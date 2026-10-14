// zero-copy fetch of MoE experts as the MOE_FETCH copy kernel does it: n experts (3 tensors each: 473600, 473600, 921600 bytes) read by
// the GPU from page-locked mapped host memory into VRAM. Varies blocks per expert, threads per block and the load width.
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

struct desc_t { const uint8_t * src[3]; uint8_t * dst[3]; int64_t size[3]; };

template <typename T>
__global__ void kcopy(const desc_t * d, const int * ex, int n) {
    const int m = blockIdx.y;
    if (m >= n) return;
    const int64_t e = ex[m];
    for (int w = 0; w < 3; ++w) {
        const T * s = (const T *) (d->src[w] + e * d->size[w]);
        T * o = (T *) (d->dst[w] + (int64_t) m * d->size[w]);
        const int64_t nn = d->size[w] / sizeof(T);
        for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < nn; i += (int64_t) gridDim.x * blockDim.x) o[i] = s[i];
    }
}

// one block per (expert, tensor) slice, all three tensors in parallel
template <typename T>
__global__ void kcopy_par(const desc_t * d, const int * ex, int n, int bpt) {
    const int m = blockIdx.y;
    if (m >= n) return;
    const int w = blockIdx.x / bpt;
    const int b = blockIdx.x % bpt;
    const int64_t e = ex[m];
    const T * s = (const T *) (d->src[w] + e * d->size[w]);
    T * o = (T *) (d->dst[w] + (int64_t) m * d->size[w]);
    const int64_t nn = d->size[w] / sizeof(T);
    for (int64_t i = (int64_t) b * blockDim.x + threadIdx.x; i < nn; i += (int64_t) bpt * blockDim.x) o[i] = s[i];
}

int main(int argc, char ** argv) {
    int fd = open(argv[1], O_RDONLY);
    const size_t region = 16ull << 30;
    uint8_t * m = (uint8_t *) mmap(nullptr, region, PROT_READ, MAP_SHARED, fd, 2ull << 30);
    volatile uint64_t sink = 0; for (size_t i = 0; i < region; i += 4096) sink += m[i];
    CK(cudaHostRegister(m, region, cudaHostRegisterPortable | cudaHostRegisterReadOnly));
    uint8_t * dm; CK(cudaHostGetDevicePointer((void **) &dm, m, 0));
    desc_t h;
    const int64_t sz[3] = { 473600, 473600, 921600 };
    uint8_t * vram; CK(cudaMalloc(&vram, 3 * 921600 * 16));
    for (int w = 0; w < 3; ++w) { h.src[w] = dm + (size_t) w * (5ull << 30); h.dst[w] = vram + (size_t) w * 921600 * 16; h.size[w] = sz[w]; }
    desc_t * dd; CK(cudaMalloc(&dd, sizeof(desc_t))); CK(cudaMemcpy(dd, &h, sizeof(h), cudaMemcpyHostToDevice));
    const int R = 400; // layers worth of fetches
    std::vector<int> exh(R * 8); srand(3); for (auto & e : exh) e = rand() % 2048;
    int * ex; CK(cudaMalloc(&ex, exh.size() * sizeof(int))); CK(cudaMemcpy(ex, exh.data(), exh.size() * sizeof(int), cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    auto run = [&](const char * name, int n, auto launch) {
        for (int r = 0; r < 20; ++r) launch(ex + r * 8, n);
        CK(cudaStreamSynchronize(s));
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < R; ++r) launch(ex + r * 8, n);
        CK(cudaStreamSynchronize(s));
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double bytes = (double) R * n * (473600 + 473600 + 921600);
        printf("%-40s n=%d: %5.1f GB/s, %6.1f us per layer\n", name, n, bytes / t / 1e9, t / R * 1e6);
    };
    for (int n : {3, 5}) {
        for (int bpe : {2, 4, 8, 16}) for (int th : {256, 1024}) {
            char nm[64]; snprintf(nm, sizeof nm, "seq uint4 bpe=%d th=%d", bpe, th);
            run(nm, n, [&](int * e, int nn) { kcopy<uint4><<<dim3(bpe, nn), th, 0, s>>>(dd, e, nn); });
        }
        for (int bpt : {2, 4, 8}) {
            char nm[64]; snprintf(nm, sizeof nm, "par uint4 blocks/tensor=%d th=256", bpt);
            run(nm, n, [&](int * e, int nn) { kcopy_par<uint4><<<dim3(3 * bpt, nn), 256, 0, s>>>(dd, e, nn, bpt); });
        }
        run("cudaMemcpyAsync x3 per expert", n, [&](int * e, int nn) {
            for (int k = 0; k < nn; ++k) for (int w = 0; w < 3; ++w)
                cudaMemcpyAsync(h.dst[w] + (size_t) k * sz[w], m + (size_t) w * (5ull << 30) + (size_t) exh[(e - ex) + k] * sz[w], sz[w], cudaMemcpyHostToDevice, s);
        });
    }
    printf("sink %llu\n", (unsigned long long) sink % 3);
}
