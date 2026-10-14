// zero-copy: a kernel reads expert-sized chunks (1.87 MB) at random offsets of the page-locked, mapped model file directly over PCIe.
// Compares: kernel reading host-mapped memory, cudaMemcpyAsync of the same chunks, and a kernel reading VRAM.
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// each block sums a slice of one chunk with 16-byte loads; grid = chunks x blocks_per_chunk
__global__ void readchunks(const uint8_t * base, const size_t * off, size_t chunk, float * out) {
    const int c = blockIdx.y;
    const uint4 * p = (const uint4 *) (base + off[c]);
    const size_t n16 = chunk / 16;
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = p[i]; acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = 1.0f;
}

int main(int argc, char ** argv) {
    const size_t chunk = 1958400;
    const int n = 256; // chunks per test (~500 MB)
    int fd = open(argv[1], O_RDONLY);
    const size_t region = 8ull << 30;
    uint8_t * m = (uint8_t *) mmap(nullptr, region, PROT_READ, MAP_SHARED, fd, 4ull << 30);
    volatile uint64_t sink = 0; for (size_t i = 0; i < region; i += 4096) sink += m[i];
    CK(cudaHostRegister(m, region, cudaHostRegisterMapped | cudaHostRegisterReadOnly));
    uint8_t * dm; CK(cudaHostGetDevicePointer((void **) &dm, m, 0));
    std::vector<size_t> off(n); srand(2); for (auto & o : off) o = (((size_t) rand() * 4096) % (region - chunk)) & ~(size_t) 255;
    size_t * doff; CK(cudaMalloc(&doff, n * sizeof(size_t))); CK(cudaMemcpy(doff, off.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
    float * out; CK(cudaMalloc(&out, 4));
    uint8_t * vram; CK(cudaMalloc(&vram, region / 4));
    std::vector<size_t> offv(n); for (int i = 0; i < n; ++i) offv[i] = off[i] % (region / 4 - chunk) & ~(size_t) 255;
    size_t * doffv; CK(cudaMalloc(&doffv, n * sizeof(size_t))); CK(cudaMemcpy(doffv, offv.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    for (int bpc : {8, 32, 128}) for (int per : {1, 4, 16}) {
        // per = chunks read by one kernel launch (like the misses of one layer)
        double best = 1e9;
        for (int r = 0; r < 3; ++r) {
            CK(cudaStreamSynchronize(s)); double t = now();
            for (int i = 0; i < n; i += per) readchunks<<<dim3(bpc, per), 256, 0, s>>>(dm, doff + i, chunk, out);
            CK(cudaStreamSynchronize(s)); t = now() - t; best = std::min(best, t);
        }
        printf("zero-copy kernel, %3d blocks/chunk, %2d chunks/launch: %5.1f GB/s (%.1f us per chunk)\n", bpc, per, n * chunk / best / 1e9, best / n * 1e6);
    }
    {
        double t = now();
        for (int i = 0; i < n; ++i) CK(cudaMemcpyAsync(vram + offv[i], m + off[i], chunk, cudaMemcpyHostToDevice, s));
        CK(cudaStreamSynchronize(s)); t = now() - t;
        printf("cudaMemcpyAsync of the same chunks: %5.1f GB/s\n", n * chunk / t / 1e9);
    }
    {
        double t = now();
        for (int i = 0; i < n; i += 4) readchunks<<<dim3(32, 4), 256, 0, s>>>(vram, doffv + i, chunk, out);
        CK(cudaStreamSynchronize(s)); t = now() - t;
        printf("VRAM kernel (reference): %6.1f GB/s\n", n * chunk / t / 1e9);
    }
    printf("sink %llu\n", (unsigned long long) sink % 3);
}
