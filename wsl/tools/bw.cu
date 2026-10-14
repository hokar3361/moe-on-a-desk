// H2D bandwidth for expert-sized chunks (1.87 MB) at random offsets of the model file: pageable mmap, cudaHostRegister'd mmap,
// pinned buffer (cudaMallocHost), and a pinned staging buffer filled by CPU memcpy threads
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char ** argv) {
    const char * path = argv[1];
    const size_t chunk = 1958400; // 1.87 MB
    const int n = 512;            // 1 GB per test
    int fd = open(path, O_RDONLY); size_t fsz = lseek(fd, 0, SEEK_END);
    const size_t region = 8ull << 30; // 8 GB window of the file
    uint8_t * m = (uint8_t *) mmap(nullptr, region, PROT_READ, MAP_SHARED, fd, 4ull << 30);
    std::vector<size_t> off(n); srand(1); for (auto & o : off) o = ((size_t) rand() * 4096) % (region - chunk);
    volatile uint64_t sink = 0; for (size_t i = 0; i < region; i += 4096) sink += m[i]; // fault in
    uint8_t * d; CK(cudaMalloc(&d, chunk * 64));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    auto run = [&](const char * name, auto f) { f(); CK(cudaStreamSynchronize(s)); double t = now(); f(); CK(cudaStreamSynchronize(s)); t = now() - t;
        printf("%-30s %6.1f GB/s  (%.1f us per 1.87 MB)\n", name, n * chunk / t / 1e9, t / n * 1e6); };
    run("pageable mmap", [&] { for (int i = 0; i < n; ++i) CK(cudaMemcpyAsync(d + (i % 64) * chunk, m + off[i], chunk, cudaMemcpyHostToDevice, s)); });
    uint8_t * pin; CK(cudaMallocHost(&pin, chunk * 64)); memcpy(pin, m, chunk * 64);
    run("pinned (cudaMallocHost)", [&] { for (int i = 0; i < n; ++i) CK(cudaMemcpyAsync(d + (i % 64) * chunk, pin + (i % 64) * chunk, chunk, cudaMemcpyHostToDevice, s)); });
    // staging: T threads memcpy 64 chunks into pinned memory, then one DMA (serial, no overlap)
    for (int T : {1, 2, 4, 8}) {
        double t = now();
        for (int b = 0; b < n; b += 64) {
            std::vector<std::thread> th;
            for (int k = 0; k < T; ++k) th.emplace_back([&, k] { for (int i = k; i < 64; i += T) memcpy(pin + i * chunk, m + off[b + i], chunk); });
            for (auto & x : th) x.join();
            CK(cudaMemcpyAsync(d, pin, chunk * 64, cudaMemcpyHostToDevice, s)); CK(cudaStreamSynchronize(s));
        }
        t = now() - t; printf("staging, %d memcpy threads      %6.1f GB/s (copy then DMA, no overlap)\n", T, n * chunk / t / 1e9);
    }
    double t = now(); cudaError_t e = cudaHostRegister(m, region, cudaHostRegisterReadOnly); t = now() - t;
    printf("cudaHostRegister 8 GB mmap: %s (%.2f s)\n", cudaGetErrorString(e), t);
    if (e == cudaSuccess) {
        run("registered mmap", [&] { for (int i = 0; i < n; ++i) CK(cudaMemcpyAsync(d + (i % 64) * chunk, m + off[i], chunk, cudaMemcpyHostToDevice, s)); });
        cudaHostUnregister(m);
    } else cudaGetLastError();
    // anonymous memory registered (a copy of the experts in our own pinned RAM)
    uint8_t * big; CK(cudaMallocHost(&big, 2ull << 30));
    for (size_t i = 0; i < (2ull << 30); i += chunk) memcpy(big + i, m + (i % (region - chunk)), std::min<size_t>(chunk, (2ull << 30) - i));
    run("pinned 2 GB, random chunks", [&] { for (int i = 0; i < n; ++i) CK(cudaMemcpyAsync(d + (i % 64) * chunk, big + (off[i] % ((2ull << 30) - chunk)), chunk, cudaMemcpyHostToDevice, s)); });
    printf("sink %llu file %zu\n", (unsigned long long) sink % 7, fsz);
}
