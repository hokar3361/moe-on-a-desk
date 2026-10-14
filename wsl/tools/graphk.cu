// cost of a tiny kernel inside a CUDA graph on this machine: a chain of N dependent tiny kernels (4 x 2560 floats each), captured
// once, launched many times; also a chain of N kernels that each read 1 MB (a small weight) to see the bandwidth-bound regime
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>
__global__ void tiny(float * x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = x[i] * 1.0001f + 0.5f; }
__global__ void readw(const float4 * w, float * y, int n4) {
    float acc = 0; for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) { float4 v = w[i]; acc += v.x + v.y + v.z + v.w; }
    if (acc == 12345.f) y[0] = acc;
}
int main() {
    float * x; cudaMalloc(&x, 4 * 2560 * sizeof(float));
    float4 * w; cudaMalloc(&w, 256 << 20);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int kind = 0; kind < 2; ++kind) for (int N : {100, 1000}) {
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int k = 0; k < N; ++k) {
            if (kind == 0) tiny<<<40, 256, 0, s>>>(x, 4 * 2560);
            else readw<<<128, 256, 0, s>>>(w + (size_t) (k % 256) * (1 << 16), x, 1 << 16); // 1 MB each
        }
        cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, 0);
        for (int r = 0; r < 3; ++r) cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        auto t0 = std::chrono::steady_clock::now();
        const int R = 20;
        for (int r = 0; r < R; ++r) cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / R;
        printf("%s x %4d in one graph: %8.1f us per launch, %.2f us per kernel\n", kind == 0 ? "tiny kernel " : "1 MB read   ", N, us, us / N);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }
    // tiny kernels with an external event record node before each (as GGML_CUDA_GPROF does)
    {
        const int N = 200; std::vector<cudaEvent_t> ev(N + 1); for (auto & e : ev) cudaEventCreate(&e);
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int k = 0; k < N; ++k) { cudaEventRecordWithFlags(ev[k], s, cudaEventRecordExternal); tiny<<<40, 256, 0, s>>>(x, 4 * 2560); }
        cudaEventRecordWithFlags(ev[N], s, cudaEventRecordExternal);
        cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&ge, g, 0);
        for (int r = 0; r < 3; ++r) cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 20; ++r) cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
        float ms = 0; cudaEventElapsedTime(&ms, ev[0], ev[N]);
        printf("tiny kernel + event node x %d: %.2f us per pair (wall), events say %.2f us per kernel\n", N, us / N, ms * 1000 / N);
    }
    // plain stream launches (no graph)
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < 1000; ++k) tiny<<<40, 256, 0, s>>>(x, 4 * 2560);
    cudaStreamSynchronize(s);
    printf("tiny kernel x 1000 without a graph: %.2f us per kernel\n", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 1000);
}
