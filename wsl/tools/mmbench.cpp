// time of ggml MUL_MAT on the CUDA backend for the small bf16 shapes of Qwen3.8-Flash-Next (hyper-connections, gates, router),
// as a chain of 200 independent matmuls in one graph (CUDA graphs on), per op
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cstdio>
#include <vector>

static double bench(ggml_backend_t be, ggml_type t, int64_t k, int64_t n, int64_t b, int reps) {
    ggml_init_params ip = { ggml_tensor_overhead() * (reps * 3 + 16) + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, t, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, b);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (int r = 0; r < reps; ++r) {
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_build_forward_expand(gf, y);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_clear(buf, 0);
    for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(be, gf);
    const int iters = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, gf);
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters / reps;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return us;
}

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    struct S { const char * name; int64_t k, n; } shapes[] = {
        { "shared_expert_gate 2560x1", 2560, 1 },
        { "hc 10240x320", 10240, 320 },
        { "hc_inject 10240x4", 10240, 4 },
        { "hc_gate 320x10240", 320, 10240 },
        { "2560x48", 2560, 48 },
        { "router 2560x512", 2560, 512 },
        { "ple_key 2560x10240", 2560, 10240 },
    };
    for (auto & s : shapes) {
        for (int b : {1, 4, 8}) {
            printf("%-28s B%d  bf16 %7.2f us   f16 %7.2f us   f32 %7.2f us\n", s.name, b,
                   bench(be, GGML_TYPE_BF16, s.k, s.n, b, 200), bench(be, GGML_TYPE_F16, s.k, s.n, b, 200), bench(be, GGML_TYPE_F32, s.k, s.n, b, 200));
        }
    }
    ggml_backend_free(be);
}
