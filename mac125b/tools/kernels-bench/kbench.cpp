// kbench: time one weight op at real shapes on Metal, and check it against the CPU backend.
// usage: kbench MODE TYPE K N NEXP NUSED B L ITERS
//   MODE gu   : MUL_MAT_ID, src1 [K,1,B] broadcast to the experts (gate/up)
//   MODE dn   : MUL_MAT_ID, src1 [K,NUSED,B] (down)
//   MODE dense: MUL_MAT [K,N] x [K,B]
//   time  : L layers with random-byte weights (each op its own stage), prints us/op
//   check : 1 layer, weights quantized from random floats, prints max rel err vs CPU
// Env GGML_METAL_CONCURRENCY_DISABLE=1 is set so every op is its own stage (as in the real graph).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

static std::vector<double> g_gpu;
static void log_cb(ggml_log_level, const char * text, void *) {
    const char * p = strstr(text, "gpu_sum=");
    if (p) g_gpu.push_back(atof(p + 8)*1000.0);   // ms -> us
}

static ggml_type type_from(const char * s) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n && strcmp(n, s) == 0) return (ggml_type) i;
    }
    fprintf(stderr, "bad type %s\n", s); exit(1);
}

struct net {
    ggml_context * ctx = nullptr;
    std::vector<ggml_tensor *> w, x, ids, y;
    ggml_cgraph * gf = nullptr;
};

static net build(const std::string & mode, ggml_type t, int K, int N, int NE, int NU, int B, int L) {
    net n;
    ggml_init_params ip = { ggml_tensor_overhead()*(8*L + 16) + ggml_graph_overhead_custom(8*L + 16, false), nullptr, true };
    n.ctx = ggml_init(ip);
    n.gf = ggml_new_graph_custom(n.ctx, 8*L + 16, false);
    for (int l = 0; l < L; ++l) {
        ggml_tensor * w, * x, * y, * id = nullptr;
        if (mode == "bat") {
            // the same work as MUL_MAT_ID at B=1 without ids: NU separate matrices, one vector each
            w = ggml_new_tensor_3d(n.ctx, t, K, N, NU);
            x = ggml_new_tensor_3d(n.ctx, GGML_TYPE_F32, K, 1, NU);
            y = ggml_mul_mat(n.ctx, w, x);
        } else if (mode == "dense") {
            w = ggml_new_tensor_2d(n.ctx, t, K, N);
            x = ggml_new_tensor_2d(n.ctx, GGML_TYPE_F32, K, B);
            y = ggml_mul_mat(n.ctx, w, x);
        } else {
            w  = ggml_new_tensor_3d(n.ctx, t, K, N, NE);
            x  = ggml_new_tensor_3d(n.ctx, GGML_TYPE_F32, K, mode == "gu" ? 1 : NU, B);
            id = ggml_new_tensor_2d(n.ctx, GGML_TYPE_I32, NU, B);
            y  = ggml_mul_mat_id(n.ctx, w, x, id);
        }
        n.w.push_back(w); n.x.push_back(x); n.ids.push_back(id); n.y.push_back(y);
        ggml_build_forward_expand(n.gf, y);
    }
    return n;
}

static void fill(net & n, const std::string & mode, ggml_type t, int K, int N, int NE, int NU, int B, bool quantize, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t l = 0; l < n.w.size(); ++l) {
        ggml_tensor * w = n.w[l];
        const size_t nb = ggml_nbytes(w);
        if (quantize) {
            const int64_t nrows = ggml_nrows(w);
            std::vector<float> f((size_t) K*nrows);
            for (auto & v : f) v = nd(rng)*0.02f;
            std::vector<uint8_t> q(nb);
            std::vector<float> imat(K, 1.0f);
            ggml_quantize_init(t);
            ggml_quantize_chunk(t, f.data(), q.data(), 0, nrows, K, imat.data());
            ggml_backend_tensor_set(w, q.data(), 0, nb);
        } else {
            // random bytes, in chunks
            const size_t CH = 64u << 20;
            std::vector<uint8_t> buf(std::min(nb, CH));
            for (size_t o = 0; o < nb; o += CH) {
                const size_t m = std::min(CH, nb - o);
                uint32_t * p = (uint32_t *) buf.data();
                for (size_t i = 0; i < m/4; ++i) p[i] = rng();
                ggml_backend_tensor_set(w, buf.data(), o, m);
            }
        }
        std::vector<float> xs(ggml_nelements(n.x[l]));
        for (auto & v : xs) v = nd(rng);
        ggml_backend_tensor_set(n.x[l], xs.data(), 0, ggml_nbytes(n.x[l]));
        if (n.ids[l]) {
            std::vector<int32_t> id((size_t) NU*B);
            for (int b = 0; b < B; ++b) {
                std::vector<int32_t> perm(NE);
                for (int e = 0; e < NE; ++e) perm[e] = e;
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int u = 0; u < NU; ++u) id[(size_t) b*NU + u] = perm[u];
            }
            // realistic overlap: the real router shares ~30% of the experts between the 4 tokens
            for (int b = 1; b < B; ++b) for (int u = 0; u < NU*3/10; ++u) id[(size_t) b*NU + u] = id[u + (size_t) (b - 1)*0];
            ggml_backend_tensor_set(n.ids[l], id.data(), 0, ggml_nbytes(n.ids[l]));
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 11) { fprintf(stderr, "usage: kbench time|check MODE TYPE K N NEXP NUSED B L ITERS\n"); return 1; }
    const std::string what = argv[1], mode = argv[2];
    const ggml_type t = type_from(argv[3]);
    const int K = atoi(argv[4]), N = atoi(argv[5]), NE = atoi(argv[6]), NU = atoi(argv[7]), B = atoi(argv[8]);
    int L = atoi(argv[9]); const int IT = atoi(argv[10]);
    setenv("GGML_METAL_CONCURRENCY_DISABLE", "1", 0);
    setenv("GGML_METAL_TIMING", "1", 0);
    ggml_log_set(log_cb, nullptr);
    const bool on_cpu = what == "timecpu";
    ggml_backend_t mtl = on_cpu ? ggml_backend_cpu_init() : ggml_backend_metal_init();
    if (on_cpu) {
        ggml_backend_cpu_set_n_threads(mtl, getenv("KB_THREADS") ? atoi(getenv("KB_THREADS")) : 12);
    }
    if (what == "check") L = 1;
    net n = build(mode, t, K, N, NE, NU, B, L);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(n.ctx, mtl);
    fill(n, mode, t, K, N, NE, NU, B, what == "check", 1234);
    if (what == "time" || what == "timecpu") {
        for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(mtl, n.gf);
        std::vector<double> ts;
        for (int i = 0; i < IT; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ggml_backend_graph_compute(mtl, n.gf);
            auto t1 = std::chrono::high_resolution_clock::now();
            ts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        const double med = ts[ts.size()/2];
        // bytes actually needed: distinct experts per layer
        double bytes = 0;
        if (mode == "dense" || mode == "bat") bytes = (double) ggml_nbytes(n.w[0]);
        else {
            std::vector<int32_t> id((size_t) NU*B);
            ggml_backend_tensor_get(n.ids[0], id.data(), 0, id.size()*4);
            std::sort(id.begin(), id.end());
            const size_t nd = std::unique(id.begin(), id.end()) - id.begin();
            bytes = (double) ggml_nbytes(n.w[0])/NE*nd;
        }
        printf("RESULT %s %s %d %d %d %d %d %.0f %.2f\n", mode.c_str(), argv[3], K, N, B, L, IT, bytes, med);
        fflush(stdout);
    } else {
        ggml_backend_graph_compute(mtl, n.gf);
        std::vector<float> ym(ggml_nelements(n.y[0]));
        ggml_backend_tensor_get(n.y[0], ym.data(), 0, ggml_nbytes(n.y[0]));
        // CPU reference on the same data
        ggml_backend_t cpu = ggml_backend_cpu_init();
        net c = build(mode, t, K, N, NE, NU, B, 1);
        ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(c.ctx, cpu);
        std::vector<uint8_t> tmp(ggml_nbytes(n.w[0]));
        ggml_backend_tensor_get(n.w[0], tmp.data(), 0, tmp.size()); ggml_backend_tensor_set(c.w[0], tmp.data(), 0, tmp.size());
        tmp.resize(ggml_nbytes(n.x[0])); ggml_backend_tensor_get(n.x[0], tmp.data(), 0, tmp.size()); ggml_backend_tensor_set(c.x[0], tmp.data(), 0, tmp.size());
        if (n.ids[0]) { tmp.resize(ggml_nbytes(n.ids[0])); ggml_backend_tensor_get(n.ids[0], tmp.data(), 0, tmp.size()); ggml_backend_tensor_set(c.ids[0], tmp.data(), 0, tmp.size()); }
        ggml_backend_graph_compute(cpu, c.gf);
        std::vector<float> yc(ym.size());
        ggml_backend_tensor_get(c.y[0], yc.data(), 0, yc.size()*4);
        double num = 0, den = 0, mx = 0;
        for (size_t i = 0; i < ym.size(); ++i) { const double d = ym[i] - yc[i]; num += d*d; den += (double) yc[i]*yc[i]; mx = std::max(mx, std::fabs(d)); }
        printf("check %s %s B=%d: rel rms err %.2e, max abs %.3e (n=%zu)\n", mode.c_str(), argv[3], B, std::sqrt(num/std::max(den, 1e-30)), mx, ym.size());
        ggml_backend_buffer_free(cbuf); ggml_backend_free(cpu);
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(mtl);
    return 0;
}
