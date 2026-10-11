// draftd: remote draft server for cross-machine speculative decoding (hypothesis A, DESIGN.md §10).
// Runs the drafter (a same-model low-bit copy) on the PC GPU; the Mac verifies.
//
// Protocol (TCP, little-endian int32/float):
//   request : int32 n_new, int32 k, n_new x int32 token
//             n_new >= 0: tokens committed since the previous request are appended
//             n_new <  0: new request, the |n_new| tokens are the whole committed history (prompt)
//   response: int32 n, n x (int32 token, float p_top1)
// The committed history lives in sequence 0; drafting copies it to sequence 1 (state + KV), drafts greedily and drops
// sequence 1, so the committed state never needs a rollback (the model is hybrid: recurrent state cannot be truncated).
//
// usage: draftd <model.gguf> [port=8095] [n_ctx=16384]
// env:   DRAFTD_CPU_MOE=N   experts of the last N layers stay in host RAM (the drafter may not fit 24 GB)
#include "llama.h"
#include "ggml-backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_all(int fd, void * buf, size_t n) {
    char * p = (char *) buf;
    while (n > 0) {
        const ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += r; n -= (size_t) r;
    }
    return true;
}

static bool write_all(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n > 0) {
        const ssize_t r = send(fd, p, n, 0);
        if (r <= 0) return false;
        p += r; n -= (size_t) r;
    }
    return true;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct drafter {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    int n_vocab = 0;
    int n_past  = 0;              // committed tokens in sequence 0
    std::vector<float> last;      // logits after the last committed token
    llama_batch batch;

    void init(const char * path, int n_ctx) {
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        static std::vector<llama_model_tensor_buft_override> ov;
        static std::vector<std::string> pats;
        if (const char * e = getenv("DRAFTD_CPU_MOE")) {
            const int n = atoi(e);
            for (int il = 48 - n; il < 48; ++il) {
                pats.push_back("blk\\." + std::to_string(il) + "\\.ffn_(gate|up|down)_exps");
            }
            for (auto & p : pats) ov.push_back({ p.c_str(), ggml_backend_cpu_buffer_type() });
            ov.push_back({ nullptr, nullptr });
            mp.tensor_buft_overrides = ov.data();
        }
        model = llama_model_load_from_file(path, mp);
        if (!model) { fprintf(stderr, "load failed\n"); exit(1); }
        auto cp = llama_context_default_params();
        cp.n_ctx      = n_ctx;
        cp.n_batch    = 512;
        cp.n_ubatch   = 512;
        cp.n_seq_max  = 2;
        cp.kv_unified = true;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "context failed\n"); exit(1); }
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        batch = llama_batch_init(512, 0, 2);
    }

    void reset() {
        llama_memory_clear(llama_get_memory(ctx), true);
        n_past = 0;
        last.clear();
    }

    // append committed tokens to sequence 0; keep the logits of the last one
    bool commit(const std::vector<int32_t> & toks) {
        for (size_t i0 = 0; i0 < toks.size(); i0 += 512) {
            const size_t n = std::min<size_t>(512, toks.size() - i0);
            batch.n_tokens = 0;
            for (size_t i = 0; i < n; ++i) {
                const int j = batch.n_tokens++;
                batch.token[j] = toks[i0 + i];
                batch.pos[j] = n_past + (int) i;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = (i0 + i + 1 == toks.size());
            }
            if (llama_decode(ctx, batch) != 0) return false;
            n_past += (int) n;
        }
        if (!toks.empty()) {
            const float * l = llama_get_logits_ith(ctx, -1);
            last.assign(l, l + n_vocab);
        }
        return true;
    }

    static void top1(const float * l, int n, int32_t & id, float & p) {
        int b = 0; float m = l[0];
        for (int i = 1; i < n; ++i) if (l[i] > m) { m = l[i]; b = i; }
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += std::exp((double) (l[i] - m));
        id = b; p = (float) (1.0/s);
    }

    // k greedy drafts after the committed history (sequence 1 is a throw-away copy)
    void draft(int k, std::vector<int32_t> & out_t, std::vector<float> & out_p) {
        out_t.clear(); out_p.clear();
        if (last.empty() || k <= 0) return;
        int32_t id; float p;
        top1(last.data(), n_vocab, id, p);
        out_t.push_back(id); out_p.push_back(p);
        if (k == 1) return;
        auto * mem = llama_get_memory(ctx);
        llama_memory_seq_rm(mem, 1, -1, -1);
        llama_memory_seq_cp(mem, 0, 1, -1, -1);
        for (int i = 1; i < k; ++i) {
            batch.n_tokens = 1;
            batch.token[0] = id;
            batch.pos[0] = n_past + i - 1;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 1;
            batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) break;
            top1(llama_get_logits_ith(ctx, -1), n_vocab, id, p);
            out_t.push_back(id); out_p.push_back(p);
        }
        llama_memory_seq_rm(mem, 1, -1, -1);
    }
};

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [port] [n_ctx]\n", argv[0]); return 1; }
    const int port  = argc > 2 ? atoi(argv[2]) : 8095;
    const int n_ctx = argc > 3 ? atoi(argv[3]) : 16384;
    drafter d;
    d.init(argv[1], n_ctx);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET; addr.sin_port = htons(port); addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) != 0 || listen(srv, 1) != 0) { perror("bind/listen"); return 1; }
    fprintf(stderr, "draftd: listening on %d\n", port);

    for (;;) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        fprintf(stderr, "draftd: client connected\n");
        std::vector<int32_t> toks, out_t;
        std::vector<float> out_p;
        double t_commit = 0, t_draft = 0; long n_req = 0, n_drafted = 0;
        for (;;) {
            int32_t hdr[2];
            if (!read_all(fd, hdr, sizeof(hdr))) break;
            const int32_t n_new = hdr[0], k = hdr[1];
            toks.resize(std::abs(n_new));
            if (!toks.empty() && !read_all(fd, toks.data(), toks.size()*sizeof(int32_t))) break;
            const double t0 = now_ms();
            if (n_new < 0) d.reset();
            if (!d.commit(toks)) { fprintf(stderr, "draftd: decode failed\n"); break; }
            const double t1 = now_ms();
            d.draft(k, out_t, out_p);
            const double t2 = now_ms();
            t_commit += t1 - t0; t_draft += t2 - t1; n_req++; n_drafted += (long) out_t.size();
            const int32_t n = (int32_t) out_t.size();
            std::vector<char> buf(4 + n*8);
            memcpy(buf.data(), &n, 4);
            for (int i = 0; i < n; ++i) { memcpy(buf.data() + 4 + 8*i, &out_t[i], 4); memcpy(buf.data() + 8 + 8*i, &out_p[i], 4); }
            if (!write_all(fd, buf.data(), buf.size())) break;
            if (n_req % 100 == 0) {
                fprintf(stderr, "draftd: %ld requests, commit %.2f ms, draft %.2f ms (%.2f ms/token)\n", n_req, t_commit/n_req, t_draft/n_req,
                        n_drafted ? t_draft/n_drafted : 0.0);
            }
        }
        close(fd);
        fprintf(stderr, "draftd: client gone\n");
    }
}
