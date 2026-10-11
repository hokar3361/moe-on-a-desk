# Row-prune study: opt-in verify-row dump (LLAMA_ROWPRUNE_DUMP=<file>, diagnostics only; no effect when unset).
#
# For every speculative verify pass of slot 0 the server appends one record to <file>:
#   - the HC residual (tensor "l_last-<il>", [n_embd*hc] = 10240 floats, stored fp16) of every verify row at the layers
#     listed in LLAMA_ROWPRUNE_LAYERS (il values, default "15,23,31,47" = after 16 / 24 / 32 / 48 layers),
#   - the row tokens [bonus, d1..dk], the target argmax / top-1 / top-2 logit / log-sum-exp / logit of the next row's
#     token for every row, the accept count, the task id and the position of row 0,
#   - the MTP top-8 candidates (id, logit, p) of every drafted token (side channel from common/speculative.cpp).
# File format: see probe.py (read_dump). Capture uses the scheduler eval callback on the target context only: the graph
# is split at the captured nodes (slower; same kernels and arithmetic), so the dump run must not be used for timing.
# Requires -np 1. Replay batches (after a checkpoint restore) and synthetic acceptance are not recorded.
#
# ANCHORS: taken from a reconstruction of the fork = upstream llama.cpp 781dbc5 + mac125b/patches/0001..0046 (tag
# mac125b-best-2026-10-11d), all patches applied cleanly with `git am`. NOT verified against the live Mac tree, NOT
# compiled with the Metal build. Coexists with tools/depthsignal/sigpatch.py (both insert before the same depth-cap
# line and keep it). Every anchor is asserted unique; the script stops without writing anything on the first mismatch.
#
# usage: python3 dumppatch.py <llama.cpp tree>
import sys

root = sys.argv[1]
P_SRV  = root + "/tools/server/server-context.cpp"
P_SPEC = root + "/common/speculative.cpp"

srv  = open(P_SRV,  encoding="utf-8", newline="").read()
spec = open(P_SPEC, encoding="utf-8", newline="").read()

if "LLAMA_ROWPRUNE_DUMP" in srv or "LLAMA_ROWPRUNE_DUMP" in spec:
    print("dumppatch: already applied (or partially: check both files)"); sys.exit(0)

def rep(s, old, new, what):
    n = s.count(old)
    if n != 1:
        print("dumppatch: anchor '%s' found %d times, expected 1; nothing written" % (what, n)); sys.exit(1)
    return s.replace(old, new)

# ---------------------------------------------------------------------------------------------------------------
# common/speculative.cpp
# ---------------------------------------------------------------------------------------------------------------

# S1. side-channel storage (file scope, before the MTP implementation)
spec = rep(spec, """struct common_speculative_impl_draft_mtp : public common_speculative_impl {
""", """// LLAMA_ROWPRUNE_DUMP side channel (mac125b/tools/rowprune): the top-8 MTP candidates of every drafted token of
// seq 0, in draft order. The server clears them before each draft() call and reads them when it records a verify pass.
std::vector<int32_t> g_rowprune_cand_id;
std::vector<float>   g_rowprune_cand_logit;
std::vector<float>   g_rowprune_cand_p;
static const bool    g_rowprune_cand_on = getenv("LLAMA_ROWPRUNE_DUMP") != nullptr;

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
""", "MTP impl struct")

# S2. draft(): record the candidates of every drafted token that enters the verify batch (just before the depth cap)
spec = rep(spec, """                if ((mtp_depth > 0 ? std::min(params.n_max, mtp_depth) : params.n_max) <= (int) result.size()) {
""", """                if (g_rowprune_cand_on && seq_id == 0) {
                    for (int c = 0; c < 8; ++c) {
                        const bool ok = c < (int) cur_p->size;
                        g_rowprune_cand_id.push_back(ok ? cur_p->data[c].id : -1);
                        g_rowprune_cand_logit.push_back(ok ? cur_p->data[c].logit : -INFINITY);
                        g_rowprune_cand_p.push_back(ok ? cur_p->data[c].p : 0.0f);
                    }
                }

                if ((mtp_depth > 0 ? std::min(params.n_max, mtp_depth) : params.n_max) <= (int) result.size()) {
""", "MTP draft depth-cap check")

# ---------------------------------------------------------------------------------------------------------------
# tools/server/server-context.cpp
# ---------------------------------------------------------------------------------------------------------------

# V1. capture state, eval callback, record writer (file scope, after the includes)
srv = rep(srv, """#include <fstream>
""", """#include <fstream>
#include <cstring>
#include <cmath>

// ---- LLAMA_ROWPRUNE_DUMP (opt-in diagnostics, mac125b/tools/rowprune; no effect when unset) ----
extern std::vector<int32_t> g_rowprune_cand_id;     // common/speculative.cpp: 8 per drafted token (seq 0)
extern std::vector<float>   g_rowprune_cand_logit;
extern std::vector<float>   g_rowprune_cand_p;

struct rowprune_state {
    FILE *   f           = nullptr;
    bool     header_done = false;
    std::vector<int32_t> layers;                // il values of the captured "l_last-<il>" tensors, ascending
    int32_t  ne_row = 0;                        // n_embd*hc
    int32_t  n_tok  = -1;                       // rows of the last captured pass
    uint64_t gen    = 0;                        // pass counter (bumped at the first captured layer)
    std::vector<std::vector<float>> h;          // per layer: n_tok*ne_row
    std::vector<uint64_t>           got;        // per layer: gen of the last capture
    std::vector<ggml_fp16_t>        h16;
    uint64_t n_rec = 0;
};
static rowprune_state g_rowprune;
static const int32_t  ROWPRUNE_K = 8;

// scheduler eval callback (target context only): ask -> split the graph after the wanted nodes; data -> copy them out.
// Must return true on the data call (false aborts the rest of the graph).
static bool rowprune_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (rowprune_state *) user_data;
    if (strncmp(t->name, "l_last-", 7) != 0) {
        return false;
    }
    const int32_t il = atoi(t->name + 7);
    int k = -1;
    for (size_t i = 0; i < st->layers.size(); ++i) {
        if (st->layers[i] == il) {
            k = (int) i;
            break;
        }
    }
    if (k < 0) {
        return false;
    }
    if (ask) {
        // verify-sized passes only (prompt batches are not split)
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[2] >= 1 && t->ne[2] <= 16 && t->ne[3] == 1;
    }
    const int32_t nt     = (int32_t) t->ne[2];
    const int32_t ne_row = (int32_t) (t->ne[0]*t->ne[1]);
    if (k == 0) {
        st->gen++;
        st->n_tok = nt;
    }
    if (nt != st->n_tok || (st->ne_row != 0 && ne_row != st->ne_row)) {
        return true;
    }
    st->ne_row = ne_row;
    st->h[k].resize((size_t) nt*ne_row);
    ggml_backend_tensor_get(t, st->h[k].data(), 0, ggml_nbytes(t));
    st->got[k] = st->gen;
    return true;
}

static bool rowprune_dump_init(int n_parallel) {
    auto & st = g_rowprune;
    if (st.f) {
        return true; // resume from sleep: keep the open file
    }
    const char * path = getenv("LLAMA_ROWPRUNE_DUMP");
    if (!path || !*path) {
        return false;
    }
    if (n_parallel != 1) {
        LOG_WRN("rowprune: LLAMA_ROWPRUNE_DUMP needs -np 1 (got %d), dump disabled\\n", n_parallel);
        return false;
    }
    const char * ls = getenv("LLAMA_ROWPRUNE_LAYERS");
    std::string spec_layers = ls ? ls : "15,23,31,47";
    st.layers.clear();
    for (size_t p = 0; p < spec_layers.size(); ) {
        size_t q = spec_layers.find(',', p);
        if (q == std::string::npos) {
            q = spec_layers.size();
        }
        if (q > p) {
            st.layers.push_back(atoi(spec_layers.substr(p, q - p).c_str()));
        }
        p = q + 1;
    }
    std::sort(st.layers.begin(), st.layers.end());
    st.layers.erase(std::unique(st.layers.begin(), st.layers.end()), st.layers.end());
    if (st.layers.empty()) {
        return false;
    }
    // appending to an existing dump is allowed only with the same layer list
    if (FILE * fr = fopen(path, "rb")) {
        char    magic[4] = {0};
        int32_t ver = 0, nl = 0;
        if (fread(magic, 1, 4, fr) == 4 && fread(&ver, 4, 1, fr) == 1 && fread(&nl, 4, 1, fr) == 1) {
            std::vector<int32_t> fl(std::max(nl, 0));
            const bool ok = memcmp(magic, "RPD1", 4) == 0 && ver == 1 && nl > 0 &&
                            fread(fl.data(), 4, nl, fr) == (size_t) nl && fl == st.layers;
            if (!ok) {
                LOG_ERR("rowprune: %s exists with another format or layer list, dump disabled\\n", path);
                fclose(fr);
                return false;
            }
            st.header_done = true;
        }
        fclose(fr);
    }
    st.f = fopen(path, "ab");
    if (!st.f) {
        LOG_ERR("rowprune: cannot open %s, dump disabled\\n", path);
        return false;
    }
    st.h.assign(st.layers.size(), {});
    st.got.assign(st.layers.size(), 0);
    LOG_INF("rowprune: dumping verify rows to %s (layers %s)\\n", path, spec_layers.c_str());
    return true;
}

// one verify pass of slot 0: rows [tok0, draft...] at batch indices i_batch, n_acc drafts accepted
static void rowprune_dump_round(llama_context * ctx, const llama_vocab * vocab, int32_t req_id, int32_t pos0,
        llama_token tok0, const llama_tokens & draft, const std::vector<int32_t> & i_batch, int32_t n_acc) {
    auto & st = g_rowprune;
    const int32_t B = (int32_t) i_batch.size();
    if (!st.f || B < 2 || B > 16 || st.n_tok != B || (int32_t) draft.size() != B - 1 || st.ne_row <= 0) {
        return;
    }
    for (size_t k = 0; k < st.layers.size(); ++k) {
        if (st.got[k] != st.gen || st.h[k].size() != (size_t) B*st.ne_row) {
            return;
        }
    }
    for (int32_t j = 0; j < B; ++j) {
        if (i_batch[j] < 0 || i_batch[j] >= B) {
            return;
        }
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int32_t> tok(B), am(B);
    std::vector<float>   lse(B), l1(B), l2(B), lnext(B);
    tok[0] = tok0;
    for (int32_t j = 1; j < B; ++j) {
        tok[j] = draft[j - 1];
    }
    for (int32_t j = 0; j < B; ++j) {
        const float * l = llama_get_logits_ith(ctx, i_batch[j]);
        if (!l) {
            return;
        }
        float   m1 = -INFINITY, m2 = -INFINITY;
        int32_t a  = 0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            const float x = l[v];
            if (x > m2) {
                if (x > m1) { m2 = m1; m1 = x; a = v; } else { m2 = x; }
            }
        }
        double z = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            z += std::exp((double) (l[v] - m1));
        }
        am[j]    = a;
        l1[j]    = m1;
        l2[j]    = m2;
        lse[j]   = (float) (m1 + std::log(z));
        lnext[j] = (j + 1 < B && tok[j + 1] >= 0 && tok[j + 1] < n_vocab) ? l[tok[j + 1]] : NAN;
    }
    const size_t nc = (size_t) (B - 1)*ROWPRUNE_K;
    const bool cand_ok = g_rowprune_cand_id.size() >= nc && g_rowprune_cand_logit.size() >= nc && g_rowprune_cand_p.size() >= nc;
    std::vector<int32_t> cid(nc, -1);
    std::vector<float>   clg(nc, -INFINITY), cpp(nc, 0.0f);
    if (cand_ok) {
        std::copy(g_rowprune_cand_id.begin(),    g_rowprune_cand_id.begin()    + nc, cid.begin());
        std::copy(g_rowprune_cand_logit.begin(), g_rowprune_cand_logit.begin() + nc, clg.begin());
        std::copy(g_rowprune_cand_p.begin(),     g_rowprune_cand_p.begin()     + nc, cpp.begin());
    }
    FILE * f = st.f;
    if (!st.header_done) {
        // file header: "RPD1", version, n_layers, layers[n_layers], ne_row, K
        const int32_t ver = 1, nl = (int32_t) st.layers.size();
        fwrite("RPD1", 1, 4, f);
        fwrite(&ver, 4, 1, f);
        fwrite(&nl, 4, 1, f);
        fwrite(st.layers.data(), 4, nl, f);
        fwrite(&st.ne_row, 4, 1, f);
        fwrite(&ROWPRUNE_K, 4, 1, f);
        st.header_done = true;
    }
    // record: "RPR1", B, n_acc, req_id, pos0, flags, tok[B], am[B], lse[B], l1[B], l2[B], lnext[B],
    //         cand_id[B-1][K], cand_logit[B-1][K], cand_p[B-1][K], h[n_layers][B][ne_row] (fp16)
    const int32_t flags = cand_ok ? 1 : 0;
    const int32_t hdr[5] = { B, n_acc, req_id, pos0, flags };
    fwrite("RPR1", 1, 4, f);
    fwrite(hdr, 4, 5, f);
    fwrite(tok.data(), 4, B, f);
    fwrite(am.data(),  4, B, f);
    fwrite(lse.data(), 4, B, f);
    fwrite(l1.data(),  4, B, f);
    fwrite(l2.data(),  4, B, f);
    fwrite(lnext.data(), 4, B, f);
    fwrite(cid.data(), 4, nc, f);
    fwrite(clg.data(), 4, nc, f);
    fwrite(cpp.data(), 4, nc, f);
    st.h16.resize(st.ne_row);
    for (size_t k = 0; k < st.layers.size(); ++k) {
        for (int32_t j = 0; j < B; ++j) {
            ggml_fp32_to_fp16_row(st.h[k].data() + (size_t) i_batch[j]*st.ne_row, st.h16.data(), st.ne_row);
            fwrite(st.h16.data(), sizeof(ggml_fp16_t), st.ne_row, f);
        }
        st.got[k] = 0;
    }
    if (++st.n_rec % 32 == 0) {
        fflush(f);
    }
}
""", "includes")

# V2. load_model(): install the callback on the target context params
srv = rep(srv, """        llama_init = common_init_from_params(params_base);
""", """        if (rowprune_dump_init(params_base.n_parallel)) {
            params_base.cb_eval           = rowprune_cb_eval;
            params_base.cb_eval_user_data = &g_rowprune;
        }

        llama_init = common_init_from_params(params_base);
""", "target init")

# V3. the draft / MTP context must not inherit the callback (its result_norm etc. would be split and synchronized)
srv = rep(srv, """                common_params params_dft = common_base_params_to_speculative(params_base);
""", """                common_params params_dft = common_base_params_to_speculative(params_base);
                params_dft.cb_eval           = nullptr;
                params_dft.cb_eval_user_data = nullptr;
""", "draft params")

# V4. clear the candidate side channel before each draft() call
srv = rep(srv, """                common_speculative_draft(spec.get());
""", """                g_rowprune_cand_id.clear();
                g_rowprune_cand_logit.clear();
                g_rowprune_cand_p.clear();
                common_speculative_draft(spec.get());
""", "draft call")

# V5. after the acceptance of a verify pass (spec_i_batch / spec_draft still describe the verified rows)
srv = rep(srv, """                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);
""", """                if (g_rowprune.f && synth_probs.empty() && !slot.spec_is_replay) {
                    rowprune_dump_round(slot.ctx_tgt, vocab, (int32_t) slot.task->id,
                            (int32_t) slot.prompt.n_tokens() - (int32_t) n_draft - 1,
                            slot.sampled, slot.spec_draft, slot.spec_i_batch, (int32_t) accepted.size() - 1);
                }

                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);
""", "accept")

open(P_SPEC, "w", encoding="utf-8", newline="").write(spec)
open(P_SRV,  "w", encoding="utf-8", newline="").write(srv)
print("dumppatch ok")
