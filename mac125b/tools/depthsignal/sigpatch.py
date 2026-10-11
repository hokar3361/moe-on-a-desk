# per-round speculative signal log (LLAMA_SPEC_SIGNAL_LOG=1|2, opt-in, diagnostics only; no effect when unset)
#
# Writes one text line per event to LLAMA_SPEC_SIGNAL_FILE (default /tmp/spec-signal.log):
#   B seq=S req=Q n_prompt=N                                  begin() of a request
#   R seq=S r=R pos0=P depth=D t_us=T                         a draft round starts (D = depth cap chosen for it)
#   D seq=S r=R k=K tok=X p=.. m=.. lm=..                     drafted token K (1-based = verify row K):
#                                                             p  = MTP top-1 prob (softmax over the sampler's top-10),
#                                                             m  = p(top1) - p(top2), lm = logit(top1) - logit(top2)
#   V seq=S r=R row=I tok=X am=Y lm=.. p1=.. H=..             verify row I (row 0 = id_last, row K = draft K):
#                                                             tok = token fed at that row, am = target argmax,
#                                                             lm = target top1-top2 logit margin,
#                                                             p1/H = target top-1 prob / entropy (level 2 only, else -1;
#                                                             truncated softmax over logits > max-20, error <= n_vocab*e^-20)
#   A seq=S r=R nacc=N dt_us=T dctl=D                         accept(): N drafts accepted, T = draft start -> accept
#
# Level 1 costs one max/second-max pass over the vocabulary per verify row (~0.1 ms/row on the host);
# level 2 adds a thresholded exp pass. Timing runs must not use the log; collect data in separate runs.
#
# ANCHORS: verified against a reconstruction of common/speculative.cpp = upstream 781dbc5 + the fork patches
# 0023..0046 that touch that file (all apply cleanly). NOT verified against the live Mac tree (~/llama.cpp), which may
# carry later edits (tested: also applies after tools/pending/dctl_bayes.py). NOT compiled here.
# Every anchor is asserted unique; the script stops without writing on the first mismatch.
import sys

p = sys.argv[1] + "/common/speculative.cpp"
s = open(p, encoding="utf-8", newline="").read()
if "LLAMA_SPEC_SIGNAL_LOG" in s:
    print("sigpatch: already applied"); sys.exit(0)

def rep(old, new, what):
    global s
    n = s.count(old)
    if n != 1:
        print("sigpatch: anchor '%s' found %d times, expected 1; nothing written" % (what, n)); sys.exit(1)
    s = s.replace(old, new)

# 1. members + helpers (after the EV planner per-seq state, which sits just before the constructor)
rep("""    std::vector<float> ev_P, ev_E;                // running chain acceptance and expected accepted, per seq
""", """    std::vector<float> ev_P, ev_E;                // running chain acceptance and expected accepted, per seq

    // ---- per-round signal log (LLAMA_SPEC_SIGNAL_LOG=1|2, opt-in, diagnostics only) ----
    int      sig_level = 0;
    FILE *   sig_f     = nullptr;
    int64_t  sig_round = 0;
    int32_t  sig_n_vocab = 0;
    std::vector<int64_t> sig_open;   // round id waiting for accept(), per seq (-1 = none)
    std::vector<int64_t> sig_t0;     // draft() start of that round, per seq
    std::vector<int32_t> sig_req;    // request counter, per seq

    // target row statistics: argmax, top1-top2 logit margin, and (full) top-1 prob + entropy
    static void sig_row_stats(const float * l, int32_t n, bool full, int32_t & am, float & lm, float & p1, float & H) {
        float m1 = -1e30f, m2 = -1e30f;
        int32_t a = 0;
        for (int32_t j = 0; j < n; ++j) {
            const float v = l[j];
            if (v > m2) {
                if (v > m1) { m2 = m1; m1 = v; a = j; } else { m2 = v; }
            }
        }
        am = a; lm = m1 - m2; p1 = -1.0f; H = -1.0f;
        if (!full) {
            return;
        }
        // truncated softmax: terms below max-20 are < e^-20 each
        double z = 0.0, sx = 0.0;
        const float cut = m1 - 20.0f;
        for (int32_t j = 0; j < n; ++j) {
            const float v = l[j];
            if (v > cut) {
                const double x = (double) (v - m1);
                const double e = std::exp(x);
                z  += e;
                sx += e*x;
            }
        }
        p1 = (float) (1.0/z);
        H  = (float) (std::log(z) - sx/z);
    }

    void sig_log_verify(const common_batch & batch_in) {
        auto * ctx_tgt = this->params.ctx_tgt;
        if (sig_n_vocab <= 0) {
            sig_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_tgt)));
        }
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (sig_open[seq_id] < 0 || i_batch_beg[seq_id] < 0) {
                continue;
            }
            for (int32_t k = i_batch_beg[seq_id]; k <= i_batch_end[seq_id]; ++k) {
                const auto & t = batch_in.tokens[k];
                if (t.seq_id != seq_id || !t.output || t.id == LLAMA_TOKEN_NULL) {
                    continue;
                }
                const float * l = llama_get_logits_ith(ctx_tgt, k);
                if (!l) {
                    continue;   // e.g. backend sampling: no host logits
                }
                int32_t am; float lm, p1, H;
                sig_row_stats(l, sig_n_vocab, sig_level >= 2, am, lm, p1, H);
                fprintf(sig_f, "V seq=%d r=%lld row=%d tok=%d am=%d lm=%.4f p1=%.5f H=%.4f\\n",
                        (int) seq_id, (long long) sig_open[seq_id], (int) (k - i_batch_beg[seq_id]), (int) t.id, (int) am, lm, p1, H);
            }
        }
    }
""", "ev_P/ev_E members")

# 2. constructor: open the log (after the depth controller init in the constructor)
rep("""        ev.init();
        dctl.init(params.draft.n_max);
""", """        ev.init();
        dctl.init(params.draft.n_max);
        if (const char * e = getenv("LLAMA_SPEC_SIGNAL_LOG")) {
            sig_level = atoi(e);
        }
        if (sig_level > 0) {
            const char * fp = getenv("LLAMA_SPEC_SIGNAL_FILE");
            sig_f = fopen(fp ? fp : "/tmp/spec-signal.log", "a");
            if (!sig_f) {
                sig_level = 0;
            }
            LOG_INF("%s: speculative signal log level %d -> %s\\n", __func__, sig_level, fp ? fp : "/tmp/spec-signal.log");
        }
        sig_open.assign(n_seq, -1);
        sig_t0.assign(n_seq, 0);
        sig_req.assign(n_seq, 0);
""", "ctor dctl.init")

# 3. destructor: close
rep("""    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
""", """    ~common_speculative_impl_draft_mtp() override {
        if (sig_f) {
            fclose(sig_f);
            sig_f = nullptr;
        }
        auto * ctx_dft = this->params.ctx_dft;
""", "dtor")

# 4. begin(): request marker
rep("""        if (merge) {
            gen[seq_id]       = false;
            def_valid[seq_id] = false;
        }
""", """        if (merge) {
            gen[seq_id]       = false;
            def_valid[seq_id] = false;
        }

        if (sig_level > 0) {
            sig_req[seq_id]++;
            sig_open[seq_id] = -1;
            fprintf(sig_f, "B seq=%d req=%d n_prompt=%d\\n", (int) seq_id, (int) sig_req[seq_id], (int) prompt.size());
        }
""", "begin reset")

# 5. draft(): round start (per seq, after the first draft-step input is queued)
rep("""            i_last[seq_id] = idx;

            ev_last_p[seq_id].clear();
""", """            i_last[seq_id] = idx;

            if (sig_level > 0) {
                sig_open[seq_id] = ++sig_round;
                sig_t0[seq_id]   = ggml_time_us();
                fprintf(sig_f, "R seq=%d r=%lld pos0=%d depth=%d t_us=%lld\\n", (int) seq_id, (long long) sig_open[seq_id],
                        (int) dp.pos0, (int) (mtp_depth > 0 ? std::min(params.n_max, mtp_depth) : params.n_max), (long long) sig_t0[seq_id]);
            }

            ev_last_p[seq_id].clear();
""", "draft round start")

# 6. draft(): every drafted token that enters the verify batch (just before the MTP depth-cap check)
rep("""                if ((mtp_depth > 0 ? std::min(params.n_max, mtp_depth) : params.n_max) <= (int) result.size()) {
""", """                if (sig_level > 0 && sig_open[seq_id] >= 0) {
                    const float p1 = cur_p->data[0].p;
                    const float p2 = cur_p->size > 1 ? cur_p->data[1].p : 0.0f;
                    const float lm = cur_p->size > 1 ? cur_p->data[0].logit - cur_p->data[1].logit : 99.0f;
                    fprintf(sig_f, "D seq=%d r=%lld k=%d tok=%d p=%.5f m=%.5f lm=%.4f\\n", (int) seq_id, (long long) sig_open[seq_id],
                            (int) result.size(), (int) id, p1, p1 - p2, lm);
                }

                if ((mtp_depth > 0 ? std::min(params.n_max, mtp_depth) : params.n_max) <= (int) result.size()) {
""", "draft depth-cap check")

# 7. process(): verify rows (target logits are on the host here: process() runs right after the target decode,
#    before the server samples the same rows)
rep("""            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }
""", """            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        if (sig_level > 0 && batch_in.has_token() && n_tokens <= 17) {
            sig_log_verify(batch_in);
        }

        return true;
    }
""", "process end")

# 8. accept(): outcome of the round
rep("""        if (dctl.enabled) {
            dctl.last_acc = n_accepted;
        }
""", """        if (sig_level > 0 && sig_open[seq_id] >= 0) {
            fprintf(sig_f, "A seq=%d r=%lld nacc=%d dt_us=%lld dctl=%d\\n", (int) seq_id, (long long) sig_open[seq_id], (int) n_accepted,
                    (long long) (ggml_time_us() - sig_t0[seq_id]), dctl.enabled ? dctl.depth : -1);
            fflush(sig_f);
            sig_open[seq_id] = -1;
        }

        if (dctl.enabled) {
            dctl.last_acc = n_accepted;
        }
""", "accept last_acc")

open(p, "w", encoding="utf-8", newline="").write(s)
print("sigpatch ok")
