# speculative.cpp: per-draft stop rule (opt-in LLAMA_SPEC_STOP=1), the MTP head's own confidence calibrated per position.
# Model: logistic regression fitted on 48 training prompts (tools/depthsignal/fit_s1.py; fixed-metric topics excluded),
# held-out AUC 0.888. Objective: maximise E[tokens] - lam * E[ms] (lam = long-run rate), multi-step look-ahead.
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
NL = chr(92) + "n"

# 1. state struct, before the depth controller struct
old = "    struct depth_ctl {"
assert s.count(old) == 1
s = s.replace(old, """    // LLAMA_SPEC_STOP=1: stop drafting when one more draft no longer pays, from the MTP head's own confidence
    struct stop_rule {
        bool enabled = false;
        static constexpr int NF = 9;
        // S1 models (fit_s1.py on train.log, 48 prompts): post = draft k known, pre = before drafting step k
        float mu_post[NF] = { 0.298195f, 0.233429f, 0.166603f, 0.301772f, 1.1594f, 0.829856f, 4.18254f, 1.55277f, 2.51477f };
        float sd_post[NF] = { 0.457466f, 0.423013f, 0.372621f, 0.459027f, 0.889215f, 0.165111f, 3.5908f, 0.828812f, 3.13468f };
        float w_post[NF + 1] = { 0.260389f, -0.00761729f, -0.06875f, -0.196675f, 0.0883201f, -0.0497614f, 2.89242f, -0.455195f, 0.0654567f, 2.19465f };
        float mu_pre[NF] = { 0.298195f, 0.233429f, 0.166603f, 0.301772f, 1.1594f, 0.829856f, 3.71632f, 1.2723f, 2.51477f };
        float sd_pre[NF] = { 0.457466f, 0.423013f, 0.372621f, 0.459027f, 0.889215f, 0.165111f, 3.71476f, 1.03591f, 3.13468f };
        float w_pre[NF + 1] = { 0.178326f, -0.0654108f, -0.0691572f, -0.0613016f, 0.396043f, -0.030329f, 0.444108f, -0.243838f, 0.141795f, 1.09217f };
        float qbar[18] = { 1.0f, 0.7827f, 0.7136f, 0.6951f, 0.7031f, 0.7329f, 0.7491f, 0.7464f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f, 0.7f };
        // verify GPU+host ms by width (index = tokens in the verify batch)
        float tv[18] = { 0.0f, 17.9f, 21.85f, 25.8f, 28.63f, 31.47f, 34.3f, 37.65f, 41.0f, 44.0f, 47.0f, 50.0f, 53.0f, 56.0f, 59.0f, 62.0f, 65.0f, 68.0f };
        float c_draft = 1.3f;
        float lam     = 0.07f;          // long-run tokens per ms, EMA over rounds
        float ema[18];                  // per-position acceptance EMA (0.9), reset per request
        int   cap = 7, dmin = 1;
        int64_t t_prev_us = 0;
        int   last_n = 0, last_acc = -1;
        bool  first_round = true;
        int   first_pos = -1;
        uint64_t rounds = 0, n_drafted = 0, n_stop_post = 0, n_stop_pre = 0;
        static float lg(float p) { p = std::min(std::max(p, 1e-4f), 1.0f - 1e-4f); return logf(p/(1.0f - p)); }
        void reset_request() {
            for (int k = 0; k < 18; ++k) ema[k] = 0.7f;
            first_round = true; t_prev_us = 0; last_acc = -1; last_n = 0; first_pos = -1;
        }
        // k 1-based; p/lm: the draft's top-1 prob and top1-top2 logit margin (post: draft k, pre: draft k-1); minlg: min lg p of drafts 1..k-1
        float q(bool post, int k, float p, float lm, float minlg, int pos_rel) const {
            const float * mu = post ? mu_post : mu_pre;
            const float * sd = post ? sd_post : sd_pre;
            const float * w  = post ? w_post  : w_pre;
            float x[NF];
            const int kk = std::min(k, 4);
            for (int j = 0; j < 4; ++j) x[j] = kk == j + 1 ? 1.0f : 0.0f;
            x[4] = lg(ema[std::min(k, 17)]);
            x[5] = log1pf((float) std::max(pos_rel, 0))/6.0f;
            const bool has = post || k >= 2;
            x[6] = has ? lg(p) : 0.0f;
            x[7] = has ? log1pf(std::max(lm, 0.0f)) : 0.0f;
            x[8] = k >= 2 ? minlg : 0.0f;
            float z = w[NF];
            for (int j = 0; j < NF; ++j) z += (x[j] - mu[j])/sd[j]*w[j];
            z = std::max(-30.0f, std::min(30.0f, z));
            return 1.0f/(1.0f + expf(-z));
        }
        // value of drafts k..k+m-1 (best m >= 1): P_{k-1}*q*qbar... minus lam*(verify growth + draft steps not yet paid)
        float value(int k, float Pprev, float qk, bool step_paid) const {
            float acc = 0.0f, best = -1e30f, P = Pprev;
            for (int j = k; j <= cap; ++j) {
                P *= j == k ? qk : qbar[std::min(j, 17)];
                const float dtv = tv[std::min(j + 1, 17)] - tv[std::min(j, 17)];
                acc += P - lam*(dtv + ((j > k || !step_paid) ? c_draft : 0.0f));
                best = std::max(best, acc);
            }
            return best;
        }
    };

""" + old)

# 2. members + init
old = "    std::vector<float> ev_P, ev_E;                // running chain acceptance and expected accepted, per seq"
assert s.count(old) == 1
s = s.replace(old, old + """
    stop_rule sr;
    std::vector<float> sr_P, sr_minlg, sr_lastp, sr_lastlm;   // per seq: chain acceptance, weakest draft, last draft p / margin
    std::vector<int>   sr_n;                                   // per seq: drafts in the last round""")
old = "        ev_P.assign(n_seq, 1.0f);"
assert s.count(old) == 1
s = s.replace(old, old + """
        sr.enabled = getenv("LLAMA_SPEC_STOP") != nullptr && atoi(getenv("LLAMA_SPEC_STOP")) != 0;
        if (sr.enabled) {
            sr.cap = params.draft.n_max;
            if (const char * m = getenv("LLAMA_SPEC_DEPTH_MIN")) sr.dmin = std::max(1, atoi(m));
            if (const char * e = getenv("LLAMA_SPEC_STOP_LAM")) sr.lam = (float) atof(e);
            sr.reset_request();
            LOG_INF("%s: draft stop rule on (cap %d, min %d)""" + NL + """", __func__, sr.cap, sr.dmin);
        }
        sr_n.assign(n_seq, 0); sr_P.assign(n_seq, 1.0f); sr_minlg.assign(n_seq, 0.0f); sr_lastp.assign(n_seq, 1.0f); sr_lastlm.assign(n_seq, 0.0f);""")

# 3. per-request reset
old = "        // LLAMA_SPEC_DEPTH_RESET=1: the depth controller starts from its priors for every request"
assert s.count(old) == 1
s = s.replace(old, """        if (sr.enabled) {
            sr.reset_request();
        }

""" + old)

# 4. round timing -> lam, at the start of draft()
old = """        if (dctl.enabled) {
            const int64_t now = ggml_time_us();
"""
assert s.count(old) == 1
s = s.replace(old, """        if (sr.enabled) {
            const int64_t now = ggml_time_us();
            if (sr.t_prev_us > 0 && sr.last_acc >= 0 && !sr.first_round) {
                const float ms = (now - sr.t_prev_us)/1000.0f;
                if (ms > 5.0f && ms < 150.0f) {
                    sr.lam = 0.9f*sr.lam + 0.1f*(sr.last_acc + 1)/ms;
                }
            }
            sr.first_round = sr.t_prev_us == 0;
            sr.t_prev_us = now;
            sr.last_acc = -1;
        }
""" + old)

# 5. per-seq chain reset at the start of drafting
old = """            ev_last_p[seq_id].clear();
            ev_P[seq_id] = 1.0f;
            ev_E[seq_id] = 1.0f;
"""
assert s.count(old) == 1
s = s.replace(old, old + """            sr_P[seq_id] = 1.0f; sr_minlg[seq_id] = 0.0f; sr_n[seq_id] = 0;
            if (sr.enabled && sr.first_pos < 0) {
                sr.first_pos = dparams[seq_id].pos0;
            }
""")

# 6. post decision (before the EV planner block) and pre decision (before the depth cap check)
old = "                // EV planner: include this token only if it improves expected tokens per round time"
assert s.count(old) == 1
s = s.replace(old, """                if (sr.enabled) {
                    const int   k   = (int) dparams.at(seq_id).result->size() + 1;
                    const float p1  = cur_p->data[0].p;
                    const float lm  = cur_p->size > 1 ? cur_p->data[0].logit - cur_p->data[1].logit : 99.0f;
                    const int   pr  = dparams.at(seq_id).pos0 - sr.first_pos;
                    const float qk  = sr.q(true, k, p1, lm, sr_minlg[seq_id], pr);
                    if (k > sr.dmin && sr.value(k, sr_P[seq_id], qk, true) <= 0.0f) {
                        sr.n_stop_post++;
                        drafting[seq_id] = false;
                        n_drafting--;
                        continue;
                    }
                    sr_P[seq_id] *= qk;
                    sr_minlg[seq_id] = k == 1 ? stop_rule::lg(p1) : std::min(sr_minlg[seq_id], stop_rule::lg(p1));
                    sr_lastp[seq_id] = p1; sr_lastlm[seq_id] = lm;
                }

""" + old)

old = """                if ((mtp_depth > 0 ? std::min(params.n_max, mtp_depth) : params.n_max) <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }
"""
assert s.count(old) == 1
s = s.replace(old, """                if (sr.enabled) {
                    sr_n[seq_id] = (int) result.size();
                }
""" + old + """
                if (sr.enabled && (int) result.size() >= sr.dmin) {
                    const int   k  = (int) result.size() + 1;   // the step being decided
                    const int   pr = dp.pos0 - sr.first_pos;
                    const float qp = sr.q(false, k, sr_lastp[seq_id], sr_lastlm[seq_id], sr_minlg[seq_id], pr);
                    if (sr.value(k, sr_P[seq_id], qp, false) <= 0.0f) {
                        sr.n_stop_pre++;
                        drafting[seq_id] = false;
                        n_drafting--;
                        continue;
                    }
                }
""")

# 7. accept(): per-position EMA, accepted count for lam
old = """        if (dctl.enabled) {
            dctl.last_acc = n_accepted;
        }"""
assert s.count(old) == 1
s = s.replace(old, old + """
        if (sr.enabled) {
            const int n = sr_n[seq_id];
            const int tried = std::min<int>(n, n_accepted + 1);
            for (int k = 1; k <= tried && k < 18; ++k) {
                sr.ema[k] = 0.9f*sr.ema[k] + 0.1f*(k <= (int) n_accepted ? 1.0f : 0.0f);
            }
            sr.last_acc = n_accepted;
            sr.rounds++; sr.n_drafted += n;
            if (sr.rounds % 200 == 0) {
                LOG_INF("%s: stop rule rounds %llu, drafts/round %.2f, stops post %llu pre %llu, lam %.4f""" + NL + """", __func__,
                        (unsigned long long) sr.rounds, (double) sr.n_drafted/sr.rounds,
                        (unsigned long long) sr.n_stop_post, (unsigned long long) sr.n_stop_pre, sr.lam);
            }
        }""")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("stoprule ok")
