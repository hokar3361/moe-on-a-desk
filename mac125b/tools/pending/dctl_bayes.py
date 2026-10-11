# depth controller, acceptance model v2 (LLAMA_SPEC_DEPTH_BAYES=1, opt-in until measured):
#  - per position k: successes s[k] and trials n[k], counted only when position k was reached (k <= accepted+1)
#  - estimate a[k] = (s[k] + w*prior_k) / (n[k] + w), prior_k = estimate of position k-1 (position 1: 0.7), w = 4
#    -> an unreached deep position is as good as the last reached one, not a fixed optimistic 0.7
#  - on a request reset only the counts are cleared; round times are kept (LLAMA_SPEC_DEPTH_KEEP_T semantics)
import sys
p = sys.argv[1] + "/common/speculative.cpp"
s = open(p).read()
if "LLAMA_SPEC_DEPTH_BAYES" in s:
    print("already applied"); sys.exit(0)

old = """        float E(int d) const {
            float e = 1.0f, P = 1.0f;
            for (int k = 1; k <= d; ++k) { P *= a[k]; e += P; }
            return e;
        }"""
assert s.count(old) == 1
s = s.replace(old, """        static bool bayes() {
            static const bool v = getenv("LLAMA_SPEC_DEPTH_BAYES") != nullptr && atoi(getenv("LLAMA_SPEC_DEPTH_BAYES")) != 0;
            return v;
        }
        float bs[17] = {}, bn[17] = {};      // successes / trials per position (bayes)
        float acc_est(int k) const {
            if (!bayes()) return a[k];
            float prior = 0.7f;
            float est = 0.7f;
            for (int j = 1; j <= k; ++j) {
                est = (bs[j] + 4.0f*prior)/(bn[j] + 4.0f);
                prior = est;
            }
            return est;
        }
        float E(int d) const {
            float e = 1.0f, P = 1.0f;
            for (int k = 1; k <= d; ++k) { P *= acc_est(k); e += P; }
            return e;
        }""")

old = """            for (int k = 0; k <= 16; ++k) { a[k] = 0.7f; if (reset_t) t[k] = 0.0f; stale[k] = true; }"""
assert s.count(old) == 1
s = s.replace(old, """            for (int k = 0; k <= 16; ++k) { a[k] = 0.7f; if (reset_t && !bayes()) t[k] = 0.0f; stale[k] = true; bs[k] = 0.0f; bn[k] = 0.0f; }""")

old = """            for (int k = 1; k <= tried; ++k) {
                const float hit = k <= (int) n_accepted ? 1.0f : 0.0f;
                dctl.a[k] = 0.9f*dctl.a[k] + 0.1f*hit;
            }"""
assert s.count(old) == 1
s = s.replace(old, """            for (int k = 1; k <= tried; ++k) {
                const float hit = k <= (int) n_accepted ? 1.0f : 0.0f;
                dctl.a[k] = 0.9f*dctl.a[k] + 0.1f*hit;
                // bayes: decayed counts so the estimate follows the text (effective window ~30 trials)
                dctl.bn[k] = 0.97f*dctl.bn[k] + 1.0f;
                dctl.bs[k] = 0.97f*dctl.bs[k] + hit;
            }""")
open(p, "w").write(s)
print("dctl_bayes ok")
