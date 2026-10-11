# depth controller round-time fixes:
#  - init() also clears the previous-round timestamp and depth, so the first round of a request (which spans prompt
#    processing / HTTP) is never measured
#  - the first measurement of an unmeasured depth is only taken when it is plausible (0.6..1.6x the prior)
#  - T(d) is monotone in d (a deeper round cannot be cheaper than a shallower one)
#  - the prior verify table follows the current verify pass (LLAMA_SPEC_DEPTH_FIX=0 restores the old behaviour)
import sys
p = sys.argv[1] + "/common/speculative.cpp"
s = open(p).read()
if "LLAMA_SPEC_DEPTH_FIX" not in s:
    old = """            depth = std::min(cap, 3);
            for (int k = 0; k <= 16; ++k) { a[k] = 0.7f; t[k] = 0.0f; }
        }"""
    assert s.count(old) == 1
    s = s.replace(old, """            depth = std::min(cap, 3);
            for (int k = 0; k <= 16; ++k) { a[k] = 0.7f; t[k] = 0.0f; }
            if (fix()) { t_prev_us = 0; last_depth = 0; last_acc = -1; }
        }
        static bool fix() {
            static const bool v = getenv("LLAMA_SPEC_DEPTH_FIX") == nullptr || atoi(getenv("LLAMA_SPEC_DEPTH_FIX")) != 0;
            return v;
        }
        // prior round time of depth d: verify of d+1 tokens + d draft steps
        static float prior(int d) {
            if (fix()) {
                // verify GPU time by width (2026-10-11 fork, 16k ctx server) + host, and 1.3 ms per draft step
                static const float v[] = { 0.0f, 18.5f, 23.0f, 26.6f, 29.3f, 32.0f, 35.0f, 38.0f, 41.0f };
                const int w = d + 1;
                const float tv = w <= 8 ? v[w] : 41.0f + 3.0f*(w - 8);
                return tv + 1.3f*d + 0.8f;
            }
            static const float v[] = { 0.0f, 21.0f, 28.8f, 31.6f, 34.3f, 37.2f, 40.1f, 43.0f, 46.0f };
            const int w = d + 1;
            const float tv = w <= 8 ? v[w] : 46.0f + 3.0f*(w - 8);
            return tv + 1.5f*d;
        }""")
    old = """        float T(int d) const {
            if (t[d] > 0.0f) return t[d];
            static const float v[] = { 0.0f, 21.0f, 28.8f, 31.6f, 34.3f, 37.2f, 40.1f, 43.0f, 46.0f };
            const int w = d + 1;
            const float tv = w <= 8 ? v[w] : 46.0f + 3.0f*(w - 8);
            return tv + 1.5f*d;
        }"""
    assert s.count(old) == 1
    s = s.replace(old, """        float T(int d) const {
            if (!fix()) {
                return t[d] > 0.0f ? t[d] : prior(d);
            }
            // monotone in d: each extra position costs at least 1 ms more
            float m = 0.0f;
            for (int k = 1; k <= d; ++k) {
                const float raw = t[k] > 0.0f ? t[k] : prior(k);
                m = std::max(raw, m + 1.0f);
            }
            return m;
        }""")
    old = """                const float prior = dctl.T(dctl.last_depth);"""
    assert s.count(old) == 1
    s = s.replace(old, """                const float prior = dctl.T(dctl.last_depth);
                // the first measurement of a depth must be plausible: one bad round would otherwise set it alone
                const bool plausible = !depth_ctl::fix() || td > 0.0f || (ms > 0.6f*depth_ctl::prior(dctl.last_depth) && ms < 1.6f*depth_ctl::prior(dctl.last_depth));""")
    old = """                if (ms < 2.0f*prior && ms < 150.0f) {
                    td = td > 0.0f ? 0.8f*td + 0.2f*ms : ms;"""
    assert s.count(old) == 1
    s = s.replace(old, """                if (plausible && ms < 2.0f*prior && ms < 150.0f) {
                    td = td > 0.0f ? 0.8f*td + 0.2f*ms : ms;""")
    open(p, "w").write(s)
print("dctlfix ok")
