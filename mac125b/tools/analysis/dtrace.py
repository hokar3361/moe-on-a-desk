# LLAMA_SPEC_DEPTH_TRACE=1: log every measured round of the depth controller
import sys
BS = chr(92)
p = sys.argv[1] + "/common/speculative.cpp"
s = open(p).read()
# undo a broken earlier attempt, if any
i = s.find("                static const bool trace = getenv(\"LLAMA_SPEC_DEPTH_TRACE\") != nullptr;")
if i >= 0:
    j = s.index("                if (ms < 2.0f*prior && ms < 150.0f) {", i)
    s = s[:i] + s[j:]
old = """                const float prior = dctl.T(dctl.last_depth);
                if (ms < 2.0f*prior && ms < 150.0f) {"""
assert s.count(old) == 1
s = s.replace(old, """                const float prior = dctl.T(dctl.last_depth);
                static const bool trace = getenv("LLAMA_SPEC_DEPTH_TRACE") != nullptr;
                if (trace) {
                    LOG_INF("dctl-trace d=%d ms=%.2f acc=%d prior=%.1f t=%.1f""" + BS + """n", dctl.last_depth, ms, dctl.last_acc, prior, td);
                }
                if (ms < 2.0f*prior && ms < 150.0f) {""")
open(p, "w").write(s)
print("dtrace ok")
