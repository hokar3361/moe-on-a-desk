# speculative.cpp: with kept round times (bayes) every measurement must be near the prior, not only the first one
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = "                const bool plausible = !depth_ctl::fix() || td > 0.0f || (ms > 0.6f*depth_ctl::prior(dctl.last_depth) && ms < 1.6f*depth_ctl::prior(dctl.last_depth));"
assert s.count(old) == 1
s = s.replace(old, """                const bool near_prior = ms > 0.6f*depth_ctl::prior(dctl.last_depth) && ms < 1.6f*depth_ctl::prior(dctl.last_depth);
                const bool plausible = !depth_ctl::fix() || (depth_ctl::bayes() ? near_prior : (td > 0.0f || near_prior));""")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
