# print stage-0 tables from s0eval JSON files. usage: s0report.py a.json [b.json ...]
import sys, json
for f in sys.argv[1:]:
    r = json.load(open(f))
    print("==", f, "pt", r["pt"], "mode", r["mode"])
    for name, groups in r["res"].items():
        for g in ["code", "prose", "narr", "story", "poem", "review"]:
            if g not in groups: continue
            x = groups[g]
            print("%-5s %-6s texts %3d starts %6d" % (name, g, x["texts"], x["starts"]))
            print("   cond   " + " ".join("%.3f" % v for v in x["cond"]))
            print("   top2   " + " ".join("%.3f" % v[0] for v in x["topk"]))
            print("   top4   " + " ".join("%.3f" % v[1] for v in x["topk"]))
            print("   top8   " + " ".join("%.3f" % v[2] for v in x["topk"]))
            print("   marg   " + " ".join("%.3f" % v for v in x["marginal"]))
            print("   sim    " + "  ".join("d%s %.3f" % (d, s["tok_per_round"]) for d, s in sorted(x["sim"].items())))
