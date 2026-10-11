# Drive a dump-enabled llama-server (see run-dump.sh) with the MTP-training prompt set of tools/mtp/gendata2.py
# (fixed-metric / bench topics excluded by its BANNED regex) or with its held-out set. Stdlib only.
# The server writes the verify-row records itself (LLAMA_ROWPRUNE_DUMP); this script only sends the requests and
# writes a sidecar (one JSON line per request, in order) so probe.py can split by prompt and report code vs prose.
#
#   python3 collect.py --gendata ~/kbench/gendata2.py --set train --n 200 --sidecar ~/rowprune/train.jsonl
#   python3 collect.py --gendata ~/kbench/gendata2.py --set held  --sidecar ~/rowprune/held.jsonl
import argparse, importlib.util, json, re, sys, time, urllib.request

FIXED = re.compile(r"lighthouse|bottle|iso.?8601|weekday", re.I)   # fixed-metric topics (gendata2 also bans these)

ap = argparse.ArgumentParser()
ap.add_argument("--gendata", required=True, help="path to tools/mtp/gendata2.py")
ap.add_argument("--set", choices=["train", "held"], default="train")
ap.add_argument("--n", type=int, default=200, help="number of prompts (train set; held uses all)")
ap.add_argument("--skip", type=int, default=0, help="skip the first N prompts of the set (resume)")
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
ap.add_argument("--sidecar", required=True)
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("gendata2", a.gendata)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)          # module level only defines lists and functions (main is __name__-guarded)
train, held = g.build()
prompts = held if a.set == "held" else train[:a.n]
prompts = prompts[a.skip:]
for p in prompts:
    assert not FIXED.search(p), p

def kind(p):
    if re.match(r"Write an? \S+ function", p):
        return "code"
    if p.startswith("Answer step by step"):
        return "qa"
    return "prose"

with open(a.sidecar, "a") as side:
    for i, p in enumerate(prompts):
        body = {"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens, "temperature": 0}
        req = urllib.request.Request(a.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        u, tm = d.get("usage", {}), d.get("timings", {})
        side.write(json.dumps({"i": a.skip + i, "set": a.set, "kind": kind(p), "prompt": p,
                               "n_out": u.get("completion_tokens"), "draft_n": tm.get("draft_n"),
                               "draft_n_accepted": tm.get("draft_n_accepted"), "wall_s": round(time.time() - t0, 2),
                               "text": d["choices"][0]["message"]["content"]}) + "\n")
        side.flush()
        print(a.skip + i, kind(p), u.get("completion_tokens"), "%.1f s" % (time.time() - t0), flush=True)
