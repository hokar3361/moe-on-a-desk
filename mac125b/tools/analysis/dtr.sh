#!/bin/bash
# depth trace of the prose request in the fixed-metric order, for two controller variants
for v in "LLAMA_SPEC_DEPTH_KEEP_T=0" "LLAMA_SPEC_DEPTH_KEEP_T=1"; do
  E="GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_DEPTH_CTL=1 LLAMA_SPEC_DEPTH_MIN=1 LLAMA_SPEC_DEPTH_RESET=1 LLAMA_SPEC_DEPTH_TRACE=1 $v|--reasoning off --spec-draft-n-max 7"
  bash ~/mtpab.sh "dt:$E" 2>&1 | head -1
  python3 - <<'PY'
import re, collections
L = open("$HOME/mtpab/dt.log").read().split("\n")
# prose = the rounds after the last "prompt eval" line
last = max(i for i, l in enumerate(L) if "prompt eval time" in l)
c = collections.Counter(); ms = collections.defaultdict(list); acc = collections.defaultdict(list)
for l in L[last:]:
    m = re.search(r"d=(\d+) ms=([\d.]+) acc=(-?\d+)", l)
    if m:
        d = int(m.group(1)); c[d] += 1; ms[d].append(float(m.group(2))); acc[d].append(int(m.group(3)))
for d in sorted(c):
    print("  depth %d rounds %3d  mean ms %.1f  mean acc %.2f" % (d, c[d], sum(ms[d])/len(ms[d]), sum(acc[d])/len(acc[d])))
PY
done
