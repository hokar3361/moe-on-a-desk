#!/bin/bash
# GPU time of one forward pass over B tokens (llama-bench pp B), with optional ablation env.
# usage: profb.sh B cfg...
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
B=$1; shift
for cfg in "$@"; do
  r=$(env $cfg GGML_METAL_TIMING=1 $HOME/llama.cpp/build/bin/llama-bench -m $M -ngl 99 -p $B -n 0 -r 15 -ub $B -b $B 2>&1 | python3 -c '
import re, sys, statistics as st, collections
rows=[[float(v) for v in m.groups()] for m in (re.search(r"nodes=(\d+) n_cb=\d+ encode=[\d.]+ gpu_span=[\d.]+ gpu_sum=([\d.]+)", l) for l in sys.stdin) if m]
c=collections.Counter(int(r[0]) for r in rows); n=c.most_common(1)[0][0]
d=[r[1] for r in rows if int(r[0])==n][3:]
print("%.2f" % st.median(d))')
  echo "B=$B $cfg gpu_ms=$r"
done
