#!/bin/bash
# GPU time of one forward over B tokens, with op classes skipped (exclusion method), model $M, build of this fork
for B in 1 4 8; do
  for cfg in "X_=0" GGML_METAL_PROF_SKIP=MUL_MAT_ID GGML_METAL_PROF_SKIP=MUL_MAT.q8_0 GGML_METAL_PROF_SKIP=MUL_MAT_ID.q4_K GGML_METAL_PROF_SKIP=MUL_MAT_ID.q5_K GGML_METAL_PROF_SKIP=MUL_MAT_ID,MUL_MAT.q8_0; do
    r=$(env $cfg GGML_ALLOC_FREE_DELAY=4 GGML_METAL_TIMING=1 $HOME/llama.cpp/build/bin/llama-bench -m $M -ngl 99 -p $B -n 0 -r 15 -ub $B -b $B 2>&1 | python3 -c '
import re, sys, statistics as st, collections
rows=[[float(v) for v in m.groups()] for m in (re.search(r"nodes=(\d+) n_cb=\d+ encode=[\d.]+ gpu_span=[\d.]+ gpu_sum=([\d.]+)", l) for l in sys.stdin) if m]
c=collections.Counter(int(r[0]) for r in rows); n=c.most_common(1)[0][0]
d=[r[1] for r in rows if int(r[0])==n][3:]
print("%.2f nodes=%d" % (st.median(d), n))')
    echo "B=$B $cfg gpu_ms=$r"
  done
done
