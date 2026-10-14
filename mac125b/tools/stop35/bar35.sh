#!/bin/bash
# stage (barrier) overhead upper bound and per-class exclusion for the small ops, model $M, build of this fork
for B in 1 4; do
  for cfg in "X_=0" GGML_METAL_PROF_NOBARRIER=1 GGML_METAL_PROF_SKIP=GATED_DELTA_NET GGML_METAL_PROF_SKIP=SSM_CONV GGML_METAL_PROF_SKIP=FLASH_ATTN_EXT GGML_METAL_PROF_SKIP=RMS_NORM GGML_METAL_PROF_SKIP=GET_ROWS GGML_METAL_PROF_SKIP=MUL_MAT.f32 GGML_METAL_PROF_SKIP=MUL_MAT.bf16; do
    r=$(env $cfg GGML_ALLOC_FREE_DELAY=4 GGML_METAL_TIMING=1 $HOME/llama.cpp/build/bin/llama-bench -m $M -ngl 99 -p $B -n 0 -r 15 -ub $B -b $B 2>&1 | python3 -c '
import re, sys, statistics as st, collections
rows=[[float(v) for v in m.groups()] for m in (re.search(r"nodes=(\d+) n_cb=\d+ encode=[\d.]+ gpu_span=[\d.]+ gpu_sum=([\d.]+)", l) for l in sys.stdin) if m]
c=collections.Counter(int(r[0]) for r in rows); n=c.most_common(1)[0][0]
d=[r[1] for r in rows if int(r[0])==n][3:]
print("%.2f" % st.median(d))')
    echo "B=$B $cfg gpu_ms=$r"
  done
done
GGML_METAL_PROF_BARLOG=1 GGML_ALLOC_FREE_DELAY=4 $HOME/llama.cpp/build/bin/llama-bench -m $M -ngl 99 -p 1 -n 0 -r 1 -ub 1 -b 1 2>&1 | grep -iE "barrier|stage" | tail -3
