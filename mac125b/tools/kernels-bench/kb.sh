#!/bin/bash
# build kbench against the fork and run the per-type suite at real shapes (B tokens)
# usage: kb.sh [B] [check]
B=${1:-4}
R=$HOME/llama.cpp
cd $HOME/kbench || exit 1
clang++ -std=c++17 -O2 -I$R/ggml/include kbench.cpp -L$R/build/bin -lggml -lggml-base -lggml-cpu -lggml-metal -Wl,-rpath,$R/build/bin -o kbench || exit 1
run() { ./kbench "$@" 2>&1 | python3 -I $HOME/kbench/kparse.py; }
if [ "$2" = "check" ]; then
  for t in iq3_s iq2_s iq2_xs iq2_xxs iq3_xxs; do run check gu $t 2560 640 16 10 $B 1 1; done
  for t in iq4_nl q2_0; do run check dn $t 640 2560 16 10 $B 1 1; done
  for t in bf16 q6_K iq4_xs q4_K q5_K iq3_s; do run check dense $t 2560 2048 1 1 $B 1 1; done
  run check dense bf16 10240 320 1 1 $B 1 1
  run check dense bf16 320 10240 1 1 $B 1 1
  exit 0
fi
# MoE: L layers so the working set (L x ~28 experts) is far beyond the SLC
for t in iq3_s iq2_s iq2_xs iq2_xxs iq3_xxs; do run time gu $t 2560 640 512 10 $B 8 30; done
for t in iq4_nl q2_0; do run time dn $t 640 2560 512 10 $B 8 30; done
# dense: real shapes (hc down/up, router, attention-ish)
run time dense bf16 10240 320 1 1 $B 96 30
run time dense bf16 320 10240 1 1 $B 96 30
run time dense bf16 2560 512 1 1 $B 96 30
for t in q6_K iq4_xs q4_K q5_K iq3_s; do run time dense $t 2560 10240 1 1 $B 24 30; done
run time dense q6_K 6144 2560 1 1 $B 24 30
run time dense q5_K 2560 248320 1 1 $B 1 30
