#!/bin/bash
# A0: top-1 agreement of a same-model low-bit drafter (experts requantized, everything else unchanged) with the target.
# imatrix from the target's own generations on training prompts; agreement measured on the 18 held-out texts.
set -x
B=$HOME/llama.cpp/build/bin
D=$HOME/models/qwen38-gsq/IQ3_XXS
M=$D/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
cd $HOME/drafter
[ -f imat.gguf ] || $B/llama-imatrix -m $M -f imat.txt -c 512 -b 512 -ngl 99 --chunks 300 -o imat.gguf > imat.log 2>&1
[ -f base.kld ] || $B/llama-perplexity -m $M -f held.txt -c 512 -b 512 -ngl 99 --kl-divergence-base base.kld > base.log 2>&1
for t in ${TYPES:-iq2_xxs iq1_m}; do
  T=$(echo $t | tr a-z A-Z)
  [ -f d_$t.gguf ] || $B/llama-quantize --allow-requantize --imatrix imat.gguf --tensor-type-file types_$t.txt $M d_$t.gguf $T 12 > q_$t.log 2>&1
  ls -la d_$t.gguf
  $B/llama-perplexity -m d_$t.gguf -f held.txt -c 512 -b 512 -ngl 99 --kl-divergence-base base.kld --kl-divergence > kld_$t.log 2>&1
  grep -E "Same top|Mean.*KLD|PPL\(Q\)|Mean PPL" kld_$t.log
done
echo A0 done
