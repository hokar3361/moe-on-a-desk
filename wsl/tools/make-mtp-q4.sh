#!/bin/bash
# Smaller draft (MTP) head: its 512 experts (2.55 GB in Q8_0) re-quantized to Q4_K (gate/up) and Q5_0 (down: 640 columns are not a
# multiple of 256, llama-quantize falls back to Q5_0); everything else stays Q8_0. 4.14 GB -> 2.98 GB, about 1.1 GB less VRAM.
# The head only drafts: every drafted token is verified by the full model, so the output of the model does not change
# (checked: same md5 at the same draft depth); only the acceptance of drafts could change (it did not measurably).
SRC=${1:-$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}
DST=${2:-$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8e4.gguf}
Q=${LLAMA_QUANTIZE:-$HOME/llama.cpp/build/bin/llama-quantize}
$Q --allow-requantize \
  --tensor-type "ffn_gate_exps=q4_k" --tensor-type "ffn_up_exps=q4_k" --tensor-type "ffn_down_exps=q4_k" \
  "$SRC" "$DST" Q8_0
