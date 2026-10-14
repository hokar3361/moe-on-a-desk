#!/bin/bash
# Chat / API server for Qwen3.8-Flash-Next (GSQ-RCO IQ3_XXS) on WSL2 + an NVIDIA GPU, OpenAI-compatible on 127.0.0.1:8080.
#   run-wsl.sh            about 22 GB of VRAM: expert cache 14 GB, draft (MTP) head with Q4_K/Q5_0 experts, draft depth 4
#   run-wsl.sh 12g        about 12 GB of VRAM: expert cache 6.7 GB, no draft head
#   run-wsl.sh stop
# Edit the paths below. CACHE_MIB overrides the expert-cache size (MiB), e.g. CACHE_MIB=9000 run-wsl.sh 12g.
MODEL_DIR=${MODEL_DIR:-$HOME/models}
M=${M:-$MODEL_DIR/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
MTP=${MTP:-$MODEL_DIR/mtp/mtp-Qwen3.8-Flash-Next-Q8e4.gguf}   # made by make-mtp-q4.sh
B=${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}

if [ "$1" = "stop" ]; then pkill -x llama-server; echo stopped; exit 0; fi
if [ "$1" = "12g" ]; then
  ARGS="--moe-cache-mib ${CACHE_MIB:-6700}"
else
  ARGS="--moe-cache-mib ${CACHE_MIB:-14000} -md $MTP --spec-type draft-mtp --spec-draft-n-max 4"
fi
pkill -x llama-server; sleep 2
exec env LLAMA_MOE_CACHE_FETCH=1 $B -m $M -c 16384 --port 8080 --host 127.0.0.1 -np 1 --reasoning off \
  -ngl 99 --cpu-moe $ARGS
