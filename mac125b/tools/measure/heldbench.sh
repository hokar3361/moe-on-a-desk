#!/bin/bash
# generality check: mean decode tok/s over the held-out prompts (never used for training nor in the fixed metric)
# usage: heldbench.sh NAME MTPFILE "ENV..."
BIN=$HOME/llama.cpp/build/bin/llama-server
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
pkill -f "[l]lama-server"; sleep 2
env $3 nohup $BIN -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 -md $2 --spec-type draft-mtp --reasoning off --spec-draft-n-max 7 > /tmp/held_$1.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
python3 -I $HOME/kbench/streambench.py http://127.0.0.1:8080/v1 m file:$HOME/mtpdata/heldout.json 400 > /tmp/held_$1.txt
pkill -f "[l]lama-server"
awk -v n=$1 '{s+=$1; c++} END {printf "%s held-out mean %.1f tok/s over %d prompts\n", n, s/c, c}' /tmp/held_$1.txt
