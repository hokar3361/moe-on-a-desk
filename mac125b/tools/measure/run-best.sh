#!/bin/bash
# mac125b best state (tag mac125b-best-2026-10-11f, pinned build in ~/llama-best):
# Qwen3.8-Flash-Next 125B IQ3_XXS + MTP head, thinking off, OpenAI-compatible server on http://127.0.0.1:${PORT:-8080}/v1
#   ~/run-best.sh            start in the background (log: ~/llama-best.log), waits until ready
#   ~/run-best.sh stop       stop it
#   PORT=8081 HOST=0.0.0.0 ~/run-best.sh   other port / listen on the LAN
# Mechanisms (all new code in the fork): graph cache per small-batch shape, delayed frees in the graph allocator,
# MTP catch-up merged into the first draft step, per-draft stop rule from the MTP head's calibrated confidence, q2_0 few-token MoE kernel, bf16 few-row MMA from 2 rows, coarse q1_0 draft head
# with exact top-32 re-scoring, spin wait on the GPU.
B=$HOME/llama-best/build/bin/llama-server
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MTP=$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PORT=${PORT:-8080}; HOST=${HOST:-127.0.0.1}
if [ "$1" = "stop" ]; then pkill -f "[l]lama-best/build/bin/llama-server"; echo stopped; exit 0; fi
if curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok; then echo "port $PORT already serves something; stop it first"; exit 1; fi
env GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1 \
    nohup $B -m $M -md $MTP --spec-type draft-mtp --spec-draft-n-max 7 --reasoning off \
    -ngl 99 -c 16384 -np 1 --host $HOST --port $PORT > $HOME/llama-best.log 2>&1 < /dev/null &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 3; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && echo "ready: http://$HOST:$PORT/v1 (log ~/llama-best.log)" || { echo "failed to start, see ~/llama-best.log"; exit 1; }
