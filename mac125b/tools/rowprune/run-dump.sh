#!/bin/bash
# Start the dump-enabled llama-server (worktree ~/llama-rp = tag mac125b-best-2026-10-11d + dumppatch.py) with the
# best-checkpoint switches, writing verify-row records to $OUT. Not for timing (the dump splits the graph).
#   MODE=fixed7 (default): fixed draft depth 7, every draft position observed (probe.py truncates to any depth <= 7)
#   MODE=ctl             : the production depth controller (as run-best.sh); the dump overhead perturbs its round times
#   OUT=~/rowprune/dump-train.bin LAYERS=15,23,31,47 ~/rowprune/run-dump.sh      ~/rowprune/run-dump.sh stop
B=$HOME/llama-rp/build/bin/llama-server
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MTP=$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PORT=${PORT:-8080}
OUT=${OUT:-$HOME/rowprune/dump-train.bin}
LAYERS=${LAYERS:-15,23,31,47}
MODE=${MODE:-fixed7}
if [ "$1" = "stop" ]; then pkill -f "[l]lama-rp/build/bin/llama-server"; echo stopped; exit 0; fi
if curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok; then echo "port $PORT already serves something; stop it first"; exit 1; fi
mkdir -p "$(dirname "$OUT")"
if [ "$MODE" = "ctl" ]; then
    DEPTH="LLAMA_SPEC_DEPTH_CTL=1 LLAMA_SPEC_DEPTH_MIN=1 LLAMA_SPEC_DEPTH_RESET=1"
else
    DEPTH="LLAMA_SPEC_MTP_DEPTH=7"
fi
env GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 $DEPTH \
    LLAMA_ROWPRUNE_DUMP="$OUT" LLAMA_ROWPRUNE_LAYERS="$LAYERS" \
    nohup $B -m $M -md $MTP --spec-type draft-mtp --spec-draft-n-max 7 --reasoning off \
    -ngl 99 -c 16384 -np 1 --port $PORT > $HOME/rowprune/server.log 2>&1 < /dev/null &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 3; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && echo "ready (dump -> $OUT, log ~/rowprune/server.log)" || { echo "failed, see ~/rowprune/server.log"; exit 1; }
grep -m1 "rowprune:" $HOME/rowprune/server.log
