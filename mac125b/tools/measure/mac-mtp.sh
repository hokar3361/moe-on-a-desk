#!/bin/bash
# Same server, same prompts, greedy; the only change between runs is MTP speculation.
set -u
BIN=$HOME/llama.cpp/build/bin/llama-server
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MTP=$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
OUT=$HOME/mtp-cmp; mkdir -p "$OUT"
COMMON="-m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1"

run() {
  local name=$1; shift
  pkill -f "[l]lama-server" 2>/dev/null; sleep 3
  nohup "$BIN" $COMMON "$@" > "$OUT/$name-server.log" 2>&1 < /dev/null &
  for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 5; done
  echo "=== $name ==="
  python3 "$HOME/bench-strata.py" --url http://127.0.0.1:8080/v1 --lengths 1024,4096 --max-tokens 256 --repeat 3 2>&1 | tail -3
  curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling."}],"max_tokens":300,"temperature":0}' \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" > "$OUT/$name-answer.txt"
  grep -i -E "draft acceptance|accepted|n_draft|spec" "$OUT/$name-server.log" | tail -3
}

run base
run mtp -md "$MTP" --spec-type draft-mtp
pkill -f "[l]lama-server"
echo "=== identical greedy answers? ==="
if cmp -s "$OUT/base-answer.txt" "$OUT/mtp-answer.txt"; then echo IDENTICAL; else echo DIFFERENT; diff "$OUT/base-answer.txt" "$OUT/mtp-answer.txt" | head -10; fi
echo MTP_CMP_DONE
