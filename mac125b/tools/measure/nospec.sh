#!/bin/bash
# ours -12a build without speculation (no -md): same calls and timings as ~/mtpab.sh
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
OUT=$HOME/mtpab; pkill -f "[l]lama-server"; sleep 2
env GGML_ALLOC_FREE_DELAY=4 LLAMA_GRAPH_CACHE=16 nohup $HOME/llama-best/build/bin/llama-server -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 --reasoning off > $OUT/nospec.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
ask() { curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":400,\"temperature\":0}" > $2
  python3 -c "import json;d=json.load(open('$2'));print('%.1f' % d.get('timings',{}).get('predicted_per_second',0))"; }
b=$(python3 $HOME/bench-strata.py --url http://127.0.0.1:8080/v1 --lengths 1024,4096 --max-tokens 256 --repeat 3 2>&1 | awk '$1 ~ /^(1024|4096)$/ {printf "%s ", $4}')
c=$(ask "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling." $OUT/nospec-code.json)
p=$(ask "Write a short story about a lighthouse keeper who finds a message in a bottle." $OUT/nospec-prose.json)
echo "nospec | bench 1K/4K: $b| code $c | prose $p"
pkill -f "[l]lama-server"
