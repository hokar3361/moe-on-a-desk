#!/bin/bash
# choose c_draft on the TRAINING prompts (never the fixed metric or the held-out set): mean decode tok/s over 24 of them
cd ~/stop35/q36
B=$HOME/llama.cpp/build/bin/llama-server; M=$HOME/models/qwen36-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
for c in 1.2 1.9; do
  sed "s/^c_draft .*/c_draft $c/" q36b.params > tune_$c.params
  pkill -f "[l]lama-server"; sleep 2
  env GGML_ALLOC_FREE_DELAY=4 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1 LLAMA_SPEC_STOP_PARAMS=$PWD/tune_$c.params nohup $B -m $M -ngl 99 -c 16384 --port 8080 \
    --host 127.0.0.1 -np 1 --reasoning off --spec-type draft-mtp --spec-draft-n-max 7 > /tmp/tune_$c.log 2>&1 < /dev/null &
  for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
  python3 - $c <<PY
import json, sys, urllib.request
ps = json.load(open("train.json"))[::2]
tot = tim = 0.0; code = []; prose = []
for p in ps:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})))
    s = r["timings"]["predicted_per_second"]; n = r["usage"]["completion_tokens"]; tot += n; tim += n / s
    (code if any(w in p.lower() for w in ("function", "code", "program", "class", "script", "implement", "sql", "query")) else prose).append(s)
print("c_draft %s: mean %.1f tok/s over %d (code-like %.1f n=%d, other %.1f n=%d)" % (sys.argv[1], tot / tim, len(ps), sum(code) / max(1, len(code)), len(code), sum(prose) / max(1, len(prose)), len(prose)))
PY
done
pkill -f "[l]lama-server"; echo TUNEDONE
