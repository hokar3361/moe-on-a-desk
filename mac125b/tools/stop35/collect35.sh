#!/bin/bash
# stop-rule data for a model: signal log at a fixed depth 7 (no depth controller, no merged catch-up), training prompts then held-out
# usage: M=<gguf> collect35.sh OUTDIR
OUT=$1; mkdir -p $OUT; PY=python3
cd ~/stop35
[ -f $OUT/train.json ] || $PY analyze.py prompts gendata2.py $OUT/train.json --n 48 --seed 1 > $OUT/prompts.log 2>&1
cp ~/mtpdata/heldout.json $OUT/held.json
run() {  # name prompts.json
  pkill -f "[l]lama-server"; sleep 2
  env LLAMA_SPEC_SIGNAL_LOG=1 LLAMA_SPEC_SIGNAL_FILE=$OUT/$1.log LLAMA_SPEC_MTP_DEPTH=7 GGML_ALLOC_FREE_DELAY=4 LLAMA_GRAPH_CACHE=16 \
    nohup $HOME/llama-sig35/build/bin/llama-server -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 --reasoning off \
    --spec-type draft-mtp --spec-draft-n-max 7 > $OUT/$1.server.log 2>&1 < /dev/null &
  for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
  python3 - "$2" <<PY
import json, sys, urllib.request
ps = json.load(open(sys.argv[1]))
for p in ps:
    p = p if isinstance(p, str) else (p.get("prompt") or p.get("content") or p)
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0}).encode()
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})).read()
print("done", len(ps))
PY
}
run train $OUT/train.json
run held $OUT/held.json
pkill -f "[l]lama-server"
$PY fit_s1.py $OUT/train.log $OUT/held.log > $OUT/fit.txt 2>&1
echo COLLECTDONE
