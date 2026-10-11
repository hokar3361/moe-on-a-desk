#!/bin/bash
# generality check (reference only, not the fixed metric): per-prompt decode tok/s on the 18 held-out prompts
# arms: start (unmodified 781dbc5, MTP depth 3), best tag, current fork
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MTP=$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
BEST="GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_DEPTH_CTL=1 LLAMA_SPEC_DEPTH_MIN=1 LLAMA_SPEC_DEPTH_RESET=1"
run() { # name bin env nmax
  pkill -f "[l]lama-server"; sleep 2
  env $3 nohup $2 -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 -md $MTP --spec-type draft-mtp --reasoning off --spec-draft-n-max $4 > /tmp/held_$1.log 2>&1 < /dev/null &
  for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
  python3 - "$1" <<PY
import json, sys, urllib.request
name = sys.argv[1]
ps = json.load(open("$HOME/mtpdata/heldout.json"))
out = []
for p in ps:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})))
    out.append(r["timings"]["predicted_per_second"])
json.dump(out, open("/tmp/held3_%s.json" % name, "w"))
print(name, " ".join("%.1f" % x for x in out))
PY
  pkill -f "[l]lama-server"
}
run e11e $HOME/llama-best/build/bin/llama-server "GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1" 7
run f11f $HOME/llama.cpp/build/bin/llama-server "GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1" 7
