#!/bin/bash
# cold n-gram table rows: new prompts (never asked before), each asked once; LLAMA_PREFETCH_TOUCH=0 vs 1 alternate per prompt
# usage: coldab.sh BIN "extra server args" tag
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MTP=$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
BIN=$1; EXTRA=$2; TAG=$3
python3 - "$TAG" > /tmp/coldprompts.txt <<PY
import sys, random
random.seed(sys.argv[1])
who = ["a glassblower","a night-shift nurse","a retired astronaut","a beekeeper","a ferry pilot","a violin repairer","a cartographer","a tea farmer",
       "a subway musician","a weather balloon launcher","a puppet maker","a desert botanist","a clock tower keeper","a salt miner","a radio operator","a kite builder"]
what = ["an unsigned letter","a key that fits no lock","a recording of their own voice","a map drawn in pencil","a photograph from the future",
        "a seed that glows","a stopped watch","a coat with someone else's name","a message carved in ice","a stranger's diary"]
random.shuffle(who); random.shuffle(what)
for i in range(8): print("Write a short story about %s who discovers %s." % (who[i], what[i % len(what)]))
PY
i=0
while read -r P; do
  t=$((i % 2)); i=$((i+1))
  pkill -f "[l]lama-server"; sleep 2
  env LLAMA_PREFETCH_TOUCH=$t GGML_ALLOC_FREE_DELAY=4 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_MTP_MERGE=1 LLAMA_SPEC_STOP=1 nohup $BIN -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 --reasoning off $EXTRA > /tmp/coldab.log 2>&1 < /dev/null &
  for k in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
  r=$(curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":400,\"temperature\":0}" | python3 -c "import json,sys;d=json.load(sys.stdin);print('%.1f %d' % (d['timings']['predicted_per_second'], d['usage']['completion_tokens']))")
  echo "touch=$t $r | $P"
done < /tmp/coldprompts.txt
pkill -f "[l]lama-server"
