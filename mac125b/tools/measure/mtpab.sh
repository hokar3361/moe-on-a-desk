#!/bin/bash
# Fixed metric (DESIGN.md 2.3), MTP on: bench 1K/4K + code/prose 400 greedy; answer must equal the baseline run.
# usage: mtpab.sh "NAME:ENV=VAL ENV2=VAL" ...
BIN=${BIN_OVERRIDE:-$HOME/llama.cpp/build/bin/llama-server}
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MTP=${MTP_OVERRIDE:-$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}
OUT=$HOME/mtpab; mkdir -p $OUT
ask() {  # prompt, file
  curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":400,\"temperature\":0}" > $2
  python3 -c "import json,sys;d=json.load(open('$2'));t=d.get('timings',{});print('%.1f' % t.get('predicted_per_second',0))"
}
for arm in "$@"; do
  # arm = NAME:ENV=VAL ...|extra server args
  name=${arm%%:*}; rest=${arm#*:}; envs=${rest%%|*}; EXTRA=""
  [ "$rest" != "$envs" ] && EXTRA=${rest#*|}
  pkill -f "[l]lama-server"; sleep 2
  env $envs nohup $BIN -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 -md $MTP ${SPEC:---spec-type draft-mtp} $EXTRA > $OUT/$name.log 2>&1 < /dev/null &
  for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
  b=$(python3 $HOME/bench-strata.py --url http://127.0.0.1:8080/v1 --lengths 1024,4096 --max-tokens 256 --repeat 3 2>&1 | awk '$1 ~ /^(1024|4096)$/ {printf "%s ", $4}')
  c=$(ask "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling." $OUT/$name-code.json)
  p=$(ask "Write a short story about a lighthouse keeper who finds a message in a bottle." $OUT/$name-prose.json)
  python3 -c "import json;open('$OUT/$name-code.txt','w').write(json.load(open('$OUT/$name-code.json'))['choices'][0]['message']['content'])"
  python3 -c "import json;open('$OUT/$name-prose.txt','w').write(json.load(open('$OUT/$name-prose.json'))['choices'][0]['message']['content'])"
  acc=$(grep -E "draft acceptance" $OUT/$name.log | tail -2 | sed -e "s/.*mean len = *//" | tr "\n" " ")
  echo "$name | bench 1K/4K: $b| code $c | prose $p | mean len (code,prose): $acc| $(grep -m1 "coarse head" $OUT/$name.log | sed -e 's/.*draft coarse/coarse/')"
done
pkill -f "[l]lama-server"
first=${1%%:*}
for arm in "$@"; do n=${arm%%:*}; for k in code prose; do cmp -s $OUT/$first-$k.txt $OUT/$n-$k.txt && echo "$n $k == $first" || echo "$n $k DIFFERS from $first"; done; done
