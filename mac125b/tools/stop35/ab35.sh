#!/bin/bash
# Fixed metric (same calls as ~/mtpab.sh: bench-strata 1K/4K, code/prose 400 greedy, server timings) for another model.
# usage: M=<gguf> ab35.sh "name|bin|ENV=V ...|extra server args" ...
OUT=$HOME/ab35; mkdir -p $OUT
ask() {
  curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":400,\"temperature\":0}" > $2
  python3 -c "import json;d=json.load(open('$2'));print('%.1f' % d.get('timings',{}).get('predicted_per_second',0))"
}
for arm in "$@"; do
  IFS='|' read -r name bin envs args <<< "$arm"
  pkill -f "[l]lama-server"; sleep 2
  env ${envs:-X_=0} nohup $bin -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 --reasoning off $args > $OUT/$name.log 2>&1 < /dev/null &
  for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; pgrep -f "[l]lama-server" >/dev/null || break; sleep 3; done
  if ! curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok; then echo "$name | failed: $(grep -iE 'error|fail' $OUT/$name.log | head -2)"; continue; fi
  b=$(python3 $HOME/bench-strata.py --url http://127.0.0.1:8080/v1 --lengths 1024,4096 --max-tokens 256 --repeat 3 2>&1 | awk '$1 ~ /^(1024|4096)$/ {printf "%s ", $4}')
  c=$(ask "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling." $OUT/$name-code.json)
  p=$(ask "Write a short story about a lighthouse keeper who finds a message in a bottle." $OUT/$name-prose.json)
  acc=$(grep -E "draft acceptance" $OUT/$name.log | tail -2 | sed -e "s/.*mean len = *//" | tr "\n" " ")
  echo "$name | bench 1K/4K: $b| code $c | prose $p | mean len: $acc"
done
pkill -f "[l]lama-server"
