#!/bin/bash
# Fixed metric on WSL (same calls as the Mac's ~/mtpab.sh: bench-strata 1K/4K, code/prose 400 greedy, server timings) plus the
# resources used while it runs: system CPU % (all 32 threads = 100 %), the server's RSS, and VRAM in use.
# usage: M=<gguf> abw.sh "name|bin|ENV=V ...|extra server args" ...
OUT=$HOME/abw; mkdir -p $OUT
ask() {
  curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":400,\"temperature\":0}" > $2
  python3 -c "import json;d=json.load(open('$2'));print('%.1f' % d.get('timings',{}).get('predicted_per_second',0))"
}
sample() {  # every second: cpu% rssGB vramGB
  local prev=($(head -1 /proc/stat))
  while true; do
    sleep 1
    local cur=($(head -1 /proc/stat))
    local idle=$(( ${cur[4]} + ${cur[5]} - ${prev[4]} - ${prev[5]} )); local tot=0
    for i in 1 2 3 4 5 6 7 8; do tot=$(( tot + ${cur[$i]} - ${prev[$i]} )); done
    prev=("${cur[@]}")
    local pid=$(pgrep -f "[l]lama-server -m" | head -1)
    local rss=0; [ -n "$pid" ] && rss=$(awk '/VmRSS/ {print $2}' /proc/$pid/status 2>/dev/null)
    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "$(( 100 * (tot - idle) / (tot > 0 ? tot : 1) )) $(( ${rss:-0} / 1048576 )) $(( vram / 1024 ))"
  done
}
for arm in "$@"; do
  IFS='|' read -r name bin envs args <<< "$arm"
  pkill -x llama-server; sleep 3
  env ${envs:-X_=0} nohup $bin -m $M -c 16384 --port 8080 --host 127.0.0.1 -np 1 --reasoning off $args > $OUT/$name.log 2>&1 < /dev/null &
  for i in $(seq 1 300); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; pgrep -f "[l]lama-server" >/dev/null || break; sleep 3; done
  if ! curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok; then echo "$name | failed: $(grep -iE 'error|fail|out of memory' $OUT/$name.log | head -2)"; continue; fi
  sample > $OUT/$name.res & SP=$!
  b=$(python3 ${BENCH:-$(dirname "$0")/../../mac125b/tools/measure/bench-strata.py} --url http://127.0.0.1:8080/v1 --lengths 1024,4096 --max-tokens 256 --repeat 3 2>&1 | awk '$1 ~ /^(1024|4096)$/ {printf "%s ", $4}')
  c=$(ask "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling." $OUT/$name-code.json)
  p=$(ask "Write a short story about a lighthouse keeper who finds a message in a bottle." $OUT/$name-prose.json)
  kill $SP 2>/dev/null
  res=$(awk '{c+=$1; if($1>cm)cm=$1; if($2>r)r=$2; if($3>v)v=$3; n++} END {printf "cpu avg %d%% max %d%% | rss max %d GB | vram max %d GB", c/n, cm, r, v}' $OUT/$name.res)
  acc=$(grep -E "draft acceptance" $OUT/$name.log | tail -2 | sed -e "s/.*mean len = *//" | tr "\n" " ")
  echo "$name | bench 1K/4K: $b| code $c | prose $p | len: $acc| $res"
done
pkill -x llama-server
