#!/bin/bash
# Reference measurement of Strata on this Mac with the fixed method of ~/mtpab.sh (same bench-strata call, same prompts,
# same timings field). usage: strataab.sh NAME CONFIG.json
name=$1; cfg=$2; S=$HOME/opt/strata-0.1.44
OUT=$HOME/strataab; mkdir -p $OUT
ask() {
  curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":400,\"temperature\":0}" > $2
  python3 -c "import json,sys;d=json.load(open('$2'));t=d.get('timings',{});print('%.1f' % (t.get('predicted_per_second') or 0))"
}
pkill -f "[l]lama-server"; pkill -f "[s]erve/server.py"; pkill -f "[b]uild-metal/strata"; sleep 3
cd $S && nohup .venv/bin/python serve/server.py --engine strata --config $cfg --port 8080 > $OUT/$name.log 2>&1 < /dev/null &
for i in $(seq 1 300); do curl -s -m 2 http://127.0.0.1:8080/v1/models | grep -q data && break; sleep 3; done
b=$(python3 $HOME/bench-strata.py --url http://127.0.0.1:8080/v1 --lengths 1024,4096 --max-tokens 256 --repeat 3 2>&1 | awk '$1 ~ /^(1024|4096)$/ {printf "%s ", $4}')
c=$(ask "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling." $OUT/$name-code.json)
p=$(ask "Write a short story about a lighthouse keeper who finds a message in a bottle." $OUT/$name-prose.json)
c2=$(ask "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling." $OUT/$name-code2.json)
p2=$(ask "Write a short story about a lighthouse keeper who finds a message in a bottle." $OUT/$name-prose2.json)
echo "$name | bench 1K/4K: $b| code $c $c2 | prose $p $p2"
pkill -f "[s]erve/server.py"; pkill -f "[b]uild-metal/strata"
