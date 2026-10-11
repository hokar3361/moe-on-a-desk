#!/bin/bash
# Reference measurement only: Sushi (MLX) on the same Mac, same prompts and bench script as ours.
S=$HOME/opt/sushi/sushi-macos-arm64/sushi
M=$HOME/models/sushi/Qwen3.8-Flash-Next-Sushi-2.6bpw
pkill -f "[l]lama-server"; pkill -f "[s]ushi-macos-arm64/sushi"; sleep 2
nohup $S serve --model $M --port 12345 --ctx-size 16384 > /tmp/sushi.log 2>&1 < /dev/null &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:12345/v1/models | grep -q . && break; sleep 3; done
curl -s http://127.0.0.1:12345/v1/models | head -c 400; echo
python3 $HOME/bench-strata.py --url http://127.0.0.1:12345/v1 --lengths 1024,4096 --max-tokens 256 --repeat 3 2>&1 | tail -4
for p in "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling." "Write a short story about a lighthouse keeper who finds a message in a bottle."; do
  t0=$(python3 -c "import time;print(time.time())")
  curl -s http://127.0.0.1:12345/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":400,\"temperature\":0}" > /tmp/sushi-ans.json
  python3 -c "import json;d=json.load(open('/tmp/sushi-ans.json'));print('usage',d.get('usage'));print('timings',d.get('timings'))"
done
grep -i -E "tok/s|mtp|accept|draft|decode" /tmp/sushi.log | tail -8
