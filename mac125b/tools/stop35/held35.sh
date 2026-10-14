#!/bin/bash
# 18 held-out prompts (not the fixed metric's), per-prompt decode tok/s: "name|bin|ENV|args" arms, model $M
run() {
  IFS='|' read -r name bin envs args <<< "$1"
  pkill -f "[l]lama-server"; sleep 2
  env ${envs:-X_=0} nohup $bin -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 --reasoning off $args > /tmp/held35_$name.log 2>&1 < /dev/null &
  for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
  python3 - "$name" <<PY
import json, sys, urllib.request
name = sys.argv[1]
ps = json.load(open(""+__import__('os').path.expanduser('~/mtpdata/heldout.json')+""))
out = []
for p in ps:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})))
    out.append([r["timings"]["predicted_per_second"], r["usage"]["completion_tokens"], r["choices"][0]["message"]["content"]])
json.dump(out, open(__import__("os").path.expanduser("~/held35_%s.json" % name), "w"))
tot = sum(x[1] for x in out); tim = sum(x[1] / x[0] for x in out)
print("%s mean %.1f tok/s | %s" % (name, tot / tim, " ".join("%.0f" % x[0] for x in out)))
PY
}
for a in "$@"; do run "$a"; done
pkill -f "[l]lama-server"
echo HELD35DONE
