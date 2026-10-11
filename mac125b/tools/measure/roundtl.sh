#!/bin/bash
# MTP round timeline: wall of each graph and the CPU gaps between graphs, one greedy request (thinking off).
# usage: roundtl.sh "ENV=VAL ..." "extra server args" prompt-kind(code|prose)
BIN=$HOME/llama.cpp/build/bin/llama-server
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
MTP=$HOME/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
P="Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling."
[ "$3" = "prose" ] && P="Write a short story about a lighthouse keeper who finds a message in a bottle."
pkill -f "[l]lama-server"; sleep 2
env GGML_METAL_TIMING=1 $1 nohup $BIN -m $M -ngl 99 -c 16384 --port 8080 --host 127.0.0.1 -np 1 -md $MTP --spec-type draft-mtp --reasoning off $2 > /tmp/rtl.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
N0=$(wc -l < /tmp/rtl.log | tr -d ' ')
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":400,\"temperature\":0}" > /tmp/rtl.json
tail -n +$N0 /tmp/rtl.log > /tmp/rtl.req.log
pkill -f "[l]lama-server"
python3 - <<'PY'
import re, json, collections
rows=[]
for l in open("/tmp/rtl.req.log", errors="replace"):
    m=re.search(r"nodes=(\d+) n_cb=\d+ encode=([\d.]+) gpu_span=[\d.]+ gpu_sum=([\d.]+) first_gpu=[\d.]+ wall=([\d.]+) ms t=([\d.]+)", l)
    if m: rows.append(dict(n=int(m.group(1)), enc=float(m.group(2)), gpu=float(m.group(3)), wall=float(m.group(4)), t=float(m.group(5))))
# classify: draft graphs are the most frequent small size; verify graphs are > 1000 nodes
cnt = collections.Counter(r["n"] for r in rows if r["n"] < 1000)
dn = cnt.most_common(1)[0][0]
# skip prompt processing: start at the first verify graph after the first draft graph
i0 = next(i for i,r in enumerate(rows) if r["n"] == dn)
i0 = next(i for i in range(i0, len(rows)) if rows[i]["n"] >= 1000)
rows = rows[i0:]
kind = lambda r: "verify" if r["n"] >= 1000 else ("draft" if r["n"] == dn else "catchup%d" % r["n"])
agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
gap_after = collections.defaultdict(float)
for a, b in zip(rows, rows[1:]):
    g = b["t"] - (a["t"] + a["wall"])
    gap_after[kind(a)] += g
for r in rows:
    k = kind(r); agg[k][0] += 1; agg[k][1] += r["wall"]; agg[k][2] += r["gpu"]; agg[k][3] += r["enc"]
nver = agg["verify"][0]
span = rows[-1]["t"] + rows[-1]["wall"] - rows[0]["t"]
d = json.load(open("/tmp/rtl.json")); tm = d.get("timings", {})
print("rounds %d  span %.1f ms  per round %.2f ms  tok/s %.1f  predicted %d" % (nver, span, span/nver, tm.get("predicted_per_second", 0), tm.get("predicted_n", 0)))
for k, (c, w, g, e) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print("  %-10s count %5d  per round: n %.2f  wall %.2f ms  gpu %.2f ms  encode %.2f ms  cpu-gap after %.2f ms" % (k, c, c/nver, w/nver, g/nver, e/nver, gap_after[k]/nver))
PY
grep -E "draft acceptance" /tmp/rtl.req.log | tail -1
