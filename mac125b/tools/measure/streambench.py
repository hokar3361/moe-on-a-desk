# engine-independent greedy decode speed: stream one chat completion and time the tokens after the first one.
# usage: streambench.py URL MODEL code|prose|file:<json list> [max_tokens]
import json, sys, time, urllib.request
CODE = "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling."
PROSE = "Write a short story about a lighthouse keeper who finds a message in a bottle."

def run(url, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t_first = t_last = None; n = 0; text = []; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    n += 1
                    text.append(piece)
    ntok = (usage or {}).get("completion_tokens") or n
    dt = (t_last - t_first) if t_first and t_last and t_last > t_first else float("nan")
    return (ntok - 1)/dt, ntok, "".join(text)

if __name__ == "__main__":
    url, model, what = sys.argv[1], sys.argv[2], sys.argv[3]
    mt = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    prompts = {"code": [CODE], "prose": [PROSE]}.get(what) or json.load(open(what[5:]))
    for p in prompts:
        tps, ntok, text = run(url, model, p, mt)
        print("%.1f tok/s  (%d tokens)  %s" % (tps, ntok, p[:50].replace("\n", " ")), flush=True)
