"""Benchmark a Strata (or any OpenAI-compatible) endpoint.

Reports prompt-processing and decode rates separately, the way Strata's paper
does, so numbers line up with its Table 5: time to first token is the prompt
path, the rest is decode.

Usage:  python3 bench-strata.py [--url URL] [--lengths 1024,4096,32768]
Stdlib only; run it on the box that serves the model.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

FILLER = (
    "def transform(records):\n"
    "    out = []\n"
    "    for r in records:\n"
    "        if r.get('active') and r['score'] > 0:\n"
    "            out.append({'id': r['id'], 'score': r['score'] * 1.5})\n"
    "    return sorted(out, key=lambda x: -x['score'])\n\n"
)


def build_prompt(approx_tokens: int) -> str:
    """Roughly approx_tokens of plausible code context plus a task."""
    body = FILLER * max(1, approx_tokens // 60)
    return (
        "Here is part of a codebase:\n\n```python\n"
        + body
        + "```\n\nWrite one short function that sums the scores. Code only."
    )


def run_once(url: str, model: str, prompt: str, max_tokens: int, api_key: str | None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url + "/chat/completions", json.dumps(body).encode(), headers
    )

    start = time.perf_counter()
    ttft = None
    produced = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    produced += 1
    total = time.perf_counter() - start

    prompt_tokens = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or produced
    decode_time = max(total - (ttft or 0), 1e-6)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "ttft": ttft or 0.0,
        "total": total,
        "decode_tps": completion / decode_time,
        "prompt_tps": (prompt_tokens / ttft) if ttft else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--lengths", default="1024,4096,32768")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    print(f"{'ctx':>8}  {'prompt tok':>10}  {'prompt t/s':>11}  {'decode t/s':>11}  {'TTFT s':>7}")
    for want in [int(x) for x in args.lengths.split(",") if x.strip()]:
        prompt = build_prompt(want)
        runs = []
        for _ in range(args.repeat):
            try:
                runs.append(run_once(args.url, args.model, prompt, args.max_tokens, args.api_key))
            except Exception as exc:  # keep going; a long context may not fit
                print(f"{want:>8}  error: {type(exc).__name__}: {exc}")
                break
        if not runs:
            continue
        dec = statistics.median(r["decode_tps"] for r in runs)
        pro = statistics.median(r["prompt_tps"] for r in runs)
        ttft = statistics.median(r["ttft"] for r in runs)
        ptok = runs[0]["prompt_tokens"]
        print(f"{want:>8}  {ptok:>10}  {pro:>11.1f}  {dec:>11.1f}  {ttft:>7.2f}")


if __name__ == "__main__":
    main()
