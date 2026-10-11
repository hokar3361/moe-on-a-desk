#!/usr/bin/env python3
# mac125b engine console: start/stop one engine at a time (start point, Sushi, -11d) and chat with it through one UI.
# Python 3.9 stdlib only. Listens on 0.0.0.0:8090; private-network clients only, plus an access key (cookie).
import difflib, hashlib, http.client, http.server, json, os, secrets, signal, socketserver, subprocess, threading, time, urllib.parse, ipaddress

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "console")
os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
KEY_FILE = os.path.join(BASE, "key")
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "w") as f:
        f.write(secrets.token_urlsafe(12))
    os.chmod(KEY_FILE, 0o600)
KEY = open(KEY_FILE).read().strip()

M   = HOME + "/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
MTP = HOME + "/models/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
LLAMA_COMMON = ["-m", M, "-md", MTP, "--spec-type", "draft-mtp", "--reasoning", "off", "-ngl", "99", "-c", "16384", "-np", "1",
                "--host", "127.0.0.1"]

ENGINES = {
    "start": {
        "label": "開始時点 (llama.cpp 781dbc5、未改造、MTP 深さ 3)",
        "cmd": [HOME + "/llama-base/build/bin/llama-server"] + LLAMA_COMMON + ["--spec-draft-n-max", "3", "--port", "8081"],
        "env": {}, "port": 8081, "health": "/health",
    },
    "sushi": {
        "label": "Sushi 1.2.1 + Sushi-2.6bpw (MLX)",
        "cmd": [HOME + "/opt/sushi/sushi-macos-arm64/sushi", "serve", "--model", HOME + "/models/sushi/Qwen3.8-Flash-Next-Sushi-2.6bpw",
                "--port", "12345", "--ctx-size", "16384"],
        "env": {}, "port": 12345, "health": "/v1/models",
    },
    "tf27b": {
        "label": "TensorFold 0.6.6 + Qwen3.8-27B MLX 4-bit + DFlash2 下書き器(別モデル、参考)",
        "cmd": [HOME + "/venv-tf/bin/tensorfold", "serve", "TensorFold/Qwen3.8-27B-MLX-4bit", "--host", "127.0.0.1", "--port", "8083",
                "--name", "m", "--no-thinking", "--no-update-check"],
        "env": {}, "port": 8083, "health": "/v1/models",
    },
    "best": {
        "label": "-11d (前の最良、タグ mac125b-best-2026-10-11d)",
        "cmd": [HOME + "/llama-11d/build/bin/llama-server"] + LLAMA_COMMON + ["--spec-draft-n-max", "7", "--port", "8082"],
        "env": {"GGML_ALLOC_FREE_DELAY": "4", "LLAMA_SPEC_MTP_MERGE": "1", "LLAMA_GRAPH_CACHE": "16", "LLAMA_SPEC_DEPTH_CTL": "1",
                "LLAMA_SPEC_DEPTH_MIN": "1", "LLAMA_SPEC_DEPTH_RESET": "1"},
        "port": 8082, "health": "/health",
    },
    "e11e": {
        "label": "-11f (このプロジェクトの最良、タグ mac125b-best-2026-10-11f、下書きの停止規則)",
        "cmd": [HOME + "/llama-best/build/bin/llama-server"] + LLAMA_COMMON + ["--spec-draft-n-max", "7", "--port", "8084"],
        "env": {"GGML_ALLOC_FREE_DELAY": "4", "LLAMA_SPEC_MTP_MERGE": "1", "LLAMA_GRAPH_CACHE": "16", "LLAMA_SPEC_STOP": "1"},
        "port": 8084, "health": "/health",
    },
}

# model facts per engine (sources: GGUF tensor types and file sizes, the Sushi pack README, the Hugging Face file listings)
META = {
    "start": {"model": "Qwen3.8-Flash-Next 125B (MoE、512 エキスパート)", "quant": "IQ3_XXS GSQ-RCO(GGUF、エキスパート平均 約 2.9 bpw、Dense 4〜6 bit・HC bf16)+ MTP Q8_0",
              "gpu": "重み 43.8 GiB + MTP 3.9 GiB", "disk": "71 GiB(n-gram 表 27.5 GB を含む)+ MTP 3.9 GiB"},
    "best":  {"model": "Qwen3.8-Flash-Next 125B (MoE、512 エキスパート)", "quant": "IQ3_XXS GSQ-RCO(開始時点と同一ファイル)+ MTP Q8_0",
              "gpu": "重み 43.8 GiB + MTP 3.9 GiB", "disk": "71 GiB(n-gram 表 27.5 GB を含む)+ MTP 3.9 GiB"},
    "e11e":  {"model": "Qwen3.8-Flash-Next 125B (MoE、512 エキスパート)", "quant": "IQ3_XXS GSQ-RCO(開始時点と同一ファイル)+ MTP Q8_0",
              "gpu": "重み 43.8 GiB + MTP 3.9 GiB", "disk": "71 GiB(n-gram 表 27.5 GB を含む)+ MTP 3.9 GiB"},
    "sushi": {"model": "Qwen3.8-Flash-Next 125B (MoE、512 エキスパート)", "quant": "Sushi-2.6bpw(エキスパート EXL3 2.6 bpw、Dense 8 bit、n-gram 表 4 bit)",
              "gpu": "重み 43.95 GiB(MTP 層を含む)", "disk": "74 GiB(n-gram 表 29.8 GB を含む)"},
    "tf27b": {"model": "Qwen3.8-27B (Dense)", "quant": "MLX 4-bit g64 + DFlash2 下書き器(読み込み時に 4 bit 化)",
              "gpu": "重み 約 15 GB + 下書き器", "disk": "15 GB + 下書き器 3.6 GB"},
}
state = {"active": None, "status": "stopped", "since": time.time(), "error": ""}
RUNS = os.path.join(BASE, "runs.jsonl")
runs_lock = threading.Lock()


def save_run(r):
    with runs_lock:
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_runs(limit=500):
    try:
        with open(RUNS, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        return rows[-limit:]
    except Exception:
        return []


def prompt_key(req):
    key = json.dumps({"m": req.get("messages", []), "max_tokens": req.get("max_tokens"), "temperature": req.get("temperature")},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def last_user(req):
    for m in reversed(req.get("messages", [])):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""

proc = {"p": None}
lock = threading.Lock()


_mem_cache = {"t": 0.0, "v": None}


def mem_snapshot():
    """system memory breakdown (GB) from vm_stat / sysctl, and the RSS of the engine processes"""
    now = time.time()
    if _mem_cache["v"] is not None and now - _mem_cache["t"] < 1.5:
        return _mem_cache["v"]
    gb = 1024.0**3
    v = {}
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        ps_ = 16384
        pages = {}
        for line in out.splitlines():
            if "page size of" in line:
                ps_ = int(line.split("page size of")[1].split()[0])
            elif ":" in line:
                k, val = line.split(":", 1)
                try:
                    pages[k.strip()] = int(val.strip().rstrip("."))
                except ValueError:
                    pass
        g = lambda k: pages.get(k, 0)*ps_/gb
        v = {"total": round(total/gb, 1), "wired": round(g("Pages wired down"), 1), "compressed": round(g("Pages occupied by compressor"), 1),
             "active": round(g("Pages active"), 1), "inactive": round(g("Pages inactive"), 1),
             "free": round(g("Pages free") + g("Pages speculative"), 1), "purgeable": round(g("Pages purgeable"), 1)}
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        for part in sw.split("  "):
            if part.strip().startswith("used"):
                x = part.split("=")[1].strip()
                v["swap"] = round(float(x[:-1])/(1024 if x.endswith("M") else 1), 2)
        rss = 0
        psout = subprocess.run(["ps", "-axo", "rss=,command="], capture_output=True, text=True).stdout
        for line in psout.splitlines():
            line = line.strip()
            if "llama-server" in line or "sushi-macos-arm64/sushi serve" in line or "tensorfold serve" in line:
                try:
                    rss += int(line.split(None, 1)[0])
                except ValueError:
                    pass
        v["engine_rss"] = round(rss*1024/gb, 1)
    except Exception as ex:
        v["error"] = str(ex)
    _mem_cache.update(t=now, v=v)
    return v


def kill_all():
    p = proc["p"]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:
            pass
        for _ in range(50):
            if p.poll() is not None:
                break
            time.sleep(0.2)
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    proc["p"] = None
    # strays: servers started outside the console (e.g. ~/run-best.sh) hold the same memory
    subprocess.call(["pkill", "-f", "[l]lama-server"])
    subprocess.call(["pkill", "-f", "[s]ushi-macos-arm64/sushi serve"])
    subprocess.call(["pkill", "-f", "[t]ensorfold serve"])
    time.sleep(2)


def healthy(name):
    e = ENGINES[name]
    try:
        c = http.client.HTTPConnection("127.0.0.1", e["port"], timeout=2)
        c.request("GET", e["health"])
        r = c.getresponse()
        body = r.read(400).decode("utf-8", "replace")
        return r.status == 200 and (ENGINES[name]["health"] == "/v1/models" or "ok" in body)
    except Exception:
        return False


def start_engine(name):
    with lock:
        state.update(active=name, status="stopping others", since=time.time(), error="")
        kill_all()
        e = ENGINES[name]
        env = dict(os.environ)
        env.update(e["env"])
        log = open(os.path.join(BASE, "logs", name + ".log"), "w")
        state.update(status="starting", since=time.time())
        proc["p"] = subprocess.Popen(e["cmd"], env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
    for _ in range(1800):
        if proc["p"] is None or state["active"] != name:
            return
        if proc["p"].poll() is not None:
            state.update(status="error", error="process exited (code %s), see log" % proc["p"].returncode)
            return
        if healthy(name):
            state.update(status="ready", since=time.time())
            return
        time.sleep(1)
    state.update(status="error", error="not ready after 1800 s")


def stop_engine():
    with lock:
        state.update(status="stopping", since=time.time())
        kill_all()
        state.update(active=None, status="stopped", since=time.time(), error="")


def log_tail(name, n=12):
    try:
        with open(os.path.join(BASE, "logs", name + ".log"), "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 6000))
            lines = f.read().decode("utf-8", "replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


PRIVATE = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        pass

    def _allowed(self):
        ip = ipaddress.ip_address(self.client_address[0])
        if not any(ip in n for n in PRIVATE):
            return False
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if q.get("key", [""])[0] == KEY:
            return True
        ck = self.headers.get("Cookie", "")
        return ("ck=" + KEY) in ck

    def _send(self, code, body, ctype="application/json", extra=None):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if not self._allowed():
            return self._send(403, "forbidden: open the URL with ?key=... from the console start message", "text/plain")
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8",
                              {"Set-Cookie": "ck=%s; Path=/; Max-Age=2592000; SameSite=Strict; HttpOnly" % KEY})
        if path == "/api/status":
            a = state["active"]
            return self._send(200, json.dumps({
                "active": a, "status": state["status"], "error": state["error"], "elapsed": round(time.time() - state["since"]),
                "engines": {k: v["label"] for k, v in ENGINES.items()}, "meta": META, "log": log_tail(a) if a else "", "mem": mem_snapshot()}))
        if path == "/api/runs":
            return self._send(200, json.dumps(load_runs(), ensure_ascii=False))
        if path.startswith("/v1/"):
            return self._proxy("GET")
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        if not self._allowed():
            return self._send(403, "forbidden", "text/plain")
        path = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(n) if n else b""
        if path == "/api/start":
            name = json.loads(body or b"{}").get("engine")
            if name not in ENGINES:
                return self._send(400, json.dumps({"error": "unknown engine"}))
            threading.Thread(target=start_engine, args=(name,), daemon=True).start()
            return self._send(200, json.dumps({"ok": True}))
        if path == "/api/runs/clear":
            with runs_lock:
                open(RUNS, "w").close()
            return self._send(200, json.dumps({"ok": True}))
        if path == "/api/stop":
            threading.Thread(target=stop_engine, daemon=True).start()
            return self._send(200, json.dumps({"ok": True}))
        if path.startswith("/v1/"):
            return self._proxy("POST", body)
        return self._send(404, "not found", "text/plain")

    def _proxy(self, method, body=b""):
        a = state["active"]
        if a is None or state["status"] != "ready":
            return self._send(503, json.dumps({"error": "no engine is ready"}))
        try:
            path = urllib.parse.urlparse(self.path).path
            capture = method == "POST" and path == "/v1/chat/completions"
            req = {}
            if capture:
                try:
                    req = json.loads(body or b"{}")
                except Exception:
                    capture = False
            t0 = time.time()
            c = http.client.HTTPConnection("127.0.0.1", ENGINES[a]["port"], timeout=600)
            c.request(method, path, body=body if method == "POST" else None, headers={"Content-Type": "application/json"})
            r = c.getresponse()
            self.send_response(r.status)
            self.send_header("Content-Type", r.getheader("Content-Type", "application/json"))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            buf = b""; raw = []; t_first = 0.0; t_last = 0.0; chunks = 0; text = []; usage = None; timings = None
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
                if not chunk:
                    break
                now = time.time()
                if capture:
                    if req.get("stream"):
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line.startswith(b"data:"):
                                continue
                            pl = line[5:].strip()
                            if pl == b"[DONE]":
                                continue
                            try:
                                j = json.loads(pl)
                            except Exception:
                                continue
                            usage = j.get("usage") or usage
                            timings = j.get("timings") or timings
                            ch = j.get("choices") or []
                            d = (ch[0].get("delta") or {}).get("content") if ch else None
                            if d:
                                if not t_first:
                                    t_first = now
                                t_last = now
                                chunks += 1
                                text.append(d)
                    else:
                        raw.append(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            t_end = time.time()
            if capture and r.status == 200:
                if not req.get("stream"):
                    try:
                        j = json.loads(b"".join(raw))
                        usage = j.get("usage")
                        timings = j.get("timings")
                        text = [j["choices"][0]["message"].get("content") or ""]
                    except Exception:
                        pass
                    t_first = t_last = t_end
                out = "".join(text)
                n_out = (usage or {}).get("completion_tokens") or chunks
                run = {"ts": t0, "engine": a, "key": prompt_key(req), "prompt": last_user(req), "n_msgs": len(req.get("messages", [])),
                       "max_tokens": req.get("max_tokens"), "temperature": req.get("temperature"),
                       "cache_prompt": req.get("cache_prompt", True), "prompt_tokens": (usage or {}).get("prompt_tokens"),
                       "out_tokens": n_out, "ttft_ms": round((t_first - t0)*1000, 1) if t_first else None,
                       "proxy_decode_tps": round((n_out - 1)/(t_last - t_first), 2) if (t_last > t_first and n_out and n_out > 1) else None,
                       "total_ms": round((t_end - t0)*1000, 1), "text": out, "text_sha": hashlib.sha1(out.encode("utf-8")).hexdigest()[:12]}
                if timings:
                    run.update(eng_prefill_tokens=timings.get("prompt_n"), eng_prefill_ms=timings.get("prompt_ms"),
                               eng_prefill_tps=timings.get("prompt_per_second"), eng_decode_tokens=timings.get("predicted_n"),
                               eng_decode_ms=timings.get("predicted_ms"), eng_decode_tps=timings.get("predicted_per_second"),
                               cache_n=timings.get("cache_n"), draft_n=timings.get("draft_n"), draft_accepted=timings.get("draft_n_accepted"))
                _mem_cache["t"] = 0.0
                m = mem_snapshot()
                run.update(mem_wired=m.get("wired"), mem_engine_rss=m.get("engine_rss"), mem_swap=m.get("swap"),
                           mem_compressed=m.get("compressed"), mem_free=m.get("free"))
                save_run(run)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as ex:
            try:
                self._send(502, json.dumps({"error": str(ex)}))
            except Exception:
                pass


class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>mac125b エンジン比較</title>
<style>
:root{--bg:#f4f5f7;--fg:#1d2330;--mut:#667085;--card:#fff;--line:#e2e5ea;--acc:#2f6fed;--ok:#1a9b55;--warn:#c98a0b;--err:#c23934;
 --c-start:#8a94a6;--c-sushi:#d08a2b;--c-best:#2f6fed;--c-tf27b:#16a39a;--c-e11e:#c0399b}
@media (prefers-color-scheme: dark){:root{--bg:#121519;--fg:#e6e8ec;--mut:#98a2b3;--card:#1b1f26;--line:#2b313b;--acc:#5b8dff;--ok:#3ccf7e;--warn:#e3b341;--err:#ff6b64;
 --c-start:#9aa4b5;--c-sushi:#e3a04a;--c-best:#5b8dff;--c-tf27b:#2cc4b9;--c-e11e:#e05cbf}}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif;background:var(--bg);color:var(--fg);
 display:grid;grid-template-rows:auto 1fr;overflow:hidden}
header{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);background:var(--card)}
header h1{font-size:15px;margin:0;white-space:nowrap}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:7px;padding:5px 10px;cursor:pointer;white-space:nowrap}
button.eng.active{border-color:var(--acc);box-shadow:0 0 0 2px color-mix(in srgb,var(--acc) 30%,transparent);font-weight:600}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button:disabled{opacity:.45;cursor:default}
.lamp{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--mut);margin-right:6px;vertical-align:middle}
.lamp.ready{background:var(--ok)}.lamp.starting{background:var(--warn)}.lamp.error{background:var(--err)}
#status{color:var(--mut);font-size:13px;flex:0 1 auto;max-width:42%;min-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
main{display:grid;grid-template-columns:minmax(300px,0.72fr) minmax(600px,1.8fr);gap:10px;padding:10px;min-height:0}
#chat,#chat .msg,.composer textarea{font-size:12px;line-height:1.4}
#chat .stat,#chat .who{font-size:10.5px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;display:flex;flex-direction:column;min-height:0;overflow:hidden}
.ph{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;padding:8px 10px;border-bottom:1px solid var(--line);font-size:12px;color:var(--mut)}
.ph b{color:var(--fg);font-size:13px}
#chat{flex:1;min-height:0;overflow-y:auto;padding:8px 12px;scroll-behavior:auto}
.msg{margin:8px 0;white-space:pre-wrap;word-wrap:break-word}
.msg .who{font-size:11px;color:var(--mut);margin-bottom:2px}
.msg.user .body{background:color-mix(in srgb,var(--acc) 10%,transparent);padding:6px 9px;border-radius:7px}
.stat{font-size:11px;color:var(--mut);margin-top:3px}
.composer{border-top:1px solid var(--line);padding:8px 10px;display:flex;gap:8px;align-items:stretch}
textarea{flex:1;height:64px;resize:none;font:inherit;border:1px solid var(--line);border-radius:7px;padding:6px 8px;background:var(--bg);color:var(--fg)}
.composer .btns{display:flex;flex-direction:column;gap:6px}
input[type=number]{width:64px;font:inherit;font-size:12px;border:1px solid var(--line);border-radius:6px;padding:2px 4px;background:var(--bg);color:var(--fg)}
select{font:inherit;font-size:12px;border:1px solid var(--line);border-radius:6px;padding:3px;background:var(--bg);color:var(--fg);max-width:100%}
#cmpbody{flex:1;min-height:0;overflow-y:auto;padding:8px 10px}
table{border-collapse:collapse;width:100%;font-size:12px}
table{border:1px solid var(--line)}
th,td{border:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:middle}
th{background:color-mix(in srgb,var(--line) 35%,transparent)}
th{color:var(--mut);font-weight:600}
td.n,th.n{text-align:right;white-space:nowrap}
.mx td.cell{width:21%}
.cellv{display:flex;align-items:center;gap:6px}
.track{flex:1;height:12px;background:color-mix(in srgb,var(--line) 60%,transparent);border-radius:4px;overflow:hidden}
.bar{height:100%;border-radius:4px}
.b-start{background:var(--c-start)}.b-sushi{background:var(--c-sushi)}.b-best{background:var(--c-best)}.b-tf27b{background:var(--c-tf27b)}.b-e11e{background:var(--c-e11e)}
.v{min-width:58px;text-align:right;font-variant-numeric:tabular-nums}
.best-v{font-weight:700}
td.best-cell{background:#dff1ff}
@media (prefers-color-scheme: dark){td.best-cell{background:#16344d}}
.q-ok{color:var(--ok);font-weight:600}.q-diff{color:var(--warn);font-weight:600}
.sec{font-weight:600;margin:10px 0 4px;font-size:12px}
.note{font-size:11px;color:var(--mut);margin-top:4px}
details{margin-top:6px}summary{cursor:pointer;font-size:12px;color:var(--mut)}
details pre{white-space:pre-wrap;max-height:220px;overflow:auto;font-size:12px;margin:4px 0;background:var(--bg);padding:6px;border-radius:6px}
.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px;vertical-align:middle}

.m-wired{background:#7c5cd6}.m-comp{background:#d0543f}.m-active{background:#2f9e6e}.m-inactive{background:#9cc5b3}
.leg{display:inline-block;width:8px;height:8px;border-radius:2px;margin:0 3px 0 6px;vertical-align:middle}
@media (max-width:900px){body{overflow:auto}main{grid-template-columns:1fr}.panel{min-height:70vh}}
</style></head><body>
<header>
  <h1>mac125b エンジン比較</h1>
  <div id="engines" style="display:flex;gap:6px"></div>
  <button id="stop">停止</button>
  <span id="status"><span class="lamp"></span>状態を取得中…</span>
  <div id="mem" title="" style="display:flex;align-items:center;gap:8px;min-width:360px;flex:1;font-size:12px;color:var(--mut)">
    <span>RAM</span><div id="membar" style="flex:1;display:flex;height:12px;border-radius:4px;overflow:hidden;background:color-mix(in srgb,var(--line) 60%,transparent)"></div>
    <span id="memtxt" style="white-space:nowrap"></span></div>
  <details id="logbox" style="margin:0"><summary>ログ</summary><pre id="log" style="position:absolute;right:14px;z-index:5;width:min(720px,90vw);max-height:300px;overflow:auto;background:var(--card);border:1px solid var(--line);padding:8px;font-size:11px"></pre></details>
</header>
<main>
  <section class="panel">
    <div class="ph"><b>チャット</b>
      <span>プリセット</span>
      <button class="preset" data-p="Write a complete Python module that implements an LRU cache class with get, put, delete and resize methods, thread safety with a lock, type hints, docstrings, and a set of unittest test cases covering every method.">コード</button>
      <button class="preset" data-p="Generate a JSON array of 15 fictional employees. Each object must have id, name, department, title, start_date (YYYY-MM-DD), salary, and a skills array of 3 strings. Output only the JSON.">JSON</button>
      <button class="preset" data-p="Write a short story about a lighthouse keeper who finds a message in a bottle.">文章</button>
      <button class="preset" data-p="地方の小さな駅を舞台に、最終列車を待つ二人の会話を中心とした短編小説を書いてください。">日本語</button>
      <button class="preset" data-p="次の条件をすべて満たす3桁の整数をすべて求め、考え方も示してください。各桁の和が15、百の位は一の位より2大きい、全体は7の倍数。">推論</button>
      <span>max <input type="number" id="maxtok" value="600" min="16" max="4096"></span>
      <span>temp <input type="number" id="temp" value="0" min="0" max="2" step="0.1"></span>
      <label><input type="checkbox" id="keep"> 会話を続ける</label>
      <label title="毎回プロンプトを処理し直し、プレフィル速度を測れるようにします"><input type="checkbox" id="nocache" checked> キャッシュなし</label>
    </div>
    <div id="chat"></div>
    <div class="composer">
      <textarea id="input" placeholder="メッセージ(Ctrl/⌘+Enter で送信)"></textarea>
      <div class="btns"><button class="primary" id="send">送信</button><button id="clear">クリア</button></div>
    </div>
  </section>
  <section class="panel" id="cmp">
    <div class="ph"><b>比較</b>
      <span>表示</span><select id="view"><option value="prompt">プロンプト別</option><option value="preset">プリセット別</option></select>
      <select id="pmetric" style="display:none"><option value="decode">デコード tok/s</option><option value="prefill">プレフィル tok/s</option><option value="ttft">最初のトークン ms</option><option value="total">応答全体 ms</option><option value="out">出力トークン</option><option value="acc">下書き受理</option><option value="wired">固定メモリ GB</option></select>
      <span>プロンプト</span><select id="grp" style="flex:1;min-width:220px"></select>
      <span>基準</span><select id="ref"><option value="start">開始時点</option><option value="best">-11d</option><option value="e11e">-11f</option><option value="sushi">Sushi</option><option value="tf27b">TF 27B</option></select>
      <button id="reload">更新</button><button id="clearruns">全消去</button>
    </div>
    <div id="cmpbody"><div class="note">まだ記録がありません。エンジンを起動して送信すると、ここに並びます(このコンソールを通った推論を自動収集)。</div></div>
  </section>
</main>
<script>
let st={active:null,status:"stopped"}, messages=[], busy=false;
const $=s=>document.querySelector(s);
const short={start:"開始時点",sushi:"Sushi",best:"-11d",e11e:"-11f",tf27b:"TF 27B"};
const ORDER=["start","sushi","best","e11e","tf27b"];
const chat=$("#chat");
function toBottom(){chat.scrollTop=chat.scrollHeight}
async function poll(){try{const r=await fetch("/api/status");st=await r.json();}catch(e){}
 const lamp=st.status==="ready"?"ready":st.status==="error"?"error":(st.status==="stopped"?"":"starting");
 const name=st.active?st.engines[st.active]:"なし";
 const msg={ready:"準備完了",starting:"起動中",stopped:"停止中",error:"エラー","stopping others":"他のエンジンを停止中",stopping:"停止中"}[st.status]||st.status;
 $("#status").innerHTML='<span class="lamp '+lamp+'"></span>'+(st.active?esc(name)+" — ":"")+msg+(st.status==="starting"?"("+st.elapsed+" 秒)":"")+(st.error?" — "+esc(st.error):"");
 $("#status").title=st.active?name:"";
 $("#log").textContent=st.log||"";
 const m=st.mem||{}; if(m.total){const parts=[["wired","m-wired","固定(GPU の重みなど)"],["compressed","m-comp","圧縮"],["active","m-active","アクティブ"],["inactive","m-inactive","非アクティブ(ファイルキャッシュ含む)"]];
  $("#membar").innerHTML=parts.map(([k,c,l])=>'<div class="'+c+'" title="'+l+' '+m[k]+' GB" style="width:'+(100*m[k]/m.total)+'%"></div>').join("");
  const used=m.wired+m.compressed+m.active;
  $("#memtxt").innerHTML='<span class="leg m-wired"></span>固定 '+m.wired+'<span class="leg m-comp"></span>圧縮 '+m.compressed+'<span class="leg m-active"></span>アクティブ '+m.active+'<span class="leg m-inactive"></span>キャッシュ '+m.inactive+' / 空き '+m.free+' / 計 '+m.total+' GB ・ スワップ '+(m.swap??0)+' GB ・ エンジン RSS '+m.engine_rss+' GB';
  $("#mem").title="固定+圧縮+アクティブ = "+used.toFixed(1)+" GB / "+m.total+" GB";}
 const box=$("#engines"); if(!box.children.length&&st.engines){for(const k of ORDER){if(!st.engines[k])continue;const b=document.createElement("button");b.className="eng";b.dataset.k=k;
  b.innerHTML='<span class="dot b-'+k+'"></span>'+short[k];b.title=st.engines[k]+" を起動";b.onclick=()=>start(k);box.appendChild(b)}}
 for(const b of box.children){b.classList.toggle("active",b.dataset.k===st.active)}
 $("#send").disabled=busy||st.status!=="ready";
}
async function start(k){if(busy)return;await fetch("/api/start",{method:"POST",body:JSON.stringify({engine:k})});messages=[];chat.innerHTML="";poll()}
$("#stop").onclick=async()=>{await fetch("/api/stop",{method:"POST"});poll()};
for(const b of document.querySelectorAll(".preset")) b.onclick=()=>{$("#input").value=b.dataset.p;$("#input").focus()};
$("#clear").onclick=()=>{messages=[];chat.innerHTML=""};
$("#input").addEventListener("keydown",e=>{if(e.key==="Enter"&&(e.metaKey||e.ctrlKey))send()});
$("#send").onclick=send;
function addMsg(who,text,cls){const d=document.createElement("div");d.className="msg "+cls;d.innerHTML='<div class="who"></div><div class="body"></div><div class="stat"></div>';
 d.querySelector(".who").textContent=who;d.querySelector(".body").textContent=text;chat.appendChild(d);toBottom();return d}
async function send(){const text=$("#input").value.trim(); if(!text||busy||st.status!=="ready")return;
 busy=true;poll(); const eng=st.active; if(!$("#keep").checked){messages=[];chat.innerHTML=""}
 messages.push({role:"user",content:text}); addMsg("あなた",text,"user"); $("#input").value="";
 const d=addMsg(short[eng],"","bot"); const body=d.querySelector(".body"), stat=d.querySelector(".stat");
 const t0=performance.now(); let tf=0, tl=0, chunks=0, out="", usage=null, etps=null;
 try{const r=await fetch("/v1/chat/completions",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:"m",messages,stream:true,
     max_tokens:+$("#maxtok").value,temperature:+$("#temp").value,cache_prompt:!$("#nocache").checked,stream_options:{include_usage:true}})});
  const rd=r.body.getReader(), dec=new TextDecoder(); let buf="";
  while(true){const {value,done}=await rd.read(); if(done)break; buf+=dec.decode(value,{stream:true});
   let i; while((i=buf.indexOf("\n"))>=0){const line=buf.slice(0,i).trim(); buf=buf.slice(i+1);
    if(!line.startsWith("data:"))continue; const p=line.slice(5).trim(); if(p==="[DONE]")continue;
    let j; try{j=JSON.parse(p)}catch(e){continue}
    if(j.usage)usage=j.usage; if(j.timings&&j.timings.predicted_per_second)etps=j.timings.predicted_per_second;
    const c=j.choices&&j.choices[0]&&j.choices[0].delta&&j.choices[0].delta.content;
    if(c){const now=performance.now(); if(!tf)tf=now; tl=now; chunks++; out+=c; body.textContent=out; toBottom()}}}
 }catch(e){body.textContent+="\n[エラー] "+e}
 const n=usage&&usage.completion_tokens?usage.completion_tokens:chunks;
 const tps=(tl>tf&&n>1)?(n-1)/((tl-tf)/1000):0, ttft=tf?(tf-t0)/1000:0;
 stat.textContent=n+" トークン / TTFT "+ttft.toFixed(2)+" 秒 / 画面計測 "+tps.toFixed(1)+" tok/s"+(etps?" / エンジン報告 "+etps.toFixed(1)+" tok/s":"");
 toBottom(); messages.push({role:"assistant",content:out});
 busy=false; poll(); setTimeout(loadRuns,300)}

let RUNS=[];
async function loadRuns(){try{RUNS=await (await fetch("/api/runs")).json()}catch(e){RUNS=[]}
 const groups={}; for(const r of RUNS){(groups[r.key]=groups[r.key]||{key:r.key,prompt:r.prompt,last:0,n:0,eng:new Set()}); const g=groups[r.key]; g.last=Math.max(g.last,r.ts); g.n++; g.eng.add(r.engine)}
 const gl=Object.values(groups).sort((a,b)=>b.last-a.last); const sel=$("#grp"), cur=sel.value; sel.innerHTML="";
 for(const g of gl){const o=document.createElement("option");o.value=g.key;o.textContent="["+ORDER.filter(e=>g.eng.has(e)).map(e=>short[e]).join("・")+"] "+g.prompt.slice(0,70)+"("+g.n+" 回)";sel.appendChild(o)}
 if(cur&&groups[cur])sel.value=cur; render()}
function prefix(a,b){let i=0;const n=Math.min(a.length,b.length);while(i<n&&a[i]===b[i])i++;return i}
function sim(a,b){const A=a.split(/\s+/).filter(x=>x), B=b.split(/\s+/).filter(x=>x); if(!A.length&&!B.length)return 1;
 const n=Math.min(A.length,1500), m=Math.min(B.length,1500); let prev=new Array(m+1).fill(0);
 for(let i=1;i<=n;i++){const cur=new Array(m+1).fill(0);for(let j=1;j<=m;j++)cur[j]=A[i-1]===B[j-1]?prev[j-1]+1:Math.max(prev[j],cur[j-1]);prev=cur}
 return 2*prev[m]/(n+m)}
function val(r,k){if(!r)return null;if(k==="prefill")return r.eng_prefill_tps??(r.prompt_tokens&&r.ttft_ms?r.prompt_tokens/(r.ttft_ms/1000):null);
 if(k==="decode")return r.eng_decode_tps??r.proxy_decode_tps; if(k==="out")return r.out_tokens; if(k==="ttft")return r.ttft_ms;
 if(k==="total")return r.total_ms; if(k==="acc")return r.draft_n?100*r.draft_accepted/r.draft_n:null;
 if(k==="wired")return r.mem_wired??null; if(k==="rss")return r.mem_engine_rss??null; if(k==="swap")return r.mem_swap??null; return null}
function fmt(k,v){if(v==null||!isFinite(v))return "-"; if(k==="wired"||k==="rss"||k==="swap")return v.toFixed(1); if(k==="out"||k==="ttft"||k==="total")return Math.round(v).toLocaleString(); if(k==="acc")return v.toFixed(0)+"%"; return v.toFixed(1)}
function renderCmp(){const key=$("#grp").value, ref=$("#ref").value, box=$("#cmpbody"); const rs=RUNS.filter(r=>r.key===key);
 if(!rs.length){box.innerHTML='<div class="note">まだ記録がありません。エンジンを起動して送信すると、ここに並びます。</div>';return}
 const latest={}; for(const r of rs)latest[r.engine]=r; const order=ORDER;
 const metrics=[["prefill","プレフィル tok/s",1],["decode","デコード tok/s",1],["ttft","最初のトークン ms",-1],["total","応答全体 ms",-1],["out","出力トークン",0],["acc","下書き受理",0],["wired","固定メモリ GB(応答終了時)",-1],["rss","エンジン RSS GB",-1],["swap","スワップ GB",-1]];
 const MT=st.meta||{};
 let h='<table class="mx"><tr><th>指標</th>'+order.map(e=>'<th><span class="dot b-'+e+'"></span>'+short[e]+(latest[e]?'':' <span class="note">(未実行)</span>')+
   (MT[e]?'<div class="note" style="font-weight:400">'+esc(MT[e].model)+'<br>'+esc(MT[e].quant)+'<br>GPU: '+esc(MT[e].gpu)+'</div>':'')+'</th>').join("")+'</tr>';
 for(const [k,label,dir] of metrics){const vs=order.map(e=>val(latest[e],k)); const ok=vs.filter(v=>v!=null&&isFinite(v)); const mx=Math.max(...ok,1e-9);
  const bestv=dir===1?Math.max(...ok):dir===-1?Math.min(...ok):null;
  h+='<tr><td>'+label+(k==="prefill"||k==="decode"?'':'')+'</td>';
  order.forEach((e,i)=>{const v=vs[i]; const proxy=(k==="prefill"&&latest[e]&&latest[e].eng_prefill_tps==null)||(k==="decode"&&latest[e]&&latest[e].eng_decode_tps==null);
   const isb=bestv!=null&&v===bestv&&ok.length>1;
   h+='<td class="cell'+(isb?' best-cell':'')+'"><div class="cellv"><div class="track"><div class="bar b-'+e+'" style="width:'+(v==null?0:Math.max(1,100*v/mx))+'%"></div></div><span class="v'+(isb?' best-v':'')+'">'+fmt(k,v)+(proxy?'*':'')+'</span></div></td>'});
  h+='</tr>'}
 h+='</table><div class="note">太字・水色=最良。* は中継側で計測(エンジンが値を報告しない場合)。</div>';
 if(latest[ref]){const R=latest[ref];
  h+='<div class="sec">基準('+short[ref]+')との比較</div><table><tr><th>エンジン</th><th>モデル</th><th>量子化</th><th>モデルサイズ(GPU 常駐 / ディスク)</th><th class="n">実行時メモリ 固定 / RSS</th><th class="n">プレフィル</th><th class="n">デコード</th><th class="n">応答全体</th><th>出力</th><th class="n">最初の相違</th><th class="n">類似度</th></tr>';
  const present=order.filter(e=>latest[e]);
  const colBest=(f,dirr)=>{const v=present.map(e=>f(latest[e])).filter(x=>x!=null&&isFinite(x));return v.length>1?(dirr>0?Math.max(...v):Math.min(...v)):null};
  const bPf=colBest(r=>val(r,"prefill"),1), bDc=colBest(r=>val(r,"decode"),1), bTt=colBest(r=>val(r,"total"),-1), bMem=colBest(r=>r.mem_wired,-1);
  const bc=(v,b)=>(b!=null&&v===b)?' best-cell':'';
  for(const e of order){const r=latest[e]; if(!r)continue; const pf=val(r,"prefill")/val(R,"prefill"), dc=val(r,"decode")/val(R,"decode"), tt=val(R,"total")/val(r,"total");
   const same=r.text===R.text, pos=prefix(r.text,R.text), sm=sim(r.text,R.text);
   const mt=MT[e]||{};
   h+='<tr><td><span class="dot b-'+e+'"></span>'+short[e]+'</td><td>'+esc(mt.model||'-')+'</td><td>'+esc(mt.quant||'-')+'</td><td>'+esc((mt.gpu||'-')+' / '+(mt.disk||'-'))+'</td><td class="n'+bc(r.mem_wired,bMem)+'">'+(r.mem_wired!=null?r.mem_wired.toFixed(1)+' / '+(r.mem_engine_rss??0).toFixed(1)+' GB':'-')+'</td><td class="n'+bc(val(r,"prefill"),bPf)+'">'+(isFinite(pf)?pf.toFixed(2)+'x':'-')+'</td><td class="n'+bc(val(r,"decode"),bDc)+'">'+(isFinite(dc)?dc.toFixed(2)+'x':'-')+'</td><td class="n'+bc(val(r,"total"),bTt)+'">'+(isFinite(tt)?tt.toFixed(2)+'x':'-')+'</td><td>'+(e===ref?'基準':(e==="tf27b"||ref==="tf27b")?'<span class="note">別モデル</span>':same?'<span class="q-ok">完全一致</span>':'<span class="q-diff">相違あり</span>')+'</td><td class="n">'+(same?'-':pos+' 文字目')+'</td><td class="n">'+(100*sm).toFixed(1)+'%</td></tr>'}
  h+='</table><div class="note">倍率は基準に対する速さ(応答全体は時間の逆数)。実行時メモリの「固定」は Mac 全体の固定メモリ(GPU に載せた重み・KV など、OS 分約 1〜2 GB を含む)、RSS はエンジンのプロセス(MLX 系は GPU バッファを含まないため小さく出る)。開始時点と -11d は同じ重みなので greedy では完全一致が期待値。Sushi は量子化が違うため相違は当然で、類似度は参考。TF 27B は別のモデル(Qwen3.8-27B)なので速度・出力とも参考値。</div>'}
 else h+='<div class="note">基準('+short[ref]+')の記録がまだありません。</div>';
 h+='<details><summary>出力本文を見比べる</summary><div style="display:grid;grid-template-columns:repeat('+Math.max(1,order.filter(e=>latest[e]).length)+',1fr);gap:8px">'+
   order.filter(e=>latest[e]).map(e=>'<div><div class="note"><span class="dot b-'+e+'"></span>'+short[e]+'('+latest[e].out_tokens+' トークン)</div><pre>'+esc(latest[e].text)+'</pre></div>').join("")+'</div></details>';
 h+='<details><summary>このプロンプトの全実行('+rs.length+' 回)</summary><table><tr><th>時刻</th><th>エンジン</th><th class="n">入力</th><th class="n">プレフィル</th><th class="n">TTFT</th><th class="n">出力</th><th class="n">デコード</th><th class="n">受理</th></tr>'+
   rs.slice().reverse().map(r=>'<tr><td>'+new Date(r.ts*1000).toLocaleTimeString()+'</td><td>'+short[r.engine]+'</td><td class="n">'+(r.prompt_tokens??'-')+'</td><td class="n">'+fmt("prefill",val(r,"prefill"))+'</td><td class="n">'+fmt("ttft",r.ttft_ms)+'</td><td class="n">'+r.out_tokens+'</td><td class="n">'+fmt("decode",val(r,"decode"))+'</td><td class="n">'+fmt("acc",val(r,"acc"))+'</td></tr>').join("")+'</table></details>';
 h+='<div class="note">プロンプト:'+esc(rs[rs.length-1].prompt.slice(0,240))+'</div>';
 box.innerHTML=h}
function esc(t){return (t||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
const PRESETS=[...document.querySelectorAll(".preset")].map(b=>({name:b.textContent.trim(),text:b.dataset.p}));
function presetOf(r){const t=(r.prompt||"").trim();for(const p of PRESETS)if(p.text.trim()===t)return p.name;return "自由入力"}
function median(a){const v=a.filter(x=>x!=null&&isFinite(x)).sort((x,y)=>x-y);if(!v.length)return null;const m=v.length>>1;return v.length%2?v[m]:(v[m-1]+v[m])/2}
function renderPreset(){const box=$("#cmpbody"), ref=$("#ref").value, k=$("#pmetric").value; const MT=st.meta||{};
 const label={decode:"デコード tok/s",prefill:"プレフィル tok/s",ttft:"最初のトークン ms",total:"応答全体 ms",out:"出力トークン",acc:"下書き受理 %",wired:"固定メモリ GB"}[k];
 const dir={decode:1,prefill:1,ttft:-1,total:-1,out:0,acc:0,wired:-1}[k];
 const names=[...PRESETS.map(p=>p.name),"自由入力"]; const order=ORDER;
 const agg={}; for(const r of RUNS){const pn=presetOf(r);(agg[pn]=agg[pn]||{});(agg[pn][r.engine]=agg[pn][r.engine]||[]).push(r)}
 let h='<table class="mx"><tr><th>プリセット / '+label+'</th>'+order.map(e=>'<th><span class="dot b-'+e+'"></span>'+short[e]+(MT[e]?'<div class="note" style="font-weight:400">'+esc(MT[e].quant)+'</div>':'')+'</th>').join("")+'</tr>';
 let any=false;
 for(const pn of names){const row=agg[pn]; if(!row)continue; any=true;
  const vals=order.map(e=>row[e]?median(row[e].map(r=>val(r,k))):null); const ok=vals.filter(v=>v!=null&&isFinite(v)); const mx=Math.max(...ok,1e-9);
  const bestv=dir===1?Math.max(...ok):dir===-1?Math.min(...ok):null; const rv=row[ref]?median(row[ref].map(r=>val(r,k))):null;
  h+='<tr><td>'+esc(pn)+'</td>';
  order.forEach((e,i)=>{const v=vals[i]; const n=row[e]?row[e].length:0;
   const ratio=(v!=null&&rv&&e!==ref&&(k==="decode"||k==="prefill"))?' <span class="note">'+(v/rv).toFixed(2)+'x</span>':(v!=null&&rv&&e!==ref&&(k==="ttft"||k==="total"))?' <span class="note">'+(rv/v).toFixed(2)+'x</span>':'';
   const isb=bestv!=null&&v===bestv&&ok.length>1;
   h+='<td class="cell'+(isb?' best-cell':'')+'"><div class="cellv"><div class="track"><div class="bar b-'+e+'" style="width:'+(v==null?0:Math.max(1,100*v/mx))+'%"></div></div><span class="v'+(isb?' best-v':'')+'">'+fmt(k==="acc"?"acc":k,v)+'</span></div><div class="note">'+(n?n+' 回'+ratio:'未実行')+'</div></td>'});
  h+='</tr>'}
 h+='</table><div class="note">各セルはそのプリセットでの中央値と回数。倍率は基準('+short[ref]+')に対する速さ(時間の指標は逆数)。太字・水色=最良。TF 27B は別モデル(参考)。プリセットの判定は送信文面の一致で行い、文面を変えたものは「自由入力」。</div>';
 if(!any)h='<div class="note">まだ記録がありません。</div>';
 box.innerHTML=h}
function render(){const v=$("#view").value; $("#grp").style.display=v==="prompt"?"":"none"; $("#pmetric").style.display=v==="preset"?"":"none";
 document.querySelectorAll("#cmp .ph > span").forEach(sp=>{if(sp.textContent==="プロンプト")sp.style.display=v==="prompt"?"":"none"});
 if(v==="preset")renderPreset();else renderCmp()}
$("#grp").onchange=render; $("#ref").onchange=render; $("#view").onchange=render; $("#pmetric").onchange=render; $("#reload").onclick=loadRuns;
$("#clearruns").onclick=async()=>{if(confirm("記録をすべて消しますか?")){await fetch("/api/runs/clear",{method:"POST"});loadRuns()}};
loadRuns(); poll(); setInterval(poll,2000);
</script></body></html>
"""

if __name__ == "__main__":
    ips = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True).stdout.strip()
    print("console: http://%s:8090/?key=%s" % (ips or "127.0.0.1", KEY), flush=True)
    # adopt an engine that is already running (e.g. after a console restart)
    for name in ENGINES:
        if healthy(name):
            state.update(active=name, status="ready", since=time.time())
            break
    S(("0.0.0.0", 8090), H).serve_forever()
