"""Fixed metric on native Windows (same calls as wsl/tools/abw.sh and mac125b/tools/measure/mtpab.sh): bench 1K/4K, code / prose 400 greedy,
decode tok/s from the server timings, plus the resources used meanwhile: system CPU % (all logical CPUs = 100 %), the server's RSS
(working set) and the GPU's dedicated memory in use.

usage: python abw.py NAME -- <server command line ...>        (the server must listen on 127.0.0.1:$ABW_PORT, default 8091)
       env: ABW_ENV="A=1 B=2" extra environment for the server, ABW_OUT=dir for logs and answers, ABW_HEALTH=/health
"""
import json, os, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ.get('BENCH', os.path.join(HERE, '..', '..', 'mac125b', 'tools', 'measure', 'bench-strata.py'))
OUT = os.environ.get('ABW_OUT', os.path.join(HERE, 'runs'))
URL = 'http://127.0.0.1:' + os.environ.get('ABW_PORT', '8091')
CODE = 'Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling.'
PROSE = 'Write a short story about a lighthouse keeper who finds a message in a bottle.'

import psutil


def get(path, timeout=2):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return r.read()


def ask(prompt, path):
    body = json.dumps({'model': 'm', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 400, 'temperature': 0}).encode()
    req = urllib.request.Request(URL + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    json.dump(d, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    t = d.get('timings', {})
    return t.get('predicted_per_second') or 0.0


def gpu_stat():
    # (VRAM used MiB, GPU utilization %)
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        a, b = o.stdout.strip().splitlines()[0].split(',')
        return int(a), int(b)
    except Exception:
        return 0, 0


def vram_mib():
    return gpu_stat()[0]


def main():
    name = sys.argv[1]
    cmd = sys.argv[sys.argv.index('--') + 1:]
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ)
    for kv in os.environ.get('ABW_ENV', '').split():
        k, v = kv.split('=', 1)
        env[k] = v
    log = open(os.path.join(OUT, name + '.log'), 'w', encoding='utf-8')
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    health = os.environ.get('ABW_HEALTH', '/health')
    t0 = time.time()
    ok = False
    while time.time() - t0 < 1800:
        if proc.poll() is not None:
            break
        try:
            if b'ok' in get(health) or b'data' in get(health):
                ok = True
                break
        except Exception:
            pass
        time.sleep(3)
    if not ok:
        print(f'{name} | failed to start (see {name}.log)')
        proc.kill()
        return
    load_s = time.time() - t0
    samples = []
    stop = threading.Event()
    p = psutil.Process(proc.pid)

    def sampler():
        psutil.cpu_percent(None)
        while not stop.is_set():
            time.sleep(1)
            try:
                rss = sum(c.memory_info().rss for c in [p] + p.children(recursive=True))
            except Exception:
                rss = 0
            vm, gu = gpu_stat()
            samples.append((psutil.cpu_percent(None), rss / 2**30, vm / 1024, gu))

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    b = subprocess.run([sys.executable, BENCH, '--url', URL + '/v1', '--lengths', '1024,4096', '--max-tokens', '256', '--repeat', '3'],
                       capture_output=True, text=True)
    rows = [l.split() for l in b.stdout.splitlines() if l.split() and l.split()[0] in ('1024', '4096')]
    bench = ' '.join(r[3] for r in rows) + ' (prompt t/s ' + ' '.join(r[2] for r in rows) + ')'
    c = ask(CODE, os.path.join(OUT, name + '-code.json'))
    pr = ask(PROSE, os.path.join(OUT, name + '-prose.json'))
    stop.set()
    th.join()
    cpu = sum(s[0] for s in samples) / max(1, len(samples))
    print(f'{name} | bench 1K/4K: {bench} | code {c:.1f} | prose {pr:.1f} | cpu avg {cpu:.0f}% max {max(s[0] for s in samples):.0f}% | '
          f'rss max {max(s[1] for s in samples):.1f} GB | vram max {max(s[2] for s in samples):.1f} GB | gpu avg {sum(s[3] for s in samples)/max(1,len(samples)):.0f}% | load {load_s:.0f} s', flush=True)
    for ch in p.children(recursive=True):
        ch.kill()
    proc.kill()
    proc.wait()
    time.sleep(3)


if __name__ == '__main__':
    main()
