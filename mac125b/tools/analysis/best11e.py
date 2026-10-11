# run-best.sh / bench-best.sh -> -11e (stop rule); console.py: "best" stays -11d (tree ~/llama-11d), new engine "e11e"
import os
H = os.path.expanduser("~")

p = H + "/run-best.sh"
s = open(p).read()
s = s.replace("tag mac125b-best-2026-10-11d", "tag mac125b-best-2026-10-11e")
old = """env GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 \\
    LLAMA_SPEC_DEPTH_CTL=1 LLAMA_SPEC_DEPTH_MIN=1 LLAMA_SPEC_DEPTH_RESET=1 \\"""
assert s.count(old) == 1
s = s.replace(old, """env GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1 \\""")
s = s.replace("# MTP catch-up merged into the first draft step, adaptive draft depth (model-based, reset per request,\n# acceptance-driven exploration),",
              "# MTP catch-up merged into the first draft step, per-draft stop rule from the MTP head's calibrated confidence,")
open(p, "w").write(s)

p = H + "/bench-best.sh"
s = open(p).read()
old = 'E="GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_DEPTH_CTL=1 LLAMA_SPEC_DEPTH_MIN=1 LLAMA_SPEC_DEPTH_RESET=1"'
assert s.count(old) == 1
s = s.replace(old, 'E="GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1"')
open(p, "w").write(s)

p = H + "/console/console.py"
s = open(p, encoding="utf-8").read()
old = '''        "cmd": [HOME + "/llama-best/build/bin/llama-server"] + LLAMA_COMMON + ["--spec-draft-n-max", "7", "--port", "8082"],'''
assert s.count(old) == 1
s = s.replace(old, '''        "cmd": [HOME + "/llama-11d/build/bin/llama-server"] + LLAMA_COMMON + ["--spec-draft-n-max", "7", "--port", "8082"],''')
s = s.replace('"label": "-11d (このプロジェクトの最良、タグ mac125b-best-2026-10-11d)",', '"label": "-11d (前の最良、タグ mac125b-best-2026-10-11d)",')
old = '''        "port": 8082, "health": "/health",
    },
}'''
assert s.count(old) == 1
s = s.replace(old, '''        "port": 8082, "health": "/health",
    },
    "e11e": {
        "label": "-11e (このプロジェクトの最良、タグ mac125b-best-2026-10-11e、下書きの停止規則)",
        "cmd": [HOME + "/llama-best/build/bin/llama-server"] + LLAMA_COMMON + ["--spec-draft-n-max", "7", "--port", "8084"],
        "env": {"GGML_ALLOC_FREE_DELAY": "4", "LLAMA_SPEC_MTP_MERGE": "1", "LLAMA_GRAPH_CACHE": "16", "LLAMA_SPEC_STOP": "1"},
        "port": 8084, "health": "/health",
    },
}''')
old = '''    "sushi": {"model": "Qwen3.8-Flash-Next 125B (MoE、512 エキスパート)", "quant": "Sushi-2.6bpw'''
assert s.count(old) == 1
s = s.replace(old, '''    "e11e":  {"model": "Qwen3.8-Flash-Next 125B (MoE、512 エキスパート)", "quant": "IQ3_XXS GSQ-RCO(開始時点と同一ファイル)+ MTP Q8_0",
              "gpu": "重み 43.8 GiB + MTP 3.9 GiB", "disk": "71 GiB(n-gram 表 27.5 GB を含む)+ MTP 3.9 GiB"},
''' + old)
s = s.replace(''' --c-start:#8a94a6;--c-sushi:#d08a2b;--c-best:#2f6fed;--c-tf27b:#16a39a}''', ''' --c-start:#8a94a6;--c-sushi:#d08a2b;--c-best:#2f6fed;--c-tf27b:#16a39a;--c-e11e:#c0399b}''')
s = s.replace(''' --c-start:#9aa4b5;--c-sushi:#e3a04a;--c-best:#5b8dff;--c-tf27b:#2cc4b9}}''', ''' --c-start:#9aa4b5;--c-sushi:#e3a04a;--c-best:#5b8dff;--c-tf27b:#2cc4b9;--c-e11e:#e05cbf}}''')
s = s.replace('''.b-tf27b{background:var(--c-tf27b)}''', '''.b-tf27b{background:var(--c-tf27b)}.b-e11e{background:var(--c-e11e)}''')
old = '''const short={start:"開始時点",sushi:"Sushi",best:"-11d",tf27b:"TF 27B"};
const ORDER=["start","sushi","best","tf27b"];'''
assert s.count(old) == 1
s = s.replace(old, '''const short={start:"開始時点",sushi:"Sushi",best:"-11d",e11e:"-11e",tf27b:"TF 27B"};
const ORDER=["start","sushi","best","e11e","tf27b"];''')
s = s.replace('''<option value="best">-11d</option>''', '''<option value="best">-11d</option><option value="e11e">-11e</option>''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("best11e ok")
