# bytes per token by tensor class for an MoE GGUF (header only): dense tensors read fully, routed experts n_used/n_expert of them
import sys, collections
sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader
r = GGUFReader(sys.argv[2])
kv = {f.name: f for f in r.fields.values()}
def val(k):
    f = kv.get(k)
    return None if f is None else f.parts[f.data[0]][0]
arch = bytes(kv["general.architecture"].parts[kv["general.architecture"].data[0]]).decode()
ne = int(val(arch + ".expert_count")); nu = int(val(arch + ".expert_used_count"))
cls = collections.defaultdict(lambda: [0, 0.0])
for t in r.tensors:
    n = t.name; b = int(t.n_bytes)
    if "_exps" in n: k = "routed " + t.tensor_type.name; per = b * nu / ne
    elif n.startswith("token_embd"): k = "token_embd (1 row)"; per = 0
    elif "nextn" in n or "mtp" in n: k = "mtp " + t.tensor_type.name; per = 0
    else: k = "dense " + t.tensor_type.name; per = b
    cls[k][0] += b; cls[k][1] += per
tot = 0
for k, (b, per) in sorted(cls.items(), key=lambda x: -x[1][1]):
    print("%-26s file %7.2f GB  per token %6.3f GB" % (k, b / 1e9, per / 1e9)); tot += per
print("arch %s experts %d used %d; per token %.3f GB -> floor at 546 GB/s %.2f ms (B=1)" % (arch, ne, nu, tot / 1e9, tot / 546e9 * 1e3))
