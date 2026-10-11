# tensor-type file for llama-quantize: every tensor keeps its current type except the routed experts (-> argv[2]).
# A tensor whose target type equals its current type is copied unchanged by llama-quantize.
# usage: python3 mktypes.py <expert type> <out file> shard1.gguf [shard2.gguf ...]
import struct, sys, re
TN = {0:"f32",1:"f16",2:"q4_0",3:"q4_1",6:"q5_0",7:"q5_1",8:"q8_0",9:"q8_1",10:"q2_K",11:"q3_K",12:"q4_K",13:"q5_K",14:"q6_K",15:"q8_K",
      16:"iq2_xxs",17:"iq2_xs",18:"iq3_xxs",19:"iq1_s",20:"iq4_nl",21:"iq3_s",22:"iq2_s",23:"iq4_xs",24:"i8",25:"i16",26:"i32",27:"i64",28:"f64",29:"iq1_m",30:"bf16",34:"tq1_0",35:"tq2_0",39:"mxfp4",40:"nvfp4",41:"q1_0",42:"q2_0"}
def rd(f, fmt):
    n = struct.calcsize(fmt); return struct.unpack("<" + fmt, f.read(n))
def rstr(f):
    (n,) = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
def skipval(f, t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sz: f.read(sz[t])
    elif t == 8: rstr(f)
    elif t == 9:
        (at,) = rd(f, "I"); (n,) = rd(f, "Q")
        for _ in range(n): skipval(f, at)
etype, out = sys.argv[1], sys.argv[2]
lines, nexp = [], 0
for path in sys.argv[3:]:
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        (ver,) = rd(f, "I"); (nt,) = rd(f, "Q"); (nkv,) = rd(f, "Q")
        for _ in range(nkv):
            rstr(f); (t,) = rd(f, "I"); skipval(f, t)
        for _ in range(nt):
            name = rstr(f); (nd,) = rd(f, "I"); rd(f, "%dQ" % nd); (t,) = rd(f, "I"); rd(f, "Q")
            if re.search(r"ffn_(gate|up|down)_exps", name):
                lines.append("%s=%s" % (re.escape(name), etype)); nexp += 1
            else:
                lines.append("%s=%s" % (re.escape(name), TN.get(t, "?%d" % t)))
open(out, "w").write("\n".join(lines) + "\n")
print("tensors %d, experts %d -> %s" % (len(lines), nexp, out))
