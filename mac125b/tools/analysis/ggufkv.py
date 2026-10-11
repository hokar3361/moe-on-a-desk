# print the scalar / small-array metadata of a GGUF file (no numpy)
import struct, sys
def rd(f, fmt):
    n = struct.calcsize(fmt); return struct.unpack("<" + fmt, f.read(n))
def rstr(f):
    (n,) = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
FMT = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
def val(f, t):
    if t in FMT: return rd(f, FMT[t])[0]
    if t == 8: return rstr(f)
    if t == 9:
        (at,) = rd(f, "I"); (n,) = rd(f, "Q")
        v = [val(f, at) for _ in range(n)]
        return v if n <= 64 else "[%d items]" % n
with open(sys.argv[1], "rb") as f:
    assert f.read(4) == b"GGUF"
    rd(f, "I"); (nt,) = rd(f, "Q"); (nkv,) = rd(f, "Q")
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, "I"); v = val(f, t)
        if "tokenizer" in k and not isinstance(v, (int, float, str)): v = "<...>"
        print(k, "=", v)
