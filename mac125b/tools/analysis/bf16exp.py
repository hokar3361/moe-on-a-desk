# exponent statistics of bf16 tensors in a GGUF (pure python): per tensor role, how many distinct exponents,
# the share covered by the best 16-wide window, and the entropy (bits) of the exponent
import struct, sys, re, collections, math, array
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
want = re.compile(sys.argv[2]) if len(sys.argv) > 2 else re.compile(r"blk\.(0|11|23|47)\.(hc_|ffn_gate_inp\.)")
path = sys.argv[1]
with open(path, "rb") as f:
    assert f.read(4) == b"GGUF"
    (ver,) = rd(f, "I"); (nt,) = rd(f, "Q"); (nkv,) = rd(f, "Q")
    align = 32
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, "I")
        if k == "general.alignment":
            (align,) = rd(f, "I")
        else:
            skipval(f, t)
    infos = []
    for _ in range(nt):
        name = rstr(f); (nd,) = rd(f, "I"); dims = rd(f, "Q"*nd); (tt,) = rd(f, "I"); (off,) = rd(f, "Q")
        infos.append((name, dims, tt, off))
    base = (f.tell() + align - 1)//align*align
    for name, dims, tt, off in infos:
        if tt != 30 or not want.search(name):
            continue
        n = 1
        for d in dims: n *= d
        f.seek(base + off)
        a = array.array("H"); a.frombytes(f.read(2*n))
        h = collections.Counter(((v >> 7) & 0xff) for v in a)
        tot = sum(h.values())
        ent = -sum(c/tot*math.log2(c/tot) for c in h.values())
        best = max(sum(h.get(e, 0) for e in range(lo, lo + 16)) for lo in range(0, 256))
        best8 = max(sum(h.get(e, 0) for e in range(lo, lo + 8)) for lo in range(0, 256))
        print("%-32s n=%8d distinct=%3d H=%.2f bits  in best 16-window %.5f  8-window %.5f" % (name, n, len(h), ent, best/tot, best8/tot))
