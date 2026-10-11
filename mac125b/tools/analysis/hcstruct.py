# structure of the hyper-connection weights (bf16): relations between the 4 stream blocks, effective rank,
# relations across layers. usage: python hcstruct.py <gguf shard with blk.N> [layers]
import struct, sys, numpy as np

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

def index(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    rd(f, "I"); (nt,) = rd(f, "Q"); (nkv,) = rd(f, "Q")
    align = 32
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, "I")
        if k == "general.alignment": (align,) = rd(f, "I")
        else: skipval(f, t)
    infos = {}
    for _ in range(nt):
        name = rstr(f); (nd,) = rd(f, "I"); dims = rd(f, "Q"*nd); (tt,) = rd(f, "I"); (off,) = rd(f, "Q")
        infos[name] = (dims, tt, off)
    base = (f.tell() + align - 1)//align*align
    return f, infos, base

def load_bf16(f, infos, base, name):
    dims, tt, off = infos[name]
    assert tt == 30, (name, tt)
    n = 1
    for d in dims: n *= d
    f.seek(base + off)
    u = np.frombuffer(f.read(2*n), dtype=np.uint16).astype(np.uint32) << 16
    a = u.view(np.float32)
    # ggml dims: ne0 = fastest. reshape to [ne1, ne0]
    return a.reshape(dims[1], dims[0]) if len(dims) == 2 else a

f, infos, base = index(sys.argv[1])
layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 23, 47]
np.set_printoptions(precision=4, suppress=True, linewidth=160)
for il in layers:
    for kind in ["attn", "ffn"]:
        nd = "blk.%d.hc_%s_down.weight" % (il, kind)
        nu = "blk.%d.hc_%s_up.weight" % (il, kind)
        ni = "blk.%d.hc_%s_inject.weight" % (il, kind)
        if nd not in infos:
            continue
        D = load_bf16(f, infos, base, nd)   # [320, 10240]: row r = output r, columns = 4 streams x 2560
        U = load_bf16(f, infos, base, nu)   # [10240, 320]
        I = load_bf16(f, infos, base, ni)   # [4, 10240]
        print("== layer %d %s: down %s up %s inject %s" % (il, kind, D.shape, U.shape, I.shape))
        Ds = [D[:, s*2560:(s+1)*2560] for s in range(4)]
        Us = [U[s*2560:(s+1)*2560, :] for s in range(4)]
        # block relations: cosine between flattened stream blocks, and best scalar fit residual
        for name, B in (("down", Ds), ("up", Us)):
            cos = np.zeros((4, 4))
            for a in range(4):
                for b in range(4):
                    va, vb = B[a].ravel(), B[b].ravel()
                    cos[a, b] = va @ vb / (np.linalg.norm(va)*np.linalg.norm(vb) + 1e-30)
            print("  %s block cosine:\n%s" % (name, cos))
            print("  %s block norms: %s, exact equal blocks: %s" % (name, [round(float(np.linalg.norm(x)), 3) for x in B],
                  [bool(np.array_equal(B[0], B[s])) for s in range(1, 4)]))
        # effective rank (singular values) of down and of the stream-summed down
        sv = np.linalg.svd(D, compute_uv=False)
        e = np.cumsum(sv**2)/np.sum(sv**2)
        print("  down singular values: max %.3f, 90%% energy at rank %d, 99%% at %d, 99.9%% at %d (of %d)" %
              (sv[0], np.searchsorted(e, 0.90) + 1, np.searchsorted(e, 0.99) + 1, np.searchsorted(e, 0.999) + 1, len(sv)))
        sv = np.linalg.svd(U, compute_uv=False)
        e = np.cumsum(sv**2)/np.sum(sv**2)
        print("  up   singular values: max %.3f, 90%% energy at rank %d, 99%% at %d, 99.9%% at %d" %
              (sv[0], np.searchsorted(e, 0.90) + 1, np.searchsorted(e, 0.99) + 1, np.searchsorted(e, 0.999) + 1))
        # sparsity / value set
        for name, X in (("down", D), ("up", U)):
            ax = np.abs(X)
            print("  %s: zeros %.4f, |x|<1e-3 %.4f, distinct values %d, max %.3f" % (name, np.mean(X == 0), np.mean(ax < 1e-3), len(np.unique(X)), ax.max()))
        print("  inject rows cosine:\n%s" % np.corrcoef(I))
