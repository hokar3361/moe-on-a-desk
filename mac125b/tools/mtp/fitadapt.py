# fit a recursion adapter: map the MTP output residual h_mtp[k] onto the target hidden h_tgt[k] (what the next draft
# step should get as h). Works in per-stream RMS-normalized space (the MTP re-normalizes h per stream anyway).
# usage: fitadapt.py pairs.bin [out.npz]
import sys, numpy as np, time
NE = 10240; HC = 4; NS = 2560
rec = np.dtype([("tok", "<i4"), ("hm", "<f2", NE), ("ht", "<f2", NE)])
d = np.memmap(sys.argv[1], dtype=rec, mode="r")
N = len(d); ntr = int(N*0.8)
print("pairs", N, "train", ntr, "test", N - ntr, flush=True)

def norm_streams(a):
    a = a.astype(np.float32).reshape(-1, HC, NS)
    a = a / np.sqrt(np.mean(a*a, axis=2, keepdims=True) + 1e-6)
    return a.reshape(-1, NE)

def batches(lo, hi, bs=4096):
    for i in range(lo, hi, bs):
        j = min(hi, i + bs)
        yield norm_streams(d["hm"][i:j]), norm_streams(d["ht"][i:j])

# baseline: how far is h_mtp from h_tgt (per stream cosine, relative error)
def stats(lo, hi, W=None, mx=None, b=None):
    err = 0.0; tot = 0.0; cos = []
    for X, Y in batches(lo, hi):
        P = X if W is None else X + (X - mx) @ W + b
        err += np.sum((Y - P)**2); tot += np.sum((Y - Y.mean(0))**2)
        Pn = P.reshape(-1, HC, NS); Yn = Y.reshape(-1, HC, NS)
        c = np.sum(Pn*Yn, 2)/(np.linalg.norm(Pn, axis=2)*np.linalg.norm(Yn, axis=2) + 1e-9)
        cos.append(c.mean())
    return err/tot, float(np.mean(cos))

t0 = time.time()
print("identity: rel err %.4f, stream cosine %.4f" % stats(ntr, N), flush=True)

# normal equations for the residual D = Y - X on centred X
mx = np.zeros(NE); md = np.zeros(NE); n = 0
for X, Y in batches(0, ntr):
    mx += X.sum(0); md += (Y - X).sum(0); n += len(X)
mx /= n; md /= n
G = np.zeros((NE, NE)); C = np.zeros((NE, NE))
for X, Y in batches(0, ntr):
    Xc = (X - mx).astype(np.float64); Dc = (Y - X - md).astype(np.float64)
    G += Xc.T @ Xc; C += Xc.T @ Dc
print("normal equations %.0f s" % (time.time() - t0), flush=True)
tr = np.trace(G)/NE
for lam in [1e-2, 1e-1, 1.0]:
    W = np.linalg.solve(G + lam*tr*np.eye(NE), C)
    b = md.astype(np.float32)
    W32 = W.astype(np.float32); mx32 = mx.astype(np.float32)
    e, c = stats(ntr, N, W32, mx32, b)
    print("ridge lam %.0e: rel err %.4f, stream cosine %.4f (%.0f s)" % (lam, e, c, time.time() - t0), flush=True)
    # low rank truncations of W
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    for r in [64, 256, 1024]:
        Wr = ((U[:, :r]*S[:r]) @ Vt[:r]).astype(np.float32)
        e, c = stats(ntr, N, Wr, mx32, b)
        print("   rank %4d: rel err %.4f, stream cosine %.4f" % (r, e, c), flush=True)
    if len(sys.argv) > 2 and lam == 1e-1:
        np.savez(sys.argv[2], W=W32, mx=mx32, b=b, U=(U[:, :256]*S[:256]).astype(np.float32), Vt=Vt[:256].astype(np.float32))
