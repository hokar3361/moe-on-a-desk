# export the recursion adapter as raw float32 for the C++ side:
#   <out>.full.bin : header [NE, 0] then mx[NE], b[NE], W[NE][NE] (row-major, p = x + W^T (x - mx) + b)
#   <out>.lr.bin   : header [NE, r] then mx[NE], b[NE], U[NE][r] (= U*S), V[r][NE]
import sys, numpy as np
NE = 10240; HC = 4; NS = 2560
rec = np.dtype([("tok", "<i4"), ("hm", "<f2", NE), ("ht", "<f2", NE)])
d = np.memmap(sys.argv[1], dtype=rec, mode="r")
N = len(d); ntr = int(N*0.8)
out = sys.argv[2]
def norm_streams(a):
    a = a.astype(np.float32).reshape(-1, HC, NS)
    a = a / np.sqrt(np.mean(a*a, axis=2, keepdims=True) + 1e-6)
    return a.reshape(-1, NE)
def batches(lo, hi, bs=4096):
    for i in range(lo, hi, bs):
        j = min(hi, i + bs)
        yield norm_streams(d["hm"][i:j]), norm_streams(d["ht"][i:j])
use_all = len(sys.argv) > 3 and sys.argv[3] == "all"
hi = N if use_all else ntr
mx = np.zeros(NE); md = np.zeros(NE); n = 0
for X, Y in batches(0, hi):
    mx += X.sum(0); md += (Y - X).sum(0); n += len(X)
mx /= n; md /= n
G = np.zeros((NE, NE)); C = np.zeros((NE, NE))
for X, Y in batches(0, hi):
    Xc = (X - mx).astype(np.float64); Dc = (Y - X - md).astype(np.float64)
    G += Xc.T @ Xc; C += Xc.T @ Dc
tr = np.trace(G)/NE
W = np.linalg.solve(G + 0.1*tr*np.eye(NE), C)
with open(out + ".full.bin", "wb") as f:
    np.array([NE, 0], dtype=np.int32).tofile(f)
    mx.astype(np.float32).tofile(f); md.astype(np.float32).tofile(f); W.astype(np.float32).tofile(f)
W1 = np.linalg.solve(G + 1.0*tr*np.eye(NE), C)
U, S, Vt = np.linalg.svd(W1, full_matrices=False)
r = 1024
with open(out + ".lr.bin", "wb") as f:
    np.array([NE, r], dtype=np.int32).tofile(f)
    mx.astype(np.float32).tofile(f); md.astype(np.float32).tofile(f)
    (U[:, :r]*S[:r]).astype(np.float32).tofile(f); Vt[:r].astype(np.float32).tofile(f)
print("exported", out, "samples", hi)
