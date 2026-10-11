#!/usr/bin/env python3
# Offline go/no-go for lossless mid-pass row pruning of the speculative verify pass (see design.md).
#
# Reads LLAMA_ROWPRUNE_DUMP files (dumppatch.py), trains probes that predict, from the HC residual of verify row j at
# an intermediate layer, whether the target will reject the next draft token d_{j+1} (argmax(row j) != d_{j+1}), and
# turns catch / false-drop rates into verify-time savings with the measured cost model.
#
#   python probe.py --dump train.bin --sidecar train.jsonl --held held.bin --held-sidecar held.jsonl \
#       --head head.bin --mtp-gguf D:\GPUNW\mac125b\mtptrain\mtp-Q8_0.gguf --out report.json
#   python probe.py --selftest           (synthetic dumps; checks the reader, labels, probes and simulator end to end)
#
# Probes (score = log-odds of "d_{j+1} rejected"; the cut is at the first row whose score exceeds tau):
#   oracle      the true label (ceiling)
#   draft       MTP top-8 statistics of d_{j+1} only (known before the verify; the same model applied BEFORE the verify
#               is the "pre" baseline = truncating the draft by its own confidence, which skips whole rows)
#   linear_h    logistic regression on the per-stream RMS-normalised residual (10240-d) alone
#   linear      the same + draft features (needs the MTP statistics of each row as a small graph input)
#   lens_cand   logit lens (the model's own final HC mixer + output head) restricted to the 8 MTP candidates of
#               position j+1 + draft features (deployable: 8*B head rows, ~0.1 ms)
#   lens_full   full-vocabulary logit lens + draft features (reference: costs a full head pass, ~1.2 ms)
#   tuned_cand  as lens_cand with a learned rank-r correction of the mixed stream (--tuned-steps > 0)
#
# Cost model (ms, DESIGN.md 9.13-9.15, B = verify rows): Tv(1)=17.93, Tv(B>=2)=23.0+2.82*(B-2); MoE per live row per
# pass 2.65 (spread evenly over the 48 layers); draft step 1.18; host 0.5. All overridable.
import argparse, json, math, os, struct, sys, tempfile
import numpy as np
import torch
import torch.nn.functional as F

NE, HC = 2560, 4
EPS = 1e-6
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------------------------ dump reader

class Dump:
    """All records of one or more dump files. Hidden states stay on disk until load_layer()."""

    def __init__(self, paths, sidecars=None, max_records=None, tag=""):
        self.paths = paths
        self.layers = None
        self.ne_row = self.K = None
        R = dict(fi=[], hoff=[], B=[], nacc=[], req=[], pos0=[], flags=[], tok=[], am=[], lse=[], l1=[], l2=[],
                 lnext=[], cid=[], clg=[], cp=[])
        n_trunc = 0
        for fi, p in enumerate(paths):
            size = os.path.getsize(p)
            with open(p, "rb") as f:
                if f.read(4) != b"RPD1":
                    sys.exit("%s: not a rowprune dump" % p)
                ver, nl = struct.unpack("<ii", f.read(8))
                layers = list(struct.unpack("<%di" % nl, f.read(4 * nl)))
                ne_row, K = struct.unpack("<ii", f.read(8))
                if self.layers is None:
                    self.layers, self.ne_row, self.K = layers, ne_row, K
                elif (layers, ne_row, K) != (self.layers, self.ne_row, self.K):
                    sys.exit("%s: layer list / row size differs from the first dump" % p)
                while True:
                    m = f.read(4)
                    if len(m) < 4:
                        break
                    if m != b"RPR1":
                        sys.exit("%s: bad record magic at %d" % (p, f.tell() - 4))
                    hdr = f.read(20)
                    if len(hdr) < 20:
                        n_trunc += 1; break
                    B, nacc, req, pos0, flags = struct.unpack("<5i", hdr)
                    nc = (B - 1) * K
                    body = f.read(4 * (6 * B + 3 * nc))
                    hb = nl * B * ne_row * 2
                    if len(body) < 4 * (6 * B + 3 * nc) or f.tell() + hb > size:
                        n_trunc += 1; break
                    ints = np.frombuffer(body, dtype=np.int32)
                    flt = np.frombuffer(body, dtype=np.float32)
                    o = 0
                    R["tok"].append(ints[o:o + B].copy()); o += B
                    R["am"].append(ints[o:o + B].copy()); o += B
                    R["lse"].append(flt[o:o + B].copy()); o += B
                    R["l1"].append(flt[o:o + B].copy()); o += B
                    R["l2"].append(flt[o:o + B].copy()); o += B
                    R["lnext"].append(flt[o:o + B].copy()); o += B
                    R["cid"].append(ints[o:o + nc].reshape(B - 1, K).copy()); o += nc
                    R["clg"].append(flt[o:o + nc].reshape(B - 1, K).copy()); o += nc
                    R["cp"].append(flt[o:o + nc].reshape(B - 1, K).copy()); o += nc
                    R["fi"].append(fi); R["hoff"].append(f.tell()); R["B"].append(B); R["nacc"].append(nacc)
                    R["req"].append(req); R["pos0"].append(pos0); R["flags"].append(flags)
                    f.seek(hb, 1)
                    if max_records and len(R["B"]) >= max_records:
                        break
            if max_records and len(R["B"]) >= max_records:
                break
        self.R = R
        self.n = len(R["B"])
        self.B = np.array(R["B"]); self.nacc = np.array(R["nacc"]); self.fi = np.array(R["fi"])
        self.req = np.array(R["req"]); self.flags = np.array(R["flags"])
        # leading matches = accepted drafts implied by the target argmax; must equal nacc for greedy decoding
        self.lead = np.array([lead_matches(R["am"][i], R["tok"][i]) for i in range(self.n)])
        self.kind = np.array(["?"] * self.n, dtype=object)
        self.reqkey = np.array(["%s:%d:%d" % (tag, a, b) for a, b in zip(self.fi, self.req)], dtype=object)
        if sidecars:
            self._attach_sidecars(sidecars)
        print("%s: %d records (%d truncated tail), layers %s, rows/record %.2f, nacc==argmax-leading %.4f, cand ok %.3f"
              % (tag or "dump", self.n, n_trunc, self.layers, self.B.mean() if self.n else 0,
                 (self.lead == self.nacc).mean() if self.n else 0, (self.flags & 1).mean() if self.n else 0))

    def _attach_sidecars(self, sidecars):
        # requests are served in order: the i-th distinct task id of file fi is the i-th sidecar line of that file
        for fi, sc in enumerate(sidecars):
            if sc is None or fi >= len(self.paths):
                continue
            lines = [json.loads(l) for l in open(sc) if l.strip()]
            idx = np.where(self.fi == fi)[0]
            order = []
            for i in idx:
                if not order or order[-1] != self.req[i]:
                    if self.req[i] not in order:
                        order.append(self.req[i])
            if len(order) != len(lines):
                print("  sidecar %s: %d lines vs %d requests in the dump; kinds not attached" % (sc, len(lines), len(order)))
                continue
            kmap = {r: lines[k]["kind"] for k, r in enumerate(order)}
            for i in idx:
                self.kind[i] = kmap[self.req[i]]

    def load_layer(self, k, rec, row):
        """fp16 [N, ne_row] residual of (record, row) pairs at dumped layer index k."""
        out = np.empty((len(rec), self.ne_row), dtype=np.float16)
        files = [open(p, "rb") for p in self.paths]
        rb = self.ne_row * 2
        order = np.lexsort((row, rec))
        last = -1; blk = None
        for t in order:
            i = rec[t]
            if i != last:
                B = self.B[i]
                f = files[self.fi[i]]
                f.seek(self.R["hoff"][i] + k * B * rb)
                blk = np.frombuffer(f.read(B * rb), dtype=np.float16).reshape(B, self.ne_row)
                last = i
            out[t] = blk[row[t]]
        for f in files:
            f.close()
        return out


def lead_matches(am, tok):
    n = 0
    while n + 1 < len(tok) and am[n] == tok[n + 1]:
        n += 1
    return n


# ------------------------------------------------------------------------------------------------ rows / features

class Rows:
    """Decision rows j = 0..B-2 of every record (row j decides on d_{j+1})."""

    def __init__(self, d: Dump):
        rec, row = [], []
        for i in range(d.n):
            for j in range(d.B[i] - 1):
                rec.append(i); row.append(j)
        self.rec = np.array(rec, dtype=np.int64); self.row = np.array(row, dtype=np.int64)
        R = d.R
        self.y = np.array([R["am"][i][j] != R["tok"][i][j + 1] for i, j in zip(rec, row)], dtype=np.float32)
        self.alive = np.array([j <= d.lead[i] for i, j in zip(rec, row)])
        self.am = np.array([R["am"][i][j] for i, j in zip(rec, row)], dtype=np.int64)
        self.nxt = np.array([R["tok"][i][j + 1] for i, j in zip(rec, row)], dtype=np.int64)
        self.cid = np.stack([R["cid"][i][j] for i, j in zip(rec, row)]) if rec else np.zeros((0, d.K), np.int32)
        self.cand_ok = np.array([(d.flags[i] & 1) == 1 for i in rec], dtype=bool)
        self.draft = draft_features(d, self.rec, self.row)
        self.n = len(rec)


def draft_features(d, rec, row):
    """MTP-side features of the drafted token d_{j+1} (top-8 at draft position j+1): known before the verify."""
    K = d.K
    X = np.zeros((len(rec), 6 + 8), dtype=np.float32)
    for t, (i, j) in enumerate(zip(rec, row)):
        if not (d.flags[i] & 1):
            X[t, 5] = 1.0
            continue
        lg = d.R["clg"][i][j].astype(np.float64)
        ok = np.isfinite(lg)
        lg = np.where(ok, lg, -1e30)
        z = lg - lg.max()
        p = np.exp(z) / np.exp(z).sum()
        top = p[0]
        m = lg[0] - (lg[1] if K > 1 and ok[1] else lg[0] - 30.0)
        H = -(p[p > 0] * np.log(p[p > 0])).sum()
        X[t, 0] = math.log(max(top, 1e-12))
        X[t, 1] = min(m, 30.0)
        X[t, 2] = H
        X[t, 3] = d.R["cp"][i][j][0]
        X[t, 4] = float(d.R["cid"][i][j][0] == d.R["tok"][i][j + 1])   # drafted token is the MTP top-1
        X[t, 6 + min(j, 7)] = 1.0                                        # draft position one-hot
    return X


# ------------------------------------------------------------------------------------------------ logit lens

def _gguf_tensors(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))
    def rs():
        (n,) = rd("Q"); return f.read(n).decode("utf-8", "replace")
    def skip(t):
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sz: f.read(sz[t])
        elif t == 8: rs()
        elif t == 9:
            (at,) = rd("I"); (n,) = rd("Q")
            for _ in range(n): skip(at)
    assert f.read(4) == b"GGUF"
    rd("I"); (nt,) = rd("Q"); (nkv,) = rd("Q")
    align = 32
    for _ in range(nkv):
        k = rs(); (t,) = rd("I")
        if k == "general.alignment": (align,) = rd("I")
        else: skip(t)
    infos = {}
    for _ in range(nt):
        name = rs(); (nd,) = rd("I"); dims = rd("Q" * nd); (tt,) = rd("I"); (off,) = rd("Q")
        infos[name] = (dims, tt, off)
    base = (f.tell() + align - 1) // align * align
    return f, base, infos


def _decode(raw, tt, n):
    if tt == 0: return np.frombuffer(raw, dtype=np.float32, count=n).astype(np.float32)
    if tt == 1: return np.frombuffer(raw, dtype=np.float16, count=n).astype(np.float32)
    if tt == 30: return (np.frombuffer(raw, dtype=np.uint16, count=n).astype(np.uint32) << 16).view(np.float32)
    if tt == 8:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
        return (b[:, 2:].view(np.int8).astype(np.float32) * b[:, :2].copy().view(np.float16).astype(np.float32)).reshape(-1)
    raise ValueError("ggml type %d not supported" % tt)


def load_out_weight(path):
    """Output head [V, NE] as fp16 on DEV: output.weight of a GGUF (MTP GGUF: Q8_0 copy of the shared head) or a .npy."""
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path).astype(np.float16)).to(DEV)
    f, base, infos = _gguf_tensors(path)
    dims, tt, off = infos["output.weight"]
    ne0, V = dims
    assert tt in (0, 1, 8, 30), tt
    rowb = {0: 4 * ne0, 1: 2 * ne0, 30: 2 * ne0, 8: ne0 // 32 * 34}[tt]
    W = torch.empty((V, ne0), dtype=torch.float16, device=DEV)
    f.seek(base + off)
    ch = 8192
    for r0 in range(0, V, ch):
        nr = min(ch, V - r0)
        a = _decode(f.read(nr * rowb), tt, nr * ne0).reshape(nr, ne0)
        W[r0:r0 + nr] = torch.from_numpy(a).to(DEV).half()
    return W


def load_hc_head(path):
    """Target's output_hc_{norm,down,up} from extract_head.py (RPHW)."""
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"RPHW"
        (n,) = struct.unpack("<I", f.read(4))
        for _ in range(n):
            (ln,) = struct.unpack("<I", f.read(4)); name = f.read(ln).decode()
            tt, nd = struct.unpack("<II", f.read(8)); dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            (nb,) = struct.unpack("<Q", f.read(8))
            a = _decode(f.read(nb), tt, int(np.prod(dims)))
            out[name] = torch.from_numpy(a.reshape(tuple(reversed(dims))).copy()).to(DEV)
    return out["output_hc_norm.weight"].reshape(HC, NE), out["output_hc_down.weight"], out["output_hc_up.weight"]


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def stream_rms(H, bs=8192):
    """per-stream RMS normalisation of [N, HC*NE] fp16 rows (the input normalisation every HC mixer applies)"""
    out = np.empty_like(H)
    with torch.no_grad():
        for s in range(0, H.shape[0], bs):
            x = torch.from_numpy(H[s:s + bs]).to(DEV).float().view(-1, HC, NE)
            out[s:s + bs] = rms(x).reshape(-1, HC * NE).half().cpu().numpy()
    return out


def hc_head_mix(h, wn, wd, wu):
    """qwen4exp build_hc_mix(res_hc, hc_head_norm, hc_head_down, hc_head_up) (il = -1, unfused path)."""
    T = h.shape[0]
    xn = rms(h.view(T, HC, NE)) * wn
    flat = xn.reshape(T, HC * NE)
    lo = F.silu((flat @ wd.t()) / HC)
    gate = (lo @ wu.t()).view(T, HC, NE)
    return (xn * torch.sigmoid(gate)).mean(1)


class Lens:
    def __init__(self, hc, W):
        self.wn, self.wd, self.wu = hc
        self.W = W
        self.tuned = {}

    def mixed(self, h, layer=None):
        m = hc_head_mix(h, self.wn, self.wd, self.wu)
        if layer is not None and layer in self.tuned:
            A, Bm, b = self.tuned[layer]
            m = m + (m @ A) @ Bm + b
        return m

    def features(self, H, nxt, cid, am=None, layer=None, bs=256):
        """per row: full-lens [logp(next), max-l(next), log1p(rank), is_top1, entropy] and candidate-restricted
        [logp_C(next), l(next)-max_C\\next, is_top_C]; plus lens argmax (for the self-test)."""
        N = H.shape[0]
        full = np.zeros((N, 5), np.float32); cand = np.zeros((N, 3), np.float32); amax = np.zeros(N, np.int64)
        with torch.no_grad():
            for s in range(0, N, bs):
                h = torch.from_numpy(H[s:s + bs]).to(DEV).float()
                m = self.mixed(h, layer)
                lg = (m.half() @ self.W.t()).float()                       # [b, V]
                nx = torch.from_numpy(nxt[s:s + bs]).to(DEV)
                ln = lg.gather(1, nx[:, None])[:, 0]
                lse = torch.logsumexp(lg, 1)
                mx, ax = lg.max(1)
                rank = (lg > ln[:, None]).sum(1).float()
                p = torch.softmax(lg, 1)
                ent = -(p * torch.log(p.clamp_min(1e-30))).sum(1)
                full[s:s + bs] = torch.stack([ln - lse, mx - ln, torch.log1p(rank), (rank == 0).float(), ent], 1).cpu().numpy()
                amax[s:s + bs] = ax.cpu().numpy()
                c = torch.from_numpy(cid[s:s + bs].astype(np.int64)).to(DEV)
                lc = lg.gather(1, c.clamp_min(0))
                lc = torch.where((c < 0) | (c == nx[:, None]), torch.full_like(lc, -1e30), lc)
                lcs = torch.cat([lc, ln[:, None]], 1)
                cand[s:s + bs] = torch.stack([ln - torch.logsumexp(lcs, 1), (ln - lc.max(1).values).clamp(-50, 50),
                                              (ln >= lc.max(1).values).float()], 1).cpu().numpy()
        return full, cand, amax

    def train_tuned(self, layer, H, am, steps, rank=128, bs=256, lr=1e-3):
        """rank-r correction of the mixed stream, trained to predict the target's final argmax (hard-label CE)."""
        A = (torch.randn(NE, rank, device=DEV) * 1e-3).requires_grad_()
        Bm = torch.zeros(rank, NE, device=DEV, requires_grad=True)
        b = torch.zeros(NE, device=DEV, requires_grad=True)
        opt = torch.optim.AdamW([A, Bm, b], lr=lr, weight_decay=0.0)
        N = H.shape[0]
        g = torch.Generator().manual_seed(0)
        for it in range(steps):
            idx = torch.randint(0, N, (bs,), generator=g).numpy()
            h = torch.from_numpy(H[idx]).to(DEV).float()
            with torch.no_grad():
                m0 = hc_head_mix(h, self.wn, self.wd, self.wu)
            m = m0 + (m0 @ A) @ Bm + b
            lg = (m @ self.W.t().float()) if self.W.dtype == torch.float32 else (m.half() @ self.W.t()).float()
            loss = F.cross_entropy(lg, torch.from_numpy(am[idx]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            if it % max(1, steps // 5) == 0:
                print("    tuned lens il=%d step %d CE %.3f" % (layer, it, loss.item()))
        self.tuned[layer] = (A.detach(), Bm.detach(), b.detach())


# ------------------------------------------------------------------------------------------------ probes

def auc(s, y):
    y = y.astype(bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class LR:
    """Logistic regression on GPU (standardised inputs, AdamW, early stop on validation AUC)."""

    def __init__(self, epochs=8, lr=None, wd=1e-4, bs=4096):
        self.epochs, self.lr, self.wd, self.bs = epochs, lr, wd, bs

    def _std(self, X):
        return (X - self.mu) / self.sd

    def fit(self, Xtr, ytr, Xva, yva):
        Xtr = torch.as_tensor(Xtr, device=DEV)
        d = Xtr.shape[1]
        acc = torch.zeros(d, device=DEV, dtype=torch.float64); acc2 = torch.zeros_like(acc)
        for s in range(0, Xtr.shape[0], 8192):
            x = Xtr[s:s + 8192].double(); acc += x.sum(0); acc2 += (x * x).sum(0)
        n = Xtr.shape[0]
        self.mu = (acc / n).float(); self.sd = ((acc2 / n - (acc / n) ** 2).clamp_min(1e-12).sqrt()).float()
        lr = self.lr or (1e-3 if d > 256 else 1e-2)
        p0 = float(np.clip(ytr.mean(), 1e-4, 1 - 1e-4))
        self.w = torch.zeros(d, device=DEV, requires_grad=True)
        self.b = torch.tensor([math.log(p0 / (1 - p0))], device=DEV, requires_grad=True)
        opt = torch.optim.AdamW([self.w, self.b], lr=lr, weight_decay=self.wd)
        yt = torch.as_tensor(ytr, device=DEV)
        best = (-1.0, self.w.detach().clone(), self.b.detach().clone())
        g = torch.Generator(device="cpu").manual_seed(0)
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g).to(DEV)
            for s in range(0, n, self.bs):
                i = perm[s:s + self.bs]
                z = self._std(Xtr[i].float()) @ self.w + self.b
                loss = F.binary_cross_entropy_with_logits(z, yt[i])
                opt.zero_grad(); loss.backward(); opt.step()
            a = auc(self.score(Xva), yva) if len(yva) else 0.0
            if a > best[0]:
                best = (a, self.w.detach().clone(), self.b.detach().clone())
        self.w = best[1]; self.b = best[2]
        return self

    def score(self, X):
        X = torch.as_tensor(X, device=DEV)
        out = []
        with torch.no_grad():
            for s in range(0, X.shape[0], 16384):
                out.append((self._std(X[s:s + 16384].float()) @ self.w + self.b).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


# ------------------------------------------------------------------------------------------------ simulator

class Cost:
    def __init__(self, a):
        self.tv1, self.tv2, self.row, self.moe = a.tv1, a.tv2, a.row_ms, a.moe_row_ms
        self.draft, self.host, self.n_layer = a.draft_ms, a.host_ms, a.n_layer
        self.probe = dict(kv.split("=") for kv in a.probe_ms.split(","))
        self.probe = {k: float(v) for k, v in self.probe.items()}

    def tv(self, B):
        return self.tv1 if B <= 1 else self.tv2 + self.row * (B - 2)

    def round(self, B, n_draft):
        return self.tv(B) + self.draft * n_draft + self.host

    def frac(self, il):
        return max(0, self.n_layer - 1 - il) / self.n_layer


def first_cut(s, Bp, tau):
    """cut r = 1 + first row j in [0, Bp-2] with score > tau, else Bp (no cut)."""
    for j in range(Bp - 1):
        if s[j] > tau:
            return j + 1
    return Bp


def simulate(recs, d, score_of, Bmax, cost, il, model, tau, pre=None):
    """Totals over records: (tokens, ms) baseline and with pruning.
    score_of(i) -> scores of rows 0..B_i-2 of record i (mid-pass probe at layer il).
    pre=(score_of_draft, tau_d): pre-verify truncation by draft confidence first (hybrid); model None = pre only."""
    tb = Tb = tp = Tp = 0.0
    probe_ms = cost.probe.get(model.split("+")[0], 0.0) if model else 0.0
    for i in recs:
        Bp = min(int(d.B[i]), Bmax + 1)
        n = min(int(d.lead[i]), Bp - 1)
        tb += n + 1
        Tb += cost.round(Bp, Bp - 1)
        r1 = Bp
        if pre is not None:
            r1 = first_cut(pre[0](i), Bp, pre[1])
        T = cost.round(r1, min(r1, Bp - 1))
        r = r1
        if model is not None:
            r2 = first_cut(score_of(i), r1, tau)
            T -= (r1 - r2) * cost.moe * cost.frac(il)
            T += probe_ms
            r = r2
        tp += (n + 1) if n < r else r
        Tp += T
    return tb, Tb, tp, Tp


def speedup(t):
    tb, Tb, tp, Tp = t
    if tb == 0 or Tp == 0:
        return float("nan")
    return (tp / Tp) / (tb / Tb) - 1.0


def best_tau(recs, d, score_of, Bmax, cost, il, model, cand_scores, pre=None):
    qs = np.unique(np.quantile(cand_scores, np.linspace(0.0, 1.0, 81))) if len(cand_scores) else np.array([0.0])
    best = (0.0, float("inf"))      # (speedup, tau): tau=inf = never cut
    for tau in qs:
        sp = speedup(simulate(recs, d, score_of, Bmax, cost, il, model, tau, pre))
        if sp > best[0]:
            best = (sp, float(tau))
    return best[1]


# ------------------------------------------------------------------------------------------------ main analysis

def split_by_request(d, frac_val, frac_test, seed=0):
    keys = sorted(set(d.reqkey))
    rng = np.random.RandomState(seed); rng.shuffle(keys)
    nv, nt = int(len(keys) * frac_val), int(len(keys) * frac_test)
    kv, kt = set(keys[:nv]), set(keys[nv:nv + nt])
    part = np.array([2 if k in kt else 1 if k in kv else 0 for k in d.reqkey])
    return part


def run(a):
    cost = Cost(a)
    paths = a.dump + (a.held or [])
    sidecars = (a.sidecar or [None] * len(a.dump)) + (a.held_sidecar or [None] * len(a.held or []))
    d = Dump(paths, sidecars, a.max_records, tag="all")
    if d.n == 0:
        sys.exit("no records")
    if a.held:
        part = split_by_request(d, 0.2, 0.0, a.seed)
        part[d.fi >= len(a.dump)] = 2
    else:
        part = split_by_request(d, 0.15, 0.15, a.seed)
    rows = Rows(d)
    rpart = part[rows.rec]
    tr = (rpart == 0) & rows.alive & rows.cand_ok if not a.train_all else (rpart == 0) & rows.cand_ok
    va = (rpart == 1) & rows.alive
    te = (rpart == 2) & rows.alive
    print("records train/val/test %d/%d/%d; decision rows (alive) train/val/test %d/%d/%d; reject rate on alive rows %.3f"
          % ((part == 0).sum(), (part == 1).sum(), (part == 2).sum(), tr.sum(), va.sum(), te.sum(), rows.y[rows.alive].mean()))
    for k in sorted(set(d.kind)):
        m = d.kind == k
        print("  kind %-6s records %6d  mean rows %.2f  mean accepted %.2f" % (k, m.sum(), d.B[m].mean(), d.lead[m].mean()))
    # per-position acceptance (P(d_{j+1} accepted | rows <= j alive))
    acc_pos = [1 - rows.y[rows.alive & (rows.row == j)].mean() for j in range(int(d.B.max()) - 1) if (rows.alive & (rows.row == j)).any()]
    print("  acceptance by draft position: " + " ".join("%.3f" % x for x in acc_pos))

    lens = None
    if a.head and a.mtp_gguf:
        lens = Lens(load_hc_head(a.head), load_out_weight(a.mtp_gguf))
        print("lens: head mixer + output head loaded (V=%d)" % lens.W.shape[0])

    # record -> row positions in `rows`, for the simulator
    first = np.full(d.n, -1, np.int64)
    for t in range(rows.n - 1, -1, -1):
        first[rows.rec[t]] = t
    def scorer(S):
        return lambda i: S[first[i]:first[i] + d.B[i] - 1]

    rec_va = np.where(part == 1)[0]; rec_te = np.where(part == 2)[0]
    report = dict(layers={}, cost=vars(a), acc_pos=acc_pos)
    depths = [int(x) for x in a.depths.split(",")]
    targets = [float(x) for x in a.fdr.split(",")]

    # draft-only model (layer independent): also the pre-verify baseline
    m_draft = LR(epochs=a.epochs).fit(rows.draft[tr], rows.y[tr], rows.draft[va], rows.y[va])
    S_draft = m_draft.score(rows.draft)
    pre_tau = {}
    report["pre"] = {}
    for D in depths:
        # pre-only: tune tau_d on validation, report test
        best = (0.0, float("inf"))
        for tau in np.unique(np.quantile(S_draft[va], np.linspace(0, 1, 81))):
            sp = speedup(simulate(rec_va, d, None, D, cost, 0, None, 0, pre=(scorer(S_draft), tau)))
            if sp > best[0]:
                best = (sp, float(tau))
        pre_tau[D] = best[1]
        sp_te = speedup(simulate(rec_te, d, None, D, cost, 0, None, 0, pre=(scorer(S_draft), best[1])))
        bk = {kd: speedup(simulate(rec_te[d.kind[rec_te] == kd], d, None, D, cost, 0, None, 0, pre=(scorer(S_draft), best[1])))
              for kd in sorted(set(d.kind[rec_te]))}
        report["pre"][D] = dict(tau=best[1], test_speedup=sp_te, by_kind=bk)
    print("\npre-verify truncation by draft confidence alone (D1-like; skips whole rows): " +
          "  ".join("depth %d %+.1f%%" % (D, 100 * report["pre"][D]["test_speedup"]) for D in depths))
    print("draft-only AUC (test) %.3f" % auc(S_draft[te], rows.y[te]))

    for k, il in enumerate(d.layers):
        print("\n=== layer il=%d (after %d of %d layers; MoE of %d layers skippable for cut rows) ===" %
              (il, il + 1, a.n_layer, a.n_layer - 1 - il))
        H = d.load_layer(k, rows.rec, rows.row)
        Hn = stream_rms(H)
        models = {"oracle": rows.y * 20.0 - 10.0, "draft": S_draft}
        lin = LR(epochs=a.epochs).fit(np.concatenate([Hn[tr], rows.draft[tr].astype(np.float16)], 1), rows.y[tr],
                                      np.concatenate([Hn[va], rows.draft[va].astype(np.float16)], 1), rows.y[va])
        models["linear"] = lin.score(np.concatenate([Hn, rows.draft.astype(np.float16)], 1))
        # residual only: deployable without passing any MTP statistics into the verify graph
        models["linear_h"] = LR(epochs=a.epochs).fit(Hn[tr], rows.y[tr], Hn[va], rows.y[va]).score(Hn)
        if lens is not None:
            full, cand, amax = lens.features(H, rows.nxt, rows.cid, layer=None)
            agree = float((amax == rows.am).mean())
            report.setdefault("lens_agree", {})[il] = agree
            print("  raw logit lens: top-1 == target final argmax on %.4f of rows%s" %
                  (agree, "  (last layer: checks the dump layout and the head; expect ~0.99+)" if il == a.n_layer - 1 else ""))
            for name, F_ in (("lens_cand", cand), ("lens_full", np.concatenate([full, cand], 1))):
                X = np.concatenate([F_, rows.draft], 1)
                models[name] = LR(epochs=a.epochs * 4).fit(X[tr], rows.y[tr], X[va], rows.y[va]).score(X)
            if a.tuned_steps > 0 and il < a.n_layer - 1:
                lens.train_tuned(il, H[(rpart == 0)], rows.am[(rpart == 0)], a.tuned_steps)
                _, cand_t, amax_t = lens.features(H, rows.nxt, rows.cid, layer=il)
                print("  tuned lens: top-1 == target final argmax on %.4f of rows" % (amax_t == rows.am).mean())
                X = np.concatenate([cand_t, rows.draft], 1)
                models["tuned_cand"] = LR(epochs=a.epochs * 4).fit(X[tr], rows.y[tr], X[va], rows.y[va]).score(X)
        del H, Hn
        L = report["layers"][il] = {}
        print("  %-11s %6s | %s | %s" % ("probe", "AUC", "  ".join("catch@FDR%-4s" % ("%g%%" % (100 * f)) for f in targets),
                                         "  ".join("depth %d: speedup @best-tau (vs pre)" % D for D in depths)))
        for name, S in models.items():
            ent = L[name] = dict(auc=auc(S[te], rows.y[te]), fdr={}, depth={})
            cells = []
            for f in targets:
                neg = S[va][rows.y[va] == 0]
                tau = float(np.quantile(neg, 1 - f)) if len(neg) else float("inf")
                pos_te, neg_te = S[te][rows.y[te] == 1], S[te][rows.y[te] == 0]
                c, fd = (pos_te > tau).mean() if len(pos_te) else float("nan"), (neg_te > tau).mean() if len(neg_te) else float("nan")
                ent["fdr"][f] = dict(tau=tau, catch=float(c), fdr_test=float(fd))
                cells.append("%.3f(%.3f)" % (c, fd))
            dcells = []
            for D in depths:
                tau = best_tau(rec_va, d, scorer(S), D, cost, il, name, S[va])
                sp = speedup(simulate(rec_te, d, scorer(S), D, cost, il, name, tau))
                # hybrid: pre-verify truncation (its own best tau) + this probe mid-pass
                pre = (scorer(S_draft), pre_tau[D])
                tau_h = best_tau(rec_va, d, scorer(S), D, cost, il, name, S[va], pre=pre)
                sp_h = speedup(simulate(rec_te, d, scorer(S), D, cost, il, name, tau_h, pre=pre))
                bykind = {}
                for kd in sorted(set(d.kind[rec_te])):
                    rr = rec_te[d.kind[rec_te] == kd]
                    s1 = speedup(simulate(rr, d, scorer(S), D, cost, il, name, tau))
                    sh = speedup(simulate(rr, d, scorer(S), D, cost, il, name, tau_h, pre=pre))
                    p0 = report["pre"][D]["by_kind"][kd]
                    bykind[kd] = dict(speedup=s1, hybrid_speedup=sh, over_pre=(1 + sh) / (1 + p0) - 1)
                over_pre = (1 + sp_h) / (1 + report["pre"][D]["test_speedup"]) - 1
                ent["depth"][D] = dict(tau=tau, speedup=sp, hybrid_tau=tau_h, hybrid_speedup=sp_h, over_pre=over_pre,
                                       by_kind=bykind)
                dcells.append("%+6.1f%% (pre+this %+5.1f%%)" % (100 * sp, 100 * sp_h))
            print("  %-11s %6.3f | %s | %s" % (name, ent["auc"], "  ".join("%-13s" % c for c in cells), "  ".join(dcells)))
            for D in depths:
                bk = ent["depth"][D]["by_kind"]
                if len(bk) > 1 or "?" not in bk:
                    print("  %-11s        depth %d by kind: %s" % ("", D, "  ".join(
                        "%s %+.1f%% (over pre %+.1f%%)" % (kk, 100 * v["speedup"], 100 * v["over_pre"]) for kk, v in bk.items())))
    report["verdict"] = verdict(report, d, a)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
        print("\nwrote", a.out)
    return report


def verdict(rep, d, a):
    """design.md section 6: GO only if a deployable probe at il <= go_max_il pays on held-out prose AND code, both on its
    own and on top of the pre-verify truncation by draft confidence (which needs no kernel work)."""
    Dp, Dc = a.go_prose_depth, a.go_code_depth
    out = []
    print("\n=== go/no-go (held-out test; prose depth %d: speedup >= +3%% and >= +2%% over pre; code depth %d: >= +2%% and"
          " >= +1%% over pre; deployable probes at il <= %d) ===" % (Dp, Dc, a.go_max_il))
    for il, L in rep["layers"].items():
        if int(il) > a.go_max_il:
            continue
        for name in ("linear_h", "linear", "lens_cand", "tuned_cand"):
            if name not in L:
                continue
            e = L[name]["depth"]

            def get(D, kind):
                if D not in e:
                    return None
                bk = e[D]["by_kind"]
                if kind in bk:
                    return bk[kind]
                return dict(speedup=e[D]["speedup"], over_pre=e[D]["over_pre"]) if (not bk or "?" in bk) else None
            p, c = get(Dp, "prose"), get(Dc, "code")
            ok = (p is not None and p["speedup"] >= 0.03 and p["over_pre"] >= 0.02 and
                  c is not None and c["speedup"] >= 0.02 and c["over_pre"] >= 0.01)
            fmt = lambda v: "n/a" if v is None else "%+.1f%% (over pre %+.1f%%)" % (100 * v["speedup"], 100 * v["over_pre"])
            print("  il=%-3s %-10s prose %-28s code %-28s %s" % (il, name, fmt(p), fmt(c), "GO" if ok else "-"))
            out.append(dict(il=il, probe=name, prose=p, code=c, go=ok))
    go = any(x["go"] for x in out)
    print("  verdict: %s" % ("GO (implement the GPU path, then confirm with the fixed metric)" if go else
                             "NO-GO (record the numbers in DESIGN.md as refuted)"))
    return dict(go=go, rows=out)


# ------------------------------------------------------------------------------------------------ self-test

def write_synthetic(path, n_req, layers, V, W, hc, rng, kinds):
    """Synthetic dump in the exact on-disk format of dumppatch.py. The final argmax is the raw lens at the last
    layer; intermediate layers are the last-layer residual plus noise that shrinks with depth."""
    K = 8
    wn, wd, wu = hc
    with open(path, "wb") as f:
        f.write(b"RPD1"); f.write(struct.pack("<ii", 1, len(layers))); f.write(struct.pack("<%di" % len(layers), *layers))
        f.write(struct.pack("<ii", NE * HC, K))
        for q in range(n_req):
            acc = 0.9 if kinds[q] == "code" else 0.6
            for rnd in range(30):
                B = 8
                hl = rng.standard_normal((B, HC * NE)).astype(np.float32)
                with torch.no_grad():
                    lg = (hc_head_mix(torch.from_numpy(hl).to(DEV), wn, wd, wu).half() @ W.t()).float().cpu().numpy()
                am = lg.argmax(1)
                tok = np.zeros(B, np.int32); tok[0] = rng.randint(V)
                good = rng.rand(B) < acc
                for j in range(1, B):
                    if good[j]:
                        tok[j] = am[j - 1]
                    else:
                        t = rng.randint(V)
                        tok[j] = t if t != am[j - 1] else (t + 1) % V
                nacc = lead_matches(am, tok)
                lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
                srt = np.sort(lg, 1)
                lnext = np.array([lg[j, tok[j + 1]] if j + 1 < B else np.nan for j in range(B)], np.float32)
                cid = rng.randint(0, V, (B - 1, K)).astype(np.int32); cid[:, 0] = tok[1:]
                clg = np.sort(rng.randn(B - 1, K).astype(np.float32), 1)[:, ::-1].copy()
                clg[:, 0] += np.where(good[1:], 3.0, 0.5)       # the draft is more confident when it is right
                cp = np.exp(clg - clg.max(1, keepdims=True)); cp /= cp.sum(1, keepdims=True)
                f.write(b"RPR1"); f.write(struct.pack("<5i", B, nacc, q + 1, 100 + rnd * 8, 1))
                for arr, dt in ((tok, np.int32), (am, np.int32), (lse, np.float32), (srt[:, -1], np.float32),
                                (srt[:, -2], np.float32), (lnext, np.float32), (cid, np.int32), (clg, np.float32),
                                (cp.astype(np.float32), np.float32)):
                    f.write(np.ascontiguousarray(arr, dtype=dt).tobytes())
                for il in layers:
                    sig = 3.0 * (1.0 - (il + 1) / 48.0)
                    h = hl + sig * rng.standard_normal(hl.shape).astype(np.float32)
                    f.write(h.astype(np.float16).tobytes())


def selftest(a):
    rng = np.random.RandomState(0)
    torch.manual_seed(0)
    V = 2048
    tmp = tempfile.mkdtemp(prefix="rowprune_")
    W = (torch.randn(V, NE, device=DEV) / math.sqrt(NE) * 8).half()
    wn = (1 + 0.1 * torch.randn(HC, NE, device=DEV)); wd = torch.randn(320, HC * NE, device=DEV) / math.sqrt(HC * NE)
    wu = torch.randn(HC * NE, 320, device=DEV) / math.sqrt(320)
    np.save(os.path.join(tmp, "W.npy"), W.float().cpu().numpy())
    with open(os.path.join(tmp, "head.bin"), "wb") as o:
        o.write(b"RPHW"); o.write(struct.pack("<I", 3))
        for name, t in (("output_hc_norm.weight", wn.reshape(-1)), ("output_hc_down.weight", wd), ("output_hc_up.weight", wu)):
            arr = t.float().cpu().numpy()
            dims = tuple(reversed(arr.shape))
            nb = name.encode(); o.write(struct.pack("<I", len(nb))); o.write(nb)
            o.write(struct.pack("<II", 0, len(dims))); o.write(struct.pack("<" + "Q" * len(dims), *dims))
            raw = arr.astype(np.float32).tobytes(); o.write(struct.pack("<Q", len(raw))); o.write(raw)
    layers = [15, 31, 47]
    kinds_tr = ["code" if q % 2 else "prose" for q in range(16)]
    kinds_he = ["code" if q % 2 else "prose" for q in range(6)]
    write_synthetic(os.path.join(tmp, "train.bin"), 16, layers, V, W, (wn, wd, wu), rng, kinds_tr)
    write_synthetic(os.path.join(tmp, "held.bin"), 6, layers, V, W, (wn, wd, wu), rng, kinds_he)
    for nm, ks in (("train.jsonl", kinds_tr), ("held.jsonl", kinds_he)):
        with open(os.path.join(tmp, nm), "w") as o:
            for i, k in enumerate(ks):
                o.write(json.dumps({"i": i, "kind": k}) + "\n")
    a.dump = [os.path.join(tmp, "train.bin")]; a.sidecar = [os.path.join(tmp, "train.jsonl")]
    a.held = [os.path.join(tmp, "held.bin")]; a.held_sidecar = [os.path.join(tmp, "held.jsonl")]
    a.head = os.path.join(tmp, "head.bin"); a.mtp_gguf = os.path.join(tmp, "W.npy")
    a.tuned_steps = 40; a.epochs = 4; a.depths = "2,5,7"
    rep = run(a)
    last = rep["layers"][47]
    assert last["oracle"]["auc"] > 0.99
    assert rep["lens_agree"][47] > 0.95, rep["lens_agree"]
    assert rep["layers"][15]["oracle"]["depth"][7]["speedup"] > 0.0
    print("\nselftest ok (tmp %s)" % tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", default=[], help="training dumps (split by request into train/val[/test])")
    ap.add_argument("--sidecar", nargs="+", help="collect.py sidecars, one per --dump file (code/prose breakdown)")
    ap.add_argument("--held", nargs="+", help="held-out dumps: the test set (prompt families never used for training)")
    ap.add_argument("--held-sidecar", nargs="+")
    ap.add_argument("--head", help="extract_head.py output (target output_hc_* tensors)")
    ap.add_argument("--mtp-gguf", help="GGUF with output.weight (MTP GGUF: Q8_0 copy of the head) or a .npy")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--tuned-steps", type=int, default=0)
    ap.add_argument("--train-all", action="store_true", help="also train on rows after the first rejection")
    ap.add_argument("--depths", default="2,3,5,7", help="simulate these draft depths (records truncated, causality)")
    ap.add_argument("--fdr", default="0.01,0.03,0.05")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tv1", type=float, default=17.93)
    ap.add_argument("--tv2", type=float, default=23.0)
    ap.add_argument("--row-ms", type=float, default=2.82)
    ap.add_argument("--moe-row-ms", type=float, default=2.65)
    ap.add_argument("--draft-ms", type=float, default=1.18)
    ap.add_argument("--host-ms", type=float, default=0.5)
    ap.add_argument("--n-layer", type=int, default=48)
    ap.add_argument("--probe-ms", default="oracle=0,draft=0.01,linear_h=0.02,linear=0.02,lens_cand=0.10,tuned_cand=0.11,lens_full=1.25")
    ap.add_argument("--go-prose-depth", type=int, default=2, help="production depth of prose (depth controller)")
    ap.add_argument("--go-code-depth", type=int, default=5, help="production depth of code (depth controller)")
    ap.add_argument("--go-max-il", type=int, default=23, help="latest probe layer that still counts (il)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a)
    else:
        if not a.dump:
            ap.error("--dump is required")
        run(a)
