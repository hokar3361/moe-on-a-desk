#!/usr/bin/env python3
# Per-round draft depth from the verifier's own signal: offline analysis of LLAMA_SPEC_SIGNAL_LOG logs.
#
#   analyze.py selftest [--sweep]                 synthetic logs -> parse -> fit -> AUC -> simulate (runs anywhere)
#   analyze.py fit --train A.log [..] --test B.log [..] [--cap 7] [--tv 1:17.9,3:25.8,6:34.3,8:41]
#                  [--draft-ms 1.3] [--host-ms 1.0] [--catchup-ms 0.5] [--dmin 1]
#                                                 fit on --train, report AUC / calibration / simulated tok/s on --test
#   analyze.py summary X.log                      acceptance per position, single-feature AUCs, measured round times
#   analyze.py prompts GENDATA2.py OUT.json [--n 48] [--seed 1]
#                                                 training prompts (gendata2.build(), fixed-metric topics excluded)
#
# Log format: see sigpatch.py. Collect with a FIXED depth (no depth controller, LLAMA_SPEC_MTP_DEPTH=D, D = cap) so
# that every round records D drafts: with greedy drafts the accepted count at any depth d <= D is min(L, d), where L
# is the logged count (censored at D). Policies are then replayed on the same rounds (the round partition of the
# text is held fixed, the same approximation as the oracle). numpy only.
import sys, math, json, random, argparse, re
import numpy as np

# ----------------------------------------------------------------------------------------------------------------
# parsing

LINE = re.compile(r"^([BRDVA]) (.*)$")

def _kv(rest):
    d = {}
    for tok in rest.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d

class Round:
    __slots__ = ("req", "idx", "pos0", "pos_rel", "D", "p", "m", "lm", "L", "cens", "vlm", "vp1", "vH", "vam", "vtok",
                 "dt_us", "t_us", "dtok")

def parse(path, req_offset=0):
    """returns list of requests, each a list of Rounds (only rounds that reached accept())"""
    reqs = []
    cur = None                # list of rounds of the current request
    open_r = {}               # r -> partial dict
    first_pos = None
    nreq = req_offset
    with open(path, "r", errors="replace") as f:
        for line in f:
            mt = LINE.match(line.strip())
            if not mt:
                continue
            kind, kv = mt.group(1), _kv(mt.group(2))
            if kind == "B":
                cur = []
                reqs.append(cur)
                nreq += 1
                open_r = {}
                first_pos = None
            elif kind == "R":
                if cur is None:
                    cur = []; reqs.append(cur); nreq += 1
                r = int(kv["r"])
                pos0 = int(kv["pos0"])
                if first_pos is None:
                    first_pos = pos0
                open_r = {r: {"pos0": pos0, "depth": int(kv["depth"]), "t_us": int(kv["t_us"]), "d": {}, "v": {}}}
            elif kind == "D":
                o = open_r.get(int(kv["r"]))
                if o is not None:
                    o["d"][int(kv["k"])] = (float(kv["p"]), float(kv["m"]), float(kv["lm"]), int(kv["tok"]))
            elif kind == "V":
                o = open_r.get(int(kv["r"]))
                if o is not None:
                    o["v"][int(kv["row"])] = (int(kv["tok"]), int(kv["am"]), float(kv["lm"]), float(kv["p1"]), float(kv["H"]))
            elif kind == "A":
                r = int(kv["r"])
                o = open_r.pop(r, None)
                if o is None or not o["d"]:
                    continue
                D = max(o["d"].keys())
                if sorted(o["d"].keys()) != list(range(1, D + 1)):
                    continue
                rd = Round()
                rd.req = nreq; rd.idx = len(cur); rd.pos0 = o["pos0"]; rd.pos_rel = o["pos0"] - first_pos
                rd.D = D
                rd.p = np.array([o["d"][k][0] for k in range(1, D + 1)])
                rd.m = np.array([o["d"][k][1] for k in range(1, D + 1)])
                rd.lm = np.array([min(o["d"][k][2], 30.0) for k in range(1, D + 1)])
                rd.dtok = [o["d"][k][3] for k in range(1, D + 1)]
                rd.L = min(int(kv["nacc"]), D)
                rd.cens = rd.L >= D
                n = D + 1
                rd.vlm = np.full(n, np.nan); rd.vp1 = np.full(n, np.nan); rd.vH = np.full(n, np.nan)
                rd.vam = [-1]*n; rd.vtok = [-1]*n
                for row, (tok, am, lm, p1, H) in o["v"].items():
                    if 0 <= row < n:
                        rd.vtok[row] = tok; rd.vam[row] = am
                        rd.vlm[row] = min(lm, 30.0)
                        rd.vp1[row] = p1 if p1 >= 0 else np.nan
                        rd.vH[row] = H if H >= 0 else np.nan
                rd.dt_us = int(kv.get("dt_us", 0)); rd.t_us = o["t_us"]
                cur.append(rd)
    return [q for q in reqs if q]

def check_consistency(reqs):
    """accept counts recomputed from the verify rows (target argmax at row k-1 == draft k) vs the logged nacc"""
    n = bad = 0
    for q in reqs:
        for rd in q:
            if rd.vam[0] < 0:
                continue
            L = 0
            while L < rd.D and rd.vam[L] == rd.dtok[L]:
                L += 1
            n += 1
            bad += L != rd.L
    return n, bad

# ----------------------------------------------------------------------------------------------------------------
# cost model

def make_tv(spec):
    pts = sorted((int(a), float(b)) for a, b in (x.split(":") for x in spec.split(",")))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    def tv(B):
        if B <= xs[0]:
            return ys[0]
        if B >= xs[-1]:
            slope = (ys[-1] - ys[-2])/(xs[-1] - xs[-2])
            return ys[-1] + slope*(B - xs[-1])
        return float(np.interp(B, xs, ys))
    return tv

class Cost:
    def __init__(self, tv, draft_ms, host_ms, catchup_ms):
        self.tv, self.c_d, self.host, self.catch = tv, draft_ms, host_ms, catchup_ms
    def round(self, depth, steps):
        # verify of depth+1 rows + draft steps + host; a round without drafts still owes the MTP catch-up
        return self.tv(depth + 1) + self.c_d*steps + self.host + (self.catch if steps == 0 else 0.0)
    def dv(self, k):
        # marginal verify cost of including draft k (width k -> k+1)
        return self.tv(k + 1) - self.tv(k)

# ----------------------------------------------------------------------------------------------------------------
# features: sample = (round, position k), label = draft k accepted given 1..k-1 accepted

def _lg(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p/(1 - p))

class Prev:
    """what the controller knows about the previous round of the same request, under the depth it actually used"""
    __slots__ = ("first", "n", "rej", "tlm", "tlm_mean", "tH", "tp1", "has_H")
    def __init__(self, rd=None, depth=None):
        self.first = rd is None
        if rd is None:
            self.n = 0; self.rej = 0; self.tlm = 0.0; self.tlm_mean = 0.0; self.tH = 0.0; self.tp1 = 0.0; self.has_H = 0
            return
        n = min(rd.L, depth)
        self.n = n
        self.rej = 1 if rd.L < depth else 0
        row = n                                   # the row whose argmax is the next round's id_last
        lm = rd.vlm[row] if row < len(rd.vlm) else np.nan
        self.tlm = 0.0 if np.isnan(lm) else float(lm)
        seg = rd.vlm[:row + 1]
        seg = seg[~np.isnan(seg)]
        self.tlm_mean = float(seg.mean()) if len(seg) else 0.0
        H = rd.vH[row] if row < len(rd.vH) else np.nan
        p1 = rd.vp1[row] if row < len(rd.vp1) else np.nan
        self.has_H = 0 if np.isnan(H) else 1
        self.tH = 0.0 if np.isnan(H) else float(H)
        self.tp1 = 0.0 if np.isnan(p1) else float(p1)

KMAX_ONEHOT = 4
FEATURE_SETS = {
    # S0: what the current controller uses (position + per-position EMA)
    "S0_pos+ema":        dict(mtp=False, ver=False),
    # S1: + MTP draft probability of this and earlier drafts (the EV planner's information)
    "S1_+mtp":           dict(mtp=True,  ver=False),
    # S3: + verifier signal of the previous round, without MTP probabilities
    "S3_+verifier":      dict(mtp=False, ver=True),
    # S2: everything
    "S2_+mtp+verifier":  dict(mtp=True,  ver=True),
}

def featurize(rd, k, prev, ema_k, fs, post=True):
    """k is 1-based. post=True: p_k is known (decision to include draft k); post=False: before drafting step k"""
    x = [1.0 if min(k, KMAX_ONEHOT) == j else 0.0 for j in range(1, KMAX_ONEHOT + 1)]
    x.append(_lg(ema_k))
    x.append(math.log1p(rd.pos_rel)/6.0)
    if fs["mtp"]:
        if post:
            x += [_lg(rd.p[k - 1]), math.log1p(max(rd.lm[k - 1], 0.0))]
        else:
            x += [_lg(rd.p[k - 2]) if k >= 2 else 0.0, math.log1p(max(rd.lm[k - 2], 0.0)) if k >= 2 else 0.0]
        x.append(min(_lg(rd.p[j]) for j in range(k - 1)) if k >= 2 else 0.0)   # weakest earlier draft
    if fs["ver"]:
        x += [float(prev.first), math.log1p(prev.n), float(prev.rej), math.log1p(max(prev.tlm, 0.0)),
              math.log1p(max(prev.tlm_mean, 0.0)), prev.tH*prev.has_H, float(prev.has_H)]
    return x

def build_samples(reqs, fs, depth_collect=None, post=True):
    """samples from collection logs: the previous round's features use the depth the collection actually drafted"""
    X, y, kk = [], [], []
    for q in reqs:
        ema = [0.7]*17
        prev = Prev()
        for rd in q:
            tried = min(rd.D, rd.L + 1)
            for k in range(1, tried + 1):
                X.append(featurize(rd, k, prev, ema[min(k, 16)], fs, post))
                y.append(1.0 if k <= rd.L else 0.0)
                kk.append(k)
            for k in range(1, tried + 1):
                ema[k] = 0.9*ema[k] + 0.1*(1.0 if k <= rd.L else 0.0)
            prev = Prev(rd, rd.D)
    return np.array(X, dtype=float), np.array(y), np.array(kk)

# ----------------------------------------------------------------------------------------------------------------
# logistic regression (Newton, L2) and metrics

class Logit:
    def fit(self, X, y, l2=1e-2, iters=30):
        self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd < 1e-9] = 1.0
        Z = np.hstack([(X - self.mu)/self.sd, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        R = l2*np.eye(Z.shape[1]); R[-1, -1] = 0.0
        for _ in range(iters):
            p = 1/(1 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = Z.T @ (p - y) + R @ w
            H = (Z*(p*(1 - p))[:, None]).T @ Z + R + 1e-9*np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-7:
                break
        self.w = w
        return self
    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Z = np.hstack([(X - self.mu)/self.sd, np.ones((len(X), 1))])
        return 1/(1 + np.exp(-np.clip(Z @ self.w, -30, 30)))
    def p1(self, x):
        z = (np.asarray(x, dtype=float) - self.mu)/self.sd
        return float(1/(1 + math.exp(-max(-30.0, min(30.0, float(z @ self.w[:-1] + self.w[-1]))))))

def auc(s, y):
    s = np.asarray(s, dtype=float); y = np.asarray(y)
    pos = y > 0.5; npos = pos.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2)/2.0
        i = j + 1
    return float((ranks[pos].sum() - npos*(npos + 1)/2)/(npos*nneg))

def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y))**2))

def reliability(p, y, nb=5):
    edges = np.quantile(p, np.linspace(0, 1, nb + 1))
    out = []
    for i in range(nb):
        m = (p >= edges[i]) & (p <= edges[i + 1]) if i == nb - 1 else (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out

# ----------------------------------------------------------------------------------------------------------------
# policies (replayed on the logged rounds)

def run_policy(reqs, cost, cap, policy):
    """policy.round(rd, state) -> (depth, steps); policy.observe(rd, depth, steps). returns tokens, ms, rounds, depth sum"""
    tok = ms = 0.0; nr = 0; dsum = 0
    for q in reqs:
        policy.begin()
        for rd in q:
            d, steps = policy.round(rd)
            d = min(d, cap, rd.D)
            steps = min(steps, rd.D)
            n_acc = min(rd.L, d)
            t = cost.round(d, steps)
            tok += n_acc + 1; ms += t; nr += 1; dsum += d
            policy.observe(rd, d, steps, n_acc, t)
    return tok, ms, nr, dsum

class Fixed:
    def __init__(self, d): self.d = d
    def begin(self): pass
    def round(self, rd): return self.d, self.d
    def observe(self, *a): pass

class Oracle:
    """depth = the number of drafts that will be accepted (knows the outcome)"""
    def __init__(self, cap, dmin): self.cap, self.dmin = cap, dmin
    def begin(self): pass
    def round(self, rd):
        d = max(self.dmin, min(rd.L, self.cap))
        return d, d
    def observe(self, *a): pass

class DepthCtl:
    """port of the fork's depth_ctl, mode 1, LLAMA_SPEC_DEPTH_FIX on, FASTX on, RESET per request (run-best.sh)"""
    V = [0.0, 18.5, 23.0, 26.6, 29.3, 32.0, 35.0, 38.0, 41.0]
    def __init__(self, cap): self.cap = min(16, cap)
    @classmethod
    def prior(cls, d):
        w = d + 1
        tv = cls.V[w] if w <= 8 else 41.0 + 3.0*(w - 8)
        return tv + 1.3*d + 0.8
    def T(self, d):
        m = 0.0
        for k in range(1, d + 1):
            raw = self.t[k] if self.t[k] > 0 else self.prior(k)
            m = max(raw, m + 1.0)
        return m
    def E(self, d):
        e, P = 1.0, 1.0
        for k in range(1, d + 1):
            P *= self.a[k]; e += P
        return e
    def best(self):
        bd, br = 1, 0.0
        for d in range(1, self.cap + 1):
            r = self.E(d)/self.T(d)
            if r > br:
                br, bd = r, d
        return bd
    def begin(self):
        self.a = [0.7]*17; self.t = [0.0]*17; self.stale = [True]*17
        self.depth = min(self.cap, 3); self.rounds_at = 0; self.last_depth = 0; self.last_ms = None
    def round(self, rd):
        if self.last_depth > 0 and self.last_ms is not None:
            ms, ld = self.last_ms, self.last_depth
            td = self.t[ld]
            prior = self.T(ld)
            plausible = td > 0 or (0.6*self.prior(ld) < ms < 1.6*self.prior(ld))
            if plausible and ms < 2.0*prior and ms < 150.0:
                self.t[ld] = 0.8*td + 0.2*ms if (td > 0 and not self.stale[ld]) else ms
                self.stale[ld] = False
        d = self.best()
        if d != self.depth and self.E(d)/self.T(d) < 1.03*self.E(self.depth)/self.T(self.depth):
            d = self.depth
        if d != self.depth:
            self.depth = d; self.rounds_at = 0
        ad = self.a[self.depth]
        period = 2 if ad >= 0.80 else 8 if ad >= 0.65 else 32
        self.rounds_at += 1
        use = self.depth + 1 if (self.rounds_at % period == 0 and self.depth < self.cap) else self.depth
        self.last_depth = use
        return use, use
    def observe(self, rd, d, steps, n_acc, t):
        self.last_ms = t
        tried = min(d, n_acc + 1)
        for k in range(1, tried + 1):
            self.a[k] = 0.9*self.a[k] + 0.1*(1.0 if k <= n_acc else 0.0)

class DepthCtlFirstSkip(DepthCtl):
    """the first round of a request is never measured (spans prompt processing)"""
    def begin(self):
        super().begin(); self.first = True
    def observe(self, rd, d, steps, n_acc, t):
        tried = min(d, n_acc + 1)
        for k in range(1, tried + 1):
            self.a[k] = 0.9*self.a[k] + 0.1*(1.0 if k <= n_acc else 0.0)
        self.last_ms = None if self.first else t
        self.first = False

class EVPlanner:
    """port of patch 0023 (LLAMA_SPEC_EV=1): bucket-calibrated MTP p, myopic per-round ratio, global calibration"""
    def __init__(self, cap, cost_table=None, draft_ms=1.5):
        self.cap = cap
        self.cost = cost_table or [0.0, 21.0, 28.8, 31.6, 34.3, 37.2, 40.1, 43.0, 46.0]
        self.draft_ms = draft_ms
        self.n = np.full(20, 4.0); self.k = np.array([4.0*(b + 0.5)/20 for b in range(20)])
    def T(self, w):
        c = self.cost
        if w < len(c):
            return c[w]
        return c[-1] + (c[-1] - c[-2])*(w - len(c) + 1)
    def rate(self, E, k, s): return E/(self.T(k + 1) + self.draft_ms*s)
    def begin(self): pass
    def round(self, rd):
        P, E, k = 1.0, 1.0, 0
        steps = 0
        self.lp = []
        for i in range(min(self.cap, rd.D)):
            steps = i + 1
            b = min(19, max(0, int(rd.p[i]*20)))
            a = self.k[b]/self.n[b]
            P1 = P*a; E1 = E + P1
            if self.rate(E1, k + 1, steps) <= self.rate(E, k, steps):
                break
            P, E, k = P1, E1, k + 1
            self.lp.append(rd.p[i])
            if self.rate(E1 + P1, k + 1, steps + 1) <= self.rate(E1, k, steps):
                break
        return k, steps
    def observe(self, rd, d, steps, n_acc, t):
        for j, p in enumerate(self.lp):
            if j > n_acc:
                break
            b = min(19, max(0, int(p*20)))
            self.n[b] += 1
            if j < n_acc:
                self.k[b] += 1

class OptionTable:
    """distribution of the post-draft estimate q_post given the pre-draft estimate q_pre (training data): drafting one
    more step buys the information p_k, so its value is E[max(0, value after seeing p_k)] - lam*draft_ms"""
    def __init__(self, q_pre, q_post, nb=10, nq=8):
        self.edges = np.quantile(q_pre, np.linspace(0, 1, nb + 1))[1:-1]
        idx = np.searchsorted(self.edges, q_pre)
        lv = (np.arange(nq) + 0.5)/nq
        self.draws = []
        for b in range(nb):
            v = q_post[idx == b]
            self.draws.append(np.quantile(v, lv) if len(v) else np.array([float(q_pre[idx == b].mean()) if (idx == b).any() else 0.5]))
    def get(self, q):
        return self.draws[int(np.searchsorted(self.edges, q))]

class SignalStop:
    """incremental stop rule. Before draft step k: run it if the expected value of the decision that follows
    (option value over the q_post spread for this q_pre) exceeds lam*draft_ms. After draft step k (p_k known):
    keep draft k if the look-ahead value of including it is positive. value = expected accepted tokens - lam * ms.
    lam = long-run tok/ms (fixed: Dinkelbach iterations on the data; online: EMA of the realized rate per request)."""
    def __init__(self, cap, cost, m_pre, m_post, fs, qbar, lam, online=False, dmin=0, lam0=0.06, opt=None):
        self.cap, self.cost, self.pre, self.post, self.fs, self.qbar = cap, cost, m_pre, m_post, fs, qbar
        self.lam_fixed, self.online, self.dmin, self.lam0, self.opt = lam, online, dmin, lam0, opt
    def begin(self):
        self.ema = [0.7]*17; self.prev = Prev(); self.prev_rd = None
        self.lam = self.lam0 if self.online else self.lam_fixed
    def _ahead(self, P, k, q_first, sunk_first):
        """best value of including drafts k..k+m-1 (m >= 1); q_first = P(draft k ok | 1..k-1 ok)"""
        best, v, Pc = -1e9, 0.0, P
        for j in range(k, self.cap + 1):
            q = q_first if j == k else self.qbar[min(j, len(self.qbar) - 1)]
            Pc *= q
            c = self.cost.dv(j) + (0.0 if (j == k and sunk_first) else self.cost.c_d)
            v += Pc - self.lam*c
            best = max(best, v)
        return best
    def round(self, rd):
        P, depth, steps = 1.0, 0, 0
        for k in range(1, min(self.cap, rd.D) + 1):
            if k > self.dmin:
                q_pre = self.pre.p1(featurize(rd, k, self.prev, self.ema[min(k, 16)], self.fs, post=False))
                if self.opt is None:
                    v = self._ahead(P, k, q_pre, False)
                else:
                    v = float(np.mean([max(0.0, self._ahead(P, k, q, True)) for q in self.opt.get(q_pre)])) - self.lam*self.cost.c_d
                if v <= 0.0:
                    break
            steps = k
            q_post = self.post.p1(featurize(rd, k, self.prev, self.ema[min(k, 16)], self.fs, post=True))
            if k > self.dmin and self._ahead(P, k, q_post, True) <= 0.0:
                break
            P *= q_post
            depth = k
        return depth, steps
    def observe(self, rd, d, steps, n_acc, t):
        tried = min(d, n_acc + 1)
        for k in range(1, tried + 1):
            self.ema[k] = 0.9*self.ema[k] + 0.1*(1.0 if k <= n_acc else 0.0)
        self.prev = Prev(rd, d)
        if self.online:
            self.lam = 0.9*self.lam + 0.1*((n_acc + 1)/t)

# ----------------------------------------------------------------------------------------------------------------
# fit + evaluate

def qbar_from(reqs, cap):
    s = np.zeros(cap + 2); n = np.zeros(cap + 2)
    for q in reqs:
        for rd in q:
            for k in range(1, min(rd.D, rd.L + 1) + 1):
                n[k] += 1; s[k] += k <= rd.L
    qb = [0.7] + [float((s[k] + 1)/(n[k] + 2)) for k in range(1, cap + 2)]
    return qb

def evaluate(train, test, args, quiet=False):
    tv = make_tv(args.tv)
    cost = Cost(tv, args.draft_ms, args.host_ms, args.catchup_ms)
    cap = args.cap
    out = {}
    pr = (lambda *a: None) if quiet else print
    ntr = sum(len(q) for q in train); nte = sum(len(q) for q in test)
    pr("rounds: train %d (%d requests), test %d (%d requests)" % (ntr, len(train), nte, len(test)))
    for name, reqs in (("train", train), ("test", test)):
        n, bad = check_consistency(reqs)
        if n:
            pr("  %s: accept count re-derived from verify rows: %d / %d rounds disagree" % (name, bad, n))
    # acceptance per position (test)
    qb = qbar_from(test, cap)
    pr("test acceptance P(k ok | 1..k-1 ok): " + " ".join("%d:%.2f" % (k, qb[k]) for k in range(1, cap + 1)))

    # ---- AUC per feature set
    models = {}
    pr("\nAUC for 'draft k accepted | 1..k-1 accepted' (fit on train, scored on test)")
    pr("  %-18s %6s %6s %6s %6s %6s  %s" % ("features", "all", "k=1", "k=2", "k=3", "k>=4", "brier"))
    for fname, fs in FEATURE_SETS.items():
        Xtr, ytr, _ = build_samples(train, fs)
        Xte, yte, kte = build_samples(test, fs)
        m = Logit().fit(Xtr, ytr)
        p = m.predict(Xte)
        row = [auc(p, yte)]
        for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 99)):
            sel = (kte >= lo) & (kte <= hi)
            row.append(auc(p[sel], yte[sel]) if sel.sum() > 10 else float("nan"))
        pr("  %-18s " % fname + " ".join("%6.3f" % v for v in row) + "  %.4f" % brier(p, yte))
        Xpre, ypre, _ = build_samples(train, fs, post=False)
        mpre = Logit().fit(Xpre, ypre)
        models[fname] = (fs, m, mpre, OptionTable(mpre.predict(Xpre), m.predict(Xtr)))
        out["auc_" + fname] = row[0]
        if fname.startswith("S2"):
            out["auc_k1"] = row[1]
            pr("    reliability S2 (pred, observed, n): " +
               " ".join("(%.2f,%.2f,%d)" % t for t in reliability(p, yte)))

    # ---- simulation
    pr("\nsimulated decode on the test rounds (T(B) %s, draft %.2f ms/step, host %.2f ms/round)" %
       (args.tv, args.draft_ms, args.host_ms))
    res = []
    def sim(name, pol):
        tok, ms, nr, dsum = run_policy(test, cost, cap, pol)
        rate = 1000.0*tok/ms
        res.append((name, rate, tok/nr, dsum/nr))
        return rate
    fixed = {d: sim("fixed d=%d" % d, Fixed(d)) for d in range(1, cap + 1)}
    best_fixed = max(fixed.values())
    r_ctl = sim("depth_ctl (current)", DepthCtlFirstSkip(cap))
    sim("EV planner 0023 (old costs)", EVPlanner(cap))
    sim("EV planner 0023 (current costs)", EVPlanner(cap, [0.0] + [tv(b) + args.host_ms for b in range(1, 17)], args.draft_ms))
    for fname in ("S1_+mtp", "S2_+mtp+verifier"):
        fs, mpost, mpre, opt = models[fname]
        qb_tr = qbar_from(train, cap)
        lam = best_fixed/1000.0
        for _ in range(6):     # Dinkelbach: lam <- achieved rate
            tok, ms, nr, dsum = run_policy(test, cost, cap, SignalStop(cap, cost, mpre, mpost, fs, qb_tr, lam, dmin=args.dmin, opt=opt))
            lam = tok/ms
        tag = fname[:2]
        out["stop_" + tag] = sim("stop rule %s (lam*)" % tag, SignalStop(cap, cost, mpre, mpost, fs, qb_tr, lam, dmin=args.dmin, opt=opt))
        out["stop_online_" + tag] = sim("stop rule %s (online lam)" % tag, SignalStop(cap, cost, mpre, mpost, fs, qb_tr, lam, online=True, dmin=args.dmin, lam0=best_fixed/1000.0, opt=opt))
        out["stop_myopic_" + tag] = sim("stop rule %s (no option value)" % tag, SignalStop(cap, cost, mpre, mpost, fs, qb_tr, lam, dmin=args.dmin))
        if args.dmin > 0:
            out["stop_d0_" + tag] = sim("stop rule %s (lam*, may skip drafts)" % tag, SignalStop(cap, cost, mpre, mpost, fs, qb_tr, lam, dmin=0, opt=opt))
    r_or1 = sim("oracle (min depth 1)", Oracle(cap, 1))
    r_or0 = sim("oracle (min depth 0)", Oracle(cap, 0))
    pr("  %-34s %8s %8s %8s %9s %9s" % ("policy", "tok/s", "tok/rnd", "depth", "vs ctl", "of gap"))
    for name, rate, tpr, dpr in res:
        gap = (rate - r_ctl)/(r_or1 - r_ctl) if r_or1 > r_ctl else float("nan")
        pr("  %-34s %8.1f %8.2f %8.2f %+8.1f%% %8.0f%%" % (name, rate, tpr, dpr, 100*(rate/r_ctl - 1), 100*gap))
    out.update(dict(ctl=r_ctl, oracle1=r_or1, oracle0=r_or0, best_fixed=best_fixed))
    if getattr(args, "export", None):
        export_model(args.export, models["S2_+mtp+verifier"], qbar_from(train, cap), cost, cap)
        pr("exported S2 models (pre, post, option table, qbar, cost) -> %s" % args.export)
    # measured round times in the logs (collection run, perturbed by the log itself)
    return out

def export_model(path, model, qbar, cost, cap):
    """plain text for a C++ loader: features are standardized (x - mu)/sd, then p = sigmoid(w . z + b);
    feature order = featurize() with the S2 set (see design.md)"""
    fs, mpost, mpre, opt = model
    with open(path, "w") as f:
        for name, m in (("post", mpost), ("pre", mpre)):
            f.write("%s %d\n" % (name, len(m.mu)))
            f.write("mu " + " ".join("%.8g" % v for v in m.mu) + "\n")
            f.write("sd " + " ".join("%.8g" % v for v in m.sd) + "\n")
            f.write("w " + " ".join("%.8g" % v for v in m.w) + "\n")   # last = bias
        f.write("opt_edges " + " ".join("%.6g" % v for v in opt.edges) + "\n")
        for b, d in enumerate(opt.draws):
            f.write("opt_draws %d " % b + " ".join("%.6g" % v for v in d) + "\n")
        f.write("qbar " + " ".join("%.6g" % v for v in qbar[:cap + 2]) + "\n")
        f.write("tv " + " ".join("%.4g" % cost.tv(b) for b in range(1, cap + 2)) + "\n")
        f.write("draft_ms %.4g host_ms %.4g\n" % (cost.c_d, cost.host))

def measured_round_times(reqs):
    by = {}
    for q in reqs:
        for a, b in zip(q, q[1:]):
            dt = (b.t_us - a.t_us)/1000.0
            if 0 < dt < 150:
                by.setdefault(a.D, []).append(dt)
    return {d: (float(np.median(v)), len(v)) for d, v in sorted(by.items())}

# ----------------------------------------------------------------------------------------------------------------
# synthetic logs (self-test)

def synth_log(path, n_req, rounds_per_req, D, seed, signal=1.0, kinds=("prose", "code")):
    """latent local difficulty z (AR(1) over rounds) drives acceptance at every position; the verifier's margin at the
    row that starts the next round and the MTP probabilities are noisy views of it. signal scales how informative
    the views are (0 = no information beyond position)."""
    rng = np.random.default_rng(seed)
    base = {"prose": [0.0, 0.66, 0.46, 0.44, 0.43, 0.42, 0.42, 0.42, 0.42],
            "code":  [0.0, 0.95, 0.80, 0.74, 0.70, 0.68, 0.66, 0.66, 0.66]}
    r_id = 0
    with open(path, "w") as f:
        for q in range(n_req):
            kind = kinds[q % len(kinds)]
            f.write("B seq=0 req=%d n_prompt=%d\n" % (q + 1, 40))
            z = rng.normal()
            pos = 40; t = 0
            for r in range(rounds_per_req):
                z = 0.8*z + 0.6*rng.normal()
                r_id += 1
                f.write("R seq=0 r=%d pos0=%d depth=%d t_us=%d\n" % (r_id, pos, D, t))
                # acceptance chain
                L = 0; alive = True
                ys = []
                for k in range(1, D + 1):
                    b = base[kind][min(k, 8)]
                    a = 1/(1 + math.exp(-(math.log(b/(1 - b)) + 0.6*z)))
                    y = alive and (rng.random() < a)
                    ys.append(y)
                    if alive and y:
                        L += 1
                    else:
                        alive = False
                # MTP probabilities: informative about its own token's acceptance, and about z
                toks = []
                for k in range(1, D + 1):
                    ok = ys[k - 1]
                    lg = 0.8 + signal*(0.55*(1.0 if ok else -1.0) + 0.3*z) + rng.normal()*1.2
                    p = 1/(1 + math.exp(-lg))
                    p2 = (1 - p)*rng.uniform(0.3, 0.9)
                    lm = max(0.01, math.log(p/max(p2, 1e-6)))
                    tok = int(rng.integers(1000, 200000)); toks.append(tok)
                    f.write("D seq=0 r=%d k=%d tok=%d p=%.5f m=%.5f lm=%.4f\n" % (r_id, k, tok, p, p - p2, lm))
                # verify rows: argmax equals the draft while accepted; the margin carries next-round difficulty
                znext_hint = 0.8*z
                for row in range(D + 1):
                    tok_in = 7 if row == 0 else toks[row - 1]
                    if row < D:
                        am = toks[row] if row < L else int(rng.integers(1000, 200000))
                    else:
                        am = int(rng.integers(1000, 200000))
                    lm_t = max(0.0, 2.0 + signal*1.0*znext_hint + rng.normal()*1.0)
                    p1 = 1/(1 + math.exp(-(lm_t - 0.5)))
                    H = max(0.0, 1.5 - 0.5*lm_t + 0.2*rng.normal())
                    f.write("V seq=0 r=%d row=%d tok=%d am=%d lm=%.4f p1=%.5f H=%.4f\n" % (r_id, row, tok_in, am, lm_t, p1, H))
                f.write("A seq=0 r=%d nacc=%d dt_us=%d dctl=-1\n" % (r_id, L, 40000))
                pos += L + 1; t += 40000

def selftest(args):
    import tempfile, os
    d = tempfile.mkdtemp(prefix="depthsignal-")
    ok = True
    sweep = [0.0, 0.5, 1.0, 1.5] if args.sweep else [1.0]
    rows = []
    for sig in sweep:
        for kind in (("prose",), ("code",)):
            tr = os.path.join(d, "train-%s-%.1f.log" % (kind[0], sig)); te = os.path.join(d, "test-%s-%.1f.log" % (kind[0], sig))
            synth_log(tr, 30, 120, args.cap, 1, sig, kind)
            synth_log(te, 15, 120, args.cap, 2, sig, kind)
            train, test = parse(tr), parse(te)
            assert sum(len(q) for q in train) == 30*120 and sum(len(q) for q in test) == 15*120, "parse lost rounds"
            n, bad = check_consistency(test)
            assert bad == 0, "verify-row consistency failed on synthetic data"
            quiet = args.sweep
            if not quiet:
                print("=== synthetic %s, signal %.1f ===" % (kind[0], sig))
            out = evaluate(train, test, args, quiet=quiet)
            rows.append((kind[0], sig, out))
            if out["oracle1"] + 1e-6 < max(out["ctl"], out["stop_S2"]):
                ok = False
                print("FAIL: a causal policy beat the oracle")
            if not quiet:
                print()
    if args.sweep:
        print("signal sweep (synthetic): AUC of the S2 model vs fraction of the oracle gap over depth_ctl")
        print("  %-6s %6s %7s %7s %7s %8s %8s %8s %8s" % ("text", "signal", "AUC S0", "AUC S1", "AUC S2", "ctl", "stop S2", "online", "oracle"))
        for kind, sig, o in rows:
            print("  %-6s %6.1f %7.3f %7.3f %7.3f %8.1f %8.1f %8.1f %8.1f   stop = %+.1f%% vs ctl, %.0f%% of gap" % (
                kind, sig, o["auc_S0_pos+ema"], o["auc_S1_+mtp"], o["auc_S2_+mtp+verifier"], o["ctl"], o["stop_S2"],
                o["stop_online_S2"], o["oracle1"], 100*(o["stop_S2"]/o["ctl"] - 1),
                100*(o["stop_S2"] - o["ctl"])/max(1e-9, o["oracle1"] - o["ctl"])))
    print("selftest %s (logs in %s)" % ("OK" if ok else "FAILED", d))
    return 0 if ok else 1

# ----------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    def common(p):
        p.add_argument("--cap", type=int, default=7)
        p.add_argument("--tv", default="1:17.9,3:25.8,6:34.3,8:41")
        p.add_argument("--draft-ms", type=float, default=1.3)
        p.add_argument("--host-ms", type=float, default=1.0)
        p.add_argument("--catchup-ms", type=float, default=0.5)
        p.add_argument("--dmin", type=int, default=1, help="minimum depth of the stop rule (0 allows no drafts)")
    p = sub.add_parser("selftest"); common(p); p.add_argument("--sweep", action="store_true")
    p = sub.add_parser("fit"); common(p); p.add_argument("--train", nargs="+", required=True); p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--export", help="write the S2 models for a C++ implementation")
    p = sub.add_parser("summary"); common(p); p.add_argument("log")
    p = sub.add_parser("prompts"); p.add_argument("gendata2"); p.add_argument("out"); p.add_argument("--n", type=int, default=48); p.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest(args)
    if args.cmd == "fit":
        train = [q for f in args.train for q in parse(f)]
        test = [q for f in args.test for q in parse(f)]
        Ds = sorted({rd.D for q in train + test for rd in q})
        print("draft depths in the logs:", Ds, "(collect at a fixed depth = cap; rounds with D < cap are censored early)")
        evaluate(train, test, args)
        return 0
    if args.cmd == "summary":
        reqs = parse(args.log)
        n, bad = check_consistency(reqs)
        print("requests %d, rounds %d, verify-row consistency %d/%d disagree" % (len(reqs), sum(len(q) for q in reqs), bad, n))
        qb = qbar_from(reqs, args.cap)
        print("P(k ok | 1..k-1 ok): " + " ".join("%d:%.2f" % (k, qb[k]) for k in range(1, args.cap + 1)))
        X, y, kk = build_samples(reqs, FEATURE_SETS["S2_+mtp+verifier"])
        names = ["k1", "k2", "k3", "k4+", "ema", "pos", "lg p_k", "log1p lm_k", "min lg p_<k",
                 "first", "log1p n_prev", "rej_prev", "log1p tlm", "log1p tlm_mean", "tH", "has_H"]
        print("single-feature AUC (sign-free, in-sample):")
        for j, nm in enumerate(names):
            a = auc(X[:, j], y)
            print("  %-16s %.3f" % (nm, max(a, 1 - a) if not math.isnan(a) else float("nan")))
        print("median round wall ms by drafted depth (log-perturbed):", measured_round_times(reqs))
        return 0
    if args.cmd == "prompts":
        import importlib.util
        spec = importlib.util.spec_from_file_location("gendata2", args.gendata2)
        g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
        train, held = g.build()
        rnd = random.Random(args.seed)
        code = [p for p in train if p.startswith("Write a ") and " function " in p]
        other = [p for p in train if p not in code]
        n_code = args.n//3
        sel = rnd.sample(code, min(n_code, len(code))) + rnd.sample(other, min(args.n - n_code, len(other)))
        json.dump(sel, open(args.out, "w"), indent=1)
        print("%d training prompts (%d code) -> %s; held-out prompts are tools/mtp/heldout.json" % (len(sel), n_code, args.out))
        return 0
    ap.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())
