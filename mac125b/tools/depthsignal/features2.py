#!/usr/bin/env python3
# Additional draft-time features for the per-draft stop rule (S1_+mtp), offline only.
#
#   python features2.py TRAIN.log HELD.log [--export NAME]
#
# Fits on TRAIN only, scores AUC per position on HELD, replays the stop rule with analyze.py's simulator and cost model
# (same T(B), draft and host ms, Dinkelbach lambda, option table). Every feature uses only what is known when draft k is
# decided: the committed text of this request (reconstructed from the log: row-0 token + accepted drafts of each round;
# prompt tokens are NOT logged), drafts 1..k-1 (and k for the post decision) with their MTP p / logit margin, and the
# outcomes of earlier rounds under the depth the replayed policy actually used.
import sys, os, math, argparse, importlib.util
from collections import defaultdict, Counter
import numpy as np

_spec = importlib.util.spec_from_file_location("analyze", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py"))
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)

NMAX = 8          # longest n-gram looked up
S1 = A.FEATURE_SETS["S1_+mtp"]

# ----------------------------------------------------------------------------------------------------------------
# text-only (policy independent) features: n-gram copy signal from the committed text of the request

class NgramIndex:
    """n-gram (n = 1..NMAX) -> Counter of next tokens, over the committed text; mirrors a C++ suffix hash table"""
    def __init__(self):
        self.t = []
        self.d = [None] + [defaultdict(Counter) for _ in range(NMAX)]
        self.last = [None] + [dict() for _ in range(NMAX)]     # most recent continuation
    def push(self, tok):
        t = self.t
        t.append(tok)
        j = len(t) - 1                    # new token at j: it continues the n-grams ending at j-1
        for n in range(1, NMAX + 1):
            if j - n < 0:
                break
            key = tuple(t[j - n:j])
            self.d[n][key][tok] += 1
            self.last[n][key] = tok
    def match(self, ctx_tail):
        """longest n such that the last n tokens of (text + ctx_tail) occurred before with a continuation.
        returns (n, most recent continuation, count of that n-gram, count of that continuation)"""
        full = self.t[-NMAX:] + list(ctx_tail)
        for n in range(min(NMAX, len(full)), 0, -1):
            key = tuple(full[-n:])
            c = self.d[n].get(key)
            if c:
                pred = self.last[n][key]
                tot = sum(c.values())
                return n, pred, tot, c
        return 0, -1, 0, None

def text_features(reqs):
    """per (req, idx): list over k = 1..D of dicts with the n-gram features"""
    ext = {}
    for q in reqs:
        ix = NgramIndex()
        for rd in q:
            ix.push(rd.vtok[0])           # id_last: committed before this round's drafts
            rows = []
            for k in range(1, rd.D + 1):
                tail = rd.dtok[:k - 1]
                n, pred, tot, c = ix.match(tail)
                dk = rd.dtok[k - 1]
                f = {}
                f["ng_n"] = n                                       # pre: context match length
                f["ng_has"] = 1.0 if n >= 2 else 0.0
                f["ng_hit"] = 1.0 if (n >= 1 and pred == dk) else 0.0
                f["ng_hit2"] = 1.0 if (n >= 2 and pred == dk) else 0.0
                f["ng_miss2"] = 1.0 if (n >= 2 and pred != dk) else 0.0
                f["ng_frac"] = (c[dk]/tot) if (c is not None and n >= 2) else 0.0   # share of earlier continuations = d_k
                f["ng_cnt"] = math.log1p(tot) if n >= 2 else 0.0
                # pre: does the n-gram prediction exist (with length >= 2) for this position; d_k unknown
                f["ng_pre"] = math.log1p(n) if n >= 2 else 0.0
                f["tok_seen"] = 0.0          # filled in the second pass below
                f["dk"] = dk
                f["ctx"] = rd.dtok[k - 2] if k >= 2 else rd.vtok[0]
                rows.append(f)
            # previous-draft hit (known before step k): hit of draft k-1
            for k in range(1, rd.D + 1):
                rows[k - 1]["ng_hit_prev"] = rows[k - 2]["ng_hit2"] if k >= 2 else 0.0
                rows[k - 1]["ng_n_prev"] = rows[k - 2]["ng_n"] if k >= 2 else 0
            ext[(rd.req, rd.idx)] = rows
            for k in range(rd.L):
                ix.push(rd.dtok[k])
            # count of the draft token in the committed text (frequency), computed after the round for the next one
    # token frequency in committed text, computed in a second pass (needs the text before the round)
    for q in reqs:
        cnt = Counter(); n_txt = 0
        for rd in q:
            cnt[rd.vtok[0]] += 1; n_txt += 1
            rows = ext[(rd.req, rd.idx)]
            for k in range(1, rd.D + 1):
                dk = rd.dtok[k - 1]
                rows[k - 1]["tok_seen"] = math.log1p(cnt[dk])
                rows[k - 1]["rep_prev"] = 1.0 if (k >= 2 and dk == rd.dtok[k - 2]) else 0.0
            for k in range(rd.L):
                cnt[rd.dtok[k]] += 1; n_txt += 1
    return ext

# ----------------------------------------------------------------------------------------------------------------
# token tables (fitted on the training requests, cross-fitted inside train to avoid in-sample leakage):
# shrunk mean residual (y - q_S1) of draft token d_k, and of the context token before it

class TokTable:
    def __init__(self, alpha=20.0):
        self.alpha = alpha
        self.s = defaultdict(float); self.n = defaultdict(float)
    def add(self, t, r):
        self.s[t] += r; self.n[t] += 1
    def get(self, t):
        n = self.n.get(t, 0.0)
        return self.s.get(t, 0.0)/(n + self.alpha)
    def export(self, min_n=30):
        return {t: self.get(t) for t, n in self.n.items() if n >= min_n}

# ----------------------------------------------------------------------------------------------------------------
# state of the controller within a request (depends on the depth the policy used)

class St:
    def __init__(self):
        self.ema = [0.7]*17
        self.prev = A.Prev()
        self.run_full = 0            # consecutive earlier rounds without a rejection (all drafted tokens accepted)
        self.since_rej = 0
        self.last_ratio = 0.5
        self.resid = 0.0             # EMA of (y - q_S1) over tried positions of this request
        self.nres = 0
        self.acc_rate = 0.7          # EMA of accepted/drafted over rounds
    def update(self, rd, d, n_acc, q_tried):
        tried = min(d, n_acc + 1)
        for k in range(1, tried + 1):
            self.ema[k] = 0.9*self.ema[k] + 0.1*(1.0 if k <= n_acc else 0.0)
        for k, q in enumerate(q_tried, start=1):
            if k > tried:
                break
            y = 1.0 if k <= n_acc else 0.0
            self.resid = 0.9*self.resid + 0.1*(y - q); self.nres += 1
        rej = n_acc < d
        self.run_full = 0 if rej else self.run_full + 1
        self.since_rej = 0 if rej else self.since_rej + 1
        self.last_ratio = n_acc/max(d, 1)
        self.acc_rate = 0.8*self.acc_rate + 0.2*self.last_ratio
        self.prev = A.Prev(rd, d)

GROUPS_ALL = ("inter", "ngram", "tok", "run", "resid", "nl", "tail", "ver")

def feat(rd, k, st, ext, tabs, groups, post=True):
    x = A.featurize(rd, k, st.prev, st.ema[min(k, 16)], S1, post)
    row = ext[(rd.req, rd.idx)][k - 1]
    lg = [A._lg(v) for v in rd.p]
    if "inter" in groups:
        # cumulative MTP confidence of the earlier drafts, previous draft's p, per-position slope of lg p
        cum = sum(math.log(max(rd.p[j], 1e-4)) for j in range(k - 1))
        x.append(cum)
        x.append(lg[k - 2] if k >= 2 else 0.0)
        if post:
            kk = min(k, 4)
            x += [lg[k - 1]*(1.0 if kk == j else 0.0) for j in (1, 2, 3)]
        else:
            x += [0.0, 0.0, 0.0]
    if "nl" in groups:
        # nonlinear terms on the logit margin (hinge basis), post only
        lm = min(rd.lm[k - 1], 30.0) if post else 0.0
        x += [max(0.0, lm - t) for t in (1.0, 2.0, 4.0, 8.0)] if post else [0.0]*4
    if "ngram" in groups:
        x.append(row["ng_pre"])
        x.append(row["ng_hit_prev"])
        if post:
            x += [row["ng_hit2"], row["ng_miss2"], row["ng_frac"], row["ng_cnt"]*row["ng_hit2"], row["tok_seen"], row["rep_prev"]]
        else:
            x += [0.0]*6
    if "tok" in groups:
        tdk, tctx = tabs
        x.append(tctx.get(row["ctx"]))
        x.append(tdk.get(row["dk"]) if post else 0.0)
    if "run" in groups:
        x += [math.log1p(st.run_full), math.log1p(st.since_rej), st.last_ratio, A._lg(min(max(st.acc_rate, 0.02), 0.98))]
    if "resid" in groups:
        x.append(st.resid)
    if "tail" in groups:
        # top-10 renormalised MTP distribution: second prob p2 = p1*exp(-lm) and the mass outside the top two
        if post:
            p1, lmk = rd.p[k - 1], min(rd.lm[k - 1], 30.0)
            p2 = p1*math.exp(-lmk)
            x += [math.log(max(p2, 1e-9)), math.log(max(1.0 - p1 - p2, 1e-6))]
        else:
            x += [0.0, 0.0]
    if "ver" in groups:
        # the S2 verifier features of the previous round (analyze.featurize, ver part)
        pv = st.prev
        x += [float(pv.first), math.log1p(pv.n), float(pv.rej), math.log1p(max(pv.tlm, 0.0)),
              math.log1p(max(pv.tlm_mean, 0.0)), pv.tH*pv.has_H, float(pv.has_H)]
    return x

# ----------------------------------------------------------------------------------------------------------------
# samples (collection depth = D, like analyze.build_samples)

def base_models(train):
    Xp, yp, _ = A.build_samples(train, S1, post=True)
    Xq, yq, _ = A.build_samples(train, S1, post=False)
    return A.Logit().fit(Xp, yp), A.Logit().fit(Xq, yq)

def build(reqs, ext, tabs, groups, s1post, post=True):
    X, y, kk = [], [], []
    for q in reqs:
        st = St()
        for rd in q:
            tried = min(rd.D, rd.L + 1)
            qs = []
            for k in range(1, tried + 1):
                X.append(feat(rd, k, st, ext, tabs, groups, post))
                y.append(1.0 if k <= rd.L else 0.0)
                kk.append(k)
                qs.append(s1post.p1(A.featurize(rd, k, st.prev, st.ema[min(k, 16)], S1, True)) if "resid" in groups else 0.0)
            st.update(rd, rd.D, rd.L, qs)
    return np.array(X, dtype=float), np.array(y), np.array(kk)

def make_tables(reqs, s1post):
    tdk, tctx = TokTable(), TokTable()
    for q in reqs:
        st = St()
        for rd in q:
            tried = min(rd.D, rd.L + 1)
            rows = None
            for k in range(1, tried + 1):
                qv = s1post.p1(A.featurize(rd, k, st.prev, st.ema[min(k, 16)], S1, True))
                yv = 1.0 if k <= rd.L else 0.0
                tdk.add(rd.dtok[k - 1], yv - qv)
                tctx.add(rd.dtok[k - 2] if k >= 2 else rd.vtok[0], yv - qv)
            st.update(rd, rd.D, rd.L, [])
    return tdk, tctx

def build_train_crossfit(train, ext, groups, s1post, post, nfold=4):
    """token tables for each training request come from the other folds"""
    if "tok" not in groups:
        return build(train, ext, (None, None), groups, s1post, post)
    Xs, ys, ks = [], [], []
    folds = [train[i::nfold] for i in range(nfold)]
    for i in range(nfold):
        others = [q for j in range(nfold) if j != i for q in folds[j]]
        tabs = make_tables(others, s1post)
        X, y, k = build(folds[i], ext, tabs, groups, s1post, post)
        Xs.append(X); ys.append(y); ks.append(k)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ks)

# ----------------------------------------------------------------------------------------------------------------
# replay policy: analyze.SignalStop with the extended features

class Stop2(A.SignalStop):
    def __init__(self, *a, ext=None, tabs=None, groups=(), s1post=None, **kw):
        super().__init__(*a, **kw)
        self.ext, self.tabs, self.groups, self.s1post = ext, tabs, groups, s1post
    def begin(self):
        super().begin(); self.st = St()
    def round(self, rd):
        st = self.st
        P, depth, steps = 1.0, 0, 0
        for k in range(1, min(self.cap, rd.D) + 1):
            if k > self.dmin:
                q_pre = self.pre.p1(feat(rd, k, st, self.ext, self.tabs, self.groups, post=False))
                if self.opt is None:
                    v = self._ahead(P, k, q_pre, False)
                else:
                    v = float(np.mean([max(0.0, self._ahead(P, k, q, True)) for q in self.opt.get(q_pre)])) - self.lam*self.cost.c_d
                if v <= 0.0:
                    break
            steps = k
            q_post = self.post.p1(feat(rd, k, st, self.ext, self.tabs, self.groups, post=True))
            if k > self.dmin and self._ahead(P, k, q_post, True) <= 0.0:
                break
            P *= q_post
            depth = k
        return depth, steps
    def observe(self, rd, d, steps, n_acc, t):
        st = self.st
        qs = []
        if "resid" in self.groups:
            for k in range(1, min(d, n_acc + 1) + 1):
                qs.append(self.s1post.p1(A.featurize(rd, k, st.prev, st.ema[min(k, 16)], S1, True)))
        st.update(rd, d, n_acc, qs)
        if self.online:
            self.lam = 0.9*self.lam + 0.1*((n_acc + 1)/t)

# ----------------------------------------------------------------------------------------------------------------

def per_pos_auc(p, y, kk):
    row = [A.auc(p, y)]
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 99)):
        s = (kk >= lo) & (kk <= hi)
        row.append(A.auc(p[s], y[s]))
    return row

def evaluate_set(train, test, ext_tr, ext_te, groups, s1post, cost, cap, best_fixed, l2=1e-2):
    Xtr, ytr, _ = build_train_crossfit(train, ext_tr, groups, s1post, True)
    Xpr, ypr, _ = build_train_crossfit(train, ext_tr, groups, s1post, False)
    tabs = make_tables(train, s1post) if "tok" in groups else (None, None)
    mpost = A.Logit().fit(Xtr, ytr, l2=l2); mpre = A.Logit().fit(Xpr, ypr, l2=l2)
    Xte, yte, kte = build(test, ext_te, tabs, groups, s1post, True)
    Xtp, ytp, _ = build(test, ext_te, tabs, groups, s1post, False)
    pp = mpost.predict(Xte)
    aucs = per_pos_auc(pp, yte, kte)
    auc_pre = A.auc(mpre.predict(Xtp), ytp)
    opt = A.OptionTable(mpre.predict(Xpr), mpost.predict(Xtr))
    qb = A.qbar_from(train, cap)
    kw = dict(ext=ext_te, tabs=tabs, groups=groups, s1post=s1post)
    lam = best_fixed/1000.0
    for _ in range(6):
        tok, ms, nr, ds = A.run_policy(test, cost, cap, Stop2(cap, cost, mpre, mpost, S1, qb, lam, dmin=1, opt=opt, **kw))
        lam = tok/ms
    tok, ms, nr, ds = A.run_policy(test, cost, cap, Stop2(cap, cost, mpre, mpost, S1, qb, lam, dmin=1, opt=opt, **kw))
    r_star = 1000*tok/ms
    tok, ms, nr, ds2 = A.run_policy(test, cost, cap, Stop2(cap, cost, mpre, mpost, S1, qb, lam, online=True, dmin=1, lam0=best_fixed/1000.0, opt=opt, **kw))
    r_on = 1000*tok/ms
    # per-request rates (lam*) for a robustness count
    per = []
    for q in test:
        t1, m1, _, _ = A.run_policy([q], cost, cap, Stop2(cap, cost, mpre, mpost, S1, qb, lam, dmin=1, opt=opt, **kw))
        per.append(1000*t1/m1)
    return dict(auc=aucs, auc_pre=auc_pre, brier=A.brier(pp, yte), star=r_star, online=r_on, depth=ds/nr,
                mpost=mpost, mpre=mpre, opt=opt, tabs=tabs, lam=lam, per=per, n=Xtr.shape[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("test")
    ap.add_argument("--tv", default="1:17.9,3:25.8,6:34.3,8:41")
    ap.add_argument("--draft-ms", type=float, default=1.3); ap.add_argument("--host-ms", type=float, default=1.0)
    ap.add_argument("--catchup-ms", type=float, default=0.5); ap.add_argument("--cap", type=int, default=7)
    ap.add_argument("--export", default=None, help="comma-separated groups of the model to print coefficients for")
    ap.add_argument("--sets", default=None, help="semicolon-separated list of group sets, e.g. 'inter;ngram;inter,ngram'")
    args = ap.parse_args()
    train = A.parse(args.train); test = A.parse(args.test)
    ext_tr = text_features(train); ext_te = text_features(test)
    cost = A.Cost(A.make_tv(args.tv), args.draft_ms, args.host_ms, args.catchup_ms)
    cap = args.cap
    fixed = max(1000*(lambda r: r[0]/r[1])(A.run_policy(test, cost, cap, A.Fixed(d))) for d in range(1, cap + 1))
    t, m, _, _ = A.run_policy(test, cost, cap, A.DepthCtlFirstSkip(cap)); r_ctl = 1000*t/m
    t, m, _, _ = A.run_policy(test, cost, cap, A.Oracle(cap, 1)); r_or = 1000*t/m
    s1post, s1pre = base_models(train)
    print("held-out: depth_ctl %.1f tok/s, oracle %.1f (+%.1f%%), best fixed %.1f" % (r_ctl, r_or, 100*(r_or/r_ctl - 1), fixed))
    if args.sets:
        sets = [tuple(s.split(",")) if s else () for s in args.sets.split(";")]
    else:
        sets = [(), ("inter",), ("nl",), ("ngram",), ("tok",), ("run",), ("resid",),
                ("inter", "nl"), ("inter", "nl", "ngram"), ("inter", "nl", "ngram", "run", "resid"), GROUPS_ALL]
    print("%-36s %4s %6s %6s %6s %6s %6s %6s %7s %7s %7s %7s %6s %5s" % (
        "groups", "nf", "AUC", "k=1", "k=2", "k=3", "k>=4", "pre", "brier", "lam*", "vs ctl", "online", "depth", "win"))
    res = {}
    base_per = None
    for g in sets:
        r = evaluate_set(train, test, ext_tr, ext_te, g, s1post, cost, cap, fixed)
        if g == ():
            base_per = r["per"]
        win = sum(a > b + 1e-9 for a, b in zip(r["per"], base_per)) if base_per else 0
        name = "S1" + ("+" + "+".join(g) if g else " (baseline)")
        print("%-36s %4d " % (name, r["n"]) + " ".join("%6.3f" % v for v in r["auc"]) + " %6.3f %7.4f %7.1f %+6.1f%% %7.1f %6.2f %2d/%d" % (
            r["auc_pre"], r["brier"], r["star"], 100*(r["star"]/r_ctl - 1), r["online"], r["depth"], win, len(test)))
        sys.stdout.flush()
        res[g] = r
    if args.export:
        g = tuple(args.export.split(","))
        r = res.get(g) or evaluate_set(train, test, ext_tr, ext_te, g, s1post, cost, cap, fixed)
        for nm, m in (("post", r["mpost"]), ("pre", r["mpre"])):
            print("%s n_features=%d" % (nm, len(m.mu)))
            print("  mu", " ".join("%.6g" % v for v in m.mu))
            print("  sd", " ".join("%.6g" % v for v in m.sd))
            print("  w ", " ".join("%.6g" % v for v in m.w))
        print("lam* %.5f tok/ms" % r["lam"])
        print("opt_edges", " ".join("%.6g" % v for v in r["opt"].edges))
        for b, d in enumerate(r["opt"].draws):
            print("opt_draws %d" % b, " ".join("%.6g" % v for v in d))
    return 0

if __name__ == "__main__":
    sys.exit(main())
