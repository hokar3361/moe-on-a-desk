# Stage 0a/0b evaluator: chain-conditional acceptance per draft position with inference-matched MTP KV.
# usage: s0eval.py mtp.gguf [trained.pt|-] sets(comma list of H18,SUBJ,TRAIN,TRAINFIT) out.json [J] [mode inf|old]
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s0common import *
from s0model import unroll
from mtp_torch import MTP, load_gguf

D = MTPDIR
dev = "cuda"
TOPK = (2, 4, 8)

def build_sets(vocab, names):
    out = {}
    dho, tokho, held = segments(os.path.join(D, "pairs2_held.bin"))
    dtr, toktr, train = segments(os.path.join(D, "pairs2.bin"))
    lab = lambda tok, s: label(vocab, tok, s)
    hold = lambda p: any(h in p for h in HOLD_SUBJECTS + HOLD_TASKS)
    if "H18" in names:
        out["H18"] = (dho, tokho, [(s, genre(lab(tokho, s))) for s in held])
    tr = [(s, lab(toktr, s)) for s in train]
    if "SUBJ" in names:
        out["SUBJ"] = (dtr, toktr, [(s, genre(p)) for s, p in tr if hold(p)])
    if "TRAIN" in names:
        out["TRAIN"] = (dtr, toktr, [(s, genre(p)) for s, p in tr])
    if "TRAINFIT" in names:
        out["TRAINFIT"] = (dtr, toktr, [(s, genre(p)) for s, p in tr if not hold(p)])
    return out

def content_start(vocab, tok, s):
    a, e, ans, hdr = s
    for k in range(ans, min(ans + 8, e)):
        if vocab[int(tok[k])] == "</think>":
            return k + 2           # skip "</think>" and the following "\n\n" (part of the prompt at inference)
    return ans

def seg_stats(m, d, tok, s, J, mode, maxlen=1600):
    a, e, ans, hdr = s
    e = min(e, a + maxlen)
    t = torch.from_numpy(tok[a:e].astype(np.int64)).to(dev)
    h = torch.from_numpy(np.asarray(d["ht"][a:e]).astype(np.float32)).to(dev)
    L = t.shape[0]
    corr = np.zeros((J, L), bool); rank = np.full((J, L), 99, np.int16)
    with torch.no_grad():
        for j, head in unroll(m, t, h, J, mode):
            lg = m.logits(head[:-1]).float()
            top = torch.topk(lg, max(TOPK), -1).indices            # [L-1, 8]
            hit = (top == t[1:, None])
            r = torch.where(hit.any(-1), hit.float().argmax(-1).long(), torch.full_like(hit[:, 0], 99, dtype=torch.long))
            rank[j - 1, :L - 1] = r.cpu().numpy()
            corr[j - 1, :L - 1] = (r == 0).cpu().numpy()
    return corr, rank, L

def accumulate(acc, corr, rank, c0, L, J):
    # chain starts s in [c0, L-2]; stage j of chain s is position s+j-1 predicting token s+j (needs s+j <= L-1)
    for s in range(c0, L - 1):
        for j in range(1, J + 1):
            p = s + j - 1
            if p > L - 2:
                break
            acc["reach"][j - 1] += 1
            for ki, k in enumerate(TOPK):
                acc["topk"][j - 1][ki] += int(rank[j - 1, p] < k)
            if not corr[j - 1, p]:
                break
            acc["acc"][j - 1] += 1
        for j in range(1, J + 1):
            p = s + j - 1
            if p > L - 2: break
            acc["marg_n"][j - 1] += 1; acc["marg"][j - 1] += int(corr[j - 1, p])
    # simulated fixed-depth rounds (greedy verify: accept the leading correct drafts, +1 bonus token)
    for dd in (2, 3, 4, 5):
        if dd > J: continue
        r = acc["sim"].setdefault(str(dd), {"rounds": 0, "tokens": 0, "reach": [0]*dd, "acc": [0]*dd})
        s = c0
        while s < L - 1:
            k = 0
            for j in range(1, dd + 1):
                p = s + j - 1
                if p > L - 2: break
                r["reach"][j - 1] += 1
                if not corr[j - 1, p]: break
                r["acc"][j - 1] += 1; k += 1
            r["rounds"] += 1; r["tokens"] += k + 1
            s += k + 1

def new_acc(J):
    return {"reach": [0]*J, "acc": [0]*J, "topk": [[0]*len(TOPK) for _ in range(J)], "marg": [0]*J, "marg_n": [0]*J, "sim": {}, "texts": 0}

def merge(dst, src):
    for k in ("reach", "acc", "marg", "marg_n"):
        dst[k] = [x + y for x, y in zip(dst[k], src[k])]
    dst["topk"] = [[x + y for x, y in zip(a, b)] for a, b in zip(dst["topk"], src["topk"])]
    for dd, r in src["sim"].items():
        q = dst["sim"].setdefault(dd, {"rounds": 0, "tokens": 0, "reach": [0]*len(r["reach"]), "acc": [0]*len(r["acc"])})
        q["rounds"] += r["rounds"]; q["tokens"] += r["tokens"]
        q["reach"] = [x + y for x, y in zip(q["reach"], r["reach"])]; q["acc"] = [x + y for x, y in zip(q["acc"], r["acc"])]
    dst["texts"] += src["texts"]

def summary(a):
    cond = [round(x/max(n, 1), 4) for x, n in zip(a["acc"], a["reach"])]
    topk = [[round(x/max(n, 1), 4) for x in row] for row, n in zip(a["topk"], a["reach"])]
    marg = [round(x/max(n, 1), 4) for x, n in zip(a["marg"], a["marg_n"])]
    sim = {dd: {"cond": [round(x/max(n, 1), 4) for x, n in zip(r["acc"], r["reach"])], "reach": r["reach"],
                "tok_per_round": round(r["tokens"]/max(r["rounds"], 1), 4)} for dd, r in a["sim"].items()}
    return {"texts": a["texts"], "starts": a["reach"][0], "cond": cond, "reach": a["reach"], "topk": topk, "marginal": marg, "sim": sim}

GROUPS = {"code": {"code"}, "prose": PROSE, "narr": NARR}
ORDER = ["all", "code", "prose", "narr"]

def run(m, sets, J, mode, vocab):
    res = {}
    for name, (d, tok, items) in sets.items():
        per = {}
        t0 = time.time()
        for s, g in items:
            c0 = content_start(vocab, tok, s)
            corr, rank, L = seg_stats(m, d, tok, s, J, mode)
            a = new_acc(J); a["texts"] = 1
            accumulate(a, corr, rank, c0 - s[0], L, J)
            for key in [g] + [gn for gn, gs in GROUPS.items() if g in gs and gn != g] + ["all"]:
                merge(per.setdefault(key, new_acc(J)), a)
        res[name] = {k: summary(v) for k, v in per.items()}
        print(name, "%.0f s" % (time.time() - t0), flush=True)
        for k in ORDER + sorted(x for x in per if x not in ORDER):
            if k in res[name]:
                r = res[name][k]
                print("  %-9s texts %3d starts %6d cond %s | sim2 %s | sim5 %s" % (k, r["texts"], r["starts"], r["cond"],
                      r["sim"].get("2", {}).get("cond"), r["sim"].get("5", {}).get("cond")), flush=True)
    return res

def load_model(gguf, pt):
    W = load_gguf(gguf, want=lambda n: n.startswith("blk.48.") or n in ("token_embd.weight", "output.weight"))
    m = MTP(W, device=dev); del W
    if pt and pt != "-":
        sd = torch.load(pt)
        with torch.no_grad():
            for n_, p_ in m.named_parameters():
                if n_ in sd: p_.copy_(sd[n_].to(p_.dtype))
        print("loaded", pt, len(sd), "tensors", flush=True)
    m.eval()
    return m

if __name__ == "__main__":
    gguf, pt, names, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
    J = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    mode = sys.argv[6] if len(sys.argv) > 6 else "inf"
    vocab = load_vocab(gguf)
    sets = build_sets(vocab, names)
    m = load_model(gguf, pt)
    res = run(m, sets, J, mode, vocab)
    json.dump({"gguf": gguf, "pt": pt, "mode": mode, "J": J, "res": res}, open(out, "w"), indent=1)
