# train the MTP block for recursive use, v2: loss only on assistant-answer positions (the target's greedy tokens in
# their original chat context), evaluation on a separate held-out dump (prompt families never used for training).
# usage: train_mtp2.py mtp.gguf train_pairs.bin held_pairs.bin out.pt epochs [lr]
import os, sys, time, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
from mtp_torch import MTP, load_gguf, HC, NE

J = 4
NLOSS = 384
NEF = HC*NE
import os
dev = os.environ.get("MTP_DEV", "mps")
rec = np.dtype([("tok", "<i4"), ("hm", "<f2", NEF), ("ht", "<f2", NEF)])

def segments(path):
    d = np.memmap(path, dtype=rec, mode="r")
    tok = np.asarray(d["tok"])
    start, user = tok[0], tok[1]
    b = [i for i in range(len(tok) - 1) if tok[i] == start and tok[i + 1] == user] + [len(tok)]
    segs = []
    for i in range(len(b) - 1):
        a, e = b[i], b[i + 1]
        # the answer starts after the last "<|im_start|> X \n" header of the request (X = assistant)
        hs = [k for k in range(a, e - 2) if tok[k] == start]
        if len(hs) < 2 or e - a < 64:
            continue
        ans = hs[-1] + 3
        segs.append((a, e, ans))
    return d, tok, segs

dtr, toktr, train = segments(sys.argv[2])
dho, tokho, held = segments(sys.argv[3])
print("train texts %d (%d answer tokens), held-out texts %d" % (len(train), sum(e - s for _, e, s in train), len(held)), flush=True)

W = load_gguf(sys.argv[1], want=lambda n: n.startswith("blk.48.") or n in ("token_embd.weight", "output.weight"))
m = MTP(W, device=dev)
del W
for n_, p_ in m.named_parameters():
    p_.requires_grad = not n_.endswith("_e")
params = [p_ for p_ in m.parameters() if p_.requires_grad]
print("trainable params %.1fM" % (sum(p_.numel() for p_ in params)/1e6), flush=True)

def load(d, tok, a, e, maxlen):
    e = min(e, a + maxlen)
    t = torch.from_numpy(tok[a:e].astype(np.int64)).to(dev)
    h = torch.from_numpy(np.asarray(d["ht"][a:e]).astype(np.float32)).to(dev)
    return t, h

def unroll(toks, ht, ans, grad, sel=None):
    L = toks.shape[0]
    pos = torch.arange(L, device=dev)
    h_prev = torch.cat([torch.zeros(1, NEF, device=dev), ht[:-1]], 0)
    losses, accs = [], []
    for j in range(1, J + 1):
        h_out, head = m(toks, h_prev, pos)
        # position p predicts token p+1; only answer tokens are targets, and step j needs a chain start at p-j+1 >= ans-1
        lo = max(j, ans - 1)
        idx = torch.arange(lo, L - 1, device=dev)
        if sel is not None and idx.numel() > sel:
            idx = idx[torch.randperm(idx.numel(), device=dev)[:sel]]
        lg = m.logits(head[idx]).float()
        tgt = toks[idx + 1]
        losses.append(F.cross_entropy(lg, tgt))
        accs.append((lg.argmax(-1) == tgt).float().mean().item())
        h_prev = torch.cat([torch.zeros(1, NEF, device=dev), h_out[:-1]], 0)
        if not grad:
            h_prev = h_prev.detach()
    return losses, accs

def evaluate():
    m.eval()
    tot = np.zeros(J); n = 0
    with torch.no_grad():
        for a, e, s in held:
            t, h = load(dho, tokho, a, e, 1600)
            _, accs = unroll(t, h, s - a, False)
            tot += np.array(accs); n += 1
    m.train()
    return tot/n

t0 = time.time()
base = evaluate()
print("held-out before: per-step top-1 agreement", np.round(base, 4), "(%.0f s)" % (time.time() - t0), flush=True)
lr = float(sys.argv[6]) if len(sys.argv) > 6 else 2e-5
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
epochs = int(sys.argv[5])
step = 0
best = base.copy()
for ep in range(epochs):
    random.seed(ep)
    order = list(train); random.shuffle(order)
    for a, e, s in order:
        t, h = load(dtr, toktr, a, e, 1100)
        if s - a >= t.shape[0] - 8:
            continue
        losses, accs = unroll(t, h, s - a, True, sel=NLOSS)
        loss = sum(losses)/J
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            print("ep %d step %d loss %.4f accs %s (%.0f s)" % (ep, step, loss.item(), np.round(accs, 3), time.time() - t0), flush=True)
    r = evaluate()
    print("held-out after epoch %d: %s (%.0f s)" % (ep, np.round(r, 4), time.time() - t0), flush=True)
    if r[1:].mean() > best[1:].mean():
        best = r
        torch.save({n_: p_.detach().cpu() for n_, p_ in m.named_parameters() if p_.requires_grad}, sys.argv[4])
        print("saved", flush=True)
