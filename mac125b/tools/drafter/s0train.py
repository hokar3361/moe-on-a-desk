# Stage 0c: retrain the MTP block for recursive use with the attention KV built as at inference
# (accepted positions: first-stage KV from the target hidden state; chain positions: recursive-state KV, diagonal mask).
# Same recipe as train_mtp3 (v3): experts frozen, 4 stages, loss weights 0.25/0.5/1/1, lr 5e-6, 1 epoch, 384 positions/stage.
# Training texts = pairs2.bin minus the subject-disjoint hold-out (s0common.HOLD_SUBJECTS / HOLD_TASKS).
# usage: s0train.py mtp.gguf out.pt [mode inf|old] [lr] [epochs]
import os, sys, time, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s0common import *
from s0model import unroll
from s0eval import build_sets, content_start
from mtp_torch import MTP, load_gguf

J, NLOSS, WEIGHTS = 4, 384, (0.25, 0.5, 1.0, 1.0)
dev = "cuda"
gguf, outp = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "inf"
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-6
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 1

vocab = load_vocab(gguf)
d, tok, items = build_sets(vocab, ["TRAINFIT"])["TRAINFIT"]
print("train texts %d, answer tokens %d, mode %s, lr %g" % (len(items), sum(s[1] - s[2] for s, _ in items), mode, lr), flush=True)
W = load_gguf(gguf, want=lambda n: n.startswith("blk.48.") or n in ("token_embd.weight", "output.weight"))
m = MTP(W, device=dev); del W
for n_, p_ in m.named_parameters():
    p_.requires_grad = not n_.endswith("_e")
params = [p_ for p_ in m.parameters() if p_.requires_grad]
print("trainable params %.1fM" % (sum(p_.numel() for p_ in params)/1e6), flush=True)
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
m.train()
t0 = time.time(); step = 0
for ep in range(epochs):
    random.seed(ep)
    order = list(items); random.shuffle(order)
    for s, g in order:
        a, e, ans, hdr = s
        e = min(e, a + 1100)
        c0 = content_start(vocab, tok, s) - a
        if c0 >= e - a - 8:
            continue
        t = torch.from_numpy(tok[a:e].astype(np.int64)).to(dev)
        h = torch.from_numpy(np.asarray(d["ht"][a:e]).astype(np.float32)).to(dev)
        L = t.shape[0]
        losses, accs = [], []
        for j, head in unroll(m, t, h, J, mode):
            # stage j at position p belongs to the chain starting at p-j+1 >= c0 and predicts token p+1
            idx = torch.arange(c0 + j - 1, L - 1, device=dev)
            if idx.numel() > NLOSS:
                idx = idx[torch.randperm(idx.numel(), device=dev)[:NLOSS]]
            lg = m.logits(head[idx]).float()
            tgt = t[idx + 1]
            losses.append(F.cross_entropy(lg, tgt))
            accs.append((lg.argmax(-1) == tgt).float().mean().item())
        loss = sum(w*l for w, l in zip(WEIGHTS, losses))/sum(WEIGHTS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            print("ep %d step %d loss %.4f accs %s (%.0f s)" % (ep, step, loss.item(), np.round(accs, 3), time.time() - t0), flush=True)
torch.save({n_: p_.detach().cpu() for n_, p_ in m.named_parameters() if p_.requires_grad}, outp)
print("saved %s (%.0f s)" % (outp, time.time() - t0), flush=True)
