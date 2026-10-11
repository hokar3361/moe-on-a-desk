# check the PyTorch MTP against llama.cpp: feed the first dumped request (tokens, target hidden shifted by one) and
# compare the output residual with the dumped h_mtp. usage: mtp_check.py mtp.gguf pairs.bin
import sys, time, numpy as np, torch
sys.path.insert(0, "$HOME/kbench")
from mtp_torch import MTP, load_gguf, HC, NE

NEF = HC*NE
rec = np.dtype([("tok", "<i4"), ("hm", "<f2", NEF), ("ht", "<f2", NEF)])
d = np.memmap(sys.argv[2], dtype=rec, mode="r")
tok = np.asarray(d["tok"][:4000])
# a request starts with <|im_start|> user; take the records up to the next request start
start, user = tok[0], tok[1]
ends = [i for i in range(2, len(tok) - 1) if tok[i] == start and tok[i + 1] == user]
L = min(ends[0] if ends else len(tok), 1400)
print("first request: %d tokens" % L, flush=True)

t0 = time.time()
W = load_gguf(sys.argv[1], want=lambda n: n.startswith("blk.48.") or n in ("token_embd.weight", "output.weight"))
print("loaded %d tensors in %.0f s" % (len(W), time.time() - t0), flush=True)
for k in sorted(W):
    if "exps" not in k:
        print("  ", k, tuple(W[k].shape))
dev = "mps"
m = MTP(W, device=dev)
del W

toks = torch.from_numpy(tok[:L].astype(np.int64)).to(dev)
ht = torch.from_numpy(np.asarray(d["ht"][:L]).astype(np.float32)).to(dev)
hm = torch.from_numpy(np.asarray(d["hm"][:L]).astype(np.float32)).to(dev)
h_in = torch.cat([torch.zeros(1, NEF, device=dev), ht[:-1]], 0)
pos = torch.arange(L, device=dev)
with torch.no_grad():
    t0 = time.time()
    h_out, head = m(toks, h_in, pos)
    torch.mps.synchronize()
    print("forward %.2f s" % (time.time() - t0), flush=True)
    a, b = h_out[1:], hm[1:]
    for s in range(HC):
        sa, sb = a[:, s*NE:(s + 1)*NE], b[:, s*NE:(s + 1)*NE]
        cos = (sa*sb).sum(1)/(sa.norm(dim=1)*sb.norm(dim=1) + 1e-9)
        rel = (sa - sb).norm(dim=1)/(sb.norm(dim=1) + 1e-9)
        print("stream %d: cosine mean %.6f min %.6f, rel err mean %.5f max %.5f" % (s, cos.mean(), cos.min(), rel.mean(), rel.max()))
    # does the MTP head predict the next token of the text (the target's greedy choice) as llama.cpp does?
    lg = m.logits(head[:-1])
    pred = lg.argmax(-1)                       # head[k] predicts token k+1
    ok = (pred[1:] == toks[2:]).float()
    print("one-step top-1 agreement with the next text token: all %.4f, last 600 (answer) %.4f" % (ok.mean().item(), ok[-600:].mean().item()))
