# Inference-matched recursive MTP unroll (stage 0a/0c).
# Real llama.cpp (fork, merged catch-up): MTP KV of every accepted position comes from the first stage (input = target
# hidden state); only the positions inside the current draft chain carry KV built from the recursive state.
# Stage j (1-based) at position p belongs to the chain that starts at s = p - j + 1. Its attention sees
#   stage-1 K/V at positions q <= s   and   stage-i K/V at position s + i - 1 (i = 2..j)   (EAGLE-3-style diagonal).
import torch, torch.nn.functional as F
from s0common import HC, NE, NEF
from mtp_torch import rms, rope_neox, NH, NKV, HD

G = NH//NKV
SCALE = 1.0/HD**0.5

def _shift(x, s):
    # y[t] = x[t - s] (zeros for t < s)
    if s == 0:
        return x
    return torch.cat([torch.zeros_like(x[:s]), x[:-s]], 0)

def attn_inf(m, x, pos, j, kv):
    T = x.shape[0]
    qf = (x @ m.wq.t()).reshape(T, NH, 2*HD)
    q, gate = qf[..., :HD], qf[..., HD:]
    q = rms(q, m.q_norm)
    k = rms((x @ m.wk.t()).reshape(T, NKV, HD), m.k_norm)
    v = (x @ m.wv.t()).reshape(T, NKV, HD)
    q, k = rope_neox(q, pos), rope_neox(k, pos)
    K1, V1 = (k, v) if j == 1 else kv[0]
    qg = q.reshape(T, NKV, G, HD)
    t = torch.arange(T, device=x.device)
    s1 = torch.einsum("tkgd,skd->kgts", qg, K1)*SCALE                       # [NKV, G, T, T]
    allowed = t[None, :] <= (t[:, None] - (j - 1))                          # key s <= query t - (j-1)
    s1 = s1.masked_fill(~allowed, float("-inf"))
    cols, vals = [s1], []
    for i in range(2, j + 1):
        Ki, Vi = (k, v) if i == j else kv[i - 1]
        sh = j - i
        Ks, Vs = _shift(Ki, sh), _shift(Vi, sh)
        d = torch.einsum("tkgd,tkd->kgt", qg, Ks)*SCALE
        if sh > 0:
            d = d.masked_fill((t < sh)[None, None, :], float("-inf"))
        cols.append(d[..., None]); vals.append(Vs)
    P = torch.softmax(torch.cat(cols, -1), -1)
    o = torch.einsum("kgts,skd->tkgd", P[..., :T], V1)
    for n, Vs in enumerate(vals):
        o = o + P[..., T + n].permute(2, 0, 1)[..., None]*Vs[:, :, None, :]
    o = o.reshape(T, NH*HD)*torch.sigmoid(gate.reshape(T, NH*HD))
    return o @ m.wo.t(), k, v

def stage(m, toks, h_in, pos, j, kv):
    T = toks.shape[0]
    e = m.tok_embd[toks].float()
    e_n = rms(e, m.enorm)[:, None, :].expand(T, HC, NE)
    h_n = rms(h_in.reshape(T, HC, NE), m.hnorm)
    res = torch.cat([e_n, h_n], dim=-1) @ m.eh_proj.t()
    mixed, inj = m.hc_mix(res, m.hc_attn_norm, m.hc_attn_down, m.hc_attn_up, m.hc_attn_inject)
    a, k, v = attn_inf(m, mixed, pos, j, kv)
    res = m.hc_combine(res, a, inj)
    mixed, inj = m.hc_mix(res, m.hc_ffn_norm, m.hc_ffn_down, m.hc_ffn_up, m.hc_ffn_inject)
    res = m.hc_combine(res, m.moe(mixed), inj)
    h_out = res.reshape(T, NEF)
    head, _ = m.hc_mix(res, m.hc_head_norm, m.hc_head_down, m.hc_head_up, None)
    return h_out, head, k, v

def unroll(m, toks, ht, J, mode="inf"):
    """yields (j, head_j) for j = 1..J. head_j[p] predicts token p+1 for the chain starting at p-j+1.
    mode 'inf' = inference-matched KV; mode 'old' = previous training (all KV from the recursive state)."""
    L = toks.shape[0]
    pos = torch.arange(L, device=toks.device)
    z = torch.zeros(1, NEF, device=toks.device)
    h_prev = torch.cat([z, ht[:-1]], 0)
    kv = []
    for j in range(1, J + 1):
        if mode == "inf":
            h_out, head, k, v = stage(m, toks, h_prev, pos, j, kv)
            kv.append((k, v))
        else:
            h_out, head = m(toks, h_prev, pos)
        yield j, head
        h_prev = torch.cat([z, h_out[:-1]], 0)
