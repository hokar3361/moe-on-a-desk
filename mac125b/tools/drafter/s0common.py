# Stage-0 common helpers: GGUF vocab decode, dump segmentation, genre labels, subject-disjoint hold-out split.
import os, sys, re, numpy as np
MTPDIR = os.environ.get("MTPDIR", r"D:\GPUNW\mac125b\mtptrain")
sys.path.insert(0, MTPDIR)
from mtp_torch import _rd, _rstr, _skip, HC, NE

NEF = HC*NE
rec = np.dtype([("tok", "<i4"), ("hm", "<f2", NEF), ("ht", "<f2", NEF)])

def load_vocab(path):
    f = open(path, "rb"); f.read(4); _rd(f, "I"); _rd(f, "Q"); (nkv,) = _rd(f, "Q")
    for _ in range(nkv):
        k = _rstr(f); (t,) = _rd(f, "I")
        if k == "tokenizer.ggml.tokens":
            _rd(f, "I"); (n,) = _rd(f, "Q")
            return [_rstr(f) for _ in range(n)]
        _skip(f, t)

def _b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
U2B = _b2u()

def decode(vocab, ids):
    out = bytearray()
    for i in ids:
        s = vocab[int(i)]
        if s.startswith("<|") and s.endswith("|>"):
            out += s.encode(); continue
        for ch in s:
            b = U2B.get(ch)
            if b is None: out += ch.encode()
            else: out.append(b)
    return out.decode("utf-8", "replace")

def segments(path):
    d = np.memmap(path, dtype=rec, mode="r")
    tok = np.asarray(d["tok"])
    start, user = tok[0], tok[1]
    b = [i for i in range(len(tok) - 1) if tok[i] == start and tok[i + 1] == user] + [len(tok)]
    segs = []
    for i in range(len(b) - 1):
        a, e = b[i], b[i + 1]
        hs = [k for k in range(a, e - 2) if tok[k] == start]
        if len(hs) < 2 or e - a < 64:
            continue
        segs.append((a, e, hs[-1] + 3, hs[-1]))   # (start, end, answer start, assistant header)
    return d, tok, segs

BANNED = re.compile(r"lighthouse|bottle|\bsea\b|ocean|sailor|iso.?8601|\bdates?\b|weekday|calendar|\bscores?\b|\brecords?\b", re.I)

def genre(prompt):
    p = prompt
    if re.search(r"Write an? \S+ function", p): return "code"
    if p.startswith("Write a poem"): return "poem"
    if p.startswith("Write a review"): return "review"
    if p.startswith("Write a short story"): return "story"
    if p.startswith("Write an essay"): return "essay"
    if p.startswith("Write a letter"): return "letter"
    if p.startswith("Write a dialogue"): return "dialogue"
    if p.startswith("Write a persuasive"): return "speech"
    if p.startswith("Write a short guide"): return "guide"
    if p.startswith("Explain"): return "explain"
    if p.startswith("Summarize"): return "summary"
    if p.startswith("Answer step by step"): return "steps"
    return "other"

PROSE = {"poem", "review", "story", "essay", "letter", "dialogue", "speech", "guide", "explain", "summary", "steps"}
# narrative / literary prose, closest to the fixed prose metric (a short story) without using it
NARR = {"poem", "review", "story", "letter", "dialogue"}

# subjects held out of training entirely (every genre), chosen before looking at any result
HOLD_SUBJECTS = ["a clockmaker who repairs a broken town clock", "a mountain village in winter", "the history of tea",
                 "a city library at night", "how glaciers shape valleys", "a lost dog finding its way home",
                 "a festival of lanterns", "the invention of the bicycle"]
HOLD_TASKS = ["merges overlapping intervals", "implements a bloom filter", "computes the edit distance between two strings"]

def label(vocab, tok, seg):
    a, e, ans, hdr = seg
    txt = decode(vocab, tok[a:hdr])
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", txt, re.S)
    p = m.group(1).strip() if m else txt
    return p
