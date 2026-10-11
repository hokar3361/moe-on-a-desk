# compare barrier-forcing node names between two barlog captures (numbers normalized), per graph
import sys, re, collections
def load(path, ngraphs):
    c = collections.Counter()
    for l in open(path, errors="replace"):
        if l.startswith("barlog "):
            c[re.sub(r"\d+", "N", l.strip()[7:])] += 1
    return c
a = load(sys.argv[1], 1); b = load(sys.argv[2], 1)
na = int(sys.argv[3]); nb = int(sys.argv[4])   # number of big graphs in each capture
print("barriers per graph: %.1f vs %.1f" % (sum(a.values())/na, sum(b.values())/nb))
keys = set(a) | set(b)
rows = sorted(((b[k]/nb - a[k]/na), k) for k in keys)
for d, k in rows[:12] + rows[-12:]:
    if abs(d) >= 0.5:
        print("%+7.1f  %6.1f -> %6.1f  %s" % (d, a[k]/na, b[k]/nb, k))
