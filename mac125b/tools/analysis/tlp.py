import re, sys, statistics as st, collections
rows = []
seg = 0
for l in open(sys.argv[1]):
    m = re.search(r"nodes=(\d+) .*encode=([\d.]+) gpu_span=([\d.]+) .*wall=([\d.]+) ms t=([\d.]+)", l)
    if m and rows and float(m[5]) - rows[-1][5] > 1000: seg += 1
    if m: rows.append((seg, int(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])))
for s in sorted(set(x[0] for x in rows)):
    r = [x for x in rows if x[0] == s]
    # verify graphs: nodes > 3000
    vi = [i for i, x in enumerate(r) if x[1] > 3000]
    vi = vi[1:]  # skip prompt
    per = collections.defaultdict(list)
    for a, b in zip(vi, vi[1:]):
        wall = r[b][5] - r[a][5]
        gv = r[a][3]; gd = sum(x[3] for x in r[a+1:b]); nd = b - a - 1
        enc = sum(x[2] for x in r[a:b])
        per[r[a][1]].append((wall, gv, gd, nd, enc, r[a][4]))
    print("segment", s)
    for k in sorted(per):
        v = per[k]
        f = lambda j: st.median([x[j] for x in v])
        print("  verify nodes %5d n=%3d  round wall %.2f | verify gpu %.2f wall %.2f | drafts %d gpu %.2f | encode %.2f | gap %.2f" % (
            k, len(v), f(0), f(1), f(5), f(3), f(2), f(4), f(0) - f(1) - f(2)))
