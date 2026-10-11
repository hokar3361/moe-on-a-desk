import re, sys, statistics as st, collections
rows = []
for l in open(sys.argv[1]):
    m = re.search(r"nodes=(\d+) .*encode=([\d.]+) gpu_span=([\d.]+) .*wall=([\d.]+) ms t=([\d.]+)", l)
    if m: rows.append((int(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])))
vi = [i for i, x in enumerate(rows) if x[0] > 3000][1:]
per = collections.defaultdict(list)
for a, b in zip(vi, vi[1:]):
    seq = rows[a:b+1]
    # gaps between consecutive graphs: next.t - (cur.t + cur.wall)   (t assumed = start)
    gaps = [seq[i+1][4] - (seq[i][4] + seq[i][3]) for i in range(len(seq)-1)]
    per[(rows[a][0], b-a-1)].append(gaps)
for k in sorted(per):
    v = per[k]
    if len(v) < 5: continue
    med = [st.median(g[i] for g in v) for i in range(len(v[0]))]
    print(k, len(v), " ".join("%.2f" % x for x in med))
