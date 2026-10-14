import sys, collections
groups = {}; events = []
for line in open(sys.argv[1]):
    if line.startswith('# group'):
        t = line.split(); groups[int(t[2])] = int(t[3]); continue
    t = line.split(); events.append((int(t[1]), int(t[2]), [int(x) for x in t[3:]]))
n = len(events)
for scale in (0.2, 0.3, 0.4, 0.5, 0.65, 1.0):
    caches = {g: collections.OrderedDict() for g in groups}
    miss = 0
    for il, g, ex in events:
        od = caches[g]; cap = max(12, int(groups[g] * scale))
        for e in ex:
            k = (il, e)
            if k in od: od.move_to_end(k)
            else:
                miss += 1; od[k] = 1
                if len(od) > cap: od.popitem(last=False)
    print(f'cache {sum(groups.values())*scale*1.87/1024:5.1f} GiB: {miss/n:.2f} misses per layer-forward')
