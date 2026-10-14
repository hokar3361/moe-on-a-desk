# offline simulation of MoE cache replacement policies on a routing trace from LLAMA_MOE_TRACE
# misses = distinct experts of a (forward, layer) not in the cache of its group; each group has its own capacity (n_slots of its LRU)
import sys, collections, heapq, math

path = sys.argv[1]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # capacity multiplier
groups = {}
events = []  # (fwd, il, g, [experts])
for line in open(path):
    if line.startswith('# group'):
        t = line.split()
        groups[int(t[2])] = int(int(t[3]) * scale)
        continue
    t = line.split()
    events.append((int(t[0]), int(t[1]), int(t[2]), [int(x) for x in t[3:]]))
n_req = sum(len(e[3]) for e in events)
print(f'{len(events)} layer-forwards, {n_req} expert uses, {len(groups)} groups, capacity {sum(groups.values())} slots (x{scale})')

def run(policy_factory, name):
    caches = {g: policy_factory(cap) for g, cap in groups.items()}
    miss = 0
    for idx, (f, il, g, ex) in enumerate(events):
        miss += caches[g].access(idx, il, ex)
    print(f'{name:34s} misses {miss:7d}  hit {100*(1-miss/n_req):5.1f}%  misses per layer-forward {miss/len(events):.2f}')
    return miss

class LRU:
    def __init__(self, cap): self.cap = cap; self.od = collections.OrderedDict()
    def access(self, t, il, ex):
        m = 0
        for e in ex:
            k = (il, e)
            if k in self.od: self.od.move_to_end(k)
            else:
                m += 1; self.od[k] = 1
                if len(self.od) > self.cap: self.od.popitem(last=False)
        return m

class LFUDecay:
    # score = decayed use count; evict the lowest score (ties: least recent)
    def __init__(self, cap, half=2000):
        self.cap = cap; self.score = {}; self.last = {}; self.lam = math.log(2) / half; self.t = 0
    def val(self, k): return self.score[k] * math.exp(-self.lam * (self.t - self.last[k]))
    def access(self, t, il, ex):
        self.t = t; m = 0
        for e in ex:
            k = (il, e)
            if k in self.score:
                self.score[k] = self.val(k) + 1; self.last[k] = t
            else:
                m += 1
                if len(self.score) >= self.cap:
                    victim = min(self.score, key=lambda x: (self.val(x), self.last[x]))
                    del self.score[victim]; del self.last[victim]
                self.score[k] = 1; self.last[k] = t
        return m

class Belady:
    def __init__(self, cap): self.cap = cap; self.cache = set(); self.heap = []
    def access(self, t, il, ex):
        m = 0
        for e in ex:
            k = (il, e)
            nu = next_use[(t, k)]
            if k not in self.cache:
                m += 1
                if len(self.cache) >= self.cap:
                    while True:
                        negn, kk = heapq.heappop(self.heap)
                        if kk in self.cache and cur_next[kk] == -negn:
                            self.cache.remove(kk); break
                self.cache.add(k)
            cur_next[k] = nu
            heapq.heappush(self.heap, (-nu, k))
        return m

# next use of each (event index, key), per group
next_use = {}
last_seen = {}
for idx in range(len(events) - 1, -1, -1):
    f, il, g, ex = events[idx]
    for e in ex:
        k = (il, e)
        next_use[(idx, k)] = last_seen.get(k, 10**12)
        last_seen[k] = idx
cur_next = {}

base = run(LRU, 'LRU')
for half in (500, 3000):
    run(lambda c, h=half: LFUDecay(c, h), f'LFU decayed (half-life {half} layer-fwd)')
run(Belady, 'Belady (optimal, needs the future)')
