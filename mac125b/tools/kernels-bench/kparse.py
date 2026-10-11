import sys, re, statistics
gpu=[]; res=None
for l in sys.stdin:
    m=re.search(r"gpu_sum=([\d.]+)", l)
    if m: gpu.append(float(m.group(1))*1000.0)
    if l.startswith("RESULT"): res=l.split()
    if l.startswith("check"): print(l.strip())
if res:
    mode,t,K,N,B,L,IT,by,wall=res[1],res[2],int(res[3]),int(res[4]),int(res[5]),int(res[6]),int(res[7]),float(res[8]),float(res[9])
    g=statistics.median(gpu[-IT:]) if len(gpu)>=IT else wall
    us=g/L
    print("%s %s K=%d N=%d B=%d: %.2f us/op (gpu), %.0f GB/s, wall %.1f us/op" % (mode,t,K,N,B,us,by/(us*1e3),wall/L))
