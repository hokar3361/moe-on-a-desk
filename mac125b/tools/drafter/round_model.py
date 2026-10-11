T_now=[17.9,23,25.8,29,31.5,34.3,37.5,41.0]
T_ideal=[11.63+1.29*i for i in range(8)]
def best(prof, T, ds=1.3, host=1.0, block=None, maxd=7):
    out=[]
    for d in range(0,maxd+1):
        E=1.0; p=1.0
        for k in range(d):
            p*=prof[min(k,len(prof)-1)]; E+=p
        dr = (block if block is not None else ds*d) if d>0 else 0
        t=T[d]+dr+host
        out.append((E/t*1000,d,E,t))
    return max(out)
def flat(a): return [a]
profiles={
 "P0 now prose [.72,.50..]":[0.72,0.50],
 "P0' [.72,.55..]":[0.72,0.55],
 "a-real [.74,.56..]":[0.74,0.56],
 "a-opt [.78,.65..]":[0.78,0.65],
 "a-qwen36conv [.81,.80..]":[0.81,0.80],
 "b-scratch-lowdata [.68,.58..]":[0.68,0.58],
 "b-litflat [.75,.72..]":[0.75,0.72],
 "flat .80":[0.8], "flat .85":[0.85], "flat .90":[0.9],"flat .95":[0.95],
}
for name,p in profiles.items():
    r=best(p,T_now); ri=best(p,T_ideal,ds=0.6,host=0.3)
    print("%-32s now %.1f tok/s (d=%d E=%.2f T=%.1f) | ideal-kernel %.1f (d=%d E=%.2f)"%(name,*r,ri[0],ri[1],ri[2]))
# (c) HyperDFlash-like: per-position profile such that E(6)=1.09*E_MTP(3) on P0, block draft cost 3 ms
import math
def E(prof,d):
    e=1;p=1
    for k in range(d): p*=prof[min(k,len(prof)-1)]; e+=p
    return e
tgt=1.09*E([0.72,0.5],3)
lo,hi=0.3,0.99
for _ in range(60):
    m=(lo+hi)/2
    if E([m],6)<tgt: lo=m
    else: hi=m
print("HyperDFlash-equiv flat a=%.3f E6=%.2f (MTP E3=%.2f)"%(m,E([m],6),E([0.72,.5],3)))
for blk in (2.0,3.0,4.0):
    r=best([m],T_now,block=blk); ri=best([m],T_ideal,block=blk*0.5,host=0.3)
    print(" block cost %.1f ms: now %.1f (d=%d E=%.2f) ideal %.1f"%(blk,r[0],r[1],r[2],ri[0]))
# (d) merged draft into verify: saves 1.0 ms per round
for name,p in [("P0",[0.72,0.5]),("a-real",[0.74,0.56])]:
    r=best(p,T_now,host=0.0); print("d-merge %s: %.1f (d=%d)"%(name,r[0],r[1]))
# n-gram mixture +0.015 everywhere
r=best([0.735,0.515],T_now); print("d-ngram +1.5pt: %.1f"%r[0])
# top-2 tree at pos1: B+1, pos1 acceptance .72->.84 but no deeper drafts on alt branch
# needed flat a for 145
for TT,ds,h,nm in [(T_now,1.3,1.0,"now"),(T_ideal,0.6,0.3,"ideal")]:
    for a in [x/100 for x in range(70,100)]:
        if best([a],TT,ds,h)[0]>=145: print("flat a for 145 (%s): %.2f"%(nm,a)); break
    else: print("145 unreachable (%s) for a<1"%nm)
print("a=0.99 now:",best([0.99],T_now))
print("---")
for p in ([0.72,0.6],[0.72,0.65],[0.72,0.72],[0.80,0.5],[0.85,0.5],[0.72,0.80]):
    r=best(p,T_now); ri=best(p,T_ideal,ds=0.6,host=0.3); print(p,"now %.1f d=%d E=%.2f | ideal %.1f"%(r[0],r[1],r[2],ri[0]))
# draft-cost sensitivity at P0 / a-opt: draft step 0.65 ms
for p in ([0.72,0.5],[0.78,0.65],[0.81,0.8]):
    print(p,"ds=0.65: %.1f"%best(p,T_now,ds=0.65)[0], "ds=2.6: %.1f"%best(p,T_now,ds=2.6)[0])
