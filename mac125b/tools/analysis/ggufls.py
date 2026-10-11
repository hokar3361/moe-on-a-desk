# minimal GGUF tensor lister (no numpy): prints per (role,type) count and bytes, and blk.3 tensors
import struct, sys, collections, re
TN = {0:"f32",1:"f16",2:"q4_0",3:"q4_1",6:"q5_0",7:"q5_1",8:"q8_0",9:"q8_1",10:"q2_K",11:"q3_K",12:"q4_K",13:"q5_K",14:"q6_K",15:"q8_K",
      16:"iq2_xxs",17:"iq2_xs",18:"iq3_xxs",19:"iq1_s",20:"iq4_nl",21:"iq3_s",22:"iq2_s",23:"iq4_xs",24:"i8",25:"i16",26:"i32",27:"i64",28:"f64",29:"iq1_m",30:"bf16",34:"tq1_0",35:"tq2_0",39:"mxfp4"}
BS = {"f32":(1,4),"f16":(1,2),"bf16":(1,2),"q4_0":(32,18),"q4_1":(32,20),"q5_0":(32,22),"q5_1":(32,24),"q8_0":(32,34),"q2_K":(256,84),"q3_K":(256,110),
      "q4_K":(256,144),"q5_K":(256,176),"q6_K":(256,210),"iq2_xxs":(256,66),"iq2_xs":(256,74),"iq3_xxs":(256,98),"iq1_s":(256,50),"iq4_nl":(32,18),
      "iq3_s":(256,110),"iq2_s":(256,82),"iq4_xs":(256,136),"iq1_m":(256,56),"mxfp4":(32,17)}
def rd(f, fmt):
    n = struct.calcsize(fmt); return struct.unpack("<" + fmt, f.read(n))
def rstr(f):
    (n,) = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
def skipval(f, t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sz: f.read(sz[t])
    elif t == 8: rstr(f)
    elif t == 9:
        (at,) = rd(f, "I"); (n,) = rd(f, "Q")
        for _ in range(n): skipval(f, at)
agg = collections.defaultdict(lambda: [0, 0])
for path in sys.argv[1:]:
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        (ver,) = rd(f, "I"); (nt,) = rd(f, "Q"); (nkv,) = rd(f, "Q")
        for _ in range(nkv):
            rstr(f); (t,) = rd(f, "I"); skipval(f, t)
        for _ in range(nt):
            name = rstr(f); (nd,) = rd(f, "I"); dims = rd(f, "Q"*nd); (tt,) = rd(f, "I"); rd(f, "Q")
            tn = TN.get(tt, str(tt)); ne = 1
            for d in dims: ne *= d
            b, s = BS.get(tn, (1, 0)); nbytes = ne // b * s
            role = re.sub(r"blk\.\d+\.", "", name)
            agg[(role, tn)][0] += 1; agg[(role, tn)][1] += nbytes
            if name.startswith("blk.3.") or name.startswith("blk.0.") or not name.startswith("blk."):
                print(name, tn, dims)
print("---- role type count GiB")
for (role, tn), (c, nb) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print("%-40s %-8s %4d %8.3f" % (role, tn, c, nb / 2**30))
