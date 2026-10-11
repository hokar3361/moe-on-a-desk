# fit the S1 (MTP-only) post/pre logistic models for the C++ stop rule; prints a header-style table
# usage: python fit_s1.py train.log held.log
import sys, importlib.util, os
spec = importlib.util.spec_from_file_location("analyze", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
import numpy as np

tr = A.parse(sys.argv[1]); te = A.parse(sys.argv[2])
fs = A.FEATURE_SETS["S1_+mtp"]
for post in (True, False):
    X, y, kk = A.build_samples(tr, fs, post=post)
    m = A.Logit().fit(X, y)
    Xt, yt, kt = A.build_samples(te, fs, post=post)
    print("%s n=%d auc_test=%.3f" % ("post" if post else "pre", len(y), A.auc(m.predict(Xt), yt)))
    print("  mu", " ".join("%.6g" % v for v in m.mu))
    print("  sd", " ".join("%.6g" % v for v in m.sd))
    print("  w ", " ".join("%.6g" % v for v in m.w))
qb = A.qbar_from(tr, 7)
print("qbar", " ".join("%.4f" % qb[k] for k in range(1, 8)))
