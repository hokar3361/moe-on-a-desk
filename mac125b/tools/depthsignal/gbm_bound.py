# upper-bound check: a nonlinear model (gradient boosting) on S1 + all features2 groups (incl. MTP tail mass, verifier);
# if this does not move AUC / replay, no logistic feature engineering on the same information will.
# usage: python gbm_bound.py TRAIN.log HELD.log
import sys, math, numpy as np
import features2 as F
from sklearn.ensemble import HistGradientBoostingClassifier
A = F.A

class GB:
    def __init__(self, m): self.m = m
    def predict(self, X): return self.m.predict_proba(np.atleast_2d(np.asarray(X, float)))[:, 1]
    def p1(self, x): return float(self.predict([x])[0])


def main():
    train = A.parse(sys.argv[1]); test = A.parse(sys.argv[2])
    ext_tr = F.text_features(train); ext_te = F.text_features(test)
    s1post, _ = F.base_models(train)
    cost = A.Cost(A.make_tv("1:17.9,3:25.8,6:34.3,8:41"), 1.3, 1.0, 0.5); cap = 7
    fixed = max(1000*(lambda r: r[0]/r[1])(A.run_policy(test, cost, cap, A.Fixed(d))) for d in range(1, cap + 1))
    t, m, _, _ = A.run_policy(test, cost, cap, A.DepthCtlFirstSkip(cap)); r_ctl = 1000*t/m
    for groups in ((), F.GROUPS_ALL):
        Xs = {}
        for post in (True, False):
            Xtr, ytr, _ = F.build_train_crossfit(train, ext_tr, groups, s1post, post)
            tabs = F.make_tables(train, s1post) if "tok" in groups else (None, None)
            Xte, yte, kte = F.build(test, ext_te, tabs, groups, s1post, post)
            Xs[post] = (Xtr, ytr, Xte, yte, kte, tabs)
        Xtr, ytr, Xte, yte, kte, tabs = Xs[True]
        mpost = GB(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(Xtr, ytr))
        Xpr, ypr, Xtp, ytp, _, _ = Xs[False]
        mpre = GB(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(Xpr, ypr))
        pp = mpost.predict(Xte)
        aucs = F.per_pos_auc(pp, yte, kte)
        opt = A.OptionTable(mpre.predict(Xpr), mpost.predict(Xtr))
        qb = A.qbar_from(train, cap)
        kw = dict(ext=ext_te, tabs=tabs, groups=groups, s1post=s1post)
        lam = fixed/1000.0
        for _ in range(4):
            tok, ms, nr, ds = A.run_policy(test, cost, cap, F.Stop2(cap, cost, mpre, mpost, A.FEATURE_SETS["S1_+mtp"], qb, lam, dmin=1, opt=opt, **kw)); lam = tok/ms
        tok, ms, nr, ds = A.run_policy(test, cost, cap, F.Stop2(cap, cost, mpre, mpost, A.FEATURE_SETS["S1_+mtp"], qb, lam, dmin=1, opt=opt, **kw))
        r = 1000*tok/ms
        print("GBM S1%s: AUC " % ("+" + "+".join(groups) if groups else "") + " ".join("%.3f" % v for v in aucs) +
              "  pre %.3f  lam* %.1f (%+.1f%% vs ctl %.1f)" % (A.auc(mpre.predict(Xtp), ytp), r, 100*(r/r_ctl - 1), r_ctl))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
