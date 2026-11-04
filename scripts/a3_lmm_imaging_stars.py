"""Heterogeneous-spread LMM (per post epoch vs pre: y ~ C(epoch) + (1|animal) + animal x epoch variance per epoch,
t on df = n_animals - 1) on every epoch bar family's stored per-session values (<name>_sessions.csv), compared with
the stored mark (<name>.csv 'mark', the pre-A3 session-weighted bootstrap). Holm across all post cells of a figure.
Reads only; nothing written to the share."""
import sys, glob, os, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")
DATA, OUT = sys.argv[1], sys.argv[2]
EP = ["acute", "subacute", "chronic"]
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty(len(p)); r = 0
    for k, i in enumerate(o):
        r = max(r, (len(p) - k) * p[i]); a[i] = min(1, r)
    return a
def het(D):
    D = D.reset_index(drop=True).copy(); vc = {}
    eps = [e for e in EP if (D.epoch == e).any()]
    for e in eps:
        D[f"a_{e}"] = np.where(D.epoch == e, D.animal, "none"); vc[e] = f"0 + C(a_{e}, Treatment('none'))"
    for meth in ("lbfgs", "powell", "nm"):
        try:
            return smf.mixedlm("value ~ C(epoch, Treatment('pre'))", D, groups=D.animal, re_formula="1",
                               vc_formula=vc).fit(reml=True, method=meth), eps
        except Exception:
            continue
    return None, eps
rows = []
for f in sorted(glob.glob(os.path.join(DATA, "*_sessions.csv"))):
    name = os.path.basename(f)[:-len("_sessions.csv")]
    main = os.path.join(DATA, name + ".csv")
    if "delta" in name or not os.path.exists(main):
        continue
    S = pd.read_csv(f); M = pd.read_csv(main)
    if not {"epoch", "position", "animal", "value"} <= set(S) or "mark" not in M:
        continue
    M["mark"] = M["mark"].fillna("").astype(str)
    for pos, D in S.groupby("position", sort=False):
        D = D[np.isfinite(D.value)]
        if (D.epoch == "pre").sum() < 2:
            continue
        m, eps = het(D)
        for e in eps:
            Dp = D[D.epoch.isin(["pre", e])]
            both = set(Dp[Dp.epoch == "pre"].animal) & set(Dp[Dp.epoch == e].animal)
            old = M[(M.epoch == e) & (M.position.astype(str) == str(pos))].mark
            r = dict(figure=name, position=pos, epoch=e, n_animals=len(both), n_sess_post=int((D.epoch == e).sum()),
                     star_old=old.iloc[0] if len(old) else "")
            if m is not None and len(both) >= 2:
                k = f"C(epoch, Treatment('pre'))[T.{e}]"
                t = m.params[k] / m.bse[k]
                r.update(beta=m.params[k], se=m.bse[k], p=2 * stats.t.sf(abs(t), len(both) - 1), converged=m.converged)
            rows.append(r)
T = pd.DataFrame(rows)
T["p_holm"] = np.nan
for _fig, ix in T.groupby("figure").groups.items():
    ok = [i for i in ix if np.isfinite(T.loc[i, "p"])]
    if ok:
        T.loc[ok, "p_holm"] = holm(T.loc[ok, "p"].values)
T["star_lmm"] = np.where(T.p_holm < .05, "**", np.where(T.p < .05, "*", ""))
T.loc[T.p.isna(), "star_lmm"] = "n/a"
T.to_csv(OUT, index=False)
sig = lambda s: s.isin(["*", "**"])
T["fam"] = T.figure.str.replace("^epoch_", "", regex=True)
summ = T.groupby("figure").apply(lambda g: pd.Series(dict(cells=len(g), old_any=int(sig(g.star_old).sum()),
    old_2=int((g.star_old == "**").sum()), lmm_any=int(sig(g.star_lmm).sum()), lmm_2=int((g.star_lmm == "**").sum()),
    lost=int((sig(g.star_old) & ~sig(g.star_lmm)).sum()), gained=int((~sig(g.star_old) & sig(g.star_lmm)).sum()))))
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 300)
print(summ.to_string()); print("TOTAL", summ.sum().to_dict())
