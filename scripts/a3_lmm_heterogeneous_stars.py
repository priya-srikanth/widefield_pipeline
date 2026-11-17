"""Heterogeneous spread: y ~ C(epoch) + (1|animal) + animal x epoch variance SEPARATE per post epoch (and per-epoch
residual variance is NOT modeled -- statsmodels can't). Tests t on df = n_animals - 1 per epoch. Plus the summary-
statistic test: one-sample t on each animal's own change (mean post - mean pre), df = 3. Holm across 18 and within epoch."""
import sys, warnings
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings("ignore")
OUT = sys.argv[1]
L = pd.read_csv(OUT + "_long.csv")  # written by scripts/a3_lmm_slope_stars.py
PRE = L[(L.arm == "pre") & (L.epoch == "acute")]
POST = L[L.arm != "pre"]
EP = ["acute", "subacute", "chronic"]
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); a = np.empty(len(p)); r = 0
    for k, i in enumerate(o):
        r = max(r, (len(p) - k) * p[i]); a[i] = min(1, r)
    return a
rows = []
for (fam, cell), post in POST.groupby(["family", "cell"], sort=False):
    D = pd.concat([PRE[(PRE.family == fam) & (PRE.cell == cell)], post])[["arm", "animal", "y"]].reset_index(drop=True)
    vc = {}
    for e in EP:
        D[f"a_{e}"] = np.where(D.arm == e, D.animal, "none")
        vc[e] = f"0 + C(a_{e}, Treatment('none'))"
    f = "y ~ C(arm, Treatment('pre'))"
    m = None
    for meth in ("lbfgs", "powell", "nm"):
        try:
            m = smf.mixedlm(f, D, groups=D.animal, re_formula="1", vc_formula=vc).fit(reml=True, method=meth); break
        except Exception:
            continue
    vcs = dict(zip(m.model.exog_vc.names, m.vcomp))
    for e in EP:
        k = f"C(arm, Treatment('pre'))[T.{e}]"
        t = m.params[k] / m.bse[k]
        ch = [D[(D.animal == a) & (D.arm == e)].y.mean() - D[(D.animal == a) & (D.arm == "pre")].y.mean()
              for a in sorted(D.animal.unique())]
        tt = stats.ttest_1samp(ch, 0)
        rows.append(dict(family=fam, epoch=e, cell=cell, het_beta=m.params[k], het_se=m.bse[k], het_t=t,
                         het_p=2 * stats.t.sf(abs(t), 3), het_var_ae=vcs[e], het_singular=vcs[e] < 1e-4 * m.scale,
                         het_conv=bool(m.converged), changes=";".join(f"{c:+.3f}" for c in ch),
                         mean_change=np.mean(ch), t_changes=tt.statistic, p_changes=tt.pvalue))
T = pd.DataFrame(rows)
for col in ("het_p", "p_changes"):
    for _f, ix in T.groupby("family").groups.items():
        T.loc[ix, col + "_holm18"] = holm(T.loc[ix, col].values)
    for _k, ix in T.groupby(["family", "epoch"]).groups.items():
        T.loc[ix, col + "_holmEp"] = holm(T.loc[ix, col].values)
def star(p, h): return np.where(h < .05, "**", np.where(p < .05, "*", ""))
for col in ("het_p", "p_changes"):
    T["star_" + col + "_18"] = star(T[col], T[col + "_holm18"]); T["star_" + col + "_ep"] = star(T[col], T[col + "_holmEp"])
P = pd.read_csv(OUT + "_pooled.csv")[["family", "epoch", "cell", "star_boot_session_old", "star_boot_animal", "star_m2"]].fillna("")
T = P.merge(T, on=["family", "epoch", "cell"]); T.to_csv(OUT + "_het.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(T[["family","epoch","cell","changes","star_boot_session_old","star_boot_animal","star_m2","het_beta","het_p","het_var_ae","het_singular","star_het_p_18","star_het_p_ep","p_changes","star_p_changes_18","star_p_changes_ep"]].round(4).to_string())
