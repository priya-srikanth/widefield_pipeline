"""Pooled-epoch LMMs per cell (all four epochs in one model), session rows:
  M1  y ~ C(epoch) + (1 | animal)
  M2  y ~ C(epoch) + (1 | animal) + (1 | animal:epoch)    one animal-x-epoch variance shared by the 3 post epochs
statsmodels MixedLM REML; epoch tests t on containment df = (n_animals - 1) * (n_epochs - 1) = 9 for M2.
Singular M2 = animal:epoch variance < 1e-4 * residual."""
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
OUT = sys.argv[1]
L = pd.read_csv(OUT + "_long.csv")  # written by scripts/a3_lmm_slope_stars.py
PRE = L[(L.arm == "pre") & (L.epoch == "acute")]
POST = L[L.arm != "pre"]
EP = ["acute", "subacute", "chronic"]


def holm(p):
    p = np.asarray(p, float)
    o, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


rows = []
for (fam, cell), post in POST.groupby(["family", "cell"], sort=False):
    D = pd.concat([PRE[(PRE.family == fam) & (PRE.cell == cell)], post])[["arm", "animal", "y"]]
    D["ae"] = D.animal + "_" + D.arm
    f = "y ~ C(arm, Treatment('pre'))"
    m1 = smf.mixedlm(f, D, groups=D.animal).fit(reml=True)
    m2 = None
    for meth in ("lbfgs", "powell", "nm"):
        try:
            m2 = smf.mixedlm(f, D, groups=D.animal, re_formula="1",
                             vc_formula={"ae": "0 + C(ae)"}).fit(reml=True, method=meth)
            break
        except Exception:  # noqa: BLE001
            continue
    na = D.animal.nunique()
    df2 = (na - 1) * (D.arm.nunique() - 1)
    vc = float(m2.vcomp[0])
    for e in EP:
        k = f"C(arm, Treatment('pre'))[T.{e}]"
        t2 = m2.params[k] / m2.bse[k]
        rows.append(dict(family=fam, epoch=e, cell=cell,
                         m1_beta=m1.params[k], m1_p=m1.pvalues[k],
                         m2_beta=m2.params[k], m2_se=m2.bse[k], m2_t=t2, m2_df=df2,
                         m2_p=2 * stats.t.sf(abs(t2), df2), var_animal=float(m2.cov_re.iloc[0, 0]),
                         var_animal_epoch=vc, var_resid=m2.scale, m2_singular=vc < 1e-4 * m2.scale,
                         m2_converged=bool(m2.converged)))
T = pd.DataFrame(rows)
for _f, idx in T.groupby("family").groups.items():
    T.loc[idx, "m2_holm"] = holm(T.loc[idx, "m2_p"].values)
    T.loc[idx, "m1_holm"] = holm(T.loc[idx, "m1_p"].values)
T["star_m2"] = np.where(T.m2_holm < 0.05, "**", np.where(T.m2_p < 0.05, "*", ""))
T["star_m1"] = np.where(T.m1_holm < 0.05, "**", np.where(T.m1_p < 0.05, "*", ""))
S = pd.read_csv(OUT + "_slope.csv")[["family", "epoch", "cell", "star_boot_session_old", "star_boot_animal",
                                     "star_lmm_df3"]].fillna("")
T = S.merge(T, on=["family", "epoch", "cell"])
T.to_csv(OUT + "_pooled.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(T.round(4).to_string())
