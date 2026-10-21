"""Mixed-effects time course of the POSITION encoding measures (Priya 10-20: "what about mixed model for position /
distance? are we using random intercept and random slope per animal").

Outcomes per session x area x window (dR2 x1e3, cortex + both hemispheres of the focus areas):
  position       full - drop:position                  (movement_first; order-free drop, the lead measure)
  side           full - shufside:position               (labels permuted WITHIN distance -> side information)
  distance       full - shufdist:position               (labels permuted WITHIN side -> distance information)
  spout:<p>      full - droppos-<p>:position            (movement_first_subset-droppos-; per-spout unique)
  far-close      mean(far spouts) - mean(close spouts)  of the per-spout values (does distance coding change?)
  contra-ipsi    mean(contra spouts) - mean(ipsi spouts)

Model: y ~ post + post:lday, sessions as observations, post = after the stroke, lday = log(days since stroke) (0 at
day 1) -> `post` = day-1 step vs pre, `post:lday` = change per e-fold of days. Random effects per animal, tried in this
order and the first that converges is reported (column `re`):
  ~post + post:lday   random intercept + random step + random recovery SLOPE
  ~post               random intercept + random step
  ~1                  random intercept only
  fixed               (last resort) OLS with fixed, sum-coded animal intercepts, when every random structure is singular
With 4 animals the random-slope variance is weakly identified (often singular -> fallback); the per-animal deviations
(BLUPs) are written out so animal-specific courses stay visible. REML, unweighted. FDR over measures within window x term.
Output: <run>/imaging_encoding/qc/encoding_mixed.csv. Curated sessions from 08-06 (August pre baseline).
"""
from __future__ import annotations

import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FOCUS = ("cortex", "SSp-m_left", "SSp-m_right", "MOs_left", "MOs_right", "MOp_left", "MOp_right", "SSp-ul_left",
         "SSp-ul_right")
WINDOWS = ("enl", "cue", "lick")
RES = ("~post + post:lday", "~post", "~1")


def _regions(folder: Path, labels: set) -> pd.DataFrame:
    fs = [f for f in glob.glob(str(folder / "sessions" / "*_regions.csv")) if Path(f).stem[:-8] in labels]
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return R[R.area.isin(FOCUS) & R.window.isin(WINDOWS)]


def outcomes(E: Path, labels: set) -> pd.DataFrame:
    rows = []
    R = _regions(E / "movement_first", labels)
    P = R.pivot_table(index=["label", "window", "area"], columns="model", values="r2")
    for name, m in (("position", "drop:position"), ("side", "shufside:position"), ("distance", "shufdist:position")):
        if m in P:
            rows.append(((P["full"] - P[m]) * 1e3).rename("y").reset_index().assign(measure=name))
    D = _regions(E / "movement_first_subset-droppos-", labels)
    Q = D.pivot_table(index=["label", "window", "area"], columns="model", values="r2")
    sp = {}
    for c in Q.columns:
        if c.startswith("droppos-") and c.endswith(":position"):
            p = c[len("droppos-"):-len(":position")]
            sp[p] = (Q["full"] - Q[c]) * 1e3
            rows.append(sp[p].rename("y").reset_index().assign(measure=f"spout:{p}"))
    if sp:
        mean = lambda keys: pd.concat([sp[k] for k in keys if k in sp], axis=1).mean(1)   # noqa: E731
        far, close = [k for k in sp if k.startswith("far")], [k for k in sp if k.startswith("close")]
        contra, ipsi = [k for k in sp if k.endswith("contra")], [k for k in sp if k.endswith("ipsi")]
        rows.append((mean(far) - mean(close)).rename("y").reset_index().assign(measure="far-close"))
        rows.append((mean(contra) - mean(ipsi)).rename("y").reset_index().assign(measure="contra-ipsi"))
    # ANGLE contest (Priya 10-20: "try this for ... position (or angle) analysis"): nokin target / executed measures,
    # from the cam4 contest folder and, when present, the 3-D polar contest (combined licks, soft bins)
    for tag, fold in (("cam4", "movement_first_subset-nokin"),
                      ("3d", "movement_first_subset-nokin_exec-polar_licks-combined_bins")):
        if not (E / fold / "sessions").exists():
            continue
        N = _regions(E / fold, labels).pivot_table(index=["label", "window", "area"], columns="model", values="r2")
        for name, (hi, lo) in (("ceiling", ("category", "none")), ("u_target", ("target+exec", "exec")),
                               ("u_exec", ("target+exec", "target")), ("u_exec_strict", ("category+exec", "category")),
                               ("u_tgtbin", ("tgtbin+exbin", "exbin")), ("u_exbin", ("tgtbin+exbin", "tgtbin"))):
            if f"nokin:{hi}" in N and f"nokin:{lo}" in N:
                rows.append(((N[f"nokin:{hi}"] - N[f"nokin:{lo}"]) * 1e3).rename("y").reset_index()
                            .assign(measure=f"{tag}:{name}"))
    L = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
    from wfield_local import epochs
    L["animal"] = L.label.str[:4]
    L["epoch"] = L.label.map(epochs.epoch_of)
    L["day"] = L.label.map(epochs.days_since_stroke)
    L["post"] = (L.day > 0).astype(float)
    L["lday"] = np.where(L.day > 0, np.log(L.day.clip(lower=1)), 0.0)
    return L



def _re_ok(r) -> bool:
    """Random effects predictable (a singular RE covariance makes statsmodels raise) -> else fall back."""
    try:
        _ = r.random_effects
        return True
    except Exception:                                       # noqa: BLE001
        return False

def fit(G: pd.DataFrame):
    import statsmodels.formula.api as smf
    for re in RES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = smf.mixedlm("y ~ post + post:lday", G, groups=G["animal"], re_formula=re).fit(reml=True)
            except Exception:                                   # noqa: BLE001 -- singular / non-converged
                continue
        if r.converged and np.all(np.isfinite(r.bse[r.fe_params.index])) and _re_ok(r):
            return r, re
    # LAST RESORT: fixed animal intercepts (sum-coded, so Intercept = mean of the animal levels) -- used when even the
    # random intercept is singular (animals with near-identical baselines)
    r = smf.ols("y ~ C(animal, Sum) + post + post:lday", G).fit()
    r.fe_params = r.params
    return r, "fixed"


def main(argv=None) -> int:
    from statsmodels.stats.multitest import fdrcorrection
    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    a = ap.parse_args(argv)
    E = oo.root(PathResolver()) / a.run / "imaging_encoding"
    labels = {x for x in ak.curated_labels() if x[-4:] >= "0806"}
    L = outcomes(E, labels)
    rows = []
    for (meas, win, area), G in L.groupby(["measure", "window", "area"]):
        if (G.post == 0).sum() < 6 or (G.post == 1).sum() < 10 or G.animal.nunique() < 2:
            continue                                    # too few pre / post sessions (e.g. a run still in progress)
        r, re = fit(G)
        if r is None:
            rows.append({"measure": meas, "window": win, "area": area, "re": re, "n": len(G)})
            continue
        rfx = r.random_effects if re != "fixed" else {}
        blup = {f"step_dev_{an}": float(v.get("post", np.nan)) for an, v in rfx.items()}
        blup.update({f"slope_dev_{an}": float(v.get("post:lday", np.nan)) for an, v in rfx.items()})
        for term in ("Intercept", "post", "post:lday"):
            rows.append({"measure": meas, "window": win, "area": area, "re": re, "n": len(G),
                         "term": "pre_level" if term == "Intercept" else term, "coef": r.fe_params[term],
                         "se": r.bse[term], "z": r.tvalues[term], "p": r.pvalues[term], **blup})
    T = pd.DataFrame(rows)
    T["q"] = np.nan
    for _, idx in T.dropna(subset=["p"]).groupby(["window", "term"]).groups.items():
        T.loc[idx, "q"] = fdrcorrection(T.loc[idx, "p"])[1]
    T.to_csv(E / "qc" / "encoding_mixed.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 230)
    for win in WINDOWS:
        S = T[(T.window == win) & (T.area == "cortex") & T.term.notna()]
        print(f"\n=== {win} window, cortex: dR2 x1e3; y ~ post + post:lday; q = FDR over measures x areas")
        print(S.pivot_table(index="measure", columns="term", values=["coef", "z", "q"]).round(3).to_string())
    print("\n  random-effect structure used:", T.groupby("re").size().to_dict())
    print(f"\n-> {E / 'qc' / 'encoding_mixed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
