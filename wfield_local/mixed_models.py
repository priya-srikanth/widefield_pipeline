"""Session-level linear mixed models for post-stroke time courses -- the one place they are defined.

WHY THIS EXISTS (Priya 2026-10-20: "should we be using the lmm approach for the imaging analyses too (as opposed to
epochs)" -> "take a close look at the current imaging analyses - decide where lmm approach would make sense and prepare
code to do so, matching the repo conventions and avoiding redundancy"). Epoch contrasts compare different post-stroke
times across animals: the chronic epoch starts at day 11 (PS92 / PS93), 15 (PS95) or 25 (PS94), and acute spans days
1-7 with large changes inside it. A model with each session at its own day -- or at its own behavioural recovery --
uses every session and asks the questions directly (DECISIONS 2026-10-20 night: mixed models are the primary readout,
epochs descriptive).

THE MODELS. One row per session (x whatever the caller groups by), sessions as observations, animal as the grouping
factor:
  time       y ~ post + post:lday        post = 1 after the stroke; lday = log(days since stroke), 0 at day 1
             -> `post` = the day-1 step vs pre, `post:lday` = change per e-fold of days (fast early, slow late)
  behaviour  y ~ post + post:deficit     deficit = (animal's August-pre mean - session) / pre mean of a behavioural
             recovery measure (default far-contra contact), 0 = pre level, 1 = total loss, 0 for pre sessions
             -> `post` = change still present at FULL behavioural recovery, `post:deficit` = change per unit deficit
             (recovery TIMING differs per animal; indexing by behaviour aligns it -- Priya 10-20)
  epoch      y ~ C(epoch)                pre reference; for comparison with the epoch tables
RANDOM EFFECTS, tried in order, first that converges with ALL standard errors finite (variance components too) and
predictable random effects wins
(reported in `re`): intercept + step + slope (Priya 10-20: "are we using random intercept and random slope per
animal"), intercept + step, intercept; LAST RESORT fixed sum-coded animal intercepts (OLS; `re` = "fixed") when even
the random intercept is singular (animals with near-identical baselines). With 4 animals the random-slope variance is
weakly identified -- it converged in ~40 % of the encoding fits; per-animal deviations (BLUPs) are returned so
animal-specific courses stay visible.

UNIT AND BASELINE. Sessions are observations within animals; inference is about the cohort of sessions given the
animals, not a population of animals (n = 4). Pre = curated AUGUST sessions only (the primary baseline, CLAUDE.md);
sessions whose epoch is None (excluded PS92 / PS93 0817, gap days) are DROPPED before `post` is set -- otherwise
they would count as pre. RULE 8: `lmm_table` prints the per-animal epoch means before any fit.

WHAT IS DELIBERATELY NOT HERE. Pixel maps (keep `beta_maps.significance_contour` / `vs_zero_contour`; a per-pixel
fixed-intercept OLS can be drawn as a descriptive companion only); serial correlation between neighbouring sessions
(not modelled -- z-values are somewhat optimistic; check by per-animal summaries); FDR is `orofacial_stats.bh_fdr`
(not a second implementation). Callers: scripts/encoding_mixed.py, scripts/within_spout_mixed.py, scripts/imaging_lmm.py.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

#: slope term per model
SLOPES = {"time": "post:lday", "behaviour": "post:deficit"}
#: minimum sessions to fit a group (fewer = a run still in progress, or a measure only some sessions carry)
MIN_PRE, MIN_POST, MIN_ANIMALS = 6, 10, 2
EPOCHS = ("pre", "acute", "subacute", "chronic")


def add_time_covariates(L: pd.DataFrame, label_col: str = "label", august_pre: bool = True) -> pd.DataFrame:
    """+ animal, epoch, day, post, lday; drops sessions with no epoch (excluded / gap days) and, by default, June pre."""
    from wfield_local import epochs
    L = L.copy()
    L["animal"] = L[label_col].str[:4]
    L["epoch"] = L[label_col].map(epochs.epoch_of)
    L = L[L.epoch.notna()]
    if august_pre:
        L = L[(L.epoch != "pre") | (L[label_col].str[-4:] >= "0806")]
    L["day"] = L[label_col].map(epochs.days_since_stroke).astype(float)
    L["post"] = (L.epoch != "pre").astype(float)
    L["lday"] = np.where(L.post == 1, np.log(L.day.clip(lower=1)), 0.0)
    return L.reset_index(drop=True)


def behaviour_deficit(sessions_csv: Path, column: str = "contact_far_contra") -> dict:
    """{label: deficit} = (animal's August-pre mean - session) / pre mean of `column` (default far-contra contact,
    from qc/position_coding_trend_recovery_sessions.csv, scripts/position_coding_trend_recovery.py)."""
    S = pd.read_csv(sessions_csv)
    pre = S[(S.epoch == "pre") & (S.label.str[-4:] >= "0806")].groupby("animal")[column].mean()
    return dict(zip(S.label, (S.animal.map(pre) - S[column]) / S.animal.map(pre)))


def add_deficit(L: pd.DataFrame, deficit: dict, label_col: str = "label") -> pd.DataFrame:
    """+ deficit (0 for pre sessions; NaN where the session has no behavioural value)."""
    L = L.copy()
    L["deficit"] = np.where(L.post == 0, 0.0, L[label_col].map(deficit))
    return L


def _re_ok(r) -> bool:
    """Random effects predictable (a singular RE covariance makes statsmodels raise on access)."""
    try:
        _ = r.random_effects
        return True
    except Exception:                                       # noqa: BLE001
        return False


def fit_lmm(G: pd.DataFrame, formula: str, slope: str | None = None, fallback: bool = True,
            ladder: tuple | None = None):
    """One group: `y ~ <rhs>` with the random-effects ladder (see module docstring). Returns (result, re) or
    (None, "failed") when nothing converges and `fallback` is False. `slope` = the term given a random slope;
    `ladder` overrides the structures tried (e.g. ("~lday", "~1") for post-stroke-only tables, where `post` is
    constant)."""
    import statsmodels.formula.api as smf
    ladder = ladder or (([f"~post + {slope}"] if slope else []) + ["~post", "~1"])
    for re in ladder:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = smf.mixedlm(formula, G, groups=G["animal"], re_formula=re).fit(reml=True)
            except Exception:                                   # noqa: BLE001 -- singular / non-converged
                continue
        # ALL standard errors finite, variance components included: a NaN SE on a variance parameter = the random
        # slope / step collapsed onto the boundary -> the simpler structure is the right one (10-20; the two scripts had
        # diverged on this -- encoding checked fixed effects only)
        if r.converged and np.all(np.isfinite(r.bse)) and _re_ok(r):
            return r, re
    if not fallback:
        return None, "failed"
    rhs = formula.split("~", 1)[1]
    r = smf.ols(f"y ~ C(animal, Sum) + {rhs}", G).fit()
    r.fe_params = r.params
    return r, "fixed"


def tidy(r, re: str, terms, n: int, slope: str | None = None) -> list[dict]:
    """Rows term / coef / se / z / p (+ per-animal step and slope deviations when random)."""
    rfx = r.random_effects if re not in ("fixed", "failed") else {}
    blup = {f"step_dev_{an}": float(v.get("post", np.nan)) for an, v in rfx.items()}
    if slope:
        blup.update({f"slope_dev_{an}": float(v.get(slope, np.nan)) for an, v in rfx.items()})
    out = []
    for t in terms:
        if t not in r.fe_params.index:
            continue
        out.append({"term": "pre_level" if t == "Intercept" else t, "coef": float(r.fe_params[t]),
                    "se": float(r.bse[t]), "z": float(r.tvalues[t]), "p": float(r.pvalues[t]), "re": re, "n": n,
                    **blup})
    return out


def animal_epoch_table(L: pd.DataFrame, by: list[str], value: str = "y") -> pd.DataFrame:
    """RULE 8: per-animal epoch means (columns pre .. chronic) for each `by` group."""
    t = L.groupby([*by, "animal", "epoch"])[value].mean().unstack("epoch")
    return t.reindex(columns=[e for e in EPOCHS if e in t.columns])


def lmm_table(L: pd.DataFrame, by: list[str], models=("time", "behaviour"), value: str = "y",
              print_animals: bool = True, fdr_by: tuple = ("window",)) -> pd.DataFrame:
    """Fit every `by` group under each model; FDR (orofacial_stats.bh_fdr) within model x term x `fdr_by`.
    `L` needs label / `value` columns and the covariates from `add_time_covariates` (+ `add_deficit` for behaviour)."""
    from wfield_local.orofacial_stats import bh_fdr
    if print_animals:
        pd.set_option("display.width", 220)
        print("  per-animal epoch means (rule 8: animals before the model)\n"
              + animal_epoch_table(L, by, value).round(4).to_string())
    rows = []
    D = L.rename(columns={value: "y"}) if value != "y" else L
    for model in models:
        for key, G in D.groupby(by):
            key = key if isinstance(key, tuple) else (key,)
            if model == "behaviour":
                G = G.dropna(subset=["deficit"])
            if (G.post == 0).sum() < MIN_PRE or (G.post == 1).sum() < MIN_POST or G.animal.nunique() < MIN_ANIMALS:
                continue
            if model == "epoch":
                G = G.assign(epoch=pd.Categorical(G.epoch, list(EPOCHS)))
                r, re = fit_lmm(G, "y ~ C(epoch)", None)
                terms = ["Intercept", *[t for t in r.fe_params.index if t.startswith("C(epoch)")]]
                sl = None
            else:
                sl = SLOPES[model]
                r, re = fit_lmm(G, f"y ~ post + {sl}", sl)
                terms = ["Intercept", "post", sl]
            for row in tidy(r, re, terms, len(G), sl):
                rows.append({"model": model, **dict(zip(by, key)), **row})
    T = pd.DataFrame(rows)
    if T.empty:
        return T
    T["q"] = np.nan
    grp = ["model", "term", *[c for c in fdr_by if c in T.columns]]
    for _, idx in T.groupby(grp).groups.items():
        T.loc[idx, "q"] = bh_fdr(T.loc[idx, "p"].to_numpy())
    return T


def summary_table(T: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Readable: one row per model x `by`, a 'coef (z, q)' cell per term, + re and n."""
    W = T.assign(cell=[f"{c:.3g} (z {z:.1f}, q {q:.3f})" for c, z, q in zip(T.coef, T.z, T.q)])
    idx = ["model", *by]
    return (W.pivot_table(index=idx, columns="term", values="cell", aggfunc="first")
            .join(T.drop_duplicates(idx).set_index(idx)[["re", "n"]]))
