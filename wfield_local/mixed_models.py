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
             index (`DEFICITS`; default ttc = far-contra time to contact, (session - pre) / pre), 0 = pre level, 0 for pre sessions
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
SLOPES = {"time": "post:lday", "behaviour": "post:deficit", "time_drift": "post:lday"}
#: time_drift = time model + a linear DRIFT term over days (pre sessions at their negative days), so a slow trend
#: present before the stroke is estimated and not credited to the stroke (Priya 10-27; use only if it costs no power)
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


def behaviour_deficit(sessions_csv: Path, column: str = "contact_far_contra", higher_is_better: bool = True,
                      relative: bool = True) -> dict:
    """{label: deficit} = (animal's August-pre mean - session) / pre mean of `column` (default far-contra contact,
    from qc/position_coding_trend_recovery_sessions.csv, scripts/position_coding_trend_recovery.py). For a column where
    larger is worse (a latency; scripts/trial_attempts.py) pass higher_is_better=False: (session - pre) / pre.
    relative=False: (pre - session) in the column's own units -- for measures whose pre level is ~0 (signed angle
    offsets, e.g. the weighted aim of scripts/iti_recovery_measure.py), where dividing by pre is meaningless."""
    S = pd.read_csv(sessions_csv)
    pre = S[(S.epoch == "pre") & (S.label.str[-4:] >= "0806")].groupby("animal")[column].mean()
    d = S.animal.map(pre) - S[column]
    if relative:
        d = d / S.animal.map(pre)
    return dict(zip(S.label, d if higher_is_better else -d))


#: Behavioural indices for the `behaviour` model, {name: (qc csv, column, higher_is_better)} (Priya 10-20). PRIMARY =
#: `ttc`: mean time to first far-contra contact on ATTEMPTED trials (scripts/trial_attempts.py; no contact = 3.5 s), the
#: execution axis. `attempt` = initiation axis (attempt rate over the canonical engaged period). `contact` = the earlier
#: per-lick contact fraction, kept for continuity. Never an executed-kinematic variable (lateral angle): it enters the
#: neural measures' construction, so indexing by it is circular (DECISIONS 10-20).
DEFICITS = {"ttc": ("trial_attempts_sessions.csv", "ttc_ms_far_contra", False),
            "attempt": ("trial_attempts_sessions.csv", "attempt_rate_far_contra", True),
            "completion": ("trial_attempts_sessions.csv", "completion_far_contra", True),
            "latency": ("trial_attempts_sessions.csv", "latency_attempt_ms_far_contra", False),
            "contact": ("position_coding_trend_recovery_sessions.csv", "contact_far_contra", True)}
DEFICIT = "ttc"


def deficit(qc_dir: Path, name: str = DEFICIT) -> dict:
    """{label: deficit} for a `DEFICITS` entry, read from `<run>/imaging_encoding/qc`."""
    csv, col, hib = DEFICITS[name]
    return behaviour_deficit(Path(qc_dir) / csv, col, hib)


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
                drift = " + day" if model == "time_drift" else ""
                r, re = fit_lmm(G, f"y ~ post + {sl}{drift}", sl)
                terms = ["Intercept", "post", sl] + (["day"] if drift else [])
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


ANIMAL_COL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
#: x axis per model: behavioural deficit or days since stroke
_X = {"behaviour": ("deficit", "behavioural deficit, relative to pre (0 = pre level; larger = worse; mixed_models.DEFICITS)"),
      "behaviour_post": ("deficit", "behavioural deficit, relative to pre (0 = pre level; larger = worse; mixed_models.DEFICITS)"),
      "time": ("day", "days since stroke (pre sessions at their negative days)"),
      "time_post": ("day", "days since stroke")}


def _bold(txt: str) -> str:
    """matplotlib mathtext bold for a plain stats string (spaces / special characters escaped)."""
    for ch in ("\\", "%", "_", "$", "{", "}"):
        txt = txt.replace(ch, "\\" + ch)
    return "$\\mathbf{" + txt.replace(" ", "\\ ") + "}$"


def plot_epoch(L: pd.DataFrame, T: pd.DataFrame, panels, path: Path, title: str = "", ylabel: str = "y",
               ncols: int = 3) -> None:
    """EPOCH-model companion of `plot_fits` (y ~ C(epoch), same random-effects ladder): per panel, session dots per
    epoch (shape = animal) + animal means, the fitted epoch levels (pre + coefficient) with 95% Wald CI; text strip =
    each post epoch's coefficient (change vs pre), BOLD and * when BH-FDR q < 0.05."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    nrows = int(np.ceil(len(panels) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.9 * nrows), squeeze=False)
    for k, (name, sel) in enumerate(panels):
        a_ = ax.flat[k]
        G, c = L, T[T.model == "epoch"]
        for col, v in sel.items():
            G, c = G[G[col] == v], c[c[col] == v]
        if G.empty or c.empty:
            a_.axis("off")
            continue
        c = c.set_index("term")
        pre = c.coef.get("pre_level", np.nan)
        for i, ep in enumerate(EPOCHS):
            g = G[G.epoch == ep]
            rng = np.random.default_rng(i)
            for an, h in g.groupby("animal"):
                mk = ANIMAL_MARKER.get(an, "o")
                a_.plot(i + rng.uniform(-0.2, 0.2, len(h)), h.y, mk, color=EPOCH_LINE[ep], ms=3, alpha=0.4, mew=0)
                a_.plot(i, h.y.mean(), mk, mfc="none", mec=EPOCH_LINE[ep], mew=1.1, ms=6)
            t = f"C(epoch)[T.{ep}]"
            lvl, se = (pre, c.se.get("pre_level", np.nan)) if ep == "pre" else (pre + c.coef.get(t, np.nan),
                                                                                c.se.get(t, np.nan))
            a_.errorbar(i + 0.3, lvl, yerr=1.96 * se, fmt="s", color="k", ms=4, capsize=3, zorder=4)
            if ep != "pre" and t in c.index:
                q = c.q.get(t, np.nan)
                txt = f"{c.coef[t]:+.3g}" + ("*" if q < 0.05 else "") + f"\nq {q:.3f}"
                a_.text(i, 1.01, txt, transform=a_.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5,
                        color=EPOCH_LINE[ep], fontweight="bold" if q < 0.05 else "normal", clip_on=False)
        a_.axhline(pre, color="k", ls=":", lw=0.8)
        a_.set_xticks(range(len(EPOCHS)), EPOCHS, fontsize=7)
        a_.set_title(f"{name}; re {c.re.iloc[0]}", fontsize=7, pad=22)
        a_.set_ylabel(ylabel, fontsize=7)
        a_.tick_params(labelsize=7)
    for a_ in ax.flat[len(panels):]:
        a_.axis("off")
    fig.suptitle(f"{title} -- EPOCH mixed model (y ~ C(epoch), animal random effects): small = sessions, open = animal "
                 "means, shape = animal; black square = fitted epoch level +/- 95% CI; text = change vs pre, bold * = "
                 "BH-FDR q < 0.05", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_size_inches()[1]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_fits(L: pd.DataFrame, T: pd.DataFrame, panels, model: str, path: Path, title: str = "",
              ylabel: str = "y", ncols: int = 3, xlabel: str | None = None) -> None:
    """Session values (o pre, ^ post, coloured by animal) vs deficit or days, with the fitted fixed effects.
    `panels` = [(panel title, {column: value} selecting rows of L and of T)]; `model` = a T.model value (time,
    behaviour, time_post, behaviour_post). One figure -> `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xcol, xlab = _X[model]
    if xlabel and model.startswith("behaviour"):                  # the caller's deficit (e.g. an angle measure)
        xlab = xlabel
    nrows = int(np.ceil(len(panels) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.3 * nrows), squeeze=False)
    for k, (name, sel) in enumerate(panels):
        a_ = ax.flat[k]
        G, c = L, T[T.model == model]
        for col, v in sel.items():
            G, c = G[G[col] == v], c[c[col] == v]
        G = G.dropna(subset=[xcol])
        if G.empty or c.empty:
            a_.axis("off")
            continue
        for an, g in G.groupby("animal"):
            for post, mk in ((0, "o"), (1, "^")):
                h = g[g.post == post]
                a_.scatter(h[xcol], h.y, s=14, marker=mk, color=ANIMAL_COL.get(an, "0.4"), alpha=0.8,
                           label=f"{an} {'post' if post else 'pre'}" if k == 0 and len(h) else None)
        c = c.set_index("term")
        if model in ("time", "behaviour"):
            sl = SLOPES[model]
            pre, step, b = c.coef["pre_level"], c.coef["post"], c.coef.get(sl, np.nan)
            xx = (np.linspace(min(0, G.deficit.min()), max(1, G.deficit.max()), 50) if model == "behaviour"
                  else np.linspace(1, max(G.day.max(), 2), 80))
            a_.plot(xx, pre + step + b * (xx if model == "behaviour" else np.log(xx)), "k-", lw=1.5,
                    label="fit: post" if k == 0 else None)
            a_.axhline(pre, color="k", ls=":", lw=1, label="fit: pre level" if k == 0 else None)
            # * = FDR q < 0.05 (lmm_table's BH within model x term), so the panel says which terms are significant
            def _term(t, label, v, c=c):
                """'label v (z, q*)' -- BOLD when the term is significant (BH-FDR q < 0.05)."""
                q = c.q.get(t, np.nan) if "q" in c else np.nan
                txt = f"{label} {v:+.3g} (z {c.z.get(t, np.nan):.1f}" + (f", q {q:.3f}" if np.isfinite(q) else "") + ")"
                return _bold(txt + "*") if np.isfinite(q) and q < 0.05 else txt
            stat = f"{_term('post', 'post', step)}\n{_term(sl, sl, b)}"   # one term per line (no overlap)
        else:
            term, icpt = (("lday", "post_level_day1") if model == "time_post"
                          else ("deficit", "post_level_full_recovery"))
            a0, b = c.coef[icpt], c.coef[term]
            xx = np.linspace(1, max(G.day.max(), 2), 80) if model == "time_post" else np.linspace(0, 1, 50)
            a_.plot(xx, a0 + b * (np.log(xx) if model == "time_post" else xx), "k-", lw=1.5,
                    label="fit" if k == 0 else None)
            stat = f"{icpt} {a0:+.3g} (z {c.z[icpt]:.1f}), {term} {b:+.3g} (z {c.z[term]:.1f})"
        a_.set_title(f"{name}\n{stat}\nre {c.re.iloc[0]}", fontsize=6.5)
        a_.axhline(0, color="0.8", lw=0.5)
        a_.set_xlabel(xlab.split(",")[0].split(" (")[0], fontsize=7)    # short per panel; full text once, below
        a_.set_ylabel(ylabel, fontsize=7)
        a_.tick_params(labelsize=7)
    for a_ in ax.flat[len(panels):]:
        a_.axis("off")
    ax.flat[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{title} -- mixed model, animal random effects ({model})", fontsize=9)
    fig.supxlabel(f"x: {xlab}", fontsize=7)        # the long label once, so neighbouring panels' labels cannot collide
    # fixed headroom in INCHES for the suptitle / supxlabel: tight_layout alone ignores them, and on a tall figure
    # (8 rows, lmm_*_lickclass) the suptitle landed on the first row's panel titles (10-27)
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0.25 / h, 1, 1 - 0.45 / h))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
