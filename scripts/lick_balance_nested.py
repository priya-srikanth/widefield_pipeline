"""Held-out confirmation of the best balance DEFINITION (Priya 10-28: "yes confirm the first cued lick timing on held out
data"). The sweep (lick_balance_variants, 14.18) picked e.g. first-cued-lick-aligned ratio 0.2..0.35 s (LOAO 0.69) out
of ~60 definitions, so its score is inflated by the selection. Here the SELECTION is inside the cross-validation:
  outer loop: hold out one animal;
  inner: on the other 3 animals, score every candidate definition by leave-one-animal-out cvR2 among those 3 and pick
         the best;
  fit OLS aim ~ chosen definition on the 3 animals, predict the held-out animal.
Pooled cvR2 over the 4 held-out animals (SSE / SST around the overall mean, as lick_recovery_markers.cv_r2), the chosen
definition per fold, and a NULL from repeating the whole procedure with every candidate permuted across sessions within
each animal (the same permutation for all candidates, so their correlations are kept). Candidate families: ALL definitions;
TIMING only (one family, the window free): raw difference, baseline-subtracted difference, gain ratio over the 8 bins.
Lick sets / targets as the sweep. Reads <OUT_W>/balvar_sessions.csv; writes balnest.csv, figure lick_balance_nested.png.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402

N_NULL = 200
TARGETS = {"cued": ["aim"], lrm.C1SET: ["aim"], "ITI": ["iti_ang"], lrm.QSET: ["iti_ang"]}


def _ols(xtr, ytr, xte):
    X = np.column_stack([np.ones(len(xtr)), xtr])
    b = np.linalg.lstsq(X, ytr, rcond=None)[0]
    return b[0] + b[1] * xte


def _loao_r2(x, y, grp):
    pred = np.full(len(y), np.nan)
    for g in np.unique(grp):
        te = grp == g
        pred[te] = _ols(x[~te], y[~te], x[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def nested(X: np.ndarray, names: list[str], y: np.ndarray, grp: np.ndarray) -> tuple[float, list[str]]:
    pred = np.full(len(y), np.nan)
    chosen = []
    for g in np.unique(grp):
        te = grp == g
        scores = [_loao_r2(X[~te, j], y[~te], grp[~te]) for j in range(X.shape[1])]
        j = int(np.nanargmax(scores))
        chosen.append(names[j])
        pred[te] = _ols(X[~te, j], y[~te], X[te, j])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), chosen


def trial_set_sentence() -> str:
    """The lick sets of the held-out definition check = the sweep's (balvar_sessions.csv of lick_balance_variants)."""
    import lick_balance_variants as bv
    return bv.trial_set_sentence()


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    S = pd.read_csv(il.OUT_W / "balvar_sessions.csv")
    allc = [c for c in S.columns if c.startswith(("diff:", "ratio:"))]
    fams = {"all definitions": allc,
            "timing: raw difference": [c for c in allc if c.startswith("diff:raw") and c.endswith("|oro")],
            "timing: baseline-sub difference": [c for c in allc if c.startswith("diff:bs") and c.endswith("|oro")],
            "timing: gain ratio": [c for c in allc if c.startswith("ratio:bs") and c.endswith("|oro")]}
    rng = np.random.default_rng(0)
    rows = []
    for st, ys in TARGETS.items():
        for yv in ys:
            G = S[(S.set == st) & (S.epoch != "pre") & (S.day > 0)].dropna(subset=[yv]).copy()
            G["logday"] = np.log(G.day)
            y = G[yv].to_numpy()
            grp = G.animal.to_numpy()
            r_days = _loao_r2(G.logday.to_numpy(), y, grp)
            for fam, cands in fams.items():
                C = G[cands]
                C = C.loc[:, C.notna().mean() > 0.9]
                X = C.to_numpy(float)
                X = np.where(np.isfinite(X), X, np.nanmean(X, 0))
                r, chosen = nested(X, list(C.columns), y, grp)
                null = []
                for _ in range(N_NULL):
                    Xs = X.copy()
                    for a in np.unique(grp):
                        ix = np.flatnonzero(grp == a)
                        Xs[ix] = X[rng.permutation(ix)]
                    null.append(nested(Xs, list(C.columns), y, grp)[0])
                null = np.asarray(null)
                rows.append({"set": st, "y": yv, "family": fam, "n_candidates": X.shape[1], "nested_loao": r,
                             "days_loao": r_days, "null95": float(np.quantile(null, 0.95)),
                             "p_null": float((1 + np.sum(null >= r)) / (1 + len(null))),
                             "chosen_per_fold": " | ".join(f"{a}:{c}" for a, c in zip(np.unique(grp), chosen, strict=True))})
                print(f"[{st} -> {yv}] {fam}: nested {r:.3f} (days {r_days:.3f}, null95 {rows[-1]['null95']:.3f}, "
                      f"p {rows[-1]['p_null']:.3g})\n    chosen: {rows[-1]['chosen_per_fold']}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(il.OUT_W / "balnest.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(st, yv) for st, ys in TARGETS.items() for yv in ys]
    fig, axs = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5), squeeze=False)
    for j, (st, yv) in enumerate(panels):
        ax = axs[0, j]
        t = R[(R.set == st) & (R.y == yv)]
        x = np.arange(len(t))
        ax.bar(x, t.nested_loao, color=["#8172B3", "0.4", "#C44E52", "#4C72B0"][: len(t)])
        ax.plot(x, t.null95, "k_", ms=16, mew=1.5)
        ax.axhline(t.days_loao.iloc[0], color="0.4", ls="--", lw=1, label="days since stroke")
        for xi, (_, r) in enumerate(t.iterrows()):
            ax.text(xi, max(r.nested_loao, 0) + 0.03, f"p {r.p_null:.2g}", ha="center", fontsize=7,
                    fontweight="bold" if r.p_null < 0.05 and r.nested_loao > r.days_loao else "normal")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1, 1)
        ax.set_xticks(x, [f.replace("timing: ", "") for f in t.family], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{st} -> {yv}\nheld-out animal, selection inside CV", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("Held-out confirmation: the balance definition (window, difference vs ratio) is CHOSEN on 3 animals and "
                 "TESTED on the 4th (4 folds, pooled cvR²); bar = null 95% from repeating the whole selection on "
                 "within-animal shuffled sessions. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_balance_nested.png", dpi=95)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
