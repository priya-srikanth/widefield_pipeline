"""BEHAVIORAL AIM DEFICIT per session, FIRST cued lick vs all licks, weighted vs far contra (Priya 10-27: switch the
behavior axis / mixed-model deficit to the first-lick weighted aim; "can I see a comparison between the weighted deficit
and the first-lick far-contra deficit vs time across animals").

THE aim definition lives in `iti_recovery_measure.py` (aim_licks / weighted / far_contra; rule 9): per cued lick, angle
atan2(lateral, sagittal) (deg; 3-D landing at the extension peak) minus the animal's PRE mean angle at the same target
spout. First lick = lick_idx 0 (first post-cue lick of the trial; joined via first_lick_preds.csv.gz). Per session:
  weighted_first / weighted_all   signed offset per spout, combined with weights far contra 3, far mid 2, others 1
  farcontra_first / farcontra_all signed offset at far contra
(- = aim shifted ipsilesional). Split-half reliability (random half of the session's licks, Spearman-Brown).
Min licks per spout: first 5, all 10 (iti_recovery_measure.MIN_LICKS). The mixed models' behavior deficit = pre -
session of the weighted aim of the WFIELD_AIM lick set (first by default; iti_recovery_measure.aim_deficit).
Outputs <E>/iti_licks/aim_deficit_sessions.csv, figures/lick_cued_aim_deficit_compare.png
(figure data figdata/lick_cued_aim_deficit_compare.pkl; --plot-only redraws from it).
Usage: python scripts/lick_aim_deficit.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_recovery_measure as rm  # noqa: E402

AN = rm.AN
FD = il.OUT / "figdata" / "lick_cued_aim_deficit_compare.pkl"


def compute() -> pd.DataFrame:
    """Per session, from THE aim definition (iti_recovery_measure: aim_licks / weighted / far_contra; rule 9): FIRST-lick
    and all-lick weighted + far-contra signed aim (MIN_LICKS per spout of each set) and split halves of the first-lick
    licks (>= 3 licks per spout per half)."""
    from wfield_local import epochs
    p = rm.aim_licks()
    rng = np.random.default_rng(0)
    rows = []
    mf, ma = rm.MIN_LICKS["first"], rm.MIN_LICKS["all"]
    for (lab, ep, a), g in p.groupby(["label", "epoch", "animal"]):
        f = g[g["first"]]
        hf = rng.random(len(f)) < 0.5
        rows.append(dict(label=lab, epoch=ep, animal=a, day=float(epochs.days_since_stroke(lab)),
                         weighted_all=rm.weighted(g, ma), weighted_first=rm.weighted(f, mf),
                         farcontra_all=rm.far_contra(g, ma), farcontra_first=rm.far_contra(f, mf),
                         wf_h1=rm.weighted(f[hf], 3), wf_h2=rm.weighted(f[~hf], 3),
                         fcf_h1=rm.far_contra(f[hf], 3), fcf_h2=rm.far_contra(f[~hf], 3),
                         n_first=len(f), n_all=len(g)))
    S = pd.DataFrame(rows).sort_values(["animal", "day"])
    S.to_csv(il.OUT / "aim_deficit_sessions.csv", index=False)
    return S


def reliability(S: pd.DataFrame) -> dict:
    out = {}
    for nm, (a, b) in {"weighted_first": ("wf_h1", "wf_h2"), "farcontra_first": ("fcf_h1", "fcf_h2")}.items():
        d = S[[a, b]].dropna()
        r = float(np.corrcoef(d[a], d[b])[0, 1])
        out[nm] = (r, 2 * r / (1 + r), len(d))
    return out


def plot(S: pd.DataFrame, rel: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    from wfield_local.mixed_models import ANIMAL_COL
    series = [("weighted_first", "weighted aim, FIRST lick", "-", 2.0), ("farcontra_first", "far-contra aim, FIRST lick",
              "--", 1.6), ("weighted_all", "weighted aim, all licks", ":", 1.2)]
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.6), squeeze=False, sharey=True)
    for j, a in enumerate(AN):
        ax = axs[0, j]
        g = S[S.animal == a]
        for col, _lab, ls, lw in series:
            ax.plot(g.day, g[col], ls, color=ANIMAL_COL[a], lw=lw, marker=ANIMAL_MARKER[a], ms=3)
        ax.set_title(a, fontsize=10)
    ax = axs[0, 4]
    for col, _lab, ls, lw in series:
        A = S.assign(dbin=np.where(S.day < 0, -5, np.round(S.day / 3) * 3)).groupby("dbin")[col]
        mu, se = A.mean(), A.sem()
        ax.errorbar(mu.index, mu, yerr=se, fmt="o" + ls, color="k", lw=lw, ms=3, capsize=2)
    ax.set_title("all animals (3-day bins; pre = left), mean +/- SEM", fontsize=9)
    for ax in axs[0]:
        ax.axhline(0, color="0.6", lw=0.7)
        ax.axvline(0, color="0.6", lw=0.7, ls=":")
        ax.set_xlabel("days since stroke", fontsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0, 0].set_ylabel("aim vs own pre landing (deg, - = ipsilesional)", fontsize=8)
    axs[0, 0].legend(handles=[Line2D([], [], color="k", ls=ls, lw=lw, label=lab) for _c, lab, ls, lw in series],
                     fontsize=7.5, frameon=False, loc="lower right")
    txt = "; ".join(f"{k.replace('_', ' ')}: split-half r {v[0]:.2f}, reliability {v[1]:.2f} ({v[2]} sessions)"
                    for k, v in rel.items())
    fig.suptitle("BEHAVIOR: cued-lick aim deficit per session vs days since stroke -- weighted (far contra x3, far mid x2, "
                 "others x1) vs far contra only, FIRST post-cue lick vs all licks.\n" + txt + ". The mixed models' "
                 "behavior deficit = pre - session of the FIRST-lick weighted aim. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_png_svg(fig, il.OUT / "figures" / "lick_cued_aim_deficit_compare.png", dpi=100)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The lick sets of the aim-deficit comparison (Priya 11-09): the cued decoding licks of the session files with a
    3-D lateral + sagittal tip position at one of the six spouts (`iti_recovery_measure.aim_licks`); FIRST = lick_idx 0;
    per spout >= MIN_LICKS licks of the set."""
    import lick_covariate_reference as cv
    from wfield_local import trial_set_text as tst
    c = "3-D lateral + sagittal tip position"
    first = cv.session_cued(first_only=True, coords=c, extra=[f">= {rm.MIN_LICKS['first']} per spout x session"])
    allk = cv.session_cued(coords=c, extra=[f">= {rm.MIN_LICKS['all']} per spout x session"])
    return tst.union(first, allk)


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    if "--plot-only" in sys.argv and FD.exists():
        S, rel = pickle.loads(FD.read_bytes())
    else:
        S = compute()
        rel = reliability(S)
        FD.parent.mkdir(exist_ok=True)
        FD.write_bytes(pickle.dumps((S, rel)))
    print(rel)
    pd.set_option("display.width", 200)
    print(S.groupby(["animal", "epoch"])[["weighted_first", "farcontra_first", "weighted_all", "farcontra_all"]].mean()
          .round(1).unstack("epoch").to_string())
    plot(S, rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
