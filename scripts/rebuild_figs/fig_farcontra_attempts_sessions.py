"""Far-contra trials, per session: attempt rate, completion, licks without contact, jaw moves without lick.

Replaces the per-animal table on the PI-deck slide "Far-contra trials: animals try and fail, the jaw reacts without
licking" with one open marker per session (shape = animal, color = epoch) and the mean of animals +- SEM.

Cohort 1 only (PS92-95, lesion LEFT -> far contra = far_R = canonical `far_contra`). Pre = August-only pre sessions.
Curated sessions only (`analysis_kit.curated_labels`). Epoch = `epoch_figures.epoch_of_day` on days from
`config.stroke_date` (cross-checked against the epoch column in the sources).

Sources (no pipeline is re-run; these are read as they are):
  attempt rate, completion   LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/qc/trial_attempts_sessions.csv
                             (scripts/trial_attempts.py): attempt_rate_far_contra = attempted / canonical (engaged)
                             trials, attempted = hit + incomplete; completion_far_contra = hit / attempted
  licks without contact,     DeepLabCut/Widefield/session_poses/cam1_licks/incomplete/sessions.csv
  jaw moves without lick     (scripts/cam1_incomplete_licks.py), position far_R: incomplete_frac = (cam4 no-contact +
                             cam1-only no-contact) / all licks; jaw_only_corr = share of jaw-QC'd trials with jaw
                             movement and no tongue event, cam1-corrected (FINDINGS 1.4, 1.7)

Run:  python fig_farcontra_attempts_sessions.py   (from the widefield_pipeline repo env, locanmf)
Out:  build_inputs/farcontra_attempts_sessions.png / .csv (per-session values)

PROVENANCE: recreated 2026-11-04 (RECONCILIATION section 7) from the share-only
  M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/fig_farcontra_attempts_sessions.py. Computation, layout and
  styling unchanged. Changes: attempt rate / completion from THIS run's trial_attempts_sessions.csv
  (`run_mode.attempts_dir()`: the run's camera classes; was the pre-rebuild adopted-run table); the cam1 incomplete
  table (a pose output, not rebuild-dependent) is read in place (--incomplete overrides);
  --out / --plot-only / figdata (`_common`); no sys.path pin of the main checkout.

    PYTHONPATH=<wf_rebuild> python scripts/rebuild_figs/fig_farcontra_attempts_sessions.py [--incomplete PATH] [--out DIR] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wfield_local import config  # noqa: E402
from wfield_local.analysis_kit import curated_labels  # noqa: E402
from wfield_local.epoch_figures import epoch_of_day  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE  # noqa: E402

import _common  # noqa: E402

NAME = "farcontra_attempts_sessions"
INC = Path(config.resolver().root("dlc")) / "session_poses" / "cam1_licks" / "incomplete" / "sessions.csv"   # in place
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
EPOCHS = ("pre", "acute", "subacute", "chronic")
PANELS = (("attempt_rate", "Attempt rate\n(engaged trials)"),
          ("completion", "Completion\n(attempt reaches spout)"),
          ("licks_no_contact", "Licks without contact\n(share of licks)"),
          ("jaw_no_lick", "Jaw moves, no lick\n(share of trials)"))


def _day(animal: str, mmdd: str) -> int:
    s = config.stroke_date(animal)
    return (date(2026, int(mmdd[:2]), int(mmdd[2:])) - date(2026, int(s[:2]), int(s[2:]))).days


def attempts_sessions() -> Path:
    """THIS run's trial_attempts_sessions.csv (routed trial_attempts; camera classes)."""
    from wfield_local import run_mode
    return run_mode.attempts_dir() / "trial_attempts_sessions.csv"


def load(att: Path, inc: Path) -> pd.DataFrame:
    A = pd.read_csv(att)
    A = A[A.animal.isin(ANIMALS)][["label", "animal", "epoch", "attempt_rate_far_contra", "completion_far_contra",
                                   "n_trials_far_contra"]]
    A = A.rename(columns={"attempt_rate_far_contra": "attempt_rate", "completion_far_contra": "completion",
                          "n_trials_far_contra": "n_trials_far_contra", "epoch": "epoch_src_att"})
    I = pd.read_csv(inc)
    I = I[I.animal.isin(ANIMALS) & (I.position == "far_R")].copy()
    I["label"] = I.animal + "_" + I.date.astype(str).str[-4:]
    I = I[["label", "incomplete_frac", "jaw_only_corr", "epoch"]].rename(
        columns={"incomplete_frac": "licks_no_contact", "jaw_only_corr": "jaw_no_lick", "epoch": "epoch_src_inc"})
    S = A.merge(I, on="label", how="outer")
    S["animal"] = S.label.str.split("_").str[0]
    cur = set(curated_labels())
    S = S[S.label.isin(cur)].copy()
    S["day"] = [_day(a, l[-4:]) for a, l in zip(S.animal, S.label)]
    S["epoch"] = [epoch_of_day(a, d) for a, d in zip(S.animal, S.day)]
    for c in ("epoch_src_att", "epoch_src_inc"):
        bad = S[S[c].notna() & (S[c] != S.epoch)]
        if len(bad):
            print(f"  WARNING epoch mismatch vs {c}:\n{bad[['label', 'day', 'epoch', c]]}")
    S = S[S.epoch.isin(EPOCHS)]
    S = S[(S.epoch != "pre") | (~S.label.map(config.early_pre))]          # August-only pre baseline
    S = S.drop(columns=["epoch_src_att", "epoch_src_inc"])
    S["epoch"] = pd.Categorical(S.epoch, EPOCHS, ordered=True)
    return S.sort_values(["animal", "day"]).reset_index(drop=True)


def trial_set_sentence() -> str:
    """The four panels' sets (trials for the attempt panels, LICKS for licks-without-contact, all trials for jaw)."""
    from wfield_local import trial_set_text as tst
    att = _common.working_period_trials(
        what="attempt rate / completion: far-contra trials of the camera working period",
        extra=["attempt = contact + incomplete, completion = contact / attempt"])
    lk = tst.events("licks without contact: every v36 cam4 lick and cam1-only lick event at far_R (all trials; no "
                    "engagement, working-period or class gate); no contact = cam4 no-contact licks + cam1-only "
                    "no-contact events", unit="licks")
    jaw = tst.events("jaw moves, no lick: every far_R trial (trial-level camera class jaw-only over all trials; no "
                     "working-period cut)", unit="trials")
    return tst.union(att, lk, jaw, unit="trials / licks")


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--incomplete", type=Path, default=INC, help="cam1 incomplete-licks sessions table")
    a = ap.parse_args(argv)
    _common.declare(trial_set_sentence())          # THE TRIAL SET of every figure (Priya 11-09)
    OUT = a.out
    S = _common.load_figdata(OUT, NAME) if a.plot_only else load(attempts_sessions(), a.incomplete)
    if not a.plot_only:
        _common.save_figdata(OUT, NAME, S)
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / "farcontra_attempts_sessions.csv", index=False, float_format="%.4f")
    rng = np.random.default_rng(0)
    off = {a: o for a, o in zip(ANIMALS, (-0.21, -0.07, 0.07, 0.21))}
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.8), sharey=True)
    summ = []
    for ax, (col, title) in zip(axs, PANELS):
        for (a, e), g in S.groupby(["animal", "epoch"], observed=True):
            v = g[col].dropna()
            x = EPOCHS.index(e) + off[a] + rng.uniform(-0.04, 0.04, len(v))
            ax.scatter(x, v, marker=MARK[a], s=26, facecolors="none", edgecolors=EPOCH_LINE[e], linewidths=1.0,
                       alpha=0.85, zorder=2)
        M = S.groupby(["animal", "epoch"], observed=True)[col].mean().unstack("epoch").reindex(columns=EPOCHS)
        mu, se = M.mean(), M.std(ddof=1) / np.sqrt(M.notna().sum())
        xs = np.arange(len(EPOCHS))
        ax.errorbar(xs, mu, yerr=se, fmt="_", ms=22, mew=3.0, color="0.15", elinewidth=2.0, capsize=0, ls="none", zorder=3)
        for e in EPOCHS:
            summ.append({"measure": col, "epoch": e, "mean_of_animals": mu[e], "sem": se[e],
                         **{a: M.loc[a, e] if a in M.index else np.nan for a in ANIMALS}})
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Pre", "Acute", "Subacute", "Chronic"], fontsize=11, rotation=30, ha="right",
                           rotation_mode="anchor")
        ax.set_xlim(-0.55, len(EPOCHS) - 0.45)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[0].set_ylabel("Fraction", fontsize=12)
    h = [Line2D([], [], ls="none", marker=MARK[a], mfc="none", mec="0.3", ms=7, label=a) for a in ANIMALS]
    h += [Line2D([], [], ls="none", marker="_", ms=16, mew=3.0, color="0.15", label="Mean of animals ± SEM")]
    h += [Line2D([], [], ls="none", marker="o", mfc="none", mec=EPOCH_LINE[e], mew=1.6, ms=7,
                 label=e.capitalize()) for e in EPOCHS]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.5)
    save_png_svg(fig, OUT / "farcontra_attempts_sessions.png", dpi=200)
    pd.set_option("display.width", 200)
    print(S.groupby(["animal", "epoch"], observed=True).size().unstack("epoch").to_string())
    print(pd.DataFrame(summ).round(3).to_string())
    print(f"-> {OUT / 'farcontra_attempts_sessions.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
