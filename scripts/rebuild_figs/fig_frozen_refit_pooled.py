"""Imaging results page Fig 4.4 (`frozen_refit_pooled.png`; PI-deck slide 12): size-matched FROZEN vs within-session
REFIT spout decoder, POOLED over the six spouts, by epoch, with the refit - frozen gap.

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from the share-only
  M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/fig_frozen_refit_pooled.py (11-02 20:18).
  Computation (pairing, boot_delta seeds, Bonferroni x9), layout and jitter seeds unchanged. Changes: argparse
  (--out / --plot-only / --src), no sys.path insert of widefield_pipeline (uses the repo on PYTHONPATH), output
  folder no longer build_inputs/, figdata pickle (<out>/data/frozen_refit_pooled_figdata.pkl).

INPUT (no decoding re-run): the per-session sidecar of the matched pooled-over-positions family (5rmo) written by
  wfield_local/epoch_accuracy.py::_frozen_vs_refit_overall(matched=True), cue window 0-2 s, working trials
  (camera gate: contact + incomplete inside the working period, stopped runs and quit tail removed), the decoders of
  imaging Figs 4.1 / 4.2:
    labcams/grant_figures/epoch/data/epoch_5rmo_frozen_refit_overall_cue_working_sessions.csv
  Default = the original's hard-coded M: path (kept; --src overrides). Each session's value is TRIAL-POOLED over
  positions; gap = refit - frozen on the same trials. Pre = June + August (11 per animal). Cohort 1 only.
REBUILD: re-render epoch family 5rmo (epoch_grant_figures --only 5rmo) under the new data, then point --src at its
  sidecar.

STATS: wfield_local.analysis_kit.boot_delta (paired animals -> sessions, animal-weighted), each post epoch vs pre per
panel; * = 95% CI excludes 0; ** = CI at alpha 0.05/9 (Bonferroni, 3 post epochs x 3 panels) excludes 0.
PLOT: sessions = open markers (shape = animal, color = epoch); filled dot +- bar = mean of animal means +- SEM (n=4);
stars in the top band (sig_marks.star_top + reserve_headroom).
OUT: <out>/frozen_refit_pooled.png / .csv (3 sections: sessions, levels, change vs pre)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

NAME = "frozen_refit_pooled"
SRC = _common.epoch_data_dir() / "epoch_5rmo_frozen_refit_overall_cue_working_sessions.csv"   # R3: this run's
EPOCHS = ["pre", "acute", "subacute", "chronic"]
ARMS = ["frozen", "refit", "gap"]
ANIMALS = ["PS92", "PS93", "PS94", "PS95"]
TITLES = {"frozen": "Frozen pre-stroke decoder", "refit": "Within-session refit decoder",
          "gap": "Refit \u2212 frozen gap"}
N_BOOT = 4000
N_COMP = 9
CHANCE = 1 / 6


def load(src):
    d = pd.read_csv(src)
    d["session_idx"] = d.groupby(["epoch", "position", "animal"]).cumcount()
    w = d.pivot_table(index=["epoch", "animal", "session_idx"], columns="position", values="value").reset_index()
    assert np.allclose(w.refit - w.frozen, w.gap), "arms not paired"
    return w[["epoch", "animal", "session_idx", *ARMS]]


def stats(w):
    from wfield_local.analysis_kit import boot_delta
    rows = []
    a = 0.05 / N_COMP
    for k, arm in enumerate(ARMS):
        pre = {an: w[(w.epoch == "pre") & (w.animal == an)][arm].tolist() for an in ANIMALS}
        for e in EPOCHS[1:]:
            post = {an: w[(w.epoch == e) & (w.animal == an)][arm].tolist() for an in ANIMALS}
            post = {an: v for an, v in post.items() if v}
            rng = np.random.default_rng(20261102 + 10 * k + EPOCHS.index(e))
            iv, o = boot_delta(post, pre, rng, n_boot=N_BOOT, draws=True)
            clo, chi = np.percentile(o, [100 * a / 2, 100 * (1 - a / 2)])
            mark = "**" if (clo > 0 or chi < 0) else ("*" if (iv.lo > 0 or iv.hi < 0) else "")
            per = {an: np.mean(post[an]) - np.mean(pre[an]) for an in post}
            rows.append(dict(panel=arm, epoch=e, delta_vs_pre=iv.point, lo95=iv.lo, hi95=iv.hi,
                             lo_bonf=clo, hi_bonf=chi, mark=mark, n_animals=iv.n_animals,
                             n_animals_up=sum(v > 0 for v in per.values()),
                             **{f"delta_{an}": per.get(an, np.nan) for an in ANIMALS}))
    return pd.DataFrame(rows)


def level_table(w):
    rows = []
    for arm in ARMS:
        for e in EPOCHS:
            m = w[w.epoch == e].groupby("animal")[arm].mean()
            rows.append(dict(panel=arm, epoch=e, mean_of_animals=m.mean(),
                             sem_animals=m.std(ddof=1) / np.sqrt(len(m)), n_animals=len(m),
                             n_sessions=int((w.epoch == e).sum()),
                             **{f"mean_{an}": m.get(an, np.nan) for an in ANIMALS}))
    return pd.DataFrame(rows)


def plot(w, st, lv, png):
    from wfield_local.orofacial_cohort import EPOCH_LINE, ANIMAL_MARKER
    from wfield_local.sig_marks import star_top, reserve_headroom
    plt.rcParams.update({"font.family": "Arial", "font.size": 12})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    rng = np.random.default_rng(3)
    for ax, arm in zip(axes, ARMS):
        for i, e in enumerate(EPOCHS):
            c = EPOCH_LINE[e]
            for an in ANIMALS:
                v = w[(w.epoch == e) & (w.animal == an)][arm].values
                x = i - 0.13 + rng.uniform(-0.09, 0.09, len(v))
                ax.scatter(x, v, marker=ANIMAL_MARKER[an], s=26, facecolors="none", edgecolors=c,
                           linewidths=0.9, alpha=0.75, zorder=2)
            r = lv[(lv.panel == arm) & (lv.epoch == e)].iloc[0]
            ax.errorbar(i + 0.2, r.mean_of_animals, yerr=r.sem_animals, fmt="o", ms=7, color=c,
                        mec="k", mew=0.6, capsize=4, elinewidth=1.6, zorder=4)
            if e != "pre":
                mk = st[(st.panel == arm) & (st.epoch == e)].iloc[0].mark
                if mk:
                    star_top(ax, i + 0.03, mk, color=c, row=0, fontsize=14)
        if arm == "gap":
            ax.axhline(0, ls="--", color="0.5", lw=1, zorder=1)
            ax.set_ylabel("Refit \u2212 frozen accuracy")
        else:
            ax.axhline(CHANCE, ls="--", color="0.5", lw=1, zorder=1)
            ax.text(3.45, CHANCE + 0.015, "chance", ha="right", va="bottom", fontsize=11, color="0.4")
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Decoding accuracy" if arm == "frozen" else "")
        ax.set_title(TITLES[arm], fontsize=13, pad=6)
        ax.set_xticks(range(4))
        ax.set_xticklabels([e.capitalize() for e in EPOCHS])
        for t, e in zip(ax.get_xticklabels(), EPOCHS):
            t.set_color(EPOCH_LINE[e])
        ax.set_xlim(-0.5, 3.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)
    axes[2].set_ylim(-0.12, 0.42)
    reserve_headroom(fig)
    h = [Line2D([], [], marker=ANIMAL_MARKER[an], ls="", mfc="none", mec="0.25", ms=7, label=an) for an in ANIMALS]
    # marker + vertical error bar only (no horizontal line): a NaN errorbar gives the legend an errorbar handle
    eb = axes[0].errorbar([np.nan], [np.nan], yerr=[0.1], fmt="o", ms=7, color="0.25", mec="k", mew=0.6,
                          capsize=4, elinewidth=1.6, label="Mean of animals \u00b1 SEM")
    h.append(eb)
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)
    save_png_svg(fig, png, dpi=200)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The 5rmo sidecar's trials: camera working rows, cue window, pooled over positions."""
    return _common.working_trials(extra=["cue window 0-2 s, trial-pooled over positions",
                                         "frozen arm trained on a size-matched subset of pre blocks; refit and frozen "
                                         "scored on the same trials"])


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--src", type=Path, default=SRC, help="epoch_5rmo_..._sessions.csv sidecar")
    a = ap.parse_args(argv)
    _common.declare(trial_set_sentence())          # THE TRIAL SET of every figure (Priya 11-09)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        fd = _common.load_figdata(a.out, NAME)
        plot(fd["w"], fd["st"], fd["lv"], a.out / f"{NAME}.png")
        return 0
    w = load(a.src)
    st, lv = stats(w), level_table(w)
    _common.save_figdata(a.out, NAME, {"w": w, "st": st, "lv": lv, "src": str(a.src)})
    plot(w, st, lv, a.out / f"{NAME}.png")
    with open(a.out / f"{NAME}.csv", "w", newline="") as f:
        f.write("# source: " + str(a.src) + "\n# section 1: per-session values (trial-pooled over positions)\n")
        w.to_csv(f, index=False)
        f.write("\n# section 2: level per epoch, mean of animal means +- SEM over animals\n")
        lv.to_csv(f, index=False)
        f.write("\n# section 3: change from pre, boot_delta (paired animals->sessions, animal-weighted, "
                f"{N_BOOT} draws); * 95% CI excl 0, ** Bonferroni x{N_COMP}\n")
        st.to_csv(f, index=False)
    pd.set_option("display.width", 250)
    print(w.groupby(["animal", "epoch"]).size().unstack()[EPOCHS])
    print(lv.round(3).to_string())
    print(st.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
