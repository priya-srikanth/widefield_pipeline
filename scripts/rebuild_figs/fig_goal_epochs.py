"""Imaging results page Fig 5.2 = goal/encoding page Fig 8.2 (`fig2b_attempt_decoder_epochs.png`): the FROZEN
pre-stroke spout decoder on far-contra trials by attempt class across recovery (acute / subacute / chronic), cue and
lick windows, per animal; solid line = pre-stroke hits, dotted = chance 1/6; classes with >= 5 trials, trial counts
under each axis; mean +- SEM over trials; no test.

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from the session-scratchpad script fig_goal_epochs.py
  (session cdd1e016, 10-24; copy in wf_attempt scripts/recovered_20261103/, where an unused `import numpy` was
  removed). Computation, layout, suptitle and dpi unchanged. Changes: argparse (--out / --plot-only / --coding-dir)
  instead of sys.argv[1], figdata pickle (<out>/data/goal_epochs_figdata.pkl = the filtered summary rows).

INPUT: attempt_coding/summary_{cue,lick}.csv (wfield_local/attempt_coding.py, WIDEFIELD_LICK_SOURCE=lp; method "dom";
  far_contra; the decoder is the LP-trained frozen decoder) in --coding-dir (default = the original's
  hard-coded M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/
  attempt_coding).
REBUILD: re-run attempt_coding.py under the new data (decision D7; Priya 11-09: the LP-trained frozen decoder is THE
  LP model) and point --coding-dir at its output.
OUT: <out>/fig2b_attempt_decoder_epochs.png
EPOCH-COLOR CONVENTION (PI decision A13, Priya 11-07): before, one panel per animal with the attempt classes as colored
lines joining acute -> subacute -> chronic. Now one panel per CLASS (columns) x window (rows), x = epoch (pre = the
animals' pre-stroke hits, the reference): each animal's value is an OPEN marker, shape = animal, color = epoch, error bar
= its SEM over trials; NO line across epochs; filled dot = mean +- SEM over animals. Same inclusion (>= 5 trials), same
numbers; trial counts (summed over animals) under each epoch.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

NAME = "goal_epochs"
PNG = "fig2b_attempt_decoder_epochs.png"
from wfield_local import attempt_coding as _ac  # noqa: E402

A_DEF = _ac.out_dir()          # the writer's folder under the run mode (review N1); was the literal default path
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
CL = [("hit", "#1b9e77"), ("incomplete", "#d95f02"), ("noattempt", "#7570b3"), ("reaction", "#e7298a"),
      ("stopped", "0.6")]
WINDOWS = ["cue", "lick"]


def load(A: Path) -> dict:
    out = {}
    for w in WINDOWS:
        s = pd.read_csv(A / f"summary_{w}.csv")
        s = s[(s.spout == "far_contra") & (s.method == "dom")]
        out[w] = s[["animal", "cls", "epoch", "n", "p_true", "p_true_sem"]].reset_index(drop=True)
    return out


def draw(summ: dict, png: Path) -> None:
    from wfield_local import cat_points as cp
    classes = [cl for cl, _c in CL]
    fig, axes = plt.subplots(2, len(classes), figsize=(16, 8.4), sharey="row", squeeze=False)
    for r, w in enumerate(WINDOWS):
        s = summ[w]
        vals = [0.0, 1.0]
        for j, cl in enumerate(classes):
            ax = axes[r, j]
            for i, e in enumerate(EP):
                key = "pre_hit" if e == "pre" else cl
                am, ntot = [], 0
                for k, an in enumerate(AN):
                    y = s[(s.animal == an) & (s.cls == key) & (s.epoch == e)] if e != "pre" else \
                        s[(s.animal == an) & (s.cls == "pre_hit")]
                    if len(y) and (e == "pre" or y.n.iloc[0] >= MIN_N):
                        v, se = float(y.p_true.iloc[0]), float(y.p_true_sem.iloc[0])
                        ax.errorbar(i - 0.2 + 0.07 * k, v, yerr=se, fmt=cp.animal_marker(an), ms=5, mfc="none",
                                    mec=cp.epoch_color(e), ecolor=cp.epoch_color(e), mew=1.0, elinewidth=0.7,
                                    capsize=1.5, zorder=3)
                        am.append(v)
                        ntot += int(y.n.iloc[0])
                        vals += [v - se, v + se]
                if am:
                    cp.epoch_mean_sem(ax, i + 0.2, am, e, ms=6.5)
                    ax.text(i, -0.12, f"n {ntot}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                            fontsize=7, color=cp.epoch_color(e))
            ax.axhline(1 / 6, color="k", ls=":", lw=1)
            ax.set_xticks(range(len(EP)))
            ax.set_xticklabels(["pre hits", *EP[1:]], fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if r == 0:
                ax.set_title(cl)
            if j == 0:
                ax.set_ylabel(f"{w} window\nP(true spout), far contra")
        axes[r, 0].set_ylim(0, 1)
        cp.fit_ylim(axes[r, 0], vals)                    # 0..1, grown only if a mean +- SEM falls outside
    fig.legend(handles=cp.legend_handles(AN, EP, sessions_label="{a}", epoch_label="{e}: mean +- SEM over animals"),
               loc="upper center", ncol=8, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("Item 1 across epochs: far-contra trials on the frozen pre-stroke decoder by trial type "
                 "(classes with >= 5 trials per animal; chance dotted; pre = the animals' pre-stroke hits)\nopen = each "
                 "animal (mean +- SEM over trials; shape = animal, color = epoch), filled = mean +- SEM over animals, no "
                 "lines across epochs; n = trials summed over animals", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    fig.subplots_adjust(hspace=0.45)
    save_png_svg(fig, png, dpi=140)
    plt.close(fig)


#: the panels' trial floor per animal x class x epoch (pre is always drawn)
MIN_N = 5


def trial_set_sentence() -> str:
    """The attempt_coding far-contra class rows this figure draws (cue and lick windows, every post epoch)."""
    return _common.attempt_classes(epochs="acute / subacute / chronic", windows=_common.windows_text(WINDOWS),
                                   min_n=MIN_N)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--coding-dir", type=Path, default=A_DEF, help="imaging_encoding/attempt_coding folder")
    a = ap.parse_args(argv)
    _common.declare(trial_set_sentence())          # THE TRIAL SET of every figure (Priya 11-09)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        summ = _common.load_figdata(a.out, NAME)["summary"]
    else:
        summ = load(a.coding_dir)
        _common.save_figdata(a.out, NAME, {"summary": summ, "src": str(a.coding_dir)})
    draw(summ, a.out / PNG)
    print("->", a.out / PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
