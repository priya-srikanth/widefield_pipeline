"""Imaging results page Fig 5.2 = goal/encoding page Fig 8.2 (`fig2b_attempt_decoder_epochs.png`): the FROZEN
pre-stroke spout decoder on far-contra trials by attempt class across recovery (acute / subacute / chronic), cue and
lick windows, per animal; solid line = pre-stroke hits, dotted = chance 1/6; classes with >= 5 trials, trial counts
under each axis; mean +- SEM over trials; no test.

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from the session-scratchpad script fig_goal_epochs.py
  (session cdd1e016, 10-24; copy in wf_attempt scripts/recovered_20261103/, where an unused `import numpy` was
  removed). Computation, layout, suptitle and dpi unchanged. Changes: argparse (--out / --plot-only / --coding-dir)
  instead of sys.argv[1], figdata pickle (<out>/data/goal_epochs_figdata.pkl = the filtered summary rows).

INPUT: attempt_coding/summary_{cue,lick}.csv (wfield_local/attempt_coding.py, WIDEFIELD_LICK_SOURCE=lp; method "dom";
  far_contra; the frozen decoder is the stored DAQ-contact model) in --coding-dir (default = the original's
  hard-coded M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/
  attempt_coding).
REBUILD: re-run attempt_coding.py under the new data (decision D7: training rule of the LP-gated stored DAQ model) and
  point --coding-dir at its output.
OUT: <out>/fig2b_attempt_decoder_epochs.png
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
    fig, axes = plt.subplots(2, 4, figsize=(14, 9), sharey="row")
    for r, w in enumerate(WINDOWS):
        s = summ[w]
        for k, an in enumerate(AN):
            ax = axes[r, k]
            x = s[(s.animal == an) & (s.cls == "pre_hit")]
            if len(x):
                ax.axhline(x.p_true.iloc[0], color="0.2", lw=1.5, label="pre-stroke hits")
            for j, (cl, col) in enumerate(CL):
                xs, ys, es, ns = [], [], [], []
                for i, e in enumerate(EP[1:]):
                    y = s[(s.animal == an) & (s.cls == cl) & (s.epoch == e)]
                    if len(y) and y.n.iloc[0] >= 5:
                        xs.append(i + (j - 2) * 0.1)
                        ys.append(y.p_true.iloc[0])
                        es.append(y.p_true_sem.iloc[0])
                        ns.append(int(y.n.iloc[0]))
                if xs:
                    ax.errorbar(xs, ys, yerr=es, fmt="-o", color=col, ms=5, lw=1.5, capsize=2, label=cl)
                    for xx, nn in zip(xs, ns):           # trial counts in a column under the axis, one row per class
                        ax.text(round(xx), -0.13 - 0.075 * j, str(nn), transform=ax.get_xaxis_transform(),
                                ha="center", va="top", fontsize=7, color=col)
            ax.axhline(1 / 6, color="k", ls=":", lw=1)
            ax.set_xticks(range(3))
            ax.set_xticklabels(EP[1:])
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(an)
            if k == 0:
                ax.set_ylabel(f"{w} window\nP(true spout), far contra")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Item 1 across epochs: far-contra trials on the frozen pre-stroke decoder by trial type "
                 "(classes with >= 5 trials; chance dotted, pre-stroke hits solid)\nnumbers under each axis = trials "
                 "per class (same colours)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.subplots_adjust(hspace=0.75)
    save_png_svg(fig, png, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--coding-dir", type=Path, default=A_DEF, help="imaging_encoding/attempt_coding folder")
    a = ap.parse_args(argv)
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
