"""Goal-question summary figures (10-24): imaging results page Fig 5.1 = goal/encoding page Fig 8.1
(`fig2_attempt_decoder.png`), plus the two other figures the same script drew for the goal page
(`fig1_binned_contest.png`, `fig3_attempt_amplitude.png`).

  fig2_attempt_decoder   FROZEN pre-stroke 6-way spout decoder on far-contra trials by attempt class (LP video:
                         pre-stroke hit, hit, incomplete, no attempt, reaction, stopped), acute, pre-cue / cue / lick
                         windows: top = P(true spout) +- SEM over trials (classes with >= 5 trials), bottom = share
                         decoded as ANY far spout.
  fig1_binned_contest    binned target vs executed contest (O2 encoding, cortex), unique dR2 per session and epoch.
  fig3_attempt_amplitude far-contra lick-window amplitude ratio (acute hits / failed attempts vs pre-stroke refs).

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from the session-scratchpad script
  fig_goal_summary.py (session cdd1e016, 10-25 06:32; copy in wf_attempt scripts/recovered_20261103/). Computation,
  layout, suptitles (including their hard-coded readings) and dpi unchanged. Changes: argparse (--out / --plot-only /
  --enc-dir / --only) instead of sys.argv[1], figdata pickle (<out>/data/goal_summary_figdata.pkl: the filtered
  rows each figure draws), figures can be drawn one at a time (--only fig2).
  NOTE (as found, not changed): the suptitles carry readings typed on 10-24 ("chronic target > executed 4/4", "failed
  attempts (orange) lose the far-contra code ..."); they are text, not computed, and will not follow new data.

INPUTS, under --enc-dir (default = the original's hard-coded
  M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding):
  attempt_coding/summary_{precue,cue,lick}.csv and attempt_coding/trials_{precue,cue,lick}_<animal>.csv.gz
      (wfield_local/attempt_coding.py, WIDEFIELD_LICK_SOURCE=lp; method "dom"; the frozen decoder is the stored
      DAQ-contact model) -> fig2
  movement_first_subset-nokin_exec-polar_licks-combined_bins/tables/region_measures.csv (O2 encoding) -> fig1
  attempt_maps/hemispheres_lick.csv (wfield_local/attempt_maps.py) -> fig3
REBUILD: re-run attempt_coding.py (decision D7: training rule of the LP-gated stored DAQ model), attempt_maps.py and
  the O2 contest bundle, then point --enc-dir at the new imaging_encoding folder.
OUT: <out>/fig1_binned_contest.png, fig2_attempt_decoder.png, fig3_attempt_amplitude.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # noqa: E402 - rebuild root (11-03)
from wfield_local.attempt_coding import trials_name as attempt_trials_name  # noqa: E402

NAME = "goal_summary"
E_DEF = _encoding_root()
#: R3: fig1's combined-encoding run, at its WRITER's routed folder (combined_encoding: lp_licks.output_dir)
def _fig1_root() -> Path:
    from wfield_local import lp_licks
    return lp_licks.output_dir(E_DEF / "_").parent
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
TC, XC = "#6a3d9a", "#e66101"
CL = ["pre_hit", "hit", "incomplete", "noattempt", "reaction", "stopped"]
CC = {"pre_hit": "0.25", "hit": "#1b9e77", "incomplete": "#d95f02", "noattempt": "#7570b3", "reaction": "#e7298a",
      "stopped": "0.65"}
WINDOWS = ["precue", "cue", "lick"]
FIGS = ("fig1", "fig2", "fig3")


# ---------------------------------------------------------------- data
def load_fig1(E: Path) -> pd.DataFrame:
    d = pd.read_csv(E / "movement_first_subset-nokin_exec-polar_licks-combined_bins/tables/region_measures.csv")
    c = d[d.pos == "cortex"].copy()
    cols = [f"nokin:u:{k}bin|{w}" for w in ("cue", "lick") for k in ("tgt", "ex")]
    return c[["animal", "epoch", *cols]].reset_index(drop=True)


def _attempt_dir():
    """attempt_coding output folder, resolved by its WRITER (review N1 / N10): never a literal path."""
    from wfield_local import attempt_coding
    return attempt_coding.out_dir()


def load_fig2(E: Path) -> dict:
    A = _attempt_dir()                           # the writer's folder (review N1); E kept for the other inputs
    summ, trials = {}, {}
    for w in WINDOWS:
        s = pd.read_csv(A / f"summary_{w}.csv")
        s = s[(s.spout == "far_contra") & (s.method == "dom") & (((s.epoch == "acute") & (s.cls != "pre_hit"))
                                                                 | (s.cls == "pre_hit"))]
        summ[w] = s[["animal", "cls", "n", "p_true", "p_true_sem"]].reset_index(drop=True)
        for an in AN:
            t = pd.read_csv(A / attempt_trials_name(w, an))
            t = t[(t.method == "dom") & (t.spout == "far_contra") & ((t.epoch == "acute") | (t.cls == "pre_hit"))]
            trials[(w, an)] = t[["cls", "pred"]].reset_index(drop=True)
    return {"summary": summ, "trials": trials}


def load_fig3(E: Path) -> pd.DataFrame:
    from wfield_local import attempt_maps
    h = pd.read_csv(attempt_maps.hemispheres_path(attempt_maps.out_dir(), "lick"))   # the writer's folder + key (R4)
    return h[h.spout == "far_contra"].set_index(["animal", "epoch", "cls"])[["ipsi", "contra"]]


# ---------------------------------------------------------------- figures
def draw_fig1(c: pd.DataFrame, png: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for r, w in enumerate(["cue", "lick"]):
        lo, hi = np.inf, -np.inf
        for k, an in enumerate(AN):
            ax = axes[r, k]
            a = c[c.animal == an]
            for col, colr, lab, dx in ((f"nokin:u:tgtbin|{w}", TC, "target bins", -0.08),
                                       (f"nokin:u:exbin|{w}", XC, "executed bins", 0.08)):
                m = [a[a.epoch == e][col].mean() for e in EP]
                for i, e in enumerate(EP):
                    v = a[a.epoch == e][col].to_numpy()
                    ax.scatter(np.full(len(v), i + dx), v, s=8, color=colr, alpha=0.35, lw=0)
                    lo, hi = min(lo, np.nanmin(v) if len(v) else lo), max(hi, np.nanmax(v) if len(v) else hi)
                ax.plot(np.arange(4) + dx, m, "-o", color=colr, label=lab, lw=2, ms=6)
            ax.axhline(0, color="0.6", lw=0.8)
            n = [int((a.epoch == e).sum()) for e in EP]
            ax.set_title(f"{an}  (n {'/'.join(map(str, n))})" if r == 0 else "", fontsize=10)
            ax.set_xticks(range(4))
            ax.set_xticklabels(EP)
            if k == 0:
                ax.set_ylabel(f"{w} window\nunique dR2 (R2 fraction)")
        for k in range(4):
            axes[r, k].set_ylim(max(lo, -45), min(hi, 80))
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Binned target vs executed contest (O2, final inputs, width 1), cortex: unique contribution of each "
                 "set beyond the other\nlines = epoch means, dots = sessions; chronic target > executed 4/4, acute "
                 "lick 3/4, subacute mixed, pre no lead", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(png, dpi=140)
    plt.close(fig)


def draw_fig2(fd: dict, png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, w in enumerate(WINDOWS):
        s = fd["summary"][w]
        ax = axes[0, j]
        for k, an in enumerate(AN):
            for i, cl in enumerate(CL):
                x = s[(s.animal == an) & (s.cls == cl)]
                if len(x) and x.n.iloc[0] >= 5:
                    ax.bar(k * 7 + i, x.p_true.iloc[0], yerr=x.p_true_sem.iloc[0], color=CC[cl], width=0.85,
                           label=cl if k == 0 else None, error_kw={"lw": 0.8})
        ax.axhline(1 / 6, color="k", ls=":", lw=1)
        ax.set_xticks([k * 7 + 2.5 for k in range(4)])
        ax.set_xticklabels(AN)
        ax.set_title({"precue": "ENL (pre-cue, before movement)", "cue": "cue window", "lick": "lick window"}[w],
                     fontsize=10)
        if j == 0:
            ax.set_ylabel("frozen decoder P(true spout)\nfar contra (chance 1/6 dotted)")
        ax = axes[1, j]
        for k, an in enumerate(AN):
            t = fd["trials"][(w, an)]
            for i, cl in enumerate(CL):
                x = t[t.cls == cl]
                if len(x) >= 5:
                    ax.bar(k * 7 + i, x.pred.isin(["far_ipsi", "far_mid", "far_contra"]).mean(), color=CC[cl],
                           width=0.85)
        ax.set_xticks([k * 7 + 2.5 for k in range(4)])
        ax.set_xticklabels(AN)
        ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel("share decoded as ANY far spout")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Item 1: far-contra trials on the frozen pre-stroke decoder, by what the animal did (acute; pre_hit = "
                 "pre-stroke reference)\nfailed attempts (orange) lose the far-contra code but are read as a FAR spout "
                 "(distance kept, side lost); no pre-cue difference", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(png, dpi=140)
    plt.close(fig)


def draw_fig3(h: pd.DataFrame, png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for j, (ref, title) in enumerate([("pre_hit", "relative to PRE-STROKE HITS (touch + water)"),
                                      ("pre_incomplete", "relative to PRE-STROKE MISSES (no touch, no water)")]):
        ax = axes[j]
        for k, an in enumerate(AN):
            for i, (e, cl, colr, lab) in enumerate([("acute", "hit", "#1b9e77", "acute hits"),
                                                    ("acute", "incomplete", "#d95f02", "acute failed attempts")]):
                try:
                    r, rf = h.loc[(an, e, cl)], h.loc[(an, "pre", ref)]
                except KeyError:
                    continue
                v = np.mean([r.ipsi / rf.ipsi, r.contra / rf.contra])
                ax.bar(k * 3 + i, v, color=colr, width=0.85, label=lab if k == 0 else None)
        ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.set_xticks([k * 3 + 0.5 for k in range(4)])
        ax.set_xticklabels(AN)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("lick-window amplitude ratio\n(mean of both hemispheres)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Item 4: how big is the far-contra response on failed attempts?\nSmaller than hits in 3/4, but vs "
                 "touch-matched pre-stroke misses only 2/4 (PS92's reference = 7 trials)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(png, dpi=140)
    plt.close(fig)


LOAD = {"fig1": load_fig1, "fig2": load_fig2, "fig3": load_fig3}
DRAW = {"fig1": (draw_fig1, "fig1_binned_contest.png"), "fig2": (draw_fig2, "fig2_attempt_decoder.png"),
        "fig3": (draw_fig3, "fig3_attempt_amplitude.png")}


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--enc-dir", type=Path, default=E_DEF, help="LP_analysis/<run>/imaging_encoding folder")
    ap.add_argument("--only", nargs="+", choices=FIGS, default=list(FIGS), help="figures to draw (default all)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        fd = _common.load_figdata(a.out, NAME)
    else:
        fd = {k: LOAD[k](_fig1_root() if (k == "fig1" and a.enc_dir == E_DEF) else a.enc_dir) for k in a.only}
        fd["src"] = str(a.enc_dir)
        _common.save_figdata(a.out, NAME, fd)
    for k in a.only:
        fn, png = DRAW[k]
        fn(fd[k], a.out / png)
        print("->", a.out / png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
