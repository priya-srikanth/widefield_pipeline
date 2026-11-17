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
  layout and dpi unchanged. Changes: argparse (--out / --plot-only / --enc-dir / --only) instead of sys.argv[1],
  figdata pickle (<out>/data/goal_summary_figdata.pkl: the filtered rows each figure draws), figures can be drawn one at
  a time (--only fig2).
  SUPTITLE READINGS ARE COMPUTED (Priya 11-06, A22): the 10-24 suptitles carried readings typed in by hand ("chronic
  target > executed 4/4, acute lick 3/4, subacute mixed, pre no lead"; "failed attempts (orange) lose the far-contra
  code but are read as a FAR spout ...; no pre-cue difference"; fig3 "smaller than hits in 3/4 ... only 2/4"). They are
  now COUNTS of animals computed from exactly the numbers the panels draw (`fig1_reading` / `fig2_reading` /
  `fig3_reading`); the interpretive words ("mixed", "distance kept, side lost") are dropped -- a count cannot carry
  them honestly. The figure states what the data show; the page caption interprets.

INPUTS, under --enc-dir (default = the original's hard-coded
  M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding):
  attempt_coding/summary_{precue,cue,lick}.csv and attempt_coding/trials_{precue,cue,lick}_<animal>.csv.gz
      (wfield_local/attempt_coding.py, WIDEFIELD_LICK_SOURCE=lp; method "dom"; the decoder is the LP-trained
      frozen decoder) -> fig2
  movement_first_subset-nokin_exec-polar_licks-combined_bins/tables/region_measures.csv (O2 encoding) -> fig1
  attempt_maps/hemispheres_lick.csv (wfield_local/attempt_maps.py) -> fig3
REBUILD: re-run attempt_coding.py (decision D7; Priya 11-09: the LP-trained frozen decoder is THE LP model),
  attempt_maps.py and the O2 contest bundle, then point --enc-dir at the new imaging_encoding folder.
OUT: <out>/fig1_binned_contest.png, fig2_attempt_decoder.png, fig3_attempt_amplitude.png
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

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


# ---------------------------------------------------------------- readings (A22: computed from what the panels draw)
FAR = ["far_ipsi", "far_mid", "far_contra"]
MIN_N = 5          # the panels' own floor (bars drawn for classes with >= 5 trials)


def fig1_reading(c: pd.DataFrame) -> str:
    """Per epoch and window: animals whose epoch-mean target-bin unique dR2 exceeds the executed-bin one (the two lines
    each panel draws), out of the animals with both."""
    parts = []
    for e in EP:
        cnt = []
        for w in ("cue", "lick"):
            n = k = 0
            for an in AN:
                a = c[(c.animal == an) & (c.epoch == e)]
                t, x = a[f"nokin:u:tgtbin|{w}"].mean(), a[f"nokin:u:exbin|{w}"].mean()
                if np.isfinite(t) and np.isfinite(x):
                    n, k = n + 1, k + int(t > x)
            cnt.append(f"{w} {k}/{n}")
        parts.append(f"{e} " + ", ".join(cnt))
    return "target > executed (epoch means; animals): " + "; ".join(parts)


def fig2_reading(fd: dict) -> str:
    """Per window: animals whose acute INCOMPLETE (failed-attempt) P(true spout) is below their pre-stroke-hit one (top
    row, both classes >= MIN_N trials), and animals whose incomplete trials are decoded as a far spout more than half
    the time (bottom row; half = chance for 3 of 6 spouts)."""
    lo, far = [], []
    for w in WINDOWS:
        s = fd["summary"][w]
        n = k = 0
        for an in AN:
            r = {cl: s[(s.animal == an) & (s.cls == cl)] for cl in ("incomplete", "pre_hit")}
            if all(len(v) and v.n.iloc[0] >= MIN_N for v in r.values()):
                n, k = n + 1, k + int(r["incomplete"].p_true.iloc[0] < r["pre_hit"].p_true.iloc[0])
        lo.append(f"{w} {k}/{n}")
        n = k = 0
        for an in AN:
            t = fd["trials"].get((w, an))
            if t is None:
                continue
            x = t[t.cls == "incomplete"]
            if len(x) >= MIN_N:
                n, k = n + 1, k + int(x.pred.isin(FAR).mean() > 0.5)
        far.append(f"{w} {k}/{n}")
    return ("failed attempts (incomplete, orange): P(true) below pre-stroke hits in " + ", ".join(lo)
            + ";\nfailed attempts decoded as a far spout on > 1/2 of trials in " + ", ".join(far) + " (animals)")


def _fig3_ratio(h: pd.DataFrame, an: str, cl: str, ref: str) -> float:
    """The bar `draw_fig3` draws: acute ``cl`` amplitude / pre ``ref`` amplitude, mean of both hemispheres."""
    try:
        r, rf = h.loc[(an, "acute", cl)], h.loc[(an, "pre", ref)]
    except KeyError:
        return np.nan
    return float(np.mean([r.ipsi / rf.ipsi, r.contra / rf.contra]))


def fig3_reading(h: pd.DataFrame) -> str:
    """Per panel (reference): animals whose acute failed-attempt amplitude ratio is below their acute-hit ratio."""
    parts = []
    for ref, lab in (("pre_hit", "vs pre-stroke hits"), ("pre_incomplete", "vs pre-stroke misses")):
        n = k = 0
        for an in AN:
            f, t = _fig3_ratio(h, an, "incomplete", ref), _fig3_ratio(h, an, "hit", ref)
            if np.isfinite(f) and np.isfinite(t):
                n, k = n + 1, k + int(f < t)
        parts.append(f"{lab} {k}/{n}")
    return "failed attempts smaller than hits (animals): " + ", ".join(parts)


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
            ax.spines["top"].set_visible(False)          # Priya 11-03: no top frame line
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
                 "set beyond the other\nlines = epoch means, dots = sessions; " + fig1_reading(c), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_svg(fig, png, dpi=140)
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
        for b in axes[:, j]:
            b.spines["top"].set_visible(False)
        if j == 0:
            ax.set_ylabel("share decoded as ANY far spout")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Item 1: far-contra trials on the frozen pre-stroke decoder, by what the animal did (acute; pre_hit = "
                 "pre-stroke reference)\n" + fig2_reading(fd), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))           # three title lines (the computed reading, A22)
    save_png_svg(fig, png, dpi=140)
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
        ax.spines["top"].set_visible(False)
    axes[0].set_ylabel("lick-window amplitude ratio\n(mean of both hemispheres)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Item 4: how big is the far-contra response on failed attempts?\n" + fig3_reading(h), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_png_svg(fig, png, dpi=140)
    plt.close(fig)


LOAD = {"fig1": load_fig1, "fig2": load_fig2, "fig3": load_fig3}
DRAW = {"fig1": (draw_fig1, "fig1_binned_contest.png"), "fig2": (draw_fig2, "fig2_attempt_decoder.png"),
        "fig3": (draw_fig3, "fig3_attempt_amplitude.png")}


def fig3_sentence() -> str:
    """fig 3: the lick-window class maps of `attempt_maps` (no-lick rows have no lick window)."""
    from wfield_local import attempt_maps as am
    from wfield_local import trial_set_text as tst
    return tst.events(
        f"far-contra lick rows ({tst.lick_text()}: first lick within the engaged cut), lick window, by camera class",
        unit="trials",
        keep=[f"acute hit / acute incomplete and the pre-stroke-hit reference: sessions with >= {am.MIN_TRIALS} such "
              "trials, session-weighted", "pre-stroke incomplete reference: trial-pooled, no per-session floor"],
        excluded=["no-lick rows (no attempt, reaction, stopped)", "subacute", "chronic", "other spouts"])


#: the trial set of each figure (they differ: encoder frames / attempt-class trials / lick-window class maps)
SENTENCES = {"fig1": lambda: _common.contest_frames(),
             "fig2": lambda: _common.attempt_classes(epochs="acute", windows=_common.windows_text(
                 ["precue", "cue", "lick"]), min_n=MIN_N),
             "fig3": fig3_sentence}


def trial_set_sentence() -> str:
    """All three figures' sets (the driver-level fallback; each figure is saved under its own, `SENTENCES`)."""
    from wfield_local import trial_set_text as tst
    return tst.union(*(f() for f in SENTENCES.values()), unit="trials / frames")


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--enc-dir", type=Path, default=E_DEF, help="LP_analysis/<run>/imaging_encoding folder")
    ap.add_argument("--only", nargs="+", choices=FIGS, default=list(FIGS), help="figures to draw (default all)")
    a = ap.parse_args(argv)
    _common.declare(trial_set_sentence())          # THE TRIAL SET of every figure (Priya 11-09)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        fd = _common.load_figdata(a.out, NAME)
    else:
        fd = {k: LOAD[k](_fig1_root() if (k == "fig1" and a.enc_dir == E_DEF) else a.enc_dir) for k in a.only}
        fd["src"] = str(a.enc_dir)
        _common.save_figdata(a.out, NAME, fd)
    for k in a.only:
        fn, png = DRAW[k]
        from wfield_local import trial_set_text as tst
        with tst.active(None, text=SENTENCES[k]()):          # each figure's own set (Priya 11-09)
            fn(fd[k], a.out / png)
        print("->", a.out / png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
