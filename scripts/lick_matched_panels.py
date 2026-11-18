"""One multi-panel like-for-like figure (Priya 11-17: "a figure that has multiple panels ... instead of multiple figures
with the same cued panel"; "pick loclin with gaussian kernel, and the two versions would be with and without like-for-like
matching to pre-stroke licks"). Columns = cued all licks, cued FIRST lick, ITI all licks, ITI FIRST lick (bout start),
ITI DOCKED licks. Rows:
  all     every post-stroke lick; the local-linear reference carries the edge line beyond the pre landing range
  inside  post-stroke licks only where the SAME animal's pre curve of the SAME lick set exists (strict like-for-like;
          lick_matched_readout.inside_curves; pre licks unchanged)
Curves = Gaussian-smoothed decoded vs landing per animal (likeforlike.smooth_curves, the loclin sigma), mean +/- SEM of 4
animals, dashed where only 3; thinner and translucent (Priya 11-17 "hard to see where the curves overlap"). Bars = 0.25-mm
bins whose change vs pre survives BH (the row's lick set). (11-17 first draft: a gauss row vs a loclin row with the same
curves; they came out almost identical, so the Gaussian reference was dropped from this figure.)
Plot only: reads the figure data of lick_matched_readout.py --ref loclin --inside-curves runs (As / Ains / BTin, first-lick
twins As1 / Ains1 / BT1in with --first-curves), passed as --src all=<pkl of the all-ITI run with --first-curves>
--src dock=<pkl of the WFIELD_ITI_FILTER=dock run>. Writes <iti_licks.OUT_W>/figures/lick_matched_panels.png (+ svg).
Panel drawing = lick_matched_readout.curve_panel."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import iti_licks as il
import lick_matched_readout as lmr
import likeforlike as lfl
from wfield_local import trial_set_text as tst
from wfield_local.figure_layout import save_png_svg

COLS = [("cued, all licks", "all", "", "trial"), ("cued, FIRST lick", "all", "1", "trial"),
        ("ITI, all licks", "all", "", "iti"), ("ITI, FIRST lick (bout start)", "all", "1", "iti"),
        ("ITI, DOCKED licks", "dock", "", "iti")]
ROWS = [("all", "As{f}", "BT{f}", "ALL post-stroke licks\n(edge line carried beyond the pre range)"),
        ("inside", "Ains{f}", "BT{f}in", "INSIDE each animal's own pre curve\n(same lick set; strict like-for-like)")]
LW, ALPHA = 1.2, 0.75          # thinner, translucent curves (Priya 11-17)


def sources() -> dict:
    src = dict(a.split("=", 1) for a in (sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--src"))
    return {k: pickle.loads(Path(v).read_bytes()) for k, v in src.items()}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    D = sources()
    col = {"pre": "k", **EPOCH_LINE}       # legend = short lines, same width / opacity as the curves (Priya 11-17)
    epl = [Line2D([], [], color=col[e], lw=LW, alpha=ALPHA, label=e) for e in lmr.EPS]
    tst.declare(text="cued licks (3-D lick table, post-cue, inside a trial's response window) and ITI licks (after a "
                     "trial's response window, before the next spout arrival); ITI docked = onset while the spout is "
                     "docked (WFIELD_ITI_FILTER=dock, frozen all-ITI decoders); first lick = cued order 0 / ITI bout "
                     "start; animals PS92-95, pre = August sessions")
    fig, axs = plt.subplots(len(ROWS), len(COLS), figsize=(4.0 * len(COLS), 3.7 * len(ROWS) + 0.6), squeeze=False,
                            sharex=True, sharey=True)      # one x / y scale across panels (Priya 11-17)
    dashed = []      # why a line is dashed: one line UNDER the figure, not inside a panel (Priya 11-17 "obscures the plot")
    for i, (_row, ak, bk, _lab) in enumerate(ROWS):
        for j, (head, lset, f, st) in enumerate(COLS):
            A = D[lset][ak.format(f=f)]
            if i == 0:
                dashed += [f"{head}: {n}" for n in lfl.dash_notes(A, st, lmr.EPS, lmr.CURVE_MIN_W)]
            lmr.curve_panel(axs[i, j], A, D[lset][bk.format(f=f)], st, head, place=True, fs=9.5, sub=False, lw=LW,
                            alpha=ALPHA, epl=epl)
            if j:
                axs[i, j].set_ylabel("")
            if i < len(ROWS) - 1:
                axs[i, j].set_xlabel("")
    sg = D["all"]["As"]["sigma"].iloc[0]
    dec = ("NON-LINEAR (RBF kernel ridge)" if lmr.lo.SOURCE == "nonlinear" else "KEY-row")    # WFIELD_LICK_DECODER
    # title: not "LIKE-FOR-LIKE" -- the curves are decoded vs landed per epoch; the matched comparison is the bars /
    # read-outs, and only the bottom row is strictly matched (Priya 11-17)
    fig.suptitle(f"Decoded vs landed lateral position, before and after the stroke (frozen {dec} lateral decoder). "
                 "A curve shifted UP = read more contralesional than pre licks landing at the same place."
                 f"\nCurves: Gaussian-smoothed per animal (sigma {sg:g} mm), mean +/- SEM of 4 animals; dashed = "
                 "only 3 animals there. Reference for the bars: local-linear on pre licks.\nBars at the top = 0.25-mm "
                 "bins whose change vs pre survives BH (paired nested bootstrap; rows acute / subacute / chronic). "
                 "PROVISIONAL", fontsize=10, y=0.995)
    fb = tst.footer_bottom(fig)
    fig.tight_layout(rect=(0.035, fb + (0.035 if dashed else 0), 1, 0.955))
    if dashed:
        txt = "Dashed lines: " + "; ".join(dashed)
        print(txt, flush=True)
        fig.text(0.5, fb + 0.004, txt, ha="center", va="bottom", fontsize=8, color="0.25", wrap=True)
    for i, (_row, _a, _b, lab) in enumerate(ROWS):    # row label left of the y-axis labels, centred on the row
        b = axs[i, 0].get_position()
        fig.text(0.008, b.y0 + b.height / 2, lab, rotation=90, ha="left", va="center", fontsize=9, fontweight="bold")
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, il.OUT_W / "figures" / "lick_matched_panels.png", dpi=100)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
