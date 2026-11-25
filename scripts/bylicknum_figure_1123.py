"""Candidate supplementary figure: kinematic change from own pre BY LICK NUMBER in the trial, both cohorts
(CC-B, 2026-11-23; suffix _bylicknum_ccB1123). Reads the cell tables written by
stroke_orofacial_pipeline scripts/manuscript/c1_bylicknum.py (cohort 1) and scripts/sixpos_bylicknum_1123.py (cohort 2).

Main figure: rows = peak speed, peak protrusion (cohort 1, xy) / peak 3-D reach (cohort 2); columns = cohort 1
contralesional cue, one panel per post epoch (grades as colors), then cohort 2 pooled over positions and far contra
(epochs as colors). Companion figure: cohort 1 ipsilesional cue. x = lick number 1-15; y = bootstrap change from the
animal's own pre at the same lick number (line) with its 95% CI (band); 0 = pre. Stars in the top band (colored by
line) = late-minus-early contrast (change at licks 6-15 minus change at licks 1-5) starred in its family.
Single-animal lines (#5, moderate acute) dashed, no band, never starred.

    python scripts/bylicknum_figure_1123.py DIR      (DIR = bylicknum/, holds c1/ and c2/; NEW files only)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wfield_local import sig_marks  # noqa: E402

D = Path(sys.argv[1])
SUF = "_bylicknum_ccB1123"
GRADE_COL = {"mild": "#f2c12e", "moderate": "#f0822e", "severe": "#b4001e"}
EP_COL = {"acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4"}
EPOCHS = ("acute", "subacute", "chronic")
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
                     "svg.fonttype": "none"})
C1 = pd.read_csv(D / "c1" / f"c1_cells{SUF}.csv")
C2 = pd.read_csv(D / "c2" / f"c2_cells{SUF}.csv")
X = np.arange(1, 16)


def _line(ax, d, col, single=False):
    d = d.assign(x=d.bin.astype(int)).sort_values("x")
    if single:
        ax.plot(d.x, d.wa1_effect, color=col, lw=1.0, ls="--", marker="o", ms=1.8, zorder=3)
        return
    ax.fill_between(d.x, d.lo, d.hi, color=col, alpha=0.18, lw=0, zorder=1)
    ax.plot(d.x, d.est, color=col, lw=1.1, marker="o", ms=1.8, zorder=3)


def _tidy(ax, ylab=None, title=None, xlab=True):
    ax.axhline(0, color="0.5", lw=0.6, ls="--", zorder=0)
    ax.axvline(5.5, color="0.85", lw=0.6, zorder=0)
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks([1, 5, 10, 15])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if ylab:
        ax.set_ylabel(ylab)
    if title:
        ax.set_title(title, fontsize=6.5, pad=10)
    if xlab:
        ax.set_xlabel("Lick number in trial")


def _mark(t, col="nb_effect"):
    """Star plus direction: down arrow = later licks worse than early (deficit builds); up = deficit shrinks."""
    return "*↓" if t[col].iloc[0] < 0 else "*↑"


def c1_panel(ax, meas, side, epoch, row):
    d = C1[(C1.measure == meas) & (C1.side == side) & (C1.epoch == epoch)]
    srow = 0
    for g, col in GRADE_COL.items():
        dl = d[(d.grade == g) & (d.kind == "licknum")].rename(columns={"nb_effect": "est", "nb_ci_lo": "lo",
                                                                      "nb_ci_hi": "hi"})
        if dl.empty:
            continue
        single = bool(dl.single_animal.all())
        _line(ax, dl, col, single)
        t = d[(d.grade == g) & (d.bin == "late_minus_early")]
        if len(t) and bool(t.star.iloc[0]):
            sig_marks.star_top(ax, 13.5 - 2.6 * srow, _mark(t), color=col, fontsize=8)
            srow += 1
        if single:
            ax.text(15.3, dl.wa1_effect.iloc[-1], "#5", color=col, fontsize=5.5, va="center", ha="left")


def c2_panel(ax, meas, pos):
    d = C2[(C2.measure == meas) & (C2.pos == pos)]
    srow = 0
    for e, col in EP_COL.items():
        dl = d[(d.epoch == e) & (d.kind == "licknum")].rename(columns={"boot_change": "est", "boot_lo": "lo",
                                                                      "boot_hi": "hi"})
        _line(ax, dl, col)
        t = d[(d.epoch == e) & (d.bin == "late_minus_early")]
        if len(t) and bool(t.star.iloc[0]):
            sig_marks.star_top(ax, 13.5 - 2.6 * srow, _mark(t, "boot_change"), color=col, fontsize=8)
            srow += 1


def save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        p = D / f"{stem}{SUF}.{ext}"
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
        fig.savefig(p, dpi=600 if ext == "png" else None)
        print("wrote", p)


ROWS = [("speed", "Peak speed change\n(c1 px/s; c2 mm/s)"), ("xy", "Protrusion / reach change\n(c1 px; c2 mm)")]
C2MEAS = {"speed": "speed", "xy": "reach"}

# ---- main figure: cohort 1 contralesional (3 epochs) + cohort 2 pooled / far contra
fig, axs = plt.subplots(2, 5, figsize=(7.0, 3.3), gridspec_kw={"wspace": 0.62, "hspace": 0.55,
                                                                 "width_ratios": [1, 1, 1, 1.08, 1.08]})
for r, (meas, _ylab) in enumerate(ROWS):
    for c, e in enumerate(EPOCHS):
        ax = axs[r, c]
        c1_panel(ax, meas, "contralesional", e, r)
        _tidy(ax, ("Peak speed change (px/s)" if meas == "speed" else "Protrusion change (px)") if c == 0 else None,
              f"Cohort 1, contra cue\n{e}" if r == 0 else None, xlab=(r == 1))
    for c, (pos, lab) in enumerate([("pooled", "all positions"), ("far_contra", "far contra")]):
        ax = axs[r, 3 + c]
        c2_panel(ax, C2MEAS[meas], pos)
        _tidy(ax, ("Peak speed change (mm/s)" if meas == "speed" else "3-D reach change (mm)") if c == 0 else None,
              f"Cohort 2, {lab}" if r == 0 else None, xlab=(r == 1))
    # shared y within cohort-1 row and within cohort-2 row
    for grp in (axs[r, :3], axs[r, 3:]):
        lo = min(a.get_ylim()[0] for a in grp)
        hi = max(a.get_ylim()[1] for a in grp)
        for a in grp:
            a.set_ylim(lo, hi)
h1 = [plt.Line2D([], [], color=c, lw=1.2, label=g) for g, c in GRADE_COL.items()]
h2 = [plt.Line2D([], [], color=c, lw=1.2, label=e) for e, c in EP_COL.items()]
fig.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.08, -0.01), ncol=3, frameon=False, fontsize=6,
           title="Cohort 1 grade", title_fontsize=6)
fig.legend(handles=h2, loc="lower right", bbox_to_anchor=(0.95, -0.01), ncol=3, frameon=False, fontsize=6,
           title="Cohort 2 epoch", title_fontsize=6)
sig_marks.reserve_headroom(fig)
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.2)
save(fig, "FigS_kinematics_bylicknum")
plt.close(fig)

# ---- companion: cohort 1 ipsilesional cue
fig, axs = plt.subplots(2, 3, figsize=(4.3, 3.3), gridspec_kw={"wspace": 0.45, "hspace": 0.55})
for r, (meas, _) in enumerate(ROWS):
    for c, e in enumerate(EPOCHS):
        ax = axs[r, c]
        c1_panel(ax, meas, "ipsilesional", e, r)
        _tidy(ax, ("Peak speed change (px/s)" if meas == "speed" else "Protrusion change (px)") if c == 0 else None,
              f"Cohort 1, ipsi cue\n{e}" if r == 0 else None, xlab=(r == 1))
    lo = min(a.get_ylim()[0] for a in axs[r])
    hi = max(a.get_ylim()[1] for a in axs[r])
    for a in axs[r]:
        a.set_ylim(lo, hi)
fig.legend(handles=h1, loc="lower center", ncol=3, frameon=False, fontsize=6)
sig_marks.reserve_headroom(fig)
fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.2)
save(fig, "FigS_kinematics_bylicknum_c1ipsi")
plt.close(fig)
