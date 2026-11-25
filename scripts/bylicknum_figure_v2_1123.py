"""Candidate supplementary figure v2: kinematic change from own pre BY LICK NUMBER in the trial, both cohorts
(CC-B, 2026-11-23; suffix _bylicknum_v2_ccB1123). Priya 11-23 changes over v1: manuscript palettes (vls_style EP for the
six-position epochs, SEV for the two-tone grades), cohort names as in the manuscript ("two-tone", "six-position"),
a third row = peak tongue angle (session MEAN per lick number; + = contralesional), and a legend that explains the
arrows and the dashed single-animal line.

Reads the v2 cell tables of stroke_orofacial_pipeline scripts/manuscript/c1_bylicknum.py (two-tone) and
scripts/sixpos_bylicknum_1123.py (six-position).

Main figure: rows = peak speed, peak protrusion (two-tone, xy) / peak 3D reach (six-position), peak tongue angle
(two-tone 2D all-lick peak angle, Fig 2G source / six-position 3D aim angle, Fig 4E definition). Columns = two-tone
contralesional cue, one panel per post epoch (grades as colors), then six-position all positions and far contra
(epochs as colors). Companion figure: two-tone ipsilesional cue. x = lick number 1-15; y = bootstrap change from the
animal's own pre at the same lick number (line) with its 95% CI (band); 0 = pre. Marks in the top band (colored by
line) = the late-minus-early contrast starred in its family, with an arrow for its sign.

    python scripts/bylicknum_figure_v2_1123.py DIR      (DIR = bylicknum/, holds c1_v2/ and c2_v2/; NEW files only)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from wfield_local import sig_marks  # noqa: E402

D = Path(sys.argv[1])
SUF = "_bylicknum_v2_ccB1123"
#: the manuscript palettes, copied exactly from manuscript_code/figures/vls_style.py (SEV, EP)
GRADE_COL = {"mild": "#F2C02A", "moderate": "#E0702A", "severe": "#8E0012"}
EP_COL = {"acute": "#AA3377", "subacute": "#4477AA", "chronic": "#228833"}
EPOCHS = ("acute", "subacute", "chronic")
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
                     "svg.fonttype": "none"})
C1 = pd.read_csv(D / "c1_v2" / f"c1_cells{SUF}.csv")
C2 = pd.read_csv(D / "c2_v2" / f"c2_cells{SUF}.csv")
ROWS = [("speed", "speed", "Δ peak speed (px/s)", "Δ peak speed (mm/s)"),
        ("xy", "reach", "Δ peak protrusion (px)", "Δ peak 3D reach (mm)"),
        ("angle", "angle", "Δ tongue angle (deg;\n+ = contralesional)", "Δ 3D angle (deg;\n+ = contralesional)")]
LEGEND_NOTE = ("Line, bootstrap change from each animal's own pre-stroke value at the same lick number; band, 95% CI; gray "
               "dashed line at 0 = pre.\n* with arrow: late-minus-early change (licks 6-15 minus licks 1-5) significant; "
               "arrow = its direction (down = deficit builds over the bout, or angle more ipsilesional;\nup = shrinks). "
               "Dashed colored line: moderate acute, #5 only (#6 had no acute session passing the lick minimum), not tested.")


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


def _mark(est):
    return "*↓" if est < 0 else "*↑"


def c1_panel(ax, meas, side, epoch):
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
            sig_marks.star_top(ax, 13.5 - 2.6 * srow, _mark(t.nb_effect.iloc[0]), color=col, fontsize=8)
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
            sig_marks.star_top(ax, 13.5 - 2.6 * srow, _mark(t.boot_change.iloc[0]), color=col, fontsize=8)
            srow += 1


def _share(axes):
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)


def save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        p = D / f"{stem}{SUF}.{ext}"
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
        fig.savefig(p, dpi=600 if ext == "png" else None)
        print("wrote", p)


h1 = [plt.Line2D([], [], color=c, lw=1.2, label=g) for g, c in GRADE_COL.items()]
h2 = [plt.Line2D([], [], color=c, lw=1.2, label=e) for e, c in EP_COL.items()]

# ---- main figure: two-tone contralesional cue (3 epochs) + six-position all positions / far contra
fig, axs = plt.subplots(3, 5, figsize=(7.0, 4.9), gridspec_kw={"wspace": 0.62, "hspace": 0.55,
                                                                 "width_ratios": [1, 1, 1, 1.08, 1.08]})
for r, (m1, m2, y1, y2) in enumerate(ROWS):
    last = r == len(ROWS) - 1
    for c, e in enumerate(EPOCHS):
        c1_panel(axs[r, c], m1, "contralesional", e)
        _tidy(axs[r, c], y1 if c == 0 else None, f"two-tone, contralesional cue\n{e}" if r == 0 else None, last)
    for c, (pos, lab) in enumerate([("pooled", "all positions"), ("far_contra", "far contra")]):
        c2_panel(axs[r, 3 + c], m2, pos)
        _tidy(axs[r, 3 + c], y2 if c == 0 else None, f"six-position\n{lab}" if r == 0 else None, last)
    _share(axs[r, :3])
    _share(axs[r, 3:])
fig.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.07, 0.085), ncol=3, frameon=False, fontsize=6,
           title="two-tone grade", title_fontsize=6)
fig.legend(handles=h2, loc="lower right", bbox_to_anchor=(0.97, 0.085), ncol=3, frameon=False, fontsize=6,
           title="six-position epoch", title_fontsize=6)
fig.text(0.07, 0.012, LEGEND_NOTE, fontsize=5.5, va="bottom", ha="left", linespacing=1.3)
sig_marks.reserve_headroom(fig)
fig.subplots_adjust(left=0.085, right=0.98, top=0.92, bottom=0.215)
save(fig, "FigS_kinematics_bylicknum_v2")
plt.close(fig)

# ---- companion: two-tone ipsilesional cue
fig, axs = plt.subplots(3, 3, figsize=(4.3, 4.9), gridspec_kw={"wspace": 0.5, "hspace": 0.55})
for r, (m1, _m2, y1, _y2) in enumerate(ROWS):
    last = r == len(ROWS) - 1
    for c, e in enumerate(EPOCHS):
        c1_panel(axs[r, c], m1, "ipsilesional", e)
        _tidy(axs[r, c], y1 if c == 0 else None, f"two-tone, ipsilesional cue\n{e}" if r == 0 else None, last)
    _share(axs[r])
fig.legend(handles=h1, loc="lower center", bbox_to_anchor=(0.55, 0.09), ncol=3, frameon=False, fontsize=6,
           title="two-tone grade", title_fontsize=6)
fig.text(0.03, 0.008, LEGEND_NOTE.replace(" (down", "\n(down").replace(";\nup", "; up").replace(". Dashed colored", ".\nDashed colored"), fontsize=4.6, va="bottom",
         ha="left", linespacing=1.3)
sig_marks.reserve_headroom(fig)
fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.235)
save(fig, "FigS_kinematics_bylicknum_v2_twotone_ipsi")
plt.close(fig)
