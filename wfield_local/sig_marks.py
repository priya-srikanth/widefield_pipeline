"""SIGNIFICANCE ASTERISK on a bar / point plot. One definition, so every figure marks significance the same way.

Priya 2026-10-25: "for all the bar graphs, the asterisks are hard to see". They were 9-12 pt, drawn AT the whisker tip,
where the animal dots and the neighbouring bar's whisker sit on top of them. Here: 13 pt bold, a white halo, and placed
just beyond the outermost of the whisker end, zero and the animal values, on the side of the effect (above an increase,
below a decrease).
"""
from __future__ import annotations

import numpy as np


def sig_star(ax, x: float, lo: float, hi: float, values=(), *, text: str = "*", fontsize: float = 13, color: str = "k"):
    """Mark x when the interval [lo, hi] excludes 0; `values` = the animal dots plotted at x (same scale as lo / hi).
    Returns the annotation, or None when the interval contains 0."""
    import matplotlib.patheffects as pe
    if not (lo > 0 or hi < 0):
        return None
    up = lo > 0
    v = [float(u) for u in values if u is not None and np.isfinite(u)]
    ext = max([hi, 0.0, *v]) if up else min([lo, 0.0, *v])
    return ax.annotate(text, (x, ext), xytext=(0, 1 if up else -1), textcoords="offset points", ha="center",
                       va="bottom" if up else "top", fontsize=fontsize, fontweight="bold", color=color, zorder=6,
                       annotation_clip=False, path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])


# ---- TOP-BAND marks (Priya 10-30, ALL figures): "place significance stars at top of figure space as possible, so they're
# visible at a glance and don't overlap with figure points or horizontal 'pre' lines". Prefer these over `sig_star`
# for new figures (same rule as orofacial_cohort `_mark_rows` / `_tidy`, 10-17):
#     star_top(ax, x, "**", color, row=0)     # x in data units; y fixed in axes fraction, row 0 = the top line
#     reserve_headroom(fig)                   # once, after all data are drawn, BEFORE tight_layout / savefig
# `reserve_headroom` grows each marked axes' y range (once per shared-y group) so the data fill the bottom part and the
# marks sit in the empty band above them. One row per mark set sharing an x (e.g. one per epoch).
STEP = 0.075        # axes fraction per row of marks
PAD = 0.03          # extra free space under the lowest row


def star_top(ax, x: float, text: str = "*", color="k", row: int = 0, fontsize: float = 12, step: float = STEP):
    """Draw `text` at data x in the top band of the axes (row 0 = top line); tag the axes for `reserve_headroom`."""
    import matplotlib.patheffects as pe
    t = ax.text(x, 0.99 - step * row, text, color=color, ha="center", va="top", fontsize=fontsize, fontweight="bold",
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
    ax._wf_mark_rows = max(getattr(ax, "_wf_mark_rows", 0), row + 1)
    ax._wf_mark_step = step
    return t


def reserve_headroom(fig) -> None:
    """Grow the y range of every axes carrying `star_top` marks so the top band is free of data (call once)."""
    done = set()
    for ax in fig.get_axes():
        if not getattr(ax, "_wf_mark_rows", 0):
            continue
        sib = ax.get_shared_y_axes().get_siblings(ax) or [ax]
        key = frozenset(id(a) for a in sib)
        if key in done:
            continue
        done.add(key)
        n = max(getattr(a, "_wf_mark_rows", 0) for a in sib)
        lo, hi = ax.get_ylim()
        frac = getattr(ax, "_wf_mark_step", STEP) * n + PAD
        ax.set_ylim(lo, hi + (hi - lo) * frac / (1 - frac))
