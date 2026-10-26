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
