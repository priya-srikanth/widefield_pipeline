"""Wrap a per-session figure onto a grid instead of one long row.

WHY THIS EXISTS. Several figures were ``subplots(1, N)`` with N = SESSIONS, so every panel shrank as
the cohort grew: ten post-stroke sessions in a single row left each one unreadable, and it gets
worse every night (Priya, 2026-08-21). Wrapping at a few columns keeps each panel legible no matter
how many sessions accumulate.

Kept in its own module rather than in ``plot_poststroke`` so the decoder/encoder/RSA figures can use
it without importing the post-stroke plotting stack, which pulls in the whole section-G world for
two four-line helpers.
"""
from __future__ import annotations

import numpy as np

#: Panels per row before wrapping. Four keeps a session panel wide enough to read its annotations;
#: five made the long "CONTROL UNDETERMINED" labels collide in G4.
DEFAULT_MAX_COLS = 4


def grid_shape(n: int, max_cols: int = DEFAULT_MAX_COLS) -> tuple[int, int]:
    """(rows, cols) for ``n`` panels, wrapping at ``max_cols``."""
    cols = max(1, min(int(n), int(max_cols)))
    return int(np.ceil(int(n) / cols)), cols


def blank_unused(axes, n: int, rows: int, cols: int) -> None:
    """Hide the cells a partial last row does not fill.

    Without this an incomplete row shows as empty BOXES with ticks, which reads as "these panels
    failed" rather than "the cohort does not divide by four".
    """
    for j in range(int(n), int(rows) * int(cols)):
        axes[j // cols][j % cols].axis("off")


def at(axes, k: int, cols: int):
    """The k-th panel in a (rows, cols) grid, in reading order."""
    return axes[k // cols][k % cols]
