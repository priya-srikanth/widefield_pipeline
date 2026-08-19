"""Shared Allen atlas overlay helpers.

region_edges() draws the region/brain outline from a label atlas. It marks BOTH
pixels of each label transition before masking to the brain, so the brain's LEFT
and ANTERIOR (top) outer borders are kept.

The earlier per-module version marked only the upper pixel (vertical) / left pixel
(horizontal) of each transition pair and then did ``edges &= valid``. At the
brain's left/top outer boundary the marked pixel falls on the background side and
was deleted, so those outlines were dropped (the open left-anterior / olfactory-
bulb border). Marking both sides fixes it: the labeled-side pixel always survives
the mask, regardless of which way the boundary faces.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def region_edges(atlas: np.ndarray) -> np.ndarray:
    valid = np.isfinite(atlas) & (atlas != 0)
    dv = atlas[:-1, :] != atlas[1:, :]
    dh = atlas[:, :-1] != atlas[:, 1:]
    edges = np.zeros_like(valid, dtype=bool)
    edges[:-1, :] |= dv
    edges[1:, :] |= dv   # also mark the lower row of each vertical pair
    edges[:, :-1] |= dh
    edges[:, 1:] |= dh   # also mark the right col of each horizontal pair
    edges &= valid       # keep edges on the labeled (brain) side
    return ndimage.binary_dilation(edges, iterations=1)


def overlay_regions(ax, edges: np.ndarray) -> None:
    """Draw Allen region boundaries as a translucent black layer over an existing image.

    Was copied identically into four map modules. `plot_lick_vs_cue_spout_maps` keeps its own
    variant deliberately -- it draws at a different alpha for a denser panel -- and that difference
    is now visible as a difference rather than hidden among four copies of the same thing.
    """
    overlay = np.zeros((*edges.shape, 4), dtype=np.float32)
    overlay[edges] = (0, 0, 0, 0.65)
    ax.imshow(overlay, interpolation="nearest")
