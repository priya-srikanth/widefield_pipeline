"""Camera-view orientation for lick PATH figures (Priya 11-03).

Rule: 2-D and 3-D path figures (x-y tongue / jaw paths, front and top views, closest-approach front views) are drawn
as the cam1 / cam4 face videos show them -- ipsilesional = image RIGHT, so contralesional (mouse right; every lesion in
PS92-95 is LEFT) = image LEFT. The SIGN convention stays + = contralesional in labels and numbers, so the lateral axis
is INVERTED: + = contra values sit on the left, and the axis ends are labelled "contra" (left) and "ipsi" (right).

Scalar change-vs-pre plots, angle traces and dot plots are NOT path views: they keep + = contra on the y axis and do
not use this module.

The 11-02 contra re-plots mirrored the paths instead (+ = contra drawn right, against the camera); those files are
superseded by the ``*_contra_cam`` renders (docs: DECISIONS.md 2026-11-03).
"""
from __future__ import annotations

#: one sentence for figure titles / notes
CAMERA_NOTE = ("camera view: contralesional (mouse right) on the image LEFT as in the face videos; lateral axis signed "
               "+ = contra and INVERTED, so + values read leftward")


def camera_x(ax, ends: bool = True, fontsize: float = 6.5, color: str = "0.25") -> None:
    """Invert ``ax``'s x axis (idempotent, safe on shared axes) so + = contra sits on the LEFT, and label the axis
    ends "contra" (left) / "ipsi" (right) just under the tick labels; the x label is pushed below them."""
    ax.xaxis.set_inverted(True)
    if not ends:
        return
    tick = max((t.get_size() for t in ax.get_xticklabels()), default=fontsize)
    dy = -(1.3 * tick + 9.5)                       # below tick length + pad + label
    for x, ha, word in ((0.0, "left", "contra"), (1.0, "right", "ipsi")):
        ax.annotate(word, xy=(x, 0.0), xycoords="axes fraction", xytext=(0.0, dy), textcoords="offset points",
                    ha=ha, va="top", fontsize=fontsize, color=color, fontweight="bold", annotation_clip=False)
    ax.xaxis.labelpad = max(ax.xaxis.labelpad, fontsize + 5.0)


def camera_x3d(ax, x_lo: float, x_hi: float, y: float, z: float, fontsize: float = 7.0, color: str = "0.25") -> None:
    """3-D axes whose x = ml (+ = mouse right = contra): write "contra" at the + end and "ipsi" at the - end of the ml
    axis (at depth ``y``, height ``z``). No inversion -- the oblique view is set so that the mouse's right already falls
    on the image left (a viewer in front of the mouse); this only names the ends."""
    ax.text(x_hi, y, z, "contra", fontsize=fontsize, color=color, fontweight="bold", ha="center", va="top")
    ax.text(x_lo, y, z, "ipsi", fontsize=fontsize, color=color, fontweight="bold", ha="center", va="top")
