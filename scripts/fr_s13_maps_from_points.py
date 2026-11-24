"""Supplementary Figure S13 (face movement speed maps, six-position cohort) drawn as VECTOR from a per-point permutation
table (`face_rhythm_aligned_maps.py --perm-points`: align, position, epoch, point, x, y, mean_log2, sig_cluster), in
the manuscript style (vls_style), instead of cropping the pipeline PNGs (`manuscript_code/figures/fr_aligned_maps_panel.py`,
whose layout this keeps: A cue-aligned 0 to +2 s, B lick-aligned -0.1 to +0.2 s; rows near ipsi, near mid, near contra,
far ipsi, far mid, far contra; columns acute / subacute / chronic minus pre; one shared colorbar, log2 change, -1..1,
RdBu_r). Points and rings as Fig 5B (`fr_fig5_col_v26b.py`): rings = the point's own outline, dark red up / dark blue
down. Image left = contralesional; vertical line = facial midline (master LP-nose midline).

    python scripts/fr_s13_maps_from_points.py <perm_points.csv> <out_stem> [--style-dir <manuscript_code/figures>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

STYLE_DIR = ("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/manuscript_code/figures")
ROWS = [("close_L", "near ipsi"), ("close_center", "near mid"), ("close_R", "near contra"),
        ("far_L", "far ipsi"), ("far_center", "far mid"), ("far_R", "far contra")]
EPS = ["acute", "subacute", "chronic"]
COLS = ["acute \u2212 pre", "subacute \u2212 pre", "chronic \u2212 pre"]
TITLES = (("cue", "cue-aligned (0 to +2 s)"), ("lick", "lick-aligned (\u22120.1 to +0.2 s)"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("points", type=Path)
    ap.add_argument("out")
    ap.add_argument("--style-dir", default=STYLE_DIR)
    ap.add_argument("--midline", type=float, default=None, help="midline x (default: the pipeline's MASTER_MIDLINE_X)")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.style_dir)
    import matplotlib.pyplot as plt
    import vls_style as S
    from matplotlib import cm, colors
    S.apply()
    if a.midline is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from wfield_local import face_rhythm_readouts as frr
        mid = float(frr.MASTER_MIDLINE_X)
    else:
        mid = a.midline
    for ext in ("pdf", "svg", "png"):
        if Path(f"{a.out}.{ext}").exists():
            raise SystemExit(f"refusing to overwrite {a.out}.{ext}")
    P = pd.read_csv(a.points)
    nb, cmb = colors.Normalize(-1, 1), "RdBu_r"
    fig = plt.figure(figsize=(7.0, 5.2))
    gs = fig.add_gridspec(6, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1], left=0.09, right=0.88, top=0.885, bottom=0.01,
                          wspace=0.03, hspace=0.06)
    g0 = P.iloc[:0]
    for b, (al, title) in enumerate(TITLES):
        for i, (pos, lab) in enumerate(ROWS):
            for c, ep in enumerate(EPS):
                ax = fig.add_subplot(gs[i, c + 4 * b])
                g = P[(P["align"] == al) & (P.position == pos) & (P.epoch == ep)].sort_values("point")
                if g.empty:
                    ax.axis("off")
                    continue
                g0 = g
                x, y, v = g.x.to_numpy(), g.y.to_numpy(), g.mean_log2.to_numpy()
                mk = g.sig_cluster.astype(bool).to_numpy()
                ax.scatter(x, y, c=v, cmap=cmb, norm=nb, s=1.6, lw=0)
                for sel, col in ((mk & (v > 0), "#67001f"), (mk & (v <= 0), "#053061")):
                    if sel.any():
                        ax.scatter(x[sel], y[sel], s=1.5, facecolor="none", edgecolor=col, lw=0.14)
                ax.axvline(mid, color="0.6", lw=0.4)
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.axis("off")
                if i == 0:
                    ax.set_title(COLS[c], fontsize=7.5, pad=3)
                if c == 0 and b == 0:
                    ax.text(-0.08, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=7.5)
        x0 = gs[0, 4 * b].get_position(fig).x0
        x1 = gs[0, 4 * b + 2].get_position(fig).x1
        fig.text((x0 + x1) / 2, 0.965, title, ha="center", va="center", fontsize=8.5)
        fig.text(x0 - 0.005, 0.965, "AB"[b], ha="right", va="center", fontsize=10, fontweight="bold")
    # common limits so every tile has the same scale
    xl = (g0.x.min() - 8, g0.x.max() + 8)
    yl = (g0.y.max() + 8, g0.y.min() - 8)
    for ax in fig.axes:
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
    cax = fig.add_axes([0.895, 0.35, 0.012, 0.3])
    fig.colorbar(cm.ScalarMappable(nb, cmb), cax=cax, ticks=[-1, 0, 1]).set_label("log$_2$ change from pre",
                                                                                 fontsize=7.5)
    S.save(fig, a.out)
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
