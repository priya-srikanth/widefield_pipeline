"""Supplementary Figure S13 (face movement speed maps, six-position cohort) drawn as VECTOR from a per-point permutation
table (`face_rhythm_aligned_maps.py --perm-points`: align, position, epoch, point, x, y, mean_log2, sig_cluster), in
the manuscript style (vls_style), instead of cropping the pipeline PNGs (`manuscript_code/figures/fr_aligned_maps_panel.py`,
whose layout this keeps: A cue-aligned 0 to +2 s, B lick-aligned -0.1 to +0.2 s; rows near ipsi, near mid, near contra,
far ipsi, far mid, far contra; columns acute / subacute / chronic minus pre; one shared colorbar, log2 change, -1..1,
RdBu_r). Points and rings as Fig 5B (`fr_fig5_col_v26b.py`): rings = the point's own outline, dark red up / dark blue
down. Image left = contralesional; vertical line = facial midline (master LP-nose midline).

    python scripts/fr_s13_maps_from_points.py <perm_points.csv> <out_stem> [--style-dir <manuscript_code/figures>]
        [--width 6.5 --height 4.85] [--marker-frac 0.85] [--supp-fonts] [--vertical-rows --left-in 0.30]

--width / --height set the page (default 7.0 x 5.2 in, the 11-23 layout). Margins, the colorbar and the title band are
fixed in inches, so a narrower page (6.5 in = the 468 pt text block, included at 100%) keeps every font size.
--marker-frac F sizes each point from the mesh: diameter = F x the median nearest-neighbor spacing at the rendered scale
(the ring is drawn on the point's own outline, so its outer edge = the point's edge and neighbors never touch). Default
(no flag) = the 11-23 fixed sizes (s 1.6 / ring 1.5, lw 0.14). --supp-fonts = supplement convention (Priya 11-23):
6.5 pt text, 7 pt bold headers, 9 pt bold panel letters (default = the 11-23 sizes 7.5 / 8.5 / 10).
--vertical-rows = row labels rotated 90 deg (bottom to top), centered on each row at the left, wrapped onto 2 lines
if they would not fit the row height (Priya 11-23); pair with a smaller --left-in so the maps get the width.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
    ap.add_argument("--width", type=float, default=7.0, help="page width, in (6.5 = manuscript text block)")
    ap.add_argument("--height", type=float, default=5.2, help="page height, in")
    ap.add_argument("--marker-frac", type=float, default=None, help="point diameter / median NN spacing")
    ap.add_argument("--supp-fonts", action="store_true", help="6.5 / 7 bold / 9 bold pt supplement fonts")
    ap.add_argument("--vertical-rows", action="store_true", help="rotate the row labels 90 deg")
    ap.add_argument("--left-in", type=float, default=0.63, help="left margin, in (0.63 = the 11-23 layout)")
    ap.add_argument("--cbar-label", default="log$_2$ change from pre", help="colorbar label (a literal \\n gives 2 lines)")
    ap.add_argument("--ring-lw", type=float, default=None, help="ring line width, pt (default: 13%% of the point)")
    ap.add_argument("--ring-color", default=None, help="one neutral ring color (default: dark red up / dark blue down)")
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
    W, H = a.width, a.height
    # inch margins = the 7.0 x 5.2 in layout (left 0.63, right 0.84, top 0.598, bottom 0.052 in)
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(6, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1], left=a.left_in / W, right=1 - 0.84 / W,
                          top=1 - 0.598 / H, bottom=0.052 / H,
                          wspace=0.03, hspace=0.06)
    FS, FT, FL, TW = (6.5, 7.0, 9.0, "bold") if a.supp_fonts else (7.5, 8.5, 10.0, "normal")
    g0 = P.iloc[:0]
    tiles = []
    for b, (al, title) in enumerate(TITLES):
        for i, (pos, lab) in enumerate(ROWS):
            for c, ep in enumerate(EPS):
                ax = fig.add_subplot(gs[i, c + 4 * b])
                g = P[(P["align"] == al) & (P.position == pos) & (P.epoch == ep)].sort_values("point")
                if g.empty:
                    ax.axis("off")
                    continue
                g0 = g
                tiles.append((ax, g))
                ax.axvline(mid, color="0.6", lw=0.4)
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.axis("off")
                if i == 0:
                    ax.set_title(COLS[c], fontsize=FS, pad=3)
                if c == 0 and b == 0 and not a.vertical_rows:
                    ax.text(-0.08, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=FS)
        x0 = gs[0, 4 * b].get_position(fig).x0
        x1 = gs[0, 4 * b + 2].get_position(fig).x1
        fig.text((x0 + x1) / 2, 1 - 0.182 / H, title, ha="center", va="center", fontsize=FT,
                 fontweight=TW)
        fig.text(x0 - 0.035 / W, 1 - 0.182 / H, "AB"[b], ha="right", va="center", fontsize=FL, fontweight="bold")
    # common limits so every tile has the same scale
    xl = (g0.x.min() - 8, g0.x.max() + 8)
    yl = (g0.y.max() + 8, g0.y.min() - 8)
    for ax in fig.axes:
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
    if a.vertical_rows:                      # one label per row, centered on the row's drawn tile, left of the maps
        fig.canvas.draw()
        ren = fig.canvas.get_renderer()
        for i, (_, lab) in enumerate(ROWS):
            cell = fig.axes[3 * i].get_position()           # the drawn (aspect-equal) tile of column 1
            t = fig.text(cell.x0 - 0.04 / W, (cell.y0 + cell.y1) / 2, lab, rotation=90, ha="right", va="center",
                         fontsize=FS, multialignment="center")
            if t.get_window_extent(ren).height > 0.95 * cell.height * H * fig.dpi:
                t.set_text(lab.replace(" ", "\n", 1))       # wrap onto 2 lines when it does not fit the row
    if a.marker_frac is None:
        d_pt, s_pt, s_ring, lw_ring = None, 1.6, 1.5, 0.14
    else:
        from scipy.spatial import cKDTree
        xy = g0[["x", "y"]].to_numpy(float)
        nn = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))       # data units (px of the mesh)
        ax0 = tiles[0][0]
        ax0.apply_aspect()
        pt_per_unit = ax0.get_position().width * W * 72 / (xl[1] - xl[0])
        d_pt = a.marker_frac * nn * pt_per_unit
        lw_ring = a.ring_lw if a.ring_lw is not None else max(0.08, 0.13 * d_pt)
        s_pt, s_ring = d_pt ** 2, (d_pt - lw_ring) ** 2                  # ring outer edge = point edge
        print(f"median NN {nn:.2f} units = {nn * pt_per_unit:.3f} pt; point {d_pt:.3f} pt, ring lw {lw_ring:.3f} pt")
    for ax, g in tiles:
        x, y, v = g.x.to_numpy(), g.y.to_numpy(), g.mean_log2.to_numpy()
        mk = g.sig_cluster.astype(bool).to_numpy()
        ax.scatter(x, y, c=v, cmap=cmb, norm=nb, s=s_pt, lw=0)
        rings = (((mk, a.ring_color),) if a.ring_color else
                 ((mk & (v > 0), "#67001f"), (mk & (v <= 0), "#053061")))
        for sel, col in rings:
            if sel.any():
                ax.scatter(x[sel], y[sel], s=s_ring, facecolor="none", edgecolor=col, lw=lw_ring)
    cax = fig.add_axes([1 - 0.735 / W, 0.5 - 0.78 / H, 0.084 / W, 1.56 / H])
    cb = fig.colorbar(cm.ScalarMappable(nb, cmb), cax=cax, ticks=[-1, 0, 1])
    cb.set_label(a.cbar_label.replace("\\n", "\n"), fontsize=FS)
    if a.supp_fonts:
        cb.ax.tick_params(labelsize=FS)
    S.save(fig, a.out)
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
