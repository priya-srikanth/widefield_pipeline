"""Figure 4 panel a: state-space change maps, full and cortex-wide-mode-removed,
with the NEW kinematics + contact + face control (frKBs).

Source (server):
  state_space_recovery/page_figs/ss_cmap_full_rest_frKBs_lick_svd.png
  state_space_recovery/page_figs/ss_cmap_nocw_rest_frKBs_lick_svd.png
Alpha flattened onto white, then the brain cells cropped out and recomposed here
with our own titles so the two blocks sit side by side with their own colour
bars (the two versions are on different scales and must not share one).
Columns kept: 'mean over spouts' (col 1) and 'far contra' (col 7) only.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from grant_panels import save, set_scale

# pre and chronic only: acute and subacute are dropped for space (the acute
# detour is carried by the trajectory panels instead).
ROWS = [('pre response', (212, 383)), ('chronic \u2212 pre', (976, 1150))]
BLOCKS = [
    ('full change',               'ss/ss_cmap_full_rest_frKBs_lick_svd_w.png',
     (43, 206), (1325, 1488), (1511, 1606)),
    ('cortex-wide mode removed',  'ss/ss_cmap_nocw_rest_frKBs_lick_svd_w.png',
     (57, 220), (1339, 1502), (1525, 1636)),
]
PAD = 8

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
#: (column, top row, bottom row, vmin, vmax, ticks, label) per block, per row
_BARS = {
    (0, 0): (1517, 191, 402, 0.0, 2.30, [0, 1, 2], 'pre response'),
    (1, 0): (1531, 191, 402, -0.60, 0.90, [-0.5, 0, 0.5], 'pre response'),
    (0, 1): (1523, 584, 1031, -0.8, 0.8, [-0.8, 0, 0.8], 'change vs pre'),
    (1, 1): (1537, 584, 1031, -0.6, 0.6, [-0.6, 0, 0.6], 'change vs pre'),
}
FS = 6.8
def _bar(fig, im, bi, ri, x, y, w, h):
    col, r0, r1, lo, hi, ticks, lab = _BARS[(bi, ri)]
    rgb = im[r0:r1 + 1, col - 1:col + 2].mean(1)[::-1] / 255.0      # bottom -> top
    cax = fig.add_axes([x, y, w, h])
    cb = ColorbarBase(cax, cmap=ListedColormap(rgb), norm=Normalize(lo, hi),
                      orientation='vertical', ticks=ticks)
    cb.outline.set_linewidth(0.5)
    cax.tick_params(labelsize=FS, length=2, width=0.5, pad=1.5)
    cax.set_yticklabels([('%g' % t).replace('-', '\u2212') if t <= 0 else '+%g' % t
                         if ri == 1 else '%g' % t for t in ticks])
    cb.set_label(lab, fontsize=FS, labelpad=2)

ims = {p: np.asarray(Image.open(p).convert('RGB')) for _, p, _, _, _ in BLOCKS}

set_scale('ss_full_nocw_frKBs', 6.6)
fig = plt.figure(figsize=(6.6, 2.0))
# geometry in figure coords
LEFT = 0.072
CELL_W, CELL_H = 0.192, 0.345
BAR_W = 0.046
GAP_IN, GAP_BLOCK = 0.002, 0.016
TOP = 0.760

def cell(x, y, w, h):
    ax = fig.add_axes([x, y, w, h]); ax.set_axis_off(); return ax

bx = LEFT
for bi, (title, path, cmean, cfar, cbar) in enumerate(BLOCKS):
    im = ims[path]
    xs = [bx, bx + CELL_W + GAP_IN]
    for ri, (rlab, (y0, y1)) in enumerate(ROWS):
        yy = TOP - ri * (CELL_H + 0.030) - CELL_H
        for ci, (cx0, cx1) in enumerate([cmean, cfar]):
            ax = cell(xs[ci], yy, CELL_W, CELL_H)
            ax.imshow(im[y0:y1, cx0-PAD:cx1+PAD])
        # colour bars, drawn natively for both rows so the tick numbers and
        # labels are one type size. Colours are sampled from the source bars
        # (centre column, top to bottom); ranges read off the source ticks.
        _bar(fig, im, bi, ri, xs[1] + CELL_W + 0.006, yy + 0.02, BAR_W*0.30, CELL_H - 0.04)
        if bi == 0:
            fig.text(LEFT - 0.012, yy + CELL_H/2, rlab, rotation=90,
                     ha='right', va='center', fontsize=7.4)
    # I / C under the bottom row of each column: image left = ipsilesional
    y_bot = TOP - (len(ROWS)-1) * (CELL_H + 0.030) - CELL_H
    for ci in range(2):
        for frac, lab in ((0.28, 'I'), (0.72, 'C')):
            fig.text(xs[ci] + frac*CELL_W, y_bot - 0.016, lab, ha='center',
                     va='top', fontsize=7.0, color='0.55')
    for ci, lab in enumerate(['mean over spouts', 'far contra']):
        fig.text(xs[ci] + CELL_W/2, TOP + 0.012, lab, ha='center', va='bottom',
                 fontsize=7.4)
    fig.text(bx + CELL_W + GAP_IN/2, TOP + 0.072, title, ha='center',
             va='bottom', fontsize=8.2, fontweight='bold')
    bx = xs[1] + CELL_W + 0.004 + BAR_W + GAP_BLOCK

save(fig, 'ss_full_nocw_frKBs')
