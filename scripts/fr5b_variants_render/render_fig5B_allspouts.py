"""Fig 5B, all six spouts (CC-B, 2026-11-26): one row of three face maps per spout, in the live 5B style.

Rows in the canonical order (near ipsi, mid, contra, then far ipsi, mid, contra). Each map: per point, epoch minus own
pre (log2 speed ratio, mean of animals), -1..1 RdBu_r, the live 5B face crop, point size from the mesh spacing, rings =
permutation-cluster points (dark red where the point's value is > 0, dark blue otherwise). Under each map: the
number of animals in it; a cell with < 2 animals has no test and is drawn as 'not testable'. The map slots have the
live 5B slot width. Imports only numpy, pandas, matplotlib, PIL and vls_style.py.

Usage: render_fig5B_allspouts.py <kit_dir> <maps_csv> <out_stem> [--set NAME=VALUE ...]   (refuses to overwrite)
   kit_dir holds master.npz, master_face.png, vls_style.py; maps_csv = position, epoch, point, log2_ratio, ring, n_animals
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vls_style as S                                   # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib import colormaps                        # noqa: E402
from matplotlib.colors import Normalize                 # noqa: E402
from matplotlib.cm import ScalarMappable                # noqa: E402
from PIL import Image                                   # noqa: E402

# ----------------------------------------------------------------------------------------------- presentation settings
FS = 6.5
FS_LETTER = 8
FIG_W_IN = 4.35                             # = the live Fig 5 column width
LEFT, RIGHT = 0.13, 0.87
X0, X1, Y0, Y1 = 60, 610, 0, 415            # face crop (must equal the live 5B)
EPS = ['acute', 'subacute', 'chronic']
EP_TITLE = '{e} − pre'
TITLE = 'face movement speed, lick-aligned, angle-matched (post-cue licks)'
LETTER = ''
ROWS = [('close_L', 'near ipsi'), ('close_center', 'near mid'), ('close_R', 'near contra'),
        ('far_L', 'far ipsi'), ('far_center', 'far mid'), ('far_R', 'far contra')]
SHOW_N = True                               # 'n = k' (animals) under each map
CB_LABEL = 'speed vs pre\n(log$_2$ fold)'; CB_TICKS = [-1, 0, 1]
POINT_FRAC, RING_LW = 0.85, 0.2
RING_UP, RING_DOWN = '#67001f', '#053061'
ROW_GAP_IN = 0.16                           # vertical space between rows (in), holds the side / n labels
# -------------------------------------------------------------------------------------------------------------------


def apply_overrides(argv):
    import ast
    out, i = [], 0
    while i < len(argv):
        if argv[i] == '--set':
            k, v = argv[i + 1].split('=', 1)
            if k not in globals(): sys.exit(f'unknown setting {k}')
            try: v = ast.literal_eval(v)
            except (ValueError, SyntaxError): pass
            globals()[k] = v; i += 2
        else:
            out.append(argv[i]); i += 1
    return out


def main(kit, maps_csv, out):
    for ext in ('pdf', 'svg', 'png'):
        if os.path.exists(f'{out}.{ext}'): sys.exit(f'refusing to overwrite {out}.{ext}')
    S.apply()
    H = np.load(os.path.join(kit, 'master.npz'))['home']
    Image.open(os.path.join(kit, 'master_face.png')).verify()   # face image not drawn (as live 5B); kit check
    M = pd.read_csv(maps_csv)
    slot_w = (RIGHT - LEFT) * FIG_W_IN / 4 * (1 - 0.06 * 3 / 4)   # live 5B map slot width (in)
    map_h = slot_w * (Y1 - Y0) / (X1 - X0)
    top_in, bot_in = 0.42, 0.12
    FH = top_in + bot_in + len(ROWS) * map_h + (len(ROWS) - 1) * ROW_GAP_IN + ROW_GAP_IN
    fig = plt.figure(figsize=(FIG_W_IN, FH))
    gs = fig.add_gridspec(len(ROWS), 4, width_ratios=[1, 1, 1, 1], wspace=0.06, hspace=(ROW_GAP_IN / map_h),
                          left=LEFT, right=RIGHT, top=1 - top_in / FH, bottom=(bot_in + ROW_GAP_IN) / FH)
    nb = Normalize(-1, 1); cmb = colormaps['RdBu_r']
    H64 = H.astype(np.float64)
    _d = np.hypot(H64[:, None, 0] - H64[None, :, 0], H64[:, None, 1] - H64[None, :, 1]); np.fill_diagonal(_d, np.inf)
    _NN = float(np.median(_d.min(1)))
    first = last = None
    for r, (pos, name) in enumerate(ROWS):
        ax = fig.add_subplot(gs[r, 0]); ax.axis('off')
        ax.text(0.5, 0.5, name, transform=ax.transAxes, ha='center', va='center', fontsize=FS, weight='bold')
        for j, e in enumerate(EPS):
            ax = fig.add_subplot(gs[r, 1 + j])
            ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_aspect('equal'); ax.axis('off')
            ax.apply_aspect()
            if first is None: first = ax
            last = ax if j == 2 and r == 0 else last
            if r == 0: ax.set_title(EP_TITLE.format(e=e), fontsize=FS, pad=2)
            g = M[(M.position == pos) & (M.epoch == e)].sort_values('point')
            if not len(g):
                ax.text(0.5, 0.5, 'not testable\n(< 2 animals)', transform=ax.transAxes, ha='center', va='center',
                        fontsize=FS - 0.5, color='0.45')
                continue
            _ptpu = ax.get_position().width * fig.get_size_inches()[0] * 72 / (X1 - X0)
            _D = POINT_FRAC * _NN * _ptpu
            v = g.log2_ratio.to_numpy(); mk = g.ring.astype(bool).to_numpy()
            ax.scatter(H[:, 0], H[:, 1], c=v, cmap=cmb, norm=nb, s=_D ** 2, lw=0)
            for sel, col in ((mk & (v > 0), RING_UP), (mk & (v <= 0), RING_DOWN)):
                if sel.any():
                    ax.scatter(H[sel, 0], H[sel, 1], s=(_D - RING_LW) ** 2, facecolor='none', edgecolor=col,
                               lw=RING_LW)
            ax.axvline(np.median(H[:, 0]), color='0.6', lw=0.4)
            if r == len(ROWS) - 1:
                ax.text(0.27, -0.02, 'contra', transform=ax.transAxes, ha='center', va='top', fontsize=FS - 0.5,
                        color='0.35')
                ax.text(0.73, -0.02, 'ipsi', transform=ax.transAxes, ha='center', va='top', fontsize=FS - 0.5,
                        color='0.35')
            if SHOW_N:
                ax.text(0.98, 0.02, f'n = {int(g.n_animals.iloc[0])}', transform=ax.transAxes, ha='right',
                        va='bottom', fontsize=FS - 1, color='0.35')
    fig.text((LEFT + RIGHT) / 2 + 0.06, 1 - 0.12 / FH, TITLE, ha='center', va='top', fontsize=FS)
    if LETTER:
        first.text(-0.08, 1.12, LETTER, transform=first.transAxes, fontsize=FS_LETTER, weight='bold')
    p = last.get_position()
    cax = fig.add_axes([p.x1 + 0.015, p.y0 + 0.1 * p.height, 0.012, 0.8 * p.height])
    cb = fig.colorbar(ScalarMappable(nb, cmb), cax=cax, ticks=CB_TICKS); cb.ax.tick_params(labelsize=FS, length=2, width=0.4)
    cb.set_label(CB_LABEL, fontsize=FS, labelpad=1); cb.outline.set_linewidth(0.4)
    S.save(fig, out)
    print('wrote', out)


if __name__ == '__main__':
    a = apply_overrides(sys.argv[1:])
    main(a[0], a[1], a[2])
