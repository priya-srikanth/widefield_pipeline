"""Fig 5B variant panel (CC-B, 2026-11-26): one row of three face maps in the live 5B style.

Panel B of render_fig5AD.py (fig5AD_facerhythm_col kit, 2026-11-25) drawn alone: same figure width, same map slot
size (the A+B row's four equal slots; the first slot carries the version label instead of the point mesh), same color
scale (-1..1 log2, RdBu_r), same face crop X0..Y1, same point size (from the mesh spacing) and ring rule (dark red
where the point's value is > 0, dark blue otherwise; ring = permutation-cluster point).
Imports only numpy, pandas, matplotlib, PIL and vls_style.py.

Usage: render_fig5B_variant.py <kit_dir> <maps_csv> <out_stem> [--set NAME=VALUE ...]   (refuses to overwrite)
   kit_dir holds master.npz, master_face.png, vls_style.py; maps_csv = epoch, point, log2_ratio, ring
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
ROW_H = 0.72                                # = the live A+B row height ratio
LEFT, RIGHT = 0.13, 0.87
X0, X1, Y0, Y1 = 60, 610, 0, 415            # face crop (must equal the live 5B)
EPS = ['acute', 'subacute', 'chronic']
EP_TITLE = '{e} − pre'
B_TITLE = 'face movement speed, far contralesional spout, lick-aligned'
VERSION_LABEL = 'successful\nlicks only'    # text in the first slot
LETTER = 'B'
B_CB_LABEL = 'speed vs pre\n(log$_2$ fold)'; B_CB_TICKS = [-1, 0, 1]
B_POINT_FRAC, B_RING_LW = 0.85, 0.2
RING_UP, RING_DOWN = '#67001f', '#053061'
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
    Image.open(os.path.join(kit, 'master_face.png')).verify()   # face image not drawn in B (as live); kept as a kit check
    MB = pd.read_csv(maps_csv)
    MAPS = {e: g.sort_values('point').log2_ratio.to_numpy() for e, g in MB.groupby('epoch')}
    MASK = {e: g.sort_values('point').ring.astype(bool).to_numpy() for e, g in MB.groupby('epoch')}
    _UNIT = (0.955 - 0.04) * 5.7 / 6.07                    # inches per height-ratio unit in the live column
    top_in, bot_in = 0.045 * 5.7, 0.16
    FH = top_in + bot_in + _UNIT * ROW_H
    fig = plt.figure(figsize=(FIG_W_IN, FH))
    g1 = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.06, left=LEFT, right=RIGHT, top=1 - top_in / FH,
                          bottom=bot_in / FH)
    ax = fig.add_subplot(g1[0]); ax.axis('off')
    ax.text(0.5, 0.5, VERSION_LABEL, transform=ax.transAxes, ha='center', va='center', fontsize=FS, weight='bold')
    nb = Normalize(-1, 1); cmb = colormaps['RdBu_r']
    H64 = H.astype(np.float64)
    _d = np.hypot(H64[:, None, 0] - H64[None, :, 0], H64[:, None, 1] - H64[None, :, 1]); np.fill_diagonal(_d, np.inf)
    _NN = float(np.median(_d.min(1)))
    for j, e in enumerate(EPS):
        ax = fig.add_subplot(g1[1 + j])
        ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_aspect('equal'); ax.axis('off')
        ax.apply_aspect()
        _ptpu = ax.get_position().width * fig.get_size_inches()[0] * 72 / (X1 - X0)
        _D = B_POINT_FRAC * _NN * _ptpu
        if e not in MAPS:
            ax.text(0.5, 0.5, 'n/a', transform=ax.transAxes, ha='center', va='center', fontsize=FS)
            continue
        v = MAPS[e]; mk = MASK[e]
        ax.scatter(H[:, 0], H[:, 1], c=v, cmap=cmb, norm=nb, s=_D ** 2, lw=0)
        for sel, col in ((mk & (np.asarray(v) > 0), RING_UP), (mk & (np.asarray(v) <= 0), RING_DOWN)):
            if sel.any():
                ax.scatter(H[sel, 0], H[sel, 1], s=(_D - B_RING_LW) ** 2, facecolor='none', edgecolor=col, lw=B_RING_LW)
        ax.text(0.27, -0.02, 'contra', transform=ax.transAxes, ha='center', va='top', fontsize=FS - 0.5, color='0.35')
        ax.text(0.73, -0.02, 'ipsi', transform=ax.transAxes, ha='center', va='top', fontsize=FS - 0.5, color='0.35')
        ax.axvline(np.median(H[:, 0]), color='0.6', lw=0.4)
        ax.set_title(EP_TITLE.format(e=e), fontsize=FS, pad=2)
        if j == 0:
            ax.text(-0.08, 1.12, LETTER, transform=ax.transAxes, fontsize=FS_LETTER, weight='bold')
            ax.text(1.55, 1.3, B_TITLE, transform=ax.transAxes, ha='center', fontsize=FS)
    p = fig.axes[-1].get_position()
    cax = fig.add_axes([p.x1 + 0.015, p.y0 + 0.1 * p.height, 0.012, 0.8 * p.height])
    cb = fig.colorbar(ScalarMappable(nb, cmb), cax=cax, ticks=B_CB_TICKS); cb.ax.tick_params(labelsize=FS, length=2, width=0.4)
    cb.set_label(B_CB_LABEL, fontsize=FS, labelpad=1); cb.outline.set_linewidth(0.4)
    S.save(fig, out)
    print('wrote', out)


if __name__ == '__main__':
    a = apply_overrides(sys.argv[1:])
    main(a[0], a[1], a[2])
