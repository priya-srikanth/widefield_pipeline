"""Figure 3P: 3-D tongue (solid) and jaw (dashed) paths, FRONT view, 2 x 3 (near top, far bottom; ipsi, mid,
contra columns), EQUAL aspect so the mouth (+) is not distorted. Same pooling as kinematics_pooled_figs.fig_paths3d
(pose3d_paths_pooled.csv). Markers on the tongue paths: bins where that epoch differs from pre (cluster permutation
per coordinate on session changes from each animal's own August pre, labels permuted within animal, p < .05/3):
open circle = side only, open triangle = down only, filled = both (paths3d_front_sig.csv)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': ['Arial', 'Liberation Sans'], 'font.size': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
R = pd.read_csv('pose3d_paths_pooled.csv'); R = R[R.n_animals >= 3]
SIG = pd.read_csv('paths3d_front_sig.csv')
COL = {'pre': '#111111', 'acute': '#e0218a', 'subacute': '#d4a017', 'chronic': '#1b7f79'}
EP = ['pre', 'acute', 'subacute', 'chronic']
GRID = [['close_L', 'close_center', 'close_R'], ['far_L', 'far_center', 'far_R']]
def path(p, m, e):
    return R[(R.position == p) & (R.measure == m) & (R.epoch == e)].set_index('x').v
# shared limits: x = side in CONTRA sign (+ = contra), drawn camera-style (contra on the left)
xs = -R[R.measure.isin(['side', 'j_side'])].v; ys = -R[R.measure.isin(['down', 'j_down'])].v
x0, x1 = min(xs.min(), 0) - 0.12, max(xs.max(), 0) + 0.12
y0, y1 = min(ys.min(), 0) - 0.15, 0.35
W_CELL = 0.92                                    # inches per cell; height follows from equal aspect
H_CELL = 1.05                                   # shorter cells (not equal aspect)
GX, GY, L, B, T = 0.14, 0.32, 0.52, 0.50, 0.22
LEG = 0.82
FW = L + 3 * W_CELL + 2 * GX + LEG; FH = B + 2 * H_CELL + GY + T
fig = plt.figure(figsize=(FW, FH))
for r, row in enumerate(GRID):
    for c, p in enumerate(row):
        ax = fig.add_axes([(L + c * (W_CELL + GX)) / FW, (B + (1 - r) * (H_CELL + GY)) / FH, W_CELL / FW, H_CELL / FH])
        for part, ls, lw in (('', '-', 1.3), ('j_', '--', 0.9)):
            for e in EP:
                A = path(p, part + 'side', e); Bd = path(p, part + 'down', e)
                ix = A.index.intersection(Bd.index)
                if not len(ix): continue
                xa, yb = -A[ix].to_numpy(), -Bd[ix].to_numpy()
                ax.plot(xa, yb, ls, color=COL[e], lw=lw, zorder=2)
                if part == '' and e != 'pre':
                    s = SIG[(SIG.position == p) & (SIG.epoch == e)]
                    ma = s[s.measure == 'side'].set_index('x').sig.reindex(ix, fill_value=False).to_numpy()
                    mb = s[s.measure == 'down'].set_index('x').sig.reindex(ix, fill_value=False).to_numpy()
                    for sel, mk, fc in ((ma & ~mb, 'o', 'none'), (mb & ~ma, '^', 'none'), (ma & mb, 'o', COL[e])):
                        if sel.any():
                            ax.scatter(xa[sel], yb[sel], marker=mk, s=4, facecolors=fc, edgecolors=COL[e],
                                       linewidths=0.4, zorder=4)
        ax.plot(0, 0, '+', color='k', ms=5, mew=0.8, zorder=5)
        ax.set_xlim(x1, x0)                           # camera view: contra (+) on the LEFT
        ax.set_ylim(y0, y1)
        ax.set_xticks([1, 0, -1]); ax.set_xticklabels(['1', '0', '−1'])
        ax.set_yticks([0, -2, -4]); ax.set_yticklabels(['0', '−2', '−4'] if c == 0 else [])
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        if r == 0: ax.set_title(['ipsi', 'mid', 'contra'][c], fontsize=7.5, fontweight='bold', pad=3)
        if c == 0:
            ax.set_ylabel('down (mm)', fontsize=7.5, labelpad=1)
            ax.text(-0.52, 0.5, ['near', 'far'][r], transform=ax.transAxes, rotation=90, ha='center', va='center',
                    fontsize=7.5, fontweight='bold')
        if r == 1:
            ax.text(0.0, -0.20, 'contra', transform=ax.transAxes, fontsize=6.5, ha='left', va='top')
            ax.text(1.0, -0.20, 'ipsi', transform=ax.transAxes, fontsize=6.5, ha='right', va='top')
fig.text((L + 1.5 * W_CELL + GX) / FW, 0.02 / FH, 'side (mm; + = contralesional)', ha='center', va='bottom', fontsize=7.5)
h = [plt.Line2D([], [], color=COL[e], lw=1.3, label=e) for e in EP]
h += [plt.Line2D([], [], color='0.3', lw=1.3, ls='-', label='tongue'), plt.Line2D([], [], color='0.3', lw=0.9, ls='--', label='jaw'),
      plt.Line2D([], [], marker='o', ls='none', ms=3, mfc='0.3', mec='0.3', label='differs\nfrom pre')]
fig.legend(handles=h, loc='upper left', bbox_to_anchor=((L + 3 * W_CELL + 2 * GX + 0.05) / FW, 1 - T / FH), frameon=False,
           fontsize=6.8, handlelength=1.6, labelspacing=0.3)
fig.savefig('Fig3O_paths_short.svg'); fig.savefig('Fig3O_paths_short.pdf'); fig.savefig('Fig3O_paths_short.png', dpi=200); fig.savefig('Fig3O_paths_short_300.png', dpi=300)
print(FW, FH, H_CELL, (x0, x1, y0, y1))
