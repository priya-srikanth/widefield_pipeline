"""ptsv2 (CC-B 2026-11-23; copy of manuscript_code/figures/fr_fig5_col_v26b.py, which is unchanged): ONLY panel B's
point styling changes, as S13 v2 (scripts/fr_s13_maps_from_points.py --marker-frac 0.85 --ring-lw 0.2), Priya 11-23:
  * point diameter = 0.85 x the median nearest-neighbor mesh spacing at the rendered scale (was a fixed s = 1.6, wider
    than the spacing, so neighbors overlapped);
  * significance rings on the point's own outline (outer edge = point edge, so rings never reach a neighbor), 0.2 pt,
    dark red up / dark blue down as before (a 0.3-0.4 pt ring would cover ~85% of a 1 pt point and hide its fill);
  * fill unchanged: RdBu_r, log2 speed ratio vs pre, one symmetric -1..1 scale for all three epochs;
  * B colorbar ticks and label at the 6.5 pt text size (were 5.5 / 5.0 pt), label with units.
Canvas, A, C, D and every input unchanged. Same usage and inputs as v26b.

v26b docstring follows. v26b (Priya 2026-11-22; copy of fr_fig5_col_v26.py, which is unchanged): amplitude palette ColorBrewer BrBG, ipsi
brown (near #DFC27D, far #8C510A), mid grey unchanged, contra teal (near #80CDC1, far #01665E): v26's purple / green sat
next to the acute (#AA3377) / chronic (#228833) epoch colours of E-G. Everything else as v26, including the direction
maps drawn on the full-contrast face (the current fr_fig5_col.py rendering, v42+). Same inputs and usage.
"""
"""v26 (Priya 2026-11-22; copy of fr_fig5_col.py, which is unchanged): panel C columns closer together (absolute
column widths: maps 0.636 in as before, gaps 0.10 / 0.22 / 0.32 in, spectrum 0.44 in, amplitude 0.64 in); spectrum
lines black; amplitude palette purple (ipsi) / grey (mid) / green (contra), near lighter, far darker (no blue / red, which
mean decreased / increased movement in B and D); position legend = one column right of the amplitude column (inside
the figure's existing width, so the right column E-G does not move); the C-to-D gap that held the old legend is closed
(D moves up, figure height shrinks by that amount; A, B, C, D sizes unchanged); D colorbar tick labels at FS-1 as B
(the tick_params call had ended up inside a comment). Same inputs and usage as fr_fig5_col.py. Run in the
stroke_orofacial env (aligned_maps_cache_point.pkl needs pandas >= 3).

Figure 5 A-D, one-column layout for the v41 Figure 5 (Priya 11-19: A+B together, C below, D below; E-G in a
second column). Same data and drawing as fr_fig4_top.py except: larger C maps with the point mesh visible over the face,
brighter A mesh, spectrum peak label on the side away from the peak.

Original docstring (fr_fig4_top.py): Figure 4 A-D redrawn from source in the manuscript style (replaces the crop of the K08 Illustrator figure).

A  FaceRhythm point mesh on the mean face (master_face.png, master.npz 'home').
B  Face movement speed per point, lick-aligned, far contralesional spout, each epoch minus pre-stroke (log2 ratio, mean of
   animals); black rings = permutation-cluster points (aligned_maps_cache_point.pkl, the pipeline's aligned_maps readout).
C  Four of the twelve frozen pre-stroke factors (lick contra = lick_right, lick ipsi = lick_left, whisking = whisk_run,
   pre-lick jaw = prep_jaw), one row each: spatial weight | direction of motion | spectrum | cue-aligned amplitude by
   spout position (pre-stroke sessions, mean +- SEM across sessions; fr_pre_timecourses_abs.npz).
   Direction colours as fr_factors_lib.report.direction_colours: hue = 2/3 x (1 - angle/90 deg) (blue lateral, green
   oblique, red vertical), opacity = weight^0.7.
D  Change in each lick factor's spatial loading per point, epoch minus pre, in units of the animal's own pre-stroke session
   SD; rings = permutation clusters p < .05 (dark red positive, dark blue negative). Values recovered from the pipeline's
   session_shape_v2 readout figure (analysis/fr_lick_loading_from_svg.py).
Image left = mouse right = contralesional (left VLS lesions).

Usage: fr_fig4_top.py <master.npz> <master_face.png> <aligned_maps_cache_point.pkl> <fr_pre_timecourses_abs.npz>
                      <fr_lick_loading_change.csv> <out_stem> [<aligned_maps_perm_points.csv>]
v43 rebuild (manifest R8, 2026-11-20): with the optional 7th argument (the pipeline's per-point permutation table,
aligned_maps/data/aligned_maps_perm_points.csv), the B rings are the PERMUTATION-CLUSTER points (sig_cluster, align
'lick', position 'far_R'), matching the legend and S14; without it, the older cached marks (per-point bootstrap +
animals agree) as before. Ring colour unchanged.
"""
import sys, pickle, colorsys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
import vls_style as S
S.apply()

mfile, face, pkl, tcf, ldf, out = sys.argv[1:7]
Z = np.load(mfile); H = Z['home']; SP = Z['spatial']; SPEC = Z['spectral']; FQ = Z['freqs']
FACE = np.asarray(Image.open(face).convert('L'))
MAPS, MASK = pickle.load(open(pkl, 'rb'))[2:4]
if len(sys.argv) > 7:   # manifest R8: rings = permutation clusters from the pipeline's per-point table
    PP = pd.read_csv(sys.argv[7])
    PP = PP[(PP['align'] == 'lick') & (PP['position'] == 'far_R')]
    for e, g in PP.groupby('epoch'):
        g = g.sort_values('point')
        assert len(g) == len(MASK[('lick', 'far_R', e)]) and (g.point.to_numpy() == np.arange(len(g))).all()
        MASK[('lick', 'far_R', e)] = g.sig_cluster.astype(bool).to_numpy()
TC = np.load(tcf); T = TC['t_ms']
LD = pd.read_csv(ldf)
if 'unit' in LD.columns:   # v42: the pipeline's own point table (session_maps_scaled_lick_points.csv), not the SVG read-back
    LD = LD.assign(factor=LD.unit.map({'lick_right': 'lick contra', 'lick_left': 'lick ipsi'}), value_sd=LD.value)
EX = [(4, 'lick\ncontra'), (5, 'lick\nipsi'), (9, 'whisking'), (10, 'pre-lick\njaw')]
EPS = ['acute', 'subacute', 'chronic']
POS = [('close_L', 'near ipsi'), ('close_center', 'near mid'), ('close_R', 'near contra'),
       ('far_L', 'far ipsi'), ('far_center', 'far mid'), ('far_R', 'far contra')]
PCOL = {'close_L': '#DFC27D', 'close_center': '#BDBDBD', 'close_R': '#80CDC1',     # ColorBrewer BrBG (v26b):
        'far_L': '#8C510A', 'far_center': '#525252', 'far_R': '#01665E'}           # ipsi brown, contra teal
FS = 6.5                                    # printed font size (figure drawn at 1:1)
X0, X1, Y0, Y1 = 60, 610, 0, 415            # crop of the 680 x 680 face image around the mesh


def face_ax(ax, bg=True):
    if bg: ax.imshow(FACE, cmap='gray', vmin=0, vmax=255, alpha=0.55)
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_aspect('equal'); ax.axis('off')


def side_labels(ax):
    ax.text(0.27, -0.02, 'contra', transform=ax.transAxes, ha='center', va='top', fontsize=FS - 0.5, color='0.35')
    ax.text(0.73, -0.02, 'ipsi', transform=ax.transAxes, ha='center', va='top', fontsize=FS - 0.5, color='0.35')
    ax.axvline(np.median(H[:, 0]), color='0.6', lw=0.4)


def dir_colours(k):
    v = SP[:, :, k].T; w = np.hypot(v[:, 0], v[:, 1]); w = w / w.max()
    ang = np.degrees(np.arctan2(np.abs(v[:, 1]), np.abs(v[:, 0])))
    rgb = np.array([colorsys.hsv_to_rgb(2 / 3 * (1 - a / 90), 1, 0.95) for a in ang])
    return np.c_[rgb, w ** 0.7]   # as fr_factors_lib.report: value 0.95, opacity = weight^0.7 (no floor; v42)


_HR = [0.72, 0.3, 2.75, 0.50, 1.45]          # v26: C-D gap 0.85 -> 0.50 (legend moved beside C)
_UNIT = (0.955 - 0.04) * 5.7 / 6.07          # inches per height-ratio unit in the v41 figure (kept)
_FH = 0.045 * 5.7 + 0.04 * 5.7 + _UNIT * sum(_HR)
fig = plt.figure(figsize=(4.35, _FH))
_gs = fig.add_gridspec(5, 1, height_ratios=_HR, hspace=0, left=0.13, right=0.87, top=1 - 0.045 * 5.7 / _FH, bottom=0.04 * 5.7 / _FH)
gs0 = [_gs[0], _gs[2], _gs[4]]   # explicit gaps: small above C, room for the C legend above D
# ---------------------------------------------------------------- row 1: A, B
g1 = gs0[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.06)
ax = fig.add_subplot(g1[0]); face_ax(ax, bg=False); ax.imshow(FACE, cmap='gray', vmin=0, vmax=255)   # face at full contrast (Priya v41)
ax.scatter(H[:, 0], H[:, 1], s=0.5, color='0.85', lw=0, alpha=0.8)   # grey mesh
ax.set_title('point mesh', fontsize=FS, pad=2); ax.text(-0.12, 1.12, 'A', transform=ax.transAxes, fontsize=8, weight='bold')
nb = Normalize(-1, 1); cmb = colormaps['RdBu_r']
from scipy.spatial import cKDTree   # noqa: E402  (ptsv2: points sized from the mesh)
_NN = float(np.median(cKDTree(H).query(H, k=2)[0][:, 1]))   # median nearest-neighbor spacing, image px
_RLW = 0.2                                                    # ring line width, pt
for j, e in enumerate(EPS):
    ax = fig.add_subplot(g1[1 + j]); face_ax(ax, bg=False)
    v = MAPS[('lick', 'far_R', e)]; mk = MASK[('lick', 'far_R', e)]
    ax.apply_aspect()
    _ptpu = ax.get_position().width * fig.get_size_inches()[0] * 72 / (X1 - X0)   # pt per image px
    _D = 0.85 * _NN * _ptpu                                                       # point diameter, pt
    if j == 0:
        print(f'B: median NN {_NN:.2f} px = {_NN * _ptpu:.3f} pt; point {_D:.3f} pt; ring {_RLW} pt')
    ax.scatter(H[:, 0], H[:, 1], c=v, cmap=cmb, norm=nb, s=_D ** 2, lw=0)
    # rings as the pipeline draws them in S15 (wfield_local.face_rhythm_figures.sign_rings: dark red up / dark blue down,
    # ring 7 vs point 4) - the v41-v43 black rings smaller than the points were hard to see and did not match S15 (Priya 11-20)
    for sel, col in ((mk & (np.asarray(v) > 0), '#67001f'), (mk & (np.asarray(v) <= 0), '#053061')):
        if sel.any():
            ax.scatter(H[sel, 0], H[sel, 1], s=(_D - _RLW) ** 2, facecolor='none', edgecolor=col, lw=_RLW)   # ring = the point's own outline: the mesh is dense at this size (Priya: rings ran into each other)
    side_labels(ax); ax.set_title(f'{e} − pre', fontsize=FS, pad=2)
    if j == 0:
        ax.text(-0.08, 1.12, 'B', transform=ax.transAxes, fontsize=8, weight='bold')
        ax.text(1.55, 1.3, 'face movement speed, far contralesional spout, lick-aligned', transform=ax.transAxes, ha='center', fontsize=FS)
p = fig.axes[-1].get_position()
cax = fig.add_axes([p.x1 + 0.015, p.y0 + 0.1 * p.height, 0.012, 0.8 * p.height])
cb = fig.colorbar(ScalarMappable(nb, cmb), cax=cax, ticks=[-1, 0, 1]); cb.ax.tick_params(labelsize=FS, length=2, width=0.4)
cb.set_label('speed vs pre\n(log$_2$ fold)', fontsize=FS, labelpad=1); cb.outline.set_linewidth(0.4)
# ---------------------------------------------------------------- C: one row per example factor
_CW = [0.636, 0.10, 0.636, 0.22, 0.44, 0.32, 0.64]   # inches: weight | gap | direction | gap | spectrum | gap | amplitude
_cb = gs0[1].get_position(fig); _FW = fig.get_size_inches()[0]; _x0 = 0.13 * _FW
_g2 = fig.add_gridspec(4, len(_CW), width_ratios=_CW, wspace=0, hspace=0.3, left=0.13, right=(_x0 + sum(_CW)) / _FW,
                       top=_cb.y1, bottom=_cb.y0)
g2 = {(r, c): _g2[r, 2 * c] for r in range(4) for c in range(4)}
for r, (k, nm) in enumerate(EX):
    w = np.hypot(SP[0, :, k], SP[1, :, k])
    ax = fig.add_subplot(g2[r, 0]); face_ax(ax, bg=False); ax.imshow(FACE, cmap='gray', vmin=0, vmax=255)
    ax.scatter(H[:, 0], H[:, 1], c=w / w.max(), cmap='magma', vmin=0, vmax=1, s=0.8, lw=0)
    ax.text(-0.06, 0.5, nm, transform=ax.transAxes, ha='right', va='center', fontsize=FS, weight='bold', linespacing=1.0)
    if r == 0:
        ax.set_title('spatial factor', fontsize=FS, pad=2); ax.text(-0.62, 1.12, 'C', transform=ax.transAxes, fontsize=8, weight='bold')
    ax = fig.add_subplot(g2[r, 1]); face_ax(ax, bg=False); ax.imshow(FACE, cmap='gray', vmin=0, vmax=255)   # full-contrast face, as the pipeline report
    wk = np.hypot(SP[0, :, k], SP[1, :, k]); wk = wk / wk.max(); o = np.argsort(wk)   # heaviest points drawn last, size by weight
    ax.scatter(H[o, 0], H[o, 1], c=dir_colours(k)[o], s=0.1 + 1.6 * wk[o], lw=0)
    if r == 0: ax.set_title('direction of motion', fontsize=FS, pad=2)
    ax = fig.add_subplot(g2[r, 2]); sp = SPEC[:, k] / SPEC[:, k].max()
    ax.plot(FQ, sp, color='k', lw=S.TRAJ_LW); ax.set_xscale('log'); ax.set_xlim(1, 60); ax.set_ylim(0, 1.45)
    pk = FQ[np.argmax(sp)]; ax.axvline(pk, ymax=1.0 / 1.45, color='0.7', lw=0.4, ls=':')
    fx = np.log(pk) / np.log(60)                    # v26: label in a band above the curve (narrower column), over the peak
    lx, lha = (0.07, 'left') if fx < 0.35 else (0.99, 'right') if fx > 0.65 else (fx, 'center')
    ax.text(lx, 0.99, f'{pk:.1f} Hz', transform=ax.transAxes, ha=lha, va='top', fontsize=FS - 0.5)
    ax.set_xticks([1, 10, 60]); ax.set_xticklabels(['1', '10', '60'] if r == 3 else []); ax.set_yticks([0, 1])
    ax.tick_params(labelsize=FS - 0.5, length=2, width=0.4)
    if r == 3: ax.set_xlabel('frequency (Hz)', fontsize=FS, labelpad=1)
    if r == 0: ax.set_title('spectral factor', fontsize=FS, pad=2)
    ax = fig.add_subplot(g2[r, 3]); sel = (T >= -500) & (T <= 2000)
    for p_, pn in POS:
        a = TC[p_][:, :, k]; n = np.isfinite(a).sum(0); m = np.nanmean(a, 0); se = np.nanstd(a, 0, ddof=1) / np.sqrt(n)
        ax.fill_between(T[sel], (m - se)[sel], (m + se)[sel], color=PCOL[p_], alpha=S.TRAJ_BAND, lw=0)
        ax.plot(T[sel], m[sel], color=PCOL[p_], lw=S.TRAJ_LW, label=pn)
    ax.axvline(0, color='0.75', lw=0.4); ax.set_xlim(-500, 2000); ax.set_xticks([0, 1000, 2000])
    ax.set_xticklabels(['0', '1', '2'] if r == 3 else []); ax.tick_params(labelsize=FS - 0.5, length=2, width=0.4)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    if r == 3: ax.set_xlabel('time from cue (s)', fontsize=FS, labelpad=1)
    if r == 0: ax.set_title('amplitude, cue-aligned', fontsize=FS, pad=2)
    if r == 0:   # v26: one column right of the amplitude column, canonical order (near ipsi, mid, contra, far ipsi, mid, contra)
        ax.legend(fontsize=FS - 1, frameon=False, handlelength=0.9, handletextpad=0.35, labelspacing=0.25, loc='upper left',
                  bbox_to_anchor=(1.04, 1.0), borderaxespad=0, ncol=1)
# ---------------------------------------------------------------- D: lick-factor loading change
g3 = gs0[2].subgridspec(2, 3, wspace=0.05, hspace=0.3)
nd = Normalize(-5, 5); cmd = colormaps['RdBu_r']
for gi, fac in enumerate(['lick contra', 'lick ipsi']):
    for j, e in enumerate(EPS):
        ax = fig.add_subplot(g3[gi, j]); face_ax(ax, bg=False)
        d = LD[(LD.factor == fac) & (LD.epoch == e)].sort_values('point')
        ax.scatter(H[:, 0], H[:, 1], c=d.value_sd.values, cmap=cmd, norm=nd, s=1.4, lw=0)
        for sgn, col in ((1, '#67001F'), (-1, '#053061')):
            q = d.ring.values == sgn
            ax.scatter(H[q, 0], H[q, 1], s=0.7, facecolor='none', edgecolor=col, lw=0.1)
        side_labels(ax)
        if gi == 0: ax.set_title(f'{e} − pre', fontsize=FS, pad=2)
        if j == 0: ax.text(-0.06, 0.5, fac.replace(' ', '\n'), ha='right', va='center', transform=ax.transAxes, fontsize=FS, weight='bold')
        if gi == 0 and j == 0: ax.text(-0.62, 1.12, 'D', transform=ax.transAxes, fontsize=8, weight='bold')
        if gi == 0 and j == 2: pD = ax.get_position()
        if gi == 1 and j == 2: pD1 = ax.get_position()
cax = fig.add_axes([pD.x1 + 0.015, pD1.y0 + 0.1 * (pD.y1 - pD1.y0), 0.012, 0.8 * (pD.y1 - pD1.y0)])
cb = fig.colorbar(ScalarMappable(nd, cmd), cax=cax, ticks=[-5, 0, 5], extend='max')   # positive values clipped at +5 SD (v42)
cb.ax.tick_params(labelsize=FS - 1, length=2, width=0.4)
cb.set_label('spatial factor vs pre (SD)', fontsize=FS - 1, labelpad=1); cb.outline.set_linewidth(0.4)
S.save(fig, out)
