"""Figure 2C: cam1 and cam4 at one timepoint, with tongue, jaw, spout and nose.

Frames and keypoints come from the production pose models (cam1
lp_cam1_ctx_prod_20261019, cam4 lp_r4_prod_20261015); the two cameras are paired
on the DAQ clock, which for this session is cam4 = cam1 + 8 frames. Moment:
PS92, 2026-08-07 (pre-stroke), far contralesional spout, trial 18, peak of the
lick. Markers are redrawn here rather than reusing the pose machine's overlay,
whose rings and type would be a couple of points once the panel is placed.
"""
import sys, json; sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from grant_panels import save, set_scale

KP = json.load(open('camframes/kp_t18.json'))
# a palette of its own: the epoch colours (grey / magenta / gold / teal) are
# reserved throughout the figures, so body parts must not borrow them
COL = {'tongue': '#ff3b1f', 'jaw': '#2b8cff', 'spout': '#b45ff0', 'nose': '#00e5ff'}
#: crop as (left, top, right, bottom) in the source frame's own pixels: the whole
#: head with a comfortable margin, without the empty fur at the edges
#: both frames cropped to the same 520 x 545 box, so the two axes match and
#: the pair can be scaled to the height of the histology panel beside it
CROP = {'cam1': (40, 55, 560, 600), 'cam4': (95, 115, 615, 660)}
LAB_DX = {'cam1': {'nose': (22, 0), 'jaw': (22, 6), 'tongue': (-22, 4), 'spout': (-22, -6)},
          'cam4': {'nose': (22, 0), 'jaw': (22, 4), 'tongue': (-22, 6), 'spout': (-22, -6)}}
TITLE = {'cam1': 'bottom view', 'cam4': 'front view'}

set_scale('cam_frames', 2.62)
fig, axes = plt.subplots(1, 2, figsize=(2.62, 1.49))
for ax, cam in zip(axes, ('cam1', 'cam4')):
    im = Image.open('camframes/PS92_20260807_far_R_t18_%s_raw.png' % cam).convert('L')
    l, t, r, b = CROP[cam]
    ax.imshow(im.crop((l, t, r, b)), cmap='gray', vmin=0, vmax=255)
    for part, (x, y, _c) in zip(KP[cam]['parts'], KP[cam]['xy']):
        ax.plot(x - l, y - t, 'o', ms=5.0, mfc='none', mec=COL[part], mew=1.1, zorder=3)
        dx, dy = LAB_DX[cam][part]
        ax.annotate(part, (x - l, y - t), (x - l + dx, y - t + dy),
                    arrowprops=dict(arrowstyle='-', color=COL[part], lw=0.7,
                                    shrinkA=1.0, shrinkB=4.0),
                    color=COL[part], fontsize=6.6, fontweight='bold',
                    ha='left' if dx > 0 else 'right', va='center', zorder=4,
                    path_effects=None)
    ax.set_title(TITLE[cam], fontsize=7.0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.subplots_adjust(wspace=0.04, left=0.005, right=0.995, top=0.90, bottom=0.01)
save(fig, 'cam_frames')
