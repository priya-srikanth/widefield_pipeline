import sys; sys.path.insert(0,'.')
from build_fig import place, place_png, place_crop, caption

PT=72.0; PAGE_W=612.0; FIG_W=7.4*PT; X0=(PAGE_W-FIG_W)/2; Y0=44.0
GX, GY = 12.0, 16.0
P='K08_2026_grant_panels/'
# v7 Fig 5H, in widefield.svg points. Tightened on 9 Nov so the crop no longer
# drags in the neighbouring panel's acute / subacute / chronic key (which ends at
# x 194) and so the panel letter 'H' from the source figure is masked out.
AREAKEY  = (232.0, 316.0, 322.0, 396.0)
AREAKEY_H = (232.0, 316.0, 11.0, 16.5)        # x, y, w, h of the stray 'H', source pts


def txt(x, y, t, size=7.0, anchor='middle', weight='normal', fill='#000', rot=None):
    a = ('<text x="%.2f" y="%.2f" font-family="Arial" font-size="%.1f" '
         'font-weight="%s" fill="%s" text-anchor="%s"' % (x, y, size, weight, fill, anchor))
    if rot is not None:
        a += ' transform="rotate(%.1f %.2f %.2f)"' % (rot, x, y)
    return a + '>%s</text>' % t

def cover(x, y, w, h):
    return '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="#fff"/>' % (x, y, w, h)

parts, y = [], Y0
def row(items):
    global y
    x=X0; rh=0
    for it in items:
        kind, src, frac, lab = it[0], it[1], it[2], it[3]
        w=FIG_W*frac
        if kind=='svg':    s,h = place(src, lab, x, y, w, label=lab)
        elif kind=='png':  s,h = place_png(src, lab, x, y, w, label=lab)
        else:              s,h = place_crop(src, lab, x, y, w, it[4], label=lab)
        parts.append(s); rh=max(rh,h); x += w + GX
    y += rh + GY

import json as _json
_LAY = _json.load(open('rows_fig4_layout.json'))
def full_row(name, letters):
    global y
    s, h = place(P + name + '.svg', name, X0, y, FIG_W, label=None)
    for k, (l_in, top_in) in _LAY[name]['axes'].items():
        if k not in letters: continue
        s += ('\n<text x="%.2f" y="%.2f" font-family="Arial" font-weight="bold" '
              'font-size="11">%s</text>' % (X0 + (l_in - 0.56)*72.0, y + (top_in - 0.02)*72.0,
                                            letters[k]))
    parts.append(s); y += h + 4.0
# THE TEST: a decoder frozen before the stroke vs one refit after
full_row('fig4_row_AB', {'a': 'A', 'b': 'B'})

# WHAT THE ACTIVITY ITSELF DOES, by region
# C: ROI time courses. The deck raster carries 1.3 pt titles, tick numbers and
# a repeated x label once placed, so those are covered and redrawn here.
_wc = FIG_W*0.60
_XC = X0 + 16.0
_sc, _hc = place_png('deck2/f3_traces_4roi.png', 'c', _XC, y, _wc, label='c', label_dx=-16)
_k = _wc/1385.0
# Geometry measured from the 1385 x 352 source: left spines at x 66/407/747/1088,
# x axis at y 305, x ticks every 53 px from each spine (-2 ... 3 s); y ticks
# (30 ... 0, step 5) at the rows below. Axes are NOT shared in the source.
_SP = [66, 407, 747, 1088]
_YT = [(45, 284), (36, 279), (54, 284), (55, 286)]     # rows of 30 and 0 per panel
_sc += '\n' + cover(_XC, y, _wc, 24*_k)                            # deck titles
_sc += '\n' + cover(_XC - 2, y + 311*_k, _wc + 4, 42*_k)           # tick numbers + x labels
for _s in _SP:
    _sc += '\n' + cover(_XC + (_s - 62)*_k, y + 24*_k, 56*_k, 287*_k) if False else ''
    _sc += '\n' + cover(_XC + (_s - 60)*_k, y + 24*_k, 54*_k, 287*_k)   # y tick numbers
for _s, (_r30, _r0), _lab in zip(_SP, _YT, ['tjM1 contra', 'tjM1 ipsi', 'SSp-m contra', 'SSp-m ipsi']):
    _sc += '\n' + txt(_XC + (_s + 133)*_k, y + 16*_k, _lab, size=7.0, weight='bold')
    for _v in (0, 10, 20, 30):
        _ry = _r0 + (_r30 - _r0)*_v/30.0
        _sc += '\n' + txt(_XC + (_s - 7)*_k, y + _ry*_k + 2.2, str(_v), size=6.6, anchor='end')
    for _j, _v in enumerate(('\u22122', '\u22121', '0', '1', '2', '3')):
        _sc += '\n' + txt(_XC + (_s + 53.2*_j)*_k, y + 305*_k + 8.6, _v, size=6.6)
_sc += '\n' + txt(_XC + _wc/2, y + 305*_k + 17.5, 'time from spontaneous lick onset (s)', size=7.0)
_sc += '\n' + txt(X0 + 6.0, y + 190*_k, '\u0394F/F (%)', size=7.0, rot=-90)
_hc = 305*_k + 20.0
parts.append(_sc); y += _hc + GY
# D + E and F + G + H: each row is ONE canvas rendered at final size by
# render_fig4_rows.py, so the plot areas in a row share their x-axis line
# exactly and the type is true size (placed at scale 1).
full_row('fig4_row_DE', {'d': 'D', 'e': 'E'})
# F: hemispheric balance against far-contra aim (new); G, H: state-space trajectory
full_row('fig4_row_FGH', {'f': 'F', 'g': 'G', 'h': 'H'})

# Allen area key beside the maps it labels. The .ai -> SVG conversion paints the
# parcellation through a luminance mask that Illustrator drops, so there the brain
# vanished and only the labels remained; placed instead as a 600 dpi raster from
# the PDF, with the source figure's own panel letter already painted out.
_w = FIG_W*0.22
# a title for J, over the maps only, so the reader knows what the change maps are
parts.append('<text x="%.2f" y="%.2f" font-family="Arial" font-weight="bold" font-size="8.5" '
             'text-anchor="middle">Regional activity change along the pre-stroke to chronic recovery axis</text>'
             % (X0 + _w + GX + FIG_W*0.72/2, y + 8.0))
y += 13.0
_s, _h = place_png('airaster/panel_areakey_nolab.png', 'i', X0, y, _w, label='i')
for _px, _py, _lab in ((228, 222, 'ALM'), (190, 282, 'tjM1')):
    _s += ('\n<text x="%.2f" y="%.2f" font-family="Arial" font-weight="bold" font-size="9" '
           'fill="#1a1a1a" text-anchor="middle">%s</text>' % (X0 + _px/750.0*_w, y + _py/750.0*_w, _lab))
for _f, _lab in ((0.33, 'I'), (0.75, 'C')):      # image left = ipsilesional
    _s += ('\n<text x="%.2f" y="%.2f" font-family="Arial" font-size="7" '
           'fill="#8c8c8c" text-anchor="middle">%s</text>'
           % (X0 + _f*_w, y + _h + 7.5, _lab))
_s += ('\n<text x="%.2f" y="%.2f" font-family="Arial" font-weight="bold" font-size="8">Reference areas</text>' % (X0 + 2.0, y + 10.0))
parts.append(_s)
_wi = FIG_W*0.72
_si, _hi = place(P+'ss_full_nocw_frKBs.svg', 'j', X0 + _w + GX, y, _wi, label='j')
parts.append(_si)
y += max(_h + 10, _hi) + GY

CAP = ("Striatal stroke leaves a motor code that is reorganised rather than restored, with the "
 "activity change loaded on contralesional orofacial sensorimotor cortex. "
 "(A) Decoding accuracy for a decoder frozen before the stroke, one refit within each post-stroke "
 "session, and their difference, on LocaNMF components (August pre baseline; asterisks, Bonferroni-corrected "
 "paired bootstrap vs pre). Position information is present after the stroke but displaced, and the gap persists "
 "through recovery. "
 "(B) Like-for-like readout offset of the first cued lick. "
 "(C) Response time courses around spontaneous (inter-trial) lick onset, by region and epoch (pre in black). "
 "(D) Lick-response AUC as a fold of each animal's own pre; activity rises at every orofacial "
 "region and stays raised. "
 "(E) Contralesional:ipsilesional balance, log2 of the contralesional / ipsilesional lick-response AUC (0 to 1 s "
 "after the first cued lick), at the far contralesional spout, tjM1 and SSp-m: the balance shifts contralesionally "
 "and does not return. "
 "(F) The same balance (SSp-m) against far contralesional aim, one point per session: sessions with a more "
 "contralesional balance had a larger aim deficit, within animals and beyond time from stroke (mixed model "
 "with log time, p < 10^\u22129; three of four animals individually). Dotted lines, per-animal fits over "
 "post-stroke sessions. "
 "(G) Position on the pre-to-chronic axis per session against days from stroke. "
 "(H) Each session in the plane of that axis and the orthogonal acute departure; acute sessions "
 "are displaced off the recovery axis rather than part-way along it. In G and H, asterisks mark "
 "epochs displaced from pre (paired bootstrap over animals, then sessions). "
 "(I) Areas for reference in J: ALM, anterolateral motor cortex; tjM1, tongue-jaw motor cortex; SSp-m and SSp-n, mouth and nose primary somatosensory cortex. Image left is the ipsilesional hemisphere (I), right the "
 "contralesional one (C), throughout. "
 "(J) Regional activity change along the pre-stroke to chronic recovery axis: lick-aligned pre-stroke response (units of pre RMS) and chronic minus pre, for the mean over "
 "spouts and the far contralesional spout, with lick kinematics, spout contact and facial movement "
 "regressed out. Left, the full change; right, the same after removing the cortex-wide mode, which "
 "alone carries most of the mean response. Contours, significant bins (nested bootstrap, "
 "max-statistic); white, tissue not analysed. A structured redistribution remains once the global "
 "mode is removed, and it is largest in tjM1 and SSp-m of both hemispheres. "
 "Faded or open markers are sessions, shape by animal; filled markers are means with their "
 "interval. Asterisks are each panel's own test, as given in Methods.")
import os
if not os.environ.get('NOCAP'):
    s,h = caption(X0, y+10, FIG_W, CAP, bold_lead='Figure 5.')
    parts.append(s); y += h + 16

H=y
svg=('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
 'xmlns:xlink="http://www.w3.org/1999/xlink" width="%.1fpt" height="%.1fpt" viewBox="0 0 %.1f %.1f">\n'
 '<rect width="100%%" height="100%%" fill="#fff"/>\n<g id="Figure4_decoder_encoder">\n%s\n</g>\n</svg>\n'
 % (PAGE_W,H,PAGE_W,H,'\n'.join(parts)))
open(('Figure5_decoder_encoder_DRAFT4_panels' if os.environ.get('NOCAP') else 'Figure5_decoder_encoder_DRAFT4')+'.svg','w',encoding='utf-8').write(svg)
print('wrote %.0f x %.0f pt (%.1f in tall), %.1f KB' % (PAGE_W,H,H/72,len(svg)/1024))
