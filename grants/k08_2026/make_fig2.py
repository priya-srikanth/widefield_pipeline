import sys; sys.path.insert(0,'.')
from build_fig import place, place_png, place_crop, caption

P = 'K08_2026_grant_panels/'

PT=72.0; PAGE_W=612.0; FIG_W=7.4*PT; X0=(PAGE_W-FIG_W)/2; Y0=44.0
GX, GY = 12.0, 15.0
CROPS = {'c': (0.0, 1095.0, 792.0, 1330.0),   # far_R row of aligned_maps_lick_perm
         'e': (0.0, 618.0, 280.0, 778.0)}     # side-specificity subplot of session_scalars

def txt(x, y, t, size=7.2, anchor='middle', weight='normal', fill='#000', rot=None):
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
    for kind,src,frac,lab in items:
        w=FIG_W*frac
        if kind=='svg':    s,h = place(src, lab, x, y, w, label=lab)
        elif kind=='crop': s,h = place_crop(src, lab, x, y, w, CROPS[lab], label=lab)
        else:              s,h = place_png(src, lab, x, y, w, label=lab)
        parts.append(s); rh=max(rh,h); x += w + GX
    y += rh + GY

def ttl(x, y0, t):
    return txt(x + 2.0, y0 + 10.0, t, size=8.0, anchor='start', weight='bold')
# A: the tracked mesh on the face
_y0 = y; _TG = 20.0; y += _TG
_wa = FIG_W*0.26
_sa, _ha = place_png('fr/fr_mesh_face.png', 'a', X0, y, _wa, label='a', label_dy=10-_TG)
_sa += '\n' + ttl(X0, _y0, 'Face mesh')
parts.append(_sa)

# B: four frozen factors. The montage carries the deck's own tiny titles and
# axis numbers (about 1.3 pt once placed), so they are covered and replaced with
# one readable label per factor plus shared column headers.
_xb = X0 + _wa + GX
_wb = FIG_W*0.68
_sb, _hb = place_png('fr/fr_4factors.png', 'b', _xb, y, _wb, label='b', label_dy=10-_TG)
_sb += '\n' + ttl(_xb, _y0, 'Face Rhythm factors (frozen, pre-stroke)')
_k = _wb/2704.0                       # png px -> output pt
CW, CH, BGX, BGY = 1332.0, 404.0, 40.0, 18.0
_cells = [('factor 4', 'lick right', '6.5 Hz'), ('factor 5', 'lick left', '6.5 Hz'),
          ('factor 9', 'whisking and running', '14.7 Hz'), ('factor 10', 'jaw, pre-lick', '2.9 Hz')]
for _i, (_f, _what, _hz) in enumerate(_cells):
    _cx = _xb + (_i % 2)*(CW+BGX)*_k
    _cy = y   + (_i//2)*(CH+BGY)*_k
    _sb += '\n' + cover(_cx, _cy, CW*_k, 62*_k)                 # the three titles
    _sb += '\n' + cover(_cx, _cy + 358*_k, CW*_k, 46*_k)        # the x-axis numbers
    _sb += '\n' + cover(_cx + 748*_k, _cy + 62*_k, 86*_k, 286*_k)   # the y-axis numbers
    _sb += '\n' + txt(_cx + 6*_k, _cy + 54*_k, '%s &#183; %s' % (_f, _what),
                       size=6.6, anchor='start', weight='bold')
    _sb += '\n' + txt(_cx + 1060*_k, _cy + 54*_k, 'peak %s' % _hz, size=6.4)
for _ci, _lab in ((168.0, 'spatial weight'), (585.0, 'direction of motion')):
    _sb += '\n' + txt(_xb + _ci*_k, y - 3.0, _lab, size=6.8)
_sb += '\n' + txt(_xb + 1060*_k, y + (2*CH + BGY)*_k + 7.0, 'frequency, 1-50 Hz (log)', size=6.4)
parts.append(_sb)
y += max(_ha, _hb) + GY + 8.0
# C: far contralesional-spout face movement, the three epoch maps plus the
# figure's own colour bar, lifted from elsewhere in the same file. The source
# titles carry the cluster statistics in 3 pt type; those move to the caption.
C_MAPS = (0.0, 1144.0, 700.0, 1310.0)   # the far_R row only; its titles sit ABOVE 1144
C_BAR  = (700.0, 455.0, 792.0, 905.0)
_y0 = y; _TG = 11.0; y += _TG
_wc = FIG_W*0.58
_LAB = 11.0                              # room above the maps for our own labels
_sc, _hc = place_crop('fr/aligned_maps_lick_perm_mesh.svg', 'c', X0, y + _LAB, _wc,
                      C_MAPS, label='c', label_dy=10-_LAB-_TG)
_sc_t = ttl(X0, _y0, 'Face movement speed change, far contra trials')
_k = _wc/(C_MAPS[2]-C_MAPS[0])
for _mx, _lab in ((117.0, 'acute &#8722; pre'), (345.0, 'subacute &#8722; pre'),
                  (573.0, 'chronic &#8722; pre')):
    _sc += '\n' + txt(X0 + _mx*_k, y + 8.0, _lab, size=7.2, weight='bold')
parts.append(_sc); parts.append(_sc_t)
# our own bar, for the same reason as panel D
import re as _re
_cbh = max(_hc*0.88, 78.0)
_cbw = _cbh*34.235/97.966
_sb2, _ = place(P + 'cbar_speed.svg', 'cbar', X0 + _wc + 3.0, y + _LAB + (_hc-_cbh)/2, _cbw)
parts.append(_sb2)
_hc += _LAB
y += _hc + GY
# D: lick-factor loading change. Drop the suptitle, cover the per-map titles,
# and label rows and columns at a size that survives the placement.
D_CROP = (0.0, 18.0, 756.0, 446.0)
_y0 = y; _TG = 11.0; y += _TG
_wd = FIG_W*0.50
_sd, _hd = place_crop('fr/session_maps_scaled_lick_mesh.svg', 'd', X0, y, _wd, D_CROP, label='d', label_dy=10-_TG)
parts.append(ttl(X0, _y0, 'Lick factor loading change'))
_k = _wd/(D_CROP[2]-D_CROP[0])
for _ty in (22.0, 226.0):                      # the two title bands, source pts
    _sd += '\n' + cover(X0, y + (_ty-18.0)*_k, 690*_k, 52*_k)
for _mx, _lab in ((115.0, 'acute &#8722; pre'), (337.0, 'subacute &#8722; pre'),
                  (560.0, 'chronic &#8722; pre')):
    _sd += '\n' + txt(X0 + _mx*_k, y + (44.0-18.0)*_k, _lab, size=7.0, weight='bold')
for _ry, _lab in ((130.0, 'contralesional'), (350.0, 'ipsilesional')):
    _sd += '\n' + txt(X0 + 6.0*_k, y + (_ry-18.0)*_k, _lab + ' lick factor',
                       size=6.8, rot=-90)
_sd += '\n' + cover(X0 + 688*_k, y, _wd - 688*_k, _hd)      # bar and its 2 pt numbers
parts.append(_sd)

# a bar of our own, at the figure's type size, placed clear of panel D so no
# cover can sit on top of it (cropping the source's 6 pt strip proved fragile)
_bh = _hd*0.80
_bwd = _bh*34.221/123.264
_sdb, _ = place(P + 'cbar_loading.svg', 'dbar', X0 + _wd + 3.0, y + _hd*0.10, _bwd)
parts.append(_sdb)

# E: side specificity, already readable
_xe = X0 + _wd + 3.0 + _bwd + GX
_we = FIG_W*0.30
_se, _he = place_crop('fr/session_scalars.svg', 'e', _xe, y, _we,
                      (0.0, 618.0, 280.0, 778.0), label='e', label_dy=10-_TG)
parts.append(ttl(_xe, _y0, 'Side specificity'))
# the source draws its stars in one straight row; cover them and redraw with a
# small vertical stagger so neighbouring epochs do not run into each other
_ke = _we/280.0
_se += '\n' + cover(_xe + 128*_ke, y + (634-618)*_ke, 136*_ke, 22*_ke)
_STARS = [(153.5, 'acute'), (166.0, 'subacute'), (180.0, 'chronic'),
          (217.0, 'acute'), (230.0, 'subacute')]
_SY = {'acute': 652.0, 'subacute': 646.5, 'chronic': 641.0}
for _sx, _ep in _STARS:
    _se += '\n' + txt(_xe + _sx*_ke, y + (_SY[_ep]-618)*_ke, '**',
                       size=8.4, weight='bold', fill=
                       {'acute':'#e0218a','subacute':'#d4a017','chronic':'#1b7f79'}[_ep])
# the source's tick numbers, axis label and factor names are ~5 pt once placed
# and use code names (lick_right / lick_left): cover and redraw at grant size
def _E(sx, sy): return _xe + sx*_ke, y + (sy - 618.0)*_ke
_c = _E(92, 746); _se += '\n' + cover(_c[0], _c[1], 188*_ke, 32*_ke)
_c = _E(96, 660); _se += '\n' + cover(_c[0], _c[1], 23*_ke, 66*_ke)
_c = _E(70, 630); _se += '\n' + cover(_c[0], _c[1], 26*_ke, 116*_ke)
for _sy, _v in ((673.2, '0'), (711.2, '\u22120.5')):
    _p = _E(118.5, _sy); _se += '\n' + txt(_p[0], _p[1] + 2.3, _v, size=6.6, anchor='end')
for _sx, _v in ((159.8, 'contra'), (223.6, 'ipsi')):
    _p = _E(_sx, 743.3); _se += '\n' + txt(_p[0], _p[1] + 8.4, _v, size=6.8)
_p = _E(191.7, 743.3); _se += '\n' + txt(_p[0], _p[1] + 16.0, 'lick factor', size=6.8)
_p = _E(84, 692); _se += '\n' + txt(_p[0], _p[1], 'side specificity (\u0394 vs pre)', size=6.8, rot=-90)
parts.append(_se)
# the source panel's key sits elsewhere in its figure, so draw ours, in the
# colours every other panel uses
_EC = [('pre', '#808080'), ('acute', '#e0218a'), ('subacute', '#d4a017'), ('chronic', '#1b7f79')]
_lx = _xe + 14.0
for _nm, _col in _EC:
    parts.append('<circle cx="%.2f" cy="%.2f" r="2.4" fill="%s" stroke="#000" stroke-width="0.4"/>'
                 % (_lx, y + _he + 17.0, _col))
    parts.append(txt(_lx + 4.5, y + _he + 19.3, _nm, size=6.6, anchor='start'))
    _lx += 9.0 + len(_nm)*3.3
y += max(_hd, _he + 12.0) + GY

CAP = ("Facial movement decomposes into rhythmic factors, and after stroke contralesional licks "
 "recruit the ipsilesional lick pattern. "
 "(A) Tracked point mesh on the face. "
 "(B) Four of twelve frozen pre-stroke factors: spatial weight, direction of motion and spectrum. "
 "(C) Face movement speed per point on far contralesional-spout trials, each epoch minus pre; "
 "rings mark permutation clusters, p < 0.05. Movement shifts off the contralesional half of the "
 "face onto the ipsilesional half and stays shifted. "
 "(D) Change in each lick factor's spatial loading per point, in units of the animal's own "
 "pre-session SD; red is more weight after stroke. "
 "(E) Side specificity: how well each lick factor's loading still matches the animal's pre-stroke "
 "movement toward its own side of the face, minus the opposite side, each session relative to its "
 "own pre. Both factors become less side-specific; the contralesional one is still below pre at "
 "chronic. Open markers are sessions, shape by animal; filled markers mean \u00b1 SEM. "
 "Asterisks, Holm-corrected within-animal permutation.")
import os
if not os.environ.get('NOCAP'):
    s,h = caption(X0, y+10, FIG_W, CAP, bold_lead='Figure 4.')
    parts.append(s); y += h + 16

H=y
svg=('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
 'xmlns:xlink="http://www.w3.org/1999/xlink" width="%.1fpt" height="%.1fpt" viewBox="0 0 %.1f %.1f">\n'
 '<rect width="100%%" height="100%%" fill="#fff"/>\n<g id="Figure3_facerhythm">\n%s\n</g>\n</svg>\n'
 % (PAGE_W,H,PAGE_W,H,'\n'.join(parts)))
open(('Figure4_facerhythm_DRAFT4_panels' if os.environ.get('NOCAP') else 'Figure4_facerhythm_DRAFT4')+'.svg','w',encoding='utf-8').write(svg)
print('wrote %.0f x %.0f pt (%.1f in tall), %.1f KB' % (PAGE_W,H,H/72,len(svg)/1024))
