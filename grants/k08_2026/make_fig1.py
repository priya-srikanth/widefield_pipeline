import sys; sys.path.insert(0,'.')
from build_fig import place, place_png, place_crop, caption

PT=72.0; PAGE_W=612.0; FIG_W=7.4*PT; X0=(PAGE_W-FIG_W)/2; Y0=44.0
GX, GY = 12.0, 15.0
IN=lambda v: v*72.0
P='K08_2026_grant_panels/'

CROP_SIXSPOUT = (34.0, 28.0, 212.0, 150.0)
CROP_HIST     = (IN(0.95), IN(0.02), IN(2.15), IN(1.30))
CROP_FLUOROA  = (IN(0.05), IN(0.10), IN(5.55), IN(1.20))
FL = 'fluoro/all_graphs.svg'
R1 = (27.8, 124.0); R2 = (135.8, 232.0)
COL = {'A': (1.4, 104.0), 'B': (105.6, 207.0), 'C': (208.8, 312.0),
       'D': (313.0, 416.0), 'E': (416.6, 518.0)}


def txt(x, y, t, size=7.0, anchor='middle', weight='normal', fill='#000', rot=None):
    a = ('<text x="%.2f" y="%.2f" font-family="Arial" font-size="%.1f" '
         'font-weight="%s" fill="%s" text-anchor="%s"' % (x, y, size, weight, fill, anchor))
    if rot is not None:
        a += ' transform="rotate(%.1f %.2f %.2f)"' % (rot, x, y)
    return a + '>%s</text>' % t

def cover(x, y, w, h):
    return '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="#fff"/>' % (x, y, w, h)

parts, y = [], Y0
#: short panel titles, drawn on the panel-letter line; content sits TG pt lower
TG = 11.0
TITLES = {'a': 'Six-position licking task', 'b': 'VLS infarct', 'c': 'Thalamic gliosis (earlier cohort, severe)', 'n': '3-D tongue and jaw paths, front view',
          'd': 'Videofluoroscopic swallow study', 'e': 'Swallow rate', 'f': 'Swallow interval',
          'g': 'Licks per swallow', 'h': 'Lick rate', 'i': 'Bolus amount', 'j': 'Esophageal transit',
          'k': 'Hit rate', 'l': 'First-lick aim', 'm': 'Tracked camera views'}
def ttl(x, y0, lab):
    t = TITLES.get(lab)
    return txt(x + 2.0, y0 + 10.0, t, size=8.0, anchor='start', weight='bold') if t else ''
def row(items):
    global y
    x=X0; rh=0; y0=y; yc=y+TG; ldy=10.0-TG
    for it in items:
        kind, src, frac, lab = it[0], it[1], it[2], it[3]
        w=FIG_W*frac
        if kind=='svg':    s,h = place(src, lab, x, yc, w, label=lab, label_dy=ldy)
        elif kind=='png':  s,h = place_png(src, lab, x, yc, w, label=lab, label_dy=ldy)
        elif kind=='key':  s,h = src(lab, x, yc, w)
        else:              s,h = place_crop(src, lab, x, yc, w, it[4], label=lab, label_dy=ldy)
        # Cover the panel letter inherited from the source figure, so each panel
        # carries only this figure's own capital letter. MASK is in the SOURCE
        # file's points, measured from the crop's top-left corner.
        mk = it[5] if (kind == 'crop' and len(it) > 5) else None
        if mk:
            c = it[4]; sc = w/(c[2]-c[0])
            fill = mk[4] if len(mk) > 4 else '#fff'
            s += ('\n<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>'
                  % (x + mk[0]*sc, yc + mk[1]*sc, mk[2]*sc, mk[3]*sc, fill))
        s += '\n' + ttl(x, y0, lab)
        parts.append(s); rh=max(rh,h); x += w + GX
    y += rh + TG + GY

def swallow_key(lab, x, y, w):
    """K drawn natively (no crop of the source legend, whose glyphs were clipped
    in Illustrator). 2 x 2, same handles and colours as the source graphs."""
    g = ['<g id="Key_EJ">']   # shared key for E-J: no panel letter
    items = [('line', '#f2c12e', 'mild stroke (n = 4)'), ('line', '#b4001e', 'severe stroke (n = 5)'),
             ('x', '#6b6b6b', 'no swallows (undefined)'), ('o', '#6b6b6b', '< 5 intervals')]
    for i, (kind, col, t) in enumerate(items):
        cx = x + 8 + (i % 2) * w * 0.48; cy = y + 8 + (i // 2) * 13
        if kind == 'line':
            g.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="1.3"/>'
                     % (cx - 6, cy, cx + 6, cy, col))
            g.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="1"/>'
                     % (cx, cy - 4, cx, cy + 4, col))
            g.append('<circle cx="%.2f" cy="%.2f" r="2.2" fill="%s"/>' % (cx, cy, col))
        elif kind == 'x':
            g.append('<path d="M%.2f %.2fL%.2f %.2fM%.2f %.2fL%.2f %.2f" stroke="%s" stroke-width="1"/>'
                     % (cx-2.5, cy-2.5, cx+2.5, cy+2.5, cx-2.5, cy+2.5, cx+2.5, cy-2.5, col))
        else:
            g.append('<circle cx="%.2f" cy="%.2f" r="2.5" fill="none" stroke="%s" stroke-width="1"/>' % (cx, cy, col))
        g.append('<text x="%.2f" y="%.2f" font-family="Arial" font-size="7.5">%s</text>'
                 % (cx + 10, cy + 2.7, t.replace('<', '&lt;')))
    g.append('</g>')
    return '\n'.join(g), 30.0


def thal_panel(lab, x, y, w):
    """C: ipsilesional thalamus in a severe animal of the earlier two-tone cohort (PS55,
    Bregma -1.34 mm), with and without GFAP, Paxinos plate outlines overlaid (the box
    alignment set per section on the alignment page). Built by thalamus_overlay_options.py."""
    s, h = place_png('airaster/panel_thal.png', 'c', x, y, w, label=lab, label_dy=10.0 - TG)
    iw = w * 1600.0 / 3240.0
    for k, t in enumerate(('DAPI / NeuN / GFAP', 'DAPI / NeuN')):
        s += '\n' + txt(x + k*(w - iw) + 3.0, y + 8.0, t, size=6.6, anchor='start', weight='bold', fill='#ffffff')
    bl = 0.5/5.8*iw
    s += ('\n<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#fff" stroke-width="1.6"/>'
          % (x + iw - 4 - bl, y + h - 5, x + iw - 4, y + h - 5))
    s += '\n' + txt(x + iw - 4 - bl/2, y + h - 8, '500 \u00b5m', size=6.0, fill='#ffffff')
    return s, h

#: each fluoroscopy column carries its own letter at the top-left of the band
FLU_MASK = (0.0, 0.0, 10.5, 12.0)

def flu(letter, band, lab, frac=0.235, mask=FLU_MASK):
    x0,x1 = COL[letter]; return ('crop', FL, frac, lab, (x0, band[0], x1, band[1]), mask)

# model
# A and B come from .ai files whose SVG conversion composites embedded images
# through feColorMatrix filters and luminance masks. Chromium renders that fine,
# Illustrator does not: it drops the filtered content and the panels open garbled.
# They are therefore placed as 600 dpi rasters rendered straight from the PDFs.
row([('png','airaster/panel_task.png',0.30,'a'),
     ('png','airaster/panel_hist.png',0.15,'b'),
     ('key', thal_panel, 0.505,'c')])
# swallowing
# D placed as a 600 dpi raster from the PDF: the .ai -> SVG conversion composites these frames
# through feColorMatrix masks that Illustrator drops (they opened blank)
row([('png','airaster/panel_fluoro.png',1.0,'d')])
row([flu('A',R1,'e'), flu('B',R1,'f'), flu('C',R1,'g'), flu('D',R1,'h')])
row([flu('C',R2,'i'), flu('E',R2,'j'),
     ('key', swallow_key, 0.50, 'key')])
# licking: hit rate, then aim, then trajectories (your order)
row([('png',P+'hit_rate_by_spout_single.png',0.62,'k')])
row([('png',P+'first_lick_aim_by_spout_single.png',0.62,'l'),
     ('svg','K08_2026_grant_panels/cam_frames.svg',0.355,'m')])
# M: three-dimensional lick paths, front view, as 2 x 3 (near on top, far
# below). Each spout's axes box is cut from the deck raster (boxes measured from
# the 1495 x 445 file: x pairs below, plot y 44-389) and placed compressed
# vertically; titles, ticks, numbers and the key are drawn here at grant size.
import base64 as _b64, io as _io
from PIL import Image as _Im
_src = _Im.open('deck2/f1_paths3d_front.png').convert('RGB')
_BOX = [(73, 292), (309, 528), (545, 764), (781, 1000), (1017, 1236), (1253, 1472)]
_Y0, _Y1 = 44, 390
_CW, _CH, _GX, _GYc = 92.0, 66.0, 12.0, 24.0
_XL = X0 + 34.0                                 # room for the y numbers and label
_sn = '<g id="Panel_n">' + txt(X0 - 10.0, y + 10.0, 'N', size=11, anchor='start', weight='bold')
_sn += '\n' + ttl(X0, y, 'n')
_top = y + TG + 9.0                             # column headers sit above the first row
for _c, _h in enumerate(('ipsi', 'mid', 'contra')):
    _sn += '\n' + txt(_XL + _c*(_CW+_GX) + _CW/2, _top - 2.0, _h, size=7.4, weight='bold')
for _r, _ring in enumerate(('near', 'far')):
    _cy = _top + _r*(_CH + _GYc)
    _sn += '\n' + txt(X0 + 2.0, _cy + _CH/2, _ring, size=7.4, weight='bold', rot=-90)
    for _c in range(3):
        _a, _b = _BOX[_r*3 + _c]
        _cell = _src.crop((_a - 1, _Y0 - 1, _b + 2, _Y1 + 1))
        _buf = _io.BytesIO(); _cell.save(_buf, 'PNG')
        _cx = _XL + _c*(_CW+_GX)
        _sn += ('\n<image x="%.2f" y="%.2f" width="%.2f" height="%.2f" preserveAspectRatio="none" '
                'xlink:href="data:image/png;base64,%s"/>' % (_cx, _cy, _CW, _CH,
                _b64.b64encode(_buf.getvalue()).decode('ascii')))
        _ky = _CH/(_Y1 - _Y0); _kx = _CW/(_b - _a + 3)
        if _r == 0 and _c == 0:   # the deck's own tiny key sits inside this box
            _sn += '\n' + cover(_cx + (76 - _a + 1)*_kx, _cy + (296 - _Y0 + 1)*_ky, 132*_kx, 92*_ky)
        # x ticks 1, 0, -1 at +40, +107, +174 px within each box
        for _off, _v in ((40, '1'), (107, '0'), (174, '\u22121')):
            _tx = _cx + (_off + 1)*_kx
            _sn += ('\n<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#000" stroke-width="0.6"/>'
                    % (_tx, _cy + _CH, _tx, _cy + _CH + 2.2))
            _sn += '\n' + txt(_tx, _cy + _CH + 8.6, _v, size=6.6)
        if _c == 0:
            for _row, _v in ((63, '0'), (197, '\u22122'), (331, '\u22124')):
                _ty = _cy + (_row - _Y0 + 1)*_ky
                _sn += ('\n<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#000" stroke-width="0.6"/>'
                        % (_cx - 2.2, _ty, _cx, _ty))
                _sn += '\n' + txt(_cx - 3.2, _ty + 2.2, _v, size=6.6, anchor='end')
        if _r == 1:
            _sn += '\n' + txt(_cx + 1.0, _cy + _CH + 16.5, 'contra', size=6.3, anchor='start')
            _sn += '\n' + txt(_cx + _CW - 1.0, _cy + _CH + 16.5, 'ipsi', size=6.3, anchor='end')
_bot = _top + 2*_CH + _GYc
_sn += '\n' + txt(_XL + (3*_CW + 2*_GX)/2, _bot + 25.0, 'side (mm; + = contralesional)', size=6.8)
_sn += '\n' + txt(X0 + 14.0, _top + _CH + _GYc/2, 'down (mm)', size=6.8, rot=-90)
# key to the right of the grid
_lx = _XL + 3*_CW + 2*_GX + 16.0
for _i, (_nm, _col) in enumerate((('pre', '#222222'), ('acute', '#e0218a'),
                                   ('subacute', '#d4a017'), ('chronic', '#1b7f79'))):
    _ly = _top + 10.0 + _i*11.0
    _sn += ('\n<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="1.4"/>'
            % (_lx, _ly, _lx + 8.0, _ly, _col))
    _sn += '\n' + txt(_lx + 11.0, _ly + 2.4, _nm, size=6.8, anchor='start')
_sn += '\n</g>'
_hn = _bot + 28.0 - y
parts.append(_sn); y += _hn + GY

CAP = ("Ventrolateral striatal stroke produces a graded, position-dependent orofacial deficit that "
 "only partly recovers. "
 "(A) Six-spout licking task and trial structure. "
 "(B) VLS infarct at day 7 (DAPI, NeuN, GFAP). "
 "(C) Ipsilesional thalamus in a severe animal from an earlier cohort, with and without GFAP; Paxinos outlines overlaid. Gliosis spans VL, VM, mediodorsal (MD) and centromedian (CM) nuclei, with a dense focus at the reticular nucleus (Rt) border. "
 "(D) Videofluoroscopy frames: oral cavity, pharynx, oesophagus and a swallow. "
 "(E-J) Swallowing measures before and after stroke, by severity (key at right): swallows per minute, swallow "
 "interval, licks per swallow, licks per minute, barium amount and oesophageal transit. "
 "(K) Hit rate by spout, change versus each animal's own pre; it collapses acutely at the far "
 "contralesional spout and returns to pre by chronic. "
 "(L) First-lick aim offset by spout: the deficit is graded across spouts, largest at the far "
 "contralesional spout, and still present at chronic. "
 "(M) The two camera views at one timepoint, with the tracked tongue, jaw, spout and nose. "
 "(N) Three-dimensional tongue and jaw paths by spout and epoch, near spouts above and far below (pre in black). "
 "In K and L, faded points are sessions and filled markers the paired nested bootstrap mean with "
 "its 95 per cent interval; asterisks mark intervals excluding zero after Bonferroni correction. "
 "Pre is drawn in L as the reference and is the zero line in K.")
import os
if not os.environ.get('NOCAP'):
    s,h = caption(X0, y+10, FIG_W, CAP, bold_lead='Figure 3.')
    parts.append(s); y += h + 16

H=y
svg=('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
 'xmlns:xlink="http://www.w3.org/1999/xlink" width="%.1fpt" height="%.1fpt" viewBox="0 0 %.1f %.1f">\n'
 '<rect width="100%%" height="100%%" fill="#fff"/>\n<g id="Figure2_behaviour">\n%s\n</g>\n</svg>\n'
 % (PAGE_W,H,PAGE_W,H,'\n'.join(parts)))
open(('Figure3_behavior_DRAFT4_panels' if os.environ.get('NOCAP') else 'Figure3_behavior_DRAFT4')+'.svg','w',encoding='utf-8').write(svg)
print('wrote %.0f x %.0f pt (%.1f in tall), %.1f KB' % (PAGE_W,H,H/72,len(svg)/1024))
