import sys; sys.path.insert(0,'.')
from build_fig import place, place_png

PT=72.0; PAGE_W=612.0; FIG_W=7.4*PT; X0=(PAGE_W-FIG_W)/2; Y0=48.0
GAP_X=12.0; GAP_Y=18.0

def placeholder(x,y,w,h,label,text):
    return ('<g id="Panel_%s">'
      '<text x="%.1f" y="%.1f" font-family="Arial" font-weight="bold" font-size="11">%s</text>'
      '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#f4f4f4" stroke="#999" '
      'stroke-dasharray="4 3"/>'
      '<text x="%.1f" y="%.1f" font-family="Arial" font-size="8.5" fill="#666" '
      'text-anchor="middle">%s</text></g>' % (label, x-10, y+10, label, x, y, w, h,
       x+w/2, y+h/2, text)), h

def build(name, rows, title_note):
    parts=[]; y=Y0
    for row in rows:
        avail = FIG_W - GAP_X*(len(row)-1)
        x = X0; rh = 0
        for item in row:
            kind, src, frac, lab = item[0], item[1], item[2], item[3]
            w = FIG_W*frac
            if kind=='svg':   s,h = place(src, lab, x, y, w, label=lab)
            elif kind=='png': s,h = place_png(src, lab, x, y, w, label=lab)
            else:             s,h = placeholder(x, y, w, item[4], lab, src)
            parts.append(s); rh=max(rh,h); x += w + GAP_X
        y += rh + GAP_Y
    total_h = y + 40
    parts.append('<g id="NOTES"><text x="%.1f" y="%.1f" font-family="Arial" font-size="8" '
                 'fill="#b00">%s</text></g>' % (X0, y+14, title_note))
    svg=('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
      'xmlns:xlink="http://www.w3.org/1999/xlink" width="%.1fpt" height="%.1fpt" '
      'viewBox="0 0 %.1f %.1f">\n<rect width="100%%" height="100%%" fill="#fff"/>\n'
      '<g id="%s">\n%s\n</g>\n</svg>\n' % (PAGE_W, total_h, PAGE_W, total_h, name, '\n'.join(parts)))
    open(name+'.svg','w',encoding='utf-8').write(svg)
    print('%s  %.0f x %.0f pt   %.1f KB' % (name, PAGE_W, total_h, len(svg)/1024))

FIG1=[
 [('ph','VLS stroke histology\n(reuse existing grant panel)',0.34,'a',150),
  ('svg','fluoro/swallow_frames.svg',0.64,'b')],
 [('svg','fluoro/per_swallow_distributions_panel_inter_swallow_interval.svg',0.32,'c'),
  ('svg','fluoro/per_swallow_distributions_panel_licks_per_interval.svg',0.32,'d'),
  ('svg','fluoro/per_swallow_distributions_panel_swallow_duration.svg',0.32,'e')],
 [('svg','fluoro/sitting_transit.svg',0.60,'f'),
  ('svg','fluoro/bolus_density.svg',0.36,'g')],
 [('ph','Six-spout task schematic\n(reuse existing grant panel)',0.30,'h',120),
  ('png','deck/f1_hitrate_by_spout.png',0.66,'i')],
 [('png','deck2/f1_camframe.png',0.46,'j')],
 [('png','deck2/f1_firstlick_aim_noweighted.png',1.0,'k')],
 [('png','deck/f1_3d_paths.png',0.74,'l')],
 [('png','deck/f1_farcontra_trial_outcomes.png',0.52,'m')],
]

FIG3=[
 [('png','deck/f3_FR_master_face.png',0.30,'a'),
  ('png','deck/f3_FR_12_factors.png',0.66,'b')],
 [('png','deck/f3_FR_lickright_lickleft.png',0.44,'c'),
  ('png','deck/f3_FR_loading_change_maps.png',0.52,'d')],
 [('png','deck2/slide95_image90.png',0.60,'e')],
 [('png','deck2/f3_cued_ROI_row.png',1.0,'f')],
 [('png','deck2/slide96_image92.png',1.0,'g')],
]

build('Figure1_model_and_deficit_DRAFT', FIG1,
 'DRAFT, everything in. a and h are placeholders for the two existing grant panels. Pare down before use.')
build('Figure3_facerhythm_encoder_DRAFT', FIG3,
 'DRAFT, everything in. e needs the pre-stroke residual level added; f should subset to tjM1 and SSp-m R/L. Pare down before use.')
