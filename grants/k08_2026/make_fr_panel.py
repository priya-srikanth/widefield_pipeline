import sys; sys.path.insert(0,'.')
from build_fig import place_png, caption

PT=72.0; PAGE_W=612.0; FIG_W=7.4*PT; X0=(PAGE_W-FIG_W)/2; Y0=44.0
parts=[]; y=Y0

s,h = place_png('fr/fr_position_unique_4col.png','a',X0,y,FIG_W*0.76,label='a')
parts.append(s); y += h + 14

CAP = ("Face movement accounts for most of the chronic rise in position coding, but a focal "
 "contralesional increase survives. Per-session encoding models were fit to widefield activity in "
 "the lick window, with and without FaceRhythm factors describing uninstructed facial movement. "
 "Maps show held-out R2 (fraction) per 2x2-pixel block, within animal then averaged over four "
 "animals; image left is ipsilesional. Row 1, pre-stroke level; rows 2 and 3, epoch minus pre. "
 "Unique variance is the full model minus the model without that group. The full model's R2 rises "
 "chronically rather than falling (first column), so the changes in unique variance are not a "
 "denominator effect. FaceRhythm occupies its own broad territory before stroke (second column). "
 "Chronically, position-unique variance rises broadly and bilaterally when FaceRhythm is left out "
 "of the model (third column), and that broad rise is largely absent once FaceRhythm is included "
 "(fourth column); what remains is a focal contralesional anterior-lateral increase. In an a "
 "priori area test this focal increase is significant in contralesional SSp-m (cue window, +0.037 "
 "[0.010, 0.067], 4 of 4 animals, Bonferroni-corrected p = .002), with ipsilesional ~0; the acute "
 "loss of position coding in SSp-m survives the same control. PROVISIONAL: the chronic effect is "
 "carried by two of the four animals (PS94, PS95) and exceeds each animal's pre-stroke drift in "
 "those two only.")

s,h = caption(X0, y+8, FIG_W, CAP, bold_lead='Figure X.')
parts.append(s); y += h + 18

H=y
svg=('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
 'xmlns:xlink="http://www.w3.org/1999/xlink" width="%.1fpt" height="%.1fpt" viewBox="0 0 %.1f %.1f">\n'
 '<rect width="100%%" height="100%%" fill="#fff"/>\n<g id="FR_encoding_panel">\n%s\n</g>\n</svg>\n'
 % (PAGE_W,H,PAGE_W,H,'\n'.join(parts)))
open('FR_encoding_panel_DRAFT.svg','w',encoding='utf-8').write(svg)
print('wrote %.0f x %.0f pt' % (PAGE_W,H))
