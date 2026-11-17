import re, html
from PIL import ImageFont
R=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',760)
B=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',760)
w=lambda t,f=R: f.getlength(t)/100.0
f='Figure5_cuedD.svg'; s=open(f,encoding='utf-8').read()
i=s.index('<g id="CAPTION">'); j=s.index('\n  </g>',i)+len('\n  </g>')
old=s[i:j]
lines=[html.unescape(re.sub('<[^>]+>','',m)) for m in re.findall(r'<tspan[^>]*>.*?</tspan>',old)]
maxw=max(w(l) for l in lines[2:]); y0=float(re.search(r'translate\(13 ([\d.]+)\)',old).group(1)); print('maxw',maxw,'y0',y0,len(lines))
HEAD='Figure 5. '
CAP=("Striatal stroke leaves a motor code that is reorganised rather than restored, with the activity change loaded on contralesional orofacial "
"sensorimotor cortex. (A) Accuracy of a decoder frozen before stroke (left) and of one refit within each post-stroke session (right), on LocaNMF "
"components (August pre baseline; dashed line, chance). (B) Refit minus frozen gap: position information is present after stroke but displaced, "
"and the gap persists through recovery. In A and B, asterisks mark Bonferroni-corrected paired bootstrap vs pre. (C) Lateral offset between the "
"decoded and executed first cued lick, over all spouts and at the far contralesional spout. (D) Response around the first cued lick in tjM1 and "
"SSp-m of each hemisphere, by epoch (pre in black; mean ± SEM over sessions); grey, the 0 to 1 s window used in E and F; bars above, time bins "
"that differ from pre (paired bootstrap, FDR q < 0.05). (E) Lick-response AUC as a fold of each animal's own pre; activity rises in every region "
"and stays raised. (F) Contralesional:ipsilesional balance, log2 of the contralesional/ipsilesional lick-response AUC at the far contralesional "
"spout, in tjM1 and SSp-m: the balance shifts contralesionally and does not return. (G) The same balance (SSp-m) against far contralesional aim, "
"one point per session: sessions with a more contralesional balance had a larger aim deficit, within animals and beyond time from stroke (mixed "
"model with log time, p < 10^−9; three of four animals individually). Dotted lines, per-animal fits over post-stroke sessions. (H) Position on "
"the pre-to-chronic axis per session against days from stroke. (I) Each session in the plane of that axis and the orthogonal acute departure; "
"acute sessions are displaced off the recovery axis rather than part-way along it. In H and I, asterisks mark epochs displaced from pre (paired "
"bootstrap over animals, then sessions). (J) Allen CCF areas for reference in K: ALM, anterolateral motor cortex; tjM1, tongue-jaw motor cortex; "
"SSp-m, SSp-n, SSp-ul, SSp-ll and SSp-tr, primary somatosensory cortex of the mouth, nose, upper limb, lower limb and trunk. Left is the "
"ipsilesional hemisphere (I) and right the contralesional (C) throughout. (K) Regional activity change along the pre-stroke to chronic recovery "
"axis: lick-aligned pre-stroke response (units of pre RMS) and chronic minus pre, for the mean over spouts and the far contralesional spout, with "
"lick kinematics, spout contact and facial movement regressed out. Left, the full change; right, the same after removing the cortex-wide mode, "
"which alone carries most of the mean response. Contours, significant bins (nested bootstrap, max-statistic); white, tissue not analysed. A "
"structured redistribution remains once the global mode is removed, and it is largest in tjM1 and SSp-m of both hemispheres. Faded or open "
"markers are sessions, shape by animal; filled markers are means with their interval.")
words=CAP.split(' '); out=[]; cur=''; first=True
for wd in words:
    t=(cur+' '+wd).strip(); lim=maxw-(w(HEAD,B) if first else 0)
    if w(t)>lim and cur: out.append(cur); cur=wd; first=False
    else: cur=t
out.append(cur)
lh=9.4; e=lambda t: html.escape(t,quote=False).replace("'",'&apos;')
g=['<g id="CAPTION">','    <g class="st32">',
 f'      <text class="st39" transform="translate(13 {y0})"><tspan x="0" y="0">{HEAD}</tspan></text>',
 f'      <text class="st111" transform="translate({13+w(HEAD,B):.1f} {y0})"><tspan x="0" y="0">{e(out[0])}</tspan></text>','    </g>']
for k,l in enumerate(out[1:],1): g.append(f'    <text class="st29" transform="translate(13 {y0+k*lh:.1f})"><tspan x="0" y="0">{e(l)}</tspan></text>')
g.append('  </g>')
s=s[:i]+'\n'.join(g)+s[j:]
open(f,'w',encoding='utf-8').write(s); print(len(out),'lines; last y',y0+(len(out)-1)*lh)
