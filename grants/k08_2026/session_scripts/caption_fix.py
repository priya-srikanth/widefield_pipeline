import re
from PIL import ImageFont
R=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',760)
B=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',760)
w=lambda t,f=R: f.getlength(t)/100.0
f='Figure3_behavior_relettered.svg'; s=open(f,encoding='utf-8').read()
i=s.index('<g id="CAPTION">'); j=s.index('</g>\n  <g>',i)+4
old=s[i:j]
lines=[re.sub('<[^>]+>','',m) for m in re.findall(r'<tspan[^>]*>.*?</tspan>',old)]
import html
maxw=max(w(html.unescape(l)) for l in lines[2:]); print('max line', maxw)
HEAD='Figure 3. '
CAP=('Ventrolateral striatal stroke produces a graded, position-dependent orofacial deficit that only partly recovers. '
 '(A) VLS infarct (*) at day 7 (DAPI, NeuN, GFAP). (B) Ipsilesional thalamus in a severe animal from an earlier cohort, '
 'with (left) and without (right) GFAP; Paxinos outlines overlaid. Arrow, area of thalamic gliosis and neuronal loss. '
 '(C) Videofluoroscopy frames of a swallow, with oral cavity, pharynx and esophagus outlined; the barium bolus is '
 'highlighted in red. (D-I) Swallowing before and after stroke, by severity (separate cohorts; key above): swallow rate, '
 'swallow interval, licks per swallow, lick rate, bolus amount and esophageal transit. (J) Six-position task. '
 '(K) Trial structure. (L) Spout-contact hit rate by spout, change versus each animal’s own pre. (M) Front and bottom '
 'camera views with tracked tongue, jaw, spout and nose. (N) First-lick aim by spout, change versus own pre; negative '
 'values are ipsilesional deviation. (O) Three-dimensional tongue (solid) and jaw (dashed) paths, front view, near spouts '
 'above and far below; markers, bins where the tongue path differs from pre (cluster permutation). In L and N, faded '
 'points are sessions and filled markers the paired nested bootstrap mean with its 95% interval; asterisks, intervals '
 'excluding zero after Bonferroni correction.')
words=CAP.split(' '); out=[]; cur=''; first=True
for wd in words:
    t=(cur+' '+wd).strip(); lim=maxw-(w(HEAD,B) if first else 0)
    if w(t)>lim and cur: out.append(cur); cur=wd; first=False
    else: cur=t
out.append(cur)
x0=394.1309; lh=9.4; ylast=1290.5725; y0=ylast-lh*(len(out)-1)
e=lambda t: html.escape(t,quote=False).replace("'",'&apos;')
g=['<g id="CAPTION">','    <g class="st80">',
 f'      <text class="st87" transform="translate({x0} {y0:.4f})"><tspan x="0" y="0">{HEAD}</tspan></text>',
 f'      <text class="st169" transform="translate({x0+w(HEAD,B):.4f} {y0:.4f})"><tspan x="0" y="0">{e(out[0])}</tspan></text>','    </g>']
for k,l in enumerate(out[1:],1):
    g.append(f'    <text class="st78" transform="translate({x0} {y0+k*lh:.4f})"><tspan x="0" y="0">{e(l)}</tspan></text>')
g.append('  </g>')
s=s[:i]+'\n'.join(g)+s[j:]
open(f,'w',encoding='utf-8').write(s)
print(len(out),'lines, top y',y0); print('\n'.join(out))
