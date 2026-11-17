import re
D='/tmp/claude-0/-home-claude/b93e55df-fbd7-56c6-87ea-844c1ae2344e/scratchpad/'
s=open(D+'f3new/Figure3_behavior.svg',encoding='utf-8').read()
L=[('st61','389.3 178.3','E','D'),('st61','505.2 178.3','F','E'),('st61','612.4 178.3','G','F'),
   ('st61','387.1 295.9','H','G'),('st61','505.9 295','I','H'),('st61','613.6 296.4','J','I'),
   ('st61','389 413.6','K','J'),('st61','485.5 412.7','L','K'),('st61','388.6 498.2','M','L'),
   ('st99','667.7 452.5) scale(1.1 1','N','M'),('st61','392.5 623.4','O','N'),('st61','393.6 759.2','P','O')]
for c,t,o,n in L:
    old=f'<text class="{c}" transform="translate({t})"><tspan x="0" y="0">{o}</tspan></text>'
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,old.replace(f'>{o}<',f'>{n}<'))
# aim panel raster with ipsi-labelled y axis
s=s.replace('xlink:href="Figure3_behavior-8.png"','xlink:href="Figure3_behavior-8_ipsilabel.png"')
# replace old paths content (after her title, up to the caption)
t='<text class="st73" transform="translate(493.1 757.2)"><tspan x="0" y="0">3-D tongue and jaw paths, front view</tspan></text>'
i=s.index(t)+len(t); j=s.index('<g id="CAPTION">')
p=open(D+'paths/Fig3O_paths_narrow.svg',encoding='utf-8').read()
p=p[p.index('>',p.index('<svg'))+1:p.rindex('</svg>')]
p=re.sub(r'<metadata>.*?</metadata>','',p,flags=re.S)
p=re.sub(r'<defs>\s*<style[^<]*</style>\s*</defs>','',p)
p=re.sub(r'id="([^"]+)"',r'id="O3d_\1"',p)
p=re.sub(r'href="#([^"]+)"',r'href="#O3d_\1"',p)
p=re.sub(r'url\(#([^)]+)\)',r'url(#O3d_\1)',p)
p=re.sub(r"font-family:\s*[^;\"]+","font-family: Arial",p)
p=p.replace('style="fill: #ffffff"/>','style="fill: none"/>',1)  # transparent background
s=s[:i]+'\n  <g id="Panel_O_paths3d" transform="translate(386 761)">'+p+'</g>\n  '+s[j:]
open(D+'f3new/Figure3_behavior_relettered.svg','w',encoding='utf-8').write(s)
print('ok',len(s))
