import re
D='/tmp/claude-0/-home-claude/b93e55df-fbd7-56c6-87ea-844c1ae2344e/scratchpad/'
s=open(D+'f3new/hers_v2.svg',encoding='utf-8').read()
i=s.index('<g id="Panel_O_paths3d">'); k=i; depth=0
for m in re.finditer(r'<g\b|</g>',s[i:]):
    depth+= 1 if m.group()!='</g>' else -1
    if depth==0: k=i+m.end(); break
p=open(D+'paths/Fig3O_paths_short.svg',encoding='utf-8').read()
p=p[p.index('>',p.index('<svg'))+1:p.rindex('</svg>')]
p=re.sub(r'<metadata>.*?</metadata>','',p,flags=re.S)
p=re.sub(r'<defs>\s*<style[^<]*</style>\s*</defs>','',p)
p=re.sub(r'id="([^"]+)"',r'id="O3d_\1"',p); p=re.sub(r'href="#([^"]+)"',r'href="#O3d_\1"',p); p=re.sub(r'url\(#([^)]+)\)',r'url(#O3d_\1)',p)
p=re.sub(r"font-family:\s*[^;\"]+","font-family: Arial",p)
p=p.replace('style="fill: #ffffff"/>','style="fill: none"/>',1)
s=s[:i]+'<g id="Panel_O_paths3d" transform="translate(388.2 761.8)">'+p+'</g>'+s[k:]
fix={'Figure3_behavior-1.png':'.375','Figure3_behavior-2.png':'.1644','Figure3_behavior-3.png':'.1644','Figure3_behavior-4.png':'.1644',
'Figure3_behavior-5.png':'.1644','Figure3_behavior-6.png':'.1001','Figure3_behavior-7.png':'.0778','Figure3_behavior-8_ipsilabel.png':'.381',
'Figure3_behavior-9.png':'.3618 -.3318','Figure3_behavior-10.png':'.3618 -.3318','Figure3_behavior-17.png':'.2075'}
for h,sc in fix.items():
    pat=re.compile(r'(<image[^>]*transform="translate\([^)]*\) )scale\([^)]*\)("[^>]*xlink:href="'+re.escape(h)+'")')
    s,n=pat.subn(r'\g<1>scale('+sc+r')\2',s); assert n==1,(h,n)
open(D+'f3new/Figure3_behavior_relettered_v2.svg','w',encoding='utf-8').write(s); print('ok')
