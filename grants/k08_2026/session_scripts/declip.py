from lxml import etree
import re, collections
NS='{http://www.w3.org/2000/svg}'
t=etree.parse('Figure4_facerhythm_Dsplit.svg',etree.XMLParser(huge_tree=True)); root=t.getroot()
style=root.find('.//'+NS+'style')
css=style.text
cls2clip={}
for m in re.finditer(r'([^{}]+)\{([^}]*)\}',css):
    c=re.search(r'clip-path:\s*url\(#([^)]+)\)',m.group(2))
    if c:
        for sel in re.findall(r'\.(st\d+)',m.group(1)): cls2clip[sel]=c.group(1)
clips={e.get('id'):e for e in root.iter(NS+'clipPath')}
kinds=collections.Counter()
for cp in list(clips.values())[:2000]:
    kinds[tuple(ch.tag.replace(NS,'') for ch in cp)]+=1
print(kinds.most_common(5))
n=0
for g in list(root.iter(NS+'g')):
    c=g.get('class')
    if c in cls2clip and len(g)==1 and g[0].tag==NS+'path' and len(g.attrib)==1:
        p=g.getparent(); i=p.index(g); p.remove(g); p.insert(i,g[0]) if False else None
        ch=g[0]; p.insert(i,ch); n+=1
print('unwrapped',n)
used=set()
s_body=etree.tostring(root).decode()
used_cls=set(re.findall(r'class="([^"]+)"',s_body)); used_cls={c for v in used_cls for c in v.split()}
used_clip={cls2clip[c] for c in used_cls if c in cls2clip}
used_clip|=set(re.findall(r'clip-path="url\(#([^)]+)\)"',s_body))
rm=0
for cid,cp in clips.items():
    if cid not in used_clip: cp.getparent().remove(cp); rm+=1
print('clipPaths removed',rm,'kept',len(clips)-rm)
t.write('Figure4_facerhythm_Dsplit_light.svg',xml_declaration=True,encoding='UTF-8')
