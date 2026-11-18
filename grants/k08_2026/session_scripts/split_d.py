from lxml import etree
import re, copy
NS='{http://www.w3.org/2000/svg}'
P=etree.XMLParser(huge_tree=True, remove_blank_text=False)
t=etree.parse('Figure4_facerhythm.svg',P); root=t.getroot()
byid={e.get('id'):e for e in root.iter() if e.get('id')}
pd=byid['Panel_d']; fig=byid['d_figure_1']
clipg=fig.getparent()                      # g.st46640 (clip-path)
def txt(e): return ''.join(e.itertext())
# drop background, colorbar, suptitle
for k in ['d_patch_1','d_axes_7','d_text_13']:
    e=byid[k]; e.getparent().remove(e)
# drop per-map source titles (lick_right -- acute ...)
n=0
for ax in [byid['d_axes_%d'%i] for i in range(1,7)]:
    for tx in list(ax.iter(NS+'text')):
        if '--' in txt(tx):
            g=tx.getparent(); (g.getparent() if g.get('id','').startswith('d_text') else g).remove(g if g.get('id','').startswith('d_text') else tx); n+=1
    for p in ax.iter(NS+'path'):            # any white axes background -> no fill
        if p.get('class')=='st590': p.set('style','fill:none'); 
print('titles removed',n)
# white cover rects at top level near D
for r in list(root.findall(NS+'rect')):
    if r.get('class')=='st590': root.remove(r); print('removed rect',r.get('x'),r.get('y'))
# her loose labels for D (top level text): row labels + contra/ipsi rows
def tr(e):
    m=re.search(r'translate\(([-\d.]+) ([-\d.]+)\)',e.get('transform','')); return (float(m.group(1)),float(m.group(2))) if m else None
row1=[];row2=[]
for e in list(root):
    if e.tag!=NS+'text': continue
    x=txt(e); p=tr(e)
    if not p: continue
    if x in('contra','ipsi') and abs(p[1]-562.2)<0.5: row1.append(e)
    elif x in('contra','ipsi') and abs(p[1]-623.9)<0.5: row2.append(e)
    elif x=='Lick contra' and abs(p[1]-551.7)<0.5: row1.append(e)
    elif x=='Lick ipsi' and abs(p[1]-610.7)<0.5: row2.append(e)
print(len(row1),len(row2))
# build row groups (no clip) inside Panel_d
g1=etree.SubElement(pd,NS+'g',id='D_row1_contra_lick'); g2=etree.SubElement(pd,NS+'g',id='D_row2_ipsi_lick')
for i in (1,2,3): g1.append(byid['d_axes_%d'%i])
for i in (4,5,6): g2.append(byid['d_axes_%d'%i])
for e in row1: g1.append(e)
for e in row2: g2.append(e)
clipg.getparent().remove(clipg)            # now empty shell (figure_1 had only these)
print('left in fig:',[c.get('id') for c in fig])
t.write('Figure4_facerhythm_Dsplit.svg',xml_declaration=True,encoding='UTF-8')
