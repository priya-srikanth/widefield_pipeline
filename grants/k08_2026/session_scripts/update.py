import re, html, sys, subprocess, docx
from PIL import ImageFont
from docx.oxml.ns import qn
import capio
R=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',1000)
Bf=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',1000)
def W(t,size,bold=False): return (Bf if bold else R).getlength(t)/1000*size
d=docx.Document('../rebase/K08_Consolidated_Aims_and_Research_Strategy_v18.docx'); caps={}
for p in d.element.body.iter(qn('w:p')):
    if p.find('.//'+qn('w:p')) is not None: continue
    t=''.join(x.text or '' for x in p.iter(qn('w:t')))
    for n in '345':
        if t.startswith('Figure %s.'%n): caps[n]=re.sub(r'\s+',' ',t).strip()
def fsize(s,cls):
    st=s[:s.index('</style>')]; v=None
    for m in re.finditer(r'([^{}]*)\{([^}]*)\}',st):
        if re.search(r'\.'+cls+r'(?![\d])',m.group(1)):
            f=re.search(r'font-size:\s*([\d.]+)',m.group(2))
            if f: v=float(f.group(1))
    return v
def wrap(words,limit_first,limit,size):
    out=[];cur='';lim=limit_first
    for w in words:
        t=(cur+' '+w).strip()
        if W(t,size)>lim and cur: out.append(cur); cur=w; lim=limit
        else: cur=t
    out.append(cur); return out
esc=lambda t: html.escape(t,quote=False).replace("'",'&apos;')
def fix_images(s,pdf):
    L=subprocess.run(['pdfimages','-list',pdf],capture_output=True,text=True).stdout.splitlines()[2:]
    ppi={}
    for l in L:
        f=l.split()
        if len(f)>13 and f[2]=='image': ppi[(int(f[3]),int(f[4]))]=(float(f[12]),float(f[13]))
    def rep(m):
        tag=m.group(0); w=int(float(re.search(r'width="([\d.]+)"',tag).group(1))); h=int(float(re.search(r'height="([\d.]+)"',tag).group(1)))
        if (w,h) not in ppi: return tag
        sc=re.search(r'scale\(([-\d.]+)(?:\s+([-\d.]+))?\)',tag)
        if not sc: return tag
        sx=float(sc.group(1)); sy=float(sc.group(2)) if sc.group(2) else sx
        nx=72/ppi[(w,h)][0]*(1 if sx>=0 else -1); ny=72/ppi[(w,h)][1]*(1 if sy>=0 else -1)
        return tag.replace(sc.group(0),'scale(%.4f %.4f)'%(nx,ny))
    return re.sub(r'<image[^>]*>',rep,s)
LIM={'3':360.0}
def update(n,src,dst,pdf=None):
    s=open(src,encoding='utf-8').read()
    if pdf: s=fix_images(s,pdf)
    i,j,kind=capio.find(s,n); seg=s[i:j]; cap=caps[n]
    if kind=='text':
        cls=re.search(r'class="([^"]+)"',seg).group(1); size=fsize(s,cls)
        x0,y0=re.search(r'translate\(([\d.]+) ([\d.]+)\)',seg).groups()
        lines=[html.unescape(t) for t in re.findall(r'<tspan[^>]*>(.*?)</tspan>',seg,flags=re.S)]
        lh=float(re.findall(r'y="([\d.]+)"',seg)[1])
        lim=LIM.get(n) or max(W(l.rstrip(),size) for l in lines)+1
        out=wrap(cap.split(' '),lim,lim,size)
        new=f'<text class="{cls}" style="font-family: Arial, ArialMT" transform="translate({x0} {y0})">'+''.join(f'<tspan x="0" y="{k*lh:g}">{esc(l)}{" " if k<len(out)-1 else ""}</tspan>' for k,l in enumerate(out))+'</text>'
        yend=float(y0)+lh*(len(out)-1); old_n=len(lines)
    else:
        texts=re.findall(r'<text class="([^"]+)" transform="translate\(([\d.]+) ([\d.]+)\)"><tspan[^>]*>(.*?)</tspan></text>',seg)
        hc,hx,hy,ht=texts[0]; bc,bx,by,bt=texts[1]; rest=texts[2:]
        lc=rest[0][0]; lx=float(rest[0][1]); lh=float(rest[1][2])-float(rest[0][2])
        size=fsize(s,lc); hsize=fsize(s,hc)
        lim=max(W(html.unescape(t[3]),size) for t in rest[:-1])+1
        head=html.unescape(ht).strip()+' '; body=cap[len(head.strip()):].strip()
        out=wrap(body.split(' '),lim-W(head,hsize,True),lim,size)
        y0=float(hy); hxw=float(hx)+W(head,hsize,True)
        g=['<g id="CAPTION">','    <g>',f'      <text class="{hc}" transform="translate({hx} {hy})"><tspan x="0" y="0">{esc(head)}</tspan></text>',
           f'      <text class="{bc}" transform="translate({hxw:.2f} {hy})"><tspan x="0" y="0">{esc(out[0])}</tspan></text>','    </g>']
        for k,l in enumerate(out[1:],1): g.append(f'    <text class="{lc}" transform="translate({lx:g} {y0+k*lh:.2f})"><tspan x="0" y="0">{esc(l)}</tspan></text>')
        g.append('  </g>'); new='\n'.join(g); yend=y0+lh*(len(out)-1); old_n=len(texts)-1
    s=s[:i]+new+s[j:]
    vb=re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"',s); H=float(vb.group(2))
    if yend+3>H: s=s.replace(vb.group(0),'viewBox="0 0 %s %.1f"'%(vb.group(1),yend+4),1)
    open(dst,'w',encoding='utf-8').write(s)
    print(n,kind,'lines',old_n,'->',len(out),'yend',round(yend,1),'H',H)
update('3','f3.svg','Figure3_behavior_relettered_v2_legend.svg','f3.pdf')
update('4','f4.svg','Figure4_facerhythm_Dsplit_light_legend.svg','f4.pdf')
update('5','f5.svg','Figure5_decoder_encoder_cuedD_legend.svg')
