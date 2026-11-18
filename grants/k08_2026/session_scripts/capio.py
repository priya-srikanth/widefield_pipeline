import re,html
def find(s,n):
    if '<g id="CAPTION"' in s:
        i=s.index('<g id="CAPTION"'); depth=0
        for m in re.finditer(r'<g\b|</g>',s[i:]):
            depth+=1 if m.group()!='</g>' else -1
            if depth==0: return i,i+m.end(),'group'
    i=s.index('>Figure %s. '%n); i=s.rindex('<text',0,i); j=s.index('</text>',i)+7
    return i,j,'text'
def text(s,n):
    i,j,k=find(s,n); seg=s[i:j]
    parts=[html.unescape(re.sub('<[^>]+>','',t)) for t in re.findall(r'<tspan[^>]*>(.*?)</tspan>',seg,flags=re.S)]
    return re.sub(r'\s+',' ',' '.join(parts)).strip()
