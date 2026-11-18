import sys, re, pathlib
from playwright.sync_api import sync_playwright
src = pathlib.Path(sys.argv[1]).resolve()
s = src.read_text(encoding='utf-8')
w = float(re.search(r'width="([\d.]+)pt"', s).group(1))
h = float(re.search(r'height="([\d.]+)pt"', s).group(1))
html = ('<!doctype html><meta charset="utf-8">'
        '<style>html,body{margin:0;padding:0}svg{display:block}</style>' + s[s.index('<svg'):])
tmp = src.with_suffix('.render.html'); tmp.write_text(html, encoding='utf-8')
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto(tmp.as_uri()); pg.wait_for_timeout(1200)
    pg.pdf(path=str(src.with_suffix('.pdf')), width='%.3fin'%(w/72.0), height='%.3fin'%(h/72.0),
           print_background=True, margin={'top':'0','bottom':'0','left':'0','right':'0'})
    pg.screenshot(path=str(src.with_suffix('.png')), full_page=True)
    b.close()
tmp.unlink()
print('wrote', src.with_suffix('.pdf').name)
