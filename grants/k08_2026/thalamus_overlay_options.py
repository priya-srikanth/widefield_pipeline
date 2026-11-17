import re, base64, io, sys
from PIL import Image
from playwright.sync_api import sync_playwright
U = '/mnt/user-data/uploads/Sabatini Lab/Manuscripts/VLS stroke methods paper/histology_panels/PS46-55_sections/'
KEEP = {'VM', 'VL', 'MD', 'MDL', 'MDM', 'MDC', 'CM', 'PF', 'Pf', 'Rt', 'VPM', 'VPL', 'PC', 'CL', 'Po', 'IMD', 'Re', 'Rh', 'ZI', 'VA'}
HI = {'VM', 'CM', 'PF', 'MD', 'MDL', 'MDM', 'MDC', 'Rt'}
def plate(fig):
    s = open(U + 'plates/plate_zoom_fig%d.svg' % fig).read()
    lines = re.findall(r'<polyline[^>]*/>', s) + re.findall(r'<path[^>]*/>', s) + re.findall(r'<polygon[^>]*/>', s)
    texts = re.findall(r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*>([^<]+)</text>', s)
    return lines, texts
def panel(png, fig, out_w=900):
    im = Image.open(U + png).convert('RGB'); W, H = im.size
    k = out_w / W; im = im.resize((out_w, int(H * k)), Image.LANCZOS); W, H = im.size
    b = io.BytesIO(); im.save(b, 'JPEG', quality=90); b64 = base64.b64encode(b.getvalue()).decode()
    sc = W / 5.8
    lines, texts = plate(fig)
    g = ['<g transform="translate(%.2f,%.2f) scale(%.4f)">' % (2.9 * sc, -3.187 * sc, sc)]
    for l in lines:
        l = re.sub(r'stroke="[^"]*"', '', l); l = re.sub(r'stroke-width="[^"]*"', '', l)
        g.append(l.replace('/>', ' fill="none" stroke="#ffffff" stroke-opacity="0.75" stroke-width="0.010"/>'))
    for x, y, t in texts:
        if t in KEEP:
            col = '#ffe14d' if t in HI else '#ffffff'
            g.append('<text x="%s" y="%s" font-family="Arial" font-weight="bold" font-size="0.13" fill="%s" '
                     'text-anchor="middle" stroke="#000" stroke-width="0.012" paint-order="stroke">%s</text>' % (x, y, col, t))
    g.append('</g>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d"><image width="%d" height="%d" '
            'href="data:image/jpeg;base64,%s"/>%s</svg>' % (W, H, W, H, b64, '\n'.join(g))), W, H
def sheet(name, items, title):
    cells = []
    for lab, png, fig in items:
        sv, W, H = panel(png, fig)
        cells.append('<div><div class="t">%s</div>%s</div>' % (lab, sv))
    html = ('<html><head><style>body{margin:8px;font-family:Arial;background:#fff}.r{display:flex;gap:10px}'
            '.t{font-size:15px;font-weight:bold;margin:2px 0}h3{margin:2px 0 6px}</style></head><body><h3>%s</h3>'
            '<div class="r">%s</div></body></html>' % (title, ''.join(cells)))
    open(name + '.html', 'w').write(html)
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1880, 'height': 900}); pg.goto('file://' + __import__('os').path.abspath(name + '.html'))
        pg.wait_for_timeout(500); pg.screenshot(path=name + '.png', full_page=True); b.close()
OPT = [
 ('optA_PS55_e100', 'Option A: PS55 (severe), Bregma -1.22 mm', 41, 'Sev2_PS55_thal_PS55_e100'),
 ('optB_PS55_e96',  'Option B: PS55 (severe), Bregma -1.34 mm', 42, 'Sev2_PS55_thal_PS55_e96'),
 ('optC_PS54_e226', 'Option C: PS54 (severe), Bregma -1.70 mm', 45, 'Sev1_PS54_thal_PS54_e226'),
 ('optD_PS51_e146', 'Option D: PS51 (moderate-severe), Bregma -1.70 mm', 45, 'Mod2_PS51_thal_PS51_e146')]
for name, title, fig, stem in OPT:
    sheet(name, [('DAPI / NeuN / GFAP', stem + '_thalamus_zoom.png', fig), ('DAPI / NeuN', stem + '_thalamus_zoom_noGFAP.png', fig)], title)
    print(name)
