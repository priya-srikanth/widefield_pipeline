import re, os, html

_PANEL_CACHE = {}


def load_panel(path, prefix):
    """Return (inner_svg_body, width, height) with all ids namespaced.

    Namespacing is done in ONE pass per pattern with a callback; the previous
    version looped over every id and re-scanned the whole body each time, which
    is O(ids x bytes) and took minutes on a 600 KB figure with thousands of ids.
    Parsed bodies are cached per path, so placing seven crops of one file reads
    and parses it once.
    """
    key = (path, prefix)
    if key in _PANEL_CACHE:
        return _PANEL_CACHE[key]
    if path in _PANEL_CACHE:
        body, w, h = _PANEL_CACHE[path]
    else:
        s = open(path, encoding='utf-8', errors='ignore').read()
        w = float(re.search(r'\bwidth="([\d.]+)(?:pt)?"', s).group(1))
        h = float(re.search(r'\bheight="([\d.]+)(?:pt)?"', s).group(1))
        s = re.sub(r'<\?xml[^>]*\?>', '', s)
        s = re.sub(r'<!DOCTYPE[^>]*>', '', s, flags=re.S)
        body = re.search(r'<svg[^>]*>(.*)</svg>\s*$', s, flags=re.S).group(1)
        _PANEL_CACHE[path] = (body, w, h)
    p = prefix + '_'
    nb = re.sub(r'\bid="([^"]+)"', lambda m: 'id="%s%s"' % (p, m.group(1)), body)
    nb = re.sub(r'url\(#([^)]+)\)', lambda m: 'url(#%s%s)' % (p, m.group(1)), nb)
    nb = re.sub(r'(xlink:href|href)="#([^"]+)"',
                lambda m: '%s="#%s%s"' % (m.group(1), p, m.group(2)), nb)
    _PANEL_CACHE[key] = (nb, w, h)
    return nb, w, h


def place(path, prefix, x, y, target_w, label=None, label_dx=-10, label_dy=10):
    """Place a whole panel SVG.

    Uses a plain <g transform="translate(...) scale(...)"> rather than a nested
    <svg viewBox>. Illustrator turns every nested <svg> into its own artboard-like
    clipping group, which is what made the composed figures open as a deep stack
    of layers; a transform group opens as one named group per panel.
    """
    body, w, h = load_panel(path, prefix)
    sc = target_w / w
    out = ['<g id="Panel_%s">' % prefix]
    if label:
        out.append('<text x="%.2f" y="%.2f" font-family="Arial" font-weight="bold" '
                   'font-size="11" fill="#000">%s</text>' % (x + label_dx, y + label_dy, str(label).upper()))
    out.append('<g transform="translate(%.3f,%.3f) scale(%.6f)">' % (x, y, sc))
    out.append(body)
    out.append('</g>')
    out.append('</g>')
    return '\n'.join(out), h * sc

def place_png(path, prefix, x, y, target_w, label=None, label_dx=-10, label_dy=10):
    """Embed a PNG as base64 so the SVG stays self-contained."""
    import base64
    from PIL import Image
    w, h = Image.open(path).size
    b64 = base64.b64encode(open(path, 'rb').read()).decode('ascii')
    sc = target_w / w
    out = ['<g id="Panel_%s">' % prefix]
    if label:
        out.append('<text x="%.2f" y="%.2f" font-family="Arial" font-weight="bold" '
                   'font-size="11" fill="#000">%s</text>' % (x + label_dx, y + label_dy, str(label).upper()))
    out.append('<image x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
               'xlink:href="data:image/png;base64,%s"/>' % (x, y, target_w, h * sc, b64))
    out.append('</g>')
    return '\n'.join(out), h * sc


def place_crop(path, prefix, x, y, target_w, crop, label=None, label_dx=-10, label_dy=10):
    """Place a rectangular region of a vector SVG. crop = (x0, y0, x1, y1) in the
    source file's own points, origin top-left. Keeps everything live vector.

    A transform group plus one explicit clipPath, for the Illustrator reason in
    `place`: no nested <svg>, so the panel opens as a single named group.
    """
    body, w, h = load_panel(path, prefix)
    x0, y0, x1, y1 = crop
    cw, ch = x1 - x0, y1 - y0
    sc = target_w / cw
    cid = 'clip_%s' % prefix
    out = ['<g id="Panel_%s">' % prefix]
    if label:
        out.append('<text x="%.2f" y="%.2f" font-family="Arial" font-weight="bold" '
                   'font-size="11" fill="#000">%s</text>' % (x + label_dx, y + label_dy, str(label).upper()))
    out.append('<clipPath id="%s"><rect x="%.3f" y="%.3f" width="%.3f" height="%.3f"/></clipPath>'
               % (cid, x, y, target_w, ch * sc))
    out.append('<g clip-path="url(#%s)">' % cid)
    out.append('<g transform="translate(%.3f,%.3f) scale(%.6f) translate(%.3f,%.3f)">'
               % (x, y, sc, -x0, -y0))
    out.append(body)
    out.append('</g></g></g>')
    return '\n'.join(out), ch * sc


def caption(x, y, width_pt, text, size=7.6, leading=9.4, bold_lead=None):
    """Wrapped figure caption as real SVG text. `bold_lead` is rendered bold at
    the start of the first line (e.g. 'Figure 1.')."""
    import textwrap
    # rough average glyph width for Arial at this size
    chars = max(int(width_pt / (size * 0.49)), 20)
    body = text if not bold_lead else text
    lines = textwrap.wrap(body, chars)
    out = ['<g id="CAPTION">']
    for i, ln in enumerate(lines):
        yy = y + i * leading
        if i == 0 and bold_lead:
            out.append('<text x="%.1f" y="%.1f" font-family="Arial" font-size="%.1f">'
                       '<tspan font-weight="bold">%s </tspan>%s</text>'
                       % (x, yy, size, bold_lead, _esc(ln)))
        else:
            out.append('<text x="%.1f" y="%.1f" font-family="Arial" font-size="%.1f">%s</text>'
                       % (x, yy, size, _esc(ln)))
    out.append('</g>')
    return '\n'.join(out), len(lines) * leading


def _esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
