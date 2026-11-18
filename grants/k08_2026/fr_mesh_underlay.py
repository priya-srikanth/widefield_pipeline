"""Draw the whole face on the FaceRhythm mesh maps.

Matplotlib writes every mesh point as a <use> of one symbol per subplot, filled
by value, so points near zero are almost white and the face shape disappears
where nothing changed. This adds a faint grey stroke to every point of the FULL
mesh layers only (the symbols used 1,282 times, one per subplot), leaving the
significance rings and their red / blue colours untouched.
"""
import collections, re, sys

STROKE = 'stroke: #c4c4c4; stroke-width: 0.11'

def underlay(src, dst, mesh_n=None):
    s = open(src, encoding='utf-8', errors='ignore').read()
    c = collections.Counter(re.findall(r'<use xlink:href="#([^"]+)"', s))
    n = mesh_n or max(c.values())
    full = {k for k, v in c.items() if v == n}
    print('%s: mesh layers %d x %d points' % (src, len(full), n))

    def fix(m):
        tag, href = m.group(0), m.group(1)
        if href not in full or 'stroke' in tag:
            return tag
        if 'style="' in tag:
            return tag.replace('style="', 'style="%s; ' % STROKE)
        return tag.replace('/>', ' style="%s"/>' % STROKE)

    out = re.sub(r'<use xlink:href="#([^"]+)"[^>]*/>', fix, s)
    open(dst, 'w', encoding='utf-8').write(out)
    print('  wrote', dst)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        underlay(p, p.replace('.svg', '_mesh.svg'))
