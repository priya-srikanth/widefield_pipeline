import re
from pathlib import Path

p = Path(__file__).parent / "imaging_page" / "imaging_results.html"
s = p.read_text(encoding="utf-8")


def label(fid):
    return fid[3:]


def reorder(order):
    """order = list of OLD ids in the new page order; new ids are the sorted old ids."""
    global s
    new_ids = sorted(order, key=lambda x: (len(x), x))
    blocks = {}
    for fid in order:
        m = re.search(r'<figure id="%s">.*?</figure>' % fid, s, re.S)
        blocks[fid] = m.group(0)
    first = min(s.index(blocks[f]) for f in order)
    last = max(s.index(blocks[f]) + len(blocks[f]) for f in order)
    seg = s[first:last]
    for f in order:
        seg = seg.replace(blocks[f], "@@SLOT@@", 1)
    assert seg.replace("@@SLOT@@", "").strip() == "", seg.replace("@@SLOT@@", "")[:200]
    out = []
    mapping = {}
    for old, new in zip(order, new_ids):
        mapping[old] = new
        b = blocks[old].replace('id="%s"' % old, 'id="@@%s"' % new, 1)
        b = b.replace('Fig %s &middot;' % label(old), 'Fig @@%s &middot;' % label(new), 1)
        out.append(b)
    s = s[:first] + "\n".join(out) + s[last:]
    # cross-references elsewhere on the page
    for old, new in mapping.items():
        if old != new:
            s = re.sub(r"Fig %s\b(?! &middot;)" % label(old), "Fig @@%s" % label(new), s)
    s = s.replace("@@", "")
    return mapping


m2 = reorder(["fig2c", "fig2", "fig2b"])
m9 = reorder(["fig9c", "fig9d", "fig9e", "fig9f", "fig9g", "fig9j", "fig9i", "fig9h", "fig9k"])
print(m2)
print(m9)
p.write_text(s, encoding="utf-8")
