"""Stable figure ids for the imaging page (Priya 10-27), the same scheme as the encoding goal page.

Each <figure> is keyed by its image name: id="f-<stem>" + data-fid. These never change when figures are
renumbered or reordered, so links from other pages / handoffs should use #f-<stem>. The section.position id
(fig<N>-<k>) is kept as an empty <span class="oldnum"> anchor at the top of the figure, so every link made
before this step still lands. In-page links are rewritten to the stable id. Writes imaging_fig_ids.csv
(fid, number, title) = the registry: fid -> current number.

Run after p_img_haufe.py. Asserts before writing.
"""
import csv
import html as H
import re
from pathlib import Path

HERE = Path(__file__).parent
PAGE = HERE / "imaging_page" / "imaging_results.html"
REG = HERE / "imaging_fig_ids.csv"

h = PAGE.read_text(encoding="utf8")
assert 'id="f-' not in h, "stable ids already applied"

old2fid, rows = {}, []
for m in re.finditer(r'<figure id="(fig\d+-\d+)">(.*?)</figure>', h, re.S):
    old, body = m.group(1), m.group(2)
    srcs = re.findall(r'<img src="([^"]+)"', body)
    assert len(srcs) == 1, (old, srcs)
    fid = Path(srcs[0]).stem
    num = re.search(r'<span class="fnum">Fig ([\d.]+)</span>', body).group(1)
    title = re.sub(r"<[^>]+>", "", re.search(r'<div class="ftitle"><span class="fnum">[^<]*</span>(.*?)</div>', body, re.S).group(1))
    old2fid[old] = fid
    rows.append((fid, num, H.unescape(title).strip()))
assert len(old2fid) == 74 and len(set(old2fid.values())) == 74, len(old2fid)

for old, fid in old2fid.items():
    h = h.replace(f'<figure id="{old}">', f'<figure id="f-{fid}" data-fid="{fid}"><span class="oldnum" id="{old}"></span>', 1)
n_links = 0


def _link(m):
    global n_links
    n_links += 1
    return f'href="#f-{old2fid[m.group(1)]}"'


h = re.sub(r'href="#(fig\d+-\d+)"', _link, h)
assert not re.search(r'href="#fig\d', h), "unresolved in-page figure link"

css_anchor = "figure { margin:0 0 18px;"
assert css_anchor in h
h = h.replace(css_anchor, ".oldnum { display:block; height:0; }\n" + css_anchor, 1)

PAGE.write_text(h, encoding="utf8")
with REG.open("w", newline="", encoding="utf8") as f:
    w = csv.writer(f)
    w.writerow(["fid", "number", "title"])
    w.writerows(rows)
print(f"{len(rows)} figures keyed by image name; {n_links} in-page links -> #f-<fid>; registry {REG.name}")
