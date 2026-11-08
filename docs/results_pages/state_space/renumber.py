"""Renumber the page's figures N.k by position (N = the section's h2 number, k = order within the section) and rewrite
every reference: the figure's own number tag and caption label, "Fig N.k" / "Figs N.k-N.m" in the text, and the text
of links to a figure id (<a href="#f-...">N.k</a>). Permanent ids (f-<image stem>) never change. Then rebuild
fig_ids.csv. Usage: python docs/results_pages/state_space/renumber.py <path to index.html>"""
import re
import subprocess
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
# old number of every figure id, and its new number by position
old, new = {}, {}
for sec in re.finditer(r'<section id="[^"]+"[^>]*>.*?</section>', s, re.S):
    m = re.search(r"<h2>(\d+)\.", sec.group(0))
    figs = re.findall(r'<figure id="(f-[^"]+)"><div class="ftitle"><span class="fnum">Fig ([0-9.]+)</span>', sec.group(0))
    for k, (fid, num) in enumerate(figs, 1):
        old[fid] = num
        new[fid] = f"{m.group(1)}.{k}" if m else num
o2n = {old[f]: new[f] for f in old}
TOK = "\x00{}\x00"
# 1. per-figure tags (by id, so duplicates of an old number cannot collide)
for fid in old:
    s = re.sub(rf'(<figure id="{re.escape(fid)}"><div class="ftitle"><span class="fnum">Fig )[0-9.]+(</span>)',
               lambda m, f=fid: m.group(1) + TOK.format(new[f]) + m.group(2), s)
    s = re.sub(rf'(<a href="#{re.escape(fid)}">)[0-9.]+(</a>)', lambda m, f=fid: m.group(1) + TOK.format(new[f]) + m.group(2), s)
# 2. caption labels and in-text references "Fig N.k" (old numbers -> new), one pass with placeholders
s = re.sub(r"(Figs? )(\d+\.\d+)((?:&ndash;|-|–)(\d+\.\d+))?",
           lambda m: m.group(1) + TOK.format(o2n.get(m.group(2), m.group(2)))
           + (m.group(3).replace(m.group(4), TOK.format(o2n.get(m.group(4), m.group(4)))) if m.group(3) else ""), s)
s = s.replace("\x00", "")
p.write_text(s, encoding="utf-8")
changed = {k: v for k, v in o2n.items() if k != v}
print("renumbered:", changed or "nothing")
subprocess.run([sys.executable, str(Path(__file__).with_name("make_fig_ids.py"))], check=False)
