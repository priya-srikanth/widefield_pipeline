"""Rebuild fig_ids.csv (permanent figure id -> current N.k number, title, image) from the page's figure tags.
Run after any renumbering: python docs/results_pages/state_space/make_fig_ids.py"""
import csv
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
s = (HERE / "index.html").read_text(encoding="utf-8")
rows = re.findall(r'<figure id="(f-[^"]+)"><div class="ftitle"><span class="fnum">Fig ([0-9.]+)</span>(.*?)</div>', s)
with open(HERE / "fig_ids.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["fid", "number", "title", "image"])
    for fid, num, title in rows:
        w.writerow([fid, num, html.unescape(re.sub("<[^>]+>", "", title)), fid[2:] + ".png"])
print(len(rows), "figures registered")
