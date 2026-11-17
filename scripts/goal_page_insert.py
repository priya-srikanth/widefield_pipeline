"""Insert / move figures on the goal page and renumber their section (links, caption headers, fig_ids.csv).

Used for the 10-28 page edits (Priya: figures numbered by their neighbors). Every figure number inside the section is
reassigned in page order as <sec>.1, <sec>.2, ...; links whose text is a changed number (with or without "Fig ") are
updated; plain-text mentions of changed numbers are PRINTED for a manual check (they may refer to other sections).
Two-pass rename (old -> temporary -> new) so swapped numbers never collide.

Usage (from Python): insert(section_id, sec_num, after_fid, html_blocks, fig_rows); renumber(section_id, sec_num)
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "_goal_page" / "index.html"
IDS = PAGE.parent / "fig_ids.csv"


def _sec(h: str, sid: str) -> tuple[int, int]:
    s = h.find(f'<section id="{sid}"')
    if s < 0:
        raise ValueError(sid)
    return s, h.find("</section>", s)


def insert(sid: str, sec_num: str, after_fid: str, blocks: str, fig_rows: list[list[str]]) -> dict:
    h = PAGE.read_text(encoding="utf-8")
    s, e = _sec(h, sid)
    i = h.find(f'<figure id="{after_fid}"', s, e)
    if i < 0:
        raise ValueError(after_fid)
    j = h.find("</figure>", i) + len("</figure>")
    h = h[:j] + blocks + h[j:]
    PAGE.write_text(h, encoding="utf-8")
    if fig_rows:
        rows = list(csv.reader(IDS.open(encoding="utf-8", newline="")))
        known = {r[0] for r in rows[1:]}
        rows += [r for r in fig_rows if r[0] not in known]
        csv.writer(IDS.open("w", encoding="utf-8", newline="")).writerows(rows)
    return renumber(sid, sec_num)


def renumber(sid: str, sec_num: str) -> dict:
    h = PAGE.read_text(encoding="utf-8")
    s, e = _sec(h, sid)
    sec = h[s:e]
    ids = re.findall(r'<figure id="([^"]+)"', sec)
    nums = re.findall(r'class="fnum">Fig ([^<]+)</span>', sec)
    if len(ids) != len(nums):
        raise RuntimeError(f"{len(ids)} figures but {len(nums)} numbers in {sid}")
    chg = {fid: (o, f"{sec_num}.{k + 1}") for k, (fid, o) in enumerate(zip(ids, nums)) if o != f"{sec_num}.{k + 1}"}
    tmp = {fid: f"{sec_num}.TMP{k}" for k, fid in enumerate(chg)}
    for stage in (0, 1):
        for fid, (o, n) in chg.items():
            a, b = (o, tmp[fid]) if stage == 0 else (tmp[fid], n)
            i = h.find(f'<figure id="{fid}"')
            j = h.find("</figure>", i)
            f = h[i:j].replace(f'class="fnum">Fig {a}</span>', f'class="fnum">Fig {b}</span>', 1)
            f = f.replace(f"Fig {a} &middot;", f"Fig {b} &middot;", 1)
            h = h[:i] + f + h[j:]
            h = re.sub(rf'(<a href="#{re.escape(fid)}">(?:Fig )?){re.escape(a)}(</a>)', rf"\g<1>{b}\g<2>", h)
    PAGE.write_text(h, encoding="utf-8")
    rows = list(csv.reader(IDS.open(encoding="utf-8", newline="")))
    for r in rows[1:]:
        if "f-" + r[0] in chg:
            r[1] = chg["f-" + r[0]][1]
    csv.writer(IDS.open("w", encoding="utf-8", newline="")).writerows(rows)
    olds = {o for o, _n in chg.values() if "." in o and "TMP" not in o and not o.split(".")[1].startswith(("X", "Y"))}
    for m in re.finditer(r"(?<![>\d.])(\d+\.\d+)(?![\d<])", h):
        if m.group(1) in olds and "&middot;" not in h[m.end():m.end() + 12]:
            print("CHECK plain mention:", m.group(1), "|",
                  re.sub("<[^>]+>", "", h[max(0, m.start() - 80):m.end() + 20]).replace("\n", " "))
    return chg
