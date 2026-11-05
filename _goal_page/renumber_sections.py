"""Renumber every goal-page figure as SECTION.POSITION in page order (Priya 10-26: "adjust naming and ordering ... why
two letters"): section N's figures become Fig N.1, N.2, ... (ids fig<N>-<k>); the key figure stays 'Key figure'.
Rewrites caption labels, ids, href="#fig..." targets, link text and every 'Fig / Figs' cross reference from one
old -> new map; Musall citations are protected. Writes the map to <page dir>/fig_map.csv (old,new) and prints any
'Fig <token>' left unmapped. The sidebar groups figures per SECTION (one collapsible list each).
Usage: python renumber_sections.py [page.html]   (default goal_page/index.html next to this script)"""
import csv
import re
import sys
from pathlib import Path

P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "goal_page" / "index.html"
s = P.read_text(encoding="utf8")
s = re.sub(r"(Musall[^<]{0,40}?)Fig", lambda m: m.group(1) + "@MUSFIG@", s)
TOK = r"\d+(?:[a-z]+\d*)?(?:\.\d+)?"

# 1. sections (h2 'N. ...') and figures in page order
marks = [(m.start(), "sec", m.group(1)) for m in re.finditer(r"<h2>(\d+)\.", s)]
marks += [(m.start(), "fig", m.group(1)) for m in re.finditer(r'<figure id="([^"]+)"', s)]
marks.sort()
sec, k, mp, ids = None, 0, {}, {}
for pos, kind, val in marks:
    if kind == "sec":
        sec, k = val, 0
        continue
    if val == "figkey":
        continue
    cap = re.search(r'<b class="h">Fig (' + TOK + r")", s[pos:pos + 4000])
    if cap is None or sec is None:
        print("SKIP (no caption label or outside a numbered section):", val)
        continue
    k += 1
    old = cap.group(1)
    if old in mp:
        raise SystemExit(f"duplicate caption label {old}")
    mp[old] = f"{sec}.{k}"
    ids[val] = f"fig{sec}-{k}"
print(len(mp), "figures renumbered")


def m_tok(t: str) -> str:
    return "@@" + mp[t] + "@@" if t in mp else t


# 2. caption labels + 'Fig / Figs' references (lists and ranges)
sep = r"(?:\s*(?:&ndash;|–|-|/|,|and|&amp;)\s*(?:Figs?\.?\s*)?)"
pat = re.compile(rf"(Figs?\.?\s+)({TOK}(?:{sep}{TOK})*)")
s = pat.sub(lambda m: m.group(1) + re.sub(TOK, lambda t: m_tok(t.group(0)), m.group(2)), s)
# 3. anchors: targets and their text
s = re.sub(r'href="#(fig[^"]+)"', lambda m: f'href="#{ids.get(m.group(1), m.group(1))}"', s)
s = re.sub(r'(<a href="#fig\d+-\d+">)(Fig )?(' + TOK + r")</a>",
           lambda m: m.group(1) + (m.group(2) or "") + (mp.get(m.group(3), m.group(3))) + "</a>", s)
s = s.replace("@@", "")
# 4. figure ids
s = re.sub(r'<figure id="([^"]+)"', lambda m: f'<figure id="{ids.get(m.group(1), m.group(1))}"', s)
# 5. sidebar: one collapsible group per SECTION (was: per leading figure number)
old_js = "var m = /^fig(\\d+)/.exec(f.id); var num = m ? m[1] : f.id;"
new_js = "var num = s.id;   // one collapsible list per section (figures are numbered section.position, 10-26)"
if old_js in s:
    s = s.replace(old_js, new_js)
else:
    print("sidebar grouping line not found (already changed?)")
s = s.replace("'Show or hide figures ' + num + 'b onwards'", "'Show or hide the other figures of this section'")
s = s.replace("@MUSFIG@", "Fig")
left = sorted(set(re.findall(r"Figs?\.?\s+(\d+[a-z]+\d*)\b", s)))
print("unmapped letter-suffix tokens after 'Fig':", left)
fid = re.findall(r'<figure id="([^"]+)"', s)
assert len(fid) == len(set(fid)), "duplicate ids"
P.write_text(s, encoding="utf8")
with open(P.parent / "fig_map.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["old", "new"])
    for o, n in mp.items():
        w.writerow([o, n])
print("map ->", P.parent / "fig_map.csv")
