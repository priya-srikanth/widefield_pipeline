"""Imaging page: renumber every figure SECTION.POSITION in page order (Priya 10-26, encoder-page format).
Section N ('<h2>N. ...') figures become Fig N.1, N.2, ... with ids fig<N>-<k>. Rewrites caption labels, figure ids,
href targets, link text and every 'Fig / Figs' cross reference from one old -> new map (written to imaging_fig_map.csv),
drops the 7.1 / 7.1.1 numbers from the section-7 subheadings (they would read as figure numbers) and makes the sidebar
one collapsible list per section. Runs last in the rebuild chain (after p_img_s1face.py)."""
import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
P = HERE / "imaging_page" / "imaging_results.html"
s = P.read_text(encoding="utf8")
assert "fig1-1" not in s, "already renumbered"
TOK = r"\d+(?:\.\d+)?[a-z]?"

marks = [(m.start(), "sec", m.group(1)) for m in re.finditer(r"<h2>(\d+)\.", s)]
marks += [(m.start(), "fig", m.group(1)) for m in re.finditer(r'<figure id="([^"]+)"', s)]
marks.sort()
sec, k, mp, ids = None, 0, {}, {}
for pos, kind, val in marks:
    if kind == "sec":
        sec, k = val, 0
        continue
    cap = re.search(r'<b class="h">Fig (' + TOK + r") &middot;", s[pos:pos + 3000])
    assert cap and sec is not None, val
    k += 1
    old = cap.group(1)
    assert old not in mp, old
    mp[old] = f"{sec}.{k}"
    ids[val] = f"fig{sec}-{k}"
print(len(mp), "figures renumbered")


def m_tok(t):
    return "@@" + mp[t] + "@@" if t in mp else t


# caption labels and 'Fig / Figs' references (lists and ranges)
sep = r"(?:\s*(?:&ndash;|–|-|/|,|and|&amp;)\s*(?:Figs?\.?\s*)?)"
pat = re.compile(rf"(Figs?\.?\s+)({TOK}(?:{sep}{TOK})*)\b")
s = pat.sub(lambda m: m.group(1) + re.sub(TOK, lambda t: m_tok(t.group(0)), m.group(2)), s)
# anchors: targets and their text (status-table links show the bare label)
s = re.sub(r'href="#(fig[^"]+)"', lambda m: f'href="#{ids.get(m.group(1), m.group(1))}"', s)
s = re.sub(r'(<a href="#fig\d+-\d+">)(Fig )?(' + TOK + r")</a>",
           lambda m: m.group(1) + (m.group(2) or "") + mp.get(m.group(3), m.group(3)) + "</a>", s)
s = s.replace("@@", "")
s = re.sub(r'<figure id="([^"]+)"', lambda m: f'<figure id="{ids.get(m.group(1), m.group(1))}"', s)
# section-7 subheadings: words only
s = re.sub(r'(<h[34] class="h7">)7\.\d(?:\.\d)? ', r"\1", s)
# sidebar: one collapsible list per section
old_js = "var m = /^fig(\\d+(?:_\\d+)?)/.exec(f.id); var num = m ? m[1] : f.id;"
assert old_js in s
s = s.replace(old_js, "var num = s.id;   // one collapsible list per section (figures numbered section.position, 10-26)")
s = s.replace("// group by figure NUMBER (Fig 7, 7b ... 7n -> one collapsible \"Fig 7\" row), Priya 10-25",
              "// one collapsible list per SECTION (Priya 10-26)")
s = s.replace("'Show or hide figures ' + num + 'b onwards'", "'Show or hide the other figures of this section'")

fid = re.findall(r'<figure id="([^"]+)"', s)
assert len(fid) == len(set(fid)), "duplicate ids"
left = sorted(set(re.findall(r"Figs?\.?\s+(\d+[a-z]|\d+\.\d+[a-z])\b", s)))
print("unmapped letter tokens after 'Fig':", left)
P.write_text(s, encoding="utf8")
with open(HERE / "imaging_fig_map.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["old", "new"])
    for o, n in mp.items():
        w.writerow([o, n])
print("map ->", HERE / "imaging_fig_map.csv")
