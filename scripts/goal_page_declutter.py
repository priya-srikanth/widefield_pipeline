"""Declutter the goal page (Priya 10-29 "go ahead with page decluttering"): collapse figures IN PLACE into one closed
line each (order and numbers unchanged):
  1. every figure with data-tier="supplementary";
  2. in method / robustness sections, every figure that is not data-tier="key".
A figure's following <details> supplements are collapsed with it. Links into a collapsed figure open it (small script)."""
import re
from pathlib import Path

PAGE = Path("C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/index.html")
METHOD_SECTIONS = {"contestpeth", "landingrelabel", "lickdecrobust", "readmaps", "variance", "frdirection"}
h = PAGE.read_text(encoding="utf-8")
assert "collapsed-fig" not in h, "already decluttered"

secs = [(m.start(), m.group(1)) for m in re.finditer(r'<section id="([^"]+)"', h)]
bounds = [(s, secs[k + 1][0] if k + 1 < len(secs) else len(h), sid) for k, (s, sid) in enumerate(secs)]


def sec_of(pos):
    for s, e, sid in bounds:
        if s <= pos < e:
            return sid
    return None


targets = []
for m in re.finditer(r'<figure id="([^"]+)"([^>]*)>', h):
    tier = re.search(r'data-tier="([^"]+)"', m.group(2))
    tier = tier.group(1) if tier else ""
    sid = sec_of(m.start())
    if tier == "supplementary" or (sid in METHOD_SECTIONS and tier != "key"):
        targets.append((m.start(), m.group(1), tier, sid))

# wrap from the end so earlier offsets stay valid
n = 0
for start, fid, tier, _sid in sorted(targets, reverse=True):
    end = h.find("</figure>", start) + len("</figure>")
    while True:
        mm = re.match(r"\s*<details", h[end:])
        if not mm:
            break
        # find the matching </details> (nested details possible)
        depth, j = 0, end
        while True:
            o = h.find("<details", j)
            c = h.find("</details>", j)
            if o != -1 and o < c:
                depth += 1
                j = o + 8
            else:
                depth -= 1
                j = c + len("</details>")
                if depth == 0:
                    break
        end = j
    blk = h[start:end]
    num = re.search(r'class="fnum">Fig ([^<]+)</span>', blk)
    num = num.group(1) if num else "?"
    title = re.search(r'<div class="ftitle">(.*?)</div>', blk, re.S)
    title = re.sub(r"<[^>]+>", "", title.group(1)) if title else fid
    title = re.sub(r"^\s*Fig\s*[0-9.]+\s*", "", title).strip()
    if len(title) > 170:
        title = title[:167].rstrip() + "&hellip;"
    label = "supplementary" if tier == "supplementary" else "method / supporting"
    wrapped = (f'<details class="collapsed-fig" data-for="{fid}"><summary><b>Fig {num}</b> &mdash; {title} '
               f'<span class="cf-tag">{label}</span></summary>{blk}</details>')
    h = h[:start] + wrapped + h[end:]
    n += 1

css = ("\n<style>details.collapsed-fig{margin:4px 0;border-left:3px solid rgba(127,127,127,.35);padding:2px 0 2px 8px}"
       "details.collapsed-fig>summary{cursor:pointer;font-size:.9em;opacity:.85}"
       "details.collapsed-fig>summary .cf-tag{font-size:.8em;opacity:.7;margin-left:6px;border:1px solid rgba(127,127,127,.4);"
       "border-radius:3px;padding:0 4px}details.collapsed-fig[open]>summary{opacity:1}</style>\n")
js = ("\n<script>(function(){function openTo(id){var el=id&&document.getElementById(id);if(!el)return;var p=el.parentElement;"
      "while(p){if(p.tagName==='DETAILS')p.open=true;p=p.parentElement;}el.scrollIntoView();}"
      "window.addEventListener('hashchange',function(){openTo(location.hash.slice(1));});"
      "document.addEventListener('DOMContentLoaded',function(){if(location.hash)openTo(location.hash.slice(1));"
      "document.querySelectorAll('a[href^=\"#\"]').forEach(function(a){a.addEventListener('click',function(){"
      "setTimeout(function(){openTo(a.getAttribute('href').slice(1));},0);});});});})();</script>\n")
i = h.rfind("</body>")
h = (h[:i] + css + js + h[i:]) if i != -1 else (h + css + js)
PAGE.write_text(h, encoding="utf-8")
vis = len(re.findall(r'<figure id=', h)) - n
print(f"collapsed {n} figures in place; still open: {vis}")
for _sid2 in [b[2] for b in bounds]:
    pass
print("by section:", {sid: sum(1 for t in targets if t[3] == sid) for _s, _e, sid in bounds})
