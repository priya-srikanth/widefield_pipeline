"""Give every figure on a results page a PERMANENT id, so links survive renumbering (Priya 10-27).

Same scheme as the goal page (wf_attempt/_goal_page): figure id = "f-<image name>" (the image file's basename), kept
forever; the printed number (data-num, "Fig N.k") is free to change on any reorder. Run AFTER the page's own build:

    python scripts/page_stable_ids.py <page/index.html> [--registry <page>/fig_ids.csv]

What it does, idempotently:
  * each <figure> with a number (data-num) gets id="f-<fid>" and data-fid; an empty <span id="figN-K"> is kept just
    inside it so a link written against the CURRENT number still lands (that number moves on a renumber, the f- id
    does not);
  * every in-page href="#figN-K" becomes href="#f-<fid>" (link text unchanged);
  * writes the registry fid,number,title (default: fig_ids.csv next to the page) -- decks and other pages cite
    "<number> [#f-<fid>]" and refresh the number from this file.
A figure whose image is reused elsewhere on the page gets "<name>_2", ... in page order; the registry keeps earlier
assignments, so a later insertion does not steal an existing id.
"""
import argparse
import csv
import html
import re
from pathlib import Path

FIG = re.compile(r'<figure\b([^>]*)>(.*?)</figure>', re.S)


def _fid(src):
    return re.sub(r'[^A-Za-z0-9_-]', '_', Path(src).stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--registry")
    a = ap.parse_args()
    page = Path(a.page)
    reg = Path(a.registry) if a.registry else page.parent / "fig_ids.csv"
    h = page.read_text(encoding="utf-8")
    prev = {}  # (image src, occurrence) -> fid from an earlier run
    if reg.exists():
        for r in csv.DictReader(open(reg, encoding="utf-8")):
            if r.get("src"):
                prev[(r["src"], int(r.get("occ") or 1))] = r["fid"]
    used, seen, num2fid, rows = set(prev.values()), {}, {}, []

    def fig(m):
        attrs, body = m.group(1), m.group(2)
        num = re.search(r'data-num="([^"]+)"', attrs)
        img = re.search(r'<img src="([^"]+)"', body)
        if not num or not img:
            return m.group(0)
        num, src = num.group(1), img.group(1)
        seen[src] = seen.get(src, 0) + 1
        key = (src, seen[src])
        fid = prev.get(key)
        if fid is None:
            base = _fid(src)
            fid, k = base, 1
            while fid in used:
                k += 1
                fid = f"{base}_{k}"
            used.add(fid)
        numid = "fig" + num.replace(".", "-")
        num2fid[numid] = fid
        attrs = re.sub(r'\s(?:id|data-fid)="[^"]*"', "", attrs)
        body = re.sub(r'^\s*<span id="fig[\d-]+" class="figanchor"></span>', "", body)
        lab = re.search(r'data-label="([^"]*)"', attrs)
        rows.append({"fid": fid, "number": num, "title": html.unescape(lab.group(1)) if lab else "", "src": src,
                     "occ": seen[src]})
        return (f'<figure id="f-{fid}" data-fid="{fid}"{attrs}><span id="{numid}" class="figanchor"></span>{body}'
                f'</figure>')

    h = FIG.sub(fig, h)
    h = re.sub(r'href="#(fig[\d-]+)"', lambda m: f'href="#f-{num2fid[m.group(1)]}"' if m.group(1) in num2fid
               else m.group(0), h)
    page.write_text(h, encoding="utf-8")
    with open(reg, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["fid", "number", "title", "src", "occ"])
        w.writeheader()
        w.writerows(rows)
    dead = sorted(set(re.findall(r'href="#(fig[\d-]+)"', h)))
    print(f"{page}: {len(rows)} figures -> stable ids; registry {reg}; unresolved numeric links: {dead or 'none'}")


if __name__ == "__main__":
    main()
