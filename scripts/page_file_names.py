"""Give every figure on a results page its FILE NAME (Priya 11-09: "we should make including file names the default for
all pages"; docs/templates/README.md).

Each `<figure>`'s caption ends with `<span class="ffile"><b>File:</b> <code><prefix>/<img src></code></span>`, the
figure's path on the share. Idempotent: an existing File line is rewritten from the figure's CURRENT `<img src>`, so run
it after every build or figure swap (as `page_stable_ids.py`). Figures without an `<img>` are left alone.

Usage: python scripts/page_file_names.py <index.html> --prefix <share folder of the page's figures>
  e.g. --prefix state_space_recovery/page_figs (relative to Widefield/, or any path that finds the file)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FILE_RE = re.compile(r' <span class="ffile"><b>File:</b> <code>[^<]*</code></span>')


def add_file_names(html: str, prefix: str) -> tuple[str, int]:
    """(html with one File line per figure caption, number of figures noted)."""
    prefix = prefix.rstrip("/")

    def one(m):
        fig = FILE_RE.sub("", m.group(0))
        img = re.search(r'<img src="([^"]+)"', fig)
        if img is None or "</figcaption>" not in fig:
            return fig
        note = f' <span class="ffile"><b>File:</b> <code>{prefix}/{img.group(1)}</code></span>'
        return fig.replace("</figcaption>", note + "</figcaption>", 1)

    # figures AND collapsible supplements (`<details class="supp-levels">` with a figcaption, as on the goal page)
    return re.subn(r'<figure\b[^>]*>.*?</figure>|<details\b[^>]*>.*?</details>', one, html, flags=re.S)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page")
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args(argv)
    p = Path(a.page)
    s, n = add_file_names(p.read_text(encoding="utf-8"), a.prefix)
    p.write_text(s, encoding="utf-8")
    print(f"{p}: {n} figures noted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
