"""EVERY DOC CROSS-REFERENCE MUST RESOLVE. Run it after moving any document.

    PYTHONPATH=$(pwd) python scripts/check_doc_links.py

WHY. This repo's documentation is its best asset and it is held together by relative links --
`DECISIONS.md` cites the status docs, the status docs cite each other and the topic docs, module
docstrings cite all of them. Moving one file breaks links in files that never mentioned the move.
Consolidating the sixteen status docs into `docs/status/` on 2026-09-21 broke 26 references in
three different shapes, and every one of them was silent: a dead relative link renders as ordinary
text on GitHub and as nothing at all in an editor.

Resolves against the file's own directory OR the repo root, because both conventions are in use
here -- markdown links are file-relative, while docstrings and prose in .py name paths from the
root. A reference that resolves under neither is broken.


Both are legitimate here: markdown links are file-relative, while docstrings and prose in .py
name paths from the repo root. A reference that resolves under neither is simply broken.
"""
import re
from pathlib import Path

ROOT = Path.cwd()
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE = re.compile(r"`([^`]*STATUS_2026[^`]*)`")
bad = []
for p in list(Path(".").rglob("*.md")) + list(Path(".").rglob("*.py")):
    if ".git" in p.parts or ".claude" in p.parts:
        continue
    try:
        s = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for c in [m for m in LINK.findall(s) if "STATUS_2026" in m] + [m.strip() for m in CODE.findall(s)]:
        c = c.split("#")[0].strip()
        if not c.endswith(".md"):
            continue
        if (p.parent / c).resolve().exists() or (ROOT / c).exists():
            continue
        bad.append((str(p), c))
print(f"{len(bad)} unresolved")
for f, c in bad:
    print(f"  {f}  ->  {c}")
