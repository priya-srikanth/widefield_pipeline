"""COMPARE TWO SVG RENDER DIRECTORIES, ignoring what matplotlib makes non-reproducible.

    PYTHONPATH=$(pwd) python scripts/compare_svg_renders.py <before-dir> <after-dir>

WHY THIS EXISTS. The repo verifies a refactor by rendering before and after and diffing the
output. That works for PNG -- 90 of 90 byte-identical across the 2026-09-21 `grant_kit`
extraction. It does NOT work for SVG: every file differs on every run, so a plain byte comparison
reports 90 differences whatever you did, which is the same as reporting nothing. Measured that
day: 89 of 89 SVGs "differed", and every one of them was identical once normalised.

MATPLOTLIB SVG IS NOT BYTE-REPRODUCIBLE, so SVG byte-comparison is not a verification method --
it always reports a difference and therefore reports nothing. Two things vary per render:
  <dc:date>...</dc:date>     the render timestamp
  id="pXXXXXXXXX" / url(#p..) / id="imageXXXXXXXX"   random element ids
Normalise both away and what is left is the drawing.
"""
import re
import sys
from pathlib import Path

DATE = re.compile(r"<dc:date>.*?</dc:date>")
IDS = re.compile(r"(?:#|\")(p[0-9a-f]{8,}|image[0-9a-f]{8,}|m[0-9a-f]{8,}|C[0-9a-f]{8,})")

def norm(p):
    s = Path(p).read_text(encoding="utf-8", errors="replace")
    s = DATE.sub("<dc:date/>", s)
    return IDS.sub("ID", s)

before, after = Path(sys.argv[1]), Path(sys.argv[2])
same = diff = miss = 0
bad = []
for f in sorted(after.glob("*.svg")):
    b = before / f.name
    if not b.exists():
        miss += 1; continue
    if norm(b) == norm(f):
        same += 1
    else:
        diff += 1; bad.append(f.name)
print(f"SVG normalised: identical={same}  differing={diff}  not-in-baseline={miss}")
for n in bad[:8]:
    print("  DIFFERS:", n)
# REFUSE A PASS ON NOTHING. Pointed at a directory that does not exist -- a typo, an unset shell
# variable -- this printed "identical=0 differing=0" and exited 0, which reads exactly like
# success. The deck fingerprint learned the same lesson from two empty files that compared equal.
if same + diff == 0:
    raise SystemExit(f"refusing: compared 0 files. Are {before} and {after} right?")
raise SystemExit(1 if diff else 0)
