"""PROVE A DECK REFACTOR CHANGED NOTHING — a semantic fingerprint of a built .pptx, and a stub
figure tree to build one against.

    PYTHONPATH=$(pwd) python scripts/deck_fingerprint.py stub-tree <dir>      # once
    PYTHONPATH=$(pwd) python scripts/deck_fingerprint.py print <deck.pptx> before.txt
    <refactor>
    PYTHONPATH=$(pwd) python scripts/deck_fingerprint.py print <deck.pptx> after.txt
    diff before.txt after.txt

WHY NOT BYTES, AND WHY NOT SLIDE COUNT. A .pptx is a zip whose bytes differ run to run -- part
ordering and timestamps -- so byte comparison always reports a difference and therefore reports
nothing. Slide count is the opposite failure: it would pass a refactor that swapped two slides'
speaker notes, which is precisely the damage a careless registry move does. This walks the built
deck and emits one line per shape -- text with its run sizes and colours, pictures as position,
size and image sha1, and each slide's RESOLVED speaker notes. Everything that reaches the reader
is in it; nothing that is an artefact of when the file was written is.

WHY A STUB TREE. On a machine that is not the analysis box, `figures_working` has a handful of
PNGs, so sections A-G place nothing and a fingerprint over that build proves only that H and I
are unchanged. The last real nightly's manifest -- written beside the deck by `_write_manifest`
-- names every figure it actually placed, so stubbing those exercises every section: 531 slides
and 840 figures here against 475 slides from the grant/epoch sets alone. The stubs' pixel content
is irrelevant; the CODE PATH is what is under test, and it is the same path before and after.
Sizes are derived from the filename rather than random, because a size that moved between the
before and after run would show up as a false diff.

TWO RULES THIS TOOL LEARNED THE HARD WAY, both of them the repo's existing lesson in a new place:

  RUN THE BEFORE TWICE. A baseline you have not shown to be reproducible is not a baseline.
  `print` refuses a fingerprint under 100 lines for the same reason: piping to a cp1252 console
  first produced two EMPTY files, which then compared equal and reported success.

  MUTATION-TEST IT. Measured 2026-09-21: swapping two `EPOCH_FIGURES` entries moves 60 lines, and
  changing three characters inside one speaker note moves 2. A verification tool nobody has shown
  can fail is decoration.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

#: Sizes the stubs cycle through. `big` and `grid` both scale by aspect ratio, so a tree of
#: identically-shaped figures would leave the fitting arithmetic untested.
SHAPES = [(1600, 1200), (2000, 900), (1200, 1600), (2400, 1400), (1000, 1000)]
FIXED_MTIME = 1_750_000_000       # constant, so the deck's manifest sidecar cannot drift


def stub_tree(dest, manifest=None):
    """Populate `dest` with one stub PNG per figure named in a deck manifest."""
    from PIL import Image

    if manifest is None:
        from wfield_local.paths import PathResolver
        manifest = (Path(PathResolver().root("labcams"))
                    / "spout_position_analysis_summary.manifest.json")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = [r["figure"] for r in json.loads(Path(manifest).read_text(encoding="utf-8"))["figures"]]
    for n in sorted(names):
        p = dest / n
        if p.exists():
            continue
        w, h = SHAPES[int(hashlib.sha1(n.encode()).hexdigest(), 16) % len(SHAPES)]
        Image.new("RGB", (w // 8, h // 8), (240, 240, 240)).save(p)
        os.utime(p, (FIXED_MTIME, FIXED_MTIME))
    print(f"{dest}: {len(list(dest.glob('*.png')))} stub PNGs from {len(names)} manifest rows")


def _runs(tf):
    out = []
    for para in tf.paragraphs:
        for r in para.runs:
            c = r.font.color
            try:
                rgb = str(c.rgb) if c and c.type is not None and c.rgb is not None else "-"
            except (AttributeError, TypeError):
                rgb = "-"
            out.append(f"{r.text!r} sz={r.font.size} b={r.font.bold} i={r.font.italic} c={rgb}")
    return out


def fingerprint(path):
    prs = Presentation(path)
    lines = [f"slides={len(prs.slides)}"]
    for i, s in enumerate(prs.slides, 1):
        # SORTED BY POSITION, not z-order: shape order inside the XML is an implementation detail
        # of the writer, while where a thing sits on the slide is what the reader sees.
        shapes = sorted(s.shapes, key=lambda sh: (sh.top or 0, sh.left or 0, sh.shape_type or 0))
        for sh in shapes:
            geom = (f"L{Emu(sh.left or 0).inches:.3f} T{Emu(sh.top or 0).inches:.3f} "
                    f"W{Emu(sh.width or 0).inches:.3f} H{Emu(sh.height or 0).inches:.3f}")
            if sh.shape_type == 13 or getattr(sh, "image", None) is not None:
                lines.append(f"{i:04d} PIC  {geom} "
                             f"img={hashlib.sha1(sh.image.blob).hexdigest()[:16]}")
            elif sh.has_text_frame:
                for r in _runs(sh.text_frame):
                    lines.append(f"{i:04d} TXT  {geom} {r}")
        if s.has_notes_slide:
            n = s.notes_slide.notes_text_frame.text
            if n.strip():
                lines.append(f"{i:04d} NOTE {hashlib.sha1(n.encode()).hexdigest()[:16]} "
                             f"len={len(n)} {n[:160]!r}")
    return lines


def main(argv):
    if len(argv) < 2 or argv[0] not in ("stub-tree", "print"):
        sys.exit(__doc__.strip().splitlines()[0] + "\n\nsee the module docstring for usage")
    if argv[0] == "stub-tree":
        return stub_tree(argv[1], argv[2] if len(argv) > 2 else None)
    lines = fingerprint(argv[1])
    if len(lines) < 100:
        sys.exit(f"refusing: only {len(lines)} fingerprint lines -- that is not a built deck")
    # EXPLICIT UTF-8 TO A FILE, never stdout: the notes carry arrows and en-dashes and the console
    # on the analysis box is cp1252.
    with open(argv[2], "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {argv[2]}: {len(lines)} lines")


if __name__ == "__main__":
    main(sys.argv[1:])
