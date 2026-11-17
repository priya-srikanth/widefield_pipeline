"""Deck crops of the LMM hidden-slide images (the "notitle" copies the PI deck places).

`drop_title` = the deck builder's rule (presentations/_source_PI_overview_2026-11-02/insert_lmm_hidden_1103.py): cut
lmm_change_fig's two-line suptitle, i.e. everything down to the end of the second ink run + 4 px (runs 1-2 are the title
lines; a gap-length rule cut the top panel row off 2 x 2 figures). `stack` = add_lick_sessz_backup_1104.py (images one
above the other, 20 px white between).

    python -m scripts.lmm_hidden.crop notitle <src.png> <dst.png>
    python -m scripts.lmm_hidden.crop stack <dst.png> <a.png> <b.png> ...
Refuses to overwrite an existing destination.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _new(dst) -> Path:
    dst = Path(dst)
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    return dst


def drop_title(src, dst) -> Path:
    dst = _new(dst)
    im = Image.open(src).convert("RGB")
    ink = (np.asarray(im.convert("L")) < 200).sum(1) > 0
    runs, i = [], 0
    while i < len(ink) and len(runs) < 2:
        if ink[i]:
            j = i
            while j < len(ink) and ink[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    im.crop((0, runs[1][1] + 4, im.width, im.height)).save(dst)
    return dst


def stack(paths, dst) -> Path:
    dst = _new(dst)
    ims = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.width for i in ims)
    out = Image.new("RGB", (w, sum(i.height for i in ims) + 20 * (len(ims) - 1)), "white")
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height + 20
    out.save(dst)
    return dst


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "notitle":
        print(drop_title(a[1], a[2]))
    elif a[0] == "stack":
        print(stack(a[2:], a[1]))
    else:
        raise SystemExit(__doc__)
