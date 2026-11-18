"""Crop the footer-fixed frozen_claims_trend.png like the current slide-147 image: the 3 panels + legend, full width, no
suptitle (top: the widest white band in the top 20 % of the canvas) and no purple trial-set footer (bottom: a few px
above the first row holding the footer color, RUN_STAMP_COLOR #8a2be2). Writes a NEW file; the old crop is kept.
Usage: python crop_s147.py <src.png> <dst.png> [pad_to_height]   (slide 147: 401 = the old crop height)
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image


def main(src: str, dst: str, pad_h: int = 0) -> int:
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(int)
    h = a.shape[0]
    ink = (a.min(2) < 235).any(1)                                     # rows with any non-white pixel
    purple = ((np.abs(a[..., 0] - 0x8A) < 40) & (np.abs(a[..., 1] - 0x2B) < 40) & (np.abs(a[..., 2] - 0xE2) < 40)).any(1)
    foot = int(np.argmax(purple)) if purple.any() else h
    # bottom: last inked row above the footer, + 4 px
    above = np.where(ink[:foot])[0]
    bot = min(foot, int(above.max()) + 5)
    # top: middle of the widest white band in the top 20 % (between suptitle and panel titles)
    lim = int(0.2 * h)
    best, run, start, cut = 0, 0, 0, 0
    for y in range(lim):
        if not ink[y]:
            if run == 0:
                start = y
            run += 1
            if run > best and start > 0:
                best, cut = run, start + run // 2
        else:
            run = 0
    out = im.crop((0, cut, im.size[0], bot))
    if pad_h and out.size[1] < pad_h:               # white-pad to the old image's height: same aspect, box unchanged
        c = Image.new("RGB", (out.size[0], pad_h), "white")
        c.paste(out, (0, (pad_h - out.size[1]) // 2))
        out = c
    out.save(dst)
    print(f"{src} -> {dst}: rows {cut}..{bot} of {h} (footer starts at {foot}), size {out.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0))
