"""Reproduction check for the rebuilt figure scripts (cohort-1 rebuild, 2026-11-03, step 3).

Compares a NEW render against a REFERENCE render:
  images  pixel-identical? max |diff| (0-255, any channel), fraction of differing pixels, and a side-by-side PNG
          (reference | new | |diff| x 8) written next to the new image as <new>_vs_<refdir>_<ref>.png. Both are compared
          as RGB (alpha dropped, as `docs/results_pages/imaging/page_figs.py` does); a size mismatch is reported and
          the side-by-side is still written (no diff stats).
          `--page-transform` first applies the results-page copy step (page_figs.py: RGB, downscale to <= 1600 px
          wide, LANCZOS) to the NEW image, for references that are the page copies of a figure.
  tables  CSV vs CSV: same columns / rows, max |diff| of every numeric column, exact match of the others.
          Lines starting with '#' are dropped; a multi-section CSV (blank-line separated, as frozen_refit_pooled.csv)
          is compared section by section.

    python scripts/rebuild_figs/compare_figs.py img REF.png NEW.png [--page-transform]
    python scripts/rebuild_figs/compare_figs.py csv REF.csv NEW.csv [--sort col1,col2]

Read-only on its inputs; writes only the side-by-side next to NEW.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np


def _rgb(path_or_img):
    from PIL import Image
    im = path_or_img if hasattr(path_or_img, "size") else Image.open(path_or_img)
    if im.mode == "RGBA":                                    # composite on white (matplotlib PNGs are opaque anyway)
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def page_transform(path):
    """`docs/results_pages/imaging/page_figs.py`: RGB, downscale to <= 1600 px wide (LANCZOS)."""
    from PIL import Image
    im = Image.open(path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    return im


def compare_images(ref, new, page=False, side_by_side=True) -> dict:
    from PIL import Image
    a = _rgb(ref)
    b = _rgb(page_transform(new)) if page else _rgb(new)
    res = {"ref": str(ref), "new": str(new), "ref_size": a.size, "new_size": b.size}
    if a.size == b.size:
        A, B = np.asarray(a, np.int16), np.asarray(b, np.int16)
        D = np.abs(A - B)
        res.update(identical=bool((D == 0).all()), max_abs_diff=int(D.max()),
                   frac_pixels_differ=float((D.max(axis=2) > 0).mean()),
                   frac_pixels_differ_gt16=float((D.max(axis=2) > 16).mean()))
        dimg = Image.fromarray(np.clip(D * 8, 0, 255).astype(np.uint8))
    else:
        res.update(identical=False, max_abs_diff=None, frac_pixels_differ=None, frac_pixels_differ_gt16=None)
        dimg = Image.new("RGB", a.size, (255, 255, 255))
    if side_by_side:
        w = a.width + b.width + dimg.width
        h = max(a.height, b.height, dimg.height)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        canvas.paste(dimg, (a.width + b.width, 0))
        tag = Path(ref).parent.name + "_" + Path(ref).stem
        sbs = Path(new).with_name(Path(new).stem + ("_page" if page else "") + f"_vs_{tag}.png")
        canvas.save(sbs)
        res["side_by_side"] = str(sbs)
    return res


def _sections(path):
    import pandas as pd
    txt = [ln for ln in Path(path).read_text().splitlines() if not ln.startswith("#")]
    blocks, cur = [], []
    for ln in txt:
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
            cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    return [pd.read_csv(io.StringIO("\n".join(b))) for b in blocks]


def compare_tables(ref, new, sort=None) -> list[dict]:
    out = []
    for k, (A, B) in enumerate(zip(_sections(ref), _sections(new))):
        r = {"section": k, "ref_shape": A.shape, "new_shape": B.shape,
             "same_columns": list(A.columns) == list(B.columns)}
        if sort:
            keys = [c for c in sort if c in A.columns and c in B.columns]
            if keys:
                A, B = A.sort_values(keys).reset_index(drop=True), B.sort_values(keys).reset_index(drop=True)
        if A.shape == B.shape and r["same_columns"]:
            worst, nonnum = 0.0, []
            for c in A.columns:
                if np.issubdtype(A[c].dtype, np.number) and np.issubdtype(B[c].dtype, np.number):
                    x, y = A[c].to_numpy(float), B[c].to_numpy(float)
                    if not np.array_equal(np.isnan(x), np.isnan(y)):
                        nonnum.append(f"{c} (NaN pattern)")
                        continue
                    m = np.nanmax(np.abs(x - y)) if np.isfinite(x).any() else 0.0
                    worst = max(worst, float(m))
                elif not A[c].astype(str).equals(B[c].astype(str)):
                    nonnum.append(c)
            r.update(max_abs_numeric_diff=worst, mismatched_columns=nonnum)
        out.append(r)
    if len(_sections(ref)) != len(_sections(new)):
        out.append({"section": "count", "ref": len(_sections(ref)), "new": len(_sections(new))})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("img", "csv"))
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--page-transform", action="store_true", help="apply page_figs.py's copy step to NEW first")
    ap.add_argument("--sort", default="", help="csv: comma-separated key columns to sort both tables by")
    a = ap.parse_args(argv)
    if a.kind == "img":
        print(compare_images(a.ref, a.new, page=a.page_transform))
    else:
        for r in compare_tables(a.ref, a.new, sort=[s for s in a.sort.split(",") if s]):
            print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
