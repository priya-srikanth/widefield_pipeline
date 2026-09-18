"""Redraw every map figure from its saved bundle -- presentation changes, no recomputation.

`epoch_figures.replot_map` has existed since the bundles did and had NO caller, so a change to how
a map is DRAWN has been costing a full re-render of the analysis behind it. That is the same
expense `rotation_maps --replot` and `reference_family_figure` already refuse to pay, and the
bundle exists precisely so this does not have to be paid: "the expensive part was the analysis, and
the bundle makes the picture cheap to revise".

OVERWRITES THE CANONICAL NAME, deliberately and unlike the library default. `replot_map` appends
`_replot` unless a name is passed, which is right for an experiment and wrong for a refresh -- the
deck places canonical filenames, so a `_replot` twin would leave the deck showing the OLD drawing
while a corrected one sat beside it unused. That is the "superseded set wearing the canonical name"
trap in reverse and just as bad.

REFUSES TO TOUCH A BUNDLE WHOSE FIGURE IS GONE. A bundle whose PNG has been retired should not
resurrect the PNG.

RUN:  python -m scripts.replot_map_figures [--dry-run] [--only SUBSTR] [--dir DIR]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--only", default=None, help="substring filter on the figure stem")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    from wfield_local import epoch_figures as ef
    from wfield_local.paths import PathResolver

    d = a.dir or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    bundles = sorted(d.glob("*_bundle.npz"))
    if a.only:
        bundles = [b for b in bundles if a.only in b.name]
    if not bundles:
        print(f"!! no bundles in {d}" + (f" matching {a.only!r}" if a.only else ""))
        return 1

    todo, skipped = [], []
    for b in bundles:
        stem = b.name.replace("_bundle.npz", "")
        (todo if (d / f"{stem}.png").exists() else skipped).append((b, stem))

    print(f"{len(todo)} map figures to redraw from bundles in {d}")
    for _, stem in skipped:
        print(f"   skip {stem}: its .png is not here (retired?) -- not resurrecting it")
    if a.dry_run:
        for _, stem in todo:
            print(f"   would redraw {stem}")
        return 0

    t0, ok, failed = time.time(), 0, []
    for b, stem in todo:
        try:
            ef.replot_map(b, name=stem)
            ok += 1
            print(f"   {stem}", flush=True)
        except Exception as ex:                                          # noqa: BLE001
            failed.append((stem, f"{type(ex).__name__}: {str(ex)[:90]}"))
            print(f"   !! {stem}: {type(ex).__name__}: {str(ex)[:90]}", flush=True)
    print(f"\nredrew {ok}/{len(todo)} in {time.time() - t0:.0f}s")
    for stem, err in failed:
        print(f"   FAILED {stem}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
