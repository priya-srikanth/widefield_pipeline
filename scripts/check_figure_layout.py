"""NO SIDECAR MAY SIT FLAT BESIDE A FIGURE. Run it against any figure output directory.

    python scripts/check_figure_layout.py <dir>                  # report; exit 1 if loose
    python scripts/check_figure_layout.py <dir> --fix --dry-run  # show the moves
    python scripts/check_figure_layout.py <dir> --fix            # make them (MOVE ONLY)

WHY THIS EXISTS. `wfield_local/figure_layout` moved every sidecar to ``<dir>/data/`` and every
vector copy to ``<dir>/svg/`` on 2026-09-21, and `find_sidecar` PREFERS the new location. A writer
that still builds ``out_dir / f"{stem}.csv"`` by hand therefore produces the worst possible
outcome and produces it silently: the fresh numbers land flat where no reader looks, the reader
resolves the pre-migration copy in ``data/``, and BOTH FILES EXIST so nothing is missing and
nothing errors. On 2026-09-22, one day after the migration, five writers were still doing exactly
this and every consumer of 15f/15g/15h/15s/10cs had been reading two-day-old numbers.

`figure_layout`'s own docstring predicted it -- *"the coverage tool counts PNGs, and a stray CSV is
invisible to every check in this repo"* -- and then no check was written. This is that check.

IT IS DELIBERATELY EMPIRICAL, not a source scan. A source scan for `with_suffix(".csv")` goes
blind the moment somebody writes the join a new way, and it cannot see a file a THIRD-PARTY script
dropped in the directory. What is actually on disk cannot be evaded by phrasing.

STALE-SHADOW is the severe class and is reported separately: a flat file NEWER than the ``data/``
copy the readers prefer. A flat file with no counterpart is merely untidy; a flat file that is
newer than the one being read means the pipeline is quietly serving old numbers.

NOT EVERY LOOSE FILE IS A STRAY, and `--ignore <glob>` exists for the ones that are not. The
working ``analysis_figures/`` root is the case to know: `nightly_figs._publish_figs` copies
the analysis artefacts to MICROSCOPE with ``Path(out).glob("*.json")`` AT THE TOP LEVEL, so
those JSONs are the SOURCE of a publish step, not sidecars of the PNGs beside them. Tidying
them into ``data/`` would make that glob match nothing and the nightly would report
``copied: 0`` without failing. Check that tree as::

    python scripts/check_figure_layout.py <analysis_figures> --ignore "*.json"
"""
from __future__ import annotations

import pathlib
import sys

#: Subdirectories that are part of the layout rather than content to inspect.
LAYOUT_DIRS = {"data", "svg", "retired"}
#: Extensions that legitimately live beside a figure at the top level.
FIGURE_EXT = {".png"}
#: Not artefacts of a renderer: notes a human left in the directory.
IGNORE_NAMES = {"README.md", "readme.md", "README.txt", "Thumbs.db", ".DS_Store"}

#: Filename patterns that are never a renderer's output. ``~$...`` is the lock file Word
#: and PowerPoint leave behind while a document is OPEN -- it appears and vanishes on its
#: own, so reporting it would make a clean tree fail whenever somebody has the deck up.
IGNORE_GLOBS = ["~$*"]

#: ARTEFACT FAMILIES THAT LIVE HERE ON PURPOSE AND ARE NOT SIDECARS OF ANY FIGURE.
#:
#: Structure cannot distinguish these from a real stray -- `exclusion_mask_painted_PS92.npy` and
#: `epoch_15g_transfer_ENL_matrix.csv` are both "a non-PNG whose stem no figure owns", and only
#: one of them should move. So the exemption is a LIST WITH A REASON rather than a heuristic that
#: would quietly widen. `test_check_figure_layout` asserts each entry still earns its place: an
#: exemption is only valid while no figure owns the stem and a reader addresses it another way.
#:
#: exclusion_mask_painted -- `paint_exclusion.mask_dir()` puts these in `grant_figures/` and
#:   `load_masks()` glob-reads them back from exactly there; `beta_maps` is the consumer. There is
#:   no `exclusion_mask_painted*.png`. Moving them breaks the mask, silently and cohort-wide.
NOT_SIDECARS = ("exclusion_mask_painted",)


def scan(root: pathlib.Path, ignore=()):
    """``(strays, shadows, n_dirs, n_png)`` for every figure directory under ``root``.

    A "figure directory" is one that directly contains at least one PNG -- that is what makes a
    loose CSV in it a sidecar rather than somebody's data folder.
    """
    strays, shadows, n_dirs, n_png = [], [], 0, 0
    root = pathlib.Path(root)
    ignore = [*IGNORE_GLOBS, *ignore]
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        if not d.is_dir() or set(d.relative_to(root).parts) & LAYOUT_DIRS:
            continue
        pngs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in FIGURE_EXT]
        if not pngs:
            continue
        n_dirs += 1
        n_png += len(pngs)
        for f in d.iterdir():
            if not f.is_file() or f.suffix.lower() in FIGURE_EXT or f.name in IGNORE_NAMES:
                continue
            if any(f.name.startswith(x) for x in NOT_SIDECARS):
                continue
            if any(f.match(g) for g in ignore):
                continue
            sub = "svg" if f.suffix.lower() == ".svg" else "data"
            twin = d / sub / f.name
            if twin.exists() and f.stat().st_mtime > twin.stat().st_mtime:
                shadows.append((f, twin))
            else:
                strays.append(f)
    return strays, shadows, n_dirs, n_png


def repair(strays, shadows, *, dry_run: bool = True):
    """Put every flat file where the layout says it goes. MOVE ONLY -- nothing is deleted.

    A stale shadow's twin is RETIRED rather than overwritten: it is the copy every reader has
    been resolving, so it is the one somebody may need in order to explain a number already
    quoted in a document. It goes to ``<sub>/retired/<name>.SUPERSEDED_<stamp>``, the same
    convention `prestroke_reference` uses for exactly this situation.
    """
    import time

    # LOCAL time, deliberately: this stamp is read by a person comparing a retired file
    # against when they remember a run happening, not parsed by anything.
    stamp = time.strftime("%Y%m%d%H%M%S")
    moves = []
    for f, twin in shadows:
        moves.append((twin, twin.parent / "retired" / f"{twin.name}.SUPERSEDED_{stamp}"))
        moves.append((f, twin))
    for f in strays:
        sub = "svg" if f.suffix.lower() == ".svg" else "data"
        dest = f.parent / sub / f.name
        # This flat copy is OLDER than the one in data/ (otherwise it would be a shadow), so
        # here it is the flat file that is superseded.
        if dest.exists():
            dest = dest.parent / "retired" / f"{f.name}.SUPERSEDED_{stamp}"
        moves.append((f, dest))
    for src, dest in moves:
        print(f"{'would move' if dry_run else 'move'}  {src}")
        print(f"            -> {dest}")
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise SystemExit(f"refusing to overwrite {dest}")
            src.replace(dest)
    return moves


def main(argv) -> int:
    rest = list(argv[1:])
    ignore = []
    while "--ignore" in rest:
        i = rest.index("--ignore")
        if i + 1 >= len(rest):
            return _die("--ignore needs a glob")
        ignore.append(rest[i + 1])
        del rest[i:i + 2]
    flags = {a for a in rest if a.startswith("--")}
    unknown = flags - {"--fix", "--dry-run"}
    if unknown:
        return _die(f"unknown flag(s): {' '.join(sorted(unknown))}")
    roots = [pathlib.Path(a) for a in rest if not a.startswith("--")]
    if not roots:
        return _die("no directory given -- refusing to report a pass on nothing")
    total_dirs = total_png = 0
    strays: list[pathlib.Path] = []
    shadows: list[tuple[pathlib.Path, pathlib.Path]] = []
    for r in roots:
        if not r.is_dir():
            return _die(f"{r} is not a directory -- refusing to report a pass on nothing")
        s, sh, nd, npng = scan(r, ignore)
        strays += s
        shadows += sh
        total_dirs += nd
        total_png += npng
    # REFUSE AN EMPTY MEASUREMENT. A tool that prints "0 problems" after looking at nothing is
    # worse than no tool: three checks written this week passed that way before being fixed.
    if total_png == 0:
        return _die(f"found no PNG under {', '.join(map(str, roots))} -- nothing was checked")

    for f, twin in sorted(shadows):
        print(f"STALE-SHADOW  {f}\n              is NEWER than {twin}, which is what readers get")
    for f in sorted(strays):
        print(f"STRAY         {f}")
    print(f"\n{total_dirs} figure dir(s), {total_png} PNG: "
          f"{len(shadows)} stale-shadow, {len(strays)} stray")
    if (shadows or strays) and "--fix" in flags:
        print()
        repair(strays, shadows, dry_run="--dry-run" in flags)
        if "--dry-run" not in flags:
            print("\nre-run without --fix to confirm the tree is clean")
    return 1 if (shadows or strays) else 0


def _die(msg: str) -> int:
    print(f"check_figure_layout: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
