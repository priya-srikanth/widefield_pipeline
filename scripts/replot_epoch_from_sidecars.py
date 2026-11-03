"""Redraw epoch bar / contrast figures from their stored sidecars, in the 11-03 session-dot style.

Priya 2026-11-03, figure rule 2: every summary at a categorical x shows its SESSION values as faded
open markers (shape = animal, colour = epoch). This replots WITHOUT recomputation: it reads
``data/<name>.csv`` + ``_sessions.csv`` + ``_meta.csv`` (`epoch_figures.replot_from_sidecars`) and
writes ``<name>_nolines.png`` beside the original. It never overwrites the original PNG and never
writes a sidecar on the share (the redraw happens in a scratch dir); it hashes every sidecar it
read before and after and fails if any changed.

    python scripts/replot_epoch_from_sidecars.py <png> [<png> ...] [--suffix _nolines] [--work DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

from wfield_local import epoch_figures as ef  # noqa: E402
from wfield_local.writeguard import assert_writable  # noqa: E402


def _sidecars(png: pathlib.Path):
    """Every sidecar the replot may read: the figure's own, and its companion bar figure's."""
    d = png.parent / "data"
    stems = {png.stem, png.stem.replace("delta", "", 1)}
    want = {f"{s}{suf}.csv" for s in stems for suf in ("", "_sessions", "_meta")}
    return sorted(d / n for n in want if (d / n).exists())


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pngs", nargs="+")
    ap.add_argument("--suffix", default="_nolines")
    ap.add_argument("--work", default=None, help="scratch dir for the redraw (default: a temp dir)")
    a = ap.parse_args(argv)
    work_root = pathlib.Path(a.work or tempfile.mkdtemp(prefix="replot_epoch_"))
    bad = 0
    for s in a.pngs:
        png = pathlib.Path(s)
        out = png.with_name(f"{png.stem}{a.suffix}.png")
        if out.resolve() == png.resolve():
            raise SystemExit("refusing to overwrite the original")
        assert_writable(out)
        cars = _sidecars(png)
        before = {p: _sha(p) for p in cars}
        try:
            q, kind, n, note = ef.replot_from_sidecars(png, out, workdir=work_root / png.stem)
            msg = f"OK   {kind:8s} dots={n:4d}  {q}" + (f"  [{note}]" if note else "")
            # round trip: the redraw's own value sidecar must equal the original byte for byte
            orig = png.parent / "data" / f"{png.stem}.csv"
            redo = work_root / png.stem / "data" / f"{png.stem}.csv"
            if orig.exists() and redo.exists():
                msg += "  roundtrip=" + ("identical" if orig.read_bytes() == redo.read_bytes() else "DIFFERS")
            print(msg, flush=True)
        except Exception as ex:                                        # noqa: BLE001
            bad += 1
            print(f"SKIP {png.name}: {type(ex).__name__}: {ex}", flush=True)
        after = {p: _sha(p) for p in cars}
        changed = [p.name for p in cars if before[p] != after[p]]
        if changed:
            raise SystemExit(f"SIDECAR CHANGED: {changed}")
        print(f"     sidecars unchanged (sha256, {len(cars)} files): {', '.join(p.name for p in cars)}",
              flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
