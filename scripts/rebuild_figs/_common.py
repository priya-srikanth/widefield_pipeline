"""Shared plumbing for the recreated results-page figure scripts in `scripts/rebuild_figs/` (cohort-1 rebuild, 11-03).

Every script here follows one contract:
  --out DIR     where the PNG (+ sidecar CSVs) go; the plotting inputs are pickled to DIR/data/<name>_figdata.pkl.
                Default = $WF_REBUILD_FIGS_OUT, else <system temp>/wf_rebuild_figs -- a NEW folder, never a page,
                deck or share folder (the originals wrote into presentations/_source_PI_overview_2026-11-02/
                build_inputs/ or a session scratchpad; the rebuild publishes by an explicit copy step instead).
  --plot-only   redraw from DIR/data/<name>_figdata.pkl without loading any data (same draw function, so the PNG
                is pixel-identical to the full run's).
Run from the repo root with the repo on the path, e.g.
    PYTHONPATH=. python scripts/rebuild_figs/fig_behaviour_hits.py --out <dir>
"""
from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path

DEFAULT_OUT = Path(os.environ.get("WF_REBUILD_FIGS_OUT") or Path(tempfile.gettempdir()) / "wf_rebuild_figs")


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output folder (default {DEFAULT_OUT}); figdata goes to <out>/data/")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw from <out>/data/<name>_figdata.pkl; no data load")
    return ap


def figdata_path(out: Path, name: str) -> Path:
    return Path(out) / "data" / f"{name}_figdata.pkl"


def save_figdata(out: Path, name: str, obj) -> Path:
    """Atomic write (tmp + replace) of everything the draw function needs."""
    p = figdata_path(out, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(p)
    return p


def load_figdata(out: Path, name: str):
    p = figdata_path(out, name)
    if not p.exists():
        raise SystemExit(f"--plot-only: no figure data at {p}; run once without --plot-only")
    return pickle.loads(p.read_bytes())
