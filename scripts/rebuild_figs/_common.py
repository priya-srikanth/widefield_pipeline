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

def default_out() -> Path:
    """$WF_REBUILD_FIGS_OUT, else <rebuild_root>/rebuild_figs/<run tag> under a rebuild root (coordinator 11-04: written
    under the run tag, so arms / blocks never share a folder), else <system temp>/wf_rebuild_figs."""
    if os.environ.get("WF_REBUILD_FIGS_OUT"):
        return Path(os.environ["WF_REBUILD_FIGS_OUT"])
    if os.environ.get("WIDEFIELD_REBUILD_ROOT", "").strip():
        from wfield_local import run_mode
        return run_mode.subdir(Path(os.environ["WIDEFIELD_REBUILD_ROOT"]) / "rebuild_figs")
    return Path(tempfile.gettempdir()) / "wf_rebuild_figs"


DEFAULT_OUT = default_out()


# R3 (fourth review, 11-03): DEFAULT INPUTS FOLLOW THE RUN. Inputs made by a routed writer default to that writer's
# folder for this run (inside the rebuild root under the profile); inputs with no routed writer (a dated backup, a
# one-off diagnostic cache) are refused in a non-default run unless given explicitly.
def nondefault_run() -> str | None:
    """The run's full output tag, or "rebuild-root" (None = a default run)."""
    from wfield_local import lp_licks, rebuild_root
    return lp_licks.output_tag() or ("rebuild-root" if rebuild_root.active() else None)


def epoch_data_dir() -> Path:
    """`grant_figures/epoch/data` of THIS run (`lp_licks.output_dir`: the tag folder; the rebuild root's under one)."""
    from wfield_local import config, lp_licks
    return lp_licks.output_dir(Path(config.resolver().root("cohort_out")) / "grant_figures" / "epoch") / "data"


def require_explicit(value, default, flag: str, *, ok_in_rebuild: bool = False) -> None:
    """SystemExit when ``value`` is the unrouted ``default`` in a non-default run (R3). ``ok_in_rebuild``: the default
    resolves inside the rebuild root (its writer writes there), so it is fine under one."""
    from wfield_local import rebuild_root
    mode = nondefault_run()
    if ok_in_rebuild and rebuild_root.active():
        return
    if mode and Path(value) == Path(default):
        raise SystemExit(f"{flag} defaults to a pre-rebuild input with no routed writer; in run {mode!r} pass "
                         f"{flag} explicitly (or run without the render profile)")


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
