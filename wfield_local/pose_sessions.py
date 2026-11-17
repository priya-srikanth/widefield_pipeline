"""The session_poses clip-folder resolver, library code.

Moved VERBATIM from `scripts/session_poses.py` on 2026-11-05 (code review 11-04, modularity item 1: `combined_encoding` imported it from scripts/). The script re-exports it.
"""
from __future__ import annotations
from pathlib import Path


def session_dir(rv, spec: str) -> tuple[str, str, Path]:
    """``animal:date`` -> the trial-subset clip folder; ``animal:date:tag`` -> ``<animal>_<date>_<tag>`` (e.g. tag
    ``full`` = every trial from a whole-video O2 run, `o2_inference collect`)."""
    animal, date, *tag = spec.split(":")
    from wfield_local import dlc_project
    name = f"{animal}_{date}" + (f"_{tag[0]}" if tag else "")
    return animal, date, dlc_project.project_dir(rv).parent / "session_poses" / name
