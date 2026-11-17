"""11-06 (_daqlk render): imaging_lmm read the mixed models' deficit table from `<encoding root>/qc` and failed in the
tagged run, where trial_attempts_sessions.csv lives in `run_mode.deficit_qc_dir()` (the run's routed trial_attempts
folder). remap_lmm, remap_checks and iti_lmm_figures had the same hard-coded path. Every deficit read must go through
`run_mode.deficit_qc_dir()`.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CALL = re.compile(r"\b(?:mm|mx|mixed_models)\.deficit\(([^)]*)\)")


def test_every_deficit_read_uses_deficit_qc_dir():
    bad = []
    for f in sorted((REPO / "scripts").rglob("*.py")):
        if "archive" in f.parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in CALL.finditer(line):
                arg = m.group(1)
                if '"qc"' in arg or "'qc'" in arg:
                    bad.append(f"{f.relative_to(REPO)}:{i}: {line.strip()}")
    assert not bad, "deficit table read from a hard-coded qc path:\n" + "\n".join(bad)
