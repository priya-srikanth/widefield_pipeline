"""A name the module never defines must not survive to run time.

WHY THIS EXISTS (2026-08-28). Three NameErrors reached production in one night, each of them a
name that was plainly undefined in the source and each of them found only by a multi-hour job
dying on it:

  * `write_animal_confusion_grid` -- the frozen-decoder CLI called a function defined 12 lines
    below its `__main__` guard. Importing the module was fine; running it was not, so every test
    passed. It wrote the decoder JSON and died before the encoder, leaving a half-finished
    artifact set behind an exit code nobody was reading.
  * `spec` in `pooled_frozen_loso` -- an extraction moved the local into the callee and left the
    result dict reading it. Killed the [lick] alignment ~30 minutes in.
  * `config` in `locanmf_contralateral` -- used but never imported, dormant only because the arm
    that calls it is not on the nightly path.

The suite could not catch any of them, because a NameError in a branch no test executes is
invisible to a test suite by construction. Static resolution catches all three in under a second,
and this is the check the nightly run needs most: the behaviour box runs these modules unattended
for hours, and the cost of the failure is not the crash but the plausible-looking partial output
it leaves on the share.

SCOPE. F821 only -- undefined name. Deliberately not a style gate: this pins one property that has
actually broken the pipeline, and a lint suite that fails for unrelated reasons is a lint suite
people learn to skip.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _ruff():
    """The interpreter's ruff, else one on PATH, else skip -- ruff is not a hard dependency."""
    r = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True)
    if r.returncode == 0:
        return [sys.executable, "-m", "ruff"]
    exe = shutil.which("ruff")
    return [exe] if exe else None


def test_no_undefined_names_anywhere_in_the_package():
    ruff = _ruff()
    if ruff is None:
        pytest.skip("ruff not installed")
    r = subprocess.run(
        ruff + ["check", "--select", "F821", "--no-cache", "--output-format=concise",
                str(ROOT / "wfield_local"), str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (
        "undefined name(s) -- these do not fail at import, they fail hours into a run:\n"
        + (r.stdout or r.stderr))
