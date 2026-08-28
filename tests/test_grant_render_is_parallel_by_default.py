"""The grant render defaults to PARALLEL, and there is exactly one definition of how many.

WHY (2026-08-28). `--jobs` defaulted to 1 while `nightly_figs` passed `--grant-jobs 8`. The nightly
was therefore fine and only DIRECT invocations paid: a full render run by hand took 6 h 35 m on a
24-core box, using about 1.5 of them. That is the shape of default that survives longest -- the
penalised path is the one a person runs while watching it, and the automated path that everyone
checks looks healthy.

TWO PROPERTIES, and the second is why this file is not one assertion:

  * PARALLEL BY DEFAULT. Not "supports --jobs". The capability existed the whole time.
  * ONE DEFINITION. `nightly_figs` must not carry its own copy of the number. It held a literal 8
    while grant_figures held 1; a repo that has been bitten by six copies of a decoder recipe and
    three of a lick discriminator does not need a fourth instance of the same lesson.

Also pinned: `1` still means serial, so the escape hatch works and a bisect can reach the old
behaviour.
"""
import argparse
import ast
import inspect
from pathlib import Path

import pytest

from wfield_local import grant_figures

ROOT = Path(__file__).resolve().parent.parent


def test_the_default_is_parallel_on_any_plausible_box():
    assert grant_figures._default_jobs() > 1, "the default render must not be serial"


def test_the_default_is_derived_from_the_box_not_hardcoded():
    """A laptop must not be told to run 8 workers because the analysis box has 24 cores."""
    src = inspect.getsource(grant_figures._default_jobs)
    assert "cpu_count" in src, "the worker count must come from the machine"


def test_jobs_1_still_selects_the_serial_path():
    src = inspect.getsource(grant_figures.main)
    assert "n_jobs > 1" in src, (
        "the parallel branch must test the RESOLVED count; testing `args.jobs` directly means a "
        "None default is falsy and silently restores the serial render")


def test_nightly_does_not_keep_its_own_copy_of_the_number():
    """`--grant-jobs` must defer, not duplicate: two defaults for one number drift apart."""
    src = (ROOT / "wfield_local" / "nightly_figs.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == "--grant-jobs" for a in node.args):
            continue
        default = next((k.value for k in node.keywords if k.arg == "default"), None)
        assert isinstance(default, ast.Constant) and default.value is None, (
            "--grant-jobs must default to None and let grant_figures._default_jobs() decide; a "
            "literal here is a second definition of the same number")
        return
    pytest.fail("could not find the --grant-jobs argument")
