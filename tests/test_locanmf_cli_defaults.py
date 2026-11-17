"""The LocaNMF CLIs default to configs/defaults.yaml ``locanmf`` (code review 2026-11-04, item 4).

run_locanmf used to carry its own literals (--loc-thresh 70, --r2-thresh 0.99) while the yaml and
batch_locanmf said 80 / 0.95, so a by-hand re-run of one session from the README fitted a different
decomposition from every other session with no warning. Both CLIs must read the same yaml values.
"""
from __future__ import annotations

import argparse
import sys

import pytest

from wfield_local import config


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def _batch_defaults() -> dict:
    """batch_locanmf builds its parser inside main(); stop at parse_args and read the defaults."""
    from wfield_local import batch_locanmf

    def _stop(self, *a, **k):
        raise _Captured(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(argparse.ArgumentParser, "parse_args", _stop)
        mp.setattr(sys, "argv", ["batch_locanmf"])
        with pytest.raises(_Captured) as ei:
            batch_locanmf.main()
    p = ei.value.parser
    return {a.dest: a.default for a in p._actions}


def test_run_locanmf_defaults_equal_yaml():
    from wfield_local import run_locanmf
    lp = config.defaults()["locanmf"]
    ns = run_locanmf.build_parser().parse_args(["--allen-dir", "x", "--output", "y", "--label", "z"])
    assert ns.loc_thresh == float(lp["loc_thresh"])
    assert ns.r2_thresh == float(lp["r2_thresh"])
    assert ns.maxrank == lp["maxrank"]


def test_batch_and_run_locanmf_agree():
    from wfield_local import run_locanmf
    lp = config.defaults()["locanmf"]
    b = _batch_defaults()
    ns = run_locanmf.build_parser().parse_args(["--allen-dir", "x", "--output", "y", "--label", "z"])
    assert b["loc"] == lp["loc_thresh"] == ns.loc_thresh
    assert b["r2"] == lp["r2_thresh"] == ns.r2_thresh
    assert b["maxrank"] == lp["maxrank"] == ns.maxrank
