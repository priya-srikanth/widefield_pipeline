"""Guards for the rest arm's engagement gate.

The gate was DOCUMENTED in four rest analyses and implemented in none of them until 2026-09-16
(Priya: *"all only on working trials?"*). The failure was silent in both directions -- the prose
asserted it, the stdout header printed "(working trials, ...)", and the only filter applied was
position agreement. These tests assert the behaviour rather than the claim.
"""
from __future__ import annotations

import numpy as np
import pytest

re_ = pytest.importorskip("wfield_local.rest_engagement")


def test_a_terminal_collapse_is_gated_out():
    """A trailing run of non-responses is the sated tail and must be excluded."""
    from wfield_local.precue_engagement_states import engagement_gate

    n = 200
    order = np.arange(n)
    responded = np.ones(n, bool)
    responded[150:] = False                      # sustained collapse to the end
    pos = np.array(["close_L"] * n)
    not_eng = engagement_gate(order, responded, pos)
    assert not_eng[-1]
    assert not not_eng[0]
    assert not_eng.sum() >= 30


def test_a_recovering_dip_is_NOT_gated_out():
    """Only the FINAL sustained collapse counts -- a mid-session patch that recovers is a motor
    problem, not satiety, and gating it would discard working trials."""
    from wfield_local.precue_engagement_states import engagement_gate

    n = 300
    order = np.arange(n)
    responded = np.ones(n, bool)
    responded[120:170] = False                   # dips, then recovers
    pos = np.array(["close_L"] * n)
    not_eng = engagement_gate(order, responded, pos)
    assert not not_eng[-1]
    assert not not_eng.any()


def test_the_gate_is_indexed_BY_CUE():
    """The collectors locate bracketing trials as indices into `cue_samples`, so a gate indexed any
    other way would be applied to the wrong trials while still looking correct."""
    import inspect

    src = inspect.getsource(re_.engaged_by_cue)
    assert "order = np.arange(n)" in src
    assert "cue_samples" in src


def test_it_fails_OPEN_and_says_so():
    """An unavailable gate must keep every period AND announce it -- never silently drop the
    session, and never silently drop the gate."""
    sess = {"label": "PS99_0101"}                # no events on disk
    eng, note = re_.engaged_by_cue(sess, np.arange(10) * 1000, np.zeros(10, int))
    assert eng.shape == (10,)
    assert eng.all()
    assert "UNGATED" in note


def test_no_cues_is_not_a_crash():
    eng, note = re_.engaged_by_cue({"label": "PS99_0101"}, np.zeros(0), np.zeros(0))
    assert eng.shape == (0,)
    assert "no cues" in note


def test_the_rest_collectors_actually_apply_the_gate():
    """The whole point: the filter must be in the CODE, not only in the docstring.

    This is the assertion whose absence let four analyses claim a gate none of them ran.
    """
    import importlib
    import inspect

    for mod, fn in (("scripts.rest_migration.rest_frozen_decoder", "_collect"),
                    ("scripts.rest_migration.shared_position_projection", "_session_terms")):
        m = pytest.importorskip(mod)
        src = inspect.getsource(getattr(m, fn))
        code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
        assert "engaged_by_cue" in code, f"{mod}.{fn} does not build the gate"
        assert "engaged[prev]" in code and "engaged[nc]" in code, \
            f"{mod}.{fn} builds the gate but does not apply it to BOTH bracketing trials"
        importlib.invalidate_caches()


def test_the_ungated_escape_hatch_exists_and_is_not_the_default():
    """`--no-engagement-gate` is for MEASURING the size of the correction, never for a result, so
    it must exist and must be opt-in."""
    import inspect

    m = pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")
    src = inspect.getsource(m.main)
    assert '"--no-engagement-gate", action="store_true"' in src
    assert "gate=not a.no_engagement_gate" in src
