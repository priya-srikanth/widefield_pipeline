"""The STOPPED trial class: the terminal quit period, which every other figure throws away.

Priya, 2026-09-11: "do we already have (or can we add) a post-stroke vs pre-stroke 'stopped' trials
pattern similarity analysis?" and then "the stopped class should basically be another frozen
decoder / encoder analysis set". `flag_engagement` had only ever been a FILTER -- every collector
selected `~not_eng` -- so promoting its complement to a first-class trial class touched four sites
that each held their own copy of the rule.

THE FAILURE MODE THESE TESTS EXIST FOR. All four copies read

    parts = [XE[engaged rows]]                      # unconditional
    if variant == "working": parts.append(XU[...])  # conditional

which is correct for `lick` and `working` and silently catastrophic for `stopped`: the engaged rows
are the LICKING trials, so a set defined as "the animal had quit" would have been mostly trials
where it did not. `_class_select` is the single implementation and these pin its contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import grant_figures as G


def _masks(variant, n_e=8, n_u=10, quit_from=6):
    sess_e = np.ones(n_e, bool)
    sess_u = np.ones(n_u, bool)
    not_eng = np.zeros(n_u, bool)
    not_eng[quit_from:] = True
    return G._class_select(variant, sess_e, sess_u, not_eng), not_eng


def test_stopped_takes_no_engaged_trials_at_all():
    """THE ONE THAT MATTERS. An engaged trial is a LICKING trial; the animal had not quit."""
    (me, mu), _ne = _masks("stopped")
    assert not me.any(), "stopped must contain no licking trials"
    assert mu.sum() == 4


def test_stopped_and_working_partition_the_unengaged_trials():
    """Not nested, not overlapping: a trial is in one class or the other, never both."""
    (_me_s, mu_s), _ = _masks("stopped")
    (_me_w, mu_w), _ = _masks("working")
    assert not (mu_s & mu_w).any(), "a trial cannot be both working and stopped"
    assert (mu_s | mu_w).all(), "every unengaged trial must land in exactly one of the two"


def test_working_is_unchanged_by_the_promotion():
    (me, mu), not_eng = _masks("working")
    assert me.all()
    assert np.array_equal(mu, ~not_eng)


def test_lick_takes_only_engaged_trials():
    (me, mu), _ = _masks("lick")
    assert me.all()
    assert not mu.any()


def test_an_animal_with_no_unengaged_trials_does_not_crash_any_class():
    """`pool_sessions` returns an EMPTY no-lick arm for a session with no misses at all."""
    for v in ("lick", "working", "stopped"):
        me, mu = G._class_select(v, np.ones(5, bool), np.zeros(0, bool), np.zeros(0, bool))
        assert len(mu) == 0
        assert me.any() == (v != "stopped")


def test_stopped_is_a_variant_for_cue_and_precue_but_never_for_lick():
    """A trial inside the quit period is a non-response, so it has no lick to align to."""
    assert "stopped" in G._variants("cue")
    assert "stopped" in G._variants("precue")
    assert "stopped" not in G._variants("lick")


def test_the_render_planner_and_the_collectors_agree_on_the_variant_list():
    """They used to hold two copies of the rule, and only one of them honoured --only-variant."""
    units = G.render_units()
    for key in ("6", "7b", "8"):
        got = {(a, v) for k, a, v in units if k == key}
        if not got:
            continue
        for align in {a for a, _v in got}:
            assert {v for a, v in got if a == align} == set(G._variants(align))


def test_class_note_names_every_class_it_is_given():
    """Fifteen inline copies of a two-branch conditional captioned `stopped` as its own complement."""
    assert "QUIT" in G._class_note("stopped").upper()
    assert G._class_note("stopped") != G._class_note("working")
    assert G._class_note("lick") != G._class_note("working")


@pytest.mark.parametrize("variant", ["lick", "working", "stopped"])
def test_session_trials_returns_the_same_rows_for_X_and_for_blk(variant):
    """The block vector must line up row-for-row with the data or the bootstrap resamples nonsense.

    `_session_trials` computes the mask ONCE for both `field` values, which is the property that
    makes that true; this pins it against a future edit that recomputes the mask per field.
    """
    n_e, n_u, n_feat = 6, 8, 3
    bd = {
        "XE": np.arange(n_e * n_feat, dtype=float).reshape(n_e, n_feat),
        "XU": np.arange(n_u * n_feat, dtype=float).reshape(n_u, n_feat) + 100,
        "BE": np.arange(n_e, dtype=np.int64),
        "BU": np.arange(n_u, dtype=np.int64) + 100,
        "GE": np.zeros(n_e, int), "GU": np.zeros(n_u, int),
        "en": np.array(["close_L"] * n_e), "un": np.array(["close_L"] * n_u),
        "not_eng": np.array([False] * 5 + [True] * 3),
    }
    X = G._session_trials(bd, 0, "close_L", variant, "X")
    B = G._session_trials(bd, 0, "close_L", variant, "blk")
    assert len(X) == len(B), f"{variant}: {len(X)} patterns against {len(B)} block ids"


def test_pre_panel_is_licking_only_for_lick_and_working():
    """The pre-stroke panel of `_collect_5c` must take NO unengaged rows except for `stopped`.

    THE REGRESSION THIS PINS, 2026-09-11. Promoting `stopped` to a trial class meant routing the
    selection rule through one helper, and routing the PRE branch through it too silently added the
    miss-while-working trials to a panel that had always been licking-only. The pooled pre-stroke
    set went 21,017 -> 22,076 trials and its accuracy 0.89 -> 0.859, which moves every pre-stroke
    position number in the deck -- and it surfaced only because a brand-new figure's pre bar
    disagreed with a number printed on an older one. That is luck, not a guard.

    `_collect_7` states the rule: a pre-stroke animal is not missing, so `working` adds nothing at
    pre except a different KIND of trial, and the reference then differs from itself.
    """
    import inspect

    src = inspect.getsource(G._collect_5c)
    pre = src[src.index("PRE: leave-one-session-out"):]
    pre = pre[:pre.index("by_day = {}")]
    assert 'if variant == "stopped":' in pre, (
        "the pre branch must special-case `stopped`; routing every variant through "
        "`_class_select` here re-adds miss-while-working trials to the pre panel")
    assert "mu_te = np.zeros(len(GU), bool)" in pre, (
        "lick/working must contribute NO unengaged rows to the pre panel")
