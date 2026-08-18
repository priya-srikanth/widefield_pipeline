"""Two adjacent blocks at the same position must not merge into one.

Priya, 2026-08-18. The pipeline started a new block only when the POSITION changed, so a far_L block
followed by another far_L block became a single CV group. Audited against the firmware's own
block_number: 118 of 4216 blocks (2.8%) across the 48 curated + 8/17 sessions.

These tests pin the rule and its two documented limits, so a future simplification back to
"split on position change" fails loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local.block_ids import DEFAULT_BLOCK_SIZE_MAX, block_ids


def test_a_run_longer_than_block_size_max_is_split():
    """The regression this exists for: 11 trials at one position cannot be one block of max 8."""
    ids = block_ids(np.array([3] * 11), block_size_max=8)
    assert len(set(ids)) == 2
    assert list(ids) == [0] * 8 + [1] * 3


def test_a_run_at_exactly_block_size_max_stays_one_block():
    """A KNOWN LIMIT, pinned so it is not mistaken for a bug.

    A 4+4 merge lands at run-length exactly block_size_max and is indistinguishable from one genuine
    maximal block. About 10 of the 118 merges are of this kind and remain merged; the run-length rule
    cannot see them and this test records that it does not pretend to.
    """
    assert len(set(block_ids(np.array([2] * 8), block_size_max=8))) == 1


def test_position_changes_still_start_a_block():
    ids = block_ids(np.array([0, 0, 0, 1, 1, 0, 0]), block_size_max=8)
    assert list(ids) == [0, 0, 0, 1, 1, 2, 2]


def test_unusable_trials_are_excluded_and_do_not_break_a_run():
    """-1 codes (unresolvable position) get no block and must not split the surrounding run."""
    ids = block_ids(np.array([1, 1, -1, 1, 1]), block_size_max=8)
    assert ids[2] == -1
    assert len({int(i) for i in ids if i >= 0}) == 1


def test_split_boundaries_do_not_exceed_block_size_max():
    rng = np.random.RandomState(0)
    codes = np.repeat(rng.randint(0, 6, 40), rng.randint(1, 15, 40))
    ids = block_ids(codes, block_size_max=DEFAULT_BLOCK_SIZE_MAX)
    sizes = np.bincount(ids[ids >= 0])
    assert sizes.max() <= DEFAULT_BLOCK_SIZE_MAX


def test_the_decoder_uses_this_rule_and_not_a_local_copy():
    """A second copy of the rule is how the two normalisations diverged elsewhere in this project."""
    import inspect

    from wfield_local import locanmf_position_decoder as d

    src = inspect.getsource(d._trial_features)
    assert "block_ids(" in src, "_trial_features must call the shared rule"
    assert "codes[k] != prev" not in src, "the old position-change-only rule is still inline"


def test_block_size_max_is_read_per_session_not_hardcoded():
    """It is a scheduler setting that could change at the rig, like the response window did."""
    import inspect

    from wfield_local import block_ids as b

    assert "gui_config.json" in inspect.getsource(b.block_size_max_for)
