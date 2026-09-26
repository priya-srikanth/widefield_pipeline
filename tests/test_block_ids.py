"""Two adjacent blocks at the same position must not merge into one.

Priya, 2026-08-18. The pipeline started a new block only when the POSITION changed, so a far_L block
followed by another far_L block became a single CV group. Audited against the firmware's own
block_number: 118 of 4216 blocks (2.8%) across the 48 curated + 8/17 sessions.

These tests pin the rule and its documented limits, so a future simplification back to
"split on position change" fails loudly.

2026-09-26: they also pin WHERE an over-long run is divided, which used to be a free choice. It is
not one any more -- blocks are now the exchangeable unit of the block-label permutation nulls, so a
split that yields a piece the scheduler could not have run (8+1 for a run of 9, against a
`block_size_min` of 4) understates the null. See DECISIONS.md, 2026-09-26.
"""
from __future__ import annotations

import numpy as np

from wfield_local.block_ids import DEFAULT_BLOCK_SIZE_MAX, block_ids


def test_a_run_longer_than_block_size_max_is_split():
    """The regression this exists for: 11 trials at one position cannot be one block of max 8."""
    ids = block_ids(np.array([3] * 11), block_size_max=8)
    assert len(set(ids)) == 2
    assert list(ids) == [0] * 6 + [1] * 5


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


def test_a_run_of_nine_splits_evenly_and_not_eight_plus_one():
    """The 2026-09-26 regression. Left-chunking made 9 into 8+1, and 1 is not a block.

    `block_size_min` is 4, so the only decompositions of 9 the scheduler could have produced are 4+5
    and 5+4. A one-trial block matters because blocks are the exchangeable unit of the block-label
    permutation nulls: a unit of one trial is trial-level shuffling for that trial, which UNDERSTATES
    the null. Measured over the 177 balanced_block_cycles sessions, left-chunking produced an
    impossible decomposition on 286 of 491 over-long runs.
    """
    ids = block_ids(np.array([3] * 9), block_size_max=8)
    assert sorted(np.bincount(ids).tolist()) == [4, 5]


def test_no_piece_falls_outside_the_schedulers_own_bounds():
    """Every block must be one the scheduler could have run: within [block_size_min, max].

    Swept over every run length a merge can produce (two blocks, so up to 2*max). Longer than that
    is not a merge and `audit` reports it as damaged labels rather than this rule absorbing it.
    """
    from wfield_local.block_ids import DEFAULT_BLOCK_SIZE_MIN, split_lengths

    for bmin, bmax in ((4, 8), (5, 8), (DEFAULT_BLOCK_SIZE_MIN, DEFAULT_BLOCK_SIZE_MAX)):
        for n in range(bmin, 2 * bmax + 1):
            pieces = split_lengths(n, bmax)
            assert sum(pieces) == n
            assert all(p <= bmax for p in pieces), f"n={n} piece over max: {pieces}"
            # A run of n has a legal decomposition only if the fewest blocks that can COVER it can
            # also be filled to the minimum. With min 5 and max 8 a run of 9 cannot: the smallest
            # merge is 5+5. Such a run is damaged data, not a merge, and no splitter can rescue it --
            # `audit` reports it through `undersized_blocks`.
            k = -(-n // bmax)
            if k * bmin <= n:
                assert all(p >= bmin for p in pieces), f"n={n} piece under min: {pieces}"


def test_an_infeasible_run_still_splits_rather_than_raising():
    """min 5 / max 8 and a run of 9: no legal decomposition exists, and this must not raise.

    `audit` is where that becomes a reported fact. The splitter's job is to return the best
    available, because a caller mid-way through a session loop cannot do anything useful with an
    exception from one damaged run.
    """
    from wfield_local.block_ids import split_lengths

    assert sorted(split_lengths(9, 8)) == [4, 5]      # best available; 4 is under a min of 5


def test_block_size_min_is_read_per_session(tmp_path):
    """Two sessions use 5 rather than 4, so this is a per-session setting like its partner."""
    import json

    from wfield_local.block_ids import DEFAULT_BLOCK_SIZE_MIN, block_size_min_for

    d = tmp_path / "PS94_20260812_120000"
    d.mkdir()
    (d / "gui_config.json").write_text(json.dumps({"timing": {"block_size_min": 5}}))
    assert block_size_min_for(d) == 5
    assert block_size_min_for(tmp_path / "nope") == DEFAULT_BLOCK_SIZE_MIN


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


def test_block_size_max_reads_a_behaviour_directory_directly(tmp_path):
    """`spout_behavior` holds the log directory, not a session mapping.

    Accepting both keeps `block_size_max_for` the single reader of `timing.block_size_max`. The
    alternative was three lines of gui_config parsing in the behaviour path, which is how the block
    definition ends up with two homes that agree until one of them is edited.
    """
    import json

    from wfield_local.block_ids import DEFAULT_BLOCK_SIZE_MAX, block_size_max_for

    d = tmp_path / "PS94_20260812_120000"
    d.mkdir()
    (d / "gui_config.json").write_text(json.dumps({"timing": {"block_size_max": 6}}))
    assert block_size_max_for(d) == 6
    # a directory with no config falls back rather than raising
    assert block_size_max_for(tmp_path / "nope") == DEFAULT_BLOCK_SIZE_MAX


def test_a_run_of_nine_is_two_blocks():
    """Priya, 2026-08-28: any run of >= 9 at one position is by definition two blocks.

    This is the run-length detector the 2026-08-18 audit measured: it recovers ~92% of the
    adjacent same-position merges (108 of 118 over 48 curated + 8/17 sessions). The residual are
    4+4 merges landing at exactly `block_size_max`, which no run-length rule can separate from one
    genuine maximal block.
    """
    from wfield_local.block_ids import block_ids

    codes = np.array([3] * 9 + [1] * 4 + [3] * 8)
    b = block_ids(codes, 8)
    assert len(set(b[:9].tolist())) == 2, "a run of 9 stayed one block"
    assert len(set(b[13:].tolist())) == 1, "a run of exactly 8 was split"
    assert len(set(b.tolist())) == 4
