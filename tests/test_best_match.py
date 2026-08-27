"""The whole-row summary: what the diagonal alone cannot say.

A diagonal of 0.2 cannot separate "the code is gone" from "the code moved to far_L" -- 0.2 against
everything, and 0.2 against itself with 0.7 elsewhere, are the same number. `_best_match` uses all
six entries of a row and separates them.
"""
import numpy as np

from wfield_local import grant_figures as gf

L = gf.CONF_LABELS
FAR_R, FAR_L = L.index("far_R"), L.index("far_L")


def test_the_diagonal_cannot_tell_these_apart_but_the_row_can():
    flat = np.full((6, 6), 0.2)
    moved = np.full((6, 6), 0.1)
    np.fill_diagonal(moved, 0.2)
    moved[FAR_R, FAR_L] = 0.7

    assert flat[FAR_R, FAR_R] == moved[FAR_R, FAR_R], "same diagonal, by construction"

    _b_flat, r_flat = gf._best_match(flat)
    b_moved, r_moved = gf._best_match(moved)
    assert b_moved[FAR_R] == FAR_L, "the row names the substitute"
    assert r_moved[FAR_R] == 2, "the true position is second behind it"
    assert r_flat[FAR_R] == 1, "a flat row leaves the diagonal tied-best"


def test_ties_go_to_the_diagonal_and_never_invent_a_move():
    """A flat row means no particular substitute. A bare argmax would return whichever position
    comes first in DISPLAY_ORDER and report a substitution that does not exist -- and would then
    disagree with `rank`, which calls the diagonal tied-best."""
    flat = np.full((6, 6), 0.2)
    best, rank = gf._best_match(flat)
    assert list(best) == list(range(6)), f"every row must match itself on a tie, got {best}"
    assert list(rank) == [1] * 6


def test_direction_is_respected_for_distances():
    """For a DISTANCE the best match is the smallest entry, not the largest. Getting this backwards
    would report every position as matching whichever it is furthest from."""
    D = np.full((6, 6), 1.0)
    D[FAR_R, FAR_L] = 0.1                     # post far_R is CLOSEST to pre far_L
    D[FAR_R, FAR_R] = 0.9
    best, _r = gf._best_match(D, higher_is_better=False)
    assert best[FAR_R] == FAR_L
    best_wrong, _r2 = gf._best_match(D, higher_is_better=True)
    assert best_wrong[FAR_R] != FAR_L, "the direction flag must actually change the answer"


def test_rows_with_too_little_data_are_declined_not_guessed():
    M = np.full((6, 6), np.nan)
    M[0, 0] = 0.5                              # a single usable entry is not a comparison
    best, rank = gf._best_match(M)
    assert best[0] == -1 and not np.isfinite(rank[0])
    assert all(b == -1 for b in best)


def test_invariance_to_a_monotone_change_across_a_row():
    """THE PROPERTY THAT MOTIVATES THIS SUMMARY. The uniform row shifts that dominate the distance
    figures are amplitude, not resemblance; argmax and rank cannot be moved by them, while the
    diagonal moves with every one."""
    rng = np.random.default_rng(0)
    M = rng.uniform(0, 1, (6, 6))
    shifted = M.copy()
    shifted[FAR_R] += 0.4                      # whole row moves together
    b0, r0 = gf._best_match(M)
    b1, r1 = gf._best_match(shifted)
    assert list(b0) == list(b1)
    assert np.allclose(r0, r1, equal_nan=True)
    assert M[FAR_R, FAR_R] != shifted[FAR_R, FAR_R], "the diagonal DID move"


def test_the_pre_ceiling_counts_sessions_not_one_averaged_matrix():
    """THE BUG PRIYA SPOTTED FROM A "1". Figure 10's PRE panel counted argmax over the MEAN of the
    eleven leave-one-out matrices, so every row contributed exactly one count and the ceiling read
    100% match self in all four animals -- printed directly under a caption telling the reader never
    to compare against 100%.

    Averaging removes the per-session noise that the post-stroke columns still carry. Here: eleven
    noisy matrices, half of which get far_R wrong, whose AVERAGE gets it right.
    """
    rng = np.random.default_rng(0)
    mats = []
    for k in range(11):
        M = np.full((6, 6), 0.1)
        np.fill_diagonal(M, 0.5)
        # far_R goes to far_L on a MINORITY of sessions -- often enough that the ceiling is not
        # perfect, rarely enough that the average still favours itself: 4 x 0.9 + 7 x 0.1 over 11
        # is 0.39, under the diagonal's 0.5.
        if k % 3 == 0:
            M[FAR_R, FAR_L] = 0.9
        M = M + rng.normal(0, 0.01, M.shape)
        mats.append(M)

    avg = np.mean(np.stack(mats), axis=0)
    b_avg, _r = gf._best_match(avg)
    assert b_avg[FAR_R] == FAR_R, "the AVERAGE matrix matches itself -- hence a 6/6 ceiling"

    per_session = [gf._best_match(M)[0][FAR_R] for M in mats]
    hits = sum(1 for j in per_session if j == FAR_R)
    assert 0 < hits < len(mats), (
        f"counting sessions individually must give a REACHABLE ceiling, got {hits}/{len(mats)}")
