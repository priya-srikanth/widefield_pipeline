"""`cd_trajectories` -- the CD must be TIME-INVARIANT, and the poles must mean what they say.

The load-bearing property is `test_direction_is_one_weight_per_component`. `position_coding_directions`
fits in (component x time sub-bin) space -- "a weight per component PER MOMENT in the window", 90 x 4
for an ENL window -- and such a vector cannot be applied to a single frame. If this module ever
inherits that shape, every trajectory becomes a projection of frame t through weights fitted for a
different moment, which is not a coherent readout and would not raise.
"""
import numpy as np
import pytest

from wfield_local import cd_trajectories as cdt


def _sig(ncomp=12, T=4000, seed=0):
    return np.random.default_rng(seed).normal(scale=0.05, size=(ncomp, T))


def test_window_means_is_one_value_per_component_not_per_sub_bin():
    sig = _sig(ncomp=12)
    X = cdt.window_means(sig, [100, 500, 900], 62)
    assert X.shape == (3, 12)                       # NOT (3, 12 * bins)


def test_window_means_marks_a_window_that_runs_off_the_end():
    sig = _sig(ncomp=5, T=200)
    X = cdt.window_means(sig, [10, 190, -1, None], 62)
    assert np.isfinite(X[0]).all()
    assert np.isnan(X[1]).all() and np.isnan(X[2]).all() and np.isnan(X[3]).all()


def test_direction_is_one_weight_per_component():
    """The property that makes a frame-by-frame projection meaningful at all."""
    rng = np.random.default_rng(1)
    ncomp = 20
    y = rng.integers(0, 6, 400)
    tmpl = rng.normal(size=(6, ncomp))
    X = tmpl[y] + rng.normal(scale=0.5, size=(400, ncomp))
    dirs = cdt.fit_directions(X, y, list(range(6)))
    assert set(dirs) == set(range(6))
    for w, _p0, _p1 in dirs.values():
        assert w.shape == (ncomp,)
        assert np.linalg.norm(w) == pytest.approx(1.0, abs=1e-9)


def test_fit_directions_omits_a_position_it_cannot_fit():
    rng = np.random.default_rng(2)
    y = np.concatenate([np.zeros(200, int), np.ones(3, int)])       # position 1 has 3 trials
    X = rng.normal(size=(203, 8))
    assert 1 not in cdt.fit_directions(X, y, [0, 1])


def test_poles_put_a_position_P_trial_at_one_and_a_not_P_trial_at_zero():
    """0 = pre-stroke not-P, 1 = pre-stroke lick at P. The whole scale rests on this."""
    rng = np.random.default_rng(3)
    ncomp, T = 10, 3000
    tmpl = rng.normal(size=(6, ncomp)) * 2.0
    y = rng.integers(0, 6, 300)
    X = tmpl[y]                                                     # noiseless: the poles are exact
    dirs = cdt.fit_directions(X, y, list(range(6)))
    w, p0, p1 = dirs[0]
    # a constant "signal" holding position 0's pattern must project to 1.0 at every frame
    sig_P = np.repeat(tmpl[0][:, None], T, axis=1)
    tr = cdt.trajectory(sig_P, [1000], w, p0, p1, 50, 50)
    assert np.nanmean(tr) == pytest.approx(1.0, abs=1e-6)
    # ...and the not-P pole must project to 0.0. That pole is the mean over not-P TRIALS, weighted
    # by how many of each other position were drawn -- NOT the unweighted mean of the five
    # templates, which differs from it by the sampling imbalance.
    sig_n = np.repeat(X[y != 0].mean(0)[:, None], T, axis=1)
    assert np.nanmean(cdt.trajectory(sig_n, [1000], w, p0, p1, 50, 50)) == pytest.approx(0.0, abs=1e-6)


def test_trajectory_marks_a_trial_too_close_to_an_edge():
    sig = _sig(ncomp=6, T=500)
    w = np.ones(6) / np.sqrt(6)
    tr = cdt.trajectory(sig, [10, 250, 495, None], w, 0.0, 1.0, 60, 60)
    assert np.isnan(tr[0]).all() and np.isnan(tr[2]).all() and np.isnan(tr[3]).all()
    assert np.isfinite(tr[1]).all()


def test_the_trailing_window_ENDS_at_the_sample_it_is_plotted_at():
    """THE BUG THIS PINS. The first version took `full[n-1:]`, which is the window ending at
    i+n-1 -- a LEADING window, the whole trace shifted one window early, so the t=0 point showed
    the POST-cue response instead of the ENL. A step input localises it exactly.
    """
    fs, w = 10.0, 2.0                       # 20-sample window
    v = np.concatenate([np.zeros(100), np.ones(100)])
    out = cdt.smooth(v, fs, w, trailing=True)
    # the sample where the step occurs (index 100) sees a window of [81..100]: 1 of 20 samples high
    assert out[100] == pytest.approx(1 / 20, abs=1e-9)
    # one full window later it is entirely inside the step
    assert out[119] == pytest.approx(1.0, abs=1e-9)
    # and BEFORE the step nothing has leaked backwards -- this is what the bug violated
    assert out[99] == pytest.approx(0.0, abs=1e-12)
    assert out[80] == pytest.approx(0.0, abs=1e-12)


def test_smoothing_preserves_the_level_and_shrinks_the_edges_rather_than_padding():
    v = np.ones(500) * 3.0
    out = cdt.smooth(v, fs=31.23, seconds=0.5)
    # a constant stays constant EVERYWHERE, including the ends -- zero-padding would dip them
    assert np.allclose(out, 3.0)


def test_smoothing_reduces_variance_without_moving_the_mean():
    rng = np.random.default_rng(4)
    v = rng.normal(size=5000)
    out = cdt.smooth(v, fs=31.23, seconds=0.5)
    assert np.std(out) < 0.5 * np.std(v)
    assert np.mean(out) == pytest.approx(np.mean(v), abs=0.02)


def test_lick_alignment_draws_only_classes_that_have_a_lick():
    """A no-lick trial's lick time is an INFERENCE; a trajectory through it has a guessed x-axis."""
    assert cdt.LICK_ALIGNED_CLASSES == ("success",)
    assert set(cdt.CLASSES) - set(cdt.LICK_ALIGNED_CLASSES) == {"miss_working", "stopped"}


def test_span_frames_covers_spout_arrival_for_the_enl_window():
    """The spout arrives ~3 s before the cue; the ENL panel must reach back past it."""
    pre_n, _post = cdt.span_frames("precue", 31.23)
    assert pre_n / 31.23 >= 3.0


# ------------------------------------------------------------------------------------------
# THE SHARED-RESPONSE SUBSPACE. K=1 was measured to be insufficient: after orthogonalising
# against a single mode, PS95's shared column still ran -0.47 to +1.83 against a
# position-specific signal of +1.0 to +1.9 -- the same order as the thing being measured.
# ------------------------------------------------------------------------------------------

def test_cim_basis_is_orthonormal_and_ranked_by_variance():
    rng = np.random.default_rng(10)
    ncomp, T = 30, 400
    t = np.linspace(-3, 4, T)
    # three nested components with clearly decreasing power
    u = np.linalg.qr(rng.normal(size=(ncomp, 3)))[0]
    G = (10 * np.outer(u[:, 0], np.exp(-((t - 0.5) ** 2)))
         + 3 * np.outer(u[:, 1], np.exp(-((t - 1.5) ** 2)))
         + 1 * np.outer(u[:, 2], np.exp(-((t - 2.5) ** 2))))
    U = cdt.condition_independent_modes(G, t)
    assert U.shape[0] == ncomp and 1 <= U.shape[1] <= cdt.CIM_KMAX
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)


def test_cim_k_grows_with_the_variance_threshold():
    rng = np.random.default_rng(11)
    t = np.linspace(-3, 4, 300)
    u = np.linalg.qr(rng.normal(size=(25, 4)))[0]
    G = sum(w * np.outer(u[:, i], np.exp(-((t - c) ** 2)))
            for i, (w, c) in enumerate([(8, 0.0), (5, 1.0), (3, 2.0), (2, 3.0)]))
    k_lo = cdt.condition_independent_modes(G, t, var=0.50).shape[1]
    k_hi = cdt.condition_independent_modes(G, t, var=0.999).shape[1]
    assert k_lo < k_hi


def test_cim_k_is_capped():
    rng = np.random.default_rng(12)
    t = np.linspace(-3, 4, 300)
    G = rng.normal(size=(40, 300))                    # flat spectrum: variance never concentrates
    assert cdt.condition_independent_modes(G, t, var=0.999, kmax=3).shape[1] <= 3


def test_cim_returns_none_for_a_flat_grand_mean():
    t = np.linspace(-3, 4, 200)
    assert cdt.condition_independent_modes(np.zeros((12, 200)), t) is None


def test_subspace_overlap_is_one_for_itself_and_zero_for_an_orthogonal_pair():
    rng = np.random.default_rng(13)
    Q = np.linalg.qr(rng.normal(size=(20, 6)))[0]
    A, B = Q[:, :3], Q[:, 3:]
    assert cdt.subspace_overlap(A, A) == pytest.approx(1.0, abs=1e-9)
    assert cdt.subspace_overlap(A, B) == pytest.approx(0.0, abs=1e-9)


def test_orthogonalising_against_the_whole_subspace_leaves_nothing_along_it():
    """THE BUG A K=1 LOOP WOULD LEAVE: w orthogonal to the first mode but not the rest."""
    rng = np.random.default_rng(14)
    ncomp = 24
    U = np.linalg.qr(rng.normal(size=(ncomp, 4)))[0]
    y = rng.integers(0, 6, 600)
    X = rng.normal(size=(600, ncomp)) + 6.0 * (U @ rng.normal(size=(4, 600))).T   # shared + noise
    dirs = cdt.fit_directions(X, y, list(range(6)), cim=U)
    for w, _p0, _p1 in dirs.values():
        assert np.abs(U.T @ w).max() < 1e-8
        assert np.linalg.norm(w) == pytest.approx(1.0, abs=1e-8)


# ------------------------------------------------------------------------------------------
# THE REST REFERENCE -- w_P = mean(P) - a baseline, so the six directions no longer cancel.
# ------------------------------------------------------------------------------------------

def test_rest_referenced_directions_do_not_cancel_the_way_one_vs_rest_does():
    """The property the whole artifact rests on: one-vs-rest directions sum to ~0, these do not."""
    rng = np.random.default_rng(15)
    ncomp = 20
    tmpl = rng.normal(size=(6, ncomp))
    shared = rng.normal(size=ncomp) * 5.0              # a big condition-independent offset
    y = rng.integers(0, 6, 900)
    X = tmpl[y] + shared + rng.normal(scale=0.3, size=(900, ncomp))
    contrast = cdt.fit_directions(X, y, list(range(6)))
    vs_rest = cdt.directions_vs_rest(X, y, {None: shared}, list(range(6)))
    n_contrast = np.linalg.norm(np.stack([w for w, _, _ in contrast.values()]).sum(0))
    n_rest = np.linalg.norm(np.stack([w for w, _, _ in vs_rest.values()]).sum(0))
    assert n_contrast < 1.0                            # six unit vectors that nearly cancel
    assert n_rest > n_contrast                         # these do not have to


def test_rest_referenced_poles_put_the_baseline_at_zero_and_the_position_at_one():
    rng = np.random.default_rng(16)
    ncomp = 15
    tmpl = rng.normal(size=(6, ncomp)) * 2.0
    base = rng.normal(size=ncomp)
    y = rng.integers(0, 6, 600)
    X = tmpl[y] + base                                  # noiseless, so the poles are exact
    dirs = cdt.directions_vs_rest(X, y, {None: base}, list(range(6)))
    for p, (w, p0, p1) in dirs.items():
        assert float(base @ w) == pytest.approx(p0, abs=1e-9)
        assert float(X[y == p].mean(0) @ w) == pytest.approx(p1, abs=1e-9)
        assert p1 > p0                                  # the position sits ABOVE its baseline


def test_rest_reference_can_be_per_position():
    rng = np.random.default_rng(17)
    ncomp = 12
    tmpl = rng.normal(size=(6, ncomp))
    per = {p: rng.normal(size=ncomp) for p in range(6)}   # each position its own baseline
    y = rng.integers(0, 6, 600)
    X = np.stack([tmpl[v] + per[int(v)] for v in y])
    dirs = cdt.directions_vs_rest(X, y, per, list(range(6)))
    assert len(dirs) == 6
    for p, (w, p0, _p1) in dirs.items():
        assert float(per[p] @ w) == pytest.approx(p0, abs=1e-9)


def test_subspace_chance_is_K_over_n_and_random_subspaces_hit_it():
    """0.61 is not "a third rotated away" -- unrelated subspaces overlap at K/n, not 0."""
    rng = np.random.default_rng(18)
    ncomp, k = 95, 2
    assert cdt.subspace_chance(ncomp, k) == pytest.approx(k / ncomp)
    got = [cdt.subspace_overlap(np.linalg.qr(rng.normal(size=(ncomp, k)))[0],
                                np.linalg.qr(rng.normal(size=(ncomp, k)))[0])
           for _ in range(200)]
    assert np.mean(got) == pytest.approx(k / ncomp, abs=0.01)
    # and the measured 0.61 is far above that -- the mode is CONSERVED, however much leaks through
    assert 0.61 > 10 * cdt.subspace_chance(ncomp, k)


def test_orthogonalising_scales_by_the_ORIGINAL_gap_not_the_collapsed_one():
    """THE close_center BUG. Recomputing the poles after rotating rescales whatever residual
    survives back up to 1, so a direction that lost most of its separation is INFLATED rather than
    drawn small. Measured on PS95: close_center's gap fell 0.737 -> 0.448 while close_L's fell only
    1.565 -> 1.383, making close_center's denominator 3.1x smaller and everything divided by it 3.1x
    bigger.

    Here one position is built to lie almost entirely in the shared subspace, so orthogonalising
    should leave it with a SMALL surviving fraction and a correspondingly small trace -- not a
    renormalised one.
    """
    rng = np.random.default_rng(19)
    ncomp = 24
    U = np.linalg.qr(rng.normal(size=(ncomp, 2)))[0]
    tmpl = rng.normal(size=(6, ncomp))
    tmpl[0] = 6.0 * U[:, 0] + 0.2 * tmpl[0]          # position 0 is nearly all shared mode
    y = rng.integers(0, 6, 1200)
    X = tmpl[y] + rng.normal(scale=0.3, size=(1200, ncomp))

    plain = cdt.fit_directions(X, y, list(range(6)))
    orth, surviving = cdt.fit_directions(X, y, list(range(6)), cim=U, with_surviving=True)

    # the contaminated position loses far more of its separation than the others
    assert surviving[0] < 0.5
    assert min(surviving[p] for p in range(1, 6)) > surviving[0]

    # and the SCALE it is divided by is the unrotated gap, so the panel draws small rather than
    # being rescaled back to 1
    g_plain = plain[0][2] - plain[0][1]
    assert (orth[0][2] - orth[0][1]) == pytest.approx(g_plain, rel=1e-9)


def test_surviving_is_absent_when_not_orthogonalising():
    rng = np.random.default_rng(20)
    y = rng.integers(0, 6, 400)
    X = rng.normal(size=(400, 10))
    _dirs, surviving = cdt.fit_directions(X, y, list(range(6)), with_surviving=True)
    assert surviving == {}
