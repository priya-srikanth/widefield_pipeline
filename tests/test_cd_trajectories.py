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
