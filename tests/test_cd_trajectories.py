"""`cd_trajectories` -- the CD must be TIME-INVARIANT, and the poles must mean what they say.

The load-bearing property is `test_direction_is_one_weight_per_component`. `position_coding_directions`
fits in (component x time sub-bin) space -- "a weight per component PER MOMENT in the window", 90 x 4
for an ENL window -- and such a vector cannot be applied to a single frame. If this module ever
inherits that shape, every trajectory becomes a projection of frame t through weights fitted for a
different moment, which is not a coherent readout and would not raise.
"""
from pathlib import Path

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


# ------------------------------------------------------------------ persisted figure data


def _fake_result(**over):
    """A result with the SHAPE `analyse_animal` returns -- tuple trace keys, int surviving keys."""
    t = 7
    res = {"animal": "PS99", "align": "precue", "method": "dom", "basis_id": "deadbeefcafe",
           "ncomp": 12, "fs": 10.0, "span": (-3.0, 4.0), "pre_n": 3, "post_n": 4,
           "positions": [0, 1], "n_sessions": 2, "errors": [], "win_s": 2.0,
           "standardised": True, "smooth_s": cdt.SMOOTH_S, "orth": True, "gate": "lick",
           "reference": "contrast", "fit_on": ["success"], "cim_k": 2,
           "surviving": {0: 0.61, 1: 0.88}, "anchor": 1.0,
           "cim_cos": {"pre": 1.0, "acute": 0.68}, "cim_chance": 0.021,
           "cim_scale": {"pre": 1.0, "acute": 1.51}, "traces": {}}
    for ep in ("pre", "acute"):
        for cd in (0, 1):
            for tr in (0, 1):
                res["traces"][(ep, cd, tr)] = {
                    "mean": np.linspace(0, 1, t) + cd, "n": 40 + tr,
                    "iqr": np.ones(t), "lo": np.zeros(t), "hi": np.full(t, 2.0)}
    res.update(over)
    return res


def test_a_saved_result_round_trips_with_TUPLE_trace_keys_and_INT_position_keys(tmp_path):
    """The one thing JSON cannot hold. `res["traces"][(ep, p, p)]` is how every layout reads a
    trace, and a dict whose keys came back as the strings `"pre|0|0"` would not raise -- it would
    draw six empty panels and look like an animal with no data.
    """
    res = _fake_result()
    cdt.save_result(res, tmp_path)
    got = cdt.load_result(tmp_path, "PS99", "precue", orth=True)

    assert set(got["traces"]) == set(res["traces"])
    assert ("pre", 0, 0) in got["traces"]
    for key, d in res["traces"].items():
        np.testing.assert_allclose(got["traces"][key]["mean"], d["mean"])
        assert got["traces"][key]["n"] == d["n"]
    assert got["surviving"] == res["surviving"]          # int keys, not "0"/"1"
    assert got["positions"] == [0, 1]
    assert got["cim_scale"]["acute"] == pytest.approx(1.51)


def test_a_saved_result_is_enough_to_DRAW_without_recomputing(tmp_path):
    """The point of the dump: a layout must accept it with nothing else on hand."""
    cdt.save_result(_fake_result(), tmp_path)
    got = cdt.load_result(tmp_path, "PS99", "precue", orth=True)
    png = cdt.figure_epochs(got, tmp_path / "replot.png")
    assert Path(png).exists() and Path(png).stat().st_size > 5000


def test_a_STALE_dump_raises_rather_than_being_redrawn(tmp_path):
    """The `COURSE_VERSION` lesson applied to the dump. A dump written under a different smoothing
    width would otherwise be redrawn beneath a title quoting the CURRENT one -- the single failure
    mode persistence adds over recomputing.
    """
    cdt.save_result(_fake_result(), tmp_path)
    old = cdt.SMOOTH_S
    try:
        cdt.SMOOTH_S = old + 0.1
        with pytest.raises(ValueError, match="STALE"):
            cdt.load_result(tmp_path, "PS99", "precue", orth=True)
    finally:
        cdt.SMOOTH_S = old
    assert cdt.load_result(tmp_path, "PS99", "precue", orth=True) is not None


def test_load_result_is_None_when_nothing_was_ever_saved(tmp_path):
    """Absent and stale are DIFFERENT facts -- None means "never rendered", the raise means "do not
    trust this one". `warn_if_store_moved` learned the same distinction the hard way (rule 10)."""
    assert cdt.load_result(tmp_path, "PS99", "precue") is None


def test_orth_and_gate_do_not_collide_in_the_saved_name(tmp_path):
    """Four analyses share an output directory; one tag per (animal, align, method, reference,
    gate, orth) or the last one written wins and the figures silently pair up wrongly."""
    tags = {cdt.result_tag("PS95", a, "dom", r, g, o)
            for a in ("precue", "cue") for r in cdt.REFERENCES
            for g in cdt.GATES for o in (False, True)}
    assert len(tags) == 2 * len(cdt.REFERENCES) * len(cdt.GATES) * 2


def test_the_reference_label_is_the_LOOKUP_not_its_source_text():
    """The inlined `f"{{...}}[res.get(...)]"` printed the dict literal plus the subscript as plain
    text, so every title on the share named no reference at all."""
    for ref, want in (("contrast", "OTHER POSITIONS"), ("restw", "ITS OWN REST")):
        title = cdt._suptitle(_fake_result(reference=ref))
        assert want in title
        assert "res.get(" not in title
        assert "'restw':" not in title


# ------------------------------------------------------------------ the three windows


def test_the_FIT_WINDOW_is_the_one_each_alignment_names():
    """Priya, 2026-09-25: *"make sure the windows are -2 to 0 for pre-cue, 0-2 cue-aligned for cue,
    0-2 lick-aligned for lick"*.

    `fit_window_mask` is the single definition and both `_anchor` and `cd_migration` read it. Pinned
    here because getting it wrong does not raise -- it silently measures the post-cue period on a
    pre-cue direction, which is exactly what `cd_migration` did until 2026-09-25.
    """
    fs = 10.0
    t = np.arange(-30, 40) / fs
    pre = cdt.fit_window_mask(t, "precue", 2.0)
    assert t[pre].min() == pytest.approx(-2.0) and t[pre].max() < 0.0
    for al in ("cue", "lick"):
        m = cdt.fit_window_mask(t, al, 2.0)
        assert t[m].min() == pytest.approx(0.0), al
        assert t[m].max() < 2.0, al


def test_the_LICK_direction_is_fitted_on_the_LICK_not_the_cue(monkeypatch):
    """THE DEFECT THIS PINS, measured 2026-09-25: `cos(w_cue, w_lick)` was 1.000000 at every
    position in two animals because `session_arms` set `ref0 = c0` for everything but `precue` -- so
    the lick arm was the cue arm replotted. `fit` must be the FIRST LICK for `lick`, and it must
    equal `at`, since the window starts where the trace is centred.
    """
    import types

    cue = np.array([100, 200, 300], int)
    licks = np.array([120, 232, 355], int)          # one lick after each cue
    codes = np.array([0, 1, 2], int)

    def fake_categorize(_s, _args=None, with_licks=True):
        cat = ["engaged"] * 3
        return (codes, cat, None, None, cue, np.array([True] * 3), licks,
                cue - 30)

    monkeypatch.setattr("wfield_local.nolick_decoder.categorize", fake_categorize)
    args = types.SimpleNamespace(post_s=2.0, fs=10.0, align="lick")
    basis = types.SimpleNamespace(basis_id="x", ncomp=4)
    arms = cdt.session_arms({"label": "PS99_0101"}, args, basis, "lick")
    assert arms["success"]["fit"] == arms["success"]["at"], (
        "the lick window must START at the lick it is centred on")
    assert list(arms["success"]["at"]) == list(licks)

    arms_cue = cdt.session_arms({"label": "PS99_0101"}, args, basis, "cue")
    assert list(arms_cue["success"]["fit"]) == list(cue)
    assert arms_cue["success"]["fit"] != arms["success"]["fit"], (
        "cue and lick must not fit the same window -- that was the bug")


def test_the_arms_cache_key_MOVES_when_session_arms_changes_meaning():
    """The key names the alignment and the args, and neither moves when `fit` changes underneath --
    the same failure `COURSE_VERSION` exists for. `ARMS_VERSION` has to reach the digest."""
    from wfield_local.locanmf_frozen_decoder import _args

    a = _args(source="roi", align="lick", post_s=2.0)
    k1 = cdt.arms_cache_kind(a, "lick")
    old = cdt.ARMS_VERSION
    try:
        cdt.ARMS_VERSION = old + 1
        k2 = cdt.arms_cache_kind(a, "lick")
    finally:
        cdt.ARMS_VERSION = old
    assert k1 != k2, "ARMS_VERSION is defined but never reaches the cache key"


def test_the_FEATURES_cache_key_moves_with_ARMS_VERSION():
    """THE FIX THAT DID NOT TAKE. `session_arms` was corrected so the lick window starts at the lick,
    `ARMS_VERSION` was bumped, the re-render ran clean -- and every lick anchor came back identical to
    two decimals, because the WINDOW MEANS live under their own key which names the alignment and the
    window LENGTH but nothing about where the window STARTS.

    Three layers of this arm have now each served stale numbers that looked plausible: the courses
    (`COURSE_VERSION`), the arms (`ARMS_VERSION`), and the features between them.
    """
    import types

    basis = types.SimpleNamespace(basis_id="abcdef123456")
    k1 = cdt.fit_cache_kind("lick", ("success",), basis, 62)
    old = cdt.ARMS_VERSION
    try:
        cdt.ARMS_VERSION = old + 1
        k2 = cdt.fit_cache_kind("lick", ("success",), basis, 62)
    finally:
        cdt.ARMS_VERSION = old
    assert k1 != k2


def test_every_module_builds_the_features_key_the_SAME_way():
    """It was an f-string in three modules, so a version bumped in one would move one reader and
    leave two on the old entries -- which is worse than not bumping it at all, because the three
    would then disagree about what the same session's features are."""
    import pathlib

    root = pathlib.Path(cdt.__file__).parent.parent
    offenders = []
    for py in [*(root / "wfield_local").glob("*.py"), *(root / "scripts").rglob("*.py")]:
        if py.name == "cd_trajectories.py":
            continue                      # the DEFINITION lives here; everyone else must call it
        if 'f"cdfit' in py.read_text(encoding="utf-8"):
            offenders.append(py.name)
    assert not offenders, (
        f"these build the features cache key by hand instead of calling `fit_cache_kind`: "
        f"{offenders}")


def test_the_window_rule_is_the_DECODERS_and_matches_them_on_every_alignment():
    """Priya, 2026-09-25: *"minimize redundancy by using the same or similar gates as the decoders,
    and use the same module/function"*.

    `session_arms` no longer restates where a window starts; it calls
    `locanmf_position_decoder.window_start`. This pins the rule itself, since the copy that used to
    live in `session_arms` is exactly what diverged: it agreed for `precue` and `cue` and put `lick`
    at the cue, making the lick coding direction identical to the cue one.
    """
    from wfield_local.locanmf_position_decoder import window_start

    ls = np.array([50, 130, 250], float)
    # cue: the cue itself
    assert window_start("cue", 100, 130, 70, ls, 20) == 100
    # lick: the FIRST LICK, never the cue
    assert window_start("lick", 100, 130, 70, ls, 20) == 130
    # lick with no lick recorded: dropped, not placed at a guess
    assert window_start("lick", 100, -1, 70, ls, 20) is None
    # a cue outside imaging coverage is dropped whatever the alignment
    for al in ("precue", "cue", "lick"):
        assert window_start(al, -1, 130, 70, ls, 20) is None


def test_session_arms_does_not_restate_the_window_rule():
    """Two copies of a rule always agree at first. This repo has collapsed six copies of a decoder
    recipe, three of a lick discriminator and two of an engaged cut; this is the seventh."""
    import inspect

    src = inspect.getsource(cdt.session_arms)
    assert "window_start(" in src
    assert "precue_window_start(" not in src, (
        "session_arms is applying the pre-cue rule itself again instead of going through "
        "`window_start`")
