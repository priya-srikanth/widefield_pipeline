"""The trial-matched pre-to-pre null must calibrate BOTH readings and must not fake either.

Three things are worth a test here, and they are the three ways this construction could mislead:

  * the match is on TRIALS, so a split that matches session count while missing trial count by a
    factor of five is a failure even though it looks matched;
  * the null must pay the SAME noise penalty the observation pays -- if it did not, the amplitude
    bias it exists to absorb would pass straight through;
  * a genuine rotation must still register, or the null has simply been made large enough to
    swallow everything.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import cd_overlap_null as con
from wfield_local import cd_trajectories as cdt


def _session(rng, ncomp, T, n_trials, basis, scale=1.0, t=None):
    """One session's ``(G, n)``: a shared time course along `basis` plus mean-of-n-trials noise."""
    t = np.linspace(-3.0, 4.0, T) if t is None else t
    course = scale * np.exp(-((t - 0.8) ** 2) / 0.5) * (t > -1.0)
    G = np.outer(basis, course)
    # the noise a mean over n trials carries, which is what makes a small session's norm bigger
    G = G + rng.normal(scale=1.0 / np.sqrt(n_trials), size=(ncomp, T))
    return (G.astype(np.float32), int(n_trials))


def test_the_split_matches_TRIALS_not_just_the_number_of_sessions():
    """The whole point of Priya's *"trial-matched?"*. Session counts are equal by construction here
    and the trial counts differ 10x between sessions, so a size-only match lands far off."""
    counts = [600, 600, 600, 80, 80, 80, 80, 90]
    rng = np.random.default_rng(0)
    iB, iA = con.trial_matched_split(counts, n_target=170.0, k_sessions=2, rng=rng)
    assert len(iB) == 2 and len(iA) == len(counts) - 2
    got = sum(counts[i] for i in iB)
    assert abs(got - 170) <= 20, got                     # two small sessions, not a 600
    assert not set(iB) & set(iA)                          # DISJOINT -- the null is not vs itself


def test_the_split_is_disjoint_and_covers_every_session():
    counts = [100, 200, 300, 400, 500]
    iB, iA = con.trial_matched_split(counts, 500.0, rng=np.random.default_rng(3))
    assert sorted([*iB, *iA]) == list(range(5))


def test_the_split_declines_a_pool_it_cannot_split():
    assert con.trial_matched_split([100], 100.0, rng=np.random.default_rng(1)) is None


def test_the_NULL_pays_the_SAME_noise_penalty_as_the_observation(_ncomp=20, _T=60):
    """THE TEST THAT MATTERS FOR THE AMPLITUDE FINDING. Every session here is drawn from ONE
    unchanging generator, so the true scale is 1.0 and the true overlap is 1.0 -- but the "epoch"
    sessions are SMALL, so their grand mean is noisy, its norm is inflated and its subspace is
    rotated by noise alone. A correctly matched null reproduces both, leaving no difference.
    """
    rng = np.random.default_rng(11)
    t = np.linspace(-3.0, 4.0, _T)
    b = rng.normal(size=_ncomp)
    b /= np.linalg.norm(b)
    # THE PRE-STROKE POOL MUST SPAN A RANGE, as the real one does (221-675 trials per session across
    # the cohort). A pool of uniformly LARGE sessions cannot match a small epoch at all -- that is a
    # property of whole-session matching, tested separately below, not of this null.
    pre = [_session(rng, _ncomp, _T, n, b, t=t) for n in (600, 550, 620, 95, 88, 92, 85)]
    ep = [_session(rng, _ncomp, _T, n, b, t=t) for n in (90, 80, 100)]     # same generator, few trials

    r = con.epoch_null(pre, ep, t, k=1, n_draw=60, rng=np.random.default_rng(2))
    assert r is not None
    # the match actually landed, or the rest of this test is measuring the wrong thing
    assert abs(np.median(r["n_B"]) - r["n_target"]) / r["n_target"] < 0.25
    # the epoch's raw ratio is inflated above the truth of 1.0 by noise alone ...
    assert np.median(r["obs_scale"]) > 1.0
    # ... and the matched null is inflated by the SAME amount, so the difference straddles zero
    assert abs(float(np.median(r["d_scale"]))) < 0.5 * (np.median(r["obs_scale"]) - 1.0) + 0.05
    # same for orientation: the observed overlap is below 1.0 although nothing rotated
    assert np.median(r["obs_cos"]) < 1.0
    assert float(np.median(r["d_cos"])) > -0.25


def test_a_REAL_rotation_still_registers_against_the_matched_null(_ncomp=20, _T=60):
    """The other failure mode: a null so generous that nothing is ever detectable. The epoch's
    shared mode is put on an ORTHOGONAL axis, matched on trials, and must read as worse."""
    rng = np.random.default_rng(12)
    t = np.linspace(-3.0, 4.0, _T)
    Q = np.linalg.qr(rng.normal(size=(_ncomp, 2)))[0]
    pre = [_session(rng, _ncomp, _T, n, Q[:, 0], t=t) for n in (600, 550, 620, 580, 500, 640)]
    ep = [_session(rng, _ncomp, _T, n, Q[:, 1], t=t) for n in (90, 80, 100)]

    r = con.epoch_null(pre, ep, t, k=1, n_draw=60, rng=np.random.default_rng(4))
    assert float(np.median(r["obs_cos"])) < float(np.median(r["null_cos"]))
    assert float(np.median(r["d_cos"])) < 0
    assert r["p_not_worse_cos"] < 0.05


def test_a_REAL_amplitude_increase_still_registers(_ncomp=20, _T=60):
    rng = np.random.default_rng(13)
    t = np.linspace(-3.0, 4.0, _T)
    b = rng.normal(size=_ncomp)
    b /= np.linalg.norm(b)
    pre = [_session(rng, _ncomp, _T, n, b, scale=1.0, t=t) for n in (600, 550, 620, 580, 500, 640)]
    ep = [_session(rng, _ncomp, _T, n, b, scale=2.0, t=t) for n in (90, 80, 100)]

    r = con.epoch_null(pre, ep, t, k=1, n_draw=60, rng=np.random.default_rng(5))
    assert float(np.median(r["d_scale"])) > 0.5
    assert r["p_not_bigger_scale"] < 0.05


def test_K_is_FIXED_across_the_comparison():
    """`subspace_overlap` normalises by the first basis's dimension and chance is K/n, so a null
    whose groups each chose their own K would differ from the observation in two ways at once."""
    rng = np.random.default_rng(14)
    T, ncomp = 60, 20
    t = np.linspace(-3.0, 4.0, T)
    b = rng.normal(size=ncomp)
    pre = [_session(rng, ncomp, T, n, b, t=t) for n in (600, 550, 620, 580)]
    ep = [_session(rng, ncomp, T, n, b, t=t) for n in (90, 80)]
    r = con.epoch_null(pre, ep, t, k=3, n_draw=10, rng=np.random.default_rng(6))
    assert r["k"] == 3
    # and the override reaches the basis itself, whatever the variance rule would have chosen
    G = cdt.pooled_mean(pre)
    assert cdt.condition_independent_modes(G, t, k=3).shape[1] == 3
    assert cdt.condition_independent_modes(G, t, k=1).shape[1] == 1


def test_the_report_names_the_ACHIEVED_match_not_the_requested_one():
    """A bad match must be visible in the output rather than absorbed into it."""
    rng = np.random.default_rng(15)
    T, ncomp = 60, 20
    t = np.linspace(-3.0, 4.0, T)
    b = rng.normal(size=ncomp)
    pre = [_session(rng, ncomp, T, n, b, t=t) for n in (600, 550, 620, 580)]
    ep = [_session(rng, ncomp, T, n, b, t=t) for n in (90, 80)]
    out = {"animal": "PS99", "align": "precue", "gate": "lick", "k": 1, "ncomp": ncomp,
           "chance": 1 / ncomp, "n_pre_sessions": len(pre),
           "n_pre_trials": sum(c[1] for c in pre),
           "epochs": {"acute": con.epoch_null(pre, ep, t, k=1, n_draw=20,
                                              rng=np.random.default_rng(7))}}
    txt = con.report(out)
    assert "170" in txt                       # the epoch's true trial total appears
    assert "NOT a p-value" in txt
    assert "matched null" in txt


def test_the_fractions_are_NOT_presented_as_p_values():
    """The draws resample which pre-stroke sessions stand in for E; they do not permute an
    exchangeable label, and they share sessions. Calling either output a p-value would overclaim,
    and the field names are what a reader of a saved result sees."""
    assert "not as a p-value" in con.__doc__
    assert "NOT a p-value" in con.report({"animal": "x", "align": "precue", "gate": "lick", "k": 1,
                                          "chance": 0.02, "n_pre_sessions": 0, "n_pre_trials": 0,
                                          "epochs": {}})
    rng = np.random.default_rng(18)
    T, ncomp = 60, 20
    t = np.linspace(-3.0, 4.0, T)
    b = rng.normal(size=ncomp)
    pre = [_session(rng, ncomp, T, n, b, t=t) for n in (600, 550, 620, 580)]
    ep = [_session(rng, ncomp, T, n, b, t=t) for n in (90, 80)]
    r = con.epoch_null(pre, ep, t, k=1, n_draw=20, rng=np.random.default_rng(8))
    assert "p_value" not in r and "p" not in r
    assert {"p_not_worse_cos", "p_not_bigger_scale"} <= set(r)


@pytest.mark.parametrize("n_target", [100.0, 500.0, 2000.0])
def test_the_matcher_gets_closer_than_a_random_split_at_every_target(n_target):
    counts = [600, 550, 620, 580, 500, 640, 90, 80, 100, 110]
    rng = np.random.default_rng(16)
    iB, _iA = con.trial_matched_split(counts, n_target, rng=rng)
    matched = abs(sum(counts[i] for i in iB) - n_target)
    rnd = np.random.default_rng(17)
    worse = 0
    for _ in range(30):
        k = len(iB)
        pick = rnd.permutation(len(counts))[:k]
        if abs(sum(counts[i] for i in pick) - n_target) > matched:
            worse += 1
    assert worse >= 20


def test_an_UNMATCHABLE_epoch_is_visible_in_the_reported_counts(_ncomp=20, _T=60):
    """Whole sessions are the only unit, so an epoch smaller than the smallest pre-stroke session
    CANNOT be matched -- pre-stroke sessions of 500-640 trials against an epoch total of 270 give a
    closest subset of 1610. The null still returns a number, so the MISMATCH has to be legible in the
    output rather than absorbed into it; that is what `n_B` and `n_target` are for, and it is why
    `report` prints the achieved match beside every row.

    This does not arise in the current cohort (smallest epoch total 454 against pre-stroke sessions
    from 221) and it is the first thing to check before reusing this construction elsewhere.
    """
    rng = np.random.default_rng(21)
    t = np.linspace(-3.0, 4.0, _T)
    b = rng.normal(size=_ncomp)
    pre = [_session(rng, _ncomp, _T, n, b, t=t) for n in (600, 550, 620, 580, 500, 640, 560)]
    ep = [_session(rng, _ncomp, _T, n, b, t=t) for n in (90, 80, 100)]
    r = con.epoch_null(pre, ep, t, k=1, n_draw=10, rng=np.random.default_rng(9))
    assert r is not None
    assert np.median(r["n_B"]) > 3 * r["n_target"]            # the mismatch is large ...
    out = {"animal": "PS99", "align": "precue", "gate": "lick", "k": 1, "ncomp": _ncomp,
           "chance": 1 / _ncomp, "n_pre_sessions": len(pre),
           "n_pre_trials": sum(c[1] for c in pre), "epochs": {"acute": r}}
    txt = con.report(out)
    assert f"{r['n_target']:.0f}" in txt                     # ... and both numbers are printed
    assert f"{np.median(r['n_B']):.0f}" in txt
