"""The block bootstrap's claims, on synthetic data where the truth is known.

The point of resampling BLOCKS rather than trials is that trials adjacent in time are correlated, so
an i.i.d. trial bootstrap reports intervals that are too narrow. That is an empirical claim about the
estimator and it is checked here rather than asserted in a docstring.
"""
import numpy as np
import pytest

from wfield_local import grant_figures as gf

LABELS = gf.CONF_LABELS


def _mus(rng, dim=20):
    """One true pattern per position, SHARED across sessions -- otherwise nothing is preserved from
    day to day and every cross-session correlation is zero by construction."""
    return {q: rng.normal(0, 1, size=dim) for q in LABELS}


def _session(rng, n_blocks=12, per_block=6, dim=20, block_sd=1.0, trial_sd=0.3, mus=None):
    """One session with REAL block structure: a per-block offset shared by its trials.

    That shared offset is the autocorrelation an i.i.d. trial bootstrap is blind to.
    """
    x, blk = {}, {}
    for q in LABELS:
        rows, ids = [], []
        mu = rng.normal(0, 1, size=dim) if mus is None else mus[q]
        for b in range(n_blocks):
            off = rng.normal(0, block_sd, size=dim)          # shared by the whole block
            rows.append(mu + off + rng.normal(0, trial_sd, size=(per_block, dim)))
            ids.append(np.full(per_block, b))
        x[q] = np.vstack(rows)
        blk[q] = np.concatenate(ids)
    return x, blk


def test_runs_to_blocks_splits_on_session_and_position():
    sess = np.array([0, 0, 0, 0, 1, 1])
    pos = np.array(["a", "a", "b", "b", "b", "b"])
    b = gf._runs_to_blocks(sess, pos)
    assert len(set(b.tolist())) == 3, b            # (0,a), (0,b), (1,b)
    assert b[0] == b[1] and b[2] == b[3] and b[4] == b[5]
    assert np.all(b < 0), "synthetic ids must be negative so they cannot collide with real ones"


def test_block_boot_returns_positions_and_respects_min_trials():
    rng = np.random.default_rng(0)
    x, blk = _session(rng)
    out = gf._block_boot(x, blk, np.random.default_rng(1))
    assert set(out) <= set(LABELS) and out
    for q, Z in out.items():
        assert Z.shape[1] == x[q].shape[1]
    # a session with almost nothing in it yields no usable position rather than a 2-trial mean
    tiny = {q: x[q][:2] for q in LABELS}
    tiny_b = {q: blk[q][:2] for q in LABELS}
    assert gf._block_boot(tiny, tiny_b, np.random.default_rng(2), min_trials=8) == {}


def test_block_bootstrap_is_wider_than_an_iid_trial_bootstrap():
    """THE REASON FOR BLOCKS. With a per-block offset shared by six trials, an i.i.d. trial
    bootstrap sees six independent samples where there is really one, and understates the spread."""
    rng = np.random.default_rng(3)
    x, blk = _session(rng, n_blocks=12, per_block=6, block_sd=1.2, trial_sd=0.2)
    q = LABELS[0]

    block_means, trial_means = [], []
    r1, r2 = np.random.default_rng(4), np.random.default_rng(5)
    for _ in range(300):
        got = gf._block_boot({q: x[q]}, {q: blk[q]}, r1)
        if q in got:
            block_means.append(got[q].mean(0)[0])
        idx = r2.integers(0, len(x[q]), size=len(x[q]))
        trial_means.append(x[q][idx].mean(0)[0])

    sd_block = float(np.std(block_means))
    sd_trial = float(np.std(trial_means))
    assert sd_block > 1.5 * sd_trial, (
        f"block bootstrap sd {sd_block:.4f} should markedly exceed the i.i.d. trial one "
        f"{sd_trial:.4f} when trials within a block share an offset")


def test_delta_ci_holds_sessions_fixed_and_brackets_a_known_shift(monkeypatch):
    """Sessions are NOT resampled: the same pre-stroke session set enters every draw. And a day
    built from the pre-stroke distribution should give a delta interval covering zero.

    THE BOOTSTRAP CACHE IS OFF HERE. `_delta_diag_ci` memoises each day under a digest of its
    inputs, and this fixture is deterministic -- so the second run of this test would replay a
    stored record instead of recomputing it, and the guard that caught the leave-one-session-out
    bug would stop executing the arithmetic it was written to check.
    """
    monkeypatch.setenv("WIDEFIELD_NO_CACHE", "1")
    rng = np.random.default_rng(6)
    mus = _mus(rng)
    pre_x, pre_b, day_x, day_b = {}, {}, {}, {}
    for s in ("0806", "0807", "0809"):
        pre_x[s], pre_b[s] = _session(rng, block_sd=0.4, trial_sd=0.3, mus=mus)
    # the "day" is drawn from the SAME distribution as the pre-stroke sessions, so nothing moved
    day_x[1], day_b[1] = _session(rng, block_sd=0.4, trial_sd=0.3, mus=mus)

    def mats(pat, ref):
        return gf._corr_matrix(gf._means(pat), gf._means(ref))

    # `_delta_diag_ci` now takes a FACTORY and SEED PARTS, not a built matrix function and a live
    # generator: it seeds and caches per DAY, so it has to make the generator itself. The property
    # under test is unchanged -- sessions held fixed, an unshifted day covering zero.
    ci = gf._delta_diag_ci(lambda _an, _rng: mats, (pre_x, day_x), (pre_b, day_b), "PS94", [1],
                           ("PS94", "cue", "lick", "test-delta"), n_boot=60)
    assert 1 in ci
    lo, hi, med = ci[1]["mean"]
    assert lo < hi and lo <= med <= hi
    assert lo <= 0.0 <= hi, f"an unshifted day should cover zero, got ({lo:.3f}, {hi:.3f})"

    # PER-POSITION RECORD, which figure 9 plots. Averaging the scalar mean per session first would
    # give the same overall number and no breakdown -- and the per-position trajectory is what the
    # deficit is about.
    pos = ci[1]["pos"]
    assert set(pos) <= set(LABELS) and len(pos) >= 4
    for q, (qlo, qhi, qmed) in pos.items():
        assert qlo < qhi and qlo <= qmed <= qhi, q
        assert qlo <= 0.0 <= qhi, f"{q}: unshifted day should cover zero, got ({qlo:.2f}, {qhi:.2f})"

    # the mean must sit inside the spread of the per-position medians, not outside it
    meds = [v[2] for v in pos.values()]
    assert min(meds) - 0.2 <= med <= max(meds) + 0.2


@pytest.mark.parametrize("field", ["X", "blk"])
def test_session_trials_mask_is_shared_between_fields(field):
    """The bootstrap is only valid if the block vector lines up row-for-row with the data. The mask
    is computed once inside _session_trials, so the two selections cannot drift apart."""
    n, dim = 40, 7
    rng = np.random.default_rng(8)
    bd = {
        "XE": rng.normal(size=(n, dim)),
        "en": np.array([LABELS[i % len(LABELS)] for i in range(n)]),
        "GE": np.zeros(n, int),
        "BE": np.arange(n, dtype=np.int64) // 4,
        "XU": np.zeros((0, dim)),
        "un": np.zeros(0, str),
        "GU": np.zeros(0, int),
        "BU": np.zeros(0, np.int64),
        "not_eng": np.zeros(0, bool),
    }
    got = gf._session_trials(bd, 0, LABELS[0], "lick", field)
    n_rows = len(gf._session_trials(bd, 0, LABELS[0], "lick"))
    assert len(got) == n_rows
