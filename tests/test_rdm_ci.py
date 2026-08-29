"""`_rdm_ci` -- the intervals figures 8b and 8g draw -- on synthetic data where truth is known.

8b is the arbiter for anything about geometry: it is the one measure in the grant set that a global
amplitude change cannot move. It shipped with no uncertainty at all, so a post-stroke session
reading 0.82 against a ceiling of 0.90 gave the reader no way to tell a real loss from nothing.
These tests check that the interval says "changed" when the geometry changed and "cannot tell" when
it did not -- and, critically, that a pure GAIN change moves neither.
"""
import numpy as np
import pytest

from wfield_local import grant_figures as gf

LABELS = gf.CONF_LABELS
DIM = 24


def _mus(rng, dim=DIM):
    return {q: rng.normal(0, 1, size=dim) for q in LABELS}


def _session(rng, mus, n_blocks=10, per_block=6, block_sd=0.35, trial_sd=0.6, gain=1.0):
    x, blk = {}, {}
    for q in LABELS:
        rows, ids = [], []
        for b in range(n_blocks):
            off = rng.normal(0, block_sd, size=DIM)
            rows.append(gain * mus[q] + off + rng.normal(0, trial_sd, size=(per_block, DIM)))
            ids.append(np.full(per_block, b))
        x[q] = np.vstack(rows)
        blk[q] = np.concatenate(ids)
    return x, blk


def _install(monkeypatch, pre_specs, day_specs, seed=0):
    """Wire synthetic sessions in behind `_collect_7` for ONE animal."""
    rng = np.random.default_rng(seed)
    mus = _mus(rng)
    pre_x, pre_b, day_x, day_b = {}, {}, {}, {}
    for i, kw in enumerate(pre_specs):
        x, b = _session(rng, mus, **kw)
        pre_x[f"p{i}"], pre_b[f"p{i}"] = x, b
    for d, kw in day_specs.items():
        use = dict(kw)
        m = use.pop("mus", mus)
        x, b = _session(rng, m, **use)
        day_x[d], day_b[d] = x, b
    an = gf.ANIMALS[0]
    days = sorted(day_x)

    def fake(align, variant, min_trials, field="X"):
        s = (pre_b, day_b) if field == "blk" else (pre_x, day_x)
        return {an: s}, days

    monkeypatch.setattr(gf, "_collect_7", fake)
    _clear_caches()
    return an, mus, days


def _clear_caches():
    """EVERY cache keyed on (align, variant, min_trials), not just the one under test.

    `_rdm_ci` anchors its intervals on `_rdm_rows`, and both memoise on the same key. Clearing only
    one leaves a stale point estimate from the PREVIOUS test driving this one's interval -- which is
    how the scrambled-geometry case first came out as "no change": correct code, a ceiling computed
    from someone else's data.
    """
    for fn in (gf._rdm_ci, gf._rdm_rows, gf._enc_ci, gf._enc_tables):
        fn.cache_clear()


@pytest.fixture(autouse=True)
def _clear():
    _clear_caches()
    yield
    _clear_caches()


def test_unchanged_geometry_gives_a_delta_interval_covering_zero(monkeypatch):
    """Post-stroke sessions drawn from the SAME means as pre. The figure must not call that a
    change -- a delta interval excluding zero here would box cells in every animal."""
    an, _mu, days = _install(monkeypatch, [{}] * 4, {1: {}, 2: {}}, seed=1)
    out, got_days = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    assert got_days == days and an in out
    for d in days:
        iv = out[an][d]["dwhole"]
        assert iv[0] < 0 < iv[1], f"day {d}: {iv} excludes zero with no change present"
        assert not gf._excludes_zero(iv)


def test_scrambled_geometry_is_called_changed(monkeypatch):
    """Post-stroke means REASSIGNED between positions: the RDM's arrangement really is different,
    and an interval that still covered zero would mean the figure can never detect anything."""
    rng = np.random.default_rng(7)
    other = {q: rng.normal(0, 1, size=DIM) for q in LABELS}
    an, _mu, days = _install(monkeypatch, [{}] * 4,
                             {1: {"mus": other}, 2: {"mus": other}}, seed=2)
    out, _d = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    ivs = [out[an][d]["dwhole"] for d in days]
    assert all(iv[1] < 0 for iv in ivs), f"a scrambled geometry must read as a LOSS, got {ivs}"


def test_a_pure_gain_change_moves_neither_the_estimate_nor_the_interval(monkeypatch):
    """THE PROPERTY 8b EXISTS FOR. Tripling every PATTERN while leaving the noise alone multiplies
    every crossnobis distance by nine, and a correlation between RDMs is invariant to that. If the
    interval moved with it, the figure would be reporting amplitude while claiming to be immune.

    Deliberately the harder version: scaling the noise as well would scale the whitener with it and
    leave the distances numerically identical, which nothing could fail. Here the signal-to-noise
    genuinely improves, so a small shift is expected -- the observed one is 0.08 against an interval
    0.17 wide, well inside what a 9x amplitude change would do if the measure tracked amplitude."""
    plain = _install(monkeypatch, [{}] * 4, {1: {}}, seed=3)
    flat, _d = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    a0 = flat[plain[0]][1]["dwhole"]

    gain = _install(monkeypatch, [{}] * 4, {1: {"gain": 3.0}}, seed=3)
    scaled, _d2 = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    a1 = scaled[gain[0]][1]["dwhole"]

    assert np.allclose(a0, a1, atol=0.12), (
        f"a 3x gain moved the interval from {a0} to {a1}; 8b is meant to be blind to it")


def test_the_ceiling_is_leave_one_out_and_not_one(monkeypatch):
    """A pre-stroke session scored against a pool CONTAINING it reads ~1 by construction, and every
    delta would then come out at about -1 regardless of the data -- the bug `_delta_diag_ci` was
    caught with before it reached a figure."""
    an, _mu, _days = _install(monkeypatch, [{}] * 5, {1: {}}, seed=4)
    out, _d = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    lo, hi, med = out[an]["PRE"]["whole"]
    assert lo < med < hi
    assert med < 0.999, "a ceiling of 1.0 means the reference was scored against itself"


def test_per_position_rows_are_reported_for_every_position(monkeypatch):
    an, _mu, days = _install(monkeypatch, [{}] * 4, {1: {}}, seed=5)
    out, _d = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    rec = out[an][days[0]]
    assert set(rec["rows"]) == set(LABELS)
    assert set(rec["drows"]) == set(LABELS)
    for q, iv in rec["rows"].items():
        assert -1.001 <= iv[0] <= iv[2] <= iv[1] <= 1.001, f"{q}: {iv}"


def test_a_lone_pre_session_is_declined_rather_than_estimated(monkeypatch):
    """One pre-stroke session cannot support a leave-one-out ceiling."""
    an, _mu, _days = _install(monkeypatch, [{}], {1: {}}, seed=6)
    out, _d = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    assert an not in out


# ---------------------------------------------------------------------------------------------
# `_anchor`: the bootstrap supplies the WIDTH, the plotted estimate supplies the LOCATION.
# ---------------------------------------------------------------------------------------------


def test_anchor_preserves_width_and_asymmetry():
    lo, hi, med = 0.30, 0.70, 0.40                      # skewed: 0.10 below, 0.30 above
    a = gf._anchor((lo, hi, med), 0.55)
    assert a[2] == 0.55
    assert a[1] - a[0] == pytest.approx(hi - lo), "the bootstrap width must survive"
    assert a[2] - a[0] == pytest.approx(med - lo), "and so must the skew"


def test_anchor_always_contains_the_estimate():
    """THE FAILURE THIS EXISTS FOR: PS92's encoder PRE read +0.57 against a percentile interval of
    [+0.32, +0.56] -- an interval that did not contain its own estimate."""
    a = gf._anchor((0.32, 0.56, 0.44), 0.57)
    assert a[0] <= 0.57 <= a[1], a


def test_anchor_respects_the_parameter_space():
    """A correlation of 0.98 shifted upward would advertise an upper limit above 1."""
    a = gf._anchor((0.72, 0.98, 0.87), 0.98, lo=-1.0, hi=1.0)
    assert a[1] <= 1.0
    b = gf._anchor((0.72, 0.98, 0.87), 0.98)
    assert b[1] > 1.0, "unbounded by default -- a gain and a difference are not correlations"


def test_anchor_declines_without_an_estimate():
    iv = (0.1, 0.5, 0.3)
    assert gf._anchor(iv, None) == iv
    assert gf._anchor(iv, np.nan) == iv
    assert gf._anchor(None, 0.4) is None


def test_every_published_interval_contains_its_own_estimate(monkeypatch):
    """The property, checked over the whole returned structure rather than one cell."""
    an, _mu, days = _install(monkeypatch, [{}] * 4, {1: {}, 2: {}}, seed=11)
    rdm, _d = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    enc, _e = gf._enc_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    n = 0
    for src in (rdm, enc):
        for rec in (src.get(an) or {}).values():
            for key, val in rec.items():
                ivs = val.values() if isinstance(val, dict) else [val]
                for iv in ivs:
                    if not iv:
                        continue
                    assert iv[0] <= iv[2] <= iv[1], f"{key}: {iv}"
                    n += 1
    assert n > 40, f"only {n} intervals checked; the structure changed shape"
    assert days

def test_the_cache_returns_what_the_uncached_path_would_have(monkeypatch, tmp_path):
    """A cached 8b interval must BE the interval, not merely arrive faster.

    The whole memoisation is only sound because the seeds are stable and scoped to one animal. If
    that ever stops being true the cached run and the fresh run diverge, and nothing downstream
    would show it -- the figure still renders, with different numbers.
    """
    from wfield_local import session_cache as sc
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path)
    an, _mu, _days = _install(monkeypatch, [{}] * 5, {1: {}}, seed=11)
    cold, _d = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    assert cold and an in cold
    assert list(tmp_path.glob("bootstrap/8b_rdm__*.pkl")), "nothing was cached"
    warm, _d2 = gf._rdm_ci("cue", "lick", 10, n_boot=40, n_loo=3)
    assert set(warm) == set(cold)
    for a in cold:
        assert set(warm[a]) == set(cold[a])
        for col in cold[a]:
            assert warm[a][col]["whole"] == cold[a][col]["whole"]
            assert warm[a][col]["rows"] == cold[a][col]["rows"]


def test_a_failed_bootstrap_is_never_memoised(tmp_path, monkeypatch):
    """An empty result is a FAILURE, and caching it makes one bad run permanent.

    This is not hypothetical: a NameError inside `_rdm_one` was swallowed by its broad `except`,
    None was pickled under twelve keys, and the corrected code then read those back and produced
    no intervals -- silently, twice.
    """
    from wfield_local import session_cache as sc
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path)
    calls = []

    def _empty():
        calls.append(1)
        return None

    for _ in range(2):
        assert gf._boot_cached("probe", ("k",), _empty) is None
    assert len(calls) == 2, "a falsy result was memoised and replayed"
    assert not list(tmp_path.glob("bootstrap/probe__*.pkl"))

    real = []
    assert gf._boot_cached("probe", ("k",), lambda: real.append(1) or {"x": 1}) == {"x": 1}
    assert gf._boot_cached("probe", ("k",), lambda: real.append(1) or {"x": 2}) == {"x": 1}
    assert len(real) == 1, "a real result was NOT memoised"


def test_the_8e_cache_returns_what_the_uncached_path_would(monkeypatch, tmp_path):
    """`_asymmetry_ci` cached per animal must BE the uncached result, not merely arrive faster.

    Same contract as the 8b round trip above, and the same reason it is worth pinning: a cached
    bootstrap that diverges from a fresh one still renders a figure, with different numbers and
    nothing to say so.
    """
    from wfield_local import session_cache as sc
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path)
    an, _mu, _days = _install(monkeypatch, [{}] * 4, {1: {}, 2: {}}, seed=7)
    cold = gf._asymmetry_ci("cue", "lick", 10, n_boot=40)
    assert cold and an in cold
    assert list(tmp_path.glob("bootstrap/8e_asym__*.pkl")), "nothing was cached"
    warm = gf._asymmetry_ci("cue", "lick", 10, n_boot=40)
    assert set(warm) == set(cold)
    for a in cold:
        assert set(warm[a]) == set(cold[a])
        for col in cold[a]:
            obs_c, sig_c = cold[a][col]
            obs_w, sig_w = warm[a][col]
            np.testing.assert_array_equal(obs_w, obs_c)
            np.testing.assert_array_equal(sig_w, sig_c)


def test_extracting_a_loop_body_keeps_every_decorator_on_its_own_function():
    """Guards a hazard that bit twice while extracting these per-animal units.

    Inserting a new function "before the next `def`" lands BETWEEN a decorator and the function it
    decorates, silently moving it to the inserted one. `@lru_cache` jumped to `_asymmetry_one` and
    `_rdm_ci` lost its `.cache_clear`; eleven top-level defs in that module are decorated, so this
    is a live trap rather than a one-off.
    """
    for name in ("_rdm_ci", "_rdm_rows", "_enc_ci", "_enc_tables"):
        assert hasattr(getattr(gf, name), "cache_clear"), f"{name} lost its cache decorator"
    for name in ("_rdm_one", "_asymmetry_one", "_disatt_one"):
        assert not hasattr(getattr(gf, name), "cache_clear"), (
            f"{name} picked up a decorator belonging to another function")
