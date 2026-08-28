"""The grant render can be fanned over processes, and its bootstraps can be reused between runs.

Priya, 2026-08-28: fix the parallelism before any further full render, and store bootstrap results
so a nightly run recomputes only the sessions that changed.

WHAT THE RENDER ACTUALLY COSTS, measured from the output mtimes of the 2026-08-28 serial run
(5.79 h wall, 1.51 of 24 cores): six bootstrap families -- 7b, 8d, 6d, 7d, 8b, 8e -- are **94.7%**
of it, and each writes exactly five files, one per (alignment, trial class), at 10-37 minutes
apiece. Collection is ~20 s against that. So the unit of parallelism is the FIGURE, not the figure
family, and the duplicated collection a fan-out costs is seconds against half-hours.

THREE PROPERTIES HAD TO BE TRUE FIRST, and none of them was:

  * **Stable seeds.** Every bootstrap seed was `abs(hash((animal, align, variant))) % 2**31`, and
    Python salts string hashing per process. Three consecutive interpreters gave 1125027485,
    2138950357 and 223190567 for the same tuple. The render had therefore never been reproducible:
    the point estimates held, every confidence interval moved. Workers make it louder, not
    different -- each is its own process with its own salt.
  * **Per-day seeds.** `_disattenuated_ci` drew every day of an animal from one stream, so a day's
    interval depended on how many days preceded it in that run, and no day's result could be
    cached and replayed.
  * **A unit decomposition that matches the filenames.** Figure 4 splits by alignment but NOT by
    trial class, and its filename carries no variant. Treating it as variant-split had two workers
    writing one path at the same time -- a torn PNG that presents as a merely missing figure.
"""
import subprocess
import sys

import numpy as np
import pytest

from wfield_local import grant_figures as G


# --------------------------------------------------------------------------- stable seeds

def test_the_seed_is_the_same_in_a_different_process():
    """The property `hash()` never had. Checked in a SUBPROCESS, because that is where it failed.

    Run in-process this passes trivially for any implementation, including the broken one.
    """
    code = "from wfield_local.grant_figures import _seed; print(_seed('PS92','cue','lick',7))"
    got = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True).stdout.strip() for _ in range(3)}
    assert len(got) == 1, f"seed varies between processes: {got}"
    assert got != {""} and int(next(iter(got))) >= 0


def test_the_seed_separates_what_should_be_separate():
    base = G._seed("PS92", "cue", "lick", 7)
    for other in (G._seed("PS93", "cue", "lick", 7), G._seed("PS92", "precue", "lick", 7),
                  G._seed("PS92", "cue", "working", 7), G._seed("PS92", "cue", "lick", 8)):
        assert other != base


def test_no_salted_hash_seeds_remain():
    """`abs(hash(...)) % 2**31` anywhere in this module is an unreproducible bootstrap."""
    src = (G.__file__ or "").replace(".pyc", ".py")
    text = open(src, encoding="utf-8").read()
    assert "abs(hash(" not in text, "a salted hash seed is back; use _seed()"


# ------------------------------------------------------------------- the unit decomposition

def test_units_never_share_an_output_path():
    """Figure 4 is window-split but not variant-split; asking for both variants raced on one file."""
    units = G.render_units()
    assert len(units) == len(set(units))
    four = [u for u in units if u[0] == "4"]
    assert four and all(v is None for _k, _a, v in four), four
    # and a variant-split figure still gets its variants
    seven_b = [u for u in units if u[0] == "7b"]
    assert {v for _k, _a, v in seven_b} == {"lick", "working"}


def test_the_expensive_families_are_scheduled_first():
    """Makespan, not correctness: a pool that starts 7b last finishes half an hour idle."""
    keys = [k for k, _a, _v in G.render_units()]
    assert keys[0] == "7b" and keys.index("8d") < keys.index("1")


def test_every_windowed_figure_is_detected_from_its_source():
    """A hand-kept list is wrong the first time somebody adds a figure."""
    windowed = {k for k, fn in G.JOBS if G._splits(fn)[0]}
    assert "7b" in windowed and "8d" in windowed and "4" in windowed
    assert "1" not in windowed and "2" not in windowed


# ------------------------------------------------------------------ the bootstrap cache

def _cache_to(tmp_path, monkeypatch):
    from wfield_local import session_cache as sc
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sc, "_disabled", lambda: False)


def test_a_cached_bootstrap_is_returned_instead_of_recomputed(tmp_path, monkeypatch):
    _cache_to(tmp_path, monkeypatch)
    calls = []

    def compute():
        calls.append(1)
        return {"far_R": (0.1, 0.9)}

    parts = (np.arange(12, dtype=float), "params")
    a = G._boot_cached("probe", parts, compute)
    b = G._boot_cached("probe", parts, compute)
    assert a == b == {"far_R": (0.1, 0.9)}
    assert len(calls) == 1, "the second call recomputed instead of reading the cache"


def test_changed_data_misses_the_cache(tmp_path, monkeypatch):
    """The key is the DATA, not a session name -- a re-preprocessed session must not hit a stale
    entry, which is the contamination class this repo keeps finding."""
    _cache_to(tmp_path, monkeypatch)
    calls = []
    x = np.arange(12, dtype=float)
    G._boot_cached("probe", (x, "p"), lambda: calls.append(1) or "a")
    y = x.copy()
    y[3] += 1e-9                      # one bit of one trial
    G._boot_cached("probe", (y, "p"), lambda: calls.append(1) or "b")
    assert len(calls) == 2, "a changed input hit the cached entry"


def test_the_digest_separates_shape_and_dtype():
    """Two arrays can share a byte string while meaning different things."""
    a = np.zeros(8, dtype=np.int64)
    assert G._digest(a) != G._digest(a.astype(np.float64))
    assert G._digest(a) != G._digest(a.reshape(2, 4))
    assert G._digest(a) == G._digest(a.copy())


def test_the_disabled_switch_turns_the_bootstrap_cache_off_too(tmp_path, monkeypatch):
    """One environment variable has to disable EVERY memoisation, or a result under suspicion
    cannot be reproduced from scratch."""
    from wfield_local import session_cache as sc
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sc, "_disabled", lambda: True)
    calls = []
    for _ in range(2):
        G._boot_cached("probe", (np.arange(4.0), "p"), lambda: calls.append(1) or "x")
    assert len(calls) == 2


# ------------------------------------------------------------ the cached value is the real value

def test_one_days_bootstrap_is_reproducible_from_its_seed():
    """A cached draw must be the draw the uncached path would have produced.

    Without this the cache is not a speed-up but a source of numbers nobody can regenerate.
    """
    def rng_of():
        return np.random.default_rng(G._seed("PS92", "cue", "lick", 3, "7bci"))

    gen = np.random.default_rng(0)
    # `_collect_7` hands back trials keyed by POSITION, per session (pre) or per day (post).
    def per_pos(n=48, dim=24):
        return ({q: gen.normal(size=(n, dim)) for q in G.CONF_LABELS},
                {q: np.arange(n) // 6 for q in G.CONF_LABELS})

    px0, pb0 = per_pos()
    px1, pb1 = per_pos()
    dx, db = per_pos()
    pre_x, pre_b = {0: px0, 1: px1}, {0: pb0, 1: pb1}
    a = G._disatt_one(pre_x, pre_b, dx, db, rng_of(), n_boot=6)
    b = G._disatt_one(pre_x, pre_b, dx, db, rng_of(), n_boot=6)
    assert a == b


@pytest.mark.parametrize("var", ["OMP_NUM_THREADS", "MKL_NUM_THREADS"])
def test_the_worker_thread_cap_is_named(var):
    """Uncapped, each worker's BLAS grabs the box: the serial render already averages 1.5 cores
    from BLAS alone, so ten unconstrained workers oversubscribe 24 rather than scale on them."""
    src = open((G.__file__ or "").replace(".pyc", ".py"), encoding="utf-8").read()
    assert var in src


def test_one_days_delta_bootstrap_is_reproducible_from_its_seed():
    """Same property for the delta family (6d, 7d, 8d, 9) -- 49% of a render between them."""
    gen = np.random.default_rng(1)

    def per_pos(n=48, dim=24):
        return ({q: gen.normal(size=(n, dim)) for q in G.CONF_LABELS},
                {q: np.arange(n) // 6 for q in G.CONF_LABELS})

    px0, pb0 = per_pos()
    px1, pb1 = per_pos()
    dx, db = per_pos()
    pre_x, pre_b = {0: px0, 1: px1}, {0: pb0, 1: pb1}

    def once():
        rng = np.random.default_rng(G._seed("PS92", "cue", "lick", "6d", 3))
        return G._delta_diag_one(G._mats_pattern("PS92", rng), pre_x, pre_b, dx, db, rng, 8)

    a, b = once(), once()
    assert a.keys() == b.keys()
    assert a.get("mean") == b.get("mean")


def test_the_delta_driver_seeds_per_day_not_per_animal():
    """A day's interval must not depend on how many days preceded it in the run.

    That dependence is what made the old shared stream uncacheable, and it also meant rendering a
    subset of days silently changed the days that remained.
    """
    src = open((G.__file__ or "").replace(".pyc", ".py"), encoding="utf-8").read()
    body = src[src.index("def _delta_diag_ci("):src.index("def _delta_diag_one(")]
    assert "_seed(*seed_parts, d)" in body, "the delta bootstrap is not seeded per day"
    assert "_boot_cached(" in body, "the delta bootstrap is not cached per day"
