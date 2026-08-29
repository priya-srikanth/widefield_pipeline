"""The suite must never write the lab's real session cache.

The share guard in conftest watches N:/ and M:/. `session_cache.CACHE_DIR` is
`E:/.widefield_session_cache` on this box -- shared lab state on a LOCAL disk, so it went
unwatched. Running the suite with a broken bootstrap wrote twelve `None` entries into it, and the
corrected code then read them back and produced empty figures.
"""
import pathlib

from wfield_local import grant_figures as gf
from wfield_local import session_cache as sc


def test_the_cache_dir_is_redirected_away_from_the_real_one():
    assert sc.CACHE_DIR != sc._default_cache_dir(), (
        "tests are pointed at the production session cache")


def test_a_bootstrap_written_during_tests_lands_in_the_temp_cache():
    """Not just that CACHE_DIR differs -- that a real write FOLLOWS it."""
    out = gf._boot_cached("isolation_probe", ("k",), lambda: {"v": 1})
    assert out == {"v": 1}
    here = list(pathlib.Path(sc.CACHE_DIR).glob("bootstrap/isolation_probe__*.pkl"))
    assert here, "the write did not land in the redirected cache"
    real = pathlib.Path(sc._default_cache_dir())
    if real.exists():
        assert not list(real.glob("bootstrap/isolation_probe__*.pkl")), (
            "a test wrote into the production cache")


def test_each_test_gets_a_cold_cache():
    """The probe above must not still be here; a warm cache would void every round-trip test."""
    assert not list(pathlib.Path(sc.CACHE_DIR).glob("bootstrap/isolation_probe__*.pkl"))
