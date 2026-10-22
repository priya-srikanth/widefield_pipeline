"""Faster epoch-map collection (`wfield_local.map_collect`, 2026-10-21).

    defined order        results keyed and sorted by session label, whatever order the workers finish in
    failures reported    a raising session is named, the rest still come back
    cache key moves      with the map code, the arguments, MAPS_VERSION and the ENL window -- and ONLY with those
    cache is used        a second call for the same session x arm does not recompute
"""
import numpy as np
import pytest

from wfield_local import map_collect


def _square(job):
    lab, x = job
    if x < 0:
        raise ValueError("negative")
    return x * x


def test_collect_serial_keys_by_label_and_reports_failures():
    got, fail = map_collect.collect([("b", 3), ("a", 2), ("c", -1)], _square, jobs=1)
    assert list(got) == ["a", "b"] and got == {"a": 4, "b": 9}
    assert len(fail) == 1 and fail[0][0][0] == "c" and "negative" in fail[0][1]


def test_collect_parallel_matches_serial():
    items = [(f"s{i:02d}", i) for i in range(12)][::-1]
    a, _ = map_collect.collect(items, _square, jobs=1)
    b, _ = map_collect.collect(items, _square, jobs=3)
    assert list(a) == list(b) and a == b


def _f():
    return 1


def _g():
    return 2


def test_cache_kind_moves_with_code_args_and_enl_window(monkeypatch):
    from wfield_local import lp_licks
    for k in (lp_licks.LICK_SOURCE_ENV, lp_licks.ENL_WINDOW_ENV):
        monkeypatch.delenv(k, raising=False)
    base = map_collect.cache_kind("prm", "precue", "working", {"post_s": 2.0}, map_collect.code_digest(_f))
    assert base == map_collect.cache_kind("prm", "precue", "working", {"post_s": 2.0}, map_collect.code_digest(_f))
    assert base != map_collect.cache_kind("prm", "precue", "working", {"post_s": 2.0}, map_collect.code_digest(_g))
    assert base != map_collect.cache_kind("prm", "precue", "working", {"post_s": 1.0}, map_collect.code_digest(_f))
    assert base != map_collect.cache_kind("prm", "cue", "working", {"post_s": 2.0}, map_collect.code_digest(_f))
    monkeypatch.setenv(lp_licks.ENL_WINDOW_ENV, "1.0")
    assert base != map_collect.cache_kind("prm", "precue", "working", {"post_s": 2.0}, map_collect.code_digest(_f))
    monkeypatch.setattr(map_collect, "MAPS_VERSION", map_collect.MAPS_VERSION + 1)
    monkeypatch.delenv(lp_licks.ENL_WINDOW_ENV)
    assert base != map_collect.cache_kind("prm", "precue", "working", {"post_s": 2.0}, map_collect.code_digest(_f))


def test_session_raw_maps_cached_computes_once(monkeypatch, tmp_path):
    from wfield_local import position_reference_maps as prm
    from wfield_local import session_cache as sc
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sc, "session_signature", lambda s, p=None: "sig%08x" % (hash(str(p)) & 0xffffffff))
    monkeypatch.setattr(sc, "_disabled", lambda: False)
    calls = []

    def fake(session, align, *, post_s=2.0, variant="working"):
        calls.append(1)
        return {"raw": {"far_R": np.ones((2, 2))}, "align": align}
    monkeypatch.setattr(prm, "session_raw_maps", fake)
    s = {"label": "PS00_0101", "mc": str(tmp_path / "nomc")}
    a = prm.session_raw_maps_cached(s, "cue", variant="working")
    b = prm.session_raw_maps_cached(s, "cue", variant="working")
    c = prm.session_raw_maps_cached(s, "lick", variant="lick")
    assert len(calls) == 2 and a["align"] == b["align"] == "cue" and c["align"] == "lick"
    np.testing.assert_array_equal(a["raw"]["far_R"], b["raw"]["far_R"])


def test_map_jobs_env(monkeypatch):
    monkeypatch.setenv(map_collect.MAP_JOBS_ENV, "3")
    assert map_collect.map_jobs() == 3
    monkeypatch.setenv(map_collect.MAP_JOBS_ENV, "0")
    assert map_collect.map_jobs() == 1


@pytest.mark.parametrize("v", ["", None])
def test_stat_sig_handles_missing(v, tmp_path):
    assert map_collect.stat_sig(v) == "-"
    assert map_collect.stat_sig(tmp_path / "nope.npy") == "nope.npy:-"
