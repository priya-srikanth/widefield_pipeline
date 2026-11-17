"""WIDEFIELD_GATE_SOURCE (the LP-inclusive engagement gate, 2026-11-01; `wfield_local/gate_source.py`).

Two things are pinned:

1. THE DEFAULT IS BYTE-IDENTICAL. With the switch unset, `gate_responded` returns its input object, no hook adds a
   row column, and on real sessions the gate arrays and the S28 quit dicts equal EXACTLY what
   `scripts/gate_lp_check.py` cached on 11-01 with the pre-option code (`WFIELD_INPUT_CACHE/gate_lp_check/`). Exact
   equality, not a tolerance (CLAUDE.md rule 9: a tolerance passes the very bug this guards against). The hooks also
   leave the map-cache source digests alone: every hook line is marked and `map_collect.code_digest` drops it.
2. THE SWITCH DOES WHAT IT SAYS. Unit tests of the OR rule and the windows; and on the same real sessions, with the
   switch on, the gate and the quit reproduce gate_lp_check's LP rule (`ne_img_lp`, `quit_lp`).

The real-session tests read DAQ .h5 + LP tables off the share (read-only) and skip when the cache or the share is absent.
"""
from __future__ import annotations

import math
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from wfield_local import gate_source

REPO = Path(__file__).resolve().parents[1]
#: Sessions pinned against gate_lp_check's cache: acute with a quit that moves under the LP rule (+2.6 min, 17 trials
#: leave the gate), chronic whose quit disappears under it, and pre-stroke with a small move.
PINNED = ("PS92_0818", "PS94_0921", "PS95_0814")
CANON_RAW = {"close_ipsi": "close_L", "close_mid": "close_center", "close_contra": "close_R",
             "far_ipsi": "far_L", "far_mid": "far_center", "far_contra": "far_R"}


@pytest.fixture
def gate_off(monkeypatch):
    monkeypatch.delenv(gate_source.GATE_SOURCE_ENV, raising=False)


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv(gate_source.GATE_SOURCE_ENV, "lp")


# ============================================================================================ unit (no I/O)
def test_default_returns_the_same_object(gate_off):
    r = np.array([True, False, True])
    assert gate_source.source() == "daq" and not gate_source.is_lp()
    assert gate_source.gate_responded("NOPE_0000", np.arange(3), r) is r       # no copy, no session lookup


def test_bad_value_refuses(monkeypatch):
    monkeypatch.setenv(gate_source.GATE_SOURCE_ENV, "video")
    with pytest.raises(ValueError):
        gate_source.source()


def test_switch_ors_by_cue_index(gate_on, monkeypatch):
    extra = np.array([False, True, False, False, True])
    monkeypatch.setattr(gate_source, "extra", lambda label, w, cue_samples=None, sr=None: {"any": extra})
    order = np.array([0, 1, 3, 4, 9])                       # 9 is out of range -> adds nothing
    resp = np.array([False, False, True, False, False])
    out = gate_source.gate_responded("X", order, resp, 3.5)
    assert out.tolist() == [False, True, True, True, False]
    assert resp.tolist() == [False, False, True, False, False]           # input untouched


def test_extra_windows_and_classes(gate_on, monkeypatch):
    """DAQ contact counts inside [cue, min(cue + w, next cue)]; an LP attempt counts with first lick <= w; a cue
    without an LP verdict, or a non-attempt class (session_lp leaves `attempted` False for jaw_only / no_try /
    unknown), adds nothing."""
    from wfield_local import config, lp_licks
    sr = 1000.0
    cues = np.array([0, 10_000, 20_000, 30_000, 40_000], np.int64)
    licks = np.array([3_400.0,          # cue 0: contact at 3.4 s       -> daq (w = 3.5)
                      13_600.0,         # cue 1: at 3.6 s               -> outside w
                      29_000.0])        # cue 2: at 9 s (next cue at 10 s) -> outside w
    monkeypatch.setattr(config, "load_sessions", lambda: [{"label": "X_0000", "h5": "x.h5"}])
    monkeypatch.setattr(gate_source, "_daq_contact_samples", lambda h5: (licks, sr))
    monkeypatch.setattr(lp_licks, "spec_token", lambda rv=None: "t")
    monkeypatch.setattr(lp_licks, "session_lp", lambda label, cs, sr_, max_rt, rv=None: {
        "drop": np.array([False, False, False, True, False]),
        "attempted": np.array([False, True, True, True, False]),     # cue 4: e.g. jaw_only -> not attempted
        "first_s": np.array([np.nan, 0.4, 3.9, 0.3, np.nan])})
    gate_source._EXTRA.clear()
    e = gate_source.extra("X_0000", 3.5, cues, sr)
    assert e["daq"].tolist() == [True, False, False, False, False]
    assert e["lp"].tolist() == [False, True, False, False, False]      # cue 2 too late, cue 3 dropped
    assert e["lp_cov"].tolist() == [True, True, True, False, True]
    assert e["any"].tolist() == [True, True, False, False, False]
    assert lp_licks.ATTEMPT_CLASSES == ("hit", "incomplete")           # incomplete yes, jaw_only no (10-21)


def test_quit_mask_default_identity(gate_off):
    """`beta_maps._quit_mask` with the switch off == the pre-option computation, on a session with a terminal quit."""
    from wfield_local.beta_maps import _quit_mask
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.precue_engagement_states import engagement_gate
    rng = np.random.default_rng(0)
    n = 300
    codes = rng.integers(0, 6, n)
    resp = rng.random(n) < 0.85
    resp[220:] = False
    idx_e, idx_n = np.flatnonzero(resp), np.flatnonzero(~resp)
    got = _quit_mask({"label": "NOPE_0000"}, idx_e, idx_n, codes[idx_e], codes[idx_n])
    order = np.concatenate([idx_e, idx_n])
    r = np.concatenate([np.ones(len(idx_e), bool), np.zeros(len(idx_n), bool)])
    pos = np.array([POSITION_NAMES.get(int(c), str(c)) for c in np.concatenate([codes[idx_e], codes[idx_n]])])
    o = np.argsort(order, kind="stable")
    by = dict(zip(order[o].tolist(), engagement_gate(order[o], r[o], pos[o]).tolist()))
    want = np.array([by[int(k)] for k in idx_n], bool)
    assert got.any() and np.array_equal(got, want)


def test_output_tag(monkeypatch):
    from wfield_local import lp_licks
    for k in (lp_licks.LICK_SOURCE_ENV, lp_licks.ENL_WINDOW_ENV, gate_source.GATE_SOURCE_ENV):
        monkeypatch.delenv(k, raising=False)
    assert lp_licks.output_tag() is None
    monkeypatch.setenv(lp_licks.LICK_SOURCE_ENV, "lp")
    assert lp_licks.output_tag() == "LP_licks"
    monkeypatch.setenv(gate_source.GATE_SOURCE_ENV, "lp")
    assert lp_licks.output_tag() == "LP_licks_gateLP"
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV)
    assert lp_licks.output_tag() == "DAQ_gateLP"


def test_session_cache_kind_untouched_by_default(gate_off, monkeypatch, tmp_path):
    from wfield_local import session_cache
    seen = []
    monkeypatch.setattr(session_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(session_cache, "session_signature", lambda s, p=None: "sig")
    monkeypatch.setattr(session_cache, "_disabled", lambda: False)
    session_cache.cached({"label": "X_0000"}, "bm-cue-working-abc", lambda: seen.append(1) or 1, verbose=False)
    names = sorted(p.name for p in tmp_path.glob("*.pkl"))
    # every kind ends in the detector part (`run_mode.session_kind_tag`, 11-06); the gate tag is what this pins
    assert names == ["X_0000__bm-cue-working-abc-daqlk__sig.pkl"]
    monkeypatch.setenv(gate_source.GATE_SOURCE_ENV, "lp")
    monkeypatch.setattr(gate_source, "cache_tag", lambda rv=None: "glpTEST")
    session_cache.cached({"label": "X_0000"}, "bm-cue-working-abc", lambda: 2, verbose=False)
    session_cache.cached({"label": "X_0000"}, "tf-cue-0123", lambda: 3, verbose=False)
    names = sorted(p.name for p in tmp_path.glob("*.pkl"))
    assert "X_0000__bm-cue-working-abc-glpTEST-daqlk__sig.pkl" in names  # gate consumers: own key under the switch
    assert "X_0000__tf-cue-0123-daqlk__sig.pkl" in names                 # trial features: shared, never gate-tagged


DIGESTED = ("wfield_local/beta_maps.py", "wfield_local/precue_engagement_states.py", "wfield_local/rest_engagement.py",
            "wfield_local/rest_by_position.py", "wfield_local/position_reference_maps.py")


@pytest.mark.parametrize("rel", DIGESTED)
def test_every_hook_line_in_a_digested_module_is_marked(rel):
    """`map_collect.code_digest` hashes these sources; an unmarked hook line would move every map-cache key."""
    src = (REPO / rel).read_text(encoding="utf-8")
    bad = [ln for ln in src.splitlines() if re.search(r"\bgate_source\b|\b_gs\.", ln) and gate_source.HOOK_MARK not in ln]
    assert not bad, bad
    plain = "def f():\n    return 1\n"
    assert gate_source.strip_hooks(plain) == plain
    assert gate_source.strip_hooks(plain + f"x = 1  {gate_source.HOOK_MARK}\n") == plain


# ============================================================================================ real sessions
def _cached(label):
    c = Path(os.environ.get("WFIELD_INPUT_CACHE", "C:/Users/SabatiniLab/wfield_cache")) / "gate_lp_check" / f"{label}.pkl"
    if not c.exists():
        pytest.skip(f"no gate_lp_check cache for {label}")
    d = pickle.load(open(c, "rb"))
    from wfield_local import config
    s = next((x for x in config.load_sessions() if x["label"] == label), None)
    if s is None or not s.get("h5") or not Path(s["h5"]).exists():
        pytest.skip(f"{label}: DAQ .h5 not reachable")
    return d, s


#: Under the switch only the QUIT moves: `hit_before` / `hit_after` / `step_sharpness` stay on the row's own `hit`
#: (the option never changes a hit rate), whereas gate_lp_check overwrote `hit` itself -- so those three are not compared.
HIT_DESCRIPTIVES = ("hit_before", "hit_after", "step_sharpness")
#: LICK-COUNT descriptives of the quit (cumulative licks at the quit, early licks / min): since 11-03 they are counted
#: with THE behaviour detector (`analysis_kit.lick_count_samples`, coordinator decision, DECISIONS "One DAQ lick
#: detector for lick COUNTS"), so they no longer equal the pinned pre-decision values. The GATE and the QUIT itself
#: (hit-based) stay pinned, exactly.
COUNT_DESCRIPTIVES = ("quit_cum_licks", "early_lpm", "early_lpt")


def _same_quit(a, b, skip=()):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert set(a) == set(b)
    for k in a:
        if k in skip:
            continue
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == y, (k, x, y)


def _img_gate(label, T, resp_col):
    from wfield_local.precue_engagement_states import engagement_gate
    k = T.k.to_numpy()
    pos = np.array([CANON_RAW[p] for p in T.pos])
    return np.asarray(engagement_gate(k, gate_source.gate_responded(label, k, T[resp_col].to_numpy(bool)), pos), bool)


@pytest.mark.parametrize("label", PINNED)
def test_default_gate_and_quit_pinned(gate_off, monkeypatch, label):
    from wfield_local import lp_licks
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV, raising=False)
    d, s = _cached(label)
    from scripts.rest_migration.engagement_decomposition import session_trials
    from scripts.rest_migration.quit_point import session_quit
    T = d["trials"]
    assert np.array_equal(_img_gate(label, T, "daq35"), T.ne_img.to_numpy(bool))
    rows = session_trials(s, 2.0)
    assert rows and all("gate_hit" not in r for r in rows)
    _same_quit(session_quit(s, rows), d["quit_daq"], skip=COUNT_DESCRIPTIVES)


@pytest.mark.parametrize("label", PINNED)
def test_switch_reproduces_gate_lp_check(gate_on, monkeypatch, label):
    from wfield_local import lp_licks
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV, raising=False)
    d, s = _cached(label)
    if not d["lp"]:
        pytest.skip(f"{label}: no LP tables")
    from scripts.rest_migration.engagement_decomposition import session_trials
    from scripts.rest_migration.quit_point import session_quit
    T = d["trials"]
    ne_lp = _img_gate(label, T, "daq35")
    assert np.array_equal(ne_lp, T.ne_img_lp.to_numpy(bool))
    assert ne_lp.sum() <= T.ne_img.sum()                     # the rule only adds responses
    rows = session_trials(s, 2.0)
    assert all("gate_hit" in r for r in rows)
    _same_quit(session_quit(s, rows), d["quit_lp"], skip=HIT_DESCRIPTIVES + COUNT_DESCRIPTIVES)
