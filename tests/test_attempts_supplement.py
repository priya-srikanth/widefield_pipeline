"""The opt-in LP attempts SUPPLEMENT (Priya 11-09, June camera tables, route 2b) and `trial_attempts.py --labels`.

    --labels               narrows the curated set; a label not available + curated under the env raises; needs --out
    switch unset           spec_token / cache_tag / session-cache file names byte-identical to before
    switch set             registered sessions keep their exact keys; a supplement session gets a key that tracks the
                           supplement file (size + mtime); spec_token never moves
    served                 `_session_lp` finds a supplement-only session; a registered one still comes from the table
    overlap refused        a label in both tables raises, never silently overrides
"""
import os

import numpy as np
import pandas as pd
import pytest

from scripts import trial_attempts as ta
from wfield_local import lp_licks
from wfield_local import session_cache as sc

COLS = ["label", "trial_id", "cls", "cls_raw", "cam4", "combined", "pos", "stop_ms", "contact"]


def _table(path, labels, cls="hit"):
    rows = [{"label": lab, "trial_id": i, "cls": cls, "cls_raw": cls, "cam4": 300.0, "combined": 250.0,
             "pos": "far_contra", "stop_ms": 3000.0, "contact": 400.0} for lab in labels for i in range(3)]
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
    return path


@pytest.fixture
def lp_env(monkeypatch, tmp_path):
    """Registered table with one August session; LP source on; supplement unset; caches cleared."""
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_")):
            monkeypatch.delenv(k, raising=False)
    reg = _table(tmp_path / "trial_attempts.csv", ["PS92_0901"])
    monkeypatch.setattr(lp_licks, "attempts_path", lambda rv=None: reg)
    monkeypatch.setattr(lp_licks, "_cam1_signature", lambda rv=None: "c1sig")
    for d in (lp_licks._ATTEMPTS, lp_licks._SUPP, lp_licks._LP):
        d.clear()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(sc, "CACHE_DIR", cache)
    monkeypatch.setattr(sc, "session_signature", lambda s, p=None: "basesig0123456789")
    monkeypatch.setattr(sc, "_disabled", lambda: False)
    monkeypatch.setenv(lp_licks.LICK_SOURCE_ENV, "lp")
    yield tmp_path, reg, cache
    for d in (lp_licks._ATTEMPTS, lp_licks._SUPP, lp_licks._LP):
        d.clear()


def _names(cache):
    return sorted(p.name for p in cache.glob("*.pkl"))


# ------------------------------------------------------------------------------------------------ --labels
def test_select_labels_default_unchanged_and_narrowed():
    avail = ["PS92_0606", "PS92_0901", "PS93_0606", "PS95_0902"]
    cur = ["PS92_0901", "PS95_0902", "PS92_0606", "PS93_0606"]
    assert ta.select_labels(avail, cur) == avail                         # default: every curated, in available order
    assert ta.select_labels(avail[1:], cur[:2]) == ["PS92_0901", "PS95_0902"]
    assert ta.select_labels(avail, cur, ["PS93_0606", "PS92_0606", "PS92_0606"]) == ["PS92_0606", "PS93_0606"]


def test_select_labels_refuses_a_label_not_curated_under_this_env():
    with pytest.raises(SystemExit, match="PS92_0606"):
        ta.select_labels(["PS92_0606", "PS92_0901"], ["PS92_0901"], ["PS92_0606"])   # e.g. PRE_FROM=0801 still set


def test_labels_needs_out():
    with pytest.raises(SystemExit, match="needs --out"):
        ta.main(["--labels", "PS92_0606"])


# ------------------------------------------------------------------------------------------------ switch unset
def test_unset_keys_identical_even_with_a_supplement_file_present(lp_env):
    tmp, reg, cache = lp_env
    tok0, tag0 = lp_licks.spec_token(), lp_licks.cache_tag()
    _table(tmp / "june.csv", ["PS92_0606"])                              # present on disk, switch unset: ignored
    assert lp_licks.supplement_path() is None
    assert lp_licks.spec_token() == tok0 and lp_licks.cache_tag() == tag0
    assert lp_licks.supplement_token("PS92_0606") == "" and lp_licks.supplement_token("PS92_0901") == ""
    assert sc._attempts_supplement_token("PS92_0606") == ""
    sc.cached({"label": "PS92_0901"}, "prm-x", lambda: 1)
    sc.cached({"label": "PS92_0606"}, "prm-x", lambda: 2)
    k = f"prm-x-{tag0}-daqlk"
    assert _names(cache) == sorted([f"PS92_0606__{k}__basesig0123456789.pkl", f"PS92_0901__{k}__basesig0123456789.pkl"])
    # the pinned token form (unchanged by this switch): lp<ver>t<tol>:<mtime>:<size>:cam1-<sig>
    st = reg.stat()
    assert tok0 == f"lp{lp_licks.LP_VERSION}t{lp_licks.LP_DAQ_TOL_MS:g}:{st.st_mtime_ns}:{st.st_size}:cam1-c1sig"


def test_unset_june_still_unavailable(lp_env, monkeypatch):
    monkeypatch.setattr(lp_licks, "lp_lick_times", lambda lab, rv=None: (np.array([]), pd.DataFrame(
        {"cue_daq": [1.0]}, index=pd.Index([0], name="trial_id"))))
    with pytest.raises(lp_licks.LPUnavailable, match="not in trial_attempts.csv$"):
        lp_licks._session_lp("PS92_0606", np.array([1000]), 1000.0, 3.5)


# ------------------------------------------------------------------------------------------------ switch set
def test_set_registered_keys_unchanged_supplement_keys_track_the_file(lp_env, monkeypatch):
    tmp, reg, cache = lp_env
    tok0, tag0 = lp_licks.spec_token(), lp_licks.cache_tag()
    sc.cached({"label": "PS92_0901"}, "prm-x", lambda: "aug")
    before = _names(cache)
    sup = _table(tmp / "june.csv", ["PS92_0606", "PS93_0606"])
    monkeypatch.setenv(lp_licks.SUPPLEMENT_ENV, str(sup))
    assert lp_licks.spec_token() == tok0 and lp_licks.cache_tag() == tag0            # shared tokens never move
    assert lp_licks.supplement_token("PS92_0901") == ""                              # registered: no key part
    st = sup.stat()
    assert lp_licks.supplement_token("PS92_0606") == f"sup:{st.st_size}:{st.st_mtime_ns}"
    assert sc.cached({"label": "PS92_0901"}, "prm-x", lambda: "recomputed") == "aug"  # warm, same file
    assert _names(cache) == before
    assert sc.cached({"label": "PS92_0606"}, "prm-x", lambda: "june1") == "june1"
    j1 = [n for n in _names(cache) if n.startswith("PS92_0606")]
    assert len(j1) == 1 and j1[0].startswith(f"PS92_0606__prm-x-{tag0}-daqlk__") and "basesig" not in j1[0]
    # rewrite the supplement (size changes): only the June key moves, the August file is untouched
    _table(sup, ["PS92_0606", "PS93_0606", "PS94_0606"])
    assert lp_licks.supplement_token("PS92_0606") != f"sup:{st.st_size}:{st.st_mtime_ns}"
    assert sc.cached({"label": "PS92_0606"}, "prm-x", lambda: "june2") == "june2"
    j2 = [n for n in _names(cache) if n.startswith("PS92_0606")]
    assert len(j2) == 1 and j2 != j1                                                 # superseded June entry pruned
    assert f"PS92_0901__prm-x-{tag0}-daqlk__basesig0123456789.pkl" in _names(cache)
    assert lp_licks.spec_token() == tok0


def test_set_but_daq_source_adds_no_key_part(lp_env, monkeypatch):
    tmp, _reg, _cache = lp_env
    monkeypatch.setenv(lp_licks.SUPPLEMENT_ENV, str(_table(tmp / "june.csv", ["PS92_0606"])))
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV)
    assert sc._attempts_supplement_token("PS92_0606") == ""


def test_session_lp_serves_supplement_sessions_and_registered_from_the_table(lp_env, monkeypatch):
    tmp, _reg, _cache = lp_env
    _table(tmp / "june.csv", ["PS92_0606"], cls="no_try")
    monkeypatch.setenv(lp_licks.SUPPLEMENT_ENV, str(tmp / "june.csv"))
    cue_t = pd.DataFrame({"cue_daq": [10.0, 20.0, 30.0]}, index=pd.Index([0, 1, 2], name="trial_id"))
    monkeypatch.setattr(lp_licks, "lp_lick_times", lambda lab, rv=None: (np.array([10.2, 20.3]), cue_t))
    cues = np.array([10_000, 20_000, 30_000])
    J = lp_licks._session_lp("PS92_0606", cues, 1000.0, 3.5)
    assert J["n_matched"] == 3 and not J["attempted"].any() and list(J["cls"]) == ["no_try"] * 3
    A = lp_licks._session_lp("PS92_0901", cues, 1000.0, 3.5)                       # registered: hit rows
    assert A["n_matched"] == 3 and A["attempted"].all()
    with pytest.raises(lp_licks.LPUnavailable, match="nor in the supplement june.csv"):
        lp_licks._session_lp("PS95_0606", cues, 1000.0, 3.5)


def test_overlap_refused(lp_env, monkeypatch):
    tmp, _reg, _cache = lp_env
    monkeypatch.setenv(lp_licks.SUPPLEMENT_ENV, str(_table(tmp / "bad.csv", ["PS92_0606", "PS92_0901"])))
    with pytest.raises(lp_licks.AttemptsSupplementOverlap, match="PS92_0901"):
        lp_licks.supplement_token("PS92_0606")
    with pytest.raises(lp_licks.AttemptsSupplementOverlap):
        sc.cached({"label": "PS92_0901"}, "prm-x", lambda: 1)


def test_missing_supplement_file_raises(lp_env, monkeypatch):
    tmp, _reg, _cache = lp_env
    monkeypatch.setenv(lp_licks.SUPPLEMENT_ENV, str(tmp / "nope.csv"))
    with pytest.raises(FileNotFoundError):
        lp_licks.supplement_token("PS92_0606")
