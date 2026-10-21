"""LP_licks (`wfield_local.lp_licks`, 2026-10-21): the video lick train that replaces DAQ contact under
WIDEFIELD_LICK_SOURCE=lp.

    the gate IS the first lick    attempted -> first lick after the cue = cue + first LP lick; earlier LP licks
                                  inside the trial are removed, a missing first lick is inserted
    no attempt = no lick          no_try / jaw_only / unknown -> nothing in [cue, cue + max_rt]; a lick after
                                  max_rt survives (late_rewarded)
    nothing else moves            pre-cue / ITI licks and dropped cues untouched
    DAQ path untouched            switch unset -> "daq"; session_cache kinds and frozen specs unchanged
    LP never fits a DAQ spec      a frozen-model miss under LP refuses rather than filing LP features as DAQ
"""
import numpy as np
import pytest

from wfield_local import lp_licks


def _first_after(t, c):
    t = np.asarray(t)
    a = t[t >= c]
    return a.min() if a.size else np.nan


def test_attempted_trial_first_lick_is_the_lp_onset():
    cue = np.array([10.0, 20.0])
    licks = np.array([9.0, 10.3, 10.6, 11.0, 25.0])          # 10.3 = a jaw / partial event before the attempt onset
    out = lp_licks.edit_train(licks, cue, drop=[False, False], attempted=[True, False],
                              first_s=[0.6, np.nan], max_rt=3.5)
    assert _first_after(out, 10.0) == pytest.approx(10.6)
    assert 9.0 in out and 11.0 in out                        # pre-cue and later licks kept


def test_missing_first_lick_is_inserted():
    out = lp_licks.edit_train(np.array([5.0]), np.array([10.0]), [False], [True], [0.45], 3.5)
    assert _first_after(out, 10.0) == pytest.approx(10.45)


def test_non_attempt_has_no_lick_within_max_rt_but_keeps_a_late_one():
    cue = np.array([10.0])
    licks = np.array([10.5, 12.0, 14.2])
    out = lp_licks.edit_train(licks, cue, [False], [False], [np.nan], 3.5)
    assert _first_after(out, 10.0) == pytest.approx(14.2)


def test_attempt_after_max_rt_counts_as_no_lick_inside_the_window():
    out = lp_licks.edit_train(np.array([13.8]), np.array([10.0]), [False], [True], [3.8], 3.5)
    assert _first_after(out, 10.0) == pytest.approx(13.8)    # beyond max_rt: not engaged, still late_rewarded


def test_dropped_cue_is_left_alone():
    licks = np.array([10.2, 10.9])
    out = lp_licks.edit_train(licks, np.array([10.0]), [True], [False], [np.nan], 3.5)
    np.testing.assert_array_equal(out, licks)


def test_default_source_is_daq_and_bad_values_raise(monkeypatch):
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV, raising=False)
    assert lp_licks.lick_source() == "daq" and not lp_licks.is_lp()
    monkeypatch.setenv(lp_licks.LICK_SOURCE_ENV, "contact")
    with pytest.raises(ValueError):
        lp_licks.lick_source()


def test_session_cache_tags_kinds_only_under_lp(monkeypatch, tmp_path):
    from wfield_local import session_cache as sc
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sc, "session_signature", lambda s, p=None: "sig")
    monkeypatch.setattr(sc, "_disabled", lambda: False)
    monkeypatch.setattr(lp_licks, "spec_token", lambda rv=None: "lp1:1:1")
    s = {"label": "PS00_0101"}
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV, raising=False)
    assert sc.cached(s, "tf-lick-abc", lambda: "daq") == "daq"
    monkeypatch.setenv(lp_licks.LICK_SOURCE_ENV, "lp")
    assert sc.cached(s, "tf-lick-abc", lambda: "lp") == "lp"       # not served the DAQ entry
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV)
    assert sc.cached(s, "tf-lick-abc", lambda: "recomputed") == "daq"  # DAQ entry not evicted
    names = sorted(p.name for p in tmp_path.glob("*.pkl"))
    assert names == ["PS00_0101__tf-lick-abc-" + lp_licks.cache_tag() + "__sig.pkl", "PS00_0101__tf-lick-abc__sig.pkl"]


def test_frozen_spec_unchanged_under_lp_and_miss_refuses(monkeypatch):
    from wfield_local import frozen_models as fm
    monkeypatch.setattr(fm, "_session_sig", lambda lab: "x")
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV, raising=False)
    daq = fm.make_spec("PS92", "position", align="lick", source="locanmf", train_labels=["PS92_0806"])
    monkeypatch.setenv(lp_licks.LICK_SOURCE_ENV, "lp")
    lp = fm.make_spec("PS92", "position", align="lick", source="locanmf", train_labels=["PS92_0806"])
    assert fm.spec_id(lp) == fm.spec_id(daq)                         # LP reuses the DAQ-frozen model
    monkeypatch.setattr(fm, "find", lambda spec: (None, None))
    monkeypatch.setattr(fm, "warn_if_store_moved", lambda log: None)
    with pytest.raises(RuntimeError, match="reuses DAQ-frozen"):
        fm.load_or_fit(lp, lambda: pytest.fail("must not fit"), log=lambda m: None)
    monkeypatch.setenv(fm.LP_FROZEN_ENV, "lp")
    sens = fm.make_spec("PS92", "position", align="lick", source="locanmf", train_labels=["PS92_0806"])
    assert sens["lick_source"] == "lp" and fm.spec_id(sens) != fm.spec_id(daq)


def test_lp_outputs_route_to_lp_licks_and_guard_refuses_daq_paths(monkeypatch):
    from pathlib import Path

    from wfield_local import writeguard
    g = Path("M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures")
    monkeypatch.delenv(lp_licks.LICK_SOURCE_ENV, raising=False)
    assert lp_licks.output_dir(g / "epoch") == g / "epoch"
    writeguard.assert_writable(g / "epoch" / "x.png")
    monkeypatch.setenv(lp_licks.LICK_SOURCE_ENV, "lp")
    assert lp_licks.output_dir(g / "epoch") == g / "LP_licks" / "epoch"
    assert lp_licks.output_dir(g) == g / "LP_licks"
    writeguard.assert_writable(g / "LP_licks" / "epoch" / "x.png")
    with pytest.raises(writeguard.WriteGuardError, match="LP_licks"):
        writeguard.assert_writable(g / "epoch" / "x.png")
    writeguard.assert_writable("C:/Users/SabatiniLab/wfield_cache/x.pkl")          # local paths unaffected
