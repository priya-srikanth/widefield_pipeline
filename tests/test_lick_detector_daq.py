"""Imaging uses THE behaviour DAQ lick detector everywhere (Priya 11-04/05, REBUILD_CODE 8.53).

Pins: the one accessor (`config.lick_detector`), that no imaging call site carries the old literal
``(2.5, 1.0, (0.001, 0.020), 0.10)`` any more, the 40 ms min-ILI fallback shared by every reader, the cache-version bumps,
the ``daqlk`` run-tag part, and the before / after onset counts on 17 real sessions
(tests/fixtures/lick_detector_onsets_20261105.csv; the 16 of the 11-04 comparison match its task1_per_session.csv exactly).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "lick_detector_onsets_20261105.csv"
OLD = (2.5, 1.0, (0.001, 0.020), 0.10)          # the retired imaging profile


def test_accessor_is_the_yaml_behaviour_profile():
    from wfield_local import config
    ld = config.defaults()["lick_detection"]
    d = config.lick_detector()
    assert d["thresh_upper"] == ld["thresh_upper"] == 2.5
    assert d["thresh_lower"] == ld["thresh_lower"] == 0.5
    assert d["lockout_s"] == tuple(ld["lockout_falling_edge_s"])
    assert d["refractory_s"] == 0.0 and d["min_ili_s"] == pytest.approx(0.040)
    assert config.lick_detector_args() == (2.5, 0.5, tuple(ld["lockout_falling_edge_s"]), 0.0)


def test_min_ili_fallback_is_40_everywhere(monkeypatch):
    from wfield_local import config
    base = config.defaults()
    no_floor = {**base, "lick_detection": {k: v for k, v in base["lick_detection"].items() if k != "min_ili_ms"}}
    monkeypatch.setattr(config, "defaults", lambda session=None: no_floor)
    assert config.lick_min_ili_s() == pytest.approx(0.040) and config.LICK_MIN_ILI_MS_DEFAULT == 40.0
    # no reader keeps its own fallback (behavior_events / restnl_mask / rest_by_position fell back to 0; daq_trials to 40)
    bad = []
    for d in ("wfield_local", "scripts"):
        for f in (REPO / d).rglob("*.py"):
            if "archive" in f.parts:
                continue
            if re.search(r'get\("min_ili_ms",\s*\d', f.read_text(encoding="utf-8", errors="replace")):
                bad.append(str(f.relative_to(REPO)))
    assert not bad, bad


def test_no_imaging_literal_left():
    """Every `_load_daq_events(h5, "lick_analog", ...)` takes the accessor's default; archived scripts are left as they
    ran (they reproduce archived results)."""
    lit = re.compile(r'2\.5,\s*1\.0,\s*\(0\.001,\s*0\.020\),\s*0\.10')
    cli = re.compile(r'"--refractory-s",\s*type=float,\s*default=0\.1')
    bad = []
    for d in ("wfield_local", "scripts"):
        for f in (REPO / d).rglob("*.py"):
            if "archive" in f.parts:
                continue
            src = f.read_text(encoding="utf-8", errors="replace")
            code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
            if f.name == "analysis_kit.py":            # its docstring quotes the retired profile on purpose
                code = code.replace("``(2.5, 1.0, (0.001, 0.020), 0.10)``", "")
            if lit.search(code) or cli.search(code):
                bad.append(str(f.relative_to(REPO)))
    assert not bad, bad


def _fake_daq(monkeypatch, volts, fs=5000.0):
    from wfield_local import daq_io

    class F:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    names = ["cue", "spout_strobe", "pco_exposure", "spout_bit0", "spout_bit1", "spout_bit2"]
    monkeypatch.setattr(daq_io, "open_daq", lambda p: F())
    monkeypatch.setattr(daq_io, "session_attrs", lambda f: (fs, "2026-08-11T00:00:00"))
    monkeypatch.setattr(daq_io, "analog_channel", lambda f, name, required=True: volts)
    monkeypatch.setattr(daq_io, "digital_bits", lambda f: (names, np.zeros((volts.size, len(names)), np.uint8)))


def _trace(onsets_ms, fs=5000.0, dur_ms=30.0, total_s=3.0):
    v = np.full(int(total_s * fs), 3.5, np.float32)
    for t in onsets_ms:
        i = int(t / 1000.0 * fs)
        v[i:i + int(dur_ms / 1000.0 * fs)] = 0.0
    return v


def test_load_daq_events_defaults_keep_fast_real_licks_and_drop_sub_floor_doubles(monkeypatch):
    """Licks 95 ms apart (10.5 Hz, PS95's fast licking) are BOTH kept now; the old 0.10 s refractory dropped the second.
    A contact re-onset 30 ms after the last (the 40 ms floor) is still dropped."""
    monkeypatch.delenv("WIDEFIELD_LICK_SOURCE", raising=False)
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    fs = 5000.0
    v = _trace([500.0, 595.0, 1500.0], fs, dur_ms=20.0)
    v[int(1.550 * fs):int(1.560 * fs)] = 0.0                       # re-touch 50 ms after the 1500 ms onset (kept)
    v[int(2.000 * fs):int(2.010 * fs)] = 0.0
    v[int(2.030 * fs):int(2.040 * fs)] = 0.0                       # 30 ms after -> below the 40 ms floor (dropped)
    _fake_daq(monkeypatch, v, fs)
    new = _load_daq_events("x.h5", "lick_analog")
    old = _load_daq_events("x.h5", "lick_analog", *OLD)
    assert new["lick_source"] == "daq"
    np.testing.assert_array_equal(new["lick_samples"] / fs, [0.5, 0.595, 1.5, 1.55, 2.0])
    np.testing.assert_array_equal(old["lick_samples"] / fs, [0.5, 1.5, 2.0])
    ld = new["lick_detection"]
    assert (ld["thresh_lower"], ld["refractory_s"], ld["min_ili_s"]) == (0.5, 0.0, pytest.approx(0.04))


def test_analysis_kit_lick_samples_routes_through_the_default(monkeypatch):
    from wfield_local import analysis_kit as ak
    from wfield_local import plot_lick_aligned_averages as pla
    seen = {}

    def fake(h5, ch, *a, **k):
        seen.update(h5=h5, ch=ch, a=a, k=k)
        return {"lick_samples": np.array([1, 2, 3])}
    monkeypatch.setattr(pla, "_load_daq_events", fake)
    assert ak.lick_samples({"h5": "x.h5"}).tolist() == [1, 2, 3]
    assert seen == {"h5": "x.h5", "ch": "lick_analog", "a": (), "k": {}}


def test_cache_versions_bumped():
    from wfield_local import gate_source, session_cache
    assert session_cache.CACHE_VERSION >= 14
    assert gate_source.GATE_VERSION >= 2
    src = (REPO / "scripts" / "disengagement_tests.py").read_text(encoding="utf-8")
    assert int(re.search(r"^BEH_VERSION = (\d+)\b", src, re.M).group(1)) >= 3      # 4: + quit_cam (11-05)
    assert re.search(r"^VERSION = 2\b", (REPO / "scripts" / "gate_lp_check.py").read_text(encoding="utf-8"), re.M)


def test_daqlk_part_in_every_tagged_run_never_the_default(monkeypatch):
    from wfield_local import run_mode
    for k in ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LOCANMF_PROJECT", "WIDEFIELD_PRE_FROM", "WIDEFIELD_TRIAL_GATE",
              "WIDEFIELD_MAP_REFS", "WIDEFIELD_LICK_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    assert run_mode.tag() == "" and run_mode.cache_tag() == ""
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    assert run_mode.tag() == "pre0801_daqlk"
    assert run_mode.cache_tag() == "daqlk"                     # pre is pooling-level; the detector is per session
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    p = run_mode.parts()
    assert p.index("daqlk") == p.index(next(x for x in p if x.startswith("sel-"))) + 1
    assert run_mode.input_run_tag(f"X:/a/{run_mode.tag()}/b.png") == run_mode.tag()


def test_rebuild_profile_tag_carries_daqlk():
    from wfield_local import render_profile as rp
    d = rp.load("cohort1_rebuild_20261103")
    assert d["notes"]["expected_run_tag"] == ("hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-"
                                              "endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk")
    for arm, a in d["arms"].items():
        assert "_daqlk" in a["expected_run_tag"], arm


def test_onset_fixture_agrees_with_the_comparison_doc():
    """The fixture's 16 comparison sessions reproduce the doc's per-animal table (LICK_DETECTOR_COMPARE_2026-11-04 §3)."""
    F = pd.read_csv(FIXTURE)
    C = F[F.in_compare_doc]
    assert len(C) == 16 and set(C.animal) == {"PS92", "PS93", "PS94", "PS95"}
    g = C.groupby("animal")[["n_old_imaging", "n_new_daq"]].sum()
    assert g.n_old_imaging.to_dict() == {"PS92": 30683, "PS93": 27395, "PS94": 24080, "PS95": 19658}
    assert g.n_new_daq.to_dict() == {"PS92": 30785, "PS93": 27542, "PS94": 24139, "PS95": 19940}
    assert (C.n_new_daq.sum(), C.n_old_imaging.sum()) == (102406, 101816)
    assert C.n_removed.sum() == 1                      # the one imaging-only onset (PS94_0819), offset threshold: 0
    assert (F.n_new_daq - F.n_old_imaging == F.n_added - F.n_removed).all()


def _h5(label):
    from wfield_local import config
    try:
        for s in config.load_sessions():
            if s.get("label") == label and s.get("h5") and Path(s["h5"]).exists():
                return s["h5"]
    except Exception:                                                  # noqa: BLE001 - no share on this box
        pass
    pytest.skip(f"{label}: DAQ .h5 not reachable")


@pytest.mark.parametrize("label", ["PS95_0917", "PS94_0819"])
def test_onset_counts_on_real_sessions(label, monkeypatch):
    """PS95_0917 = the largest change (+117, 1.6 %); PS94_0819 = the one onset the old detector had and the new lacks."""
    monkeypatch.delenv("WIDEFIELD_LICK_SOURCE", raising=False)
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    row = pd.read_csv(FIXTURE).set_index("label").loc[label]
    h5 = _h5(label)
    assert _load_daq_events(h5, "lick_analog")["lick_samples"].size == row.n_new_daq
    assert _load_daq_events(h5, "lick_analog", *OLD)["lick_samples"].size == row.n_old_imaging
