"""ONE ENL window, and it follows the lick source (Priya 11-04: "every ENL analysis follows the lick source").

`lp_licks.enl_post_s()` is the single definition: WIDEFIELD_ENL_WINDOW_S if set, else 1.0 s under LP, else the DAQ
window decode.precue_post_s (2.0 s). These tests pin it under each source, check every site switched on 11-04 resolves
to it, and check the scripts that gate on their own lick times read the licks of the window's source
(`lp_licks.gate_licks_s`). No data, no share: everything the sites read is monkeypatched.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wfield_local import config, lp_licks

REPO = Path(__file__).resolve().parents[1]
DAQ_W = float(config.defaults()["decode"]["precue_post_s"])


@pytest.fixture
def daq(monkeypatch):
    for k in (lp_licks.LICK_SOURCE_ENV, lp_licks.ENL_WINDOW_ENV):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.fixture
def lp(daq):
    daq.setenv(lp_licks.LICK_SOURCE_ENV, "lp")
    return daq


# ---------------------------------------------------------------- the shared definition


def test_daq_window_is_the_config_window(daq):
    assert DAQ_W == 2.0
    assert lp_licks.enl_post_s() == DAQ_W
    assert lp_licks.enl_window_s() is None                 # default run: no override, keys unchanged


def test_lp_window_is_one_second(lp):
    assert lp_licks.enl_post_s() == lp_licks.LP_ENL_WINDOW_S == 1.0
    assert lp_licks.enl_window_s() == 1.0


def test_override_wins_under_both_sources(daq):
    daq.setenv(lp_licks.ENL_WINDOW_ENV, "1.5")
    assert lp_licks.enl_post_s() == 1.5                    # the matched DAQ arm
    daq.setenv(lp_licks.LICK_SOURCE_ENV, "lp")
    assert lp_licks.enl_post_s() == 1.5                    # LP at another window
    assert lp_licks.enl_post_s(source="daq") == 1.5


def test_forced_source_ignores_the_env_source(lp):
    assert lp_licks.enl_post_s(source="daq") == DAQ_W      # cohort-2-style / DAQ-only callers under an LP shell
    lp.setenv(lp_licks.LICK_SOURCE_ENV, "daq")
    assert lp_licks.enl_post_s(source="lp") == 1.0
    with pytest.raises(ValueError):
        lp_licks.enl_post_s(source="cam")


@pytest.mark.parametrize("src,want", [("daq", 2.0), ("lp", 1.0)])
def test_post_s_for(daq, src, want):
    daq.setenv(lp_licks.LICK_SOURCE_ENV, src)
    assert lp_licks.post_s_for("precue") == want
    assert lp_licks.post_s_for("precue-rt2-f3") == want    # pass-2 variant names are the pre-cue window too
    dp = config.defaults()["decode"]
    assert lp_licks.post_s_for("cue") == float(dp["cue_post_s"])
    assert lp_licks.post_s_for("lick") == float(dp["lick_post_s"])


@pytest.mark.parametrize("src", ["daq", "lp"])
def test_apply_enl_window_agrees_with_enl_post_s(daq, src):
    """The feature builders' window (`apply_enl_window` on the config value) IS the shared one."""
    daq.setenv(lp_licks.LICK_SOURCE_ENV, src)
    a = lp_licks.apply_enl_window(SimpleNamespace(align="precue", post_s=DAQ_W))
    assert a.post_s == lp_licks.enl_post_s()


def test_frozen_spec_carries_the_window(daq):
    """The frozen pre-cue spec's post_s is the shared window, whatever post_s the caller passed."""
    from wfield_local import frozen_models as fm
    kw = dict(align="precue", source="locanmf", train_labels=["PS92_0811"], train_data=np.zeros((2, 2)))
    assert fm.make_spec("PS92", "position", post_s=DAQ_W, **kw)["post_s"] == DAQ_W
    daq.setenv(lp_licks.LICK_SOURCE_ENV, "lp")
    assert fm.make_spec("PS92", "position", post_s=DAQ_W, **kw)["post_s"] == 1.0
    assert fm.make_spec("PS92", "position", post_s=lp_licks.enl_post_s(), **kw)["post_s"] == 1.0


# ---------------------------------------------------------------- the gate's licks follow the window's source


def test_gate_licks_daq_is_the_daq_train(daq):
    licks, keep = lp_licks.gate_licks_s("X_0811", [10.0, 20.0], [5.0, 1.0, 3.0], 3.5)
    assert licks.tolist() == [1.0, 3.0, 5.0] and keep.tolist() == [True, True]


def test_gate_licks_lp_is_the_session_lp_train(lp):
    seen = {}

    def fake(label, cue_samples, sr, max_rt, rv=None):
        seen.update(label=label, cues=np.asarray(cue_samples) / sr, max_rt=max_rt)
        return {"lick_samples": np.round(np.array([9.5, 10.2]) * sr).astype(np.int64),
                "drop": np.array([False, True])}
    lp.setattr(lp_licks, "session_lp", fake)
    licks, keep = lp_licks.gate_licks_s("X_0811", [10.0, 20.0], [5.0], 3.5)
    assert np.allclose(licks, [9.5, 10.2]) and keep.tolist() == [True, False]
    assert seen["label"] == "X_0811" and np.allclose(seen["cues"], [10.0, 20.0]) and seen["max_rt"] == 3.5


def _fake_session(monkeypatch, tmp_path, src):
    """trial_bounds / DAQ licks / LP train for one fake session: three cues at 10, 20, 30 s.

    DAQ licks: one at 9.5 s (inside every pre-cue window of cue 10), the post-cue licks.
    LP licks: one at 19.5 s (inside cue 20's window), the post-cue licks; cue 30 has no LP verdict."""
    from wfield_local import dlc_frames
    from wfield_local import trial_windows as TW
    monkeypatch.setenv(lp_licks.LICK_SOURCE_ENV, src)
    monkeypatch.delenv(lp_licks.ENL_WINDOW_ENV, raising=False)
    b = pd.DataFrame({"cue_s": [10.0, 20.0, 30.0], "strobe_s": [5.0, 15.0, 25.0],
                      "pos_name": ["close_L", "close_R", "far_L"]})
    monkeypatch.setattr(TW, "trial_bounds", lambda animal, date, rv: b.copy())
    monkeypatch.setattr(dlc_frames, "lick_onsets", lambda *a, **k: np.array([9.5, 10.3, 20.3, 30.3]))

    def fake_lp(label, cue_samples, sr, max_rt, rv=None):
        return {"lick_samples": np.round(np.array([10.3, 19.5, 20.3, 30.3]) * sr).astype(np.int64),
                "drop": np.array([False, False, True])}
    monkeypatch.setattr(lp_licks, "session_lp", fake_lp)
    d = tmp_path / "sessions" / "PS92" / "20260811"
    d.mkdir(parents=True)
    (d / "PS92_20260811_1200_trials.csv").write_text("x\n")
    return SimpleNamespace(root=lambda key: str(tmp_path))


@pytest.mark.parametrize("src,kept", [("daq", [20.0, 30.0]), ("lp", [10.0])])
def test_enl_movement_table_gates_on_the_window_source(monkeypatch, tmp_path, src, kept):
    EM = importlib.import_module("scripts.enl_movement_multisession")
    rv = _fake_session(monkeypatch, tmp_path, src)
    s = {"label": "PS92_0811", "h5": str(tmp_path / "20260811" / "x.h5")}
    _a, _d, tr = EM.enl_table(s, rv, lp_licks.enl_post_s(), 3.5)
    assert tr.cue_s.tolist() == kept


@pytest.mark.parametrize("src,kept", [("daq", [20.0, 30.0]), ("lp", [10.0])])
def test_enl_video_trials_gate_on_the_window_source(monkeypatch, tmp_path, src, kept):
    VS = importlib.import_module("scripts.enl_video_spout_check")
    rv = _fake_session(monkeypatch, tmp_path, src)
    tr = VS.enl_trials("PS92", "20260811", rv, label="PS92_0811")
    assert tr.cue_s.tolist() == kept


# ---------------------------------------------------------------- every switched site resolves to the shared window


@pytest.mark.parametrize("src", ["daq", "lp"])
def test_enl_lick_rates_window(daq, src):
    daq.setenv(lp_licks.LICK_SOURCE_ENV, src)
    m = importlib.import_module("scripts.enl_lick_rates")
    assert m._post_s() == lp_licks.enl_post_s()


@pytest.mark.parametrize("src", ["daq", "lp"])
def test_within_spout_enl_window_is_the_lp_window(daq, src):
    """Its licks are the encoding bundle's video licks: LP window (1.0 s, unchanged) under either env source."""
    daq.setenv(lp_licks.LICK_SOURCE_ENV, src)
    m = importlib.reload(importlib.import_module("scripts.within_spout_deviation"))
    assert m.ENL_S == (-lp_licks.enl_post_s(source="lp"), 0.0) == (-1.0, 0.0)


@pytest.mark.parametrize("src", ["daq", "lp"])
def test_precue_lickfree_control_is_the_daq_window(daq, src):
    """DAQ-contact control: DAQ window (2.0 s, unchanged) under either env source."""
    daq.setenv(lp_licks.LICK_SOURCE_ENV, src)
    m = importlib.reload(importlib.import_module("wfield_local.precue_lickfree"))
    assert m.PRE_S == lp_licks.enl_post_s(source="daq") == DAQ_W


def _src(rel):
    return (REPO / rel).read_text(encoding="utf-8")


#: (file, the shared-window call it must contain) for the sites a static check covers (their callers load data)
SITES = {
    "wfield_local/precue_engagement_states.py": "lp_licks.enl_post_s()",
    "scripts/enl_movement_removed_decode.py": "post_s=lp_licks.enl_post_s()",
    "scripts/enl_movement_multisession.py": "win_s, lead_s = lp_licks.enl_post_s(), 0.6",
    "scripts/enl_fr_removed_decode.py": "win_s, lead_s = lp_licks.enl_post_s(), 0.6",
    "scripts/enl_video_spout_check.py": "win_s = lp_licks.enl_post_s()",
    "scripts/enl_lick_rates.py": "return lp_licks.enl_post_s()",
    "scripts/cohort2_qc/quiet_task_events.py": 'ENL_S = _lp_licks.enl_post_s(source="daq")',
    "scripts/within_spout_deviation.py": 'ENL_S = (-_lp_licks.enl_post_s(source="lp"), 0.0)',
    "scripts/null_potent_session.py": '("ENL", "precue", lp_licks.enl_post_s())',
    "wfield_local/precue_lickfree.py": 'PRE_S = _lp_licks.enl_post_s(source="daq")',
    "wfield_local/attempt_coding.py": "post_s = lp_licks.post_s_for(align)",
    "wfield_local/attempt_maps.py": "post_s = post_s or lp_licks.post_s_for(align)",
    "wfield_local/cd_trajectories.py": 'lp_licks.apply_enl_window(_args(source="roi", align=align, post_s=post_s))',
    "wfield_local/cd_overlap_null.py": "post_s = lp_licks.post_s_for(align)",
    "scripts/component_exclusion_audit.py": "post_s = lp_licks.post_s_for(align)",
    "wfield_local/position_axes.py": "post_s = lp_licks.post_s_for(align)",
    "wfield_local/position_coding_directions.py": "post_s = lp_licks.post_s_for(align)",
    "wfield_local/nightly_figs.py": "str(lp_licks.post_s_for(al))",
    "wfield_local/locanmf_analysis_deck.py": "post = lp_licks.post_s_for(align)",
    "scripts/locanmf_refit_test/refit_vs_projection.py": "post_s=lp_licks.post_s_for(align)",
}


@pytest.mark.parametrize("rel", sorted(SITES))
def test_site_uses_the_shared_window(rel):
    assert SITES[rel] in _src(rel), f"{rel}: the 11-04 shared-window call is gone"


#: The ONLY places allowed to read decode.precue_post_s (or decode.{align}_post_s) directly: the definition itself and
#: precue_engagement_states' `cue` branch (a post-cue window that has always used that value).
_READ = re.compile(r"""\[\s*["']precue_post_s["']\s*\]|get\(\s*["']precue_post_s["']|\{al(ign)?\}_post_s["']""")
_ALLOWED = {"wfield_local/lp_licks.py": 2, "wfield_local/precue_engagement_states.py": 1}


def test_no_other_direct_reads_of_the_config_window():
    found = {}
    for p in sorted([*(REPO / "wfield_local").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]):
        rel = p.relative_to(REPO).as_posix()
        if "/archive/" in rel:
            continue
        code = "\n".join(ln for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
                         if not ln.lstrip().startswith("#"))
        n = len(_READ.findall(code))
        if n:
            found[rel] = n
    assert found == _ALLOWED, found


def test_quiet_task_events_npz_records_its_window():
    t = _src("scripts/cohort2_qc/quiet_task_events.py")
    assert "enl_s=float(ENL_S)" in t and "_npz_enl_s(f) == ENL_S" in t
