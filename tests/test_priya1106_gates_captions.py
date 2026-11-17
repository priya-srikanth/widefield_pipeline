"""Priya 11-06, A20-A23 (fix/priya1106-gates-captions). Synthetic inputs only, no share reads.

  A20 kinematics on the CAMERA WORKING PERIOD: `trial_classes.working_period_trials` / `keep_working_period` /
      `gate_kin_licks` (one definition, shared with Fig 0.2 / 1b); jaw_only_incomplete_figs, pose3d_cue_traces and the
      3-D per-lick scripts use it; a session without camera classes contributes nothing; WFIELD_KIN_GATE=legacy keeps
      the old gates in their own folders.
  A21 recovery_timing's hit rate = trial_attempts.session_summary's frac_hit (contact / working-period trials).
  A22 fig_goal_summary suptitle readings computed from the drawn numbers.
  A23 exec_vs_neural_match's relabel tag derived from the design at call time, mismatches / empty runs refused.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ("jaw_only_incomplete_figs", "pose3d_cue_traces", "pose3d_summary_fig", "pose3d_spout_error", "pose3d_phase",
           "recovery_timing", "exec_vs_neural_match", "fig_goal_summary", "trial_attempts")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Clean run-mode env; scripts importable by name; restore sys.path / os.environ and drop ONLY the script modules."""
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "rebuild_figs"))
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    yield
    for m in SCRIPTS:
        sys.modules.pop(m, None)


from wfield_local import trial_classes as tc  # noqa: E402

# one session, cue order: 2 contact, incomplete, jaw_only, no_response, stopped run, after the last camera lick
CLS = np.array(["contact", "contact", "incomplete", "jaw_only", "no_response", "stopped", "stopped_licking",
                "after_end", "unknown"], dtype=object)


@pytest.fixture
def session(monkeypatch):
    """PS92_0810 with the classes above; PS93_0813 with no video (all unknown)."""
    full = np.arange(len(CLS)) * 10.0
    T = pd.DataFrame({"trial_id": np.arange(len(CLS)) + 1, "position": ["far_R"] * len(CLS), "cue_daq": full + 0.004,
                      "cls": ["raw"] * len(CLS)})
    monkeypatch.setattr(tc, "session_trials", lambda lab, *a, **k: T if lab.startswith("PS92") else None)
    monkeypatch.setattr(tc, "daq_cue_seconds", lambda lab: full)
    monkeypatch.setattr(tc, "cue_classes_for", lambda lab, cs, w=None: CLS.copy() if lab.startswith("PS92")
                        else np.array(["unknown"] * len(cs), dtype=object))
    return T


# ------------------------------------------------------------------------------------------------ A20 shared helpers
def test_working_period_trials_is_the_camera_period(session):
    W = tc.working_period_trials("PS92_20260810")             # long label accepted
    assert W.trial_id.tolist() == [1, 2, 3, 4, 5]
    assert W.cls.tolist() == ["contact", "contact", "incomplete", "jaw_only", "no_response"]   # classified, not raw
    assert tc.working_period_ids("PS92_0810") == {1, 2, 3, 4, 5}
    # the same set the 11-05 helper gives (one definition)
    ids, has = tc.working_trial_ids("PS92_0810")
    assert has and {k for k, v in ids.items() if v} == tc.working_period_ids("PS92_0810")


def test_no_camera_classes_contributes_nothing(session):
    assert tc.working_period_trials("PS93_20260813").empty
    df = pd.DataFrame({"session": ["PS92_20260810"] * 3 + ["PS93_20260813"] * 2, "trial_id": [1, 6, 9, 1, 2]})
    out = tc.keep_working_period(df)
    assert out.session.tolist() == ["PS92_20260810"] and out.trial_id.tolist() == [1]


def test_kin_gate_switch_and_folders(session, monkeypatch):
    df = pd.DataFrame({"session": ["PS92_20260810"] * 3, "trial_id": [1, 6, 9]})
    assert tc.kin_gate() == "working" and tc.gate_kin_licks(df).trial_id.tolist() == [1]
    assert tc.kin_out_name("phase") == "phase"
    monkeypatch.setenv(tc.KIN_GATE_ENV, "legacy")
    assert tc.gate_kin_licks(df) is df and tc.kin_out_name("phase") == "phase_legacygate"
    assert tc.kin_gate_signature() == "legacy"
    monkeypatch.setenv(tc.KIN_GATE_ENV, "engaged")
    with pytest.raises(ValueError):
        tc.kin_gate()


def test_kin_gate_signature_follows_the_selection(monkeypatch):
    monkeypatch.setattr(tc, "class_source_signature", lambda *a, **k: "csig")
    a = tc.kin_gate_signature()
    monkeypatch.setenv(tc.END_ENV, "contact")
    b = tc.kin_gate_signature()
    assert a.startswith("working|") and a.endswith("|csig") and a != b


# ------------------------------------------------------------------------------------------------ A20 consumers
def test_jaw_only_figs_default_is_the_working_period(session, monkeypatch, tmp_path):
    import jaw_only_incomplete_figs as jf
    from wfield_local import orofacial_cohort as oc
    monkeypatch.setattr(oc, "engagement_keep", lambda *a, **k: pytest.fail("behavior gate used under the default"))
    r = jf.session_job(("PS92_20260810", str(tmp_path), str(tmp_path), str(tmp_path)))
    (c,) = r["counts"]
    assert c["n_trials"] == 5 and c["share_unknown"] == 0
    assert sum(c[f"share_{k}"] for k in ("contact", "incomplete", "jaw_only", "no_response")) == pytest.approx(1)
    assert jf.session_job(("PS93_20260813", str(tmp_path), str(tmp_path), str(tmp_path)))["counts"] == []
    assert jf.denom() == jf.WORKING_DENOM and "camera working period" in jf.data_line()
    # legacy: the behaviour gate, its own denominator text
    monkeypatch.setenv(tc.KIN_GATE_ENV, "legacy")
    monkeypatch.setattr(oc, "engagement_keep", lambda *a, **k: ({1, 2, 9}, "reference"))
    monkeypatch.setattr(tc, "session_trials", lambda lab, *a, **k: pd.DataFrame(
        {"trial_id": [1, 2, 3, 9], "position": "far_R", "cls": ["contact", "contact", "no_response", "unknown"]}))
    (c,) = jf.session_job(("PS92_20260810", str(tmp_path), str(tmp_path), str(tmp_path)))["counts"]
    assert c["n_trials"] == 3 and "engaged" in jf.denom()


def test_jaw_only_figs_writes_to_the_derived_pose_area():
    import re
    src = (REPO / "scripts" / "jaw_only_incomplete_figs.py").read_text(encoding="utf-8")
    assert re.search(r'session_poses_out\(\s*"jaw_only_incomplete_figs"', src)


def test_cue_traces_keep_ids_and_set(session, monkeypatch, tmp_path):
    import pose3d_cue_traces as ct
    from wfield_local import orofacial_cohort as oc
    monkeypatch.setattr(oc, "engagement_keep", lambda *a, **k: ({7}, "reference"))
    assert ct.keep_ids("PS92", "20260810", tmp_path) == {1, 2, 3, 4, 5}
    assert ct.keep_ids("PS93", "20260813", tmp_path) == set()          # no classes: nothing, not "ungated"
    assert ct.all_set() == "working" and ct.out_dir(tmp_path, "cue") == tmp_path / "cue_traces"
    assert "camera working period" in ct.data_line("cue")
    monkeypatch.setenv(tc.KIN_GATE_ENV, "legacy")
    assert ct.keep_ids("PS92", "20260810", tmp_path) == {7}
    assert ct.all_set() == "engaged" and ct.out_dir(tmp_path, "first_lick") == tmp_path / "lick_traces_legacygate"


def test_kin_outputs_routed_under_a_rebuild_root(monkeypatch, tmp_path):
    from wfield_local import lp_licks
    d3 = tmp_path / "session_poses" / "pose3d"
    assert tc.kin_out_dir(d3, "phase") == d3 / "phase" and tc.kin_cache_dir(d3, "phase") == d3 / "_cache" / "phase"
    monkeypatch.setenv("WIDEFIELD_REBUILD_ROOT", str(tmp_path / "rb"))
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    base = tmp_path / "rb" / "session_poses_derived" / lp_licks.output_tag()
    assert tc.kin_out_dir(d3, "phase") == base / "pose3d_phase"            # never the session_poses input tree
    assert tc.kin_cache_dir(d3, "phase") == base / "pose3d_cache" / "phase"
    monkeypatch.setenv(tc.KIN_GATE_ENV, "legacy")
    assert tc.kin_out_dir(d3, "figures", "pose3d_summary_fig") == base / "pose3d_summary_fig_legacygate"


def test_kinematics_pooled_reads_either_set():
    src = (REPO / "scripts" / "kinematics_pooled_figs.py").read_text(encoding="utf-8")
    assert 'sset = "working" if (T.set == "working").any() else "engaged"' in src


def _kin_tables():
    L = pd.DataFrame({"session": ["PS92_20260810"] * 4 + ["PS93_20260813"] * 2, "animal": ["PS92"] * 4 + ["PS93"] * 2,
                      "date": ["20260810"] * 4 + ["20260813"] * 2, "epoch": "pre",
                      "position": ["far_L", "far_L", "far_R", "far_R", "far_L", "far_R"],
                      "trial_id": [1, 6, 2, 9, 1, 2], "reach_mm": [1.0, 100.0, 3.0, 100.0, 5.0, 5.0],
                      "speed_peak": [10.0, 0.0, 30.0, 0.0, 1.0, 1.0]})
    J = pd.DataFrame({"session": ["PS92_20260810"] * 4, "animal": "PS92", "date": "20260810", "epoch": "pre",
                      "position": ["far_L", "far_R", "far_L", "far_R"], "trial_id": [1, 2, 6, 9],
                      "lr_wide_mm": [1.0, -1.0, 50.0, 50.0], "open_max_mm": [2.0, 4.0, 99.0, 99.0]})
    return L, J


def test_summary_fig_rows_are_over_working_trials(session):
    import pose3d_summary_fig as sf
    L, J = _kin_tables()
    K = sf.gated_sessions(L, J)
    assert K.session.tolist() == ["PS92_20260810"]                     # PS93: no camera classes, absent
    r = K.iloc[0]
    assert r["reach_mm|far_L"] == 1.0 and r["reach_mm|far_R"] == 3.0 and r["speed_peak|far_R"] == 30.0
    assert r["open_max_mm|far_L"] == 2.0 and r["jaw_far_dir_mm"] == pytest.approx(1.0)
    assert r["jaw_far_bias_mm"] == pytest.approx(0.0)


def test_spout_error_gated_licks(session, tmp_path):
    import pose3d_spout_error as se
    L, _J = _kin_tables()
    (tmp_path / "kinematics").mkdir()
    L.to_csv(tmp_path / "kinematics" / "licks.csv.gz", index=False)
    G = se.gated_licks(tmp_path)
    assert G.trial_id.tolist() == [1, 2] and G.index.tolist() == [0, 1]   # reset: add_peak_errors indexes positionally
    assert "camera working period" in se.data_line()


def test_three_d_scripts_route_through_the_one_gate():
    for name in ("pose3d_phase", "pose3d_spout_error", "pose3d_vs_cam4", "exec_bins_calibration"):
        src = (REPO / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert "tc.gate_kin_licks(" in src, name
    pv = (REPO / "scripts" / "rebuild_figs" / "fig_pose_validation_sessions.py").read_text(encoding="utf-8")
    assert "tc.gate_kin_licks(" in pv and 'tc.kin_out_dir(P3, "vs_cam4")' in pv      # same lick set as pose3d_vs_cam4
    for name in ("pose3d_phase", "pose3d_spout_error"):                 # cached results keyed on the gate
        src = (REPO / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert "extra=tc.kin_gate_signature()" in src, name


# ------------------------------------------------------------------------------------------------ A21
def test_recovery_hit_rate_is_session_summary_frac_hit(monkeypatch, tmp_path):
    import recovery_timing as rt
    import trial_attempts as ta
    # the per-trial camera table of one session: 3 contact, 1 incomplete, 1 jaw_only, 1 no_try inside the period,
    # 2 stopped + 1 after_end outside it -> frac_hit = 3 / 6 (stopped / after-end not in the denominator)
    cls = ["hit", "hit", "hit", "incomplete", "jaw_only", "no_try", "stopped", "stopped", "after_end"]
    n = len(cls)
    D = pd.DataFrame({"label": "PS92_0810", "pos": "far_contra", "cls": cls, "cls_raw": cls,
                      "canonical": ~pd.Series(cls).isin(ta.RUN_EXCLUDED), "attempted": pd.Series(cls).isin(
                          ta.ATTEMPT_CLASSES), "n_licks": 1, "n_nocontact": 0, "contact": 300.0, "combined": 250.0,
                      "has_cam1": True, "cam4": 250.0, "jaw_onset_ms": np.nan, "angle_first": 0.0}, index=range(n))
    W = ta.session_summary(D)
    H = rt.far_contra_hit(W)
    assert H.hit.tolist() == [pytest.approx(3 / 6)]
    with pytest.raises(SystemExit):                                   # a non-camera table is refused
        rt.far_contra_hit(W.drop(columns=[c for c in W.columns if c.startswith("frac_stopped_")]))
    # series(): meta from the per-trial table, the rate from the sessions table, early pre dropped
    D.assign(animal="PS92", epoch="pre", day=-5).to_csv(tmp_path / "trial_attempts.csv", index=False)
    W.to_csv(tmp_path / "trial_attempts_sessions.csv", index=False)
    pd.DataFrame({"label": ["PS92_0810"], "window": ["lick"], "cls2": ["hit"], "ret": [0.9], "n": [12]}).to_csv(
        tmp_path / "decoder_retention_sessions.csv", index=False)
    monkeypatch.setattr(rt._lp_licks, "attempts_path", lambda rv=None: tmp_path / "trial_attempts.csv")
    monkeypatch.setattr(rt, "A", tmp_path)
    S = rt.series("lick")
    assert S.hit.tolist() == [pytest.approx(0.5)] and S.beh.tolist() == [pytest.approx(1.0)] and S.neu.tolist() == [0.9]


# ------------------------------------------------------------------------------------------------ A22
def test_goal_summary_readings_are_computed():
    import fig_goal_summary as gs
    rows = []
    for an, (t, x) in zip(gs.AN, [(2, 1), (2, 1), (1, 2), (np.nan, 1)]):
        for e in gs.EP:
            rows.append({"animal": an, "epoch": e, "nokin:u:tgtbin|cue": t, "nokin:u:exbin|cue": x,
                         "nokin:u:tgtbin|lick": x, "nokin:u:exbin|lick": t})
    s1 = gs.fig1_reading(pd.DataFrame(rows))
    assert "chronic cue 2/3, lick 1/3" in s1 and "4/4" not in s1
    summ = {w: pd.DataFrame([{"animal": an, "cls": cl, "n": 10, "p_true": p, "p_true_sem": 0.01}
                             for an, ps in zip(gs.AN, [(0.2, 0.5), (0.6, 0.5), (0.1, 0.5), (0.1, 0.5)])
                             for cl, p in zip(("incomplete", "pre_hit"), ps)]) for w in gs.WINDOWS}
    summ["precue"].loc[summ["precue"].animal == "PS95", "n"] = 3               # below the drawn floor: not counted
    trials = {(w, an): pd.DataFrame({"cls": ["incomplete"] * 6, "pred": ["far_mid"] * k + ["close_mid"] * (6 - k)})
              for w in gs.WINDOWS for an, k in zip(gs.AN, (6, 6, 1, 0))}
    s2 = gs.fig2_reading({"summary": summ, "trials": trials})
    assert "precue 2/3, cue 3/4, lick 3/4" in s2 and "far spout on > 1/2 of trials in precue 2/4" in s2
    h = pd.DataFrame([{"animal": an, "epoch": e, "cls": cl, "ipsi": v, "contra": v}
                      for an, f in zip(gs.AN, (0.5, 0.5, 2.0, 0.5))
                      for e, cl, v in (("acute", "hit", 1.0), ("acute", "incomplete", f), ("pre", "pre_hit", 1.0),
                                       ("pre", "pre_incomplete", 1.0))]).set_index(["animal", "epoch", "cls"])
    assert gs.fig3_reading(h) == "failed attempts smaller than hits (animals): vs pre-stroke hits 3/4, vs pre-stroke misses 3/4"
    src = (REPO / "scripts" / "rebuild_figs" / "fig_goal_summary.py").read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]                                    # below the module docstring
    for typed in ("chronic target > executed 4/4", "distance kept, side lost", "only 2/4"):
        assert typed not in code


# ------------------------------------------------------------------------------------------------ A23
def test_relabel_tag_follows_the_design(monkeypatch, tmp_path):
    import exec_vs_neural_match as ev
    from wfield_local import frozen_encoder as fe
    monkeypatch.setattr(fe, "SPONT", True)
    monkeypatch.setattr(fe, "LICKS_ALL", "split")
    assert ev.relabel_tag() == "v7g3_t3d_spont_trial"                 # what the render's a_relabel_trial writes
    with pytest.raises(SystemExit):
        ev.relabel_tag("v7g3_t3d_trial")                                # stale non-spont tag under split: refused
    monkeypatch.setattr(fe, "SPONT", False)
    monkeypatch.setattr(fe, "LICKS_ALL", "none")
    assert ev.relabel_tag() == "v7g3_t3d_trial"
    monkeypatch.setattr(fe, "out_root", lambda: tmp_path)
    with pytest.raises(SystemExit):                                    # empty relabel run: refused, not 0 trials
        ev.encoder_trials()
    src = (REPO / "scripts" / "exec_vs_neural_match.py").read_text(encoding="utf-8")
    assert 'tag: str = _design_tag("v7g3_t3d_trial")' not in src          # never frozen at import
