"""Priya 11-04 lick-timing rules (docs/status/CUE70_CHECK_2026-11-04.md, REBUILD_CODE_20261103.md section 8.43):

1. CUE + 70 ms: post-cue from cue + 70 ms (camera licks by PEAK, DAQ contacts by onset); pre-cue ends AT the cue; a lick
   whose peak is in cue .. cue + 70 ms is NEITHER (excluded_0_70). Peak, not onset: onsets at 60-70 ms are responses.
2. RESPONSE WINDOW: the camera classes count licks / contacts / jaw onsets only up to min(cue + max_rt, stop), max_rt =
   decode.max_rt_s (the DAQ engaged rule's source).
The encoder's whole-session lick lists keep every lick (only the analysis label changes)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.pose_kinematics_demo as D  # noqa: E402
from wfield_local import config  # noqa: E402
from wfield_local import trial_classes as TC  # noqa: E402


# ----------------------------------------------------------------------------------------------- rule 1: the 0-70 split
def test_one_definition_of_the_split():
    assert TC.POSTCUE_START_MS == TC.CONTACT_FROM_CUE_S * 1000.0 == 70.0
    assert TC.PRECUE_END_AFTER_CUE_MS == 0.0 and TC.EXCLUDED_CONTEXT == "excluded_0_70"
    assert D.PRECUE_END_AFTER_CUE_MS is TC.PRECUE_END_AFTER_CUE_MS and D.POSTCUE_START_MS is TC.POSTCUE_START_MS
    from wfield_local import daq_trials
    assert daq_trials.CONTACT_FROM_CUE_S == TC.CONTACT_FROM_CUE_S


def test_precue_split_is_by_peak():
    # a strobe .. cue + 70 ms table (as every v36 licks_precue.csv written before 11-04)
    L = pd.DataFrame({"trial_id": [1, 1, 1, 1, 2], "on_ms": [-900.0, -40.0, -10.0, 55.0, -5.0],
                      "t_ms": [-850.0, -0.1, 20.0, 69.9, 0.0]})
    P, X = D.split_precue_by_peak(L)
    assert P.t_ms.tolist() == [-850.0, -0.1]                       # peak before the cue: pre-cue (onset irrelevant)
    assert X.t_ms.tolist() == [20.0, 69.9, 0.0]                    # cue <= peak < cue + 70 ms: excluded
    assert len(P) + len(X) == len(L)                               # nothing lost: every peak in exactly one table
    e = pd.DataFrame(columns=["trial_id", "t_ms"])
    assert all(len(x) == 0 for x in D.split_precue_by_peak(e))


def test_precue_pass_detects_to_70_and_splits(monkeypatch):
    # the pass still DETECTS strobe .. cue + 70 ms (the lick features stay what they were); only the split moves
    seen = {}

    def fake_epoch(clean, idx, frame, df, a, e, lick_s, cue_s):
        seen["end"] = np.asarray(e, float)
        return pd.DataFrame({"trial_id": [7, 7], "t_ms": [-300.0, 30.0], "t_from_anchor_ms": [700.0, 1030.0]})
    # the chain lives in wfield_local.kinematics_chain since 11-05 (the script re-exports it); `precue_licks` stays in the
    # script and reads the script's name, so patch both
    from wfield_local import kinematics_chain
    monkeypatch.setattr(kinematics_chain, "epoch_licks", fake_epoch)
    monkeypatch.setattr(D, "epoch_licks", fake_epoch)
    idx = pd.DataFrame({"trial_k": [0, 0], "strobe_ms": [-1000.0, -1000.0]})
    P, X = D.precue_and_excluded_licks(None, idx, None, None)
    assert (seen["end"] == 70.0).all()
    assert P.t_ms.tolist() == [-300.0] and X.t_ms.tolist() == [30.0]
    assert "t_from_strobe_ms" in P and "t_from_strobe_ms" in X
    assert D.precue_licks(None, idx, None, None).t_ms.tolist() == [-300.0]


def test_daq_contacts_floor_uses_the_constant(monkeypatch):
    from wfield_local import dlc_frames

    class RV:
        def root(self, name):
            return str(self.tmp)
    rv = RV()
    import tempfile
    rv.tmp = Path(tempfile.mkdtemp())
    d = rv.tmp / "sessions" / "PS92" / "20260806"
    d.mkdir(parents=True)
    (d / "PS92_20260806_s1_trials.csv").write_text("x\n")
    monkeypatch.setattr(dlc_frames, "lick_onsets", lambda *a, **k: np.array([10.05, 10.0699, 10.07, 10.5, 14.0]))
    b = pd.DataFrame({"trial_id": [1], "cue_s": [10.0], "stop_s": [13.5]})
    got = D.daq_contacts_ms("PS92", "20260806", b, rv)[1]
    assert np.allclose(got, [70.0, 500.0])                         # 0-70 ms contacts are not post-cue


def test_precue_rate_denominator_ends_at_the_cue(tmp_path):
    from wfield_local import orofacial_cohort as oc
    tid = np.repeat([1, 2], 3)
    np.savez(tmp_path / "windows.npz", ix_trial_id=tid, ix_t_ms=np.tile([-3000.0, 0.0, 4000.0], 2),
             ix_strobe_ms=np.repeat([-2000.0, -500.0], 3), ix_stop_ms=np.repeat([3000.0, 3500.0], 3))
    dur, _ = oc._windows(tmp_path)
    assert np.allclose(dur.pre_s.to_numpy(), [2.0, 0.5])          # strobe .. CUE (was strobe .. cue + 70 ms)
    assert np.allclose(dur.resp_s.to_numpy(), [2.93, 3.43])       # cue + 70 ms .. stop: unchanged


def test_precue_rate_numerator_drops_the_excluded_licks():
    # session_metrics splits the pre-cue table by peak before counting (old tables still hold the 0-70 ms peaks)
    import inspect

    from wfield_local import orofacial_cohort as oc
    src = inspect.getsource(oc.session_metrics)
    assert "P, X = split_precue_by_peak(P)" in src
    assert src.index("split_precue_by_peak(P)") < src.index('"precue_rate_hz"')


# ---------------------------------------------------------------------------------- encoder lists: every lick kept
def _encoder_session(tmp_path, A, monkeypatch):
    from wfield_local import dlc_frames
    sd = tmp_path / "v36"
    sd.mkdir()
    A.to_csv(sd / "licks_all.csv", index=False)
    d = tmp_path / "sessions" / "PS92" / "20260806"
    d.mkdir(parents=True, exist_ok=True)
    (d / "PS92_20260806_s1_trials.csv").write_text("x\n")
    monkeypatch.setattr(dlc_frames, "lick_onsets", lambda *a, **k: np.array([5.0]))

    class RV:
        def root(self, name):
            return str(tmp_path)
    b = pd.DataFrame({"strobe_s": [0.0], "cue_s": [3.0], "stop_s": [6.0]})
    return RV(), sd, b


def test_encoder_context_licks_unchanged_by_the_relabel(tmp_path, monkeypatch):
    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_outputs as oo
    on = np.array([-800.0, -20.0, 10.0, 60.0, 4000.0])      # ms from the cue
    pk = on + 40.0
    sb, sp = np.full(5, -3000.0), np.full(5, 3000.0)
    # the 10 ms onset (peak 50 ms) sat in licks_precue before 11-04 and sits in licks_excluded_0_70 after: a post-cue
    # onset outside licks.csv = the encoder's LICKS_ALL post_extra list (encoder-classes), by the ONSET label both times
    base = pd.DataFrame({"daq_on_s": 3.0 + on / 1000.0})
    old = base.assign(context=oo.onset_context(on, sb, sp),                     # a pre-11-04 licks_all.csv
                      table=["licks_precue", "licks_precue", "licks_precue", "licks", "licks_iti"])
    new = base.assign(context=oo.lick_context(pk, sb, sp), onset_context=oo.onset_context(on, sb, sp),
                      table=["licks_precue", "licks_excluded_0_70", "licks_excluded_0_70", "licks", "licks_iti"])
    assert "excluded_0_70" in set(new.context)                                  # the relabel did happen
    rv, sd, b = _encoder_session(tmp_path, old, monkeypatch)
    ctx_old, c_old = ce._context_licks(rv, "PS92", "20260806", sd, b, np.zeros(0))
    new.to_csv(sd / "licks_all.csv", index=False)
    ctx_new, c_new = ce._context_licks(rv, "PS92", "20260806", sd, b, np.zeros(0))
    for k in ("pre_cue", "iti", "post_extra"):
        assert np.array_equal(ctx_old[k], ctx_new[k])
    assert np.allclose(ctx_new["post_extra"], [3.01])                           # the 10 ms onset keeps its kernel
    assert np.allclose(ctx_new["pre_cue"], [2.2, 2.98])                         # the -20 ms onset lick is still there
    assert np.array_equal(c_old, c_new)


# ------------------------------------------------------------------------------ rule 2: the response window (classes)
MAX_RT = None


def _rt():
    return float(config.defaults()["decode"]["max_rt_s"]) * 1000.0


def test_max_rt_is_the_engaged_rules_source():
    assert TC.max_rt_ms() == _rt() == 3500.0
    assert np.allclose(TC.response_end_ms([3000.0, 5000.0, np.nan]), [3000.0, 3500.0, 3500.0])


# trial: (position, stop_ms, n_daq_contacts, late DAQ contact ms list, cam4 licks [(on, peak)], cam1 events
#         [(on, peak)], cam4_moved, cam1_moved, jaw_onset_ms, expected class)
CASES = {
    1: ("far_R", 5000.0, 1, [3600.0], [], [], 0, 0, np.nan, "no_response"),         # contact only after max_rt
    2: ("far_R", 5000.0, 1, [3600.0], [(800.0, 850.0)], [], 1, 1, 100.0, "incomplete"),  # + an earlier miss
    3: ("far_R", 5000.0, 2, [3600.0], [], [], 0, 0, np.nan, "contact"),             # one contact in, one after
    4: ("far_R", 5000.0, 0, [], [(3550.0, 3600.0)], [], 0, 0, np.nan, "no_response"),   # video lick after max_rt
    5: ("far_R", 5000.0, 0, [], [], [(3400.0, 3510.0)], 0, 0, np.nan, "no_response"),   # cam1 event peak after
    6: ("far_R", 5000.0, 0, [], [], [], 1, 0, 3700.0, "no_response"),               # jaw moved only after max_rt
    7: ("far_R", 5000.0, 0, [], [], [], 0, 1, 3400.0, "jaw_only"),                  # jaw inside
    8: ("far_R", 3000.0, 0, [], [(2950.0, 2990.0)], [], 0, 0, np.nan, "incomplete"),     # stop earlier than max_rt
    9: ("far_R", 5000.0, 0, [], [(50.0, 65.0)], [], 0, 0, np.nan, "no_response"),   # peak in 0-70 ms: neither
    10: ("far_R", 5000.0, 0, [], [(62.0, 90.0)], [], 0, 0, np.nan, "incomplete"),   # onset 62, peak 90: a response
    11: ("far_R", 5000.0, 0, [], [], [(20.0, 60.0)], 0, 0, np.nan, "no_response"),  # cam1 peak in 0-70 ms
    12: ("far_R", 5000.0, 0, [], [(3450.0, 3500.0)], [], 0, 0, np.nan, "incomplete"),    # peak AT max_rt: inside
}
CUE0 = 100.0                                                       # trial t's cue at CUE0 * t s


def _rw_session(tmp_path, monkeypatch):
    v36 = tmp_path / "v36" / "sessions"
    sd = v36 / "PS92" / "20260818"
    sd.mkdir(parents=True)
    c1 = tmp_path / "cam1"
    c1.mkdir()
    qc = tmp_path / "v36" / "imaging_encoding" / "qc"
    qc.mkdir(parents=True)
    pd.DataFrame([{"trial_id": t, "position": c[0], "n_daq_contacts": c[2]} for t, c in CASES.items()]
                 ).to_csv(sd / "trials.csv", index=False)
    tid = np.repeat(list(CASES), 2)
    stop = np.repeat([c[1] for c in CASES.values()], 2)
    np.savez(sd / "windows.npz", ix_trial_id=tid, ix_t_ms=np.tile([0.0, 4.0], len(CASES)),
             ix_daq_s=tid * CUE0 + np.tile([0.0, 0.004], len(CASES)), ix_strobe_ms=np.full(len(tid), -1000.0),
             ix_stop_ms=stop)
    pd.DataFrame([{"trial_id": t, "on_ms": a, "t_ms": p, "contact": False} for t, c in CASES.items() for a, p in c[4]],
                 columns=["trial_id", "on_ms", "t_ms", "contact"]).to_csv(sd / "licks.csv", index=False)
    pd.DataFrame([{"trial_id": t, "on_ms": a, "t_peak_ms": p, "cam4_matched": False, "in_3d": False, "contact": False,
                   "jaw_only_trial": False, "droplet": False} for t, c in CASES.items() for a, p in c[5]],
                 columns=["trial_id", "on_ms", "t_peak_ms", "cam4_matched", "in_3d", "contact", "jaw_only_trial",
                          "droplet"]).to_csv(c1 / "PS92_20260818_events.csv", index=False)
    pd.DataFrame([{"session": "PS92_20260818", "trial_id": t, "cam4_moved": c[6], "cam1_moved": c[7],
                   "jaw_onset_ms": c[8], "onset_at_cue": False, "jaw_reonset_ms": np.nan} for t, c in CASES.items()]
                 ).to_csv(qc / "jaw_onsets.csv", index=False)
    # the DAQ contacts AFTER max_rt (the contact table's n_daq_contacts already holds them)
    daq = np.sort([t * CUE0 + x / 1000.0 for t, c in CASES.items() for x in c[3]])
    monkeypatch.setattr(TC, "_daq_contact_onsets", lambda a, d: daq)
    TC.jaw_table.cache_clear()
    return v36, c1


def test_response_window_classes(tmp_path, monkeypatch):
    v36, c1 = _rw_session(tmp_path, monkeypatch)
    T = TC.session_trials("PS92_0818", v36=v36, c1=c1).set_index("trial_id")
    assert T.cls.to_dict() == {t: c[-1] for t, c in CASES.items()}
    assert T.loc[3, "n_contacts"] == 1 and T.loc[1, "n_contacts"] == 0
    assert T.loc[6, "jaw_judged"]                                  # judged, and still inside the window


def test_response_window_needs_no_daq_when_stop_is_early(tmp_path, monkeypatch):
    times = pd.DataFrame({"trial_id": [1, 2], "cue_daq": [10.0, 20.0], "stop_ms": [3000.0, 3500.0]})

    def boom(*a):
        raise AssertionError("no DAQ read needed")
    monkeypatch.setattr(TC, "_daq_contact_onsets", boom)
    assert TC._late_contacts("PS92", "20260818", times).tolist() == [0, 0]


def test_cam1_events_need_a_peak(tmp_path, monkeypatch):
    v36, c1 = _rw_session(tmp_path, monkeypatch)
    f = c1 / "PS92_20260818_events.csv"
    pd.read_csv(f).drop(columns="t_peak_ms").to_csv(f, index=False)
    with pytest.raises(ValueError, match="t_peak_ms"):
        TC.session_trials("PS92_0818", v36=v36, c1=c1)


def test_last_lick_anchor_follows_the_class():
    # a lick after max_rt no longer makes its trial a lick trial, so it no longer moves the session's last-lick end
    cls_old = ["contact", "no_response", "incomplete"]             # trial 2 = a lick only after max_rt (old rule)
    cls_new = ["contact", "no_response", "no_response"]
    assert TC.before_session_end(cls_old, [0, 1, 2]).tolist() == [True, True, True]
    assert TC.before_session_end(cls_new, [0, 1, 2]).tolist() == [True, False, False]


def test_class_signature_covers_the_new_rules():
    for f in ("_late_contacts", "response_end_ms", "max_rt_ms"):
        assert f in TC.CLASS_CODE_ROOTS
