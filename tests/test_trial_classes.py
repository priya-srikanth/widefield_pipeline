"""wfield_local.trial_classes: THE trial classification (Priya 11-03). One synthetic session, one trial per rule."""
import numpy as np
import pandas as pd

from wfield_local import trial_classes as TC

# trial_id: (position, n_daq_contacts, cam4 lick on_ms list, cam1-only event on_ms list, cam1 saw a tongue,
#            cam4_moved, cam1_moved, onset_at_cue, jaw_reonset_ms, expected class)
CASES = {
    1: ("far_R", 2, [150.0], [], False, 1, 1, False, np.nan, "contact"),
    2: ("far_R", 0, [200.0], [], False, 1, np.nan, False, np.nan, "incomplete"),        # cam4 lick, no contact
    3: ("far_R", 0, [], [300.0], False, 0, 0, False, np.nan, "incomplete"),              # cam1-only lick
    4: ("far_R", 0, [], [], True, 1, 1, False, np.nan, "incomplete"),                    # cam1 saw the tongue
    5: ("far_R", 0, [], [], False, 0, 1, False, np.nan, "jaw_only"),                     # cam4 OR cam1
    6: ("far_R", 0, [], [], False, 0, 0, False, np.nan, "no_response"),
    7: ("far_R", 0, [], [], False, np.nan, np.nan, False, np.nan, "unknown"),            # neither camera judged
    8: ("close_center", 0, [], [], False, 1, 1, False, np.nan, "no_response"),           # jaw unjudgeable: default (11-03)
    9: ("far_R", 0, [], [], False, 1, 1, True, np.nan, "unknown"),                       # moving at cue, no re-onset
    10: ("far_R", 0, [], [], False, 1, 1, True, 400.0, "jaw_only"),                     # ... with a re-onset
    11: ("far_R", 0, [-300.0], [], False, 0, 0, False, np.nan, "no_response"),          # pre-cue lick: outside window
    12: ("far_R", 0, [5000.0], [], False, 0, 0, False, np.nan, "no_response"),          # after the stop: outside
}
STOP_MS, STROBE_MS = 3500.0, -1000.0


def _session(tmp_path):
    v36 = tmp_path / "v36" / "sessions"
    sd = v36 / "PS92" / "20260818"
    sd.mkdir(parents=True)
    c1 = tmp_path / "cam1"
    c1.mkdir()
    qc = tmp_path / "v36" / "imaging_encoding" / "qc"
    qc.mkdir(parents=True)
    pd.DataFrame([{"trial_id": t, "position": c[0], "n_daq_contacts": c[1]} for t, c in CASES.items()]
                 ).to_csv(sd / "trials.csv", index=False)
    tid = np.repeat(list(CASES), 3)
    np.savez(sd / "windows.npz", ix_trial_id=tid, ix_t_ms=np.tile([0.0, 10.0, 20.0], len(CASES)),
             ix_daq_s=tid * 10.0 + np.tile([0.0, 0.01, 0.02], len(CASES)),
             ix_strobe_ms=np.full(len(tid), STROBE_MS), ix_stop_ms=np.full(len(tid), STOP_MS))
    pd.DataFrame([{"trial_id": t, "on_ms": x, "contact": c[1] > 0} for t, c in CASES.items() for x in c[2]],
                 columns=["trial_id", "on_ms", "contact"]).to_csv(sd / "licks.csv", index=False)
    ev = [{"trial_id": t, "on_ms": x, "cam4_matched": False, "in_3d": False, "contact": False, "jaw_only_trial": False,
           "droplet": False} for t, c in CASES.items() for x in c[3]]
    ev += [{"trial_id": t, "on_ms": np.nan, "cam4_matched": True, "in_3d": False, "contact": False,
            "jaw_only_trial": True, "droplet": False} for t, c in CASES.items() if c[4]]
    pd.DataFrame(ev).to_csv(c1 / "PS92_20260818_events.csv", index=False)
    pd.DataFrame([{"session": "PS92_20260818", "trial_id": t, "cam4_moved": c[5], "cam1_moved": c[6],
                   "jaw_onset_ms": 100.0, "onset_at_cue": c[7], "jaw_reonset_ms": c[8]} for t, c in CASES.items()]
                 ).to_csv(qc / "jaw_onsets.csv", index=False)
    TC.jaw_table.cache_clear()
    return v36, c1


def test_every_rule(tmp_path):
    v36, c1 = _session(tmp_path)
    T = TC.session_trials("PS92_0818", v36=v36, c1=c1).set_index("trial_id")
    got = T.cls.to_dict()
    assert got == {t: c[-1] for t, c in CASES.items()}
    assert (T.window == "cue_stop").all()
    assert not T.loc[8, "jaw_judged"] and T.loc[6, "jaw_judged"]
    assert TC.in_gate(T.cls, "working").sum() == 6          # 1, 2, 3, 4, 5, 10 (the jaw-inclusive gate, as on main)
    assert TC.in_sel(T.cls, "working").sum() == 4           # 1, 2, 3, 4 (production in_working: jaw_only 5, 10 out)
    assert TC.in_gate(T.cls, "camera_lick").sum() == 4      # 1, 2, 3, 4


def test_trial_window_counts_pre_cue_licks(tmp_path, monkeypatch):
    v36, c1 = _session(tmp_path)
    monkeypatch.setattr(TC, "_pre_cue_contacts", lambda a, d, T: set())    # no raw DAQ in the test
    T = TC.session_trials("PS92_0818", window="trial", v36=v36, c1=c1).set_index("trial_id")
    assert T.loc[11, "cls"] == "incomplete"                 # the pre-cue lick now counts
    assert T.loc[12, "cls"] == "no_response"                # after the stop: still outside


def test_missing_contact_table_raises(tmp_path):
    v36, c1 = _session(tmp_path)
    f = v36 / "PS92" / "20260818" / "trials.csv"
    pd.read_csv(f).drop(columns="n_daq_contacts").to_csv(f, index=False)
    try:
        TC.session_trials("PS92_0818", v36=v36, c1=c1)
    except ValueError as e:
        assert "n_daq_contacts" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_selection_tag():
    assert TC.selection_tag() == "sel-camera-cue_stop"                # the base tag, as on main (class part: run tags)
    assert TC.selection_tag("daq", "trial") == "sel-daq-trial"


def test_end_of_session_cutoff():
    # two sessions; order = trial index. A: last camera lick at 3 (incomplete), last contact at 1, last reference-spout
    # lick at 0. B: no lick at all -> nothing kept (every trial is after its "last" lick).
    cls = ["contact", "contact", "jaw_only", "incomplete", "jaw_only", "no_response", "jaw_only", "jaw_only"]
    order = [0, 1, 2, 3, 4, 5, 0, 1]
    sess = ["A"] * 6 + ["B"] * 2
    pos = ["close_L", "far_R", "far_R", "far_R", "far_R", "far_R", "far_R", "far_R"]
    keep = lambda a: TC.before_session_end(cls, order, session=sess, position=pos, anchor=a).tolist()  # noqa: E731
    assert keep("lick") == [True] * 4 + [False] * 4
    assert keep("contact") == [True, True] + [False] * 6
    assert keep("close") == [True] + [False] * 7
    assert keep("none") == [True] * 8
    # the jaw-inclusive in_working, as on main (production narrows it: & camera_lick, `in_sel`)
    assert TC.in_working(cls, order, session=sess).tolist() == [True, True, True, True, False, False, False, False]
    assert TC.DEFAULT_END == "lick"


def test_selection_tag_end():
    assert TC.selection_tag() == "sel-camera-cue_stop"                 # no end anchor: no end part
    assert TC.selection_tag(end="lick") == "sel-camera-cue_stop-endlick"
