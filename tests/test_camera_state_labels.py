"""CAMERA-LABELED behavioral states (WIDEFIELD_STATE_LABELS=camera, CC-B 2026-11-28).

Pins: (a) DAQ is the default and the DAQ mode is byte-identical to before (segments hash computed on origin/main
2524aff4; the frozen spec is exactly the pre-change `make_spec` call, no extra key, align "segments", run tag
unchanged); (b) the camera rules on synthetic events: licking anchored on the first FUSED lick after the cue,
running dropped by any fused lick / jaw-only onset in the window, quiet dropped by a fused lick / jaw-only onset in the
window or its rest lick buffer (0.5 s before / 1.0 s after each event), segments outside video coverage dropped;
(c) the camera spec carries the mode, so camera and DAQ entries cannot collide.
"""
import hashlib

import numpy as np
import pytest

from wfield_local import frozen_models as fm
from wfield_local import locomotor_decoder as ld
from wfield_local import locomotor_state as ls

FS = 5000.0


def _events():
    """Two quiet periods, two running bouts, licks after two cues (DAQ samples)."""
    s = lambda t: int(t * FS)                                    # noqa: E731
    return {"fs": FS,
            "quiet_starts": np.array([s(10), s(40)]), "quiet_stops": np.array([s(14), s(44)]),
            "running_starts": np.array([s(20), s(60)]), "running_stops": np.array([s(24), s(63)]),
            "lick_onsets": np.array([s(30.2), s(30.4), s(50.3), s(61.5)]),
            }


CUES = np.array([int(30.0 * FS), int(50.0 * FS), int(61.0 * FS)])


def _h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        a = np.asarray(a)
        m.update((a.astype(str) if a.dtype == object else a.astype(np.int64)).tobytes())
    return m.hexdigest()[:16]


# ------------------------------------------------------------------ (a) DAQ mode unchanged
def test_default_is_daq_and_bad_value_raises(monkeypatch):
    monkeypatch.delenv(ls.STATE_LABELS_ENV, raising=False)
    assert ls.state_labels() == "daq" and not ls.camera_labels_on() and ls.label_title_suffix() == ""
    monkeypatch.setenv(ls.STATE_LABELS_ENV, "camera")
    assert ls.camera_labels_on() and "fused four-camera licks" in ls.label_title_suffix()
    monkeypatch.setenv(ls.STATE_LABELS_ENV, "video")
    with pytest.raises(ValueError):
        ls.state_labels()


def test_daq_segments_byte_identical_to_main():
    """Hash computed with origin/main 2524aff4's `three_way_segments` on these events (before this branch)."""
    smp, lab, per, nd = ls.three_way_segments(_events(), lick_mode="postcue", cue_samples=CUES, lick_window_s=1.0)
    assert (_h(smp, lab, per), nd) == (DAQ_HASH, DAQ_NDROP)


DAQ_HASH = "05980692f9fb75d1"
DAQ_NDROP = 3


def _no_store(monkeypatch):
    seen = []

    def fake(spec, fit, *, meta=None, refreeze=None, log=print):
        seen.append(spec)
        return fit(), fm.STATUS_NEW
    monkeypatch.setattr(fm, "load_or_fit", fake)
    monkeypatch.setattr(fm, "_session_sig", lambda lab: "sig")
    monkeypatch.setattr(fm.config, "animal_of", lambda lab: lab.split("_")[0])
    return seen


def _got(seed=0):
    rng = np.random.default_rng(seed)

    def sess(n):
        y = np.array(list(ls.THREE_WAY) * n)
        return rng.normal(size=(len(y), 4)) + np.array([list(ls.THREE_WAY).index(v) for v in y])[:, None], y
    return {lab: (*sess(30), None, d) for lab, d in (("PSX_0801", -3), ("PSX_0805", -1), ("PSX_0820", 3))}


def test_daq_spec_is_the_pre_change_spec(monkeypatch):
    monkeypatch.delenv(ls.STATE_LABELS_ENV, raising=False)
    seen = _no_store(monkeypatch)
    got = _got()
    ld.frozen_state_models("PSX", "abc", got, scaling="session_bal_em")
    pre = {k: v for k, v in got.items() if v[3] <= 0}
    want = fm.make_spec("PSX", "state", align="segments", source="joint_abc", train_labels=list(pre), basis_id="abc",
                        n_features=4, post_s=ls.SEGMENT_S, zscore=True,
                        train_data={lab: v[:3] for lab, v in sorted(pre.items())},
                        scaling="session_bal_em", scaling_params=ld.bal_em_params())
    assert seen[0] == want and "state_labels" not in seen[0] and seen[0]["align"] == "segments"


def test_daq_run_tag_has_no_state_part(monkeypatch):
    from wfield_local import run_mode
    monkeypatch.delenv(ls.STATE_LABELS_ENV, raising=False)
    assert run_mode.statelab_part() == "" and "statecam" not in run_mode.parts()
    monkeypatch.setenv(ls.STATE_LABELS_ENV, "camera")
    assert run_mode.statelab_part() == "statecam" and "statecam" in run_mode.parts()
    assert "statecam" not in run_mode.cache_parts()                     # output-only


# ------------------------------------------------------------------ (c) camera spec
def test_camera_spec_carries_the_mode(monkeypatch):
    from wfield_local import camera_state_labels as csl
    seen = _no_store(monkeypatch)
    monkeypatch.setattr(csl, "attempts_path", lambda: "X:/attempts.csv")
    got = _got()
    monkeypatch.delenv(ls.STATE_LABELS_ENV, raising=False)
    ld.frozen_state_models("PSX", "abc", got, scaling="session_bal_em")
    monkeypatch.setenv(ls.STATE_LABELS_ENV, "camera")
    ld.frozen_state_models("PSX", "abc", got, scaling="session_bal_em")
    daq, cam = seen
    assert cam["align"] == "segments-camlabels" and cam["state_labels"]["mode"] == "camera"
    assert cam["state_labels"]["rest_lick_buffer_s"] == [0.5, 1.0]
    assert fm.spec_id(daq) != fm.spec_id(cam)


# ------------------------------------------------------------------ (b) camera rules
def _cam(ev=None, cam=(), jaw=(), cov=None, buffer_s=(0.5, 1.0)):
    ev = ev or _events()
    s = lambda t: np.round(np.asarray(t, float) * FS).astype(np.int64)  # noqa: E731
    lo, hi = (None, None) if cov is None else (s([c[0] for c in cov]), s([c[1] for c in cov]))
    return ls.camera_three_way_segments(ev, cue_samples=CUES, cam_licks=s(cam), jaw_onsets=s(jaw),
                                        cov_lo=lo, cov_hi=hi, buffer_s=buffer_s, lick_window_s=1.0)


def _times(smp, lab, which):
    return sorted(float(x) / FS for x, y in zip(smp, lab) if y == which)


def test_camera_licking_anchors_on_first_fused_lick_after_cue():
    smp, lab, _p, _n, _i = _cam(cam=[30.18, 30.4, 50.29])
    assert _times(smp, lab, "licking") == [30.18, 50.29]
    # the DAQ licks of `events` play no part: no fused lick after cue 61 -> no licking window there
    assert 61.5 not in _times(smp, lab, "licking")


def test_camera_running_dropped_by_fused_lick_or_jaw_in_window():
    base = _times(*_cam()[:2], "running")
    assert 23.0 in base                                     # bout 20-24 tiled backwards: 23, 22, 21, 20
    smp, lab, *_ = _cam(cam=[23.5])
    assert 23.0 not in _times(smp, lab, "running") and 22.0 in _times(smp, lab, "running")
    smp, lab, *_ = _cam(jaw=[21.2])
    assert 21.0 not in _times(smp, lab, "running") and 22.0 in _times(smp, lab, "running")
    smp, lab, *_ = _cam(jaw=[24.0])                         # at t1 (exclusive): kept
    assert 23.0 in _times(smp, lab, "running")


def test_camera_quiet_buffer_edges():
    base = _times(*_cam()[:2], "quiet")
    assert base == [10.0, 11.0, 12.0, 13.0, 40.0, 41.0, 42.0, 43.0]
    # a lick 0.4 s AFTER window 13-14 ends: it excludes [13.9, 15.4], which overlaps [13, 14) only
    smp, lab, *_ = _cam(cam=[14.4])
    assert _times(smp, lab, "quiet") == [10.0, 11.0, 12.0, 40.0, 41.0, 42.0, 43.0]
    # a jaw-only onset 0.4 s BEFORE window 40-41 excludes [39.1, 40.6] -> drops 40, keeps 41
    smp, lab, *_ = _cam(jaw=[39.6])
    assert 40.0 not in _times(smp, lab, "quiet") and 41.0 in _times(smp, lab, "quiet")
    # 1.0 s before the window start exactly: [t0-1.0, ...) is inclusive -> dropped; 1.01 s before -> kept
    smp, lab, *_ = _cam(cam=[9.0])
    assert 10.0 not in _times(smp, lab, "quiet")
    smp, lab, *_ = _cam(cam=[8.99])
    assert 10.0 in _times(smp, lab, "quiet")


def test_camera_coverage_cut_and_info():
    smp, lab, _p, _n, info = _cam(cov=[(0.0, 12.0), (19.5, 70.0)])
    q = _times(smp, lab, "quiet")
    assert q == [10.0, 11.0, 40.0, 41.0, 42.0, 43.0] and info["uncovered"] == 2


def test_camera_mode_needs_postcue(monkeypatch):
    from wfield_local import locomotor_features as lf
    monkeypatch.setenv(ls.STATE_LABELS_ENV, "camera")
    monkeypatch.setattr(lf, "_load_cue_events", lambda h5: {"cue_samples": CUES})
    with pytest.raises(ValueError):
        lf.segment_features({"label": "PSX_0801", "h5": None}, np.zeros((2, 10)), _events(), fs_img=31.23,
                            lick_mode="bout")
