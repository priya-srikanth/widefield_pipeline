"""`cam1_licks.detached_static`: a raw cam1 tongue run that never comes back to the mouth and does not move is masked
before cleaning (Priya 11-18); a lick leaving the mouth, a lips lick and a moving point in front are kept."""
import numpy as np

from wfield_local import cam1_licks as C1


def _runs(*runs, gap=10):                    # runs 44 ms apart: beyond the 24 ms bridge
    out, lat, tid, t = [], [], [], []
    tt = 0.0
    for trial, pts in runs:
        for o, la in pts:
            out.append(o); lat.append(la); tid.append(trial); t.append(tt); tt += 4.0
        tt += 4.0 * gap
    n = len(out)
    return np.array(out), np.array(lat), np.ones(n, bool), np.array(tid), np.array(t)


def test_masks_only_detached_stationary_runs():
    drop = [(50 + 0.5 * k % 2, 3.0) for k in range(20)]                # in front, static
    lick = [(-45 + 10 * k, 0.0) for k in range(10)]                    # leaves the mouth line
    lips = [(-50 + 0.3 * k, 1.0) for k in range(8)]                    # static at the lips
    moving = [(40 + 6 * k, 0.0) for k in range(8)]                     # in front but moving (spread > 6 px)
    out, lat, ok, tid, t = _runs((1, drop), (1, lick), (2, lips), (2, moving))
    m = C1.detached_static(out, lat, ok, tid, t)
    n = [20, 10, 8, 8]
    assert m[:n[0]].all() and not m[n[0]:].any()


def test_runs_split_at_gaps_and_trials():
    out, lat, ok, tid, t = _runs((1, [(-40.0, 0.0)] * 3 + [(60.0, 0.0)] * 5))   # one run: touches the mouth -> kept
    assert not C1.detached_static(out, lat, ok, tid, t).any()
    ok2 = ok.copy(); ok2[3] = False                                      # a 1-frame flicker: still one run -> kept
    assert not C1.detached_static(out, lat, ok2, tid, t).any()
    m = C1.detached_static(out, lat, ok2, tid, t, max_gap_ms=0)        # no bridging: the far part is its own run
    assert m[4:].all() and not m[:4].any()


def test_cam4_lick_keeps_a_detached_run():
    import pandas as pd
    out, lat, ok, tid, t = _runs((1, [(50.0, 0.0)] * 6), (1, [(55.0, 0.0)] * 6))
    W = pd.DataFrame({"trial_id": [1], "on_ms": [t[7]], "off_ms": [t[8]]})      # a cam4 lick over the 2nd run
    keep = C1.keep_frames(pd.DataFrame({"trial_id": tid, "t_ms": t}), W)
    m = C1.detached_static(out, lat, ok, tid, t, keep=keep)
    assert m[:6].all() and not m[6:].any()
