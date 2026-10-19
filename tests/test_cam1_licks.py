import numpy as np
import pandas as pd

from wfield_local import cam1_licks as C1


def _trial(out, tid=1, t0=0.0, stop=3500.0, pos="far_L"):
    n = len(out)
    return dict(out_px=np.asarray(out, float), lat_px=np.zeros(n), present=np.isfinite(out),
                trial_id=np.full(n, tid), t_ms=t0 + 4.0 * np.arange(n), src_frame=np.arange(n) + 100 * tid,
                stop_ms=np.full(n, stop), position=np.full(n, pos))


def _cat(*ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def test_runs_sizes_and_classes():
    nan = np.nan
    d = _trial([nan, 5, 12, 15, 9, nan, nan, 10, 40, 60, 30, nan, 3, 4, nan])
    E = C1.detect(**d)
    assert E[["on_ms", "off_ms", "n_frames"]].values.tolist() == [[4, 16, 4], [28, 40, 4]]
    assert E.size_class.tolist() == ["lips", "out"]                    # max out 15 <= 20 px; 60 > 20
    assert E.t_peak_ms.tolist() == [12.0, 36.0] and E.lick_idx.tolist() == [0, 1]   # the 2-frame run is dropped


def test_window_and_trial_breaks():
    d1 = _trial([10, 10, 10, 10], tid=1, t0=-8.0)                       # starts before the cue: only 2 frames after
    d2 = _trial([10, 10, 10, 10], tid=2, stop=4.0)                      # trial stops after 2 frames
    d3 = _trial([10, 10, 10, 10, 10, 10], tid=3)
    d3["src_frame"][3:] += 5                                            # a video-frame break splits the run
    E = C1.detect(**_cat(d1, d2, d3))
    assert E.trial_id.tolist() == [3, 3] and E.n_frames.tolist() == [3, 3]


def test_overlap_contact_and_scale():
    E = pd.DataFrame({"trial_id": [1, 1, 2], "on_ms": [100, 400, 100], "off_ms": [140, 420, 140]})
    W = pd.DataFrame({"trial_id": [1], "on_ms": [130.0], "off_ms": [200.0]})
    assert C1.overlaps(E, W).tolist() == [True, False, False]
    assert C1.contact_flags(E, {1: [430.0], 2: [50.0]}).tolist() == [False, True, False]
    assert np.isclose(C1.mm_per_px({"far_L": 90.0, "far_R": -90.0}, {"far_L": -2.0, "far_R": 2.0}), 4 / 180)
