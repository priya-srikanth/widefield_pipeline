import numpy as np
import pandas as pd
import pytest

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


def test_droplet_flag_static_point_away_from_the_mouth_only():
    E = pd.DataFrame({"spread_px": [1.0, 1.0, 1.0, 6.0, 1.0], "range_px": [3.0, 3.0, 3.0, 40.0, 3.0],
                      "med_out_px": [40.0, -60.0, 40.0, 40.0, -20.0]})
    cam1_only = np.array([True, True, False, True, True])
    # static + away from the mouth = droplet; static AT the mouth opening = tongue at the lips; seen by cam4 = tongue held
    # on the spout; moving = a lick
    assert C1.droplet_flags(E, cam1_only).tolist() == [True, False, False, False, True]


def test_span_all_keeps_events_outside_cue_to_stop():
    # Priya 10-20 (encoding window): licks before the cue / after the stop are detected with span="all" only
    d1 = _trial([10, 10, 10, 10], tid=1, t0=-40.0)                      # entirely before the cue
    d2 = _trial([10, 10, 10, 10], tid=2, stop=4.0)                      # trial stops after 2 frames
    post = C1.detect(**_cat(d1, d2))
    whole = C1.detect(**_cat(d1, d2), span="all")
    assert post.empty
    assert whole.trial_id.tolist() == [1, 2] and whole.n_frames.tolist() == [4, 4]
    with pytest.raises(ValueError):
        C1.detect(**_cat(d1), span="pre")
