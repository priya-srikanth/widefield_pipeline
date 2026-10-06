"""Multi-view LP export: the split rule on toy labels, the CSV layout LP 2.4.2 parses, pairing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wfield_local import lp_multiview as M

KP = ["nose", "jaw", "tongue", "spout"]


def _toy():
    # 5 moments x 4 keypoints, two views.
    # m0: paired; nose hidden in BOTH, tongue hidden in cam1 only, jaw/spout labelled in both
    # m1: paired; everything labelled in both
    # m2: cam4 only (unpaired); tongue blank in cam4
    # m3: cam1 only (unpaired); all labelled
    # m4: paired; cam1 frame is all_occluded, cam4 labels jaw only
    lab1 = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]], bool)
    lab4 = np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]], bool)
    pres = {"cam1": np.array([1, 1, 0, 1, 1], bool), "cam4": np.array([1, 1, 1, 0, 1], bool)}
    occ = {"cam1": np.array([0, 0, 0, 0, 1], bool), "cam4": np.zeros(5, bool)}
    return {"cam1": lab1, "cam4": lab4}, pres, occ


def test_split_rule_on_toy_labels():
    lab, pres, occ = _toy()
    v = M.split_visibility(lab, pres, occ)
    c1, c4 = v["cam1"], v["cam4"]
    # m0: nose hidden in every labelled view -> 1 in both; tongue hidden in cam1 but seen in cam4 -> 0
    assert c1[0].tolist() == [1, 2, 0, 2]
    assert c4[0].tolist() == [1, 2, 2, 2]
    assert c1[1].tolist() == c4[1].tolist() == [2, 2, 2, 2]
    # m2: cam1 has no frame -> all 0; in cam4 (the only labelled view) a blank tongue is occluded -> 1
    assert c1[2].tolist() == [0, 0, 0, 0]
    assert c4[2].tolist() == [2, 2, 1, 2]
    # m3: cam4 has no frame -> all 0
    assert c4[3].tolist() == [0, 0, 0, 0]
    assert c1[3].tolist() == [2, 2, 2, 2]
    # m4: all_occluded on cam1 -> 1 everywhere there, even where cam4 labels the jaw; cam4 blanks hidden in
    # every labelled view (cam1 labels nothing) -> 1
    assert c1[4].tolist() == [1, 1, 1, 1]
    assert c4[4].tolist() == [1, 2, 1, 1]


def test_unlabelled_view_never_gets_a_code_from_its_stale_points():
    lab, pres, occ = _toy()
    lab["cam1"][2] = True            # garbage points on a row that is not labelled in cam1
    v = M.split_visibility(lab, pres, occ)
    assert v["cam1"][2].tolist() == [0, 0, 0, 0]
    assert v["cam4"][2].tolist() == [2, 2, 1, 2]     # and they must not count as "labelled elsewhere"


def test_csv_layout_matches_lp_parser(tmp_path):
    """x, y, visible per keypoint in that order; LP reshapes to (N, K, 3) -- emulate it exactly."""
    lab, pres, occ = _toy()
    v = M.split_visibility(lab, pres, occ)
    xy = np.where(lab["cam4"][..., None], 10.0, np.nan) * np.ones((5, 4, 2))
    idx = [M.lp_image_path("PS92", "20260820", "cam4", a) for a in range(5)]
    df = M.view_frame(xy, v["cam4"], idx, KP)
    p = tmp_path / "CollectedData_cam4.csv"
    df.to_csv(p)
    back = pd.read_csv(p, header=[0, 1, 2], index_col=0)
    assert [c[2] for c in back.columns[:3]] == ["x", "y", "visible"]
    raw = back.to_numpy(float).reshape(len(back), -1, 3)
    assert (raw[:, :, 2] == v["cam4"]).all()
    assert set(np.unique(raw[:, :, 2])) <= {0, 1, 2}
    # coordinates only where visible 2 (LP warns on visible 1 with coordinates)
    assert np.isnan(raw[:, :, 0][v["cam4"] != 2]).all()
    assert not np.isnan(raw[:, :, 0][v["cam4"] == 2]).any()
    assert list(back.index) == idx


def test_same_basename_in_every_view_and_session_parse():
    a = M.lp_image_path("PS93", "20260904", "cam1", 1234)
    b = M.lp_image_path("PS93", "20260904", "cam4", 1234)
    assert a.split("/")[-1] == b.split("/")[-1]
    folder = a.split("/")[1]
    assert folder.rsplit("_", 1) == ["PS93_20260904", "cam1"]     # LP's <session>_<view> discovery


def test_map_frame_inverts_and_pairs_within_tolerance():
    t4 = {"fs_daq": 5000.0, "slope_daqSample_per_camFrame": 20.0, "intercept_daqSample": 1000.0}
    t1 = {"fs_daq": 5000.0, "slope_daqSample_per_camFrame": 20.0004, "intercept_daqSample": 1500.0}
    f4 = np.array([100, 5000, 90000])
    f1 = M.map_frame(f4, t4, t1)
    assert np.abs(M.map_frame(f1, t1, t4) - f4).max() <= 1
    r1 = pd.DataFrame({"animal": "A", "date": "d", "anchor": [100, 5001, 7000]})
    r4 = pd.DataFrame({"animal": "A", "date": "d", "anchor": [100, 5000, 8000]})
    m = M.pair_moments({"cam1": r1, "cam4": r4}, tol=1)
    assert m.paired.sum() == 2 and len(m) == 4
    assert sorted(m.delta.dropna().astype(int).tolist()) == [0, 1]
    m0 = M.pair_moments({"cam1": r1, "cam4": r4}, tol=0)
    assert m0.paired.sum() == 1 and len(m0) == 5
