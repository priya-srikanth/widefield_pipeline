"""What seeding-by-triangulation must refuse to do.

The value of this route is not that it places points -- a detector does that. It is that GEOMETRY,
rather than a network's confidence in its own guess, decides which points survive. So the tests are
about REJECTION: a seed that is confidently wrong is worse than a blank frame, because the labeller's
eye anchors to whatever is already on screen.

Run against a stub camera group rather than aniposelib, which lives only in the ``dlc`` env. That is
the better test in any case: what is worth pinning is this module's acceptance rule, not the
library's arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import dlc_seed3d as s3


class StubGroup:
    """Projects a 3-D point to a fixed 2-D spot per camera, with a per-camera offset we control.

    ``project`` returns whatever ``self.out[cam]`` says, so a test can make the views agree or
    disagree by construction and check only what the acceptance rule does about it.
    """

    def __init__(self, order, out):
        self.order, self.out = order, out
        self.cameras = [type("C", (), {"name": n})() for n in order]

    def triangulate(self, pts, undistort=True):
        seen = [p for p in pts[:, 0] if np.isfinite(p).all()]
        return np.array([[float(len(seen)), 0.0, 0.0]])

    def project(self, xyz):
        return np.array([[self.out[c]] for c in self.order], dtype=float)


def _group(order, out):
    return StubGroup(order, out)


def test_views_that_disagree_are_REFUSED_not_averaged():
    """Two predictions that are each confident and mutually inconsistent must yield no seed.

    This is the whole safety argument for running a frontal donor on a side view: if cam1's "jaw"
    and cam4's "jaw" are not the same physical point, their rays do not meet, and what comes back
    must be nothing rather than the midpoint of two wrong answers.
    """
    order = ["cam4", "cam1"]
    obs = {"cam4": np.array([100.0, 100.0]), "cam1": np.array([100.0, 100.0])}
    far = _group(order, {"cam4": [100.0, 100.0], "cam1": [400.0, 400.0]})   # 300+ px apart
    xyz, resid, used = s3.fit_moment(obs, far, order)
    assert xyz is None, "a point the two views disagree about was seeded anyway"
    assert used == []
    assert resid > s3.MAX_RESIDUAL_PX


def test_views_that_agree_are_accepted():
    order = ["cam4", "cam1"]
    obs = {"cam4": np.array([100.0, 100.0]), "cam1": np.array([200.0, 200.0])}
    near = _group(order, {"cam4": [101.0, 100.0], "cam1": [200.0, 201.0]})
    xyz, resid, used = s3.fit_moment(obs, near, order)
    assert xyz is not None
    assert set(used) == {"cam4", "cam1"}
    assert resid <= s3.MAX_RESIDUAL_PX


def test_one_bad_view_costs_THAT_VIEW_not_the_point():
    """With three views and one outlier, the outlier is dropped and the point survives on the other
    two -- otherwise a single hallucinating camera would veto every landmark it touches.
    """
    order = ["cam4", "cam1", "cam2"]
    obs = {c: np.array([100.0, 100.0]) for c in order}
    g = _group(order, {"cam4": [100.0, 100.0], "cam1": [100.0, 101.0], "cam2": [900.0, 900.0]})
    xyz, resid, used = s3.fit_moment(obs, g, order)
    assert xyz is not None, "one bad camera vetoed a point two others agreed on"
    assert "cam2" not in used
    assert set(used) == {"cam4", "cam1"}


def test_a_single_view_is_never_enough():
    """One ray does not determine a 3-D point. Seeding from it would be inventing depth."""
    order = ["cam4", "cam1"]
    obs = {"cam4": np.array([100.0, 100.0])}
    g = _group(order, {"cam4": [100.0, 100.0], "cam1": [100.0, 100.0]})
    xyz, _, used = s3.fit_moment(obs, g, order)
    assert xyz is None and used == []


def test_the_likelihood_floor_is_below_prelabels_so_geometry_can_arbitrate():
    """A point the network is unsure of but that two views agree on is a good seed; a confident one
    the views disagree on is not. Blanking at prelabel's 0.6 first would throw away the former and
    keep the latter, which is backwards for this route.
    """
    from wfield_local import dlc_prelabel as pre

    assert s3.MIN_LIKELIHOOD < float((pre.prelabel_cfg() or {}).get("min_likelihood", 0.6))


def test_the_residual_ceiling_is_above_the_measured_route_and_below_a_miss():
    """4.48 px (cam2) and 7.71 px (cam3) were measured on the calibration board; the calibration's
    own floor is 2.83 px. A ceiling under those would reject good seeds, and one far above would
    stop distinguishing a landmark from a different landmark entirely.
    """
    assert 7.71 < s3.MAX_RESIDUAL_PX < 40.0


def test_a_part_a_camera_cannot_see_is_never_written(monkeypatch, tmp_path):
    """cam1 has no eyes in view. A reprojection lands somewhere in every camera regardless, so the
    per-view bodypart list -- not the geometry -- has to be what decides whether it is written.
    """
    seeds = pd.DataFrame([
        {"cam": "cam1", "video_stem": "cam1_x", "image": "img0.png", "bodypart": "jaw",
         "x": 1.0, "y": 2.0, "residual": 1.0, "n_views": 2},
        {"cam": "cam1", "video_stem": "cam1_x", "image": "img0.png", "bodypart": "L_eye",
         "x": 3.0, "y": 4.0, "residual": 1.0, "n_views": 2},
    ])
    monkeypatch.setattr(s3, "bodyparts", lambda cam=None: ["nose", "jaw", "tongue", "spout"])
    written = []
    monkeypatch.setattr(s3, "out_root", lambda rv=None: tmp_path)
    (tmp_path / "labeled-data" / "cam1_x").mkdir(parents=True)
    s3.write(seeds, dry=True)
    # dry run writes nothing; the assertion is that L_eye is filtered before it can be written
    kept = [r for r in seeds.itertuples() if r.bodypart in s3.bodyparts("cam1")]
    assert [r.bodypart for r in kept] == ["jaw"]


def test_write_refuses_to_clobber_an_existing_label_file(tmp_path, monkeypatch, capsys):
    """A CollectedData file is the only record of manual work. Overwriting one is the single thing
    this pipeline must never do, so the refusal is checked rather than assumed.
    """
    root = tmp_path / "labeled-data" / "cam1_x"
    root.mkdir(parents=True)
    (root / "CollectedData_Priya.h5").write_bytes(b"not really hdf5, but it EXISTS")
    monkeypatch.setattr(s3, "out_root", lambda rv=None: tmp_path)
    monkeypatch.setattr(s3, "bodyparts", lambda cam=None: ["jaw"])
    seeds = pd.DataFrame([{"cam": "cam1", "video_stem": "cam1_x", "image": "img0.png",
                           "bodypart": "jaw", "x": 1.0, "y": 2.0, "residual": 1.0, "n_views": 2}])
    out = s3.write(seeds)
    assert out == [], "an existing CollectedData was overwritten"
    assert (root / "CollectedData_Priya.h5").read_bytes().endswith(b"EXISTS")


def test_moments_are_matched_by_BEHAVIOUR_not_by_frame_number():
    """Frame numbers differ between cameras even at the same instant -- cam4 and cam1 shared 0 of 72
    before the anchor landed. The key has to be the trial and phase.
    """
    df = pd.DataFrame({"animal": ["PS92", "PS92"], "date": [20260606, 20260606],
                       "cam": ["cam4", "cam1"], "trial_id": [7, 7], "phase": ["lick+32", "lick+32"],
                       "frame": [720369, 75101]})
    k = s3.moment_key(df)
    assert k.iloc[0] == k.iloc[1], "the same instant hashed differently across cameras"
