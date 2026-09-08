"""A pre-label is a claim a human will trust, so the tests are about what must NOT be written.

The donor network is confidently wrong in two specific ways on this rig -- eyes it cannot see, and a
tongue it puts on the spout -- and neither is separable by likelihood. A blank cell is what the
labelling GUI shows as "place this"; a wrong point invites being accepted rather than checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import dlc_prelabel as pl

DONOR_BPS = ["nose", "jaw", "tongue", "L_whiskers_1", "L_whiskers_2", "L_whiskers_3",
             "R_whiskers_1", "R_whiskers_2", "R_whiskers_3", "L_eye", "R_eye",
             "L_spout", "R_spout"]


def _pred(n=3, **over):
    """A donor prediction table: every bodypart confident at (100, 200) unless overridden."""
    cols, data = [], []
    for bp in DONOR_BPS:
        x, y, p = over.get(bp, (100.0, 200.0, 0.99))
        for c, v in (("x", x), ("y", y), ("likelihood", p)):
            cols.append((bp, c))
            data.append(np.full(n, v, dtype=float))
    return pd.DataFrame(np.column_stack(data), columns=pd.MultiIndex.from_tuples(cols))


def _x(labels, bp):
    return labels[(pl.SCORER, bp, "x")].to_numpy()


def test_anything_listed_as_never_prelabel_stays_blank_however_confident(monkeypatch):
    """The withholding mechanism, tested independently of who is currently on the list.

    `tongue` was on it for one afternoon on the strength of a single overlay, and came off after
    twelve more overlays and the lick-onset scoring. The list is a config decision; that it is
    OBEYED is the invariant.
    """
    real = pl.prelabel_cfg()
    monkeypatch.setattr(pl, "prelabel_cfg", lambda: {**real, "never_prelabel": ["jaw"]})
    labels = pl.to_labels(_pred(jaw=(50.0, 60.0, 1.0)), 1.0, 1.0)
    assert np.isnan(_x(labels, "jaw")).all()
    assert not np.isnan(_x(labels, "nose")).any()


def test_the_tongue_is_predicted_at_its_own_scale():
    """Scored against DAQ lick onsets the tongue is found on 2x as many true tongue-out frames at
    native scale as at 0.45, while the nose goes the other way (8% -> 97%). One scale for the whole
    frame would have to sacrifice one of them."""
    assert pl.scale_for("tongue") == 1.0
    assert pl.scale_for("nose") == float(pl.prelabel_cfg()["scale"])
    assert pl.scale_for("tongue") != pl.scale_for("nose")


def test_every_bodypart_is_predicted_at_exactly_one_scale():
    """A bodypart in two groups would be filled twice and the merge would hide which pass won."""
    from wfield_local.dlc_frames import bodyparts
    groups = pl.scale_groups()
    flat = [bp for bps in groups.values() for bp in bps]
    assert sorted(flat) == sorted(bodyparts())
    assert len(flat) == len(set(flat))


def test_only_restricts_which_bodyparts_a_pass_fills():
    labels = pl.to_labels(_pred(), 1.0, 1.0, only=["nose"])
    assert not np.isnan(_x(labels, "nose")).any()
    assert np.isnan(_x(labels, "jaw")).all()


def test_merging_scale_passes_never_lets_one_overwrite_another():
    a = pl.to_labels(_pred(nose=(10.0, 10.0, 0.99)), 1.0, 1.0, only=["nose"])
    b = pl.to_labels(_pred(jaw=(20.0, 20.0, 0.99)), 1.0, 1.0, only=["jaw"])
    merged = pl.merge_scales([a, b])
    assert _x(merged, "nose")[0] == 10.0
    assert _x(merged, "jaw")[0] == 20.0

    # Order must not matter -- each pass only ever fills cells the others left NaN.
    other = pl.merge_scales([b, a])
    assert _x(other, "nose")[0] == 10.0 and _x(other, "jaw")[0] == 20.0


def test_a_camera_only_ever_gets_columns_for_what_it_can_see():
    """The eyes are out of frame on cam4 yet returned at 0.44-0.79 in its top corners, and cam1
    looks up from below so has no nose. Anatomy decides the columns -- not a second exclusion list
    that could disagree with it."""
    cam4 = {c[1] for c in pl.to_labels(_pred(), 1.0, 1.0, cam="cam4").columns}
    assert "L_eye" not in cam4 and "R_eye" not in cam4
    assert "nose" in cam4
    assert "L_spout" not in cam4 and "R_spout" not in cam4, "the two donor spouts collapse into one"

    cam1 = {c[1] for c in pl.to_labels(_pred(), 1.0, 1.0, cam="cam1").columns}
    assert "nose" not in cam1, "cam1 is a bottom view -- the nose is not in it"
    assert not any("whisker" in b for b in cam1), "no identifiable whiskers from below"
    assert {"jaw", "tongue", "spout"} <= cam1

    cam2 = {c[1] for c in pl.to_labels(_pred(), 1.0, 1.0, cam="cam2").columns}
    assert "L_eye" in cam2 and "R_eye" not in cam2, "a side view sees ONE eye"
    assert not any(b.startswith("R_whisker") for b in cam2)


def test_the_side_views_carry_opposite_laterality():
    """PS93's deficit is on the RIGHT, so which camera holds which side is not a cosmetic label."""
    from wfield_local.dlc_frames import bodyparts, role

    assert role("cam2") == "side_left" and role("cam3") == "side_right"
    left, right = set(bodyparts("cam2")), set(bodyparts("cam3"))
    assert "L_eye" in left and "R_eye" in right
    assert all(b.startswith("L_") for b in left if "whisker" in b)
    assert all(b.startswith("R_") for b in right if "whisker" in b)


def test_low_confidence_points_are_left_blank_rather_than_placed():
    labels = pl.to_labels(_pred(nose=(10.0, 20.0, 0.10)), 1.0, 1.0)
    assert np.isnan(_x(labels, "nose")).all()

    labels = pl.to_labels(_pred(nose=(10.0, 20.0, 0.95)), 1.0, 1.0)
    assert not np.isnan(_x(labels, "nose")).any()


def test_the_threshold_is_inclusive_at_its_own_value():
    thresh = float(pl.prelabel_cfg()["min_likelihood"])
    labels = pl.to_labels(_pred(nose=(10.0, 20.0, thresh)), 1.0, 1.0)
    assert not np.isnan(_x(labels, "nose")).any()


def test_the_spout_takes_whichever_donor_point_fires_harder():
    """Which of the old fixed spouts responds depends on where the ONE moving spout is."""
    p = _pred(L_spout=(400.0, 500.0, 0.30), R_spout=(255.0, 505.0, 0.90))
    assert _x(pl.to_labels(p, 1.0, 1.0), "spout")[0] == 255.0

    p = _pred(L_spout=(439.0, 500.0, 0.88), R_spout=(255.0, 505.0, 0.31))
    assert _x(pl.to_labels(p, 1.0, 1.0), "spout")[0] == 439.0


def test_the_spout_is_blank_when_neither_donor_point_is_confident():
    p = _pred(L_spout=(400.0, 500.0, 0.2), R_spout=(255.0, 505.0, 0.3))
    assert np.isnan(_x(pl.to_labels(p, 1.0, 1.0), "spout")).all()


def test_coordinates_come_back_in_ORIGINAL_pixels():
    """Inference runs on a downscaled copy; a label in scaled pixels lands ~2x off on the real frame.

    This is the whole scale trick's failure mode: get the resize right, forget the inverse, and every
    seeded point is silently wrong by a factor of two.
    """
    labels = pl.to_labels(_pred(nose=(144.0, 288.0, 0.99)), 320 / 680, 320 / 680)
    assert _x(labels, "nose")[0] == pytest.approx(144.0 * 680 / 320)


def test_non_square_scaling_uses_each_axis_separately():
    """cam2/cam3 are 600x450; one scale factor for both axes would skew every y."""
    labels = pl.to_labels(_pred(nose=(100.0, 100.0, 0.99)), 0.5, 0.25)
    assert _x(labels, "nose")[0] == pytest.approx(200.0)
    assert labels[(pl.SCORER, "nose", "y")].to_numpy()[0] == pytest.approx(400.0)


def test_the_output_columns_are_exactly_this_projects_bodyparts():
    labels = pl.to_labels(_pred(), 1.0, 1.0)
    from wfield_local.dlc_frames import bodyparts
    assert [c[1] for c in labels.columns][::2] == bodyparts()
    assert labels.columns.names == ["scorer", "bodyparts", "coords"]


def test_a_donor_bodypart_this_project_does_not_have_is_simply_absent():
    """And one this project HAS but the donor lacks comes back blank, not as a KeyError."""
    p = _pred().drop(columns=[("jaw", "x"), ("jaw", "y"), ("jaw", "likelihood")])
    labels = pl.to_labels(p, 1.0, 1.0)
    assert np.isnan(_x(labels, "jaw")).all()
    assert not np.isnan(_x(labels, "nose")).any()


def test_existing_labels_are_never_overwritten(tmp_path, monkeypatch):
    """The entire point is to save manual work; a re-run must not discard it."""
    root = tmp_path / "labeled-data" / "cam4_stem"
    root.mkdir(parents=True)
    (root / f"CollectedData_{pl.SCORER}.h5").write_bytes(b"corrected by hand")
    monkeypatch.setattr(pl, "out_root", lambda rv=None: tmp_path)

    index = pd.DataFrame({"video_stem": ["cam4_stem"] * 3,
                          "image": [f"img{i}.png" for i in range(3)],
                          "path": [""] * 3})
    written = pl.write_labels(pl.to_labels(_pred(), 1.0, 1.0), index, rv=None)
    assert written == []
    assert (root / f"CollectedData_{pl.SCORER}.h5").read_bytes() == b"corrected by hand"


def test_written_labels_carry_dlcs_three_level_row_index(tmp_path, monkeypatch):
    """DLC reads rows as ('labeled-data', <video stem>, <image>); anything else it ignores."""
    pytest.importorskip("tables", reason="the .h5 write needs pytables (present in the dlc env)")
    (tmp_path / "labeled-data" / "cam4_stem").mkdir(parents=True)
    monkeypatch.setattr(pl, "out_root", lambda rv=None: tmp_path)
    index = pd.DataFrame({"video_stem": ["cam4_stem"] * 2,
                          "image": ["img0000001.png", "img0000002.png"], "path": ["", ""]})
    pl.write_labels(pl.to_labels(_pred(n=2), 1.0, 1.0), index, rv=None)

    got = pd.read_hdf(tmp_path / "labeled-data" / "cam4_stem" / f"CollectedData_{pl.SCORER}.h5")
    assert list(got.index[0]) == ["labeled-data", "cam4_stem", "img0000001.png"]


def test_a_missing_pytables_fails_before_anything_is_written(tmp_path, monkeypatch):
    """A run that wrote CSVs then died on the first to_hdf leaves folders that LOOK labelled.

    pytables is in the `dlc` env and not in `locanmf`, so running this from the wrong env is the
    likely mistake, and it has to be a refusal rather than a half-written labelling set.
    """
    import builtins
    real = builtins.__import__

    def no_tables(name, *a, **kw):
        if name == "tables":
            raise ImportError("no pytables")
        return real(name, *a, **kw)

    (tmp_path / "labeled-data" / "cam4_stem").mkdir(parents=True)
    monkeypatch.setattr(pl, "out_root", lambda rv=None: tmp_path)
    monkeypatch.setattr(builtins, "__import__", no_tables)
    index = pd.DataFrame({"video_stem": ["cam4_stem"], "image": ["img1.png"], "path": [""]})
    with pytest.raises(RuntimeError, match="dlc"):
        pl.write_labels(pl.to_labels(_pred(n=1), 1.0, 1.0), index, rv=None)
    monkeypatch.undo()
    assert list((tmp_path / "labeled-data" / "cam4_stem").iterdir()) == []


def test_coverage_reports_what_the_labeller_still_has_to_place(monkeypatch):
    real = pl.prelabel_cfg()
    monkeypatch.setattr(pl, "prelabel_cfg", lambda: {**real, "never_prelabel": ["jaw"]})
    cov = pl.coverage(pl.to_labels(_pred(), 1.0, 1.0))
    assert cov["jaw"] == 0.0, "a withheld part must read as 0%, which is what prints 'by hand'"
    assert cov["nose"] == 1.0


def test_the_donor_project_is_referenced_read_only():
    """`analyze_videos` writes into a project dir; the donor is irreplaceable source data."""
    d = pl.donor()
    assert d["project"] != d["local_copy"]
    assert "MICROSCOPE" in d["project"] and "MICROSCOPE" not in d["local_copy"]
