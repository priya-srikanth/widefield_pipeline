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


def test_the_tongue_is_never_pre_labelled_however_confident():
    """It fires at 0.99 on the SPOUT while the tongue is out beside it."""
    labels = pl.to_labels(_pred(tongue=(50.0, 60.0, 1.0)), 1.0, 1.0)
    assert np.isnan(_x(labels, "tongue")).all()


def test_the_eyes_never_reach_the_output_at_all():
    """Not in frame on cam4, yet returned at 0.44-0.79 in the top corners."""
    labels = pl.to_labels(_pred(), 1.0, 1.0)
    got = {c[1] for c in labels.columns}
    assert "L_eye" not in got and "R_eye" not in got
    assert "L_spout" not in got and "R_spout" not in got, "the two donor spouts collapse into one"


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


def test_coverage_reports_the_withheld_parts_as_zero():
    cov = pl.coverage(pl.to_labels(_pred(), 1.0, 1.0))
    assert cov["tongue"] == 0.0
    assert cov["nose"] == 1.0


def test_the_donor_project_is_referenced_read_only():
    """`analyze_videos` writes into a project dir; the donor is irreplaceable source data."""
    d = pl.donor()
    assert d["project"] != d["local_copy"]
    assert "MICROSCOPE" in d["project"] and "MICROSCOPE" not in d["local_copy"]
