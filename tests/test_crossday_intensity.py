"""crossday_intensity is cohort-scoped (Priya 2026-11-05: cohort2's cross-day ROI intensity = cohort2 animals only)."""
import numpy as np

from wfield_local import crossday_intensity as ci
from wfield_local import paths
from wfield_local import preprocess_deck as pd


def _session(root, date, folder):
    d = root / date / folder / "motion_corrected" / "wfield_local_results"
    d.mkdir(parents=True)
    fa = np.full((2, 40, 40), 100.0)
    fa[:, 10:30, 10:30] = 1000.0
    np.save(d / "frames_average.npy", fa)


def test_analyze_keeps_only_the_given_animals_and_tags_490(tmp_path):
    _session(tmp_path, "20260811", "PS92_20260811_100000")
    _session(tmp_path, "20261104", "PS121_490_20261104_122657")
    _session(tmp_path, "20261104", "PS118_20261104_095017")
    ci.LED490.clear()
    rows = ci.analyze(tmp_path, animals=["PS118", "PS121"])
    assert sorted(r[0] for r in rows) == ["PS118", "PS121"]
    assert ci.LED490 == {("PS121", "20261104")}
    assert len(ci.analyze(tmp_path)) == 3                     # no filter = every animal, as before


def test_out_dir_is_scoped_only_for_a_cohort_with_a_subdir(monkeypatch):
    monkeypatch.setattr(paths, "cohort_spec", lambda name=None: {"output_subdir": ""})
    assert ci.out_dir_for("M:/x/xday") == "M:/x/xday"
    monkeypatch.setattr(paths, "cohort_spec", lambda name=None: {"output_subdir": "cohort2"})
    assert ci.out_dir_for("M:/x/xday/") == "M:/x/xday/cohort2"


def test_cohort2_deck_never_falls_back_to_the_unscoped_figure(tmp_path, monkeypatch):
    xday, lab = tmp_path / "xday", tmp_path / "labcams"
    xday.mkdir()
    lab.mkdir()
    (xday / "crossday_raw_intensity.png").write_bytes(b"cohort1")
    monkeypatch.setattr(paths, "cohort_spec", lambda name=None: {"output_subdir": "cohort2"})
    assert pd._crossday_intensity(str(xday), str(lab)) is None
    (xday / "cohort2").mkdir()
    (xday / "cohort2" / "crossday_raw_intensity.png").write_bytes(b"cohort2")
    assert pd._crossday_intensity(str(xday), str(lab)).endswith("cohort2/crossday_raw_intensity.png")
    monkeypatch.setattr(paths, "cohort_spec", lambda name=None: {"output_subdir": ""})
    assert pd._crossday_intensity(str(xday), str(lab)).endswith("xday/crossday_raw_intensity.png")
