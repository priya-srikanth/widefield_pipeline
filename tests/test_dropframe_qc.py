"""Tests for the Blackfly/Bonsai dropped-frame QC (wfield_local.dropframe_qc)."""
import numpy as np
import pytest

from wfield_local import dropframe_qc as dq
from wfield_local.writeguard import WriteGuardError


def _write_csv(path, ids, ts, gpio=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i, t in zip(ids, ts):
            fh.write(f"{i},{t},{gpio}\n")


def test_analyze_csv_detects_a_drop(tmp_path):
    # ids 100..109 with 105 missing (one dropped frame); its timestamp gap is doubled (8 ms).
    ids = [100, 101, 102, 103, 104, 106, 107, 108, 109]
    ts = [0, 4, 8, 12, 16, 24, 28, 32, 36]           # in ms; the 104->106 step is 8 ms
    ts_ns = [t * 1_000_000 for t in ts]
    f = tmp_path / "PS92" / "cam1_2026-01-01T00_00_00.csv"
    _write_csv(f, ids, ts_ns)
    s = dq.analyze_csv(f)
    assert s["cam"] == "cam1" and s["recording"] == "2026-01-01T00_00_00"
    assert s["rows"] == 9 and s["id_span"] == 10 and s["dropped"] == 1
    assert s["drop_pct"] == 10.0
    assert s["gap_events"] == 1 and s["max_gap_frames"] == 1
    assert s["ts_gaps_gt_1p5x"] == 1 and s["max_dt_ms"] == 8.0


def test_analyze_csv_clean_contiguous(tmp_path):
    n = 500
    ids = list(range(1000, 1000 + n))
    ts_ns = [i * 4_000_000 for i in range(n)]        # exactly 4 ms apart
    f = tmp_path / "PS93" / "cam2_2026-01-01T01_02_03.csv"
    _write_csv(f, ids, ts_ns)
    s = dq.analyze_csv(f)
    assert s["dropped"] == 0 and s["gap_events"] == 0 and s["max_gap_frames"] == 0
    assert s["ts_gaps_gt_1p5x"] == 0 and s["mean_dt_ms"] == 4.0 and s["cam"] == "cam2"


def test_scan_and_summary_roundtrip(tmp_path):
    for animal, cam in (("PS92", "cam1"), ("PS94", "cam1")):
        ids = list(range(0, 100))
        _write_csv(tmp_path / animal / f"{cam}_2026-01-01T00_00_00.csv", ids,
                   [i * 4_000_000 for i in range(100)])
    rows = dq.scan(tmp_path, "20260101")
    assert [r["session"] for r in rows] == ["PS92", "PS94"]        # ordered by animal
    csv_path, txt_path = dq.write_summary(rows, tmp_path, "20260101")
    head, first = csv_path.read_text().splitlines()[:2]
    assert head == ",".join(dq.CSV_COLUMNS)
    assert first.startswith("20260101,PS92,cam1,")
    assert "Dropped-frame QC - 20260101" in txt_path.read_text()
    assert "No dropped frames" in txt_path.read_text()


def test_scan_animals_filter(tmp_path):
    for a in ("PS92", "PS94"):
        _write_csv(tmp_path / a / "cam1_2026-01-01T00_00_00.csv",
                   range(100), [i * 4_000_000 for i in range(100)])
    rows = dq.scan(tmp_path, "20260101", animals=["PS94"])
    assert [r["session"] for r in rows] == ["PS94"]


def test_write_summary_is_guarded():
    # writing the summary into another person's MICROSCOPE folder must be refused (ground rule 1)
    with pytest.raises(WriteGuardError):
        dq.write_summary([], "N:/MICROSCOPE/Rich/data", "20260101")
