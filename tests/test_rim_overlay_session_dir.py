"""lick_rim_overlay reads lick times from the run's own ROI extract (`il.SESS_NAME`), not a hardcoded sessions_roi2
that no run writes (D9 11-08: e_rim_overlay skipped for a missing input)."""
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "scripts" / "lick_rim_overlay.py").read_text(encoding="utf-8")


def test_no_hardcoded_roi_version():
    assert '"sessions_roi2"' not in SRC
    assert "il.XOUT / il.SESS_NAME" in SRC
