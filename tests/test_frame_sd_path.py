"""frame_sd_path: WFIELD_FRAME_SD_DIR wins; unset -> in_path_x (lane E 11-25)."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_frame_sd_env(monkeypatch, tmp_path):
    import iti_licks as il
    monkeypatch.setenv("WFIELD_FRAME_SD_DIR", str(tmp_path))
    assert il.frame_sd_path("locanmf") == tmp_path / "basis_locanmf" / "frame_sd.csv"


def test_frame_sd_default(monkeypatch):
    import iti_licks as il
    monkeypatch.delenv("WFIELD_FRAME_SD_DIR", raising=False)
    importlib.reload(il)
    assert il.frame_sd_path("svdbasis") == il.in_path_x("basis_svdbasis/frame_sd.csv")
