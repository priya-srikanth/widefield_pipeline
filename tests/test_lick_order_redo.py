"""iti_lick_order.preds_path: WFIELD_LICK_ORDER_REDO=1 rebuilds the cache in WFIELD_ITI_OUT_W (never an upstream copy)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402


def test_redo_writes_to_out_w(tmp_path, monkeypatch):
    monkeypatch.setattr(il, "OUT_W", tmp_path / "build")
    monkeypatch.setenv("WFIELD_ITI_OUT_W", str(tmp_path / "build"))
    monkeypatch.setenv(lo.REDO_ENV, "1")
    p = lo.preds_path()
    assert p.parent == tmp_path / "build" and p.name.startswith("lick_order_preds")


def test_redo_without_redirect_refuses(monkeypatch):
    monkeypatch.delenv("WFIELD_ITI_OUT_W", raising=False)
    monkeypatch.setenv(lo.REDO_ENV, "1")
    with pytest.raises(SystemExit):
        lo.preds_path()
