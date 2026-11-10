"""lick_enl_target_test titles read the ENL window (lp_licks.enl_post_s), never a literal "no lick 2 s" (Priya 11-09)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = (REPO / "scripts" / "lick_enl_target_test.py").read_text(encoding="utf-8")


def test_no_literal_window_in_suptitles():
    sup = re.findall(r"fig\.suptitle\((.*?)fontsize", SRC, flags=re.S)
    assert sup and not [s for s in sup if "no lick 2 s" in s or "ENL, 1 s" in s or "W 1.0 s" in s]


def test_window_follows_enl_post_s(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import lick_enl_target_test as t
    finally:
        sys.path.remove(str(REPO / "scripts"))
    from wfield_local import lp_licks
    monkeypatch.setenv(lp_licks.ENL_WINDOW_ENV, "0.75")
    assert t.enl_window_s() == 0.75
    monkeypatch.delenv(lp_licks.ENL_WINDOW_ENV)
    assert t.enl_window_s() == lp_licks.LP_ENL_WINDOW_S
    assert t.lick_free_rule(1.0) == "sliding lick-free 1 s window"      # primary rule (ead.LF_PRIMARY = slide)
