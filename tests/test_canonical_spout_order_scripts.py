"""Per-spout figures use the canonical spout order (Priya 10-25): close ipsi, close mid, close contra, far ipsi, far mid,
far contra = `orofacial_cohort.POS_ORDER`. The 11-08 settings audit found three scripts with their own contra-first /
scrambled literals; they now take the shared constant."""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

from wfield_local.orofacial_cohort import POS_ORDER

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_pos_order_is_the_canonical_order():
    assert POS_ORDER == ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")


def test_combined_encoding_figures_pos6():
    assert importlib.import_module("combined_encoding_figures").POS6 == POS_ORDER


def test_frozen_spout_maps_rows():
    assert importlib.import_module("frozen_spout_maps").SPOUTS == POS_ORDER


def test_encoding_key_figures_lick_deficit_rows_follow_pos_order():
    src = inspect.getsource(importlib.import_module("encoding_key_figures").lick_deficits)
    assert 'sp = ("far_contra", "far_mid", "close_contra")' not in src
    assert "for p in POS if p in" in src
