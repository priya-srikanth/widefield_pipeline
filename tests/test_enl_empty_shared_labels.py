"""A post-stroke epoch with no position common to all three ENL arms is NOT SCORABLE, not a crash.

2026-09-26/27: `enl_decode` raised `max() arg is an empty sequence` (majority_class_floor over an empty
label set) on every post-stroke epoch, three enl_decode_figure steps failed in cascade, and the deck's
failed-steps gate refused to publish two nights running.
"""
from __future__ import annotations

import math

import numpy as np

from wfield_local import enl_decode as E
from wfield_local import nolick_analysis as na


def _scored():
    return {"y": np.array([0, 1, 0]), "pred": np.array([0, 0, 1]), "n_total": 3, "coverage": 1.0,
            "g": np.array([0, 1, 2])}


def test_score_shared_with_no_shared_position_is_a_skipped_result():
    out = E._score_shared(_scored(), target_frac=None, n_perm=10, labels=[])
    assert "skipped" in out and out["positions_used"] == 0 and out["n_total"] == 3


def test_score_shared_still_scores_when_positions_are_shared():
    out = E._score_shared(_scored(), target_frac=None, n_perm=10, labels=[0, 1])
    assert "skipped" not in out and out["positions_used"] == 2 and "balanced_accuracy" in out


def test_majority_class_floor_over_no_labels_is_nan():
    assert math.isnan(na.majority_class_floor(np.array([0, 1, 0]), labels=[]))
    assert na.majority_class_floor(np.array([0, 1, 0]), labels=[0, 1]) == 2 / 3
