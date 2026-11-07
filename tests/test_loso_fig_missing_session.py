"""11-07: under the camera gate a session with no camera working trials (no behaviour video, PS93_0813) drops out of the
frozen decoder's held-out set but stays in ``labels``; `_loso_fig` raised KeyError drawing it (g5_joint_xsession)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from wfield_local.locanmf_frozen_decoder import _loso_fig  # noqa: E402


def _res(labels, scored):
    return {"labels": labels, "per_session": {lab: 0.5 for lab in scored},
            "within_session": {lab: 0.55 for lab in labels}, "transfer_cost": -0.05,
            "ood": {"engaged_H": 0.6, "nolick_H": 0.8, "shuffle_H": 0.95}}


def test_session_without_loso_score_is_skipped(tmp_path):
    res = {"PS93": _res(["PS93_0812", "PS93_0813", "PS93_0814"], ["PS93_0812", "PS93_0814"]),
           "PS92": _res(["PS92_0812"], ["PS92_0812"])}
    p = _loso_fig(res, tmp_path, align="lick", basis="joint")
    assert p.exists()
