"""scripts/trial_attempts.session_summary: the attempt-rate denominator is the canonical period, the execution
composite (time to contact) and components use attempted trials only, and the lateral error is referenced to the
animal's own August pre."""
import numpy as np
import pandas as pd
import pytest

from scripts import trial_attempts as ta


def _trials(label, pos, rows):
    """rows: (cls, canonical, contact_ms, angle)"""
    out = []
    for i, (cls, can, con, ang) in enumerate(rows):
        att = cls in ("hit", "incomplete", "jaw_only")
        out.append({"label": label, "trial_id": i, "pos": pos, "cls": cls, "attempted": att, "canonical": can,
                    "contact": con, "n_licks": 1 if cls in ("hit", "incomplete") else 0,
                    "n_nocontact": 1 if cls == "incomplete" else 0, "cam4": 100.0 if att else np.nan,
                    "combined": 90.0 if att else np.nan, "has_cam1": True, "angle_first": ang})
    return out


def test_denominators_and_composite():
    rows = (_trials("PS92_0806", "far_contra", [("hit", True, 200.0, 0.0), ("hit", True, 400.0, 2.0)])
            + _trials("PS92_0901", "far_contra", [("hit", True, 500.0, 10.0), ("incomplete", True, np.nan, 12.0),
                                                  ("no_try", True, np.nan, np.nan),
                                                  ("no_try", False, np.nan, np.nan),     # quit tail: not counted
                                                  ("hit", False, 5000.0, 14.0)]))        # attempted, outside canonical
    W = ta.session_summary(pd.DataFrame(rows)).set_index("label")
    r = W.loc["PS92_0901"]
    assert r.attempt_rate_far_contra == pytest.approx(2 / 3)               # canonical: 2 attempted of 3
    assert r.n_attempted_far_contra == 3                                   # attempted counts outside canonical too
    assert r.ttc_ms_far_contra == pytest.approx((500 + 3500 + 3500) / 3)   # no contact and late contact -> 3.5 s
    assert r.completion_far_contra == pytest.approx(2 / 3)
    assert r.lateral_bias_far_contra == pytest.approx(12.0 - 1.0)          # vs August-pre mean angle 1.0
    assert W.loc["PS92_0806"].lateral_bias_far_contra == pytest.approx(0.0)
