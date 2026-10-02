"""lick_reference: deviation from the contact-lick reference is ~0 for the reference's own contact licks and
equals the angle offset for a lick that departs from it, at the peak and over the phase."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wfield_local import lick_reference as LR


def _tables(offset_nocontact=10.0):
    rows, ph = [], []
    for k, (pos, base) in enumerate([("far_L", 5.0)] * 4 + [("far_R", -20.0)] * 4):
        contact = k % 4 != 3
        a = base + (0.0 if contact else offset_nocontact)
        r = 150.0
        rows.append({"trial_id": k, "lick_idx": 0, "position": pos, "contact": contact,
                     "ap_px": r * np.cos(np.radians(a)), "lr_px": r * np.sin(np.radians(a))})
        for g in np.linspace(0, 1, 5):
            prot = 20.0 if g in (0.0, 1.0) else 120.0                       # ends inside the lip zone
            ph.append({"trial_id": k, "lick_idx": 0, "position": pos, "phase": g, "angle_deg": a + 3 * g,
                       "protrusion_px": prot})
    return pd.DataFrame(rows), pd.DataFrame(ph)


def test_deviation_zero_for_contact_and_offset_for_departures():
    pl, ph = _tables(offset_nocontact=10.0)
    ref = LR.build_reference([pl], [ph])
    assert ref["n"].to_dict() == {"far_L": 3, "far_R": 3}
    np.testing.assert_allclose(ref["peak"]["far_L"], 5.0)
    pl2, ph2 = LR.add_deviation(pl, ph, ref, "session")
    np.testing.assert_allclose(pl2.loc[pl2.contact, "dev_session_deg"], 0.0, atol=1e-9)
    np.testing.assert_allclose(pl2.loc[~pl2.contact, "dev_session_deg"], 10.0, atol=1e-9)
    inner = ph2[ph2.protrusion_px >= LR.LIP_ZONE_PX].merge(pl2[["trial_id", "contact"]], on="trial_id")
    np.testing.assert_allclose(inner.loc[inner.contact, "dev_session_deg"], 0.0, atol=1e-9)
    np.testing.assert_allclose(inner.loc[~inner.contact, "dev_session_deg"], 10.0, atol=1e-9)
    assert ph2.loc[ph2.protrusion_px < LR.LIP_ZONE_PX, "dev_session_deg"].isna().all()
