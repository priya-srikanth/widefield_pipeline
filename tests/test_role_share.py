"""behavior_states.role_share_sessions / role_share_summary (jaw-only role share, Priya 11-20)."""
import numpy as np
import pandas as pd

from wfield_local import behavior_states as bs


def _table():
    rows = []
    # PS1 pre: 6 jaw-only (4 disengaged, 2 partial) + 2 near-mid no-response (dropped) + 5 contact engaged
    for _i in range(4):
        rows.append(("PS1_20260806", "PS1", "pre", "jaw_only", "far_L", "disengaged"))
    for _i in range(2):
        rows.append(("PS1_20260806", "PS1", "pre", "jaw_only", "far_R", "partial"))
    for _i in range(2):
        rows.append(("PS1_20260806", "PS1", "pre", "no_response", "close_center", "disengaged"))
    for _i in range(5):
        rows.append(("PS1_20260806", "PS1", "pre", "contact", "far_L", "engaged"))
    # PS2 pre: 5 jaw-only all engaged; one session with 2 jaw-only (below the minimum)
    for _i in range(5):
        rows.append(("PS2_20260806", "PS2", "pre", "jaw_only", "close_L", "engaged"))
    for _i in range(2):
        rows.append(("PS2_20260807", "PS2", "pre", "jaw_only", "close_L", "disengaged"))
    return pd.DataFrame(rows, columns=["label", "animal", "epoch", "cls", "position", "role"])


def test_sessions_drop_near_mid_and_count_classes():
    S = bs.role_share_sessions(_table())
    a = S[(S.label == "PS1_20260806") & (S.cls == "jaw_only")].iloc[0]
    assert a.n_trials == 6 and np.isclose(a.share_disengaged, 4 / 6) and np.isclose(a.share_partial, 2 / 6)
    assert S[(S.label == "PS1_20260806") & (S.cls == "no_response")].empty      # near mid dropped
    assert S[(S.label == "PS1_20260806") & (S.cls == "attempt")].iloc[0].share_engaged == 1.0


def test_summary_is_animal_weighted_with_min_trials():
    M = bs.role_share_summary(bs.role_share_sessions(_table()))
    r = M[(M.cls == "jaw_only") & (M.epoch == "pre")].iloc[0]
    assert r.n_animals == 2 and r.n_sessions == 2                                 # 2-trial session excluded
    assert np.isclose(r.disengaged, (4 / 6 + 0) / 2) and np.isclose(r.engaged, (0 + 1) / 2)
    assert np.isclose(r["disengaged_PS1"], 4 / 6)
    assert M[(M.cls == "jaw_only") & (M.epoch == "pooled")].iloc[0].n_sessions == 2
