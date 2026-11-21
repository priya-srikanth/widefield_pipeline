"""v37 spout-position rule (Priya 11-04, docs/status/SPOUT_POSITION_AUDIT_2026-11-04.md; executed 11-21 with D1 = (a)):
the session spout frame from the NO-LICK ENL medians, and the per-trial spout reference = that session median."""
import numpy as np
import pandas as pd

from wfield_local import orofacial_outputs as oo
from wfield_local import spout_frame as SF
from wfield_local import tongue_kinematics as tk

POS = {"close_L": (300.0, 200.0), "far_L": (340.0, 150.0), "close_center": (250.0, 205.0),
       "far_center": (250.0, 145.0), "close_R": (200.0, 200.0), "far_R": (160.0, 150.0)}


def _session(n_per=500, lick_shift=8.0):
    """Six positions, one trial each: ENL frames (strobe+300 .. cue, tongue absent) at the true spout; licking frames
    (after the cue, tongue present) with the spout pushed by ``lick_shift`` px -- the whole-window median moves, the
    ENL median does not."""
    rows, ix = [], []
    for k, (p, (x, y)) in enumerate(POS.items()):
        for i in range(n_per):
            t = -1000.0 + 4.0 * i                       # strobe at -1500: ENL = -1200 .. 0
            lick = t >= 0
            rows.append({"spout_x": x + (lick_shift if lick else 0.0), "spout_y": y, "spout_likelihood": 0.99,
                         "tongue_likelihood": 0.95 if lick else 0.05})
            ix.append({"trial_k": k, "trial_id": k + 1, "position": p, "t_ms": t, "strobe_ms": -1500.0})
    return pd.DataFrame(rows), pd.DataFrame(ix)


def test_v37_frame_uses_enl_medians_and_sets_the_trial_reference():
    flat, ix = _session()
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in ix.groupby("trial_k")]
    fr37, src37 = oo.session_spout_frame(flat, ix, spans, cleaning="v37")
    fr36, src36 = oo.session_spout_frame(flat, ix, spans, cleaning="v36")
    assert src37 == "enl_no_lick" and src36 == "whole_window"
    assert fr36.ref_medians is None and set(fr37.ref_medians) == set(POS)
    for p, (x, _y) in POS.items():
        assert abs(fr37.positions[p][0] - x) < 1e-9                 # parked spout, licks ignored
        assert fr36.positions[p][0] > x                             # whole window pulled by the pushed spout


def test_v37_falls_back_when_enl_cannot_build_the_frame():
    flat, ix = _session()
    flat.loc[ix.position.isin(["close_center", "far_center"]) & (ix.t_ms < 0), "spout_likelihood"] = 0.5
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in ix.groupby("trial_k")]
    fr, src = oo.session_spout_frame(flat, ix, spans, cleaning="v37")
    assert src == "whole_window_fallback" and "close_center" not in fr.ref_medians


def test_trial_spout_reference_takes_the_session_median():
    f = tk.SpoutFrame((250.0, 230.0), (0.0, 1.0))
    trials = [tk.Trial(trial_id=1, cue_frame=10.0, position="far_L"), tk.Trial(trial_id=2, cue_frame=60.0, position="far_R")]
    n = 120
    sx, sy, sl = np.full(n, 360.0), np.full(n, 150.0), np.full(n, 0.99)      # a pushed / mis-tracked spout in-trial
    p = tk.params(None)
    old = tk.trial_spout_reference(trials, (sx, sy, sl), 250.0, f, p=p)
    new = tk.trial_spout_reference(trials, (sx, sy, sl), 250.0, f, p=p, medians={"far_L": (340.0, 150.0, 999)})
    assert old.spout_x_img.iloc[0] == 360.0
    assert new.spout_x_img.iloc[0] == 340.0 and new.spout_n_frames.iloc[0] == 999
    assert list(new.spout_ref_source) == ["enl_session", "trial"]          # far_R has no ENL median -> per-trial
    assert new.spout_x_img.iloc[1] == 360.0


def test_cleaning_version_is_v37():
    assert oo.CLEANING_VERSION == "v37" and oo.run_name("LP_r4_heatmap_20261013") == "lp_r4_heatmap_20261013__clean_v37"
    assert SF.SpoutFrame.__dataclass_fields__["ref_medians"].default is None
