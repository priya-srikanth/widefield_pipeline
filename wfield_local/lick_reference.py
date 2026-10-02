"""Tongue-direction DEVIATION from a reference of successful (spout-contact) licks at the same spout position.

Why (Priya 2026-10-02, after `tongue_spout_rays_*.png`): on cam4, "tongue angle - spout-tip angle" is ~±30-35° even
on CONTACT licks -- the spout-tip label is the tube's top front edge beside the mouth (60-100 px out, ±40°) while
the tongue-tip label is the tongue's distal end pressed past the spout (140-185 px, near vertical). In this front
view the two are not the same image location, so that difference is a per-position constant, not accuracy.
Measured instead: how far a lick's direction departs from the TYPICAL SUCCESSFUL lick at that position --
  * at the peak: angle - median peak angle of the reference contact licks (per position);
  * over the lick: phase angle - mean phase angle of the reference contact licks (per position x phase).
Contact licks of the reference sit near 0 by construction. Two references, both kept:
  * ``session`` -- the same session's contact licks: how misses differ from hits, immune to between-session
    camera / head shifts;
  * ``pre``     -- pooled PRE-STROKE contact licks of the same animal: a post-stroke drift away from the pre-stroke
    successful path (e.g. tongue running image-left of where it used to go), even when contact is normal. Valid
    because the spout positions are set by the Zabers; still carries any head-pose / camera difference between
    sessions (the spout frame is rebuilt per session, which absorbs translation but not rotation).
Angle = atan2(lr, ap) in the spout frame (+ = image-right = mouse LEFT on cam4), as `tongue_kinematics`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIP_ZONE_PX = 30.0      # phase points closer to the mouth than this are left out of reference and deviation


def peak_angle(per_lick: pd.DataFrame) -> pd.Series:
    """Unsmoothed tongue angle at each lick's peak, from the spout-frame components."""
    return pd.Series(np.degrees(np.arctan2(per_lick.lr_px, per_lick.ap_px)), index=per_lick.index)


def build_reference(per_licks: list[pd.DataFrame], phases: list[pd.DataFrame]) -> dict:
    """Reference from one or more sessions' tables (each per_lick with `contact`, each lick_phase): contact licks
    only. Returns {"peak": Series position -> median angle, "phase": Series (position, phase) -> mean angle,
    "n": Series position -> n contact licks}."""
    pl = pd.concat(per_licks, ignore_index=True)
    c = pl[pl.contact == True].assign(angle=lambda d: peak_angle(d))           # noqa: E712
    ph = pd.concat(phases, ignore_index=True)
    keys = c[["trial_id", "lick_idx", "position"]].drop_duplicates()
    # sessions are concatenated: (trial_id, lick_idx) is unique within a session, and trial ids do not repeat
    # across a reference set in practice; the merge on position too keeps a stray collision position-consistent
    cp = ph.merge(keys, on=["trial_id", "lick_idx", "position"])
    cp = cp[cp.protrusion_px >= LIP_ZONE_PX]
    return {"peak": c.groupby("position").angle.median(),
            "phase": cp.groupby(["position", "phase"]).angle_deg.mean(),
            "n": c.groupby("position").size()}


def add_deviation(per_lick: pd.DataFrame, lick_phase: pd.DataFrame, ref: dict, name: str):
    """Copies of the tables with `dev_<name>_deg`: per lick (peak angle - reference peak angle of its position) and
    per phase point (angle - reference angle at that position and phase; NaN inside the lip zone)."""
    pl = per_lick.copy()
    pl[f"dev_{name}_deg"] = peak_angle(pl) - pl.position.map(ref["peak"])
    ph = lick_phase.copy()
    r = ref["phase"].rename("ref_angle").reset_index()
    ph = ph.merge(r, on=["position", "phase"], how="left")
    ph[f"dev_{name}_deg"] = np.where(ph.protrusion_px >= LIP_ZONE_PX, ph.angle_deg - ph.ref_angle, np.nan)
    return pl, ph.drop(columns="ref_angle")
