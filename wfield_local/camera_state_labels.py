"""Per-session CAMERA events for the camera-labeled state decoder (`locomotor_state.camera_three_way_segments`).

LOAD-ONLY. Reuses the 11-28 state census loaders (scripts/state_census_ccB1128.py, branch ccb/state-census-1128):

  fused licks   `session_licks.session_onsets_s(label, col="daq_on_s")` -- the adopted fused four-camera train
                (`session_licks.fused_run()`, env WIDEFIELD_FUSED_LICKS), cam1-only events dropped as everywhere
  jaw-only      the four-camera `trial_attempts.csv` rows with cls == "jaw_only" (3d_mouth rule), ``jaw_onset_ms`` from
                the cue, put on the DAQ clock with `orofacial_outputs.trial_ms_to_daq` through cam4's windows index
                (the same mapping the fused licks use)
  coverage      the union of cam4's per-trial windows on the DAQ clock (adjacent windows <= 8 ms apart merged)

All three are returned in DAQ SAMPLES at the behavior-events rate. A session without a fused lick table raises
`session_licks.SessionLicksUnavailable` -- the caller drops the session (never a silent fall-back to DAQ licks).
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache

import numpy as np

#: The four-camera trial_attempts table (jaw-only onsets). Override with env WIDEFIELD_STATE_JAW_ATTEMPTS.
ATTEMPTS_ENV = "WIDEFIELD_STATE_JAW_ATTEMPTS"
ATTEMPTS_GLOB = ("{root}/LP_analysis/*/imaging_encoding/qc/"
                 "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70/trial_attempts.csv")
DEFAULT_ROOT = "M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam"
#: Two cam4 trial windows closer than this are one covered stretch.
MERGE_GAP_S = 0.008


def attempts_path() -> str:
    p = os.environ.get(ATTEMPTS_ENV, "").strip()
    if p:
        return p
    root = os.environ.get("WIDEFIELD_REBUILD_ROOT", "").strip() or DEFAULT_ROOT
    f = sorted(glob.glob(ATTEMPTS_GLOB.format(root=root)))
    if len(f) != 1:
        raise FileNotFoundError(f"trial_attempts (jaw-only onsets): expected one match, got {f}; set {ATTEMPTS_ENV}")
    return f[0]


@lru_cache(maxsize=2)
def _jaw_table(path: str):
    import pandas as pd
    A = pd.read_csv(path, usecols=["label", "trial_id", "cls", "jaw_onset_ms"], low_memory=False)
    return A[(A.cls == "jaw_only") & np.isfinite(A.jaw_onset_ms)].copy()


def merge_windows(lo, hi, gap_s=MERGE_GAP_S):
    """Sorted, merged coverage intervals from per-trial windows (seconds)."""
    o = np.argsort(np.asarray(lo, float), kind="stable")
    mlo, mhi = [], []
    for a, b in zip(np.asarray(lo, float)[o], np.asarray(hi, float)[o]):
        if mlo and a <= mhi[-1] + gap_s:
            mhi[-1] = max(mhi[-1], b)
        else:
            mlo.append(a)
            mhi.append(b)
    return np.asarray(mlo, float), np.asarray(mhi, float)


def session_camera_events(label: str, fs: float, rv=None) -> dict:
    """``{"cam_licks", "jaw_onsets", "cov_lo", "cov_hi"}`` in DAQ samples, plus ``"n_fused"`` / ``"n_jaw"``."""
    from wfield_local import config
    from wfield_local import orofacial_outputs as oo
    from wfield_local import session_licks as SL
    from wfield_local.paths import PathResolver

    rv = rv or PathResolver()
    an, mmdd = label.split("_")
    d8 = config.date8(mmdd)
    fused = np.sort(np.asarray(SL.session_onsets_s(label, rv, col="daq_on_s"), float))
    fused = fused[np.isfinite(fused)]
    ix = SL._windows_ix(SL._cam4_dir(rv, an, d8))
    w = ix.groupby("trial_id").daq_s.agg(["min", "max"]).dropna()
    mlo, mhi = merge_windows(w["min"].to_numpy(float), w["max"].to_numpy(float))
    J = _jaw_table(attempts_path())
    J = J[J.label == label]
    jaw = (oo.trial_ms_to_daq(ix, J.trial_id.to_numpy(), J.jaw_onset_ms.to_numpy(float)) if len(J)
           else np.zeros(0))
    jaw = np.sort(jaw[np.isfinite(jaw)])
    to_s = lambda t: np.round(np.asarray(t, float) * fs).astype(np.int64)   # noqa: E731
    return {"cam_licks": to_s(fused), "jaw_onsets": to_s(jaw), "cov_lo": to_s(mlo), "cov_hi": to_s(mhi),
            "n_fused": int(len(fused)), "n_jaw": int(len(jaw))}
