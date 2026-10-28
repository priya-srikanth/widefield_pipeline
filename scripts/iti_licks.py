"""SELF-INITIATED (ITI) 3-D LICKS: pre-lick ramp timing + target-free execution coding (Priya 10-25 night: "let's also do
the inter-trial 3d licks, pre-lick ramp timing"; proposed 10-25 as analyses 3 and 5).

WHY. Every trial lick follows a cue and has a target, so the trial readouts mix the plan / target with the movement and
with the cue response. ITI licks (whole-session 3-D table `pose3d/kinematics_all/licks.csv.gz`, context iti) have NO
trial, NO cue and NO reward schedule; the spout is still in place (ITI licks carry its position), so they are
target-free in the TASK sense but can still be aimed at the spout present. Two questions:
 (A) RAMP TIMING: how early does ALM / tjM1 activity rise before a self-initiated lick, and does that change after the
     stroke? (Bout-start ITI licks: no lick in the preceding 2 s -- 1 s as a secondary set, kind iti_bout_gap1 --, no cue /
     spout move within [-2, +0.75] s; ONSET-aligned, baseline -1.5..-1.0 s.) Trial first
     licks are a secondary set (a cue precedes them by the reaction time; read within RT terciles).
 (B) EXECUTION CODING WITHOUT A TRIAL: cross-validated decoding of each ITI lick's 3-D lateral direction (side_ml, mm)
     from area activity, per lag around the lick peak (-0.5 .. +0.5 s), vs n-matched in-trial (post-cue) licks of the
     same session; and the WITHIN-SPOUT direction (side_ml minus its spout's session mean: pure execution variation,
     no spout identity). Plus a FROZEN version: decoder trained on each animal's August pre ITI licks (pre sessions
     leave-one-session-out), applied to every post session. If ITI execution coding is intact while the trial
     far-contra code is lost, plan / target and execution dissociate.
ROIs (Priya 10-25: "ALM and tjM1 ... are the anterolateral 1/3 to 1/2 of MOs and MOp"): per hemisphere, the ROI_FRAC (v1 40 %; v2 ALM 40 % / tjM1 50 %) most
anterolateral pixels of MOs (ALM) / MOp (tjM1) on the eroded stat mask, ranked by z(anterior) + z(lateral distance from
the midline) within the area (Allen grid shared by all sessions; figure lick_iti_rois.png to check placement). _right =
contralesional (image right), _left = ipsilesional.
Signals: Allen-area / ROI means of the adopted hemo-corrected SVD (`combined_encoding.load_imaging`, the encoder's
targets), slow drift removed (the encoder's slow basis), dF/F x1e3. Lick times on the DAQ clock from the session's
licks_all.csv via lick_uid (NOT the kinematics table's daq_peak_s, which is rounded to 4 significant digits = up to
0.5 s off in a 1 h session): ramp PETHs are aligned to lick ONSET, direction decoding to the 3-D PEAK; the clock match to
the behaviour licks is checked per session and recorded.
Calcium indicator lag delays every rise; the pre -> post CHANGE is the readout, not the absolute lead.
Usage: python scripts/iti_licks.py [--jobs 3] [--labels ...]   (extract + summarise)
       python scripts/iti_licks.py --plot-only [--key ...]       (redraw from figdata/*.pkl only)
       python scripts/iti_licks.py --summarise                   (from saved sessions)
Output: <E>/iti_licks/ (sessions/<label>.npz, tables, figures)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

OUT = E / "iti_licks"
# ROI VERSION (Priya 10-27, after lick_roi_check / lick_roi_foci and the MOp decile profile: the pre-map hot spot in MOp
# runs past 40 %; literature check vs Guo 2014 / Mayrhofer 2019 / Esmaeili 2021 + lick_roi_candidates.png; Priya chose
# ALM unchanged, tjM1 50 % -- ALM 45 % starts to take in whisker M2): v1 = ALM / tjM1 = anterolateral 40 % (session
# files in sessions/); v2 = ALM 40 % of MOs, tjM1 50 % of MOp (sessions_roi2/). WFIELD_ROI_VERSION selects; the v1 files
# are kept unchanged.
ROI_VERSION = os.environ.get("WFIELD_ROI_VERSION", "8")      # default = the FINAL ROIs (v1 = 40/40, kept)
# FINAL (Priya 10-27): v2 = ALM 40 % / tjM1 50 % (45 / 50 considered; ALM 45 % takes in part of whisker M2).
# v3 (Priya 10-27: "can we expand the rim anteriorly? posteriorly it matters more" / "shrink the excluded area"): same
# fractions as v2, but ROI / AREA AVERAGES use a THINNER rim anteriorly -- pixels anterior of RIM_AP_MIN keep everything
# except the outer RIM_ANT_PX of the brain mask (instead of the 16-px stat-mask rim); posteriorly unchanged. Measured
# before the change (6 sessions): the anterior MOs rim has half the lick response but no worse single-lick SNR (1.29 vs
# 1.13). Per-pixel MAP statistics keep beta_maps.stat_mask() (its rim artefacts are shared across animals).
RIM_AP_MIN, RIM_ANT_PX = 1.0, 4
_ROI = {"1": ({"ALM": 0.4, "tjM1": 0.4}, "sessions", "session_sd.csv", "session_sd_split.csv", ""),
        "2": ({"ALM": 0.4, "tjM1": 0.5}, "sessions_roi2", "session_sd_roi2_tj50.csv", "session_sd_split_roi2_tj50.csv",
              ""),
        "3": ({"ALM": 0.4, "tjM1": 0.5}, "sessions_roi3", "session_sd_roi3_rimant.csv", "session_sd_split_roi3_rimant.csv",
              "ant"),
        # v4 (Priya 10-27: "give the orofacial S1 strip"; "I don't want the front edge slivers - only expand laterally"):
        # v3 is ABANDONED (front-edge slivers). v4 = the thin rim ONLY at the LATERAL edge: a pixel of the 16-px rim is
        # kept when it is >= RIM_ANT_PX inside the brain mask AND the mask ends within 16 px LATERALLY (moving away from
        # the midline along its row) -- i.e. it is rim because of the lateral edge, not the front / back edge -- AND it
        # lies anterior of RIM_LAT_AP_MIN (the lateral MOp / tjM1 edge and the orofacial S1 strip; posterior rim kept out)
        "4": ({"ALM": 0.4, "tjM1": 0.5}, "sessions_roi4", "session_sd_roi4_rimlat.csv",
              "session_sd_split_roi4_rimlat.csv", "lat"),
        # v5 (Priya 10-27 sketch on lick_rim_overlay: ALM reaches the FRONT edge): as v4 + the anterior thin-rim MOs pixels
        # that continue the ALM wedge (same anterolateral rank threshold as ALM) are added to ALM (and MOs); no other
        # front-edge pixels (no PL / FRP / medial MOs slivers)
        "5": ({"ALM": 0.4, "tjM1": 0.5}, "sessions_roi5", "session_sd_roi5_rimlat_almfront.csv",
              "session_sd_split_roi5_rimlat_almfront.csv", "lat+almfront"),
        # v6 (Priya 10-27: "keep the lateral extent but I'm worried about edge artefact anteriorly"): as v4, but a lateral-
        # edge pixel is NOT added when it is also within 16 px of the FRONT edge (the mask ends within 16 px going
        # anterior) -- removes v4's detached anterolateral ALM sliver; nothing added at the front edge (v5 rejected)
        "6": ({"ALM": 0.4, "tjM1": 0.5}, "sessions_roi6", "session_sd_roi6_rimlat_nofront.csv",
              "session_sd_split_roi6_rimlat_nofront.csv", "lat-nofront"),
        # v7 (Priya 10-27: "I don't want to be right up on FRP, medium rim is better"; "v6 doesn't have the lateral
        # extent in tjM1 and SSp-m"): lateral thin rim as v4 (full lateral extent) + ALM continued toward the front edge
        # with a MEDIUM rim:
        # front-edge MOs pixels >= RIM_FRONT_PX inside the brain mask that pass the ALM anterolateral rank are added to
        # ALM (and MOs); ALM / tjM1 kept as one connected region each
        "7": ({"ALM": 0.4, "tjM1": 0.5}, "sessions_roi7", "session_sd_roi7_rimlat_almfront8.csv",
              "session_sd_split_roi7_rimlat_almfront8.csv", "lat-almfront8"),
        # v8 (Priya 10-27: "v7 just needs to extend the front horizontal line until it intersects the lateral edge - no
        # zig zag"): as v7, but ALM's front boundary is ONE horizontal line -- the most anterior row of v7's medium-rim
        # front -- carried laterally to the thin (4-px) lateral rim; ALM = MOs pixels at or behind that row that pass the
        # ALM rank, within the 4-px-eroded brain mask, >= FRP_GAP_PX from FRP; one connected region
        "8": ({"ALM": 0.4, "tjM1": 0.5}, "sessions_roi8fr",   # sessions_roi8/ = env-less first run (superseded)
               "session_sd_roi8_rimlat_almline.csv",
              "session_sd_split_roi8_rimlat_almline.csv", "lat-almline")}
RIM_FRONT_PX = 8
FRP_GAP_PX = 10           # 0.19 mm
RIM_LAT_AP_MIN = -0.5
ROI_FRAC, SESS_NAME, SD_NAME, SPLIT_NAME, ROI_THIN_ANT_RIM = _ROI[ROI_VERSION]   # SD files live in zscore_check/
LAGS_P = np.round(np.arange(-1.5, 0.7501, 0.05), 3)          # PETH lags (s from lick peak)
LAGS_C = np.round(np.arange(-1.0, 2.5001, 0.05), 3)          # cue-aligned lags (trial first licks)
LAGS_D = np.round(np.arange(-1.0, 1.0001, 0.05), 3)          # decoding lags (-1..+1 s: binned-decoder sweep, 10-26)
LAGS_D_SHOW = 0.5                                            # per-lag curves drawn / within-session decoded to +-0.5 s
BASE = (-1.5, -1.0)
BOUT_GAP = 1.0                     # extraction: ITI onsets with no lick for >= 1 s (each lick's gap is saved)
GAP_MAIN = 2.0                     # summaries: >= 2 s, so the -1.5..-1.0 s baseline starts >= 0.5 s after the last lick
#                                    (with 1 s the acute baseline still carried the previous licks' decay)
CUE_EXCL = (-2.0, 0.75)
N_TRIAL_MAX = 1500
FR_KEEP = ["fr_lick", "fr_right", "fr_left", "fr_close", "fr_far"]   # FaceRhythm master_v1 mode-A factors (no "other")
LAGS_FRC = np.round(np.arange(0.0, 1.5001, 0.05), 3)       # cue window (frozen Musall cue window 0..1.5 s)
LAGS_FRL = np.round(np.arange(-0.1, 1.0001, 0.05), 3)      # first-lick window (-0.1..1.0 s)
RAMP_ROIS = ["ALM_right", "ALM_left", "tjM1_right", "tjM1_left", "MOs_right", "MOs_left", "MOp_right", "MOp_left",
             "SSp-m_right", "SSp-m_left", "cortex"]
EP4 = ["pre", "acute", "subacute", "chronic"]


# Figure wording shared by every decoder figure (Priya 10-26: say which decoder frame and what is measured).
YR = "r (decoded vs actual lateral tongue-tip position)"
FROZEN_TAG = "FROZEN pre-stroke decoder (fit once on the animal's August pre licks, never refit)"
REFIT_TAG = "PER-SESSION decoder (refit within each session, cross-validated)"
TGT_LAB = {"direction": "lateral tongue-tip position (3-D, mm)",
           "relative": "lateral position relative to the reference spout"}

def lick_table() -> pd.DataFrame:
    from wfield_local.paths import PathResolver
    p = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
    return pd.read_csv(p, usecols=["session", "context", "lick_uid", "side_ml", "out_ap", "down_dv", "position",
                                   "trial_id", "reach_mm", "speed_peak"])


def roi_masks(parea, pstat, names, flat, shape, frac: dict | None = None) -> dict:
    """{name: boolean over brain pixels}: every Allen area (stat mask) + ALM / tjM1 per hemisphere (module docstring).
    ROI_THIN_ANT_RIM: "lat" (v4) = the 16-px rim is kept at the LATERAL edge (anterior of RIM_LAT_AP_MIN) except the
    outer RIM_ANT_PX; "ant" (v3, abandoned) = the same anterior of RIM_AP_MIN."""
    H, W = (int(x) for x in shape)
    r, c = flat // W, flat % W
    if ROI_THIN_ANT_RIM:
        from scipy import ndimage

        from wfield_local import beta_maps
        from wfield_local.allen_register import BREGMA_OFFSET, RESOLUTION
        bmask = beta_maps.brain_mask()
        thin = ndimage.binary_erosion(bmask, iterations=RIM_ANT_PX).ravel()[flat]
        ap_mm = (BREGMA_OFFSET[1] - r) * RESOLUTION
        if ROI_THIN_ANT_RIM.startswith("lat"):
            step = beta_maps.STAT_ERODE_PX
            out_c = np.where(c >= BREGMA_OFFSET[0], c + step, c - step)          # 16 px further from the midline
            ok_c = (out_c >= 0) & (out_c < W)
            lat_edge = ~ok_c | ~bmask[r, np.clip(out_c, 0, W - 1)]
            add = lat_edge & (ap_mm > RIM_LAT_AP_MIN)
            if ROI_THIN_ANT_RIM == "lat-nofront":
                up_r = r - step                                                   # 16 px anterior (image up)
                front_edge = (up_r < 0) | ~bmask[np.clip(up_r, 0, H - 1), c]
                add &= ~front_edge
        else:                                                                     # "ant" (v3, abandoned)
            add = ap_mm > RIM_AP_MIN
        front = thin & ~pstat & ~add & (ap_mm > RIM_AP_MIN) & (parea != 0)   # anterior thin-rim candidates (v5)
        if ROI_THIN_ANT_RIM == "lat-almline":                                    # v8: pool for the line rule
            frp_ = np.zeros(H * W, bool)
            frp_[flat[np.isin(parea, [k for k, v in names.items() if str(v).startswith("FRP")])]] = True
            gap_ = ndimage.distance_transform_edt(~frp_.reshape(H, W)).ravel()[flat] >= FRP_GAP_PX
            pool4 = thin & gap_ & (parea != 0)
            front8 = front & ndimage.binary_erosion(bmask, iterations=RIM_FRONT_PX).ravel()[flat] & gap_
        if ROI_THIN_ANT_RIM == "lat-almfront8":                                  # v7: medium front rim
            front &= ndimage.binary_erosion(bmask, iterations=RIM_FRONT_PX).ravel()[flat]
            # ... and >= FRP_GAP_PX from the frontal pole (Priya: "I don't want to be right up on FRP")
            frp = np.zeros(H * W, bool)
            frp[flat[np.isin(parea, [k for k, v in names.items() if str(v).startswith("FRP")])]] = True
            front &= ndimage.distance_transform_edt(~frp.reshape(H, W)).ravel()[flat] >= FRP_GAP_PX
        pstat = pstat | (thin & add & (parea != 0))   # labelled cortex only
    m = {names.get(int(a), str(a)): (parea == a) & pstat for a in sorted(np.unique(parea[pstat])) if a != 0}
    left = np.zeros(len(flat), bool)
    right = np.zeros(len(flat), bool)
    for n, mk in m.items():
        left |= mk & n.endswith("_left")
        right |= mk & n.endswith("_right")
    mid = (np.median(c[left]) + np.median(c[right])) / 2.0 if left.any() and right.any() else W / 2.0
    for src, dst in (("MOs", "ALM"), ("MOp", "tjM1")):
        for h in ("left", "right"):
            mk = m.get(f"{src}_{h}")
            if mk is None or not mk.any():
                continue
            ant = -r[mk].astype(float)                      # anterior = top of the image
            lat = np.abs(c[mk] - mid).astype(float)
            u = (ant - ant.mean()) / (ant.std() or 1) + (lat - lat.mean()) / (lat.std() or 1)
            sel = np.zeros(len(flat), bool)
            thr = np.quantile(u, 1 - (frac or ROI_FRAC)[dst])
            sel[np.flatnonzero(mk)[u >= thr]] = True
            if ROI_THIN_ANT_RIM in ("lat+almfront", "lat-almfront8") and dst == "ALM":
                # front-edge MOs pixels of this hemisphere that continue the wedge: same rank, same z-scoring
                cand = front & (parea == next(k for k, v in names.items() if v == f"MOs_{h}"))
                ua = (-r[cand].astype(float) - ant.mean()) / (ant.std() or 1) \
                    + (np.abs(c[cand] - mid).astype(float) - lat.mean()) / (lat.std() or 1)
                idx = np.flatnonzero(cand)[ua >= thr]
                sel[idx] = True
                m[f"MOs_{h}"] = m[f"MOs_{h}"].copy()
                m[f"MOs_{h}"][idx] = True
            if ROI_THIN_ANT_RIM == "lat-almline" and dst == "ALM":
                code = next(k for k, v in names.items() if v == f"MOs_{h}")

                def _u(ix, ant=ant, lat=lat):
                    return (-r[ix].astype(float) - ant.mean()) / (ant.std() or 1) \
                        + (np.abs(c[ix] - mid).astype(float) - lat.mean()) / (lat.std() or 1)
                c8 = front8 & (parea == code)
                c8 = np.flatnonzero(c8)[_u(c8) >= thr]
                line = int(r[c8].min()) if len(c8) else int(r[sel].min())          # v7's most anterior ALM row
                pool = pool4 & (parea == code) & (r >= line)
                idx = np.flatnonzero(pool)[_u(pool) >= thr]
                sel[idx] = True
                sel &= r >= line
                m[f"MOs_{h}"] = m[f"MOs_{h}"].copy()
                m[f"MOs_{h}"][idx] = True
            if ROI_THIN_ANT_RIM in ("lat-nofront", "lat-almfront8", "lat-almline"):   # v6-v8: one connected ROI
                from scipy import ndimage
                g = np.zeros(H * W, bool)
                g[flat[sel]] = True
                lab_, n_ = ndimage.label(g.reshape(H, W))
                if n_ > 1:
                    keep = np.argmax(np.bincount(lab_.ravel())[1:]) + 1
                    sel &= (lab_.ravel()[flat] == keep)
            m[f"{dst}_{h}"] = sel
    m["cortex"] = pstat.copy()
    if ROI_THIN_ANT_RIM in ("lat+almfront", "lat-almfront8", "lat-almline"):
        m["cortex"] |= m.get("ALM_left", False) | m.get("ALM_right", False)
    return m


def _behaviour(label: str):
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    try:
        return ce.load_behaviour(an, date, PathResolver())
    except FileNotFoundError as ex:                      # no cam1 licks: cam4 timing (as frozen_encoder.session_design)
        if "cam1_licks" not in str(ex) or not os.environ.get(ce.LICKS_ENV):
            raise
        saved = os.environ.pop(ce.LICKS_ENV)
        try:
            return ce.load_behaviour(an, date, PathResolver())
        finally:
            os.environ[ce.LICKS_ENV] = saved


def _sample(R: np.ndarray, ft: np.ndarray, t: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """(n events, n lags, n columns) of R interpolated at t + lags."""
    tt = (t[:, None] + lags[None, :]).ravel()
    out = np.empty((len(tt), R.shape[1]), np.float32)
    for j in range(R.shape[1]):
        out[:, j] = np.interp(tt, ft, R[:, j])
    return out.reshape(len(t), len(lags), R.shape[1])


def extract(item) -> dict:
    """One session -> <out>/sessions/<label>.npz (module-level for spawn). item = (label, lick table path or None)."""
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    label, out = item
    f = Path(out) / SESS_NAME / f"{label}.npz"
    if f.exists():
        return {"label": label, "cached": True}
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    B = _behaviour(label)
    T = B.trials
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    K = lick_table()
    K = K[K.session == f"{an}_{date}"].copy()
    # FULL-PRECISION TIMES: the kinematics_all table stores daq_peak_s to 4 significant digits (0.1 s resolution after
    # 100 s, 1 s after 1000 s); lick_uid = the row of the session's licks_all.csv, which has the exact DAQ onset / peak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["context", "segment", "daq_on_s", "daq_peak_s"])
    uid = K.lick_uid.to_numpy(int)
    if not (A.context.to_numpy()[uid] == K.context.to_numpy()).all():
        raise RuntimeError(f"{label}: lick_uid join to licks_all.csv does not match on context")
    K["daq_peak_s"] = A.daq_peak_s.to_numpy(float)[uid]
    K["daq_on_s"] = A.daq_on_s.to_numpy(float)[uid]
    # ITI licks carry the label of the trial WINDOW they fall in: "tail" = after that trial's response window (label =
    # the trial just finished), "lead" = before the next trial's spout arrives (label = the UPCOMING trial). The spout
    # is retracted during the ITI (Priya 10-26), so the label is not a spout in place: use the PREVIOUS trial's spout
    # (the last spout the animal licked at) for every ITI lick -- trial_id for tail, trial_id - 1 for lead
    seg = A.segment.astype(str).to_numpy()[uid]
    pos_of_trial = dict(zip(T.trial_id.to_numpy(), T.pos.astype(str).to_numpy()))
    prev_tid = np.where(seg == "lead", K.trial_id.to_numpy() - 1, K.trial_id.to_numpy())
    K["prev_pos"] = [pos_of_trial.get(t_, "") for t_ in prev_tid]
    K = K.sort_values("daq_peak_s").reset_index(drop=True)
    # every lick ONSET (all cam4 licks of the windows + the behaviour licks) for the bout rule
    t_all = np.sort(np.concatenate([A.daq_on_s.to_numpy(float), np.asarray(B.lick_on_s, float)]))
    t_all = t_all[np.isfinite(t_all)]
    pc = K[K.context == "post_cue"].daq_peak_s.to_numpy(float)
    bp = np.sort(np.asarray(B.lick_peak_s, float)[np.isfinite(B.lick_peak_s)])
    if len(pc) and len(bp):
        j = np.clip(np.searchsorted(bp, pc), 1, len(bp) - 1)
        clock = float(np.median(np.minimum(np.abs(pc - bp[j - 1]), np.abs(pc - bp[j]))))
    else:
        clock = np.nan
    ev = np.sort(np.concatenate([T.cue_s.to_numpy(float), T.strobe_s.to_numpy(float)]))
    ev = ev[np.isfinite(ev)]
    lo, hi = ft[0] - LAGS_P[0], ft[-1] - LAGS_P[-1]

    def near_event(t, a, b):
        j = np.searchsorted(ev, t + a)
        return (j < len(ev)) & (ev[np.minimum(j, len(ev) - 1)] <= t + b)
    # (A) bout-start ITI licks
    iti = K[(K.context == "iti") & np.isfinite(K.side_ml) & np.isfinite(K.daq_on_s)].copy()
    ti = iti.daq_peak_s.to_numpy(float)
    to = iti.daq_on_s.to_numpy(float)                    # ramp PETHs are ONSET-aligned
    k = np.searchsorted(t_all, to - 0.05)                # onsets within 50 ms = the same lick (cam4 vs behaviour copy)
    gap = to - np.where(k > 0, t_all[np.maximum(k - 1, 0)], -np.inf)
    bout = (gap >= BOUT_GAP) & ~np.array([near_event(t, *CUE_EXCL) for t in to], bool) & (to > lo) & (to < hi)
    # licks FOLLOWING the bout start within the post-onset window (0.05..0.75 s): a longer bout sums more lick
    # responses, so the post-onset amplitude is also read on ISOLATED licks (none following)
    nfol = np.searchsorted(t_all, to + 0.75, side="right") - np.searchsorted(t_all, to + 0.05)
    # trial first licks (engaged trials), ONSET-aligned (behaviour lick onsets)
    bo = np.sort(np.asarray(B.lick_on_s, float)[np.isfinite(B.lick_on_s)])
    Eg = T[T.engaged & np.isfinite(T.cue_s)]
    fl, rt, fpos = [], [], []
    for c, s, p in zip(Eg.cue_s.to_numpy(float), Eg.stop_s.to_numpy(float), Eg.pos.to_numpy()):
        s = s if np.isfinite(s) else c + 3.5
        j = np.searchsorted(bo, c)
        if j < len(bo) and bo[j] <= s and lo < bo[j] < hi and c + LAGS_C[0] > ft[0] and c + LAGS_C[-1] < ft[-1]:
            fl.append(bo[j]); rt.append(bo[j] - c); fpos.append(p)
    fl, rt = np.asarray(fl), np.asarray(rt)
    # (B) decoding sets: all ITI licks with 3-D direction; post-cue licks (random subset, seeded)
    dlo, dhi = ft[0] - LAGS_D[0], ft[-1] - LAGS_D[-1]
    di = iti[(ti > dlo) & (ti < dhi)]
    tr = K[(K.context == "post_cue") & np.isfinite(K.side_ml) & (K.daq_peak_s > dlo) & (K.daq_peak_s < dhi)]
    if len(tr) > N_TRIAL_MAX:
        tr = tr.iloc[np.sort(np.random.default_rng(0).choice(len(tr), N_TRIAL_MAX, replace=False))]
    # CONTROLS (Priya 10-26): FaceRhythm factors around each bout start (pre-lick face movement) and per trial in the
    # cue / first-lick windows (trial-to-trial variance of the lick factor, Fig 1.16); 3-D reach / speed per bout start
    from wfield_local import frozen_encoder as fe
    from wfield_local.paths import PathResolver as _PR
    try:
        frc = fe._fr_columns(an, date, _PR(), ft)
        FRM = np.column_stack([frc[v][:, 0] for v in FR_KEEP])
    except Exception as ex:                              # no FaceRhythm for this session: NaN placeholders
        print(f"  {label}: no FaceRhythm ({ex})", flush=True)
        FRM = np.full((len(ft), len(FR_KEEP)), np.nan)
    FRM = np.where(np.isfinite(FRM), FRM, np.nan)

    def fr_at(t, lags):
        tt = (np.asarray(t, float)[:, None] + lags[None, :]).ravel()
        o = np.empty((len(tt), FRM.shape[1]), np.float32)
        for j_ in range(FRM.shape[1]):
            ok_ = np.isfinite(FRM[:, j_])
            o[:, j_] = np.interp(tt, ft[ok_], FRM[ok_, j_]) if ok_.sum() > 10 else np.nan
        return o.reshape(len(t), len(lags), FRM.shape[1])
    f16 = np.float16
    np.savez_compressed(
        f, cols=np.array(cols), epoch=epochs.epoch_of(label), animal=an, clock_med_s=clock,
        bout_peth=_sample(R, ft, to[bout], LAGS_P).astype(f16), bout_side=iti.side_ml.to_numpy(float)[bout], bout_gap=gap[bout],
        bout_pos=iti.position.to_numpy(dtype=str)[bout], bout_nfol=nfol[bout], bout_t=to[bout],
        bout_reach=iti.reach_mm.to_numpy(float)[bout], bout_speed=iti.speed_peak.to_numpy(float)[bout],
        bout_fr=fr_at(to[bout], LAGS_P), fr_names=np.array(FR_KEEP),
        first_fr_cue=fr_at(fl - rt, LAGS_FRC) if len(fl) else np.zeros((0, len(LAGS_FRC), len(FR_KEEP)), np.float32),
        first_fr_lick=fr_at(fl, LAGS_FRL) if len(fl) else np.zeros((0, len(LAGS_FRL), len(FR_KEEP)), np.float32),
        first_peth=_sample(R, ft, fl, LAGS_P).astype(f16) if len(fl) else np.zeros((0, len(LAGS_P), len(cols)), f16),
        first_cue=_sample(R, ft, fl - rt, LAGS_C).astype(f16) if len(fl) else np.zeros((0, len(LAGS_C), len(cols)), f16),
        first_rt=rt, first_pos=np.asarray(fpos, str),
        iti_X=_sample(R, ft, di.daq_peak_s.to_numpy(float), LAGS_D).astype(f16), iti_side=di.side_ml.to_numpy(float),
        iti_pos=di.prev_pos.to_numpy(dtype=str), iti_t=di.daq_peak_s.to_numpy(float),
        tr_X=_sample(R, ft, tr.daq_peak_s.to_numpy(float), LAGS_D).astype(f16), tr_side=tr.side_ml.to_numpy(float),
        tr_pos=tr.position.to_numpy(dtype=str), tr_t=tr.daq_peak_s.to_numpy(float))
    return {"label": label, "n_bout": int(bout.sum()), "n_iti": len(di), "n_first": len(fl), "clock_med_s": clock}


# ------------------------------------------------------------------------------------------------ summaries


def _load(label: str, sess_dir: Path | None = None) -> dict:
    z = np.load((sess_dir or OUT / SESS_NAME) / f"{label}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def _bout(d: dict, gap: float, isolated: bool = False) -> np.ndarray:
    """Bout-start ITI PETHs whose preceding no-lick gap is >= gap s; isolated = no lick FOLLOWING within 0.75 s."""
    m = d["bout_gap"] >= gap
    if isolated:
        m &= d["bout_nfol"] == 0
    return d["bout_peth"][m]


def ramp_metrics(P: np.ndarray, lags: np.ndarray) -> tuple[float, float, float]:
    """(amplitude at the lick, t50, t20) of a baseline-subtracted PETH: t_x = the first lag after the LAST time (in
    [-1.0, 0]) the PETH was below x * amplitude (sign-aware); NaN if the amplitude is not positive-going."""
    A = float(P[(lags >= -0.1) & (lags <= 0.1)].mean())
    out = [A]
    for x in (0.5, 0.2):
        w = (lags >= -1.0) & (lags <= 0.0)
        v, L = P[w] * np.sign(A), lags[w]
        below = np.flatnonzero(v < x * abs(A))
        out.append(float(L[below[-1] + 1]) if len(below) and below[-1] + 1 < len(L) else (float(L[0]) if not len(below)
                                                                                                else np.nan))
    return out[0], out[1], out[2]


def _cv_r(X: np.ndarray, y: np.ndarray, t: np.ndarray, k: int = 5) -> float:
    """Ridge (alpha by inner CV), 5 contiguous time-block folds; r(prediction, y)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    o = np.argsort(t)
    X, y = X[o], y[o]
    folds = np.array_split(np.arange(len(y)), k)
    pred = np.full(len(y), np.nan)
    for te in folds:
        trn = np.setdiff1d(np.arange(len(y)), te)
        sc = StandardScaler().fit(X[trn])
        m = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(sc.transform(X[trn]), y[trn])
        pred[te] = m.predict(sc.transform(X[te]))
    return float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else np.nan


def _within(side: np.ndarray, pos: np.ndarray) -> np.ndarray:
    w = side.astype(float).copy()
    for q in np.unique(pos):
        m = pos == q
        w[m] -= w[m].mean()
    return w


def decode_session(d: dict, rng) -> list[dict]:
    rows = []
    ni, nt = len(d["iti_side"]), len(d["tr_side"])
    if ni < 50:
        return rows
    sub = np.sort(rng.choice(nt, ni, replace=False)) if nt >= ni else np.arange(nt)
    sets = {"iti": (d["iti_X"].astype(float), d["iti_side"], d["iti_pos"], d["iti_t"]),
            "trial": (d["tr_X"].astype(float)[sub], d["tr_side"][sub], d["tr_pos"][sub], d["tr_t"][sub])}
    feat = [i for i, c in enumerate(d["cols"]) if c not in ("cortex",) and not c.startswith(("ALM", "tjM1"))]
    for sname, (X, side, pos, t) in sets.items():
        for target, y in (("direction", side), ("relative", _within(side, pos))):
            for li, lag in enumerate(LAGS_D):
                if abs(lag) > LAGS_D_SHOW + 1e-9:
                    continue
                rows.append({"set": sname, "target": target, "lag": float(lag), "n": len(y),
                             "r": _cv_r(X[:, li, feat], y, t)})
    return rows


def frozen_decode(D: dict) -> pd.DataFrame:
    """FROZEN ITI direction decoder per animal: ridge on the animal's August pre ITI licks (features = area activity
    averaged over lags 0..0.2 s), pre sessions scored leave-one-session-out, post sessions with the full pre fit."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    w = (LAGS_D >= 0.0) & (LAGS_D <= 0.2)
    rows = []
    for a in AN:
        S = {x: d for x, d in D.items() if str(d["animal"]) == a and len(d["iti_side"]) >= 20}
        pre = [x for x, d in S.items() if str(d["epoch"]) == "pre"]
        if len(pre) < 3:
            continue
        feat = [i for i, c in enumerate(next(iter(S.values()))["cols"]) if c != "cortex" and not c.startswith(("ALM", "tjM1"))]

        def xy(x, tg, S=S, feat=feat):
            d = S[x]
            y = d["iti_side"] if tg == "direction" else _within(d["iti_side"], d["iti_pos"])   # iti_pos = PREVIOUS trial spout
            return d["iti_X"].astype(float)[:, w][:, :, feat].mean(1), y
        for tg in ("direction", "relative"):
            for x in S:
                trn = [p_ for p_ in pre if p_ != x]
                Xt = np.vstack([xy(p_, tg)[0] for p_ in trn])
                yt = np.concatenate([xy(p_, tg)[1] for p_ in trn])
                sc = StandardScaler().fit(Xt)
                m = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(sc.transform(Xt), yt)
                X, y = xy(x, tg)
                pr = m.predict(sc.transform(X))
                rows.append({"label": x, "animal": a, "epoch": str(S[x]["epoch"]), "target": tg, "n": len(y),
                             "r": float(np.corrcoef(pr, y)[0, 1]) if np.std(pr) > 0 else np.nan})
    return pd.DataFrame(rows)


def ramp_figure(plt, D: dict, RR: pd.DataFrame, kind: str, iso: bool, colors: dict, path, what: str) -> None:
    """Pooled PETH per epoch (compute -> figdata <stem>.pkl -> draw); see `ramp_compute`."""
    from figdata import save_fd
    F = ramp_compute(D, RR, kind, iso, what)
    pd.DataFrame(F["sig_recs"]).to_csv(OUT / f"sig_bins_{Path(path).stem}.csv", index=False)
    save_fd(Path(path).stem, F)
    ramp_draw(plt, F, colors, path)


def ramp_compute(D: dict, RR: pd.DataFrame, kind: str, iso: bool, what: str) -> dict:
    """Pooled PETH per epoch: mean of animals of session means, each animal's PETHs scaled by its pre amplitude; band =
    SEM over animals (Priya 10-26). Returns the figure data (curves + per-bin tests per ROI)."""
    shown = (LAGS_P >= -1.5) & (LAGS_P <= 0.75)
    sig_recs, panels = [], []
    for roi in RAMP_ROIS:
        long, curv = [], []
        for ep in EP4:
            curves = []
            for a in AN:
                cs, labs = [], []
                for x, d in D.items():
                    P = _bout(d, GAP_MAIN, isolated=iso)
                    if str(d["animal"]) != a or str(d["epoch"]) != ep or len(P) < 10 or roi not in list(d["cols"]):
                        continue
                    P = P.astype(float)[:, :, list(d["cols"]).index(roi)]
                    P = P - P[:, (LAGS_P >= BASE[0]) & (LAGS_P <= BASE[1])].mean(1, keepdims=True)
                    cs.append(P.mean(0))
                    labs.append(x)
                pre_a = RR[(RR.kind == kind) & (RR.roi == roi) & (RR.animal == a) & (RR.epoch == "pre")].amp.mean()
                if cs and np.isfinite(pre_a) and abs(pre_a) > 0:
                    curves.append(np.mean(cs, 0) / pre_a)
                    long += [pd.DataFrame({"label": x, "animal": a, "epoch": ep, "lag": LAGS_P[shown], "v": c[shown] / pre_a})
                             for x, c in zip(labs, cs)]
            if curves:
                cv = np.asarray(curves)
                mu = np.nanmean(cv, 0)
                se = np.nanstd(cv, 0, ddof=1) / np.sqrt(len(cv)) if len(cv) > 1 else None
                curv.append((ep, mu, se, len(curves)))
        S = sig_compute(pd.concat(long, ignore_index=True), value="v", tag={"roi": roi, "kind": kind}) if long else None
        if S is not None:
            sig_recs += S["recs"]
        panels.append({"roi": roi, "curves": curv, "sig": S})
    return {"lags": LAGS_P, "panels": panels, "what": what, "sig_recs": sig_recs}


def ramp_draw(plt, F: dict, colors: dict, path) -> None:
    fig, axs = plt.subplots(2, len(F["panels"]) // 2 + 1, figsize=(19, 7.5), squeeze=False)
    lags = F["lags"]
    for i, p in enumerate(F["panels"]):
        ax = axs.flat[i]
        for ep, mu, se, n in p["curves"]:
            ax.plot(lags, mu, color=colors[ep], lw=2, label=f"{ep} ({n} animals)")
            if se is not None:
                ax.fill_between(lags, mu - se, mu + se, color=colors[ep], alpha=0.18, lw=0)
        ax.axvline(0, color="0.6", lw=0.6)
        ax.axhline(0, color="0.8", lw=0.6)
        if p["sig"] is not None:
            sig_draw(ax, p["sig"], colors)
            _headroom([ax])
        ax.set_title(p["roi"], fontsize=9)
        ax.set_xlim(-1.5, 0.75)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in axs.flat[len(F["panels"]):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7, frameon=False)
    axs.flat[0].set_ylabel("dF/F / animal's pre amplitude at the lick")
    fig.suptitle(f"NEURAL, NO DECODER: dF/F around SELF-INITIATED licks ({F['what']}; no cue / spout move within -2..+0.75 s), "
                 "lick-ONSET aligned, baseline -1.5..-1.0 s; line = mean of animals, band = SEM over animals, each scaled by its pre amplitude "
                 f"at the lick. ALM = anterolateral {ROI_FRAC['ALM']:.0%} of MOs, tjM1 = anterolateral "
                 f"{ROI_FRAC['tjM1']:.0%} of MOp; _right = contralesional.\n" + SIG_NOTE +
                 " Tests on session PETHs (/ animal's pre amplitude).", fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=110)
    plt.close(fig)


SIG_Q = 0.05
SIG_NOTE = ("Bars above the curves: time bins where that epoch differs from pre\n(paired nested bootstrap animals > "
            "sessions per bin, BH-FDR q = 0.05 across the bins of the curve), one row per epoch in its colour.")


def sig_bars(ax, t: pd.DataFrame, colors: dict, value: str = "r", rows_top: float = 0.975, row_h: float = 0.04,
             tag: dict | None = None) -> list[dict]:
    """Per-time-bin significance of each post epoch vs pre on one curve (Priya 10-26: "mark with lines (colored per
    epoch) where in the time bins curves differ from pre"). ``t`` = one row per session x lag (columns label, animal,
    epoch, lag, ``value``). Per post epoch: `analysis_kit.boot_delta_columns` (the SAME paired animals > sessions draw as
    every dot-plot star, vectorised over lags) -> `boot_p_columns` -> `bh` (q = SIG_Q) across that curve's lags.
    Contiguous significant runs are drawn as thick segments in the epoch colour in a strip at the top of the axes (x in
    data, y in axes coordinates), one row per epoch; each segment spans half a bin either side. Returns one record per
    epoch x lag for the tests CSV. = `sig_compute` (the test) + `sig_draw` (the strip)."""
    S = sig_compute(t, value=value, tag=tag)
    sig_draw(ax, S, colors, rows_top=rows_top, row_h=row_h)
    return S["recs"]


def sig_compute(t: pd.DataFrame, value: str = "r", tag: dict | None = None) -> dict:
    """The test half of `sig_bars` (no axes): {"rows": [(k, epoch, [(x0, x1), ...])], "recs": [...]}; picklable, so a
    figure-data cache can hold it and `sig_draw` redraws the strip without re-running the bootstrap."""
    from wfield_local import analysis_kit as ak
    W = t.groupby(["label", "animal", "epoch", "lag"])[value].mean().unstack("lag").sort_index(axis=1)
    lags = W.columns.to_numpy(float)
    half = np.median(np.diff(lags)) / 2 if len(lags) > 1 else 0.05
    pre = {a: g.to_numpy(float) for a, g in W.xs("pre", level="epoch").groupby(level="animal")} \
        if "pre" in W.index.get_level_values("epoch") else {}
    recs, rows = [], []
    for k, ep in enumerate([e for e in EP4 if e != "pre"]):
        if ep not in W.index.get_level_values("epoch") or not pre:
            continue
        post = {a: g.to_numpy(float) for a, g in W.xs(ep, level="epoch").groupby(level="animal")}
        res = ak.boot_delta_columns(post, pre, np.random.default_rng(0))
        if res is None:
            continue
        point, lo, hi, draws, n_an = res
        sig = ak.bh(ak.boot_p_columns(draws), SIG_Q)
        segs = []
        i = 0
        while i < len(lags):
            if sig[i]:
                j = i
                while j + 1 < len(lags) and sig[j + 1]:
                    j += 1
                segs.append((float(lags[i] - half), float(lags[j] + half)))
                i = j + 1
            else:
                i += 1
        rows.append((k, ep, segs))
        recs += [dict(**(tag or {}), epoch=ep, lag=float(lg), delta=float(pt), lo=float(lo_), hi=float(hi_),
                      sig_bh=bool(s), n_animals=n_an) for lg, pt, lo_, hi_, s in zip(lags, point, lo, hi, sig)]
    return {"rows": rows, "recs": recs}


def sig_draw(ax, S: dict, colors: dict, rows_top: float = 0.975, row_h: float = 0.04) -> None:
    """Draw a `sig_compute` result: one strip row per epoch at the top of the axes (x data, y axes)."""
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for k, ep, segs in S["rows"]:
        y = rows_top - k * row_h
        for x0, x1 in segs:
            ax.plot([x0, x1], [y, y], color=colors[ep], lw=2.6, alpha=0.5, solid_capstyle="butt", transform=tr,
                    clip_on=False)


def _headroom(axs, frac: float = 0.2) -> None:
    """Raise every axis top by ``frac`` of its range so the significance strip sits above the data."""
    for ax in np.ravel(axs):
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + frac * (hi - lo))


def plot_direction(DR: pd.DataFrame, FZ: pd.DataFrame) -> None:
    """Page 3.7: within-session direction decoding per lag (session-pooled mean +- SEM) + the frozen pre-ITI decoder.
    Called by summarise(); `--plot-direction` redraws it from decode_sessions.csv / decode_frozen_sessions.csv
    (re-running the per-bin tests); `--plot-only` redraws it from figdata/lick_direction_decoding.pkl alone."""
    from figdata import save_fd
    F = direction_compute(DR, FZ)
    pd.DataFrame(F["sig_recs"]).to_csv(OUT / "sig_bins_3_7.csv", index=False)
    save_fd("lick_direction_decoding", F)
    direction_draw(F)


def direction_compute(DR: pd.DataFrame, FZ: pd.DataFrame) -> dict:
    """Figure data of page 3.7: per (target, set) panel the per-epoch session-pooled curves + per-bin tests; per target
    the frozen decoder's animal x epoch table."""
    panels, frozen, sig_recs = {}, {}, []
    for tg in ("direction", "relative"):
        for s_ in ("iti", "trial"):
            t = DR[(DR.target == tg) & (DR.set == s_)]
            S = sig_compute(t, tag={"figure": "3.7", "set": s_, "target": tg})
            sig_recs += S["recs"]
            curves = []
            for ep in EP4:
                g = t[t.epoch == ep].groupby(["label", "lag"]).r.mean().groupby("lag")   # session-pooled, as 3.8
                m, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
                if len(m):
                    curves.append((ep, m.index.to_numpy(), m.to_numpy(), (m - se).to_numpy(), (m + se).to_numpy(),
                                   t[t.epoch == ep].label.nunique()))
            panels[(tg, s_)] = {"sig": S, "curves": curves}
        f = FZ[FZ.target == tg] if not FZ.empty else FZ
        frozen[tg] = f.groupby(["animal", "epoch"]).r.mean().unstack("epoch").reindex(columns=EP4) if not f.empty else None
    return {"panels": panels, "frozen": frozen, "sig_recs": sig_recs}


def direction_draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.7]})
    for i, tg in enumerate(("direction", "relative")):
        for j, s_ in enumerate(("iti", "trial")):
            ax = axs[i, j]
            p = F["panels"][(tg, s_)]
            sig_draw(ax, p["sig"], colors)
            for ep, x, m, lo, hi, n in p["curves"]:
                ax.plot(x, m, color=colors[ep], lw=2, label=f"{ep} ({n} sessions)")
                ax.fill_between(x, lo, hi, color=colors[ep], alpha=0.18, lw=0)
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            rel = "relative to the PREVIOUS trial's spout" if s_ == "iti" else "within the spout in place"
            ax.set_title(f"PER-SESSION decoder | {'ITI (self-initiated; spout retracted)' if s_ == 'iti' else 'in-trial post-cue (n-matched)'}"
                         f":\n{'lateral tongue-tip position (3-D, mm)' if tg == 'direction' else 'lateral position ' + rel}",
                         fontsize=9)
            ax.set_xlabel("lag from lick peak (s)")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        ax = axs[i, 2]                                     # FROZEN pre-stroke ITI decoder, per animal across epochs
        x = np.arange(len(EP4))
        P = F["frozen"][tg]
        if P is not None:
            for _a, v in P.iterrows():
                ax.plot(x, v.values, color="0.6", lw=0.8, marker="o", ms=3)
            ax.plot(x, P.mean().values, color="k", lw=2.2, marker="o")
            for k_, ep in enumerate(EP4):
                ax.scatter([k_], [P[ep].mean()], s=60, color=colors[ep], zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(["pre (LOSO)", "acute", "sub", "chronic"], fontsize=8)
        ax.set_title("FROZEN pre-stroke decoder (ITI licks, never refit)\n(lags 0..0.2 s; grey = animals)", fontsize=9)
        ax.axhline(0, color="0.8", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    _headroom([axs[0, 0]])                              # y is shared: raise once
    axs[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    axs[0, 0].set_ylabel("r, decoded vs actual\nlateral position", fontsize=9)
    axs[1, 0].set_ylabel("r, decoded vs actual\nlateral position (relative)", fontsize=9)
    fig.suptitle("NEURAL: decoding each lick's lateral tongue-tip position (3-D, mm) from cortical area activity; y = r between "
                 "decoded and actual position.\nLeft / middle: " + REFIT_TAG + " (ridge, 5 time-block folds), per lag, mean over "
                 "sessions +- SEM.\n" + SIG_NOTE + "\nRight: " + FROZEN_TAG + ", applied unchanged to every session (pre "
                 "scored leave-one-session-out).", fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "figures" / "lick_direction_decoding.png", dpi=110)
    plt.close(fig)


RAMP_FIGS = (("iti_bout", False, "lick_iti_ramp_peth.png", "bout-start ITI licks: no lick for 2 s"),
             ("iti_isolated", True, "lick_iti_ramp_peth_isolated.png",
              "ISOLATED ITI licks: no lick for 2 s before AND none within 0.75 s after"))


def ramp_figures(D: dict | None, RR: pd.DataFrame | None, plot_only: bool = False) -> None:
    """Pages 3.16 / 3.19 (bout-start and isolated ITI-lick PETHs); `--ramp` redraws them from ramp_sessions.csv;
    plot_only=True redraws them from figdata/<figure>.pkl alone (no session load, no test)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors   # the epoch palette used across the page
    for kind, iso, fname, what in RAMP_FIGS:
        if plot_only:
            from figdata import load_fd
            ramp_draw(plt, load_fd(Path(fname).stem), colors, OUT / "figures" / fname)
        else:
            ramp_figure(plt, D, RR, kind, iso, colors, OUT / "figures" / fname, what)


def summarise(labels: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")

    from wfield_local import analysis_kit as ak
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    D = {x: _load(x) for x in labels if (OUT / SESS_NAME / f"{x}.npz").exists()}
    meta = pd.DataFrame([{"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "clock_med_s": float(d["clock_med_s"]),
                          "n_bout": len(d["bout_side"]), "n_iti": len(d["iti_side"]), "n_first": len(d["first_rt"])}
                         for x, d in D.items()])
    meta.to_csv(OUT / "sessions_meta.csv", index=False)
    print("== sessions per animal x epoch (n bout-start ITI licks, summed)\n"
          + meta.pivot_table(index="animal", columns="epoch", values="n_bout", aggfunc="sum").reindex(columns=EP4).to_string())
    print(f"clock check (median |3-D post-cue peak - behaviour lick peak|, s): max over sessions {meta.clock_med_s.max():.3f}")
    bl = pd.DataFrame([{"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]),
                        "nfol": float(d["bout_nfol"][d["bout_gap"] >= GAP_MAIN].mean()),
                        "isolated": float((d["bout_nfol"][d["bout_gap"] >= GAP_MAIN] == 0).mean())}
                       for x, d in D.items() if (d["bout_gap"] >= GAP_MAIN).sum() >= 10])
    bl.to_csv(OUT / "bout_length_sessions.csv", index=False)
    print("\n== per animal (rule 8): licks FOLLOWING a bout start within 0.75 s (mean) / share isolated\n"
          + bl.pivot_table(index="animal", columns="epoch", values=["nfol", "isolated"]).round(2).to_string())
    # (A) ramp metrics per session x ROI (bout-start ITI licks, >= 10 per session) and trial first licks
    rr = []
    for x, d in D.items():
        cols = list(d["cols"])
        for kind, P in (("iti_bout", _bout(d, GAP_MAIN)), ("iti_bout_gap1", _bout(d, BOUT_GAP)),
                        ("iti_isolated", _bout(d, GAP_MAIN, isolated=True)), ("trial_first", d["first_peth"])):
            if len(P) < 10:
                continue
            P = P.astype(float)
            b = P[:, (LAGS_P >= BASE[0]) & (LAGS_P <= BASE[1]), :].mean(1, keepdims=True)
            m = (P - b).mean(0)
            for roi in RAMP_ROIS:
                if roi in cols:
                    v = m[:, cols.index(roi)]
                    A, t50, t20 = ramp_metrics(v, LAGS_P)
                    rr.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "kind": kind, "roi": roi,
                               "n": len(P), "amp": A, "t50": t50, "t20": t20,
                               "early": float(v[(LAGS_P >= -0.6) & (LAGS_P <= -0.2)].mean()),
                               "peak": float(v[(LAGS_P >= 0.1) & (LAGS_P <= 0.4)].mean())})
    RR = pd.DataFrame(rr)
    # AMPLITUDE-FREE readouts (t50 / t20 are fractions of the amplitude AT the lick, which rose post-stroke -- a larger
    # post-onset response alone moves t50 later): early ramp (-0.6..-0.2 s) and post-onset peak (0.1..0.4 s), each in
    # units of the animal's PRE amplitude at the lick (same kind x ROI), so epochs compare on one scale per animal
    pa = RR[RR.epoch == "pre"].groupby(["kind", "roi", "animal"]).amp.mean()
    den = np.array([pa.get((k, r_, a), np.nan) for k, r_, a in zip(RR.kind, RR.roi, RR.animal)])
    den = np.where(np.abs(den) > 0, den, np.nan)
    RR["early_n"], RR["peak_n"] = RR.early / den, RR.peak / den
    RR.to_csv(OUT / "ramp_sessions.csv", index=False)
    print("\n== per animal (rule 8): t50 (s from lick ONSET), bout-start ITI licks\n"
          + RR[RR.kind == "iti_bout"].pivot_table(index=["roi", "animal"], columns="epoch", values="t50")
          .reindex(columns=EP4).round(2).to_string())
    bt = []
    for (kind, roi), t in RR.groupby(["kind", "roi"]):
        for val in ("t50", "t20", "amp", "early_n", "peak_n"):
            ref = {a: t[(t.animal == a) & (t.epoch == "pre")][val].dropna().to_numpy() for a in AN}
            for ep in EP4[1:]:
                post = {a: t[(t.animal == a) & (t.epoch == ep)][val].dropna().to_numpy() for a in AN}
                com = [a for a in AN if len(ref[a]) and len(post[a])]
                if len(com) < 3:
                    continue
                iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                bt.append({"kind": kind, "roi": roi, "value": val, "epoch": ep, "n_animals": len(com),
                           "pre": float(np.mean([ref[a].mean() for a in com])), "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                           "n_down": sum(post[a].mean() < ref[a].mean() for a in com)})
    BT = pd.DataFrame(bt, columns=["kind", "roi", "value", "epoch", "n_animals", "pre", "delta", "lo", "hi", "n_down"])
    BT.to_csv(OUT / "ramp_change.csv", index=False)
    print("\n== per animal (rule 8): EARLY RAMP (-0.6..-0.2 s) / animal's pre amplitude, bout-start ITI licks\n"
          + RR[RR.kind == "iti_bout"].pivot_table(index=["roi", "animal"], columns="epoch", values="early_n")
          .reindex(columns=EP4).round(2).to_string())
    print("\n== ramp change vs pre (nested bootstrap; t: negative = EARLIER rise; n_down = animals whose mean fell)\n"
          + BT[BT.value != "amp"].round(3).to_string(index=False))
    ramp_figures(D, RR)
    # (B) decoding: within session, per lag
    rng = np.random.default_rng(0)
    dr = []
    for x, d in D.items():
        for r_ in decode_session(d, rng):
            dr.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), **r_})
    DR = pd.DataFrame(dr)
    DR.to_csv(OUT / "decode_sessions.csv", index=False)
    if DR.empty:
        print("no session with >= 50 ITI licks")
        return
    win = DR[(DR.lag >= 0.0) & (DR.lag <= 0.2)].groupby(["label", "animal", "epoch", "set", "target"]).r.mean().reset_index()
    print("\n== per animal (rule 8): decoding r of lateral direction, lags 0..0.2 s, mean over sessions\n"
          + win.pivot_table(index=["target", "set", "animal"], columns="epoch", values="r").reindex(columns=EP4).round(2)
          .to_string())
    db = []
    for (s_, tg), t in win.groupby(["set", "target"]):
        ref = {a: t[(t.animal == a) & (t.epoch == "pre")].r.dropna().to_numpy() for a in AN}
        for ep in EP4[1:]:
            post = {a: t[(t.animal == a) & (t.epoch == ep)].r.dropna().to_numpy() for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            if len(com) < 3:
                continue
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
            db.append({"set": s_, "target": tg, "epoch": ep, "n_animals": len(com),
                       "pre": float(np.mean([ref[a].mean() for a in com])), "delta": iv.point, "lo": iv.lo, "hi": iv.hi})
    DB = pd.DataFrame(db)
    DB.to_csv(OUT / "decode_change.csv", index=False)
    print("\n== decoding change vs pre (nested bootstrap)\n" + DB.round(3).to_string(index=False))
    FZ = frozen_decode(D)
    plot_direction(DR, FZ)
    FZ.to_csv(OUT / "decode_frozen_sessions.csv", index=False)
    if not FZ.empty:
        print("\n== per animal (rule 8): FROZEN pre-ITI decoder, r per epoch (pre = leave-one-session-out)\n"
              + FZ.pivot_table(index=["target", "animal"], columns="epoch", values="r").reindex(columns=EP4).round(2)
              .to_string())
        fb = []
        for tg, t in FZ.groupby("target"):
            ref = {a: t[(t.animal == a) & (t.epoch == "pre")].r.dropna().to_numpy() for a in AN}
            for ep in EP4[1:]:
                post = {a: t[(t.animal == a) & (t.epoch == ep)].r.dropna().to_numpy() for a in AN}
                com = [a for a in AN if len(ref[a]) and len(post[a])]
                if len(com) >= 3:
                    iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                    fb.append({"target": tg, "epoch": ep, "n_animals": len(com), "delta": iv.point, "lo": iv.lo,
                               "hi": iv.hi})
        pd.DataFrame(fb).to_csv(OUT / "decode_frozen_change.csv", index=False)
        print("\n== frozen decoder change vs pre\n" + pd.DataFrame(fb).round(3).to_string(index=False))
    json.dump({"n_sessions": len(D)}, open(OUT / "summary.json", "w"))
    print("->", OUT)


def roi_figure(label: str | None, plot_only: bool = False) -> None:
    """lick_iti_rois.png: the ALM / tjM1 ROI placement on one session's grid; plot_only=True redraws it from
    figdata/lick_iti_rois.pkl (no imaging load)."""
    from figdata import load_fd, save_fd
    if plot_only:
        F = load_fd("lick_iti_rois")
    else:
        from frozen_area_maps import area_image, outline

        from wfield_local import combined_encoding as ce
        _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
        M = roi_masks(parea, pstat, names, flat, shape)
        H, W = (int(x) for x in shape)
        img = np.zeros(H * W)
        for v, n in enumerate(("ALM_left", "ALM_right", "tjM1_left", "tjM1_right"), 1):
            g = np.zeros(H * W, bool)
            g[flat[M[n]]] = True
            img[g] = v
        ai, _nm = area_image(label)
        F = {"label": label, "img": img.reshape(H, W), "outline": outline(ai), "roi_frac": ROI_FRAC["tjM1"], "roi_frac_alm": ROI_FRAC["ALM"]}
        save_fd("lick_iti_rois", F)
    roi_draw(F)


def roi_draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = F["img"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(np.where(img > 0, img, np.nan), cmap="Set1", vmin=1, vmax=9)
    ax.imshow(F["outline"], cmap="Greys", vmin=0, vmax=1.2, alpha=0.6)
    ax.set_title(f"ROIs: ALM = anterolateral {F.get('roi_frac_alm', F['roi_frac']):.0%} of MOs, tjM1 = "
                 f"{F['roi_frac']:.0%} of MOp\nimage left = ipsilesional", fontsize=9)
    ax.axis("off")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "lick_iti_rois.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ KEY figures (Priya 10-26)
# "3.7 frozen decoder and 3.8 are KEY results": frozen-decoder time courses for ITI and in-trial licks; summaries pooled
# over SESSIONS (one dot per session, mean +- SEM), raw and within-animal normalised to pre, with the nested-bootstrap
# test (animals -> sessions, analysis_kit.boot_delta) of each epoch against pre.
KEY_ROIS = ["ALM_right", "ALM_left", "tjM1_right", "tjM1_left", "SSp-m_right", "SSp-m_left", "cortex"]


def frozen_decode_lags(D: dict, set_name: str) -> pd.DataFrame:
    """FROZEN decoder per LAG: ridge trained on each animal's August PRE licks of the set (ITI or in-trial post-cue) at
    that lag, pre sessions scored leave-one-session-out, post sessions with the full pre fit. Targets: lateral direction
    and direction RELATIVE to the previous trial's spout (ITI) / the spout in place (in-trial)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    kx, ky, kp = ("iti_X", "iti_side", "iti_pos") if set_name == "iti" else ("tr_X", "tr_side", "tr_pos")
    rows = []
    for a in AN:
        S = {x: d for x, d in D.items() if str(d["animal"]) == a and len(d[ky]) >= 20}
        pre = [x for x, d in S.items() if str(d["epoch"]) == "pre"]
        if len(pre) < 3:
            continue
        feat = [i for i, c in enumerate(next(iter(S.values()))["cols"]) if c != "cortex" and not c.startswith(("ALM", "tjM1"))]
        Y = {tg: {x: (S[x][ky] if tg == "direction" else _within(S[x][ky], S[x][kp])) for x in S}
             for tg in ("direction", "relative")}
        for li, lag in enumerate(LAGS_D):
            X = {x: S[x][kx][:, li, :][:, feat].astype(float) for x in S}
            for tg in ("direction", "relative"):
                fits = {}
                for held in [*pre, None]:
                    trn = [q for q in pre if q != held]
                    Xt = np.vstack([X[q] for q in trn])
                    sc = StandardScaler().fit(Xt)
                    fits[held] = (sc, RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(sc.transform(Xt),
                                                                                np.concatenate([Y[tg][q] for q in trn])))
                for x in S:
                    sc, m = fits[x if x in pre else None]
                    pr = m.predict(sc.transform(X[x]))
                    rows.append({"label": x, "animal": a, "epoch": str(S[x]["epoch"]), "set": set_name, "target": tg,
                                 "lag": float(lag), "r": float(np.corrcoef(pr, Y[tg][x])[0, 1]) if np.std(pr) > 0 else np.nan})
    return pd.DataFrame(rows)


def _star(ax, x):
    """Significance star in a strip ABOVE the data (axes y 0.95; session_dots adds the headroom), so it never sits on a
    dot, an error bar or a reference line (Priya 10-26)."""
    import matplotlib.patheffects as pe
    ax.text(x, 0.95, "*", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=14, fontweight="bold",
            zorder=6, path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])


def session_dots(ax, T: pd.DataFrame, val: str, colors: dict, rng) -> list[dict]:
    """One dot per session per epoch (epoch colour), mean +- SEM over sessions, * = nested-bootstrap CI of the change
    from pre excludes 0. T: label, animal, epoch, <val>. Returns the test rows. = `dots_compute` + `dots_draw`."""
    S = dots_compute(T, val, rng)
    dots_draw(ax, S, colors)
    return S["tests"]


def dots_compute(T: pd.DataFrame, val: str, rng) -> dict:
    """The data + test half of `session_dots` (no axes; draws the jitter from ``rng`` in the same order): picklable."""
    from wfield_local import analysis_kit as ak
    out, stars, eps = [], [], []
    ref = {a: T[(T.animal == a) & (T.epoch == "pre")][val].dropna().to_numpy() for a in AN}
    for k, ep in enumerate(EP4):
        v = T[T.epoch == ep][val].dropna().to_numpy()
        if not len(v):
            continue
        xj = k + rng.uniform(-0.18, 0.18, len(v))
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        eps.append({"k": k, "ep": ep, "x": xj, "v": v, "m": m, "se": se})
        if ep != "pre":
            post = {a: T[(T.animal == a) & (T.epoch == ep)][val].dropna().to_numpy() for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            if len(com) >= 3:
                iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                out.append({"value": val, "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi, "n_sessions": len(v)})
                if iv.lo > 0 or iv.hi < 0:
                    stars.append(k)
    return {"eps": eps, "stars": stars, "tests": out}


def dots_draw(ax, S: dict, colors: dict) -> None:
    """Draw a `dots_compute` result (dots, mean +- SEM, n, headroom, stars)."""
    for e in S["eps"]:
        k = e["k"]
        ax.scatter(e["x"], e["v"], s=12, color=colors[e["ep"]], alpha=0.55, lw=0, zorder=3)
        ax.errorbar(k + 0.3, e["m"], yerr=e["se"], fmt="o", color="k", ms=6, capsize=5, lw=2.2, capthick=2.2, zorder=5)
        ax.text(k, 0.01, f"n={len(e['v'])}", ha="center", va="bottom", fontsize=6.5, color="0.4",
                transform=ax.get_xaxis_transform())          # x in data, y in axes: always at the bottom
    ax.set_xticks(range(len(EP4)))
    ax.set_xticklabels(["pre", "acute", "sub", "chronic"], fontsize=8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))                 # headroom: a star strip above every dot and reference line
    for k in S["stars"]:
        _star(ax, k)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _norm_to_pre(T: pd.DataFrame, val: str, how: str = "ratio") -> pd.Series:
    pre = T[T.epoch == "pre"].groupby("animal")[val].mean()
    den = T.animal.map(pre)
    return T[val] / den if how == "ratio" else T[val] - den


def _auc(v: np.ndarray, t0: float, t1: float) -> float:
    m = (LAGS_P >= t0 - 1e-9) & (LAGS_P <= t1 + 1e-9)
    return float(np.trapezoid(v[m], LAGS_P[m])) if hasattr(np, "trapezoid") else float(np.trapz(v[m], LAGS_P[m]))


def plot_frozen_timecourse(FL: pd.DataFrame) -> None:
    """Page 3.8: the frozen pre-stroke decoder per lag (session-pooled mean +- SEM) with per-bin significance bars;
    called by key_figures, `--plot-frozen-tc` redraws it from decode_frozen_lags_sessions.csv (re-running the per-bin
    tests); `--plot-only` redraws it from figdata/lick_frozen_timecourse.pkl alone."""
    from figdata import save_fd
    panels, sig_recs = {}, []
    for tg in ("direction", "relative"):
        for st in ("iti", "trial"):
            t = FL[(FL.target == tg) & (FL.set == st)]
            S = sig_compute(t, tag={"figure": "3.8", "set": st, "target": tg})
            sig_recs += S["recs"]
            curves = []
            for ep in EP4:
                g = t[t.epoch == ep].groupby("lag").r
                m, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
                if len(m):
                    curves.append((ep, m.index.to_numpy(), m.to_numpy(), (m - se).to_numpy(), (m + se).to_numpy(),
                                   t[t.epoch == ep].label.nunique()))
            panels[(tg, st)] = {"sig": S, "curves": curves}
    pd.DataFrame(sig_recs).to_csv(OUT / "sig_bins_3_8.csv", index=False)
    F = {"panels": panels}
    save_fd("lick_frozen_timecourse", F)
    frozen_tc_draw(F)


def frozen_tc_draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    set_lab = {"iti": "ITI (self-initiated; spout retracted; reference = previous trial's spout)",
               "trial": "in-trial post-cue (reference = spout in place)"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharey=True, sharex=True)
    for i, tg in enumerate(("direction", "relative")):
        for j, st in enumerate(("iti", "trial")):
            ax = axs[i, j]
            p = F["panels"][(tg, st)]
            sig_draw(ax, p["sig"], colors)
            for ep, x, m, lo, hi, n in p["curves"]:
                ax.plot(x, m, color=colors[ep], lw=2, label=f"{ep} ({n} sessions)")
                ax.fill_between(x, lo, hi, color=colors[ep], alpha=0.18, lw=0)
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.set_title(f"{set_lab[st]}\n{TGT_LAB[tg]}", fontsize=8.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 1:
                ax.set_xlabel("lag from lick peak (s)")
            if j == 0:
                ax.set_ylabel("FROZEN decoder r\n(decoded vs actual lateral position)", fontsize=8.5)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    _headroom([axs[0, 0]])                              # y is shared: raise once
    fig.suptitle("NEURAL: " + FROZEN_TAG + ", per lag from the lick peak; y = r between decoded and actual lateral "
                 "tongue-tip position.\nTrained on each animal's PRE-stroke licks of the same kind (ITI or in-trial; pre "
                 "sessions leave-one-session-out), applied unchanged to every session. Line = mean over sessions (all animals "
                 "pooled), band = SEM.\n" + SIG_NOTE, fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "figures" / "lick_frozen_timecourse.png", dpi=110)
    plt.close(fig)


KEY_SPEC = (("resp_auc", "response AUC 0..0.5 s (dF/F x1e3 * s)"), ("resp_auc_n", "response AUC / animal's pre response AUC"),
            ("build_auc", "build-up AUC -0.4..-0.05 s before tongue visible"),
            ("build_auc_n", "build-up AUC / animal's pre response AUC"))


def key_figures(D: dict, RR: pd.DataFrame) -> None:
    """KEY figures (3.7 / 3.8 family): compute every statistic, save one figdata file per figure, then draw
    (`key_figures_plot` redraws them from the figdata alone)."""
    from figdata import save_fd
    rng = np.random.default_rng(1)
    # ---- 3.7: frozen decoder time courses + session summaries
    FL = pd.concat([frozen_decode_lags(D, "iti"), frozen_decode_lags(D, "trial")])
    FL.to_csv(OUT / "decode_frozen_lags_sessions.csv", index=False)
    plot_frozen_timecourse(FL)
    W = FL[(FL.lag >= 0.0) & (FL.lag <= 0.2)].groupby(["label", "animal", "epoch", "set", "target"]).r.mean().reset_index()
    tests, panels = [], {}
    for j, (st, tg) in enumerate([(a, b) for a in ("iti", "trial") for b in ("direction", "relative")]):
        T = W[(W.set == st) & (W.target == tg)].copy()
        T["r_norm"] = _norm_to_pre(T, "r")
        for i, val in enumerate(("r", "r_norm")):
            S = dots_compute(T, val, rng)
            panels[(i, j)] = (st, tg, S)
            for r_ in S["tests"]:
                tests.append({"set": st, "target": tg, **r_})
    pd.DataFrame(tests).to_csv(OUT / "decode_frozen_session_tests.csv", index=False)
    save_fd("lick_frozen_summary", {"panels": panels})
    print("\n== frozen decoder (lags 0..0.2), change vs pre, session-pooled figure tests\n"
          + pd.DataFrame(tests).round(3).to_string(index=False))
    # ---- 3.8: per-session PETH mean +- SEM, raw and normalised; amplitude-free summaries
    rows = []
    for x, d in D.items():
        P = _bout(d, GAP_MAIN)
        if len(P) < 10:
            continue
        P = P.astype(float)
        P = P - P[:, (LAGS_P >= BASE[0]) & (LAGS_P <= BASE[1]), :].mean(1, keepdims=True)
        m = P.mean(0)
        for roi in KEY_ROIS:
            if roi in list(d["cols"]):
                rows.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "roi": roi,
                             "peth": m[:, list(d["cols"]).index(roi)]})
    PE = pd.DataFrame(rows)
    pa = RR[(RR.kind == "iti_bout") & (RR.epoch == "pre")].groupby(["roi", "animal"]).amp.mean()
    shown = (LAGS_P >= -1.5) & (LAGS_P <= 0.75)
    sig_recs, rp = [], {}
    for j, roi in enumerate(KEY_ROIS):
        for i, norm in enumerate((False, True)):
            long, curves = [], []
            for ep in EP4:
                t = PE[(PE.roi == roi) & (PE.epoch == ep)]
                if t.empty:
                    continue
                M = np.stack(t.peth.to_numpy())
                if norm:
                    M = M / np.array([pa.get((roi, a), np.nan) for a in t.animal])[:, None]
                mu, se = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt(np.isfinite(M[:, 0]).sum())
                curves.append((ep, mu, se, len(t)))
                long += [pd.DataFrame({"label": x, "animal": a, "epoch": ep, "lag": LAGS_P[shown], "v": m_[shown]})
                         for x, a, m_ in zip(t.label, t.animal, M)]
            S = sig_compute(pd.concat(long, ignore_index=True), value="v", tag={"roi": roi, "normalised": norm}) \
                if long else None
            if S is not None:
                sig_recs += S["recs"]
            rp[(i, j)] = {"roi": roi, "norm": norm, "curves": curves, "sig": S}
    save_fd("lick_iti_ramp_sessions", {"lags": LAGS_P, "panels": rp, "rois": list(KEY_ROIS)})
    pd.DataFrame(sig_recs).to_csv(OUT / "sig_bins_3_24.csv", index=False)
    # AUC readouts (Priya 10-26): response = AUC 0..0.5 s after lick onset (= the first frame the tongue is visible,
    # cam4 `on_frame`); build-up = AUC -0.4..-0.05 s BEFORE the tongue is visible. dF/F x1e3 * s; normalised versions
    # divide by the animal's PRE response AUC (one scale for both rows, pre response = 1)
    rows = []
    for _i, r_ in PE.iterrows():
        v = r_.peth
        rows.append({"label": r_.label, "animal": r_.animal, "epoch": r_.epoch, "roi": r_.roi,
                     "resp_auc": _auc(v, 0.0, 0.5), "build_auc": _auc(v, -0.4, -0.05)})
    T0 = pd.DataFrame(rows)
    den = T0[T0.epoch == "pre"].groupby(["roi", "animal"]).resp_auc.mean()
    d_ = np.array([den.get((r_, a_), np.nan) for r_, a_ in zip(T0.roi, T0.animal)])
    T0["resp_auc_n"], T0["build_auc_n"] = T0.resp_auc / d_, T0.build_auc / d_
    T0.to_csv(OUT / "ramp_auc_sessions.csv", index=False)
    tests, sp_ = [], {}
    for j, roi in enumerate(KEY_ROIS):
        T = T0[T0.roi == roi]
        for i, (val, _ylab) in enumerate(KEY_SPEC):
            S = dots_compute(T, val, rng)
            sp_[(i, j)] = S
            for r_ in S["tests"]:
                tests.append({"roi": roi, **r_})
    pd.DataFrame(tests).to_csv(OUT / "ramp_session_tests.csv", index=False)
    save_fd("lick_iti_ramp_summary", {"panels": sp_, "rois": list(KEY_ROIS), "spec": KEY_SPEC})
    key_figures_plot(only=("lick_iti_ramp_sessions", "lick_frozen_summary", "lick_iti_ramp_summary"))
    print("\n== ramp per-session tests (change vs pre)\n" + pd.DataFrame(tests).round(3).to_string(index=False))


KEY_STEMS = ("lick_frozen_timecourse", "lick_frozen_summary", "lick_iti_ramp_sessions", "lick_iti_ramp_summary")


def key_figures_plot(only: tuple = KEY_STEMS) -> None:
    """Draw the KEY figures from their figdata files alone (no session load, no fit, no bootstrap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figdata import load_fd

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    if "lick_frozen_timecourse" in only:
        frozen_tc_draw(load_fd("lick_frozen_timecourse"))
    if "lick_frozen_summary" in only:
        F = load_fd("lick_frozen_summary")
        fig, axs = plt.subplots(2, 4, figsize=(17, 8))
        for (i, j), (st, tg, S) in sorted(F["panels"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
            ax = axs[i, j]
            if i == 1:
                ax.axhline(1, color="0.7", lw=0.8, ls="--")
            dots_draw(ax, S, colors)
            if i == 0:
                ax.set_title(f"{'ITI' if st == 'iti' else 'in-trial post-cue'}: {TGT_LAB[tg]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(("FROZEN decoder: " + YR, "same r / animal's pre mean (pre = 1)")[i])
        fig.suptitle("NEURAL: " + FROZEN_TAG + " (lags 0..0.2 s); y = r between decoded and actual lateral tongue-tip position.\n"
                     "One dot per session, all animals pooled; black = mean +- SEM over sessions. Top: r; bottom: r normalised "
                     "within animal to its own pre mean. * = change from pre with "
                     "95% nested-bootstrap CI (animals > sessions) excluding 0.", fontsize=9.5, wrap=True)
        fig.tight_layout(rect=(0.01, 0, 0.99, 0.9))
        fig.savefig(OUT / "figures" / "lick_frozen_summary.png", dpi=110)
        plt.close(fig)
    if "lick_iti_ramp_sessions" in only:
        F = load_fd("lick_iti_ramp_sessions")
        lags = F["lags"]
        fig, axs = plt.subplots(2, len(F["rois"]), figsize=(21, 7.5), sharex=True)
        for j, _roi in enumerate(F["rois"]):
            for i in (0, 1):
                ax = axs[i, j]
                p = F["panels"][(i, j)]
                for ep, mu, se, n in p["curves"]:
                    ax.plot(lags, mu, color=colors[ep], lw=2, label=f"{ep} ({n})")
                    ax.fill_between(lags, mu - se, mu + se, color=colors[ep], alpha=0.2, lw=0)
                if p["sig"] is not None:
                    sig_draw(ax, p["sig"], colors)
                    _headroom([ax])
                ax.axvline(0, color="0.6", lw=0.6)
                ax.axhline(0, color="0.8", lw=0.6)
                ax.set_xlim(-1.5, 0.75)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if i == 0:
                    ax.set_title(p["roi"], fontsize=9)
                if j == 0:
                    ax.set_ylabel("dF/F x1e3 (baseline-subtracted)" if not p["norm"] else "/ animal's pre amplitude at the lick")
                if i == 1:
                    ax.set_xlabel("time from lick onset (s)")
        axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
        fig.suptitle("NEURAL, NO DECODER: dF/F around self-initiated licks (first ITI lick after >= 2 s without licking), per-session mean "
                     "PETH, all animals pooled: line = mean over sessions, band = SEM (n sessions in legend). Top: raw; bottom: "
                     "each session divided by its animal's PRE amplitude at the lick (within-animal normalised).\n" + SIG_NOTE,
                     fontsize=9.5, wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(OUT / "figures" / "lick_iti_ramp_sessions.png", dpi=110)
        plt.close(fig)
    if "lick_iti_ramp_summary" in only:
        F = load_fd("lick_iti_ramp_summary")
        spec = F["spec"]
        fig, axs = plt.subplots(len(spec), len(F["rois"]), figsize=(21, 13))
        for j, roi in enumerate(F["rois"]):
            for i, (_val, ylab) in enumerate(spec):
                ax = axs[i, j]
                ax.axhline(0, color="0.8", lw=0.6)
                dots_draw(ax, F["panels"][(i, j)], colors)
                if i == 0:
                    ax.set_title(roi, fontsize=9)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=8)
        fig.suptitle("NEURAL, NO DECODER: self-initiated licks, per-session dF/F AUCs (one dot per session, all animals pooled; black = mean +- SEM "
                     "over sessions). Response = AUC 0..0.5 s after lick onset (first frame the tongue is visible); build-up = AUC "
                     "-0.4..-0.05 s before it; raw and divided by the animal's PRE response AUC. * = change from pre with 95% "
                     "nested-bootstrap CI (animals > sessions) excluding 0.", fontsize=9.5, wrap=True)
        fig.tight_layout(rect=(0.01, 0, 0.99, 0.93))
        fig.savefig(OUT / "figures" / "lick_iti_ramp_summary.png", dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------------------------------------ CONTROLS (Priya 10-26)
# "lick kinematics of these licks (a bigger or faster tongue movement could drive a bigger response) and face movement
# before the lick" + "trial-to-trial variance of the FaceRhythm lick factor per epoch" (Fig 1.16 reading).
CTRL_ROIS = ["tjM1_right", "tjM1_left", "SSp-m_right", "SSp-m_left", "ALM_right", "cortex"]


def _per_lick(d: dict, roi: str, w=(0.1, 0.4)) -> np.ndarray:
    """Per bout-start (gap >= GAP_MAIN) baseline-subtracted mean activity in window w for one ROI."""
    m = d["bout_gap"] >= GAP_MAIN
    P = d["bout_peth"][m].astype(float)[:, :, list(d["cols"]).index(roi)]
    P = P - P[:, (LAGS_P >= BASE[0]) & (LAGS_P <= BASE[1])].mean(1, keepdims=True)
    return P[:, (LAGS_P >= w[0]) & (LAGS_P <= w[1])].mean(1)


def _per_lick_fr(d: dict, k: int, w) -> np.ndarray:
    m = d["bout_gap"] >= GAP_MAIN
    F = d["bout_fr"][m].astype(float)[:, :, k]
    F = F - np.nanmean(F[:, (LAGS_P >= BASE[0]) & (LAGS_P <= BASE[1])], 1, keepdims=True)
    return np.nanmean(F[:, (LAGS_P >= w[0]) & (LAGS_P <= w[1])], 1)


_KFULL = {}


def _more_kin(label: str, t_on: np.ndarray) -> pd.DataFrame:
    """Extra TONGUE kinematics for licks at exact DAQ ONSET times: 3-D side_ml / out_ap / down_dv (mm, tip at the peak) from
    kinematics_all, and the cam4 lick duration (off - on, ms) and peak retraction speed from licks_all.csv. NaN where a
    lick has no 3-D row. (Priya 10-26: regress the response on "FR plus tongue kinematics plus jaw kinematics"; jaw is
    not in the per-lick tables -- per-trial only -- so it is not here yet.)"""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["daq_on_s", "on_ms", "off_ms", "max_retract_velocity_y_px_per_s"])
    if "K" not in _KFULL:
        pth = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
        _KFULL["K"] = pd.read_csv(pth, usecols=["session", "lick_uid", "side_ml", "out_ap", "down_dv"])
    K = _KFULL["K"]
    K = K[K.session == f"{an}_{date}"].drop_duplicates("lick_uid").set_index("lick_uid")
    uid = pd.Series(np.arange(len(A)), index=A.daq_on_s.to_numpy(float))
    uid = uid[~uid.index.duplicated()]
    u = uid.reindex(np.asarray(t_on, float)).to_numpy()
    out = pd.DataFrame(np.nan, index=range(len(t_on)), columns=["side_ml", "out_ap", "down_dv", "dur_ms", "retract"])
    ok = np.isfinite(u)
    ui = u[ok].astype(int)
    out.loc[ok, ["side_ml", "out_ap", "down_dv"]] = K.reindex(ui)[["side_ml", "out_ap", "down_dv"]].to_numpy(float)
    out.loc[ok, "dur_ms"] = (A.off_ms.to_numpy(float) - A.on_ms.to_numpy(float))[ui]
    out.loc[ok, "retract"] = A.max_retract_velocity_y_px_per_s.to_numpy(float)[ui]
    return out


def controls(D: dict, RR: pd.DataFrame) -> None:
    """Kinematics / face / FR-variance controls: compute every regression + test, save one figdata file per figure,
    then draw (`controls_plot` redraws them from the figdata alone)."""
    from figdata import save_fd
    rng = np.random.default_rng(2)
    D = {x: d for x, d in D.items() if (d["bout_gap"] >= GAP_MAIN).sum() >= 10 and "bout_reach" in d}
    pa = RR[(RR.kind == "iti_bout") & (RR.epoch == "pre")].groupby(["roi", "animal"]).amp.mean()
    # per-lick table
    L = []
    for x, d in D.items():
        m = d["bout_gap"] >= GAP_MAIN
        base = pd.DataFrame({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]),
                             "reach": d["bout_reach"][m], "speed": d["bout_speed"][m]})
        mk = _more_kin(x, np.asarray(d["bout_t"], float)[m])
        for c in mk:
            base[c] = mk[c].to_numpy()
        for roi in CTRL_ROIS:
            base[f"amp_{roi}"] = _per_lick(d, roi) / pa.get((roi, str(d["animal"])), np.nan)
            base[f"early_{roi}"] = _per_lick(d, roi, (-0.6, -0.2)) / pa.get((roi, str(d["animal"])), np.nan)
        names = list(d["fr_names"])
        for k, v in enumerate(names):
            base[f"{v}_pre"] = _per_lick_fr(d, k, (-0.6, -0.2))
            base[f"{v}_post"] = _per_lick_fr(d, k, (0.1, 0.4))
        L.append(base)
    L = pd.concat(L, ignore_index=True)
    # z-scores of kinematics / face within animal, against the animal's PRE licks
    for c in ["reach", "speed", "side_ml", "out_ap", "down_dv", "dur_ms", "retract", *[c for c in L if c.startswith("fr_")]]:
        mu = L[L.epoch == "pre"].groupby("animal")[c].mean()
        sd = L[L.epoch == "pre"].groupby("animal")[c].std()
        L[f"z_{c}"] = (L[c] - L.animal.map(mu)) / L.animal.map(sd)
    # (1) kinematics-adjusted and range-matched post-onset response
    for roi in CTRL_ROIS:
        adj = np.full(len(L), np.nan)
        matched = np.zeros(len(L), bool)
        for _a, g in L.groupby("animal"):
            pre = g[(g.epoch == "pre")].dropna(subset=[f"amp_{roi}", "z_reach", "z_speed"])
            if len(pre) < 30:
                continue
            Xp = np.column_stack([np.ones(len(pre)), pre.z_reach, pre.z_speed])
            beta = np.linalg.lstsq(Xp, pre[f"amp_{roi}"].to_numpy(), rcond=None)[0]
            ok = g[["z_reach", "z_speed"]].notna().all(1)
            Xg = np.column_stack([np.ones(ok.sum()), g.z_reach[ok], g.z_speed[ok]])
            adj[g.index[ok]] = g[f"amp_{roi}"][ok].to_numpy() - Xg @ beta + beta[0]   # residual + pre intercept
            lo_r, hi_r = np.nanpercentile(pre.reach, [10, 90])
            lo_s, hi_s = np.nanpercentile(pre.speed, [10, 90])
            matched[g.index] = ((g.reach >= lo_r) & (g.reach <= hi_r) & (g.speed >= lo_s) & (g.speed <= hi_s)).to_numpy()
        L[f"adj_{roi}"] = adj
        L[f"matched_{roi}"] = np.where(matched, L[f"amp_{roi}"], np.nan)
    # (1b) the same regression with post-onset FACE movement added (Priya 10-26: "add FR adjustment"): reach + speed +
    # FaceRhythm lick / right / left factors in the response window (0.1..0.4 s), per animal on PRE licks
    fcols = [c for c in ("z_fr_lick_post", "z_fr_right_post", "z_fr_left_post") if c in L]
    for roi in CTRL_ROIS:
        adj = np.full(len(L), np.nan)
        cols = ["z_reach", "z_speed", *fcols]
        for _a, g in L.groupby("animal"):
            pre = g[g.epoch == "pre"].dropna(subset=[f"amp_{roi}", *cols])
            if len(pre) < 30:
                continue
            Xp = np.column_stack([np.ones(len(pre)), *[pre[c] for c in cols]])
            beta = np.linalg.lstsq(Xp, pre[f"amp_{roi}"].to_numpy(), rcond=None)[0]
            ok = g[cols].notna().all(1)
            Xg = np.column_stack([np.ones(ok.sum()), *[g[c][ok] for c in cols]])
            adj[g.index[ok]] = g[f"amp_{roi}"][ok].to_numpy() - Xg @ beta + beta[0]
        L[f"adjfr_{roi}"] = adj
    # (1c) FULL tongue kinematics (3-D position, reach, speed, duration, retraction) + post-onset face, per animal on PRE
    full = ["z_reach", "z_speed", "z_side_ml", "z_out_ap", "z_down_dv", "z_dur_ms", "z_retract", *fcols]
    for roi in CTRL_ROIS:
        adj = np.full(len(L), np.nan)
        for _a, g in L.groupby("animal"):
            pre = g[g.epoch == "pre"].dropna(subset=[f"amp_{roi}", *full])
            if len(pre) < 60:
                continue
            Xp = np.column_stack([np.ones(len(pre)), *[pre[c] for c in full]])
            beta = np.linalg.lstsq(Xp, pre[f"amp_{roi}"].to_numpy(), rcond=None)[0]
            ok = g[full].notna().all(1)
            Xg = np.column_stack([np.ones(ok.sum()), *[g[c][ok] for c in full]])
            adj[g.index[ok]] = g[f"amp_{roi}"][ok].to_numpy() - Xg @ beta + beta[0]
        L[f"adjfull_{roi}"] = adj
    S = L.groupby(["label", "animal", "epoch"]).mean(numeric_only=True).reset_index()
    S.to_csv(OUT / "controls_sessions.csv", index=False)
    print("\n== per animal (rule 8): 3-D reach (mm) / peak speed of bout-start ITI licks, session means\n"
          + S.pivot_table(index="animal", columns="epoch", values=["reach", "speed"]).reindex(columns=EP4, level=1)
          .round(2).to_string())
    tests = []

    def panel_row(cells, items, tag):
        """cells = [(row, col)]; computes the session dots of each item, returns [(row, col, title, ref1, S)]."""
        out = []
        for rc, (val, ttl, ref1) in zip(cells, items):
            S_ = dots_compute(S, val, rng)
            for r_ in S_["tests"]:
                tests.append({"figure": tag, **r_})
            out.append((*rc, ttl, ref1, S_))
        return out
    # figure A: kinematics + adjusted response
    A = []
    A += panel_row([(0, 0), (0, 1)], [("reach", "3-D reach (mm)", None), ("speed", "peak tongue speed", None)], "kin")
    A += panel_row([(0, 2), (0, 3)], [("z_reach", "reach, z vs animal's pre licks", 0), ("z_speed", "speed, z vs animal's pre", 0)],
                   "kin")
    row = [(1, c) for c in range(6)]
    A += panel_row(row, [(f"adj_{r}", f"{r}: post-onset, KINEMATICS-ADJUSTED", None) for r in CTRL_ROIS], "adj")
    row = [(2, c) for c in range(6)]
    A += panel_row(row, [(f"matched_{r}", f"{r}: post-onset, licks in pre reach+speed range", None) for r in CTRL_ROIS],
                   "matched")
    row = [(3, c) for c in range(6)]
    A += panel_row(row, [(f"adjfr_{r}", f"{r}: post-onset, KINEMATICS + FACE-ADJUSTED", None) for r in CTRL_ROIS], "adj_fr")
    row = [(4, c) for c in range(6)]
    A += panel_row(row, [(f"adjfull_{r}", f"{r}: post-onset, FULL TONGUE + FACE-ADJUSTED", None) for r in CTRL_ROIS],
                   "adj_full")
    save_fd("lick_iti_kinematics_control", {"panels": A, "off": [(0, 4), (0, 5)]})
    # figure B: face movement before / after the lick, and build-up adjusted for it
    for roi in CTRL_ROIS:
        adj = np.full(len(L), np.nan)
        cols = [c for c in ("z_fr_lick_pre", "z_fr_right_pre", "z_fr_left_pre") if c in L]
        for _a, g in L.groupby("animal"):
            pre = g[g.epoch == "pre"].dropna(subset=[f"early_{roi}", *cols])
            if len(pre) < 30 or not cols:
                continue
            Xp = np.column_stack([np.ones(len(pre)), *[pre[c] for c in cols]])
            beta = np.linalg.lstsq(Xp, pre[f"early_{roi}"].to_numpy(), rcond=None)[0]
            ok = g[cols].notna().all(1)
            Xg = np.column_stack([np.ones(ok.sum()), *[g[c][ok] for c in cols]])
            adj[g.index[ok]] = g[f"early_{roi}"][ok].to_numpy() - Xg @ beta + beta[0]
        L[f"earlyadj_{roi}"] = adj
    S = L.groupby(["label", "animal", "epoch"]).mean(numeric_only=True).reset_index()
    S.to_csv(OUT / "controls_sessions.csv", index=False)
    B = []
    fr_items = [(f"z_{v}_pre", f"{v} BEFORE the lick (-0.6..-0.2 s), z", 0) for v in FR_KEEP if f"z_{v}_pre" in S]
    B += panel_row([(0, c) for c in range(len(fr_items))], fr_items, "face_pre")
    fr_post = [(f"z_{v}_post", f"{v} AFTER onset (0.1..0.4 s), z", 0) for v in FR_KEEP if f"z_{v}_post" in S]
    B += panel_row([(1, c) for c in range(len(fr_post))], fr_post, "face_post")
    offB = [(0, c) for c in range(len(fr_items), 6)] + [(1, c) for c in range(len(fr_post), 6)]
    B += panel_row([(2, c) for c in range(6)], [(f"earlyadj_{r}", f"{r}: early build-up, FACE-ADJUSTED", None)
                                                for r in CTRL_ROIS], "early_adj")
    save_fd("lick_iti_face_control", {"panels": B, "off": offB})
    # figure C: trial-to-trial (within-spout) variance of the FaceRhythm factors in the cue / first-lick windows
    V = []
    for x, d in D.items():
        if "first_fr_cue" not in d or not len(d["first_rt"]):
            continue
        pos = d["first_pos"]
        for wname, key in (("cue", "first_fr_cue"), ("lick", "first_fr_lick")):
            A_ = d[key].astype(float)
            for k, v in enumerate(d["fr_names"]):
                Y = A_[:, :, k]
                if np.isnan(Y).all():
                    continue
                R = Y - np.nanmean(Y, 0, keepdims=True)
                W_ = Y.copy()
                for q in np.unique(pos):
                    m = pos == q
                    W_[m] = Y[m] - np.nanmean(Y[m], 0, keepdims=True)
                V.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "window": wname, "factor": v,
                          "var_within": float(np.nanvar(W_)), "var_total": float(np.nanvar(R)),
                          "mean": float(np.nanmean(Y))})
    V = pd.DataFrame(V)
    V.to_csv(OUT / "fr_variance_sessions.csv", index=False)
    C, offC = [], []
    for i, wname in enumerate(("cue", "lick")):
        for j, (fac, val) in enumerate((("fr_lick", "var_within"), ("fr_lick", "var_total"), ("fr_lick", "mean"),
                                        ("fr_right", "var_within"))):
            T = V[(V.window == wname) & (V.factor == fac)].copy()
            if T.empty:
                offC.append((i, j))
                continue
            T["norm"] = _norm_to_pre(T, val)
            S_ = dots_compute(T, "norm", rng)
            for r_ in S_["tests"]:
                tests.append({"figure": f"frvar_{wname}_{fac}_{val}", **r_})
            C.append((i, j, f"{wname} window: {fac} {val.replace('var_', 'variance, ')}\n(/ animal's pre mean)", 1, S_))
    save_fd("fr_lick_variance", {"panels": C, "off": offC})
    controls_plot()
    T = pd.DataFrame(tests)
    T.to_csv(OUT / "controls_tests.csv", index=False)
    print("\n== controls: change vs pre (nested bootstrap animals > sessions)\n" + T.round(3).to_string(index=False))


CTRL_FIGS = {   # figdata stem: (grid, figsize, tight rect, ylabels {(row, col): text}, suptitle)
    "lick_iti_kinematics_control": (
        (5, 6), (21, 18), (0.01, 0, 0.99, 0.92), {(k, 0): "dF/F / animal's pre amplitude" for k in (1, 2, 3, 4)},
        "NEURAL + LICK: does the larger post-onset response to self-initiated licks follow bigger / faster licks? Row 1: "
        "3-D reach and peak speed of the same bout-start ITI licks (one dot per session). Row 2: post-onset response "
        "after regressing out reach + speed (per-animal fit on PRE licks; residual + pre intercept). Row 3: only post "
        "licks inside the animal's pre 10-90 % reach AND speed range. Row 4: as row 2 with post-onset face movement (FaceRhythm lick, right, left factors, 0.1..0.4 s) added to the regression. Row 5: FULL tongue kinematics (3-D lateral / outward / height, reach, peak speed, lick duration, retraction speed) + the same face factors. Jaw kinematics are not in the per-lick tables yet. * = nested-bootstrap change from pre."),
    "lick_iti_face_control": (
        (3, 6), (21, 11), (0.01, 0, 0.99, 0.92), {(2, 0): "/ animal's pre amplitude"},
        "NEURAL + FACE: face movement (FaceRhythm master factors, z vs each animal's pre licks) before and after "
        "self-initiated licks, one dot per session; row 3: the early build-up after regressing out pre-lick face "
        "movement (fr_lick, fr_right, fr_left; per-animal fit on PRE licks). * = nested-bootstrap change from pre."),
    "fr_lick_variance": (
        (2, 4), (16, 8), (0.01, 0, 0.99, 0.91), {},
        "FACE: trial-to-trial variability of the FaceRhythm lick factor in the encoder's cue (0..1.5 s) and first-lick "
        "(-0.1..1.0 s) windows, per session, normalised to each animal's pre mean. var_within = variance after "
        "removing each spout's mean time course (what the trial-by-trial R2 of Fig 1.16 sees); var_total = all "
        "trials. * = nested-bootstrap change from pre."),
}


def controls_plot(only: tuple = tuple(CTRL_FIGS)) -> None:
    """Draw the three control figures from their figdata files alone (no session load, no regression, no bootstrap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figdata import load_fd

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    for stem in only:
        grid, size, rect, ylabs, title = CTRL_FIGS[stem]
        F = load_fd(stem)
        fig, axs = plt.subplots(*grid, figsize=size)
        for i, j, ttl, ref1, S_ in F["panels"]:
            ax = axs[i, j]
            if ref1 is not None:
                ax.axhline(ref1, color="0.7", lw=0.8, ls="--")
            dots_draw(ax, S_, colors)
            ax.set_title(ttl, fontsize=8.5)
        for i, j in F["off"]:
            axs[i, j].axis("off")
        for (i, j), t in ylabs.items():
            axs[i, j].set_ylabel(t)
        fig.suptitle(title, fontsize=9.5, wrap=True)
        fig.tight_layout(rect=rect)
        fig.savefig(OUT / "figures" / f"{stem}.png", dpi=110)
        plt.close(fig)


# ------------------------------------------------------------------------------------------------ BINNED decoder (Priya 10-26)
# "a decoder accuracy version that uses binned time dynamics (sweep the best one, e.g. -1..1 s in 0.25 s bins) for a
# single-number decoder comparison across sessions"; feature sets: all areas (main), orofacial sensorimotor, and a
# visual / retrosplenial comparison set (not a null). The configuration (window x bin width) is chosen per lick type x
# feature set by PRE-stroke leave-one-session-out r only, then applied unchanged to every session.
BIN_WINDOWS = ((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.0), (0.0, 0.5), (-0.25, 0.25), (0.0, 0.25))
BIN_WIDTHS = (0.1, 0.25, 0.5)
OROFACIAL = ("ALM", "tjM1", "MOs", "MOp", "SSp-m", "SSp-n", "SSp-un", "SSs", "SSp-bfd")


def feature_sets(cols) -> dict:
    cols = list(cols)
    allf = [i for i, c in enumerate(cols) if c != "cortex" and not c.startswith(("ALM", "tjM1"))]
    oro = [i for i, c in enumerate(cols) if c.rsplit("_", 1)[0] in OROFACIAL]
    vis = [i for i, c in enumerate(cols) if c.startswith(("VIS", "RSP"))]
    return {"all areas": allf, "orofacial": oro, "visual + RSP": vis}


def _binned(X: np.ndarray, win, bw) -> np.ndarray | None:
    """(n, lags, areas) -> (n, bins * areas): mean activity in consecutive bins of width bw across win."""
    edges = np.round(np.arange(win[0], win[1] + 1e-9, bw), 3)
    if len(edges) < 2:
        return None
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (LAGS_D >= a - 1e-9) & (LAGS_D <= b + 1e-9)
        if not m.any():
            return None
        out.append(X[:, m, :].mean(1))
    return np.concatenate(out, 1)


def binned_frozen(D: dict, set_name: str) -> pd.DataFrame:
    """Frozen binned decoder r per session for every (feature set, window, bin width); pre sessions leave-one-out."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    kx, ky, kp = ("iti_X", "iti_side", "iti_pos") if set_name == "iti" else ("tr_X", "tr_side", "tr_pos")
    rows = []
    for a in AN:
        S = {x: d for x, d in D.items() if str(d["animal"]) == a and len(d[ky]) >= 20}
        pre = [x for x, d in S.items() if str(d["epoch"]) == "pre"]
        if len(pre) < 3:
            continue
        fsets = feature_sets(next(iter(S.values()))["cols"])
        for fname, feat in fsets.items():
            if not feat:
                continue
            Xs = {x: S[x][kx].astype(float)[:, :, feat] for x in S}
            for win in BIN_WINDOWS:
                for bw in BIN_WIDTHS:
                    if bw > (win[1] - win[0]) + 1e-9:
                        continue
                    F = {x: _binned(Xs[x], win, bw) for x in S}
                    if any(v is None for v in F.values()):
                        continue
                    for tg in ("direction", "relative"):
                        Y = {x: (S[x][ky] if tg == "direction" else _within(S[x][ky], S[x][kp])) for x in S}
                        fits = {}
                        for held in [*pre, None]:
                            trn = [q for q in pre if q != held]
                            Xt = np.vstack([F[q] for q in trn])
                            sc = StandardScaler().fit(Xt)
                            fits[held] = (sc, RidgeCV(alphas=np.logspace(0, 5, 11)).fit(
                                sc.transform(Xt), np.concatenate([Y[q] for q in trn])))
                        for x in S:
                            sc, m = fits[x if x in pre else None]
                            pr = m.predict(sc.transform(F[x]))
                            rows.append({"label": x, "animal": a, "epoch": str(S[x]["epoch"]), "set": set_name,
                                         "features": fname, "window": f"{win[0]:g}..{win[1]:g}", "bin": bw, "target": tg,
                                         "r": float(np.corrcoef(pr, Y[x])[0, 1]) if np.std(pr) > 0 else np.nan})
    return pd.DataFrame(rows)


def binned_figures(D: dict) -> None:
    """Binned frozen decoder: fit every configuration, select on PRE, test; save figdata lick_binned_{sweep,summary}.pkl,
    then draw (`binned_plot` redraws them from the figdata alone)."""
    from figdata import save_fd
    B = pd.concat([binned_frozen(D, "iti"), binned_frozen(D, "trial")])
    B.to_csv(OUT / "decode_binned_sessions.csv", index=False)
    B["config"] = B.window + " s / " + B.bin.map("{:g}".format) + " s bins"
    # selection on PRE only: mean over animals of each animal's mean pre (LOSO) r
    pre = B[B.epoch == "pre"].groupby(["set", "features", "target", "config", "animal"]).r.mean() \
        .groupby(["set", "features", "target", "config"]).mean().rename("pre_r").reset_index()
    best = pre.loc[pre.groupby(["set", "features", "target"]).pre_r.idxmax()]
    best.to_csv(OUT / "decode_binned_selected.csv", index=False)
    print("\n== binned decoder: configuration selected on PRE leave-one-session-out r\n" + best.round(3).to_string(index=False))
    # sweep figure: pre r and chronic normalised r per config
    norm = B.copy()
    pm = norm[norm.epoch == "pre"].groupby(["set", "features", "target", "config", "animal"]).r.mean()
    norm["r_norm"] = norm.r / [pm.get(k, np.nan) for k in zip(norm.set, norm.features, norm.target, norm.config, norm.animal)]
    chron = norm[norm.epoch == "chronic"].groupby(["set", "features", "target", "config", "animal"]).r_norm.mean() \
        .groupby(["set", "features", "target", "config"]).mean().rename("chronic_norm").reset_index()
    cfgs = sorted(B.config.unique(), key=lambda c: (c.split(" ")[0], float(c.split("/ ")[1].split(" ")[0])))
    rowsk = [(st, f) for st in ("iti", "trial") for f in ("all areas", "orofacial", "visual + RSP")]
    mats = []
    for T, val in ((pre, "pre_r"), (chron, "chronic_norm")):
        M = np.full((len(rowsk), len(cfgs)), np.nan)
        for i, (st, f) in enumerate(rowsk):
            t = T[(T.set == st) & (T.features == f) & (T.target == "direction")].set_index("config")[val]
            M[i] = [t.get(c, np.nan) for c in cfgs]
        mats.append(M)
    boxes = [(cfgs.index(r_.config), rowsk.index((r_.set, r_.features))) for _i, r_ in best[best.target == "direction"].iterrows()
             if (r_.set, r_.features) in rowsk]
    save_fd("lick_binned_sweep", {"mats": mats, "cfgs": cfgs, "rowsk": rowsk, "boxes": boxes})
    # summary at the selected configuration: one dot per session, raw and normalised
    rng = np.random.default_rng(3)
    tests, panels = [], {}
    for j, (st, f) in enumerate(rowsk):
        sel = best[(best.set == st) & (best.features == f) & (best.target == "direction")]
        if sel.empty:
            panels[j] = None
            continue
        cfg = sel.config.iloc[0]
        T = norm[(norm.set == st) & (norm.features == f) & (norm.target == "direction") & (norm.config == cfg)]
        T = T.groupby(["label", "animal", "epoch"])[["r", "r_norm"]].mean().reset_index()
        panels[j] = (st, f, cfg, [dots_compute(T, val, rng) for val in ("r", "r_norm")])
        for S_ in panels[j][3]:
            for r_ in S_["tests"]:
                tests.append({"set": st, "features": f, "config": cfg, **r_})
    pd.DataFrame(tests).to_csv(OUT / "decode_binned_tests.csv", index=False)
    save_fd("lick_binned_summary", {"panels": panels})
    binned_plot()
    print("\n== binned decoder at the selected configuration: change vs pre\n" + pd.DataFrame(tests).round(3).to_string(index=False))


def binned_plot() -> None:
    """Draw lick_binned_sweep / lick_binned_summary from their figdata files alone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figdata import load_fd

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    F = load_fd("lick_binned_sweep")
    cfgs, rowsk = F["cfgs"], F["rowsk"]
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    for ax, M, (val, ttl, cmap) in zip(axs, F["mats"], (("pre_r", "pre-stroke r (leave-one-session-out)", "viridis"),
                                                       ("chronic_norm", "chronic r / animal's pre (frozen)", "RdBu_r"))):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=(0 if val == "pre_r" else 0.5), vmax=(np.nanmax(M) if val == "pre_r" else 1.5))
        ax.set_yticks(range(len(rowsk)))
        ax.set_yticklabels([f"{'ITI' if st == 'iti' else 'in-trial'} | {f}" for st, f in rowsk], fontsize=8)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=60, ha="right", fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="w" if val == "pre_r" and M[i, j] < 0.5 * np.nanmax(M) else "k")
        for cx, ry in F["boxes"]:
            ax.add_patch(plt.Rectangle((cx - 0.5, ry - 0.5), 1, 1, fill=False, ec="k", lw=2))
        ax.set_title(ttl, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("NEURAL: " + FROZEN_TAG + ", BINNED (lateral tongue-tip position; r = decoded vs actual) -- sweep of window x bin width (boxed = configuration selected "
                 "on PRE r only). Left: pre-stroke leave-one-session-out r; right: chronic r normalised to each animal's pre.",
                 fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "figures" / "lick_binned_sweep.png", dpi=110)
    plt.close(fig)
    F = load_fd("lick_binned_summary")
    fig, axs = plt.subplots(2, 6, figsize=(22, 8))
    for j in range(6):
        p = F["panels"].get(j)
        if p is None:
            axs[0, j].axis("off"); axs[1, j].axis("off")
            continue
        st, f, cfg, Ss = p
        for i, ylab in enumerate(("FROZEN binned decoder: " + YR, "same r / animal's pre mean (pre = 1)")):
            ax = axs[i, j]
            if i == 1:
                ax.axhline(1, color="0.7", lw=0.8, ls="--")
            dots_draw(ax, Ss[i], colors)
            if i == 0:
                ax.set_title(f"{'ITI' if st == 'iti' else 'in-trial'} | {f}\n{cfg}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(ylab)
    fig.suptitle("NEURAL: " + FROZEN_TAG + ", BINNED, at the configuration selected on pre data; y = r between decoded and "
                 "actual lateral tongue-tip position.\n(One number per session; dots = sessions, all animals pooled; black = mean +- SEM). * = nested-bootstrap change "
                 "from pre (animals > sessions) excluding 0.", fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0.01, 0, 0.99, 0.91))
    fig.savefig(OUT / "figures" / "lick_binned_summary.png", dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--rois-only", action="store_true")
    ap.add_argument("--key", action="store_true", help="only the KEY figures, from saved sessions + ramp_sessions.csv")
    ap.add_argument("--controls", action="store_true", help="only the kinematics / face / FR-variance controls")
    ap.add_argument("--binned", action="store_true", help="only the binned frozen-decoder sweep + summary")
    ap.add_argument("--plot-direction", action="store_true", help="redraw page 3.7 from the saved CSVs")
    ap.add_argument("--plot-frozen-tc", action="store_true", help="redraw page 3.8 from decode_frozen_lags_sessions.csv")
    ap.add_argument("--ramp", action="store_true", help="redraw pages 3.16 / 3.19 (ITI-lick PETHs)")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw from figdata/<figure>.pkl ONLY (no session load, no fit, no bootstrap): every figure of "
                         "this script, or only the family of --rois-only / --ramp / --plot-direction / --plot-frozen-tc / "
                         "--key / --controls / --binned when one is given")
    a = ap.parse_args(argv)
    if a.plot_only:
        fam = {"rois": a.rois_only, "ramp": a.ramp, "dir": a.plot_direction, "tc": a.plot_frozen_tc, "key": a.key,
               "ctrl": a.controls, "bin": a.binned}
        alls = not any(fam.values())
        if alls or fam["rois"]:
            roi_figure(None, plot_only=True)
        if alls or fam["ramp"]:
            ramp_figures(None, None, plot_only=True)
        if alls or fam["dir"]:
            from figdata import load_fd
            direction_draw(load_fd("lick_direction_decoding"))
        if alls or fam["key"] or fam["tc"]:
            key_figures_plot(KEY_STEMS if (alls or fam["key"]) else ("lick_frozen_timecourse",))
        if alls or fam["ctrl"]:
            controls_plot()
        if alls or fam["bin"]:
            binned_plot()
        return 0
    assert_writable(OUT)
    (OUT / SESS_NAME).mkdir(parents=True, exist_ok=True)
    run = E / "frozen_encoder_gate-attempted_v7g3_t3d_frA" / "joint" / "sessions"
    labels = sorted(a.labels or (p.stem for p in run.glob("*.json")))
    labels = [x for x in labels if x[-4:] >= "0801"]
    if a.rois_only:
        roi_figure(labels[0])
        return 0
    if a.ramp:
        ramp_figures({x: _load(x) for x in labels if (OUT / SESS_NAME / f"{x}.npz").exists()},
                     pd.read_csv(OUT / "ramp_sessions.csv"))
        return 0
    if a.plot_direction:
        plot_direction(pd.read_csv(OUT / "decode_sessions.csv"), pd.read_csv(OUT / "decode_frozen_sessions.csv"))
        return 0
    if a.plot_frozen_tc:
        plot_frozen_timecourse(pd.read_csv(OUT / "decode_frozen_lags_sessions.csv"))
        return 0
    if a.binned:
        binned_figures({x: _load(x) for x in labels if (OUT / SESS_NAME / f"{x}.npz").exists()})
        return 0
    if a.controls:
        D = {x: _load(x) for x in labels if (OUT / SESS_NAME / f"{x}.npz").exists()}
        controls(D, pd.read_csv(OUT / "ramp_sessions.csv"))
        return 0
    if a.key:
        D = {x: _load(x) for x in labels if (OUT / SESS_NAME / f"{x}.npz").exists()}
        key_figures(D, pd.read_csv(OUT / "ramp_sessions.csv"))
        return 0
    if not a.summarise:
        roi_figure(labels[0])
        todo = [(x, str(OUT)) for x in labels if not (OUT / SESS_NAME / f"{x}.npz").exists()]
        print(f"[iti_licks] {len(labels)} sessions, {len(todo)} to extract -> {OUT}", flush=True)
        got, fail = ak.fan_sessions(todo, extract, jobs=a.jobs, label="extract")
        for it, r in got:
            if not r.get("cached"):
                print(f"  {it[0]}: bout {r['n_bout']}, iti {r['n_iti']}, first {r['n_first']}, clock {r['clock_med_s']:.3f} s",
                      flush=True)
        for it, err in fail:
            print(f"  !! {it[0]}: {err}", flush=True)
    summarise(labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
