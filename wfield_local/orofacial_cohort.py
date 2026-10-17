"""Pre- vs post-stroke orofacial kinematics across the cohort, from the organised pose outputs (`orofacial_outputs`).

    python -m wfield_local.orofacial_cohort [--run lp_r4_heatmap_20261013__clean_v35] [--jobs 4]

Reads every CURATED session (`analysis_kit.curated_sessions`, the same pre + post set the imaging analyses use -- early
June 6/1-6/5 and 8/5 excluded) that has a processed session folder, reduces each to per-session numbers, and writes

    <LP_analysis>/<run>/cohort/tables/prepost/  session_metrics.csv     one row per session (animal, epoch, day, metrics)
                                                position_metrics.csv    one row per session x spout position
                                                lick_phase.csv          lick-peak-aligned mean protrusion / velocity per
                                                                        session x position (measured-peak licks)
                                                epoch_by_animal.csv     per-animal epoch mean + SEM over sessions (rule 8)
                                                epoch_deltas.csv        paired animal-weighted change vs pre (`boot_delta`)
    <LP_analysis>/<run>/cohort/figures/prepost/ epochs, timecourse (session level), epochs_by_position_<group>,
                                                timecourse_by_position_<group> (group = engagement / kinematics /
                                                accuracy), by_position, angle_tuning, lick_phase, phase_portrait,
                                                mismatch (.png)

Re-run as sessions arrive; every number is recomputed from the session folders (nothing cached here).

POSITIONS are named and ordered by the lesion: close ipsi / close mid / close contra, far ipsi / far mid / far contra
(Priya, 2026-10-16). ipsi / contra come from `animals.yaml stroke_laterality` (all four lesions are LEFT: L = ipsi,
R = contra; PS93 also has a pre-existing right orofacial deficit). Colours follow the widefield figures: positions
`spout_behavior.position_style` (hue = side, lightness = ring), epochs the grey ramp `epoch_figures.EPOCH_GREY`, animals
`config.animal_color`. Error bars are SEM over SESSIONS within an animal (none where an animal has one session).

WHAT EACH METRIC IS (per session; post-cue = cue + 70 ms .. DAQ trial stop, `licks.csv`):
  resp_rate          fraction of trials with >= 1 post-cue lick (engagement as much as ability)
  licks_per_resp     post-cue licks per RESPONDED trial
  post_rate_hz       post-cue licks / summed response-window time
  latency_ms         median time of the first post-cue LP lick's PEAK (comparable to a spout contact)
  latency_contact_ms median time of the first post-cue DAQ spout contact (video-independent)
  ili_ms             median inter-lick interval inside a trial, intervals < 300 ms (within-bout rhythm)
  neighbor_frac      fraction of licks with another lick in the same trial within 250 ms (bout continuation vs
                     isolated licks -- the flanks of the lick-phase plot)
  lick_dur_ms        median lick duration (onset .. offset)
  protrusion_px      median per-lick max protrusion, MEASURED peaks only (`peak_interp` False; Priya 10-15: be wary
                     of quantities read off interpolated points). Compare WITHIN a position: the spouts sit at
                     different image distances from the mouth (far mid ~109 px vs far sides ~132 px, PS92)
  reach_frac         protrusion / mouth-spout distance -- comparable ACROSS positions (> 1 = past the spout)
  vel_protrude       median peak protrusion speed (px/s), measured peaks
  vel_retract        median peak retraction speed (px/s), measured peaks
  contact_frac       fraction of post-cue licks with a DAQ spout contact
  nocontact_frac     1 - contact_frac: the tongue came out and missed (tongue-side mismatch)
  jaw_no_lick_frac   fraction of jaw-QC'd trials with a jaw movement but no tongue lick (jaw-side mismatch,
                     `tongue_jaw_mismatch.csv candidate_no_lick_with_jaw_move`)
  angle_gain_far     executed-angle vs target from the far side positions: (angle far L - far R) / (spout far L - far R)
  ipsi_bias_deg      mean over positions of the median executed peak angle, signed + = toward the IPSILESIONAL side --
                     a target-independent turn of the tongue
  precue_rate_hz     spontaneous licks, position strobe (spout arrived) .. cue + 70 ms, per second (`licks_precue.csv`)
  precue_trial_frac  fraction of trials with >= 1 pre-cue lick
  iti_rate_hz        inter-trial licks per second (`licks_iti.csv`)
  peak_interp_frac   fraction of post-cue licks whose peak is interpolated / imputed (tracking-quality context)
At n=4, the per-animal table is printed before any bootstrap (CLAUDE.md rule 8).
"""
from __future__ import annotations

import argparse
import os
import textwrap
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN = "lp_r4_heatmap_20261013__clean_v35"
EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")
POS_ORDER = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
METRICS = ("resp_rate", "licks_per_resp", "post_rate_hz", "precue_rate_hz", "precue_trial_frac", "iti_rate_hz",
           "latency_ms", "latency_contact_ms", "ili_ms", "neighbor_frac", "lick_dur_ms",
           "protrusion_px", "reach_frac", "vel_protrude", "vel_retract",
           "contact_frac", "nocontact_frac", "jaw_no_lick_frac", "angle_gain_far", "ipsi_bias_deg", "peak_interp_frac")
LABEL = {"resp_rate": "responded trials (frac)", "licks_per_resp": "licks / responded trial",
         "post_rate_hz": "post-cue lick rate (Hz)", "precue_rate_hz": "pre-cue lick rate (Hz)",
         "precue_trial_frac": "trials with pre-cue lick (frac)", "iti_rate_hz": "ITI lick rate (Hz)",
         "latency_ms": "first LP lick peak (ms from cue)", "latency_contact_ms": "first spout contact (ms from cue)",
         "ili_ms": "within-bout ILI (ms)", "lick_dur_ms": "lick duration (ms)",
         "neighbor_frac": "licks with a neighbour < 250 ms (frac)",
         "protrusion_px": "max protrusion, measured (px)", "reach_frac": "reach (protrusion / spout distance)",
         "vel_protrude": "peak protrusion speed (px/s)", "vel_retract": "peak retraction speed (px/s)",
         "contact_frac": "licks with spout contact (frac)", "nocontact_frac": "no-contact licks (frac)",
         "jaw_no_lick_frac": "jaw move, no lick (frac trials)", "angle_gain_far": "angle gain, far ipsi vs contra",
         "ipsi_bias_deg": "tongue angle bias, + = ipsi (deg)", "peak_interp_frac": "interpolated peaks (frac)"}
LABEL["peak_angle_deg"] = "executed peak angle (deg; + = L / image right)"
LABEL["delta_angle_deg"] = "peak angle - spout angle (deg)"
LABEL["walk_d_angle"] = "lick 3 - lick 1 angle (deg; + = toward L)"
LABEL["walk_err_reduction"] = "|error| lick 1 - |error| lick 3 (deg; + = closer)"
LABEL["walk_n_trials"] = "trials with licks 1 & 3 measured"
for _c, _l in (("l1_angle_deg", "lick 1 angle"), ("l2_angle_deg", "lick 2 angle"), ("l1_prot_px", "lick 1 protrusion"),
               ("l2_prot_px", "lick 2 protrusion")):
    LABEL[f"{_c}_dpre"] = f"{_l}, change vs pre" + (" (deg; + = L)" if "angle" in _c else " (px)")


def delta_from_pre(Pm: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    """Adds ``<col>_dpre`` = session value minus that animal's mean PRE-stroke value at the same position
    (stroke_orofacial Family 8.1's per-animal, side-matched baseline subtraction). Pre sessions keep their spread
    around 0 -- the noise floor every post epoch is read against."""
    out = Pm.copy()
    for c in cols:
        base = out[out.epoch == "pre"].groupby(["animal", "pos"])[c].mean()
        out[f"{c}_dpre"] = out[c] - base.reindex(pd.MultiIndex.from_frame(out[["animal", "pos"]])).to_numpy()
    return out
#: per-position figures, one file per group (rows = metric, columns = the six positions)
POS_GROUPS = {"engagement": ("resp_rate", "licks_per_resp", "precue_rate_hz", "ili_ms", "neighbor_frac",
                             "latency_ms", "latency_contact_ms"),
              "kinematics": ("protrusion_px", "reach_frac", "vel_protrude", "vel_retract", "lick_dur_ms"),
              "accuracy": ("contact_frac", "nocontact_frac", "jaw_no_lick_frac", "peak_angle_deg")}
#: epoch colours for this module's line plots (angle tuning, by position, lick phase): the grey ramp's light "pre"
#: disappears on them (Priya 10-16: "ok to set another convention there"). Animal colours stay `config.animal_color`.
#: Chosen to collide with NO animal colour (PS92 blue, PS93 red, PS94 orange, PS95 green -- Priya 10-16: "confusing for
#: animals colors to be the same as the pre/acute/subacute/chronic colors").
EPOCH_LINE = {"pre": "k", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}
ILI_MAX_MS = 300.0
FRAME_MS = 4.0
PHASE_HALF = 38                       # frames each side of the lick peak (+-152 ms)
FPS = 250.0
PORTRAIT_HALF_MS = 72.0
ERR_T_MS = (-48.0, -24.0, 0.0, 24.0, 48.0)   # where 2-D SEM crosses are drawn on trajectories / phase portraits               # phase portrait: about one lick cycle (median ILI ~150 ms)


def pos_key(position: str, laterality: str) -> str:
    """`close_L` -> 'close_ipsi' / 'close_mid' / 'close_contra', relative to the lesion side."""
    ring, s = position.split("_", 1)
    side = "mid" if s == "center" else ("ipsi" if s == laterality else "contra")
    return f"{ring}_{side}"


def pos_disp(key: str) -> str:
    return key.replace("_", " ")


def raw_position(key: str, laterality: str) -> str:
    """Inverse of `pos_key` -- for the position palette, which is keyed by the rig's names."""
    ring, side = key.split("_")
    if side == "mid":
        return f"{ring}_center"
    other = "R" if laterality == "L" else "L"
    return f"{ring}_{laterality if side == 'ipsi' else other}"


def _windows(sd: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-trial durations: pre-cue, response, ITI seconds; per-trial first row + first t_ms, for frame lookups)."""
    w = np.load(sd / "windows.npz")
    ix = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "t_ms", "strobe_ms", "stop_ms")})
    ix["row"] = np.arange(len(ix))
    g = ix.groupby("trial_id", sort=True).agg(t0=("t_ms", "min"), t1=("t_ms", "max"), strobe=("strobe_ms", "first"),
                                               stop=("stop_ms", "first"), row0=("row", "min"))
    t1 = g.t1 + FRAME_MS
    pre = (70.0 - g.strobe).where(np.isfinite(g.strobe), np.nan).clip(lower=0)
    resp = (g.stop.clip(upper=t1) - 70.0).clip(lower=0)
    iti = (g.strobe - g.t0).clip(lower=0).fillna(0) + (t1 - g.stop).clip(lower=0).fillna(0)
    return pd.DataFrame({"pre_s": pre / 1000, "resp_s": resp / 1000, "iti_s": iti / 1000}), g[["row0", "t0"]]


def _first_contacts(animal: str, date: str) -> pd.Series:
    """{trial_id: first DAQ spout-contact time, ms from the cue, inside cue + 70 ms .. trial stop}."""
    from scripts import pose_kinematics_demo as D
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    lick = np.sort(D.daq_lick_onsets_s(animal, date, rv))
    b = TW.trial_bounds(animal, date, rv)
    out = {}
    for t, c, s in zip(b.trial_id, b.cue_s, b.stop_s):
        if np.isfinite(c):
            k = np.searchsorted(lick, c + 0.07)
            if k < len(lick) and (not np.isfinite(s) or lick[k] <= s):
                out[t] = (lick[k] - c) * 1000.0
    return pd.Series(out, dtype=float)


#: below this protrusion the tongue direction is not defined (tip at the lips / mouth fill = 0 px): angle = NaN
ANGLE_MIN_PX = 20.0


def _lick_phase(sd: Path, meas: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """{position: (mean protrusion, mean velocity, mean angle, n licks)} around each measured lick PEAK (+-152 ms).
    Protrusion = distance of the cleaned tongue from the mouth, which reproduces `protrusion_px` at the peak frame;
    angle = `tongue_kinematics.compute_signed_tongue_angle_deg` (+ = image right = the L positions), NaN while the tip
    is within ANGLE_MIN_PX of the mouth."""
    from wfield_local import tongue_kinematics as tk
    c = np.load(sd / "clean.npz")
    tx, ty = c["tongue_x"].astype(float), c["tongue_y"].astype(float)
    dist = np.hypot(tx, ty)
    o = c["tongue_origin"]
    ang = tk.compute_signed_tongue_angle_deg(tx + o[0], ty + o[1],
                                             tk.SpoutFrame(tuple(c["spout_origin"]), tuple(c["spout_ap_axis"])))
    rad = np.deg2rad(ang)
    lat_xy, out_xy = dist * np.sin(rad), dist * np.cos(rad)       # tip in the mouth frame, every tracked frame
    out_m = dist >= ANGLE_MIN_PX                                   # tongue OUT (in-mouth frames excluded)
    ang_m0 = np.where(np.isfinite(dist), np.where(out_m, ang, 0.0), np.nan)     # in-mouth frames counted as 0 deg
    ang = np.where(out_m, ang, np.nan)
    prot_out, lat_out = np.where(out_m, dist, np.nan), np.where(out_m, lat_xy, np.nan)
    outxy_out = np.where(out_m, out_xy, np.nan)
    n = len(dist)
    r = (rows.row0.reindex(meas.trial_id).to_numpy() +
         np.round((meas.t_ms.to_numpy() - rows.t0.reindex(meas.trial_id).to_numpy()) / FRAME_MS)).astype(int)
    off = np.arange(-PHASE_HALF, PHASE_HALF + 1)
    ok = (r - PHASE_HALF >= 0) & (r + PHASE_HALF < n)
    out = {}
    li = meas.lick_idx.to_numpy()
    sets = {"postcue": np.ones(len(meas), bool), "first": li == 0, "second": li == 1}
    for (st, sm), pos in ((x, q) for x in sets.items() for q in meas.position.unique()):
        m = ok & sm & (meas.position.to_numpy() == pos)
        if m.sum() < 5:
            continue
        idx = r[m][:, None] + off[None, :]
        seg = dist[idx]
        vel = np.gradient(seg, axis=1) * FPS
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[(st, pos)] = ({"protrusion_px": np.nanmean(seg, 0), "velocity_px_s": np.nanmean(vel, 0),
                                   "speed_abs_px_s": np.nanmean(np.abs(vel), 0), "angle_deg": np.nanmean(ang[idx], 0),
                                   "angle_mouth0_deg": np.nanmean(ang_m0[idx], 0), "lat_px": np.nanmean(lat_xy[idx], 0),
                                   "out_px": np.nanmean(out_xy[idx], 0), "protrusion_out_px": np.nanmean(prot_out[idx], 0),
                                   "lat_out_px": np.nanmean(lat_out[idx], 0),
                                   "out_out_px": np.nanmean(outxy_out[idx], 0),
                                   "velocity_out_px_s": np.nanmean(np.where(out_m[idx], vel, np.nan), 0)},
                                  int(m.sum()))
    return out


NEIGHBOR_MS = 250.0
#: stroke_orofacial's lick-phase normalisation (`tongue_pertrial_features._assign_phase_peak_to_peak_shifted`): inside a
#: bout of >= 2 licks, each peak-to-peak interval maps to phase 0.5 (peak) -> 1 == 0 (midway) -> 0.5 (next peak);
#: N_PHASE_BINS bins, a bin needs MIN_PHASE_POINTS samples, light circular smoothing (1 bin) when plotted.
N_PHASE_BINS = 72
MIN_PHASE_POINTS = 5                   # INTERVALS contributing to a bin (each interval is interpolated onto every bin)
#: interval fraction u (0 = first peak, 1 = next peak) of each phase-bin centre: phase = (u - 0.5) mod 1
U_CENTERS = np.mod((np.arange(N_PHASE_BINS) + 0.5) / N_PHASE_BINS + 0.5, 1.0)
PHASE_MODES = ("first_multi", "all_multi")      # first multi-lick bout of each trial (their default) / every one
#: lick sets for the per-lick metrics (Priya 10-16: "first lick vs all post-cue licks in trial period vs all licks")
LICK_SETS = {"first": "first post-cue lick of each trial", "postcue": "all post-cue licks (cue + 70 ms .. stop)",
             "all": "all licks: pre-cue + post-cue + ITI"}
#: lick-by-lick "walk" toward the target (Priya 10-16): per-lick-number medians, lick 1 .. WALK_MAX within a trial
WALK_MAX = 8
WALK_MIN_LICKS = 3
LICKSET_METRICS = ("protrusion_px", "peak_angle_deg", "delta_angle_deg", "vel_protrude", "vel_retract")


def _neighbor_frac(g: pd.DataFrame) -> float:
    """Fraction of licks with another lick of the same trial within NEIGHBOR_MS (either side)."""
    if not len(g):
        return np.nan
    o = g.sort_values(["trial_id", "t_ms"])
    d = o.groupby("trial_id").t_ms.diff()
    prev_ok = d < NEIGHBOR_MS
    next_ok = (-o.groupby("trial_id").t_ms.diff(-1)) < NEIGHBOR_MS
    return float((prev_ok | next_ok).mean())


def _phase_norm(sd: Path, L: pd.DataFrame, rows: pd.DataFrame, spout_angle: dict) -> list[dict]:
    """Phase-normalised angle / angle error / protrusion per position and mode (stroke_orofacial's method, see
    N_PHASE_BINS). Uses peak-to-peak intervals of >= 2-lick post-cue bouts whose BOTH peaks are measured (theirs: all
    detected peaks -- here an imputed apex reads as a dip at the peak); angle is NaN within ANGLE_MIN_PX of the mouth."""
    from wfield_local import tongue_kinematics as tk
    c = np.load(sd / "clean.npz")
    tx, ty = c["tongue_x"].astype(float), c["tongue_y"].astype(float)
    dist = np.hypot(tx, ty)
    o = c["tongue_origin"]
    ang = tk.compute_signed_tongue_angle_deg(tx + o[0], ty + o[1],
                                             tk.SpoutFrame(tuple(c["spout_origin"]), tuple(c["spout_ap_axis"])))
    ang_mouth0 = np.where(np.isfinite(dist), np.nan_to_num(ang, nan=0.0), np.nan)   # in-mouth frames count as 0
    ang = np.where(dist >= ANGLE_MIN_PX, ang, np.nan)
    B = L[L.multi_bout_rank.notna()].copy()
    if not len(B):
        return []
    B["r"] = (rows.row0.reindex(B.trial_id).to_numpy() +
              np.round((B.t_ms.to_numpy() - rows.t0.reindex(B.trial_id).to_numpy()) / FRAME_MS)).astype(int)
    out = []
    for mode in PHASE_MODES:
        Bm = B if mode == "all_multi" else B[B.multi_bout_rank == 0]
        for pos, gp in Bm.groupby("position"):
            sums = {k: np.zeros(N_PHASE_BINS) for k in ("angle", "prot", "angle0", "protout")}
            cnt = {k: np.zeros(N_PHASE_BINS) for k in ("angle", "prot", "angle0", "protout")}
            n_pk = 0
            for _, gb in gp.groupby(["trial_id", "bout_idx"]):
                gb = gb.sort_values("r")
                r, imp = gb.r.to_numpy(), gb.peak_interp.astype(bool).to_numpy()
                n_pk += int((~imp).sum())
                for k_, (i0, i1) in enumerate(zip(r[:-1], r[1:])):
                    # an interval touching an interpolated / imputed peak is skipped: the apex time is imputed but
                    # the cleaned trace there is the occluded, flattened value -> a dip at phase 0.5 (Priya 10-16)
                    if i1 <= i0 or imp[k_] or imp[k_ + 1]:
                        continue
                    # Each interval is INTERPOLATED onto the bin centres (stroke_orofacial's
                    # `densify_phase_rows_via_interp`), not binned frame by frame: a ~40-frame interval steps 1/40 of a
                    # cycle per frame, coarser than 72 bins, so the bin just before the peak was filled only by rare
                    # long (missed-lick / slow) intervals sitting in a trough -> a spurious dip at phase ~0.49 (10-16).
                    idx = np.arange(i0, i1 + 1)
                    u = (idx - i0) / float(i1 - i0)
                    for k, v in (("angle", ang[idx]), ("prot", dist[idx]), ("angle0", ang_mouth0[idx]),
                                 ("protout", np.where(dist[idx] >= ANGLE_MIN_PX, dist[idx], np.nan))):
                        vi = np.interp(U_CENTERS, u, v)                    # NaN wherever a neighbour is NaN
                        ok = np.isfinite(vi)
                        sums[k][ok] += vi[ok]
                        cnt[k][ok] += 1
            with np.errstate(all="ignore"):
                ma = np.where(cnt["angle"] >= MIN_PHASE_POINTS, sums["angle"] / cnt["angle"], np.nan)
                mp = np.where(cnt["prot"] >= MIN_PHASE_POINTS, sums["prot"] / cnt["prot"], np.nan)
                m0 = np.where(cnt["angle0"] >= MIN_PHASE_POINTS, sums["angle0"] / cnt["angle0"], np.nan)
                mpo = np.where(cnt["protout"] >= MIN_PHASE_POINTS, sums["protout"] / cnt["protout"], np.nan)
            centers = (np.arange(N_PHASE_BINS) + 0.5) / N_PHASE_BINS
            for k in range(N_PHASE_BINS):
                out.append({"position": pos, "mode": mode, "phase": centers[k], "angle_deg": ma[k],
                            "delta_angle_deg": ma[k] - spout_angle.get(pos, np.nan), "protrusion_px": mp[k],
                            "angle_mouth0_deg": m0[k], "protrusion_out_px": mpo[k],
                            "n_peaks": n_pk})
    return out


def _lickset_rows(sets: dict, spout_angle: dict, lat: str, base: dict) -> list[dict]:
    """Per lick set (first / postcue / all) x position: the per-lick metrics, measured peaks only."""
    out = []
    for name, D in sets.items():
        if not len(D):
            continue
        Dm = D[~D.peak_interp.astype(bool)] if "peak_interp" in D else D
        for pos, g in Dm.groupby("position"):
            pa = float(g.peak_angle_deg.median())
            out.append({**base, "set": name, "position": pos, "pos": pos_key(pos, lat), "n_licks": int(len(g)),
                        "protrusion_px": float(g.protrusion_max_measured_px.median()),
                        "peak_angle_deg": pa, "delta_angle_deg": pa - spout_angle.get(pos, np.nan),
                        "vel_protrude": float(g.max_velocity_y_px_per_s.median()),
                        "vel_retract": float(g.max_retract_velocity_y_px_per_s.median())})
    return out


#: Family 8.2 (stroke_orofacial `figures/ipsi_shift.py`): cue-aligned continuous tongue traces, binned
CUE_BIN_MS = 20.0
CUE_WIN_MS = (-500.0, 3000.0)
CUE_MIN_FRAMES = 25                   # frames per bin per session (sparse pre-cue bins drew spikes at 5)


def _cue_traces(sd: Path, lat_side: str, keep: set | None = None) -> list[dict]:
    """Per position: cue-aligned mean LATERAL tongue position (mouth frame, + = toward the IPSILESIONAL side; in-mouth
    frames INCLUDED at 0), mean tongue ANGLE (+ = ipsi; in-mouth frames EXCLUDED, < ANGLE_MIN_PX) and the same angle with
    in-mouth frames counted as 0 deg (Priya 10-16; stroke_orofacial's convention), in CUE_BIN_MS bins
    pooled over every frame of every trial at that position (stroke_orofacial Family 8.2 used the per-frame x)."""
    from wfield_local import tongue_kinematics as tk
    c = np.load(sd / "clean.npz")
    w = np.load(sd / "windows.npz")
    tx, ty = c["tongue_x"].astype(float), c["tongue_y"].astype(float)
    dist = np.hypot(tx, ty)
    o = c["tongue_origin"]
    ang = tk.compute_signed_tongue_angle_deg(tx + o[0], ty + o[1],
                                             tk.SpoutFrame(tuple(c["spout_origin"]), tuple(c["spout_ap_axis"])))
    sgn = 1.0 if lat_side == "L" else -1.0                # angles are + toward the L positions (image right)
    lat_xy = sgn * dist * np.sin(np.deg2rad(np.nan_to_num(ang, nan=0.0)))
    lat_xy[~np.isfinite(dist)] = np.nan
    ang_out = np.where(dist >= ANGLE_MIN_PX, sgn * ang, np.nan)
    ang_m0 = np.where(np.isfinite(dist), np.where(dist >= ANGLE_MIN_PX, sgn * ang, 0.0), np.nan)   # in-mouth = 0 deg
    t = w["ix_t_ms"].astype(float)
    pos = w["ix_position"].astype(str)
    edges = np.arange(CUE_WIN_MS[0], CUE_WIN_MS[1] + CUE_BIN_MS, CUE_BIN_MS)
    b = np.digitize(t, edges) - 1
    inwin = (b >= 0) & (b < len(edges) - 1)
    if keep is not None:                                  # engaged trials only
        inwin &= np.isin(w["ix_trial_id"].astype(int), np.fromiter(keep, int))
    out = []
    for p in np.unique(pos):
        m = inwin & (pos == p)
        for col, v in (("lat_px", lat_xy), ("angle_deg", ang_out), ("angle_mouth0_deg", ang_m0), ("protrusion_px", dist),
                       ("protrusion_out_px", np.where(dist >= ANGLE_MIN_PX, dist, np.nan)),
                       ("lat_out_px", np.where(dist >= ANGLE_MIN_PX, lat_xy, np.nan))):
            ok = m & np.isfinite(v)
            ssum = np.bincount(b[ok], weights=v[ok], minlength=len(edges) - 1)
            n = np.bincount(b[ok], minlength=len(edges) - 1)
            with np.errstate(all="ignore"):
                mu = np.where(n >= CUE_MIN_FRAMES, ssum / n, np.nan)
            for k in range(len(edges) - 1):
                out.append({"position": p, "var": col, "t_ms": edges[k] + CUE_BIN_MS / 2, "value": mu[k]})
    return out


def _repo_on_path() -> None:
    """Workers import `scripts.pose_kinematics_demo`; make that independent of the launch directory."""
    import sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)


def session_metrics(job):
    """(session row, position rows, lick-phase rows) for one session folder; module-level for `fan_sessions`."""
    _repo_on_path()
    sd, animal, date, epoch, day, lat = job
    rd = lambda n: pd.read_csv(sd / n) if (sd / n).exists() else pd.DataFrame()   # noqa: E731
    L, P, Q, T, M = rd("licks.csv"), rd("licks_precue.csv"), rd("licks_iti.csv"), rd("trials.csv"), \
        rd("tongue_jaw_mismatch.csv")
    if not len(T) or "t_ms" not in L:
        return None
    dur, rows = _windows(sd)
    # ENGAGEMENT GATE (Priya 10-17): every table below, the per-trial window durations (rate denominators) and the
    # cue-aligned frames keep ENGAGED trials only; `--all-trials` is the control
    n_tr_all = len(T)
    keep, gate = engagement_keep(animal, date, sd)
    if keep is not None:
        L, P, Q, T, M = (d[d.trial_id.astype(int).isin(keep)] if len(d) and "trial_id" in d else d
                         for d in (L, P, Q, T, M))
        dur = dur[dur.index.astype(int).isin(keep)]
        if not len(T) or not len(L):
            return None
    contact1 = _first_contacts(animal, date)
    meas = L[~L.peak_interp.astype(bool)]
    by_trial = L.groupby("trial_id")
    resp_trials = by_trial.size()
    ili = L.sort_values(["trial_id", "t_ms"]).groupby("trial_id").t_ms.diff()
    pos_med = meas.groupby("position")[["peak_angle_deg", "spout_angle_deg"]].median()
    far = pos_med.reindex([f"far_{lat}", f"far_{'R' if lat == 'L' else 'L'}"])          # ipsi, contra
    dsp = far.spout_angle_deg.iloc[0] - far.spout_angle_deg.iloc[1]
    gain_far = (float((far.peak_angle_deg.iloc[0] - far.peak_angle_deg.iloc[1]) / dsp)
                if far.notna().all().all() and abs(dsp) > 10 else np.nan)
    # angles are + toward image right = the L positions; flip so + = ipsilesional whatever the lesion side
    sgn = 1.0 if lat == "L" else -1.0
    jq = M[M.jaw_pass_qc.astype(bool)] if len(M) and "jaw_pass_qc" in M else pd.DataFrame()
    n_tr = int(len(T))

    def kin(g, gm):
        return {"protrusion_px": float(gm.protrusion_max_measured_px.median()) if len(gm) else np.nan,
                "reach_frac": float(gm.reach_frac.median()) if len(gm) else np.nan,
                "vel_protrude": float(gm.max_velocity_y_px_per_s.median()) if len(gm) else np.nan,
                "vel_retract": float(gm.max_retract_velocity_y_px_per_s.median()) if len(gm) else np.nan,
                "lick_dur_ms": float((g.off_ms - g.on_ms).median()) if len(g) else np.nan,
                "contact_frac": float(g.contact.astype(float).mean()) if len(g) else np.nan,
                "neighbor_frac": _neighbor_frac(g),
                "latency_ms": float(g.groupby("trial_id").t_ms.min().median()) if len(g) else np.nan}

    row = {"animal": animal, "date": date, "label": f"{animal}_{date[4:]}", "epoch": epoch, "day": day,
           "n_trials_all": n_tr_all, "n_trials_used": len(T), "engagement_gate": gate,
           "n_trials": n_tr, "n_licks": int(len(L)), "n_precue_licks": int(len(P)), "n_iti_licks": int(len(Q)),
           "resp_rate": len(resp_trials) / n_tr,
           "licks_per_resp": float(resp_trials.mean()) if len(resp_trials) else np.nan,
           "post_rate_hz": len(L) / dur.resp_s.sum() if dur.resp_s.sum() > 0 else np.nan,
           "precue_rate_hz": len(P) / dur.pre_s.sum() if dur.pre_s.sum() > 0 else np.nan,
           "precue_trial_frac": (P.trial_id.nunique() / n_tr) if len(P) else 0.0,
           "iti_rate_hz": len(Q) / dur.iti_s.sum() if dur.iti_s.sum() > 0 else np.nan,
           "latency_contact_ms": float(contact1.median()) if len(contact1) else np.nan,
           "ili_ms": float(ili[ili < ILI_MAX_MS].median()) if ili.notna().any() else np.nan,
           **kin(L, meas),
           "jaw_no_lick_frac": float(jq.candidate_no_lick_with_jaw_move.astype(float).mean()) if len(jq) else np.nan,
           "angle_gain_far": gain_far,
           "ipsi_bias_deg": sgn * float(pos_med.peak_angle_deg.mean()) if len(pos_med) else np.nan,
           "peak_interp_frac": float(L.peak_interp.astype(float).mean())}
    row["nocontact_frac"] = 1.0 - row["contact_frac"]
    prow = []
    tpos = T.set_index("trial_id").position
    for pos, g in L.groupby("position"):
        tids = tpos[tpos == pos].index
        jp = jq[jq.position == pos] if len(jq) else jq
        rt = g.groupby("trial_id").size()
        ili_p = g.sort_values(["trial_id", "t_ms"]).groupby("trial_id").t_ms.diff()
        pre_s = dur.pre_s.reindex(tids).sum()
        r = {"animal": animal, "date": date, "epoch": epoch, "day": day, "position": pos, "pos": pos_key(pos, lat),
             "n_trials": int(len(tids)), "n_licks": int(len(g)), **kin(g, meas[meas.position == pos]),
             "resp_rate": len(rt) / len(tids) if len(tids) else np.nan,
             "licks_per_resp": float(rt.mean()) if len(rt) else np.nan,
             "precue_rate_hz": (int(P.trial_id.isin(tids).sum()) / pre_s) if pre_s > 0 else np.nan,
             "ili_ms": float(ili_p[ili_p < ILI_MAX_MS].median()) if ili_p.notna().any() else np.nan,
             "latency_contact_ms": float(contact1.reindex(tids).median()),
             "jaw_no_lick_frac": float(jp.candidate_no_lick_with_jaw_move.astype(float).mean()) if len(jp) else np.nan,
             "peak_angle_deg": float(meas[meas.position == pos].peak_angle_deg.median()),
             "spout_angle_deg": float(meas[meas.position == pos].spout_angle_deg.median()),
             "spout_dist_px": float(g.spout_dist_px.median())}
        r["nocontact_frac"] = 1.0 - r["contact_frac"]
        gm_ = g[~g.peak_interp.astype(bool)]
        a1 = gm_[gm_.lick_idx == 0].drop_duplicates("trial_id").set_index("trial_id").peak_angle_deg
        a3 = gm_[gm_.lick_idx == 2].drop_duplicates("trial_id").set_index("trial_id").peak_angle_deg
        both = a1.index.intersection(a3.index)
        spa = r["spout_angle_deg"]
        if len(both) >= WALK_MIN_LICKS and np.isfinite(spa):
            d13 = a3[both] - a1[both]
            red = (a1[both] - spa).abs() - (a3[both] - spa).abs()
            r.update({"walk_d_angle": float(d13.median()), "walk_err_reduction": float(red.median()),
                      "walk_n_trials": int(len(both))})
        else:
            r.update({"walk_d_angle": np.nan, "walk_err_reduction": np.nan, "walk_n_trials": int(len(both))})
        tp = T[T.position == pos]
        r.update({"l1_t_ms": float(tp.lick1_t_peak_ms.median()), "l2_t_ms": float(tp.lick2_t_peak_ms.median()),
                  "l12_dt_ms": float((tp.lick2_t_peak_ms - tp.lick1_t_peak_ms).median()),
                  "l1_prot_px": float(tp.lick1_y_peak.median()), "l2_prot_px": float(tp.lick2_y_peak.median()),
                  "l1_angle_deg": float(tp.lick1_angle_at_ypeak.median()),
                  "l2_angle_deg": float(tp.lick2_angle_at_ypeak.median())})
        prow.append(r)
    t = np.arange(-PHASE_HALF, PHASE_HALF + 1) * FRAME_MS
    sp = {r["position"]: r["spout_angle_deg"] for r in prow}
    ph = [{"animal": animal, "date": date, "epoch": epoch, "set": st, "position": pos, "pos": pos_key(pos, lat),
           "t_ms": tt, **{k: float(v[i]) for k, v in dd.items()},
           "delta_angle_deg": float(dd["angle_deg"][i]) - sp.get(pos, np.nan), "n_licks": n}
          for (st, pos), (dd, n) in _lick_phase(sd, meas, rows).items() for i, tt in enumerate(t)]
    base = {"animal": animal, "date": date, "epoch": epoch}
    phn = [{**base, "pos": pos_key(x["position"], lat), **x} for x in _phase_norm(sd, L, rows, sp)]
    first = L.sort_values("t_ms").groupby("trial_id", as_index=False).head(1)
    allk = pd.concat([d for d in (P, L, Q) if len(d)], ignore_index=True)
    ls = _lickset_rows({"first": first, "postcue": L, "all": allk}, sp, lat, base)
    wk = []
    for pos, g in meas.groupby("position"):
        for k in range(WALK_MAX):
            gk = g[g.lick_idx == k]
            if len(gk) >= WALK_MIN_LICKS:
                pa = float(gk.peak_angle_deg.median())
                wk.append({**base, "position": pos, "pos": pos_key(pos, lat), "lick_n": k + 1, "peak_angle_deg": pa,
                           "delta_angle_deg": pa - sp.get(pos, np.nan), "spout_angle_deg": sp.get(pos, np.nan),
                           "protrusion_px": float(gk.protrusion_max_measured_px.median()), "n_licks": int(len(gk))})
    # max signed angle DURING the lick (tongue >= 50 % of that lick's peak distance, within the lick: the near-mouth
    # frames, whose angle is artificially large, are excluded -- tongue_kinematics angle.max_signed_*), licks 1-5
    for pos, g in T.groupby("position"):
        for k in range(1, 6):
            c = f"angle_max_signed_lick{k}"
            if c in g and g[c].notna().sum() >= WALK_MIN_LICKS:
                mx = float(g[c].median())
                hit = [r for r in wk if r["position"] == pos and r["lick_n"] == k]
                if hit:
                    hit[0].update(max_angle_deg=mx, max_delta_angle_deg=mx - sp.get(pos, np.nan))
    ct = [{**base, "pos": pos_key(x["position"], lat), **x} for x in _cue_traces(sd, lat, keep)]
    return row, prow, ph, phn, ls, wk, ct


def collect(rv, run: str = RUN, jobs: int | None = None):
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs
    from wfield_local import orofacial_outputs as oo
    lat = {a: str(v.get("stroke_laterality", "L")) for a, v in config.animals().items()}
    items = []
    for s in ak.curated_sessions(order="sorted"):
        a = config.animal_of(s["label"])
        date = "2026" + s["label"].split("_")[-1]
        sd = oo.session_dir(rv, run, a, date)
        if (sd / "manifest.json").exists():
            items.append((sd, a, date, epochs.epoch_of(s["label"]), epochs.days_since_stroke(s["label"]), lat[a]))
    res, fail = ak.fan_sessions(items, session_metrics, jobs=jobs, key=lambda it: (it[1], it[2]), label="session")
    if fail:
        raise RuntimeError(f"{len(fail)} session(s) failed -- refusing to write a partial table: {fail}")
    rows = [v for _, v in res if v is not None]
    return (pd.DataFrame([v[0] for v in rows]), pd.DataFrame([p for v in rows for p in v[1]]),
            pd.DataFrame([x for v in rows for x in v[2]]), pd.DataFrame([x for v in rows for x in v[3]]),
            pd.DataFrame([x for v in rows for x in v[4]]), pd.DataFrame([x for v in rows for x in v[5]]),
            pd.DataFrame([x for v in rows for x in v[6]]), lat)


def _sem(x) -> float:
    x = pd.Series(x).dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def epoch_tables(S: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-animal epoch means (+ SEM over sessions), and the paired animal-weighted change vs pre (`boot_delta`)."""
    from wfield_local import analysis_kit as ak
    g = S.groupby(["animal", "epoch"])
    A = g[list(METRICS)].mean()
    A = A.join(g[list(METRICS)].agg(_sem).add_suffix("_sem")).reset_index()
    A.insert(2, "n_sessions", g.size().to_numpy())
    rng = np.random.default_rng(seed)
    out = []
    for m in METRICS:
        pre = {a: x[m].dropna().tolist() for a, x in S[S.epoch == "pre"].groupby("animal", sort=True)}
        for ep in EPOCH_ORDER[1:]:
            post = {a: x[m].dropna().tolist() for a, x in S[S.epoch == ep].groupby("animal", sort=True)}
            pre_e = {a: v for a, v in pre.items() if v and post.get(a)}
            post_e = {a: v for a, v in post.items() if v and a in pre_e}
            iv = ak.boot_delta(post_e, pre_e, rng) if len(post_e) >= 2 else None
            out.append({"metric": m, "epoch": ep, "n_animals": len(post_e),
                        "delta": iv.point if iv else (float(np.mean([np.mean(post_e[a]) - np.mean(pre_e[a])
                                                                   for a in post_e])) if post_e else np.nan),
                        "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan,
                        "animals": ",".join(sorted(post_e))})
    return A, pd.DataFrame(out)


# --------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _palettes():
    from wfield_local import config
    from wfield_local.epoch_figures import EPOCH_GREY
    return config.animal_color(), dict(EPOCH_GREY)


def _pos_color(key: str, lat: str):
    from wfield_local.spout_behavior import position_style
    return position_style(raw_position(key, lat))[0]


def _grid(n: int, nc: int, w: float, h: float):
    plt = _plt()
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(w * nc, h * nr), squeeze=False)
    for ax in axs.flat[n:]:
        ax.axis("off")
    return plt, fig, axs


def _mark_rows(ax, n: int, step: float = 0.075) -> None:
    """Tag an axes that carries n rows of significance marks / bars along its top: `_tidy` reserves headroom for them."""
    ax._wf_mark_rows = max(getattr(ax, "_wf_mark_rows", 0), n)
    ax._wf_mark_step = step


def _tidy(fig) -> None:
    """Layout rules for every figure (Priya 10-17: legends must not obscure data or significance marks; marks must not
    overlap data). (1) Axes tagged by `_mark_rows` get headroom above the data: the y range grows so the data fill the
    bottom part and the marks sit in the empty band. (2) Every axes legend moves OUTSIDE the data: into an unused
    (axis-off) panel when the figure has one, else to the right of the last panel of its row."""
    axes = fig.get_axes()
    done = set()
    for ax in axes:
        n = getattr(ax, "_wf_mark_rows", 0)
        if n and ax.get_visible():
            key = id(ax.get_shared_y_axes().get_siblings(ax)[0]) if ax.get_shared_y_axes().get_siblings(ax) else id(ax)
            if key in done:
                continue
            done.add(key)
            lo, hi = ax.get_ylim()
            frac = getattr(ax, "_wf_mark_step", 0.075) * n + 0.03   # fraction of the new range kept free at the top
            ax.set_ylim(lo, hi + (hi - lo) * frac / (1 - frac))
    for ax in axes:                                     # readable sizes for grants / papers (Priya 10-17)
        if not ax.axison:
            continue
        from matplotlib.ticker import AutoLocator, MaxNLocator
        for axis in (ax.xaxis, ax.yaxis):               # fewer ticks so the larger labels do not collide
            if type(axis.get_major_locator()) is AutoLocator:
                axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        for t_ in ax.get_xticklabels() + ax.get_yticklabels():
            t_.set_fontsize(max(t_.get_fontsize(), 9))
        for lab in (ax.xaxis.label, ax.yaxis.label):
            lab.set_fontsize(max(lab.get_fontsize(), 10))
        ax.title.set_fontsize(max(ax.title.get_fontsize(), 10.5))
        # wrap long labels / titles to the panel so larger fonts do not run into the neighbouring panel
        import textwrap as _tw
        bb = ax.get_position()
        w_in, h_in = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
        for lab, span in ((ax.title, w_in), (ax.yaxis.label, h_in), (ax.xaxis.label, w_in)):
            txt = lab.get_text()
            if txt and "\n" not in txt:
                nch = max(int(span * 72 / (lab.get_fontsize() * 0.55)), 12)
                if len(txt) > nch:
                    lab.set_text("\n".join(_tw.wrap(txt, nch)))
    spare = [a for a in axes if not a.axison and not a.has_data() and a.get_legend() is None]
    for ax in axes:
        leg = ax.get_legend()
        if leg is None or not ax.axison:
            continue
        h, lbl = ax.get_legend_handles_labels()
        fs = max(leg.get_texts()[0].get_fontsize() if leg.get_texts() else 9, 9)
        leg.remove()
        if not h:
            continue
        if spare:
            tgt = spare.pop(0)
            tgt.legend(h, lbl, loc="center left", fontsize=fs + 1, frameon=False)
            continue
        y0 = ax.get_position().y0
        row = [a for a in axes if a.axison and abs(a.get_position().y0 - y0) < 1e-3]
        last = max(row, key=lambda a: a.get_position().x0)
        last.legend(h, lbl, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=fs, frameon=False)


def save_png_svg(fig, path: Path, dpi: int = 130) -> None:
    """PNG at `path` + an editable SVG copy in `<dir>/svg/<stem>.svg` (Priya 10-17: grant / paper figures)."""
    path = Path(path)
    fig.savefig(path, dpi=dpi)
    (path.parent / "svg").mkdir(exist_ok=True)
    import matplotlib as mpl
    with mpl.rc_context({"svg.fonttype": "none"}):              # text stays text in Illustrator
        fig.savefig(path.parent / "svg" / (path.stem + ".svg"))


def _save(plt, fig, path: Path) -> None:
    _tidy(fig)
    fig.tight_layout()
    save_png_svg(fig, path)
    plt.close(fig)


def fig_epochs(A: pd.DataFrame, D: pd.DataFrame, path: Path, title: str) -> None:
    col, _ = _palettes()
    plt, fig, axs = _grid(len(METRICS), 5, 3.5, 2.8)
    xi = {e: i for i, e in enumerate(EPOCH_ORDER)}
    for ax, m in zip(axs.flat, METRICS):
        for k, (a, g) in enumerate(A.groupby("animal")):
            g = g[g.epoch.isin(xi)].assign(x=lambda d: d.epoch.map(xi)).sort_values("x")
            ax.errorbar(g.x + (k - 1.5) * 0.06, g[m], yerr=g[f"{m}_sem"], fmt="o-", color=col.get(a, "k"), ms=4,
                        lw=1.2, capsize=2, elinewidth=0.8, label=a)
        lab = []
        for e in EPOCH_ORDER[1:]:
            r = D[(D.metric == m) & (D.epoch == e)]
            if len(r) and np.isfinite(r.lo.iloc[0]):
                star = "*" if r.lo.iloc[0] > 0 or r.hi.iloc[0] < 0 else ""
                lab.append(f"{e[:3]} {r.delta.iloc[0]:+.3g}{star}")
        ax.set_xticks(range(len(EPOCH_ORDER)), EPOCH_ORDER, fontsize=7)
        ax.set_title(LABEL[m] + ("\n" + "; ".join(lab) if lab else "\n"), fontsize=7.5)
        ax.set_ylabel(LABEL[m], fontsize=6.5)
        ax.tick_params(labelsize=7)
    axs.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title + "\nlicks: ALL POST-CUE licks (cue + 70 ms .. trial stop; measured peaks for kinematics) unless "
                 "the metric names pre-cue / ITI; per-animal epoch mean +- SEM over sessions; title: paired change vs pre "
                 "(animal-weighted "
                 "bootstrap, * = 95% CI excludes 0; only epochs with >= 2 animals)", fontsize=9)
    _save(plt, fig, path)


def fig_timecourse(S: pd.DataFrame, path: Path, title: str) -> None:
    col, _ = _palettes()
    plt, fig, axs = _grid(len(METRICS), 5, 3.5, 2.6)
    for ax, m in zip(axs.flat, METRICS):
        for a, g in S.sort_values("day").groupby("animal"):
            ax.plot(g.day, g[m], "o-", ms=2.5, lw=0.9, color=col.get(a, "k"), label=a)
        ax.axvline(0.5, color="0.4", ls="--", lw=0.8, zorder=0)
        ax.set_title(LABEL[m], fontsize=8)
        ax.set_ylabel(LABEL[m], fontsize=6.5)
        ax.tick_params(labelsize=7)
    axs.flat[0].legend(fontsize=7, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("days since stroke", fontsize=8)
    fig.suptitle(title + "\nlicks: ALL POST-CUE licks unless the metric names pre-cue / ITI; one point per session; "
                 "dashed = lesion", fontsize=9)
    _save(plt, fig, path)


def _by_pos_panel(ax, P: pd.DataFrame, m: str, ecol: dict, lat: str) -> None:
    x = np.arange(len(POS_ORDER))
    for e in EPOCH_ORDER:
        g = P[P.epoch == e]
        if not len(g):
            continue
        mu = g.groupby("pos")[m].mean().reindex(POS_ORDER)
        se = g.groupby("pos")[m].agg(_sem).reindex(POS_ORDER)
        ax.errorbar(x, mu.to_numpy(), yerr=se.to_numpy(), fmt="o-", color=ecol[e], ms=3.5, lw=1.2, capsize=2,
                    elinewidth=0.8, label=f"{e} ({g.date.nunique()})", )
    ax.axvline(2.5, color="0.85", lw=0.8, zorder=0)
    if m == "peak_angle_deg":
        _spout_markers(ax, P)
    ax.set_xticks(x, [pos_disp(p).replace(" ", "\n") for p in POS_ORDER], fontsize=6)
    for t, p in zip(ax.get_xticklabels(), POS_ORDER):
        t.set_color(_pos_color(p, lat))
    ax.tick_params(labelsize=7)


def _per_animal_figure(P: pd.DataFrame, rows: tuple, lat: dict, path: Path, title: str, note: str) -> None:
    ecol = EPOCH_LINE
    animals = sorted(P.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(rows), len(animals), figsize=(3.4 * len(animals), 2.4 * len(rows)), squeeze=False)
    for j, a in enumerate(animals):
        for i, m in enumerate(rows):
            _by_pos_panel(axs[i][j], P[P.animal == a], m, ecol, lat[a])
            if i == 0:
                axs[i][j].set_title(a, fontsize=9)
            axs[i][j].set_ylabel(LABEL[m], fontsize=7 if j == 0 else 6)
        axs[0][j].legend(fontsize=6, frameon=False)
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 150)), fontsize=9)
    _save(plt, fig, path)


def fig_by_position(Pm, lat, path, title):
    _per_animal_figure(Pm, ("contact_frac", "reach_frac", "protrusion_px", "latency_ms", "latency_contact_ms",
                            "vel_protrude", "vel_retract"), lat, path, title,
                       "licks: ALL POST-CUE licks (measured peaks); by spout position (close ipsi / mid / contra | far "
                       "ipsi / mid / contra), epoch mean +- SEM over "
                       "sessions; compare protrusion WITHIN a position (spouts sit at different image distances), "
                       "reach across positions")


def fig_mismatch(Pm, lat, path, title):
    _per_animal_figure(Pm, ("nocontact_frac", "jaw_no_lick_frac"), lat, path, title,
                       "tongue-jaw mismatch, ALL POST-CUE licks / trials: no-contact licks (tongue out, spout missed) and "
                       "jaw movement without a "
                       "lick; epoch mean +- SEM over sessions")


def fig_angle_tuning(Pm: pd.DataFrame, lat: dict, path: Path, title: str) -> None:
    ecol = EPOCH_LINE
    animals = sorted(Pm.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(1, len(animals), figsize=(3.6 * len(animals), 4.0), squeeze=False)
    for ax, a in zip(axs[0], animals):
        g = Pm[Pm.animal == a]
        for e in EPOCH_ORDER:
            ge = g[g.epoch == e]
            if not len(ge):
                continue
            # close and far sit at the SAME image angle (~+-40 deg): one line per ring (far solid, close dashed)
            for ring, ls in (("far", "-"), ("close", "--")):
                gr = ge[ge.pos.str.startswith(ring)]
                mu = gr.groupby("pos")[["spout_angle_deg", "peak_angle_deg"]].mean().sort_values("spout_angle_deg")
                se = gr.groupby("pos").peak_angle_deg.agg(_sem).reindex(mu.index)
                ax.plot(mu.spout_angle_deg, mu.peak_angle_deg, ls, color=ecol[e], lw=1.3,
                        label=f"{e} ({ge.date.nunique()})" if ring == "far" else None)
                for p, r in mu.iterrows():
                    ax.errorbar(r.spout_angle_deg, r.peak_angle_deg, yerr=se[p], fmt="o", ms=4, capsize=2,
                                color=ecol[e], elinewidth=0.8)
        ax.plot([-50, 50], [-50, 50], color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(-50, 50)
        ax.set_title(a, fontsize=9)
        ax.set_xlabel("spout angle (deg; + = image right = L positions)", fontsize=7.5)
        ax.legend(fontsize=6.5, frameon=False)
        ax.tick_params(labelsize=7)
    axs[0][0].set_ylabel("executed peak tongue angle (deg)", fontsize=8)
    fig.suptitle(title + "\nexecuted vs target angle, licks: ALL POST-CUE licks: per-position median (measured peaks), "
                 "epoch mean +- SEM over "
                 "sessions\nfar = solid, close = dashed; grey = identity; left = contra positions, right = ipsi",
                 fontsize=9)
    _save(plt, fig, path)


def _phase_fig(PH: pd.DataFrame, lat: dict, path: Path, title: str, portrait: bool) -> None:
    ecol = EPOCH_LINE
    animals = sorted(PH.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(animals), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER), 2.3 * len(animals)),
                            squeeze=False, sharex=True, sharey="row" if not portrait else False)
    for i, a in enumerate(animals):
        for j, p in enumerate(POS_ORDER):
            ax = axs[i][j]
            for e in EPOCH_ORDER:
                g = PH[(PH.animal == a) & (PH.pos == p) & (PH.epoch == e)]
                if not len(g):
                    continue
                piv_d = g.pivot_table(index="date", columns="t_ms", values="protrusion_px")
                piv_v = g.pivot_table(index="date", columns="t_ms", values="velocity_px_s")
                t = piv_d.columns.to_numpy()
                md, mv = piv_d.mean().to_numpy(), piv_v.mean().to_numpy()
                if portrait:
                    # one lick cycle only: +-152 ms also holds the previous lick's retraction and the next lick's
                    # protrusion, which drew as inner arcs (Priya 10-16)
                    w = np.abs(t) <= PORTRAIT_HALF_MS
                    if len(piv_d) > 1:
                        sd_d = (piv_d.std(ddof=1) / np.sqrt(len(piv_d))).to_numpy()
                        sd_v = (piv_v.std(ddof=1) / np.sqrt(len(piv_v))).to_numpy()
                        _sem_ribbon(ax, md[w], mv[w] / 1000, sd_d[w], sd_v[w] / 1000, ecol[e])
                    t, md, mv = t[w], md[w], mv[w]
                    ax.plot(md, mv / 1000, color=ecol[e], lw=1.3, label=f"{e} ({len(piv_d)})")
                    ax.plot(md[t == 0], mv[t == 0] / 1000, "o", color=ecol[e], ms=3)
                else:
                    sd = (piv_d.std(ddof=1) / np.sqrt(len(piv_d))).to_numpy() if len(piv_d) > 1 else None
                    ax.plot(t, md, color=ecol[e], lw=1.3, label=f"{e} ({len(piv_d)})")
                    if sd is not None:
                        ax.fill_between(t, md - sd, md + sd, color=ecol[e], alpha=0.25, lw=0)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=8.5, color=_pos_color(p, lat[a]))
            if j == 0:
                ax.set_ylabel(f"{a}\n" + ("protrusion speed (px/ms)" if portrait else "protrusion (px)"), fontsize=8)
            if portrait:
                ax.axhline(0, color="0.85", lw=0.8, zorder=0)
            else:
                ax.axvline(0, color="0.85", lw=0.8, zorder=0)
            ax.tick_params(labelsize=6.5)
        axs[i][0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("protrusion (px)" if portrait else "ms from lick peak", fontsize=7.5)
    fig.suptitle(title + ("\nlick phase portrait, licks: ALL POST-CUE licks (measured peaks): protrusion speed vs "
                          "protrusion, mean lick per epoch, "
                          f"+-{PORTRAIT_HALF_MS:.0f} ms around the peak (one cycle); dot = peak; band = SEM over "
                          "sessions; IN-MOUTH FRAMES INCLUDED (0 px)" if portrait else
                          "\nlick-peak-aligned protrusion, licks: ALL POST-CUE licks (measured peaks), epoch mean +- SEM "
                          "over sessions; IN-MOUTH FRAMES "
                          "INCLUDED (0 px)"),
                 fontsize=9)
    _save(plt, fig, path)


def fig_epochs_by_position(Pm: pd.DataFrame, lat: dict, metrics: tuple, path: Path, title: str) -> None:
    """Rows = metric, columns = the six positions; x = epoch; one line per animal, mean +- SEM over sessions."""
    col, _ = _palettes()
    plt = _plt()
    fig, axs = plt.subplots(len(metrics), len(POS_ORDER), figsize=(2.5 * len(POS_ORDER), 2.2 * len(metrics)),
                            squeeze=False, sharey="row")
    from wfield_local.orofacial_stats import within_animal_tests
    W = within_animal_tests(Pm, metrics)                  # per animal: post-epoch sessions vs its own pre sessions
    for j, p in enumerate(POS_ORDER):
        for i, m in enumerate(metrics):
            ax = axs[i][j]
            for k, (a, g) in enumerate(Pm[Pm.pos == p].groupby("animal")):
                mu = g.groupby("epoch")[m].mean().reindex(EPOCH_ORDER)
                se = g.groupby("epoch")[m].agg(_sem).reindex(EPOCH_ORDER)
                x = np.arange(len(EPOCH_ORDER)) + (k - 1.5) * 0.07
                ok = mu.notna().to_numpy()
                ax.errorbar(x[ok], mu[ok], yerr=se[ok], fmt="o-", color=col.get(a, "k"), ms=3.5, lw=1.1, capsize=2,
                            elinewidth=0.7, label=a)
                _mark_rows(ax, 4)
                sig = W[(W.metric == m) & (W.pos == p) & (W.animal == a) & W.sig]
                for e in sig.epoch:                      # * in the animal's colour above its significant epochs
                    xe = EPOCH_ORDER.index(e) + (k - 1.5) * 0.07
                    ax.text(xe, 0.985 - 0.065 * k, "*", color=col.get(a, "k"), ha="center", va="top", fontsize=10,
                            fontweight="bold", transform=ax.get_xaxis_transform())
            if m == "peak_angle_deg":
                ax.axhline(Pm[Pm.pos == p].spout_angle_deg.mean(), color="0.55", ls="--", lw=1.0, zorder=0)
            ax.set_xticks(range(len(EPOCH_ORDER)), [e[:3] for e in EPOCH_ORDER], fontsize=6.5)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=9, color=_pos_color(p, "L"))
            if j == 0:
                ax.set_ylabel(LABEL[m], fontsize=7)
    axs[0][0].legend(fontsize=6, frameon=False)
    note = ("by spout position: per-animal epoch mean +- SEM over sessions; licks: ALL POST-CUE licks (cue + 70 ms .. "
            "trial stop), measured peaks for kinematics; dashed = spout angle. * (animal colour, one row per animal) = "
            "that animal's sessions in the epoch differ from ITS OWN pre-stroke sessions: two-sided Mann-Whitney on "
            "sessions, BH-FDR within each metric (all animals x positions x epochs), q < 0.05")
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 150)), fontsize=9)
    _save(plt, fig, path)


def fig_timecourse_by_position(Pm: pd.DataFrame, metrics: tuple, path: Path, title: str) -> None:
    """Rows = metric, columns = the six positions; x = days since stroke; one line per animal (one point per session)."""
    col, _ = _palettes()
    plt = _plt()
    fig, axs = plt.subplots(len(metrics), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER), 2.1 * len(metrics)),
                            squeeze=False, sharey="row", sharex=True)
    for j, p in enumerate(POS_ORDER):
        for i, m in enumerate(metrics):
            ax = axs[i][j]
            for a, g in Pm[Pm.pos == p].sort_values("day").groupby("animal"):
                ax.plot(g.day, g[m], "o-", ms=2.2, lw=0.8, color=col.get(a, "k"), label=a)
            ax.axvline(0.5, color="0.6", ls="--", lw=0.7, zorder=0)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=9, color=_pos_color(p, "L"))
            if j == 0:
                ax.set_ylabel(LABEL[m], fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("days since stroke", fontsize=7)
    axs[0][0].legend(fontsize=6, frameon=False)
    fig.suptitle(title + "\nby spout position; licks: ALL POST-CUE licks (measured peaks for kinematics); one point per "
                 "session; dashed = lesion", fontsize=9)
    _save(plt, fig, path)


def fig_lick_phase_value(PH: pd.DataFrame, lat: dict, col: str, ylabel: str, path: Path, title: str,
                         note: str, zero_line: bool = True, delta_pre: bool = False,
                         xlabel: str = "ms from lick peak", spout: dict | None = None, spout_ipsi: bool = False) -> None:
    """Rows = animals, columns = the six positions: ``col`` around the lick peak, epoch mean +- SEM over sessions."""
    ecol = EPOCH_LINE
    animals = sorted(PH.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(animals), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER), 2.3 * len(animals)),
                            squeeze=False, sharex=True, sharey=True)
    for i, a in enumerate(animals):
        for j, p in enumerate(POS_ORDER):
            ax = axs[i][j]
            pre_curve = _pre_curve(PH, a, p, "t_ms", col) if delta_pre else None
            if delta_pre and pre_curve is None:
                continue
            for e in EPOCH_ORDER:
                g = PH[(PH.animal == a) & (PH.pos == p) & (PH.epoch == e)]
                if not len(g):
                    continue
                piv = g.pivot_table(index="date", columns="t_ms", values=col)
                if pre_curve is not None:
                    piv = piv.sub(pre_curve, axis=1)
                t, mu = piv.columns.to_numpy(), piv.mean().to_numpy()
                ax.plot(t, mu, color=ecol[e], lw=1.3, label=f"{e} ({len(piv)})")
                if len(piv) > 1:
                    se = (piv.std(ddof=1) / np.sqrt(piv.notna().sum())).to_numpy()
                    ax.fill_between(t, mu - se, mu + se, color=ecol[e], alpha=0.2, lw=0)
            ax.axvline(0, color="0.85", lw=0.8, zorder=0)
            if zero_line:
                ax.axhline(0, color="0.85", lw=0.8, zorder=0)
            _spout_line(ax, spout, a, p, lat, spout_ipsi)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=8.5, color=_pos_color(p, lat[a]))
            if j == 0:
                ax.set_ylabel(f"{a}\n" + "\n".join(textwrap.wrap(ylabel, 28)), fontsize=7.5)
        axs[i][0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel(xlabel, fontsize=7.5)
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 150)), fontsize=9)
    _save(plt, fig, path)


def _circ_smooth(y: np.ndarray, sigma_bins: float = 1.0) -> np.ndarray:
    """NaN-aware circular Gaussian smoothing over phase bins (stroke_orofacial PHASE_CURVE_SIGMA_BINS = 1)."""
    k = np.arange(-3, 4)
    w = np.exp(-0.5 * (k / sigma_bins) ** 2)
    v = np.nan_to_num(y)
    m = np.isfinite(y).astype(float)
    num = sum(wi * np.roll(v, ki) for wi, ki in zip(w, k))
    den = sum(wi * np.roll(m, ki) for wi, ki in zip(w, k))
    with np.errstate(all="ignore"):
        out = num / den
    out[~np.isfinite(y)] = np.nan
    return out


def spout_angles(Pm: pd.DataFrame) -> dict:
    """{(animal, pos): mean spout angle over sessions} (image sign: + = image right = the L positions)."""
    return Pm.groupby(["animal", "pos"]).spout_angle_deg.mean().to_dict()


def _spout_line(ax, spout: dict | None, animal: str, pos: str, lat: dict, ipsi: bool) -> None:
    """Dashed reference at the spout angle -- where the tongue would point if it aimed at the spout (Priya 10-16: on
    every raw-angle plot). ``ipsi``: the plot's angles are signed + = ipsilesional (Family 8.2), so flip for an R lesion."""
    if not spout or (animal, pos) not in spout or not np.isfinite(spout[(animal, pos)]):
        return
    v = spout[(animal, pos)] * (1.0 if (not ipsi or lat.get(animal, "L") == "L") else -1.0)
    ax.axhline(v, color="0.55", ls="--", lw=1.0, zorder=0)


def _sem_ribbon(ax, x, y, sx, sy, color, alpha: float = 0.22) -> None:
    """Faded band along a 2-D path, half-width = the SEM projected on the path's normal at each point
    (sqrt((sx nx)^2 + (sy ny)^2)) -- the spread across sessions perpendicular to where the tongue is going (Priya
    10-16, replaces the crosses)."""
    x, y, sx, sy = (np.asarray(v, float) for v in (x, y, sx, sy))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy)
    if ok.sum() < 3:
        return
    x, y, sx, sy = x[ok], y[ok], sx[ok], sy[ok]
    dx, dy = np.gradient(x), np.gradient(y)
    nrm = np.hypot(dx, dy)
    nrm[nrm == 0] = np.nan
    nx, ny = -dy / nrm, dx / nrm
    nx, ny = np.nan_to_num(nx), np.nan_to_num(ny)
    w = np.sqrt((sx * nx) ** 2 + (sy * ny) ** 2)
    ax.fill(np.r_[x + w * nx, (x - w * nx)[::-1]], np.r_[y + w * ny, (y - w * ny)[::-1]], color=color, alpha=alpha,
            lw=0, zorder=2)


def _spout_markers(ax, P: pd.DataFrame) -> None:
    """Spout angle per position on a position x-axis (grey dashes + bars)."""
    if "spout_angle_deg" in P:
        sp = P.groupby("pos").spout_angle_deg.mean()
    elif {"peak_angle_deg", "delta_angle_deg"} <= set(P.columns):
        sp = (P.peak_angle_deg - P.delta_angle_deg).groupby(P.pos).mean()
    else:
        return
    ax.plot(np.arange(len(POS_ORDER)), sp.reindex(POS_ORDER).to_numpy(), marker="_", ms=12, mew=1.6,
            color="0.55", ls="--", lw=1.0, zorder=0, label="spout angle")


def _pre_curve(D: pd.DataFrame, animal: str, pos: str, xcol: str, col: str):
    """This animal's mean pre-stroke curve at one position (mean over pre sessions), or None."""
    g = D[(D.animal == animal) & (D.pos == pos) & (D.epoch == "pre")]
    return g.pivot_table(index="date", columns=xcol, values=col).mean() if len(g) else None


def fig_phase_norm(PHN: pd.DataFrame, lat: dict, col: str, mode: str, ylabel: str, path: Path, title: str,
                   note: str, delta_pre: bool = False, spout: dict | None = None) -> None:
    """Rows = animals, columns = positions; x = normalised lick phase (0.5 = peak); epoch mean +- SEM over sessions."""
    ecol = EPOCH_LINE
    D = PHN[PHN["mode"] == mode]
    animals = sorted(D.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(animals), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER), 2.3 * len(animals)),
                            squeeze=False, sharex=True, sharey=True)
    for i, a in enumerate(animals):
        for j, p in enumerate(POS_ORDER):
            ax = axs[i][j]
            ax.axvline(0.5, color="0.8", ls="--", lw=0.8, zorder=0)
            if col != "protrusion_px" or delta_pre:
                ax.axhline(0, color="0.85", lw=0.8, zorder=0)
            _spout_line(ax, spout, a, p, lat, False)
            pre_curve = _pre_curve(D, a, p, "phase", col) if delta_pre else None
            if delta_pre and pre_curve is None:
                continue
            for e in EPOCH_ORDER:
                g = D[(D.animal == a) & (D.pos == p) & (D.epoch == e)]
                if not len(g):
                    continue
                piv = g.pivot_table(index="date", columns="phase", values=col)
                if pre_curve is not None:
                    piv = piv.sub(pre_curve, axis=1)
                x = piv.columns.to_numpy()
                mu = _circ_smooth(piv.mean().to_numpy())
                ax.plot(x, mu, color=ecol[e], lw=1.4, label=f"{e} ({len(piv)})", zorder=3)
                if len(piv) > 1:
                    se = _circ_smooth((piv.std(ddof=1) / np.sqrt(piv.notna().sum())).to_numpy())
                    ax.fill_between(x, mu - se, mu + se, color=ecol[e], alpha=0.2, lw=0, zorder=2)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=8.5, color=_pos_color(p, lat[a]))
            if j == 0:
                ax.set_ylabel(f"{a}\n" + "\n".join(textwrap.wrap(ylabel, 28)), fontsize=7.5)
        axs[i][0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("lick phase (0 = between peaks, 0.5 = peak)", fontsize=7.5)
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 150)), fontsize=9)
    _save(plt, fig, path)


def fig_xy_trajectory(PH: pd.DataFrame, Pm: pd.DataFrame, lat: dict, path: Path, title: str,
                      which: str = "postcue") -> None:
    """Mean tongue-tip path over one lick (+-PORTRAIT_HALF_MS around the peak) in the mouth frame (lateral, + = image
    right = L positions; out of the mouth up), one curve per spout position in its palette colour, x = that position's
    median spout location; rows = animals, columns = epochs."""
    animals = sorted(PH.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(animals), len(EPOCH_ORDER), figsize=(3.3 * len(EPOCH_ORDER), 3.3 * len(animals)),
                            squeeze=False, sharex=True, sharey=True)
    W = PH[(PH.t_ms.abs() <= PORTRAIT_HALF_MS) & (PH.set == which)]
    for i, a in enumerate(animals):
        for j, e in enumerate(EPOCH_ORDER):
            ax = axs[i][j]
            ax.plot(0, 0, "k+", ms=8, zorder=1)
            g = W[(W.animal == a) & (W.epoch == e)]
            if not len(g):
                ax.axis("off")
                continue
            for p in POS_ORDER:
                gp = g[g.pos == p]
                if not len(gp):
                    continue
                c = _pos_color(p, lat[a])
                mu = gp.groupby("t_ms")[["lat_px", "out_px"]].mean()
                ax.plot(mu.lat_px, -mu.out_px, color=c, lw=1.5, zorder=3, label=pos_disp(p))
                if gp.date.nunique() > 1:              # SEM over sessions as a ribbon along the path
                    se = gp.groupby("t_ms")[["lat_px", "out_px"]].agg(_sem).reindex(mu.index)
                    _sem_ribbon(ax, mu.lat_px, -mu.out_px, se.lat_px, se.out_px, c)
                if 0.0 in mu.index:
                    ax.plot(mu.lat_px[0.0], -mu.out_px[0.0], "o", color=c, ms=3.5, zorder=4)
                sp = Pm[(Pm.animal == a) & (Pm.epoch == e) & (Pm.pos == p)]
                if len(sp):
                    d, an = sp.spout_dist_px.mean(), np.deg2rad(sp.spout_angle_deg.mean())
                    ax.plot(d * np.sin(an), -d * np.cos(an), "x", color=c, ms=7, mew=1.8, zorder=4)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(e, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{a}\nfrom mouth (px; camera view)", fontsize=8)
            n = g.date.nunique()
            ax.text(0.02, 0.98, f"{n} session{'s' if n != 1 else ''}", transform=ax.transAxes, fontsize=6.5,
                    va="top")
    for ax in axs[-1]:
        ax.set_xlabel("lateral (px; + = image right = L)", fontsize=7.5)
    h, lb = axs[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, lb, loc="lower center", fontsize=7.5, frameon=False, ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.9, bottom=0.07)
    lab = {"postcue": "all post-cue licks", "first": "FIRST post-cue lick", "second": "SECOND post-cue lick"}[which]
    note = (f"licks: {lab}; mean tongue-tip path over one lick (+-{PORTRAIT_HALF_MS:.0f} ms around the peak, measured "
            "peaks), as the camera sees it (mouth at top); + = mouth, dot = peak, x = spout; band = SEM over sessions "
            "(perpendicular to the path); IN-MOUTH FRAMES INCLUDED (at the mouth)")
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 120)), fontsize=9)
    _save(plt, fig, path)


def fig_walk(WK: pd.DataFrame, lat: dict, col: str, ylabel: str, path: Path, title: str, note: str) -> None:
    """Rows = animals, columns = positions; x = lick number within the trial; epoch mean +- SEM over sessions. With
    col = peak_angle_deg the dashed line is the spout angle (where the tongue would have to go)."""
    ecol = EPOCH_LINE
    animals = sorted(WK.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(animals), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER), 2.3 * len(animals)),
                            squeeze=False, sharex=True, sharey=True)
    for i, a in enumerate(animals):
        for j, p in enumerate(POS_ORDER):
            ax = axs[i][j]
            g0 = WK[(WK.animal == a) & (WK.pos == p)]
            if col in ("delta_angle_deg", "max_delta_angle_deg"):
                ax.axhline(0, color="0.8", lw=0.9, zorder=0)
            elif len(g0):
                ax.axhline(g0.spout_angle_deg.mean(), color="0.6", ls="--", lw=0.9, zorder=0)
            for k, e in enumerate(EPOCH_ORDER):
                g = g0[g0.epoch == e]
                if not len(g):
                    continue
                mu = g.groupby("lick_n")[col].mean()
                se = g.groupby("lick_n")[col].agg(_sem)
                ax.errorbar(mu.index + (k - 1.5) * 0.08, mu, yerr=se, fmt="o-", color=ecol[e], ms=3, lw=1.2,
                            capsize=1.5, elinewidth=0.7, zorder=3, label=f"{e} ({g.date.nunique()})")
            ax.set_xticks(range(1, WALK_MAX + 1))
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=8.5, color=_pos_color(p, lat[a]))
            if j == 0:
                ax.set_ylabel(f"{a}\n" + "\n".join(textwrap.wrap(ylabel, 28)), fontsize=7.5)
        axs[i][0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("lick number in trial", fontsize=7.5)
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 150)), fontsize=9)
    _save(plt, fig, path)


def fig_lick12(Pm: pd.DataFrame, lat: dict, path: Path, title: str) -> None:
    """Lick 1 vs lick 2 (LP-detected) per position: peak time, peak protrusion, peak angle; solid = lick 1, dashed =
    lick 2; epoch colours; mean +- SEM over sessions."""
    ecol = EPOCH_LINE
    animals = sorted(Pm.animal.unique())
    rows = (("l1_t_ms", "l2_t_ms", "peak time (ms from cue)"), ("l1_prot_px", "l2_prot_px", "peak protrusion (px)"),
            ("l1_angle_deg", "l2_angle_deg", "peak angle (deg; + = L)"))
    plt = _plt()
    fig, axs = plt.subplots(len(rows) + 1, len(animals), figsize=(3.5 * len(animals), 2.4 * (len(rows) + 1)),
                            squeeze=False)
    x = np.arange(len(POS_ORDER))
    for j, a in enumerate(animals):
        P = Pm[Pm.animal == a]
        for i, (c1, c2, lab) in enumerate(rows + (("l12_dt_ms", None, "lick 2 - lick 1 (ms)"),)):
            ax = axs[i][j]
            for e in EPOCH_ORDER:
                g = P[P.epoch == e]
                if not len(g):
                    continue
                for c, ls, mk in ((c1, "-", "o"), (c2, "--", "s")):
                    if c is None:
                        continue
                    mu = g.groupby("pos")[c].mean().reindex(POS_ORDER)
                    se = g.groupby("pos")[c].agg(_sem).reindex(POS_ORDER)
                    ax.errorbar(x, mu.to_numpy(), yerr=se.to_numpy(), fmt=mk + ls, color=ecol[e], ms=3.2, lw=1.1,
                                capsize=2, elinewidth=0.7, zorder=3,
                                label=(f"{e} ({g.date.nunique()})" if c == c1 else None))
            ax.axvline(2.5, color="0.85", lw=0.8, zorder=0)
            if c1 == "l1_angle_deg":
                _spout_markers(ax, P)
            ax.set_xticks(x, [pos_disp(q).replace(" ", "\n") for q in POS_ORDER], fontsize=6)
            for t, q in zip(ax.get_xticklabels(), POS_ORDER):
                t.set_color(_pos_color(q, lat[a]))
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(a, fontsize=9)
            ax.set_ylabel(lab, fontsize=7 if j == 0 else 6)
        axs[0][j].legend(fontsize=6, frameon=False)
    fig.suptitle(title + "\nlick 1 (solid, o) vs lick 2 (dashed, s), LP-detected post-cue licks; per-position median per "
                 "session, epoch mean +- SEM over sessions", fontsize=9)
    _save(plt, fig, path)


# --------------------------------------------------------------------------- COHORT minus-pre summaries (Priya 10-17)

from wfield_local.orofacial_stats import cluster_permutation, session_change_tests  # noqa: E402


def _animal_delta_curves(D: pd.DataFrame, xcol: str, col: str, pos: str, weight: str = "session",
                         raw: bool = False, cover: bool = True) -> dict:
    """{epoch: (rows DataFrame, {animal: animal-mean curve})}. Every session's curve minus ITS ANIMAL's mean pre-stroke
    curve at this position. weight='session': rows = the sessions (each session counts once; animals with more sessions
    weigh more -- Priya 10-17); weight='animal': rows = each animal's mean over its sessions (CLAUDE.md rule 9: the
    convention for CHANGES, which keeps a many-session animal from deciding the result)."""
    out: dict = {}
    for a, ga in D[D.pos == pos].groupby("animal"):
        piv_all = ga.pivot_table(index=["epoch", "date"], columns=xcol, values=col)
        if "pre" not in piv_all.index.get_level_values(0):
            continue
        pre = (_covered(piv_all.loc["pre"]) if cover else piv_all.loc["pre"]).mean()
        if raw:                                         # RAW level (no subtraction), same rows / coverage rules
            pre = pd.Series(0.0, index=piv_all.columns)
        for e in EPOCH_ORDER:
            if e in piv_all.index.get_level_values(0):
                dd = piv_all.loc[e] - pre
                if weight == "animal" and cover:
                    dd = _covered(dd)
                rows, am, an = out.setdefault(e, ([], {}, []))
                am[a] = dd.mean()
                if weight == "session":
                    rows.extend(r for _, r in dd.iterrows())
                    an.extend([a] * len(dd))
                else:
                    rows.append(dd.mean())
                    an.append(a)
    res = {}
    for e, (r, am, an) in out.items():
        R = pd.DataFrame(r).reset_index(drop=True)
        need = max(MIN_COVER_N, int(np.ceil(MIN_COVER_FRAC * len(R)))) if cover else 1
        bad = R.notna().sum() < need                    # too few sessions (animals) carry this bin
        R.loc[:, bad] = np.nan
        am = {a: c.where(~bad.reindex(c.index, fill_value=True)) for a, c in am.items()}
        res[e] = (R, am, np.asarray(an))
    return res


#: coverage rule for change curves (Priya 10-17, far-ipsi spike at the cue): a bin is drawn only when at least
#: MIN_COVER_FRAC of the epoch's rows (sessions, or animals when animal-weighted) and at least MIN_COVER_N of them have
#: data there. Tongue-OUT traces before ~+100 ms from the cue exist in only a few sessions (the tongue is rarely out),
#: so without this one acute session of one animal drew the whole pre-cue segment.
MIN_COVER_FRAC = 0.25
MIN_COVER_N = 3
MIN_LICKS_SESSION_POS = 20
ENGAGED_ONLY = True


def _apply_config() -> None:
    """Inclusion thresholds live in configs/defaults.yaml `orofacial_cohort:` (Priya 10-17, as in stroke_orofacial's
    min_trials_per_bin / min_sessions_per_bin / min_trials_with_detected_lick); the module constants are the fallback."""
    global MIN_COVER_FRAC, MIN_COVER_N, MIN_LICKS_SESSION_POS, CUE_MIN_FRAMES, MIN_PHASE_POINTS, WALK_MIN_LICKS
    global ENGAGED_ONLY
    try:
        from wfield_local import config
        c = (config.defaults() or {}).get("orofacial_cohort") or {}
    except Exception:                                           # tests / no config: keep the defaults above
        c = {}
    MIN_COVER_FRAC = float(c.get("min_cover_frac", MIN_COVER_FRAC))
    MIN_COVER_N = int(c.get("min_cover_n", MIN_COVER_N))
    MIN_LICKS_SESSION_POS = int(c.get("min_licks_per_session_position", MIN_LICKS_SESSION_POS))
    CUE_MIN_FRAMES = int(c.get("cue_min_frames_per_bin", CUE_MIN_FRAMES))
    MIN_PHASE_POINTS = int(c.get("min_phase_intervals_per_bin", MIN_PHASE_POINTS))
    WALK_MIN_LICKS = int(c.get("walk_min_licks", WALK_MIN_LICKS))
    # env var so spawned workers follow `--all-trials` (the CLI flag cannot reach a fresh interpreter otherwise)
    ENGAGED_ONLY = bool(c.get("engaged_only", ENGAGED_ONLY)) and not os.environ.get("OROFACIAL_ALL_TRIALS")


def _cue_table(sd: Path) -> pd.DataFrame:
    """Per orofacial trial: trial_id, position, cue time on the DAQ clock (s), from windows.npz."""
    w = np.load(sd / "windows.npz")
    ix = pd.DataFrame({"trial_id": w["ix_trial_id"].astype(int), "position": w["ix_position"].astype(str),
                       "t_ms": w["ix_t_ms"].astype(float), "daq_s": w["ix_daq_s"].astype(float)})
    ix["cue_daq"] = ix.daq_s - ix.t_ms / 1000.0
    return ix.groupby("trial_id", sort=True).agg(position=("position", "first"), cue_daq=("cue_daq", "median")
                                                  ).reset_index()


def _align_by_order(cue: pd.DataFrame, bt: pd.DataFrame) -> np.ndarray | None:
    """Row of the behaviour table for every orofacial trial, by aligning the two TRIAL ORDERS within each DAQ recording
    segment (a > 300 s gap between cues = a restart, e.g. PS92_0812's crash): the integer row offset whose positions
    agree best. Accepted only with >= 98 % position agreement over >= 80 % of the segment AND a consistent clock
    (SD of DAQ cue minus the log's device trial end < 2 s -- chance alignments do not keep time). None if any
    segment fails."""
    lp = bt.pos_name.to_numpy(str)
    t_end = bt.device_trial_end_ms.to_numpy(float) / 1000.0 if "device_trial_end_ms" in bt else None
    seg = np.r_[0, np.cumsum(np.diff(cue.cue_daq.to_numpy()) > 300.0)]
    out = np.full(len(cue), -1)
    for sg in np.unique(seg):
        idx = np.flatnonzero(seg == sg)
        pos = cue.position.to_numpy(str)[idx]
        n, best = len(idx), None
        for k in range(-len(bt), len(bt)):
            jj = np.arange(n) + k
            ok = (jj >= 0) & (jj < len(bt))
            if ok.sum() < 0.8 * n:
                continue
            ag = float((lp[jj[ok]] == pos[ok]).mean())
            if best is None or ag > best[1]:
                best = (k, ag)
        if best is None or best[1] < 0.98:
            return None
        jj = np.arange(n) + best[0]
        ok = (jj >= 0) & (jj < len(bt))
        if t_end is not None and ok.sum() > 2:
            if np.std(cue.cue_daq.to_numpy()[idx][ok] - t_end[jj[ok]]) > 2.0:
                return None
        out[idx[ok]] = jj[ok]
    return out


def engagement_keep(animal: str, date: str, sd: Path) -> tuple[set | None, str]:
    """(engaged orofacial trial_ids or None, gate label) for one session folder.

    The gate is the behaviour pipeline's per-trial `engaged` column. Its trial_ids equal ours (DAQ cue order) when it was
    built from the DAQ; when the DAQ quality gate sent it to the GUI LOG instead (source = GUI: 1-based ids, the log's
    own trial set) they do not -- measured 10-17: PS92_0812 (log 280 trials incl. the 41-min imaging gap vs 225 DAQ cues)
    and PS92_0929. Then the trials are matched by ORDER within each recording segment (`_align_by_order`; both
    sessions: 100 % position agreement, clock consistent) and the engaged flags carried across. A session that matches
    neither way stays ungated and says so."""
    if not ENGAGED_ONLY:
        return None, "off (all trials)"
    from wfield_local.paths import PathResolver
    d = Path(PathResolver().root("behavior_out")) / "sessions" / animal / date
    c = sorted(d.glob(f"{animal}_{date}_*_trials.csv"))
    if not c:
        return None, "none (no behaviour table)"
    bt = pd.read_csv(c[-1])
    if "engaged" not in bt or "trial_id" not in bt:
        return None, "none (no engaged column)"
    cue = _cue_table(sd)
    m = cue.merge(bt[["trial_id", "pos_name", "engaged"]].astype({"trial_id": int}), on="trial_id", how="left")
    bad = int(m.pos_name.isna().sum() + (m.dropna(subset=["pos_name"]).position
                                         != m.dropna(subset=["pos_name"]).pos_name).sum())
    if bad <= 0.01 * max(len(m), 1):
        return set(m.loc[m.engaged.fillna(False).astype(bool), "trial_id"].astype(int)), "reference"
    rows = _align_by_order(cue, bt.reset_index(drop=True))
    if rows is None:
        return None, f"none (trial ids disagree on {bad} trials and order alignment failed)"
    eng = bt["engaged"].astype(bool).to_numpy()
    keep = {int(t) for t, r in zip(cue.trial_id, rows) if r >= 0 and eng[r]}
    return keep, f"reference (matched by trial order: ids disagreed on {bad} trials)"


def engaged_trial_ids(animal: str, date: str) -> set | None:
    """trial_ids the BEHAVIOUR pipeline flagged engaged (`spout_behavior.session_metrics` -> `reference_engagement`:
    judged at the reference positions only, terminal non-recovering collapse -- the post-stroke-valid gate; the older
    `flag_engagement` counts far-position motor failure as disengagement). Read from that pipeline's persisted
    `<behavior_out>/sessions/<animal>/<date>/<sid>_trials.csv`, the same table `trial_windows.trial_bounds` takes its
    trial_ids from. None when the session has no such table (then nothing can be gated)."""
    from wfield_local.paths import PathResolver
    d = Path(PathResolver().root("behavior_out")) / "sessions" / animal / date
    c = sorted(d.glob(f"{animal}_{date}_*_trials.csv"))
    if not c:
        return None
    t = pd.read_csv(c[-1])
    if "engaged" not in t or "trial_id" not in t:
        return None
    return set(t.loc[t["engaged"].astype(bool), "trial_id"].astype(int))


_apply_config()


def _covered(piv: pd.DataFrame) -> pd.DataFrame:
    """Blank columns (bins) where fewer than MIN_COVER_N of an animal's sessions (all of them, if it has fewer) have
    data, before averaging them."""
    return piv.loc[:, piv.notna().sum() >= min(MIN_COVER_N, len(piv))].reindex(columns=piv.columns)


def fig_cohort_delta(rows: list, path: Path, title: str, note: str, xlabel: str, weight: str = "session") -> None:
    """Rows = (DataFrame, x column, value column, y label, smooth?); columns = the six positions. Per epoch: bold = mean
    over ANIMALS of each animal's change from its own pre-stroke curve, band = SEM over animals, faint = each animal."""
    ecol = EPOCH_LINE
    plt = _plt()
    fig, axs = plt.subplots(len(rows), len(POS_ORDER), figsize=(2.7 * len(POS_ORDER), 2.5 * len(rows)),
                            squeeze=False, sharex=True, sharey="row")
    rlo: dict = {}
    rhi: dict = {}
    for i, row in enumerate(rows):
        D, xcol, col, ylab, smooth = row[:5]
        raw = len(row) > 5 and bool(row[5])             # 6th field True = plot the RAW level; bars = the change test
        for j, p in enumerate(POS_ORDER):
            ax = axs[i][j]
            ax.axhline(0, color="0.8", lw=0.8, zorder=0)
            if xcol in ("t_ms",):
                ax.axvline(0, color="0.85", lw=0.8, zorder=0)
            elif xcol == "phase":
                ax.axvline(0.5, color="0.85", ls="--", lw=0.8, zorder=0)
            cur = _animal_delta_curves(D, xcol, col, p, weight, raw=raw)
            cur_s = (cur if weight == "session" and not raw else _animal_delta_curves(D, xcol, col, p, "session"))
            if "pre" in cur_s:                                  # cluster permutation vs the pre sessions (on SESSIONS)
                _mark_rows(ax, 3, step=0.045)
                Mp, _, anp = cur_s["pre"]
                for k, e in enumerate(EPOCH_ORDER[1:]):
                    if e not in cur_s:
                        continue
                    Me, _, ane = cur_s[e]
                    cols_ = Mp.columns.intersection(Me.columns)
                    # bins where BOTH displayed curves (this epoch and pre, in the figure's own weighting) pass the
                    # coverage rule; significant bins outside them are drawn LIGHTER, over a faint dashed curve
                    # (Priya 10-17: "if it's significant even with fewer sessions, plot it, in a different shade")
                    shown = np.ones(len(cols_), bool)
                    if e in cur and "pre" in cur:
                        shown = (cur[e][0].mean().reindex(cols_).notna()
                                 & cur["pre"][0].mean().reindex(cols_).notna()).to_numpy()
                    mask = cluster_permutation(np.vstack([Mp[cols_].to_numpy(float), Me[cols_].to_numpy(float)]),
                                               np.r_[np.zeros(len(Mp), bool), np.ones(len(Me), bool)],
                                               np.r_[anp, ane], n_perm=CLUSTER_PERM)
                    if mask.any():                       # one solid bar per significant run of bins
                        xs = cols_.to_numpy(float)
                        dx = np.median(np.diff(xs)) if len(xs) > 1 else 1.0
                        for part, al in ((mask & shown, 0.5), (mask & ~shown, 0.18)):
                            edges = np.flatnonzero(np.diff(np.r_[0, part.astype(int), 0]))
                            for s0, s1 in zip(edges[::2], edges[1::2]):
                                ax.plot([xs[s0] - dx / 2, xs[s1 - 1] + dx / 2], [0.975 - 0.04 * k] * 2,
                                        color=EPOCH_LINE[e], lw=2.6, alpha=al, transform=ax.get_xaxis_transform(),
                                        solid_capstyle="butt", zorder=4)
            for e in EPOCH_ORDER:
                if e not in cur or e == "pre" and len(cur) == 1:
                    continue
                M, am, _ = cur[e]                               # rows (sessions or animals) x x
                x = M.columns.to_numpy(float)
                f = (lambda v: _circ_smooth(v)) if smooth else (lambda v: v)
                for c_ in am.values():                          # faint = each animal's mean
                    ax.plot(x, f(c_.reindex(M.columns).to_numpy(float)), color=ecol[e], lw=0.6, alpha=0.3, zorder=1)
                mu = f(M.mean().to_numpy(float))
                lo_cov = _animal_delta_curves(D, xcol, col, p, weight, raw=raw, cover=False).get(e)
                if lo_cov is not None:                          # low-coverage bins: faint dashed, outside the rule
                    mlo = f(lo_cov[0].mean().reindex(M.columns).to_numpy(float))
                    show = np.isnan(mu) & np.isfinite(mlo)
                    if show.any():
                        grow = show | np.r_[show[1:], False] | np.r_[False, show[:-1]]   # join onto the solid line
                        ax.plot(x, np.where(grow, np.where(np.isnan(mu), mlo, mu), np.nan), color=ecol[e], lw=1.0,
                                ls=(0, (2, 1.5)), alpha=0.5, zorder=2.5)
                n_sess = int(D[(D.pos == p) & (D.epoch == e)][["animal", "date"]].drop_duplicates().shape[0])
                lab = f"{e} (n={len(M) if weight == 'session' else n_sess} sessions)"
                ax.plot(x, mu, color=ecol[e], lw=1.6, zorder=3, label=lab)
                se = np.zeros_like(mu)
                if len(M) > 1:
                    se = f((M.std(ddof=1) / np.sqrt(M.notna().sum())).to_numpy(float))
                    ax.fill_between(x, mu - se, mu + se, color=ecol[e], alpha=0.2, lw=0, zorder=2)
                if np.isfinite(mu - se).any():
                    rlo[i] = min(rlo.get(i, np.inf), np.nanmin(mu - se))
                    rhi[i] = max(rhi.get(i, -np.inf), np.nanmax(mu + se))
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=9, color=_pos_color(p, "L"))
            if j == 0:
                ax.set_ylabel("\n".join(textwrap.wrap(ylab, 26)), fontsize=7.5)
        axs[i][0].legend(fontsize=6, frameon=False)
        if i in rlo:                                    # y range from the bold means +- SEM (faint lines may clip)
            lo_, hi_ = (rlo[i], rhi[i]) if raw else (min(rlo[i], 0.0), max(rhi[i], 0.0))
            pad = 0.08 * (hi_ - lo_ or 1.0)
            axs[i][0].set_ylim(lo_ - pad, hi_ + pad)
    for ax in axs[-1]:
        ax.set_xlabel(xlabel, fontsize=7.5)
    full = (note + ("; SESSION-WEIGHTED: bold = mean over SESSIONS (each = its change from its animal's own pre-stroke "
                    "curve), band = SEM over sessions (animals with more sessions weigh more)" if weight == "session" else
                    "; ANIMAL-WEIGHTED: bold = mean over animals (each = its sessions' mean change from its own pre-stroke "
                    "curve), band = SEM over animals") + "; faint = individual animals (y range set by the bold means, so faint lines "
                    "can run off the panel)" + (
                    "; bars at top = cluster-permutation p < 0.05 vs pre (test on SESSIONS whatever the weighting; labels "
                    "permuted within animal; one bar row per epoch: acute / subacute / chronic); LOW COVERAGE (fewer "
                    f"than {MIN_COVER_FRAC:.0%} / {MIN_COVER_N} of the rows carry the bin in this weighting): faint "
                    "dashed curve and lighter bar"))
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(full, 150)), fontsize=9)
    _save(plt, fig, path)


CLUSTER_PERM = 500                       # permutations per cluster test (cohort trajectory figures)
COHORT_SCALARS = ("contact_frac", "reach_frac", "protrusion_px", "peak_angle_deg", "delta_angle_deg", "latency_ms",
                  "licks_per_resp", "neighbor_frac", "vel_protrude")


STAR = {"both": "*", "MWU only": "(*)", "bootstrap only": "b"}


def _stars(ax, st: pd.DataFrame, x: np.ndarray) -> None:
    """Significance marks above each position, one row per post epoch: * = session MWU (BH-FDR q < 0.05) AND the
    animal-weighted bootstrap CI excludes 0; (*) = MWU only; b = bootstrap only."""
    _mark_rows(ax, 3)
    for k, e in enumerate(EPOCH_ORDER[1:]):
        for _, r in st[st.epoch == e].iterrows():
            mk = STAR.get(r.agree)
            if mk and r.pos in POS_ORDER:
                ax.text(x[POS_ORDER.index(r.pos)] + (k - 1) * 0.12, 0.99 - 0.075 * k, mk, color=EPOCH_LINE[e],
                        ha="center", va="top", fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())


def fig_cohort_scalars(Pm: pd.DataFrame, path: Path, title: str, jaw: pd.DataFrame | None = None,
                       weight: str = "session", stats: pd.DataFrame | None = None) -> None:
    """Per-position metrics as change from each animal's own pre-stroke mean: x = position, line = epoch; bold = mean
    over animals, error = SEM over animals, faint markers = each animal. Optional last panel: jaw steering FAR_DIR."""
    ecol = EPOCH_LINE
    ms = [m for m in COHORT_SCALARS if m in Pm]
    n = len(ms) + (1 if jaw is not None else 0)
    plt = _plt()
    nc = 3
    fig, axs = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(4.6 * nc, 3.0 * np.ceil(n / nc)), squeeze=False)
    for ax in axs.flat[n:]:
        ax.axis("off")
    x = np.arange(len(POS_ORDER))
    for ax, m in zip(axs.flat, ms):
        base = Pm[Pm.epoch == "pre"].groupby(["animal", "pos"])[m].mean()
        d = Pm.assign(_d=Pm[m] - base.reindex(pd.MultiIndex.from_frame(Pm[["animal", "pos"]])).to_numpy())
        A = d.groupby(["epoch", "animal", "pos"])._d.mean()            # one value per animal x epoch x position
        Ssn = d.groupby(["epoch", "animal", "date", "pos"])._d.mean()     # one value per session x position
        for k, e in enumerate(EPOCH_ORDER[1:]):
            if e not in A.index.get_level_values(0):
                continue
            Ma = A.loc[e].unstack("pos").reindex(columns=list(POS_ORDER))                  # animals
            M = Ssn.loc[e].unstack("pos").reindex(columns=list(POS_ORDER)) if weight == "session" else Ma
            xo = x + (k - 1) * 0.12
            for _, r in Ma.iterrows():                                   # faint = each animal
                ax.plot(xo, r.to_numpy(float), "o", color=ecol[e], ms=2.5, alpha=0.35, zorder=1)
            se = M.std(ddof=1) / np.sqrt(M.notna().sum()) if len(M) > 1 else M.mean() * np.nan
            lab = f"{e} (n={int(Ssn.loc[e].index.droplevel('pos').unique().size)} sessions)"
            ax.errorbar(xo, M.mean().to_numpy(float), yerr=se.to_numpy(float), fmt="o-", color=ecol[e], ms=4,
                        lw=1.4, capsize=2, elinewidth=0.8, zorder=3, label=lab)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.axvline(2.5, color="0.85", lw=0.8, zorder=0)
        if stats is not None:
            _stars(ax, stats[stats.metric == m], x)
        ax.set_xticks(x, [pos_disp(q).replace(" ", "\n") for q in POS_ORDER], fontsize=6.5)
        for t_, q in zip(ax.get_xticklabels(), POS_ORDER):
            t_.set_color(_pos_color(q, "L"))
        ax.set_title(LABEL[m] + " -- change vs pre", fontsize=8)
        ax.set_ylabel(f"change vs pre: {LABEL[m]}", fontsize=6.5)
        ax.tick_params(labelsize=7)
    if jaw is not None and len(jaw):
        ax = axs.flat[len(ms)]
        c = "lr_wide_far_dir"
        base = jaw[jaw.epoch == "pre"].groupby("animal")[c].mean()
        d = jaw.assign(_d=jaw[c] - jaw.animal.map(base))
        A = d.groupby(["epoch", "animal"])._d.mean()
        for k, e in enumerate(EPOCH_ORDER):
            if e not in A.index.get_level_values(0):
                continue
            v = A.loc[e] if weight == "animal" else d[d.epoch == e]._d
            ax.plot(np.full(len(v), k) + np.linspace(-0.08, 0.08, len(v)), v.to_numpy(float), "o", color=ecol[e],
                    ms=3, alpha=0.4)
            ax.errorbar(k, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0, fmt="s",
                        color=ecol[e], ms=6, capsize=3)
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        if stats is not None:                           # cohort marks for the jaw readout (far pair)
            _mark_rows(ax, 1)
            for k, e in enumerate(EPOCH_ORDER):
                r = stats[(stats.metric == c) & (stats.pos == "far_pair") & (stats.epoch == e)]
                if len(r) and STAR.get(r.agree.iloc[0]):
                    ax.text(k, 0.98, STAR[r.agree.iloc[0]], color=ecol[e], ha="center", va="top", fontsize=11,
                            fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(EPOCH_ORDER)), EPOCH_ORDER, fontsize=7)
        ax.set_title("jaw steering FAR_DIR -- change vs pre", fontsize=8)
        ax.set_ylabel("change vs pre: jaw FAR_DIR (px)", fontsize=6.5)
        ax.tick_params(labelsize=7)
    axs.flat[0].legend(fontsize=6.5, frameon=False)
    note = ("licks: ALL POST-CUE licks (measured peaks for kinematics); change from each animal's OWN pre-stroke mean at "
            "that position; " + ("SESSION-WEIGHTED: bold = mean +- SEM over sessions (animals with more sessions weigh "
            "more)" if weight == "session" else "ANIMAL-WEIGHTED: bold = mean +- SEM over animals") +
            "; faint = individual animals. Jaw panel: every far-spout trial (orofacial_jaw_lateral)" + (
            ". Marks (one row per epoch): * = session Mann-Whitney BH-FDR q < 0.05 AND animal-weighted bootstrap CI "
            "excludes 0; (*) = MWU only; b = bootstrap only" if stats is not None else ""))
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 150)), fontsize=9)
    _save(plt, fig, path)


def main(argv=None) -> int:
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--all-trials", action="store_true",
                    help="control: no engagement gate; writes tables/prepost_alltrials, figures/prepost_alltrials")
    a = ap.parse_args(argv)
    global ENGAGED_ONLY
    if a.all_trials:
        os.environ["OROFACIAL_ALL_TRIALS"] = "1"
        ENGAGED_ONLY = False
    rv = PathResolver()
    base = oo.root(rv) / a.run / "cohort"
    sfx = "" if ENGAGED_ONLY else "_alltrials"
    tables, figs = base / "tables" / f"prepost{sfx}", base / "figures" / f"prepost{sfx}"
    writeguard.assert_writable(base)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    S, Pm, PH, PHN, LS, WK, CT, lat = collect(rv, a.run, a.jobs)
    few = Pm[Pm.n_licks < MIN_LICKS_SESSION_POS][["animal", "date", "pos"]]
    print(f"min licks per session x position = {MIN_LICKS_SESSION_POS}: {len(few)} of {len(Pm)} session-positions dropped")
    if len(few):
        key = lambda d: d.animal + "|" + d.date.astype(str) + "|" + d.pos            # noqa: E731
        drop = set(key(few))
        Pm = Pm[~key(Pm).isin(drop)]
        PH, PHN, CT = (d[~key(d).isin(drop)] for d in (PH, PHN, CT))
    PHa = PH[PH.set == "postcue"]
    SP = spout_angles(Pm)
    A, D = epoch_tables(S)
    pd.set_option("display.width", 250)
    print(f"{len(S)} curated sessions with pose outputs:")
    print(S.groupby(["animal", "epoch"]).size().unstack(fill_value=0).reindex(columns=list(EPOCH_ORDER), fill_value=0))
    print("\nPER-ANIMAL epoch means (rule 8 -- read before the deltas):")
    print(A.set_index(["animal", "epoch"])[["n_sessions", *METRICS]].round(3).to_string())
    print("\nchange vs pre (animal-weighted; CI only with >= 2 animals):")
    print(D.round(4).to_string(index=False))
    S.to_csv(tables / "session_metrics.csv", index=False)
    Pm.to_csv(tables / "position_metrics.csv", index=False)
    PH.to_csv(tables / "lick_phase.csv", index=False)
    PHN.to_csv(tables / "lick_phase_normalised.csv", index=False)
    LS.to_csv(tables / "lickset_metrics.csv", index=False)
    WK.to_csv(tables / "lick_walk.csv", index=False)
    CT.to_csv(tables / "cue_aligned_traces.csv", index=False)
    A.to_csv(tables / "epoch_by_animal.csv", index=False)
    D.to_csv(tables / "epoch_deltas.csv", index=False)
    animals = ", ".join(f"{x} {int((S.animal == x).sum())}" for x in sorted(S.animal.unique()))
    gate_txt = ("ENGAGED trials only (behaviour pipeline's reference-position gate)" if ENGAGED_ONLY else
                "ALL trials (no engagement gate -- control)")
    if ENGAGED_ONLY and "engagement_gate" in S:
        print(f"engagement gate: {int(S.n_trials_used.sum())} of {int(S.n_trials_all.sum())} trials kept")
        for _, r_ in S[S.engagement_gate != "reference"].iterrows():
            print(f"  {r_.label}: {r_.engagement_gate}")
    t = f"Orofacial kinematics, pre vs post stroke -- {a.run} -- {gate_txt} -- sessions: {animals}"
    fig_epochs(A, D, figs / "epochs.png", t)
    fig_timecourse(S, figs / "timecourse.png", t)
    fig_by_position(Pm, lat, figs / "by_position.png", t)
    fig_mismatch(Pm, lat, figs / "mismatch.png", t)
    fig_angle_tuning(Pm, lat, figs / "angle_tuning.png", t)
    for name, ms in POS_GROUPS.items():
        fig_epochs_by_position(Pm, lat, ms, figs / f"epochs_by_position_{name}.png", t)
        fig_timecourse_by_position(Pm, ms, figs / f"timecourse_by_position_{name}.png", t)
    _phase_fig(PHa, lat, figs / "lick_phase.png", t, portrait=False)
    _phase_fig(PHa, lat, figs / "phase_portrait.png", t, portrait=True)
    for mode in PHASE_MODES:
        which = "first multi-lick bout per trial" if mode == "first_multi" else "every multi-lick bout"
        for col, ylab, what in (("angle_deg", "tongue angle (deg)",
                                 "tongue angle (+ = image right = L); IN-MOUTH FRAMES EXCLUDED (< 20 px) -- reads: given the tongue is out, is its trajectory different"),
                                ("angle_mouth0_deg", "tongue angle (deg)",
                                 "tongue angle; IN-MOUTH FRAMES INCLUDED as 0 deg (stroke_orofacial convention) -- reads: trajectory AND how much of the time the tongue is out (lick rate / duration also enter)"),
                                ("delta_angle_deg", "angle - spout angle (deg)",
                                 "angle ERROR vs the spout; IN-MOUTH FRAMES EXCLUDED (< 20 px)"),
                                ("protrusion_px", "protrusion (px)", "protrusion; IN-MOUTH FRAMES INCLUDED (0 px)")):
            fig_phase_norm(PHN, lat, col, mode, ylab, figs / f"phase_norm_{col}_{mode}.png", t, spout=(
                SP if col in ("angle_deg", "angle_mouth0_deg") else None), note=
                           f"licks: POST-CUE licks in multi-lick bouts ({which}); {what} vs NORMALISED lick phase "
                           f"(stroke_orofacial method); epoch mean +- SEM "
                           f"over sessions; each peak-to-peak interval interpolated onto the phase grid")
            if col in ("protrusion_px", "angle_deg"):
                fig_phase_norm(PHN, lat, col, mode, f"{ylab} -- change vs pre", figs / f"phase_norm_{col}_{mode}_minus_pre.png",
                               t, f"licks: POST-CUE licks in multi-lick bouts ({which}); {what} MINUS this animal's mean "
                               f"pre-stroke curve, normalised lick phase; "
                               "epoch mean +- SEM over sessions (pre = 0 +- its session spread)", delta_pre=True)
    for name, desc in LICK_SETS.items():
        _per_animal_figure(LS[LS.set == name], LICKSET_METRICS, lat, figs / f"by_position_licks_{name}.png", t,
                           f"per-lick metrics -- {desc}; measured peaks; epoch mean +- SEM over sessions")
    fig_lick12(Pm, lat, figs / "lick1_vs_lick2.png", t)
    # Family 8.2 (stroke_orofacial ipsi_shift): cue-aligned continuous lateral position / angle, raw and minus pre
    for var, ylab, inm in (("lat_px", "lateral tongue position (px; + = ipsi)", "IN-MOUTH FRAMES INCLUDED (at 0) -- reads: trajectory AND how much of the time the tongue is out (lick rate / duration also enter)"),
                           ("angle_deg", "tongue angle (deg; + = ipsi)", "IN-MOUTH FRAMES EXCLUDED (< 20 px) -- reads: given the tongue is out, is its trajectory different"),
                           ("angle_mouth0_deg", "tongue angle (deg; + = ipsi)", "IN-MOUTH FRAMES INCLUDED as 0 deg -- reads: trajectory AND how much of the time the tongue is out (lick rate / duration also enter)")):
        Cv = CT[CT["var"] == var].rename(columns={"value": var})
        what = {"lat_px": "lateral tongue position", "angle_deg": "tongue angle",
                "angle_mouth0_deg": "tongue angle (in-mouth = 0)"}[var]
        fig_lick_phase_value(Cv, lat, var, ylab, figs / f"cue_{var}.png", t,
                             f"NOT lick-selected: EVERY FRAME of every trial at each position, cue-aligned; {what} "
                             f"({CUE_BIN_MS:.0f}-ms bins; + = toward the "
                             f"IPSILESIONAL side); epoch mean +- SEM over sessions; {inm}" + ("; dashed = spout angle" if var != "lat_px" else ""),
                             xlabel="ms from cue", spout=(SP if var != "lat_px" else None), spout_ipsi=True)
        fig_lick_phase_value(Cv, lat, var, f"{ylab} -- change vs pre", figs / f"cue_{var}_minus_pre.png", t,
                             f"stroke_orofacial Family 8.2, NOT lick-selected (every frame of every trial): cue-aligned "
                             f"{what} MINUS this animal's mean pre-stroke curve "
                             f"at the same position; pre = 0 +- its session spread; epoch mean +- SEM; {inm}",
                             delta_pre=True, xlabel="ms from cue")
    cols12 = ("l1_angle_deg", "l2_angle_deg", "l1_prot_px", "l2_prot_px")
    _per_animal_figure(delta_from_pre(Pm, cols12), tuple(f"{c}_dpre" for c in cols12), lat,
                       figs / "lick1_lick2_minus_pre.png", t,
                       "licks: FIRST and SECOND post-cue lick of each trial; peak angle / protrusion as CHANGE from this "
                       "animal's pre-stroke mean at the "
                       "same position (stroke_orofacial Family 8.1); pre = 0 +- its session spread; epoch mean +- SEM")
    for which in ("postcue", "first", "second"):
        fig_xy_trajectory(PH, Pm, lat, figs / f"lick_xy_trajectory_{which}.png", t, which=which)
    _per_animal_figure(Pm, ("walk_err_reduction", "walk_d_angle", "walk_n_trials"), lat, figs / "lick_walk_paired.png",
                       t, "licks: post-cue lick 1 vs lick 3 of the SAME trial; does the tongue walk toward the target "
                       "WITHIN a trial? "
                       "(measured peaks): + error reduction = lick 3 closer to the spout angle; per-session median, "
                       "epoch mean +- SEM over sessions")
    fig_walk(WK, lat, "peak_angle_deg", "angle at peak protrusion (deg; + = L)", figs / "lick_walk_angle.png", t,
             "licks: POST-CUE licks by lick number in the trial (lick n exists only in trials with >= n licks); peak "
             "tongue angle (does the tongue walk toward the target?); dashed = spout "
             "angle; per-lick-number median per session, epoch mean +- SEM over sessions; measured peaks")
    fig_walk(WK, lat, "delta_angle_deg", "angle - spout (deg)", figs / "lick_walk_angle_error.png", t,
             "licks: POST-CUE licks by lick number; angle ERROR (peak angle - spout angle; 0 = on target); epoch mean "
             "+- SEM")
    WK5 = WK[WK.lick_n <= 5]
    if "max_angle_deg" in WK5:
        fig_walk(WK5, lat, "max_angle_deg", "max angle in lick (deg; + = L)", figs / "lick_walk_max_angle.png", t,
                 "licks: POST-CUE licks 1-5 by lick number; MAX signed angle DURING each lick, only frames with the tongue "
                 ">= 50 % of that lick's peak distance (near-mouth frames, whose angle is artificially large, excluded); "
                 "dashed = spout angle; epoch mean +- SEM")
        fig_walk(WK5, lat, "max_delta_angle_deg", "max angle - spout (deg)", figs / "lick_walk_max_angle_error.png",
                 t, "licks: POST-CUE licks 1-5 by lick number; MAX angle during the lick minus the spout angle (0 = on "
                 "target; tongue >= 50 % of its peak distance only); epoch mean +- SEM")
    fig_walk(WK, lat, "protrusion_px", "protrusion (px)", figs / "lick_walk_protrusion.png", t,
             "licks: POST-CUE licks by lick number; peak protrusion; epoch mean +- SEM over sessions")
    for col, ylab in (("protrusion_px", "protrusion change vs pre (px)"), ("angle_deg", "angle change vs pre (deg)")):
        fig_lick_phase_value(PHa, lat, col, ylab, figs / f"lick_phase_{col}_minus_pre.png", t,
                             f"licks: ALL POST-CUE licks (measured peaks); lick-peak-aligned {col.split('_')[0]} MINUS "
                             f"this animal's mean pre-stroke curve; epoch "
                             "mean +- SEM over sessions (pre = 0 +- its session spread); IN-MOUTH FRAMES "
                             + ("INCLUDED (0 px)" if col == "protrusion_px" else "EXCLUDED (< 20 px)"), delta_pre=True)
    fig_lick_phase_value(PHa, lat, "angle_deg", "tongue angle (deg)", figs / "lick_phase_angle.png", t,
                         "licks: ALL POST-CUE licks (measured peaks); lick-peak-aligned tongue angle (+ = image right = "
                         "L / ipsi positions); dashed = spout angle; mean lick (measured "
                         f"peaks), epoch mean +- SEM over sessions; IN-MOUTH FRAMES EXCLUDED (< {ANGLE_MIN_PX:.0f} px)",
                         spout=SP)
    fig_lick_phase_value(PHa, lat, "delta_angle_deg", "angle - spout angle (deg)", figs / "lick_phase_angle_delta.png",
                         t, "licks: ALL POST-CUE licks (measured peaks); lick-peak-aligned angle ERROR: tongue angle "
                         "minus that position's spout angle (0 = on "
                         "target; + = toward image right / L), epoch mean +- SEM over sessions; IN-MOUTH FRAMES EXCLUDED (< 20 px)")
    # ---- COHORT minus-pre summaries (Priya 10-17: "a combined figure of the delta minus pre-stroke analyses ...
    # all the animals"), under figures/prepost/cohort_*.png
    CTw = CT.pivot_table(index=["animal", "date", "epoch", "pos", "t_ms"], columns="var", values="value").reset_index()
    ST = cohort_stats(Pm, base, tables)
    for wgt, suf in (("session", ""), ("animal", "_animalw")):
        _cohort_figs(CTw, PHa, PHN, Pm, base, figs, t, wgt, suf, PH=PH, stats=ST)
    (tables / "provenance.json").write_text(json.dumps({"run": a.run, "n_sessions": int(len(S)),
                                                        "code_commit": oo.code_commit(),
                                                        "sessions": S.label.tolist()}, indent=1), encoding="utf-8")
    print(f"\ntables -> {tables}\nfigures -> {figs}")
    return 0


def _path_sig(W: pd.DataFrame, p: str, e: str, xc: str, yc: str) -> pd.Series | None:
    """Boolean per t_ms: the epoch's path differs from pre in x(t) OR y(t) (cluster permutation on per-session changes
    from each animal's own pre path, labels permuted within animal, each coordinate at ALPHA/2)."""
    from wfield_local.orofacial_stats import ALPHA
    g = W[(W.pos == p) & W.epoch.isin(["pre", e])]
    if (g.epoch == e).sum() == 0:
        return None
    out = None
    for c in (xc, yc):
        pv = g.pivot_table(index=["animal", "date", "epoch"], columns="t_ms", values=c)
        an = pv.index.get_level_values(0).to_numpy()
        ep = pv.index.get_level_values(2).to_numpy()
        base = pv[ep == "pre"].groupby(level=0).mean()
        ch = pv.to_numpy(float) - base.reindex(an).to_numpy(float)
        m = cluster_permutation(ch, ep == e, an, n_perm=CLUSTER_PERM, alpha=ALPHA / 2)
        out = m if out is None else (out | m)
    return pd.Series(out, index=pv.columns)


def fig_cohort_paths(PH: pd.DataFrame, Pm: pd.DataFrame, path: Path, title: str, portrait: bool = False,
                     within: bool = False, inmouth: str = "incl") -> None:
    """COHORT lick trajectory per position, pre vs each post epoch overlaid: x-y path of the tongue tip (camera view) or
    the phase portrait (protrusion speed vs protrusion); mean over SESSIONS (all animals), band = SEM over sessions
    perpendicular to the path; licks: ALL POST-CUE licks, measured peaks, +-PORTRAIT_HALF_MS around the peak."""
    W = PH[PH.t_ms.abs() <= PORTRAIT_HALF_MS]
    plt = _plt()
    fig, axs = plt.subplots(1, len(POS_ORDER), figsize=(3.0 * len(POS_ORDER), 4.6), squeeze=False,
                            sharex=not portrait, sharey=True)
    if inmouth == "incl":
        xc, yc = ("protrusion_px", "velocity_px_s") if portrait else ("lat_px", "out_px")
    else:                                                # tongue OUT frames only (>= ANGLE_MIN_PX from the mouth)
        xc, yc = ("protrusion_out_px", "velocity_out_px_s") if portrait else ("lat_out_px", "out_out_px")
    for j, p in enumerate(POS_ORDER):
        ax = axs[0][j]
        for e in EPOCH_ORDER:
            g = W[(W.pos == p) & (W.epoch == e)]
            if not len(g):
                continue
            px = g.pivot_table(index=["animal", "date"], columns="t_ms", values=xc)
            py = g.pivot_table(index=["animal", "date"], columns="t_ms", values=yc)
            if within:
                # WITHIN-ANIMAL (Priya 10-17): every session minus ITS animal's mean pre path, added to the cohort pre path,
                # so a difference between epochs is within-animal (PS93's pre-existing deficit cannot shift it)
                gp = W[(W.pos == p) & (W.epoch == "pre")]
                prex = gp.pivot_table(index=["animal", "date"], columns="t_ms", values=xc)
                prey = gp.pivot_table(index=["animal", "date"], columns="t_ms", values=yc)
                ax_, ay_ = prex.groupby(level=0).mean(), prey.groupby(level=0).mean()
                px = (px - ax_.reindex(px.index.get_level_values(0)).set_axis(px.index)) + prex.mean()
                py = (py - ay_.reindex(py.index.get_level_values(0)).set_axis(py.index)) + prey.mean()
            mx, my = px.mean(), py.mean()
            sx, sy = px.std(ddof=1) / np.sqrt(len(px)), py.std(ddof=1) / np.sqrt(len(py))
            X, Y = (mx, my / 1000) if portrait else (mx, -my)
            SX, SY = (sx, sy / 1000) if portrait else (sx, sy)
            ax.plot(X.to_numpy(), Y.to_numpy(), color=EPOCH_LINE[e], lw=1.6, zorder=3, label=f"{e} (n={len(px)} sessions)")
            if 0.0 in X.index:
                ax.plot(X[0.0], Y[0.0], "o", color=EPOCH_LINE[e], ms=3.5, zorder=4)
            if within and e != "pre":
                sig = _path_sig(W, p, e, xc, yc)
                if sig is not None and sig.any():
                    ts = sig.index[sig.to_numpy()]
                    ts = ts[np.isclose(np.mod(ts.to_numpy(float), 12.0), 0.0)]   # one circle per 12 ms
                    ax.plot(X.reindex(ts).to_numpy(), Y.reindex(ts).to_numpy(), "o", ms=4.5, mfc="white",
                            mec=EPOCH_LINE[e], mew=1.2, zorder=5)
            if len(px) > 1:
                _sem_ribbon(ax, X.to_numpy(), Y.to_numpy(), SX.to_numpy(), SY.to_numpy(), EPOCH_LINE[e], alpha=0.18)
        if not portrait:
            ax.plot(0, 0, "k+", ms=8)
            sp = Pm[Pm.pos == p]
            if len(sp):
                d, an = sp.spout_dist_px.mean(), np.deg2rad(sp.spout_angle_deg.mean())
                ax.plot(d * np.sin(an), -d * np.cos(an), "x", color="0.4", ms=8, mew=2)
            ax.set_aspect("equal")
        else:
            ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(pos_disp(p), fontsize=9, color=_pos_color(p, "L"))
        ax.set_xlabel("protrusion (px)" if portrait else "lateral (px; + = L)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    axs[0][0].set_ylabel("protrusion speed (px/ms)" if portrait else "from mouth (px; camera view)", fontsize=8)
    axs[0][0].legend(fontsize=6, frameon=False)
    what = ("phase portrait (protrusion speed vs protrusion)" if portrait else "x-y tongue-tip path, as the camera sees "
            "it (+ = mouth, x = spout)")
    note = (f"COHORT lick trajectory, pre vs post epochs: {what}; licks: ALL POST-CUE licks (measured peaks), "
            f"+-{PORTRAIT_HALF_MS:.0f} ms around the peak; mean over SESSIONS, band = SEM over sessions; dot = peak; "
            + ("IN-MOUTH FRAMES INCLUDED" if inmouth == "incl" else f"IN-MOUTH FRAMES EXCLUDED (tongue out >= "
                                                                   f"{ANGLE_MIN_PX:.0f} px only)")
            + ("; open circles = where that epoch's path differs from pre (cluster permutation on the x(t) and y(t) "
               "changes, sessions, labels permuted within animal, p < 0.025 each)" if within else "")
            + ("; WITHIN-ANIMAL: each post-epoch path = the cohort pre path + the mean of each "
                                          "session's change from ITS animal's own pre path" if within else
                                          "; raw paths pooled over animals"))
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 150)), fontsize=9)
    _save(plt, fig, path)


STATS_METHODS = """# How the statistics were run (orofacial pre- vs post-stroke figures)

Generated by `wfield_local/orofacial_cohort.py` / `wfield_local/orofacial_stats.py`; design follows
stroke_orofacial's `docs/STATS_PIPELINE.md`, chosen by Priya 2026-10-17. All numbers PRELIMINARY until every animal's
sessions are in.

* **Trials = ENGAGED only** (Priya 10-17): the behaviour pipeline's reference-position gate
  (`spout_behavior.reference_engagement`: judged at close ipsi / close mid only, terminal non-recovering collapse, so a
  far-position motor deficit is not deleted as "disengagement"). Sessions whose trial ids disagree with the behaviour
  table stay ungated and are listed in the run log. Ungated control: `figures/prepost_alltrials`.
* **Unit = the session.** Each session's value (per spout position) is first expressed as its CHANGE from that animal's
  own pre-stroke mean at the same position, so animal-to-animal differences in level do not enter.
* **Headline (cohort scalar figures, `*` / `(*)`):** two-sided Mann-Whitney U, post-epoch sessions vs pre-stroke
  sessions (all animals pooled), Benjamini-Hochberg FDR within each metric across positions x epochs, q < 0.05.
  Trial-level tests are NOT used (stroke_orofacial measured 39-46 % false positives for them under the null).
* **Robustness (`*` vs `(*)` / `b`):** the animal-weighted bootstrap of this repo (`analysis_kit.boot_delta`: animals
  resampled, then sessions within animals; each animal weighted equally), 95 % CI. `*` = both agree; `(*)` = Mann-Whitney
  only (often carried by the animals with many sessions); `b` = bootstrap only (consistent across animals, but too few /
  too spread sessions for the Mann-Whitney).
* **Per-animal figures (`epochs_by_position_*`, `*` in an animal's colour):** that animal's post-epoch sessions vs its
  own pre-stroke sessions, two-sided Mann-Whitney on sessions, BH-FDR within each metric across animals x positions x
  epochs.
* **Trajectories (bars along the top of cohort trajectory figures, session-weighted versions):** cluster permutation.
  Per time / phase bin, Welch t of post-epoch session change curves vs pre-session change curves; clusters = contiguous
  bins with |t| above the two-sided p < 0.05 threshold and one sign; cluster mass = sum |t|; null = the largest cluster
  mass after permuting epoch / pre labels WITHIN each animal (500 permutations); clusters with p < 0.05 are drawn.
* **Caveats.** Sessions are not independent within an animal, which the within-animal permutation and the bootstrap
  partly address. Small chronic shifts can reflect slow drift (head position / camera framing over months) rather than
  the stroke: compare against the pre-stroke drift between early (June) and late (August) baseline sessions.
"""


def cohort_stats(Pm: pd.DataFrame, base: Path, tables: Path) -> pd.DataFrame:
    """Session MWU + animal-weighted bootstrap for every cohort scalar (and jaw FAR_DIR); writes stats.csv and a
    plain-language RESULTS.md of the significant cells."""
    ST = session_change_tests(Pm, COHORT_SCALARS)
    jaw_csv = base / "tables" / ("jaw_lateral" if ENGAGED_ONLY else "jaw_lateral_alltrials") / "sessions.csv"
    if jaw_csv.exists():
        J = pd.read_csv(jaw_csv, dtype={"date": str}).assign(pos="far_pair")
        ST = pd.concat([ST, session_change_tests(J, ["lr_wide_far_dir", "angle_far_dir"])], ignore_index=True)
    out = tables / "stats"
    out.mkdir(parents=True, exist_ok=True)
    ST.to_csv(out / "change_vs_pre_stats.csv", index=False, float_format="%.4g")
    lines = ["# Change vs pre -- significant cells (auto-generated by `orofacial_cohort`; re-written every run)", "",
             "Headline: two-sided Mann-Whitney U on per-session changes (each session minus its animal's own pre-stroke "
             "mean) vs the pre sessions, BH-FDR within each metric across positions x epochs. Robustness: animal-weighted "
             "bootstrap (analysis_kit.boot_delta). `agree`: both / MWU only / bootstrap only. PRELIMINARY until every "
             "animal's sessions are in.", "",
             "| metric | position | epoch | change (mean over sessions) | sessions (animals) | q (FDR) | bootstrap 95% CI "
             "| agree |", "|---|---|---|---|---|---|---|---|"]
    for _, r in ST[ST.agree != "neither"].sort_values(["metric", "epoch", "pos"]).iterrows():
        lines.append(f"| {LABEL.get(r.metric, r.metric)} | {r.pos.replace('_', ' ')} | {r.epoch} | {r.mean_change:+.3g} | "
                     f"{r.n_sessions} ({r.n_animals}) | {r.q_fdr:.3g} | [{r.boot_lo:+.3g}, {r.boot_hi:+.3g}] | {r.agree} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "STATS_METHODS.md").write_text(STATS_METHODS, encoding="utf-8")
    print(f"stats: {int((ST.agree == 'both').sum())} cells significant by both tests, "
          f"{int((ST.agree == 'MWU only').sum())} MWU only, {int((ST.agree == 'bootstrap only').sum())} bootstrap only "
          f"-> {out}")
    return ST


#: matched in-mouth conventions (Priya 10-17: never mix excluded / included within one figure)
INMOUTH = {"excl": ("in-mouth EXCLUDED (tongue out, >= 20 px): given the tongue is out, is its trajectory different",
                    {"protrusion": "protrusion_out_px", "lateral": "lat_out_px", "angle": "angle_deg"}),
           "incl": ("in-mouth INCLUDED (in-mouth = 0): trajectory AND how much of the time the tongue is out",
                    {"protrusion": "protrusion_px", "lateral": "lat_px", "angle": "angle_mouth0_deg"})}


def _cohort_figs(CTw, PHa, PHN, Pm, base: Path, figs: Path, t: str, wgt: str, suf: str, PH=None,
                 stats: pd.DataFrame | None = None) -> None:
    """The COHORT minus-pre figures for one weighting (session / animal), each multi-row figure in ONE in-mouth convention
    (two files: _inmouth_excl / _inmouth_incl)."""
    PHf = PHN[PHN["mode"] == "first_multi"]
    for tag, (desc, c) in INMOUTH.items():
        fz = f"_inmouth_{tag}{suf}"
        fig_cohort_delta([(CTw, "t_ms", c["protrusion"], "protrusion (px)", False),
                          (CTw, "t_ms", c["lateral"], "lateral tongue position (px; + = ipsi)", False),
                          (CTw, "t_ms", c["angle"], "tongue angle (deg; + = ipsi)", False)],
                         figs / f"cohort_cue_minus_pre{fz}.png", t,
                         f"COHORT, change vs pre: cue-aligned protrusion / lateral position / angle (Family 8.2), NOT "
                         f"lick-selected (every frame of every trial); {desc}", "ms from cue", wgt)
        if PH is not None:
            PHf1 = PH[PH.set == "first"]
            fig_cohort_delta([(PHa, "t_ms", c["protrusion"], "protrusion (px)", False),
                              (PHa, "t_ms", c["lateral"], "lateral tip position (px; + = L)", False),
                              (PHa, "t_ms", c["angle"], "tongue angle (deg; + = L)", False)],
                             figs / f"cohort_lick_phase_minus_pre{fz}.png", t,
                             f"COHORT, change vs pre: lick-peak-aligned protrusion / lateral position / angle, licks: ALL "
                             f"POST-CUE licks (measured peaks); {desc}", "ms from lick peak", wgt)
            fig_cohort_delta([(PHf1, "t_ms", c["protrusion"], "protrusion (px), FIRST lick", False),
                              (PHf1, "t_ms", c["angle"], "tongue angle (deg; + = L), FIRST lick", False),
                              (PHa, "t_ms", c["protrusion"], "protrusion (px), ALL post-cue licks", False),
                              (PHa, "t_ms", c["angle"], "tongue angle (deg; + = L), ALL post-cue licks", False)],
                             figs / f"cohort_lick_phase_first_vs_all_minus_pre{fz}.png", t,
                             f"COHORT, change vs pre: lick-peak-aligned protrusion / angle -- licks: rows 1-2 the FIRST "
                             f"post-cue lick of each trial, rows 3-4 ALL post-cue licks (measured peaks); {desc}",
                             "ms from lick peak", wgt)
        fig_cohort_delta([(PHf, "phase", c["protrusion"], "protrusion (px)", True),
                          (PHf, "phase", c["angle"], "tongue angle (deg; + = L)", True)],
                         figs / f"cohort_phase_norm_minus_pre{fz}.png", t,
                         f"COHORT, change vs pre: protrusion / angle vs NORMALISED lick phase, licks: POST-CUE licks in the "
                         f"FIRST multi-lick bout per trial; {desc}", "lick phase (0 = between peaks, 0.5 = peak)", wgt)
    if PH is not None:
        fig_cohort_delta([(PHa, "t_ms", "lat_px", "lateral tip position (px; + = L), change vs pre", False),
                          (PHa, "t_ms", "out_px", "tip distance out along the mouth axis (px), change vs pre", False),
                          (PHa, "t_ms", "speed_abs_px_s", "protrusion SPEED |velocity| (px/s), RAW level", False, True),
                          (PHa, "t_ms", "speed_abs_px_s", "protrusion SPEED |velocity| (px/s), change vs pre", False)],
                         figs / f"cohort_trajectory_delta_minus_pre{suf}.png", t,
                         "COHORT lick TRAJECTORY: rows 1-2 change vs pre of the lateral and out-of-mouth tip position; "
                         "row 3 the RAW protrusion speed |velocity| per epoch (pre drawn as its own curve), row 4 its change "
                         "vs pre (slower = negative). Signed velocity is not shown: slower-out and slower-back cancel in it "
                         "(Priya 10-17). Bars on rows 3 and 4 are the same within-animal change test. Licks: ALL POST-CUE "
                         "licks (measured peaks); IN-MOUTH INCLUDED (tip at the mouth = 0)", "ms from lick peak", wgt)
        if wgt == "session":
            for within, wz in ((True, "_within"), (False, "")):
                for im in ("incl", "excl"):
                    fig_cohort_paths(PHa, Pm, figs / f"cohort_trajectory_xy{wz}_inmouth_{im}.png", t, within=within,
                                     inmouth=im)
                    fig_cohort_paths(PHa, Pm, figs / f"cohort_phase_portrait{wz}_inmouth_{im}.png", t, portrait=True,
                                     within=within, inmouth=im)
    jaw_csv = base / "tables" / ("jaw_lateral" if ENGAGED_ONLY else "jaw_lateral_alltrials") / "sessions.csv"
    jaw = pd.read_csv(jaw_csv, dtype={"date": str}) if jaw_csv.exists() else None
    fig_cohort_scalars(Pm, figs / f"cohort_metrics_minus_pre{suf}.png",
                       t + "\nCOHORT, change vs pre: per-position lick metrics", jaw, wgt,
                       stats if wgt == "session" else None)

if __name__ == "__main__":
    raise SystemExit(main())
