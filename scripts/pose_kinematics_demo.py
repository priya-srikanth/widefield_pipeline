"""First run of the ported kinematics on the PS93 0908 cue clip, DLC round 3 vs LP: clean -> spout frame ->
tongue v7 kinematics -> jaw per-trial -> tongue-jaw mismatch. A smoke test of the chain, NOT a result: every px
threshold is still the old rig's, and the clip holds 24 trials (4 per position).

    python -m scripts.pose_kinematics_demo --lp <LP csv for the clip>

The clip is 24 cue windows (cue -0.5 .. +3.5 s). Windows are joined with 2,200 NaN frames (8.8 s): the tongue
module slices each trial to +8 s and would otherwise overwrite the neighbouring window in its session arrays.
The spout frame comes from DLC's spout in these windows (all inside position strobe .. trial end).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import jaw_kinematics as jk
from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import tongue_jaw_mismatch as tm
from wfield_local import tongue_kinematics as tk

SEP = 2200


def daq_contacts_ms(animal: str, date: str, bounds: pd.DataFrame, rv=None) -> dict:
    """{trial_id: DAQ spout-contact onsets (ms from the cue) inside cue + 70 ms .. trial stop} -- every contact is a
    lick; incomplete licks make none. ``bounds`` = `trial_windows.trial_bounds`."""
    from wfield_local import dlc_frames
    from wfield_local.paths import PathResolver
    rv = rv or PathResolver()
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
    licks = dlc_frames.lick_onsets(animal, date, sid, rv)
    return {t: (licks[(licks >= c + 0.07) & (licks <= s)] - c) * 1000.0
            for t, c, s in zip(bounds.trial_id, bounds.cue_s, bounds.stop_s)}


def run(pose_csv: Path, idx: pd.DataFrame, frame: SF.SpoutFrame | None, stop_after_cue_s: dict | None = None,
        contacts_ms: dict | None = None, params_override: dict | None = None):
    """Tongue zero = the MOUTH (spout-frame origin); jaw keeps the data-driven baseline. With
    ``stop_after_cue_s`` ({trial_id: stop - cue, s}), every post-cue window ends at that trial's own stop."""
    # ``pose_csv``: a DLC-format CSV path, or the flat pose DataFrame itself (`orofacial_outputs` passes the window rows)
    df = (pose_csv if isinstance(pose_csv, pd.DataFrame) else oc.read_pose(pose_csv)).iloc[:len(idx)]
    clean = {"tongue": oc.clean_windows(df, idx, "tongue", sep=SEP, x0y0=tuple(frame.origin) if frame else None),
             "jaw": oc.clean_windows(df, idx, "jaw", sep=SEP)}
    cues = clean["tongue"][2]
    pos = idx.groupby("trial_k", sort=True).position.first().to_numpy()
    tid = idx.groupby("trial_k", sort=True).trial_id.first().to_numpy()
    fps = clean["tongue"][0].fps
    stops = np.array([cues[k] + stop_after_cue_s[t] * fps if stop_after_cue_s and t in stop_after_cue_s else np.nan
                      for k, t in enumerate(tid)])
    trials_df = pd.DataFrame({"trial_id": tid, "cue_frame": cues.astype(float), "position": pos, "stop_frame": stops})
    trials = tk.trials_from_frame(trials_df)
    sfr = tk.SpoutFrame(tuple(frame.origin), tuple(frame.ap_axis)) if frame is not None else None
    # OURS: the model's own spout, laid out like the cleaned tongue arrays (padded clip index), for reach accuracy
    T = tk.from_clean(clean["tongue"][0], trials, spout_frame=sfr, spout_xy=_spout_arrays(clean, df), contacts_ms=contacts_ms,
                      params_override=params_override)
    J, _ = jk.jaw_pertrial(clean["jaw"][0].y_final, clean["jaw"][0].fps, trials_df)
    M = tm.classify_from_clean(clean["jaw"][0], clean["tongue"][0], trials_df)
    return T, J, M, clean


#: Window keys of the out-of-response passes (`epoch_licks`): their STARTS move to the segment's anchor; their ends are
#: clipped to the segment end by `trial_windows.end_at_stop`.
_EPOCH_KEYS = ("trial_slice_win_ms", "lick12_detect_win_ms", "peak_velocity_detect_win_ms", "lick_count_detect_win_ms",
               "lick_count_apply_win_ms", "licking_dyn_apply_win_ms", "angle_apply_win_ms")
#: The post-cue pass detects from cue + 70 ms (stroke_orofacial parity) to the DAQ trial stop; the pre-cue pass ends here.
PRECUE_END_AFTER_CUE_MS = 70.0
#: Per-lick reach fields that assume the spout sits at the trial's position -- blanked outside strobe .. trial stop,
#: where the spout is moving or retracted.
_REACH_COLS = ("spout_angle_deg", "spout_dist_px", "delta_angle_deg", "tip_to_spout_px", "reach_frac")


def epoch_licks(clean: dict, idx: pd.DataFrame, frame: SF.SpoutFrame | None, df: pd.DataFrame,
                start_ms: np.ndarray, end_ms: np.ndarray, lick_s: np.ndarray | None = None,
                cue_s: dict | None = None, min_ms: float = 50.0) -> pd.DataFrame:
    """Licks in ONE segment per trial window, [start_ms, end_ms) from that trial's cue (arrays per ``trial_k``, NaN =
    skip) -- the SAME v7 detector + per-lick features as the post-cue pass, on the same cleaned trace, each segment
    re-anchored at its start and cut at its end (OURS, 2026-10-16). The post-cue pass keeps its stroke_orofacial
    windows (cue + 70 ms .. DAQ trial stop); these passes cover the rest of each window (`precue_licks`,
    `iti_licks`), so every lick peak lands in exactly one table.

    ``lick_s`` (all DAQ lick-contact onsets, s) + ``cue_s`` ({trial_id: cue DAQ s}) -> per-lick ``contact``. Returns
    per-lick rows like licks.csv with ``t_ms`` from the CUE and ``t_from_anchor_ms`` from the segment start."""
    ct = clean["tongue"][0]
    cues = clean["tongue"][2]
    fps = float(ct.fps)
    g = idx.groupby("trial_k", sort=True)
    tid, pos = g.trial_id.first().to_numpy(), g.position.first().to_numpy()
    start_ms, end_ms = np.asarray(start_ms, float), np.asarray(end_ms, float)
    ok = np.isfinite(start_ms) & np.isfinite(end_ms) & np.isfinite(cues) & (end_ms - start_ms >= min_ms)
    if not ok.any():
        return pd.DataFrame()
    trials_df = pd.DataFrame({"trial_id": tid[ok], "cue_frame": (cues + start_ms * fps / 1000.0)[ok],
                              "position": pos[ok], "stop_frame": (cues + end_ms * fps / 1000.0)[ok]})
    contacts = None
    if lick_s is not None and cue_s is not None:
        contacts = {}
        for t, a, e in zip(tid[ok], start_ms[ok], end_ms[ok]):
            c = cue_s.get(t, np.nan)
            if np.isfinite(c):
                rel = (np.asarray(lick_s, float) - c) * 1000.0
                contacts[t] = rel[(rel >= a) & (rel < e)] - a
    big = [0.0, 1.0e6]
    override = {k: big for k in _EPOCH_KEYS}
    override["bout_table"] = {"detect_win_ms": big, "plot_win_ms": big}
    T = tk.from_clean(ct, tk.trials_from_frame(trials_df), spout_frame=(
        tk.SpoutFrame(tuple(frame.origin), tuple(frame.ap_axis)) if frame is not None else None),
        spout_xy=_spout_arrays(clean, df), contacts_ms=contacts, params_override=override)
    L = T.per_lick.copy()
    if not len(L):
        return L
    L = L.rename(columns={"t_ms": "t_from_anchor_ms"})
    a_of, e_of = L.trial_id.map(dict(zip(tid[ok], start_ms[ok]))), L.trial_id.map(dict(zip(tid[ok], end_ms[ok])))
    L.insert(L.columns.get_loc("t_from_anchor_ms"), "t_ms", L.t_from_anchor_ms + a_of)
    return L[(L.t_ms >= a_of) & (L.t_ms < e_of)].reset_index(drop=True)


def precue_licks(clean: dict, idx: pd.DataFrame, frame: SF.SpoutFrame | None, df: pd.DataFrame,
                 lick_s: np.ndarray | None = None, cue_s: dict | None = None) -> pd.DataFrame:
    """SPONTANEOUS licks with the spout in place before the cue: position strobe (the spout HAS arrived -- Priya,
    2026-10-16) .. cue + 70 ms, where the post-cue pass starts (Priya: "detect spontaneous licks, at the very least
    after spout arrival but before cue"). Strobe -> cue is median ~7 s, up to ~80 s (a lick resets the no-lick timer).
    ``t_from_strobe_ms`` = time from spout arrival."""
    g = idx.groupby("trial_k", sort=True)
    L = epoch_licks(clean, idx, frame, df, g.strobe_ms.first().to_numpy(float),
                    np.full(g.ngroups, PRECUE_END_AFTER_CUE_MS), lick_s, cue_s)
    return L.rename(columns={"t_from_anchor_ms": "t_from_strobe_ms"})


def iti_licks(clean: dict, idx: pd.DataFrame, frame: SF.SpoutFrame | None, df: pd.DataFrame,
              lick_s: np.ndarray | None = None, cue_s: dict | None = None) -> pd.DataFrame:
    """Licks in the INTER-TRIAL interval, the parts of each trial window outside strobe .. stop (Priya, 2026-10-16:
    "detect licks in the ITI"): ``segment`` 'tail' = this trial's DAQ stop .. its window end (where the next trial's
    window begins, `o2_inference.trial_frame_spans`), 'lead' = window start .. this trial's strobe (the ~1 s before the
    spout is in place). With the post-cue and pre-cue tables this covers every frame of every window; the reach fields
    (`_REACH_COLS`) are blanked, the spout being in motion or retracted. A lick straddling two windows can be missed
    (the windows are cleaned separately)."""
    g = idx.groupby("trial_k", sort=True)
    t0, t1 = g.t_ms.min().to_numpy(float), g.t_ms.max().to_numpy(float) + 4.0
    strobe, stop = g.strobe_ms.first().to_numpy(float), g.stop_ms.first().to_numpy(float)
    parts = []
    for name, a, e in (("lead", t0, strobe), ("tail", np.where(np.isfinite(stop), stop + 1e-6, np.nan), t1)):
        L = epoch_licks(clean, idx, frame, df, a, e, lick_s, cue_s)
        if len(L):
            L.insert(0, "segment", name)
            parts.append(L)
    if not parts:
        return pd.DataFrame()
    L = pd.concat(parts, ignore_index=True).sort_values(["trial_id", "t_ms"], kind="stable").reset_index(drop=True)
    for c in _REACH_COLS:
        if c in L.columns:
            L[c] = np.nan
    return L


def _spout_arrays(clean: dict, df: pd.DataFrame) -> tuple:
    """The model's spout (x, y, likelihood) laid out like the cleaned tongue arrays (padded clip index)."""
    pos_map = clean["tongue"][1]
    n = len(clean["tongue"][0].y_final)
    out = []
    for c_ in ("spout_x", "spout_y", "spout_likelihood"):
        a_ = np.full(n, np.nan)
        a_[pos_map] = df[c_].to_numpy(float)
        out.append(a_)
    return tuple(out)


def daq_lick_onsets_s(animal: str, date: str, rv=None) -> np.ndarray:
    """Every DAQ lick-contact onset of the session (DAQ s), for the out-of-response passes' ``lick_s``."""
    from wfield_local import dlc_frames
    from wfield_local.paths import PathResolver
    rv = rv or PathResolver()
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
    return np.asarray(dlc_frames.lick_onsets(animal, date, sid, rv), float)


def main(argv=None) -> int:
    from wfield_local import dlc_project
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lp", type=Path, required=True)
    a = ap.parse_args(argv)
    rv = PathResolver()
    d = dlc_project.project_dir(rv).parent / "lp_vs_dlc_cue_traces" / "PS93_20260908"
    idx = pd.read_csv(d / "cue_windows_index.csv")
    # spout frame from DLC's spout over the clip (one span per window)
    dlc = oc.read_pose(d / "cue_windows_DLC.csv")
    xyp = dlc[["spout_x", "spout_y", "spout_likelihood"]].to_numpy()
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(xyp, spans))
    print(f"spout frame: origin {frame.origin.round(1)}  ap {frame.ap_axis.round(3)}  rms {frame.rms_px:.2f} px  "
          f"mouse-left sign {frame.mouse_left_sign:+d}")
    from wfield_local import trial_windows as TW
    b = TW.trial_bounds("PS93", "20260908", rv)
    stop_after = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    rows = []
    for name, f in (("DLC", d / "cue_windows_DLC.csv"), ("LP", a.lp)):
        T, J, M, c = run(f, idx, frame, stop_after)
        tg = c["tongue"][0]
        conf = tg.fill_method[c["tongue"][1]] == oc.FILL_NONE
        yv = tg.y_final[c["tongue"][1]][conf]
        xv = tg.x_final[c["tongue"][1]][conf]
        print(f"{name} tongue rel. MOUTH (raw confident frames): y p5/50/95 {np.percentile(yv, [5, 50, 95]).round(0)}  "
              f"x p5/50/95 {np.percentile(xv, [5, 50, 95]).round(0)};  kept-lick peak y p5/50/95 "
              f"{np.percentile(T.per_lick.y, [5, 50, 95]).round(0) if len(T.per_lick) else '-'}")
        pt = T.per_trial.merge(J[["trial_id", "jaw_pass_qc", "jaw_peak_selected_deflection"]], on="trial_id")
        mm = M.trials
        pt = pt.merge(mm[["trial_id", "candidate_no_lick_with_jaw_move"]], on="trial_id", how="left")
        pt["model"] = name
        rows.append(pt)
        pl = T.per_lick
        print(f"\n== {name}: {len(pt)} trials, {len(pl)} kept licks")
    out = pd.concat(rows, ignore_index=True)
    order = {p: i for i, p in enumerate(tk.POSITIONS)}
    summ = (out.groupby(["model", "position"])
            .agg(n=("trial_id", "size"), kept_licks=("n_kept_licks", "mean"), lick1_ms=("lick1_t_peak_ms", "median"),
                 lick1_angle=("lick1_angle_at_ypeak", "median"), angle_max_l1=("angle_max_signed_lick1", "median"),
                 vxy_l1=("vxy_peak_lick1", "median"), jaw_moved=("jaw_pass_qc", "mean"),
                 mismatch=("candidate_no_lick_with_jaw_move", "sum"))
            .reset_index().sort_values(["model", "position"], key=lambda c: c.map(order) if c.name == "position" else c))
    print("\nper position (medians; angle + = image-right = mouse LEFT on cam4):")
    print(summ.round(1).to_string(index=False))
    out.to_csv(d / "kinematics_demo_per_trial.csv", index=False)
    print(f"-> {d / 'kinematics_demo_per_trial.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
