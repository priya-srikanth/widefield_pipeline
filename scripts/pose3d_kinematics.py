"""The cam4 cohort's tongue / jaw readouts redone in 3-D (Priya 10-19: cross-check, then direction vs target, reach + speed,
jaw opening), on the SAME lick events as the cam4 cohort.

    python -m scripts.pose3d_kinematics [ANIMAL:YYYYMMDD ...]      # default: every session in session_poses/pose3d/

Licks = the cam4 production licks (LP_analysis v36 ``licks.csv``: on_ms .. off_ms windows), so every 3-D number belongs to a
lick the cam4 tables already count. Within each window, on the 3-D tongue (both cameras + single-camera fill):
  * PEAK IN 3-D (Priya 10-19: "peak time may need to be revisited based on 3-D position"): the frame where the tip is
    FARTHEST FROM THE MOUTH (E, below). The cam4 peak is the max of cam4's 'out' axis, which is 99 % the DOWN axis
    (`scripts.pose3d_crosscheck`: r 0.99 with -dv, ~0 with ap). ``dt_peak_ms`` = 3-D peak - cam4 peak.
  * MOUTH E = the spout "mouth zero": the stage's mouth SET POINT (Priya 10-19), i.e. the origin of the 3-D frame. It is
    set at the rig by bringing the spout as close as possible to the front of the mouth, so the targets sit 2 / 4 mm from
    it (`geometry` dist_close / dist_far). NOT a body landmark: the resting JAW label (the chin, ~1.4 mm in front of and
    1.3 mm below it) made the close spouts look like they sat on the mouth, and the jaw and the resting NOSE can both move
    after the stroke (Priya 10-19: "don't use resting nose"). It is re-entered daily, so it carries that day's placement
    (August pre: within 0.12-0.18 mm of a nose-anchored version; both acute days 0.66 mm, mostly ~0.5 mm further back --
    question to Priya 10-19). ``nose_*`` / ``jaw_*`` = the session's resting nose / jaw relative to it (diagnostics only).
  * 1. DIRECTION vs TARGET, from E at the 3-D peak. NOT a horizontal azimuth (10-19): with the jaw as E it sat AT the close_center spout
    and the far spouts lay beside / below it, while licks go mostly DOWN (elevation -50 to -75 deg), so atan2(ml, ap) is
    set by tiny ap differences (close_L target came out at -119 deg, 'behind' the mouth). Instead: ``ang3d_err`` = the
    3-D angle between E->tip and E->target; ``frontal`` = atan2(ml, down), the side tilt of a downward lick (what cam4's
    angle approximates, now in mm; + = mouse RIGHT), ``frontal_err_ipsi`` = -(frontal - target's) (+ = toward mouse
    LEFT = ipsilesional for all four animals); ``sagittal`` = atan2(out, down), the forward tilt, and its error.
    Origin-free, at the closest approach to the spout: ``miss_mm``, ``lat_err_mm`` (+ = outward; centre + = mouse left),
    ``ap_err_mm``, ``dv_err_mm``.
  * 2. REACH + SPEED: ``reach_mm`` = |tip - E| at the 3-D peak and its components (``out_ap``, ``down_dv``, ``side_ml``);
    ``speed_peak`` = max 3-D tip speed from onset to the 3-D peak (central difference over consecutive 4 ms frames, mm/s).
Jaw per TRIAL (as `wfield_local.orofacial_jaw_lateral`, but cue + 70 ms .. the trial's stop, measured jaw frames): rest = the trial's
pre-cue median; ``open_max_mm`` = largest 3-D distance from rest and its components there; ``lr_wide_mm`` = median sideways
displacement (+ = mouse LEFT = ipsi) over frames opened > ``OPEN_MM`` (1.5 mm = cam4's 50 px at ~34 px / mm). Session
FAR_DIR = (far_L - far_R) / 2 of the per-trial medians (+ = toward the target side), FAR_BIAS = (far_L + far_R) / 2.
Output `session_poses/pose3d/kinematics/`: licks.csv.gz, jaw_trials.csv.gz, sessions.csv (per session x position medians +
jaw FAR_DIR), by_animal.csv (each animal's post sessions minus its AUGUST pre median -- rule 8, printed first).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
OPEN_MM = 1.5
JAW_WIN_MS = (70.0, 3500.0)   # cue + 70 ms .. the TRIAL'S stop (end of its response window, usually 3500 ms; Priya 10-19);
                              # the second value is only the fallback when stop is missing (cam4 jaw_lateral used 3000)
LICK_MEASURES = ["miss_mm", "lat_err_mm", "ap_err_mm", "dv_err_mm", "ang3d_err", "midline", "midline_err_ipsi", "frontal",
                 "frontal_err_ipsi", "sagittal",
                 "sagittal_err", "reach_mm", "out_ap", "down_dv", "side_ml", "speed_peak", "dt_peak_ms", "share_single_view"]


def rest_points(S: pd.DataFrame) -> dict:
    """Session medians over pre-cue frames (cue - 500 .. cue) of the 3-D nose and jaw (mm)."""
    pre = S[(S.t_ms < 0) & (S.t_ms >= -500)]
    return {p: pre[[f"{p}_ml", f"{p}_ap", f"{p}_dv"]].median().to_numpy(float) for p in ("nose", "jaw")}


def session_licks(S: pd.DataFrame, C: pd.DataFrame, geo: dict, E: np.ndarray) -> pd.DataFrame:
    from wfield_local import spout_world as SW
    tgt = {p: SW.commanded_world(geo, p) for p in SW.POSITIONS}
    g = {k: v for k, v in S.groupby("trial_id")}
    rows = []
    for r in C.itertuples():
        f = g.get(r.trial_id)
        if f is None or r.position not in tgt:
            continue
        w = f[(f.t_ms >= r.on_ms - 0.5) & (f.t_ms <= r.off_ms + 0.5)]
        P = w[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
        ok = np.isfinite(P).all(1)
        if ok.sum() < 3:
            continue
        t, P = w.t_ms.to_numpy(float)[ok], P[ok]
        src = w.tongue_src.to_numpy()[ok] if "tongue_src" in w else np.ones(len(P))
        R = P - E
        d = np.linalg.norm(R, axis=1)
        k = int(np.argmax(d))
        T = tgt[r.position]
        dm = np.linalg.norm(P - T, axis=1)
        m = int(np.argmin(dm))
        sp = np.full(len(P), np.nan)                                   # central difference on consecutive frames only
        cons = (np.abs(t[2:] - t[:-2] - 8.0) < 0.5)
        sp[1:-1][cons] = np.linalg.norm(P[2:][cons] - P[:-2][cons], axis=1) / 0.008
        Tt = T - E
        cosang = float(R[k] @ Tt / (np.linalg.norm(R[k]) * np.linalg.norm(Tt)))
        fr, fr_t = np.degrees(np.arctan2(R[k, 0], -R[k, 2])), np.degrees(np.arctan2(Tt[0], -Tt[2]))      # side tilt
        # angle OUT OF THE MIDLINE (sagittal) PLANE through the mouth zero and the close / far centre spouts (Priya 10-19);
        # that plane is ml = 0 here (centre azimuth 0, head roll 0), so the angle is asin(ml / |r|)
        md = float(np.degrees(np.arcsin(np.clip(R[k, 0] / np.linalg.norm(R[k]), -1, 1))))
        md_t = float(np.degrees(np.arcsin(np.clip(Tt[0] / np.linalg.norm(Tt), -1, 1))))
        sg, sg_t = np.degrees(np.arctan2(R[k, 1], -R[k, 2])), np.degrees(np.arctan2(Tt[1], -Tt[2]))      # forward tilt
        out_sign = np.sign(T[0]) if abs(T[0]) > 1e-6 else -1.0
        rows.append({"trial_id": r.trial_id, "lick_idx": r.lick_idx, "position": r.position,
                     "contact": bool(getattr(r, "contact", False)), "cam4_matched": bool(getattr(r, "cam4_matched", True)), "t_peak_cam4": r.t_ms, "t_peak_3d": t[k],
                     "dt_peak_ms": t[k] - r.t_ms, "n_3d": int(ok.sum()), "share_single_view": float(np.mean(src > 1)),
                     "reach_mm": float(d[k]), "side_ml": R[k, 0], "out_ap": R[k, 1], "down_dv": -R[k, 2],
                     "ang3d_err": float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
                     "midline": md, "midline_target": md_t, "midline_err_ipsi": -(md - md_t),
                     "frontal": fr, "frontal_target": fr_t, "frontal_err_ipsi": -(fr - fr_t),
                     "sagittal": sg, "sagittal_target": sg_t, "sagittal_err": sg - sg_t,
                     "miss_mm": float(dm[m]), "lat_err_mm": float(out_sign * (P[m, 0] - T[0])),
                     "ap_err_mm": float(P[m, 1] - T[1]), "dv_err_mm": float(P[m, 2] - T[2]),
                     "speed_peak": float(np.nanmax(sp[:k + 1])) if np.isfinite(sp[:k + 1]).any() else np.nan})
    return pd.DataFrame(rows)


def combined_events(rv, S: pd.DataFrame, C4: pd.DataFrame, animal: str, date: str) -> pd.DataFrame:
    """`pose3d.lick_events` on the session's 3-D track, each flagged ``cam4_matched`` (overlaps a cam4 lick window of the
    same trial) and ``contact`` (a DAQ spout-contact onset within the event +- 20 ms)."""
    import scripts.pose_kinematics_demo as D
    from wfield_local import pose3d as P3
    from wfield_local import trial_windows as TW
    E = P3.lick_events(S)
    win = {k: g[["on_ms", "off_ms"]].to_numpy(float) for k, g in C4.groupby("trial_id")}
    E["cam4_matched"] = [bool(((w := win.get(r.trial_id, np.empty((0, 2))))[:, 0] <= r.off_ms).any()
                              and ((w[:, 0] <= r.off_ms) & (w[:, 1] >= r.on_ms)).any()) for r in E.itertuples()]
    cont = D.daq_contacts_ms(animal, date, TW.trial_bounds(animal, date, rv), rv)
    E["contact"] = [bool(((c := np.asarray(cont.get(r.trial_id, []), float)) >= r.on_ms - 20).any()
                         and ((c >= r.on_ms - 20) & (c <= r.off_ms + 20)).any()) for r in E.itertuples()]
    return E


def session_jaw(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, f in S.groupby("trial_id"):
        J = f[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
        t = f.t_ms.to_numpy(float)
        stop = float(f.stop_ms.iloc[0]) if np.isfinite(f.stop_ms.iloc[0]) else JAW_WIN_MS[1]
        pre = (t < 0) & (t >= -500) & np.isfinite(J).all(1)
        win = (t >= JAW_WIN_MS[0]) & (t <= stop) & np.isfinite(J).all(1)
        if pre.sum() < 5 or win.sum() < 5:
            continue
        D = J[win] - np.median(J[pre], axis=0)
        o = np.linalg.norm(D, axis=1)
        k = int(np.argmax(o))
        wide = o > OPEN_MM
        rows.append({"trial_id": tid, "position": f.position.iloc[0], "open_max_mm": float(o[k]),
                     "open_max_side_ml": D[k, 0], "open_max_back_ap": -D[k, 1], "open_max_down_dv": -D[k, 2],
                     "lr_wide_mm": float(np.median(-D[wide, 0])) if wide.sum() >= 3 else np.nan, "n_wide": int(wide.sum())})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    from scripts.apparatus_mouth import _rig
    from wfield_local import epochs
    from wfield_local import pose3d as P3
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--licks", choices=["cam4", "combined"], default="cam4",
                    help="cam4 = the cam4 production lick events (default); combined = events from the 3-D track "
                         "(`pose3d.lick_events`), flagged cam4_matched, DAQ contact; output kinematics_combined/")
    ap.add_argument("--all", action="store_true", help="every session with 3-D frames, not just the curated cohort")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    from wfield_local import analysis_kit as ak
    cur = None if a.all else ak.curated_ids()
    sess = a.sessions or sorted(f.name.replace("_frames.csv.gz", "").replace("_", ":", 1) for f in d3.glob("PS*_frames.csv.gz")
                                if cur is None or f.name.replace("_frames.csv.gz", "") in cur)
    out = d3 / ("kinematics" if a.licks == "cam4" else "kinematics_combined")
    out.mkdir(exist_ok=True)
    bad = P3.failed_sessions(d3)                      # spout fit / camera-timing gates (`pose3d.failed_sessions`)
    frames, rest = {}, {}
    for s in sess:                                    # pass 1: frames + resting nose / jaw per session
        animal, date = s.split(":")
        if f"{animal}_{date}" in bad:
            print(f"{s}: {bad[f'{animal}_{date}']} -- skipped")
            continue
        if not (Path(rv.root("orofacial")) / V36 / "sessions" / animal / date / "licks.csv").exists():
            print(f"{s}: no v36 licks -- skipped")
            continue
        frames[s] = pd.read_csv(d3 / f"{animal}_{date}_frames.csv.gz").drop_duplicates(["trial_id", "t_ms"])
        rest[s] = rest_points(frames[s])
    E = np.zeros(3)                                   # the spout mouth zero = the frame's origin (Priya 10-19)
    LL, JJ, SS = [], [], []
    for s, S in frames.items():
        animal, date = s.split(":")
        C = pd.read_csv(Path(rv.root("orofacial")) / V36 / "sessions" / animal / date / "licks.csv")
        if a.licks == "combined":
            C = combined_events(rv, S, C, animal, date)
        geo = _rig(rv, animal, date)["geometry"]
        ep = epochs.epoch_of(f"{animal}_{date[4:]}")
        L, J = session_licks(S, C, geo, E), session_jaw(S)
        for X in (L, J):
            X["animal"], X["date"], X["session"], X["epoch"] = animal, date, f"{animal}_{date}", ep
        LL.append(L)
        JJ.append(J)
        row = {"session": f"{animal}_{date}", "animal": animal, "date": date, "epoch": ep,
               "cam4_licks": len(C), "licks_3d": len(L),
               "not_in_cam4": int((~L.cam4_matched).sum()) if len(L) else 0,
               **{f"{part}_{ax}": v for part in ("nose", "jaw") for ax, v in zip(("ml", "ap", "dv"), rest[s][part])}}
        jm = J.groupby("position").lr_wide_mm.median()
        if {"far_L", "far_R"} <= set(jm.index):
            row["jaw_far_dir_mm"] = (jm["far_L"] - jm["far_R"]) / 2       # + = toward the target side
            row["jaw_far_bias_mm"] = (jm["far_L"] + jm["far_R"]) / 2      # + = ipsi (mouse left)
        # CLOSE side spouts too (10-20): cam4's jaw readout leaves them out (the rod hides part of the chin in cam4),
        # but 3-D sees the jaw there 81-98 % of frames at ~1 px (both cameras)
        if {"close_L", "close_R"} <= set(jm.index):
            row["jaw_close_dir_mm"] = (jm["close_L"] - jm["close_R"]) / 2
            row["jaw_close_bias_mm"] = (jm["close_L"] + jm["close_R"]) / 2
        for p, g in L.groupby("position"):
            for m in LICK_MEASURES:
                row[f"{m}|{p}"] = float(g[m].median())
        for p, g in J.groupby("position"):
            row[f"open_max_mm|{p}"] = float(g.open_max_mm.median())
        SS.append(row)
        print(f"{s} ({ep}): {len(L)} / {len(C)} cam4 licks with >= 3 3-D frames; {len(J)} jaw trials")
    L, J, S = pd.concat(LL, ignore_index=True), pd.concat(JJ, ignore_index=True), pd.DataFrame(SS)
    L.to_csv(out / "licks.csv.gz", index=False, float_format="%.4g")
    J.to_csv(out / "jaw_trials.csv.gz", index=False, float_format="%.4g")
    S.to_csv(out / "sessions.csv", index=False)
    # rule 8: each animal's post sessions minus its AUGUST pre median, per position, before any pooling
    num = [c for c in S.columns if "|" in c or c.startswith(("jaw_far", "nose_", "jaw_ml", "jaw_ap", "jaw_dv"))]
    base = S[(S.epoch == "pre") & (S.date >= "20260801")].groupby("animal")[num].median()
    rows = []
    for r in S[S.epoch.isin(["acute", "subacute", "chronic"])].itertuples(index=False):
        if r.animal not in base.index:
            continue
        b = base.loc[r.animal]
        rows.append({"animal": r.animal, "session": r.session, "epoch": r.epoch,
                     **{c: S.loc[S.session == r.session, c].iloc[0] - b[c] for c in num}})
    B = pd.DataFrame(rows)
    B.to_csv(out / "by_animal.csv", index=False)
    if len(B):
        show = ["jaw_far_dir_mm", "nose_ap", "nose_dv"] + [f"{m}|{p}" for m in ("miss_mm", "lat_err_mm", "ang3d_err", "frontal_err_ipsi", "sagittal_err", "reach_mm", "out_ap",
                                                          "down_dv", "speed_peak", "open_max_mm")
                                     for p in ("far_L", "far_R", "far_center")]
        show = [c for c in show if c in B]
        pd.set_option("display.width", 250)
        print("\nchange vs the animal's August pre median (mm, deg, mm/s):")
        print(B.set_index(["animal", "session", "epoch"])[show].T.round(2).to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
