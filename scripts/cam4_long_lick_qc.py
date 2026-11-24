"""How often does ONE cam4 lick window hold TWO licks? (11-22, Priya: measure before changing the split rule.)

Trigger: the clip agent's PS95 0821 far_L t259 note ("cam4 calls one long lick where cam1, the sides, the 3-D track and
the DAQ see two; the fused train follows cam4"). Read-only: cached tables of the adopted sync60 run, no pose recompute.

Per cam4 post-cue lick of the adopted run (``sessions/<an>/<date>/licks.csv``: the cue .. stop lick set whose
``on_ms .. off_ms`` window is what `scripts/pose3d_kinematics.py` measures for Fig 4B / 4E):
  n3d      3-D lick events (``pose3d_dir()/<an>_<date>_licks.csv``, `pose3d.lick_events`, cue .. stop, 60 ms floor)
           whose PEAK lies inside the window
  ndaq     DAQ spout-contact onsets inside the window +- 20 ms (`kinematics_chain.daq_contacts_ms`), counted as distinct
           licks only when >= 60 ms apart (the video-lick floor; a bounce inside one lick is not a second lick)
  nfused   fused four-camera licks (`session_licks`, the adopted gapfix60 train) whose peak lies inside the window
  overlap  the NEXT cam4 lick of the same trial starts before this window ends (windows overlap)
  win_ms   off_ms - on_ms
Flags: ``two_3d`` = n3d >= 2; ``two_daq`` = ndaq >= 2; ``both`` = two_3d AND two_daq (the strict "two licks" call);
``fused_split`` = nfused >= 2 (the fused train already has two licks there).

Reach sensitivity for the first lick (Fig 4B): for an affected first lick, ``reach_first3d`` = reach_mm of the first 3-D
event in the window (the reach the split lick would report), next to the cam4-window ``reach_mm`` of
``kinematics/licks.csv.gz`` (the max over the whole window).

    python scripts/cam4_long_lick_qc.py [--jobs 8] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60/"
           "qc_cam4_long_lick_20261122")
TOL_MS = 20.0
FLOOR_MS = 60.0


def _distinct(t: np.ndarray, floor: float = FLOOR_MS) -> int:
    t = np.sort(t)
    if not len(t):
        return 0
    n, last = 1, t[0]
    for x in t[1:]:
        if x - last >= floor:
            n, last = n + 1, x
    return n


def session_rows(session: str) -> pd.DataFrame:
    from wfield_local import kinematics_chain as KC
    from wfield_local import lp_licks
    from wfield_local import orofacial_outputs as oo
    from wfield_local import session_licks as SL
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    from wfield_local.pose3d import pose3d_dir
    rv = PathResolver()
    an, date = session.split("_")
    C = pd.read_csv(oo.session_dir(rv, lp_licks.input_run(), an, date) / "licks.csv",
                    usecols=["trial_id", "position", "lick_idx", "t_ms", "on_ms", "off_ms"])
    E = pd.read_csv(pose3d_dir() / f"{an}_{date}_licks.csv", usecols=["trial_id", "t_onset_ms", "t_peak_ms", "reach_mm"])
    F = SL.session_licks(session, rv)
    F = F[F.context == "post_cue"]
    cont = KC.daq_contacts_ms(an, date, TW.trial_bounds(an, date, rv), rv)
    eg = {k: g.sort_values("t_peak_ms") for k, g in E.groupby("trial_id")}
    fg = {k: g.t_ms.to_numpy(float) for k, g in F.groupby("trial_id")}
    C = C.sort_values(["trial_id", "on_ms"]).reset_index(drop=True)
    nxt = C.groupby("trial_id").on_ms.shift(-1)
    rows = []
    for r, nx in zip(C.itertuples(), nxt):
        e = eg.get(r.trial_id)
        ein = e[(e.t_peak_ms >= r.on_ms) & (e.t_peak_ms <= r.off_ms)] if e is not None else None
        c = np.asarray(cont.get(r.trial_id, []), float)
        cin = c[(c >= r.on_ms - TOL_MS) & (c <= r.off_ms + TOL_MS)]
        f = fg.get(r.trial_id, np.zeros(0))
        rows.append({"session": session, "animal": an, "date": date, "trial_id": r.trial_id, "position": r.position,
                     "lick_idx": r.lick_idx, "t_ms": r.t_ms, "on_ms": r.on_ms, "off_ms": r.off_ms,
                     "win_ms": r.off_ms - r.on_ms,
                     "n3d": 0 if ein is None else len(ein), "ndaq_raw": len(cin), "ndaq": _distinct(cin),
                     "nfused": int(((f >= r.on_ms) & (f <= r.off_ms)).sum()),
                     "overlap": bool(np.isfinite(nx) and nx < r.off_ms), "next_on_ms": nx,
                     "t3d_first": np.nan if ein is None or not len(ein) else float(ein.t_peak_ms.iloc[0]),
                     "t3d_second": np.nan if ein is None or len(ein) < 2 else float(ein.t_peak_ms.iloc[1]),
                     "reach_first3d": np.nan if ein is None or not len(ein) else float(ein.reach_mm.iloc[0])})
    return pd.DataFrame(rows)


def _worker(session):
    return session_rows(session)


def _rate(d: pd.DataFrame) -> pd.Series:
    return pd.Series({"n_licks": len(d), "two_3d_pct": 100 * d.two_3d.mean(), "two_daq_pct": 100 * d.two_daq.mean(),
                      "both_pct": 100 * d.both.mean(), "either_pct": 100 * (d.two_3d | d.two_daq).mean(),
                      "overlap_pct": 100 * d.overlap.mean(),
                      "peak_in_overlap_pct": 100 * d.peak_in_overlap.mean(),
                      "no3d_peak_in_kin_pct": 100 * d.loc[d.in_kin, "no3d_peak"].mean() if d.in_kin.any() else np.nan,
                      "both_fused_split_pct": 100 * d.loc[d.both, "fused_split"].mean() if d.both.any() else np.nan})


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.pose3d import pose3d_dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    K = pd.read_csv(pose3d_dir() / "kinematics" / "licks.csv.gz",
                    usecols=["session", "epoch", "trial_id", "lick_idx", "t_peak_cam4", "t_peak_3d", "reach_mm",
                             "speed_peak"])
    sessions = sorted(K.session.unique())
    res, fail = ak.fan_sessions(sessions, _worker, jobs=a.jobs)
    if fail:
        print("FAILED:", fail)
    L = pd.concat([v for _, v in res if v is not None and len(v)], ignore_index=True)
    ep = K.drop_duplicates("session").set_index("session").epoch
    L["epoch"] = L.session.map(ep)
    L["two_3d"], L["two_daq"] = L.n3d >= 2, L.ndaq >= 2
    L["both"] = L.two_3d & L.two_daq
    L["fused_split"] = L.nfused >= 2
    k = K.rename(columns={"t_peak_cam4": "t_ms", "reach_mm": "reach_win", "speed_peak": "speed_win"})
    L = L.merge(k[["session", "trial_id", "t_ms", "t_peak_3d", "reach_win", "speed_win"]],
                on=["session", "trial_id", "t_ms"], how="left")
    L["in_kin"] = L.reach_win.notna()
    # the window's reach maximum falls in the part shared with the NEXT cam4 window (overlapping windows)
    L["peak_in_overlap"] = L.overlap & (L.t_peak_3d >= L.next_on_ms)
    # no 3-D event peak inside the window (e.g. a window that starts after the tongue's peak): reach from a fragment
    L["no3d_peak"] = L.n3d == 0
    L.to_csv(a.out / "cam4_windows.csv.gz", index=False)
    order = ["pre", "acute", "subacute", "chronic"]
    T = {"overall": _rate(L).to_frame("all").T,
         "epoch": L.groupby("epoch").apply(_rate).reindex([e for e in order if e in set(L.epoch)]),
         "animal": L.groupby("animal").apply(_rate),
         "animal_epoch": L.groupby(["animal", "epoch"]).apply(_rate),
         "position": L.groupby("position").apply(_rate),
         "first_lick": L[L.lick_idx == 0].groupby("epoch").apply(_rate)}
    for n, t in T.items():
        t.round(3).to_csv(a.out / f"rates_{n}.csv")
        print(f"== {n}\n{t.round(2).to_string()}\n")
    W = L.groupby("both").win_ms.describe(percentiles=[.1, .5, .9])
    W.to_csv(a.out / "window_length_by_flag.csv")
    print("== cam4 window length (ms) by strict flag\n", W.round(1).to_string())
    # Fig 4B sensitivity: session medians of first-lick reach with affected first licks re-measured on the first 3-D event
    F = L[(L.lick_idx == 0) & L.reach_win.notna()].copy()
    F["reach_split"] = np.where(F.both & F.reach_first3d.notna(), F.reach_first3d, F.reach_win)
    S = F[~F.date.str[4:].lt("0806")].groupby(["session", "epoch", "position"])[["reach_win", "reach_split"]].median()
    S["d"] = S.reach_split - S.reach_win
    S.to_csv(a.out / "reach_first_sensitivity.csv")
    print("== first-lick reach, session x position medians: split - window (mm)\n",
          S.d.describe(percentiles=[.5, .9, .99]).round(4).to_string())
    print("by epoch mean |d|:", S.groupby("epoch").d.apply(lambda x: x.abs().mean()).round(4).to_dict())
    print("written", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
