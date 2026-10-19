"""Do the cam4 (2-D) cohort lick measures agree with the 3-D tongue (`wfield_local.pose3d`)? Lick by lick, then pre -> post.

    python -m scripts.pose3d_crosscheck [ANIMAL:YYYYMMDD ...]      # default: every session with session_poses/pose3d/<s>_frames.csv.gz

Each cam4 production lick (LP_analysis v36 ``sessions/<a>/<d>/licks.csv``: peak ``t_ms``, ``lr_px`` lateral in the cam4 mouth
frame, + = image right = mouse LEFT; ``ap_px`` the cam4 'out' axis, which on a frontal camera is mostly DOWN; ``protrusion_px``;
``peak_angle_deg``; ``tip_to_spout_px``) is matched to the 3-D frame of the same trial and time (same 4 ms grid). Reports, per
session: how many cam4 licks have a 3-D point at their peak; Pearson r and slope of cam4 lateral vs -ml (3-D, mm; sign
flipped so both are + = mouse left), cam4 'out' vs -dv and vs ap, protrusion vs 3-D reach (distance from the mouth set point),
the cam4 peak angle vs -ml, and the cam4 tip-to-spout px vs the 3-D tip-to-spout distance at the same frame. The CHANGE check
(per position, post session minus the animal's August pre median): does cam4 move the same way as 3-D? Output `session_poses/pose3d/crosscheck/{per_lick.csv.gz, agreement.csv, changes.csv}`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
PAIRS = (("lr_px", "neg_ml", "lateral: cam4 lr vs -ml"),
         ("ap_px", "neg_dv", "cam4 'out' vs -dv"),
         ("ap_px", "ap", "cam4 'out' vs ap"),
         ("protrusion_px", "reach3d", "protrusion vs 3-D reach"),
         ("peak_angle_deg", "neg_ml", "peak angle vs -ml"),
         ("tip_to_spout_px", "tip_spout3d", "tip-to-spout px vs 3-D"))


def session_licks(rv, animal: str, date: str) -> pd.DataFrame | None:
    from scripts.apparatus_mouth import _rig
    from wfield_local import spout_world as SW
    f3 = Path(rv.root("dlc")) / "session_poses" / "pose3d" / f"{animal}_{date}_frames.csv.gz"
    f4 = Path(rv.root("orofacial")) / V36 / "sessions" / animal / date / "licks.csv"
    if not (f3.exists() and f4.exists()):
        return None
    S = pd.read_csv(f3, usecols=["trial_id", "t_ms", "position", "tongue_ml", "tongue_ap", "tongue_dv", "tongue_src"])
    S = S.drop_duplicates(["trial_id", "t_ms"])
    C = pd.read_csv(f4)
    M = C.merge(S, on=["trial_id", "t_ms"], how="left", suffixes=("", "_3d"))
    geo = _rig(rv, animal, date)["geometry"]
    tgt = {p: SW.commanded_world(geo, p) for p in SW.POSITIONS}
    P = M[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    T = np.array([tgt.get(p, [np.nan] * 3) for p in M.position])
    M["neg_ml"], M["neg_dv"], M["ap"] = -P[:, 0], -P[:, 2], P[:, 1]
    M["reach3d"] = np.linalg.norm(P, axis=1)
    M["tip_spout3d"] = np.linalg.norm(P - T, axis=1)
    M["session"], M["animal"], M["date"] = f"{animal}_{date}", animal, date
    return M


def agreement(M: pd.DataFrame) -> dict:
    row = {"n_cam4_licks": len(M), "with_3d": float(np.isfinite(M.neg_ml).mean())}
    for a, b, name in PAIRS:
        ok = np.isfinite(M[a]) & np.isfinite(M[b])
        if ok.sum() > 20:
            x, y = M.loc[ok, b].to_numpy(float), M.loc[ok, a].to_numpy(float)
            row[f"r | {name}"] = round(float(np.corrcoef(x, y)[0, 1]), 3)
            row[f"slope | {name}"] = round(float(np.polyfit(x, y, 1)[0]), 2)
    return row


def main(argv=None) -> int:
    from wfield_local import epochs
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", metavar="ANIMAL:YYYYMMDD")
    a = ap.parse_args(argv)
    rv = PathResolver()
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    sess = a.sessions or sorted(f.name.replace("_frames.csv.gz", "").replace("_", ":", 1) for f in d3.glob("PS*_frames.csv.gz"))
    out = d3 / "crosscheck"
    out.mkdir(exist_ok=True)
    licks, rows = [], []
    for s in sess:
        animal, date = s.split(":")
        M = session_licks(rv, animal, date)
        if M is None:
            print(f"{s}: no 3-D frames or no v36 licks -- skipped")
            continue
        try:
            ep = epochs.epoch_of(f"{animal}_{date[4:]}")
        except Exception:                                                     # noqa: BLE001  label formats differ
            ep = None
        M["epoch"] = ep
        licks.append(M)
        rows.append({"session": f"{animal}_{date}", "epoch": ep, **agreement(M)})
    L = pd.concat(licks, ignore_index=True)
    L.to_csv(out / "per_lick.csv.gz", index=False, float_format="%.4g")
    A = pd.DataFrame(rows)
    A.to_csv(out / "agreement.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(A.set_index("session").T.to_string())
    # changes: per position, each session minus the animal's AUGUST pre median (the primary baseline, 10-18)
    meas = ["lr_px", "neg_ml", "ap_px", "neg_dv", "ap", "protrusion_px", "reach3d", "peak_angle_deg", "tip_to_spout_px",
            "tip_spout3d"]
    G = L.groupby(["animal", "session", "date", "epoch", "position"], dropna=False)[meas].median().reset_index()
    base = G[(G.epoch == "pre") & (G.date >= "20260801")].groupby(["animal", "position"])[meas].median()
    ch = []
    for r in G.itertuples():
        if r.epoch in (None, "pre") or (r.animal, r.position) not in base.index:
            continue
        b = base.loc[(r.animal, r.position)]
        ch.append({"session": r.session, "epoch": r.epoch, "position": r.position,
                   **{f"d_{m}": round(float(getattr(r, m) - b[m]), 3) for m in meas}})
    Ch = pd.DataFrame(ch)
    Ch.to_csv(out / "changes.csv", index=False)
    if len(Ch):
        print("\nchange vs the animal's August pre median (cam4 px / deg; 3-D mm):")
        print(Ch.to_string(index=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
