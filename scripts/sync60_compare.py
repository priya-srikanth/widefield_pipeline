"""Whole-session (first..last DAQ sync pulse) re-run at the 60 ms video-lick floor (11-22) vs the 11-21 run.

Per session, old = `_3net_20261121` (trial windows, 40 ms floor; fusion `..._3net_20261121_gapfix`, cam4 v37,
cam1_licks_v37) and new = `_3net_20261121_sync60` (sync-span windows, 60 ms floor; fusion
`..._3net_20261121_sync60_gapfix`, cam4 `__clean_v37_sync60`, cam1_licks_v37_sync60):

* kept fused licks (`camera_licks.lick_train`) by camera combination (`table_gaps_check.composition`) + side-only reasons
* 3-D licks (`<s>_licks.csv` rows), cam4 `licks_all.csv` rows, cam1 `events_all.csv` rows
* frames in the 3-D table

READ ONLY on both runs; writes ``<pose3d_4cam_3net_20261121_sync60>/qc_sync60/compare_by_session.csv``.

    python -m scripts.sync60_compare [--jobs 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DLC = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses")
OROF = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis")
SIDE = "lp_side_r2b_prod_20261118"
RUNS = {"old": {"tag": "_3net_20261121", "cam4": "lp_r4_heatmap_20261013__clean_v37", "cam1": "cam1_licks_v37"},
        "new": {"tag": "_3net_20261121_sync60", "cam4": "lp_r4_heatmap_20261013__clean_v37_sync60",
                "cam1": "cam1_licks_v37_sync60"}}


def _rows(path: Path, **kw) -> float:
    return float(len(pd.read_csv(path, **kw))) if path.exists() else float("nan")


def one(sid: str) -> dict:
    from scripts.table_gaps_check import composition
    from wfield_local import camera_licks as CL
    a, d = sid.split("_")
    row = {"session": sid}
    for k, r in RUNS.items():
        f = DLC / "camera_licks" / f"{SIDE}{r['tag']}_gapfix" / f"{sid}_fused.csv"
        if f.exists():
            U = pd.read_csv(f)
            row.update({f"{k}_{c}": v for c, v in composition(CL.lick_train(U)).items()})
            so = CL.side_only_mask(U) & ~U.guarded.astype(bool).to_numpy()
            for rs in ("two_side_views", "cut_fragment", "one_view_tracked", "unconfirmed_one_side_view"):
                row[f"{k}_side_only_{rs}"] = int((so & (U.reason == rs).to_numpy()).sum())
        row[f"{k}_licks3d"] = _rows(DLC / f"pose3d_4cam{r['tag']}" / f"{sid}_licks.csv", usecols=[0])
        row[f"{k}_cam4_licks_all"] = _rows(OROF / r["cam4"] / "sessions" / a / d / "licks_all.csv", usecols=[0])
        row[f"{k}_cam1_events_all"] = _rows(DLC / r["cam1"] / f"{sid}_events_all.csv", usecols=[0])
    return row


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    new3 = DLC / f"pose3d_4cam{RUNS['new']['tag']}"
    sids = sorted(p.name[:13] for p in new3.glob("PS*_licks.csv"))
    res, fails = ak.fan_sessions(sids, one, jobs=a.jobs)
    for it, e in fails:
        print("FAILED", it, e)
    C = pd.DataFrame([v for _, v in res]).sort_values("session")
    out = new3 / "qc_sync60"
    assert "MICROSCOPE" in str(out) and "Priya" in str(out)
    out.mkdir(parents=True, exist_ok=True)
    C.to_csv(out / "compare_by_session.csv", index=False)
    num = C.drop(columns="session")
    print(num.sum().to_string())
    print(f"-> {out / 'compare_by_session.csv'} ({len(C)} sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
