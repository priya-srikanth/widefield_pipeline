"""Does 3-D add information over cam4 2-D? Two tests on the SAME licks (Priya 10-19: "run the 2d vs 3d on what we have").

    python -m scripts.pose3d_vs_cam4 [--animals PS92]      # after scripts.pose3d_kinematics (cam4 lick events)

Licks = cam4 production lick events with a 3-D point at their peak (`kinematics/licks.csv.gz` joined to the v36
`licks.csv`). Feature sets, each at the lick's own peak:
  cam4 2-D : lateral ``lr_px``, vertical ('out') ``ap_px`` (+ ``protrusion_px``, ``peak_angle_deg``)
  3-D      : ``side_ml``, ``out_ap``, ``down_dv`` at the 3-D peak (mm, mouth-zero frame)
1. DECODING the target spout (6 classes, chance 1/6) from each lick's endpoint: multinomial logistic regression on
   standardised features, leave-one-SESSION-out over the animal's AUGUST pre sessions; balanced accuracy. Also per pair of
   neighbouring positions (close vs far on each side, the depth contrast cam4 cannot see).
2. EFFECT SIZE of each post session vs pre: (post median - mean of the August pre session medians) / SD of those
   medians, per position, for matched measures: lateral (cam4 lr_px vs 3-D side), protrusion vs 3-D reach, cam4 peak angle
   vs 3-D midline angle, plus 3-D only (out, down, miss). Larger |z| = the measure separates the epoch from day-to-day
   variation more. Output `session_poses/pose3d/vs_cam4/{decoding.csv, effect_sizes.csv}`, printed per animal first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
FEATS = {"cam4 2-D": ["lr_px", "ap_px"], "cam4 2-D + angle": ["lr_px", "ap_px", "peak_angle_deg"],
         "3-D": ["side_ml", "out_ap", "down_dv"]}
PAIRS = {"L close vs far": ("close_L", "far_L"), "centre close vs far": ("close_center", "far_center"),
         "R close vs far": ("close_R", "far_R")}
MATCHED = (("lateral", "lr_px", "side_ml"), ("protrusion / reach", "protrusion_px", "reach_mm"),
           ("angle", "peak_angle_deg", "midline"), ("3-D only: out", None, "out_ap"),
           ("3-D only: down", None, "down_dv"), ("3-D only: miss", None, "miss_mm"))


def load(rv, animals) -> pd.DataFrame:
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    K = pd.read_csv(d3 / "kinematics" / "licks.csv.gz", dtype={"date": str})
    if animals:
        K = K[K.animal.isin(animals)]
    out = []
    for s, g in K.groupby("session"):
        a, d = s.split("_")
        C = pd.read_csv(Path(rv.root("orofacial")) / V36 / "sessions" / a / d / "licks.csv",
                        usecols=["trial_id", "lick_idx", "lr_px", "ap_px", "protrusion_px", "peak_angle_deg"])
        out.append(g.merge(C, on=["trial_id", "lick_idx"], how="inner"))
    return pd.concat(out, ignore_index=True)


def decode(L: pd.DataFrame, feats: list[str], classes) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    D = L[L.position.isin(classes)].dropna(subset=feats)
    yt, yp = [], []
    for s in D.session.unique():
        tr, te = D[D.session != s], D[D.session == s]
        if tr.position.nunique() < len(classes) or not len(te):
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[feats], tr.position)
        yt += list(te.position)
        yp += list(m.predict(te[feats]))
    return float(balanced_accuracy_score(yt, yp)) if yt else np.nan


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--pre-from", default="20260801")
    a = ap.parse_args(argv)
    rv = PathResolver()
    L = load(rv, a.animals)
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d" / "vs_cam4"
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    dec, eff = [], []
    for animal, A in L.groupby("animal"):
        pre = A[(A.epoch == "pre") & (A.date >= a.pre_from)]
        if pre.session.nunique() < 3:
            print(f"{animal}: < 3 August pre sessions -- decoding skipped")
        else:
            for name, f in FEATS.items():
                row = {"animal": animal, "features": name, "n_sessions": pre.session.nunique(),
                       "6 positions": decode(pre, f, POS)}
                for pn, pr in PAIRS.items():
                    row[pn] = decode(pre, f, pr)
                dec.append(row)
        med = A.groupby(["session", "epoch", "date", "position"])[
            ["lr_px", "side_ml", "protrusion_px", "reach_mm", "peak_angle_deg", "midline", "out_ap", "down_dv", "miss_mm"]
        ].median().reset_index()
        base = med[(med.epoch == "pre") & (med.date >= a.pre_from)]
        mu, sd = base.groupby("position").mean(numeric_only=True), base.groupby("position").std(numeric_only=True)
        for r in med[med.epoch != "pre"].itertuples(index=False):
            for name, c2, c3 in MATCHED:
                row = {"animal": animal, "session": r.session, "epoch": r.epoch, "position": r.position, "measure": name}
                for lab, c in (("z_cam4", c2), ("z_3d", c3)):
                    row[lab] = (getattr(r, c) - mu.loc[r.position, c]) / sd.loc[r.position, c] if c else np.nan
                eff.append(row)
    Dc, E = pd.DataFrame(dec), pd.DataFrame(eff)
    Dc.to_csv(out / "decoding.csv", index=False)
    E.to_csv(out / "effect_sizes.csv", index=False)
    print("1. decoding the target spout from the lick endpoint (balanced accuracy, leave-one-session-out, August pre; "
          "chance 6 positions 0.17, pairs 0.50):")
    print(Dc.round(3).to_string(index=False))
    if len(E):
        print("\n2. effect size vs pre day-to-day spread (|z|; post session minus the August pre mean / SD of pre "
              "session medians):")
        P = E.assign(abs_cam4=E.z_cam4.abs(), abs_3d=E.z_3d.abs()).pivot_table(
            index=["animal", "measure"], columns="position", values=["abs_cam4", "abs_3d"], aggfunc="median")
        print(P.round(1).to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
