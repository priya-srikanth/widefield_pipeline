"""Six-position (cohort 2, mice #9-#12) 3-D kinematics BY LICK NUMBER within the trial (CC-B, 2026-11-23;
suffix _bylicknum_ccB1123).

Question (Priya 11-23): is the post-stroke deficit present from the first licks of a trial, with a later deficit that
builds over the bout?

Licks: production cam3-on fused four-camera list (`kinematics_fused_postcue/licks.csv.gz`): lick peak from cue + 70 ms
to the DAQ trial stop, >= 60 ms apart, below-camera-only protrusions not licks, kinematics only from licks seen by >= 2
cameras; August pre only (config.early_pre dropped). Lick number = `lick_idx` + 1 (order among ALL post-cue licks of
the trial). Speed = `speed_peak` (onset -> extension peak, ungated set, as production); reach = `reach_mm` (camera
working-period trial gate, as production all-lick reach; ids from sixpos_first5/working_ids.json).

Per session x position x lick-number bin: the MEDIAN of the per-lick values (cell kept if >= 5 licks). Bins 1, 2, 3-5,
6-10, 11-15, 16+ and individual lick numbers 1-15. Change = session value minus the animal's own pre mean at that
position AND bin. Groups: "pooled" = per session, the mean of the change over the positions present; "far_contra" =
the far contralesional spout alone. Trend contrast per session x position = mean(change 6-10, 11-15) minus mean(change
1, 2, 3-5) (all five bins needed), pooled as above; slope = OLS slope of the change over lick numbers 1-15 (per 10
licks, >= 8 lick numbers).

Stats: orofacial_stats.session_change_tests (deltas_given; session MWU + 4,000-draw paired nested animal -> session
analysis_kit.boot_delta, seed 0). Family = the late-minus-early cells (2 measures x 2 groups x 3 epochs, BH once);
the per-bin and slope cells get their own BH families (descriptive). Star = q < 0.05 AND the CI excludes 0.

    python scripts/sixpos_bylicknum_1123.py --out DIR      (NEW files only; refuses to overwrite)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
LICKS = P3D / "kinematics_fused_postcue/licks.csv.gz"
WORKING_IDS = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ccB_caption_review_2026-11-23"
                   "/sixpos_first5/working_ids.json")
LR_POS = dict(zip(["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"],
                  ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]))
MS = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}
EPOCHS = ("acute", "subacute", "chronic")
BINS = [("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11-15", 11, 15), ("16+", 16, 10**6)]
EARLY, LATE = ["1", "2", "3-5"], ["6-10", "11-15"]
MIN_N = 5
SUFFIX = "_bylicknum_v2_ccB1123"
#: angle = the Fig 4E all-lick 3-D aim angle, atan2(side_ml, hypot(out_ap, down_dv)) in degrees, + = contralesional
#: (aim_tables_from_kinematics + iti_recovery_measure.aim_licks); session MEAN per bin, medians for speed / reach
MEAS = {"speed": "speed_peak", "reach": "reach_mm", "angle": "angle_deg"}
AGG = {"speed": "median", "reach": "median", "angle": "mean"}


def save(df: pd.DataFrame, out: Path, name: str):
    p = out / f"{name}{SUFFIX}.csv"
    if p.exists():
        sys.exit(f"refusing to overwrite {p}")
    df.to_csv(p, index=False, lineterminator="\n")
    print("wrote", p, len(df), flush=True)


def binlab(n: int):
    for lab, lo, hi in BINS:
        if lo <= n <= hi:
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-bout", type=int, default=0,
                    help="sensitivity: keep only trials with >= this many post-cue licks (late bin = 6-10 only)")
    a = ap.parse_args()
    out = a.out
    global SUFFIX, LATE
    if a.min_bout:
        SUFFIX = f"_bout{a.min_bout}" + SUFFIX
        LATE = ["6-10"]
    out.mkdir(parents=True, exist_ok=True)
    from wfield_local import config
    from wfield_local import orofacial_stats as st
    L = pd.read_csv(LICKS, usecols=["trial_id", "lick_idx", "position", "reach_mm", "speed_peak", "side_ml", "out_ap", "down_dv", "animal", "date",
                                    "session", "epoch"], dtype={"date": str})
    L = L[~((L.epoch == "pre") & L.date.map(config.early_pre))].reset_index(drop=True)
    L["pos"] = L.position.map(LR_POS)
    L["angle_deg"] = np.degrees(np.arctan2(L.side_ml, np.hypot(L.out_ap, L.down_dv)))   # + = right = contralesional
    L["n"] = L.lick_idx + 1
    L["bin"] = L.n.map(binlab)
    if a.min_bout:   # same trials supply every lick number 1..min_bout (removes the bout-length selection)
        nb = L.groupby(["session", "trial_id"]).n.transform("max")
        L = L[(nb >= a.min_bout) & (L.n <= a.min_bout)].reset_index(drop=True)
        print(f"min bout {a.min_bout}: {len(L)} licks kept", flush=True)
    ids = json.load(open(WORKING_IDS))
    keep = np.zeros(len(L), bool)
    sv, tid = L.session.astype(str).to_numpy(), L.trial_id.to_numpy(float)
    for s in np.unique(sv):
        m = sv == s
        keep[m] = np.isin(tid[m], np.asarray(ids.get(s, []), float))
    L["working"] = keep
    print(f"licks {len(L)}, working-period {keep.sum()}", flush=True)

    # bout length = post-cue licks per trial (all licks, with or without 3-D support)
    T = L.groupby(["animal", "session", "date", "epoch", "pos", "trial_id"]).n.max().rename("bout").reset_index()
    B = T.groupby(["animal", "session", "date", "epoch", "pos"]).agg(
        bout_median=("bout", "median"), bout_mean=("bout", "mean"), frac_ge6=("bout", lambda x: (x >= 6).mean()),
        frac_ge11=("bout", lambda x: (x >= 11).mean()), n_trials=("bout", "size")).reset_index()
    save(B.assign(animal_ms=B.animal.map(MS)), out, "c2_bout_length_per_session")
    cnt = []
    for meas, col in MEAS.items():
        d = L[L.working] if meas == "reach" else L
        d = d[d[col].notna()]
        cnt.append(d.groupby(["epoch", "bin"]).size().rename("n_licks").reset_index().assign(measure=meas))
        cnt.append(d[d.pos == "far_contra"].groupby(["epoch", "bin"]).size().rename("n_licks").reset_index()
                   .assign(measure=meas + "_far_contra"))
    C = pd.concat(cnt).pivot_table(index=["measure", "epoch"], columns="bin", values="n_licks").reset_index()
    save(C, out, "c2_lick_counts_per_bin")

    # per session x position x bin medians
    g = ["animal", "session", "date", "epoch", "pos"]
    rows = []
    for meas, col in MEAS.items():
        d = (L[L.working] if meas == "reach" else L).dropna(subset=[col])
        s = d.groupby(g + ["bin"])[col].agg([AGG[meas], "size"]).reset_index().assign(kind="bin")
        s = s.rename(columns={AGG[meas]: "value"})
        rows.append(s.assign(measure=meas))
        k = d[d.n <= 15].groupby(g + ["n"])[col].agg([AGG[meas], "size"]).reset_index()
        k = k.rename(columns={AGG[meas]: "value"})
        rows.append(k.assign(bin=k.n.astype(str), kind="licknum", measure=meas).drop(columns="n"))
    S = pd.concat(rows, ignore_index=True).rename(columns={"size": "n_licks"})
    S = S[S.n_licks >= MIN_N].copy()
    base = S[S.epoch == "pre"].groupby(["measure", "kind", "bin", "animal", "pos"]).value.mean().rename("pre_mean")
    S = S.join(base, on=["measure", "kind", "bin", "animal", "pos"]).dropna(subset=["pre_mean"])
    S["chg"] = S.value - S.pre_mean
    save(S.assign(animal_ms=S.animal.map(MS)), out, "c2_session_position_bin_values")

    # per session x position contrast and slope
    gm = ["measure"] + g
    W = S[S.kind == "bin"].pivot_table(index=gm, columns="bin", values="chg").reset_index()
    W["late_minus_early"] = np.where(W[EARLY + LATE].notna().all(1), W[LATE].mean(1) - W[EARLY].mean(1), np.nan)
    W = W.reindex(columns=list(dict.fromkeys(list(W.columns))))
    K = S[S.kind == "licknum"].assign(n=lambda x: x.bin.astype(int))
    sl = K.groupby(gm)[["n", "chg"]].apply(lambda x: np.polyfit(x.n, x.chg, 1)[0] * 10 if len(x) >= 8 else np.nan)
    W = W.merge(sl.rename("slope_per10").reset_index(), on=gm, how="left")
    save(W.assign(animal_ms=W.animal.map(MS)), out, "c2_session_position_contrast")

    # long table: metric = measure|kind|bin, two groups
    lg = pd.concat([S[gm + ["kind", "bin", "chg"]],
                    W.melt(id_vars=gm, value_vars=["late_minus_early", "slope_per10"], var_name="bin",
                           value_name="chg").assign(kind="trend")], ignore_index=True).dropna(subset=["chg"])
    lg["metric"] = lg.measure + "|" + lg.kind + "|" + lg.bin
    pooled = lg.groupby(["metric", "animal", "session", "date", "epoch"]).agg(
        chg=("chg", "mean"), n_pos=("pos", "nunique")).reset_index().assign(pos="pooled")
    fc = lg[lg.pos == "far_contra"][["metric", "animal", "session", "date", "epoch", "pos", "chg"]].assign(n_pos=1)
    G = pd.concat([pooled, fc], ignore_index=True)
    save(G.assign(animal_ms=G.animal.map(MS)), out, "c2_session_group_values")
    P = G.pivot_table(index=["animal", "session", "date", "epoch", "pos"], columns="metric", values="chg").reset_index()
    metrics = sorted(G.metric.unique())
    R = st.session_change_tests(P, metrics, deltas_given=True, n_boot=4000)
    R[["measure", "kind", "bin"]] = R.metric.str.split("|", expand=True)
    R["fam"] = np.where(R.kind == "trend", np.where(R.bin == "late_minus_early", "trend", "slope"), R.kind)
    parts = []
    for _fam, d in R.groupby("fam"):
        d = st.apply_family_fdr(d.copy())
        parts.append(d)
    R = pd.concat(parts, ignore_index=True)
    R["star"] = (R.q_fdr < 0.05) & R.boot_sig
    save(R, out, "c2_cells")
    # per-animal changes (ground rule: per-animal table before the bootstrap)
    pa = []
    for (metric, pos), d in G.groupby(["metric", "pos"]):
        for a, x in d.groupby("animal"):
            pre = x[x.epoch == "pre"].chg.mean()
            for e in EPOCHS:
                if (x.epoch == e).any():
                    pa.append(dict(metric=metric, pos=pos, epoch=e, animal=a, animal_ms=MS[a],
                                   change=x[x.epoch == e].chg.mean() - pre, n_sessions=int((x.epoch == e).sum())))
    save(pd.DataFrame(pa), out, "c2_per_animal")
    t = R[R.fam == "trend"]
    print(t[["measure", "pos", "epoch", "n_sessions", "n_animals", "boot_change", "boot_lo", "boot_hi", "q_fdr",
             "star"]].to_string(), flush=True)
    print(R[R.kind == "bin"].pivot_table(index=["measure", "pos", "epoch"], columns="bin", values="boot_change")
          .round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
