"""Why is PS94's target-unique R2 ~10x the other animals'? (Priya 10-25.) Two checks, pre-stroke August sessions:

1. BLOCK / STATE ALIGNMENT. Spout positions come in blocks (~6 trials); the drift removal only takes out components
   slower than 10 min, so state that differs between spouts (running, pushing) can be credited to the per-spout TARGET
   terms. Per trial (engaged trials): forward running speed before the cue (-2..0 s), running fraction (> 1.5 mm/s,
   -2..+1.5 s) and push fraction (backward speed > 0.3 mm/s, 0..+1.5 s), from the DAQ treadmill channel. Per animal:
     eta2 within session   share of the trial variance explained by spout within each session (any alignment, incl.
                           block-random state) -- vs a CIRCULAR-SHIFT null (positions shifted by half the session: block
                           structure kept, alignment broken)
     eta2 systematic       share explained by the spout MEAN pooled over sessions after removing session means (what a
                           frozen pooled target kernel could absorb)
2. TRACKING QUALITY. Per animal: share of trials with a 3-D first lick, the 3-D first lick's single-view share (one camera
   only), and cam1 coverage (trial_attempts.has_cam1).

Usage: python scripts/state_spout_check.py
Output: <E>/qc/state_spout_check.csv (+ printed tables)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

KIN = Path("C:/Users/SabatiniLab/wfield_cache/pose3d_inputs/pose3d_kin_licks_snapshot_20261023.csv.gz")
RUN_ON, PUSH_ON = 1.5, 0.3


def trial_state(label: str) -> pd.DataFrame:
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.paths import PathResolver
    an = config.animal_of(label)
    B = ce.load_behaviour(an, "2026" + label.split("_")[-1], PathResolver())
    t, v = (np.asarray(x, float) for x in B.analog["running"])
    T = B.trials
    if "engaged" in T:
        T = T[T.engaged.astype(bool)]
    rows = []
    for _, r in T.iterrows():
        c = r.cue_s
        if not np.isfinite(c):
            continue
        pre = (t >= c - 2.0) & (t < c)
        win = (t >= c - 2.0) & (t < c + 1.5)
        post = (t >= c) & (t < c + 1.5)
        if not pre.any() or not post.any():
            continue
        rows.append({"label": label, "animal": an, "trial_id": r.trial_id, "pos": r.pos,
                     "speed_pre": float(np.nanmean(np.maximum(v[pre], 0))),
                     "run_frac": float(np.nanmean(v[win] > RUN_ON)),
                     "push_frac": float(np.nanmean(-v[post] > PUSH_ON))})
    return pd.DataFrame(rows)


def eta2(df: pd.DataFrame, col: str, pos: str = "pos") -> tuple[float, float]:
    """(within-session eta2 averaged over sessions, systematic eta2 of the session-centred spout means)."""
    w, num, den = [], 0.0, 0.0
    for _, g in df.groupby("label"):
        y = g[col].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 12 or np.nanvar(y) == 0:
            continue
        g = g[ok]
        y = y[ok]
        m = y.mean()
        sst = ((y - m) ** 2).sum()
        ssb = sum(len(h) * (h[col].mean() - m) ** 2 for _, h in g.groupby(pos))
        w.append(ssb / sst if sst > 0 else np.nan)
        num += 0.0
        den += sst
    c = df.dropna(subset=[col]).copy()
    c["yc"] = c[col] - c.groupby("label")[col].transform("mean")
    sst = (c.yc ** 2).sum()
    ssb = sum(len(h) * h.yc.mean() ** 2 for _, h in c.groupby(pos))
    return float(np.nanmean(w)) if w else np.nan, float(ssb / sst) if sst > 0 else np.nan


def main() -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    labels = sorted(p.stem for p in (E / "frozen_encoder_gate-attempted_v7g3_t3d" / "joint" / "sessions").glob("*.json")
                    if p.stem[-4:] >= "0801" and epochs.epoch_of(p.stem) == "pre")
    res, fail = ak.fan_sessions(labels, trial_state, jobs=4, label="state")
    for it, err in fail:
        print("  !!", it, err)
    D = pd.concat([r for _, r in res], ignore_index=True)
    rng = np.random.default_rng(0)
    out = []
    for an, g in D.groupby("animal"):
        for col in ("speed_pre", "run_frac", "push_frac"):
            w, s = eta2(g, col)
            sh = g.copy()
            sh["pos_shift"] = sh.groupby("label").pos.transform(lambda p: np.roll(p.to_numpy(), len(p) // 2))
            ws, ss = eta2(sh, col, "pos_shift")
            out.append({"animal": an, "measure": col, "mean": g[col].mean(), "eta2_within": w, "eta2_within_shift_null": ws,
                        "eta2_systematic": s, "eta2_systematic_shift_null": ss, "n_trials": len(g),
                        "n_sessions": g.label.nunique()})
    _ = rng
    R = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    print("== 1. state x spout (pre-stroke, engaged trials)\n" + R.round(4).to_string(index=False))
    # 2. tracking quality
    K = pd.read_csv(KIN)
    K = K[(K.lick_idx == 0) & (K.epoch == "pre")]
    K["label"] = K.animal + "_" + K.date.astype(str).str[-4:]
    K = K[K.label.str[-4:] >= "0806"]
    A = pd.read_csv(E / "qc" / "trial_attempts.csv")
    A = A[(A.epoch == "pre") & (A.label.str[-4:] >= "0806") & A.attempted.astype(bool)]
    trk = []
    for an in sorted(A.animal.unique()):
        a_ = A[A.animal == an]
        k_ = K[K.animal == an]
        n3 = k_.merge(a_[["label", "trial_id"]], on=["label", "trial_id"]).shape[0]
        trk.append({"animal": an, "attempted_trials": len(a_), "share_with_3d_first_lick": n3 / max(len(a_), 1),
                    "single_view_share_median": float(k_.share_single_view.median()) if len(k_) else np.nan,
                    "cam1_coverage": float(a_.has_cam1.astype(bool).mean()) if "has_cam1" in a_ else np.nan})
    TQ = pd.DataFrame(trk)
    print("\n== 2. tracking quality (pre-stroke, attempted trials)\n" + TQ.round(3).to_string(index=False))
    R.to_csv(E / "qc" / "state_spout_check.csv", index=False)
    TQ.to_csv(E / "qc" / "state_spout_tracking.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
