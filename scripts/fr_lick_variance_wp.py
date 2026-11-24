"""Movement page Fig 3.4 on the camera WORKING PERIOD (lane E, 11-24; Priya 11-23 decision 2: trial gate = the camera
working period, `canonical` of the camera-tagged attempts table, never the DAQ-contact engaged gate).

Same quantity as `iti_licks.controls` figure C (`fr_lick_variance`): trial-to-trial variance of the FaceRhythm factors
in the frozen encoder's cue window (0..1.5 s) and first-lick window (-0.1..1.0 s), per session, normalized to the
animal's own pre mean; var_within = after removing each spout's mean time course. What changes is the TRIAL SET: the
session files' first-lick rows (`iti_licks.extract`, first_*; built under the extraction run's engagement gate) are
given their EXACT trial ids (the extraction rule re-applied to the session's behavior table, `first_row_trials`) and
only rows of WORKING-PERIOD trials (`canonical` == True: stopped runs,
early-run, after-end and unknown trials removed) are kept. Rows whose trial cannot be recovered are dropped and counted (audit).
Nothing is refit; FaceRhythm values are the session files' (legacy, `iti_licks.FRVAR_FIG`).

Writes (under iti_licks.OUT_W, i.e. WFIELD_ITI_OUT_W): fr_variance_wp_sessions.csv (both trial sets: `set` = all_rows /
working), fr_variance_wp_tests.csv (change vs own pre, analysis_kit.boot_delta via iti_licks.dots_compute, both sets),
fr_variance_wp_audit.csv (per session: rows, matched, kept), figdata/fr_lick_variance_wp.pkl,
figures/fr_lick_variance_wp.png.
Usage: WFIELD_ITI_OUT_W=<new dir> python scripts/fr_lick_variance_wp.py [--attempts CSV] [--jobs 3]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

ATT_ENV = "FRVAR_ATTEMPTS_CSV"
STEM = "fr_lick_variance_wp"


def default_attempts() -> Path:
    from wfield_local import run_mode
    q = run_mode.encoding_out_root() / "qc" / "jaw3dmouth_ccB1123"
    return q / "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk" / "trial_attempts.csv"


def first_row_trials(label: str) -> dict:
    """Trial ids of one session file's first-lick rows, EXACT: the extraction rule of `iti_licks._first_lick_times`
    (first behavior lick onset in cue .. stop, cue + 3.5 s without a stop) applied to EVERY trial of the session's
    behavior table, then the stored ``first_rt`` matched as an exact subsequence (`iti_licks.subseq_match`); -1 =
    unmatched. Module-level for spawn."""
    d = il._load(label)
    B = il._behaviour(label)
    T = B.trials
    bo = np.sort(np.asarray(B.lick_on_s, float)[np.isfinite(B.lick_on_s)])
    tid, rt = [], []
    for t, c, s in zip(T.trial_id.to_numpy(int), T.cue_s.to_numpy(float), T.stop_s.to_numpy(float)):
        if not np.isfinite(c):
            continue
        s = s if np.isfinite(s) else c + 3.5
        j = np.searchsorted(bo, c)
        if j < len(bo) and bo[j] <= s:
            tid.append(t); rt.append(bo[j] - c)
    m = il.subseq_match(np.asarray(rt, float), np.asarray(d["first_rt"], float)) if len(d["first_rt"]) else np.zeros(0, int)
    tid = np.asarray(tid, int)
    return {"label": label, "trial_id": np.where(m >= 0, tid[np.maximum(m, 0)] if len(tid) else -1, -1)}


def trial_ids(labels: list[str], jobs: int) -> dict:
    """{label: trial id per first-lick row}, cached in OUT_W/fr_variance_wp_trialids.csv (one behavior load per session,
    ~2 min cold: the DAQ trial bounds)."""
    from wfield_local import analysis_kit as ak
    f = il.OUT_W / "fr_variance_wp_trialids.csv"
    have = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["label", "row", "trial_id"])
    todo = [x for x in labels if x not in set(have.label)]
    got, fail = ak.fan_sessions(todo, first_row_trials, jobs=jobs, label="frvar_wp") if todo else ([], [])
    for it, err in fail:
        print(f"  !! {it}: {err}", flush=True)
    new = [pd.DataFrame({"label": r["label"], "row": np.arange(len(r["trial_id"])), "trial_id": r["trial_id"]})
           for _it, r in got]
    have = pd.concat([have, *new], ignore_index=True) if new else have
    have.to_csv(f, index=False)
    return {x: g.sort_values("row").trial_id.to_numpy(int) for x, g in have.groupby("label")}


def rows_of(x: str, d: dict, keep: np.ndarray, trial_set: str) -> list[dict]:
    """`iti_licks.controls` figure C rows of one session (legacy FR source), restricted to ``keep``."""
    pos = np.asarray(d["first_pos"])[keep]
    out = []
    for wname, key in (("cue", "first_fr_cue"), ("lick", "first_fr_lick")):
        A_ = d[key + il.FRVAR_SRC[il.FRVAR_FIG]].astype(float)[keep]
        for k, v in enumerate(d["fr_names"]):
            Y = A_[:, :, k]
            if not len(Y) or np.isnan(Y).all():
                continue
            R = Y - np.nanmean(Y, 0, keepdims=True)
            W_ = Y.copy()
            for q in np.unique(pos):
                m = pos == q
                W_[m] = Y[m] - np.nanmean(Y[m], 0, keepdims=True)
            out.append({"set": trial_set, "label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]),
                        "window": wname, "factor": str(v), "n_trials": int(keep.sum()),
                        "var_within": float(np.nanvar(W_)), "var_total": float(np.nanvar(R)),
                        "mean": float(np.nanmean(Y))})
    return out


def main(argv=None) -> int:
    from wfield_local import config
    from figdata import save_fd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempts", default=os.environ.get(ATT_ENV) or str(default_attempts()))
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    if il.OUT_W == il.OUT:
        raise SystemExit("set WFIELD_ITI_OUT_W to a NEW folder (this script never writes into the production iti_licks)")
    A = pd.read_csv(a.attempts, usecols=["label", "trial_id", "pos", "first_lick_ms", "cls", "canonical"])
    A["canonical"] = A.canonical.astype(str).str.lower().isin(["true", "1", "1.0"])
    A = A.sort_values(["label", "trial_id"])
    labels = sorted(p.stem for p in (il.XOUT / il.SESS_NAME).glob("*.npz"))
    labels = [x for x in labels if not config.early_pre(x)]
    TID = trial_ids(labels, a.jobs)
    V, audit = [], []
    for x in labels:
        d = il._load(x)
        if "first_fr_cue" not in d or not len(d["first_rt"]):
            continue
        cand = A[A.label == x].reset_index(drop=True)
        t = TID.get(x, np.full(len(d["first_rt"]), -1))
        ok = t >= 0
        ci = cand.set_index("trial_id")
        canon = np.zeros(len(t), bool)
        canon[ok] = ci.canonical.reindex(t[ok]).fillna(False).to_numpy(bool)
        m = t
        cls = ci.cls.reindex(t[ok]).fillna("not_in_table").value_counts().to_dict() if ok.any() else {}
        audit.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "rows": len(m),
                      "matched": int(ok.sum()), "kept_working": int(canon.sum()),
                      "dropped_not_working": int((ok & ~canon).sum()),
                      "working_trials_with_lick_in_table": int((cand.canonical & cand.first_lick_ms.notna()).sum()),
                      **{f"cls_{k}": v for k, v in cls.items()}})
        V += rows_of(x, d, np.ones(len(m), bool), "all_rows")
        V += rows_of(x, d, canon, "working")
        print(f"  {x}: rows {len(m)}, matched {ok.sum()}, working {canon.sum()}", flush=True)
    V = pd.DataFrame(V)
    AU = pd.DataFrame(audit)
    V.to_csv(il.OUT_W / "fr_variance_wp_sessions.csv", index=False)
    AU.to_csv(il.OUT_W / "fr_variance_wp_audit.csv", index=False)
    print("\n== audit per animal x epoch (rows / matched / kept)\n"
          + AU.groupby(["animal", "epoch"])[["rows", "matched", "kept_working", "dropped_not_working"]].sum()
          .to_string())
    tests = []
    for trial_set in ("all_rows", "working"):
        rng = np.random.default_rng(2)             # = iti_licks.controls (jitter and test order unchanged)
        C, offC = [], []
        Vs = V[V.set == trial_set]
        for i, wname in enumerate(("cue", "lick")):
            for j, (fac, val) in enumerate((("fr_lick", "var_within"), ("fr_lick", "var_total"), ("fr_lick", "mean"),
                                            ("fr_right", "var_within"))):
                T = Vs[(Vs.window == wname) & (Vs.factor == fac)].copy()
                if T.empty:
                    offC.append((i, j))
                    continue
                T["norm"] = il._norm_to_pre(T, val)
                S_ = il.dots_compute(T, "norm", rng)
                for r_ in S_["tests"]:
                    tests.append({"set": trial_set, "figure": f"frvar_{wname}_{fac}_{val}", **r_})
                C.append((i, j, f"{wname} window: {fac} {val.replace('var_', 'variance, ')}\n(/ animal's pre mean)", 1,
                          S_))
        if trial_set == "working":
            save_fd(STEM, {"panels": C, "off": offC})
    T = pd.DataFrame(tests)
    T.to_csv(il.OUT_W / "fr_variance_wp_tests.csv", index=False)
    print("\n== change vs own pre (boot_delta, animals -> sessions)\n" + T.round(3).to_string(index=False))
    grid, size, rect, ylabs, _t = il.CTRL_FIGS["fr_lick_variance"]
    il.CTRL_FIGS[STEM] = (grid, size, rect, ylabs,
                          "FACE: trial-to-trial variability of the FaceRhythm lick factor in the frozen encoder's cue "
                          "(0..1.5 s) and first-lick (-0.1..1.0 s) windows, per session, normalized to each animal's pre "
                          "mean; first licks of camera WORKING-PERIOD trials (camera-tagged attempts table, canonical). "
                          "var_within = after removing each spout's mean time course; var_total = all trials. "
                          "* = boot_delta 95% CI of the change from the animal's own pre excludes 0.")
    il.controls_plot(only=(STEM,))
    print("->", il.OUT_W / "figures" / f"{STEM}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
