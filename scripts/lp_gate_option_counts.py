"""What WIDEFIELD_GATE_SOURCE=lp changes, per session and per animal (2026-11-01; `wfield_local/gate_source.py`).

NO NEW EXTRACTION: reads `scripts/gate_lp_check.py`'s per-session cache (WFIELD_INPUT_CACHE/gate_lp_check/<label>.pkl,
curated PS92-95, August-only pre, epochs per `epochs.epoch_of`), whose LP rule is the option's rule
(tests/test_lp_gate_option.py checks the gate and the quit against it on three sessions, exactly).

Per session, on the full behavioural trial list (DAQ cues with a resolved position):
  img gate (engagement_gate, responded within decode.max_rt_s = 3.5 s), three response rules:
    DAQ      DAQ contact                           -- the DAQ arm, and the gate_lp_check baseline
    LPonly   LP attempt (LP-covered trials only)   -- what the LP_licks arm's imaging gate is TODAY (its engaged arm is
                                                      the LP-attempted arm; uncovered trials are not in its features)
    DAQorLP  DAQ contact OR LP attempt             -- the option
  n_flip_daq   trials whose engaged status differs DAQ vs DAQorLP (all trials; the rule only adds responses, so every
               flip is not-engaged -> engaged)
  n_flip_lp    the same, LPonly vs DAQorLP, on LP-covered trials (what switching the LP arm actually moves)
  S28 quit (2.0 s scoring, `quit_point.session_quit`): DAQ vs DAQorLP -- shift (min), quit lost.

    python scripts/lp_gate_option_counts.py        -> per-session / per-animal tables printed, CSV written to
                                                      <E>/gate_lp_check/lp_gate_option_counts.csv
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("WIDEFIELD_PRE_FROM", "0801")          # AUGUST-ONLY pre (before config import)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_lp_check import ANIMALS, CACHE, CANON, EPS, OUT, labels  # noqa: E402

RAW = {v: k for k, v in CANON.items()}


def one(d):
    from wfield_local.precue_engagement_states import engagement_gate
    T = d["trials"]
    r = {"label": d["label"], "animal": d["animal"], "epoch": d["epoch"], "lp": bool(d["lp"]), "n": len(T),
         "n_ne_daq": int(T.ne_img.sum())}
    qd, ql = d.get("quit_daq"), d.get("quit_lp")
    r["quit_daq"] = int(qd is not None and not qd["censored"])
    if not d["lp"]:                       # no LP tables: the option's gate = DAQ contact only -> nothing moves
        r.update(n_cov=0, n_ne_daqlp=r["n_ne_daq"], n_flip_daq=0, quit_daqlp=r["quit_daq"], shift_min=0.0,
                 quit_lost=0)
        return r
    k = T.k.to_numpy()
    pos = np.array([RAW[p] for p in T.pos])
    daqlp = T.ne_img_lp.to_numpy(bool)
    r["n_ne_daqlp"] = int(daqlp.sum())
    r["n_flip_daq"] = int((daqlp != T.ne_img.to_numpy(bool)).sum())
    cov = T.lp_cov.to_numpy(bool)
    r["n_cov"] = int(cov.sum())
    att = (T.lp_att35.to_numpy(bool) & cov)
    lponly = np.asarray(engagement_gate(k[cov], att[cov], pos[cov]), bool)
    both = np.asarray(engagement_gate(k[cov], (T.daq35.to_numpy(bool) | att)[cov], pos[cov]), bool)
    r["n_ne_lponly_cov"] = int(lponly.sum())
    r["n_ne_daqlp_cov"] = int(both.sum())
    r["n_flip_lp"] = int((lponly != both).sum())
    r["n_flip_lp_to_eng"] = int((lponly & ~both).sum())
    r["n_flip_lp_to_ne"] = int((~lponly & both).sum())
    if qd is not None and ql is not None:
        r["quit_daqlp"] = int(not ql["censored"])
        r["shift_min"] = (ql["time_engaged_s"] - qd["time_engaged_s"]) / 60.0
        r["quit_lost"] = int(r["quit_daq"] == 1 and r["quit_daqlp"] == 0)
    return r


def main():
    labs = labels()
    D = {}
    for lab in labs:
        f = CACHE / f"{lab}.pkl"
        if not f.exists():
            raise SystemExit(f"missing gate_lp_check cache for {lab}: run `python scripts/gate_lp_check.py extract`")
        D[lab] = pickle.load(open(f, "rb"))
    S = pd.DataFrame([one(D[lab]) for lab in labs])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(f"{len(S)} sessions; without LP tables: {S.label[~S.lp].tolist()}")
    S["moved"] = (S.shift_min.abs() > 1e-9).astype(int)
    agg = dict(sessions=("label", "size"), trials=("n", "sum"), ne_daq=("n_ne_daq", "sum"),
               ne_daqlp=("n_ne_daqlp", "sum"), flip_daq=("n_flip_daq", "sum"),
               sess_flip_daq=("n_flip_daq", lambda x: int((x > 0).sum())),
               cov=("n_cov", "sum"), ne_lponly=("n_ne_lponly_cov", "sum"), ne_daqlp_cov=("n_ne_daqlp_cov", "sum"),
               flip_lp=("n_flip_lp", "sum"), flip_lp_to_eng=("n_flip_lp_to_eng", "sum"),
               flip_lp_to_ne=("n_flip_lp_to_ne", "sum"),
               quits_daq=("quit_daq", "sum"), quits_daqlp=("quit_daqlp", "sum"), quits_moved=("moved", "sum"),
               quits_lost=("quit_lost", "sum"), shift_mean=("shift_min", "mean"), shift_max=("shift_min", "max"))
    A = S.groupby(["animal", "epoch"]).agg(**agg).reset_index()
    A["epoch"] = pd.Categorical(A.epoch, EPS, ordered=True)
    A = A.sort_values(["animal", "epoch"])
    print("\n=== PER ANIMAL x EPOCH (img gate trial counts; quits)")
    print(A.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    E = S.groupby("epoch").agg(**agg).reindex(list(EPS)).reset_index()
    print("\n=== POOLED BY EPOCH (sums over animals)")
    print(E.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== sessions where anything moves")
    cols = ["label", "epoch", "n", "n_ne_daq", "n_ne_daqlp", "n_flip_daq", "n_ne_lponly_cov", "n_ne_daqlp_cov",
            "n_flip_lp", "quit_daq", "quit_daqlp", "shift_min"]
    print(S[(S.n_flip_daq > 0) | (S.n_flip_lp.fillna(0) > 0) | (S.moved > 0)][cols].to_string(index=False))
    from wfield_local.writeguard import assert_writable
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / "lp_gate_option_counts.csv", index=False)          # per session; the tables above re-derive
    print("->", OUT / "lp_gate_option_counts.csv")


if __name__ == "__main__":
    assert ANIMALS == ("PS92", "PS93", "PS94", "PS95")
    main()
