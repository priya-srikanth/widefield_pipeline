"""Chance level for the in-mouth lick (cam1) <-> 3-D jaw match of `s10a_onsets_analysis.py`: the same rise test
(max of the 3-D jaw displacement in t +- W minus its min in t - W - 80 .. t, matched if >= 0.20 mm) at a pseudo-event
time t = cue - 300 ms (spout docked, before the cue), on licked canonical trials with no lick or in-mouth event in
cue - 0.5 .. cue (`ctrl_ok`-like: no fused lick or in-mouth event in cue - 2.0 .. cue). Per animal x epoch.

    python s10a_mouth_match_chance.py <s10a_onsets_dir>
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W_MS = (40.0, 60.0, 80.0)
T_PSEUDO = -300.0
RISE_UM = 200.0


def main(od: str) -> None:
    od = Path(od)
    W = pd.read_csv(od / "s10a_trials_ccB1123.csv.gz", usecols=["label", "animal", "epoch", "trial_id", "ctrl_ok"])
    W = W[W.ctrl_ok.astype(str).str.lower().eq("true")]
    rows = []
    for lab, g in W.groupby("label", sort=True):
        J = pickle.loads((od / "cache_ccB1123" / f"{lab}.pkl").read_bytes())["jaw3d"]
        if J is None:
            continue
        tid = J["trial_id"]
        want = set(g.trial_id.astype(int))
        brk = np.flatnonzero(np.diff(tid)) + 1
        for a, b in zip(np.r_[0, brk], np.r_[brk, len(tid)]):
            t_id = int(tid[a])
            if t_id not in want:
                continue
            tt, dd = J["t_daq_ms"][a:b], J["d_um"][a:b].astype(float)
            r = {"label": lab, "animal": g.animal.iloc[0], "epoch": g.epoch.iloc[0], "trial_id": t_id}
            for w in W_MS:
                win = (tt >= T_PSEUDO - w) & (tt <= T_PSEUDO + w)
                pre = (tt >= T_PSEUDO - w - 80) & (tt <= T_PSEUDO)
                ok = np.isfinite(dd[win]).any() and np.isfinite(dd[pre]).any()
                r[f"match_{int(w)}"] = (np.nanmax(dd[win]) - np.nanmin(dd[pre]) >= RISE_UM) if ok else np.nan
            rows.append(r)
        print(lab, flush=True)
    R = pd.DataFrame(rows)
    cols = [f"match_{int(w)}" for w in W_MS]
    A = R.groupby(["animal", "epoch"])[cols].mean().reset_index()
    A["n"] = R.groupby(["animal", "epoch"]).size().to_numpy()
    A.to_csv(od / "tables_ccB1123" / "mouth_match_chance_animal_epoch.csv", index=False, float_format="%.4g")
    E = A.groupby("epoch")[cols].mean().reset_index()
    E.to_csv(od / "tables_ccB1123" / "mouth_match_chance_epoch.csv", index=False, float_format="%.4g")
    print(A.round(3).to_string())
    print(E.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
