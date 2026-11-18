"""Why does 'jaw without visible tongue' (AR-HMM 21 + 22) fall post-stroke when jaw-only TRIALS rise? (Priya 11-17)
(1) which syllable groups fill the response window (cue .. cue + min(stop, 2 s)) on jaw-only vs contact vs incomplete
trials, pre vs post; (2) share of 21 / 22 frames within 0.5 s of a camera lick (lick-bout context). Scratch; locanmf."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import analysis_kit as ak

S = Path(sys.argv[1])
Zd = S / "arhmm_local_fit" / "states_pooled_all_s1"
FEAT = S / "arhmm_feat_v3" / "sessions"
EPOCHS = json.load(open("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/arhmm_20261117/inputs/epochs.json"))
GROUPS = {"quiet": [5, 8, 19], "nose_slow": [1], "lick_ipsi": [0, 7, 9, 23], "lick_central": [10, 12, 18],
          "lick_contra": [2, 15, 20], "jaw_no_tongue": [21, 22], "push": [11, 13], "walk_slow": [4], "walk": [6]}
G = np.full(32, "other", dtype=object)
for g, ks in GROUPS.items():
    G[ks] = g


def one(lab):
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import trial_classes as tc
    from wfield_local.paths import PathResolver
    T = tc.session_trials(lab)
    if T is None:
        return None
    z = np.load(Zd / f"{lab}.npz")["z"].astype(int)
    t = np.load(FEAT / f"{lab}.npz", allow_pickle=True)["t_daq_s"]
    rows = []
    for r in T.itertuples():
        a = r.cue_daq
        b = a + min(r.stop_ms / 1000.0, 2.0) if np.isfinite(r.stop_ms) else a + 2.0
        i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
        zz = z[i0:i1]
        zz = zz[zz >= 0]
        if not len(zz):
            continue
        row = {"label": lab, "trial_id": r.trial_id, "cls": r.cls}
        gz = G[zz]
        for g in [*GROUPS, "other"]:
            row[g] = float(np.mean(gz == g))
        rows.append(row)
    animal, date = lab.split("_")
    L = pd.read_csv(Path(oo.session_dir(PathResolver(), oc.RUN, animal, date)) / "licks_all.csv", usecols=["daq_on_s"])
    lk = np.sort(L.daq_on_s.dropna().to_numpy(float))
    ctx = {}
    for k in (21, 22):
        tk = t[z == k]
        if len(tk) and len(lk):
            j = np.clip(np.searchsorted(lk, tk), 1, len(lk) - 1)
            dmin = np.minimum(np.abs(tk - lk[j - 1]), np.abs(tk - lk[j]))
            ctx[k] = (len(tk), float(np.mean(dmin <= 0.5)))
    return pd.DataFrame(rows), ctx


def epoch(lab):
    e = EPOCHS.get(lab)
    return None if e is None or (e == "pre" and lab[-4:] < "0801") else e


if __name__ == "__main__":
    labs = [Path(f).stem for f in sorted(glob.glob(str(Zd / "*.npz"))) if epoch(Path(f).stem)]
    res, fail = ak.fan_sessions(labs, one)
    print("failed:", fail)
    W = pd.concat([r[1][0] for r in res if r[1] is not None])
    W["animal"], W["epoch"] = W.label.str[:4], W.label.map(epoch)
    W.to_csv(S / "poststroke_states" / "response_window_groups_per_trial.csv", index=False)
    pd.set_option("display.width", 220)
    print("\ntrials per class x epoch:")
    print(W.groupby(["cls", "epoch"]).size().unstack()[["pre", "acute", "subacute", "chronic"]].to_string())
    cols = [*GROUPS, "other"]
    print("\nresponse-window composition (mean share of frames; animal-weighted), by class x pre/post:")
    W["pp"] = np.where(W.epoch == "pre", "pre", "post")
    A = W.groupby(["cls", "pp", "animal"])[cols].mean().groupby(["cls", "pp"]).mean()
    print(A.round(3).to_string())
    print("\njaw-only trials, per animal (share of window in each group):")
    print(W[W.cls == "jaw_only"].groupby(["animal", "pp"])[cols].mean().round(3).to_string())
    print("\n21 / 22 frames within 0.5 s of a camera lick (frame-weighted, by epoch):")
    rows = []
    for lab, r in res:
        if r is None:
            continue
        for k, (n, f) in r[1].items():
            rows.append({"label": lab, "state": k, "n": n, "near_lick": f, "epoch": epoch(lab)})
    C = pd.DataFrame(rows)
    print(C.groupby(["state", "epoch"]).apply(lambda g: np.average(g.near_lick, weights=g.n)).unstack().round(3)
          .to_string())
