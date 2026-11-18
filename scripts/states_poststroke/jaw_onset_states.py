"""States at the jaw onset (+-0.15 s) of jaw-only / incomplete / contact trials, pre vs post. Scratch."""
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
EP = json.load(open("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/arhmm_20261117/inputs/epochs.json"))


def one(lab):
    from wfield_local import trial_classes as tc
    T = tc.session_trials(lab)
    z = np.load(Zd / f"{lab}.npz")["z"].astype(int)
    t = np.load(FEAT / f"{lab}.npz", allow_pickle=True)["t_daq_s"]
    out = []
    for r in T.itertuples():
        if not np.isfinite(r.jaw_onset_ms):
            continue
        c = r.cue_daq + r.jaw_onset_ms / 1000
        zz = z[np.searchsorted(t, c - 0.15):np.searchsorted(t, c + 0.15)]
        for k in zz[zz >= 0]:
            out.append((lab, r.cls, int(k)))
    return out


if __name__ == "__main__":
    labs = [Path(f).stem for f in sorted(glob.glob(str(Zd / "*.npz")))
            if EP.get(Path(f).stem) and not (EP[Path(f).stem] == "pre" and Path(f).stem[-4:] < "0801")]
    res, fail = ak.fan_sessions(labs, one)
    D = pd.DataFrame([x for _l, r in res for x in r], columns=["label", "cls", "state"])
    D["pp"] = np.where(D.label.map(EP) == "pre", "pre", "post")
    P = D.groupby(["cls", "pp"]).state.value_counts(normalize=True).rename("share").reset_index()
    for (c, pp), g in P.groupby(["cls", "pp"]):
        print(c, pp, "n frames", int((D.cls.eq(c) & D.pp.eq(pp)).sum()), "|",
              ", ".join(f"{s}:{v:.2f}" for s, v in zip(g.state, g.share) if v >= 0.04))
