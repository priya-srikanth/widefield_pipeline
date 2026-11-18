"""Per-state running + cue timing for Priya's 11-17 review questions (4 vs 6 vs 13; 10/12/18 running). Scratch."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import analysis_kit as ak

S = Path(sys.argv[1])
SD = S / "arhmm_local_fit" / "states_pooled_all_s1"
FEAT = S / "arhmm_feat_v3" / "sessions"
labs = sorted(Path(f).stem for f in glob.glob(str(SD / "*.npz")))[::2]


def one(lab):
    z = np.load(SD / f"{lab}.npz")["z"].astype(int)
    F = np.load(FEAT / f"{lab}.npz", allow_pickle=True)
    cols = [str(c) for c in F["features"]]
    X, t = F["X"], F["t_daq_s"]
    b = ak.session_behavior(lab[:5] + lab[-4:])
    cue = np.sort(b.cue_samples / b.sample_rate) if b is not None else np.array([])
    on = np.r_[0, np.flatnonzero(np.diff(z)) + 1]
    on = on[z[on] >= 0]
    rows = []
    for i in on:
        if cue.size:
            j = np.searchsorted(cue, t[i]) - 1
            dt = t[i] - cue[j] if j >= 0 else np.nan
        else:
            dt = np.nan
        rows.append({"session": lab, "state": z[i], "dt_cue": dt})
    fr = pd.DataFrame({"state": z, "run_fwd": X[:, cols.index("run_fwd")], "run_back": X[:, cols.index("run_back")]})
    return pd.DataFrame(rows), fr[fr.state >= 0]


if __name__ == "__main__":
    res, fail = ak.fan_sessions(labs, one)
    print("failed:", fail)
    out = [r for _k, r in res]
    ons = pd.concat([o[0] for o in out if o is not None])
    frm = pd.concat([o[1] for o in out if o is not None])
    g = frm.groupby("state")
    T = pd.DataFrame({"run_fwd_mm_s": g.run_fwd.mean(), "run_back_mm_s": g.run_back.mean(),
                      "frac_fwd_gt3": g.run_fwd.apply(lambda v: (v > 3).mean()),
                      "frac_back_gt0.5": g.run_back.apply(lambda v: (v > 0.5).mean())})
    o = ons.groupby("state").dt_cue
    T["onsets"] = o.size()
    T["frac_onset_0_1s_after_cue"] = o.apply(lambda v: ((v >= 0) & (v < 1)).mean())
    T["median_s_after_cue"] = o.median()
    print(T.round(3).to_string())
    T.to_csv(S / "arhmm_state_check.csv")
