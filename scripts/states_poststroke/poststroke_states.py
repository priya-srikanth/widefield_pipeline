"""Post-stroke readouts of the behavioral states, delta vs each animal's OWN August pre (11-17). Scratch; locanmf env.

A. AR-HMM (pooled_all seed 1): fraction of decoded frames in each of Priya's 11-17 syllable groups, per session.
B. GLM-HMM (design v2, K = 5): mean posterior occupancy of each state per session (+ the state profile, to name them).
Per-animal table first (CLAUDE.md rule 8), then `analysis_kit.boot_delta` (animals -> sessions, paired) per epoch with
the bootstrap p and Holm over the family. Writes CSVs next to this script's scratch outputs.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import analysis_kit as ak

S = Path(sys.argv[1])
OUT = S / "poststroke_states"
OUT.mkdir(exist_ok=True)
EPOCHS = json.load(open("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/arhmm_20261117/inputs/epochs.json"))
POST = ["acute", "subacute", "chronic"]
pd.set_option("display.width", 220)

GROUPS = {"quiet": [5, 8, 19], "nose_slow": [1], "lick_ipsi": [0, 7, 9, 23], "lick_central": [10, 12, 18],
          "lick_contra": [2, 15, 20], "jaw_no_tongue": [21, 22], "push": [11, 13], "walk_slow": [4], "walk": [6]}


def epoch(lab):
    e = EPOCHS.get(lab)
    if e == "pre" and lab[-4:] < "0801":
        return None                                    # August-only pre baseline (orofacial / encoding rule, 10-18)
    return e


def readout(T: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    T = T.copy()
    T["animal"], T["epoch"] = T.label.str[:4], T.label.map(epoch)
    T = T[T.epoch.notna()]
    print(f"\n######## {name}: sessions per animal x epoch")
    print(T.groupby(["animal", "epoch"]).size().unstack()[["pre", *POST]].to_string())
    rows = []
    for c in cols:
        m = T.groupby(["animal", "epoch"])[c].mean().unstack()[["pre", *POST]]
        d = m[POST].sub(m["pre"], axis=0)
        print(f"\n--- {c}: per-animal mean (pre) and delta vs own pre")
        print(pd.concat([m[["pre"]], d.add_prefix("d_")], axis=1).round(3).to_string())
        pre = {a: g[c].to_numpy() for a, g in T[T.epoch == "pre"].groupby("animal")}
        for e in POST:
            post = {a: g[c].to_numpy() for a, g in T[T.epoch == e].groupby("animal")}
            iv, dr = ak.boot_delta(post, pre, np.random.default_rng(0), draws=True)
            n_same = int((np.sign(d[e]) == np.sign(iv.point)).sum())
            rows.append({"readout": name, "quantity": c, "epoch": e, "pre_mean": m["pre"].mean(), "delta": iv.point,
                         "lo": iv.lo, "hi": iv.hi, "p_boot": ak.boot_p(dr), "animals_same_sign": n_same,
                         "n_animals": iv.n_animals})
    R = pd.DataFrame(rows)
    R["p_holm"] = ak.holm(R.p_boot.to_numpy())
    R["ci_excl_0"] = (R.lo > 0) | (R.hi < 0)
    return R


# ---- A. AR-HMM syllable groups
Zd = S / "arhmm_local_fit" / "states_pooled_all_s1"
rows = []
for f in sorted(glob.glob(str(Zd / "*.npz"))):
    z = np.load(f)["z"]
    z = z[z >= 0]
    r = {"label": Path(f).stem}
    for g, ks in GROUPS.items():
        r[g] = np.isin(z, ks).mean()
    rows.append(r)
A = pd.DataFrame(rows)
A.to_csv(OUT / "arhmm_group_usage_per_session.csv", index=False)
RA = readout(A, list(GROUPS), "AR-HMM group usage (fraction of frames)")

# ---- B. GLM-HMM K = 5 occupancy (mean posterior per session)
d = pd.read_csv(S / "glmhmm_v1" / "states_pooled_all_K5_v2.csv.gz")
pc = [c for c in d if c.startswith("p_state")]
d["attempt"] = d.cls.isin(["contact", "incomplete"])
d["no_resp"] = d.cls.eq("no_response")
d["run"] = (d.run_frac_precue.fillna(0) + d.run_frac_iti.fillna(0)) / 2
prof = pd.DataFrame({c: [np.average(d[v], weights=d[c]) for v in ("attempt", "no_resp", "run")] for c in pc},
                    index=["attempt", "no_resp", "run"]).T
prof["occupancy"] = d[pc].mean().to_numpy()
print("\n######## GLM-HMM K = 5 state profile (posterior-weighted, all trials)")
print(prof.round(3).to_string())
B = d.groupby("label")[pc].mean().reset_index()
B.to_csv(OUT / "glmhmm_K5_occupancy_per_session.csv", index=False)
RB = readout(B, pc, "GLM-HMM K5 occupancy (mean posterior)")

R = pd.concat([RA, RB])
R.to_csv(OUT / "poststroke_delta_vs_own_pre.csv", index=False)
print("\n######## SUMMARY (boot_delta, paired animals -> sessions; Holm within each readout)")
print(R.round(3).to_string(index=False))
