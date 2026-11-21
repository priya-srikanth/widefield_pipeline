"""Which GLM-HMM role do JAW-ONLY (and, for comparison, NO-RESPONSE) trials fall into? Six-position cohort, PS92-PS95.

Trials = every trial of the analysis sessions in the GLM-HMM table (`behavior_states.glmhmm_table`: all trials, all
classes; August pre, acute, subacute, chronic), whole sessions AND first 95 min. Role = the argmax state's role
(production K5 v2, merged 0+1; refits from s3: ipsimid = outcome observed on ipsi / middle spouts only, init = response
initiation). Share = fraction of that class's trials in each role, per epoch and pooled; plus per-spout counts.
Caveat: at the near-middle spout (close_center) the spout occludes the chin, so jaw-only cannot be called there; a
no-lick close_center trial is classed no_response (FINDINGS 9.1b).
Usage: WF_REPO=... python s6_jaw_only_by_role.py <out_dir>  -> jaw_only_by_glmhmm_role.csv
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline")))
import pandas as pd  # noqa: E402
from wfield_local import behavior_states as B  # noqa: E402

OUT = Path(sys.argv[1])
f = OUT / "jaw_only_by_glmhmm_role.csv"
if f.exists():
    sys.exit(f"refusing to overwrite {f}")
G = B.glmhmm_table()
roles = B.glmhmm_roles(G)
pc = B._pcols(G)
G["role_production"] = [roles[int(pc[i][7:])] for i in G[pc].to_numpy().argmax(1)]
G = B.add_session_time(G, B.load_collected()["trials"])
for v in ("ipsimid", "init"):
    R = pd.read_csv(OUT / f"glmhmm_refit_{v}_K5.csv.gz", usecols=["label", "trial_id", "role"])
    G = G.merge(R.rename(columns={"role": f"role_{v}"}), on=["label", "trial_id"], how="left")
rows = []
for match, D in (("all", G), ("first95", G[~(G.t_min > B.COMMON_SESSION_MIN)])):
    for model in ("production", "ipsimid", "init"):
        for cls in ("jaw_only", "no_response"):
            for ep in ("pre", "acute", "subacute", "chronic", "pooled"):
                for spout in ("all", "close_center", "other_spouts"):
                    d = D[D.cls == cls]
                    if ep != "pooled":
                        d = d[d.epoch == ep]
                    if spout == "close_center":
                        d = d[d.position == "close_center"]
                    elif spout == "other_spouts":
                        d = d[d.position != "close_center"]
                    sh = d[f"role_{model}"].value_counts(normalize=True)
                    r = {"match": match, "model": model, "class": cls, "epoch": ep, "spout": spout, "n_trials": len(d),
                         "n_sessions": d.label.nunique()}
                    r.update({f"share_{k}": float(sh.get(k, 0.0)) for k in B.ROLES})
                    rows.append(r)
T = pd.DataFrame(rows)
# per-session share (animal-weighted check): pooled jaw-only share in engaged role, mean over animals
T.to_csv(f, index=False)
pd.set_option("display.width", 200)
print(T[(T.match == "all") & (T.spout == "all")].round(3).to_string(index=False))
print(T[(T.match == "all") & (T.spout == "close_center") & (T.model == "production")].round(3).to_string(index=False))
