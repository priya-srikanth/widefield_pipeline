"""Per-lick MISS rate by spout side x epoch -- the PS92-95 check of the AR-HMM cohort's "licks and misses" result
(github.com/htong1/Priya_ARHMM2026, PS46-50: contra miss-rate 25 % -> 40-90 % acute, ~45-50 % plateau; Priya 10-23).

    python scripts/lick_contact_by_epoch.py

Two levels, per animal x epoch (August pre; session means first):
  TRIAL  existing `scripts/trial_attempts.py` table (hit = DAQ contact / incomplete = LP lick without contact):
         far-contra attempt rate, completion at far / close contra and ipsi.
  LICK   v36 `licks_all.csv`: fraction of LP licks with onset cue 0..4 s on RIGHT-spout (contra; all lesions left) /
         LEFT-spout (ipsi) trials that made NO spout contact (`contact`); sessions with >= 10 such licks.
-> stdout + ``<run>/imaging_encoding/qc/lick_contact_by_epoch.csv``.
"""
import sys, pathlib
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wfield_local.paths import PathResolver
from wfield_local import orofacial_outputs as oo, orofacial_cohort as oc, epochs
rv = PathResolver()
q = oo.root(rv) / oc.RUN / "imaging_encoding" / "qc"
S = pd.read_csv(q / "trial_attempts_sessions.csv")
S = S[(S.epoch != "pre") | S.label.str.contains("_08")]
EP = ["pre", "acute", "subacute", "chronic"]
cols = ["attempt_rate_far_contra", "completion_far_contra", "completion_close_contra", "completion_far_ipsi", "completion_close_ipsi"]
print("TRIAL level (existing trial_attempts; session means, August pre):")
print(S.groupby(["animal", "epoch"])[cols].mean().reindex(EP, level="epoch").round(2).to_string())
rows = []
base = pathlib.Path(rv.root("orofacial")) / oc.RUN / "sessions"
for a in ("PS92", "PS93", "PS94", "PS95"):
    for d in sorted(p.name for p in (base / a).iterdir() if p.is_dir()):
        ep = epochs.epoch_of(f"{a}_{d[4:]}")
        if ep not in EP or (ep == "pre" and d[4:6] != "08"):
            continue
        f = base / a / d / "licks_all.csv"
        if not f.exists():
            continue
        L = pd.read_csv(f, usecols=["position", "contact", "t_ms"])
        L = L[(L.t_ms >= 0) & (L.t_ms < 4000)].dropna(subset=["position"])
        for side, pos in (("contra", ("far_R", "close_R")), ("ipsi", ("far_L", "close_L"))):
            m = L.position.isin(pos)
            if m.sum() >= 10:
                rows.append({"animal": a, "date": d, "epoch": ep, "side": side, "n": int(m.sum()),
                             "miss": 1 - L.contact[m].astype(float).mean()})
R = pd.DataFrame(rows)
print("\nLICK level: fraction of licks on R (contra) / L (ipsi) spout trials WITHOUT contact (cue 0-4 s; session means):")
T = R.groupby(["animal", "side", "epoch"]).miss.mean().unstack("epoch")[EP].round(2)
T["n_sessions"] = R.groupby(["animal", "side"]).date.nunique()
print(T.to_string())
R.to_csv(q / "lick_contact_by_epoch.csv", index=False, float_format="%.4f")
print(f"-> {q / 'lick_contact_by_epoch.csv'}")
