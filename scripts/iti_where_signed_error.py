"""Signed lateral decoding error per lick, ITI vs in-trial (Priya 10-26: "the predicted lateral deviation (with
   directionality) per lick in ITI vs in-trial licks"). Frozen pre-stroke decoder, per-lick predictions from
   `iti_where_scatter.py` (where_preds.csv). err = decoded - actual lateral (mm; + = contralesional); change vs the
   animal's PRE error for the same spout (in-trial: target; ITI: nearest spout), mean over spouts per session.
   "on pre landing" = licks within 0.25 mm of the animal's pre landing mean for that spout (actual position held
   fixed, so the decoder's pull toward the middle for off-centre licks cannot make the shift).
Usage: python scripts/iti_where_signed_error.py <iti_licks dir>; writes where_signed_error_tests.csv there.
"""
import sys

import numpy as np
import pandas as pd

from wfield_local import analysis_kit as ak

E = sys.argv[1]
pd.set_option("display.width", 220)
p = pd.read_csv(f"{E}/where_preds.csv")
p["spout"] = np.where(p["set"] == "trial", p["pos"], p["near_pos"])
p = p.dropna(subset=["act_lat", "dec_lat", "spout"])
p["err"] = p.dec_lat - p.act_lat
pre_land = p[p.epoch == "pre"].groupby(["set", "animal", "spout"]).act_lat.mean().rename("land_pre")
p = p.join(pre_land, on=["set", "animal", "spout"])
p["on"] = (p.act_lat - p.land_pre).abs() < 0.25
EP = ["acute", "subacute", "chronic"]
rows = []
for sel, d in (("all licks", p), ("on pre landing", p[p.on])):
    # session x spout mean error, minus the animal's pre mean for that spout, then mean over spouts per session
    s = d.groupby(["set", "animal", "epoch", "label", "spout"]).err.agg(["mean", "size"]).reset_index()
    s = s[s["size"] >= 5]
    pre = s[s.epoch == "pre"].groupby(["set", "animal", "spout"])["mean"].mean().rename("pre")
    s = s.join(pre, on=["set", "animal", "spout"]).dropna(subset=["pre"])
    s["d"] = s["mean"] - s["pre"]
    ses = s.groupby(["set", "animal", "epoch", "label"]).d.mean().reset_index()
    for st in ("iti", "trial"):
        x = ses[ses.set == st]
        tab = x[x.epoch.isin(EP)].groupby(["animal", "epoch"]).d.mean().unstack("epoch")[EP]
        print(f"\n== {st}, {sel}: change in signed error vs pre (mm, + = contralesional), per animal")
        print(tab.round(2).to_string())
        prd = {a: g.d.to_numpy() for a, g in x[x.epoch == "pre"].groupby("animal")}
        for ep in EP:
            post = {a: g.d.to_numpy() for a, g in x[x.epoch == ep].groupby("animal")}
            ci = ak.boot_delta(post, prd, np.random.default_rng(0))
            n_up = int((tab[ep] > 0).sum())
            rows.append(dict(set=st, licks=sel, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi, n_contra=n_up,
                             n=int(tab[ep].notna().sum())))
    # absolute level for orientation: pre signed error of on-target licks per spout
out = pd.DataFrame(rows)
print("\n== pooled (paired nested bootstrap animals -> sessions; n_contra = animals with + mean)")
print(out.round(3).to_string(index=False))
out.to_csv(f"{E}/where_signed_error_tests.csv", index=False)
