"""Slide 21 check: position/state change as a fraction of each animal's OWN pre level (vs the absolute delta)."""
import numpy as np
import pandas as pd
from wfield_local.analysis_kit import boot_delta, boot_p, holm

F = ("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/rebuild_figs/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_"
     "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/state_position_sessions.csv")
D = pd.read_csv(F)
EP = ["acute", "subacute", "chronic"]
for metric in ("retained", "balanced_acc"):
    print(f"\n=== {metric}: relative change = post / own pre mean - 1 ===")
    res, keys = {}, []
    for ro, g in D.groupby("readout"):
        pm = g[g.epoch == "pre"].groupby("animal")[metric].mean()
        tab = g.groupby(["animal", "epoch"])[metric].mean().unstack()[EP].div(pm, axis=0) - 1
        print(ro, "per animal:\n", tab.round(3).to_string())
        pre = {a: list(x[metric] / pm[a] - 1) for a, x in g[g.epoch == "pre"].groupby("animal")}
        for e in EP:
            post = {a: list(x[metric] / pm[a] - 1) for a, x in g[g.epoch == e].groupby("animal")}
            rng = np.random.default_rng(0)
            iv, dr = boot_delta(post, pre, rng, draws=True)
            res[(ro, e)] = (iv, boot_p(dr))
            keys.append((ro, e))
    ph = holm([res[k][1] for k in keys])
    for k, pa in zip(keys, ph):
        iv, p = res[k]
        print(f"  {k[0]:8s} {k[1]:8s} rel {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}] p {p:.4f} Holm {pa:.4f}")
