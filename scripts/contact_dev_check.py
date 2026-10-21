"""Direct check of the contact confound in the within-spout toward-midline slope (Priya 10-21).
Per session x lateral spout, on the trials the within-spout analysis uses (cam4, lick-aligned): does the window's DAQ
contact count rise with the lick's deviation TOWARD THE MIDLINE (within spout), raw and with the analysis' other
covariates (reach, n_licks, latency) held fixed? Reads behaviour only (local input cache)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

LATERAL = ("close_contra", "close_ipsi", "far_contra", "far_ipsi")


def one(label):
    from scripts import within_spout_deviation as W
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.paths import PathResolver
    B = ce.load_behaviour(config.animal_of(label), "2026" + label.split("_")[-1], PathResolver())
    E = B.trials[B.trials.engaged & np.isfinite(B.trials.cue_s)].reset_index(drop=True)
    T = W.trial_table(B, E, "", "lick")
    rows = []
    for p in LATERAL:
        G = T[T.pos == p].dropna(subset=["lat_mean", "reach_mean"])
        if len(G) < W.MIN_TRIALS:
            continue
        mid = (G.lat_mean - G.lat_mean.mean()).to_numpy() * W.side_sign([p])[0] / 10.0   # per 10 deg toward midline
        c = G.n_contact.to_numpy(float)
        X = np.column_stack([np.ones(len(G)), mid, G.reach_mean, G.n_licks, G.latency]).astype(float)
        b = np.linalg.lstsq(X, c, rcond=None)[0]
        rows.append({"label": label, "pos": p, "n": len(G), "hit_frac": float((c > 0).mean()),
                     "mean_contacts": float(c.mean()), "lat_sd": float(G.lat_mean.std()),
                     "r_mid_contact": float(np.corrcoef(mid, c)[0, 1]) if c.std() > 0 else np.nan,
                     "slope_raw": float(np.polyfit(mid, c, 1)[0]) if c.std() > 0 else np.nan,
                     "slope_adj": float(b[1])})
    return rows


def main():
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    labels = [x for x in ce.sessions_available() if x in set(ak.curated_labels()) and x[-4:] >= "0806"]
    got, fail = ak.fan_sessions(labels, one, jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    for it, err in fail:
        print("FAILED", it, err)
    D = pd.DataFrame([r for _, rr in got for r in rr])
    D["animal"], D["epoch"] = D.label.str[:4], D.label.map(epochs.epoch_of)
    D.to_csv(sys.argv[2] if len(sys.argv) > 2 else "contact_dev_check.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    order = ["pre", "acute", "subacute", "chronic"]
    for col, title in (("slope_adj", "contacts per 10 deg TOWARD MIDLINE, reach / n_licks / latency held fixed"),
                       ("r_mid_contact", "raw correlation, toward-midline deviation vs contact count"),
                       ("hit_frac", "fraction of analysed trials with >= 1 contact")):
        t = D.groupby(["pos", "animal", "epoch"])[col].mean().unstack("epoch").reindex(columns=order)
        print(f"\n=== {title} (session means)\n" + t.round(3).to_string())
    print("\n=== all four lateral spouts pooled, per animal (session x spout means)")
    print(D.groupby(["animal", "epoch"])[["slope_adj", "r_mid_contact", "hit_frac"]].mean().unstack("epoch")
          .reindex(columns=order, level=1).round(3).to_string())


if __name__ == "__main__":
    main()
