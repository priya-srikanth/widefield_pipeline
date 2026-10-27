"""ATTRIBUTION (not a correction): how much of the like-for-like read-out shift could a pan-cortical change in the lick
response SIZE produce through the frozen decoder? (Priya 10-27: "a pan-cortical activity change may be biological, i
wouldn't want to get rid of it" -> "you can run the prediction step").

Frozen KEY-row lateral decoder per lick set (z-scored Allen areas, lags 0..0.2 s; ridge fit on the animal's pre licks,
as `iti_lick_order.predictions`). Per animal: m = mean activity vector of its pre licks (the lick-response PATTERN).
Per session: g = projection of (session mean - m) onto m, / |m|^2 -- the response-size change along the pre pattern
(g = +0.5: the average lick response is 1.5x its pre size in that pattern); the GAIN-ONLY prediction = decoder output
for (1 + g) m minus decoder output for m (mm). Compared with the observed like-for-like shift
(lick_matched_readout.py, scope all spouts) per session; change vs pre by the paired nested bootstrap.
Outputs <E>/iti_licks/gain_prediction_sessions.csv, figures/lick_gain_prediction.png.
Usage: python scripts/lick_gain_prediction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_first_lick as fl  # noqa: E402
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
SETS = {"iti": "ITI", "trial": "cued"}


def run() -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from iti_decoder_contrib import feature_names
    from wfield_local import orofacial_cohort as oc
    SD = pd.read_csv(il.OUT / "zscore_check" / "session_sd.csv", index_col=0)
    names = feature_names()
    D = wd.load("areas")
    rows = []
    for st in SETS:
        for a in AN:
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                pos = np.array([q if q in fl.POS or "_" not in q else oc.pos_key(q, "L")
                                for q in d[f"{st}_pos"].astype(str)])
                ok = np.isfinite(C[:, 0]) & np.isin(pos, fl.POS)
                if ok.sum() >= 20:
                    S[x] = (d["epoch"], X[ok] / SD.loc[x, names].to_numpy(float)[None, :], C[ok, 0])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            Xp = np.vstack([S[q][1] for q in pre])
            sc = StandardScaler().fit(Xp)
            m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xp), np.concatenate([S[q][2] for q in pre]))
            mu = Xp.mean(0)
            base = float(m.predict(sc.transform(mu[None, :]))[0])
            unit = float(m.predict(sc.transform(2 * mu[None, :]))[0]) - base     # read-out shift for g = +1
            for x, (ep, X, _y) in S.items():
                dlt = X.mean(0) - mu
                g = float(dlt @ mu / (mu @ mu))
                full = float(m.predict(sc.transform(X.mean(0)[None, :]))[0]) - base
                rows.append(dict(set=st, animal=a, label=x, epoch=ep, g=g, pred_gain=g * unit, unit_shift=unit,
                                 pred_full=full, cos_delta_pattern=float(dlt @ mu / (np.linalg.norm(dlt) *
                                                                                np.linalg.norm(mu) + 1e-12))))
    return pd.DataFrame(rows)


def main() -> int:
    G = run()
    M = pd.read_csv(il.OUT / "matched_sessions.csv")
    M = M[(M.scope == "all spouts") & (M.cls == "all")][["set", "label", "resid"]]
    G = G.merge(M, on=["set", "label"], how="left")
    G.to_csv(il.OUT / "gain_prediction_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print("== decoder read-out shift for a pure doubling of the pre lick-response pattern (mm), per animal")
    print(G.groupby(["set", "animal"]).unit_shift.first().round(3).unstack().to_string())
    print("== per animal x epoch (rule 8): response-size change g, gain-only predicted shift, observed like-for-like")
    print(G.groupby(["set", "animal", "epoch"])[["g", "pred_gain", "resid", "cos_delta_pattern"]].mean()
          .round(3).unstack("epoch").to_string())
    for st in SETS:
        g = G[G.set == st]
        for val in ("g", "pred_gain", "resid"):
            prd = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            out = []
            for ep in EP:
                post = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                out.append(f"{ep} {ci.point:+.3f} [{ci.lo:+.3f},{ci.hi:+.3f}]")
            print(f"{SETS[st]:5s} {val:9s} change vs pre: " + " | ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
