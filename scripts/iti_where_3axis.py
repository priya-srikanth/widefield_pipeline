"""The where-decoder on all THREE axes in one run: lateral, forward (out_ap) and down (down_dv) (Priya 10-27: "run the
3-axis version but low priority"). Sagittal extension = sqrt(forward^2 + down^2) is ~|down| (r 0.96 over all licks),
forward is nearly independent of it (r -0.05), so this mostly re-states the outward and sagittal runs in one frame.

Same decoder as 3.15 / KEY row: per animal x lick set, ridge on Allen areas (lags 0..0.2 s), z-scored within session
(`zscore_check/session_sd.csv`; --unscaled for raw), trained on August pre licks, pre sessions leave-one-session-out,
one ridge per axis. Per session: r per axis (/ animal's pre mean) and the offset beyond the pre pull per axis
(decoded - (a + b * actual), a, b fit on the animal's pre licks). Change vs pre: paired nested bootstrap.
+ lateral = contralesional; + forward = further forward; + down = further down (larger down_dv, as the lick table).
Output <E>/iti_licks/where_3axis[_unscaled]_{sessions,tests}.csv. Usage: python scripts/iti_where_3axis.py [--unscaled]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
AXES = {"lat": 0, "fwd": 1, "down": 4}                       # columns of iti_where_decoder.coords


def main(argv=None) -> int:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from iti_decoder_contrib import feature_names
    unscaled = "--unscaled" in (argv if argv is not None else sys.argv[1:])
    SD = pd.read_csv(il.OUT / "zscore_check" / "session_sd.csv", index_col=0)
    names = feature_names()
    D = wd.load("areas")
    rows = []
    for st in ("iti", "trial"):
        for a in AN:
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                ok = np.isfinite(C[:, list(AXES.values())]).all(1)
                if ok.sum() >= 20:
                    Xs = X[ok] if unscaled else X[ok] / SD.loc[x, names].to_numpy(float)[None, :]
                    S[x] = (d["epoch"], Xs, C[ok][:, list(AXES.values())])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            fits = {}
            for held in [*pre, None]:
                trn = [q for q in pre if q != held]
                Xt, Yt = np.vstack([S[q][1] for q in trn]), np.vstack([S[q][2] for q in trn])
                sc = StandardScaler().fit(Xt)
                fits[held] = (sc, [RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), Yt[:, k])
                                   for k in range(len(AXES))])
            preds = {}
            for x, (ep, X, Y) in S.items():
                sc, ms = fits[x if x in pre else None]
                preds[x] = (ep, Y, np.column_stack([m.predict(sc.transform(X)) for m in ms]))
            Ya = np.vstack([preds[q][1] for q in pre])
            Pa = np.vstack([preds[q][2] for q in pre])
            pull = [np.polyfit(Ya[:, k], Pa[:, k], 1) for k in range(len(AXES))]
            for x, (ep, Y, P) in preds.items():
                for k, ax_ in enumerate(AXES):
                    b, c = pull[k]
                    rows.append(dict(set=st, animal=a, label=x, epoch=ep, axis=ax_, n=len(Y), slope_pre=b,
                                     r=float(np.corrcoef(P[:, k], Y[:, k])[0, 1]),
                                     beyond=float(np.mean(P[:, k] - (c + b * Y[:, k])))))
    R = pd.DataFrame(rows)
    pre = R[R.epoch == "pre"].groupby(["set", "animal", "axis"])[["r", "beyond"]].mean()
    key = list(zip(R.set, R.animal, R.axis))
    R["r_norm"] = R.r / [pre.r.get(k, np.nan) for k in key]
    R["beyond_d"] = R.beyond - [pre.beyond.get(k, np.nan) for k in key]
    sfx = "_unscaled" if unscaled else ""
    R.to_csv(il.OUT / f"where_3axis{sfx}_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"== {'UNSCALED' if unscaled else 'Z-SCORED'} features. Pre pull slope per axis (1 = no pull)")
    print(R.groupby(["set", "axis", "animal"]).slope_pre.first().unstack().round(2).to_string())
    print("\n== per animal (rule 8): offset beyond the pre pull, change vs pre (mm)")
    print(R.pivot_table(index=["set", "axis", "animal"], columns="epoch", values="beyond_d")[EP].round(2).to_string())
    tests = []
    for (st, ax_), g in R.groupby(["set", "axis"]):
        for val in ("r_norm", "beyond_d"):
            prd = {a: h[val].to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: h[val].to_numpy() for a, h in g[g.epoch == ep].groupby("animal")}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = g[g.epoch == ep].groupby("animal")[val].mean() - g[g.epoch == "pre"].groupby("animal")[val].mean()
                tests.append(dict(set=st, axis=ax_, measure=val, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                  n_pos=int((am > 0).sum()), n=int(am.notna().sum())))
    T = pd.DataFrame(tests)
    T.to_csv(il.OUT / f"where_3axis{sfx}_tests.csv", index=False)
    print("\n== change vs pre (paired nested bootstrap); r_norm = r / pre - 1 shown as delta of the ratio; n_pos = animals > pre")
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
