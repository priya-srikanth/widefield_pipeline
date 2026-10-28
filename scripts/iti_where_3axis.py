"""The where-decoder on all THREE axes in one run: lateral, forward (out_ap) and down (down_dv) (Priya 10-27: "run the
3-axis version but low priority"). Sagittal extension = sqrt(forward^2 + down^2) is ~|down| (r 0.96 over all licks),
forward is nearly independent of it (r -0.05), so this mostly re-states the outward and sagittal runs in one frame.

Same decoder as 3.15 / KEY row: per animal x lick set, ridge on Allen areas WITH the ALM / tjM1 split (every area
decoder since 10-27 late; iti_where_decoder.area_basis), lags 0..0.2 s, z-scored within session (frame-level SDs;
--unscaled for raw), trained on August pre licks, pre sessions leave-one-session-out, one ridge per axis. Per session:
r per axis (/ animal's pre mean) and the LIKE-FOR-LIKE offset per axis (lfl; replaces 'offset beyond the pre pull',
Priya 10-27 late "yes replace beyond pull"): mean over the session's licks of decoded - the animal's PRE licks of the set
that landed at the same place on that axis (`likeforlike.offset`, Gaussian by default / WFIELD_LFL_REF=loclin);
lfl_d = minus the animal's pre mean. Change vs pre: paired nested bootstrap.
+ lateral = contralesional; + forward = further forward; + down = further down (larger down_dv, as the lick table).
Output <E>/iti_licks/where_3axis[_unscaled]_{sessions,tests}.csv and per-lick where_3axis[_unscaled]_preds<PRED_TAG>.csv.gz
(basis-specific cache name, il.pred_file). Usage: python scripts/iti_where_3axis.py [--unscaled]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
import likeforlike as lfl  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
AXES = {"lat": 0, "fwd": 1, "down": 4}                       # columns of iti_where_decoder.coords


def main(argv=None) -> int:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    unscaled = "--unscaled" in (argv if argv is not None else sys.argv[1:])
    il.split_only("iti_where_3axis")
    # Allen areas in the basis (split by default, Priya 10-27 late) + frame SDs: the shared loader
    D, names, SD = wd.area_basis()
    rows, lick_rows = [], []
    POSL = {}
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
                    POSL[x] = np.asarray(d[f"{st}_pos"])[ok]
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
            for x, (ep, Y, P) in preds.items():           # per-LICK predictions (histograms / recovery / overcorrection)
                lick_rows.append(pd.DataFrame({"set": st, "animal": a, "label": x, "epoch": ep, "pos": POSL[x],
                                               **{f"act_{k}": Y[:, i] for i, k in enumerate(AXES)},
                                               **{f"dec_{k}": P[:, i] for i, k in enumerate(AXES)}}))
            for x, (ep, Y, P) in preds.items():
                for k, ax_ in enumerate(AXES):
                    rows.append(dict(set=st, animal=a, label=x, epoch=ep, axis=ax_, n=len(Y),
                                     r=float(np.corrcoef(P[:, k], Y[:, k])[0, 1])))
    R = pd.DataFrame(rows)
    sfx_ = "_unscaled" if unscaled else ""
    L = pd.concat(lick_rows, ignore_index=True)
    L.to_csv(il.pred_file(f"where_3axis{sfx_}_preds.csv.gz"), index=False)
    for ax_ in AXES:                               # like-for-like offset per lick, per axis (same-place pre reference)
        L[f"lfl_{ax_}"] = lfl.offset(L, f"act_{ax_}", f"dec_{ax_}", lfl.SIGMA_MM, f"3-axis {ax_} (mm)")[0]
    M = L.groupby(["set", "label"])[[f"lfl_{ax_}" for ax_ in AXES]].mean()
    R["lfl"] = [M[f"lfl_{ax_}"].get((s_, x_), np.nan) for s_, x_, ax_ in zip(R.set, R.label, R.axis)]
    pre = R[R.epoch == "pre"].groupby(["set", "animal", "axis"])[["r", "lfl"]].mean()
    key = list(zip(R.set, R.animal, R.axis))
    R["r_norm"] = R.r / [pre.r.get(k, np.nan) for k in key]
    R["lfl_d"] = R.lfl - [pre.lfl.get(k, np.nan) for k in key]
    sfx = "_unscaled" if unscaled else ""
    R.to_csv(il.OUT / f"where_3axis{sfx}_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"== {'UNSCALED' if unscaled else 'Z-SCORED'} features ({il.basis_label()})")
    print("\n== per animal (rule 8): like-for-like offset (same place), change vs pre (mm)")
    print(R.pivot_table(index=["set", "axis", "animal"], columns="epoch", values="lfl_d")[EP].round(2).to_string())
    tests = []
    for (st, ax_), g in R.groupby(["set", "axis"]):
        for val in ("r_norm", "lfl_d"):
            prd = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal")}
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
