"""Is the frozen lick-position decoder's post-stroke r loss more than ELAPSED-TIME DRIFT? (handoff 00f to-do 3)

Representations drift with time even without a lesion, and every post session is further in time from the pre training
sessions than the pre sessions are from each other. Single-session decoders make the gap explicit:
  pre -> pre    train a ridge decoder on ONE pre session, test on another pre session: r vs |day gap| (the drift curve)
  pre -> post   train on ONE pre session, test on a post session: r at that gap
Per animal x lick set x axis (lateral, sagittal): fit r = a + b * log(1 + gap) on the pre -> pre pairs; for every
pre -> post pair, residual = r - drift prediction at its gap (extrapolated beyond the pre range for chronic gaps, which
the log fit makes conservative-to-neutral; reported separately). Per post session: mean residual over its training
sessions; change per epoch = nested bootstrap of those residuals (pre->pre residuals as the reference).
Features = Allen areas, lags 0..0.2 s (as 3.15), z-scored within session (`zscore_check/session_sd.csv`).
--june (Priya 10-27): add the June 6-8 pre sessions (`iti_june_ref.py`, a separate folder; never in the August
baseline) as a NO-STROKE LONG-GAP reference: pairs August pre <-> June (both directions; ~60 days, about the chronic
gap). The drift curve is then fit on pre -> pre AND June pairs, so chronic gaps are interpolated, not extrapolated; the
June residual against the August-only curve says whether that extrapolation was right. Caveat: June is earlier in
training, so June <-> August drift includes learning (makes the test conservative).
Output <E>/iti_licks/where_drift[_june]_{pairs,tests}.csv, figures/iti_where_drift[_june].png.
Usage: python scripts/iti_where_drift.py [--june]
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
AXES = {"lat": 0, "sag": 3}


def _day(label: str) -> int:
    md = label.split("_")[-1]
    return date(2026, int(md[:2]), int(md[2:])).toordinal()


def pairs(june: bool = False) -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from iti_decoder_contrib import feature_names
    S = pd.read_csv(il.OUT / "zscore_check" / "session_sd.csv", index_col=0)
    names = feature_names()
    D = wd.load("areas")
    if june:
        from iti_june_ref import JUNE
        DJ = wd.load("areas", JUNE / "sessions")
        for d in DJ.values():
            d["epoch"] = "june"
        D = {**D, **DJ}
        S = pd.concat([S, pd.read_csv(JUNE / "session_sd.csv", index_col=0)])
    rows = []
    for st in ("iti", "trial"):
        for a in AN:
            ses = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                ok = np.isfinite(C[:, [0, 3]]).all(1)
                if ok.sum() >= 40:
                    ses[x] = (d["epoch"], X[ok] / S.loc[x, names].to_numpy(float)[None, :], C[ok])
            pre = [x for x, v in ses.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            jun = [x for x, v in ses.items() if v[0] == "june"]
            for ax_, k in AXES.items():
                for tr in pre + jun:
                    _e, Xt, Ct = ses[tr]
                    sc = StandardScaler().fit(Xt)
                    m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), Ct[:, k])
                    for te, (ep, Xe, Ce) in ses.items():
                        if te == tr or (tr in jun and ep != "pre") or (tr in jun and te in jun):
                            continue   # June-trained decoders are tested on August pre only (no-stroke long gaps)
                        r = float(np.corrcoef(m.predict(sc.transform(Xe)), Ce[:, k])[0, 1])
                        rows.append(dict(set=st, animal=a, axis=ax_, train=tr, test=te,
                                         epoch="june" if tr in jun else ep,
                                         gap=abs(_day(te) - _day(tr)), r=r))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    june = "--june" in (argv if argv is not None else sys.argv[1:])
    sfx = "_june" if june else ""
    P = pairs(june)
    ref_eps = ["pre", "june"] if june else ["pre"]
    out = []
    for _key, g in P.groupby(["set", "animal", "axis"]):
        p8 = g[g.epoch == "pre"]
        b8, c8 = np.polyfit(np.log1p(p8.gap), p8.r, 1)          # August-only curve (the 3.36 one)
        pp = g[g.epoch.isin(ref_eps)]
        b, c = np.polyfit(np.log1p(pp.gap), pp.r, 1)
        g = g.assign(pred=c + b * np.log1p(g.gap), slope=b, max_pre_gap=pp.gap.max(),
                     pred_aug=c8 + b8 * np.log1p(g.gap))
        out.append(g.assign(resid=g.r - g.pred, resid_aug=g.r - g.pred_aug))
    P = pd.concat(out, ignore_index=True)
    P.to_csv(il.OUT / f"where_drift{sfx}_pairs.csv", index=False)
    pd.set_option("display.width", 220)
    if june:
        print("== EXTRAPOLATION CHECK: June pairs (no stroke, ~60 d) vs the AUGUST-only drift curve; residual < 0 = "
              "the 8-day curve UNDER-predicts long-gap drift")
        jj = P[P.epoch == "june"]
        print(jj.groupby(["set", "axis", "animal"]).agg(gap=("gap", "mean"), r=("r", "mean"),
                                                        resid_aug=("resid_aug", "mean")).round(3).to_string())
    print("== drift slope within pre (r per log(1 + day gap)) and the largest pre gap, per animal")
    print(P.groupby(["set", "axis", "animal"])[["slope", "max_pre_gap"]].first().round(3).to_string())
    ses = P.groupby(["set", "axis", "animal", "epoch", "test"]).agg(resid=("resid", "mean"), r=("r", "mean"),
                                                                    gap=("gap", "mean")).reset_index()
    print("\n== per animal (rule 8): mean residual r beyond the pre drift curve")
    print(ses.pivot_table(index=["set", "axis", "animal"], columns="epoch", values="resid")
          .reindex(columns=[*ref_eps, *EP]).round(3).to_string())
    tests = []
    for (st, ax_), g in ses.groupby(["set", "axis"]):
        prd = {a: h.resid.to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
        for ep in EP:
            post = {a: h.resid.to_numpy() for a, h in g[g.epoch == ep].groupby("animal")}
            ci = ak.boot_delta(post, prd, np.random.default_rng(0))
            am = g[g.epoch == ep].groupby("animal").resid.mean() - g[g.epoch == "pre"].groupby("animal").resid.mean()
            tests.append(dict(set=st, axis=ax_, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                              n_down=int((am < 0).sum()), n=int(am.notna().sum()), mean_gap=g[g.epoch == ep].gap.mean()))
    T = pd.DataFrame(tests)
    T.to_csv(il.OUT / f"where_drift{sfx}_tests.csv", index=False)
    print("\n== r beyond elapsed-time drift, change vs pre (paired nested bootstrap)")
    print(T.round(3).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(4, 4, figsize=(16, 13), squeeze=False, sharex=True)
    for i, (st, ax_) in enumerate([("iti", "lat"), ("trial", "lat"), ("iti", "sag"), ("trial", "sag")]):
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = ses[(ses.set == st) & (ses.axis == ax_) & (ses.animal == a)]
            for ep in [*ref_eps, *EP]:
                h = g[g.epoch == ep]
                ax.scatter(h.gap, h.r, s=14, color=colors.get(ep, "#8e44ad"), label=ep + (" (no stroke)" if ep == "june"
                                                                                       else ""),
                           alpha=0.8, marker="D" if ep == "june" else "o")
            pp = P[(P.set == st) & (P.axis == ax_) & (P.animal == a)]
            if len(pp):
                xx = np.linspace(0, pp.gap.max(), 100)
                ax.plot(xx, pp.slope.iloc[0] * np.log1p(xx) + (pp.pred - pp.slope * np.log1p(pp.gap)).iloc[0], "k--",
                        lw=1, label="drift fit (pre + June)" if june else "pre drift fit")
                ax.axvline(pp.max_pre_gap.iloc[0], color="0.7", lw=0.6, ls=":")
            ax.set_title(f"{a} {'ITI' if st == 'iti' else 'cued'} {'LATERAL' if ax_ == 'lat' else 'SAGITTAL'}",
                         fontsize=9)
            if j == 0:
                ax.set_ylabel("r, single-pre-session decoder\n(mean over training sessions)", fontsize=8)
            if i == 3:
                ax.set_xlabel("days between training and test session", fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False)
    t = T.set_index(["set", "axis", "epoch"])
    jl = (" JUNE no-stroke reference added (diamonds: August pre <-> June pairs, ~60 d), fit on pre + June pairs, so "
          "chronic gaps are within the reference range.\n") if june else ""
    fig.suptitle(jl + "NEURAL: is the frozen decoder's post-stroke r loss more than elapsed-time DRIFT? Decoders trained on ONE "
                 "pre session (z-scored Allen areas, lags 0..0.2 s), tested on every other session; dashed = log fit to "
                 "pre -> pre pairs\n(dotted = largest pre gap; beyond it the fit is extrapolated). Change in r beyond "
                 "the drift curve vs pre, chronic: ITI lateral "
                 f"{t.loc[('iti', 'lat', 'chronic'), 'point']:+.3f} [{t.loc[('iti', 'lat', 'chronic'), 'lo']:+.3f}, "
                 f"{t.loc[('iti', 'lat', 'chronic'), 'hi']:+.3f}]; cued lateral {t.loc[('trial', 'lat', 'chronic'), 'point']:+.3f}"
                 f" [{t.loc[('trial', 'lat', 'chronic'), 'lo']:+.3f}, {t.loc[('trial', 'lat', 'chronic'), 'hi']:+.3f}]. "
                 "PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if june else 0.94))
    fig.savefig(il.OUT / "figures" / f"iti_where_drift{sfx}.png", dpi=95)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
