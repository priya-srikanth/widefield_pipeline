"""Is the frozen lick-position decoder's post-stroke r loss more than ELAPSED-TIME DRIFT? (handoff 00f to-do 3)

Representations drift with time even without a lesion, and every post session is further in time from the pre training
sessions than the pre sessions are from each other. Single-session decoders make the gap explicit:
  pre -> pre    train a ridge decoder on ONE pre session, test on another pre session: r vs |day gap| (the drift curve)
  pre -> post   train on ONE pre session, test on a post session: r at that gap
Per animal x lick set x axis (lateral, sagittal): fit r = a + b * log(1 + gap) on the pre -> pre pairs; for every
pre -> post pair, residual = r - drift prediction at its gap (extrapolated beyond the pre range for chronic gaps, which
the log fit makes conservative-to-neutral; reported separately). Per post session: mean residual over its training
sessions; change per epoch = nested bootstrap of those residuals (pre->pre residuals as the reference).
Features = Allen areas with the ALM / tjM1 split (every area decoder since 10-27 late; iti_where_decoder.area_basis),
lags 0..0.2 s (as 3.15), z-scored within session (frame-level SDs; June: june_ref/ SD + split tables, iti_june_ref.py).
--june (Priya 10-27): add the June 6-8 pre sessions (`iti_june_ref.py`, a separate folder; never in the August
baseline) as a NO-STROKE LONG-GAP reference: pairs August pre <-> June (both directions; ~60 days, about the chronic
gap). The drift curve is then fit on pre -> pre AND June pairs, so chronic gaps are interpolated, not extrapolated; the
June residual against the August-only curve says whether that extrapolation was right. Caveat: June is earlier in
training, so June <-> August drift includes learning (makes the test conservative).
Output <E>/iti_licks/where_drift[_june]_{pairs,tests}.csv, figures/lick_where_drift[_june].png.
Figure data <E>/iti_licks/figdata/lick_where_drift[_june].pkl; --plot-only redraws from it alone (no decoder fits,
no bootstrap).
Usage: python scripts/iti_where_drift.py [--june] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
AXES = {"lat": 0, "sag": 3}
MIN_LICKS = 40      # a session enters the single-session decoders with >= this many tracked licks of the set


def _day(label: str) -> int:
    md = label.split("_")[-1]
    return date(2026, int(md[:2]), int(md[2:])).toordinal()


def pairs(june: bool = False) -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    # Allen areas in the basis (split by default, Priya 10-27 late) + frame SDs: the shared loader
    D, names, S = wd.area_basis()
    if june:
        from iti_june_ref import JUNE
        DJ, _nj, SJ = wd.area_basis(JUNE / il.SESS_NAME)     # June: its own SD / split tables (june_ref/)
        for d in DJ.values():
            d["epoch"] = "june"
        D = {**D, **DJ}
        S = pd.concat([S, SJ])
    rows = []
    for st in ("iti", "trial"):
        for a in AN:
            ses = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                ok = np.isfinite(C[:, [0, 3]]).all(1)
                if ok.sum() >= MIN_LICKS:
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


def compute(june: bool) -> dict:
    """Decoder pairs, drift fits, residual tests (CSVs + per-animal tables); returns the figure data."""
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
    fits = {}
    for st, ax_ in [("iti", "lat"), ("trial", "lat"), ("iti", "sag"), ("trial", "sag")]:
        for a in AN:
            pp = P[(P.set == st) & (P.axis == ax_) & (P.animal == a)]
            if len(pp):
                xx = np.linspace(0, pp.gap.max(), 100)
                fits[(st, ax_, a)] = (xx, pp.slope.iloc[0] * np.log1p(xx) + (pp.pred - pp.slope * np.log1p(pp.gap)).iloc[0],
                                      pp.max_pre_gap.iloc[0])
    return dict(june=june, ref_eps=ref_eps, ses=ses, T=T, fits=fits)


def plot(F: dict) -> None:
    june, ref_eps, ses, T, fits = F["june"], F["ref_eps"], F["ses"], F["T"], F["fits"]
    sfx = "_june" if june else ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(4, 5, figsize=(20, 13), squeeze=False, sharex=True)
    marks = dict(zip(AN, "osD^"))
    for i, (st, ax_) in enumerate([("iti", "lat"), ("trial", "lat"), ("iti", "sag"), ("trial", "sag")]):
        # POOLED, PRE-NORMALISED (Priya 10-28): each animal's r / its mean pre (pre -> pre pair) r, drift fit scaled the
        # same; black dashed = mean of the animals' scaled fits on a common gap axis
        axp = axs[i, 4]
        xg = np.linspace(0, ses.gap.max(), 200)
        curves = []
        for a in AN:
            g = ses[(ses.set == st) & (ses.axis == ax_) & (ses.animal == a)]
            pre_m = g[g.epoch == "pre"].r.mean()
            if not np.isfinite(pre_m) or not len(g):
                continue
            for ep in [*ref_eps, *EP]:
                h = g[g.epoch == ep]
                axp.scatter(h.gap, h.r / pre_m, s=11, color=colors.get(ep, "#8e44ad"), marker=marks[a], alpha=0.7, lw=0)
            if (st, ax_, a) in fits:
                xx, yy, _mx = fits[(st, ax_, a)]
                b_ = (yy[-1] - yy[0]) / np.log1p(xx[-1])            # the fit is c + b log(1 + gap)
                curves.append((yy[0] + b_ * np.log1p(xg)) / pre_m)
        if curves:
            axp.plot(xg, np.mean(curves, axis=0), "k--", lw=1.4)
        axp.axhline(1, color="0.8", lw=0.6)
        tt = T.set_index(["set", "axis", "epoch"])
        axp.set_title(f"POOLED {'ITI' if st == 'iti' else 'cued'} {'LATERAL' if ax_ == 'lat' else 'SAGITTAL'}, r / pre\n"
                      "beyond drift vs pre: " + ", ".join(
                          f"{e[:3]} {tt.loc[(st, ax_, e), 'point']:+.2f}" + ("*" if tt.loc[(st, ax_, e), 'lo'] > 0 or
                                                                            tt.loc[(st, ax_, e), 'hi'] < 0 else "")
                          for e in EP if (st, ax_, e) in tt.index), fontsize=8.5)
        if i == 3:
            axp.set_xlabel("days between training and test session", fontsize=8)
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = ses[(ses.set == st) & (ses.axis == ax_) & (ses.animal == a)]
            for ep in [*ref_eps, *EP]:
                h = g[g.epoch == ep]
                ax.scatter(h.gap, h.r, s=14, color=colors.get(ep, "#8e44ad"), label=ep + (" (no stroke)" if ep == "june"
                                                                                       else ""),
                           alpha=0.8, marker="D" if ep == "june" else "o")
            if (st, ax_, a) in fits:
                xx, yy, mx = fits[(st, ax_, a)]
                ax.plot(xx, yy, "k--", lw=1, label="drift fit (pre + June)" if june else "pre drift fit")
                ax.axvline(mx, color="0.7", lw=0.6, ls=":")
            ax.set_title(f"{a} {'ITI' if st == 'iti' else 'cued'} {'LATERAL' if ax_ == 'lat' else 'SAGITTAL'}",
                         fontsize=9)
            if j == 0:
                ax.set_ylabel("r, single-pre-session decoder\n(mean over training sessions)", fontsize=8)
            if i == 3:
                ax.set_xlabel("days between training and test session", fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False)
    from matplotlib.lines import Line2D
    axs[0, 4].legend([Line2D([], [], ls="", marker=marks[a], color="0.4") for a in AN] +
                     [Line2D([], [], color="k", ls="--")], [*AN, "mean scaled drift fit"], fontsize=7, frameon=False,
                     loc="lower left")
    t = T.set_index(["set", "axis", "epoch"])
    jl = (" JUNE no-stroke reference added (diamonds: August pre <-> June pairs, ~60 d), fit on pre + June pairs, so "
          "chronic gaps are within the reference range.\n") if june else ""
    fig.suptitle(jl + "NEURAL: is the frozen decoder's post-stroke r loss more than elapsed-time DRIFT? Decoders trained on ONE "
                 f"pre session (z-scored {il.basis_label()}, lags 0..0.2 s), tested on every other session; dashed = log fit to "
                 "pre -> pre pairs\n(dotted = largest pre gap; beyond it the fit is extrapolated). Change in r beyond "
                 "the drift curve vs pre, chronic: ITI lateral "
                 f"{t.loc[('iti', 'lat', 'chronic'), 'point']:+.3f} [{t.loc[('iti', 'lat', 'chronic'), 'lo']:+.3f}, "
                 f"{t.loc[('iti', 'lat', 'chronic'), 'hi']:+.3f}]; cued lateral {t.loc[('trial', 'lat', 'chronic'), 'point']:+.3f}"
                 f" [{t.loc[('trial', 'lat', 'chronic'), 'lo']:+.3f}, {t.loc[('trial', 'lat', 'chronic'), 'hi']:+.3f}]. "
                 "PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if june else 0.94))
    save_png_svg(fig, il.OUT / "figures" / f"lick_where_drift{sfx}.png", dpi=95)
    plt.close(fig)


def trial_set_sentence(june: bool = False) -> str:
    """ITI + cued licks of every session with >= MIN_LICKS tracked licks of the set (Priya 11-09); --june adds the June
    6-8 pre sessions (iti_june_ref.py) as a no-stroke reference (decoders trained on them are tested on August pre only)."""
    xs = ["+ June 6-8 pre sessions as a no-stroke reference (never trained into the August baseline)"] if june else []
    return wd.both_caption("", min_n=MIN_LICKS, iti_extra=xs, cued_extra=xs)


def main(argv=None) -> int:
    from wfield_local import trial_set_text as tst
    il.split_only("iti_where_drift")   # outputs not basis-tagged: split basis only (10-27 late)
    argv = argv if argv is not None else sys.argv[1:]
    june = "--june" in argv
    tst.declare(text=trial_set_sentence(june))
    stem = f"lick_where_drift{'_june' if june else ''}"
    if "--plot-only" in argv or plot_only():   # redraw from figdata only: no decoder fits, no bootstrap
        plot(load_fd(stem))
        return 0
    F = compute(june)
    save_fd(stem, F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
