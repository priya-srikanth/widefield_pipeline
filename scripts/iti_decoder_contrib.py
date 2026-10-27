"""Which areas move the frozen decoder's lick position after the stroke? (Priya 10-26: "it's counter intuitive that
cued licks are read as being closer ... when there is MORE tjM1 and ALM activity").

The frozen where-decoder (`iti_where_decoder.py`, Allen areas, lags 0..0.2 s from the lick peak, ridge on standardised
features, fit on ALL of the animal's August pre licks) is linear: decoded = c + sum_i w_i (x_i - mu_i) / sd_i. So the
change of the mean decoded position from pre to an epoch splits exactly into per-area contributions
w_i / sd_i * (mean_epoch x_i - mean_pre x_i). Per animal, lick set (ITI / in-trial) and axis (lateral, sagittal);
mean of animals reported, per-animal table first. Pre is scored with the all-pre fit here (an exact split needs one
model), so the totals differ slightly from the leave-one-session-out numbers on the page.
Also writes the decoder weights per area (w / sd, mm per dF/F unit) and the mean feature change, so a sign can be read
as "this area going up pushes the decoded lick further / more contralesional".
Outputs <E>/iti_licks/decoder_contrib.csv, figures/lick_decoder_contrib.png. Usage: python scripts/iti_decoder_contrib.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
AXES = {"lat": 0, "sag": 3}                  # columns of iti_where_decoder.coords


def feature_names() -> list[str]:
    x = sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))[0]
    z = il._load(x)
    return [str(c) for c in z["cols"] if c != "cortex" and not str(c).startswith(("ALM", "tjM1"))]


def contributions() -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    names = feature_names()
    D = wd.load("areas")
    rows = []
    for st in ("iti", "trial"):
        for a in AN:
            S = {x: (d["epoch"], *d[st]) for x, d in D.items() if d["animal"] == a}
            S = {x: (ep, X[ok], C[ok]) for x, (ep, X, C) in S.items() for ok in [np.isfinite(C[:, [0, 3]]).all(1)]
                 if ok.sum() >= 20}
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            Xp = np.vstack([S[q][1] for q in pre])
            Yp = np.vstack([S[q][2] for q in pre])
            sc = StandardScaler().fit(Xp)
            # session means first, then epoch means (sessions weighted equally, like the page's per-session numbers)
            sm = pd.DataFrame([dict(label=x, epoch=ep, **dict(zip(names, X.mean(0)))) for x, (ep, X, _C) in S.items()])
            em = sm.groupby("epoch")[names].mean()
            for ax_, k in AXES.items():
                m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xp), Yp[:, k])
                w = m.coef_ / sc.scale_
                for ep in EP:
                    if ep not in em.index:
                        continue
                    dx = (em.loc[ep] - em.loc["pre"]).to_numpy()
                    for n, wi, di in zip(names, w, dx):
                        rows.append(dict(set=st, animal=a, axis=ax_, epoch=ep, area=n, weight=wi, dx=di,
                                         contrib=wi * di))
    return pd.DataFrame(rows)


def plot(T: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("trial", "sag", "cued in-trial licks, SAGITTAL (- = read closer)"),
              ("iti", "lat", "ITI licks, LATERAL (+ = read more contralesional)"),
              ("trial", "lat", "cued in-trial licks, LATERAL"), ("iti", "sag", "ITI licks, SAGITTAL")]
    fig, axs = plt.subplots(len(panels), len(EP), figsize=(5.2 * len(EP), 4.2 * len(panels)), squeeze=False)
    for i, (st, ax_, ttl) in enumerate(panels):
        G = T[(T.set == st) & (T.axis == ax_)]
        order = (G[G.epoch == "chronic"].groupby("area").contrib.mean().abs().sort_values(ascending=False)
                 .index[:14].tolist())
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            g = G[(G.epoch == ep) & G.area.isin(order)]
            per = g.pivot_table(index="area", columns="animal", values="contrib").reindex(order)
            mu = per.mean(1)
            yy = np.arange(len(order))[::-1]
            ax.barh(yy, mu.to_numpy(), color=["#c0392b" if v > 0 else "#2166ac" for v in mu], alpha=0.75)
            for c in per.columns:
                ax.scatter(per[c].to_numpy(), yy, s=9, color="k", alpha=0.5, zorder=3)
            ax.axvline(0, color="0.5", lw=0.7)
            ax.set_yticks(yy)
            ax.set_yticklabels(order, fontsize=7.5)
            tot = G[G.epoch == ep].groupby("animal").contrib.sum()
            ax.set_title(f"{ttl}\n{ep}: total shift {tot.mean():+.2f} mm (animals {', '.join(f'{v:+.2f}' for v in tot)})",
                         fontsize=8.5)
            ax.set_xlabel("contribution to the decoded shift (mm)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("NEURAL: which areas move the FROZEN pre-stroke decoder's lick position (Allen areas, lags 0..0.2 s). "
                 "Contribution = the area's pre-stroke decoder weight x its change in mean activity (epoch - pre);\n"
                 "bars = mean of animals, dots = animals; the 14 areas with the largest chronic contribution per row; "
                 "contributions sum to the shift of the mean decoded position.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(il.OUT / "figures" / "lick_decoder_contrib.png", dpi=95)
    plt.close(fig)


def main() -> int:
    T = contributions()
    T.to_csv(il.OUT / "decoder_contrib.csv", index=False)
    pd.set_option("display.width", 220)
    tot = T.groupby(["set", "axis", "animal", "epoch"]).contrib.sum().unstack("epoch")[EP]
    print("== per animal (rule 8): total shift of the mean decoded position (mm)\n" + tot.round(2).to_string())
    for st, ax_ in (("trial", "sag"), ("iti", "lat")):
        G = T[(T.set == st) & (T.axis == ax_) & (T.epoch == "chronic")]
        s = G.groupby("area").agg(contrib=("contrib", "mean"), weight=("weight", "mean"), dx=("dx", "mean"),
                                  n_same=("contrib", lambda v: int((np.sign(v) == np.sign(v.mean())).sum())))
        print(f"\n== {st} {ax_}, chronic: top areas (mean of animals; weight in mm per dF/F unit)")
        print(s.reindex(s.contrib.abs().sort_values(ascending=False).index).head(12).round(3).to_string())
    plot(T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
