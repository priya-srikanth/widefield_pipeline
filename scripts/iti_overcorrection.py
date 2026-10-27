"""Is the chronic ipsilesional read-out of far-contra cued licks a downstream OVER-CORRECTION? (Priya 10-27: "run the
overcorrection analyses you proposed").

Chronically the frozen decoder reads far-contra cued licks ipsilesional of where they land (beyond its pre pull,
FINDINGS 4d0b) while the tongue lands near its pre landing. A downstream contralesional correction predicts:
  (1) SPOUT SPECIFICITY: a shift of the cortex -> landing transform -- which spouts carry the ipsilesional read-out?
      Per animal x spout x epoch: beyond-pull offset = (decoded - pre decoded mean at the spout) - b x (landing - pre
      landing mean at the spout), b = the animal's pre pull slope (cued, lateral).
  (2) INTACT LICK-TO-LICK COUPLING with a shifted offset: within a spout and session, the slope of decoded on landing
      (cortex still tracks each lick's deviation) back at its pre value, while the offset stays shifted. A degraded /
      remapped code predicts a lower slope instead.
Per session x spout (>= MIN licks): offset (as 1) and within-session slope; change vs pre per epoch with the paired nested
bootstrap (animals > sessions); per-animal table first (rule 8). Decoder = the KEY-row where-decoder on z-scored Allen
areas (zscore_check/where_preds_z.csv), cued licks.
Outputs <E>/iti_licks/overcorrection_{sessions,tests}.csv, figures/cued_overcorrection[_angle].png.
--angle: the same in lick ANGLE (atan2(lateral, sagittal), degrees), the primary behaviour measure (Priya 10-27: "keep
angle and lateral analyses, but make primary the one that is most consistent").
Usage: python scripts/iti_overcorrection.py [--angle]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
MIN = 15


def sessions(angle: bool = False) -> pd.DataFrame:
    p = pd.read_csv(il.OUT / "zscore_check" / "where_preds_z.csv")
    if angle:   # lick ANGLE atan2(lateral, sagittal), degrees; decoded angle from the two decoded axes
        p = p.dropna(subset=["act_sag", "dec_sag"]).copy()
        p["act_lat"] = np.degrees(np.arctan2(p.act_lat, p.act_sag))
        p["dec_lat"] = np.degrees(np.arctan2(p.dec_lat, p.dec_sag))
    p = p[(p.set == "trial") & p.pos.isin(POS)].dropna(subset=["act_lat", "dec_lat"])
    rows = []
    for a, g in p.groupby("animal"):
        pre = g[g.epoch == "pre"]
        b, _c = np.polyfit(pre.act_lat, pre.dec_lat, 1)
        pm = pre.groupby("pos")[["act_lat", "dec_lat"]].mean()
        for (lab, ep, sp), h in g.groupby(["label", "epoch", "pos"]):
            if len(h) < MIN or sp not in pm.index:
                continue
            da = h.act_lat.mean() - pm.loc[sp, "act_lat"]
            dd = h.dec_lat.mean() - pm.loc[sp, "dec_lat"]
            sl = np.polyfit(h.act_lat, h.dec_lat, 1)[0] if h.act_lat.std() > 1e-6 else np.nan
            rows.append(dict(animal=a, label=lab, epoch=ep, spout=sp, n=len(h), b_pre=b, landing=da, decoded=dd,
                             beyond=dd - b * da, slope=sl))
    return pd.DataFrame(rows)


def tests(S: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sp in POS:
        g = S[S.spout == sp]
        for val in ("beyond", "slope", "landing"):
            prd = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            for ep in EPS[1:]:
                post = {a: h[val].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                post = {a: v for a, v in post.items() if len(v)}
                if not post:
                    continue
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = (g[g.epoch == ep].groupby("animal")[val].mean() - g[g.epoch == "pre"].groupby("animal")[val].mean()
                      ).dropna()
                out.append(dict(spout=sp, measure=val, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                n_dir=int((am * np.sign(ci.point) > 0).sum()), n=len(am)))
    return pd.DataFrame(out)


def plot(S: pd.DataFrame, T: pd.DataFrame, angle: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    u = "deg" if angle else "mm"
    meas = [("landing", f"BEHAVIOUR: {'angle' if angle else 'landing'} vs pre at the spout ({u})"),
            ("beyond", f"NEURAL: decoded offset beyond the pre pull ({u})"),
            ("slope", "NEURAL: within-session slope, decoded on landing")]
    fig, axs = plt.subplots(len(meas), len(POS), figsize=(3.0 * len(POS), 2.9 * len(meas) + 1.2), squeeze=False,
                            sharey="row")
    for i, (val, lab) in enumerate(meas):
        for j, sp in enumerate(POS):
            ax = axs[i, j]
            g = S[S.spout == sp]
            am = g.groupby(["animal", "epoch"])[val].mean()
            for a in AN:
                ax.plot(range(4), [am.get((a, e), np.nan) for e in EPS], "-", color="0.6", lw=0.8)
                for x, e in enumerate(EPS):
                    ax.plot(x, am.get((a, e), np.nan), "o", color=EPOCH_LINE[e], ms=3.5)
            mu = [am.xs(e, level="epoch").mean() if e in am.index.get_level_values("epoch") else np.nan for e in EPS]
            ax.plot(range(4), mu, "k-", lw=2)
            for x, e in enumerate(EPS[1:], start=1):
                t = T[(T.spout == sp) & (T.measure == val) & (T.epoch == e)]
                if len(t):
                    t = t.iloc[0]
                    sig = t.lo > 0 or t.hi < 0
                    ax.errorbar(x + 0.15, mu[0] + t.point, yerr=[[t.point - t.lo], [t.hi - t.point]], fmt="D",
                                color="k", ms=4, capsize=2)
                    ax.text(x + 0.15, mu[0] + t.hi, f"{t.point:+.2f}{'*' if sig else ''}\n{t.n_dir}/{t.n}",
                            ha="center", va="bottom", fontsize=6.5, fontweight="bold" if sig else "normal")
            ax.axhline(mu[0] if val == "slope" else 0, color="0.7", lw=0.6, ls=":")
            ax.set_xticks(range(4), [e[:3] for e in EPS], fontsize=7.5)
            if i == 0:
                ax.set_title(sp.replace("_", " "), fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.5)
    fig.suptitle("Is the chronic ipsilesional read-out of far-contra cued licks a downstream over-correction? Per spout "
                 "(canonical order), cued licks, frozen where-decoder (z-scored Allen areas).\nRow 1 = where the tongue "
                 "lands vs its pre landing at that spout; row 2 = decoded offset beyond the decoder's pre pull (- = read "
                 "ipsilesional of the landing); row 3 = within-session slope of decoded on landing (lick-to-lick coupling)."
                 "\nGrey = animals, black = mean of animals, diamond = pre + change with 95% paired nested bootstrap CI "
                 "(* = excludes 0, bold), k/n = animals changing that way. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_size_inches()[1]))
    fig.savefig(il.OUT / "figures" / f"cued_overcorrection{'_angle' if angle else ''}.png", dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    angle = "--angle" in (argv if argv is not None else sys.argv[1:])
    sfx = "_angle" if angle else ""
    S = sessions(angle)
    S.to_csv(il.OUT / f"overcorrection{sfx}_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): beyond-pull offset, per spout x epoch (mm; - = read ipsilesional of the landing)")
    print(S.pivot_table(index=["spout", "animal"], columns="epoch", values="beyond").reindex(columns=EPS).round(2)
          .to_string())
    print("\n== per animal: within-session slope of decoded on landing")
    print(S.pivot_table(index=["spout", "animal"], columns="epoch", values="slope").reindex(columns=EPS).round(2)
          .to_string())
    T = tests(S)
    T.to_csv(il.OUT / f"overcorrection{sfx}_tests.csv", index=False)
    print("\n== change vs pre (paired nested bootstrap)")
    print(T.round(3).to_string(index=False))
    plot(S, T, angle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
