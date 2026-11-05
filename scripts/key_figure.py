"""KEY FIGURE of the goal page (Priya 10-28: "yes to your A and B; and use the newer behavior recovery metric").

One figure, four panels, every value read from outputs other scripts already wrote (no new statistics except the
far-contra code-on-hits change, which uses the same paired nested bootstrap as everything else):
  A  BEHAVIOUR recovers: far-contra FIRST-cued-lick aim offset (deg vs the animal's pre landing, - = ipsilesional;
     `iti_recovery_measure`, the behaviour axis everywhere) per post session over days, per animal (thin) + mean of
     animals per epoch (bold, at the epoch's median day).
  B  CUED licks, fraction of pre: the FROZEN pre-stroke far-contra spout code on far-contra HITS (`recovery_timing.series`
     'neu', the previous key figure's quantity), the FROZEN lateral lick decoder r / pre, and the same decoder REFIT per
     epoch (r / matched-n pre) -- `lick_refit_decoders` (refit_decoders_tests.csv).
  C  ITI licks: frozen vs epoch-refit lateral lick decoder, as B.
  D  ITI like-for-like read-out offset (mm, + = contralesional) against pre licks that landed at the same place: any spout
     (12.1, matched_gauss_tests.csv) and the same previous-trial target (12.15, sametarget_tests.csv); per animal + mean
     of animals with 95 % paired nested bootstrap CI.
Reading: behaviour returns; the frozen pre-stroke read-out stays low (ITI) while a decoder refit after the stroke reads
position again (the code is there, read differently); ITI licks are read ~0.25 mm contralesional of where they land.
Outputs <E>/iti_licks/figures/key_figure.png; figdata/key_figure.pkl (--plot-only redraws from it).
Usage: python scripts/key_figure.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]


def _frac(T: pd.DataFrame, st: str, kind: str) -> pd.DataFrame:
    """Refit-decoder tests -> fraction of pre per epoch (pre = 1), CI scaled the same way, per-animal fractions."""
    t = T[(T.set == st) & (T.target == "lateral") & (T.kind == kind)]
    rows = [dict(epoch="pre", y=1.0, lo=1.0, hi=1.0, n_dir=np.nan, n=np.nan, **{a: 1.0 for a in AN})]
    for _, r in t.iterrows():
        rows.append(dict(epoch=r.epoch, y=1 + r.delta / r.pre_r, lo=1 + r.lo / r.pre_r, hi=1 + r.hi / r.pre_r,
                         n_dir=r.n_down, n=r.n, **{a: 1 + r[f"animal_{a}"] / r.pre_r for a in AN}))
    return pd.DataFrame(rows).set_index("epoch").reindex(EP)


def compute() -> dict:
    import iti_recovery_measure as rm
    from recovery_timing import series
    Aim = pd.read_csv(rm.sessions_csv("first"))[["animal", "label", "epoch", "day", "far_contra_signed"]]
    Aim = Aim.rename(columns={"far_contra_signed": "aim"})
    # previous key-figure quantity: frozen far-contra code on far-contra HITS (1 = pre by construction)
    B = series("lick")
    ref = {a: B[(B.animal == a) & (B.epoch == "pre")].neu.dropna().to_numpy() for a in AN}
    code = [dict(epoch="pre", y=1.0, lo=1.0, hi=1.0, n_dir=np.nan, n=np.nan, **{a: 1.0 for a in AN})]
    for ep in EP[1:]:
        post = {a: B[(B.animal == a) & (B.epoch == ep)].neu.dropna().to_numpy() for a in AN}
        com = [a for a in AN if len(post[a]) and len(ref[a])]
        iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
        am = {a: float(post[a].mean() - ref[a].mean()) for a in com}
        code.append(dict(epoch=ep, y=1 + iv.point, lo=1 + iv.lo, hi=1 + iv.hi,
                         n_dir=sum(v < 0 for v in am.values()), n=len(com),
                         **{a: (1 + am[a]) if a in am else np.nan for a in AN}))
    code = pd.DataFrame(code).set_index("epoch")
    R = pd.read_csv(il.OUT / "refit_decoders_tests.csv")
    dec = {(st, k): _frac(R, st, k) for st in ("trial", "iti") for k in ("frozen", "epoch refit")}
    M = pd.read_csv(il.OUT / "matched_gauss_tests.csv")
    M = M[(M.scope == "all spouts") & (M.cls == "all") & (M.set == "iti") & (M.measure == "resid")].set_index("epoch")
    ST = pd.read_csv(il.OUT / "sametarget_tests.csv")
    ST = ST[(ST.scope == "all spouts") & (ST.cls == "all") & (ST.set == "iti") & (ST.measure == "tt")].set_index("epoch")
    off = {"any spout (12.1)": M, "same previous-trial target (12.15)": ST}
    print("== per animal (rule 8): far-contra code on hits, fraction of pre\n" + code.round(2).to_string())
    for k, v in dec.items():
        print(f"== {k}: lateral decoder r / pre\n" + v.round(2).to_string())
    for k, v in off.items():
        print(f"== ITI offset, {k}\n" + v[["point", "lo", "hi", "n_dir", "n", *[f"animal_{a}" for a in AN]]].round(3)
              .to_string())
    return dict(Aim=Aim, code=code, dec=dec, off=off)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    Aim, code, dec, off = F["Aim"], F["code"], F["dec"], F["off"]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    # A behaviour
    ax = axs[0]
    for a in AN:
        g = Aim[Aim.animal == a].sort_values("day")
        ax.plot(g.day, g.aim, color="0.75", lw=0.8, marker=ANIMAL_MARKER.get(a, "o"), ms=3, zorder=1)
    E_ = Aim.groupby(["animal", "epoch"]).agg(day=("day", "median"), v=("aim", "mean")).reset_index()
    Mn = E_.groupby("epoch").agg(day=("day", "median"), v=("v", "mean")).reindex(EP)
    ax.plot(Mn.day, Mn.v, color="k", lw=2.2, zorder=3)
    for ep in EP:
        ax.scatter(Mn.loc[ep, "day"], Mn.loc[ep, "v"], s=70, color=col[ep], zorder=4, edgecolor="k")
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("day from stroke")
    ax.set_ylabel("far-contra first-lick aim offset\n(deg vs pre landing; - = ipsilesional)")
    rem = (Mn.loc["chronic", "v"] - Mn.loc["pre", "v"]) / (Mn.loc["acute", "v"] - Mn.loc["pre", "v"])
    ax.set_title(f"A  BEHAVIOUR mostly recovers: {rem:.0%} of the acute aim\ndeficit remains chronically (grey = "
                 "animals; bold = mean)", fontsize=10, loc="left")
    # B, C decoders
    x = np.arange(len(EP))
    for ax, st, ttl in ((axs[1], "trial", "B  CUED licks: frozen read-out vs refit"),
                        (axs[2], "iti", "C  ITI licks: frozen read-out vs refit")):
        lines = [(dec[(st, "frozen")], "frozen pre-stroke lateral decoder", "#444444", "-", "o"),
                 (dec[(st, "epoch refit")], "decoder REFIT per epoch (vs matched-n pre)", "#2b8cbe", "-", "s")]
        if st == "trial":
            lines.insert(0, (code, "frozen far-contra spout code, far-contra hits", "#b2182b", "--", "D"))
        for k, (D, lab, c, ls, mk) in enumerate(lines):
            xx = x + (k - (len(lines) - 1) / 2) * 0.07
            ax.plot(xx, D.y, color=c, ls=ls, marker=mk, lw=2, ms=6, label=lab, zorder=3)
            ax.vlines(xx, D.lo, D.hi, color=c, lw=1.4, zorder=2)
            for a in AN:
                ax.scatter(xx[1:], D[a].to_numpy()[1:], s=9, color=c, alpha=0.35, marker=ANIMAL_MARKER.get(a, "o"),
                           lw=0, zorder=1)
        ax.axhline(1, color="0.5", lw=0.7, ls="--")
        ax.set_xticks(x, EP)
        ax.set_ylabel("fraction of pre-stroke level")
        ax.set_title(ttl + "\n(mean of animals, 95% nested CI; small = animals)", fontsize=10, loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    # D ITI offset
    ax = axs[3]
    for k, (lab, T) in enumerate(off.items()):
        c = ["#1a9641", "#7b3294"][k]
        xx = x[1:] + (k - 0.5) * 0.18
        T = T.reindex(EP[1:])
        ax.errorbar(xx, T.point, yerr=[T.point - T.lo, T.hi - T.point], fmt="o-", color=c, ms=7, lw=2, capsize=3,
                    label=lab, zorder=3)
        for a in AN:
            ax.scatter(xx, T[f"animal_{a}"], s=12, color=c, alpha=0.4, marker=ANIMAL_MARKER.get(a, "o"), lw=0)
        for xi, (_, r) in zip(xx, T.iterrows()):
            ax.text(xi + (0.09 if k else -0.09), r.point, f"{r.point:+.2f}\n{int(r.n_dir)}/{int(r.n)}",
                    ha="left" if k else "right", va="center", fontsize=7, color=c)
    ax.scatter([0], [0], s=60, color="k", zorder=3)
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.set_xticks(x, EP)
    ax.set_ylabel("ITI read-out minus pre licks that landed\nat the same place (mm, + = contralesional)")
    ax.set_title("D  ITI licks are read contralesional of where\nthey land (like-for-like, Gaussian reference)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for a_ in axs:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.suptitle("KEY FIGURE: behaviour mostly recovers; the FROZEN pre-stroke read-out stays down for the far-contra spout "
                 "code (~70%) and for ITI licks (~75%),\nwhile a decoder REFIT after the stroke reads lick position at ~95-100% "
                 "of pre (the code is there, read differently); ITI licks are read ~0.25 mm contralesional of where they "
                 "land,\neven against pre licks of the same previous-trial target. Decoders: z-scored Allen areas with the ALM / tjM1 split, lags 0..0.2 s; "
                 "pre = August pre sessions (leave-one-session-out). PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(il.OUT_W / "figures" / "key_figure.png", dpi=105)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("key_figure"))
        return 0
    F = compute()
    save_fd("key_figure", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
