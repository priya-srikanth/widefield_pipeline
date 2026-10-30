"""DEVIATION OF SUCCESSFUL LICKS ACROSS EPOCHS (Priya 10-30: "i like the compensation score and showing deviation on
successful chronic licks - can we plot that across all epochs?"). Figure-only follow-up of kinematic_normality.py
(pre-registered in docs/status/KINEMATIC_NORMALITY_PREREG_20261030.md); reads its saved trial table, refits nothing.

HITS only (first post-cue lick of engaged hit trials), PS92-95, August pre, epochs as in kinematic_normality.
  A  normality = fraction of hits inside the animal's pre 95 % contour at that spout (pre scored leave-one-session-out),
     per animal + mean of animals, all spouts.
  B  the same per spout (canonical order close_ipsi, close_mid, close_contra, far_ipsi, far_mid, far_contra) x epoch,
     mean of animals; per-cell boot_delta vs pre (* = CI excludes 0 AND >= 3/4 animals same sign).
  C  WHICH features deviate: per feature, signed z vs the animal's pre hits at the same spout (pre mean / SD), session
     mean, mean over spouts with the feature, then mean of animals; * as in B. ALL LATERAL SIGNS + = CONTRALESIONAL:
     angle as is; landing lateral error re-signed from "+ outward of the spout" (outward = contralesional at contra
     spouts, ipsilesional at ipsi spouts and at the centre, where + = mouse left); jaw lateral lr_wide_mm (+ ipsi)
     negated. log_lat + = slower.
  D  distribution of d2 / k on hits per epoch (pooled trials, animals equally weighted), pre 95 % line.

    python scripts/kn_hits_deviation.py [--plot-only]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kinematic_normality as kn  # noqa: E402

OUT = kn.OUT
FIG, FD = kn.FIG, kn.FD
EPS = ["pre", "acute", "subacute", "chronic"]
AN, POS = kn.AN, kn.POS
FEATS = ["angle", "sagittal", "reach_mm", "log_speed", "lat_err_mm", "log_miss", "log_lat", "jaw_lead", "lr_wide_mm"]
FLAB = {"angle": "lick angle at extension peak (+ contra)", "sagittal": "sagittal angle", "reach_mm": "reach", "log_speed": "peak speed (log)",
        "lat_err_mm": "landing lateral error (+ contra)", "log_miss": "closest approach (log)",
        "log_lat": "latency (log; + slower)", "jaw_lead": "jaw onset lead", "lr_wide_mm": "jaw lateral (+ contra)"}


def _boot(sess: pd.DataFrame, col: str, rng) -> dict:
    """{epoch: (point, lo, hi, n_same, per-animal deltas)} vs pre, session values, animals -> sessions."""
    from wfield_local import analysis_kit as ak
    pre = {a: g[col].dropna().to_list() for a, g in sess[sess.epoch == "pre"].groupby("animal")}
    pre = {a: v for a, v in pre.items() if v}
    out = {}
    for ep in EPS[1:]:
        post = {a: g[col].dropna().to_list() for a, g in sess[sess.epoch == ep].groupby("animal")}
        post = {a: v for a, v in post.items() if v}
        iv = ak.boot_delta(post, pre, rng)
        if iv is None:
            continue
        d = {a: np.mean(post[a]) - np.mean(pre[a]) for a in sorted(set(post) & set(pre))}
        ns = int(sum(np.sign(v) == np.sign(iv.point) for v in d.values()))
        out[ep] = (iv.point, iv.lo, iv.hi, ns, d)
    return out


def _sig(b) -> bool:
    return b is not None and (b[1] > 0 or b[2] < 0) and b[3] >= 3


def build() -> dict:
    T = pd.read_csv(OUT / "trials.csv.gz")
    T = T[T.is_hit.astype(bool) & T.epoch.isin(EPS) & T.animal.isin(AN)].copy()
    T["d2k"] = T["d2k"].astype(float)
    out_sign = np.where(T.pos.str.endswith("contra"), 1.0, -1.0)           # outward -> + contralesional
    T["lat_err_mm"] = T["lat_err_mm"] * out_sign
    T["lr_wide_mm"] = -T["lr_wide_mm"]                                     # + ipsilesional -> + contralesional
    rng = np.random.default_rng(0)
    # A: all spouts
    SA = T.groupby(["animal", "epoch", "label"]).inside.mean().rename("norm").reset_index()
    A = SA.groupby(["animal", "epoch"]).norm.mean().unstack().reindex(columns=EPS)
    bA = _boot(SA, "norm", rng)
    # B: per spout
    SB = T.groupby(["animal", "epoch", "label", "pos"]).inside.mean().rename("norm").reset_index()
    B, bB = {}, {}
    for p in POS:
        g = SB[SB.pos == p]
        B[p] = g.groupby(["animal", "epoch"]).norm.mean().unstack().reindex(columns=EPS).mean(0)
        bB[p] = _boot(g, "norm", rng)
    # C: feature z vs the animal's pre hits at the same spout
    Z = []
    for (a, p), g in T.groupby(["animal", "pos"]):
        pre = g[g.epoch == "pre"]
        for f in FEATS:
            if f not in kn.FEATS.get(p, []):
                continue
            mu, sd = pre[f].mean(), pre[f].std()
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = (g[f] - mu) / sd
            Z.append(pd.DataFrame({"animal": a, "pos": p, "epoch": g.epoch, "label": g.label, "feat": f, "z": z}))
    Z = pd.concat(Z, ignore_index=True).dropna()
    SC = Z.groupby(["feat", "animal", "epoch", "label", "pos"]).z.mean().reset_index()
    SC = SC.groupby(["feat", "animal", "epoch", "label"]).z.mean().reset_index()      # mean over spouts per session
    C, bC = {}, {}
    for f in FEATS:
        g = SC[SC.feat == f]
        C[f] = g.groupby(["animal", "epoch"]).z.mean().unstack().reindex(columns=EPS).mean(0)
        bC[f] = _boot(g, "z", rng)
    # C far contra only
    SCf = Z[Z.pos == "far_contra"].groupby(["feat", "animal", "epoch", "label"]).z.mean().reset_index()
    Cf, bCf = {}, {}
    for f in FEATS:
        g = SCf[SCf.feat == f]
        if g.empty:
            continue
        Cf[f] = g.groupby(["animal", "epoch"]).z.mean().unstack().reindex(columns=EPS).mean(0)
        bCf[f] = _boot(g, "z", rng)
    # D: d2/k distributions, animals equally weighted (sample equal n per animal x epoch)
    D = {}
    for ep in EPS:
        v = []
        for a in AN:
            x = T[(T.animal == a) & (T.epoch == ep)].d2k.dropna().to_numpy()
            if len(x):
                v.append(np.random.default_rng(1).choice(x, 400, replace=len(x) < 400))
        D[ep] = np.concatenate(v) if v else np.array([])
    k_typ = 8
    from scipy.stats import chi2
    thr = chi2.ppf(0.95, k_typ) / k_typ
    n = T.groupby(["animal", "epoch"]).size().unstack().reindex(columns=EPS)
    F = {"A": A, "bA": bA, "B": B, "bB": bB, "C": C, "bC": bC, "Cf": Cf, "bCf": bCf, "D": D, "thr": thr, "n": n}
    FD.mkdir(parents=True, exist_ok=True)
    with open(FD / "kn_hits_deviation.pkl", "wb") as fh:
        pickle.dump(F, fh)
    return F


def _star_band(ax, xs, flags):
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for x, f in zip(xs, flags):
        if f:
            ax.text(x, 0.97, "*", transform=tr, ha="center", va="top", fontsize=14, fontweight="bold")


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(17, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.15, 1.25])
    # A
    ax = fig.add_subplot(gs[0, 0])
    A = F["A"]
    for a in AN:
        if a in A.index:
            ax.plot(range(4), A.loc[a].to_numpy(float), "-", marker=kn.MK[a], color=kn.ACOL[a], lw=1.2, ms=6, label=a)
    ax.plot(range(4), A.mean(0).to_numpy(float), "-o", color="k", lw=2.6, ms=7, label="mean of animals")
    ax.set_xticks(range(4), EPS)
    ax.set_ylim(0.15, 1.12)
    ax.set_ylabel("fraction of successful licks inside\nthe pre-stroke 95 % range")
    ax.set_title("A. Movement normality on HITS, all spouts", fontsize=11, loc="left")
    _star_band(ax, range(1, 4), [_sig(F["bA"].get(e)) for e in EPS[1:]])
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    # B
    ax = fig.add_subplot(gs[0, 1])
    M = np.array([F["B"][p].to_numpy(float) for p in POS])
    im = ax.imshow(M, vmin=0.3, vmax=0.95, cmap="viridis", aspect="auto")
    for i, p in enumerate(POS):
        for j, e in enumerate(EPS):
            if np.isfinite(M[i, j]):
                s = "*" if j and _sig(F["bB"][p].get(e)) else ""
                ax.text(j, i, f"{M[i, j]:.2f}{s}", ha="center", va="center", fontsize=9,
                        color="w" if M[i, j] < 0.7 else "k")
    ax.set_xticks(range(4), EPS)
    ax.set_yticks(range(6), [p.replace("_", " ") for p in POS])
    ax.set_title("B. Normality on hits by spout (mean of animals)\n* = change vs pre, CI excl. 0 and >= 3/4 animals",
                 fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction inside pre 95 %")
    # C (all spouts) and C' (far contra)
    for col, (key, bkey, ttl) in enumerate([("C", "bC", "C. WHICH features deviate on hits, all spouts"),
                                            ("Cf", "bCf", "C'. same, far-contra hits")]):
        ax = fig.add_subplot(gs[col, 2])
        fs = [f for f in FEATS if f in F[key]]
        M = np.array([F[key][f].to_numpy(float) for f in fs])
        vm = 1.5
        im = ax.imshow(M, vmin=-vm, vmax=vm, cmap="RdBu_r", aspect="auto")
        for i, f in enumerate(fs):
            for j, e in enumerate(EPS):
                if np.isfinite(M[i, j]):
                    s = "*" if j and _sig(F[bkey][f].get(e)) else ""
                    ax.text(j, i, f"{M[i, j]:+.2f}{s}", ha="center", va="center", fontsize=8.5,
                            color="w" if abs(M[i, j]) > 0.9 else "k")
        ax.set_xticks(range(4), EPS)
        ax.set_yticks(range(len(fs)), [FLAB[f] for f in fs], fontsize=8.5)
        ax.set_title(ttl + "\nz vs own pre hits at the same spout;\nlateral + = contralesional; mean of animals",
                     fontsize=10, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.8, label="z (pre SD units)")
    # D
    ax = fig.add_subplot(gs[1, 0:2])
    cols = {"pre": "0.3", "acute": "#c0392b", "subacute": "#e67e22", "chronic": "#2471a3"}
    for e in EPS:
        x = np.sort(F["D"][e])
        if len(x):
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=cols[e], lw=2, label=e)
    ax.axvline(F["thr"], color="k", ls=":", lw=1)
    ax.text(F["thr"], 0.04, "  pre 95 % contour (k = 8)", fontsize=8.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 30)
    ax.set_xlabel("distance of each successful lick from the animal's pre-stroke licks at that spout (d2 / k, log)")
    ax.set_ylabel("cumulative fraction of hits")
    ax.set_title("D. Distribution of the deviation on hits (animals equally weighted)", fontsize=11, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.suptitle("Successful licks stay less normal after the stroke: compensation score on HITS across epochs "
                 "(PS92-95, August pre; first post-cue lick). n = 4, PROVISIONAL", fontsize=12)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "kn_fig5_hits_deviation.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "kn_fig5_hits_deviation.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        with open(FD / "kn_hits_deviation.pkl", "rb") as fh:
            F = pickle.load(fh)
    else:
        F = build()
    pd.set_option("display.width", 200)
    print("A per animal:\n", F["A"].round(3).to_string())
    print("n hits:\n", F["n"].to_string())
    for name, b in [("A all", F["bA"])] + [(f"B {p}", F["bB"][p]) for p in POS] + \
            [(f"C {f}", F["bC"][f]) for f in FEATS if f in F["bC"]] + \
            [(f"Cfc {f}", F["bCf"][f]) for f in FEATS if f in F["bCf"]]:
        print(name, {e: (round(v[0], 3), round(v[1], 3), round(v[2], 3), v[3]) for e, v in b.items()})
    plot(F)


if __name__ == "__main__":
    main()
