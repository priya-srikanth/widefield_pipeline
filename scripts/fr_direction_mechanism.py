"""WHY do FaceRhythm's direction factors explain less orofacial cortex after the stroke? (Priya 10-25: "can we test / delve
into this ... it could be critical to interpretation".)

Fig 16n-16o: chronically the side / distance FR factors add ~0 to the per-session encoder in SSp-m (pre +0.011-0.016) and
take less of the position code, while position | directionless FR rises contralesionally. Three explanations:
  (a) BEHAVIOUR  the face's side-specific movement itself shrinks / stops differing by spout (e.g. lateral jaw / tongue
                 steering lost; jaw FAR_DIR collapses acutely, OROFACIAL_FINDINGS);
  (c) MEASUREMENT the frozen pre-stroke FR vocabulary (master_pre_v1) no longer captures the post-stroke lateral movement;
  (b) NEURAL     the face moves as before, but orofacial cortex stops covarying with it.
Per session (O2 bundle inputs: the encoder's own imaging, behaviour and raw FR factors), engaged trials, cue window
0 .. 1.5 s (FR / cortex = window mean; cortex minus the 0.5 s pre-cue mean):
  side index   fr_lick_left - fr_lick_right;  distance index  fr_lick_far_nose - fr_lick_fast_close;  lick = sum of the
               six lick-locked factors (master_v1 "lick")
  (a) eta2_side / eta2_dist   share of trial variance explained by the spout (between-spout / total);
      eta2_tongue             the same for the 3-D tongue's lateral (ml) position (mean of the window's licks)
      sd_side                 within-spout SD of the side index (how much side-specific face movement there is at all)
  (c) r_side_tongue           within-spout correlation of the side index with the tongue's ml (spout means removed)
  (b) r_<area>_side           within-spout PARTIAL correlation of the area response with the side index, lick amount
                              partialled out (SSp-m right / left, contra - ipsi, MOp right / left); same for distance
Per animal x epoch, epoch - pre with the nested bootstrap (analysis_kit.boot_delta). Rule 8: per-animal tables printed.

Usage: python scripts/fr_direction_mechanism.py [--jobs 6]
Output: <E>/fr_encoding_summary/direction_mechanism/ (sessions.csv, bootstrap.csv, figures/fr_direction_mechanism.png)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

BUNDLE = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/enc_v36_full_20261024/inputs")
LICK = ("fr_lick_ongoing_7Hz", "fr_lick_right", "fr_lick_left", "fr_lick_fast_close", "fr_lick_far_nose", "fr_lick_slow_late")
AREAS = ("SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left")
T0, T1, B0 = 0.0, 1.5, -0.5
EPS = ["acute", "subacute", "chronic"]
MET = ["eta2_side", "eta2_dist", "eta2_tongue", "sd_side", "r_side_tongue",
       "r_SSp-m_right_side", "r_SSp-m_left_side", "r_SSp-m_c-i_side", "r_MOp_right_side", "r_MOp_left_side",
       "r_SSp-m_right_dist", "r_SSp-m_left_dist", "r_SSp-m_c-i_dist"]


def eta2(y, g):
    ok = np.isfinite(y)
    y, g = y[ok], g[ok]
    if len(y) < 20:
        return np.nan
    tot = ((y - y.mean()) ** 2).sum()
    btw = sum(((y[g == k].mean() - y.mean()) ** 2) * (g == k).sum() for k in np.unique(g))
    return float(btw / tot) if tot > 0 else np.nan


def demean(y, g):
    y = y.astype(float).copy()
    for k in np.unique(g):
        m = g == k
        y[m] = y[m] - np.nanmean(y[m])
    return y


def pcorr(x, y, z):
    """Correlation of x and y after regressing both on z (and an intercept); finite rows only."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 20:
        return np.nan
    Z = np.c_[np.ones(ok.sum()), z[ok]]
    rx = x[ok] - Z @ np.linalg.lstsq(Z, x[ok], rcond=None)[0]
    ry = y[ok] - Z @ np.linalg.lstsq(Z, y[ok], rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def session(pk: Path) -> dict | None:
    from wfield_local import config, epochs
    lab = pk.stem
    if lab[-4:] < "0801":
        return None
    d = pickle.loads(pk.read_bytes())
    fr = d.get("fr")
    if not fr or "X" not in fr:
        return None
    V, ft, Ub, _flat, parea, names = d["imaging"][:6]
    X, fn = np.asarray(fr["X"], float), fr["names"]
    ix = {n: i for i, n in enumerate(fn)}
    b = d["behaviour"]
    tr = b.trials[b.trials.engaged].reset_index(drop=True)
    code = {v: k for k, v in names.items()}
    U = {a: Ub[parea == code[a]].mean(0) for a in AREAS if a in code}
    ml = np.asarray(b.exec_xyz[0], float) if b.exec_xyz is not None else np.full(len(b.lick_peak_s), np.nan)
    rows = []
    for _, t in tr.iterrows():
        w = np.flatnonzero((ft >= t.cue_s + T0) & (ft < t.cue_s + T1))
        bl = np.flatnonzero((ft >= t.cue_s + B0) & (ft < t.cue_s))
        if len(w) < 10 or len(bl) < 3:
            continue
        f = np.nanmean(X[w], 0)
        r = {"pos": t.pos, "side": f[ix["fr_lick_left"]] - f[ix["fr_lick_right"]],
             "dist": f[ix["fr_lick_far_nose"]] - f[ix["fr_lick_fast_close"]], "lick": sum(f[ix[k]] for k in LICK)}
        for a, u in U.items():
            r[a] = float((V[w] @ u).mean() - (V[bl] @ u).mean())
        lk = (b.lick_peak_s >= t.cue_s + T0) & (b.lick_peak_s < t.cue_s + T1) & np.isfinite(ml)
        r["tongue_ml"] = float(ml[lk].mean()) if lk.any() else np.nan
        rows.append(r)
    D = pd.DataFrame(rows)
    if len(D) < 40:
        return None
    g = D.pos.to_numpy()
    D["SSp-m_c-i"] = D["SSp-m_right"] - D["SSp-m_left"]
    out = {"label": lab, "animal": config.animal_of(lab), "epoch": epochs.epoch_of(lab), "n_trials": len(D),
           "eta2_side": eta2(D.side.to_numpy(), g), "eta2_dist": eta2(D.dist.to_numpy(), g),
           "eta2_tongue": eta2(D.tongue_ml.to_numpy(), g), "sd_side": float(np.nanstd(demean(D.side.to_numpy(), g)))}
    s, di, lk = (demean(D[c].to_numpy(), g) for c in ("side", "dist", "lick"))
    tm = demean(D.tongue_ml.to_numpy(), g)
    ok = np.isfinite(s) & np.isfinite(tm)
    out["r_side_tongue"] = float(np.corrcoef(s[ok], tm[ok])[0, 1]) if ok.sum() >= 20 else np.nan
    for a in ("SSp-m_right", "SSp-m_left", "SSp-m_c-i", "MOp_right", "MOp_left"):
        y = demean(D[a].to_numpy(), g)
        out[f"r_{a}_side"] = pcorr(y, s, lk)
        if a.startswith("SSp-m"):
            out[f"r_{a}_dist"] = pcorr(y, di, lk)
    return out


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed

    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--plot-only", action="store_true", help="redraw the summary figure from sessions.csv / bootstrap.csv")
    a = ap.parse_args(argv)
    out = E / "fr_encoding_summary" / "direction_mechanism"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        summary_figure(pd.read_csv(out / "sessions.csv"), pd.read_csv(out / "bootstrap.csv"), out)
        return 0
    got = Parallel(n_jobs=a.jobs)(delayed(session)(pk) for pk in sorted(BUNDLE.glob("PS*_*.pkl")))
    S = pd.DataFrame([g for g in got if g is not None])
    S.to_csv(out / "sessions.csv", index=False)
    print(f"{len(S)} sessions", flush=True)
    pd.set_option("display.width", 240)
    A = S.groupby(["animal", "epoch"])[MET].mean()
    print("== per animal x epoch (rule 8)")
    print(A.reindex(["pre", "acute", "subacute", "chronic"], level=1).round(3).to_string())
    rows = []
    for m in MET:
        pre = {x: S[(S.epoch == "pre") & (S.animal == x)][m].dropna().to_numpy() for x in AN}
        for ep in EPS:
            post = {x: S[(S.epoch == ep) & (S.animal == x)][m].dropna().to_numpy() for x in AN}
            com = [x for x in AN if len(pre[x]) and len(post[x])]
            if not com:
                continue
            iv, dr = ak.boot_delta({x: post[x] for x in com}, {x: pre[x] for x in com}, np.random.default_rng(0), draws=True)
            rows.append({"metric": m, "epoch": ep, "pre": float(np.mean([pre[x].mean() for x in com])), "point": iv.point,
                         "lo": iv.lo, "hi": iv.hi, "p": float(ak.boot_p(dr)),
                         "n_down": sum(post[x].mean() < pre[x].mean() for x in com), "n": len(com)})
    B = pd.DataFrame(rows)
    B.to_csv(out / "bootstrap.csv", index=False)
    print("== epoch - pre (pre level, change [95% nested CI], p, animals down)")
    print(B.round(3).to_string(index=False))

    groups = [("(a) behaviour: spout-specificity", ["eta2_side", "eta2_dist", "eta2_tongue"]),
              ("(a) amount of side-specific face movement", ["sd_side"]),
              ("(c) FR side index tracks the tongue", ["r_side_tongue"]),
              ("(b) cortex covaries with the FR side index", ["r_SSp-m_right_side", "r_SSp-m_left_side", "r_SSp-m_c-i_side",
                                                              "r_MOp_right_side", "r_MOp_left_side"]),
              ("(b) cortex covaries with the FR distance index", ["r_SSp-m_right_dist", "r_SSp-m_left_dist", "r_SSp-m_c-i_dist"])]
    EP4 = ["pre", "acute", "subacute", "chronic"]
    MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    fig, axs = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.6))
    for ax, (ttl, ms) in zip(axs, groups):
        for k, m in enumerate(ms):
            for an in AN:
                y = [A.loc[(an, e), m] if (an, e) in A.index else np.nan for e in EP4]
                x = np.arange(4) + (k - (len(ms) - 1) / 2) * 0.15
                ax.plot(x, y, marker=MK[an], color=f"C{k}", lw=0.7, ms=4, alpha=0.8, label=m if an == "PS92" else None)
        ax.set_xticks(range(4)); ax.set_xticklabels(EP4, fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=6.5, frameon=False, loc="best")
    fig.suptitle("NEURAL + FACE: why FaceRhythm's direction factors explain less orofacial cortex after the stroke. Per animal "
                 "(markers) x epoch, engaged trials, cue window 0-1.5 s.\nside = fr_lick_left - fr_lick_right; within-spout "
                 "correlations (spout means removed); cortex partial correlations control lick amount. Table + bootstrap: "
                 "direction_mechanism/bootstrap.csv", fontsize=9)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.08, wspace=0.25)
    fig.savefig(out / "figures" / "fr_direction_mechanism.png", dpi=115)
    plt.close(fig)
    summary_figure(S, B, out)
    print("->", out)
    return 0


SUMMARY = [("(a) face side index differs by spout", "eta2_side", "eta2 (0-1)"),
           ("(a) tongue lateral position differs by spout", "eta2_tongue", "eta2 (0-1)"),
           ("(a) within-spout side movement (SD)", "sd_side", "% of pre"),
           ("(c) FR side index tracks tongue (within spout)", "r_side_tongue", "r"),
           ("(b) contralesional SSp-m vs FR side | lick", "r_SSp-m_right_side", "partial r"),
           ("(b) ipsilesional SSp-m vs FR side | lick", "r_SSp-m_left_side", "partial r"),
           ("(b) SSp-m contra - ipsi vs FR side | lick", "r_SSp-m_c-i_side", "partial r"),
           ("(b) SSp-m contra - ipsi vs FR distance | lick", "r_SSp-m_c-i_dist", "partial r")]


def summary_figure(S: pd.DataFrame, B: pd.DataFrame, out: Path) -> None:
    """Epoch - pre per metric: bar = nested-bootstrap point, whisker = 95 % CI, dots = per-animal change (rule 8)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    A = S.groupby(["animal", "epoch"])[[m for _, m, _ in SUMMARY]].mean()
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.2))
    for ax, (ttl, m, unit) in zip(axs.ravel(), SUMMARY):
        pre_lvl = float(B[(B.metric == m)].pre.iloc[0])
        sc = 100.0 / pre_lvl if unit == "% of pre" else 1.0
        for k, ep in enumerate(EPS):
            r = B[(B.metric == m) & (B.epoch == ep)].iloc[0]
            sig = not (r.lo <= 0 <= r.hi)
            ax.bar(k, r.point * sc, color="#c0392b" if sig and r.point < 0 else "#2471a3" if sig else "#b3b3b3", width=0.6)
            ax.plot([k, k], [r.lo * sc, r.hi * sc], color="k", lw=1.2)
            for j, an in enumerate(AN):
                if (an, ep) in A.index and (an, "pre") in A.index:
                    ax.plot(k - 0.18 + 0.12 * j, (A.loc[(an, ep), m] - A.loc[(an, "pre"), m]) * sc, MK[an], color="k",
                            ms=3.5, mfc="white", mew=0.8, label=an if k == 0 else None)
            if sig:
                ax.text(k, max(r.hi, 0) * sc, "*", ha="center", va="bottom", fontsize=12)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(3)); ax.set_xticklabels(EPS, fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel(f"change from pre ({unit})" if unit == "% of pre" else f"change from pre ({unit})", fontsize=8)
        ax.text(0.02, 0.97, f"pre = {pre_lvl:.3g}" if unit != "% of pre" else f"pre SD = {pre_lvl:.0f} (FR units)",
                transform=ax.transAxes, fontsize=7.5, va="top")
        ax.margins(y=0.25)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    fig.suptitle("NEURAL + FACE: why FaceRhythm's direction factors explain less orofacial cortex after the stroke "
                 "(engaged trials, cue window 0-1.5 s; side index = fr_lick_left - fr_lick_right).\nBar = epoch - pre, "
                 "nested bootstrap (animals > sessions); whisker 95 % CI; * CI excludes 0 (red = decrease, blue = increase); "
                 "dots = per-animal change. Lesion left: right SSp-m = contralesional.", fontsize=9.5)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.06, wspace=0.32, hspace=0.38)
    fig.savefig(out / "figures" / "fr_direction_mechanism_summary.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
