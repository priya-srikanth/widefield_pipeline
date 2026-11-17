"""Step 2 of the ITI negative-AUC diagnostic (see neg_auc_diag.py). Read-only on production; writes tables to <out_dir>.

usage: python neg_auc_diag.py --analyze <all.pkl> <dock.pkl> <out_dir> [<fig_dir>]
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
EP = ["acute", "subacute", "chronic"]
PROD = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/"
            "imaging_encoding/iti_licks")
TAG = ("hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-"
       "stopA2jexrt3500c70_daqlk")


def trap(v, L, w0, w1):
    m = (L >= w0 - 1e-9) & (L <= w1 + 1e-9)
    return float(np.trapezoid(v[m], L[m]))


def features(C: pd.DataFrame, L: np.ndarray) -> pd.DataFrame:
    """Per row: AUCs, alternative baselines, per-side diagnostics."""
    out = C.drop(columns=["curve_c", "curve_i", "raw_c", "raw_i"]).copy()
    pre_m = (L >= -2.0) & (L <= -0.5)
    for s in ("c", "i"):
        cv = np.stack(C[f"curve_{s}"].to_numpy())
        rw = np.stack(C[f"raw_{s}"].to_numpy())
        for w0, w1 in ((0, 1.0), (0, 0.5), (0.5, 1.0)):
            out[f"{s}_{w0:g}-{w1:g}"] = [trap(v, L, w0, w1) for v in cv]
        for nm, (b0, b1) in {"bl": (-0.5, -0.25), "be": (-1.5, -1.0)}.items():   # late / early alternative baselines
            bm = (L >= b0 - 1e-9) & (L <= b1 + 1e-9)
            cb = cv - np.nanmean(cv[:, bm], 1, keepdims=True)
            out[f"{s}_{nm}_0-1"] = [trap(v, L, 0, 1.0) for v in cb]
            out[f"{s}_{nm}_0-0.5"] = [trap(v, L, 0, 0.5) for v in cb]
        out[f"{s}_early"] = np.nanmean(cv[:, (L >= -2.0) & (L <= -1.5)], 1)      # level 1 s before the baseline
        out[f"{s}_prelick"] = np.nanmean(cv[:, (L >= -0.5) & (L <= -0.05)], 1)
        sl = []
        for v in cv:
            y = v[pre_m]
            ok = np.isfinite(y)
            sl.append(np.polyfit(L[pre_m][ok], y[ok], 1)[0] if ok.sum() > 5 else np.nan)
        out[f"{s}_slope"] = sl                                                  # dF/F x1e3 per s, -2..-0.5 s
        # linear extrapolation of the pre-trend into the window (relative to the baseline centre -0.75 s)
        out[f"{s}_drift_0-1"] = out[f"{s}_slope"] * 1.25
        out[f"{s}_drift_0-0.5"] = out[f"{s}_slope"] * 0.5
        out[f"{s}_peak"] = np.nanmax(cv[:, (L >= 0) & (L <= 0.5)], 1)
        out[f"{s}_min_late"] = np.nanmin(cv[:, (L >= 0.5) & (L <= 1.0)], 1)
        out[f"{s}_rawbase"] = np.nanmean(rw[:, (L >= -1.0) & (L <= -0.5)], 1)   # vs session mean (drift-removed = 0)
    return out


def add_metrics(F: pd.DataFrame) -> pd.DataFrame:
    def logr(c, i):
        ok = (c > 0) & (i > 0)
        return np.where(ok, np.log2(np.clip(c, 1e-12, None) / np.clip(i, 1e-12, None)), np.nan)

    def bi(c, i):
        den = np.abs(c) + np.abs(i)
        return np.where(den > 0, (c - i) / den, np.nan)
    F["logr_0-1"] = logr(F["c_0-1"], F["i_0-1"])
    F["logr_0-0.5"] = logr(F["c_0-0.5"], F["i_0-0.5"])
    F["bi_0-1"] = bi(F["c_0-1"], F["i_0-1"])
    F["bi_0-0.5"] = bi(F["c_0-0.5"], F["i_0-0.5"])
    F["diff_0-1"] = F["c_0-1"] - F["i_0-1"]
    F["logr_bl_0-1"] = logr(F["c_bl_0-1"], F["i_bl_0-1"])
    F["logr_be_0-1"] = logr(F["c_be_0-1"], F["i_be_0-1"])
    F["logr_0-1_n10"] = np.where(F.n >= 10, F["logr_0-1"], np.nan)
    F["logr_0-1_n15"] = np.where(F.n >= 15, F["logr_0-1"], np.nan)
    return F


METRICS = {"logr_0-1": "PRODUCTION log2(c/i) 0..1 s", "logr_0-0.5": "log2(c/i) 0..0.5 s",
           "bi_0-1": "bounded (c-i)/(|c|+|i|) 0..1 s", "bi_0-0.5": "bounded 0..0.5 s", "diff_0-1": "c - i 0..1 s",
           "logr_bl_0-1": "log2 0..1 s, baseline -0.5..-0.25 s", "logr_be_0-1": "log2 0..1 s, baseline -1.5..-1.0 s",
           "logr_0-1_n10": "log2 0..1 s, rows >= 10 licks", "logr_0-1_n15": "log2 0..1 s, rows >= 15 licks"}


def lr_test(full, red, df):
    import statsmodels.formula.api as smf
    from scipy import stats
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f1 = smf.mixedlm(full, df, groups=df["animal"]).fit(reml=False)
        f0 = smf.mixedlm(red, df, groups=df["animal"]).fit(reml=False)
    k = len(f1.fe_params) - len(f0.fe_params)
    lr = 2 * (f1.llf - f0.llf)
    conv = bool(f1.converged and f0.converged and np.isfinite(lr))
    nw = sum("Convergence" in type(x.message).__name__ or "converge" in str(x.message).lower() for x in w)
    return float(lr), float(stats.chi2.sf(lr, k)) if np.isfinite(lr) else np.nan, conv, nw


def run_tests(F: pd.DataFrame, mode: str, sets) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    tests, models = [], []
    for (st, a), g in F[F.set.isin(sets)].groupby(["set", "pair"]):
        for v in METRICS:
            Gall = g.dropna(subset=[v]).rename(columns={v: "y"})
            G = Gall[Gall.spout.isin(POS)]
            for name, full, red, df in (("pre spout", "y ~ C(spout)", "y ~ 1", G[G.epoch == "pre"]),
                                        ("spout x epoch", "y ~ C(spout) * C(epoch)", "y ~ C(spout) + C(epoch)", G),
                                        ("epoch", "y ~ C(spout) + C(epoch)", "y ~ C(spout)", G)):
                try:
                    lr, p, conv, nw = lr_test(full, red, df)
                except Exception as ex:  # noqa: BLE001
                    lr, p, conv, nw = np.nan, np.nan, False, -1
                    print("FAILED", mode, st, a, v, name, ex)
                models.append({"mode": mode, "set": st, "pair": a, "metric": v, "test": name, "lr": lr, "p": p,
                               "converged": conv, "n_warn": nw, "n_rows": len(df)})
            for p in [*POS, "all", "pooled"]:
                if p == "all":
                    h = G.groupby(["label", "animal", "epoch"]).y.mean().reset_index()
                else:
                    h = Gall[Gall.spout == p]
                ref = {an: t.y.to_numpy() for an, t in h[h.epoch == "pre"].groupby("animal")}
                for ep in EP:
                    post = {an: t.y.to_numpy() for an, t in h[h.epoch == ep].groupby("animal")}
                    com = sorted(an for an in set(post) & set(ref) if len(post[an]) and len(ref[an]))
                    if len(com) < 3:
                        continue
                    iv = ak.boot_delta({an: post[an] for an in com}, {an: ref[an] for an in com},
                                       np.random.default_rng(0))
                    if iv is None:
                        continue
                    tests.append({"mode": mode, "set": st, "pair": a, "metric": v, "spout": p, "epoch": ep,
                                  "delta": iv.point, "lo": iv.lo, "hi": iv.hi, "sig": bool(iv.lo > 0 or iv.hi < 0),
                                  "n_animals": len(com)})
    return pd.DataFrame(tests), pd.DataFrame(models)


def main(argv):
    pa, pd_, out = argv[0], argv[1], Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    A = pickle.load(open(pa, "rb"))
    Dk = pickle.load(open(pd_, "rb"))
    L = A["L"]
    FA = add_metrics(features(A["C"], L)).assign(mode="all")
    FD = add_metrics(features(Dk["C"], L)).assign(mode="dock")
    # the docked run only changes the iti set; cued / cue identical -> keep its iti rows
    FD = FD[FD.set == "iti"]
    F = pd.concat([FA, FD], ignore_index=True)
    F.to_pickle(out / "features.pkl")
    # reproduction check vs production csv
    for mode, f in (("all", PROD / TAG / "lateral_ratio_sessions.csv"),
                    ("dock", PROD / "lateral_ratio_dock_20261117" / TAG / "lateral_ratio_sessions.csv")):
        P = pd.read_csv(f)
        m = F[F["mode"] == mode].merge(P, on=["label", "set", "spout", "pair"], suffixes=("", "_p"))
        print(f"== repro {mode}: {len(m)} matched rows; max|dAUC c 0-1| {np.nanmax(np.abs(m['c_0-1'] - m['c_0-1_p'])):.2e}; "
              f"undefined agree {(m['logr_0-1'].isna() == m['logr_0-1_p'].isna()).mean():.4f}")
    if len(argv) > 3:
        figs(F, A, Dk, Path(argv[3]))
    sets_all = ["cued", "cue", "iti"]
    T1, M1 = run_tests(FA, "all", sets_all)
    T2, M2 = run_tests(FD, "dock", ["iti"])
    pd.concat([T1, T2]).to_csv(out / "alt_tests.csv", index=False)
    pd.concat([M1, M2]).to_csv(out / "alt_models.csv", index=False)
    print("done")


def figs(F, A, Dk, fd: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd.mkdir(parents=True, exist_ok=True)
    L = A["L"]
    for mode, src in (("all", A["C"]), ("dock", Dk["C"])):
        C = src[src.set == "iti"].reset_index(drop=True)
        und = ~((np.array([trap(v, L, 0, 1) for v in C.curve_c]) > 0) & (np.array([trap(v, L, 0, 1) for v in C.curve_i]) > 0))
        fig, axs = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
        for k, a in enumerate(["ALM", "tjM1", "SSp-m", "SSp-n", "orofacial"]):
            for r, s in enumerate(("c", "i")):
                ax = axs[r, k]
                sel = (C.pair == a).to_numpy() & (C.spout != "pooled").to_numpy()
                for msk, col, lab in ((sel & ~und, "0.3", "defined"), (sel & und, "tab:red", "undefined")):
                    if msk.sum() == 0:
                        continue
                    X = np.stack(C.loc[msk, f"curve_{s}"].to_numpy())
                    mu = np.nanmean(X, 0)
                    se = np.nanstd(X, 0) / np.sqrt(len(X))
                    ax.plot(L, mu, color=col, label=f"{lab} (n={msk.sum()})")
                    ax.fill_between(L, mu - se, mu + se, color=col, alpha=0.2, lw=0)
                ax.axvspan(-1.0, -0.5, color="0.9")
                ax.axhline(0, color="0.6", lw=0.7)
                ax.axvline(0, color="0.6", lw=0.7)
                ax.set_xlim(-2, 2)
                ax.set_title(f"{a} {'contra (right)' if s == 'c' else 'ipsi (left)'}", fontsize=9)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if k == 0:
                    ax.set_ylabel("dF/F x1e3, baseline -1..-0.5 s")
                ax.legend(fontsize=7, frameon=False)
        fig.suptitle(f"ITI bout starts ({'docked spout' if mode == 'dock' else 'all ITI'}): session x spout mean curves, "
                     "rows with an undefined 0..1 s log ratio (either AUC <= 0) vs defined; mean +- SEM over rows. "
                     "Grey band = baseline. Diagnostic (neg_auc_diag 11-17)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(fd / f"neg_auc_curves_iti_{mode}.png", dpi=90)
        plt.close(fig)
