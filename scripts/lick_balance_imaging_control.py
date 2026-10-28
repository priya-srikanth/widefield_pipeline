"""Is the post-stroke shift in the HEMISPHERIC BALANCE of the lick response an IMAGING artefact?  (Priya 10-28)

The claim (scripts/lick_lateral_ratio.py, lateral_ratio_sessions.csv): log2(contra AUC / ipsi AUC) of the lick-evoked dF/F,
X_right (contralesional) vs X_left (ipsilesional), rises post-stroke (e.g. cued licks to contra-side targets, tjM1 +0.20).
Alternative: a ONE-SIDED imaging change on the lesioned side --
  (a) baseline fluorescence F0 (dF/F = dF / F0, so an F0 fall on one side inflates that side's dF/F by the same factor:
      the artefact predicts  delta logr = - delta log2(F0_R / F0_L));
  (b) haemodynamics: the 415 nm (isosbestic) channel's own lick-evoked response, contra vs ipsi, on the SAME licks;
  (c) vessel / edge contamination: the balance recomputed on ROI pixels that are in the stat (interior) mask and NOT in
      the top 5 % vesselness (iti_hemo_check's definition) -- "clean" pixels.

SAME LICKS, WINDOWS AND ESTIMATOR AS lick_lateral_ratio: cued = first cued lick (times = <OUT_W>/cued_first_lick_times.csv,
the rows of cued_wide/<label>.npz, with its spout), ITI = bout starts with gap >= GAP_MAIN (2 s); LAGS_W, baseline
-1.0..-0.5 s per lick (nanmean), session-mean curve, trapezoid AUC 0..1.0 s (and 0..0.5 s); orofacial = mean of the four
areas' curves per hemisphere; MIN_N licks per cell. Signals: adopted (hemo_meegkit_hpfit SVTcorr; must REPRODUCE
lateral_ratio_sessions.csv -- checked), raw470 (uncorrected functional half of raw SVT), raw415 (415 half), each projected
on the ROI means of U_atlas, slow drift (>= 600 s cosines) removed, x1e3 -- exactly iti_hemo_check / wide_session.
RAW SVT channels are interleaved (K, 2T); functional slot = hemo_variants.functional_channel(label) (0 for PS92_0828).
F0 = mean of frames_average_atlas (raw counts, per channel) over the ROI's pixels.

415 BALANCE: the 415 lick response is small and sign-variable, so a log ratio of it is undefined in many sessions; it is
reported as (c415 - i415) / the animal's PRE mean neural (adopted) response ((c+i)/2), i.e. in units of the neural
response it could contaminate (and, where both are > 0, as log2 too).
Tests: per-animal tables first; analysis_kit.boot_delta change vs pre; tracking = per-session change from the animal's pre
mean, per-animal correlations, then MixedLM dN ~ dX + (1|animal) (sessions as observations: ANTI-CONSERVATIVE), and
logr ~ C(epoch) [+ covariates] + (1|animal) to see whether the epoch shift survives adjustment.
Outputs (il.OUT_W): balimg_sessions.csv (per session x signal x mask x set x spout x pair AUCs), balimg_f0.csv,
balimg_measures.csv, balimg_change.csv, balimg_track.csv, balimg_models.csv, figures/balimg_control.png.
Usage: python scripts/lick_balance_imaging_control.py [--jobs 6] [--summarise]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_cued_ramp as cr  # noqa: E402
from fr_musall import AN  # noqa: E402

OROFACIAL = ["ALM", "tjM1", "SSp-m", "SSp-n"]
PAIRS = [*OROFACIAL, "orofacial"]
HEMI = ("right", "left")                     # right = contralesional, left = ipsilesional
SIGNALS = ("adopted", "raw470", "raw415")
MASKS = ("full", "clean")
WINDOWS = ((0.0, 1.0), (0.0, 0.5))
L = cr.LAGS_W
BASE = cr.BASE_SPOUT
TOP_FRAC, VESSEL_SIGMA = 0.05, 12.0          # = iti_hemo_check
EP = il.EP4
POST = EP[1:]
SIDES = ("side_ipsi", "side_mid", "side_contra")
SESS = il.OUT_W / "balimg_sessions.csv"
F0F = il.OUT_W / "balimg_f0.csv"
_TT = None


def _times(label: str) -> np.ndarray:
    global _TT
    if _TT is None:
        T = pd.read_csv(il.OUT_W / "cued_first_lick_times.csv", float_precision="round_trip")
        _TT = {x: g.sort_values("row").t.to_numpy(float) for x, g in T.groupby("label")}
    return _TT[label]


def _auc(m: np.ndarray, w0: float, w1: float) -> float:
    w = (L >= w0) & (L <= w1)
    return float(np.trapezoid(m[w], L[w]))


def work(label: str) -> dict:
    """One session (module-level for spawn): per signal x mask x set x spout x pair contra / ipsi AUC; F0 per ROI."""
    from scipy import ndimage

    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.hemo_variants import functional_channel
    d = il._load(label)
    z = np.load(il.OUT_W / cr.WIDE / f"{label}.npz")
    tl, pos = _times(label), np.asarray(z["pos"], str)
    if len(tl) != len(pos):
        raise RuntimeError(f"{label}: {len(tl)} cached first-lick times vs {len(pos)} cued_wide rows")
    bt = np.asarray(d["bout_t"], float)[np.asarray(d["bout_gap"], float) >= il.GAP_MAIN]
    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    n = len(ft)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    if list(M) != list(z["cols"]):
        raise RuntimeError(f"{label}: ROI columns differ from cued_wide")
    fc = functional_channel(label)
    man = json.loads((res / "hemo_meegkit_hpfit" / "manifest.json").read_text())
    if int(man.get("functional_channel", fc)) != fc:
        raise RuntimeError(f"{label}: functional slot {fc} vs manifest {man.get('functional_channel')}")
    # vessel pixels (iti_hemo_check definition): top 5 % of the high-passed NEGATED mean 470 image within the stat mask
    fa = np.load(res / "allen_aligned_affine8v1" / "frames_average_atlas.npy").astype(np.float64)   # (2, H, W)
    img = np.nan_to_num(fa[fc])
    vess = (-img - ndimage.gaussian_filter(-img, VESSEL_SIGMA)).ravel()[flat]
    vessel = pstat & (vess >= np.quantile(vess[pstat], 1 - TOP_FRAC))
    keys, cols_m = [], []
    for a in OROFACIAL:
        for h in HEMI:
            c = f"{a}_{h}"
            keys += [(c, "full"), (c, "clean")]
            cols_m += [M[c], M[c] & pstat & ~vessel]
    fl = fa.reshape(2, -1)[:, flat]
    f0 = {(k, ch): float(np.nanmean(fl[slot][m])) for k, m in zip(keys, cols_m)
          for ch, slot in (("470", fc), ("415", (fc + 1) % 2))}
    npix = {k: int(m.sum()) for k, m in zip(keys, cols_m)}
    Ra = np.stack([Ub[m].mean(0) for m in cols_m], 1)                        # (K, 16)
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    if svt.shape[1] // 2 < n:
        raise RuntimeError(f"{label}: raw SVT {svt.shape[1] // 2} pairs < {n} frames")
    rows, val = [], {}
    for sig in SIGNALS:
        if sig == "adopted":
            R = V[:n].astype(np.float64) @ Ra.astype(np.float64)
        else:
            slot = fc if sig == "raw470" else (fc + 1) % 2
            S = np.asarray(svt[:, slot::2][:, :n], np.float32)
            R = (Ra.T.astype(np.float32) @ S).T.astype(np.float64)
            del S
        R -= Q @ (Q.T @ R)
        R *= 1e3

        def samp(t, R=R):
            X = il._sample(R, ft, t, L).astype(np.float64)
            out = (t[:, None] + L[None, :] < ft[0]) | (t[:, None] + L[None, :] > ft[-1])
            X[out] = np.nan
            return X
        Xc, Xi = samp(tl), samp(bt)
        if sig == "adopted":                    # reproduce the stored cued_wide traces (float16) for the full masks
            j = [list(z["cols"]).index(k[0]) for k in keys if k[1] == "full"]
            fi = [i for i, k in enumerate(keys) if k[1] == "full"]
            ref = z["lick"].astype(np.float64)[:, :, j]
            val["maxdiff"] = float(np.nanmax(np.abs(Xc[:, :, fi] - ref))) if len(ref) else np.nan
        for st, X, P in (("cued", Xc, pos), ("iti", Xi, None)):
            groups = {"pooled": np.ones(len(X), bool)}
            if P is not None:
                groups.update({g: np.array([p.endswith(g.split("_")[1]) for p in P], bool) for g in SIDES})
            for g, sel in groups.items():
                if sel.sum() < cr.MIN_N:
                    continue
                Y = X[sel]
                Y = Y - np.nanmean(Y[:, (L >= BASE[0]) & (L <= BASE[1]), :], 1, keepdims=True)
                m = np.nanmean(Y, 0)                                          # (lags, 16)
                for msk in MASKS:
                    idx = {(a, h): keys.index((f"{a}_{h}", msk)) for a in OROFACIAL for h in HEMI}
                    for a in PAIRS:
                        aa = OROFACIAL if a == "orofacial" else [a]
                        mc = m[:, [idx[(q, "right")] for q in aa]].mean(1)
                        mi = m[:, [idx[(q, "left")] for q in aa]].mean(1)
                        r = {"label": label, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "signal": sig,
                             "mask": msk, "set": st, "spout": g, "pair": a, "n": int(sel.sum())}
                        for w0, w1 in WINDOWS:
                            r[f"c_{w0:g}-{w1:g}"], r[f"i_{w0:g}-{w1:g}"] = _auc(mc, w0, w1), _auc(mi, w0, w1)
                        rows.append(r)
        del R
    frows = []
    for msk in MASKS:
        for a in PAIRS:
            aa = OROFACIAL if a == "orofacial" else [a]
            r = {"label": label, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "mask": msk, "pair": a}
            for ch in ("470", "415"):
                for h in HEMI:
                    r[f"f0_{ch}_{h}"] = float(np.mean([f0[((f"{q}_{h}", msk), ch)] for q in aa]))
            for h in HEMI:
                r[f"npix_{h}"] = int(sum(npix[(f"{q}_{h}", msk)] for q in aa))
            frows.append(r)
    return {"rows": rows, "f0": frows, "maxdiff": val.get("maxdiff", np.nan), "func": fc}


def extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    res, fails = ak.fan_sessions(il.labels(), work, jobs=jobs)
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    S = pd.DataFrame([r for _x, v in res for r in v["rows"]])
    F = pd.DataFrame([r for _x, v in res for r in v["f0"]])
    md = pd.Series({x: v["maxdiff"] for x, v in res})
    print(f"== validation: max |adopted re-sampled - cued_wide stored| over sessions = {md.max():.4f} (float16 store)")
    S.to_csv(SESS, index=False, float_format="%.10g")
    F.to_csv(F0F, index=False, float_format="%.10g")


# ------------------------------------------------------------------------------------------------------------ SUMMARY
W = "0-1"
CELLS = (("cued", "side_contra"), ("cued", "pooled"), ("iti", "pooled"))
MEAS = {"neural": "neural balance log2(c/i), adopted", "clean": "neural balance, clean pixels (interior, non-vessel)",
        "raw470": "uncorrected 470 balance log2(c/i)", "dF": "dF balance = neural + F0 log ratio (undo F0 scaling)",
        "f0_470": "F0 470 log2(R/L)", "f0_415": "F0 415 log2(R/L)", "h415": "415 balance (c-i) / pre neural resp",
        "h415_lr": "415 log2(c/i) (where both > 0)"}


def _lr(c, i):
    c, i = np.asarray(c, float), np.asarray(i, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((c > 0) & (i > 0), np.log2(c / i), np.nan)


def measures() -> pd.DataFrame:
    S = pd.read_csv(SESS)
    F = pd.read_csv(F0F)
    # reproduction of lateral_ratio_sessions.csv (adopted, full masks)
    LR = pd.read_csv(il.OUT_W / "lateral_ratio_sessions.csv")
    a = S[(S.signal == "adopted") & (S["mask"] == "full")].merge(LR, on=["label", "set", "spout", "pair"], suffixes=("", "_lr"))
    print(f"== reproduction vs lateral_ratio_sessions.csv: {len(a)} rows; max |c - c_lr| {np.abs(a[f'c_{W}'] - a[f'c_{W}_lr']).max():.4f}, "
          f"max |i - i_lr| {np.abs(a[f'i_{W}'] - a[f'i_{W}_lr']).max():.4f} (AUC scale ~{a[f'c_{W}'].abs().median():.1f})")
    k = ["label", "animal", "epoch", "set", "spout", "pair"]
    P = S.pivot_table(index=k, columns=["signal", "mask"], values=[f"c_{W}", f"i_{W}"])
    out = pd.DataFrame(index=P.index)
    out["neural"] = _lr(P[(f"c_{W}", "adopted", "full")], P[(f"i_{W}", "adopted", "full")])
    out["clean"] = _lr(P[(f"c_{W}", "adopted", "clean")], P[(f"i_{W}", "adopted", "clean")])
    out["raw470"] = _lr(P[(f"c_{W}", "raw470", "full")], P[(f"i_{W}", "raw470", "full")])
    out["h415_lr"] = _lr(P[(f"c_{W}", "raw415", "full")], P[(f"i_{W}", "raw415", "full")])
    out["c415"], out["i415"] = P[(f"c_{W}", "raw415", "full")], P[(f"i_{W}", "raw415", "full")]
    out["cn"], out["in"] = P[(f"c_{W}", "adopted", "full")], P[(f"i_{W}", "adopted", "full")]
    out = out.reset_index()
    pre_resp = out[out.epoch == "pre"].assign(r=lambda t: (t.cn + t["in"]) / 2) \
        .groupby(["animal", "set", "spout", "pair"]).r.mean().rename("pre_resp")
    out = out.join(pre_resp, on=["animal", "set", "spout", "pair"])
    out["h415"] = (out.c415 - out.i415) / out.pre_resp
    Ff = F[F["mask"] == "full"].assign(f0_470=lambda t: np.log2(t.f0_470_right / t.f0_470_left),
                                        f0_415=lambda t: np.log2(t.f0_415_right / t.f0_415_left))
    out = out.merge(Ff[["label", "pair", "f0_470", "f0_415"]], on=["label", "pair"], how="left")
    out["dF"] = out.neural + out.f0_470
    out.to_csv(il.OUT_W / "balimg_measures.csv", index=False, float_format="%.8g")
    return out


def _dpre(g: pd.DataFrame, v: str) -> pd.Series:
    """Per session: value minus its animal's PRE-session mean."""
    return g[v] - g.animal.map(g[g.epoch == "pre"].groupby("animal")[v].mean())


def _boot(g: pd.DataFrame, v: str, ep: str):
    from wfield_local import analysis_kit as ak
    ref = {a: g[(g.animal == a) & (g.epoch == "pre")][v].dropna().to_numpy() for a in AN}
    post = {a: g[(g.animal == a) & (g.epoch == ep)][v].dropna().to_numpy() for a in AN}
    com = [a for a in AN if len(ref[a]) and len(post[a])]
    if len(com) < 3:
        return None, {}
    iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
    return iv, {a: post[a].mean() - ref[a].mean() for a in com}


def summarise() -> None:
    import statsmodels.formula.api as smf
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    D = measures()
    print("== fraction of sessions with an undefined log ratio (an AUC <= 0): "
          + ", ".join(f"{m} {D[m].isna().mean():.3f}" for m in ("neural", "clean", "raw470", "h415_lr")))
    # ---- per-animal tables (rule 8)
    for st, sp in CELLS:
        for a in ("tjM1", "orofacial"):
            g = D[(D.set == st) & (D.spout == sp) & (D.pair == a)]
            print(f"\n-- per animal x epoch means, {st} {sp} {a} (window 0..1.0 s)")
            print(g.groupby(["animal", "epoch"])[list(MEAS)].mean().unstack("epoch")
                  .reorder_levels([1, 0], axis=1).reindex(columns=EP, level=0).T.round(3).to_string())
    # ---- change vs pre (boot_delta)
    rows = []
    for (st, sp, a), g in D.groupby(["set", "spout", "pair"]):
        if (st, sp) not in CELLS:
            continue
        for v in MEAS:
            for ep in POST:
                iv, per = _boot(g, v, ep)
                if iv is None:
                    continue
                rows.append({"set": st, "spout": sp, "pair": a, "measure": v, "epoch": ep, "delta": iv.point, "lo": iv.lo,
                             "hi": iv.hi, "sig": bool(iv.lo > 0 or iv.hi < 0), "n_animals": iv.n_animals,
                             **{f"d_{an}": x for an, x in per.items()}})
    C = pd.DataFrame(rows)
    C.to_csv(il.OUT_W / "balimg_change.csv", index=False, float_format="%.6g")
    C["s"] = [f"{d:+.3f}{'*' if s else ''}" for d, s in zip(C.delta, C.sig)]
    for st, sp in CELLS:
        print(f"\n== change vs pre (boot_delta, animals>sessions; * = 95% CI excludes 0): {st} {sp}")
        print(C[(C.set == st) & (C.spout == sp)].pivot_table(index=["pair", "measure"], columns="epoch", values="s",
                                                               aggfunc="first").reindex(columns=POST).to_string())
    # ---- tracking: per-session change from the animal's pre mean
    trk, mods = [], []
    for (st, sp, a), g in D.groupby(["set", "spout", "pair"]):
        if (st, sp) not in CELLS:
            continue
        g = g.copy()
        for v in MEAS:
            g[f"d_{v}"] = _dpre(g, v)
        for x in ("f0_470", "f0_415", "h415"):
            for an, h in g.groupby("animal"):
                h = h.dropna(subset=["d_neural", f"d_{x}"])
                r = float(np.corrcoef(h["d_neural"], h[f"d_{x}"])[0, 1]) if len(h) > 3 else np.nan
                trk.append({"set": st, "spout": sp, "pair": a, "x": x, "animal": an, "n": len(h), "r": r,
                            "slope": float(np.polyfit(h[f"d_{x}"], h["d_neural"], 1)[0]) if len(h) > 3 else np.nan})
            h = g.dropna(subset=["d_neural", f"d_{x}"])
            try:
                f = smf.mixedlm(f"d_neural ~ d_{x}", h, groups=h["animal"]).fit(reml=True)
                mods.append({"set": st, "spout": sp, "pair": a, "model": f"d_neural ~ d_{x} + (1|animal)",
                             "term": f"d_{x}", "coef": f.params[f"d_{x}"], "se": f.bse[f"d_{x}"],
                             "p": f.pvalues[f"d_{x}"], "n": len(h)})
            except Exception as ex:  # noqa: BLE001
                mods.append({"set": st, "spout": sp, "pair": a, "model": f"d_neural ~ d_{x}: FAILED {ex}"})
        # epoch effect with / without the imaging covariates
        h = g.dropna(subset=["neural", "f0_470", "f0_415", "h415"])
        for cov in ("", " + f0_470", " + f0_415", " + h415", " + f0_470 + f0_415 + h415"):
            try:
                f = smf.mixedlm(f"neural ~ C(epoch, Treatment('pre')){cov}", h, groups=h["animal"]).fit(reml=True)
                for ep in POST:
                    t = f"C(epoch, Treatment('pre'))[T.{ep}]"
                    if t in f.params:
                        mods.append({"set": st, "spout": sp, "pair": a, "model": f"neural ~ epoch{cov} + (1|animal)",
                                     "term": ep, "coef": f.params[t], "se": f.bse[t], "p": f.pvalues[t], "n": len(h)})
            except Exception as ex:  # noqa: BLE001
                mods.append({"set": st, "spout": sp, "pair": a, "model": f"epoch{cov}: FAILED {ex}"})
        trk_g = g
        trk_g.to_csv(il.OUT_W / f"balimg_deltas_{st}_{sp}_{a}.csv", index=False, float_format="%.6g") \
            if (st, sp) == CELLS[0] and a in ("tjM1", "orofacial") else None
    T, Mo = pd.DataFrame(trk), pd.DataFrame(mods)
    T.to_csv(il.OUT_W / "balimg_track.csv", index=False, float_format="%.6g")
    Mo.to_csv(il.OUT_W / "balimg_models.csv", index=False, float_format="%.6g")
    for st, sp in CELLS:
        print(f"\n== per-animal r(session d_neural, d_x) {st} {sp}")
        print(T[(T.set == st) & (T.spout == sp)].pivot_table(index=["pair", "x"], columns="animal", values="r").round(2).to_string())
        print(f"-- mixed models {st} {sp} (sessions as observations: anti-conservative)")
        print(Mo[(Mo.set == st) & (Mo.spout == sp)].drop(columns=["set", "spout"]).round(4).to_string(index=False))
    figure(D, C)


def figure(D: pd.DataFrame, C: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    st, sp = CELLS[0]
    mk = dict(zip(AN, ["o", "s", "^", "D"]))
    ec = {"pre": "0.55", "acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4"}
    cols = [("f0_470", "F0 470 log2(R/L)"), ("f0_415", "F0 415 log2(R/L)"), ("h415", "415 balance (c-i)/pre resp"),
            ("neural", "neural balance log2(c/i)"), ("clean", "neural, clean pixels"), ("dF", "neural + F0 ratio (dF)")]
    fig, axs = plt.subplots(3, 6, figsize=(21, 11))
    for r, a in enumerate(("tjM1", "orofacial")):
        for c, (v, lab) in enumerate(cols):
            ax = axs[r, c]
            t = C[(C.set == st) & (C.spout == sp) & (C.pair == a) & (C.measure == v)].set_index("epoch").reindex(POST)
            x = np.arange(len(POST))
            ax.bar(x, t.delta, color=[ec[e] for e in POST], alpha=0.35, width=0.6)
            ax.errorbar(x, t.delta, yerr=[t.delta - t.lo, t.hi - t.delta], fmt="none", ecolor="k", capsize=3)
            for j, an in enumerate(AN):
                ax.scatter(x + (j - 1.5) * 0.08, t.get(f"d_{an}", pd.Series(np.nan, index=POST)), marker=mk[an],
                           facecolor="none", edgecolor="k", s=28, zorder=3, label=an if (r, c) == (0, 0) else None)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xticks(x, POST)
            ax.set_title(f"{a}: {lab}", fontsize=9)
            if c == 0:
                ax.set_ylabel("change vs pre\n(bar = boot_delta, CI; symbols = animals)")
    axs[0, 0].legend(fontsize=7, frameon=False)
    # scatter: per-session change of the neural balance vs the imaging changes
    for c, (a, x, xl) in enumerate([("tjM1", "f0_470", "d F0 470 log2(R/L)"), ("tjM1", "f0_415", "d F0 415 log2(R/L)"),
                                    ("tjM1", "h415", "d 415 balance"), ("orofacial", "f0_470", "d F0 470 log2(R/L)"),
                                    ("orofacial", "f0_415", "d F0 415 log2(R/L)"), ("orofacial", "h415", "d 415 balance")]):
        ax = axs[2, c]
        g = D[(D.set == st) & (D.spout == sp) & (D.pair == a)].copy()
        for v in ("neural", x):
            g[f"d_{v}"] = _dpre(g, v)
        for an in AN:
            for ep in EP:
                h = g[(g.animal == an) & (g.epoch == ep)]
                ax.scatter(h[f"d_{x}"], h.d_neural, marker=mk[an], color=ec[ep], s=22, alpha=0.8,
                           label=ep if (c == 0 and an == AN[0]) else None)
        if x.startswith("f0"):
            lim = np.nanmax(np.abs(g[f"d_{x}"])) * 1.1
            xx = np.linspace(-lim, lim, 3)
            ax.plot(xx, -xx, color="k", ls=":", lw=1, label="pure F0 artefact" if c == 0 else None)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel(xl)
        ax.set_title(f"{a}: per session, change from animal's pre mean", fontsize=9)
        if c == 0:
            ax.set_ylabel("d neural balance log2(c/i)")
    axs[2, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Is the post-stroke hemispheric balance shift an imaging artefact? ({st} first licks, {sp} targets, "
                 f"AUC 0..1.0 s, baseline -1.0..-0.5 s; right = contralesional)\nF0 = mean raw fluorescence per ROI; 415 = "
                 "isosbestic lick response on the same licks; clean = interior non-vessel pixels; boot_delta animals>sessions",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = il.OUT_W / "figures" / "balimg_control.png"
    fig.savefig(out, dpi=130)
    print(f"== wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args(argv)
    if not a.summarise or not SESS.exists():
        extract(min(a.jobs, 8))
    summarise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
