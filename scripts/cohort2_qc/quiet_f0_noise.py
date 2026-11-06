"""T22: per-area baseline fluorescence F0 (tonic proxy) + the noise floor under T20's fast band (cohort1, 2026-11-01).

(1) TONIC PROXY. Priya's hypothesis (quiet-events handoff section 7): the VLS stroke reduces thalamocortical drive to
IPSILESIONAL orofacial cortex. dF/F cannot see a tonic level; F0 can, weakly. Per session: F0 = the pipeline's per-pixel
temporal mean image (`frames_average`, raw counts, as scripts/iti_f0_check.py) on the atlas grid
(`allen_aligned_affine8v1/frames_average_atlas.npy` = the same mean image under the session's registration), both
channels; the 470 slot is `hemo_variants.functional_channel(label)` (0 for PS92_0828). Areas = the quiet analyses' masks
(brain mask & beta_maps.brain_mask, eroded stat mask, `iti_licks.roi_masks`: Allen areas per hemisphere + ALM / tjM1 =
anterolateral 40 % of MOs / MOp). Groups as quiet_area_spikes: orofacial = ALM + tjM1 + SSp-m + SSp-n (pixel union),
limb = SSp-ul + SSp-ll, cortex = stat mask. _left = ipsilesional (image left). Measures: log10 F470, log10 F415, log10
(F470 / F415) (crude haemodynamics / expression normalisation: 415 is isosbestic, so the ratio cancels expression and
common light losses, but more blood lowers 415 more than 470 and RAISES the ratio). Contrast: ipsi - contra (log10),
change vs the animal's August pre.

(2) NOISE FLOOR UNDER T20. T20's synchronisation index = log10(PSD 1-4 / 8-15 Hz) of the quiet engaged trace
(quiet_fluct spectra, v2_quiet/fluct/sessions). Floor = mean PSD 12-15 Hz per trace (white-noise estimate; fs 31.25 Hz).
Shot noise predicts floor ~ 1 / F0 (PSD of dF/F), i.e. slope -1 of log floor on log F0. T20 contrast (orofacial ipsi -
contra) recomputed as: A original; B floor-subtracted log10((P1-4 - fl) / (P8-15 - fl)); C fast band 4-8 Hz
log10(P1-4 / P4-8); D floor-subtracted 4-8. Diagnostic: the 7-11 Hz peak in the corrected trace is also in the 415
channel (T10 spectra, v2_quiet/hemo415/sessions psd_raw415) -- a non-calcium (cardiac / brain-motion) component -- so
E = ipsi - contra log10 415 power 7-11 Hz tests whether T20's ipsilesional fast-power drop is in 415 too.

Per-animal tables first (CLAUDE.md rule 8); `mixed_models.lmm_table` epoch model vs August pre, FDR within measure.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_f0_noise.py [--jobs 4]
Output: M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes/v2_quiet/f0_noise/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import GROUPS as AREA_GROUPS  # noqa: E402
from quiet_fluct import FL, TRACE_COLS  # noqa: E402
from wfield_local import config  # noqa: E402

V2Q = FL.parent
OUT = V2Q / "f0_noise"
H415 = V2Q / "hemo415" / "sessions"
EP = ["pre", "acute", "subacute", "chronic"]
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSp-ul", "SSp-ll"]
B_LO, B_HI, B_FAST, B_FLOOR, B_CARD = (1.0, 4.0), (8.0, 15.0), (4.0, 8.0), (12.0, 15.0), (7.0, 11.0)


def labels():
    from wfield_local import epochs
    out = []
    for f in sorted((FL / "sessions").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or (ep == "pre" and config.early_pre(f.stem)) or f.stem[:4] not in ("PS92", "PS93", "PS94", "PS95"):
            continue
        out.append(f.stem)
    return out


def f0_session(label):
    """One session -> {column: (F470, F415, npix)} on the atlas grid, quiet-analysis masks. Module-level for spawn."""
    from iti_licks import roi_masks
    from wfield_local import beta_maps, config
    from wfield_local.hemo_variants import functional_channel
    s = next(x for x in config.load_sessions() if x["label"] == label)
    at = Path(s["mc"]) / "wfield_local_results" / "allen_aligned_affine8v1"
    fa = np.load(at / "frames_average_atlas.npy").astype(np.float64)                  # (2, H, W)
    mask = np.load(at / "allen_brain_mask_native_grid.npy").astype(bool)               # as combined_encoding._load_imaging
    area = np.load(at / "allen_area_atlas_native_grid.npy")
    names = {int(c): n for c, n in json.load(open(at / "allen_area_names.json"))}
    bm, sm = beta_maps.brain_mask(), beta_maps.stat_mask()
    if bm is not None and bm.shape == mask.shape:
        mask = mask & bm
    flat = np.flatnonzero(mask.ravel())
    pstat = sm.ravel()[flat] if sm is not None and sm.shape == mask.shape else np.ones(len(flat), bool)
    m = roi_masks(area.ravel()[flat], pstat, names, flat, mask.shape)
    fc = functional_channel(label)
    b, v = fa[fc].ravel()[flat], fa[1 - fc].ravel()[flat]
    out = {}
    cols = [f"{a}_{h}" for a in AREAS for h in ("left", "right")]
    cols += [f"{g}_{h}" for g in AREA_GROUPS for h in ("left", "right")] + list(AREA_GROUPS) + ["cortex"]
    for c in cols:
        if c == "cortex":
            mk = m["cortex"]
        elif c in AREA_GROUPS:
            mk = np.any([m[f"{a}_{h}"] for a in AREA_GROUPS[c] for h in ("left", "right") if f"{a}_{h}" in m], 0)
        elif c.rsplit("_", 1)[0] in AREA_GROUPS:
            g, h = c.rsplit("_", 1)
            mk = np.any([m[f"{a}_{h}"] for a in AREA_GROUPS[g] if f"{a}_{h}" in m], 0)
        else:
            mk = m.get(c, np.zeros(len(flat), bool))
        ok = mk & np.isfinite(b) & np.isfinite(v) & (b > 0) & (v > 0)
        out[c] = (float(b[ok].mean()), float(v[ok].mean()), int(ok.sum())) if ok.any() else (np.nan, np.nan, 0)
    return out


def bandmean(fr, P, band):
    return np.nanmean(P[..., (fr >= band[0]) & (fr < band[1])], -1)


def per_animal(S, col, ep=EP[1:]):
    return S[S.epoch != "pre"].pivot_table(index="animal", columns="epoch", values=col, aggfunc="mean").reindex(columns=ep)


def delta_vs_pre(S, cols, by=("animal",)):
    pre = S[S.epoch == "pre"].groupby(list(by))[cols].mean()
    for c in cols:
        S[f"d_{c}"] = S[c].to_numpy() - pre[c].reindex(pd.MultiIndex.from_frame(S[list(by)]) if len(by) > 1
                                                       else S[by[0]]).to_numpy()
    return S


def lmm(W, cols, measure):
    from wfield_local import mixed_models as mm
    L = pd.concat([W[["label"]].assign(measure=measure, column=c, y=W[c].to_numpy()) for c in cols])
    L = mm.add_time_covariates(L, august_pre=True)
    return mm.lmm_table(L.dropna(subset=["y"]), ["measure", "column"], models=("epoch",), print_animals=True,
                        fdr_by=("measure",))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230)
    labs = labels()
    print(f"{len(labs)} cohort1 sessions (August pre + post)")
    # ---------------------------------------------------------------------------------------------- (1) F0
    fcsv = OUT / "f0_sessions.csv"
    if fcsv.exists():
        F = pd.read_csv(fcsv)
    else:
        res, fail = ak.fan_sessions(labs, f0_session, jobs=args.jobs)
        if fail:
            print("FAILED:", fail)
        F = pd.DataFrame([dict(label=k, column=c, f470=v[0], f415=v[1], npix=v[2]) for k, d in res for c, v in d.items()])
        F.to_csv(fcsv, index=False)
    F["animal"] = F.label.str[:4]
    F["epoch"] = F.label.map(epochs.epoch_of)
    F["l470"], F["l415"] = np.log10(F.f470), np.log10(F.f415)
    F["lratio"] = F.l470 - F.l415
    W = F.pivot_table(index=["label", "animal", "epoch"], columns="column", values=["l470", "l415", "lratio"])
    C = pd.DataFrame(index=W.index)
    for m in ("l470", "l415", "lratio"):
        for g in ("orofacial", "limb"):
            C[f"{m}|{g}_ipsi"] = W[(m, f"{g}_left")]
            C[f"{m}|{g}_contra"] = W[(m, f"{g}_right")]
            C[f"{m}|{g}_ipsi_minus_contra"] = W[(m, f"{g}_left")] - W[(m, f"{g}_right")]
        for a in AREAS:
            C[f"{m}|{a}_ipsi_minus_contra"] = W[(m, f"{a}_left")] - W[(m, f"{a}_right")]
        C[f"{m}|cortex"] = W[(m, "cortex")]
        C[f"{m}|orofacial_ipsi_minus_cortex"] = W[(m, "orofacial_left")] - W[(m, "cortex")]
    C = C.reset_index()
    C = C[C.epoch.notna()].reset_index(drop=True)
    cc = [c for c in C.columns if "|" in c]
    C = delta_vs_pre(C, cc)
    C.to_csv(OUT / "f0_contrasts_sessions.csv", index=False)
    print("\n==== (1) F0: change vs August pre (log10 units; 0.01 = 2.3 %), animal x epoch mean (rule 8)")
    show = [f"{m}|{x}" for m in ("l470", "l415", "lratio") for x in
            ("orofacial_ipsi", "orofacial_contra", "orofacial_ipsi_minus_contra", "limb_ipsi", "limb_contra",
             "limb_ipsi_minus_contra", "cortex")]
    for c in show:
        print(f"\n-- d {c}\n" + per_animal(C, f"d_{c}").round(4).to_string())
    print("\n-- pre-stroke LEVEL of ipsi - contra (log10), per animal")
    print(C[C.epoch == "pre"].groupby("animal")[[f"{m}|orofacial_ipsi_minus_contra" for m in ("l470", "l415", "lratio")]]
          .mean().round(4).to_string())
    T_f0 = []
    for m in ("l470", "l415", "lratio"):
        cols = [f"{m}|{x}" for x in ("orofacial_ipsi_minus_contra", "limb_ipsi_minus_contra", "orofacial_ipsi",
                                     "orofacial_contra", "limb_ipsi", "limb_contra", "cortex",
                                     *[f"{a}_ipsi_minus_contra" for a in AREAS[:4]])]
        T_f0.append(lmm(C, cols, f"F0_{m}"))
    T_f0 = pd.concat(T_f0)
    # ---------------------------------------------------------------------------------------------- (2) noise floor
    rows = []
    PSD = {}
    for k in labs:
        z = np.load(FL / "sessions" / f"{k}.npz")
        fr, P = z["freqs"], z["psd"]
        PSD[k] = P
        h = None
        if (H415 / f"{k}.npz").exists():
            zh = np.load(H415 / f"{k}.npz", allow_pickle=True)
            h = {str(r): i for i, r in enumerate(zh["rois"])}, zh["psd_raw415"], zh["psd_raw470"]
        for j, c in enumerate(TRACE_COLS):
            p = P[j]
            lo, hi, fa, fl = (float(bandmean(fr, p, b)) for b in (B_LO, B_HI, B_FAST, B_FLOOR))
            cb = (fr >= 6.0) & (fr < 12.0)
            r = dict(label=k, animal=k[:4], epoch=epochs.epoch_of(k), column=c, p_lo=lo, p_hi=hi, p_fast=fa, floor=fl,
                     card_corr=float(bandmean(fr, p, B_CARD)), peak_hz=float(fr[cb][np.argmax(p[cb])]))
            if h is not None and c in h[0]:
                r["card415"] = float(bandmean(fr, h[1][h[0][c]], B_CARD))
                r["floor415"] = float(bandmean(fr, h[1][h[0][c]], B_FLOOR))
                r["floor470raw"] = float(bandmean(fr, h[2][h[0][c]], B_FLOOR))
            rows.append(r)
    N = pd.DataFrame(rows)
    sub = lambda a, b: np.where((a > 0) & (b > 0), np.log10(np.clip(a, 1e-12, None) / np.clip(b, 1e-12, None)), np.nan)  # noqa: E731
    N["sync_A"] = np.log10(N.p_lo / N.p_hi)
    N["sync_B"] = sub(N.p_lo - N.floor, N.p_hi - N.floor)
    N["sync_C"] = np.log10(N.p_lo / N.p_fast)
    N["sync_D"] = sub(N.p_lo - N.floor, N.p_fast - N.floor)
    N["lfloor"] = np.log10(N.floor)
    N["lcard415"] = np.log10(N.card415)
    N["lp_hi"] = np.log10(N.p_hi)
    N["lp_hiex"] = np.log10((N.p_hi - N.floor).clip(lower=1e-12))
    N["lp_fast"] = np.log10(N.p_fast)
    # attach the trace's own F0 (orofacial_left/right, limb_left/right, orofacial, limb, cortex)
    fmap = F.set_index(["label", "column"])
    N["l470"] = [fmap.l470.get((a, b), np.nan) for a, b in zip(N.label, N.column)]
    N["l415"] = [fmap.l415.get((a, b), np.nan) for a, b in zip(N.label, N.column)]
    N.to_csv(OUT / "noise_sessions.csv", index=False)
    print("\n==== (2) noise floor (mean PSD 12-15 Hz) vs F0, orofacial + limb hemisphere traces")
    print("negative-excess sessions (P8-15 <= floor or P1-4 <= floor):",
          int(N.sync_B.isna().sum()), "of", len(N), "| (P4-8 <= floor):", int(N.sync_D.isna().sum()))
    H = N[N.column.isin(["orofacial_left", "orofacial_right", "limb_left", "limb_right"])].copy()
    Hc = H.copy()
    for c in ("lfloor", "l470", "l415"):
        Hc[c] = Hc[c] - Hc.groupby(["animal", "column"])[c].transform("mean")
    fit = {}
    for an, g in Hc.groupby("animal"):
        fit[an] = np.polyfit(g.l470, g.lfloor, 1)[0]
    fit["pooled"] = np.polyfit(Hc.l470, Hc.lfloor, 1)[0]
    print("slope of log10 floor on log10 F0_470 within animal x hemisphere-ROI (shot noise predicts -1):",
          {k: round(v, 2) for k, v in fit.items()})
    print("r(within):", round(np.corrcoef(Hc.l470, Hc.lfloor)[0, 1], 3))
    pre = H[H.epoch == "pre"]
    print("pre-stroke floors (x1e-3, corr470 / raw470 / raw415) and F0 by hemisphere, animal means:")
    print((pre.groupby(["animal", "column"])[["floor", "floor470raw", "floor415"]].mean() * 1e3).round(3)
          .join(pre.groupby(["animal", "column"])[["l470", "l415"]].mean().round(3)).to_string())
    print("\n7-11 Hz peak frequency (corrected trace, orofacial bilateral), animal x epoch mean:")
    print(N[N.column == "orofacial"].pivot_table(index="animal", columns="epoch", values="peak_hz").reindex(columns=EP)
          .round(2).to_string())
    # T20 ipsi - contra contrasts, per variant
    V = N.pivot_table(index=["label", "animal", "epoch"], columns="column",
                      values=["sync_A", "sync_B", "sync_C", "sync_D", "lfloor", "lcard415", "lp_hi", "lp_hiex",
                              "lp_fast", "l470"])
    K = pd.DataFrame(index=V.index)
    for m in ("sync_A", "sync_B", "sync_C", "sync_D", "lfloor", "lcard415", "lp_hi", "lp_hiex", "lp_fast", "l470"):
        for g in ("orofacial", "limb"):
            K[f"{m}|{g}_ipsi_minus_contra"] = V[(m, f"{g}_left")] - V[(m, f"{g}_right")]
    # floor predicted from F0 (shot noise, slope -1): residual ipsi - contra floor after removing -1 x dF0
    for g in ("orofacial", "limb"):
        K[f"lfloor_resid_F0|{g}_ipsi_minus_contra"] = K[f"lfloor|{g}_ipsi_minus_contra"] + K[f"l470|{g}_ipsi_minus_contra"]
    K = K.reset_index()
    kc = [c for c in K.columns if "|" in c]
    K = delta_vs_pre(K, kc)
    K.to_csv(OUT / "t20_contrasts_sessions.csv", index=False)
    print("\n==== T20 orofacial ipsi - contra, change vs pre, animal x epoch mean (rule 8)")
    for c in kc:
        if c.endswith("orofacial_ipsi_minus_contra"):
            print(f"\n-- d {c}\n" + per_animal(K, f"d_{c}").round(3).to_string())
    print("\n-- pre-stroke LEVELS orofacial ipsi - contra:")
    print(K[K.epoch == "pre"].groupby("animal")[[c for c in kc if c.endswith("orofacial_ipsi_minus_contra")]]
          .mean().round(3).to_string())
    T_n = pd.concat([lmm(K, [c for c in kc if c.startswith(m + "|")], m)
                     for m in ("sync_A", "sync_B", "sync_C", "sync_D", "lfloor", "lfloor_resid_F0", "lcard415",
                               "lp_hi", "lp_hiex", "lp_fast")])
    T = pd.concat([T_f0, T_n])
    T.to_csv(OUT / "lmm_f0_noise.csv", index=False)
    print("\n==== LMM (epoch vs August pre), FDR within measure")
    print(T[T.term != "pre_level"][["measure", "column", "term", "coef", "se", "p", "q", "n"]].round(4).to_string())
    figure(C, K, N, H, Hc, fit, T, PSD, fr)
    (OUT / "README.txt").write_text(__doc__)


def _strip(a, S, col, T, measure, column, title, ylab=None):
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    animals = sorted(S.animal.unique())
    G = S[S.epoch != "pre"]
    for k, ep in enumerate(EP[1:]):
        g = G[G.epoch == ep]
        for an, x in g.groupby("animal"):
            a.plot(np.full(len(x), k + (animals.index(an) - 1.5) * 0.09), x[col], ANIMAL_MARKER[an], mfc="none",
                   mec=EPOCH_LINE[ep], ms=4, alpha=0.75, ls="none")
        m = g.groupby("animal")[col].mean()
        a.bar(k, m.mean(), width=0.55, color=EPOCH_LINE[ep], alpha=0.3)
        t = T[(T.measure == measure) & (T.column == column) & T.term.str.contains(ep)]
        if len(t) and np.isfinite(t.q.iloc[0]) and float(t.q.iloc[0]) < 0.05:
            q = float(t.q.iloc[0])
            sig_marks.star_top(a, k, "***" if q < 0.001 else "**" if q < 0.01 else "*")
    a.axhline(0, color="0.5", lw=0.8)
    a.set_xticks(range(3))
    a.set_xticklabels(EP[1:], fontsize=8)
    a.set_title(title, fontsize=8.5)
    if ylab:
        a.set_ylabel(ylab, fontsize=8.5)


def figure(C, K, N, H, Hc, fit, T, PSD, fr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    fig = plt.figure(figsize=(21, 13.5))
    gs = fig.add_gridspec(3, 6, hspace=0.62, wspace=0.42)
    # row A: F0
    rowA = [("l470", "orofacial_ipsi_minus_contra", "F470 orofacial\nipsi - contra"),
            ("l415", "orofacial_ipsi_minus_contra", "F415 orofacial\nipsi - contra"),
            ("lratio", "orofacial_ipsi_minus_contra", "F470/F415 orofacial\nipsi - contra"),
            ("lratio", "limb_ipsi_minus_contra", "F470/F415 limb\nipsi - contra"),
            ("l470", "cortex", "F470 whole cortex"),
            ("lratio", "cortex", "F470/F415 whole cortex")]
    for i, (m, c, t) in enumerate(rowA):
        a = fig.add_subplot(gs[0, i])
        _strip(a, C, f"d_{m}|{c}", T, f"F0_{m}", f"{m}|{c}", t, "change vs pre (log10)" if i == 0 else None)
    # row B: T20 contrast variants
    rowB = [("sync_A", "A  T20 index (1-4 / 8-15 Hz)\nipsi - contra, as published"),
            ("sync_B", "B  floor-subtracted\n(1-4 - fl) / (8-15 - fl)"),
            ("sync_C", "C  fast band 4-8 Hz\n1-4 / 4-8"),
            ("sync_D", "D  floor-subtracted 4-8\n(1-4 - fl) / (4-8 - fl)"),
            ("lp_hiex", "8-15 Hz power above floor\nipsi - contra"),
            ("lcard415", "415 nm 7-11 Hz power\nipsi - contra (non-calcium)")]
    for i, (m, t) in enumerate(rowB):
        a = fig.add_subplot(gs[1, i])
        c = f"{m}|orofacial_ipsi_minus_contra"
        _strip(a, K, f"d_{c}", T, m, c, t, "orofacial ipsi - contra,\nchange vs pre (log10)" if i == 0 else None)
    # row C: diagnostics
    a = fig.add_subplot(gs[2, 0:2])
    for an, g in Hc.groupby("animal"):
        for ep, x in g.groupby("epoch"):
            a.plot(x.l470, x.lfloor, ANIMAL_MARKER[an], mfc="none", mec=EPOCH_LINE[ep], ms=4, alpha=0.6, ls="none")
    xx = np.array([Hc.l470.min(), Hc.l470.max()])
    a.plot(xx, -xx, color="0.4", ls="--", lw=1)
    a.plot(xx, fit["pooled"] * xx, color="k", lw=1.2)
    a.set_xlabel("log10 F0 (470), centred within animal x hemisphere ROI", fontsize=8.5)
    a.set_ylabel("log10 noise floor (PSD 12-15 Hz), centred", fontsize=8.5)
    a.set_title(f"Noise floor vs F0 (orofacial + limb hemisphere traces): slope {fit['pooled']:.2f} "
                f"(dashed = -1, pure shot noise)", fontsize=8.5)
    a = fig.add_subplot(gs[2, 2:4])
    for ep in ("pre", "acute"):
        for c, ls in (("orofacial_left", "-"), ("orofacial_right", "--")):
            j = TRACE_COLS.index(c)
            per_an = {}
            for k, P in PSD.items():
                if N.loc[(N.label == k), "epoch"].iloc[0] == ep:
                    per_an.setdefault(k[:4], []).append(P[j])
            med = np.median([np.median(v, 0) for v in per_an.values()], 0)
            a.semilogy(fr[1:], med[1:], color=EPOCH_LINE[ep], ls=ls, lw=1.3)
    a.axvspan(*B_FLOOR, color="0.85", zorder=0)
    a.axvspan(*B_CARD, color="#fde0dd", zorder=0, alpha=0.6)
    a.set_xlabel("Hz", fontsize=8.5)
    a.set_ylabel("PSD ((dF/F %)^2 / Hz)", fontsize=8.5)
    a.set_title("Orofacial quiet PSD, median of animals (grey = floor band 12-15 Hz,\npink = 7-11 Hz peak, also "
                "in 415 nm); solid = ipsi, dashed = contra", fontsize=8.5)
    a = fig.add_subplot(gs[2, 4])
    _strip(a, K, "d_lfloor|orofacial_ipsi_minus_contra", T, "lfloor", "lfloor|orofacial_ipsi_minus_contra",
           "noise floor orofacial\nipsi - contra", "change vs pre (log10)")
    a = fig.add_subplot(gs[2, 5])
    _strip(a, K, "d_lfloor_resid_F0|orofacial_ipsi_minus_contra", T, "lfloor_resid_F0",
           "lfloor_resid_F0|orofacial_ipsi_minus_contra", "floor ipsi - contra after\nshot-noise F0 term (slope -1)")
    hand = [Line2D([], [], marker=ANIMAL_MARKER[a], ls="none", mfc="none", mec="k", label=a) for a in ANIMAL_MARKER]
    hand += [Line2D([], [], color=EPOCH_LINE[e], lw=3, label=e) for e in EP]
    hand += [Line2D([], [], color="k", ls="-", label="ipsi (PSD)"), Line2D([], [], color="k", ls="--", label="contra (PSD)")]
    fig.legend(handles=hand, loc="lower center", ncol=len(hand), fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    sig_marks.reserve_headroom(fig)
    fig.suptitle("T22 (cohort1): row 1 = baseline fluorescence F0 per area (tonic proxy; ipsi = lesioned left hemisphere); "
                 "row 2 = T20 orofacial ipsi - contra synchronisation contrast with noise-floor / fast-band variants; "
                 "row 3 = noise-floor diagnostics.\nSessions (shape = animal, colour = epoch), bar = mean of animals; "
                 "stars = LMM epoch vs August pre, FDR within measure", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.08)
    fig.savefig(OUT / "quiet_f0_noise.png", dpi=105)
    print("wrote", OUT / "quiet_f0_noise.png")


if __name__ == "__main__":
    main()
