"""Is the larger post-stroke ITI-lick response HAEMODYNAMIC, a FILTER effect, MOTION / EDGE / VESSEL, or an F0 artefact?

The claim under test (scripts/iti_licks.py, ramp_auc_sessions.csv / area_auc_sessions.csv): the cortical response to each
self-initiated (bout-start ITI, >= 2 s without licking) lick -- AUC 0..0.5 s after lick ONSET of the per-session mean PETH,
baseline -1.5..-1.0 s, ROI means of the ADOPTED hemo-corrected SVD (hemo_meegkit_hpfit SVTcorr), slow drift (>= 600 s
cosines) removed, dF/F x1e3 -- is ~2x pre acutely and still elevated chronically. Already excluded: F0 fall, bout length,
kinematics, face movement, lick count. Here the SAME licks (the session npz's bout_t / bout_gap), SAME ROIs (`roi_masks`),
SAME drift removal, SAME window and SAME estimator are recomputed in four SIGNAL VARIANTS:

  adopted   hemo_meegkit_hpfit/SVTcorr.npy           -- must reproduce the published numbers (sanity check, (a))
  orig      the ORIGINAL pipeline SVTcorr.npy          -- zero-phase 0.1 Hz high-pass (filter sensitivity, (b))
  raw470    UNCORRECTED functional half of raw SVT.npy (c)
  raw415    the 415 nm half of raw SVT.npy             (d) -- haemodynamics (+ a small NEGATIVE calcium term at 415 nm,
                                                        see hemo_map_control docstring) + motion

RAW SVT LAYOUT. `SVT.npy` is (K, 2T), the two channels INTERLEAVED in time; the functional slot is
`hemo_variants.functional_channel(label)` (1 for the cohort, 0 for PS92_0828), the other slot is 415 -- exactly how
`hemo_variants.compute` builds SVTcorr from it (and the slot is checked against each session's hemo manifest). UNITS:
wfield `approximate_svd(divide_by_average=True)` divides each frame by its OWN channel's session-mean image
(frames_average[c]) before projecting on the shared U, so both halves of SVT are dF/F on the same basis as SVTcorr: no
extra normalisation is needed, and U_atlas projects all four variants identically. Frame time: the 415 frame of a pair is
~16 ms from its 470 frame; both use the pair's frame time `ft` (16 ms against a 500 ms AUC window).

ALL THE ESTIMATORS ARE LINEAR, so per lick the baseline-subtracted response AUC is computed once in COMPONENT space
(a_k = AUC_0..0.5(P_k) - 0.5 * baseline_k) and projected: ROI = a @ Ra (Ra = ROI-mean of U_atlas), pixel map = U_b @ mean(a).

  (e) MOTION / EDGE / VESSEL (adopted and 415 pixel maps): EDGE = brain-mask pixels outside the eroded stat mask (the
      16-px rim `beta_maps.stat_mask` removes), INTERIOR = stat mask. VESSEL = top 5 % (within the interior) of a
      vesselness map = high-passed (sigma 12 px) NEGATED mean 470 image (`frames_average_atlas`, as in
      hemo_map_control.vessel_template but on the functional slot), vs the other 95 %. VAR415 = top 5 % of interior
      pixels by 415 temporal variance (drift removed) vs the other 95 %. Motion / haemodynamic artefacts concentrate at
      edges and on vessels, so a post/pre change that is LARGER there than in the interior / non-vessel pixels would flag one.
  (f) F0 per Allen area: mean of `frames_average_atlas` (raw counts, both channels) over each area's stat-mask pixels,
      plus the whole-frame central-60 % mean of iti_f0_check; change vs pre per area.
  (g) global state: treadmill speed (DAQ, `behavior_events.session_speed`) in -1..+1 s around each lick and in the
      baseline window; FaceRhythm |factor| in the same windows (the npz's bout_fr).
  (h) baseline / gap match: each post session's licks re-weighted so its no-lick GAP distribution (bins 2-3, 3-5, 5-10,
      10-20, >= 20 s) matches its animal's pooled pre distribution; response recomputed.

Epoch comparisons: per-animal table first (CLAUDE.md rule 8), then `analysis_kit.boot_delta` (paired animals > sessions).
Normalisation for comparing variants: every variant's response is divided by the animal's PRE ADOPTED response (same ROI),
so "415 change" is read in units of the corrected pre response (415's own pre response is near zero / sign-variable and a
ratio to it is meaningless).
Usage: python scripts/iti_hemo_check.py [--labels ...] [--jobs 3]     (extract, cached, then summarise)
       python scripts/iti_hemo_check.py --summarise
Output: <iti_licks OUT>/hemo_check/*.csv, cache/<label>.npz ; figures <OUT>/figures/iti_hemo_check_*.png
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
from fr_musall import AN  # noqa: E402

HC = il.OUT / "hemo_check"
CACHE = HC / "cache"
CACHE_VERSION = 1
VARIANTS = ("adopted", "orig", "raw470", "raw415")
VLAB = {"adopted": "corrected ADOPTED (meegkit_hpfit)", "orig": "corrected ORIGINAL (zero-phase HP)",
        "raw470": "UNCORRECTED 470 nm", "raw415": "415 nm (haemodynamic)"}
MAP_VARIANTS = ("adopted", "raw415")
EP = il.EP4
POST = EP[1:]
TOP_FRAC = 0.05
VESSEL_SIGMA = 12.0
RUN_THRESH = 3.0                    # mm/s, the cohort running threshold (configs segmentation; sedentary mice)
GAP_BINS = np.array([2.0, 3.0, 5.0, 10.0, 20.0, np.inf])
SHOW_ROIS = ["key_mean", "tjM1_right", "tjM1_left", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "cortex"]
PIXSETS = ("edge", "interior", "vessel", "nonvessel", "var415", "nonvar415")


def _session(label: str) -> dict:
    from wfield_local import config
    return next(x for x in config.load_sessions() if x["label"] == label)


def _comp_auc(X: np.ndarray, ft: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per lick: (baseline-subtracted response AUC 0..0.5 s per column, PETH (n, lags, cols) baseline-subtracted)."""
    P = il._sample(X, ft, t, il.LAGS_P).astype(np.float64)
    P -= P[:, (il.LAGS_P >= il.BASE[0]) & (il.LAGS_P <= il.BASE[1]), :].mean(1, keepdims=True)
    m = (il.LAGS_P >= -1e-9) & (il.LAGS_P <= 0.5 + 1e-9)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trap(P[:, m, :], il.LAGS_P[m], axis=1), P


def work(label: str) -> dict:
    """One session -> CACHE/<label>.npz (module-level for spawn)."""
    from scipy import ndimage

    from wfield_local import combined_encoding as ce
    from wfield_local.hemo_variants import ADOPTED, functional_channel
    f = CACHE / f"{label}.npz"
    if f.exists() and int(np.load(f)["cache_version"]) == CACHE_VERSION:
        return {"label": label, "cached": True}
    d = il._load(label)
    s = _session(label)
    res = Path(s["mc"]) / "wfield_local_results"
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    n = len(ft)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    if cols != list(d["cols"]):
        raise RuntimeError(f"{label}: ROI columns differ from the saved session npz")
    Ra = np.stack([Ub[M[c]].mean(0) for c in cols], 1).astype(np.float64)        # (K, n_roi)
    fc = functional_channel(label)
    man = json.loads((res / f"hemo_{ADOPTED}" / "manifest.json").read_text())
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    if abs(svt.shape[1] // 2 - n) > 2 and abs(svt.shape[1] // 2 - V.shape[0]) > 2:
        raise RuntimeError(f"{label}: raw SVT {svt.shape[1]} samples vs corrected {V.shape[0]} frames")
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    t = np.asarray(d["bout_t"], float)
    keep = d["bout_gap"] >= il.GAP_MAIN
    out = dict(cache_version=CACHE_VERSION, label=label, animal=str(d["animal"]), epoch=str(d["epoch"]), cols=np.array(cols),
               bout_t=t, bout_gap=np.asarray(d["bout_gap"], float), func_slot=fc,
               manifest_func=int(man.get("functional_channel", -1)), n_frames=n, svt_pairs=svt.shape[1] // 2)
    # pixel sets on the brain mask (flat indices on the shared 540 x 640 grid)
    H, W = (int(x) for x in shape)
    edge, interior = ~pstat, pstat.copy()
    fa = np.load(res / "allen_aligned_affine8v1" / "frames_average_atlas.npy").astype(np.float64)   # (2, H, W)
    img = np.nan_to_num(fa[fc])
    vess = (-img - ndimage.gaussian_filter(-img, VESSEL_SIGMA)).ravel()[flat]
    vessel = interior & (vess >= np.quantile(vess[interior], 1 - TOP_FRAC))
    sets = {"edge": edge, "interior": interior, "vessel": vessel, "nonvessel": interior & ~vessel}
    for v in VARIANTS:
        if v == "adopted":
            X = V[:n].astype(np.float64)
        elif v == "orig":
            X = np.load(res / "SVTcorr.npy", mmap_mode="r").T[:n].astype(np.float64)
        else:
            slot = fc if v == "raw470" else (fc + 1) % 2
            X = np.asarray(svt[:, slot::2], np.float64).T[:n]
        if X.shape[0] < n:
            raise RuntimeError(f"{label} {v}: {X.shape[0]} frames < {n}")
        X -= Q @ (Q.T @ X)
        X *= 1e3
        if v == "raw415":                                # 415 temporal variance per brain pixel -> VAR415 set
            C = X.T @ X / len(X)
            var = np.einsum("pk,pk->p", Ub.astype(np.float64) @ C, Ub)
            v415 = interior & (var >= np.quantile(var[interior], 1 - TOP_FRAC))
            sets["var415"], sets["nonvar415"] = v415, interior & ~v415
        a, P = _comp_auc(X, ft, t)                        # (n_lick, K), (n_lick, lags, K)
        out[f"roi_{v}"] = (a @ Ra).astype(np.float32)     # per-lick ROI response AUC
        out[f"peth_{v}"] = (P[keep].mean(0) @ Ra).astype(np.float32) if keep.any() else np.zeros((len(il.LAGS_P), len(cols)), np.float32)
        out[f"comp_{v}"] = a.astype(np.float32)           # per-lick component AUC (pixel sets computed in summary)
        del X, P
    out["Ub_sets"] = np.stack([(Ub[sets[k]].mean(0) if sets[k].any() else np.full(Ub.shape[1], np.nan))
                               for k in ("edge", "interior", "vessel", "nonvessel")], 1).astype(np.float32)
    out["set_frac"] = np.array([sets[k].mean() for k in PIXSETS])
    for k in ("var415", "nonvar415"):
        out[f"Ub_{k}"] = Ub[sets[k]].mean(0).astype(np.float32)
    # pixel maps (mean over the >= 2 s gap licks), adopted + 415, full grid float16
    for v in MAP_VARIANTS:
        mp = np.full(H * W, np.nan, np.float32)
        if keep.any():
            mp[flat] = Ub @ out[f"comp_{v}"][keep].mean(0)
        out[f"map_{v}"] = mp.reshape(H, W).astype(np.float16)
    out["pstat_img"] = np.zeros(H * W, bool)
    out["pstat_img"][flat[pstat]] = True
    out["pstat_img"] = out["pstat_img"].reshape(H, W)
    # (f) F0 per area (raw counts, both channels) + whole-frame central 60 % (iti_f0_check definition)
    area_names = [c for c in cols if c != "cortex" and not c.startswith(("ALM_", "tjM1_"))]
    fl = fa.reshape(2, -1)[:, flat]
    out["f0_area_names"] = np.array(area_names)
    out["f0_area"] = np.array([[np.nanmean(fl[ch][M[c]]) for c in area_names] for ch in (fc, (fc + 1) % 2)])
    fn = np.load(res / "frames_average.npy")
    hh, ww = fn.shape[-2:]
    cc = fn[:, int(hh * 0.2):int(hh * 0.8), int(ww * 0.2):int(ww * 0.8)]
    out["f0_whole"] = np.array([np.nanmean(cc[fc]), np.nanmean(cc[(fc + 1) % 2])])
    out["f0_brain"] = np.array([np.nanmean(fl[fc][pstat]), np.nanmean(fl[(fc + 1) % 2][pstat])])
    # (g) running speed around each lick
    try:
        from wfield_local.behavior_events import session_speed
        rt, rv = session_speed(Path(s["h5"]))
        cs = np.concatenate([[0.0], np.cumsum(rv)])

        def wmean(a_, b_):
            i0, i1 = np.searchsorted(rt, t + a_), np.searchsorted(rt, t + b_)
            return np.where(i1 > i0, (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1), np.nan)
        out["run_win"], out["run_base"] = wmean(-1.0, 1.0), wmean(*il.BASE)
    except Exception as ex:                                # noqa: BLE001
        print(f"  {label}: no running ({ex})", flush=True)
        out["run_win"] = out["run_base"] = np.full(len(t), np.nan)
    HC.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"{label}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(f)
    return {"label": label, "n_bout": int(keep.sum()), "func": fc, "manifest_func": out["manifest_func"]}


# ------------------------------------------------------------------------------------------------ summaries


def _boot(T: pd.DataFrame, val: str, ep: str):
    from wfield_local import analysis_kit as ak
    ref = {a: T[(T.animal == a) & (T.epoch == "pre")][val].dropna().to_numpy() for a in AN}
    post = {a: T[(T.animal == a) & (T.epoch == ep)][val].dropna().to_numpy() for a in AN}
    com = [a for a in AN if len(ref[a]) and len(post[a])]
    if len(com) < 3:
        return None
    return ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))


def _change_table(T: pd.DataFrame, keys: list[str], vals: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in T.groupby(keys, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        for val in vals:
            for ep in POST:
                iv = _boot(g, val, ep)
                if iv is None:
                    continue
                pre = g[g.epoch == "pre"].groupby("animal")[val].mean().mean()
                rows.append({**dict(zip(keys, k)), "value": val, "epoch": ep, "pre": pre, "delta": iv.point,
                             "lo": iv.lo, "hi": iv.hi, "sig": iv.excludes_zero, "n_animals": iv.n_animals})
    return pd.DataFrame(rows, columns=[*keys, "value", "epoch", "pre", "delta", "lo", "hi", "sig", "n_animals"])


def _per_animal(T: pd.DataFrame, idx: list[str], val: str) -> str:
    return (T.groupby([*idx, "animal", "epoch"])[val].mean().unstack("epoch").reindex(columns=EP).round(2).to_string())


def _panel(ax, T: pd.DataFrame, val: str, colors: dict, ref: float | None = 1.0) -> None:
    """Epoch on x, one thin line per animal (animal mean over sessions), black = mean of animals, boot_delta CI text."""
    x = np.arange(len(EP))
    A = T.groupby(["animal", "epoch"])[val].mean().unstack("epoch").reindex(columns=EP)
    for a, r in A.iterrows():
        ax.plot(x, r.to_numpy(float), color="0.6", lw=0.9, marker="o", ms=2.5)
        ax.text(x[-1] + 0.08, r.to_numpy(float)[-1], a[-2:], fontsize=5.5, color="0.5", va="center")
    mu = A.mean(0).to_numpy(float)
    ax.plot(x, mu, color="k", lw=2)
    for i, ep in enumerate(EP):
        ax.scatter(x[i], mu[i], color=colors[ep], s=40, zorder=5, edgecolor="k", lw=0.6)
    if ref is not None:
        ax.axhline(ref, color="0.75", lw=0.7, ls="--")
    for i, ep in enumerate(POST, start=1):
        iv = _boot(T, val, ep)
        if iv is not None:
            ax.text(i, 1.0, f"{iv.point:+.2f}\n[{iv.lo:+.2f},{iv.hi:+.2f}]" + (" *" if iv.excludes_zero else ""),
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.8, color=colors[ep])
    ax.set_xticks(x)
    ax.set_xticklabels(["pre", "acute", "sub", "chronic"], fontsize=7)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.25 * (hi - lo))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def summarise(labels: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    from wfield_local.writeguard import assert_writable
    assert_writable(HC)
    (il.OUT / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    C = {x: dict(np.load(CACHE / f"{x}.npz", allow_pickle=False)) for x in labels if (CACHE / f"{x}.npz").exists()}
    print(f"{len(C)} sessions cached")
    # --- channel-slot checks
    ch = pd.DataFrame([{"label": x, "func_slot": int(c["func_slot"]), "manifest_func": int(c["manifest_func"]),
                        "n_frames": int(c["n_frames"]), "svt_pairs": int(c["svt_pairs"])} for x, c in C.items()])
    ch.to_csv(HC / "channel_slots.csv", index=False)
    bad = ch[ch.func_slot != ch.manifest_func]
    print(f"functional slot: {ch.func_slot.value_counts().to_dict()}; disagreements with hemo manifest: {len(bad)}"
          + (f" {list(bad.label)}" if len(bad) else ""))
    print(f"|raw SVT pairs - corrected frames| max {int((ch.svt_pairs - ch.n_frames).abs().max())}")
    # --- (a)-(d): session x variant x ROI response (>= 2 s gap licks, >= 10 per session)
    rows, peth_rows = [], []
    for x, c in C.items():
        keep = c["bout_gap"] >= il.GAP_MAIN
        if keep.sum() < 10:
            continue
        cols = list(c["cols"])
        for v in VARIANTS:
            r = c[f"roi_{v}"][keep].astype(float).mean(0)
            for j, roi in enumerate(cols):
                rows.append({"label": x, "animal": str(c["animal"]), "epoch": str(c["epoch"]), "variant": v, "roi": roi,
                             "resp": r[j], "n": int(keep.sum())})
            peth_rows.append({"label": x, "animal": str(c["animal"]), "epoch": str(c["epoch"]), "variant": v,
                              "peth": c[f"peth_{v}"][:, [cols.index(k) for k in il.KEY_ROIS]].astype(float).mean(1)})
    R = pd.DataFrame(rows)
    key = R[R.roi.isin(il.KEY_ROIS)].groupby(["label", "animal", "epoch", "variant"]).resp.mean().reset_index()
    R = pd.concat([R, key.assign(roi="key_mean", n=np.nan)], ignore_index=True)
    # (a) reproduction against the published tables
    pub = pd.read_csv(il.OUT / "ramp_auc_sessions.csv")
    m = R[R.variant == "adopted"].merge(pub[["label", "roi", "resp_auc"]], on=["label", "roi"])
    pa = pd.read_csv(il.OUT / "area_auc_sessions.csv")
    m2 = R[R.variant == "adopted"].merge(pa[["label", "area", "resp"]].rename(columns={"area": "roi", "resp": "pub"}),
                                         on=["label", "roi"])
    rep = {"n_key": len(m), "max_abs_diff_key": float((m.resp - m.resp_auc).abs().max()),
           "median_rel_diff_key": float(((m.resp - m.resp_auc).abs() / m.resp_auc.abs().clip(lower=1e-6)).median()),
           "r_key": float(np.corrcoef(m.resp, m.resp_auc)[0, 1]),
           "n_area": len(m2), "max_abs_diff_area": float((m2.resp - m2.pub).abs().max()),
           "r_area": float(np.corrcoef(m2.resp, m2.pub)[0, 1]),
           "sd_published_key": float(m.resp_auc.std())}
    json.dump(rep, open(HC / "reproduction.json", "w"), indent=1)
    print("\n== (a) REPRODUCTION of ramp_auc_sessions / area_auc_sessions by the adopted variant\n", json.dumps(rep, indent=1))
    # normalise every variant by the animal's PRE ADOPTED response (same ROI) and by its own pre
    den = R[(R.variant == "adopted") & (R.epoch == "pre")].groupby(["roi", "animal"]).resp.mean()
    R["resp_rel_adopted"] = R.resp / np.array([den.get((r_, a_), np.nan) for r_, a_ in zip(R.roi, R.animal)])
    own = R[R.epoch == "pre"].groupby(["variant", "roi", "animal"]).resp.mean()
    R["resp_rel_own"] = R.resp / np.array([own.get((v_, r_, a_), np.nan) for v_, r_, a_ in zip(R.variant, R.roi, R.animal)])
    R.to_csv(HC / "variant_resp_sessions.csv", index=False)
    S = R[R.roi.isin(SHOW_ROIS)]
    print("\n== per animal (rule 8): response / animal's PRE ADOPTED response, by variant")
    print(_per_animal(S[S.roi.isin(["key_mean", "tjM1_right", "SSp-m_right", "cortex"])], ["roi", "variant"], "resp_rel_adopted"))
    print("\n== per animal (rule 8): raw response AUC (dF/F x1e3 s), key_mean and cortex")
    print(_per_animal(S[S.roi.isin(["key_mean", "cortex"])], ["roi", "variant"], "resp"))
    CH = _change_table(S, ["variant", "roi"], ["resp", "resp_rel_adopted", "resp_rel_own"])
    CH.to_csv(HC / "variant_resp_change.csv", index=False)
    print("\n== change vs pre (boot_delta), resp_rel_adopted\n"
          + CH[CH.value == "resp_rel_adopted"].pivot_table(index=["roi", "variant"], columns="epoch",
                                                            values=["delta", "lo", "hi"]).round(2).to_string())
    # cortex-wide: per Allen area, change in raw AUC per variant; count of significant areas
    areas = [r_ for r_ in R.roi.unique() if r_ not in ("cortex", "key_mean")]      # = iti_area_maps set
    AC = _change_table(R[R.roi.isin(areas)], ["variant", "roi"], ["resp"])
    AC.to_csv(HC / "variant_area_change.csv", index=False)
    cnt = AC.assign(up=AC.sig & (AC.delta > 0), down=AC.sig & (AC.delta < 0)).groupby(["variant", "epoch"])[["up", "down"]].sum()
    cnt["n_areas"] = AC.groupby(["variant", "epoch"]).size()
    cnt.to_csv(HC / "variant_area_counts.csv")
    print("\n== cortex-wide: Allen areas with a significant response change (raw AUC)\n" + cnt.to_string())
    # --- figure 1: variants x ROIs
    fig, axs = plt.subplots(len(VARIANTS), len(SHOW_ROIS), figsize=(2.5 * len(SHOW_ROIS), 2.4 * len(VARIANTS)))
    for i, v in enumerate(VARIANTS):
        for j, roi in enumerate(SHOW_ROIS):
            ax = axs[i, j]
            _panel(ax, S[(S.variant == v) & (S.roi == roi)], "resp_rel_adopted", colors)
            ax.axhline(0, color="0.85", lw=0.6)
            if i == 0:
                ax.set_title(roi, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{VLAB[v]}\n/ pre ADOPTED resp.", fontsize=7)
    fig.suptitle("NEURAL, NO DECODER: self-initiated (ITI) lick response (AUC 0..0.5 s after onset, baseline -1.5..-1.0 s, "
                 ">= 2 s no-lick gap) in four signal variants -- same licks, ROIs, drift removal and window. Every variant is "
                 "divided by the animal's PRE response in the ADOPTED corrected signal (1 = pre corrected response). Grey = "
                 "animals, black = mean of animals; text = change vs pre, paired nested bootstrap (animals > sessions) 95% CI, "
                 "* excludes 0. key_mean = mean of ALM, tjM1, SSp-m (both sides) and cortex, as iti_f0_check.",
                 fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(il.OUT / "figures" / "iti_hemo_check_variants.png", dpi=110)
    plt.close(fig)
    # --- figure 2: key_mean PETH per variant, mean of animals
    PE = pd.DataFrame(peth_rows)
    fig, axs = plt.subplots(1, len(VARIANTS), figsize=(4 * len(VARIANTS), 3.3), sharey=True)
    for i, v in enumerate(VARIANTS):
        ax = axs[i]
        for ep in EP:
            t = PE[(PE.variant == v) & (PE.epoch == ep)]
            if t.empty:
                continue
            A = np.stack([np.mean(np.stack(g.peth.to_numpy()), 0) for _a, g in t.groupby("animal")])
            ax.plot(il.LAGS_P, A.mean(0), color=colors[ep], lw=2, label=f"{ep} ({len(A)} animals)")
        ax.axvline(0, color="0.6", lw=0.6)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_title(VLAB[v], fontsize=9)
        ax.set_xlabel("time from lick onset (s)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("dF/F x1e3 (baseline-subtracted)\nkey_mean ROI")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("NEURAL, NO DECODER: ITI-lick PETH (key ROIs averaged) per signal variant; mean of animals (each animal = mean "
                 "of its sessions). Same y scale across variants.", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(il.OUT / "figures" / "iti_hemo_check_peth.png", dpi=110)
    plt.close(fig)
    # --- (e) pixel sets
    prow = []
    for x, c in C.items():
        keep = c["bout_gap"] >= il.GAP_MAIN
        if keep.sum() < 10:
            continue
        Ubs = np.column_stack([c["Ub_sets"], c["Ub_var415"], c["Ub_nonvar415"]])     # (K, 6) in PIXSETS order
        for v in VARIANTS:
            val = c[f"comp_{v}"][keep].astype(float).mean(0) @ Ubs
            for k, s_ in enumerate(PIXSETS):
                prow.append({"label": x, "animal": str(c["animal"]), "epoch": str(c["epoch"]), "variant": v, "set": s_,
                             "resp": val[k], "frac": float(c["set_frac"][k])})
    PX = pd.DataFrame(prow)
    den = PX[(PX.variant == "adopted") & (PX.epoch == "pre") & (PX.set == "interior")].groupby("animal").resp.mean()
    PX["resp_rel"] = PX.resp / PX.animal.map(den)
    own = PX[PX.epoch == "pre"].groupby(["variant", "set", "animal"]).resp.mean()
    PX["resp_rel_own"] = PX.resp / np.array([own.get(k_, np.nan) for k_ in zip(PX.variant, PX.set, PX.animal)])
    W = PX.pivot_table(index=["label", "animal", "epoch", "variant"], columns="set", values="resp").reset_index()
    W["edge_over_interior"] = W.edge / W.interior
    W["vessel_over_nonvessel"] = W.vessel / W.nonvessel
    W["var415_over_rest"] = W.var415 / W.nonvar415
    PX.to_csv(HC / "pixelset_sessions.csv", index=False)
    W.to_csv(HC / "pixelset_ratios_sessions.csv", index=False)
    print("\n== (e) per animal (rule 8): pixel-set response / animal's PRE ADOPTED interior response")
    print(_per_animal(PX[PX.variant.isin(MAP_VARIANTS)], ["variant", "set"], "resp_rel"))
    print("\n== (e) per animal: response in set / its own pre (growth factor), adopted")
    print(_per_animal(PX[PX.variant == "adopted"], ["set"], "resp_rel_own"))
    PC = pd.concat([_change_table(PX[PX.variant.isin(VARIANTS)], ["variant", "set"], ["resp_rel", "resp_rel_own"]),
                    _change_table(W.melt(id_vars=["label", "animal", "epoch", "variant"],
                                         value_vars=["edge_over_interior", "vessel_over_nonvessel", "var415_over_rest"],
                                         var_name="set", value_name="ratio"), ["variant", "set"], ["ratio"])])
    PC.to_csv(HC / "pixelset_change.csv", index=False)
    print("\n== (e) change vs pre: set ratios (edge/interior, vessel/non-vessel, var415/rest)\n"
          + PC[PC.value == "ratio"].round(3).to_string(index=False))
    print("\n== (e) change vs pre: growth vs own pre per set (adopted, raw415)\n"
          + PC[(PC.value == "resp_rel_own") & PC.variant.isin(MAP_VARIANTS)].round(3).to_string(index=False))
    # pooled pixel change maps (mean over animals of animal epoch-mean minus animal pre-mean)
    maps = {}
    for v in MAP_VARIANTS:
        per = {}
        for c in C.values():
            if (c["bout_gap"] >= il.GAP_MAIN).sum() < 10:
                continue
            per.setdefault((str(c["animal"]), str(c["epoch"])), []).append(c[f"map_{v}"].astype(np.float32))
        for ep in POST:
            dd = [np.nanmean(per[(a, ep)], 0) - np.nanmean(per[(a, "pre")], 0) for a in AN if (a, ep) in per and (a, "pre") in per]
            maps[(v, ep)] = np.nanmean(dd, 0) if dd else None
        maps[(v, "pre")] = np.nanmean([np.nanmean(per[(a, "pre")], 0) for a in AN if (a, "pre") in per], 0)
    any_c = next(iter(C.values()))
    stat_img = any_c["pstat_img"]
    fig, axs = plt.subplots(len(MAP_VARIANTS) + 1, 4, figsize=(15, 3.6 * (len(MAP_VARIANTS) + 1)))
    lim = np.nanpercentile(np.abs(maps[("adopted", "pre")]), 99)
    for i, v in enumerate(MAP_VARIANTS):
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            mp = maps[(v, ep)]
            if mp is None:
                ax.axis("off")
                continue
            im = ax.imshow(mp, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.contour(stat_img, levels=[0.5], colors="k", linewidths=0.5)
            ax.set_title(f"{VLAB[v]}\n{'pre mean' if ep == 'pre' else ep + ' - pre'}", fontsize=8)
            ax.axis("off")
    fig.colorbar(im, ax=axs[:len(MAP_VARIANTS), :], fraction=0.015).set_label("response AUC (dF/F x1e3 s)")
    for j, (s_, lab) in enumerate((("edge_over_interior", "edge / interior"), ("vessel_over_nonvessel", "vessel / non-vessel"),
                                   ("var415_over_rest", "top-5% 415-variance / rest"))):
        ax = axs[-1, j]
        _panel(ax, W[W.variant == "adopted"].rename(columns={s_: "val"}), "val", colors, ref=None)
        ax.set_title(f"ADOPTED: {lab} response ratio", fontsize=8)
    ax = axs[-1, 3]
    _panel(ax, PX[(PX.variant == "raw415") & (PX.set == "interior")], "resp_rel", colors, ref=0.0)
    ax.set_title("415 interior response / pre ADOPTED interior", fontsize=8)
    fig.suptitle("MOTION / EDGE / VESSEL check: ITI-lick response maps (AUC 0..0.5 s), mean of animals, one colour scale "
                 "(adopted pre 99th pct); contour = eroded stat mask (outside = EDGE rim). Image left = ipsilesional. Bottom: "
                 "per-session ratios of the adopted response in artefact-prone pixels to the rest (grey = animals, black = mean; "
                 "text = change vs pre, nested bootstrap). VESSEL = top 5 % of the high-passed negated mean 470 image; "
                 "415-variance = top 5 % of interior pixels by 415 variance.", fontsize=9, wrap=True)
    fig.savefig(il.OUT / "figures" / "iti_hemo_check_edge_vessel.png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    # --- (f) F0 per area
    frows = []
    for x, c in C.items():
        for k, nm in enumerate(c["f0_area_names"]):
            frows.append({"label": x, "animal": str(c["animal"]), "epoch": str(c["epoch"]), "area": str(nm),
                          "f0_470": float(c["f0_area"][0, k]), "f0_415": float(c["f0_area"][1, k])})
        for nm, arr in (("WHOLE_central60", c["f0_whole"]), ("BRAIN_statmask", c["f0_brain"])):
            frows.append({"label": x, "animal": str(c["animal"]), "epoch": str(c["epoch"]), "area": nm,
                          "f0_470": float(arr[0]), "f0_415": float(arr[1])})
    F = pd.DataFrame(frows)
    for ch_ in ("f0_470", "f0_415"):
        pre = F[F.epoch == "pre"].groupby(["area", "animal"])[ch_].mean()
        F[f"{ch_}_rel"] = F[ch_] / np.array([pre.get(k_, np.nan) for k_ in zip(F.area, F.animal)])
    F.to_csv(HC / "f0_area_sessions.csv", index=False)
    print("\n== (f) per animal (rule 8): F0 / pre, whole frame and brain stat mask")
    print(_per_animal(F[F.area.isin(["WHOLE_central60", "BRAIN_statmask"])], ["area"], "f0_470_rel"))
    print(_per_animal(F[F.area.isin(["WHOLE_central60", "BRAIN_statmask"])], ["area"], "f0_415_rel"))
    FC = _change_table(F, ["area"], ["f0_470_rel", "f0_415_rel"])
    FC.to_csv(HC / "f0_area_change.csv", index=False)
    s470 = FC[FC.value == "f0_470_rel"].pivot_table(index="area", columns="epoch", values="delta")
    print("\n== (f) F0_470 change vs pre (fraction), distribution over areas\n" + s470.describe().round(3).to_string())
    print(FC[FC.area.isin(["WHOLE_central60", "BRAIN_statmask"])].round(3).to_string(index=False))
    # dF-equivalent response per area: resp(adopted) x F0_area / pre F0_area
    A_ = R[(R.variant == "adopted") & R.roi.isin(areas)].rename(columns={"roi": "area"}).merge(
        F[["label", "area", "f0_470_rel"]], on=["label", "area"])
    A_["resp_dF"] = A_.resp * A_.f0_470_rel
    DFc = _change_table(A_, ["area"], ["resp", "resp_dF"])
    DFc.to_csv(HC / "resp_dF_area_change.csv", index=False)
    print("\n== (f) areas significant UP in dF/F vs in dF (per-area F0):\n"
          + DFc.assign(up=DFc.sig & (DFc.delta > 0)).groupby(["value", "epoch"]).up.sum().unstack().to_string())
    try:
        from frozen_area_maps import area_image, outline, paint
        img, nm = area_image(next(iter(C)))
        ol = outline(img)
        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        for i, ch_ in enumerate(("f0_470_rel", "f0_415_rel")):
            c_ = FC[FC.value == ch_]
            for j, ep in enumerate(POST):
                ax = axs[i, j]
                ce_ = c_[c_.epoch == ep].set_index("area")
                im = ax.imshow(paint(img, nm, ce_.delta.to_dict()), cmap="RdBu_r", vmin=-0.3, vmax=0.3, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                w = ce_.loc["WHOLE_central60"] if "WHOLE_central60" in ce_.index else None
                ax.set_title(f"F0 {ch_[3:6]} nm, {ep} - pre (fraction of pre)\n{int(ce_.sig.sum())} areas sig.; whole frame "
                             f"{w.delta:+.2f}" if w is not None else "", fontsize=8)
                ax.axis("off")
            fig.colorbar(im, ax=axs[i, :], fraction=0.02).set_label("F0 / pre - 1")
        fig.suptitle("Raw fluorescence F0 (frames_average on the atlas grid, raw counts) per Allen area, change vs the animal's "
                     "pre mean (mean of animals, paired nested bootstrap). Image left = ipsilesional.", fontsize=9, wrap=True)
        fig.savefig(il.OUT / "figures" / "iti_hemo_check_f0_areas.png", dpi=105, bbox_inches="tight")
        plt.close(fig)
    except Exception as ex:                                # noqa: BLE001
        print(f"F0 area map figure skipped: {ex}")
    # --- (g) state around licks + (h) gap-matched response
    grow, hrow = [], []
    pre_gap = {}
    for c in C.values():
        if str(c["epoch"]) == "pre":
            g = c["bout_gap"][c["bout_gap"] >= il.GAP_MAIN]
            pre_gap.setdefault(str(c["animal"]), []).append(g)
    nb = len(GAP_BINS) - 1
    pdist = {a: np.bincount(np.clip(np.digitize(np.concatenate(v), GAP_BINS) - 1, 0, nb - 1), minlength=nb)
             / len(np.concatenate(v)) for a, v in pre_gap.items()}
    for x, c in C.items():
        keep = c["bout_gap"] >= il.GAP_MAIN
        if keep.sum() < 10:
            continue
        an, ep = str(c["animal"]), str(c["epoch"])
        grow.append({"label": x, "animal": an, "epoch": ep, "run_win": np.nanmean(c["run_win"][keep]),
                     "run_base": np.nanmean(c["run_base"][keep]), "gap_median": float(np.median(c["bout_gap"][keep])),
                     "frac_running": float(np.nanmean(c["run_win"][keep] > RUN_THRESH))})
        if an not in pdist:
            continue
        gb = np.clip(np.digitize(c["bout_gap"][keep], GAP_BINS) - 1, 0, len(GAP_BINS) - 2)   # inf gap (first lick) -> top bin
        cnt_ = np.bincount(gb, minlength=len(GAP_BINS) - 1)[:len(GAP_BINS) - 1].astype(float)
        w = np.where(cnt_[gb] > 0, pdist[an][gb] / np.maximum(cnt_[gb], 1), 0.0)
        cols = list(c["cols"])
        ki = [cols.index(k) for k in il.KEY_ROIS]
        for v in ("adopted", "raw415"):
            r = c[f"roi_{v}"][keep].astype(float)
            hrow.append({"label": x, "animal": an, "epoch": ep, "variant": v,
                         "resp_key": r[:, ki].mean(1).mean(), "resp_key_gapmatched": float(w @ r[:, ki].mean(1) / w.sum()),
                         "resp_cortex": r[:, cols.index("cortex")].mean(),
                         "resp_cortex_gapmatched": float(w @ r[:, cols.index("cortex")] / w.sum())})
    G = pd.DataFrame(grow)
    G.to_csv(HC / "state_sessions.csv", index=False)
    print("\n== (g) per animal (rule 8): treadmill speed (mm/s) in -1..+1 s around the lick; share of licks while running "
          "(> 3 mm/s = RUN_THRESH); median no-lick gap (s)")
    for val in ("run_win", "frac_running", "gap_median"):
        print(val + "\n" + G.groupby(["animal", "epoch"])[val].mean().unstack().reindex(columns=EP).round(2).to_string())
    GC = _change_table(G.assign(k="all"), ["k"], ["run_win", "run_base", "frac_running", "gap_median"])
    GC.to_csv(HC / "state_change.csv", index=False)
    print(GC.round(3).to_string(index=False))
    Hh = pd.DataFrame(hrow)
    for v_ in ("resp_key", "resp_key_gapmatched", "resp_cortex", "resp_cortex_gapmatched"):
        base = v_.replace("_gapmatched", "")
        den = Hh[(Hh.variant == "adopted") & (Hh.epoch == "pre")].groupby("animal")[base].mean()
        Hh[v_ + "_rel"] = Hh[v_] / Hh.animal.map(den)
    Hh.to_csv(HC / "gapmatch_sessions.csv", index=False)
    print("\n== (h) per animal (rule 8): key_mean response / pre adopted, all vs gap-matched to the animal's pre gap distribution")
    print(_per_animal(Hh, ["variant"], "resp_key_rel"))
    print(_per_animal(Hh, ["variant"], "resp_key_gapmatched_rel"))
    HCh = _change_table(Hh, ["variant"], [c_ for c_ in Hh.columns if c_.endswith("_rel")])
    HCh.to_csv(HC / "gapmatch_change.csv", index=False)
    print(HCh.round(3).to_string(index=False))
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    _panel(axs[0], G, "run_win", colors, ref=None)
    axs[0].set_title("treadmill speed -1..+1 s (mm/s)", fontsize=8)
    _panel(axs[1], G, "gap_median", colors, ref=None)
    axs[1].set_title("median no-lick gap before the lick (s)", fontsize=8)
    _panel(axs[2], Hh[Hh.variant == "adopted"], "resp_key_rel", colors)
    axs[2].set_title("ADOPTED key_mean response / pre (all licks)", fontsize=8)
    _panel(axs[3], Hh[Hh.variant == "adopted"], "resp_key_gapmatched_rel", colors)
    axs[3].set_title("same, GAP-MATCHED to the animal's pre distribution", fontsize=8)
    fig.suptitle("BEHAVIOUR + NEURAL: global state around ITI licks and the response after matching the no-lick gap "
                 "(grey = animals, black = mean; text = change vs pre, nested bootstrap).", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(il.OUT / "figures" / "iti_hemo_check_state_gap.png", dpi=110)
    plt.close(fig)
    print("->", HC)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args(argv)
    from wfield_local import analysis_kit as ak
    from wfield_local.writeguard import assert_writable
    assert_writable(HC)
    CACHE.mkdir(parents=True, exist_ok=True)
    labels = a.labels or sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))
    if not a.summarise:
        res, fail = ak.fan_sessions(labels, work, jobs=min(a.jobs, 3))
        for f_ in fail:
            print("FAILED", f_)
    summarise(labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
