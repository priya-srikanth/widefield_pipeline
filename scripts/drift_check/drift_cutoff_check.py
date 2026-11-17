"""Drift-cutoff check (2026-11-03): could within-session slow drift be removed from the SHARED corrected signal without
removing block-level (spout-position) information, and at what cutoff?

Signal: `hemo_meegkit_hpfit_lag_refitT_swapmask` (set WIDEFIELD_HEMO_VARIANT; read via config.svtcorr_path), all 120
curated cohort 1 sessions. Nothing here writes to an adopted product, the frozen store or the shared session cache.

    python scripts/drift_check/drift_cutoff_check.py blocks   [--jobs 4]   # item 1 + 2 (DAQ only)
    python scripts/drift_check/drift_cutoff_check.py measure  [--jobs 4] [labels]   # items 3 + 4 (imaging)
    python scripts/drift_check/drift_cutoff_check.py replace  [--jobs 4] [--cutoffs 15 20] [--write 20]
    python scripts/drift_check/drift_cutoff_check.py plot     # --plot-only: figures + tables from the CSVs

Definitions
* BLOCK: run of consecutive trials at the same position (DAQ position code, `classify_cues_with_backup`, the decoder's
  own source). Duration = first trial's strobe -> last trial's stop (`trial_windows.trial_bounds` stop_s, matched to
  the cue by time). For the regressors the blocks PARTITION the task span (`drift_dct.block_segments`).
* OUT-OF-TRIAL mask: the hpfit fit mask (`filter_acausality_test.fit_mask`, strobedetrend: strobe-0.25 s -> cue+4 s
  excluded).
* BANDS by DCT period over the whole record. BLOCK BAND per animal: periods from the animal's median block duration to
  2 x its 95th-percentile block duration (a block of length D puts its energy at periods >~ D, fundamental 2D).
* ARMS (stacked on the swapmask signal; pass `replace` builds the REPLACEMENT): `blk<c>` = block-protected cosines
  with period >= c min (c = 10, 15, 20, 30); `cos10` = cosines >= 10 min, no block protection.
* DECODING: `filter_acausality_test` ROI features + its decoder (Allen-ROI, block GroupKFold, LR C=0.5), pre-cue
  (lick-free ENL, 2 s ending at the cue) and cue (2 s after the cue). Features are linear in the signal (no per-trial
  baseline), so each arm's features are base minus the features of its drift part -- one `_trial_features` call per
  alignment serves every arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/drift_cutoff_check_20261103")
VARIANT_ENV = "meegkit_hpfit_lag_refitT_swapmask"
FS = 31.23
CUTOFFS = (10, 15, 20, 30)
# blk = per-BLOCK protection (the literal spec; measured ill-posed), pos = per-POSITION protection, cos = unprotected
ARMS = ([(f"pos{c}", c, "position") for c in CUTOFFS] + [(f"blk{c}", c, "block") for c in (10, 20)]
        + [(f"cos{c}", c, None) for c in (10, 20)])
BANDS_MIN = [(">43", 43, np.inf), ("20-43", 20, 43), ("10-20", 10, 20), ("5-10", 5, 10), ("2-5", 2, 5),
             ("1-2", 1, 2), ("<1", 0, 1)]
POS_ORDER = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPOCHS = ("pre", "acute", "subacute", "chronic")


def _env_check():
    v = os.environ.get("WIDEFIELD_HEMO_VARIANT")
    if v != VARIANT_ENV:
        raise SystemExit(f"set WIDEFIELD_HEMO_VARIANT={VARIANT_ENV} (got {v!r})")
    sc = os.environ.get("WIDEFIELD_SESSION_CACHE", "")
    if "swapmask" not in sc:
        raise SystemExit("set WIDEFIELD_SESSION_CACHE to the swapmask cache (never the shared one)")
    if os.environ.get("WIDEFIELD_EPOCH_FROZEN") != "0":
        raise SystemExit("set WIDEFIELD_EPOCH_FROZEN=0")
    # the env's editable install points at the MAIN checkout; running this file puts scripts/drift_check (not the
    # worktree root) on sys.path, so without PYTHONPATH=<worktree> the workers import the wrong wfield_local
    import wfield_local
    root = Path(__file__).resolve().parents[2]
    if Path(wfield_local.__file__).resolve().parents[1] != root:
        raise SystemExit(f"wfield_local resolves to {wfield_local.__file__}; set PYTHONPATH={root}")


def _session(label):
    from wfield_local import analysis_kit as ak
    return {s["label"]: s for s in ak.curated_sessions()}[label]


def _pos_key(code):
    from wfield_local import daq_trials
    from wfield_local import orofacial_cohort as oc
    return oc.pos_key(daq_trials.POS_NAMES[int(code)], "L")         # all four lesions LEFT (animals.yaml)


def _events(s):
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    cue = _lc(s["h5"])
    return cue, np.asarray(classify_cues_with_backup(s, cue))


# ------------------------------------------------------------------------------------------------ item 1 + 2
def blocks_worker(label):
    """Per-block rows + per-session row (DAQ only)."""
    from wfield_local import config, epochs
    from wfield_local import trial_windows as TW
    s = _session(label)
    cue, codes = _events(s)
    sr = float(cue["sample_rate_hz"])
    cs = np.asarray(cue["cue_samples"], float) / sr
    ss = np.asarray(cue["strobe_samples"], float) / sr
    j = np.searchsorted(ss, cs, side="right") - 1
    st = np.where(j >= 0, ss[np.clip(j, 0, len(ss) - 1)], cs - 3.0)
    st = np.where(cs - st > 60, cs - 3.0, st)
    animal, mmdd = label.split("_")
    b = TW.trial_bounds(animal, config.date8(mmdd))
    bc = b.cue_s.to_numpy(float)
    k = np.clip(np.searchsorted(bc, cs), 1, len(bc) - 1)
    k = np.where(np.abs(bc[k - 1] - cs) < np.abs(bc[k] - cs), k - 1, k)
    matched = np.abs(bc[k] - cs) < 0.5
    stop = np.where(matched, b.stop_s.to_numpy(float)[k], cs + 3.5)
    pos_tb = b.pos_name.astype(str).to_numpy()[k]
    from wfield_local import daq_trials
    agree = np.mean([daq_trials.POS_NAMES.get(int(c), "?") == p for c, p, m in zip(codes, pos_tb, matched) if m])
    ok = codes >= 0
    idx = np.flatnonzero(ok)
    c = codes[idx]
    t0, t1 = st[idx[0]], np.nanmax(stop[idx])
    span = t1 - t0
    rows, i = [], 0
    ep = epochs.epoch_of(label)
    while i < c.size:
        e = i
        while e + 1 < c.size and c[e + 1] == c[i]:
            e += 1
        a, z = st[idx[i]], stop[idx[e]]
        rows.append(dict(label=label, animal=animal, epoch=ep, block=len(rows), pos=_pos_key(c[i]),
                         n_trials=e - i + 1, dur_s=z - a, start_s=a, stop_s=z,
                         t_mid_norm=((a + z) / 2 - t0) / span, t_start_norm=(a - t0) / span))
        i = e + 1
    # trial-level normalised time (for the order checks)
    trials = [dict(label=label, animal=animal, epoch=ep, pos=_pos_key(cc), t_norm=(cs[ii] - t0) / span)
              for ii, cc in zip(idx, c)]
    # dominant block-alternation period: periodogram of the six position indicators on a 1 s grid (task span)
    grid = np.arange(t0, t1, 1.0)
    P = np.zeros((grid.size, 6))
    starts = np.array([r["start_s"] for r in rows])
    which = np.clip(np.searchsorted(starts, grid, side="right") - 1, 0, len(rows) - 1)
    for g, w in enumerate(which):
        P[g, POS_ORDER.index(rows[w]["pos"])] = 1.0
    P -= P.mean(0)
    F = (np.abs(np.fft.rfft(P, axis=0)) ** 2).sum(1)
    fr = np.fft.rfftfreq(grid.size, 1.0)
    per_min = np.where(fr > 0, 1.0 / np.maximum(fr, 1e-12) / 60.0, np.inf)
    sel = (per_min >= 1) & (per_min <= 60)
    peak = float(per_min[sel][np.argmax(F[sel])]) if sel.any() else np.nan
    n_frames = np.load(config.svtcorr_path(s["mc"]), mmap_mode="r").shape[1]
    sess = dict(label=label, animal=animal, epoch=ep, n_trials=int(idx.size), n_blocks=len(rows),
                task_span_min=span / 60, imaging_min=n_frames / FS / 60, peak_period_min=peak,
                pos_agree_trial_bounds=float(agree), unmatched_stop=int((~matched).sum()))
    return dict(blocks=rows, trials=trials, session=sess)


def _eta(t, w, pp):
    """Duration-weighted eta^2 of block time by position label."""
    mu = np.average(t, weights=w)
    between = sum(w[pp == q].sum() * (np.average(t[pp == q], weights=w[pp == q]) - mu) ** 2 for q in np.unique(pp))
    return between / np.sum(w * (t - mu) ** 2)


def _order_stats(TR: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """Per animal (and animal x epoch): is position systematically related to time in session?"""
    from scipy.stats import friedmanchisquare
    out = []
    for key, g in list(TR.groupby("animal")) + list(TR.groupby(["animal", "epoch"])):
        an, ep = (key, "all") if isinstance(key, str) else key
        M = g.groupby(["label", "pos"]).t_norm.mean().unstack("pos").reindex(columns=list(POS_ORDER)).dropna()
        n = len(M)
        row = dict(animal=an, epoch=ep, n_sessions=n)
        for p in POS_ORDER:
            row[f"mean_t_{p}"] = M[p].mean() if n else np.nan
            row[f"sd_t_{p}"] = M[p].std() if n > 1 else np.nan
        if n >= 3:
            chi, pv = friedmanchisquare(*[M[p].to_numpy() for p in POS_ORDER])
            row.update(friedman_chi2=chi, friedman_p=pv, kendall_W=chi / (n * (len(POS_ORDER) - 1)))
        # within-session: eta^2 of trial time by position, vs block-label permutation null
        etas, nulls = [], []
        rng = np.random.default_rng(0)
        for _lab, bb in B[B.label.isin(g.label.unique())].groupby("label"):
            t, p = bb.t_mid_norm.to_numpy(), bb.pos.to_numpy()
            w = bb.dur_s.to_numpy()
            etas.append(_eta(t, w, p))
            nulls.append(np.mean([_eta(t, w, rng.permutation(p)) for _ in range(200)]))
        row.update(eta2_within=np.mean(etas), eta2_null=np.mean(nulls))
        # first block position: fraction of sessions starting at the modal position
        fb = B[(B.block == 0) & B.label.isin(g.label.unique())].pos
        row.update(first_block_modal=fb.mode().iat[0] if len(fb) else "", first_block_modal_frac=
                   fb.value_counts(normalize=True).iat[0] if len(fb) else np.nan)
        # time-only cross-session decoder (leave one session out): 10 time bins -> modal position in the others
        acc = []
        for lab in g.label.unique():
            tr, te = g[g.label != lab], g[g.label == lab]
            bins = np.linspace(0, 1, 11)
            ti, te_i = np.digitize(tr.t_norm, bins), np.digitize(te.t_norm, bins)
            mode = tr.assign(b=ti).groupby("b").pos.agg(lambda x: x.value_counts().index[0])
            pred = pd.Series(te_i).map(mode).to_numpy()
            acc.append(np.mean(pred == te.pos.to_numpy()))
        row.update(time_only_loso_acc=np.mean(acc) if acc else np.nan)
        out.append(row)
    return pd.DataFrame(out)


def cmd_blocks(a):
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    labels = a.labels or [s["label"] for s in ak.curated_sessions()]
    got, fail = ak.fan_sessions(labels, blocks_worker, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED {it}: {err}", flush=True)
    B = pd.DataFrame([r for _k, v in got for r in v["blocks"]])
    TR = pd.DataFrame([r for _k, v in got for r in v["trials"]])
    S = pd.DataFrame([v["session"] for _k, v in got])
    B.to_csv(OUT / "blocks.csv", index=False, float_format="%.5g")
    TR.to_csv(OUT / "trial_times.csv", index=False, float_format="%.5g")
    S.to_csv(OUT / "sessions.csv", index=False, float_format="%.5g")
    _summarise_blocks(B, S, TR)
    if fail:
        print(f"{len(fail)} sessions FAILED -- tables are partial", flush=True)
    return 0


def _summarise_blocks(B, S, TR):
    g = B.groupby(["animal", "epoch"])
    T = pd.DataFrame({"n_sessions": g.label.nunique(), "n_blocks": g.size(),
                      "trials_median": g.n_trials.median(), "trials_max": g.n_trials.max(),
                      "dur_median_s": g.dur_s.median(), "dur_p90_s": g.dur_s.quantile(0.9),
                      "dur_p95_s": g.dur_s.quantile(0.95), "dur_max_s": g.dur_s.max()})
    gs = S.groupby(["animal", "epoch"])
    T["task_span_min_median"] = gs.task_span_min.median()
    T["imaging_min_median"] = gs.imaging_min.median()
    T["peak_period_min_median"] = gs.peak_period_min.median()
    T = T.reindex(pd.MultiIndex.from_product([ANIMALS, EPOCHS]), fill_value=np.nan).dropna(how="all").reset_index()
    T = T.rename(columns={"level_0": "animal", "level_1": "epoch"})
    T.to_csv(OUT / "block_summary_animal_epoch.csv", index=False, float_format="%.4g")
    ga = B.groupby("animal").dur_s
    band = pd.DataFrame({"median_s": ga.median(), "p95_s": ga.quantile(0.95), "max_s": ga.max()})
    band["band_lo_s"], band["band_hi_s"] = band.median_s, 2 * band.p95_s
    # CYCLE band: positions recur every ~6 blocks, so a position's indicator has its fundamental at the alternation
    # period (Fig 1c): from the median block to the animal's 95th-percentile session peak period
    band["cycle_hi_s"] = S.groupby("animal").peak_period_min.quantile(0.95) * 60
    band.to_csv(OUT / "block_band_per_animal.csv", float_format="%.5g")
    O = _order_stats(TR, B)
    O.to_csv(OUT / "spout_order.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(T.round(1).to_string(index=False))
    print(band.round(1).to_string())
    print(O[["animal", "epoch", "n_sessions", "kendall_W", "friedman_p", "eta2_within", "eta2_null",
             "first_block_modal", "first_block_modal_frac", "time_only_loso_acc"]].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ item 3 + 4
def _load_inputs(s):
    """SVTcorr (K, T) float64, out-of-trial mask, block segments, ROI matrix + regions, brain-mean vector."""
    import glob

    from wfield_local import config
    from wfield_local import drift_dct as dd
    from wfield_local.filter_acausality_test import MASK_SPEC, fit_mask
    from wfield_local.locanmf_crossanimal_dff import _frames as _fr
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    C = np.load(config.svtcorr_path(s["mc"])).astype(np.float64)
    cue, codes = _events(s)
    lk = _ll(s["h5"], "lick_analog")
    _c, _l, csmp = _fr(s, cue, lk)
    mask, _d = fit_mask(s, C.shape[1], csmp, cue, **MASK_SPEC["strobedetrend"])
    mask = mask[:C.shape[1]]
    segs = dd.block_segments(cue, codes, csmp, C.shape[1], FS)
    ad = glob.glob(s["mc"] + "/wfield_local_results/allen_aligned_affine8v1")[0]
    U = np.load(f"{ad}/U_atlas.npy")
    atlas = np.load(f"{ad}/allen_area_atlas_native_grid.npy").reshape(-1)
    bm = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool).reshape(-1)
    Uf = U.reshape(-1, U.shape[2])[:, :C.shape[0]]
    R, regs = [], []
    for lab in np.unique(atlas):                              # = filter_acausality_test.roi_signal
        if lab == 0:
            continue
        pix = (atlas == lab) & bm
        if pix.sum() < 20:
            continue
        R.append(np.nanmean(Uf[pix], 0))
        regs.append(int(lab))
    ub = np.nanmean(Uf[bm], 0)
    del U, Uf
    return C, mask, segs, np.asarray(R), np.asarray(regs), ub, cue, codes


def _band_masks(n, lo_s, hi_s):
    from wfield_local import drift_dct as dd
    per = dd.dct_periods_s(n, FS)
    return (per >= lo_s) & (per < hi_s) & np.isfinite(per)


def _decode_arms(s, sig_base, drift_sigs: dict, regs, align):
    """{arm: acc} for base + every arm from ONE _trial_features call (features are linear in the signal)."""
    from types import SimpleNamespace

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wfield_local.filter_acausality_test import _max_rt
    from wfield_local.locanmf_position_decoder import _trial_features
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER
    names = ["base"] + list(drift_sigs)
    sig = np.vstack([sig_base] + [drift_sigs[k] for k in drift_sigs]).astype(np.float32)
    tag = np.arange(sig.shape[0])                     # row index; arm = row // n_roi (Allen labels can be large)
    args = SimpleNamespace(source="roi", align=align, baseline="none", pre_s=1.0, post_s=2.0, fs=FS, max_rt=_max_rt())
    X, y, g, _xn, _yn, fr = _trial_features(s, args, signal=sig, feat_region=tag)
    ng = min(5, int(np.unique(g).size))
    out = {}
    if ng < 2 or len(np.unique(y)) < len(DISPLAY_ORDER):
        return {k: np.nan for k in names}, int(len(y))
    arm_of = np.asarray(fr) // len(regs)
    Xb = X[:, arm_of == 0]
    for i, k in enumerate(names):
        Xa = Xb if i == 0 else Xb - X[:, arm_of == i]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        out[k] = float(accuracy_score(y, cross_val_predict(clf, Xa, y, cv=GroupKFold(ng), groups=g)))
    return out, int(len(y))


def measure_worker(label):
    from scipy.fft import dct, idct

    from wfield_local import drift_dct as dd
    from wfield_local import epochs
    s = _session(label)
    band = pd.read_csv(OUT / "block_band_per_animal.csv", index_col=0)
    animal = label.split("_")[0]
    blo, bhi = float(band.loc[animal, "band_lo_s"]), float(band.loc[animal, "band_hi_s"])
    chi = float(band.loc[animal, "cycle_hi_s"])
    C, mask, segs, R, regs, ub, _cue, _codes = _load_inputs(s)
    K, N = C.shape
    row = dict(label=label, animal=animal, epoch=epochs.epoch_of(label), n_frames=N, minutes=N / FS / 60,
               mask_frac=float(mask.mean()), n_blocks=len(segs), block_band_lo_s=blo, block_band_hi_s=bhi,
               cycle_band_hi_s=chi)
    Cc = C - C.mean(1, keepdims=True)
    Z = dct(Cc, norm="ortho", axis=1)                          # (K, N)
    zb = ub @ Z                                                # brain-mean DCT coefficients
    bands = {nm: (lo * 60, hi * 60) for nm, lo, hi in BANDS_MIN}
    bands["block"] = (blo, bhi)
    bands["cycle"] = (blo, chi)
    bands["10-43"] = (600.0, 2580.0)
    tot_all = {"comp": float((Z ** 2).sum()), "bm": float((zb ** 2).sum())}
    bm_t = ub @ Cc
    tot_oot = {"comp": float(((Cc[:, mask]) ** 2).sum() / mask.sum() * N), "bm": float(np.sum(bm_t[mask] ** 2)
                                                                                         / mask.sum() * N)}
    # ---- item 3: variance by band, all frames (Parseval) and out-of-trial frames (band component on mask frames)
    bmask = {nm: _band_masks(N, lo, hi) for nm, (lo, hi) in bands.items()}
    for nm, sel in bmask.items():
        row[f"v3_comp_all_{nm}"] = float((Z[:, sel] ** 2).sum()) / tot_all["comp"]
        row[f"v3_bm_all_{nm}"] = float((zb[sel] ** 2).sum()) / tot_all["bm"]
        Zs = np.where(sel[None, :], Z, 0.0)
        xs = idct(Zs, norm="ortho", axis=1)
        row[f"v3_comp_oot_{nm}"] = float((xs[:, mask] ** 2).sum() / mask.sum() * N) / tot_oot["comp"]
        bs = ub @ xs
        row[f"v3_bm_oot_{nm}"] = float(np.sum(bs[mask] ** 2) / mask.sum() * N) / tot_oot["bm"]
        del Zs, xs
    # ---- item 4: arms
    sig_base = R @ C
    seg_ix = [(g["start"], g["stop"], g["code"]) for g in segs]

    def block_means(x):                                       # (n_feat, n_blocks)
        return np.stack([x[:, a:b].mean(1) for a, b, _c in seg_ix], 1)
    codes_b = np.array([c for _a, _b, c in seg_ix])
    w_b = np.array([b - a for a, b, _c in seg_ix], float)
    feat_base = np.vstack([sig_base, bm_t[None, :]])
    BM0 = block_means(feat_base)

    def pos_var(M):                                           # duration-weighted between-position variance per feature
        mu = (M * w_b).sum(1, keepdims=True) / w_b.sum()
        return sum(w_b[codes_b == q].sum() * (((M[:, codes_b == q] * w_b[codes_b == q]).sum(1)
                   / w_b[codes_b == q].sum()) - mu[:, 0]) ** 2 for q in np.unique(codes_b)) / w_b.sum()
    drift_sigs = {}
    for arm, cut, prot in ARMS:
        beta, B, info = dd.fit_cosine_drift(C, mask, FS, cut * 60.0, segs, protect=prot)
        kmax = info["kmax"]
        Za = Z.copy()
        Za[:, 1:kmax + 1] -= beta[:, 1:]
        zba = ub @ Za
        for nm in ("block", "cycle", "1-2", "2-5", "5-10", "10-20", "20-43", ">43", "10-43"):
            sel = bmask[nm]
            den_c, den_b = (Z[:, sel] ** 2).sum(), (zb[sel] ** 2).sum()
            row[f"{arm}_ret_comp_{nm}"] = float((Za[:, sel] ** 2).sum() / den_c) if den_c > 0 else np.nan
            row[f"{arm}_ret_bm_{nm}"] = float((zba[sel] ** 2).sum() / den_b) if den_b > 0 else np.nan
        row[f"{arm}_removed_comp_total"] = 1 - float((Za ** 2).sum()) / tot_all["comp"]
        row[f"{arm}_removed_bm_total"] = 1 - float((zba ** 2).sum()) / tot_all["bm"]
        row[f"{arm}_kmax"] = kmax
        row[f"{arm}_robust_kept"] = info["frac_kept_robust"]
        row[f"{arm}_n_blocks_fit"] = info["n_blocks_fit"]
        dsig = (R @ beta[:, 1:]) @ B[:, 1:].T                  # ROI signal of the removed drift
        dbm = (ub @ beta[:, 1:]) @ B[:, 1:].T
        drift_sigs[arm] = dsig
        BM1 = BM0 - block_means(np.vstack([dsig, dbm[None, :]]))
        r = [np.corrcoef(BM0[i], BM1[i])[0, 1] for i in range(BM0.shape[0])]
        vr = BM1.var(1) / BM0.var(1)
        pr = pos_var(BM1) / pos_var(BM0)
        row[f"{arm}_bmean_r_roi_median"] = float(np.nanmedian(r[:-1]))
        row[f"{arm}_bmean_r_roi_min"] = float(np.nanmin(r[:-1]))
        row[f"{arm}_bmean_r_bm"] = float(r[-1])
        row[f"{arm}_bmean_var_ratio_roi_median"] = float(np.nanmedian(vr[:-1]))
        row[f"{arm}_bmean_var_ratio_bm"] = float(vr[-1])
        row[f"{arm}_bmean_posvar_ratio_roi_median"] = float(np.nanmedian(pr[:-1]))
        row[f"{arm}_bmean_posvar_ratio_bm"] = float(pr[-1])
        del Za, B
    for align in ("precue", "cue"):
        acc, n = _decode_arms(s, sig_base, drift_sigs, regs, align)
        row[f"n_trials_{align}"] = n
        for k, v in acc.items():
            row[f"dec_{align}_{k}"] = v
    print(f"  {label}: " + "  ".join(f"{k[4:]} {row[k]:.3f}" for k in row if k.startswith("dec_")), flush=True)
    return row


def cmd_measure(a):
    from wfield_local import analysis_kit as ak
    labels = a.labels or [s["label"] for s in ak.curated_sessions()]
    f = OUT / "measure_sessions.csv"
    prev = pd.read_csv(f) if f.exists() and not a.fresh else pd.DataFrame()
    todo = [x for x in labels if prev.empty or x not in set(prev.label)]
    print(f"measure: {len(todo)} to do, {len(labels) - len(todo)} reused", flush=True)
    rows = []
    for i in range(0, len(todo), 8):                           # checkpoint every 8 sessions
        got, fail = ak.fan_sessions(todo[i:i + 8], measure_worker, jobs=a.jobs)
        for it, err in fail:
            print(f"  FAILED {it}: {str(err)[-600:]}", flush=True)
        rows += [v for _k, v in got]
        D = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("label", keep="last")
        D.sort_values("label").to_csv(f, index=False, float_format="%.6g")
    return 0


# ------------------------------------------------------------------------------------------------ replacement arm
def replace_worker(item):
    """item = (label, cutoffs tuple, write_cutoff or 0, family 'dctpos'|'dctblk'). Replacement (polynomial -> DCT)
    vs stacked (hpfit signal minus the same DCT fit)."""
    label, cutoffs, write_c, fam = item
    from wfield_local import drift_dct as dd
    from wfield_local import epochs
    from wfield_local import hemo_variants as hv
    s = _session(label)
    C, mask, segs, R, regs, ub, _cue, _codes = _load_inputs(s)
    res = Path(s["mc"]) / "wfield_local_results"
    bad = np.load(res / "swapped_pairs.npz")["bad"]
    func = hv.functional_channel(s)
    out = []
    for c in cutoffs:
        var = f"{fam}{c}_lag"
        rep, _T, rc, meta = hv.compute(s, var, refit_t=True, verbose=False, func=func, bad_pairs=bad)
        rep = rep.astype(np.float64)
        beta, B, _info = dd.fit_cosine_drift(C, mask, FS, c * 60.0, segs, protect=hv.VARIANTS[var]["dct_protect"])
        stk = C - beta[:, 1:] @ B[:, 1:].T
        stk -= stk.mean(1, keepdims=True)
        rep -= rep.mean(1, keepdims=True)
        sr, ss_, sb = R @ rep, R @ stk, R @ C
        brep, bstk, bbase = ub @ rep, ub @ stk, ub @ (C - C.mean(1, keepdims=True))
        row = dict(label=label, animal=label[:4], epoch=epochs.epoch_of(label), family=fam, cutoff_min=c,
                   r2_comp_rep_vs_stk=1 - float(((rep - stk) ** 2).sum() / (stk ** 2).sum()),
                   r2_comp_base_vs_stk=1 - float(((C - C.mean(1, keepdims=True) - stk) ** 2).sum() / (stk ** 2).sum()),
                   r_bm_rep_vs_stk=float(np.corrcoef(brep, bstk)[0, 1]),
                   r_bm_rep_vs_base=float(np.corrcoef(brep, bbase)[0, 1]),
                   r_roi_rep_vs_stk_median=float(np.median([np.corrcoef(sr[i], ss_[i])[0, 1] for i in range(len(sr))])),
                   r_roi_rep_vs_stk_min=float(np.min([np.corrcoef(sr[i], ss_[i])[0, 1] for i in range(len(sr))])),
                   var_ratio_rep_over_stk=float((rep ** 2).sum() / (stk ** 2).sum()),
                   rcoeff_median=float(np.median(rc)), n_position_blocks=meta.get("n_position_blocks"))
        # slow-band comparison: power at periods >= 5 min in each
        from scipy.fft import dct
        sel = _band_masks(rep.shape[1], 300.0, np.inf)
        row["slow5_power_rep_over_stk_bm"] = float((dct(brep, norm="ortho")[sel] ** 2).sum()
                                                   / (dct(bstk, norm="ortho")[sel] ** 2).sum())
        acc, _n = _decode_arms(s, sb, {"stk": sb - ss_, "rep": sb - sr}, regs, "precue")
        acc2, _n2 = _decode_arms(s, sb, {"stk": sb - ss_, "rep": sb - sr}, regs, "cue")
        row.update({f"dec_precue_{k}": v for k, v in acc.items()})
        row.update({f"dec_cue_{k}": v for k, v in acc2.items()})
        if write_c and c == write_c:
            d = hv.out_dir(s, var, True, "_swapmask")
            if d.exists():
                row["written"] = f"EXISTS (not touched): {d}"
            else:
                hv.write(s, var, refit_t=True, verbose=False, func=func, bad_pairs=bad, tag="_swapmask")
                row["written"] = str(d)
        print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        out.append(row)
        del rep, stk
    return out


def cmd_replace(a):
    from wfield_local import analysis_kit as ak
    B = pd.read_csv(OUT / "blocks.csv")
    mx = B.groupby(["label", "animal", "epoch"]).dur_s.max().reset_index()
    labels = []
    for an in ANIMALS:
        for ep in ("pre", "acute"):
            g = mx[(mx.animal == an) & (mx.epoch == ep)].sort_values("dur_s", ascending=False)
            labels += g.label.head(1).tolist()                 # the session with the longest block
    labels = a.labels or labels
    print("replacement sessions:", labels, flush=True)
    items = [(lab, tuple(a.cutoffs), a.write or 0, a.family) for lab in labels]
    got, fail = ak.fan_sessions(items, replace_worker, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED {it}: {str(err)[-600:]}", flush=True)
    D = pd.DataFrame([r for _k, v in got for r in v])
    f = OUT / "replacement_vs_stacked.csv"
    if f.exists():
        D = pd.concat([pd.read_csv(f), D]).drop_duplicates(["label", "family", "cutoff_min"], keep="last")
    D.sort_values(["label", "cutoff_min"]).to_csv(f, index=False, float_format="%.6g")
    print(D.round(4).to_string(index=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["blocks", "measure", "replace", "plot"])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[20])
    ap.add_argument("--write", type=int, default=0, help="also WRITE hemo_<family><c>_lag_refitT_swapmask for this c")
    ap.add_argument("--family", default="dctpos", choices=["dctpos", "dctblk"])
    a = ap.parse_args(argv)
    a.jobs = min(a.jobs, 4)
    if a.cmd == "plot":
        sys.path.insert(0, str(Path(__file__).parent))
        import drift_cutoff_plots
        return drift_cutoff_plots.main(OUT)
    _env_check()
    return {"blocks": cmd_blocks, "measure": cmd_measure, "replace": cmd_replace}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
