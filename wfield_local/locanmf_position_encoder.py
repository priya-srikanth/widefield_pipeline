"""Position ENCODER (reverse of the decoder): predict expected neural activity from intended spout
position. The pre/post-stroke tool -- fit pre-stroke, then post-stroke residual (observed - predicted)
per intended position = the lesion's effect, computable even on no-lick/failed trials.

  spatial: ridge  one-hot(position) -> LocaNMF component activity (2s post-lick, no-baseline), block-CV.
    -> predicted per-position cortical map (footprint-reconstructed) + cross-validated encoding R^2 by region.
  temporal: per-position expected time-course of pooled SSp / MO activity (lick-aligned) -- "expected
    dynamics" per position; the linear encoder's prediction at each time bin.

    python -m wfield_local.locanmf_position_encoder --date 0605 --output "<dir>"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from collections import defaultdict

from wfield_local import config
from wfield_local.locanmf_cue_lick_analysis import SESSIONS, ANIMAL_COLOR
from wfield_local.locanmf_position_decoder import _trial_features, _build_signal
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES, DISPLAY_ORDER, _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue, _classify_cues
from wfield_local.behavior_position import classify_cues_with_backup
from wfield_local.locanmf_crossanimal_dff import _frames

FS = 31.23


def _sess(label):
    return next(s for s in SESSIONS if s["label"] == label)


def _args(post_s=2.0, baseline="none", bins=1):
    """Feature spec for the ENCODER.

    ``bins=1`` (a single window mean) is DELIBERATE and explicit, not inherited. The decoder adopted
    sub-binned features on 2026-08-14 because they measurably help CLASSIFICATION, and `_bins_for`
    falls back to `defaults.yaml decode.bins` when args carry no `bins` -- which would silently have
    turned this encoder into a model of each component's 8-bin TIME COURSE rather than its mean
    activity, changing what R^2, EV and the FEVE ceiling mean without anyone choosing it. The
    sub-binning gain was measured for decoding only; whether a time-course target improves the
    forward model is untested, so the encoder stays on the mean until it is.
    """
    return SimpleNamespace(source="locanmf", align="lick", baseline=baseline, pre_s=1.0,
                           post_s=post_s, fs=FS, max_rt=float(config.defaults()["decode"]["max_rt_s"]), bins=bins)


def _names(s):
    return {int(k): v for k, v in json.load(open(glob.glob(
        f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}


def encode_spatial(label):
    """Ridge encoder position->activity; returns CV R^2 per component, per-position predicted activity (B)."""
    s = _sess(label)
    X, y, g, _, _, reg = _trial_features(s, _args(2.0))
    pos = np.array(DISPLAY_ORDER); P = np.stack([(y == p).astype(float) for p in pos], 1)
    pred = np.zeros_like(X); ng = min(5, int(np.unique(g).size))
    for tr, te in GroupKFold(ng).split(X, y, g):
        pred[te] = Ridge(alpha=1.0).fit(P[tr], X[tr]).predict(P[te])
    r2 = 1 - ((X - pred) ** 2).sum(0) / np.maximum(((X - X.mean(0)) ** 2).sum(0), 1e-12)
    B = Ridge(alpha=1.0).fit(P, X).coef_.T                       # (6, ncomp) predicted activity per position
    # noise ceiling per component = between-position var / total var (max R^2 a position-only model can reach)
    gm = X.mean(0); betw = np.zeros(X.shape[1]); wit = np.zeros(X.shape[1])
    for p in pos:
        mm = y == p
        # AN ABANDONED POSITION CONTRIBUTES NO BETWEEN-CLASS SS. Skipping is not a
        # convenience: X[mm].mean(0) on an empty mask is nan, and `mm.sum() * (...)` is
        # then 0 * nan = nan, NOT 0. One such position turned every region's pooled
        # `expl` into nan, and `nan > floor` is False, so the 58-session FEVE figure
        # collapsed to ZERO regions (2026-08-20/21). Post-stroke animals abandoning a
        # position outright is the phenotype, so this arrived with the science.
        if not mm.any():
            continue
        mu = X[mm].mean(0); betw += mm.sum() * (mu - gm) ** 2; wit += ((X[mm] - mu) ** 2).sum(0)
    ceiling = betw / (betw + wit + 1e-12)
    return dict(label=label, r2=r2, B=B, reg=reg, pos=pos, cv_r2=float(np.mean(r2)), ceiling=ceiling)


def _quiet_baseline_local(s, sig, nbins=24):
    """TIME-LOCAL quiet (rest) baseline per component, (ncomp, T): bin the session into nbins, take the
    median of quiet (no-lick/no-move) frames per bin, interpolate to every frame -> tracks slow drift
    (photobleaching / state). Falls back to a session-constant mean if no quiet mask. This is the stable
    cross-session reference for the pre/post-stroke residual."""
    T = sig.shape[1]
    qf = glob.glob(f"{s['mc']}/quiet_affine8v1/*quiet_frame.npy")
    if not qf:
        return np.repeat(sig.mean(1, keepdims=True), T, axis=1)
    q = np.load(qf[0]).astype(bool); L = min(q.shape[0], T); qm = np.zeros(T, bool); qm[:L] = q[:L]
    qi = np.where(qm)[0]
    edges = np.linspace(0, T, nbins + 1); cent = (edges[:-1] + edges[1:]) / 2
    bm = np.full((sig.shape[0], nbins), np.nan)
    for b in range(nbins):
        bq = qi[(qi >= edges[b]) & (qi < edges[b + 1])]
        if len(bq):
            bm[:, b] = np.median(sig[:, bq], axis=1)
    base = np.empty((sig.shape[0], T))
    for c in range(sig.shape[0]):
        good = ~np.isnan(bm[c])
        base[c] = np.interp(np.arange(T), cent[good], bm[c, good]) if good.sum() >= 2 else (np.nanmean(bm[c]) if good.any() else sig[c].mean())
    return base


def _engaged_frames(s, post_s=2.0):
    """First-lick frame + position for engaged (cue+lick) trials, with the post window length (frames)."""
    cue = _load_cue(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, _ = _frames(s, cue, lk); codes = classify_cues_with_backup(s, cue)
    ls = np.sort(lf); j = np.searchsorted(ls, cf, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cf
    post_n = int(round(post_s * FS)); fr, y = [], []
    for k in range(cf.size):
        if codes[k] >= 0 and first[k] > 0 and 0 < rt[k] <= 2 * FS:
            fr.append(int(first[k])); y.append(int(codes[k]))
    return np.array(fr), np.array(y), post_n


def fig_predicted_maps(label, out):
    """Predicted per-position pixel-ΔF/F map = A @ C (the TRUE data reconstruction, not the footprint-
    scaled s*C which reweights components), relative to the TIME-LOCAL quiet (rest) baseline. Diverging
    colormap so ΔF/F can go negative (blue=below rest). A@C is the cross-session-comparable frame for the
    pre/post-stroke residual; footprint scaling is a within-session per-component normalization, not used here."""
    s = _sess(label); e = encode_spatial(label)
    C = np.load(f"{config.locanmf_dir(s['mc'])}/{label}_locanmf_C.npy")       # RAW C (not footprint-scaled)
    base = _quiet_baseline_local(s, C)                                            # time-local rest baseline on raw C
    fr, y, post_n = _engaged_frames(s)
    feats = np.array([C[:, f:f + post_n].mean(1) - base[:, f:f + post_n].mean(1) for f in fr])
    B = np.stack([feats[y == p].mean(0) for p in DISPLAY_ORDER])                  # 6 x ncomp raw-C activity above rest
    Ar = np.load(f"{config.locanmf_dir(s['mc'])}/{label}_locanmf_A.npy"); H, Wd = Ar.shape[:2]
    Af = np.nan_to_num(Ar.reshape(-1, Ar.shape[2]))
    mask = np.load(glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_brain_mask_native_grid.npy")[0]).astype(bool)
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    maps = [(Af @ B[p]).reshape(H, Wd) for p in range(6)]   # A@C true pixel-dF/F above (time-local) rest
    vmax = np.nanpercentile([np.abs(m[mask]) for m in maps], 99)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, p in zip(axes.ravel(), range(6)):
        m = maps[p].astype(float); m[~mask] = np.nan
        im = ax.imshow(m[y0:y1, x0:x1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)    # diverging: red=above rest, blue=below
        ax.set_title(POSITION_NAMES[DISPLAY_ORDER[p]], fontsize=11); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"{label}: ENCODER expected activity per intended position (TIME-LOCAL quiet baseline; "
                 f"red=above rest, blue=below; single-trial CV R^2={e['cv_r2']:.3f})", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_encoder_predicted_maps_{label}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_encoding_r2(label, out, topn=16):
    """Explained variance by region, two views (region-level via summed SS over each region's components,
    consistent with the FEVE heatmaps):
      LEFT  absolute — EXPLAINABLE (noise ceiling = between-position SS / single-trial SS) vs CAPTURED
            (encoder CV SS / single-trial SS); both are fractions of single-trial variance.
      RIGHT normalized-to-1.0 — FEVE = captured / EXPLAINABLE per region (what fraction of the explainable
            variance the encoder actually explains; 1.0 = all of it). This removes the ceiling height so
            you see encoder QUALITY per region; low absolute capture is mostly a low ceiling, not failure."""
    fe = _region_feve(label)
    regs = sorted(fe, key=lambda r: fe[r]["expl"] / max(fe[r]["tot"], 1e-12), reverse=True)[:topn]
    expl = np.array([fe[r]["expl"] / max(fe[r]["tot"], 1e-12) for r in regs])
    cap = np.array([fe[r]["cap"] / max(fe[r]["tot"], 1e-12) for r in regs])
    feve = np.array([fe[r]["cap"] / fe[r]["expl"] if fe[r]["expl"] > 0 else np.nan for r in regs])
    cols = ["#c0392b" if r.startswith(("SSp", "SSs")) else "#2980b9" if r.startswith("MO") else "#555" for r in regs]
    y = np.arange(len(regs))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.barh(y + 0.0, expl, 0.4, color="#bbbbbb", label="explainable (noise ceiling)")
    ax.barh(y - 0.42, cap, 0.4, color=cols, label="captured (CV)")
    ylab = [f"{r} (n={fe[r]['n']})" + ("*" if fe[r]["n"] == 1 else "") for r in regs]
    ax.set_yticks(y); ax.set_yticklabels(ylab, fontsize=8); ax.set_xlabel("fraction of single-trial variance")
    ax.set_title("absolute: explainable vs captured  (* = single component, noisier)"); ax.legend(fontsize=8)
    ax = axes[1]
    ax.barh(y, feve, 0.55, color=cols)
    ax.axvline(1.0, color="k", ls="--", lw=1.0)
    lo = min(0.0, np.nanmin(feve) - 0.05) if np.isfinite(feve).any() else 0.0
    hi = max(1.15, (np.nanmax(feve) + 0.08) if np.isfinite(feve).any() else 1.15)
    ax.set_xlim(lo, hi); ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("FEVE = captured / explainable  (1.0 dashed = all explainable variance explained)")
    ax.set_title("normalized to explainable (per region)")
    for yi, v in zip(y, feve):
        if np.isfinite(v):
            ax.text(min(v, hi) + 0.012, yi, f"{v:.2f}", va="center", fontsize=7)
    fig.suptitle(f"{label}: encoder explained variance by region — absolute vs ceiling-normalized to 1.0 "
                 f"(SSp/SSs red, MO blue; sorted by explainable variance)", fontsize=11)
    fig.tight_layout(); p = out / f"locanmf_encoder_r2_by_region_{label}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_temporal_encoder(label, out, pre_s=1.0, post_s=1.5):
    """Per-position expected time-course of pooled SSp / MO activity (lick-aligned)."""
    s = _sess(label); names = _names(s)
    sig, reg = _build_signal(s, "locanmf"); T = sig.shape[1]
    rn = np.array([names.get(int(reg[i]), "?") for i in range(sig.shape[0])])
    ssp = np.where(np.char.startswith(rn.astype(str), "SSp"))[0]
    mo = np.array([i for i in range(len(rn)) if rn[i].startswith(("MOp", "MOs"))])
    cue = _load_cue(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, _ = _frames(s, cue, lk); codes = classify_cues_with_backup(s, cue)
    ls = np.sort(lf); j = np.searchsorted(ls, cf, side="right"); first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cf
    pre = int(pre_s * FS); post = int(post_s * FS); tax = np.arange(-pre, post) / FS
    keep = [k for k in range(cf.size) if codes[k] >= 0 and first[k] > 0 and 0 < rt[k] <= 2 * FS and first[k] - pre >= 0 and first[k] + post <= T]
    flk = np.array([int(first[k]) for k in keep]); yk = np.array([int(codes[k]) for k in keep])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, idx, tit in [(axes[0], ssp, "SSp (pooled)"), (axes[1], mo, "MO (pooled)")]:
        if len(idx) == 0:
            ax.set_title(f"{tit}: none"); continue
        pooled = sig[idx][:, :].mean(0)                      # 1 x T pooled activity
        for p in DISPLAY_ORDER:
            fp = flk[yk == p]
            if len(fp) < 5:                                  # guard before stacking (empty -> ValueError)
                continue
            tr = np.stack([pooled[f - pre:f + post] for f in fp])
            m = tr.mean(0) - tr[:, :pre].mean()
            ax.plot(tax, m, label=POSITION_NAMES[p])
        ax.axvline(0, color="k", lw=1); ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel("time from first lick (s)"); ax.set_title(f"{tit} expected activity by position"); ax.legend(fontsize=7)
    axes[0].set_ylabel("pooled dF/F (baseline-sub)")
    fig.suptitle(f"{label}: TEMPORAL encoder -- expected activity time-course per intended position", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_encoder_temporal_{label}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_ev_by_position(labels, out, tag):
    """Total (whole-cortex) held-out explained variance per spout position, per session: how much
    better the encoder predicts each position's trials than the grand mean (R^2 restricted to that
    position's held-out trials, summed over all components). High = that position is distinctly encoded."""
    posn = [POSITION_NAMES[c] for c in DISPLAY_ORDER]; pos = np.array(DISPLAY_ORDER)
    fig, ax = plt.subplots(figsize=(11, 5.5)); x = np.arange(6); w = 0.8 / max(len(labels), 1)
    summary = {}
    for i, lab in enumerate(labels):
        # ev_per_position, NOT a fourth inline copy. This function kept its own until 2026-08-22,
        # and so reported PS94_0820 far_R = 1.0 -- a perfect score for a position with ZERO engaged
        # trials -- for a full day after the shared implementation had been fixed. The consolidation
        # that created ev_per_position folded in fig_ev_by_position_animal and the EV matrix and
        # missed this one, which is exactly how a duplicate outlives the fix to its twin.
        ev = ev_per_position(lab)
        ax.bar(x + (i - (len(labels) - 1) / 2) * w, ev, w, label=lab[:4])
        for k in range(6):                      # a position the animal never visited: say so
            if not np.isfinite(ev[k]):
                ax.text(x[k] + (i - (len(labels) - 1) / 2) * w, 0, "n/a", rotation=90,
                        ha="center", va="bottom", fontsize=6, color="#888888")
        summary[lab] = {posn[k]: round(float(ev[k]), 3) for k in range(6)}
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(posn, rotation=45, ha="right")
    ax.set_ylabel("explained variance (held-out R^2, per position)"); ax.legend(fontsize=9, title="session")
    ax.set_title(f"Encoder: total explained variance per spout position, per session ({tag})")
    fig.tight_layout(); p = out / f"locanmf_encoder_ev_by_position_{tag}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    for lab, d in summary.items():
        print(f"  {lab}: {d}")
    return p


def fig_ev_ceiling_by_position(labels, out, tag):
    """Unified per-position (whole-cortex) explained variance RELATIVE TO CEILING. Left: noise ceiling
    (explainable var) per position; right: captured/ceiling (1 = encoder captures all explainable).
    Center positions often have ~0 ceiling (activity ~ grand mean) -> low raw EV is no signal, not failure."""
    posn = [POSITION_NAMES[c] for c in DISPLAY_ORDER]; pos = np.array(DISPLAY_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5)); w = 0.8 / max(len(labels), 1)
    for i, lab in enumerate(labels):
        s = _sess(lab)
        X, y, g, _, _, _ = _trial_features(s, _args(2.0))
        P = np.stack([(y == p).astype(float) for p in pos], 1)
        pred = np.zeros_like(X); ng = min(5, int(np.unique(g).size))
        for tr, te in GroupKFold(ng).split(X, y, g):
            pred[te] = Ridge(alpha=1.0).fit(P[tr], X[tr]).predict(P[te])
        xbar = X.mean(0); ceil = []; cap = []
        for p in pos:
            m = y == p
            if not m.any():
                # DELIBERATE nan, not an accidental one: the animal never went here, so
                # there is no ceiling and no capture to report. Recorded as missing so the
                # figure shows a gap rather than a fabricated zero.
                ceil.append(np.nan); cap.append(np.nan)
                continue
            mu = X[m].mean(0)
            betw = m.sum() * ((mu - xbar) ** 2).sum(); tot = betw + ((X[m] - mu) ** 2).sum()
            ceil.append(betw / max(tot, 1e-12)); cap.append(1 - ((X[m] - pred[m]) ** 2).sum() / max(tot, 1e-12))
        ceil = np.array(ceil); ratio = np.where(ceil > 0.05, np.array(cap) / ceil, np.nan)
        x = np.arange(6) + (i - (len(labels) - 1) / 2) * w
        axes[0].bar(x, ceil, w, label=lab[:4]); axes[1].bar(x, ratio, w, label=lab[:4])
    for ax, t in [(axes[0], "noise ceiling (explainable var) per position"),
                  (axes[1], "captured / ceiling per position (1 = all explainable captured)")]:
        ax.set_xticks(range(6)); ax.set_xticklabels(posn, rotation=45, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10); ax.legend(fontsize=8); ax.axhline(0, color="k", lw=0.5)
    axes[1].axhline(1, color="grey", ls="--", lw=0.8)
    fig.suptitle(f"Encoder explained variance per spout position: ceiling vs ceiling-normalized ({tag}; "
                 f"ratio shown where ceiling>0.05)", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_encoder_ev_ceiling_by_position_{tag}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_encoder_vs_svd(label, out):
    """VALIDATION: encoder (LocaNMF reconstruction) vs SVD pixel per-position map, matched cue-aligned
    pre-cue delta -> the only difference is the LocaNMF basis. Per-position spatial r quantifies fidelity."""
    s = _sess(label); mc = s["mc"]; W = int(round(FS))
    ad = glob.glob(f"{mc}/wfield_local_results/allen_aligned_affine8v1")[0]
    U = np.load(f"{ad}/U_atlas.npy"); SVT = np.load(config.svtcorr_path(mc))
    mask = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool); H, Wd = mask.shape; mk = mask.reshape(-1)
    Uf = U.reshape(-1, U.shape[2])
    Ar = np.load(f"{config.locanmf_dir(mc)}/{label}_locanmf_A.npy"); Af = np.nan_to_num(Ar.reshape(-1, Ar.shape[2]))
    C = np.load(f"{config.locanmf_dir(mc)}/{label}_locanmf_C.npy"); T = SVT.shape[1]
    cue = _load_cue(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, _ = _frames(s, cue, lk); codes = classify_cues_with_backup(s, cue)
    keep = [k for k in range(cf.size) if codes[k] >= 0 and int(cf[k]) - W >= 0 and int(cf[k]) + W <= T]
    cfk = np.array([int(cf[k]) for k in keep]); yk = np.array([int(codes[k]) for k in keep])
    dSVT = np.array([SVT[:, c:c + W].mean(1) - SVT[:, c - W:c].mean(1) for c in cfk])
    dC = np.array([C[:, c:c + W].mean(1) - C[:, c - W:c].mean(1) for c in cfk])
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    fig, axes = plt.subplots(2, 6, figsize=(20, 7)); rs = []
    for j, p in enumerate(DISPLAY_ORDER):
        m = yk == p; svd = Uf @ dSVT[m].mean(0); loc = Af @ dC[m].mean(0)
        r = np.corrcoef(svd[mk], loc[mk])[0, 1]; rs.append(r); vmax = np.nanpercentile(np.abs(svd[mk]), 99)
        for row, img, tg in [(0, svd, "SVD pixel"), (1, loc, "LocaNMF recon")]:
            im = img.reshape(H, Wd).astype(float); im[~mask] = np.nan
            ax = axes[row][j]; ax.imshow(im[y0:y1, x0:x1], cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{POSITION_NAMES[p]}\nr={r:.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(tg, fontsize=10)
    fig.suptitle(f"{label}: encoder (LocaNMF recon, bottom) vs SVD pixel (top) per position — cue-aligned pre-cue delta "
                 f"(median r={np.median(rs):.3f})", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_encoder_vs_svd_{label}.png"; fig.savefig(p, dpi=120); plt.close(fig)
    return p


def _region_feve(label):
    """Per-region FEVE substrate (fraction of EXPLAINABLE variance the encoder explains), computed at the
    REGION level by summing sums-of-squares over that region's components (not by averaging per-component
    fractions). For each LocaNMF component (2 s post-lick, no-baseline, block-CV ridge position->activity):
      sstot = Σ_trials (X - grand_mean)^2 ;  ssres = Σ (X - pred)^2 ;  betw = Σ_pos n_pos (mu_pos - gm)^2.
    EXPLAINABLE SS (noise ceiling, the most a position-only model could ever explain) = betw.
    CAPTURED SS (what the encoder actually explains) = sstot - ssres.
    Region FEVE = Σ_comp captured / Σ_comp explainable  (1.0 = encoder captures ALL explainable variance).
    Returns region -> dict(expl, cap, tot, n)."""
    s = _sess(label); names = _names(s)
    X, y, g, _, _, reg = _trial_features(s, _args(2.0))
    pos = np.array(DISPLAY_ORDER); P = np.stack([(y == p).astype(float) for p in pos], 1)
    pred = np.zeros_like(X); ng = min(5, int(np.unique(g).size))
    for tr, te in GroupKFold(ng).split(X, y, g):
        pred[te] = Ridge(alpha=1.0).fit(P[tr], X[tr]).predict(P[te])
    gm = X.mean(0); sstot = ((X - gm) ** 2).sum(0); ssres = ((X - pred) ** 2).sum(0)
    betw = np.zeros(X.shape[1])
    for p in pos:
        mm = y == p
        if not mm.any():                 # see the note in _ceiling: 0 * nan = nan
            continue
        mu = X[mm].mean(0); betw += mm.sum() * (mu - gm) ** 2
    cap = sstot - ssres                                    # captured SS per component
    rn = np.array([names.get(int(reg[i]), "?") for i in range(X.shape[1])])
    out = {}
    for r in sorted(set(rn)):
        m = rn == r
        if m.sum() < 1:                       # include every region with >=1 component (single-comp = noisier,
            continue                          # flagged via n in the figures, not dropped -> consistent region sets)
        out[r] = dict(expl=float(betw[m].sum()), cap=float(cap[m].sum()), tot=float(sstot[m].sum()), n=int(m.sum()))
    return out


def _feve_regions(res, floor):
    """Common region axis across sessions/animals: pool explainable + total SS over ALL provided sessions,
    keep regions whose pooled explainable FRACTION (expl/tot) exceeds `floor` (i.e. there is non-trivial
    position-explainable signal to normalize against), sorted by pooled explainable variance (desc)."""
    pe, pt = defaultdict(float), defaultdict(float)
    dropped = 0
    for d in res.values():
        for r, v in d.items():
            # DEFENCE IN DEPTH. The arithmetic that produced nan here is fixed above, but a
            # single non-finite value silently deletes the ENTIRE region axis (nan > floor
            # is False for every region), which is far too much damage for one bad session
            # to do. Skip the contribution and count it instead.
            if not (np.isfinite(v["expl"]) and np.isfinite(v["tot"])):
                dropped += 1
                continue
            pe[r] += v["expl"]; pt[r] += v["tot"]
    if dropped:
        print(f"  !! {dropped} non-finite region contribution(s) skipped when building the "
              f"FEVE axis -- a session produced nan SS; the axis is built without it.",
              flush=True)
    regs = [r for r in pe if pt[r] > 0 and pe[r] / pt[r] > floor]
    return sorted(regs, key=lambda r: pe[r], reverse=True)


def _feve_pct(d, regs, floor):
    """Per-region FEVE in PERCENT for one (pooled) session-group dict region->summed-SS; NaN where that
    group's own explainable fraction is below `floor` (nothing meaningful to normalize against)."""
    out = []
    for r in regs:
        v = d.get(r)
        if v is None or v["expl"] <= 0 or v["expl"] / max(v["tot"], 1e-12) <= floor:
            out.append(np.nan)
        else:
            out.append(100.0 * v["cap"] / v["expl"])
    return np.array(out)


def _pool_region(dicts):
    """Sum per-region SS across a list of per-session region dicts -> one pooled region dict."""
    agg = defaultdict(lambda: dict(expl=0.0, cap=0.0, tot=0.0, n=0))
    for d in dicts:
        for r, v in d.items():
            a = agg[r]; a["expl"] += v["expl"]; a["cap"] += v["cap"]; a["tot"] += v["tot"]; a["n"] += v["n"]
    return agg


def _feve_label_colors(regs):
    return ["#c0392b" if r.startswith(("SSp", "SSs")) else "#2980b9" if r.startswith("MO") else "#333" for r in regs]


def _feve_heatmap(ax, M, rows, regs, fig):
    """Shared FEVE heatmap: rows x regions, cell = FEVE% (0-100 viridis; <0 and NaN greyed), annotated."""
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#dddddd")
    Mc = np.where(M < 0, np.nan, M)                                    # below-0 FEVE shown as 'no capture' grey
    im = ax.imshow(Mc, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(regs))); ax.set_xticklabels(regs, rotation=60, ha="right", fontsize=6.5)
    for t, c in zip(ax.get_xticklabels(), _feve_label_colors(regs)):
        t.set_color(c)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(regs)):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.2,
                    color="white" if (0 <= v < 60) else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01); cb.set_label("FEVE — % of explainable variance captured", fontsize=8)
    return im


#: A FEVE figure with fewer regions than this, built from sessions that did have data, is
#: degenerate rather than informative -- the region axis has collapsed, not the biology.
MIN_FEVE_REGIONS = 10


def _degenerate_feve(res, regs, out_path, verbose=True) -> bool:
    """Is this about to replace a good figure with a collapsed one?

    The pooled FEVE figure is rebuilt per-day by the nightly, so a run in which most sessions fail
    to load overwrites a 64-region heatmap with a one-region strip -- and renders it perfectly
    happily. Slide 21 read as "empty" for exactly that reason (Priya, 2026-08-20); on the same
    inputs a manual run recovers 64 regions, so the collapse is a property of that RUN, not of the
    data, and it must not be allowed to land silently.

    Returns True (and says so) when the figure should not be written.
    """
    if len(regs) >= MIN_FEVE_REGIONS or not res:
        return False
    if verbose:
        print(f"  !! FEVE region axis collapsed to {len(regs)} region(s) from {len(res)} session(s) "
              f"-- REFUSING to overwrite {os.path.basename(str(out_path))}. Something failed to load "
              f"in this run; re-run the encoder alone and check the per-session skips.", flush=True)
    return True


def _feve_name(f) -> str:
    """The figure a FEVE builder writes, for log lines where it refused and returned None."""
    return ("locanmf_encoder_feve_by_region_pooled.png" if f is fig_region_feve_pooled
            else "locanmf_encoder_feve_by_region_sessions.png")


def _feve_log(f, q) -> str:
    """What to say about a FEVE builder that returned ``q``.

    A builder returns None when the degenerate-collapse guard refused to overwrite a good figure.
    Reporting that as "wrote" is how a refusal came to read as a success in the 2026-08-21 nightly
    log -- "REFUSING to overwrite X" on one line, "wrote X" on the next. The file on disk was
    intact; only the report lied, which is the harder kind to catch, so the decision lives here
    where a test can hold it.
    """
    return (f"wrote {q.name}" if q else
            f"kept existing (guard refused to overwrite): {_feve_name(f)}")


def fig_region_feve_pooled(res, out, floor=0.02):
    """ACROSS-SESSIONS: per region, FEVE (% of explainable/ceiling variance the encoder captures) pooled
    within each animal (SS summed over all of that animal's sessions). Heatmap animal x region; 100% = the
    encoder captures all position-explainable variance in that region. Grey = <0 (no capture) or no
    explainable signal. Regions sorted by pooled explainable variance; SSp red / MO blue x-labels."""
    by = defaultdict(list)
    for lab in res:
        by[lab[:4]].append(lab)
    animals = sorted(by); regs = _feve_regions(res, floor)
    # REFUSE BEFORE PLOTTING, not after. The degenerate check used to sit below the
    # figure build, so an empty axis died inside imshow with "Invalid shape (0,) for
    # image data" instead of being declined -- a guard that only works when there is
    # something to guard. Seen 2026-08-22 when a mistyped --pool-dates matched no
    # session at all.
    if not animals or not regs:
        print(f"  !! locanmf_encoder_feve_by_region_pooled.png: nothing to plot "
              f"({len(animals)} animal(s), {len(regs)} region(s)) from {len(res)} "
              f"session(s) -- NOT written, the existing figure is left alone.",
              flush=True)
        return None
    if _degenerate_feve(res, regs, out / "locanmf_encoder_feve_by_region_pooled.png"):
        return None
    pooled = {a: _pool_region([res[l] for l in by[a]]) for a in animals}
    rows = [f"{a} (n={len(by[a])})" for a in animals]
    M = np.array([_feve_pct(pooled[a], regs, floor) for a in animals])
    fig, ax = plt.subplots(figsize=(max(12, 0.30 * len(regs)), 0.6 * len(animals) + 2.6))
    _feve_heatmap(ax, M, rows, regs, fig)
    ax.set_title(f"Encoder FEVE by region — pooled per animal across ALL sessions "
                 f"(100% = encoder captures all position-explainable variance; SSp red / MO blue labels; "
                 f"regions with explainable frac >{floor}, sorted by explainable variance)", fontsize=9.5)
    fig.tight_layout(); p = out / "locanmf_encoder_feve_by_region_pooled.png"; fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig_region_feve_sessions(res, out, floor=0.02):
    """INDIVIDUAL SESSIONS: same FEVE-by-region heatmap, one row per session (grouped by animal) -> shows
    session-to-session stability of each region's ceiling-normalized encoding within an animal."""
    by = defaultdict(list)
    for lab in res:
        by[lab[:4]].append(lab)
    animals = sorted(by); regs = _feve_regions(res, floor)
    # REFUSE BEFORE PLOTTING, not after. The degenerate check used to sit below the
    # figure build, so an empty axis died inside imshow with "Invalid shape (0,) for
    # image data" instead of being declined -- a guard that only works when there is
    # something to guard. Seen 2026-08-22 when a mistyped --pool-dates matched no
    # session at all.
    if not animals or not regs:
        print(f"  !! locanmf_encoder_feve_by_region_sessions.png: nothing to plot "
              f"({len(animals)} animal(s), {len(regs)} region(s)) from {len(res)} "
              f"session(s) -- NOT written, the existing figure is left alone.",
              flush=True)
        return None
    if _degenerate_feve(res, regs, out / "locanmf_encoder_feve_by_region_sessions.png"):
        return None
    labs = [l for a in animals for l in sorted(by[a], key=lambda x: x[-4:])]
    rows = [f"{l[:4]} {l[-4:-2]}/{l[-2:]}" for l in labs]
    M = np.array([_feve_pct(res[l], regs, floor) for l in labs])
    fig, ax = plt.subplots(figsize=(max(12, 0.30 * len(regs)), 0.42 * len(labs) + 2.6))
    _feve_heatmap(ax, M, rows, regs, fig)
    # separators between animals
    seen = 0
    for a in animals[:-1]:
        seen += len(by[a]); ax.axhline(seen - 0.5, color="k", lw=1.2)
    ax.set_title(f"Encoder FEVE by region — individual sessions (grouped by animal) "
                 f"(% of explainable variance captured; 100% = all; regions explainable frac >{floor})", fontsize=9.5)
    fig.tight_layout(); p = out / "locanmf_encoder_feve_by_region_sessions.png"; fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig_quiet_drift(labels, out, tag):
    """Time-local quiet (rest) baseline over the session, pooled over components, per session. Shows the
    slow drift (photobleaching/state) the time-local baseline tracks; small in dF/F, but the right
    reference for long continuous sessions and the pre/post-stroke residual."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for lab in labels:
        s = _sess(lab); sig, _ = _build_signal(s, "locanmf"); T = sig.shape[1]
        pooled = _quiet_baseline_local(s, sig).mean(0)
        ax.plot(np.arange(T) / FS / 60, pooled, label=f"{lab[:4]} ({T / FS / 60:.0f}min)")
    ax.set_xlabel("session time (min)"); ax.set_ylabel("pooled quiet (rest) baseline (dF/F)")
    ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8)
    ax.set_title(f"Time-local quiet baseline drift over session ({tag}) — small in dF/F; time-local tracks any residual")
    fig.tight_layout(); p = out / f"locanmf_encoder_quiet_drift_{tag}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p


def ev_per_position(label):
    """Held-out per-position encoder EV for ONE session, as a length-6 array over DISPLAY_ORDER.

    ONE implementation: `fig_ev_by_position_animal` had this inline and the EV matrix needed the
    same numbers. Ridge from a one-hot position design onto the feature matrix, scored per position
    with GroupKFold over blocks -- the same CV the decoders use, so the two are comparable.
    """
    pos = np.array(DISPLAY_ORDER)
    X, y, g, _, _, _ = _trial_features(_sess(label), _args(2.0))
    P = np.stack([(y == p).astype(float) for p in pos], 1)
    pred = np.zeros_like(X)
    ng = min(5, int(np.unique(g).size))
    for tr, te in GroupKFold(ng).split(X, y, g):
        pred[te] = Ridge(alpha=1.0).fit(P[tr], X[tr]).predict(P[te])
    xbar = X.mean(0)
    sstot = ((X - xbar) ** 2).sum(1)
    ssres = ((X - pred) ** 2).sum(1)
    # A POSITION WITH NO TRIALS HAS NO EV, AND MUST NOT SCORE 1.0.
    # Both sums are 0.0 over an empty mask, so `1 - 0.0/max(0.0, 1e-12)` is exactly 1.0: a PERFECT
    # score for a position the animal never visited. PS94_0820 reported far_R = 1.0 that way with
    # zero engaged trials there, beside neighbours between -0.04 and +0.21 -- and this feeds the
    # per-position x date EV matrix, so the brightest cell in the figure sat exactly where the
    # animal had abandoned the spout. An abandoned position is the post-stroke phenotype; reporting
    # it as perfectly encoded inverts the reading.
    out = []
    for p in pos:
        m = y == p
        out.append(1 - ssres[m].sum() / max(sstot[m].sum(), 1e-12) if m.any() else np.nan)
    return np.array(out)


def fig_ev_matrix(by_animal, out, name="locanmf_encoder_ev_matrix.png"):
    """Encoded variance per POSITION x SESSION, one panel per animal, on ONE colour scale.

    The per-animal bar charts this summarises put six grouped bars per position on one axis and
    asked the reader to track a position across sessions by colour. As a matrix, a position that
    degrades over days is a column that changes colour, and the lesion is a horizontal rule.

    ONE COLOUR SCALE across animals, so panels are comparable -- which is the whole point of a
    summary and is exactly what the per-session figures could not offer.
    """
    from wfield_local import config

    posn = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    animals = sorted(by_animal)
    rows = {a: sorted(by_animal[a], key=lambda x: x[-4:]) for a in animals}
    M = {a: np.full((len(rows[a]), len(posn)), np.nan) for a in animals}
    for a in animals:
        for i, lab in enumerate(rows[a]):
            try:
                M[a][i] = ev_per_position(lab)
            except Exception as ex:                                   # noqa: BLE001
                print(f"  ev_matrix {lab}: skip ({type(ex).__name__} {str(ex)[:50]})", flush=True)
    finite = np.concatenate([M[a][np.isfinite(M[a])] for a in animals]) if animals else np.array([])
    if not finite.size:
        return None
    vmax = float(np.nanpercentile(finite, 98))
    vmin = min(0.0, float(np.nanpercentile(finite, 2)))

    fig, axes = plt.subplots(1, len(animals), figsize=(3.15 * len(animals) + 1.4, 5.6),
                             squeeze=False)
    for k, a in enumerate(animals):
        ax = axes[0][k]
        # Grey, not transparent: a position the animal abandoned must LOOK missing rather
        # than blend into the page. Matches the FEVE heatmap's bad colour.
        _cm = plt.cm.viridis.copy(); _cm.set_bad("#dddddd")
        im = ax.imshow(np.ma.masked_invalid(M[a]), cmap=_cm, vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(posn)))
        ax.set_xticklabels(posn, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows[a])))
        ax.set_yticklabels([l[-4:] for l in rows[a]], fontsize=6.5)
        # the lesion, as a rule between the last pre-stroke row and the first post-stroke one
        phases = [config.session_phase(a, l[-4:]) for l in rows[a]]
        for i in range(1, len(phases)):
            if phases[i - 1] == "pre" and phases[i] != "pre":
                ax.axhline(i - 0.5, color="firebrick", lw=2.0)
                ax.text(len(posn) - 0.4, i - 0.5, " lesion", color="firebrick", fontsize=6.5,
                        va="center")
        for i, ph in enumerate(phases):        # excluded sessions belong to neither phase
            if ph == "excluded":
                ax.text(-0.9, i, "excl", fontsize=5.5, color="grey", va="center", ha="right")
        ax.set_title(a, fontsize=10, fontweight="bold")
        if k == 0:
            ax.set_ylabel("session (MMDD)", fontsize=8)
    # DEDICATED AXES on the right, not `ax=axes[...]`, which steals width from the panels and lets
    # the bar land on top of them (Priya, 2026-08-20).
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.905, 0.18, 0.015, 0.62])
    fig.colorbar(im, cax=cax, label="encoder held-out EV (per position)")
    fig.suptitle(
        "ENCODED VARIANCE PER SPOUT POSITION, per session. Ridge from a one-hot position design onto "
        "the feature matrix, scored per position with the same block GroupKFold the decoders use. "
        "ONE colour scale across all animals, so the panels are comparable -- which the per-session "
        "bar charts could not offer. A position that degrades across days is a COLUMN that changes "
        "colour; the red rule is the lesion.", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 0.88, 0.90))
    p = out / name
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_ev_by_position_animal(animal, labels, out):
    """Per-position held-out EV for ONE animal across its sessions — one graph, sessions distinguished by
    colour (grouped bars), labelled by date. The per-animal-across-sessions view of fig_ev_by_position."""
    posn = [POSITION_NAMES[c] for c in DISPLAY_ORDER]; pos = np.array(DISPLAY_ORDER)
    fig, ax = plt.subplots(figsize=(11, 5.5)); x = np.arange(6); w = 0.8 / max(len(labels), 1)
    for i, lab in enumerate(labels):
        ev = ev_per_position(lab)          # one implementation, shared with fig_ev_matrix
        ax.bar(x + (i - (len(labels) - 1) / 2) * w, ev, w, label=lab[5:9])
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(posn, rotation=45, ha="right")
    ax.set_ylabel("explained variance (held-out R^2, per position)"); ax.legend(fontsize=9, title="session")
    ax.set_title(f"{animal} — encoder explained variance per spout position, across sessions")
    fig.tight_layout(); p = out / f"locanmf_encoder_ev_by_position_animal_{animal}.png"
    fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_ev_ceiling_by_position_animal(animal, labels, out):
    """Per-position EV RELATIVE TO CEILING for ONE animal across sessions (one graph, sessions by colour).
    Left: noise ceiling (explainable var) per position; right: captured/ceiling. Per-animal fig_ev_ceiling."""
    posn = [POSITION_NAMES[c] for c in DISPLAY_ORDER]; pos = np.array(DISPLAY_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5)); w = 0.8 / max(len(labels), 1)
    for i, lab in enumerate(labels):
        X, y, g, _, _, _ = _trial_features(_sess(lab), _args(2.0))
        P = np.stack([(y == p).astype(float) for p in pos], 1)
        pred = np.zeros_like(X); ng = min(5, int(np.unique(g).size))
        for tr, te in GroupKFold(ng).split(X, y, g):
            pred[te] = Ridge(alpha=1.0).fit(P[tr], X[tr]).predict(P[te])
        xbar = X.mean(0); ceil = []; cap = []
        for p in pos:
            m = y == p; mu = X[m].mean(0)
            betw = m.sum() * ((mu - xbar) ** 2).sum(); tot = betw + ((X[m] - mu) ** 2).sum()
            ceil.append(betw / max(tot, 1e-12)); cap.append(1 - ((X[m] - pred[m]) ** 2).sum() / max(tot, 1e-12))
        ceil = np.array(ceil); ratio = np.where(ceil > 0.05, np.array(cap) / ceil, np.nan)
        x = np.arange(6) + (i - (len(labels) - 1) / 2) * w
        axes[0].bar(x, ceil, w, label=lab[5:9]); axes[1].bar(x, ratio, w, label=lab[5:9])
    for ax, t in [(axes[0], "noise ceiling (explainable var) per position"),
                  (axes[1], "captured / ceiling per position (1 = all explainable captured)")]:
        ax.set_xticks(range(6)); ax.set_xticklabels(posn, rotation=45, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10); ax.legend(fontsize=8, title="session"); ax.axhline(0, color="k", lw=0.5)
    axes[1].axhline(1, color="grey", ls="--", lw=0.8)
    fig.suptitle(f"{animal} — encoder EV per position across sessions: ceiling vs ceiling-normalized "
                 f"(ratio shown where ceiling>0.05)", fontsize=12)
    fig.tight_layout(); p = out / f"locanmf_encoder_ev_ceiling_by_position_animal_{animal}.png"
    fig.savefig(p, dpi=130); plt.close(fig); return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--date", default="0605")
    ap.add_argument("--pool-dates", default="", help="comma MMDD subset for the pooled FEVE figures "
                    "(default empty = ALL sessions; e.g. 0605,0606,0607,0608 for the 6/5-onward deck)")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for s in [x for x in SESSIONS if x["label"].endswith(args.date)]:
        lab = s["label"]
        try:
            e = encode_spatial(lab)
            print(f"{lab}: CV encoding R^2 mean={e['cv_r2']:.3f}", flush=True)
            for f in (fig_predicted_maps, fig_encoding_r2, fig_temporal_encoder, fig_encoder_vs_svd):
                print("  wrote", f(lab, args.output).name, flush=True)
        except Exception as ex:
            print(f"{lab}: FAILED {type(ex).__name__}: {str(ex)[:80]}", flush=True)
    labs = [x["label"] for x in SESSIONS if x["label"].endswith(args.date)]
    for f in (fig_ev_by_position, fig_ev_ceiling_by_position, fig_quiet_drift):
        try:
            print("wrote", f(labs, args.output, args.date).name, flush=True)
        except Exception as ex:
            print(f"{f.__name__}: FAILED {type(ex).__name__}: {str(ex)[:80]}", flush=True)
    # FEVE by region — pooled-per-animal and per-session. Default ALL sessions; --pool-dates restricts
    # (the 6/5-onward deck pools only 6/5-6/8, which share each animal's consistent 6/6-referenced alignment).
    pool = set(args.pool_dates.split(",")) if args.pool_dates else None
    all_labs = [x["label"] for x in SESSIONS if pool is None or x["label"][-4:] in pool]
    res = {}
    for lab in all_labs:
        try:
            res[lab] = _region_feve(lab)
        except Exception as ex:
            print(f"  {lab} feve skip: {type(ex).__name__}: {str(ex)[:60]}", flush=True)
    for f in (fig_region_feve_pooled, fig_region_feve_sessions):
        try:
            print(_feve_log(f, f(res, args.output)), flush=True)
        except Exception as ex:
            print(f"{f.__name__}: FAILED {type(ex).__name__}: {str(ex)[:80]}", flush=True)
    # per-animal EV-per-position across the pooled sessions (one graph/animal, sessions by colour)
    by_animal = {}
    for lab in all_labs:
        by_animal.setdefault(lab[:4], []).append(lab)
    # THE EV MATRIX HAD NO CALLER. It was written, the deck renders it `if exists()`, and nothing
    # in the repo produced it -- so the slide was served by a PNG from 2026-08-19 and would have
    # gone on aging silently, because "figure missing" and "figure stale" look identical to a deck
    # that only checks existence. Found 2026-08-22 while re-running the encoder.
    try:
        q = fig_ev_matrix(by_animal, args.output)
        print("wrote" if q else "ev_matrix: nothing finite to plot", q.name if q else "", flush=True)
    except Exception as ex:                                       # noqa: BLE001
        print(f"fig_ev_matrix: FAILED {type(ex).__name__}: {str(ex)[:80]}", flush=True)
    for a, labs_a in sorted(by_animal.items()):
        for f in (fig_ev_by_position_animal, fig_ev_ceiling_by_position_animal):
            try:
                print("wrote", f(a, sorted(labs_a), args.output).name, flush=True)
            except Exception as ex:
                print(f"{f.__name__} {a}: FAILED {type(ex).__name__}: {str(ex)[:70]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
