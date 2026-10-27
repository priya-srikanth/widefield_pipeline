"""WHERE IN CORTEX the same-place read-out offset comes from, per decoder basis (Priya 10-27: "run the chronic offset
map"; why does the chronic offset persist in Allen areas / LocaNMF but not in the pre-stroke SVD basis?).

The frozen pre-stroke lateral decoder is linear in its (scaled) features h: dec = b0 + beta . h, with beta = ridge coef /
StandardScaler scale (iti_lick_order.predictions, all-pre fit). Per lick, the like-for-like change is
  dh_i = h_i - sum_j K(act_i - act_j) h_j / sum_j K       (j = the animal's pre licks of the same set; Gaussian, SIGMA)
so the offset = beta . dh exactly (same-place Gaussian reference as lick_matched_readout.py --ref gauss, but every lick
scored by the ALL-pre decoder instead of leave-one-pre-session-out; the sanity column compares the two).
Per basis the features are f = M x of pixel activity x, with a spatial matrix B such that M B = I:
  areas    M = area means (stat mask, iti_licks.roi_masks), B = area indicators
  svd      M = P' (the animal's pre-stroke spatial basis, orthonormal), B = P
  locanmf  M = s * pinv(A) (joint pre-stroke LocaNMF footprints A, footprint scale s), B = A / s
(each further divided by its per-session scaling sd, folded into h). Pixel CONTRIBUTION MAP
  c(p) = w(p) * dx(p),  w = M' beta (decoder weight per pixel),  dx = B dh (activity change at matched landing)
sums over pixels to beta . dh = the offset. Split: GLOBAL part = sum_p w(p) * mean(dx) (a uniform cortex-wide change)
vs PATTERN part = the rest. Per animal: epoch mean of session means of dh, minus the same for pre.
Outputs <E>/iti_licks/offset_contrib/{summary_<tag>.csv, maps_<tag>.npz}, figures/lick_offset_contrib_map.png
(figdata = the npz + csv; --plot-only redraws).
Usage: WFIELD_LICK_DECODER=<areas|svd|locanmf> [WFIELD_LOCANMF_MASK=1] python scripts/lick_offset_contrib_map.py
       python scripts/lick_offset_contrib_map.py --plot-only
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = il.OUT / "offset_contrib"
SIGMA = 0.1                      # mm, the held-out-chosen Gaussian width of the same-place reference
EP = ["acute", "subacute", "chronic"]
AN = ["PS92", "PS93", "PS94", "PS95"]
H, W = 540, 640
ACC: dict = {}


def _hook(st, a, S, fit):
    sc, m = fit
    beta = m.coef_ / sc.scale_
    pre = [x for x, v in S.items() if v[0] == "pre"]
    Xp = np.vstack([S[x][1] for x in pre])
    ap = np.concatenate([S[x][2] for x in pre])
    sess = {}
    for x, (ep, X, act, _o, _p) in S.items():
        K = np.exp(-0.5 * ((act[:, None] - ap[None, :]) / SIGMA) ** 2)
        if x in pre:                                   # a pre session is referenced to the OTHER pre sessions
            own = np.concatenate([np.full(len(S[q][2]), q == x) for q in pre])
            K[:, own] = 0
        ok = K.sum(1) > 1e-6
        if ok.sum() < 8:
            continue
        dh = X[ok] - (K[ok] @ Xp) / K[ok].sum(1, keepdims=True)
        sess[x] = (ep, dh.mean(0))
    ACC[(st, a)] = (beta, sess)


def _grid(a: str):
    """flat brain-pixel indices, stat mask over them, area masks over them (one pre session of the animal)."""
    from iti_licks import roi_masks
    from wfield_local import combined_encoding as ce
    lab = sorted(p.stem for p in (il.OUT / "sessions").glob(f"{a}_*.npz"))[0]
    _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(lab)
    if tuple(int(s) for s in shape) != (H, W):
        raise ValueError(f"{a}: imaging grid {shape} != {(H, W)}")
    return flat, pstat, roi_masks(parea, pstat, names, flat, shape)


def spatial(source: str, a: str, keep=None):
    """(B (npix, K), Mt (npix, K), pixel flat indices on the H x W grid) for one animal and basis."""
    if source == "areas":
        from iti_decoder_contrib import feature_names
        flat, _pstat, M = _grid(a)
        names = feature_names()
        B = np.stack([M[n] for n in names], 1).astype(float)          # indicators over brain pixels
        Mt = B / B.sum(0, keepdims=True)
        return B, Mt, flat
    if source == "svd":
        from iti_basis_decoder import SPATIAL
        flat, pstat, _M = _grid(a)
        P = np.asarray(pickle.loads((SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], float)
        return P, P, flat[pstat]
    from wfield_local import joint_locanmf as jl
    from wfield_local.joint_locanmf import _footprint_scale
    b = jl.load(a, warn_stale=False)
    Araw = np.asarray(b.A, float).reshape(-1, b.ncomp)
    inside = np.isfinite(Araw).any(1)
    A = np.nan_to_num(Araw[inside])
    s = _footprint_scale(np.asarray(b.A), b.ncomp)
    Mt = (A @ np.linalg.pinv(A.T @ A)) * s[None, :]                    # (s pinv(A))'
    B = A / s[None, :]
    if keep is not None:
        B, Mt = B[:, keep], Mt[:, keep]
    return B, Mt, np.flatnonzero(inside)


def compute():
    import os

    import iti_lick_order as lo
    source = lo.SOURCE
    lo.CONTRIB_HOOK = _hook
    P = lo.predictions()                                # fills ACC; P = the LOSO predictions (sanity)
    rows, maps = [], {}
    for (st, a), (beta, sess) in sorted(ACC.items()):
        keep = lo.locanmf_inmask(a) if lo.MASK_LOC else None
        B, Mt, pix = spatial(source, a, keep)
        if B.shape[1] != len(beta):
            raise ValueError(f"{a} {source}: {B.shape[1]} spatial columns vs {len(beta)} features")
        w = Mt @ beta
        by = {ep: np.mean([v for e, v in sess.values() if e == ep], 0) for ep in ["pre", *EP]
              if any(e == ep for e, _ in sess.values())}
        for ep in EP:
            if ep not in by:
                continue
            dh = by[ep] - by["pre"]
            dx = B @ dh
            c = w * dx
            glob = float(w.sum() * dx.mean())
            img = np.full(H * W, np.nan)
            img[pix] = c
            maps[(st, a, ep)] = img.reshape(H, W).astype(np.float32)
            wimg = np.full(H * W, np.nan)
            wimg[pix] = w
            maps[(st, a, "w")] = wimg.reshape(H, W).astype(np.float32)
            g = P[(P.set == st) & (P.animal == a)]
            rows.append(dict(source=lo.SRC_TAG.lstrip("_") or "areas", set=st, animal=a, epoch=ep,
                             offset=float(beta @ dh), check=float(c.sum()), global_part=glob,
                             pattern_part=float(c.sum()) - glob, w_sum=float(w.sum()), dx_mean=float(dx.mean()),
                             n_pre=sum(e == "pre" for e, _ in sess.values()),
                             n_ep=sum(e == ep for e, _ in sess.values()),
                             loso_dec_change=float(g[g.epoch == ep].dec.mean() - g[g.epoch == "pre"].dec.mean())))
    OUT.mkdir(exist_ok=True)
    tag = lo.SRC_TAG.lstrip("_") or "areas"
    np.savez_compressed(OUT / f"maps_{tag}.npz", **{"|".join(k): v for k, v in maps.items()})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / f"summary_{tag}.csv", index=False)       # one file per basis: runs go in parallel
    print(os.environ.get("WFIELD_LICK_DECODER"), tag)
    print(R.round(3).to_string())
    return R


TAGS = [("areas", "Allen areas"), ("locanmfF", "LocaNMF"), ("locanmfFM", "LocaNMF, glue / MOB comps dropped"),
        ("svdF", "pre-stroke SVD")]


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("summary_*.csv"))], ignore_index=True)
    tags = [(t, lab) for t, lab in TAGS if (OUT / f"maps_{t}.npz").exists()]
    for st, stlab in (("iti", "ITI licks"), ("trial", "cued licks")):
        fig, axs = plt.subplots(len(tags), 4, figsize=(15, 3.6 * len(tags)), squeeze=False)
        for i, (t, lab) in enumerate(tags):
            Z = np.load(OUT / f"maps_{t}.npz")
            ims = {ep: [Z[f"{st}|{a}|{ep}"] for a in AN if f"{st}|{a}|{ep}" in Z] for ep in [*EP, "w"]}
            mean = {ep: np.nanmean(np.stack(v), 0) for ep, v in ims.items() if v}
            lim = np.nanpercentile(np.abs(np.stack([mean[e] for e in EP if e in mean])), 99.5)
            for j, ep in enumerate(EP):
                ax = axs[i, j]
                if ep in mean:
                    im = ax.imshow(mean[ep], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                    r = R[(R.source == t) & (R.set == st) & (R.epoch == ep)]
                    ax.set_title(f"{lab}: {ep} - pre\noffset {r.offset.mean():+.2f} mm (global {r.global_part.mean():+.2f}, "
                                 f"pattern {r.pattern_part.mean():+.2f})", fontsize=8)
                ax.axis("off")
            plt.colorbar(im, ax=axs[i, 2], fraction=0.04, label="mm per pixel (+ = contralesional)")
            ax = axs[i, 3]
            wl = np.nanpercentile(np.abs(mean["w"]), 99.5)
            im = ax.imshow(mean["w"], cmap="PuOr_r", vmin=-wl, vmax=wl, interpolation="nearest")
            ax.set_title(f"{lab}: decoder weight per pixel (pre fit)", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle(f"NEURAL: where the same-place read-out offset comes from, {stlab}. Contribution per pixel = decoder "
                     "weight x activity change vs pre licks landing at the same place (Gaussian, 0.1 mm);\nsums to the "
                     "offset. Global = a uniform cortex-wide change x the summed weight; pattern = the rest. Mean of "
                     "animals; ipsi (lesion) = image LEFT. PROVISIONAL", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(il.OUT / "figures" / f"lick_{'iti' if st == 'iti' else 'cued'}_offset_contrib_map.png", dpi=90)
        plt.close(fig)


def main() -> int:
    if "--plot-only" not in sys.argv:
        compute()
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
