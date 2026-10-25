"""Musall-style EVENT-KERNEL beta maps of the FROZEN pre-stroke encoder (Priya 10-25: "have we been looking at beta weights
of event kernels" -> "show all 6 spout position kernels"; "lick onset and contact have a lot of venous contamination").
From the pixel (shared-SVD basis) frozen run: each animal's pre-stroke weights beta (columns x 200 basis targets + cortex)
mapped back to pixels through its basis P (stat pixels x 200, orthonormal): pixel kernel = P @ (beta[col, :200] /
scale[col]), dF/F per unit of the raw regressor. Event kernels use a raised-cosine basis (0.1 s spacing; the cue / target
kernels 0.4 s after 2 s), so column k is the kernel's value at lag = start + k * spacing. Mean of the 4 animals (same
Allen grid). VENOUS CONTAMINATION: the sagittal sinus runs along the midline; a band of MIDLINE_PX pixels either side of
the hemispheric boundary is masked (grey) -- crude (the transverse sinus and other vessels stay). Image left = ipsilesional.

Figures: kernel_maps.png (selected kernels), kernel_maps_target_6spouts.png (the target kernel of every spout = its
DEVIATION from the shared cue kernel, at several lags). Spout-specific LICK kernels are not in this model (one shared lick
kernel; spout shape enters through the templates) -> they come from the per-epoch kernel fits (`epoch_kernels.py`).
Usage: python scripts/frozen_kernel_maps.py [--tag gate-attempted_v7g3_t3d_frA_basis]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

POS = ["close_contra", "close_ipsi", "close_mid", "far_contra", "far_ipsi", "far_mid"]   # sorted spout order of the blocks
POS_SHOW = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]   # canonical spout order (orofacial_cohort.POS_ORDER)
NCUE = 35                                     # columns per cue / target kernel
MIDLINE_PX = 10
VESSEL_Q = 92.0           # vessel mask = pixels above this percentile of the mean vesselness template (within brain)
VESSEL_DILATE = 2
#: kernels shown with the vessel mask (the venous-looking ones); target / cue / FR maps are shown unmasked
VESSEL_PANELS = ("lick onset", "contact")


def cue_lag_index(t: float) -> int:
    """cue / target kernel columns: 0 .. 2 s every 0.1 s (21), then 2 .. 7 s every 0.4 s."""
    return int(round(t / 0.1)) if t <= 2.0 else 21 + int(round((t - 2.0) / 0.4)) - 1


PANELS = [("cue, 0.2 s", "cue", cue_lag_index(0.2)), ("cue, 1.0 s", "cue", cue_lag_index(1.0)),
          ("target far contra, 1.0 s", "pos_cue", POS.index("far_contra") * NCUE + cue_lag_index(1.0)),
          ("target far ipsi, 1.0 s", "pos_cue", POS.index("far_ipsi") * NCUE + cue_lag_index(1.0)),
          ("lick onset (spout-locked), 0 s", "lick_onset:spl", 5), ("lick onset (deviation), 0 s", "lick_onset:dev", 5),
          ("lick onset (deviation), +0.2 s", "lick_onset:dev", 7), ("contact, 0 s", "contact", 5),
          ("FaceRhythm lick", "fr_lick", 0), ("FaceRhythm right (contra)", "fr_right", 0),
          ("FaceRhythm left (ipsi)", "fr_left", 0), ("running (first hinge, lag 0)", "running", 0)]
TLAGS = (0.3, 0.6, 1.0, 2.0)
TARGET = [(f"{q.replace('_', ' ')}, {t:g} s", "pos_cue", POS.index(q) * NCUE + cue_lag_index(t)) for t in TLAGS for q in POS_SHOW]


def midline_mask(img: np.ndarray, names: list, px: int = MIDLINE_PX) -> np.ndarray:
    """True within px columns of the left / right hemisphere boundary in each row (sagittal sinus)."""
    hemi = np.zeros(img.shape)
    for i, n in enumerate(names):
        hemi[img == i] = -1 if n.endswith("_left") else (1 if n.endswith("_right") else 0)
    m = np.zeros(img.shape, bool)
    for r in range(img.shape[0]):
        row = hemi[r]
        L, R = np.flatnonzero(row < 0), np.flatnonzero(row > 0)
        if len(L) and len(R):
            mid = (L.max() + R.min()) / 2.0
            c = np.arange(img.shape[1])
            m[r] = np.abs(c - mid) <= px
    return m


def vessel_mask(shape) -> np.ndarray:
    """Union over animals of each animal's vessel mask: its pre-stroke sessions' vesselness templates (high-passed negated
    mean fluorescence on the atlas grid, `hemo_map_control.vessel_template`) averaged, thresholded at VESSEL_Q (within the
    brain), dilated by VESSEL_DILATE px. Cached in <E>/vessel_mask.npz."""
    from scipy import ndimage

    from wfield_local import config, epochs
    from wfield_local.hemo_map_control import vessel_template
    cache = E / "vessel_mask.npz"
    if cache.exists():
        z = np.load(cache)
        if tuple(z["mask"].shape) == tuple(shape):
            return z["mask"]
    sess = [s for s in config.load_sessions() if epochs.epoch_of(s["label"]) == "pre" and s["label"][-4:] >= "0806"]
    out = np.zeros(shape, bool)
    for an in AN:
        ts = [t for t in (vessel_template(s) for s in sess if s["label"].startswith(an)) if t is not None]
        if not ts:
            continue
        T = np.nanmean(ts, 0)
        brain = np.abs(T) > 0
        thr = np.nanpercentile(T[brain], VESSEL_Q)
        out |= ndimage.binary_dilation(T > thr, iterations=VESSEL_DILATE)
        print(f"  vessel mask {an}: {len(ts)} sessions, {int((T > thr).sum())} px", flush=True)
    np.savez_compressed(cache, mask=out)
    return out


def kernel_maps(run: Path, panels: list) -> tuple[dict, tuple]:
    from wfield_local import combined_encoding as ce
    maps, shape = {}, None
    for an in AN:
        fz, sp = run / "fit" / f"{an}_joint.npz", run / "fit" / f"spatial_{an}.pkl"
        if not fz.exists() or not sp.exists():
            continue
        z = np.load(fz, allow_pickle=True)
        P = pickle.loads(sp.read_bytes())["P"].astype(np.float64)
        K = P.shape[1]
        lab = sorted(p.stem for p in (run / "joint" / "sessions").glob(f"{an}_*.json"))[0]
        _V, _ft, _Ub, flat, _pa, _nm, shp, pstat = ce.load_imaging(lab)
        H, W = (int(x) for x in shp)
        shape = (H, W)
        pix = flat[pstat]
        cv = list(z["col_var"])
        for title, var, k in panels:
            idx = [i for i, v in enumerate(cv) if v == var]
            if k >= len(idx):
                continue
            c = idx[k]
            g = np.full(H * W, np.nan)
            g[pix] = P @ (z["beta_all"][c, :K] / z["scale"][c])
            maps.setdefault(title, []).append(g.reshape(H, W))
    return maps, shape


def draw(maps, titles, path, suptitle, ol, mask, ncol=4, shared_rows=False, mask_for=None):
    import matplotlib.pyplot as plt
    nrow = int(np.ceil(len(titles) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.1 * nrow), squeeze=False)
    Ms = {t: (np.where(mask, np.nan, np.nanmean(maps[t], 0)) if mask is not None and (mask_for is None or any(
        k in t for k in mask_for)) else np.nanmean(maps[t], 0)) * 1e3 for t in titles}
    for i, (ax, t) in enumerate(zip(axs.flat, titles)):
        M = Ms[t]
        if shared_rows:                                     # one colour scale per row (same lag, all spouts)
            row = titles[(i // ncol) * ncol:(i // ncol + 1) * ncol]
            lim = np.nanpercentile(np.abs(np.stack([Ms[r] for r in row])), 99) or 1.0
        else:
            lim = np.nanpercentile(np.abs(M), 99) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        if ol.shape == M.shape:
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
        ax.set_title(f"{t}  (n={len(maps[t])})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if not shared_rows or i % ncol == ncol - 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.set_label("beta x1e3", fontsize=6)
    for ax in axs.flat[len(titles):]:
        ax.axis("off")
    import textwrap
    fig.suptitle(textwrap.fill(suptitle, 40 * ncol), fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.96, top=0.88, bottom=0.02, wspace=0.12, hspace=0.25)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("->", path)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    from frozen_area_maps import area_image, outline
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA_basis")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    maps, shape = kernel_maps(run, PANELS + TARGET)
    img, nm = area_image(sorted(p.stem for p in (run / "joint" / "sessions").glob("PS92_*.json"))[0])
    ol = outline(img)
    base = (f"NEURAL: EVENT-KERNEL beta maps of the FROZEN pre-stroke encoder (Musall et al. 2019 style), {a.tag}: mean of 4 "
            "animals' pre-stroke weights mapped to pixels through each animal's SVD basis (dF/F x1e3 per unit). Image left = "
            "ipsilesional.")
    draw(maps, [t for t, _v, _k in PANELS if t in maps], run / "figures" / "kernel_maps.png",
         base + " Target = per-spout DEVIATION from the shared cue kernel; lick-onset deviation = the lick-locked response "
         "beyond the spout template. Lick-onset and contact kernels look venous-contaminated (vessel-shaped stripes); read "
         "them with that caution.", ol, None)
    draw(maps, [t for t, _v, _k in TARGET if t in maps], run / "figures" / "kernel_maps_target_6spouts.png",
         base + " TARGET kernels of all six spouts (canonical order: close ipsi, mid, contra, then far ipsi, mid, contra): each spout's deviation from the shared "
         "cue kernel, at 0.3 / 0.6 / 1.0 / 2.0 s after the cue; one colour scale per row.", ol, None, ncol=6, shared_rows=True)
    print("grid", shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
