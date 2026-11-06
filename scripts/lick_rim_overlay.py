"""THE ROI RIM vs ACTIVITY (Priya 10-27: "can I see it with SSp-m/n activity superimposed"; "only expand laterally").

Pre-stroke lick response per pixel (ITI bout starts with >= 2 s quiet; dF/F x1e3, 0..0.3 s minus -1.5..-1.0 s; one map per
session from the session's adopted hemo-corrected SVD (Ub @ V), drift removed as `iti_licks.extract`; mean of the
animal's PRE sessions, then mean of animals), drawn on the brain mask, with:
  grey dashed  the 16-px stat-mask rim (excluded from ROI / area averages in ROI v1-v2)
  cyan         pixels ROI v4 adds back (thin rim, lateral edge only, anterior of RIM_LAT_AP_MIN)
  black / green ALM / tjM1 (v4); magenta SSp-m, orange SSp-n (Allen)
Output figures/lick_rim_overlay.png (figdata figdata/lick_rim_overlay.pkl; --plot-only redraws).
Usage: WFIELD_ROI_VERSION=4 python scripts/lick_rim_overlay.py [--jobs 8] [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import iti_licks as il  # noqa: E402

STEM = "lick_rim_overlay"


def one(item):
    from wfield_local import combined_encoding as ce
    label = item[0]
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    z = il._load(label, il.XOUT / "sessions_roi2")            # lick times only (identical in every ROI version)
    t = z["bout_t"][z["bout_gap"] >= 2.0]
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    Vd = V.astype(np.float64) - Q @ (Q.T @ V.astype(np.float64))
    lags = il.LAGS_P
    b = (lags >= -1.5) & (lags <= -1.0)
    w = (lags >= 0.0) & (lags <= 0.3)
    X = il._sample(Vd, ft, t, lags)                              # (licks, lags, K)
    dv = X[:, w].mean(1).mean(0) - X[:, b].mean(1).mean(0)       # (K,)
    img = np.full(int(shape[0]) * int(shape[1]), np.nan)
    img[flat] = (Ub.astype(np.float64) @ dv) * 1e3
    return {"label": label, "animal": label[:4], "img": img.reshape(shape).astype(np.float32), "n": len(t)}


def compute(jobs: int) -> dict:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    labels = sorted(p.stem for p in (il.XOUT / "sessions_roi2").glob("*.npz"))
    pre = [x for x in labels if str(il._load(x, il.XOUT / "sessions_roi2")["epoch"]) == "pre"]
    got, fail = ak.fan_sessions([(x,) for x in pre], one, jobs=jobs, label="rim overlay", log=lambda m: None)
    for it, err in fail:
        print("  !!", it[0], err, flush=True)
    by = {}
    for _it, r in got:
        by.setdefault(r["animal"], []).append(r["img"])
    mean = np.nanmean(np.stack([np.nanmean(np.stack(v), 0) for v in by.values()]), 0)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(pre[0])
    M = il.roi_masks(parea, pstat, names, flat, shape)
    H, W = shape

    def im(x):
        g = np.zeros(H * W, bool)
        g[flat] = x
        return g.reshape(H, W)
    lab = np.array([str(names.get(int(a), "")).rsplit("_", 1)[0] for a in parea])
    return {"mean": mean, "n_sess": {a: len(v) for a, v in by.items()}, "rim": im(~pstat), "added": im(M["cortex"] & ~pstat),
            "ALM": im(M["ALM_left"] | M["ALM_right"]), "tjM1": im(M["tjM1_left"] | M["tjM1_right"]),
            "SSp-m": im(lab == "SSp-m"), "SSp-n": im(lab == "SSp-n"), "brain": im(np.ones(len(flat), bool)),
            "areas": im(np.ones(len(flat), bool)), "parea": (parea, flat, (H, W)), "roi_version": il.ROI_VERSION}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parea, flat, (H, W) = F["parea"]
    ar = np.zeros(H * W)
    ar[flat] = parea
    ar = ar.reshape(H, W)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    v = F["mean"]
    lim = np.nanpercentile(v[F["brain"]], 99)
    for ax, zoom in zip(axs, (False, True)):
        ax.imshow(np.where(F["brain"], v, np.nan), cmap="magma", vmin=0, vmax=lim, interpolation="nearest")
        bnd = np.zeros_like(ar)
        bnd[1:, :] += ar[1:, :] != ar[:-1, :]
        bnd[:, 1:] += ar[:, 1:] != ar[:, :-1]
        ax.contour(bnd > 0, levels=[0.5], colors="0.75", linewidths=0.3)
        ax.contour(F["rim"].astype(float) * F["brain"], levels=[0.5], colors="w", linewidths=1.0, linestyles="--")
        ax.contour(F["added"].astype(float), levels=[0.5], colors="c", linewidths=1.6)
        ax.contour(F["ALM"].astype(float), levels=[0.5], colors="w", linewidths=1.6)
        ax.contour(F["tjM1"].astype(float), levels=[0.5], colors="lime", linewidths=1.6)
        ax.contour(F["SSp-m"].astype(float), levels=[0.5], colors="m", linewidths=1.6)
        ax.contour(F["SSp-n"].astype(float), levels=[0.5], colors="orange", linewidths=1.6)
        ax.axis("off")
        if zoom:
            ax.set_xlim(0, W // 2 + 20)
            ax.set_ylim(H * 0.75, H * 0.12)
            ax.set_title("zoom: lesioned (left) hemisphere, anterolateral", fontsize=10)
        else:
            ax.set_title("whole brain", fontsize=10)
    fig.suptitle(f"PRE-stroke lick response per pixel (ITI bout starts, 2 s quiet; dF/F x1e3, 0-0.3 s minus -1.5..-1.0 s; mean "
                 f"of animals, sessions {F['n_sess']}).\nWhite dashed = 16-px rim excluded from ROI / area averages; CYAN = "
                 f"pixels ROI v{F['roi_version']} adds back (lateral edge only); white = ALM, green = tjM1, magenta = SSp-m, "
                 "orange = SSp-n. ipsi (lesion) = image LEFT. PROVISIONAL", fontsize=9)
    save_png_svg(fig, il.OUT / "figures" / f"{STEM}.png", dpi=95)
    plt.close(fig)


def main() -> int:
    if figdata.plot_only():
        F = figdata.load_fd(STEM)
    else:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
        F = compute(jobs)
        figdata.save_fd(STEM, F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
