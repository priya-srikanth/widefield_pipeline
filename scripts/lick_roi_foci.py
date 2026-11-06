"""WHERE ARE THE PRE-STROKE MAP FOCI, relative to ALM / tjM1? (Priya 10-27: "let's look at these to decide on ALM/tjM1":
the pre per-spout close-mid / close-contra maps, the ITI lateral and sagittal maps)

For each PRE Haufe map (mean of animals, held-out epoch decoder; figdata of iti_decoder_pattern_stats lat / sag and
iti_spout_pattern_stats cued + ITI): per hemisphere and sign, the FOCUS = the top 3 % of |map| pixels of that sign in
the hemisphere; printed: its Allen-area composition, its centroid (image row / col and distance from the midline), and the
fraction inside ALM / tjM1 / rest of MOs / rest of MOp. Figure: every map with the Allen outline, ALM (black), tjM1
(green) and the foci (cyan) -> figures/lick_roi_foci.png; table roi_foci.csv.
Usage: python scripts/lick_roi_foci.py
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import iti_licks as il  # noqa: E402

TOP = 0.03
MAPS = [("lick_decoder_pattern_stats_lat", ("iti", "lat", "pre", "level"), "ITI lateral"),
        ("lick_decoder_pattern_stats_lat", ("trial", "lat", "pre", "level"), "cued lateral"),
        ("lick_decoder_pattern_stats_sag", ("iti", "sag", "pre", "level"), "ITI sagittal"),
        ("lick_decoder_pattern_stats_sag", ("trial", "sag", "pre", "level"), "cued sagittal"),
        ("lick_cued_spout_pattern_stats", ("close_mid", "pre", "level"), "cued close mid"),
        ("lick_cued_spout_pattern_stats", ("close_contra", "pre", "level"), "cued close contra"),
        ("lick_cued_spout_pattern_stats", ("far_contra", "pre", "level"), "cued far contra"),
        ("lick_cued_spout_pattern_stats", ("far_mid", "pre", "level"), "cued far mid")]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import combined_encoding as ce
    lab0 = il.labels()[0]
    _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(lab0)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    H, W = shape
    r, c = flat // W, flat % W
    hemi = {h: np.zeros(len(flat), bool) for h in ("left", "right")}
    for n, mk in M.items():
        for h in hemi:
            if n.endswith(f"_{h}") and not n.startswith(("ALM", "tjM1")):
                hemi[h] |= mk
    mid = (np.median(c[hemi["left"]]) + np.median(c[hemi["right"]])) / 2.0
    area = np.full(len(flat), "", object)
    for n, mk in M.items():
        if n != "cortex" and not n.startswith(("ALM", "tjM1")):
            area[mk] = n.rsplit("_", 1)[0]
    pos = {p: i for i, p in enumerate(flat)}
    rows, imgs = [], []
    for fd, key, lab in MAPS:
        D = figdata.load_fd(fd)
        idx = np.array([pos[p] for p in D["grid"]["pix"]])
        v = np.full(len(flat), np.nan)
        v[idx] = D["res"][key]["point"]
        foc = np.zeros(len(flat), bool)
        for h, hm in hemi.items():
            for sgn in (1, -1):
                m = hm & np.isfinite(v) & (sgn * v > 0)
                if m.sum() < 50:
                    continue
                a = np.where(m, sgn * v, -np.inf)
                thr = np.quantile(a[m], 1 - TOP * hm.sum() / m.sum()) if TOP * hm.sum() < m.sum() else -np.inf
                f = m & (a >= thr)
                if f.sum() < 10 or np.nanmax(np.abs(v[f])) < 0.5 * np.nanpercentile(np.abs(v), 99):
                    continue                             # weak lobe: not a focus
                foc |= f
                comp = pd.Series(area[f]).value_counts(normalize=True)
                rows.append(dict(map=lab, hemi=h, sign="+" if sgn > 0 else "-", n_pix=int(f.sum()),
                                 peak=float(np.nanmax(np.abs(v[f]))), row=float(r[f].mean()),
                                 ml_px=float(np.abs(c[f] - mid).mean()),
                                 in_ALM=float(M[f"ALM_{h}"][f].mean()), in_tjM1=float(M[f"tjM1_{h}"][f].mean()),
                                 in_MOs_rest=float((M[f"MOs_{h}"] & ~M[f"ALM_{h}"])[f].mean()),
                                 in_MOp_rest=float((M[f"MOp_{h}"] & ~M[f"tjM1_{h}"])[f].mean()),
                                 areas=", ".join(f"{k} {x:.0%}" for k, x in comp.head(3).items())))
        imgs.append((lab, v, foc))
    R = pd.DataFrame(rows)
    R.to_csv(il.OUT / "roi_foci.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(R.round(2).to_string())

    def img(x):
        g = np.full(H * W, np.nan)
        g[flat] = x
        return g.reshape(H, W)
    roiA = img((M["ALM_left"] | M["ALM_right"]).astype(float))
    roiT = img((M["tjM1_left"] | M["tjM1_right"]).astype(float))
    ar = np.zeros(len(flat))
    for i, n in enumerate(sorted(set(area) - {""}), 1):
        ar[area == n] = i
    arimg = img(ar)
    fig, axs = plt.subplots(2, 4, figsize=(17, 9), layout="constrained")
    for ax, (lab, v, foc) in zip(axs.ravel(), imgs):
        lim = np.nanpercentile(np.abs(v), 99)
        ax.imshow(img(v), cmap=figdata.PRE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.contour(arimg, levels=np.arange(0.5, ar.max() + 1), colors="0.6", linewidths=0.3)
        ax.contour(roiA, levels=[0.5], colors="k", linewidths=1.3)
        ax.contour(roiT, levels=[0.5], colors="g", linewidths=1.3)
        ax.contour(img(foc.astype(float)), levels=[0.5], colors="c", linewidths=1.2)
        ax.set_title(f"PRE {lab}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Pre-stroke Haufe maps (mean of animals): ALM (black) / tjM1 (green) = anterolateral 40 % of MOs / MOp; cyan = "
                 "map FOCI (top 3 % |map| per hemisphere and sign).\nGrey = Allen areas. Orange = more active for "
                 "contralesional (lateral) / longer (sagittal) licks / that spout. ipsi (lesion) = image LEFT. PROVISIONAL",
                 fontsize=10)
    save_png_svg(fig, il.OUT / "figures" / "lick_roi_foci.png", dpi=95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
