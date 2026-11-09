"""DOES THE ALM / tjM1 PLACEMENT FIT OUR MAPS? (Priya 10-27: "should we review the ALM/tjM1 split location?"; "is the
0.4 cutoff supported by maps")

ALM / tjM1 = the anterolateral ROI_FRAC (0.4) of MOs / MOp per hemisphere (`iti_licks.roi_masks`, ranked by
z(anterior) + z(lateral distance from the midline); Priya 10-25: "the anterolateral 1/3 to 1/2"). Check against the PRE
lateral-position Haufe maps (mean of animals; `iti_decoder_pattern_stats` figdata, pre level, ITI and cued) and the
sagittal ones: (1) overlay the ROI outlines; (2) per hemisphere, |map| within MOs / MOp ranked by the same
anterolateral score: the mean |map| in the top f of pixels vs the rest, for f = 0.2 .. 0.6 (a supported cutoff = the
anterolateral part carries clearly more than the rest, and the curve has no strong gain beyond f).
Output figures/lick_roi_check.png, roi_check.csv. Usage: python scripts/lick_roi_check.py
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

FRACS = (0.2, 0.3, 0.4, 0.5, 0.6)


def trial_set_sentence() -> str:
    """THE lick set of the figure (Priya 11-09): the licks behind the PRE Haufe maps it reads."""
    import lick_refit_decoders as rd
    return rd.pre_pattern_sets()


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import combined_encoding as ce
    lab0 = il.labels()[0]
    _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(lab0)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    H, W = shape
    r, c = flat // W, flat % W
    left = np.zeros(len(flat), bool)
    right = np.zeros(len(flat), bool)
    for n, mk in M.items():
        left |= mk & n.endswith("_left")
        right |= mk & n.endswith("_right")
    mid = (np.median(c[left]) + np.median(c[right])) / 2.0
    maps = {}
    for ax_ in ("lat", "sag"):
        D = figdata.load_fd(f"lick_decoder_pattern_stats_{ax_}")
        g = D["grid"]
        pos = {p: i for i, p in enumerate(flat)}
        idx = np.array([pos[p] for p in g["pix"]])           # stat-mask pixel -> brain-pixel index
        for st in ("iti", "trial"):
            v = np.full(len(flat), np.nan)
            v[idx] = D["res"][(st, ax_, "pre", "level")]["point"]
            maps[(ax_, st)] = v
    rows = []
    for (ax_, st), v in maps.items():
        for src, dst in (("MOs", "ALM"), ("MOp", "tjM1")):
            for h in ("left", "right"):
                mk = M[f"{src}_{h}"] & np.isfinite(v)
                ant = -r[mk].astype(float)
                lat = np.abs(c[mk] - mid).astype(float)
                u = (ant - ant.mean()) / (ant.std() or 1) + (lat - lat.mean()) / (lat.std() or 1)
                a = np.abs(v[mk])
                row = dict(map=ax_, set=st, area=f"{src}_{h}", roi=dst)
                for f in FRACS:
                    top = u >= np.quantile(u, 1 - f)
                    row[f"ratio_{f}"] = float(a[top].mean() / a[~top].mean())
                row["corr_score_absmap"] = float(np.corrcoef(u, a)[0, 1])
                rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(il.OUT / "roi_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(2).to_string())

    fig, axs = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    roi = np.zeros(len(flat))
    for v_, n in enumerate(("ALM_left", "ALM_right", "tjM1_left", "tjM1_right"), 1):
        roi[M[n]] = v_
    mos = np.zeros(len(flat))
    for n in ("MOs_left", "MOs_right", "MOp_left", "MOp_right"):
        mos[M[n]] = 1

    def img(v):
        g_ = np.full(H * W, np.nan)
        g_[flat] = v
        return g_.reshape(H, W)
    for i, ax_ in enumerate(("lat", "sag")):
        for j, st in enumerate(("iti", "trial")):
            ax = axs[i, j]
            v = maps[(ax_, st)]
            lim = np.nanpercentile(np.abs(v), 99)
            im = ax.imshow(img(v), cmap=figdata.PRE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
            ax.contour(img(mos), levels=[0.5], colors="0.3", linewidths=0.6)
            ax.contour(img((roi == 1) | (roi == 2)), levels=[0.5], colors="k", linewidths=1.4)
            ax.contour(img((roi == 3) | (roi == 4)), levels=[0.5], colors="g", linewidths=1.4)
            rr = R[(R['map'] == ax_) & (R['set'] == st)]
            ax.set_title(f"pre {'lateral' if ax_ == 'lat' else 'sagittal'} map, {'ITI' if st == 'iti' else 'cued'} licks\n"
                         f"|map| top-40% / rest: ALM {rr[rr.roi == 'ALM']['ratio_0.4'].mean():.2f}, "
                         f"tjM1 {rr[rr.roi == 'tjM1']['ratio_0.4'].mean():.2f} (mean of hemispheres)", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.04, label="pre level (pattern / pre RMS)")
    fig.suptitle("ALM (black) / tjM1 (green) ROIs = anterolateral 40 % of MOs / MOp (gray) over the PRE Haufe maps (mean of "
                 "animals, held-out epoch decoder).\nSign: orange = more active for contralesional (lateral) / longer "
                 "(sagittal) licks. ipsi (lesion) = image LEFT. PROVISIONAL", fontsize=9)
    save_png_svg(fig, il.OUT / "figures" / "lick_roi_check.png", dpi=100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
