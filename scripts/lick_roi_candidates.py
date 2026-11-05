"""CANDIDATE ALM / tjM1 ROIs drawn over the pre ITI sagittal map (Priya 10-27 sketch: extend ALM medially / posteriorly
to the blue line, tjM1 posteriorly to the red line; "the regions are chosen based on known anatomical coordinates too").

Candidates (per hemisphere, within the Allen MOs (ALM) / MOp (tjM1) masks on the stat mask):
  v1    anterolateral 40 % / 40 % (z(anterior) + z(lateral) rank; the ROIs used so far)
  v2    anterolateral 50 % / 60 % (same rank)
  AP    ALM = MOs anterior of AP_ALM; tjM1 = MOp anterior of AP_TJ (an anterior-only cut, which reaches medially)
  AP2   anterior-weighted rank 2 z(anterior) + z(lateral), 60 % / 70 %
Literature centres (mm from bregma): ALM AP +2.5 ML 1.5 (Guo 2014), tjM1 AP +1.8 ML 2.6 (Mayrhofer 2019), tongue / jaw
ICMS AP +2.4 ML 2.0 (Komiyama 2010) as markers. Atlas: bregma at pixel (320, 270), 0.0194 mm / px (allen_register).
Output figures/lick_roi_candidates.png. Usage: python scripts/lick_roi_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import iti_licks as il  # noqa: E402

AP_ALM, AP_TJ = 1.9, 1.0
ML_ALM_MIN = 1.25        # proposal: ALM lateral of the rostral forelimb area (RFA ~AP +2.0, ML ~1.0)
# NEAREST PUBLISHED CENTRE (Esmaeili et al. 2021 Neuron, Petersen lab, targeting coordinates + Fig 3A probe map;
# Mayrhofer et al. 2019 tjM1; Guo 2014 ALM): every frontal cortex pixel (MOs / MOp, AP > AP_MIN) goes to the nearest
# centre within R_MAX mm; ALM and tjM1 are the pixels won by those centres (wM2 / wM1 / mPFC centres compete only).
CENTRES = {"ALM": [(2.5, 1.5)], "tjM1": [(2.0, 2.0), (1.8, 2.6)], "wM2": [(2.0, 1.0)], "wM1": [(1.0, 1.0)],
           "mPFC": [(2.0, 0.5)]}
R_MAX, AP_MIN = 0.9, 0.8
LIT = {"ALM (Guo 2014)": (2.5, 1.5, "k"), "tjM1 (Mayrhofer 2019)": (1.8, 2.6, "g"), "tongue/jaw ICMS (Komiyama 2010)": (2.4, 2.0, "m"),
       "wM2 / RFA (Esmaeili 2021; approx.)": (2.0, 1.0, "b"), "CFA (forelimb M1)": (0.5, 1.25, "c"),
       "tjM1 (Esmaeili 2021)": (2.0, 2.0, "lime"), "wM1 (Esmaeili 2021)": (1.0, 1.0, "y")}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import combined_encoding as ce
    from wfield_local.allen_register import BREGMA_OFFSET as BO
    from wfield_local.allen_register import RESOLUTION as RES
    lab0 = il.labels()[0]
    _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(lab0)
    H, W = shape
    r, c = flat // W, flat % W
    ap = (BO[1] - r) * RES
    ml = np.abs(c - BO[0]) * RES
    base = il.roi_masks(parea, pstat, names, flat, shape, frac={"ALM": 0.4, "tjM1": 0.4})

    def ranked(src, h, f, wa=1.0):
        mk = base[f"{src}_{h}"]
        a, la = ap[mk], ml[mk]
        u = wa * (a - a.mean()) / a.std() + (la - la.mean()) / la.std()
        sel = np.zeros(len(flat), bool)
        sel[np.flatnonzero(mk)[u >= np.quantile(u, 1 - f)]] = True
        return sel

    cands = {}
    for nm, spec in {"v1: ALM 40 % / tjM1 40 % anterolateral": ("rank", 0.4, 0.4, 1.0),
                     "ALM 40 % / tjM1 50 %": ("rank", 0.4, 0.5, 1.0),
                     "ALM 45 % / tjM1 50 %": ("rank", 0.45, 0.5, 1.0),
                     "ALM 45 % / tjM1 50 % over the PRE cued LATERAL map": ("rank", 0.45, 0.5, 1.0)}.items():
        A = np.zeros(len(flat), bool)
        T = np.zeros(len(flat), bool)
        for h in ("left", "right"):
            if spec[0] == "near":
                fr = (base[f"MOs_{h}"] | base[f"MOp_{h}"]) & (ap > AP_MIN)
                dist = {k: np.min([np.hypot(ap - a0, ml - m0) for a0, m0 in v], 0) for k, v in CENTRES.items()}
                Dm = np.stack(list(dist.values()))
                win = np.array(list(dist))[np.argmin(Dm, 0)]
                ok = fr & (Dm.min(0) <= R_MAX)
                A |= ok & (win == "ALM")
                T |= ok & (win == "tjM1")
            elif spec[0] == "apml":
                A |= base[f"MOs_{h}"] & (ap > AP_ALM) & (ml > ML_ALM_MIN)
                T |= base[f"MOp_{h}"] & (ap > AP_TJ)
            elif spec[0] == "ap":
                A |= base[f"MOs_{h}"] & (ap > AP_ALM)
                T |= base[f"MOp_{h}"] & (ap > AP_TJ)
            else:
                A |= ranked("MOs", h, spec[1], spec[3])
                T |= ranked("MOp", h, spec[2], spec[3])
        cands[nm] = (A, T)
    D = figdata.load_fd("lick_decoder_pattern_stats_sag")
    pos = {p: i for i, p in enumerate(flat)}
    idx = np.array([pos[p] for p in D["grid"]["pix"]])
    v = np.full(len(flat), np.nan)
    v[idx] = D["res"][("iti", "sag", "pre", "level")]["point"]
    D2 = figdata.load_fd("lick_decoder_pattern_stats_lat")
    v2 = np.full(len(flat), np.nan)
    v2[idx] = D2["res"][("trial", "lat", "pre", "level")]["point"]

    def img(x):
        g = np.full(H * W, np.nan)
        g[flat] = x
        return g.reshape(H, W)
    mo = img((base["MOs_left"] | base["MOs_right"]).astype(float) + 2 * (base["MOp_left"] | base["MOp_right"]))
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.6), layout="constrained")
    for ax, (nm, (A, T)) in zip(axs, cands.items()):
        vv = v2 if "LATERAL" in nm else v
        lv = np.nanpercentile(np.abs(vv), 99)
        ax.imshow(img(vv), cmap=figdata.PRE_CMAP, vmin=-lv, vmax=lv, interpolation="nearest")
        ax.contour(mo, levels=[0.5, 1.5], colors="0.45", linewidths=0.6)
        ax.contour(img(A.astype(float)), levels=[0.5], colors="k", linewidths=1.6)
        ax.contour(img(T.astype(float)), levels=[0.5], colors="g", linewidths=1.6)
        for a0, m0, col in LIT.values():
            for s in (-1, 1):
                ax.plot(BO[0] + s * m0 / RES, BO[1] - a0 / RES, "o", mfc="none", mec=col, ms=9, mew=1.8)
        ax.plot(BO[0], BO[1], "+", color="r", ms=10)
        cA = [f"{ap[A & base[f'MOs_{h}']].mean():+.2f}/{ml[A & base[f'MOs_{h}']].mean():.2f}" for h in ("left",)]
        cT = [f"{ap[T & base[f'MOp_{h}']].mean():+.2f}/{ml[T & base[f'MOp_{h}']].mean():.2f}" for h in ("left",)]
        ax.set_title(f"{nm}\nALM centroid AP/ML {cA[0]} mm, tjM1 {cT[0]} mm", fontsize=9)
        ax.set_xlim(60, 580)
        ax.set_ylim(420, 60)
        ax.axis("off")
    fig.suptitle("Candidate ALM (black) / tjM1 (green) ROIs over the PRE ITI sagittal Haufe map (orange = more active for longer "
                 "licks). Grey = Allen MOs / MOp. Circles = literature centres: black ALM AP +2.5 ML 1.5 (Guo 2014), green "
                 "tjM1 AP +1.8 ML 2.6 (Mayrhofer 2019),\nmagenta tongue/jaw ICMS AP +2.4 ML 2.0 (Komiyama 2010), blue wM2 / RFA AP +2.0 ML 1.0 (Esmaeili 2021), "
                 "lime tjM1 AP +2.0 ML 2.0 (Esmaeili 2021), yellow wM1 AP +1.0 ML 1.0, cyan CFA AP +0.5 ML 1.25; red + = bregma. Last panel: PRE cued lateral map. ipsi (lesion) = image LEFT. PROVISIONAL", fontsize=9)
    fig.savefig(il.OUT / "figures" / "lick_roi_candidates.png", dpi=95)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
