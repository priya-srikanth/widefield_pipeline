"""Scan ANY session (cohort1 or cohort2) for frames that carry the other LED's image, from the SVD products only.

`swapped_frames.py` finds them by BRIGHTNESS: a swapped frame jumps by F_other / F_own - 1. That only works when
the two channels differ in brightness (RiboL1: 470 ~2x 415). In GCaMP6s, 415 and 470 are about equally bright, so
here the per-PIXEL ratio map is used as well. P = F_other / F_own - 1 (from `frames_average`) is not flat: vessels
and parenchyma, and the two LEDs' illumination fields, differ. A swapped frame's dF/F deviation from its neighbours
IS that map. So each frame's deviation from an 11-frame running median, e_t (K components), is scored on:

  m_t = brain-mean deviation / mean(P)                 -> 1 for a swap (meaningful only if |mean P| is large)
  q_t = <deviation, P0> / <P0, P0>, P0 = P - mean(P)   -> 1 for a swap, from the SPATIAL pattern alone

P0 is first projected onto the session's SVD basis (the SVT can only represent what U spans). A frame is called
swapped when q_t is within 0.1-1.5 (partial = a band of rows lit by the other LED) and > 8 robust SD from zero. The impact is the cortex-mean 0.1-14 Hz 470~415
slope before vs after interpolating flagged frames (the hemo fit's band); on a clean session it should not move.

    conda activate locanmf
    python scripts/cohort2_qc/swap_scan.py --cohort2              # the four 10-29 sessions
    python scripts/cohort2_qc/swap_scan.py --cohort1 [--every 4]  # registered cohort1 sessions (every Nth)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, median_filter
from scipy.signal import butter, filtfilt

FS = 31.23
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029")
C2 = Path("M:/MICROSCOPE/Priya/Widefield/labcams/20261029")
C2_SESS = ["PS118_20261029_112630", "PS120_20261029_133200", "PS121_20261029_160848", "PS122_20261029_192217"]


def slope(a, b):
    bb, aa = butter(2, [0.1 / (FS / 2), 14 / (FS / 2)], btype="band")
    x, y = filtfilt(bb, aa, a), filtfilt(bb, aa, b)
    return float(x @ y / (y @ y))


def scan(res: Path, label: str, func: int = 1) -> dict:
    iso = 1 - func
    fa = np.load(res / "frames_average.npy")
    brain = binary_erosion(fa[func] > 0.45 * fa[func].max(), iterations=4)
    U = np.load(res / "U.npy", mmap_mode="r")
    K = U.shape[-1]
    Ub = np.nan_to_num(np.asarray(U).reshape(-1, K)[brain.reshape(-1)]).astype(np.float64)
    del U
    um = Ub.mean(0)
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    n = svt.shape[1] // 2
    out = dict(label=label, n_pairs=n, ratio_470_415=float(fa[func][brain].mean() / fa[iso][brain].mean()))
    flagged = np.zeros(n, bool)
    trace = {}
    for ch, nm in ((iso, "415"), (func, "470")):
        S = np.asarray(svt[:, ch::2], np.float64)[:, :n]
        trace[nm] = um @ S
        E = S - median_filter(S, size=(1, 11), mode="nearest")
        P = fa[1 - ch][brain] / fa[ch][brain] - 1
        P0 = P - P.mean()
        c = np.linalg.lstsq(Ub, P0, rcond=None)[0]
        P0h = Ub @ c
        q = (Ub.T @ P0h) @ E / (P0h @ P0h)
        m = (um @ E) / P.mean() if abs(P.mean()) > 0.02 else np.full(n, np.nan)
        rsd = np.median(np.abs(q)) * 1.4826
        hit = (q > 0.1) & (q < 1.5) & (np.abs(q) > 8 * rsd)       # partial (rolling-shutter band) swaps too
        flagged |= hit
        out[f"{nm}_meanP"] = float(P.mean())
        out[f"{nm}_P0_captured"] = float((P0h @ P0h) / (P0 @ P0))
        out[f"{nm}_q_rsd"] = float(rsd)
        out[f"{nm}_n_swapped"] = int(hit.sum())
        out[f"{nm}_q_median_hits"] = float(np.median(q[hit])) if hit.any() else None
        out[f"{nm}_m_median_hits"] = float(np.nanmedian(m[hit])) if hit.any() else None
        out[f"{nm}_n_q_gt_0.5"] = int((q > 0.5).sum())
    out["swapped_frac_pct"] = float(100 * flagged.mean())
    out["slope_as_built"] = slope(trace["470"], trace["415"])
    bad = binary_dilation(flagged, iterations=2)
    if bad.any():
        i = np.arange(n)
        for k in trace:
            trace[k] = trace[k].copy()
            trace[k][bad] = np.interp(i[bad], i[~bad], trace[k][~bad])
    out["slope_swaps_removed"] = slope(trace["470"], trace["415"])
    return out


def cohort1_sessions(every: int):
    from wfield_local import config
    ss = config.load_sessions(machine="analysis_desktop")
    return [s for k, s in enumerate(sorted(ss, key=lambda s: s["label"])) if k % every == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort1", action="store_true")
    ap.add_argument("--cohort2", action="store_true")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    jobs = []
    if a.cohort2:
        jobs += [(C2 / s / "motion_corrected" / "wfield_local_results", s[:5] + "_1029", 1) for s in C2_SESS]
    if a.cohort1:
        from wfield_local.hemo_variants import functional_channel
        for s in cohort1_sessions(a.every):
            jobs.append((Path(s["mc"]) / "wfield_local_results", s["label"], functional_channel(s)))
    rows = []
    keys = ["label", "ratio_470_415", "415_n_swapped", "470_n_swapped", "swapped_frac_pct", "415_q_median_hits",
            "470_q_median_hits", "415_m_median_hits", "470_m_median_hits", "415_q_rsd", "415_P0_captured",
            "slope_as_built", "slope_swaps_removed"]
    for res, lab, func in jobs:
        if not (res / "SVT.npy").is_file():
            print(f"{lab}: no SVT at {res}", flush=True)
            continue
        try:
            r = scan(res, lab, func)
        except Exception as e:                     # one bad session must not stop the sweep
            print(f"{lab}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(r)
        print(" ".join(f"{k}={r[k]:.3g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys), flush=True)
    dest = a.out or OUT / f"swap_scan_{'c1' if a.cohort1 else ''}{'c2' if a.cohort2 else ''}.csv"
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print("wrote", dest, json.dumps({"sessions": len(rows)}))


if __name__ == "__main__":
    main()
