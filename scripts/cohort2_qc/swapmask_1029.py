"""Exclude the other-LED ("swapped") frames from the 10-29 cohort2 analysis (Priya 2026-10-30: "for our current
non-cam-log analysis exploration, can we exclude the swapped frames from 10/29 analysis").

Per session, a pair is BAD if, in either channel, either detector fires:
  * brightness: cortex-mean dF/F jumps > 5% from its 11-frame running median (swapped_frames.py; catches the
    RiboL1 swaps, where the LEDs differ ~1.5-2x in brightness);
  * pattern: the deviation scores q in 0.1-1.5 on the other-LED / own-LED pixel-ratio map (swap_scan.py; catches
    swaps when the LEDs are equally bright and the PARTIAL rolling-shutter ones, a band of rows lit by the other LED).
Bad pairs are dilated by +-2 pairs (the frames around a spike carry its SVD / filter ringing).

Writes, per session, into its wfield_local_results/ (new files only; the adopted products are untouched):
  swapped_pairs.npz  -- bad (bool per pair), core (undilated), and per-detector hits
  hemo_meegkit_hpfit_refitT_swapmask/  -- the adopted variant rebuilt with bad pairs interpolated in both channels
                                          and T refitted (hemo_variants.compute(bad_pairs=...))
Analyses of 10-29 should read the _swapmask SVTcorr and drop epochs touching `bad` where they need exactness.

    conda activate locanmf
    python scripts/cohort2_qc/swapmask_1029.py [--mask-only]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, median_filter

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/20261029")
DAQ = Path("M:/MICROSCOPE/Priya/Widefield/DAQ_recorder_output/20261029")
SESS = {"PS118": ("PS118_20261029_112630", "PS118_20261029_114909.h5"),
        "PS120": ("PS120_20261029_133200", "PS120_20261029_135728.h5"),
        "PS121": ("PS121_20261029_160848", "PS121_20261029_163740.h5"),
        "PS122": ("PS122_20261029_192217", "PS122_20261029_192406.h5")}
FUNC, ISO = 1, 0


def detect(res: Path, func: int = FUNC) -> dict:
    iso = 1 - func
    fa = np.load(res / "frames_average.npy")
    brain = binary_erosion(fa[func] > 0.45 * fa[func].max(), iterations=4)
    U = np.load(res / "U.npy")
    K = U.shape[-1]
    Ub = np.nan_to_num(U.reshape(-1, K)[brain.reshape(-1)]).astype(np.float64)
    del U
    um = Ub.mean(0)
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    n = svt.shape[1] // 2
    hits = {}
    for ch, nm in ((iso, "415"), (func, "470")):
        S = np.asarray(svt[:, ch::2], np.float64)[:, :n]
        E = S - median_filter(S, size=(1, 11), mode="nearest")
        hits[f"bright_{nm}"] = np.abs(um @ E) > 0.05
        P0 = fa[1 - ch][brain] / fa[ch][brain] - 1
        P0 = P0 - P0.mean()
        P0h = Ub @ np.linalg.lstsq(Ub, P0, rcond=None)[0]
        q = (Ub.T @ P0h) @ E / (P0h @ P0h)
        rsd = np.median(np.abs(q)) * 1.4826
        hits[f"pattern_{nm}"] = (q > 0.1) & (q < 1.5) & (np.abs(q) > 8 * rsd)
    core = np.any(np.vstack(list(hits.values())), 0)
    return dict(core=core, bad=binary_dilation(core, iterations=2), **hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-only", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    from wfield_local import hemo_variants as hv
    for animal, (sdir, h5) in SESS.items():
        if a.only and animal not in a.only:
            continue
        mc = ROOT / sdir / "motion_corrected"
        res = mc / "wfield_local_results"
        m = detect(res)
        np.savez(res / "swapped_pairs.npz", **m)
        summ = {k: int(v.sum()) for k, v in m.items()}
        print(f"{animal}: " + json.dumps(summ) + f"  ({100 * m['bad'].mean():.3f}% of pairs excluded)", flush=True)
        if a.mask_only:
            continue
        s = dict(label=f"{animal}_1029", mc=str(mc), h5=str(DAQ / h5), regime="B", fmdir=None)
        hv.write(s, hv.ADOPTED, refit_t=True, func=FUNC, bad_pairs=m["bad"], tag="_swapmask")


if __name__ == "__main__":
    main()
