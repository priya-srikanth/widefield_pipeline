"""Native-space (no Allen) cue / lick maps for the 10-29 cohort2 sessions (Priya 2026-10-30: "can we just see raw
maps without allen alignment"). The pipeline's map scripts need the Allen-aligned basis (U_atlas.npy), which waits on
landmarks; these use the session's own U, in camera pixels, native orientation.

Signal: hemo_meegkit_hpfit_refitT_swapmask SVTcorr (other-LED frames excluded, §0c-0e of the QC doc). Events from the
DAQ (cue TTL, lick_analog via the pipeline's loaders, mapped to frames by `_frames`). Spout position per trial =
the DAQ strobe code at the cue (agrees 100% with trials.csv pos_idx on all four sessions); 0 close_center,
1 close_L, 2 close_R, 3 far_center, 4 far_L, 5 far_R. No lesion yet, so positions are shown as L / centre / R
(the GUI's names), close row then far row. Trials whose window touches an excluded pair are dropped.

Per session, one figure: per position, the cue response (0-1 s minus -1-0 s), plus all trials, the first-lick
response (0-0.4 s minus -1-0 s), and L-minus-R (close and far). One colour scale per figure (the 99th percentile
of |all-trial cue response|), so the panels compare. Plus a 4-session summary (all-trial cue | first lick).

    conda activate locanmf
    python scripts/cohort2_qc/native_maps_1029.py
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

FS = 31.23
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/20261029")
DAQ = Path("M:/MICROSCOPE/Priya/Widefield/DAQ_recorder_output/20261029")
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/native_maps_20261029")
SESS = {"PS118": ("PS118_20261029_112630", "PS118_20261029_114909.h5", "Thy1-GCaMP6s"),
        "PS120": ("PS120_20261029_133200", "PS120_20261029_135728.h5", "Thy1-GCaMP6s"),
        "PS121": ("PS121_20261029_160848", "PS121_20261029_163740.h5", "RiboL1-jGCaMP8s"),
        "PS122": ("PS122_20261029_192217", "PS122_20261029_192406.h5", "RiboL1-jGCaMP8s")}
CODE = {0: "close_center", 1: "close_L", 2: "close_R", 3: "far_center", 4: "far_L", 5: "far_R"}
ORDER = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
VARIANT = "hemo_meegkit_hpfit_refitT_swapmask"


def window_mean(C, frames, lo, hi, bad):
    """(K,) mean of C over [f+lo, f+hi) frames, averaged over events; events touching a bad pair dropped."""
    a, b = int(round(lo * FS)), int(round(hi * FS))
    keep = [f for f in frames if f + min(a, -int(FS)) >= 0 and f + b < C.shape[1]
            and not bad[f + min(a, -int(FS)):f + b].any()]
    if not keep:
        return None, 0
    idx = np.concatenate([np.arange(f + a, f + b) for f in keep])
    return np.asarray(C[:, idx], np.float64).reshape(C.shape[0], len(keep), b - a).mean((1, 2)), len(keep)


def response(C, frames, lo, hi, bad):
    post, n = window_mean(C, frames, lo, hi, bad)
    pre, _ = window_mean(C, frames, -1.0, 0.0, bad)
    return (None, 0) if post is None else (post - pre, n)


def session(animal):
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    sd, h5, geno = SESS[animal]
    mc = ROOT / sd / "motion_corrected"
    res = mc / "wfield_local_results"
    s = dict(label=f"{animal}_1029", mc=str(mc), h5=str(DAQ / h5), regime="B", fmdir=None)
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, _lf, csmp = _frames(s, cue, lk)
    st = np.asarray(cue["strobe_samples"])
    pos = np.asarray(cue["strobe_codes"])[np.clip(np.searchsorted(st, cue["cue_samples"]) - 1, 0, None)]
    ls = np.sort(lk["lick_samples"])
    j = np.searchsorted(ls, cue["cue_samples"])
    has = (j < len(ls)) & ((ls[np.minimum(j, len(ls) - 1)] - cue["cue_samples"]) / cue["sample_rate_hz"] < 1.5)
    first = np.searchsorted(csmp, ls[j[has]])
    fa = np.load(res / "frames_average.npy")
    H, W = fa.shape[1:]
    brain = binary_erosion(fa[1] > 0.45 * fa[1].max(), iterations=4)
    U = np.nan_to_num(np.load(res / "U.npy"))
    U = U.reshape(-1, U.shape[-1])
    C = np.load(res / VARIANT / "SVTcorr.npy", mmap_mode="r")
    bad = np.load(res / "swapped_pairs.npz")["bad"][:C.shape[1]]
    ok = cf >= 0
    maps, ns = {}, {}
    for code, nm in CODE.items():
        r, n = response(C, cf[ok & (pos == code)], 0.0, 1.0, bad)
        maps[nm], ns[nm] = r, n
    maps["all trials"], ns["all trials"] = response(C, cf[ok], 0.0, 1.0, bad)
    maps["first lick"], ns["first lick"] = response(C, first, 0.0, 0.4, bad)
    img = {}
    for k, v in maps.items():
        if v is None:
            continue
        m = 100 * (U @ v).reshape(H, W)
        m[~brain] = np.nan
        img[k] = m
    for d in ("close", "far"):
        img[f"{d}: L - R"] = img[f"{d}_L"] - img[f"{d}_R"]
        ns[f"{d}: L - R"] = min(ns[f"{d}_L"], ns[f"{d}_R"])
    return img, ns, geno, brain


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for animal in SESS:
        img, ns, geno, brain = session(animal)
        lim = np.nanpercentile(np.abs(img["all trials"]), 99)
        panels = ORDER + ["all trials", "first lick", "close: L - R", "far: L - R"]
        fig, ax = plt.subplots(2, 5, figsize=(20, 8.6))
        slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)]
        for k, (r, c) in zip(panels, slots):
            im = ax[r, c].imshow(img[k], cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[r, c].set_title(f"{k}  (n {ns[k]})", fontsize=10)
            ax[r, c].axis("off")
        fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6, label="corrected dF/F %  (post - pre)")
        fig.suptitle(f"{animal} 10-29 ({geno}): cue response 0-1 s minus -1-0 s by spout position; first lick 0-0.4 s; "
                     f"native camera pixels (no Allen), swap-masked correction; one scale ±{lim:.2f}%", fontsize=11)
        fig.savefig(OUT / f"{animal}_1029_native_cue_lick_maps.png", dpi=110)
        plt.close(fig)
        summary[animal] = (img, ns, geno)
        print(animal, {k: ns[k] for k in panels}, f"scale ±{lim:.2f}%", flush=True)
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))
    for j, (animal, (img, ns, geno)) in enumerate(summary.items()):
        for i, k in enumerate(("all trials", "first lick")):
            lim = np.nanpercentile(np.abs(img[k]), 99)
            im = ax[i, j].imshow(img[k], cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[i, j].set_title(f"{animal} {geno}\n{k} (n {ns[k]}), ±{lim:.2f}%", fontsize=10)
            ax[i, j].axis("off")
            fig.colorbar(im, ax=ax[i, j], shrink=0.7)
    fig.suptitle("10-29 native maps: cue (0-1 s) and first-lick (0-0.4 s) responses; per-panel scale (genotypes differ "
                 "~5x in amplitude)", fontsize=11)
    fig.savefig(OUT / "cohort2_1029_native_summary.png", dpi=110)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
