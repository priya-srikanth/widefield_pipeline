"""Channel-SWAPPED frames in the clean-pairs frame map, and what they do to the hemo coefficient. cohort2 10-29.

Found 2026-10-30 while asking whether 415 is isosbestic for RiboL1-jGCaMP8s (iso_check.py: RiboL1 415 had a flat
spectrum, no heartbeat, coherence ~0 with 470). Cause: a few dozen frames per channel per session carry the OTHER
LED's image. In cortex-mean dF/F a swapped frame jumps by F_other / F_own - 1, which is what is measured:

    PS121 415 +94% (swap predicts +114%), 470 -54% (-53%); PS122 +56% (+54%), -38% (-35%); PS120 +19%/-13%
    (+13%/-11%); PS118 ~1% (415 and 470 equally bright, so a swap is invisible in brightness).

They are not at the frame map's parity jumps, not motion (motion_correction_shifts), not locked to behaviour events.
Under 0.3% of frames, but in RiboL1 they carry ~97% of the 415 fast-band variance, so the 0.1-14 Hz hpfit coefficient
collapses (cortex-mean slope 0.10 / 0.43). Interpolated out, the RiboL1 415 behaves as isosbestic: heartbeat
coherence 0.97, r 0.85-0.90, slope ~2.2. Prints per session: swapped-frame counts / amplitudes vs the swap prediction,
the 0.1-14 Hz slope before / after, and the slope by session third (415 bleaching makes it drift; see the doc).

    conda activate locanmf
    python scripts/cohort2_qc/swapped_frames.py
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, median_filter
from scipy.signal import butter, coherence, filtfilt

FS = 31.23
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/20261029")
SESS = ["PS118_20261029_112630", "PS120_20261029_133200", "PS121_20261029_160848", "PS122_20261029_192217"]
JUMP = 0.05          # |frame - 11-frame running median| in cortex-mean dF/F that flags a frame (swaps are 11-114%)


def band(x, lo, hi):
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def slope(x, y):
    return float(x @ y / (y @ y))


def main():
    for s in SESS:
        res = ROOT / s / "motion_corrected" / "wfield_local_results"
        fa = np.load(res / "frames_average.npy")
        brain = binary_erosion(fa[1] > 0.45 * fa[1].max(), iterations=4)
        F415, F470 = fa[0][brain].mean(), fa[1][brain].mean()
        U = np.load(res / "U.npy")
        um = np.nan_to_num(U.reshape(-1, U.shape[-1])[brain.reshape(-1)]).mean(0)
        del U
        svt = np.load(res / "SVT.npy", mmap_mode="r")
        a, b = um @ np.asarray(svt[:, 1::2], np.float64), um @ np.asarray(svt[:, 0::2], np.float64)
        bad = np.zeros(len(a), bool)
        for y, nm, pred in ((b, "415", F470 / F415 - 1), (a, "470", F415 / F470 - 1)):
            amp = y - median_filter(y, 11)
            k = np.abs(amp) > JUMP
            bad |= k
            print(f"{s[:5]} {nm}: {k.sum():3d} flagged frames, median jump {100*np.median(amp[k]) if k.any() else 0:+5.0f}%"
                  f"  (a swap predicts {100*pred:+.0f}%)")
        bad = binary_dilation(bad, iterations=2)
        i = np.arange(len(a))
        a2, b2 = a.copy(), b.copy()
        for y in (a2, b2):
            y[bad] = np.interp(i[bad], i[~bad], y[~bad])
        f, C = coherence(a2, b2, fs=FS, nperseg=1024)
        hb = (f > 8) & (f < 13)
        print(f"      0.1-14 Hz slope 470~415: as built {slope(band(a, .1, 14), band(b, .1, 14)):.2f} -> swaps removed "
              f"{slope(band(a2, .1, 14), band(b2, .1, 14)):.2f} (r {np.corrcoef(band(a2, .1, 14), band(b2, .1, 14))[0, 1]:.2f});"
              f" heartbeat coherence {C[hb].max():.2f} at {f[hb][np.argmax(C[hb])]:.1f} Hz")
        th = np.array_split(i, 3)
        sl = [slope(band(a2, .1, 5)[t], band(b2, .1, 5)[t]) for t in th]
        lv = [1 + b2[t].mean() for t in th]
        print(f"      0.1-5 Hz slope by third {'/'.join(f'{v:.2f}' for v in sl)}; 415 F / session mean "
              f"{'/'.join(f'{v:.2f}' for v in lv)}", flush=True)


if __name__ == "__main__":
    main()
