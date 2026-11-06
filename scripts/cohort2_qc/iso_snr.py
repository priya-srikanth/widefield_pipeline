"""Absolute 415/470 brain-ROI counts and shot-noise ΔF/F floor, cohort2 2026-10-29.

Answers the "415 dimmed a TON — raise the LED?" question: prints baseline counts above the camera dark
offset, the ratio to offset, and the single-frame shot-noise ΔF/F floor (100/sqrt(photoelectrons),
PCO edge ~0.46 e-/count). The RiboL1 415 is the brightest of the cohort => usable as is (see
docs/NEW_COHORT_2026-10-29.md). Reads raw .dat only (sampled); run in the `wfield` env."""
import numpy as np, re
from pathlib import Path
from scipy.ndimage import binary_erosion

S = [("PS118 Thy1",     "PS118_20261029_112630"),
     ("PS120 Thy1 dim", "PS120_20261029_133200"),
     ("PS121 RiboL1",   "PS121_20261029_160848"),
     ("PS122 RiboL1",   "PS122_20261029_192217")]
root = Path("E:/labcams_data/20261029")
GAIN = 0.46  # PCO edge ~0.46 e-/count (approx)
print(f"{'sess':14s} {'dark':>6s} {'470_F':>8s} {'415_F':>8s} {'415-dark':>9s} "
      f"{'415/dark':>8s} {'shot_dFF%':>9s} {'end415%':>8s}")
for name, sess in S:
    dat = next((root / sess / "raw_widefield_data").glob("*_uint16.dat"))
    m = re.search(r"_(\d+)_(\d+)_(\d+)_uint16", dat.name); H, W = int(m[2]), int(m[3])
    nphys = dat.stat().st_size // (H * W * 2); npair = nphys // 2
    mm = np.memmap(dat, dtype=np.uint16, mode="r", shape=(nphys, H, W))
    a = np.asarray(mm[0:40:2], float).mean(); b = np.asarray(mm[1:40:2], float).mean()
    fo, io = (1, 0) if b >= a else (0, 1)                      # func = brighter, iso = 415
    samp = np.linspace(0, npair - 1, 120).astype(int)
    F0 = np.mean([mm[2 * k + fo] for k in samp], 0)
    brain = binary_erosion(F0 > 0.45 * F0.max(), iterations=4)
    func = F0[brain].mean()
    iso = np.mean([mm[2 * k + io][brain].mean() for k in samp])
    dark = min(F0[:5, :5].mean(), F0[:5, -5:].mean(), F0[-5:, :5].mean(), F0[-5:, -5:].mean())
    iso_sig = iso - dark
    shot = 100.0 / np.sqrt(max(iso_sig, 1) * GAIN)
    tsamp = np.linspace(0, npair - 1, 30).astype(int)
    it = np.array([mm[2 * k + io][brain].mean() for k in tsamp])
    print(f"{sess[:14]:14s} {dark:6.0f} {func:8.0f} {iso:8.0f} {iso_sig:9.0f} "
          f"{iso/max(dark,1):8.1f} {shot:9.3f} {it[-1]/it[0]*100-100:+8.0f}")
