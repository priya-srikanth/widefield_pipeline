"""Is the RiboL1-8s 415 BELOW its isosbestic point, i.e. does 415 fall when calcium rises? (2026-10-31)

Priya 10-31: jGCaMP8s's isosbestic point is probably ~440 nm. Then 415 is on the Ca-FREE side: calcium makes 415 go
DOWN, so 415 carries inverted calcium. Prediction: in the band where calcium dominates 470 and hemodynamics are
weak (2-8 Hz), the 415-470 cross-spectrum phase is near 180 deg (anti-phase) in RiboL1, near 0 in Thy1-6s (isosbestic
~405-410, 415 just above it) and in PS120 (no GCaMP: pure hemodynamics, in phase). Coherence magnitude alone (§0g)
cannot see the sign. Raw dF/F channels (SVT), cortex mean and the corrected lick hotspot (top 2%), whole session,
swap pairs interpolated in cohort1. Also the real part of the coherency (signed: +1 in phase, -1 anti-phase) and the
event-locked 415 at the 470 lick peak.

    conda activate locanmf
    python scripts/cohort2_qc/iso_phase.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import csd, welch

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from indicator_kinetics_spread import FS, IND, OUT, VAR, events, sessions  # noqa: E402

BANDS = [(0.1, 0.5), (0.5, 2), (2, 5), (5, 8), (8, 13)]


def one(s):
    from wfield_local import hemo_variants as hv
    res = Path(s["mc"]) / "wfield_local_results"
    func = s["func"]
    br = brain_mask(np.load(res / "frames_average.npy"), func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    del U
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    C = np.load(res / VAR / "SVTcorr.npy", mmap_mode="r")
    n = C.shape[1]
    a = np.asarray(SVT[:, func::2][:, :n], np.float64)
    b = np.asarray(SVT[:, 1 - func::2][:, :n], np.float64)
    if s["cohort"] == "cohort1" and (res / "swapped_pairs.npz").exists():
        bad = np.load(res / "swapped_pairs.npz")["bad"][:n]
        a, b = hv._interp_pairs(a, bad), hv._interp_pairs(b, bad)
    cf, fl, _ = events(s, n)
    Cc = np.asarray(C, np.float64)
    m = Ub @ (Cc[:, fl[:, None] + np.arange(0, 16)].mean(2) - Cc[:, cf[:, None] + np.arange(-31, 0)].mean(2)).mean(1)
    del Cc
    hot = m >= np.percentile(m, 98)
    out = dict(label=s["label"], indicator=IND[s["animal"]])
    for nm, w in (("cortex", Ub.mean(0)), ("hot", Ub[hot].mean(0))):
        x, y = w @ a, w @ b
        if s["animal"] == "PS122" and s["date"] == "20261029":
            x, y = x[:int(110 * 60 * FS)], y[:int(110 * 60 * FS)]
        f, Pxy = csd(x, y, fs=FS, nperseg=1024)
        _, Pxx = welch(x, fs=FS, nperseg=1024)
        _, Pyy = welch(y, fs=FS, nperseg=1024)
        for lo, hi in BANDS:
            k = (f >= lo) & (f < hi)
            S = Pxy[k].sum()
            coh = S / np.sqrt(Pxx[k].sum() * Pyy[k].sum())
            out[f"{nm}_{lo}-{hi}_phase_deg"] = float(np.degrees(np.angle(S)))
            out[f"{nm}_{lo}-{hi}_coh_real"] = float(coh.real)
            out[f"{nm}_{lo}-{hi}_coh_abs"] = float(abs(coh))
    return out


def main():
    from wfield_local.parallel import fan_out
    res, fail = fan_out(sessions(), one, jobs=4, label="session")
    if fail:
        print("FAILED", fail)
    R = sorted((r for _, r in res), key=lambda r: (r["indicator"], r["label"]))
    (OUT / "iso_phase.json").write_text(json.dumps(R, indent=1))
    for nm in ("cortex", "hot"):
        print(f"\n{nm}: signed coherence (real part; +1 in phase, -1 anti-phase) and phase (deg) per band")
        print(f"{'session':11s} ind  " + "  ".join(f"{lo}-{hi}Hz".rjust(14) for lo, hi in BANDS))
        for r in R:
            print(f"{r['label']:11s} {r['indicator']:4s} " + "  ".join(
                f"{r[f'{nm}_{lo}-{hi}_coh_real']:+.2f} ({r[f'{nm}_{lo}-{hi}_phase_deg']:+4.0f})".rjust(14)
                for lo, hi in BANDS))


if __name__ == "__main__":
    main()
