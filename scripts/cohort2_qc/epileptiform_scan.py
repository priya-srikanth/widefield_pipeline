"""Scan every recording (cohort1 + cohort2) for epileptiform-like discharges (2026-10-31).

Priya 10-31, after the PS122 10-29 video: "yes looks like epileptiform discharges with motor twitch. do any other
recordings have this". The PS122 events: cortex-wide 470 rise within one pair to ~+6.5%, gone by ~130 ms, 415 flat,
then a ~2-s two-channel undershoot (QC doc §0o).

Detector, on the cortex-mean 470 dF/F (U . SVT over the brain mask, raw per channel; the same in both cohorts):
  d = x - running median over 2 s;   events = scipy find_peaks(d) with
    prominence  > PROM %              (PS122 median +6.5%; lick responses are a few % and slow)
    half-width  < MAX_W pairs (250 ms) (GCaMP6s interictal events are ~200 ms; behavioural responses are longer)
    rise        : d[peak] - d[peak - 3] > 0.6 * d[peak]       (up within ~100 ms)
    415         : |415 change at the peak| < 0.35 x the 470 change   (not light / blood / a swap)
  and, cohort1 (never repaired before the SVD): no event within +-3 pairs of a swapped pair
  (wfield_local_results/swapped_pairs.npz, from the cohort1 sweep).
Per session: count, rate, times (pair index + min), amplitude, half-width. Sessions with >= MIN_MAP events also get
the mean peak map (saved .npy) for a figure.

    conda activate locanmf
    python scripts/cohort2_qc/epileptiform_scan.py [--jobs 6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

LABCAMS = Path("M:/MICROSCOPE/Priya/Widefield/labcams")
OUT = LABCAMS / "cohort2" / "_tmp_qc_figures_20261031" / "epileptiform_scan"
FS = 31.23
PROM, MAX_W, MIN_MAP = 4.0, 7.8, 3


def sessions():
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    out = [(s["label"], str(Path(s["mc"]) / "wfield_local_results"), int(hv.functional_channel(s)), "cohort1")
           for s in config.load_sessions(machine="analysis_desktop")]
    for d in ("20261029", "20261030"):
        for p in sorted((LABCAMS / d).glob("PS*")):
            res = p / "motion_corrected" / "wfield_local_results"
            if (res / "SVT.npy").exists():
                func = int(json.loads((res / "local_wfield_summary.json").read_text())["functional_channel"])
                out.append((f"{p.name.split('_')[0]}_{d[4:]}", str(res), func, "cohort2"))
    return out


def one(item):
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    label, res, func, cohort = item
    res = Path(res)
    fa = np.load(res / ("frames_average_raw.npy" if (res / "frames_average_raw.npy").exists() else "frames_average.npy"))
    br = brain_mask(fa, func, res=res)
    U = np.load(res / "U.npy", mmap_mode="r")
    Ub = np.nan_to_num(np.asarray(U).reshape(-1, U.shape[-1])[br.reshape(-1)]).astype(np.float32)
    del U
    um = Ub.mean(0).astype(np.float64)
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    S = np.asarray(SVT[:, func::2], np.float32)
    x = 100 * (um @ S)
    y = 100 * (um @ np.asarray(SVT[:, 1 - func::2], np.float64))
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    d = x - median_filter(x, 63, mode="nearest")
    e = y - median_filter(y, 63, mode="nearest")
    pk, pr = find_peaks(d, prominence=PROM, width=(None, MAX_W), rel_height=0.5, wlen=63)
    keep = (pk > 70) & (pk < n - 70)
    keep &= (d[pk] - d[np.maximum(pk - 3, 0)]) > 0.6 * d[pk]
    keep &= np.abs(e[pk]) < 0.35 * d[pk]
    bad = None
    if (res / "swapped_pairs.npz").exists() and cohort == "cohort1":
        bad = np.load(res / "swapped_pairs.npz")["bad"][:n]
        keep &= np.array([not bad[max(0, p - 3):p + 4].any() for p in pk], bool)
    pk = pk[keep]
    w = pr["widths"][keep]
    out = dict(label=label, cohort=cohort, res=str(res), func=func, n_pairs=int(n), dur_min=n / FS / 60,
               n_events=int(len(pk)), per_hour=float(len(pk) / (n / FS / 3600)),
               pairs=pk.tolist(), t_min=np.round(pk / FS / 60, 3).tolist(),
               amp=np.round(d[pk], 2).tolist(), halfwidth_ms=np.round(w / FS * 1000, 0).tolist(),
               amp415=np.round(e[pk], 2).tolist())
    if len(pk) >= MIN_MAP:
        m = Ub @ (S[:, pk].mean(1) - S[:, pk[:, None] + np.arange(-6, -2)].mean((1, 2)))
        img = np.full(br.size, np.nan, np.float32)
        img[br.reshape(-1)] = 100 * m
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(OUT / f"{label}_peakmap.npy", img.reshape(br.shape))
        lags = np.arange(-31, 63)
        ok = pk[(pk > 31) & (pk < n - 63)]
        out["wave470"] = np.round((d[ok[:, None] + lags]).mean(0), 3).tolist()
        out["wave415"] = np.round((e[ok[:, None] + lags]).mean(0), 3).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    from wfield_local.parallel import fan_out
    from wfield_local.writeguard import assert_writable
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    items = sessions()
    res, fail = fan_out(items, one, jobs=a.jobs, label="session")
    R = sorted((r for _, r in res), key=lambda r: (r["cohort"], r["label"]))
    (OUT / "epileptiform_scan.json").write_text(json.dumps(R, indent=1))
    if fail:
        print("FAILED:", fail)
    print(f"{len(R)} sessions scanned")
    for r in R:
        if r["n_events"]:
            print(f"{r['cohort']} {r['label']:12s} {r['n_events']:4d} events ({r['per_hour']:5.1f}/h)  "
                  f"amp med {np.median(r['amp']):5.1f}%  hw med {np.median(r['halfwidth_ms']):4.0f} ms  "
                  f"t(min) {r['t_min'][:12]}", flush=True)


if __name__ == "__main__":
    main()
