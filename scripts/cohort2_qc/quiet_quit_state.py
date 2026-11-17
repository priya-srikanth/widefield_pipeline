"""Do the animals look ASLEEP after they quit? Quit-period quiet vs engaged quiet, cortex and face (cohort1, 2026-11-01).

Priya 11-01: "do the animals look 'asleep' in the quit period?" No eye / pupil tracking exists (side-camera eye points
deferred 09-25), and head-fixed mice can sleep with their eyes open (NREM: pupil constricts, eyes often stay open), so
eye closure would not settle it anyway. The signatures available here:
  * CORTEX: NREM-like state = large slow fluctuations. Welch power of the quiet trace (dF/F %, not detrended; 4-s
    segments, linear detrend) in 0.5-1, 1-4 (delta) and 4-8 Hz, and the robust SD of the 2-s-median-detrended trace.
  * FACE: sleep = a still face. Median cam4 FaceRhythm point speed (own basis, summed over the 20 point PCs) and the
    fraction of frames above the session's 90th percentile of face speed (any-state percentile).
FRAMES: quiet exactly as quiet_spikes_v2 (no DAQ contact lick / running within 1.5 s before or after, no swapped pair
within 0.5 s); ENGAGED = before the quit, QUIT = after it. Sessions with >= 60 s of quit-period quiet. August sessions.
Per session: quit / engaged ratio of each measure; per animal x epoch medians.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_quit_state.py [--jobs 6]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QS_FR_BASIS", "own")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import columns, session_traces  # noqa: E402
from quiet_spikes_v2 import V2, busy_frames, detrend  # noqa: E402
from wfield_local import config  # noqa: E402

OUTQ = V2 / "quit_state"
COLS = ["orofacial", "limb", "cortex"]
BANDS = ((0.5, 1.0), (1.0, 4.0), (4.0, 8.0))
SEG_S, MIN_QUIT_S = 4.0, 60.0


def band_power(y, m, fs):
    from scipy.signal import welch
    nseg = int(round(SEG_S * fs))
    e = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    P, W, fr = None, 0, None
    for a, b in zip(e[::2], e[1::2]):
        if b - a < nseg:
            continue
        fr, p = welch(y[:, a:b], fs=fs, nperseg=nseg, noverlap=nseg // 2, detrend="linear", axis=-1)
        w = 1 + (b - a - nseg) // (nseg // 2)
        P = p * w if P is None else P + p * w
        W += w
    if not W:
        return np.full((y.shape[0], len(BANDS)), np.nan), 0.0
    P /= W
    return np.stack([P[:, (fr >= lo) & (fr < hi)].mean(1) for lo, hi in BANDS], 1), W * SEG_S / 2


def extract(label):
    f = OUTQ / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    from quiet_spikes_fr import face_regressors
    Xr, ft, fs, rest, engaged, quit_s = session_traces(label, detrend=False)
    busy, badw = busy_frames(label, ft, fs)
    quiet = ~busy & ~badw & np.isfinite(Xr).all(0)
    cols = columns()
    ci = [cols.index(c) for c in COLS]
    Xd = detrend(Xr, fs)
    out = dict(quit_s=np.nan if quit_s is None else float(quit_s), fs=fs)
    try:
        from quiet_area_spikes import DETREND_S
        _, fsp = face_regressors(label, ft, int(round(DETREND_S * fs)) | 1)      # summed point-PC speed per frame
    except Exception as e:  # noqa: BLE001 -- no face video (PS93 0813) is a missing value, not a failure
        print(label, "no face:", e)
        fsp = np.full(len(ft), np.nan)
    hi = np.nanpercentile(fsp, 90) if np.isfinite(fsp).any() else np.nan
    for name, m in (("eng", quiet & engaged), ("quit", quiet & ~engaged)):
        bp, secs = band_power(Xr[ci], m, fs)
        out[f"{name}_bp"] = bp
        out[f"{name}_psd_s"] = secs
        out[f"{name}_min"] = m.sum() / fs / 60
        out[f"{name}_sd"] = 1.4826 * np.median(np.abs(Xd[ci][:, m] - np.median(Xd[ci][:, m], 1, keepdims=True)), 1) \
            if m.sum() > 100 else np.full(len(ci), np.nan)
        fm = m & np.isfinite(fsp)
        out[f"{name}_face_speed"] = float(np.median(fsp[fm])) if fm.sum() > 100 else np.nan
        out[f"{name}_face_move"] = float(np.mean(fsp[fm] > hi)) if fm.sum() > 100 else np.nan
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)
    return label


def summarise():
    import pandas as pd
    from wfield_local import epochs
    pd.set_option("display.width", 220)
    rows = []
    for f in sorted((OUTQ / "sessions").glob("*.npz")):
        lab = f.stem
        ep = epochs.epoch_of(lab)
        if ep is None or config.early_pre(lab):
            continue
        z = np.load(f)
        r = dict(label=lab, animal=lab[:4], epoch=ep, eng_min=float(z["eng_min"]), quit_min=float(z["quit_min"]))
        ok = r["quit_min"] * 60 >= MIN_QUIT_S
        for j, c in enumerate(COLS):
            r[f"{c}_sd_ratio"] = float(z["quit_sd"][j] / z["eng_sd"][j]) if ok else np.nan
            for k, (lo, hi) in enumerate(BANDS):
                r[f"{c}_{lo:g}-{hi:g}Hz_ratio"] = float(z["quit_bp"][j, k] / z["eng_bp"][j, k]) if ok else np.nan
        for m in ("face_speed", "face_move"):
            r[f"eng_{m}"], r[f"quit_{m}"] = float(z[f"eng_{m}"]), float(z[f"quit_{m}"])
            r[f"{m}_ratio"] = r[f"quit_{m}"] / r[f"eng_{m}"] if ok and r[f"eng_{m}"] > 0 else np.nan
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUTQ / "quit_state_sessions.csv", index=False)
    EP = ["pre", "acute", "subacute", "chronic"]
    print("sessions with >= 60 s quit-period quiet, animal x epoch:\n",
          S[S.quit_min * 60 >= MIN_QUIT_S].pivot_table(index="animal", columns="epoch", values="label",
                                                       aggfunc="count").reindex(columns=EP).to_string())
    print("\nquit-period quiet minutes (median):\n",
          S.pivot_table(index="animal", columns="epoch", values="quit_min", aggfunc="median").reindex(columns=EP)
          .round(1).to_string())
    for m in ["cortex_sd_ratio", "orofacial_sd_ratio", "cortex_0.5-1Hz_ratio", "cortex_1-4Hz_ratio", "cortex_4-8Hz_ratio",
              "orofacial_1-4Hz_ratio", "face_speed_ratio", "face_move_ratio"]:
        T = S.pivot_table(index="animal", columns="epoch", values=m, aggfunc="median").reindex(columns=EP)
        T.loc["all sessions"] = [S.loc[S.epoch == e, m].median() for e in EP]
        print(f"\n{m} (quit / engaged quiet; median over sessions):\n", T.round(2).to_string())
    print("\nengaged-quiet face speed by epoch (median, any session):\n",
          S.pivot_table(index="animal", columns="epoch", values="eng_face_speed", aggfunc="median")
          .reindex(columns=EP).round(3).to_string())
    (OUTQ / "README.txt").write_text(__doc__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = sorted(f.stem for f in (V2 / "sessions").glob("*.npz") if not config.early_pre(f.stem))
        res, fail = fan_sessions(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
    summarise()


if __name__ == "__main__":
    main()
