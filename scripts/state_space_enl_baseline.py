"""Per-trial lick-free ENL baseline in the cohort SVD basis, for the state-space maps (Priya 11-08: "cue minus lick-free
ENL baseline and lick minus lick-free ENL baseline may also be helpful").

THE WINDOW IS `wfield_local.enl_window`'s (the ENL decoders' and the encoder's "enl" window, rule 9): length
`lp_licks.enl_post_s("lp")` (1.0 s, LP licks), per trial [cue - 1.0 s, cue) when it holds no LP lick
(`lp_licks.lp_lick_times`), else the last 1.0 s of the latest lick-free gap (>= 1 s) in [strobe + settle, cue)
(`sliding_lags`, no span limit), else the trial is dropped. Priya 11-08: "make it the >= 1 s lick-free ENL" / "we'll
need > 1 s before cue" -- the per-trial files stop at -1.0 s, so the window is taken from the CONTINUOUS signal: the
session's SVTcorr, drift removed (cosines >= 600 s incl. the constant, as the extraction), projected onto the cohort SVD
on the analysis mask, averaged over the frames in [cue + lo, cue + hi). One (K,) vector per trial, used as (a) the
PRE-CUE map's window (vs rest / none) and (b) the baseline of the cue- and lick-aligned responses (cue / lick minus ENL).
Note (F12): the ENL window carries the pre-cue position code, so subtracting it gives the cue / lick-evoked increment
over THIS trial's pre-cue state.

Run under the rebuild profile in child processes (its write fence); the parent writes
<state_space_recovery>/enl_baseline_svd/<label>.npz: b (n_rows, K) NaN for dropped rows, lo / hi (lag window), info.

    python scripts/state_space_enl_baseline.py [--jobs 3] [LABELS ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o2_state_space_lnmf as L  # noqa: E402
import state_space_sim as S  # noqa: E402

OUT = S.ROOT / "enl_baseline_svd"
BUNDLE = S.SVD.parents[1]


def _worker(label):
    """Module-level (spawn): one session -> its arrays (the parent writes)."""
    L._env(L.CACHE)
    import pickle

    from wfield_local import combined_encoding as ce
    from wfield_local import enl_window, lp_licks
    rows = np.load(BUNDLE / "rows" / f"{label}.npz", allow_pickle=True)
    cue = rows["cue_s"].astype(float)
    meta = json.loads((BUNDLE / "bundle.json").read_text())
    try:
        licks, _t = lp_licks.lp_lick_times(label)
    except Exception as ex:                                           # noqa: BLE001
        return {"label": label, "ok": False, "reason": f"{type(ex).__name__}: {str(ex)[:120]}"}
    strobe = enl_window.strobe_for_cues(label, cue)
    win = lp_licks.enl_post_s("lp")
    lo, hi = enl_window.sliding_lags(cue, strobe, np.asarray(licks, float), win, end_lag=0.0)
    src = Path(meta["lnmf_scratch"]) / "inputs" / f"{label}.pkl"
    if not src.exists():
        src = Path(meta["enc_scratch"]) / "inputs" / f"{label}.pkl"
    x = pickle.loads(src.read_bytes())
    V, ft = np.asarray(x["imaging"][0], np.float64), np.asarray(x["imaging"][1], float)
    del x
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    V -= Q @ (Q.T @ V)
    msk = np.load(BUNDLE / "mask.npy").ravel().astype(bool)
    U = np.load(Path(meta["uatlas_root"]) / meta["uatlas"][label])
    u = np.nan_to_num(U.reshape(-1, U.shape[2])).astype(np.float64)[msk]
    del U
    Uc = np.load(S.SVD.parent / "basis" / "U_cohort.npy").astype(np.float64)
    R = V @ (Uc.T @ u).T                                               # (T, K_cohort): the signal the per-trial files hold
    b = np.full((len(cue), R.shape[1]), np.nan, np.float32)
    for i in np.flatnonzero(np.isfinite(lo)):
        m = (ft >= cue[i] + lo[i]) & (ft < cue[i] + hi[i])
        if m.any():
            b[i] = R[m].mean(0)
    ok = np.isfinite(b).all(1)
    slid = np.isfinite(lo) & (np.abs(lo + win) > 1e-3)
    info = {"n_rows": int(len(cue)), "n_kept": int(ok.sum()), "n_slid": int(slid.sum()),
            "n_no_strobe": int(np.isnan(strobe).sum()), "win_s": float(win),
            "lo_median": float(np.nanmedian(lo)) if np.isfinite(lo).any() else None}
    return {"label": label, "ok": bool(ok.any()), "b": b.astype(np.float32), "lo": lo, "hi": hi, "info": json.dumps(info)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = a.labels or [x for x in (BUNDLE / "labels.txt").read_text().split() if x]
    todo = [lab for lab in labels if not (OUT / f"{lab}.npz").exists()]
    from wfield_local import analysis_kit as kit
    if a.jobs > 1 and len(todo) > 1:
        got, fail = kit.fan_sessions(todo, _worker, jobs=a.jobs, key=lambda x: x)
    else:                         # never in THIS process: the profile's write fence would block the writes below
        from concurrent.futures import ProcessPoolExecutor
        got, fail = [], []
        with ProcessPoolExecutor(1) as ex:
            for lab in todo:
                try:
                    got.append((lab, ex.submit(_worker, lab).result()))
                except Exception as e:                                 # noqa: BLE001
                    fail.append((lab, e))
    kept = []
    for _i, r in got:
        np.savez(OUT / f"{r['label']}.npz", **{k: v for k, v in r.items() if k != "label"})
        if r["ok"]:
            i = json.loads(r["info"])
            kept.append(i["n_kept"] / max(i["n_rows"], 1))
    bad = [r["label"] for _i, r in got if not r["ok"]]
    print(f"{len(got)} done; trials kept (lick-free >= 1 s ENL window): median {np.median(kept) if kept else np.nan:.2f} "
          f"of rows; {len(bad)} without a baseline {bad}; {len(fail)} failed {[(i, str(e)[:80]) for i, e in fail]}")


if __name__ == "__main__":
    main()
