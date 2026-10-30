"""CUE-TO-LICK per-trial cache (Priya 10-29, tasks A-D): per ENGAGED trial, raw cue-aligned and first-lick-aligned area
traces for the eight areas of the restw trace figure, both hemispheres, written to <OUT_W>/cue_to_lick/sessions/<label>.npz.

TRACE  = exactly the construction of `lick_precue_restw_traces.session_worker` / `revup_neural.session_worker` (ROI v8
         masks, area means of U_atlas on the adopted SVT, slow drift basis removed (incl. the constant, so the raw level is
         relative to the session mean), x1e3, NO per-trial baseline), with the ALM / tjM1 / MOs_rest / MOp_rest split
         (`iti_licks.with_rest`). Stored RAW (not restw-subtracted) so task D can decompose raw = (raw - restw) + restw;
         the session restw vector is stored next to it (read from <OUT_W>/precue_restw/<label>.npz, the Fig 14.21 numbers).
TRIALS = the precue_restw engaged-trial set (checked equal); the first lick per trial (time, phase in_window / late /
         never, licks in -2..0 s before the cue, quiet time before the cue) is READ from the rev-up cache
         <OUT_W>/revup/sessions/<label>.npz (revup_neural.session_worker's rule), so both analyses share one definition.
GRID   = 0.025 s (the cue -> lick windows are short: median ~0.2 s). Cue-aligned -3..+4 s; lick-aligned -2..+1 s.
Usage: imported by lick_cue_to_lick.py / lick_cue_to_lick_precue.py (`extract(jobs)`), or run directly [--jobs 4].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = il.OUT_W / "cue_to_lick"
SESS = OUT / "sessions"
VERSION = 1
DT = 0.025
LAGS_C = np.round(np.arange(-3.0, 4.0 + 1e-9, DT), 4)
LAGS_L = np.round(np.arange(-2.0, 1.0 + 1e-9, DT), 4)
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "MOs_rest", "MOp_rest", "SSp-bfd", "VISp"]
REF = {"MOs_rest", "MOp_rest", "SSp-bfd", "VISp"}
HEMI = (("right", "contralesional (right)"), ("left", "ipsilesional (left)"))
COLS = [f"{a}_{h}" for a in AREAS for h, _ in HEMI if not (a == "VISp" and h == "left")]


def session_worker(label: str) -> dict:
    """One session (module-level for spawn) -> <OUT_W>/cue_to_lick/sessions/<label>.npz."""
    f = SESS / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == VERSION:
            return {"label": label, "cached": True}
    from wfield_local import combined_encoding as ce
    src = np.load(il.OUT_W / "precue_restw" / f"{label}.npz")
    rv = np.load(il.OUT_W / "revup" / "sessions" / f"{label}.npz")
    d = il._load(label)
    # == lick_precue_restw_traces.session_worker's trace construction
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    if cols != list(d["cols"]):
        raise ValueError(f"{label}: ROI columns differ from the session file")
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    Rs, cs = il.with_rest(R, cols, label, basis="split")
    scols = [str(c) for c in src["cols"]]
    if cs != scols:
        raise ValueError(f"{label}: columns differ from the precue_restw cache")
    idx = [cs.index(c) for c in COLS]
    R = Rs[:, idx]
    restw = np.asarray(src["restw"], float)[idx]
    pos = rv["pos"].astype(str)
    if not np.array_equal(pos, src["pos"].astype(str)):
        raise ValueError(f"{label}: rev-up trials differ from the precue_restw cache")
    cue, fl = np.asarray(rv["cue"], float), np.asarray(rv["fl"], float)

    def samp(t, lags):
        X = il._sample(R, ft, t, lags).astype(np.float32)
        out = (t[:, None] + lags[None, :] < ft[0]) | (t[:, None] + lags[None, :] > ft[-1])
        X[out] = np.nan
        return X
    Xc = samp(cue, LAGS_C)
    ok = np.isfinite(fl)
    Xl = np.full((len(cue), len(LAGS_L), len(COLS)), np.nan, np.float32)
    if ok.any():
        Xl[ok] = samp(fl[ok], LAGS_L)
    # sanity vs precue_restw: trial -1..0 s mean on the 0.05 s points vs the stored p10 (exact)
    on05 = np.isclose(np.round(LAGS_C / 0.05), LAGS_C / 0.05)
    w = (LAGS_C >= -1.0 - 1e-9) & (LAGS_C <= 1e-9) & on05
    maxdiff = float(np.nanmax(np.abs(np.nanmean(Xc[:, w], 1) - np.asarray(src["p10"], float)[:, idx])))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, version=VERSION, cols=np.array(COLS), pos=pos, cue=cue, stop=rv["stop"], fl=fl,
                        phase=rv["phase"].astype(str), quiet=rv["quiet"], npre=rv["npre"], Xc=Xc, Xl=Xl, restw=restw,
                        restw_ok=bool(src["restw_ok"]), animal=str(d["animal"]), epoch=str(d["epoch"]), maxdiff=maxdiff)
    return {"label": label, "cached": False, "n": len(cue), "maxdiff": maxdiff}


def extract(jobs: int = 4) -> list:
    from wfield_local import analysis_kit as ak
    labs = il.labels()
    res, fails = ak.fan_sessions(labs, session_worker, jobs=min(jobs, 4))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    md = [v.get("maxdiff", np.nan) for _x, v in res if not v.get("cached")]
    if md:
        print(f"== sanity: max |trial -1..0 s mean (0.05 s points) - precue_restw p10| = {np.nanmax(md):.3f}")
    return labs


if __name__ == "__main__":
    if "--one" in sys.argv:
        import time
        t0 = time.time()
        print(session_worker(sys.argv[sys.argv.index("--one") + 1]), f"{time.time() - t0:.1f} s")
    else:
        extract(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4)
