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
RESTnl VARIANT (`extract(jobs, "restnl")` / `--rest-variant restnl`): the CURRENT engaged gate and the LICK-FREE rest
level, both from <OUT_W>/precue_restnl/<label>.npz (lick_precue_restw.py --rest-variant restnl), written to
<OUT_W>/cue_to_lick/sessions_restnl/<label>.npz (`restw` = the RESTnl level, `restw_old` = RESTW). The rev-up cache holds
the OLD gate, so the first lick per trial is rebuilt here with revup_neural.session_worker's rule, verbatim (`_first_licks`;
checked equal to the rev-up cache wherever the two trial sets agree).
Usage: imported by lick_cue_to_lick.py / lick_cue_to_lick_precue.py (`extract(jobs)`), or run directly [--jobs 4]
[--rest-variant restnl].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import restnl  # noqa: E402

from wfield_local import mask_keys  # noqa: E402

OUT = il.OUT_W / "cue_to_lick"
SESS = OUT / "sessions"
SESS_NL = OUT / restnl.cache("sessions_restnl")   # per RESTnl rule (strict = sessions_restnl, wr = _wr)
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
            mask_keys.check_obj(z, f)             # stale-read guard: built on the current analysis masks (11-09)
            return {"label": label, "cached": True}
    from wfield_local import combined_encoding as ce
    src = np.load(il.OUT_W / "precue_restw" / f"{label}.npz")
    mask_keys.check_obj(src, il.OUT_W / "precue_restw" / f"{label}.npz")
    rv = np.load(il.OUT_W / "revup" / "sessions" / f"{label}.npz")
    mask_keys.check_obj(rv, il.OUT_W / "revup" / "sessions" / f"{label}.npz")   # stale-read guard (11-09)
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
                        restw_ok=bool(src["restw_ok"]), animal=str(d["animal"]), epoch=str(d["epoch"]), maxdiff=maxdiff, **mask_keys.stamp())
    return {"label": label, "cached": False, "n": len(cue), "maxdiff": maxdiff}


def sess_dir(var: str = "restw"):
    return SESS if var == "restw" else SESS_NL


# ------------------------------------------------------------------------------------------------ trial-set captions
# Priya 11-09: "EVERY figure must state EXACTLY which trials were included / excluded". The trials of this cache (and of
# iti_licks.extract's first licks) are `combined_encoding.load_behaviour`'s trials.engaged = the ENGAGEMENT gate of
# `orofacial_cohort.engagement_keep` (OROFACIAL_GATE), NOT the camera working classes; the words come from that switch.
def engaged_text() -> str:
    """The engaged-trial gate (trials.engaged of `combined_encoding.load_behaviour`) in words, for this process."""
    from wfield_local import orofacial_cohort as oc
    if not oc.ENGAGED_ONLY:
        return "all trials (no engagement gate)"
    g = oc.ENGAGEMENT_GATE
    if g == "reference":
        try:
            tail = f" (>= {oc._eng_cfg('tail_min_misses')} trailing misses)"
        except Exception:                                              # noqa: BLE001 - a caption never kills a figure
            tail = ""
        return ("engaged trials (behavior pipeline reference-position gate, spout_behavior.reference_engagement: "
                f"terminal non-recovering collapse and sated tail{tail} removed; not the camera classes)")
    import trial_attempts as ta
    att = " + ".join(ta.ATTEMPT_CLASSES)
    if g == "canonical_attempted":
        return f"attempted trials ({att}, trial_attempts.csv) inside the reference-gate engaged period"
    return f"attempted trials ({att}, trial_attempts.csv)"


def first_lick_rule() -> str:
    """The first-lick rule of this cache (`_first_licks`, revup_neural.session_worker's rule) in words."""
    return "first lick = the first behavior-table lick onset in cue..trial stop (revup_neural rule)"


def rest_text(var: str = "restw") -> str:
    return ("sessions with a RESTnl (lick-free rest) level" if var == "restnl"
            else "sessions with a RESTW level (restw_ok; trial set = the precue_restw cache's engaged trials)")


def _first_licks(label: str, B, T, cue: np.ndarray, stop: np.ndarray, t_end: float) -> dict:
    """revup_neural.session_worker's first-lick rule, verbatim: in_window = first behavior lick onset in cue..stop;
    else late = first ITI video lick (licks_all.csv context iti) after stop and before the next trial's strobe; quiet =
    time since the last lick (behavior + video) before the cue; npre = licks in -2..0 s."""
    import pandas as pd

    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    st_all = np.sort(T.strobe_s.to_numpy(float)[np.isfinite(T.strobe_s.to_numpy(float))])
    j = np.searchsorted(st_all, cue, side="right")
    nxt = np.where(j < len(st_all), st_all[np.minimum(j, len(st_all) - 1)], t_end)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["context", "daq_on_s"])
    bo = np.sort(np.asarray(B.lick_on_s, float)[np.isfinite(B.lick_on_s)])
    ao = np.sort(A.daq_on_s.to_numpy(float)[np.isfinite(A.daq_on_s.to_numpy(float))])
    iti_on = np.sort(A.daq_on_s.to_numpy(float)[(A.context == "iti").to_numpy() & np.isfinite(A.daq_on_s.to_numpy(float))])
    t_all = np.unique(np.concatenate([bo, ao]))
    n = len(cue)
    fl, phase = np.full(n, np.nan), np.full(n, "never", dtype=object)
    quiet, npre = np.full(n, np.nan), np.zeros(n, int)
    for i in range(n):
        k = np.searchsorted(bo, cue[i])
        if k < len(bo) and bo[k] <= stop[i]:
            fl[i], phase[i] = bo[k], "in_window"
        else:
            k = np.searchsorted(iti_on, stop[i], side="right")
            if k < len(iti_on) and iti_on[k] < nxt[i]:
                fl[i], phase[i] = iti_on[k], "late"
        k = np.searchsorted(t_all, cue[i]) - 1
        quiet[i] = cue[i] - t_all[k] if k >= 0 else np.inf
        npre[i] = int(((t_all >= cue[i] - 2.0) & (t_all < cue[i])).sum())
    return {"fl": fl, "phase": phase.astype(str), "quiet": quiet, "npre": npre}


def session_worker_restnl(label: str) -> dict:
    """RESTnl variant (module-level for spawn) -> <OUT_W>/cue_to_lick/sessions_restnl/<label>.npz."""
    f = SESS_NL / f"{label}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == VERSION:
            mask_keys.check_obj(z, f)             # stale-read guard (11-09)
            return {"label": label, "cached": True}
    from wfield_local import combined_encoding as ce
    src = np.load(il.OUT_W / restnl.cache("precue_restnl") / f"{label}.npz")
    mask_keys.check_obj(src, il.OUT_W / restnl.cache("precue_restnl") / f"{label}.npz")
    old = np.load(il.OUT_W / "precue_restw" / f"{label}.npz")
    d = il._load(label)
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
        raise ValueError(f"{label}: columns differ from the precue_restnl cache")
    idx = [cs.index(c) for c in COLS]
    R = Rs[:, idx]
    B = il._behaviour(label)
    T = B.trials
    Eg = T[T.engaged.astype(bool) & np.isfinite(T.cue_s.to_numpy(float))].reset_index(drop=True)
    pos = np.asarray(Eg.pos.astype(str).to_numpy(), str)
    if not np.array_equal(pos, src["pos"].astype(str)):
        raise ValueError(f"{label}: engaged trials differ from the precue_restnl cache")
    cue = Eg.cue_s.to_numpy(float)
    stop = Eg.stop_s.to_numpy(float)
    stop = np.where(np.isfinite(stop), stop, cue + 3.5)
    FL = _first_licks(label, B, T, cue, stop, ft[-1])
    fl = FL["fl"]
    # check against the rev-up cache (old gate) on the trials both sets share (matched by cue time)
    rv = np.load(il.OUT_W / "revup" / "sessions" / f"{label}.npz")
    mask_keys.check_obj(rv, il.OUT_W / "revup" / "sessions" / f"{label}.npz")   # stale-read guard (11-09)
    rc = np.asarray(rv["cue"], float)
    jj = np.searchsorted(cue, rc).clip(0, len(cue) - 1)
    hit = np.abs(cue[jj] - rc) < 1e-6
    a, b = np.asarray(rv["fl"], float)[hit], fl[jj[hit]]
    fl_mismatch = int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum()
                      + (rv["phase"].astype(str)[hit] != FL["phase"][jj[hit]]).sum())

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
    on05 = np.isclose(np.round(LAGS_C / 0.05), LAGS_C / 0.05)
    w = (LAGS_C >= -1.0 - 1e-9) & (LAGS_C <= 1e-9) & on05
    maxdiff = float(np.nanmax(np.abs(np.nanmean(Xc[:, w], 1) - np.asarray(src["p10"], float)[:, idx])))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, version=VERSION, cols=np.array(COLS), pos=pos, cue=cue, stop=stop, fl=fl,
                        phase=FL["phase"], quiet=FL["quiet"], npre=FL["npre"], Xc=Xc, Xl=Xl,
                        restw=np.asarray(src["restw"], float)[idx], restw_ok=bool(src["restw_ok"]),
                        restw_old=np.asarray(old["restw"], float)[[[str(c) for c in old["cols"]].index(c) for c in COLS]],
                        animal=str(d["animal"]), epoch=str(d["epoch"]), maxdiff=maxdiff, fl_mismatch=fl_mismatch,
                        n_shared=int(hit.sum()), **mask_keys.stamp())
    return {"label": label, "cached": False, "n": len(cue), "maxdiff": maxdiff, "fl_mismatch": fl_mismatch,
            "n_shared": int(hit.sum())}


def extract(jobs: int = 4, var: str = "restw") -> list:
    from wfield_local import analysis_kit as ak
    labs = il.labels()
    worker = session_worker_restnl if var == "restnl" else session_worker
    res, fails = ak.fan_sessions(labs, worker, jobs=min(jobs, 4))
    if fails:
        raise SystemExit(f"{len(fails)} sessions failed: {fails}")
    md = [v.get("maxdiff", np.nan) for _x, v in res if not v.get("cached")]
    if md:
        print(f"== sanity: max |trial -1..0 s mean (0.05 s points) - precue p10| = {np.nanmax(md):.3f}")
    fm = [(x, v["fl_mismatch"], v["n_shared"]) for x, v in res if "fl_mismatch" in v]
    if fm:
        print(f"== first-lick rule vs rev-up cache on shared trials: {sum(m for _x, m, _n in fm)} mismatches over "
              f"{sum(n for _x, _m, n in fm)} shared trials")
    return labs


if __name__ == "__main__":
    if "--one" in sys.argv:
        import time
        t0 = time.time()
        print(session_worker(sys.argv[sys.argv.index("--one") + 1]), f"{time.time() - t0:.1f} s")
    else:
        extract(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4, restnl.variant())
