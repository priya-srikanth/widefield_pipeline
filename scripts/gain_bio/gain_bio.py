"""GAIN_BIO (Priya 11-04): is the post-stroke global SD rise (SD_RISE_2026-11-04) a biological gain or measurement?
Read-only diagnostic, per session. Writes only <out>/sessions/<label>.npz (+ <out>/sessions.csv, the session list).

PROVENANCE: ported 2026-11-05 from M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/gain_bio_1104/gain_bio.py
  (run 11-04, 107 / 107 sessions; docs/status/GAIN_BIO_2026-11-04.md sections 2 / 7; deck slide 102). Computation
  unchanged. Changes:
  - output root = gb_common.out_root() (GAIN_BIO_OUT / --out, else lp_licks.output_dir(<cohort_out>/gain_bio): under
    the render profile <rebuild root>/labcams/<run tag>/gain_bio), not the script's own folder;
  - quiet-event input = gb_common.quiet_dir() (--quiet-dir, else THIS run's routed quiet_spikes_v2 output, v2_quiet),
    not the adopted labcams/cohort1_quiet_spikes/v2_quiet;
  - the hemo manifest = config.hemo_manifest_path (the selected variant, the one combined_encoding.load_imaging reads
    via hemo_dir()), not hemo_{ADOPTED};
  - the session list = gb_common.session_table() (--sessions CSV, else the 107-session rule; written to
    <out>/sessions.csv), not sd_rise_1104/tables/f0_sessions.csv;
  - no repo / cache path in code (WFIELD_INPUT_CACHE comes from the caller's env);
  - a RebuildFenceError in the face / running loaders is raised, not turned into NaN movement columns.
  - main() refuses an empty / missing quiet-events folder (the original silently gave R5far = no frames: near_event_mask
    returns None when a session file or the pre SD is missing).
  Priya 11-06 (VERSION 2; the 11-04 run = VERSION 1):
  - quiet events = the LP version (gb_common.QUIET_SUBDIR v2_quiet_lp: quiet_spikes_v2 with QS_LICK_SOURCE=daq+lp);
  - TASK = cue .. cue + 2 s of the camera WORKING PERIOD cues only (gb_common.task_cues, TASK_ROWS "period"; --task-rows);
  - the 415 comparison is made like with like downstream (summarise / gain_stats / fig_sessions: gb_common.NONCA vs
    CA_REF); this script's per-session numbers are unchanged by that.
  The 11-04 run read hemo_meegkit_hpfit (46 split-basis areas + cortex); a rerun reads whatever the env selects (the
  render profile: meegkit_hpfit_lag_refitT_swapmask; area_basis.ROI_VERSION; the camera trial gate's quit for the
  engaged frames).

    PYTHONPATH=<repo>;<repo>/scripts python scripts/gain_bio/gain_bio.py [--jobs 3] [--out DIR] [--quiet-dir DIR]
        [--sessions CSV] [LABEL ...]

SIGNALS (all dF/F %, on the same U_atlas basis; `combined_encoding.load_imaging` + raw SVT.npy, slots by
`hemo_variants.functional_channel`, the quiet_hemo_check loader logic):
  corr      production hemo-corrected 470 (the selected hemo variant's SVTcorr) -- what the decoder / lick slides see
  corr_hp   corr, 0.1 Hz high-pass (butter 2, zero phase)
  r470_hp   UNCORRECTED 470 (functional half of raw SVT), same high-pass
  r415_hp   415 nm isosbestic half of raw SVT, same high-pass  (non-calcium control)
AREAS: this repo's scripts/iti_licks.roi_masks (the quiet-events arm's masks: every Allen area per hemisphere on the
eroded stat mask + ALM / tjM1 = anterolateral 40 % of MOs / MOp, + cortex); _left = ipsilesional (image left).
FRAME SETS (frames within 3 of a swapped pair dropped from all):
  ALL     every frame
  R5      restdock05 quiet frame AND engaged (rest_centre_1104/rest_stats.py R5 rule)
  R5far   R5 and > 2 s from any quiet event (v2 detector on LP quiet frames, orofacial_L/R + limb_L/R, both
          polarities, > 4 x the animal's August-pre robust SD; the R5q rule with a 2 s window both sides and the limb
          columns added)
  TASK    cue .. cue + 2 s of every camera-working-period cue (gb_common.task_cues; engaged frames)
PER SET: area SD per signal; pixel variance map per signal (diag(U C U'), C = component covariance over the set),
binned 4 x 4 px onto the 540 x 640 atlas grid.
MOVEMENT / AROUSAL (R5 frames): cam4 FaceRhythm point-PC face speed (quiet_spikes_fr.face_regressors, own basis; sum of
the 20 PC speeds), 2 s centred smoothing; treadmill speed (behavior_events.session_speed) 2 s smoothed. No pupil
tracking exists for cohort 1 (quiet_quit_state.py docstring). R5 split into chunks of <= 62 frames (2 s) inside
contiguous R5 runs with a constant far / near flag; per chunk: n, far flag, mean face speed, mean running speed,
and per signal x area the sum and sum of squares of (x - session R5 mean), so any speed-matched subset's variance can
be rebuilt exactly.
FACE REGRESSION (R5, corr_hp): ridge (alpha 1e3 on z-scored regressors) of each area trace on the FaceRhythm design
(factors + point PCs + PC speeds, 2-s high-passed, lags -0.3 / 0 / +0.3 s) + smoothed face and treadmill speed;
10 contiguous folds, out of fold; residual SD and R2 per area.
F0: frames_average_atlas 470 / 415, binned like the maps, and per area.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_common  # noqa: E402

SIGS = ("corr", "corr_hp", "r470_hp", "r415_hp")
SETS = ("ALL", "R5", "R5far", "TASK")
BIN = 4
CHUNK = 62
EV_K, EV_COLS, FAR_S = 4.0, ("orofacial_left", "orofacial_right", "limb_left", "limb_right"), 2.0
TASK_S = 2.0
VERSION = 2                       # 2 = Priya 11-06: LP quiet events, camera-working-period TASK


def OUTD() -> Path:
    return gb_common.out_root() / "sessions"


def V2S() -> Path:
    return gb_common.quiet_dir() / "sessions"


def _paths():
    gb_common.add_paths()
    os.environ["QS_FR_BASIS"] = "own"


def _fence_error(ex) -> bool:
    from wfield_local.rebuild_root import RebuildFenceError
    return isinstance(ex, RebuildFenceError)


def event_sd():
    from wfield_local import epochs
    acc = {}
    for f in sorted(V2S().glob("*.npz")):
        if epochs.epoch_of(f.stem) == "pre" and f.stem[-4:] >= "0806":
            acc.setdefault(f.stem[:4], []).append(np.load(f)["sd_q"])
    return {a: np.nanmedian(v, 0) for a, v in acc.items()}


def near_event_mask(label, T, sd):
    f = V2S() / f"{label}.npz"
    if not f.exists() or sd is None:
        return None, 0
    z = np.load(f)
    cols, fs = list(z["cols"]), float(z["fs"])
    w = int(round(FAR_S * fs))
    m = np.zeros(T, bool)
    n = 0
    for kind in ("pos", "neg"):
        a = z[kind]
        for c in EV_COLS:
            i = cols.index(c)
            r = a[(a[:, 0] == i) & (a[:, 2] > EV_K * sd[i])]
            for p in r[:, 1].astype(int):
                m[max(0, p - w):min(T, p + w + 1)] = True
            n += len(r)
    return m, n


def binmap(vals, flat, shape):
    H, W = shape
    g = np.zeros(H * W)
    c = np.zeros(H * W)
    g[flat] = vals
    c[flat] = 1
    g = g.reshape(H // BIN, BIN, W // BIN, BIN).sum((1, 3))
    c = c.reshape(H // BIN, BIN, W // BIN, BIN).sum((1, 3))
    out = np.full(g.shape, np.nan, np.float32)
    ok = c >= BIN * BIN / 2
    out[ok] = g[ok] / c[ok]
    return out


def smooth_nan(x, w):
    from scipy.ndimage import uniform_filter1d
    ok = np.isfinite(x)
    s = uniform_filter1d(np.where(ok, x, 0.0), w, mode="nearest")
    d = uniform_filter1d(ok.astype(float), w, mode="nearest")
    out = s / np.maximum(d, 1e-9)
    out[d < 0.5] = np.nan
    return out


def task_mask(ft, cue_s, keep, w, n):
    """``(mask, n_cue)``: frames [i, i + w) from the first frame at or after each KEPT cue (``cue_s`` s, ``keep``
    bool per cue); a cue whose first frame i is not in 0 <= i < n - w is dropped whole, as in the 11-04 run."""
    mask = np.zeros(n, bool)
    n_cue = 0
    for i in np.searchsorted(ft, np.asarray(cue_s, float)[np.asarray(keep, bool)]):
        if 0 <= i < n - w:
            mask[i:i + w] = True
            n_cue += 1
    return mask, n_cue


def chunks(mask, flag):
    """[(start, stop)] runs of `mask` with constant `flag`, cut into <= CHUNK frames."""
    out = []
    m = mask.astype(np.int8)
    key = np.where(mask, 1 + flag.astype(np.int8), 0)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], key, [0]])) != 0)
    for a, b in zip(edges[:-1], edges[1:]):
        if not m[a]:
            continue
        for s in range(a, b, CHUNK):
            out.append((s, min(b, s + CHUNK)))
    return out


def run(label):
    _paths()
    from wfield_local.parallel import pin_blas
    pin_blas()
    f = OUTD() / f"{label}.npz"
    if f.exists() and int(np.load(f)["version"]) == VERSION:
        return str(f)
    from scipy.signal import butter, sosfiltfilt

    from iti_licks import roi_masks
    from wfield_local import combined_encoding as ce
    from wfield_local import config, rest_engagement
    from wfield_local.analysis_kit import session_behavior
    from wfield_local.behavior_events import session_speed
    from wfield_local.hemo_variants import functional_channel
    from wfield_local.quiet_periods import quiet_frame_path
    s = next(x for x in config.load_sessions() if x["label"] == label)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    n, K = V.shape
    fs = 1.0 / np.median(np.diff(ft))
    res = Path(s["mc"]) / "wfield_local_results"
    fc = functional_channel(label)
    man = json.loads(Path(config.hemo_manifest_path(res)).read_text())
    if int(man.get("functional_channel", -1)) != fc:
        raise RuntimeError(f"{label}: functional slot {fc} vs manifest {man.get('functional_channel')}")
    # ---------------------------------------------------------------- signals (T x K float32)
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    if svt.shape[0] != K or svt.shape[1] // 2 < n:
        raise RuntimeError(f"{label}: SVT {svt.shape} vs V {V.shape}")
    sos = butter(2, 0.1, "highpass", fs=fs, output="sos")
    S = {"corr": np.asarray(V, np.float32)}
    S["corr_hp"] = sosfiltfilt(sos, S["corr"].astype(np.float64), axis=0).astype(np.float32)
    for nm, slot in (("r470_hp", fc), ("r415_hp", 1 - fc)):
        x = np.asarray(svt[:, slot::2][:, :n], np.float64).T
        S[nm] = sosfiltfilt(sos, x - x.mean(0), axis=0).astype(np.float32)
        del x
    del svt
    # ---------------------------------------------------------------- areas
    M = roi_masks(parea, pstat, names, flat, shape)
    anames = sorted(M)
    W = np.array([Ub[M[a]].mean(0) if M[a].any() else np.full(K, np.nan) for a in anames], np.float64)
    npix = np.array([int(M[a].sum()) for a in anames])
    # ---------------------------------------------------------------- frame sets
    bad = np.zeros(n, bool)
    sw = res / "swapped_pairs.npz"
    if sw.exists():
        b = np.load(sw)["bad"][:n].astype(float)
        bad[:len(b)] = np.convolve(b, np.ones(7), "same") > 0
    qp = quiet_frame_path(s["mc"])
    qf = np.zeros(n, bool)
    if qp is not None:
        q = np.load(qp).astype(bool)
        qf[:min(n, len(q))] = q[:n]
    eng, gate_note = rest_engagement.engaged_frame_mask(s, n)
    eng = np.asarray(eng, bool)[:n]
    R5 = qf & eng & ~bad
    sd_ev = event_sd().get(label[:4])
    near, n_ev = near_event_mask(label, n, sd_ev)
    R5far = R5 & ~near if near is not None else np.zeros(n, bool)
    TASK = np.zeros(n, bool)
    bh = session_behavior(label)
    n_cue, task_note = 0, "no behaviour"
    if bh is not None:
        cue_s = np.asarray(bh.cue_samples, float) / bh.sample_rate
        keep, task_note = gb_common.task_cues(label, cue_s)
        TASK, n_cue = task_mask(ft, cue_s, keep, int(round(TASK_S * fs)), n)
        TASK &= eng & ~bad
    SETM = {"ALL": ~bad, "R5": R5, "R5far": R5far, "TASK": TASK}
    out = dict(version=VERSION, label=label, areas=np.array(anames), npix=npix, fs=fs, T=n, fc=fc,
               n_events=n_ev, n_cue=n_cue, gate_note=str(gate_note), sigs=np.array(SIGS), sets=np.array(SETS),
               task_rows=gb_common.task_rows(), task_note=str(task_note), quiet_src=str(gb_common.quiet_dir()))
    # ---------------------------------------------------------------- per set: area SD + pixel variance maps
    Ubd = Ub.astype(np.float64)
    for st in SETS:
        m = SETM[st]
        out[f"{st}_n"] = int(m.sum())
        for sg in SIGS:
            if m.sum() < 200:
                out[f"{st}_{sg}_sd"] = np.full(len(anames), np.nan)
                out[f"{st}_{sg}_map"] = np.full((shape[0] // BIN, shape[1] // BIN), np.nan, np.float32)
                continue
            X = S[sg][m].astype(np.float64)
            C = np.cov(X, rowvar=False)
            av = np.einsum("ak,kl,al->a", np.nan_to_num(W), C, np.nan_to_num(W))
            av[~np.isfinite(W).all(1)] = np.nan
            out[f"{st}_{sg}_sd"] = 100 * np.sqrt(av)
            pv = np.einsum("pk,pk->p", Ubd @ C, Ubd)
            out[f"{st}_{sg}_map"] = binmap(1e4 * pv, flat, shape)          # (dF/F %)^2
            del X
    # ---------------------------------------------------------------- movement / arousal at rest
    from quiet_area_spikes import DETREND_S
    from quiet_spikes_fr import face_regressors
    w2 = int(round(2.0 * fs))
    try:
        Rf, sp = face_regressors(label, ft, int(round(DETREND_S * fs)) | 1)
        face = smooth_nan(np.asarray(sp, float), w2)
        fr_note = "ok"
    except Exception as ex:                                     # no FaceRhythm for this session
        if _fence_error(ex):
            raise
        Rf, face, fr_note = None, np.full(n, np.nan), f"face: {ex}"
    try:
        tt, rs = session_speed(Path(s["h5"]))
        run_sp = smooth_nan(np.interp(ft, tt, rs), w2)
    except Exception as ex:
        if _fence_error(ex):
            raise
        run_sp, fr_note = np.full(n, np.nan), fr_note + f"; run: {ex}"
    out["move_note"] = fr_note
    for st in ("R5", "TASK"):
        m = SETM[st]
        out[f"{st}_face_med"] = float(np.nanmedian(face[m])) if m.any() else np.nan
        out[f"{st}_face_nanfrac"] = float(np.mean(~np.isfinite(face[m]))) if m.any() else np.nan
        out[f"{st}_run_med"] = float(np.nanmedian(run_sp[m])) if m.any() else np.nan
        out[f"{st}_run_p95"] = float(np.nanpercentile(run_sp[m], 95)) if m.any() else np.nan
    # chunks (R5)
    ch = chunks(R5, R5far)
    out["ch_se"] = np.array(ch, np.int64).reshape(-1, 2)
    out["ch_far"] = np.array([bool(R5far[a]) for a, _ in ch])
    out["ch_face"] = np.array([np.nanmean(face[a:b]) if np.isfinite(face[a:b]).any() else np.nan for a, b in ch])
    out["ch_run"] = np.array([np.nanmean(run_sp[a:b]) if np.isfinite(run_sp[a:b]).any() else np.nan for a, b in ch])
    Wn = np.nan_to_num(W)
    for sg in SIGS:
        X = 100 * (S[sg][R5].astype(np.float64) @ Wn.T)                # R5 frames x areas
        mu = X.mean(0)
        idx = np.cumsum(R5) - 1                                          # frame -> row in X
        S1 = np.zeros((len(ch), len(anames)), np.float32)
        S2 = np.zeros((len(ch), len(anames)), np.float32)
        for j, (a, b) in enumerate(ch):
            d = X[idx[a]:idx[b - 1] + 1] - mu
            S1[j], S2[j] = d.sum(0), (d * d).sum(0)
        out[f"ch_{sg}_S1"], out[f"ch_{sg}_S2"] = S1, S2
        out[f"R5_{sg}_mu"] = mu
        if sg == "corr_hp" and Rf is not None:
            out.update(face_regression(X, Rf[R5], face[R5], run_sp[R5], fs))
        del X
    # ---------------------------------------------------------------- F0
    faa = np.load(res / "allen_aligned_affine8v1" / "frames_average_atlas.npy").astype(np.float64)
    f470, f415 = faa[fc].ravel()[flat], faa[1 - fc].ravel()[flat]
    out["f0_470_map"], out["f0_415_map"] = binmap(f470, flat, shape), binmap(f415, flat, shape)
    out["f0_470_area"] = np.array([f470[M[a]].mean() if M[a].any() else np.nan for a in anames])
    out["f0_415_area"] = np.array([f415[M[a]].mean() if M[a].any() else np.nan for a in anames])
    gb_common.guard(OUTD())
    np.savez_compressed(f, **out)
    return str(f)


def face_regression(X, R, face, run_sp, fs):
    """Out-of-fold ridge of R5 area traces (corr_hp) on the lagged FaceRhythm design + smoothed speeds."""
    lags = [int(round(x * fs)) for x in (-0.3, 0.0, 0.3)]
    n = len(R)
    cols = []
    for k in lags:
        Y = np.full_like(R, np.nan)
        if k >= 0:
            Y[:n - k] = R[k:]
        else:
            Y[-k:] = R[:k]
        cols.append(Y)
    D = np.column_stack(cols + [face[:, None], run_sp[:, None]])
    ok = np.isfinite(D).all(1)
    if ok.sum() < 2000:
        return {"fr_ok_frac": float(ok.mean())}
    D, Xo = D[ok], X[ok]
    D = (D - D.mean(0)) / np.where(D.std(0) > 0, D.std(0), 1.0)
    fold = np.minimum((np.arange(len(D)) * 10) // len(D), 9)
    P = np.zeros_like(Xo)
    for k in range(10):
        tr, te = fold != k, fold == k
        A = D[tr]
        mu = Xo[tr].mean(0)
        B = np.linalg.solve(A.T @ A + 1e3 * np.eye(A.shape[1]), A.T @ (Xo[tr] - mu))
        P[te] = D[te] @ B + mu
    resid = Xo - P
    return {"fr_ok_frac": float(ok.mean()), "fr_raw_sd": Xo.std(0), "fr_res_sd": resid.std(0),
            "fr_r2": 1 - resid.var(0) / Xo.var(0)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=3)
    gb_common.common_args(ap)
    ap.add_argument("only", nargs="*")
    a = ap.parse_args(argv)
    gb_common.apply_args(a)                 # env, so the spawned workers resolve the same folders
    _paths()
    from wfield_local import analysis_kit as ak
    if not sorted(V2S().glob("*.npz")):
        raise SystemExit(f"no quiet-events v2 sessions in {V2S()}: run scripts/cohort2_qc/quiet_spikes_v2.py under the "
                         f"same env with QS_LICK_SOURCE={gb_common.QUIET_LICK_SOURCE} first, or pass --quiet-dir")
    if gb_common.quiet_dir().name != gb_common.QUIET_SUBDIR:
        print(f"WARNING: quiet events from {gb_common.quiet_dir()}, not the LP version ({gb_common.QUIET_SUBDIR}; "
              "Priya 11-06)", flush=True)
    out = gb_common.guard(gb_common.out_root())
    S = gb_common.session_table()
    S.to_csv(out / "sessions.csv", index=False)
    print("out", out, "| quiet", gb_common.quiet_dir(), "| task rows", gb_common.task_rows(), flush=True)
    labs = a.only or sorted(S.label)
    print(len(labs), "sessions", flush=True)
    got, fail = ak.fan_sessions(labs, run, jobs=min(a.jobs, 3), label="gain_bio")
    for it, err in fail:
        print("FAILED", it, err, flush=True)
    print("done", len(got), "failed", len(fail), flush=True)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
