"""O2 stage `project`: every session on the frozen set (temporal weights, normaliser D, per-window spectral share and point speed, point PCs) -> proj/<key>.npz."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import BUSY_S, CHUNK, EDGE_BINS, POINTS_CHUNK, PRECUE_S, RESP_S
from .cp import _scaled, normaliser, project_rows
from .files import _atomic_savez, _sidecar_module, load_master, load_plan, npz_memmap, npz_small, out_dir, plan_views
from .sample import session_group, spec_of


def task_basis(sd: Path, t: dict, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """(A, B, basis, matched cosines) a task is SCORED on (Priya 10-22): post-stroke sessions on the full pre-stroke fit;
    a PRE-STROKE session on the fit that LEFT ITS DAY OUT (``loso/rank<R>_<day>.npz``, components in the full fit's
    order), so pre-stroke numbers are never in-sample and pre vs post is a fair comparison (as the frozen decoders'
    held-out pre evaluation). A pre task without its left-out basis is refused when the plan has leave-one-out jobs;
    a plan without them (``--no-loso``) scores pre in-sample and says so (basis "full_in_sample")."""
    R = A.shape[1]
    if t["epoch"] != "pre":
        return A, B, "full", np.ones((2, R))
    f = sd / "loso" / f"rank{R:02d}_{session_group(t['key'])}.npz"
    if f.exists():
        z = np.load(f)
        return z["A"].astype(np.float64), z["B"].astype(np.float64), "loso", z["cos"]
    if load_plan(sd).get("loso_jobs"):
        sys.exit(f"{t['key']} is pre-stroke and {f.name} is missing: run the loso array for rank {R} first")
    return A, B, "full_in_sample", np.ones((2, R))


# ------------------------------------------------------------------------------------- O2: project / fitsession
def window_labels(t: np.ndarray, cues: np.ndarray, positions: np.ndarray, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """Class of each time bin: -1 none, 0 between trials (outside BUSY_S of every engaged cue), 1 + 2 p + w for position
    p's pre-cue (w = 0, PRECUE_S) / response (w = 1, RESP_S) window of the latest cue whose pre-cue window has started.
    ``edge`` bins at each end are never between-trial. -> (labels, sorted unique positions)."""
    o = np.argsort(cues)
    cues, positions = cues[o], positions[o]
    pos = np.unique(positions)
    pidx = np.searchsorted(pos, positions)
    j = np.searchsorted(cues, t - PRECUE_S[0], side="right") - 1
    rel = np.where(j >= 0, t - cues[np.clip(j, 0, None)], np.inf)
    lab = np.full(len(t), -1)
    near = np.zeros(len(t), bool)
    k = np.searchsorted(cues, t)
    for kk in (k - 1, k):
        ok = (kk >= 0) & (kk < len(cues))
        d = np.where(ok, t - cues[np.clip(kk, 0, len(cues) - 1)], np.inf)
        near |= ok & (d >= BUSY_S[0]) & (d <= BUSY_S[1])
    lab[~near] = 0
    lab[:edge] = lab[len(t) - edge:] = -1
    for w, (a, b) in enumerate((PRECUE_S, RESP_S)):
        m = (rel >= a) & (rel < b)
        lab[m] = 1 + 2 * pidx[np.clip(j[m], 0, None)] + w
    return lab, pos


class ClassMeans:
    """Running mean of rows per label ``lab`` (rows with label < 0 or any NaN skipped), fed in chunks by ``add(start,
    rows)`` in any order -- so it can ride on another pass over the same file."""

    def __init__(self, lab: np.ndarray, n_lab: int, shape: tuple):
        self.lab, self.shape = lab, tuple(shape)
        self.tot, self.cnt = np.zeros((n_lab, *shape), np.float64), np.zeros(n_lab)

    def add(self, c0: int, x: np.ndarray) -> None:
        lb = self.lab[c0:c0 + len(x)]
        ok = (lb >= 0) & np.isfinite(x.reshape(len(x), -1)).all(1)
        for v in np.unique(lb[ok]):
            mm = ok & (lb == v)
            self.tot[v] += x[mm].sum(0)
            self.cnt[v] += mm.sum()

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        mean = (self.tot / np.maximum(self.cnt, 1).reshape(-1, *([1] * len(self.shape)))).astype(np.float32)
        mean[self.cnt == 0] = np.nan
        return mean, self.cnt.astype(int)


def _class_means(get, n: int, lab: np.ndarray, n_lab: int, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Mean of rows ``get(slice)`` per label (chunked; rows with any NaN skipped). -> (means (n_lab, *shape), counts)."""
    acc = ClassMeans(lab, n_lab, shape)
    for c0 in range(0, n, CHUNK):
        acc.add(c0, get(slice(c0, c0 + CHUNK)))
    return acc.result()


def _power_dict(mean: np.ndarray, cnt: np.ndarray, pos: np.ndarray) -> dict:
    return {"power_between": mean[0], "power_precue": mean[1::2], "power_response": mean[2::2],
            "power_positions": pos, "power_n_bins": cnt}


def window_power(S, s: float, t5: np.ndarray, cues: np.ndarray, positions: np.ndarray) -> dict:
    """Per-point SPECTRAL SHARE by window from the 5 Hz session spectrogram ``S`` (n5, 2, P, F), scaled. FR normalises
    every frame by its mean power over the whole face (normalization_factor 0.95), so this is how a frame's movement is
    DISTRIBUTED over points and frequencies, not how much the face moves (that is `window_speed`)."""
    lab, pos = window_labels(t5, cues, positions, EDGE_BINS)
    mean, cnt = _class_means(lambda sl: np.asarray(S[sl], np.float32) / s, len(t5), lab, 1 + 2 * len(pos), S.shape[1:])
    return _power_dict(mean, cnt, pos)


def window_speed(Xp: np.ndarray, t25: np.ndarray, cues: np.ndarray, positions: np.ndarray) -> dict:
    """Per-point movement SPEED (px / s, |d displacement / dt| of the 25 Hz points, RMS within each window class) --
    the ABSOLUTE amount of movement the normalised spectrogram cannot show. ``Xp`` (bins, 2P) point-major."""
    P = Xp.shape[1] // 2
    v = np.diff(Xp.reshape(len(Xp), P, 2), axis=0) * (1.0 / np.median(np.diff(t25)))
    sp2 = np.vstack([np.full((1, P), np.nan), (v ** 2).sum(-1)])                # speed^2 per bin, first bin NaN
    lab, pos = window_labels(t25, cues, positions, 5 * EDGE_BINS)
    mean, cnt = _class_means(lambda sl: sp2[sl], len(t25), lab, 1 + 2 * len(pos), (P,))
    rms = np.sqrt(mean)
    return {"speed_between": rms[0], "speed_precue": rms[1::2], "speed_response": rms[2::2], "speed_n_bins": cnt}


def points_stream(pf: Path, comps: np.ndarray, cues: np.ndarray, positions: np.ndarray,
                  t25: np.ndarray | None = None, bad: np.ndarray | None = None) -> dict:
    """`_centred_disp` + PCs + `window_speed` for a whole task WITHOUT holding its points: two sequential passes over
    ``disp`` in ``POINTS_CHUNK`` blocks (the session mean; then per block the centred rows on the frozen PCs and the
    speed^2 per bin, the block's first bin differenced against the previous block's last). Same numbers as the
    in-memory path (test); peak memory one block instead of ~5 GB at 230k bins x 1282 points."""
    D = npz_memmap(pf, "disp")
    n, P = D.shape[0], D.shape[1]
    if t25 is None:                                  # else the TRUE times (`face_rhythm_timing`); ``bad`` -> NaN
        t25 = npz_small(pf, ("t_daq_s",))["t_daq_s"]
    tot, cnt = np.zeros(2 * P), 0
    for c0 in range(0, n, POINTS_CHUNK):
        x = np.asarray(D[c0:c0 + POINTS_CHUNK], np.float32).reshape(-1, 2 * P)
        ok = np.isfinite(x).all(1)
        tot += x[ok].sum(0)
        cnt += ok.sum()
    mu = (tot / max(cnt, 1)).astype(np.float32)
    lab, pos = window_labels(t25, cues, positions, 5 * EDGE_BINS)
    if bad is not None:
        lab[bad[:len(lab)]] = -1
    acc = ClassMeans(lab, 1 + 2 * len(pos), (P,))
    pcs = np.empty((n, comps.shape[0]), np.float32)
    inv_dt, prev = 1.0 / np.median(np.diff(t25)), None
    for c0 in range(0, n, POINTS_CHUNK):
        x = np.asarray(D[c0:c0 + POINTS_CHUNK], np.float32).reshape(-1, 2 * P) - mu
        pcs[c0:c0 + len(x)] = x @ comps.T                                     # NaN bins stay NaN
        xx = x.reshape(len(x), P, 2)
        full = xx if prev is None else np.concatenate([prev[None], xx])
        sp2 = ((np.diff(full, axis=0) * inv_dt) ** 2).sum(-1)
        if prev is None:
            sp2 = np.vstack([np.full((1, P), np.nan, np.float32), sp2])        # first bin of the session: NaN
        acc.add(c0, sp2)
        prev = xx[-1]
    mean, cnts = acc.result()
    rms = np.sqrt(mean)
    if bad is not None:
        pcs[bad[:len(pcs)]] = np.nan
    return {"points_pc": pcs, "points_t_daq_s": t25, "speed_between": rms[0], "speed_precue": rms[1::2],
            "speed_response": rms[2::2], "speed_n_bins": cnts}


def cmd_project(sd: Path, task: int, loso: bool = False) -> None:
    """Project one task on the frozen basis -> ``proj/<key>.npz``. Since 10-23 (Priya) EVERY session, pre included, is
    projected on the ONE full frozen basis: the same regressor definitions for the encoder and every expression
    analysis (held-out bases differ slightly per pre day). ``loso``: a PRE task on the basis that left its day out ->
    ``proj_loso/<key>.npz``, used only where in-sample would flatter pre-stroke -- FaceRhythm's own scoring
    (reconstruction error, EVR gap / noise floor; post tasks are not written there). Pilot sets made before 10-23 hold
    pre tasks on their held-out basis in ``proj/`` (``basis`` field says which)."""
    t = load_plan(sd)["tasks"][task - 1]
    key, animal = t["key"], t["animal"]
    if loso and t["epoch"] != "pre":
        print(f"{key}: post-stroke -- proj/ already is its scoring basis (full); nothing for proj_loso/")
        return
    dst = sd / ("proj_loso" if loso else "proj") / f"{key}.npz"
    if dst.exists():
        print("done already", dst)
        return
    A, B, scales, M = load_master(sd)
    if loso:
        A, B, basis, bcos = task_basis(sd, t, A, B)
    else:
        basis = "full_in_sample" if t["epoch"] == "pre" else "full"
        bcos = np.ones((2, A.shape[1]))
    s = float(scales[animal])
    od = out_dir(sd)
    t0 = time.time()
    St = spec_of(sd, key, "trials")
    n_tr, n_t = St.shape[:2]
    pr = load_plan(sd)["params"]
    fr_norm, fr_ex = float(pr.get("vqt_norm", 0.95)), float(pr.get("one_over_f", 0.5))
    freqs = M["freqs"].astype(np.float64)
    dfn = lambda x: normaliser(x * s, freqs, fr_norm, fr_ex)                   # rows are scaled by 1 / s
    if plan_views(sd):                                  # multi-view: D = mean of the views' own D (provisional)
        fmv = _sidecar_module("face_rhythm_multiview")
        divs = json.loads((sd / "view_scales.json").read_text())[animal]
        dfn = lambda x: fmv.view_normaliser(x, St.P, divs, s, normaliser, freqs, fr_norm, fr_ex)
    Wt, Et, _, Dt = project_rows(_scaled(St.reshape(n_tr * n_t, *St.shape[2:]), s), A, B, row_fn=dfn)
    meta = npz_small(od / f"{key}_trials.npz", ("t_ms", "trial_id", "position", "cue_daq_s"))
    tse = npz_small(od / f"{key}_session.npz", ("t_daq_s", "rate_hz"))
    frt = _sidecar_module("face_rhythm_timing")
    tm = frt.load(od.parent, key)                      # TRUE bin times + lossy bins (10-23); linear stamps if absent
    if tm is not None:
        tse["t_daq_s"] = tm["t5"][:len(tse["t_daq_s"])]
    bad5 = frt.lossy(tm["drop5"][:len(tse["t_daq_s"])]) if tm is not None else np.zeros(len(tse["t_daq_s"]), bool)
    Ss = spec_of(sd, key, "session")
    # ONE pass over the session spectrogram (6-9 GB on the share): the window power means (`window_power`) ride on the
    # projection's chunked read instead of a second read (10-22)
    lab, pos = window_labels(tse["t_daq_s"], meta["cue_daq_s"], meta["position"], EDGE_BINS)
    lab[bad5] = -1
    acc = ClassMeans(lab, 1 + 2 * len(pos), Ss.shape[1:])
    Ws, Es, _, Ds = project_rows(_scaled(Ss, s, on_chunk=lambda c0, x: acc.add(c0, x.reshape(len(x), *Ss.shape[1:]))),
                                 A, B, row_fn=dfn)
    pw = _power_dict(*acc.result(), pos)
    if "points_components" in M:
        pw.update(points_stream(od / f"{key}_points.npz", M["points_components"], meta["cue_daq_s"], meta["position"],
                                t25=tm["t25"] if tm is not None else None,
                                bad=frt.lossy(tm["drop25"]) if tm is not None else None))
    Ws[bad5], Es[bad5] = np.nan, np.nan
    bad_tr = frt.lossy(tm["drop_trials"]).ravel() if tm is not None else np.zeros(len(Wt), bool)
    Wt[bad_tr], Et[bad_tr] = np.nan, np.nan
    _atomic_savez(dst, trials=Wt.reshape(n_tr, n_t, -1), trials_err=Et.reshape(n_tr, n_t), session=Ws, session_err=Es,
                  trials_D=Dt.reshape(n_tr, n_t), session_D=Ds,
                  session_t_daq_s=tse["t_daq_s"], session_rate_hz=tse["rate_hz"], spec_hash=str(M["spec_hash"]),
                  timing=frt.TIMING_VERSION if tm is not None else "linear", session_lossy=bad5,
                  trials_lossy=bad_tr.reshape(n_tr, n_t),
                  animal=animal, epoch=str(t["epoch"]), date=t["date"], basis=basis, basis_cos=bcos, **meta, **pw)
    print(f"{key} [{t['epoch']}, basis {basis}]: trials err median {np.nanmedian(Et):.3f}, session err median {np.nanmedian(Es):.3f}, "
          f"{(time.time() - t0) / 60:.1f} min -> {dst}")
