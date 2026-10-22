"""FaceRhythm factor set fitted on PRE-STROKE sessions of all four animals, frozen, and every session projected onto it
(Priya 2026-10-21: one "master pre-stroke" factor set across animals; DECISIONS.md 10-21 evening). Pipeline overview,
inputs / outputs of every stage and run order: docs/FACE_RHYTHM_PIPELINE.md.

    desktop (locanmf)  python scripts/face_rhythm_factors.py plan   --bundle fr_cam4_20261021 --set master_pre_v1
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py build                    one job: the fit tensor
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py fit --rank R             array over ranks
    desktop            python scripts/face_rhythm_factors.py report --set ... [--face img]  rank sweep + factor gallery
    desktop            python scripts/face_rhythm_factors.py select --set ... --rank R      FREEZE (refuses to refreeze)
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py project --task i         array over tasks
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py fitsession --task i      array over tasks
    then the session readouts: scripts/face_rhythm_readouts.py (wfield_local/face_rhythm_readouts.py)

Inputs (`o2_face_rhythm.py`, `<bundle>/out/<key>_{trials,session}.npz`, float16, UNCOMPRESSED -> memory-mapped here):
trials spec (trial, 151 bins at 25 Hz cue -2..+4 s, xy, point, freq), session spec (5 Hz bin, xy, point, freq). Points
are the master mesh warped to each session: point i is the same face location in every session and animal.
The O2 commands need only numpy (+ scipy for the Hungarian match); everything that needs the repo runs in `plan`.

plan       every task's epoch (`wfield_local.epochs.epoch_of`). ROWS: engaged trials balanced over positions + between-trial
           5 Hz bins (outside cue -3 .. +6 s of engaged trials, not the first / last 2 s) = ``between_frac`` x the trial
           rows. MASTER rows: per animal ``trials_per_animal`` trials drawn across its pre-stroke (August) sessions --
           EQUAL rows per animal. SESSION rows (for `fitsession`): ``session_trials`` per task, seeded per task.
build      the master fit tensor (rows, 2 x points, freqs) float32, trial windows at 12.5 Hz (pairs of 25 Hz bins; time
           is not shared across sessions, so the rate only sets the row count). Each ANIMAL is divided by ONE scalar,
           the RMS of its own fit rows (stored; applied to all its sessions, pre and post): face size differs between
           animals (mesh QC: PS93's face is smaller); one scalar removes that without flattening which points move
           most (a per-point scale would amplify the still forehead) and keeps post-stroke amplitude changes visible.
fit        non-negative CP (HALS, numpy, chunked) of rank R: ``inits`` random starts (best kept; similarity between
           starts = stability, Williams et al. 2018) and LEAVE-ONE-ANIMAL-OUT (fit on three animals, project the fourth
           with spatial + spectral fixed; held-out vs in-fit error, factor similarity to the full fit).
select     freezes rank R as ``master.npz`` (spatial (2, P, R), spectral (F, R), animal scales, spec hash of plan + rank +
           init). A frozen set is a REFERENCE (rule 10, docs/FROZEN_MODELS.md): never refitted once used.
project    per task, spatial + spectral fixed: non-negative temporal weights per bin for the trial windows (25 Hz) and
           the whole session (5 Hz), each bin's relative reconstruction error, and per-point MOVEMENT POWER (scaled
           spectrogram averaged over pre-cue / response / between-trial bins, by position) -> ``proj/<key>.npz``.
           Window means come from the 5 Hz session spectrogram: a 5 Hz bin is the mean of five 25 Hz bins, so means
           over windows aligned to 0.2 s are exact, and one pass over the session file covers every trial.
fitsession per task, the session's OWN rank-R fit on its plan rows, the frozen set's fit on the same rows, and the own
           factors matched to the frozen ones (Hungarian on spatial x spectral cosine) -> ``own/<key>.npz``: the
           explained-variance gap (own - frozen) and factor "morphing" over recovery (the 2pRAM manuscript's Fig 4
           analyses iii and the factor-angle analysis, here against one cross-animal pre-stroke basis).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

EPS = 1e-12
TRIAL_DS = 2                                        # 25 -> 12.5 Hz for the fit rows only
CHUNK = 256                                         # rows per einsum chunk
PRECUE_S, RESP_S = (-2.0, 0.0), (0.0, 4.0)          # movement-power windows (s from cue)
BUSY_S = (-3.0, 6.0)                                # "between trials" = outside this around every engaged cue
EDGE_BINS = 10                                      # 2 s at 5 Hz kept out of the between-trial pool at each end


# --------------------------------------------------------------------------------------------- npz access (no copies)
def npz_memmap(path: Path, name: str) -> np.memmap:
    """Memory-map array ``name`` of an UNCOMPRESSED npz (the runner writes them with np.savez)."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(f"{name}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{name} is compressed -- cannot memory-map")
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        h = fh.read(30)
        n_name, n_extra = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(fh)
        off = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")


def npz_small(path: Path, names: tuple) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in names if k in z.files}


def rows_of(spec: np.ndarray) -> np.ndarray:
    """(..., 2, P, F) -> (n, 2P, F) float32 (xy-major: row index xy * P + point)."""
    a = np.asarray(spec, dtype=np.float32)
    return a.reshape(-1, a.shape[-3] * a.shape[-2], a.shape[-1])


class _scaled:
    """Lazy (n, 2P, F) float32 view of a (n, 2, P, F) float16 memmap divided by the animal's scale."""

    def __init__(self, S, s):
        self.S, self.s = S, s
        self.shape = (S.shape[0], S.shape[-3] * S.shape[-2], S.shape[-1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sl):
        return rows_of(self.S[sl]) / self.s


# ------------------------------------------------------------------------------------------------- non-negative CP
def _mttkrp(X, A, B, C, mode: str) -> np.ndarray:
    """Matricised tensor times Khatri-Rao product for X (T, P, F) ~ sum_r C[:, r] o A[:, r] o B[:, r]."""
    T, R = X.shape[0], A.shape[1]
    if mode == "C":
        out = np.empty((T, R), np.float64)
        for s in range(0, T, CHUNK):
            out[s:s + CHUNK] = np.einsum("tpr,pr->tr", X[s:s + CHUNK] @ B, A)
        return out
    out = np.zeros((A.shape[0] if mode == "A" else B.shape[0], R), np.float64)
    for s in range(0, T, CHUNK):
        x, c = X[s:s + CHUNK], C[s:s + CHUNK]
        if mode == "A":
            out += np.einsum("tpr,tr->pr", x @ B, c)
        else:
            out += np.einsum("tfr,tr->fr", np.swapaxes(x, 1, 2) @ A, c)
    return out


def _hals_cols(Fm: np.ndarray, M: np.ndarray, G: np.ndarray) -> None:
    for r in range(Fm.shape[1]):
        Fm[:, r] = np.maximum(EPS, Fm[:, r] + (M[:, r] - Fm @ G[:, r]) / max(G[r, r], EPS))


def ncp_hals(X: np.ndarray, rank: int, seed: int, n_iter: int = 300, tol: float = 1e-6, log=None) -> dict:
    """Non-negative CP of X (T, P, F) by HALS. Returns A (P, R), B (F, R) with unit-norm columns, C (T, R), rel_err
    (Frobenius, uncentred: EVR = 1 - rel_err^2, the paper's definition) and the error trace."""
    X = X.astype(np.float32, copy=False)
    T, P, F = X.shape
    rng = np.random.default_rng(seed)
    nX2 = float(sum(np.einsum("tpf,tpf->", X[s:s + CHUNK], X[s:s + CHUNK], dtype=np.float64)
                    for s in range(0, T, CHUNK)))
    A, B, C = rng.random((P, rank)), rng.random((F, rank)), rng.random((T, rank))
    A /= np.linalg.norm(A, axis=0)
    B /= np.linalg.norm(B, axis=0)
    G = (A.T @ A) * (B.T @ B)
    C *= np.sqrt(nX2 / max(float(np.sum(G * (C.T @ C))), EPS))
    trace = []
    for it in range(n_iter):
        A32, B32 = A.astype(np.float32), B.astype(np.float32)
        _hals_cols(A, _mttkrp(X, A32, B32, C.astype(np.float32), "A"), (B.T @ B) * (C.T @ C))
        A32 = A.astype(np.float32)
        _hals_cols(B, _mttkrp(X, A32, B32, C.astype(np.float32), "B"), (A.T @ A) * (C.T @ C))
        na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
        A /= na
        B /= nb
        C *= na * nb
        G = (A.T @ A) * (B.T @ B)
        Mc = _mttkrp(X, A.astype(np.float32), B.astype(np.float32), None, "C")
        _hals_cols(C, Mc, G)
        err2 = nX2 - 2 * float(np.sum(Mc * C)) + float(np.sum(G * (C.T @ C)))
        trace.append(np.sqrt(max(err2, 0.0) / nX2))
        if log and (it % 25 == 0):
            log(f"    it {it}: rel err {trace[-1]:.4f}")
        if it > 10 and abs(trace[-2] - trace[-1]) < tol * trace[-1]:
            break
    return {"A": A, "B": B, "C": C, "rel_err": trace[-1], "trace": np.array(trace), "n_iter": it + 1}


def project_rows(X, A: np.ndarray, B: np.ndarray, n_iter: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temporal weights (n, R) >= 0 per row of X (n, P, F) with A, B fixed (exact per-row NNLS by coordinate descent on
    the R x R system), each row's relative error ||x - x^|| / ||x||, and ||x||^2 (for pooled EVR). NaN rows -> NaN."""
    n, R = X.shape[0], A.shape[1]
    W, E, N2 = np.full((n, R), np.nan, np.float32), np.full(n, np.nan, np.float32), np.full(n, np.nan)
    G = (A.T @ A) * (B.T @ B)
    A32, B32 = A.astype(np.float32), B.astype(np.float32)
    for s in range(0, n, CHUNK):
        x = np.asarray(X[s:s + CHUNK], np.float32)
        ok = np.isfinite(x).all(axis=(1, 2))
        if not ok.any():
            continue
        x = x[ok]
        M = np.einsum("tpr,pr->tr", x @ B32, A32).astype(np.float64)
        c = np.maximum(EPS, np.linalg.lstsq(G, M.T, rcond=None)[0].T)
        for _ in range(n_iter):
            for r in range(R):
                c[:, r] = np.maximum(0.0, c[:, r] + (M[:, r] - c @ G[:, r]) / max(G[r, r], EPS))
        nx2 = np.einsum("tpf,tpf->t", x, x, dtype=np.float64)
        e2 = nx2 - 2 * np.sum(M * c, 1) + np.einsum("tr,rs,ts->t", c, G, c)
        idx = np.arange(s, min(s + CHUNK, n))[ok]
        W[idx], N2[idx] = c, nx2
        E[idx] = np.sqrt(np.maximum(e2, 0) / np.maximum(nx2, EPS))
    return W, E, N2


def pooled_evr(E: np.ndarray, N2: np.ndarray) -> float:
    """Uncentred EVR over rows from `project_rows`' per-row errors: 1 - sum(err^2 ||x||^2) / sum(||x||^2)."""
    ok = np.isfinite(E) & np.isfinite(N2)
    return float(1 - np.sum(E[ok] ** 2 * N2[ok]) / max(np.sum(N2[ok]), EPS))


def match(ref: list, F: list) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian match of F's components to ref's on the product of per-mode cosines. -> (order: for each ref
    component the matched F column, cos (n_modes, R_ref) per mode for the matched pairs)."""
    from scipy.optimize import linear_sum_assignment
    cs = []
    for a, b in zip(ref, F):
        a = a / np.maximum(np.linalg.norm(a, axis=0), EPS)
        b = b / np.maximum(np.linalg.norm(b, axis=0), EPS)
        cs.append(a.T @ b)
    i, j = linear_sum_assignment(-np.prod(cs, axis=0))
    order = np.empty(len(i), int)
    order[i] = j
    return order, np.array([c[i, j] for c in cs])


def similarity(F1: list, F2: list) -> float:
    """Stability (Williams et al. 2018): mean over matched components of the product of per-mode cosines."""
    _, cos = match(F1, F2)
    return float(np.prod(cos, axis=0).mean())


# ----------------------------------------------------------------------------------------------- rows (plan / O2)
def session_candidates(key: str, z, n_frames: int) -> tuple[list, list]:
    """A task's engaged trials [(key, trial, position)] and between-trial 5 Hz bins [(key, bin)] (its bundle trial
    grid ``z``: position, cue_daq_s, daq_fit frame -> DAQ s)."""
    trials = [(key, i, str(p)) for i, p in enumerate(z["position"])]
    n5 = int(n_frames) // 50
    t5 = z["daq_fit"][0] * (np.arange(n5) * 50 + 24.5) + z["daq_fit"][1]
    busy = np.zeros(n5, bool)
    busy[:EDGE_BINS] = busy[-EDGE_BINS:] = True
    for c in z["cue_daq_s"]:
        busy |= (t5 >= c + BUSY_S[0]) & (t5 <= c + BUSY_S[1])
    return trials, [(key, int(b)) for b in np.nonzero(~busy)[0]]


def draw_rows(trials: list, between: list, n_trials: int, between_frac: float, rng) -> dict:
    """Positions balanced (n_trials // n_positions each, or all a position has) + between_frac x the trial rows of
    between-trial bins -> {"trials": {key: [trial, ...]}, "between": {key: [bin, ...]}} (sorted)."""
    pos = sorted({p for _, _, p in trials})
    per = n_trials // max(len(pos), 1)
    pick = []
    for p in pos:
        g = [t for t in trials if t[2] == p]
        pick += [g[i] for i in rng.choice(len(g), min(per, len(g)), replace=False)]
    nb = min(int(round(between_frac * len(pick) * (151 // TRIAL_DS))), len(between))
    bsel = [between[i] for i in rng.choice(len(between), nb, replace=False)] if nb else []
    out = {"trials": {}, "between": {}}
    for k, i, _ in pick:
        out["trials"].setdefault(k, []).append(int(i))
    for k, b in bsel:
        out["between"].setdefault(k, []).append(int(b))
    return {kind: {k: sorted(v) for k, v in sorted(d.items())} for kind, d in out.items()}


def gather_rows(od: Path, picks: dict) -> tuple[np.ndarray, list]:
    """The rows a `draw_rows` pick names, from the outputs: trial windows at 12.5 Hz + session bins. -> (X (n, 2P, F)
    float32 UNSCALED, index [(key, kind, trial, bin)]). Rows with any NaN are dropped."""
    X, index = [], []
    for key, tr in picks["trials"].items():
        S = npz_memmap(od / f"{key}_trials.npz", "spec")
        for ti in tr:
            x = rows_of(S[ti])                                           # (151, 2P, F)
            nb = (x.shape[0] // TRIAL_DS) * TRIAL_DS
            x = x[:nb].reshape(-1, TRIAL_DS, *x.shape[1:]).mean(1)
            ok = np.isfinite(x).all(axis=(1, 2))
            X.append(x[ok])
            index += [(key, "trial", int(ti), int(j)) for j in np.nonzero(ok)[0]]
    for key, bins in picks["between"].items():
        S = npz_memmap(od / f"{key}_session.npz", "spec")
        b = np.sort(np.asarray(bins, int))
        x = rows_of(S[b])
        ok = np.isfinite(x).all(axis=(1, 2))
        X.append(x[ok])
        index += [(key, "between", -1, int(v)) for v in b[ok]]
    return np.concatenate(X).astype(np.float32), index


# ---------------------------------------------------------------------------------------- paths inside the bundle
def set_dir() -> Path:
    """On O2 this file lives in <bundle>/factors/<set>/ (copied there by `plan`)."""
    return Path(__file__).resolve().parent


def out_dir(sd: Path) -> Path:
    return Path(os.environ.get("FR_OUT", sd.parent.parent / "out"))


def load_plan(sd: Path) -> dict:
    return json.loads((sd / "plan.json").read_text())


def load_master(sd: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """(A (2P, R), B (F, R), animal scales, the npz as a dict)."""
    M = dict(np.load(sd / "master.npz"))
    A = M["spatial"].reshape(-1, M["spatial"].shape[-1]).astype(np.float64)
    return A, M["spectral"].astype(np.float64), json.loads(str(M["scales_json"])), M


def _atomic_savez(dst: Path, **arrs) -> None:
    dst.parent.mkdir(exist_ok=True)
    tmp = dst.with_suffix(".part.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, dst)


# ------------------------------------------------------------------------------------------- O2: build / fit
def cmd_build(sd: Path) -> None:
    pl = load_plan(sd)
    od = out_dir(sd)
    rows, index, scale = [], [], {}
    for a, picks in pl["fit"].items():
        Xa, ix = gather_rows(od, picks)
        scale[a] = float(np.sqrt(np.mean(Xa.astype(np.float64) ** 2)))
        rows.append(Xa / scale[a])
        index += [(a, *r) for r in ix]
        print(f"  {a}: {len(Xa)} rows, scale {scale[a]:.4g}", flush=True)
    X = np.concatenate(rows).astype(np.float32)
    np.save(sd / "fit_tensor.npy", X)
    idx = np.array(index, dtype=object)
    np.savez(sd / "fit_index.npz", animal=idx[:, 0].astype(str), key=idx[:, 1].astype(str),
             kind=idx[:, 2].astype(str), trial=idx[:, 3].astype(int), bin=idx[:, 4].astype(int))
    (sd / "scales.json").write_text(json.dumps(scale, indent=1))
    print(f"fit tensor {X.shape} ({X.nbytes / 1e9:.1f} GB) -> {sd}")


def cmd_fit(sd: Path, rank: int) -> None:
    pl = load_plan(sd)
    X = np.ascontiguousarray(np.load(sd / "fit_tensor.npy", mmap_mode="r"))
    animals = np.load(sd / "fit_index.npz")["animal"]
    n_init, n_iter = pl["params"]["inits"], pl["params"]["n_iter"]
    t0 = time.time()
    fits = []
    for k in range(n_init):
        f = ncp_hals(X, rank, seed=1000 * rank + k, n_iter=n_iter)
        fits.append(f)
        print(f"rank {rank} init {k}: rel err {f['rel_err']:.4f} ({f['n_iter']} it, {(time.time() - t0) / 60:.0f} min)",
              flush=True)
    best = int(np.argmin([f["rel_err"] for f in fits]))
    b = fits[best]
    stab = [similarity([b["A"], b["B"]], [f["A"], f["B"]]) for i, f in enumerate(fits) if i != best]
    loao = {}
    for a in np.unique(animals):
        m = animals == a
        f = ncp_hals(X[~m], rank, seed=7 + rank, n_iter=n_iter)
        _, e_out, n2 = project_rows(X[m], f["A"], f["B"])
        _, e_in, _ = project_rows(X[m], b["A"], b["B"])
        loao[str(a)] = {"held_out_err": float(np.sqrt(1 - pooled_evr(e_out, n2))),
                        "in_fit_err": float(np.sqrt(1 - pooled_evr(e_in, n2))),
                        "similarity_to_full": similarity([b["A"], b["B"]], [f["A"], f["B"]])}
        print(f"  leave out {a}: held-out err {loao[str(a)]['held_out_err']:.4f} vs in-fit "
              f"{loao[str(a)]['in_fit_err']:.4f}, factor similarity {loao[str(a)]['similarity_to_full']:.3f}", flush=True)
    np.savez(sd / f"rank{rank:02d}.npz", A=b["A"], B=b["B"], C=b["C"].astype(np.float32), best_init=best,
             rel_errs=np.array([f["rel_err"] for f in fits]), trace=b["trace"])
    summ = {"rank": rank, "rel_err": float(b["rel_err"]), "evr_uncentred": float(1 - b["rel_err"] ** 2),
            "rel_errs": [float(f["rel_err"]) for f in fits], "best_init": best,
            "stability_median": float(np.median(stab)) if stab else None, "stability": stab, "loao": loao,
            "minutes": (time.time() - t0) / 60}
    (sd / f"rank{rank:02d}.json").write_text(json.dumps(summ, indent=1))
    print(f"rank {rank}: err {b['rel_err']:.4f}, EVR {summ['evr_uncentred']:.3f}, stability "
          f"{summ['stability_median']}, {summ['minutes']:.0f} min")


# ------------------------------------------------------------------------------------- O2: project / fitsession
def window_power(S, s: float, t5: np.ndarray, cues: np.ndarray, positions: np.ndarray) -> dict:
    """Per-point movement power (scaled spectrogram means) from the 5 Hz session spectrogram ``S`` (n5, 2, P, F):
    between trials (outside BUSY_S of every engaged cue), and per position pre-cue (PRECUE_S) and response (RESP_S).
    One pass; each bin goes to at most one class (the latest cue whose pre-cue window has started)."""
    o = np.argsort(cues)
    cues, positions = cues[o], positions[o]
    pos = np.unique(positions)
    pidx = np.searchsorted(pos, positions)
    j = np.searchsorted(cues, t5 - PRECUE_S[0], side="right") - 1         # latest cue with cue + PRECUE_S[0] <= t
    rel = np.where(j >= 0, t5 - cues[np.clip(j, 0, None)], np.inf)
    lab = np.full(len(t5), -1)                                           # -1 none, 0 between, 1 + 2 p + w
    near = np.zeros(len(t5), bool)
    k = np.searchsorted(cues, t5)
    for kk in (k - 1, k):
        ok = (kk >= 0) & (kk < len(cues))
        d = np.where(ok, t5 - cues[np.clip(kk, 0, len(cues) - 1)], np.inf)
        near |= ok & (d >= BUSY_S[0]) & (d <= BUSY_S[1])
    lab[~near] = 0
    lab[:EDGE_BINS] = lab[-EDGE_BINS:] = -1
    for w, (a, b) in enumerate((PRECUE_S, RESP_S)):
        m = (rel >= a) & (rel < b)
        lab[m] = 1 + 2 * pidx[np.clip(j[m], 0, None)] + w
    n_lab = 1 + 2 * len(pos)
    tot, cnt = np.zeros((n_lab, *S.shape[1:]), np.float64), np.zeros(n_lab)
    for c0 in range(0, len(t5), CHUNK):
        x = np.asarray(S[c0:c0 + CHUNK], np.float32) / s
        lb = lab[c0:c0 + CHUNK]
        ok = (lb >= 0) & np.isfinite(x).all(axis=(1, 2, 3))
        for v in np.unique(lb[ok]):
            mm = ok & (lb == v)
            tot[v] += x[mm].sum(0)
            cnt[v] += mm.sum()
    mean = (tot / np.maximum(cnt, 1)[:, None, None, None]).astype(np.float32)
    mean[cnt == 0] = np.nan
    return {"power_between": mean[0], "power_precue": mean[1::2], "power_response": mean[2::2],
            "power_positions": pos, "power_n_bins": cnt.astype(int)}


def cmd_project(sd: Path, task: int) -> None:
    t = load_plan(sd)["tasks"][task - 1]
    key, animal = t["key"], t["animal"]
    dst = sd / "proj" / f"{key}.npz"
    if dst.exists():
        print("done already", dst)
        return
    A, B, scales, M = load_master(sd)
    s = float(scales[animal])
    od = out_dir(sd)
    t0 = time.time()
    St = npz_memmap(od / f"{key}_trials.npz", "spec")
    n_tr, n_t = St.shape[:2]
    Wt, Et, _ = project_rows(_scaled(St.reshape(n_tr * n_t, *St.shape[2:]), s), A, B)
    Ss = npz_memmap(od / f"{key}_session.npz", "spec")
    Ws, Es, _ = project_rows(_scaled(Ss, s), A, B)
    meta = npz_small(od / f"{key}_trials.npz", ("t_ms", "trial_id", "position", "cue_daq_s"))
    tse = npz_small(od / f"{key}_session.npz", ("t_daq_s", "rate_hz"))
    pw = window_power(Ss, s, tse["t_daq_s"], meta["cue_daq_s"], meta["position"])
    _atomic_savez(dst, trials=Wt.reshape(n_tr, n_t, -1), trials_err=Et.reshape(n_tr, n_t), session=Ws, session_err=Es,
                  session_t_daq_s=tse["t_daq_s"], session_rate_hz=tse["rate_hz"], spec_hash=str(M["spec_hash"]),
                  animal=animal, epoch=str(t["epoch"]), date=t["date"], **meta, **pw)
    print(f"{key} [{t['epoch']}]: trials err median {np.nanmedian(Et):.3f}, session err median {np.nanmedian(Es):.3f}, "
          f"{(time.time() - t0) / 60:.1f} min -> {dst}")


def cmd_fitsession(sd: Path, task: int, inits: int = 3) -> None:
    pl = load_plan(sd)
    t = pl["tasks"][task - 1]
    key, animal = t["key"], t["animal"]
    dst = sd / "own" / f"{key}.npz"
    if dst.exists():
        print("done already", dst)
        return
    A, B, scales, M = load_master(sd)
    R = A.shape[1]
    t0 = time.time()
    X, _ = gather_rows(out_dir(sd), pl["session_rows"][key])
    X /= float(scales[animal])
    _, Ef, N2 = project_rows(X, A, B)
    evr_frozen = pooled_evr(Ef, N2)
    fits = [ncp_hals(X, R, seed=zlib.crc32(key.encode()) + k, n_iter=pl["params"]["n_iter"]) for k in range(inits)]
    f = fits[int(np.argmin([x["rel_err"] for x in fits]))]
    order, cos = match([A, B], [f["A"], f["B"]])
    _atomic_savez(dst, A_own=f["A"][:, order].astype(np.float32), B_own=f["B"][:, order].astype(np.float32),
                  evr_own=1 - f["rel_err"] ** 2, evr_frozen=evr_frozen, cos_spatial=cos[0], cos_spectral=cos[1],
                  rel_errs_own=np.array([x["rel_err"] for x in fits]), n_rows=len(X), spec_hash=str(M["spec_hash"]),
                  animal=animal, epoch=str(t["epoch"]), date=t["date"])
    print(f"{key} [{t['epoch']}]: EVR own {1 - f['rel_err'] ** 2:.3f} vs frozen {evr_frozen:.3f}; matched cosine "
          f"spatial {cos[0].mean():.3f} spectral {cos[1].mean():.3f}; {len(X)} rows, {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------------------------- desktop: plan / select
def _sbatch(b: str, opts: str, cmd: str) -> str:
    thr = opts.split(" -c ")[1].split()[0]
    parts, lines, i = opts.split(), [], 0
    while i < len(parts):
        step = 1 if parts[i].startswith("--") else 2
        lines.append(" ".join(parts[i:i + step]))
        i += step
    return ("#!/bin/bash -l\n#SBATCH -p short\n" + f"#SBATCH -o {b}/logs/%x_%A_%a.out\n#SBATCH -e {b}/logs/%x_%A_%a.err\n"
            + "".join(f"#SBATCH {x}\n" for x in lines)
            + "set -eo pipefail\nmodule load conda/miniforge3/24.11.3-0\nconda activate fr\n"
            + f"export OMP_NUM_THREADS={thr} MKL_NUM_THREADS={thr} OPENBLAS_NUM_THREADS={thr}\n"
            + f"python {b}/fr_factors.py {cmd}\n")


def cmd_plan(bundle: str, name: str, trials_per_animal: int, session_trials: int, between_frac: float, seed: int,
             ranks: list[int], inits: int, n_iter: int, user: str) -> Path:
    import pandas as pd
    from wfield_local import epochs, writeguard
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    bd = oi.bundle_dir(rv, bundle)
    sd = bd / "factors" / name
    writeguard.assert_writable(sd.parent)
    if (sd / "master.npz").exists():
        sys.exit(f"{sd} is FROZEN (master.npz exists) -- choose a new set name")
    T = pd.read_csv(bd / "tasks.tsv", sep="\t", dtype=str)
    rng = np.random.default_rng(seed)
    tasks, cand = [], {}
    for r in T.itertuples():
        tasks.append({"key": r.key, "animal": r.animal, "date": r.date, "epoch": epochs.epoch_of(f"{r.animal}_{r.date[4:]}")})
        cand[r.key] = session_candidates(r.key, np.load(bd / "trials" / f"{r.key}.npz"), int(r.n_frames))
    fit = {}
    for a in sorted({t["animal"] for t in tasks}):
        keys = [t["key"] for t in tasks if t["animal"] == a and t["epoch"] == "pre"]
        if not keys:
            continue
        picks = draw_rows(sum((cand[k][0] for k in keys), []), sum((cand[k][1] for k in keys), []),
                          trials_per_animal, between_frac, rng)
        fit[a] = {"sessions": keys, **picks}
        n_tr = sum(map(len, picks["trials"].values()))
        print(f"  {a}: {len(keys)} pre sessions, {n_tr} trials, {sum(map(len, picks['between'].values()))} between bins")
    session_rows = {t["key"]: draw_rows(*cand[t["key"]], session_trials, between_frac,
                                        np.random.default_rng([seed, zlib.crc32(t["key"].encode())])) for t in tasks}
    pl = {"set": name, "bundle": bundle, "created": time.strftime("%Y-%m-%d %H:%M"),
          "params": {"trials_per_animal": trials_per_animal, "session_trials": session_trials,
                     "between_frac": between_frac, "seed": seed, "trial_ds": TRIAL_DS, "ranks": ranks, "inits": inits,
                     "n_iter": n_iter, "precue_s": PRECUE_S, "resp_s": RESP_S, "busy_s": BUSY_S,
                     "scale": "per animal: RMS of its fit rows"},
          "fit": fit, "session_rows": session_rows, "tasks": tasks}
    (sd / "logs").mkdir(parents=True, exist_ok=True)
    (sd / "plan.json").write_text(json.dumps(pl, indent=1), encoding="utf-8", newline="\n")
    (sd / "fr_factors.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    b = f"{oi.scratch(user)}/o2/{bundle}/factors/{name}"
    n = len(tasks)
    jobs = {"run_build.sbatch": ("-J frf_build -c 4 --mem=40G -t 0-03:00:00", "build"),
            "run_fit.sbatch": (f"-J frf_fit -c 16 --mem=64G -t 0-12:00:00 --array={','.join(map(str, ranks))}",
                               "fit --rank $SLURM_ARRAY_TASK_ID"),
            "run_project.sbatch": (f"-J frf_proj -c 4 --mem=16G -t 0-02:00:00 --array=1-{n}%40",
                                   "project --task $SLURM_ARRAY_TASK_ID"),
            "run_fitsession.sbatch": (f"-J frf_own -c 8 --mem=24G -t 0-04:00:00 --array=1-{n}%40",
                                      "fitsession --task $SLURM_ARRAY_TASK_ID")}
    for fn, (opts, cmd) in jobs.items():
        (sd / fn).write_text(_sbatch(b, opts, cmd), encoding="utf-8", newline="\n")
    rows = sum(len(v) for a in fit.values() for v in a["trials"].values()) * (151 // TRIAL_DS) + \
        sum(len(v) for a in fit.values() for v in a["between"].values())
    print(f"plan -> {sd}  (master fit rows ~{rows}, tensor ~{rows * 2 * 1282 * 36 * 4 / 1e9:.1f} GB float32; {n} tasks)")
    return sd


def spec_hash(sd: Path, rank: int, best_init: int) -> str:
    pl = load_plan(sd)
    blob = json.dumps({"fit": pl["fit"], "params": {k: v for k, v in pl["params"].items() if k != "session_trials"},
                       "rank": rank, "best_init": best_init,
                       "scales": json.loads((sd / "scales.json").read_text())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cmd_select(sd: Path, rank: int) -> None:
    R = np.load(sd / f"rank{rank:02d}.npz")
    h = spec_hash(sd, rank, int(R["best_init"]))
    m = sd / "master.npz"
    if m.exists():
        old = str(np.load(m)["spec_hash"])
        sys.exit(f"{m} already frozen (spec {old}); {'same spec, nothing to do' if old == h else 'REFUSED: different spec'}")
    pl = load_plan(sd)
    first = next(iter(pl["fit"].values()))["sessions"][0]
    meta = npz_small(out_dir(sd) / f"{first}_trials.npz", ("freqs", "home"))
    P = len(R["A"]) // 2
    np.savez(m, spatial=R["A"].reshape(2, P, rank).astype(np.float32), spectral=R["B"].astype(np.float32),
             freqs=meta["freqs"], home=meta["home"], scales_json=(sd / "scales.json").read_text(), rank=rank,
             spec_hash=h, set=pl["set"], frozen=time.strftime("%Y-%m-%d %H:%M"))
    print(f"FROZEN {pl['set']} rank {rank} spec {h} -> {m}")


def cmd_report(sd: Path, face: Path | None = None) -> None:
    """Factor-set QC into ``<set>/report/``: the rank sweep (rank*.json) and, once frozen, the factor gallery (spatial
    weight on the master face, spectrum, pre-stroke trial averages by position from proj/). Session readouts (epochs,
    EVR gap, morphing, movement maps) are `scripts/face_rhythm_readouts.py`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    rd = sd / "report"
    rd.mkdir(exist_ok=True)
    S = [json.loads(f.read_text()) for f in sorted(sd.glob("rank*.json"))]
    if S:
        rk = [x["rank"] for x in S]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].plot(rk, [x["evr_uncentred"] for x in S], "o-")
        ax[0].set(xlabel="rank", ylabel="EVR (1 - |S - S^|^2 / |S|^2)", title="fit")
        ax[1].plot(rk, [x["stability_median"] or np.nan for x in S], "o-")
        ax[1].set(xlabel="rank", ylabel="similarity to best start", title="stability across starts", ylim=(0, 1.02))
        for a in sorted(S[0]["loao"]):
            ax[2].plot(rk, [x["loao"][a]["held_out_err"] / x["loao"][a]["in_fit_err"] for x in S], "o-", label=a)
        ax[2].axhline(1, color="k", lw=0.6)
        ax[2].set(xlabel="rank", ylabel="held-out / in-fit error", title="leave one animal out")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(rd / "rank_sweep.png", dpi=120)
        plt.close(fig)
        pd.DataFrame([{k: v for k, v in x.items() if k not in ("loao", "stability", "rel_errs")} for x in S]
                     ).to_csv(rd / "rank_sweep.csv", index=False, float_format="%.4f")
    if not (sd / "master.npz").exists():
        print(f"report: rank sweep only (not frozen yet) -> {rd}")
        return
    _, B, _, M = load_master(sd)
    Wsp, freqs, home, R = M["spatial"], M["freqs"], M["home"], int(M["rank"])
    mag = np.hypot(Wsp[0], Wsp[1])                                       # (P, R) weight per point
    tc, t_ms = {}, None
    for f in sorted((sd / "proj").glob("*.npz")) if (sd / "proj").exists() else []:
        z = np.load(f)
        if str(z["epoch"]) != "pre":
            continue
        t_ms = z["t_ms"]
        for pos in np.unique(z["position"]):
            tc.setdefault(str(pos), []).append(np.nanmean(z["trials"][z["position"] == pos], 0))
    img = plt.imread(face) if face and Path(face).exists() else None
    fig, axs = plt.subplots(R, 3, figsize=(11, 2.3 * R), gridspec_kw={"width_ratios": [1.1, 1, 1.4]})
    for k, (a0, a1, a2) in enumerate(np.atleast_2d(axs)):
        if img is not None:
            a0.imshow(img, cmap="gray")
        else:
            a0.invert_yaxis()
        a0.scatter(home[:, 0], home[:, 1], c=mag[:, k], s=3, cmap="magma", vmin=0)
        a0.set_aspect("equal")
        a0.axis("off")
        a0.set_title(f"factor {k}: where (image right = ipsilesional)", fontsize=8)
        a1.plot(freqs, B[:, k])
        a1.set_xscale("log")
        a1.set_title(f"peak {freqs[np.argmax(B[:, k])]:.1f} Hz", fontsize=8)
        for pos, xs in sorted(tc.items()):
            a2.plot(t_ms / 1000, np.nanmean(xs, 0)[:, k], lw=1, label=pos)
        a2.axvline(0, color="k", lw=0.5)
        a2.set_title("pre-stroke trials, by position (cue = 0 s)", fontsize=8)
        if k == 0 and tc:
            a2.legend(fontsize=6, ncol=2)
    fig.suptitle(f"{M['set']} rank {R} (spec {M['spec_hash']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(rd / "factor_gallery.png", dpi=110)
    plt.close(fig)
    print(f"report -> {rd}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "build", "fit", "select", "project", "fitsession", "report"])
    ap.add_argument("--bundle", default="fr_cam4_20261021")
    ap.add_argument("--set", default="master_pre_v1")
    ap.add_argument("--dir", type=Path, default=None, help="factor-set folder (default: this file's folder on O2, "
                                                           "or <bundle>/factors/<set> on the desktop)")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--face", type=Path, default=None, help="report: master face image under the spatial maps")
    ap.add_argument("--trials-per-animal", type=int, default=60)
    ap.add_argument("--session-trials", type=int, default=60)
    ap.add_argument("--between-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranks", type=int, nargs="*", default=[6, 8, 10, 12, 14, 16])
    ap.add_argument("--inits", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--user", default="ps150")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cmd_plan(a.bundle, a.set, a.trials_per_animal, a.session_trials, a.between_frac, a.seed, a.ranks, a.inits,
                 a.n_iter, a.user)
        return 0
    sd = a.dir
    if sd is None:
        if (set_dir() / "plan.json").exists():
            sd = set_dir()
        else:
            from wfield_local import o2_inference as oi
            from wfield_local.paths import PathResolver
            sd = oi.bundle_dir(PathResolver(), a.bundle) / "factors" / a.set
    task = a.task or (int(os.environ["SLURM_ARRAY_TASK_ID"]) if a.cmd in ("project", "fitsession") else None)
    {"build": lambda: cmd_build(sd), "fit": lambda: cmd_fit(sd, a.rank), "select": lambda: cmd_select(sd, a.rank),
     "project": lambda: cmd_project(sd, task), "fitsession": lambda: cmd_fitsession(sd, task),
     "report": lambda: cmd_report(sd, a.face)}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
