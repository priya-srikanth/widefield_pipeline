"""FaceRhythm factor set fitted on PRE-STROKE sessions of all four animals, frozen, and every session projected onto it
(Priya 2026-10-21: one "master pre-stroke" factor set across animals; DECISIONS.md 10-21 evening, FACE_RHYTHM_PLAN.md A0).

    <locanmf> python scripts/face_rhythm_factors.py plan    --bundle fr_cam4_20261021 --set master_pre_v1   (desktop)
    <fr, O2>  python <bundle>/factors/<set>/fr_factors.py build                (one job: the fit tensor)
    <fr, O2>  python <bundle>/factors/<set>/fr_factors.py fit --rank R         (array over ranks: inits + held-out animal)
    <locanmf> python scripts/face_rhythm_factors.py select --bundle ... --set ... --rank R   (freeze; refuses to refreeze)
    <fr, O2>  python <bundle>/factors/<set>/fr_factors.py project              (array over tasks)
    (figures -- factor gallery, residual by epoch -- next: `report`, not written yet)

Data (`o2_face_rhythm.py` outputs, `<bundle>/out/<key>_{trials,session}.npz`, float16, uncompressed -> memory-mapped
here): trials spec (trial, 151 bins at 25 Hz cue -2..+4 s, xy, point, freq), session spec (5 Hz bin, xy, point, freq).
Points are the master mesh warped to each session, so point i is the same face location everywhere.

plan     every task's epoch (`wfield_local.epochs.epoch_of`); FIT SET = pre-stroke sessions (August; June is not in the
         bundle) of each animal: ``trials_per_animal`` engaged trials, positions balanced, drawn across sessions, plus
         ``between_frac`` x as many 5 Hz bins from between trials (outside cue -3 .. +6 s of engaged trials) so
         non-task movement (grooming, rest) is in the repertoire. EQUAL rows per animal. Seeded. Writes plan.json,
         this script as fr_factors.py and the sbatch files into ``<bundle>/factors/<set>/``.
build    the fit tensor X (rows, 2 x points, freqs), float32; trial windows averaged to 12.5 Hz (pairs of 25 Hz bins:
         time is not shared across sessions, so the temporal rate only sets how many rows). Each ANIMAL is divided by
         one scalar, the RMS of its own fit rows (stored; applied to every session of that animal, pre and post):
         face size differs between animals (mesh QC: PS93's face is smaller), and one scalar per animal removes that
         without flattening which points move most (a per-point scale would amplify the still forehead) and keeps
         post-stroke amplitude changes visible.
fit      non-negative CP (HALS, numpy, chunked) of rank R: ``inits`` random starts (best kept; factor similarity
         between starts = stability, Williams et al. 2018), and LEAVE-ONE-ANIMAL-OUT: fit on three animals, project the
         fourth with the spatial + spectral factors fixed, its error vs the in-fit error. Rank chosen from the sweep
         (`select`): error flattening, stability, held-out animal, interpretability (the paper's criteria + one).
select   freezes rank R as ``master.npz`` (spatial (2, P, R), spectral (F, R), animal scales, spec hash of plan +
         rank + init). A frozen set is a REFERENCE (rule 10, docs/FROZEN_MODELS.md): never refitted once used; a
         different spec under the same set name is refused.
project  per task, spatial + spectral fixed, non-negative temporal weights per bin (NNLS via HALS on the small R x R
         system) for the trial windows (25 Hz) and the whole session (5 Hz), plus each bin's relative reconstruction
         error: post-stroke movement outside the pre-stroke repertoire shows up there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

EPS = 1e-12
TRIAL_DS = 2                                        # 25 -> 12.5 Hz for the fit rows only
CHUNK = 256                                         # rows per einsum chunk


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


def ncp_hals(X: np.ndarray, rank: int, seed: int, n_iter: int = 300, tol: float = 1e-6,
             fixed: tuple | None = None, log=None) -> dict:
    """Non-negative CP of X (T, P, F) by HALS. ``fixed=(A, B)`` keeps spatial + spectral fixed (projection fit).
    Returns A (P, R), B (F, R) unit-norm columns, C (T, R), rel_err (Frobenius, uncentred) and the error trace."""
    X = X.astype(np.float32, copy=False)
    T, P, F = X.shape
    rng = np.random.default_rng(seed)
    nX2 = float(sum(np.einsum("tpf,tpf->", X[s:s + CHUNK], X[s:s + CHUNK], dtype=np.float64)
                    for s in range(0, T, CHUNK)))
    if fixed is None:
        A, B = rng.random((P, rank)), rng.random((F, rank))
    else:
        A, B = (np.array(f, np.float64) for f in fixed)
    A /= np.linalg.norm(A, axis=0)
    B /= np.linalg.norm(B, axis=0)
    C = rng.random((T, rank))
    G = (A.T @ A) * (B.T @ B)
    C *= np.sqrt(nX2 / max(float(np.sum(G * (C.T @ C))), EPS))
    trace = []
    for it in range(n_iter):
        if fixed is None:
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


def project_rows(X: np.ndarray, A: np.ndarray, B: np.ndarray, n_iter: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Temporal weights (n, R) >= 0 per row of X (n, P, F) with A, B fixed, and each row's relative error
    ||x - x^|| / ||x||. NaN rows -> NaN. Rows are independent (exact per-bin NNLS by coordinate descent)."""
    n, R = X.shape[0], A.shape[1]
    W, E = np.full((n, R), np.nan, np.float32), np.full(n, np.nan, np.float32)
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
        W[idx] = c
        E[idx] = np.sqrt(np.maximum(e2, 0) / np.maximum(nx2, EPS))
    return W, E


def similarity(F1: list, F2: list) -> float:
    """Williams-style stability: Hungarian-matched mean over components of the product of per-mode cosines."""
    from scipy.optimize import linear_sum_assignment
    S = np.ones((F1[0].shape[1], F2[0].shape[1]))
    for a, b in zip(F1, F2):
        a = a / np.maximum(np.linalg.norm(a, axis=0), EPS)
        b = b / np.maximum(np.linalg.norm(b, axis=0), EPS)
        S *= a.T @ b
    i, j = linear_sum_assignment(-S)
    return float(S[i, j].mean())


# ---------------------------------------------------------------------------------------- paths inside the bundle
def set_dir() -> Path:
    """On O2 this file lives in <bundle>/factors/<set>/ (copied by `plan`)."""
    return Path(__file__).resolve().parent


def out_dir(sd: Path) -> Path:
    return Path(os.environ.get("FR_OUT", sd.parent.parent / "out"))


def load_plan(sd: Path) -> dict:
    return json.loads((sd / "plan.json").read_text())


# ------------------------------------------------------------------------------------------------ build / fit (O2)
def cmd_build(sd: Path) -> None:
    pl = load_plan(sd)
    od = out_dir(sd)
    rows, index, scale = [], [], {}
    for a, picks in pl["fit"].items():
        Xa = []
        for key, tr in picks["trials"].items():
            S = npz_memmap(od / f"{key}_trials.npz", "spec")
            for ti in tr:
                x = rows_of(S[ti])                                   # (151, 2P, F)
                nb = (x.shape[0] // TRIAL_DS) * TRIAL_DS
                x = x[:nb].reshape(-1, TRIAL_DS, *x.shape[1:]).mean(1)
                ok = np.isfinite(x).all(axis=(1, 2))
                Xa.append(x[ok])
                index += [(a, key, "trial", int(ti), int(j)) for j in np.nonzero(ok)[0]]
        for key, bins in picks["between"].items():
            S = npz_memmap(od / f"{key}_session.npz", "spec")
            x = rows_of(S[np.sort(np.asarray(bins, int))])
            ok = np.isfinite(x).all(axis=(1, 2))
            Xa.append(x[ok])
            index += [(a, key, "between", -1, int(b)) for b in np.sort(bins)[ok]]
        Xa = np.concatenate(Xa)
        scale[a] = float(np.sqrt(np.mean(Xa.astype(np.float64) ** 2)))
        rows.append(Xa / scale[a])
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
    X = np.load(sd / "fit_tensor.npy", mmap_mode="r")
    X = np.ascontiguousarray(X)
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
        _, e_out = project_rows(X[m], f["A"], f["B"])
        _, e_in = project_rows(X[m], b["A"], b["B"])
        loao[str(a)] = {"held_out_err": float(np.sqrt(np.nanmean(e_out ** 2))),
                        "in_fit_err": float(np.sqrt(np.nanmean(e_in ** 2))),
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


def cmd_project(sd: Path, task: int) -> None:
    pl = load_plan(sd)
    t = pl["tasks"][task - 1]
    key, animal = t["key"], t["animal"]
    dst = sd / "proj" / f"{key}.npz"
    if dst.exists():
        print("done already", dst)
        return
    M = np.load(sd / "master.npz")
    A, B = M["spatial"].reshape(-1, M["spatial"].shape[-1]).astype(np.float64), M["spectral"].astype(np.float64)
    s = float(json.loads(str(M["scales_json"]))[animal])
    od = out_dir(sd)
    t0 = time.time()
    St = npz_memmap(od / f"{key}_trials.npz", "spec")
    n_tr, n_t = St.shape[:2]
    Wt, Et = project_rows(_scaled(St.reshape(n_tr * n_t, *St.shape[2:]), s), A, B)
    Ss = npz_memmap(od / f"{key}_session.npz", "spec")
    Ws, Es = project_rows(_scaled(Ss, s), A, B)
    meta = npz_small(od / f"{key}_trials.npz", ("t_ms", "trial_id", "position", "cue_daq_s"))
    tse = npz_small(od / f"{key}_session.npz", ("t_daq_s", "rate_hz"))
    (sd / "proj").mkdir(exist_ok=True)
    tmp = dst.with_suffix(".part.npz")
    np.savez(tmp, trials=Wt.reshape(n_tr, n_t, -1), trials_err=Et.reshape(n_tr, n_t), session=Ws, session_err=Es,
             session_t_daq_s=tse["t_daq_s"], session_rate_hz=tse["rate_hz"], spec_hash=str(M["spec_hash"]),
             animal=animal, epoch=str(t["epoch"]), **meta)
    os.replace(tmp, dst)
    print(f"{key} [{t['epoch']}]: trials err median {np.nanmedian(Et):.3f}, session err median {np.nanmedian(Es):.3f}, "
          f"{(time.time() - t0) / 60:.1f} min -> {dst}")


class _scaled:
    """Lazy (n, 2P, F) float32 view of a (n, 2, P, F) float16 memmap divided by the animal's scale."""

    def __init__(self, S, s):
        self.S, self.s = S, s
        self.shape = (S.shape[0], S.shape[-3] * S.shape[-2], S.shape[-1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sl):
        return rows_of(self.S[sl]) / self.s


# ------------------------------------------------------------------------------------------- plan / select (desktop)
def cmd_plan(bundle: str, name: str, trials_per_animal: int, between_frac: float, seed: int, ranks: list[int],
             inits: int, n_iter: int, user: str) -> Path:
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
    tasks, by_animal = [], {}
    for r in T.itertuples():
        ep = epochs.epoch_of(f"{r.animal}_{r.date[4:]}")
        tasks.append({"key": r.key, "animal": r.animal, "date": r.date, "epoch": ep})
        if ep == "pre":
            by_animal.setdefault(r.animal, []).append(r)
    fit = {}
    for a, rs in sorted(by_animal.items()):
        pool, between = [], []
        for r in rs:
            z = np.load(bd / "trials" / f"{r.key}.npz", allow_pickle=False)
            pool += [(r.key, i, p) for i, p in enumerate(z["position"])]
            n5 = int(r.n_frames) // 50
            t5 = z["daq_fit"][0] * (np.arange(n5) * 50 + 24.5) + z["daq_fit"][1]
            busy = np.zeros(n5, bool)
            busy[:10] = busy[-10:] = True                                   # not the first / last 2 s
            for c in z["cue_daq_s"]:
                busy |= (t5 >= c - 3.0) & (t5 <= c + 6.0)
            between += [(r.key, int(b)) for b in np.nonzero(~busy)[0]]
        P = pd.DataFrame(pool, columns=["key", "trial", "position"])
        per = trials_per_animal // P.position.nunique()
        pick = pd.concat([g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 31)))
                          for _, g in P.groupby("position")])
        n_between = int(round(between_frac * len(pick) * (151 // TRIAL_DS)))
        bsel = [between[i] for i in rng.choice(len(between), min(n_between, len(between)), replace=False)]
        fit[a] = {"sessions": sorted({r.key for r in rs}),
                  "trials": {k: sorted(int(x) for x in g.trial) for k, g in pick.groupby("key")},
                  "between": {k: sorted(b for kk, b in bsel if kk == k) for k in sorted({kk for kk, _ in bsel})}}
        print(f"  {a}: {len(rs)} pre sessions, {len(pick)} trials "
              f"({pick.position.value_counts().sort_index().to_dict()}), {len(bsel)} between-trial bins")
    pl = {"set": name, "bundle": bundle, "created": time.strftime("%Y-%m-%d %H:%M"),
          "params": {"trials_per_animal": trials_per_animal, "between_frac": between_frac, "seed": seed,
                     "trial_ds": TRIAL_DS, "ranks": ranks, "inits": inits, "n_iter": n_iter,
                     "scale": "per animal: RMS of its fit rows"},
          "fit": fit, "tasks": tasks}
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "logs").mkdir(exist_ok=True)
    (sd / "plan.json").write_text(json.dumps(pl, indent=1), encoding="utf-8", newline="\n")
    (sd / "fr_factors.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    b = f"{oi.scratch(user)}/o2/{bundle}/factors/{name}"
    head = "#!/bin/bash -l\n#SBATCH -p short\n#SBATCH -o {b}/logs/%x_%A_%a.out\n#SBATCH -e {b}/logs/%x_%A_%a.err\n"
    env = "set -eo pipefail\nmodule load conda/miniforge3/24.11.3-0\nconda activate fr\n"
    jobs = {"run_build.sbatch": ("-J frf_build -c 4 --mem=40G -t 0-03:00:00", "build"),
            "run_fit.sbatch": (f"-J frf_fit -c 16 --mem=64G -t 0-12:00:00 --array={','.join(map(str, ranks))}",
                               "fit --rank $SLURM_ARRAY_TASK_ID"),
            "run_project.sbatch": (f"-J frf_proj -c 4 --mem=16G -t 0-02:00:00 --array=1-{len(tasks)}%40",
                                   "project --task $SLURM_ARRAY_TASK_ID")}
    for fn, (opts, cmd) in jobs.items():
        thr = opts.split(" -c ")[1].split()[0]
        txt = head.format(b=b) + "".join(f"#SBATCH {o}\n" for o in _sbatch_opts(opts)) + env + \
            f"export OMP_NUM_THREADS={thr} MKL_NUM_THREADS={thr} OPENBLAS_NUM_THREADS={thr}\n" + \
            f"python {b}/fr_factors.py {cmd}\n"
        (sd / fn).write_text(txt, encoding="utf-8", newline="\n")
    rows = sum(len(v) for a in fit.values() for v in a["trials"].values()) * (151 // TRIAL_DS) + \
        sum(len(v) for a in fit.values() for v in a["between"].values())
    print(f"plan -> {sd}  (fit rows ~{rows}, tensor ~{rows * 2 * 1282 * 36 * 4 / 1e9:.1f} GB float32; "
          f"{len(tasks)} tasks to project)")
    print(f"O2: sbatch {b}/run_build.sbatch; then sbatch --dependency=afterok:<build> {b}/run_fit.sbatch")
    return sd


def _sbatch_opts(s: str) -> list[str]:
    parts, out = s.split(), []
    i = 0
    while i < len(parts):
        if parts[i].startswith("--"):
            out.append(parts[i])
            i += 1
        else:
            out.append(f"{parts[i]} {parts[i + 1]}")
            i += 2
    return out


def spec_hash(sd: Path, rank: int, best_init: int) -> str:
    pl = load_plan(sd)
    blob = json.dumps({"fit": pl["fit"], "params": pl["params"], "rank": rank, "best_init": best_init,
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
    P = len(R["A"]) // 2
    scales = json.loads((sd / "scales.json").read_text())
    with np.load(next(iter(_outs(sd, pl)))) as z:
        freqs = z["freqs"]
        home = z["home"]
    np.savez(m, spatial=R["A"].reshape(2, P, rank).astype(np.float32), spectral=R["B"].astype(np.float32),
             freqs=freqs, home=home, scales_json=json.dumps(scales), rank=rank, spec_hash=h,
             set=pl["set"], frozen=time.strftime("%Y-%m-%d %H:%M"))
    print(f"FROZEN {pl['set']} rank {rank} spec {h} -> {m}")


def _outs(sd: Path, pl: dict):
    od = out_dir(sd)
    for a in pl["fit"].values():
        for k in a["sessions"]:
            yield od / f"{k}_trials.npz"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "build", "fit", "select", "project"])
    ap.add_argument("--bundle", default="fr_cam4_20261021")
    ap.add_argument("--set", default="master_pre_v1")
    ap.add_argument("--dir", type=Path, default=None, help="factor-set folder (default: this file's folder on O2, "
                                                           "or <bundle>/factors/<set> on the desktop)")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--trials-per-animal", type=int, default=60)
    ap.add_argument("--between-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranks", type=int, nargs="*", default=[6, 8, 10, 12, 14, 16])
    ap.add_argument("--inits", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--user", default="ps150")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cmd_plan(a.bundle, a.set, a.trials_per_animal, a.between_frac, a.seed, a.ranks, a.inits, a.n_iter, a.user)
        return 0
    sd = a.dir
    if sd is None:
        if (set_dir() / "plan.json").exists():
            sd = set_dir()
        else:
            from wfield_local import o2_inference as oi
            from wfield_local.paths import PathResolver
            sd = oi.bundle_dir(PathResolver(), a.bundle) / "factors" / a.set
    if a.cmd == "build":
        cmd_build(sd)
    elif a.cmd == "fit":
        cmd_fit(sd, a.rank)
    elif a.cmd == "select":
        cmd_select(sd, a.rank)
    else:
        cmd_project(sd, a.task or int(os.environ["SLURM_ARRAY_TASK_ID"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
