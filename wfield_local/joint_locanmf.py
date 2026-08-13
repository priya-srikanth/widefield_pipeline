"""A joint LocaNMF basis that is BUILT ONCE, SEEDED, and SAVED — not refit on every run.

``joint_basis.build_joint_basis`` gives one shared SVD basis across an animal's sessions; this module
runs LocaNMF on it once and persists the result, so every downstream figure uses the same components.

WHY PERSIST (both reasons measured, not assumed):

  1. LocaNMF is STOCHASTIC. Two runs of identical code over identical sessions returned different
     component counts (PS92 123 vs 128, PS93 93 vs 96, PS94 95 vs 98, PS95 137 vs 135) and moved the
     cross-session RSA by up to 0.054 — larger than the gap between the bases we were choosing among.
     Randomness enters via ``randomized_svd`` (numpy global RNG) and ``torch.randperm`` in LocaNMF's
     initialisation, so seeding both makes a rebuild reproducible.
  2. A basis refit over "all curated sessions" is a DIFFERENT BASIS every time the session set grows.
     Refitting nightly would silently make last week's numbers incomparable with this week's. The basis
     must be a fixed reference frame, which is also what the post-stroke comparison requires: the
     reference has to be fixed BEFORE the manipulation.

REFITTING IS EXPECTED, and is a versioned event rather than an overwrite. ``basis_id`` hashes the
session set, their input signatures, the rank, the LocaNMF params and the seed, so a refit over a
different session set lands in a NEW directory and cannot be confused with the old one. Results carry
the id (``meta['basis_id']``), so a figure can never silently mix two bases. Nothing is deleted.

    build(animal, labels)        -> fit + save, returns a Basis
    load(animal, basis_id=None)  -> newest saved basis for the animal (or a specific one)
    Basis.signal(label)          -> (ncomp, T) footprint-scaled components for one session

Storage: ``WIDEFIELD_JOINT_BASIS_DIR`` (default ``C:/wf_local/joint_bases``). Derived data — safe to
delete, costs only a rebuild.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from wfield_local import config, joint_basis
from wfield_local.locanmf_crossanimal_dff import _footprint_scale

BASIS_DIR = Path(os.environ.get("WIDEFIELD_JOINT_BASIS_DIR", "C:/wf_local/joint_bases"))
SEED = 0
FORMAT_VERSION = 1          # bump if the on-disk layout or the fitting logic changes


def _stat_sig(path) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "MISSING"


def _session_sig(s) -> str:
    """Signature of the INPUTS the basis is built from, so an upstream re-preprocess invalidates it."""
    ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
    u_atlas = f"{ad[0]}/U_atlas.npy" if ad else ""
    svt = f"{s['mc']}/wfield_local_results/SVTcorr.npy"
    return f"{_stat_sig(u_atlas)}|{_stat_sig(svt)}"


def basis_id(sessions, rank, seed=SEED) -> str:
    """Stable id over everything that determines the fitted basis."""
    lp = config.defaults()["locanmf"]
    parts = [f"fmt{FORMAT_VERSION}", f"rank={rank}", f"seed={seed}",
             "lp=" + repr(sorted(lp.items()))]
    for s in sorted(sessions, key=lambda x: x["label"]):
        parts.append(f"{s['label']}:{_session_sig(s)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


class Basis:
    """A fitted joint basis: shared footprints A, per-session component time courses C."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.basis_id = self.manifest["basis_id"]
        self._A = None

    @property
    def labels(self):
        return list(self.manifest["labels"])

    @property
    def ncomp(self) -> int:
        return int(self.manifest["ncomp"])

    @property
    def A(self):
        """Shared spatial footprints (H, W, ncomp). Memory-mapped — 180 MB otherwise."""
        if self._A is None:
            self._A = np.load(self.root / "A.npy", mmap_mode="r")
        return self._A

    def signal(self, label):
        """(ncomp, T) footprint-scaled components for one session — the same scaling
        ``locanmf_position_decoder._build_signal`` applies to per-session LocaNMF, so features from a
        joint basis are directly comparable with the per-session ones."""
        if label not in self.manifest["labels"]:
            raise KeyError(f"{label} is not in basis {self.basis_id} "
                           f"(has {len(self.manifest['labels'])} sessions)")
        C = np.load(self.root / f"C_{label}.npy")
        return _footprint_scale(np.asarray(self.A), C.shape[0])[:, None] * C

    def __repr__(self):
        return (f"<Basis {self.basis_id} {self.manifest['animal']} "
                f"{self.ncomp} comps, {len(self.labels)} sessions, rank {self.manifest['rank']}>")


def _seed_everything(seed=SEED):
    """LocaNMF draws from the numpy global RNG (sklearn ``randomized_svd``) and from torch
    (``randperm`` in its initialisation). Seed both. NB CUDA reductions are not bitwise deterministic,
    so a rebuild can still differ in the last digits — but not in component identity."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build(animal, sessions, rank=None, seed=SEED, out_dir=None, verbose=True):
    """Fit one joint LocaNMF over ``sessions`` (list of session dicts) and save it.

    Returns a ``Basis``. Re-running with the same inputs returns the existing one rather than refitting
    — pass ``out_dir`` to force a separate location."""
    labels = [s["label"] for s in sessions]
    bid = basis_id(sessions, rank, seed)
    root = Path(out_dir) if out_dir else BASIS_DIR / animal / bid
    if (root / "manifest.json").exists():
        if verbose:
            print(f"[joint_locanmf] {animal}: reusing existing basis {bid}", flush=True)
        return Basis(root)

    from wfield.local_nmf import compute_locaNMF
    _seed_everything(seed)
    mcs = [s["mc"] for s in sessions]
    Uj, Vj, _sv = joint_basis.build_joint_basis(mcs, rank=rank, labels=labels, verbose=verbose)

    ad = glob.glob(f"{sessions[0]['mc']}/wfield_local_results/allen_aligned_affine8v1")[0]
    atlas = np.load(f"{ad}/allen_area_atlas_native_grid.npy")
    mask = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool)
    H, W = atlas.shape

    bounds, t, cols = {}, 0, []
    for lab in labels:
        n = Vj[lab].shape[1]
        bounds[lab] = (t, t + n); t += n
        cols.append(Vj[lab])
    Vcat = np.concatenate(cols, axis=1)

    lp = config.defaults()["locanmf"]
    t0 = time.time()
    A, C, _reg = compute_locaNMF(Uj.reshape(H, W, -1).astype(np.float32), Vcat.astype(np.float32),
                                 atlas.astype(np.int32), mask, minrank=1, maxrank=lp["maxrank"],
                                 min_pixels=100, loc_thresh=lp["loc_thresh"],
                                 r2_thresh=lp["r2_thresh"], nonnegative_temporal=False, device="auto")
    A = np.asarray(A); C = np.asarray(C)
    elapsed = time.time() - t0

    tmp = root.with_name(root.name + f".{os.getpid()}.tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "A.npy", A.astype(np.float32))
    np.save(tmp / "U_joint.npy", Uj.astype(np.float32))
    for lab in labels:
        a, b = bounds[lab]
        np.save(tmp / f"C_{lab}.npy", C[:, a:b].astype(np.float32))
    manifest = {
        "basis_id": bid, "animal": animal, "labels": labels, "ncomp": int(A.shape[2]),
        "rank": int(rank) if rank else int(Uj.shape[1]), "seed": int(seed),
        "format_version": FORMAT_VERSION, "locanmf_params": lp,
        "session_signatures": {s["label"]: _session_sig(s) for s in sessions},
        "n_frames": {lab: int(bounds[lab][1] - bounds[lab][0]) for lab in labels},
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fit_seconds": round(elapsed, 1),
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, root)                      # atomic publish; a killed fit leaves only a .tmp
    if verbose:
        print(f"[joint_locanmf] {animal}: {A.shape[2]} shared components in {elapsed:.0f}s "
              f"-> {root}", flush=True)
    return Basis(root)


def load(animal, bid=None):
    """Newest saved basis for ``animal``, or a specific ``basis_id``. Raises if none exists —
    a missing basis should be an explicit build, never a silent refit."""
    d = BASIS_DIR / animal
    if bid:
        root = d / bid
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"no joint basis {bid} for {animal} under {d}")
        return Basis(root)
    cands = [p for p in sorted(d.glob("*")) if (p / "manifest.json").exists()] if d.exists() else []
    if not cands:
        raise FileNotFoundError(f"no joint basis for {animal} under {d} — build one first")
    cands.sort(key=lambda p: json.loads((p / "manifest.json").read_text())["built_utc"])
    return Basis(cands[-1])


def listing():
    """Every saved basis, newest last — for provenance when a figure's numbers are questioned."""
    out = []
    for man in sorted(BASIS_DIR.glob("*/*/manifest.json")):
        m = json.loads(man.read_text())
        out.append({k: m[k] for k in ("animal", "basis_id", "ncomp", "rank", "built_utc")}
                   | {"n_sessions": len(m["labels"])})
    return sorted(out, key=lambda r: r["built_utc"])
