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

def _basis_dir() -> Path:
    """Where fitted joint bases live on THIS machine (override: WIDEFIELD_JOINT_BASIS_DIR).

    Derived from the machine's own working-figure root rather than a literal, for the same reason
    nightly_figs and session_cache were fixed: a drive-letter default is only correct on the machine
    it was written on."""
    env = os.environ.get("WIDEFIELD_JOINT_BASIS_DIR")
    if env:
        return Path(env)
    try:
        from wfield_local import config
        return Path(config.resolver().root("figures_working")).parent / "joint_bases"
    except Exception:                              # noqa: BLE001
        return Path("C:/wf_local/joint_bases")


BASIS_DIR = _basis_dir()
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
    svt = config.svtcorr_path(s['mc'])
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

    @property
    def regions(self):
        """Allen region label per component (ncomp,) — DERIVED from the footprints, not stored.

        ``compute_locaNMF`` returns a region assignment that ``build`` did not keep, and refitting a
        basis just to recover it would cost hours AND produce a different basis (LocaNMF is
        stochastic). It does not need to be stored: LocaNMF CONSTRAINS each component to one atlas
        region, so a component's support already identifies its region. Take the atlas label carrying
        the most footprint weight; ties cannot occur in practice because the support is contained in a
        single region by construction.

        Cached to ``regions.npy`` on first use so the atlas load happens once per basis.
        """
        cache = self.root / "regions.npy"
        if cache.exists():
            return np.load(cache)
        from wfield_local.locanmf_cue_lick_analysis import SESSIONS
        s = next(x for x in SESSIONS if x["label"] == self.labels[0])
        ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")[0]
        atlas = np.load(f"{ad}/allen_area_atlas_native_grid.npy").reshape(-1)
        A = np.nan_to_num(np.asarray(self.A, dtype=np.float32).reshape(-1, self.ncomp))
        labs = np.unique(atlas[atlas != 0])
        # weight per (region, component), then argmax over regions
        W = np.stack([np.abs(A[atlas == l]).sum(0) for l in labs])       # (nregions, ncomp)
        reg = labs[np.argmax(W, axis=0)].astype(int)
        try:
            np.save(cache, reg)
        except OSError:                                # read-only store -> recompute next time
            pass
        return reg

    def signal(self, label):
        """(ncomp, T) footprint-scaled components for one session — the same scaling
        ``locanmf_position_decoder._build_signal`` applies to per-session LocaNMF, so features from a
        joint basis are directly comparable with the per-session ones."""
        if label not in self.manifest["labels"]:
            raise KeyError(f"{label} is not in basis {self.basis_id} "
                           f"(has {len(self.manifest['labels'])} sessions)")
        C = np.load(self.root / f"C_{label}.npy")
        return _footprint_scale(np.asarray(self.A), C.shape[0])[:, None] * C

    @property
    def _operator(self):
        """``pinv(A) @ U_joint`` (ncomp, R) — projects joint-basis temporal weights onto the frozen
        footprints. Contracted in THIS order deliberately: ``pinv(A) @ (U_joint @ V)`` would
        materialize an (npix, T) movie (~276 GB for a long session), the same OOM that had to be fixed
        in ``locanmf_frozen_decoder.project_C_fixed_A``."""
        if getattr(self, "_op", None) is None:
            Uj = np.nan_to_num(np.load(self.root / "U_joint.npy")).astype(np.float32)
            Araw = np.asarray(self.A, dtype=np.float32).reshape(-1, self.ncomp)
            # LocaNMF writes NaN OUTSIDE each footprint's region, so a pixel finite for ANY component
            # is inside the brain. Restrict the least-squares to those rows: LocaNMF never fit the
            # out-of-mask pixels, and including them (where every footprint is 0 but the movie is not)
            # drags the solution toward explaining signal no component can carry.
            inside = np.isfinite(Araw).any(axis=1)
            # Within the mask, NaN means "this component has no weight here" -> 0, not missing.
            A = np.nan_to_num(Araw[inside])
            # pinv(A) = pinv(A'A) A' -- exact at any rank, but takes the pseudo-inverse of the
            # (ncomp, ncomp) Gram matrix rather than a (345600, ncomp) one: far cheaper, and
            # numerically far better behaved (the tall-matrix SVD failed to converge outright).
            G = np.asarray(A.T @ A, dtype=np.float64)
            self._op = np.linalg.pinv(G) @ np.asarray(A.T @ Uj[inside], dtype=np.float64)
            self._inside = inside
        return self._op

    def project(self, session, with_diagnostics=False, allow_cross_animal=False):
        """Components for a session that was NOT in the fit, with the footprints A held FIXED.

        This is what makes the basis a usable reference frame rather than a description of the
        sessions it was built from: a new pre-stroke day, and every post-stroke day, is expressed in
        the SAME components so its geometry is comparable to the reference set.

        ``variance_captured`` is the fraction of the session's own energy that survives projection onto
        the joint subspace. It is the honest health check on applying a frozen basis to new data —
        a post-stroke session whose activity has moved outside the pre-stroke subspace will show it
        here, and a low value means the resulting components under-describe that session rather than
        that its representation changed. Always report it alongside any projected result.
        """
        an = session["label"].split("_")[0]
        if an != self.manifest["animal"] and not allow_cross_animal:
            # NOT caught by variance_captured: a PS92 basis spans a PS95 session at 97%, because both
            # are cortex on the same Allen grid. The diagnostic detects a session leaving the basis's
            # subspace, not a session belonging to a different mouse. Only the label can catch that.
            raise ValueError(f"{session['label']} is {an} but this basis is for "
                             f"{self.manifest['animal']}; pass allow_cross_animal=True to override")
        u, v = joint_basis._load_session(session["mc"])
        Uj = np.load(self.root / "U_joint.npy")
        if u.shape[0] != Uj.shape[0]:
            raise ValueError(f"{session['label']} is not on the basis pixel grid "
                             f"({u.shape[0]} px vs {Uj.shape[0]}) — needs allen_aligned_affine8v1")
        P = Uj.T @ u                                  # (R, K) — never touches pixel-space time series
        C = self._operator @ (P @ v)                  # (ncomp, R)(R, K)(K, T)
        C = _footprint_scale(np.asarray(self.A), self.ncomp)[:, None] * C
        if not with_diagnostics:
            return C
        W = np.asarray(v @ v.T, dtype=np.float64)     # (K, K)
        den = float(np.sum(np.asarray(u.T @ u, dtype=np.float64) * W))
        num = float(np.sum(np.asarray(P.T @ P, dtype=np.float64) * W))
        return C, {"basis_id": self.basis_id, "label": session["label"],
                   "variance_captured": num / max(den, 1e-30), "in_fit": session["label"] in self.labels}

    def signal_or_project(self, session):
        """``signal`` when the session is in the fit, ``project`` when it is not — so callers do not
        have to branch, and cannot accidentally refit."""
        lab = session["label"] if isinstance(session, dict) else session
        if lab in self.manifest["labels"]:
            return self.signal(lab)
        if not isinstance(session, dict):
            raise KeyError(f"{lab} is not in basis {self.basis_id}; pass the session dict to project it")
        return self.project(session)

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


def basis_is_current(basis, sessions, rank=None, seed=SEED):
    """Does this saved basis match what its OWN session set would produce from today's inputs?

    ``basis_id`` makes two bases distinguishable on disk; it does NOT stop a stale one being loaded.
    ``load`` returns the newest saved basis regardless, so when the inputs changed underneath -- the
    2026-08-14 flip to the meegkit_hpfit SVTcorr, plus the LocaNMF refit -- every joint figure kept
    being built on a basis fitted to superseded data while the ROI figures had already moved. Silent,
    and only visible by recomputing the id.

    Returns (ok, expected_id). ``sessions`` should be the full registered list; the basis's own labels
    select from it, so this asks "same inputs?", never "same session set?".
    """
    want = [s for s in sessions if s["label"] in set(basis.labels)]
    if len(want) != len(basis.labels):
        return False, None
    # `basis_id` hashes the rank ARGUMENT verbatim, and `build` is normally called with rank=None
    # (resolve-to-default), while the manifest records the RESOLVED rank. Recomputing with the
    # resolved value therefore mismatches a basis that is perfectly current -- it flagged all four
    # freshly-built bases as stale, which is the cry-wolf failure that makes a guard worthless.
    # Accept either form; the id is what identifies the basis, not the spelling of its rank.
    cands = [rank] if rank is not None else [None, basis.manifest.get("rank", basis.ncomp)]
    for r in cands:
        exp = basis_id(want, r, seed)
        if exp == basis.basis_id:
            return True, exp
    return False, basis_id(want, cands[0], seed)


def load(animal, bid=None, sessions=None, warn_stale=True):
    """Newest saved basis for ``animal``, or a specific ``basis_id``. Raises if none exists —
    a missing basis should be an explicit build, never a silent refit.

    Pass ``sessions`` to have the loaded basis checked against today's inputs; a mismatch is WARNED
    about loudly rather than raised, because a fixed reference frame is the point (the post-stroke
    comparison needs the reference fixed BEFORE the manipulation) and a legitimately frozen basis will
    mismatch as soon as anything upstream is reprocessed. What must not happen is not noticing.
    """
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
    b = Basis(cands[-1])
    if warn_stale and sessions is not None:
        ok, exp = basis_is_current(b, sessions)
        if not ok:
            print(f"[joint_locanmf] *** STALE BASIS for {animal}: loaded {b.basis_id} "
                  f"(built {b.manifest.get('built_utc', '?')[:16]}) but today's inputs give "
                  f"{exp}.\n"
                  f"    Its figures describe SUPERSEDED data. Rebuild with build() -- which mints a "
                  f"NEW basis_id and leaves this one in place -- or pass the old bid deliberately.",
                  flush=True)
    return b


def listing():
    """Every saved basis, newest last — for provenance when a figure's numbers are questioned."""
    out = []
    for man in sorted(BASIS_DIR.glob("*/*/manifest.json")):
        m = json.loads(man.read_text())
        out.append({k: m[k] for k in ("animal", "basis_id", "ncomp", "rank", "built_utc")}
                   | {"n_sessions": len(m["labels"])})
    return sorted(out, key=lambda r: r["built_utc"])
