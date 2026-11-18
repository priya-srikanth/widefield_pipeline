"""AR-HMM of whole-session orofacial + locomotor behaviour (port of Hillary Tong's jax-moseq pipeline,
github.com/htong1/Priya_ARHMM2026 `model/moseq.py` + `model/design.py`, adapted; decisions 11-16 in
docs/status/ENCODING_DESIGN_TIM_STATES_2026-11-16.md). RUNS IN THE `arhmm` CONDA ENV (CPU JAX + jax-moseq); imports
only numpy / jax_moseq so it never needs the analysis env.

    <arhmm python> -m wfield_local.arhmm_model scan  --feat <features dir> --out <dir>    # kappa -> median duration
    <arhmm python> -m wfield_local.arhmm_model fit   --feat <dir> --out <dir> --frame pooled_all|pre --kappa K
    <arhmm python> -m wfield_local.arhmm_model decode --feat <dir> --out <dir> --frame pooled_all

INPUT: `arhmm_features` session npz files (25 Hz bins of the ENCODER's behaviour variables: 3-D tongue, cam4 jaw,
cam4 nose, treadmill fwd / back, + 12 FR factors).

DESIGN (changes from hers, each deliberate):
- BLOCK WEIGHTS: after pooled z-scoring, every modality block (tongue / jaw / nose / run / fr) is scaled to the SAME
  total variance (each column x 1 / sqrt(block size)). Hers weighted DLC x2 (a sweep, lick-side NMI) because 6 DLC
  columns competed with 10 FR ones; here 12 FR columns would otherwise swamp 4 tongue columns. A sweep can follow.
  The z-scoring, block weights and PCA are computed ONCE on the fit set and applied UNCHANGED to every session (Priya
  11-16: "is this ok for pre vs post-stroke behavior?" -> yes, because it is a fixed transform: a post-stroke session
  with smaller licks or more running keeps those differences; per-session z-scoring would erase them and is never
  done). Caveats: the scale reflects the fit set's composition (pooled_all: equal sessions per animal x epoch; pre:
  pre only), and values beyond +/- 8 pooled SD are clipped.
- PCA to PCA_VAR of the pooled fit set's variance (hers: 90 %); the transform is saved and applied to every session.
- FIT FRAME (Priya 11-16): ``pooled_all`` = one vocabulary pooled over the 4 animals and ALL epochs (equal sessions per
  animal x epoch, FIT_PER_CELL), so post-stroke syllables can exist; ``pre`` = August-pre only (hers), kept for the
  frozen log-likelihood recovery score.
- GAPS: sequences split at any NaN bin; segments shorter than MIN_SEG_S dropped.
- KAPPA re-scanned at 25 Hz (hers 1e8 at 12.5 Hz -> 0.72 s median); target TARGET_MEDIAN_S.
- K upper bound NUM_STATES = 24 (hers); nlags 3 (= 120 ms at 25 Hz).
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FS_HZ = 25.0
PCA_VAR = 0.90
NUM_STATES = 24
NLAGS = 3
MIN_SEG_S = 2.0                   # 11-16: was 5 s; with the jaw fill, short pieces between rare gaps are kept
FIT_PER_CELL = 2                  # sessions per animal x epoch in the pooled fit set
TARGET_MEDIAN_S = 0.5
RUN_FLOOR = 0.3                   # mm/s, both running directions (Priya 11-16 "yes floor is ok"; `load_features`)
BLOCKS = {"tongue": ("tongue_",), "jaw": ("jaw_",), "nose": ("nose_",), "run": ("run_",), "fr": ("fr_",)}


# ---------------------------------------------------------------------------------------------------------- design ----
def block_of(col: str) -> str:
    for b, pre in BLOCKS.items():
        if col.startswith(pre):
            return b
    return "other"


def segments(X: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """(start, stop) index runs where every column is finite, at least ``min_len`` long."""
    ok = np.isfinite(X).all(1)
    d = np.diff(np.r_[0, ok.astype(int), 0])
    st, sp = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a, b) for a, b in zip(st, sp) if b - a >= min_len]


class Design:
    """Pooled z-score -> equal-variance block weights -> PCA; ``transform`` projects any session."""

    def __init__(self, columns, mats):
        self.columns = list(columns)
        P = np.vstack([m[np.isfinite(m).all(1)] for m in mats]).astype(np.float64)
        self.mu, self.sd = P.mean(0), P.std(0) + 1e-9
        blk = np.array([block_of(c) for c in self.columns])
        self.w = np.array([1.0 / np.sqrt((blk == b).sum()) for b in blk])
        Z = np.clip((P - self.mu) / self.sd, -8, 8) * self.w
        self.zm = Z.mean(0)
        Zc = Z - self.zm
        ev, V = np.linalg.eigh(Zc.T @ Zc / len(Zc))
        o = np.argsort(ev)[::-1]
        ev, V = ev[o], V[:, o]
        var = ev / ev.sum()
        n = int(np.searchsorted(np.cumsum(var), PCA_VAR)) + 1
        self.comps, self.var_ratio = V[:, :n].T, var[:n]

    def transform(self, X):
        Z = np.clip((X - self.mu) / self.sd, -8, 8) * self.w - self.zm
        return (Z @ self.comps.T).astype(np.float32)


    FIELDS = ("columns", "mu", "sd", "w", "zm", "comps", "var_ratio")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.FIELDS}

    @classmethod
    def from_dict(cls, d: dict) -> "Design":
        o = cls.__new__(cls)
        for k in cls.FIELDS:
            setattr(o, k, d[k])
        return o


#: Models are saved with the design as PLAIN DATA (`Design.to_dict`), never the class: this file runs as
#: ``python -m``, so its classes live in ``__main__``, and relabelling ``Design.__module__`` made pickle import a SECOND
#: copy of the module and refuse ("not the same object") -- all six O2 fits of 11-17 were lost at the save step.


def load_model(path):
    """A saved model dict with ``design`` rebuilt as a `Design`. Also reads the 11-16 check-fit file that pickled the
    class itself (``__main__.Design``)."""
    import pickle as _p

    class _U(_p.Unpickler):
        def find_class(self, module, name):
            if name == "Design":
                return Design
            return super().find_class(module, name)
    with open(path, "rb") as fh:
        M = _U(fh).load()
    if isinstance(M.get("design"), dict):
        M["design"] = Design.from_dict(M["design"])
    return M


# ----------------------------------------------------------------------------------------------------------- moseq ----
def _jax():
    import jax
    jax.config.update("jax_platform_name", "cpu")
    return jax


def hypparams(latent_dim, num_states=NUM_STATES, kappa=1e8, nlags=NLAGS):
    """MoSeq-style transition + AR hyperparameters (hers, unchanged)."""
    return ({"num_states": num_states, "alpha": 5.7, "gamma": 1e3, "kappa": float(kappa)},
            {"latent_dim": latent_dim, "nlags": nlags, "S_0_scale": 0.01, "K_0_scale": 10.0})


def make_data(seqs):
    """Padded (N, Tmax, D) + mask (padding, unlike her truncate-to-shortest default, so no data are dropped)."""
    jnp = _jax().numpy
    L = max(len(s) for s in seqs)
    x = np.zeros((len(seqs), L, seqs[0].shape[1]), np.float32)
    m = np.zeros((len(seqs), L), np.float32)
    for i, s in enumerate(seqs):
        x[i, :len(s)], m[i, :len(s)] = s, 1.0
    return {"x": jnp.asarray(x), "mask": jnp.asarray(m)}


def fit(data, kappa, num_iters, seed=0, num_states=NUM_STATES):
    jr = _jax().random
    from jax_moseq.models import arhmm
    tr, ar = hypparams(int(data["x"].shape[-1]), num_states, kappa)
    model = arhmm.init_model(data=data, trans_hypparams=tr, ar_hypparams=ar, seed=jr.PRNGKey(seed))
    import time
    t0 = time.time()
    for i in range(num_iters):
        model = arhmm.resample_model(data, **model)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  kappa {kappa:.0e}: iter {i + 1}/{num_iters} ({time.time() - t0:.0f} s)", flush=True)
    return model


def decode(data, model, num_passes=5):
    jr = _jax().random
    from jax_moseq.models import arhmm
    states = arhmm.init_states(jr.PRNGKey(0), data["x"], data["mask"], model["params"])
    m = {"seed": jr.PRNGKey(1), "states": states, "params": model["params"], "hypparams": model["hypparams"]}
    for _ in range(num_passes):
        m = arhmm.resample_model(data, **m, states_only=True)
    return np.asarray(m["states"]["z"])


def durations_s(z, mask) -> np.ndarray:
    out = []
    for s, mk in zip(np.asarray(z), np.asarray(mask)[:, NLAGS:]):
        s = s[mk > 0]
        ch = np.flatnonzero(np.diff(s) != 0)
        out.append(np.diff(np.r_[-1, ch, len(s) - 1]))
    return np.concatenate(out) / FS_HZ


# ------------------------------------------------------------------------------------------------------------ sets ----
def load_features(feat_dir, run_floor: float = 0.0):
    """{label: {X, columns}}; ``run_floor`` (mm/s) zeroes run_fwd / run_back below it -- the treadmill rest-noise floor
    (rest p90 ~0.2 mm/s; 0.3 = the push variable's threshold), so sign-split jitter at rest is not modelled as
    forward / backward movement (Priya 11-16 discussion; 0 = raw smoothed speed)."""
    out = {}
    for f in sorted(glob.glob(str(Path(feat_dir) / "sessions" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        X = z["X"].astype(np.float64)
        cols = [str(c) for c in z["features"]]
        if run_floor > 0:
            for c in ("run_fwd", "run_back"):
                j = cols.index(c)
                X[:, j] = np.where(X[:, j] < run_floor, 0.0, X[:, j])
        out[str(z["label"])] = {"X": X, "columns": cols}
    return out


def fit_labels(feats: dict, frame: str, epochs: dict, seed: int = 0, per_cell: int = FIT_PER_CELL) -> list[str]:
    """``pooled_all``: FIT_PER_CELL sessions per animal x epoch (random, seeded); ``pre``: every August-pre session."""
    rng = np.random.default_rng(seed)
    if frame == "pre":
        return sorted(lab for lab in feats if epochs.get(lab) == "pre" and lab[-4:] >= "0801")
    cells = {}
    for lab in sorted(feats):
        e = epochs.get(lab)
        if e is None or (e == "pre" and lab[-4:] < "0801"):
            continue
        cells.setdefault((lab.split("_")[0], e), []).append(lab)
    pick = []
    for _k, labs in sorted(cells.items()):
        pick += list(rng.choice(labs, size=min(per_cell, len(labs)), replace=False))
    return sorted(pick)


#: segments are cut into CHUNK_BINS-long pieces overlapping by NLAGS (keypoint-MoSeq's batching), because `make_data`
#: pads every sequence to the longest: whole-session segments (up to 182k bins) padded 772 segments to 27x the real
#: frames (pooled_all fit: 141 M padded vs 5.2 M real; measured 11-17). 1500 bins (60 s) -> 1.16x. Each chunk's states
#: cover [start + NLAGS, stop), so the overlapping chunks tile the segment exactly; the only cost is one forced state
#: re-draw per 60 s.
CHUNK_BINS = 1500
DITHER = 0.1          # 11-17 test (6 sessions): side-within-contact NMI .08 -> .36, median 0.12 -> 0.44 s


def holdout_fold(labs: list[str], fold: int) -> tuple[list[str], list[str]]:
    """Fold ``fold`` (1-based) holds out each animal's ``fold``-th fit session (date order); an animal with fewer
    sessions sits that fold out. Used on the ``pre`` frame so every pre-stroke day is scored by a model that never saw
    it (8 folds cover 31 sessions; the in-sample pre model would make the baseline fit look better than any post day).
    Returns (train, held_out)."""
    by = {}
    for lab in sorted(labs):
        by.setdefault(lab.split("_")[0], []).append(lab)
    held = sorted(v[fold - 1] for v in by.values() if len(v) >= fold)
    return [lab for lab in labs if lab not in held], held


def chunks(a: int, b: int, size: int = CHUNK_BINS) -> list[tuple[int, int]]:
    if b - a <= size:
        return [(a, b)]
    step = size - NLAGS
    k = int(np.ceil((b - a - NLAGS) / step))
    return [(a + i * step, min(a + i * step + size, b)) for i in range(k)]


def seqs_of(feats, labels, design, dither: float = 0.0):
    """PCA sequences (chunked). ``dither`` = SD of a FIXED Gaussian added in PC space (seeded by session + chunk start,
    so a fit and its decode see the same values): a floor on each state's AR noise. Without it the smooth 25 Hz PCs
    (median frame-to-frame change 0.01-0.04 vs PC SD 0.27-1.06) let states predict frames to ~0.007, so any change in
    dynamics out-bids the stickiness and kappa had to reach ~1e13 for 0.5 s syllables (11-17 scan)."""
    import zlib
    seqs, owner = [], []
    for lab in labels:
        X = feats[lab]["X"]
        for a0, b0 in segments(X, int(MIN_SEG_S * FS_HZ)):
            for a, b in chunks(a0, b0):
                Y = design.transform(X[a:b])
                if dither > 0:
                    rng = np.random.default_rng(zlib.crc32(f"{lab}:{a}".encode()))
                    Y = (Y + dither * rng.standard_normal(Y.shape)).astype(np.float32)
                seqs.append(Y)
                owner.append((lab, a, b))
    return seqs, owner


def per_session(z, owner, feats) -> dict:
    """Chunk states (N, T - NLAGS) -> {label: per-bin state, -1 = not modelled}; chunks tile each segment."""
    out = {}
    for (lab, s0, s1), zi in zip(owner, np.asarray(z)):
        full = out.setdefault(lab, np.full(len(feats[lab]["X"]), -1, np.int16))
        full[s0 + NLAGS:s1] = zi[: s1 - s0 - NLAGS]
    return out


# -------------------------------------------------------------------------------------------------------------- cli ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("cmd", choices=("scan", "fit", "decode", "score"))
    ap.add_argument("--feat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", required=True, help="json {label: epoch} (written by the analysis env)")
    ap.add_argument("--frame", default="pooled_all", choices=("pooled_all", "pre"))
    ap.add_argument("--kappa", type=float, default=1e8)
    ap.add_argument("--kappas", default="1e6,1e7,1e8,1e9,1e10", help="scan: comma list (one value per O2 task)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=FIT_PER_CELL)
    ap.add_argument("--run-floor", type=float, default=RUN_FLOOR)
    ap.add_argument("--tag", default=None, help="model name (default <frame>_s<seed>)")
    ap.add_argument("--label", default=None, help="decode: one session only (an O2 array task)")
    ap.add_argument("--scan-sessions", type=int, default=4)
    ap.add_argument("--save-states", action="store_true", help="scan: save each kappa's states per session")
    ap.add_argument("--dither", type=float, default=DITHER, help="PC-space noise floor SD (see seqs_of)")
    ap.add_argument("--holdout-fold", type=int, default=0,
                    help="fit: 1-based; drop each animal's k-th fit session (held-out pre baseline); tag gets _f<k>")
    ap.add_argument("--fit-labels", default=None, help="comma list: fit on exactly these sessions (sanity checks)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.frame}_s{a.seed}" + (f"_f{a.holdout_fold}" if a.holdout_fold else "")
    feats = load_features(a.feat, a.run_floor)
    epochs = json.loads(Path(a.epochs).read_text())
    if a.cmd == "score":
        # frame-by-frame MARGINAL log-likelihood of each session under this model (forward algorithm over every
        # state path; padded frames add 0), per frame = total / masked frames after the lags. Each model has its own
        # PCA basis, so compare likelihoods only WITHIN one model (held-out pre day vs post days under the same fold;
        # `behavior_states.frozen_distance`). Rows land in ll_<tag>.csv beside the model.
        from jax_moseq.models import arhmm
        M = load_model(out / f"model_{tag}.pkl")
        jnp = _jax().numpy
        P = {k: jnp.asarray(M["params"][k]) for k in ("Ab", "Q", "pi")}
        dest = out / f"ll_{tag}.csv"
        done = set(pd.read_csv(dest).label) if dest.exists() else set()
        rows = []
        for lab in ([a.label] if a.label else sorted(feats)):
            if lab in done:
                continue
            seqs, _own = seqs_of(feats, [lab], M["design"], M.get("dither", 0.0))
            if not seqs:
                continue
            d = make_data(seqs)
            n = float(np.asarray(d["mask"])[:, NLAGS:].sum())
            ll = float(arhmm.marginal_log_likelihood(d["mask"], d["x"], **P))
            rows.append({"label": lab, "model": tag, "ll_total": ll, "n_frames": n, "ll_per_frame": ll / n,
                         "in_fit": lab in set(M.get("fit_labels", [])), "held_out": lab in set(M.get("held_out", []))})
            print(f"scored {lab}: {ll / n:.3f} / frame", flush=True)
        if rows:
            new = pd.DataFrame(rows)
            allr = pd.concat([pd.read_csv(dest), new]) if dest.exists() else new
            tmp = dest.with_suffix(".csv.tmp")
            allr.to_csv(tmp, index=False)
            tmp.replace(dest)
        return 0
    if a.cmd == "decode":
        M = load_model(out / f"model_{tag}.pkl")
        jnp = _jax().numpy
        tojx = lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x                # noqa: E731
        params = _jax().tree_util.tree_map(tojx, M["params"])
        hyp = _jax().tree_util.tree_map(tojx, M["hypparams"])
        sd = out / f"states_{tag}"
        sd.mkdir(exist_ok=True)
        for lab in ([a.label] if a.label else sorted(feats)):
            f = sd / f"{lab}.npz"
            if f.exists():
                continue
            seqs, owner = seqs_of(feats, [lab], M["design"], M.get("dither", 0.0))
            if not seqs:
                continue
            z = decode(make_data(seqs), {"params": params, "hypparams": hyp})
            np.savez_compressed(f, z=per_session(z, owner, feats)[lab])
            print("decoded", lab, flush=True)
        return 0
    labs = (a.fit_labels.split(",") if a.fit_labels
            else fit_labels(feats, a.frame, epochs, seed=a.seed, per_cell=a.per_cell))
    held = []
    if a.holdout_fold:
        labs, held = holdout_fold(labs, a.holdout_fold)
        print(f"fold {a.holdout_fold}: holding out {held}", flush=True)
    design = Design(feats[labs[0]]["columns"], [feats[lab]["X"] for lab in labs])
    print(f"{a.frame}: {len(labs)} fit sessions; PCA {len(design.var_ratio)} dims ({design.var_ratio.sum():.2f})",
          flush=True)
    if a.cmd == "scan":
        # spread over the (sorted, animal-contiguous) fit set so every animal is in the scan, not only the first
        step = max(1, len(labs) // a.scan_sessions)
        scan_labs = labs[::step][: a.scan_sessions]
        seqs, own = seqs_of(feats, scan_labs, design, a.dither)
        data = make_data(seqs)
        for k in (float(x) for x in a.kappas.split(",")):
            m = fit(data, k, a.iters, seed=a.seed)
            if a.save_states:       # the last sample's states per session, for scoring against independent labels
                sd = out / f"scan_states_k{k:.0e}_d{a.dither:g}"
                sd.mkdir(exist_ok=True)
                for lab, zz in per_session(np.asarray(m["states"]["z"]), own, feats).items():
                    np.savez_compressed(sd / f"{lab}.npz", z=zz)
            d = durations_s(m["states"]["z"], data["mask"])
            row = {"kappa": k, "dither": a.dither, "median_s": float(np.median(d)), "mean_s": float(np.mean(d)),
                   "n_used": int(len(np.unique(np.asarray(m["states"]["z"])))), "iters": a.iters,
                   "sessions": scan_labs}
            (out / f"scan_{a.frame}_k{k:.0e}_d{a.dither:g}.json").write_text(json.dumps(row, indent=1))
            print(row, flush=True)
        return 0
    seqs, _owner = seqs_of(feats, labs, design, a.dither)
    data = make_data(seqs)
    m = fit(data, a.kappa, a.iters, seed=a.seed)
    d = durations_s(m["states"]["z"], data["mask"])
    print(f"fit {tag}: median syllable {np.median(d):.2f} s; saving", flush=True)
    # plain numpy throughout (params AND hypparams) and the design as plain data, so the analysis env (no JAX) can read
    # the model; serialized BEFORE the file is opened, so a failure cannot leave an empty model file behind
    tonp = lambda x: np.asarray(x) if hasattr(x, "shape") else x                           # noqa: E731
    blob = pickle.dumps({"params": _jax().tree_util.tree_map(tonp, m["params"]),
                         "hypparams": _jax().tree_util.tree_map(tonp, m["hypparams"]),
                         "design": design.to_dict(), "fit_labels": labs, "held_out": held, "kappa": a.kappa, "iters": a.iters,
                         "seed": a.seed, "run_floor": a.run_floor, "per_cell": a.per_cell, "dither": a.dither,
                         "median_dur_s": float(np.median(d))})
    tmp = out / f"model_{tag}.pkl.tmp"
    tmp.write_bytes(blob)
    tmp.replace(out / f"model_{tag}.pkl")
    print(f"fit {tag} done: median syllable {np.median(d):.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
