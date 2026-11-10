"""ALIGNED-DECODER ARM of the frozen vs refit spout decoding: is the post-stroke code the OLD code RESTORED, the old code
RE-EMBEDDED in a different cortical pattern, or REPLACED by a new code? (Priya 11-09; Gallego et al. 2020 Nat Neurosci
style latent alignment; the earlier label-free CCA version, FINDINGS 0c item 20 / REEMBEDDING_PREREG_20261030, failed its
pre->pre gate because spout-pooled trajectories do not pin the spout dimensions.)

DATA = the existing frozen / refit analysis, lick-aligned arm (5r / 5rm, `grant_confusion._collect_5c("lick", "lick")`),
captured through the D6 hook (`refit_matched_pre.capture`): per session the refit inputs (X, y, blocks) -- each
animal's joint-LocaNMF components in 8 sub-bins of the 2 s after the first lick (bin-major: feature b * C + c), z-scored
per feature on the session's engaged trials; cued trials with a camera-seen lick (LP licks, LP-inclusive gate); August
pre. Render profile `cohort1_rebuild_20261103`, block `epoch_accuracy`, WIDEFIELD_GATE_SOURCE=lp (= the D6 lick run's
folder) and WIDEFIELD_EPOCH_FROZEN=0 (no stored model is read or written; the frozen predictions of the capture are not
used here).

LATENTS. Per animal, k = 10 PCs (also 5 and 20) of single-trial activity: samples = every (trial, bin), variables =
the C components. Reference = PCA of the animal's pre sessions (the scored pre session left out).
PER SESSION, 2-fold cross-validation (GroupKFold over the scheduler's position blocks): the CALIBRATION half fits
everything session-specific, the other half is scored; session value = mean of the two folds. Every decoder is
`locanmf_frozen_decoder._pipe` (standardize + multinomial logistic, C = 0.5) on the k latents x 8 bins.
  frozen    calibration-free: held-out trials projected on the reference PCs, the FROZEN pre decoder. The frozen decoder
            is trained on a SIZE-MATCHED random set of reference trials (whole blocks, n = the calibration half), as the
            5rm matched frozen arm -- so frozen and refit see the same number of training trials.
  (a) cov   label-free covariance alignment: the session's OWN k PCs (calibration half), whitened by the own latent
            covariance and re-coloured by the reference latent covariance: A = S_own^-1/2 Q S_ref^1/2. Q = the
            orthogonal factor of W_own' W_ref (own -> reference axis correspondence through the SHARED joint-basis
            components; no labels; with session-z-scored features the spout-pooled mean trajectory is ~0 and cannot fix
            the axes). Then the frozen decoder.
  (b) pro   orthogonal Procrustes (with one isotropic scale) of the own condition-averaged latent trajectories (per
            spout x 8 bins, calibration half, spouts with >= 3 calibration trials) onto the reference ones -> frozen
            decoder.
  refit     own PCs, decoder trained on the calibration half's labels (same size as the frozen training set).
  controls  rotation-only (no scaling): (a) Q alone, (b) R alone; (b) fit after SHUFFLING the calibration labels
            (N_SHUF permutations, mean); pre -> pre = the same for each held-out pre session against its LOSO reference
            (the ceiling); PAIRWISE mode (k = 10): reference = ONE pre session, so every (target, reference) pair has a
            lag in days -> the pre -> pre ceiling at MATCHED LAG.
READOUTS: accuracy (chance 1/6; balanced accuracy beside it); gap fraction (aligned - frozen) / (refit - frozen) of
animal-epoch means, per animal and per epoch with the nested animal -> session bootstrap (`analysis_kit
.paired_animal_draws`, animal-weighted); change vs pre (`boot_delta`) of frozen, aligned, refit, A = aligned - frozen,
R = refit - aligned, G = refit - frozen; index = dA / dG (lesion-attributable part of the gap closed).
CORTICAL DECOMPOSITION (pooled reference, k = 10): the readout the frozen decoder applies, carried to pixels through
the joint-basis footprints (A_c / footprint scale; weights in session-z units): reference readout W_ref beta vs aligned
readout W_own A beta on the target's components. Loading share = pattern energy in an ROI / whole brain, ROI v8
(`area_basis.roi_masks`: tjM1 = anterolateral 50 % of MOp, SSp-m), _left = IPSILESIONAL, _right = CONTRALESIONAL.
Hemisphere RE-WEIGHTING = the contra share changes; WITHIN-HEMISPHERE ROTATION = the cosine between the aligned and
reference patterns inside a hemisphere falls. Both read against pre -> pre.
VERDICT per epoch and alignment (stated before any post number was looked at, except the gate): gate = pre -> pre
alignment EFFICIENCY (aligned - 1/6) / (frozen - 1/6) of animal means >= 0.7 in >= 3 / 4 animals (the 10-30 prereg's
G1; Priya 11-09 after the first run -- the gate first written, pre -> pre gap fraction >= 0.7, divides by the ~0 pre
gap of size-matched decoders and cannot pass when nothing has moved; it is still reported, `gate_pre_fraction`);
RESTORED = frozen and G not changed vs pre; RE-EMBEDDED = G changed, index >= 0.67 in >= 3 / 4
animals and R not changed; REPLACED = G changed, index <= 0.33 in >= 3 / 4 and A not changed; else PARTIAL.

SVD ARM (`--basis svd`, Priya 11-09: LocaNMF primary, SVD as the bridge to the state-space maps): the same pipeline on
the state-space extract in the COHORT SVD basis (100 components shared by all animals, glue-free brain mask;
`state_space_sim` with WF_SS_BASIS=svd, WF_SS_ALIGN=lick): first-lick trials, 8 bins of 0.1 s from -0.1 to +0.7 s
after the first lick (the extract ends at +0.7 s), z-scored per feature per session like the decoder features; blocks
= runs of one spout in trial order. Its trial set is the state-space one, not the decoder's. ROI loadings through
U_cohort on the same pixel grid. Outputs <state_space_recovery>/aligned_decoder_svd/.

    python scripts/aligned_decoder.py                 # capture (cached) + score + summarise
    python scripts/aligned_decoder.py --summarise     # stats + figure from the rows already written
    python scripts/aligned_decoder.py --basis svd     # the SVD arm
Outputs (rebuild root, LP lick run folder): <epoch data>/aligned_decoder/{inputs_lick_lick.pkl, rois.npz, rows.csv,
animals.csv, tests.csv, lag.csv, verdict.csv, figdata.pkl, aligned_decoder.png}. EXPLORATORY.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROFILE, BLOCK = "cohort1_rebuild_20261103", "epoch_accuracy"
ALIGN, VARIANT = "lick", "lick"
K_LIST, K_MAIN = (5, 10, 20), 10
N_SHUF = 5
MIN_COND = 3
EPOCHS = ("pre", "acute", "subacute", "chronic")
ARMS = ("frozen", "cov", "cov_rot", "pro", "pro_rot", "pro_shuf", "refit")
ALNS = ("cov", "pro")
ROIS = ("tjM1_left", "tjM1_right", "SSp-m_left", "SSp-m_right")


def setup_env() -> Path:
    from wfield_local import render_profile
    render_profile.apply(PROFILE, block=BLOCK)
    os.environ["WIDEFIELD_GATE_SOURCE"] = "lp"
    os.environ["WIDEFIELD_EPOCH_FROZEN"] = "0"
    from wfield_local import config, lp_licks
    return lp_licks.output_dir(Path(config.resolver().root("cohort_out")) / "grant_figures" / "epoch") / "data" / \
        "aligned_decoder"


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def _day(label: str) -> int:
    import datetime as dt
    mmdd = label.split("_")[-1]
    return dt.date(2026, int(mmdd[:2]), int(mmdd[2:])).toordinal()


# ---------------------------------------------------------------------------------------------------------------- inputs

def load_inputs(out: Path) -> list[dict]:
    f = out / f"inputs_{ALIGN}_{VARIANT}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    import refit_matched_pre as rmp

    from wfield_local import epochs, writeguard
    cap = rmp.capture(ALIGN, VARIANT)
    for c in cap:
        c["epoch"] = epochs.epoch_of(c["label"])
        c["day"] = _day(c["label"])
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(f, "wb") as fh:
        pickle.dump(cap, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return cap


SVD_BINS = slice(9, 17)          # state-space lick bins are 0.1 s from -1.0 s: -0.1 .. +0.7 s after the first lick


def svd_inputs() -> list[dict]:
    """The state-space SVD extract as the capture's records (animal, label, pre, X bin-major, y, blk, epoch, day)."""
    import state_space_sim as S
    data, _cols = S.load_sessions()
    out = []
    for an in sorted(data):
        for i, (lab, ep, _dss, Y, pos) in enumerate(data[an]):
            if ep not in EPOCHS:
                continue
            X3 = np.asarray(Y, float)[:, SVD_BINS]
            ok = np.isfinite(X3).all((1, 2)) & np.isin(pos, S.POS)
            X3, p = X3[ok], np.asarray(pos)[ok]
            if len(p) < 20:
                continue
            sd = X3.std(0)
            X3 = (X3 - X3.mean(0)) / np.where(sd > 0, sd, 1.0)                  # session z, as the decoder features
            y = np.array([S.POS.index(q) for q in p])
            blk = np.concatenate([[0], np.cumsum(y[1:] != y[:-1])]) + i * 10_000_000   # unique within the animal
            out.append(dict(animal=an, label=lab, pre=ep == "pre", X=X3.reshape(len(y), -1), y=y,
                            blk=blk.astype(np.int64), epoch=ep, day=_day(lab)))
    return out


def svd_grams() -> dict:
    """ROI Gram matrices of the cohort SVD basis (U_cohort on the brain-mask pixels; ROI v8 masks on the same grid)."""
    import state_space_sim as S

    from wfield_local import area_basis, beta_maps
    Uc = np.asarray(np.load(S.SVD.parent / "basis" / "U_cohort.npy", mmap_mode="r"), np.float64)
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    shape = atlas.shape
    flat = np.flatnonzero(np.load(S.SVD.parent / "basis" / "mask.npy"))                # U_cohort rows (flat bool mask)
    sm = beta_maps.stat_mask()
    pstat = sm.ravel()[flat].astype(bool)
    M = area_basis.roi_masks(atlas.ravel()[flat], pstat, S._atlas_names(), flat, shape)
    return _grams_from(Uc, M, pstat)


def _grams_from(F, M, pstat) -> dict:
    left = np.zeros(len(pstat), bool)
    right = np.zeros(len(pstat), bool)
    for n, m in M.items():
        if n in ("ALM_left", "ALM_right", "tjM1_left", "tjM1_right"):
            continue
        left |= m & n.endswith("_left")
        right |= m & n.endswith("_right")
    masks = {r: M[r] for r in ROIS}
    masks.update(left=left, right=right, all=pstat.astype(bool))
    return {r: F[m].T @ F[m] for r, m in masks.items()}


def roi_grams(animal: str, ncomp: int) -> dict:
    """{roi: F_r' F_r (C x C)} with F = joint-basis footprints / footprint scale on the stat-mask pixels; ROIs = ROI v8
    tjM1 / SSp-m per hemisphere, the two hemispheres, and the whole brain."""
    import json

    from wfield_local import area_basis, beta_maps, config, joint_locanmf
    from wfield_local.locanmf_crossanimal_dff import _footprint_scale
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    basis = joint_locanmf.load(animal, sessions=SESSIONS)
    if basis.ncomp != ncomp:
        raise RuntimeError(f"{animal}: basis {basis.basis_id} has {basis.ncomp} components, features {ncomp}")
    A = np.asarray(basis.A, dtype=np.float64)
    H, W, C = A.shape
    s = next(x for x in config.load_sessions() if x["label"] == basis.labels[0])
    at = Path(s["mc"]) / "wfield_local_results" / "allen_aligned_affine8v1"
    area = np.load(at / "allen_area_atlas_native_grid.npy")
    names = {int(c): n for c, n in json.load(open(at / "allen_area_names.json"))}
    bm, sm = beta_maps.brain_mask(), beta_maps.stat_mask()
    if bm is None or bm.shape != (H, W):
        raise RuntimeError(f"{animal}: brain mask {None if bm is None else bm.shape} vs footprints {(H, W)}")
    flat = np.flatnonzero(bm.ravel())
    pstat = sm.ravel()[flat]
    M = area_basis.roi_masks(area.ravel()[flat], pstat, names, flat, (H, W))
    F = np.nan_to_num(A.reshape(-1, C)[flat]) / np.where(_footprint_scale(A, C) > 0, _footprint_scale(A, C), 1.0)
    return _grams_from(F, M, pstat)


# ---------------------------------------------------------------------------------------------------------------- model

def _pca(X3: np.ndarray, k: int):
    """X3 (n, B, C) -> (W (C x k), variances (k,)); samples = every (trial, bin)."""
    Z = X3.reshape(-1, X3.shape[-1])
    Z = Z - Z.mean(0)
    _u, s, vt = np.linalg.svd(Z, full_matrices=False)
    return vt[:k].T, (s[:k] ** 2) / max(len(Z) - 1, 1)


def _lat(X3, W):
    return X3 @ W


def _flat(L3):
    return L3.reshape(len(L3), -1)


def _cond(L3, y, classes):
    return {int(q): L3[y == q].mean(0) for q in classes if (y == q).sum() >= MIN_COND}


def _procrustes(Mt: dict, Mr: dict):
    common = sorted(set(Mt) & set(Mr))
    if len(common) < 2:
        return None, None
    Pt = np.concatenate([Mt[q] for q in common])
    Pr = np.concatenate([Mr[q] for q in common])
    u, s, vt = np.linalg.svd(Pt.T @ Pr)
    R = u @ vt
    return R, float(s.sum() / max((Pt ** 2).sum(), 1e-12))


def _sqrtm(S, inv=False):
    w, v = np.linalg.eigh((S + S.T) / 2)
    w = np.clip(w, 1e-9, None)
    return (v * (w ** (-0.5 if inv else 0.5))) @ v.T


def _matched_rows(y, blk, n_target, rng):
    """Whole blocks in random order until >= n_target rows (the `_matched_frozen` rule)."""
    keep = np.zeros(len(y), bool)
    for b in rng.permutation(np.unique(blk)):
        keep |= blk == b
        if keep.sum() >= n_target:
            break
    return np.flatnonzero(keep)


def _scores(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(np.mean(p == y)), float(balanced_accuracy_score(y, p))


class Reference:
    """PCA + condition averages + covariance of one reference set (a list of pre sessions), for one k."""

    def __init__(self, sessions: list[dict], k: int, B: int):
        X3 = np.concatenate([s["X"].reshape(len(s["y"]), B, -1) for s in sessions])
        self.y = np.concatenate([s["y"] for s in sessions]).astype(int)
        self.blk = np.concatenate([np.asarray(s["blk"]).astype(np.int64) for s in sessions])  # unique across sessions
        self.W, self.var = _pca(X3, k)
        self.L = _lat(X3, self.W)
        self.S = np.cov(self.L.reshape(-1, k).T)
        self.M = _cond(self.L, self.y, np.unique(self.y))

    def decoder(self, n_target, rng):
        from wfield_local.locanmf_frozen_decoder import _pipe
        rows = _matched_rows(self.y, self.blk, n_target, rng)
        if len(np.unique(self.y[rows])) < 2:
            return None
        return _pipe().fit(_flat(self.L[rows]), self.y[rows])


def score_session(T: dict, refs: dict, k: int, B: int, want_maps: bool, grams: dict | None) -> dict | None:
    """All arms on one target session against one Reference, 2-fold; returns fold-mean scores (+ maps)."""
    from sklearn.model_selection import GroupKFold

    from wfield_local.locanmf_frozen_decoder import _pipe
    y = np.asarray(T["y"]).astype(int)
    blk = np.asarray(T["blk"])
    X3 = T["X"].reshape(len(y), B, -1)
    if np.unique(blk).size < 2:
        return None
    ref: Reference = refs["ref"]
    acc = {a: [] for a in ARMS}
    bal = {a: [] for a in ARMS}
    maps = []
    for f, (cal, tst) in enumerate(GroupKFold(2).split(X3, y, blk)):
        if len(np.unique(y[cal])) < 2:
            continue
        rng = np.random.default_rng(_seed(T["label"], refs["tag"], k, f))
        D = ref.decoder(len(cal), rng)
        if D is None:
            continue
        Wt, _v = _pca(X3[cal], k)
        Lc, Ls = _lat(X3[cal], Wt), _lat(X3[tst], Wt)
        St = np.cov(Lc.reshape(-1, k).T)
        u, _s, vt = np.linalg.svd(Wt.T @ ref.W)
        Q = u @ vt
        A_cov = _sqrtm(St, inv=True) @ Q @ _sqrtm(ref.S)
        R, sc = _procrustes(_cond(Lc, y[cal], np.unique(y)), ref.M)
        pred = {"frozen": D.predict(_flat(_lat(X3[tst], ref.W))),
                "cov": D.predict(_flat(Ls @ A_cov)), "cov_rot": D.predict(_flat(Ls @ Q))}
        if R is not None:
            pred["pro"] = D.predict(_flat(Ls @ (sc * R)))
            pred["pro_rot"] = D.predict(_flat(Ls @ R))
            sh = []
            for j in range(N_SHUF):
                yp = np.random.default_rng(_seed(T["label"], refs["tag"], k, f, "shuf", j)).permutation(y[cal])
                Rs, ss = _procrustes(_cond(Lc, yp, np.unique(y)), ref.M)
                if Rs is not None:
                    sh.append(_scores(y[tst], D.predict(_flat(Ls @ (ss * Rs)))))
            if sh:
                a_, b_ = np.mean(sh, 0)
                acc["pro_shuf"].append(a_)
                bal["pro_shuf"].append(b_)
        pred["refit"] = _pipe().fit(_flat(Lc), y[cal]).predict(_flat(Ls))
        for a, p in pred.items():
            a_, b_ = _scores(y[tst], p)
            acc[a].append(a_)
            bal[a].append(b_)
        if want_maps and grams is not None:
            maps.append(readout_maps(D, ref.W, Wt, {"cov": A_cov, "pro": None if R is None else sc * R}, B, k, grams))
    if not acc["frozen"]:
        return None
    out = {f"acc_{a}": float(np.mean(v)) if v else np.nan for a, v in acc.items()}
    out.update({f"bal_{a}": float(np.mean(v)) if v else np.nan for a, v in bal.items()})
    out["n"] = len(y)
    if maps:
        keys = set().union(*maps)
        out.update({kk: float(np.nanmean([m.get(kk, np.nan) for m in maps])) for kk in keys})
    return out


def readout_maps(D, Wr, Wt, A: dict, B: int, k: int, G: dict) -> dict:
    """Loading shares / within-hemisphere cosines of the aligned readout vs the reference readout (module docstring)."""
    sc = D.named_steps["standardscaler"].scale_
    beta = (D.named_steps["logisticregression"].coef_ / sc).reshape(-1, B, k)        # classes x bins x k
    bet = beta.reshape(-1, k).T                                                       # k x (classes * bins)
    Eref = Wr @ bet

    def share(E, r):
        return float(np.trace(E.T @ G[r] @ E) / np.trace(E.T @ G["all"] @ E))

    def cos(Ea, Eb, r):
        return float(np.trace(Ea.T @ G[r] @ Eb) / np.sqrt(np.trace(Ea.T @ G[r] @ Ea) * np.trace(Eb.T @ G[r] @ Eb)))
    out = {}
    for r in (*ROIS, "right"):
        out[f"share_ref_{r}"] = share(Eref, r)
    for name, M in A.items():
        if M is None:
            continue
        E = Wt @ M @ bet
        for r in (*ROIS, "right"):
            out[f"share_{name}_{r}"] = share(E, r)
            out[f"dshare_{name}_{r}"] = out[f"share_{name}_{r}"] - out[f"share_ref_{r}"]
        for h in ("left", "right"):
            out[f"cos_{name}_{h}"] = cos(E, Eref, h)
    return out


def unit_job(unit) -> list[dict]:
    """One target session: pooled reference at every k (maps at K_MAIN) + pairwise single-pre references at K_MAIN."""
    T, pres, B, grams = unit
    rows = []
    others = [P for P in pres if P["label"] != T["label"]]
    if not others:
        return rows
    for k in K_LIST:
        if k >= T["X"].shape[1] // B:
            continue
        ref = Reference(others, k, B)
        r = score_session(T, {"ref": ref, "tag": "pool"}, k, B, want_maps=k == K_MAIN, grams=grams)
        if r:
            rows.append(dict(mode="pool", ref="pool", lag=float(T["day"] - np.mean([P["day"] for P in others])),
                             k=k, **r))
    for P in others:
        ref = Reference([P], K_MAIN, B)
        r = score_session(T, {"ref": ref, "tag": P["label"]}, K_MAIN, B, want_maps=False, grams=None)
        if r:
            rows.append(dict(mode="pair", ref=P["label"], lag=float(T["day"] - P["day"]), k=K_MAIN, **r))
    for r in rows:
        r.update(animal=T["animal"], label=T["label"], epoch=T["epoch"], day=T["day"])
    return rows


# ---------------------------------------------------------------------------------------------------------------- summary

def per_session(R: pd.DataFrame, k=K_MAIN, mode="pool") -> pd.DataFrame:
    D = R[(R["mode"] == mode) & (R.k == k)].copy()
    for a in ALNS:
        D[f"A_{a}"] = D[f"acc_{a}"] - D.acc_frozen
        D[f"R_{a}"] = D.acc_refit - D[f"acc_{a}"]
    D["G"] = D.acc_refit - D.acc_frozen
    return D


def _by_animal(D, ep, col):
    g = D[D.epoch == ep]
    return {a: g[g.animal == a][col].dropna().tolist() for a in sorted(g.animal.unique())}


def fraction_draws(D, ep, aln, rng, n_boot=2000):
    """Gap fraction (aligned - frozen) / (refit - frozen) of animal-weighted means, nested animal -> session draws
    (`paired_animal_draws` against a zero pre arm, so it IS the canonical draw); per-animal fractions."""
    from wfield_local import analysis_kit as ak
    g = D[D.epoch == ep]
    rec = {a: list(zip(g[g.animal == a][f"A_{aln}"], g[g.animal == a].G)) for a in sorted(g.animal.unique())}
    rec = {a: v for a, v in rec.items() if v}
    if not rec:
        return None
    zero = {a: [(0.0, 0.0)] for a in rec}

    def red(v, _rng):
        return np.asarray(v, float).mean(0)

    def comb(d):
        d = np.asarray(d)
        return float(d[:, 0].mean() / d[:, 1].mean()) if abs(d[:, 1].mean()) > 1e-9 else np.nan
    point, draws, _an = ak.paired_animal_draws(rec, zero, red, rng, n_boot, combine=comb)
    draws = draws[np.isfinite(draws)]
    per = {a: (float(np.mean([x for x, _ in v])) / float(np.mean([y for _, y in v]))
               if abs(np.mean([y for _, y in v])) > 1e-9 else np.nan) for a, v in rec.items()}
    return point, np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan), per


def index_draws(D, ep, aln, rng, n_boot=2000):
    """dA / dG (epoch minus pre, animal-weighted) from the same paired draws; per-animal ratios."""
    from wfield_local import analysis_kit as ak

    def recs(e):
        g = D[D.epoch == e]
        return {a: list(zip(g[g.animal == a][f"A_{aln}"], g[g.animal == a].G)) for a in sorted(g.animal.unique())}
    post, pre = recs(ep), recs("pre")

    def red(v, _rng):
        return np.asarray(v, float).mean(0)

    def comb(d):
        d = np.asarray(d)
        return float(d[:, 0].mean() / d[:, 1].mean()) if abs(d[:, 1].mean()) > 1e-9 else np.nan
    point, draws, animals = ak.paired_animal_draws(post, pre, red, rng, n_boot, combine=comb)
    draws = draws[np.isfinite(draws)]
    per = {}
    for a in animals:
        da = np.mean([x for x, _ in post[a]]) - np.mean([x for x, _ in pre[a]])
        dg = np.mean([y for _, y in post[a]]) - np.mean([y for _, y in pre[a]])
        per[a] = float(da / dg) if abs(dg) > 1e-9 else np.nan
    return point, np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan), per


def summarise(out: Path) -> None:
    from wfield_local import analysis_kit as ak
    R = pd.read_csv(out / "rows.csv")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    tests, verdict, frac_rows = [], [], []
    for k in K_LIST:
        D = per_session(R, k)
        if D.empty:
            continue
        cols = [f"acc_{a}" for a in ARMS] + ["G"] + [f"A_{a}" for a in ALNS]
        print(f"\n== k = {k}: per-animal session means (accuracy; chance 1/6), pooled pre reference")
        print(D.groupby(["animal", "epoch"])[cols].mean().reindex(EPOCHS, level=1).round(3).to_string())
        for aln in ALNS:
            for ep in EPOCHS:
                got = fraction_draws(D, ep, aln, np.random.default_rng(_seed("frac", k, aln, ep)))
                if got is None:
                    continue
                pt, (lo, hi), per = got
                frac_rows.append(dict(k=k, alignment=aln, epoch=ep, fraction=pt, lo=lo, hi=hi,
                                      **{f"f_{a}": v for a, v in per.items()}))
        for ep in EPOCHS[1:]:
            for col in cols + [f"R_{a}" for a in ALNS] + [f"bal_{a}" for a in ARMS]:
                iv = ak.boot_delta(_by_animal(D, ep, col), _by_animal(D, "pre", col),
                                   np.random.default_rng(_seed("delta", k, col, ep)))
                per = {a: float(np.mean(v)) - float(np.mean(_by_animal(D, "pre", col).get(a, [np.nan])))
                       for a, v in _by_animal(D, ep, col).items() if v}
                if iv is None:
                    continue
                tests.append(dict(k=k, epoch=ep, quantity=col, change=iv.point, lo=iv.lo, hi=iv.hi,
                                  change_sig=bool(iv.lo > 0 or iv.hi < 0),
                                  k_same=int(sum(np.sign(v) == np.sign(iv.point) for v in per.values())),
                                  **{f"d_{a}": round(v, 3) for a, v in per.items()}))
            for aln in ALNS:
                pt, (lo, hi), per = index_draws(D, ep, aln, np.random.default_rng(_seed("index", k, aln, ep)))
                tests.append(dict(k=k, epoch=ep, quantity=f"index_{aln}", change=pt, lo=lo, hi=hi,
                                  change_sig=np.nan, k_same=np.nan, **{f"d_{a}": round(v, 3) for a, v in per.items()}))
    F = pd.DataFrame(frac_rows)
    T = pd.DataFrame(tests)
    F.to_csv(out / "animals.csv", index=False, float_format="%.4g")
    T.to_csv(out / "tests.csv", index=False, float_format="%.4g")
    print("\n== gap fraction (aligned - frozen) / (refit - frozen), per epoch (pre = the pre -> pre ceiling)")
    print(F.round(3).to_string(index=False))
    show = T[T.quantity.isin(["acc_frozen", "acc_cov", "acc_pro", "acc_refit", "G", "A_cov", "A_pro", "R_cov",
                              "R_pro", "index_cov", "index_pro", "acc_pro_shuf", "acc_cov_rot", "acc_pro_rot"])]
    print("\n== change vs pre (boot_delta; index = dA / dG paired draws)")
    print(show.round(3).to_string(index=False))

    # verdicts (rule fixed in the docstring)
    for k in K_LIST:
        for aln in ALNS:
            f_pre = F[(F.k == k) & (F.alignment == aln) & (F.epoch == "pre")]
            if f_pre.empty:
                continue
            fp = [v for c, v in f_pre.iloc[0].items() if c.startswith("f_") and np.isfinite(v)]
            pre = per_session(R, k)
            pre = pre[pre.epoch == "pre"].groupby("animal")[[f"acc_{aln}", "acc_frozen"]].mean()
            eff = ((pre[f"acc_{aln}"] - 1 / 6) / (pre.acc_frozen - 1 / 6)).to_numpy()
            gate = sum(v >= 0.7 for v in eff) >= 3
            for ep in EPOCHS[1:]:
                t = T[(T.k == k) & (T.epoch == ep)].set_index("quantity")
                if "G" not in t.index:
                    continue
                sig = lambda q, t=t: bool(t.loc[q, "change_sig"]) if q in t.index else False  # noqa: E731
                idx = t.loc[f"index_{aln}"]
                per = [v for c, v in idx.items() if c.startswith("d_") and np.isfinite(v)]
                if not gate:
                    v = "UNINFORMATIVE (gate)"
                elif not sig("G") and not sig("acc_frozen"):
                    v = "RESTORED"
                elif not sig("G"):
                    v = "no extra gap (frozen changed with refit)"
                elif sum(x >= 0.67 for x in per) >= 3 and not sig(f"R_{aln}"):
                    v = "RE-EMBEDDED"
                elif sum(x <= 0.33 for x in per) >= 3 and not sig(f"A_{aln}"):
                    v = "REPLACED"
                else:
                    v = "PARTIAL"
                verdict.append(dict(k=k, alignment=aln, epoch=ep, gate_efficiency=np.round(eff, 2).tolist(),
                                    gate_pass=gate, gate_pre_fraction=np.round(fp, 2).tolist(), G_change=t.loc["G", "change"], G_sig=sig("G"),
                                    index=idx["change"], index_lo=idx["lo"], index_hi=idx["hi"],
                                    index_per_animal=np.round(per, 2).tolist(), verdict=v))
    V = pd.DataFrame(verdict)
    V.to_csv(out / "verdict.csv", index=False, float_format="%.4g")
    print("\n== verdicts")
    print(V.to_string(index=False))

    # lag-matched ceiling (pairwise, k = K_MAIN)
    P = per_session(R, K_MAIN, "pair")
    if not P.empty:
        P["lag_bin"] = pd.cut(P.lag.abs(), [0, 7, 14, 28, 120], labels=["1-7", "8-14", "15-28", ">28"])
        L = P.groupby(["epoch", "lag_bin"], observed=True)[["acc_frozen", "acc_cov", "acc_pro", "acc_refit", "G",
                                                            "A_cov", "A_pro"]].mean()
        L["frac_cov"] = L.A_cov / L.G
        L["frac_pro"] = L.A_pro / L.G
        L["n_pairs"] = P.groupby(["epoch", "lag_bin"], observed=True).size()
        L.to_csv(out / "lag.csv", float_format="%.4g")
        print("\n== pairwise (one pre session as reference), by |lag| in days: pre rows = pre -> pre ceiling")
        print(L.round(3).to_string())

    # cortical decomposition
    D = per_session(R, K_MAIN)
    mc = [c for c in D.columns if c.startswith(("dshare_", "cos_", "share_ref_"))]
    if mc:
        print("\n== cortical decomposition (k = 10, pooled reference): per-animal epoch means")
        print(D.groupby(["animal", "epoch"])[mc].mean().reindex(EPOCHS, level=1).round(3).to_string())
        dec = []
        for ep in EPOCHS[1:]:
            for col in mc:
                if col.startswith("share_ref_"):
                    continue
                iv = ak.boot_delta(_by_animal(D, ep, col), _by_animal(D, "pre", col),
                                   np.random.default_rng(_seed("map", col, ep)))
                per = {a: float(np.mean(v)) - float(np.mean(_by_animal(D, "pre", col).get(a, [np.nan])))
                       for a, v in _by_animal(D, ep, col).items() if v}
                if iv is not None:
                    dec.append(dict(epoch=ep, quantity=col, change=iv.point, lo=iv.lo, hi=iv.hi,
                                    change_sig=bool(iv.lo > 0 or iv.hi < 0),
                                    k_same=int(sum(np.sign(v) == np.sign(iv.point) for v in per.values())),
                                    **{f"d_{a}": round(v, 3) for a, v in per.items()}))
        Dm = pd.DataFrame(dec)
        Dm.to_csv(out / "maps.csv", index=False, float_format="%.4g")
        print("\n== cortical decomposition: change vs pre -> pre (boot_delta)")
        print(Dm.round(3).to_string(index=False))
    with open(out / "figdata.pkl", "wb") as fh:
        pickle.dump({"R": R, "F": F, "T": T, "V": V}, fh)
    plot(out, R, F)


def plot(out: Path, R: pd.DataFrame, F: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.figure_layout import save_png_svg
    from wfield_local.orofacial_cohort import EPOCH_LINE
    mark = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    D = per_session(R, K_MAIN)
    arms = [("frozen", "frozen"), ("cov", "(a) cov"), ("pro", "(b) Procrustes"), ("refit", "refit"),
            ("pro_shuf", "(b) shuffled"), ("cov_rot", "(a) rotation"), ("pro_rot", "(b) rotation")]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [3, 1.3, 1.6]})
    ax = axs[0]
    rng = np.random.default_rng(0)
    for j, ep in enumerate(EPOCHS):
        g = D[D.epoch == ep]
        for i, (a, _lab) in enumerate(arms):
            x = i + (j - 1.5) * 0.19
            for an, gg in g.groupby("animal"):
                v = gg[f"acc_{a}"].dropna()
                ax.scatter(x + rng.uniform(-.04, .04, len(v)), v, s=12, marker=mark.get(an, "o"),
                           facecolors="none", edgecolors=EPOCH_LINE[ep], lw=.7, alpha=.7)
            am = g.groupby("animal")[f"acc_{a}"].mean()
            ax.errorbar(x, am.mean(), yerr=am.std(ddof=1) / np.sqrt(len(am)) if len(am) > 1 else 0, fmt="o",
                        color=EPOCH_LINE[ep], ms=6, capsize=0, lw=1.6, label=ep if i == 0 else None)
    ax.axhline(1 / 6, color="0.6", lw=.8, ls=":")
    ax.set_xticks(range(len(arms)), [lab for _a, lab in arms], fontsize=9)
    ax.set_ylabel("spout accuracy, held-out half")
    ax.set_title(f"Lick window, k = {K_MAIN} latents: frozen / aligned / refit (size-matched training)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=4)
    ax = axs[1]
    T = pd.read_csv(out / "tests.csv")
    pre = D[D.epoch == "pre"].groupby("animal")
    for i, aln in enumerate(ALNS):
        eff = ((pre[f"acc_{aln}"].mean() - 1 / 6) / (pre.acc_frozen.mean() - 1 / 6)).round(2)
        ax.text(i, -0.95, "gate (pre->pre efficiency)\n" + " / ".join(f"{v:.2f}" for v in eff), ha="center",
                fontsize=7.5, color="0.3")
        for j, ep in enumerate(EPOCHS[1:]):
            r = T[(T.k == K_MAIN) & (T.epoch == ep) & (T.quantity == f"index_{aln}")]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (j - 1) * 0.22
            ax.errorbar(x, r.change, yerr=[[r.change - r.lo], [r.hi - r.change]], fmt="o", color=EPOCH_LINE[ep],
                        capsize=0, lw=1.6, ms=6, label=ep if i == 1 else None)
            for c, v in r.items():
                if c.startswith("d_") and np.isfinite(v):
                    ax.scatter(x + .06, np.clip(v, -1.1, 2.1), s=16, marker=mark.get(c[2:], "o"), facecolors="none",
                               edgecolors=EPOCH_LINE[ep], lw=.8)
    ax.axhline(0, color="0.6", lw=.8)
    ax.axhline(1, color="0.6", lw=.8, ls=":")
    for y in (0.33, 0.67):
        ax.axhline(y, color="0.8", lw=.6, ls="--")
    ax.set_ylim(-1.15, 2.15)
    ax.set_xticks(range(len(ALNS)), ["(a) covariance", "(b) Procrustes"])
    ax.set_ylabel("share of the stroke-added gap closed\nindex = d(aligned - frozen) / d(refit - frozen)")
    ax.set_title("Re-embedding index vs pre (bars 95 % CI)\n>= .67 re-embedded, <= .33 replaced", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax = axs[2]
    P = per_session(R, K_MAIN, "pair")
    for ep in EPOCHS:
        g = P[P.epoch == ep]
        if g.empty:
            continue
        b = g.groupby(pd.cut(g.lag.abs(), [0, 7, 14, 28, 60, 120]), observed=True)
        x = [iv.mid for iv in b.size().index]
        ax.plot(x, b.acc_pro.mean() - b.acc_frozen.mean(), "o-", color=EPOCH_LINE[ep], label=f"{ep} (b) - frozen")
        ax.plot(x, b.acc_refit.mean() - b.acc_frozen.mean(), "s--", color=EPOCH_LINE[ep], alpha=.6, ms=4)
    ax.axhline(0, color="0.6", lw=.8)
    ax.set_xlabel("|lag| target - reference pre session (days)")
    ax.set_ylabel("accuracy gain over frozen")
    ax.set_title("Pairwise references: solid = (b) aligned, dashed = refit", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_png_svg(fig, out / "aligned_decoder.png", dpi=170)
    print(f"wrote {out / 'aligned_decoder.png'}", flush=True)


# ---------------------------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N sessions per animal (timing)")
    ap.add_argument("--basis", choices=("lnmf", "svd"), default="lnmf")
    a = ap.parse_args(argv)
    if a.basis == "svd":
        os.environ["WF_SS_BASIS"], os.environ["WF_SS_ALIGN"] = "svd", "lick"
        import state_space_sim as S
        out = S.ROOT / "aligned_decoder_svd"
    else:
        out = setup_env()
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    if a.summarise:
        summarise(out)
        return 0
    import time
    t0 = time.time()
    cap = [c for c in (svd_inputs() if a.basis == "svd" else load_inputs(out)) if c.get("epoch") in EPOCHS]
    cap = [c for c in cap if a.animals is None or c["animal"] in a.animals]
    B = 8
    print(f"inputs: {len(cap)} sessions ({time.time() - t0:.0f} s); "
          f"{ {an: sum(c['animal'] == an for c in cap) for an in sorted({c['animal'] for c in cap})} }", flush=True)
    gf = out / "rois.npz"
    grams = {}
    saved = dict(np.load(gf, allow_pickle=True)) if gf.exists() else {}
    for an in sorted({c["animal"] for c in cap}):
        C = next(c for c in cap if c["animal"] == an)["X"].shape[1] // B
        key = f"{an}"
        if key in saved:
            grams[an] = saved[key].item()
        else:
            grams[an] = svd_grams() if a.basis == "svd" else roi_grams(an, C)
            saved[key] = np.array(grams[an], dtype=object)
    np.savez(gf, **saved)
    units = []
    for an in sorted({c["animal"] for c in cap}):
        mine = [c for c in cap if c["animal"] == an]
        pres = [c for c in mine if c["pre"]]
        units += [(T, pres, B, grams[an]) for T in (mine[: a.limit] if a.limit else mine)]
    got, fail = ak.fan_sessions(units, unit_job, jobs=min(a.jobs, len(units)), key=lambda u: u[0]["label"])
    for u, err in fail:
        print(f"  FAILED {u[0]['label']}: {err}", flush=True)
    R = pd.DataFrame([r for _u, rr in got for r in rr])
    R.to_csv(out / "rows.csv", index=False, float_format="%.5g")
    print(f"scored {len(units)} sessions in {time.time() - t0:.0f} s -> {out / 'rows.csv'}", flush=True)
    summarise(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
