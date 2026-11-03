"""PRE-LAUNCH AIM TRACKING (Priya 10-31): does cortex carry the RE-AIMED launch command BEFORE the tongue moves?

An ACROSS-SPOUT launch-angle decoder is trained on each animal's pooled August pre licks and FROZEN; in post sessions, at
the SAME TARGET (spout; ITI: previous trial's spout) the decoded angle either follows the deviated launch (cortex carries
the aim) or stays at the target (aim error set downstream). Tracking index TI = delta(decoded / b_sp) / delta(executed),
delta = epoch - August pre, b_sp = the pre across-target calibration slope (0 = reads the target, 1 = follows execution
as far as it follows angle across targets pre-stroke). Same decomposition as lick_enl_offset_explore B7 (pre-cue).

DESIGN (fixed before any number): docs/status/PRELAUNCH_AIM_TRACKING_PREREG_20261031.md -- read it first.

SIGNS: theta_L (launch, first 12 ms), theta_E (landing) in deg, + = CONTRALESIONAL (mouse right). left hemisphere =
ipsilesional = image left. Windows in s from TONGUE onset (jaw variant: from JAW onset).

Usage (bash env per the repo; outputs only under <v36>/imaging_encoding/prelaunch_aim_tracking/):
  python scripts/prelaunch_aim_tracking.py extract [--jobs 4] [--animals PS92]   per-session features (cached npz)
  python scripts/prelaunch_aim_tracking.py fit [--jobs 4] [--animals PS92] [--nperm 200]   decoders, nulls, frozen
  python scripts/prelaunch_aim_tracking.py summarise                              tracking, stats, figdata, figures
  python scripts/prelaunch_aim_tracking.py plot   (or any subcommand with --plot-only)   figures from figdata only
Env: PAT_OUT redirects all outputs (smoke test); WIDEFIELD_FROZEN_MODEL_DIR redirects the frozen store (smoke test).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

V36 = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36")
E = V36 / "imaging_encoding"
OUT = Path(os.environ.get("PAT_OUT") or (E / "prelaunch_aim_tracking"))
FEAT = E / "prelaunch_aim_tracking" / "features"          # feature cache is shared by the smoke test (deterministic)
FIGS = OUT / "figures"
FD = OUT / "figdata"
FITD = OUT / "fit"
S25 = E / "prelick_hemisphere_tuning"
WLC = E / "within_lick_correction"
LCOMP = E / "learned_compensation"
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
EDGES = np.round(np.arange(-0.80, 0.3001, 0.05), 3)
NB = len(EDGES) - 1                                           # 22 bins
CENT = np.round((EDGES[:-1] + EDGES[1:]) / 2, 3)
FINE = np.round(np.arange(-0.80 + 0.005, 0.30, 0.01), 4)      # 5 samples per bin
BASE = (-0.80, -0.50)
WINS = {"PRE": (-0.50, -0.05), "PRE10": (-0.50, -0.10), "POST": (0.05, 0.30)}
QUIET = 1.0
SAME_LICK = 0.05
CUE_EXCL = (-2.0, 0.75)
BLK = 8
ALPHAS = np.logspace(-1, 5, 13)
K_SVD = (10, 20, 40)
K0 = 20
MIN_PRE_SESS, MIN_PRE_LICKS, MIN_SESS, MIN_CELL = 3, 150, 10, 3
SEED = 0
SETS = ("cued", "jaw", "iti")
BASIS = {"PS92": "76d884873920", "PS93": "0141bb830372", "PS94": "a5e70d5ecd36", "PS95": "a9170190179c"}
FR_ENV = {"COMBINED_ENCODING_FR": "master_pre_v1", "COMBINED_ENCODING_FR_GROUPS": "master_v1:A"}


def _bins_of(w) -> np.ndarray:
    return (EDGES[:-1] >= w[0] - 1e-9) & (EDGES[1:] <= w[1] + 1e-9)


BB = _bins_of(BASE)


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _animals() -> list[str]:
    a = _arg("--animals")
    return [x for x in a.split(",")] if a else AN


def _b(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def labels() -> list[str]:
    """S25's imaging sessions (curated, PS92-95, licks_scored), pre restricted to AUGUST."""
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    S = pd.read_csv(S25 / "extract_summary.csv")
    cur = set(ak.curated_labels())
    out = []
    for lab in sorted(S.label):
        if lab[:4] not in AN or lab not in cur:
            continue
        ep = epochs.epoch_of(lab)
        if ep not in EPS or (ep == "pre" and lab[-4:-2] != "08"):
            continue
        out.append(lab)
    return out


# ================================================================================================ extraction
class Item(tuple):
    def __repr__(self):
        return str(self[0])


def _items(labs: list[str]) -> list:
    LS = pd.read_csv(WLC / "licks_scored.csv.gz", low_memory=False,
                     usecols=["sid", "label", "lick_uid", "trial_id", "lick_idx", "context", "pos", "onset_ok",
                              "theta_L", "theta_E"])
    LS = LS[LS.label.isin(labs)]
    IT = pd.read_csv(LCOMP / "iti_licks.csv.gz", low_memory=False,
                     usecols=["label", "lick_uid", "onset_ok", "theta_L", "theta_E", "daq_on_s", "gap", "phase",
                              "since_stop", "prev_pos"])
    IT = IT[IT.label.isin(labs)]
    DA = pd.read_csv(E / "iti_licks" / "iti_dock_audit.csv", usecols=["label", "onset", "dock01"])
    DA = DA[DA.label.isin(labs)]
    return [Item((lab, LS[LS.label == lab].reset_index(drop=True), IT[IT.label == lab].reset_index(drop=True),
                  DA[DA.label == lab].reset_index(drop=True))) for lab in labs]


_NMF = {}


def _nmf_op(an: str):
    """(operator pinv(A) U_joint (ncomp, R), footprint scale (ncomp,), U_joint (npix, R)) of the animal's joint basis."""
    if an not in _NMF:
        from wfield_local import joint_locanmf as jl
        from wfield_local.locanmf_crossanimal_dff import _footprint_scale
        b = jl.load(an, BASIS[an])
        op = np.asarray(b._operator, np.float64)
        Uj = np.nan_to_num(np.load(b.root / "U_joint.npy")).astype(np.float64)
        _NMF[an] = (op, _footprint_scale(np.asarray(b.A), b.ncomp), Uj, b.basis_id)
    return _NMF[an]


def _binned(R: np.ndarray, ft: np.ndarray, t: np.ndarray) -> np.ndarray:
    import iti_licks as il
    if not len(t):
        return np.zeros((0, NB, R.shape[1]), np.float32)
    X = il._sample(R, ft, np.asarray(t, float), FINE)
    return X.reshape(len(t), NB, 5, X.shape[-1]).mean(2).astype(np.float32)


def extract(item) -> dict:
    """One session -> FEAT/<label>.npz: per lick set (cued / jaw / iti) the 22 bins (-0.80..+0.30 s) of the Allen split
    features (/ session frame SD), the SVD temporal components (V-space; + the session's 8 x 8 block-mean U), the
    joint-LocaNMF components (/ frame SD), FaceRhythm factors at the bin centres, and per-lick metadata."""
    label, LS, IT, DA = item
    f = FEAT / f"{label}.npz"
    if f.exists() and "--redo" not in sys.argv:
        return {"label": label, "cached": True}
    t0 = time.time()
    for k, v in FR_ENV.items():
        os.environ.setdefault(k, v)
    import iti_licks as il
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    z = np.load(S25 / "sessions" / f"{label}.npz", allow_pickle=True)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    V = V.astype(np.float64)
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    V -= Q @ (Q.T @ V)
    V *= 1e3
    Ub = np.asarray(Ub, np.float64)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[c]].mean(0) for c in cols], 1)
    R = V @ Ra
    fnames = il.area_names(cols)
    sd = il.area_sd().loc[label, fnames].to_numpy(float)
    # joint LocaNMF components (V-space): C = V @ (scale * pinv(A) U_joint[mask]^T U)^T
    op, fsc, Uj, bid = _nmf_op(an)
    P = Uj[flat].T @ Ub                                            # (R, K)
    Wn = fsc[:, None] * (op @ P)                                   # (ncomp, K)
    Cn = V @ Wn.T
    nsd = Cn.std(0)
    nsd[nsd == 0] = 1
    VtV = V.T @ V
    vcap = float(np.sum((P.T @ P) * VtV) / max(np.sum((Ub.T @ Ub) * VtV), 1e-30))
    # SVD 8 x 8 blocks
    W = int(shape[1])
    rr, cc = flat // W, flat % W
    bidx = (rr // BLK) * (W // BLK + 1) + cc // BLK
    ub, inv = np.unique(bidx, return_inverse=True)
    Ublk = np.zeros((len(ub), Ub.shape[1]))
    np.add.at(Ublk, inv, Ub)
    cnt = np.bincount(inv, minlength=len(ub))
    Ublk /= cnt[:, None]
    # behaviour / licks
    B = il._behaviour(label)
    T = B.trials.sort_values("trial_id").set_index("trial_id")
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s"])
    on_all = A.daq_on_s.to_numpy(float)
    t_all = np.sort(np.concatenate([on_all, np.asarray(B.lick_on_s, float)]))
    t_all = t_all[np.isfinite(t_all)]
    ev = np.sort(np.concatenate([T.cue_s.to_numpy(float), T.strobe_s.to_numpy(float)]))
    ev = ev[np.isfinite(ev)]
    lo, hi = ft[0] - EDGES[0] + 0.02, ft[-1] - EDGES[-1] - 0.02

    def prev_gap(t):
        j = np.searchsorted(t_all, t - SAME_LICK)
        return t - np.where(j > 0, t_all[np.maximum(j - 1, 0)], -np.inf)
    # previous scored lick's theta_L (any context), by onset
    LS = LS.copy()
    LS["on"] = on_all[LS.lick_uid.to_numpy(int)]
    LS = LS[np.isfinite(LS.on)].sort_values("on").reset_index(drop=True)
    on_s, th_s = LS.on.to_numpy(float), LS.theta_L.to_numpy(float)

    def prev_theta(t):
        j = np.searchsorted(on_s, np.asarray(t, float) - SAME_LICK) - 1
        return np.where(j >= 0, th_s[np.maximum(j, 0)], np.nan)
    first = LS[(LS.context == "post_cue") & (LS.lick_idx == 0)].drop_duplicates("trial_id").set_index("trial_id")
    out = {"label": label, "animal": an, "epoch": epochs.epoch_of(label), "fnames": np.array(fnames),
           "blocks": ub.astype(np.int64), "block_n": cnt.astype(np.int64), "Ublk": Ublk.astype(np.float32),
           "nmf_basis": bid, "nmf_vcap": vcap}
    meta = {}
    # ---- CUED (S25's set) and JAW
    tid = z["first_trial_id"].astype(int)
    meta["cued"] = pd.DataFrame({"t": z["first_t"].astype(float), "target": z["first_pos"].astype(str),
                                 "theta_L": z["first_theta_L"].astype(float), "rt": z["first_rt"].astype(float),
                                 "trial_id": tid, "t_tongue": z["first_t"].astype(float)})
    jt = z["firstjaw_t"].astype(float)
    meta["jaw"] = pd.DataFrame({"t": jt - z["firstjaw_rt"].astype(float) + z["firstjaw_jaw_ms"].astype(float) / 1000,
                                "target": z["firstjaw_pos"].astype(str), "theta_L": z["firstjaw_theta_L"].astype(float),
                                "rt": z["firstjaw_rt"].astype(float), "trial_id": z["firstjaw_trial_id"].astype(int),
                                "t_tongue": jt})
    for nm in ("cued", "jaw"):
        D = meta[nm]
        D["theta_E"] = D.trial_id.map(first.theta_E).astype(float)
        D["prev_target"] = [str(T.pos.get(x - 1, "none")) for x in D.trial_id]
        D["prev_theta"] = prev_theta(D.t_tongue.to_numpy())
        D["gap"] = prev_gap(D.t_tongue.to_numpy())
        D["since_stop"], D["dock01"] = np.nan, False
        meta[nm] = D[(D.t > lo) & (D.t < hi)].reset_index(drop=True)
    # ---- ITI: docked, bout-first (1 s), onset_ok, theta_L, canonical previous spout, no cue / strobe nearby
    I = IT[(IT.phase == "docked") & _b(IT.onset_ok) & np.isfinite(IT.theta_L) & IT.prev_pos.isin(POS)
           & (IT.gap >= QUIET)].copy()
    t_on = on_all[I.lick_uid.to_numpy(int)]
    clock = float(np.nanmax(np.abs(t_on - I.daq_on_s.to_numpy(float)))) if len(I) else np.nan
    I["t"] = t_on
    I["gap_s25"] = prev_gap(t_on)
    j = np.searchsorted(ev, t_on + CUE_EXCL[0])
    near = (j < len(ev)) & (ev[np.minimum(j, len(ev) - 1)] <= t_on + CUE_EXCL[1])
    I = I[(I.gap_s25 >= QUIET) & ~near & (I.t > lo) & (I.t < hi)].copy()
    don = DA.onset.to_numpy(float)
    if len(don) and len(I):
        o = np.argsort(don)
        don_s, dfl = don[o], DA.dock01.to_numpy(bool)[o]
        k = np.clip(np.searchsorted(don_s, I.t.to_numpy()), 1, max(len(don_s) - 1, 1))
        kk = np.where(np.abs(don_s[np.minimum(k, len(don_s) - 1)] - I.t.to_numpy()) <= np.abs(don_s[k - 1] - I.t.to_numpy()),
                      np.minimum(k, len(don_s) - 1), k - 1)
        I["dock01"] = (np.abs(don_s[kk] - I.t.to_numpy()) < 0.002) & dfl[kk]
    else:
        I["dock01"] = False
    meta["iti"] = pd.DataFrame({"t": I.t.to_numpy(float), "target": I.prev_pos.astype(str).to_numpy(),
                                "theta_L": I.theta_L.to_numpy(float), "theta_E": I.theta_E.to_numpy(float),
                                "rt": np.nan, "trial_id": -1, "t_tongue": I.t.to_numpy(float),
                                "prev_target": I.prev_pos.astype(str).to_numpy(), "prev_theta": prev_theta(I.t.to_numpy()),
                                "gap": I.gap_s25.to_numpy(float), "since_stop": I.since_stop.to_numpy(float),
                                "dock01": I.dock01.to_numpy(bool)})
    summ = {"label": label, "animal": an, "epoch": out["epoch"], "nmf_vcap": vcap, "iti_clock_maxdiff_s": clock,
            "n_blocks": len(ub), "K_svd": V.shape[1]}
    for nm in SETS:
        D = meta[nm]
        t = D.t.to_numpy(float)
        Xa = _binned(R, ft, t)
        out[f"{nm}_allen"] = (il.area_features(Xa.astype(np.float64), cols, label) / sd).astype(np.float32)
        out[f"{nm}_V"] = _binned(V, ft, t)
        out[f"{nm}_nmf"] = (_binned(Cn, ft, t) / nsd).astype(np.float32)
        try:
            fr = il.fr_unblanked(an, date, (t[:, None] + CENT[None, :]).ravel()) if len(t) else np.zeros((0, 5))
            out[f"{nm}_fr"] = fr.reshape(len(t), NB, -1).astype(np.float32)
        except Exception as ex:                                      # noqa: BLE001  (no FaceRhythm projection)
            summ[f"fr_err_{nm}"] = f"{type(ex).__name__}: {str(ex)[:80]}"
            out[f"{nm}_fr"] = np.full((len(t), NB, 5), np.nan, np.float32)
        for c in D.columns:
            v = D[c].to_numpy()
            out[f"{nm}_m_{c}"] = v.astype(str) if v.dtype == object else v
        summ[f"n_{nm}"] = len(D)
    summ["fr_ok_cued"] = float(np.isfinite(out["cued_fr"]).all((1, 2)).mean()) if len(meta["cued"]) else np.nan
    FEAT.mkdir(parents=True, exist_ok=True)
    tmp = FEAT / f"{label}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(f)
    return {**summ, "s": round(time.time() - t0, 1)}


def cmd_extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(FEAT / "x.npz")
    FEAT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    labs = [x for x in labels() if x[:4] in _animals()]
    print(f"extract: {len(labs)} sessions, {jobs} workers", flush=True)
    t0 = time.time()
    res, fails = ak.fan_sessions(_items(labs), extract, jobs=jobs, key=lambda it: it[0], label="extract")
    for it, err in fails:
        print(f"  !! {it}: {err}", flush=True)
    S = pd.DataFrame([v for _k, v in res])
    f = OUT / "extract_summary.csv"
    if f.exists():
        old = pd.read_csv(f)
        S = pd.concat([old[~old.label.isin(S.label)], S], ignore_index=True)
    S.sort_values("label").to_csv(f, index=False)
    print(f"extract done {(time.time() - t0) / 60:.1f} min, {len(fails)} failed", flush=True)
    if fails:
        raise SystemExit("extraction failures")


# ================================================================================================ decoders
def _prep_fit(X: np.ndarray, kind: str, K: int = 0) -> dict:
    mu = X.mean(0)
    if kind == "std":
        sd = X.std(0)
        sd[sd == 0] = 1
        return {"kind": "std", "mu": mu, "sd": sd}
    Xc = X - mu
    # PCA via the n x n Gram matrix (n << p)
    G = Xc @ Xc.T
    w, U = np.linalg.eigh(G)
    o = np.argsort(w)[::-1][:K]
    comps = (U[:, o].T @ Xc) / np.sqrt(np.maximum(w[o], 1e-12))[:, None]
    S = Xc @ comps.T
    ssd = S.std(0)
    ssd[ssd == 0] = 1
    return {"kind": "pca", "mu": mu, "comps": comps, "smu": S.mean(0), "ssd": ssd}


def _prep_apply(p: dict, X: np.ndarray, K: int | None = None) -> np.ndarray:
    if p["kind"] == "std":
        return (X - p["mu"]) / p["sd"]
    C = p["comps"] if K is None else p["comps"][:K]
    smu, ssd = (p["smu"], p["ssd"]) if K is None else (p["smu"][:K], p["ssd"][:K])
    return ((X - p["mu"]) @ C.T - smu) / ssd


def _ridge_all(Ztr, Ytr, Zte):
    """Held-out predictions for every alpha and column: (nalpha, nte, m); plus the SVD pieces for the weights."""
    zm, ym = Ztr.mean(0), Ytr.mean(0)
    U, s, Vt = np.linalg.svd(Ztr - zm, full_matrices=False)
    H = U.T @ (Ytr - ym)                                          # (r, m)
    G = (Zte - zm) @ Vt.T                                         # (nte, r)
    d = s[None, :] / (s[None, :] ** 2 + ALPHAS[:, None])          # (nalpha, r)
    P = np.stack([G @ (da[:, None] * H) for da in d]) + ym
    return P, (zm, ym, Vt, s, H)


def _coef(pieces, ai: int, col: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    zm, ym, Vt, s, H = pieces
    c = Vt.T @ ((s / (s ** 2 + ALPHAS[ai])) * H[:, col])
    return c, float(ym[col]), zm


def _colr(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """corr per alpha x column: P (nalpha, n, m), Y (n, m) -> (nalpha, m)."""
    Pc = P - P.mean(1, keepdims=True)
    Yc = Y - Y.mean(0)
    num = np.einsum("anm,nm->am", Pc, Yc)
    den = np.sqrt((Pc ** 2).sum(1) * (Yc ** 2).sum(0)[None])
    return np.where(den > 0, num / np.where(den > 0, den, 1), -np.inf)


def _raw_model(prep: dict, coef: np.ndarray, ym: float, zm: np.ndarray, K: int | None, alpha: float) -> dict:
    """Linear model in raw feature space: pred = X @ w + b."""
    if prep["kind"] == "std":
        w = coef / prep["sd"]
        b = ym - (prep["mu"] / prep["sd"]) @ coef - zm @ coef
    else:
        Kk = len(coef)
        C, smu, ssd = prep["comps"][:Kk], prep["smu"][:Kk], prep["ssd"][:Kk]
        w = C.T @ (coef / ssd)
        b = ym - prep["mu"] @ w - (smu / ssd) @ coef - zm @ coef
    return {"w": w.astype(np.float64), "b": float(b), "alpha": float(alpha)}


def _fit_block(X, Y, groups, kind, K):
    """Inner leave-one-group-out alpha per column, refit on all of X: (alpha index per column, model for column 0, prep)."""
    prep = _prep_fit(X, kind, K)
    Z = _prep_apply(prep, X, K if kind == "pca" else None)
    gs = np.unique(groups)
    Pin = np.full((len(ALPHAS),) + Y.shape, np.nan)
    for g in gs:
        te = groups == g
        Pin[:, te, :], _ = _ridge_all(Z[~te], Y[~te], Z[te])
    ai = np.argmax(_colr(Pin, Y), 0)
    _, pieces = _ridge_all(Z, Y, Z[:1])
    c, ym, zm = _coef(pieces, int(ai[0]), 0)
    return ai, _raw_model(prep, c, ym, zm, K if kind == "pca" else None, ALPHAS[ai[0]]), prep, Z, pieces


def loso_multi(X, Y, sess, kind="std", K=0) -> dict:
    """Leave-one-session-out held-out predictions for every column of Y (observed + permutations), inner leave-one-
    session-out alpha per column; preprocessing (scaler / PCA) fitted on the outer training sessions and reused inside
    the inner loop (label-free). Returns preds (n, m), alpha index (nfold, m), per-fold raw models of column 0."""
    labs = list(np.unique(sess))
    pred = np.full(Y.shape, np.nan)
    ais, models = [], {}
    for s in labs:
        te = sess == s
        tr = ~te
        ai, _m0, prep, Ztr, pieces = _fit_block(X[tr], Y[tr], sess[tr], kind, K)
        Zte = _prep_apply(prep, X[te], K if kind == "pca" else None)
        zm, ym, Vt, sv, H = pieces
        G = (Zte - zm) @ Vt.T
        for a in np.unique(ai):
            cols = ai == a
            d = sv / (sv ** 2 + ALPHAS[a])
            pred[np.ix_(te, cols)] = G @ (d[:, None] * H[:, cols]) + ym[cols]
        ais.append(ai)
        c, ymc, zmc = _coef(pieces, int(ai[0]), 0)
        models[s] = _raw_model(prep, c, ymc, zmc, K if kind == "pca" else None, ALPHAS[ai[0]])
    return {"pred": pred, "ai": np.array(ais), "models": models}


def _r(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _cent(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    return v - pd.DataFrame(v).groupby(g).transform("mean").to_numpy() if v.ndim == 2 else \
        v - pd.Series(v).groupby(g).transform("mean").to_numpy()


def _perm_within(y, g, rng):
    y = y.copy()
    for q in np.unique(g):
        i = np.flatnonzero(g == q)
        y[i] = y[rng.permutation(i)]
    return y


def _digest(*arrs) -> str:
    import hashlib
    h = hashlib.sha1()
    for a in arrs:
        a = np.asarray(a)
        h.update("|".join(a.astype(str)).encode() if a.dtype.kind in "OUS" else
                 np.ascontiguousarray(a.astype(np.float64)).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ per-animal data
def load_animal(an: str) -> dict:
    labs = [x for x in labels() if x[:4] == an]
    D = {s: {"meta": [], "allen": [], "V": [], "nmf": [], "fr": [], "sess": []} for s in SETS}
    blocks = {}
    for lab in labs:
        f = FEAT / f"{lab}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{lab}: not extracted")
        z = np.load(f, allow_pickle=False)
        blocks[lab] = (z["blocks"], z["block_n"], z["Ublk"])
        for s in SETS:
            n = len(z[f"{s}_m_t"])
            m = pd.DataFrame({c[len(s) + 3:]: z[c] for c in z.files if c.startswith(f"{s}_m_")})
            m.insert(0, "label", lab)
            m.insert(1, "animal", an)
            m.insert(2, "epoch", str(z["epoch"]))
            D[s]["meta"].append(m)
            for k in ("allen", "V", "nmf", "fr"):
                D[s][k].append(z[f"{s}_{k}"])
            D[s]["sess"].append(np.full(n, lab))
    fnames = list(np.load(FEAT / f"{labs[0]}.npz")["fnames"])
    # common 8 x 8 blocks: present in every session with >= half of the 64 pixels
    common = None
    for _lab, (b, n, _u) in blocks.items():
        ok = set(b[n >= BLK * BLK // 2].tolist())
        common = ok if common is None else common & ok
    common = np.array(sorted(common), np.int64)
    out = {"labels": labs, "fnames": fnames, "common_blocks": common}
    for s in SETS:
        d = D[s]
        meta = pd.concat(d["meta"], ignore_index=True)
        out[s] = {"meta": meta, "allen": np.concatenate(d["allen"]).astype(np.float64),
                  "nmf": np.concatenate(d["nmf"]).astype(np.float64), "fr": np.concatenate(d["fr"]).astype(np.float64),
                  "Vlist": d["V"], "Ulist": [blocks[lab][2][np.searchsorted(blocks[lab][0], common)] for lab in labs],
                  "sess": np.concatenate(d["sess"])}
    return out


def _window(X: np.ndarray, w: str, base: bool = True) -> np.ndarray:
    """(n, 22, p) -> (n, nbins_w * p) flattened, minus the BASE mean (if base)."""
    Xw = X[:, _bins_of(WINS[w])]
    if base:
        Xw = Xw - X[:, BB].mean(1, keepdims=True)
    return Xw.reshape(len(X), -1)


def _svd_feats(S: dict, w: str) -> np.ndarray:
    rows = []
    for Vb, Ub in zip(S["Vlist"], S["Ulist"]):
        Vb = Vb.astype(np.float64)
        Vw = Vb[:, _bins_of(WINS[w])] - Vb[:, BB].mean(1, keepdims=True)       # (n, nb, K)
        rows.append((Vw @ Ub.astype(np.float64).T).astype(np.float32).reshape(len(Vb), Vw.shape[1] * len(Ub)))
    return np.concatenate(rows)


def _bin_feats(X: np.ndarray, b: int) -> np.ndarray:
    """Single-bin decoder features: bin b minus BASE mean; bins inside BASE keep raw levels (prereg 6)."""
    return X[:, b] if BB[b] else X[:, b] - X[:, BB].mean(1)


#: specs made by run_decoder in this process, keyed (animal, align, source) -- for the cued_PRE lookup
_SPECS: dict = {}


# ------------------------------------------------------------------------------------------------ one decoder
def run_decoder(an, arm, name, X, meta, kind="std", K=0, nperm=0, spec_src="", n_feat=None, basis_id=None,
                train_mask=None, log=print) -> tuple[np.ndarray, dict, dict]:
    """LOSO (pre) + frozen all-pre model -> predictions for every lick; gate row (if nperm); payload."""
    from wfield_local import frozen_models as fm
    tm = (meta.epoch == "pre").to_numpy() if train_mask is None else train_mask
    ok = np.isfinite(X).all(1) & np.isfinite(meta.theta_L.to_numpy(float))
    pre = tm & ok
    labs_pre = sorted(set(meta.label[pre]))
    row = {"animal": an, "arm": arm, "decoder": name, "n_pre": int(pre.sum()), "n_pre_sess": len(labs_pre)}
    pred = np.full(len(meta), np.nan)
    if len(labs_pre) < MIN_PRE_SESS or pre.sum() < MIN_PRE_LICKS:
        row["skipped"] = "too few pre"
        return pred, row, {}
    Xp, y = X[pre], meta.theta_L.to_numpy(float)[pre]
    sess = meta.label.to_numpy(str)[pre]
    cell = np.char.add(np.char.add(sess, "|"), meta.target.to_numpy(str)[pre])
    dig = _digest(Xp, y, cell)
    spec = fm.make_spec(an, "prelaunch_aim", align=f"{arm}_{name.split(':')[-1]}", source=spec_src,
                        train_labels=labs_pre, n_features=int(n_feat or X.shape[1]), basis_id=basis_id,
                        train_data={"xy_sha1": dig})  # training-data fingerprint (FREEZE_VERSION 2, rebuild 11-03)
    _SPECS[(an, spec["align"], spec_src)] = spec        # the cued_PRE model is looked up again below (find)
    obs = {}

    def fit():
        L = loso_multi(Xp, y[:, None], sess, kind, K)
        _ai, full, _p, _z, _pc = _fit_block(Xp, y[:, None], sess, kind, K)
        obs["L"] = L
        return {"full": full, "loso": L["models"], "xy_sha1": dig}
    P, status = fm.load_or_fit(spec, fit, meta={"script": "prelaunch_aim_tracking.py", "decoder": name, "arm": arm,
                                                "n_trials": int(len(y))}, log=log)
    if P.get("xy_sha1") != dig:
        log(f"[frozen] !! {an} {arm} {name}: stored model trained on DIFFERENT data (same pre sessions); stored used")
    row["frozen"] = status
    # predictions: pre = LOSO sibling of its session, others = full
    sall = meta.label.to_numpy(str)
    for lab in np.unique(sall):
        i = np.flatnonzero((sall == lab) & ok)
        if not len(i):
            continue
        m = P["loso"][lab] if (lab in P["loso"] and tm[i].all()) else P["full"]
        pred[i] = X[i] @ m["w"] + m["b"]
    row["alpha_full"] = P["full"]["alpha"]
    row["alpha_edge_frac"] = float(np.mean([m["alpha"] in (ALPHAS[0], ALPHAS[-1]) for m in P["loso"].values()]))
    pp = pred[pre]
    row["r_pool"] = _r(pp, y)
    row["r_sess_mean"] = float(np.nanmean([_r(pp[sess == s], y[sess == s]) for s in labs_pre]))
    row["partial_r"] = _r(_cent(pp, cell), _cent(y, cell))
    if nperm:
        import zlib
        rng = np.random.default_rng([AN.index(an), SETS.index(arm), zlib.crc32(name.encode()) % 99991])
        Ys = [y] + [_perm_within(y, sess, rng) for _ in range(nperm)] + [_perm_within(y, cell, rng) for _ in range(nperm)]
        L = loso_multi(Xp, np.column_stack(Ys), sess, kind, K)
        Pr = L["pred"]
        r_obs_refit = _r(Pr[:, 0], y)
        rs = np.array([_r(Pr[:, j], Ys[j]) for j in range(1, nperm + 1)])
        pr = np.array([_r(_cent(Pr[:, j], cell), _cent(Ys[j], cell)) for j in range(nperm + 1, 2 * nperm + 1)])
        row.update(r_pool_permpath=r_obs_refit, null95=float(np.nanpercentile(rs, 95)),
                   p_gate=float((1 + np.sum(rs >= row["r_pool"])) / (1 + nperm)),
                   partial_null95=float(np.nanpercentile(pr, 95)),
                   p_partial=float((1 + np.sum(pr >= row["partial_r"])) / (1 + nperm)))
    return pred, row, P


def fit_animal(an: str) -> dict:
    """Every decoder of one animal (module-level for spawn). Writes FITD/<an>_{preds_<set>.csv.gz, gate.csv, ...}."""
    for k, v in FR_ENV.items():
        os.environ.setdefault(k, v)
    t0 = time.time()
    nperm = int(_arg("--nperm", 200))
    logf = open(FITD / f"{an}_log.txt", "w")

    def log(m):
        print(f"[{an}] {m}", flush=True)
        logf.write(m + "\n")
        logf.flush()
    from wfield_local import frozen_models as fm
    log(f"frozen local_dir = {fm.local_dir()} exists={fm.local_dir().exists()}")
    D = load_animal(an)
    fnames = D["fnames"]
    left = np.array([n.endswith("_left") for n in fnames])
    log(f"{len(D['labels'])} sessions; common blocks {len(D['common_blocks'])}; load {time.time() - t0:.0f} s")
    gate, refit = [], []
    preds = {}
    bid = BASIS[an]
    for arm in ("cued", "iti", "jaw"):
        S = D[arm]
        meta = S["meta"].copy()
        Pd = {}

        def go(name, X, kind="std", K=0, nperm_=0, src="", basis_id=None, mask=None, arm=arm, meta=meta, Pd=Pd):
            p, row, payload = run_decoder(an, arm, name, X, meta, kind, K, nperm_, src, basis_id=basis_id,
                                          train_mask=mask, log=log)
            Pd[name] = p
            gate.append(row)
            log(f"{arm} {name}: r_pool {row.get('r_pool', np.nan):.3f} p {row.get('p_gate', np.nan)} "
                f"partial {row.get('partial_r', np.nan):.3f} ({row.get('frozen', row.get('skipped'))}) "
                f"{time.time() - t0:.0f} s")
            return payload
        Al, Nm, Fr = S["allen"], S["nmf"], S["fr"]
        wins = ("PRE", "POST", "PRE10") if arm != "jaw" else ("PRE",)
        for w in wins:
            nul = nperm if w in ("PRE", "POST") else 0
            pay = go(f"allen:{w}", _window(Al, w), nperm_=nul, src="allen_split_r8")
            if w == "PRE" and pay:                                          # per-bin decomposition of the primary
                Xw = Al[:, _bins_of(WINS["PRE"])] - Al[:, BB].mean(1, keepdims=True)
                nb = Xw.shape[1]
                sall = meta.label.to_numpy(str)
                C = np.full((len(meta), nb + 1), np.nan)
                for lab in np.unique(sall):
                    i = np.flatnonzero(sall == lab)
                    m = pay["loso"][lab] if (lab in pay["loso"] and (meta.epoch.to_numpy()[i] == "pre").all()) else pay["full"]
                    Wb = m["w"].reshape(nb, -1)
                    C[i, :nb] = np.einsum("nbp,bp->nb", Xw[i], Wb)
                    C[i, nb] = m["b"]
                for b_ in range(nb):
                    meta[f"c_bin{b_}"] = C[:, b_]
                meta["c_int"] = C[:, nb]
            sv = _svd_feats(S, w)
            go(f"svd{K0}:{w}", sv, "pca", K0, nul, src=f"svd_blk{BLK}_K{K0}")
            if w == "PRE":
                for Kx in K_SVD:
                    if Kx != K0:
                        go(f"svd{Kx}:{w}", sv, "pca", Kx, 0, src=f"svd_blk{BLK}_K{Kx}")
            del sv
            go(f"nmf:{w}", _window(Nm, w), nperm_=nul, src="joint", basis_id=bid)
        if arm != "jaw":
            go("allen:PREnb", _window(Al, "PRE", base=False), src="allen_split_r8_nobase")
            for h, msk in (("left", left), ("right", ~left)):
                go(f"allen_{h}:PRE", _window(Al[:, :, msk], "PRE"), src=f"allen_split_r8_{h}")
            # face: face-only decoder; neural residualised on face (OLS fitted on all pre licks with FaceRhythm)
            Fw = _window(Fr, "PRE")
            okf = np.isfinite(Fw).all(1)
            Xf = np.where(okf[:, None], Fw, np.nan)
            go("face:PRE", Xf, src="fr_master_v1A")
            Xn = _window(Al, "PRE")
            pre_f = okf & (meta.epoch == "pre").to_numpy()
            Xr = np.full_like(Xn, np.nan)
            if pre_f.sum() > 100:
                Dm = np.column_stack([np.ones(pre_f.sum()), Fw[pre_f]])
                Bc = np.linalg.lstsq(Dm, Xn[pre_f], rcond=None)[0]
                Xr[okf] = Xn[okf] - np.column_stack([np.ones(okf.sum()), Fw[okf]]) @ Bc
            go("allen_faceres:PRE", Xr, src="allen_split_r8_faceres")
        for b_ in range(NB):
            go(f"bin{b_:02d}", _bin_feats(Al, b_), src="allen_split_r8_singlebin")
        # within-session refit (Allen PRE): 5 folds stratified by target, inner leave-one-fold-out alpha
        if arm != "jaw":
            Xw = _window(Al, "PRE")
            y = meta.theta_L.to_numpy(float)
            rng = np.random.default_rng([AN.index(an), SETS.index(arm), 7])
            for lab in sorted(meta.label.unique()):
                i = np.flatnonzero(meta.label.to_numpy() == lab)
                if len(i) < 40:
                    continue
                tg = meta.target.to_numpy(str)[i]
                fo = np.empty(len(i), int)
                for v in np.unique(tg):
                    ix = np.flatnonzero(tg == v)
                    fo[rng.permutation(ix)] = np.arange(len(ix)) % 5
                L = loso_multi(Xw[i], y[i][:, None], fo.astype(str), "std")
                refit.append({"animal": an, "arm": arm, "label": lab, "epoch": meta.epoch.iloc[i[0]], "n": len(i),
                              "r_refit": _r(L["pred"][:, 0], y[i])})
        for k, v in Pd.items():
            meta[f"p_{k}"] = v
        preds[arm] = meta
    # cued all-pre Allen PRE decoder applied to ITI licks
    from wfield_local import frozen_models as fm2
    Mc = preds["cued"]
    labs_pre = sorted(set(Mc.label[Mc.epoch == "pre"]))
    # the spec run_decoder made for the cued PRE decoder in THIS run (it carries the training-data fingerprint,
    # which cannot be rebuilt here); none -> the ITI cross-application is skipped, loudly (rebuild 11-03)
    spec = _SPECS.get((an, "cued_PRE", "allen_split_r8"))
    if spec is None:
        print(f"  !! {an}: no cued_PRE decoder spec in this run ({len(labs_pre)} pre) -- p_cuedx:PRE not written",
              flush=True)
    d, _o = fm2.find(spec) if spec is not None else (None, None)
    if d is not None:
        Pc = fm2._read(d)
        preds["iti"]["p_cuedx:PRE"] = _window(D["iti"]["allen"], "PRE") @ Pc["full"]["w"] + Pc["full"]["b"]
    for arm, M in preds.items():
        M.to_csv(FITD / f"{an}_preds_{arm}.csv.gz", index=False, float_format="%.6g")
    pd.DataFrame(gate).to_csv(FITD / f"{an}_gate.csv", index=False)
    pd.DataFrame(refit).to_csv(FITD / f"{an}_refit.csv", index=False)
    log(f"done {(time.time() - t0) / 60:.1f} min")
    logf.close()
    return {"animal": an, "min": round((time.time() - t0) / 60, 1)}


def cmd_fit(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import frozen_models as fm
    from wfield_local import writeguard
    writeguard.assert_writable(FITD / "x.csv")
    FITD.mkdir(parents=True, exist_ok=True)
    d = fm.local_dir()
    kinds = sorted({p.name.split("_")[0] for a in AN if (d / a).exists() for p in (d / a).iterdir()}) if d.exists() else []
    print(f"frozen local_dir = {d}  exists={d.exists()}  kinds present={kinds}", flush=True)
    if "--smoke" in sys.argv:
        if not os.environ.get("WIDEFIELD_FROZEN_MODEL_DIR") or not os.environ.get("PAT_OUT"):
            raise SystemExit("--smoke needs WIDEFIELD_FROZEN_MODEL_DIR and PAT_OUT (scratch store / outputs)")
        print("SMOKE TEST: scratch frozen store + scratch outputs", flush=True)
    elif not d.exists() or not kinds:
        raise SystemExit("frozen store absent or empty -- stop (prereg 3)")
    ans = _animals()
    res, fails = ak.fan_sessions(ans, fit_animal, jobs=min(jobs, len(ans)), label="animal")
    for it, err in fails:
        print(f"  !! {it}: {err}", flush=True)
    print(res, flush=True)
    if fails:
        raise SystemExit("fit failures")


# ================================================================================================ tracking
def calib(D: pd.DataFrame, pcol: str, tcol: str, key: str, rng=None) -> dict:
    """Pre calibration of one animal: b_sp (cell means), b_all (licks), SD ratio, b_sp bootstrap over pre sessions."""
    pre = D[D.epoch == "pre"]
    c = pre.groupby(["label", key]).agg(p=(pcol, "mean"), t=(tcol, "mean"), n=(pcol, "size"))
    c = c[c.n >= MIN_CELL].reset_index()
    out = {"n_cells": len(c)}
    if len(c) < 4:
        return {**out, "b_sp": np.nan, "b_all": np.nan, "b_lo": np.nan, "b_hi": np.nan, "sd_ratio": np.nan}
    out["b_sp"] = float(np.polyfit(c.t, c.p, 1)[0])
    out["b_all"] = float(np.polyfit(pre[tcol], pre[pcol], 1)[0])
    out["sd_ratio"] = float(pre[pcol].std() / pre[tcol].std())
    rng = rng or np.random.default_rng(SEED)
    labs = c.label.unique()
    bs = []
    for _ in range(200):
        pick = rng.choice(labs, len(labs), replace=True)
        cc = pd.concat([c[c.label == x] for x in pick])
        if cc.t.std() > 0:
            bs.append(np.polyfit(cc.t, cc.p, 1)[0])
    out["b_lo"], out["b_hi"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan, np.nan)
    return out


def track_licks(D: pd.DataFrame, pcol: str, tcol: str = "theta_L", key: str = "target", cal_from: dict | None = None
                ) -> tuple[pd.DataFrame, dict]:
    """Per lick d_dec (raw), d_dec* (/ b_sp), d_exe vs the animal's pre means at the same key cell (equal pre-session
    weight; a pre lick's own session left out). Returns (licks with the d columns, {animal: calibration})."""
    D = D.dropna(subset=[pcol, tcol]).copy()
    D["d_dec"], D["d_decs"], D["d_exe"] = np.nan, np.nan, np.nan
    cal = {}
    for a, g in D.groupby("animal"):
        # a lick SUBSET (far contra, RT tercile, ITI filters) keeps the calibration of the decoder on the arm's FULL lick
        # set (cal_from): a slope across the targets of a one-target subset is undefined
        c = dict(cal_from[a]) if cal_from is not None and a in cal_from else calib(g, pcol, tcol, key)
        cal[a] = c
        pre = g[g.epoch == "pre"]
        cm = pre.groupby(["label", key]).agg(p=(pcol, "mean"), t=(tcol, "mean"), n=(pcol, "size")).reset_index()
        cm = cm[cm.n >= MIN_CELL]
        ref_all = cm.groupby(key)[["p", "t"]].mean()
        rp = g[key].map(ref_all.p).to_numpy(float)
        rt = g[key].map(ref_all.t).to_numpy(float)
        for lab in pre.label.unique():
            o = cm[cm.label != lab].groupby(key)[["p", "t"]].mean()
            i = (g.label == lab).to_numpy()
            rp[i] = g.loc[i, key].map(o.p).to_numpy(float)
            rt[i] = g.loc[i, key].map(o.t).to_numpy(float)
        D.loc[g.index, "d_dec"] = g[pcol].to_numpy() - rp
        D.loc[g.index, "d_exe"] = g[tcol].to_numpy() - rt
        b = c["b_sp"]
        good = np.isfinite(b) and b > 0 and np.isfinite(c["b_lo"]) and c["b_lo"] > 0
        c["calibrated"] = bool(good)
        D.loc[g.index, "d_decs"] = (g[pcol].to_numpy() - rp) / b if (np.isfinite(b) and b > 0) else np.nan
    return D, cal


def sessions_of(L: pd.DataFrame) -> pd.DataFrame:
    S = L.dropna(subset=["d_dec", "d_exe"]).groupby(["label", "animal", "epoch"]).agg(
        n=("d_exe", "size"), d_dec=("d_dec", "mean"), d_decs=("d_decs", "mean"), d_exe=("d_exe", "mean")).reset_index()
    return S[S.n >= MIN_SESS]


def tracking_stats(S: pd.DataFrame, cal: dict, tag: dict) -> tuple[list, list]:
    """Per-animal table + pooled TI per post epoch (boot_delta_columns on [d_dec*, d_exe, d_dec])."""
    from wfield_local import analysis_kit as ak
    animals = [a for a in AN if cal.get(a, {}).get("calibrated")]
    per, pooled = [], []
    for (a, ep), g in S.groupby(["animal", "epoch"]):
        per.append({**tag, "animal": a, "epoch": ep, "n_sessions": len(g), "n_licks": int(g.n.sum()),
                    "d_decs": g.d_decs.mean(), "d_dec": g.d_dec.mean(), "d_exe": g.d_exe.mean(),
                    "b_sp": cal.get(a, {}).get("b_sp"), "calibrated": cal.get(a, {}).get("calibrated")})
    P = pd.DataFrame(per)
    for ep in EPS[1:]:
        rng = np.random.default_rng(SEED)
        pre = {a: S[(S.animal == a) & (S.epoch == "pre")][["d_decs", "d_exe", "d_dec"]].to_numpy(float) for a in animals}
        post = {a: S[(S.animal == a) & (S.epoch == ep)][["d_decs", "d_exe", "d_dec"]].to_numpy(float) for a in animals}
        pre = {a: v for a, v in pre.items() if len(v)}
        post = {a: v for a, v in post.items() if len(v)}
        row = {**tag, "epoch": ep, "n_animals": len(set(pre) & set(post)),
               "n_uncalibrated": len([a for a in AN if a in cal and not cal[a].get("calibrated")])}
        r = ak.boot_delta_columns(post, pre, rng) if set(pre) & set(post) else None
        if r is None:
            pooled.append(row)
            continue
        obs, lo, hi, dr, n = r
        ti_d = dr[:, 0] / np.where(dr[:, 1] == 0, np.nan, dr[:, 1])
        per_a = {a: (post[a][:, 0].mean() - pre[a][:, 0].mean(), post[a][:, 1].mean() - pre[a][:, 1].mean())
                 for a in sorted(set(pre) & set(post))}
        sgn = np.sign(obs[0])
        est = bool((lo[1] > 0 or hi[1] < 0) and abs(obs[1]) >= 3)
        row.update(d_decs=obs[0], d_decs_lo=lo[0], d_decs_hi=hi[0], d_exe=obs[1], d_exe_lo=lo[1], d_exe_hi=hi[1],
                   d_dec_raw=obs[2], d_dec_raw_lo=lo[2], d_dec_raw_hi=hi[2],
                   TI=obs[0] / obs[1] if obs[1] != 0 else np.nan,
                   TI_lo=float(np.nanpercentile(ti_d, 2.5)), TI_hi=float(np.nanpercentile(ti_d, 97.5)),
                   p_num=ak.boot_p(dr[:, 0]), p_full=ak.boot_p(dr[:, 0] - dr[:, 1]),
                   frac_den_small=float(np.mean(np.abs(dr[:, 1]) < 1)), estimable=est,
                   n_same_sign=int(sum(np.sign(v[0]) == sgn for v in per_a.values())),
                   **{f"TI_{a}": (v[0] / v[1] if v[1] != 0 else np.nan) for a, v in per_a.items()},
                   **{f"dnum_{a}": v[0] for a, v in per_a.items()}, **{f"dexe_{a}": v[1] for a, v in per_a.items()})
        pooled.append(row)
    return P.to_dict("records"), pooled


def _load_preds(arm: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(FITD / f"{a}_preds_{arm}.csv.gz", low_memory=False) for a in AN
                      if (FITD / f"{a}_preds_{arm}.csv.gz").exists()], ignore_index=True)


def analysis(name, arm, D, pcol, tcol="theta_L", key="target", sel=None, store=None):
    cal_full = None
    if sel is not None:
        _l, cal_full = track_licks(D, pcol, tcol, "target")
    L = D if sel is None else D[sel]
    L, cal = track_licks(L, pcol, tcol, key, cal_from=cal_full)
    S = sessions_of(L)
    tag = {"analysis": name, "arm": arm, "decoder": pcol[2:], "angle": tcol, "key": key}
    per, pooled = tracking_stats(S, cal, tag)
    cal_rows = [{**tag, "animal": a, **c} for a, c in cal.items()]
    if store is not None:
        store[name] = {"L": L, "S": S, "cal": cal, "per": per, "pooled": pooled}
    return per, pooled, cal_rows


def cmd_summarise() -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    import likeforlike as lfl
    writeguard.assert_writable(OUT / "x.csv")
    logf = open(OUT / "summarise_log.txt", "w")

    def P(m=""):
        print(m, flush=True)
        logf.write(str(m) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    G = pd.concat([pd.read_csv(FITD / f"{a}_gate.csv") for a in AN if (FITD / f"{a}_gate.csv").exists()],
                  ignore_index=True)
    G.to_csv(OUT / "gate_all.csv", index=False)
    # ---- GATE FIRST
    gate_rows = []
    for (arm, dec), g in G[G.p_gate.notna()].groupby(["arm", "decoder"], sort=False):
        npass = int((g.p_gate < 0.05).sum())
        gate_rows.append({"arm": arm, "decoder": dec, "n_animals": len(g), "n_pass": npass, "gate_pass": npass >= 3,
                          **{f"r_{r.animal}": r.r_pool for r in g.itertuples()},
                          **{f"p_{r.animal}": r.p_gate for r in g.itertuples()},
                          **{f"partial_{r.animal}": r.partial_r for r in g.itertuples()},
                          **{f"ppartial_{r.animal}": r.p_partial for r in g.itertuples()}})
    GT = pd.DataFrame(gate_rows)
    GT.to_csv(OUT / "gate.csv", index=False)
    P("== GATE (pre, pooled LOSO r vs 200 within-session permutations; pass = p < 0.05 in >= 3/4)")
    P(GT.round(3).to_string(index=False))
    gj = {f"{r.arm}|{r.decoder}": bool(r.gate_pass) for r in GT.itertuples()}
    with open(OUT / "gate.json", "w") as fh:
        json.dump(gj, fh, indent=1)
    P("== frozen status: " + G.groupby("frozen").size().to_dict().__repr__())
    G[["animal", "arm", "decoder", "frozen", "alpha_full", "alpha_edge_frac"]].to_csv(OUT / "frozen_status.csv", index=False)
    # ---- data
    Dc, Di, Dj = _load_preds("cued"), _load_preds("iti"), _load_preds("jaw")
    for D in (Dc, Di, Dj):
        D["target"] = D.target.astype(str)
        D["prev_target"] = D.prev_target.astype(str)
    store, per_all, pooled_all, cal_all = {}, [], [], []

    def add(*a, **k):
        per, pooled, cal = analysis(*a, **k, store=store)
        per_all.extend(per)
        pooled_all.extend(pooled)
        cal_all.extend(cal)
    # PRIMARY
    for arm, D in (("cued", Dc), ("iti", Di)):
        add("primary", arm, D, "p_allen:PRE")
    # calibration first
    CAL = pd.DataFrame(cal_all)
    P("\n== CALIBRATION (pre; b_sp = slope of session x target-cell mean decoded on mean theta_L; primary normaliser)")
    P(CAL.round(3).to_string(index=False))
    # per-animal table FIRST (rule 8)
    PA = pd.DataFrame(per_all)
    P("\n== PER-ANIMAL TABLE (primary: Allen PRE; d = vs pre at the same target; d_decs = d_dec / b_sp, deg)")
    P(PA.round(2).to_string(index=False))
    PR = pd.DataFrame(pooled_all)
    PR["p_holm"] = ak.holm(PR.p_num.to_numpy(float))
    PR["tracking"] = (PR.p_holm < 0.05) & ((PR.d_decs_lo > 0) | (PR.d_decs_hi < 0)) & (PR.n_same_sign >= 3)
    PR.to_csv(OUT / "primary_holm.csv", index=False)
    P("\n== PRIMARY pooled (boot_delta_columns; TI = delta d_decs / delta d_exe; Holm over the 6 numerator p)")
    P(PR[["arm", "epoch", "n_animals", "d_decs", "d_decs_lo", "d_decs_hi", "d_exe", "d_exe_lo", "d_exe_hi", "TI",
          "TI_lo", "TI_hi", "p_num", "p_holm", "p_full", "estimable", "n_same_sign", "tracking"]].round(3).to_string(index=False))
    # ---- like-for-like offset (primary)
    lrows = []
    for arm in ("cued", "iti"):
        D = (Dc if arm == "cued" else Di).dropna(subset=["p_allen:PRE", "theta_L"]).copy()
        D["set"] = arm
        D["offset"], _o = lfl.offset(D, "theta_L", "p_allen:PRE", lfl.SIGMA_DEG, what=f"{arm} PRE", method="nw")
        S = D.groupby(["label", "animal", "epoch"]).offset.mean().reset_index()
        for ep in EPS[1:]:
            rng = np.random.default_rng(SEED)
            pre = {a: g.offset.dropna().to_numpy() for a, g in S[S.epoch == "pre"].groupby("animal")}
            post = {a: g.offset.dropna().to_numpy() for a, g in S[S.epoch == ep].groupby("animal")}
            post = {a: v for a, v in post.items() if len(v)}
            iv = ak.boot_delta(post, {a: v for a, v in pre.items() if len(v)}, rng)
            lrows.append({"arm": arm, "epoch": ep, "lfl_offset": iv.point if iv else np.nan,
                          "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan})
    LF = pd.DataFrame(lrows)
    LF.to_csv(OUT / "likeforlike_offset.csv", index=False)
    P("\n== like-for-like offset (decoded - pre reference at the same ACTUAL angle; raw deg)")
    P(LF.round(3).to_string(index=False))
    # ---- SECONDARY
    sec_per, sec_pool = [], []

    def sec(*a, **k):
        per, pooled, cal = analysis(*a, **k, store=store)
        sec_per.extend(per)
        sec_pool.extend(pooled)
        cal_all.extend(cal)
    for arm, D in (("cued", Dc), ("iti", Di)):
        for dec in ("allen:POST", "allen:PRE10", "allen:PREnb", "allen_left:PRE", "allen_right:PRE", "face:PRE",
                    "allen_faceres:PRE", f"svd{K0}:PRE", f"svd{K0}:POST", f"svd{K0}:PRE10", "svd10:PRE", "svd40:PRE",
                    "nmf:PRE", "nmf:POST", "nmf:PRE10"):
            if f"p_{dec}" in D:
                sec(f"dec {dec}", arm, D, f"p_{dec}")
        sec("POST theta_E", arm, D, "p_allen:POST", tcol="theta_E")
        sec("far contra only", arm, D, "p_allen:PRE", sel=(D.target == "far_contra"))
        sec("far contra only POST", arm, D, "p_allen:POST", sel=(D.target == "far_contra"))
        D2 = D.copy()
        D2["tkey"] = D2.target + "|" + D2.prev_target
        sec("history cells (target x prev)", arm, D2, "p_allen:PRE", key="tkey")
        for b_ in range(NB):
            sec(f"bin {CENT[b_]:+.3f}", arm, D, f"p_bin{b_:02d}")
    # ITI subsets
    sec("ITI gap >= 2 s", "iti", Di, "p_allen:PRE", sel=(Di.gap >= 2.0))
    sec("ITI dock01", "iti", Di, "p_allen:PRE", sel=_b(Di.dock01))
    sec("ITI since_stop >= 1.5 s", "iti", Di, "p_allen:PRE", sel=(Di.since_stop >= 1.5))
    if "p_cuedx:PRE" in Di:
        sec("cued decoder on ITI", "iti", Di, "p_cuedx:PRE")
    # jaw-referenced (cued only) + tongue-referenced decoder on the same licks
    for dec in ("allen:PRE", f"svd{K0}:PRE", "nmf:PRE"):
        sec(f"JAW {dec}", "jaw", Dj, f"p_{dec}")
    for b_ in range(NB):
        sec(f"JAW bin {CENT[b_]:+.3f}", "jaw", Dj, f"p_bin{b_:02d}")
    key_j = set(zip(Dj.label, Dj.trial_id))
    selc = np.array([(l_, t_) in key_j for l_, t_ in zip(Dc.label, Dc.trial_id)])
    sec("tongue decoder on jaw licks", "cued", Dc, "p_allen:PRE", sel=selc)
    # RT terciles (cued)
    Dc["rt_terc"] = np.nan
    for _a, g in Dc.groupby("animal"):
        q = np.nanpercentile(g[g.epoch == "pre"].rt, [100 / 3, 200 / 3])
        Dc.loc[g.index, "rt_terc"] = np.digitize(g.rt, q)
    for t_ in range(3):
        sec(f"RT tercile {t_ + 1}", "cued", Dc, "p_allen:PRE", sel=(Dc.rt_terc == t_))
    RT = Dc.groupby(["animal", "epoch"]).agg(n=("rt", "size"), rt_med=("rt", "median"),
                                              frac_rt_gt_05=("rt", lambda s: float((s > 0.5).mean())),
                                              frac_rt_gt_08=("rt", lambda s: float((s > 0.8).mean()))).reset_index()
    RT.to_csv(OUT / "rt_baseline_overlap.csv", index=False)
    P("\n== CUED RT vs BASE (-0.80..-0.50 s from tongue): RT > 0.5 s -> the cue falls inside / before BASE ends; "
      "RT > 0.8 s -> BASE entirely post-cue")
    P(RT.round(3).to_string(index=False))
    SP, SQ = pd.DataFrame(sec_per), pd.DataFrame(sec_pool)
    SP.to_csv(OUT / "secondary_per_animal.csv", index=False)
    SQ.to_csv(OUT / "secondary_pooled.csv", index=False)
    pd.DataFrame(cal_all).to_csv(OUT / "calibration.csv", index=False)
    PA.to_csv(OUT / "per_animal.csv", index=False)
    P("\n== SECONDARY pooled (descriptive)")
    show = ["analysis", "arm", "epoch", "n_animals", "d_decs", "d_decs_lo", "d_decs_hi", "d_exe", "TI", "TI_lo", "TI_hi",
            "p_num", "estimable", "n_same_sign"]
    P(SQ[[c for c in show if c in SQ]].round(3).to_string(index=False))
    # ---- per-bin decomposition of the primary decoder
    brows = []
    for arm in ("cued", "iti"):
        D = Dc if arm == "cued" else Di
        Lp, cal = track_licks(D, "p_allen:PRE")
        nb = int(_bins_of(WINS["PRE"]).sum())
        for b_ in list(range(nb)) + ["int"]:
            col = f"c_bin{b_}" if b_ != "int" else "c_int"
            Lb, _c = track_licks(D.assign(_c=D[col]), "_c")
            Lb["d_decs"] = Lb.d_dec / Lb.animal.map({a: c["b_sp"] for a, c in cal.items()})
            Lb = Lb[Lb.animal.map({a: bool(c.get("calibrated")) for a, c in cal.items()}).fillna(False).astype(bool)]
            S = Lb.groupby(["label", "animal", "epoch"]).d_decs.mean().reset_index()
            for ep in EPS[1:]:
                rng = np.random.default_rng(SEED)
                pre = {a: g.d_decs.to_numpy() for a, g in S[S.epoch == "pre"].groupby("animal")}
                post = {a: g.d_decs.to_numpy() for a, g in S[S.epoch == ep].groupby("animal")}
                iv = ak.boot_delta(post, pre, rng)
                brows.append({"arm": arm, "bin": b_ if b_ == "int" else float(CENT[_bins_of(WINS["PRE"])][b_]),
                              "epoch": ep, "delta": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan,
                              "hi": iv.hi if iv else np.nan})
    BD = pd.DataFrame(brows)
    BD.to_csv(OUT / "perbin_decomposition.csv", index=False)
    P("\n== per-bin decomposition of the primary decoder's delta d_dec* (deg-equivalent; 'int' = intercept term)")
    P(BD.round(3).to_string(index=False))
    # ---- dose control (cell level, with intercept) and history regression
    drows, hrows = [], []
    for arm in ("cued", "iti"):
        D = Dc if arm == "cued" else Di
        L, cal = track_licks(D, "p_allen:PRE")
        C = L.groupby(["label", "animal", "epoch", "target"]).agg(n=("d_exe", "size"), x=("d_exe", "mean"),
                                                                  y=("d_decs", "mean")).reset_index()
        C = C[(C.n >= MIN_CELL) & np.isfinite(C.y)]
        for (a, ep), g in C[C.epoch != "pre"].groupby(["animal", "epoch"]):
            if len(g) < 4:
                continue
            sl, ic = np.polyfit(g.x, g.y, 1)
            drows.append({"arm": arm, "animal": a, "epoch": ep, "n_cells": len(g), "slope": sl, "intercept": ic,
                          "slope_origin": float((g.x * g.y).sum() / (g.x ** 2).sum())})
        L = L[np.isfinite(L.d_decs)].copy()
        L["cell"] = L.label + "|" + L.target
        for (a, ep), g in L[L.epoch != "pre"].groupby(["animal", "epoch"]):
            g = g.dropna(subset=["prev_theta"])
            if len(g) < 50:
                continue
            dd, de, dp = (_cent(g[c].to_numpy(float), g.cell.to_numpy()) for c in ("d_decs", "d_exe", "prev_theta"))
            Bc = np.linalg.lstsq(np.column_stack([de, dp]), dd, rcond=None)[0]
            hrows.append({"arm": arm, "animal": a, "epoch": ep, "n": len(g), "b_exe_within": Bc[0], "b_prev_within": Bc[1]})
    DO, HI = pd.DataFrame(drows), pd.DataFrame(hrows)
    DO.to_csv(OUT / "dose_control.csv", index=False)
    HI.to_csv(OUT / "history_regression.csv", index=False)
    P("\n== DOSE control (post session x target cells: d_dec* ~ 1 + d_exe; uniform shift -> intercept)")
    P(DO.round(3).to_string(index=False))
    P(DO.groupby(["arm", "epoch"])[["slope", "intercept", "slope_origin"]].agg(["mean", lambda s: int((s > 0).sum())])
      .round(3).to_string())
    P("\n== HISTORY (lick level, within session x target, post): d_dec* ~ d_exe + previous lick theta_L")
    P(HI.round(3).to_string(index=False))
    # refit
    RF = pd.concat([pd.read_csv(FITD / f"{a}_refit.csv") for a in AN if (FITD / f"{a}_refit.csv").exists()])
    RF.to_csv(OUT / "refit_sessions.csv", index=False)
    P("\n== within-session refit r (Allen PRE; BIASED near zero signal, read only as change)")
    P(RF.groupby(["arm", "animal", "epoch"]).r_refit.mean().unstack().round(3).to_string())
    # ENL pre-cue reference
    enl = pd.DataFrame()
    try:
        V = pd.read_csv(E / "iti_licks" / "enl_offx_variants.csv")
        enl = V[V.variant.str.startswith("B7")][["variant", "epoch", "delta", "lo", "hi"]]
        P("\n== ENL pre-cue reference (lick_enl_offset_explore B7; 3-D first-lick angle; raw deg)")
        P(enl.round(2).to_string(index=False))
    except FileNotFoundError:
        P("ENL B7 table not found")
    # ---- figdata
    ES = pd.read_csv(OUT / "extract_summary.csv") if (OUT / "extract_summary.csv").exists() else \
        pd.read_csv(E / "prelaunch_aim_tracking" / "extract_summary.csv")
    fd = {"GT": GT, "G": G, "CAL": pd.DataFrame(cal_all), "PA": PA, "PR": PR, "SP": SP, "SQ": SQ, "BD": BD, "DO": DO,
          "HI": HI, "RT": RT, "LF": LF, "ENL": enl, "ES": ES,
          "cells": {arm: store_cells(Dc if arm == "cued" else Di) for arm in ("cued", "iti")},
          }
    # primary session tables per arm
    fd["prim_sess"] = {}
    for arm, D in (("cued", Dc), ("iti", Di)):
        L, cal = track_licks(D, "p_allen:PRE")
        fd["prim_sess"][arm] = sessions_of(L)
        L2, _ = track_licks(D, "p_allen:POST")
        fd["prim_sess"][arm + "_POST"] = sessions_of(L2)
        L3, _ = track_licks(D[D.target == "far_contra"], "p_allen:PRE", cal_from=cal)
        S3 = L3.dropna(subset=["d_dec", "d_exe"]).groupby(["label", "animal", "epoch"]).agg(
            n=("d_exe", "size"), d_decs=("d_decs", "mean"), d_exe=("d_exe", "mean")).reset_index()
        fd["prim_sess"][arm + "_fc"] = S3[S3.n >= MIN_CELL]          # far contra: >= 3 licks per session (figure F)
    FD.mkdir(parents=True, exist_ok=True)
    with open(FD / "prelaunch_aim_tracking.pkl", "wb") as fh:
        pickle.dump(fd, fh)
    logf.close()
    cmd_plot()


def store_cells(D: pd.DataFrame) -> pd.DataFrame:
    """Pre session x target cell means of the primary decoder (calibration panel)."""
    pre = D[D.epoch == "pre"].dropna(subset=["p_allen:PRE"])
    C = pre.groupby(["animal", "label", "target"]).agg(p=("p_allen:PRE", "mean"), t=("theta_L", "mean"),
                                                       n=("theta_L", "size")).reset_index()
    return C[C.n >= MIN_CELL]


# ================================================================================================ figures
def cmd_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from within_lick_correction import reserve_headroom, star_top
    from wfield_local import orofacial_cohort as oc
    with open(FD / "prelaunch_aim_tracking.pkl", "rb") as fh:
        fd = pickle.load(fh)
    FIGS.mkdir(parents=True, exist_ok=True)
    EC, AM = oc.EPOCH_LINE, oc.ANIMAL_MARKER
    GT, G, PR, SQ = fd["GT"], fd["G"], fd["PR"], fd["SQ"]
    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 7, width_ratios=[1.25, 0.12, 1, 0.12, 1, 0.12, 1.15], hspace=0.55, wspace=0.05,
                          left=0.05, right=0.99, top=0.9, bottom=0.12)
    # A gate
    ax = fig.add_subplot(gs[0, 0])
    decs = [("cued", "allen:PRE"), ("cued", f"svd{K0}:PRE"), ("cued", "nmf:PRE"), ("iti", "allen:PRE"),
            ("iti", f"svd{K0}:PRE"), ("iti", "nmf:PRE"), ("jaw", "allen:PRE")]
    for i, (arm, dec) in enumerate(decs):
        g = G[(G.arm == arm) & (G.decoder == dec)]
        for j, a in enumerate(AN):
            r = g[g.animal == a]
            if not len(r):
                continue
            x = i + (j - 1.5) * 0.16
            ax.scatter(x, r.r_pool.iloc[0], marker=AM[a], facecolor="none", edgecolor="k", s=36, zorder=3)
            ax.plot([x - 0.07, x + 0.07], [r.null95.iloc[0]] * 2, color="0.5", lw=1.5)
        gt = GT[(GT.arm == arm) & (GT.decoder == dec)]
        if len(gt):
            star_top(ax, i, f"{int(gt.n_pass.iloc[0])}/4", fontsize=9)
    ax.set_xticks(range(len(decs)), [f"{a}\n{d.split(':')[0]}" for a, d in decs], fontsize=8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("pre held-out r (pooled LOSO)")
    ax.set_title("A. pre gate, pre-launch window\n(grey tick = null 95th pct; top = animals p < 0.05)", fontsize=9, pad=18)
    # B calibration
    ax = fig.add_subplot(gs[0, 2])
    for arm, col in (("cued", "#2166ac"), ("iti", "#b2182b")):
        C = fd["cells"][arm]
        for a in AN:
            c = C[C.animal == a]
            ax.scatter(c.t, c.p, marker=AM[a], facecolor="none", edgecolor=col, s=14, lw=0.7, alpha=0.7)
        cal = fd["CAL"]
        cal = cal[(cal.analysis == "primary") & (cal.arm == arm)]
        xs = np.linspace(-30, 30, 2)
        bm = cal.b_sp.mean()
        ax.plot(xs, bm * xs + C.p.mean() - bm * C.t.mean(), color=col, lw=1.5)
        btxt = getattr(ax, "_btxt", "") + f"\n{arm} ({'blue' if arm == 'cued' else 'red'}): b_sp " + \
            ", ".join(f"{v:.2f}" for v in cal.b_sp)
        ax._btxt = btxt
    ax.plot([-30, 30], [-30, 30], color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("actual theta_L, cell mean (deg, + contra)")
    ax.set_ylabel("decoded (deg; cells beyond +-30 not shown)")
    ax.set_title("B. calibration (pre, session x target cells)" + ax._btxt, fontsize=8, pad=6)
    ax.set_xlim(-32, 32)
    ax.set_ylim(-30, 30)
    # C / D tracking index
    for k, (arm, ttl) in enumerate((("cued", "C. CUED pre-launch"), ("iti", "D. ITI pre-launch"))):
        ax = fig.add_subplot(gs[0, 4 + 2 * k])
        _ti_panel(ax, PR[PR.arm == arm], EC, AM, star_top)
        extra = ""
        if arm == "cued" and len(fd["ENL"]):
            e = fd["ENL"]
            e = e[e.epoch == "acute"]
            t_ = e[e.variant.str.contains("target")].delta
            b_ = e[e.variant.str.contains("behaviour")].delta
            if len(t_) and len(b_):
                extra = f"\npre-cue ENL acute: decoded {t_.iloc[0]:+.2f} vs executed {b_.iloc[0]:+.1f} deg"
        ax.set_title(f"{ttl}: tracking index{extra}", fontsize=9, pad=18)
        if k == 0:
            ax.set_ylabel("TI = d decoded* / d executed")
    # E positive control
    ax = fig.add_subplot(gs[1, 0])
    Q = SQ[SQ.analysis == "dec allen:POST"]
    xt = []
    for i, arm in enumerate(("cued", "iti")):
        q = Q[Q.arm == arm].set_index("epoch")
        for j, ep in enumerate(EPS[1:]):
            x = i * 3.6 + j
            xt.append((x, f"{arm}\n{ep}"))
            if ep not in q.index or not np.isfinite(q.loc[ep, "TI"]):
                continue
            r = q.loc[ep]
            ax.errorbar(x, r.TI, yerr=[[max(r.TI - r.TI_lo, 0)], [max(r.TI_hi - r.TI, 0)]], fmt="o", color=EC[ep], ms=7,
                        capsize=3, mfc=EC[ep] if r.estimable else "w")
            for a in AN:
                v = r.get(f"TI_{a}", np.nan)
                if np.isfinite(v):
                    ax.scatter(x + 0.25, np.clip(v, -1.5, 2.5), marker=AM[a], facecolor="none", edgecolor=EC[ep], s=26)
    ax.set_xticks([x for x, _ in xt], [s for _, s in xt], fontsize=7)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axhline(1, color="0.6", lw=0.6, ls=":")
    ax.set_ylim(-1.6, 2.6)
    ax.set_ylabel("TI (post-contact window)")
    ax.set_title("E. positive control: post-contact +0.05..+0.30 s", fontsize=9, pad=6)
    # F far contra
    for k, arm in enumerate(("cued", "iti")):
        ax = fig.add_subplot(gs[1, 2 + 2 * k])
        S = fd["prim_sess"][f"{arm}_fc"]
        A = S.groupby(["animal", "epoch"])[["d_decs", "d_exe"]].mean().reset_index()
        pre = A[A.epoch == "pre"].set_index("animal")
        for r in A[A.epoch != "pre"].itertuples():
            if r.animal not in pre.index:
                continue
            ax.scatter(r.d_exe - pre.loc[r.animal, "d_exe"], r.d_decs - pre.loc[r.animal, "d_decs"],
                       marker=AM[r.animal], facecolor="none", edgecolor=EC[r.epoch], s=40, lw=1.2)
        lim = 40
        ax.plot([-lim, lim], [-lim, lim], color="0.6", ls=":", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(0, color="0.6", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("d executed theta_L vs pre (deg)")
        if k == 0:
            ax.set_ylabel("d decoded / b_sp vs pre (deg)")
        ax.set_title(f"F{k + 1}. far contra, {arm}: decoded vs executed shift\n(sessions >= 3 licks; dotted = full tracking, 0 = target)",
                     fontsize=9, pad=6)
    # legend axis
    ax = fig.add_subplot(gs[1, 6])
    ax.axis("off")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=AM[a], mfc="none", mec="k", label=a) for a in AN]
    h += [Line2D([], [], ls="", marker="o", color=EC[e], label=e) for e in EPS]
    h += [Line2D([], [], ls="", marker="o", mfc="k", mec="k", label="pooled (filled) +- 95 % CI")]
    ax.legend(handles=h, loc="center", fontsize=9, frameon=False)
    ax.text(0, 0.02, "open = animal (session-pooled); per-animal TI clipped to [-1.5, 2.5]\nNEURAL. n = 4. PROVISIONAL",
            transform=ax.transAxes, fontsize=8, va="bottom")
    fig.suptitle("Does pre-launch cortex follow the re-aimed launch at the same target? Frozen across-spout launch-angle "
                 "decoder (Allen areas, -0.50..-0.05 s from tongue onset)", fontsize=11)
    reserve_headroom(fig)
    fig.savefig(FIGS / "pat_primary.png", dpi=110)
    plt.close(fig)
    _plot_secondary(fd, plt, EC, AM)
    print(f"== figures in {FIGS}", flush=True)


def _ti_panel(ax, Q, EC, AM, star_top):
    q = Q.set_index("epoch")
    for j, ep in enumerate(EPS[1:]):
        if ep not in q.index or not np.isfinite(q.loc[ep].get("TI", np.nan)):
            continue
        r = q.loc[ep]
        ax.errorbar(j, r.TI, yerr=[[max(r.TI - r.TI_lo, 0)], [max(r.TI_hi - r.TI, 0)]], fmt="o", color=EC[ep], ms=8,
                    capsize=3, mfc=EC[ep] if r.estimable else "w")
        for i, a in enumerate(AN):
            v = r.get(f"TI_{a}", np.nan)
            if np.isfinite(v):
                ax.scatter(j + 0.22 + 0.06 * i, np.clip(v, -1.5, 2.5), marker=AM[a], facecolor="none",
                           edgecolor=EC[ep], s=28)
        if bool(r.get("tracking", False)):
            star_top(ax, j, "*")
        else:
            star_top(ax, j, f"p {r.p_holm:.2f}" if "p_holm" in r else "", fontsize=7)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axhline(1, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(range(3), EPS[1:])
    ax.set_xlim(-0.5, 2.8)
    ax.set_ylim(-1.6, 2.6)


def _plot_secondary(fd, plt, EC, AM):
    SQ, BD = fd["SQ"], fd["BD"]
    fig = plt.figure(figsize=(19, 10))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 0.1, 1, 0.1, 1.3], hspace=0.5, wspace=0.05, left=0.05, right=0.99,
                          top=0.9, bottom=0.08)
    # A-C per-bin single-bin decoders: d decoded* and d executed by bin, acute + chronic
    for k, (arm, pref, ttl) in enumerate((("cued", "bin ", "A. CUED, tongue-referenced"),
                                         ("jaw", "JAW bin ", "B. CUED, JAW-referenced"),
                                         ("iti", "bin ", "C. ITI, tongue-referenced"))):
        ax = fig.add_subplot(gs[k // 2 * 1, (k % 2) * 2] if k < 2 else gs[1, 0])
        for ep in ("acute", "chronic"):
            q = SQ[(SQ.arm == arm) & (SQ.analysis.str.startswith(pref)) & (SQ.epoch == ep)].copy()
            if not len(q):
                continue
            q["t"] = q.analysis.str.split().str[-1].astype(float)
            q = q.sort_values("t")
            ax.errorbar(q.t, q.d_decs, yerr=[q.d_decs - q.d_decs_lo, q.d_decs_hi - q.d_decs], fmt="o-", ms=3,
                        color=EC[ep], lw=1, capsize=0, label=f"{ep}: d decoded*")
            ax.plot(q.t, q.d_exe, ls=":", color=EC[ep], lw=1.5, label=f"{ep}: d executed")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.axvspan(BASE[0], BASE[1], color="0.92", zorder=0)
        ax.set_xlabel("bin centre (s from " + ("JAW" if arm == "jaw" else "tongue") + " onset)")
        ax.set_ylabel("deg vs pre at same target")
        ax.set_title(f"{ttl}: single-bin decoders\n(grey = BASE; dotted = executed shift)", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    # D per-bin decomposition
    ax = fig.add_subplot(gs[1, 2])
    for arm, mk in (("cued", "o"), ("iti", "s")):
        q = BD[(BD.arm == arm) & (BD.bin != "int")].copy()
        q["bin"] = q.bin.astype(float)
        for ep in ("acute", "chronic"):
            g = q[q.epoch == ep]
            ax.errorbar(g.bin + (0.006 if arm == "iti" else 0), g.delta, yerr=[g.delta - g.lo, g.hi - g.delta],
                        fmt=mk, ms=4, color=EC[ep], mfc="none" if arm == "iti" else EC[ep], capsize=0,
                        label=f"{arm} {ep}")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("bin centre (s from tongue onset)")
    ax.set_ylabel("contribution to d decoded* (deg)")
    ax.set_title("D. primary decoder: per-bin share of the decoded change", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    # E sensitivity forest (acute + chronic TI numerator)
    ax = fig.add_subplot(gs[:, 4])
    names = ["primary"] + [n for n in ("dec allen:PRE10", "dec allen:PREnb", "dec allen_left:PRE", "dec allen_right:PRE",
                                       "dec allen_faceres:PRE", "dec face:PRE", f"dec svd{K0}:PRE", "dec svd10:PRE",
                                       "dec svd40:PRE", "dec nmf:PRE", "history cells (target x prev)",
                                       "far contra only", "RT tercile 1", "RT tercile 2", "RT tercile 3",
                                       "tongue decoder on jaw licks", "JAW allen:PRE", "ITI gap >= 2 s", "ITI dock01",
                                       "ITI since_stop >= 1.5 s", "cued decoder on ITI")]
    PRm = fd["PR"].assign(analysis="primary")
    allq = pd.concat([PRm, SQ], ignore_index=True)
    y = 0
    yt = []
    for n in names:
        for arm in ("cued", "jaw", "iti"):
            q = allq[(allq.analysis == n) & (allq.arm == arm)]
            if not len(q):
                continue
            for k, ep in enumerate(("acute", "chronic")):
                r = q[q.epoch == ep]
                if not len(r) or not np.isfinite(r.d_decs.iloc[0]):
                    continue
                r = r.iloc[0]
                ax.errorbar(r.d_decs, y + k * 0.3, xerr=[[r.d_decs - r.d_decs_lo], [r.d_decs_hi - r.d_decs]], fmt="o",
                            color=EC[ep], ms=4, capsize=0)
                ax.scatter(r.d_exe, y + k * 0.3, marker="|", color=EC[ep], s=40)
            yt.append((y + 0.15, f"{n} [{arm}]"))
            y += 1
    ax.set_yticks([v for v, _ in yt], [s for _, s in yt], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("d decoded* (dot, 95 % CI) and d executed (bar), deg vs pre at same target")
    ax.set_title("E. sensitivities: acute (upper) / chronic (lower)", fontsize=9)
    fig.suptitle("Pre-launch aim tracking: time course, jaw reference, sensitivities (NEURAL; n = 4; PROVISIONAL)",
                 fontsize=11)
    fig.savefig(FIGS / "pat_secondary.png", dpi=110)
    plt.close(fig)


# ================================================================================================ main
def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if "--plot-only" in sys.argv or cmd == "plot":
        cmd_plot()
        return 0
    jobs = min(4, int(_arg("--jobs", 4)))
    if cmd == "extract":
        cmd_extract(jobs)
    elif cmd == "fit":
        cmd_fit(jobs)
    elif cmd == "summarise":
        cmd_summarise()
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
