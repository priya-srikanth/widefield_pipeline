"""PRE-LAUNCH LAUNCH-ANGLE READOUT FROM ALM + tjM1 PIXELS, POOLED PRE SESSIONS (Priya 10-30: "why area pools instead
of an SVD / pixel readout within the full Allen ALM / tjM1 areas? averaging pixels may cancel a spatial pattern").
Add-on to `prelick_hemisphere_tuning.py`, whose area-pool gate failed (0/4 animals PRE).

DESIGN (fixed before any number): docs/status/PRELICK_SVD_TUNING_PREREG_20261030.md -- read it first.

Two changes against the area analysis and only two: features = 2 x 2-binned pixels of ALM U tjM1 (ROI v8) per
hemisphere; the animal's AUGUST pre sessions are pooled into one decoder (leave-one-SESSION-out). Trials, launch angle,
covariates, windows, baseline: IDENTICAL -- the lick set is read from the area analysis's per-session cache and every
signal step reuses its functions.

SIGN / NAMING:
  y      within-spout launch angle residualised on spout FE + reach + peak speed / 100 + RT, per session (deg).
  r      corr(prediction, y) within ONE session (unsigned information readout; + = predicts the angle).
  hemi   c = contralesional = RIGHT hemisphere (image right); i = ipsilesional = LEFT (image left). D = r_c - r_i.
  pre    per-session r from the frozen LOSO siblings (out-of-session); post = the frozen full-pre decoder's r.

Usage (bash env per the repo; outputs only under <v36>/imaging_encoding/prelick_svd_tuning/):
  python scripts/prelick_svd_tuning.py extract [--jobs 4] [--only PS92] [--labels a,b]   per-session pixel features
  python scripts/prelick_svd_tuning.py fit [--only PS92]          pooled-pre LOSO + permutations + frozen + post scores
  python scripts/prelick_svd_tuning.py refit [--jobs 4]           per-session within-session CV refit r
  python scripts/prelick_svd_tuning.py summarise                  gate, per-animal table, boot_delta, Holm, figdata
  python scripts/prelick_svd_tuning.py plot  (or --plot-only)     figures from figdata only
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prelick_hemisphere_tuning as pht  # noqa: E402

AN = pht.AN
EPS = pht.EPS
HEMI = pht.HEMI                       # {"c": "right", "i": "left"}
WINS = pht.WINS                       # PRE / POST
BIN = 2
KS = (5, 10, 20)
K0 = 10
KMAX = max(KS)
ALPHAS = 10.0 ** np.arange(-1, 6)
N_PERM, N_PERM_SENS = 200, 100
SEED = 0
REFIT_FOLDS, REFIT_REP, REFIT_INNER = 5, 5, 3
MIN_N = pht.MIN_N

OUT = Path(os.environ.get("PST_OUT") or (pht.OUT.parent / "prelick_svd_tuning"))
FEAT = OUT / "features"
FIGS = OUT / "figures"
FD = OUT / "figdata"
SRC = pht.SESS                        # the area analysis's per-session lick sets


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _is_pre_aug(label: str, ep: str) -> bool:
    from wfield_local import config
    return ep == "pre" and not config.early_pre(label)


def labels() -> list[str]:
    from wfield_local import epochs
    out = []
    for f in sorted(SRC.glob("*.npz")):
        lab = f.stem
        if lab[:4] not in AN or ".tmp" in lab:
            continue
        ep = epochs.epoch_of(lab)
        if ep not in EPS or (ep == "pre" and not _is_pre_aug(lab, ep)):
            continue
        out.append(lab)
    only = _arg("--only")
    if only:
        out = [x for x in out if x[:4] in only.split(",")]
    if _arg("--labels"):
        out = [x for x in out if x in _arg("--labels").split(",")]
    return out


# ================================================================================================ extraction
def _target(theta, reach, speed, rt, pos) -> np.ndarray:
    """Within-spout launch angle residualised on reach / peak speed / RT (the area slope model's 'within' design)."""
    n = len(theta)
    Z = np.column_stack([np.ones(n), reach, speed / 100, rt, pht._spout_fe(pos)])
    return pht._resid(Z, theta[:, None].astype(float))[:, 0]


def _block_ids(flat_ix: np.ndarray, W: int) -> np.ndarray:
    r, c = flat_ix // W, flat_ix % W
    return (r // BIN) * ((W + BIN - 1) // BIN) + (c // BIN)


def extract(label: str) -> dict:
    """One session -> FEAT/<label>.npz: per lick y + PRE / POST (minus BASE) window means of the 2 x 2-binned ALM U
    tjM1 pixels per hemisphere (computed in V space and projected through the binned U rows: every step is linear) +
    the MOTOR area pool (from the area analysis's own traces)."""
    f = FEAT / f"{label}.npz"
    if f.exists() and "--redo" not in sys.argv:
        return {"label": label, "cached": True}
    t0 = time.time()
    from wfield_local import combined_encoding as ce
    z = np.load(SRC / f"{label}.npz", allow_pickle=True)
    t = z["first_t"].astype(float)
    n = len(t)
    pos = z["first_pos"].astype(str)
    th, re, sp, rt = (z[f"first_{k}"].astype(float) for k in ("theta_L", "reach", "speed_peak", "rt"))
    out = {"label": label, "n": n, "pos": pos, "theta_L": th, "trial_id": z["first_trial_id"].astype(float),
           "y": _target(th, re, sp, rt, pos) if n >= MIN_N else np.full(n, np.nan)}
    cols = [str(c) for c in z["cols"]]
    bB = pht._bins_of(pht.BASE)
    Xa = z["first_X"].astype(np.float64)                                    # (n, NB, 18) area traces
    summ = {"label": label, "n": n}
    for hk, h in HEMI.items():
        mot = (Xa[:, :, cols.index(f"ALM_{h}")] + Xa[:, :, cols.index(f"tjM1_{h}")]) / 2
        for w, ws in WINS.items():
            out[f"motor_{hk}_{w}"] = mot[:, pht._bins_of(ws)].mean(1) - mot[:, bB].mean(1)
    if n < MIN_N:
        np.savez_compressed(f, **out)
        return {**summ, "skipped": "n<MIN_N"}
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = pht._masks(parea, pstat, names, flat, shape)
    V = V.astype(np.float64)
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    V -= Q @ (Q.T @ V)
    V *= 1e3
    Xb = pht._binned(V, ft, t).astype(np.float64)                            # (n, NB, K)
    del V
    base = Xb[:, bB].mean(1)
    Vw = {w: Xb[:, pht._bins_of(ws)].mean(1) - base for w, ws in WINS.items()}
    W = int(shape[1])
    for hk, h in HEMI.items():
        roi = M[f"ALM_{h}"] | M[f"tjM1_{h}"]
        bid = _block_ids(flat[roi], W)
        ub, inv = np.unique(bid, return_inverse=True)
        Ur = np.asarray(Ub[roi], np.float64)
        Ubin = np.zeros((len(ub), Ur.shape[1]))
        np.add.at(Ubin, inv, Ur)
        Ubin /= np.bincount(inv, minlength=len(ub))[:, None]
        out[f"blocks_{hk}"] = ub.astype(np.int64)
        out[f"col_med_{hk}"] = float(np.median(flat[roi] % W))
        for w in WINS:
            out[f"X_{hk}_{w}"] = (Vw[w] @ Ubin.T).astype(np.float32)
        # exactness check against the area analysis's ALM trace (same window arithmetic, area mean of U rows)
        Ra = np.asarray(Ub[M[f"ALM_{h}"]], np.float64).mean(0)
        a_new = Vw["PRE"] @ Ra
        a_old = Xa[:, pht._bins_of(WINS["PRE"]), cols.index(f"ALM_{h}")].mean(1) \
            - Xa[:, bB, cols.index(f"ALM_{h}")].mean(1)
        summ[f"alm_check_maxdiff_{hk}"] = float(np.max(np.abs(a_new - a_old)))
        summ[f"alm_check_sd_{hk}"] = float(np.std(a_old))
        summ[f"n_blocks_{hk}"] = len(ub)
    FEAT.mkdir(parents=True, exist_ok=True)
    tmp = FEAT / f"{label}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(f)
    return {**summ, "s": round(time.time() - t0, 1)}


def cmd_extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    FEAT.mkdir(parents=True, exist_ok=True)
    labs = labels()
    print(f"{len(labs)} sessions -> {FEAT}", flush=True)
    t0 = time.time()
    res, fail = ak.fan_sessions(labs, extract, jobs=jobs)
    S = pd.DataFrame([v for _k, v in res])
    f = OUT / "extract_summary.csv"
    if f.exists() and not _arg("--only") and not _arg("--labels"):
        pass
    elif f.exists():
        old = pd.read_csv(f)
        S = pd.concat([old[~old.label.isin(S.label)], S]).sort_values("label")
    S.to_csv(f, index=False)
    print(S.drop(columns=[c for c in S.columns if c.startswith("col_")], errors="ignore").round(4).to_string(index=False))
    print(f"extracted {(time.time() - t0) / 60:.1f} min; failed {[str(x[0]) for x in fail][:5]}", flush=True)
    if fail:
        for x in fail[:3]:
            print(x, flush=True)
        raise SystemExit(1)


# ================================================================================================ decoder primitives
def _centre(X: np.ndarray, sess: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Within session x spout centring on all licks of the set (label-free)."""
    X = np.array(X, np.float64, copy=True)
    key = pd.Series(sess).astype(str) + "|" + pd.Series(pos).astype(str)
    for _k, ix in key.groupby(key).groups.items():
        ix = np.asarray(ix)
        X[ix] -= X[ix].mean(0)
    return X


def pca_fit(X: np.ndarray, k: int = KMAX) -> tuple[np.ndarray, np.ndarray]:
    """(mean, components (k, p)): covariance PCA, randomized SVD with a fixed seed (bitwise reproducible here)."""
    from sklearn.utils.extmath import randomized_svd
    mu = X.mean(0)
    k = min(k, X.shape[0] - 1, X.shape[1])
    _u, _s, vt = randomized_svd(X - mu, n_components=k, n_iter=7, random_state=0)
    return mu, vt


def _ridge_preds(Ztr, Ytr, Zte, alphas=ALPHAS) -> np.ndarray:
    """(A, n_te, m) ridge predictions for every alpha and every column of Ytr (intercept = training mean)."""
    ym = Ytr.mean(0)
    s, Vv = np.linalg.eigh(Ztr.T @ Ztr)
    B = Vv.T @ (Ztr.T @ (Ytr - ym))                                            # (K, m)
    out = np.empty((len(alphas), len(Zte), Ytr.shape[1]))
    for i, a in enumerate(alphas):
        out[i] = Zte @ (Vv @ (B / (s + a)[:, None])) + ym
    return out


def _colcorr(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """corr over rows, per column; P (..., n, m), Y (n, m) -> (..., m); 0 where undefined."""
    Pc = P - P.mean(-2, keepdims=True)
    Yc = Y - Y.mean(0)
    num = (Pc * Yc).sum(-2)
    den = np.sqrt((Pc ** 2).sum(-2) * (Yc ** 2).sum(0))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def _inner_alpha(Z, Y, groups) -> np.ndarray:
    """Per column of Y: the alpha maximising the mean held-out r over the inner groups (sessions or folds)."""
    gs = np.unique(groups)
    acc = np.zeros((len(ALPHAS), Y.shape[1]))
    for g in gs:
        te = groups == g
        acc += _colcorr(_ridge_preds(Z[~te], Y[~te], Z[te]), Y[te])
    return np.argmax(acc, 0)


def _std(Ztr, *Zs):
    mu, sd = Ztr.mean(0), Ztr.std(0)
    sd[sd == 0] = 1
    return (mu, sd), [(z - mu) / sd for z in (Ztr, *Zs)]


def _fit_model(X, y, sess, K, pca=None) -> dict:
    """PCA (K; None = no PCA) + standardise + ridge with alpha by inner leave-one-session-out. Real y only."""
    if K is None:
        mu, comps = np.zeros(X.shape[1]), np.eye(X.shape[1])
    else:
        mu, comps = pca if pca is not None else pca_fit(X)
        comps = comps[:K]
    Z = (X - mu) @ comps.T
    (zm, zs), (Zs,) = _std(Z)
    Y = y[:, None]
    ia = int(_inner_alpha(Zs, Y, sess)[0])
    ym = float(y.mean())
    s, Vv = np.linalg.eigh(Zs.T @ Zs)
    w = (Vv @ ((Vv.T @ (Zs.T @ (y - ym))) / (s + ALPHAS[ia])))
    return {"mu": mu.astype(np.float32), "comps": comps.astype(np.float32), "zm": zm, "zs": zs,
            "alpha": float(ALPHAS[ia]), "w": w, "ym": ym, "K": K}


def _predict(m: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - m["mu"].astype(np.float64)) @ m["comps"].astype(np.float64).T
    return ((Z - m["zm"]) / m["zs"]) @ m["w"] + m["ym"]


def _r(a, b) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pix_weights(m: dict) -> np.ndarray:
    """Ridge weights back-projected through the PCA to the pixel blocks (deg per dF/F x1e3 unit)."""
    return m["comps"].astype(np.float64).T @ (m["w"] / m["zs"])


# ================================================================================================ per-animal fit
def _load_animal(an: str, labs: list[str]) -> dict:
    from wfield_local import epochs
    A = {"label": [], "y": [], "pos": [], "epoch": []}
    X = {(hk, w): [] for hk in HEMI for w in WINS}
    Mo = {(hk, w): [] for hk in HEMI for w in WINS}
    blocks = {}
    for lab in labs:
        z = np.load(FEAT / f"{lab}.npz", allow_pickle=True)
        n = int(z["n"])
        if n < MIN_N or "X_c_PRE" not in z:
            continue
        for hk in HEMI:
            b = z[f"blocks_{hk}"]
            if hk in blocks and not np.array_equal(blocks[hk], b):
                raise SystemExit(f"{lab}: pixel blocks differ from the animal's first session ({hk})")
            blocks[hk] = b
            for w in WINS:
                X[(hk, w)].append(z[f"X_{hk}_{w}"])
                Mo[(hk, w)].append(z[f"motor_{hk}_{w}"][:, None])
        A["label"] += [lab] * n
        A["y"].append(z["y"])
        A["pos"].append(z["pos"].astype(str))
        A["epoch"] += [epochs.epoch_of(lab)] * n
    A = {"label": np.array(A["label"]), "y": np.concatenate(A["y"]), "pos": np.concatenate(A["pos"]),
         "epoch": np.array(A["epoch"])}
    A["X"] = {k: _centre(np.concatenate(v), A["label"], A["pos"]) for k, v in X.items()}
    A["M"] = {k: _centre(np.concatenate(v), A["label"], A["pos"]) for k, v in Mo.items()}
    A["blocks"] = blocks
    return A


def _perms(y, sess, pos, rng, n) -> np.ndarray:
    """(len(y), n) permutations of y within session x spout cells."""
    key = pd.Series(sess).astype(str) + "|" + pd.Series(pos).astype(str)
    groups = [np.asarray(ix) for _k, ix in key.groupby(key).groups.items()]
    P = np.empty((len(y), n))
    for j in range(n):
        yp = y.copy()
        for ix in groups:
            yp[ix] = y[rng.permutation(ix)]
        P[:, j] = yp
    return P


def _loso_multi(X, Y, sess, K_list, pcas) -> dict:
    """{K: (n_sess, m) per-session held-out r} for every column of Y (real + permutations); PCA per outer fold
    (label-free, computed once and reused); alpha per column by inner LOSO on the training sessions."""
    us = sorted(np.unique(sess))
    R = {K: np.full((len(us), Y.shape[1]), np.nan) for K in K_list}
    for i, s in enumerate(us):
        te = sess == s
        for K in K_list:
            if K is None:
                Z = X
            else:
                mu, comps = pcas[s]
                Z = (X - mu) @ comps[:K].T
            _, (Ztr, Zte) = _std(Z[~te], Z[te])
            ia = _inner_alpha(Ztr, Y[~te], sess[~te])
            P = _ridge_preds(Ztr, Y[~te], Zte)                                   # (A, n_te, m)
            Pb = P[ia, :, np.arange(Y.shape[1])].T                                 # (n_te, m)
            R[K][i] = _colcorr(Pb, Y[te])
    return {"labels": us, "R": R}


def _digest(*arrs) -> str:
    h = hashlib.sha1()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes())
    return h.hexdigest()


def fit_animal(an: str) -> dict:
    """One animal: pooled-pre LOSO (+ permutation nulls), frozen full + LOSO decoders (rule 10), post frozen r."""
    from wfield_local import frozen_models as fm
    t0 = time.time()
    labs = [x for x in labels() if x[:4] == an]
    A = _load_animal(an, labs)
    pre = np.array([_is_pre_aug(lab, ep) for lab, ep in zip(A["label"], A["epoch"])])
    pre_labs = sorted(set(A["label"][pre]))
    yp, sp, pp = A["y"][pre], A["label"][pre], A["pos"][pre]
    rng = np.random.default_rng([SEED, int(an[2:])])
    Yall = np.column_stack([yp, _perms(yp, sp, pp, rng, N_PERM)])
    rows_loso, rows_null, rows_post, weights, status = [], [], [], {}, []
    feats = [(hk, w, "pix") for hk in HEMI for w in WINS] + [(hk, w, "motor") for hk in HEMI for w in WINS]
    for hk, w, kind in feats:
        X = (A["X"] if kind == "pix" else A["M"])[(hk, w)]
        Xp = X[pre]
        K_list = list(KS) if kind == "pix" else [None]
        pcas = {s: pca_fit(Xp[sp != s]) for s in pre_labs} if kind == "pix" else {}
        L = _loso_multi(Xp, Yall, sp, K_list, pcas)
        full_pca = pca_fit(Xp) if kind == "pix" else None
        for K in K_list:
            tag = f"pix{K}" if kind == "pix" else "motor"
            Rk = L["R"][K]
            nperm = N_PERM if (K in (K0, None)) else N_PERM_SENS
            obs = float(np.nanmean(Rk[:, 0]))
            null = np.nanmean(Rk[:, 1:1 + nperm], 0)
            # ---- frozen reference (rule 10): full-pre + LOSO siblings, real y only
            src = (f"pix_almtjm1_r{pht_roi()}_{HEMI[hk]}_bin{BIN}_K{K}" if kind == "pix"
                   else f"motorpool_r{pht_roi()}_{HEMI[hk]}")
            dig = _digest(Xp, yp)
            spec = fm.make_spec(an, "prelaunch_svd", align=f"tongue_{w}", source=src, train_labels=pre_labs,
                                n_features=Xp.shape[1], zscore=True, train_data={"xy_sha1": dig, "sess": sp})  # training-data fingerprint (FREEZE_VERSION 2, rebuild 11-03)

            def fit(Xp=Xp, yp=yp, sp=sp, K=K, pcas=pcas, full_pca=full_pca, dig=dig, hk=hk, kind=kind):
                return {"full": _fit_model(Xp, yp, sp, K, full_pca),
                        "loso": {s: _fit_model(Xp[sp != s], yp[sp != s], sp[sp != s], K, pcas.get(s))
                                 for s in pre_labs},
                        "xy_sha1": dig, "blocks": A["blocks"].get(hk) if kind == "pix" else None}
            P, st = fm.load_or_fit(spec, fit, meta={"script": "prelick_svd_tuning.py", "n_trials": int(len(yp)),
                                                     "window": w, "hemi": HEMI[hk], "features": tag})
            warn = "" if P.get("xy_sha1") == dig else " !! stored model trained on DIFFERENT trial data"
            status.append({"animal": an, "hemi": hk, "win": w, "feat": tag, "status": st, "warn": warn,
                           "spec_id": fm.spec_id(spec)})
            print(f"[frozen] {an} {hk} {w} {tag}: {st}{warn}", flush=True)
            # pre per-session r from the frozen LOSO siblings; consistency with the permutation path's real column
            for i, s in enumerate(L["labels"]):
                q = sp == s
                r_f = _r(_predict(P["loso"][s], Xp[q]), yp[q])
                rows_loso.append({"label": s, "animal": an, "hemi": hk, "win": w, "feat": tag, "r": r_f,
                                  "r_permpath": float(Rk[i, 0]), "n": int(q.sum())})
            pred_all = np.concatenate([_predict(P["loso"][s], Xp[sp == s]) for s in L["labels"]])
            y_all = np.concatenate([yp[sp == s] for s in L["labels"]])
            rows_null.append({"animal": an, "hemi": hk, "win": w, "feat": tag, "r_loso": obs,
                              "r_loso_concat": _r(pred_all, y_all), "n_perm": nperm,
                              "null_mean": float(np.mean(null)), "null95": float(np.percentile(null, 95)),
                              "p": float((1 + np.sum(null >= obs)) / (1 + nperm)), "n_pre_sessions": len(pre_labs),
                              "alpha_full": P["full"]["alpha"]})
            if kind == "pix":
                weights[(hk, w, K)] = pix_weights(P["full"]).astype(np.float32)
            # post sessions: frozen full-pre decoder
            for s in sorted(set(A["label"][~pre])):
                q = A["label"] == s
                rows_post.append({"label": s, "animal": an, "hemi": hk, "win": w, "feat": tag,
                                  "r": _r(_predict(P["full"], X[q]), A["y"][q]), "n": int(q.sum())})
    return {"loso": rows_loso, "null": rows_null, "post": rows_post, "weights": weights, "blocks": A["blocks"],
            "status": status, "s": round(time.time() - t0, 1)}


def pht_roi() -> str:
    import iti_licks as il
    return str(il.ROI_VERSION)


def cmd_fit(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import frozen_models as fm
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    d = fm.local_dir()
    kinds = set()
    if d.exists():
        for a in d.iterdir():
            if a.is_dir():
                kinds |= {x.name.split("_")[0] for x in a.iterdir()}
    print(f"[frozen] local_dir = {d}; exists = {d.exists()}; kinds present = {sorted(kinds)}", flush=True)
    if os.environ.get("PST_SMOKE_FROZEN"):          # smoke test only: a throwaway store in the scratchpad
        print("[frozen] SMOKE store (throwaway) -- not the reference", flush=True)
    elif not d.exists() or not kinds:
        raise SystemExit("frozen store ABSENT or EMPTY -- stopping before any fit (rule 10)")
    ans = [a for a in AN if (not _arg("--only")) or a in _arg("--only").split(",")]
    t0 = time.time()
    res, fail = ak.fan_sessions(ans, fit_animal, jobs=min(jobs, len(ans)), label="animal")
    if fail:
        for x in fail:
            print(x, flush=True)
        raise SystemExit("fit failed")
    tag = "" if not _arg("--only") else "_" + _arg("--only").replace(",", "_")
    pd.DataFrame([r for _a, v in res for r in v["loso"]]).to_csv(OUT / f"loso_pre{tag}.csv", index=False)
    pd.DataFrame([r for _a, v in res for r in v["null"]]).to_csv(OUT / f"perm_null{tag}.csv", index=False)
    pd.DataFrame([r for _a, v in res for r in v["post"]]).to_csv(OUT / f"sessions_frozen{tag}.csv", index=False)
    pd.DataFrame([r for _a, v in res for r in v["status"]]).to_csv(OUT / f"frozen_status{tag}.csv", index=False)
    import pickle
    (OUT / "weights").mkdir(exist_ok=True)
    for a, v in res:
        (OUT / "weights" / f"{a}.pkl").write_bytes(pickle.dumps({"weights": v["weights"], "blocks": v["blocks"]}))
    print(f"fit {(time.time() - t0) / 60:.1f} min ({[(a, v['s']) for a, v in res]})", flush=True)


# ================================================================================================ per-session refit
def _strat_folds(pos, rng, k):
    f = np.empty(len(pos), int)
    for v in np.unique(pos):
        ix = np.flatnonzero(pos == v)
        f[rng.permutation(ix)] = np.arange(len(ix)) % k
    return f


def refit_session(label: str) -> list:
    """Within-session 5-fold CV (stratified by spout, 5 assignments averaged): PCA (K = 10) + ridge inside the folds,
    alpha by 3-fold inner CV on the training trials."""
    from wfield_local import config, epochs
    z = np.load(FEAT / f"{label}.npz", allow_pickle=True)
    if int(z["n"]) < MIN_N or "X_c_PRE" not in z:
        return []
    y, pos = z["y"].astype(float), z["pos"].astype(str)
    sess = np.array([label] * len(y))
    rows = []
    for hk in HEMI:
        for w in WINS:
            for tag in ("pix10", "motor"):
                X = _centre(z[f"X_{hk}_{w}"] if tag == "pix10" else z[f"motor_{hk}_{w}"][:, None], sess, pos)
                rng = np.random.default_rng([SEED, int(label[2:4]), int(label[-4:])])
                rs = []
                for _rep in range(REFIT_REP):
                    f = _strat_folds(pos, rng, REFIT_FOLDS)
                    pred = np.empty(len(y))
                    for k in range(REFIT_FOLDS):
                        tr, te = f != k, f == k
                        if tag == "pix10":
                            mu, comps = pca_fit(X[tr], K0)
                            Z = (X - mu) @ comps.T
                        else:
                            Z = X
                        _, (Ztr, Zte) = _std(Z[tr], Z[te])
                        inner = _strat_folds(pos[tr], rng, REFIT_INNER)
                        ia = _inner_alpha(Ztr, y[tr, None], inner)
                        pred[te] = _ridge_preds(Ztr, y[tr, None], Zte)[ia[0], :, 0]
                    rs.append(_r(pred, y))
                rows.append({"label": label, "animal": config.animal_of(label), "epoch": epochs.epoch_of(label),
                             "hemi": hk, "win": w, "feat": tag, "r": float(np.nanmean(rs)), "n": len(y)})
    return rows


def cmd_refit(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    t0 = time.time()
    res, fail = ak.fan_sessions(labels(), refit_session, jobs=jobs)
    if fail:
        raise SystemExit(f"refit failed: {fail[:3]}")
    tag = "" if not _arg("--only") else "_" + _arg("--only").replace(",", "_")
    pd.DataFrame([r for _k, v in res for r in v]).to_csv(OUT / f"sessions_refit{tag}.csv", index=False)
    print(f"refit {(time.time() - t0) / 60:.1f} min", flush=True)


# ================================================================================================ summarise
def _with_meta(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import epochs
    S = S.copy()
    S["epoch"] = S.label.map(epochs.epoch_of)
    S["day"] = S.label.map(epochs.days_since_stroke)
    return S


def _with_D(S: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    W = S.pivot_table(index=["label", "animal", "epoch"] + keys, columns="hemi", values="r").reset_index()
    W = W.dropna(subset=["c", "i"])
    W["r"] = W.c - W.i
    W["hemi"] = "c-i"
    return pd.concat([S, W.drop(columns=["c", "i"])], ignore_index=True)


def cmd_summarise() -> None:
    import lick_correction as lc
    from figdata import save_fd
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    pd.set_option("display.width", 250)
    log = open(OUT / "summarise_log.txt", "w", encoding="utf-8")

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
    N = pd.read_csv(OUT / "perm_null.csv")
    LP = pd.read_csv(OUT / "loso_pre.csv")
    FP = pd.read_csv(OUT / "sessions_frozen.csv")
    RF = pd.read_csv(OUT / "sessions_refit.csv")
    ST = pd.read_csv(OUT / "frozen_status.csv")
    P("== frozen status ==")
    P(ST.groupby("status").size().to_string())
    if ST.warn.notna().any() and (ST.warn.fillna("") != "").any():
        P("!! xy mismatch:", ST[ST.warn.fillna("") != ""].to_string(index=False))
    P(f"\nfrozen-LOSO r vs permutation-path r: max |diff| = {np.nanmax(np.abs(LP.r - LP.r_permpath)):.2e}")
    # ---------------- GATE
    P("\n== GATE (August pre, pooled LOSO r = mean held-out per-session r; null = 200 within session x spout "
      "permutations; K = 10 pixels) ==")
    G = N[N.feat == f"pix{K0}"].copy()
    P(G[["win", "animal", "hemi", "r_loso", "r_loso_concat", "null_mean", "null95", "p", "n_pre_sessions",
         "alpha_full"]].sort_values(["win", "animal", "hemi"]).round(4).to_string(index=False))
    gate = {}
    for w in WINS:
        g = G[G.win == w]
        per = g.groupby("animal").p.min()
        gate[w] = {"animals_pass": int((per < 0.05).sum()),
                   "animals_pass_bonf2": int((per < 0.025).sum()),
                   "hemis_pass": {f"{r.animal}_{r.hemi}": bool(r.p < 0.05) for r in g.itertuples()}}
        gate[w]["passed"] = gate[w]["animals_pass"] >= 3
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1))
    G.to_csv(OUT / "gate.csv", index=False)
    P(f"GATE PRE: {gate['PRE']['animals_pass']}/4 animals (Bonferroni x2: {gate['PRE']['animals_pass_bonf2']}/4) -> "
      f"{'PASSED' if gate['PRE']['passed'] else 'FAILED'}; POSITIVE CONTROL POST: {gate['POST']['animals_pass']}/4 "
      f"(Bonf x2 {gate['POST']['animals_pass_bonf2']}/4) -> {'PASSED' if gate['POST']['passed'] else 'FAILED'}")
    P("\n== sensitivity K = 5 / 20 (100 permutations) and MOTOR area pool (200) ==")
    P(N[N.feat != f"pix{K0}"][["feat", "win", "animal", "hemi", "r_loso", "null95", "p"]]
      .sort_values(["feat", "win", "animal", "hemi"]).round(4).to_string(index=False))
    # ---------------- per-session long table: pre = frozen-LOSO r, post = frozen full r
    S = _with_meta(pd.concat([LP[["label", "animal", "hemi", "win", "feat", "r", "n"]],
                              FP[["label", "animal", "hemi", "win", "feat", "r", "n"]]], ignore_index=True))
    S = S[S.epoch.isin(EPS)]
    S2 = _with_D(S, ["win", "feat"])
    S2["score"] = "frozen"
    R2 = _with_D(RF[RF.epoch.isin(EPS)].assign(day=np.nan), ["win", "feat"])
    R2["score"] = "refit"
    ALL = pd.concat([S2, R2], ignore_index=True)
    ALL.to_csv(OUT / "sessions_long.csv", index=False)
    KEYS = ["score", "win", "feat", "hemi"]
    PA = lc.per_animal(ALL, KEYS, ["r"])
    PA.to_csv(OUT / "per_animal.csv", index=False)
    P("\n== PER-ANIMAL (rule 8): frozen r (pre = held-out LOSO), K = 10 pixels; session means / n sessions ==")
    for w in WINS:
        q = PA[(PA.score == "frozen") & (PA.win == w) & (PA.feat == f"pix{K0}")]
        P(f"-- {w} --")
        P(q.pivot_table(index=["hemi", "animal"], columns="epoch", values="r_mean").reindex(columns=EPS)
          .round(3).to_string())
        P(q.pivot_table(index=["hemi", "animal"], columns="epoch", values="r_count").reindex(columns=EPS)
          .to_string())
    P("\n-- noise floor: SD over pre sessions of the held-out r (K = 10) --")
    nf = S[(S.epoch == "pre") & (S.feat == f"pix{K0}")].groupby(["win", "animal", "hemi"]).r.std().unstack()
    P(nf.round(3).to_string())
    nf.to_csv(OUT / "noise_floor.csv")
    t0 = time.time()
    B = pht._change(lc.boot_table(ALL, KEYS, ["r"]).fillna({"n_pos": 0, "n_paired": 0}))
    B.to_csv(OUT / "boot.csv", index=False)
    P(f"boot {(time.time() - t0) / 60:.1f} min")
    # ---------------- primary: P1 / P2 / P3, PRE, K = 10, frozen; Holm within epoch (and over 9)
    rng = np.random.default_rng(SEED)
    rows = []
    Pm = ALL[(ALL.score == "frozen") & (ALL.win == "PRE") & (ALL.feat == f"pix{K0}")]
    for hk, nm in (("c", "P1 r_c (contralesional, right)"), ("i", "P2 r_i (ipsilesional, left)"),
                   ("c-i", "P3 D = r_c - r_i")):
        gm = Pm[Pm.hemi == hk]
        pre = {a: list(h.r.dropna()) for a, h in gm[gm.epoch == "pre"].groupby("animal", sort=True)}
        for ep in EPS[1:]:
            cur = {a: list(h.r.dropna()) for a, h in gm[gm.epoch == ep].groupby("animal", sort=True)}
            iv, dr = ak.boot_delta(cur, pre, rng, draws=True)
            common = sorted(set(cur) & set(pre))
            da = [np.mean(cur[a]) - np.mean(pre[a]) for a in common]
            rows.append({"hemi": hk, "name": nm, "epoch": ep, "delta": iv.point if iv else np.nan,
                         "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan, "p": ak.boot_p(dr),
                         "n_pos": int(np.sum(np.array(da) > 0)), "n_paired": len(common),
                         "per_animal": " ".join(f"{a}:{x:+.3f}" for a, x in zip(common, da))})
    H = pd.DataFrame(rows)
    H["p_holm_epoch"] = np.nan
    for ep in EPS[1:]:
        q = H.epoch == ep
        H.loc[q, "p_holm_epoch"] = ak.holm(H.loc[q, "p"].to_numpy())
    H["p_holm_9"] = ak.holm(H.p.to_numpy())
    H["n_same"] = np.where(H.delta > 0, H.n_pos, H.n_paired - H.n_pos)
    H["change"] = ((H.lo > 0) | (H.hi < 0)) & (H.n_same >= 3)
    H.to_csv(OUT / "primary_holm.csv", index=False)
    P(f"\n== PRIMARY (frozen r, PRE, K = 10 pixels) boot_delta vs August pre; gate PRE "
      f"{gate['PRE']['animals_pass']}/4 -> {'confirmatory' if gate['PRE']['passed'] else 'DESCRIPTIVE (gate failed)'} ==")
    P(H.round(3).to_string(index=False))
    P("\n== boot lines (levels + changes), K = 10 / motor, frozen and refit ==")
    show = B[B.feat.isin([f"pix{K0}", "motor"])]
    P(show[["score", "win", "feat", "hemi", "epoch", "level", "level_lo", "level_hi", "delta", "delta_lo", "delta_hi",
            "n_same", "change", "per_animal_delta"]].round(3).to_string(index=False))
    # ---------------- atlas for the weight maps
    import pickle
    Wt = {a: pickle.loads((OUT / "weights" / f"{a}.pkl").read_bytes()) for a in AN
          if (OUT / "weights" / f"{a}.pkl").exists()}
    save_fd("prelick_svd_tuning", {"N": N, "G": G, "gate": gate, "ALL": ALL, "B": B, "H": H, "PA": PA, "W": Wt,
                                   "atlas": _atlas()}, root=FD)
    P("figdata saved")
    log.close()


def _atlas() -> dict:
    """Allen area grid + brain mask (identical across sessions) for the weight-map outlines."""
    from wfield_local import config
    s = next(x for x in config.load_sessions() if x["label"] == labels()[0])
    at = Path(s["mc"]) / "wfield_local_results" / "allen_aligned_affine8v1"
    return {"area": np.load(at / "allen_area_atlas_native_grid.npy"),
            "mask": np.load(at / "allen_brain_mask_native_grid.npy").astype(bool)}


# ================================================================================================ figures
HCOL = pht.HCOL
HLAB = {"c": "contralesional (RIGHT) hemisphere", "i": "ipsilesional (LEFT) hemisphere", "c-i": "D = r_c - r_i"}


def _wrap(s, n=110):
    return pht._wrap(s, n)


def epoch_panel(ax, S, B, sel, hemi, row=0, holm=None):
    """Sessions = open markers (shape = animal, colour = epoch); pooled level = filled marker in the epoch colour with
    95 % CI; no lines between epochs; top-band * = change vs pre (CI excl 0 AND >= 3/4 animals), ** = also Holm."""
    import within_lick_correction as wlc
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    s = pht._sel(S, **sel)
    s = s[s.hemi == hemi]
    rng = np.random.default_rng(7)
    for an, h in s.groupby("animal"):
        v = h.groupby(["epoch", "label"]).r.mean().dropna()
        ep_of = list(v.index.get_level_values(0))
        xi = np.array([EPS.index(e) for e in ep_of], float) + rng.uniform(-0.12, 0.12, len(v)) - 0.12
        ax.scatter(xi, v.to_numpy(float), s=16, facecolors="none", edgecolors=[EPOCH_LINE[e] for e in ep_of],
                   alpha=0.7, marker=ANIMAL_MARKER.get(an, "o"), linewidths=0.8, zorder=2)
    b = pht._sel(B, **sel)
    b = b[(b.hemi == hemi) & (b.metric == "r")].set_index("epoch").reindex(EPS)
    for i, ep in enumerate(EPS):
        if not np.isfinite(b.loc[ep, "level"]):
            continue
        ax.errorbar(i + 0.18, b.loc[ep, "level"], yerr=[[b.loc[ep, "level"] - b.loc[ep, "level_lo"]],
                                                       [b.loc[ep, "level_hi"] - b.loc[ep, "level"]]],
                    color=EPOCH_LINE[ep], lw=2, marker="o", ms=7, mfc=EPOCH_LINE[ep], capsize=3, zorder=4)
        if ep != "pre" and b.loc[ep, "change"] == 1:
            txt = "*"
            if holm is not None:
                hh = holm[(holm.hemi == hemi) & (holm.epoch == ep)]
                if len(hh) and hh.p_holm_epoch.iloc[0] < 0.05:
                    txt = "**"
            wlc.star_top(ax, i + 0.18, txt, "k", row=row)
    ax.set_xticks(np.arange(len(EPS)), EPS, fontsize=8)
    ax.set_xlim(-0.6, len(EPS) - 0.4)
    ax.tick_params(labelsize=8)
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)


def gate_panel(ax, G, feat=f"pix{K0}"):
    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    xs = {}
    x = 0
    for w in WINS:
        for hk in ("c", "i"):
            xs[(w, hk)] = x
            x += 1
        x += 0.6
    for r in G[G.feat == feat].itertuples():
        x0 = xs[(r.win, r.hemi)] + (AN.index(r.animal) - 1.5) * 0.17
        ax.plot([x0 - 0.07, x0 + 0.07], [r.null95, r.null95], color="0.45", lw=1.6, zorder=2)
        ax.scatter(x0, r.r_loso, marker=ANIMAL_MARKER[r.animal], s=42, edgecolors=HCOL[r.hemi],
                   facecolors=HCOL[r.hemi] if r.p < 0.05 else "none", linewidths=1.2, zorder=3)
    ax.set_xticks(list(xs.values()), [f"{w} {'contra (R)' if h == 'c' else 'ipsi (L)'}" for w, h in xs],
                  fontsize=8)
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    ax.set_ylabel("pre LOSO r (held-out session)", fontsize=8)
    ax.tick_params(labelsize=8)


def pix_vs_pool_panel(ax, N):
    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    cells = [(w, hk) for w in WINS for hk in ("c", "i")]
    for j, (w, hk) in enumerate(cells):
        for feat, dx in ((f"pix{K0}", -0.17), ("motor", 0.17)):
            q = N[(N.win == w) & (N.hemi == hk) & (N.feat == feat)]
            for r in q.itertuples():
                ax.scatter(j * 1.4 + dx + (AN.index(r.animal) - 1.5) * 0.04, r.r_loso,
                           marker=ANIMAL_MARKER[r.animal], s=26, facecolors="none",
                           edgecolors=HCOL[hk] if feat != "motor" else "0.45", linewidths=0.9, zorder=3)
            ax.scatter(j * 1.4 + dx, q.r_loso.mean(), marker="_", s=260, color=HCOL[hk] if feat != "motor" else "0.3",
                       linewidths=2.2, zorder=4)
    ax.set_xticks([j * 1.4 for j in range(len(cells))],
                  [f"{w} {h}: pix | pool" for w, h in cells], fontsize=8)
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)
    ax.set_ylabel("pre LOSO r", fontsize=8)
    ax.tick_params(labelsize=8)


def weight_map(ax, Wd, atlas, w, K=K0, crop=True):
    """Both hemispheres' pixel weights on one brain image (left hemisphere = image LEFT = ipsilesional)."""
    from wfield_local.atlas_overlay import overlay_regions, region_edges
    area, mask = atlas["area"], atlas["mask"]
    H, W = area.shape
    W2 = (W + BIN - 1) // BIN
    img = np.full((H, W), np.nan)
    rr, cc = np.mgrid[0:H, 0:W]
    bid = (rr // BIN) * W2 + (cc // BIN)
    vmax = 0
    for hk in HEMI:
        if (hk, w, K) not in Wd["weights"]:
            continue
        wt = Wd["weights"][(hk, w, K)].astype(float)
        # per-hemisphere NORMALISED (w / max |w|): the inner-CV alpha differs between hemispheres (0.1 vs 1e5 on the
        # standardised PCs), so absolute weights are not comparable across them; the map shows the spatial pattern only
        wt = wt / (np.nanmax(np.abs(wt)) or 1)
        bl = Wd["blocks"][hk]
        lut = dict(zip(bl.tolist(), wt.tolist()))
        sel = np.isin(bid, bl) & mask
        img[sel] = [lut[b] for b in bid[sel]]
        vmax = max(vmax, np.nanpercentile(np.abs(wt), 99))
    ax.imshow(np.where(mask, 0.0, np.nan), cmap="Greys", vmin=0, vmax=4)
    im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    overlay_regions(ax, region_edges(np.where(mask, area, 0)))
    if crop:
        ys, xs_ = np.where(np.isfinite(img))
        if len(ys):
            ax.set_ylim(ys.max() + 40, max(ys.min() - 40, 0))
            ax.set_xlim(max(xs_.min() - 40, 0), min(xs_.max() + 40, W))
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def fig_primary(D):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import within_lick_correction as wlc
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    ALL, B, H, N, gate = D["ALL"], D["B"], D["H"], D["N"], D["gate"]
    fig = plt.figure(figsize=(16, 17), layout="constrained")
    gs = fig.add_gridspec(4, 13, height_ratios=[1, 1, 1, 1.15])  # col 8 = gap so y labels never sit on data
    axA = fig.add_subplot(gs[0, 0:7])
    gate_panel(axA, N)
    axA.set_title(_wrap(f"A  Gate (pre-stroke, pooled August sessions): held-out-session r of the K = {K0} pixel "
                        f"decoder per animal x hemisphere; dash = 95th pct of 200 within session x spout permutations; "
                        f"filled = p < 0.05. PRE {gate['PRE']['animals_pass']}/4 animals pass (gate "
                        f"{'PASSED' if gate['PRE']['passed'] else 'FAILED'}); POST positive control "
                        f"{gate['POST']['animals_pass']}/4 ({'PASSED' if gate['POST']['passed'] else 'FAILED'}).",
                        95), fontsize=9, loc="left")
    axE = fig.add_subplot(gs[0, 8:13])
    pix_vs_pool_panel(axE, N)
    axE.set_title(_wrap("E  Pixels vs the MOTOR area pool (ALM + tjM1 mean, 1 feature), same pooled-pre LOSO "
                        "pipeline: per animal (open) and mean (bar); coloured = pixels, grey = pool", 60),
                  fontsize=9, loc="left")
    sel = {"score": "frozen", "feat": f"pix{K0}"}
    axs = {}
    for r, (w, letter) in enumerate((("PRE", "B"), ("POST", "C")), start=1):
        for c, hk in enumerate(("c", "i")):
            ax = fig.add_subplot(gs[r, c * 4:(c + 1) * 4])
            epoch_panel(ax, ALL, B, {**sel, "win": w}, hk, holm=H if w == "PRE" else None)
            ax.set_title(_wrap(f"{letter}{c + 1}  {w}: frozen-decoder r, {HLAB[hk]}", 55), fontsize=9,
                         color=HCOL[hk], loc="left")
            if hk == "c":
                ax.set_ylabel("r (pre: held-out LOSO; post: frozen)", fontsize=8)
            axs[(w, hk)] = ax
        ax = fig.add_subplot(gs[r, 9:13])
        epoch_panel(ax, ALL, B, {**sel, "win": w}, "c-i", holm=H if w == "PRE" else None)
        ax.set_title(_wrap(f"D{r}  {w}: hemisphere difference D = r_c - r_i", 55), fontsize=9, loc="left")
        ax.set_ylabel("D (r)", fontsize=8)
    for w in WINS:
        axs[(w, "i")].sharey(axs[(w, "c")])
        axs[(w, "i")].tick_params(labelleft=False)
    W = D["W"]
    for j, an in enumerate(AN):
        ax = fig.add_subplot(gs[3, (0, 3, 7, 10)[j]:(3, 6, 10, 13)[j]])
        if an in W:
            im = weight_map(ax, W[an], D["atlas"], "PRE")
            fig.colorbar(im, ax=ax, shrink=0.8, orientation="horizontal", label="weight / max |weight| (per hemisphere)")
        else:
            ax.set_axis_off()
        ax.set_title(f"F{j + 1}  {an}: PRE weights\n(image left = ipsi, right = contra)", fontsize=9, loc="left")
    wlc.reserve_headroom(fig)
    h = [Line2D([], [], ls="none", marker=m, mfc="none", mec="0.3", label=a) for a, m in ANIMAL_MARKER.items()]
    h += [Line2D([], [], ls="none", marker="o", color=EPOCH_LINE[e], label=f"{e} (filled = pooled, 95 % CI)")
          for e in EPS]
    h += [Line2D([], [], ls="none", marker="s", color=HCOL[k], label=HLAB[k]) for k in ("c", "i")]
    fig.legend(handles=h, loc="outside lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(_wrap("Pre-launch launch-angle readout from ALM + tjM1 PIXELS (2 x 2-binned, PCA K = 10 inside the "
                       "training sessions, ridge), August pre sessions pooled (leave-one-session-out). Cued FIRST "
                       "licks, engaged, no lick in the prior 1 s; target = within-spout launch angle residualised on "
                       "reach / peak speed / RT. Windows from tongue onset: PRE -0.30..-0.05 s, POST +0.05..+0.30 s, "
                       "minus BASE -0.80..-0.50 s. Open = sessions (shape = animal, colour = epoch). Top-band * = "
                       "change vs pre (CI excludes 0 AND >= 3/4 animals); ** = also Holm over P1-P3 within the epoch. "
                       + ("" if gate["PRE"]["passed"] else "Gate FAILED: post-stroke panels are DESCRIPTIVE."), 200),
                 fontsize=9.5)
    return fig


def fig_secondary(D):
    import matplotlib.pyplot as plt

    import within_lick_correction as wlc
    ALL, B = D["ALL"], D["B"]
    rows = [("refit", f"pix{K0}", "per-session REFIT r (within-session 5-fold CV, K = 10)"),
            ("frozen", "motor", "MOTOR area pool (1 feature), frozen pooled-pre decoder"),
            ("frozen", "pix5", "pixels, K = 5, frozen"), ("frozen", "pix20", "pixels, K = 20, frozen")]
    fig, axs = plt.subplots(len(rows), 4, figsize=(17, 3.3 * len(rows)), layout="constrained", sharey="row")
    for r, (sc, ft, tt) in enumerate(rows):
        for c, (w, hk) in enumerate([(w, hk) for w in WINS for hk in ("c", "i")]):
            ax = axs[r, c]
            epoch_panel(ax, ALL, B, {"score": sc, "feat": ft, "win": w}, hk)
            ax.set_title(_wrap(f"{chr(65 + r * 4 + c)}  {tt}; {w}; {HLAB[hk]}", 52), fontsize=8.5,
                         color=HCOL[hk], loc="left")
        axs[r, 0].set_ylabel("r", fontsize=8)
    wlc.reserve_headroom(fig)
    fig.suptitle(_wrap("Secondary readouts (descriptive): refit r separates loss of the frozen frame (frozen down, "
                       "refit unchanged) from loss of the code (both down); the MOTOR pool through the same pooled "
                       "pipeline separates the gain from pooling sessions from the gain from pixels; K sensitivity. "
                       "Same marker conventions as the primary figure; stars = change criterion, uncorrected.", 190),
                 fontsize=9.5)
    return fig


def fig_weights_post(D):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.5), layout="constrained")
    for j, an in enumerate(AN):
        if an in D["W"]:
            im = weight_map(axs[j], D["W"][an], D["atlas"], "POST")
            fig.colorbar(im, ax=axs[j], shrink=0.8, orientation="horizontal", label="weight / max |weight| (per hemisphere)")
        else:
            axs[j].set_axis_off()
        axs[j].set_title(f"{chr(65 + j)}  {an}: POST weights (left = ipsi)", fontsize=9, loc="left")
    fig.suptitle("Pixel weight maps of the full-pre POST-window decoder (ridge weights back-projected through PCA K = 10)",
                 fontsize=9.5)
    return fig


def cmd_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figdata import load_fd
    D = load_fd("prelick_svd_tuning", root=FD)
    FIGS.mkdir(parents=True, exist_ok=True)
    for name, fn in (("pst_primary", fig_primary), ("pst_secondary", fig_secondary),
                     ("pst_weights_POST", fig_weights_post)):
        fig = fn(D)
        fig.savefig(FIGS / f"{name}.png", dpi=130)
        plt.close(fig)
        print(f"wrote {FIGS / name}.png", flush=True)


def main() -> int:
    from wfield_local import parallel
    parallel.pin_blas()
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in (_arg("--jobs"), _arg("--only"),
                                                                             _arg("--labels"))]
    cmd = args[0] if args else "plot"
    jobs = min(4, int(_arg("--jobs", 4)))
    if "--plot-only" in sys.argv or cmd == "plot":
        cmd_plot()
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    {"extract": lambda: cmd_extract(jobs), "fit": lambda: cmd_fit(jobs), "refit": lambda: cmd_refit(jobs),
     "summarise": lambda: (cmd_summarise(), cmd_plot())}.get(cmd, lambda: (_ for _ in ()).throw(SystemExit(__doc__)))()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
