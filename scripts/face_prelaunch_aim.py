"""WHAT in the FACE carries the pre-launch aim, WHEN, and does face movement explain the frozen-decoder OFFSET?
Pre-registered: docs/status/FACE_PRELAUNCH_AIM_PREREG_20261031.md (follow-up to S26, prelaunch_aim_tracking.py).

Behaviour-only decoders (Q1-Q4) of launch angle theta_L (+ = contralesional) from face signals in S26's lick sets (cued,
jaw-referenced cued, docked ITI), S26's ridge / LOSO / frozen pipeline and tracking index (TI). Face families:
  PTS    FaceRhythm mesh point-displacement PCs 1-20 (frozen pre PCA, 25 Hz, timing-valid)        PRIMARY
  JAW    cam4 cleaned jaw lateral (+ = contralesional = image LEFT), opening, both velocities      PRIMARY
  FRall  master_pre_v1 absolute factors grouped (lick, left, right, close, far + 6 'other')         LEAK-EXPOSED (acausal VQT)
  FR groups / single 'other' factors / drop-one; NOSE (cam4 raw nose); JAW3D (cam1+cam4 jaw ml / dv); FACEALL; beyond-jaw
Q5: cortical features of the frozen KEY lateral decoder (iti_lick_order, Allen split r8) partialled on the face around
each lick (regression fitted on PRE licks; LOSO for pre), re-scored -> docked ITI same-place offset
(lick_matched_readout --ref loclin, WFIELD_ITI_FILTER=dock) and cued matched-error offset (lick_balance_matched_error).

  python scripts/face_prelaunch_aim.py extract [--jobs 4] [--animals PS93]
  python scripts/face_prelaunch_aim.py fit [--jobs 4] [--animals PS93] [--nperm 200] [--smoke]
  python scripts/face_prelaunch_aim.py summarise
  python scripts/face_prelaunch_aim.py offset [--jobs 4]          (Q5; reproduction check first, stops if it fails)
  python scripts/face_prelaunch_aim.py plot   (or --plot-only)     figures from figdata only
Env: FPA_OUT redirects outputs (smoke test); WIDEFIELD_FROZEN_MODEL_DIR redirects the frozen store (smoke test).
Ipsilesional = left hemisphere; in face video ipsi = image RIGHT.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prelaunch_aim_tracking as pat  # noqa: E402

V36 = pat.V36
E = pat.E          # pat.E = run_mode.encoding_out_root(...): the TAGGED encoding tree (routed, coordinator 11-04)
OUT = Path(os.environ.get("FPA_OUT") or (E / "face_prelaunch_aim"))
FEAT = E / "face_prelaunch_aim" / "features"          # deterministic, shared with the smoke test
S26F = pat.FEAT
FIGS, FD, FITD, OFFD = OUT / "figures", OUT / "figdata", OUT / "fit", OUT / "offset"
AN, EPS, POS = pat.AN, pat.EPS, pat.POS
ARMS = ("cued", "iti", "jaw")
EDGES, CENT, FINE, BB, NB = pat.EDGES, pat.CENT, pat.FINE, pat.BB, pat.NB
WINS = {"PRE": (-0.50, -0.05), "PRE10": (-0.50, -0.10), "POST": (0.05, 0.30)}
SEED = 0
FRD = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1")
P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d")
# master_v1 grouping (wfield_local/fr_groups.py): raw factor index -> grouped traces (order fixed)
FR_LICK = (2, 4, 5, 7, 8, 11)
FR_OTHER = (0, 1, 3, 6, 9, 10)
FR_NAMES = ["lick", "left", "right", "close", "far"] + [f"o{k}" for k in FR_OTHER]
FR_LAB = {"o0": "sniff_prelick", "o1": "nose_prelick_slow", "o3": "nose_steps", "o6": "run_onset_jaw",
          "o9": "whisk_run", "o10": "prep_jaw"}
LEAK_FWHM, LEAK_SUPPORT = 0.35, 0.70        # FR lick-factor half-height half-width / support (s), prereg section 0
MAIN = ("PTS", "JAW", "FRall")
GATED = ("PTS", "JAW", "FRall", "NOSE", "JAW3D", "FR_lick", "FR_left", "FR_right", "FR_side", "FR_close", "FR_far",
         "FR_other", "PTS_bj", "FRall_bj")
SINGLE = ("PTS", "JAW", "FRall")            # single-bin time courses
MAX_GAP_PTS = 0.1


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _animals() -> list[str]:
    a = _arg("--animals")
    return a.split(",") if a else AN


def labels() -> list[str]:
    return sorted(p.stem for p in S26F.glob("PS9*.npz") if p.stem[:4] in AN and ".tmp" not in p.stem)


# ================================================================================================ extraction
def _binmean(F: np.ndarray) -> np.ndarray:
    """(n, 110, k) fine samples -> (n, 22, k) bin means; NaN where < 50 % (3 of 5) of a bin's samples are finite."""
    n, _, k = F.shape
    G = F.reshape(n, NB, 5, k)
    c = np.isfinite(G).sum(2)
    with np.errstate(invalid="ignore"):
        m = np.nansum(G, 2) / np.maximum(c, 1)
    return np.where(c >= 3, m, np.nan).astype(np.float32)


def _nearest(tsrc: np.ndarray, vals: np.ndarray, t: np.ndarray, tol: float) -> np.ndarray:
    """Value of the nearest source sample within tol s (vals NaN = not measured)."""
    j = np.clip(np.searchsorted(tsrc, t), 1, len(tsrc) - 1)
    jl = j - 1
    k = np.where(np.abs(tsrc[j] - t) < np.abs(tsrc[jl] - t), j, jl)
    out = vals[k].astype(float)
    out[np.abs(tsrc[k] - t) > tol] = np.nan
    return out


def _vel(P: np.ndarray) -> np.ndarray:
    """velocity on the 10-ms fine grid (per lick row), NaN-propagating central difference."""
    V = np.full_like(P, np.nan)
    V[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 0.02
    return V


def session_signals(an: str, date: str, want3d: bool = True):
    """Loader of one session's face / pose signals on the DAQ clock: dict of (t, values) or callables."""
    from wfield_local import face_rhythm_outputs as fro
    files = fro.session_keys(FRD, an, date)
    out = {}
    if files:
        t_fr, W = fro.combined_series(files, "factors", None, absolute=True)
        tp, Wp = [], []
        for f in files:
            z = np.load(f)
            tp.append(z["points_t_daq_s"].astype(float))
            Wp.append(z["points_pc"].astype(float))
        tp, Wp = np.concatenate(tp), np.concatenate(Wp)
        o = np.argsort(tp, kind="stable")
        out["fr"] = (t_fr, W)
        out["pts"] = (tp[o], Wp[o])
    sd = V36 / "sessions" / an / date
    cl = np.load(sd / "clean.npz")
    wz = np.load(sd / "windows.npz")
    td = wz["ix_daq_s"].astype(float)
    o = np.argsort(td, kind="stable")
    td = td[o]
    jx, jy, jf = cl["jaw_x"][o].astype(float), cl["jaw_y"][o].astype(float), cl["jaw_fill"][o]
    rest = cl["jaw_rest"].astype(float)
    meas = np.isfinite(jx) & np.isfinite(jy) & (jf == 0)
    lat = np.where(meas, -(jx - rest[0]), np.nan)                  # + = contralesional = image LEFT
    opn = np.where(meas, jy - rest[1], np.nan)
    pose = wz["pose"][o]
    nok = pose[:, 0, 2] >= 0.9
    nx = np.where(nok, pose[:, 0, 0], np.nan).astype(float)
    ny = np.where(nok, pose[:, 0, 1], np.nan).astype(float)
    nmed = np.nanmedian(nx)
    out["jaw"] = (td, np.column_stack([lat, opn]))
    out["nose"] = (td, np.column_stack([-(nx - nmed), ny - np.nanmedian(ny)]))
    # 3-D jaw: frames on (trial_id, t_daq_ms from the cue); cue DAQ time per trial from the cam4 windows index
    f3 = P3D / f"{an}_{date}_frames.csv.gz"
    if want3d and f3.exists():
        tid = wz["ix_trial_id"][o]
        tdm = wz["ix_t_daq_ms"][o].astype(float)
        cue = pd.Series(td - tdm / 1000.0).groupby(tid).median()
        F = pd.read_csv(f3, usecols=["trial_id", "t_daq_ms", "jaw_ml", "jaw_dv"])
        F["t"] = F.trial_id.map(cue) + F.t_daq_ms / 1000.0
        F = F.dropna(subset=["t"]).sort_values("t")
        F = F[~F.t.round(4).duplicated()]
        out["jaw3d"] = (F.t.to_numpy(float), F[["jaw_ml", "jaw_dv"]].to_numpy(float))
    return out


def extract(label: str) -> dict:
    f = FEAT / f"{label}.npz"
    if f.exists() and "--redo" not in sys.argv:
        return {"label": label, "cached": True}
    from wfield_local import face_rhythm_outputs as fro
    t0 = time.time()
    an, date = label[:4], "2026" + label[-4:]
    z = np.load(S26F / f"{label}.npz", allow_pickle=False)
    sig = session_signals(an, date)
    out, summ = {"label": label}, {"label": label, "animal": an, "epoch": str(z["epoch"])}
    for s in ARMS:
        t = z[f"{s}_m_t"].astype(float)
        n = len(t)
        T = (t[:, None] + FINE[None, :]).ravel()
        if "fr" in sig and n:
            Ffr = fro.to_frames(*sig["fr"], T, 0.5).reshape(n, len(FINE), -1)
            Fpt = fro.to_frames(*sig["pts"], T, MAX_GAP_PTS).reshape(n, len(FINE), -1)
        else:
            Ffr = np.full((n, len(FINE), 12), np.nan)
            Fpt = np.full((n, len(FINE), 20), np.nan)
        J = np.stack([_nearest(sig["jaw"][0], sig["jaw"][1][:, k], T, 0.003).reshape(n, -1) for k in range(2)], -1) \
            if n else np.zeros((0, len(FINE), 2))
        N = np.stack([_nearest(sig["nose"][0], sig["nose"][1][:, k], T, 0.003).reshape(n, -1) for k in range(2)], -1) \
            if n else np.zeros((0, len(FINE), 2))
        if "jaw3d" in sig and n:
            J3 = np.stack([_nearest(sig["jaw3d"][0], sig["jaw3d"][1][:, k], T, 0.003).reshape(n, -1) for k in range(2)], -1)
        else:
            J3 = np.full((n, len(FINE), 2), np.nan)
        J = np.concatenate([J, np.stack([_vel(J[..., 0]), _vel(J[..., 1])], -1)], -1)
        N = np.concatenate([N, np.stack([_vel(N[..., 0]), _vel(N[..., 1])], -1)], -1)
        J3 = np.concatenate([J3, np.stack([_vel(J3[..., 0]), _vel(J3[..., 1])], -1)], -1)
        for k_, A in (("fr", Ffr), ("pts", Fpt), ("jaw", J), ("nose", N), ("jaw3d", J3)):
            out[f"{s}_{k_}"] = _binmean(A)
            summ[f"ok_{s}_{k_}"] = float(np.isfinite(out[f"{s}_{k_}"]).all((1, 2)).mean()) if n else np.nan
        summ[f"n_{s}"] = n
        # QC: 3-D jaw dv vs cam4 jaw opening on common fine samples (a pairing check, behaviour only)
        ok = np.isfinite(J[..., 1]) & np.isfinite(J3[..., 1])
        summ[f"r_jaw3d_cam4_{s}"] = float(np.corrcoef(J[..., 1][ok], J3[..., 1][ok])[0, 1]) if ok.sum() > 100 else np.nan
    FEAT.mkdir(parents=True, exist_ok=True)
    tmp = FEAT / f"{label}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(f)
    return {**summ, "s": round(time.time() - t0, 1)}


def cmd_extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(FEAT / "x.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    labs = [x for x in labels() if x[:4] in _animals()]
    print(f"extract: {len(labs)} sessions, {jobs} workers", flush=True)
    t0 = time.time()
    res, fails = ak.fan_sessions(labs, extract, jobs=jobs, label="extract")
    for it, err in fails:
        print(f"  !! {it}: {err}", flush=True)
    S = pd.DataFrame([v for _k, v in res])
    f = OUT / "extract_summary.csv"
    if f.exists():
        old = pd.read_csv(f)
        S = pd.concat([old[~old.label.isin(S.label)], S], ignore_index=True)
    S.sort_values("label").to_csv(f, index=False)
    print(S.drop(columns=[c for c in S.columns if c.startswith("ok_jaw3d")], errors="ignore").describe().round(3).T.to_string())
    print(f"extract done {(time.time() - t0) / 60:.1f} min, {len(fails)} failed", flush=True)
    if fails:
        raise SystemExit("extraction failures")


# ================================================================================================ per-animal data
def _fr11(F: np.ndarray) -> np.ndarray:
    lick = F[..., list(FR_LICK)].sum(-1, keepdims=True)
    return np.concatenate([lick, F[..., [5]], F[..., [4]], F[..., [7]], F[..., [8]], F[..., list(FR_OTHER)]], -1)


def load_animal(an: str) -> dict:
    labs = [x for x in labels() if x[:4] == an]
    out = {"labels": labs}
    for s in ARMS:
        meta, fam = [], {k: [] for k in ("fr", "pts", "jaw", "nose", "jaw3d")}
        for lab in labs:
            z = np.load(S26F / f"{lab}.npz", allow_pickle=False)
            m = pd.DataFrame({c[len(s) + 3:]: z[c] for c in z.files if c.startswith(f"{s}_m_")})
            m.insert(0, "label", lab)
            m.insert(1, "animal", an)
            m.insert(2, "epoch", str(z["epoch"]))
            meta.append(m)
            y = np.load(FEAT / f"{lab}.npz")
            for k in fam:
                fam[k].append(y[f"{s}_{k}"].astype(np.float64))
        M = pd.concat(meta, ignore_index=True)
        F = {k: np.concatenate(v) for k, v in fam.items()}
        fr = _fr11(F["fr"])
        fams = {"PTS": F["pts"], "JAW": F["jaw"], "FRall": fr, "NOSE": F["nose"], "JAW3D": F["jaw3d"],
                "FR_lick": fr[..., [0]], "FR_left": fr[..., [1]], "FR_right": fr[..., [2]], "FR_side": fr[..., [1, 2]],
                "FR_close": fr[..., [3]], "FR_far": fr[..., [4]], "FR_other": fr[..., 5:],
                "FACEALL": np.concatenate([F["pts"], F["jaw"], fr], -1)}
        for i, nm in enumerate(FR_NAMES):
            if nm.startswith("o"):
                fams[f"FR_{nm}"] = fr[..., [i]]
            fams[f"FRdrop_{nm}"] = None if nm.startswith("o") else np.delete(fr, [i], -1)
        fams = {k: v for k, v in fams.items() if v is not None}
        out[s] = {"meta": M, "fam": fams}
    return out


def _window(X: np.ndarray, w: str, base: bool = True) -> np.ndarray:
    Xw = X[:, pat._bins_of(WINS[w])]
    if base:
        Xw = Xw - X[:, BB].mean(1, keepdims=True)
    return Xw.reshape(len(X), -1)


def _bin_feats(X: np.ndarray, b: int) -> np.ndarray:
    return X[:, b] if BB[b] else X[:, b] - X[:, BB].mean(1)


# ------------------------------------------------------------------------------------------------ frozen helpers
def _spec(an, align, source, labs_pre, n_feat, train_data):
    from wfield_local import frozen_models as fm
    return fm.make_spec(an, "face_aim", align=align, source=source, train_labels=labs_pre, n_features=int(n_feat),
                        train_data=train_data)  # training-data fingerprint (FREEZE_VERSION 2, rebuild 11-03)


def frozen_resid(an, arm, w, name, X, Z, meta, log=print):
    """X residualised on Z (OLS with intercept) fitted on PRE licks with both finite; pre sessions LOSO, post the all-pre
    fit; frozen (kind face_aim). Applied as X - (Z - mean_Z_train) B (pre means kept). Returns X_res (NaN where either
    is not finite)."""
    from wfield_local import frozen_models as fm
    ok = np.isfinite(X).all(1) & np.isfinite(Z).all(1)
    pre = ok & (meta.epoch == "pre").to_numpy()
    sess = meta.label.to_numpy(str)
    labs_pre = sorted(set(sess[pre]))
    Xr = np.full_like(X, np.nan)
    if len(labs_pre) < pat.MIN_PRE_SESS or pre.sum() < 100:
        return Xr

    def ols(i):
        zm = Z[i].mean(0)
        D = np.column_stack([np.ones(i.sum()), Z[i] - zm])
        return {"B": np.linalg.lstsq(D, X[i], rcond=None)[0][1:], "zm": zm}

    def fit():
        return {"full": ols(pre), "loso": {lab: ols(pre & (sess != lab)) for lab in labs_pre},
                "xy_sha1": pat._digest(X[pre], Z[pre])}
    spec = _spec(an, f"{arm}_{w}", f"reg-{name}-x", labs_pre, X.shape[1] + Z.shape[1],
                 {"xy_sha1": pat._digest(X[pre], Z[pre]), "sess": sess[pre]})   # "-x": no source is a prefix of another (siblings glob)
    P, status = fm.load_or_fit(spec, fit, meta={"script": "face_prelaunch_aim.py", "what": f"resid {name}"}, log=log)
    for lab in np.unique(sess):
        i = np.flatnonzero((sess == lab) & ok)
        if not len(i):
            continue
        m = P["loso"][lab] if (lab in P["loso"] and (meta.epoch.to_numpy()[i] == "pre").all()) else P["full"]
        Xr[i] = X[i] - (Z[i] - m["zm"]) @ m["B"]
    return Xr


def run_decoder(an, arm, name, X, meta, nperm=0, train_mask=None, log=print):
    """S26's run_decoder (pat.run_decoder) with kind face_aim. Pre = LOSO sibling, post = all-pre model."""
    from wfield_local import frozen_models as fm
    tm = (meta.epoch == "pre").to_numpy() if train_mask is None else train_mask
    ok = np.isfinite(X).all(1) & np.isfinite(meta.theta_L.to_numpy(float))
    pre = tm & ok
    labs_pre = sorted(set(meta.label[pre]))
    row = {"animal": an, "arm": arm, "decoder": name, "n_pre": int(pre.sum()), "n_pre_sess": len(labs_pre),
           "n_all": int(ok.sum())}
    pred = np.full(len(meta), np.nan)
    if len(labs_pre) < pat.MIN_PRE_SESS or pre.sum() < pat.MIN_PRE_LICKS:
        row["skipped"] = "too few pre"
        return pred, row
    Xp, y = X[pre], meta.theta_L.to_numpy(float)[pre]
    sess = meta.label.to_numpy(str)[pre]
    cell = np.char.add(np.char.add(sess, "|"), meta.target.to_numpy(str)[pre])
    dig = pat._digest(Xp, y, cell)
    fam, w = (name.split(":") + [""])[:2]
    spec = _spec(an, f"{arm}_{w}", f"{fam}-x", labs_pre, X.shape[1], {"xy_sha1": dig})

    def fit():
        L = pat.loso_multi(Xp, y[:, None], sess, "std")
        _ai, full, _p, _z, _pc = pat._fit_block(Xp, y[:, None], sess, "std", 0)
        return {"full": full, "loso": L["models"], "xy_sha1": dig}
    P, status = fm.load_or_fit(spec, fit, meta={"script": "face_prelaunch_aim.py", "decoder": name, "arm": arm,
                                                "n_trials": int(len(y))}, log=log)
    if P.get("xy_sha1") != dig:
        log(f"[frozen] !! {an} {arm} {name}: stored model trained on DIFFERENT data (same pre sessions); stored used")
        row["hash_mismatch"] = True
    row["frozen"] = status
    sall = meta.label.to_numpy(str)
    for lab in np.unique(sall):
        i = np.flatnonzero((sall == lab) & ok)
        if not len(i):
            continue
        m = P["loso"][lab] if (lab in P["loso"] and tm[i].all()) else P["full"]
        pred[i] = X[i] @ m["w"] + m["b"]
    row["alpha_full"] = P["full"]["alpha"]
    row["alpha_edge_frac"] = float(np.mean([m["alpha"] in (pat.ALPHAS[0], pat.ALPHAS[-1]) for m in P["loso"].values()]))
    pp = pred[pre]
    row["r_pool"] = pat._r(pp, y)
    row["r_sess_mean"] = float(np.nanmean([pat._r(pp[sess == s], y[sess == s]) for s in labs_pre]))
    row["partial_r"] = pat._r(pat._cent(pp, cell), pat._cent(y, cell))
    if nperm:
        rng = np.random.default_rng([AN.index(an), ARMS.index(arm), zlib.crc32(name.encode()) % 99991])
        Ys = [y] + [pat._perm_within(y, sess, rng) for _ in range(nperm)] + \
            [pat._perm_within(y, cell, rng) for _ in range(nperm)]
        L = pat.loso_multi(Xp, np.column_stack(Ys), sess, "std")
        Pr = L["pred"]
        rs = np.array([pat._r(Pr[:, j], Ys[j]) for j in range(1, nperm + 1)])
        pr = np.array([pat._r(pat._cent(Pr[:, j], cell), pat._cent(Ys[j], cell)) for j in range(nperm + 1, 2 * nperm + 1)])
        row.update(null95=float(np.nanpercentile(rs, 95)), p_gate=float((1 + np.sum(rs >= row["r_pool"])) / (1 + nperm)),
                   partial_null95=float(np.nanpercentile(pr, 95)),
                   p_partial=float((1 + np.sum(pr >= row["partial_r"])) / (1 + nperm)))
    return pred, row


def within_spout(an, arm, name, X, meta, nperm, log=print) -> dict:
    """Q4: within-spout decoder on PRE licks: y and X centred within session x target cell, LOSO, null = permutations
    of y within cell (not frozen: never scores post)."""
    pre = (meta.epoch == "pre").to_numpy() & np.isfinite(X).all(1) & np.isfinite(meta.theta_L.to_numpy(float))
    row = {"animal": an, "arm": arm, "decoder": name, "n_pre": int(pre.sum())}
    sess = meta.label.to_numpy(str)[pre]
    if len(set(sess)) < pat.MIN_PRE_SESS or pre.sum() < pat.MIN_PRE_LICKS:
        row["skipped"] = "too few pre"
        return row
    cell = np.char.add(np.char.add(sess, "|"), meta.target.to_numpy(str)[pre])
    keep = pd.Series(cell).map(pd.Series(cell).value_counts()).to_numpy() >= pat.MIN_CELL
    sess, cell = sess[keep], cell[keep]
    Xc = pat._cent(X[pre][keep], cell)
    y = pat._cent(meta.theta_L.to_numpy(float)[pre][keep], cell)
    rng = np.random.default_rng([AN.index(an), ARMS.index(arm), zlib.crc32(("ws" + name).encode()) % 99991])
    Ys = [y] + [pat._perm_within(y, cell, rng) for _ in range(nperm)]
    L = pat.loso_multi(Xc, np.column_stack(Ys), sess, "std")
    r0 = pat._r(L["pred"][:, 0], y)
    rs = np.array([pat._r(L["pred"][:, j], Ys[j]) for j in range(1, nperm + 1)])
    row.update(r_within=r0, null95=float(np.nanpercentile(rs, 95)) if nperm else np.nan,
               p_within=float((1 + np.sum(rs >= r0)) / (1 + nperm)) if nperm else np.nan, n_used=int(len(y)))
    return row


def fit_animal(an: str) -> dict:
    t0 = time.time()
    nperm = int(_arg("--nperm", 200))
    FITD.mkdir(parents=True, exist_ok=True)
    logf = open(FITD / f"{an}_log.txt", "w")

    def log(m):
        print(f"[{an}] {m}", flush=True)
        logf.write(m + "\n")
        logf.flush()
    from wfield_local import frozen_models as fm
    log(f"frozen local_dir = {fm.local_dir()} exists={fm.local_dir().exists()}")
    D = load_animal(an)
    log(f"{len(D['labels'])} sessions; load {time.time() - t0:.0f} s")
    gate, within = [], []
    for arm in ARMS:
        S = D[arm]
        meta = S["meta"].copy()
        Fm = S["fam"]
        Pd = {}

        def go(name, X, nperm_=0, arm=arm, meta=meta, Pd=Pd):
            p, row = run_decoder(an, arm, name, X, meta, nperm_, log=log)
            Pd[name] = p
            gate.append(row)
            log(f"{arm} {name}: n {row.get('n_pre')} r {row.get('r_pool', np.nan):.3f} p {row.get('p_gate', np.nan)} "
                f"partial {row.get('partial_r', np.nan):.3f} ({row.get('frozen', row.get('skipped'))}) {time.time() - t0:.0f} s")
        wins = ("PRE", "PRE10", "POST") if arm != "jaw" else ("PRE",)
        for w in wins:
            Xw = {k: _window(v, w) for k, v in Fm.items()}
            Xw["PTS_bj"] = frozen_resid(an, arm, w, "PTS_bj", Xw["PTS"], Xw["JAW"], meta, log)
            Xw["FRall_bj"] = frozen_resid(an, arm, w, "FRall_bj", Xw["FRall"], Xw["JAW"], meta, log)
            for k in GATED:
                go(f"{k}:{w}", Xw[k], nperm)
            if w == "PRE":
                for k in Xw:
                    if k not in GATED:
                        go(f"{k}:{w}", Xw[k], 0)
                for k in MAIN:
                    go(f"{k}:PREnb", _window(Fm[k], "PRE", base=False), 0)
            # Q4 within-spout (pre only)
            for k in ("PTS", "JAW", "FRall", "NOSE", "JAW3D", "PTS_bj"):
                r = within_spout(an, arm, f"{k}:{w}", Xw[k], meta, nperm, log)
                within.append(r)
                log(f"{arm} WITHIN {k}:{w}: r {r.get('r_within', np.nan):.3f} p {r.get('p_within', np.nan)}")
            del Xw
        for k in SINGLE:
            for b_ in range(NB):
                go(f"{k}:bin{b_:02d}", _bin_feats(Fm[k], b_), 0)
        for k, v in Pd.items():
            meta[f"p_{k}"] = v
        meta.to_csv(FITD / f"{an}_preds_{arm}.csv.gz", index=False, float_format="%.6g")
    pd.DataFrame(gate).to_csv(FITD / f"{an}_gate.csv", index=False)
    pd.DataFrame(within).to_csv(FITD / f"{an}_within.csv", index=False)
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
        if not os.environ.get("WIDEFIELD_FROZEN_MODEL_DIR") or not os.environ.get("FPA_OUT"):
            raise SystemExit("--smoke needs WIDEFIELD_FROZEN_MODEL_DIR and FPA_OUT")
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


# ================================================================================================ summarise
def _load_preds(arm: str) -> pd.DataFrame:
    return pd.concat([pd.read_csv(FITD / f"{a}_preds_{arm}.csv.gz", low_memory=False) for a in AN
                      if (FITD / f"{a}_preds_{arm}.csv.gz").exists()], ignore_index=True)


def analysis(name, arm, D, pcol, sel=None, store=None):
    cal_full = None
    if sel is not None:
        _l, cal_full = pat.track_licks(D, pcol, "theta_L", "target")
    L = D if sel is None else D[sel]
    L, cal = pat.track_licks(L, pcol, "theta_L", "target", cal_from=cal_full)
    S = pat.sessions_of(L)
    tag = {"analysis": name, "arm": arm, "decoder": pcol[2:]}
    per, pooled = pat.tracking_stats(S, cal, tag)
    if store is not None:
        store[(name, arm, pcol[2:])] = {"S": S, "cal": cal}
    return per, pooled, [{**tag, "animal": a, **c} for a, c in cal.items()]


def recovery(S: pd.DataFrame, tag: dict) -> list:
    """Q3: per-animal r over POST sessions + within-animal MixedLM (log day)."""
    import statsmodels.formula.api as smf
    import iti_licks as _il
    R = pd.read_csv(_il.OUT_W / "recmark_sessions.csv", usecols=["label", "day", "aim", "iti_ang"]).drop_duplicates("label")
    G = S.merge(R, on="label", how="left")
    G = G[(G.epoch != "pre") & (G.day > 0)].dropna(subset=["d_decs"]).copy()
    G["logday"] = np.log(G.day)
    out = []
    for xv, yv in (("d_exe", "d_decs"), ("aim", "d_decs"), ("iti_ang", "d_decs"), ("d_decs", "d_exe")):
        g = G.dropna(subset=[xv, yv]).copy()
        if g.animal.nunique() < 3:
            continue
        rr = {a: (np.corrcoef(h[xv], h[yv])[0, 1] if len(h) > 2 else np.nan) for a, h in g.groupby("animal")}
        g["w"] = g[xv] - g.groupby("animal")[xv].transform("mean")
        g["b"] = g.groupby("animal")[xv].transform("mean")
        g["yy"] = g[yv]
        try:
            fit = smf.mixedlm("yy ~ w + b + logday", g, groups=g["animal"]).fit(reml=False)
            slope, p = fit.params["w"], fit.pvalues["w"]
        except Exception:                                          # noqa: BLE001
            slope, p = np.nan, np.nan
        out.append({**tag, "x": xv, "y": yv, "n_sessions": len(g), "slope_within": slope, "p_within": p,
                    "n_pos": int(sum(v > 0 for v in rr.values() if np.isfinite(v))),
                    **{f"r_{a}": rr.get(a, np.nan) for a in AN}})
    return out


def cmd_summarise() -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT / "x.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    logf = open(OUT / "summarise_log.txt", "w")

    def P(m):
        print(m, flush=True)
        logf.write(m + "\n")
    G = pd.concat([pd.read_csv(FITD / f"{a}_gate.csv") for a in AN if (FITD / f"{a}_gate.csv").exists()], ignore_index=True)
    W = pd.concat([pd.read_csv(FITD / f"{a}_within.csv") for a in AN if (FITD / f"{a}_within.csv").exists()], ignore_index=True)
    G["pass"] = G.p_gate < 0.05
    gate_rows = []
    for (arm, dec), g in G.dropna(subset=["p_gate"]).groupby(["arm", "decoder"]):
        gate_rows.append({"arm": arm, "decoder": dec, "n_pass": int(g["pass"].sum()), "n": len(g),
                          "gate_pass": bool(g["pass"].sum() >= 3),
                          **{f"r_{r.animal}": r.r_pool for r in g.itertuples()},
                          **{f"p_{r.animal}": r.p_gate for r in g.itertuples()},
                          **{f"partial_{r.animal}": r.partial_r for r in g.itertuples()}})
    GT = pd.DataFrame(gate_rows)
    GT.to_csv(OUT / "gate.csv", index=False)
    G.to_csv(OUT / "gate_all.csv", index=False)
    P("== GATE (pooled LOSO r vs within-session permutations; pass p < .05 in >= 3/4)\n" + GT.round(3).to_string(index=False))
    WG = []
    for (arm, dec), g in W.dropna(subset=["p_within"]).groupby(["arm", "decoder"]):
        WG.append({"arm": arm, "decoder": dec, "n_pass": int((g.p_within < 0.05).sum()),
                   "n_pass_bonf": int((g.p_within < 0.05 / 3).sum()), "n": len(g),
                   **{f"r_{r.animal}": r.r_within for r in g.itertuples()},
                   **{f"p_{r.animal}": r.p_within for r in g.itertuples()}})
    WT = pd.DataFrame(WG)
    WT.to_csv(OUT / "within_spout.csv", index=False)
    P("== Q4 WITHIN-SPOUT (pre; r vs within-cell permutations)\n" + WT.round(3).to_string(index=False))
    G[["animal", "arm", "decoder", "frozen", "alpha_full", "alpha_edge_frac"]].to_csv(OUT / "frozen_status.csv", index=False)
    P("== frozen status: " + repr(G.groupby("frozen").size().to_dict()))
    per_all, pooled_all, cal_all, rec = [], [], [], []
    store = {}
    Dc = {arm: _load_preds(arm) for arm in ARMS}
    decs = sorted({c[2:] for arm in ARMS for c in Dc[arm].columns if c.startswith("p_")})
    for arm in ARMS:
        D = Dc[arm]
        for dec in decs:
            if f"p_{dec}" not in D or D[f"p_{dec}"].notna().sum() == 0:
                continue
            pe, po, ca = analysis("all", arm, D, f"p_{dec}", store=store)
            per_all += pe
            pooled_all += po
            cal_all += ca
            if dec.endswith(":PRE") and dec.split(":")[0] in ("PTS", "JAW", "FRall", "PTS_bj", "FRall_bj"):
                rec += recovery(store[("all", arm, dec)]["S"], {"arm": arm, "decoder": dec})
                if arm == "cued":                               # far contra (calibration from the full set)
                    pe, po, _c = analysis("far_contra", arm, D, f"p_{dec}", sel=(D.target == "far_contra").to_numpy())
                    per_all += pe
                    pooled_all += po
    PA, PR, CAL, REC = pd.DataFrame(per_all), pd.DataFrame(pooled_all), pd.DataFrame(cal_all), pd.DataFrame(rec)
    PA.to_csv(OUT / "per_animal.csv", index=False)
    PR.to_csv(OUT / "tracking.csv", index=False)
    CAL.to_csv(OUT / "calibration.csv", index=False)
    REC.to_csv(OUT / "recovery.csv", index=False)
    cols = ["animal", "epoch", "n_sessions", "n_licks", "d_decs", "d_dec", "d_exe", "b_sp", "calibrated"]
    for arm, dec in (("cued", "PTS:PRE"), ("cued", "JAW:PRE"), ("iti", "PTS:PRE"), ("jaw", "PTS:PRE"), ("jaw", "JAW:PRE"),
                     ("cued", "FRall:PRE"), ("iti", "JAW:PRE"), ("iti", "FRall:PRE")):
        t = PA[(PA.analysis == "all") & (PA.arm == arm) & (PA.decoder == dec)]
        P(f"\n== PER ANIMAL {arm} {dec}\n" + t[cols].round(3).to_string(index=False))
    # primary families
    fam = {"A": [("cued", "PTS:PRE", "acute"), ("cued", "PTS:PRE", "subacute"), ("cued", "JAW:PRE", "acute"),
                 ("cued", "JAW:PRE", "subacute"), ("iti", "PTS:PRE", "acute"), ("iti", "PTS:PRE", "subacute")],
           "B": [("jaw", "PTS:PRE", "acute"), ("jaw", "PTS:PRE", "subacute"), ("jaw", "JAW:PRE", "acute"),
                 ("jaw", "JAW:PRE", "subacute")]}
    hrows = []
    for f, cells in fam.items():
        rows = []
        for i, (arm, dec, ep) in enumerate(cells):
            r = PR[(PR.analysis == "all") & (PR.arm == arm) & (PR.decoder == dec) & (PR.epoch == ep)]
            g = GT[(GT.arm == arm) & (GT.decoder == dec)]
            post = GT[(GT.arm == arm) & (GT.decoder == dec.replace(":PRE", ":POST"))]
            rr = r.iloc[0].to_dict() if len(r) else {}
            rows.append({"family": f, "cell": f"{f}{i + 1}", "arm": arm, "decoder": dec, "epoch": ep,
                         "gate_pass": bool(g.gate_pass.iloc[0]) if len(g) else None,
                         "post_pass": bool(post.gate_pass.iloc[0]) if len(post) else None,
                         **{k: rr.get(k, np.nan) for k in ("n_animals", "n_uncalibrated", "d_decs", "d_decs_lo", "d_decs_hi",
                                                           "d_exe", "d_exe_lo", "d_exe_hi", "d_dec_raw", "TI", "TI_lo",
                                                           "TI_hi", "p_num", "p_full", "estimable", "n_same_sign")}})
        H = pd.DataFrame(rows)
        H["p_holm"] = ak.holm(H.p_num.to_numpy(float))
        H["tracking"] = (((H.d_decs_lo > 0) | (H.d_decs_hi < 0)) & (H.n_same_sign >= 3) & (H.p_holm < 0.05)
                         & (H.gate_pass == True))                                    # noqa: E712
        hrows.append(H)
    HH = pd.concat(hrows, ignore_index=True)
    HH.to_csv(OUT / "primary_holm.csv", index=False)
    P("\n== PRIMARY families A (tongue-referenced) / B (pre-jaw), TI numerator, Holm within family\n" + HH.round(3).to_string(index=False))
    show = PR[(PR.analysis == "all") & PR.decoder.str.endswith((":PRE", ":PRE10", ":POST", ":PREnb"))]
    P("\n== all decoders, pooled TI\n" + show[["arm", "decoder", "epoch", "n_animals", "n_uncalibrated", "d_decs", "d_decs_lo",
                                                "d_decs_hi", "d_exe", "TI", "TI_lo", "TI_hi", "p_num", "p_full",
                                                "n_same_sign"]].round(3).to_string(index=False))
    P("\n== Q3 recovery\n" + REC.round(3).to_string(index=False))
    # per-bin
    BINR = []
    for arm in ARMS:
        for k in SINGLE:
            for b_ in range(NB):
                dec = f"{k}:bin{b_:02d}"
                for r in PR[(PR.analysis == "all") & (PR.arm == arm) & (PR.decoder == dec)].itertuples():
                    gg = G[(G.arm == arm) & (G.decoder == dec)]
                    BINR.append({"arm": arm, "family": k, "bin": b_, "t": CENT[b_], "epoch": r.epoch, "TI": r.TI,
                                 "TI_lo": r.TI_lo, "TI_hi": r.TI_hi, "d_decs": r.d_decs, "d_decs_lo": r.d_decs_lo,
                                 "d_decs_hi": r.d_decs_hi, "n_same_sign": r.n_same_sign, "n_animals": r.n_animals,
                                 "n_uncal": r.n_uncalibrated, "r_mean": gg.r_pool.mean(),
                                 "leak": "fwhm" if (k == "FRall" and abs(CENT[b_]) <= LEAK_FWHM) else
                                 ("support" if (k == "FRall" and abs(CENT[b_]) <= LEAK_SUPPORT) else "")})
    BR = pd.DataFrame(BINR)
    BR.to_csv(OUT / "perbin.csv", index=False)
    P("\n== per-bin TI (acute)\n" + BR[BR.epoch == "acute"].round(3).to_string(index=False))
    # reproduction of S26 face decoder (cached features, S26 frozen model)
    s26 = pd.read_csv(pat.OUT / "secondary_pooled.csv") if (pat.OUT / "secondary_pooled.csv").exists() else pd.DataFrame()
    if len(s26):
        P("\n== S26 face:PRE rows (reference)\n" + s26[s26.analysis.astype(str).str.contains("face")].round(3).to_string(index=False))
    # mean jaw-opening / lateral traces (pre licks) per arm for the time-course figure (behaviour)
    traces = {}
    for arm in ARMS:
        rows = []
        for a in AN:
            ff = []
            D_ = Dc[arm][(Dc[arm].animal == a) & (Dc[arm].epoch == "pre")]
            labs = sorted(D_.label.unique())
            for lab in labs:
                y = np.load(FEAT / f"{lab}.npz")[f"{arm}_jaw"]
                ff.append(np.nanmean(y, 0))
            if ff:
                rows.append(np.nanmean(np.stack(ff), 0))
        traces[arm] = np.stack(rows) if rows else None
    FD.mkdir(parents=True, exist_ok=True)
    fd = {"GT": GT, "G": G, "WT": WT, "W": W, "PA": PA, "PR": PR, "CAL": CAL, "HH": HH, "BR": BR, "REC": REC,
          "traces": traces, "sessions": {k: v["S"] for k, v in store.items() if k[2].endswith(":PRE") and
                                         k[2].split(":")[0] in ("PTS", "JAW", "FRall")}}
    with open(FD / "face_prelaunch_aim.pkl", "wb") as f:
        pickle.dump(fd, f)
    logf.close()
    plot_all()


# ================================================================================================ figures
AM = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}


def _epc():
    from wfield_local import orofacial_cohort as oc
    return oc.EPOCH_LINE


def plot_all() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from within_lick_correction import reserve_headroom, star_top
    with open(FD / "face_prelaunch_aim.pkl", "rb") as f:
        fd = pickle.load(f)
    EC = _epc()
    FIGS.mkdir(parents=True, exist_ok=True)
    GT, G, PR, WT, W = fd["GT"], fd["G"], fd["PR"], fd["WT"], fd["W"]
    fams = ["PTS", "JAW", "FRall", "NOSE", "JAW3D", "PTS_bj", "FRall_bj"]
    flab = {"PTS": "points PCs", "JAW": "cam4 jaw", "FRall": "FaceRhythm (leak-exp.)", "NOSE": "nose", "JAW3D": "3-D jaw",
            "PTS_bj": "points | jaw", "FRall_bj": "FR | jaw"}
    # ---------------- primary figure
    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(3, 7, width_ratios=[1, 0.18, 1, 0.18, 1, 0.18, 1], hspace=0.6, wspace=0.05)
    axA = fig.add_subplot(gs[0, 0:3])
    xs = np.arange(len(fams))
    for _j, (arm, w, dx, col) in enumerate((("cued", "PRE", -0.27, "k"), ("iti", "PRE", -0.09, "#7570b3"),
                                           ("jaw", "PRE", 0.09, "#1b9e77"), ("cued", "POST", 0.27, "0.6"))):
        for i, fm_ in enumerate(fams):
            g = G[(G.arm == arm) & (G.decoder == f"{fm_}:{w}")]
            for r in g.itertuples():
                axA.scatter(i + dx, r.r_pool, marker=AM[r.animal], facecolor="none", edgecolor=col, s=26, lw=1)
                axA.plot([i + dx - 0.05, i + dx + 0.05], [r.null95] * 2, color=col, lw=0.8)
            t = GT[(GT.arm == arm) & (GT.decoder == f"{fm_}:{w}")]
            if len(t) and t.gate_pass.iloc[0]:
                star_top(axA, i + dx, "*", color=col, fontsize=10)
    axA.set_xticks(xs, [flab[f] for f in fams], fontsize=8, rotation=20)
    axA.axhline(0, color="k", lw=0.5)
    axA.set_ylabel("pre LOSO r (angle)")
    axA.set_title("A  pre-stroke gate: r per animal (open), bar = null 95th; * = 3/4 pass\n"
                  "black cued PRE, purple ITI PRE, green pre-JAW, grey cued POST (control)", fontsize=9, loc="left")
    # B / C / D: TI by epoch per family
    for k, (arm, title, gcol) in enumerate((("cued", "B  CUED, tongue-referenced PRE", 4), ("iti", "C  ITI docked PRE", 6),
                                            ("jaw", "D  CUED, pre-JAW window", 0))):
        ax = fig.add_subplot(gs[0 if k < 2 else 1, gcol if k < 2 else 0])
        ax.set_xlim(-0.6, 3.6)
        for i, fm_ in enumerate(["PTS", "JAW", "FRall", "PTS_bj"]):
            for e, ep in enumerate(EPS[1:]):
                x = i + (e - 1) * 0.25
                r = PR[(PR.analysis == "all") & (PR.arm == arm) & (PR.decoder == f"{fm_}:PRE") & (PR.epoch == ep)]
                if not len(r) or not np.isfinite(r.TI.iloc[0]) or r.n_animals.iloc[0] < 3:
                    continue
                r = r.iloc[0]
                ax.errorbar(x, np.clip(r.TI, -2, 3), yerr=[[np.clip(r.TI - r.TI_lo, 0, 5)], [np.clip(r.TI_hi - r.TI, 0, 5)]],
                            fmt="o", color=EC[ep], ms=6, capsize=2, zorder=4)
                for a in AN:
                    v = r.get(f"TI_{a}", np.nan)
                    if np.isfinite(v):
                        ax.scatter(x + 0.08, np.clip(v, -2, 3), marker=AM[a], facecolor="none", edgecolor=EC[ep], s=18, lw=0.8)
                if (r.d_decs_lo > 0 or r.d_decs_hi < 0) and r.n_same_sign >= 3 and r.n_animals >= 3:
                    star_top(ax, x, "*", color=EC[ep], fontsize=10)
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(1, color="0.6", lw=0.5, ls="--")
        ax.set_xticks(range(4), ["points", "jaw", "FR", "points|jaw"], fontsize=8)
        ax.set_ylim(-2.1, 3.1)
        ax.set_ylabel("tracking index (0 = target, 1 = launch)", fontsize=8)
        ax.set_title(title + "\nfilled = pooled [95 % CI] (>= 3 calibrated animals)\nopen = animals; * numerator CI excl. 0, >= 3/4", fontsize=8.5,
                     loc="left")
    # E: Q5 offset (from offset outputs if present)
    axE = fig.add_subplot(gs[1, 2:5])
    of = OFFD / "pooled.csv"
    if of.exists():
        O = pd.read_csv(of)
        O = O[O.variant == "A_FR"]
        for j, meas in enumerate(("iti_dock_sameplace", "cued_matched_error")):
            for e, ep in enumerate(EPS[1:]):
                r = O[(O.measure == meas) & (O.epoch == ep)]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = j * 4 + e
                for dx, (pt, lo, hi, mfc) in ((-0.15, (r.d_before, r.d_before_lo, r.d_before_hi, "none")),
                                              (0.15, (r.d_after, r.d_after_lo, r.d_after_hi, EC[ep]))):
                    axE.errorbar(x + dx, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", mfc=mfc, mec=EC[ep], color=EC[ep],
                                 ms=6, capsize=2)
                if r.p_holm < 0.05:
                    star_top(axE, x, "*", color=EC[ep])
        axE.set_xticks([0, 1, 2, 4, 5, 6], ["acute", "subacute", "chronic"] * 2, fontsize=8)
        axE.text(1, -0.12, "ITI docked same-place", transform=axE.get_xaxis_transform(), ha="center", fontsize=8)
        axE.text(5, -0.12, "cued matched-error", transform=axE.get_xaxis_transform(), ha="center", fontsize=8)
    axE.axhline(0, color="k", lw=0.5)
    axE.set_ylabel("decoder offset change vs pre (mm)")
    axE.set_title("E  frozen-decoder offset (change vs pre): BEFORE (open) vs AFTER face partialling\n"
                  "(filled; FR 11 traces, pre-fit regression); * = Holm p < .05 for after - before", fontsize=8.5, loc="left")
    # F: Q4 within-spout
    axF = fig.add_subplot(gs[1, 6])
    wf = ["PTS", "JAW", "FRall", "NOSE", "JAW3D", "PTS_bj"]
    for i, fm_ in enumerate(wf):
        for dx, (arm, w, col) in zip((-0.3, -0.1, 0.1, 0.3), (("cued", "PRE10", "k"), ("jaw", "PRE", "#1b9e77"),
                                                            ("iti", "PRE10", "#7570b3"), ("cued", "POST", "0.6"))):
            g = W[(W.arm == arm) & (W.decoder == f"{fm_}:{w}")]
            for r in g.itertuples():
                axF.scatter(i + dx, r.r_within, marker=AM[r.animal], facecolor="none", edgecolor=col, s=20, lw=0.9)
            t = WT[(WT.arm == arm) & (WT.decoder == f"{fm_}:{w}")]
            if len(t) and t.n_pass.iloc[0] >= 3:
                star_top(axF, i + dx, "*", color=col, fontsize=9)
    axF.axhline(0, color="k", lw=0.5)
    axF.set_xticks(range(len(wf)), ["pts", "jaw", "FR", "nose", "3D", "pts|jaw"], fontsize=7, rotation=30)
    axF.set_title("F  within-spout r (pre)\nblack cued PRE10, green pre-jaw,\npurple ITI PRE10, grey POST", fontsize=8, loc="left")
    # G: per-bin time course (cued, acute) PTS / JAW / FR
    BR = fd["BR"]
    for k, (arm, ref) in enumerate((("cued", "tongue"), ("jaw", "JAW"), ("iti", "tongue"))):
        ax = fig.add_subplot(gs[2, 2 * k])
        for fm_, col in (("PTS", "#1f78b4"), ("JAW", "#33a02c"), ("FRall", "#e31a1c")):
            b = BR[(BR.arm == arm) & (BR.family == fm_) & (BR.epoch == "acute")].sort_values("t").copy()
            if not len(b):
                continue
            few = b.n_animals < 3                                  # bins with < 3 calibrated animals are not drawn
            b.loc[few, ["TI", "TI_lo", "TI_hi"]] = np.nan
            ax.plot(b.t, np.clip(b.TI, -2, 3), color=col, lw=2, label=fm_)
            ax.fill_between(b.t, np.clip(b.TI_lo, -2, 3), np.clip(b.TI_hi, -2, 3), color=col, alpha=0.12, lw=0)
        ax.axvspan(-LEAK_FWHM, LEAK_FWHM, color="#e31a1c", alpha=0.05, lw=0)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylim(-2.1, 3.1)
        ax.set_xlabel(f"bin centre (s from {ref} onset)")
        ax.set_ylabel("acute TI (single-bin decoders)", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.set_title(f"{'GHI'[k]}  per-bin acute TI, {arm} (bins with >= 3 calibrated animals)\n"
                     "pink band = FaceRhythm leak half-width (+-0.35 s)", fontsize=8.5, loc="left")
    h = [Line2D([], [], marker=AM[a], ls="", mfc="none", mec="k", label=a) for a in AN] + \
        [Line2D([], [], marker="o", ls="", color=EC[e], label=e) for e in EPS[1:]]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("What in the face carries the pre-launch aim (BEHAVIOUR), when, and the cortical offset after face "
                 "partialling (NEURAL, E). n = 4, PROVISIONAL", fontsize=11)
    for ax in fig.get_axes():
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    reserve_headroom(fig)
    save_png_svg(fig, FIGS / "fpa_primary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    plot_secondary(fd)


def plot_secondary(fd) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from within_lick_correction import reserve_headroom, star_top
    EC = _epc()
    G, GT, PR = fd["G"], fd["GT"], fd["PR"]
    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.4, 0.12, 1, 0.12, 1], hspace=0.5, wspace=0.05)
    # A: FR group decomposition (cued PRE gate r, single groups + drop-one)
    ax = fig.add_subplot(gs[0, 0])
    decs = ["FR_lick", "FR_left", "FR_right", "FR_side", "FR_close", "FR_far", "FR_other"] + \
        [f"FR_o{k}" for k in FR_OTHER] + [f"FRdrop_{n}" for n in ("lick", "left", "right", "close", "far")] + ["FRall", "NOSE"]
    for i, d in enumerate(decs):
        for dx, arm, col in ((-0.15, "cued", "k"), (0.15, "iti", "#7570b3")):
            g = G[(G.arm == arm) & (G.decoder == f"{d}:PRE")]
            for r in g.itertuples():
                ax.scatter(i + dx, r.r_pool, marker=AM[r.animal], facecolor="none", edgecolor=col, s=18, lw=0.8)
            t = GT[(GT.arm == arm) & (GT.decoder == f"{d}:PRE")]
            if len(t) and t.gate_pass.iloc[0]:
                star_top(ax, i + dx, "*", color=col, fontsize=9)
    ax.set_xticks(range(len(decs)), [d.replace("FR_", "").replace("FRdrop_", "-") .replace("o0", FR_LAB["o0"])
                                     .replace("o10", FR_LAB["o10"]).replace("o1", FR_LAB["o1"]).replace("o3", FR_LAB["o3"])
                                     .replace("o6", FR_LAB["o6"]).replace("o9", FR_LAB["o9"]) for d in decs],
                  rotation=60, fontsize=7, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("pre LOSO r (PRE window)")
    ax.set_title("A  FaceRhythm decomposition (leak-exposed)\nsingle groups, '-' = drop one; black cued, purple ITI",
                 fontsize=9, loc="left")
    # B: acute TI by group (cued)
    ax = fig.add_subplot(gs[0, 2:5])
    for i, d in enumerate(decs):
        for e, ep in enumerate(EPS[1:]):
            r = PR[(PR.analysis == "all") & (PR.arm == "cued") & (PR.decoder == f"{d}:PRE") & (PR.epoch == ep)]
            if not len(r) or not np.isfinite(r.TI.iloc[0]) or r.n_animals.iloc[0] < 3:
                continue
            r = r.iloc[0]
            x = i + (e - 1) * 0.25
            ax.errorbar(x, np.clip(r.TI, -2, 3), yerr=[[np.clip(r.TI - r.TI_lo, 0, 5)], [np.clip(r.TI_hi - r.TI, 0, 5)]],
                        fmt="o", color=EC[ep], ms=4, capsize=1.5)
    ax.set_xticks(range(len(decs)), [d.replace("FR_", "").replace("FRdrop_", "-") for d in decs], rotation=60, fontsize=7,
                  ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1, color="0.6", lw=0.5, ls="--")
    ax.set_ylim(-2.1, 3.1)
    ax.set_title("B  CUED PRE tracking index by FR group\npooled, 95 % CI; acute / subacute / chronic", fontsize=9, loc="left")
    # C: Q3 recovery scatter, cued PTS / JAW
    for k, dec in enumerate(("PTS:PRE", "JAW:PRE")):
        ax = fig.add_subplot(gs[1, 2 * k])
        S = fd["sessions"].get(("all", "cued", dec))
        CAL = fd["CAL"]
        okA = set(CAL[(CAL.arm == "cued") & (CAL.decoder == dec) & (CAL.calibrated == True)].animal)  # noqa: E712
        if S is not None:
            S = S[S.animal.isin(okA)]
            for r in S.itertuples():
                ax.scatter(r.d_exe, np.clip(r.d_decs, -40, 40), marker=AM[r.animal], facecolor="none",
                           edgecolor=EC[r.epoch], s=22, lw=0.9)
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_xlabel("launch deviation at the target (deg, session)")
        ax.set_ylabel("face-decoded deviation (deg-equiv., session)")
        ax.set_title(f"{'CD'[k]}  CUED {dec.split(':')[0]}: sessions (open), colour = epoch\ncalibrated animals only: "
                     + ", ".join(sorted(okA)), fontsize=9, loc="left")
    # E: pre jaw-opening trace (behaviour) to time the jaw
    ax = fig.add_subplot(gs[1, 4])
    for arm, col in (("cued", "k"), ("iti", "#7570b3"), ("jaw", "#1b9e77")):
        T = fd["traces"].get(arm)
        if T is None:
            continue
        for row in T:
            ax.plot(CENT, row[:, 1], color=col, lw=0.6, alpha=0.35)
        ax.plot(CENT, np.nanmean(T[:, :, 1], 0), color=col, lw=2.2, label=arm)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("s from reference onset (tongue; green: jaw)")
    ax.set_ylabel("cam4 jaw y vs rest (px; rises = opening)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("E  when the jaw opens (pre; faint = animals)", fontsize=9, loc="left")
    for a_ in fig.get_axes():
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.suptitle("Face pre-launch aim: FaceRhythm decomposition, recovery, jaw timing (BEHAVIOUR; n = 4; PROVISIONAL)", fontsize=11)
    reserve_headroom(fig)
    save_png_svg(fig, FIGS / "fpa_secondary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ================================================================================================ Q5 offset
def main() -> int:
    # coordinator 11-04: routed (outputs under the run tag), no longer refused in a tagged run
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = int(_arg("--jobs", 4))
    if "--plot-only" in sys.argv or cmd == "plot":
        plot_all()
        if (OFFD / "pooled.csv").exists():
            import face_offset_partial as fop
            fop.plot()
        return 0
    if cmd == "extract":
        cmd_extract(jobs)
    elif cmd == "fit":
        cmd_fit(jobs)
    elif cmd == "summarise":
        cmd_summarise()
    elif cmd == "offset":
        import face_offset_partial as fop
        return fop.run(jobs)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
