"""RE-EMBEDDED OLD CODE OR A NEW CODE? (lit item 2.3; Gallego 2020 Nat Neurosci, Degenhart 2020 Nat Biomed Eng, Makin &
Krakauer 2023 eLife). Pre-registration: docs/status/REEMBEDDING_PREREG_20261030.md (design fixed there before any
post-stroke number; read it first -- this docstring only maps it to code).

Per session, cued FIRST licks (iti_licks session files: first_peth onset-aligned, first_cue cue-aligned), 45 split-basis
Allen areas / frame SD, centred on the calibration half. k = 10 PCs of the lick-window (-0.5..+0.75 s) activity.
Three decoders scored on the same held-out half (2 folds of interleaved 10-trial blocks):
  (i)   FROZEN  pre PCs + frozen pre decoder (frozen_models.load_or_fit, kind 'reembed'; pre sessions -> LOSO siblings)
  (ii)  ALIGNED own PCs -> CCA map (Gallego) of the label-free trial-averaged trajectories (lick -0.5..0.75 + cue
        -0.5..2.0, pooled over spouts) onto the pre reference trajectories -> the SAME frozen decoder
        (secondary: orthogonal Procrustes)
  (iii) REFIT   own PCs, same bins / C, trained on the calibration half
Primary target spout (6 classes, balanced accuracy above chance); secondary side_ml (ridge r). Families: lick (4 lick
bins, PRIMARY) and cue (3 cue bins). Licks: matched (side_ml AND reach within the animal's pre 2.5-97.5 % for that
spout; PRIMARY) and all. Controls: time-shuffled alignment (G2), label-permuted (G3), fold split-half (G4), random
orthogonal area rotation of pre sessions (C5, code check).
Stats: per-animal table first (rule 8); analysis_kit.boot_delta vs pre->pre; index = dA / dG from paired draws.
Outputs (only) under <E>/reembedding/: sessions_cache.pkl, reembed_sessions.csv, reembed_animals.csv,
reembed_tests.csv, reembed_gate.csv, reembed_areas.csv, figdata/*.pkl, figures/*.png.
Usage: python scripts/reembedding_test.py [--jobs 3] [--reload]   (load + run + summarise + plot)
       python scripts/reembedding_test.py --summarise                (stats + figures from reembed_sessions.csv)
       python scripts/reembedding_test.py --plot-only                (figures from figdata only)
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from fr_musall import E  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402
from wfield_local.writeguard import assert_writable  # noqa: E402

OUT = E / "reembedding"
FIGD = OUT / "figdata"
FIG = OUT / "figures"
CACHE = OUT / "sessions_cache.pkl"
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic", "june"]
POST = ["acute", "subacute", "chronic", "june"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
ORO = ("ALM", "tjM1", "SSp-m", "SSp-n", "SSs")
JUNE_DIR = il.OUT / "june_ref" / il.SESS_NAME
TIMES = il.OUT / "cued_first_lick_times.csv"
# ---- fixed by the pre-registration
K = 10
C_LOG = 0.1
ALPHA_SIDE = 10.0
LICK_BINS = ((-0.2, 0.0), (0.0, 0.15), (0.15, 0.3), (0.3, 0.5))
CUE_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5))
WIN_L = (-0.5, 0.75)          # lick-aligned window: PCA samples + lick trajectory
WIN_C = (-0.5, 2.0)           # cue-aligned trajectory window
BLOCK = 10
Q = (2.5, 97.5)
MATCH_S = 0.03
MIN_CLS = 3                   # licks per spout in a scored half for the spout to count
MIN_SPOUTS = 3
MIN_FOLD = 15                 # scored licks per half
SEED = 20261030
LAG_L = il.LAGS_P[(il.LAGS_P >= WIN_L[0] - 1e-9) & (il.LAGS_P <= WIN_L[1] + 1e-9)]
LAG_C = il.LAGS_C[(il.LAGS_C >= WIN_C[0] - 1e-9) & (il.LAGS_C <= WIN_C[1] + 1e-9)]
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
EPCOL = {"pre": "#555555", "acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4", "june": "#8c8c8c"}
DEC = {"i": "(i) frozen", "ii": "(ii) aligned (CCA)", "iii": "(iii) refit"}


# ================================================================================================ load
def _kin(label: str, t: np.ndarray) -> pd.DataFrame:
    """side_ml, reach of licks at DAQ onsets t: nearest cam4 onset within MATCH_S (lick_vigor_perspout._kin rule)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s"])
    Kt = il.lick_table()
    Kt = Kt[Kt.session == f"{an}_{date}"].drop_duplicates("lick_uid").set_index("lick_uid")
    on = A.daq_on_s.to_numpy(float)
    out = pd.DataFrame(np.nan, index=range(len(t)), columns=["side", "reach"])
    if not len(on) or not len(t):
        return out
    o = np.argsort(on)
    j = np.clip(np.searchsorted(on[o], t), 1, len(on) - 1)
    jj = np.where(np.abs(on[o][j - 1] - t) <= np.abs(on[o][j] - t), j - 1, j)
    u = np.where(np.abs(on[o][jj] - t) <= MATCH_S, o[jj], np.nan)
    ok = np.isfinite(u)
    out.loc[ok, ["side", "reach"]] = Kt.reindex(u[ok].astype(int))[["side_ml", "reach_mm"]].to_numpy(float)
    return out


def _session(item):
    """(label, sess_dir, times or None) -> per-lick arrays (module-level for fan_sessions)."""
    from wfield_local.epochs import days_since_stroke
    label, sdir, tt = item
    sdir = Path(sdir) if sdir else None
    z = il._load(label, sdir)
    rt = z["first_rt"].astype(float)
    if tt is not None and len(tt) == len(rt):
        fl = np.asarray(tt, float)
    else:
        _c, fl = il._first_lick_times(label, rt)
    names = il.area_names(z["cols"])
    sd = il.area_sd(sdir).loc[label, names].to_numpy(float)
    il_ = np.isin(np.round(il.LAGS_P, 3), np.round(LAG_L, 3))
    ic_ = np.isin(np.round(il.LAGS_C, 3), np.round(LAG_C, 3))
    Xl = il.area_features(z["first_peth"][:, il_].astype(float), z["cols"], label, sdir) / sd
    Xc = il.area_features(z["first_cue"][:, ic_].astype(float), z["cols"], label, sdir) / sd
    kin = _kin(label, fl)
    ok = np.isfinite(fl) & np.isfinite(Xl).all((1, 2)) & np.isfinite(Xc).all((1, 2)) & np.isin(z["first_pos"], POS)
    day = days_since_stroke(label)
    return {"label": label, "animal": str(z["animal"]), "epoch": "june" if sdir else str(z["epoch"]),
            "day": np.nan if day is None else float(day), "names": names, "t": fl[ok],
            "pos": z["first_pos"].astype(str)[ok], "side": kin.side.to_numpy()[ok], "reach": kin.reach.to_numpy()[ok],
            "Xl": Xl[ok].astype(np.float32), "Xc": Xc[ok].astype(np.float32)}


def load(jobs: int, reload: bool = False) -> dict:
    if CACHE.exists() and not reload:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    cur = set(ak.curated_labels())
    T = pd.read_csv(TIMES, float_precision="round_trip")
    TT = {x: g.sort_values("row").t.to_numpy(float) for x, g in T.groupby("label")}
    main = [x for x in il.labels() if x in cur and x[:4] in AN]
    june = [p.stem for p in sorted(JUNE_DIR.glob("*.npz")) if p.stem in cur and p.stem[:4] in AN]
    items = [(x, "", TT.get(x)) for x in main] + [(x, str(JUNE_DIR), None) for x in june]
    res, fails = ak.fan_sessions(items, _session, jobs=jobs, key=lambda it: (it[1], it[0]), label="reembed load")
    if fails:
        raise RuntimeError(f"failed sessions: {[f[0][0] for f in fails]}: {fails[0][1]}")
    S = {r["label"]: r for _it, r in res}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(S, f, protocol=pickle.HIGHEST_PROTOCOL)
    return S


# ================================================================================================ model pieces
def _bins(Z: np.ndarray, lags: np.ndarray, bins) -> np.ndarray:
    """(n, lags, k) -> (n, k * n_bins), bin-major (feature index b * k + j)."""
    return np.concatenate([Z[:, (lags >= a - 1e-9) & (lags < b - 1e-9)].mean(1) for a, b in bins], 1)


def _pca(Xs: np.ndarray, k: int = K) -> np.ndarray:
    """Centred samples (n, F) -> (F, k) top-k PCs, sign fixed (largest |loading| positive)."""
    C = Xs.T @ Xs / max(len(Xs) - 1, 1)
    w, v = np.linalg.eigh(C)
    P = v[:, np.argsort(w)[::-1][:k]]
    s = np.sign(P[np.argmax(np.abs(P), 0), np.arange(k)])
    return P * s


def _clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C_LOG, max_iter=3000))


def _ridge():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA_SIDE))


def _traj(Zl: np.ndarray, Zc: np.ndarray) -> np.ndarray:
    """label-free trial-averaged latent trajectories (lick window, then cue window), (T, k), centred over time."""
    T = np.vstack([Zl.mean(0), Zc.mean(0)])
    return T - T.mean(0)


def cca_map(Lb: np.ndarray, La: np.ndarray) -> np.ndarray:
    """Gallego 2020: k x k map M with Lb @ M ~ La (canonical weights B A^-1)."""
    Qa, Ra = np.linalg.qr(La)
    Qb, Rb = np.linalg.qr(Lb)
    U, _s, Vt = np.linalg.svd(Qa.T @ Qb)
    A = np.linalg.solve(Ra, U)
    B = np.linalg.solve(Rb, Vt.T)
    return B @ np.linalg.inv(A)


def proc_map(Lb: np.ndarray, La: np.ndarray) -> np.ndarray:
    U, _s, Vt = np.linalg.svd(Lb.T @ La)
    return U @ Vt


def build_ref(sessions: list[dict]) -> dict:
    """Pre reference from pre sessions (each centred on its own lick-window mean): PCs, decoders, trajectories."""
    mus = [s["Xl"].reshape(-1, s["Xl"].shape[-1]).mean(0) for s in sessions]
    P = _pca(np.vstack([(s["Xl"] - m).reshape(-1, len(m)) for s, m in zip(sessions, mus)]))
    Zl = [(s["Xl"] - m) @ P for s, m in zip(sessions, mus)]
    Zc = [(s["Xc"] - m) @ P for s, m in zip(sessions, mus)]
    y = np.concatenate([s["pos"] for s in sessions])
    side = np.concatenate([s["side"] for s in sessions])
    Fl = np.vstack([_bins(z, LAG_L, LICK_BINS) for z in Zl])
    Fc = np.vstack([_bins(z, LAG_C, CUE_BINS) for z in Zc])
    clf_l, clf_c = _clf().fit(Fl, y), _clf().fit(Fc, y)
    ok = np.isfinite(side)
    rid = _ridge().fit(Fl[ok], side[ok])
    traj = np.mean([_traj(a, b) for a, b in zip(Zl, Zc)], 0)
    return {"P": P, "clf_lick": clf_l, "clf_cue": clf_c, "ridge": rid, "traj": traj}


def frozen_ref(an: str, pre: list[dict]) -> dict:
    """RULE 10: the animal's pre reference (+ LOSO siblings) via frozen_models.load_or_fit."""
    from wfield_local import frozen_models as fm
    labs = sorted(s["label"] for s in pre)
    pre = sorted(pre, key=lambda s: s["label"])
    h = hashlib.sha1()
    for s in pre:
        h.update(np.ascontiguousarray(s["Xl"]).tobytes() + np.ascontiguousarray(s["Xc"]).tobytes()
                 + "|".join(s["pos"]).encode())
    dig = h.hexdigest()
    spec = fm.make_spec(an, "reembed", align="first_lick", source=f"areas_split_r{il.ROI_VERSION}_k{K}",
                        train_labels=labs, post_s=WIN_L[1], n_features=len(pre[0]["names"]))

    def fit():
        return {"full": build_ref(pre), "loso": {x: build_ref([s for s in pre if s["label"] != x]) for x in labs},
                "xy_sha1": dig}
    P, status = fm.load_or_fit(spec, fit, meta={"script": "reembedding_test.py", "k": K, "C": C_LOG,
                                                "lick_bins": LICK_BINS, "cue_bins": CUE_BINS})
    print(f"[frozen] {an} reembed: {status}", flush=True)
    if P.get("xy_sha1") != dig:
        print(f"[frozen] !! {an}: stored reference trained on DIFFERENT data than today's load -- stored one used",
              flush=True)
    return P


# ================================================================================================ scoring
def _balacc(y, p):
    cls, n = np.unique(y, return_counts=True)
    keep = cls[n >= MIN_CLS]
    if len(keep) < MIN_SPOUTS:
        return np.nan
    m = np.isin(y, keep)
    if m.sum() < MIN_FOLD:
        return np.nan
    acc = np.mean([np.mean(p[m][y[m] == c] == c) for c in keep])
    return acc - 1.0 / len(keep)


def _r(y, p):
    ok = np.isfinite(y)
    if ok.sum() < MIN_FOLD or np.std(y[ok]) == 0 or np.std(p[ok]) == 0:
        return np.nan
    return float(np.corrcoef(y[ok], p[ok])[0, 1])


def _folds(t: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(t))
    return (r // BLOCK) % 2


def _share(W: np.ndarray) -> np.ndarray:
    """(F, m) effective area weights -> per-area share of the summed squared weight."""
    e = (W ** 2).sum(1)
    return e / e.sum() if e.sum() > 0 else e * np.nan


def _lat_weights(clf, k: int) -> np.ndarray:
    """Decoder weights per latent: (k, classes * bins) (standardisation folded in)."""
    sc, lr = clf[0], clf[-1]
    Wf = lr.coef_ / sc.scale_[None, :]                   # (classes, k * bins)
    nb = Wf.shape[1] // k
    return np.concatenate([Wf[:, b * k:(b + 1) * k].T for b in range(nb)], 1)


def score_session(s: dict, R: dict, keep: np.ndarray, rng, rot: np.ndarray | None = None) -> dict:
    """The three decoders + controls on one session (2-fold); keep = the analysed licks (matched / all)."""
    folds = _folds(s["t"])
    Xl, Xc = s["Xl"].astype(float), s["Xc"].astype(float)
    if rot is not None:
        Xl, Xc = Xl @ rot, Xc @ rot
    y, side = s["pos"], s["side"]
    res = {k_: [] for k_ in ("i", "ii", "iii", "ii_proc", "ii_tshuf", "perm_i", "perm_ii", "perm_iii",
                             "c_i", "c_ii", "c_iii", "s_i", "s_ii", "s_iii")}
    shares = []
    for f in (0, 1):
        cal, sc = keep & (folds != f), keep & (folds == f)
        if cal.sum() < MIN_FOLD or sc.sum() < MIN_FOLD:
            for v in res.values():
                v.append(np.nan)
            continue
        mu = Xl[cal].reshape(-1, Xl.shape[-1]).mean(0)
        Lc, Cc = Xl - mu, Xc - mu
        P = R["P"]
        Ps = _pca(Lc[cal].reshape(-1, Lc.shape[-1]))
        own_l, own_c = Lc @ Ps, Cc @ Ps
        M = cca_map(_traj(own_l[cal], own_c[cal]), R["traj"])
        Mp = proc_map(_traj(own_l[cal], own_c[cal]), R["traj"])
        Tb = _traj(own_l[cal], own_c[cal])
        Ms = cca_map(Tb[rng.permutation(len(Tb))], R["traj"])
        F_i = _bins(Lc @ P, LAG_L, LICK_BINS)
        F_own = _bins(own_l, LAG_L, LICK_BINS)
        F_ii, F_pr, F_sh = (_bins(own_l @ m_, LAG_L, LICK_BINS) for m_ in (M, Mp, Ms))
        clf = R["clf_lick"]
        res["i"].append(_balacc(y[sc], clf.predict(F_i[sc])))
        res["ii"].append(_balacc(y[sc], clf.predict(F_ii[sc])))
        res["ii_proc"].append(_balacc(y[sc], clf.predict(F_pr[sc])))
        res["ii_tshuf"].append(_balacc(y[sc], clf.predict(F_sh[sc])))
        refit = _clf().fit(F_own[cal], y[cal])
        res["iii"].append(_balacc(y[sc], refit.predict(F_own[sc])))
        # G3 label-permuted (within each half)
        yp = y.copy()
        for m_ in (cal, sc):
            ix = np.flatnonzero(m_)
            yp[ix] = y[rng.permutation(ix)]
        res["perm_i"].append(_balacc(yp[sc], clf.predict(F_i[sc])))
        res["perm_ii"].append(_balacc(yp[sc], clf.predict(F_ii[sc])))
        res["perm_iii"].append(_balacc(yp[sc], _clf().fit(F_own[cal], yp[cal]).predict(F_own[sc])))
        # cue family
        G_i = _bins(Cc @ P, LAG_C, CUE_BINS)
        G_own = _bins(own_c, LAG_C, CUE_BINS)
        G_ii = _bins(own_c @ M, LAG_C, CUE_BINS)
        res["c_i"].append(_balacc(y[sc], R["clf_cue"].predict(G_i[sc])))
        res["c_ii"].append(_balacc(y[sc], R["clf_cue"].predict(G_ii[sc])))
        res["c_iii"].append(_balacc(y[sc], _clf().fit(G_own[cal], y[cal]).predict(G_own[sc])))
        # side_ml (secondary target)
        okc = cal & np.isfinite(side)
        res["s_i"].append(_r(side[sc], R["ridge"].predict(F_i[sc])))
        res["s_ii"].append(_r(side[sc], R["ridge"].predict(F_ii[sc])))
        res["s_iii"].append(_r(side[sc], _ridge().fit(F_own[okc], side[okc]).predict(F_own[sc]))
                            if okc.sum() >= MIN_FOLD else np.nan)
        # spatial readout: effective area weights of the aligned frozen decoder
        Wl = _lat_weights(clf, K)
        shares.append(_share(Ps @ M @ Wl))
    out = {k_: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k_, v in res.items()}
    for k_ in ("i", "ii", "iii"):
        out[f"{k_}_f0"], out[f"{k_}_f1"] = res[k_][0], res[k_][1]
    out["n"] = int(keep.sum())
    out["share"] = np.nanmean(shares, 0) if shares else None
    return out


def matched_mask(S: dict, an: str) -> dict:
    """{label: bool per lick}: pre = licks with kinematics; others = side AND reach inside the animal's pre 2.5-97.5 %
    for that spout."""
    pre = [s for s in S.values() if s["animal"] == an and s["epoch"] == "pre"]
    P = pd.DataFrame({"pos": np.concatenate([s["pos"] for s in pre]), "side": np.concatenate([s["side"] for s in pre]),
                      "reach": np.concatenate([s["reach"] for s in pre])}).dropna()
    lo, hi = P.groupby("pos").quantile(Q[0] / 100), P.groupby("pos").quantile(Q[1] / 100)
    out = {}
    for x, s in S.items():
        if s["animal"] != an:
            continue
        fin = np.isfinite(s["side"]) & np.isfinite(s["reach"])
        if s["epoch"] == "pre":
            out[x] = fin
            continue
        m = fin.copy()
        for v in ("side", "reach"):
            a_ = lo[v].reindex(s["pos"]).to_numpy(float)
            b_ = hi[v].reindex(s["pos"]).to_numpy(float)
            m &= (s[v] >= a_) & (s[v] <= b_)
        out[x] = m
    return out


def run(S: dict) -> tuple[pd.DataFrame, dict]:
    rows, shares = [], {}
    names = next(iter(S.values()))["names"]
    rng0 = np.random.default_rng(SEED)
    Qrot = np.linalg.qr(rng0.standard_normal((len(names), len(names))))[0]
    ref_share = {}
    for an in AN:
        sess = {x: s for x, s in sorted(S.items()) if s["animal"] == an}
        pre = [s for s in sess.values() if s["epoch"] == "pre"]
        if len(pre) < 3:
            print(f"{an}: {len(pre)} pre sessions -- skipped", flush=True)
            continue
        Pz = frozen_ref(an, pre)
        ref_share[an] = _share(Pz["full"]["P"] @ _lat_weights(Pz["full"]["clf_lick"], K))
        mm = matched_mask(sess, an)
        for x, s in sess.items():
            R = Pz["loso"][x] if s["epoch"] == "pre" else Pz["full"]
            for lk, keep in (("matched", mm[x]), ("all", np.ones(len(s["pos"]), bool))):
                rng = np.random.default_rng([SEED, int(hashlib.md5(f"{x}{lk}".encode()).hexdigest()[:8], 16)])
                o = score_session(s, R, keep, rng)
                sh = o.pop("share")
                rows.append({"animal": an, "label": x, "epoch": s["epoch"], "day": s["day"], "licks": lk,
                             "n_all": len(s["pos"]), **o})
                if sh is not None:
                    shares[(x, lk)] = sh
                if s["epoch"] == "pre" and lk == "matched":       # C5: random orthogonal area rotation
                    o2 = score_session(s, R, keep, np.random.default_rng([SEED, 5]), rot=Qrot)
                    rows[-1]["rot_i"], rows[-1]["rot_ii"] = o2["i"], o2["ii"]
        print(f"{an}: {len(sess)} sessions scored", flush=True)
    D = pd.DataFrame(rows)
    return D, {"shares": shares, "ref_share": ref_share, "names": names}


# ================================================================================================ summarise
def _dict(D, ep, col):
    g = D[(D.epoch == ep)].dropna(subset=[col])
    return {a: g[g.animal == a][col].to_list() for a in AN if (g.animal == a).any()}


def _valid(D):
    return D.dropna(subset=["i", "ii", "iii"]).assign(A=lambda d: d.ii - d.i, R=lambda d: d.iii - d.ii,
                                                      G=lambda d: d.iii - d.i, A_tshuf=lambda d: d.ii_tshuf - d.i,
                                                      A_proc=lambda d: d.ii_proc - d.i, G_proc=lambda d: d.iii - d.i,
                                                      cA=lambda d: d.c_ii - d.c_i, cG=lambda d: d.c_iii - d.c_i,
                                                      sA=lambda d: d.s_ii - d.s_i, sG=lambda d: d.s_iii - d.s_i)


def summarise(D: pd.DataFrame, X: dict) -> dict:
    out = {}
    for lk in ("matched", "all"):
        V = _valid(D[D.licks == lk])
        # ---- per-animal table FIRST (rule 8)
        A = (V.groupby(["animal", "epoch"])[["i", "ii", "iii", "A", "R", "G", "ii_tshuf", "ii_proc", "c_i", "c_ii", "c_iii",
                                            "s_i", "s_ii", "s_iii"]].mean().reset_index())
        A["n_sess"] = V.groupby(["animal", "epoch"]).size().to_numpy()
        A["ratio_raw"] = A.A / A.G
        for a in AN:
            p = A[(A.animal == a) & (A.epoch == "pre")]
            if len(p):
                for c in ("A", "G", "R"):
                    A.loc[A.animal == a, f"d{c}"] = A.loc[A.animal == a, c] - float(p[c].iloc[0])
        A["index"] = A.dA / A.dG
        A["epoch"] = pd.Categorical(A.epoch, EPS, ordered=True)
        A = A.sort_values(["epoch", "animal"])
        print(f"\n===== PER-ANIMAL ({lk} licks): above-chance balanced accuracy, spout, lick family =====")
        with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
            print(A[["animal", "epoch", "n_sess", "i", "ii", "iii", "A", "R", "G", "dA", "dR", "dG", "index",
                     "ratio_raw"]].to_string(index=False))
        A.assign(licks=lk).to_csv(OUT / f"reembed_animals_{lk}.csv", index=False)
        # ---- boot_delta vs pre
        tests = []
        for ep in POST:
            for col in ("i", "ii", "iii", "A", "R", "G", "A_tshuf", "A_proc", "c_i", "c_ii", "c_iii", "cA", "cG",
                        "s_i", "s_ii", "s_iii", "sA", "sG"):
                post, pre = _dict(V, ep, col), _dict(V, "pre", col)
                iv = ak.boot_delta(post, pre, np.random.default_rng(SEED))
                if iv is None:
                    continue
                per = {a: np.mean(post[a]) - np.mean(pre[a]) for a in post if a in pre}
                nup = sum(v > 0 for v in per.values())
                tests.append({"licks": lk, "epoch": ep, "q": col, "delta": iv[0] if isinstance(iv, tuple) else iv.point,
                              "lo": _lo(iv), "hi": _hi(iv), "n_animals": len(per), "n_up": nup,
                              **{f"d_{a}": per.get(a, np.nan) for a in AN}})
            # index from PAIRED draws (same seed, same session sets -> same animal / session draw)
            for nA, nG, tag in (("A", "G", "index"), ("cA", "cG", "c_index"), ("sA", "sG", "s_index"),
                                ("A_tshuf", "G", "index_tshuf"), ("A_proc", "G", "index_proc")):
                pa, qa = _dict(V, ep, nA), _dict(V, "pre", nA)
                pg, qg = _dict(V, ep, nG), _dict(V, "pre", nG)
                ia, da = ak.boot_delta(pa, qa, np.random.default_rng(SEED), draws=True)
                ig, dg = ak.boot_delta(pg, qg, np.random.default_rng(SEED), draws=True)
                if ia is None or ig is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = da / dg
                per = {a: (np.mean(pa[a]) - np.mean(qa[a])) / (np.mean(pg[a]) - np.mean(qg[a])) for a in pa if a in qa}
                tests.append({"licks": lk, "epoch": ep, "q": tag, "delta": _pt(ia) / _pt(ig),
                              "lo": np.nanpercentile(r, 2.5), "hi": np.nanpercentile(r, 97.5), "n_animals": len(per),
                              "n_up": sum(v > 0.5 for v in per.values()), **{f"d_{a}": per.get(a, np.nan) for a in AN}})
        T = pd.DataFrame(tests)
        T["change"] = ((T.lo > 0) | (T.hi < 0)) & (np.maximum(T.n_up, T.n_animals - T.n_up) >= 3)
        T.to_csv(OUT / f"reembed_tests_{lk}.csv", index=False)
        print(f"\n===== boot_delta vs pre->pre ({lk}) =====")
        with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
            print(T[T.q.isin(["i", "ii", "iii", "A", "R", "G", "index", "index_tshuf", "index_proc", "c_index",
                              "s_index"])][["epoch", "q", "delta", "lo", "hi", "n_up", "n_animals", "change",
                                            "d_PS92", "d_PS93", "d_PS94", "d_PS95"]].to_string(index=False))
        out[lk] = {"V": V, "A": A, "T": T}
    out["gate"] = gate(D)
    out["areas"] = areas(D, X)
    return out


def _pt(iv):
    return iv.point if hasattr(iv, "point") else iv[0]


def _lo(iv):
    return iv.lo if hasattr(iv, "lo") else iv[1]


def _hi(iv):
    return iv.hi if hasattr(iv, "hi") else iv[2]


def gate(D: pd.DataFrame) -> dict:
    V = _valid(D[D.licks == "matched"])
    pre = V[V.epoch == "pre"]
    g1 = pre.groupby("animal").apply(lambda g: g.ii.mean() / g.i.mean(), include_groups=False)
    post = V[V.epoch.isin(["acute", "subacute", "chronic"])]
    perm = D[D.licks == "matched"][["perm_i", "perm_ii", "perm_iii"]].mean()
    Vf = V.dropna(subset=["ii_f0", "ii_f1", "i_f0", "i_f1"])
    a0, a1 = Vf.ii_f0 - Vf.i_f0, Vf.ii_f1 - Vf.i_f1
    g4 = float(np.corrcoef(a0, a1)[0, 1]) if len(Vf) > 3 else np.nan
    rot = pre.dropna(subset=["rot_i"])
    G = {"G1_eff": g1.to_dict(), "G1_pass": bool((g1 >= 0.7).sum() >= 3),
         "G2_A_real_post": float(post.A.mean()), "G2_A_tshuf_post": float(post.A_tshuf.mean()),
         "G3_perm": perm.to_dict(), "G3_pass": bool((perm.abs() <= 0.03).all()),
         "G4_r_A": g4, "G4_n": int(len(Vf)), "G4_pass": bool(g4 >= 0.3),
         "C5_i": float(rot.i.mean()), "C5_rot_i": float(rot.rot_i.mean()), "C5_ii": float(rot.ii.mean()),
         "C5_rot_ii": float(rot.rot_ii.mean()), "C5_max_abs_ii_diff": float((rot.ii - rot.rot_ii).abs().max()),
         "fold_a0": a0.to_numpy(), "fold_a1": a1.to_numpy(), "fold_ep": Vf.epoch.to_numpy(),
         "pre_eff_sessions": (pre.ii / pre.i).to_numpy(), "pre_animal": pre.animal.to_numpy()}
    print("\n===== GATE =====")
    for k_, v in G.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k_}: {v}")
    pd.DataFrame([{k_: (str(v) if isinstance(v, dict) else v) for k_, v in G.items() if not isinstance(v, np.ndarray)}]
                 ).to_csv(OUT / "reembed_gate.csv", index=False)
    return G


def areas(D: pd.DataFrame, X: dict) -> dict:
    names = list(X["names"])
    hemi_r = np.array([n.endswith("_right") for n in names])
    oro = np.array([n.rsplit("_", 1)[0] in ORO for n in names])
    rows = []
    for (x, lk), sh in sorted(X["shares"].items()):
        if lk != "matched" or sh is None or not np.isfinite(sh).all():
            continue
        r = D[(D.label == x) & (D.licks == lk)].iloc[0]
        rows.append({"label": x, "animal": r.animal, "epoch": r.epoch, "contra": sh[hemi_r].sum(), "oro": sh[oro].sum(),
                     "oro_contra": sh[oro & hemi_r].sum(), "oro_ipsi": sh[oro & ~hemi_r].sum(),
                     **{n: v for n, v in zip(names, sh)}})
    Sdf = pd.DataFrame(rows)
    Sdf.to_csv(OUT / "reembed_areas_sessions.csv", index=False)
    tests = []
    for ep in POST:
        for col in ("contra", "oro", "oro_contra", "oro_ipsi"):
            post, pre = _dict(Sdf, ep, col), _dict(Sdf, "pre", col)
            iv = ak.boot_delta(post, pre, np.random.default_rng(SEED))
            if iv is None:
                continue
            per = {a: np.mean(post[a]) - np.mean(pre[a]) for a in post if a in pre}
            tests.append({"epoch": ep, "q": col, "delta": _pt(iv), "lo": _lo(iv), "hi": _hi(iv), "n_animals": len(per),
                          "n_up": sum(v > 0 for v in per.values()), **{f"d_{a}": per.get(a, np.nan) for a in AN}})
    T = pd.DataFrame(tests)
    T["change"] = ((T.lo > 0) | (T.hi < 0)) & (np.maximum(T.n_up, T.n_animals - T.n_up) >= 3)
    T.to_csv(OUT / "reembed_areas_tests.csv", index=False)
    print("\n===== aligned read-out area shares: change vs pre (matched) =====")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(Sdf.groupby(["animal", "epoch"])[["contra", "oro", "oro_contra", "oro_ipsi"]].mean().to_string())
        print(T.to_string(index=False))
    # per-area mean share per epoch (animal-weighted) and change vs pre
    M = Sdf.groupby(["animal", "epoch"])[names].mean().groupby("epoch").mean()
    ref = np.mean([v for v in X["ref_share"].values()], 0)
    return {"S": Sdf, "T": T, "M": M, "names": names, "ref": ref}


# ================================================================================================ figures
def _star_top(ax, x, text="*", color="k", row=0, step=0.075):
    """Mirrors wfield_local.sig_marks.star_top (Priya 10-30: stars in a reserved top band)."""
    try:
        from wfield_local.sig_marks import star_top
        return star_top(ax, x, text, color=color, row=row)
    except ImportError:
        import matplotlib.patheffects as pe
        ax.text(x, 0.99 - step * row, text, color=color, ha="center", va="top", fontsize=12, fontweight="bold",
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
        ax._wf_mark_rows = max(getattr(ax, "_wf_mark_rows", 0), row + 1)


def _reserve(fig, step=0.075, pad=0.03):
    try:
        from wfield_local.sig_marks import reserve_headroom
        return reserve_headroom(fig)
    except ImportError:
        for ax in fig.get_axes():
            n = getattr(ax, "_wf_mark_rows", 0)
            if n:
                lo, hi = ax.get_ylim()
                frac = step * n + pad
                ax.set_ylim(lo, hi + (hi - lo) * frac / (1 - frac))


def fig_data(R: dict, D: pd.DataFrame, X: dict) -> dict:
    lk = "matched"
    V, A, T = R[lk]["V"], R[lk]["A"], R[lk]["T"]
    lev = {}
    for col in ("i", "ii", "iii", "A", "R", "G", "c_i", "c_ii", "c_iii", "s_i", "s_ii", "s_iii"):
        for ep in EPS:
            g = V[V.epoch == ep].dropna(subset=[col])
            iv = ak.boot_ci(g.groupby("animal")[col].apply(list).to_dict(), np.random.default_rng(SEED)) \
                if len(g) else None
            lev[(col, ep)] = None if iv is None else (_pt(iv), _lo(iv), _hi(iv))
    fd = {"A": A, "T": T, "lev": lev, "gate": R["gate"], "areas": {k_: v for k_, v in R["areas"].items() if k_ != "S"},
          "areaS": R["areas"]["S"][["animal", "epoch", "contra", "oro", "oro_contra", "oro_ipsi"]],
          "A_all": R["all"]["A"], "T_all": R["all"]["T"]}
    try:
        from figdata import imaging_grid  # noqa: F401
        from frozen_area_maps import area_image
        img, nm = area_image(V.label.iloc[0])
        fd["img"], fd["img_names"] = img, nm
    except Exception as ex:  # noqa: BLE001
        print(f"area image unavailable ({ex}); map panel skipped", flush=True)
    return fd


def _ep_axis(ax, eps):
    ax.set_xticks(range(len(eps)))
    ax.set_xticklabels([e if e != "june" else "June\n(no stroke,\n~60 d)" for e in eps], fontsize=8)
    ax.set_xlim(-0.5, len(eps) - 0.5)


def draw_decoders(F: dict, path: Path, fam: str = "") -> None:
    import matplotlib.pyplot as plt
    A, lev = F["A"], F["lev"]
    eps = [e for e in EPS if (A.epoch == e).any()]
    cols = [f"{fam}{c}" for c in ("i", "ii", "iii")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, col, k_ in zip(axs, cols, ("i", "ii", "iii")):
        for a in AN:
            g = A[A.animal == a].set_index("epoch").reindex(eps)
            ax.plot(range(len(eps)), g[col].to_numpy(float), "-o", color=ACOL[a], lw=1, ms=4, alpha=0.8, label=a)
        for j, ep in enumerate(eps):
            v = lev.get((col, ep))
            if v:
                ax.errorbar(j + 0.18, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="s", color="k", ms=6, capsize=3,
                            zorder=5)
        ax.axhline(0, color="0.7", lw=0.8, ls=":")
        T = F["T"]
        for j, ep in enumerate(eps):
            r = T[(T.epoch == ep) & (T.q == col)]
            if len(r) and bool(r.change.iloc[0]):
                _star_top(ax, j, "*")
        _ep_axis(ax, eps)
        ax.set_title(DEC[k_] if k_ != "ii" else "(ii) aligned (CCA, label-free)", fontsize=10)
    axs[0].set_ylabel("spout decoding, balanced accuracy\nabove chance" if fam != "s_" else "r (decoded vs landing)")
    _reserve(fig)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fam_txt = {"": "first-lick window", "c_": "cue window", "s_": "landing side_ml (ridge r)"}[fam]
    fig.suptitle(f"Re-embedding test: three decoders on each session ({fam_txt}; kinematics-matched first licks)\n"
                 "lines = animals; black = pooled level (boot_ci); * = change vs pre->pre (boot_delta, CI excl. 0, "
                 ">= 3/4 animals)", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_index(F: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    A, T, lev = F["A"], F["T"], F["lev"]
    eps = [e for e in EPS if (A.epoch == e).any()]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, col, ttl in ((axs[0], "A", "gap closed by alignment  A = (ii) - (i)"),
                         (axs[1], "R", "gap NOT closed  R = (iii) - (ii)")):
        pre = A[A.epoch == "pre"][col]
        ax.axhspan(pre.min(), pre.max(), color="0.85", zorder=0, label="pre->pre range (animals)")
        for a in AN:
            g = A[A.animal == a].set_index("epoch").reindex(eps)
            ax.plot(range(len(eps)), g[col].to_numpy(float), "-o", color=ACOL[a], lw=1, ms=4, label=a)
        for j, ep in enumerate(eps):
            v = lev.get((col, ep))
            if v:
                ax.errorbar(j + 0.18, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="s", color="k", ms=6, capsize=3)
            r = T[(T.epoch == ep) & (T.q == col)]
            if len(r) and bool(r.change.iloc[0]):
                _star_top(ax, j, "*")
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        _ep_axis(ax, eps)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("balanced accuracy difference")
    ax = axs[2]
    pe = [e for e in eps if e != "pre"]
    for a in AN:
        g = A[A.animal == a].set_index("epoch").reindex(pe)
        ax.plot(range(len(pe)), np.clip(g["index"].to_numpy(float), -1.5, 2.5), "o", color=ACOL[a], ms=6)
    for j, ep in enumerate(pe):
        r = T[(T.epoch == ep) & (T.q == "index")]
        if len(r):
            lo, hi = np.clip([r.lo.iloc[0], r.hi.iloc[0]], -1.5, 2.5)
            ax.errorbar(j + 0.2, np.clip(r.delta.iloc[0], -1.5, 2.5), yerr=[[max(0, np.clip(r.delta.iloc[0], -1.5, 2.5) - lo)],
                        [max(0, hi - np.clip(r.delta.iloc[0], -1.5, 2.5))]], fmt="s", color="k", ms=6, capsize=3)
        rg = T[(T.epoch == ep) & (T.q == "G")]
        if len(rg) and not bool(rg.change.iloc[0]):
            ax.text(j, -1.4, "no extra\ngap (dG)", ha="center", va="bottom", fontsize=7, color="0.4")
    ax.axhspan(0.67, 2.5, color="#d9ead3", zorder=0)
    ax.axhspan(-1.5, 0.33, color="#f4cccc", zorder=0)
    ax.text(len(pe) - 0.55, 2.4, "re-embedded old code", ha="right", va="top", fontsize=8, color="#38761d")
    ax.text(len(pe) - 0.55, 0.28, "new code", ha="right", va="top", fontsize=8, color="#990000")
    ax.set_ylim(-1.5, 2.5)
    _ep_axis(ax, pe)
    ax.set_title("re-embedding index dA / dG (vs pre->pre)\nclipped to [-1.5, 2.5]; black = paired-draw CI",
                 fontsize=10)
    _reserve(fig)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Re-embedding index (kinematics-matched first licks, spout, first-lick window). Index interpreted only "
                 "when dG (frozen -> refit gap growth) is a change", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_areas(F: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    ar = F["areas"]
    names, M = ar["names"], ar["M"]
    fig = plt.figure(figsize=(13, 4.6))
    have_img = "img" in F

    def to_area(v):
        d = dict(zip(names, v))
        out = {}
        for h in ("left", "right"):
            for n in set(n_.rsplit("_", 1)[0] for n_ in names):
                if n in ("ALM", "MOs_rest"):
                    out[f"MOs_{h}"] = d.get(f"ALM_{h}", 0) + d.get(f"MOs_rest_{h}", 0)
                elif n in ("tjM1", "MOp_rest"):
                    out[f"MOp_{h}"] = d.get(f"tjM1_{h}", 0) + d.get(f"MOp_rest_{h}", 0)
                elif f"{n}_{h}" in d:
                    out[f"{n}_{h}"] = d[f"{n}_{h}"]
        return out
    panels = []
    if "pre" in M.index:
        panels.append(("pre: aligned read-out share\n(held-out pre sessions)", M.loc["pre", names].to_numpy(float), "lev"))
    for ep in ("chronic", "subacute"):
        if ep in M.index and "pre" in M.index:
            panels.append((f"{ep} - pre (share)", M.loc[ep, names].to_numpy(float) - M.loc["pre", names].to_numpy(float),
                           "chg"))
    if have_img:
        from frozen_area_maps import outline, paint
        img, nm = F["img"], F["img_names"]
        ol = outline(img)
        lv = max(np.nanmax(np.abs(p[1])) for p in panels if p[2] == "lev") if panels else 1
        cv = max([np.nanmax(np.abs(p[1])) for p in panels if p[2] == "chg"] or [1])
        for j, (ttl, v, kind) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels) + 1, j + 1)
            im = ax.imshow(paint(img, nm, to_area(v)), cmap="Oranges" if kind == "lev" else "RdBu_r",
                           vmin=0 if kind == "lev" else -cv, vmax=lv if kind == "lev" else cv)
            ax.imshow(ol, cmap="gray_r", alpha=0.6, vmin=0, vmax=1)
            ax.set_title(ttl, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.02, 0.02, "ipsilesional", transform=ax.transAxes, fontsize=7, va="bottom")
            ax.text(0.98, 0.02, "contralesional", transform=ax.transAxes, fontsize=7, va="bottom", ha="right")
            fig.colorbar(im, ax=ax, fraction=0.04)
    ax = fig.add_subplot(1, len(panels) + 1, len(panels) + 1)
    S, T = F["areaS"], ar["T"]
    eps = [e for e in EPS if (S.epoch == e).any()]
    for col, mk, ttl in (("contra", "o", "contralesional share"), ("oro", "s", "orofacial share")):
        g = S.groupby(["animal", "epoch"])[col].mean().unstack().reindex(columns=eps)
        for a in AN:
            if a in g.index:
                ax.plot(range(len(eps)), g.loc[a].to_numpy(float), "-" + mk, color=ACOL[a], lw=0.8, ms=4, alpha=0.8)
        ax.plot(range(len(eps)), g.mean().to_numpy(float), "-" + mk, color="k", lw=2, ms=6, label=ttl + " (mean)")
        for j, ep in enumerate(eps):
            r = T[(T.epoch == ep) & (T.q == col)]
            if len(r) and bool(r.change.iloc[0]):
                _star_top(ax, j, "*" if col == "contra" else "+", row=0 if col == "contra" else 1)
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    _ep_axis(ax, eps)
    ax.set_ylabel("share of aligned read-out weight")
    ax.set_title("group shares (animals)\n* contra, + orofacial: change vs pre", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    _reserve(fig)
    fig.suptitle("Where the ALIGNED frozen decoder reads the spout code: effective area weights (own PCs x CCA map x "
                 "frozen weights), share of squared weight; ALM / tjM1 drawn inside MOs / MOp on the map", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_gate(F: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    G = F["gate"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    ax = axs[0]
    for j, a in enumerate(AN):
        v = G["pre_eff_sessions"][G["pre_animal"] == a]
        ax.plot(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), v, "o", color=ACOL[a], ms=4, alpha=0.6)
        ax.plot(j, G["G1_eff"].get(a, np.nan), "_", color="k", ms=20, mew=2)
    ax.axhline(0.7, color="r", ls="--", lw=1)
    ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.set_xticks(range(len(AN)))
    ax.set_xticklabels(AN)
    ax.set_title(f"G1 alignment efficiency pre->pre\n(ii)/(i) above chance; pass = {G['G1_pass']}", fontsize=9)
    ax = axs[1]
    ax.bar([0, 1], [G["G2_A_real_post"], G["G2_A_tshuf_post"]], color=["#1f77b4", "0.6"], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CCA, time-locked", "CCA, time-shuffled"], fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("G2 gap closed A, post epochs\n(shuffled correspondence must not close it)", fontsize=9)
    ax = axs[2]
    p = G["G3_perm"]
    ax.bar(range(3), [p["perm_i"], p["perm_ii"], p["perm_iii"]], color="0.5", width=0.6)
    ax.axhspan(-0.03, 0.03, color="#d9ead3", zorder=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["(i)", "(ii)", "(iii)"])
    ax.set_title(f"G3 label-permuted: above chance\npass = {G['G3_pass']}", fontsize=9)
    ax = axs[3]
    for ep in EPS:
        m = G["fold_ep"] == ep
        ax.plot(G["fold_a0"][m], G["fold_a1"][m], "o", ms=3, color=EPCOL[ep], label=ep, alpha=0.7)
    ax.set_xlabel("A, half 0")
    ax.set_ylabel("A, half 1")
    ax.set_title(f"G4 split-half reliability of A\nr = {G['G4_r_A']:.2f} (n = {G['G4_n']}); pass = {G['G4_pass']}",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Stability gate (matched first licks). C5 code check: random area rotation of pre sessions -> (i) "
                 f"{G['C5_i']:.3f} -> {G['C5_rot_i']:.3f}, (ii) {G['C5_ii']:.3f} -> {G['C5_rot_ii']:.3f} "
                 f"(max |diff| {G['C5_max_abs_ii_diff']:.1e})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot(F: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    draw_decoders(F, FIG / "reembed_decoders.png")
    draw_decoders(F, FIG / "reembed_decoders_cue.png", "c_")
    draw_decoders(F, FIG / "reembed_decoders_side.png", "s_")
    draw_index(F, FIG / "reembed_index.png")
    draw_areas(F, FIG / "reembed_areas.png")
    draw_gate(F, FIG / "reembed_gate.png")
    print(f"figures -> {FIG}", flush=True)


# ================================================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        with open(FIGD / "reembed.pkl", "rb") as f:
            plot(pickle.load(f))
        return 0
    if a.summarise:
        D = pd.read_csv(OUT / "reembed_sessions.csv")
        with open(OUT / "reembed_shares.pkl", "rb") as f:
            X = pickle.load(f)
    else:
        S = load(min(a.jobs, 3), a.reload)
        print(f"{len(S)} sessions loaded", flush=True)
        D, X = run(S)
        D.to_csv(OUT / "reembed_sessions.csv", index=False)
        with open(OUT / "reembed_shares.pkl", "wb") as f:
            pickle.dump(X, f)
    R = summarise(D, X)
    F = fig_data(R, D, X)
    FIGD.mkdir(parents=True, exist_ok=True)
    with open(FIGD / "reembed.pkl", "wb") as f:
        pickle.dump(F, f)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
