"""PRE-LAUNCH HEMISPHERIC TUNING TO LAUNCH ANGLE (Priya 10-30; RECOVERY_MECHANISMS_LIT_20261030 1b "discriminating
analyses"): which hemisphere's PRE-LAUNCH cortical activity carries the lick's launch angle, and does that change after
the stroke? Contralesional take-over (routes A / B / D) predicts the RIGHT hemisphere gains IPSIVERSIVE (rightward)
tuning before launch; the ipsilesional bypass predicts the LEFT hemisphere gains weight; feedback / reafference predicts
changes only after contact.

DESIGN (fixed before any post-stroke number): docs/status/PRELICK_HEMISPHERE_TUNING_PREREG_20261030.md -- read it first.

SIGN CONVENTIONS (every table / figure):
  angle   launch angle theta_L from within_lick_correction (licks_scored.csv.gz; first 12 ms of protrusion), deg,
          + = CONTRALESIONAL (mouse right). within = theta_L minus the session x spout mean of the lick set.
  b       tuning slope, dF/F x 1e3 per 10 deg: + = MORE activity for more CONTRALESIONAL (rightward) launches.
          Pre-stroke contraversive tuning: left (ipsilesional) hemisphere b > 0, right (contralesional) b < 0.
  hemi    c = contralesional = RIGHT hemisphere (image right); i = ipsilesional = LEFT (image left). D = b_c - b_i.
  windows s from TONGUE onset (licks_all daq_on_s): BASE -0.80..-0.50 (subtracted), PRE -0.30..-0.05, POST +0.05..+0.30;
          PREJAW = PRE relative to JAW onset (jaw_onsets.csv).

Usage (bash env per the repo; outputs only under <v36>/imaging_encoding/prelick_hemisphere_tuning/):
  python scripts/prelick_hemisphere_tuning.py run [--jobs 3]     per-session extraction (cached npz) + per-session fits
  python scripts/prelick_hemisphere_tuning.py gate               August-pre stability gate + noise floor
  python scripts/prelick_hemisphere_tuning.py summarise          per-animal tables, boot_delta, Holm, recovery link
  python scripts/prelick_hemisphere_tuning.py plot  (or any subcommand with --plot-only)   figures from figdata only
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lick_correction as lc  # noqa: E402

V36 = "lp_r4_heatmap_20261013__clean_v36"
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
AREAS7 = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs", "MOp_rest", "MOs_rest"]
CTRL_A = ["RSPd", "VISpm"]
AREAS = AREAS7 + CTRL_A
POOLS = {"POOL": ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"], "MOTOR": ["ALM", "tjM1"], "SENS": ["SSp-m", "SSp-n", "SSs"],
         "CTRL": CTRL_A}
FEATS = list(POOLS) + AREAS
HEMI = {"c": "right", "i": "left"}
EDGES = np.round(np.arange(-0.80, 0.6001, 0.05), 3)
NB = len(EDGES) - 1
CENT = np.round((EDGES[:-1] + EDGES[1:]) / 2, 3)
FINE = np.round(np.arange(-0.80 + 0.005, 0.60, 0.01), 4)          # 5 samples per 0.05 s bin
BASE = (-0.80, -0.50)
WINS = {"PRE": (-0.30, -0.05), "POST": (0.05, 0.30)}
QUIET_FIRST, QUIET_ITI, SAME_LICK = 1.0, 2.0, 0.05
CUE_EXCL = (-2.0, 0.75)
MIN_N = 40
K_FOLD, N_REP, N_PERM = 5, 5, 20
SEED = 0
SETS = ("first", "later", "iti", "firstjaw")


def out_root() -> Path:
    from wfield_local import run_mode
    return run_mode.encoding_out_root(V36) / "prelick_hemisphere_tuning"     # routed (coordinator 11-04)


OUT = out_root()
SESS = OUT / "sessions"
FIGS = OUT / "figures"
FD = OUT / "figdata"
WLC = OUT.parent / "within_lick_correction"


def _bins_of(w: tuple) -> np.ndarray:
    return (EDGES[:-1] >= w[0] - 1e-9) & (EDGES[1:] <= w[1] + 1e-9)


# ================================================================================================ extraction
def _items() -> list:
    """(label, sid, licks_scored subset, jaw subset) per imaging session (curated, PS92-95, in licks_scored)."""
    from wfield_local import analysis_kit as ak
    L = pd.read_csv(WLC / "licks_scored.csv.gz", usecols=["sid", "label", "animal", "epoch", "lick_uid", "trial_id",
                                                          "lick_idx", "context", "pos", "onset_ok", "theta_L",
                                                          "reach", "speed_peak", "contact"])
    L = L[L.animal.isin(AN)]
    labs = sorted(set(lc.imaging_labels()) & set(ak.curated_labels()) & set(L.label.unique()))
    labs = [x for x in labs if x[:4] in AN]
    from wfield_local import run_mode
    J = pd.read_csv(run_mode.behaviour_qc_dir() / "jaw_onsets.csv", dtype={"date": str},
                    usecols=["session", "trial_id", "jaw_onset_ms", "onset_at_cue", "jaw3d_side_ipsi_mm"])
    items = []
    for lab in labs:
        g = L[L.label == lab]
        sid = g.sid.iloc[0]
        items.append(Item((lab, sid, g.reset_index(drop=True), J[J.session == sid].reset_index(drop=True))))
    return items


class Item(tuple):
    def __repr__(self):
        return str(self[0])


def _masks(parea, pstat, names, flat, shape) -> dict:
    import iti_licks as il
    M = il.roi_masks(parea, pstat, names, flat, shape)
    out = {}
    for h in ("left", "right"):
        for a in ("ALM", "tjM1", "SSp-m", "SSp-n", "SSs", "RSPd", "VISpm"):
            out[f"{a}_{h}"] = M[f"{a}_{h}"]
        out[f"MOp_rest_{h}"] = M[f"MOp_{h}"] & ~M[f"tjM1_{h}"]
        out[f"MOs_rest_{h}"] = M[f"MOs_{h}"] & ~M[f"ALM_{h}"]
    return out


def _binned(R, ft, t) -> np.ndarray:
    import iti_licks as il
    X = il._sample(R, ft, np.asarray(t, float), FINE)                 # (n, 140, cols)
    return X.reshape(len(t), NB, 5, X.shape[-1]).mean(2).astype(np.float32)


def extract(item) -> dict:
    """One imaging session -> SESS/<label>.npz: 0.05-s bin means (-0.8..+0.6 s from tongue / jaw onset) of the 18 area
    traces for each lick set, plus per-lick covariates."""
    label, sid, Ls, J = item
    f = SESS / f"{label}.npz"
    if f.exists() and "--redo" not in sys.argv:
        return {"label": label, "cached": True}
    t0 = time.time()
    import iti_licks as il
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = sid.split("_")
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = _masks(parea, pstat, names, flat, shape)
    cols = [f"{a}_{h}" for h in ("left", "right") for a in AREAS]
    Ra = np.stack([Ub[M[c]].mean(0) for c in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    del V, Ub
    B = il._behaviour(label)
    T = B.trials.set_index("trial_id")
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["trial_id", "lick_idx", "context", "segment", "daq_on_s"])
    A["lick_uid"] = np.arange(len(A))
    t_all = np.sort(np.concatenate([A.daq_on_s.to_numpy(float), np.asarray(B.lick_on_s, float)]))
    t_all = t_all[np.isfinite(t_all)]
    ev = np.sort(np.concatenate([T.cue_s.to_numpy(float), T.strobe_s.to_numpy(float)]))
    ev = ev[np.isfinite(ev)]
    lo, hi = ft[0] - EDGES[0] + 0.01, ft[-1] - EDGES[-1] - 0.01

    def prev_gap(t):
        j = np.searchsorted(t_all, t - SAME_LICK)
        return t - np.where(j > 0, t_all[np.maximum(j - 1, 0)], -np.inf)

    D = Ls[Ls.onset_ok.astype(bool) & np.isfinite(Ls.theta_L) & np.isfinite(Ls.reach)
           & np.isfinite(Ls.speed_peak)].copy()
    D = D.merge(A[["lick_uid", "daq_on_s", "segment"]], on="lick_uid", how="left", validate="one_to_one")
    D = D[np.isfinite(D.daq_on_s) & (D.daq_on_s > lo) & (D.daq_on_s < hi)].copy()
    D["t"] = D.daq_on_s.astype(float)
    D["engaged"] = D.trial_id.map(T.engaged).fillna(False).astype(bool)
    D["cue_s"] = D.trial_id.map(T.cue_s).astype(float)
    D["gap"] = prev_gap(D.t.to_numpy())
    out = {"cols": np.array(cols), "animal": an, "epoch": epochs.epoch_of(label), "edges": EDGES}
    n = {}
    cue = D[(D.context == "post_cue") & D.engaged]
    # FIRST cued licks: clean single-lick window (no other onset in the 1.0 s before tongue onset)
    F = cue[(cue.lick_idx == 0) & (cue.gap >= QUIET_FIRST)].copy()
    F["rt"] = F.t - F.cue_s
    F = F[np.isfinite(F.rt)]
    # LATER cued licks: previous lick (same trial) interval + launch angle
    Lt = cue[cue.lick_idx >= 1].copy()
    allc = A[A.context == "post_cue"].set_index(["trial_id", "lick_idx"]).daq_on_s
    allc = allc[~allc.index.duplicated()]
    key = list(zip(Lt.trial_id, Lt.lick_idx - 1))
    Lt["prev_ili"] = Lt.t.to_numpy() - allc.reindex(key).to_numpy(float)
    th = D[D.context == "post_cue"].set_index(["trial_id", "lick_idx"]).theta_L
    th = th[~th.index.duplicated()]
    Lt["prev_ang"] = th.reindex(key).to_numpy(float)
    Lt = Lt[np.isfinite(Lt.prev_ili) & (Lt.prev_ili > 0) & (Lt.prev_ili < 1.0)]
    # ITI first licks: >= 2 s quiet, no cue / strobe in [t - 2.0, t + 0.75]; group = previous trial's spout
    It = D[(D.context == "iti") & (D.gap >= QUIET_ITI)].copy()
    j = np.searchsorted(ev, It.t.to_numpy() + CUE_EXCL[0])
    near = (j < len(ev)) & (ev[np.minimum(j, len(ev) - 1)] <= It.t.to_numpy() + CUE_EXCL[1])
    It = It[~near].copy()
    ptid = np.where(It.segment.astype(str) == "lead", It.trial_id - 1, It.trial_id)
    It["pos"] = [str(T.pos.get(x, "")) for x in ptid]
    It = It[It.pos != ""]
    # JAW-referenced first licks: jaw onset after the cue, before tongue onset, not at the cue
    Jx = J.drop_duplicates("trial_id").set_index("trial_id")
    Fj = F.copy()
    Fj["jaw_ms"] = Fj.trial_id.map(Jx.jaw_onset_ms).astype(float)
    Fj["at_cue"] = Fj.trial_id.map(Jx.onset_at_cue).astype(object)
    Fj["t_jaw"] = Fj.cue_s + Fj.jaw_ms / 1000
    Fj = Fj[np.isfinite(Fj.t_jaw) & (Fj.at_cue != True) & (Fj.t_jaw > Fj.cue_s) & (Fj.t_jaw < Fj.t)  # noqa: E712
            & (Fj.t_jaw > lo)]
    F["jaw_lat"] = -F.trial_id.map(Jx.jaw3d_side_ipsi_mm).astype(float)          # + = contralesional
    for nm, S, tcol in (("first", F, "t"), ("later", Lt, "t"), ("iti", It, "t"), ("firstjaw", Fj, "t_jaw")):
        n[nm] = len(S)
        out[f"{nm}_X"] = _binned(R, ft, S[tcol].to_numpy()) if len(S) else np.zeros((0, NB, len(cols)), np.float32)
        for c in ("trial_id", "lick_idx", "theta_L", "reach", "speed_peak", "t", "contact"):
            out[f"{nm}_{c}"] = S[c].to_numpy(float)
        out[f"{nm}_pos"] = S.pos.astype(str).to_numpy()
        for c in ("rt", "jaw_lat", "prev_ili", "prev_ang", "jaw_ms"):
            if c in S:
                out[f"{nm}_{c}"] = S[c].to_numpy(float)
        if nm == "firstjaw":
            out[f"{nm}_rt"] = (S.t - S.cue_s).to_numpy(float)
    SESS.mkdir(parents=True, exist_ok=True)
    tmp = SESS / f"{label}.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(f)
    return {"label": label, **{f"n_{k}": v for k, v in n.items()}, "s": round(time.time() - t0, 1)}


# ================================================================================================ per-session fits
def _feat_values(X: np.ndarray, cols: list[str]) -> dict:
    """{(hemi, feat): (n, NB)} area / pool traces (pool = mean of member areas)."""
    v = {}
    for hk, h in HEMI.items():
        for a in AREAS:
            v[(hk, a)] = X[:, :, cols.index(f"{a}_{h}")].astype(np.float64)
        for p, mem in POOLS.items():
            v[(hk, p)] = np.mean([v[(hk, a)] for a in mem], 0)
    return v


def _spout_fe(pos: np.ndarray) -> np.ndarray:
    lv = sorted(set(pos))
    return np.column_stack([(pos == p).astype(float) for p in lv[1:]]) if len(lv) > 1 else np.zeros((len(pos), 0))


def _resid(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    b, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return Y - Z @ b


def slopes(ang: np.ndarray, Z: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FWL: slope of each column of Y on ang/10 given covariates Z (incl. intercept) and the partial r."""
    ra = _resid(Z, ang[:, None] / 10.0)[:, 0]
    RY = _resid(Z, Y)
    den = float(ra @ ra)
    b = (ra @ RY) / den if den > 0 else np.full(Y.shape[1], np.nan)
    sy = np.sqrt((RY ** 2).sum(0))
    pr = np.where(sy > 0, (ra @ RY) / (np.sqrt(den) * np.where(sy > 0, sy, 1)), np.nan)
    return b, pr


def _folds(pos: np.ndarray, rng) -> np.ndarray:
    f = np.empty(len(pos), int)
    for v in np.unique(pos):
        ix = np.flatnonzero(pos == v)
        f[rng.permutation(ix)] = np.arange(len(ix)) % K_FOLD
    return f


def _cv_r(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    p = X.shape[1]
    pred = np.empty(len(y))
    for k in range(K_FOLD):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or tr.sum() < p + 2:
            return np.nan
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        Zt, Ze = (X[tr] - mu) / sd, (X[te] - mu) / sd
        yc = y[tr] - y[tr].mean()
        w = np.linalg.solve(Zt.T @ Zt + p * np.eye(p), Zt.T @ yc)
        pred[te] = Ze @ w + y[tr].mean()
    if pred.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1])


def decode(X: np.ndarray, y: np.ndarray, pos: np.ndarray, rng) -> dict:
    """Within-spout CV ridge decode: r averaged over N_REP fold assignments; null = N_PERM within-spout permutations."""
    Xc = X - pd.DataFrame(X).groupby(pos).transform("mean").to_numpy()
    r = np.nanmean([_cv_r(Xc, y, _folds(pos, rng)) for _ in range(N_REP)])
    f0 = _folds(pos, rng)
    nul = []
    for _ in range(N_PERM):
        yp = y.copy()
        for v in np.unique(pos):
            ix = np.flatnonzero(pos == v)
            yp[ix] = y[rng.permutation(ix)]
        nul.append(_cv_r(Xc, yp, f0))
    rn = float(np.nanmean(nul))
    return {"r": float(r), "r_null": rn, "r_ex": float(r - rn), "r_null95": float(np.nanpercentile(nul, 95))}


def analyse(item) -> dict:
    """One session's npz -> long rows: tuning slopes (T), decode (Dc), time course (TC)."""
    label = item
    z = np.load(SESS / f"{label}.npz", allow_pickle=True)
    from wfield_local import config, epochs
    an, ep = config.animal_of(label), epochs.epoch_of(label)
    day = epochs.days_since_stroke(label)
    base = {"label": label, "animal": an, "epoch": ep, "day": day}
    cols = [str(c) for c in z["cols"]]
    bB = _bins_of(BASE)
    rng = np.random.default_rng([int(label[2:4]), int(label[-4:])])
    T, Dc, TC = [], [], []
    for st in SETS:
        X = z[f"{st}_X"]
        n = len(X)
        if n < MIN_N:
            continue
        pos = z[f"{st}_pos"].astype(str)
        ang = z[f"{st}_theta_L"].astype(float)
        within = ang - pd.Series(ang).groupby(pos).transform("mean").to_numpy()
        par = (z[f"{st}_trial_id"].astype(int) % 2).astype(int)
        V = _feat_values(X, cols)
        keys = list(V)
        basev = {k: V[k][:, bB].mean(1) for k in keys}
        cov = [np.ones(n), z[f"{st}_reach"], z[f"{st}_speed_peak"] / 100]
        if f"{st}_rt" in z and st in ("first", "firstjaw"):
            cov.append(z[f"{st}_rt"])
        if st == "later":
            pa = z["later_prev_ang"].astype(float)
            has = np.isfinite(pa).astype(float)
            cov += [z["later_prev_ili"], np.where(has > 0, pa, 0.0), has]
        C0 = np.column_stack(cov)
        FE = _spout_fe(pos)
        designs = {"within": np.column_stack([C0, FE]), "total": C0}
        for w, ws in WINS.items():
            bw = _bins_of(ws)
            Y = np.column_stack([V[k][:, bw].mean(1) - basev[k] for k in keys])
            Yraw = np.column_stack([V[k][:, bw].mean(1) for k in keys])
            Bm = np.column_stack([basev[k] for k in keys])
            for at, Z in designs.items():
                variants = [("main", Z, Y, None)]
                if at == "within" and st == "first":
                    jl = z["first_jaw_lat"].astype(float)
                    ok = np.isfinite(jl)
                    if ok.sum() >= MIN_N:
                        variants.append(("jawlat", np.column_stack([Z, jl]), Y, ok))
                for vn, Zv, Yv, sub in variants:
                    s = np.ones(n, bool) if sub is None else sub
                    b, pr = slopes(ang[s], Zv[s], Yv[s])
                    for (hk, ft_), bb, rr in zip(keys, b, pr):
                        T.append({**base, "set": st, "angle": at, "win": w, "model": vn, "hemi": hk, "feat": ft_,
                                  "half": "full", "b": bb, "pr": rr, "n": int(s.sum())})
                    if vn == "main" and at == "within":
                        for hv in (0, 1):
                            q = par == hv
                            if q.sum() >= MIN_N // 2:
                                bh, _ = slopes(ang[q], Zv[q], Yv[q])
                                for (hk, ft_), bb in zip(keys, bh):
                                    T.append({**base, "set": st, "angle": at, "win": w, "model": vn, "hemi": hk,
                                              "feat": ft_, "half": "even" if hv == 0 else "odd", "b": bb,
                                              "pr": np.nan, "n": int(q.sum())})
                # S3: baseline as covariate (one fit per column)
                if at == "within" and st == "first":
                    for j, (hk, ft_) in enumerate(keys):
                        b, pr = slopes(ang, np.column_stack([Z, Bm[:, j]]), Yraw[:, [j]])
                        T.append({**base, "set": st, "angle": at, "win": w, "model": "basecov", "hemi": hk,
                                  "feat": ft_, "half": "full", "b": b[0], "pr": pr[0], "n": n})
            # decode (within-spout angle), per hemisphere: 7 orofacial areas; control RSPd + VISpm
            for hk in HEMI:
                for fs, mem in (("oro7", AREAS7), ("ctrl", CTRL_A)):
                    Xd = np.column_stack([V[(hk, a)][:, bw].mean(1) - basev[(hk, a)] for a in mem])
                    d = decode(Xd, within, pos, rng)
                    Dc.append({**base, "set": st, "win": w, "hemi": hk, "fs": fs, "half": "full", "n": n, **d})
                    if st == "first" and fs == "oro7":
                        for hv in (0, 1):
                            q = par == hv
                            d = decode(Xd[q], within[q], pos[q], rng)
                            Dc.append({**base, "set": st, "win": w, "hemi": hk, "fs": fs,
                                       "half": "even" if hv == 0 else "odd", "n": int(q.sum()), **d})
        # time course (first / firstjaw, within angle, pools)
        if st in ("first", "firstjaw"):
            pk = [k for k in keys if k[1] in POOLS]
            for ib in range(NB):
                Y = np.column_stack([V[k][:, ib] - basev[k] for k in pk])
                b, pr = slopes(ang, designs["within"], Y)
                for (hk, ft_), bb, rr in zip(pk, b, pr):
                    TC.append({**base, "set": st, "t": float(CENT[ib]), "hemi": hk, "feat": ft_, "b": bb, "pr": rr})
        # behaviour descriptors of the set
        T.append({**base, "set": st, "angle": "beh", "win": "-", "model": "beh", "hemi": "-", "feat": "within_sd",
                  "half": "full", "b": float(np.std(within)), "pr": np.nan, "n": n})
    return {"T": T, "D": Dc, "TC": TC}


# ================================================================================================ run
def cmd_run(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    SESS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = _items()
    print(f"{len(items)} imaging sessions", flush=True)
    res, fail = ak.fan_sessions(items, extract, jobs=jobs, key=lambda x: x[0])
    S = pd.DataFrame([v for _k, v in res])
    S.to_csv(OUT / "extract_summary.csv", index=False)
    print(f"extracted {(time.time() - t0) / 60:.1f} min; failed {[str(f[0]) for f in fail][:5]}", flush=True)
    if fail:
        raise SystemExit(1)
    labs = sorted(str(it[0]) for it in items)
    res, fail = ak.fan_sessions(labs, analyse, jobs=jobs)
    if fail:
        raise SystemExit(f"analyse failed: {fail[:3]}")
    pd.DataFrame([r for _k, v in res for r in v["T"]]).to_csv(OUT / "tuning_sessions.csv.gz", index=False,
                                                              float_format="%.6g")
    pd.DataFrame([r for _k, v in res for r in v["D"]]).to_csv(OUT / "decode_sessions.csv", index=False,
                                                              float_format="%.6g")
    pd.DataFrame([r for _k, v in res for r in v["TC"]]).to_csv(OUT / "timecourse_sessions.csv.gz", index=False,
                                                               float_format="%.6g")
    print(f"analysed {(time.time() - t0) / 60:.1f} min", flush=True)


def _aug(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.epoch.isin(EPS) & d.animal.isin(AN)]
    return d[(d.epoch != "pre") | (d.label.str[-4:-2] == "08")].copy()


def load_T() -> pd.DataFrame:
    T = _aug(pd.read_csv(OUT / "tuning_sessions.csv.gz"))
    return T


def with_diff(T: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Append hemi 'c-i' rows (b_c - b_i) per label x keys."""
    W = T[T.hemi.isin(["c", "i"])].pivot_table(index=["label", "animal", "epoch", "day"] + keys, columns="hemi",
                                               values="b").reset_index()
    W = W.dropna(subset=["c", "i"])
    W["b"] = W.c - W.i
    W["hemi"] = "c-i"
    return pd.concat([T, W.drop(columns=["c", "i"])], ignore_index=True)


# ================================================================================================ gate
def gate_table(T: pd.DataFrame, D: pd.DataFrame) -> dict:
    P = T[(T.epoch == "pre") & (T.set == "first") & (T.angle == "within") & (T.model == "main")
          & (T.feat == "POOL") & T.hemi.isin(["c", "i"])]
    rows = []
    for (w, an, hk), g in P.groupby(["win", "animal", "hemi"], sort=True):
        h = g.pivot_table(index="label", columns="half", values="b")
        bo, be = h["odd"].mean(), h["even"].mean()
        ok = bool(np.sign(bo) == np.sign(be) and abs(bo + be) / 2 > abs(bo - be))
        est_sd = float(((h["odd"] - h["even"]) / 2).std())
        rows.append({"win": w, "animal": an, "hemi": hk, "b_odd": bo, "b_even": be, "b_full": h["full"].mean(),
                     "n_sessions": len(h), "passed": ok, "est_sd": est_sd, "between_sd": float(h["full"].std()),
                     "per_session_odd_even_r": float(h[["odd", "even"]].corr().iloc[0, 1]) if len(h) > 2 else np.nan})
    G = pd.DataFrame(rows)
    GA = G.groupby(["win", "animal"])["passed"].all().rename("animal_pass").reset_index()
    Pd = D[(D.epoch == "pre") & (D.set == "first") & (D.fs == "oro7")]
    drow = []
    for (w, an, hk), g in Pd.groupby(["win", "animal", "hemi"], sort=True):
        h = g.pivot_table(index="label", columns="half", values="r_ex")
        drow.append({"win": w, "animal": an, "hemi": hk, "rex_odd": h["odd"].mean(), "rex_even": h["even"].mean(),
                     "rex_full": h["full"].mean(), "passed": bool(h["odd"].mean() > 0 and h["even"].mean() > 0)})
    GD = pd.DataFrame(drow)
    GDA = GD.groupby(["win", "animal"])["passed"].all().rename("animal_pass").reset_index()
    return {"G": G, "GA": GA, "GD": GD, "GDA": GDA}


def cmd_gate() -> dict:
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    T = load_T()
    D = _aug(pd.read_csv(OUT / "decode_sessions.csv"))
    g = gate_table(T, D)
    pd.set_option("display.width", 250)
    for k in g:
        g[k].to_csv(OUT / f"gate_{k}.csv", index=False)
    print("== GATE (August pre; first cued licks, within-spout, POOL slope; odd / even trials) ==")
    print(g["G"].round(3).to_string(index=False))
    print(g["GA"].to_string(index=False))
    print("\n== decode gate (r_ex both halves > 0, per hemisphere) ==")
    print(g["GD"].round(3).to_string(index=False))
    summ = {w: int(g["GA"][g["GA"].win == w].animal_pass.sum()) for w in WINS}
    summ_d = {w: int(g["GDA"][g["GDA"].win == w].animal_pass.sum()) for w in WINS}
    (OUT / "gate.json").write_text(json.dumps({"slope_animals_pass": summ, "decode_animals_pass": summ_d,
                                               "interpret_slope": {w: summ[w] >= 3 for w in WINS},
                                               "interpret_decode": {w: summ_d[w] >= 3 for w in WINS}}, indent=1))
    print(f"\nanimals passing (slope): {summ}; (decode): {summ_d}", flush=True)
    return g


# ================================================================================================ summarise
def _change(B: pd.DataFrame) -> pd.DataFrame:
    B = B.copy()
    nn = B.n_paired - B.n_pos
    B["n_same"] = np.where(B.delta > 0, B.n_pos, nn)
    B["change"] = ((B.delta_lo > 0) | (B.delta_hi < 0)) & (B.n_same >= 3)
    return B


PRIM = {"set": "first", "angle": "within", "model": "main", "win": "PRE", "feat": "POOL", "half": "full"}


def _sel(D, **kw):
    for k, v in kw.items():
        D = D[D[k] == v]
    return D


def cmd_summarise() -> None:
    from figdata import save_fd
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    pd.set_option("display.width", 250)
    g = cmd_gate()
    gate = json.loads((OUT / "gate.json").read_text())
    T = load_T()
    T = T[T.half == "full"]
    KT = ["set", "angle", "win", "model", "feat"]
    T2 = with_diff(T[T.model != "beh"], KT)
    T2.to_csv(OUT / "tuning_sessions_with_diff.csv.gz", index=False, float_format="%.6g")
    PA = lc.per_animal(T2, KT + ["hemi"], ["b", "pr"])
    PA.to_csv(OUT / "per_animal.csv", index=False)
    print("\n== PER-ANIMAL (rule 8): PRIMARY first cued licks, within-spout launch angle, POOL; tuning slope b "
          "(dF/F x1e3 per 10 deg, + = more activity for more contralesional launches); c = right (contralesional), "
          "i = left (ipsilesional), c-i = D ==")
    for w in WINS:
        q = _sel(PA, set="first", angle="within", model="main", win=w, feat="POOL")
        print(f"-- {w} --")
        print(q.pivot_table(index=["hemi", "animal"], columns="epoch", values="b_mean").reindex(columns=EPS)
              .round(3).to_string(), flush=True)
        print(q.pivot_table(index=["hemi", "animal"], columns="epoch", values="b_count").reindex(columns=EPS)
              .to_string(), flush=True)
    Dd = _aug(pd.read_csv(OUT / "decode_sessions.csv"))
    Dd = Dd[Dd.half == "full"]
    DW = Dd.pivot_table(index=["label", "animal", "epoch", "day", "set", "win", "fs"], columns="hemi",
                        values="r_ex").reset_index().dropna(subset=["c", "i"])
    DW["c-i"] = DW.c - DW.i
    DL = DW.melt(id_vars=["label", "animal", "epoch", "day", "set", "win", "fs"], value_vars=["c", "i", "c-i"],
                 var_name="hemi", value_name="r_ex")
    PAD = lc.per_animal(DL, ["set", "win", "fs", "hemi"], ["r_ex"])
    PAD.to_csv(OUT / "decode_per_animal.csv", index=False)
    print("\n== PER-ANIMAL decode r_ex (r - null), first licks, oro7 ==")
    for w in WINS:
        q = _sel(PAD, set="first", win=w, fs="oro7")
        print(f"-- {w} --")
        print(q.pivot_table(index=["hemi", "animal"], columns="epoch", values="r_ex_mean").reindex(columns=EPS)
              .round(3).to_string(), flush=True)
    t0 = time.time()
    B = _change(lc.boot_table(T2, KT + ["hemi"], ["b"]).fillna({"n_pos": 0, "n_paired": 0}))
    B.to_csv(OUT / "boot.csv", index=False)
    BD = _change(lc.boot_table(DL, ["set", "win", "fs", "hemi"], ["r_ex"]).fillna({"n_pos": 0, "n_paired": 0}))
    BD.to_csv(OUT / "decode_boot.csv", index=False)
    print(f"boot {(time.time() - t0) / 60:.1f} min", flush=True)
    # primary + Holm
    rng = np.random.default_rng(SEED)
    rows = []
    Pm = _sel(T2, **{k: v for k, v in PRIM.items() if k != "half"})
    for hk, nm in (("c", "P1 b_c (contralesional)"), ("i", "P2 b_i (ipsilesional)"), ("c-i", "P3 D = b_c - b_i")):
        gm = Pm[Pm.hemi == hk]
        pre = {a: list(h.b) for a, h in gm[gm.epoch == "pre"].groupby("animal", sort=True)}
        for ep in EPS[1:]:
            cur = {a: list(h.b) for a, h in gm[gm.epoch == ep].groupby("animal", sort=True)}
            iv, dr = ak.boot_delta(cur, pre, rng, draws=True)
            common = sorted(set(cur) & set(pre))
            da = [np.mean(cur[a]) - np.mean(pre[a]) for a in common]
            rows.append({"hemi": hk, "name": nm, "epoch": ep, "delta": iv.point if iv else np.nan,
                         "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan, "p": ak.boot_p(dr),
                         "n_pos": int(np.sum(np.array(da) > 0)), "n_paired": len(common),
                         "per_animal": " ".join(f"{a}:{x:+.3f}" for a, x in zip(common, da))})
    H = pd.DataFrame(rows)
    H["p_holm"] = ak.holm(H.p.to_numpy())
    H["n_same"] = np.where(H.delta > 0, H.n_pos, H.n_paired - H.n_pos)
    H["change"] = ((H.lo > 0) | (H.hi < 0)) & (H.n_same >= 3)
    H.to_csv(OUT / "primary_holm.csv", index=False)
    print("\n== PRIMARY (first cued licks, PRE window, within-spout, POOL), boot_delta vs August pre, Holm over 9; "
          f"gate animals passing (PRE): {gate['slope_animals_pass']['PRE']} / 4 ==")
    print(H.round(3).to_string(index=False), flush=True)
    print("\n== boot_delta lines (selected) ==")
    show = B[(B.epoch != "pre") & (B.set == "first") & (B.angle == "within") & (B.model == "main")
             & B.feat.isin(["POOL", "MOTOR", "SENS", "CTRL"])]
    print(show[["win", "feat", "hemi", "epoch", "level", "delta", "delta_lo", "delta_hi", "n_same", "change",
                "per_animal_delta"]].round(3).to_string(index=False), flush=True)
    # recovery link: aim = far-contra FIRST-lick launch error e_L (within_lick_correction sessions.csv)
    WS = pd.read_csv(WLC / "sessions.csv")
    aim = _sel(WS, context="cued", group="far_contra", order="first", outcome="any").set_index("label").e_L
    TRT = Pm[Pm.hemi.isin(["c", "i", "c-i"])].pivot_table(index=["label", "animal", "epoch", "day", "win"],
                                                          columns="hemi", values="b").reset_index()
    TRT.columns = [str(c) for c in TRT.columns]
    TRT = TRT.rename(columns={"c": "b_c", "i": "b_i", "c-i": "D"})
    TRD = DW[(DW.set == "first") & (DW.fs == "oro7")].rename(columns={"c": "dec_c", "i": "dec_i", "c-i": "dec_D"})
    TRT = TRT.merge(TRD[["label", "win", "dec_c", "dec_i", "dec_D"]], on=["label", "win"], how="left")
    TR = lc.track(TRT, ["win"], ["b_c", "b_i", "D", "dec_c", "dec_i", "dec_D"], aim)
    TR.to_csv(OUT / "track.csv", index=False)
    print("\n== recovery link (post sessions; aim = far-contra first-lick launch error; log(day) in the model) ==")
    keep = [c for c in TR.columns if c in ("win", "metric", "n", "aim_slope_w", "aim_lo", "aim_hi", "aim_p")
            or c.startswith("r_aim_")]
    print(TR[keep].round(3).to_string(index=False), flush=True)
    TC = _aug(pd.read_csv(OUT / "timecourse_sessions.csv.gz"))
    TCB = []
    rng2 = np.random.default_rng(SEED + 1)
    for (st, ft_, hk, tt), h in TC.groupby(["set", "feat", "hemi", "t"], sort=True):
        for ep in EPS:
            cur = {a: list(x.b) for a, x in h[h.epoch == ep].groupby("animal", sort=True)}
            lev = ak.boot_ci(cur, rng2, n_boot=500) if cur else None
            TCB.append({"set": st, "feat": ft_, "hemi": hk, "t": tt, "epoch": ep,
                        "level": lev.point if lev else np.nan, "lo": lev.lo if lev else np.nan,
                        "hi": lev.hi if lev else np.nan})
    TCB = pd.DataFrame(TCB)
    TCB.to_csv(OUT / "timecourse_levels.csv", index=False)
    beh = _sel(load_T(), model="beh", half="full")
    save_fd("prelick_hemisphere_tuning", {"T": T2, "B": B, "H": H, "DL": DL, "BD": BD, "gate": g, "gate_json": gate,
                                          "TR": TR, "TCB": TCB, "TC": TC, "beh": beh}, root=FD)
    print("figdata saved", flush=True)


# ================================================================================================ figures
HCOL = {"c": "#e66101", "i": "#5e3c99", "c-i": "k"}
HLAB = {"c": "contralesional (RIGHT) hemisphere", "i": "ipsilesional (LEFT) hemisphere", "c-i": "D = c - i"}


def _wrap(s, n=110):
    import textwrap
    return "\n".join(textwrap.fill(p, n) for p in s.split("\n"))


def epoch_panel(ax, S, B, sel, hemi, val="b", off=0.0, row=0, color=None, holm=None, epoch_colours=False):
    """Open session markers (shape = animal; colour = epoch for a single series, hemisphere colour otherwise) + pooled
    level with 95 % CI (thick); top-band star where the change vs pre meets the criterion."""
    import within_lick_correction as wlc
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    color = color or HCOL[hemi]
    x = np.arange(len(EPS)) + off
    s = _sel(S, **sel)
    s = s[s.hemi == hemi]
    rng = np.random.default_rng(7)
    for an, h in s.groupby("animal"):
        v = h.groupby(["epoch", "label"])[val].mean().dropna()
        ep_of = v.index.get_level_values(0)
        xi = np.array([x[EPS.index(e)] for e in ep_of], float) + rng.uniform(-0.04, 0.04, len(v)) + 0.1
        cs = [EPOCH_LINE[e] for e in ep_of] if epoch_colours else [color] * len(v)
        ax.scatter(xi, v.to_numpy(float), s=12, facecolors="none", edgecolors=cs, alpha=0.6,
                   marker=ANIMAL_MARKER.get(an, "o"), linewidths=0.8, zorder=2)
    b = _sel(B, **sel)
    b = b[(b.hemi == hemi) & (b.metric == val)].set_index("epoch").reindex(EPS)
    ax.errorbar(x, b.level, yerr=[b.level - b.level_lo, b.level_hi - b.level], color=color, lw=2.2, marker="o",
                ms=5, capsize=3, zorder=4, label=HLAB.get(hemi, hemi))
    for i, ep in enumerate(EPS[1:], start=1):
        if ep in b.index and b.loc[ep, "change"] == 1:
            txt = "*"
            if holm is not None:
                hh = holm[(holm.hemi == hemi) & (holm.epoch == ep)]
                if len(hh) and hh.p_holm.iloc[0] < 0.05:
                    txt = "**"
            wlc.star_top(ax, x[i], txt, color, row=row)
    ax.set_xticks(np.arange(len(EPS)), EPS, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.axhline(0, color="0.6", lw=0.7, zorder=0)


def _legend_markers(fig):
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    h = [Line2D([], [], ls="none", marker=m, mfc="none", mec="0.3", label=a) for a, m in ANIMAL_MARKER.items()]
    h += [Line2D([], [], color=HCOL[k], lw=2.2, marker="o", label=HLAB[k]) for k in ("c", "i")]
    fig.legend(handles=h, loc="outside lower center", ncol=6, fontsize=8, frameon=False)


def _gate_txt(D, w):
    gj = D["gate_json"]
    return (f"gate {w}: slope {gj['slope_animals_pass'][w]}/4 animals pass"
            + ("" if gj["interpret_slope"][w] else " -> NOT interpreted (descriptive)"))


def fig_primary(D):
    import matplotlib.pyplot as plt

    import within_lick_correction as wlc
    T, B, H, DL, BD = D["T"], D["B"], D["H"], D["DL"], D["BD"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for r, w in enumerate(WINS):
        sel = {"set": "first", "angle": "within", "model": "main", "win": w, "feat": "POOL"}
        ax = axs[r, 0]
        epoch_panel(ax, T, B, sel, "c", off=-0.12, row=0, holm=H if w == "PRE" else None)
        epoch_panel(ax, T, B, sel, "i", off=0.12, row=1, holm=H if w == "PRE" else None)
        ax.set_ylabel("tuning slope b (dF/F x1e3 per 10 deg)", fontsize=8)
        ax.set_title(_wrap(f"{'AD'[r]}. {w}: orofacial POOL slope on WITHIN-spout launch angle, per hemisphere "
                           f"(+ = more activity for more contralesional launches)", 60), fontsize=9)
        ax = axs[r, 1]
        epoch_panel(ax, T, B, sel, "c-i", row=0, color="k", holm=H if w == "PRE" else None, epoch_colours=True)
        ax.set_title(_wrap(f"{'BE'[r]}. {w}: hemisphere difference D = b_c - b_i (pre expected < 0: each hemisphere "
                           f"contraversive)", 60), fontsize=9)
        ax = axs[r, 2]
        ds = {"set": "first", "win": w, "fs": "oro7"}
        epoch_panel(ax, DL, BD, ds, "c", val="r_ex", off=-0.12, row=0)
        epoch_panel(ax, DL, BD, ds, "i", val="r_ex", off=0.12, row=1)
        ax.set_ylabel("decode r - null r", fontsize=8)
        ax.set_title(_wrap(f"{'CF'[r]}. {w}: CV ridge decode of within-spout launch angle from the 7 orofacial areas "
                           f"of ONE hemisphere", 60), fontsize=9)
    for ax in axs[:, 0]:
        ax.sharey(axs[0, 0])
    wlc.reserve_headroom(fig)
    _legend_markers(fig)
    fig.suptitle(_wrap("Pre-launch vs post-contact hemispheric tuning to the LAUNCH angle (cued FIRST post-cue "
                       "licks, engaged trials, no lick in the 1 s before onset; PS92-95, August pre). Windows from "
                       "tongue onset: PRE -0.30..-0.05 s, POST +0.05..+0.30 s, minus BASE -0.80..-0.50 s. Model: "
                       "spout fixed effects (within-spout angle) + reach + peak speed + reaction time. Thick = pooled "
                       "session level, 95 % nested bootstrap CI; open markers = sessions (shape = animal). Top-band * "
                       "= change vs pre (CI excludes 0 AND >= 3/4 animals); ** = also Holm over the 9 primary tests. "
                       + _gate_txt(D, "PRE") + "; " + _gate_txt(D, "POST") + ".", 190), fontsize=9.5)
    return fig


def fig_areas(D, w):
    import matplotlib.pyplot as plt

    import within_lick_correction as wlc
    T, B = D["T"], D["B"]
    feats = ["ALM", "tjM1", "MOp_rest", "MOs_rest", "SSp-m", "SSp-n", "SSs", "MOTOR", "SENS", "RSPd", "VISpm", "CTRL"]
    fig, axs = plt.subplots(3, 4, figsize=(16, 11), layout="constrained", sharey=True)
    for ax, f in zip(axs.ravel(), feats):
        sel = {"set": "first", "angle": "within", "model": "main", "win": w, "feat": f}
        epoch_panel(ax, T, B, sel, "c", off=-0.12, row=0)
        epoch_panel(ax, T, B, sel, "i", off=0.12, row=1)
        ax.set_title(f + (" (control)" if f in CTRL_A + ["CTRL"] else ""), fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("b (dF/F x1e3 per 10 deg)", fontsize=8)
    wlc.reserve_headroom(fig)
    _legend_markers(fig)
    fig.suptitle(_wrap(f"Area split, {w} window: tuning slope on the within-spout launch "
                       f"angle per area and hemisphere (first cued licks; MOTOR = ALM + tjM1, SENS = SSp-m + SSp-n + "
                       f"SSs, CTRL = RSPd + VISpm). Same conventions as the primary figure. A command route (D) predicts the "
                       f"contralesional change in MOTOR at least as much as SENS; the same change in CTRL = "
                       f"non-specific. " + _gate_txt(D, w) + ".", 190), fontsize=9.5)
    return fig


def fig_timecourse(D):
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    TCB = D["TCB"]
    feats = ["POOL", "MOTOR", "SENS", "CTRL"]
    fig, axs = plt.subplots(len(feats), 3, figsize=(15, 13), layout="constrained", sharex="col")
    for r, f in enumerate(feats):
        for c, (st, hk) in enumerate((("first", "c"), ("first", "i"), ("firstjaw", "c"))):
            ax = axs[r, c]
            for ep in EPS:
                q = TCB[(TCB.set == st) & (TCB.feat == f) & (TCB.hemi == hk) & (TCB.epoch == ep)].sort_values("t")
                ax.plot(q.t, q.level, color=EPOCH_LINE[ep], lw=2 if ep == "pre" else 1.5, label=ep)
                ax.fill_between(q.t, q.lo, q.hi, color=EPOCH_LINE[ep], alpha=0.12, lw=0)
            ax.axhline(0, color="0.6", lw=0.7)
            ax.axvline(0, color="0.4", lw=0.8, ls="--")
            for _w, (a, b) in WINS.items():
                ax.axvspan(a, b, color="0.85", alpha=0.35, lw=0, zorder=0)
            ref = "jaw onset" if st == "firstjaw" else "tongue onset"
            ax.set_title(f"{f}, {HLAB[hk]}, from {ref}", fontsize=9)
            ax.tick_params(labelsize=8)
            if r == len(feats) - 1:
                ax.set_xlabel(f"time from {ref} (s; 0.05 s bins)", fontsize=8)
        axs[r, 0].set_ylabel("b (dF/F x1e3 / 10 deg)", fontsize=8)
    axs[0, 0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle(_wrap("Time course of the within-spout launch-angle tuning slope (bin minus BASE -0.80..-0.50 "
                       "s), first cued licks; pooled session level with 95 % nested bootstrap CI (descriptive). Grey "
                       "bands = PRE and POST windows. Right column: the contralesional hemisphere aligned to JAW onset "
                       "(PREJAW control; the jaw leads the tongue by 52-120 ms). GCaMP6s lags spiking by ~0.1-0.2 s.",
                       190), fontsize=9.5)
    return fig


def fig_gate(D):
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    G = D["gate"]["G"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for c, w in enumerate(WINS):
        ax = axs[c]
        q = G[G.win == w]
        for r in q.itertuples():
            ax.scatter(r.b_odd, r.b_even, marker=ANIMAL_MARKER[r.animal], s=60, edgecolors=HCOL[r.hemi],
                       facecolors=HCOL[r.hemi] if r.passed else "none", linewidths=1.2)
        lim = np.nanmax(np.abs(q[["b_odd", "b_even"]].to_numpy())) * 1.2 if len(q) else 1
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.7)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.axvline(0, color="0.8", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("b, ODD trials (animal mean over pre sessions)", fontsize=8)
        ax.set_ylabel("b, EVEN trials", fontsize=8)
        ax.set_title(_wrap(f"{chr(97 + c)}. {w}: split-half POOL slope per animal x hemisphere (filled = passes)", 55),
                     fontsize=9)
    ax = axs[2]
    q = G.copy()
    xs = {("PRE", "c"): 0, ("PRE", "i"): 1, ("POST", "c"): 2.5, ("POST", "i"): 3.5}
    for r in q.itertuples():
        x0 = xs[(r.win, r.hemi)]
        ax.scatter(x0 - 0.12, r.est_sd, marker=ANIMAL_MARKER[r.animal], facecolors="none", edgecolors=HCOL[r.hemi])
        ax.scatter(x0 + 0.12, r.between_sd, marker=ANIMAL_MARKER[r.animal], facecolors=HCOL[r.hemi],
                   edgecolors=HCOL[r.hemi])
    ax.set_xticks(list(xs.values()), [f"{w}\n{h}" for w, h in xs], fontsize=8)
    ax.set_ylabel("SD of b (dF/F x1e3 / 10 deg)", fontsize=8)
    ax.set_title(_wrap("c. noise floor: per-session estimation SD (open, from odd - even) vs between-session SD "
                       "(filled), August pre", 55), fontsize=9)
    _legend_markers(fig)
    fig.suptitle(_wrap("Stability gate (pre-registered): a hemisphere passes if its odd- and even-trial animal "
                       "means agree in sign and their mean exceeds their difference; an animal passes if both "
                       "hemispheres pass; post changes interpreted only if >= 3/4 animals pass. "
                       + _gate_txt(D, "PRE") + "; " + _gate_txt(D, "POST") + ".", 170), fontsize=9.5)
    return fig


def fig_secondary(D):
    import matplotlib.pyplot as plt

    import within_lick_correction as wlc
    T, B = D["T"], D["B"]
    rows = [("first", "total", "main", "PRE", "first licks, TOTAL angle (target + launch; no spout FE)"),
            ("firstjaw", "within", "main", "PRE", "first licks, PREJAW (-0.30..-0.05 s from JAW onset)"),
            ("first", "within", "basecov", "PRE", "first licks, baseline as covariate (not subtracted)"),
            ("first", "within", "jawlat", "PRE", "first licks, + jaw lateral covariate (side spouts)"),
            ("later", "within", "main", "PRE", "LATER cued licks (+ previous-lick interval and angle)"),
            ("iti", "within", "main", "PRE", "ITI first licks (>= 2 s quiet; within previous-trial spout)")]
    fig, axs = plt.subplots(len(rows), 2, figsize=(12, 3.0 * len(rows)), layout="constrained")
    for r, (st, at, md, w, tt) in enumerate(rows):
        sel = {"set": st, "angle": at, "model": md, "win": w, "feat": "POOL"}
        epoch_panel(axs[r, 0], T, B, sel, "c", off=-0.12, row=0)
        epoch_panel(axs[r, 0], T, B, sel, "i", off=0.12, row=1)
        epoch_panel(axs[r, 1], T, B, sel, "c-i", color="k", epoch_colours=True)
        axs[r, 0].set_title(_wrap(f"{'ACEGIKMOQ'[r]}. {tt}: b per hemisphere", 70), fontsize=9)
        axs[r, 1].set_title(_wrap(f"{'BDFHJLNPR'[r]}. {tt}: D = b_c - b_i", 70), fontsize=9)
        axs[r, 0].set_ylabel("b (POOL)", fontsize=8)
    wlc.reserve_headroom(fig)
    _legend_markers(fig)
    fig.suptitle(_wrap("Secondary lick sets and sensitivities, PRE window, orofacial POOL (same conventions as the primary figure; "
                       "stars = the pre-registered change criterion, no multiple-comparison correction).", 120),
                 fontsize=9.5)
    return fig


def cmd_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figdata import load_fd
    D = load_fd("prelick_hemisphere_tuning", root=FD)
    FIGS.mkdir(parents=True, exist_ok=True)
    for name, fn in (("pht_fig1_primary", fig_primary), ("pht_fig2_areas_PRE", lambda d: fig_areas(d, "PRE")),
                     ("pht_fig3_areas_POST", lambda d: fig_areas(d, "POST")), ("pht_fig4_timecourse", fig_timecourse),
                     ("pht_fig5_gate", fig_gate), ("pht_fig6_secondary", fig_secondary)):
        fig = fn(D)
        fig.savefig(FIGS / f"{name}.png", dpi=140)
        plt.close(fig)
        print(f"wrote {FIGS / name}.png", flush=True)


def main() -> int:
    # coordinator 11-04: routed (outputs under the run tag), no longer refused in a tagged run
    from wfield_local import parallel
    parallel.pin_blas()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args else "plot"
    jobs = 3
    if "--jobs" in sys.argv:
        jobs = min(3, int(sys.argv[sys.argv.index("--jobs") + 1]))
    if "--plot-only" in sys.argv or cmd == "plot":
        cmd_plot()
        return 0
    if cmd == "run":
        cmd_run(jobs)
    elif cmd == "gate":
        cmd_gate()
    elif cmd == "summarise":
        cmd_summarise()
        cmd_plot()
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
