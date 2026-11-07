"""WITHIN-LICK CORRECTIONS (analysis 2.1, RECOVERY_MECHANISMS_LIT_20261030; Bollu et al. 2021): is recovery carried by
re-aiming INSIDE one protrusion (launch direction vs where the tongue ends up; corrective submovements = extra speed
peaks)? Between-lick corrections are `lick_correction.py` and are not redone here.

DESIGN (fixed before any post-stroke number): docs/status/WITHIN_LICK_CORRECTION_PREREG_20261030.md -- read it first.

SIGN CONVENTIONS (every table / figure):
  angle   MIDLINE angle asin(ml / |v|), deg, + = CONTRALESIONAL (mouse right; sign forced so the August-pre far-contra
          endpoint mean is > 0). theta_L = launch (SG-smoothed displacement, first tracked frame -> frame 3 = 12 ms);
          theta_E = endpoint (raw tip at the extension peak, from the mouth zero = kinematics_all `midline`).
  e_L/e_E error vs the animal's AUGUST-pre mean of the same angle on CONTACT post-cue licks at that spout (deg; - =
          ipsilesional of the pre-stroke hit pattern).
  corr    e_E - e_L (deg; + = endpoint more contralesional than the launch; at far contra + = TOWARD the target).
  toward  -sign(e_L) * corr for |e_L| >= 5 deg (+ = moved toward the target during the lick).
  g_w     OLS slope of corr on -e_L (+ spout FE): 1 = endpoint fully back on the pre pattern, 0 = keeps launch error.
  CSM     corrective submovements = sum over the two phases (before / after the extension peak) of max(0, speed peaks
          - 1), prominence >= theta_s (noise-calibrated on August pre, `csm_threshold.json`); CSM_lat on |v_ml|.
  csm_false  share of licks where a noise surrogate (smooth track + the session's noise) gains a CSM.
  hyp_mm  max projection of the tip on the direction of the commanded target spout minus the spout distance (- = short).
  HEMISPHERE (neural): c = contralesional = '_right', i = ipsilesional = '_left'.

Usage (bash env per the repo; outputs only under <v36>/imaging_encoding/within_lick_correction/):
  python scripts/within_lick_correction.py run [--jobs 3] [--redo]   per-lick extraction (pre first -> threshold ->
                                                                       post) + neural per-session fits
  python scripts/within_lick_correction.py summarize                 session cells, per-animal tables, boot_delta,
                                                                       recovery link; writes figdata
  python scripts/within_lick_correction.py plot   (or any subcommand with --plot-only)   figures from figdata only
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lick_correction as lc  # noqa: E402

V36 = "lp_r4_heatmap_20261013__clean_v36"
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
GROUPS = {"all": POS, "far_contra": ["far_contra"], "other5": POS[:5], **{p: [p] for p in POS if p != "far_contra"}}
N_LAUNCH = 3            # frames (12 ms) for the launch direction
MIN_PEAK_FR = 4         # extension peak must be >= 4 frames after the run start for an aim score
ONSET_TOL = 8.5         # ms: run must start / end within 2 frames of the window edge
MIN_CSM_FR = 10
TOWARD_MIN = 5.0        # deg
CLEAN_CORR = 10.0       # deg: |corr| below this and no CSM = "clean" lick (neural N2)
MIN_N = 10
DT = 0.004
GRID = list(range(30, 201, 10))
FALSE_MAX = 0.05
AREAS_N = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
WINS = {"w0.0-0.2": (0.0, 0.2), "w0.2-0.35": (0.2, 0.35)}
PRE_W = (-0.2, -0.1)
TAUS = (0.0, 0.6)
SEED = 0


def out_root() -> Path:
    # coordinator 11-04: routed -- <imaging_encoding>/<tag>/within_lick_correction in a tagged run
    from wfield_local import run_mode
    return run_mode.encoding_out_root(V36) / "within_lick_correction"


OUT = out_root()
LICKS = OUT / "licks"
FIGS = OUT / "figures"
FD = OUT / "figdata"


# ================================================================================================ per-lick extraction
def _sessions() -> list[str]:
    """Curated PS92-95 sessions present in kinematics_all (= passing the pose3d session gates), as PSxx_YYYYMMDD."""
    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    p = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
    have = set(pd.read_csv(p, usecols=["session"]).session.unique())
    cur = ak.curated_ids()
    return sorted(s for s in have & cur if s[:4] in AN)


def _label(sid: str) -> str:
    return f"{sid[:4]}_{sid[-4:]}"


def _is_aug_pre_sid(sid: str) -> bool:
    from wfield_local import config
    from wfield_local import epochs
    return epochs.epoch_of(_label(sid)) == "pre" and not config.early_pre(sid)


def _track(t: np.ndarray, P: np.ndarray, src: np.ndarray):
    """Fill isolated one-frame gaps (linear), return the longest consecutive (4 ms) run: (t, P, src) or None."""
    P = P.copy()
    ok = np.isfinite(P).all(1)
    n = len(t)
    for i in range(1, n - 1):
        if not ok[i] and ok[i - 1] and ok[i + 1] and abs(t[i + 1] - t[i - 1] - 8.0) < 0.5:
            P[i] = (P[i - 1] + P[i + 1]) / 2
            src = src.copy()
            src[i] = 9                                       # interpolated
    ok = np.isfinite(P).all(1)
    best, cur_a = (0, 0), None
    for i in range(n):
        if ok[i] and cur_a is not None and abs(t[i] - t[i - 1] - 4.0) < 0.5:
            pass
        elif ok[i]:
            cur_a = i
        else:
            cur_a = None
        if cur_a is not None and i + 1 - cur_a > best[1] - best[0]:
            best = (cur_a, i + 1)
    a, b = best
    if b - a < 3:
        return None
    return t[a:b], P[a:b], src[a:b]


def _peaks(x: np.ndarray, k: int) -> tuple[tuple, tuple]:
    """Prominences of the local maxima of x (zero-padded) before / after frame k."""
    from scipy.signal import find_peaks, peak_prominences
    xp = np.r_[0.0, x, 0.0]
    pk, _ = find_peaks(xp)
    if not len(pk):
        return (), ()
    pr = peak_prominences(xp, pk)[0]
    idx = pk - 1
    return (tuple(np.round(pr[idx < k], 2)), tuple(np.round(pr[idx > k], 2)))


def csm_count(pre: tuple, post: tuple, th: float) -> int:
    a = sum(1 for p in pre if p >= th)
    b = sum(1 for p in post if p >= th)
    return max(0, a - 1) + max(0, b - 1)


def session_licks(sid: str) -> dict:
    """One session -> LICKS/<sid>.pkl: one row per post-cue / ITI lick (see the module docstring)."""
    from scipy.signal import savgol_filter

    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import spout_world as SW
    from wfield_local.paths import PathResolver
    f = LICKS / f"{sid}.pkl"
    if f.exists() and "--redo" not in sys.argv:
        return {"sid": sid, "cached": True}
    t0 = time.time()
    rv = PathResolver()
    an, date = sid.split("_")
    from scripts.apparatus_mouth import _rig
    geo = _rig(rv, an, date)["geometry"]
    tgt = {p: np.asarray(SW.commanded_world(geo, p), float) for p in SW.POSITIONS}
    S = pd.read_csv(Path(rv.root("dlc")) / "session_poses" / "pose3d" / f"{sid}_frames.csv.gz",
                    usecols=["trial_id", "position", "t_ms", "tongue_src", "tongue_ml", "tongue_ap", "tongue_dv"]
                    ).drop_duplicates(["trial_id", "t_ms"]).sort_values(["trial_id", "t_ms"])
    A = pd.read_csv(Path(rv.root("orofacial")) / V36 / "sessions" / an / date / "licks_all.csv",
                    usecols=["trial_id", "position", "lick_idx", "on_ms", "off_ms", "context", "contact"])
    A["lick_uid"] = np.arange(len(A))
    A = A[A.context.isin(["post_cue", "iti"])]
    g = {k: (v.t_ms.to_numpy(float), v[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float),
             v.tongue_src.to_numpy(int)) for k, v in S.groupby("trial_id")}
    tracks = []
    for r in A.itertuples():
        fr = g.get(r.trial_id)
        if fr is None:
            continue
        sel = (fr[0] >= r.on_ms - 0.5) & (fr[0] <= r.off_ms + 0.5)
        if sel.sum() < 3:
            continue
        tr = _track(fr[0][sel], fr[1][sel], fr[2][sel])
        if tr is None:
            continue
        tracks.append((r, int(sel.sum()), tr))
    # session noise SD (upper bound): residual from SG(7, 2) on fully tracked post-cue licks, interior frames
    res = []
    for r, _nw, (t, P, _src) in tracks:
        if r.context == "post_cue" and len(P) >= 9 and t[0] - r.on_ms <= ONSET_TOL and r.off_ms - t[-1] <= ONSET_TOL:
            res.append((P - savgol_filter(P, 7, 2, axis=0))[3:-3])
    sd = np.concatenate(res).std(0) / np.sqrt(1 - 1 / 3) if res else np.full(3, np.nan)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    rows = []
    for r, nw, (t, P, src) in tracks:
        n = len(P)
        rr = np.linalg.norm(P, axis=1)
        k = int(np.argmax(rr))
        lag_on, lag_off = t[0] - r.on_ms, r.off_ms - t[-1]
        onset_ok = lag_on <= ONSET_TOL
        full_ok = bool(onset_ok and lag_off <= ONSET_TOL and n >= MIN_CSM_FR)
        row = {"sid": sid, "lick_uid": int(r.lick_uid), "trial_id": int(r.trial_id), "lick_idx": int(r.lick_idx),
               "context": r.context, "position": r.position, "pos": oc.pos_key(r.position, "L"),
               "contact": bool(r.contact) if pd.notna(r.contact) else False, "lick_ms": r.off_ms - r.on_ms,
               "n_win": nw, "n_run": n, "lag_on": lag_on, "lag_off": lag_off, "onset_ok": bool(onset_ok),
               "full_ok": full_ok, "k": k, "single_share": float(np.mean((src > 1) & (src != 9))),
               "reach": float(rr[k]), "theta_E": float(np.degrees(np.arcsin(np.clip(P[k, 0] / rr[k], -1, 1))))}
        if n >= 5:
            Ps = savgol_filter(P, 5, 2, axis=0)
            V = savgol_filter(P, 5, 2, deriv=1, axis=0) / DT
            sp = np.linalg.norm(V, axis=1)
            row["speed_peak"] = float(sp.max())
            if onset_ok and k >= MIN_PEAK_FR:
                d = Ps[N_LAUNCH] - Ps[0]
                nd = np.linalg.norm(d)
                row["theta_L"] = float(np.degrees(np.arcsin(np.clip(d[0] / nd, -1, 1)))) if nd > 0.05 else np.nan
                seg = np.linalg.norm(np.diff(Ps[:k + 1], axis=0), axis=1).sum()
                ch = np.linalg.norm(Ps[k] - Ps[0])
                row["curv"] = float(seg / ch) if ch > 0.2 else np.nan
                row["prot_ms"] = float(t[k] - t[0])
                h = Ps[k] - Ps[k - 3]
                row["theta_H"] = float(np.degrees(np.arcsin(np.clip(h[0] / np.linalg.norm(h), -1, 1))))
                # normalised path onset -> extension peak (11 points) for the trajectory figure
                q = np.linspace(0, k, 11)
                row["path_ml"] = tuple(np.round(np.interp(q, np.arange(k + 1), Ps[:k + 1, 0]), 3))
                row["path_r"] = tuple(np.round(np.interp(q, np.arange(k + 1), np.linalg.norm(Ps[:k + 1], axis=1)), 3))
            if full_ok:
                row["pk_sp"] = _peaks(sp, k)
                row["pk_ml"] = _peaks(np.abs(V[:, 0]), k)
                if np.isfinite(sd).all():
                    T9 = savgol_filter(P, 9, 3, axis=0)
                    Tn = T9 + rng.normal(0, 1, T9.shape) * sd
                    for nm, X in (("sm", T9), ("nz", Tn)):
                        Vx = savgol_filter(X, 5, 2, deriv=1, axis=0) / DT
                        row[f"sur_{nm}_sp"] = _peaks(np.linalg.norm(Vx, axis=1), k)
                        row[f"sur_{nm}_ml"] = _peaks(np.abs(Vx[:, 0]), k)
        if r.context == "post_cue" and r.position in tgt:
            T = tgt[r.position]
            dm = np.linalg.norm(P - T, axis=1)
            m = int(np.argmin(dm))
            row["miss_mm"] = float(dm[m])
            row["theta_C"] = float(np.degrees(np.arcsin(np.clip(P[m, 0] / max(np.linalg.norm(P[m]), 1e-9), -1, 1))))
            u = T / np.linalg.norm(T)
            row["hyp_mm"] = float((P @ u).max() - np.linalg.norm(T))
        rows.append(row)
    L = pd.DataFrame(rows)
    L["noise_ml"], L["noise_ap"], L["noise_dv"] = sd
    L["epoch"] = epochs.epoch_of(_label(sid))
    L["animal"], L["label"] = an, _label(sid)
    LICKS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".pkl.tmp")
    L.to_pickle(tmp)
    tmp.replace(f)
    return {"sid": sid, "n": len(L), "s": round(time.time() - t0, 1)}


#: THE CAMERA WORKING PERIOD (Priya 11-05, after (f) 2): licks of trials after the camera quit or inside a stopped run
#: are dropped (`trial_classes.working_trial_ids`: first cue -> last camera lick, minus the stopped runs; an ITI lick
#: belongs to the trial it follows, by its licks_all trial_id). A session without camera classes is left out and listed.
#: Applied when the per-lick caches are READ, so the extraction caches stay valid. Until 11-05 every post-cue / ITI lick
#: of the session was scored, including licks in stopped time.
_NO_CAMERA: set = set()


def working_mask(sid: str, trial_id) -> np.ndarray | None:
    """Per lick (its trial_id): inside the camera working period; None for a session without camera classes."""
    from wfield_local import trial_classes as tc
    wk, has = tc.working_trial_ids(_label(sid))
    if not has:
        return None
    return np.array([bool(wk.get(int(t), False)) for t in np.asarray(trial_id)], bool)


def load_licks(sids: list[str], cols: list[str] | None = None) -> pd.DataFrame:
    out = []
    for s in sids:
        d = pd.read_pickle(LICKS / f"{s}.pkl")
        m = working_mask(s, d["trial_id"].to_numpy())
        if m is None:
            _NO_CAMERA.add(s)
            continue
        d = d[m]
        out.append(d[[c for c in cols if c in d]] if cols else d)
    if _NO_CAMERA:
        print(f"  camera working period: no camera classes, left out: {sorted(_NO_CAMERA)}", flush=True)
    return pd.concat(out, ignore_index=True)


def calibrate(L: pd.DataFrame) -> dict:
    """Threshold rule (prereg section 2), AUGUST-pre post-cue licks only."""
    from wfield_local import config
    C = L[L.full_ok & (L.context == "post_cue") & L.sur_sm_sp.notna()]
    assert set(C.epoch) == {"pre"} and all(not config.early_pre(s) for s in C.sid.unique()), "calibration must be Aug pre"
    out = {"n_licks": int(len(C)), "sessions": sorted(C.sid.unique()), "grid": {}}
    for kind in ("sp", "ml"):
        tab = []
        for th in GRID:
            sm = np.array([csm_count(*x, th) for x in C[f"sur_sm_{kind}"]])
            nz = np.array([csm_count(*x, th) for x in C[f"sur_nz_{kind}"]])
            obs = np.array([csm_count(*x, th) for x in C[f"pk_{kind}"]])
            tab.append({"theta": th, "false": float(np.mean(nz > sm)), "obs_share": float(np.mean(obs >= 1)),
                        "sur_smooth_share": float(np.mean(sm >= 1))})
        out["grid"][kind] = tab
        ok = [r["theta"] for r in tab if r["false"] <= FALSE_MAX]
        out[f"theta_{kind}"] = float(min(ok)) if ok else None
    return out


# ================================================================================================ neural (per session)
def neural_session(item) -> dict:
    """One imaging session: N1 gamma (CSM flag; |corr|), N2 mean responses (all scorable / clean / CSM licks)."""
    label, sgn, tg, th = item
    from wfield_local import orofacial_cohort as oc
    z = np.load(lc.SESS / f"{label}.npz", allow_pickle=True)
    an, ep = str(z["animal"]), str(z["epoch"])
    sid = f"{an}_2026{label[-4:]}"
    cols = list(z["cols"])
    M = pd.DataFrame({k: z[k] for k in ["trial_id", "lick_idx", "lick_uid", "side_ml", "reach_mm", "t"]})
    M["row"] = np.arange(len(M))
    M["pos"] = [oc.pos_key(p, "L") for p in z["position"]]
    M["contact_n"] = z["contact"].astype(bool)
    M["ang"] = sgn * np.degrees(np.arcsin(np.clip(M.side_ml / M.reach_mm, -1, 1)))
    # next post-cue lick of the same trial (GCaMP6s overlap covariates)
    M = M.sort_values(["trial_id", "lick_idx"]).reset_index(drop=True)
    nx = M.shift(-1)
    has = (nx.trial_id == M.trial_id) & (nx.lick_idx == M.lick_idx + 1) & ((nx.t - M.t) <= lc.ILI_MAX) & \
        np.isfinite(nx.ang.astype(float))
    M["has_next"] = has.astype(float)
    M["next_ang"] = np.where(has, nx.ang.astype(float), np.nan)
    M["next_contact"] = np.where(has, nx.contact_n.astype(float), 0.0)
    M["next_ili"] = np.where(has, (nx.t - M.t).astype(float), 0.0)
    W = pd.read_pickle(LICKS / f"{sid}.pkl")
    W = W[(W.context == "post_cue")]
    D = M.merge(W, on="lick_uid", how="inner", suffixes=("", "_w"))
    D["theta_L"] *= sgn
    D["theta_E"] *= sgn
    D["e_L"] = D.theta_L - [tg.get((an, p), (np.nan, np.nan))[0] for p in D.pos]
    D["e_E"] = D.theta_E - [tg.get((an, p), (np.nan, np.nan))[1] for p in D.pos]
    D["corr"] = D.e_E - D.e_L
    D = D[D.onset_ok & D.full_ok & np.isfinite(D.theta_L) & np.isfinite(D["corr"]) & D.pk_sp.notna()
          & np.isfinite(D.speed_peak)].reset_index(drop=True)
    base = {"label": label, "animal": an, "epoch": ep, "n": len(D)}
    if len(D) < 2 * MIN_N:
        return {"G": [], "N": [], "n": len(D)}
    D["csm"] = [csm_count(*x, th) for x in D.pk_sp]
    D["csm_f"] = (D.csm >= 1).astype(float)
    D["next_ang"] = D.next_ang.fillna(D.next_ang.mean() if D.has_next.any() else 0.0)
    feats = {f"{a}|{h}": cols.index(f"{a}_{'right' if h == 'c' else 'left'}") for a in AREAS_N for h in ("c", "i")}
    for c in ("RSPd_right", "RSPd_left"):
        if c in cols:
            feats[c.replace("_right", "|c").replace("_left", "|i")] = cols.index(c)
    rows = D.row.to_numpy(int)
    F = lc._spout_fe(D.pos.to_numpy())
    cov = np.column_stack([D.theta_L, D.theta_L ** 2 / 10, D.theta_E, D.reach, D.speed_peak / 100,
                           D.contact_n.astype(float), (D.lick_idx == 0).astype(float), F, D.next_ang, D.next_contact,
                           D.next_ili, D.has_next])
    first = (D.lick_idx == 0).to_numpy()
    clean = ((D.csm == 0) & (D["corr"].abs() < CLEAN_CORR)).to_numpy()
    csmf = D.csm_f.to_numpy()
    G, N = [], []
    for tau in TAUS:
        X3 = lc._deconv(z["X"][rows].astype(float), tau)
        lv = {}
        for k, j in feats.items():
            pre = X3[:, (lc.LAGS_X >= PRE_W[0] - 1e-9) & (lc.LAGS_X < PRE_W[1] - 1e-9), j].mean(1)
            lv[k] = {w: X3[:, (lc.LAGS_X >= a - 1e-9) & (lc.LAGS_X < b - 1e-9), j].mean(1) - pre
                     for w, (a, b) in WINS.items()}
        for h in ("c", "i"):
            lv[f"POOL|{h}"] = {w: sum(lv[f"{a}|{h}"][w] for a in AREAS_N) for w in WINS}
        for k in lv:
            for w in WINS:
                y = lv[k][w]
                for sub, sm in (("all", np.ones(len(y), bool)), ("first", first)):
                    nc, nf = int(csmf[sm].sum()), int((1 - csmf[sm]).sum())
                    r = {**base, "tau": tau, "feat": k, "win": w, "subset": sub, "n_csm": nc, "n_free": nf}
                    if nc >= MIN_N and nf >= MIN_N:
                        X = np.column_stack([np.ones(sm.sum()), cov[sm], csmf[sm]])
                        b, se = lc._ols(X, y[sm])
                        r.update({"gamma_csm": b[-1], "gamma_csm_se": se[-1]})
                    if sm.sum() >= 2 * MIN_N:
                        X = np.column_stack([np.ones(sm.sum()), cov[sm], D["corr"].abs().to_numpy()[sm] / 10])
                        b, se = lc._ols(X, y[sm])
                        r.update({"gamma_corr": b[-1], "gamma_corr_se": se[-1]})
                    G.append(r)
                N.append({**base, "tau": tau, "feat": k, "win": w, "resp_all": float(y.mean()),
                          "resp_clean": float(y[clean].mean()) if clean.sum() >= MIN_N else np.nan,
                          "resp_csm": float(y[csmf == 1].mean()) if csmf.sum() >= MIN_N else np.nan,
                          "n_clean": int(clean.sum()), "n_csm": int(csmf.sum())})
    return {"G": G, "N": N, "n": len(D)}


def imaging_items(sgn: float, tg: dict, th: float) -> list:
    sids = set(_sessions())
    labs = sorted(p.stem for p in lc.SESS.glob("*.npz"))
    return [(lab, sgn, tg, th) for lab in labs if f"{lab[:4]}_2026{lab[-4:]}" in sids]


# ================================================================================================ run
def cmd_run(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    LICKS.mkdir(parents=True, exist_ok=True)
    sids = _sessions()
    pre = [s for s in sids if _is_aug_pre_sid(s)]
    rest = [s for s in sids if s not in pre]
    print(f"{len(sids)} sessions ({len(pre)} August pre first)", flush=True)
    t0 = time.time()
    res, fail = ak.fan_sessions(pre, session_licks, jobs=jobs)
    if fail:
        raise SystemExit(f"pre extraction failed: {fail[:3]}")
    cal = calibrate(load_licks(pre))
    (OUT / "csm_threshold.json").write_text(json.dumps(cal, indent=1))
    print(f"pre extracted {(time.time() - t0) / 60:.1f} min; theta_sp {cal['theta_sp']} theta_ml {cal['theta_ml']} "
          f"(n {cal['n_licks']})", flush=True)
    for kind in ("sp", "ml"):
        print(pd.DataFrame(cal["grid"][kind]).round(3).to_string(index=False), flush=True)
    res, fail = ak.fan_sessions(rest, session_licks, jobs=jobs)
    print(f"all extracted {(time.time() - t0) / 60:.1f} min; failed {fail[:3]}", flush=True)
    if fail:
        raise SystemExit(1)
    pd.DataFrame([v for _k, v in res]).to_csv(OUT / "extract_summary.csv", index=False)
    if cal["theta_sp"] is None:
        print("CSM NOT RESOLVABLE by the pre-registered rule -> neural CSM part skipped", flush=True)
        return
    # neural per-session fits (need the sign + targets: computed exactly as in summarise)
    L = prepare(load_licks(sids, ["sid", "label", "animal", "epoch", "context", "pos", "contact", "onset_ok",
                                  "theta_L", "theta_E", "theta_C", "theta_H", "lick_idx"]))
    sgn, TG = L.attrs["sgn"], L.attrs["TG"]
    tg ={(a, p): (float(x), float(y)) for a, p, x, y in zip(TG.animal, TG.pos, TG.tgt_L, TG.tgt_E)}
    items = imaging_items(sgn, tg, cal["theta_sp"])
    res, fail = ak.fan_sessions(items, neural_session, jobs=jobs, key=lambda x: x[0])
    if fail:
        raise SystemExit(f"neural failed: {[f[0][0] if isinstance(f[0], tuple) else f for f in fail][:3]}")
    G = pd.DataFrame([r for _k, v in res for r in v["G"]])
    Nn = pd.DataFrame([r for _k, v in res for r in v["N"]])
    G.to_csv(OUT / "neural_gamma_sessions.csv", index=False)
    Nn.to_csv(OUT / "neural_resp_sessions.csv", index=False)
    print(f"neural done {(time.time() - t0) / 60:.1f} min ({len(items)} sessions)", flush=True)


# ================================================================================================ summarise
def prepare(L: pd.DataFrame) -> pd.DataFrame:
    """Curated / epoch filter, sign, targets, errors, correction."""
    from wfield_local import config
    from wfield_local import epochs
    L = L[L.epoch.isin(EPS) & L.animal.isin(AN)].copy()
    L["aug_pre"] = (L.epoch == "pre") & (~L.sid.map(config.early_pre))
    L = L[(L.epoch != "pre") | L.aug_pre].copy()
    P = L[L.aug_pre & (L.context == "post_cue")]
    sgn = float(np.sign(P[P.pos == "far_contra"].theta_E.mean()))
    for c in ("theta_L", "theta_E", "theta_C", "theta_H"):
        L[c] = sgn * L[c]
    if "path_ml" in L:
        L["path_ml"] = [tuple(sgn * np.asarray(x)) if isinstance(x, tuple) else x for x in L.path_ml]
    P = L[L.aug_pre & (L.context == "post_cue") & L.contact & L.onset_ok]
    TG = P.groupby(["animal", "pos"]).agg(tgt_L=("theta_L", "mean"), tgt_E=("theta_E", "mean"),
                                          n_tgt=("theta_L", "count")).reset_index()
    L = L.merge(TG[["animal", "pos", "tgt_L", "tgt_E"]], on=["animal", "pos"], how="left")
    cue = L.context == "post_cue"
    L["e_L"] = np.where(cue, L.theta_L - L.tgt_L, np.nan)
    L["e_E"] = np.where(cue, L.theta_E - L.tgt_E, np.nan)
    L["corr"] = L.e_E - L.e_L
    L["toward"] = np.where(L.e_L.abs() >= TOWARD_MIN, -np.sign(L.e_L) * L["corr"], np.nan)
    L["day"] = L.label.map({lab: epochs.days_since_stroke(lab) for lab in sorted(L.label.unique())})
    L.attrs["sgn"], L.attrs["TG"] = sgn, TG
    return L


def add_csm(L: pd.DataFrame, th: float, thm: float) -> pd.DataFrame:
    f = L.full_ok & L.pk_sp.notna()
    for nm, t_, col in (("csm", th, "pk_sp"), ("csm_lo", th * 2 / 3, "pk_sp"), ("csm_hi", th * 3 / 2, "pk_sp"),
                        ("csm_lat", thm, "pk_ml")):
        L[nm] = np.nan
        L.loc[f, nm] = [csm_count(*x, t_) for x in L.loc[f, col]]
    g = f & L.sur_sm_sp.notna()
    L["csm_false"] = np.nan
    L.loc[g, "csm_false"] = [float(csm_count(*a, th) > csm_count(*b, th)) for a, b in zip(L.loc[g, "sur_nz_sp"],
                                                                                       L.loc[g, "sur_sm_sp"])]
    return L


BMET = ["e_L", "e_E", "corr", "toward", "abs_corr", "g_w", "csm_share", "csm_mean", "csm_lat_share", "csm_false",
        "csm_share_lo", "csm_share_hi", "csm_excess", "curv", "prot_ms", "lick_ms", "hyp_mm", "miss_mm", "speed_peak",
        "reach", "onset_cov", "full_cov", "single_share", "theta_C_e"]


def cell(c: pd.DataFrame) -> dict:
    A = c[c.onset_ok & np.isfinite(c.e_L) & np.isfinite(c.e_E)]
    C = c[c.full_ok & np.isfinite(c.csm)]
    o = {"n": len(c), "n_aim": len(A), "n_csm": len(C), "onset_cov": float(c.onset_ok.mean()),
         "full_cov": float(c.full_ok.mean()), "single_share": float(c.single_share.mean()),
         "lick_ms": float(c.lick_ms.mean())}
    if len(A) >= MIN_N:
        o.update({"e_L": A.e_L.mean(), "e_E": A.e_E.mean(), "corr": A["corr"].mean(), "abs_corr": A["corr"].abs().mean(),
                  "curv": A.curv.mean(), "prot_ms": A.prot_ms.mean()})
        tw = A.toward.dropna()
        o["toward"] = tw.mean() if len(tw) >= MIN_N else np.nan
        X = np.column_stack([np.ones(len(A)), -A.e_L.to_numpy(float), lc._spout_fe(A.pos.to_numpy())])
        o["g_w"] = lc._ols(X, A["corr"].to_numpy(float))[0][1]
    elif len(c) >= MIN_N and c.onset_ok.sum() >= MIN_N:
        o.update({"curv": c.curv.mean(), "prot_ms": c.prot_ms.mean()})
    if len(C) >= MIN_N:
        o.update({"csm_share": float((C.csm >= 1).mean()), "csm_mean": float(C.csm.mean()),
                  "csm_lat_share": float((C.csm_lat >= 1).mean()), "csm_share_lo": float((C.csm_lo >= 1).mean()),
                  "csm_share_hi": float((C.csm_hi >= 1).mean()), "csm_false": float(C.csm_false.mean())})
        o["csm_excess"] = o["csm_share"] - o["csm_false"]
    if len(c) >= MIN_N:
        for m in ("hyp_mm", "miss_mm", "speed_peak", "reach"):
            v = c[m].dropna() if m in c else pd.Series(dtype=float)
            if len(v) >= MIN_N:
                o[m] = float(v.mean())
        if "theta_C" in c:
            v = (c.theta_C - c.tgt_E).dropna()
            if len(v) >= MIN_N:
                o["theta_C_e"] = float(v.mean())
    return o


ORDERS = {"any": None, "first": True, "later": False}
OUTC = {"any": None, "hit": True, "miss": False}


def sessions_table(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lab, an, ep, day), g in L.groupby(["label", "animal", "epoch", "day"], sort=True, dropna=False):
        cue = g[g.context == "post_cue"]
        for gn, ps in GROUPS.items():
            gg = cue[cue.pos.isin(ps)]
            for on, of in ORDERS.items():
                h = gg if of is None else gg[(gg.lick_idx == 0) == of]
                for un, uf in OUTC.items():
                    if (on != "any" or un != "any") and gn not in ("all", "far_contra", "other5"):
                        continue
                    c = h if uf is None else h[h.contact == uf]
                    if len(c) < MIN_N:
                        continue
                    rows.append({"label": lab, "animal": an, "epoch": ep, "day": day, "context": "cued", "group": gn,
                                 "order": on, "outcome": un, **cell(c)})
        it = g[g.context == "iti"]
        if len(it) >= MIN_N:
            rows.append({"label": lab, "animal": an, "epoch": ep, "day": day, "context": "iti", "group": "iti",
                         "order": "any", "outcome": "any", **cell(it)})
    return pd.DataFrame(rows)


KEYS = ["context", "group", "order", "outcome"]
PRIMARY = [("e_L", "P1 launch error"), ("corr", "P2 within-lick correction"), ("csm_share", "P3 CSM share")]


def _change(B: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered change: CI excludes 0 AND >= 3/4 animals same sign as the pooled delta."""
    B = B.copy()
    nn = B.n_paired - B.n_pos
    same = np.where(B.delta > 0, B.n_pos, nn)
    B["n_same"] = same
    B["change"] = ((B.delta_lo > 0) | (B.delta_hi < 0)) & (same >= 3)
    return B


def primary_holm(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(SEED)
    g = S[(S.context == "cued") & (S.group == "far_contra") & (S.order == "any") & (S.outcome == "any")]
    rows = []
    for m, nm in PRIMARY:
        gm = g.dropna(subset=[m])
        pre = {a: list(h[m]) for a, h in gm[gm.epoch == "pre"].groupby("animal", sort=True)}
        for ep in EPS[1:]:
            cur = {a: list(h[m]) for a, h in gm[gm.epoch == ep].groupby("animal", sort=True)}
            iv, dr = ak.boot_delta(cur, pre, rng, draws=True)
            rows.append({"metric": m, "name": nm, "epoch": ep, "delta": iv.point if iv else np.nan,
                         "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan, "p": ak.boot_p(dr)})
    H = pd.DataFrame(rows)
    H["p_holm"] = ak.holm(H.p.to_numpy())
    return H


def neural_summary() -> dict:
    from wfield_local import config
    G = pd.read_csv(OUT / "neural_gamma_sessions.csv")
    Nn = pd.read_csv(OUT / "neural_resp_sessions.csv")
    aug = lambda d: d[(d.epoch != "pre") | (~d.label.map(config.early_pre))]  # noqa: E731
    G, Nn = aug(G), aug(Nn)
    GL = G.melt(id_vars=["label", "animal", "epoch", "tau", "feat", "win", "subset"],
                value_vars=["gamma_csm", "gamma_corr"], var_name="term", value_name="v").dropna(subset=["v"])
    # hemisphere difference c - i
    GW = GL.pivot_table(index=["label", "animal", "epoch", "tau", "win", "subset", "term"], columns="feat",
                        values="v").reset_index()
    for a in AREAS_N + ["POOL", "RSPd"]:
        if f"{a}|c" in GW and f"{a}|i" in GW:
            GW[f"{a}|c-i"] = GW[f"{a}|c"] - GW[f"{a}|i"]
    GL = GW.melt(id_vars=["label", "animal", "epoch", "tau", "win", "subset", "term"], var_name="feat",
                 value_name="v").dropna(subset=["v"])
    NL = Nn.melt(id_vars=["label", "animal", "epoch", "tau", "feat", "win"],
                 value_vars=["resp_all", "resp_clean", "resp_csm"], var_name="set", value_name="v").dropna(subset=["v"])
    PAG = lc.per_animal(GL, ["tau", "win", "subset", "term", "feat"], ["v"])
    PAN = lc.per_animal(NL, ["tau", "win", "set", "feat"], ["v"])
    PAG.to_csv(OUT / "neural_gamma_per_animal.csv", index=False)
    PAN.to_csv(OUT / "neural_resp_per_animal.csv", index=False)
    print("\n== PER-ANIMAL (rule 8): gamma_csm (CSM - CSM-free at matched kinematics), raw, 0-0.2 s, all licks, "
          "POOL (dF/F x1e3; c = contralesional, i = ipsilesional) ==")
    q = PAG[(PAG.tau == 0) & (PAG.win == "w0.0-0.2") & (PAG.subset == "all") & (PAG.term == "gamma_csm")
            & PAG.feat.str.startswith("POOL")]
    print(q.pivot_table(index=["feat", "animal"], columns="epoch", values="v_mean").reindex(columns=EPS).round(2)
          .to_string(), flush=True)
    print("\n== PER-ANIMAL: mean lick response, raw, 0-0.2 s, POOL, all scorable vs clean (no CSM, |corr| < 10) ==")
    q = PAN[(PAN.tau == 0) & (PAN.win == "w0.0-0.2") & PAN.feat.str.startswith("POOL")]
    print(q.pivot_table(index=["feat", "set", "animal"], columns="epoch", values="v_mean").reindex(columns=EPS)
          .round(2).to_string(), flush=True)
    BG = _change(lc.boot_table(GL.rename(columns={"v": "v"}), ["tau", "win", "subset", "term", "feat"], ["v"])
                 .dropna(subset=["level"], how="all").fillna({"n_pos": 0, "n_paired": 0}))
    BN = _change(lc.boot_table(NL, ["tau", "win", "set", "feat"], ["v"]).fillna({"n_pos": 0, "n_paired": 0}))
    BG.to_csv(OUT / "neural_gamma_boot.csv", index=False)
    BN.to_csv(OUT / "neural_resp_boot.csv", index=False)
    # carried share = 1 - delta(clean) / delta(all), pooled points
    W = BN[BN.epoch != "pre"].pivot_table(index=["tau", "win", "feat", "epoch"], columns="set",
                                          values="delta").reset_index()
    CH = BN[BN.epoch != "pre"].pivot_table(index=["tau", "win", "feat", "epoch"], columns="set",
                                           values="change").reset_index()
    W = W.merge(CH, on=["tau", "win", "feat", "epoch"], suffixes=("", "_change"))
    W["carried_share"] = 1 - W.resp_clean / W.resp_all
    W["carried"] = (W.resp_all_change == 1) & (W.resp_clean_change == 0) & (W.carried_share >= 0.5)
    W.to_csv(OUT / "neural_carried.csv", index=False)
    return {"BG": BG, "BN": BN, "GL": GL, "NL": NL, "W": W, "PAG": PAG, "PAN": PAN}


def cmd_summarise() -> None:
    from figdata import save_fd

    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    cal = json.loads((OUT / "csm_threshold.json").read_text())
    th, thm = cal["theta_sp"], cal["theta_ml"]
    sids = _sessions()
    L = prepare(load_licks(sids))
    sgn, TG = L.attrs["sgn"], L.attrs["TG"]
    print(f"sign factor {sgn:+.0f}; theta_sp {th} theta_ml {thm}; {len(L)} licks, {L.label.nunique()} sessions",
          flush=True)
    TG.to_csv(OUT / "targets.csv", index=False)
    if th is not None:
        L = add_csm(L, th, thm if thm is not None else th)
    else:
        for c in ("csm", "csm_lo", "csm_hi", "csm_lat", "csm_false"):
            L[c] = np.nan
    keep = [c for c in L.columns if not c.startswith(("pk_", "sur_", "path_"))]
    L[keep].to_csv(OUT / "licks_scored.csv.gz", index=False, float_format="%.5g")
    S = sessions_table(L)
    S.to_csv(OUT / "sessions.csv", index=False)
    PA = lc.per_animal(S, KEYS, BMET)
    PA.to_csv(OUT / "per_animal.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== PER-ANIMAL (rule 8): cued post-cue, FAR CONTRA, any order / outcome (deg; + = contralesional; "
          "corr + = toward the far-contra target) ==")
    q = PA[(PA.context == "cued") & (PA.group == "far_contra") & (PA.order == "any") & (PA.outcome == "any")]
    print(q[["animal", "epoch", "e_L_mean", "e_E_mean", "corr_mean", "toward_mean", "g_w_mean", "csm_share_mean",
             "csm_false_mean", "csm_lat_share_mean", "curv_mean", "hyp_mm_mean", "e_L_count"]].round(3)
          .to_string(index=False), flush=True)
    t0 = time.time()
    Sb = S[(S.group.isin(["all", "far_contra", "other5", "iti"])) | ((S.order == "any") & (S.outcome == "any"))]
    B = _change(lc.boot_table(Sb, KEYS, BMET).fillna({"n_pos": 0, "n_paired": 0}))
    B.to_csv(OUT / "boot.csv", index=False)
    print(f"boot {(time.time() - t0) / 60:.1f} min", flush=True)
    H = primary_holm(S)
    H.to_csv(OUT / "primary_holm.csv", index=False)
    print("\n== PRIMARY (far contra, cued, any order / outcome), boot_delta vs August pre + Holm over 9 ==")
    print(H.round(3).to_string(index=False), flush=True)
    aim = pd.read_csv(lc.OUT / "corr_session_aim.csv").set_index("label").aim
    TR = lc.track(S[(S.context == "cued") & (S.group == "far_contra") & (S.order.isin(["any", "first", "later"]))
                    & (S.outcome == "any")], ["group", "order"],
                  ["e_L", "e_E", "corr", "toward", "g_w", "csm_share", "csm_excess"], aim)
    TR.to_csv(OUT / "track.csv", index=False)
    # mean normalised paths (far contra + far ipsi, cued, onset-scored), per animal x epoch x spout
    pp = L[(L.context == "post_cue") & L.pos.isin(["far_contra", "far_ipsi", "far_mid"]) & L.path_ml.notna()]
    PATH = []
    for (an, ep, pos), h in pp.groupby(["animal", "epoch", "pos"]):
        PATH.append({"animal": an, "epoch": ep, "pos": pos, "n": len(h),
                     "ml": np.mean(np.stack(h.path_ml.to_numpy()), 0), "r": np.mean(np.stack(h.path_r.to_numpy()), 0)})
    FN = neural_summary() if (OUT / "neural_gamma_sessions.csv").exists() else None
    save_fd("within_lick_correction", {"S": S, "B": B, "H": H, "TR": TR, "PATH": pd.DataFrame(PATH), "cal": cal,
                                       "sgn": sgn, "FN": FN}, root=FD)
    print("figdata saved", flush=True)


# ================================================================================================ figures
STEP, PADF = 0.075, 0.03


def star_top(ax, x, text="*", color="k", row=0, fontsize=12):
    """As main's `sig_marks.star_top` (not in this worktree): mark in a reserved top band, never on data."""
    try:
        from wfield_local.sig_marks import star_top as st
        return st(ax, x, text, color, row=row, fontsize=fontsize)
    except ImportError:
        import matplotlib.patheffects as pe
        t = ax.text(x, 0.99 - STEP * row, text, color=color, ha="center", va="top", fontsize=fontsize,
                    fontweight="bold", transform=ax.get_xaxis_transform(), clip_on=False, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="w")])
        ax._wf_mark_rows = max(getattr(ax, "_wf_mark_rows", 0), row + 1)
        return t


def reserve_headroom(fig):
    try:
        from wfield_local.sig_marks import reserve_headroom as rh
        return rh(fig)
    except ImportError:
        for ax in fig.get_axes():
            n = getattr(ax, "_wf_mark_rows", 0)
            if n:
                lo, hi = ax.get_ylim()
                frac = STEP * n + PADF
                ax.set_ylim(lo, hi + (hi - lo) * frac / (1 - frac))


def _wrap(s, n=110):
    import textwrap
    return "\n".join(textwrap.fill(p, n) for p in s.split("\n"))


def _sel(D, **kw):
    for k, v in kw.items():
        D = D[D[k] == v]
    return D


def epoch_panel(ax, S, B, m, sel, color="k", row=0, off=0.0, label=None, animals=True, holm=None):
    """SESSIONS (Priya 10-30: one open marker per session, marker = animal, small deterministic jitter) + pooled level
    and 95 % CI (thick); star (top band) where the epoch's change vs pre meets the pre-registered criterion (CI excludes
    0 AND >= 3/4 animals same sign)."""
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    x = np.arange(len(EPS)) + off
    s = _sel(S, **sel)
    if animals:
        # Priya 10-30: marker shape = animal, colour = the usual EPOCH colour (single-series panels); in panels with
        # several series (hit / miss, hemispheres, sets) the series colour is kept so the series stay distinguishable
        rng = np.random.default_rng(7)
        single = color == "k"
        for an, h in s.groupby("animal"):
            v = h.groupby(["epoch", "label"])[m].mean().dropna()
            ep_of = v.index.get_level_values(0)
            xi = np.array([x[EPS.index(e)] for e in ep_of], float)
            xi = xi + rng.uniform(-0.05, 0.05, len(xi)) + 0.09
            cs = [EPOCH_LINE[e] for e in ep_of] if single else [color] * len(xi)
            ax.scatter(xi, v.to_numpy(float), s=11, facecolors="none", edgecolors=cs, alpha=0.6,
                       marker=ANIMAL_MARKER.get(an, "o"), linewidths=0.8, zorder=2)
    b = _sel(B[B.metric == m], **sel).set_index("epoch").reindex(EPS)
    # pooled level + 95 % CI, NO connecting line between epochs (Priya 10-30); epoch colour in single-series panels
    for i, ep in enumerate(EPS):
        if ep not in b.index or not np.isfinite(b.loc[ep, "level"]):
            continue
        c_ = EPOCH_LINE[ep] if color == "k" else color
        lv = b.loc[ep]
        ax.errorbar(x[i] - 0.06, lv.level, yerr=[[lv.level - lv.level_lo], [lv.level_hi - lv.level]], color=c_,
                    mfc=c_, mec="k" if color == "k" else c_, ls="none", lw=2.2, marker="o", ms=6, capsize=3,
                    label=label if i == 0 else None, zorder=4)
    for i, ep in enumerate(EPS[1:], start=1):
        if ep in b.index and b.loc[ep, "change"] == 1:
            txt = "*"
            if holm is not None:
                hh = holm[(holm.metric == m) & (holm.epoch == ep)]
                if len(hh) and hh.p_holm.iloc[0] < 0.05:
                    txt = "**"
            star_top(ax, x[i], txt, color, row=row)
    ax.set_xticks(np.arange(len(EPS)), EPS, fontsize=8)
    ax.tick_params(labelsize=8)


def _pre_line(ax, S, m, sel):
    s = _sel(S, **sel)
    v = s[s.epoch == "pre"].groupby("animal")[m].mean().mean()
    if np.isfinite(v):
        ax.axhline(v, color="0.6", lw=0.8, ls=":", zorder=0)


def fig_primary(D):
    import matplotlib.pyplot as plt
    S, B, H = D["S"], D["B"], D["H"]
    sel = {"context": "cued", "group": "far_contra", "order": "any", "outcome": "any"}
    panels = [("e_L", "A. LAUNCH error e_L (deg; - = ipsilesional of the pre hit pattern) -- P1", True),
              ("e_E", "B. ENDPOINT error e_E at the extension peak (deg)", False),
              ("corr", "C. WITHIN-LICK correction e_E - e_L (deg; + = toward the far-contra target) -- P2", True),
              ("csm_share", "D. share of licks with >= 1 corrective submovement (CSM) -- P3", True),
              ("csm_false", "E. noise control: share a noise surrogate gains a CSM (same threshold)", False),
              ("g_w", "F. within-lick GAIN (slope of correction on -launch error; 1 = full return)", False)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.4), layout="constrained")
    for ax, (m, tt, prim) in zip(axs.ravel(), panels):
        epoch_panel(ax, S, B, m, sel, holm=H if prim else None)
        if m in ("e_L", "e_E", "corr"):
            ax.axhline(0, color="0.5", lw=0.7)
        else:
            _pre_line(ax, S, m, sel)
        ax.set_title(_wrap(tt, 52), fontsize=9)
    axs[1, 1].sharey(axs[1, 0])
    reserve_headroom(fig)
    th = D["cal"].get("theta_sp")
    fig.suptitle(_wrap(f"Within-lick correction at the FAR-CONTRA spout (cued post-cue licks, all orders and "
                       f"outcomes; PS92-95, August pre). Filled = pooled session level with 95 % nested bootstrap CI; "
                       f"open markers = SESSIONS (shape = animal: o PS92, s PS93, ^ PS94, D PS95; color = epoch). Top-band * = change vs pre (CI excludes 0 "
                       f"AND >= 3/4 animals same sign); ** = also Holm over the 9 primary tests. Angles + = "
                       f"contralesional. CSM threshold {th} mm/s speed-peak prominence (noise-calibrated on August "
                       f"pre). Dotted = pre level.", 175), fontsize=9.5)
    return fig


def fig_spouts(D):
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    S, B = D["S"], D["B"]
    mets = [("e_L", "A. launch error e_L, change vs pre (deg)"), ("corr", "B. within-lick correction, change vs pre "
            "(deg)"), ("csm_share", "C. CSM share, change vs pre"), ("e_E", "D. endpoint error e_E, change vs pre "
                                                                         "(deg)")]
    fig, axs = plt.subplots(2, 2, figsize=(15, 9.6), layout="constrained")
    for ax, (m, tt) in zip(axs.ravel(), mets):
        for j, ep in enumerate(EPS[1:]):
            off = (j - 1) * 0.22
            b = B[(B.metric == m) & (B.context == "cued") & (B.order == "any") & (B.outcome == "any")
                  & B.group.isin(POS) & (B.epoch == ep)].set_index("group").reindex(POS)
            x = np.arange(len(POS)) + off
            ax.errorbar(x, b.delta, yerr=[b.delta - b.delta_lo, b.delta_hi - b.delta], fmt="o", color=EPOCH_LINE[ep],
                        ms=6, lw=1.8, capsize=2.5, label=ep, zorder=4)
            s = S[(S.context == "cued") & (S.order == "any") & (S.outcome == "any") & S.group.isin(POS)]
            rng = np.random.default_rng(7 + j)
            for an, h in s.groupby("animal"):                    # sessions: each post session minus its animal's pre
                pre = h[h.epoch == "pre"].groupby("group")[m].mean()
                cur = h[h.epoch == ep].set_index("group")[m]
                d = (cur - pre.reindex(cur.index)).dropna()
                xi = np.array([POS.index(g) for g in d.index], float) + off + 0.06 + rng.uniform(-0.03, 0.03, len(d))
                ax.plot(xi, d.to_numpy(float), ls="none", marker=ANIMAL_MARKER[an], mfc="none", mec=EPOCH_LINE[ep],
                        ms=3, alpha=0.5, mew=0.7)
            for i, p in enumerate(POS):
                if p in b.index and b.loc[p, "change"] == 1:
                    star_top(ax, x[i], "*", EPOCH_LINE[ep], row=j)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xticks(range(len(POS)), [p.replace("_", "\n") for p in POS], fontsize=8)
        ax.set_title(tt, fontsize=9.5)
        ax.tick_params(labelsize=8)
    reserve_headroom(fig)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=3, fontsize=9, frameon=False,
               title="epoch (pooled change vs August pre, 95 % CI; open markers = sessions minus own pre)", title_fontsize=9)
    fig.suptitle(_wrap("Per spout (canonical order), cued post-cue licks, all orders / outcomes. Top-band * "
                       "(one row per epoch, epoch color) = change vs pre by the pre-registered criterion. Angles + = "
                       "contralesional; correction + = endpoint more contralesional than launch.", 170), fontsize=9.5)
    return fig


def fig_split(D):
    import matplotlib.pyplot as plt
    S, B = D["S"], D["B"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.4), layout="constrained")
    cols = {"hit": "#1a9850", "miss": "#d73027"}
    for r, on in enumerate(["first", "later"]):
        for c, (m, tt) in enumerate([("e_L", "launch error e_L (deg)"), ("corr", "within-lick correction (deg)"),
                                     ("csm_share", "CSM share")]):
            ax = axs[r, c]
            for j, (oc_, col) in enumerate(cols.items()):
                epoch_panel(ax, S, B, m, {"context": "cued", "group": "far_contra", "order": on, "outcome": oc_},
                            color=col, row=j, off=(j - 0.5) * 0.12, label=f"lick {oc_.upper()}s (contact = "
                                                                          f"{'yes' if oc_ == 'hit' else 'no'})")
            if m != "csm_share":
                ax.axhline(0, color="0.5", lw=0.7)
            ax.set_title(f"{'ABCDEF'[r * 3 + c]}. far contra, {on.upper()} licks: {tt}", fontsize=9.5)
    reserve_headroom(fig)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(_wrap("Far-contra spout split by lick order (FIRST post-cue lick vs LATER licks) and outcome "
                       "(HIT = the lick contacted the spout). Filled = pooled (95 % CI), open markers = sessions; top-band * "
                       "(row per outcome, its color) = change vs pre (CI excludes 0 AND >= 3/4 same sign).", 170),
                 fontsize=9.5)
    return fig


def fig_secondary(D):
    import matplotlib.pyplot as plt
    S, B = D["S"], D["B"]
    cols = {"far contra": ({"context": "cued", "group": "far_contra"}, "#b2182b"),
            "other 5 spouts": ({"context": "cued", "group": "other5"}, "#2166ac"),
            "ITI licks": ({"context": "iti", "group": "iti"}, "0.4")}
    mets = [("curv", "A. path curvature (path / chord, onset -> peak)"), ("prot_ms", "B. protrusion duration (ms)"),
            ("hyp_mm", "C. hypometria (mm past (+) / short of (-) the spout)"),
            ("speed_peak", "D. peak tip speed (mm/s)"), ("csm_lat_share", "E. lateral-velocity CSM share"),
            ("csm_share", "F. CSM share (cued vs ITI)"), ("full_cov", "G. tracking: share of licks fully tracked"),
            ("single_share", "H. tracking: share of single-camera frames")]
    fig, axs = plt.subplots(2, 4, figsize=(19, 9.4), layout="constrained")
    for ax, (m, tt) in zip(axs.ravel(), mets):
        for j, (nm, (sel, col)) in enumerate(cols.items()):
            if m in ("hyp_mm",) and sel["context"] == "iti":
                continue
            if len(_sel(B[B.metric == m], **sel, order="any", outcome="any").dropna(subset=["level"])) == 0:
                continue
            epoch_panel(ax, S, B, m, {**sel, "order": "any", "outcome": "any"}, color=col, row=j,
                        off=(j - 1) * 0.12, label=nm)
        ax.set_title(_wrap(tt, 46), fontsize=9.5)
    reserve_headroom(fig)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(_wrap("Secondary kinematics and tracking controls (all orders / outcomes). Filled = pooled "
                       "(95 % CI), thin = animals; top-band * (row per line) = change vs pre (CI excludes 0 AND >= 3/4 "
                       "same sign).", 170), fontsize=9.5)
    return fig


def fig_paths(D):
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    P = D["PATH"]
    fig, axs = plt.subplots(3, 4, figsize=(16, 11.5), layout="constrained", sharex="row", sharey="row")
    for r, pos in enumerate(["far_ipsi", "far_mid", "far_contra"]):
        for c, an in enumerate(AN):
            ax = axs[r, c]
            for ep in EPS:
                h = P[(P.animal == an) & (P.epoch == ep) & (P.pos == pos)]
                if not len(h):
                    continue
                ml, rr = h.ml.iloc[0], h.r.iloc[0]
                ax.plot(ml, rr, color=EPOCH_LINE[ep], lw=2, label=f"{ep} (n {int(h.n.iloc[0])})")
                ax.plot(ml[0], rr[0], "o", color=EPOCH_LINE[ep], ms=4)
            ax.axvline(0, color="0.8", lw=0.6)
            ax.set_title(f"{an}, {pos.replace('_', ' ')}", fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="lower left")
            ax.tick_params(labelsize=8)
            if c == 0:
                ax.set_ylabel("distance from the mouth zero (mm)", fontsize=8)
            if r == 2:
                ax.set_xlabel("ml (mm; + = contralesional)", fontsize=8)
    fig.suptitle(_wrap("Mean tongue path from the first tracked frame (dot) to the extension peak, cued post-cue "
                       "licks at the three FAR spouts, per animal and epoch (11 time-normalized points, SG-smoothed). "
                       "A launch bias that is corrected inside the lick bends the path toward + ml late; a straight "
                       "biased path means no within-lick correction.", 170), fontsize=9.5)
    return fig


def fig_pre_signatures(D):
    """PRE-STROKE validation of the CSM measure against Bollu et al. 2021 (Priya 10-30: "what figure is this"):
    CSM share per spout (more at FAR targets) and cued vs ITI licks (more for cue-evoked licks), August pre sessions;
    noise-surrogate false rate alongside."""
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    S = D["S"]
    P = S[(S.epoch == "pre") & (S.order == "any") & (S.outcome == "any")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained", gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axs[0]
    rng = np.random.default_rng(5)
    q = P[(P.context == "cued") & P.group.isin(POS)]
    for j, an in enumerate(AN):
        h = q[q.animal == an]
        for m, col in (("csm_share", "k"), ("csm_false", "0.6")):
            xs = np.array([POS.index(g) for g in h.group], float) - 0.2 + 0.08 * j + (0.0 if m == "csm_share" else 0.4)
            ax.scatter(xs + rng.uniform(-0.02, 0.02, len(xs)), h[m], s=14, marker=ANIMAL_MARKER[an], facecolors="none",
                       edgecolors=col, linewidths=0.8, alpha=0.75)
    for i, p in enumerate(POS):
        for m, col, dx in (("csm_share", "k", 0.15), ("csm_false", "0.6", 0.55)):
            v = q[q.group == p].groupby("animal")[m].mean().mean()
            ax.plot(i + dx, v, "o", color=col, mec="k", ms=7, zorder=5)
    ax.set_xticks(np.arange(len(POS)) + 0.2, [p.replace("_", "\n") for p in POS], fontsize=9)
    ax.set_ylabel("share of licks with >= 1 corrective submovement", fontsize=9)
    ax.set_title("A. Pre-stroke, cued licks by spout: black = real licks, gray = noise surrogate (false rate)",
                 fontsize=10, loc="left")
    ax = axs[1]
    for k, (ctx, grp) in enumerate((("cued", "all"), ("iti", "iti"))):
        h = P[(P.context == ctx) & (P.group == grp)]
        for j, an in enumerate(AN):
            g = h[h.animal == an]
            ax.scatter(k - 0.15 + 0.07 * j + rng.uniform(-0.02, 0.02, len(g)), g.csm_share, s=14,
                       marker=ANIMAL_MARKER[an], facecolors="none", edgecolors="k", linewidths=0.8, alpha=0.75)
        ax.plot(k + 0.25, h.groupby("animal").csm_share.mean().mean(), "o", color="k", ms=7, zorder=5)
    ax.set_xticks([0, 1], ["cued licks", "ITI licks"])
    ax.set_xlim(-0.5, 1.6)
    ax.set_title("B. Pre-stroke, cued vs ITI", fontsize=10, loc="left")
    fig.suptitle(_wrap("Validation of the corrective-submovement measure on PRE-STROKE licks (August; open markers = "
                       "sessions, shape = animal; filled = mean of animals): as in Bollu et al. 2021, corrections are "
                       "more frequent at FAR spouts and for cue-evoked licks, and well above the noise-surrogate rate.",
                       150), fontsize=10)
    return fig


def fig_paths_pooled(D):
    """Fig 5b (Priya 10-30): Fig 5 pooled -- per epoch and far spout, the mean over ANIMALS of each animal's mean
    path (animals equally weighted), thin = animals."""
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    P = D["PATH"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6), layout="constrained", sharey=True)
    for c, pos in enumerate(["far_ipsi", "far_mid", "far_contra"]):
        ax = axs[c]
        for ep in EPS:
            h = P[(P.epoch == ep) & (P.pos == pos)]
            if not len(h):
                continue
            ML, R = np.stack(h.ml.to_numpy()), np.stack(h.r.to_numpy())
            for i in range(len(h)):
                ax.plot(ML[i], R[i], color=EPOCH_LINE[ep], lw=0.6, alpha=0.3)
            ml, rr = ML.mean(0), R.mean(0)
            ax.plot(ml, rr, color=EPOCH_LINE[ep], lw=2.6, label=f"{ep} ({len(h)} animals, {int(h.n.sum())} licks)")
            ax.plot(ml[0], rr[0], "o", color=EPOCH_LINE[ep], ms=5)
        ax.axvline(0, color="0.8", lw=0.6)
        ax.set_title(pos.replace("_", " "), fontsize=11)
        ax.set_xlabel("ml (mm; + = contralesional)", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax.tick_params(labelsize=8)
    axs[0].set_ylabel("distance from the mouth zero (mm)", fontsize=9)
    fig.suptitle(_wrap("POOLED mean tongue path (first tracked frame = dot -> extension peak), cued post-cue "
                       "licks at the three FAR spouts: thick = mean of the four animals' mean paths (animals equally "
                       "weighted), thin = animals. Paths shift ipsilesionally from the launch and stay roughly "
                       "parallel: the bias is set at launch, not corrected inside the lick.", 150), fontsize=10)
    return fig


def fig_neural(D):
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    FN = D["FN"]
    BG, GL, BN, NL = FN["BG"], FN["GL"], FN["BN"], FN["NL"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.6), layout="constrained")
    hc = {"contralesional (right)": ("|c", "#b2182b"), "ipsilesional (left)": ("|i", "#2166ac")}
    base = {"tau": 0.0, "win": "w0.0-0.2"}
    for c, (sub, term, tt) in enumerate([("all", "gamma_csm", "A. CSM - CSM-free lick response at matched "
                                                              "kinematics, all licks"),
                                         ("first", "gamma_csm", "B. same, FIRST licks only (no preceding lick)"),
                                         ("all", "gamma_corr", "C. response per 10 deg of |within-lick correction|")]):
        ax = axs[0, c]
        for j, (nm, (sfx, col)) in enumerate(hc.items()):
            sel = {**base, "subset": sub, "term": term, "feat": f"POOL{sfx}"}
            S_ = GL.rename(columns={"v": "v"})
            epoch_panel(ax, S_, BG.assign(metric="v"), "v", sel, color=col, row=j, off=(j - 0.5) * 0.12, label=nm)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_title(_wrap(tt + " (orofacial pool, raw, 0-0.2 s)", 50), fontsize=9.5)
        ax.set_ylabel("dF/F x1e3", fontsize=8)
    for c, (sfx, hn) in enumerate([("|c", "contralesional"), ("|i", "ipsilesional")]):
        ax = axs[1, c]
        for j, (st, col) in enumerate([("resp_all", "k"), ("resp_clean", "#1a9850"), ("resp_csm", "#d73027")]):
            epoch_panel(ax, NL, BN.assign(metric="v"), "v", {**base, "set": st, "feat": f"POOL{sfx}"}, color=col,
                        row=j, off=(j - 1) * 0.12, label={"resp_all": "all scorable licks",
                                                          "resp_clean": "clean licks (no CSM, |corr| < 10 deg)",
                                                          "resp_csm": "CSM licks"}[st])
        ax.set_title(f"{'DE'[c]}. mean lick response, {hn} orofacial pool (raw, 0-0.2 s)", fontsize=9.5)
        ax.set_ylabel("dF/F x1e3", fontsize=8)
    ax = axs[1, 2]
    W = FN["W"]
    w = W[(W.tau == 0) & (W.win == "w0.0-0.2")]
    names = [f"{a}|{h}" for a in AREAS_N + ["POOL"] for h in ("c", "i")]
    for ep in EPS[1:]:
        from wfield_local.orofacial_cohort import EPOCH_LINE
        ww = w[w.epoch == ep].set_index("feat").reindex(names)
        ax.plot(ww.resp_all, ww.resp_clean, ls="none", marker="o", color=EPOCH_LINE[ep], label=ep, ms=6)
    lim = np.nanmax(np.abs(w[["resp_all", "resp_clean"]].to_numpy(float))) * 1.1 if len(w) else 1
    ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.8)
    ax.axhline(0, color="0.85", lw=0.6)
    ax.axvline(0, color="0.85", lw=0.6)
    ax.set_xlabel("change vs pre, ALL licks (dF/F x1e3)", fontsize=8)
    ax.set_ylabel("change vs pre, CLEAN licks", fontsize=8)
    ax.set_title(_wrap("F. each area x hemisphere: on the diagonal = the rise is NOT carried by correction licks; "
                       "near 0 on y = carried", 50), fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _ = ANIMAL_MARKER
    reserve_headroom(fig)
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(_wrap("NEURAL (imaging sessions): is the larger post-stroke lick response carried by "
                       "correction licks? Row 1: per-session OLS coefficient of the CSM flag (or |correction|) on the "
                       "lick-evoked response (0-0.2 s minus -0.2..-0.1 s) with launch / endpoint angle, reach, speed, "
                       "contact, order, spout and the NEXT lick's angle / contact / interval as covariates. Row 2: mean "
                       "responses. Pooled 95 % CI + sessions; top-band * = change vs pre (CI excludes 0 AND >= 3/4).",
                       175), fontsize=9.5)
    return fig


def cmd_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figdata import load_fd
    D = load_fd("within_lick_correction", root=FD)
    FIGS.mkdir(parents=True, exist_ok=True)
    figs = {"wlc_fig1_primary_far_contra": fig_primary, "wlc_fig2_per_spout": fig_spouts,
            "wlc_fig3_order_outcome": fig_split, "wlc_fig4_secondary_tracking": fig_secondary,
            "wlc_fig5_mean_paths": fig_paths, "wlc_fig5b_mean_paths_pooled": fig_paths_pooled,
            "wlc_fig0_pre_signatures": fig_pre_signatures}
    if D.get("FN") is not None:
        figs["wlc_fig6_neural"] = fig_neural
    for name, fn in figs.items():
        fig = fn(D)
        save_png_svg(fig, FIGS / f"{name}.png", dpi=110)
        plt.close(fig)
        print(f"-> {FIGS / name}.png", flush=True)


def main() -> int:
    # coordinator 11-04: routed (out_root under the run tag), no longer refused in a tagged run
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args else "run"
    if "--plot-only" in sys.argv or cmd == "plot":
        cmd_plot()
        return 0
    jobs = min(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 3, 3)
    if cmd == "run":
        cmd_run(jobs)
    elif cmd == "summarise":
        cmd_summarise()
        cmd_plot()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
