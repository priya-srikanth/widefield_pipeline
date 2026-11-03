"""Q5 of the face pre-launch prereg (docs/status/FACE_PRELAUNCH_AIM_PREREG_20261031.md section 7): does partialling the
FACE out of the cortical features change the frozen KEY lateral decoder's post-stroke OFFSET?

Called as `python scripts/face_prelaunch_aim.py offset [--jobs 4]` (this module sets WFIELD_ITI_FILTER=dock and
redirects iti_licks.OUT_W to <face_prelaunch_aim>/offset/_itiw BEFORE importing the offset scripts, so nothing of theirs
is overwritten; it only CALLS their functions).

Steps
  1. collect   the decoder's per-lick inputs exactly as iti_lick_order.predictions (Allen split r8, lags 0..0.2 s from the
               lick peak, frame-SD z-scored; ITI filter none = all ITI for training) + each lick's peak time t
  2. before    the original decoder (RidgeCV on August pre, pre LOSO) -> must equal the lick_order_preds cache
  3. REPRODUCTION CHECK (gate): docked ITI same-place offset (lick_matched_readout --ref loclin, dock filter) and cued
               matched-error offset (lick_balance_matched_error off_spout; per-lick read-out rebuilt like
               lick_offset_contrib_map.session_dh, sigma 0.1 mm) from the recomputed 'before' must reproduce the
               published values to 0.01 mm; else STOP.
  4. face covariates per lick (FR 11 traces / 5 groups / + points PCs / + jaw), window means [-0.2, 0) and [0, 0.2] s
  5. variants  A (pre-fit regression, LOSO for pre, original decoder), B (decoder refit on residualised pre),
               C (per-session regression), planted-signal control; offsets per variant; boot_delta_columns before/after
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import face_prelaunch_aim as fpa  # noqa: E402

OFFD = fpa.OFFD
os.environ["WFIELD_ITI_FILTER"] = "dock"
os.environ["WFIELD_ITI_OUT_W"] = str(OFFD / "_itiw")
_argv = list(sys.argv)
sys.argv = [sys.argv[0], "--ref", "loclin"]
import iti_first_lick as fl  # noqa: E402
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
import lick_balance_matched_error as lbme  # noqa: E402
import lick_matched_readout as lmr  # noqa: E402

sys.argv = _argv

AN, EPS = fpa.AN, fpa.EPS
EP = EPS[1:]
PUBLISHED = {("iti_dock_sameplace", "acute"): 0.298, ("iti_dock_sameplace", "subacute"): 0.216,
             ("iti_dock_sameplace", "chronic"): 0.290, ("cued_matched_error", "acute"): -0.035,
             ("cued_matched_error", "subacute"): -0.033, ("cued_matched_error", "chronic"): -0.019}
TOL = 0.01
SIGMA_RO = 0.1        # lick_offset_contrib_map.SIGMA (lateral same-place read-out)
MIN_REF = 8
GRID = np.round(np.arange(-0.195, 0.2, 0.01), 3)        # 40 samples: [-0.2, 0) and [0, 0.2)
COVSETS = ("FR", "FR5", "FRPTS", "FRJAW")
PRIMARY_COV = "FR"
MEASURES = ("iti_dock_sameplace", "cued_matched_error", "cued_sameplace", "iti_dock_matched_error")


def log(m):
    print(m, flush=True)
    with open(OFFD / "offset_log.txt", "a") as f:
        f.write(m + "\n")


# ------------------------------------------------------------------------------------------------ 1. collect
JOBS = [4]


def collect() -> dict:
    f = OFFD / "collect.pkl"
    if f.exists():
        return pd.read_pickle(f)
    from wfield_local import orofacial_cohort as oc
    assert lo.SOURCE == "areas_split", lo.SOURCE
    assert lo.ITI_FILTER == "", lo.ITI_FILTER
    D, names, SD = wd.area_basis(basis="split", jobs=JOBS[0])
    out = {}
    for st in lo.SETS:
        for a in AN:
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                t = il._load(x)["iti_t" if st == "iti" else "tr_t"].astype(float)
                idx = lo._bout_idx(x, t) if st == "iti" else fl._lick_idx(x, t)
                pos = np.array([q if q in lo.POS or "_" not in q else oc.pos_key(q, "L") for q in d[f"{st}_pos"].astype(str)])
                ok = np.isfinite(C[:, 0]) & np.isfinite(idx) & np.isin(pos, lo.POS)
                if st == "iti":
                    ok &= lo.iti_keep(x, t)
                if ok.sum() >= 20:
                    sd = SD.loc[x, names].to_numpy(float)
                    S[x] = (d["epoch"], X[ok] / sd[None, :], C[ok, 0], idx[ok], pos[ok], t[ok])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            out[(st, a)] = S
    pd.to_pickle(out, f)
    return out


def fit_dec(S: dict, X: dict) -> dict:
    """The pre-LOSO / all-pre ridge read-out. ONE definition (Priya 11-03): `frozen_readouts.fit_all`, the same
    StandardScaler + RidgeCV construction, called from inside the stored spec of `refit_frozen` (kind face_aim, which
    carries the training-data fingerprint) -- so this read-out is stored, never refitted per render."""
    from wfield_local import frozen_readouts
    pre = [x for x, v in S.items() if v[0] == "pre"]
    return frozen_readouts.fit_all(pre, {q: X[q] for q in pre}, {q: S[q][2] for q in pre}, np.logspace(-1, 5, 13))


def score(C: dict, X: dict, fits: dict) -> pd.DataFrame:
    """Prediction table in the lick_order_preds layout and order (sets, animals, sessions)."""
    rows = []
    for st in lo.SETS:
        for a in AN:
            if (st, a) not in C:
                continue
            S = C[(st, a)]
            pre = [x for x, v in S.items() if v[0] == "pre"]
            for x, (ep, _X, y, idx, pos, _t) in S.items():
                sc, m = fits[(st, a)][x if x in pre else None]
                rows.append(pd.DataFrame({"set": st, "animal": a, "label": x, "epoch": ep, "pos": pos, "order": idx,
                                          "act": y, "dec": m.predict(sc.transform(X[(st, a)][x]))}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------------------------ 3. offsets
def measure_sameplace(P: pd.DataFrame, tag: str) -> pd.DataFrame:
    f = OFFD / f"preds_{tag}.csv.gz"
    P.to_csv(f, index=False)
    lmr.lo.preds_path = lambda: f
    Q = lmr.per_lick()
    S = lmr.sessions(Q)
    S = S[(S.scope == "all spouts") & (S.cls == "all")]
    out = S[["set", "animal", "label", "epoch", "n", "resid"]].rename(columns={"resid": "value"})
    out["measure"] = np.where(out.set == "iti", "iti_dock_sameplace", "cued_sameplace")
    return out.drop(columns="set")


def readouts(P: pd.DataFrame) -> pd.DataFrame:
    """Per-lick like-for-like read-out (lick_offset_contrib_map.session_dh, lateral rule): dec - Gaussian (0.1 mm) mean
    of the held-out-decoded pre licks of OTHER sessions at the same landing place; sessions with >= 8 such licks."""
    rows = []
    for (st, _a), g in P.groupby(["set", "animal"], sort=False):
        pre = g[g.epoch == "pre"]
        ap, dp, lp = pre.act.to_numpy(float), pre.dec.to_numpy(float), pre.label.to_numpy(str)
        for x, h in g.groupby("label", sort=False):
            ref = lp != x
            act = h.act.to_numpy(float)
            K = np.exp(-0.5 * ((act[:, None] - ap[None, ref]) / SIGMA_RO) ** 2)
            ok = K.sum(1) > 1e-6
            if ok.sum() < MIN_REF:
                continue
            Kn = K[ok] / K[ok].sum(1, keepdims=True)
            hh = h[ok]
            rows.append(pd.DataFrame({"set": st, "label": x, "act": hh.act.to_numpy(), "order": hh.order.to_numpy(),
                                      "pos": hh.pos.to_numpy(), "readout": hh.dec.to_numpy(float) - Kn @ dp[ref]}))
    return pd.concat(rows, ignore_index=True)


def measure_matched(P: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    T = readouts(P)

    def _ro(L, T=T):
        T = T.copy()
        key = ["set", "label", "order", "pos", "act"]
        L = L.copy()
        for df in (T, L):
            df["act"] = df.act.round(3)
            df["dup"] = df.groupby(key).cumcount()
        return L.merge(T, on=[*key, "dup"], how="left")
    lbme._readouts = _ro
    lbme.OUT = il.OUT / "itifilter_dock"                 # read-only: the docked baltarget_licks copy (cued rows = all)
    S, _T, _K = lbme.compute()
    S = S[["set", "animal", "label", "epoch", "n", "off_spout"]].rename(columns={"off_spout": "value"}).dropna(subset=["value"])
    S["measure"] = np.where(S.set == "iti", "iti_dock_matched_error", "cued_matched_error")
    return S.drop(columns="set"), T


def offsets(P: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    A = measure_sameplace(P, tag)
    B, T = measure_matched(P)
    return pd.concat([A, B], ignore_index=True), T


def tests(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rows = []
    for m, g in S.groupby("measure"):
        pre = {a: h.value.to_numpy() for a, h in sorted(g[g.epoch == "pre"].groupby("animal"))}
        for ep in EP:
            post = {a: h.value.to_numpy() for a, h in sorted(g[g.epoch == ep].groupby("animal")) if a in pre}
            if not post:
                continue
            ci = ak.boot_delta(post, pre, np.random.default_rng(0))
            rows.append({"measure": m, "epoch": ep, "point": ci.point, "lo": ci.lo, "hi": ci.hi,
                         **{a: post[a].mean() - pre[a].mean() for a in post}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ 4. face covariates
def facecov(item) -> dict:
    """One session -> OFFD/facecov/<label>.npz: per set, (n, 2 windows, traces) of FR raw (12), points (20), jaw (2)."""
    label, tt = item
    f = OFFD / "facecov" / f"{label}.npz"
    if f.exists():
        return {"label": label, "cached": True}
    from wfield_local import face_rhythm_outputs as fro
    an, date = label[:4], "2026" + label[-4:]
    sig = fpa.session_signals(an, date, want3d=False)
    out = {}
    for st, t in tt.items():
        n = len(t)
        T = (t[:, None] + GRID[None, :]).ravel()
        fr = fro.to_frames(*sig["fr"], T, 0.5).reshape(n, len(GRID), -1) if "fr" in sig else np.full((n, len(GRID), 12), np.nan)
        pt = fro.to_frames(*sig["pts"], T, fpa.MAX_GAP_PTS).reshape(n, len(GRID), -1) if "pts" in sig else \
            np.full((n, len(GRID), 20), np.nan)
        jw = np.stack([fpa._nearest(sig["jaw"][0], sig["jaw"][1][:, k], T, 0.003).reshape(n, -1) for k in range(2)], -1)
        for k_, A in (("fr", fr), ("pts", pt), ("jaw", jw)):
            G = A.reshape(n, 2, 20, A.shape[-1])
            c = np.isfinite(G).sum(2)
            with np.errstate(invalid="ignore"):
                out[f"{st}_{k_}"] = np.where(c >= 10, np.nansum(G, 2) / np.maximum(c, 1), np.nan).astype(np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return {"label": label, "cached": False}


def covariates(C: dict, jobs: int) -> dict:
    from wfield_local import analysis_kit as ak
    labs = sorted({x for S in C.values() for x in S})
    items = [(x, {st: C[(st, x[:4])][x][5] for st in lo.SETS if (st, x[:4]) in C and x in C[(st, x[:4])]}) for x in labs]
    res, fails = ak.fan_sessions(items, facecov, jobs=jobs, key=lambda it: it[0], label="facecov")
    if fails:
        raise SystemExit(f"facecov failures: {fails[:3]}")
    Z = {}
    for (st, _a), S in C.items():
        for x in S:
            z = np.load(OFFD / "facecov" / f"{x}.npz")
            fr = fpa._fr11(z[f"{st}_fr"].astype(float))                       # (n, 2, 11)
            fr5 = fr[..., :5]
            Z[(st, x)] = {"FR": fr.reshape(len(fr), -1), "FR5": fr5.reshape(len(fr), -1),
                          "FRPTS": np.concatenate([fr.reshape(len(fr), -1), z[f"{st}_pts"].reshape(len(fr), -1)], 1),
                          "FRJAW": np.concatenate([fr.reshape(len(fr), -1), z[f"{st}_jaw"].reshape(len(fr), -1)], 1)}
    return Z


# ------------------------------------------------------------------------------------------------ 5. partialling
def _ols(X, Z):
    zm = Z.mean(0)
    D = np.column_stack([np.ones(len(Z)), Z - zm])
    return {"B": np.linalg.lstsq(D, X, rcond=None)[0][1:], "zm": zm}


def partial_A(C, Xb, Z, cov) -> tuple[dict, dict]:
    """Variant A: per (set, animal) regression of features on face, PRE licks only (LOSO for pre sessions), frozen;
    x_res = x - (z - zm_train) B; licks without face covariates are left unchanged (counted)."""
    from wfield_local import frozen_models as fm
    Xr, stats = {}, {}
    for (st, a), S in C.items():
        pre = [x for x, v in S.items() if v[0] == "pre"]
        Zs = {x: Z[(st, x)][cov] for x in S}
        okz = {x: np.isfinite(Zs[x]).all(1) for x in S}

        def fit(pre=pre, Zs=Zs, okz=okz, S=S, st=st, a=a):
            def one(labs):
                return _ols(np.vstack([Xb[(st, a)][q][okz[q]] for q in labs]), np.vstack([Zs[q][okz[q]] for q in labs]))
            return {"full": one(pre), "loso": {h: one([q for q in pre if q != h]) for h in pre}}
        nf = Xb[(st, a)][pre[0]].shape[1] + Zs[pre[0]].shape[1]
        spec = fm.make_spec(a, "face_aim", align=f"offset_{st}", source=f"partial_{cov}", train_labels=pre, n_features=nf,
                            train_data=({q: Xb[(st, a)][q][okz[q]] for q in pre}, {q: Zs[q][okz[q]] for q in pre}))  # training-data fingerprint (FREEZE_VERSION 2, rebuild 11-03)
        Pm, status = fm.load_or_fit(spec, fit, meta={"script": "face_offset_partial.py", "what": f"partial {cov} {st}"}, log=log)
        out, nmiss = {}, 0
        for x in S:
            m = Pm["loso"][x] if x in pre else Pm["full"]
            X = Xb[(st, a)][x].copy()
            o = okz[x]
            X[o] = X[o] - (Zs[x][o] - m["zm"]) @ m["B"]
            nmiss += int((~o).sum())
            out[x] = X
        Xr[(st, a)] = out
        stats[(st, a)] = {"status": status, "n_noface": nmiss, "n": int(sum(len(v) for v in Zs.values()))}
    return Xr, stats


def partial_C(C, Xb, Z, cov) -> tuple[dict, int]:
    """Variant C: per-SESSION regression B_s (its own licks), applied as x - (z - zm_pre_animal) B_s; zm_pre LOSO for
    pre sessions; sessions with < 3 x n_covariates face licks use the animal's pre-fit (A) regression (counted)."""
    Xr, nfall = {}, 0
    for (st, a), S in C.items():
        pre = [x for x, v in S.items() if v[0] == "pre"]
        Zs = {x: Z[(st, x)][cov] for x in S}
        okz = {x: np.isfinite(Zs[x]).all(1) for x in S}
        out = {}
        for x in S:
            others = [q for q in pre if q != x]
            zm = np.vstack([Zs[q][okz[q]] for q in others]).mean(0)
            o = okz[x]
            X = Xb[(st, a)][x].copy()
            if o.sum() >= 3 * Zs[x].shape[1]:
                B = _ols(X[o], Zs[x][o])["B"]
            else:
                nfall += 1
                B = _ols(np.vstack([Xb[(st, a)][q][okz[q]] for q in others]), np.vstack([Zs[q][okz[q]] for q in others]))["B"]
            X[o] = X[o] - (Zs[x][o] - zm) @ B
            out[x] = X
        Xr[(st, a)] = out
    return Xr, nfall


def refit_frozen(C, Xr, cov) -> dict:
    """Variant B: decoder refit on the residualised pre features, frozen (kind face_aim)."""
    from wfield_local import frozen_models as fm
    fits = {}
    for (st, a), S in C.items():
        pre = [x for x, v in S.items() if v[0] == "pre"]
        spec = fm.make_spec(a, "face_aim", align=f"offset_{st}", source=f"refitdec_{cov}", train_labels=pre,
                            n_features=Xr[(st, a)][pre[0]].shape[1],
                            train_data=({q: Xr[(st, a)][q] for q in pre}, {q: S[q] for q in pre}))  # training-data fingerprint (FREEZE_VERSION 2, rebuild 11-03)
        P, _s = fm.load_or_fit(spec, lambda S=S, X=Xr[(st, a)]: fit_dec(S, X), log=log,
                               meta={"script": "face_offset_partial.py", "what": f"refit decoder {cov} {st}"})
        fits[(st, a)] = P
    return fits


def planted(C, Xb, Z, fits, target: dict) -> dict:
    """Plant a face-driven component along the decoder's weight direction in every lick: x += k c u, c = (z - zm_pre) g
    (g = the post - pre mean face change, unit), k per animal so the acute mean decoded shift = target[set]."""
    Xp = {}
    for (st, a), S in C.items():
        sc, m = fits[(st, a)][None]
        w = m.coef_ / sc.scale_
        u = w / np.linalg.norm(w)
        pre = [x for x, v in S.items() if v[0] == "pre"]
        Zs = {x: np.nan_to_num(Z[(st, x)][PRIMARY_COV] - 0.0) for x in S}
        zpre = np.vstack([Zs[q] for q in pre]).mean(0)
        post = [x for x in S if S[x][0] != "pre"]
        g = np.vstack([Zs[q] for q in post]).mean(0) - zpre if post else np.ones_like(zpre)
        g = g / max(np.linalg.norm(g), 1e-12)
        cs = {x: (Zs[x] - zpre) @ g for x in S}
        ac = [x for x in S if S[x][0] == "acute"] or post
        mc = np.concatenate([cs[x] for x in ac]).mean() * float(u @ w) if ac else 0.0
        k = target[st] / mc if abs(mc) > 1e-9 else 0.0
        Xp[(st, a)] = {x: Xb[(st, a)][x] + k * cs[x][:, None] * u[None, :] for x in S}
    return Xp


def compare(Sb: pd.DataFrame, Sa: pd.DataFrame, variant: str) -> tuple[list, list]:
    from wfield_local import analysis_kit as ak
    M = Sb.merge(Sa[["measure", "label", "value"]], on=["measure", "label"], suffixes=("_b", "_a"))
    per, pooled = [], []
    for m, g in M.groupby("measure"):
        for (a, ep), h in g.groupby(["animal", "epoch"]):
            per.append({"variant": variant, "measure": m, "animal": a, "epoch": ep, "n_sessions": len(h),
                        "n_licks": int(h.n.sum()), "before": h.value_b.mean(), "after": h.value_a.mean()})
        pre = {a: np.column_stack([h.value_b, h.value_a, h.value_a - h.value_b]) for a, h in sorted(g[g.epoch == "pre"].groupby("animal"))}
        for ep in EP:
            post = {a: np.column_stack([h.value_b, h.value_a, h.value_a - h.value_b])
                    for a, h in sorted(g[g.epoch == ep].groupby("animal")) if a in pre}
            if not post:
                continue
            obs, lo_, hi_, dr, n = ak.boot_delta_columns(post, pre, np.random.default_rng(0))
            am = {a: (post[a][:, 0].mean() - pre[a][:, 0].mean(), post[a][:, 1].mean() - pre[a][:, 1].mean()) for a in post}
            ratio = dr[:, 1] / np.where(dr[:, 0] == 0, np.nan, dr[:, 0])
            sgn = np.sign(obs[0])
            pooled.append({"variant": variant, "measure": m, "epoch": ep, "n_animals": n,
                           "d_before": obs[0], "d_before_lo": lo_[0], "d_before_hi": hi_[0],
                           "d_after": obs[1], "d_after_lo": lo_[1], "d_after_hi": hi_[1],
                           "d_diff": obs[2], "d_diff_lo": lo_[2], "d_diff_hi": hi_[2], "p_diff": ak.boot_p(dr[:, 2]),
                           "retained": obs[1] / obs[0] if obs[0] != 0 else np.nan,
                           "retained_lo": float(np.nanpercentile(ratio, 2.5)), "retained_hi": float(np.nanpercentile(ratio, 97.5)),
                           "n_after_same_sign": int(sum(np.sign(v[1]) == sgn for v in am.values())),
                           **{f"before_{a}": v[0] for a, v in am.items()}, **{f"after_{a}": v[1] for a, v in am.items()}})
    return per, pooled


def run(jobs: int) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import frozen_models as fm
    from wfield_local import writeguard
    writeguard.assert_writable(OFFD / "x.csv")
    (OFFD / "_itiw").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t0 = time.time()
    d = fm.local_dir()
    kinds = sorted({p.name.split("_")[0] for a in AN if (d / a).exists() for p in (d / a).iterdir()}) if d.exists() else []
    log(f"frozen local_dir = {d}  exists={d.exists()}  kinds present={kinds}")
    if not os.environ.get("WIDEFIELD_FROZEN_MODEL_DIR") and (not d.exists() or not kinds):
        raise SystemExit("frozen store absent or empty -- stop")
    JOBS[0] = min(jobs, 4)
    C = collect()
    log(f"collected {len(C)} set x animal cells, {sum(len(v) for v in C.values())} session-sets, {time.time() - t0:.0f} s")
    Xb = {k: {x: v[1] for x, v in S.items()} for k, S in C.items()}
    fits0 = {k: fit_dec(S, Xb[k]) for k, S in C.items()}
    Pb = score(C, Xb, fits0)
    cache = pd.read_csv(lo.preds_path())
    same = len(cache) == len(Pb) and (cache[["set", "label", "pos"]].astype(str).to_numpy() == Pb[["set", "label", "pos"]].astype(str).to_numpy()).all()
    dd = float(np.nanmax(np.abs(cache.dec.to_numpy(float) - Pb.dec.to_numpy(float)))) if same else np.nan
    log(f"REPRO 1: recomputed 'before' predictions vs lick_order_preds cache: rows equal {same}, max |d dec| {dd:.2e}")
    # published-path reproduction (original cache) and recomputed-before
    S_cache, T_cache = offsets(cache, "cache")
    S_b, T_b = offsets(Pb, "before")
    Tt = tests(S_b)
    Tc = tests(S_cache)
    log("== tests from the ORIGINAL cache\n" + Tc.round(3).to_string(index=False))
    log("== tests from the RECOMPUTED before\n" + Tt.round(3).to_string(index=False))
    terms = pd.read_csv(il.OUT / "offset_contrib" / "terms_licks_areas_splitr8.csv.gz",
                        usecols=["set", "label", "act", "order", "pos", "readout"])
    terms["set"] = terms.set.replace({"cued": "trial"})
    key = ["set", "label", "order", "pos", "act"]
    a1, b1 = terms.copy(), T_b.copy()
    for df in (a1, b1):
        df["act"] = df.act.round(3)
        df["dup"] = df.groupby(key).cumcount()
    mm = a1.merge(b1, on=[*key, "dup"], suffixes=("_pub", "_me"))
    log(f"REPRO 2: per-lick read-out vs terms_licks: {len(mm)} of {len(terms)} matched, max |diff| "
        f"{np.nanmax(np.abs(mm.readout_pub - mm.readout_me)):.2e}")
    rep = []
    for (m, ep), v in PUBLISHED.items():
        r = Tt[(Tt.measure == m) & (Tt.epoch == ep)]
        got = float(r.point.iloc[0]) if len(r) else np.nan
        rep.append({"measure": m, "epoch": ep, "published": v, "recomputed": got, "ok": bool(abs(got - v) <= TOL)})
    R = pd.DataFrame(rep)
    R.to_csv(OFFD / "reproduction.csv", index=False)
    log("== REPRODUCTION CHECK (tolerance 0.01 mm)\n" + R.round(4).to_string(index=False))
    if not R.ok.all():
        log("REPRODUCTION FAILED -- stopping before any after-partialling number (coordinator rule)")
        return 2
    if "--repro-only" in sys.argv:
        return 0
    # ---- face covariates + variants
    Z = covariates(C, jobs)
    allS, per_all, pooled_all, notes = [S_b.assign(variant="before")], [], [], []
    for cov in COVSETS:
        Xr, st_ = partial_A(C, Xb, Z, cov)
        notes += [{"variant": f"A_{cov}", "set": k[0], "animal": k[1], **v} for k, v in st_.items()]
        Sa, _ = offsets(score(C, Xr, fits0), f"A_{cov}")
        allS.append(Sa.assign(variant=f"A_{cov}"))
        pe, po = compare(S_b, Sa, f"A_{cov}")
        per_all += pe
        pooled_all += po
        log(f"variant A_{cov} done {time.time() - t0:.0f} s")
        if cov == PRIMARY_COV:
            fitsB = refit_frozen(C, Xr, cov)
            Sbv, _ = offsets(score(C, Xr, fitsB), f"B_{cov}")
            allS.append(Sbv.assign(variant=f"B_{cov}"))
            pe, po = compare(S_b, Sbv, f"B_{cov}")
            per_all += pe
            pooled_all += po
            Xc, nfall = partial_C(C, Xb, Z, cov)
            notes.append({"variant": f"C_{cov}", "n_fallback_sessions": nfall})
            Sc, _ = offsets(score(C, Xc, fits0), f"C_{cov}")
            allS.append(Sc.assign(variant=f"C_{cov}"))
            pe, po = compare(S_b, Sc, f"C_{cov}")
            per_all += pe
            pooled_all += po
            # planted-signal control
            target = {"iti": PUBLISHED[("iti_dock_sameplace", "acute")], "trial": 0.164}
            Xp = planted(C, Xb, Z, fits0, target)
            Sp, _ = offsets(score(C, Xp, fits0), "planted_raw")
            Xpr, _s = partial_A_planted(C, Xp, Z, cov)
            Spr, _ = offsets(score(C, Xpr, fits0), "planted_A")
            Ta, Tpl, Tpa = tests(Sa), tests(Sp), tests(Spr)
            pl = []
            for r in Tt.itertuples():
                def g(T_, r=r):
                    q = T_[(T_.measure == r.measure) & (T_.epoch == r.epoch)].point
                    return float(q.iloc[0]) if len(q) else np.nan
                eff = g(Tpl) - r.point
                rem = g(Tpa) - g(Ta)
                pl.append({"measure": r.measure, "epoch": r.epoch, "planted_effect": eff, "remaining_after_A": rem,
                           "removed_frac": 1 - rem / eff if eff != 0 else np.nan})
            PL = pd.DataFrame(pl)
            PL.to_csv(OFFD / "planted.csv", index=False)
            log("== PLANTED-SIGNAL control (must remove >= 90 %)\n" + PL.round(3).to_string(index=False))
    PA, PO = pd.DataFrame(per_all), pd.DataFrame(pooled_all)
    PO["family_D"] = (PO.variant == f"A_{PRIMARY_COV}") & PO.measure.isin(["iti_dock_sameplace", "cued_matched_error"])
    PO["p_holm"] = np.nan
    PO.loc[PO.family_D, "p_holm"] = ak.holm(PO.loc[PO.family_D, "p_diff"].to_numpy(float))
    PO["survives"] = ((PO.d_after_lo > 0) | (PO.d_after_hi < 0)) & (np.sign(PO.d_after) == np.sign(PO.d_before)) & \
        (PO.n_after_same_sign >= 3) & (PO.retained_lo >= 0.5)
    PO["face_carried"] = (PO.d_after_lo <= 0) & (PO.d_after_hi >= 0) & (PO.retained_hi < 0.5)
    PA.to_csv(OFFD / "per_animal.csv", index=False)
    PO.to_csv(OFFD / "pooled.csv", index=False)
    pd.concat(allS, ignore_index=True).to_csv(OFFD / "sessions.csv", index=False)
    pd.DataFrame(notes).to_csv(OFFD / "variant_notes.csv", index=False)
    log("== PER ANIMAL (change vs pre per animal is in pooled before_/after_ columns)\n" +
        PA[PA.variant == f"A_{PRIMARY_COV}"].round(3).to_string(index=False))
    log("== POOLED (family D = A_FR x {iti_dock_sameplace, cued_matched_error})\n" + PO.round(3).to_string(index=False))
    log("== notes\n" + pd.DataFrame(notes).to_string(index=False))
    plot()
    log(f"offset done {(time.time() - t0) / 60:.1f} min")
    return 0


def partial_A_planted(C, Xp, Z, cov):
    """Variant A on the planted features, NOT frozen (a control on synthetic data)."""
    Xr = {}
    for (st, a), S in C.items():
        pre = [x for x, v in S.items() if v[0] == "pre"]
        Zs = {x: Z[(st, x)][cov] for x in S}
        okz = {x: np.isfinite(Zs[x]).all(1) for x in S}
        out = {}
        for x in S:
            trn = [q for q in pre if q != x]
            m = _ols(np.vstack([Xp[(st, a)][q][okz[q]] for q in trn]), np.vstack([Zs[q][okz[q]] for q in trn]))
            X = Xp[(st, a)][x].copy()
            o = okz[x]
            X[o] = X[o] - (Zs[x][o] - m["zm"]) @ m["B"]
            out[x] = X
        Xr[(st, a)] = out
    return Xr, None


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from within_lick_correction import reserve_headroom, star_top

    from wfield_local import orofacial_cohort as oc
    EC = oc.EPOCH_LINE
    PO = pd.read_csv(OFFD / "pooled.csv")
    PA = pd.read_csv(OFFD / "per_animal.csv")
    variants = ["A_FR", "A_FR5", "A_FRPTS", "A_FRJAW", "B_FR", "C_FR"]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.6), gridspec_kw={"wspace": 0.35})
    for j, m in enumerate(MEASURES):
        ax = axs[j]
        for e, ep in enumerate(EP):
            base = e * (len(variants) + 2)
            r0 = PO[(PO.measure == m) & (PO.epoch == ep) & (PO.variant == "A_FR")]
            if len(r0):
                r0 = r0.iloc[0]
                ax.errorbar(base, r0.d_before, yerr=[[r0.d_before - r0.d_before_lo], [r0.d_before_hi - r0.d_before]],
                            fmt="o", mfc="none", mec=EC[ep], color=EC[ep], ms=7, capsize=2)
            for i, v in enumerate(variants):
                r = PO[(PO.measure == m) & (PO.epoch == ep) & (PO.variant == v)]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = base + 1 + i
                ax.errorbar(x, r.d_after, yerr=[[r.d_after - r.d_after_lo], [r.d_after_hi - r.d_after]], fmt="o",
                            color=EC[ep], ms=6 if v == "A_FR" else 4, capsize=2)
                if v == "A_FR" and np.isfinite(r.p_holm) and r.p_holm < 0.05:
                    star_top(ax, x, "*", color=EC[ep])
            # per-animal after (primary) as open markers
            for a in fpa.AN:
                rr = PO[(PO.measure == m) & (PO.epoch == ep) & (PO.variant == "A_FR")]
                if len(rr) and f"after_{a}" in rr and np.isfinite(rr[f"after_{a}"].iloc[0]):
                    ax.scatter(base + 1.45, rr[f"after_{a}"].iloc[0], marker=fpa.AM[a], facecolor="none",
                               edgecolor=EC[ep], s=18, lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ticks = [e * (len(variants) + 2) + k for e in range(3) for k in range(len(variants) + 1)]
        short = {"A_FR": "A", "A_FR5": "A5", "A_FRPTS": "A+pts", "A_FRJAW": "A+jaw", "B_FR": "B", "C_FR": "C"}
        ax.set_xticks(ticks, (["bef."] + [short[v] for v in variants]) * 3, rotation=90, fontsize=7)
        for e, ep in enumerate(EP):
            ax.text(e * (len(variants) + 2) + 3, -0.24, ep, transform=ax.get_xaxis_transform(), ha="center", fontsize=8,
                    color=EC[ep])
        ax.set_title({"iti_dock_sameplace": "ITI docked, same-place offset",
                      "cued_matched_error": "cued, matched-error offset",
                      "cued_sameplace": "cued, same-place offset (secondary)",
                      "iti_dock_matched_error": "ITI docked, matched-error (secondary)"}[m] +
                     "\nchange vs pre (mm), 95 % CI; open = before", fontsize=9, loc="left")
        ax.set_ylabel("decoder offset change (mm, + = contralesional)" if j == 0 else "")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("Frozen-decoder offset before vs after partialling the face out of the cortical features (NEURAL; "
                 "\nA = pre-fit regression [primary, FR 11 traces], A5 = 5 FR groups, +pts / +jaw = plus points / jaw, "
                 "B = decoder refit, C = per-session regression; "
                 "n = 4, PROVISIONAL)", fontsize=10, y=1.06)
    reserve_headroom(fig)
    (fpa.FIGS).mkdir(parents=True, exist_ok=True)
    fig.savefig(fpa.FIGS / "fpa_offset.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    _ = PA
