"""NOISE-MATCHED TRANSFER + ROTATION POWER for the lick-direction code (follow-up to lick_code_subspace.py, 10-29).

WHY. lick_code_subspace found post->pre transfer ~ post->post and an ITI coding-axis cosine (pre vs post) that is low
but equally low for the June no-stroke reference -> 'no re-map'. Priya: too strong. Post data carry LESS direction
information, so a weaker post decoder transferring 'equally' is what a NOISIER but unrotated code would also give, and
the axis test may lack power. Two questions:
 (1) NOISE-MATCHED NULL. What do transfer / axis measures look like for a PRE code degraded to the post epoch's
     decodability (noisier, NOT rotated)? A rotation predicts real cross-transfer BELOW this null.
 (2) POWER. Inject known rotations of the pre direction code at the real session structure: which rotation does each
     measure detect with 80 % power, and how large are the observed post / June changes in rotation units?

DESIGN (session = unit; every post session is compared with every pre session at the SAME structure).
  Lick sets as lick_code_subspace (cued_first, iti_all, iti_far1); matched post licks (inside the pre side_ml range per
  spout); the frozen lick decoders' feature basis (45 z-scored area features), target y = side_ml (+ = contralesional).
  Per animal x set x post epoch (june = no-stroke reference, acute, subacute, chronic):
    m = min(median matched licks per post session, median licks per pre session); every unit is ONE session
    subsampled to min(its count, m) licks (real post sessions: R_SUB subsamples averaged; pre units: N_REP).
  REAL unit = a post session s:   within = 5 contiguous time-fold CV r inside s
                                  fwd    = the FROZEN full pre decoder (rule 10, frozen_models kind 'lick_code',
                                           lick_code_subspace.frozen_pre -> frozen-hit) on s      (= pre->post)
                                  back   = a decoder trained on s, tested on every pre session    (= post->pre)
                                  cos_enc = cos(encoding axis of s, encoding axis of all pre)
                                  sub    = class-mean subspace overlap (k = 2) of s vs all pre
  NULL unit = a pre session p (pseudo-post; same subsample size m):
                                  fwd uses the FROZEN leave-p-out sibling of the same frozen payload; back tests on the
                                  other pre sessions; axes vs the pre-minus-p reference. So pre->D vs pre->post differ
                                  only by n_pre-1 vs n_pre training sessions.
    DEGRADATION ('matched'): X_p + sigma * E_perm, E = p's own residual after regressing its features on
      [y, spout one-hot] (session-centred), rows shuffled -> noise with the session's real covariance and no
      direction information. sigma chosen per animal x set x epoch by bisection so the mean null within-session r
      equals the mean real post within-session r (sigma = 0 when post is not less decodable than pre at equal m).
    ROTATION (power): the pre direction code = C_ref (coefficients of the session-centred features on [y, spout
      one-hot], from the pre-minus-p reference; rank 6), row space B. It is rotated by theta toward a random
      orthonormal U orthogonal to B (a new U per rep), U drawn in proportion to the session's residual covariance
      (E' g): Rot = I + (cos-1)(BB'+UU') + sin(UB'-BU'); every principal angle between injected and original code =
      theta. Two variants:
        rigid (PRIMARY): the whole session-centred cloud rotated in span(B, U) -- code AND its noise move together,
              within-session decodability preserved exactly, total variance preserved.
        sig: only the code component Z_p C_ref rotated (X_p + Z_p C_ref (Rot' - I)), then the session's shuffled
              residual added until within-session r is back at the unrotated level (a rotated code in less noisy
              directions otherwise becomes MORE decodable; with isotropic U this needed sigma ~15, hence the
              covariance-weighted U).
      theta in THETAS, with and without the noise-matched degradation.
  Decoders trained on degraded / rotated SYNTHETIC pre data and on post sessions are NOT pre-stroke references:
  refit in memory, NOT frozen (rule 10 covers the reference only; the reference here is the frozen lick_code payload).
TESTS.
  Per animal: real epoch mean vs the null pseudo-epoch distribution (n_post pre units drawn with replacement, each the
    mean of R_SUB random reps -- same averaging as a real session); p_below = share of null pseudo-epochs <= real.
  Pooled: analysis_kit.boot_delta(real post sessions, null pre sessions; animals paired), + a pooled p_below on the
    animal-mean statistic.
  POWER per animal and pooled: share of rotated pseudo-epochs (n_post units at theta) below the 5th percentile of the
    theta = 0 pseudo-epoch distribution (one-sided 5 %, the same test as p_below). theta80 = smallest theta with power
    >= 0.8 (linear interpolation on the grid). EQUIVALENT ROTATION of the real data = theta at which the mean
    noise-matched rotated unit equals the real epoch mean. POST-BEYOND-JUNE power: rotated pseudo-epochs at
    theta_June + delta vs theta_June (grid), post structure.
Outputs <OUT_W>/lick_code_noisematch/*.csv + figdata/noisematch.pkl; figures <OUT_W>/figures/lick_code_noisematch.png,
lick_code_power.png. Reads lick_code_subspace/licks_cache.pkl (run lick_code_subspace.py first); writes nothing there.
ENLv LICKS (Priya 10-29): LICK_CODE_FAMILY=enlv (run lick_code_subspace.py with it first) -> sets enlv_on1 /
enlv_bout, epochs acute / subacute / chronic, every output file suffixed _enlv (lick_code_subspace.FAMILY).
Usage: python scripts/lick_code_noisematch.py [--jobs 4]      (compute + figures)
       python scripts/lick_code_noisematch.py --plot-only
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_code_subspace as lcs  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN, EP, POS, SETS, W, SFX = lcs.AN, lcs.EP, lcs.POS, lcs.SETS, lcs.W, lcs.SFX   # LICK_CODE_FAMILY=enlv -> ENLv
THETAS = [0, 10, 20, 30, 45, 60, 90]
N_REP = 6           # pre units per session (subsample + noise permutation + rotation direction)
R_SUB = 4           # subsamples per real post session (and reps averaged per null pseudo-session)
N_FOLD = 5
MIN_SESS = 30
N_PSEUDO = 4000
ALPHA = 0.05
MET = ["within_r", "fwd_r", "fwd_slope", "fwd_icpt", "back_r", "back_slope", "back_icpt", "asym", "cos_enc", "sub"]
LOWER = ["fwd_r", "back_r", "cos_enc", "sub", "within_r"]       # a rotation lowers these
POWER_MET = ["cos_enc", "sub", "fwd_r", "back_r"]
ROTS = ["rigid", "sig"]   # rigid = whole session cloud rotated in span(B, U) (PRIMARY; SNR kept exactly); sig = code
PRIMARY = "rigid"         # component only, residual noise added back until decodability is re-matched
OUT = il.OUT_W / "lick_code_noisematch"
FIGD = OUT / "figdata"
FIG = il.OUT_W / "figures"


# ------------------------------------------------------------------------------------------------ pieces
def _design(y, pos, sess=None):
    """Session-centred [y, spout one-hot] design."""
    Z = np.column_stack([y.astype(float), (pos[:, None] == np.array(POS)[None, :]).astype(float)])
    sess = np.zeros(len(y)) if sess is None else sess
    for s in np.unique(sess):
        m = sess == s
        Z[m] -= Z[m].mean(0)
    return Z


def _within(X, y, t):
    o = np.argsort(t)
    X, y = X[o], y[o]
    dec = np.full(len(y), np.nan)
    for te in np.array_split(np.arange(len(y)), N_FOLD):
        tr = np.setdiff1d(np.arange(len(y)), te)
        dec[te] = lcs._pred(lcs._fit(X[tr], y[tr]), X[te])
    return lcs._score(dec, y)["r"]


def _rotation(B, U, theta):
    th = np.deg2rad(theta)
    F = B.shape[0]
    return (np.eye(F) + (np.cos(th) - 1) * (B @ B.T + U @ U.T) + np.sin(th) * (U @ B.T - B @ U.T))


class Ref:
    """A pre reference (all pre sessions, or pre minus one): encoding axis, class subspace, code coefficients."""

    def __init__(self, Xc, yc, pos, sess, Z):
        self.b = lcs.enc_axis(Xc, yc)
        self.U = lcs.class_sub(Xc, pos)
        self.C = np.linalg.lstsq(Z, Xc, rcond=None)[0]
        _u, s, vt = np.linalg.svd(self.C, full_matrices=False)
        r = int((s > s[0] * 1e-6).sum())
        self.B = vt[:r].T


def _evaluate(X, y, t, pos, model, ref: Ref, back_sets, within_only=False) -> dict:
    out = {"within_r": _within(X, y, t)}
    if within_only:
        return out
    f = lcs._score(lcs._pred(model, X), y)
    M = lcs._fit(X, y)
    bk = [lcs._score(lcs._pred(M, Xq), yq) for Xq, yq in back_sets]
    Xc, yc = X - X.mean(0), y - y.mean()
    out.update({"fwd_r": f["r"], "fwd_slope": f["slope"], "fwd_icpt": f["intercept"],
                "back_r": float(np.nanmean([b["r"] for b in bk])),
                "back_slope": float(np.nanmean([b["slope"] for b in bk])),
                "back_icpt": float(np.nanmean([b["intercept"] for b in bk])),
                "cos_enc": lcs.cos(lcs.enc_axis(Xc, yc), ref.b), "sub": lcs.overlap(lcs.class_sub(Xc, pos), ref.U)})
    out["asym"] = out["fwd_r"] - out["back_r"]
    return out


# ------------------------------------------------------------------------------------------------ worker
def _cell(item):
    """(set, animal) -> (units DataFrame, sigma DataFrame). Module-level for analysis_kit.fan_sessions."""
    st, a = item
    from wfield_local import parallel
    parallel.pin_blas() if hasattr(parallel, "pin_blas") else None
    L, X, _names = lcs.load(1)
    L = lcs.add_match(L)
    g = L[(L.set == st) & (L.animal == a)]
    Xg = X[g.index.to_numpy()]
    g = g.reset_index(drop=True)
    sess, y, t, pos = g.label.to_numpy(), g.y.to_numpy(float), g.t.to_numpy(float), g.pos.to_numpy()
    matched = g.matched.to_numpy()
    mpre = (g.epoch == "pre").to_numpy()
    P = lcs.frozen_pre(a, st, Xg[mpre], y[mpre], sess[mpre])          # RULE 10: frozen-hit expected
    pre = sorted(g[mpre].label.unique())
    rng = np.random.default_rng(1000 + 10 * list(SETS).index(st) + AN.index(a))
    Xc, yc = lcs._centre(Xg, y, sess)
    Zall = _design(y, pos, sess)
    refs = {}
    for key in ["__all__", *pre]:
        m = mpre & (sess != key)
        refs[key] = Ref(Xc[m], yc[m], pos[m], sess[m], Zall[m])
    pre_sets = {q: (Xg[sess == q], y[sess == q]) for q in pre}
    F = Xg.shape[1]
    rows, sig = [], []
    for ep in EP:
        cnt = g[(g.epoch == ep) & g.matched].groupby("label").size()
        post = sorted(cnt[cnt >= MIN_SESS].index)
        if not post or len(pre) < 3:
            continue
        pre_cnt = g[mpre].groupby("label").size()
        m_ep = int(min(np.median(cnt[post]), np.median(pre_cnt)))
        # ---- real post sessions
        for s in post:
            idx_all = np.flatnonzero((sess == s) & matched)
            for r in range(R_SUB):
                ii = np.sort(rng.choice(idx_all, min(len(idx_all), m_ep), replace=False))
                ev = _evaluate(Xg[ii], y[ii], t[ii], pos[ii], P["full"], refs["__all__"], list(pre_sets.values()))
                rows.append({"epoch": ep, "kind": "real", "theta": 0, "noise": "-", "label": s, "rep": r,
                             "n": len(ii), **ev})
        # ---- pre units (fixed random numbers across sigma / theta = common random numbers)
        units = []
        for p in pre:
            idx_all = np.flatnonzero(sess == p)
            for r in range(N_REP):
                ii = np.sort(rng.choice(idx_all, min(len(idx_all), m_ep), replace=False))
                Xs, ys, ps = Xg[ii], y[ii], pos[ii]
                Z = _design(ys, ps)
                Xsc = Xs - Xs.mean(0)
                E = Xsc - Z @ np.linalg.lstsq(Z, Xsc, rcond=None)[0]
                E = E[rng.permutation(len(ii))]
                ref = refs[p]
                # target directions drawn in proportion to the session's own residual covariance (patterns the
                # cortex actually expresses), not isotropic: an isotropic direction in 45 correlated z-scored
                # features is nearly noise-free and makes a rotated code absurdly decodable
                Ur = E.T @ rng.standard_normal((len(ii), ref.B.shape[1]))
                Ur -= ref.B @ (ref.B.T @ Ur)
                Ur = np.linalg.qr(Ur)[0]
                units.append({"p": p, "rep": r, "ii": ii, "Z": Z, "E": E, "U": Ur})

        def mean_within(sigma, units=units):
            return float(np.nanmean([_within(Xg[u["ii"]] + sigma * u["E"], y[u["ii"]], t[u["ii"]]) for u in units]))
        target = float(np.nanmean([r_["within_r"] for r_ in rows if r_["epoch"] == ep and r_["kind"] == "real"]))
        w0 = mean_within(0.0)
        sigma = 0.0
        if np.isfinite(target) and w0 > target:
            lo_, hi_ = 0.0, 1.0
            while mean_within(hi_) > target and hi_ < 64:
                lo_, hi_ = hi_, hi_ * 2
            for _ in range(8):
                mid = 0.5 * (lo_ + hi_)
                if mean_within(mid) > target:
                    lo_ = mid
                else:
                    hi_ = mid
            sigma = 0.5 * (lo_ + hi_)
        w_matched = mean_within(sigma) if sigma > 0 else w0
        sig.append({"epoch": ep, "n_post": len(post), "n_pre": len(pre), "m": m_ep, "target_within": target,
                    "pre_within_m": w0, "sigma": sigma, "matched_within": w_matched})
        print(f"   {st} {a} {ep}: m {m_ep} within post {target:.3f} pre {w0:.3f} -> sigma {sigma:.3f} "
              f"({w_matched:.3f})", flush=True)
        def make(u, th, sg, rot):
            ii, ref = u["ii"], refs[u["p"]]
            if not th:
                return Xg[ii] + sg * u["E"]
            R = _rotation(ref.B, u["U"], th)
            if rot == "sig":    # rotate ONLY the pre code component Z C_ref; noise geometry untouched
                return Xg[ii] + (u["Z"] @ ref.C) @ (R.T - np.eye(F)) + sg * u["E"]
            Xm = Xg[ii] + sg * u["E"]     # RIGID: the whole session-centred cloud rotated in span(B, U)
            mu = Xm.mean(0)
            return mu + (Xm - mu) @ R.T

        for noise, sg in (("none", 0.0), ("matched", sigma)):
            w_cond = w0 if noise == "none" else w_matched
            for rot in ("sig", "rigid"):
                for th in THETAS:
                    sg_th = sg
                    if rot == "sig" and th:
                        # a rotated code can land in lower-noise directions and become MORE decodable: add the
                        # session's own shuffled residual noise until within-session r is back at this condition's
                        # unrotated level, so only the DIRECTION changes
                        def mw(s_, th=th, units=units):
                            return float(np.nanmean([_within(make(u, th, s_, "sig"), y[u["ii"]], t[u["ii"]])
                                                     for u in units]))
                        if mw(sg) > w_cond:
                            lo_, hi_ = sg, max(2 * sg, 0.5)
                            while mw(hi_) > w_cond and hi_ < 64:
                                lo_, hi_ = hi_, hi_ * 2
                            for _ in range(7):
                                mid = 0.5 * (lo_ + hi_)
                                lo_, hi_ = (mid, hi_) if mw(mid) > w_cond else (lo_, mid)
                            sg_th = 0.5 * (lo_ + hi_)
                    for u in units:
                        ii, ref = u["ii"], refs[u["p"]]
                        back = [v for q, v in pre_sets.items() if q != u["p"]]
                        ev = _evaluate(make(u, th, sg_th, rot), y[ii], t[ii], pos[ii], P["loso"][u["p"]], ref, back)
                        rows.append({"epoch": ep, "kind": "sim", "rot": rot, "theta": th, "noise": noise,
                                     "label": u["p"], "rep": u["rep"], "n": len(ii), "sigma": sg_th, **ev})
    U_ = pd.DataFrame(rows)
    S_ = pd.DataFrame(sig)
    for D in (U_, S_):
        D.insert(0, "animal", a)
        D.insert(0, "set", st)
    print(f"   {st} {a} done", flush=True)
    return U_, S_


# ------------------------------------------------------------------------------------------------ statistics
def _pools(U, st, a, ep, noise, theta, met, rot="sig"):
    """null / rotated pool: array (n_pre sessions, N_REP) of unit values."""
    h = U[(U.set == st) & (U.animal == a) & (U.epoch == ep) & (U.kind == "sim") & (U.noise == noise)
          & (U.theta == theta) & (U.rot == rot)]
    if h.empty:
        return None
    return h.pivot_table(index="label", columns="rep", values=met).to_numpy(float)


def _real(U, st, a, ep, met):
    h = U[(U.set == st) & (U.animal == a) & (U.epoch == ep) & (U.kind == "real")]
    return h.groupby("label")[met].mean().to_numpy(float) if len(h) else None


def _pseudo(pool, n, rng, n_draw=N_PSEUDO):
    """n_draw pseudo-epoch means: n sessions with replacement, each = mean of R_SUB random reps (as a real session)."""
    ns, nr = pool.shape
    si = rng.integers(0, ns, (n_draw, n))
    k = min(R_SUB, nr)
    ri = np.argsort(rng.random((n_draw, n, nr)), axis=2)[:, :, :k]
    v = pool[si[:, :, None], ri]
    return np.nanmean(np.nanmean(v, 2), 1)


def tables(U: pd.DataFrame, S: pd.DataFrame):
    rng = np.random.default_rng(0)
    PA, PP, PW, PWP, EQ, BJ = [], [], [], [], [], []
    for st in SETS:
        for ep in EP:
            nulls_animal = {}
            for a in AN:
                srow = S[(S.set == st) & (S.animal == a) & (S.epoch == ep)]
                if srow.empty:
                    continue
                n_post = int(srow.n_post.iloc[0])
                r = {"set": st, "animal": a, "epoch": ep, **srow.iloc[0][["n_post", "n_pre", "m", "sigma"]].to_dict()}
                for met in MET:
                    real = _real(U, st, a, ep, met)
                    p0 = _pools(U, st, a, ep, "none", 0, met)
                    pm = _pools(U, st, a, ep, "matched", 0, met)
                    r[f"{met}_real"] = np.nanmean(real)
                    r[f"{met}_null0"] = np.nanmean(p0)
                    r[f"{met}_nullm"] = np.nanmean(pm)
                    ps = _pseudo(pm, n_post, rng)
                    r[f"{met}_p_below"] = float(np.mean(ps <= np.nanmean(real)))
                    nulls_animal.setdefault(met, {})[a] = (ps, np.nanmean(real))
                for mt in ("fwd", "back"):
                    for kk in ("real", "nullm"):
                        r[f"{mt}_ratio_{kk}"] = r[f"{mt}_r_{kk}"] / r[f"within_r_{kk}"]
                PA.append(r)
                # ---- power + equivalent rotation (per animal)
                for rot in ROTS:
                    for met in POWER_MET:
                        for noise in ("none", "matched"):
                            p0 = _pools(U, st, a, ep, noise, 0, met, rot)
                            thr = np.nanpercentile(_pseudo(p0, n_post, rng), 100 * ALPHA)
                            for th in THETAS:
                                pl = _pools(U, st, a, ep, noise, th, met, rot)
                                PW.append({"set": st, "epoch": ep, "animal": a, "rot": rot, "measure": met,
                                           "noise": noise, "theta": th,
                                           "power": float(np.mean(_pseudo(pl, n_post, rng) < thr)),
                                           "mean": float(np.nanmean(pl))})
            # ---- pooled real vs noise-matched null
            for met in MET:
                if met not in nulls_animal:
                    continue
                d = nulls_animal[met]
                pooled_null = np.mean([v[0] for v in d.values()], 0)
                pooled_real = np.mean([v[1] for v in d.values()])
                for noise in ("matched", "none"):
                    post = {a: _real(U, st, a, ep, met) for a in d}
                    pre = {}
                    for a in d:
                        pl = _pools(U, st, a, ep, noise, 0, met)
                        pre[a] = np.nanmean(pl, 1)
                    post = {a: v[np.isfinite(v)] for a, v in post.items() if np.isfinite(v).any()}
                    pre = {a: v[np.isfinite(v)] for a, v in pre.items() if a in post}
                    ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                    dd = {a: post[a].mean() - pre[a].mean() for a in post}
                    pt = ci.point if ci is not None else np.mean(list(dd.values()))
                    PP.append({"set": st, "epoch": ep, "measure": met, "null": noise,
                               "real": np.mean([post[a].mean() for a in post]),
                               "null_level": np.mean([pre[a].mean() for a in post]), "delta": pt,
                               "lo": ci.lo if ci is not None else np.nan, "hi": ci.hi if ci is not None else np.nan,
                               "n_same_sign": int(sum(np.sign(v) == np.sign(pt) for v in dd.values())), "n": len(dd),
                               "p_below_pooled": float(np.mean(pooled_null <= pooled_real)) if noise == "matched"
                               else np.nan, **{f"d_{a}": dd.get(a, np.nan) for a in AN}})
            # ---- pooled power (mean over animals of pseudo-epoch means)
            ani = [a for a in AN if not S[(S.set == st) & (S.animal == a) & (S.epoch == ep)].empty]
            if not ani:
                continue
            npost = {a: int(S[(S.set == st) & (S.animal == a) & (S.epoch == ep)].n_post.iloc[0]) for a in ani}
            for rot in ROTS:
                for met in POWER_MET:
                    for noise in ("none", "matched"):
                        null = np.mean([_pseudo(_pools(U, st, a, ep, noise, 0, met, rot), npost[a], rng)
                                        for a in ani], 0)
                        thr = np.nanpercentile(null, 100 * ALPHA)
                        for th in THETAS:
                            alt = np.mean([_pseudo(_pools(U, st, a, ep, noise, th, met, rot), npost[a], rng)
                                           for a in ani], 0)
                            PWP.append({"set": st, "epoch": ep, "rot": rot, "measure": met, "noise": noise,
                                        "theta": th, "power": float(np.mean(alt < thr)), "n_animals": len(ani)})
                # ---- equivalent rotation (noise-matched curve)
                for met in POWER_MET:
                    for a in ani + ["pooled"]:
                        aa = ani if a == "pooled" else [a]
                        curve = np.array([np.mean([np.nanmean(_pools(U, st, x, ep, "matched", th, met, rot))
                                                   for x in aa]) for th in THETAS])
                        real = np.mean([np.nanmean(_real(U, st, x, ep, met)) for x in aa])
                        EQ.append({"set": st, "epoch": ep, "rot": rot, "measure": met, "animal": a, "real": real,
                                   "curve0": curve[0], "curve90": curve[-1], "theta_eq": _theta_eq(curve, real)})
    PA, PP, PW, PWP, EQ = map(pd.DataFrame, (PA, PP, PW, PWP, EQ))
    # ---- theta80
    T80 = []
    for keys, h in PW.groupby(["set", "epoch", "animal", "rot", "measure", "noise"]):
        T80.append({**dict(zip(["set", "epoch", "animal", "rot", "measure", "noise"], keys)), "theta80": _theta80(h)})
    for keys, h in PWP.groupby(["set", "epoch", "rot", "measure", "noise"]):
        T80.append({**dict(zip(["set", "epoch", "rot", "measure", "noise"], keys)), "animal": "pooled",
                    "theta80": _theta80(h)})
    T80 = pd.DataFrame(T80)
    # ---- post beyond June: delta detectable on top of the June-equivalent angle (pooled, noise-matched, post structure)
    for st in SETS:
        for rot in ROTS:
            for met in POWER_MET:
                q = EQ[(EQ.set == st) & (EQ.rot == rot) & (EQ.measure == met) & (EQ.animal == "pooled")
                       & (EQ.epoch == "june")]
                if q.empty or not np.isfinite(q.theta_eq.iloc[0]):
                    continue
                thj = q.theta_eq.iloc[0]
                th1 = max([th for th in THETAS if th <= thj], default=0)
                for ep in [e for e in EP if e != "june"]:
                    ani = [a for a in AN if not S[(S.set == st) & (S.animal == a) & (S.epoch == ep)].empty]
                    if not ani:
                        continue
                    npost = {a: int(S[(S.set == st) & (S.animal == a) & (S.epoch == ep)].n_post.iloc[0])
                             for a in ani}
                    null = np.mean([_pseudo(_pools(U, st, a, ep, "matched", th1, met, rot), npost[a], rng)
                                    for a in ani], 0)
                    thr = np.nanpercentile(null, 100 * ALPHA)
                    pw = {}
                    for th2 in [th for th in THETAS if th > th1]:
                        alt = np.mean([_pseudo(_pools(U, st, a, ep, "matched", th2, met, rot), npost[a], rng)
                                       for a in ani], 0)
                        pw[th2] = float(np.mean(alt < thr))
                    d80 = next((th2 - th1 for th2, v in pw.items() if v >= 0.8), np.nan)
                    eqp = EQ[(EQ.set == st) & (EQ.rot == rot) & (EQ.measure == met) & (EQ.animal == "pooled")
                             & (EQ.epoch == ep)]
                    BJ.append({"set": st, "rot": rot, "measure": met, "epoch": ep, "theta_june_eq": thj,
                               "grid_base": th1, "theta_post_eq": eqp.theta_eq.iloc[0] if len(eqp) else np.nan,
                               "delta80_beyond_june": d80, **{f"power_{th2}": v for th2, v in pw.items()}})
    return PA, PP, PW, PWP, EQ, T80, pd.DataFrame(BJ)


def _theta_eq(curve, real):
    """theta where the (monotone-enforced, decreasing) mean rotated curve equals the real value."""
    c = np.minimum.accumulate(np.asarray(curve, float))
    if not np.isfinite(real) or not np.isfinite(c).all():
        return np.nan
    if real >= c[0]:
        return 0.0
    if real <= c[-1]:
        return float(THETAS[-1])
    return float(np.interp(-real, -c, THETAS))


def _theta80(h):
    h = h.sort_values("theta")
    th, pw = h.theta.to_numpy(float), h.power.to_numpy(float)
    for i in range(1, len(th)):
        if pw[i] >= 0.8:
            if i == 1:          # already reached at the smallest grid angle: report it (= '<= 10 deg'), no
                return th[1]    # interpolation toward theta = 0, where power is alpha by construction
            return float(th[i - 1] + (0.8 - pw[i - 1]) / (pw[i] - pw[i - 1]) * (th[i] - th[i - 1]))
    return np.nan


# ------------------------------------------------------------------------------------------------ run
def run(jobs):
    t0 = time.time()
    items = [(st, a) for st in SETS for a in AN]
    res, fails = ak.fan_sessions(items, _cell, jobs=jobs, key=lambda it: (list(SETS).index(it[0]), it[1]),
                                 label="lick_code_noisematch")
    if fails:
        raise RuntimeError(f"failed cells: {[f[0] for f in fails]} {fails[0][1]}")
    U = pd.concat([v[0] for _k, v in res], ignore_index=True)
    S = pd.concat([v[1] for _k, v in res], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGD.mkdir(parents=True, exist_ok=True)
    U.to_csv(OUT / f"units{SFX}.csv", index=False)
    S.to_csv(OUT / f"sigma_animals{SFX}.csv", index=False)
    print(f"cells done in {time.time() - t0:.0f} s", flush=True)
    finish(U, S)
    print(f"total runtime {time.time() - t0:.0f} s", flush=True)


def finish(U, S):
    PA, PP, PW, PWP, EQ, T80, BJ = tables(U, S)
    VJ = vs_june(U)
    for name, D in (("noisematch_animals", PA), ("noisematch_pooled", PP), ("power_animals", PW),
                    ("power_pooled", PWP), ("equiv_rotation", EQ), ("theta80", T80), ("power_beyond_june", BJ),
                    ("deficit_vs_june", VJ)):
        D.to_csv(OUT / f"{name}{SFX}.csv", index=False)
    with open(FIGD / f"noisematch{SFX}.pkl", "wb") as f:
        pickle.dump({"S": S, "PA": PA, "PP": PP, "PW": PW, "PWP": PWP, "EQ": EQ, "T80": T80, "BJ": BJ, "VJ": VJ}, f)
    report(S, PA, PP, T80, EQ, BJ)
    print("\n== DEFICIT (real session minus its epoch's noise-matched null mean): post epoch vs June (boot_delta)")
    print(VJ.round(3).to_string(index=False))
    figures()


def vs_june(U: pd.DataFrame) -> pd.DataFrame:
    """Is the post deficit below the noise-matched null larger than June's (the no-stroke ~2-month reference)?
    Per session: real value minus the animal's noise-matched theta = 0 null mean FOR THAT EPOCH (each epoch has its own
    lick count and sigma); boot_delta(post-epoch deficits, June deficits), animals paired."""
    out = []
    for st in SETS:
        for met in ("fwd_r", "back_r", "fwd_slope", "back_slope", "asym", "cos_enc", "sub"):
            dfc = {}
            for ep in EP:
                for a in AN:
                    real = _real(U, st, a, ep, met)
                    pool = _pools(U, st, a, ep, "matched", 0, met, PRIMARY)
                    if real is None or pool is None:
                        continue
                    v = real - np.nanmean(pool)
                    if np.isfinite(v).any():
                        dfc.setdefault(ep, {})[a] = v[np.isfinite(v)]
            if "june" not in dfc:
                continue
            for ep in [e for e in EP if e != "june" and e in dfc]:
                post = {a: v for a, v in dfc[ep].items() if a in dfc["june"]}
                ref = {a: dfc["june"][a] for a in post}
                ci = ak.boot_delta(post, ref, np.random.default_rng(0))
                d = {a: post[a].mean() - ref[a].mean() for a in post}
                pt = ci.point if ci is not None else np.mean(list(d.values()))
                out.append({"set": st, "measure": met, "epoch": ep,
                            "june_deficit": np.mean([ref[a].mean() for a in post]),
                            "post_deficit": np.mean([post[a].mean() for a in post]), "delta_vs_june": pt,
                            "lo": ci.lo if ci is not None else np.nan, "hi": ci.hi if ci is not None else np.nan,
                            "n_same_sign": int(sum(np.sign(v) == np.sign(pt) for v in d.values())), "n": len(d),
                            **{f"d_{a}": d.get(a, np.nan) for a in AN}})
    return pd.DataFrame(out)


def report(S, PA, PP, T80, EQ, BJ):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    print("\n== PER ANIMAL noise matching (within-session r at m licks)")
    print(S.round(3).to_string(index=False))
    cols = ["set", "animal", "epoch", "n_post", "sigma"]
    for met in ("within_r", "fwd_r", "back_r", "asym", "fwd_slope", "back_slope", "fwd_icpt", "cos_enc", "sub"):
        print(f"\n== PER ANIMAL {met}: real vs null (none / noise-matched); p_below = vs matched null")
        print(PA[cols + [f"{met}_real", f"{met}_null0", f"{met}_nullm", f"{met}_p_below"]].round(3).to_string(index=False))
    print("\n== POOLED real minus null (boot_delta, animals paired)")
    print(PP.round(3).to_string(index=False))
    print("\n== theta80 (deg; NaN = not reached by 90)")
    print(T80.pivot_table(index=["set", "rot", "measure", "noise", "epoch"], columns="animal", values="theta80")
          .round(0).to_string())
    print("\n== equivalent rotation (noise-matched curve, deg)")
    print(EQ.pivot_table(index=["set", "rot", "measure", "animal"], columns="epoch", values="theta_eq")
          .round(0).to_string())
    print("\n== post beyond June (pooled, noise-matched, post structure)")
    print(BJ.round(2).to_string(index=False))


# ------------------------------------------------------------------------------------------------ figures
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.epoch_figures import EPOCH_GREY as _EG
    EG = {**_EG, "june": "#6baed6"}
    with open(FIGD / f"noisematch{SFX}.pkl", "rb") as f:
        F = pickle.load(f)
    PA, PWP, T80, EQ = F["PA"], F["PWP"], F["T80"], F["EQ"]
    FIG.mkdir(parents=True, exist_ok=True)
    x0 = {ep: i for i, ep in enumerate(EP)}
    mets = [("within_r", "within-session r (matched by construction)"), ("fwd_r", "pre -> X r (frozen pre decoder)"),
            ("back_r", "X -> pre r (X-trained)"), ("asym", "asymmetry: (pre->X) minus (X->pre) r"),
            ("cos_enc", "encoding-axis cosine vs pre"), ("sub", "class-mean subspace overlap vs pre")]
    fig, axs = plt.subplots(len(SETS), len(mets), figsize=(22, 11), squeeze=False)
    for i, st in enumerate(SETS):
        for j, (met, lab) in enumerate(mets):
            ax = axs[i, j]
            g = PA[PA.set == st]
            for ep in EP:
                h = g[g.epoch == ep]
                if h.empty:
                    continue
                x = x0[ep]
                ax.bar(x - 0.2, h[f"{met}_null0"].mean(), 0.18, color="#d9d9d9", edgecolor="k", lw=0.4)
                ax.bar(x, h[f"{met}_nullm"].mean(), 0.18, color="#a6d8a6", edgecolor="k", lw=0.4)
                ax.bar(x + 0.2, h[f"{met}_real"].mean(), 0.18, color=EG[ep], edgecolor="k", lw=0.4)
                for _, r in h.iterrows():
                    ax.plot([x, x + 0.2], [r[f"{met}_nullm"], r[f"{met}_real"]], color="0.4", lw=0.6)
                    ax.plot(x + 0.2, r[f"{met}_real"], lcs.MARK[r.animal], color="w", mec="k", ms=4.5)
                    ax.plot(x, r[f"{met}_nullm"], lcs.MARK[r.animal], color="w", mec="#3a7d3a", ms=3.5)
            ax.set_xticks(range(len(EP)), EP, fontsize=8)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(W(f"{SETS[st]}: {lab}", 40), fontsize=8.5)
    hand = [Line2D([], [], marker="s", ls="", color="#d9d9d9", mec="k", label="pre unit, not degraded"),
            Line2D([], [], marker="s", ls="", color="#a6d8a6", mec="k", label="pre unit, NOISE-MATCHED (null)"),
            Line2D([], [], marker="s", ls="", color="0.5", mec="k", label="real post / June session")] + [
        Line2D([], [], marker=lcs.MARK[a], ls="", color="w", mec="k", label=a) for a in AN]
    fig.legend(handles=hand, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(W("Fig 1. Noise-matched null for the lick-direction code. Each unit = one session subsampled to the "
                   "same lick count; pre sessions degraded with their own shuffled residual noise until within-session "
                   "decodability equals the post epoch's (green). A rotated post code predicts real transfer / "
                   "axis cosine BELOW green; a noisier but unrotated code predicts real ~ green. Bars = mean over "
                   "animals, lines join each animal's null and real.", 170), fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    save_png_svg(fig, FIG / f"lick_code_noisematch{SFX}.png", dpi=120)
    plt.close(fig)

    # ---- power figure
    pm = [("cos_enc", "encoding-axis cosine"), ("sub", "class-mean subspace overlap"), ("fwd_r", "pre -> X transfer r"),
          ("back_r", "X -> pre transfer r")]
    fig, axs = plt.subplots(len(SETS), len(pm) + 1, figsize=(22, 11), squeeze=False)
    for i, st in enumerate(SETS):
        for j, (met, lab) in enumerate(pm):
            ax = axs[i, j]
            for ep in EP:
                for noise, ls in (("matched", "-"), ("none", ":")):
                    h = PWP[(PWP.set == st) & (PWP.epoch == ep) & (PWP.measure == met) & (PWP.noise == noise)
                            & (PWP.rot == PRIMARY)]
                    if len(h):
                        ax.plot(h.theta, h.power, ls, marker="o" if noise == "matched" else None, ms=3,
                                color=EG[ep], lw=1.6 if noise == "matched" else 1)
            ax.axhline(0.8, color="#b0413e", lw=0.8, ls="--")
            ax.axhline(ALPHA, color="0.6", lw=0.6)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(THETAS)
            ax.set_xlabel("injected rotation (deg)", fontsize=8)
            t8 = T80[(T80.set == st) & (T80.measure == met) & (T80.animal == "pooled") &
                     (T80.noise == "matched") & (T80.rot == PRIMARY)]
            txt = ", ".join(f"{r.epoch} {r.theta80:.0f}" if np.isfinite(r.theta80) else f"{r.epoch} >90"
                            for r in t8.itertuples())
            ax.set_title(W(f"{SETS[st]}: {lab}. Pooled power; theta80 (noise-matched): {txt}", 48), fontsize=8.5)
        ax = axs[i, -1]
        for k, (met, off, mc) in enumerate((("fwd_r", -0.2, "#2c7fb8"), ("back_r", 0.2, "#e08214"))):
            e = EQ[(EQ.set == st) & (EQ.measure == met) & (EQ.rot == PRIMARY)]
            for ep in EP:
                h = e[e.epoch == ep]
                for _, r in h.iterrows():
                    if r.animal == "pooled":
                        ax.plot(x0[ep] + off, r.theta_eq, "_", color=mc, ms=18, mew=2.5,
                                label=f"{met} pooled" if ep == EP[0] else None)
                    else:
                        ax.plot(x0[ep] + off + 0.05 * (AN.index(r.animal) - 1.5), r.theta_eq, lcs.MARK[r.animal],
                                color="w", mec=mc, ms=5)
            _ = k
        ax.set_xticks(range(len(EP)), EP, fontsize=8)
        ax.set_ylim(-2, 60)
        ax.set_ylabel("equivalent rotation (deg)", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_title(W(f"{SETS[st]}: real transfer deficit vs the noise-matched null, in injected-rotation units "
                       "(blue pre->X, orange X->pre; markers = animals). theta80 for these measures ~10 deg pooled",
                       48), fontsize=8.5)
    hand = [Line2D([], [], color=EG[ep], lw=1.6, label=ep) for ep in EP] + [
        Line2D([], [], color="0.3", ls="-", marker="o", ms=3, label="with noise-matched degradation"),
        Line2D([], [], color="0.3", ls=":", label="no degradation"),
        Line2D([], [], color="#b0413e", ls="--", label="80 % power")]
    fig.legend(handles=hand, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(W("Fig 2. Power to detect a rotated lick-direction code. The pre code (session-centred features on "
                   "[side_ml, spout], rank 6) is rotated rigidly (code + its noise, decodability preserved) by theta "
                   "toward random orthogonal, covariance-weighted directions inside held-out pre sessions at each "
                   "epoch's real session count and lick count; power = share of pooled pseudo-epochs below the 5th "
                   "percentile of the unrotated ones (solid = with the noise-matched degradation). Right: the real "
                   "June / post transfer expressed as the injected rotation giving the same mean r.", 170),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    save_png_svg(fig, FIG / f"lick_code_power{SFX}.png", dpi=120)
    plt.close(fig)
    print("figures written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--tables-only", action="store_true", help="re-derive tables + figures from units.csv")
    a = ap.parse_args()
    if il.ITI_SUBSET or il.AREA_BASIS != "split":
        raise SystemExit("run without WFIELD_ITI_FILTER / with the split basis")
    if a.tables_only:
        finish(pd.read_csv(OUT / f"units{SFX}.csv"), pd.read_csv(OUT / f"sigma_animals{SFX}.csv"))
        return
    if a.plot_only:
        figures()
        return
    run(min(a.jobs, 4))


if __name__ == "__main__":
    main()
