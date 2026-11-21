"""CSM diagnostic, part 2 (2026-11-21, after `pose3d_csm_examples`): the size of the RESIDUAL four-camera view-switch
jumps and candidate fixes, evaluated on the same licks. READ ONLY on production; inputs = `pose3d_csm_examples refuse`
pickles (csm_examples/fused/<sid>.pkl: production's fusion re-run in memory, every >= 2-camera subset of the USED views
re-triangulated with production's shifted cameras, i.e. AFTER the per-session side-tongue offsets) + the gate-4 per-lick
pickles; outputs only in csm_examples/fixes/.

    python -m scripts.pose3d_csm_examples fixes

JUMP SIZE: frames where production used all four views; offset of each subset's point vs the all-view point
(d_S = X_S - X_all, mm). Per session: bias = mean vector, scatter = RMS about it; pose dependence by reach quintile
(|X_all|, per session), midline angle tercile and spout. Scale: a CSM peak needs ~40 mm/s x 4 ms = 0.16 mm per frame.

FIXES (four-camera; the two-camera run is shown raw and through the same smoother, at the SAME threshold):
  raw  production track (the gate-4 measure).
  a1   per-session constant subset-bias correction: X - b_S (b_S = mean d_S on that session's all-view frames).
  a2   per-session REACH-BINNED correction: X - b_S(reach quintile) (fallback: a1 when a bin has < 20 frames).
       Neither adds samples; one-view fills are left as they are.
  b    one view set per lick: the modal multi-view set of the lick; frames whose used views do not contain it -> NaN
       (re-triangulated from that set otherwise); then the same run rule (one-frame gaps only).
  c    CSM peaks within the near-change window (`pose3d_csm_examples`) masked (a bound, from the peaks table).
  d0   Kalman / RTS smoother on the raw track; d = the same on the a2-corrected track. Constant-velocity model per axis
       (discrete white-noise acceleration, q), measurement variance PER VIEW-SET CLASS (each used subset, each one-view
       fill camera; two-camera: src 1 / 2 / 3), initial velocity variance -- all three FIT by EM on every full_ok post-cue
       lick (no hand tuning, nothing chosen toward a CSM rate). Runs only on the lick's tracked run (one-frame gaps are
       prediction steps, no frame outside the run is produced: no extrapolation). Posterior SD kept. The CSM measure is
       then applied unchanged to the smoothed positions.
  e1   one-view fills (`fill_n`: one camera's ray at an interpolated depth) treated as missing, same run rule (2cam:
       src 2 / 3 likewise); e1_a2_smoother = a2 + the smoother on that track.
  e2   a speed peak on the run's LAST frame not counted (it exists only through `_peaks`' zero padding).
NOISE CONTROL: the gate-4 surrogate (SG(9,3) track + iid noise at the session SD) and a SWITCH-AWARE surrogate: the same
plus, at every frame, the measured subset bias of that frame's REAL view set at its reach (10-bin interpolation, finer than
the a2 correction's 5 bins, so a2 is not graded against its own table). False = surrogate gains a CSM over the clean
SG(9,3) track passed through the SAME pipeline. Threshold per pipeline recalibrated by the gate-4 rule (min theta on the
30..200 grid with false <= 0.05) under the switch-aware surrogate.
VALIDATION of the smoother: TRUTH licks = runs where production used all four views on every frame. (1) view dropping:
each truth lick takes the view-set sequence of a random real lick of the same spout and session (fills -> all views) and
its points become X_S of those subsets: RMS error vs truth and false CSMs, raw vs a2 vs d0 vs d. (2) injection: a
minimum-jerk lateral submovement (28 ms, peak speed 40 / 60 / 100 / 150 mm/s) added to the truth track mid-protrusion
(or mid-retraction): recovery = CSM count rises by >= 1, raw vs smoothed, with and without the view dropping.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pose3d_csm_examples as EX  # noqa: E402

DT = 0.004
AX = ("ml", "ap", "dv")
ALL = 15
GRID = list(range(30, 201, 10))
ONSET_TOL = 8.5
MIN_CSM_FR = 10
INJ_V = (40.0, 60.0, 100.0, 150.0)
INJ_D = 0.028
POS_ORDER = EX.POS_ORDER
NAME = {1: "c1", 2: "c4", 4: "c2", 8: "c3"}


def sname(s: int) -> str:
    if s >= 100:
        return NAME[s - 100] + " fill"
    return "+".join(NAME[b] for b in (1, 2, 4, 8) if s & b)


def csm_of(P: np.ndarray, th) -> int | list:
    k = int(np.argmax(np.linalg.norm(P, axis=1)))
    pk = EX.peaks_idx(EX.speed(P), k)
    if np.ndim(th):
        return [len(EX.csm_peaks(pk, t)) for t in th]
    return len(EX.csm_peaks(pk, th))


def csm_noend(P: np.ndarray, th) -> list:
    """CSM count with a speed peak on the run's LAST frame not counted (the zero-padded boundary of `_peaks`)."""
    k = int(np.argmax(np.linalg.norm(P, axis=1)))
    pk = [p for p in EX.peaks_idx(EX.speed(P), k) if p[0] != len(P) - 1]
    return [len(EX.csm_peaks(pk, t)) for t in th]


def track(W, t, P, on, off):
    """`_track` + the gate-4 full_ok rule: (t, P, idx (-1 = interpolated), full_ok) or None."""
    tr = W._track(t, P, np.arange(len(t)) + 1000)
    if tr is None:
        return None
    tt, PP, ss = tr
    full = (tt[0] - on <= ONSET_TOL) and (off - tt[-1] <= ONSET_TOL) and len(PP) >= MIN_CSM_FR
    return tt, PP, np.where(ss == 9, -1, ss - 1000), bool(full)


# ------------------------------------------------------------------------------------------------ bias tables
def bias_tables(Fd: pd.DataFrame):
    """Per subset: constant bias, reach-quintile biases (5 bins) and 10-bin biases from all-view frames of a session."""
    a = Fd[(Fd.used == ALL) & (Fd.nv == 4)]
    Xa = a[[f"X_{x}" for x in AX]].to_numpy(float)
    r = np.linalg.norm(Xa, axis=1)
    e5 = np.quantile(r, np.linspace(0, 1, 6)) if len(r) > 50 else None
    e10 = np.quantile(r, np.linspace(0, 1, 11)) if len(r) > 100 else None
    out = {}
    for s in EX.SUBSETS:
        if s == ALL:
            continue
        D = a[[f"S{s}_{x}" for x in AX]].to_numpy(float) - Xa
        ok = np.isfinite(D).all(1)
        if ok.sum() < 20:
            continue
        b = D[ok].mean(0)
        ent = {"b": b, "n": int(ok.sum()), "scatter": float(np.sqrt(((D[ok] - b) ** 2).sum(1).mean()))}
        for nm, e in (("b5", e5), ("b10", e10)):
            if e is None:
                continue
            nb = len(e) - 1
            idx = np.clip(np.searchsorted(e, r, side="right") - 1, 0, nb - 1)
            B = np.tile(b, (nb, 1))
            for i in range(nb):
                k = ok & (idx == i)
                if k.sum() >= 20:
                    B[i] = D[k].mean(0)
            ent[nm] = B
            ent[f"e{nm[1:]}"] = e
        out[s] = ent
    return out


def bias_at(tab: dict, s: np.ndarray, X: np.ndarray, kind: str) -> np.ndarray:
    """(n, 3) bias of each frame's used set s at its reach; 0 for all-view / fills / unknown sets."""
    out = np.zeros((len(s), 3))
    r = np.linalg.norm(X, axis=1)
    for v in np.unique(s):
        if v not in tab:
            continue
        k = s == v
        ent = tab[v]
        if kind == "a1" or f"b{kind[1:]}" not in ent:
            out[k] = ent["b"]
            continue
        B, e = ent[f"b{kind[1:]}"], ent[f"e{kind[1:]}"]
        if kind == "b10":                      # interpolated between bin centres
            c = (e[:-1] + e[1:]) / 2
            out[k] = np.column_stack([np.interp(r[k], c, B[:, j]) for j in range(3)])
        else:
            i = np.clip(np.searchsorted(e, r[k], side="right") - 1, 0, len(B) - 1)
            out[k] = B[i]
    return out


# ------------------------------------------------------------------------------------------------ per-session worker
def worker(sid: str) -> dict:
    EX._env()
    import within_lick_correction as W
    from scipy.signal import savgol_filter
    from wfield_local.paths import PathResolver
    D = EX.dirs()
    rv = PathResolver()
    an, date = sid.split("_")
    Fd = pd.read_pickle(D["out"] / "fused" / f"{sid}.pkl")
    tab = bias_tables(Fd)
    # jump table rows (all-view frames)
    a = Fd[(Fd.used == ALL) & (Fd.nv == 4)]
    Xa = a[[f"X_{x}" for x in AX]].to_numpy(float)
    L4 = pd.read_pickle(D["4cam_licks"] / f"{sid}.pkl").set_index("lick_uid")
    pos_of = L4.pos.to_dict()
    jump = {"Xa_reach": np.linalg.norm(Xa, axis=1), "angle": np.degrees(np.arcsin(np.clip(Xa[:, 0] /
                                                                                         np.linalg.norm(Xa, axis=1), -1, 1))),
            "pos": a.lick_uid.map(pos_of).to_numpy(), "D": {}}
    for s in EX.SUBSETS:
        if s != ALL:
            jump["D"][s] = (a[[f"S{s}_{x}" for x in AX]].to_numpy(float) - Xa).astype(np.float32)
    # view-set frequency on multi-view lick frames
    mv = Fd[Fd.nv >= 2]
    freq = mv.used.value_counts().to_dict()
    nfill = int((Fd.nv == -1).sum())
    A = pd.read_csv(Path(rv.root("orofacial")) / EX.V36 / "sessions" / an / date / "licks_all.csv",
                    usecols=["trial_id", "on_ms", "off_ms"])
    A["lick_uid"] = np.arange(len(A))
    A = A.set_index("lick_uid")
    sd = L4[["noise_ml", "noise_ap", "noise_dv"]].iloc[0].to_numpy(float)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    lick_rows, sm_items, truth, donors, e1_items = [], [], [], {}, []
    for uid, g in Fd.groupby("lick_uid", sort=True):
        g = g.sort_values("t_ms")
        on, off = A.loc[uid, "on_ms"], A.loc[uid, "off_ms"]
        t = g.t_ms.to_numpy(float)
        X = g[[f"X_{x}" for x in AX]].to_numpy(float)
        nv, used = g.nv.to_numpy(), g.used.to_numpy()
        code = np.where(nv >= 2, used, np.where(nv == -1, 100 + used, 0))
        row = {"sid": sid, "animal": an, "lick_uid": int(uid), "pos": pos_of.get(uid)}
        tr = track(W, t, X, on, off)
        if tr is None:
            row["full_ok"] = False
            lick_rows.append(row)
            continue
        tt, P, idx, full = tr
        row["full_ok"] = full
        has = np.isfinite(X).all(1)
        # fix b: modal multi-view set
        if (nv >= 2).any():
            M = int(pd.Series(used[nv >= 2]).mode().iloc[0])
            Xb = np.full_like(X, np.nan)
            okb = (nv >= 2) & ((used & M) == M)
            if M == ALL:
                Xb[okb] = X[okb]
            else:
                Xb[okb] = g[[f"S{M}_{x}" for x in AX]].to_numpy(float)[okb]
            row["b_set"] = M
            row["b_lost"] = float((has & ~np.isfinite(Xb).all(1)).sum() / max(has.sum(), 1))
            tb = track(W, t, Xb, on, off)
            row["b_full_ok"] = bool(tb is not None and tb[3])
            if row["b_full_ok"]:
                row["b_csm40"], row["b_csm50"] = csm_of(tb[1], [40, 50])
        if not full:
            lick_rows.append(row)
            continue
        cfr = np.where(idx >= 0, code[np.clip(idx, 0, None)], -1)       # -1 = interpolated frame
        row["raw_csm40"], row["raw_csm50"] = csm_of(P, [40, 50])
        row["e2_csm40"], row["e2_csm50"] = csm_noend(P, [40, 50])
        row["fill_share"] = float(np.mean((cfr >= 100)))
        row["fill_last"] = bool(cfr[-1] >= 100)
        # fix e1: one-view fills (depth interpolated by `fill_n`) treated as missing, then the same run rule
        Xe = X.copy()
        Xe[nv == -1] = np.nan
        te = track(W, t, Xe, on, off)
        row["e1_full_ok"] = bool(te is not None and te[3])
        if row["e1_full_ok"]:
            row["e1_csm40"], row["e1_csm50"] = csm_of(te[1], [40, 50])
            ce = np.where(te[2] >= 0, code[np.clip(te[2], 0, None)], -1)
            Pe = te[1] - bias_at(tab, np.where(ce > 0, ce, 0), te[1], "b5")
            Ye = Pe.copy()
            Ye[te[2] < 0] = np.nan
            e1_items.append({"uid": int(uid), "cls": ce, "Y": {"a2": Ye}})
        corr = {}
        for kind in ("a1", "a2"):
            Bk = bias_at(tab, np.where(cfr > 0, cfr, 0), P, kind if kind == "a1" else "b5")
            Pc = P - Bk
            corr[kind] = Pc
            row[f"{kind}_csm40"], row[f"{kind}_csm50"] = csm_of(Pc, [40, 50])
        # measured Y for the smoother: NaN on interpolated frames
        Yraw = P.copy()
        Yraw[idx < 0] = np.nan
        Ya2 = corr["a2"].copy()
        Ya2[idx < 0] = np.nan
        # surrogates (gate-4 iid and switch-aware), on the same lick
        T9 = savgol_filter(P, 9, 3, axis=0)
        noise = rng.normal(0, 1, T9.shape) * sd
        Bsw = bias_at(tab, np.where(cfr > 0, cfr, 0), T9, "b10")
        Tn = T9 + noise
        Tsw = T9 + Bsw + noise
        Tsw_a2 = Tsw - bias_at(tab, np.where(cfr > 0, cfr, 0), Tsw, "b5")
        row["sur_raw_clean"] = tuple(csm_of(T9, GRID))
        row["sur_raw_iid"] = tuple(csm_of(Tn, GRID))
        row["sur_raw_sw"] = tuple(csm_of(Tsw, GRID))
        row["sur_a2_sw"] = tuple(csm_of(Tsw_a2, GRID))
        msk = idx < 0
        sm_items.append({"uid": int(uid), "cls": cfr, "Y": {"raw": Yraw, "a2": Ya2,
                                                           "s_clean": np.where(msk[:, None], np.nan, T9),
                                                           "s_iid": np.where(msk[:, None], np.nan, Tn),
                                                           "s_sw": np.where(msk[:, None], np.nan, Tsw),
                                                           "s_sw_a2": np.where(msk[:, None], np.nan, Tsw_a2)}})
        donors.setdefault(row["pos"], []).append(cfr)
        # truth licks, nothing interpolated: STRICT = every run frame all four views (truth = production point);
        # C14 = every run frame used cam1 AND cam4 as multi-view (truth = the cam1 + cam4-only point, a constant set)
        strict = bool((cfr == ALL).all())
        c14 = bool(((cfr > 0) & (cfr < 100) & ((cfr & 3) == 3)).all())
        if strict or c14:
            Sx = {s: g[[f"S{s}_{x}" for x in AX]].to_numpy(float)[idx] for s in EX.SUBSETS if s != ALL}
            Pt = P if strict else Sx[3]
            if np.isfinite(Pt).all():
                truth.append({"uid": int(uid), "pos": row["pos"], "P": Pt, "S": Sx, "strict": strict,
                              "base": ALL if strict else 3, "Pall": P if strict else None})
        lick_rows.append(row)
    # two-camera tracks for the smoother (same lick windows, its own frames / src classes)
    S2 = EX.load_frames(D, "2cam", sid)
    L2 = pd.read_pickle(D["2cam_licks"] / f"{sid}.pkl")
    L2 = L2[L2.lick_uid.isin(Fd.lick_uid.unique()) & L2.full_ok]
    gg = {k: v for k, v in S2.groupby("trial_id")}
    two = []
    for r in L2.itertuples():
        v = gg.get(r.trial_id)
        if v is None:
            continue
        on, off = A.loc[r.lick_uid, "on_ms"], A.loc[r.lick_uid, "off_ms"]
        v = v[(v.t_ms >= on - 0.5) & (v.t_ms <= off + 0.5)]
        tr = track(W, v.t_ms.to_numpy(float), v[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float), on, off)
        if tr is None or not tr[3]:
            continue
        tt, P, idx, _ = tr
        src = v.tongue_src.to_numpy(int)
        cls = np.where(idx >= 0, src[np.clip(idx, 0, None)], -1)
        Y = P.copy()
        Y[idx < 0] = np.nan
        it2 = {"uid": int(r.lick_uid), "pos": r.pos, "cls": cls, "Y": {"raw": Y},
               "raw_csm40": csm_of(P, 40), "raw_csm50": csm_of(P, 50)}
        it2["e2_csm40"], it2["e2_csm50"] = csm_noend(P, [40, 50])
        Xe = v[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float).copy()
        Xe[np.isin(src, (2, 3))] = np.nan                                   # one-camera fills -> missing
        te = track(W, v.t_ms.to_numpy(float), Xe, on, off)
        it2["e1_full_ok"] = bool(te is not None and te[3])
        if it2["e1_full_ok"]:
            it2["e1_csm40"], it2["e1_csm50"] = csm_of(te[1], [40, 50])
        two.append(it2)
    # truth-lick validation inputs: donor sequences of the same spout
    for tl in truth:
        dn = donors.get(tl["pos"], [])
        tl["donor"] = dn[rng.integers(len(dn))] if dn else None
        tl["inj_sign"] = float(rng.choice([-1.0, 1.0]))
    btab = {s: {"b": e["b"], "n": e["n"], "scatter": e["scatter"], "b5": e.get("b5"), "e5": e.get("e5")}
            for s, e in tab.items()}
    return {"sid": sid, "licks": pd.DataFrame(lick_rows), "sm": sm_items, "e1": e1_items, "two": two, "truth": truth, "jump": jump,
            "freq": freq, "nfill": nfill, "tab": btab, "attrs": Fd.attrs}


# ------------------------------------------------------------------------------------------------ Kalman / RTS + EM
A_ = np.array([[1.0, DT], [0.0, 1.0]])
Q0 = np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
Q0i = np.linalg.inv(Q0)


def rts_group(Y: np.ndarray, R: np.ndarray, q: float, p0v: float):
    """Batch RTS over licks of equal length. Y (L, n, 3) NaN = missing; R (L, n) measurement variance (NaN missing).
    Returns smoothed means (L, n, 3, 2), covariances (L, n, 2, 2) and lag-one cross covariances (L, n-1, 2, 2)."""
    L, n, _ = Y.shape
    Q = q * Q0
    mf = np.zeros((L, n, 3, 2))
    Pf = np.zeros((L, n, 2, 2))
    mp = np.zeros_like(mf)
    Pp = np.zeros_like(Pf)
    m = np.zeros((L, 3, 2))
    m[:, :, 0] = np.nan_to_num(Y[:, 0])
    P = np.tile(np.diag([1e2, p0v]), (L, 1, 1))
    for t in range(n):
        if t > 0:
            m = m @ A_.T
            P = A_ @ P @ A_.T + Q
        mp[:, t], Pp[:, t] = m, P
        obs = np.isfinite(R[:, t]) & np.isfinite(Y[:, t]).all(1)
        if obs.any():
            S = P[obs, 0, 0] + R[obs, t]
            K = P[obs, :, 0] / S[:, None]                                   # (l, 2)
            innov = Y[obs, t] - m[obs, :, 0]                                # (l, 3)
            m[obs] = m[obs] + innov[:, :, None] * K[:, None, :]
            P[obs] = P[obs] - K[:, :, None] * P[obs, 0, :][:, None, :]
        mf[:, t], Pf[:, t] = m, P
    ms, Ps = mf.copy(), Pf.copy()
    Cs = np.zeros((L, max(n - 1, 0), 2, 2))
    for t in range(n - 2, -1, -1):
        J = Pf[:, t] @ A_.T @ np.linalg.inv(Pp[:, t + 1])
        ms[:, t] = mf[:, t] + np.einsum("lij,lkj->lki", J, ms[:, t + 1] - mp[:, t + 1])
        Ps[:, t] = Pf[:, t] + J @ (Ps[:, t + 1] - Pp[:, t + 1]) @ np.transpose(J, (0, 2, 1))
        Cs[:, t] = Ps[:, t + 1] @ np.transpose(J, (0, 2, 1))               # cov(x_{t+1}, x_t)
    return ms, Ps, Cs


def smooth_all(items: list, key: str, cls_index: dict, rvar: np.ndarray, q: float, p0v: float, em_stats=False):
    """Smooth every item's Y[key] (grouped by length). Returns list of (pos (n,3), sd (n,)) and optionally EM sums."""
    out = [None] * len(items)
    by_n = {}
    for i, it in enumerate(items):
        by_n.setdefault(len(it["cls"]), []).append(i)
    st = {"q": [0.0, 0], "r": np.zeros((len(rvar), 2)), "v0": [0.0, 0]}
    for n, ii in by_n.items():
        Y = np.stack([items[i]["Y"][key] for i in ii])
        C = np.stack([items[i]["cls"] for i in ii])
        ci = np.vectorize(lambda c: cls_index.get(int(c), -1))(C)
        R = np.where(ci >= 0, rvar[np.clip(ci, 0, None)], np.nan)
        ms, Ps, Cs = rts_group(Y, R, q, p0v)
        for j, i in enumerate(ii):
            out[i] = (ms[j, :, :, 0], np.sqrt(Ps[j, :, 0, 0]))
        if em_stats:
            # q: E[trace(Q0^-1 (x_t - A x_{t-1})(.)^T)] / 2 per axis
            for t in range(1, n):
                m1, m0 = ms[:, t], ms[:, t - 1]                              # (l, 3, 2)
                S11 = Ps[:, t][:, None] + np.einsum("lai,laj->laij", m1, m1)
                S00 = Ps[:, t - 1][:, None] + np.einsum("lai,laj->laij", m0, m0)
                S10 = Cs[:, t - 1][:, None] + np.einsum("lai,laj->laij", m1, m0)
                M = S11 - A_ @ np.swapaxes(S10, -1, -2) - S10 @ A_.T + A_ @ S00 @ A_.T
                st["q"][0] += float(np.einsum("ij,laji->", Q0i, M)) / 2
                st["q"][1] += M.shape[0] * 3
            ok = np.isfinite(Y).all(2) & (ci >= 0)
            res = (Y - ms[:, :, :, 0]) ** 2 + Ps[:, :, 0, 0][:, :, None]
            for c in range(len(rvar)):
                k = ok & (ci == c)
                st["r"][c, 0] += float(res[k].sum())
                st["r"][c, 1] += int(k.sum()) * 3
            st["v0"][0] += float((Ps[:, 0, 1, 1][:, None] + ms[:, 0, :, 1] ** 2).sum())
            st["v0"][1] += ms.shape[0] * 3
    return out, st


def em_fit(items: list, key: str, max_iter: int = 400, tol: float = 2e-4, sub: int = 5, n_full: int = 5) -> dict:
    """EM for (q, measurement variance per class, initial velocity variance). Stage 1 on every ``sub``-th lick until the
    largest relative parameter change per iteration is < ``tol`` (or ``max_iter``); stage 2 = ``n_full`` iterations on
    all licks from there. EM moves q slowly (25 iterations from q = 1e5 were still rising ~1 % per iteration)."""
    classes = sorted({int(c) for it in items for c in it["cls"] if c >= 0})
    ci = {c: i for i, c in enumerate(classes)}
    rvar = np.full(len(classes), 0.05)
    q, p0v = 1e5, 1e4
    hist = []
    stages = [(items[::sub], max_iter, "sub"), (items, n_full, "full")]
    for its, n_it, nm in stages:
        for it in range(n_it):
            _, st = smooth_all(its, key, ci, rvar, q, p0v, em_stats=True)
            q1 = st["q"][0] / st["q"][1]
            r1 = np.maximum(np.where(st["r"][:, 1] > 0, st["r"][:, 0] / np.maximum(st["r"][:, 1], 1), rvar), 1e-6)
            v1 = st["v0"][0] / st["v0"][1]
            ch = max(abs(q1 / q - 1), abs(v1 / p0v - 1), float(np.max(np.abs(r1 / rvar - 1))))
            q, rvar, p0v = q1, r1, v1
            hist.append({"stage": nm, "iter": it, "q": q, "p0v": p0v, "max_rel_change": ch,
                         **{f"r_{c}": float(rvar[ci[c]]) for c in classes}})
            if nm == "sub" and ch < tol:
                break
    return {"classes": classes, "ci": ci, "rvar": rvar, "q": q, "p0v": p0v, "hist": hist}


# ------------------------------------------------------------------------------------------------ driver
def share_tab(L: pd.DataFrame, cols: dict, mask_col="full_ok") -> pd.DataFrame:
    rows = []
    for pos in POS_ORDER:
        g = L[L.pos == pos]
        r = {"pos": pos}
        for nm, c in cols.items():
            v = g[c].dropna() if c in g else pd.Series(dtype=float)
            r[nm] = float((v >= 1).mean()) if len(v) else np.nan
        rows.append(r)
    T = pd.DataFrame(rows)
    far = T[T.pos.isin(["far_ipsi", "far_contra"])].mean(numeric_only=True)
    close = T[T.pos.isin(["close_ipsi", "close_contra"])].mean(numeric_only=True)
    T = pd.concat([T, pd.DataFrame([{"pos": "far_lateral", **far}, {"pos": "close_lateral", **close},
                                    {"pos": "far_minus_close", **(far - close)}])], ignore_index=True)
    return T


def calib(series_clean, series_sur) -> tuple[float | None, list]:
    c = np.array(list(series_clean))
    s = np.array(list(series_sur))
    fr = [float(np.mean(s[:, i] > c[:, i])) for i in range(len(GRID))]
    ok = [GRID[i] for i in range(len(GRID)) if fr[i] <= 0.05]
    return (float(min(ok)) if ok else None), fr


def run(sids: list | None = None, out: Path | None = None, jobs: int = 6) -> None:
    from wfield_local import analysis_kit as ak
    D = EX.dirs()
    out = Path(out) if out else D["out"] / "fixes"
    out.mkdir(parents=True, exist_ok=True)
    sids = sids or EX.sessions(D)
    res, fail = ak.fan_sessions(sids, worker, jobs=min(jobs, len(sids)))
    if fail:
        raise SystemExit(f"failed: {[(j, str(e)[:400]) for j, e in fail]}")
    R = [v for _, v in res]
    print("fusion re-run vs production table, max |diff| mm:", max(v["attrs"]["table_match_maxabs_mm"] for v in R))
    # ---------------- jump sizes
    rows, prow = [], []
    for v in R:
        J = v["jump"]
        for s, Dd in J["D"].items():
            ok = np.isfinite(Dd).all(1)
            if ok.sum() < 20:
                continue
            b = Dd[ok].mean(0)
            rows.append({"sid": v["sid"], "subset": sname(s), "s": s, "n": int(ok.sum()), "bias_ml": b[0],
                         "bias_ap": b[1], "bias_dv": b[2], "bias_mm": float(np.linalg.norm(b)),
                         "scatter_mm": float(np.sqrt(((Dd[ok] - b) ** 2).sum(1).mean())),
                         "freq_multiview": v["freq"].get(s, 0) / max(sum(v["freq"].values()), 1)})
            r = J["Xa_reach"][ok]
            qs = np.quantile(r, np.linspace(0, 1, 6))
            bi = np.clip(np.searchsorted(qs, r, side="right") - 1, 0, 4)
            ang = J["angle"][ok]
            ai = np.digitize(ang, np.quantile(ang, [1 / 3, 2 / 3]))
            for nm, lab in (("reach_q", bi), ("angle_t", ai)):
                for i in np.unique(lab):
                    k = lab == i
                    if k.sum() >= 20:
                        bb = Dd[ok][k].mean(0)
                        prow.append({"sid": v["sid"], "subset": sname(s), "by": nm, "bin": int(i), "n": int(k.sum()),
                                     "bias_ml": bb[0], "bias_ap": bb[1], "bias_dv": bb[2],
                                     "bias_mm": float(np.linalg.norm(bb)), "reach_mm": float(r[k].mean())})
            for pos in POS_ORDER:
                k = J["pos"][ok] == pos
                if k.sum() >= 20:
                    bb = Dd[ok][k].mean(0)
                    prow.append({"sid": v["sid"], "subset": sname(s), "by": "pos", "bin": pos, "n": int(k.sum()),
                                 "bias_ml": bb[0], "bias_ap": bb[1], "bias_dv": bb[2], "bias_mm": float(np.linalg.norm(bb)),
                                 "reach_mm": float(r[k].mean())})
    JT = pd.DataFrame(rows)
    PT = pd.DataFrame(prow)
    JT.to_csv(out / "jump_subset_vs_allview_sessions.csv", index=False, float_format="%.4f")
    PT.to_csv(out / "jump_pose_dependence_sessions.csv", index=False, float_format="%.4f")
    js = JT.groupby("subset").agg(n_sessions=("sid", "nunique"), freq_med=("freq_multiview", "median"),
                                  bias_med=("bias_mm", "median"), bias_max=("bias_mm", "max"),
                                  scatter_med=("scatter_mm", "median"),
                                  ml=("bias_ml", "median"), ap=("bias_ap", "median"), dv=("bias_dv", "median"))
    js = js.sort_values("freq_med", ascending=False)
    js.to_csv(out / "jump_summary.csv", float_format="%.4f")
    # pose dependence summary: per subset, median over sessions of (max - min bias magnitude across reach quintiles)
    pdp = []
    for (sid, sub, by), g in PT.groupby(["sid", "subset", "by"]):
        if len(g) < 2:
            continue
        B = g[["bias_ml", "bias_ap", "bias_dv"]].to_numpy()
        pdp.append({"sid": sid, "subset": sub, "by": by,
                    "range_mm": float(np.max(np.linalg.norm(B[:, None] - B[None], axis=2)))})
    PD = pd.DataFrame(pdp).groupby(["subset", "by"]).range_mm.median().unstack()
    PD.to_csv(out / "jump_pose_dependence_summary.csv", float_format="%.4f")
    # reach profile pooled (median over sessions per quintile)
    RP = PT[PT.by == "reach_q"].groupby(["subset", "bin"])[["bias_ml", "bias_ap", "bias_dv", "bias_mm", "reach_mm"]].median()
    RP.to_csv(out / "jump_reach_profile.csv", float_format="%.4f")
    nfill = sum(v["nfill"] for v in R)
    # ---------------- per-lick fixes
    L = pd.concat([v["licks"] for v in R], ignore_index=True)
    items, keymap = [], []
    for v in R:
        for it in v["sm"]:
            items.append(it)
            keymap.append((v["sid"], it["uid"]))
    fit4 = em_fit(items, "raw")
    fit4a = em_fit(items, "a2")
    two = [it for v in R for it in v["two"]]
    two_sid = [v["sid"] for v in R for _ in v["two"]]
    fit2 = em_fit(two, "raw")
    e1 = [it for v in R for it in v["e1"]]
    e1_key = [(v["sid"], it["uid"]) for v in R for it in v["e1"]]
    fit_e1 = em_fit(e1, "a2")
    with open(out / "em_fits.json", "w") as f:
        json.dump({k: {"q": fx["q"], "p0v": fx["p0v"], "classes": [sname(c) if k != "2cam" else str(c)
                                                                    for c in fx["classes"]],
                       "rvar_mm2": [float(x) for x in fx["rvar"]],
                       "sd_mm": [float(np.sqrt(x)) for x in fx["rvar"]], "n_iter": len(fx["hist"]),
                       "converge": fx["hist"][-3:], "trace_q": [h["q"] for h in fx["hist"]]}
                   for k, fx in (("4cam_raw", fit4), ("4cam_a2", fit4a), ("2cam", fit2), ("4cam_e1_a2", fit_e1))},
                  f, indent=1, default=str)
    sm = {}
    for nm, key, fx in (("d0", "raw", fit4), ("d", "a2", fit4a), ("s_clean_d0", "s_clean", fit4),
                        ("s_sw_d0", "s_sw", fit4), ("s_clean_d", "s_clean", fit4a), ("s_sw_d", "s_sw_a2", fit4a)):
        sm[nm] = smooth_all(items, key, fx["ci"], fx["rvar"], fx["q"], fx["p0v"])[0]
    sme1 = smooth_all(e1, "a2", fit_e1["ci"], fit_e1["rvar"], fit_e1["q"], fit_e1["p0v"])[0]
    E1 = pd.DataFrame([{"sid": s_, "lick_uid": u_, **dict(zip(("de1_csm40", "de1_csm50"), csm_of(sme1[i][0], [40, 50])))}
                       for i, (s_, u_) in enumerate(e1_key)])
    psd = {}
    rows = []
    for i, (sid, uid) in enumerate(keymap):
        r = {"sid": sid, "lick_uid": uid}
        for nm in ("d0", "d"):
            Pm, sdv = sm[nm][i]
            r[f"{nm}_csm40"], r[f"{nm}_csm50"] = csm_of(Pm, [40, 50])
            psd.setdefault(nm, []).append(np.median(sdv))
        for nm in ("s_clean_d0", "s_sw_d0", "s_clean_d", "s_sw_d"):
            r[nm] = tuple(csm_of(sm[nm][i][0], GRID))
        rows.append(r)
    L = L.merge(pd.DataFrame(rows), on=["sid", "lick_uid"], how="left").merge(E1, on=["sid", "lick_uid"], how="left")
    sm2 = smooth_all(two, "raw", fit2["ci"], fit2["rvar"], fit2["q"], fit2["p0v"])[0]
    T2 = pd.DataFrame([{"sid": s, "lick_uid": it["uid"], "pos": it["pos"], "two_raw_csm40": it["raw_csm40"],
                        "two_raw_csm50": it["raw_csm50"], "two_e2_csm40": it["e2_csm40"],
                        "two_e2_csm50": it["e2_csm50"], "two_e1_full_ok": it["e1_full_ok"],
                        "two_e1_csm40": it.get("e1_csm40", np.nan), "two_e1_csm50": it.get("e1_csm50", np.nan),
                        **dict(zip(("two_d_csm40", "two_d_csm50"), csm_of(sm2[i][0], [40, 50])))}
                       for i, (s, it) in enumerate(zip(two_sid, two))])
    # fix c from the peaks table
    PK = pd.read_pickle(D["out"] / "tables" / "peaks.pkl")
    PK = PK[(PK.run == "4cam") & (PK.kind == "csm")]
    for th in (40, 50):
        c = PK[(PK.th == th) & ~PK.near_chg].groupby(["sid", "lick_uid"]).size().rename(f"c_csm{th}")
        L = L.merge(c.reset_index(), on=["sid", "lick_uid"], how="left")
        L.loc[L.full_ok & L[f"c_csm{th}"].isna(), f"c_csm{th}"] = 0
        L.loc[~L.full_ok.astype(bool), f"c_csm{th}"] = np.nan
    L.to_pickle(out / "licks_fixes.pkl")
    T2.to_pickle(out / "licks_two_smoothed.pkl")
    # ---------------- calibration under the surrogates
    F = L[L.full_ok.astype(bool) & L.sur_raw_clean.notna()]
    cal = {}
    for nm, cc, ss in (("raw_iid", "sur_raw_clean", "sur_raw_iid"), ("raw_switch", "sur_raw_clean", "sur_raw_sw"),
                       ("a2_switch", "sur_raw_clean", "sur_a2_sw"), ("d0_switch", "s_clean_d0", "s_sw_d0"),
                       ("d_switch", "s_clean_d", "s_sw_d")):
        th, fr = calib(F[cc], F[ss])
        cal[nm] = {"theta": th, "false_at_40": fr[GRID.index(40)], "false_at_50": fr[GRID.index(50)],
                   "false_by_pos_at_40": {p: float(np.mean(np.array(list(F[F.pos == p][ss]))[:, GRID.index(40)] >
                                                           np.array(list(F[F.pos == p][cc]))[:, GRID.index(40)]))
                                          for p in POS_ORDER}}
    json.dump(cal, open(out / "calibration_switch_surrogate.json", "w"), indent=1)

    def at_theta(Lc, col_grid, th):
        if th is None:
            return pd.Series(np.nan, index=Lc.index)
        return Lc[col_grid].map(lambda x: x[GRID.index(int(th))] if isinstance(x, tuple) else np.nan)

    # CSM share at the calibrated theta for each pipeline: recompute counts at theta from stored grids is not possible
    # for real tracks (stored at 40 / 50 only) -> compute for the needed thetas here
    SH = {}
    for th in (40, 50):
        cols = {"2cam_raw": f"two_raw_csm{th}", "2cam_smoothed": f"two_d_csm{th}",
                "2cam_e2_no_end_peak": f"two_e2_csm{th}", "2cam_e1_no_fills": f"two_e1_csm{th}"}
        A2 = share_tab(T2, cols)
        A4 = share_tab(L[L.full_ok.astype(bool)], {"4cam_raw": f"raw_csm{th}", "a1_const_bias": f"a1_csm{th}",
                                                   "a2_reach_bias": f"a2_csm{th}", "c_mask_near_switch": f"c_csm{th}",
                                                   "d0_smoother": f"d0_csm{th}", "d_a2_plus_smoother": f"d_csm{th}",
                                                   "e2_no_end_peak": f"e2_csm{th}", "e1_no_fills": f"e1_csm{th}",
                                                   "e1_a2_smoother": f"de1_csm{th}"})
        Lb = L[L.b_full_ok.fillna(False).astype(bool)]
        Ab = share_tab(Lb, {"b_one_set_per_lick": f"b_csm{th}"})
        SH[th] = A2.merge(A4, on="pos").merge(Ab, on="pos")
        SH[th].to_csv(out / f"csm_share_fixes_theta{th}.csv", index=False, float_format="%.4f")
    bl = L.groupby("pos").agg(b_lost_frames=("b_lost", "mean"), b_full_ok=("b_full_ok", "mean"),
                              full_ok=("full_ok", "mean"), e1_full_ok=("e1_full_ok", "mean"),
                              fill_share=("fill_share", "mean"), fill_last=("fill_last", "mean")).reindex(list(POS_ORDER))
    bl = bl.join(T2.groupby("pos").two_e1_full_ok.mean())
    bl.to_csv(out / "fix_b_frames_lost.csv", float_format="%.4f")
    # ---------------- validation on truth licks
    val_rows, inj_rows = [], []
    vitems, vmeta = [], []
    for v in R:
        for tl in v["truth"]:
            P = tl["P"]
            n = len(P)
            dn = tl["donor"]
            if dn is None:
                continue
            seq = np.r_[dn, np.repeat(dn[-1], max(0, n - len(dn)))][:n].astype(int)
            b0 = tl["base"]
            seq = np.where((seq >= 100) | (seq <= 0), b0, seq)             # fills / interpolated -> the truth set
            Pm = P.copy()
            for s in np.unique(seq):
                if s != b0:
                    k = seq == s
                    Pm[k] = (tl["S"][s] if s != ALL else np.full_like(P, np.nan))[k]
            ok = np.isfinite(Pm).all(1)                                     # set not available at that frame
            Pm[~ok] = P[~ok]
            seq[~ok] = b0
            tab = {s: {"b": e["b"], "b5": e["b5"], "e5": e["e5"]} for s, e in v["tab"].items() if e.get("b5") is not None}
            Pm_a2 = Pm - bias_at(tab, seq, Pm, "b5")
            k = int(np.argmax(np.linalg.norm(P, axis=1)))
            # injection: minimum-jerk lateral pulse centred 3 frames after the extension peak (early retraction, low
            # speed), else 3 frames before it
            c0 = k + 3 if k + 7 < n else max(k - 3, 3)
            tt = (np.arange(n) - c0) * DT + INJ_D / 2
            s_ = np.clip(tt / INJ_D, 0, 1)
            prof = 10 * s_ ** 3 - 15 * s_ ** 4 + 6 * s_ ** 5
            base = {"sid": v["sid"], "uid": tl["uid"], "pos": tl["pos"], "n": n, "strict": tl["strict"]}
            for nm, Y, sq in (("truth", P, np.full(n, b0)), ("mimic", Pm, seq), ("mimic_a2", Pm_a2, seq)):
                vitems.append({"cls": sq, "Y": {"raw": Y, "a2": Y}})
                vmeta.append((base, nm, None))
                for vv in INJ_V:
                    Ai = vv * INJ_D / 1.875
                    Yi = Y.copy()
                    Yi[:, 0] += tl["inj_sign"] * Ai * prof
                    vitems.append({"cls": sq, "Y": {"raw": Yi, "a2": Yi}})
                    vmeta.append((base, nm, vv))
    if vitems:
        sv0 = smooth_all(vitems, "raw", fit4["ci"], fit4["rvar"], fit4["q"], fit4["p0v"])[0]
        sva = smooth_all(vitems, "a2", fit4a["ci"], fit4a["rvar"], fit4a["q"], fit4a["p0v"])[0]
        recs = []
        for i, (base, nm, vv) in enumerate(vmeta):
            Y = vitems[i]["Y"]["raw"]
            recs.append({**base, "series": nm, "inj": vv if vv else 0.0, "Y": Y, "d0": sv0[i][0], "d": sva[i][0]})
        V = pd.DataFrame(recs)
        for (sid, uid), g in V.groupby(["sid", "uid"]):
            gt = g[(g.series == "truth") & (g.inj == 0)].iloc[0]
            truthP = gt.Y
            c_truth = csm_of(truthP, 40)
            c_truth_d0 = csm_of(gt.d0, 40)
            c_truth_d = csm_of(gt.d, 40)
            for _, r in g[g.inj == 0].iterrows():
                for proc, Pp, ref in (("raw", r.Y, c_truth), ("d0", r.d0, c_truth_d0), ("d", r.d, c_truth_d)):
                    if r.series == "mimic_a2" and proc != "raw":
                        continue
                    lab = "a2" if (r.series == "mimic_a2") else proc
                    val_rows.append({"sid": sid, "uid": uid, "pos": r.pos, "strict": r.strict, "series": r.series,
                                     "proc": lab,
                                     "rms_mm": float(np.sqrt(((Pp - truthP) ** 2).sum(1).mean())),
                                     "csm40": csm_of(Pp, 40), "csm40_ref": ref, "csm40_truth_raw": c_truth})
            for _, r in g[g.inj > 0].iterrows():
                g0 = g[(g.series == r.series) & (g.inj == 0)].iloc[0]
                for proc in ("raw", "d0", "d"):
                    if r.series == "mimic_a2" and proc != "raw":
                        continue
                    P1 = r.Y if proc == "raw" else r[proc]
                    P0 = g0.Y if proc == "raw" else g0[proc]
                    inj_rows.append({"sid": sid, "uid": uid, "pos": r.pos, "strict": r.strict, "series": r.series,
                                     "proc": proc,
                                     "v_inj": r.inj, "recovered": csm_of(P1, 40) >= csm_of(P0, 40) + 1})
    VR = pd.DataFrame(val_rows)
    IR = pd.DataFrame(inj_rows)
    VR.to_csv(out / "validation_view_drop.csv", index=False, float_format="%.4f")
    IR.to_csv(out / "validation_injection.csv", index=False)
    vs = VR.groupby(["strict", "series", "proc"]).agg(n=("uid", "size"), rms_med=("rms_mm", "median"),
                                            rms_mean=("rms_mm", "mean"),
                                            csm_share=("csm40", lambda x: float((x >= 1).mean())),
                                            false_gain=("csm40", "size")).reset_index()
    fg = VR.assign(fg=(VR.csm40 > VR.csm40_ref)).groupby(["strict", "series", "proc"]).fg.mean().rename(
        "false_gain_vs_own_truth")
    vs = vs.drop(columns="false_gain").merge(fg.reset_index(), on=["strict", "series", "proc"])
    vs.to_csv(out / "validation_view_drop_summary.csv", index=False, float_format="%.4f")
    isum = IR.groupby(["series", "proc", "v_inj"]).recovered.mean().unstack()
    isum["n_licks"] = IR.groupby(["series", "proc"]).uid.nunique()
    # conditional: of the pulses the RAW track detects (same lick, same series, same size), share still detected
    piv = IR[IR.series == "truth"].pivot_table(index=["sid", "uid", "v_inj"], columns="proc", values="recovered",
                                               aggfunc="first").reset_index()
    cond = {pr: piv[piv.raw.astype(bool)].groupby("v_inj")[pr].mean() for pr in ("d0", "d") if pr in piv}
    pd.DataFrame(cond).to_csv(out / "validation_injection_conditional.csv", float_format="%.3f")
    isum.to_csv(out / "validation_injection_summary.csv", float_format="%.3f")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("one-view fill frames in lick windows:", nfill)
        print(js.round(3).to_string())
        print(PD.round(3).to_string())
        print(RP.round(3).to_string())
        print(json.dumps({k: (v["q"], v["p0v"]) for k, v in (("4cam_raw", fit4), ("4cam_a2", fit4a), ("2cam", fit2))}))
        print("median posterior SD mm:", {k: float(np.median(v)) for k, v in psd.items()})
        for th in (40, 50):
            print(f"theta {th}")
            print(SH[th].round(3).to_string())
        print(bl.round(3).to_string())
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "false_by_pos_at_40"} for k, v in cal.items()}))
        print(pd.DataFrame({k: v["false_by_pos_at_40"] for k, v in cal.items()}).round(3).to_string())
        print(vs.round(3).to_string())
        print(isum.round(3).to_string())
        print("conditional on raw detection (truth series):")
        print(pd.DataFrame(cond).round(3).to_string())
