"""Follow-ups to the per-spout toward-midline results (Priya 10-21, items 2-3), cam4 lick-aligned, same trials / response
windows / drift filter as scripts/within_spout_deviation.py. Per session, MOs and SSp-m (left / right / bilateral mean):

(2) ECCENTRICITY TUNING: on the four lateral spouts, within-spout responses (spout means removed; covariates reach,
    n_licks, latency, n_contact) fitted by
      A  free slope per spout on the lateral angle (4 params)
      B  one curve in ECCENTRICITY e = |lateral angle| shared by all four spouts: e, e^2 (2 params)
      B' one curve in the SIGNED angle a: a, a^2 (2 params)
    5-fold CV R2 (contiguous folds); B close to A = the per-spout sign pattern is one eccentricity tuning curve.
    Peak e* = -g1 / (2 g2) when g2 < 0.
(3) FIRST LICK: per-spout toward-midline slope on the FIRST lick's angle with first-lick reach, n_licks, latency and a
    0/1 FIRST-LICK CONTACT covariate (a contact between the first lick's onset and the next lick's onset, <= 250 ms);
    counts of miss-then-correct trials (first lick no contact, a later contact in the window).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

LATERAL = ("close_contra", "close_ipsi", "far_contra", "far_ipsi")
REGS = ("MOs", "SSp-m")


def _cv_r2(y, X, k=5):
    n = len(y)
    f = np.minimum(np.arange(n) * k // n, k - 1)
    pred = np.zeros_like(y)
    for i in range(k):
        tr, te = f != i, f == i
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        pred[te] = X[te] @ b
    return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def _demean(M, pos):
    M = M.copy()
    for p in np.unique(pos):
        m = pos == p
        M[m] -= M[m].mean(0)
    return M


def one(label):
    import os
    from scripts import within_spout_deviation as W
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.paths import PathResolver
    os.environ[W.ALIGN_ENV] = "lick"
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    B = ce.load_behaviour(config.animal_of(label), "2026" + label.split("_")[-1], PathResolver())
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0].astype(np.float32)
    V = V - Q @ (Q.T @ V)
    E = B.trials[B.trials.engaged & np.isfinite(B.trials.cue_s)].reset_index(drop=True)
    T = W.trial_table(B, E, "", "lick")
    cue = E.cue_s.to_numpy(float)[T.k.to_numpy()]
    t0 = T.t0.to_numpy(float)

    def _mean(ref, a, b):
        i0, i1 = np.searchsorted(ft, ref + a), np.searchsorted(ft, ref + b)
        return np.stack([V[x:y].mean(0) if y > x else np.full(V.shape[1], np.nan, V.dtype) for x, y in zip(i0, i1)])
    Rk = _mean(t0, *W.LICK_WIN_S) - _mean(cue, *W.BASE_S)
    ok = np.isfinite(Rk).all(1)
    T, Rk = T[ok].reset_index(drop=True), Rk[ok]
    inv = {v: k for k, v in names.items()}
    cols, cn = [], []
    for r in REGS:
        sides = [Ub[(parea == inv[f"{r}_{s}"]) & pstat].mean(0) for s in ("left", "right") if f"{r}_{s}" in inv]
        if len(sides) == 2:
            cols += [sides[0], sides[1], (sides[0] + sides[1]) / 2]
            cn += [f"{r}_left", f"{r}_right", f"{r} bilateral"]
    R = Rk @ np.stack(cols, 1) * 1e3                            # x1e-3 dF
    # first-lick contact: a contact between the first lick's onset and the next lick (<= 250 ms)
    on = np.sort(np.asarray(B.lick_on_s, float))
    con = np.sort(np.asarray(B.contact_s, float))
    nxt = np.array([on[np.searchsorted(on, x, side="right")] if np.searchsorted(on, x, side="right") < len(on)
                    else x + 0.25 for x in t0[ok]])
    end = np.minimum(nxt, t0[ok] + 0.25)
    T["first_contact"] = (np.searchsorted(con, end) - np.searchsorted(con, t0[ok])) > 0
    T["miss_then_correct"] = (~T.first_contact) & (T.n_contact > 0)
    rows = []
    lat_m = T.pos.isin(LATERAL).to_numpy() & np.isfinite(T.lat_mean).to_numpy()
    if lat_m.sum() >= 40 and len(np.unique(T.pos[lat_m])) == 4:
        G, RG, pos = T[lat_m].reset_index(drop=True), R[lat_m], T.pos[lat_m].to_numpy()
        a = G.lat_mean.to_numpy(float) / 10.0                  # per 10 deg, + = toward the mouse's LEFT
        e = np.abs(a)
        C = np.column_stack([G.reach_mean, G.n_licks, G.latency, G.n_contact]).astype(float)
        Xa = np.column_stack([np.where(pos == p, a, 0.0) for p in LATERAL])
        Yd = _demean(RG, pos)
        base = _demean(C, pos)
        r2 = {"null": _cv_r2(Yd, base), "A_per_spout": _cv_r2(Yd, np.column_stack([base, _demean(Xa, pos)])),
              "B_ecc": _cv_r2(Yd, np.column_stack([base, _demean(np.column_stack([e, e ** 2]), pos)])),
              "Bs_signed": _cv_r2(Yd, np.column_stack([base, _demean(np.column_stack([a, a ** 2]), pos)]))}
        gb = np.linalg.lstsq(np.column_stack([base, _demean(np.column_stack([e, e ** 2]), pos)]), Yd, rcond=None)[0]
        spout_e = {p: float(np.median(e[pos == p])) * 10 for p in LATERAL}
        for i, c in enumerate(cn):
            g1, g2 = gb[-2, i], gb[-1, i]
            rows.append({"label": label, "test": "tuning", "area": c, **{k: float(v[i]) for k, v in r2.items()},
                         "g1": g1, "g2": g2, "peak_deg": float(-g1 / (2 * g2) * 10) if g2 < 0 else np.nan,
                         **{f"ecc_{p}": spout_e[p] for p in LATERAL}})
    for p in LATERAL:                                           # (3) first-lick slope with first-lick contact
        m = (T.pos == p).to_numpy() & np.isfinite(T.lat_first).to_numpy() & np.isfinite(T.reach_first).to_numpy()
        if m.sum() < W.MIN_TRIALS:
            continue
        G = T[m]
        sg = 1.0 if p.endswith("contra") else -1.0
        X = np.column_stack([G.lat_first / 10.0 * sg, G.reach_first, G.n_licks, G.latency,
                             G.first_contact.astype(float)]).astype(float)
        X = X - X.mean(0)
        Y = R[m] - R[m].mean(0)
        b = np.linalg.lstsq(X, Y, rcond=None)[0]
        for i, c in enumerate(cn):
            rows.append({"label": label, "test": "first_lick", "pos": p, "area": c, "b_mid_first": float(b[0, i]),
                         "b_first_contact": float(b[4, i]), "n": int(m.sum()),
                         "frac_first_contact": float(G.first_contact.mean()),
                         "frac_miss_then_correct": float(G.miss_then_correct.mean()),
                         "frac_no_contact": float((G.n_contact == 0).mean())})
    return rows


def main():
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    labels = [x for x in ce.sessions_available() if x in set(ak.curated_labels()) and x[-4:] >= "0806"]
    got, fail = ak.fan_sessions(labels, one, jobs=int(sys.argv[1]))
    for it, err in fail:
        print("FAILED", it, err)
    D = pd.DataFrame([r for _, rr in got for r in rr])
    D["animal"], D["epoch"] = D.label.str[:4], D.label.map(epochs.epoch_of)
    D.to_csv(sys.argv[2], index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    order = ["pre", "acute", "subacute", "chronic"]
    U = D[(D.test == "tuning") & D.area.str.endswith("bilateral")]
    print("=== (2) CV R2 x100 of within-spout responses, bilateral, session means (A per-spout slopes, B eccentricity curve)")
    for c in ("A_per_spout", "B_ecc", "Bs_signed"):
        U = U.assign(**{f"{c}_gain": (U[c] - U.null) * 100})
    print(U.groupby(["area", "animal", "epoch"])[["A_per_spout_gain", "B_ecc_gain", "Bs_signed_gain"]].mean()
          .unstack("epoch").reindex(columns=order, level=1).round(2).to_string())
    print("\n  eccentricity curve: curvature g2 (<0 = peaked) and peak (deg), session medians; spout eccentricities (deg)")
    print(U.groupby(["area", "animal", "epoch"])[["g2", "peak_deg"]].median().unstack("epoch")
          .reindex(columns=order, level=1).round(2).to_string())
    print(U.groupby("animal")[[f"ecc_{p}" for p in LATERAL]].median().round(1).to_string())
    F = D[(D.test == "first_lick") & D.area.str.endswith("bilateral")]
    print("\n=== (3) FIRST-LICK toward-midline slope (x1e-3 dF / 10 deg) with first-lick contact covariate, session means")
    print(F.groupby(["pos", "area", "animal", "epoch"]).b_mid_first.mean().unstack("epoch").reindex(columns=order)
          .loc[["far_ipsi", "close_contra", "far_contra"]].round(2).to_string())
    print("\n  trial fractions (session means): first lick touched / miss-then-correct / no contact in window")
    print(F[F.area == "MOs bilateral"].groupby(["pos", "animal", "epoch"])[
        ["frac_first_contact", "frac_miss_then_correct", "frac_no_contact"]].mean().unstack("epoch")
          .reindex(columns=order, level=1).round(2).to_string())


if __name__ == "__main__":
    main()
