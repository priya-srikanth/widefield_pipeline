"""Can the state-space readouts tell gain, rotation, shift, new dimensions and trial spread apart? (Priya 11-07, plan
docs/STATE_SPACE_RECOVERY_PLAN.md, "method check before the render lands".)

Uses ONLY pre-stroke sessions. Plants a known change into one held-out pre session at a time, then scores it against
every other pre session of the same animal, single session vs single session at matched trial counts. The UNCHANGED
held-out sessions give the null. Real trial noise, real trial counts; the acute-sized subsample asks what survives when
far-contra trials are few.

Data: per-trial area extracts (`first_cue`: first-lick trials x 71 lags (-1..2.5 s) x 63 Allen areas incl. ALM / tjM1),
binned to 0.1 s over -0.5..2.0 s. Left hemisphere = ipsilesional (all four lesions left).

Planted changes (to every trial of the test session; mu = the reference grand mean, P = the reference top-k space):
  none                      the null
  amp a                     mu + a (Y - mu)
  gain_rand s               mu + g o (Y - mu), log g ~ N(0, s) per area
  gain_ipsi g               left-hemisphere areas x g
  rot th                    rotation by th degrees in a random plane inside P
  shift f                   + a condition-independent time course along a random direction in P (norm = f x the shared
                            trajectory's norm)
  newdim f                  + a spout-specific time course along a direction OUTSIDE P (f = added share of variance)
  spread f                  + a per-trial random offset inside P, f x the pre trial-to-trial SD (local search)

Readouts (session vs session, median over the comparison sessions):
  out_var   share of the test session's centered variance outside P
  r, amp    pattern correlation and amplitude ratio vs the comparison session
  rot_gain  held-out (leave one spout out) R2 of rotation-after-per-area-gain minus per-area-gain alone (ladder L4 - L3)
  gain_sd   SD of log per-area gains (gain fitted per area)
  shared    norm of the spout-averaged trajectory, ratio
  spread    trial-to-trial spread inside P around each spout mean, ratio
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ITI = ("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
       "iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801/sessions_roi8fr")
OUT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery/sim")
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
K = 8
MET = ["out_var", "out_resid", "r", "amp", "rot_gain", "gain_sd", "shared", "shared_dist", "spread", "spread_norm"]
LAGS = np.round(np.arange(-1.0, 2.5 + 1e-9, 0.05), 3)
BINS = [(a, a + 0.1) for a in np.round(np.arange(-0.5, 2.0, 0.1), 2)]
PLANTS = ([("none", 0)] + [("amp", a) for a in (1.25, 1.5)] + [("gain_rand", s) for s in (0.15, 0.3)]
          + [("gain_ipsi", g) for g in (0.8, 0.6)] + [("rot", t) for t in (10, 20, 30, 45)] + [("rot12", t) for t in (10, 20, 30)]
          + [("shift", f) for f in (0.25, 0.5)] + [("newdim", f) for f in (0.05, 0.15)]
          + [("spread", f) for f in (0.5, 1.0)])


def load(path):
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["cols"]]
    keep = [i for i, c in enumerate(cols) if c != "cortex"]
    x = z["first_cue"].astype(float)[:, :, keep]                      # trials x lags x areas
    lag = LAGS[: x.shape[1]]
    xb = np.stack([np.nanmean(x[:, (lag >= a - 1e-9) & (lag < b - 1e-9)], axis=1) for a, b in BINS], axis=1)
    pos = np.array([str(p) for p in z["first_pos"]])
    return xb, pos, [cols[i] for i in keep], str(z["epoch"]), str(z["animal"])


def spout_means(Y, pos):
    return np.stack([Y[pos == p].mean(0) if (pos == p).any() else np.full(Y.shape[1:], np.nan) for p in POS])


def flat(M):
    return M.reshape(-1, M.shape[-1])


def topk(M, k=K):
    F = flat(M[np.isfinite(M).all(axis=(1, 2))])
    return np.linalg.svd(F - F.mean(0), full_matrices=False)[2][:k].T


def fit_gain(X, T):
    Tm, Xm = T.mean(0), X.mean(0)
    den = ((T - Tm) ** 2).sum(0)
    g = np.where(den > 0, ((T - Tm) * (X - Xm)).sum(0) / den, 1.0)
    return g, Xm - g * Tm


def rot_beyond_gain(X, T, P):
    """Held-out R2 (leave one spout out): L4 (per-area gain, then Procrustes in P) minus L3 (per-area gain)."""
    ok = [s for s in range(6) if np.isfinite(X[s]).all() and np.isfinite(T[s]).all()]
    sse3 = sse4 = sst = 0.0
    for h in ok:
        tr = [s for s in ok if s != h]
        Xtr, Ttr = flat(X[tr]), flat(T[tr])
        g, b = fit_gain(Xtr, Ttr)
        Ytr, Yte = Ttr * g + b, T[h] * g + b
        mu = Ytr.mean(0)
        u, _, vt = np.linalg.svd(((Ytr - mu) @ P).T @ ((Xtr - mu) @ P))
        Z = (Yte - mu) @ P
        pred4 = Yte + (Z @ (u @ vt) - Z) @ P.T
        sst += ((X[h] - Xtr.mean(0)) ** 2).sum()
        sse3 += ((X[h] - Yte) ** 2).sum()
        sse4 += ((X[h] - pred4) ** 2).sum()
    return (sse3 - sse4) / sst


def spread(Y, pos, P):
    d = []
    for p in POS:
        y = Y[pos == p]
        if len(y) > 1:
            z = flat(y - y.mean(0)) @ P
            d.append((z ** 2).sum() / len(y))
    return float(np.mean(d))


def readouts(Ya, pa, Yb, pb, P):
    A, B = spout_means(Ya, pa), spout_means(Yb, pb)
    ok = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(B).all(axis=(1, 2))
    Af, Bf = flat(A[ok]), flat(B[ok])
    Ac, Bc = Af - Af.mean(0), Bf - Bf.mean(0)
    g, b0 = fit_gain(Af, Bf)
    As, Bs = A[ok].mean(0), B[ok].mean(0)
    # outside-P variance of what per-area gains CANNOT explain: separates new dimensions from region gains
    R = Af - (Bf * g + b0)
    amp = np.linalg.norm(Ac) / np.linalg.norm(Bc)
    return {"out_var": float(1 - ((Ac @ P) ** 2).sum() / (Ac ** 2).sum()),
            "r": float((Ac * Bc).sum() / np.linalg.norm(Ac) / np.linalg.norm(Bc)),
            "amp": float(amp),
            "out_resid": float((R - R @ P @ P.T).var() / Ac.var()),
            "rot_gain": float(rot_beyond_gain(A, B, P)),
            "gain_sd": float(np.std(np.log(np.clip(g, 1e-3, None)))),
            "shared": float(np.linalg.norm(As - As.mean(0)) / np.linalg.norm(Bs - Bs.mean(0))),
            "shared_dist": float(np.linalg.norm(As - Bs) / np.linalg.norm(Bs - Bs.mean(0))),
            "spread": spread(Ya, pa, P) / spread(Yb, pb, P),
            "spread_norm": spread(Ya, pa, P) / spread(Yb, pb, P) / amp ** 2}


def plant(Y, pos, kind, lvl, mu, P, ref_means, cols, rng):
    Yc = Y - mu
    if kind == "none":
        return Y
    if kind == "amp":
        return mu + lvl * Yc
    if kind == "gain_rand":
        return mu + Yc * np.exp(rng.normal(0, lvl, Y.shape[-1]))
    if kind == "gain_ipsi":
        g = np.array([lvl if c.endswith("_left") else 1.0 for c in cols])
        return mu + Yc * g
    if kind in ("rot", "rot12"):
        q = np.eye(P.shape[1])[:, :2] if kind == "rot12" else np.linalg.qr(rng.normal(size=(P.shape[1], 2)))[0]
        u, v = P @ q[:, 0], P @ q[:, 1]
        th = np.deg2rad(lvl)
        R = (np.eye(len(u)) + (np.cos(th) - 1) * (np.outer(u, u) + np.outer(v, v))
             + np.sin(th) * (np.outer(v, u) - np.outer(u, v)))
        return mu + Yc @ R.T
    shared = np.nanmean(ref_means, axis=0)                              # bins x areas
    if kind == "shift":
        w = P @ rng.normal(size=P.shape[1])
        w /= np.linalg.norm(w)
        prof = np.linalg.norm(shared - shared.mean(0), axis=1)
        add = np.outer(prof / np.linalg.norm(prof), w) * lvl * np.linalg.norm(shared - shared.mean(0))
        return Y + add
    if kind == "newdim":
        w = rng.normal(size=Y.shape[-1])
        w -= P @ (P.T @ w)
        w /= np.linalg.norm(w)
        code = ref_means - shared                                       # spout x bins x areas
        h = code @ P[:, 0]                                              # spout x bins: the main position code
        tot = (flat(np.nan_to_num(ref_means - flat(ref_means).mean(0))) ** 2).sum() / flat(ref_means).shape[0]
        h = h / np.sqrt(np.nanmean(h ** 2)) * np.sqrt(lvl * tot)
        add = np.stack([np.outer(np.nan_to_num(h[POS.index(p)]), w) for p in pos])
        return Y + add
    if kind == "spread":
        sd = np.sqrt(spread(Y, pos, P) / Y.shape[1] / P.shape[1])
        off = (rng.normal(0, lvl * sd, size=(len(Y), P.shape[1])) @ P.T)[:, None, :]
        return Y + off
    raise ValueError(kind)


def subsample(Y, pos, n, rng):
    idx = np.concatenate([rng.permutation(np.flatnonzero(pos == p))[:n] for p in POS])
    return Y[idx], pos[idx]


def run(n_per_spout, seed=0):
    rng = np.random.default_rng(seed)
    sess = {}
    for f in sorted(glob.glob(ITI + "/PS9*.npz")):
        Y, pos, cols, ep, an = load(f)
        if ep == "pre":
            sess.setdefault(an, []).append((Path(f).stem, Y, pos))
    rows = []
    for an, S in sorted(sess.items()):
        for i, (lab, Y, pos) in enumerate(S):
            others = [s for j, s in enumerate(S) if j != i]
            ref_means = np.nanmean(np.stack([spout_means(y, p) for _, y, p in others]), axis=0)
            P = topk(ref_means)
            mu = flat(ref_means[np.isfinite(ref_means).all(axis=(1, 2))]).mean(0)
            for kind, lvl in PLANTS:
                Yp = plant(Y, pos, kind, lvl, mu, P, ref_means, cols, rng)
                vals = []
                for _, Yo, po in others:
                    if n_per_spout:
                        n = min(n_per_spout, *[int((pos == p).sum()) for p in POS], *[int((po == p).sum()) for p in POS])
                        a, pa = subsample(Yp, pos, n, rng)
                        b, pb = subsample(Yo, po, n, rng)
                    else:
                        a, pa, b, pb = Yp, pos, Yo, po
                    vals.append(readouts(a, pa, b, pb, P))
                med = pd.DataFrame(vals).median().to_dict()
                rows.append({"animal": an, "label": lab, "plant": kind, "level": lvl, "n_per_spout": n_per_spout, **med})
    return pd.DataFrame(rows), cols


def detection(df):
    """Share of planted sessions beyond the pooled null's 2.5-97.5 % range, per readout (z within animal)."""
    mets = MET
    z = df.copy()
    for _an, g in df.groupby("animal"):
        nul = g[g.plant == "none"]
        for m in mets:
            z.loc[g.index, m] = (g[m] - nul[m].mean()) / nul[m].std(ddof=1)
    nul = z[z.plant == "none"]
    lo, hi = nul[mets].quantile(0.025), nul[mets].quantile(0.975)
    det = z.groupby(["plant", "level"])[mets].apply(lambda g: ((g < lo) | (g > hi)).mean())
    return det, z


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", nargs="+", type=int, default=[0, 15], help="trials per spout (0 = all, matched by session)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    order = list(dict.fromkeys(PLANTS))
    for n in a.n:
        df, _ = run(n)
        df.to_csv(out / f"sim_readouts_n{n}.csv", index=False)
        det, _ = detection(df)
        det = det.reindex(order)
        det.to_csv(out / f"sim_detection_n{n}.csv")
        with __import__("pandas").option_context("display.width", 200, "display.precision", 2):
            print(f"\n=== detection rate, n per spout = {n or 'all'} (null row = false-positive rate) ===")
            print(det)
            print("\nmedian readouts:")
            print(df.groupby(["plant", "level"])[MET]
                  .median().reindex(order))


if __name__ == "__main__":
    main()
