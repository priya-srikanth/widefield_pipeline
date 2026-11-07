"""Change vs each animal's own pre-stroke sessions, the way every FaceRhythm readout reports it.

Two steps, always in this order (CLAUDE.md rules 8 and 9):
  1. per animal: the mean of its sessions in each epoch, relative to its own pre-stroke sessions -- PRINTED FIRST, so a
     cell carried by one animal is visible before any interval is;
  2. the change vs pre by `analysis_kit.boot_delta` (paired animals -> sessions bootstrap, animals weighted equally,
     95 % interval). There is no second bootstrap here: this module only arranges the inputs.

Marks (Priya 10-24 "mark with asterisks where appropriate"): ``*`` = the 95 % interval excludes 0 (one planned test);
``**`` = also significant after a Holm correction over the FAMILY the figure tests at once (e.g. all 12 factors of an
epoch x trial group in the survey), from the bootstrap p of the same draws (`analysis_kit.boot_p`, crude at n = 4:
used to rank and correct, not quoted as an exact probability).

Scales: LOG2 for amplitudes (halving and doubling equally large, Priya 10-24; pre averages exactly 0), ratio / pre or
difference - pre for quantities that are already relative (EVR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wfield_local.face_rhythm.common import POST_EPOCHS


def log2_vs_pre(S: pd.DataFrame, value: str, by: list[str], floor: float = 1e-12) -> pd.Series:
    """log2(value) minus the mean log2 value of the same animal's PRE sessions within each ``by`` group (``by``
    includes "animal"); pre sessions average exactly 0. ``floor`` guards log2(0)."""
    lg = np.log2(np.maximum(S[value], floor))
    return lg - lg.where(S.epoch == "pre").groupby([S[b] for b in by]).transform("mean")


def by_animal(d: pd.DataFrame, value: str) -> dict[str, list]:
    """{animal: [session values]} in the frame's row order -- the input `boot_delta` takes."""
    return {an: list(g[value]) for an, g in d.groupby("animal")}


def epoch_change(d: pd.DataFrame, value: str, rng, n_boot: int | None = None, epochs=POST_EPOCHS,
                 baseline: float = 0.0) -> list[dict]:
    """One row per post epoch for ONE quantity (``d`` = its sessions, all epochs, column ``value`` already relative
    to pre): per-animal epoch means (one column per animal), their mean and SEM across animals, the `boot_delta`
    change vs pre (delta, ci_lo, ci_hi, n_animals, excludes_0) and how many animals sit below / above ``baseline``.
    ``rng`` is advanced in epoch order, so a caller looping over quantities reproduces the same draws every run."""
    from wfield_local import analysis_kit as ak
    pre = by_animal(d[d.epoch == "pre"], value)
    rows = []
    for ep in epochs:
        post = by_animal(d[d.epoch == ep], value)
        per = {an: float(np.mean(v)) for an, v in sorted(post.items())}
        iv, draws = ak.boot_delta(post, pre, rng, n_boot or ak.N_BOOT, draws=True)
        vals = list(per.values())
        rows.append({"epoch": ep, **per,
                     "mean": float(np.mean(vals)) if vals else np.nan,
                     "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                     "delta": iv.point if iv else np.nan, "ci_lo": iv.lo if iv else np.nan,
                     "ci_hi": iv.hi if iv else np.nan, "n_animals": iv.n_animals if iv else len(per),
                     "excludes_0": bool(iv and (iv.lo > 0 or iv.hi < 0)),
                     "p_boot": ak.boot_p(draws) if iv else np.nan,
                     "n_below": sum(v < baseline for v in vals), "n_above": sum(v > baseline for v in vals)})
    return rows


def add_holm(T: pd.DataFrame, family: list[str], p: str = "p_boot") -> pd.DataFrame:
    """Holm-adjust ``p`` within each family (rows sharing the ``family`` columns, e.g. ["epoch", "group"] for the 12
    factors of one epoch x trial group) -> columns ``p_holm`` and ``mark`` ("", "*", "**")."""
    from wfield_local import analysis_kit as ak
    T = T.copy()
    T["p_holm"] = np.nan
    for _, idx in T.groupby(family).groups.items():
        T.loc[idx, "p_holm"] = ak.holm(T.loc[idx, p].to_numpy(float))
    T["mark"] = [("**" if ph < 0.05 else "*") if ex else "" for ex, ph in zip(T.excludes_0, T.p_holm.fillna(1.0))]
    return T


def mark(row) -> str:
    """"*" / "**" of a stats row: its ``mark`` column when Holm-adjusted (`add_holm`), else "*" if the interval
    excludes 0."""
    if row is None:
        return ""
    m = row.get("mark") if hasattr(row, "get") else None
    if isinstance(m, str):
        return m
    return "*" if row["excludes_0"] else ""


def tick_label(epoch: str, row: dict | pd.Series | None, fmt: str = "+.2f", interval: bool = True) -> str:
    """Epoch tick label carrying its statistics: ``acute\\n-1.35*\\n[-1.56, -1.13]`` (* = interval excludes 0)."""
    if row is None or not np.isfinite(row["delta"]):
        return epoch[:5]
    s = f"{epoch[:5]}\n{row['delta']:{fmt}}{mark(row)}"
    return s + (f"\n[{row['ci_lo']:{fmt}}, {row['ci_hi']:{fmt}}]" if interval else "")


# ---- permutation tests over the face mesh (Priya 11-07: cluster mass as the main map test, max-statistic for points) ----

def mesh_adjacency(home: np.ndarray, factor: float = 1.6) -> list[np.ndarray]:
    """Neighbours of each mesh point: points within ``factor`` x the median nearest-neighbour distance (the mesh is a
    near-regular grid, so this is the 8-neighbourhood)."""
    from scipy.spatial import cKDTree
    home = np.asarray(home, float)
    tree = cKDTree(home)
    d, _ = tree.query(home, k=2)
    r = factor * float(np.median(d[:, 1]))
    return [np.array([j for j in nb if j != i], int) for i, nb in enumerate(tree.query_ball_point(home, r))]


def clusters(t: np.ndarray, adj: list[np.ndarray], thr: float) -> tuple[np.ndarray, list[float]]:
    """Connected clusters of points with t > thr and, separately, t < -thr. -> (label per point, -1 = none; mass per
    cluster = sum of |t| over its points)."""
    t = np.asarray(t, float)
    lab = np.full(len(t), -1, int)
    mass = []
    for sign in (1, -1):
        sup = np.isfinite(t) & (sign * t > thr)
        for s in np.flatnonzero(sup):
            if lab[s] >= 0:
                continue
            c = len(mass)
            stack, lab[s], m = [s], c, 0.0
            while stack:
                i = stack.pop()
                m += abs(t[i])
                for j in adj[i]:
                    if sup[j] and lab[j] < 0:
                        lab[j] = c
                        stack.append(j)
            mass.append(m)
    return lab, mass


def max_cluster_mass(t: np.ndarray, adj: list[np.ndarray], thr: float) -> float:
    return max(clusters(t, adj, thr)[1], default=0.0)


def map_permutation(obs: np.ndarray, null: np.ndarray, adj: list[np.ndarray], thr: float = 2.0) -> dict:
    """Family-wise tests of one change map against its permutation null. ``obs`` (P,), ``null`` (n, P) = the same
    statistic under label shuffles. Each point is studentised by its null SD (t); then
      max-statistic  point p = share of permutations whose max |t| over the face >= the point's |t|
      cluster mass   clusters of |t| > ``thr`` (sign-separate, mesh neighbours); cluster p = share of permutations whose
                     largest cluster mass >= the cluster's (+1 / +1 throughout).
    -> t, p_point (FWE), cluster label per point, cluster masses, p per cluster, sig_cluster (P,) bool, sig_point."""
    obs, null = np.asarray(obs, float), np.asarray(null, float)
    sd = np.nanstd(null, 0)
    sd = np.where(sd > 0, sd, np.nan)
    t = obs / sd
    tn = null / sd[None, :]
    n = len(null)
    mx = np.nanmax(np.abs(tn), 1)
    p_point = (1 + (mx[None, :] >= np.abs(t)[:, None]).sum(1)) / (1 + n)
    lab, mass = clusters(t, adj, thr)
    nmax = np.array([max_cluster_mass(tn[i], adj, thr) for i in range(n)])
    p_cl = np.array([(1 + (nmax >= m).sum()) / (1 + n) for m in mass])
    sig_cl = np.zeros(len(t), bool)
    for c, pc in enumerate(p_cl):
        if pc < 0.05:
            sig_cl |= lab == c
    return {"t": t, "p_point": p_point, "sig_point": p_point < 0.05, "label": lab, "mass": np.array(mass),
            "p_cluster": p_cl, "sig_cluster": sig_cl, "n_perm": n, "thr": thr}


def session_shuffle_test(per: list[tuple[np.ndarray, int]], adj: list[np.ndarray], n_perm: int, rng,
                         thr: float = 2.0) -> dict | None:
    """Epoch vs reference over sessions, family-wise over the face. ``per`` = one (X, n_ref) per animal: X (sessions, P)
    with that animal's n_ref REFERENCE sessions first (pre, or early pre for the drift control). Statistic = mean over
    animals of (mean of the other sessions - mean of the reference sessions); null = the same with each animal's session
    labels shuffled within the animal (counts kept). -> `map_permutation` result + "mean", "n_animals", "n_sessions",
    "n_ref"; None with < 2 animals."""
    per = [(np.asarray(X, float), int(n)) for X, n in per if 0 < n < len(X)]
    if len(per) < 2:
        return None

    def stat(idx):
        return np.mean([np.nanmean(X[i[n:]], 0) - np.nanmean(X[i[:n]], 0) for (X, n), i in zip(per, idx)], 0)

    obs = stat([np.arange(len(X)) for X, _ in per])
    null = np.array([stat([rng.permutation(len(X)) for X, _ in per]) for _ in range(n_perm)])
    r = map_permutation(obs, null, adj, thr)
    r.update(mean=obs, n_animals=len(per), n_sessions=sum(len(X) - n for X, n in per), n_ref=sum(n for _, n in per))
    return r


def drift_split(X_pre_in_date_order: np.ndarray) -> tuple[np.ndarray, int]:
    """DRIFT CONTROL (as `fr_pixel_significance.py --drift-control`): one animal's pre sessions in date order -> (X, n_ref)
    with the EARLY half (first n // 2) as the reference and the LATE half as the 'epoch': the chronological member of
    the pre half-splits. A post-stroke cluster that the pre drift also shows is not a stroke effect."""
    X = np.asarray(X_pre_in_date_order, float)
    return X, len(X) // 2


def chain_adjacency(n: int) -> list[np.ndarray]:
    """Neighbours along a 1-D axis (frequency bins) for `map_permutation`."""
    return [np.array([j for j in (i - 1, i + 1) if 0 <= j < n], int) for i in range(n)]
