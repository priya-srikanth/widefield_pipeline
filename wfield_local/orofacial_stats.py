"""Statistics for the orofacial pre- vs post-stroke cohort figures (Priya 2026-10-17: "start including stats and results;
see stroke_orofacial").

Follows stroke_orofacial's inferential hierarchy (`stroke_orofacial_pipeline/docs/STATS_PIPELINE.md`), adapted:

* **Headline = session Mann-Whitney** (`session_change_tests`): per (metric, position, post epoch), the per-session
  CHANGE from each animal's own pre-stroke mean in that epoch vs the same quantity in the pre-stroke sessions (which sit
  around 0 by construction), two-sided Mann-Whitney U on SESSIONS. BH-FDR within each metric (all positions x epochs).
  Trial-grain tests are excluded on purpose (stroke_orofacial measured 39-46 % type-I error under H0).
* **Robustness = this repo's animal-weighted bootstrap** (`analysis_kit.boot_delta`, CLAUDE.md rule 9): the change with a
  nested animals -> sessions CI. Sessions are not independent across animals, so where the session MWU and the bootstrap
  disagree the table says so (`agree`), and an effect carried by one animal shows up there first.
* **Trajectories = cluster permutation** (`cluster_permutation`): per bin, Welch t of epoch-session changes vs pre-session
  changes; clusters of contiguous bins with |t| above the two-sided p < 0.05 threshold; cluster mass = sum |t|; null =
  max cluster mass after permuting epoch / pre labels WITHIN each animal (an animal's sessions never move to another
  animal). Clusters with permutation p < 0.05 are drawn as a bar above the curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 0.05


def bh_fdr(p) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN p stay NaN)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(pv))
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def _deltas(P: pd.DataFrame, m: str) -> pd.DataFrame:
    """Per-session change of metric m from each animal's own pre-stroke mean at the same position."""
    base = P[P.epoch == "pre"].groupby(["animal", "pos"])[m].mean()
    d = P[["animal", "date", "epoch", "pos", m]].copy()
    d["delta"] = d[m] - base.reindex(pd.MultiIndex.from_frame(d[["animal", "pos"]])).to_numpy()
    return d.groupby(["animal", "date", "epoch", "pos"], as_index=False).delta.mean()


def session_change_tests(P: pd.DataFrame, metrics, epochs=("acute", "subacute", "chronic"), seed: int = 0,
                         n_boot: int = 2000) -> pd.DataFrame:
    """One row per (metric, position, epoch): session MWU headline (+ BH-FDR within metric) and the animal-weighted
    bootstrap change with its 95% CI."""
    from scipy.stats import mannwhitneyu

    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        if m not in P:
            continue
        d = _deltas(P, m)
        block = []
        for pos, gp in d.groupby("pos", sort=True):
            pre = gp[gp.epoch == "pre"]
            for e in epochs:
                post = gp[gp.epoch == e]
                x, y = post.delta.dropna().to_numpy(), pre.delta.dropna().to_numpy()
                r = {"metric": m, "pos": pos, "epoch": e, "n_sessions": len(x), "n_pre_sessions": len(y),
                     "n_animals": int(post.animal.nunique()), "median_change": float(np.median(x)) if len(x) else np.nan,
                     "mean_change": float(np.mean(x)) if len(x) else np.nan}
                if len(x) >= 2 and len(y) >= 2:
                    u, p = mannwhitneyu(x, y, alternative="two-sided")
                    r.update(mwu_U=float(u), p=float(p))
                else:
                    r.update(mwu_U=np.nan, p=np.nan)
                post_a = {a: g.delta.dropna().tolist() for a, g in post.groupby("animal", sort=True)}
                pre_a = {a: g.delta.dropna().tolist() for a, g in pre.groupby("animal", sort=True)}
                pre_a = {a: v for a, v in pre_a.items() if v and post_a.get(a)}
                post_a = {a: v for a, v in post_a.items() if v and a in pre_a}
                iv = ak.boot_delta(post_a, pre_a, rng, n_boot=n_boot) if len(post_a) >= 2 else None
                r.update(boot_change=iv.point if iv else np.nan, boot_lo=iv.lo if iv else np.nan,
                         boot_hi=iv.hi if iv else np.nan)
                block.append(r)
        q = bh_fdr([b["p"] for b in block])
        for b, qq in zip(block, q):
            b["q_fdr"] = float(qq)
            b["headline_sig"] = bool(np.isfinite(qq) and qq < ALPHA)
            b["boot_sig"] = bool(np.isfinite(b["boot_lo"]) and (b["boot_lo"] > 0 or b["boot_hi"] < 0))
            b["agree"] = ("both" if b["headline_sig"] and b["boot_sig"] else "MWU only" if b["headline_sig"]
                          else "bootstrap only" if b["boot_sig"] else "neither")
        rows += block
    return pd.DataFrame(rows)


def _welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column Welch t of a (n_a x bins) vs b (n_b x bins), NaN-aware."""
    ma, mb = np.nanmean(a, 0), np.nanmean(b, 0)
    va, vb = np.nanvar(a, 0, ddof=1), np.nanvar(b, 0, ddof=1)
    na, nb = np.sum(np.isfinite(a), 0), np.sum(np.isfinite(b), 0)
    with np.errstate(all="ignore"):
        t = (ma - mb) / np.sqrt(va / na + vb / nb)
    return np.where((na >= 2) & (nb >= 2), t, np.nan)


def _clusters(t: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """(start, stop, mass) of contiguous runs with |t| > thr and a consistent sign."""
    out, i, n = [], 0, len(t)
    while i < n:
        if np.isfinite(t[i]) and abs(t[i]) > thr:
            s, j = np.sign(t[i]), i
            while j < n and np.isfinite(t[j]) and abs(t[j]) > thr and np.sign(t[j]) == s:
                j += 1
            out.append((i, j, float(np.sum(np.abs(t[i:j])))))
            i = j
        else:
            i += 1
    return out


def cluster_permutation(curves: np.ndarray, is_post: np.ndarray, animal: np.ndarray, n_perm: int = 1000,
                        seed: int = 0) -> np.ndarray:
    """Boolean mask over bins: clusters where post-epoch session curves differ from pre-session curves (permutation
    p < ALPHA). ``curves`` = sessions x bins (each already a change from its animal's own pre mean); labels are permuted
    within animal."""
    from scipy.stats import t as tdist
    curves = np.asarray(curves, float)
    is_post = np.asarray(is_post, bool)
    mask = np.zeros(curves.shape[1], bool)
    if is_post.sum() < 2 or (~is_post).sum() < 2:
        return mask
    thr = float(tdist.ppf(1 - ALPHA / 2, max(int(is_post.sum() + (~is_post).sum() - 2), 1)))
    obs = _clusters(_welch_t(curves[is_post], curves[~is_post]), thr)
    if not obs:
        return mask
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(animal == a) for a in np.unique(animal)]
    null = np.zeros(n_perm)
    lab = is_post.copy()
    for k in range(n_perm):
        for g in groups:
            lab[g] = rng.permutation(is_post[g])
        cl = _clusters(_welch_t(curves[lab], curves[~lab]), thr)
        null[k] = max((c[2] for c in cl), default=0.0)
    for s, e, mass in obs:
        if (np.sum(null >= mass) + 1) / (n_perm + 1) < ALPHA:
            mask[s:e] = True
    return mask


def within_animal_tests(P: pd.DataFrame, metrics) -> pd.DataFrame:
    """Per animal x metric x position x post epoch: two-sided Mann-Whitney of that animal's post-epoch sessions vs its own
    pre sessions; BH-FDR within each metric (all animals x positions x epochs). Columns: animal, metric, pos, epoch, p,
    q_fdr, sig (q < ALPHA)."""
    from scipy.stats import mannwhitneyu
    rows = []
    for m in metrics:
        if m not in P:
            continue
        block = []
        for (a, pos), g in P.groupby(["animal", "pos"], sort=True):
            pre = g.loc[g.epoch == "pre", m].dropna().to_numpy()
            for e in ("acute", "subacute", "chronic"):
                post = g.loc[g.epoch == e, m].dropna().to_numpy()
                p = (float(mannwhitneyu(post, pre, alternative="two-sided").pvalue)
                     if len(post) >= 2 and len(pre) >= 2 else np.nan)
                block.append({"animal": a, "metric": m, "pos": pos, "epoch": e, "n_post": len(post), "n_pre": len(pre),
                              "p": p})
        q = bh_fdr([b["p"] for b in block])
        for b, qq in zip(block, q):
            b["q_fdr"] = float(qq)
            b["sig"] = bool(np.isfinite(qq) and qq < ALPHA)
        rows += block
    return pd.DataFrame(rows, columns=["animal", "metric", "pos", "epoch", "n_post", "n_pre", "p", "q_fdr", "sig"])
