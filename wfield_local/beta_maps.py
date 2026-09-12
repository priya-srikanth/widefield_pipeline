"""WHERE the position code lives in cortex, and where it moves -- Haufe-transformed decoder maps.

Priya, 2026-09-12, after the representational families had answered "moved toward WHAT" and none of
them could answer "moved WHERE": "Could we do a beta weights mapping, as was done in this paper?"
(Musall et al., Nat Neurosci 2022) -- logistic decoders on the temporal component matrix SVT, with
beta convolved with the spatial component matrix U to make cortical maps.

THE DESIGN IS THAT PAPER'S IDEA WITH THREE CHANGES, EACH MEASURED RATHER THAN ASSUMED.

1. FIT ON SVT, NOT ON LocaNMF -- theirs, and it is right. `U @ beta` is then a genuine 540 x 640
   pixel map at full spatial resolution, with no component-space intermediary. It also costs
   nothing: SVT beats LocaNMF features by 0.06-0.09 balanced accuracy (0.915/0.918/0.952/0.880
   against 0.845/0.830/0.861/0.801 on PS94's pre-stroke sessions), because LocaNMF is a constrained
   re-description of the same rank-100 data and the constraints cost a little.

2. L2, NOT L1. The paper uses L1 and it is the wrong choice HERE, because we are not doing feature
   selection -- we are making a map, and the map has to reproduce. Measured on PS94's eleven
   curated pre-stroke sessions, far-contralateral:

                              same block (<=10d)   across the Jun/Aug gap   split-half of the mean
       L1                           0.315                  0.235                    0.688
       L2                           0.357                  0.271                    0.715
       L2 + HAUFE                   0.853                  0.709                    0.960

   L1 chooses among correlated predictors and WHICH one it chooses is free to change between days.
   The sparsity buys nothing and costs reproducibility.

3. HAUFE-TRANSFORM THE WEIGHTS, which the paper does not, and which is the single biggest factor in
   that table -- single-session maps go from r = 0.32 to 0.85.

WHY THE TRANSFORM MATTERS, and it is not a detail (Haufe et al. 2014, NeuroImage). A decoder weight
vector is a FILTER: part of its job is CANCELLING correlated noise, so a channel carrying no signal
at all can take a large weight purely because subtracting it cleans up a channel that does -- a
"suppressor". A beta map can therefore be large where there is no signal and small where signal is
strong but redundant. Neither is a thing to point at on a cortical map. The transform

    A = Cov(X) @ beta            A[i] = cov(channel i, decoder output)

converts the filter into a PATTERN: "does this channel actually carry the signal the decoder
extracted", which IS the anatomical question.

    READ THE PATTERN FOR "WHERE IS THE SIGNAL". READ THE FILTER FOR "WHAT DOES THE DECODER USE".
    They are different questions and this module returns the pattern. A pattern map is NOT evidence
    that those pixels are necessary for decoding; that claim needs the filter.

RELIABILITY IS PART OF THE RESULT, not a caveat, because it differs sharply by epoch. Split-half of
the epoch-mean map (PS94 / PS92, far-contralateral, post-cue):

    pre        0.960 / 0.920          subacute   0.924 / 0.856
    acute      0.531 / 0.667          <- THE WEAK LINK, and the epoch the story is about

A post-minus-pre difference inherits the WORSE side, so an acute difference map carries ~0.53-0.67
reliability and not the pre-stroke 0.96. Two causes, confounded: only four acute sessions, and the
code is weakest acutely (frozen balanced accuracy 0.52), so there is less signal for the pattern to
reflect. PS94's individual acute maps agree at r = 0.161 against PS92's 0.661 at the same n, so it
is not only sample size. Every figure built on this must print the per-epoch reliability.

BALANCING. The `working` class is uniform over positions by construction -- 16.1-17.2% at every
position, every animal -- because the scheduler presents positions evenly and `working` recovers the
full schedule. So pre-cue and post-cue need NO balancing. The `lick` class does: far-contralateral
runs 9.2-14.2% against ~19% for near positions, and that skew IS the deficit, so an unbalanced
lick-aligned fit would map how often each spout was attempted rather than what cortex did.

U IS SESSION-SPECIFIC AND BETA IS THEREFORE NOT COMPARABLE ACROSS SESSIONS. Only `U @ A` is -- the
pixel map, on the shared Allen grid. Every average and every difference in this module happens AFTER
the map is formed, never on the coefficients. Getting that backwards would silently compare two
different bases.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

#: Regularisation, matching the position decoder's measured choice (DECISIONS, 2026-08-11: C=0.5 is
#: the argmax over 10 sessions and the optimum is flat over C in [0.1, 1.0]).
C_PENALTY = 0.5

#: Folds for the within-session fit. Grouped by the scheduler's position blocks, as everywhere here.
N_SPLITS = 5

#: Map shape on the shared Allen grid.
MAP_SHAPE = (540, 640)


def _balanced_index(y, rng):
    """Row indices with every class down-sampled to the rarest -- for the LICK arm only.

    Returns all rows unchanged when the classes are already within 10% of each other, so the
    pre-cue and post-cue arms (16.1-17.2% per position) are untouched and their numbers do not move
    for a correction they do not need.
    """
    y = np.asarray(y)
    cls, cnt = np.unique(y, return_counts=True)
    if len(cls) < 2 or cnt.min() / cnt.max() > 0.9:
        return np.arange(len(y))
    keep = []
    for c in cls:
        idx = np.flatnonzero(y == c)
        keep.append(rng.choice(idx, size=int(cnt.min()), replace=False))
    return np.sort(np.concatenate(keep))


def haufe_map(session, target, align, *, post_s=2.0, balance=False, seed=0,
              c_penalty=C_PENALTY, n_splits=N_SPLITS, filter_map=False):
    """``(540, 640)`` cortical map for one position in one session, or None.

    ``target`` is the numeric position code. ``filter_map=True`` returns the raw decoder FILTER
    instead of the Haufe pattern -- "what the decoder uses" rather than "where the signal is". The
    default is the pattern; see the module docstring for why they answer different questions.
    """
    from wfield_local import joint_basis
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached

    u, v = joint_basis._load_session(session["mc"])
    X, y, g, *_ = trial_features_cached(
        session, _args("locanmf", align, post_s), signal=np.asarray(v),
        feat_region=np.arange(v.shape[0]), signal_key=f"svt:rank{v.shape[0]}")
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
    if balance:
        keep = _balanced_index(y, np.random.default_rng(seed))
        X, y, g = X[keep], y[keep], g[keep]
    K = u.shape[1]
    if X.shape[1] % K:
        raise ValueError(f"{X.shape[1]} features is not a multiple of {K} SVT components")
    n_bins = X.shape[1] // K
    k = min(n_splits, len(set(g.tolist())))
    if k < 2 or target not in set(y.tolist()):
        return None
    acc, n = np.zeros(X.shape[1]), 0
    for tr, _te in GroupKFold(n_splits=k).split(X, y, groups=g):
        if target not in set(y[tr].tolist()) or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=c_penalty, max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        cl = list(m.classes_)
        if target not in cl:
            continue
        # UNDO THE SCALER so the coefficients live in FEATURE space; `U` multiplies feature-space
        # coefficients, and a map built from standardised ones is scaled by 1/sd per component.
        b = np.asarray(m.coef_)[cl.index(target)] / np.where(sc.scale_ > 0, sc.scale_, 1.0)
        if not filter_map:
            Xc = X[tr] - X[tr].mean(0)
            b = (Xc.T @ (Xc @ b)) / max(len(tr) - 1, 1)     # Haufe: A = Cov(X) beta
        acc += b
        n += 1
    if not n:
        return None
    # BINS AVERAGED into one map. The per-bin maps are the code's trajectory WITHIN the window and
    # are worth having separately; this is the summary, and averaging is right because the bins
    # share a spatial basis.
    coef = (acc / n).reshape(n_bins, K).mean(0)
    return (u @ coef).reshape(MAP_SHAPE)


def session_maps(session, align, *, post_s=2.0, balance=False, seed=0,
                 c_penalty=C_PENALTY, n_splits=N_SPLITS, filter_map=False):
    """``{position name: (540, 640) map}`` for one session -- ALL positions, ONE load.

    `haufe_map` loads the session's U and SVT on every call, which is the expensive part (~100 MB
    each); calling it six times per session multiplies that by six for no reason, and a cohort
    render is ~90 sessions. This loads once, builds the trial features once, and refits per
    position -- the fits themselves are cheap next to the I/O.
    """
    from wfield_local import joint_basis
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached

    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    u, v = joint_basis._load_session(session["mc"])
    X, y, g, *_ = trial_features_cached(
        session, _args("locanmf", align, post_s), signal=np.asarray(v),
        feat_region=np.arange(v.shape[0]), signal_key=f"svt:rank{v.shape[0]}")
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
    if balance:
        keep = _balanced_index(y, np.random.default_rng(seed))
        X, y, g = X[keep], y[keep], g[keep]
    K = u.shape[1]
    if X.shape[1] % K:
        raise ValueError(f"{X.shape[1]} features is not a multiple of {K} SVT components")
    n_bins = X.shape[1] // K
    k = min(n_splits, len(set(g.tolist())))
    if k < 2:
        return {}
    # ONE PASS OVER THE FOLDS for all six positions: the fit is multinomial, so every position's
    # coefficients come out of the SAME model. Refitting per position would give six identical
    # models and six times the cost.
    acc = {q: np.zeros(X.shape[1]) for q in CONF_LABELS}
    n = {q: 0 for q in CONF_LABELS}
    for tr, _te in GroupKFold(n_splits=k).split(X, y, groups=g):
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=c_penalty, max_iter=4000).fit(sc.transform(X[tr]), y[tr])
        cl = list(m.classes_)
        Xc = X[tr] - X[tr].mean(0) if not filter_map else None
        inv = np.where(sc.scale_ > 0, sc.scale_, 1.0)
        for q in CONF_LABELS:
            t = code_of.get(q)
            if t is None or t not in cl:
                continue
            b = np.asarray(m.coef_)[cl.index(t)] / inv
            if not filter_map:
                b = (Xc.T @ (Xc @ b)) / max(len(tr) - 1, 1)
            acc[q] += b
            n[q] += 1
    out = {}
    for q in CONF_LABELS:
        if n[q]:
            coef = (acc[q] / n[q]).reshape(n_bins, K).mean(0)
            out[q] = (u @ coef).reshape(MAP_SHAPE)
    return out


def atlas_edges(session=None):
    """Allen region boundaries on the shared grid, for overlaying on these maps.

    THE MAPS ARE ALL ON ONE GRID -- `joint_basis._load_session` returns the affine8v1
    Allen-aligned `U_atlas`, which is why a map from PS92 and a map from PS95 can be averaged at
    all -- so ONE atlas serves every panel and the first session that has one is as good as any.
    Returns None rather than raising if no session carries the atlas: outlines are a reading aid,
    and a missing one should cost the outlines, not the figure.
    """
    import glob

    from wfield_local.atlas_overlay import region_edges
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    for s in ([session] if session is not None else SESSIONS):
        ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
        if not ad:
            continue
        f = f"{ad[0]}/allen_area_atlas_native_grid.npy"
        try:
            return region_edges(np.load(f))
        except Exception as ex:                                        # noqa: BLE001
            # SAID OUT LOUD. A silently skipped atlas costs every panel its CCF outlines, and a
            # cortical map without them is not readable as anatomy -- the figure would still be
            # produced, which is exactly how this would go unnoticed.
            print(f"  !! atlas edges from {s['label']}: {type(ex).__name__} {str(ex)[:70]}",
                  flush=True)
            continue
    return None


def map_corr(a, b):
    """Pearson r between two pixel maps, over the pixels finite in both."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or not np.std(a[ok]) or not np.std(b[ok]):
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def split_half_reliability(maps, n_draw=20, seed=0):
    """Median split-half r of a set of session maps -- the CEILING their mean can reach.

    THE NUMBER THAT DECIDES WHETHER A DIFFERENCE MAP MEANS ANYTHING, and it is not constant across
    epochs: 0.92-0.96 pre-stroke against 0.53-0.67 acutely. A post-minus-pre difference inherits the
    worse side, so this is reported per epoch and printed on the figure rather than assumed.
    """
    labs = sorted(maps)
    if len(labs) < 4:
        return None
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_draw):
        p = list(rng.permutation(labs))
        h1 = np.mean([maps[x] for x in p[:len(p) // 2]], axis=0)
        h2 = np.mean([maps[x] for x in p[len(p) // 2:]], axis=0)
        r = map_corr(h1, h2)
        if np.isfinite(r):
            rs.append(r)
    return float(np.median(rs)) if rs else None


@lru_cache(maxsize=8)
def maps_by_epoch(align, variant, post_s=2.0):
    """``{animal: {epoch: {position: [per-session maps]}}}`` plus per-epoch reliability.

    CACHED because it fits a decoder per (session, position) and loads each session's U and SVT --
    the expensive part -- and every figure built on this reads the same store.
    """
    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    balance = (variant == "lick")
    out, rel = {}, {}
    for an in ANIMALS:
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        per = {}
        for s in [x for x in SESSIONS if x["label"] in want]:
            d = _day(an, s["label"].split("_")[-1])
            if d is None:
                continue
            e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
            if e is None:
                continue
            try:
                got = session_maps(s, align, post_s=post_s, balance=balance)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! beta-map {s['label']}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
                continue
            for q, m in got.items():
                per.setdefault(e, {}).setdefault(q, {})[s["label"]] = m
        if per:
            out[an] = per
            rel[an] = {e: {q: split_half_reliability(v) for q, v in by_q.items()}
                       for e, by_q in per.items()}
    return out, rel
