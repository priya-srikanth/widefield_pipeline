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

ONE-VS-REST MAKES THE SIX POSITIONS NON-INDEPENDENT, AND THAT LIMITS WHAT A SINGLE ROW MEANS.
Priya, 2026-09-12: "in acute there may be less ss-ul/ll activity in far-center trials, which makes
the near ipsi acute trial map look as though there is a relative *increase* in ss-ul/ll activity
compared to pre-stroke." Correct, and it is structural rather than incidental.

`A[i] = cov(pixel i, decoder output)` is computed on TRIAL-MEAN-CENTRED data, so the reference is
the average over all six positions in that session. That is not a baseline anyone chose -- it is
what covariance means -- but it has a consequence: if one position loses drive, THE REFERENCE FALLS,
and every other position's map gains an apparent increase it did not earn. A loss at one position
propagates with opposite sign into the other five.

WHAT THAT INVALIDATES HERE. The acute amplitude "increases" -- near middle 1.53 post-cue, 2.82
post-lick, 3.30 pre-cue, far ipsi 1.09 -- cannot be read as those positions gaining anything. They
are consistent with being the shadow of far-middle (0.48) and far-contralateral (0.47) losing
amplitude. The FALLS are the safer half of the figure: the mechanism works against them, not for
them. Do not quote a row that goes UP.

THE FIX IS A PER-POSITION REFERENCE, which makes the six maps independent, and it is not this
module's quantity. `framemap_event_maps` already writes, per session and PER POSITION, a
`post-cue mean - pre-cue mean` map in Allen pixels (123 `*_spout_positions_1s_pre_post_delta_maps.npz`
on the share). That reference is WITHIN TRIAL, so far-contralateral's loss cannot leak into
near-ipsilateral's map. Build that arm before drawing conclusions about any position other than the
two that fall.

RELIABILITY IS PART OF THE RESULT, not a caveat, because it differs sharply by epoch. Split-half of
the epoch-mean map (PS94 / PS92, far-contralateral, post-cue):

    pre        0.960 / 0.920          subacute   0.924 / 0.856
    acute      0.531 / 0.667          <- THE WEAK LINK, and the epoch the story is about

A post-minus-pre difference inherits the WORSE side, so an acute difference map carries ~0.53-0.67
reliability and not the pre-stroke 0.96. Two causes, confounded: only four acute sessions, and the
code is weakest acutely (frozen balanced accuracy 0.52), so there is less signal for the pattern to
reflect. PS94's individual acute maps agree at r = 0.161 against PS92's 0.661 at the same n, so it
is not only sample size. Every figure built on this must print the per-epoch reliability.

A BUG FOUND AND FIXED 2026-09-12 -- kept as the record, because the symptom was subtle.

    EVERY ARM WAS FITTED ON LICKING TRIALS ONLY, including the two labelled `working`.

`session_maps` takes `X, y, g` from `trial_features_cached`, and those are the ENGAGED (licking)
trials: the no-lick trials that make the `working` class uniform come back separately as
`Xn, yn` (see `_trial_features`'s `base_out`) and are discarded here. So the `variant` argument
currently selects only whether class weighting is applied, not which trials are used.

WHAT IT COSTS. The reasoning below -- that pre-cue and post-cue need no balancing because `working`
is uniform at 16.1-17.2% per position -- DOES NOT APPLY to what is actually being fitted. Measured
far-contralateral trials per session in the pre-cue arm:

    PS92 acute 0-5 (med 4)     PS93 acute 4-12 (med 9)
    PS94 acute 0-8 (med 1)     PS95 acute 1                against 66-119 at the best position

That is the deficit itself -- the animal does not lick far-contralateral acutely -- and it is why
`MIN_TRIALS_PER_CLASS` refuses the far-contra acute cell in the pre-cue figure. The refusal is
correct; the labelling is not.

FIXED by stacking the no-lick arm (minus the terminal quit period, via the same
`engagement_gate` the rest of the deck uses) for `working`. The effect is confined to exactly where
it should be -- pre-stroke far-contra goes 87 -> 91 trials because a pre-stroke animal rarely
misses, while ACUTE far-contra goes 0 -> 105 and 0 -> 73, recovering a cell the floor had correctly
refused.

BALANCING, as designed (and correct once the above is fixed). The `working` class is uniform over
positions by construction -- 16.1-17.2% at every position, every animal -- because the scheduler
presents positions evenly and `working` recovers the full schedule. So pre-cue and post-cue would
need NO balancing. The `lick` class does: far-contralateral runs 9.2-14.2% against ~19% for near
positions, and that skew IS the deficit, so an unbalanced lick-aligned fit would map how often each
spout was attempted rather than what cortex did.

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

#: Fewest trials a POSITION needs before its own map is drawn. Applied per position, not per
#: session: a position the animal has stopped attempting should lose ITS map, not everyone else's.
#:
#: THE ACUTE LICK ARM IS WHY THIS EXISTS (Priya, 2026-09-12: "the far R n may be low"). Acutely the
#: animal barely licks far-contralateral -- PS94 and PS92 fall to FIVE far-contra lick trials while
#: the other positions still have 70 -- and a six-class decoder cannot estimate a pattern for a
#: class it has seen five times. `Cov(X) @ beta` on that is a near-degenerate covariance and
#: produces a large smooth low-rank blob: far-contra read 4.31x its pre-stroke amplitude in the
#: post-lick arm, where the post-cue arm gave 0.47 for the same position. Not biology.
#:
#: TWENTY, MEASURED (Priya, 2026-09-12: "what is the right floor in terms of trial n? 3? 5?").
#: Sub-sampling one position of a pre-stroke session down to n and correlating its map against the
#: same session's full-data map, median over 5 draws:
#:
#:      n            3     5     8    12    15    20    30    50
#:      PS94_0606  0.66  0.82  0.70  0.86  0.82  0.94  0.94  0.98
#:      PS94_0607  0.73  0.77  0.86  0.86  0.95  0.96  0.96  0.99
#:      PS92_0606  0.69 -0.08  0.65  0.79  0.68  0.81  0.94    --
#:
#: Below ~20 it is ERRATIC rather than merely noisy -- PS92 at n=5 returns r = -0.08, a map
#: ANTI-correlated with its own full-data version, because the estimate is dominated by which
#: particular trials were drawn. Twenty is the first n where all three clear 0.81; thirty is where
#: they are consistently >= 0.94. And `class_weight="balanced"` UP-weights a rare class, so a
#: five-trial position is more influential rather than less, which argues for the floor not against.
MIN_TRIALS_PER_CLASS = 20

#: Percentile of the IN-MASK between-animal `se` used to floor the t-statistic's denominator.
#: Not 5, and the difference is measured rather than chosen (2026-09-12): with four animals the
#: in-mask `se` distribution has a long thin left tail, and a 5th-percentile floor still lets
#: pixels where four animals happen to agree produce a t large enough to seed a cluster. 25 is a
#: variance-regularisation floor in the spirit of SAM's s0 -- it changes nothing for the pixels
#: that carry a real effect (their `se` is well above it) and removes the ones that are significant
#: only because their denominator vanished.
SE_FLOOR_PCT = 25.0

#: Two-tailed p for the CLUSTER-FORMING threshold, converted to a t against the between-animal df
#: at call time. NOT a hard-coded t (it was 2.0, which at df=3 is p=0.14): measured 2026-09-12 on
#: the acute-minus-pre evoked maps, t>2 put 18.8% of the brain above threshold for near-ipsi -- a
#: position with no detectable change -- against the ~14% a df=3 null predicts. The threshold was
#: admitting noise wholesale and then relying on the cluster step to sort it out. At the df=3 value
#: (t=3.18) the near positions fall to 1.0-7.1% suprathreshold while far-contra stays at 85.7%.
CLUSTER_FORMING_P = 0.05


def _balanced_index(y, rng):
    """Row indices with every class down-sampled to the rarest. KEPT FOR REFERENCE, NOT USED.

    THIS IS THE WRONG WAY TO BALANCE HERE and the numbers say so plainly (Priya, 2026-09-12: "why
    are we decreasing far R to 5 after balancing? seems like we are throwing away valuable data").
    Down-sampling to the rarest class discards 325 of 355 acute post-lick trials to keep 30, and it
    punishes every position for one position's scarcity: far-contralateral having 5 trials drops the
    other five from ~70 each to 5 each, destroying five perfectly good maps to fix one.

    `class_weight="balanced"` achieves the identical thing -- each class contributing equally to the
    loss, so the fit cannot be pulled by base rates -- while keeping every trial. The paper
    down-samples because it balances on TWO factors at once (choice AND correctness), which is
    awkward to express as weights, and because it has >= 250 trials so the cost is small. Neither
    applies here.
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


def _runs_to_blocks(y):
    """Block ids for the no-lick arm: a new block wherever the position changes."""
    y = np.asarray(y)
    return np.concatenate([[0], np.cumsum(y[1:] != y[:-1])]).astype(np.int64) if len(y) else         np.zeros(0, np.int64)


def _quit_mask(session, idx_e, idx_n, y, yn):
    """Which NO-LICK trials fall inside the terminal quit period -- the `working` class excludes them.

    Same gate the rest of the deck uses (`precue_engagement_states.engagement_gate`): a sated
    animal's late misses are DISENGAGEMENT, not a spatial deficit, and folding them into `working`
    would put the quit period back into a class defined to exclude it.
    """
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.precue_engagement_states import engagement_gate

    idx_e, idx_n = np.asarray(idx_e), np.asarray(idx_n)
    order = np.concatenate([idx_e, idx_n])
    responded = np.concatenate([np.ones(len(idx_e), bool), np.zeros(len(idx_n), bool)])
    pos = np.array([POSITION_NAMES.get(int(c), str(c)) for c in
                    np.concatenate([np.asarray(y), np.asarray(yn)])])
    o = np.argsort(order, kind="stable")
    ne = engagement_gate(order[o], responded[o], pos[o])
    by_idx = {int(k): bool(b) for k, b in zip(order[o], ne)}
    return np.array([by_idx.get(int(k), False) for k in idx_n], bool)


def session_maps(session, align, *, post_s=2.0, balance=False, seed=0, variant="lick",
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
    X, y, g, Xn, yn, _reg, idx_e, idx_n = trial_features_cached(
        session, _args("locanmf", align, post_s), signal=np.asarray(v),
        feat_region=np.arange(v.shape[0]), signal_key=f"svt:rank{v.shape[0]}",
        with_indices=True)
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
    if variant == "working" and len(yn):
        # THE `working` CLASS IS ENGAGED + MISS-WHILE-WORKING, and leaving the second half out is
        # the bug this call site had until 2026-09-12: `trial_features_cached` returns the engaged
        # trials FIRST and the no-lick trials separately, so taking only the first three returns
        # silently made every arm a LICK-trial map. The symptom was figure 14 refusing the
        # far-contralateral ACUTE cell -- 0-9 trials a session against 66-119 elsewhere, which is
        # the deficit, not a defect.
        keep_n = ~_quit_mask(session, idx_e, idx_n, y, yn)
        if keep_n.any():
            Xn = np.asarray(Xn)[keep_n]
            X = np.vstack([X, Xn])
            y = np.concatenate([y, np.asarray(yn)[keep_n]])
            # NO BLOCK IDS FOR THE NO-LICK ARM, by the same rule `_pooled_bundle` uses: a new block
            # wherever the position changes in that session's trial order. Coarser than the real
            # blocks, never finer, so the grouped CV cannot become too permissive.
            g = np.concatenate([g, _runs_to_blocks(np.asarray(yn)[keep_n]) + int(g.max()) + 1])
    # NO DOWN-SAMPLING. `balance` now selects CLASS WEIGHTING, which removes the base-rate pull
    # without discarding a single trial -- see `_balanced_index` for why the other way is wrong.
    cw = "balanced" if balance else None
    _cls, _cnt = np.unique(y, return_counts=True)
    if len(_cls) < 2:
        return {}, {}
    K = u.shape[1]
    if X.shape[1] % K:
        raise ValueError(f"{X.shape[1]} features is not a multiple of {K} SVT components")
    n_bins = X.shape[1] // K
    k = min(n_splits, len(set(g.tolist())))
    if k < 2:
        return {}, {}
    # ONE PASS OVER THE FOLDS for all six positions: the fit is multinomial, so every position's
    # coefficients come out of the SAME model. Refitting per position would give six identical
    # models and six times the cost.
    acc = {q: np.zeros(X.shape[1]) for q in CONF_LABELS}
    n = {q: 0 for q in CONF_LABELS}
    for tr, _te in GroupKFold(n_splits=k).split(X, y, groups=g):
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=c_penalty, max_iter=4000, class_weight=cw).fit(
            sc.transform(X[tr]), y[tr])
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
    out, used = {}, {}
    by_code = {int(c): q for q in CONF_LABELS if (c := code_of.get(q)) is not None}
    counts = {by_code[int(c)]: int(v) for c, v in zip(_cls, _cnt) if int(c) in by_code}
    for q in CONF_LABELS:
        # PER POSITION. A position the animal has stopped attempting loses its OWN map and takes
        # nothing else with it -- the failure the down-sampling version inflicted on all six.
        if n[q] and counts.get(q, 0) >= MIN_TRIALS_PER_CLASS:
            coef = (acc[q] / n[q]).reshape(n_bins, K).mean(0)
            out[q] = (u @ coef).reshape(MAP_SHAPE)
            used[q] = counts.get(q, 0)
    return out, used


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


@lru_cache(maxsize=1)
def brain_mask():
    """The Allen brain mask on the shared grid -- the pixels a statistic is ALLOWED to exist in.

    WHY THIS IS NOT OPTIONAL, and the bug it exists to prevent (found 2026-09-12). The per-pixel
    statistic is `|mean| / se` where `se` is the between-animal SD over n=4. OUTSIDE THE BRAIN every
    animal's map is ~0, so the mean is ~0 AND the SD is ~0, and their ratio is whatever the
    denominator floor allows -- unbounded. `cluster_permutation` duly found enormous "significant"
    clusters tracing the image border, and because those same clusters appear in every permuted
    draw they inflated the null's cluster-mass threshold far enough to suppress every real interior
    effect. One artefact, manufacturing a false positive and hiding the true ones at once.

    `allen_brain_mask_native_grid.npy` sits beside `allen_area_atlas_native_grid.npy` in each
    session's `allen_aligned_affine8v1` -- 540 x 640 uint8, 207,213 of 345,600 pixels non-zero, i.e.
    40% of the frame is not brain. ONE MASK SERVES EVERY PANEL for the same reason one atlas does:
    these maps are all on the shared Allen grid, which is what makes a PS92 map and a PS95 map
    averageable in the first place.

    Returns None if no session carries one. Unlike `atlas_edges`, whose absence costs only a
    reading aid, a missing mask must STOP the test rather than let it fall back -- see
    `cluster_permutation`, which raises.
    """
    import glob

    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    for s in SESSIONS:
        ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
        if not ad:
            continue
        try:
            m = np.load(f"{ad[0]}/allen_brain_mask_native_grid.npy").astype(bool)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! brain mask from {s['label']}: {type(ex).__name__} {str(ex)[:70]}",
                  flush=True)
            continue
        if m.shape == MAP_SHAPE and m.any():
            return m
    return None


def map_corr(a, b):
    """Pearson r between two pixel maps, over the pixels finite in both."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or not np.std(a[ok]) or not np.std(b[ok]):
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def cluster_permutation(pre_by_animal, post_by_animal, *, n_perm=500, t_thresh=None, seed=0,
                        se_floor_pct=SE_FLOOR_PCT, mask=None):
    """Boolean mask: pixels in a cluster larger than 95% of clusters obtainable by relabelling.

    THE TEST THE DIFFERENCE COLUMN NEEDS. 540 x 640 is 345,600 pixels, so an uncorrected per-pixel
    threshold means nothing -- thousands of pixels pass at p<0.05 by construction. The field
    standard is a cluster-based permutation test and it is what this is: threshold the per-pixel t
    map, sum |t| within each connected cluster, and compare the largest observed cluster mass
    against the null distribution of largest cluster masses obtained by SHUFFLING THE EPOCH LABELS.

    LABELS ARE SHUFFLED WITHIN ANIMAL, never across. Each animal has its own baseline map and its
    own number of sessions; pooling the relabelling would let a between-animal difference stand in
    for a between-epoch one, which is the difference the test is supposed to be measuring.

    ``pre_by_animal`` / ``post_by_animal`` are ``{animal: [session maps]}``. The statistic is the
    same one the figure draws -- each animal's epoch mean, then the mean over animals -- so the
    test is testing the picture rather than a convenient relative of it.

    EVERYTHING HAPPENS INSIDE THE BRAIN MASK, and that is a FIX rather than a refinement
    (2026-09-12; Priya, of the first version's output: "this contour of significance looks like
    total artifact ... hard to believe nothing else is significant?"). The old statistic was
    `|mean| / (se + 1e-12)`, and outside the brain both the mean and the between-animal SD are ~0,
    so `t` was unbounded on 138,387 empty pixels. Clusters formed on the image border -- and
    because they formed in every PERMUTED draw too, they inflated the null's cluster-mass threshold
    far enough to suppress every interior effect. The artefact manufactured a false positive and
    hid the true ones with the same mechanism. `t` is now identically zero outside `brain_mask()`,
    so no cluster can form there, in the observed map or in any null draw.

    THE DENOMINATOR IS FLOORED AT A PERCENTILE OF THE IN-MASK `se`, not at 1e-12. With four animals
    an interior pixel where all four happen to agree gets an `se` near zero and the same unbounded
    `t`, which is the border bug wearing a different hat. The floor is recomputed FOR EACH DRAW
    from that draw's own `se` distribution, so it is a property of the relabelling rather than a
    constant imported from the observed data -- which is what keeps the null exchangeable.

    THE CLUSTER-FORMING THRESHOLD COMES FROM THE df, not from habit. `t_thresh=None` resolves to
    the two-tailed `CLUSTER_FORMING_P` point of a t distribution on (n_animals - 1) df -- 3.18 at
    n=4. The previous hard-coded 2.0 is p=0.14 there, and it showed: on the acute-minus-pre evoked
    maps it put 18.8% of the brain above threshold for NEAR-IPSILATERAL, a position with no
    detectable change, against the ~14% a df=3 null predicts. At 3.18 the near positions drop to
    1.0-7.1% while far-contralateral holds at 85.7%.

    POSITIVE AND NEGATIVE CLUSTERS ARE LABELLED SEPARATELY. Thresholding `|t|` lets a region that
    went UP and a region that went DOWN merge into a single cluster wherever they touch, and the
    merged mass is then compared against a null built the same way -- so a large increase can carry
    an adjacent decrease over the line with it. Each polarity is labelled on its own and both feed
    one shared null of maximum cluster mass, which is the standard two-tailed construction.

    A CLUSTER FILLING THE MASK IS AN ANSWER, not a failure of the test. Far-contralateral's acute
    change is 85.7% of the brain in one cluster, and that is what a position the animal has stopped
    attempting looks like: the whole task-evoked response goes, rather than a piece of it. The
    informative comparison is against the near positions on the same figure, whose clusters are
    focal and small.

    NO PARAMETRIC ASSUMPTION IS MADE about the CLUSTER statistic. With 11 pre-stroke and ~5 acute
    sessions there are C(16,5) = 4,368 distinct relabellings per animal, so 500 draws sample the
    null honestly. The t distribution enters only in choosing where to cut, which is a convention
    every cluster test needs and which the permutation then corrects around.
    """
    from scipy import stats
    from scipy import ndimage

    animals = sorted(set(pre_by_animal) & set(post_by_animal))
    if not animals:
        return None
    if mask is None:
        mask = brain_mask()
    # RAISES RATHER THAN FALLING BACK. Running this test on the whole frame is precisely the bug
    # it was written to fix, and a fallback that silently reinstates it would be indistinguishable
    # from the fix working -- the caller prints the exception and the panel simply gets no contour,
    # which is the honest outcome when the mask is unavailable.
    if mask is None or np.asarray(mask).shape != MAP_SHAPE or not np.any(mask):
        raise ValueError("cluster_permutation needs the Allen brain mask "
                         "(allen_brain_mask_native_grid.npy); refusing to test off-brain pixels")
    mask = np.asarray(mask, bool)
    if t_thresh is None:
        # df = n_animals - 1, and with a single animal there is no between-animal df at all: fall
        # back to the same number n=2 would give rather than to an unbounded one.
        t_thresh = float(stats.t.ppf(1 - CLUSTER_FORMING_P / 2, max(len(animals) - 1, 1)))
    rng = np.random.default_rng(seed)

    def _stat(assign):
        """Mean-over-animals of (post mean - pre mean), and its per-pixel t INSIDE THE MASK.

        `t` is identically 0 outside the mask, so `ndimage.label(t > thresh)` cannot form an
        off-brain cluster and off-brain pixels contribute nothing to any cluster's mass.
        """
        d = []
        for an in animals:
            allm = list(pre_by_animal[an]) + list(post_by_animal[an])
            idx = assign[an]
            a = np.mean([allm[i] for i in range(len(allm)) if not idx[i]], axis=0)
            b = np.mean([allm[i] for i in range(len(allm)) if idx[i]], axis=0)
            d.append(b - a)
        d = np.stack(d)
        m = d.mean(0)
        if len(animals) < 2:
            # ONE ANIMAL HAS NO BETWEEN-ANIMAL SE, so the spatial SD stands in for it. Still
            # in-mask: the spatial SD over the whole frame is dominated by the empty 40%.
            se = np.full(MAP_SHAPE, float(np.nanstd(m[mask])), float)
        else:
            se = d.std(0, ddof=1) / np.sqrt(len(animals))
        t = np.zeros(MAP_SHAPE, float)
        inm = se[mask]
        floor = float(np.nanpercentile(inm, se_floor_pct)) if np.isfinite(inm).any() else 0.0
        if not np.isfinite(floor) or floor <= 0:
            # Degenerate draw (every animal identical): no evidence of anything, not infinite
            # evidence of everything.
            return m, t
        # SIGNED, so the polarities can be labelled apart below.
        t[mask] = m[mask] / np.maximum(inm, floor)
        return m, t

    def _clusters(t):
        """``[(label_image, index, mass)]`` for each polarity -- masses of every cluster in `t`."""
        out = []
        for sgn in (1.0, -1.0):
            lab, n_lab = ndimage.label((sgn * t) > t_thresh)
            if n_lab:
                idx = np.arange(1, n_lab + 1)
                out.append((lab, idx, ndimage.sum(np.abs(t), lab, index=idx)))
        return out

    true_assign = {an: np.array([False] * len(pre_by_animal[an]) + [True] * len(post_by_animal[an]))
                   for an in animals}
    _m, t = _stat(true_assign)
    observed = _clusters(t)
    if not observed:
        return np.zeros(MAP_SHAPE, bool)

    null = []
    for _ in range(n_perm):
        assign = {}
        for an in animals:
            n_all = len(pre_by_animal[an]) + len(post_by_animal[an])
            k = len(post_by_animal[an])
            v = np.zeros(n_all, bool)
            v[rng.choice(n_all, k, replace=False)] = True
            assign[an] = v
        _mm, tt = _stat(assign)
        # ONE null over BOTH polarities: the draw's largest cluster whichever way it points.
        null.append(max((float(np.max(mass)) for _l, _i, mass in _clusters(tt)), default=0.0))
    cut = float(np.percentile(null, 95))
    keep = np.zeros(MAP_SHAPE, bool)
    for lab, idx, mass in observed:
        for i, mm in zip(idx, mass):
            if mm > cut:
                keep |= (lab == i)
    return keep


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
    out, rel, n_out = {}, {}, {}
    for an in ANIMALS:
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        per, ntr = {}, {}
        for s in [x for x in SESSIONS if x["label"] in want]:
            d = _day(an, s["label"].split("_")[-1])
            if d is None:
                continue
            e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
            if e is None:
                continue
            try:
                got, used = session_maps(s, align, post_s=post_s, balance=balance,
                                         variant=variant)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! beta-map {s['label']}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
                continue
            if not got:
                print(f"  .. beta-map {s['label']} {e}: no position reached "
                      f"{MIN_TRIALS_PER_CLASS} trials", flush=True)
                continue
            for q, m in got.items():
                per.setdefault(e, {}).setdefault(q, {})[s["label"]] = m
                ntr.setdefault(e, {}).setdefault(q, {})[s["label"]] = used.get(q, 0)
        if per:
            out[an] = per
            n_out[an] = ntr
            rel[an] = {e: {q: split_half_reliability(v) for q, v in by_q.items()}
                       for e, by_q in per.items()}
    return out, rel, n_out
