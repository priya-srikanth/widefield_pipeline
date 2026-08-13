"""Which cortical regions carry PRE-CUE position information — stated only in ways the data support.

Two separate facts make the obvious answer ("look at the decoder weights") invalid here:

  * SUPPORT OVERLAP (LocaNMF): components share pixels (median footprint cosine 0.977), so the
    amplitude split within an overlapping group is arbitrary — see DECISIONS.md "Projecting NEW
    sessions". A per-component importance is not a fact about cortex.
  * ACTIVITY CORRELATION (ROIs and LocaNMF alike): the noise covariance of 66 Allen ROIs has a
    participation ratio of ~1.5 effective dimensions, one mode holding 80.7% of the variance. With
    predictors that collinear, regression weights are not identifiable — L2 spreads weight over
    correlated regions, L1 picks one arbitrarily — and a linear decoder routinely puts LARGE weights
    on regions carrying NO signal, purely to cancel shared noise (Haufe et al., NeuroImage 2014).

Overlap can be handled by grouping. Collinearity cannot be "fixed": it is a true statement that the
data do not identify a unique answer. So this module never ranks features by weight. It provides three
claims that survive both problems:

  ENCODING EV (safe by construction) — per region, how much of THAT region's own activity the position
    variable explains, cross-validated. Univariate per target, so neither problem applies. Several
    regions scoring high is a real redundancy result, not an artifact.

  SUFFICIENCY vs NECESSITY (reported as a pair, never alone) — accuracy from a region family ALONE
    versus the accuracy lost when it is REMOVED. With redundant regions each is misleading by itself:
    two regions carrying the same information both look sufficient and neither looks necessary. The
    GAP between them is the redundancy measurement, which is the quantity a stroke study actually
    wants — redundancy predicts what survives a lesion.

  HAUFE ACTIVATION MAPS — the decoder is one linear functional on the movie, ``w' pinv(A) M``. That
    functional is invariant to how amplitude is split among collinear components even though ``w`` is
    not, so mapping back to PIXELS gives a well-defined object where per-component weights do not.
    The Haufe transform ``a = Cov(X) w`` converts the extraction FILTER into an activation PATTERN —
    where the signal is, rather than which features the classifier used to cancel noise.

Everything here works identically for ``source='roi'`` and ``source='locanmf'``; run both. They will
not agree feature-by-feature (they cannot — different bases) but should agree on the regional story.
"""
from __future__ import annotations

import glob
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

# Allen acronym prefixes -> the families claims are made about. Deliberately coarse: a family must be
# big enough that "this family carries position information" is a statement the data can resolve.
FAMILIES = {
    "SSp":  ("SSp",),                 # primary somatosensory (incl. barrel/limb/mouth subfields)
    "SSs":  ("SSs",),                 # supplemental somatosensory
    # MOp and MOs are kept SEPARATE: for a maintained pre-cue plan they are different claims (MOs is
    # the premotor/planning area, MOp the executor), and pooling them would hide which one carries it.
    "MOp":  ("MOp",),                 # primary motor
    "MOs":  ("MOs",),                 # secondary/premotor
    "VIS":  ("VIS",),                 # visual areas
    "RSP":  ("RSP",),                 # retrosplenial
    "PTLp": ("PTLp", "VISa", "VISrl"),  # posterior parietal
    "ACA":  ("ACA",),                 # anterior cingulate
    "AUD":  ("AUD",),                 # auditory
}
CHANCE = 1.0 / len(DISPLAY_ORDER)
C_REG = 0.5          # matches locanmf_position_decoder (measured; see DECISIONS "Decoder regularization")


def region_names(session):
    """{allen_label:int -> acronym:str} for this session's atlas."""
    p = glob.glob(f"{session['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")
    if not p:
        return {}
    # the file is a LIST of [label, acronym] pairs, not an object -- same read as
    # locanmf_position_decoder, which iterates it directly
    return {int(k): v for k, v in json.load(open(p[0]))}


def family_columns(feat_reg, names):
    """{family -> column indices}. A feature belongs to a family if its Allen region acronym starts
    with one of the family's prefixes."""
    out = {}
    for fam, prefs in FAMILIES.items():
        cols = [i for i in range(len(feat_reg))
                if any(names.get(int(feat_reg[i]), "").startswith(p) for p in prefs)]
        if cols:
            out[fam] = np.array(cols)
    return out


def _folds(g, k=5):
    return GroupKFold(min(k, int(np.unique(g).size)))


def encoding_ev(X, y, g, k=5):
    """Cross-validated R^2 of predicting each FEATURE from position, plus a split-half noise ceiling.

    The model is just the per-position mean fitted on training blocks and applied to held-out blocks,
    which is the saturated position model — anything richer would fit the same 6 conditions. R^2 is
    computed against the test fold's own total variance and CAN be negative (a feature whose position
    means do not generalise), which is informative and is not clipped.

    ``ceiling`` is the Spearman-Brown-corrected split-half correlation of the 6-vector of position
    means, SQUARED so it is on the same scale as R^2. NB an earlier version of this idea in the RSA
    code was invalid because it used half the data without the SB correction; do not remove it.
    """
    pos = np.array(DISPLAY_ORDER)
    n, p = X.shape
    pred = np.full_like(X, np.nan, dtype=float)
    for tr, te in _folds(g, k).split(X, y, groups=g):
        gm = X[tr].mean(0)
        for c in pos:
            m = y[tr] == c
            mu = X[tr][m].mean(0) if m.sum() else gm
            pred[te[y[te] == c]] = mu
    ok = np.isfinite(pred).all(1)
    resid = ((X[ok] - pred[ok]) ** 2).sum(0)
    total = ((X[ok] - X[ok].mean(0)) ** 2).sum(0)
    ev = 1.0 - resid / np.maximum(total, 1e-30)

    ub = np.unique(g); half = set(ub[::2].tolist())
    mA = np.array([gi in half for gi in g]); mB = ~mA
    def _profile(m):
        return np.stack([X[m & (y == c)].mean(0) if (m & (y == c)).sum() else np.full(p, np.nan)
                         for c in pos])                       # (6, p)
    A_, B_ = _profile(mA), _profile(mB)
    ceil = np.full(p, np.nan)
    for j in range(p):
        a, b = A_[:, j], B_[:, j]
        if np.isfinite(a).all() and np.isfinite(b).all():
            a, b = a - a.mean(), b - b.mean()
            den = np.linalg.norm(a) * np.linalg.norm(b)
            if den > 1e-30:
                r = float(a @ b / den)
                sb = 2 * r / (1 + r) if r > -1 else np.nan    # Spearman-Brown: half-data -> full-data
                ceil[j] = np.clip(sb, -1, 1) ** 2
    return ev, ceil


def _decode(X, y, g, k=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C_REG))
    return float(accuracy_score(y, cross_val_predict(clf, X, y, cv=_folds(g, k), groups=g)))


def sufficiency_necessity(X, y, g, fam_cols, k=5):
    """{family -> {alone, without, full, redundancy}}.

    ``alone``  = accuracy using ONLY that family      (sufficiency)
    ``without``= accuracy with that family REMOVED    (necessity, as full - without)
    ``redundancy`` = alone - (full - without): how much the family's information is ALSO carried
    elsewhere. Large redundancy means both readings are individually misleading, which is exactly the
    case that must not be reported as a ranking."""
    full = _decode(X, y, g, k)
    out = {"_full": full, "_chance": CHANCE}
    for fam, cols in fam_cols.items():
        rest = np.setdiff1d(np.arange(X.shape[1]), cols)
        alone = _decode(X[:, cols], y, g, k) if len(cols) >= 2 else np.nan
        without = _decode(X[:, rest], y, g, k) if len(rest) >= 2 else np.nan
        drop = full - without
        out[fam] = {"n_features": int(len(cols)), "alone": alone, "without": without,
                    "necessity_drop": drop, "redundancy": alone - drop}
    return out


def haufe_patterns(X, y, g):
    """(nfeat, nclass) ACTIVATION patterns, one column per position, in the features' own units.

    Fit on all trials — this describes the model, it is not a performance estimate, so no CV is
    needed and using all data makes the covariance better determined. Weights are un-standardised
    before the transform, since the classifier is fit on z-scored features but the activation pattern
    must live in the data's units to be mapped to pixels.
    """
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=C_REG).fit(scaler.transform(X), y)
    W = clf.coef_ / scaler.scale_[None, :]              # (nclass, nfeat) in original units
    cov = np.cov(X, rowvar=False)
    A = cov @ W.T                                       # Haufe: activation = Cov(X) . w
    order = [int(np.where(clf.classes_ == c)[0][0]) for c in DISPLAY_ORDER]
    return A[:, order], [POSITION_NAMES[c] for c in DISPLAY_ORDER]


def pattern_to_pixels(pattern, session, source, basis=None):
    """Map an (nfeat, nclass) activation pattern into (H, W, nclass) Allen-space images.

    This is the step that makes the map well defined: per-feature values are basis-dependent, but the
    pixel image of the linear functional is not.
    """
    ad = glob.glob(f"{session['mc']}/wfield_local_results/allen_aligned_affine8v1")[0]
    atlas = np.load(f"{ad}/allen_area_atlas_native_grid.npy")
    mask = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool)
    H, W = atlas.shape
    out = np.full((H, W, pattern.shape[1]), np.nan)
    if source == "roi":
        labs = [l for l in np.unique(atlas) if l != 0 and ((atlas == l) & mask).sum() >= 20]
        for i, l in enumerate(labs):
            if i >= pattern.shape[0]:
                break
            out[(atlas == l) & mask] = pattern[i]
        return out
    A = np.nan_to_num(np.asarray(basis.A if basis is not None else
                                 np.load(f"{session['mc']}/locanmf_affine8v1_final/"
                                         f"{session['label']}_locanmf_A.npy"), dtype=np.float32))
    A = A.reshape(-1, A.shape[-1])
    img = (A @ pattern).reshape(H, W, -1)               # footprint-weighted sum = the pixel functional
    img[~mask] = np.nan
    return img


# --------------------------------------------------------------------------------------------------
# driver + figure
# --------------------------------------------------------------------------------------------------

def analyse_session(session, source, align="precue", fs=31.23):
    """All three analyses for one session. Returns None if the session is unusable."""
    from types import SimpleNamespace

    from wfield_local.locanmf_position_decoder import _trial_features

    args = SimpleNamespace(source=source, align=align, baseline="none", pre_s=1.0, post_s=2.0,
                           fs=fs, max_rt=2.0)
    X, y, g, _, _, feat_reg = _trial_features(session, args)
    if len(y) < 30 or len(np.unique(y)) < len(DISPLAY_ORDER):
        return None
    names = region_names(session)
    fam = family_columns(feat_reg, names)
    ev, ceil = encoding_ev(X, y, g)
    ev_fam = {f: {"ev": float(np.nanmean(ev[c])), "ceiling": float(np.nanmean(ceil[c])),
                  "n": int(len(c))} for f, c in fam.items()}
    pat, posnames = haufe_patterns(X, y, g)
    return {"label": session["label"], "source": source, "align": align, "n_trials": int(len(y)),
            "n_features": int(X.shape[1]), "ev_by_family": ev_fam, "ev_per_feature": ev.tolist(),
            "sufficiency": sufficiency_necessity(X, y, g, fam),
            "pattern": pat, "positions": posnames, "session": session}


def figure(per_session, out_png, title):
    """One figure per animal x basis: encoding EV, sufficiency-vs-necessity, and the six activation
    maps. The maps are averaged over sessions in Allen space, which is meaningful because the pixel
    functional -- unlike the per-feature weights -- is basis- and session-comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = [r for r in per_session if r]
    if not good:
        return None
    fams = [f for f in FAMILIES if any(f in r["ev_by_family"] for r in good)]
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.15], hspace=0.45, wspace=0.25)

    # --- encoding EV -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :3])
    x = np.arange(len(fams))
    ev = [np.nanmean([r["ev_by_family"][f]["ev"] for r in good if f in r["ev_by_family"]]) for f in fams]
    ce = [np.nanmean([r["ev_by_family"][f]["ceiling"] for r in good if f in r["ev_by_family"]]) for f in fams]
    ax.bar(x, ev, 0.62, color="#3b7dd8", label="position EV (block-CV $R^2$)")
    ax.plot(x, ce, "k_", ms=18, mew=2, label="split-half ceiling (SB-corrected)")
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("variance explained")
    ax.set_title("Encoding: how much of each region's own activity position explains\n"
                 "(univariate per region — unaffected by collinearity)", fontsize=9)
    ax.legend(fontsize=7)

    # --- sufficiency vs necessity ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, :3])
    alone = [np.nanmean([r["sufficiency"][f]["alone"] for r in good if f in r["sufficiency"]]) for f in fams]
    drop = [np.nanmean([r["sufficiency"][f]["necessity_drop"] for r in good if f in r["sufficiency"]]) for f in fams]
    full = float(np.nanmean([r["sufficiency"]["_full"] for r in good]))
    ax.bar(x - 0.2, alone, 0.4, color="#3b7dd8", label="alone (sufficiency)")
    ax.bar(x + 0.2, drop, 0.4, color="#d1495b", label="accuracy lost if removed (necessity)")
    ax.axhline(full, color="k", ls="--", lw=1, label=f"all features ({full:.2f})")
    ax.axhline(CHANCE, color="gray", ls=":", lw=1, label=f"chance ({CHANCE:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylabel("decode accuracy"); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Decoding: sufficiency AND necessity — the gap between them is redundancy.\n"
                 "Neither alone supports a ranking of regions.", fontsize=9)

    # --- notes -------------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0:2, 3:]); ax.axis("off")
    n_sess = len(good)
    ax.text(0, 1, "\n".join([
        f"{title}",
        f"{n_sess} sessions, {good[0]['n_features']} features, pre-cue window (2 s ending at the cue)",
        "",
        "WHAT CAN BE CLAIMED",
        "  • Encoding EV per region — univariate per target, so neither footprint overlap nor",
        "    activity correlation affects it. Several regions scoring high is real redundancy.",
        "  • Sufficiency and necessity TOGETHER. Redundant regions each look sufficient and",
        "    neither looks necessary; the gap quantifies that.",
        "  • Activation maps (below) — the decoder is one linear functional on the movie, and",
        "    its PIXEL image is well defined even though per-feature weights are not.",
        "",
        "WHAT CANNOT",
        "  • 'Component X mattered most' — footprints overlap (median cosine 0.977), so the",
        "    amplitude split within an overlapping group is arbitrary.",
        "  • 'Region R had the biggest weight' — 66 ROIs carry ~1.5 effective noise dimensions;",
        "    weights are unidentifiable, and a decoder loads on signal-free regions to cancel",
        "    shared noise (Haufe et al. 2014). Maps below are ACTIVATIONS, not weights.",
    ]), va="top", ha="left", fontsize=7.6, family="monospace")

    # --- activation maps ---------------------------------------------------------------------
    imgs = [r["_img"] for r in good if r.get("_img") is not None]
    if imgs:
        stack = np.nanmean(np.stack(imgs, 0), 0)
        v = np.nanpercentile(np.abs(stack), 99)
        for i, nm in enumerate(good[0]["positions"]):
            axm = fig.add_subplot(gs[2, i])
            axm.imshow(stack[:, :, i], cmap="RdBu_r", vmin=-v, vmax=v)
            axm.set_title(nm, fontsize=8); axm.axis("off")
        fig.text(0.5, 0.335, "Haufe activation patterns per position "
                             "(where the pre-cue position signal is; red = higher for that position)",
                 ha="center", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png
