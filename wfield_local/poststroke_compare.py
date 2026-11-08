"""Post-stroke vs pre-stroke, POSITION-MATCHED, in the four readout conditions.

WHY MATCHING IS THE WHOLE ANALYSIS. On 8/17 both animals stopped attempting the far positions --
PS94 has ZERO engaged trials at far_center and far_R (from ~106 each pre-stroke), PS95 has 10 and 1.
So a frozen 6-position model is being scored on 4 positions, and the raw accuracy drop (PS94 0.699
vs a 0.937 pre-stroke band) is mostly TRIAL COMPOSITION, not coding. Every comparison here is
restricted to positions the animal still attempts post-stroke, in BOTH phases, so the question is
"did coding change where behavior survived" rather than "did behavior change".

Three questions (Priya, 2026-08-18):

  1. Do post-stroke PRE-CUE no-lick trials look like pre-stroke LICKING or pre-stroke NON-LICKING
     trials? Tested directly: train a discriminator on pre-stroke engaged-vs-undetected pre-cue
     patterns, POSITION-BALANCED so it cannot simply learn "far", then apply it to post-stroke
     undetected trials and report which side they fall on.
  2. How similar are the POST-CUE models? Per-position mean-pattern correlation between phases,
     plus the frozen decoder's per-position recall.
  3. At the PRESERVED positions, does POST-LICK look the same?
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config
from wfield_local import nolick_analysis as na
from wfield_local.locanmf_frozen_decoder import (
    _pipe,
    frozen_decoder_models,
    frozen_scored_frame,
    pool_sessions,
)
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

MIN_POST = 20          # a position needs this many post-stroke trials to be "preserved" (`preserved_positions`)

#: Positions used to judge post-stroke ENGAGEMENT: close_L and close_center only (Priya,
#: 2026-08-18). Deliberately NOT every position that survives on 8/17. close_R is CONTRALESIONAL
#: (both lesions are left-sided) and far_L carries the distance effect, so either could be impaired
#: -- and a reference position that is itself impaired turns the gate back into the thing it exists
#: to avoid, labelling motor failure as disengagement. These two are the least likely to be affected
#: by a left VLS lesion, so a miss there is the strongest available evidence of genuine
#: disengagement. Empirical: revisit if an animal turns out to miss them.
REFERENCE_POSITIONS = [c for c in DISPLAY_ORDER
                       if POSITION_NAMES[c] in ("close_L", "close_center")]


def excluded_labels(animal):
    """Sessions for `animal` whose phase is neither pre nor post.

    Currently PS92_0817 / PS93_0817: the 8/16 lesion produced no deficit and was redone AFTER that
    session, so it is baseline-like but not a clean baseline, and post-lesion but not post-deficit.
    They are analysable and must never be pooled.
    """
    out = []
    for s in config.load_sessions():
        lab = s["label"]
        if not lab.startswith(animal + "_"):
            continue
        if config.session_phase(animal, lab.split("_")[-1]) == "excluded":
            out.append(lab)
    return sorted(out)


def _pooled(animal, align, source="roi", post_labels=None):
    """Pre-stroke pool + a comparison session, aligned and z-scored together.

    `post_labels` overrides the comparison arm. Leave it None for real post-stroke work -- the default
    is `phase_labels("post")`, which is the only sanctioned pool. Pass it ONLY to analyze sessions that
    resolve to the 'excluded' phase, and label the output accordingly; `excluded_labels()` supplies
    them. Results built this way must not enter any pooled post-stroke summary (Priya, 2026-08-18).
    """
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    if post_labels is None:
        # THE SAME POOL THE FROZEN DECODER WAS FROZEN AGAINST, by calling the same function rather
        # than by rebuilding the same list. `config.pooled_labels` exists because three call sites
        # built it independently and identically wrongly; a fourth open-coded copy here would make
        # the stored model a permanent MISS instead of a hit, because for ROI features `_align_many`
        # intersects region x bin columns across the pool and a different pool is a different
        # `n_features` and therefore a different spec id.
        labs = config.pooled_labels(animal)
        post = [l for l in labs if l not in set(pre)]
    else:
        post = [l for l in post_labels if l.startswith(animal)]
    if not pre or not post:
        return None
    # the ROI builder that also keeps cue indices (main 48f2cc6d), so the pool maps onto the camera working trials
    from wfield_local import working_trials as _wt
    feat = _wt.roi_features_with_indices() if source == "roi" else None
    p = pool_sessions(pre + post, source=source, align=align, post_s=2.0, features=feat)
    if p is None:
        return None
    XE, YE, GE, BE, XU, YU, kept, _common, GU = p
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}
    post_i = {i for i, l in enumerate(kept) if l not in set(pre)}
    return {"XE": XE, "YE": YE, "GE": GE, "BE": BE, "XU": XU, "YU": YU.astype(int), "GU": GU,
            "kept": kept, "pre_i": pre_i, "post_i": post_i,
            # WHAT THE FROZEN MODEL IS KEYED ON, recorded at pool time. `post_i` is MUTATED by
            # `poststroke_section_g` to isolate one session at a time; `align`/`source` are not, and
            # `frozen` must key on the pre-stroke training set rather than on whatever comparison
            # arm the caller has narrowed to, or every narrowing would look like a new model.
            "align": align, "source": source, "_features": feat,
            # THE POOLED FEATURE COLUMNS (ROI: region x bin intersected over the pool), part of every section G stored
            # model's identity (`_g_store`, 11-02 pass 2): an excluded-session pool can have a different column set.
            "common": list(_common) if _common is not None else None,
            # DELIBERATELY NOT the frozen model itself. It is loaded lazily and memoised on first
            # use, so a caller that never needs it never pays for it.
            "_frozen_cache": {}}


def lick_pipe():
    """The lick-vs-no-lick discriminator's estimator. NOT the position decoder.

    Same hyperparameters as `locanmf_frozen_decoder._pipe`, and that coincidence is the trap: this
    model has a two-class label space (1 = engaged, 0 = no detected lick) and that one has six
    positions, so they are different objects that happen to be configured alike. Anything keyed on
    "the frozen model" must never serve one for the other -- `frozen_models` would need
    ``kind="lick_discriminator"``, not ``kind="decoder"``.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def balanced_lick_sample(d, keep, rng, *, exclude_session=None):
    """Position-balanced PRE-stroke engaged-vs-no-lick training set: ``(X, y, groups)`` or None.

    Within each kept position both classes are subsampled to the same count, so POSITION carries no
    information about the label. Without that the discriminator answers "far" and looks like an
    answer: undetected trials are overwhelmingly far pre-stroke, and post-stroke no-lick trials are
    far too. A position with fewer than 5 of either class is skipped.

    ``groups`` are session ids, for callers that want GroupKFold -- an ungrouped fold puts trials
    from one session on both sides, and session identity is exactly what a whole-brain feature
    vector can memorize.

    THREE COPIES OF THIS EXISTED (`looks_like_which`, `fits_engaged_distribution.balanced_fit`,
    `undetected_state_split`), agreeing only by coincidence, and one of them did not: `balanced_fit`
    shares a single RandomState across its leave-one-out loop, so by the time it takes the full-pool
    fit the generator has already been advanced by every fold before it. The other two each start a
    fresh RandomState(seed) and therefore draw the SAME sample as each other.

    THE RNG IS AN ARGUMENT, NOT CREATED HERE, so that difference is preserved exactly rather than
    quietly unified. Making `balanced_fit` deterministic per fold would be defensible and would
    change published numbers; that is a decision, not a refactor, and this is the refactor.
    """
    e = np.isin(d["GE"], list(d["pre_i"])) & np.isin(d["YE"], keep)
    u = np.isin(d["GU"], list(d["pre_i"])) & np.isin(d["YU"], keep)
    if exclude_session is not None:
        e &= d["GE"] != exclude_session
        u &= d["GU"] != exclude_session
    Xe, ye, ge = d["XE"][e], d["YE"][e], d["GE"][e]
    Xu, yu, gu = d["XU"][u], d["YU"][u], d["GU"][u]
    xs, lab, grps = [], [], []
    for c in keep:
        ie, iu = np.flatnonzero(ye == c), np.flatnonzero(yu == c)
        n = min(len(ie), len(iu))
        if n < 5:
            continue
        se = rng.choice(ie, n, replace=False)
        su = rng.choice(iu, n, replace=False)
        xs.append(Xe[se]); lab.append(np.ones(n)); grps.append(ge[se])
        xs.append(Xu[su]); lab.append(np.zeros(n)); grps.append(gu[su])
    if not xs:
        return None
    return np.vstack(xs), np.concatenate(lab), np.concatenate(grps)


def _lick_train(d, keep):
    """``d`` with its XE / XU in the engagement-state discriminator's TRAINING frame (bal_em over the two STATE
    classes, `spout_scaling.state_train_frame`): each pre-stroke session centered on the equal-weight mean of its engaged
    and no-lick means over the ``keep`` positions. OPT-IN (``WIDEFIELD_STATE_DISC_SCALING=state_bal_em``), default OFF:
    ``d`` itself (it fails the discriminator's own pre-stroke check; DECISIONS 2026-11-03, `spout_scaling`)."""
    from wfield_local import spout_scaling as _ss
    mode = _ss.state_disc_scaling()
    if not mode:
        return d
    if mode == _ss.SCALING_SPOUT_FRAME:
        return _spout_train_d(d)
    XE, XU = _ss.state_train_frame(d["XE"], d["GE"], d["XU"], d["GU"], np.isin(d["YE"], keep),
                                   np.isin(d["YU"], keep), sorted(d["pre_i"]))
    dT = dict(d)
    dT["XE"], dT["XU"] = XE, XU
    return dT


def _lick_scored(d, keep, clf, y_train, sessions, base):
    """``(XE, XU)``: ``base``'s matrices with each session in ``sessions`` moved into ``clf``'s label-free EM
    scoring frame (center from that session's PRODUCTION-frame engaged + no-lick rows at the ``keep`` positions,
    equal weight over the two states, production SD). ``base``'s own arrays under session_z."""
    from wfield_local import spout_scaling as _ss
    mode = _ss.state_disc_scaling()
    if not mode:
        return base["XE"], base["XU"]
    if mode == _ss.SCALING_SPOUT_FRAME:
        # the scored sessions in the frozen SPOUT decoder's scoring frame (`frozen_X`, main 26fd5b44): the
        # discriminator itself plays no part in the centre
        XEs, XUs = frozen_X(d)
    else:
        XEs, XUs = _ss.state_scored(clf, _ss.model_prior(clf, y_train), d["XE"], d["GE"], d["XU"], d["GU"],
                                    np.isin(d["YE"], keep), np.isin(d["YU"], keep), sorted(sessions))
    oE, oU = np.array(base["XE"], float), np.array(base["XU"], float)
    mE, mU = np.isin(d["GE"], list(sessions)), np.isin(d["GU"], list(sessions))
    oE[mE], oU[mU] = XEs[mE], XUs[mU]
    return oE, oU


def _work_txt() -> str:
    """The run's camera working classes in words, for the section G captions ("contact + incomplete" in production;
    "contact" under the in_working_contact supplement, Priya 11-07; `trial_classes.sel_class_text`)."""
    from wfield_local import trial_classes as _tc
    return _tc.sel_class_text()


def _state_frame_tag():
    """The engagement-state discriminators' frame, recorded in every output that uses it (main 4a37cbc8): "spout_frame/
    working" (section G's spout frame is the stored cue / precue decoder's own frame, so its center is always the
    camera working rows -- or "spout_frame/contact" under WIDEFIELD_SPOUT_SCORE_CENTRE=contact; the lick window and the
    DAQ gate: "spout_frame/engaged"), "session_z", or "state_bal_em". The camera-class discriminator (`g4_camclass`)
    records its own frame (`g4_frame_tag`)."""
    from wfield_local import spout_scaling as _ss
    from wfield_local import trial_classes as _tc
    from wfield_local import working_trials as _wt
    mode = _ss.state_disc_scaling()
    if mode == _ss.SCALING_SPOUT_FRAME:
        cam = _tc.camera_gate_on() and _wt.is_camera(_tc.working_selection())
        return "spout_frame/" + (_ss.score_centre_rows() if cam else "engaged")
    return mode or "session_z"


def _spout_train_d(d):
    """``d`` with every pre-stroke session in the frozen SPOUT decoder's TRAINING frame (its true spout-class means
    over the rows the model was trained on -- working rows for cue / precue, engaged rows for the lick window -- and
    that row set's SD), for the state discriminators' ``spout_frame`` mode (main 26fd5b44, Priya 11-03, item B). Pre
    sessions the spout model was not trained on (no camera classes) leave ``pre_i``. ``d`` itself when no stored
    model."""
    from wfield_local import label_shift
    from wfield_local.locanmf_frozen_decoder import working_rows
    models = frozen(d)
    if not models or "pi_train" not in models:
        return d
    R = working_rows(d["_frozen_cache"].get("work")) or (d["XE"], d["YE"], d["GE"])
    RX, RY, RG = R
    XE, XU = np.array(d["XE"], float), np.array(d["XU"], float)
    keep = set()
    for i in sorted(d["pre_i"]):
        if d["kept"][i] not in models["loso"] or not np.any(RG == i):
            continue
        c = label_shift.balanced_centre(RX[RG == i], RY[RG == i], range(6))
        sd = label_shift.session_sd(RX[RG == i])
        XE[d["GE"] == i] = (XE[d["GE"] == i] - c) / sd
        if len(d["GU"]):
            XU[d["GU"] == i] = (XU[d["GU"] == i] - c) / sd
        keep.add(i)
    dT = dict(d)
    dT["XE"], dT["XU"], dT["pre_i"] = XE, XU, keep
    return dT


#: the rules `preserved_positions` can choose positions by (recorded per session in section G)
PRESERVED_RULES = ("camera_working", "daq_engaged")


def _engaged_counts(d, idx) -> dict:
    """{position code: DAQ-engaged rows} over the sessions ``idx``."""
    m = np.isin(d["GE"], list(idx))
    return {c: int((d["YE"][m] == c).sum()) for c in DISPLAY_ORDER}


def _working_counts(d, idx):
    """{position code: camera WORKING rows (engaged + no-lick)} over the sessions ``idx``, or None when the camera
    rule does not apply (no camera grouping for this pool, or a session in ``idx`` without camera classes)."""
    cg = _camera_groups(d)
    idx = list(idx)
    if cg is None or not idx or any(i not in cg["have"] for i in idx):
        return None
    me = np.isin(d["GE"], idx) & cg["work_e"]
    mu = np.isin(d["GU"], idx) & cg["work_u"]
    YU = np.asarray(d["YU"])
    return {c: int((d["YE"][me] == c).sum() + (YU[mu] == c).sum()) for c in DISPLAY_ORDER}


def _position_counts(d, idx):
    """``(rule, {position code: n})``: the counts `preserved_positions` thresholds (see its docstring)."""
    wc = _working_counts(d, idx)
    if wc is not None:
        return "camera_working", wc
    return "daq_engaged", _engaged_counts(d, idx)


def preserved_detail(d, session) -> dict:
    """One session's preserved set with the rule that chose it and the per-position counts used (section G record)."""
    rule, n = _position_counts(d, [session])
    return {"keep": [c for c in DISPLAY_ORDER if n[c] >= MIN_POST], "rule": rule,
            "counts": {POSITION_NAMES[c]: n[c] for c in DISPLAY_ORDER}, "min_post": MIN_POST}


def preserved_positions(d, session=None, combine="intersection"):
    """Positions the animal still attempts post-stroke -- the only ones a comparison can use.

    PER SESSION, because "still attempts" is a behavioral state that changes day to day (Priya,
    2026-08-19). Pass `session` (an index into d["kept"]) for one session's set.

    WITHOUT `session`, sessions are combined by INTERSECTION: positions attempted on EVERY
    post-stroke day. An earlier version pooled the trial counts across sessions, which is the UNION,
    and it produced a concrete error -- PS95 attempted far_center/far_R on 8/18 (99 and 84 trials) but
    not on 8/17 (10 and 1), so the pooled set was six positions and PS95's 8/17 numbers were computed
    over a position with ONE engaged trial. It also silently moved that result's chance level from
    0.25 to 0.167 when 8/18 was registered, with nothing about 8/17 having changed.

    A pooled statistic has to be defensible for every session inside it, which is what the
    intersection guarantees and the union cannot.

    WHICH ROWS ARE COUNTED (Priya 11-07, "we should NOT gate on successful licks"): under the camera grouping
    (`_camera_groups`: camera gate, a camera working selection, cue / precue pool with classes) a position is preserved
    when the session has >= MIN_POST camera WORKING rows there (engaged + no-lick rows in the selection's classes:
    contact | incomplete in production, contact only under in_working_contact) -- the rows the lick-only arm scores
    (`_lickonly_working`). Before, the DAQ-engaged count chose the positions, so far contra (33-68 incomplete licks,
    < 20 DAQ licks) was dropped although its working rows were there. Otherwise (DAQ gate, lick window, a session
    without camera classes) the DAQ-engaged count, as before. `preserved_detail` reports the rule and the counts.
    """
    def _for(idx):
        return {c for c, n in _position_counts(d, idx)[1].items() if n >= MIN_POST}

    if session is not None:
        return [c for c in DISPLAY_ORDER if c in _for([session])]
    per = [_for([i]) for i in sorted(d["post_i"])]
    if not per:
        return []
    if combine == "union":                       # never the default; here only to be explicit
        keep = set.union(*per)
    elif combine == "pooled":                    # the old behaviour, kept so it can be compared
        keep = _for(list(d["post_i"]))
    else:
        keep = set.intersection(*per)
    return [c for c in DISPLAY_ORDER if c in keep]


def frozen(d):
    """The stored pre-stroke decoder for this pool, memoised on ``d``.

    Returns ``{"full": pipeline, "loso": {label: pipeline}}``, or None when the pool did not record
    what it was built from (an `excluded_labels` pool, or a dict assembled by a test). A None here
    means the caller fits locally exactly as it did before -- this is a shortcut, never a new source
    of truth, so a pool it cannot key for degrades to the old behavior rather than to a wrong model.

    MEMOISED PER POOL, which is where the saving is. `poststroke_section_g` mutates only
    ``d["post_i"]`` and calls back in once per post-stroke session per arm per alignment, so the
    identical pre-stroke model was being refitted on every one of those passes. The pre-stroke
    training set does not depend on ``post_i`` at all.
    """
    cache = d.get("_frozen_cache")
    if cache is None or "align" not in d:
        _refuse_unkeyed("frozen(d)")
        return None
    if "models" not in cache:
        # NO SILENT IN-MEMORY FALLBACK (fifth review D-B1): a store error (WriteGuardError, the LP "loads DAQ-frozen"
        # refusal, an unreadable entry, a centre refusal) propagates; it used to print "fitting locally" and let every
        # caller refit a pre-stroke decoder in memory, on the session_z frame, untagged
        # trained on the pre-stroke CAMERA WORKING trials for cue / precue (Priya 11-03 late, main 48f2cc6d `pool_work`:
        # engaged AND no-lick rows of class contact | incomplete); the engaged trials for the lick window
        from wfield_local.locanmf_frozen_decoder import pool_work
        cache["work"] = pool_work(d["XE"], d["YE"], d["GE"], d["XU"], d["YU"], d["GU"], d["kept"],
                                  d.get("_features"), d["align"])
        models, _status = frozen_decoder_models(
            d["XE"], d["YE"], d["GE"], d["kept"], d["pre_i"],
            align=d["align"], source=d["source"], work=cache["work"])
        cache["models"] = models
    return cache["models"]


class SectionGRefused(SystemExit):
    """A section G reference that cannot come from the frozen store in a non-default run (a SystemExit, as
    `frozen_models.LegacyStoreError`: no `except Exception` may turn it into a silent in-memory fit)."""


def _refuse_unkeyed(what: str) -> None:
    """A pool that records no identity (an `excluded_labels` pool, a test dict) can only be fitted in memory: allowed
    in the DEFAULT run (the pre-rebuild behavior), REFUSED in a tagged run (the render profile)."""
    from wfield_local import lp_licks
    if lp_licks.output_tag():
        raise SectionGRefused(f"section G {what}: the pool records no identity, so its pre-stroke reference cannot "
                              f"come from the frozen store; refusing an untagged in-memory fit in run "
                              f"{lp_licks.output_tag()!r}")


#: STORED BY DEFAULT (11-02 pass 2; Priya: "yes cache all frozen decoders"; CLAUDE.md rule 10). Section G's OTHER
#: pre-stroke references -- the ones `frozen(d)` (kind `decoder`) cannot serve -- go through `frozen_models.load_or_fit`
#: too, one kind each, memoised on the pool like `frozen`:
#:   g_lickonly_decoder   `decode_matched` lick-only arm: the position decoder on pre engaged trials at the session's
#:                        PRESERVED positions ({"full", "loso"}); align carries the preserved set
#:   g_lick_disc          `looks_like_which` / `undetected_state_split`: the lick-vs-no-lick discriminator on the
#:                        position-balanced sample drawn with a FRESH RandomState(seed) (both draw the same sample, so
#:                        they share one stored model); align carries the preserved set and the seed
#:   g_lick_disc_folds    `fits_engaged_distribution`: the leave-one-session-out discriminators AND the full one, drawn
#:                        from ONE RandomState(seed) advancing fold by fold, then the full fit -- stored as one payload
#:                        so the draws (and the published numbers) are exactly the in-memory ones
#: Every align also carries a digest of the pooled feature columns. OPT-OUT WIDEFIELD_SECTION_G_FROZEN=0 fits in memory
#: as before. Under WIDEFIELD_LICK_SOURCE=lp the store serves the DAQ-trained model (training set = the animal's DAQ pre
#: sessions), LP-trained with WIDEFIELD_LP_FROZEN=lp; a DAQ-trained spec that no DAQ run stored RAISES (the readout is
#: then reported failed in the log, never refitted on LP trials under a DAQ name).
#: Still in memory: the fallbacks taken when `frozen(d)` is None (a pool that records no identity, or a failed lookup),
#: and the within-pre CV estimates of `looks_like_which` (not a reference applied to post-stroke data).
SECTION_G_FROZEN_ENV = "WIDEFIELD_SECTION_G_FROZEN"


def _keep_tag(keep) -> str:
    return "keep" + "-".join(str(int(c)) for c in sorted(keep))


def _g_store(d, kind, extra, fit, meta=None, scaling=None):
    """The stored model for one section G reference (see SECTION_G_FROZEN_ENV), or None (opted out / unkeyable pool:
    the caller fits in memory exactly as before)."""
    import hashlib

    from wfield_local import frozen_models
    cache = d.get("_frozen_cache")
    if not frozen_models.family_on(SECTION_G_FROZEN_ENV):
        return None
    if cache is None or "align" not in d:
        _refuse_unkeyed(kind)
        return None
    key = (kind, extra)
    if key in cache:
        return cache[key]
    kept = d["kept"]
    animal = config.animal_of(kept[0])
    train = sorted(kept[i] for i in d["pre_i"])
    if frozen_models.daq_trained_in_lp():
        train = sorted(x for x in config.phase_labels("pre") if config.animal_of(x) == animal)
    fdig = hashlib.sha1(repr(d.get("common")).encode()).hexdigest()[:8]
    # TRAINING-DATA FINGERPRINT (rebuild 11-03): every section G reference is fitted on rows of the animal's PRE pool
    # (engaged, plus no-lick where an arm uses it), so the fingerprint covers that whole pre pool -- the PS92 11-02
    # mismatch (pre trial features changed under an unchanged spec) now mints a new id instead of loading the old fit.
    pre_ix = sorted(d["pre_i"])
    me = np.isin(np.asarray(d["GE"]), pre_ix)
    pool = [d["XE"][me], d["YE"][me], np.asarray(d["GE"])[me]]
    if d.get("XU") is not None and d.get("GU") is not None and len(d["XU"]):
        mu = np.isin(np.asarray(d["GU"]), pre_ix)
        pool += [d["XU"][mu], np.asarray(d["YU"])[mu] if d.get("YU") is not None else None]
    spec = frozen_models.make_spec(animal, kind, align=f"{d['align']}-{extra}-f{fdig}", source=d["source"],
                                   train_labels=train, post_s=2.0, n_features=int(d["XE"].shape[1]),
                                   train_data=pool,
                                   **__import__("wfield_local.spout_scaling", fromlist=["x"]).spec_kw(scaling))
    payload, status = frozen_models.load_or_fit(spec, fit, meta=meta or {})
    print(f"  [section_g] {animal} {kind} {frozen_models._slug(spec)} {status} "
          f"(trained on {spec.get('lick_source', 'daq')} pre trials)", flush=True)
    cache[key] = payload
    return payload
def _engaged_classes(d):
    """Camera class per pooled engaged row of section G pool ``d`` for the bal_em center (None unless the run's center
    is the camera working set), memoised on the pool."""
    from wfield_local.locanmf_frozen_decoder import engaged_row_classes
    cache = d.setdefault("_frozen_cache", {})
    if "cls_e" not in cache:
        cache["cls_e"] = engaged_row_classes(d["kept"], len(d["YE"]), source=d.get("source", "roi"),
                                             align=d.get("align", "cue"))
    return cache["cls_e"]


def frozen_X(d):
    """``(XE, XU)`` in the frame the STORED models of `frozen` score in, memoised on ``d``.

    Under ``session_bal_em`` (production since 2026-11-03, `spout_scaling`) each session is centered by the label-free
    EM center of its engaged trials under its scoring model (LOSO for a pre session, ``full`` otherwise); under
    ``WIDEFIELD_SPOUT_SCALING=session_z``, or when no stored model is available (local fallback fits), it is
    ``(d["XE"], d["XU"])`` itself. The scored frame does not depend on ``post_i``, which `poststroke_section_g`
    mutates, so one frame serves every narrowing of the pool.
    """
    cache = d.get("_frozen_cache")
    models = frozen(d)
    if models is None or cache is None:
        return d["XE"], d["XU"]
    # the memo carries the scored frame's centre rows (coordinator 11-04, `spout_scaling.score_centre_rows`)
    xkey = "X" + __import__("wfield_local.spout_scaling", fromlist=["x"]).score_centre_tag()
    if xkey not in cache:
        cache[xkey] = frozen_scored_frame(models, d["XE"], d["GE"], d["XU"], d["GU"], d["kept"], d["pre_i"],
                                          cls_e=None if cache.get("work") is not None else _engaged_classes(d),
                                          work=cache.get("work"))
    return cache[xkey]


#: SECTION G's GROUPS (main 4a37cbc8, Priya 11-04, item G). "camera" (default): the pre panel of `crossed_confusion` and
#: the pre band of `decode_matched` (both arms; the lick-only arm since follow-on c) are the pre-stroke camera WORKING
#: trials (contact + incomplete, engaged and no-lick rows: the rows the cue / precue decoders are trained and centred
#: on), the engaged-only post arm of `crossed_confusion` is the post session's working trials, and `crossed_confusion`
#: adds one panel per camera group (`working_trials.G_GROUPS`: contact / incomplete / jaw_only / none / after_end) for
#: pre and post, with each group's frozen-decoder accuracy -- the failed-attempt readout. "daq": the pre-11-04 DAQ-engaged
#: rows and no by-class panels (robustness). The lick window keeps the lick gate. Sessions without camera classes
#: (PS93_0813) leave the camera panels.
SECTION_G_GROUPS_ENV = "WIDEFIELD_SECTION_G_GROUPS"


def section_g_groups() -> str:
    import os
    v = os.environ.get(SECTION_G_GROUPS_ENV, "camera").strip().lower()
    if v in ("camera", "cam", "classes"):
        return "camera"
    if v in ("daq", "engaged", "old"):
        return "daq"
    raise ValueError(f"{SECTION_G_GROUPS_ENV}={v!r}: expected camera or daq")


def _camera_groups(d):
    """``{"grp_e", "grp_u", "jj_e", "jj_u", "have", "work_e", "work_u"}`` for a pool under the camera grouping (cue /
    precue, the camera gate with a camera working selection, a builder with cue indices, a working-trained stored
    model), memoised on ``d``; else None (the DAQ grouping)."""
    from wfield_local import trial_classes as _tc
    from wfield_local import working_trials as wt
    cache = d.get("_frozen_cache")
    if (section_g_groups() != "camera" or d.get("align") == "lick" or not _tc.camera_gate_on()
            or not wt.is_camera(_tc.working_selection()) or not getattr(d.get("_features"), "indices", None)
            or cache is None):
        return None
    if "groups" not in cache:
        frozen(d)
        work = cache.get("work")
        if work is None:
            cache["groups"] = None
        else:
            ge, gu, je, ju, have = wt.row_groups(d["kept"], d["_features"].indices, d["GE"], d["GU"],
                                                 YE=d["YE"], YU=d["YU"])
            cache["groups"] = {"grp_e": ge, "grp_u": gu, "jj_e": je, "jj_u": ju, "have": have,
                               "work_e": np.asarray(work["work_e"], bool), "work_u": np.asarray(work["work_u"], bool)}
    return cache["groups"]


def _conf_panel(y, p, nl, labels, jj=None):
    """One row-normalized confusion panel (rows = true position): the dict `crossed_confusion` stores per phase.
    With ``jj`` (the rows' jaw_judged flags) it also records n, accuracy and the number of jaw-unjudged rows."""
    M = np.full((len(labels), len(labels)), np.nan)
    n, n_nolick = [], []
    for i, c in enumerate(labels):
        sel = y == c
        n.append(int(sel.sum()))
        # how much of this row is carried by trials with no detected lick: a row that is 100%
        # no-lick is a different kind of evidence from one that is 10%, and the matrix cannot
        # show that by itself
        n_nolick.append(int(nl[sel].sum()))
        if sel.sum():
            for j, c2 in enumerate(labels):
                M[i, j] = float((p[sel] == c2).mean())
    out = {"matrix": M.tolist(), "n_per_true_position": n, "n_nolick_per_true_position": n_nolick,
           "positions": [POSITION_NAMES[c] for c in labels]}
    if jj is not None:
        out["n_jaw_unjudged"] = int((~np.asarray(jj, bool)).sum())
        out["accuracy"] = float(np.mean(p == y)) if len(y) else None
        out["n"] = int(len(y))
    return out


def _loso_rows(d, models, XEf, XUf, me, mu):
    """``(y, pred, from_nolick)`` for PRE-session rows (engaged mask ``me``, no-lick mask ``mu``), each session scored by
    its leave-that-session-out stored model in the stored models' frame."""
    y = np.concatenate([d["YE"][me], d["YU"][mu].astype(d["YE"].dtype)])
    G = np.concatenate([d["GE"][me], d["GU"][mu]])
    X = np.vstack([XEf[me], XUf[mu]]) if len(y) else np.zeros((0, XEf.shape[1]))
    p = np.empty_like(y)
    for i in np.unique(G):
        m = G == i
        p[m] = models["loso"][d["kept"][int(i)]].predict(X[m])
    return y, p, np.concatenate([np.zeros(int(me.sum()), bool), np.ones(int(mu.sum()), bool)])


def _by_class_panels(d, models, clf, XEf, XUf, cg, labels):
    """Section G's per-camera-group panels (main 4a37cbc8, Priya 11-04, item G): for every group in
    `working_trials.G_GROUPS`, the frozen decoder's confusion + accuracy on the PRE sessions' trials of that group (each
    pre session by its LOSO model) and on the POST session's (the full model). Engaged and no-lick rows both count: the
    group is the camera class, not the DAQ lick. Sessions without camera classes contribute nothing."""
    from wfield_local import working_trials as wt
    lo = [i for i in d["pre_i"] if d["kept"][i] in models["loso"] and i in cg["have"]]
    post = [i for i in d["post_i"] if i in cg["have"]]
    out = {"pre": {}, "post": {}, "groups": list(wt.G_ALL), "group_labels": dict(wt.G_LABELS),
           "post_sessions_without_camera_classes": [d["kept"][i] for i in sorted(set(d["post_i"]) - set(post))],
           "pre_sessions_scored": [d["kept"][i] for i in sorted(lo)]}
    for g in wt.G_ALL:
        me = np.isin(d["GE"], lo) & (cg["grp_e"] == g)
        mu = np.isin(d["GU"], lo) & (cg["grp_u"] == g)
        y, p, nl = _loso_rows(d, models, XEf, XUf, me, mu)
        out["pre"][g] = _conf_panel(y, p, nl, labels, jj=np.concatenate([cg["jj_e"][me], cg["jj_u"][mu]]))
        me = np.isin(d["GE"], post) & (cg["grp_e"] == g)
        mu = np.isin(d["GU"], post) & (cg["grp_u"] == g)
        y = np.concatenate([d["YE"][me], d["YU"][mu].astype(d["YE"].dtype)])
        X = np.vstack([XEf[me], XUf[mu]])
        p = clf.predict(X) if len(y) else np.empty(0, dtype=y.dtype)
        nl = np.concatenate([np.zeros(int(me.sum()), bool), np.ones(int(mu.sum()), bool)])
        out["post"][g] = _conf_panel(y, p, nl, labels, jj=np.concatenate([cg["jj_e"][me], cg["jj_u"][mu]]))
    return out


def _proj_X(d):
    """``(XE, XU)`` for the pattern readouts in the projection frame (main 26fd5b44 / 4a37cbc8, item C;
    `spout_scaling.projection_arrays`), memoised on ``d`` per projection tag; ``d``'s own rows when the frame is off."""
    from wfield_local import spout_scaling as _ss
    from wfield_local import working_trials as _wt
    if not _ss.projection_frame_on():
        return d["XE"], d["XU"]
    cache = d.get("_frozen_cache")
    key = "proj_" + _ss.projection_tag()
    if cache is not None and key in cache:
        return cache[key]
    feat = d.get("_features")
    out = _ss.projection_arrays(d["XE"], d["YE"], d["GE"], d["XU"], d["GU"], d["pre_i"], _pipe,
                                work=_wt.projection_work(d["kept"], feat, d["GE"], d["GU"], d.get("align")),
                                YU=d["YU"], cls_e=_wt.projection_cls_e(d["kept"], feat, d["GE"], d["GU"],
                                                                       d.get("align")), labels=d["kept"])
    if cache is not None:
        cache[key] = out
    return out


def preserved_positions_by_session(d):
    """{session label -> preserved positions} for every post-stroke session, for reporting."""
    return {d["kept"][i]: preserved_positions(d, session=i) for i in sorted(d["post_i"])}


def decode_matched(d, keep, post_all_trials=True):
    """Frozen pre-stroke decoder, scored on the SAME positions in both phases.

    POST ARM USES ALL TRIALS by default (Priya, 2026-08-18/19): the missing licks ARE the phenotype,
    and with the no-lick trials included every position has data, so the arm is scored over all SIX
    positions rather than the lick-defined `keep` set. An engaged-only version excludes the positions
    the lesion abolished -- PS94 has ZERO engaged and ~105 no-lick trials at far_center and far_R --
    so it can only ever describe the part of the phenotype that still works.

    `keep` still governs the LICK-ONLY arm, where it is forced: a position with no engaged trials
    cannot be decoded from engaged trials. The two arms therefore differ in chance level and callers
    must say which is shown.
    """
    pos = list(DISPLAY_ORDER) if post_all_trials else list(keep)
    kp = np.isin(d["YE"], pos)
    tr = np.isin(d["GE"], list(d["pre_i"])) & kp
    te = np.isin(d["GE"], list(d["post_i"])) & kp
    if tr.sum() < 50 or te.sum() < 20:
        return None
    # THE STORED MODEL, BUT ONLY ON THE ALL-TRIALS ARM. With `post_all_trials` the position set is
    # the full DISPLAY_ORDER and `kp` selects everything, so `tr` is precisely "all pre-stroke
    # engaged trials" -- the definition `frozen_decoder_models` freezes. The LICK-ONLY arm is a
    # DIFFERENT model: `keep` drops the positions the lesion abolished (4-way for PS94 and PS95, so
    # a different chance level), and serving it the 6-way model would silently change what its
    # accuracy means. It keeps fitting locally.
    models = frozen(d) if post_all_trials else None
    pre_band_rows = None              # (y, session) of the pre band when it is not the engaged rows (camera grouping)
    lo_w = None                       # the lick-only arm on camera working rows (follow-on c), when it applies
    # THE STORED MODEL SCORES IN ITS OWN FRAME (`frozen_X`; bal_em since 2026-11-03). The lick-only arm's local fit
    # stays on the production frame.
    XEf, XUf = frozen_X(d) if models is not None else (d["XE"], d["XU"])
    if models is not None:
        # a pre session with no LOSO model (dropped from a camera-working training set: no video, PS93_0813) has no
        # pre-band entry (main 48f2cc6d)
        tr = tr & np.isin(d["GE"], [i for i in d["pre_i"] if d["kept"][i] in models["loso"]])
        clf = models["full"]
        pre_pred = np.empty_like(d["YE"][tr])
        _gi = d["GE"][tr]
        for i in sorted(d["pre_i"]):
            m = _gi == i
            if m.sum():
                pre_pred[m] = models["loso"][d["kept"][i]].predict(XEf[tr][m])
        # THE PRE BAND UNDER THE CAMERA GROUPING (main 4a37cbc8, Priya 11-04, item G): the pre sessions' camera WORKING
        # trials (contact + incomplete, engaged and no-lick rows), each by its LOSO model
        cg = _camera_groups(d)
        if cg is not None:
            _lo = [i for i in d["pre_i"] if d["kept"][i] in models["loso"]]
            _me = np.isin(d["GE"], _lo) & cg["work_e"] & kp
            _mu = np.isin(d["GU"], _lo) & cg["work_u"] & np.isin(d["YU"], pos)
            _y, pre_pred, _nl = _loso_rows(d, models, XEf, XUf, _me, _mu)
            pre_band_rows = (_y, np.concatenate([d["GE"][_me], d["GU"][_mu]]))
    else:
        # THE LOCAL (class-filtered) FIT, bal_em since 2026-11-03 (`spout_scaling.FrozenFrame` over the ``pos``
        # classes): trained on pre sessions centred on the equal-weight mean of their true class means over
        # ``pos``; each scored session centred by the label-free EM centre of ALL its engaged rows under the
        # ``len(pos)``-way model. Under session_z this is the pre-11-03 fit and LOSO exactly.
        from wfield_local import spout_scaling as _ss
        # STORED BY DEFAULT (rebuild 11-02 pass 2, kind g_lickonly_decoder keyed on the preserved set; through the
        # FrozenFrame store hook since the main merge 11-03, payload {"full", "loso": {pooled index}} on its frame)
        _sc = _ss.spout_scaling()
        # FOLLOW-ON (c) (coordinator 11-04): UNDER THE CAMERA GROUPING the class-filtered fit is on the camera WORKING
        # rows (`_lickonly_working`): trained on the pre sessions' working rows at the ``pos`` spouts, pre band = their
        # working rows (LOSO), post arm = the post session's working rows at ``pos``. Else (DAQ grouping, lick window)
        # the engaged-row fit below, unchanged.
        lo_w = _lickonly_working(d, pos, _sc)

        def _store_lo(fit_all):
            got = _g_store(d, "g_lickonly_decoder", f"{_keep_tag(pos)}-ff", fit_all,
                           meta={"n_train": int(tr.sum())}, scaling=_sc)
            return got if got is not None else fit_all()

        if lo_w is not None:
            clf, pre_pred, pre_band_rows, post_rows_w = lo_w
            ff = None
        else:
            ff = _ss.FrozenFrame(d["XE"], d["YE"], d["GE"], sorted(d["pre_i"]), _pipe, classes=pos, train_mask=kp,
                                 loso=True, store=_store_lo, cls_e=_engaged_classes(d),
                                 labels=dict(enumerate(d["kept"])))
            clf = ff.full
        if ff is None:
            pass
        elif ff.scaling or ff.loso:
            pre_pred = np.empty_like(d["YE"][tr])
            _gi = d["GE"][tr]
            for i in sorted(d["pre_i"]):
                m = _gi == i
                if m.sum():
                    pre_pred[m] = ff.loso[i].predict(ff.apply(d["XE"][tr][m], i, "loso"))
            XEf = ff.apply_rows(d["XE"], d["GE"])
            XUf = ff.apply_rows(d["XU"], d["GU"]) if len(d["GU"]) else d["XU"]
        else:
            # pre-stroke baseline under the SAME restriction, leave-one-session-out
            pre_pred = cross_val_predict(_pipe(), d["XE"][tr], d["YE"][tr],
                                         cv=LeaveOneGroupOut(), groups=d["GE"][tr])
    per = {}
    _pre_y, _pre_g = pre_band_rows if pre_band_rows is not None else (d["YE"][tr], d["GE"][tr])
    for i in sorted(d["pre_i"]):
        m = _pre_g == i
        if m.sum():
            per[d["kept"][i]] = float(accuracy_score(_pre_y[m], pre_pred[m]))
    if post_all_trials:
        teu = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], pos)
        Xpost = np.vstack([XEf[te], XUf[teu]])
        ypost = np.concatenate([d["YE"][te], d["YU"][teu]])
    elif not post_all_trials and lo_w is not None:
        Xpost, ypost = post_rows_w
    else:
        Xpost, ypost = XEf[te], d["YE"][te]
    post_pred = clf.predict(Xpost)
    out = na.evaluate_arm(ypost, post_pred, n_perm=1000, labels=pos)
    out["accuracy"] = float(accuracy_score(ypost, post_pred))
    out["post_arm"] = ("ALL trials" if post_all_trials else
                       f"lick-only spouts, camera WORKING trials ({_work_txt()})" if lo_w is not None
                       else "lick-only")
    out["positions_scored"] = [POSITION_NAMES[c] for c in pos]
    out["n_post"] = len(ypost)
    v = np.array(list(per.values()), float)
    out["pre_band"] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                       "n_sessions": len(per)}
    out["below_every_pre_session"] = bool(out["accuracy"] < out["pre_band"]["min"])
    out["pre_recall"] = na.per_position_recall(_pre_y, pre_pred, labels=pos)
    out["pre_band_rows"] = (f"camera WORKING ({_work_txt()})" if pre_band_rows is not None
                            else "engaged")
    return out


def _lickonly_working(d, pos, sc):
    """``(clf, pre_pred, (pre_y, pre_g), (Xpost, ypost))`` for `decode_matched`'s LICK-ONLY arm on the camera WORKING
    rows (follow-on c, coordinator 11-04), or None (not the camera grouping: `_camera_groups`). A FrozenFrame over the
    working rows R (engaged + no-lick, contact | incomplete) restricted to the ``pos`` spouts for training; each scored
    session centered, label-free, on ALL its R rows (or its contact rows, `spout_scaling.score_centre_rows`); pre band =
    each pre session's R rows at ``pos`` by its leave-one-out model; post = the post sessions' R rows at ``pos`` by the
    full model. STORED (kind g_lickonly_decoder; the align carries "ffwork" + the run's selection tag, so its ids never
    meet the engaged-row arm's)."""
    from wfield_local import spout_scaling as _ss
    from wfield_local import trial_classes as _tc
    cg = _camera_groups(d)
    if cg is None:
        return None
    we, wu = cg["work_e"], cg["work_u"]
    RX = np.vstack([d["XE"][we], d["XU"][wu]])
    RY = np.concatenate([d["YE"][we], d["YU"][wu].astype(d["YE"].dtype)])
    RG = np.concatenate([d["GE"][we], d["GU"][wu]])
    RC = np.concatenate([cg["grp_e"][we], cg["grp_u"][wu]])
    kR = np.isin(RY, pos)
    pre_w = sorted(i for i in d["pre_i"] if np.any((RG == i) & kR))

    def _store(fit_all):
        got = _g_store(d, "g_lickonly_decoder", f"{_keep_tag(pos)}-ffwork-{_tc.run_selection()}", fit_all,
                       meta={"n_train": int((np.isin(RG, pre_w) & kR).sum()), "trials": "camera_working"},
                       scaling=sc)
        return got if got is not None else fit_all()

    ff = _ss.FrozenFrame(RX, RY, RG, pre_w, _pipe, classes=list(pos), train_mask=kR, loso=True, store=_store,
                         cls_e=RC, labels=dict(enumerate(d["kept"])))
    # THE SCORED-FRAME CENTRE KIND (`spout_scaling.score_centre_rows`), on this arm's ``pos``-spout model: wt = the
    # true-label balanced centre over the ``pos`` spouts' working rows (its training frame is balanced over ``pos``
    # too; >= 20 rows per spout or it refuses), working = the label-free EM (FrozenFrame's own), contact = the EM on
    # the contact rows; cto / n are not defined for a spout-subset model and refuse
    kind = _ss.score_centre_rows()
    if ff.scaling:
        if kind == "contact":
            ff.cm = RC == "contact"
        elif kind == "wt":
            pos_sorted = tuple(sorted(int(c) for c in pos))

            def _wt(i, which, _p=pos_sorted):
                m = (RG == i) & np.isin(RY, _p)
                if not m.any():
                    return None
                mu_, _sd = _ss.true_label_centre(RX[m], np.searchsorted(_p, RY[m]), d["kept"][int(i)],
                                                 n_classes=len(_p))
                return mu_, __import__("wfield_local.label_shift", fromlist=["x"]).session_sd(RX[RG == i])
            ff.centre_fn = _wt
        elif kind != "working":
            raise ValueError(f"{_ss.SCORE_CENTRE_ENV}={kind}: not defined for the lick-only arm's spout-subset model")
    pre_y, pre_g, pre_p = [], [], []
    for i in pre_w:
        if i not in ff.loso:
            continue
        m = (RG == i) & kR
        pre_y.append(RY[m]); pre_g.append(RG[m])
        pre_p.append(ff.loso[i].predict(ff.apply(RX[m], i, "loso")))
    post = (~np.isin(RG, list(d["pre_i"]))) & np.isin(RG, list(d["post_i"])) & kR
    Xpost = ff.apply_rows(RX[post], RG[post]) if post.any() else RX[post]
    cat = (lambda a, dt: np.concatenate(a) if a else np.zeros(0, dt))
    return (ff.full, cat(pre_p, RY.dtype), (cat(pre_y, RY.dtype), cat(pre_g, RG.dtype)), (Xpost, RY[post]))


def looks_like_which(d, keep, seed=0):
    """Do POST-stroke no-lick trials resemble PRE-stroke LICKING or PRE-stroke NON-LICKING trials?

    A discriminator trained on pre-stroke engaged vs undetected would otherwise learn POSITION --
    undetected trials are overwhelmingly far pre-stroke, and post-stroke no-lick trials are far too,
    so an unmatched classifier would answer "far" and look like an answer. The two training classes
    are therefore position-BALANCED: within each kept position, both classes are subsampled to the
    same count, so position carries no information about the label.
    """
    rng = np.random.RandomState(seed)
    pre_e = np.isin(d["GE"], list(d["pre_i"])) & np.isin(d["YE"], keep)
    pre_u = np.isin(d["GU"], list(d["pre_i"])) & np.isin(d["YU"], keep)
    post_u = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], keep)
    if pre_u.sum() < 30 or post_u.sum() < 20:
        return {"note": "too few no-lick trials to discriminate",
                "n_pre_undetected": int(pre_u.sum()), "n_post_undetected": int(post_u.sum())}
    # OPTIONAL STATE-CLASS bal_em (WIDEFIELD_STATE_DISC_SCALING=state_bal_em; DEFAULT OFF, 11-03 -- it fails this
    # discriminator's own pre-stroke check, see `spout_scaling.state_disc_scaling`): train on the state-balanced pre
    # frame, score each post session in its label-free EM frame (`_lick_train` / `_lick_scored`); pre rows stay in the
    # training frame. The sample draws the SAME indices either way (`rng` sees only counts). Identity when OFF.
    dT = _lick_train(d, keep)
    sample = balanced_lick_sample(dT, keep, rng)
    if sample is None:
        return {"note": "no position had >=5 of both classes"}
    X, y, grp = sample
    # STORED BY DEFAULT (kind g_lick_disc, 11-02 pass 2): the fit on this fresh-RandomState(seed) sample (on the state
    # training frame `dT`, main 11-03; its scaling enters the spec)
    from wfield_local import spout_scaling as _ss
    clf = _g_store(d, "g_lick_disc", f"{_keep_tag(keep)}-seed{seed}", lambda: lick_pipe().fit(X, y),
                   meta={"n_train": int(len(y))}, scaling=_ss.state_disc_scaling())
    if clf is None:
        clf = lick_pipe().fit(X, y)
    XEs, XUs = _lick_scored(d, keep, clf, y, sorted(d["post_i"]), dT)
    # the pre reference rows: the sessions the frame kept (all of them unless spout_frame dropped a no-camera one)
    pre_e = np.isin(d["GE"], list(dT["pre_i"])) & np.isin(d["YE"], keep)
    pre_u = np.isin(d["GU"], list(dT["pre_i"])) & np.isin(d["YU"], keep)
    # how well does it separate the two PRE-stroke states at all? without this the post-stroke
    # answer is uninterpretable -- a coin-flip discriminator says nothing.
    # GROUPED by session. An ungrouped cv=5 puts trials from one session on both sides of a fold, and
    # session identity is exactly what a whole-brain feature vector can memorise; every other CV in
    # this pipeline groups for that reason. Falls back to ungrouped only if the balanced training set
    # somehow spans fewer than two sessions, and says so in the output.
    def _mk():
        return lick_pipe()

    # BOTH estimates, because they answer different questions and the grouped one can be too coarse
    # here: GroupKFold holds out a whole SESSION, and with ~11 pre-stroke sessions a fold can lose an
    # entire position's worth of one class. Ungrouped = "can these two states be separated at all";
    # grouped = "can they be separated in a session the model has never seen", which is the
    # conservative number and the one to quote when the answer must generalise across days.
    sep_trial = float(accuracy_score(y, cross_val_predict(_mk(), X, y, cv=5)))
    ngrp = len(np.unique(grp))
    if ngrp >= 2:
        sep_session = float(accuracy_score(y, cross_val_predict(
            _mk(), X, y, cv=GroupKFold(n_splits=min(5, ngrp)), groups=grp)))
        sep_grouped = True
    else:
        sep_session = float("nan")
        sep_grouped = False
    sep = sep_session if sep_grouped else sep_trial
    p_post = float(clf.predict(XUs[post_u]).mean())

    # THE CONTROL THIS TEST IS WORTHLESS WITHOUT. The boundary is trained entirely on PRE-stroke
    # data, so post-stroke trials are out of distribution: any global post-lesion shift could push
    # them to one side whatever their trial state. Applying the SAME boundary to post-stroke ENGAGED
    # trials says whether it still tracks licking after the lesion. If engaged and undetected land at
    # the same rate, the classifier is reporting "post-stroke", not "licking", and the headline
    # number means nothing.
    post_e = np.isin(d["GE"], list(d["post_i"])) & np.isin(d["YE"], keep)
    p_post_eng = float(clf.predict(XEs[post_e]).mean()) if post_e.sum() >= 20 else float("nan")
    # ALL pre-stroke trials of each class, not the position-balanced training subsample -- these two
    # are the scale the post-stroke rate is read against, so they must describe the population and
    # not the sample the boundary was drawn from. (They were `Xe`/`Xu`, bound by the sampling loop
    # that `balanced_lick_sample` replaced on 2026-08-28; caught by ruff F821, which is why that
    # guard exists -- an undefined name here fails hours into a run, not at import.)
    pre_e_held = float(clf.predict(XEs[pre_e]).mean())   # pre-stroke engaged, for scale
    pre_u_held = float(clf.predict(XUs[pre_u]).mean())   # pre-stroke undetected, for scale
    gap = p_post_eng - p_post if np.isfinite(p_post_eng) else float("nan")

    # Interval on the CONTROL gap. It decides whether the headline number may be read at all, so a
    # point estimate either side of a 0.10 threshold was never enough. Stratified bootstrap: resample
    # the post-stroke engaged and no-lick arms independently, since they are separate samples.
    gap_ci = [float("nan"), float("nan")]
    if np.isfinite(p_post_eng):
        pe = clf.predict(XEs[post_e]).astype(float)
        pu = clf.predict(XUs[post_u]).astype(float)
        boots = [pe[rng.randint(0, len(pe), len(pe))].mean()
                 - pu[rng.randint(0, len(pu), len(pu))].mean() for _ in range(2000)]
        gap_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    return {"n_train_per_class": int((y == 1).sum()), "n_post_undetected": int(post_u.sum()),
            "pre_separability_cv": sep,
            "pre_separability_cv_grouped_by_session": sep_grouped,
            "pre_separability_cv_session_grouped": sep_session,
            "pre_separability_cv_trialwise": sep_trial,
            "n_train_sessions": int(ngrp),
            "control_gap_ci95": gap_ci,
            "post_undetected_frac_classified_ENGAGED_like": p_post,
            "CONTROL_post_engaged_frac_engaged_like": p_post_eng,
            "n_post_engaged": int(post_e.sum()),
            "scale_pre_engaged": pre_e_held, "scale_pre_undetected": pre_u_held,
            "engaged_minus_undetected_post": gap,
            # the WHOLE interval must clear zero: a positive point estimate whose interval spans
            # zero is not a control that passed, it is a control that was not measured precisely
            # enough to say either way
            "state_disc_frame": _state_frame_tag(),
            "boundary_still_discriminates_post": bool(
                np.isfinite(gap) and gap > 0.10 and gap_ci[0] > 0),
            "reads_as": ("engaged-like (licking)" if p_post > 0.5 else "undetected-like (non-licking)")}



def fits_engaged_distribution(d, keep, seed=0, n_boot=2000):
    """Does the POST-stroke no-lick session fall inside the PRE-stroke ENGAGED distribution?

    See the module note on `looks_like_which`: that function's control assumes post-stroke engaged and
    no-lick trials should separate, which is precisely what the execution-failure hypothesis denies.
    This one makes no such assumption. The reference distributions come from PRE-stroke sessions, where
    the truth is known, and the post-stroke session is placed against them.

    Returns per-session reference values, their ranges, and where the post-stroke value sits.
    """
    rng = np.random.RandomState(seed)
    pre_e_all = np.isin(d["GE"], list(d["pre_i"])) & np.isin(d["YE"], keep)
    pre_u_all = np.isin(d["GU"], list(d["pre_i"])) & np.isin(d["YU"], keep)
    # OPTIONAL STATE-CLASS bal_em (as `looks_like_which`; DEFAULT OFF): training frame for the fits, each scored
    # session (a held-out pre session, or the post session) in its own label-free EM frame. Identity when OFF.
    dT = _lick_train(d, keep)

    def balanced_fit(exclude_session=None):
        """Position-balanced engaged-vs-no-lick discriminator, optionally holding out one session.

        ONE `rng` ACROSS EVERY CALL, deliberately preserved in the 2026-08-28 de-duplication: each
        leave-one-out fold advances the generator, so the fits here are independent draws rather
        than one sample reused, and the full-pool fit below is NOT the same sample that
        `looks_like_which` and `undetected_state_split` take (each of those starts a fresh
        RandomState(seed)). Making this deterministic per fold would be defensible and would move
        published numbers, so it is a decision to take separately rather than a side effect of
        removing duplication.
        """
        sample = balanced_lick_sample(dT, keep, rng, exclude_session=exclude_session)
        if sample is None:
            return None
        X, y, _grp = sample
        return lick_pipe().fit(X, y), y

    def _fit_folds():
        # THE IN-MEMORY SEQUENCE, in order: every leave-one-session-out fold, then the full pool -- one advancing rng.
        loso = {d["kept"][i]: balanced_fit(exclude_session=i) for i in sorted(dT["pre_i"])}
        full = balanced_fit()
        # the generator's state AFTER every fit: the post-value bootstrap below continues from it
        return {"loso": loso, "full": full, "rng_after": rng.get_state()}

    # STORED BY DEFAULT (kind g_lick_disc_folds, 11-02 pass 2): all fold models + the full one, fitted ONCE in the
    # in-memory order, so the advancing-RNG draws are exactly the in-memory ones; the stored generator state then
    # puts the bootstrap of the post-stroke value on the same draws as the in-memory path.
    folds = _g_store(d, "g_lick_disc_folds", f"{_keep_tag(keep)}-seed{seed}-v2", _fit_folds,
                     scaling=__import__("wfield_local.spout_scaling", fromlist=["x"]).state_disc_scaling())
    if folds is not None:
        rng.set_state(folds["rng_after"])
    ref = {"engaged": {}, "nolick": {}}
    for i in sorted(dT["pre_i"]):
        got = folds["loso"].get(d["kept"][i]) if folds is not None else balanced_fit(exclude_session=i)
        if got is None:
            continue
        clf, _y = got
        hXE, hXU = _lick_scored(d, keep, clf, _y, [i], dT)
        me = pre_e_all & (d["GE"] == i)
        mu = pre_u_all & (d["GU"] == i)
        if me.sum() >= 20:
            ref["engaged"][d["kept"][i]] = float(clf.predict(hXE[me]).mean())
        if mu.sum() >= 10:
            ref["nolick"][d["kept"][i]] = float(clf.predict(hXU[mu]).mean())
    if len(ref["engaged"]) < 3 or len(ref["nolick"]) < 3:
        return {"note": "too few pre-stroke sessions to build a reference distribution",
                "n_engaged_sessions": len(ref["engaged"]), "n_nolick_sessions": len(ref["nolick"])}

    got_all = folds["full"] if folds is not None else balanced_fit()
    clf_all = got_all[0] if got_all is not None else None
    post_u = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], keep)
    post_e = np.isin(d["GE"], list(d["post_i"])) & np.isin(d["YE"], keep)
    if post_u.sum() < 10 or clf_all is None:
        return {"note": "too few post-stroke no-lick trials", "n_post_nolick": int(post_u.sum())}
    pXE, pXU = _lick_scored(d, keep, clf_all, got_all[1], sorted(d["post_i"]), dT)
    pred_u = clf_all.predict(pXU[post_u]).astype(float)
    P = float(pred_u.mean())

    out = {"post_value": P, "n_post_nolick": int(post_u.sum()), "state_disc_frame": _state_frame_tag(),
           "post_engaged_value": float(clf_all.predict(pXE[post_e]).mean())
                                 if post_e.sum() >= 20 else float("nan"),
           "reference_engaged_per_session": ref["engaged"],
           "reference_nolick_per_session": ref["nolick"]}
    # a bootstrap interval on the post-stroke value itself, so the comparison is interval-to-interval
    boots = [pred_u[rng.randint(0, len(pred_u), len(pred_u))].mean() for _ in range(n_boot)]
    out["post_value_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    for arm in ("engaged", "nolick"):
        v = np.array(list(ref[arm].values()), float)
        lo, hi = float(v.min()), float(v.max())
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        out[arm] = {"n_sessions": len(v), "mean": mu, "sd": sd, "min": lo, "max": hi,
                    "z_of_post": (P - mu) / sd if sd else float("nan"),
                    "post_inside_range": bool(lo <= P <= hi),
                    # fraction of pre-stroke sessions this post value exceeds
                    "percentile_of_post": float((v < P).mean())}
    ie, inl = out["engaged"]["post_inside_range"], out["nolick"]["post_inside_range"]
    e_lo, e_hi = out["engaged"]["min"], out["engaged"]["max"]
    n_lo, n_hi = out["nolick"]["min"], out["nolick"]["max"]
    # Where P sits BETWEEN the two references is a real and common state, and the first version of
    # this verdict collapsed it into "OFF-SCALE", which reads as a failed measurement. PS94 pre-cue
    # is exactly this: 0.662, above every pre-stroke no-lick session and below every engaged one.
    between = (not ie) and (not inl) and (n_hi < P < e_lo or e_hi < P < n_lo)
    if between:
        span = abs(e_lo - n_hi) if n_hi < e_lo else abs(n_lo - e_hi)
        frac = (P - n_hi) / span if (span and n_hi < e_lo) else float("nan")
        out["fraction_of_gap_toward_engaged"] = float(frac)
    out["verdict"] = (
        "post-stroke NO-LICK trials fall inside the pre-stroke ENGAGED distribution and OUTSIDE the "
        "pre-stroke no-lick one -> indistinguishable from successful trials: plan intact, execution "
        "failed" if (ie and not inl) else
        "post-stroke no-lick trials look like ordinary pre-stroke FAILURES" if (inl and not ie) else
        "AMBIGUOUS: the post value sits inside both reference distributions -- they overlap too much "
        "to separate" if (ie and inl) else
        (f"INTERMEDIATE: above every pre-stroke no-lick session and below every engaged one "
         f"({out.get('fraction_of_gap_toward_engaged', float('nan')):.0%} of the way toward engaged) "
         f"-> shifted toward successful trials but not indistinguishable from them") if between else
        "OFF-SCALE: outside BOTH pre-stroke distributions and outside the gap between them, so "
        "neither reference describes it")
    return out

def pattern_similarity(d, keep, post_all_trials=True):
    """POST arm uses ALL trials by default and is scored over every position -- see decode_matched.

    Per-position mean-pattern correlation between phases -- is the code the SAME code?

    ``r_pre_loo`` IS THE CEILING AND ``r`` MUST NOT BE READ WITHOUT IT (added 2026-08-25). A bare
    pre-vs-post correlation invites comparison with 1.0, and 1.0 is unreachable: two mean patterns
    estimated on DIFFERENT DAYS differ by ordinary day-to-day drift and by however noisily each was
    estimated, in a healthy animal with no lesion at all. ``r_pre_loo`` measures exactly that -- each
    pre-stroke session's own mean pattern against the pooled mean of the OTHER pre-stroke sessions,
    averaged -- so it is the value ``r`` would take if the lesion changed nothing. On PS94's post-cue
    window the equivalent quantity in the grant figures is ~0.75-0.86, not 1.0, and reading a post
    value of 0.7 against 1.0 rather than against 0.8 turns "unchanged" into "30% lost".

    ONE SESSION AGAINST A POOL, deliberately, because that is the shape of the post comparison too.
    Splitting the pre-stroke TRIALS at random instead would put both sides on the same days and
    measure only trial noise, giving a ceiling no across-day comparison can reach -- the error this
    docstring exists to prevent recurring.
    """
    # ALL six positions when the post arm is all-trials: restricting to `keep` would silently drop
    # far_center and far_R for PS94, which are the positions this comparison most needs to describe.
    # THE PROJECTION FRAME (main 26fd5b44, Priya 11-03, item C; `_proj_X`): identity when off
    _PX, _PU = _proj_X(d)
    pos = list(DISPLAY_ORDER) if post_all_trials else list(keep)
    out = {}
    pre_sessions = sorted(d["pre_i"])
    for c in pos:
        a = _PX[np.isin(d["GE"], list(d["pre_i"])) & (d["YE"] == c)]
        b = _PX[np.isin(d["GE"], list(d["post_i"])) & (d["YE"] == c)]
        if post_all_trials:
            bu = _PU[np.isin(d["GU"], list(d["post_i"])) & (d["YU"] == c)]
            if len(bu):
                b = np.vstack([b, bu]) if len(b) else bu
        if len(a) < 5 or len(b) < 5:
            continue
        ma, mb = a.mean(0), b.mean(0)
        loo = []
        for s in pre_sessions:
            h = _PX[(d["GE"] == s) & (d["YE"] == c)]
            r = _PX[np.isin(d["GE"], [x for x in pre_sessions if x != s]) & (d["YE"] == c)]
            if len(h) >= 5 and len(r) >= 5:
                mh, mr = h.mean(0), r.mean(0)
                if np.std(mh) and np.std(mr):
                    loo.append(float(np.corrcoef(mh, mr)[0, 1]))
        out[POSITION_NAMES[c]] = {"r": float(np.corrcoef(ma, mb)[0, 1]),
                                  "r_pre_loo": (float(np.mean(loo)) if loo else None),
                                  "n_pre_sessions": len(loo),
                                  "n_pre": len(a), "n_post": len(b)}
    return out




def crossed_confusion(d, labels=DISPLAY_ORDER, post_all_trials=False,
                      include_nolick=True):
    """Full confusion of the FROZEN pre-stroke decoder applied to post-stroke trials.

    Scalar accuracy discards what a lateralised lesion should produce: WHERE the errors go. A
    contralesional deficit that shifts far_R trials onto far_L looks identical, in accuracy, to one
    that scatters them. The 6x6 distinguishes those, and its diagonal is the per-position recall
    table, so the two are one object rather than two.

    Rows are TRUE position, columns PREDICTED, row-normalized. Rows with no post-stroke trials are
    returned as NaN rather than zeros -- PS94 has none at far_center or far_R, and a row of zeros
    would read as "always wrong" instead of "never attempted".

    ``include_nolick=False`` drops the two no-lick panels. Pass it for LICK-ALIGNED data: a trial
    with no lick has no lick to align to, so a "no-lick, lick-aligned" panel is not a weak result
    but an undefined one. The caller's other loop has guarded this since it was written ("no lick,
    no alignment point"); this argument lets the confusion loop guard it too.
    """
    tr = np.isin(d["GE"], list(d["pre_i"]))
    te = np.isin(d["GE"], list(d["post_i"]))
    # THE STORED MODEL. No class filter and no restriction on `tr`, so this is the frozen decoder
    # exactly. It matters most here: `poststroke_section_g` mutates only `d["post_i"]` and calls
    # back in once per post-stroke session per arm per alignment, and the pre-stroke fit -- which
    # does not depend on `post_i` at all -- was redone on every one of those passes.
    models = frozen(d)
    clf = models["full"] if models else _pipe().fit(d["XE"][tr], d["YE"][tr])
    # the frame the STORED models score in (`frozen_X`; bal_em since 2026-11-03); production frame for local fits
    XEf, XUf = frozen_X(d) if models else (d["XE"], d["XU"])

    # PRE: engaged only, leave-one-session-out -- the reference for a code with a successful movement.
    if models:                  # pre sessions without a LOSO model (no camera classes) have no pre panel rows
        tr = tr & np.isin(d["GE"], [i for i in d["pre_i"] if d["kept"][i] in models["loso"]])
    # UNDER THE CAMERA GROUPING (main 4a37cbc8, Priya 11-04, item G; `_camera_groups`) the pre panel is the pre sessions'
    # camera WORKING trials (contact + incomplete, engaged AND no-lick rows), each scored by its LOSO model.
    cg = _camera_groups(d) if models else None
    pre_y = d["YE"][tr]
    if models:
        pre_p = np.empty_like(pre_y)
        _gi = d["GE"][tr]
        for i in sorted(d["pre_i"]):
            m = _gi == i
            if m.sum():
                pre_p[m] = models["loso"][d["kept"][i]].predict(XEf[tr][m])
    else:
        pre_p = cross_val_predict(_pipe(), d["XE"][tr], d["YE"][tr], cv=LeaveOneGroupOut(),
                                  groups=d["GE"][tr])
    pre_nolick = np.zeros(len(pre_y), bool)
    if cg is not None:
        _lo = [i for i in d["pre_i"] if d["kept"][i] in models["loso"]]
        pre_y, pre_p, pre_nolick = _loso_rows(d, models, XEf, XUf, np.isin(d["GE"], _lo) & cg["work_e"],
                                              np.isin(d["GU"], _lo) & cg["work_u"])

    # PRE-NO-LICK: the matched control (Priya, 2026-08-19). Comparing post-stroke NO-LICK trials
    # against a pre-stroke ENGAGED panel confounds the lesion with the absence of a movement -- the
    # post rows have no lick and the pre rows do. Pre-stroke no-lick trials are the same KIND of
    # trial, so this panel differs from the post panel in phase alone.
    #
    # Trained on the OTHER pre-stroke sessions' engaged trials and applied to this session's no-lick
    # ones, which mirrors the post panel exactly (frozen on pre-stroke engaged, applied to no-lick).
    # Training on engaged trials of the SAME session would let the decoder see that session's noise.
    pre_u = np.isin(d["GU"], list(d["pre_i"]))
    pre_nl_y, pre_nl_p = d["YU"][pre_u], np.full(int(pre_u.sum()), -1)
    if pre_u.sum():
        for gsess in np.unique(d["GU"][pre_u]):
            hold = d["GU"][pre_u] == gsess
            trn = tr & (d["GE"] != gsess)
            if trn.sum() < 30 or not hold.sum():
                continue
            # `models["loso"][label]` IS this fit -- all pre-stroke engaged trials except this
            # session's -- keyed by label rather than by the pooled index, which depends on the
            # order the pool was assembled and this does not.
            mdl = (models["loso"].get(d["kept"][int(gsess)]) if models else None)
            if models and mdl is None:
                continue              # no LOSO model: dropped from the camera-working training set (no video)
            if mdl is None:
                mdl = _pipe().fit(d["XE"][trn], d["YE"][trn])
                pre_nl_p[hold] = mdl.predict(d["XU"][pre_u][hold])
            else:
                pre_nl_p[hold] = mdl.predict(XUf[pre_u][hold])
    keep_nl = pre_nl_p >= 0
    pre_nl_y, pre_nl_p = pre_nl_y[keep_nl], pre_nl_p[keep_nl]

    # POST: engaged, or engaged + no-lick when post_all_trials. The no-lick trials are the ONLY ones
    # that exist at an abandoned position, so without them those rows cannot be filled at all.
    if post_all_trials:
        teu = np.isin(d["GU"], list(d["post_i"]))
        post_y = np.concatenate([d["YE"][te], d["YU"][teu]])
        post_X = np.vstack([XEf[te], XUf[teu]])
        post_nolick = np.concatenate([np.zeros(int(te.sum()), bool), np.ones(int(teu.sum()), bool)])
    elif cg is not None:
        # the engaged-only arm under the camera grouping: the post session's camera WORKING trials
        teu = np.isin(d["GU"], list(d["post_i"])) & cg["work_u"]
        te = te & cg["work_e"]
        post_y = np.concatenate([d["YE"][te], d["YU"][teu].astype(d["YE"].dtype)])
        post_X = np.vstack([XEf[te], XUf[teu]])
        post_nolick = np.concatenate([np.zeros(int(te.sum()), bool), np.ones(int(teu.sum()), bool)])
    else:
        post_y, post_X = d["YE"][te], XEf[te]
        post_nolick = np.zeros(len(post_y), bool)
    post_p = clf.predict(post_X)

    # POST-NO-LICK: the post-stroke no-lick trials ON THEIR OWN, which is the panel that pairs with
    # `pre_nolick`. The all-trials `post` matrix MIXES engaged and no-lick rows, so it cannot answer
    # "what does the code look like when no movement happened" -- it records how many of each row
    # were no-lick (n_nolick_per_true_position) but not how those trials alone were classified.
    # Computed in both arms: the trials exist regardless of which arm the rest of the record uses.
    post_u = np.isin(d["GU"], list(d["post_i"]))
    post_nl_y = d["YU"][post_u]
    post_nl_p = clf.predict(XUf[post_u]) if post_u.sum() else np.empty(0, dtype=post_nl_y.dtype)

    out = {"post_arm": ("ALL trials (engaged + no-lick)" if post_all_trials else
                        f"camera WORKING trials ({_work_txt()})" if cg is not None else "engaged only"),
           "pre_arm": (f"camera WORKING trials ({_work_txt()}), leave-one-session-out" if cg is not None
                       else "engaged only, leave-one-session-out"),
           "groups": "camera" if cg is not None else "daq",
           "pre_nolick_arm": "PRE-stroke NO-LICK trials, decoder trained on other sessions' engaged",
           "post_nolick_arm": "POST-stroke NO-LICK trials only, frozen pre-stroke decoder"}
    phases = [("pre", pre_y, pre_p, pre_nolick), ("post", post_y, post_p, post_nolick)]
    if include_nolick:
        phases[1:1] = [("pre_nolick", pre_nl_y, pre_nl_p, np.ones(len(pre_nl_y), bool))]
        phases.append(("post_nolick", post_nl_y, post_nl_p, np.ones(len(post_nl_y), bool)))
    else:
        out.pop("pre_nolick_arm", None)
        out.pop("post_nolick_arm", None)
    for phase, y, p, nl in phases:
        out[phase] = _conf_panel(y, p, nl, labels)
    if cg is not None:
        out["by_class"] = _by_class_panels(d, models, clf, XEf, XUf, cg, labels)
    return out


#: POST-STROKE ENGAGEMENT FILTERING IS RETIRED (Priya, 2026-08-18).
#:
#: There is no valid post-stroke construction of "disengaged", so no analysis may split on one.
#:
#: The pre-stroke gate was invalid for the obvious reason (it reads motor failure as lost motivation
#: -- it called 59% of PS94 8/17 disengaged against 6.8% for a reference-position gate). The
#: reference-position gate that replaced it is better but ALSO not validated: its 29 "disengaged"
#: PS94 trials are undetected trials falling where the trailing response rate at close_L/close_center
#: dipped below 0.5 over 15 reference trials, and a short run of MOTOR failures produces that dip
#: just as readily as a motivational lapse. Nothing in the spout data distinguishes them.
#:
#: And the construction has no general form: in a severe stroke EVERY position may be impaired, so
#: there is no spared reference to anchor engagement on and spout contact cannot define it at all.
#:
#: Consequence for results already computed: `undetected_state_split`'s comparison class was never
#: established, so its output (PS94 working-minus-disengaged -0.060) is UNINTERPRETABLE, not
#: negative. It must not be quoted as evidence for a global post-stroke shift -- which is how I first
#: read it. PS95 returned UNDECIDABLE (0 disengaged) and that reading was right for the wrong reason.
#:
#: Post-stroke analyses therefore use ALL trials (nolick_analysis.SANCTIONED_MISMATCHES). The
#: question the split was meant to answer -- execution failure vs disengagement -- is better asked
#: WITHIN the post-stroke session by contrasting undetected trials at IMPAIRED vs PRESERVED
#: positions, which needs no engagement label. DLC tongue-protrusion data settles it directly.
POSTSTROKE_ENGAGEMENT_FILTERING = False


def poststroke_engagement(s, reference_positions, window=15, min_rate=0.5):
    """Engagement judged ONLY at positions the animal can still reach (Priya, 2026-08-18).

    THE PRE-STROKE GATE IS INVALID AFTER A LESION. `flag_engagement` calls an animal disengaged when
    its trailing response rate collapses -- and post-stroke that collapses BECAUSE the animal cannot
    reach the far positions. It would therefore label motor failure as disengagement, i.e. label the
    effect being measured as the confound, and every "no plan formed" conclusion downstream would be
    circular.

    Judged instead at the PRESERVED positions (PS94/PS95 8/17: close_L, close_center, close_R,
    far_L). A miss there is genuine disengagement; a miss at an impaired position says nothing about
    motivation. Which positions count is empirical per animal and per session, so it is passed in
    rather than hardcoded.

    RETAINED AS A DESCRIPTIVE STATISTIC ONLY -- see POSTSTROKE_ENGAGEMENT_FILTERING above. Reporting
    "the animal responded on 89% (PS94) / 97% (PS95) of trials at spared positions" is sound and worth
    showing. SPLITTING trials on it is not, because a local dip in that rate is indistinguishable from
    a run of motor failures. Callers that filter on the returned mask are the error this guards.

    Returns (engaged_bool per trial, info). Trials at non-reference positions inherit the state of
    the reference-position trials around them, since that is what the estimate is FOR.

    THE RULE ITSELF NOW LIVES IN ONE PLACE: this delegates to
    `precue_engagement_states.engagement_gate`, which supplies the non-recovery requirement and the
    backdating to the start of the run of misses that trips the gate. ``min_rate`` is therefore no
    longer read (the shared gate owns its own ``MIN_RATE``), and ``window`` survives only as the
    "too few reference trials to gate at all" guard. Both are kept in the signature so existing
    callers do not break; neither now changes the answer.
    """
    from wfield_local import nolick_decoder as nd

    args = nd._args(align="cue")
    # 2.0 s HERE IS NOT THE ENGAGED CUT. It is the boundary nd.categorize needs to produce the
    # THREE-arm split, and the next line unions engaged + late_rewarded -- which IS the task's
    # response window. Reading it as "this module missed the 3.5 s change" is the obvious mistake;
    # raising it to 3.5 would collapse the late arm into engaged and silently change what
    # `responded` means. Spelled out 2026-08-22, when eleven other modules genuinely had missed it.
    args.max_rt = 2.0
    codes, cat, _blk, _rt_s, _cue_f, _pre_gate = nd.categorize(s, args)
    codes = np.asarray(codes)
    responded = np.array([c in ("engaged", "late_rewarded") for c in cat], bool)
    from wfield_local import gate_source  # [gate-source hook] WIDEFIELD_GATE_SOURCE (see that module)
    responded = gate_source.gate_responded(s["label"], np.arange(codes.size), responded)  # [gate-source hook]
    is_ref = np.isin(codes, list(reference_positions))

    eng = np.ones(codes.size, bool)
    from wfield_local import trial_classes as _tc
    if _tc.camera_gate_on():
        # CAMERA TRIAL GATE (rebuild 11-03, `trial_classes`): engaged = the cue's camera class is in GATES["working"],
        # per trial (no reference positions, no quit tail). ``info["stopped"]`` = class in STOPPED_CLASSES (a SUSTAINED
        # stop, Priya 11-03); isolated no_response and "unknown" cues are neither: take stopped from there, never ~eng.
        from wfield_local.plot_spout_trial_averages import _load_daq_events
        cue = _load_daq_events(s["h5"])
        ccls = _tc.cue_classes_for(s["label"], np.asarray(cue["cue_samples"], float) / float(cue["sample_rate_hz"]),
                                   window="cue_stop")       # the no-lick definition: cue_stop (Priya FINAL 11-03)
        if ccls.size != codes.size:
            raise RuntimeError(f"camera trial gate: {ccls.size} classified cues vs {codes.size} categorized")
        eng = _tc.in_sel(ccls, "miss_working")   # its callers split the NO-LICK arm: no contact class (review M6)
        stopped = np.isin(ccls, list(_tc.STOPPED_CLASSES))
        return eng, {"gate": f"trial_classes camera ({_tc.run_selection()})", "stopped": stopped,
                     "frac_working": float(eng.mean()) if eng.size else None,
                     "frac_stopped": float(stopped.mean()) if stopped.size else None}
    ref_idx = np.flatnonzero(is_ref & (codes >= 0))
    if ref_idx.size < window:
        return eng, {"note": "too few reference-position trials to gate", "n_ref": int(ref_idx.size)}
    # DELEGATED 2026-09-08 (Priya: "replace poststroke_engagement to be consistent with the same
    # engagement gate we use everywhere else"). This used to carry its own copy of the rule -- a
    # rolling reference-rate threshold with no non-recovery requirement and no backdating -- so it
    # agreed with `engagement_gate` on the easy sessions and diverged on exactly the hard ones:
    # a dip that recovers tripped it, and the run of misses that trips it stayed "engaged" up to
    # the crossing point. Same inputs, one definition. `engagement_gate` compares with `np.isin`,
    # so the integer position codes used here work unchanged.
    from wfield_local.precue_engagement_states import engagement_gate
    not_eng = engagement_gate(np.arange(codes.size), responded, codes,
                              reference=tuple(reference_positions))
    eng = ~np.asarray(not_eng, bool)
    return eng, {"n_ref": int(ref_idx.size), "ref_response_rate": float(responded[ref_idx].mean()),
                 "frac_disengaged": float((~eng).mean()),
                 "gate": "precue_engagement_states.engagement_gate",
                 "reference_positions": sorted(int(c) for c in reference_positions)}


def nolick_state_vector(kept, args, reference_positions=None, with_stopped=False):
    """Engagement state + lick category for every trial in the pooled NO-LICK arm, in pool order.

    ``with_stopped=True`` appends the STOPPED mask: ``~state`` under the DAQ rule, the camera STOPPED_CLASSES under the
    camera trial gate (rebuild 11-03), where a row can be neither ("unknown").

    Built by asking `_trial_features` which cues it kept (``with_indices``) and indexing the
    per-cue arrays at those positions -- never by rebuilding the filter. That reconciliation is
    exact: PS94 8/17's no-lick arm is 354 = 353 undetected + 1 late_rewarded, PS95's 273 = 265 + 8,
    which is precisely the discrepancy that made the first attempt refuse to run.
    """
    from wfield_local import nolick_decoder as nd
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.locanmf_position_decoder import _trial_features

    refs = REFERENCE_POSITIONS if reference_positions is None else reference_positions
    eng, cats, stop = [], [], []
    for lab in kept:
        s_ = next((x for x in SESSIONS if x["label"] == lab), None)
        if s_ is None:
            continue
        *_rest, _idx_e, idx_nl = _trial_features(s_, args, with_indices=True)
        ca = nd._args(align=args.align)
        ca.max_rt = args.max_rt
        _codes, cat, _blk, _rt, _cf, _pre = nd.categorize(s_, ca)
        st, _info = poststroke_engagement(s_, refs)
        eng.append(np.asarray(st)[idx_nl])
        cats.append(np.asarray(cat, dtype=object)[idx_nl])
        stop.append(np.asarray(_info["stopped"] if "stopped" in _info else ~np.asarray(st, bool))[idx_nl])
    out = (np.concatenate(eng) if eng else np.array([], bool),
           np.concatenate(cats) if cats else np.array([], dtype=object))
    if with_stopped:
        out = out + (np.concatenate(stop).astype(bool) if stop else np.array([], bool),)
    return out


def undetected_state_split(d, keep, args, seed=0):
    """THE ARBITRATING TEST: is the engaged-like reading real, or a global post-stroke shift?

    Priya (2026-08-18) read the numbers the other way round from me and was right: post-stroke
    undetected trials classify at 0.71-0.86 against pre-stroke undetected at 0.085-0.147, so they do
    not resemble unengaged trials. My control asked whether the boundary separates post-stroke
    engaged from post-stroke DISENGAGED -- but under a valid gate there are almost no disengaged
    trials to separate (PS95: none at all), so the absence of separation was the absence of a second
    class, not a blind boundary.

    This splits the post-stroke no-lick trials by the REFERENCE-POSITION gate (close_L,
    close_center -- positions a left VLS lesion should spare) and scores each subset:

        working trials engaged-like AND disengaged trials lower -> execution failure, boundary works
        both equally high                                       -> global shift, not interpretable
        too few disengaged trials                               -> UNDECIDABLE, and reported as such
    """
    base = looks_like_which(d, keep, seed=seed)
    if "post_undetected_frac_classified_ENGAGED_like" not in base:
        return base
    # A FRESH RandomState(seed), so this is the SAME sample `looks_like_which` drew a few lines
    # above -- the two really are one model, refitted, and the de-duplication preserves that.
    rng = np.random.RandomState(seed)
    dT = _lick_train(d, keep)        # optional state-class bal_em (default OFF), as `looks_like_which`
    sample = balanced_lick_sample(dT, keep, rng)
    if sample is None:
        # The old code called np.vstack([]) here and raised. It survived only because
        # `looks_like_which` returns early on the same condition and its result is checked above --
        # a guard held in place by a caller rather than by this function.
        return dict(base, split={"note": "no position had >=5 of both classes"})
    X, y, _grp = sample
    # the SAME stored model `looks_like_which` loaded (same sample, same spec; kind g_lick_disc)
    clf = _g_store(d, "g_lick_disc", f"{_keep_tag(keep)}-seed{seed}", lambda: lick_pipe().fit(X, y),
                   meta={"n_train": int(len(y))},
                   scaling=__import__("wfield_local.spout_scaling", fromlist=["x"]).state_disc_scaling())
    if clf is None:
        clf = lick_pipe().fit(X, y)

    # stopped: ~state under the DAQ rule (unchanged); the camera STOPPED_CLASSES under the camera trial gate (11-03)
    state, cats, stopped_v = nolick_state_vector(d["kept"], args, with_stopped=True)
    out = dict(base)
    if state.size != d["GU"].size:
        out["split"] = {"note": f"state vector {state.size} vs pooled no-lick {d['GU'].size}; "
                                f"refusing to pair them"}
        return out
    post_m = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], keep)
    _XEs, XUs = _lick_scored(d, keep, clf, y, sorted(d["post_i"]), dT)
    p = clf.predict(XUs)
    w = post_m & state
    dis = post_m & stopped_v
    res = {"n_working": int(w.sum()), "n_disengaged": int(dis.sum()),
           "n_late_rewarded_in_arm": int((cats[post_m] == "late_rewarded").sum())}
    if w.sum() >= 10:
        res["working_frac_engaged_like"] = float(p[w].mean())
    if dis.sum() >= 10:
        res["disengaged_frac_engaged_like"] = float(p[dis].mean())
    if "working_frac_engaged_like" in res and "disengaged_frac_engaged_like" in res:
        gap = res["working_frac_engaged_like"] - res["disengaged_frac_engaged_like"]
        res["working_minus_disengaged"] = gap
        res["verdict"] = ("boundary STILL tracks state -> engaged-like WORKING trials are a real "
                          "execution-failure signature" if gap > 0.10 else
                          "no separation -> consistent with a global post-stroke shift")
    else:
        res["verdict"] = ("UNDECIDABLE: too few disengaged trials under the reference-position gate. "
                          "That is itself informative -- the animal was working almost throughout.")
    out["split"] = res
    return out


def impaired_nolick_readout(d, keep, alignment="precue", n_perm=2000):
    """Was a PLAN formed on post-stroke trials where no lick was detected? (Priya, 2026-08-18.)

    This replaces `undetected_state_split`, which needed a "disengaged" label that has no valid
    post-stroke construction (see POSTSTROKE_ENGAGEMENT_FILTERING). The question it was asking is
    answerable WITHIN the post-stroke session and without any engagement label, by splitting the
    no-lick trials on something measured rather than inferred: whether the TRUE position was one the
    animal still reaches (preserved) or one it has stopped reaching (impaired).

    Apply the frozen pre-stroke decoder to post-stroke NO-LICK trials and ask whether it still reads
    out the correct position:

      * above chance at IMPAIRED positions -> the position was represented and the lick did not
        happen: EXECUTION failure with the plan intact. This is the strong result, because those are
        exactly the trials a motor-output account predicts and a "no plan" account does not.
      * at chance everywhere -> no readable plan on no-lick trials; execution failure is not
        supported (though absence of decodable signal is weaker evidence than presence).

    The preserved-position arm is the internal control: no-lick trials there are the animal's own
    baseline lapses at positions it can still reach, so it calibrates the impaired arm within the
    same session, the same recording, and the same decoder.

    CAVEAT that cannot be removed from spout data alone: "no lick detected" is not "no tongue
    protrusion". PS93's rightward bias already produces licks that occur without reaching far_L. DLC
    tongue tracking replaces this inference with a measurement; until then a null here is ambiguous
    between "no plan" and "plan plus a protrusion the spout never saw".
    """
    impaired = [c for c in DISPLAY_ORDER if c not in keep]
    tr = np.isin(d["GE"], list(d["pre_i"]))
    # THE STORED MODEL. `keep` splits the OUTPUT into preserved and impaired arms below; it does not
    # restrict training, so `tr` is all pre-stroke engaged trials and this is the frozen decoder.
    _m = frozen(d)
    clf = _m["full"] if _m else _pipe().fit(d["XE"][tr], d["YE"][tr])
    _XUf = frozen_X(d)[1] if _m else d["XU"]          # the stored model's scoring frame (bal_em since 2026-11-03)
    post_u = np.isin(d["GU"], list(d["post_i"]))
    out = {"preserved_positions": [POSITION_NAMES[c] for c in keep],
           "impaired_positions": [POSITION_NAMES[c] for c in impaired]}
    for name, labs in (("preserved", keep), ("impaired", impaired)):
        m = post_u & np.isin(d["YU"], labs)
        if m.sum() < MIN_POST:
            out[name] = {"n": int(m.sum()), "note": f"fewer than {MIN_POST} no-lick trials"}
            continue
        y, pred = d["YU"][m], clf.predict(_XUf[m])
        # Scored over ALL six labels: the frozen decoder may predict any position, and restricting
        # the label set would hand it a chance level it was not operating under.
        r = na.evaluate_arm(y, pred, n_perm=n_perm, labels=list(DISPLAY_ORDER))
        r["n"] = int(m.sum())
        r["per_position"] = na.per_position_recall(y, pred, labels=list(labs))
        out[name] = r
    a, b = out.get("impaired", {}), out.get("preserved", {})
    if "balanced_accuracy" not in a:
        out["verdict"] = "UNDECIDABLE -- too few no-lick trials at impaired positions"
        return out
    if "balanced_accuracy" in b:
        out["impaired_minus_preserved"] = float(a["balanced_accuracy"] - b["balanced_accuracy"])

    # MULTIPLICITY: this function is run per alignment x arm, so a whole pass is 2 alignments x
    # 2 arms x n_animals tests. A single uncorrected p just under 0.05 is not a result, and the first
    # version of this verdict declared "execution failure" from PS95's p=0.017 (x8 = 0.14).
    p = a.get("bal_p", 1.0)
    out["bal_p_bonferroni_x8"] = float(min(1.0, p * 8))
    survives = out["bal_p_bonferroni_x8"] < 0.05

    if alignment == "precue":
        # The clean index of a PLAN: before the cue there is no movement to explain the signal.
        out["verdict"] = (
            "PRE-CUE position code present on no-lick trials at impaired positions -> plan formed, "
            "movement failed" if survives else
            "no decodable PRE-CUE plan on impaired no-lick trials -> execution failure NOT supported "
            "by this test" + ("" if p >= 0.05 else " once corrected for the 8 tests in a pass"))
    else:
        # POST-cue is NOT a plan readout on these trials. The spout is physically present throughout,
        # so a position-specific sensory response exists whether or not anything was planned, and
        # post-cue decoding is in any case largely lick-driven -- which is why the pre-stroke
        # dissociation used pre-cue survival as the discriminating quantity.
        out["verdict"] = (
            "post-cue position code present on impaired no-lick trials -- AMBIGUOUS: the spout is "
            "present, so this may be a sensory response rather than a plan. Read the pre-cue arm."
            if survives else
            "no post-cue position code on impaired no-lick trials")
    out["verdict"] += ". CAVEAT: 'no lick detected' is not 'no tongue protrusion' -- DLC settles it."
    return out


def _within_accuracy(label, X, y, gb, ng, is_post, pos_key, align):
    """Block-CV within-session accuracy for one session, memoized on ITS OWN inputs.

    Only PRE-STROKE sessions are cached. A post-stroke session's row is the thing under examination
    and is cheap (there are few of them), while the pre-stroke rows are the reference band and are
    recomputed identically every night -- that asymmetry is where the time goes.

    ``pos_key`` is in the cache key because the band is POSITION-MATCHED to the post-stroke arm: the
    same pre-stroke session legitimately has a different accuracy when scored over four positions
    than over six, and serving one for the other would reintroduce the chance-level mismatch this
    function exists to prevent.
    """
    def _compute():
        return float(accuracy_score(
            y, cross_val_predict(_pipe(), X, y, cv=GroupKFold(ng), groups=gb)))

    if is_post:
        return _compute()
    from wfield_local import config, session_cache
    rec = [s for s in config.load_sessions() if s["label"] == label]
    if not rec:                       # unregistered: compute rather than guess at a cache identity
        return _compute()
    # ALIGN IS PART OF THE IDENTITY, and leaving it out was a silent cross-window collision. `X` is
    # the pre-cue, post-cue or post-lick feature window, and the same pre-stroke session has a
    # different within-session accuracy in each. Keyed only on positions and folds, all three hashed
    # to ONE entry, so whichever alignment ran first served the other two: G2c drew the POST-cue band
    # on its pre-cue panel, the plan/execution dissociation the figure exists to show was flattened,
    # and the post-stroke z-scores were measured against the wrong reference.
    #
    # Nothing about the computation was wrong; the KEY was. Every other cached kind here already
    # carries the alignment (`evoked_amp__{align}__...`, `spatial_area_matrix__{align}__...`,
    # `fixed_scale_maps__{align}__...`) -- this one was the exception. `params=None` with everything
    # result-changing folded into `kind` is the house idiom exactly so this stays checkable by eye.
    return session_cache.cached(
        rec[0], f"within_acc__{align}__{'-'.join(str(p) for p in pos_key)}__k{ng}",
        _compute, params=None)


def recoding_test(d, keep, min_trials=40, n_splits=5, post_all_trials=True):
    """Is the position code LOST, or RECODED? Frozen pre-stroke decoder vs a within-session one.

    THE DISTINCTION THIS EXISTS TO DRAW. `decode_matched` shows PS94's frozen pre-stroke decoder
    falling below every pre-stroke session on 8/17, and that was reported as a decoding deficit. But a
    frozen decoder fails for two quite different reasons: the information is gone, or the information
    is there in a DIFFERENT code that the old model cannot read. Training a decoder on the post-stroke
    session itself separates them -- if it recovers normal accuracy, the information is intact and only
    the mapping changed.

    THE COMPARISON MUST BE POSITION-MATCHED, and this is the trap it was built to avoid. PS94 8/17 has
    engaged trials at 4 positions where its pre-stroke sessions have 6, so an unmatched within-session
    comparison pits a 4-way problem (chance 0.25) against 6-way ones (chance 0.167) and inflates the
    post-stroke side. That is the same trial-composition error that produced a spurious 'PS94 neural
    deficit' headline earlier in this project, running the other way. Pre-stroke sessions are therefore
    restricted to the SAME positions before their band is computed.

    Both arms use GroupKFold on the real position blocks, so the within-session number carries the same
    block-CV convention as everything else in the deck.

    THE PRE-STROKE BAND IS MEMOIZED PER SESSION, NOT FROZEN (2026-08-25). Recomputing every
    pre-stroke session's within-session accuracy every night costs real time, and the obvious fix --
    freeze the band the way `nolick_reference_prestroke.json` is frozen -- is WRONG here, for exactly
    the reason given above: on the lick-only arm the pre-stroke sessions are deliberately re-scored
    over the POST-stroke session's preserved positions, which change night to night. A frozen band
    would silently stop being position-matched. So the cache key carries the POSITION SET: an
    identical set reuses the value, a changed set recomputes.
    """
    # POST-STROKE SESSIONS USE ALL TRIALS (Priya, 2026-08-18): the missing licks ARE the phenotype, so
    # filtering to engaged trials removes the effect being measured -- and post-stroke that is not a
    # small correction, PS94 8/18 is only 40% engaged. Pre-stroke keeps the engaged cut, which is the
    # mismatch declared in nolick_analysis.SANCTIONED_MISMATCHES. An earlier version of this function
    # used engaged trials on BOTH sides, contradicting the decision it was written under.
    all_pos = list(DISPLAY_ORDER)
    scored = all_pos if post_all_trials else list(keep)
    rows = []
    for i in sorted(d["pre_i"]) + sorted(d["post_i"]):
        is_post = i in d["post_i"]
        if is_post and post_all_trials:
            # ALL SIX POSITIONS. Restricting to `keep` would drop exactly the positions the lesion
            # abolished -- the ones this arm exists to examine (Priya, 2026-08-19). PS94 far_center
            # and far_R have ZERO engaged trials and ~105 no-lick trials each.
            me = (d["GE"] == i) & np.isin(d["YE"], all_pos)
            mu = (d["GU"] == i) & np.isin(d["YU"], all_pos)
            X = np.vstack([d["XE"][me], d["XU"][mu]])
            y = np.concatenate([d["YE"][me], d["YU"][mu]])
            # the block vector must be filtered by the SAME position set as the trials it labels --
            # this line still said `keep` after the trials moved to all_pos, giving 600 trials and
            # 597 block ids
            ge = np.asarray(d["BE"][i])[np.isin(d["YE"][d["GE"] == i], all_pos)]
            # no-lick trials carry no block id in BE; give them their own groups so they cannot
            # straddle a CV fold with the engaged trials they sit between
            gb = np.concatenate([ge, np.arange(len(ge), len(ge) + int(mu.sum()))])
            m = np.ones(len(y), bool)
        else:
            # pre-stroke sessions are scored over whichever set this arm uses, so the band and the
            # post value are always on the same positions and the same chance level
            pos = all_pos if post_all_trials else keep
            m = (d["GE"] == i) & np.isin(d["YE"], pos)
            if m.sum() < min_trials:
                continue
            y, X = d["YE"][m], d["XE"][m]
            gb = np.asarray(d["BE"][i])[np.isin(d["YE"][d["GE"] == i], pos)]
        if len(y) < min_trials:
            continue
        ng = min(n_splits, int(np.unique(gb).size))
        if ng < 2:
            continue
        # `scored` IS the position set both arms are matched on, so it is the right cache key --
        # deriving a second expression here would be one more place for the two to drift apart.
        acc = _within_accuracy(d["kept"][i], X, y, gb, ng, is_post,
                               pos_key=tuple(sorted(scored)), align=d["align"])
        rows.append({"label": d["kept"][i], "within_accuracy": acc, "n": len(y),
                     "post": is_post})
    pre = np.array([r["within_accuracy"] for r in rows if not r["post"]], float)
    post = [r for r in rows if r["post"]]
    if len(pre) < 3 or not post:
        return {"note": "not enough sessions", "n_pre": len(pre), "n_post": len(post)}
    band = {"mean": float(pre.mean()), "sd": float(pre.std(ddof=1)),
            "min": float(pre.min()), "max": float(pre.max()), "n": len(pre)}
    out = {"n_positions": len(scored), "chance": 1.0 / len(scored), "within_pre_band": band,
           "positions_scored": [POSITION_NAMES[c] for c in scored],
           "post_arm": "ALL trials" if post_all_trials else "engaged only", "per_session": rows}
    for r in post:
        z = (r["within_accuracy"] - band["mean"]) / band["sd"] if band["sd"] else float("nan")
        out.setdefault("post", {})[r["label"]] = {
            "within_accuracy": r["within_accuracy"], "z": float(z),
            "inside_pre_range": bool(band["min"] <= r["within_accuracy"] <= band["max"])}
    # DIRECTION MATTERS. The first version tested only `inside_pre_range` and so called PS95 cue
    # (z=+1.4, ABOVE every pre-stroke session) "impaired" -- outside on the HIGH side is the opposite
    # of impairment. Only a value below the pre-stroke range is evidence of degraded information.
    vals = list(out.get("post", {}).values())
    below = [v for v in vals if v["z"] < 0 and not v["inside_pre_range"]]
    above = [v for v in vals if v["z"] > 0 and not v["inside_pre_range"]]
    if below:
        out["verdict"] = ("within-session decoding is ALSO impaired -> the position information "
                          "itself is degraded, not merely recoded")
    elif above:
        out["verdict"] = ("within-session decoding is ABOVE the pre-stroke range -> the information "
                          "is intact and if anything sharper; the frozen decoder fails because the "
                          "CODE CHANGED: RECODING, not loss")
    else:
        out["verdict"] = ("within-session decoding is NORMAL -> the position information is INTACT "
                          "and the frozen decoder fails because the CODE CHANGED: RECODING, not loss")
    out["information_degraded"] = bool(below)
    return out


# ------------------------------------------------------------------------------------------ G4 camera-class
#: SECTION G4's PRIMARY discriminator (Priya 11-04, follow-on a): a CAMERA-CLASS state discriminator. The two training
#: classes are camera classes, not the DAQ lick: WORKING (contact + incomplete, any DAQ side) vs NONE (no response; the
#: close-mid unjudged default included, as section G's "none"), position-balanced exactly as `balanced_lick_sample`
#: (within each kept spout both classes subsampled to the same count, >= 5; positives drawn first; same rng use).
#: jaw_only and after_end ("quit") are never trained on; they are SCORED as their own groups, as are contact /
#: incomplete / none separately. The only existing version was the check script
#: labcams/decoder_mix_checks/g4_camclass_mixshift_1104/g4common.py (`cam_balanced_sample`), ported here.
#: "daq" = the production DAQ lick-vs-no-lick discriminators (`looks_like_which` / `fits_engaged_distribution`), kept
#: as the labelled robustness arm.
G4_DISC_ENV = "WIDEFIELD_G4_DISC"
#: THE G4 FRAME'S CENTRE ROWS (Priya 11-04 via a8): a plain per-session z-score (mean / SD) on the session's camera
#: CONTACT rows (default; class contact, engaged and no-lick arms), on its WORKING rows (contact + incomplete), or
#: "engaged" = the production engaged-trial z-score (`pool_sessions`, identity here). The z frame is kept (no bal_em).
#: Carried in the stored discriminator's spec (`_g_store` extra) and in every output (`g4_frame_tag`).
G4_CENTRE_ENV = "WIDEFIELD_G4_CENTRE"
G4_CENTRES = ("contact", "working", "engaged")
#: the classes of the camera-class discriminator (part of its spec): positives / negatives
G4_POS = ("contact", "incomplete")
G4_NEG = ("none",)
#: minimum rows per scored cell (animal x epoch x group x spout), as in the G4 check
G4_MIN_N = 10


def g4_disc() -> str:
    import os
    v = os.environ.get(G4_DISC_ENV, "camera").strip().lower()
    if v in ("camera", "cam", "camclass"):
        return "camera"
    if v in ("daq", "lick", "old"):
        return "daq"
    raise ValueError(f"{G4_DISC_ENV}={v!r}: expected camera or daq")


def g4_centre() -> str:
    import os
    v = os.environ.get(G4_CENTRE_ENV, "contact").strip().lower()
    if v not in G4_CENTRES:
        raise ValueError(f"{G4_CENTRE_ENV}={v!r}: expected one of {G4_CENTRES}")
    return v


def g4_frame_tag() -> str:
    """The camera-class discriminator's identity: "camclass/z-<center>/pos-contact+incomplete/neg-none"."""
    return f"camclass/z-{g4_centre()}/pos-{'+'.join(G4_POS)}/neg-{'+'.join(G4_NEG)}"


def g4_frame(d):
    """``(dT, groups)`` for the camera-class discriminator, memoised on ``d`` per center: ``dT`` = ``d`` with every
    session's rows z-scored on its center rows (`g4_centre`; identity for "engaged") and ``pre_i`` narrowed to the pre
    sessions WITH camera classes and center rows; ``groups`` = `_camera_groups(d)`. None when the camera grouping does
    not apply (DAQ gate, lick window, no classes)."""
    from wfield_local import label_shift
    cg = _camera_groups(d)
    if cg is None:
        return None
    cache = d["_frozen_cache"]
    key = ("g4_frame", g4_centre())
    if key in cache:
        return cache[key]
    c = g4_centre()
    XE, XU = np.array(d["XE"], float), np.array(d["XU"], float)
    keep = set()
    for i in range(len(d["kept"])):
        if i not in cg["have"]:
            continue
        me, mu = d["GE"] == i, d["GU"] == i
        if c == "engaged":
            keep.add(i)
            continue
        sel = ("contact",) if c == "contact" else G4_POS
        C = np.vstack([d["XE"][me & np.isin(cg["grp_e"], sel)], d["XU"][mu & np.isin(cg["grp_u"], sel)]])
        if len(C) < 2:
            continue
        m_, sd = C.mean(0), label_shift.session_sd(C)
        XE[me] = (d["XE"][me] - m_) / sd
        if mu.any():
            XU[mu] = (d["XU"][mu] - m_) / sd
        keep.add(i)
    dT = dict(d)
    dT["XE"], dT["XU"] = XE, XU
    dT["pre_i"] = {i for i in d["pre_i"] if i in keep}
    dT["post_i"] = {i for i in d["post_i"] if i in keep}
    cache[key] = (dT, cg)
    return cache[key]


def cam_balanced_sample(dT, ge, gu, keep, rng, *, exclude_session=None):
    """The camera-class analog of `balanced_lick_sample` (ported from g4common.cam_balanced_sample, 11-04): pre
    WORKING (contact | incomplete) vs NONE rows (engaged and no-lick rows stacked), within each kept spout both
    subsampled to the same count (>= 5), positives drawn first, same ``rng`` use. ``(X, y, groups)`` or None."""
    X = np.vstack([dT["XE"], dT["XU"]])
    Y = np.concatenate([dT["YE"], np.asarray(dT["YU"]).astype(dT["YE"].dtype)])
    G = np.concatenate([dT["GE"], dT["GU"]])
    g = np.concatenate([ge, gu])
    m = np.isin(G, list(dT["pre_i"])) & np.isin(Y, keep)
    if exclude_session is not None:
        m &= G != exclude_session
    pos = m & np.isin(g, list(G4_POS))
    neg = m & np.isin(g, list(G4_NEG))
    xs, lab, grps = [], [], []
    for c in keep:
        ie, iu = np.flatnonzero(pos & (Y == c)), np.flatnonzero(neg & (Y == c))
        n = min(len(ie), len(iu))
        if n < 5:
            continue
        se = rng.choice(ie, n, replace=False)
        su = rng.choice(iu, n, replace=False)
        xs.append(X[se]); lab.append(np.ones(n)); grps.append(G[se])
        xs.append(X[su]); lab.append(np.zeros(n)); grps.append(G[su])
    if not xs:
        return None
    return np.vstack(xs), np.concatenate(lab), np.concatenate(grps)


def _g4_rows(dT, cg, sessions, keep):
    """All rows of ``sessions`` at the ``keep`` spouts, both arms stacked: ``(X, Y, group, jaw_judged, side)``."""
    me = np.isin(dT["GE"], list(sessions)) & np.isin(dT["YE"], keep)
    mu = np.isin(dT["GU"], list(sessions)) & np.isin(dT["YU"], keep)
    X = np.vstack([dT["XE"][me], dT["XU"][mu]])
    Y = np.concatenate([dT["YE"][me], np.asarray(dT["YU"])[mu].astype(dT["YE"].dtype)])
    g = np.concatenate([cg["grp_e"][me], cg["grp_u"][mu]])
    jj = np.concatenate([cg["jj_e"][me], cg["jj_u"][mu]])
    side = np.array(["lick"] * int(me.sum()) + ["nolick"] * int(mu.sum()), dtype=object)
    return X, Y, g, jj, side


def _g4_cells(clf, X, Y, g, jj, keep, min_n=G4_MIN_N):
    """Per camera group: the fraction classified WORKING-like over all its rows (spout-balanced: the mean of per-spout
    fractions over spouts with >= ``min_n`` rows, so a pooled value is never a spout-mix contrast), plus per spout
    {n, frac, thin}. Thin cells (< ``min_n``) are KEPT with ``thin`` = True and left out of the balanced mean."""
    from wfield_local import working_trials as wt
    out = {}
    pr = clf.predict(X).astype(float) if len(X) else np.zeros(0)
    for grp in wt.G_ALL:
        m = g == grp
        per = {}
        for c in keep:
            mc = m & (Y == c)
            n = int(mc.sum())
            per[POSITION_NAMES[int(c)]] = {"n": n, "frac": float(pr[mc].mean()) if n else None, "thin": n < min_n,
                                           "n_jaw_unjudged": int((~jj[mc]).sum())}
        ok = [v["frac"] for v in per.values() if not v["thin"]]
        out[grp] = {"n": int(m.sum()), "frac_pooled": float(pr[m].mean()) if m.any() else None,
                    "frac_spout_balanced": float(np.mean(ok)) if ok else None, "n_spouts_ok": len(ok),
                    "per_spout": per}
    return out


def g4_camclass(d, keep, seed=0, min_n=G4_MIN_N):
    """SECTION G4, camera-class discriminator (the PRIMARY since 11-04; follow-on a). Trains WORKING vs NONE on the pre
    sessions (position-balanced, `cam_balanced_sample`) in the z frame centered on `g4_centre` rows; scores every post
    session row by camera group (contact / incomplete / jaw_only / none / after_end), spout-balanced; the PRE reference
    per group = each pre session's rows scored by the discriminator that left it out (one RandomState(seed) across the
    folds, then the full fit, as `fits_engaged_distribution`). Each post group is read against the SAME group pre-stroke
    (``pre_reference``); a group with no pre cell >= ``min_n`` is read against pre WORKING, labeled
    ``pre_reference_is`` = "working". STORED (kind g_cam_disc; the spec carries keep, seed, center, class set and the
    run's selection tag). None when the camera grouping does not apply."""
    from wfield_local import trial_classes as _tc
    got = g4_frame(d)
    if got is None:
        return None
    dT, cg = got
    rng = np.random.RandomState(seed)

    def _fit(exclude_session=None):
        s = cam_balanced_sample(dT, cg["grp_e"], cg["grp_u"], keep, rng, exclude_session=exclude_session)
        if s is None:
            return None
        X, y, _g = s
        return lick_pipe().fit(X, y)

    def _fit_folds():
        loso = {d["kept"][i]: _fit(exclude_session=i) for i in sorted(dT["pre_i"])}
        return {"loso": loso, "full": _fit()}

    extra = (f"{_keep_tag(keep)}-seed{seed}-{g4_frame_tag().replace('/', '_')}-{_tc.run_selection()}")
    folds = _g_store(d, "g_cam_disc", extra, _fit_folds, meta={"frame": g4_frame_tag()})
    if folds is None:
        folds = _fit_folds()
    if folds["full"] is None:
        return {"note": "no spout had >= 5 pre WORKING and >= 5 pre NONE rows", "frame": g4_frame_tag()}
    # PRE reference: every pre session by the discriminator that left it out
    pre_cells = []
    for i in sorted(dT["pre_i"]):
        clf_i = folds["loso"].get(d["kept"][i])
        if clf_i is None:
            continue
        X, Y, g, jj, _s = _g4_rows(dT, cg, [i], keep)
        pre_cells.append(_g4_cells(clf_i, X, Y, g, jj, keep, min_n=1))
    from wfield_local import working_trials as wt
    pre_ref = {}
    for grp in wt.G_ALL:
        # pooled over pre sessions, per spout (n summed; fraction weighted by n), then spout-balanced at the floor
        per = {}
        for c in keep:
            p = POSITION_NAMES[int(c)]
            n = sum(pc_[grp]["per_spout"][p]["n"] for pc_ in pre_cells)
            k = sum((pc_[grp]["per_spout"][p]["frac"] or 0.0) * pc_[grp]["per_spout"][p]["n"] for pc_ in pre_cells)
            per[p] = {"n": int(n), "frac": (k / n) if n else None, "thin": n < min_n}
        ok = [v["frac"] for v in per.values() if not v["thin"]]
        pre_ref[grp] = {"frac_spout_balanced": float(np.mean(ok)) if ok else None, "per_spout": per,
                        "per_session": [pc_[grp]["frac_spout_balanced"] for pc_ in pre_cells]}
    work_ref = {}
    for c in keep:
        p = POSITION_NAMES[int(c)]
        n = sum(pre_ref[gq]["per_spout"][p]["n"] for gq in G4_POS)
        k = sum((pre_ref[gq]["per_spout"][p]["frac"] or 0.0) * pre_ref[gq]["per_spout"][p]["n"] for gq in G4_POS)
        work_ref[p] = {"n": int(n), "frac": (k / n) if n else None, "thin": n < min_n}
    okw = [v["frac"] for v in work_ref.values() if not v["thin"]]
    X, Y, g, jj, _s = _g4_rows(dT, cg, dT["post_i"], keep)
    post = _g4_cells(folds["full"], X, Y, g, jj, keep, min_n=min_n)
    for grp in wt.G_ALL:
        same = pre_ref[grp]["frac_spout_balanced"]
        post[grp]["pre_reference_is"] = grp if same is not None else "working"
        post[grp]["pre_reference"] = same if same is not None else (float(np.mean(okw)) if okw else None)
    return {"frame": g4_frame_tag(), "disc": "camera", "min_n": int(min_n), "thin_cells": "kept, flagged thin, "
            "left out of the spout-balanced means", "post": post, "pre_reference": pre_ref,
            "pre_working_reference": {"frac_spout_balanced": float(np.mean(okw)) if okw else None,
                                      "per_spout": work_ref},
            "group_labels": dict(wt.G_LABELS), "n_pre_sessions": len(pre_cells),
            "post_sessions": [d["kept"][i] for i in sorted(dT["post_i"])]}
