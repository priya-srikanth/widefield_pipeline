"""Pre-cue activity across five lick/engagement states, in the shared joint-LocaNMF basis.

Priya, 2026-08-20. REPORTED, not a filter: nothing here changes what any existing analysis uses.
``POSTSTROKE_ENGAGEMENT_FILTERING`` stays False and the all-trials post-stroke rule is untouched.

THE FIVE STATES, all in the PRE-CUE window:

  1  pre-stroke,  LICK          engaged, movement happened
  2  pre-stroke,  NO-LICK       pre-stroke this is essentially the sated/disengaged state
  4  post-stroke, LICK          engaged, movement happened
  5  post-stroke, NO-LICK, ENGAGED       the animal was working; this trial produced no lick
  6  post-stroke, NO-LICK, NOT ENGAGED   after the animal quit

THE TEST THAT MAKES IT MORE THAN FIVE NUMBERS. A discriminator is trained on the PRE-STROKE
lick-vs-no-lick contrast -- i.e. on data where engagement is not in question and there is no motor
deficit to confuse it with -- and states 5 and 6 are pushed through it. If the engagement gate is
capturing a real state, 6 should score DISENGAGED and 5 should not. If 5 scores engaged while
producing no movement, that is the "plan formed, execution failed" reading with an independent
witness. If 5 and 6 score alike, the gate is not capturing anything the cortex shows, and that
is a real and reportable outcome.

UNDER THE CAMERA TRIAL GATE THE TWO TRAINING CLASSES ARE CAMERA CLASSES (Priya 11-05, DECISIONS_PENDING_MERGE_2026-11-04
(f) 1): WORKING (contact + incomplete, from either DAQ arm) vs NO_RESPONSE. The DAQ no-lick side used to hold incomplete,
jaw_only, no_response and the STOPPED runs together; stopped runs sit at session end, so training on them puts time in
session into the label. Stopped, stopped_licking, jaw_only, after_end and unknown are now SCORED only, each as its own
class (``discriminator["by_class"]``). The stored model's align carries `CAMERA_DISC_TAG`, so it is a new frozen id; the
DAQ gate keeps the old lick-vs-no-lick training set and its id (the labelled DAQ row). The late window and SPATIAL_MISS
below still apply to both classes.

WHY THE TRAINING SET IS THE LATE WINDOW. Pre-stroke no-lick trials sit late in the session -- 79%,
85% and 78% of them in the final quarter for PS92, PS94 and PS95 -- because pre-stroke that IS what
a no-lick trial is (Priya, 2026-08-20). Taking both classes from the same late window therefore does
NOT strip the state; it isolates it, and it removes elapsed time as an explanation at the same time.
State 6 is terminal by construction, so without that control a discriminator trained on early-vs-late
would separate 5 from 6 by the clock alone and look like a state readout.

PS93 IS TREATED SEPARATELY. Its no-lick trials are not mainly satiety: 90% of its EARLY no-lick
trials are at far_L/far_center, i.e. spatial misses (Priya, 2026-08-20, confirmed by count). Those
are excluded from its disengaged class, or its axis would be a mixture of "disengaged" and "missed a
hard position".

THE TRIAL UNIVERSE IS THE IMAGING PIPELINE'S OWN. The engaged/no-lick split, the positions and the
trial ordering all come from ``_trial_features`` via ``with_indices``, never from the behaviour
trial table. The two disagree in ways that would silently mis-label trials: different free-reward
handling, and a response window of ``decode.max_rt_s`` here against the session's real 3.5 s there.
One universe, one definition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config, joint_locanmf
from wfield_local import gate_source  # [gate-source hook] WIDEFIELD_GATE_SOURCE (see that module)
from wfield_local.locanmf_frozen_decoder import _pipe, pool_sessions
from wfield_local.locanmf_position_decoder import trial_features_cached
from wfield_local.paths import PathResolver
from wfield_local.plot_lick_aligned_averages import POSITION_NAMES

#: "reliable locations" the engagement gate is judged at (Priya, 2026-08-20).
REFERENCE = ("close_L", "close_center")
#: positions whose no-lick trials are spatial misses rather than disengagement, per animal.
SPATIAL_MISS = {"PS93": ("far_L", "far_center")}
LATE_FRACTION = 0.75      # "late window" = final quarter of a session
WINDOW = 15               # trailing window, in REFERENCE trials
MIN_RATE = 0.5            # reference response rate below this = collapsed
CHANCE = 1.0 / 6.0

STATES = ("1_pre_lick", "2_pre_nolick", "4_post_lick", "5_post_nolick_engaged",
          "6_post_nolick_notengaged")

#: THE DISCRIMINATOR'S TRAINING CLASSES UNDER THE CAMERA GATE (Priya 11-05, DECISIONS_PENDING_MERGE_2026-11-04 (f) 1,
#: camera-class question 8): WORKING (contact + incomplete, either DAQ arm) vs NO_RESPONSE (no-lick arm). Fixed class
#: names, not `trial_classes.in_sel` -- the in_working_jaw robustness selection would put jaw_only into the trained
#: side, and the decision says jaw_only is scored-only. STOPPED is never trained on (stop runs sit at session end, so
#: training on them mixes time in session into the label); it is scored as its own class, as are the others below.
TRAIN_POS = ("contact", "incomplete")
TRAIN_NEG = ("no_response",)
SCORED_ONLY = ("jaw_only", "stopped", "stopped_licking", "after_end", "unknown")
#: the class tag in the stored discriminator's align (`<align>-disc-<tag>`), so no camera-class model can ever be served
#: for the DAQ lick-vs-no-lick one or vice versa -- also under `frozen_models.daq_trained_in_lp`, whose train_fp is a
#: constant and therefore does not move with the training rows
CAMERA_DISC_TAG = "cls-work-v-noresp-min15"
#: MINIMUM TRAINING ROWS PER CLASS (all-pre fit and each leave-one-session-out refit): 20 for the DAQ lick-vs-no-lick
#: model (unchanged, so is its id); 15 for the camera classes (Priya 11-05 ~22:10: "let's say 15", relayed as a minimum of
#: 15 trials; pre-stroke no_response is scarce -- class tables PS92 16 / PS94 25 late). The floor is part of the camera
#: model's identity (the "-min15" in `CAMERA_DISC_TAG`). The held-out session's own floor (D2) stays 20.
MIN_TRAIN = {"daq": 20, "camera": 15}


def camera_train_sets(cls_e, cls_u, eng_window, nl_window, dis_window):
    """``(keep_eng, keep_pos_u, keep_dis)`` of the state discriminator under the camera gate (`TRAIN_POS` /
    `TRAIN_NEG`). ``cls_e`` / ``cls_u`` = camera class per engaged / no-lick row (`trial_classes.pool_row_classes`);
    ``eng_window`` / ``nl_window`` = the pre-stroke late-window rows of each arm; ``dis_window`` = the no-lick rows the
    DAQ design allowed on the disengaged side (late window minus SPATIAL_MISS). Positives: engaged rows AND no-lick rows
    of class contact | incomplete (the no-lick arm holds the incomplete licks); negatives: no-lick rows of class
    no_response only. Rows of every other class (stopped, stopped_licking, jaw_only, after_end, unknown) are in none."""
    ce = np.asarray(cls_e, dtype=object)
    cu = np.asarray(cls_u, dtype=object)
    keep_eng = np.asarray(eng_window, bool) & np.isin(ce, TRAIN_POS)
    keep_pos_u = np.asarray(nl_window, bool) & np.isin(cu, TRAIN_POS)
    keep_dis = np.asarray(dis_window, bool) & np.isin(cu, TRAIN_NEG)
    return keep_eng, keep_pos_u, keep_dis


def disc_train_rows(XE, XU, GE, GU, keep_eng, keep_pos_u, keep_dis, exclude=None):
    """``(X, y, groups)`` of the state discriminator: engaged-side rows (``XE[keep_eng]`` then ``XU[keep_pos_u]``,
    y = 0) stacked over the disengaged rows (``XU[keep_dis]``, y = 1), optionally without session ``exclude``. With
    ``keep_pos_u`` all False (the DAQ gate) this is exactly the pre-11-05 stack, so the DAQ model's id is unchanged."""
    e, pu, dn = np.asarray(keep_eng, bool), np.asarray(keep_pos_u, bool), np.asarray(keep_dis, bool)
    if exclude is not None:
        e, pu, dn = e & (GE != exclude), pu & (GU != exclude), dn & (GU != exclude)
    X = np.vstack([XE[e], XU[pu], XU[dn]]) if pu.any() else np.vstack([XE[e], XU[dn]])
    y = np.concatenate([np.zeros(int(e.sum() + pu.sum())), np.ones(int(dn.sum()))])
    g = np.concatenate([GE[e], GU[pu], GU[dn]])
    return X, y, g


def class_scores(model, XEs, XUs, cls_e, cls_u, pre_e, pre_u, post_e, post_u):
    """``{"pre": {class: {"n", "p_disengaged"}}, "post": {...}}``: the discriminator's mean P(disengaged) per camera
    class, engaged and no-lick rows pooled, for the camera classes `TRAIN_POS` + `TRAIN_NEG` + `SCORED_ONLY`."""
    ce, cu = np.asarray(cls_e, dtype=object), np.asarray(cls_u, dtype=object)
    out = {}
    for ph, me, mu in (("pre", pre_e, pre_u), ("post", post_e, post_u)):
        out[ph] = {}
        for c in TRAIN_POS + TRAIN_NEG + SCORED_ONLY:
            X = np.vstack([XEs[np.asarray(me, bool) & (ce == c)], XUs[np.asarray(mu, bool) & (cu == c)]])
            out[ph][c] = {"n": int(len(X)),
                          "p_disengaged": float(np.mean(model.predict_proba(X)[:, 1])) if len(X) else None}
    return out


def _disc():
    """The state discriminator. Plain regularised LR: the axis must stay interpretable."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))


def features_with_indices(basis, nolick_ref="cue"):
    """``joint_features``, but it also keeps each session's trial INDICES and reaction times.

    The indices are what let the engagement gate be computed on exactly the trials the features
    come from, in their true within-session order -- including the fact that the pre-cue alignment
    DROPS trials with no lick-free window, so the sequence has gaps that must not be closed up.

    ``.rts`` is the same idea for first-lick latency: seconds from cue, aligned row-for-row with
    ``.indices[lab][0]`` and therefore with the engaged feature rows. It is what
    `position_coding_directions` splits post-stroke lick trials on at 2.0 s.

    RT IS REQUESTED UNCONDITIONALLY, not behind a flag. Every caller of this builder shares one disk
    cache, and a per-caller flag would give them different `feature_cache_kind` digests -- the
    split-the-cache-by-caller failure `_FEATURE_ARGS` is explicitly shaped to avoid. One float per
    trial is not worth two copies of a ~1 MB-per-session store, nor a second cold build.
    """
    idx, rts, vc = {}, {}, {}

    def _feat(s, args):
        # CACHED, AND THE PROJECTION IS DEFERRED BEHIND THE CACHE (2026-08-27). This builder had no
        # memoisation of any kind while its sibling `joint_xsession.joint_features` did, and
        # `grant_figures` constructs it at FIVE independent call sites -- each a fresh closure, so
        # each re-projected every session and rebuilt every trial matrix. Measured ~5x rebuilds per
        # distinct session across a render.
        lab = s["label"]
        src = joint_locanmf.BasisSource(basis, s)
        out = trial_features_cached(s, args, signal_fn=src.signal, signal_key=src.key,
                                    with_indices=True, with_rt=True, nolick_ref=nolick_ref)
        vc[lab] = src.variance_captured()
        idx[lab] = (out[6], out[7])
        rts[lab] = out[8]
        return out[:6]

    _feat.indices = idx
    _feat.rts = rts
    _feat.variance_captured = vc
    # THE FEATURE SPACE IS PART OF A FROZEN MODEL'S IDENTITY. Same session, same args, different
    # basis -> different features (256 columns vs 380, measured), so a model frozen against one basis
    # must never be served for another. `frozen_models` reads this off the callable.
    _feat.basis_id = basis.basis_id
    return _feat


def engagement_gate(order, responded, positions, reference=REFERENCE):
    """Per-trial engaged/not-engaged for ONE session, from a NON-RECOVERING reference collapse.

    ``order`` is the trial index of each entry (gaps allowed), already sorted. Returns a bool array
    aligned to it, True = NOT engaged.

    Requiring non-recovery is what separates satiety from a motor patch: PS94_0817's reference rate
    drops around trial 420 and is back near 0.95 by 480, and that session should not be called
    disengaged at all. Only the FINAL sustained collapse counts.

    FAIL CLOSED under the camera trial gate (rebuild 11-03): this IS the DAQ quit-tail rule, so under
    gate=camera it refuses unless the caller opened `trial_classes.daq_quit_rule(reason)` (a measure of
    engagement, a labelled DAQ robustness row, or bookkeeping a camera-wired caller overrides).
    """
    from wfield_local import trial_classes as _tc
    _tc.refuse_daq_gate("precue_engagement_states.engagement_gate")
    ref = np.isin(positions, reference)
    roll = np.full(len(order), np.nan)
    hist: list[bool] = []
    for i in range(len(order)):
        if ref[i]:
            hist.append(bool(responded[i]))
            if len(hist) > WINDOW:
                hist.pop(0)
        if hist:
            roll[i] = float(np.mean(hist))
    low = np.nan_to_num(roll, nan=1.0) < MIN_RATE
    out = np.zeros(len(order), bool)
    if low.any() and low[-1]:
        i = len(low) - 1
        while i > 0 and low[i - 1]:
            i -= 1
        # BACKDATED TO THE START OF THE RUN THAT TRIPPED IT (Priya, 2026-09-07), matching the
        # behaviour gate. `low` turns True where the trailing reference mean CROSSES MIN_RATE, which
        # takes about eight reference misses to accumulate -- so without this the first misses of a
        # quit are scored `poststroke_miss_working`, a class whose whole meaning is that the animal
        # was still trying. The onset is the trial after the last reference RESPONSE: "last
        # demonstrably engaged" is a fact about the animal, where the crossing is an artefact of
        # WINDOW and MIN_RATE and would move if either were retuned.
        #
        # Moves 408 of 3797 post-stroke miss_working trials (10.7%) into poststroke_stopped. Both
        # classes are analysed downstream, so this is a RELABEL rather than an exclusion -- but a
        # miss-while-working class holding trials from after the animal quit is contaminated in
        # exactly the way the behavioural one was.
        resp_arr = np.asarray(responded, bool)
        prior_ok = np.flatnonzero(ref[:i] & resp_arr[:i])
        i = int(prior_ok[-1]) + 1 if prior_ok.size else 0
        out[i:] = True
    return out


def _session_tables(feat, kept):
    """Per session: ordered trial index, responded, position name, and late-window flag."""
    tables = {}
    for si, lab in enumerate(kept):
        got = feat.indices.get(lab)
        if got is None:
            continue
        idx_e, idx_n = got
        order = np.concatenate([idx_e, idx_n])
        responded = np.concatenate([np.ones(len(idx_e), bool), np.zeros(len(idx_n), bool)])
        o = np.argsort(order, kind="stable")
        order, responded = order[o], responded[o]
        responded = gate_source.gate_responded(lab, order, responded)  # [gate-source hook] identity unless =lp
        tables[si] = {"order": order, "responded": responded, "label": lab}
    return tables


def _positions_for(tables, si, y_eng, g_eng, y_nl, g_nl, idx_e, idx_n):
    """Position NAME per entry of ``tables[si]['order']``."""
    pos_by_idx = {}
    for k, p in zip(idx_e, y_eng[g_eng == si]):
        pos_by_idx[int(k)] = POSITION_NAMES.get(int(p), str(p))
    for k, p in zip(idx_n, y_nl[g_nl == si]):
        pos_by_idx[int(k)] = POSITION_NAMES.get(int(p), str(p))
    return np.array([pos_by_idx.get(int(k), "?") for k in tables[si]["order"]])


def run_animal(animal, align="precue", post_s=None, verbose=True):
    """The five states plus the pre-stroke discriminator, for one animal. ``None`` if unavailable."""
    # THE ENL WINDOW (Priya 11-04): the lick source's window (`lp_licks.enl_post_s`: 1.0 s under LP, decode.precue_post_s
    # = 2.0 s under DAQ), never precue_post_s read directly. The `cue` alignment keeps the value it always had.
    from wfield_local import lp_licks
    post_s = post_s or (lp_licks.enl_post_s() if lp_licks.is_precue(align)
                        else float(config.defaults()["decode"]["precue_post_s"]))
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    if not pre or not post:
        return None
    try:
        from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
        basis = joint_locanmf.load(animal, sessions=_ALL)
    except FileNotFoundError as ex:
        print(f"[precue_states] {animal}: {ex}", flush=True)
        return None

    feat = features_with_indices(basis)
    pooled = pool_sessions(pre + post, source="locanmf", align=align, post_s=post_s, features=feat)
    if pooled is None:
        return None
    XE, YE, GE, _BE, XU, YU, kept, _common, GU = pooled
    YU = YU.astype(int)
    pre_i = {i for i, l in enumerate(kept) if l in set(pre)}

    tables = _session_tables(feat, kept)
    if not tables:
        return None

    # ---- engagement gate + late-window flag, per session, on the IMAGING trial universe ----------
    not_eng_nl, late_nl, sess_nl = [], [], []          # aligned to XU rows
    late_e = []                                        # aligned to XE rows
    from wfield_local import trial_classes as _tc
    camera = _tc.camera_gate_on()
    for si in range(len(kept)):
        t = tables.get(si)
        if t is None:
            continue
        idx_e, idx_n = feat.indices[kept[si]]
        pos = _positions_for(tables, si, YE, GE, YU, GU, idx_e, idx_n)
        # under the camera gate the DAQ mask is not built at all (states 5 / 6 come from the camera classes below)
        ne = np.zeros(len(t["order"]), bool) if camera else engagement_gate(t["order"], t["responded"], pos)
        span = max(int(t["order"].max()), 1)
        late = t["order"] > LATE_FRACTION * span
        by_idx_ne = {int(k): bool(v) for k, v in zip(t["order"], ne)}
        by_idx_late = {int(k): bool(v) for k, v in zip(t["order"], late)}
        for k in idx_n:
            not_eng_nl.append(by_idx_ne.get(int(k), False))
            late_nl.append(by_idx_late.get(int(k), False))
            sess_nl.append(si)
        for k in idx_e:
            late_e.append(by_idx_late.get(int(k), False))
    not_eng_nl = np.array(not_eng_nl, bool)
    late_nl, late_e = np.array(late_nl, bool), np.array(late_e, bool)
    if len(not_eng_nl) != len(XU) or len(late_e) != len(XE):
        print(f"[precue_states] {animal}: trial bookkeeping mismatch "
              f"(nolick {len(not_eng_nl)}/{len(XU)}, eng {len(late_e)}/{len(XE)}) -- skipped",
              flush=True)
        return None

    e_pre = np.isin(GE, list(pre_i))
    u_pre = np.isin(GU, list(pre_i))
    # STATES 5 / 6: the DAQ quit-tail split (default, unchanged) or, under the camera trial gate (rebuild 11-03,
    # `trial_classes`), 5 = post no-lick rows in GATES["working"], 6 = post no-lick rows in STOPPED_CLASSES (a SUSTAINED
    # stop, Priya 11-03); isolated no_response and "unknown" rows are in neither. The pre-stroke discriminator: no gate.
    nl_eng, nl_not = ~not_eng_nl, not_eng_nl
    if camera:
        _ce, _cu = _tc.pool_row_classes(feat, kept, len(YE), len(YU))
        nl_eng, nl_not = _tc.in_sel(_cu, "miss_working"), np.isin(_cu, list(_tc.STOPPED_CLASSES))   # review M6
    states = {
        "1_pre_lick": (XE[e_pre], YE[e_pre], GE[e_pre]),
        "2_pre_nolick": (XU[u_pre], YU[u_pre], GU[u_pre]),
        "4_post_lick": (XE[~e_pre], YE[~e_pre], GE[~e_pre]),
        "5_post_nolick_engaged": (XU[~u_pre & nl_eng], YU[~u_pre & nl_eng],
                                  GU[~u_pre & nl_eng]),
        "6_post_nolick_notengaged": (XU[~u_pre & nl_not], YU[~u_pre & nl_not],
                                     GU[~u_pre & nl_not]),
    }

    # ---- A. position decoding: ONE frozen pre-stroke decoder, applied to every state -------------
    # bal_em since 2026-11-03 (`spout_scaling.FrozenFrame`): trained on class-balanced pre sessions, each scored
    # session centred by the label-free EM centre of its engaged trials under the scoring model (identity under
    # WIDEFIELD_SPOUT_SCALING=session_z, where this is the pre-11-03 code exactly).
    # STORED BY DEFAULT (rebuild 2026-11-03, rule 10; kind `engagement_states`, `_stored`, through the FrozenFrame store
    # hook; payload {"full", "loso"} on the FrozenFrame training frame, spec + scaling): opt out with
    # WIDEFIELD_ENGSTATES_FROZEN=0, which fits exactly this in memory.
    from wfield_local import spout_scaling as _ss
    _sc = _ss.spout_scaling()
    ff = _ss.FrozenFrame(XE, YE, GE, sorted(pre_i), _pipe, loso=_sc is not None, labels=dict(enumerate(kept)),  # noqa: E501
                         cls_e=_ce if camera else None,       # EM centre rows: camera WORKING rows (Priya 11-03)
                         store=lambda fit_all: _stored(animal, f"{align}-posdec-ff", basis, kept, pre_i,
                                                       int(XE.shape[1]), (XE[e_pre], YE[e_pre], GE[e_pre]), fit_all,
                                                       post_s, scaling=_sc))
    clf = ff.full
    dec = {}
    for k, (X, y, g) in states.items():
        if len(y) < 10:
            dec[k] = {"n": len(y), "accuracy": None}
            continue
        if k == "1_pre_lick":       # its own training data: leave-one-session-out instead
            if ff.scaling:
                p = np.empty_like(y)
                for i in np.unique(g):
                    m = g == i
                    p[m] = ff.loso[i].predict(ff.apply(X[m], i, "loso"))
            else:
                p = cross_val_predict(_pipe(), X, y, cv=LeaveOneGroupOut(), groups=g)
        else:
            p = clf.predict(ff.apply_rows(X, g))
        dec[k] = {"n": len(y), "accuracy": float(np.mean(p == y))}

    # ---- C. the pre-stroke state discriminator, trained in the LATE window -----------------------
    miss = SPATIAL_MISS.get(animal, ())
    nl_names = np.array([POSITION_NAMES.get(int(p), str(p)) for p in YU])
    keep_dis = u_pre & late_nl & ~np.isin(nl_names, miss)
    keep_eng = e_pre & late_e
    # no-lick-arm rows trained on the ENGAGED side: none under the DAQ gate (the arrays below are then byte-identical to
    # the pre-11-05 training set, and so is its stored id); the camera WORKING rows of the no-lick arm under camera
    keep_pos_u = np.zeros(len(XU), bool)
    if camera:
        # CAMERA CLASSES (Priya 11-05, DECISIONS_PENDING_MERGE (f) 1): WORKING (contact + incomplete) vs NO_RESPONSE.
        # STOPPED trials are scored as their own class and never trained on -- stop runs sit at session end, so
        # training on them puts time in session into the label; jaw_only / after_end / stopped_licking / unknown are
        # scored-only too. The late window and the SPATIAL_MISS exclusion are kept as they were (the decision names the
        # classes only).
        keep_eng, keep_pos_u, keep_dis = camera_train_sets(_ce, _cu, e_pre & late_e, u_pre & late_nl, keep_dis)
    disc = {"n_engaged_train": int(keep_eng.sum() + keep_pos_u.sum()), "n_disengaged_train": int(keep_dis.sum()),
            "excluded_positions": list(miss)}
    if camera:
        disc.update(train_classes={"engaged": list(TRAIN_POS), "disengaged": list(TRAIN_NEG)},
                    scored_only=list(SCORED_ONLY), train_tag=CAMERA_DISC_TAG)
    # OPTIONAL STATE-CLASS bal_em (`spout_scaling.state_*`; WIDEFIELD_STATE_DISC_SCALING=state_bal_em, DEFAULT OFF since
    # it fails the discriminator's own pre-stroke check, 11-03): each pre session's rows centred on the equal-weight mean
    # of its two TRUE state-class means; a scored session centred, label-free, by the EM over its engaged + no-lick rows.
    # The input arrays themselves when OFF (the production engaged-trial z-score).
    # THE SPOUT bal_em FRAME (main 26fd5b44 / 4a37cbc8, items B; WIDEFIELD_STATE_DISC_SCALING=spout_frame, DEFAULT OFF):
    # pre sessions in the position decoder's TRAINING frame, scored sessions in its label-free scoring frame
    # (`_spout_frame_block`); else the state-class frame / the production z (unchanged).
    _spf, disc_scaling, e_pre_d, u_pre_d, pre_d, ff_s, keep_eng, keep_dis = _spout_frame_block(
        disc, ff, feat, kept, XE, YE, GE, XU, YU, GU, pre_i, align, e_pre, u_pre, keep_eng, keep_dis)
    keep_pos_u = keep_pos_u & u_pre_d             # the pre sessions the frame block kept
    if _spf:
        XEt, XUt = np.array(XE, float), np.array(XU, float)
        for i in sorted(pre_d):
            XEt[GE == i] = ff_s.train_apply(XE[GE == i], i)
            if np.any(GU == i):
                XUt[GU == i] = ff_s.train_apply(XU[GU == i], i)
    else:
        # (the opt-in state-class centre takes its engaged class from the engaged arm only; the no-lick-arm working rows
        # of the camera classes are not in it -- identity under the production session_z, where this returns XE / XU)
        XEt, XUt = _ss.state_train_frame(XE, GE, XU, GU, keep_eng, keep_dis, sorted(pre_i))
    floor = MIN_TRAIN["camera" if camera else "daq"]
    disc["min_train_per_class"] = floor
    if keep_dis.sum() >= floor and keep_eng.sum() + keep_pos_u.sum() >= floor:
        Xd, yd, gd = disc_train_rows(XEt, XUt, GE, GU, keep_eng, keep_pos_u, keep_dis)

        def _fit_disc():
            # the all-pre discriminator AND the per-held-out-session refits of the D2 drift control (below), one
            # payload: they are determined by the same training pool (main 11-03: on the state training frame XEt / XUt)
            loso = {}
            for si in sorted(pre_d):
                hold = e_pre_d & (GE == si)
                tr_e, tr_d = keep_eng & (GE != si), keep_dis & (GU != si)
                tr_pu = keep_pos_u & (GU != si)
                if hold.sum() < 20 or tr_e.sum() + tr_pu.sum() < floor or tr_d.sum() < floor:
                    continue
                Xl, yl, _gl = disc_train_rows(XEt, XUt, GE, GU, keep_eng, keep_pos_u, keep_dis, exclude=si)
                loso[kept[si]] = _disc().fit(Xl, yl)   # keyed by LABEL (merge review B1), never pooled id
            return {"full": _disc().fit(Xd, yd), "loso": loso}

        # the class tag in the stored id under the camera gate (`CAMERA_DISC_TAG`); "<align>-disc" = the DAQ model, as before
        dpay = _stored(animal, f"{align}-disc" + (f"-{CAMERA_DISC_TAG}" if camera else ""), basis, kept, pre_i, int(XE.shape[1]),
                       (Xd, yd, gd, sorted(pre_i), XE[e_pre], GE[e_pre]), _fit_disc, post_s,
                       scaling=disc_scaling)
        model = dpay["full"]
        _pi = _ss.model_prior(model, yd)
        _all_e, _all_u = np.ones(len(XE), bool), np.ones(len(XU), bool)
        _post = sorted(set(range(len(kept))) - set(pre_i))
        if _spf:
            XEs, XUs = XEt.copy(), XUt.copy()
            for i in _post:
                XEs[GE == i] = ff_s.apply(XE[GE == i], i, "full")
                if np.any(GU == i):
                    XUs[GU == i] = ff_s.apply(XU[GU == i], i, "full")
        else:
            XEs, XUs = _ss.state_scored(model, _pi, XEt, GE, XUt, GU, _all_e, _all_u, _post)
        # states 5 / 6 by the run's gate (camera classes under the camera gate, rebuild 11-03; DAQ quit tail otherwise)
        s_states = {
            "1_pre_lick": XEs[e_pre_d], "2_pre_nolick": XUs[u_pre_d], "4_post_lick": XEs[~e_pre],
            "5_post_nolick_engaged": XUs[~u_pre & nl_eng],
            "6_post_nolick_notengaged": XUs[~u_pre & nl_not]}
        if len(np.unique(gd)) > 1:
            cvp = cross_val_predict(_disc(), Xd, yd, cv=LeaveOneGroupOut(), groups=gd,
                                    method="predict_proba")[:, 1]
            disc["train_auc_loso"] = _auc(yd, cvp)
        for k in states:
            X = s_states[k]
            disc[k] = (float(np.mean(model.predict_proba(X)[:, 1])) if len(X) else None)
        if camera:
            # EVERY CAMERA CLASS SCORED ON ITS OWN ((f) 1: stopped is its own class), both DAQ arms, pre and post.
            # Pre rows of the TRAINED classes are in-sample; read them as the training fit, not as a test.
            disc["by_class"] = class_scores(model, XEs, XUs, _ce, _cu, e_pre_d, u_pre_d, ~e_pre, ~u_pre)
        # D2. THE STRONGER DRIFT CONTROL: PRE-STROKE LICK trials, early vs late.
        #
        # One state throughout, spanning the whole session, and thousands of trials instead of the
        # 53-147 late trials state 5 can offer -- which was the weak point of D1 (Priya, 2026-08-20:
        # "there are no early no-lick trials pre-stroke"). Exactly so: pre-stroke no-lick is 78-85%
        # late, PS92 has 28 early ones, so the control cannot live inside the no-lick class. The
        # LICK class can carry it.
        #
        # LOSO over pre-stroke sessions, refitting each time, so early and late trials are scored
        # by a model that never saw that session. Scoring in-sample late trials against out-of-sample
        # early ones would manufacture exactly the difference being tested for.
        pre_e, pre_l = [], []
        for si in sorted(pre_d):
            hold = e_pre_d & (GE == si)
            tr_e, tr_d = keep_eng & (GE != si), keep_dis & (GU != si)
            tr_pu = keep_pos_u & (GU != si)
            if hold.sum() < 20 or tr_e.sum() + tr_pu.sum() < floor or tr_d.sum() < floor:
                continue
            _yd = np.concatenate([np.zeros(int(tr_e.sum() + tr_pu.sum())), np.ones(int(tr_d.sum()))])
            m = dpay["loso"][kept[si]]                 # stored with the discriminator (same condition, same fit)
            # the held-out session in ITS OWN EM frame under this leave-one-out model (identity under session_z)
            if _spf:
                _XEh = np.array(XE, float)
                _XEh[GE == si] = ff_s.apply(XE[GE == si], si, "loso")
            else:
                _XEh, _XUh = _ss.state_scored(m, _ss.model_prior(m, _yd), XE, GE, XU, GU, _all_e, _all_u, [si])
            pr = m.predict_proba(_XEh[hold])[:, 1]
            lh = late_e[hold]
            pre_e.extend(pr[~lh])
            pre_l.extend(pr[lh])
        if pre_e and pre_l:
            disc["drift_control_pre_lick"] = {
                "n_early": len(pre_e), "n_late": len(pre_l),
                "early_p_disengaged": float(np.mean(pre_e)),
                "late_p_disengaged": float(np.mean(pre_l))}

        # D1. DRIFT CONTROL: both halves are ENGAGED, so any separation here is time, not state.
        eng5 = ~u_pre & nl_eng
        if eng5.sum() >= 20:
            early = XUs[eng5 & ~late_nl]
            lateX = XUs[eng5 & late_nl]
            disc["drift_control"] = {
                "n_early": len(early), "n_late": len(lateX),
                "early_p_disengaged": (float(np.mean(model.predict_proba(early)[:, 1]))
                                       if len(early) else None),
                "late_p_disengaged": (float(np.mean(model.predict_proba(lateX)[:, 1]))
                                      if len(lateX) else None)}
    if verbose:
        print(f"  {animal}: " + "  ".join(
            f"{k.split('_', 1)[0]}n={dec[k]['n']}" for k in STATES), flush=True)
    return {"animal": animal, "align": align, "basis_id": basis.basis_id, "ncomp": basis.ncomp,
            "chance": CHANCE, "decode": dec, "discriminator": disc,
            "sessions": list(kept), "n_pre_sessions": len(pre_i)}


ENGSTATES_FROZEN_ENV = "WIDEFIELD_ENGSTATES_FROZEN"


def _stored(animal, align, basis, kept, pre_i, n_features, train_data, fit, post_s, scaling=None):
    """``fit()`` through the frozen store (kind ``engagement_states``; default), or in memory when
    WIDEFIELD_ENGSTATES_FROZEN=0. Converted in the cohort-1 rebuild (2026-11-03): this module refitted its pre-stroke
    position decoder and state discriminator in memory on every render (rule 10)."""
    from wfield_local import frozen_models
    if not frozen_models.family_on(ENGSTATES_FROZEN_ENV):
        return fit()
    from wfield_local import spout_scaling
    spec = frozen_models.make_spec(animal, "engagement_states", align=align, source="locanmf",
                                   train_labels=[kept[i] for i in sorted(pre_i)], post_s=float(post_s),
                                   basis_id=basis.basis_id, n_features=n_features, train_data=train_data,
                                   **spout_scaling.spec_kw(scaling))
    payload, status = frozen_models.load_or_fit(spec, fit, meta={"n_pre_sessions": len(pre_i)})
    print(f"  [precue_states] {animal} {align}: engagement_states {frozen_models._slug(spec)} {status}", flush=True)
    return payload


def _spout_frame_block(disc, ff, feat, kept, XE, YE, GE, XU, YU, GU, pre_i, align, e_pre, u_pre, keep_eng,
                       keep_dis):
    """The discriminator's frame bookkeeping (main 26fd5b44 / 4a37cbc8, items B), kept out of `run_animal` so that
    function's body stays as on the other branches. Returns ``(on, spec scaling, e_pre_d, u_pre_d, pre_d, frame,
    keep_eng, keep_dis)``: the discriminator's PRE sets (states 1 / 2, training, the D2 loop) and, when
    ``WIDEFIELD_STATE_DISC_SCALING=spout_frame``, the FrozenFrame its rows are centred by -- `_working_frame` (camera
    WORKING rows, engaged + no-lick; default `spout_scaling.state_disc_centre`) or part A's ``ff`` (engaged rows,
    WIDEFIELD_STATE_DISC_CENTRE=engaged). A pre session without working rows leaves the pre sets. Records
    ``disc["frame"]`` / ``disc["centre_rows"]``; the spec scaling carries the centre."""
    from wfield_local import spout_scaling as _ss
    mode = _ss.state_disc_scaling()
    on = mode == _ss.SCALING_SPOUT_FRAME and bool(ff.scaling)
    disc["frame"] = "spout_frame" if on else (mode or "session_z")
    if not on:
        return False, mode, e_pre, u_pre, set(pre_i), None, keep_eng, keep_dis
    ffw, pre_w = _working_frame(feat, kept, XE, YE, GE, XU, YU, GU, pre_i, align)
    disc["centre_rows"] = "engaged" if ffw is None else "working"
    e_pre_d, u_pre_d, pre_d = e_pre, u_pre, set(pre_i)
    if ffw is not None:
        drop = sorted(set(pre_i) - set(pre_w))
        if drop:
            disc["pre_sessions_without_working_rows"] = [kept[i] for i in drop]
            keep_eng = keep_eng & ~np.isin(GE, drop)
            keep_dis = keep_dis & ~np.isin(GU, drop)
            e_pre_d, u_pre_d, pre_d = np.isin(GE, pre_w), np.isin(GU, pre_w), set(pre_w)
    return (True, f"{_ss.SCALING_SPOUT_FRAME}/{disc['centre_rows']}", e_pre_d, u_pre_d, pre_d,
            ff if ffw is None else ffw, keep_eng, keep_dis)


def _working_frame(feat, kept, XE, YE, GE, XU, YU, GU, pre_i, align):
    """``(FrozenFrame over the camera WORKING rows, pre sessions that have them)`` for the discriminators'
    ``spout_frame`` (main 4a37cbc8, Priya 11-04, item B): the rows the frozen cue / precue spout decoders are trained
    and centred on (contact + incomplete, engaged AND no-lick rows; rebuild: live classes, `trial_classes.in_sel`), so
    ``train_apply`` centres a pre session on its working rows' true spout-class means and ``apply`` centres any session
    on the label-free EM centre of its working rows. ``(None, pre_i)`` when the centre is the engaged rows:
    ``WIDEFIELD_STATE_DISC_CENTRE=engaged``, the lick window, the DAQ gate / a non-camera working selection, or a
    builder without cue indices."""
    from wfield_local import spout_scaling as _ss
    from wfield_local import trial_classes as _tc
    from wfield_local import working_trials as wt
    if (_ss.state_disc_centre() != "working" or align == "lick" or not _tc.camera_gate_on()
            or not wt.is_camera(_tc.working_selection()) or not getattr(feat, "indices", None)):
        return None, sorted(pre_i)
    ce, cu = _tc.pool_row_classes(feat, kept, len(YE), len(YU))
    we, wu = _tc.in_sel(ce, "working"), _tc.in_sel(cu, "working")
    RX = np.vstack([XE[we], XU[wu]])
    RY = np.concatenate([YE[we], np.asarray(YU)[wu].astype(np.asarray(YE).dtype)])
    RG = np.concatenate([np.asarray(GE)[we], np.asarray(GU)[wu]])
    RC = np.concatenate([ce[we], cu[wu]])
    pre_w = sorted(i for i in pre_i if np.any(RG == i))
    return _ss.FrozenFrame(RX, RY, RG, pre_w, _pipe, loso=True, cls_e=RC, labels=dict(enumerate(kept))), pre_w


def _auc(y, p):
    """ROC AUC without importing another metric; ties handled by rank averaging."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    r = np.argsort(np.argsort(np.asarray(p, float))) + 1.0
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def figure(results, out, align="precue"):
    """Two panels: position decoding per state, and the pre-stroke discriminator's verdict."""
    animals = [a for a in sorted(results) if results[a]]
    if not animals:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2))
    w = 0.8 / max(len(animals), 1)
    x = np.arange(len(STATES))
    cols = plt.get_cmap("tab10")(np.linspace(0, 0.4, max(len(animals), 2)))
    for k, an in enumerate(animals):
        r = results[an]
        acc = [(r["decode"].get(s, {}) or {}).get("accuracy") for s in STATES]
        ns = [(r["decode"].get(s, {}) or {}).get("n", 0) for s in STATES]
        xs = x + (k - (len(animals) - 1) / 2) * w
        axes[0].bar(xs, [a if a is not None else 0 for a in acc], w, color=cols[k],
                    edgecolor="k", linewidth=0.4, label=an)
        for xi, a, n in zip(xs, acc, ns):
            if a is None:
                axes[0].text(xi, 0.01, f"n={n}", ha="center", va="bottom", fontsize=6,
                             rotation=90, color="firebrick")
        d = r["discriminator"]
        pv = [d.get(s) for s in STATES]
        axes[1].bar(xs, [p if p is not None else 0 for p in pv], w, color=cols[k],
                    edgecolor="k", linewidth=0.4, label=an)
    axes[0].axhline(CHANCE, color="k", ls=":", lw=1.2)
    axes[0].set_ylabel("position decoding accuracy (frozen pre-stroke decoder)")
    axes[0].set_title(f"A. Position information in the {align} window, by state\n"
                      "dotted = chance (1/6)", fontsize=10)
    axes[1].axhline(0.5, color="k", ls=":", lw=1.2)
    # the camera-class discriminator (Priya 11-05 (f) 1) names its classes; the DAQ one keeps its original label
    cam = any((results[a].get("discriminator") or {}).get("train_classes") for a in animals)
    axes[1].set_ylabel("P(disengaged) under the PRE-STROKE " + (
        "working (contact + incomplete) vs no-response discriminator" if cam else "lick/no-lick discriminator"))
    axes[1].set_title("B. Where each state falls on the pre-stroke ENGAGEMENT axis\n"
                      "trained late-window only, so it is not a clock"
                      + ("; stopped / jaw-only never trained on" if cam else ""), fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n", 1) for s in STATES], fontsize=7.5)
        ax.legend(fontsize=8)
    fig.suptitle("Pre-cue activity across five lick/engagement states, shared joint-LocaNMF basis. "
                 "REPORTED ONLY — no existing analysis filters on this.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = Path(out) / f"precue_states_{align}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--align", default="precue", choices=("precue", "cue"))
    args = ap.parse_args(argv)
    from wfield_local import run_mode
    # run-mode routing (rebuild 11-03): a tagged run (camera gate, hemo variant ...) writes figures_working/<tag>/,
    # never over the default precue_states_<align>.{png,json}; identity for the default run
    out = run_mode.routed_output(args.output or Path(PathResolver().root("figures_working")))
    out.mkdir(parents=True, exist_ok=True)
    animals = config.normalize_animals(args.animals) or [a for a in config.animals()]
    results = {}
    for an in animals:
        try:
            results[an] = run_animal(an, align=args.align)
        except Exception as ex:                                           # noqa: BLE001
            print(f"  !! {an}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
            results[an] = None
    p = figure(results, out, align=args.align)
    (out / f"precue_states_{args.align}.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"wrote {p} and precue_states_{args.align}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
