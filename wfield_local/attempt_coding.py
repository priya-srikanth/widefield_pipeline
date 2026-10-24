"""Is the spout PLAN there when the animal attempts and fails? Video attempt classes on frozen pre-stroke directions.

    WIDEFIELD_LICK_SOURCE=lp python -m wfield_local.attempt_coding [--animals PS92 ...] [--aligns precue cue lick]

Priya 10-24, goal question: what does cortex "look like" post-stroke when the animal attempts a movement it cannot
execute (or executes differently)? This is item 1 of that plan, and the redo of `position_coding_directions` /
`miss_vs_stopped` that their own docstrings asked for: the August classes were defined by SPOUT CONTACT, which cannot
split "tongue out, no contact" (a motor failure) from "no attempt" (a plan failure). The video attempt classes
(`scripts/trial_attempts.py`, rebuilt 10-24 on complete cam1) can.

CLASSES (post-stroke, per spout; pre-stroke reference = video HITS at that spout):
  hit          attempted, contact (may be executed differently -- that is item 2's question)
  incomplete   attempted (tongue out within max_rt), no contact          -> "cannot execute"
  noattempt    no attempt, animal still working (cls no_try)              -> plan not initiated
  reaction     jaw movement judged a reaction, not an attempt (cls jaw_only; Priya's blind review 10-21), still working
  stopped      after the session's final sustained collapse of attempting at the reference spouts (engagement_gate on
               VIDEO attempts) -- the "no task state" floor
  late         attempted after max_rt (reported for n only)

READOUTS, per trial, from directions fitted on pre-stroke hits only (`position_coding_directions.direction`):
  own          projection on ITS spout's axis, pole-normalised (0 = pre-stroke hits elsewhere, 1 = pre-stroke hits here)
               -- the August number, kept for comparison. No "own minus other axes" index: the six one-vs-rest axes
               sum to ~0 (STATUS_2026-09-24_CD_TRAJECTORIES), so the other axes' mean is nearly a linear function of
               own and the index adds nothing (caught by the unit test, 10-24). A trial-wide state change is handled
               by the CONTRAST, not by the readout: every class is compared with `stopped` in the same sessions.
  p_true, pred the STORED frozen pre-stroke position decoder (`locanmf_frozen_decoder.frozen_decoder_models`, the
               DAQ-frozen model LP_licks reuses -- Priya 10-21): probability of the true spout and the decoded
               spout, per class (the "distance kept, side lost" confusion of the LP frozen decoders, now split by
               what the animal did). Pre-stroke rows use the stored leave-that-session-out model.
Pre-stroke hits are scored LEAVE-ONE-SESSION-OUT (directions refitted without that day), so the 1.0 reference is not
in-sample.

WINDOWS: precue = ENL (2 s before the cue, lick-free) is the clean plan test -- nothing of THIS trial's movement is in
it. cue = 0-post_s after the cue: contains the attempt's movement for hit / incomplete, none for noattempt / stopped,
so a class difference there mixes plan and movement. lick = aligned to the first VIDEO lick (measured for hit AND
incomplete -- the LP source times incomplete trials by their own tongue), inferred (session median RT) for the rest.

OUTPUT <run>/imaging_encoding/attempt_coding/: trials_<align>_<animal>.csv.gz (one row per trial x method),
summary_<align>.csv (animal x epoch x spout x class: mean / sem / n, sessions), contrasts_<align>.csv (within-session
permutation tests of the planned contrasts, per animal), figures/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

#: spout names: POSITION_NAMES (mouse-side L/R) -> lesion-side names (all lesions LEFT: contra = the mouse's RIGHT)
LESION_NAME = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
               "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
SPOUTS = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
CLASSES = ("pre_hit", "hit", "incomplete", "noattempt", "reaction", "stopped", "late")
POST_EPOCHS = ("acute", "subacute", "chronic")
#: "dom" only: the mean-difference direction has nothing fitted, so it is a summary of the pre-stroke hits rather than a
#: model (CLAUDE.md rule 10 freezes every pre-stroke DECODER; the 6-way readout below is the frozen one). The August
#: "lr" variant would be an unfrozen in-memory fit and is not carried.
METHODS = ("dom",)
ALIGNS = ("precue", "cue", "lick")
#: planned contrasts (a, b): the plan-intact test is incomplete / noattempt vs stopped; hit vs incomplete asks whether
#: failing to reach changes the code; incomplete vs pre_hit is "how much of the normal signature".
CONTRASTS = (("incomplete", "stopped"), ("noattempt", "stopped"), ("reaction", "stopped"),
             ("incomplete", "noattempt"), ("hit", "incomplete"))
MIN_N = 5           # per class x spout x session to enter a contrast
N_PERM = 2000


def classify(cls_e, cls_u, eng_pre_e, eng_pre_u, not_eng_u):
    """Class per engaged (XE) row and per no-lick (XU) row.

    Under the LP source, XE = attempted within max_rt (hit / incomplete) and XU = everything else. ``cls_*`` are the
    attempts classes of those rows ("" = unmatched), ``eng_pre_*`` True on pre-stroke sessions, ``not_eng_u`` the
    video engagement gate (True = after the session's final quit)."""
    ce = np.where(eng_pre_e, np.where(cls_e == "hit", "pre_hit", ""),
                  np.where(np.isin(cls_e, ("hit", "incomplete")), cls_e, ""))
    ce = ce.astype(object)
    cu = np.full(len(cls_u), "", dtype=object)
    post_u = ~eng_pre_u
    cu[post_u & not_eng_u & ~np.isin(cls_u, ("", "unknown"))] = "stopped"
    work = post_u & ~not_eng_u
    cu[work & (cls_u == "no_try")] = "noattempt"
    cu[work & (cls_u == "jaw_only")] = "reaction"
    cu[work & np.isin(cls_u, ("hit", "incomplete"))] = "late"
    return ce, cu


def fit_axes(X, spout, method):
    """Per-spout unit directions + poles from pre-stroke hits (rows X, spout names). {spout: (w, p0, p1)}."""
    from wfield_local.position_coding_directions import direction, poles
    W = {}
    for P in SPOUTS:
        mp = spout == P
        if mp.sum() < 5 or (~mp).sum() < 5:
            continue
        w = direction(X[mp], X[~mp], method)
        W[P] = (w, *poles(X[mp], X[~mp], w))
    return W


def score(X, spout, W):
    """Pole-normalised projection of each row on its own spout's axis; NaN where that spout has no axis."""
    from wfield_local.position_coding_directions import project
    own = np.full(len(X), np.nan)
    for P, (w, p0, p1) in W.items():
        m = spout == P
        if m.any():
            own[m] = project(X[m], w, p0, p1)
    return own


def perm_test(vals, cls, sess, a, b, n_perm=N_PERM, seed=0):
    """Mean(a) - mean(b), session-balanced: per session with >= MIN_N of each, the difference of class means; the
    statistic is the mean over sessions, the null shuffles class labels WITHIN each session. Returns
    (diff, p_two_sided, n_sessions, n_a, n_b)."""
    rng = np.random.default_rng(seed)
    groups = []
    for s in np.unique(sess):
        m = (sess == s) & np.isin(cls, (a, b)) & np.isfinite(vals)
        ia = cls[m] == a
        if ia.sum() >= MIN_N and (~ia).sum() >= MIN_N:
            groups.append((vals[m], ia))
    if not groups:
        return np.nan, np.nan, 0, 0, 0
    stat = np.mean([v[i].mean() - v[~i].mean() for v, i in groups])
    null = np.empty(n_perm)
    for k in range(n_perm):
        d = []
        for v, i in groups:
            j = rng.permutation(i)
            d.append(v[j].mean() - v[~j].mean())
        null[k] = np.mean(d)
    p = (1 + np.sum(np.abs(null) >= abs(stat))) / (n_perm + 1)
    return float(stat), float(p), len(groups), int(sum(i.sum() for _, i in groups)), \
        int(sum((~i).sum() for _, i in groups))


def _cue_classes(labels, max_rt):
    """{label: (cls per DAQ cue, trial_id per DAQ cue)} from the LP matcher (same matching the features use)."""
    from wfield_local import lp_licks
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events
    out = {}
    for lab in labels:
        s = next((x for x in SESSIONS if x["label"] == lab), None)
        if s is None:
            continue
        cue = _load_cue_events(s["h5"])
        lp = lp_licks.session_lp(lab, cue["cue_samples"], cue["sample_rate_hz"], max_rt)
        out[lab] = (np.asarray(lp["cls"], object).astype(str), np.asarray(lp["trial_id"], int))
    return out


def run_animal(animal, align="precue", verbose=True):
    """Trial table for one animal and window (None if unavailable)."""
    from wfield_local import config, joint_locanmf, lp_licks
    from wfield_local.locanmf_frozen_decoder import pool_sessions, _args
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    from wfield_local.position_coding_directions import _gate_all
    from wfield_local.precue_engagement_states import features_with_indices
    if not lp_licks.is_lp():
        raise RuntimeError(f"set {lp_licks.LICK_SOURCE_ENV}=lp: the classes are video attempts")
    post_s = float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    if not pre or not post:
        return None
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
    basis = joint_locanmf.load(animal, sessions=_ALL)
    feat = features_with_indices(basis, nolick_ref=("would_be_lick" if align == "lick" else "cue"))
    pooled = pool_sessions(pre + post, source="locanmf", align=align, post_s=post_s, features=feat)
    if pooled is None:
        return None
    XE, YE, GE, _BE, XU, YU, kept, _c, GU = pooled
    YU = YU.astype(int)
    g = _gate_all(feat, kept, XE, YE, GE, XU, YU, GU)
    if g is None:
        raise RuntimeError(f"{animal} {align}: trial bookkeeping mismatch")
    not_eng, _fnl, _fe = g

    # attempts class + trial_id per feature row, through the trial (DAQ cue) indices the features kept
    cc = _cue_classes(kept, _args("locanmf", align, post_s).max_rt)
    cls_e, cls_u, tid_e, tid_u = [], [], [], []
    for lab in kept:
        ie, inl = feat.indices[lab]
        c, t = cc.get(lab, (np.array([], str), np.array([], int)))
        cls_e += [c[k] if k < len(c) else "" for k in ie]
        tid_e += [int(t[k]) if k < len(t) else -1 for k in ie]
        cls_u += [c[k] if k < len(c) else "" for k in inl]
        tid_u += [int(t[k]) if k < len(t) else -1 for k in inl]
    cls_e, cls_u = np.array(cls_e, object).astype(str), np.array(cls_u, object).astype(str)
    if len(cls_e) != len(XE) or len(cls_u) != len(XU):
        raise RuntimeError(f"{animal} {align}: class rows {len(cls_e)}/{len(XE)}, {len(cls_u)}/{len(XU)}")

    pre_set = set(pre)
    pre_i = np.array([lab in pre_set for lab in kept])
    ce, cu = classify(cls_e, cls_u, pre_i[GE], pre_i[GU], not_eng)
    X = np.vstack([XE, XU])
    G = np.concatenate([GE, GU]).astype(int)
    C = np.concatenate([ce, cu]).astype(str)
    T = np.concatenate([tid_e, tid_u]).astype(int)
    sp = np.array([LESION_NAME.get(POSITION_NAMES.get(int(v), str(v)), "?") for v in np.concatenate([YE, YU])])
    keep = C != ""
    X, G, C, T, sp = X[keep], G[keep], C[keep], T[keep], sp[keep]

    A = pd.read_csv(lp_licks.attempts_path(), usecols=["label", "epoch", "day"]).drop_duplicates("label")
    ep = dict(zip(A.label, A.epoch)); day = dict(zip(A.label, A.day))
    lab_row = np.array(kept, dtype=object)[G]
    rows = []
    ref = C == "pre_hit"
    for meth in METHODS:
        W = fit_axes(X[ref], sp[ref], meth)
        own = score(X, sp, W)
        # pre-stroke hits: leave-one-session-out, so the reference is not scored in-sample
        for s in np.unique(G[ref]):
            tr, te = ref & (G != s), ref & (G == s)
            own[te] = score(X[te], sp[te], fit_axes(X[tr], sp[tr], meth))
        rows.append(pd.DataFrame({"method": meth, "own": own}))
    from wfield_local.locanmf_frozen_decoder import frozen_decoder_models
    pre_list = [i for i in range(len(kept)) if pre_i[i]]
    models, status = frozen_decoder_models(XE, YE, GE, kept, pre_list, align=align, source="locanmf",
                                           post_s=post_s, basis_id=getattr(feat, "basis_id", None))
    if verbose:
        print(f"[attempt_coding] {animal} {align}: frozen decoder {status}", flush=True)
    code = np.concatenate([YE, YU]).astype(int)[keep]
    p_true = np.full(len(X), np.nan)
    pred = np.full(len(X), "", dtype=object)
    for s in np.unique(G):
        m = G == s
        mdl = models["loso"].get(kept[s], models["full"])
        pr = mdl.predict_proba(X[m]); cl = [int(c) for c in mdl.classes_]
        idx = np.array([cl.index(c) if c in cl else -1 for c in code[m]])
        p_true[m] = np.where(idx >= 0, pr[np.arange(m.sum()), np.clip(idx, 0, None)], np.nan)
        pred[m] = [LESION_NAME.get(POSITION_NAMES.get(cl[j], str(cl[j])), "?") for j in pr.argmax(1)]
    base = pd.DataFrame({"animal": animal, "align": align, "label": lab_row,
                         "epoch": [ep.get(l, "?") for l in lab_row], "day": [day.get(l, np.nan) for l in lab_row],
                         "trial_id": T, "spout": sp, "cls": C, "p_true": p_true, "pred": pred})
    out = pd.concat([pd.concat([base, r], axis=1) for r in rows], ignore_index=True)
    if verbose:
        n = out[out.method == METHODS[0]].groupby(["epoch", "cls"]).size().unstack(fill_value=0)
        print(f"[attempt_coding] {animal} {align}: {len(kept)} sessions\n{n}", flush=True)
    return out


def summarise(T):
    """animal x epoch x spout x class x method: mean / sem / n of own and p_true, decoded-spout shares; n sessions."""
    rows = []
    for key, d in T.groupby(["animal", "align", "method", "epoch", "spout", "cls"]):
        r = dict(zip(["animal", "align", "method", "epoch", "spout", "cls"], key))
        r["n"] = len(d); r["n_sessions"] = d.label.nunique()
        for v in ("own", "p_true"):
            x = d[v].to_numpy(float); x = x[np.isfinite(x)]
            r[v] = float(x.mean()) if len(x) else np.nan
            r[v + "_sem"] = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
        for s in SPOUTS:
            r["pred_" + s] = float((d.pred == s).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def contrasts(T):
    """Planned within-session permutation contrasts, per animal x epoch x spout x method x readout."""
    rows = []
    for key, d in T[T.cls != "pre_hit"].groupby(["animal", "align", "method", "epoch", "spout"]):
        for a, b in CONTRASTS:
            for v in ("p_true", "own"):
                diff, p, ns, na, nb = perm_test(d[v].to_numpy(float), d.cls.to_numpy(str), d.label.to_numpy(str), a, b)
                if ns:
                    rows.append(dict(zip(["animal", "align", "method", "epoch", "spout"], key),
                                     a=a, b=b, readout=v, diff=diff, p=p, n_sessions=ns, n_a=na, n_b=nb))
    return pd.DataFrame(rows)


def freeze_reference(animal, align):
    """Fit and store the DAQ-path frozen decoder this module reads, when it is not stored yet (DAQ mode only).

    Under LP the ENL window is 1.0 s (`lp_licks.LP_ENL_WINDOW_S`) and the frozen pre-cue spec carries the window, so the
    LP run needs the DAQ 1.0 s model, frozen by a DAQ pass FIRST -- the same order as the 10-21 DAQ_ENL1s arm, which
    froze them for the families it rendered but not for the joint-basis decoder used here. Run as
    ``WIDEFIELD_ENL_WINDOW_S=1.0 python -m wfield_local.attempt_coding --freeze`` (lick source unset). A new spec gets
    its own id; the canonical 2.0 s models are untouched (CLAUDE.md rule 10)."""
    from wfield_local import config, joint_locanmf, lp_licks
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
    from wfield_local.locanmf_frozen_decoder import frozen_decoder_models, pool_sessions
    from wfield_local.precue_engagement_states import features_with_indices
    if lp_licks.is_lp():
        raise RuntimeError("--freeze runs the DAQ path: unset WIDEFIELD_LICK_SOURCE")
    post_s = float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    basis = joint_locanmf.load(animal, sessions=_ALL)
    feat = features_with_indices(basis)
    XE, YE, GE, _BE, _XU, _YU, kept, _c, _GU = pool_sessions(pre + post, source="locanmf", align=align,
                                                              post_s=post_s, features=feat)
    pre_list = [i for i, lab in enumerate(kept) if lab in set(pre)]
    _m, status = frozen_decoder_models(XE, YE, GE, kept, pre_list, align=align, source="locanmf", post_s=post_s,
                                       basis_id=getattr(feat, "basis_id", None))
    print(f"[attempt_coding] freeze {animal} {align} (ENL window {lp_licks.enl_window_s()}): {status}", flush=True)
    return status


def out_dir():
    from wfield_local import lp_licks
    return lp_licks._run_dir() / "imaging_encoding" / "attempt_coding"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--animals", nargs="+", default=["PS92", "PS93", "PS94", "PS95"])
    ap.add_argument("--aligns", nargs="+", default=list(ALIGNS), choices=ALIGNS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--freeze", action="store_true", help="DAQ mode: store the frozen decoders the LP run reads, then exit")
    a = ap.parse_args(argv)
    if a.freeze:
        for an in a.animals:
            for align in a.aligns:
                freeze_reference(an, align)
        return 0
    od = a.out or out_dir()
    od.mkdir(parents=True, exist_ok=True)
    for align in a.aligns:
        parts = []
        for an in a.animals:
            f = od / f"trials_{align}_{an}.csv.gz"
            if f.exists():
                parts.append(pd.read_csv(f)); continue
            T = run_animal(an, align)
            if T is None:
                print(f"[attempt_coding] {an} {align}: unavailable", flush=True); continue
            T.to_csv(f, index=False); parts.append(T)
        if not parts:
            continue
        T = pd.concat(parts, ignore_index=True)
        summarise(T).to_csv(od / f"summary_{align}.csv", index=False)
        contrasts(T).to_csv(od / f"contrasts_{align}.csv", index=False)
        print(f"[attempt_coding] {align}: wrote summary / contrasts ({len(T)} rows)", flush=True)
    (od / "run.json").write_text(json.dumps({"lick_source": os.environ.get("WIDEFIELD_LICK_SOURCE", ""),
                                             "animals": a.animals, "aligns": a.aligns}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
