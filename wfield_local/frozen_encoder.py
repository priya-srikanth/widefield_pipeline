"""FROZEN pre-stroke encoder in Allen-area space (Priya 10-22: "yes - and do frozen cue-aligned and lick-aligned also").

`combined_encoding` fits every session on its own (kernels, penalties, z-scoring and the movement-first
orthogonalisation all per session), so its split of an acute change into task / movement parts cannot tell FEWER /
SMALLER movements from a CHANGED cortical response to the same movement. Here one model per animal is fitted on its
August pre-stroke sessions and applied, unchanged, to every session:

- regressors: the encoder's full-model variables (`combined_encoding.GROUP_OF`: cue, arrival, the six position
  deviation blocks, lick kernels, lick angle, tongue, jaw, nose, running) from `raw_blocks`, in RAW units, slow drift
  projected out per session exactly as the encoder does (`_slow_basis`, DRIFT_MIN_PERIOD_S; the constant is in the
  basis, so every column and target is centred per session);
- SCALING: each column divided by the animal's POOLED pre-stroke SD (not the session's own), so a post-stroke session
  with smaller / rarer licks keeps smaller regressors -- the point of freezing. Pose traces are in LP pixels on a fixed
  rig (v36 timing / June rescale), so their units are comparable across sessions;
- NO ORTHOGONALISATION: all groups are fitted jointly (one ridge, the encoder's penalty groups `PEN_GROUP_OF`), so no
  group is credited by ordering. The per-session encoder residualises task on movement (movement_first), a second
  reason it credits movement; this model does not;
- target: area activity Rd = Y @ Ra (the encoder's area averages on the eroded stat mask, + "cortex"), per area name;
  areas absent from any of the animal's sessions are dropped for that animal;
- penalties: per penalty group, coordinate search (2 passes over ALPHA_GRID, as the encoder) on LEAVE-ONE-PRE-SESSION-
  OUT R^2 summed over areas; pre sessions are then scored with the model fitted WITHOUT them, post sessions with the
  model fitted on all of the animal's August pre sessions.

GROUPS (STATS_VERSION 4, 10-22 afternoon; see FGROUP): timing / target / exec_locked / exec_within / lick / other /
contact, built by splitting every lick-related movement column into the animal's PRE-STROKE spout-locked template (s)
and the deviation from it (d) -- a behaviour-only first pass over the pre sessions (`template_job`, saved as
fit/templates.pkl).

Per session the output matches `combined_encoding`'s session files where the decomposition reads them: `peth` (cue-
aligned) and `peth_lick` (first post-cue lick), parts data / full / each group / spout / lick_all / pos (joint and the
ordered variants; data / full only for the reduced models), x position x area x offset (PETH_S), plus the frozen
model's R^2 (all / cue / lick windows, cortex). `data - full` = activity the
FROZEN model does not predict: at acute far contra, negative = cortex responds LESS than the pre-stroke mapping
predicts from the movements actually made.

Usage (same env as the encoder run whose trials / licks it should use, e.g. OROFACIAL_GATE=attempted
COMBINED_ENCODING_LICKS=combined): python -m wfield_local.frozen_encoder --all --jobs 6 [--tag gate-attempted]
Output: <E>/frozen_encoder_<tag>/<variant>/sessions/<label>.{npz,json}, fit/<animal>_<variant>.npz,
fit/pre_stats.pkl.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

STATS_VERSION = 7   # 2: FGROUP split (contact / direction / movement); 3: protrusion -> direction; 4: lick / spout split;
#                     5: running capped at the pre p99 of ALL frames (superseded); 6: p99 of RUNNING frames (RUN_CAP_Q);
#                     7: default running = forward-only hinges from RUN_KNOT0_DEFAULT mm/s (spline_thr), no push term

#: RUNNING CAP (Priya 10-22 evening: "yes add running cap"). A frozen model only interpolates within the pre-stroke
#: range: PS94 ran far harder after the stroke (running SD 5.6-6.3 vs 1.4-2.9 pre; p99 20-23 vs 7-12), the linear
#: running kernel extrapolated and drove chronic frozen R2 to -0.4 .. -0.6. Every running column is clipped to
#: +/- the animal's pooled pre-session RUN_CAP_Q percentile of |speed| over its RUNNING frames (trial frames, lag-0
#: column, |speed| > the pipeline's running threshold `segmentation.running.thresh_speed_mm_s`, 3 mm/s) before
#: centring. NOT the percentile of all frames (STATS_VERSION 5): PS95 runs on 1 % of pre frames, so its all-frame p99
#: (2.8 mm/s) sat at the edge of standing still, the cap removed 63 % of its pre running SD, the per-mm/s kernel
#: tripled and post-stroke cue / lick-window R2 fell to -0.65 (chronic). The cap = the speeds the animal REACHED when
#: it ran, independent of how often it ran (PS92 / 93 / 94 / 95: 14.8 / 17.3 / 13.2 / 16.8 mm/s).
RUN_CAP_Q = 99.0


def _run_thresh() -> float:
    from wfield_local import config
    return float(config.defaults()["segmentation"]["running"]["thresh_speed_mm_s"])


def _ce():
    from wfield_local import combined_encoding as ce
    return ce


#: FROZEN-MODEL GROUPS (Priya 10-22, "yes change as you suggest"): finer than the encoder's task / instructed /
#: uninstructed, because two things the encoder lumps into "instructed" are not movement magnitude:
#:   contact    the spout-CONTACT kernel (inside the encoder's `licks` block): a touch / outcome event that happens only
#:              when the lick reaches the target -- acute far contra has few, so its missing contact-evoked activity
#:              would otherwise be credited to "movement";
#:   direction  WHERE the tongue / jaw / nose go, on either axis of position: tongue x / y, tongue PROTRUSION (reach
#:              magnitude -- unsigned but it scales with spout DISTANCE, the other position axis; 10-22 audit, Priya
#:              "ensure our variable groups appropriately include or exclude ... directionless ... vs direction-
#:              specific"), jaw x, nose x, per-lick executed angle (within-spout deviation, zero mean per spout):
#:              target-correlated, since the tongue goes toward the spout (the 10-19 no-kinematics reasoning);
#:   movement   DIRECTIONLESS timing and vigour only: lick onsets (cam4 + cam1-only), tongue speed (unsigned), jaw
#:              opening (y) and speed (unsigned), nose y (up / down orienting), running. Residual target information
#:              in these (e.g. wider opening for far spouts) is MEASURED, not assumed: `scripts/frozen_group_leakage.py`;
#:   task       cue, arrival, the six position deviation blocks.
#: v3 layout above was REPLACED 10-22 afternoon (STATS_VERSION 4; Priya: "should we have a 'lick' group and a 'spout
#: position information' group?" -> "yes you can build the modified model"). Why: the leakage check
#: (`scripts/frozen_group_leakage.py`) found the "directionless" group decoding the spout at 0.60-0.69 (chance 0.17;
#: jaw opening 0.40-0.57, lick timing ~0.3), and licks / tongue / jaw are redundant readouts of one act, so their
#: separate drops split credit arbitrarily. The split is now BY CONSTRUCTION, not by variable name:
#:   every lick-related movement column x (lick onsets, tongue speed, jaw opening / speed, and the direction traces
#:   tongue x / y / protrusion, jaw x, nose x) = SPOUT-LOCKED part s + DEVIATION d, where s = the animal's PRE-STROKE
#:   mean trace for that spout at that time from the cue (`TEMPLATE_S`; pooled over its August pre sessions, each
#:   session centred) and d = x - s. s is "what the animal's licking looks like at this spout"; d is everything else,
#:   INCLUDING post-stroke changes in vigour / timing at a spout (s is frozen with the model, so a weaker acute lick
#:   at far contra lands in d = the LICK group, not in the spout group).
#: Groups:
#:   timing       cue, arrival (shared kernels; spout-independent)
#:   target       pos_enl, pos_cue (the six position blocks: the spout itself)
#:   exec_locked  spout-locked part s of every lick-related column (executed, but fully predictable from the spout)
#:   exec_within  deviation d of the DIRECTION traces + the per-lick executed angle (already within-spout): where the
#:                tongue / jaw go relative to that spout's usual path (nose moved to "other", grouping v2)
#:   lick         deviation d of lick onsets, tongue speed, jaw opening / speed: vigour and timing that is not
#:                about which spout (leakage-checked: should decode the spout ~ chance in pre sessions)
#:   other        nose x / y (both parts, grouping v2), running, push
#:   contact      the spout-contact kernel
#: Super-groups: task = timing + target; spout = target + exec_locked + exec_within; movement = lick + exec_locked +
#: exec_within + other. target vs exec_locked are perfectly confounded by design (both follow the spout); only
#: exec_within separates them, so read target / executed only as the ordered bracket (target_first / exec_first).
SPLIT_VARS = ("lick_onset", "tongue_mag", "jaw_mag", "tongue_dir", "jaw_dir", "nose_dir")
_LICK_BASE = ("lick_onset", "tongue_mag", "jaw_mag")
TEMPLATE_S = (-4.0, 8.0)                                 # cue-relative span of the spout-locked templates (s)
#: NOSE IS "OTHER" (Priya 10-23: "why is nose direction in the lick group? did we show it's lick locked?" -> no: only
#: 4 / 21 / 29 / 13 % of nose x's pre-stroke variance (PS92-95) is in its spout-locked template, vs 44-61 % for lick
#: onsets; "nose_dir" = nose lateral POSITION, "nose_mag" = nose vertical position, both mouth-relative). Both parts of
#: nose x go to "other" with nose y and running (uninstructed). Grouping only: same columns / statistics / joint fit.
_NOSE_OTHER = {"nose_dir:spl", "nose_dir:dev"}
FGROUP = {"cue": "timing", "arrival": "timing", "pos_enl": "target", "pos_cue": "target",
          **{f"{v}:spl": "exec_locked" for v in SPLIT_VARS},
          **{f"{v}:dev": ("lick" if v in _LICK_BASE else "exec_within") for v in SPLIT_VARS},
          **{k: "other" for k in _NOSE_OTHER},
          "exec_angle": "exec_within", "nose_mag": "other", "running": "other", "push": "other", "contact": "contact"}
#: grouping revision, saved with outputs (1 = up to 10-23 evening, nose x in exec_locked / exec_within; 2 = nose in other;
#: 3 = contact its own category, out of "nontask" -> task_first / movement_first order task vs movement only)
GROUPING_VERSION = 3
FGROUPS = ("timing", "target", "exec_locked", "exec_within", "lick", "other", "contact")
#: CONTACT IS ITS OWN CATEGORY (grouping v3, Priya 10-23 "ok split out contact"): touch feedback from the animal's own
#: lick is neither an experimenter-controlled task event nor a movement, so it is no longer inside "nontask" -- the
#: task-first / movement-first bracket now orders task vs MOVEMENT with contact split jointly (in neither); the old
#: "non-task" (movement + contact) is still joint - task_only.
SUPER = {"task": ("timing", "target"), "spout": ("target", "exec_locked", "exec_within"),
         "movement": ("lick", "exec_locked", "exec_within", "other"),
         "nontask": ("lick", "exec_locked", "exec_within", "other")}
#: penalty groups as the encoder's PEN_GROUP_OF (task / lick / orofacial / uninstructed); s and d of a variable share
#: its penalty group
_FPEN_BASE = {"cue": "task", "arrival": "task", "pos_enl": "task", "pos_cue": "task",
              "lick_onset": "lick", "contact": "lick", "exec_angle": "lick",
              "tongue_mag": "orofacial", "tongue_dir": "orofacial", "jaw_mag": "orofacial", "jaw_dir": "orofacial",
              "nose_mag": "uninstructed", "nose_dir": "uninstructed", "running": "uninstructed",
              "push": "uninstructed"}
FPEN = {v: _FPEN_BASE[v.split(":")[0]] for v in FGROUP}

#: FACERHYTHM IN THE FROZEN MODEL (Priya 10-25, handoff 000t plan item 3: "should we be mixing frozen and refit encoder
#: vs FR models" -> build the frozen pre-stroke encoder WITH FR). Opt-in: FROZEN_FR=1 plus the encoder's own FR envs
#: (COMBINED_ENCODING_FR = factor set, e.g. master_pre_v1; COMBINED_ENCODING_FR_GROUPS = <grouping>:<mode>, see
#: fr_groups). One lag-0 column per grouped FR trace (as the per-session encoder enters FR), raw absolute expression,
#: scaled by the animal's pooled pre SD like every other column (so a smaller post-stroke face movement keeps a smaller
#: regressor). Groups: "fr" = lick sum + other factors (directionless); "fr_dir" = left / right / close / far (mode A
#: only: lick DIRECTION through FR). Both are MOVEMENT (super-groups movement / nontask), so movement_first credits
#: shared task / FR variance to movement; own penalty group "fr" (searched like the others). no_fr / no_fr_dir = their
#: unique R2 in the FROZEN frame (the model-based decoupling test). Frames where FR is blanked enter at the session mean.
FROZEN_FR_ENV = "FROZEN_FR"
_FR_DIR_PARTS = ("left", "centre", "right", "close", "far")


def _fr_vars() -> dict:
    if os.environ.get(FROZEN_FR_ENV, "") not in ("1", "true", "yes"):
        return {}
    from wfield_local import fr_groups
    gs = fr_groups.spec()
    if not gs or not os.environ.get("COMBINED_ENCODING_FR"):
        raise ValueError(f"{FROZEN_FR_ENV}=1 needs COMBINED_ENCODING_FR (set) and {fr_groups.ENV} (<grouping>:<mode>)")
    G = fr_groups.GROUPINGS[gs[0]]
    out = {}
    for part in fr_groups.MODES[gs[1]]:
        if part == "other":
            out.update({f"fr_other_{k}": "fr" for k in G["other"]})
        elif G.get(part):
            out[f"fr_{part}"] = "fr_dir" if part in _FR_DIR_PARTS else "fr"
    return out


FR_VARS = _fr_vars()
#: FR SPOUT-LOCKED SPLIT (Priya 10-25: "can we group by lick position information vs no lick position information across
#: FR and kinematics?"). FROZEN_FR_SPLIT=1 splits every FR column like the pose columns (SPLIT_VARS): s = the animal's
#: pre-stroke spout-locked template, d = the deviation. Without it the FR lick sum counts as "directionless" although
#: its spout-locked amplitude differs by spout. Groups then: fr_locked (all FR s parts, position information), fr_dir
#: (side / distance d parts), fr (lick sum + other d parts, no position information).
FR_SPLIT = bool(FR_VARS) and os.environ.get("FROZEN_FR_SPLIT", "") in ("1", "true", "yes")
if FR_VARS:
    if FR_SPLIT:
        _frmap = {**{f"{v}:spl": "fr_locked" for v in FR_VARS}, **{f"{v}:dev": g for v, g in FR_VARS.items()}}
        SPLIT_VARS = (*SPLIT_VARS, *FR_VARS)
    else:
        _frmap = dict(FR_VARS)
    _frg = tuple(g for g in ("fr_locked", "fr", "fr_dir") if g in _frmap.values())
    FGROUP.update(_frmap)
    FGROUPS = (*FGROUPS, *_frg)
    SUPER["movement"] = (*SUPER["movement"], *_frg)
    SUPER["nontask"] = (*SUPER["nontask"], *_frg)
    if len(_frg) > 1:
        SUPER["fr_all"] = _frg
    FPEN.update({v: "fr" for v in _frmap})
    #: LICK-POSITION INFORMATION across pose AND FR (Priya 10-25): posinfo = everything that carries where the lick goes
    #: (pose spout-locked parts + pose direction deviations / executed angle + FR spout-locked parts + FR side / distance);
    #: noposinfo = movement without it (pose lick timing / speed / opening deviations, nose, running + FR lick sum / other
    #: deviations). Contact and the task groups are in neither. Without FROZEN_FR_SPLIT the FR lick sum sits in noposinfo
    #: with its spout-locked part (approximate).
    SUPER["posinfo"] = ("exec_locked", "exec_within", *(g for g in ("fr_locked", "fr_dir") if g in _frg))
    SUPER["noposinfo"] = ("lick", "other", *(g for g in ("fr",) if g in _frg))


#: PER-SPOUT LICK KERNELS (Priya 10-25: "should we have done spout-position lick kernels for our current models" -> "i.e.
#: the frozen model"). FROZEN_LICKSPOUT=1 replaces the lick-onset spout-locked + deviation columns (s + d = the recorded
#: lick-onset lag columns) by SIX copies of the recorded columns, each masked to one spout's trial frames (spout = the
#: TARGET spout of the trial; relabel runs use the relabelled spout like the templates). Group "lickspout": executed,
#: spout-specific, lick-locked -> in the spout / movement / posinfo super-groups and on the EXECUTED side of the ordered
#: target / executed bracket. Penalty group "lick". The other pose variables keep their s / d split.
LICKSPOUT = os.environ.get("FROZEN_LICKSPOUT", "") in ("1", "true", "yes")
LICKSPOUT_POS = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")   # canonical order
if LICKSPOUT:
    for _k in ("lick_onset:spl", "lick_onset:dev"):
        FGROUP.pop(_k, None)
    FGROUP.update({f"lick_onset@{q}": "lickspout" for q in LICKSPOUT_POS})
    FGROUPS = (*FGROUPS, "lickspout")
    for _g in ("spout", "movement", "nontask", "posinfo"):
        if _g in SUPER:
            SUPER[_g] = (*SUPER[_g], "lickspout")
    FPEN.update({f"lick_onset@{q}": "lick" for q in LICKSPOUT_POS})
_EXEC_SIDE = ("exec_locked", "exec_within", *(("lickspout",) if LICKSPOUT else ()))
#: LANDING-SPLIT LICK KERNELS (Priya 10-25: "let's try the 3d landing ... lick kernels"): with FROZEN_LICKSPOUT, the six
#: copies of the lick-onset kernel are assigned by where each lick LANDED, not by the trial's target spout: each lick's
#: soft weight over the six spouts' pre-stroke landing centres (3-D peak, within-spout SD units, w ~ exp(-d^2 / 2), width
#: 1; the encoder's executed-bin rule, `exec_bins_calib`). Licks without a 3-D peak and cam1-only licks keep the target
#: spout (the old rule), so the six copies still sum to the recorded lick-onset columns. Same group / penalty.
LICKLAND = os.environ.get("FROZEN_LICKLAND", "") in ("1", "true", "yes", "landing")
LICKLAND_CALIB = os.environ.get("FROZEN_LICKLAND_CALIB",
                                "C:/Users/SabatiniLab/wfield_cache/pose3d_inputs/exec_bins_calib_20261023.csv")
if LICKLAND and not LICKSPOUT:
    raise ValueError("FROZEN_LICKLAND needs FROZEN_LICKSPOUT=1")
#: LAGGED FACERHYTHM (Priya 10-25: "sure let's try lagged kernel"): FROZEN_FR_LAGS = comma-separated lags in s (e.g.
#: "-0.3,-0.2,-0.1,0.1,0.2,0.3"); each FR variable keeps its lag-0 column and gains <v>:facelead (lags > 0: the face
#: value from BEFORE the frame -- face precedes cortex) and <v>:cortexlead (lags < 0: the face value from AFTER the frame
#: -- cortex precedes the face). Same group / penalty as the variable. Variants no_frfacelead / no_frcortexlead drop one
#: side: unique(cortexlead) = future face beyond present + past = cortex leading. Slow calcium biases absolute timing
#: toward cortex lagging (constant across epochs: read the pre -> post CHANGE). Not with FROZEN_FR_SPLIT.
FR_LAGS = tuple(float(x) for x in os.environ.get("FROZEN_FR_LAGS", "").split(",") if x.strip())
if FR_LAGS:
    if not FR_VARS or FR_SPLIT:
        raise ValueError("FROZEN_FR_LAGS needs FROZEN_FR (FaceRhythm) and not FROZEN_FR_SPLIT")
    for _v, _g in list(FR_VARS.items()):
        for _side in ("facelead", "cortexlead"):
            FGROUP[f"{_v}:{_side}"] = _g
            FPEN[f"{_v}:{_side}"] = "fr"
_LAGDROP = {"no_frfacelead": ":facelead", "no_frcortexlead": ":cortexlead"} if FR_LAGS else {}


def lick_landing_weights(B, animal: str, date: str, rv, width: float = 1.0) -> np.ndarray:
    """(n_licks, 6) soft landing weights over LICKSPOUT_POS for each lick of ``B.lick_on_s`` (rows NaN = no 3-D peak).
    Re-joins the session's lick table (same rows / order as `combined_encoding._load_behaviour`) to the 3-D lick table on
    (trial_id, lick_idx)."""
    import pandas as pd

    ce = _ce()
    from wfield_local import orofacial_outputs as oo
    L = pd.read_csv(oo.session_dir(rv, ce._run(), animal, date) / "licks.csv")
    L = L[L.trial_id.isin(set(B.trials.trial_id))]
    if len(L) != len(B.lick_on_s):
        raise ValueError(f"{animal}_{date}: lick table has {len(L)} rows, behaviour {len(B.lick_on_s)} licks")
    path = os.environ.get(ce.EXEC3D_TABLE_ENV)
    if not path:
        from wfield_local.paths import PathResolver
        path = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics" / "licks.csv.gz"
    K = pd.read_csv(path, usecols=["session", "trial_id", "lick_idx", "side_ml", "out_ap", "down_dv"])
    K = K[K.session == f"{animal}_{date}"].drop_duplicates(["trial_id", "lick_idx"])
    X = L[["trial_id", "lick_idx"]].merge(K, on=["trial_id", "lick_idx"], how="left")[
        ["side_ml", "out_ap", "down_dv"]].to_numpy(float)
    C = pd.read_csv(LICKLAND_CALIB)
    C = C[(C.calib_for == "default") & (C.animal == animal)].set_index("position").reindex(list(LICKSPOUT_POS))
    G = C[["ml", "ap", "down"]].to_numpy(float)
    sc = C[["sd_ml", "sd_ap", "sd_down"]].iloc[0].to_numpy(float)
    d = (((X / sc)[:, None, :] - (G / sc)[None]) ** 2).sum(2)
    with np.errstate(invalid="ignore"):
        e = np.exp(-0.5 * (d - np.nanmin(d, 1, keepdims=True)) / width ** 2)
        return e / e.sum(1, keepdims=True)


def _fr_columns(animal: str, date: str, rv, ft: np.ndarray) -> dict:
    """{FR variable: (n_frames, 1) raw column on every imaging frame} in FR_VARS order (NaN = blanked)."""
    ce = _ce()
    from wfield_local import face_rhythm_outputs as fro
    fr_set = os.environ["COMBINED_ENCODING_FR"]
    sig = fro.fr_signals(animal, date, rv, ft, fdir=fro.factor_dir(rv, set_name=fr_set), absolute=ce._fr_absolute())
    missing = [v for v in FR_VARS if v not in sig]
    if missing:
        raise ValueError(f"FaceRhythm {fr_set}: grouped traces {missing} missing (got {list(sig)})")
    out = {v: np.asarray(sig[v][1], float)[:, None] for v in FR_VARS}
    if FR_LAGS:                                          # FROZEN_FR_LAGS: face leading / cortex leading copies
        from wfield_local import movement_encoding as ME
        fs = ME.frame_fs(ft)
        for v in FR_VARS:
            x = out[v][:, 0]
            for side, lags in (("facelead", [lg for lg in FR_LAGS if lg > 0]), ("cortexlead", [lg for lg in FR_LAGS if lg < 0])):
                out[f"{v}:{side}"] = np.column_stack([ME._shift(x, int(round(lg * fs))) for lg in lags])
    return out


def split_blocks(raw: dict) -> dict:
    """The encoder's raw blocks split into the FGROUP variables. Column order inside each encoder block is fixed by
    `combined_encoding.raw_blocks`: licks = [lick-onset kernel, contact kernel] (equal widths, same lags); tongue =
    x, y, prot, speed; jaw = x, y, speed; nose = x, y; each analog variable len(CLAGS) columns."""
    ce = _ce()
    if os.environ.get("COMBINED_ENCODING_LICKS") == "lips" or os.environ.get("COMBINED_ENCODING_LICKS_ALL"):
        raise ValueError("frozen_encoder: lips / licks-all kernels change the licks block layout; not supported")
    nl = len(ce.CLAGS)
    lk = raw["licks"]
    if lk.shape[1] % 2:
        raise ValueError(f"licks block has {lk.shape[1]} columns; expected [onset, contact] of equal width")
    h = lk.shape[1] // 2
    out = {v: raw[v] for v in ("cue", "arrival", "pos_enl", "pos_cue", "running", "exec_angle")}
    out["lick_onset"], out["contact"] = lk[:, :h], lk[:, h:]
    out["tongue_dir"], out["tongue_mag"] = raw["tongue"][:, :3 * nl], raw["tongue"][:, 3 * nl:]   # x, y, prot | speed
    out["jaw_dir"], out["jaw_mag"] = raw["jaw"][:, :nl], raw["jaw"][:, nl:]
    out["nose_dir"], out["nose_mag"] = raw["nose"][:, :nl], raw["nose"][:, nl:]
    for k, w in (("tongue_mag", 1), ("jaw_mag", 2), ("nose_mag", 1), ("tongue_dir", 3), ("jaw_dir", 1),
                 ("nose_dir", 1)):
        if out[k].shape[1] != w * nl:
            raise ValueError(f"{k}: {out[k].shape[1]} columns, expected {w * nl}")
    return out


#: spout-locked templates, in bins of TEMPLATE_BIN_S (one imaging frame at the cohort's 31.25 Hz). TWO-PART TIME AXIS
#: (10-22 afternoon): CUE-locked (TEMPLATE_S) before the trial's first post-cue lick (and on trials without one),
#: LICK-locked (LICK_TEMPLATE_S, from that first lick) after it. Cue-locked only, the first v4 leakage check left the
#: lick group decoding the spout at 0.38-0.42 (chance 0.17) and exec_within at ~0.53: first-lick latency differs by
#: spout, so a cue-locked mean smears each spout's lick-locked profile and the deviation keeps spout structure.
#: A bin needs >= TEMPLATE_MIN_N pre trials at that spout to get a template value (else s = 0 there: all of x is d).
TEMPLATE_BIN_S = 0.032
TEMPLATE_MIN_N = 20
LICK_TEMPLATE_S = 6.0


def _template_bins() -> tuple[int, int]:
    """(k0, K): absolute bin indices k0 .. k0 + K - 1; the first Kc are cue-locked, the rest lick-locked."""
    k0 = int(np.floor(TEMPLATE_S[0] / TEMPLATE_BIN_S))
    kc = int(np.ceil(TEMPLATE_S[1] / TEMPLATE_BIN_S)) - k0 + 1
    return k0, kc + int(np.ceil(LICK_TEMPLATE_S / TEMPLATE_BIN_S)) + 1


def template_index(t_cue: np.ndarray, t_lick: np.ndarray) -> np.ndarray:
    """Absolute template bin of each frame from its time since the trial's cue and since its first post-cue lick
    (NaN = none / not yet): lick-locked once the lick has happened, cue-locked before; k0 - 1 = outside the template."""
    k0, K = _template_bins()
    kc = int(np.ceil(TEMPLATE_S[1] / TEMPLATE_BIN_S)) - k0 + 1
    after = np.isfinite(t_lick) & (t_lick >= 0)
    kcue = np.round(np.nan_to_num(t_cue, nan=-1e9) / TEMPLATE_BIN_S).astype(np.int64)
    klk = k0 + kc + np.round(np.where(after, t_lick, 0.0) / TEMPLATE_BIN_S).astype(np.int64)
    k = np.where(after, klk, kcue)
    bad = np.where(after, klk >= k0 + K, (kcue < k0) | (kcue >= k0 + kc))
    return np.where(bad, k0 - 1, k)


def template_sums(Xc: np.ndarray, pos_of: np.ndarray, k_of: np.ndarray, pos_order: list) -> tuple:
    """Per spout x cue-relative bin: column sums and frame counts of session-centred raw columns Xc (n, c).
    pos_of: each frame's trial spout ("" = none); k_of: its cue-relative bin index (absolute, from _template_bins)."""
    k0, K = _template_bins()
    S = np.zeros((len(pos_order), K, Xc.shape[1]))
    N = np.zeros((len(pos_order), K))
    kk = k_of - k0
    for pi, p in enumerate(pos_order):
        m = (pos_of == p) & (kk >= 0) & (kk < K)
        np.add.at(S[pi], kk[m], Xc[m])
        np.add.at(N[pi], kk[m], 1.0)
    return S, N


def template_means(S: np.ndarray, N: np.ndarray) -> np.ndarray:
    ok = N >= TEMPLATE_MIN_N
    return np.where(ok[..., None], S / np.maximum(N, 1.0)[..., None], 0.0)


def spout_split(Xc: np.ndarray, pos_of: np.ndarray, k_of: np.ndarray, pos_order: list, M: np.ndarray) -> tuple:
    """(s, d): s = the spout's template at each frame's cue-relative bin (0 outside the template / for unknown spouts),
    d = Xc - s. M: (spouts, K, c) template means in pos_order."""
    k0, K = _template_bins()
    s = np.zeros_like(Xc)
    kk = k_of - k0
    for pi, p in enumerate(pos_order):
        m = (pos_of == p) & (kk >= 0) & (kk < K)
        s[m] = M[pi][kk[m]]
    return s, Xc - s


def _centred(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(np.where(np.isfinite(x), x, np.nan), 0)
    x = x - np.where(np.isfinite(mu), mu, 0.0)[None, :]
    return np.where(np.isfinite(x), x, 0.0)


def session_design(label: str, templates: dict | None = None, template_only: bool = False, spatial: dict | None = None,
                   basis_only: bool = False, relabel: dict | None = None) -> dict:
    """Raw (drift-removed, session-centred) design X, area targets Rd, column variables, trial alignment frames.
    ``templates`` = the animal's spout-locked templates (`build_templates`), required for the STATS_VERSION 4 layout;
    ``template_only`` = return only this session's template sums (behaviour pass, no imaging targets);
    ``spatial`` = {"P": (stat pixels x R) projection, "names"} for basis / grid targets instead of Allen areas (Rd stays
    None above MAX_DENSE_TARGETS: Y and Ra are returned and consumers project on demand); ``basis_only`` = return this
    session's variance-weighted spatial components (the shared-basis pass); ``relabel`` = {spout q: spout j}: q's
    trials get spout j's TARGET columns and j's spout-locked templates (the executed traces stay as recorded: s + d = x),
    for the relabel test (`frozen_relabel`); ``frame_pos`` keeps the TRUE spout of every frame."""
    ce = _ce()
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = config.animal_of(label), "2026" + label.split("_")[-1]
    V, ft, Ub, _flat, parea, names, _shape, pstat = ce.load_imaging(label)
    try:
        B = ce.load_behaviour(animal, date, rv)
    except FileNotFoundError as ex:
        # NO cam1 LICK EVENTS for this session (PS95 all, PS94_1005; cam1 unfinished): cam4 lick timing, as LP_licks
        # and the per-session encoder runs do; recorded as lick_set "cam4". Any other missing file is still an error.
        if "cam1_licks" not in str(ex) or not os.environ.get(ce.LICKS_ENV):
            raise
        saved = os.environ.pop(ce.LICKS_ENV)
        try:
            B = ce.load_behaviour(animal, date, rv)
        finally:
            os.environ[ce.LICKS_ENV] = saved
    T = B.trials
    E = T[T.engaged & np.isfinite(T.cue_s)].reset_index(drop=True)
    st = np.where(np.isfinite(E.strobe_s), E.strobe_s, E.cue_s - 3.0)
    rows = np.zeros(len(ft), bool)
    fpos = np.full(len(ft), "", dtype=object)                          # each frame's trial position (drop-spout)
    ftri = np.full(len(ft), -1)                                        # ... and its trial (later trial wins, as fpos)
    for i, (a, b_, p_) in enumerate(zip(st - 0.5, E.stop_s.to_numpy(float) + 1.0, E.pos.to_numpy())):  # as run_session
        m_ = (ft >= a) & (ft <= b_)
        rows |= m_
        fpos[m_] = p_
        ftri[m_] = i
    ridx = np.flatnonzero(rows)
    ftr = ft[ridx]
    pos_of = fpos[ridx]
    # template time axis: cue-locked until the trial's first post-cue lick (same rule as first_f below), then
    # lick-locked (TEMPLATE_BIN_S comment)
    ex0 = getattr(B, "extra_lick_on_s", None)
    lk0 = np.sort(np.concatenate([np.asarray(B.lick_on_s, float), np.asarray([] if ex0 is None else ex0, float)]))
    cue_t = E.cue_s.to_numpy(float)
    fl_t = np.full(len(E), np.nan)
    for i, (c, stp) in enumerate(zip(cue_t, E.stop_s.to_numpy(float))):
        stp = stp if np.isfinite(stp) else c + 3.5
        j = np.searchsorted(lk0, c)
        if j < len(lk0) and lk0[j] < stp:
            fl_t[i] = lk0[j]
    tri = ftri[ridx]
    k_of = template_index(ftr - cue_t[tri], ftr - fl_t[tri])
    pos_order = sorted(T.pos.unique())
    if basis_only:                                       # spatial-basis pass: this session's weighted spatial components
        Q = np.linalg.qr(ce._slow_basis(ftr, ce.DRIFT_MIN_PERIOD_S))[0]
        Y = V[ridx].astype(np.float64)
        Y -= Q @ (Q.T @ Y)
        w, Ue = np.linalg.eigh(Y.T @ Y)                  # Y'Y = Ue diag(w) Ue'  ->  pixel covariance = (Ub Ue sqrt w)(...)'
        return {"label": label, "animal": animal, "n": len(Y),
                "UbL": (Ub[pstat].astype(np.float64) @ (Ue * np.sqrt(np.clip(w, 0, None)))).astype(np.float32)}
    if relabel:
        lab_r = np.array([relabel.get(p_, p_) for p_ in T.pos.to_numpy()], dtype=object)
        raw = split_blocks(ce.raw_blocks(B, ft, pos_perm={"pos_enl": lab_r, "pos_cue": lab_r}))
        pos_split = np.array([relabel.get(p_, p_) for p_ in pos_of], dtype=object)
    else:
        raw = split_blocks(ce.raw_blocks(B, ft))
        pos_split = pos_of
    if FR_VARS:                                          # FaceRhythm columns (FROZEN_FR); split like pose if FR_SPLIT
        raw.update(_fr_columns(animal, date, rv, ft))
    xc = {v: _centred(raw[v][ridx].astype(np.float64)) for v in SPLIT_VARS}
    if template_only:
        run0 = np.abs(raw["running"][ridx, 0].astype(np.float64))         # lag-0 |speed| on trial frames (RUN_CAP_Q)
        return {"label": label, "animal": animal, "pos_order": pos_order,
                "sums": {v: template_sums(xc[v], pos_of, k_of, pos_order) for v in SPLIT_VARS},
                "run_abs": run0[np.isfinite(run0)].astype(np.float32),
                "push_abs": _push_abs(raw["running"][ridx, 0])}
    if templates is None:
        raise ValueError(f"{label}: the STATS_VERSION {STATS_VERSION} layout needs the animal's spout-locked templates")
    raw["push"] = push_columns(raw["running"], templates.get("push_cap"))     # from the SIGNED speed, before
    raw["running"] = running_basis(raw["running"], templates.get("running_cap"),
                                   _knots(templates.get("running_knots")), _run_mode())
    parts = {}
    Ms = {}
    for v in SPLIT_VARS:
        M = np.stack([templates["mean"][v][templates["pos_order"].index(p)] if p in templates["pos_order"]
                      else np.zeros(templates["mean"][v].shape[1:]) for p in pos_order])
        Ms[v] = M
        parts[f"{v}:spl"], parts[f"{v}:dev"] = spout_split(xc[v], pos_split, k_of, pos_order, M)
    lickland_info = None
    if LICKSPOUT:                                        # one masked copy of the recorded lick-onset columns per spout
        x_ = parts["lick_onset:spl"] + parts["lick_onset:dev"]
        if LICKLAND:                                     # FROZEN_LICKLAND: copies by each lick's 3-D landing
            W = lick_landing_weights(B, animal, date, rv)
            tg = np.array([LICKSPOUT_POS.index(p_) if p_ in LICKSPOUT_POS else -1 for p_ in np.asarray(B.lick_pos)])
            miss = ~np.isfinite(W).all(1)
            W[miss] = 0.0
            W[miss & (tg >= 0), tg[miss & (tg >= 0)]] = 1.0          # no 3-D peak: the target spout (old rule)
            # `movement_encoding.build_design` mean-centres an amplitude modulator, so a weighted kernel = centred-amp kernel
            # + mean weight x the plain kernel (weights sum to 1 per lick -> the six copies sum to the plain kernel)
            on_ = np.asarray(B.lick_on_s, float)
            inr = (on_ >= ft[0]) & (on_ <= ft[-1])
            plain = ce._event_cols(on_, ft, ce.LAGS["lick"])[ridx].astype(np.float64)
            cols = [_centred(ce._event_cols(on_, ft, ce.LAGS["lick"], amp=W[:, k_])[ridx].astype(np.float64)
                             + (W[inr, k_].mean() if inr.any() else 0.0) * plain) for k_ in range(len(LICKSPOUT_POS))]
            resid = x_ - np.sum(cols, 0)                 # cam1-only licks (and unknown-spout licks): target-masked
            for k_, q_ in enumerate(LICKSPOUT_POS):
                parts[f"lick_onset@{q_}"] = cols[k_] + resid * (np.asarray(pos_split) == q_)[:, None]
            lickland_info = {"n_licks": int(len(W)), "share_3d": float(1 - miss.mean()) if len(W) else float("nan"),
                             "share_off_target": float(np.mean(W[~miss].argmax(1) != tg[~miss])) if (~miss).any() else 0.0,
                             "resid_share": float(np.linalg.norm(resid) / max(np.linalg.norm(x_), 1e-12))}
        else:
            for q_ in LICKSPOUT_POS:
                parts[f"lick_onset@{q_}"] = x_ * (np.asarray(pos_split) == q_)[:, None]
    vars_ = [v for v in FGROUP if v != "push" or _push_on()]
    blocks = [parts[v] if v in parts else raw[v][ridx].astype(np.float64) for v in vars_]
    X = np.hstack(blocks)
    # unknown frames (NaN pose) contribute nothing: the session's column mean, which the centring below removes
    # (the encoder's `_z` maps NaN to 0 AFTER z-scoring = the same thing)
    mu = np.nanmean(np.where(np.isfinite(X), X, np.nan), 0)
    X = np.where(np.isfinite(X), X, np.where(np.isfinite(mu), mu, 0.0)[None, :])
    col_var = np.concatenate([[v] * b.shape[1] for v, b in zip(vars_, blocks)])
    Q = np.linalg.qr(ce._slow_basis(ftr, ce.DRIFT_MIN_PERIOD_S))[0]
    X -= Q @ (Q.T @ X)
    if spatial is None:                                  # Allen-area targets (the v1-v4 default)
        areas = sorted({int(a) for a in np.unique(parea[pstat]) if a != 0})
        Ra = np.stack([Ub[(parea == a) & pstat].mean(0) for a in areas] + [Ub[pstat].mean(0)], 1)
        area_names = [names.get(a, str(a)) for a in areas] + ["cortex"]
    else:                                                # SPATIAL targets: projection P (stat pixels x R) + cortex mean
        Ra = np.column_stack([np.asarray((spatial["P"].T @ Ub[pstat].astype(np.float64))).T, Ub[pstat].mean(0)])
        area_names = list(spatial["names"]) + ["cortex"]
    Y = V[ridx].astype(np.float64)
    Y -= Q @ (Q.T @ Y)
    Rd = Y @ Ra if Ra.shape[1] <= MAX_DENSE_TARGETS else None      # large target sets stay lazy (Y @ Ra on demand)
    ex_ = getattr(B, "extra_lick_on_s", None)
    lk = np.sort(np.concatenate([np.asarray(B.lick_on_s, float), np.asarray([] if ex_ is None else ex_, float)]))
    cue_f = np.searchsorted(ftr, E.cue_s.to_numpy(float))
    first_f = np.full(len(E), -1)
    for i, (c, stp) in enumerate(zip(E.cue_s.to_numpy(float), E.stop_s.to_numpy(float))):
        stp = stp if np.isfinite(stp) else c + 3.5
        j = np.searchsorted(lk, c)
        if j < len(lk) and lk[j] < stp:
            first_f[i] = int(np.searchsorted(ftr, lk[j]))
    win = {"all": np.ones(len(ridx), bool), "cue": np.zeros(len(ridx), bool), "lick": np.zeros(len(ridx), bool)}
    for c in E.cue_s.to_numpy(float):
        win["cue"] |= (ftr >= c) & (ftr < c + 1.0)
    for f in first_f[first_f >= 0]:
        t0 = ftr[min(f, len(ftr) - 1)]
        win["lick"] |= (ftr >= t0 - 0.1) & (ftr < t0 + 1.0)
    return {"label": label, "animal": animal, "epoch": epochs.epoch_of(label), "day": epochs.days_since_stroke(label),
            "X": X, "Rd": Rd, "area_names": area_names, "col_var": col_var, "pos": E.pos.to_numpy(),
            "cue_f": cue_f, "first_f": first_f, "fs": float(1.0 / np.median(np.diff(ft))), "win": win,
            "trial_id": E.trial_id.to_numpy(int),
            "lick_set": "combined" if ex_ is not None and len(ex_) else "cam4", "n_trials": len(E),
            "frame_pos": pos_of, "frame_trial": tri, "pos_order": pos_order,
            "pos_counts": {str(p_): int((T.pos == p_).sum()) for p_ in pos_order},
            "Y": Y if Rd is None else None, "Ra": Ra if Rd is None else None,
            "spl_swap": spl_swap_deltas(Ms, templates, pos_of, k_of, pos_order, col_var, ftr) if spatial else None,
            "lickland": lickland_info}


def swap_template(M: np.ndarray, N: np.ndarray, qi: int) -> np.ndarray:
    """M with spout qi's template replaced by the OTHER spouts' pooled template (frame-count weighted per bin, so
    the bins where only some spouts have data use those): 'spout q's locked movement as the typical other spout'."""
    w = np.asarray(N, float).copy()
    w[qi] = 0.0
    den = w.sum(0)
    pooled = np.einsum("pk,pkc->kc", w, M) / np.where(den > 0, den, 1.0)[:, None]
    out = M.copy()
    out[qi] = np.where((den > 0)[:, None], pooled, 0.0)
    return out


def spl_swap_deltas(Ms: dict, templates: dict, pos_of, k_of, pos_order: list, col_var, ftr) -> dict:
    """{spout q: (frame index of q's frames, delta (n_q, p))}: the change of the design X on q's frames when q's
    spout-locked templates are swapped for the other spouts' pooled ones (`swap_template`), deviation columns
    unchanged, after the same slow-drift projection as X. With the frozen drop-one-spout fit (target q folded into
    the others) this removes the WHOLE spout-specific response of q (Priya 10-23: target and the spout-locked lick are
    the same variable on on-target licks)."""
    if "count" not in templates:
        return {}
    ce = _ce()
    Q = np.linalg.qr(ce._slow_basis(ftr, ce.DRIFT_MIN_PERIOD_S))[0]
    tpos = templates["pos_order"]
    Nall = np.asarray(templates["count"], float)
    N = np.stack([Nall[tpos.index(p)] if p in tpos else np.zeros(Nall.shape[1:]) for p in pos_order])
    out = {}
    for qi, q in enumerate(pos_order):
        rows = np.flatnonzero(pos_of == q)
        if len(rows) < 10:
            continue
        D = np.zeros((len(rows), len(col_var)))
        for v in SPLIT_VARS:
            Msw = swap_template(Ms[v], N, qi)
            s0, _ = spout_split(np.zeros((len(rows), Ms[v].shape[2])), pos_of[rows], k_of[rows], pos_order, Ms[v])
            s1, _ = spout_split(np.zeros((len(rows), Ms[v].shape[2])), pos_of[rows], k_of[rows], pos_order, Msw)
            if (col_var == f"{v}:spl").any():           # absent for lick_onset under FROZEN_LICKSPOUT
                D[:, col_var == f"{v}:spl"] = s1 - s0
        D -= Q[rows] @ (Q[rows].T @ D)                    # the slow-drift projection, restricted to q's rows
        out[q] = (rows, D)
    return out


def template_job(label: str) -> dict:
    """One pre session's spout-locked template sums (behaviour only)."""
    return session_design(label, template_only=True)


def build_templates(parts: list[dict], run_thresh: float | None = None) -> dict:
    """{animal: {"pos_order", "mean": {var: (spouts, K, c)}, "n_sessions", "labels"}} pooled over the animal's pre
    sessions (sums / counts added, so every pre trial weighs the same)."""
    by: dict = {}
    for t in sorted(parts, key=lambda t: t["label"]):
        by.setdefault(t["animal"], []).append(t)
    out = {}
    for an, ts in by.items():
        order = sorted({p for t in ts for p in t["pos_order"]})
        mean = {}
        for v in SPLIT_VARS:
            S = N = None
            for t in ts:
                s_, n_ = t["sums"][v]
                ix = [order.index(p) for p in t["pos_order"]]
                if S is None:
                    S = np.zeros((len(order),) + s_.shape[1:])
                    N = np.zeros((len(order),) + n_.shape[1:])
                S[ix] += s_
                N[ix] += n_
            mean[v] = template_means(S, N)
            count = N                                       # same frame counts for every split variable
        ra = [t["run_abs"] for t in ts if t.get("run_abs") is not None and len(t["run_abs"])]
        ra = np.concatenate(ra) if ra else np.zeros(0)
        if len(ra):
            ra = ra[ra > (run_thresh if run_thresh is not None else _run_thresh())]   # running frames, pooled
        cap = float(np.percentile(ra, RUN_CAP_Q)) if len(ra) else None
        pa = [t["push_abs"] for t in ts if t.get("push_abs") is not None and len(t["push_abs"])]
        push_cap = float(np.percentile(np.concatenate(pa), RUN_CAP_Q)) if pa else None
        rt = run_thresh if run_thresh is not None else (_run_thresh() if len(ra) else None)
        knots = [float(rt)] + [float(np.percentile(ra, q)) for q in RUN_KNOT_Q] if len(ra) else []
        out[an] = {"pos_order": order, "mean": mean, "n_sessions": len(ts), "labels": [t["label"] for t in ts],
                   "bin_s": TEMPLATE_BIN_S, "span_s": TEMPLATE_S, "min_n": TEMPLATE_MIN_N,
                   "running_cap": cap, "running_cap_q": RUN_CAP_Q, "running_knots": knots,
                   "count": count, "push_cap": push_cap}
    return out


def cap_running(x: np.ndarray, cap: float | None) -> np.ndarray:
    """Running columns clipped to +/- cap (sign kept; NaN stays NaN); cap None = unchanged (RUN_CAP_Q comment)."""
    if cap is None:
        return x
    return np.clip(x, -cap, cap)


#: SUBLINEAR RUNNING TEST (Priya 10-23: "yes test sublinear running"). With the linear kernel PS94's post-stroke fit
#: was better under a LOWER cap (9.4 mm/s: chronic R2 0.53) than a higher one (13.2: 0.30), as if cortex saturated
#: with speed. `FROZEN_RUNNING=spline` replaces the linear running term with a PIECEWISE-LINEAR curve in speed: the
#: capped speed s plus hinges max(0, s - k) at knots k = the running threshold and the RUN_KNOT_Q percentiles of the
#: animal's pre RUNNING frames (all below the cap). Every function is flat above the cap, so the frozen model still
#: never extrapolates; the fitted curve's shape (`running_curve`) is the readout. Default "linear" = v6.
RUN_KNOT_Q = (25.0, 50.0, 75.0)
#: RESULT of "spline" (10-23, `_v6spl`): the fitted curve jumps between 0 and ~3-5 mm/s and is flat / slightly falling
#: above in all four animals; held-out pre R2 +0.03 .. +0.10 and PS94 post fixed (chronic 0.30 -> 0.60), but the cue /
#: lick windows of later post-stroke epochs got WORSE in PS92 / PS93 / PS95 (by ~0.1-0.15): the steep part sits BELOW
#: the pipeline's 3 mm/s running threshold (fidgets, treadmill creep). "spline_thr" (Priya: "launch the threshold
#: variant") = no running response below the threshold: hinges max(0, s - k) at k = the threshold and the RUN_KNOT_Q
#: percentiles of the pre running frames only (no linear s term), still flat above the cap.
#: v7 DEFAULT (10-23, six models side by side, `scripts/frozen_running_compare.py` -> qc/frozen_running_compare_v6_to_
#: v7fwd15push.csv): spline_thr from 1.5 mm/s. Forward-only beats signed linear in every animal x epoch x window (PS95
#: chronic cue-window R2 .00 -> .15, PS94 chronic all frames .30 -> .62); onset 1 vs 1.5 mm/s within .02 everywhere and
#: the 1-1.5 band takes no variance from the lick group (Priya chose 1.5: below it the treadmill moves with cue / lick
#: jolts); the acute far-contra decomposition is unchanged in all six models. Reproduce older runs with
#: FROZEN_RUNNING=linear (v6), =spline / =spline_thr with FROZEN_RUN_KNOT0=3 (`_v6spl`, `_v6thr3`), =1 (`_v6thr1`).
RUN_MODE_DEFAULT = "spline_thr"
RUN_KNOT0_DEFAULT = 1.5


def _run_mode() -> str:
    m = os.environ.get("FROZEN_RUNNING", RUN_MODE_DEFAULT)
    if m not in ("linear", "spline", "spline_thr"):
        raise ValueError(f"FROZEN_RUNNING={m!r}: expected linear | spline | spline_thr")
    return m


def _knots(knots: list | None) -> list | None:
    """The template knots with the first (= the response onset for spline_thr) overridden by FROZEN_RUN_KNOT0 (mm/s),
    if set: Priya 10-23 "is 3 mm too high a threshold for this" -- 3 mm/s is the behaviour segmentation's SUSTAINED-
    running threshold, the rest floor is ~0.1-0.5 mm/s (pre trial-frame p90), and the spline's response was mostly
    reached by 2 mm/s. Higher knots stay the pre running-frame percentiles. Default RUN_KNOT0_DEFAULT (v7)."""
    k0 = _knot0()
    if not knots:
        return knots
    return sorted({k0, *[k for k in knots[1:] if k > k0]})


def _knot0() -> float:
    return float(os.environ.get("FROZEN_RUN_KNOT0") or RUN_KNOT0_DEFAULT)


def _run_tag() -> str:
    """Running-model tag stored with the saved statistics (never reused across modes / knot overrides / push)."""
    return (_run_mode() + f"|k0={_knot0():g}" + ("|push" if _push_on() else "")
            + ("|tongue3d" if os.environ.get("COMBINED_ENCODING_TONGUE", "") == "3d" else "")    # TONGUE_ENV
            + (f"|fr={os.environ['COMBINED_ENCODING_FR']}:{os.environ['COMBINED_ENCODING_FR_GROUPS']}" if FR_VARS else "")
            + ("|frsplit" if FR_SPLIT else "") + ("|lickspout" if LICKSPOUT else "") + ("|lickland" if LICKLAND else "")
            + (f"|frlags={','.join(f'{x:g}' for x in FR_LAGS)}" if FR_LAGS else ""))


#: BACKWARD / POSTURAL TREADMILL MOVEMENT (Priya 10-23, after the side-camera clips: the negative speed at bout ends is
#: the animal pushing a paw forward, and the post-cue dip is upper-limb extension / a posture change -- real movements,
#: "should we really throw these out or might they be helpful?"). The threshold running models already ignore
#: backward speed (forward hinges only) and the signed linear v6 treated it as NEGATIVE running, although cortex is
#: HIGHER during pushes. `FROZEN_PUSH=1` adds its own variable "push" (group other, penalty uninstructed): backward
#: speed beyond PUSH_FLOOR, capped at the animal's pooled pre-stroke RUN_CAP_Q percentile of backward speed.
PUSH_FLOOR = 0.3


def _push_on() -> bool:
    return bool(os.environ.get("FROZEN_PUSH"))


#: FREE PUSH (Priya 10-23: "if push changes, wouldn't it be useful to include it somehow then?"). A FROZEN push kernel
#: fails post-stroke (v7 test: chronic cue-window R2 PS95 .15 -> -.80): pushes are rare pre-stroke (0.6-3.9 % of trial
#: frames) and 2-5x more frequent after, so the pre kernel rests on few events that coincide with bout ends / cue
#: posture / licks. Leaving push out leaves its activity unexplained (residual, "data - model"). `FROZEN_PUSH=free`:
#: push stays in the pre fit (the frozen groups are estimated with push accounted for), but on EVERY session (pre and
#: post) the push columns' betas are re-fitted to that session's residual (data - frozen non-push prediction), ridge
#: with the frozen push-group alpha, cross-validated over PUSH_FREE_FOLDS contiguous blocks (no in-sample gain). Areas
#: only; the frozen parts are then read with push accounted for per session.
PUSH_FREE_FOLDS = 5


def _push_free() -> bool:
    return os.environ.get("FROZEN_PUSH", "").strip().lower() == "free"


def free_push(Zv: np.ndarray, R: np.ndarray, beta: np.ndarray, qm: np.ndarray, alpha: float,
              folds: int = PUSH_FREE_FOLDS) -> tuple[np.ndarray, np.ndarray]:
    """(frozen non-push prediction, cross-validated per-session push prediction) on the variant design Zv."""
    base = Zv[:, ~qm] @ beta[~qm]
    out = np.zeros_like(base)
    if not qm.any():
        return base, out
    r, Zp = R - base, Zv[:, qm]
    edges = np.linspace(0, len(Zp), folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.ones(len(Zp), bool)
        tr[a:b] = False
        coef = np.linalg.solve(Zp[tr].T @ Zp[tr] + alpha * np.eye(qm.sum()), Zp[tr].T @ r[tr])
        out[a:b] = Zp[a:b] @ coef
    return base, out


def _push_abs(x0: np.ndarray) -> np.ndarray:
    """Backward speed magnitudes (lag-0 column, beyond PUSH_FLOOR) for the pooled cap."""
    b = -np.asarray(x0, np.float64)
    b = b[np.isfinite(b) & (b > PUSH_FLOOR)]
    return b.astype(np.float32)


def push_columns(x: np.ndarray, cap: float | None) -> np.ndarray:
    """Signed lagged running columns -> backward-movement columns: max(0, -x - PUSH_FLOOR), capped (NaN kept)."""
    b = np.maximum(-np.asarray(x, np.float64) - PUSH_FLOOR, 0.0)
    b = np.where(np.isfinite(x), b, np.nan)
    return b if cap is None else np.minimum(b, max(cap - PUSH_FLOOR, 0.0))


def _run_blocks(s: np.ndarray, knots: list, mode: str) -> list:
    """The running basis blocks of capped speed s: spline = [s, hinge k...], spline_thr = [hinge k...] (k[0] = the
    running threshold, so nothing below it)."""
    hinges = [np.maximum(s - k, 0.0) for k in knots]
    return hinges if mode == "spline_thr" else [s, *hinges]


def running_basis(x: np.ndarray, cap: float | None, knots: list | None, mode: str) -> np.ndarray:
    """(T, L) lagged running columns -> blocks of L lag columns each (the transform commutes with the lag shift):
    "spline" [s | hinge k1 | hinge k2 ...], "spline_thr" [hinge k1 | ...]; "linear" = capped columns unchanged."""
    s = cap_running(x, cap)
    if mode == "linear" or not knots:
        return s
    return np.hstack(_run_blocks(s, knots, mode))


def running_curve(beta: np.ndarray, knots: list, speeds: np.ndarray, cap: float | None,
                  mode: str = "spline") -> np.ndarray:
    """Fitted response vs speed from the running betas IN RAW UNITS (blocks as `running_basis`), summed over lags."""
    s = cap_running(np.asarray(speeds, float), cap)
    blocks = _run_blocks(s, knots, mode)
    b = np.asarray(beta, float).reshape(len(blocks), -1).sum(1)
    return sum(bi * blk for bi, blk in zip(b, blocks))


#: SPATIAL TARGETS (Priya 10-22: "is predicting allen areas really what we want? not a downsampled map?" -> the
#: literature route, Musall et al. 2019, fits in SVD space and reconstructs pixel maps; every session has its OWN SVD
#: basis, so a frozen model needs a basis common to the animal's sessions; Priya: "ok proceed"):
#:   basis  per animal, the top components of its pooled PRE-session data on the shared Allen-aligned stat mask (every
#:          session's U_atlas lives on the same 540 x 640 grid, 129,770 stat pixels): each pre session contributes
#:          Ub Ue sqrt(w) (its spatial components weighted by its trial-frame variance, drift removed), the stack's
#:          leading left singular vectors up to BASIS_VAR of the variance (<= BASIS_KMAX) are the targets; a session's
#:          targets are its data projected on them; maps = basis x (anything in target space)
#:   grid   GRID_BLOCK x GRID_BLOCK pixel block means on the stat mask (blocks with >= GRID_MIN_PIX pixels): the
#:          transparent check on the basis
#: Both add the cortex mean as a last target (scores / Musall readouts are reported on it).
BASIS_VAR = 0.9999                 # in practice K = BASIS_KMAX: 95 % needed only ~6 components (global patterns);
#                                     the spatial detail lives in the low-variance tail (4 PS92 sessions: 99.9 % = 91, 200 = 99.975 %)
BASIS_KMAX = 200
GRID_BLOCK = 8
GRID_MIN_PIX = 16


def basis_job(label: str) -> dict:
    return session_design(label, basis_only=True)


def build_basis(parts: list[dict]) -> dict:
    """{animal: spatial dict} from the pre sessions' weighted components (basis_job)."""
    by: dict = {}
    for t in sorted(parts, key=lambda t: t["label"]):
        by.setdefault(t["animal"], []).append(t)
    out = {}
    for an, ts in by.items():
        B = np.hstack([t["UbL"] for t in ts])                       # (stat pixels, 100 x sessions), float32
        G = (B.T.astype(np.float64) @ B.astype(np.float64))
        w, Qe = np.linalg.eigh(G)
        o = np.argsort(w)[::-1]
        w, Qe = np.clip(w[o], 0, None), Qe[:, o]
        cum = np.cumsum(w) / w.sum()
        K = int(min(BASIS_KMAX, np.searchsorted(cum, BASIS_VAR) + 1))
        W = ((B.astype(np.float64) @ Qe[:, :K]) / np.sqrt(w[:K])).astype(np.float32)   # orthonormal columns
        out[an] = {"kind": "basis", "P": W, "names": [f"b{j:03d}" for j in range(K)], "evr": cum[:K],
                   "labels": [t["label"] for t in ts]}
    return out


def grid_spatial(label: str) -> dict:
    """Block-mean projection on the stat mask (same for every session: the Allen-aligned grid is shared)."""
    from scipy import sparse
    ce = _ce()
    _V, _ft, _Ub, flat, _parea, _names, shape, pstat = ce.load_imaging(label)
    H, Wd = (int(x) for x in shape)
    pix = flat[pstat]
    r, c = pix // Wd, pix % Wd
    nbc = -(-Wd // GRID_BLOCK)
    blk = (r // GRID_BLOCK) * nbc + c // GRID_BLOCK
    ub, inv, cnt = np.unique(blk, return_inverse=True, return_counts=True)
    keep = cnt >= GRID_MIN_PIX
    col = np.cumsum(keep) - 1
    m = keep[inv]
    P = sparse.csr_matrix((1.0 / cnt[inv][m], (np.flatnonzero(m), col[inv][m])), shape=(len(pix), int(keep.sum())))
    return {"kind": "grid", "P": P, "names": [f"g{b}" for b in ub[keep]], "block": GRID_BLOCK, "blocks": ub[keep],
            "shape": (H, Wd)}


#: targets above this count are never materialised as (frames x targets) arrays (a 2,022-bin grid would be ~2.7 GB per
#: session in float64): statistics and PETHs are projected from Y (frames x SVD components) instead
MAX_DENSE_TARGETS = 300


def _spatial_of(spath: str | None, animal: str) -> dict | None:
    if not spath:
        return None
    import pickle
    return pickle.loads((Path(spath) / f"spatial_{animal}.pkl").read_bytes())


def _templates_of(tpath: str, animal: str) -> dict:
    import pickle
    return pickle.loads(Path(tpath).read_bytes())[animal]


def stats_job(item):
    """Sufficient statistics of one session (column layout + per-area Gram terms), for the per-animal fit.
    item = (label, path of fit/templates.pkl)."""
    from wfield_local import config
    label, tpath, spath = (*item, None)[:3]
    an = config.animal_of(label)
    d = session_design(label, _templates_of(tpath, an), spatial=_spatial_of(spath, an))
    X, Rd = d["X"], d["Rd"]
    if Rd is None:                                       # lazy targets: X'Y Ra and diag(Ra' Y'Y Ra)
        XtY = (X.T @ d["Y"]) @ d["Ra"]
        YtY = np.einsum("kr,kl,lr->r", d["Ra"], d["Y"].T @ d["Y"], d["Ra"])
    else:
        XtY, YtY = X.T @ Rd, (Rd ** 2).sum(0)
    return {"label": label, "animal": d["animal"], "epoch": d["epoch"], "col_var": d["col_var"],
            "area_names": d["area_names"], "n": len(X), "XtX": X.T @ X, "XtY": XtY, "YtY": YtY,
            "sxx": (X ** 2).sum(0), "pos_order": d["pos_order"], "pos_counts": d["pos_counts"]}


def _subset(s: dict, keep: list[str]) -> tuple:
    ix = [s["area_names"].index(a) for a in keep]
    return s["XtY"][:, ix], s["YtY"][ix]


#: FROZEN VARIANTS (STATS_VERSION 4 groups, see FGROUP). All from the same pre-stroke Gram terms; each is a column
#: transform C of the scaled design (Z' = Z @ C).
#:   joint            everything jointly (the ridge splits shared variance) -- main
#:   no_<g>           joint refit WITHOUT group / super-group g: unique(g) = joint - no_<g> (order-free; shared variance
#:                    credited to nobody). g in timing, target, exec_locked, exec_within, lick, other, contact, and the
#:                    super-groups spout (target + exec_locked + exec_within), task (timing + target), movement (lick +
#:                    exec_* + other: no_movement = the TASK-ONLY model)
#:   <g>_only         that group / super-group alone: task_only (timing + target), lick_only, spout_only, other_only,
#:                    movement_only; no_task = all non-task alone (no_movement keeps contact: not "task alone")
#:   target_first     exec_locked + exec_within residualised on target (pre projection): spout-shared -> target
#:   exec_first       target residualised on exec_locked + exec_within: spout-shared -> executed. The pair BRACKETS
#:                    target vs executed (confounded by design; only exec_within separates them)
#:   task_first       all non-task residualised on task (timing + target): shared -> task
#:   movement_first   task residualised on all non-task: shared -> non-task. The pair brackets Musall's task vs movement
#: Musall: alone = <g>_only / no_task (non-task); unique / task-independent = joint - no_<g> (non-task: joint - task_only);
#: task-aligned = alone - task-independent.
_GROUPSETS = {**{g: (g,) for g in FGROUPS}, **SUPER}
_DROP = {f"no_{g}": gs for g, gs in _GROUPSETS.items() if g != "nontask"}
#: SHARED-VARIANCE CHECK (Priya 10-23, PS93's unique contact ~0 although it misses most): drop contact AND within-spout
#: direction together; unique(contact + exec_within) >> unique(contact) + unique(exec_within) = variance they share
_DROP["no_contact_exec_within"] = ("contact", "exec_within")
_ONLY = {"task_only": SUPER["task"], "lick_only": ("lick",), "spout_only": SUPER["spout"], "other_only": ("other",),
         "movement_only": SUPER["movement"]}
_ONLY.update({f"{g}_only": SUPER[g] for g in ("posinfo", "noposinfo") if g in SUPER})   # FR runs (FR_VARS block)
if FR_VARS:
    _ONLY["fr_only"] = tuple(g for g in FGROUPS if g.startswith("fr"))       # FaceRhythm alone (Musall top row)
#: PER-VARIABLE DROPS (Priya 10-25, Musall et al. 2019 bottom row = single-variable unique dR2): novar_<base> drops every
#: column of ONE base variable (its s and d parts / per-spout copies together), everything else joint.
_VAR_BASES = sorted({v.split(":")[0].split("@")[0] for v in FGROUP})
_VDROP = {f"novar_{b}": b for b in _VAR_BASES}
_ORDERED = {"target_first": (("target",), _EXEC_SIDE),     # (A, B): residualise B on A
            "exec_first": (_EXEC_SIDE, ("target",)),
            "task_first": (SUPER["task"], SUPER["nontask"]),
            "movement_first": (SUPER["nontask"], SUPER["task"])}
VARIANTS = ("joint", *_ORDERED, *_DROP, *_ONLY, *_VDROP, *_LAGDROP)
#: variants whose per-group PETH parts are saved (the others save data + full only: their parts are not a split)
PART_VARIANTS = ("joint", *_ORDERED)


#: FROZEN DROP-ONE-SPOUT (Priya 10-22: "should we do a leave-one-spout out but without removing movement first" -> yes):
#: `<base>|drop:<p>` = base variant with spout p's TARGET position deviation constrained to the trial-weighted mean of
#: the others, exactly `combined_encoding.raw_blocks(drop_pos=p)`: with deviation blocks D_q = X_q - m_q * shared, the
#: drop-p columns are D_q + m_q / (1 - m_p) * D_p (q != p) and D_p is removed -- a column recombination, so it is one
#: more transform of the same Gram terms (m = the animal's POOLED pre trial shares, frozen like everything else). Scored
#: on every session as the loss of variance vs the base (window offset free, all frames and spout p's own trials).
#: target_first = target BEFORE the executed spout-locked part (upper bound on spout-specific target coding),
#: exec_first = after it (lower bound), joint = ridge split. The executed spout-locked columns are NOT dropped (they
#: are the animal's behaviour at that spout, not its response to the spout).
DROP_BASES = ("joint", "target_first", "exec_first")


def _drop_transform(col_var: np.ndarray, pos_order: list, frac: dict, drop: str) -> tuple[np.ndarray, np.ndarray]:
    ce = _ce()
    p = len(col_var)
    C = np.eye(p)
    gone = []
    k = pos_order.index(drop)
    mp = frac[drop]
    for var in ce.POS_VARS:
        idx = np.flatnonzero(col_var == var)
        if len(idx) % len(pos_order):
            raise ValueError(f"{var}: {len(idx)} columns not divisible by {len(pos_order)} positions")
        ch = idx.reshape(len(pos_order), -1)                       # raw_blocks order: one chunk per position
        for j, q in enumerate(pos_order):
            if j != k:
                C[ch[k], ch[j]] += frac[q] / max(1.0 - mp, 1e-9)
        gone.extend(ch[k].tolist())
    keep = np.setdiff1d(np.arange(p), gone)
    return C[:, keep], keep


#: ordered projections: eigenvalues of the A-block Gram below this fraction of the largest are treated as zero (a
#: pseudo-inverse). The v3 code solved with a ~1e-6 ridge; the position deviation blocks are near-collinear by
#: construction (their trial-weighted sum is ~0), so the projection had HUGE weights along that null direction which
#: cancel in pre sessions but not when a post session's spout shares differ -> task_first scored -1.06 on PS95_0818
#: while joint scored 0.36 (10-22 afternoon). 1e-8 was still too permissive: PS95's target block keeps near-null
#: directions at ~1e-7 / 1e-5 of the largest eigenvalue (every real direction is > 1e-2), so target_first scored
#: -163 on PS95_0818 in the first v4 run; 1e-3 drops them (they carry no pre-stroke variance).
PROJ_RCOND = 1e-3


def _projection(Gaa: np.ndarray, Gab: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(Gaa)
    keep = w > PROJ_RCOND * max(w.max(), 1e-300)
    return (U[:, keep] / w[keep]) @ (U[:, keep].T @ Gab)


def _variant_transform(variant: str, col_var: np.ndarray, Gt: np.ndarray, pos: dict | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """((p, p') column transform of the scaled design, (p',) source column of each variant column) for ``variant``
    (Gt = pooled scaled pre Gram, for the ordered projections; ``pos`` = {"order", "frac"} for drop variants)."""
    base, _, drop = variant.partition("|drop:")
    if drop:
        C1, src1 = _drop_transform(col_var, pos["order"], pos["frac"], drop)
        C2, src2 = _variant_transform(base, col_var[src1], C1.T @ Gt @ C1)
        return C1 @ C2, src1[src2]
    p = len(col_var)
    eye = np.eye(p)
    if variant == "joint":
        return eye, np.arange(p)
    grp = np.array([FGROUP[v] for v in col_var])
    if variant in _DROP or variant in _ONLY:
        keep = ~np.isin(grp, _DROP[variant]) if variant in _DROP else np.isin(grp, _ONLY[variant])
        return eye[:, keep], np.flatnonzero(keep)
    if variant in _LAGDROP:                              # FROZEN_FR_LAGS: drop one lag side of every FR variable
        keep = ~np.char.endswith(np.asarray(col_var).astype(str), _LAGDROP[variant])
        return eye[:, keep], np.flatnonzero(keep)
    if variant in _VDROP:
        base_ = np.array([v.split(":")[0].split("@")[0] for v in col_var])
        keep = base_ != _VDROP[variant]
        return eye[:, keep], np.flatnonzero(keep)
    if variant in _ORDERED:
        A, Bg = _ORDERED[variant]
        a, b = np.isin(grp, A), np.isin(grp, Bg)
        if not a.any() or not b.any():                   # nothing to residualise: = joint
            return eye, np.arange(p)
        C = eye.copy()
        C[np.ix_(a, b)] = -_projection(Gt[np.ix_(a, a)], Gt[np.ix_(a, b)])     # residualise b on a
        return C, np.arange(p)
    raise ValueError(variant)


def fit_animal(stats: list[dict], alpha_grid=None, pen_of=None, variant: str = "joint") -> dict:
    """Ridge on the animal's pre sessions (scaled by pooled pre SD), penalties by leave-one-session-out R^2.
    Returns scale, C (variant column transform), col_var of the variant columns, areas, alphas, beta_all (p', R) and
    beta_loso {label: (p', R)}."""
    ce = _ce()
    alpha_grid = alpha_grid or ce.ALPHA_GRID
    pen_of = pen_of or FPEN
    col_var = stats[0]["col_var"]
    for s in stats:
        if len(s["col_var"]) != len(col_var) or (s["col_var"] != col_var).any():
            raise ValueError(f"{s['label']}: column layout differs from {stats[0]['label']}")
    areas = [a for a in stats[0]["area_names"] if all(a in s["area_names"] for s in stats)]
    n = sum(s["n"] for s in stats)
    sd = np.sqrt(sum(s["sxx"] for s in stats) / n)
    ok0 = sd > 1e-9
    scale = np.where(ok0, sd, np.inf)
    G0 = [s["XtX"] / np.outer(scale, scale) for s in stats]
    b0 = []
    yy = []
    for s in stats:
        xy, y2 = _subset(s, areas)
        b0.append(xy / scale[:, None])
        yy.append(y2)
    order = list(stats[0]["pos_order"])
    cnt = {q: sum(s["pos_counts"].get(q, 0) for s in stats) for q in order}
    frac = {q: cnt[q] / max(sum(cnt.values()), 1) for q in order}
    C, src = _variant_transform(variant, col_var, sum(G0), {"order": order, "frac": frac})
    vcol = col_var[src]
    ok = ok0[src]
    G = [C.T @ g @ C for g in G0]
    b = [C.T @ x for x in b0]
    Gt, bt = sum(G), sum(b)
    pen_col = np.array([pen_of[v] for v in vcol])
    groups = sorted(set(pen_col))

    def solve(Gm, bm, al):
        lam = np.array([al[g] for g in pen_col]) * ok + (~ok) * 1.0
        return np.linalg.solve(Gm + np.diag(lam) + 1e-8 * np.eye(len(lam)), bm)

    def loso_r2(al):
        err = tot = 0.0
        for k in range(len(stats)):
            beta = solve(Gt - G[k], bt - b[k], al)
            err += (yy[k] - 2 * (beta * b[k]).sum(0) + np.einsum("pr,pq,qr->r", beta, G[k], beta)).sum()
            tot += yy[k].sum()
        return 1.0 - err / tot
    alphas = {g: 100.0 for g in groups}
    if len(stats) >= 2:
        for _ in range(2):
            for g in groups:
                alphas[g] = max(alpha_grid, key=lambda a: loso_r2({**alphas, g: a}))
    beta_all = solve(Gt, bt, alphas)
    beta_loso = {s["label"]: solve(Gt - G[k], bt - b[k], alphas) for k, s in enumerate(stats)} if len(stats) >= 2 else {}
    return {"variant": variant, "scale": scale, "C": C, "vcol": vcol, "areas": areas, "alphas": alphas,
            "beta_all": beta_all, "beta_loso": beta_loso, "col_var": col_var, "src": src,
            "loso_r2": loso_r2(alphas) if len(stats) >= 2 else np.nan, "pre_labels": [s["label"] for s in stats],
            "pos_order": order, "pos_frac": frac}


def _peth(R: np.ndarray, frames: np.ndarray, off: np.ndarray) -> np.ndarray:
    idx = frames[:, None] + off[None, :]
    ok = (idx >= 0) & (idx < len(R))
    seg = np.where(ok[..., None], R[np.clip(idx, 0, len(R) - 1)], np.nan)
    return np.nanmean(seg, 0).T                                   # (areas, off)


#: MUSALL-STYLE READOUTS (Priya 10-22: "let's add trial to trial and task averaged r2 readouts a la musall"), per
#: window aligned on the cue (0 .. 1.5 s) or the first post-cue lick (-0.1 .. +1.0 s), per area, held out like every
#: frozen score: FRAME = every frame of every trial (condition-locked + single-trial variance; window offset free);
#: TRIAL = each spout's mean segment removed from data AND prediction (single-trial variance only, Musall's
#: "single-trial"); AVG = the per-spout mean segments only, offset free per area (Musall's "trial-averaged").
MUSALL_WINDOWS = {"cue": (0.0, 1.5), "lick": (-0.1, 1.0)}


def musall_r2(D: np.ndarray, P: np.ndarray, pos: np.ndarray) -> dict:
    """D, P: (trials, t, areas) data / prediction segments (NaN = outside the record); pos: (trials,) spout labels.
    -> {"frame", "trial", "avg": (areas,) R^2}."""
    ok = np.isfinite(D) & np.isfinite(P)
    D, P = np.where(ok, D, np.nan), np.where(ok, P, np.nan)

    def r2(y, yh):
        return 1.0 - np.nansum((y - yh) ** 2, axis=(0, 1)) / np.maximum(np.nansum(y ** 2, axis=(0, 1)), 1e-12)
    Dc, Pc = D - np.nanmean(D, axis=(0, 1)), P - np.nanmean(P, axis=(0, 1))
    out = {"frame": r2(Dc, Pc)}
    Dd, Pd = D.copy(), P.copy()
    Dm, Pm = [], []
    for q in np.unique(pos):
        m = pos == q
        dm, pm = np.nanmean(D[m], 0, keepdims=True), np.nanmean(P[m], 0, keepdims=True)
        Dd[m] -= dm
        Pd[m] -= pm
        Dm.append(dm[0])
        Pm.append(pm[0])
    out["trial"] = r2(Dd, Pd)
    Dm, Pm = np.stack(Dm), np.stack(Pm)                         # (spouts, t, areas)
    out["avg"] = r2(Dm - np.nanmean(Dm, axis=(0, 1)), Pm - np.nanmean(Pm, axis=(0, 1)))
    return out


def _segments(R: np.ndarray, frames: np.ndarray, off: np.ndarray) -> np.ndarray:
    idx = frames[:, None] + off[None, :]
    ok = (idx >= 0) & (idx < len(R))
    return np.where(ok[..., None], R[np.clip(idx, 0, len(R) - 1)], np.nan)


def trial_means(R: np.ndarray, frames: np.ndarray, off: np.ndarray) -> np.ndarray:
    """(trials, areas) mean of R over each trial's window ``frames + off``; NaN for a trial without an anchor (frame < 0)
    or a window entirely outside the record."""
    out = np.full((len(frames), R.shape[1]), np.nan)
    ok = frames >= 0
    if ok.any():
        seg = _segments(R, frames[ok], off)
        with np.errstate(invalid="ignore"):
            out[ok] = np.nanmean(seg, 1)
    return out


def load_trial_parts(run_dir, variant: str = "joint", areas=None):
    """Long table of the saved per-trial parts of one frozen run: label, trial_id, pos, window, area, part, value.
    Join on (label, trial_id) with `trial_attempts.csv` for any trial-level split. Sessions predicted before the
    per-trial output existed (no ``trial_parts`` in the npz) are skipped with a note."""
    import pandas as pd
    rows, old = [], []
    for f in sorted((Path(run_dir) / variant / "sessions").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        if "trial_parts" not in z.files:
            old.append(f.stem)
            continue
        T, ar = z["trial_parts"], list(z["areas"])
        keep = [i for i, a in enumerate(ar) if areas is None or a in areas]
        for ci, part in enumerate(z["parts"]):
            for wi, w in enumerate(z["trial_windows"]):
                for ai in keep:
                    rows.append(pd.DataFrame({"label": f.stem, "trial_id": z["trial_id"], "pos": z["trial_pos"],
                                              "window": w, "area": ar[ai], "part": part, "value": T[ci, :, wi, ai]}))
    if old:
        print(f"[frozen] {len(old)} session(s) have no per-trial parts (predicted before 10-24): {old[:5]} ...", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def predict_job(job):
    """Apply the animal's frozen models (one per variant) to one session; PETHs of data and each part, cue- and
    lick-aligned, written to <out>/<variant>/sessions/<label>.{npz,json}."""
    label, fits, out_dir, tpath, spath = (*job, None)[:5]
    ce = _ce()
    from wfield_local import config
    an = config.animal_of(label)
    d = session_design(label, _templates_of(tpath, an), spatial=_spatial_of(spath, an))
    f0 = next(iter(fits.values()))
    if spath:
        if _push_free():
            raise ValueError("FROZEN_PUSH=free is implemented for Allen-area targets only")
        return _predict_spatial(label, fits, out_dir, d, f0)
    if len(d["col_var"]) != len(f0["col_var"]) or (d["col_var"] != f0["col_var"]).any():
        raise ValueError(f"{label}: column layout differs from the animal's pre fit")
    ix = [d["area_names"].index(a) for a in f0["areas"]]
    Rd = d["Rd"][:, ix]
    Z = d["X"] / f0["scale"]
    off = np.arange(int(ce.PETH_S[0] * d["fs"]), int(ce.PETH_S[1] * d["fs"]) + 1)
    pos_order = sorted(set(d["pos"]))
    ci_ = f0["areas"].index("cortex")
    summary = {"label": label}
    # SSE subsets for the drop-spout scores: window x (all frames | each spout's own trials), window offset free
    subsets = {(w, q): wm & ((d["frame_pos"] == q) if q != "all" else True)
               for w, wm in d["win"].items() for q in ["all", *pos_order]}
    sst = {f"{w}|{q}": (((Rd[m] - Rd[m].mean(0)) ** 2).sum(0)).tolist() for (w, q), m in subsets.items() if m.any()}
    drop_sse = {}

    def _sse(pred):
        out = {}
        for (w, q), m in subsets.items():
            if m.any():
                r = Rd[m] - pred[m]
                out[f"{w}|{q}"] = (((r - r.mean(0)) ** 2).sum(0)).tolist()
        return out
    for variant, fit in fits.items():
        beta = fit["beta_loso"].get(label, fit["beta_all"])
        frozen_from = "loso" if label in fit["beta_loso"] else "all_pre"
        Zv = Z @ fit["C"]                                 # the variant's design (effective weights: Z @ (C @ beta))
        qm = np.asarray(fit["vcol"]) == "push"
        pfree = None
        if _push_free() and qm.any():                     # per-session cross-validated push (FREE PUSH above)
            base, pfree = free_push(Zv, Rd, beta, qm, fit["alphas"][FPEN["push"]])
            full = base + pfree
        else:
            full = Zv @ beta

        def contrib(m, Zv=Zv, beta=beta, qm=qm, pfree=pfree):
            x = Zv[:, m & ~qm] @ beta[m & ~qm]
            if (m & qm).any():
                x = x + (pfree if pfree is not None else Zv[:, m & qm] @ beta[m & qm])
            return x
        if "|drop:" in variant or variant in DROP_BASES:
            drop_sse[variant] = _sse(full)
        if "|drop:" in variant:
            continue
        gcol = np.array([FGROUP[v] for v in fit["vcol"]])
        comp = {"data": Rd, "full": full}
        if variant in PART_VARIANTS:                      # a reduced model's parts are not a split of anything
            for g, gs in (*((g, (g,)) for g in FGROUPS), ("spout", SUPER["spout"]), ("lick_all", ("lick", "exec_locked"))):
                m = np.isin(gcol, gs)
                comp[g] = contrib(m) if m.any() else np.zeros_like(Rd)
            pm = np.isin(fit["vcol"], list(ce.POS_VARS))
            comp["pos"] = contrib(pm) if pm.any() else np.zeros_like(Rd)
            if qm.any():
                comp["push"] = contrib(qm)
        peth = np.full((len(comp), len(pos_order), len(ix), len(off)), np.nan, np.float32)
        peth_lick = np.full_like(peth, np.nan)
        n_lick = {}
        for ci, R in enumerate(comp.values()):
            for pi, p in enumerate(pos_order):
                m = d["pos"] == p
                peth[ci, pi] = _peth(R, d["cue_f"][m], off)
                fl = d["first_f"][m & (d["first_f"] >= 0)]
                n_lick[p] = int(len(fl))
                if len(fl):
                    peth_lick[ci, pi] = _peth(R, fl, off)
        musall = {}
        for al, (a0, a1) in MUSALL_WINDOWS.items():
            fr_ = d["cue_f"] if al == "cue" else d["first_f"]
            keep = fr_ >= 0
            if keep.sum() < 2:
                continue
            offm = np.arange(int(round(a0 * d["fs"])), int(round(a1 * d["fs"])))
            mr = musall_r2(_segments(Rd, fr_[keep], offm), _segments(comp["full"], fr_[keep], offm), d["pos"][keep])
            musall[al] = {k: [float(x) for x in v] for k, v in mr.items()}
        r2 = {}
        for w, wm in d["win"].items():
            y, yh = Rd[wm, ci_], comp["full"][wm, ci_]
            r2[w] = float(1.0 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)) if wm.any() else None
        sess = Path(out_dir) / variant / "sessions"
        sess.mkdir(parents=True, exist_ok=True)
        # PER-TRIAL PARTS (Priya 10-24: "save outputs so we can re-plot or re-analyse without re-doing calculations"):
        # every part (data, full, each FGROUP part ...) averaged over each trial's Musall window, so trial-level
        # questions (attempt class, landing, touch) are a re-read joined on trial_id, not a refit.
        tr_parts = np.full((len(comp), len(d["pos"]), len(MUSALL_WINDOWS), len(ix)), np.nan, np.float32)
        for wi, (al, (a0, a1)) in enumerate(MUSALL_WINDOWS.items()):
            fr_ = d["cue_f"] if al == "cue" else d["first_f"]
            offm = np.arange(int(round(a0 * d["fs"])), int(round(a1 * d["fs"])))
            for ci, R in enumerate(comp.values()):
                tr_parts[ci, :, wi] = trial_means(R, fr_, offm)
        np.savez_compressed(sess / f"{label}.npz", peth=peth, peth_lick=peth_lick, peth_off_s=off / d["fs"],
                            trial_parts=tr_parts, trial_id=d["trial_id"], trial_pos=np.asarray(d["pos"], dtype=str),
                            trial_windows=np.array(list(MUSALL_WINDOWS)), parts=np.array(list(comp)),
                            areas=np.array(fit["areas"]))
        meta = {"label": label, "animal": d["animal"], "epoch": d["epoch"], "day": d["day"],
                "order": f"frozen_pre:{variant}", "variant": variant, "frozen_from": frozen_from,
                "pre_labels": fit["pre_labels"], "alphas": fit["alphas"], "loso_r2_pre": fit["loso_r2"],
                "gate": os.environ.get("OROFACIAL_GATE", "reference"), "lick_set": d["lick_set"],
                "n_trials": d["n_trials"], "n_lick_aligned_by_pos": n_lick, "pos_order": pos_order,
                "peth_parts": list(comp), "areas": fit["areas"], "frozen_r2_cortex": r2, "r2_musall": musall,
                "stats_version": STATS_VERSION, "grouping_version": GROUPING_VERSION, "run_tag": _run_tag(),
                "push": ("free" if pfree is not None else "frozen") if qm.any() else None}
        (sess / f"{label}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        summary[variant] = r2
        summary["frozen_from"] = frozen_from
    if drop_sse:
        dd = Path(out_dir) / "dropspout" / "sessions"
        dd.mkdir(parents=True, exist_ok=True)
        (dd / f"{label}.json").write_text(json.dumps(
            {"label": label, "animal": d["animal"], "epoch": d["epoch"], "areas": f0["areas"], "pos_order": pos_order,
             "sst": sst, "sse": drop_sse, "lick_set": d["lick_set"], "stats_version": STATS_VERSION},
            default=str), encoding="utf-8")
    return summary


#: evoked-response windows for the spatial outputs (as scripts/lp_encoder_decompose.py): (response, baseline), s
EVOKED_WINDOWS = {"lick": ((0.0, 1.0), (-0.3, -0.1)), "cue": ((0.5, 1.5), (-0.5, 0.0))}


#: PIXEL PETH vs TRIAL-BY-TRIAL (Priya 10-25: Musall et al. 2019 PETH / trial-by-trial prediction maps at pixel level).
#: FROZEN_PIX_MUSALL=1: for the spatial-target runs, save per MUSALL window the target-space second moments that
#: `musall_r2` needs, so a pixel's R2 = 1 - diag(W E W') / diag(W S W') (W = the animal's basis):
#:   avg    per-spout trial-averaged traces, each centred on its own mean over spouts x time (data and prediction apart)
#:   trial  single trials minus their spout's mean (data and prediction apart)
#: keys mus_<window>_<avg|trial>_<E|S>; only trials whose window lies inside the record. Variants: joint, the alone
#: (<g>_only) and the group drops (no_<g>) -- the per-variable drops keep all-frames moments only.
PIX_MUSALL = os.environ.get("FROZEN_PIX_MUSALL", "") in ("1", "true", "yes")


def _pix_musall_variant(variant: str) -> bool:
    return variant == "joint" or variant.endswith("_only") or (variant.startswith("no_") and "|" not in variant)


def musall_moments(d: dict, Z: np.ndarray, Wf: np.ndarray, R: np.ndarray, fs: float) -> dict:
    P = Z @ Wf                                           # frames x targets prediction
    out = {}
    for al, (a0, a1) in MUSALL_WINDOWS.items():
        fr_ = d["cue_f"] if al == "cue" else d["first_f"]
        offm = np.arange(int(round(a0 * fs)), int(round(a1 * fs)))
        idx = fr_[:, None] + offm[None, :]
        ok = (fr_ >= 0) & (idx.min(1) >= 0) & (idx.max(1) < len(R))
        if ok.sum() < 2:
            continue
        idx, pos = idx[ok], np.asarray(d["pos"])[ok]
        D, Q = R[idx], P[idx]                            # (trials, t, targets)
        Dt, Qt = D.copy(), Q.copy()
        Dm, Qm = [], []
        for q in np.unique(pos):
            m = pos == q
            dm, qm = D[m].mean(0), Q[m].mean(0)
            Dt[m] -= dm
            Qt[m] -= qm
            Dm.append(dm)
            Qm.append(qm)
        Dm, Qm = np.stack(Dm), np.stack(Qm)
        Dm -= Dm.mean((0, 1))
        Qm -= Qm.mean((0, 1))
        for kind, (a_, b_) in (("avg", (Dm, Qm)), ("trial", (Dt, Qt))):
            A = a_.reshape(-1, a_.shape[-1])
            Bm = (a_ - b_).reshape(-1, a_.shape[-1])
            out[f"mus_{al}_{kind}_E"] = (Bm.T @ Bm).astype(np.float32)
            out[f"mus_{al}_{kind}_S"] = (A.T @ A).astype(np.float32)
    return out


def residual_moment(Sxx: np.ndarray, Sxy: np.ndarray, Syy: np.ndarray, Wf: np.ndarray) -> np.ndarray:
    """(Y - Z Wf)'(Y - Z Wf) from Z'Z, Z'Y, Y'Y: the residual second moment in target space (R x R)."""
    SW = Sxy.T @ Wf
    return Syy - SW - SW.T + Wf.T @ Sxx @ Wf


def _predict_spatial(label: str, fits: dict, out_dir: str, d: dict, f0: dict) -> dict:
    """Spatial targets (basis / grid): per variant, the EVOKED response (response window minus baseline, cue- and
    lick-aligned) of the data and of each group's part, per spout x target, from PETHs computed in the small spaces --
    part PETH = PETH(Z) @ weights, data PETH = PETH(Y) @ Ra -- so no (frames x targets) array is built. R^2 and the
    Musall readouts on the cortex-mean target. No drop-spout (area runs have it)."""
    ce = _ce()
    ix = [d["area_names"].index(a) for a in f0["areas"]]
    lazy = d["Rd"] is None
    Ra = d["Ra"][:, ix] if lazy else None
    Rd = None if lazy else d["Rd"][:, ix]
    ci_ = f0["areas"].index("cortex")
    y_c = (d["Y"] @ Ra[:, ci_]) if lazy else Rd[:, ci_]
    Z = d["X"] / f0["scale"]
    fs = d["fs"]
    off = np.arange(int(ce.PETH_S[0] * fs), int(ce.PETH_S[1] * fs) + 1)
    offs = off / fs
    pos_order = sorted(set(d["pos"]))
    wins = {al: ((offs >= r0) & (offs < r1), (offs >= b0) & (offs <= b1))
            for al, ((r0, r1), (b0, b1)) in EVOKED_WINDOWS.items()}

    def evk(P):                                           # (..., off) -> (...,) response minus baseline
        return {al: np.nanmean(P[..., w], -1) - np.nanmean(P[..., b], -1) for al, (w, b) in wins.items()}
    # PETHs of the regressors and of the data, per spout and alignment, in the small spaces
    PZ, PD, n_al = {}, {}, {}
    for p in pos_order:
        m = d["pos"] == p
        for al in ("cue", "lick"):
            fr_ = d["cue_f"][m] if al == "cue" else d["first_f"][m & (d["first_f"] >= 0)]
            n_al[(p, al)] = int(len(fr_))
            if not len(fr_):
                continue
            PZ[(p, al)] = _peth(Z, fr_, off)                                  # (p, off)
            PD[(p, al)] = (Ra.T @ _peth(d["Y"], fr_, off)) if lazy else _peth(Rd, fr_, off)   # (R, off)
    # PIXEL R2 (Priya 10-23: unique R2 maps at pixel / SVD level): all-frames second moments in TARGET space, so a
    # pixel's SSE / SST is diag(W E W') without a frames x pixels array (`scripts/frozen_r2_maps.py`)
    Sxx = Z.T @ Z
    Rfull = ((d["Y"] @ Ra) if lazy else Rd) if PIX_MUSALL else None      # frames x targets, for the PETH / trial moments
    Sxy = (Z.T @ d["Y"]) @ Ra if lazy else Z.T @ Rd
    Syy = (Ra.T @ (d["Y"].T @ d["Y"]) @ Ra) if lazy else Rd.T @ Rd
    summary = {"label": label}
    for variant, fit in fits.items():
        if "|drop:" in variant:
            continue
        beta = fit["beta_loso"].get(label, fit["beta_all"])
        frozen_from = "loso" if label in fit["beta_loso"] else "all_pre"
        Ce = fit["C"]
        gcol = np.array([FGROUP[v] for v in fit["vcol"]])
        parts = {"full": Ce @ beta}
        if variant in PART_VARIANTS:
            for g, gs in (*((g, (g,)) for g in FGROUPS), ("spout", SUPER["spout"]), ("lick_all", ("lick", "exec_locked"))):
                mm = np.isin(gcol, gs)
                parts[g] = Ce[:, mm] @ beta[mm] if mm.any() else np.zeros((Z.shape[1], len(ix)))
        names = ["data", *parts]
        ev = {al: np.full((len(names), len(pos_order), len(ix)), np.nan, np.float32) for al in ("cue", "lick")}
        for pi, p in enumerate(pos_order):
            for al in ("cue", "lick"):
                if (p, al) not in PZ:
                    continue
                e_d = evk(PD[(p, al)])[al]
                ev[al][0, pi] = e_d
                for k, Wm in enumerate(parts.values(), start=1):
                    ev[al][k, pi] = evk(Wm.T @ PZ[(p, al)])[al]
        pred_c = Z @ parts["full"][:, ci_]
        r2 = {}
        for w, wm in d["win"].items():
            yy, yh = y_c[wm], pred_c[wm]
            r2[w] = float(1.0 - ((yy - yh) ** 2).sum() / max(((yy - yy.mean()) ** 2).sum(), 1e-12)) if wm.any() else None
        musall = {}
        for al, (a0, a1) in MUSALL_WINDOWS.items():
            fr_ = d["cue_f"] if al == "cue" else d["first_f"]
            keep = fr_ >= 0
            if keep.sum() < 2:
                continue
            offm = np.arange(int(round(a0 * fs)), int(round(a1 * fs)))
            mr = musall_r2(_segments(y_c[:, None], fr_[keep], offm), _segments(pred_c[:, None], fr_[keep], offm),
                           d["pos"][keep])
            musall[al] = {k: [float(x) for x in v] for k, v in mr.items()}
        sess = Path(out_dir) / variant / "sessions"
        sess.mkdir(parents=True, exist_ok=True)
        E = residual_moment(Sxx, Sxy, Syy, parts["full"])
        extra = musall_moments(d, Z, parts["full"], Rfull, fs) if (PIX_MUSALL and Rfull is not None
                                                                   and _pix_musall_variant(variant)) else {}
        np.savez_compressed(sess / f"{label}.npz", evoked_cue=ev["cue"], evoked_lick=ev["lick"],
                            resid_mom=E.astype(np.float32), data_mom=Syy.astype(np.float32), n_frames=len(Z), **extra)
        meta = {"label": label, "animal": d["animal"], "epoch": d["epoch"], "day": d["day"],
                "order": f"frozen_pre:{variant}", "variant": variant, "frozen_from": frozen_from,
                "pre_labels": fit["pre_labels"], "alphas": fit["alphas"], "loso_r2_pre": fit["loso_r2"],
                "gate": os.environ.get("OROFACIAL_GATE", "reference"), "lick_set": d["lick_set"],
                "n_trials": d["n_trials"], "n_aligned": {f"{p}|{al}": n for (p, al), n in n_al.items()},
                "pos_order": pos_order, "evoked_parts": names, "targets": f0["areas"], "evoked_windows": EVOKED_WINDOWS,
                "frozen_r2_cortex": r2, "r2_musall": musall, "areas": ["cortex"], "stats_version": STATS_VERSION}
        (sess / f"{label}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        summary[variant] = r2
        summary["frozen_from"] = frozen_from
    # PER-POSITION unique R2 maps (Priya 10-23 "and what about per-position unique r2"): for each frozen drop-one-spout
    # fit base|drop:q, the residual second moments of base and drop on spout q's OWN trial frames and on all frames
    # (dR2 = (SSE[drop] - SSE[base]) / SST, as `summarise_drop`; maps in scripts/frozen_r2_maps.py)
    drops = [v for v in fits if "|drop:" in v]
    if drops:
        fp = d["frame_pos"]
        mom = {"all": (Sxx, Sxy, Syy)}
        for q in sorted({v.partition("|drop:")[2] for v in drops}):
            m_ = fp == q
            if m_.sum() < 10:
                continue
            Zq = Z[m_]
            Yq = (d["Y"][m_] @ Ra) if lazy else Rd[m_]
            mom[q] = (Zq.T @ Zq, Zq.T @ Yq, Yq.T @ Yq)
        pix = {f"data|{k}": v[2].astype(np.float32) for k, v in mom.items()}
        for v in drops:
            base, _, q = v.partition("|drop:")
            for sub in ("all", q):
                if sub not in mom:
                    continue
                for tag, fv in (("base", fits[base]), ("drop", fits[v])):
                    key = f"{base}|{q}|{sub}|{tag}"
                    if key in pix:
                        continue
                    b_ = fv["beta_loso"].get(label, fv["beta_all"])
                    pix[key] = residual_moment(*mom[sub], fv["C"] @ b_).astype(np.float32)
            # WHOLE-SPOUT: the drop fit AND q's spout-locked templates swapped for the other spouts' (own trials)
            sw = (d.get("spl_swap") or {}).get(q)
            if sw is not None and q in mom:
                rows, D = sw
                fv = fits[v]
                Zq = Z[rows] + D / fv["scale"]
                Yq = (d["Y"][rows] @ Ra) if lazy else Rd[rows]
                b_ = fv["beta_loso"].get(label, fv["beta_all"])
                pix[f"{base}|{q}|{q}|whole"] = residual_moment(Zq.T @ Zq, Zq.T @ Yq, Yq.T @ Yq,
                                                               fv["C"] @ b_).astype(np.float32)
        dd = Path(out_dir) / "dropspout" / "sessions"
        dd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dd / f"{label}_pix.npz", **pix)
        (dd / f"{label}_pix.json").write_text(json.dumps({"label": label, "animal": d["animal"], "epoch": d["epoch"],
                                                          "targets": f0["areas"], "n_frames": {k: int((fp == k).sum())
                                                          for k in mom if k != "all"}}, indent=1), encoding="utf-8")
    return summary


def summarise_drop(out: Path) -> None:
    """Frozen drop-one-spout table: per session, base, spout q, window, subset (all frames | q's own trials), area:
    dR2 = (SSE[base|drop:q] - SSE[base]) / SST; then animal x epoch means (rule 8: animals first).
    -> <out>/dropspout/frozen_dropspout{,_by_animal}.csv"""
    import pandas as pd
    rows = []
    for js in sorted((Path(out) / "dropspout" / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if m["label"][-4:] < "0801":
            continue
        for base in DROP_BASES:
            if base not in m["sse"]:
                continue
            for q in m["pos_order"]:
                v = f"{base}|drop:{q}"
                if v not in m["sse"]:
                    continue
                for key, sst in m["sst"].items():
                    w, sub = key.split("|")
                    if sub not in ("all", q):
                        continue
                    d = (np.asarray(m["sse"][v][key]) - np.asarray(m["sse"][base][key])) / np.maximum(sst, 1e-12)
                    for ai, area in enumerate(m["areas"]):
                        rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "base": base,
                                     "spout": q, "window": w, "subset": "all" if sub == "all" else "own",
                                     "area": area, "dR2": float(d[ai])})
    if not rows:
        print("  (no drop-spout files)")
        return
    T = pd.DataFrame(rows)
    A = (T.groupby(["animal", "epoch", "base", "spout", "window", "subset", "area"])
         .agg(dR2=("dR2", "mean"), n_sess=("label", "nunique")).reset_index())
    dd = Path(out) / "dropspout"
    T.to_csv(dd / "frozen_dropspout.csv", index=False, float_format="%.5g")
    A.to_csv(dd / "frozen_dropspout_by_animal.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    order = ["pre", "acute", "subacute", "chronic"]
    for area in ("cortex", "SSp-m_left"):
        S = A[(A.spout == "far_contra") & (A.window == "lick") & (A.subset == "own") & (A["area"] == area)]
        if len(S):
            P = (S.pivot_table(index=["base", "animal"], columns="epoch", values="dR2") * 1e3)
            print(f"\n=== frozen drop far_contra, {area}, lick window, far-contra trials, dR2 x1e3")
            print(P.reindex(columns=[c for c in order if c in P.columns]).round(1).to_string())
    print(f"\n-> {dd / 'frozen_dropspout_by_animal.csv'}")


def summarise_musall(out: Path, variants) -> None:
    """Per variant x animal x epoch x align x measure (frame / trial / avg) x area, mean over sessions (rule 8: animals
    first) -> <out>/frozen_musall_r2{,_by_animal}.csv; prints cortex + contralesional SSp-m, both windows."""
    import pandas as pd
    rows = []
    for v in variants:
        for js in sorted((Path(out) / v / "sessions").glob("*.json")):
            m = json.loads(js.read_text(encoding="utf-8"))
            for al, d_ in (m.get("r2_musall") or {}).items():
                for meas, vals in d_.items():
                    for area, x in zip(m["areas"], vals):
                        rows.append({"variant": v, "label": m["label"], "animal": m["animal"], "epoch": m["epoch"],
                                     "align": al, "measure": meas, "area": area, "r2": x})
    if not rows:
        print("  (no Musall readouts)")
        return
    T = pd.DataFrame(rows)
    A = T.groupby(["variant", "animal", "epoch", "align", "measure", "area"]).r2.mean().reset_index()
    T.to_csv(Path(out) / "frozen_musall_r2.csv", index=False, float_format="%.4f")
    A.to_csv(Path(out) / "frozen_musall_r2_by_animal.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 240)
    order = [e for e in ("pre", "acute", "subacute", "chronic") if e in set(A.epoch)]
    for area in ("cortex", "SSp-m_left"):
        S = A[A["area"] == area].pivot_table(index=["align", "measure", "variant"], columns=["animal", "epoch"], values="r2")
        S = S.reindex(columns=pd.MultiIndex.from_product([sorted(set(A.animal)), order]))
        print(f"\n=== frozen Musall-style R2, {area} (held out; pre = leave-one-session-out)")
        print(S.round(3).to_string())
    print(f"\n-> {Path(out) / 'frozen_musall_r2_by_animal.csv'}")


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tag", default="gate-attempted")
    ap.add_argument("--variants", nargs="+", default=[v for v in VARIANTS if v not in _VDROP and v not in _LAGDROP],
                    choices=VARIANTS)   # novar_* / lag drops opt-in
    ap.add_argument("--no-drop-spout", action="store_true", help="skip the frozen drop-one-spout variants")
    ap.add_argument("--reuse-stats", action="store_true", help="reuse fit/pre_stats.pkl (skip the pre statistics pass)")
    ap.add_argument("--summarise-drop", action="store_true", help="only rebuild the drop-spout table from saved files")
    ap.add_argument("--target", default="areas", choices=("areas", "basis", "grid"),
                    help="Allen areas (default) | per-animal shared SVD basis | 8 x 8 pixel grid (SPATIAL TARGETS)")
    a = ap.parse_args(argv)
    if a.target != "areas":
        if a.variants == list(VARIANTS):
            a.variants = [*PART_VARIANTS, *_DROP, "task_only"]   # parts (maps) + drop variants (unique-R2 maps)
    if a.summarise_drop:
        summarise_drop(oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}")
        return 0
    labels = sorted(a.labels or (_ce().sessions_available() if a.all else []))
    labels = [x for x in labels if x[-4:] >= "0801"]                       # August-only pre baseline (primary)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    assert_writable(out)
    pre = [x for x in labels if epochs.epoch_of(x) == "pre"]
    print(f"[frozen] {len(labels)} sessions, {len(pre)} August pre -> {out}", flush=True)
    import pickle
    sp = out / "fit" / "pre_stats.pkl"
    tp = out / "fit" / "templates.pkl"
    (out / "fit").mkdir(parents=True, exist_ok=True)
    spath = str(out / "fit") if a.target != "areas" else None
    spat_ok = a.target == "areas" or all((out / "fit" / f"spatial_{config.animal_of(x)}.pkl").exists() for x in pre)
    saved = pickle.loads(sp.read_bytes()) if (a.reuse_stats and sp.exists() and tp.exists() and spat_ok) else None
    if saved is not None and any(s_.get("stats_version") != STATS_VERSION or s_.get("run_mode", "linear") != _run_tag()
                                 for s_ in saved):
        print(f"  saved statistics are not STATS_VERSION {STATS_VERSION} / running {_run_tag()}: recomputing", flush=True)
        saved = None
    if saved is not None:                                # a re-scoring run: the pre statistics are already saved
        got = [(s_["label"], s_) for s_ in saved if s_["label"] in pre and s_.get("pos_order")]
        print(f"  reusing {len(got)} pre-session statistics from {sp}", flush=True)
    else:
        # pass 1 (behaviour only): the animals' SPOUT-LOCKED templates from their pre sessions (FGROUP comment)
        tparts, tfail = ak.fan_sessions(pre, template_job, jobs=a.jobs, label="template")
        if tfail:
            raise RuntimeError(f"template pass failed: {tfail}")
        tp.write_bytes(pickle.dumps(build_templates([t for _l, t in tparts]), protocol=pickle.HIGHEST_PROTOCOL))
        print(f"  spout-locked templates -> {tp}", flush=True)
        if a.target == "basis":                          # pass 2: the per-animal shared spatial basis
            bparts, bfail = ak.fan_sessions(pre, basis_job, jobs=a.jobs, label="basis")
            if bfail:
                raise RuntimeError(f"basis pass failed: {bfail}")
            for an_, sd_ in build_basis([b for _l, b in bparts]).items():
                (out / "fit" / f"spatial_{an_}.pkl").write_bytes(pickle.dumps(sd_, protocol=pickle.HIGHEST_PROTOCOL))
                print(f"  {an_} shared basis: K = {len(sd_['names'])} components, {sd_['evr'][-1]:.3f} of pre variance",
                      flush=True)
        elif a.target == "grid":
            g_ = grid_spatial(pre[0])
            for an_ in sorted({config.animal_of(x) for x in labels}):
                (out / "fit" / f"spatial_{an_}.pkl").write_bytes(pickle.dumps(g_, protocol=pickle.HIGHEST_PROTOCOL))
            print(f"  grid: {len(g_['names'])} blocks of {GRID_BLOCK} x {GRID_BLOCK} px", flush=True)
        got, fail = ak.fan_sessions([(x, str(tp), spath) for x in pre], stats_job, jobs=a.jobs)
        got = [(it[0], s_) for it, s_ in got]
        for s_ in (s_ for _l, s_ in got):
            s_["stats_version"], s_["run_mode"] = STATS_VERSION, _run_tag()
        for it, err in fail:
            print(f"  !! stats {it}: {err}", flush=True)
        (out / "fit").mkdir(parents=True, exist_ok=True)
        # the pre sessions' sufficient statistics are KEPT: the ordered (movement- / task-first) bounds are linear
        # transforms of the same Gram terms, so they need no second pass over the data
        sp.write_bytes(pickle.dumps([s_ for _l, s_ in sorted(got, key=lambda kv: kv[0])],
                                    protocol=pickle.HIGHEST_PROTOCOL))
    by_an: dict = {}
    for _lab, s in got:
        by_an.setdefault(s["animal"], []).append(s["label"])
    # one fit job per ANIMAL (variants + drop-spout fits are independent across animals; serial in the main process
    # they took ~1.5 min / animal for the variants alone while the workers sat idle). Items stay small: the worker
    # reads the statistics from the pickle written above.
    fit_items = [(an, str(sp), tuple(sorted(by_an[an])), tuple(a.variants), not a.no_drop_spout) for an in sorted(by_an)]
    fres, ffail = ak.fan_sessions(fit_items, fit_animal_job, jobs=min(a.jobs, len(fit_items)), label="fit")
    if ffail:
        raise RuntimeError(f"frozen fits failed: {[(it[0], err) for it, err in ffail]}")
    fits: dict = {}
    for (an, _sp, labs, _v, _d), fa in fres:
        fits[an] = fa
        for v in a.variants:
            f = fa[v]
            np.savez_compressed(out / "fit" / f"{an}_{v}.npz", scale=f["scale"], C=f["C"], vcol=f["vcol"],
                                beta_all=f["beta_all"], col_var=f["col_var"], areas=np.array(f["areas"]))
            print(f"  {an} {v}: {len(labs)} pre sessions, LOSO R^2 (areas summed) {f['loso_r2']:.3f}, "
                  f"alphas {f['alphas']}", flush=True)
    jobs = [(x, fits[config.animal_of(x)], str(out), str(tp), spath) for x in labels if config.animal_of(x) in fits]
    res, fail = _fan(jobs, a.jobs)
    for it, err in fail:
        print(f"  !! predict {it[0]}: {err}", flush=True)
    if not a.no_drop_spout and a.target == "areas":
        summarise_drop(out)
    summarise_musall(out, a.variants)
    for _it, r in sorted(res, key=lambda kv: kv[1]["label"]):
        print(f"  {r['label']} [{r['frozen_from']}] frozen R^2 cortex (all / cue / lick): "
              + " | ".join(f"{v} " + " ".join(f"{x:.3f}" if x is not None else "-" for x in r[v].values())
                           for v in a.variants), flush=True)
    return 0 if not fail else 1


def fit_animal_job(item):
    """All of one animal's frozen fits (each variant, then the drop-one-spout fits under DROP_BASES), from the saved
    pre statistics. Module-level for spawn."""
    import pickle
    an, sp, labs, variants, drop = item
    st = sorted((s for s in pickle.loads(Path(sp).read_bytes()) if s["label"] in set(labs)), key=lambda s: s["label"])
    fits = {v: fit_animal(st, variant=v) for v in variants}
    if drop:                                             # frozen drop-one-spout, each ordering (DROP_BASES)
        for base in DROP_BASES:
            if base in fits:
                for q in st[0]["pos_order"]:
                    fits[f"{base}|drop:{q}"] = fit_animal(st, variant=f"{base}|drop:{q}")
    return fits


def _fan(jobs, n):
    from wfield_local import parallel
    res, fail = parallel.fan_out(jobs, predict_job, jobs=n, label="frozen", log=lambda m: print(m[:120], flush=True))
    res = sorted(res, key=lambda kv: kv[1]["label"])            # rule 9: never completion order
    fail = sorted(fail, key=lambda kv: kv[0][0])
    return res, fail


if __name__ == "__main__":
    raise SystemExit(main())
