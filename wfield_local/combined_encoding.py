"""Combined task + movement ENCODING MODEL of pixel-level widefield activity, Musall et al. 2019 (Nat Neurosci) Fig. 4-5
style (Priya 2026-10-18: "better isolate spout position effects in ENL and cue / lick-aligned analyses with a combined
model ... the Musall fig 4 analysis of reduced vs single variable models ... task-aligned and unique contribution maps
and cvR2 / delta R2"; Fig. 5 PETH vs trial-by-trial; executed vs intended angle).

    python -m wfield_local.combined_encoding PS92_0811 [PS93_0821 ...] [--order movement_first|task_first] [--jobs 3]
    python -m wfield_local.combined_encoding --all [--jobs 3]          # every curated session with LP pose + imaging

DATA. Pixel-level, as Musall: the model is fitted to the session's hemodynamically corrected SVD temporal components
(the ADOPTED `hemo_meegkit_hpfit/SVTcorr`, 100 components, 31.25 Hz -- kept at the full rate: tongue movements are
fast) and every R^2 is evaluated PER PIXEL on the shared Allen grid through `U_atlas` (exactly, from the component-
space residual covariance: R^2_pix = 1 - diag(U C_err U') / diag(U C_tot U')), so maps average across animals. Region
values = the same sums restricted to an Allen area. Rows = imaging frames of ENGAGED trials (the cohort's reference-
position gate), position strobe - 0.5 s .. trial stop + 1 s. CV = 10 folds of contiguous trial blocks (Musall: 10-fold).

VARIABLES (group: variable = regressors; event kernels on a 0.1-s raised-cosine basis):
  task         pos_enl   per-position spout-ARRIVAL kernel (strobe, 0..2 s) + per-position SPOUT-PRESENT state
                         (strobe .. cue; the ENL period is 2.5-49 s long because licks reset it)
               pos_cue   per-position CUE kernel, cue .. the trial's stop (no overlap with the ENL state)
               (no reward variable since 10-18: reward comes with the cue on engaged trials)
  instructed   licks     LP lick-onset + DAQ spout-contact kernels (-0.5 .. 2 s, GCaMP6s), shared across positions
               exec_angle lick-onset kernel MODULATED by the lick's executed peak angle minus its position's mean (the
                         within-position deviation; measured peaks, reach >= 60 px) -- "where the tongue went, beyond
                         where the spout was"
               tongue    tongue tip x / y (mouth frame), protrusion, |speed| (250 fps -> frame means; lags 0/0.1/0.2 s)
               jaw       jaw x / y, |speed|
  uninstructed nose      nose x / y relative to the mouth (head / posture)
               running   treadmill speed
Position specificity lives ONLY in the task group: lick kernels are shared, pose is continuous, so "position" =
position-specific activity beyond what the tongue / jaw / body did.

ORTHOGONALISATION (Musall: video vs movement, uninstructed vs instructed, by QR). ``movement_first`` (primary, Priya
10-18): uninstructed is residualised on instructed, task on all movement -> variance shared between movement and
position is credited to MOVEMENT; position's dR^2 is what movement cannot explain (conservative). ``task_first``:
movement residualised on task (sensitivity). The full model's predictions are the same either way.

MEASURES (per variable and per group; windows all / enl (pre-cue 2 s) / cue (0..1 s) / lick (+-0.2 s of a lick peak);
inside a window the held-out residual's window MEAN is removed -- each model gets a free window offset):
  alone      cvR^2 of a model with only that variable (UNORTHOGONALISED) -- upper bound of its linear information
  dR2        full cvR^2 minus cvR^2 with the variable SHUFFLED in the full (orthogonalised) model -- its unique
             information, with variance SHARED with later groups credited to it (orthogonalisation order).
  dR2_drop   full cvR^2 minus a refit WITHOUT the variable, unorthogonalised -- shared variance credited to nobody (the
             conservative, order-free unique contribution). Every reduced model has its OWN ridge penalties (fix
             2026-10-18; borrowing the full model's made "alone" negative for jaw / instructed). Shuffle = circular shift (>= 60 s) of its columns; for pos_enl / pos_cue the
             position LABELS are permuted across trials (only position SPECIFICITY is removed; the shared
             arrival / cue response stays) -- stated in every figure.
  Fig. 5     task-independent = cvR^2(task + variable) - cvR^2(task alone); task-aligned = alone - task-independent.
             PETH partition: the FULL model fitted once, its prediction split by group weights (no refit), cue-aligned
             per position, per Allen area. Time-resolved SSE (cue-aligned) per model for time-resolved dR^2.
Collinearity: Musall's cumulative subspace angles (|diag R| of the QR of the column-normalised design).

Outputs: <LP_analysis>/<run>/imaging_encoding/<order>/sessions/<label>.npz (+ .json summary). Figures:
`scripts/combined_encoding_figures.py`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SMOKE_ENV = "COMBINED_ENCODING_SMOKE"
ONLY_ENV = "COMBINED_ENCODING_ONLY"      # comma-separated model-name prefixes: fit only those (+ full); own folder   # =1: fit only the models the post-fit code needs (a fast end-to-end check)
#: VIDEO (Priya 10-18, O2 motion-energy pilot `me_pilot_20261018`): comma-separated cameras, e.g. "cam1,cam4". When set
#: AND the session has `video_motion_<cam>.npz`, the top VIDEO_K motion-energy components per camera enter as an extra
#: uninstructed-movement variable, BLANKED outside position strobe .. trial stop (the spout repositions between trials;
#: DECISIONS 2026-10-06 -- no pixel mask, treadmill kept as locomotion). Two readouts, the primary model untouched:
#:   video:full                      Musall placement: video ORTHOGONALISED against every other variable, added last ->
#:                                   video:full - full = what video explains that nothing else does;
#:   video:raw:full / video:drop:*   order-free with raw video IN -> does position / task / target / executed survive
#:                                   video (compare with raw:full - drop:* without video)?
VIDEO_ENV = "COMBINED_ENCODING_VIDEO"
VIDEO_K = 50
#: FR (Priya 10-21: FaceRhythm factors in place of a large motion-energy PC set, if ME adds little on top): the FROZEN
#: factor-set name, e.g. "master_pre_v1" (`scripts/face_rhythm_factors.py`; loader `wfield_local.face_rhythm_outputs`:
#: 25 Hz trial windows + 5 Hz elsewhere, interpolated onto the frames, blanked outside strobe .. trial end like video).
#: Placement as VIDEO -- added last, orthogonalised against every other variable for fr:full, raw in the order-free
#: models: fr:full, fr:alone, fr:raw:full, fr:drop:<group | position>. With VIDEO on as well: frvideo:raw:full (both)
#: and frvideo:alone -> FR beyond ME = frvideo:raw:full - video:raw:full, ME beyond FR = frvideo:raw:full - fr:raw:full.
FR_ENV = "COMBINED_ENCODING_FR"
#: FR_POINTS (with FR): K = number of the set's frozen point-displacement PCs entering as "frpts" (dense POSITION; the
#: paper's best combination was points + FR). Models frpts:alone, frpts:raw:full (position instead of FR),
#: frboth:raw:full (both) -> FR beyond position = frboth - frpts:raw:full; position beyond FR = frboth - fr:raw:full.
FR_POINTS_ENV = "COMBINED_ENCODING_FR_POINTS"
#: FR_SHARE: "1" = the factors' per-frame SHARE (FR's normalised spectrogram, the paper's regressors). Default (unset) =
#: ABSOLUTE expression, weights x the bin's FR normaliser D (Priya 10-21: "absolute fr in encoding"): how much of each
#: movement is happening, not its share of the frame (a share drops whenever any OTHER movement grows).
FR_SHARE_ENV = "COMBINED_ENCODING_FR_SHARE"


def _fr_tag(fr_set: str) -> str:
    """Folder-safe FR set name (`face_rhythm_outputs.set_tag`: ``<bundle>/<set>`` of another camera -> ``<cam>-<set>``)."""
    from wfield_local import face_rhythm_outputs as fro
    return fro.set_tag(fr_set)


def _fr_absolute() -> bool:
    return os.environ.get(FR_SHARE_ENV, "") not in ("1", "true", "yes")
RUN_ENV = "COMBINED_ENCODING_RUN"     # set by --run; an env var so spawned workers see it (tests/test_parallel_results...)
#: LICK-MATCHED subsampling (Priya 10-19: "both executed and target increase chronically" -- chronic sessions have more
#: usable licks, and every held-out gain improves with more data). Path to a JSON {label: target usable licks}; whole
#: ENGAGED trials are dropped at random (seeded) until the session's usable-lick count reaches the target (kept trials
#: stay in time order, so the contiguous-block folds are unchanged in kind). Outputs -> <order>..._matched/.
MATCH_ENV = "COMBINED_ENCODING_MATCH"
SUFFIX_ENV = "COMBINED_ENCODING_SUFFIX"   # appended to the output folder (a rerun that must not overwrite, e.g. "_tr")
#: 3-D EXECUTED SET (Priya 10-19, pilot). cam4's "reach" (protrusion_px) is 99 % the DOWN axis and blind to FORWARD
#: (pose3d crosscheck), while the target set carries spout distance -- so a forward-reach signal could only be credited to
#: target. Modes, all restricted to the SAME licks (cam4-usable AND in the 3-D lick table), so target and executed are
#: fitted on identical licks in every arm:  cam4m  = the cam4 executed set (angle, angle^2, reach) on those licks;
#: 3d = 3-D angle out of the midline plane atan2(ml, hypot(ap, dv)), its square, 3-D reach (mm); 3dxyz = the 3-D peak
#: ml / ap / dv (forward = ap is what cam4 cannot see: far vs close centre differ mainly there, 1.7 vs 0.6 mm on PS92).
#: The table is pose3d `kinematics/licks.csv.gz` (cam4 lick events; joined on trial_id + lick_idx); EXEC3D_TABLE_ENV
#: overrides its path (a snapshot, so a rerunning pose3d chain is never read mid-write). Outputs -> ..._exec-<mode>/.
EXEC_ENV = "COMBINED_ENCODING_EXEC"
EXEC3D_TABLE_ENV = "COMBINED_ENCODING_EXEC3D_TABLE"
#: xyz2 = COORDINATE-MATCHED contest: executed = the 3-D peak ml / ap / dv AND target = the spout's 3-D ml / ap / dv
#: (session median per position, from the pose3d frames; table via TARGET3D_TABLE_ENV). Needed because 3dxyz pits a
#: LINEAR 3-D executed set against target angle / angle^2 / distance: with six positions both are 3-D subspaces of the
#: 5-D between-position space, so a win can come from the coordinate system alone, not from tracking the tongue.
#: polar = COORDINATE-MATCHED in angles (Priya 10-20: "a 3d version that has 3d angle and reach"): executed AND target
#: each = [lateral angle out of the sagittal plane, its square, forward tilt within the sagittal plane, 3-D distance from
#: the mouth zero], the tongue peak vs the spout (same 3-D frame; the spout table's dv is + up, the tongue's down_dv
#: + down). 3dxyz is RETIRED (coordinate mismatch: linear 3-D executed vs angular target); kept for the record only.
EXEC_MODES = ("cam4m", "3d", "3dxyz", "xyz2", "polar")
TARGET3D_TABLE_ENV = "COMBINED_ENCODING_TARGET3D_TABLE"
#: LICK EVENT SET for the lick TIMING kernel (Priya 10-20: "use combined cam1/4 lick events and consider +- lips-only
#: kernel"). cam4 misses small licks (cam1 adds 3-7 % real ones; incomplete licks rise post-stroke, PS92 far_R share
#: .17 -> .86 acute), so their cortical activity was left to position / executed / traces -- a post-stroke bias.
#:   cam4      (default) LP cam4 lick onsets + DAQ contacts (as before)
#:   combined  + cam1-ONLY events (session_poses/cam1_licks/<a>_<d>_events.csv: not cam4-matched, not droplets) in the
#:             same lick kernel
#:   lips      as combined, but cam1-only events with size_class == 'lips' get their OWN kernel (inside `licks`)
#: TIMING ONLY: cam1-only events have no angle / reach (no cam4 trace; no usable 3-D depth below ~1 mm), so they never
#: enter the executed / target sets. Outputs -> ..._licks-<mode>/.
LICKS_ENV = "COMBINED_ENCODING_LICKS"
LICK_MODES = ("cam4", "combined", "lips")
#: BINNED target-vs-executed contest (Priya 10-20, agreed): the continuous contest scales ONE lick kernel by 3-4 smooth
#: terms, so a win could come from the assumed (quadratic) shape. Here both sides are six FREE lick kernels on the same
#: licks: lick_bin_tgt = one kernel per spout of the lick's trial; lick_bin_exec = one kernel per EXECUTED bin = the
#: spout nearest the tongue peak (from spout GEOMETRY, never from the licks' own spout labels -- calibrating bins on
#: target-labelled lick centroids would re-import the target post-stroke). Nearest in 3-D mm when EXEC_ENV has a 3-D
#: set and TARGET3D_TABLE_ENV is given; else in the cam4 (angle, reach) plane, each axis scaled by the six spouts' spread.
#: Both blocks deviation-coded like the position blocks (rank 5 x kernel). Matched in dof, model class and licks;
#: `nokin:category+exbin` is the binned STRICT executed. Built only when set (adds ~12 kernels); outputs -> ..._bins/.
#: WHOLE-SESSION licks (10-20 night; the other window's licks_all.csv): until now every lick source started at the cue,
#: but the 2 s pre-cue (ENL) window -- contact-free by design (the ENL restarts on any contact) -- holds no-contact
#: licks on 5-44 % of trials (PS95 .. PS93, pre-stroke), and contacts end right before it. Modes:
#:   shared  pre-cue + ITI licks join the post-cue lick TIMING kernel (one lick response wherever the lick happens);
#:           contacts over the whole trial windows (raw DAQ channel) in the contact kernel
#:   split   post-cue kernel as before + its OWN pre-cue kernel + its own ITI kernel (all inside `licks`) -- the
#:           spontaneous-vs-cued contrast is split vs shared (Priya 10-20)
#: Timing only (no angle). Outputs -> ..._licksall-<mode>/.
LICKS_ALL_ENV = "COMBINED_ENCODING_LICKS_ALL"
BINS_ENV = "COMBINED_ENCODING_BINS"
#: CALIBRATED executed bins (10-20, measured on the 3-D tables): the tongue peak does not land ON the spout table's
#: position -- it lands ~2 mm deeper (the table marks the rod's lower edge) and laterally compressed (~0.4x at close,
#: ~0.6x at far spouts), so raw nearest-spout put most close-spout licks in FAR bins (PS92 pre agreement 0.46). The
#: calibration table (`scripts/exec_bins_calibration.py`) gives, per animal x spout, where that animal's tongue lands
#: PRE-STROKE (median 3-D peak over its August pre sessions; for a pre session, excluding that session) and the pooled
#: within-spout SD per axis; executed bin = nearest calibrated landing point in SD units. Pre-stroke-only, so
#: post-stroke departures stay visible. Used when given and the executed set is 3-D.
BINS_CALIB_ENV = "COMBINED_ENCODING_BINS_CALIB"
#: SOFT-BIN WIDTH (Priya 10-24, sensitivity sweep 0.5 / 1 / 2): the softmax temperature of the executed landing-zone
#: weights in units of the pooled within-spout SD (1 = default: the weights are the posterior of "which spout's pre-stroke
#: landing distribution this lick came from"; -> 0 = hard nearest-zone bins; large = all weights 1/6). Read at design
#: time, so one O2 bundle serves every width; a width != 1 adds `_bw<w>` to the output folder.
BINS_WIDTH_ENV = "COMBINED_ENCODING_BINS_WIDTH"


def bins_width() -> float:
    w = float(os.environ.get(BINS_WIDTH_ENV) or 1.0)
    if not w > 0:
        raise ValueError(f"{BINS_WIDTH_ENV} must be > 0, got {w}")
    return w


def _match_trials(T: pd.DataFrame, B, label: str, path: str, seed: int) -> pd.DataFrame:
    """Engaged trials subsampled to the session's lick target (usable licks = finite executed angle)."""
    target = json.loads(Path(path).read_text(encoding="utf-8")).get(label)
    T = T.copy()
    eng = np.flatnonzero(T.engaged.to_numpy() & np.isfinite(T.cue_s.to_numpy()))
    if target is None or not len(eng):
        return T
    ok = np.isfinite(B.lick_angle) if B.lick_angle is not None else np.ones(len(B.lick_on_s), bool)
    on = np.asarray(B.lick_on_s, float)[ok]
    n_l = np.array([((on >= T.cue_s.iat[i]) & (on <= T.stop_s.iat[i] + 1.0)).sum() for i in eng])
    order = np.random.default_rng(seed + 7).permutation(len(eng))
    keep = order[: int(np.searchsorted(np.cumsum(n_l[order]), target)) + 1]
    drop = np.setdiff1d(np.arange(len(eng)), keep)
    T.loc[T.index[eng[drop]], "engaged"] = False
    return T


def _run() -> str:
    """The LP run whose pose / clean / licks feed the movement regressors (MIN_REACH_PX is in px, so the run matters)."""
    import os

    from wfield_local import orofacial_cohort as oc
    return os.environ.get(RUN_ENV) or oc.RUN


FOLDS = 10
BASIS_S = 0.1
#: Kernels (Priya 10-18). cue: per-position, cue .. that trial's STOP (lags to 7 s = the longest trial, zeroed after
#: each trial's own stop -- Musall: "stimulus onset until the end of the trial"); a 2-s kernel had forced the position
#: dR2 to exactly 0 from 2 s while the signal was still high. lick: -0.5 .. +2 s (GCaMP6s decays far slower than
#: Musall's 6f, whose motor kernels were -0.5 .. +1 s). No reward variable: reward comes with the cue on engaged trials
#: (absorbed by the cue kernels) and its consumption is licking.
LAGS = {"strobe": (0.0, 2.0), "cue": (0.0, 7.0), "lick": (-0.5, 2.0)}
CUE_FINE_S, CUE_COARSE_SPACING = 2.0, 0.4
CLAGS = (0.0, 0.1, 0.2)
MIN_REACH_PX = 60.0
SHIFT_MIN_S = 60.0
#: SLOW-DRIFT REMOVAL (Priya 10-18; scripts/encoding_drift_check.py). The adopted preprocessing keeps everything faster
#: than ~43 min by design (trial-contrast analyses are blind to it), but this model explains every frame and
#: cross-validates on contiguous blocks, so slow drift gave PS92_0824 R2 -0.22 (an early-session hemodynamic-correction
#: residual of within-session bleaching; raw channels normal). Cosine regressors with period >= DRIFT_MIN_PERIOD_S over
#: the session are projected out of Y AND of every design column before fitting (Frisch-Waugh-Lovell: identical to
#: including them in every model as nuisance, and the slow variance also leaves the R2 denominator). 600 s: 5x the
#: slowest position block (121 s); a 300 s basis visibly followed engagement-scale activity bursts (PS94_0917).
DRIFT_MIN_PERIOD_S = 600.0
ALPHA_GRID = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
WINDOWS = ("all", "enl", "cue", "lick")
PETH_S = (-3.0, 3.0)
RESP_BINS = {"c0-0.5": (0.0, 0.5), "c0.5-1.5": (0.5, 1.5), "c1.5-3": (1.5, 3.0)}     # s after the cue
GROUP_OF = {"cue": "task", "arrival": "task", "pos_enl": "task", "pos_cue": "task",
            "licks": "instructed", "exec_angle": "instructed",
            "tongue": "instructed", "jaw": "instructed", "nose": "uninstructed", "running": "uninstructed"}
GROUPS = ("task", "instructed", "uninstructed")
#: DISTANCE (Priya 10-18): every angle variant also carries distance -- target = the spout's distance from the mouth,
#: executed = the lick's reach (peak protrusion) -- so target and executed each have angle, angle^2 and distance.
#: VARIANT variables (Priya 10-18), never in the full model: angle_cue / angle_enl = the cue / ENL blocks modulated by
#: each trial's TARGET (spout) angle and angle^2 (smooth angular tuning instead of 6 categories); tongue_ps = tongue
#: protrusion + speed only (the tongue model WITHOUT x / y, which is a position proxy: the tongue goes to the spout).
#: lick_angle_abs = lick kernel x each lick's ABSOLUTE executed angle (and angle^2): the EXECUTED competitor to the
#: target angle -- does cortex track where the tongue went or where the spout was (they separate post-stroke, when
#: licks aim ipsi)? Subsumes exec_angle (absolute = position mean + within-position deviation).
#: lick_angle_tgt = the SAME lick kernel on the SAME licks as lick_angle_abs, scaled by the TARGET (spout) angle of the
#: lick's trial instead of the tongue's angle (Priya 10-18): target vs executed then differ ONLY in which angle scales the
#: kernel (angle_cue / angle_enl are cue- / ENL-locked and can explain any time in the trial; the lick-locked pair cannot).
EXTRA_GROUP = {"angle_cue": "task", "angle_enl": "task", "tongue_ps": "instructed", "lick_angle_abs": "instructed",
               "lick_angle_tgt": "instructed", "jaw_nd": "instructed", "nose_nd": "uninstructed",
               "dir_kin": "instructed", "kinall": "instructed", "lick_bin_tgt": "instructed",
               "lick_bin_exec": "instructed"}
#: PENALTY groups (2026-10-19, Priya: fix the shared penalty), separate from the CREDIT groups above. One ridge penalty per
#: credit group let the continuous tongue / jaw traces (well constrained, best at a weak penalty) set the penalty of the
#: lick-EVENT terms that share the "instructed" group (lick kernels + per-lick target / executed scalings: many weights
#: from a limited number of licks), which then overfit -- lick-window R2 FELL when traces were added (PS92_0811:
#: target + exec 0.440 -> + traces 0.395; full model 0.378). Event kernels and continuous traces now get their own penalty.
PEN_GROUP_OF = {"cue": "task", "arrival": "task", "pos_enl": "task", "pos_cue": "task", "angle_cue": "task",
                "angle_enl": "task",
                "licks": "lick", "exec_angle": "lick", "lick_angle_abs": "lick", "lick_angle_tgt": "lick",
                "lick_bin_tgt": "lick", "lick_bin_exec": "lick",
                "tongue": "orofacial", "jaw": "orofacial", "tongue_ps": "orofacial", "jaw_nd": "orofacial",
                "dir_kin": "orofacial", "kinall": "orofacial",
                "nose": "uninstructed", "nose_nd": "uninstructed", "running": "uninstructed", "video": "video", "fr": "fr", "frpts": "frpts"}
PEN_GROUPS = ("task", "lick", "orofacial", "uninstructed")
#: DIRECTIONLESS comparison (Priya 10-19: "wouldn't we expect executed to explain more than target ... is it an
#: appropriate comparison?"). The base of the target / executed models kept jaw x / y and nose x / y, so lateral jaw
#: (which steers with the tongue) carried EXECUTED direction frame by frame and made the lick-locked executed angle
#: redundant while target stayed novel. `nodir:*` models use a base WITHOUT any directional kinematic -- jaw_nd = jaw
#: opening (y) + speed, nose_nd = nose y, tongue_ps = protrusion + speed -- and compare: target (lick kernel x spout
#: angle / angle^2 / distance), executed (lick kernel x lick angle / angle^2 / reach), dir_kin = executed direction as
#: CONTINUOUS kinematics (tongue x / y, jaw x, nose x), their pairs, and the 6-category ceiling.
NODIR_BASE = ("cue", "arrival", "licks", "tongue_ps", "jaw_nd", "nose_nd", "running")
#: NO-KINEMATICS comparison (Priya 10-19: "doesn't jaw and tongue protrusion also have some close vs far information"):
#: protrusion IS executed distance and jaw opening scales with reach, so the nodir base still made the executed REACH
#: redundant. `nokin:*` = task timing + lick TIMING kernels + running only (no tongue / jaw / nose traces): target
#: (spout angle / angle^2 / distance) vs executed (lick angle / angle^2 / reach) on the same lick kernel and licks is then
#: the symmetric spatial contest; `kinall` = the LICK EXECUTION as continuous kinematics (every tongue + jaw trace).
#: NOSE is NOT executed movement (head / snout posture, uninstructed; its lateral part includes pre-cue ORIENTING toward
#: the spout, i.e. target-correlated) -- Priya 10-19 -- so it enters only as a separate check (`nokin:kin+nose*`). kinall
#: has many more columns than the 3-amplitude per-lick variables: read it by UNIQUE shares, not raw gains.
NOKIN_BASE = ("cue", "arrival", "licks", "running")
NOKIN_TR = ("nokin:none", "nokin:category", "nokin:target", "nokin:exec", "nokin:target+exec",
            "nokin:category+exec", "nokin:tgtbin", "nokin:exbin", "nokin:tgtbin+exbin", "nokin:category+exbin")
ORDERS = {"movement_first": ("instructed", "uninstructed", "task"),
          "task_first": ("task", "instructed", "uninstructed"),
          "musall": ("instructed", "uninstructed", "task")}
#: groups left UNORTHOGONALISED under an order. "musall" = Musall et al. 2019 as in their code (churchlandlab/ridgeModel
#: delayDec_RegressModel.m: "orthogonalize spontaneous from operant movement regressors"; task regressors raw; video
#: orthogonalised against everything, last): instructed-uninstructed shared variance goes to INSTRUCTED, task-movement
#: shared variance to NOBODY (the shuffle refit lets raw task / movement absorb each other). Our movement_first also
#: residualises task against movement, crediting instructed with the task-shared (cue / lick-locked evoked) variance.
#: Only the SHUFFLE dR2 depends on the order (drop / alone / variant models use raw columns), so a musall run needs only
#: `--only shuf:` (Priya 10-19).
ORTH_RAW = {"musall": ("task",)}
POS_VARS = ("pos_enl", "pos_cue")
#: time-resolved held-out SSE (cue-aligned) is computed for these models
TR_MODELS = ("full", "alone:task", "alone:instructed", "alone:uninstructed",
             "shuf:task", "shuf:instructed", "shuf:uninstructed", "shuf:pos_enl", "shuf:pos_cue",
             "shuf:exec_angle", "drop:task", "drop:instructed", "drop:uninstructed", "drop:pos_cue",
             "drop:exec_angle", "shuf:position", "drop:position", "shufside:position", "shufdist:position")


# =============================================================================== inputs


#: LOCAL INPUT CACHE (Priya 10-21, "should we make our jobs more efficient?"): CACHE_ENV = a directory on a LOCAL disk;
#: load_imaging / load_behaviour then keep a pickled copy of each session's inputs there and later runs read it in seconds
#: instead of over the share (a run's cost is the network read, ~1 GB per session, not the fit). The key covers the
#: session, every COMBINED_ENCODING_* switch, the source of the modules that build the inputs, and the mtimes of the
#: session's source files (imaging files, the orofacial session dir, the DAQ h5, any env-named table), so a changed input
#: or loader makes a new entry. A disposable copy of INPUTS, never results; delete the directory any time. Off when unset.
CACHE_ENV = "WFIELD_INPUT_CACHE"
#: TONGUE_ENV=3d (Priya 10-24 "we should probably use 3d tongue traces"): the continuous tongue block (tongue_x / tongue_y
#: / tongue_prot / tongue_speed) from the per-frame 3-D pose (pose3d/<animal>_<date>_frames.csv.gz, mouth-centred mm:
#: x = ml lateral, y = ap forward, prot = 3-D distance, speed = |d prot / dt|) instead of cam4 px, and the per-lick
#: within-spout angle (`exec_angle`) from the 3-D lateral angle out of the sagittal plane (kinematics licks table) instead
#: of the cam4 peak angle. Frames pair with the cam4 pose rows exactly on (trial_id, t_ms). Tongue IN the mouth (cam4
#: tongue_x == 0, no 3-D point) = 0, the cam4 convention; tongue OUT on cam4 but not triangulated = NaN (skipped by the
#: frame binning). Unset = cam4 (default; all runs before 10-24).
TONGUE_ENV = "COMBINED_ENCODING_TONGUE"


def pose3d_frames_path(rv, animal: str, date: str) -> Path:
    return Path(rv.root("dlc")) / "session_poses" / "pose3d" / f"{animal}_{date}_frames.csv.gz"


def tongue_3d_traces(ix: pd.DataFrame, cam4_tx: np.ndarray, F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x = ml, y = ap, prot = 3-D distance) mm on the cam4 pose rows ``ix`` (trial_id, t_ms); see TONGUE_ENV."""
    J = ix[["trial_id", "t_ms"]].merge(F[["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv"]]
                                       .drop_duplicates(["trial_id", "t_ms"]), on=["trial_id", "t_ms"], how="left")
    ml, ap, dv = (J[c].to_numpy(float) for c in ("tongue_ml", "tongue_ap", "tongue_dv"))
    have = np.isfinite(ml) & np.isfinite(ap) & np.isfinite(dv)
    inside = (np.asarray(cam4_tx, float) == 0) & ~have
    out = []
    for v in (ml, ap, np.sqrt(ml ** 2 + ap ** 2 + dv ** 2)):
        out.append(np.where(have, v, np.where(inside, 0.0, np.nan)))
    return out[0], out[1], out[2]


def lick_angle_3d(L: pd.DataFrame, session: str) -> np.ndarray:
    """Per-lick 3-D lateral angle out of the sagittal plane (deg; pose3d `midline`), NaN without a 3-D peak."""
    path = os.environ.get(EXEC3D_TABLE_ENV)
    if not path:
        from wfield_local.paths import PathResolver
        path = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics" / "licks.csv.gz"
    K = pd.read_csv(path, usecols=["session", "trial_id", "lick_idx", "side_ml", "out_ap", "down_dv"])
    K = K[K.session == session].drop_duplicates(["trial_id", "lick_idx"])
    J = L[["trial_id", "lick_idx"]].merge(K, on=["trial_id", "lick_idx"], how="left")
    ml, ap, dv = (J[c].to_numpy(float) for c in ("side_ml", "out_ap", "down_dv"))
    return np.degrees(np.arctan2(ml, np.hypot(ap, dv)))
_CACHE_CODE = ("wfield_local/combined_encoding.py", "wfield_local/orofacial_cohort.py",
               "wfield_local/orofacial_outputs.py", "wfield_local/trial_windows.py", "wfield_local/beta_maps.py",
               "wfield_local/daq_trials.py", "scripts/pose_kinematics_demo.py")


def _cached(kind: str, label: str, sources, build):
    """build() through the CACHE_ENV disk cache (see CACHE_ENV); plain build() when it is unset."""
    root = os.environ.get(CACHE_ENV)
    if not root:
        return build()
    import hashlib
    import pickle
    repo = Path(__file__).resolve().parents[1]
    code = hashlib.sha1(b"".join((repo / f).read_bytes() for f in _CACHE_CODE if (repo / f).exists())).hexdigest()
    env = sorted((k, v) for k, v in os.environ.items() if k.startswith(("COMBINED_ENCODING_", "OROFACIAL_")))
    mt = []
    for f in sources:
        f = Path(f)
        try:
            mt.append((str(f), f.stat().st_mtime_ns, f.stat().st_size))
        except OSError:
            mt.append((str(f), None, None))
    key = hashlib.sha1(repr((kind, label, code, env, sorted(mt))).encode()).hexdigest()[:20]
    f = Path(root) / kind / f"{label}_{key}.pkl"
    if f.exists():
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:                                       # a torn / stale-format entry: rebuild it
            pass
    v = build()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{f.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(v, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, f)                                          # atomic: concurrent workers never read a partial file
    return v


def _env_tables() -> list:
    return [v for k, v in os.environ.items() if k.startswith("COMBINED_ENCODING_") and v and Path(v).is_file()]


def load_imaging(label: str):
    """(V, ft, Ub, flat, area, names, shape, pstat) -- `_load_imaging`, through the CACHE_ENV local cache."""
    if not os.environ.get(CACHE_ENV):
        return _load_imaging(label)
    from wfield_local import config
    from wfield_local.hemo_variants import ADOPTED
    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    at = res / "allen_aligned_affine8v1"
    fmdir = s["fmdir"] or s["mc"]
    src = [res / f"hemo_{ADOPTED}" / "SVTcorr.npy", at / "U_atlas.npy", at / "allen_brain_mask_native_grid.npy",
           at / "allen_area_atlas_native_grid.npy", s["h5"], *glob.glob(f"{fmdir}/*cleanpairs_*")]
    return _cached("imaging", label, src, lambda: _load_imaging(label))


def _load_imaging(label: str):
    """(V (T, K) float32 adopted hemo-corrected SVD temporal components, frame DAQ s, U_atlas (P_brain, K) float32,
    brain-mask flat indices, area labels per brain pixel, area names {code: name})."""
    from wfield_local import config
    from wfield_local.hemo_variants import ADOPTED
    from wfield_local.locanmf_lick_aligned import _corrected_frame_samples
    from wfield_local.plot_spout_trial_averages import _load_daq_events
    s = next(x for x in config.load_sessions() if x["label"] == label)
    if s.get("regime") != "B":
        raise ValueError(f"{label}: regime {s.get('regime')} (corrected frame map needed)")
    res = Path(s["mc"]) / "wfield_local_results"
    V = np.load(res / f"hemo_{ADOPTED}" / "SVTcorr.npy").T.astype(np.float32)
    at = res / "allen_aligned_affine8v1"
    U = np.load(at / "U_atlas.npy", mmap_mode="r")
    mask = np.load(at / "allen_brain_mask_native_grid.npy").astype(bool)
    area = np.load(at / "allen_area_atlas_native_grid.npy")
    names = {int(c): n for c, n in json.load(open(at / "allen_area_names.json"))}
    # THE IMAGING-MAP MASKS (Priya 10-18: "is encoder analysis using the masked maps we used for the imaging map
    # analyses"): fit / maps on `beta_maps.brain_mask()` (MOB + the hand-painted fibre glue removed); region / cortex
    # values on the ERODED `beta_maps.stat_mask()` (16-px rim out: partial volume + warp jitter, consistent across
    # animals -- see its docstring). Both on the shared 540 x 640 grid.
    from wfield_local import beta_maps
    bm, sm = beta_maps.brain_mask(), beta_maps.stat_mask()
    if bm is not None and bm.shape == mask.shape:
        mask = mask & bm
    flat = np.flatnonzero(mask.ravel())
    pstat = (sm.ravel()[flat] if sm is not None and sm.shape == mask.shape else np.ones(len(flat), bool))
    Ub = np.asarray(U.reshape(-1, U.shape[-1])[flat], np.float32)
    cue = _load_daq_events(s["h5"])
    fmdir = s["fmdir"] or s["mc"]
    fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
    off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
    ft = np.asarray(_corrected_frame_samples(fm, cue["pco_samples"], int(off)), float) / float(cue["sample_rate_hz"])
    n = min(len(ft), len(V))
    return V[:n], ft[:n], Ub, flat, area.ravel()[flat], names, mask.shape, pstat


@dataclass
class Behaviour:
    trials: pd.DataFrame          # trial_id, pos, strobe_s, cue_s, stop_s, engaged
    lick_on_s: np.ndarray
    lick_peak_s: np.ndarray
    lick_pos: np.ndarray
    lick_angle_within: np.ndarray  # executed angle minus position mean; NaN = not usable
    contact_s: np.ndarray
    reward_s: np.ndarray
    analog: dict                   # name -> (t_s, values)
    gate: str
    target_angle: dict = None      # position -> the session's spout angle (deg, + = ipsi / image right), LP geometry
    lick_angle: np.ndarray = None  # each lick's absolute executed peak angle (deg); NaN = not usable
    target_dist: dict = None       # position -> the session's spout distance from the mouth (px), LP geometry
    lick_reach: np.ndarray = None  # each lick's executed peak protrusion (px); NaN where lick_angle is NaN
    exec_amps: list = None         # EXEC_ENV: the executed set's per-lick amplitudes (replace angle / angle^2 / reach)
    target_amps: list = None       # EXEC_ENV xyz2: the target set's per-lick amplitudes (spout 3-D ml / ap / dv)
    extra_lick_on_s: np.ndarray = None   # LICKS_ENV: cam1-only lick onsets added to the lick timing kernel
    lips_on_s: np.ndarray = None         # LICKS_ENV=lips: cam1-only lips-class onsets (their own kernel)
    exec_xyz: list = None          # 3-D EXEC_ENV: each lick's tongue peak [ml, ap, down] (mm, mouth zero); NaN = unusable
    target_xyz: list = None        # ... and its spout [ml, ap, down] (mm), when TARGET3D_TABLE_ENV is given
    session: str = None            # "<animal>_<yyyymmdd>" (the 3-D tables' key)


def load_behaviour(animal: str, date: str, rv) -> Behaviour:
    """`_load_behaviour`, through the CACHE_ENV local cache."""
    if not os.environ.get(CACHE_ENV):
        return _load_behaviour(animal, date, rv)
    from wfield_local import daq_io
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    sd = oo.session_dir(rv, _run(), animal, date)
    src = [p for p in Path(sd).iterdir() if p.is_file()] if Path(sd).is_dir() else [sd]
    src += [daq_io.session_h5(animal, date, rv) or "", *_env_tables(),       # + the engagement gate's inputs
            *(Path(rv.root("behavior_out")) / "sessions" / animal / date).glob(f"{animal}_{date}_*_trials.csv")]
    if oc.ENGAGEMENT_GATE != "reference":
        src.append(oo.root(rv) / oc.RUN / "imaging_encoding" / "qc" / "trial_attempts.csv")
    if os.environ.get(TONGUE_ENV, "") == "3d":                  # a rewritten 3-D frame / lick table makes a new entry
        src.append(pose3d_frames_path(rv, animal, date))
        src.append(os.environ.get(EXEC3D_TABLE_ENV) or
                   Path(rv.root("dlc")) / "session_poses" / "pose3d" / "kinematics" / "licks.csv.gz")
    return _cached("behaviour", f"{animal}_{date}", src, lambda: _load_behaviour(animal, date, rv))


def _load_behaviour(animal: str, date: str, rv) -> Behaviour:
    from wfield_local import daq_io, daq_trials
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import trial_windows as TW
    from wfield_local.behavior_events import session_speed
    oc._repo_on_path()
    import scripts.pose_kinematics_demo as D
    lat = "L"
    sd = oo.session_dir(rv, _run(), animal, date)
    keep, gate = oc.engagement_keep(animal, date, sd)
    b = TW.trial_bounds(animal, date, rv)
    cue_t = oc._cue_table(sd).set_index("trial_id")
    T = pd.DataFrame({"trial_id": b.trial_id.astype(int), "pos": [oc.pos_key(p, lat) for p in b.pos_name],
                      "strobe_s": b.strobe_s.to_numpy(float), "cue_s": b.cue_s.to_numpy(float),
                      "stop_s": b.stop_s.to_numpy(float)})
    T["engaged"] = True if keep is None else T.trial_id.isin(keep)
    T = T[T.trial_id.isin(cue_t.index)].reset_index(drop=True)
    L = pd.read_csv(sd / "licks.csv")
    L = L[L.trial_id.isin(cue_t.index)]
    c0 = L.trial_id.map(cue_t.cue_daq).to_numpy(float)
    on_s, pk_s = c0 + L.on_ms.to_numpy(float) / 1000.0, c0 + L.t_ms.to_numpy(float) / 1000.0
    pos = np.array([oc.pos_key(p, lat) for p in L.position])
    ang = L.peak_angle_deg.to_numpy(float).copy()
    ok = (~L.peak_interp.astype(bool).to_numpy()) & (L.protrusion_px.to_numpy(float) >= MIN_REACH_PX) & np.isfinite(ang)
    ang[~ok] = np.nan
    if os.environ.get(TONGUE_ENV, "") == "3d":                  # exec_angle from the 3-D lateral angle (TONGUE_ENV)
        a3 = lick_angle_3d(L, f"{animal}_{date}")
        within = np.where(ok & np.isfinite(a3), a3, np.nan)
    else:
        within = ang.copy()
    for p in np.unique(pos):                         # within-position deviation of the executed angle
        m = pos == p
        if np.isfinite(within[m]).any():
            within[m] = within[m] - np.nanmean(within[m])
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    cue_of = dict(zip(b.trial_id.astype(int), b.cue_s.to_numpy(float)))
    contact_s = np.concatenate([cue_of[t] + np.asarray(v, float) / 1000.0 for t, v in contacts.items()
                                if t in cue_of and len(v)]) if contacts else np.zeros(0)
    h5 = daq_io.session_h5(animal, date, rv)
    reward_s = np.asarray(daq_trials.decode(h5)["reward_s"], float)
    # analog pose on the pose rows (DAQ s): tongue / jaw cleaned (mouth- / rest-relative), nose relative to the mouth
    S = oo.load_session(rv, _run(), animal, date)
    t_pose = S["index"].daq_s.to_numpy(float)
    cl, pose = S["clean"], S["pose"]
    o = np.load(sd / "clean.npz")["tongue_origin"].astype(float)
    tx, ty = cl.tongue_x.to_numpy(float), cl.tongue_y.to_numpy(float)
    prot = np.hypot(tx, ty)
    tongue_mode = os.environ.get(TONGUE_ENV, "")
    if tongue_mode not in ("", "cam4", "3d"):
        raise ValueError(f"{TONGUE_ENV}={tongue_mode!r}; expected cam4 | 3d")
    if tongue_mode == "3d":                                       # see TONGUE_ENV
        f3 = pose3d_frames_path(rv, animal, date)
        if not f3.exists():                                       # never mix cam4 / 3-D tongue silently
            raise FileNotFoundError(f"{TONGUE_ENV}=3d needs {f3}")
        F3 = pd.read_csv(f3, usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv"])
        tx, ty, prot = tongue_3d_traces(S["index"], tx, F3)
    jx, jy = cl.jaw_x.to_numpy(float), cl.jaw_y.to_numpy(float)
    nlik = pose.nose_likelihood.to_numpy(float) >= 0.6
    nx = np.where(nlik, pose.nose_x.to_numpy(float) - o[0], np.nan)
    ny = np.where(nlik, pose.nose_y.to_numpy(float) - o[1], np.nan)
    fps = 250.0
    sp = lambda v: np.abs(np.gradient(v)) * fps                                   # noqa: E731
    run_t, run_v = session_speed(h5)
    analog = {"tongue_x": (t_pose, tx), "tongue_y": (t_pose, ty), "tongue_prot": (t_pose, prot),
              "tongue_speed": (t_pose, sp(prot)), "jaw_x": (t_pose, jx), "jaw_y": (t_pose, jy),
              "jaw_speed": (t_pose, np.hypot(np.gradient(jx), np.gradient(jy)) * fps),
              "nose_x": (t_pose, nx), "nose_y": (t_pose, ny), "running": (np.asarray(run_t, float),
                                                                          np.asarray(run_v, float))}
    tgt, tdist = {}, {}
    for col, dst in (("spout_angle_deg", tgt), ("spout_dist_px", tdist)):
        if col in L:
            for p_, g_ in L.assign(pk=pos).groupby("pk"):
                a_ = g_[col].to_numpy(float)
                if np.isfinite(a_).any():
                    dst[p_] = float(np.nanmedian(a_))
    reach = np.where(np.isfinite(ang), L.protrusion_px.to_numpy(float), np.nan)
    exec_amps = target_amps = exec_xyz = target_xyz = None
    mode = os.environ.get(EXEC_ENV, "")
    if mode:
        ang, reach, exec_amps, exec_xyz = _exec_3d(L, ang, reach, mode, f"{animal}_{date}")
        if mode in ("xyz2", "polar") or os.environ.get(TARGET3D_TABLE_ENV):
            sml, sap, sdv = _target_3d(L, f"{animal}_{date}")
            target_xyz = [sml, sap, -sdv]                       # spout dv is + up -> down = -dv
            if mode == "xyz2":
                target_amps = [sml, sap, sdv]
            elif mode == "polar":
                target_amps = _polar(*target_xyz)
    extra, lips = _cam1_lick_events(rv, animal, date, cue_t, os.environ.get(LICKS_ENV, "cam4"))
    ctx = None
    if os.environ.get(LICKS_ALL_ENV):
        ctx, contact_s = _context_licks(rv, animal, date, sd, b, contact_s)
    B_ = Behaviour(T, on_s, pk_s, pos, within, np.asarray(contact_s, float), reward_s, analog, gate, tgt, ang,
                   tdist, reach, exec_amps, target_amps, extra, lips, exec_xyz, target_xyz, f"{animal}_{date}")
    B_.context_lick_on_s = ctx
    return B_


def _context_licks(rv, animal: str, date: str, sd: Path, b: pd.DataFrame, contact_s: np.ndarray):
    """LICKS_ALL_ENV: ({'pre_cue': onsets, 'iti': onsets} in DAQ s from the cam4 whole-session table licks_all.csv,
    contacts over the whole trial windows (strobe - 0.5 .. stop + 3 s) from the raw DAQ channel)."""
    from wfield_local import dlc_frames
    f = sd / "licks_all.csv"
    if not f.exists():                                  # never mix event sets silently
        raise FileNotFoundError(f"{LICKS_ALL_ENV} needs {f} (python -m wfield_local.orofacial_outputs all-licks)")
    A = pd.read_csv(f, usecols=["context", "daq_on_s"])
    ctx = {c: np.sort(A.loc[A.context == c, "daq_on_s"].to_numpy(float)) for c in ("pre_cue", "iti")}
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
    raw = np.sort(np.asarray(dlc_frames.lick_onsets(animal, date, sid, rv), float))
    st = np.where(np.isfinite(b.strobe_s), b.strobe_s, b.cue_s - 3.0).astype(float) - 0.5
    en = b.stop_s.to_numpy(float) + 3.0
    keep = np.zeros(len(raw), bool)
    for a_, e_ in zip(st, en):
        keep |= (raw >= a_) & (raw <= e_)
    return ctx, raw[keep]


def _cam1_lick_events(rv, animal: str, date: str, cue_t: pd.DataFrame, mode: str):
    """LICKS_ENV: (extra lick onsets, lips onsets) in DAQ s from the cam1 2-D detector; (None, None) for cam4."""
    if mode not in LICK_MODES:
        raise ValueError(f"{LICKS_ENV}={mode!r}; expected one of {LICK_MODES}")
    if mode == "cam4":
        return None, None
    f = Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / f"{animal}_{date}_events.csv"
    if not f.exists():                                  # never mix event sets silently
        raise FileNotFoundError(f"{LICKS_ENV}={mode} needs {f}")
    E = pd.read_csv(f)
    only = ~E.cam4_matched.fillna(False).astype(bool) & ~E.droplet.fillna(False).astype(bool)
    E = E[only & E.trial_id.isin(cue_t.index)]
    t = E.trial_id.map(cue_t.cue_daq).to_numpy(float) + E.on_ms.to_numpy(float) / 1000.0
    is_lips = (E.size_class == "lips").to_numpy()
    if mode == "combined":
        return np.sort(t[np.isfinite(t)]), None
    return np.sort(t[np.isfinite(t) & ~is_lips]), np.sort(t[np.isfinite(t) & is_lips])


def _polar(ml, ap, down) -> list:
    """[lateral angle, its square, forward tilt, 3-D distance] from mouth-zero-relative ml / ap / down (mm). Lateral =
    out of the sagittal plane atan2(ml, hypot(ap, down)); tilt = forward of straight down atan2(ap, down); both stable
    because licks run mostly down (unlike atan2(ml, ap), which wraps). Lateral is centred before squaring."""
    ml, ap, down = (np.asarray(x, float) for x in (ml, ap, down))
    lat = np.degrees(np.arctan2(ml, np.hypot(ap, down)))
    lat_c = lat - np.nanmean(lat)
    return [lat_c, lat_c ** 2, np.degrees(np.arctan2(ap, down)), np.sqrt(ml ** 2 + ap ** 2 + down ** 2)]


def _target_3d(L: pd.DataFrame, session: str) -> list:
    """xyz2: each lick's spout 3-D position (session median per position) -> [ml, ap, dv] per lick."""
    path = os.environ.get(TARGET3D_TABLE_ENV)
    if not path:
        raise FileNotFoundError(f"{EXEC_ENV}=xyz2 needs {TARGET3D_TABLE_ENV} (per-session spout 3-D positions)")
    S = pd.read_csv(path)
    S = S[S.session == session].set_index("position")
    if S.empty:
        raise FileNotFoundError(f"no spout 3-D positions for {session} in {path}")
    return [L.position.map(S[c]).to_numpy(float) for c in ("spout_ml", "spout_ap", "spout_dv")]


def _exec_3d(L: pd.DataFrame, ang: np.ndarray, reach: np.ndarray, mode: str, session: str):
    """EXEC_ENV: restrict the usable licks to those with a 3-D peak and return (angle, reach, executed amplitudes,
    tongue peak [ml, ap, down] mm)."""
    if mode not in EXEC_MODES:
        raise ValueError(f"{EXEC_ENV}={mode!r}; expected one of {EXEC_MODES}")
    path = os.environ.get(EXEC3D_TABLE_ENV)
    if not path:
        from wfield_local.paths import PathResolver
        path = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics" / "licks.csv.gz"
    K = pd.read_csv(path, usecols=["session", "trial_id", "lick_idx", "reach_mm", "side_ml", "out_ap", "down_dv"])
    K = K[K.session == session].drop_duplicates(["trial_id", "lick_idx"])
    if K.empty:
        raise FileNotFoundError(f"no 3-D licks for {session} in {path}")
    J = L[["trial_id", "lick_idx"]].merge(K, on=["trial_id", "lick_idx"], how="left")
    ml, ap, dv, r3 = (J[c].to_numpy(float) for c in ("side_ml", "out_ap", "down_dv", "reach_mm"))
    ok = np.isfinite(ang) & np.isfinite(ml) & np.isfinite(ap) & np.isfinite(dv) & np.isfinite(r3)
    ang = np.where(ok, ang, np.nan)
    reach = np.where(ok, reach, np.nan)
    xyz = [np.where(ok, x, np.nan) for x in (ml, ap, dv)]           # kinematics table: down_dv is + down
    if mode == "cam4m":
        return ang, reach, None, xyz
    if mode in ("3dxyz", "xyz2"):
        return ang, reach, xyz, xyz
    if mode == "polar":
        return ang, reach, [np.where(ok, x, np.nan) for x in _polar(ml, ap, dv)], xyz
    # mode == "3d": angle out of the midline plane = pose3d `midline` (r = 1.0000). NOT atan2(ml, ap): licks run mostly
    # DOWN, ap ~ 0, so the horizontal angle wraps (SD 77 deg on PS92 0807 vs 12 deg for this one)
    a3 = np.where(ok, np.degrees(np.arctan2(ml, np.hypot(ap, dv))), np.nan)
    a3 = a3 - np.nanmean(a3)                                         # centred before squaring, as the cam4 angle
    return ang, reach, [a3, a3 ** 2, np.where(ok, r3, np.nan)], xyz


def exec_bins(B: Behaviour, weights: bool = False):
    """BINS_ENV: (target label, executed label, frame) per lick; '' = lick not usable. Executed label = the spout NEAREST
    the tongue peak, from spout geometry only (see BINS_ENV). Usable = exactly the licks of the continuous contest."""
    n = len(B.lick_on_s)
    empty = np.full(n, "", object)
    if B.lick_angle is None:
        return (empty, empty.copy(), "none") + ((np.zeros((n, 0)), []) if weights else ())
    usable = np.isfinite(B.lick_angle)
    scale = None
    calib = os.environ.get(BINS_CALIB_ENV)
    if B.exec_xyz is not None and calib and B.session:
        C = pd.read_csv(calib)
        C = C[C.animal == B.session[:4]]
        C = C[C.calib_for == (B.session if (C.calib_for == B.session).any() else "default")]
        if C.empty:
            raise FileNotFoundError(f"{BINS_CALIB_ENV}: no calibration for {B.session} in {calib}")
        Ex, frame = np.column_stack(B.exec_xyz), "3d_calibrated"
        geo = {r.position: np.array([r.ml, r.ap, r.down]) for r in C.itertuples()}
        scale = C[["sd_ml", "sd_ap", "sd_down"]].iloc[0].to_numpy(float)
    elif B.exec_xyz is not None and B.target_xyz is not None:
        Ex, Sx, frame = np.column_stack(B.exec_xyz), np.column_stack(B.target_xyz), "3d_mm"
        geo = {p: np.nanmedian(Sx[B.lick_pos == p], 0) for p in np.unique(B.lick_pos)}
    elif B.target_angle and B.target_dist and B.lick_reach is not None:
        Ex, frame = np.column_stack([B.lick_angle, B.lick_reach]), "cam4_angle_reach"
        geo = {p: np.array([B.target_angle[p], B.target_dist[p]]) for p in B.target_angle if p in B.target_dist}
    else:
        return (empty, empty.copy(), "none") + ((np.zeros((n, 0)), []) if weights else ())
    P = sorted(p for p, g in geo.items() if np.isfinite(g).all())
    if len(P) < 2:
        return (empty, empty.copy(), "none") + ((np.zeros((n, 0)), []) if weights else ())
    G = np.array([geo[p] for p in P])
    if scale is None:
        scale = np.ones(G.shape[1]) if frame == "3d_mm" else np.maximum(G.std(0), 1e-9)
    ok = usable & np.isfinite(Ex).all(1) & np.isin(B.lick_pos, P)
    d = (((Ex[ok] / scale)[:, None, :] - (G / scale)[None]) ** 2).sum(2)
    tgt, ex = empty.copy(), empty.copy()
    tgt[ok] = B.lick_pos[ok]
    ex[ok] = np.array(P, object)[d.argmin(1)]
    if weights:
        # SOFT bins (Priya 10-20: "licks may not land nearest their own spout"): each lick's weight on every landing
        # zone = Gaussian of its distance in SD units, rows sum to 1 -- no hard misassignment between overlapping zones
        Wt = np.zeros((n, len(P)))
        e = np.exp(-0.5 * (d - d.min(1, keepdims=True)) / bins_width() ** 2)
        Wt[ok] = e / e.sum(1, keepdims=True)
        return tgt, ex, frame, Wt, P
    return tgt, ex, frame


# =============================================================================== design


def _event_cols(times, ft, lags, amp=None, spacing=None) -> np.ndarray:
    from wfield_local import movement_encoding as ME
    r = ME.EventRegressor("e", np.asarray(times, float), lags, "g", amplitude=amp, basis_spacing_s=spacing or BASIS_S)
    return ME.build_design(ME.MovementInputs(ft, [r])).X.astype(np.float32)


def _truncate_after(X: np.ndarray, starts, stops, ft) -> np.ndarray:
    """Zero kernel rows after each event's own end (frames past the stop of the most recent event)."""
    starts, stops = np.asarray(starts, float), np.asarray(stops, float)
    o = np.argsort(starts)
    starts, stops = starts[o], stops[o]
    k = np.searchsorted(starts, ft, side="right") - 1
    past = (k >= 0) & (ft > np.where(k >= 0, stops[np.clip(k, 0, None)], np.inf))
    X = X.copy()
    X[past] = 0.0
    return X


def _state_cols(starts, stops, ft) -> np.ndarray:
    col = np.zeros(len(ft), np.float32)
    for a, b in zip(starts, stops):
        if np.isfinite(a) and np.isfinite(b) and b > a:
            col[(ft >= a) & (ft < b)] = 1.0
    return col[:, None]


def _analog_cols(t, v, ft) -> np.ndarray:
    from wfield_local import movement_encoding as ME
    base = ME.bin_to_frames(t, v, ft)
    fs = ME.frame_fs(ft)
    return np.column_stack([ME._shift(base, int(round(lag * fs))) for lag in CLAGS]).astype(np.float32)


def raw_blocks(B: Behaviour, ft, pos_perm: dict | None = None, drop_pos: str | None = None) -> dict:
    """{variable: (T, p) raw columns}. ``pos_perm`` {variable: permuted position per trial} for the label shuffle.
    ``drop_pos`` = p: the position blocks with spout p's OWN response removed (p constrained to the mean of the others)."""
    T = B.trials
    pos_order = sorted(T.pos.unique())
    out = {}
    # CUE vs POSITION (Priya 10-18: "cue and position are different. the cue is an audio cue that signals reward"):
    #   cue      ONE shared kernel, every trial, cue .. that trial's stop (auditory cue / reward anticipation)
    #   arrival  ONE shared spout-arrival kernel (strobe, 0 .. 2 s: the move itself, its sound / vibration) + ONE shared
    #            spout-present state (strobe .. cue)
    #   pos_*    per-position DEVIATIONS: position p's kernel minus p's share of the all-trial kernel (deviation coding),
    #            orthogonal to the shared kernels by construction -> "position" = only what differs between spouts.
    def enl_block(mask):
        return np.hstack([_event_cols(T.strobe_s[mask], ft, LAGS["strobe"]),
                          _state_cols(T.strobe_s[mask], T.cue_s[mask], ft)])

    def cue_block(mask):                            # cue .. this trial's stop (no overlap with the ENL state)
        # basis every 0.1 s for the first CUE_FINE_S (the fast cue / lick-onset dynamics), every CUE_COARSE_SPACING
        # after (GCaMP6s, slow) -- ~30 instead of 70 columns per kernel (2026-10-18 speed-up, same span)
        t_ = T.cue_s[mask]
        X = np.hstack([_event_cols(t_, ft, (LAGS["cue"][0], CUE_FINE_S)),
                       _event_cols(t_, ft, (CUE_FINE_S, LAGS["cue"][1]), spacing=CUE_COARSE_SPACING)])
        return _truncate_after(X, t_, T.stop_s[mask], ft)
    allm = np.ones(len(T), bool)
    shared = {"pos_enl": enl_block(allm), "pos_cue": cue_block(allm)}
    out["arrival"], out["cue"] = shared["pos_enl"], shared["pos_cue"]
    for var in POS_VARS:
        lab = T.pos.to_numpy() if not pos_perm or var not in pos_perm else pos_perm[var]
        cols = []
        if drop_pos is not None:
            # DROP-p (Priya 10-19, per-spout maps): the six deviation blocks sum to ZERO, so removing p's block alone loses
            # nothing (the other five rebuild it). Instead deviation-code the OTHER spouts among themselves --
            # D_q = X_q - m_q / (1 - m_p) * (shared - X_p), zero on p's trials -- so p's trials get only the shared
            # response = the trial-weighted mean of the others: exactly what is specific to p is removed, and no other
            # spout's trials are mislabelled (unlike the one-vs-rest relabelling).
            mp_ = lab == drop_pos
            Xex = enl_block(mp_) if var == "pos_enl" else cue_block(mp_)
            rest = shared[var] - Xex
            frac_rest = max(1.0 - mp_.mean(), 1e-9)
            for p in pos_order:
                if p == drop_pos:
                    continue
                m = lab == p
                Xp = enl_block(m) if var == "pos_enl" else cue_block(m)
                cols.append(Xp - (m.mean() / frac_rest) * rest)
            out[var] = np.hstack(cols)
            continue
        for p in pos_order:
            m = lab == p
            Xp = enl_block(m) if var == "pos_enl" else cue_block(m)
            cols.append(Xp - (m.mean() if len(m) else 0.0) * shared[var])
        out[var] = np.hstack(cols)
    lick_t = B.lick_on_s
    extra = getattr(B, "extra_lick_on_s", None)
    if extra is not None and len(extra):                 # LICKS_ENV: cam1-only events join the lick timing kernel
        lick_t = np.sort(np.concatenate([np.asarray(B.lick_on_s, float), extra]))
    ctx = getattr(B, "context_lick_on_s", None)
    mode_all = os.environ.get(LICKS_ALL_ENV, "")
    if mode_all and mode_all not in ("shared", "split"):
        raise ValueError(f"{LICKS_ALL_ENV}={mode_all!r}; expected shared | split")
    if ctx and mode_all == "shared":                     # one lick kernel wherever the lick happens
        lick_t = np.sort(np.concatenate([lick_t, ctx["pre_cue"], ctx["iti"]]))
    lk = [_event_cols(lick_t, ft, LAGS["lick"]), _event_cols(B.contact_s, ft, LAGS["lick"])]
    if ctx and mode_all == "split":                      # spontaneous (pre-cue) and ITI licks: their own kernels
        lk += [_event_cols(ctx["pre_cue"], ft, LAGS["lick"]), _event_cols(ctx["iti"], ft, LAGS["lick"])]
    lips = getattr(B, "lips_on_s", None)
    if lips is not None and len(lips):                   # LICKS_ENV=lips: lips-only licks get their own kernel
        lk.append(_event_cols(lips, ft, LAGS["lick"]))
    out["licks"] = np.hstack(lk)
    out["exec_angle"] = _event_cols(B.lick_on_s, ft, LAGS["lick"], amp=B.lick_angle_within)
    out["tongue"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("tongue_x", "tongue_y", "tongue_prot",
                                                                         "tongue_speed")])
    out["jaw"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("jaw_x", "jaw_y", "jaw_speed")])
    out["nose"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("nose_x", "nose_y")])
    out["running"] = _analog_cols(*B.analog["running"], ft)
    out["tongue_ps"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("tongue_prot", "tongue_speed")])
    out["jaw_nd"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("jaw_y", "jaw_speed")])
    out["nose_nd"] = _analog_cols(*B.analog["nose_y"], ft)
    out["dir_kin"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("tongue_x", "tongue_y", "jaw_x", "nose_x")])
    out["kinall"] = np.hstack([out["tongue"], out["jaw"]])                 # lick execution only -- no nose
    if B.lick_angle is not None and np.isfinite(B.lick_angle).any():
        la = B.lick_angle - np.nanmean(B.lick_angle)
        la2 = la ** 2 - np.nanmean(la ** 2)
        amps = [la, la2]
        if B.lick_reach is not None and np.isfinite(B.lick_reach).any():
            amps.append(B.lick_reach - np.nanmean(B.lick_reach))      # DISTANCE (Priya 10-18): executed reach
        if getattr(B, "exec_amps", None) is not None:                  # EXEC_ENV 3-D set (same licks, centred)
            amps = [x - np.nanmean(x) for x in B.exec_amps]
        out["lick_angle_abs"] = np.hstack([_event_cols(B.lick_on_s, ft, LAGS["lick"], amp=x) for x in amps])
        if B.target_angle:
            ta = np.array([B.target_angle.get(p, np.nan) for p in B.lick_pos], float)
            ta = np.where(np.isfinite(B.lick_angle), ta, np.nan)          # exactly the licks lick_angle_abs uses
            if np.isfinite(ta).any():
                tc = ta - np.nanmean(ta)
                tamps = [tc, tc ** 2 - np.nanmean(tc ** 2)]
                if B.target_dist:                                      # target DISTANCE: the spout's
                    td = np.array([B.target_dist.get(p, np.nan) for p in B.lick_pos], float)
                    td = np.where(np.isfinite(ta), td, np.nan)
                    if np.isfinite(td).any():
                        tamps.append(td - np.nanmean(td))
                if getattr(B, "target_amps", None) is not None:        # EXEC_ENV xyz2: spout 3-D ml / ap / dv
                    tx = [np.where(np.isfinite(ta), x, np.nan) for x in B.target_amps]
                    tamps = [x - np.nanmean(x) for x in tx]
                out["lick_angle_tgt"] = np.hstack([_event_cols(B.lick_on_s, ft, LAGS["lick"], amp=x) for x in tamps])
    if os.environ.get(BINS_ENV):                     # binned contest: six free lick kernels per side (see BINS_ENV)
        tl, el, frame_, Wt, P_ = exec_bins(B, weights=True)
        use = tl != ""
        if use.any():
            allk = _event_cols(B.lick_on_s[use], ft, LAGS["lick"])
            for name, lab in (("lick_bin_tgt", tl), ("lick_bin_exec", el)):
                if name == "lick_bin_exec" and frame_ == "3d_calibrated":
                    # SOFT executed bins: lick kernel x each lick's weight on landing zone p. The event builder CENTRES
                    # the modulator, so each column is already (weighted - mean-weight x all-lick) = deviation-coded;
                    # weights sum to 1 over p, so the six blocks sum to zero
                    out[name] = np.hstack([_event_cols(B.lick_on_s[use], ft, LAGS["lick"],
                                                       amp=Wt[use, P_.index(p)] if p in P_ else np.zeros(use.sum()))
                                           for p in pos_order])
                    continue
                out[name] = np.hstack([_event_cols(B.lick_on_s[lab == p], ft, LAGS["lick"])
                                       - (np.sum(lab == p) / use.sum()) * allk for p in pos_order])
    if B.target_angle:                              # trial-wise target angle (centred), its square, target distance
        a = np.array([B.target_angle.get(p, np.nan) for p in T.pos], float)
        ok = np.isfinite(a)
        a = np.where(ok, a - np.nanmean(a), 0.0)
        a2 = np.where(ok, a ** 2 - np.mean(a[ok] ** 2), 0.0)
        tamps = [a, a2]
        if B.target_dist:                           # DISTANCE (Priya 10-18: angle AND distance both shape behaviour)
            dd = np.array([B.target_dist.get(p, np.nan) for p in T.pos], float)
            if np.isfinite(dd[ok]).all():
                tamps.append(np.where(ok, dd - np.mean(dd[ok]), 0.0))
        for name, blk in (("angle_cue", cue_block), ("angle_enl", enl_block)):
            cols = []
            for amp in tamps:
                X = np.zeros_like(shared["pos_cue" if name == "angle_cue" else "pos_enl"])
                for p in pos_order:
                    m = (T.pos.to_numpy() == p) & ok
                    if m.any():
                        X += float(amp[m][0]) * blk(m)
                cols.append(X)
            out[name] = np.hstack(cols)
    return out


def _z(X, rows):
    """z-score with fit-row statistics; NaN -> 0 (unknown contributes nothing); constant columns -> 0."""
    Xr = X[rows]
    mu = np.nanmean(Xr, 0)
    sd = np.nanstd(Xr, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, np.inf)
    Z = (X - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)


def _slow_basis(t: np.ndarray, min_period_s: float) -> np.ndarray:
    """Cosine (DCT-II style) regressors over [t0, t1] with period >= min_period_s, incl. the constant."""
    t = np.asarray(t, float)
    T = max(t[-1] - t[0], 1e-9)
    n = int(np.floor(2 * T / min_period_s))
    u = (t - t[0]) / T
    return np.column_stack([np.cos(np.pi * k * u) for k in range(n + 1)])


def _residualise(X, E, rows):
    """X minus its least-squares projection on E (fit rows), re-z-scored (Musall's QR orthogonalisation)."""
    if E is None or E.shape[1] == 0:
        return X
    Er, Xr = E[rows].astype(np.float64), X[rows].astype(np.float64)
    G = Er.T @ Er + 1e-6 * np.eye(E.shape[1])
    beta = np.linalg.solve(G, Er.T @ Xr)
    return _z((X - E @ beta.astype(np.float32)), rows)


# =============================================================================== Gram engine


class Engine:
    """Per-fold / per-window sufficient statistics of a column universe A (rows = fit frames) and Y."""

    def __init__(self, A: np.ndarray, Y: np.ndarray, fold: np.ndarray, win: dict, chunk: int = 20000):
        self.P, self.K = A.shape[1] + 1, Y.shape[1]
        nf = int(fold.max()) + 1
        self.G = np.zeros((nf, self.P, self.P))
        self.b = np.zeros((nf, self.P, self.K))
        self.Gw = {w: np.zeros((nf, self.P, self.P)) for w in win}
        self.bw = {w: np.zeros((nf, self.P, self.K)) for w in win}
        self.YYw = {w: np.zeros((nf, self.K, self.K)) for w in win}
        self.sYw = {w: np.zeros((nf, self.K)) for w in win}
        self.sZw = {w: np.zeros((nf, self.P)) for w in win}
        self.nw = {w: np.zeros(nf) for w in win}
        for s in range(0, len(Y), chunk):
            a = np.hstack([np.ones((min(chunk, len(Y) - s), 1), np.float32), A[s:s + chunk]])
            y, f = Y[s:s + chunk], fold[s:s + chunk]
            for k in np.unique(f):
                m = f == k
                ak, yk = a[m], y[m]
                self.G[k] += (ak.T @ ak).astype(np.float64)
                self.b[k] += (ak.T @ yk).astype(np.float64)
                for w, wm in win.items():
                    mm = wm[s:s + chunk][m]
                    if mm.any():
                        aw, yw = ak[mm], yk[mm]
                        self.Gw[w][k] += (aw.T @ aw).astype(np.float64)
                        self.bw[w][k] += (aw.T @ yw).astype(np.float64)
                        self.YYw[w][k] += (yw.T @ yw).astype(np.float64)
                        self.sYw[w][k] += yw.sum(0)
                        self.sZw[w][k] += aw.sum(0)
                        self.nw[w][k] += mm.sum()
        self.Gt, self.bt = self.G.sum(0), self.b.sum(0)

    def betas(self, idx, pen):
        """Per-fold ridge solutions for universe columns ``idx`` (intercept added, unpenalised); ``pen`` per idx."""
        ii = np.r_[0, np.asarray(idx) + 1]
        lam = np.diag(np.r_[0.0, pen])
        out = []
        for k in range(self.G.shape[0]):
            Gtr = self.Gt[np.ix_(ii, ii)] - self.G[k][np.ix_(ii, ii)]
            btr = self.bt[ii] - self.b[k][ii]
            out.append(np.linalg.solve(Gtr + lam + 1e-8 * np.eye(len(ii)), btr))
        return ii, out

    def err_cov(self, idx, pen, w, fitted=None):
        """Summed held-out residual covariance (K, K) in window w, and the window's total covariance."""
        ii, B = fitted if fitted is not None else self.betas(idx, pen)
        E = np.zeros((self.K, self.K))
        se = np.zeros(self.K)
        for k, bk in enumerate(B):
            G, b = self.Gw[w][k][np.ix_(ii, ii)], self.bw[w][k][ii]
            E += self.YYw[w][k] - bk.T @ b - b.T @ bk + bk.T @ G @ bk
            se += self.sYw[w][k] - bk.T @ self.sZw[w][k][ii]
        n = self.nw[w].sum()
        mu = self.sYw[w].sum(0) / max(n, 1)
        Tc = self.YYw[w].sum(0) - n * np.outer(mu, mu)
        # WINDOW OFFSET FREE (2026-10-18): every model is fitted to whole trials but scored inside a window; without
        # this, a model that cannot track the window's MEAN level (running alone in the cue window) scored R^2 -0.6
        # and shuffled-model differences exceeded the full model's R^2. One common offset per component per window.
        E = E - np.outer(se, se) / max(n, 1)
        return E, Tc

    def full_beta(self, idx, pen):
        ii = np.r_[0, np.asarray(idx) + 1]
        lam = np.diag(np.r_[0.0, pen])
        return ii, np.linalg.solve(self.Gt[np.ix_(ii, ii)] + lam + 1e-8 * np.eye(len(ii)), self.bt[ii])


def _diag_quad(U, C, chunk=50000):
    """diag(U C U') for U (P, K)."""
    out = np.empty(len(U))
    for s in range(0, len(U), chunk):
        u = U[s:s + chunk].astype(np.float64)
        out[s:s + chunk] = np.einsum("ij,ij->i", u @ C, u)
    return out


# =============================================================================== one session


def group_bundled_fr(X: np.ndarray, fr_set: str, grouped: bool = False) -> np.ndarray:
    """Bundled RAW FaceRhythm factors (frames x k) -> the COMBINED_ENCODING_FR_GROUPS grouping (same as `fr_signals`
    applies when reading from the share); unchanged when the env is unset or the bundle stored grouped columns (older
    bundles, `grouped` missing = True)."""
    from wfield_local import fr_groups
    gs = fr_groups.spec()
    if not gs or grouped:
        return X
    sig = fr_groups.group({f"fr_{j}": (None, X[:, j]) for j in range(X.shape[1])}, *gs,
                          set_name=Path(fr_set).name)
    return np.column_stack([v for _t, v in sig.values()])


def session_inputs(label: str, cams=None, fr: str | None = None) -> dict:
    """Everything `run_session` reads from the share, for one session (O2 bundles, `scripts/o2_encoding.py`):
    imaging (load_imaging tuple), behaviour (Behaviour), epoch / day, and -- if `cams` -- the video motion-energy
    matrix (blanked outside strobe .. stop, as `video_signals`) with its column names, or None if absent; the same for
    the FaceRhythm factor time courses of set ``fr`` (default: FR_ENV)."""
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = config.animal_of(label), "2026" + label.split("_")[-1]
    imaging = load_imaging(label)
    out = {"label": label, "imaging": imaging, "behaviour": load_behaviour(animal, date, rv),
           "epoch": epochs.epoch_of(label), "day": epochs.days_since_stroke(label), "video": None, "fr": None}
    fr = fr or os.environ.get(FR_ENV)
    if fr:
        from wfield_local import face_rhythm_outputs as fro
        from wfield_local import fr_groups
        held = os.environ.pop(fr_groups.ENV, None)     # bundle the RAW factors; FR_GROUPS is applied at design time
        try:
            sig = fro.fr_signals(animal, date, rv, imaging[1], fdir=fro.factor_dir(rv, set_name=fr),
                                 absolute=_fr_absolute())
        except FileNotFoundError:
            sig = {}
        finally:
            if held is not None:
                os.environ[fr_groups.ENV] = held
        if sig:
            out["fr"] = {"set": fr, "absolute": _fr_absolute(), "names": list(sig), "grouped": False,
                         "X": np.column_stack([sig[k_][1] for k_ in sig]).astype(np.float32)}
        kp = int(os.environ.get(FR_POINTS_ENV) or 0)
        if kp:
            try:
                sp = fro.fr_signals(animal, date, rv, imaging[1], fdir=fro.factor_dir(rv, set_name=fr),
                                    kind="points", k=kp)
                out["frpts"] = {"set": fr, "k": kp, "names": list(sp),
                                "X": np.column_stack([sp[k_][1] for k_ in sp]).astype(np.float32)}
            except FileNotFoundError:
                pass
    if cams:
        from scripts.movement_encoding_session import video_signals
        try:
            sig = video_signals(animal, date, rv, imaging[1], list(cams), k=VIDEO_K)
        except FileNotFoundError:
            sig = {}
        if sig:
            names_ = sorted(sig)
            out["video"] = {"cams": list(cams), "names": names_,
                            "X": np.column_stack([sig[k_][1] for k_ in names_]).astype(np.float32)}
    return out


def run_session(label: str, order: str = "movement_first", seed: int = 0, inputs: dict | None = None) -> dict:
    """``inputs`` (from `session_inputs`) = run without touching the share (O2); None = load everything here."""
    from wfield_local import config
    animal = config.animal_of(label)
    date = "2026" + label.split("_")[-1]
    rng = np.random.default_rng(seed)
    if inputs is None:
        from wfield_local import epochs
        from wfield_local.paths import PathResolver
        rv = PathResolver()
        V, ft, Ub, flat, parea, names, shape, pstat = load_imaging(label)
        B = load_behaviour(animal, date, rv)
        epoch_, day_ = epochs.epoch_of(label), epochs.days_since_stroke(label)
    else:
        rv = None
        V, ft, Ub, flat, parea, names, shape, pstat = inputs["imaging"]
        B = inputs["behaviour"]
        epoch_, day_ = inputs["epoch"], inputs["day"]
    T = B.trials
    if os.environ.get(MATCH_ENV):
        T = _match_trials(T, B, label, os.environ[MATCH_ENV], seed)
    E = T[T.engaged & np.isfinite(T.cue_s)].reset_index(drop=True)
    st = np.where(np.isfinite(E.strobe_s), E.strobe_s, E.cue_s - 3.0)
    rows = np.zeros(len(ft), bool)
    tr_of = np.full(len(ft), -1)
    for k, (a, b_) in enumerate(zip(st - 0.5, E.stop_s.to_numpy(float) + 1.0)):
        m = (ft >= a) & (ft <= b_)
        rows |= m
        tr_of[m] = k
    ridx = np.flatnonzero(rows)
    # folds = contiguous trial blocks
    blk = np.array_split(np.arange(len(E)), FOLDS)
    f_of_trial = np.zeros(len(E), int)
    for k, bl in enumerate(blk):
        f_of_trial[bl] = k
    fold = f_of_trial[tr_of[ridx]]
    # windows
    ftr = ft[ridx]
    win = {"all": np.ones(len(ridx), bool), "enl": np.zeros(len(ridx), bool), "cue": np.zeros(len(ridx), bool),
           "lick": np.zeros(len(ridx), bool)}
    # extra scoring windows (Priya 10-20: acute far-contra licking often starts > 1.5 s after the cue, so a fixed early
    # cue window mixes waiting and licking): cue2lick = cue .. the trial's FIRST lick (cam4 + cam1-only onsets when
    # loaded), capped at stop = the waiting / initiation phase (length varies: ~0.15 s prompt .. seconds delayed);
    # cuestop = cue .. stop (~3.5 s), everything after the cue; firstlick = first lick -0.1 .. +1.0 s. Scoring only --
    # fits are unchanged.
    win["cue2lick"] = np.zeros(len(ridx), bool)
    win["cuestop"] = np.zeros(len(ridx), bool)
    win["firstlick"] = np.zeros(len(ridx), bool)   # first post-cue lick - 0.1 .. + 1.0 s (Priya 10-20): same movement
                                                   # phase on prompt and delayed trials; trials with no lick: none
    ex_ = getattr(B, "extra_lick_on_s", None)
    lk_all = np.sort(np.concatenate([np.asarray(B.lick_on_s, float), np.asarray([] if ex_ is None else ex_, float)]))
    for s_, c, stp in zip(st, E.cue_s.to_numpy(float), E.stop_s.to_numpy(float)):
        win["enl"] |= (ftr >= max(s_, c - 2.0)) & (ftr < c)
        win["cue"] |= (ftr >= c) & (ftr < c + 1.0)
        stp = stp if np.isfinite(stp) else c + 3.5
        i = np.searchsorted(lk_all, c)
        first = lk_all[i] if i < len(lk_all) and lk_all[i] < stp else stp
        win["cue2lick"] |= (ftr >= c) & (ftr < first)
        win["cuestop"] |= (ftr >= c) & (ftr < stp)
        if first < stp:
            win["firstlick"] |= (ftr >= first - 0.1) & (ftr < first + 1.0)
    pk = np.sort(B.lick_peak_s)
    j = np.searchsorted(pk, ftr)
    d = np.minimum(np.abs(ftr - pk[np.clip(j - 1, 0, len(pk) - 1)]), np.abs(ftr - pk[np.clip(j, 0, len(pk) - 1)]))
    win["lick"] = d <= 0.2
    # raw blocks -> z -> orthogonalised (order) -> universe
    raw = {v: _z(X, ridx)[ridx] for v, X in raw_blocks(B, ft).items()}
    # POSITION IS ONE VARIABLE (Priya 10-18): the spout sits in one place strobe .. stop, so ONE relabelling is applied
    # to both periods (pos_enl / pos_cue are its pre- and post-cue regressors; `shuf:position` / `drop:position` test
    # it whole, the per-period models are the temporal breakdown).
    _p = rng.permutation(E.pos.to_numpy())
    perm = {v: _p for v in POS_VARS}
    full_perm = T.pos.to_numpy().copy()
    eng_idx = np.flatnonzero(T.engaged.to_numpy() & np.isfinite(T.cue_s.to_numpy()))
    rawp = {}
    for v in POS_VARS:                                      # label-permuted position blocks (engaged trials permuted)
        lab = full_perm.copy()
        lab[eng_idx] = perm[v]
        rawp[v] = _z(raw_blocks(B, ft, {v: lab})[v], ridx)[ridx]

    # CONSTRAINED permutations (Priya 10-18, "how would we isolate position-specific information"): labels permuted
    # only among trials at the same DISTANCE (removes SIDE, keeps distance) or on the same SIDE (removes DISTANCE) ->
    # dR2 maps of where side vs distance is coded, from the same fitted model, no new regressors.
    def _perm_within(labels, key):
        out = labels.copy()
        grp = np.array([key(x) for x in labels])
        for gv in np.unique(grp):
            ii = np.flatnonzero(grp == gv)
            out[ii] = labels[ii][rng.permutation(len(ii))]
        return out
    rawc = {}
    # ONE-VS-REST per position (Priya 10-18: "a position signal for EACH position"): position p's label is reassigned
    # to a random set of engaged trials of the same size (the rest keep their labels in proportion) -> what is specific
    # to p is removed; variance-based, so positions are not forced to trade off against each other.
    for pp in sorted(set(full_perm[eng_idx])):
        lab = full_perm.copy()                          # ONE relabelling for both periods
        sub = full_perm[eng_idx].copy()
        isp = sub == pp
        # MINIMAL SWAP (fix 2026-10-19): a random same-size set of trials gets label p; only the p-trials that lost p
        # take the displaced labels. The earlier version re-permuted EVERY other trial's label too, so each
        # "one-vs-rest" model destroyed nearly all position information and the six per-spout maps were identical.
        pick = np.zeros(len(sub), bool)
        pick[rng.permutation(len(sub))[: int(isp.sum())]] = True
        gain, lose = pick & ~isp, isp & ~pick                 # non-p trials that become p / p trials that lose p
        new = sub.copy()
        new[gain] = pp
        new[lose] = sub[gain][rng.permutation(int(gain.sum()))]
        lab[eng_idx] = new
        for v in POS_VARS:
            rawc[(f"shufpos-{pp}", v)] = _z(raw_blocks(B, ft, {v: lab})[v], ridx)[ridx]
    if any(o.startswith("droppos") for o in (os.environ.get(ONLY_ENV) or "").split(",")):   # built only when asked
        for pp in sorted(set(full_perm[eng_idx])):   # DROP-p: spout p's own response removed (raw_blocks drop_pos)
            Rb = raw_blocks(B, ft, drop_pos=pp)
            for v in POS_VARS:
                rawc[(f"droppos-{pp}", v)] = _z(Rb[v], ridx)[ridx]
    for kind, key in (("shufside", lambda x: x.split("_")[0]), ("shufdist", lambda x: x.split("_", 1)[1])):
        lab = full_perm.copy()                          # ONE relabelling for both periods
        lab[eng_idx] = _perm_within(full_perm[eng_idx], key)
        for v in POS_VARS:
            rawc[(kind, v)] = _z(raw_blocks(B, ft, {v: lab})[v], ridx)[ridx]
    seq = ORDERS[order]
    orth, earlier = {}, None
    for g in seq:
        vs = [v for v in GROUP_OF if GROUP_OF[v] == g]
        Xg = np.hstack([raw[v] for v in vs])
        Xo = (Xg if (earlier is None or g in ORTH_RAW.get(order, ()))
              else _residualise(Xg, earlier, slice(None)))
        c = 0
        for v in vs:
            orth[v] = Xo[:, c:c + raw[v].shape[1]]
            c += raw[v].shape[1]
        earlier = Xg if earlier is None else np.hstack([earlier, Xg])
    # VIDEO (optional, see VIDEO_ENV)
    vid_raw = vid_orth = None
    cams = [c for c in (os.environ.get(VIDEO_ENV) or "").split(",") if c]
    if cams and inputs is not None:
        vi = inputs.get("video")
        sig = ({n_: None for n_ in vi["names"]} if vi and set(cams) <= set(vi["cams"]) else {})
    elif cams:
        from scripts.movement_encoding_session import video_signals
        try:
            sig = video_signals(animal, date, rv, ft, cams, k=VIDEO_K)
        except FileNotFoundError:
            sig = {}
    else:
        sig = {}
    if sig:
        Xv = (inputs["video"]["X"].astype(np.float64) if inputs is not None
              else np.column_stack([sig[k_][1] for k_ in sorted(sig)]))
        vid_raw = _z(Xv, ridx)[ridx]                            # blanked (NaN) frames -> 0
        blank = ~np.isfinite(Xv[ridx]).any(1)
        vid_orth = _residualise(vid_raw, np.hstack([raw[v] for v in GROUP_OF]), slice(None))
        vid_orth[blank] = 0.0                                   # residualising must not put signal back into blanked frames
    # FR (optional, see FR_ENV): same placement as video
    fr_raw = fr_orth = None
    fr_set = os.environ.get(FR_ENV) or ""
    if fr_set:
        if inputs is not None:
            fi = inputs.get("fr")
            Xf = (fi["X"].astype(np.float64) if fi and fi["set"] == fr_set
                  and fi.get("absolute", False) == _fr_absolute() else None)
            if Xf is not None:
                Xf = group_bundled_fr(Xf, fr_set, grouped=fi.get("grouped", True))
        else:
            from wfield_local import face_rhythm_outputs as fro
            try:
                sig_fr = fro.fr_signals(animal, date, rv, ft, fdir=fro.factor_dir(rv, set_name=fr_set),
                                        absolute=_fr_absolute())
                Xf = np.column_stack([sig_fr[k_][1] for k_ in sig_fr])
            except FileNotFoundError:
                Xf = None
        if Xf is not None:
            fr_raw = _z(Xf, ridx)[ridx]                         # blanked (NaN) frames -> 0
            blank_fr = ~np.isfinite(Xf[ridx]).any(1)
            fr_orth = _residualise(fr_raw, np.hstack([raw[v] for v in GROUP_OF]), slice(None))
            fr_orth[blank_fr] = 0.0
    frpts_raw = None
    kp = int(os.environ.get(FR_POINTS_ENV) or 0)
    if fr_raw is not None and kp:
        if inputs is not None:
            pi_ = inputs.get("frpts")
            Xp = pi_["X"].astype(np.float64) if pi_ and pi_["set"] == fr_set and pi_["k"] == kp else None
        else:
            from wfield_local import face_rhythm_outputs as fro
            try:
                sp = fro.fr_signals(animal, date, rv, ft, fdir=fro.factor_dir(rv, set_name=fr_set), kind="points", k=kp)
                Xp = np.column_stack([sp[k_][1] for k_ in sp])
            except FileNotFoundError:
                Xp = None
        if Xp is not None:
            frpts_raw = _z(Xp, ridx)[ridx]
    # shuffled orthogonalised blocks
    shift_min = int(SHIFT_MIN_S / np.median(np.diff(ft)))
    shuf = {}
    for v in GROUP_OF:
        if v in POS_VARS:
            gi = seq.index("task")
            prior = ([] if "task" in ORTH_RAW.get(order, ())
                     else [w for w in GROUP_OF if seq.index(GROUP_OF[w]) < gi])
            Epr = np.hstack([raw[w] for w in prior]) if prior else None
            shuf[v] = _residualise(rawp[v], Epr, slice(None)) if Epr is not None else rawp[v]
        else:
            s_ = int(rng.integers(shift_min, max(shift_min + 1, len(ridx) - shift_min)))
            shuf[v] = np.roll(orth[v], s_, axis=0)
    # universe layout
    blocks, layout = [], {}

    def add(key, X):
        layout[key] = np.arange(sum(b.shape[1] for b in blocks), sum(b.shape[1] for b in blocks) + X.shape[1])
        blocks.append(X)
    for (kind, v), Xc in rawc.items():                    # constrained-permutation blocks, orthogonalised like shuf
        gi = seq.index("task")
        prior = ([] if "task" in ORTH_RAW.get(order, ())
                 else [w for w in GROUP_OF if seq.index(GROUP_OF[w]) < gi])
        Epr = np.hstack([raw[w] for w in prior]) if prior else None
        add((kind, v), _residualise(Xc, Epr, slice(None)) if Epr is not None else Xc)
    for v in GROUP_OF:
        add(("orth", v), orth[v])
        add(("shuf", v), shuf[v])
        if GROUP_OF[v] != seq[0]:                       # orthogonalised groups also keep their raw columns
            add(("raw", v), raw[v])
        else:
            layout[("raw", v)] = layout[("orth", v)]
    for v in EXTRA_GROUP:
        if v in raw:
            add(("raw", v), raw[v])
    if vid_raw is not None:
        add(("raw", "video"), vid_raw)
        add(("orth", "video"), vid_orth)
    if fr_raw is not None:
        add(("raw", "fr"), fr_raw)
        add(("orth", "fr"), fr_orth)
    if frpts_raw is not None:
        add(("raw", "frpts"), frpts_raw)
    A = np.hstack(blocks)
    Y = V[ridx]
    Q = np.linalg.qr(_slow_basis(ft[ridx], DRIFT_MIN_PERIOD_S))[0]
    Q = Q.astype(np.float32)
    Y = (Y - Q @ (Q.T @ Y)).astype(np.float32)
    for c0 in range(0, A.shape[1], 256):           # in place, in column chunks (A is GBs; no full-size temporary)
        sl = slice(c0, min(c0 + 256, A.shape[1]))
        A[:, sl] -= Q @ (Q.T @ A[:, sl])
    eng = Engine(A, Y, fold, win)
    grp_of_col = np.empty(A.shape[1], object)                  # CREDIT group (task / instructed / uninstructed)
    pen_of_col = np.empty(A.shape[1], object)                  # PENALTY group (PEN_GROUP_OF)
    for (_kind, v), ix in layout.items():
        grp_of_col[ix] = {**GROUP_OF, **EXTRA_GROUP, "video": "video", "fr": "fr", "frpts": "frpts"}[v]
        pen_of_col[ix] = PEN_GROUP_OF[v]

    def cols(spec):
        """spec: list of (kind, var) -> universe column indices."""
        return np.concatenate([layout[s] for s in spec])
    # penalties: per group, coordinate search on the full model's held-out R^2 (all window, summed components)
    full_spec = [("orth", v) for v in GROUP_OF]
    alphas = {g: 100.0 for g in PEN_GROUPS}

    def score(spec, al, w="all"):
        ix = cols(spec)
        pen = np.array([al[pen_of_col[i]] for i in ix])
        Ecov, Tc = eng.err_cov(ix, pen, w)
        return 1.0 - np.trace(Ecov) / np.trace(Tc)
    for _ in range(2):
        for g in PEN_GROUPS:
            best = max(ALPHA_GRID, key=lambda a: score(full_spec, {**alphas, g: a}))
            alphas[g] = best
    alphas["video"] = 100.0                                   # start value; every video model searches its own
    alphas["fr"] = alphas["frpts"] = 100.0                    # likewise for the FaceRhythm models

    # 2x2-pixel blocks for the stored maps (sums of SSE / SST per block -> exact block R^2)
    rr, cc = np.unravel_index(flat, shape)
    blk_id_full = (rr // 2) * ((shape[1] + 1) // 2) + (cc // 2)
    ublk, blk = np.unique(blk_id_full, return_inverse=True)

    def bsum(x):
        return np.bincount(blk, weights=x, minlength=len(ublk))
    # models
    models = {"full": full_spec}
    for v in GROUP_OF:
        models[f"alone:{v}"] = [("raw", v)]
        models[f"shuf:{v}"] = [("shuf", v) if w == v else ("orth", w) for w in GROUP_OF]
    for kind, v in rawc:
        models[f"{kind}:{v}"] = [(kind, v) if w == v else ("orth", w) for w in GROUP_OF]
    for kind in sorted({k for k, _v in rawc}):          # both periods relabelled together = position as one variable
        models[f"{kind}:position"] = [(kind, w) if w in POS_VARS else ("orth", w) for w in GROUP_OF]
    models["shuf:position"] = [("shuf", w) if w in POS_VARS else ("orth", w) for w in GROUP_OF]
    for g in GROUPS:
        vs = [v for v in GROUP_OF if GROUP_OF[v] == g]
        models[f"alone:{g}"] = [("raw", v) for v in vs]
        models[f"shuf:{g}"] = [("shuf", w) if GROUP_OF[w] == g else ("orth", w) for w in GROUP_OF]
    task_raw = [("raw", v) for v in GROUP_OF if GROUP_OF[v] == "task"]
    for v in GROUP_OF:
        if GROUP_OF[v] != "task":
            models[f"task+:{v}"] = task_raw + [("raw", v)]
    for g in ("instructed", "uninstructed"):
        models[f"task+:{g}"] = task_raw + [("raw", v) for v in GROUP_OF if GROUP_OF[v] == g]
    # DROP (2026-10-18): the full model refit WITHOUT a variable / group, UNORTHOGONALISED (all other columns raw) --
    # dR2_drop = full - drop credits shared variance to NOBODY (the conservative unique contribution, order-free);
    # the shuffle dR2 credits it to the earlier group of the orthogonalisation order.
    for v in GROUP_OF:
        models[f"drop:{v}"] = [("raw", w) for w in GROUP_OF if w != v]
    for g in GROUPS:
        models[f"drop:{g}"] = [("raw", w) for w in GROUP_OF if GROUP_OF[w] != g]
    models["drop:position"] = [("raw", w) for w in GROUP_OF if w not in POS_VARS]
    # VARIANTS (order-free, raw columns like the drop models)
    models["raw:full"] = [("raw", w) for w in GROUP_OF]
    if ("raw", "angle_cue") in layout:
        for pv, av in (("pos_cue", "angle_cue"), ("pos_enl", "angle_enl")):
            models[f"angle:{pv}"] = [("raw", w) for w in GROUP_OF if w != pv] + [("raw", av)]
    notxy = [("raw", w) for w in GROUP_OF if w != "tongue"] + [("raw", "tongue_ps")]
    models["notxy:full"] = notxy
    for pv in POS_VARS:
        models[f"notxy:drop:{pv}"] = [x for x in notxy if x != ("raw", pv)]
    models["notxy:drop:position"] = [x for x in notxy if x[1] not in POS_VARS]
    if ("raw", "angle_cue") in layout:
        models["angle:position"] = ([("raw", w) for w in GROUP_OF if w not in POS_VARS]
                                    + [("raw", "angle_cue"), ("raw", "angle_enl")])
    if ("raw", "lick_angle_abs") in layout:               # position replaced by WHERE EACH LICK WENT (executed)
        models["lickangle:position"] = ([("raw", w) for w in GROUP_OF if w not in POS_VARS and w != "exec_angle"]
                                        + [("raw", "lick_angle_abs")])
    if ("raw", "lick_angle_tgt") in layout:               # ... or by WHERE THAT LICK'S SPOUT WAS (target), same kernel
        models["licktarget:position"] = ([("raw", w) for w in GROUP_OF if w not in POS_VARS and w != "exec_angle"]
                                         + [("raw", "lick_angle_tgt")])
    # TARGET vs EXECUTED inside the tongue-without-x/y model (Priya 10-18: "most invested in the target vs executed angle
    # question"): tongue x/y is itself a direction signal and would absorb directional variance from both competitors
    if ("raw", "angle_cue") in layout:
        models["notxy:angle:position"] = ([x for x in notxy if x[1] not in POS_VARS]
                                          + [("raw", "angle_cue"), ("raw", "angle_enl")])
    if ("raw", "lick_angle_abs") in layout:
        models["notxy:lickangle:position"] = ([x for x in notxy if x[1] not in POS_VARS and x[1] != "exec_angle"]
                                              + [("raw", "lick_angle_abs")])
    if ("raw", "lick_angle_tgt") in layout:
        models["notxy:licktarget:position"] = ([x for x in notxy if x[1] not in POS_VARS and x[1] != "exec_angle"]
                                               + [("raw", "lick_angle_tgt")])
    # BOTH on the lick kernel (Priya 10-18): unique target = both - lickangle, unique executed = both - licktarget
    # (target and executed angles correlate ~0.8, so their separate gains over no-position overlap)
    if ("raw", "lick_angle_tgt") in layout and ("raw", "lick_angle_abs") in layout:
        both = [("raw", "lick_angle_tgt"), ("raw", "lick_angle_abs")]
        models["lickboth:position"] = [("raw", w) for w in GROUP_OF if w not in POS_VARS and w != "exec_angle"] + both
        models["notxy:lickboth:position"] = [x for x in notxy if x[1] not in POS_VARS and x[1] != "exec_angle"] + both

    # PER-MODEL PENALTIES (2026-10-18 fix): every reduced model gets its own ridge penalty per group present (one
    # coordinate pass over ALPHA_GRID from the full model's values, same held-out criterion). Borrowing the full
    # model's penalties made single-variable models over- / under-fit (the full model's instructed penalty ranged 10 ..
    # 1000 by session), so "alone" went NEGATIVE for jaw / instructed and was not the upper bound it is meant to be.
    if vid_raw is not None:
        rawfull = [("raw", w) for w in GROUP_OF]
        models["video:full"] = full_spec + [("orth", "video")]
        models["video:alone"] = [("raw", "video")]
        models["video:raw:full"] = rawfull + [("raw", "video")]
        for g in (*GROUPS, "position"):
            drop = POS_VARS if g == "position" else [w for w in GROUP_OF if GROUP_OF[w] == g]
            models[f"video:drop:{g}"] = [x for x in rawfull if x[1] not in drop] + [("raw", "video")]
        for name in ("angle:position", "lickangle:position", "licktarget:position", "lickboth:position"):
            if name in models:
                models[f"video:{name}"] = models[name] + [("raw", "video")]
    if fr_raw is not None:
        rawfull = [("raw", w) for w in GROUP_OF]
        models["fr:full"] = full_spec + [("orth", "fr")]
        models["fr:alone"] = [("raw", "fr")]
        models["fr:raw:full"] = rawfull + [("raw", "fr")]
        for g in (*GROUPS, "position"):
            drop = POS_VARS if g == "position" else [w for w in GROUP_OF if GROUP_OF[w] == g]
            models[f"fr:drop:{g}"] = [x for x in rawfull if x[1] not in drop] + [("raw", "fr")]
        if vid_raw is not None:
            models["frvideo:raw:full"] = rawfull + [("raw", "fr"), ("raw", "video")]
            models["frvideo:alone"] = [("raw", "fr"), ("raw", "video")]
        if frpts_raw is not None:
            models["frpts:alone"] = [("raw", "frpts")]
            models["frpts:raw:full"] = rawfull + [("raw", "frpts")]
            models["frboth:raw:full"] = rawfull + [("raw", "fr"), ("raw", "frpts")]
    if all(("raw", v) in layout for v in ("jaw_nd", "nose_nd", "dir_kin")):
        nb = [("raw", v) for v in NODIR_BASE]
        tg, ex, kin = [("raw", "lick_angle_tgt")], [("raw", "lick_angle_abs")], [("raw", "dir_kin")]
        cat = [("raw", v) for v in POS_VARS]
        models["nodir:none"] = nb
        models["nodir:category"] = nb + cat
        if ("raw", "lick_angle_tgt") in layout and ("raw", "lick_angle_abs") in layout:
            models["nodir:target"] = nb + tg
            models["nodir:exec"] = nb + ex
            models["nodir:target+exec"] = nb + tg + ex
            models["nodir:kin+target"] = nb + kin + tg
            models["nodir:kin+exec"] = nb + kin + ex
            models["nodir:kin+target+exec"] = nb + kin + tg + ex
        models["nodir:kin"] = nb + kin
    if ("raw", "kinall") in layout:
        kb = [("raw", v) for v in NOKIN_BASE]
        ka, cat = [("raw", "kinall")], [("raw", v) for v in POS_VARS]
        models["nokin:none"] = kb
        models["nokin:category"] = kb + cat
        models["nokin:kin"] = kb + ka
        models["nokin:kin+category"] = kb + ka + cat
        if ("raw", "lick_angle_tgt") in layout and ("raw", "lick_angle_abs") in layout:
            tg, ex = [("raw", "lick_angle_tgt")], [("raw", "lick_angle_abs")]
            models["nokin:target"] = kb + tg
            models["nokin:exec"] = kb + ex
            models["nokin:target+exec"] = kb + tg + ex
            # STRICT executed (Priya 10-20: "execution coding once we account for the spout position"): executed beyond
            # ANY spout-specific pattern (all six position kernels) = lick-to-lick execution within a spout
            models["nokin:category+exec"] = kb + cat + ex
            models["nokin:kin+target"] = kb + ka + tg
            models["nokin:kin+exec"] = kb + ka + ex
            models["nokin:kin+target+exec"] = kb + ka + tg + ex
            nz = [("raw", "nose")]
            models["nokin:kin+nose"] = kb + ka + nz
            models["nokin:kin+nose+target"] = kb + ka + nz + tg
        if ("raw", "lick_bin_tgt") in layout and ("raw", "lick_bin_exec") in layout:      # BINS_ENV
            tb, eb = [("raw", "lick_bin_tgt")], [("raw", "lick_bin_exec")]
            models["nokin:tgtbin"] = kb + tb
            models["nokin:exbin"] = kb + eb
            models["nokin:tgtbin+exbin"] = kb + tb + eb
            models["nokin:category+exbin"] = kb + cat + eb
    only = [x for x in (os.environ.get(ONLY_ENV) or "").split(",") if x]
    if only == ["trmodels"]:                         # exactly the time-resolved model set
        models = {k: v for k, v in models.items() if k in TR_MODELS}
    elif only:                                       # fit a named SUBSET only (own output folder, see out_dir)
        models = {k: v for k, v in models.items() if k == "full" or any(k.startswith(x) for x in only)}
    if os.environ.get(SMOKE_ENV):                   # end-to-end check of the post-fit code without the full model set
        keep = {"full", *TR_MODELS, "raw:full", "angle:position", "lickangle:position", "notxy:full",
                "notxy:drop:position", "notxy:angle:position", "notxy:lickangle:position", "licktarget:position",
                "notxy:licktarget:position", "lickboth:position", "notxy:lickboth:position",
                *(k for k in models if k.startswith(("video:", "fr:", "frvideo:", "frpts:", "frboth:")))}
        models = {k: v for k, v in models.items() if k in keep}
    model_alphas = {"full": dict(alphas)}
    for name, spec in models.items():
        if name == "full":
            continue
        if name.split(":")[0].startswith("shuf"):
            # a SHUFFLE keeps the full model's structure and penalties (the comparison is "same model, the variable's
            # information destroyed"); searching here only cost time (2026-10-18 speed-up)
            model_alphas[name] = dict(alphas)
            continue
        al = dict(alphas)
        present = sorted({pen_of_col[i] for i in cols(spec)}, key=(*PEN_GROUPS, "video", "fr", "frpts").index)
        for g in present:
            al[g] = max(ALPHA_GRID, key=lambda a: score(spec, {**al, g: a}))
        model_alphas[name] = al
    # evaluate: per model x window -> pixel SSE (as e), shared total t per window
    areas = sorted({int(a) for a in np.unique(parea[pstat]) if a != 0})
    res = {"label": label, "animal": animal, "date": date, "epoch": epoch_,
           "day": day_, "order": order, "alphas": alphas, "penalty_groups": list(PEN_GROUPS), "gate": B.gate,
           "exec_set": os.environ.get(EXEC_ENV) or "cam4", "lick_set": os.environ.get(LICKS_ENV) or "cam4",
           "n_trials": int(len(E)), "n_frames": int(len(ridx)),
           "window_frames": {w_: int(m_.sum()) for w_, m_ in win.items()}, "fs": float(1 / np.median(np.diff(ft))),
           "video": (cams if vid_raw is not None else None), "video_k": VIDEO_K,
           "fr": (fr_set if fr_raw is not None else None),
           "fr_measure": ("absolute" if _fr_absolute() else "share") if fr_raw is not None else None,
           "fr_k": (int(fr_raw.shape[1]) if fr_raw is not None else 0),
           "frpts_k": (int(frpts_raw.shape[1]) if frpts_raw is not None else 0),
           "drift_min_period_s": DRIFT_MIN_PERIOD_S, "drift_n_basis": int(Q.shape[1])}
    if os.environ.get(BINS_ENV):                     # how often the tongue went to its own spout, per spout
        tl_, el_, fr_, Wt_, P_ = exec_bins(B, weights=True)
        u_ = tl_ != ""
        res["bins"] = {"frame": fr_, "agree": float(np.mean(tl_[u_] == el_[u_])) if u_.any() else float("nan"),
                       "confusion": pd.crosstab(pd.Series(tl_[u_], name="target"),
                                                pd.Series(el_[u_], name="exec")).to_dict(),
                       # descriptive: mean soft weight on each landing zone, per target spout (where licks went)
                       "soft_weight": ({t: dict(zip(P_, Wt_[u_ & (tl_ == t)].mean(0).round(3).tolist()))
                                        for t in sorted(set(tl_[u_]))} if len(P_) else {})}
    maps, region = {}, []
    tot_pix = {}
    amask = {a: (parea == a) & pstat for a in areas}          # region values on the eroded stat mask
    for name, spec in models.items():
        ix = cols(spec)
        pen = np.array([model_alphas[name][pen_of_col[i]] for i in ix])
        fitted = eng.betas(ix, pen)
        for w in win:                                     # WINDOWS + cue2lick / cuestop
            Ecov, Tc = eng.err_cov(ix, pen, w, fitted)
            e = _diag_quad(Ub, Ecov)
            if ("TOTAL", w) not in maps:
                t = _diag_quad(Ub, Tc)
                maps[("TOTAL", w)] = bsum(t)
                tot_pix[w] = t
            t = tot_pix[w]
            maps[(name, w)] = bsum(e)
            region.append({"model": name, "window": w, "area": "cortex", "r2": 1 - e[pstat].sum() / t[pstat].sum()})
            for a, m in amask.items():
                region.append({"model": name, "window": w, "area": names.get(a, str(a)),
                               "r2": 1 - e[m].sum() / t[m].sum()})
    # collinearity: cumulative subspace angles of the full design (normalised columns)
    ixf = cols(full_spec) + 1
    Gn = eng.Gt[np.ix_(ixf, ixf)]
    dsc = np.sqrt(np.clip(np.diag(Gn), 1e-12, None))
    Gc = Gn / np.outer(dsc, dsc)
    csa = np.full(len(ixf), np.nan)
    for jit in (1e-6, 1e-4, 1e-2):                  # exactly collinear columns need more jitter (1e-10 / 1e-6 gave NaN)
        try:
            csa = np.abs(np.diag(np.linalg.cholesky(Gc + jit * np.eye(len(ixf)))))
            res["collinearity_jitter"] = jit
            break
        except np.linalg.LinAlgError:
            continue
    res["collinearity_min"] = {v: float(np.nanmin(csa[np.isin(ixf - 1, layout[("orth", v)])])) for v in GROUP_OF}
    # Fig. 5: full model fitted ONCE, prediction split by group; cue-aligned per position, per area (+ cortex)
    ix = cols(full_spec)
    pen = np.array([alphas[pen_of_col[i]] for i in ix])
    ii, beta = eng.full_beta(ix, pen)
    Afull = A[:, ii[1:] - 1]
    gcol = grp_of_col[ix]
    Ra = np.stack([Ub[(parea == a) & pstat].mean(0) for a in areas] + [Ub[pstat].mean(0)], 1)   # stat mask
    comp = {"data": Y.astype(np.float64), "full": Afull @ beta[1:] + beta[0]}
    for g in GROUPS:
        comp[g] = Afull[:, gcol == g] @ beta[1:][gcol == g]
    vcol = np.empty(len(ix), object)                      # variable of each full-model column
    for (kind_, v_), ixs in layout.items():
        if kind_ == "orth":
            vcol[np.isin(ix, ixs)] = v_
    pm = np.isin(vcol, list(POS_VARS))
    comp_pos = Afull[:, pm] @ beta[1:][pm]                 # position-specific deviations only (not in the PETH parts)
    off = np.arange(int(PETH_S[0] * res["fs"]), int(PETH_S[1] * res["fs"]) + 1)
    cue_f = np.searchsorted(ftr, E.cue_s.to_numpy(float))
    pos_order = sorted(E.pos.unique())
    # PER-POSITION RESPONSE MAPS (Priya 10-18): the full model's TASK-component prediction per position, ABSOLUTE (not
    # minus the mean over positions: a mean reference couples the six -- "the reference is the claim", 09-12), averaged
    # over cue-aligned bins and over the no-lick-agnostic ENL state (strobe + 0.3 s .. cue); block maps on the grid.
    comp["pos"] = comp_pos
    st_f = np.searchsorted(ftr, np.where(np.isfinite(E.strobe_s), E.strobe_s, E.cue_s - 3.0).astype(float) + 0.3)
    for p in pos_order:
        sel = np.flatnonzero((E.pos == p).to_numpy())
        for lab, (a0, a1) in RESP_BINS.items():
            idx = np.concatenate([np.arange(cue_f[i] + int(a0 * res["fs"]), cue_f[i] + int(a1 * res["fs"]))
                                  for i in sel]) if len(sel) else np.array([], int)
            idx = idx[(idx >= 0) & (idx < len(Y))]
            if len(idx):
                maps[(f"resp:{p}", lab)] = bsum(Ub @ comp["task"][idx].mean(0)) / np.maximum(bsum(np.ones(len(Ub))), 1)
                maps[(f"respdev:{p}", lab)] = bsum(Ub @ comp["pos"][idx].mean(0)) / np.maximum(bsum(np.ones(len(Ub))), 1)
        idx = np.concatenate([np.arange(st_f[i], cue_f[i]) for i in sel if cue_f[i] > st_f[i]]) if len(sel) else []
        idx = np.asarray(idx, int)
        idx = idx[(idx >= 0) & (idx < len(Y))]
        if len(idx):
            maps[(f"resp:{p}", "enl")] = bsum(Ub @ comp["task"][idx].mean(0)) / np.maximum(bsum(np.ones(len(Ub))), 1)
            maps[(f"respdev:{p}", "enl")] = bsum(Ub @ comp["pos"][idx].mean(0)) / np.maximum(bsum(np.ones(len(Ub))), 1)
    peth = np.full((len(comp), len(pos_order), Ra.shape[1], len(off)), np.nan, np.float32)
    for ci, Cm in enumerate(comp.values()):
        R = (Cm @ Ra).astype(np.float32)
        for pi, p in enumerate(pos_order):
            idx = cue_f[(E.pos == p).to_numpy()][:, None] + off[None, :]
            ok = (idx >= 0) & (idx < len(R))
            seg = np.where(ok[..., None], R[np.clip(idx, 0, len(R) - 1)], np.nan)
            peth[ci, pi] = np.nanmean(seg, 0).T
    # time-resolved held-out SSE (cue-aligned, all trials) per group model, cortex + areas
    # + the fair target-vs-executed models when fitted (Priya 10-19: the FULL executed set -- angle, angle^2, reach -- on
    # TRIAL-TO-TRIAL variance: within a position the target is constant, so only an executed code can explain it)
    tr_models = [m for m in (*TR_MODELS, *NOKIN_TR) if m in models]
    Rd = Y.astype(np.float64) @ Ra
    # TWO ALIGNMENTS (Priya 10-20: "do all trial to trial and lick variability analyses aligned to lick"): cue (as
    # before) and the trial's FIRST post-cue lick (cam4 + cam1-only onsets when loaded; trials with no lick before stop
    # are left out of the lick alignment). Same offsets (PETH_S); readouts pick their own span (e.g. -0.1 .. +1 s).
    first_f = np.full(len(E), -1)
    for i_, (c_, stp_) in enumerate(zip(E.cue_s.to_numpy(float), E.stop_s.to_numpy(float))):
        stp_ = stp_ if np.isfinite(stp_) else c_ + 3.5
        j_ = np.searchsorted(lk_all, c_)
        if j_ < len(lk_all) and lk_all[j_] < stp_:
            first_f[i_] = int(np.searchsorted(ftr, lk_all[j_]))
    pos_all = E.pos.to_numpy()
    al = {}
    for name_, (f_, keep_) in (("cue", (cue_f, np.ones(len(E), bool))), ("lick", (first_f, first_f >= 0))):
        idx_ = f_[keep_][:, None] + off[None, :]
        okm_ = (idx_ >= 0) & (idx_ < len(Rd))
        idc_ = np.clip(idx_, 0, len(Rd) - 1)
        pos_ = pos_all[keep_]

        def _dev(X, okm_=okm_, pos_=pos_):
            """(trials, off, areas) minus each position's mean over its trials (frames outside the record excluded)."""
            Xm = np.where(okm_[..., None], X, np.nan)
            out = np.empty_like(Xm)
            for p_ in np.unique(pos_):
                m_ = pos_ == p_
                out[m_] = Xm[m_] - np.nanmean(Xm[m_], 0, keepdims=True)
            return out
        ddev_ = _dev(Rd[idc_])
        al[name_] = {"okm": okm_, "idc": idc_, "dev": _dev, "ddev": ddev_,
                     "sst": np.nansum(np.where(okm_[..., None], (Rd[idc_] - Rd.mean(0)) ** 2, np.nan), 0).T,
                     "sst_tr": np.nansum(np.where(okm_[..., None], ddev_ ** 2, np.nan), 0).T,
                     "sse": np.zeros((len(tr_models), Ra.shape[1], len(off))),
                     "sse_tr": np.zeros((len(tr_models), Ra.shape[1], len(off)))}
    for mi, mname in enumerate(tr_models):
        spec = models[mname]
        ixm = cols(spec)
        penm = np.array([model_alphas[mname][pen_of_col[i]] for i in ixm])
        iim, Bk = eng.betas(ixm, penm)
        pred = np.zeros_like(Rd)
        Am = A[:, iim[1:] - 1]
        for k, bk in enumerate(Bk):
            m = fold == k
            pred[m] = (Am[m] @ bk[1:] + bk[0]) @ Ra
        for a_ in al.values():
            okm_, idc_ = a_["okm"], a_["idc"]
            a_["sse"][mi] = np.nansum(np.where(okm_[..., None], (Rd[idc_] - pred[idc_]) ** 2, np.nan), 0).T
            # SINGLE-TRIAL (trial-to-trial) error (Priya 10-19, Musall): each position's trial-averaged response is
            # removed from data AND prediction first, so what is scored is the variability AROUND the condition mean
            a_["sse_tr"][mi] = np.nansum(np.where(okm_[..., None], (a_["ddev"] - a_["dev"](pred[idc_])) ** 2, np.nan),
                                         0).T
    sse, sst, sse_tr, sst_tr = al["cue"]["sse"], al["cue"]["sst"], al["cue"]["sse_tr"], al["cue"]["sst_tr"]
    # LICK-ALIGNED PETH (Priya 10-22: "can we do the same analysis but for lick-aligned encoder"): the same full-model
    # parts as `peth`, aligned on each trial's FIRST post-cue lick (first_f; trials without one left out), so the
    # decomposition of the post-lick response does not depend on the cue-to-lick latency (which rises post-stroke).
    peth_lick = np.full_like(peth, np.nan)
    n_lick_pos = np.zeros(len(pos_order), int)
    for ci, Cm in enumerate(comp.values()):
        R = (Cm @ Ra).astype(np.float32)
        for pi, p in enumerate(pos_order):
            fl = first_f[((E.pos == p).to_numpy()) & (first_f >= 0)]
            n_lick_pos[pi] = len(fl)
            if not len(fl):
                continue
            idx = fl[:, None] + off[None, :]
            ok = (idx >= 0) & (idx < len(R))
            seg = np.where(ok[..., None], R[np.clip(idx, 0, len(R) - 1)], np.nan)
            peth_lick[ci, pi] = np.nanmean(seg, 0).T
    res["n_trials_lick_aligned"] = int((first_f >= 0).sum())
    res["n_lick_aligned_by_pos"] = dict(zip(pos_order, n_lick_pos.tolist()))
    res["target_angle"] = B.target_angle
    res.update({"tr_models": tr_models, "pos_order": pos_order, "peth_parts": list(comp), "model_alphas": model_alphas,
                "areas": [names.get(a, str(a)) for a in areas] + ["cortex"]})
    return {"res": res, "maps": maps, "flat": ublk, "shape": ((shape[0] + 1) // 2, (shape[1] + 1) // 2),
            "region": pd.DataFrame(region),
            "peth": peth, "peth_lick": peth_lick, "peth_off_s": off / res["fs"], "sse": sse, "sst": sst,
            "sse_trial": sse_tr, "sst_trial": sst_tr,
            "sse_lick": al["lick"]["sse"], "sst_lick": al["lick"]["sst"],
            "sse_trial_lick": al["lick"]["sse_tr"], "sst_trial_lick": al["lick"]["sst_tr"]}


def out_dir(rv, order: str) -> Path:
    from wfield_local import orofacial_outputs as oo
    cams = [c for c in (os.environ.get(VIDEO_ENV) or "").split(",") if c]
    # a VIDEO run never writes into the primary folder: <order>_video-<cams>
    only = [x.rstrip(":") for x in (os.environ.get(ONLY_ENV) or "").split(",") if x]
    fr_set = os.environ.get(FR_ENV) or ""
    return oo.root(rv) / _run() / "imaging_encoding" / (order + (f"_video-{'-'.join(cams)}" if cams else "")
                                                         + (f"_fr-{_fr_tag(fr_set)}" if fr_set else "")
                                                         + ("share" if fr_set and not _fr_absolute() else "")
                                                         + (f"_pts{os.environ[FR_POINTS_ENV]}"
                                                            if fr_set and os.environ.get(FR_POINTS_ENV) else "")
                                                         + (f"_subset-{'-'.join(only)}" if only else "")
                                                         + ("_matched" if os.environ.get(MATCH_ENV) else "")
                                                         + (f"_exec-{os.environ[EXEC_ENV]}" if os.environ.get(EXEC_ENV)
                                                            else "")
                                                         + (f"_licks-{os.environ[LICKS_ENV]}"
                                                            if os.environ.get(LICKS_ENV, "cam4") != "cam4" else "")
                                                         + ("_bins" if os.environ.get(BINS_ENV) else "")
                                                         + (f"_bw{bins_width():g}" if os.environ.get(BINS_ENV)
                                                            and bins_width() != 1.0 else "")
                                                         + (f"_licksall-{os.environ[LICKS_ALL_ENV]}"
                                                            if os.environ.get(LICKS_ALL_ENV) else "")
                                                         + os.environ.get(SUFFIX_ENV, ""))


def save(out: dict, d: Path) -> None:
    from wfield_local import writeguard
    (d / "sessions").mkdir(parents=True, exist_ok=True)
    writeguard.assert_writable(d)
    lab = out["res"]["label"]
    keys = sorted(out["maps"])
    np.savez_compressed(d / "sessions" / f"{lab}.npz",
                        map_keys=np.array([f"{a}|{b}" for a, b in keys]),
                        maps=np.stack([out["maps"][k] for k in keys]).astype(np.float32),
                        flat=out["flat"], shape=np.array(out["shape"]), peth=out["peth"],
                        peth_off_s=out["peth_off_s"], sse=out["sse"], sst=out["sst"],
                        sse_trial=out["sse_trial"], sst_trial=out["sst_trial"],
                        **{k: out[k] for k in ("sse_lick", "sst_lick", "sse_trial_lick", "sst_trial_lick", "peth_lick") if k in out})
    out["region"].assign(label=lab).to_csv(d / "sessions" / f"{lab}_regions.csv", index=False, float_format="%.5f")
    (d / "sessions" / f"{lab}.json").write_text(json.dumps(out["res"], indent=1, default=str), encoding="utf-8")


def _worker(job):
    from wfield_local.orofacial_cohort import _repo_on_path
    _repo_on_path()
    label, order = job
    from wfield_local.paths import PathResolver
    d = out_dir(PathResolver(), order)
    out = run_session(label, order)
    save(out, d)
    r = out["region"]
    full = r[(r.model == "full") & (r.window == "all") & (r.area == "cortex")].r2.iloc[0]
    return {"label": label, "full_cvR2": float(full), "n_frames": out["res"]["n_frames"]}


def sessions_available() -> list[str]:
    from wfield_local import analysis_kit as ak
    from wfield_local import config
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    out = []
    for s in ak.curated_sessions(order="sorted"):
        a = config.animal_of(s["label"])
        date = "2026" + s["label"].split("_")[-1]
        if (oo.session_dir(rv, _run(), a, date) / "manifest.json").exists() and s.get("regime") == "B":
            out.append(s["label"])
    return out


def main(argv=None) -> int:
    from wfield_local import parallel
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--order", choices=tuple(ORDERS), default="movement_first")
    ap.add_argument("--jobs", type=int, default=3, help="sessions in parallel (each ~3-4 GB)")
    ap.add_argument("--run", default=None, help="LP run (default: orofacial_cohort.RUN); outputs go under it")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated model-name prefixes to fit (+ full), e.g. "
                    f"'nodir:' (sets {ONLY_ENV}; outputs -> <order>_subset-<prefixes>/)")
    ap.add_argument("--video", default=None, help="cameras for the video variable, e.g. cam1,cam4 (sets "
                    f"{VIDEO_ENV}; outputs -> <order>_video-<cams>/; sessions without motion energy are skipped)")
    ap.add_argument("--fr", default=None, help="FaceRhythm factor set for the fr variable, e.g. master_pre_v1 (sets "
                    f"{FR_ENV}; outputs -> <order>[_video-..]_fr-<set>/; sessions without projections are skipped)")
    ap.add_argument("--fr-share", action="store_true", help="with --fr: the factors' per-frame SHARE (the paper) "
                    f"instead of absolute expression (sets {FR_SHARE_ENV}; folder ..._fr-<set>share)")
    ap.add_argument("--fr-points", type=int, default=0, help="with --fr: K point-displacement PCs as frpts (sets "
                    f"{FR_POINTS_ENV}; outputs -> ..._fr-<set>_pts<K>/)")
    a = ap.parse_args(argv)
    if a.run:
        os.environ[RUN_ENV] = a.run
    if a.fr:
        os.environ[FR_ENV] = a.fr
    if a.fr_points:
        os.environ[FR_POINTS_ENV] = str(a.fr_points)
    if a.fr_share:
        os.environ[FR_SHARE_ENV] = "1"
    if a.video:
        os.environ[VIDEO_ENV] = a.video
    if a.only:
        os.environ[ONLY_ENV] = a.only
    from wfield_local.paths import PathResolver
    labels = sessions_available() if a.all else a.labels
    if a.video:                                             # only sessions with every camera's motion energy
        from scripts.session_poses import session_dir
        from wfield_local import config as _cfg
        rv_ = PathResolver()

        def _has(x):
            an = _cfg.animal_of(x)
            d = session_dir(rv_, f"{an}:2026{x.split('_')[-1]}:full")[2]
            return all((d / f"video_motion_{c}.npz").exists() for c in a.video.split(","))
        labels = [x for x in labels if _has(x)]
        print(f"video run ({a.video}): {len(labels)} sessions with motion energy")
    if a.fr:                                                # only sessions with a FaceRhythm projection
        from wfield_local import config as _cfg
        from wfield_local import face_rhythm_outputs as fro
        fd = fro.factor_dir(PathResolver(), set_name=a.fr)
        labels = [x for x in labels if fro.session_keys(fd, _cfg.animal_of(x), "2026" + x.split("_")[-1])]
        print(f"fr run ({a.fr}): {len(labels)} sessions with FaceRhythm projections")
    if a.skip_done:
        d = out_dir(PathResolver(), a.order) / "sessions"
        labels = [x for x in labels if not (d / f"{x}.json").exists()]
    res, fail = parallel.fan_out([(x, a.order) for x in labels], _worker, jobs=a.jobs, label="session")
    for _it, r in sorted(res, key=lambda r: r[0][0]):
        print(f"  {r['label']}: full cvR2 (cortex, all frames) = {r['full_cvR2']:.3f}  ({r['n_frames']} frames)")
    for it, err in fail:
        print(f"  FAILED {it[0]}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
