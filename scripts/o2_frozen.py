"""FROZEN-ENCODER FRAMES on HMS O2 (lane F, 11-24; Priya: the movement page's page-lineage frames on the four-camera
root, wave 1 = the conclusion sections). The desktop does the cheap, pooled part of a frame; O2 does the per-session
part. Every O2 task replays the session's recorded loader values (`o2_partial.Recorder` / `Replay`), never the share.

    python scripts/o2_frozen.py bundle  --frame spont_frA --name <bundle> --user ps150 [--jobs 3]
    python scripts/o2_frozen.py check   --frame spont_frA --name <bundle> --mode predict --out <scratch> LABEL
    python scripts/o2_frozen.py collect --frame spont_frA --name <bundle> --mode predict|svf|partial

WHAT RUNS WHERE (measured 11-24 on the four-camera env: one `fit_animal` 2-9 s; one `session_design` ~60 s, of which
`load_imaging` ~22 s; the 11-22 four-camera frA frame took 247 min locally at --jobs 3, i.e. the per-session
predictions dominate):
  desktop (bundle)  `frozen_encoder.main` itself, unchanged: templates, (basis), the pre statistics, every animal x
                    variant fit (fit/ files written into the frame folder on the share, as a local run writes them);
                    only its per-session PREDICT fan-out is replaced by a capture pass that records each session's
                    loader values + its design digest (the pre sessions are recorded during the statistics pass, so
                    they are not built twice). The fits are pickled per animal into the bundle.
  O2 (one array task = one session, three job types sharing the same recorded inputs)
    predict  `frozen_encoder.predict_job` -> <variant>/sessions/<label>.{npz,json}, dropspout/, joint/<label>_trials.npz
    svf      `session_vs_frozen.self_fit_job` (the per-session self-fit; frA frame only)
    partial  `partial_refit.job` (the frozen partial refits; frA frame only; fits.pkl = `partial_refit.frozen_fits`)
  desktop (collect) copies the outputs into the frame folder (refuses foreign / differing files), then the run's own
                    summaries: `summarise_drop` / `summarise_musall`, `session_vs_frozen --plot-only`,
                    `partial_refit --plot-only`.

GUARANTEES (as o2_partial / o2_encoding): env.json = every setting, applied by the runner BEFORE wfield_local is
imported; the runner REFUSES unless the bundled code computes the recorded identity (run tag / selection / area basis +
frozen design: design name, running-model tag incl. FR / lickspout / lickland / te, FR variables). The frozen run tag must
equal the v36 page-lineage frame's (`V36_RUN_MODE`, read from its fit/pre_stats.pkl) -- the same frame, four-camera
inputs; a NEW Allen-area frame is on the ALM / tjM1 SPLIT area basis (Priya 11-26; run tag + "|areas-split", env.json
FROZEN_AREA_BASIS=split; `resolve_frame_area_basis`), an existing frame folder keeps its recorded basis. A split frame's
inputs pickles carry the session's area masks (`area_masks`, image pixel ids, built at bundle time: the v8 rim needs the
share's brain mask) and its imaging pickles keep the stat mask + those pixels; the O2 task REFUSES without them. Each O2
task checks the design it builds against the digest recorded at bundle time (exact, or float-equal to
1e-9 of each column's sum |x|). The attempts table the per-trial files join (`lp_licks.attempts_path`) is shipped in the
bundle (tables/trial_attempts.csv, sha recorded) and patched in on O2. Analysis-mask digest checked on every pickle.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o2_partial as op  # noqa: E402  (Recorder / Replay / identity / digests)

PROFILE = "cohort1_rebuild_20261123_fourcam"
INPUT_CACHE = "C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104"
#: the v36 page-lineage env (o2_partial.env_noFR) with the four-camera inputs: fused four-camera licks, no 10-23 cam4
#: EXEC3D snapshot (it would override the fused post-cue kinematics; dropped as in G4). Gate set HERE, per bundle.
#: Priya 11-24: every pose regressor 3-D (jaw / nose from the four-camera frames, despiked; combined_encoding.JAW_ENV).
POSE_ENV = "COMBINED_ENCODING_POSE"
BASE_ENV = {"OROFACIAL_GATE": "attempted", "COMBINED_ENCODING_LICKS": "fourcam", "COMBINED_ENCODING_ONLY": "full",
            "COMBINED_ENCODING_SUFFIX": "_gate-attempted", "COMBINED_ENCODING_TONGUE": "3d",
            "COMBINED_ENCODING_JAW": "3d", "COMBINED_ENCODING_NOSE": "3d", "COMBINED_ENCODING_POSE": "xyz3d"}
POP_ENV = ("COMBINED_ENCODING_LICKS_ALL", "COMBINED_ENCODING_EXEC", "COMBINED_ENCODING_TARGET3D_TABLE",
           "COMBINED_ENCODING_EXEC3D_TABLE", "COMBINED_ENCODING_FR", "COMBINED_ENCODING_FR_GROUPS",
           "COMBINED_ENCODING_BINS_CALIB", "COMBINED_ENCODING_ENL_SOURCE")
#: the landing-zone calibration the TE / lickland designs read (`frozen_encoder.LICKLAND_CALIB`): the FOUR-CAMERA one
#: (Priya OK 11-24; `scripts/exec_bins_calibration.py` on kinematics_fused_postcue, lick_key join), named explicitly
#: so it travels in the bundle (@tables). The 10-23 cam4-era snapshot was the v36 frames' (default under the cache).
CALIB = {"FROZEN_LICKLAND_CALIB": "C:/Users/SabatiniLab/wfield_cache/pose3d_inputs/exec_bins_calib_fourcam_20261124.csv"}
FR_A = {"FROZEN_FR": "1", "COMBINED_ENCODING_FR": "master_pre_v1", "COMBINED_ENCODING_FR_GROUPS": "master_v1:A"}
SPLIT, NONE = {"FROZEN_LICKS_ALL": "split"}, {"FROZEN_LICKS_ALL": "none"}
FR_LAGS = "-0.3,-0.2,-0.1,0.1,0.2,0.3"                  # the v36 spont_frA_frlag frame's FROZEN_FR_LAGS
_LS_VARIANTS = ("joint target_first exec_first no_target no_spout no_task no_movement no_lickspout no_exec_locked "
                "no_exec_within no_lick no_other no_spont no_contact no_fr no_fr_dir task_only lick_only spout_only "
                "other_only movement_only fr_only novar_arrival novar_contact novar_cue novar_exec_angle novar_fr_close "
                "novar_fr_far novar_fr_left novar_fr_lick novar_fr_other_0 novar_fr_other_1 novar_fr_other_10 "
                "novar_fr_other_3 novar_fr_other_6 novar_fr_other_9 novar_fr_right novar_jaw_dir novar_jaw_mag "
                "novar_lick_onset novar_licks_iti novar_licks_pre novar_nose_dir novar_nose_mag novar_pos_cue "
                "novar_pos_enl novar_push novar_running novar_tongue_dir novar_tongue_mag").split()
SVF_VARIANTS = ("joint", "no_target", "no_spout", "no_task", "no_movement", "no_fr", "no_fr_dir", "no_lickspout",
                "no_spont", "no_other")
#: frame key -> tag, FROZEN env, frozen_encoder.main args (the v36 queue's glue-out commands, minus --all / --jobs /
#: --tag), O2 job types. Wave 1 = the conclusion sections (Priya / lead 11-24).
FRAMES = {
    "spont_frA": {"tag": "gate-attempted_v7g3_t3d_spont_frA", "env": {**SPLIT, **FR_A}, "args": [],
                  "jobs": ("predict", "svf", "partial"), "wave": 1},
    # SENSITIVITY FRAME (Priya 11-25 "let's do it"; lane E's RESULTS_frameA_check_20261125: frame A's larger chronic
    # residual sits in the 3-D nose position columns (PS92 / 94 / 95) and 3-D jaw_mag (PS93)): frame A with the jaw and
    # nose entering as 3-D SPEED only (combined_encoding POSE=xyz3d_spd; no jaw / nose position columns; tongue, FR,
    # licks, gate, pre set, modes unchanged). Built from the wave-1 inputs (bundle --inputs-from o2f_wave1_20261124).
    "spont_frA_spd": {"tag": "gate-attempted_v7g3_t3d_spont_frA_spd", "env": {**SPLIT, **FR_A, POSE_ENV: "xyz3d_spd"},
                      "args": [], "jobs": ("predict", "svf", "partial"), "wave": 1.5},
    # LAGGED FACERHYTHM (coordinator 11-25, Fig 22.2 contest_fr_lags): frame A's exact design (xyz3d, positions kept)
    # plus the FROZEN_FR_LAGS copies of the v36 page-lineage spont_frA_frlag frame (run tag |frlags=-0.3,...,0.3; its
    # variant list, RENDER_QUEUE_DAQLK_20261105 t_frame_frlag). Lag copies are built outside the recorded loader
    # (`frozen_encoder.fr_lag_columns`), so the wave-1 inputs replay for it too.
    "spont_frA_frlag": {"tag": "gate-attempted_v7g3_t3d_spont_frA_frlag", "env": {**SPLIT, **FR_A, "FROZEN_FR_LAGS": FR_LAGS},
                        "args": ["--variants", *"joint no_fr no_fr_dir no_frfacelead no_frcortexlead no_movement".split()],
                        "jobs": ("predict", "svf", "partial"), "wave": 1.5},
    "spont_frA_basis": {"tag": "gate-attempted_v7g3_t3d_spont_frA_basis", "env": {**SPLIT, **FR_A},
                        "args": ["--target", "basis", "--no-drop-spout", "--variants", *"joint no_target no_spout "
                                 "no_task no_movement no_fr no_fr_dir no_fr_all no_lick no_exec_within no_other "
                                 "no_spont".split()], "jobs": ("predict",), "wave": 1},
    # BASIS TWIN of spont_frA_spd (Priya 11-26, option a): spont_frA_basis's design + variants with the speed-only jaw /
    # nose layout, so encoder_split_rows (ALM / tjM1 rows) and frozen_residual_significance (pixel residual maps) read
    # <spd run>_basis. Built from the wave-1a2 spont_frA_basis recorded loaders (bundle --inputs-from o2f_wave1a2_20261124
    # --inputs-frame spont_frA_basis).
    "spont_frA_spd_basis": {"tag": "gate-attempted_v7g3_t3d_spont_frA_spd_basis",
                            "env": {**SPLIT, **FR_A, POSE_ENV: "xyz3d_spd"},
                            "args": ["--target", "basis", "--no-drop-spout", "--variants", *"joint no_target no_spout "
                                     "no_task no_movement no_fr no_fr_dir no_fr_all no_lick no_exec_within no_other "
                                     "no_spont".split()], "jobs": ("predict",), "wave": 1.5},
    "frA_te": {"tag": "gate-attempted_v7g3_t3d_frA_te", "env": {**NONE, **FR_A, "FROZEN_DESIGN": "te", **CALIB},
               "args": ["--no-drop-spout"], "jobs": ("predict",), "wave": 1},
    "frA_te_basis": {"tag": "gate-attempted_v7g3_t3d_frA_te_basis", "env": {**NONE, **FR_A, "FROZEN_DESIGN": "te", **CALIB},
                     "args": ["--no-drop-spout", "--target", "basis", "--variants", *"joint target_first exec_first "
                              "no_timing no_target no_execution no_contact no_other no_task no_movement task_only "
                              "target_only execution_only other_only movement_only".split()],
                     "jobs": ("predict",), "wave": 1},
    "spont_frA_lsland": {"tag": "gate-attempted_v7g3_t3d_spont_frA_lsland",
                         "env": {**SPLIT, **FR_A, "FROZEN_LICKSPOUT": "1", "FROZEN_LICKLAND": "1", **CALIB},
                         "args": ["--variants", *"joint target_first exec_first no_target no_spout no_task no_movement "
                                  "no_lickspout no_exec_locked no_exec_within no_lick no_fr no_spont no_other".split()],
                         "jobs": ("predict",), "wave": 1},
    "spont_frA_ls_basis": {"tag": "gate-attempted_v7g3_t3d_spont_frA_ls_basis",
                           "env": {**SPLIT, **FR_A, "FROZEN_LICKSPOUT": "1"},
                           "args": ["--target", "basis", "--variants", *_LS_VARIANTS], "jobs": ("predict",), "wave": 1},
    "spont_basis": {"tag": "gate-attempted_v7g3_t3d_spont_basis", "env": {**SPLIT},
                    "args": ["--target", "basis"], "jobs": ("predict",), "wave": 1},
    "spont_frAsplit_basis": {"tag": "gate-attempted_v7g3_t3d_spont_frAsplit_basis",
                             "env": {**SPLIT, **FR_A, "FROZEN_FR_SPLIT": "1"},
                             "args": ["--target", "basis", "--no-drop-spout", "--variants", *"joint no_posinfo "
                                      "no_noposinfo no_target no_spout no_fr_all no_fr_dir".split()],
                             "jobs": ("predict",), "wave": 1},
}
#: the v36 page-lineage frames' running-model tags (`frozen_encoder._run_tag`, stored in each fit/pre_stats.pkl): the
#: bundle refuses an env that does not reproduce it (same frame definition; only the inputs are four-camera)
_K0 = "spline_thr|k0=1.5"
_FRA = "|fr=master_pre_v1:master_v1:A"
V36_RUN_MODE = {
    "spont_frA": f"{_K0}|tongue3d{_FRA}|licksall-split",
    "spont_frA_spd": f"{_K0}|tongue3d{_FRA}|licksall-split",
    "spont_frA_frlag": f"{_K0}|tongue3d{_FRA}|frlags={FR_LAGS}|licksall-split",
    "spont_frA_basis": f"{_K0}|tongue3d{_FRA}|licksall-split",
    "spont_frA_spd_basis": f"{_K0}|tongue3d{_FRA}|licksall-split",
    "frA_te": f"{_K0}|te|tongue3d{_FRA}",
    "frA_te_basis": f"{_K0}|te|tongue3d{_FRA}",
    "spont_frA_lsland": f"{_K0}|tongue3d{_FRA}|lickspout|lickland|licksall-split",
    "spont_frA_ls_basis": f"{_K0}|tongue3d{_FRA}|lickspout|licksall-split",
    "spont_basis": f"{_K0}|tongue3d|licksall-split",
    "spont_frAsplit_basis": f"{_K0}|tongue3d{_FRA}|frsplit|licksall-split",
}
V36_E = ("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
         "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-"
         "stopA2jexrt3500c70_daqlk")
MEM = {"predict": "16G", "svf": "16G", "partial": "16G"}       # basis frames: 32G (set per frame in COMMANDS)
TIME = {"predict": "0-04:00:00", "svf": "0-02:00:00", "partial": "0-03:00:00"}
ENVKEYS = ("O2F_RDIR", "O2F_IMDIR", "O2F_MASK")


def expected_run_tag(key: str) -> str:
    """The v36 frame's running-model tag with this bundle's 3-D jaw / nose parts (the only intended design change):
    the pose layout (xyz3d, or xyz3d_spd = speed-only jaw / nose) is the frame's own setting (FRAMES env)."""
    layout = os.environ.get(POSE_ENV, "").strip()
    if layout not in ("xyz3d", "xyz3d_spd"):
        raise SystemExit(f"{POSE_ENV}={layout!r}: every O2 frozen frame is on a 3-D pose layout")
    extra = f"|jaw3d|{layout}|nose3d" + (f"={os.environ['COMBINED_ENCODING_NOSE_FRAMES']}"
                                         if os.environ.get("COMBINED_ENCODING_NOSE_FRAMES") else "")
    from wfield_local import frozen_encoder as fe
    split = f"{fe._AREA_TAG}split" if not is_basis(key) and fe.area_basis() == "split" else ""   # AREA BASIS (11-26)
    return V36_RUN_MODE[key].replace("|tongue3d", "|tongue3d" + extra) + split


def resolve_frame_area_basis(key: str, out_root=None) -> str | None:
    """AREA BASIS of an Allen-area frame (`frozen_encoder.resolve_area_basis`, Priya 11-26): an existing frame folder keeps
    its recorded basis, a NEW one is split unless the frame's env names one; set in the environment BEFORE the identity /
    run tag are recorded (env.json carries FROZEN_AREA_BASIS to O2). Spatial-target frames: none (unset)."""
    from wfield_local import frozen_encoder as fe
    if is_basis(key):
        os.environ.pop(fe.AREA_BASIS_ENV, None)
        return None
    b = fe.resolve_area_basis(FRAMES[key]["env"].get(fe.AREA_BASIS_ENV), frame_out(key, out_root))
    os.environ[fe.AREA_BASIS_ENV] = b
    return b


def is_basis(key: str) -> bool:
    return "--target" in FRAMES[key]["args"] and "basis" in FRAMES[key]["args"]


def apply_frame_env(key: str, nose_frames: str = "") -> None:
    """Desktop: the four-camera render profile, then the frame's settings (BEFORE wfield_local's env-reading modules)."""
    from wfield_local import render_profile
    for k in [k for k in os.environ if k.startswith("FROZEN_")]:
        del os.environ[k]
    render_profile.apply(PROFILE)
    for k in POP_ENV:
        os.environ.pop(k, None)
    os.environ.update({**BASE_ENV, **FRAMES[key]["env"], "WFIELD_INPUT_CACHE": INPUT_CACHE})
    if nose_frames:                                    # a re-triangulated 3-D nose (combined_encoding.NOSE_FRAMES_ENV)
        os.environ["COMBINED_ENCODING_NOSE_FRAMES"] = nose_frames
    # the epoch boundaries the desktop uses (`epochs.boundaries_path`, <cohort_out>/epoch_boundaries.json): named
    # explicitly so the file travels in the bundle (@tables) -- on O2 the share path is absent and `epoch_of` would
    # fall back to the stored spec (pre / post labels pick LOSO vs all-pre fits). Found by the audited check 11-24.
    if not os.environ.get("WIDEFIELD_EPOCH_BOUNDARIES"):
        from wfield_local import epochs
        bp = epochs.boundaries_path()
        if bp is None or not Path(bp).is_file():
            raise SystemExit(f"epoch boundaries file missing: {bp}")
        os.environ["WIDEFIELD_EPOCH_BOUNDARIES"] = str(bp)


def declared_out_root(rmeta: dict):
    """The frame-folder parent a bundle's run.json DECLARES, or None (a frame routed to the share root at bundle time).
    Two spellings exist: ``test_out_root`` (this code; a TEST session subset, "" for a routed full bundle) and
    ``out_root`` (the rule-11 bundles of 05e363a4 / ca1ad428, e.g. o2f_wave1a2_20261124 / o2f_wave1b2_20261124, whose
    frames live in <bundle>/frames). Both set and different -> SystemExit (11-26)."""
    o, t = str(rmeta.get("out_root") or ""), str(rmeta.get("test_out_root") or "")
    if o and t and Path(o) != Path(t):
        raise SystemExit(f"run.json declares two frame roots: out_root {o} vs test_out_root {t}; refusing")
    return o or t or None


def frame_out(key: str, out_root=None) -> Path:
    """The frame folder: the routed share root (`frozen_encoder.out_root`), or ``out_root`` (a TEST bundle only)."""
    from wfield_local import frozen_encoder as fe
    return Path(out_root or fe.out_root()) / f"frozen_encoder_{FRAMES[key]['tag']}"


# --------------------------------------------------------------------------- inputs REUSED from an earlier bundle
#: `bundle --inputs-from <bundle>` (11-25, the speed-only sensitivity frame): every session's loader values are REPLAYED
#: from that bundle's runs/<frame>/inputs/<label>.pkl + imaging/<label>.pkl (o2_partial.Replay, as an O2 task does) for
#: the templates, the pre statistics and the capture pass alike, so no share / cache read and no re-extraction; the new
#: bundle's inputs/<label>.pkl is THIN (this frame's design digest + sums only) and names the source bundle. The O2
#: runner reads the loader values from the SIBLING bundle on scratch (<scratch>/o2/<source bundle>), else from the
#: recorded absolute path (the desktop pilot). The analysis-mask digest of every source pickle must be this bundle's.
INPUTS_FROM_ENV = "O2F_INPUTS_FROM"                     # desktop: the source bundle's runs/<frame> folder


def _reuse_src():
    """(runs/<frame> dir, imaging dir, {bundle, frame, path}) of the bundle whose inputs are replayed, or None."""
    p = os.environ.get(INPUTS_FROM_ENV)
    if not p:
        return None
    r = Path(p)
    return r, r.parent.parent / "imaging", {"bundle": r.parent.parent.name, "frame": r.name, "path": str(r.parent.parent)}


def _source_pickles(sb: Path, frame: str, label: str, mask_sig: str) -> tuple[dict, dict]:
    """(inputs, imaging) pickles of ``label`` in source bundle ``sb``; both must carry this bundle's mask digest."""
    from wfield_local import mask_keys
    inp = pickle.loads((sb / "runs" / frame / "inputs" / f"{label}.pkl").read_bytes())
    img = pickle.loads((sb / "imaging" / f"{label}.pkl").read_bytes())
    for d_, what in ((inp, "inputs"), (img, "imaging")):
        if d_.get(mask_keys.FIELD) != mask_sig:
            raise RuntimeError(f"{label}: source {what} pickle masks {d_.get(mask_keys.FIELD)} != this bundle's {mask_sig}")
    return inp, img


def _replay_of(label: str):
    """(context, source meta): the Replay of the source bundle's recorded loaders, or a null context without reuse. The
    replayed imaging is the source bundle's REDUCED pixel set, so split area masks come from the source inputs pickle
    (refused when it has none: a plain bundle's imaging cannot carry a split frame)."""
    import contextlib

    from wfield_local import frozen_encoder as fe
    src = _reuse_src()
    if src is None:
        return contextlib.nullcontext(), None
    r, _imdir, meta = src
    inp, img = _source_pickles(r.parent.parent, r.name, label, os.environ["O2F_MASK"])
    stack = contextlib.ExitStack()
    stack.enter_context(op.Replay({**img["memo"], **inp["memo"]}, inp["files"], inp["globs"]))
    stack.enter_context(fe.bundled_area_masks({label: inp["area_masks"]} if inp.get("area_masks") else {}))
    return stack, meta


def template_replay_job(label: str) -> dict:
    """`frozen_encoder.template_job` on the source bundle's replayed loaders (bundle --inputs-from)."""
    from wfield_local import frozen_encoder as fe
    rep, _src = _replay_of(label)
    with rep:
        return fe.session_design(label, template_only=True)


# ------------------------------------------------------------------------------------------------- capture (desktop)
def _save_capture(label: str, rec, d: dict, rdir: Path, imdir: Path, mask_sig: str) -> dict:
    """o2_partial.capture_worker's output files from a finished Recorder + the design it saw; ``rec`` None = the
    loaders were replayed from another bundle (INPUTS_FROM_ENV): a thin inputs pickle naming it, no imaging pickle."""
    from wfield_local import combined_encoding as ce
    from wfield_local import mask_keys
    if rec is None:
        src = _reuse_src()[2]
        out = {mask_keys.FIELD: mask_sig, "label": label, "inputs_from": src, "design": op.design_digest(d),
               "design_sums": op.design_sums(d), "n_frames": int(len(d["X"])), "n_cols": int(d["X"].shape[1]),
               **({"area_masks": d["area_masks"]} if d.get("area_masks") is not None else {})}
        dst = rdir / "inputs" / f"{label}.pkl"
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(out, protocol=5))
        os.replace(tmp, dst)
        return {"label": label, "MB": round(dst.stat().st_size / 1e6), "loaders": ["replayed:" + src["bundle"]]}
    imaging = {k: v for k, v in rec.memo.items() if v[0] == "load_imaging"}
    other = {k: v for k, v in rec.memo.items() if v[0] != "load_imaging"}
    ip = imdir / f"{label}.pkl"
    hemo = ce.hemo_dir()
    # the pixels kept (o2_partial._reduce): stat mask (plain / spatial frames: no key, as before) or stat + split areas
    pix = {"pixels": "stat+split"} if d.get("area_masks") is not None else {}
    if ip.exists():
        old = pickle.loads(ip.read_bytes())
        if old[mask_keys.FIELD] != mask_sig or old["hemo"] != hemo or set(old["memo"]) != set(imaging):
            raise RuntimeError(f"{ip}: built on other masks / hemo / keys; use a new bundle name")
        if old.get("pixels") != pix.get("pixels"):
            raise RuntimeError(f"{ip}: imaging reduced to {old.get('pixels', 'stat')} pixels, this frame needs "
                               f"{pix.get('pixels', 'stat')}: a split and a plain frame need separate bundles")
    else:
        tmp = ip.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({mask_keys.FIELD: mask_sig, "hemo": hemo, "label": label, "memo": imaging, **pix},
                                     protocol=5))
        os.replace(tmp, ip)
    out = {mask_keys.FIELD: mask_sig, "label": label, "memo": other, "files": rec.files, "globs": rec.globs,
           "design": op.design_digest(d), "design_sums": op.design_sums(d), "n_frames": int(len(d["X"])),
           "n_cols": int(d["X"].shape[1]),
           # SPLIT frames: the session's area masks (image pixel ids), built here where the share is visible; the O2
           # task serves them (`frozen_encoder.bundled_area_masks`) and refuses without them (11-26)
           **({"area_masks": d["area_masks"]} if d.get("area_masks") is not None else {})}
    dst = rdir / "inputs" / f"{label}.pkl"
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(out, protocol=5))
    os.replace(tmp, dst)
    return {"label": label, "MB": round((dst.stat().st_size + ip.stat().st_size) / 1e6),
            "loaders": sorted({v[0] for v in rec.memo.values()})}


def _recording(label: str):
    """(Recorder, holder): the Recorder over the loaders + a session_design wrapper keeping the design it returns."""
    from wfield_local import frozen_encoder as fe
    holder: list = []
    orig = fe.session_design

    def sd(*a, **k):
        d = orig(*a, **k)
        if not k.get("template_only") and not k.get("basis_only"):
            holder.append(d)
        return d
    return op.Recorder(label), holder, orig, sd


def stats_capture_job(item):
    """Worker (replaces `frozen_encoder.stats_job` during bundle): the statistics of one PRE session, exactly as
    stats_job, while recording that session's loader values + design digest (so capture never rebuilds it)."""
    from wfield_local import frozen_encoder as fe
    from wfield_local import mask_keys
    rdir, imdir, mask_sig = Path(os.environ["O2F_RDIR"]), Path(os.environ["O2F_IMDIR"]), os.environ["O2F_MASK"]
    if mask_keys.signature() != mask_sig:
        raise RuntimeError(f"{item[0]}: worker masks differ from the bundle's {mask_sig}")
    rec, holder, orig, sd = _recording(item[0])
    rep, src = _replay_of(item[0])
    fe.session_design = sd
    try:
        with rep:
            if src is None:
                with rec:
                    out = fe.stats_job(item)
            else:                                        # replayed loaders: nothing to record (INPUTS_FROM_ENV)
                out = fe.stats_job(item)
    finally:
        fe.session_design = orig
    _save_capture(item[0], None if src else rec, holder[-1], rdir, imdir, mask_sig)
    return out


def capture_job(job):
    """Worker: one post (or not yet captured) session's design under the Recorder, with predict_job's exact arguments."""
    label, tpath, spath = job
    from wfield_local import config, mask_keys
    from wfield_local import frozen_encoder as fe
    rdir, imdir, mask_sig = Path(os.environ["O2F_RDIR"]), Path(os.environ["O2F_IMDIR"]), os.environ["O2F_MASK"]
    if mask_keys.signature() != mask_sig:
        raise RuntimeError(f"{label}: worker masks differ from the bundle's {mask_sig}")
    an = config.animal_of(label)
    rep, src = _replay_of(label)
    with rep:
        if src is None:
            with op.Recorder(label) as rec:
                d = fe.session_design(label, fe._templates_of(tpath, an), spatial=fe._spatial_of(spath, an))
        else:
            rec = None
            d = fe.session_design(label, fe._templates_of(tpath, an), spatial=fe._spatial_of(spath, an))
    return _save_capture(label, rec, d, rdir, imdir, mask_sig)


_FAN_STATE: dict = {}


def _capture_fan(jobs, n):
    """Replaces `frozen_encoder._fan` (the predict fan-out) during bundle: keep each animal's fits, capture the sessions
    not yet recorded. Returns (no results, failures) so main's tail runs unchanged."""
    from wfield_local import analysis_kit as ak
    from wfield_local import config
    rdir = Path(os.environ["O2F_RDIR"])
    (rdir / "fits").mkdir(exist_ok=True)
    done_an = set()
    todo = []
    for job in jobs:
        label, fits, _out, tp, spath = job[:5]
        an = config.animal_of(label)
        if an not in done_an:
            (rdir / "fits" / f"{an}.pkl").write_bytes(pickle.dumps(fits, protocol=pickle.HIGHEST_PROTOCOL))
            done_an.add(an)
        _FAN_STATE.setdefault("tp", tp)
        _FAN_STATE.setdefault("spath", spath)
        if not (rdir / "inputs" / f"{label}.pkl").exists():
            todo.append((label, tp, spath))
    _FAN_STATE["labels"] = sorted(j[0] for j in jobs)
    print(f"[o2_frozen] fits of {sorted(done_an)} kept; {len(todo)} of {len(jobs)} sessions to capture", flush=True)
    t0 = time.time()
    got, fail = ak.fan_sessions(todo, capture_job, jobs=n, key=lambda j: j[0], label="capture")
    print(f"[o2_frozen] capture {len(got)} ok, {len(fail)} failed in {(time.time() - t0) / 60:.1f} min", flush=True)
    return [], [((it[0],), err) for it, err in fail]


def cmd_bundle(a) -> Path:
    import datetime

    from wfield_local import frozen_encoder as fe
    from wfield_local import lp_licks, mask_keys, o2_bundle, writeguard
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    key = a.frame
    F = FRAMES[key]
    area_b = resolve_frame_area_basis(key, a.out_root)  # BEFORE the run tag / identity are recorded
    want = expected_run_tag(key)
    if a.labels and not a.out_root:
        raise SystemExit("a session subset needs --out-root (a TEST bundle never writes the share frame folder)")
    src_bundle = None
    if a.inputs_from:                                    # replay another bundle's recorded loaders (INPUTS_FROM_ENV)
        src_bundle = Path(a.inputs_from) if Path(a.inputs_from).is_absolute() else op._bundle_dir(a.inputs_from)
        src_frame = a.inputs_frame or key
        src_r = src_bundle / "runs" / src_frame
        if not (src_r / "labels.txt").exists() or not (src_bundle / "imaging").is_dir():
            raise SystemExit(f"--inputs-from: {src_r} has no labels.txt / {src_bundle}/imaging")
        src_meta = json.loads((src_bundle / "bundle.json").read_text())
        if src_meta.get(o2_bundle.MASK_FIELD) != mask_keys.signature():
            raise SystemExit(f"--inputs-from: {src_bundle} masks {src_meta.get(o2_bundle.MASK_FIELD)} != "
                             f"{mask_keys.signature()}")
        os.environ[INPUTS_FROM_ENV] = str(src_r)
    if fe._run_tag() != want:
        raise SystemExit(f"{key}: this env's frozen run tag {fe._run_tag()!r} != the v36 frame's + 3-D jaw / nose {want!r}")
    repo = Path(__file__).resolve().parents[1]
    dirty = op._git(repo, "status", "--porcelain", "--", "wfield_local", "configs", "scripts")
    if dirty:
        raise SystemExit(f"uncommitted changes in {repo} (the bundle archives HEAD):\n{dirty}")
    commit = op._git(repo, "rev-parse", "HEAD")
    out = frame_out(key, a.out_root)
    if any((out / "joint" / "sessions").glob("*.json")):
        raise SystemExit(f"{out} already holds session outputs: not a bundle target (never overwrite a frame)")
    d = Path(a.bundle_root) / a.name if a.bundle_root else op._bundle_dir(a.name)
    writeguard.assert_writable(d)
    rdir = d / "runs" / key
    for sub in (*(() if src_bundle else (d / "imaging",)), d / "logs", rdir / "inputs", rdir / "fit", rdir / "fits",
                rdir / "tables"):
        sub.mkdir(parents=True, exist_ok=True)
    ident = op.identity()
    mask_sig = mask_keys.signature()
    meta_f = d / "bundle.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if meta:
        try:
            o2_bundle.verify_masks(meta, mask_sig)
        except o2_bundle.BundleMismatch as ex:
            raise SystemExit(f"refusing to extend {d}: {ex}") from None
        if meta.get("commit") != commit:
            raise SystemExit(f"{d} was built with code {meta.get('commit')}; HEAD is {commit}: use a new bundle name")
    rmeta_f = rdir / "run.json"
    if rmeta_f.exists() and json.loads(rmeta_f.read_text())["identity"] != ident:
        raise SystemExit(f"{rdir} holds another env: {json.loads(rmeta_f.read_text())['identity']} vs {ident}")
    os.environ.update({"O2F_RDIR": str(rdir), "O2F_IMDIR": str(d / "imaging"), "O2F_MASK": mask_sig})
    # frozen_encoder.main, unchanged except: the pre statistics also record; the predict fan-out captures; summaries
    # wait for collect (they read the predictions)
    saved = (fe.stats_job, fe._fan, fe.summarise_drop, fe.summarise_musall, fe.template_job)
    fe.stats_job, fe._fan = stats_capture_job, _capture_fan
    fe.summarise_drop = fe.summarise_musall = lambda *x, **k: None
    if src_bundle:                                       # the template pass replays too (no share / cache read)
        fe.template_job = template_replay_job
    t0 = time.time()
    try:
        rc = fe.main([*(a.labels or ["--all"]), "--jobs", str(a.jobs), "--tag", F["tag"], "--reuse-stats", *F["args"],
                      *(["--out-root", str(a.out_root)] if a.out_root else [])])
    finally:
        fe.stats_job, fe._fan, fe.summarise_drop, fe.summarise_musall, fe.template_job = saved
    t_main = (time.time() - t0) / 60
    if rc:
        print(f"[o2_frozen] frozen_encoder.main returned {rc} (capture failures above)", flush=True)
    labels = _FAN_STATE.get("labels", [])
    ok = sorted(p.stem for p in (rdir / "inputs").glob("*.pkl"))
    missing = sorted(set(labels) - set(ok))
    if missing:
        raise SystemExit(f"{len(missing)} sessions without inputs: {missing[:8]} (re-run bundle: it resumes)")
    (rdir / "labels.txt").write_text("".join(f"{x}\n" for x in labels), encoding="utf-8", newline="\n")
    # the frame's pooled files the tasks read
    fit = out / "fit"
    shutil.copy2(fit / "templates.pkl", rdir / "fit" / "templates.pkl")
    for f in fit.glob("spatial_*.pkl"):
        shutil.copy2(f, rdir / "fit" / f.name)
    for f in fit.glob("*_joint.npz"):                       # session_vs_frozen reads <an>_joint.npz
        shutil.copy2(f, rdir / "fit" / f.name)
    att = Path(lp_licks.attempts_path())
    shutil.copy2(att, rdir / "tables" / "trial_attempts.csv")
    from wfield_local import combined_encoding as ce
    att_sha = ce.file_sha256(rdir / "tables" / "trial_attempts.csv")
    if "partial" in F["jobs"]:
        import partial_refit as pr
        (rdir / "partial_refit").mkdir(exist_ok=True)
        fp = rdir / "partial_refit" / "fits.pkl"
        if not fp.exists():
            pr.frozen_fits(out, dest=fp)
    env = {}
    for k, v in op.settings().items():
        if k.startswith("O2F_"):
            continue
        if Path(v).is_absolute() and Path(v).is_dir():
            raise SystemExit(f"{k}={v} names a directory: O2 cannot see it")
        if Path(v).is_absolute() and Path(v).is_file():
            shutil.copy2(v, rdir / "tables" / Path(v).name)
            v = f"@tables/{Path(v).name}"
        env[k] = v
    (rdir / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8", newline="\n")
    st = (fit / "pre_stats.pkl").stat()
    rmeta = {"frame": key, "tag": F["tag"], "args": F["args"], "jobs": list(F["jobs"]), "basis": is_basis(key),
             "identity": ident, "run_tag": fe._run_tag(), "v36_run_mode": V36_RUN_MODE[key], "frame_dir": str(out),
             "area_basis": area_b,
             "variants": _variants(out), "svf_variants": [v for v in SVF_VARIANTS if v in _variants(out)],
             "n_sessions": len(labels), "commit": commit, "test_out_root": str(a.out_root or ""), "attempts_table": str(att), "attempts_sha256": att_sha,
             "inputs_from": (_reuse_src()[2] if src_bundle else None),
             "pre_stats": {"size": st.st_size, "mtime": st.st_mtime}, "bundle_main_min": round(t_main, 1),
             "created": datetime.datetime.now().isoformat(timespec="seconds")}
    rmeta_f.write_text(json.dumps(rmeta, indent=1), encoding="utf-8", newline="\n")
    from wfield_local import rebuild_root
    meta.update({"name": a.name, "commit": commit, "code_root": str(repo), "profile": PROFILE,
                 "rebuild_root": str(rebuild_root.root() or ""), o2_bundle.MASK_FIELD: mask_sig,
                 "frames": sorted(set(meta.get("frames", [])) | {key})})
    meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    code = d / "code"
    if not code.exists():
        code.mkdir()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tar = Path(td) / "code.tar"
            subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tar), commit, "wfield_local",
                            "configs", "scripts"], check=True)
            shutil.unpack_archive(str(tar), str(code))
    (d / "frozen_runner.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    rv = PathResolver()
    b = f"{oi.scratch(a.user)}/o2/{a.name}"
    (d / "run_frozen.sbatch").write_text(SBATCH.format(name=a.name, b=b), encoding="utf-8", newline="\n")
    try:
        share_b = oi.o2_path(d, rv)
    except ValueError as ex:                             # a local pilot bundle (--bundle-root off the share): no O2 commands
        print(f"[o2_frozen] no COMMANDS.md: {ex}", flush=True)
    else:
        write_commands(d, meta, b, share_b, a.user)
    print(f"[o2_frozen] {key}: {len(labels)} sessions bundled -> {rdir} (frozen_encoder.main {t_main:.1f} min)")
    return d


def _variants(out: Path) -> list:
    """The frame's fitted variants (fit/<animal>_<variant>.npz of its first animal)."""
    fs = sorted(out.glob("fit/PS92_*.npz"))
    return sorted(f.stem.split("_", 1)[1] for f in fs)


# ------------------------------------------------------------------------------------------------------ O2 task
RUNNER = r'''"""Runs on O2: one session of one frozen frame from the bundle (o2_frozen.py). FRAME + MODE + SLURM_ARRAY_TASK_ID
from the environment; or: python frozen_runner.py FRAME MODE --label LABEL [--out DIR] [--audit]."""
import json, os, sys
from pathlib import Path

b = Path(__file__).resolve().parent
args = sys.argv[1:]
pos = [x for x in args if not x.startswith("--")]
frame = pos[0] if len(pos) > 0 else os.environ["FRAME"]
mode = pos[1] if len(pos) > 1 else os.environ["MODE"]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i] in ("--label", "--out")}
r = b / "runs" / frame
env = json.loads((r / "env.json").read_text())
for k in [k for k in os.environ if k.startswith(("WIDEFIELD_", "WFIELD_", "LOCANMF_", "COMBINED_ENCODING_", "OROFACIAL_",
                                                 "FROZEN_"))]:
    del os.environ[k]                            # ONLY the bundle's settings apply
for k, v in env.items():
    os.environ[k] = str(r / v[1:]) if isinstance(v, str) and v.startswith("@") else v
sys.path[:0] = [str(b / "code"), str(b / "code" / "scripts")]
import o2_frozen
if "--label" in opt:
    label = opt["--label"]
else:
    labels = [x.strip() for x in (r / "labels.txt").read_text().splitlines() if x.strip()]
    label = labels[int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1]
raise SystemExit(o2_frozen.run_task(b, frame, mode, label, out=opt.get("--out"), audit="--audit" in args))
'''

SBATCH = """#!/bin/bash -l
#SBATCH -J frz_{name}
#SBATCH -p short
#SBATCH -c 2
#SBATCH -o {b}/logs/%x_%A_%a.out
#SBATCH -e {b}/logs/%x_%A_%a.err
# FRAME + MODE via sbatch --export=ALL,FRAME=<frame>,MODE=<predict|svf|partial>; --mem / -t on the sbatch line
# (COMMANDS.md). Login shell + conda activate (as run_enc.sbatch).
set -eo pipefail
module load conda/miniforge3/26.9.4-1
conda activate lp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "task $SLURM_ARRAY_TASK_ID FRAME=$FRAME MODE=$MODE on $(hostname)"
python {b}/frozen_runner.py "$FRAME" "$MODE"
"""


def _done_marker(base: Path, mode: str, label: str) -> Path:
    return {"predict": base / "out" / "joint" / "sessions" / f"{label}.json",
            "svf": base / "svf" / "sessions" / f"{label}.json",
            "partial": base / "partial_refit" / "sessions" / f"{label}.json"}[mode]


def run_task(b, frame: str, mode: str, label: str, out=None, audit: bool = False) -> int:
    """O2 (and `check`): identity + masks, replay the session's loaders, run the job type -> its session files under
    runs/<frame>/out (predict), .../svf, .../partial_refit; ``out`` = another base folder (the desktop check)."""
    b = Path(b)
    meta = json.loads((b / "bundle.json").read_text())
    r = b / "runs" / frame
    rmeta = json.loads((r / "run.json").read_text())
    op.verify_identity(rmeta["identity"])
    from wfield_local import mask_keys, o2_bundle
    inp = pickle.loads((r / "inputs" / f"{label}.pkl").read_bytes())
    src = inp.get("inputs_from")
    if src:                                              # loader values of a SIBLING bundle (INPUTS_FROM_ENV)
        sb = b.parent / src["bundle"]
        if not (sb / "runs" / src["frame"] / "inputs" / f"{label}.pkl").exists():
            sb = Path(src["path"])
        full, img = _source_pickles(sb, src["frame"], label, meta[o2_bundle.MASK_FIELD])
        inp = {**full, **{k: inp[k] for k in ("design", "design_sums", "n_frames", "n_cols", "area_masks") if k in inp}}
        print(f"{label}: loader values replayed from {sb} (frame {src['frame']})", flush=True)
    else:
        img = pickle.loads((b / "imaging" / f"{label}.pkl").read_bytes())
    o2_bundle.check_inputs(meta, inp, label)
    o2_bundle.check_inputs(meta, img, label + " (imaging)")
    # 11-26: the task's analysis-mask digest = the bundle's (just verified on every input pickle; the desktop computed it
    # from the masks at bundle time). Recomputing it reads the brain / stat / painted masks off the share, which O2 does
    # not have: a BASIS frame's predict (`_spatial_of` -> check_saved_masks) failed the audited check on exactly those
    # reads (wave-1a2 spont_frA_basis and spont_frA_spd_basis, PS95_0812).
    _sig = meta[o2_bundle.MASK_FIELD]
    mask_keys.signature = lambda *a_, **k_: _sig
    if rmeta.get("area_basis") == "split" and not inp.get("area_masks"):     # as combined_encoding.area_masks_for
        raise o2_bundle.BundleMismatch(f"{label}: split-basis frame but the bundled inputs carry no area_masks; rebuild "
                                       "the bundle -- the split basis cannot be built on O2")
    base = Path(out) if out else r
    if _done_marker(base, mode, label).exists():
        print("done already", label)
        return 0
    if audit:
        op._OWN.extend({str(x).replace("\\", "/").lower() for x in (b, b.resolve(), base, base.resolve())})
        sys.addaudithook(op._audit)
    from wfield_local import config, lp_licks, parallel
    from wfield_local import frozen_encoder as fe
    parallel.pin_blas()                       # as the desktop's fan_sessions workers (BLAS_THREADS): same rounding
    att = r / "tables" / "trial_attempts.csv"
    lp_licks.attempts_path = lambda rv=None: att                  # the shipped gate / class table (write_trials)
    an = config.animal_of(label)
    tp = str(r / "fit" / "templates.pkl")
    spath = str(r / "fit") if rmeta["basis"] else None
    t0 = time.time()
    with op.Replay({**img["memo"], **inp["memo"]}, inp["files"], inp["globs"]),             fe.bundled_area_masks({label: inp["area_masks"]} if inp.get("area_masks") else {}):
        if mode == "partial":                 # partial_refit.job rebuilds the design with patched nulls: check it first
            d0 = fe.session_design(label, fe._templates_of(tp, an))
            dd = op.same_design(d0, inp)
            del d0
            if not dd:
                raise o2_bundle.BundleMismatch(f"{label}: the replayed design differs from the one built at bundle time")
            fits = r / "partial_refit" / "fits.pkl"
            sc = fits.parent / (fits.name + mask_keys.SIDECAR_SUFFIX)
            if not sc.exists() or sc.read_text(encoding="utf-8").strip() != meta[o2_bundle.MASK_FIELD]:
                raise o2_bundle.BundleMismatch(f"{fits}: mask sidecar missing or not the bundle's")
            import partial_refit as pr
            (base / "partial_refit").mkdir(parents=True, exist_ok=True)
            if out:
                shutil.copy2(fits, base / "partial_refit" / "fits.pkl")
                shutil.copy2(sc, base / "partial_refit" / sc.name)
            pr.job((label, str(r), str(base / "partial_refit" / "fits.pkl")))
        else:
            orig, seen = fe.session_design, []

            def checked(*a, **k):             # the task's own design build, checked against the recorded digest
                dsg = orig(*a, **k)
                if not seen:
                    dd_ = op.same_design(dsg, inp)
                    if not dd_:
                        raise o2_bundle.BundleMismatch(f"{label}: the replayed design differs from the bundle's")
                    seen.append(dd_)
                return dsg
            fe.session_design = checked
            try:
                if mode == "predict":
                    fits = pickle.loads((r / "fits" / f"{an}.pkl").read_bytes())
                    fe.predict_job((label, fits, str(base / "out"), tp, spath, False))
                elif mode == "svf":
                    import session_vs_frozen as svf
                    res = svf.self_fit_job((label, str(r), tuple(rmeta["svf_variants"])))
                    sdir = base / "svf" / "sessions"
                    sdir.mkdir(parents=True, exist_ok=True)
                    (sdir / f"{label}.json").write_text(json.dumps(res))
                else:
                    raise SystemExit(f"unknown mode {mode}")
            finally:
                fe.session_design = orig
            dd = seen[0] if seen else "?"
    if audit:
        print(f"share / cache reads outside the bundle during the task: {len(op._SHARE_READS)} "
              f"{sorted(set(op._SHARE_READS))[:20]}")
        if op._SHARE_READS:
            return 3
    print(f"{frame} {mode} {label}: design {dd} ok; done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------- commands
def write_commands(d: Path, meta: dict, b: str, share_b: str, user: str) -> None:
    blocks = []
    for key in meta["frames"]:
        rm = json.loads((d / "runs" / key / "run.json").read_text())
        n = rm["n_sessions"]
        mem_p = "32G" if rm["basis"] else MEM["predict"]
        for mode in rm["jobs"]:
            mem = mem_p if mode == "predict" else MEM[mode]
            blocks.append(f"""### {key} / {mode}  (frame {rm['tag']}; {n} sessions; --mem={mem} -t {TIME[mode]})
Login node, ONE task first (the smoke test of this job type):
```bash
sbatch --array=1 --mem={mem} -t {TIME[mode]} --export=ALL,FRAME={key},MODE={mode} {b}/run_frozen.sbatch
```
then all (task 1 is skipped as done):
```bash
sbatch --array=1-{n}%60 --mem={mem} -t {TIME[mode]} --export=ALL,FRAME={key},MODE={mode} {b}/run_frozen.sbatch
```
Desktop after the rsync back: `python scripts/o2_frozen.py collect --frame {key} --name {meta['name']} --mode {mode}`
""")
    reuse_txt = ""
    for key in meta["frames"]:
        rm = json.loads((d / "runs" / key / "run.json").read_text())
        src = rm.get("inputs_from")
        if not src:
            continue
        sb = f"{b.rsplit('/', 1)[0]}/{src['bundle']}"
        share_sb = f"{share_b.rsplit('/', 1)[0]}/{src['bundle']}"
        reuse_txt += f"""
## Inputs of {key} are REUSED from bundle {src['bundle']} (frame {src['frame']}): the runner reads that bundle's
## runs/{src['frame']}/inputs/<label>.pkl + imaging/<label>.pkl from the SIBLING folder on scratch, {sb}.
## Check it is still there (scratch purge) BEFORE submitting; if not, restore it from the share (transfer node):
```bash
ls {sb}/imaging | wc -l; ls {sb}/runs/{src['frame']}/inputs | wc -l     # both must print {rm['n_sessions']}
# only if missing:
mkdir -p {sb}/runs/{src['frame']}
rsync -ah {share_sb}/imaging/ {sb}/imaging/
rsync -ah {share_sb}/runs/{src['frame']}/inputs/ {sb}/runs/{src['frame']}/inputs/
```
"""
    txt = f"""# O2 frozen-encoder frames: {meta['name']}

code {meta['commit']} ({Path(meta['code_root']).name}); profile {meta['profile']}; analysis masks {meta['mask_sig']}
Per frame: runs/<frame>/env.json (applied before wfield_local is imported; the runner REFUSES unless the bundled code
computes the recorded identity), runs/<frame>/inputs/<label>.pkl + imaging/<label>.pkl (the session's loader values),
runs/<frame>/fits/<animal>.pkl (the frame's fits, built on the desktop), fit/ (templates, spatial basis, joint fits),
tables/ (attempts table + calibration files). No share reads.

## Transfer node  (ssh {user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b}
rsync -ah {share_b}/ {b}/
```
{reuse_txt}## Jobs
""" + "\n".join(blocks) + f"""
## Back to the share (transfer node), per frame when its arrays are done
```bash
rsync -ah {b}/runs/ {share_b}/runs/
```
"""
    (d / "COMMANDS.md").write_text(txt, encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------------------------- collect
def cmd_collect(a) -> int:
    import datetime

    from wfield_local import frozen_encoder as fe
    from wfield_local import mask_keys, o2_bundle, writeguard
    key, mode = a.frame, a.mode
    d = op._bundle_dir(a.name)
    meta = json.loads((d / "bundle.json").read_text())
    r = d / "runs" / key
    rmeta = json.loads((r / "run.json").read_text())
    oroot = declared_out_root(rmeta)      # out_root (rule-11 bundles) or test_out_root (TEST bundles)
    if not oroot:
        raise SystemExit(f"refusing: bundle {a.name} frame {key} declares no out-root (out_root / test_out_root empty; "
                         f"frame_dir {rmeta.get('frame_dir')}); collecting it would route to fe.out_root(), the "
                         "production default -- rebuild the bundle with --out-root (rule 11)")
    resolve_frame_area_basis(key, oroot)  # the frame's recorded basis (identity below), read at the DECLARED root
    try:
        o2_bundle.verify_masks(meta)
        op.verify_identity(rmeta["identity"])
    except o2_bundle.BundleMismatch as ex:
        raise SystemExit(f"refusing: this bundle is not of the current run ({ex})") from None
    # 11-26 (combined 11eb7cbf + ea88178d, Priya): FROZEN_OUT_ROOT = the bundle's DECLARED root (out_root or
    # test_out_root) for EVERY mode, so the svf / partial summaries (`fr_musall.E` = fe.out_root() at THEIR import below)
    # and the predict summaries land beside the frame; apply_frame_env cleared it, and unset they would write
    # <production imaging_encoding>/frozen_encoder_<tag>/... A bundle that declares no root is refused above.
    if "fr_musall" in sys.modules:
        raise SystemExit("fr_musall imported before collect set FROZEN_OUT_ROOT")
    os.environ["FROZEN_OUT_ROOT"] = str(oroot)
    out = frame_out(key, oroot)
    if str(out) != rmeta["frame_dir"]:
        raise SystemExit(f"frame folder {out} differs from the bundle's {rmeta['frame_dir']}")
    st = (out / "fit" / "pre_stats.pkl").stat()
    if st.st_size != rmeta["pre_stats"]["size"] or abs(st.st_mtime - rmeta["pre_stats"]["mtime"]) > 1:
        raise SystemExit(f"{out}/fit/pre_stats.pkl changed since the bundle was built (frame re-fit?)")
    labels = [x.strip() for x in (r / "labels.txt").read_text().splitlines() if x.strip()]
    src = {"predict": r / "out", "svf": r / "svf", "partial": r / "partial_refit"}[mode]
    have = {lab for lab in labels if _done_marker(r, mode, lab).exists()}
    missing = sorted(set(labels) - have)
    if missing and not a.partial:
        raise SystemExit(f"{len(missing)} of {len(labels)} sessions missing: {missing[:8]} (--partial to collect anyway)")
    # the summary root (predict: fe.out_root(); svf / partial: fr_musall.E) = the frame's parent, checked BEFORE any write
    if mode == "predict":
        sroot = fe.out_root()
    else:
        import fr_musall
        sroot = fr_musall.E
    if Path(sroot) / out.name != out:
        raise SystemExit(f"summary root {sroot} is not the frame's parent {out.parent}: refusing")
    dst_root = {"predict": out, "svf": out / "session_vs_frozen", "partial": out / "partial_refit"}[mode]
    writeguard.assert_writable(dst_root)
    files = [f for f in src.rglob("*") if f.is_file() and f.suffix in (".json", ".npz", ".csv")]
    if mode == "partial":
        files = [f for f in files if f.parent.name == "sessions"]
    rel = [(f, dst_root / f.relative_to(src)) for f in files]
    clash = [str(t) for f, t in rel if t.exists() and t.read_bytes() != f.read_bytes()]
    if clash:
        raise SystemExit(f"{len(clash)} destination files exist and differ (move them aside first): {clash[:5]}")
    for f, t in rel:
        t.parent.mkdir(parents=True, exist_ok=True)
        s = f.read_bytes()
        if f.suffix == ".json":               # a doubled task can leave a trailing byte (o2_encoding)
            txt = s.decode("utf-8")
            obj, end = json.JSONDecoder().raw_decode(txt)
            s = json.dumps(obj).encode("utf-8") if txt[end:].strip() else s
        t.write_bytes(s)
    if mode == "partial":
        fits = out / "partial_refit" / "fits.pkl"
        if fits.exists() and fits.read_bytes() != (r / "partial_refit" / "fits.pkl").read_bytes():
            raise SystemExit(f"{fits} exists and differs from the bundle's frozen fits: move it aside first")
        shutil.copy2(r / "partial_refit" / "fits.pkl", fits)
        mask_keys.write_sidecar(fits, meta[o2_bundle.MASK_FIELD])
    (dst_root / f"collected_o2_{a.name}_{mode}.json").write_text(json.dumps(
        {"bundle": str(d), "frame": key, "mode": mode, "commit": meta["commit"], "n_files": len(rel),
         "missing": missing, "collected": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1),
        encoding="utf-8")
    print(f"collected {len(rel)} files -> {dst_root}; summarising", flush=True)
    if mode == "predict":
        if is_basis(key) or "--no-drop-spout" in FRAMES[key]["args"]:
            pass
        else:
            fe.summarise_drop(out)
        fe.summarise_musall(out, rmeta["variants"])
    elif mode == "svf":
        import session_vs_frozen as svf
        svf.main(["--tag", FRAMES[key]["tag"], "--plot-only", "--variants", *rmeta["svf_variants"]])
    else:
        import partial_refit as pr
        pr.main(["--tag", FRAMES[key]["tag"], "--plot-only"])
    return 0


# ------------------------------------------------------------------------------------------------------- check
def cmd_check(a) -> int:
    """The bundled task in a clean subprocess (ONLY env.json + the bundle's code, share reads audited), outputs under
    --out; prints its time. Peak RAM: run it under a monitor (the desktop) or read sacct MaxRSS (O2)."""
    d = op._bundle_dir(a.name)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WIDEFIELD_", "WFIELD_", "LOCANMF_",
                                                                    "COMBINED_ENCODING_", "OROFACIAL_", "FROZEN_",
                                                                    "O2F_"))}
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, str(d / "frozen_runner.py"), a.frame, a.mode, "--label", a.label, "--out",
                        str(a.out), "--audit"], env=env, cwd=str(a.out), stdout=sys.stdout, stderr=sys.stderr)
    print(f"bundled task rc {p.returncode} in {(time.time() - t0) / 60:.1f} min")
    return p.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--frame", required=True, choices=sorted(FRAMES))
    b.add_argument("--name", required=True)
    b.add_argument("--user", required=True)
    b.add_argument("--jobs", type=int, default=3)
    b.add_argument("--out-root", default=None, help="frame folder parent instead of the share's routed root (required "
                   "for a session subset; a full build may declare <bundle>/frames, rule 11)")
    b.add_argument("--bundle-root", default=None, help="parent folder of the bundle instead of <rebuild root>/o2 "
                   "(a local pilot); the O2 commands assume the share's o2 folder")
    b.add_argument("--inputs-from", default=None, help="bundle (name under <rebuild root>/o2, or a path) whose recorded "
                   "loader values are replayed for every pass instead of reading the share (INPUTS_FROM_ENV)")
    b.add_argument("--inputs-frame", default=None, help="frame of --inputs-from to replay (default: this frame's key)")
    b.add_argument("--nose-frames", default="", help="session_poses run folder of a re-triangulated 3-D nose "
                   "(COMBINED_ENCODING_NOSE_FRAMES; default the adopted run)")
    b.add_argument("labels", nargs="*", help="TEST bundles only: a session subset (default --all)")
    c = sub.add_parser("check")
    c.add_argument("--frame", required=True, choices=sorted(FRAMES))
    c.add_argument("--name", required=True)
    c.add_argument("--nose-frames", default="", help="as the bundle's (its run.json run_tag shows it)")
    c.add_argument("--mode", required=True, choices=("predict", "svf", "partial"))
    c.add_argument("--out", required=True)
    c.add_argument("label")
    k = sub.add_parser("collect")
    k.add_argument("--frame", required=True, choices=sorted(FRAMES))
    k.add_argument("--name", required=True)
    k.add_argument("--nose-frames", default="", help="as the bundle's (its run.json run_tag shows it)")
    k.add_argument("--mode", required=True, choices=("predict", "svf", "partial"))
    k.add_argument("--partial", action="store_true", help="collect although some sessions are missing")
    a = ap.parse_args(argv)
    apply_frame_env(a.frame, getattr(a, "nose_frames", "") or "")   # BEFORE wfield_local's env-reading modules
    if a.cmd == "bundle":
        cmd_bundle(a)
        return 0
    return cmd_check(a) if a.cmd == "check" else cmd_collect(a)


if __name__ == "__main__":
    raise SystemExit(main())
