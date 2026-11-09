"""The pre-stroke decoder and encoder as STORED OBJECTS, not recipes re-executed nightly.

WHY (Priya, 2026-08-27, "let's freeze the decoder"). Until now nothing in this repo persisted a
fitted frozen model. `pooled_frozen_loso` fitted `_pipe()` inline on every call, `pooled_frozen_encoder`
fitted its Ridge inline, and every JSON on the server held RESULT NUMBERS -- accuracies, confusions,
EV -- with no weights behind them. `save_session_decoder` is the single exception and nothing ever
loads its `.joblib` back. "The frozen pre-stroke decoder" was a recipe, not an object.

THAT IS WHY THE CONTAMINATION WAS POSSIBLE. On 2026-08-26 the frozen models were found to have been
training on post-stroke sessions -- by 8/26 roughly 30-39% of the training data -- because the code
was written when "curated" meant pre-stroke and post-stroke nights joined the pool silently. A model
refitted every run has no identity to check: there is nothing to ask "what were you trained on?".
A stored model carries its own answer, and adding a session cannot retroactively change what an
already-frozen model saw. Freezing is not only a time saving; it removes the class of bug.

WHAT DETERMINES A MODEL'S IDENTITY -- and what, importantly, does NOT

    animal            one model per animal; the bases and feature spaces are per-animal
    kind              decoder | encoder
    align             precue (ENL) | cue | lick
    source            roi | locanmf, plus the joint BASIS ID when features come from one
    post_s, zscore    window and per-session scaling
    alpha             encoder only
    train_labels      the PRE-STROKE session set, each with its input signature

TRIAL INCLUSION IS NOT IN THE SPEC, and this is the part worth being explicit about because it looks
like it should be. Training always uses the pre-stroke ENGAGED trials (`XE`); "all", "lick + miss
while working" and "lick only" select which POST-STROKE trials are pushed through the finished model.
They are scoring-time populations, not training-time variants. So the inventory is
4 animals x 3 alignments x 2 sources = 24 decoders and 24 encoders, not 72 of each -- and because
every population is scored by the SAME model, per-class results stay summable, which is the property
`position_coding_directions` already relies on for its raw-count confusions.

A CHANGED PRE-STROKE SET MINTS A NEW MODEL; IT NEVER MUTATES THE OLD ONE. `spec_id` hashes the
training set and its input signatures, so a new pre-stroke session, a re-curation, or an upstream
re-preprocess (the 2026-08-14 SVTcorr flip changed the DATA under unchanged labels) lands in a NEW
directory. Nothing is deleted, and `load_or_fit` reports a spec change LOUDLY rather than resolving
it: silently refitting would move a reference that post-stroke results are already quoted against,
and silently reusing would score today's data against a model built from data that is gone. Same
contract as `prestroke_reference`, whose reasoning this module follows for an estimator rather than
a JSON payload.
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path

from wfield_local import config

#: Bump when the MEANING of a frozen model changes -- when the fitting logic is altered such that an
#: old artifact, though built from the same sessions, no longer answers the same question. Mtimes
#: cannot see a code change; this is the lesson `session_cache.CACHE_VERSION` records.
FREEZE_VERSION = 2          # 2 = training-data fingerprint + selection + hemo variant in every spec (2026-11-03)
#: LP_licks only: "lp" freezes LP-TRAINED sensitivity models under their own spec (see `load_or_fit`).
LP_FROZEN_ENV = "WIDEFIELD_LP_FROZEN"

STATUS_NEW, STATUS_HIT, STATUS_SPEC_CHANGED = "frozen-new", "frozen-hit", "SPEC-CHANGED"

#: Values of a per-family opt-out env var (WIDEFIELD_<FAMILY>_FROZEN) that turn the stored default OFF (11-02 pass 2).
OPT_OUT_VALUES = ("0", "false", "no", "off")


def family_on(env_name) -> bool:
    """True unless the family's opt-out env var is 0 / false / no / off: the stored model is the DEFAULT (rule 10).

    One definition for every family converted on 11-02 (`docs/status/FROZEN_ALL_20261102.md` pass 2); the opt-out
    exists only to reproduce an old in-memory render."""
    return os.environ.get(env_name, "").strip().lower() not in OPT_OUT_VALUES


def daq_trained_in_lp() -> bool:
    """True under WIDEFIELD_LICK_SOURCE=lp ONLY with WIDEFIELD_LP_FROZEN=daq (the opt-in cross-check): the LP arm then
    LOADS the DAQ-trained model (fitted by a DAQ run on the DAQ arm's pre set) and refuses to fit one (`load_or_fit`).

    PRIYA 11-03: the LP arm's DEFAULT is the LP-TRAINED model (trained on the LP arm's own pre trials; spec carries
    lick_source=lp). Until 11-03 (FROZEN_DEFAULT / FROZEN_ALL 11-02) the default was DAQ-trained and "lp" opted in;
    "lp" is still accepted and now simply names the default."""
    from wfield_local import lp_licks
    return lp_licks.is_lp() and os.environ.get(LP_FROZEN_ENV, "").strip().lower() == "daq"


def lp_trained() -> bool:
    """True under WIDEFIELD_LICK_SOURCE=lp unless WIDEFIELD_LP_FROZEN=daq: the spec carries lick_source=lp."""
    from wfield_local import lp_licks
    return lp_licks.is_lp() and not daq_trained_in_lp()


def _stat_sig(path) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "MISSING"


def _session_sig(label) -> str:
    """Signature of the INPUTS one training session contributes.

    Labels alone are not enough and this repo has the scar: on 2026-08-14 the switch to the
    meegkit_hpfit SVTcorr changed the underlying data while every label stayed identical.
    """
    sess = {s["label"]: s for s in config.load_sessions()}
    s = sess.get(label)
    if s is None:
        return "UNKNOWN"
    ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
    u_atlas = f"{ad[0]}/U_atlas.npy" if ad else ""
    return f"{_stat_sig(u_atlas)}|{_stat_sig(config.svtcorr_path(s['mc']))}"


_MISSING = object()


def data_fingerprint(obj) -> str:
    """sha1[:16] of training data: arrays by dtype / shape / bytes, sequences in order, dicts by sorted key, scalars by
    repr. Deterministic across processes (never hashes object ids)."""
    import numpy as np
    h = hashlib.sha1()

    def walk(o):
        if isinstance(o, np.ndarray) or (hasattr(o, "__array__") and not isinstance(o, (str, bytes))):
            a = np.ascontiguousarray(np.asarray(o))
            if a.dtype == object:
                h.update(f"O|{a.shape}|".encode())
                for x in a.ravel():
                    walk(x)
                return
            h.update(f"A|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())
        elif isinstance(o, dict):
            h.update(f"D{len(o)}|".encode())
            for k in sorted(o, key=repr):
                h.update(f"k{k!r}|".encode())
                walk(o[k])
        elif isinstance(o, (list, tuple)):
            h.update(f"L{len(o)}|".encode())
            for x in o:
                walk(x)
        else:
            h.update(f"S{o!r}|".encode())

    walk(obj)
    return h.hexdigest()[:16]


#: Fields `daq_trained_in_lp` ignores when it looks up the DAQ-trained model: the LP run cannot rebuild the DAQ run's
#: training arrays, so it matches the DAQ model on everything else (`find_daq_trained`). SELECTION IS MATCHED (review
#: M7, 11-03): a camera-gate (or gate-LP) cross-check must never load a DAQ model frozen under another selection.
_DAQ_LOOKUP_IGNORES = ("train_fp",)
DAQ_RUN_FP = "from-daq-run"


def make_spec(animal, kind, *, align, source, train_labels, post_s=2.0, zscore=True,
              basis_id=None, alpha=None, n_features=None, scaling=None, scaling_params=None, train_data=_MISSING,
              selection=None, lp_only=False):
    """The full identity of one frozen model, as a plain dict (JSON-serialisable, order-stable).

    ``scaling`` (2026-11-02) names HOW the features were scaled when ``zscore`` alone does not say it. Added because
    the state decoder's spec recorded ``zscore=True`` while its features were never z-scored per session
    (`locomotor_decoder.by_animal_day`): the fix (scaling="session_z") must not load the old, unscaled models filed
    under the same identity. ADDED ONLY WHEN GIVEN, so every existing spec -- and its id -- is byte-identical.
    ``scaling_params`` (2026-11-03) records the scaling's own settings (bal_em: passes and the min-mass guard, which
    is recorded even when OFF). Also added only when given.

    TRAINING-DATA FINGERPRINT (FREEZE_VERSION 2, cohort-1 rebuild 2026-11-03, Priya: yes). ``train_data`` is REQUIRED:
    the actual training arrays (X / y / groups, in training order, or a data signature the caller already computes)
    -> ``train_fp`` = `data_fingerprint`. Also always carried: ``selection`` (default
    `trial_classes.run_selection()`, the run's trial-selection tag) and ``hemo_variant`` (`config.hemo_variant()`).
    Why: the spec hashed labels + SVTcorr mtime:size only, so a changed trial set under unchanged labels (PS92 section
    G, 11-02 21:23) and a partial hemo switch (trap T4) both served models trained on other data. This changes EVERY id
    by design: the rebuild fits into a NEW store (`_write` refuses the legacy one). Under `daq_trained_in_lp` the LP
    run cannot rebuild the DAQ run's arrays, so ``train_fp`` = DAQ_RUN_FP and `load_or_fit` finds the DAQ-trained model
    by everything else (`find_daq_trained`).

    ``lp_only=True`` (review H1, 11-03): a kind that is LP-only BY NATURE (video-defined training rows: attempt_axes,
    iti_readout). The spec always carries lick_source=lp and ALWAYS fingerprints the real rows, even under
    WIDEFIELD_LP_FROZEN=daq -- there is no DAQ-trained model to look up, and the constant DAQ_RUN_FP would let a later
    run HIT a model fitted on other rows."""
    if train_data is _MISSING:
        raise TypeError(f"make_spec({kind!r}): train_data is required (training-data fingerprint, FREEZE_VERSION "
                        f"{FREEZE_VERSION}; docs/FROZEN_MODELS.md)")
    labs = sorted(train_labels)
    # ONE ANIMAL PER MODEL, checked rather than assumed. Every caller pools per animal, but the animal
    # is the first field of the identity and a pooled-across-animals training set would produce a model
    # filed under one animal's name that had seen another's data -- the exact shape of label-asserts-a-
    # property-nothing-verifies that this module exists to end.
    others = {config.animal_of(x) for x in labs} - {animal}
    if others:
        raise ValueError(f"frozen model for {animal} was given training sessions from "
                         f"{sorted(others)}: {[x for x in labs if config.animal_of(x) != animal]}")
    # ENL WINDOW (WIDEFIELD_ENL_WINDOW_S, 2026-10-21): a pre-cue model's window IS its post_s, and the features it is
    # applied to are built with the overridden window (`lp_licks.apply_enl_window`) -- so the spec must carry it, or a
    # 1.0 s run would load the 2.0 s model. The matched DAQ 1.0 s run freezes these; LP_licks reuses them.
    # 11-02 pass 2: ALSO an align that STARTS WITH "precue-" (the pass-2 kinds put variant fields after the window name:
    # nolick "precue-rt2-f..", coding "precue-dom+dom_orth", section G "precue-keep..", attempt "precue-dom"). Without
    # it a WIDEFIELD_ENL_WINDOW_S=1.0 run HIT the 2.0 s model (found by the nolick ENL 1.0 render: precue cells differed
    # stored vs in memory). Ids without the override (the default DAQ 2.0 s runs) are unchanged.
    from wfield_local import lp_licks
    if (align == "precue" or str(align).startswith("precue-")) and lp_licks.enl_window_s() is not None:
        post_s = lp_licks.enl_window_s()
    spec = {
        "freeze_version": FREEZE_VERSION,
        "animal": animal,
        "kind": kind,
        "align": align,
        "source": source,
        "basis_id": basis_id,
        "post_s": float(post_s),
        "zscore": bool(zscore),
        "alpha": None if alpha is None else float(alpha),
        "n_features": None if n_features is None else int(n_features),
        "train_labels": labs,
        "train_sigs": {lab: _session_sig(lab) for lab in labs},
    }
    if scaling is not None:
        spec["scaling"] = str(scaling)
    if scaling_params is not None:
        spec["scaling_params"] = {str(k): (None if v is None else float(v) if isinstance(v, float) else v)
                                  for k, v in sorted(dict(scaling_params).items())}
    # LP-TRAINED models (the LP arm's DEFAULT since 11-03, Priya; see `daq_trained_in_lp`): lick_source is ADDED ONLY
    # THEN, so every DAQ spec -- and its id, which names the stored directory -- is byte-identical to before, and the
    # LP-trained ids are the same ids the 11-02 WIDEFIELD_LP_FROZEN=lp runs froze.
    if lp_trained() or lp_only:
        spec["lick_source"] = "lp"
    from wfield_local import trial_classes
    spec["hemo_variant"] = config.hemo_variant() or "zerophase"
    spec["selection"] = str(selection) if selection is not None else trial_classes.run_selection()
    spec["train_fp"] = DAQ_RUN_FP if (daq_trained_in_lp() and not lp_only) else data_fingerprint(train_data)
    return spec


def _daq_lookup_selection(sel):
    """The selection as the DAQ-trained lookup compares it (review N2, 11-03): the camera-vs-DAQ rule and its options are
    matched (M7), but NOT the gate-LP suffix -- the DAQ run that froze the model never runs the LP-inclusive gate, so an
    LP run with WIDEFIELD_GATE_SOURCE=lp must still find the model frozen under ``sel-daq-quittail``."""
    s = str(sel) if sel is not None else sel
    return s[: -len("-gateLP")] if isinstance(s, str) and s.endswith("-gateLP") else s


def base_id(spec) -> str:
    """spec_id without the fields a DAQ-trained lookup from the LP arm cannot know (`_DAQ_LOOKUP_IGNORES`), with the
    selection normalized by `_daq_lookup_selection`."""
    d = {k: v for k, v in spec.items() if k not in _DAQ_LOOKUP_IGNORES}
    if "selection" in d:
        d["selection"] = _daq_lookup_selection(d["selection"])
    return spec_id(d)


def spec_id(spec) -> str:
    return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _slug(spec) -> str:
    """Readable directory name. The id is what makes it correct; the prefix is for a human."""
    src = spec["source"] if not spec.get("basis_id") else f"joint{spec['basis_id'][:6]}"
    return f"{spec['kind']}_{spec['align']}_{src}_{spec_id(spec)}"


def local_dir() -> Path:
    """Where frozen models live on THIS machine (override: WIDEFIELD_FROZEN_MODEL_DIR).

    Derived from the machine's own working-figure root rather than a literal, for the reason
    `joint_locanmf._basis_dir` and `session_cache` were both fixed: a drive-letter default is only
    correct on the machine it was written on.
    """
    env = os.environ.get("WIDEFIELD_FROZEN_MODEL_DIR")
    if env:
        return Path(env)
    from wfield_local import rebuild_root
    if rebuild_root.active():                            # a rebuild's store lives in its rebuild root, never beside
        return rebuild_root.root() / (rebuild_root.spec().get("stores") or {}).get("frozen_models", "frozen_models")
    # SIBLING OF THE JOINT BASES, deliberately, rather than a second literal fallback of its own.
    # Both are machine-local derived artifacts rooted at the working-figure parent, and deriving this
    # one from that one means there is a single place where "where does derived data live on this
    # box" is answered -- and no drive-letter literal here for `test_no_hardcoded_machine_paths` to
    # have to be widened for.
    from wfield_local import joint_locanmf

    return Path(joint_locanmf.BASIS_DIR).parent / "frozen_models"


def server_dir():
    """Where frozen models are published on MICROSCOPE, or None if the share is unreachable.

    READ-ONLY from here, exactly as `joint_locanmf.server_basis_dir` is: fitting is machine-local and
    publishing is a separate, byte-verified act.
    """
    try:
        return Path(config.resolver().root("cohort_out")) / "frozen_models"
    except Exception:                                    # noqa: BLE001
        return None


def _roots():
    return [(local_dir(), "local"), (server_dir(), "server")]


_ANNOUNCED: set[str] = set()


def legacy_local_dir() -> Path:
    """The store a PREVIOUS derivation of `local_dir` would have used.

    DERIVED FROM `joint_locanmf.FALLBACK_BASIS_DIR`, NOT REPEATED AS A LITERAL. `local_dir` is that
    directory's sibling, so any box that fitted models while its machine profile had no
    `figures_working` put them there -- and deriving it means the pointer follows if that fallback
    ever moves. (It also keeps the drive-letter literal in the one module
    `tests/test_no_hardcoded_machine_paths.py` sanctions for it, which is not an accident: that
    test exists because a literal is only correct on the machine it was written on, and this whole
    incident was a derived path moving out from under a store.)
    """
    from wfield_local import joint_locanmf

    return Path(joint_locanmf.FALLBACK_BASIS_DIR).parent / "frozen_models"


def warn_if_store_moved(log=print) -> bool:
    """Say so, once per process, when `local_dir()` does not exist. Returns True if it warned.

    **A MISS AND A MOVED STORE LOOK IDENTICAL TO `load_or_fit`, AND THIS COST TWO WEEKS.** On
    2026-09-18, commit 889c5e0 gave the analysis desktop its own machine profile and set
    `figures_working` for it. `local_dir()` is derived from that root, so it moved from
    `C:/wf_local/frozen_models` -- where 48 models had been fitted on 2026-08-28 -- to a path that
    did not exist. `find()` started returning None, `load_or_fit` did exactly what it is supposed
    to do on a miss (refit and store), and nothing anywhere said a word. Every night rebuilt the
    models it already had, and the SPEC-CHANGED warning could not fire for a set of models the
    code could no longer see, so a change to the pre-stroke training set would have gone
    unannounced too.

    Deriving the store path from config is still right -- a drive-letter literal is only correct
    on the machine it was written on. What was missing is that a DERIVED path can move, and an
    absent root is categorically different from an empty one: an empty store means "nothing frozen
    yet", an absent one means "you are not looking where the models are".
    """
    root = local_dir()
    if root.exists() or "moved" in _ANNOUNCED:
        return False
    _ANNOUNCED.add("moved")
    log(f"[frozen] !! the local model store {root} DOES NOT EXIST. Every lookup will miss and "
        f"every model will be refitted -- and a changed training set cannot be reported, because "
        f"`siblings` has nothing to compare against.")
    legacy = legacy_local_dir()
    if legacy.exists() and legacy != root:
        n = len(list(legacy.rglob("manifest.json")))
        log(f"[frozen]    {n} model(s) are sitting in {legacy}, the path this box used "
            f"before its machine profile set `figures_working`. Publish them "
            f"(`WIDEFIELD_FROZEN_MODEL_DIR={legacy} python -m wfield_local.publish_basis "
            f"--what frozen`) or point at them with that variable. Do NOT delete them: they are "
            f"the reference whatever was scored before the move was scored against.")
    return True


def find(spec):
    """``(dir, origin)`` for a model matching this EXACT spec, or ``(None, None)``.

    Local is preferred when both have it, because the local one is not read over SMB.

    **A SPEC_ID IS A HASH OF THE INPUTS, NOT OF THE FITTED MODEL, AND THE DIFFERENCE IS MEASURABLE
    ACROSS MACHINES.** This docstring used to say "two directories with the same id hold the same
    model". They hold a model fitted from the same inputs, which is not the same claim. Measured
    2026-09-21 over all 48 models, comparing the analysis desktop's 2026-08-28 fits against the
    behavior box's: coefficients agree to a cosine of 0.99997 but differ by up to 0.54% of the
    largest coefficient, and two `LogisticRegression` pipelines for one spec disagreed on 0.7% of
    random probes. `lbfgs` stops at ``tol=1e-4``, and a different BLAS build reaches that tolerance
    by a different path -- 589 iterations against 620 for the same spec.

    Consequences worth keeping straight:

    - **Within a machine the fit IS bitwise reproducible** (`tests/test_parallel_results_are_
      ordered.py` pins it across BLAS thread counts), so a re-run reproduces a re-run.
    - **Across machines it is reproducible only to the optimizer's tolerance.** Preferring local
      over server therefore means two boxes can legitimately be using slightly different models
      for the same spec. Publishing so that every box reads ONE store is what removes that, and is
      why the whole set was published on 2026-09-21.
    - `publish_tree`'s MISMATCH is therefore a REAL signal, not pickle noise: publishing the same
      spec from a second machine would report it, and it would be right to.
    """
    want = _slug(spec)
    for root, origin in _roots():
        if root is None:
            continue
        d = Path(root) / spec["animal"] / want
        if (d / "manifest.json").exists():
            return d, origin
    return None, None


def _manifest_mask_sig():
    """The analysis-mask digest a new manifest records (`mask_keys.signature`; None when it cannot be computed)."""
    try:
        from wfield_local import mask_keys
        return mask_keys.signature()
    except Exception:                                    # noqa: BLE001
        return None


def _daq_mask_mismatch(manifest) -> str | None:
    """Why a stored DAQ-trained model may NOT serve the LP cross-check on mask grounds, or None (11-09).

    THE CONSTANT DAQ_RUN_FP HOLE. `find_daq_trained` matches on every spec field except ``train_fp``, so the training
    fingerprint -- the one field that sees the masks (features on the stat mask, CD's occluded-component drop) -- is
    never compared: a DAQ model frozen in the rebuild before the glue-mask fix (glue IN) would serve a glue-out LP run.
    The manifest (not the spec: ids stay byte-identical) records ``mask_sig`` since 11-09; it must equal the current
    masks. A manifest without it is refused inside a rebuild root and accepted at the default root."""
    from wfield_local import mask_keys
    try:
        mask_keys.check(manifest.get(mask_keys.FIELD), "manifest.json")
    except RuntimeError:
        have = manifest.get(mask_keys.FIELD)
        return f"masks {have}" if have else "no mask_sig: frozen before the 11-09 glue-mask fix"
    return None


def find_daq_trained(spec):
    """``(dir, origin)`` of THE DAQ-trained model the LP arm's cross-check loads (``train_fp`` = DAQ_RUN_FP): the stored
    model whose spec equals ``spec`` on every field except `_DAQ_LOOKUP_IGNORES`. Raises if more than one distinct
    training fingerprint matches (ambiguous: two DAQ runs froze different training sets under the same identity)."""
    want = base_id(spec)
    prefix = _slug(spec).rsplit("_", 1)[0] + "_"
    hits, stale = {}, []
    for root, origin in _roots():
        if root is None:
            continue
        d = Path(root) / spec["animal"]
        if not d.exists():
            continue
        for p in sorted(d.glob(f"{prefix}*")):
            if ".SUPERSEDED_" in p.name or not (p / "manifest.json").exists():
                continue
            try:
                man = json.loads((p / "manifest.json").read_text())
                ms = man.get("spec") or {}
            except Exception:                            # noqa: BLE001
                continue
            if ms.get("train_fp") in (None, DAQ_RUN_FP) or base_id(ms) != want:
                continue
            why = _daq_mask_mismatch(man)
            if why:
                stale.append(f"{p} ({why})")
                continue
            hits.setdefault(ms["train_fp"], (p, origin))
    if not hits and stale:
        raise RuntimeError(f"[frozen] {spec['animal']} {spec['kind']}/{spec['align']}: the only DAQ-trained model(s) "
                           f"matching the LP cross-check were trained on other analysis masks: {stale}. Re-freeze the "
                           f"DAQ run first (GLUE_MASK_FIX 2026-11-09).")
    if len(hits) > 1:
        raise RuntimeError(f"[frozen] {spec['animal']} {spec['kind']}/{spec['align']}: {len(hits)} DAQ-trained models "
                           f"match the LP cross-check (training fingerprints {sorted(hits)}); point at one by id")
    return next(iter(hits.values())) if hits else (None, None)


def siblings(spec):
    """Every stored model for the same (animal, kind, align, source, post_s, lick_source), newest last.

    What makes a SPEC CHANGE visible. Without this, a changed pre-stroke set is indistinguishable
    from a first run -- both simply find nothing -- and the reference would move with no one
    noticing, which is the failure this module exists to prevent.

    WINDOW AND LICK SOURCE ARE PART OF THE GROUP (11-02). The directory prefix carries only kind / align / source, so a
    1.0 s ENL model and the 2.0 s one, or the LP-trained model and the DAQ-trained one, used to be each other's
    "siblings": the first fit of either reported SPEC-CHANGED ("11 sessions -> 11", nothing added or removed) although
    they are different references by design, and `refreeze` would have retired the other one. They are now compared on
    the manifest's `post_s` / `lick_source` as well (a DAQ spec has no `lick_source`; a manifest without `post_s` stays
    a sibling). Warning / supersede scope only: no spec, spec_id or stored model changes.
    """
    prefix = _slug(spec).rsplit("_", 1)[0] + "_"
    want_post, want_src = spec.get("post_s"), spec.get("lick_source")
    out = []
    seen = set()
    for root, origin in _roots():
        if root is None:
            continue
        d = Path(root) / spec["animal"]
        if not d.exists():
            continue
        for p in sorted(d.glob(f"{prefix}*")):
            if not (p / "manifest.json").exists() or p.name in seen:
                continue
            try:
                m = json.loads((p / "manifest.json").read_text())
            except Exception:                            # noqa: BLE001
                continue                                 # a half-copied model is not a candidate
            ms = m.get("spec", {}) or {}
            if ms.get("lick_source") != want_src or ("post_s" in ms and want_post is not None
                                                     and float(ms["post_s"]) != float(want_post)):
                continue                                 # another window / lick source: a different reference
            seen.add(p.name)
            out.append({"dir": p, "origin": origin, "frozen_utc": m.get("frozen_utc", ""),
                        "spec": m.get("spec", {}), "meta": m.get("meta", {})})
    return sorted(out, key=lambda r: r["frozen_utc"])


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def supersede(d: Path, reason: str) -> Path:
    """Retire a frozen model by RENAMING it, never by deleting or overwriting.

    The old weights are what an earlier deck was scored against, so they stay reachable; ``reason``
    goes in the name because a superseded model with no recorded reason is indistinguishable from a
    stray directory. Matches the convention already on disk for the nolick reference.
    """
    d = Path(d)
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in reason)[:60] or "unspecified"
    dest = d.with_name(f"{d.name}.SUPERSEDED_{safe}_{_stamp()}")
    d.rename(dest)
    return dest


def load_or_fit(spec, fit, *, meta=None, refreeze=None, log=print):
    """Return ``(payload, status)``: the stored model for this spec, fitting it only the first time.

    ``fit`` is a zero-argument callable returning a picklable payload -- for the decoder, a dict of
    ``{"full": pipeline, "loso": {held_out_label: pipeline}}`` -- called ONLY when no model for this
    exact spec exists. ``meta`` is recorded alongside for provenance and is not part of the identity.

    Status is ``frozen-new`` (fitted and stored), ``frozen-hit`` (loaded verbatim, the fast path), or
    ``SPEC-CHANGED`` -- a model exists for this animal/kind/align/source but was frozen against a
    DIFFERENT pre-stroke input set. In that case a NEW model is fitted and stored under its own id,
    the old one is left untouched, and the change is reported loudly: the caller can then decide
    whether today's set or the stored one is the right reference, and `supersede` retires the loser
    by name. Passing ``refreeze='<reason>'`` retires the existing model deliberately.
    """
    d, origin = find(spec)
    if d is None and spec.get("train_fp") == DAQ_RUN_FP:
        d, origin = find_daq_trained(spec)
    if d is None:
        # A MISS MAY MEAN THE STORE MOVED RATHER THAN "NEW SPEC". See `warn_if_store_moved`.
        warn_if_store_moved(log)
    if d is not None and not refreeze:
        try:
            payload = _read(d)
            if origin == "server":
                log(f"[frozen] {spec['animal']} {_slug(spec)} served from MICROSCOPE")
            return payload, STATUS_HIT
        except Exception as ex:                          # noqa: BLE001
            log(f"[frozen] !! {d.name} unreadable ({type(ex).__name__}); refitting")

    # LP_licks REUSES THE DAQ-FROZEN MODELS (Priya 10-21: no new frozen set; pre-stroke LP and DAQ trials agree except
    # PS93 far_ipsi / far_mid). Under WIDEFIELD_LICK_SOURCE=lp a MISS must therefore never fit: the fit would see
    # LP-gated pre-stroke features and file them under a DAQ spec. Only an explicit LP-trained spec (the PS93
    # sensitivity, WIDEFIELD_LP_FROZEN=lp -> make_spec adds lick_source) may be fitted here. SINCE 11-03 the LP-trained
    # spec is the LP arm's DEFAULT; this refusal now guards only the DAQ-trained cross-check (WIDEFIELD_LP_FROZEN=daq).
    from wfield_local import lp_licks
    if lp_licks.is_lp() and spec.get("lick_source") != "lp":
        raise RuntimeError(f"[frozen] the LP arm's DAQ-trained cross-check ({LP_FROZEN_ENV}=daq) loads DAQ-frozen models, "
                           f"but none is stored for {_slug(spec)} ({spec['animal']}); run the DAQ path first, or unset "
                           f"{LP_FROZEN_ENV} for the LP-trained default")
    prior = [s for s in siblings(spec) if s["dir"].name != _slug(spec)]
    if refreeze:
        for s in prior:
            old = supersede(s["dir"], refreeze)
            log(f"[frozen] superseded {s['dir'].name} -> {old.name} ({refreeze})")
    elif prior:
        was = prior[-1]["spec"].get("train_labels", [])
        now = spec["train_labels"]
        added, gone = sorted(set(now) - set(was)), sorted(set(was) - set(now))
        log(f"[frozen] !! {spec['animal']} {spec['kind']}/{spec['align']}: the stored model was "
            f"frozen against a DIFFERENT pre-stroke input set "
            f"({len(was)} sessions -> {len(now)}"
            + (f"; added {added}" if added else "")
            + (f"; removed {gone}" if gone else "")
            + "). A NEW model is being frozen under its own id and the old one is untouched -- "
              "nothing already published is rescored. If the OLD set is the right reference, point "
              "at it by id; if the new one is, retire the old with refreeze='<reason>'.")

    payload = fit()
    _write(spec, payload, meta or {}, log=log)
    return payload, (STATUS_SPEC_CHANGED if (prior and not refreeze) else STATUS_NEW)


def _read(d: Path):
    import joblib

    return joblib.load(Path(d) / "model.joblib")


class LegacyStoreError(SystemExit):
    """A SystemExit on purpose: many collectors wrap each animal in ``except Exception`` and print, which would turn a
    refused write into a silently missing animal. This stops the run instead."""


def _store_versions(root: Path, animal: str) -> set:
    out = set()
    d = Path(root) / animal
    if not d.exists():
        return out
    for p in d.glob("*/manifest.json"):
        try:
            out.add(int((json.loads(p.read_text()).get("spec") or {}).get("freeze_version", 1)))
        except Exception:                                # noqa: BLE001
            continue
    return out


def _refuse_legacy_store(spec) -> None:
    """A fingerprinted (FREEZE_VERSION >= 2) model is never written into a store that holds older-version models
    (the pre-rebuild store, e.g. C:/Users/SabatiniLab/frozen_models): the rebuild freezes into a FRESH store named
    with WIDEFIELD_FROZEN_MODEL_DIR (rebuild order G3), so the two generations can never be confused or superseded
    into each other. Raises; the existing models are untouched."""
    old = {v for v in _store_versions(local_dir(), spec["animal"]) if v < int(spec.get("freeze_version", 1))}
    if old:
        raise LegacyStoreError(f"[frozen] refusing to freeze a v{spec.get('freeze_version')} (fingerprinted) model into "
                           f"{local_dir()}, which holds v{sorted(old)} models for {spec['animal']}: set "
                           "WIDEFIELD_FROZEN_MODEL_DIR to a fresh store (docs/FROZEN_MODELS.md section 8)")


def _write(spec, payload, meta, log=print):
    import joblib

    from wfield_local import writeguard

    d = local_dir() / spec["animal"] / _slug(spec)
    writeguard.assert_writable(d.parent)
    _refuse_legacy_store(spec)
    d.mkdir(parents=True, exist_ok=True)
    # CONCURRENCY, since 2026-08-28: stages now fan out over animals and sessions, and two workers
    # that both miss the same spec will both fit and both land here. `session_cache` solved this
    # already and this is the same solution -- write to a PID-named temp, publish with `os.replace`,
    # and treat a failed publish as "someone else already wrote the same bytes". The payload is a
    # function of the spec, so their copy is equivalent to ours by construction.
    #
    # BOTH FILES, not just the model. `manifest.json` was a bare `write_text`: a reader arriving
    # mid-write got a truncated file, and a manifest is what `find`/`listing` read to answer "what
    # was this trained on" -- the question the freeze exists to make answerable.
    man = json.dumps({"spec": spec, "spec_id": spec_id(spec),
                      "frozen_utc": dt.datetime.utcnow().isoformat(), "meta": meta,
                      "mask_sig": _manifest_mask_sig()},  # provenance + the DAQ_RUN_FP lookup's mask check (11-09)
                     indent=2, default=str)
    for name, dump in (("model.joblib", lambda t: joblib.dump(payload, t)),
                       ("manifest.json", lambda t: t.write_text(man, encoding="utf-8"))):
        tmp = d / f"{name}.{os.getpid()}.tmp"
        dump(tmp)
        try:
            os.replace(tmp, d / name)
        except OSError:
            # Windows refuses the replace while another process holds the destination open for
            # reading. The destination is already correct; drop our copy rather than fail a run.
            try:
                tmp.unlink()
            except OSError:
                pass
    log(f"[frozen] {spec['animal']} {_slug(spec)}: FROZEN on {len(spec['train_labels'])} "
        f"pre-stroke session(s)")
    return d


def listing(animal=None):
    """Every stored frozen model, newest last -- provenance when a number is questioned."""
    out = []
    animals = set()
    for root, _o in _roots():
        if root is None or not Path(root).exists():
            continue
        animals |= {p.name for p in Path(root).iterdir() if p.is_dir()}
    for an in sorted(a for a in animals if animal is None or a == animal):
        for root, origin in _roots():
            if root is None:
                continue
            d = Path(root) / an
            if not d.exists():
                continue
            for p in sorted(d.glob("*")):
                man = p / "manifest.json"
                if not man.exists():
                    continue
                try:
                    m = json.loads(man.read_text())
                except Exception:                        # noqa: BLE001
                    continue
                sp = m.get("spec", {})
                out.append({"animal": an, "kind": sp.get("kind"), "align": sp.get("align"),
                            "source": sp.get("source"), "basis_id": sp.get("basis_id"),
                            "n_train": len(sp.get("train_labels", [])),
                            "spec_id": m.get("spec_id"), "frozen_utc": m.get("frozen_utc", ""),
                            "origin": origin, "superseded": ".SUPERSEDED_" in p.name})
    return sorted(out, key=lambda r: r["frozen_utc"])
