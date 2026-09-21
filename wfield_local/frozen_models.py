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
FREEZE_VERSION = 1

STATUS_NEW, STATUS_HIT, STATUS_SPEC_CHANGED = "frozen-new", "frozen-hit", "SPEC-CHANGED"


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


def make_spec(animal, kind, *, align, source, train_labels, post_s=2.0, zscore=True,
              basis_id=None, alpha=None, n_features=None):
    """The full identity of one frozen model, as a plain dict (JSON-serialisable, order-stable)."""
    labs = sorted(train_labels)
    # ONE ANIMAL PER MODEL, checked rather than assumed. Every caller pools per animal, but the animal
    # is the first field of the identity and a pooled-across-animals training set would produce a model
    # filed under one animal's name that had seen another's data -- the exact shape of label-asserts-a-
    # property-nothing-verifies that this module exists to end.
    others = {config.animal_of(x) for x in labs} - {animal}
    if others:
        raise ValueError(f"frozen model for {animal} was given training sessions from "
                         f"{sorted(others)}: {[x for x in labs if config.animal_of(x) != animal]}")
    return {
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
        return Path(config.resolver().root("labcams")) / "frozen_models"
    except Exception:                                    # noqa: BLE001
        return None


def _roots():
    return [(local_dir(), "local"), (server_dir(), "server")]


#: A store the PREVIOUS derivation of `local_dir` would have used. `joint_locanmf._basis_dir`
#: falls back to this literal when `figures_working` is not set for the machine, so any box that
#: fitted models while its profile was incomplete put them here.
_LEGACY_LOCAL = Path("C:/wf_local/frozen_models")

_ANNOUNCED: set[str] = set()


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
    if _LEGACY_LOCAL.exists() and _LEGACY_LOCAL != root:
        n = len(list(_LEGACY_LOCAL.rglob("manifest.json")))
        log(f"[frozen]    {n} model(s) are sitting in {_LEGACY_LOCAL}, the path this box used "
            f"before its machine profile set `figures_working`. Publish them "
            f"(`WIDEFIELD_FROZEN_MODEL_DIR={_LEGACY_LOCAL} python -m wfield_local.publish_basis "
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
    - **Across machines it is reproducible only to the optimiser's tolerance.** Preferring local
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


def siblings(spec):
    """Every stored model for the same (animal, kind, align, source), newest last.

    What makes a SPEC CHANGE visible. Without this, a changed pre-stroke set is indistinguishable
    from a first run -- both simply find nothing -- and the reference would move with no one
    noticing, which is the failure this module exists to prevent.
    """
    prefix = _slug(spec).rsplit("_", 1)[0] + "_"
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


def _write(spec, payload, meta, log=print):
    import joblib

    from wfield_local import writeguard

    d = local_dir() / spec["animal"] / _slug(spec)
    writeguard.assert_writable(d.parent)
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
                      "frozen_utc": dt.datetime.utcnow().isoformat(), "meta": meta},
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
