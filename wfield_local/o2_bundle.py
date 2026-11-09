"""What an O2 encoding bundle (`scripts/o2_encoding.py`) records and what its runner verifies (coordinator 11-04: the
bundle must reproduce THIS rebuild run exactly on O2).

  settings()     every setting `run_session` depends on, from this process's env: the full render-profile env
                 (WIDEFIELD_* / WFIELD_* / LOCANMF_*: hemo variant, LocaNMF projection, trial gate / window / end / stopped,
                 working trials, PRE_FROM, LP_FROZEN, ...) + COMBINED_ENCODING_* / OROFACIAL_*; minus this machine's
                 store / cache paths and the per-submit options. -> env.json (the runner applies it BEFORE importing
                 wfield_local: hemo_variants.ADOPTED etc. resolve at import).
  identity()     what the env computes to in THIS code: the run tag (`run_mode.tag()`), the selection
                 (`trial_classes.run_selection()`), the area basis (`area_basis.tag()`). -> bundle.json at bundle time;
                 the runner recomputes it from env.json with the bundled code and REFUSES on any difference (`verify`).
"""
from __future__ import annotations

import os

RECORD_PREFIXES = ("WIDEFIELD_", "WFIELD_", "LOCANMF_", "COMBINED_ENCODING_", "OROFACIAL_")
#: machine-local store / cache locations and render scopes: never part of a bundle's settings (O2 has its own paths;
#: the rebuild root's fence is a desktop guard)
LOCAL_ONLY = frozenset({"WIDEFIELD_REBUILD_ROOT", "WFIELD_INPUT_CACHE", "WIDEFIELD_SESSION_CACHE",
                        "WIDEFIELD_FROZEN_MODEL_DIR", "WIDEFIELD_MAP_JOBS", "WIDEFIELD_ONLY_ANIMALS"})
#: per-submit options (sbatch --export), never part of a bundle's recorded input settings
SUBMIT_ENVS = frozenset({"COMBINED_ENCODING_ONLY", "COMBINED_ENCODING_VIDEO", "COMBINED_ENCODING_FR",
                         "COMBINED_ENCODING_FR_POINTS", "COMBINED_ENCODING_FR_SHARE", "COMBINED_ENCODING_MATCH",
                         "COMBINED_ENCODING_BINS", "COMBINED_ENCODING_BINS_WIDTH", "COMBINED_ENCODING_FR_GROUPS",
                         "COMBINED_ENCODING_SUFFIX", "COMBINED_ENCODING_ATTEMPTS"})


def settings(env=None) -> dict:
    """{name: value} of every recorded setting set (non-empty) in ``env`` (default os.environ), sorted."""
    env = os.environ if env is None else env
    return {k: v for k, v in sorted(env.items())
            if k.startswith(RECORD_PREFIXES) and v and k not in LOCAL_ONLY and k not in SUBMIT_ENVS}


def identity() -> dict:
    """The run identity this process's env computes to (see module docstring)."""
    from wfield_local import area_basis, run_mode
    from wfield_local import trial_classes as tc
    return {"run_tag": run_mode.tag(), "selection": tc.run_selection(), "area_basis": area_basis.tag()}


class BundleMismatch(RuntimeError):
    pass


#: RUN-TAG PARTS THAT ARE NOT ENCODER INPUTS (REBUILD_CODE 8.54, coordinator 11-05): a `run_mode` part listed here names
#: a setting that no `combined_encoding.session_inputs` pickle depends on, so a bundle recorded WITHOUT it is still the
#: current run's bundle. Each entry needs a DEFINITIVE check that the inputs are byte-identical with and without it:
#:   daqlk  imaging DAQ lick detector (8.53). Checked 11-05: bundle enc_rebuild_20261104b's inputs (c149decf, old imaging
#:          detector) rebuilt under 90b7ddd5 for PS95_0917, PS94_0819, PS93_0814 are equal array by array (8.54: two files
#:          byte-identical, PS93_0814 differs only in the pickle memo of numpy dtype objects); the
#:          encoder's DAQ licks are `dlc_frames.lick_onsets` = `daq_trials.decode` with the yaml profile in both.
#: Exact part names only (no prefix / fuzzy match); every other part must match exactly, and so must the selection and
#: the area basis.
NON_INPUT_TAG_PARTS = ("daqlk",)


def tag_without_non_input_parts() -> str:
    """This process's run tag with the `NON_INPUT_TAG_PARTS` removed (the tag an input-identical older bundle recorded)."""
    from wfield_local import run_mode
    return "_".join(p for p in run_mode.parts() if p not in NON_INPUT_TAG_PARTS)


def verify(recorded: dict) -> dict:
    """Recompute `identity()` and REFUSE (BundleMismatch) unless it equals the recorded one key by key -- or differs ONLY
    in the run tag, by exactly the current run's `NON_INPUT_TAG_PARTS` (recorded tag == the current tag without them).
    The accepted case returns the CURRENT identity plus ``recorded_run_tag`` / ``accepted_parts`` (the outputs belong to
    the current run's folders)."""
    from wfield_local import run_mode
    now = identity()
    bad = {k: (recorded.get(k), now.get(k)) for k in now if recorded.get(k) != now.get(k)}
    if set(bad) == {"run_tag"} and recorded.get("run_tag") == tag_without_non_input_parts():
        extra = [p for p in run_mode.parts() if p in NON_INPUT_TAG_PARTS]
        print(f"[o2_bundle] accepted: the recorded run tag differs from this run's only by non-input part(s) {extra} "
              f"(o2_bundle.NON_INPUT_TAG_PARTS)", flush=True)
        return {**now, "recorded_run_tag": recorded["run_tag"], "accepted_parts": extra}
    if bad:
        raise BundleMismatch("bundle settings do not reproduce the recorded run identity: "
                             + "; ".join(f"{k}: recorded {a!r} vs computed {b!r}" for k, (a, b) in bad.items()))
    return now


# ------------------------------------------------------------------------------------------------ analysis masks (11-09)
#: THE BUNDLE'S MASKS. `inputs/<label>.pkl` carries the imaging tuple (flat pixel index, stat mask, area masks), so its
#: content depends on the brain / stat / painted masks, and `identity()` does not see them (the rebuild glue-mask bug,
#: GLUE_MASK_FIX_2026-11-09). O2 has no share and cannot recompute the masks, so the digest
#: (`mask_keys.signature()`) is computed at BUNDLE time and recorded twice: in bundle.json (``mask_sig``) and in every
#: inputs pickle (``mask_sig``). The desktop compares bundle.json with its own current masks (`verify_masks`); the
#: runner compares each pickle with bundle.json (`check_inputs`), i.e. against the bundle's own record.
MASK_FIELD = "mask_sig"


def verify_masks(meta: dict, now: str | None = None) -> None:
    """DESKTOP side: REFUSE (BundleMismatch) a bundle whose recorded masks differ from this process's current masks, or
    a bundle with no ``mask_sig`` recorded inside a rebuild root (every rebuild bundle before 11-09 was built with the
    glue in). A legacy bundle at the default root (no rebuild_root recorded, none active) is accepted."""
    from wfield_local import mask_keys, rebuild_root
    have = meta.get(MASK_FIELD)
    if not have:
        if meta.get("rebuild_root") or rebuild_root.active():
            raise BundleMismatch(f"bundle {meta.get('name', '')!r} records no {MASK_FIELD}: built before the 11-09 "
                                 f"glue-mask fix inside a rebuild root (glue IN). Rebuild its inputs under a new name.")
        return
    now = mask_keys.signature() if now is None else now
    if have != now:
        raise BundleMismatch(f"bundle {meta.get('name', '')!r} was built on analysis masks {have}, current masks are "
                             f"{now}. Rebuild its inputs under a new name.")


def check_inputs(meta: dict, inputs: dict, label: str = "") -> None:
    """RUNNER side (no share): REFUSE an inputs pickle whose ``mask_sig`` is not the bundle's recorded one. A bundle
    with no record is refused when it was built inside a rebuild root, accepted otherwise (legacy default-root)."""
    rec = meta.get(MASK_FIELD)
    if not rec:
        if meta.get("rebuild_root"):
            raise BundleMismatch(f"{label}: bundle built in a rebuild root with no {MASK_FIELD} (before the 11-09 "
                                 f"glue-mask fix: glue IN). Rebuild its inputs under a new name.")
        return
    have = inputs.get(MASK_FIELD) if isinstance(inputs, dict) else None
    if have != rec:
        raise BundleMismatch(f"{label}: inputs pickle built on analysis masks {have!r}, the bundle records {rec!r}.")
