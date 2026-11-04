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


def verify(recorded: dict) -> dict:
    """Recompute `identity()` and REFUSE (BundleMismatch) unless it equals the recorded one key by key."""
    now = identity()
    bad = {k: (recorded.get(k), now.get(k)) for k in now if recorded.get(k) != now.get(k)}
    if bad:
        raise BundleMismatch("bundle settings do not reproduce the recorded run identity: "
                             + "; ".join(f"{k}: recorded {a!r} vs computed {b!r}" for k, (a, b) in bad.items()))
    return now
