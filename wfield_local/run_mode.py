"""ONE definition of "which non-default mode is this run in", for output folders, fixed-name files and cache keys.

WHY (cohort-1 rebuild, 2026-11-03; traps T12 / T18 / T19 / T21 / T24 of docs/status/rebuild_20261103). Before this,
only the lick source / ENL window / LP gate reached an output path (`lp_licks.output_tag`). A run under another hemo
variant, an August-only pre (WIDEFIELD_PRE_FROM), a camera trial selection, a non-default map reference set or an
in-memory (``WIDEFIELD_<FAMILY>_FROZEN=0``) refit wrote to the DEFAULT folders and file names, so it overwrote the
adopted figures, its fixed-name JSONs were read by the next default render, and its per-session skip-if-exists files
were served to the other mode.

THE RULE. Every non-default mode contributes one PART; the default run has no parts, so every default path, file name
and cache key is byte-identical to before (pinned by tests/test_run_mode.py). Parts, in this order:

    hemo        ``hv-<variant>``        `config.hemo_tag()`: any hemo variant other than `config.UNTAGGED_HEMO_VARIANT`
                                         (the variant every untagged output holds), so flipping ``defaults.yaml
                                         hemo.variant`` STILL tags the new outputs
    lnmf        ``lnmfproj``            per-session LocaNMF served by the fixed-footprint projection (`locanmf_session`)
    pre         ``pre<MMDD>``           WIDEFIELD_PRE_FROM (`config.pre_from`)
    selection   ``sel-<source>-<win>``  a trial selection other than the untagged DAQ quit-tail rule
                                         (`trial_classes.run_selection_tag`)
    refs        ``refs-<names>``        WIDEFIELD_MAP_REFS (`position_reference_maps.REFS_ENV`)
    inmem       ``inmem-<families>``    any ``WIDEFIELD_<FAMILY>_FROZEN`` opt-out (in-memory pre-stroke refit)
    lptrained   ``lpdaqtrained``        WIDEFIELD_LP_FROZEN=daq (the LP arm's DAQ-trained cross-check)

`lp_licks.output_tag` = its own LP / ENL / gate tag + these parts, so `lp_licks.output_dir` routes every renderer that
already uses it and `writeguard.assert_writable` REFUSES any share write of a tagged run that is not under the tag
folder. CACHE KEYS use only the parts that change per-session results (`cache_parts`: hemo, lnmf, selection);
pre / refs / inmem / lptrained are pooling- or reference-level and never enter a per-session cache.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

#: Frozen-family opt-out env vars are WIDEFIELD_<FAMILY>_FROZEN; LP_FROZEN is a lick-source choice, not an opt-out.
_FROZEN_RE = re.compile(r"^WIDEFIELD_([A-Z0-9_]+)_FROZEN$")
_NOT_OPT_OUT = {"LP"}


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", str(s)).strip("-")


def hemo_part() -> str:
    from wfield_local import config
    return config.hemo_tag()


def lnmf_part() -> str:
    from wfield_local import locanmf_session
    return "lnmfproj" if locanmf_session.projection_on() else ""


def pre_part() -> str:
    from wfield_local import config
    pf = config.pre_from()
    return f"pre{_safe(pf)}" if pf else ""


def selection_part() -> str:
    from wfield_local import trial_classes
    return trial_classes.run_selection_tag() or ""


def refs_part() -> str:
    v = os.environ.get("WIDEFIELD_MAP_REFS", "").strip()
    if not v:
        return ""
    names = [x for x in re.split(r"[,\s]+", v) if x]
    return "refs-" + "+".join(_safe(x) for x in names)


def inmem_part() -> str:
    from wfield_local.frozen_models import OPT_OUT_VALUES
    fams = sorted(m.group(1).lower() for k, val in os.environ.items()
                  if (m := _FROZEN_RE.match(k)) and m.group(1) not in _NOT_OPT_OUT
                  and val.strip().lower() in OPT_OUT_VALUES)
    return ("inmem-" + "+".join(fams)) if fams else ""


def lptrained_part() -> str:
    from wfield_local import frozen_models
    return "lpdaqtrained" if frozen_models.daq_trained_in_lp() else ""


_ALL = (("hemo", hemo_part), ("lnmf", lnmf_part), ("pre", pre_part), ("selection", selection_part),
        ("refs", refs_part), ("inmem", inmem_part), ("lptrained", lptrained_part))
_CACHE = ("hemo", "lnmf", "selection")


def parts(include=None, exclude=()) -> list[str]:
    """Non-empty mode parts in the fixed order above ([] for the default run)."""
    out = []
    for name, fn in _ALL:
        if (include is not None and name not in include) or name in exclude:
            continue
        p = fn()
        if p:
            out.append(p)
    return out


def tag(include=None, exclude=()) -> str:
    """"" for the default run, else the parts joined by "_" (a single path segment)."""
    return "_".join(parts(include, exclude))


def cache_parts() -> list[str]:
    return parts(include=_CACHE)


def cache_tag() -> str:
    """Suffix for per-session cache KINDS / file keys ("" by default): hemo variant, LocaNMF projection, selection."""
    return "_".join(cache_parts())


def describe() -> str:
    return tag() or "default"


def keyed_name(name: str, *, cache: bool = False) -> str:
    """``name`` with the run's tag inserted before the suffix (``x.json`` -> ``x__<tag>.json``); ``name`` unchanged by
    default. For FIXED-NAME files read by another step (grant 2 / 2b / 4, 3a / 5b, 3b / 5 JSONs) and per-session
    skip-if-exists files. ``cache=True`` uses only the per-session cache parts."""
    t = cache_tag() if cache else tag()
    if not t:
        return name
    p = Path(name)
    return f"{p.stem}__{t}{p.suffix}"


def keyed_path(path, *, cache: bool = False) -> Path:
    p = Path(path)
    return p.with_name(keyed_name(p.name, cache=cache))


def subdir(root, *, cache: bool = False) -> Path:
    """``root/<tag>`` for a tagged run, ``root`` by default -- for skip-if-exists per-session folders and output roots
    that do not go through `lp_licks.output_dir`."""
    t = cache_tag() if cache else tag()
    return Path(root) / t if t else Path(root)


def require_default(what: str) -> None:
    """Raise unless this is a default run: for writers whose output name cannot carry a tag (refuse, do not overwrite)."""
    t = tag()
    if t:
        raise RuntimeError(f"{what}: refusing to run in mode {t!r} -- its outputs have fixed default names and would "
                           "overwrite the default ones; give it an explicit output root or unset the mode")
