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
    projall     ``projall``             WIDEFIELD_JOINT_PROJECT_ALL forced on (the adopted-variant control of the
                                         projection method; review M1) -- output AND cache
    csig        ``csig-<sha>``          CACHE ONLY, under the camera gate: `trial_classes.class_source_signature` (the
                                         class tables + class code; review H4). Never in an output folder.

`lp_licks.output_tag` = its own LP / ENL / gate tag + these parts, so `lp_licks.output_dir` routes every renderer that
already uses it and `writeguard.assert_writable` REFUSES any share write of a tagged run that is not under the tag
folder. CACHE KEYS use only the parts that change per-session results (`cache_parts`: hemo, lnmf, selection,
projall, csig);
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


def drift_part() -> str:
    """The encoder / ITI slow-drift cutoff SENSITIVITY arm (Priya 11-03): ``drift<p>s`` when
    COMBINED_ENCODING_DRIFT_MIN_PERIOD_S sets a period other than the primary 600 s; "" otherwise. It enters both the
    output tag and the cache keys, so the 1500 s arm never meets the primary outputs."""
    import sys
    v = os.environ.get("COMBINED_ENCODING_DRIFT_MIN_PERIOD_S", "").strip()
    p = float(v) if v else 600.0
    ce = sys.modules.get("wfield_local.combined_encoding")
    if ce is not None and float(ce.DRIFT_MIN_PERIOD_S) != p:
        # the constant is read at IMPORT: an env set after the import would tag outputs with a cutoff never used
        raise RuntimeError(f"COMBINED_ENCODING_DRIFT_MIN_PERIOD_S = {p:g} s but combined_encoding was imported with "
                           f"{ce.DRIFT_MIN_PERIOD_S:g} s: set the env (render profile) before importing analysis code")
    return "" if p == 600.0 else f"drift{p:g}s"


def emcentre_part() -> str:
    """``emc-engaged`` when the camera trial gate runs with the bal_em centre on every DAQ-engaged row (the
    alternative to its default camera-working centre, `spout_scaling.centre_rows`); "" for each gate's default."""
    from wfield_local import spout_scaling, trial_classes
    if spout_scaling.spout_scaling() is None or not trial_classes.camera_gate_on():
        return ""
    return "emc-engaged" if spout_scaling.centre_rows() == "engaged" else ""


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


def projall_part() -> str:
    from wfield_local import joint_locanmf
    on = os.environ.get(joint_locanmf.PROJECT_ALL_ENV, "").strip().lower() in ("1", "true", "yes", "on")
    return "projall" if on else ""


def csig_part() -> str:
    from wfield_local import trial_classes
    return f"csig-{trial_classes.class_source_signature()}" if trial_classes.camera_gate_on() else ""


_ALL = (("hemo", hemo_part), ("lnmf", lnmf_part), ("drift", drift_part), ("pre", pre_part), ("selection", selection_part), ("emcentre", emcentre_part),
        ("refs", refs_part), ("inmem", inmem_part), ("lptrained", lptrained_part), ("projall", projall_part),
        ("csig", csig_part))
_CACHE = ("hemo", "lnmf", "drift", "selection", "emcentre", "projall", "csig")
#: parts that enter cache keys only, never an output path / tag
_CACHE_ONLY = ("csig",)


def parts(include=None, exclude=()) -> list[str]:
    """Non-empty mode parts in the fixed order above ([] for the default run). Cache-only parts (`_CACHE_ONLY`) are
    returned only when named in ``include``."""
    out = []
    for name, fn in _ALL:
        if (include is not None and name not in include) or name in exclude:
            continue
        if include is None and name in _CACHE_ONLY:
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


def _out_tag() -> str:
    """The FULL output tag: `lp_licks.output_tag()` (LP source / ENL window / gate-LP + every output part here), so a
    fixed-name file or a routed root of an LP / ENL / gate-LP run is keyed too (review M3, 11-03) and matches the
    segment `writeguard` requires. "" for the default run."""
    from wfield_local import lp_licks
    return lp_licks.output_tag() or ""


def routed_output(path, *, make: bool = True, guard: bool = True, lp_family: bool = False) -> Path:
    """THE output folder of a writer in this run (review H2 / N4 / N5, 11-03): ``path`` (an explicit ``--output`` or the
    writer's default) with the run's output tag appended unless the tag is already one of its segments -- so a nightly
    passing ``--output <figures_working>`` and the readers (`grant_kit._fig_root`) resolve the same tagged folder; the
    folder is created (``make``) and `writeguard.assert_writable` is called (``guard``). The default run: ``path``."""
    p = Path(path)
    t = lp_family_tag() if lp_family else _out_tag()
    if lp_family:
        _LP_FAMILY_ROOTS.add(_norm(p.parent if t and p.name == t else p))
    if t and t not in p.parts:
        p = p / t
    if make and t:
        p.mkdir(parents=True, exist_ok=True)
    if guard:
        from wfield_local.writeguard import assert_writable
        assert_writable(p / "_")
    return p


def keyed_name(name: str, *, cache: bool = False) -> str:
    """``name`` with the run's tag inserted before the suffix (``x.json`` -> ``x__<tag>.json``); ``name`` unchanged by
    default. For FIXED-NAME files read by another step (grant 2 / 2b / 4, 3a / 5b, 3b / 5 JSONs) and per-session
    skip-if-exists files. ``cache=True`` uses only the per-session cache parts."""
    t = cache_tag() if cache else _out_tag()
    if not t:
        return name
    p = Path(name)
    return f"{p.stem}__{t}{p.suffix}"


def keyed_path(path, *, cache: bool = False) -> Path:
    p = Path(path)
    return p.with_name(keyed_name(p.name, cache=cache))


#: roots routed with ``lp_family=True`` in this process (T1, third review): `writeguard` accepts a write under such a
#: root when it carries `lp_family_tag()` (or nothing, for the plain LP run), the same path `subdir` resolved
_LP_FAMILY_ROOTS: set = set()
#: content-keyed DATA folders registered in this process (T2): e.g. the ITI per-session extracts (`iti_licks.XOUT`),
#: keyed by the data parts of the mode only; `writeguard` treats them like the frozen store (not tag-checked)
_DATA_STORES: set = set()


def register_data_store(path) -> Path:
    p = Path(path)
    _DATA_STORES.add(_norm(p))
    return p


def _norm(p) -> str:
    import os
    return os.path.normpath(str(p)).replace("\\", "/").lower().rstrip("/") + "/"


def lp_family_root_ok(c: str) -> bool:
    """True if the normalised path ``c`` lies under a registered LP-family root, at that root's routed location."""
    t = lp_family_tag().lower()
    return any(c.startswith(r) and ((not t) or c[len(r):].startswith(t + "/")) for r in _LP_FAMILY_ROOTS)


def data_store_ok(c: str) -> bool:
    return any(c.startswith(r) for r in _DATA_STORES)


def lp_family_tag() -> str:
    """The output tag of an LP-ONLY family (attempt_coding, attempt_maps, ...; review N1, 11-03): `lp_licks.output_tag()`
    without its plain "LP_licks" part (the LP source is implicit there), so the default LP run keeps the pre-rebuild
    folder and every OTHER mode (ENL window, gate-LP, hemo variant, camera gate ...) still gets its own."""
    t = _out_tag()
    if t.startswith("LP_licks"):
        t = t[len("LP_licks"):].lstrip("_")
    return t


def subdir(root, *, cache: bool = False, lp_family: bool = False, make: bool = False,
           include=None) -> Path:
    """``root/<tag>`` for a tagged run, ``root`` by default -- for skip-if-exists per-session folders and output roots
    that do not go through `lp_licks.output_dir`."""
    if include is not None:
        t = "_".join(parts(include=include))
    elif cache:
        t = cache_tag()
    else:
        t = lp_family_tag() if lp_family else _out_tag()
    if lp_family:
        _LP_FAMILY_ROOTS.add(_norm(root))
    out = Path(root) / t if t else Path(root)
    if make and t:                       # review N5: a tagged writer's folder is created; default paths untouched
        out.mkdir(parents=True, exist_ok=True)
    return out


class with_env:
    """Context manager: set (value) / unset (None) environment variables for the block, restoring them after (review 7B:
    the path of another block's output, e.g. the june_august_drift block, which unsets WIDEFIELD_PRE_FROM)."""

    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


#: the rebuild's PRIMARY pre set (render profile cohort1_rebuild_20261103: WIDEFIELD_PRE_FROM=0801); a block that keeps
#: its own pre set (june_august_drift, 12.1b) reads the primary run's outputs under this pre tag (review 7B)
PRIMARY_PRE_FROM = "0801"


def primary_output_dir(default):
    """`lp_licks.output_dir(default)` of the PRIMARY run (WIDEFIELD_PRE_FROM = `PRIMARY_PRE_FROM`) when this run is
    tagged; ``default`` itself in the default run. For a pre-less block reading a primary output (review 7B B1 / N2)."""
    from wfield_local import lp_licks
    if not lp_licks.output_tag():
        return Path(default)
    with with_env(WIDEFIELD_PRE_FROM=PRIMARY_PRE_FROM):
        return Path(lp_licks.output_dir(Path(default)))


def june_august_drift_dir(labcams):
    """The june_august_drift output folder (Fig 4.3): ``<labcams>/chronic_stability/june_august_drift`` in the default
    run; in a tagged run ``<labcams>/chronic_stability/<the block's own tag: PRE_FROM unset>/june_august_drift`` -- the
    one definition the script writes and the deck reads (review 7B B1)."""
    from wfield_local import lp_licks
    d = Path(labcams) / "chronic_stability" / "june_august_drift"
    with with_env(WIDEFIELD_PRE_FROM=None):
        return Path(lp_licks.output_dir(d)) if lp_licks.output_tag() else d


def require_default(what: str) -> None:
    """Raise unless this is a default run: for writers whose output name cannot carry a tag (refuse, do not overwrite).
    R5 (fourth review): the FULL output tag (`lp_licks.output_tag`: LP source, gate-LP, ENL window AND the run-mode
    parts), so an LP / gate-LP / ENL run refuses too."""
    from wfield_local import lp_licks
    t = lp_licks.output_tag()
    if t:
        raise RuntimeError(f"{what}: refusing to run in mode {t!r} -- its outputs have fixed default names and would "
                           "overwrite the default ones; give it an explicit output root or unset the mode")


# ------------------------------------------------------------------------------------- inputs given as paths (N5)
#: path segments that are run tags (output-folder tags of a non-default run): the LP / ENL / gate-LP names and the
#: run-mode parts' prefixes
_TAG_PREFIXES = ("hv-", "lnmfproj", "sel-", "pre0", "LP_licks", "DAQ_ENL", "DAQ_gateLP", "drift", "emc-", "refs-",
                 "inmem-", "projall", "lpdaqtrained")


def input_run_tag(path) -> str:
    """The run tag a path was written under: its (last) path segment that is a run tag; "" = the default run."""
    segs = [s for s in Path(path).parts if s.startswith(_TAG_PREFIXES)]
    return segs[-1] if segs else ""


def require_input_tag(path, what: str) -> str:
    """For a script that READS an input given as a path (fifth review N5): refuse an input written under another run
    tag than this run's (`lp_licks.output_tag`), and say which tag it read. Returns the tag ("" = default)."""
    from wfield_local import lp_licks
    got, want = input_run_tag(path), (lp_licks.output_tag() or "")
    print(f"[{what}] input run tag: {got or 'default'} (this run: {want or 'default'})", flush=True)
    if got != want:
        raise SystemExit(f"{what}: input {path} was written under run tag {got or 'default'!r}, this run is "
                         f"{want or 'default'!r} -- refusing to mix runs (point it at this run's folder)")
    return got

