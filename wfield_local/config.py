"""Config loader — single source of truth for animals, sessions, paths, and analysis defaults.

Reads `configs/*.yaml` (mirrors stroke_orofacial_pipeline). `load_sessions()` produces the list-of-dicts
`SESSIONS` structure the pipeline consumes, replacing the previously-hardcoded list in
`locanmf_cue_lick_analysis.py`. All MMDD dates are strings (YAML would parse leading-zero `0606` as octal).
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import yaml

from wfield_local.paths import PathResolver

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# The cohort is a single 2026 season, so MMDD <-> YYYYMMDD conversion just (un)prepends the year.
COHORT_YEAR = "2026"

# Per-field implicit root for a session entry's relative paths (see configs/sessions.yaml).
_SESSION_FIELD_ROOT = {"mc": "labcams", "fmdir": "labcams",
                       "h5": "daq_recorder_output", "behavior_trials": "behavior_logs"}

#: A session path may override its implicit root with an explicit ``<root>:<relpath>`` prefix, e.g.
#: ``labcams:20260805/PS93_.../ps93_reviewed_trials.csv``. Needed because a field's natural root is not
#: always its actual location: PS93 8/5's recovered-position CSV sits under LABCAMS, not behavior_logs,
#: and was stored as a machine-pinned absolute ``M:/MICROSCOPE/...`` path. That resolved only on the
#: analysis box (where M: IS MICROSCOPE); on the imaging box M: is standby, so the file did not exist
#: and the maps step failed for that session -- CLAUDE.md rule 3 exists for exactly this.
#: ``{3,}`` lowercase deliberately: a Windows drive letter (``M:/…``, ``C:/…``) must NOT match.
_ROOT_PREFIX = re.compile(r"^(?P<root>[a-z_]{3,}):(?P<rel>.+)$")


@functools.lru_cache(maxsize=None)
def _load(name):
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=None)
def resolver(machine: str | None = None) -> PathResolver:
    """Cached PathResolver for the current (or given) machine."""
    return PathResolver(machine=machine)


def animals() -> dict:
    return _load("animals.yaml")["animals"]


def date_policy() -> dict:
    return _load("animals.yaml")["date_policy"]


def locanmf_dir_name(variant: str | None = None) -> str:
    """Directory name for a session's LocaNMF outputs, from ``defaults.yaml locanmf.output_dir_name``.

    LocaNMF is fitted to ``SVTcorr``, so a DIFFERENT hemodynamic/drift-removal variant gives a
    DIFFERENT decomposition -- different component count, different footprints. The directory
    therefore has to name the variant it came from, or two incompatible decompositions end up
    indistinguishable on disk. See ``docs/PREPROCESSING_DECISION.md``.

    Override per call with ``variant``, or globally with ``WIDEFIELD_LOCANMF_VARIANT``, which is how
    an analysis can be pointed at a non-default decomposition without editing config.
    """
    v = variant or os.environ.get("WIDEFIELD_LOCANMF_VARIANT")
    if v:
        return f"locanmf_affine8v1_hemo_{v}"
    return str(defaults()["locanmf"]["output_dir_name"])


def locanmf_dir(mc, variant: str | None = None) -> str:
    """Absolute path to a session's LocaNMF outputs (``<mc>/<locanmf_dir_name()>``).

    Used instead of a hardcoded literal in every consumer: the name was written out 39 times across 22
    modules, so renaming it -- which adopting a new drift-removal variant REQUIRES -- meant editing all
    of them. Now it is one config key.
    """
    return f"{str(mc).rstrip('/')}/{locanmf_dir_name(variant)}"


def hemo_variant(variant: str | None = None) -> str | None:
    """The drift-removal variant whose ``SVTcorr`` analyses read, or None for the original.

    ``zerophase``/``none``/null all mean "the bare ``SVTcorr.npy``", because that file IS the
    zerophase product -- it is what the pipeline wrote. Override per call, or globally with
    ``WIDEFIELD_HEMO_VARIANT``.
    """
    v = variant if variant is not None else os.environ.get("WIDEFIELD_HEMO_VARIANT")
    if v is None:
        v = (defaults().get("hemo") or {}).get("variant")
    return None if v in (None, "", "none", "zerophase") else str(v)


def svtcorr_path(mc, variant: str | None = None) -> str:
    """Absolute path to the ``SVTcorr`` an analysis should read, for one session's ``mc`` dir.

    WHY THIS EXISTS. ``wfield_local_results/SVTcorr.npy`` was hardcoded in at least five analysis
    modules (joint_basis, joint_locanmf, locanmf_position_decoder, locanmf_decoder_weights,
    locanmf_frozen_decoder). That file is the ZEROPHASE product, so after adopting ``meegkit_hpfit``
    every one of them would have gone on silently reading the superseded data -- no error, no missing
    file, just the old result. The variant now comes from one config key.

    NOTE ON ``_refitT``. For the HYBRID variants (``fit_drift: filtfilt``) the saved ``T`` already IS
    the high-pass-fitted ``T``, so reusing it and refitting are equivalent -- measured on PS94_0812 at
    3.5e-5 max relative difference. The canonical directory is therefore the unsuffixed
    ``hemo_<variant>``; ``_refitT`` remains meaningful only for non-hybrid variants.
    """
    base = f"{str(mc).rstrip('/')}/wfield_local_results"
    v = hemo_variant(variant)
    return f"{base}/SVTcorr.npy" if v is None else f"{base}/hemo_{v}/SVTcorr.npy"


def svtcorr_in(results_dir, variant: str | None = None) -> str:
    """Like :func:`svtcorr_path` but from a ``wfield_local_results`` dir rather than an ``mc`` dir.

    The activity-map steps are handed the results dir directly, so they cannot use ``svtcorr_path``.
    They read the SAME variant as the decoders -- maps and decoders describing different data would be
    a quiet inconsistency in every deck that shows them side by side.
    """
    v = hemo_variant(variant)
    base = str(results_dir).rstrip("/\\")
    return f"{base}/SVTcorr.npy" if v is None else f"{base}/hemo_{v}/SVTcorr.npy"


def analysis_sessions(animals=None, curated_only=True, sessions=None):
    """Registered sessions for a cross-session analysis, with curation applied to the PRE side ONLY.

    ONE PLACE, because this selection has been written wrong three times in three modules and each
    time it deleted data silently rather than failing:

      * `evoked_amplitude`      `set(curated_dates()) | {"0817"}`
      * `spatial_reorganisation` `set(curated_dates()) | {"0817", "0818"}`
      * the Section G runner     `for an in ("PS94", "PS95")`

    Every one was correct on the day it was written and wrong at the next acquisition. A date literal
    is a snapshot of the study; the study keeps moving.

    THE INVARIANT: curation exists to keep noisy early sessions out of the pre-stroke REFERENCE BAND
    (PS95_0605 has a mean |amplitude| of 16.3 against ~0.53 elsewhere, which put PS95's band at
    [0.15, 18.09] -- a band no post-stroke value could fall outside). Applied to a session that is
    not `pre`, it deletes the measurement instead. So: a session whose phase is not "pre" is NEVER
    removed by curation; a non-curated PRE date still is.

    Pinned in both directions in tests/test_stroke_phase.py.
    """
    rows = list(sessions if sessions is not None else load_sessions())
    if curated_only:
        keep = set(curated_dates())
        rows = [x for x in rows
                if x["label"].split("_")[-1] in keep
                or session_phase(x["label"][:4], x["label"].split("_")[-1]) != "pre"]
    if animals:
        rows = [x for x in rows if x["label"][:4] in set(animals)]
    return rows


def curated_dates(machine: str | None = None, phase: str = "pre") -> list[str]:
    """The LIVE curated cross-session date set: every REGISTERED date minus ``cross_session_exclude``.

    "6/6-6/8 + 8/6 onward" is an emergent property, not a list: this is really "everything registered
    in sessions.yaml that animals.yaml does not exclude", so a newly registered night joins
    automatically. Sorted MMDD strings.

    STROKE-AWARE since 2026-08-17. "A newly registered night joins automatically" was right while
    every night was pre-stroke; the moment a lesion exists it becomes a live hazard, because this
    list feeds the joint bases, the frozen decoder's training pool and the no-lick reference. With
    any `stroke_date` set, ``phase="pre"`` (the DEFAULT) returns only dates up to and including the
    cutoff, ``phase="post"`` only dates after it, and ``phase="all"`` the old behaviour. While the
    whole cohort is pre-stroke nothing changes.

    This is the ONLY curated-set accessor. A static ``date_policy.cross_session`` list used to sit
    beside it in animals.yaml; it lagged five nights behind the registered sessions and the deck
    builder read it, so a hand-run deck quietly covered 5 dates where the nightly covered 9. Both the
    list and its accessor were deleted on 2026-08-13.
    """
    exclude = set(date_policy().get("cross_session_exclude", []))
    registered = sorted({s["label"].split("_")[1] for s in load_sessions(machine)})
    dates = [d for d in registered if d not in exclude]
    cut = stroke_cutoff()
    if cut is None:                       # whole cohort still pre-stroke: unchanged behaviour
        return dates
    if phase == "all":
        return dates
    if phase == "post":
        return [d for d in dates if d > cut]
    return [d for d in dates if d <= cut]


def stroke_date(animal: str) -> str | None:
    """MMDD of this animal's LAST PRE-STROKE session, or None if it has no lesion yet.

    Convention is animals.yaml's: the lesion is induced AFTER that day's session, so sessions ON
    stroke_date are baseline and only sessions AFTER are post-stroke.
    """
    v = (animals().get(animal) or {}).get("stroke_date")
    if v in (None, "", "null"):
        return None
    v = str(v)
    return v[4:] if len(v) == 8 else v          # accept YYYYMMDD or MMDD


def stroke_cutoff() -> str | None:
    """The EARLIEST stroke date across the cohort, or None if nobody is lesioned yet.

    Deliberately the earliest, not per-animal: `curated_dates()` feeds POOLED pre-stroke references
    (joint bases, the frozen decoder's training set, the no-lick reference), and a single post-stroke
    session leaking into any of them silently destroys the comparison those references exist to
    support. The conservative cutoff means a staggered cohort loses a little late pre-stroke data
    from the pooled set rather than risking contamination; per-animal work should call
    `stroke_date(animal)` directly.
    """
    ds = [d for d in (stroke_date(a) for a in animals()) if d]
    return min(ds) if ds else None


def animal_excluded_dates(animal: str) -> set[str]:
    """Dates dropped for this animal entirely — neither pre-stroke nor post-stroke.

    animals.yaml has carried an `exclude` list since the config was written and nothing read it.
    It earns its keep on 2026-08-17: PS92 and PS93 were lesioned on 8/16 PM but showed no overt
    deficit, so their stroke is being REDONE after the 8/17 session. That session is therefore
    neither baseline (a lesion was attempted) nor post-stroke (it did not take), and Priya's
    instruction is to exclude it from both while still showing the individual session data.

    A date here is invisible to every POOLED phase set. Per-session figures are built from the
    session registry, not from these lists, so they are unaffected — which is the intent.
    """
    v = (animals().get(animal) or {}).get("exclude") or []
    return {str(d)[4:] if len(str(d)) == 8 else str(d) for d in v}


def session_phase(animal: str, date: str) -> str:
    """'pre' | 'post' | 'excluded' for one animal on one date (MMDD).

    Three states, not two, because a lesion that did not take leaves sessions that belong to
    neither. Collapsing that into a binary would force such a session into a pool where it is
    misleading in one direction or the other.
    """
    date = str(date)[4:] if len(str(date)) == 8 else str(date)
    if date in animal_excluded_dates(animal):
        return "excluded"
    sd = stroke_date(animal)
    if sd is None:
        return "pre"
    return "pre" if date <= sd else "post"


def phase_labels(phase: str = "pre", machine: str | None = None) -> list[str]:
    """``animal_MMDD`` labels for one phase, resolved PER ANIMAL.

    Use this, not ``curated_dates()``, to build a pooled label list once any animal is lesioned.
    `curated_dates()` is a cohort-wide date list and cannot express "8/17 is post-stroke for PS94
    and PS95 but excluded for PS92 and PS93" — which is the actual state of this cohort.
    """
    exclude = set(date_policy().get("cross_session_exclude", []))
    out = []
    for s in load_sessions(machine):
        an, d = s["label"].split("_")[0], s["label"].split("_")[1]
        if d in exclude:
            continue
        if session_phase(an, d) == phase:
            out.append(s["label"])
    return sorted(out)


def animal_color() -> dict:
    """Per-animal matplotlib color (single source of truth for figure coloring)."""
    return {a: v.get("color", "k") for a, v in animals().items()}


def _deep_merge(base: dict, over: dict) -> dict:
    """Return a new dict = base with `over` merged in recursively (over wins; base untouched)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def defaults(session: str | None = None) -> dict:
    """Cohort-wide analysis parameters from `configs/defaults.yaml` (single source of truth).

    When `session` (a label like ``PS93_0807``) is given, that session's block from
    `configs/session_overrides.yaml` (under top-level ``overrides:``) is deep-merged on top —
    the defaults+per-session-overrides pattern from stroke_orofacial_pipeline. Returns the raw
    cohort defaults (the cached dict) when no session or no matching override exists.
    """
    base = _load("defaults.yaml")
    if not session:
        return base
    ov = (_load("session_overrides.yaml") or {}).get("overrides") or {}
    return _deep_merge(base, ov[session]) if ov.get(session) else base


def paths() -> dict:
    return _load("paths.yaml")


def _split_tokens(spec) -> list[str]:
    """Flatten a date/animal spec (str or list, comma- or space-separated) to a token list."""
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = [spec]
    out = []
    for s in spec:
        out.extend(str(s).replace(",", " ").split())
    return out


def _to8(d, year: str = COHORT_YEAR) -> str:
    """Normalize a date token to YYYYMMDD (accepts MMDD by prepending the cohort year).

    Rejects tokens of the wrong length or with an out-of-range month/day, so typos like `2026`
    (meant as a year) fail loudly instead of silently becoming `20262026`."""
    s = str(d)
    if s.isdigit() and len(s) in (4, 8):
        eight = s if len(s) == 8 else f"{year}{s}"
        mm, dd = int(eight[4:6]), int(eight[6:8])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return eight
    raise ValueError(f"bad date token {d!r} (want MMDD or YYYYMMDD)")


_RANGE_RE = re.compile(r"^(\d{4}|\d{8})(?:\.\.|[-:])(\d{4}|\d{8})$")


def expand_dates(spec, *, width: int = 4, available=None) -> list[str]:
    """Expand a date spec into a sorted, de-duplicated list of dates (the shared knob grammar).

    One knob used by BOTH the preprocessing and analysis CLIs. `spec` is a str or list; tokens are
    comma- or space-separated and each is one of:
      - a single date: `MMDD` (`0806`) or `YYYYMMDD` (`20260806`),
      - an inclusive range: `START-END` (also `START..END` / `START:END`), e.g. `0806-0808`,
      - `all`: every date in `available`.
    `width` selects the output form (4 = MMDD, 8 = YYYYMMDD). `available` (any width) is the set a
    range or `all` is resolved against — ranges are intersected with it (so they respect month
    boundaries and never invent nonexistent dates), and `all` requires it. Explicit single tokens
    pass through verbatim (so a freshly-acquired date not yet in `available` still processes)."""
    avail8 = sorted({_to8(d) for d in available}) if available is not None else None
    out8 = []
    for tok in _split_tokens(spec):
        if tok.lower() == "all":
            if avail8 is None:
                raise ValueError("date 'all' requires a set of available dates")
            out8.extend(avail8)
            continue
        m = _RANGE_RE.match(tok)
        if m:
            lo, hi = _to8(m.group(1)), _to8(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            if avail8 is not None:
                out8.extend(d for d in avail8 if lo <= d <= hi)
            else:   # no available set: contiguous integer expand (single-month ranges only)
                out8.extend(f"{lo[:4]}{n:04d}" for n in range(int(lo[4:]), int(hi[4:]) + 1))
            continue
        out8.append(_to8(tok))
    uniq = sorted(set(out8))
    return uniq if width == 8 else [d[4:] for d in uniq]


def normalize_animals(spec) -> list[str] | None:
    """Animal subset from a `--only` spec: a list, or None for 'all'/empty (i.e. no filter)."""
    toks = _split_tokens(spec)
    if not toks or any(t.lower() == "all" for t in toks):
        return None
    return toks


def _filter_set(explicit, env_name):
    """Resolve a subset filter: explicit arg (list/str) wins, else the env var, else None.

    Accepts a comma/space-separated string or a list; returns a set or None (no filter).
    """
    val = explicit if explicit is not None else os.environ.get(env_name)
    if not val:
        return None
    if isinstance(val, str):
        val = val.replace(",", " ").split()
    return {str(v) for v in val}


def load_sessions(machine: str | None = None, animals=None, dates=None) -> list[dict]:
    """Flatten `configs/sessions.yaml` into the pipeline's SESSIONS list:
    `dict(label, mc, h5, regime, fmdir, [behavior_trials])`, ordered by (date, animal).

    Root-relative path fields (mc/h5/fmdir/behavior_trials) are resolved to absolute
    paths for the current machine via the PathResolver (M: on the analysis box, N: on
    the imaging box). `fmdir` defaults to None; `behavior_trials` is included only when set.

    Optional subset filters (for running analysis on part of the cohort): `animals` (e.g.
    ["PS93"]) and `dates` (MMDD, e.g. ["0807"]). Each falls back to the `WIDEFIELD_ONLY_ANIMALS`
    / `WIDEFIELD_ONLY_DATES` env var (comma/space list) when not passed — so `nightly_figs --only`
    can scope the whole analysis via the environment its subprocesses inherit.
    """
    raw = _load("sessions.yaml")["sessions"]
    rv = resolver(machine)
    aset = _filter_set(animals, "WIDEFIELD_ONLY_ANIMALS")
    dset = _filter_set(dates, "WIDEFIELD_ONLY_DATES")

    def rp(field, val):
        """Resolve a session path against its implicit root, or an explicit ``<root>:<rel>`` one."""
        if val is None:
            return None
        m = _ROOT_PREFIX.match(str(val))
        if m:
            return rv.resolve(m.group("root"), m.group("rel"))
        return rv.resolve(_SESSION_FIELD_ROOT[field], val)

    rows = []
    for animal in raw:
        if aset and animal not in aset:
            continue
        for date, e in raw[animal].items():
            date = str(date)
            if dset and date not in dset:
                continue
            entry = dict(label=f"{animal}_{date}",
                         mc=rp("mc", e["mc"]), h5=rp("h5", e["h5"]),
                         regime=e.get("regime"), fmdir=rp("fmdir", e.get("fmdir")))
            if e.get("behavior_trials"):
                entry["behavior_trials"] = rp("behavior_trials", e["behavior_trials"])
            rows.append((date, animal, entry))
    rows.sort(key=lambda r: (r[0], r[1]))   # (date, animal)
    return [entry for _, _, entry in rows]


# NB there is deliberately NO `cross_session_dates()` any more. It returned the hand-maintained
# `date_policy.cross_session` list, which lagged the registered sessions by five nights and silently
# shrank any deck built outside the nightly. Use :func:`curated_dates`, which derives the set.
