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


def curated_dates(machine: str | None = None) -> list[str]:
    """The LIVE curated cross-session date set: every REGISTERED date minus ``cross_session_exclude``.

    "6/6-6/8 + 8/6 onward" is an emergent property, not a list: this is really "everything registered
    in sessions.yaml that animals.yaml does not exclude", so a newly registered night joins
    automatically. Sorted MMDD strings.

    This is the ONLY curated-set accessor. A static ``date_policy.cross_session`` list used to sit
    beside it in animals.yaml; it lagged five nights behind the registered sessions and the deck
    builder read it, so a hand-run deck quietly covered 5 dates where the nightly covered 9. Both the
    list and its accessor were deleted on 2026-08-13.
    """
    exclude = set(date_policy().get("cross_session_exclude", []))
    registered = sorted({s["label"].split("_")[1] for s in load_sessions(machine)})
    return [d for d in registered if d not in exclude]


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
        return None if val is None else rv.resolve(_SESSION_FIELD_ROOT[field], val)

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
