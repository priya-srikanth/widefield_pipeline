"""Config loader — single source of truth for animals, sessions, paths, and analysis defaults.

Reads `configs/*.yaml` (mirrors stroke_orofacial_pipeline). `load_sessions()` produces the list-of-dicts
`SESSIONS` structure the pipeline consumes, replacing the previously-hardcoded list in
`locanmf_cue_lick_analysis.py`. All MMDD dates are strings (YAML would parse leading-zero `0606` as octal).
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

from wfield_local.paths import PathResolver

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

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


def animal_color() -> dict:
    """Per-animal matplotlib color (single source of truth for figure coloring)."""
    return {a: v.get("color", "k") for a, v in animals().items()}


def defaults() -> dict:
    return _load("defaults.yaml")


def paths() -> dict:
    return _load("paths.yaml")


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


def cross_session_dates() -> list[str]:
    """Curated cross-session MMDD set from animals.yaml date_policy (6/6-6/8 + 8/6 onward)."""
    return [str(d) for d in date_policy().get("cross_session", [])]
