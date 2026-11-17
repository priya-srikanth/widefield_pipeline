"""Per-session memo for loaders of ORIGINAL data (raw DAQ h5): one read per (label, input environment), a copy per call.

Profile 11-08: one section G session unit called `trial_classes.daq_cue_seconds` 297 times, 1778 of its 2097 s
re-reading and re-decoding the same raw DAQ h5 files. A raw file never changes, so the result can only change with the
environment that picks the session list; the key holds every WIDEFIELD_* / WFIELD_* variable. Each call returns its own
copy, so a caller that edits the array cannot change what the next caller gets.

Kept OUT of `trial_classes` on purpose: the camera-class digest (`trial_classes.class_source_signature`, via
`map_collect.closure`) hashes the class module's own top-level definitions, so a speed-only memo written there moved
csig (3d7bbe921e -> edce41f2a5, 11-08) and invalidated every camera-gated cache entry. Applied there as a hook-marked
decorator that names nothing in that module, the digest is the pre-memo one (tests/test_daq_cue_seconds_memo.py).

`per_path` (11-09) is the path-keyed sibling, for a loader that takes a raw file path instead of a session label.
"""
from __future__ import annotations

import functools
import os


def _env_key() -> tuple:
    return tuple(sorted((k, v) for k, v in os.environ.items() if k.startswith(("WIDEFIELD_", "WFIELD_"))))


def per_env(fn):
    """Memoize ``fn(label)`` per (label, input environment); every call gets ``.copy()`` of the stored result."""
    cache: dict = {}

    @functools.wraps(fn)
    def wrapper(label):
        key = (label, _env_key())
        if key not in cache:
            cache[key] = fn(label)
        return cache[key].copy()
    wrapper.cache_clear = cache.clear
    return wrapper


#: `per_path` clears its store past this many entries (a whole cohort is ~110 sessions; one DAQ event dict ~1 MB).
PER_PATH_MAX = 512


def _file_key(path) -> tuple | None:
    """(normalized absolute path, size, mtime_ns) of an input file, or None when it cannot be stat-ed (no memo)."""
    p = os.path.normcase(os.path.abspath(os.fspath(path)))
    try:
        st = os.stat(p)
    except (OSError, TypeError, ValueError):
        return None
    return (p, st.st_size, st.st_mtime_ns)


def per_path(fn):
    """Memoize ``fn(path, *args, **kwargs)`` per (file, args): one read per raw file, a deep copy per call.

    Profile 11-09 (grant unit 10[cue/working], 2447 s): `plot_spout_trial_averages._load_daq_events` ran 322 times on
    the same 108 DAQ h5 files (1264 s). The key is the normalized absolute path PLUS the file's size and mtime (a
    rewritten file reloads) and every other argument; the loader reads no environment variable, so none enters the key.
    Each call returns ``copy.deepcopy`` of the stored result, so a caller that edits an array cannot change what the
    next caller gets. A path that cannot be stat-ed is passed straight through (the loader raises its own error).
    """
    import copy
    cache: dict = {}

    @functools.wraps(fn)
    def wrapper(path, *args, **kwargs):
        fk = _file_key(path)
        if fk is None:
            return fn(path, *args, **kwargs)
        key = (fk, args, tuple(sorted(kwargs.items())))
        try:
            hit = key in cache
        except TypeError:                                  # unhashable extra argument: no memo
            return fn(path, *args, **kwargs)
        if not hit:
            if len(cache) >= PER_PATH_MAX:
                cache.clear()
            cache[key] = fn(path, *args, **kwargs)
        return copy.deepcopy(cache[key])
    wrapper.cache_clear = cache.clear
    wrapper.cache_info = lambda: len(cache)
    return wrapper
