"""Per-session memo for loaders of ORIGINAL data (raw DAQ h5): one read per (label, input environment), a copy per call.

Profile 11-08: one section G session unit called `trial_classes.daq_cue_seconds` 297 times, 1778 of its 2097 s
re-reading and re-decoding the same raw DAQ h5 files. A raw file never changes, so the result can only change with the
environment that picks the session list; the key holds every WIDEFIELD_* / WFIELD_* variable. Each call returns its own
copy, so a caller that edits the array cannot change what the next caller gets.

Kept OUT of `trial_classes` on purpose: the camera-class digest (`trial_classes.class_source_signature`, via
`map_collect.closure`) hashes the class module's own top-level definitions, so a speed-only memo written there moved
csig (3d7bbe921e -> edce41f2a5, 11-08) and invalidated every camera-gated cache entry. Applied there as a hook-marked
decorator that names nothing in that module, the digest is the pre-memo one (tests/test_daq_cue_seconds_memo.py).
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
