"""Per-session RESULT cache keyed on the input FILES (Priya 10-21: "use saved arrays, caches, parallelization ... the
things that make this most efficient and less redundant").

`session_cache` keys imaging sessions by their SVD / basis signature; the pose / 3-D steps instead depend on a handful of
files per session (the 3-D frames, v36 licks.csv, a cam1 windows.npz, ...). Here a result is stored with the size +
mtime of every input file and a code VERSION string; a run recomputes a session only when one of those changed, so a
new O2 batch costs only its own sessions:

    T = result_cache.cached(cache_dir, "PS93_20260922", [f_frames, f_licks], "jaw_onsets-v1", lambda: compute(...))

Values are pickled (pandas objects, dicts, arrays) to ``<cache_dir>/<key>.pkl``, written atomically. Bump the version
whenever the computation changes. ``WFIELD_RESULT_CACHE=0`` disables reads (always recompute; still writes).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path


def signature(inputs, version: str) -> tuple:
    sig = [str(version)]
    for f in inputs:
        f = Path(f)
        if f.exists():
            st = f.stat()
            sig.append((f.name, st.st_size, st.st_mtime_ns))
        else:
            sig.append((f.name, None, None))
    return tuple(sig)


def code_version(*modules, extra="") -> str:
    """A version string from the SOURCE of the given modules (+ ``extra``, e.g. a parameter repr): editing any of them
    invalidates the cache without anyone remembering to bump a number."""
    import hashlib
    import inspect
    h = hashlib.sha1(str(extra).encode())
    for m in modules:
        h.update(Path(inspect.getfile(m)).read_bytes())
    return h.hexdigest()[:12]


def cached(cache_dir, key: str, inputs, version: str, compute):
    cache_dir = Path(cache_dir)
    f = cache_dir / f"{key}.pkl"
    sig = signature(inputs, version)
    if f.exists() and os.environ.get("WFIELD_RESULT_CACHE", "1") != "0":
        try:
            with open(f, "rb") as h:
                got = pickle.load(h)
            if got.get("sig") == sig:
                return got["value"]
        except Exception:
            pass
    value = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # CONCURRENT WRITERS (2026-10-21): the first LP_licks render ran 3 arm processes x 2 map workers that reached the
    # same sessions at the same moment; a shared ".pkl.tmp" name and an unguarded replace raised WinError 32 and DROPPED
    # four June sessions from some arms. Per-process tmp name, and a failed publish is not an error -- the value is
    # correct either way and another writer's copy is equivalent (same inputs, same version) -- as in session_cache.
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as h:
        pickle.dump({"sig": sig, "value": value}, h, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        os.replace(tmp, f)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return value
