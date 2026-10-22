"""Faster pixel-map collection for the epoch map families (Priya, 2026-10-21: "there is plenty of free CPU and RAM -
could we make this faster in the future" -> "yes").

MEASURED BEFORE CHANGING ANYTHING (10-21): the 15r / 14 collectors (`position_reference_maps.maps_by_epoch`,
`beta_maps.maps_by_epoch`) walked ~120 sessions ONE AT A TIME in one process, ~13 s per session per arm for the
position maps alone (`session_raw_maps`, PS94_0820 lick: 13.3 s, 30 MB pickled), and kept the per-session maps only in
an in-process lru_cache. A full `--only 15r 15e 14m` render ran ~5 h at ~12 % CPU on a 24-thread / 64 GB box, and
it rebuilt every map even when only the SIGNIFICANCE TESTS had changed (the 10-21 normalised-test rerender).

TWO FIXES LIVE HERE (the third, concurrent arms, is `epoch_grant_figures --parallel-arms`):

  1. A DISK CACHE per session x arm (`session_cache`, kind ``<prefix>-<align>-<variant>-<digest>``). The digest covers
     the MAP-BUILDING code only -- the source of the functions that build a session's maps and of the modules they call
     (`code_digest`), the arguments, a `MAPS_VERSION` constant, the whole of `defaults.yaml` and the effective ENL
     window -- so editing a significance test, a figure or a comment elsewhere keeps the cache warm, and editing the
     map code invalidates it without anyone remembering to bump a number. The session's DATA files are in the key via
     `session_signature` (SVT, U, LocaNMF C, h5, behaviour trials) plus the extra inputs each caller passes
     (U_atlas, quiet-frame mask). Under WIDEFIELD_LICK_SOURCE=lp the session cache tags the kind itself.
  2. THE SESSION LOOP FANNED OVER CORES (`collect`, rule 6), results re-ordered by session label so nothing downstream
     sees completion order (rule 9). WIDEFIELD_MAP_JOBS sets the worker count (default min(4, default_jobs())); 1 =
     serial in-process, which is also what a worker of `--parallel-arms` gets unless told otherwise.

Disable the cache with WIDEFIELD_NO_CACHE=1 (session_cache's switch).
"""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

#: Bump when the per-session map computation changes in a way `code_digest` cannot see (data-format changes in a
#: helper module that is not hashed, a dependency upgrade that changes results).
MAPS_VERSION = 1
MAP_JOBS_ENV = "WIDEFIELD_MAP_JOBS"


def map_jobs() -> int:
    from wfield_local import parallel
    v = os.environ.get(MAP_JOBS_ENV, "").strip()
    return max(1, int(v)) if v else max(1, min(4, parallel.default_jobs()))


def code_digest(*objs) -> str:
    """sha1 of the SOURCE of each function / module given (order-sensitive)."""
    h = hashlib.sha1()
    for o in objs:
        try:
            src = inspect.getsource(o)
        except (OSError, TypeError):
            src = repr(o)
        h.update(src.encode())
    return h.hexdigest()[:12]


def stat_sig(path) -> str:
    """size:mtime of an input file ('-' if absent), for the cache params."""
    if not path:
        return "-"
    p = Path(path)
    if not p.exists():
        return f"{p.name}:-"
    st = p.stat()
    return f"{p.name}:{st.st_size}:{st.st_mtime_ns}"


def cache_kind(prefix: str, align: str, variant: str, spec: dict, code: str) -> str:
    """``<prefix>-<align>-<variant>-<digest>``: digest of spec + code + MAPS_VERSION + defaults.yaml + ENL window."""
    from wfield_local import config, lp_licks
    full = dict(spec)
    full["maps_version"] = MAPS_VERSION
    full["code"] = code
    full["defaults"] = repr(config.defaults())
    full["enl_window_s"] = lp_licks.enl_window_s() if align == "precue" else None
    digest = hashlib.sha1(repr(sorted(full.items(), key=repr)).encode()).hexdigest()[:12]
    return f"{prefix}-{align}-{variant}-{digest}"


def collect(items, worker, *, label="session", jobs=None):
    """``({item[0]: result}, [(item, error)])`` -- `worker(item)` over `items`, fanned out when jobs > 1.

    ``item[0]`` must be the session label: results are keyed (and so re-ordered) by it, never by completion order."""
    from wfield_local import parallel
    items = list(items)
    jobs = map_jobs() if jobs is None else jobs
    if jobs <= 1 or len(items) <= 1:
        res, fail = [], []
        for it in items:
            try:
                res.append((it, worker(it)))
            except Exception as ex:                                     # noqa: BLE001
                fail.append((it, f"{type(ex).__name__}: {str(ex)[:120]}"))
    else:
        res, fail = parallel.fan_out(items, worker, jobs=jobs, label=label,
                                     log=lambda m: print(m[:100], flush=True))
    return {it[0]: r for it, r in sorted(res, key=lambda kv: kv[0][0])}, fail
