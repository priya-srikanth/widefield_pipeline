"""One definition of how this pipeline fans work over cores, and how it pins the arithmetic.

WHY A MODULE. `grant_figures` grew a process pool on 2026-08-28 and the stages around it did not,
so the nightly spent 9 h 37 m of which one stage was parallel. Adding a pool per stage would have
put four copies of the same three decisions in the tree -- how many workers, how many BLAS threads,
what to do when a unit dies -- and this repo has already had to collapse six copies of a decoder
recipe, three of a lick discriminator and two of an engaged cut. The copies always agree at first.

WHAT THE CALLER STILL OWNS: the UNIT. That is a per-stage judgement and not a mechanical one --
`poststroke_section_g` fans over animals rather than the six (animal, tag) pairs available, because
an animal's two tags share a frozen-decoder spec and `load_or_fit` writes it.
"""
from __future__ import annotations

import concurrent.futures as cf
import os

#: Pinned so a figure does not depend on how many cores drew it. See
#: `tests/test_render_is_machine_independent.py`: the same code and data rendered 289,032 B with
#: BLAS uncapped and 288,642 B at 2 threads. NOT derived from `cpu_count` -- deriving it would
#: reintroduce exactly the machine dependence it removes.
BLAS_THREADS = 2


def default_jobs(cap: int = 8) -> int:
    """Workers when the caller does not say. PARALLEL, because serial defaults hide.

    `- 2` leaves the box usable while a stage runs; the cap holds peak RSS down (grant workers
    measured ~1.4 GB each) and leaves room for the rest of the nightly.
    """
    return max(1, min(cap, (os.cpu_count() or 4) - 2))


def pin_blas(threads: int = BLAS_THREADS):
    """Fix BLAS threading for THIS process; returns a handle to restore, or None.

    ENVIRONMENT VARIABLES ALONE ARE NOT ENOUGH, which cost a wrong fix on 2026-08-28: OpenBLAS
    reads them when numpy is IMPORTED, so setting them in a `main()` that runs after the module's
    own `import numpy` changes nothing. A process pool gets away with it only because it SPAWNS --
    each worker imports numpy afresh. In-process needs a runtime control, so `threadpoolctl`
    re-threads the already-loaded library; the variables are still set, for the workers.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        import threadpoolctl
    except ImportError:                                  # pragma: no cover
        print("  !! threadpoolctl missing: BLAS stays at its import-time thread count, so this "
              "run is NOT reproducible against one that had it", flush=True)
        return None
    return threadpoolctl.threadpool_limits(limits=threads, user_api="blas")


def fan_out(items, worker, *, jobs=None, label="unit", log=print):
    """Run `worker(item)` over `items` in spawned processes. Returns (results, failures).

    `worker` must be a MODULE-LEVEL callable (spawn pickles by name) and should return whatever the
    caller wants collected. Results come back in COMPLETION order paired with their item, so a
    caller that needs input order must sort.

    A raising unit is caught, recorded, and reported by name -- never swallowed. Callers decide
    whether a partial result may be written; most should refuse, because a short output file looks
    identical to a legitimately short one.
    """
    items = list(items)
    n = min(len(items), default_jobs() if jobs is None else jobs)
    pin_blas()
    if n <= 1 or len(items) <= 1:
        results, failures = [], []
        for it in items:
            try:
                results.append((it, worker(it)))
            except Exception as ex:                      # noqa: BLE001
                failures.append((it, f"{type(ex).__name__}: {ex}"))
                log(f"  !! {label} {it} FAILED: {type(ex).__name__}: {ex}")
        return results, failures

    log(f"  {len(items)} {label}(s) over {n} worker(s)")
    results, failures = [], []
    ctx = __import__("multiprocessing").get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
        futs = {pool.submit(worker, it): it for it in items}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            it = futs[fut]
            try:
                results.append((it, fut.result()))
                log(f"  [{i}/{len(items)}] {label} {it}: ok")
            except Exception as ex:                      # noqa: BLE001
                failures.append((it, f"{type(ex).__name__}: {ex}"))
                log(f"  !! [{i}/{len(items)}] {label} {it} FAILED: {type(ex).__name__}: {ex}")
    return results, failures
