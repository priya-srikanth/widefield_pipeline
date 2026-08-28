"""A figure must not depend on how many cores drew it.

MEASURED 2026-08-28. The same code and the same data, `grant_figures --only 5d --window cue
--variant lick`, rendered three ways:

    parallel -j6, BLAS 2 threads    288,642 B   d0d2d0d8...
    serial   -j1, BLAS 2 threads    288,642 B   d0d2d0d8...   <- identical to the parallel one
    serial   -j1, BLAS uncapped     289,032 B   d29f5ecb...   <- differs

Same BLAS with different `-j` is byte-identical; same `-j` with different BLAS is not. So the
parallel fan-out was never the variable -- BLAS thread count is. Threads change the order of a
floating-point reduction, the last bits move, and a bootstrap CI lands a pixel elsewhere.

WHY THAT MATTERED. `_run_parallel` capped BLAS per worker; the serial path capped nothing. The
render was therefore machine-dependent (24-core box vs laptop) AND path-dependent (`--jobs 1` vs
the default), which is exactly the reproducibility d084ede set out to establish when it made the
bootstrap seeds stable and per-day. Stable seeds are necessary and were not sufficient.
"""
import ast
import inspect
from pathlib import Path

import pytest

from wfield_local import grant_figures

ROOT = Path(__file__).resolve().parent.parent


def test_pin_blas_actually_changes_the_loaded_library():
    """BEHAVIOURAL, because the structural version of this test passed while the fix did not work.

    The first attempt set OMP_NUM_THREADS and friends inside `main()`. A test asserting `_pin_blas`
    is *called* there passed; the render still produced the uncapped bytes, because OpenBLAS reads
    those variables when numpy is IMPORTED and `main()` runs long after this module's own import.
    `_run_parallel` only appeared to validate the approach because it SPAWNS -- its workers import
    numpy afresh with the variables already set.

    So this asserts the observable property -- the loaded BLAS reports the pinned thread count --
    rather than the presence of a call that may do nothing.
    """
    tp = pytest.importorskip("threadpoolctl")
    before = [d["num_threads"] for d in tp.threadpool_info() if d["user_api"] == "blas"]
    if not before:
        pytest.skip("no BLAS backend visible to threadpoolctl")
    ctl = grant_figures._pin_blas(1)
    try:
        after = [d["num_threads"] for d in tp.threadpool_info() if d["user_api"] == "blas"]
        assert after and all(n == 1 for n in after), (
            f"BLAS still at {after} after _pin_blas(1); environment variables alone cannot "
            f"re-thread an already-imported OpenBLAS, which is why this test is not structural")
    finally:
        if ctl is not None:
            ctl.restore_original_limits()


def test_pin_blas_runs_on_the_serial_path_too():
    """Capping only inside _run_parallel makes `--jobs 1` disagree with the default."""
    tree = ast.parse(inspect.getsource(grant_figures.main))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_pin_blas" in called, "the escape hatch must not change the answer"


def test_pin_blas_covers_every_backend_numpy_might_use():
    src = inspect.getsource(grant_figures._pin_blas)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert var in src, f"{var} unpinned -- one uncapped backend is enough to move the bits"


def test_the_thread_count_is_a_constant_not_a_core_count():
    """Deriving it from cpu_count would reintroduce exactly the machine dependence."""
    assert isinstance(grant_figures.BLAS_THREADS, int) and grant_figures.BLAS_THREADS >= 1
    src = inspect.getsource(grant_figures._pin_blas)
    assert "cpu_count" not in src, "BLAS_THREADS must not vary with the machine"
