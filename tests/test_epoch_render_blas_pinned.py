"""`epoch_grant_figures.main` pins BLAS to `parallel.BLAS_THREADS` BEFORE any family runs (CC-B 11-28).

WHY. `parallel.fan_out` pins the calling process and never restores it, so an arm's BLAS thread count depended on
whether a fanned family (14m / 15r map collects) had already run in that process. The 13s state decoder projects every
session onto the joint basis in float32 BLAS; measured on PS92_0806 (debug record run_records/2026-11-28/
rerender_refreeze_ccB1128/debug_13s): 24 threads -> signal digest e9005e399985, 2 threads -> bdd4f07fd40b (max rel diff
4.6e-5), so `--only 13s` and `--only 14m 13s` froze DIFFERENT state_segments ids for the same data.

BEHAVIORAL (the lesson of test_render_is_machine_independent.py): asserts the loaded BLAS reports the pinned count at
the moment the first family starts, not that a call is present.
"""
import pytest

threadpoolctl = pytest.importorskip("threadpoolctl")

from wfield_local import epoch_grant_figures as egf  # noqa: E402
from wfield_local import parallel  # noqa: E402


class _Stop(Exception):
    pass


def test_blas_is_pinned_when_the_first_family_runs(monkeypatch, tmp_path):
    seen = {}

    def first_family(want, out):
        seen["threads"] = parallel.blas_threads()
        raise _Stop

    monkeypatch.setattr(egf, "_behaviour_families", first_family)
    monkeypatch.setattr(egf, "assert_writable", lambda p: None)
    from wfield_local import lp_licks
    monkeypatch.setattr(lp_licks, "output_dir", lambda p: p)
    with threadpoolctl.threadpool_limits(limits=None):            # restores this test process's limits afterwards
        threadpoolctl.threadpool_limits(limits=max(parallel.BLAS_THREADS + 1, 3), user_api="blas")
        if parallel.blas_threads() == parallel.BLAS_THREADS:      # pragma: no cover - single-core BLAS build
            pytest.skip("BLAS cannot be set to a count other than BLAS_THREADS here")
        with pytest.raises(_Stop):
            egf.main(["--output", str(tmp_path), "--only", "1c"])
    assert seen["threads"] == parallel.BLAS_THREADS
