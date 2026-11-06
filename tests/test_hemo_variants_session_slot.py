"""`hemo_variants.compute / write / CLI` default to the SESSION'S 470 slot (code review 11-04 item 3, Priya OK 11-05).

The default used to be the cohort FUNC = 1, so five diagnostics that call `hv.compute` without ``func``
(precue_significance, hemo_residual_check, postcue_window_test, precue_window_sweep, filter_acausality_test) built
PS92_0828's corrected signal with 415 and 470 swapped. Production (`preprocess`) passes the slot and was never affected.
"""
import inspect

from wfield_local import hemo_variants as hv


def test_the_default_is_the_session_slot():
    assert inspect.signature(hv.compute).parameters["func"].default is None
    assert inspect.signature(hv.write).parameters["func"].default is None
    assert hv.functional_channel("PS92_0828") == 0
    assert hv.functional_channel("PS92_0806") == hv.FUNC == 1


def test_compute_resolves_the_slot_before_slicing():
    src = inspect.getsource(hv.compute)
    assert "if func is None:\n        func = functional_channel(session)" in src
    assert src.index("func = functional_channel(session)") < src.index("svt[:, func::2]")


def test_cli_default_is_per_session():
    src = inspect.getsource(hv.main)
    assert '"--functional-channel", type=int, default=None' in src
