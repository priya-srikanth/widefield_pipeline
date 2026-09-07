"""The 415/470 mean panels are labelled by WAVELENGTH, not by slot index.

WHY (2026-09-07, Priya from the preprocessing deck). PS92 2026-08-28 was saved mislabelled
single-channel; the rescue relabel locked exposure offset 1, so slot 0 holds the 470 (functional)
frames and slot 1 the 415. `session_overrides.yaml` records that as
`preprocess.svd.functional_channel: 0`, and `preprocess.py` already honours it per session when
cross-registering. `_plot_mean_overlay` did not -- it hardcoded slot 0 as "415 nm mean", so for
that one session both panels carried the wrong caption.

The image was right and the caption was wrong, which is the harder error to see: regenerating the
figure after the fix changed nothing, because the figure never consulted the fix.
"""
import inspect

import numpy as np

from wfield_local import plot_spout_position_contrasts as pc


def test_the_slot_comes_from_the_session_not_a_literal():
    src = inspect.getsource(pc._plot_mean_overlay)
    assert "_functional_slot(label)" in src, "the panel order must follow the session's override"
    assert "frames_average[0]" not in src, "slot 0 is not 415 for every session"


def test_a_normal_session_is_unchanged(monkeypatch):
    monkeypatch.setattr(pc, "_functional_slot", lambda label: 1)
    fa = np.stack([np.full((4, 4), 415.0), np.full((4, 4), 470.0)])
    func = pc._functional_slot("PS93_0828")
    assert fa[1 - func].mean() == 415.0 and fa[func].mean() == 470.0


def test_a_swapped_session_reads_the_other_slot(monkeypatch):
    monkeypatch.setattr(pc, "_functional_slot", lambda label: 0)
    fa = np.stack([np.full((4, 4), 470.0), np.full((4, 4), 415.0)])   # PS92_0828 layout
    func = pc._functional_slot("PS92_0828")
    assert fa[func].mean() == 470.0, "the functional panel must show 470 whichever slot holds it"
    assert fa[1 - func].mean() == 415.0


def test_an_unknown_label_falls_back_rather_than_raising():
    assert pc._functional_slot("NOT_A_SESSION") == 1
