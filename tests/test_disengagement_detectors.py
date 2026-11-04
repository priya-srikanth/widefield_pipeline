"""Re-verify at 3a772a26, item 1: disengagement_tests' RESPONSE measures (hit, latency, n_post, n_pre2) stay on the deck's
discriminator (`analysis_kit.lick_samples`) -- equal to the pre-change values on a real session -- while the lick COUNTS
(ILIs, bouts, licks / bout) use the behaviour detector (`analysis_kit.lick_count_samples`)."""
from __future__ import annotations

import numpy as np
import pytest


def _inputs(lab):
    from wfield_local import analysis_kit as ak
    try:
        b = ak.session_behavior(lab)
    except Exception as ex:                                             # noqa: BLE001 - no share on this box
        pytest.skip(f"{lab}: {type(ex).__name__}")
    if b is None:
        pytest.skip(f"{lab}: no behaviour")
    sr = b.sample_rate
    cs = np.asarray(b.cue_samples, float) / sr
    order = np.array([r["order"] for r in b.trials], int)
    cue = cs[order]
    nxt = np.append(cs[1:], np.inf)[order]
    lk_b = np.sort(np.asarray(ak.lick_samples(b.session), float) / sr)
    lk_a = np.sort(np.asarray(b.lick_samples, float) / sr)
    return b, cue, nxt, cs[0], lk_a, lk_b


@pytest.mark.parametrize("lab", ["PS93_0908"])
def test_response_measures_match_the_pre_change_values(lab, monkeypatch):
    # the script does os.environ.setdefault("WIDEFIELD_PRE_FROM", "0801") at import: set it through monkeypatch first so
    # the import cannot leak it into later tests (it did: lp_licks / rebuild_root / share_writers ran in mode pre0801)
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    dt = pytest.importorskip("scripts.disengagement_tests")
    b, cue, nxt, t0, lk_a, lk_b = _inputs(lab)
    deliv = np.zeros(len(b.trials))
    new = dt._trial_rows(b.trials, cue, nxt, t0, 3.5, deliv, lk_a, lk_b)
    old = dt._trial_rows(b.trials, cue, nxt, t0, 3.5, deliv, lk_b, lk_b)   # pre-change: one detector (B) for all
    for k in ("hit", "n_post", "n_pre2"):
        assert [r[k] for r in new] == [r[k] for r in old], k
    np.testing.assert_array_equal([r["lat"] for r in new], [r["lat"] for r in old])
    assert len(lk_a) != len(lk_b)                                   # the detectors really differ on this session
    assert [r["lpb"] for r in new] != [r["lpb"] for r in old]        # ... and the counts moved with detector A
