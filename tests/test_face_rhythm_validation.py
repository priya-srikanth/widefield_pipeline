"""wfield_local.face_rhythm_validation on synthetic trials: a factor planted as a bump at every left lick is recovered by the
lick-triggered average (peak at lag 0, left only), lick rates on the trial grid integrate to the lick count."""
import numpy as np
import pandas as pd

from wfield_local import face_rhythm_validation as frv


def _proj(cues, lick_t, lick_dir):
    t_ms = np.arange(-2000.0, 4001.0, 40.0)
    W = np.zeros((len(cues), len(t_ms), 2))
    for t, d in zip(lick_t, lick_dir):
        i = int(np.searchsorted(cues, t + 2.0, side="right") - 1)
        if d == "left":
            W[i, :, 0] += np.exp(-0.5 * ((cues[i] + t_ms / 1000 - t) / 0.05) ** 2)
    W[..., 1] = 1.0
    return {"trials": W, "trials_D": np.ones(W.shape[:2]), "t_ms": t_ms, "cue_daq_s": np.asarray(cues, float)}


def test_lick_triggered_and_rates():
    cues = np.array([10.0, 20.0, 30.0])
    lick_t = np.array([10.5, 10.7, 20.6, 21.0, 30.4, 30.9])
    lick_dir = np.array(["left", "right", "left", "right", "left", "centre"])
    z = _proj(cues, lick_t, lick_dir)
    L = pd.DataFrame({"daq_on_s": lick_t, "direction": lick_dir, "freq_hz": np.nan})
    X, used = frv.lick_triggered(z, L)
    assert len(used) == 6 and X.shape == (6, len(frv.LAGS), 2)
    left = (used.direction == "left").to_numpy()
    peak = frv.LAGS[np.argmax(X[left].mean(0)[:, 0])]
    assert abs(peak) < 0.041                                       # bump at the lick
    assert X[~left][:, np.abs(frv.LAGS) < 0.01, 0].max() < 0.2     # not at the other licks
    B = frv.bin_table(z, L.assign(trial_id=0), None)
    per_trial = B.groupby("trial").rate_all.sum() * 0.04           # integral of a rate = count
    assert np.allclose(per_trial.to_numpy(), [2, 2, 2], atol=0.05)
