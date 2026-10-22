"""fr_groups: grouped FaceRhythm regressors (lick amplitude = sum of lick factors, side-pooled direction, others)."""
import numpy as np
import pytest

from wfield_local import fr_groups


def _sig(n=50, k=10):
    t = np.arange(n) * 0.032
    rng = np.random.default_rng(0)
    return {f"fr_{j}": (t, rng.normal(size=n)) for j in range(k)}


def test_mode_a_sums_and_keeps_others():
    sig = _sig()
    out = fr_groups.group(sig, "pilot_v1", "A", set_name="pilot_pre_PS92_PS93")
    G = fr_groups.GROUPINGS["pilot_v1"]
    assert list(out) == ["fr_lick", "fr_left", "fr_centre", "fr_right"] + [f"fr_other_{k}" for k in G["other"]]
    np.testing.assert_allclose(out["fr_lick"][1], sum(sig[f"fr_{k}"][1] for k in G["lick"]))
    np.testing.assert_allclose(out["fr_left"][1], sig["fr_0"][1] + sig["fr_5"][1])
    np.testing.assert_array_equal(out["fr_other_2"][1], sig["fr_2"][1])


def test_mode_b_has_no_direction_traces():
    out = fr_groups.group(_sig(), "pilot_v1", "B")
    assert not any(k in out for k in ("fr_left", "fr_centre", "fr_right"))
    assert "fr_lick" in out


def test_blanked_frames_stay_blank_in_sums():
    sig = _sig()
    v = sig["fr_4"][1].copy()
    v[3] = np.nan
    sig["fr_4"] = (sig["fr_4"][0], v)
    out = fr_groups.group(sig, "pilot_v1", "A")
    assert np.isnan(out["fr_lick"][1][3]) and np.isnan(out["fr_centre"][1][3])
    assert np.isfinite(out["fr_left"][1][3])


def test_wrong_set_and_bad_env_refused(monkeypatch):
    with pytest.raises(ValueError):
        fr_groups.group(_sig(), "pilot_v1", "A", set_name="master_pre_v1")       # factor numbers shift between fits
    monkeypatch.setenv(fr_groups.ENV, "pilot_v1:C")
    with pytest.raises(ValueError):
        fr_groups.spec()
    monkeypatch.setenv(fr_groups.ENV, "pilot_v1:B")
    assert fr_groups.spec() == ("pilot_v1", "B")
