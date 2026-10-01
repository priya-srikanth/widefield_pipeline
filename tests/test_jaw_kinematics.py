"""jaw_kinematics: the per-trial jaw scalars do what stroke_orofacial's jaw_pertrial (v7a.5) does.

Synthetic jaw traces with exact arithmetic (baseline alternates +-0.5 px, so mean 0 and population sd 0.5),
then a parity check against THEIR `_compute_per_trial`, loaded from the sibling repo by file path.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import config
from wfield_local import jaw_kinematics as jk

FPS = 250.0
SRC = Path(__file__).resolve().parents[2] / "stroke_orofacial_pipeline" / "src" / "stroke_orofacial" / "dlc_kinematics"


@pytest.fixture(autouse=True)
def _no_yaml_overrides(monkeypatch):
    """Pin the ported DEFAULTS: a future `orofacial_kinematics.jaw` block in defaults.yaml must not move these."""
    monkeypatch.setattr(config, "defaults", lambda session=None: {})


def _trace(n=12000):
    return np.where(np.arange(n) % 2 == 0, 0.5, -0.5)


def _trials(cues, positions=None):
    positions = positions or ["far_L"] * len(cues)
    return pd.DataFrame({"trial_id": np.arange(len(cues)) + 1, "cue_frame": cues, "position": positions})


def test_opening_bump_after_cue_passes_qc_with_exact_features():
    y = _trace()
    c = 2000
    y[c + 25:c + 75] += 15.0                                   # 50 frames (200 ms) open, starting 100 ms post-cue
    t, meta = jk.jaw_pertrial(y, FPS, _trials([c]))
    r = t.iloc[0]
    assert r.jaw_baseline_mean == pytest.approx(0.0, abs=1e-12)
    assert r.jaw_baseline_sd == pytest.approx(0.5)
    assert r.jaw_peak_pos_deflection == pytest.approx(15.5)
    assert r.jaw_peak_neg_deflection == pytest.approx(-0.5)
    assert r.jaw_peak_abs_deflection == pytest.approx(15.5)
    assert r.jaw_peak_selected_deflection == pytest.approx(15.5)
    assert r.jaw_thresh_abs == 4.0 and r.jaw_thresh_px_used == "px_floor"     # 4 * 0.5 = 2 < 4 px floor
    assert r.jaw_n_frames_over_thresh == 50
    assert bool(r.jaw_pass_qc)
    assert meta["sustained_window_n_frames"] == 8 and meta["detect_win_ms"] == [0.0, 5000.0]


def test_no_movement_fails_and_single_frame_outlier_does_not_pass():
    y = _trace()
    y[6000 + 300] += 50.0                                      # one-frame tracking outlier in trial 2
    t, _ = jk.jaw_pertrial(y, FPS, _trials([2000, 6000]))
    quiet, outlier = t.iloc[0], t.iloc[1]
    assert not quiet.jaw_pass_qc and quiet.jaw_n_frames_over_thresh == 0
    assert outlier.jaw_peak_pos_deflection == pytest.approx(50.5)
    assert outlier.jaw_n_frames_over_thresh == 1
    assert not outlier.jaw_pass_qc                             # 1 frame is not >= 5 of 8


def test_density_is_five_of_eight_not_a_contiguous_run():
    y = _trace()
    c = 2000
    y[c + 50:c + 58] += np.array([10, 10, 0, 10, 0, 10, 10, 0])       # 5 supra in 8, not contiguous
    assert bool(jk.jaw_pertrial(y, FPS, _trials([c]))[0].iloc[0].jaw_pass_qc)
    y2 = _trace()
    y2[c + 50:c + 58] += np.array([10, 10, 0, 10, 0, 10, 0, 0])       # 4 in 8 -> fail
    assert not jk.jaw_pertrial(y2, FPS, _trials([c]))[0].iloc[0].jaw_pass_qc


def test_negative_deflection_selected_with_sign_and_sd_threshold_wins_on_noisy_baseline():
    y = _trace() * 4.0                                         # +-2 px: sd 2 -> 4 * 2 = 8 > 4 px floor
    c = 2000
    y[c + 100:c + 140] -= 20.0
    r = jk.jaw_pertrial(y, FPS, _trials([c]))[0].iloc[0]
    assert r.jaw_thresh_px_used == "sd_scaled" and r.jaw_thresh_abs == pytest.approx(8.0)
    assert r.jaw_peak_selected_deflection == pytest.approx(-22.0)
    assert bool(r.jaw_pass_qc)


def test_degenerate_baselines_give_nan_none_and_fail():
    y = _trace()
    y[4000 - 125:4001] = 0.0                                   # perfectly flat baseline (v3.4 baseline fill)
    y[4100:4200] += 30.0                                       # would pass if the baseline were judgeable
    t, _ = jk.jaw_pertrial(y, FPS, _trials([0, 4000, np.nan]))
    for _, r in t.iterrows():
        assert r.jaw_thresh_px_used == "none" and not r.jaw_pass_qc and r.jaw_n_frames_over_thresh == 0
        assert np.isnan(r.jaw_peak_abs_deflection) and np.isnan(r.jaw_thresh_abs)
    assert np.isnan(t.iloc[0].jaw_baseline_sd)                 # cue at frame 0: only 1 baseline frame
    assert t.iloc[1].jaw_baseline_sd == 0.0
    assert t.iloc[2].cue_frame_idx == -1


def test_float_cue_frame_rounds_to_nearest_and_positions_carry_through():
    y = _trace()
    y[2025:2075] += 15.0
    t, _ = jk.jaw_pertrial(y, FPS, _trials([2000.4, 1999.6], ["close_R", "far_center"]))
    assert list(t.cue_frame_idx) == [2000, 2000]
    assert list(t.position) == ["close_R", "far_center"]
    assert t.jaw_peak_pos_deflection.tolist() == pytest.approx([15.5, 15.5])


def test_yaml_block_deep_merges_over_defaults(monkeypatch):
    monkeypatch.setattr(config, "defaults",
                        lambda session=None: {"orofacial_kinematics": {"jaw": {"deflection_threshold_min_px": 9.0}}})
    p = jk.params()
    assert p["deflection_threshold_min_px"] == 9.0 and p["sustained_window_n_frames"] == 8
    assert jk.params({"sustained_window_n_frames": 4})["sustained_window_n_frames"] == 4


# --------------------------------------------------------------------------- parity with the source module

def _load_source_jaw_pertrial():
    """Import THEIR jaw_pertrial.py by path. Its pyarrow + stroke_orofacial imports are stubbed for the duration
    of the import only (none is used by `_compute_per_trial`); nothing is installed and sys.modules is restored."""
    path = SRC / "jaw_pertrial.py"
    if not path.exists():
        pytest.skip(f"source repo not found at {path}")
    names = ["pyarrow", "pyarrow.parquet", "stroke_orofacial", "stroke_orofacial.animals",
             "stroke_orofacial.config_loader", "stroke_orofacial.dlc_kinematics",
             "stroke_orofacial.dlc_kinematics._writers", "stroke_orofacial.dlc_kinematics.tongue_pertrial",
             "stroke_orofacial.manifest", "stroke_orofacial.paths"]
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for n in names:
            m = types.ModuleType(n)
            m.__getattr__ = lambda attr: object()               # PEP 562: any `from n import attr` resolves
            sys.modules[n] = m
        spec = importlib.util.spec_from_file_location("_src_jaw_pertrial", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod                            # @dataclass resolves its module by name
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.modules.pop("_src_jaw_pertrial", None)
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def test_parity_with_source_compute_per_trial():
    src = _load_source_jaw_pertrial()
    rng = np.random.default_rng(7)
    n = 40000
    y = np.cumsum(rng.normal(0, 0.3, n)) * 0.2 + rng.normal(0, 1.0, n)
    for s in rng.integers(0, n - 200, 120):                    # openings of random size and sign
        y[s:s + rng.integers(3, 80)] += rng.choice([-1, 1]) * rng.uniform(1, 20)
    y[rng.random(n) < 0.03] = np.nan                           # dropouts
    y[5000:5200] = np.nan                                      # one trial with no baseline at all
    y[9000:9126] = 0.0                                         # one with a flat baseline
    cues = np.r_[5125, 9125, 30, n - 20, rng.integers(200, n - 200, 200)]
    kw = dict(detect_win_ms=(0.0, 5000.0), baseline_win_ms=(-500.0, 0.0), sd_mult=4.0, min_px=4.0,
              density_thr=0.625, window_n=8)
    ours, _ = jk.jaw_pertrial(y, FPS, _trials(list(cues)))
    for k, c in enumerate(cues):
        ev = types.SimpleNamespace(trial_index=k, side="L", tone_frame_idx=int(c))
        theirs = src._compute_per_trial(ev, y, FPS, n, **kw)
        for col in jk.JAW_COLUMNS:
            a, b = ours.iloc[k][col], theirs[col]
            if b is None:
                assert np.isnan(a), (k, col)
            elif isinstance(b, str):
                assert a == b, (k, col)
            else:
                assert a == b, (k, col, a, b)                  # exact: same arithmetic, same order
