"""tongue_detect: each ported stage does what stroke_orofacial's v7 code does, on synthetic tongue traces.

Two kinds of test:
  * behaviour -- small hand-built traces with a known answer (a lick train, a one-frame jump, an isolated
    two-frame blip, a frozen plateau, an x snap-back), asserting what each stage is FOR;
  * parity -- the SOURCE modules loaded straight from `../stroke_orofacial_pipeline` by file path (their
    package is not installed here and must not be imported normally; see `load_source`), run on randomised
    synthetic sessions and compared for EXACT equality. A tolerance would hide a transcription slip.

`synth_session` and the `source` fixture are imported by test_tongue_kinematics.py.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401  -- must be imported BEFORE the pyarrow stub below (pandas probes pyarrow)
import pytest

from wfield_local import tongue_detect as td

FPS = 250.0
SRC = Path(__file__).resolve().parents[2] / "stroke_orofacial_pipeline" / "src" / "stroke_orofacial"


def P(**over):
    """Detection params (DEFAULTS only -- tests must not depend on configs/defaults.yaml) with overrides."""
    return td._deep_merge(copy.deepcopy(td.DEFAULTS), over)


# --------------------------------------------------------------------------- synthetic data

def synth_session(seed=0, n_trials=6, trial_len=2500, extras=True):
    """A session shaped like `orofacial_clean` output: baseline-subtracted, baseline-fill frames at exactly 0,
    each lick a visible half-sine y bump (raw frames, sometimes a PCHIP-filled first/last frame) with the
    tongue invisible (baseline-fill) between licks. With ``extras``, randomly per trial: a one-frame y jump
    inside a lick, an isolated 2-frame high blip, an x snap-back-to-rest frame, a |x| > bound frame, a tall
    low-likelihood FIRST lick (exercises gate E + imputation), a close lick pair (exercises gate M), a ~300 ms
    held protrusion that wobbles without coming back down (peaks with no low context: gate D), a flat-topped
    protrusion (gate F), an apex-then-shoulder lick that only the unbounded "legacy" detector accepts.

    Returns y, x, fill_method, likelihood, cue frames, truth list of (trial, peak frame, peak time ms).
    """
    rng = np.random.default_rng(seed)
    N = 1000 + n_trials * trial_len + 2500
    y, x = np.zeros(N), np.zeros(N)
    fm = np.full(N, 3, np.int8)
    lk = rng.uniform(0.0, 0.3, N)
    cues = [1000 + k * trial_len for k in range(n_trials)]
    truth = []
    for k, c in enumerate(cues):
        t = c + int(rng.integers(30, 60))
        n_l = int(rng.integers(3, 9))
        dx = rng.uniform(-60, 60)
        tall_first = extras and rng.random() < 0.4
        close_at = int(rng.integers(1, n_l)) if extras and rng.random() < 0.4 else -1
        for j in range(n_l):
            dur = int(rng.integers(20, 29)) if j != close_at else int(rng.integers(8, 11))   # 80-112 ms
            h = rng.uniform(150, 220) if not (tall_first and j == 0) else rng.uniform(380, 450)
            u = (np.arange(dur) + 0.5) / dur
            seg = slice(t, t + dur)
            y[seg] = h * np.sin(np.pi * u) + rng.normal(0, 2, dur)
            x[seg] = dx * np.sin(np.pi * u) + rng.normal(0, 1.5, dur)
            fm[seg] = 0
            lk[seg] = rng.uniform(0.9, 1.0, dur) if not (tall_first and j == 0) else rng.uniform(0.5, 0.7, dur)
            for e in (t, t + dur - 1):
                if rng.random() < 0.4:
                    fm[e] = 1
                    lk[e] = 0.3
            pk = t + int(np.argmax(y[seg]))
            truth.append((k, pk, (pk - c) / FPS * 1000.0))
            if extras and rng.random() < 0.15:
                y[t + dur // 2 + 1] += 160.0                      # one-frame jump
            if extras and abs(dx) > 40 and rng.random() < 0.3:
                x[t + dur // 2] = rng.normal(0, 2)                 # x snaps back to rest for one frame
            if extras and rng.random() < 0.05:
                x[t + dur // 2] = 140.0 * np.sign(dx or 1.0)       # |x| beyond the hard bound
            gap = int(rng.integers(15, 26)) if j + 1 != close_at else int(rng.integers(0, 3))   # ILI 140-212 ms
            t += dur + gap
        s0 = c + 495                                               # after the train (ends <= +484 frames)
        if extras and rng.random() < 0.5 and np.all(fm[s0 - 5:s0 + 40] == 3):
            if rng.random() < 0.5:                                 # flat-topped protrusion -> gate F
                shape = np.r_[np.linspace(10, 150, 6), np.full(14, 150.0), np.linspace(150, 10, 6)]
                noise = rng.normal(0, 0.3, shape.size)
            else:                                                  # apex, shoulder, late fall -> only the
                shape = np.r_[np.linspace(10, 180, 8), [165.0], np.full(15, 160.0), np.linspace(160, 10, 8)]
                noise = rng.normal(0, 0.5, shape.size)             # unbounded "legacy" detector keeps it
            y[s0:s0 + shape.size] = shape + noise
            x[s0:s0 + shape.size] = rng.normal(0, 1.5, shape.size)
            fm[s0:s0 + shape.size] = 0
            lk[s0:s0 + shape.size] = rng.uniform(0.9, 1.0, shape.size)
        if extras and rng.random() < 0.5:                          # tongue held out ~300 ms, wobbling
            h0 = c + int(rng.integers(560, 630))          # after the train, inside 3 s
            m = 75
            ramp = np.r_[np.linspace(5, 170, 8), 170 + 25 * np.sin(np.arange(m - 8) / 4.0)]
            y[h0:h0 + m] = ramp + rng.normal(0, 2, m)
            x[h0:h0 + m] = rng.normal(0, 2, m)
            fm[h0:h0 + m] = 0
            lk[h0:h0 + m] = rng.uniform(0.9, 1.0, m)
        if extras and rng.random() < 0.5:                          # isolated short high blip
            b = c + int(rng.integers(715, 740))
            if np.all(fm[b - 10:b + 12] == 3):
                y[b:b + 2] = 200.0
                fm[b:b + 2] = 0
                lk[b:b + 2] = 0.95
    return y, x, fm, lk, cues, truth


# --------------------------------------------------------------------------- source loader

class _Stub(types.ModuleType):
    """A module whose every attribute exists (a dummy class) -- satisfies `from X import a, b` for the source's
    I/O-side imports (pyarrow, their config/manifest/paths, their writers), none of which the ported
    functions touch."""
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


_STUBS = ["pyarrow", "pyarrow.parquet", "stroke_orofacial.animals", "stroke_orofacial.config_loader",
          "stroke_orofacial.manifest", "stroke_orofacial.paths", "stroke_orofacial.dlc_kinematics._writers"]
_SRC_MODS = ["_detector", "_preclean", "_detector_v2", "_gates", "_angle", "tongue_pertrial"]


def load_source():
    """Load their dlc_kinematics modules by FILE PATH (importlib), with package shells + stubs in sys.modules.

    Returns (mods, restore). Their package is never installed or imported normally; ``restore()`` puts
    sys.modules back. The shells must stay registered while source functions run, because two of them import
    lazily at call time (`_angle._extract_cell50_own_bout_scalars`, `_extract_licking_dynamics_traces`).
    """
    names = ["stroke_orofacial", "stroke_orofacial.dlc_kinematics", *_STUBS,
             *[f"stroke_orofacial.dlc_kinematics.{m}" for m in _SRC_MODS]]
    saved = {n: sys.modules[n] for n in names if n in sys.modules}

    def restore():
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)

    pkg = types.ModuleType("stroke_orofacial")
    pkg.__path__ = [str(SRC)]
    dk = types.ModuleType("stroke_orofacial.dlc_kinematics")
    dk.__path__ = [str(SRC / "dlc_kinematics")]
    sys.modules["stroke_orofacial"] = pkg
    sys.modules["stroke_orofacial.dlc_kinematics"] = dk
    for n in _STUBS:
        sys.modules[n] = _Stub(n)
    mods = {}
    try:
        for m in _SRC_MODS:
            name = f"stroke_orofacial.dlc_kinematics.{m}"
            spec = importlib.util.spec_from_file_location(name, SRC / "dlc_kinematics" / f"{m}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mods[m] = mod
    except Exception:
        restore()
        raise
    return mods, restore


@pytest.fixture(scope="module", name="source")
def source_fixture():
    if not (SRC / "dlc_kinematics" / "tongue_pertrial.py").exists():
        pytest.skip(f"stroke_orofacial source not found at {SRC}")
    mods, restore = load_source()
    # A STALE sibling checkout must skip, not fail (2026-10-07: the behaviour box's stroke_orofacial was ~90
    # commits behind, pre-dating `_v7_cleaned_bundle_arrays`, and the parity test's AttributeError blocked an
    # unrelated widefield push). Parity is only meaningful against the reference version the port follows.
    missing = [f"{m}.{a}" for m, attrs in _REQUIRED_SYMBOLS.items() for a in attrs
               if m in mods and not hasattr(mods[m], a)]
    if missing:
        restore()
        pytest.skip(f"stale stroke_orofacial reference at {SRC} (missing {', '.join(missing)}) -- "
                    "update that checkout to its main to run the parity tests")
    yield mods
    restore()


#: Symbols the parity tests call that only recent stroke_orofacial versions have.
_REQUIRED_SYMBOLS = {"tongue_pertrial": ["_v7_cleaned_bundle_arrays"]}


def their_params(p):
    """Our DEFAULTS-shaped dict -> their gates/preclean blocks (drop the one key we added)."""
    g = {k: v for k, v in p["gates"].items() if k != "e_min_robust_std_px"}
    return p["preclean"], p["detector"], p["detector_v2"], g


def masks(fm):
    return td.interp_masks(fm)


# =========================================================================== behaviour: pre-clean

def _cl(y, intp=None, base=None):
    n = len(y)
    intp = np.zeros(n, bool) if intp is None else intp
    base = np.zeros(n, bool) if base is None else base
    return td._classify_raw_clusters(np.asarray(y, float), intp, base, n, FPS, params=P()["preclean"])


def test_cluster_labels():
    assert _cl([20, 40, 60, 80, 100, 80, 60, 40, 20])[0].label == "keep_low"
    assert "A=1" in _cl([20, 60, 100, 140, 100, 60, 20])[0].label            # dips low -> real lick shape
    # a narrow high plateau surrounded by baseline: no shape, no attached low neighbour -> artifact
    n = 40
    y = np.zeros(n)
    base = np.ones(n, bool)
    y[15:20], base[15:20] = 140.0, False
    c = [c for c in _cl(y, base=base) if c.y_max > 0][0]
    assert c.label.startswith("artifact_narrow")
    # same but only two frames -> the isolated-short override names it
    y2, b2 = np.zeros(n), np.ones(n, bool)
    y2[15:17], b2[15:17] = 200.0, False
    assert [c for c in _cl(y2, base=b2) if c.y_max > 0][0].label.startswith("artifact_isolated_short")
    # a long frozen plateau at any height above flat_min_y
    y3 = np.full(30, 50.0) + np.linspace(0, 1, 30)
    assert _cl(y3)[0].label.startswith("artifact_flat")


def test_preclean_wipes_isolated_blip_and_its_pchip_neighbours():
    n = 60
    y, x = np.zeros(n), np.zeros(n)
    intp, base = np.zeros(n, bool), np.ones(n, bool)
    y[28:30], base[28:30] = 200.0, False                     # 2-frame isolated blip
    y[26:28] = [120.0, 180.0]                                 # PCHIP fill anchored on it
    intp[26:28], base[26:28] = True, False
    yc, xc, bc, d = td.preclean_trace(y, x, intp, base, n, FPS, params=P()["preclean"])
    assert np.all(yc[26:30] == 0) and np.all(bc[26:30])
    assert d.n_artifact_clusters == 1 and d.n_pchip_extended == 2
    assert np.all(xc[bc] == 0.0)                              # Rule 1: x = x0 (= 0) at baseline frames


def test_preclean_wipes_single_frame_outlier_inside_a_lick():
    # A 100 ms, 150 px lick. (On a much narrower/steeper bump the rule ALSO takes the lick's edge frames, whose
    # +-20 ms in-cluster median sits > 100 px above them -- theirs does the same; a px threshold to retune.)
    n = 90
    y, x = np.zeros(n), np.zeros(n)
    intp, base = np.zeros(n, bool), np.ones(n, bool)
    u = (np.arange(25) + 0.5) / 25
    y[30:55], base[30:55] = 150 * np.sin(np.pi * u), False
    y[42] += 160.0
    yc, _, bc, d = td.preclean_trace(y, x, intp, base, n, FPS, params=P()["preclean"])
    assert d.n_frame_outliers == 1 and yc[42] == 0.0 and bc[42]
    assert np.all(yc[30:42] == y[30:42]) and np.all(yc[43:55] == y[43:55])   # the rest of the lick is kept


def test_bracketed_interp_run_is_wiped():
    n = 10
    yc = np.arange(n, dtype=float) + 1
    intp = np.zeros(n, bool)
    intp[3:6] = True
    base = np.zeros(n, bool)
    base[2] = base[6] = True
    assert td._wipe_bracketed_interp(yc, intp, base, n) == 3
    assert np.all(yc[3:6] == 0) and np.all(base[2:7])


def test_x_rules_hard_bound_and_snap_back():
    n = 60
    y, x = np.zeros(n), np.zeros(n)
    intp, base = np.zeros(n, bool), np.ones(n, bool)
    base[10:50] = False
    y[10:50] = 60.0 + np.arange(40)                           # a long low raw stretch (keep_low)
    x[10:50] = 40.0 + 0.1 * np.arange(40)
    x[20] = 150.0                                             # Rule 3: |x| > 110
    x[30] = 2.0                                               # Rule 4: snaps toward x0 = 0 between two 40s
    yc, xc, bc, d = td.preclean_trace(y, x, intp, base, n, FPS, params=P()["preclean"])
    assert d.n_x_hard_outliers == 1 and d.n_x_joint_outliers == 1
    assert abs(xc[20] - 41.0) < 0.05 and abs(xc[30] - 42.0) < 0.05    # re-PCHIP'd from the neighbours
    assert yc[30] == 0.0 and bc[30] and not bc[20]            # joint outlier wipes y too; hard one does not


# =========================================================================== behaviour: detectors

def _train(n_licks=5, gap=25, dur=15, h=180.0):
    n = 200 + n_licks * (dur + gap)
    y = np.zeros(n)
    base = np.ones(n, bool)
    peaks = []
    t = 40
    for _ in range(n_licks):
        u = (np.arange(dur) + 0.5) / dur
        y[t:t + dur] = h * np.sin(np.pi * u)
        base[t:t + dur] = False
        peaks.append(t + dur // 2)
        t += dur + gap
    t_ms = (np.arange(n) - 20) / FPS * 1000.0                # frame 20 = cue; all licks inside [70, 3000]
    return y, np.zeros(n), base, t_ms, peaks


def test_three_detectors_find_each_lick_and_merge_dedupes():
    y, x, base, t_ms, peaks = _train()
    p = P()
    win = (70.0, 3000.0)
    lm = td.detect_local_maxima(y, x, t_ms, FPS, is_baseline_fill=base, detect_win_ms=win,
                                detector_params=p["detector"], params=p["detector_v2"])
    lg = td.slope_detect_lick_peaks(y, x, t_ms, FPS, detect_win_ms=win, detector_params=p["detector"])
    i_lo = int(np.argmax(t_ms >= 70.0))
    assert np.array_equal(lm.peak_indices + i_lo, peaks)      # indices are detect-window-local
    assert np.array_equal(lg.peak_indices + i_lo, peaks)
    merged = td.merge_detector_outputs([("lmax", lm), ("bounded", td.PeakList.empty()),
                                        ("legacy", td.legacy_peaks_to_peak_list(lg, p["detector"], FPS))],
                                       np.ones(len(y)), FPS, params=p["detector_v2"])
    assert len(merged) == len(peaks) and {m.source for m in merged} == {"lmax"}
    assert all(m.rise_start_frame < m.frame <= m.fall_end_frame for m in merged)


def test_bounded_detector_accepts_retraction_without_fall_run():
    # a rise that ends in the tongue vanishing (next frame baseline) has no fall run: the legacy detector
    # rejects it, the bounded one accepts it via implicit baseline retraction.
    n = 120
    y = np.zeros(n)
    base = np.ones(n, bool)
    y[40:50], base[40:50] = np.linspace(20, 200, 10), False
    t_ms = (np.arange(n) - 20) / FPS * 1000.0
    p = P()
    bp = dict(p["detector"], max_fall_search_ms=52.0, accept_implicit_baseline_fall=True)
    p_no_smooth = dict(bp, do_smooth_y_for_detection=False)
    lg = td.slope_detect_lick_peaks(y, np.zeros(n), t_ms, FPS, detect_win_ms=(70, 3000),
                                    detector_params=dict(p["detector"], do_smooth_y_for_detection=False))
    bd = td.slope_detect_lick_peaks(y, np.zeros(n), t_ms, FPS, detect_win_ms=(70, 3000),
                                    detector_params=p_no_smooth, is_baseline_fill=base)
    assert len(lg) == 0 and len(bd) == 1


# =========================================================================== behaviour: gates

def test_gate_F_rejects_flat_plateau_and_gate_D_needs_both_sides():
    g = P()["gates"]
    n = 100
    intp = np.zeros(n, bool)
    y = np.full(n, 150.0)
    base = np.zeros(n, bool)
    rej, reason, _ = td.gate_F(y, intp, base, 50, n, FPS, params=g)
    assert rej and reason.startswith("F_raw_flat")
    y2, b2 = np.zeros(n), np.ones(n, bool)
    y2[40:100], b2[40:100] = 160.0, False                    # left baseline present, right never returns low
    keep, why = td.gate_D(y2, intp, b2, 60, n, FPS, params=g)
    assert not keep and why == "D_right"


def test_gate_E_flags_tall_low_confidence_and_abstains():
    g = P()["gates"]
    peers = [150.0, 160.0, 170.0, 180.0, 165.0]
    assert td.gate_E(400.0, 0.6, peers, params=g)[0]
    assert not td.gate_E(400.0, 0.95, peers, params=g)[0]    # confident outliers are real tall licks
    assert td.gate_E(400.0, 0.6, peers[:3], params=g)[1] == "abstain_few_peers"


def test_parabolic_excise_recovers_vertex_under_a_spike():
    g = P()["gates"]
    n = 80
    f = np.arange(n, dtype=float)
    y = 200.0 - 0.5 * (f - 40.3) ** 2
    y[40] = 500.0
    intp, base = np.zeros(n, bool), np.zeros(n, bool)
    r = td.parabolic_excise(y, np.zeros(n), intp, base, 40, n, FPS, params=g)
    assert r is not None and r["new_peak_frame"] == 40 and abs(r["new_peak_y"] - 200.0) < 1e-6


def test_m_gate_drops_lower_confidence_and_relaxes_early():
    g = P()["gates"]
    n = 400
    lik = np.full(n, 0.95)
    lik[100:110] = 0.4                                        # the peak at frame 105 is the weaker one
    intp, base = np.zeros(n, bool), np.zeros(n, bool)

    def dec(t, pf, yv):
        return {"t_ms": t, "peak_frame_local": pf, "y": yv, "keep": True, "gate": "", "reason": ""}
    late = [dec(1000.0, 90, 150.0), dec(1070.0, 105, 160.0)]   # 70 ms apart, late -> < 80 ms -> one goes
    td.run_m_gate(late, None, lik, intp, base, n, FPS, None, params=g)
    assert [d["keep"] for d in late] == [True, False] and late[1]["gate"] == "M"
    early = [dec(100.0, 90, 150.0), dec(170.0, 105, 160.0)]    # 70 ms apart inside 250 ms -> 60 ms rule
    td.run_m_gate(early, None, lik, intp, base, n, FPS, None, params=g)
    assert all(d["keep"] for d in early)


def test_interp_masks_and_params_merge():
    fm = np.array([0, 1, 2, 3])
    i, b = td.interp_masks(fm)
    assert i.tolist() == [False, True, True, False] and b.tolist() == [False, False, False, True]
    p = td.params({"preclean": {"high_thr_px": 99.0}})
    assert p["preclean"]["high_thr_px"] == 99.0 and p["preclean"]["shape_low_thr_px"] == 80.0
    assert td.DEFAULTS["preclean"]["high_thr_px"] == 130.0    # DEFAULTS untouched by the merge


# =========================================================================== parity vs source

def _eq(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("seed", range(6))
def test_parity_preclean_detect_gates(source, seed):
    """Every detection stage, on every trial slice of a randomised session, equals theirs exactly."""
    pre, det, det2, gat = their_params(P())
    sp, sd, sd2, sg, stp = (source[m] for m in ("_preclean", "_detector", "_detector_v2", "_gates",
                                                "tongue_pertrial"))
    y, x, fm, lk, cues, _ = synth_session(seed)
    intp, base = masks(fm)
    for c in cues:
        lo, hi = c + 17, c + 2000 + 1                         # = their slice for (70, 8000) ms, shortened
        sl = slice(lo, hi)
        t_ms = (np.arange(lo, hi) - c) / FPS * 1000.0
        n = hi - lo
        ours = td.preclean_trace(y[sl], x[sl], intp[sl], base[sl], n, FPS, params=P()["preclean"])
        theirs = sp.preclean_trace(y[sl], x[sl], intp[sl], base[sl], n, FPS, params=pre)
        for a, b in zip(ours[:3], theirs[:3]):
            _eq(a, b)
        assert [c_.label for c_ in ours[3].clusters] == [c_.label for c_ in theirs[3].clusters]
        for f in ("n_artifact_clusters", "n_artifact_frames", "n_interp_wiped", "n_frame_outliers",
                  "n_pchip_extended", "n_x_hard_outliers", "n_x_joint_outliers", "n_x_pchip_extended"):
            assert getattr(ours[3], f) == getattr(theirs[3], f), f
        yc, xc, bc, _ = ours
        win = (70.0, 3000.0)
        a = td.detect_local_maxima(yc, xc, t_ms, FPS, is_baseline_fill=bc, detect_win_ms=win,
                                   detector_params=det, params=det2)
        b = sd2.detect_local_maxima(yc, xc, t_ms, FPS, is_baseline_fill=bc, detect_win_ms=win,
                                    detector_params=det, params=det2)
        for f in ("peak_times_ms", "peak_y", "peak_x", "peak_indices", "rise_starts", "fall_ends"):
            _eq(getattr(a, f), getattr(b, f))
        for extra in ({}, {"max_fall_search_ms": 52.0, "accept_implicit_baseline_fall": True}):
            dp = dict(det, **extra)
            a = td.slope_detect_lick_peaks(yc, xc, t_ms, FPS, detect_win_ms=win, detector_params=dp,
                                           is_baseline_fill=bc if extra else None)
            b = sd._slope_detect_lick_peaks(yc, xc, t_ms, FPS, detect_win_ms=win, detector_params=dp,
                                            is_baseline_fill=bc if extra else None)
            for f in ("peak_times_ms", "peak_y", "peak_x", "rise_starts", "turnover", "peak_indices", "fall_ends"):
                _eq(getattr(a, f), getattr(b, f))
            pa = td.legacy_peaks_to_peak_list(a, det, FPS)
            pb = stp._legacy_peaks_to_peak_list(b, det, FPS)
            _eq(pa.fall_ends, pb.fall_ends)
        # gates at every raw frame that could be a peak, and the imputers at each
        for pf in np.flatnonzero((~bc) & (yc > 100))[::3]:
            assert td.gate_F(yc, intp[sl], bc, pf, n, FPS, params=P()["gates"])[:2] == \
                sg.gate_F(yc, intp[sl], bc, pf, n, FPS, params=gat)[:2]
            assert td.gate_D(yc, intp[sl], bc, pf, n, FPS, params=P()["gates"]) == \
                sg.gate_D(yc, intp[sl], bc, pf, n, FPS, params=gat)
            for fn in ("parabolic_excise", "spline_excise"):
                ra = getattr(td, fn)(yc, x[sl], intp[sl], bc, pf, n, FPS, params=P()["gates"])
                rb = getattr(sg, fn)(yc, x[sl], intp[sl], bc, pf, n, FPS, params=gat)
                assert (ra is None) == (rb is None)
                if ra is not None:
                    for k in ("new_peak_y", "new_peak_x", "new_peak_frame", "outlier_lo", "outlier_hi"):
                        _eq(ra[k], rb[k])
                    _eq(ra["interp_y"], rb["interp_y"])


# --------------------------------------------------------------------------- Rule 4, retuned (swap mode)

def _swap_params():
    return P(preclean={"x_jump_mode": "swap", "x_jump_thr_px": 20.0})["preclean"]


def test_swap_rule_reinterpolates_a_one_frame_tip_hop_in_either_direction():
    x = np.full(40, -30.0)
    x[10] = -1.0          # hops 29 px to the other side of the spout (PS93 0814 trial 30)
    x[25] = -60.0         # and one AWAY from the midline -- theirs (toward rest only) would miss it
    raw = np.ones(40, bool)
    f = td._find_x_swaps(x, raw, 40, params=_swap_params())
    assert list(np.flatnonzero(f)) == [10, 25]


def test_swap_rule_catches_short_bursts_but_not_real_lateral_motion():
    x = np.full(40, -30.0)
    x[10:12] = 0.0                                   # a 2-frame hop -> flagged
    x[20:40] = -30.0 + np.arange(20) * 4.0           # a steady 4 px/frame sweep (real motion) -> not flagged
    raw = np.ones(40, bool)
    f = td._find_x_swaps(x, raw, 40, params=_swap_params())
    assert list(np.flatnonzero(f)) == [10, 11]


def test_swap_mode_keeps_y_and_interpolates_x():
    n = 60
    x = np.full(n, -30.0)
    x[30] = 0.0
    is_interp = np.zeros(n, bool)
    is_base = np.zeros(n, bool)
    is_base[:5] = True
    out = td._clean_x_trace(x, is_interp, is_base, is_base.copy(), n, 250.0, params=_swap_params())
    x_clean, x_hard, x_joint, _ext, is_base_aug, _x0, x_swap = out
    assert x_swap[30] and not x_joint.any()
    assert abs(x_clean[30] + 30.0) < 1e-6           # re-interpolated from its neighbours
    assert not is_base_aug[30]                      # y NOT wiped (the distance from the mouth is real)


def test_default_mode_is_theirs_for_parity():
    assert td.DEFAULTS["preclean"]["x_jump_mode"] == "snapback"
