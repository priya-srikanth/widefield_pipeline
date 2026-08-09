"""Tests for the spout behavior-session figures (wfield_local.spout_behavior)."""
import numpy as np
import pandas as pd
import pytest

from wfield_local import config, spout_behavior as sb

TRIAL_COLS = [
    "trial_id", "device_trial_start_ms", "device_trial_end_ms", "trial_start_gui_iso",
    "trial_end_gui_iso", "pos_idx", "pos_name", "pos_dist_mm_before_trial", "pos_dist_mm_after_trial",
    "adaptive_advance_this_trial", "adaptive_decrease_this_trial", "free_reward_trial",
    "free_reward_delivered", "lick_in_response_window", "hit", "miss", "reward_delivered", "reward_type",
]
NAME = {0: "close_center", 1: "close_L", 2: "close_R", 3: "far_center", 4: "far_L", 5: "far_R"}


def _trial(tid, pos_idx, *, hit, free_trial=0, free_delivered=0):
    resp = 1 if hit else 0
    return {
        "trial_id": tid, "device_trial_start_ms": 1000 * tid, "device_trial_end_ms": 1000 * tid + 500,
        "trial_start_gui_iso": "", "trial_end_gui_iso": "", "pos_idx": pos_idx,
        "pos_name": NAME[pos_idx], "pos_dist_mm_before_trial": 2.0, "pos_dist_mm_after_trial": 2.0,
        "adaptive_advance_this_trial": 0, "adaptive_decrease_this_trial": 0,
        "free_reward_trial": free_trial, "free_reward_delivered": free_delivered,
        "lick_in_response_window": resp, "hit": int(hit), "miss": int(not hit),
        "reward_delivered": 1 if (hit or free_delivered) else 0, "reward_type": "auto",
    }


def _write_session(tmp_path, name, trials, events=None):
    d = tmp_path / name
    d.mkdir()
    rows = [_trial(**t) for t in trials]
    # prepend the phantom setup row (trial_id 0, neither hit nor miss)
    phantom = _trial(0, trials[0]["pos_idx"], hit=False)
    phantom["hit"] = phantom["miss"] = phantom["lick_in_response_window"] = phantom["reward_delivered"] = 0
    pd.DataFrame([phantom] + rows, columns=TRIAL_COLS).to_csv(d / "trials.csv", index=False)
    if events is not None:
        pd.DataFrame(events).to_csv(d / "events.csv", index=False)
    return d


# --------------------------------------------------------------------------- load_trials

def test_load_trials_drops_phantom_and_keeps_scored(tmp_path):
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=1, pos_idx=1, hit=True), dict(tid=2, pos_idx=4, hit=False)])
    df = sb.load_trials(d)
    assert len(df) == 2                       # phantom trial_id 0 dropped
    assert list(df["trial_id"]) == [1, 2]
    assert df["responded"].tolist() == [True, False]


def test_is_free_uses_delivery_not_designation(tmp_path):
    # designated free but NOT delivered -> still a scored accuracy trial (kept, not is_free)
    d = _write_session(tmp_path, "PS92_20260806_120000", [
        dict(tid=1, pos_idx=1, hit=True),
        dict(tid=2, pos_idx=4, hit=False, free_trial=1, free_delivered=0),
        dict(tid=3, pos_idx=2, hit=True, free_trial=1, free_delivered=1),
    ])
    df = sb.load_trials(d)
    assert df["is_free"].tolist() == [False, False, True]        # only the delivered one
    assert df["free_designated"].tolist() == [False, True, True]


# --------------------------------------------------------------------------- engagement gate

def test_engagement_all_responded():
    engaged, info = sb.flag_engagement([True] * 30, window=15, min_rate=0.5, tail_min_misses=6)
    assert engaged.all()
    assert info["n_disengaged"] == 0 and info["tail_start"] is None


def test_engagement_terminal_sated_tail():
    seq = [True] * 20 + [False] * 8
    engaged, info = sb.flag_engagement(seq, window=15, min_rate=0.5, tail_min_misses=6)
    assert info["tail_start"] == 20
    assert not engaged[20:].any()               # the trailing sated run is excluded
    assert engaged[:20].all()


def test_engagement_short_miss_run_not_a_tail():
    # a trailing run shorter than tail_min_misses is NOT called a satiation tail
    seq = [True] * 20 + [False] * 3
    engaged, info = sb.flag_engagement(seq, window=15, min_rate=0.5, tail_min_misses=6)
    assert info["tail_start"] is None


def test_engagement_midsession_collapse_flagged():
    seq = [True] * 10 + [False] * 10 + [True] * 10
    engaged, info = sb.flag_engagement(seq, window=8, min_rate=0.5, tail_min_misses=6)
    assert info["tail_start"] is None           # not terminal (ends responding)
    assert info["n_disengaged"] > 0             # the collapse is still flagged
    assert engaged[0] and engaged[-1]


def test_engagement_never_responded():
    engaged, info = sb.flag_engagement([False] * 12, window=15, min_rate=0.5, tail_min_misses=6)
    assert not engaged.any() and info["tail_start"] == 0


# --------------------------------------------------------------------------- metrics

def test_session_metrics_excludes_disengaged_from_accuracy(tmp_path):
    # 12 close_L hits, then a 8-trial sated tail of misses at far_L. Engaged hit rate = 1.0;
    # raw includes the tail misses.
    trials = [dict(tid=i + 1, pos_idx=1, hit=True) for i in range(12)]
    trials += [dict(tid=13 + i, pos_idx=4, hit=False) for i in range(8)]
    d = _write_session(tmp_path, "PS92_20260806_120000", trials)
    params = config.defaults()["behavior"]
    m = sb.session_metrics(sb.load_trials(d), None, params)
    assert m["n_disengaged"] == 8
    assert m["hit_rate_engaged"] == pytest.approx(1.0)
    assert m["hit_rate_all"] < 1.0
    per = m["per_position"].set_index("pos_name")
    assert per.loc["close_L", "hit_rate"] == pytest.approx(1.0)
    # far_L misses were all in the sated tail -> 0 engaged trials there
    assert per.loc["far_L", "trials_engaged"] == 0


def test_per_position_covers_all_six(tmp_path):
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=i + 1, pos_idx=i % 6, hit=True) for i in range(30)])
    m = sb.session_metrics(sb.load_trials(d), None, config.defaults()["behavior"])
    assert sorted(m["per_position"]["pos_idx"]) == [0, 1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- latency

def test_first_lick_latency_from_events(tmp_path):
    events = [
        {"device_t_ms": 1000, "event_name": "cue", "trial_id": 1},
        {"device_t_ms": 1300, "event_name": "lick_on", "trial_id": 1},   # 0.3 s
        {"device_t_ms": 1500, "event_name": "lick_on", "trial_id": 1},
        {"device_t_ms": 2000, "event_name": "cue", "trial_id": 2},
        {"device_t_ms": 2050, "event_name": "lick_on", "trial_id": 2},   # 0.05 s
    ]
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=1, pos_idx=1, hit=True), dict(tid=2, pos_idx=2, hit=True)], events)
    tr = sb.load_trials(d)
    lat = sb.first_lick_latency_s(d, tr, max_s=5.0)
    assert lat.iloc[0] == pytest.approx(0.3)
    assert lat.iloc[1] == pytest.approx(0.05)


def test_latency_nan_without_events(tmp_path):
    d = _write_session(tmp_path, "PS92_20260806_120000", [dict(tid=1, pos_idx=1, hit=True)])
    lat = sb.first_lick_latency_s(d, sb.load_trials(d), max_s=5.0)
    assert lat.isna().all()


def test_latency_capped(tmp_path):
    events = [
        {"device_t_ms": 1000, "event_name": "cue", "trial_id": 1},
        {"device_t_ms": 9000, "event_name": "lick_on", "trial_id": 1},   # 8 s > cap
    ]
    d = _write_session(tmp_path, "PS92_20260806_120000", [dict(tid=1, pos_idx=1, hit=True)], events)
    lat = sb.first_lick_latency_s(d, sb.load_trials(d), max_s=5.0)
    assert lat.isna().all()


# --------------------------------------------------------------------------- discovery + figures

class _StubRV:
    def __init__(self, root):
        self._root = str(root)

    def root(self, name):
        return self._root


def test_discover_sessions_filters(tmp_path):
    for n in ("PS92_20260806_120000", "PS93_20260806_130000", "PS92_20260807_120000"):
        _write_session(tmp_path, n, [dict(tid=1, pos_idx=1, hit=True)])
    (tmp_path / "not_a_session").mkdir()        # no trials.csv -> ignored
    rv = _StubRV(tmp_path)
    assert len(sb.discover_sessions(rv, "20260806")) == 2
    got = sb.discover_sessions(rv, "20260806", animals={"PS92"})
    assert [p.name for p in got] == ["PS92_20260806_120000"]
    assert len(sb.discover_sessions(rv, None)) == 3


def test_plot_session_writes_and_dry(tmp_path):
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=i + 1, pos_idx=i % 6, hit=(i % 5 != 0)) for i in range(60)])
    out = tmp_path / "out"
    params = config.defaults()["behavior"]
    png, csv = sb.plot_session(d, out, params, dry=True)
    assert not png.exists() and not csv.exists()          # dry-run writes nothing
    png, csv = sb.plot_session(d, out, params, dry=False)
    assert png.exists() and csv.exists()
    cols = pd.read_csv(csv)
    assert len(cols) == 6 and "hit_rate" in cols.columns


def test_wilson_ci_bounds():
    lo, hi = sb._wilson(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = sb._wilson(10, 10)
    assert 0 <= lo <= 1 and 0 <= hi <= 1 and lo < hi


# --------------------------------------------------------------------------- lick microstructure

def _events(per_trial):
    """Build events rows. per_trial: {tid: {"cue_ms":, "licks_ms":[...], "resets": n}}."""
    rows = []
    for tid, d in per_trial.items():
        rows.append({"device_t_ms": d["cue_ms"], "event_name": "cue", "trial_id": tid})
        for lm in d.get("licks_ms", []):
            rows.append({"device_t_ms": lm, "event_name": "lick_on", "trial_id": tid})
        for _ in range(d.get("resets", 0)):
            rows.append({"device_t_ms": d["cue_ms"] - 500, "event_name": "pre_cue_reset_by_lick",
                         "trial_id": tid})
    return rows


def test_segment_bouts():
    # licks at 0,0.1,0.2 (one bout of 3), gap, 1.0,1.15 (bout of 2), gap, 3.0 (singleton dropped)
    onsets = np.array([0.0, 0.1, 0.2, 1.0, 1.15, 3.0])
    bouts = sb.segment_bouts(onsets, max_ili_s=0.3, min_bout_licks=2)
    assert len(bouts) == 2
    assert bouts[0] == (0.0, 0.2, 3)
    assert bouts[1] == (1.0, 1.15, 2)
    assert sb.segment_bouts(np.array([]), 0.3, 2) == []


def test_load_gui_licks_and_latency(tmp_path):
    ev = _events({1: {"cue_ms": 1000, "licks_ms": [1300, 1450], "resets": 3},
                  2: {"cue_ms": 2000, "licks_ms": [2100], "resets": 0}})
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=1, pos_idx=1, hit=True), dict(tid=2, pos_idx=2, hit=True)], ev)
    gui = sb.load_gui_licks(d)
    assert gui["all_s"].size == 3
    assert gui["cue_by_trial"].loc[1] == pytest.approx(1.0)
    assert int(gui["precue_reset_by_trial"].loc[1]) == 3
    lat = sb.first_lick_latency_s(d, sb.load_trials(d), 5.0, gui=gui)
    assert lat.iloc[0] == pytest.approx(0.3)


def test_lick_microstructure_metrics(tmp_path):
    # trial 1 (close_L): licks at cue+0,+0.1,+0.2,+0.3 -> 4 post licks, rate ~10 Hz; 2 anticipatory resets
    ev = _events({1: {"cue_ms": 1000, "licks_ms": [1000, 1100, 1200, 1300], "resets": 2},
                  2: {"cue_ms": 5000, "licks_ms": [5000, 5100], "resets": 0}})
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=1, pos_idx=1, hit=True), dict(tid=2, pos_idx=1, hit=True)], ev)
    m = sb.lick_microstructure(d, sb.load_trials(d), config.defaults()["behavior"])
    per = m["per_position"].set_index("pos_name")
    assert per.loc["close_L", "licks_per_trial"] == pytest.approx(3.0)   # (4 + 2) / 2 trials
    assert per.loc["close_L", "anticipatory_licks"] == pytest.approx(1.0)  # (2 + 0) / 2
    assert per.loc["close_L", "lick_rate_hz"] == pytest.approx(10.0)     # median ILI 0.1 s
    assert m["session"]["n_licks"] == 6
    assert len(m["raster"]) == 2


def test_lick_microstructure_none_without_events(tmp_path):
    d = _write_session(tmp_path, "PS92_20260806_120000", [dict(tid=1, pos_idx=1, hit=True)])
    assert sb.lick_microstructure(d, sb.load_trials(d), config.defaults()["behavior"]) is None


def test_min_session_trials_skips_aborted(tmp_path):
    d = _write_session(tmp_path, "PS93_20260806_203549", [dict(tid=1, pos_idx=2, hit=False)])
    png, csv = sb.plot_session(d, tmp_path / "out", config.defaults()["behavior"])
    assert png is None and csv is None                     # too few trials -> skipped, nothing written
    assert not (tmp_path / "out").exists()


def test_plot_licking_smoke(tmp_path):
    ev = _events({i + 1: {"cue_ms": 10000 * (i + 1), "licks_ms": [10000 * (i + 1) + 100 * k
                          for k in range(4)], "resets": 1} for i in range(30)})
    trials = [dict(tid=i + 1, pos_idx=i % 6, hit=True) for i in range(30)]
    d = _write_session(tmp_path, "PS92_20260806_120000", trials, ev)
    out = tmp_path / "out"
    png, csv = sb.plot_licking(d, d.name, out, config.defaults()["behavior"],
                               sb.load_trials(d), sb.load_gui_licks(d), rv=None)
    assert png.exists() and csv.exists()
    assert (out / "sessions" / "PS92" / "20260806").is_dir()   # nested by animal/date


def test_daq_h5_for_missing(tmp_path):
    class _RV:
        def root(self, name):
            return str(tmp_path / "daq")
    assert sb._daq_h5_for(_RV(), "PS92", "20260806") is None   # no dir -> graceful None
