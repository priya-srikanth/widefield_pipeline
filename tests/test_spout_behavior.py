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


def test_curated_mmdds_includes_future_onward_dates():
    # available spans May/June training + the analysis window + freshly-added 8/8, 8/9
    avail = {"0505", "0529", "0601", "0605", "0606", "0607", "0608", "0611", "0805",
             "0806", "0807", "0808", "0809"}
    got = sb._curated_mmdds(avail)
    # keep 6/6-6/8 anchors + everything 8/6 onward (incl. new 8/8/8/9); drop training + 8/5
    assert got == ["0606", "0607", "0608", "0806", "0807", "0808", "0809"]
    # a date not yet recorded is simply absent (only 'available' dates appear)
    assert sb._curated_mmdds({"0606", "0807"}) == ["0606", "0807"]


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


def test_load_trials_auto_switches_to_log_when_daq_undercovers(monkeypatch, tmp_path):
    """A crashed recorder (DAQ << log) makes the BEHAVIOUR loader use the full-session log on the
    default 'auto' path (PS92 8/12); explicit --source daq must NOT switch (decode stays DAQ-only)."""
    d = tmp_path / "PS92_20260812_concat"
    d.mkdir()
    daq_df = pd.DataFrame({"pos_idx": [0] * 10, "responded": [True] * 10, "source": ["DAQ"] * 10})
    log_df = pd.DataFrame({"pos_idx": [0] * 20, "responded": [True] * 20, "source": ["log"] * 20})
    monkeypatch.setattr(sb, "_daq_trials_for", lambda sd, rv, params: daq_df)
    monkeypatch.setattr(sb, "load_gui_trials", lambda sd: log_df)
    params = config.defaults()["behavior"]
    auto = sb.load_trials(d, rv=object(), params=params, source="auto")
    assert len(auto) == 20 and auto["source"].iloc[0] == "log"        # crashed recorder -> full log
    daq = sb.load_trials(d, rv=object(), params=params, source="daq")
    assert len(daq) == 10 and daq["source"].iloc[0] == "DAQ"          # explicit daq: never switches


def test_load_trials_auto_keeps_daq_when_it_covers_the_log(monkeypatch, tmp_path):
    """Full DAQ coverage (the normal case) keeps DAQ primary — the switch is only for a crashed recorder."""
    d = tmp_path / "PS92_20260807_120000"
    d.mkdir()
    daq_df = pd.DataFrame({"pos_idx": [0] * 100, "responded": [True] * 100, "source": ["DAQ"] * 100})
    log_df = pd.DataFrame({"pos_idx": [0] * 101, "responded": [True] * 101, "source": ["log"] * 101})
    monkeypatch.setattr(sb, "_daq_trials_for", lambda sd, rv, params: daq_df)
    monkeypatch.setattr(sb, "load_gui_trials", lambda sd: log_df)
    out = sb.load_trials(d, rv=object(), params=config.defaults()["behavior"], source="auto")
    assert out["source"].iloc[0] == "DAQ"                            # 100/101 is covered -> DAQ stays


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

def test_the_gate_fires_on_a_reference_position_collapse(tmp_path):
    """Disengagement is judged at the REFERENCE positions (close_L, close_center) only.

    REWRITTEN 2026-08-28 with the gate. It used to build a terminal run of misses at far_L and
    assert they were excluded -- which is exactly what must NOT happen now: after a lesion a run of
    far misses is the deficit, and deleting it writes the effect off as the confound. The
    companion test below pins that direction.
    """
    trials = [dict(tid=i + 1, pos_idx=1, hit=True) for i in range(20)]
    trials += [dict(tid=21 + i, pos_idx=1, hit=False) for i in range(16)]   # close_L collapse
    d = _write_session(tmp_path, "PS92_20260806_120000", trials)
    m = sb.session_metrics(sb.load_trials(d), None, config.defaults()["behavior"])
    assert m["n_disengaged"] > 0, "a sustained collapse AT REFERENCE must be caught"
    assert m["hit_rate_all"] < m["hit_rate_engaged"]


def test_a_terminal_run_of_far_misses_is_NOT_disengagement(tmp_path):
    """The whole point of the change: a position-specific failure is the result, not the confound.

    `flag_engagement` judged the trailing rate over ALL positions, so post-stroke it called the
    animal disengaged precisely because it could not reach the far spouts -- and those trials were
    then dropped from the hit rate that was supposed to measure the deficit. Measured on the real
    cohort: it excluded 380 of PS94_0817's 643 trials where the reference-judged gate excludes 0.
    """
    trials = [dict(tid=i + 1, pos_idx=1, hit=True) for i in range(12)]
    trials += [dict(tid=13 + i, pos_idx=4, hit=False) for i in range(12)]   # far_L, all missed
    d = _write_session(tmp_path, "PS92_20260806_120000", trials)
    m = sb.session_metrics(sb.load_trials(d), None, config.defaults()["behavior"])
    assert m["n_disengaged"] == 0, "far-position misses were written off as disengagement"
    per = m["per_position"].set_index("pos_name")
    assert per.loc["far_L", "trials_engaged"] == 12          # they COUNT, and they are misses
    assert per.loc["far_L", "hit_rate"] == pytest.approx(0.0)


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


def test_daq_h5_for_prefers_concat(tmp_path):
    d = tmp_path / "20260812"
    d.mkdir()
    for n in ("PS92_20260812_152628.h5", "PS92_20260812_161746.h5", "PS92_20260812_concat.h5"):
        (d / n).write_bytes(b"x")
    rv = _StubRV(tmp_path)
    # a rejoined crash day -> the _concat session wins over the raw segments (which sort first)
    assert sb._daq_h5_for(rv, "PS92", "20260812").name == "PS92_20260812_concat.h5"
    (d / "PS92_20260812_concat.h5").unlink()                        # no concat -> first sorted
    assert sb._daq_h5_for(rv, "PS92", "20260812").name == "PS92_20260812_152628.h5"


def test_discover_sessions_collapses_crash_to_concat(tmp_path):
    for n in ("PS92_20260812_152647", "PS92_20260812_161800", "PS92_20260812_concat",
              "PS93_20260812_100000"):
        _write_session(tmp_path, n, [dict(tid=1, pos_idx=1, hit=True)])
    rv = _StubRV(tmp_path)
    # PS92's two raw crash dirs are dropped in favor of the concat; PS93 (no concat) untouched
    assert {p.name for p in sb.discover_sessions(rv, "20260812")} == {
        "PS92_20260812_concat", "PS93_20260812_100000"}
    assert [p.name for p in sb.discover_sessions(rv, "20260812", {"PS92"})] == ["PS92_20260812_concat"]


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


def test_load_licks_gui_fallback_and_latency(tmp_path):
    ev = _events({1: {"cue_ms": 1000, "licks_ms": [1300, 1450], "resets": 3},
                  2: {"cue_ms": 2000, "licks_ms": [2100], "resets": 0}})
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=1, pos_idx=1, hit=True), dict(tid=2, pos_idx=2, hit=True)], ev)
    gui = sb.load_gui_licks(d)
    assert gui["gui_lick_s"].size == 3 and gui["sync_s"].size == 0
    assert gui["cue_by_trial"].loc[1] == pytest.approx(1.0)
    assert int(gui["precue_reset_by_trial"].loc[1]) == 3
    licks = sb.load_licks(d, rv=None)      # no rv -> GUI fallback (no DAQ events)
    assert licks["source"] == "GUI" and licks["all_s"].size == 3
    assert licks["cue_next_by_trial"].loc[1] == pytest.approx(2.0)   # next cue
    lat = sb.first_lick_latency_s(d, sb.load_trials(d), 5.0, licks=licks)
    assert lat.iloc[0] == pytest.approx(0.3)


def test_load_licks_daq_primary_via_sync(tmp_path):
    from wfield_local import behavior_events as be
    # events.csv: 25 sync pulses (device ms), cues, and a GUI lick that DIFFERS from the DAQ lick
    sync_ms = list(range(500, 500 + 400 * 25, 400))
    rows = [{"device_t_ms": t, "event_name": "sync", "trial_id": -1} for t in sync_ms]
    rows += [{"device_t_ms": 1000, "event_name": "cue", "trial_id": 1},
             {"device_t_ms": 5000, "event_name": "cue", "trial_id": 2},
             {"device_t_ms": 1310, "event_name": "lick_on", "trial_id": 1}]   # GUI lick @1.31 s
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=1, pos_idx=1, hit=True), dict(tid=2, pos_idx=1, hit=True)], rows)

    class _RV:
        def root(self, name):
            return str(tmp_path / "server")
    rv = _RV()
    # canonical DAQ events (5000 Hz): sync samples == device_ms*5 (identity map), DAQ lick @1.30 s
    be.save_events({"schema_version": 2, "fs": 5000.0, "n_samples": 30000,
                    "lick_onsets": np.array([6500], np.int64),          # 1.30 s
                    "sync_samples": (np.array(sync_ms) * 5).astype(np.int64),
                    "daq_h5": "x"}, be.events_path(rv, "PS92", "20260806"))
    licks = sb.load_licks(d, rv=rv)
    assert licks["source"] == "DAQ"                     # DAQ licks used, mapped onto the device clock
    assert licks["all_s"].tolist() == pytest.approx([1.30])   # the DAQ lick, NOT the GUI 1.31


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


def test_lick_microstructure_per_position_gated_to_engaged(tmp_path):
    """The per-position lick aggregates use engaged trials only (same gate as per-position hit rate).

    A sated terminal tail licks nothing; ungated it drags close_L's licks/trial down, gated it doesn't.
    Session-level scalars stay over the whole session either way.
    """
    params = config.defaults()["behavior"]
    # LONG ENOUGH TO COLLAPSE THE REFERENCE RATE. The gate no longer has a fixed "terminal run of
    # N misses" rule -- it fires when the trailing reference-position response rate stays below
    # MIN_RATE and does not recover, which takes more than `tail_min_misses` trials.
    tail = 16
    n_good = 40
    ev, trials = {}, []
    for i in range(n_good):                                  # engaged: 4 licks/trial at close_L
        tid, cue = i + 1, 10000 * (i + 1)
        ev[tid] = {"cue_ms": cue, "licks_ms": [cue + 100 * k for k in range(4)], "resets": 2}
        trials.append(dict(tid=tid, pos_idx=1, hit=True))
    for j in range(tail):                                    # sated tail: no licks, no resets
        tid = n_good + j + 1
        ev[tid] = {"cue_ms": 10000 * tid, "licks_ms": [], "resets": 0}
        trials.append(dict(tid=tid, pos_idx=1, hit=False))
    d = _write_session(tmp_path, "PS92_20260806_120000", trials, _events(ev))
    tr = sb.load_trials(d)
    m = sb.session_metrics(tr, None, params)
    # PART of the tail, not all of it, and that is a property of a rate-collapse gate rather than
    # a defect: it fires where the trailing reference rate crosses MIN_RATE, which a run of misses
    # takes ~8 reference trials to do, then extends to the end. The old gate marked from the first
    # miss because it carried a separate "terminal run of N" rule; the reference-judged gate has
    # none, which is the trade for not calling a far-position deficit disengagement.
    assert 0 < m["n_disengaged"] < tail
    assert not m["scored"]["engaged"].to_numpy()[-1], "the end of a collapse must be excluded"

    ungated = sb.lick_microstructure(d, tr, params)
    gated = sb.lick_microstructure(d, tr, params, engaged_ids=sb._engaged_ids(m))
    u = ungated["per_position"].set_index("pos_name")
    g = gated["per_position"].set_index("pos_name")
    # DERIVED FROM THE GATE, not hardcoded to "the whole tail". The gate excludes the part of the
    # collapse after the reference rate crosses MIN_RATE, so some zero-lick tail trials remain
    # engaged and the gated mean lands between the ungated one and the clean 4.0. Asserting 4.0
    # would be asserting the old gate's semantics with the new gate's name on it.
    kept = n_good + tail - m["n_disengaged"]
    assert u.loc["close_L", "licks_per_trial"] == pytest.approx(4 * n_good / (n_good + tail))
    assert g.loc["close_L", "licks_per_trial"] == pytest.approx(4 * n_good / kept)
    assert (g.loc["close_L", "licks_per_trial"]
            > u.loc["close_L", "licks_per_trial"])          # gating moves it the right way
    assert g.loc["close_L", "trials_engaged"] == kept
    assert (gated["session"]["n_pos_gated"] == m["n_disengaged"]
            and gated["session"]["pos_engagement_gated"])
    assert not ungated["session"]["pos_engagement_gated"]
    # session-level + raster span the WHOLE session in both cases (they describe the recording)
    assert ungated["session"]["n_licks"] == gated["session"]["n_licks"] == 4 * n_good
    assert len(gated["raster"]) == n_good + tail


def test_engaged_ids_matches_session_metrics(tmp_path):
    """`_engaged_ids` and `session_metrics` must agree on which trials are engaged.

    The tail is built at a REFERENCE position and long enough to collapse the reference rate,
    because that is what the gate now judges on.
    """
    params = config.defaults()["behavior"]
    trials = ([dict(tid=i + 1, pos_idx=1, hit=True) for i in range(20)]
              + [dict(tid=21 + j, pos_idx=1, hit=False) for j in range(16)])
    d = _write_session(tmp_path, "PS92_20260806_120000", trials)
    m = sb.session_metrics(sb.load_trials(d), None, params)
    ids = sb._engaged_ids(m)
    assert len(ids) == m["n_engaged"] < len(trials)
    assert ids == set(sorted(ids))                       # a set of trial_ids, not positions
    assert max(ids) < 37


def test_lick_pos_panels_match_across_session_metric_families():
    """The per-session by-position lick panels are the same three families the per-animal
    across-session figure tracks — so a session reads directly against the cross-day trend."""
    across = [col for _p, table, col, _lab in sb.POS_METRICS if table == "lick"]
    assert [col for col, _t, _y in sb.LICK_POS_PANELS] == across == [
        "licks_per_trial", "lick_rate_hz", "anticipatory_licks"]


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
                               sb.load_trials(d), sb.load_licks(d, rv=None), rv=None)
    assert png.exists() and csv.exists()
    assert (out / "sessions" / "PS92" / "20260806").is_dir()   # nested by animal/date


def _mini_adf(animal, dates):
    """Minimal per-animal across-session frame with every column the plotters read."""
    rows = []
    for i, d in enumerate(dates):
        r = {"animal": animal, "date": d, "hit_rate": 0.7, "close": 0.8, "far": 0.6,
             "n_engaged": 50, "n_disengaged": 5}
        for idx in sb.IDX_ORDER:
            nm = sb.POS_BY_IDX[idx]["name"]
            r[nm] = 0.7
            r[f"hitall__{nm}"] = 0.5
            for prefix, _t, _c, _l in sb.POS_METRICS:
                r[f"{prefix}__{nm}"] = 0.5 + 0.01 * i
        rows.append(r)
    return pd.DataFrame(rows)


def test_plot_animal_metric_series_writes_split_figures(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    # a lesion between the two dates so the boundary line (_mark_stroke) is exercised
    base = {a: dict(v) for a, v in config.animals().items()}
    base["PS94"]["stroke_date"] = "20260817"
    monkeypatch.setattr(config, "animals", lambda: base)

    written = sb.plot_animal_metric_series("PS94", _mini_adf("PS94", ["20260816", "20260818"]), tmp_path)
    assert {p.name for p in written} == {
        f"PS94_{s}_across_sessions.png"
        for s in ("hit", "latency", "licks_per_trial", "lick_rate", "anticipatory", "session")}
    for p in written:
        assert p.exists() and p.stat().st_size > 0
    # deck's expected stems and the producer's stems must agree
    from wfield_local import behavior_deck as bd
    assert {s for s, _t, _sub in bd.ACROSS_METRICS} == {p.name.split("PS94_")[1].rsplit(
        "_across_sessions", 1)[0] for p in written}


def test_plot_animal_raster_grid_tiles_sessions_on_a_shared_axis(tmp_path):
    """The tiled raster grid writes one page per <=8 sessions, all normalised to the animal's longest
    session, from DAQ-or-log trials (here the log fallback)."""
    pytest.importorskip("matplotlib")
    logs = tmp_path / "logs"
    logs.mkdir()
    for name, n in (("PS92_20260817_100000", 30), ("PS92_20260818_110000", 60)):   # different lengths
        _write_session(logs, name, [dict(tid=i + 1, pos_idx=i % 6, hit=(i % 2 == 0)) for i in range(n)])
    params = config.defaults()["behavior"]
    sessions = [logs / "PS92_20260817_100000", logs / "PS92_20260818_110000"]
    written = sb.plot_animal_raster_grid("PS92", sessions, tmp_path / "out", params, rv=None)
    assert [p.name for p in written] == ["PS92_raster_grid_p1.png"]     # 2 sessions -> one page
    assert written[0].exists() and written[0].stat().st_size > 0


def test_stroke_boundary_anchors_on_first_post_keeping_excluded_on_pre_side(monkeypatch):
    """PS92/PS93: pre, EXCLUDED (8/17), post (8/18). The lesion line sits before the first POST (8/18),
    NOT before the first non-pre (8/17) -- so the excluded 8/17 stays on the PRE side and reads as
    excluded, not post-stroke. Their effective lesion is 8/18 (re-lesioned after the 8/17 session)."""
    dates = ["20260814", "20260817", "20260818", "20260819"]
    phases = {"20260814": "pre", "20260817": "excluded", "20260818": "post", "20260819": "post"}
    monkeypatch.setattr(sb.config, "session_phase", lambda a, d: phases[d])
    bx, excl = sb._stroke_boundary("PS92", dates)
    assert bx == 1.5                        # between 8/17 (idx 1) and 8/18 (idx 2)
    assert excl == [1]                      # 8/17 is shaded...
    assert excl[0] < bx                     # ...and sits to the LEFT of the lesion line


def test_stroke_boundary_no_excluded_sits_before_first_post(monkeypatch):
    """PS94/PS95: pre then post (8/17) with no excluded session -> line just before 8/17."""
    dates = ["20260814", "20260817", "20260818"]
    phases = {"20260814": "pre", "20260817": "post", "20260818": "post"}
    monkeypatch.setattr(sb.config, "session_phase", lambda a, d: phases[d])
    bx, excl = sb._stroke_boundary("PS95", dates)
    assert bx == 0.5 and excl == []


def test_stroke_boundary_none_when_no_post_yet(monkeypatch):
    """An excluded session with no post session yet draws no line but is still reported as excluded."""
    dates = ["20260814", "20260817"]
    phases = {"20260814": "pre", "20260817": "excluded"}
    monkeypatch.setattr(sb.config, "session_phase", lambda a, d: phases[d])
    bx, excl = sb._stroke_boundary("PS92", dates)
    assert bx is None and excl == [1]


def test_daq_h5_for_missing(tmp_path):
    class _RV:
        def root(self, name):
            return str(tmp_path / "daq")
    assert sb._daq_h5_for(_RV(), "PS92", "20260806") is None   # no dir -> graceful None


def test_run_expands_a_date_range_into_each_session(tmp_path, monkeypatch):
    """A range/list must select sessions on every date in it — `discover_sessions` matches ONE
    literal date, so an unexpanded spec silently selected nothing (and the run looked successful)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in ("PS92_20260606_120000", "PS92_20260607_120000", "PS92_20260608_120000",
                 "PS92_20260805_120000"):
        _write_session(logs, name, [dict(tid=i + 1, pos_idx=i % 6, hit=True) for i in range(30)])
    rv = sb.PathResolver(machine="analysis")
    monkeypatch.setattr(rv, "root", lambda n: str(logs if n == "behavior_logs" else tmp_path / "out"))
    seen = []
    monkeypatch.setattr(sb, "plot_session",
                        lambda s, *a, **k: (seen.append(s.name), (None, None))[1])
    sb.run("0606-0608", rv, dry=True)
    assert sorted(seen) == ["PS92_20260606_120000", "PS92_20260607_120000", "PS92_20260608_120000"]
    seen.clear()
    sb.run("20260606,20260805", rv, dry=True)          # comma list, YYYYMMDD form
    assert sorted(seen) == ["PS92_20260606_120000", "PS92_20260805_120000"]


def test_cohort_and_deck_span_all_animals_even_with_only_subset(tmp_path, monkeypatch):
    """A per-animal night (`camera_nightly --only PS92 PS93`) must NOT shrink the cohort figure or the
    standing deck to that subset -- they are cohort-wide artifacts. Regression: the `--only` animals
    were forwarded into cohort_summary + the deck build, silently dropping PS94/PS95 from the deck."""
    rv = sb.PathResolver(machine="analysis")
    monkeypatch.setattr(rv, "root", lambda n: str(tmp_path / "out"))
    captured = {}
    monkeypatch.setattr(sb, "cohort_summary",
                        lambda rv_, dates, animals, out_dir, dry=False: captured.__setitem__("cohort", animals))
    import wfield_local.behavior_deck as bd
    monkeypatch.setattr(bd, "build_behavior_deck",
                        lambda root, out, animals=None: (captured.__setitem__("deck", animals),
                                                         {"out": str(out), "slides": 0,
                                                          "figures_present": 0, "figures_missing": 0})[1])
    sb.run(None, rv, animals=["PS92", "PS93"], cohort=True, from_spec="curated", dry=False)
    assert captured["cohort"] is None, "cohort figure was scoped to the --only subset"
    assert captured["deck"] is None, "standing deck was scoped to the --only subset"


def test_run_warns_when_a_date_spec_matches_nothing(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_session(logs, "PS92_20260606_120000", [dict(tid=1, pos_idx=1, hit=True)])
    rv = sb.PathResolver(machine="analysis")
    monkeypatch.setattr(rv, "root", lambda n: str(logs if n == "behavior_logs" else tmp_path / "out"))
    sb.run("20991231", rv, dry=True)
    assert "matched no sessions" in capsys.readouterr().out


def test_run_ignores_non_animal_dirs_when_resolving_a_range(tmp_path, monkeypatch):
    """The behavior-log root also holds non-animal dirs (`test_<date>_*`); they must not reach
    expand_dates, which rejects an 'unknown' date token and would abort the whole run."""
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_session(logs, "PS92_20260606_120000", [dict(tid=1, pos_idx=1, hit=True)])
    _write_session(logs, "test_20260602_173841", [dict(tid=1, pos_idx=1, hit=True)])
    rv = sb.PathResolver(machine="analysis")
    monkeypatch.setattr(rv, "root", lambda n: str(logs if n == "behavior_logs" else tmp_path / "out"))
    seen = []
    monkeypatch.setattr(sb, "plot_session",
                        lambda s, *a, **k: (seen.append(s.name), (None, None))[1])
    sb.run("0606-0608", rv, dry=True)                 # must not raise
    assert seen == ["PS92_20260606_120000"]


# ------------------------------------------------------------------- cumulative task raster

def _daq_trials(specs, *, t0=1234.5, iti_s=20.0):
    """A synthetic DAQ-shaped trial table (the columns `daq_trials.build_trials` emits).

    ``specs`` = list of dicts with pos_idx / responded / rewarded / free. Cues start at ``t0`` on the
    DAQ clock (i.e. NOT at zero — the recorder starts before the task) so the time axis is tested
    against the session's first trial, not the clock's origin.
    """
    rows = []
    for k, sp in enumerate(specs):
        resp = bool(sp.get("responded", True))
        rows.append({
            "trial_id": k + 1, "pos_idx": sp.get("pos_idx", 0),
            "pos_name": NAME.get(sp.get("pos_idx", 0), "?"),
            "cue_s": t0 + iti_s * k, "trial_start_s": t0 + iti_s * k - 3.0,
            "lick_in_response_window": int(resp), "hit": int(resp), "miss": int(not resp),
            "latency_s": 0.4 if resp else np.nan, "n_licks_post": 3 if resp else 0, "n_licks_pre": 0,
            "reward_delivered": int(sp.get("rewarded", resp)), "responded": resp,
            "is_free": bool(sp.get("free", False)), "free_designated": bool(sp.get("free", False)),
            "source": "DAQ",
        })
    return pd.DataFrame(rows)


def _by_label(ax):
    """Scatter collections of a raster axis, keyed by their legend label without the count."""
    return {c.get_label().split(" (")[0]: c for c in ax.collections}


def test_raster_rows_are_the_gui_order_and_labels():
    """Row order mirrors the rig GUI's `position_labels` (pos_idx 0..5), not the L/center/R bar
    order — the figure exists to be read beside the GUI display."""
    assert [sb.POS_BY_IDX[i]["name"] for i in sb.RASTER_ROW_ORDER] == [
        "close_center", "close_L", "close_R", "far_center", "far_L", "far_R"]
    assert sorted(sb.RASTER_ROW_ORDER) == sorted(sb.IDX_ORDER)      # same six positions, other order


def test_raster_outcome_follows_the_lick_not_the_reward():
    """GREEN = the animal LICKED, not "water arrived".

    Under `reward_mode: auto_after_delay` water is delivered on most trials whatever the animal does
    (withheld only by `auto_hold_after_miss`), so a rewarded non-response is still a MISS. PS92 8/21
    is the scale of the gap: 397 rewards against 310 hits.
    """
    mk = sb.raster_markers(_daq_trials([
        {"pos_idx": 0, "responded": True, "rewarded": True},     # hit, auto-rewarded -> earned
        {"pos_idx": 3, "responded": False, "rewarded": True},    # no lick but watered  -> MISS
        {"pos_idx": 5, "responded": False, "rewarded": False},   # reward held after misses -> miss
    ]))
    assert mk["outcome"].tolist() == ["earned", "miss", "miss"]


def test_reward_provenance_does_not_reach_the_raster():
    """The teal free/auto/manual ring was built and then dropped (Priya, 2026-08-22).

    Kept as a test because the columns it read (`is_free`, `reward_delivered`) are still on the
    trial table, so a future change could quietly reintroduce reward into a figure that is about
    the animal's behaviour. A free-rewarded hit and a plain hit must be indistinguishable here.
    """
    mk = sb.raster_markers(_daq_trials([
        {"pos_idx": 1, "responded": True, "rewarded": True, "free": True},
        {"pos_idx": 2, "responded": True, "rewarded": True},
    ]))
    assert mk["outcome"].tolist() == ["earned", "earned"]
    assert "unearned" not in mk.columns and "is_free" not in mk.columns


def test_raster_times_are_minutes_from_the_first_trial():
    mk = sb.raster_markers(_daq_trials([{"pos_idx": i % 6} for i in range(4)], t0=900.0, iti_s=60.0))
    # t=0 is the first trial (its start, 3 s before its cue), not the DAQ clock's zero
    assert mk["t_min"].iloc[0] == pytest.approx(3.0 / 60.0)
    assert mk["t_min"].iloc[-1] == pytest.approx(3.0 / 60.0 + 3.0)


def test_raster_keeps_the_disengaged_tail():
    """Explicitly NOT engagement-gated: the run of red at the end is the point of the figure."""
    specs = ([{"pos_idx": 1, "responded": True} for _ in range(20)]
             + [{"pos_idx": 1, "responded": False, "rewarded": False} for _ in range(16)])
    trials = _daq_trials(specs)
    m = sb.session_metrics(trials, None, config.defaults()["behavior"])
    assert m["n_engaged"] < len(trials)                     # the gate drops the collapse ...
    mk = sb.raster_markers(trials)
    assert len(mk) == len(trials)                           # ... the raster keeps every trial
    assert (mk["outcome"].to_numpy()[-16:] == "miss").all()


def test_raster_drops_unpaired_positions():
    """A cue with no preceding strobe decodes as pos_idx -1: it has no row, so it is dropped rather
    than drawn on an invented one."""
    mk = sb.raster_markers(_daq_trials([{"pos_idx": -1}, {"pos_idx": 2}]))
    assert mk["pos_idx"].tolist() == [2]


def test_raster_markers_none_without_a_clock():
    assert sb.raster_markers(pd.DataFrame()) is None
    no_clock = _daq_trials([{"pos_idx": 0}]).drop(columns=["cue_s", "trial_start_s"])
    assert sb.raster_markers(no_clock) is None


def test_cumulative_raster_colours_match_the_gui_legend():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    mk = sb.raster_markers(_daq_trials([
        {"pos_idx": 0, "responded": True}, {"pos_idx": 4, "responded": False, "rewarded": True}]))
    fig, ax = plt.subplots()
    sb._cumulative_raster(ax, mk)
    cols = _by_label(ax)
    assert mcolors.to_hex(cols["hit / earned reward"].get_facecolor()[0]) == sb.RASTER_COLORS["earned"]
    assert mcolors.to_hex(cols["miss"].get_facecolor()[0]) == sb.RASTER_COLORS["miss"]
    assert set(cols) == {"hit / earned reward", "miss"}, "green and red only -- no ring"
    assert ax.get_ylim() == (len(sb.RASTER_ROW_ORDER) - 0.5, -0.5)               # pos 0 on top
    plt.close(fig)


def test_cumulative_raster_renders_without_any_free_reward(tmp_path):
    """The common case -- no free water all session -- must still render."""
    import matplotlib.pyplot as plt
    trials = _daq_trials([{"pos_idx": i % 6, "responded": (i % 4 != 0)} for i in range(40)])
    mk = sb.raster_markers(trials)
    fig, ax = plt.subplots()
    sb._cumulative_raster(ax, mk)
    assert "free / auto / manual reward" not in _by_label(ax)
    plt.close(fig)
    png = sb.plot_cumulative_raster("PS92_20260806_120000", tmp_path / "out", trials)
    assert png.exists() and png.stat().st_size > 0
    assert png.name == "PS92_20260806_120000_task_raster.png"
    assert png.parent == tmp_path / "out" / "sessions" / "PS92" / "20260806"


def test_plot_session_writes_the_task_raster(tmp_path):
    """The raster is written on the log-fallback path too (device clock), beside the other figures."""
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [{"tid": i + 1, "pos_idx": i % 6, "hit": (i % 5 != 0)} for i in range(60)])
    out = tmp_path / "out"
    sb.plot_session(d, out, config.defaults()["behavior"])
    raster = out / "sessions/PS92/20260806/PS92_20260806_120000_task_raster.png"
    assert raster.exists() and raster.stat().st_size > 0
