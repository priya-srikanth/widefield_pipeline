"""wfield_local.face_rhythm_timing (true bin times + local frame loss from per-frame DAQ times) and its use in
scripts/face_rhythm_factors.py: a video that drops a stretch of frames gets correct bin times where the linear
frame -> DAQ fit is off, the lossy bins are found and NaN'd in projections, and the plan's balanced run / still draw."""
import json

import numpy as np

from wfield_local.face_rhythm import factors as f
from wfield_local import face_rhythm_timing as frt


def _fd_with_dropout(n_kept=20000, gap_at=12000, drop_every=2):
    """Per-frame DAQ times at 250 fps; from frame ``gap_at`` on, only every ``drop_every``-th frame was recorded."""
    t = np.arange(n_kept + (n_kept - gap_at) * (drop_every - 1)) / 250.0
    keep = np.r_[np.arange(gap_at), np.arange(gap_at, len(t), drop_every)][:n_kept]
    return t[keep] + 5.0


def test_true_bin_times_and_local_drop():
    fd = _fd_with_dropout()
    fc = np.arange(0, 20000, 50) + 24.5                              # 5 Hz bin centres (frames)
    t5 = frt.bin_times(fd, fc)
    lin = np.polyval(np.polyfit(np.arange(len(fd)), fd, 1), fc)
    assert np.abs(lin - t5).max() > 1.0                               # the linear fit is seconds off
    assert np.allclose(t5[:200], fc[:200] / 250.0 + 5.0, atol=1e-3)   # before the dropout: exact
    d = frt.local_drop(fd, t5)
    clean = t5 < 5.0 + 12000 / 250.0 - frt.LOSSY_HALF_S
    lossy = t5 > 5.0 + 12000 / 250.0 + frt.LOSSY_HALF_S
    assert (d[clean] < 0.01).all() and np.allclose(d[lossy], 0.5, atol=0.02)
    assert not frt.lossy(d[clean]).any() and frt.lossy(d[lossy]).all() and frt.lossy(np.array([np.nan])).all()


def test_draw_rows_balances_running_and_still():
    trials = [("k", i, "L") for i in range(20)]
    between = [("k", b, "run") for b in range(30)] + [("k", b, "still") for b in range(30, 500)] + \
        [("k", b, "other") for b in range(500, 600)]
    a = f.draw_rows(trials, between, 10, 0.4, np.random.default_rng(0), ds=5, run_share=0.5)
    nb = round(0.4 * 10 * (151 // 5))
    assert a["n_between"]["other"] == 0 and sum(a["n_between"].values()) == nb
    assert a["n_between"]["run"] == 30 and a["n_between"]["still"] == nb - 30     # run pool short: still fills


def test_project_uses_true_times_and_nans_lossy_bins(tmp_path):
    from tests.test_face_rhythm_factors import R, _bundle
    sd, _, _, plan = _bundle(tmp_path)
    f.cmd_build(sd)
    f.cmd_fit(sd, R)
    pl = json.loads((sd / "plan.json").read_text())
    pl["loso_jobs"] = []
    (sd / "plan.json").write_text(json.dumps(pl))
    f.cmd_select(sd, R)
    key = "PS93_20260820_cam4"
    i = next(j for j, t in enumerate(plan["tasks"], 1) if t["key"] == key)
    n5 = 300
    t5 = np.arange(n5) / 5.0 + 7.0                                     # the "true" times differ from the stamps
    drop5 = np.zeros(n5)
    drop5[250:] = 0.3                                                  # a lossy tail
    tr = np.load(tmp_path / "out" / f"{key}_trials.npz")
    g = tr["cue_daq_s"][:, None] + tr["t_ms"][None, :] / 1000.0
    dtr = np.zeros(g.shape)
    dtr[-1] = 0.3                                                      # the last trial in the lossy tail
    (tmp_path / "timing").mkdir()
    np.savez(tmp_path / "timing" / f"{key}.npz", t5=t5, t25=np.arange(5 * n5) / 25.0 + 7.0, drop5=drop5,
             drop25=np.zeros(5 * n5), drop_trials=dtr, dropped_frac=0.01, version=frt.TIMING_VERSION)
    f.cmd_project(sd, i)
    p = np.load(sd / "proj" / f"{key}.npz")
    assert str(p["timing"]) == frt.TIMING_VERSION and np.allclose(p["session_t_daq_s"], t5)
    assert np.isnan(p["session"][250:]).all() and np.isfinite(p["session"][:250]).all()
    assert np.isnan(p["trials"][-1]).all() and np.isfinite(p["trials"][0]).all()
    assert p["session_lossy"][250:].all() and p["trials_lossy"][-1].all()


def test_precue_rows_union_of_arrival_and_cue_minus_4():
    """Per trial: 5 Hz bins in [strobe - 1, strobe + 2] OR [cue - 4, cue - 2), each bin once, none inside the 25 Hz
    grid (>= cue - 2 s); the between pool excludes all of it through the cue."""
    n5 = 600
    t5 = np.arange(n5) / 5.0                                           # true times = bin / 5
    z = {"position": np.array(["far_L", "far_R"]), "cue_daq_s": np.array([30.0, 80.0]),
         "t_ms": np.arange(-2000.0, 4001.0, 40.0), "daq_fit": np.array([1 / 250.0, 0.0])}
    tm = {"t5": t5, "drop5": np.zeros(n5), "drop_trials": np.zeros((2, 151))}
    strobe = np.array([27.5, 70.0])                                    # lead 2.5 s (overlaps cue-4..-2), lead 10 s
    tr, between = f.session_candidates("k", z, n5 * 50, tm, None, strobe)
    b0, b1 = np.array(tr[0][3]), np.array(tr[1][3])
    assert len(set(b0)) == len(b0) and np.allclose(t5[b0], np.arange(26.0, 28.0, 0.2))      # union: [26, 28)
    want1 = np.r_[np.arange(69.0, 72.0001, 0.2), np.arange(76.0, 78.0, 0.2)]               # arrival + cue-4..-2
    assert np.allclose(np.sort(t5[b1]), np.sort(want1))
    assert (t5[np.r_[b0, b1]] < np.repeat([28.0, 78.0], [len(b0), len(b1)])).all()          # before the grid
    bt = t5[[b for _, b, _ in between]]
    assert not ((bt >= 69.0) & (bt <= 86.0)).any() and not ((bt >= 26.0) & (bt <= 36.0)).any()
    picks = f.draw_rows(tr, between, 2, 0.5, np.random.default_rng(0))
    assert sorted(picks["arrival"]["k"]) == sorted(np.r_[b0, b1].tolist())


def test_even_draw_spreads_over_days_and_fills_short_ones():
    items = [(f"PS92_2026080{d}_cam4", i, "p") for d, n in ((1, 50), (2, 3), (3, 50)) for i in range(n)]
    got = f.even_draw(items, 30, np.random.default_rng(0))
    per = {d: sum(1 for k, *_ in got if k.startswith(f"PS92_2026080{d}")) for d in (1, 2, 3)}
    assert sum(per.values()) == 30 and per[2] == 3 and abs(per[1] - per[3]) <= 1     # short day gives all
    split = [("PS92_20260812_cam4_v1", i, "p") for i in range(10)] + [("PS92_20260812_cam4_v2", i, "p") for i in range(10)] \
        + [("PS92_20260813_cam4", i, "p") for i in range(20)]
    got = f.even_draw(split, 10, np.random.default_rng(1))
    assert sum(1 for k, *_ in got if "0812" in k) == 5                                   # a split day counts once
    tr = [(f"PS92_2026080{d}_cam4", i, pos) for d in (1, 2) for i, pos in enumerate(["L", "R"] * 20)]
    a = f.draw_rows(tr, [], 8, 0.0, np.random.default_rng(0), even_days=True)
    assert {k: len(v) for k, v in a["trials"].items()} == {"PS92_20260801_cam4": 4, "PS92_20260802_cam4": 4}
