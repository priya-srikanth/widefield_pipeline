"""`session_licks`: the one session lick-table loader (legacy cam4 licks_all vs the fused four-camera train), the lick
KEY join that replaces the lick_uid row join, the 60 ms video-lick floor on the fused train, and the format guard."""
import numpy as np
import pandas as pd
import pytest

from wfield_local import session_licks as SL


def _fused(rows):
    cols = ["trial_id", "position", "on_ms", "off_ms", "t_peak_ms", "seen_cam4", "seen_cam1", "seen_cam2", "seen_cam3"]
    U = pd.DataFrame(rows, columns=cols)
    S = U[[c for c in cols if c.startswith("seen_")]]
    U["n_views"] = S.sum(1)
    U["guarded"] = False
    U["cams"] = ["+".join(sorted(c[5:] for c in S.columns if r[c])) for _, r in S.iterrows()]
    U["lick_idx"] = U.groupby("trial_id").cumcount()
    return U


def _ix(trials=(1, 2), cue_s=(10.0, 20.0)):
    rows = []
    for t, c in zip(trials, cue_s):
        for tm in np.arange(-3000.0, 6000.0, 4.0):
            rows.append({"trial_id": t, "t_ms": tm, "daq_s": c + tm / 1000.0, "strobe_ms": -2500.0, "stop_ms": 3000.0})
    return pd.DataFrame(rows)


def test_fused_run_env(monkeypatch):
    monkeypatch.delenv(SL.FUSED_ENV, raising=False)
    assert SL.fused_run() == SL.FUSED_DEFAULT
    monkeypatch.setenv(SL.FUSED_ENV, "off")
    assert SL.fused_run() is None and not SL.fused_on() and SL.signature() == ""
    monkeypatch.setenv(SL.FUSED_ENV, "lp_side_x_gapfix60")
    assert SL.fused_run() == "lp_side_x_gapfix60" and SL.fused_on()
    monkeypatch.setenv(SL.FUSED_ENV, "../escape")
    with pytest.raises(ValueError):
        SL.fused_run()


def test_lick_key_and_key_join_survive_reorder_and_added_licks():
    A = pd.DataFrame({"trial_id": [1, 1, 2, 2], "t_ms": [100.0, 260.0, -5000.0, 12345.6]})
    A["lick_key"] = SL.lick_key(A.trial_id, A.t_ms)
    assert A.lick_key.tolist() == ["1:100.0", "1:260.0", "2:-5000.0", "2:12345.6"]
    K = A.iloc[[3, 0]].reset_index(drop=True)[["trial_id", "lick_key"]]
    # the lick table is re-ordered and gains two licks: the key still finds the same licks
    A2 = pd.concat([A, pd.DataFrame({"trial_id": [1, 2], "t_ms": [50.0, 800.0]})], ignore_index=True)
    A2["lick_key"] = SL.lick_key(A2.trial_id, A2.t_ms)
    A2 = A2.sample(frac=1.0, random_state=3).reset_index(drop=True)
    r = SL.rows_for(K, A2, "t")
    assert A2.t_ms.to_numpy()[r].tolist() == [12345.6, 100.0]
    # a key the table does not hold is an ERROR, never a silent miss
    K2 = K.assign(lick_key=["1:999.0", K.lick_key[1]])
    with pytest.raises(SL.LickJoinError):
        SL.rows_for(K2, A2, "t")


def test_legacy_lick_uid_is_verified_not_trusted():
    A = pd.DataFrame({"trial_id": [1, 1, 2], "t_ms": [100.0, 260.0, -22296.0]})
    K = pd.DataFrame({"trial_id": [1, 2], "lick_uid": [1, 2], "t_peak_cam4": [260.0, -2.23e4]})   # %.4g peak
    assert SL.rows_for(K, A, "t").tolist() == [1, 2]
    A_new = pd.concat([pd.DataFrame({"trial_id": [1], "t_ms": [40.0]}), A], ignore_index=True)    # one lick added
    with pytest.raises(SL.LickJoinError):
        SL.rows_for(K, A_new, "t")                         # the old row index now points at another lick


def test_apply_floor_merges_sub_60ms_peaks_keeps_priority_peak():
    U = _fused([(1, "far_L", 0, 40, 20, False, True, False, False),
                (1, "far_L", 35, 90, 70, True, False, True, False),       # 50 ms after: one lick, cam4 peak wins
                (1, "far_L", 200, 260, 230, True, True, True, True),
                (2, "far_L", 0, 30, 10, True, False, False, False),
                (2, "far_L", 40, 80, 75, False, False, True, True)])      # 65 ms: two licks
    F = SL.apply_floor(U)
    t1 = F[F.trial_id == 1]
    assert len(t1) == 2 and t1.t_peak_ms.tolist() == [70, 230]
    m = t1.iloc[0]
    assert (m.on_ms, m.off_ms) == (0, 90) and m.cams == "cam1+cam2+cam4" and m.n_views == 3 and m.floor_merged == 1
    assert len(F[F.trial_id == 2]) == 2 and F.lick_idx.tolist() == [0, 1, 0, 1]
    assert SL.apply_floor(U, min_ili_ms=0).floor_merged.sum() == 0


def test_fused_table_context_and_daq_clock():
    U = _fused([(1, "far_L", -1200, -1100, -1150, True, False, False, False),   # pre-cue
                (1, "far_L", 30, 80, 50, False, True, False, False),            # 0-70 ms: excluded
                (1, "far_L", 300, 360, 330, True, True, False, False),          # post-cue
                (1, "far_L", 4000, 4040, 4020, False, False, True, True),       # after stop: ITI
                (9, "far_L", 100, 140, 120, True, False, False, False)])        # trial without windows: no clock
    A = SL.fused_table(U, _ix())
    assert A.context.tolist() == ["pre_cue", "excluded_0_70", "post_cue", "iti"]
    assert A.table.tolist() == ["licks_precue", "licks_excluded_0_70", "licks", "licks_iti"]
    # 30 ms is off the 4 ms frame grid: the nearest frame's DAQ time (32 ms), as licks_all (`trial_ms_to_daq`)
    np.testing.assert_allclose(A.daq_on_s, [10 - 1.2, 10.032, 10.3, 14.0])
    np.testing.assert_allclose(A.daq_peak_s, [10 - 1.15, 10.05, 10.33, 14.02], atol=0.0021)   # nearest frame
    assert A.lick_key.tolist() == ["1:-1150.0", "1:50.0", "1:330.0", "1:4020.0"]
    # 11-22: lick_idx counts within (trial, table), as licks_all: the post-cue lick is lick 0 of the post-cue table
    assert (A.source == "fused").all() and A.lick_idx.tolist() == [0, 0, 0, 0]
    assert A.segment.tolist() == ["", "", "", "tail"]          # ITI after the stop = the trial's tail


def test_fused_lick_idx_first_post_cue_lick_is_zero():
    U = _fused([(1, "far_L", -1200, -1100, -1150, True, False, False, False),
                (1, "far_L", -800, -760, -780, True, False, False, False),
                (1, "far_L", 300, 360, 330, True, True, False, False),
                (1, "far_L", 450, 500, 470, True, True, False, False),
                (1, "far_L", 600, 650, 620, True, True, False, False)])
    A = SL.fused_table(U, _ix()).sort_values("t_ms")
    assert A[A.context == "pre_cue"].lick_idx.tolist() == [0, 1]
    assert A[A.context == "post_cue"].lick_idx.tolist() == [0, 1, 2]


def test_renumber_lick_idx_within_table_in_peak_order():
    A = pd.DataFrame({"session": ["s"] * 5, "trial_id": [1, 1, 1, 2, 2], "table": ["licks_precue", "licks", "licks",
                                                                                   "licks", "licks"],
                      "t": [-5.0, 2.0, 1.0, 3.0, 4.0]}, index=[10, 11, 12, 13, 14])
    assert SL.renumber_lick_idx(A, ["session", "trial_id"], "t").tolist() == [0, 1, 0, 0, 1]


def test_session_licks_fused_refuses_old_format_and_reads_new(tmp_path, monkeypatch):
    sd = tmp_path / "cam4"
    sd.mkdir()
    ix = _ix()
    np.savez(sd / "windows.npz", ix_trial_id=ix.trial_id.to_numpy(), ix_t_ms=ix.t_ms.to_numpy(),
             ix_daq_s=ix.daq_s.to_numpy(), ix_strobe_ms=ix.strobe_ms.to_numpy(), ix_stop_ms=ix.stop_ms.to_numpy())
    fd = tmp_path / "fused"
    fd.mkdir()
    U = _fused([(1, "far_L", 300, 360, 330, True, True, False, False),
                (1, "far_L", 380, 420, 370, False, True, False, False),        # 40 ms after: floor merges it
                (2, "far_L", 500, 540, 520, False, False, True, False)])       # one side view: keep False
    U["keep"] = [True, True, False]
    U.to_csv(fd / "PS92_20260818_fused.csv", index=False)
    monkeypatch.setenv(SL.FUSED_ENV, "fused")
    monkeypatch.setattr(SL, "_cam4_dir", lambda rv, a, d: sd)
    monkeypatch.setattr(SL, "fused_dir", lambda rv=None, run=None: fd)
    SL._MEMO.clear()
    with pytest.raises(SL.FusedFormatError):
        SL.session_licks("PS92_0818", rv=object())
    SL.write_manifest(fd, bundle="b")
    A = SL.session_licks("PS92_0818", rv=object())
    assert len(A) == 1 and A.t_ms.iloc[0] == 330 and A.off_ms.iloc[0] == 420 and A.floor_merged.iloc[0] == 1
    np.testing.assert_allclose(SL.session_onsets_s("PS92_0818", rv=object()), [10.3])


def test_drop_cam1_only_rows():
    A = pd.DataFrame({"cams": ["cam1", "cam1+cam4", "cam4", "cam1+cam2"], "x": range(4)})
    assert SL.drop_cam1_only(A).x.tolist() == [1, 2, 3]
    assert SL.drop_cam1_only(A.drop(columns="cams")).x.tolist() == [0, 1, 2, 3]       # legacy table: unchanged
