"""o2_inference / o2_pose_runner: paths map to research.files, the job text is what SLURM needs, the CSV reads back
through `orofacial_clean.read_pose`, and -- the one that matters -- the standalone runner predicts EXACTLY what the
local predictor + prior predicts (needs DeepLabCut and the share; skipped otherwise)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import o2_inference as oi
from wfield_local import o2_pose_runner as runner
from wfield_local import orofacial_clean as oc


class _RV:
    def __init__(self, root):
        self._r = root

    def root(self, name):
        return self._r


def test_o2_path_maps_the_share_root_and_refuses_others():
    rv = _RV("Q:/MICROSCOPE/Priya")                       # a drive that does not exist: names kept as given
    assert oi.o2_path("Q:/MICROSCOPE/Priya/Behavior_Cameras/Widefield/20260908/PS93/cam4_x.avi", rv) == \
        "/n/files/Neurobio/MICROSCOPE/Priya/Behavior_Cameras/Widefield/20260908/PS93/cam4_x.avi"
    with pytest.raises(ValueError):
        oi.o2_path("C:/Users/x/video.avi", rv)


def test_o2_path_uses_the_on_disk_case(tmp_path):
    """O2 is Linux: paths.yaml says Behavior_Cameras/Widefield, the share has Behavior_cameras/widefield."""
    root = tmp_path / "Priya"
    (root / "Behavior_cameras" / "widefield" / "20260908").mkdir(parents=True)
    got = oi.o2_path(root / "Behavior_Cameras" / "Widefield" / "20260908" / "PS93" / "cam4_x.avi", _RV(str(root)))
    assert got == "/n/files/Neurobio/MICROSCOPE/Priya/Behavior_cameras/widefield/20260908/PS93/cam4_x.avi"


def test_job_text_has_array_requeue_and_resumable_runner():
    p = {"partition": "gpu", "gres": "gpu:1", "cpus_per_task": 6, "mem": "24G", "time": "24:00:00",
         "max_concurrent": 4, "exclude_nodes": "", "conda_module": "conda/x", "conda_env": "dlc3", "batch": 16,
         "chunk": 25000, "scratch_root": "/n/scratch/users/{u0}/{user}", "transfer_host": "transfer.rc.hms.harvard.edu",
         "deeplabcut_version": "3.0.1"}
    s = oi.sbatch_text(p, "ab123", "run1", 3)
    assert "#SBATCH --array=1-3%4" in s and "--requeue" in s and "/n/scratch/users/a/ab123/o2/run1" in s
    assert "o2_pose_runner.py --bundle" in s and "--batch 16 --chunk 25000" in s and "--exclude" not in s
    assert s.startswith("#!/bin/bash -l") and "\nset -eo pipefail\n" in s and "set -euo" not in s
    assert "\nconda activate dlc3\n" in s
    tasks = pd.DataFrame({"animal": ["PS93"], "date": ["20260908"], "video": ["cam4_x.avi"],
                          "video_o2": ["/n/files/Neurobio/MICROSCOPE/Priya/B/20260908/PS93/cam4_x.avi"]})
    c = oi.commands_text(p, "ab123", "run1", "/n/files/Neurobio/MICROSCOPE/Priya/DLC/o2/run1", tasks)
    assert "ssh ab123@transfer.rc.hms.harvard.edu" in c and "sbatch /n/scratch/users/a/ab123/o2/run1/run_pose.sbatch" in c
    assert "password" not in c.lower() and "sshpass" not in c


def test_csv_reads_back_through_read_pose(tmp_path):
    P = np.random.default_rng(0).random((7, 4, 3)).astype(np.float32)
    parts = ["nose", "jaw", "tongue", "spout"]
    runner.write_csv(tmp_path / "x.csv", P, parts, "DLC_test")
    d = oc.read_pose(tmp_path / "x.csv")
    assert len(d) == 7
    np.testing.assert_allclose(d["tongue_x"], P[:, 2, 0], atol=1e-4)
    np.testing.assert_allclose(d["spout_likelihood"], P[:, 3, 2], atol=1e-4)


_P = {"partition": "gpu", "gres": "gpu:1", "cpus_per_task": 6, "mem": "24G", "time": "24:00:00",
      "max_concurrent": 4, "exclude_nodes": "", "conda_module": "conda/x", "conda_env": "dlc3", "batch": 16,
      "chunk": 25000, "scratch_root": "/n/scratch/users/{u0}/{user}", "transfer_host": "transfer.rc.hms.harvard.edu",
      "deeplabcut_version": "3.0.1"}


def test_lp_job_text_runs_the_lp_runner_in_its_own_env():
    p = {**_P, "kind": "lp", "conda_env": "lp", "reader": "opencv", "precision": "fp32",
         "lightning_pose_version": "2.4.2"}
    s = oi.sbatch_text(p, "ab123", "lp1", 2)
    b = "/n/scratch/users/a/ab123/o2/lp1"
    assert "#SBATCH -J lp_lp1" in s and "#SBATCH --array=1-2%4" in s and "\nconda activate lp\n" in s
    assert f"python {b}/o2_lp_runner.py --model {b}/model --video " in s and "--reader opencv --precision fp32" in s
    assert "o2_pose_runner" not in s and "--chunk" not in s
    tasks = pd.DataFrame({"animal": ["PS93"], "date": ["20260814"], "video": ["cam4_x.avi"],
                          "video_o2": ["/n/files/Neurobio/MICROSCOPE/Priya/B/20260814/PS93/cam4_x.avi"]})
    c = oi.commands_text(p, "ab123", "lp1", "/n/files/x/o2/lp1", tasks)
    assert 'pip install "lightning-pose==2.4.2"' in c and "conda create -y -n lp" in c
    assert "sbatch --parsable" in c and "--dependency=afterok:$SMOKE" in c and "o2_pose_runner" not in c
    assert "--files-from=" in c and "srun -p gpu" not in c                    # no interactive GPU wait
    sm = oi.smoke_sbatch_text(p, "ab123", "lp1", "/n/scratch/v.avi")
    assert "o2_lp_runner.py --model" in sm and "--bench 2000" in sm and "\nconda activate lp\n" in sm
    # the DLC job is unchanged by the refactor
    sd = oi.sbatch_text({**_P, "kind": "dlc"}, "ab123", "run1", 3)
    assert "#SBATCH -J pose_run1" in sd and "o2_pose_runner.py --bundle" in sd and "--batch 16 --chunk 25000" in sd


def test_lp_runner_csv_to_pose_and_outputs_follow_the_shared_contract(tmp_path):
    from wfield_local import o2_lp_runner as lpr
    parts = ["nose", "jaw", "tongue", "spout"]
    P = np.random.default_rng(1).random((9, 4, 3)).astype(np.float32)
    cols = pd.MultiIndex.from_product([["heatmap_tracker"], parts, ["x", "y", "likelihood"]])
    pd.DataFrame(P.reshape(9, -1), columns=cols).to_csv(tmp_path / "lp.csv")          # LP's own CSV layout
    pose, got_parts, scorer = lpr.lp_csv_to_pose(tmp_path / "lp.csv")
    assert got_parts == parts and scorer == "heatmap_tracker"
    np.testing.assert_allclose(pose, P, atol=1e-6)
    lpr.write_outputs(tmp_path / "out", "cam4_x", pose, parts, "LP_r4", {"fps": 250.0, "wall_s": 1.0, "batch": None})
    info = json.loads((tmp_path / "out" / "cam4_x_done.json").read_text())
    assert info["n_frames"] == 9 and info["scorer"] == "LP_r4"
    z = np.load(tmp_path / "out" / "cam4_x_LP_r4.npz")
    assert list(z["parts"]) == parts and z["pose"].shape == (9, 4, 3)
    d = oc.read_pose(tmp_path / "out" / "cam4_x_LP_r4.csv")
    np.testing.assert_allclose(d["tongue_x"], P[:, 2, 0], atol=1e-4)


def test_params_lp_block_overrides_shared_keys(monkeypatch):
    monkeypatch.setattr(oi.config, "defaults", lambda: {"o2_inference": {**_P, "lp": {"conda_env": "lp", "mem": "32G"}}})
    lp, dlc = oi.params("lp"), oi.params("dlc")
    assert lp["kind"] == "lp" and lp["conda_env"] == "lp" and lp["mem"] == "32G" and lp["partition"] == "gpu"
    assert dlc["kind"] == "dlc" and dlc["conda_env"] == "dlc3" and "lp" not in dlc


def test_widefield_sessions_come_from_the_registry(monkeypatch):
    monkeypatch.setattr(oi.config, "session_phase", lambda a, d: "excluded" if d == "0817" else "pre")
    monkeypatch.setattr(oi.config, "load_sessions", lambda: [{"label": "PS93_0814"}, {"label": "PS92_0602"}, {"label": "PS92_0817"},
                                                             {"label": "PS93_0814"}])
    assert oi.widefield_sessions() == [("PS92", "20260602"), ("PS93", "20260814")]


def _local_setup():
    pytest.importorskip("deeplabcut")
    from wfield_local import dlc_prior, dlc_train
    from wfield_local.dlc_iti_frames import current_snapshot
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    try:
        td, snap = current_snapshot(dlc_train.train_project(rv), None)
        vid = oi.videos_for(rv, "PS93", "20260908")[0]
    except Exception as e:                       # share not mounted / no model here
        pytest.skip(f"needs the share and the trained model: {e}")
    return rv, td, snap, vid, dlc_prior, dlc_train


def test_runner_matches_local_predictor_with_prior(tmp_path):
    rv, td, snap, vid, dlc_prior, dlc_train = _local_setup()
    import cv2

    from wfield_local.dlc_hard_frames import pose_predictor
    box = dlc_prior.active(rv, cam="cam4") or {}
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "bundle.json").write_text(json.dumps({
        "parts": list(dlc_train.parts()), "scorer": "DLC_t", "pytorch_config": str(td / "pytorch_config.yaml"),
        "snapshot": str(snap), "prior": {k: [float(v) for v in b] for k, b in box.items()}}))
    n = 150
    out = runner.run_video(runner.load_bundle(tmp_path / "b"), vid, tmp_path / "out", batch=16, chunk=64,
                           max_frames=n)
    got = oc.read_pose(out)
    cap = cv2.VideoCapture(str(vid))
    frames = [cap.read()[1] for _ in range(n)]
    cap.release()
    with dlc_prior.apply(rv, cam="cam4"):
        ref = pose_predictor(rv)(frames)                         # (n, 4, 3), parts in dlc_train order
    for k, bp in enumerate(dlc_train.parts()):
        np.testing.assert_allclose(got[f"{bp}_x"], ref[:, k, 0], atol=1e-3)
        np.testing.assert_allclose(got[f"{bp}_y"], ref[:, k, 1], atol=1e-3)
        np.testing.assert_allclose(got[f"{bp}_likelihood"], ref[:, k, 2], atol=1e-3)
    done = json.loads(next((tmp_path / "out").glob("*_done.json")).read_text())
    assert done["n_frames"] == n and set(done["prior"]) == set(box)
    # resumable: a second call finds every chunk and only re-assembles
    out2 = runner.run_video(runner.load_bundle(tmp_path / "b"), vid, tmp_path / "out", batch=16, chunk=64,
                            max_frames=n)
    assert Path(out2).read_text() == Path(out).read_text()


def _bounds(spans_s):
    """Trial bounds with strobe / cue / stop in DAQ s (fs 1000 Hz, 250 fps, frame 0 = t 0)."""
    return pd.DataFrame([{"trial_id": k + 1, "pos_name": "far_R", "strobe_s": a, "cue_s": c, "stop_s": z,
                          "stop_source": "daq_trial_end"} for k, (a, c, z) in enumerate(spans_s)])


_TPL = {"fs_daq": 1000.0, "fps_cam": 250.0, "slope_daqSample_per_camFrame": 4.0, "intercept_daqSample": 0.0}


def test_trial_windows_are_cut_at_the_next_trials_start_so_no_frame_is_in_two_trials():
    # stop + 3 s of trial 1 (13 s) runs past trial 2's strobe - 0.5 s window start (11.5 s)
    b = _bounds([(2.0, 4.0, 10.0), (12.0, 14.0, 20.0), (30.0, 32.0, 38.0)])
    sp = oi.trial_frame_spans(b, _TPL, n_full=100_000)
    frames = np.concatenate([np.arange(f0, f1) for f0, f1, _ in sp])
    assert len(frames) == len(np.unique(frames))
    assert sp[0][1] == sp[1][0] == round(11.5 * 250)        # trial 1 ends exactly where trial 2 begins
    assert sp[1][1] == round(23.0 * 250)                     # no neighbour inside -> stop + 3 s kept
    assert sp[2][1] == round(41.0 * 250) and sp[0][0] == round(1.5 * 250)
    assert [c for *_, c in sp] == [1000, 3500, 8000]


def test_full_to_windows_writes_every_video_frame_at_most_once(tmp_path, monkeypatch):
    from wfield_local import trial_windows as TW
    tdir = tmp_path / "cam4" / "PSX"
    tdir.mkdir(parents=True)
    np.savez(tdir / "20260101.npz", **_TPL)
    monkeypatch.setattr(TW, "trial_bounds", lambda *a, **k: _bounds([(2.0, 4.0, 10.0), (12.0, 14.0, 20.0)]))
    pose = np.random.default_rng(0).random((8000, 4, 3))
    oi.full_to_windows(pose, ["nose", "jaw", "tongue", "spout"], "DLC_x", "PSX", "20260101", tmp_path / "out",
                       _RV(tmp_path))
    idx = pd.read_csv(tmp_path / "out" / "windows_index.csv")
    assert idx.src_frame.is_unique
    assert idx.groupby("trial_k").src_frame.max().iloc[0] < idx.groupby("trial_k").src_frame.min().iloc[1]


def test_split_day_cuts_each_trial_from_the_video_that_covers_it(tmp_path, monkeypatch):
    """PS92 0812-style day: video 1 covers DAQ 0-20 s, video 2 starts at DAQ 100 s (its frame 0); trials in each
    segment come from their own video, with that video's frame numbers, and nothing is cut across the gap."""
    from wfield_local import trial_windows as TW
    monkeypatch.setattr(TW, "trial_bounds", lambda *a, **k: _bounds([(2.0, 4.0, 10.0), (102.0, 104.0, 110.0)]))
    t1 = {**_TPL, "segment_daq_sec": np.array([0.0, 20.0])}
    t2 = {**_TPL, "intercept_daqSample": 100_000.0, "segment_daq_sec": np.array([100.0, 130.0])}
    p1 = np.zeros((10_000, 4, 3))
    p2 = np.ones((10_000, 4, 3))
    k = oi.full_to_windows(p1, ["nose", "jaw", "tongue", "spout"], "LP_x", "PSX", "20260101", tmp_path / "out",
                           _RV(tmp_path), model="LP", videos=[(p1, t1, "cam4_a.avi"), (p2, t2, "cam4_b.avi")])
    assert k == 2
    idx = pd.read_csv(tmp_path / "out" / "windows_index.csv")
    assert set(idx.video) == {"cam4_a.avi", "cam4_b.avi"}
    a, b = idx[idx.video == "cam4_a.avi"], idx[idx.video == "cam4_b.avi"]
    assert (a.trial_id == 1).all() and (b.trial_id == 2).all()
    assert b.src_frame.min() >= 0 and b.src_frame.max() < 10_000            # video 2's own frame numbers
    assert abs(b.loc[b.t_ms.abs().idxmin(), "src_frame"] - 1000) <= 1     # cue at DAQ 104 s = video-2 frame 1000
    poses = oc.read_pose(tmp_path / "out" / "windows_LP.csv")
    assert (poses.loc[idx.video == "cam4_a.avi", "jaw_x"] == 0).all() and (poses.loc[idx.video == "cam4_b.avi", "jaw_x"] == 1).all()
