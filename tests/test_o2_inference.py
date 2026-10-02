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
    rv = _RV("M:/MICROSCOPE/Priya")
    assert oi.o2_path("M:/MICROSCOPE/Priya/Behavior_Cameras/Widefield/20260908/PS93/cam4_x.avi", rv) == \
        "/n/files/Neurobio/MICROSCOPE/Priya/Behavior_Cameras/Widefield/20260908/PS93/cam4_x.avi"
    with pytest.raises(ValueError):
        oi.o2_path("C:/Users/x/video.avi", rv)


def test_job_text_has_array_requeue_and_resumable_runner():
    p = {"partition": "gpu", "gres": "gpu:1", "cpus_per_task": 6, "mem": "24G", "time": "24:00:00",
         "max_concurrent": 4, "exclude_nodes": "", "conda_module": "conda/x", "conda_env": "dlc3", "batch": 16,
         "chunk": 25000, "scratch_root": "/n/scratch/users/{u0}/{user}", "transfer_host": "transfer.rc.hms.harvard.edu",
         "deeplabcut_version": "3.0.1"}
    s = oi.sbatch_text(p, "ab123", "run1", 3)
    assert "#SBATCH --array=1-3%4" in s and "--requeue" in s and "/n/scratch/users/a/ab123/o2/run1" in s
    assert "o2_pose_runner.py --bundle" in s and "--batch 16 --chunk 25000" in s and "--exclude" not in s
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
