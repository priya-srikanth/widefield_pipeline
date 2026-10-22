"""scripts/face_rhythm_factors.py on a synthetic bundle in the runner's file formats: planted non-negative factors are
recovered, the leave-one-animal-out / freeze / projection / own-fit / movement-power steps run, and refreezing is
refused. Small sizes (seconds)."""
import json

import numpy as np
import pytest

from scripts import face_rhythm_factors as f

P, F, R = 30, 10, 3


def _bundle(tmp_path):
    rng = np.random.default_rng(1)
    out, sd = tmp_path / "out", tmp_path / "factors" / "test"
    out.mkdir(parents=True)
    sd.mkdir(parents=True)
    A = rng.random((2 * P, R)) ** 3
    B = np.full((F, R), 0.05)
    B[[2, 5, 8], range(R)] = 1.0
    plan = {"set": "test", "params": {"inits": 2, "n_iter": 150, "trial_ds": 2, "point_pcs": 4}, "fit": {},
            "points_fit": {}, "session_rows": {}, "tasks": []}
    modes = rng.standard_normal((2, 2 * P))                  # planted 2-D displacement structure
    for a, sc in {"PS92": 1.0, "PS93": 0.6, "PS94": 1.3}.items():
        for d, ep in (("20260810", "pre"), ("20260820", "acute")):
            key, ntr, n5 = f"{a}_{d}_cam4", 6, 300
            cues = 10.0 + 9.0 * np.arange(ntr)
            mk = lambda C, shp, sc=sc: (sc * np.einsum("tr,pr,fr->tpf", C, A, B) + 0.01 * rng.random((len(C), 2 * P, F))
                                 ).reshape(shp + (2, P, F)).astype(np.float16)
            np.savez(out / f"{key}_trials.npz", spec=mk(rng.random((ntr * 151, R)) ** 2, (ntr, 151)),
                     t_ms=np.arange(-2000.0, 4001, 40.0, dtype=np.float32), trial_id=np.arange(ntr),
                     position=np.array(["far_L", "far_R"] * (ntr // 2)), cue_daq_s=cues,
                     freqs=np.geomspace(1, 60, F), home=rng.random((P, 2)) * 100)
            np.savez(out / f"{key}_session.npz", spec=mk(rng.random((n5, R)) ** 2, (n5,)),
                     t_daq_s=np.arange(n5) / 5.0 + 0.1, rate_hz=np.float32(5))
            n25 = 5 * n5
            disp = (rng.standard_normal((n25, 2)) @ modes + 3.0).reshape(n25, P, 2).astype(np.float16)
            np.savez(out / f"{key}_points.npz", disp=disp, viol_frac=np.zeros(n25, np.float16),
                     t_daq_s=np.arange(n25) / 25.0, home=np.zeros((P, 2)))
            plan["tasks"].append({"key": key, "animal": a, "date": d, "epoch": ep})
            plan["session_rows"][key] = {"trials": {key: [0, 1, 2, 3]}, "between": {key: list(range(150, 300, 3))}}
        k = f"{a}_20260810_cam4"
        plan["fit"][a] = {"sessions": [k], "trials": {k: [0, 1, 2, 3]}, "between": {k: list(range(150, 300, 3))}}
        plan["points_fit"][a] = {k: list(range(0, 1500, 2))}
    (sd / "plan.json").write_text(json.dumps(plan))
    return sd, A, B, plan


def test_pipeline_end_to_end(tmp_path):
    sd, A, B, plan = _bundle(tmp_path)
    f.cmd_build(sd)
    f.cmd_fit(sd, R)
    f.cmd_select(sd, R)
    with pytest.raises(SystemExit, match="already frozen"):
        f.cmd_select(sd, R)
    M = np.load(sd / "master.npz")
    assert f.similarity([A, B], [M["spatial"].reshape(-1, R), M["spectral"]]) > 0.99
    for i in range(1, len(plan["tasks"]) + 1):
        f.cmd_project(sd, i)
        f.cmd_fitsession(sd, i, inits=2)
    p = np.load(sd / "proj" / "PS93_20260820_cam4.npz")
    assert p["trials"].shape == (6, 151, R) and p["session"].shape == (300, R)
    assert np.nanmedian(p["session_err"]) < 0.2
    assert p["power_response"].shape == (2, 2, P, F) and list(p["power_positions"]) == ["far_L", "far_R"]
    assert p["points_pc"].shape == (1500, 4) and p["points_t_daq_s"].shape == (1500,)
    assert M["points_explained"][:2].sum() > 0.99                # the planted 2-D structure
    assert abs(float(np.nanmean(p["points_pc"]))) < 0.1          # sessions are centred on their own mean
    o = np.load(sd / "own" / "PS93_20260820_cam4.npz")
    assert o["evr_own"] >= o["evr_frozen"] - 0.01            # a session's own fit is at least as good
    assert o["cos_spatial"].min() > 0.95                     # same structure -> no morphing


def test_window_power_classes():
    t5 = np.arange(0, 60, 0.2)
    S = np.ones((len(t5), 2, 1, 1), np.float16)
    S[(t5 >= 20) & (t5 < 24)] = 3                            # response window of the cue at 20 s
    pw = f.window_power(S, 1.0, t5, np.array([20.0, 40.0]), np.array(["a", "b"]))
    assert np.allclose(pw["power_response"][0], 3) and np.allclose(pw["power_response"][1], 1)
    assert np.allclose(pw["power_between"], 1) and pw["power_n_bins"][0] > 0
    assert pw["power_n_bins"][2] == 20                       # 4 s at 5 Hz, response of position "a"


def test_draw_rows_balanced_and_seeded():
    trials = [("k", i, "L" if i % 3 else "R") for i in range(30)]
    between = [("k", b) for b in range(500)]
    a = f.draw_rows(trials, between, 10, 0.25, np.random.default_rng(0))
    b = f.draw_rows(trials, between, 10, 0.25, np.random.default_rng(0))
    assert a == b and len(a["trials"]["k"]) == 10
    assert len(a["between"]["k"]) == round(0.25 * 10 * 75)
