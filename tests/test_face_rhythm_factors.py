"""The factor stages (wfield_local/face_rhythm/factors/, launcher scripts/face_rhythm_factors.py) on a synthetic bundle in the runner's file formats: planted non-negative factors are
recovered, the leave-one-animal-out / freeze / projection / own-fit / movement-power steps run, and refreezing is
refused. Small sizes (seconds)."""
import json

import numpy as np
import pytest

from wfield_local.face_rhythm import factors as f

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
    fj = json.loads((sd / f"rank{R:02d}.json").read_text())       # per-factor stability, every start's factors kept
    fz = np.load(sd / f"rank{R:02d}.npz")
    assert len(fj["stability_per_factor_min"]) == R and min(fj["stability_per_factor_min"]) > 0.9
    assert fz["A_starts"].shape[0] == len(fj["rel_errs"]) and fz["B_starts"].shape[1:] == fz["B"].shape
    f.cmd_splithalf(sd, R, inits=1)                                # disjoint halves recover the same planted factors
    sh = json.loads((sd / "splithalf" / f"rank{R:02d}.json").read_text())
    assert sh["similarity_halves"] > 0.9 and min(sh["similarity_half_to_full"]) > 0.9 and sum(sh["n_trials"]) == 12
    pl = json.loads((sd / "plan.json").read_text())                # leave one pre DAY out (the post-stroke noise floor)
    pl["loso_jobs"], pl["role"] = [[R, f"{a}_20260810"] for a in ("PS92", "PS93", "PS94")], "pilot"
    (sd / "plan.json").write_text(json.dumps(pl))
    for j in (1, 2, 3):
        f.cmd_loso(sd, j)
    lo = json.loads((sd / "loso" / f"rank{R:02d}_PS92_20260810.json").read_text())
    assert lo["n_session_rows"] > 0 and lo["evr_in_fit"] > 0.95
    assert lo["evr_held_out"] > lo["evr_in_fit"] - 0.02          # shared planted structure: held out ~ in fit
    f.cmd_report(sd)
    assert (sd / "report" / "loso_by_rank.csv").exists()
    assert (sd / "report" / "stability_per_factor.csv").exists()
    f.cmd_select(sd, R)
    assert str(np.load(sd / "master.npz")["role"]) == "pilot"
    with pytest.raises(SystemExit, match="already frozen"):
        f.cmd_select(sd, R)
    M = np.load(sd / "master.npz")
    assert f.similarity([A, B], [M["spatial"].reshape(-1, R), M["spectral"]]) > 0.99
    for i in range(1, len(plan["tasks"]) + 1):
        f.cmd_project(sd, i)
        f.cmd_fitsession(sd, i, inits=2)
    assert str(np.load(sd / "proj" / "PS93_20260810_cam4.npz")["basis"]) == "full_in_sample"   # ONE basis (10-23)
    for i in range(1, len(plan["tasks"]) + 1):
        f.cmd_project(sd, i, loso=True)                       # held-out pre projections, FR scoring only
    assert str(np.load(sd / "proj_loso" / "PS93_20260810_cam4.npz")["basis"]) == "loso"
    assert not (sd / "proj_loso" / "PS93_20260820_cam4.npz").exists()                   # post: not written there
    assert str(np.load(sd / "own" / "PS93_20260810_cam4.npz")["basis"]) == "loso"
    p = np.load(sd / "proj" / "PS93_20260820_cam4.npz")
    assert str(p["basis"]) == "full"                                                    # post: the full pre fit
    assert p["trials"].shape == (6, 151, R) and p["session"].shape == (300, R)
    assert np.nanmedian(p["session_err"]) < 0.2
    assert p["power_response"].shape == (2, 2, P, F) and list(p["power_positions"]) == ["far_L", "far_R"]
    assert p["points_pc"].shape == (1500, 4) and p["points_t_daq_s"].shape == (1500,)
    assert M["points_explained"][:2].sum() > 0.99                # the planted 2-D structure
    assert abs(float(np.nanmean(p["points_pc"]))) < 0.1          # sessions are centred on their own mean
    o = np.load(sd / "own" / "PS93_20260820_cam4.npz")
    assert o["evr_own"] >= o["evr_frozen"] - 0.01            # a session's own fit is at least as good
    assert o["cos_spatial"].min() > 0.95                     # same structure -> no morphing


def test_pre_task_without_its_left_out_basis_is_refused(tmp_path):
    sd, _, _, plan = _bundle(tmp_path)
    f.cmd_build(sd)
    f.cmd_fit(sd, R)
    pl = json.loads((sd / "plan.json").read_text())
    pl["loso_jobs"] = [[R, "PS92_20260810"]]
    (sd / "plan.json").write_text(json.dumps(pl))
    f.cmd_select(sd, R)
    pre = next(i for i, t in enumerate(plan["tasks"], 1) if t["epoch"] == "pre")
    f.cmd_project(sd, pre)                                     # proj/: the full frozen basis, no refusal (10-23)
    with pytest.raises(SystemExit, match="loso array"):
        f.cmd_project(sd, pre, loso=True)                      # proj_loso/ needs the left-out-day basis


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
    c = f.draw_rows(trials, between, 20, 0.25, np.random.default_rng(0), ds=4)   # 2x trials at half the row rate
    assert c["ds"] == 4 and len(c["trials"]["k"]) == 20
    assert len(c["between"]["k"]) == round(0.25 * 20 * (151 // 4))             # same trial-row share



def test_normaliser_inverts_fr_per_bin_normalisation():
    """FR (VQT_Analyzer): s_norm = s / (nf m + 1 - nf), m = mean over rows of the power summed over frequencies, then x
    freqs^ex. `normaliser` must recover D = nf m + 1 - nf from the saved rows."""
    rng = np.random.default_rng(3)
    freqs, nf, ex = np.geomspace(1, 60, F), 0.95, 0.5
    raw = rng.random((40, 2 * P, F)) * rng.uniform(0.05, 20, (40, 1, 1))      # bins with very different movement
    D = nf * raw.sum(-1).mean(-1) + (1 - nf)
    saved = raw / D[:, None, None] * freqs[None, None] ** ex
    assert np.allclose(f.normaliser(saved, freqs, nf, ex), D, rtol=1e-4)


def test_streamed_points_and_one_pass_power_match_the_in_memory_path(tmp_path, monkeypatch):
    """project's streaming (points in blocks, power means riding on the projection pass) = the in-memory helpers."""
    rng = np.random.default_rng(3)
    n, Pn = 2600, 7
    t25 = np.arange(n) / 25.0
    disp = rng.standard_normal((n, Pn, 2)).astype(np.float16)
    disp[[5, 1000, 1999]] = np.nan                                       # NaN bins, one at a block edge
    pf = tmp_path / "x_points.npz"
    np.savez(pf, disp=disp, t_daq_s=t25)
    comps = rng.standard_normal((3, 2 * Pn)).astype(np.float32)
    cues, pos = np.array([20.0, 45.0, 70.0]), np.array(["a", "b", "a"])
    monkeypatch.setattr(f.project, "POINTS_CHUNK", 1000)                  # several blocks
    monkeypatch.setattr(f.fit, "POINTS_CHUNK", 1000)
    st = f.points_stream(pf, comps, cues, pos)
    Xp, _ = f.fit._centred_disp(pf)
    ref = f.window_speed(Xp, t25, cues, pos)
    assert np.allclose(st["points_pc"], Xp @ comps.T, equal_nan=True, atol=1e-5)
    for k in ("speed_between", "speed_precue", "speed_response"):
        assert np.allclose(st[k], ref[k], equal_nan=True, rtol=1e-5)
    assert np.array_equal(st["speed_n_bins"], ref["speed_n_bins"])
    S = rng.random((600, 2, Pn, 4)).astype(np.float16)                   # power means in the projection's pass
    t5 = np.arange(600) / 5.0
    lab, ps = f.window_labels(t5, cues, pos, f.EDGE_BINS)
    acc = f.ClassMeans(lab, 1 + 2 * len(ps), S.shape[1:])
    X = f.cp._scaled(S, 2.0, on_chunk=lambda c0, x: acc.add(c0, x.reshape(len(x), *S.shape[1:])))
    f.project_rows(X, rng.random((2 * Pn, 2)), rng.random((4, 2)))
    one, ref2 = f.project._power_dict(*acc.result(), ps), f.window_power(S, 2.0, t5, cues, pos)
    for k in ("power_between", "power_precue", "power_response"):
        assert np.allclose(one[k], ref2[k], equal_nan=True, rtol=1e-6)


def test_gather_rows_uses_the_plans_row_spacing(tmp_path):
    """A pick's ``ds`` sets the fit-row spacing (plan --trial-ds); picks without it (older plans) use TRIAL_DS."""
    k = "PS92_20260810_cam4"
    sd = _bundle(tmp_path)[0]
    X4, ix4 = f.gather_rows(sd, {"trials": {k: [0, 1]}, "between": {}, "ds": 4})
    X2, _ = f.gather_rows(sd, {"trials": {k: [0, 1]}, "between": {}})
    assert len(X4) == 2 * (151 // 4) and len(X2) == 2 * (151 // f.TRIAL_DS)
    assert {r[3] for r in ix4} == set(range(151 // 4))


def test_deployed_set_runs_its_own_copy_and_old_imports_resolve(tmp_path):
    """O2 has no repo: `deploy_code` puts the launcher + package + sidecars in the set folder, and the launcher there
    imports the COPY (a set runs exactly the code deployed into it); old `scripts.face_rhythm_factors` names resolve."""
    import subprocess
    import sys

    from scripts import face_rhythm_factors as launcher
    assert launcher.cmd_project is f.cmd_project and launcher.npz_memmap is f.npz_memmap
    assert launcher.POINTS_CHUNK == f.config.POINTS_CHUNK and launcher._scaled is f.cp._scaled
    sd, *_ = _bundle(tmp_path)
    written = f.deploy_code(sd)
    assert (sd / "fr_factors.py") in written and (sd / f.LIB / "cp.py").exists()
    assert all((sd / f"{m}.py").exists() for m in f.SIDECAR_MODULES)
    code = (f"import sys; sys.path.insert(0, r'{sd}'); import fr_factors; "
            f"print(fr_factors._pkg.__file__); fr_factors._pkg.cli.main(['build', '--dir', r'{sd}'])")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert str(sd / f.LIB) in r.stdout.splitlines()[0]
    assert (sd / "fit_tensor.npy").exists() and (sd / "points_pca.npz").exists()
