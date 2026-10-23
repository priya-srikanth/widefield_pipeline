"""wfield_local.face_rhythm_multiview + the multi-view path of scripts/face_rhythm_factors.py: views stack along the point
axis (trial rows by index, session bins by nearest DAQ time), view balance gives every view equal energy, and a planted
factor spanning two cameras is recovered and projected end to end. Small synthetic bundles (seconds)."""
import json

import numpy as np
import pytest

from scripts import face_rhythm_factors as f
from wfield_local import face_rhythm_multiview as fmv

F, R = 10, 3
PV = {"cam4": 30, "cam1": 18}


def test_view_key_and_session_map():
    assert fmv.view_key("PS92_20260812_cam4_v2", "cam4", "cam1") == "PS92_20260812_cam1_v2"
    with pytest.raises(ValueError):
        fmv.view_key("PS92_20260812_cam1", "cam4", "cam2")
    t_ref = np.arange(10) / 5.0
    t_view = np.arange(12) / 5.0 + 0.03                      # 30 ms offset, two extra bins
    m = fmv.session_row_map(t_ref, t_view)
    assert np.array_equal(m, np.arange(10))
    m = fmv.session_row_map(t_ref, np.r_[t_view[:4], t_view[6:]])     # a gap: bins 4, 5 missing in the view
    assert m[4] == -1 and m[5] == -1 and m[6] == 4


def test_viewstack_indexing_reshape_and_balance():
    rng = np.random.default_rng(0)
    a = rng.random((4, 5, 2, 3, F)).astype(np.float16)
    b = rng.random((4, 5, 2, 2, F)).astype(np.float16)
    V = fmv.ViewStack([a, b], None, [1.0, 2.0])
    assert V.shape == (4, 5, 2, 5, F) and V.slices() == [slice(0, 3), slice(3, 5)]
    x = V[1]
    assert np.allclose(x[:, :, :3], a[1].astype(np.float32)) and np.allclose(x[:, :, 3:], b[1].astype(np.float32) / 2)
    W = V.reshape(20, 2, 5, F)
    assert W.shape == (20, 2, 5, F) and np.allclose(W[5:10], V[1])
    s = fmv.ViewStack([a[0], b[0]], [None, np.array([0, -1, 2, 3, 4])])
    y = s[0:3]
    assert np.isnan(y[1, :, 3:]).all() and np.isfinite(y[1, :, :3]).all()
    X = np.concatenate([np.full((50, 2, 3, F), 2.0), np.full((50, 2, 2, F), 0.5)], axis=2)
    d = fmv.balance_divisors(X, [3, 2])
    e = [float(((X[:, :, sl] / dv) ** 2).sum()) for sl, dv in zip(fmv.view_slices([3, 2]), d)]
    assert np.isclose(e[0], e[1])                                     # equal energy per view
    with pytest.raises(ValueError, match="trial grid"):
        fmv.check_trials_match(np.array([1.0, 2.0]), np.array([1.0, 2.5]), "x")


def _two_view_bundle(tmp_path):
    """cam4 bundle (the set's) + a cam1 bundle; ONE planted factor set over both views' points."""
    rng = np.random.default_rng(1)
    root = tmp_path / "o2"
    od = {c: root / f"fr_{c}_x" / "out" for c in PV}
    for d in od.values():
        d.mkdir(parents=True)
    sd = root / "fr_cam4_x" / "factors" / "mv"
    sd.mkdir(parents=True)
    Ptot = sum(PV.values())
    A = rng.random((2 * Ptot, R)) ** 3                                   # xy-major over the STACKED points
    B = np.full((F, R), 0.05)
    B[[2, 5, 8], range(R)] = 1.0
    A4 = A.reshape(2, Ptot, R)
    plan = {"set": "mv", "params": {"inits": 2, "n_iter": 150, "trial_ds": 2}, "fit": {}, "session_rows": {},
            "tasks": [], "views": {"ref": "cam4", "cams": [{"cam": "cam1", "bundle": "fr_cam1_x"}]}}
    for a, sc in {"PS92": 1.0, "PS93": 0.7}.items():
        for d, ep in (("20260810", "pre"), ("20260820", "acute")):
            ntr, n5 = 6, 300
            cues = 10.0 + 9.0 * np.arange(ntr)
            Ct, Cs = rng.random((ntr * 151, R)) ** 2, rng.random((n5, R)) ** 2
            o = 0
            for c, p in PV.items():
                Av = A4[:, o:o + p].reshape(2 * p, R)
                vs = 1.0 if c == "cam4" else 3.0                     # cam1's run on a different scale
                mk = lambda C, shp, Av=Av, vs=vs, p=p, sc=sc: (vs * sc * np.einsum("tr,pr,fr->tpf", C, Av, B)
                                                        + 0.005 * rng.random((len(C), 2 * p, F))
                                                        ).reshape(shp + (2, p, F)).astype(np.float16)
                key = f"{a}_{d}_{c}"
                np.savez(od[c] / f"{key}_trials.npz", spec=mk(Ct, (ntr, 151)),
                         t_ms=np.arange(-2000.0, 4001, 40.0, dtype=np.float32), trial_id=np.arange(ntr),
                         position=np.array(["far_L", "far_R"] * (ntr // 2)), cue_daq_s=cues,
                         freqs=np.geomspace(1, 60, F), home=rng.random((p, 2)) * 100)
                np.savez(od[c] / f"{key}_session.npz", spec=mk(Cs, (n5,)),
                         t_daq_s=np.arange(n5) / 5.0 + 0.1 + (0.02 if c == "cam1" else 0.0), rate_hz=np.float32(5))
                o += p
            k4 = f"{a}_{d}_cam4"
            np.savez(od["cam4"] / f"{k4}_points.npz", disp=np.zeros((1500, PV["cam4"], 2), np.float16),
                     viol_frac=np.zeros(1500, np.float16), t_daq_s=np.arange(1500) / 25.0,
                     home=np.zeros((PV["cam4"], 2)))
            plan["tasks"].append({"key": k4, "animal": a, "date": d, "epoch": ep})
            plan["session_rows"][k4] = {"trials": {k4: [0, 1, 2, 3]}, "between": {k4: list(range(150, 300, 3))}}
        k = f"{a}_20260810_cam4"
        plan["fit"][a] = {"sessions": [k], "trials": {k: [0, 1, 2, 3, 4, 5]}, "between": {k: list(range(150, 300, 3))}}
    (sd / "plan.json").write_text(json.dumps(plan))
    return sd, A, B


def test_multiview_pipeline_end_to_end(tmp_path):
    sd, A, B = _two_view_bundle(tmp_path)
    f.cmd_build(sd)
    vs = json.loads((sd / "view_scales.json").read_text())
    assert vs["PS92"][1] > 2 * vs["PS92"][0]                           # cam1's larger run scale is divided out
    assert np.load(sd / "fit_tensor.npy").shape[1] == 2 * sum(PV.values())
    f.cmd_fit(sd, R, loao=False)
    f.cmd_select(sd, R)
    M = np.load(sd / "master.npz")
    assert list(M["view_cams"]) == ["cam4", "cam1"] and list(M["view_P"]) == [PV["cam4"], PV["cam1"]]
    assert M["home"].shape == (sum(PV.values()), 2)
    # recovered up to each view's balance scale: compare per view (cosines are scale-free within a view)
    Ar = M["spatial"].reshape(-1, R)
    for sl in fmv.view_slices([PV["cam4"], PV["cam1"]]):
        idx = np.r_[np.arange(sl.start, sl.stop), sum(PV.values()) + np.arange(sl.start, sl.stop)]
        assert f.similarity([A[idx], B], [Ar[idx], M["spectral"]]) > 0.95
    i = next(j for j, t in enumerate(json.loads((sd / "plan.json").read_text())["tasks"], 1) if t["epoch"] == "acute")
    f.cmd_project(sd, i)
    p = np.load(sd / "proj" / "PS92_20260820_cam4.npz")
    assert p["trials"].shape == (6, 151, R) and p["session"].shape == (300, R)
    assert np.nanmedian(p["session_err"]) < 0.2 and np.isfinite(p["trials_D"]).all()
    assert p["power_response"].shape[-2] == sum(PV.values())         # window power over every view's points
    f.cmd_report(sd)                                                   # multi-view gallery: reference view + shares
