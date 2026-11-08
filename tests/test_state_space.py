"""State-space arm (exploratory, branch analysis/state-space-recovery-20261107): pin the pieces every readout is built
on, on synthetic data -- the default cue binning, the dimensionality transforms, the ONE pre -> chronic frame
(`state_space_recovery_trials.frames` / `xy`) and the ONE session loader (`state_space_sim.load_sessions`).
Handoff cleanup item 8 (docs/status/STATUS_2026-11-07_STATE_SPACE_HANDOFF.md)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def mods():
    """scripts/ on sys.path; the state-space modules imported fresh under the default env, then dropped again."""
    env, path, before = dict(os.environ), list(sys.path), set(sys.modules)
    for k in ("WF_SS_ALIGN", "WF_SS_K", "WF_SS_XFORM"):
        os.environ.pop(k, None)
    sys.path.insert(0, str(SCRIPTS))
    import state_space_recovery_trials as R
    import state_space_sim as S
    yield S, R
    for m in set(sys.modules) - before:
        f = getattr(sys.modules.get(m), "__file__", None) or ""
        if f and Path(f).resolve().parent == SCRIPTS:
            del sys.modules[m]
    sys.path[:] = path
    os.environ.clear()
    os.environ.update(env)


def test_cue_binning_is_the_original(mods):
    """Default (cue) binning = 0.1 s means over -0.5..2.0 s of first_cue, the binning of every result before 11-07."""
    S, _ = mods
    rng = np.random.default_rng(0)
    z = {"first_cue": rng.normal(size=(4, len(S.LAGS), 3))}
    got = S._bin(z, [0, 1, 2], "cue")
    want = np.stack([z["first_cue"][:, (S.LAGS >= a - 1e-9) & (S.LAGS < b - 1e-9)].mean(1) for a, b in S.BINS], 1)
    assert got.shape == (4, 25, 3)
    np.testing.assert_allclose(got, want)


def test_transforms(mods, monkeypatch):
    S, _ = mods
    rng = np.random.default_rng(1)
    xb = rng.normal(size=(30, 5, 4))
    pos = np.array(S.POS * 5)
    w = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2)
    monkeypatch.setitem(S._XSTATS, "PSX", (np.array([1.0, 2.0, 4.0, 8.0]), w))
    monkeypatch.setattr(S, "XFORM", "gsr")
    assert np.abs(S._xform(xb, pos, "PSX") @ w).max() < 1e-12            # the cortex-wide direction is gone
    monkeypatch.setattr(S, "XFORM", "z")
    np.testing.assert_allclose(S._xform(xb, pos, "PSX"), xb / [1.0, 2.0, 4.0, 8.0])
    monkeypatch.setattr(S, "XFORM", "spout")
    out = S._xform(xb, pos, "PSX")
    np.testing.assert_allclose(np.mean([out[pos == p].mean(0) for p in S.POS], 0), 0, atol=1e-12)


def _sessions(S, n_area=12, n_bin=6, n_trial=6):
    """Pre / acute / chronic sessions whose trials all sit exactly on their epoch's template."""
    rng = np.random.default_rng(2)
    T0 = rng.normal(size=(6, n_bin, n_area))
    D, A = rng.normal(size=T0.shape), rng.normal(size=T0.shape)
    pos = np.repeat(np.array(S.POS), n_trial)
    out = []
    for i, (ep, T) in enumerate([("pre", T0), ("pre", T0), ("acute", T0 + A), ("chronic", T0 + D)]):
        Y = np.stack([T[S.POS.index(p)] for p in pos])
        out.append((f"S{i}", ep, i, Y, pos))
    return out, T0, D, A


def test_frame_axes(mods):
    S, R = mods
    sess, T0, D, A = _sessions(S)
    P, mu, fr = R.frames(sess)
    assert list(fr) == S.POS
    for si, p in enumerate(S.POS):
        f = fr[p]
        x_of = {}
        for _lab, ep, _d, Y, pos in sess:
            q = ((Y[pos == p] - mu) @ P).reshape((pos == p).sum(), -1) - f["zp"]
            x, y = R.xy(q, f)
            x_of[ep] = (x.mean(), y.mean())
        np.testing.assert_allclose(x_of["pre"], (0, 0), atol=1e-9)        # pre at the origin
        np.testing.assert_allclose(x_of["chronic"], (1, 0), atol=1e-9)    # chronic at (1, 0)
        assert x_of["acute"][1] > 0                                        # acute departs along +y by construction
        assert 0 < f["orth"] <= 1
        np.testing.assert_allclose(f["Tpre"], T0[si])
        assert abs(f["u"] @ f["v"]) < 1e-9


def test_load_sessions_order_and_hook(mods, monkeypatch, tmp_path):
    S, _ = mods
    labs = ["PS92_0820", "PS92_0806", "PS93_0806", "PS92_0812"]
    for lab in labs:
        (tmp_path / f"{lab}.npz").write_bytes(b"")
    monkeypatch.setattr(S, "SRC", str(tmp_path))
    pos = np.array(S.POS)
    monkeypatch.setattr(S, "load", lambda f: (np.zeros((6, 2, 3)), pos, ["a", "b", "c"], "pre", Path(f).stem[:4]))
    data, cols = S.load_sessions(lambda lab, Y, p: (Y, p, lab))
    assert cols == ["a", "b", "c"]
    assert list(data) == ["PS92", "PS93"]
    assert [s[0] for s in data["PS92"]] == ["PS92_0806", "PS92_0812", "PS92_0820"]     # by day
    assert all(s[5] == s[0] for v in data.values() for s in v)                           # the hook's extra column


def test_paths_match_the_rebuild_routing(mods):
    """S.ENCODING / S.XTAG / S.WTAG / S.ITI = what the repo's routing gives under the render profile (computed in a
    SUBPROCESS so the profile never enters this process)."""
    import json
    import subprocess
    S, _ = mods
    code = ("import json, os\n"
            "from wfield_local import render_profile as rp\n"
            "env = dict(os.environ); rp.apply('%s', environ=env); os.environ.clear(); os.environ.update(env)\n"
            "from wfield_local import orofacial_outputs as oo, run_mode\n"
            "print(json.dumps({'enc': oo.encoding_root().as_posix(), 'w': run_mode.tag(),\n"
            "                  'x': run_mode.tag(include=('hemo', 'lnmf', 'drift', 'pre', 'projall'))}))\n") % S.PROFILE
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    got = json.loads(out.strip().splitlines()[-1])
    assert got["w"] == S.WTAG and got["x"] == S.XTAG
    assert Path(got["enc"]).as_posix().lower() == S.ENCODING.as_posix().lower()
    assert Path(S.ITI).parent.name == S.XTAG and S.ITI_W.name == S.WTAG


def test_evoked_and_hemi(mods):
    """S.evoked = response-window mean minus baseline-window mean (BAL_BINS of the alignment); S.hemi = right / left means."""
    S, _ = mods
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(5, 25, 4))
    base, resp = S.BAL_BINS["cue"]
    np.testing.assert_allclose(S.evoked(Y), Y[:, resp].mean(1) - Y[:, base].mean(1))
    np.testing.assert_allclose(S.evoked(Y[0]), Y[0, resp].mean(0) - Y[0, base].mean(0))
    h = S.hemi(Y[0, 0], ["SSp-m_right", "SSp-m_left", "VISp_right", "VISp_left"], ("SSp-m",))
    assert h == {"contra": Y[0, 0, 0], "ipsi": Y[0, 0, 1]}


def test_lnmf_basis(mods, monkeypatch, tmp_path):
    """WF_SS_BASIS=lnmf: component names from the basis regions, June kept apart, per-animal cols, _lnmf tag."""
    S, _ = mods
    monkeypatch.setattr(S, "BASIS", "lnmf")
    monkeypatch.setattr(S, "LNMF_MASK", "none")                        # the loader itself; the mask: its own test
    monkeypatch.setattr(S, "SRC", str(tmp_path / "sessions"))
    monkeypatch.setattr(S, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(S, "_NAMES", {3: "MOp_left", -3: "MOp_right", 17: "SSp-m_left", -17: "SSp-m_right"})
    assert S.tag().endswith("_lnmf")
    (tmp_path / "sessions").mkdir()
    rng = np.random.default_rng(0)
    for lab, reg in (("PS92_0606", [3, 3, -3, 17, -17]), ("PS92_0806", [3, 3, -3, 17, -17]), ("PS93_0806", [3, -17])):
        n = 12
        np.savez_compressed(tmp_path / "sessions" / f"{lab}.npz",
                            first_cue=rng.normal(size=(n, len(S.LAGS), len(reg))).astype(np.float32),
                            first_peth=rng.normal(size=(n, len(S.LAGS_P), len(reg))).astype(np.float32),
                            first_pos=np.array(S.POS * 2), first_rt=np.full(n, 0.3), epoch="pre",
                            animal=lab[:4], regions=np.array(reg))
    files = sorted(str(f) for f in (tmp_path / "sessions").glob("*.npz"))
    S.prefill(files[1:])                        # 2 files -> spawned workers: must write into THIS cache, not the module's
    assert all((tmp_path / "cache" / a / f"{lab}.npz").exists() for a in S.SEGMENTS for lab in ("PS92_0806", "PS93_0806"))
    data, cols = S.load_sessions()
    assert cols == {"PS92": ["MOp#1_left", "MOp#2_left", "MOp#1_right", "SSp-m#1_left", "SSp-m#1_right"],
                    "PS93": ["MOp#1_left", "SSp-m#1_right"]}
    assert [s[0] for s in data["PS92"]] == ["PS92_0806"]                       # June left out by default
    data, _ = S.load_sessions(june=True)
    assert [(s[0], s[1]) for s in data["PS92"]] == [("PS92_0606", "june"), ("PS92_0806", "pre")]
    assert (tmp_path / "cache" / S.ALIGN / "PS92_0806.npz").exists()            # binned once
    v = np.array([1.0, 3.0, 10.0, 100.0, 1000.0])
    h = S.hemi(v, S.cols_of(cols, "PS92"), ("SSp-m",))
    assert h == {"contra": 1000.0, "ipsi": 100.0}                               # '#k' stripped for the parent area
    assert S.hemi(v, cols["PS92"])["ipsi"] == pytest.approx((1 + 3 + 100) / 3)


def test_align_maps_recovers_a_shared_code(mods):
    """Loops in a per-animal basis (LocaNMF): one shared pre code (6 spouts x bins x 8 latents) seen through a different
    random mixing into 20 'components' per animal (+ trial noise) -> the pre-only Procrustes alignment lines the animals
    up (leave-one-animal-out r ~ 1), the aligned post epoch reproduces the planted post change, and the PCA plane runs."""
    S, R = mods
    import state_space_trajectories as TJ
    rng = np.random.default_rng(1)
    nb, d = len(S.BINS), 8
    code = rng.normal(size=(6, nb, d))
    post = code * 1.5                                                  # a pure gain change, shared by all animals
    data = {}
    for k, an in enumerate(["A1", "A2", "A3", "A4"]):
        mix = np.linalg.qr(rng.normal(size=(20, d)))[0].T              # latent -> this animal's 20 components
        sess = []
        for ep, c, n in (("pre", code, 3), ("acute", code, 2), ("subacute", code, 2), ("chronic", post, 2)):
            for j in range(n):
                pos = np.repeat(S.POS, 20)
                Y = c[[S.POS.index(p) for p in pos]] @ mix + 0.05 * rng.normal(size=(len(pos), nb, 20)) + k
                sess.append((f"{an}_{ep}{j}", ep, j, Y, pos))
        data[an] = sess
    maps, loo = TJ.align_maps(data)
    assert min(loo.values()) > 0.98
    a = {an: (R.template([s for s in sess if s[1] == "chronic"]) - R.template([s for s in sess if s[1] == "pre"]))
         @ maps[an] for an, sess in data.items()}
    for an in a:                                                       # aligned post changes agree across animals
        assert np.corrcoef(a[an].ravel(), a["A1"].ravel())[0, 1] > 0.98
    proj, pooled, ve, boot, loo2 = TJ.loops(data, align=True)
    assert pooled["chronic"].shape == (6, nb, 2) and ve > 0.3


def test_svd_basis_switches(mods, monkeypatch):
    """WF_SS_BASIS=svd: shared components -> cross-animal scripts allowed only where named areas are not needed; the
    hemisphere split reads area means through the basis weights; outputs tagged _svd."""
    S, _ = mods
    monkeypatch.setattr(S, "BASIS", "svd")
    assert S.tag().endswith("_svd")
    W = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])                  # 2 components x areas MOp_left, MOp_right, SSp_left
    monkeypatch.setattr(S, "_SVDW", {"W": W, "names": ["MOp_left", "MOp_right", "SSp_left"]})
    v = np.array([[1.0, 3.0]])                                          # 1 trial x 2 components
    h = S.hemi(v, ["svd#1_both", "svd#2_both"])
    assert h["contra"][0] == 3.0 and h["ipsi"][0] == 1.5                 # right: MOp 3; left: mean(MOp 1, SSp 2)
    assert S.hemi(v, [], sel={"SSp"})["ipsi"][0] == 2.0 and np.isnan(S.hemi(v, [], sel={"SSp"})["contra"])
    S.allen_only("x", shared_ok=True)
    with pytest.raises(SystemExit):
        S.allen_only("x")
    monkeypatch.setattr(S, "BASIS", "lnmf")
    with pytest.raises(SystemExit):
        S.allen_only("x", shared_ok=True)


def test_lnmf_brain_mask_drops_occluded(mods, monkeypatch):
    """LocaNMF on the brain mask: occluded components (glue / bulb) dropped at load, tag _lnmfbm; WF_SS_LNMF_MASK=none
    keeps them (tag _lnmf)."""
    S, _ = mods
    monkeypatch.setattr(S, "BASIS", "lnmf")
    monkeypatch.setattr(S, "_lnmf_keep", lambda an: np.array([True, False, True]))
    xb, cols = S._mask_cols(np.arange(12.0).reshape(2, 2, 3), ["a#1_left", "b#1_left", "c#1_right"], "PS92")
    assert cols == ["a#1_left", "c#1_right"] and xb.shape == (2, 2, 2) and xb[0, 0, 1] == 2.0
    assert S.tag().endswith("_lnmfbm")
    with pytest.raises(ValueError):
        S._mask_cols(np.zeros((1, 1, 2)), ["a", "b"], "PS92")
    monkeypatch.setattr(S, "LNMF_MASK", "none")
    assert S.tag().endswith("_lnmf") and S._mask_cols(xb, cols, "PS92")[1] == cols
