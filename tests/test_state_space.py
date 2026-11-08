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
    monkeypatch.setattr(S, "ITI", str(tmp_path))
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
