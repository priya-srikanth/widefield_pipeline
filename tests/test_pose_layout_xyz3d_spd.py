"""COMBINED_ENCODING_POSE=xyz3d_spd (Priya 11-25): frame A's full 3-D layout with the jaw and nose entering as their 3-D
SPEED only (no jaw / nose position columns; tongue unchanged). Checks: the layout widths; the frozen variables drop
jaw_dir / nose_dir at import (subprocess: the switch is read at import like the other design switches); split_blocks
widths; nose_speed derived from an xyz3d behaviour equals the load-time formula; run tags / folder suffixes differ from
xyz3d; xyz3d itself is unchanged (pinned widths + tag); the G4 runner's lock release handles SIGTERM."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ENV3D = {"COMBINED_ENCODING_TONGUE": "3d", "COMBINED_ENCODING_JAW": "3d", "COMBINED_ENCODING_NOSE": "3d"}


def _clear(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("COMBINED_ENCODING_", "FROZEN_")):
            monkeypatch.delenv(k)


def test_layout_widths_and_xyz3d_pinned(monkeypatch):
    from wfield_local import combined_encoding as ce
    _clear(monkeypatch)
    for k, v in ENV3D.items():
        monkeypatch.setenv(k, v)
    spd = ce.POSE_LAYOUTS["xyz3d_spd"]
    assert spd["jaw"] == ("jaw_speed",) and spd["nose"] == ("nose_speed",)
    assert spd["tongue"] == ce.POSE_LAYOUTS["xyz3d"]["tongue"] and spd["tongue_ps"] == ce.POSE_LAYOUTS["xyz3d"]["tongue_ps"]
    assert spd["dir_n"] == {"tongue": 2, "jaw": 0, "nose": 0} and spd["dir_kin"] == ("tongue_ml", "tongue_ap")
    assert spd["jaw_nd"] == ("jaw_speed",) and spd["nose_nd"] == ("nose_speed",)
    # xyz3d pinned (frame A's layout, unchanged)
    x = ce.POSE_LAYOUTS["xyz3d"]
    assert x["jaw"] == ("jaw_ml", "jaw_ap", "jaw_dv", "jaw_speed") and x["nose"] == ("nose_ml", "nose_ap", "nose_dv")
    assert x["dir_n"] == {"tongue": 2, "jaw": 2, "nose": 2}
    monkeypatch.setenv("COMBINED_ENCODING_POSE", "xyz3d_spd")
    assert ce.pose_layout() == "xyz3d_spd"
    monkeypatch.setenv("COMBINED_ENCODING_NOSE", "cam4")
    with pytest.raises(ValueError):
        ce.pose_layout()
    monkeypatch.setenv("COMBINED_ENCODING_NOSE", "3d")
    monkeypatch.setenv("COMBINED_ENCODING_POSE", "xyz3d_other")
    with pytest.raises(ValueError):
        ce.pose_layout()


def test_run_tag_and_folder_suffix(monkeypatch):
    from wfield_local import combined_encoding as ce
    from wfield_local import frozen_encoder as fe
    _clear(monkeypatch)
    for k, v in ENV3D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("COMBINED_ENCODING_POSE", "xyz3d")
    t3 = fe._run_tag()
    assert "|xyz3d|" in t3 + "|" and "xyz3d_spd" not in t3
    monkeypatch.setenv("COMBINED_ENCODING_POSE", "xyz3d_spd")
    ts = fe._run_tag()
    assert "|xyz3d_spd" in ts and ts.replace("|xyz3d_spd", "|xyz3d") == t3
    assert ce.pose_layout() == "xyz3d_spd"


def test_nose_speed_derived_equals_load_formula():
    from wfield_local import combined_encoding as ce
    rng = np.random.default_rng(0)
    n = 40
    t = np.arange(n) / 250.0
    ml, ap, dv = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    ml[7] = np.nan
    analog = {"nose_ml": (t, ml), "nose_ap": (t, ap), "nose_dv": (t, dv)}
    got = ce.derived_analog(analog, "nose_speed")
    assert got is not None and got[0] is t
    want = np.linalg.norm(np.column_stack([np.gradient(v) for v in (ml, ap, dv)]), axis=1) * 250.0   # jaw_3d_xyz's
    np.testing.assert_array_equal(np.isnan(got[1]), np.isnan(want))
    np.testing.assert_allclose(np.nan_to_num(got[1]), np.nan_to_num(want), rtol=0, atol=0)
    assert ce.derived_analog({"nose_ml": (t, ml)}, "nose_speed") is None
    assert ce.derived_analog(analog, "jaw_speed") is None
    assert len(ce.speed_3d([1.0], [1.0], [1.0])) == 1 and np.isnan(ce.speed_3d([1.0], [1.0], [1.0])).all()


def test_pose_block_derives_nose_speed_for_spd(monkeypatch):
    """A behaviour recorded under xyz3d (nose_ml / ap / dv, no nose_speed) serves the speed-only nose block."""
    from wfield_local import combined_encoding as ce
    _clear(monkeypatch)
    for k, v in ENV3D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("COMBINED_ENCODING_POSE", "xyz3d_spd")
    n = 300
    t = np.arange(n) / 250.0
    rng = np.random.default_rng(1)
    analog = {f"nose_{a}": (t, rng.normal(size=n)) for a in ("ml", "ap", "dv")}
    analog["jaw_speed"] = (t, np.abs(rng.normal(size=n)))

    class B:
        pass
    b = B()
    b.analog = analog
    ft = np.arange(0.0, 1.0, 1 / 30)
    out = {}
    ce._pose_block(out, b, ft, "nose", ce.POSE_LAYOUTS["xyz3d_spd"])
    ce._pose_block(out, b, ft, "jaw", ce.POSE_LAYOUTS["xyz3d_spd"])
    nl = len(ce.CLAGS)
    assert out["nose"].shape == (len(ft), nl) and out["jaw"].shape == (len(ft), nl)
    sp = ce.speed_3d(*(analog[f"nose_{a}"][1] for a in ("ml", "ap", "dv")))
    ref = {}
    ce._pose_block(ref, type("B2", (), {"analog": {"nose_speed": (t, sp)}})(), ft, "nose", {"nose": ("nose_speed",)})
    np.testing.assert_array_equal(out["nose"], ref["nose"])
    with pytest.raises(KeyError):
        ce._pose_block({}, type("B3", (), {"analog": {}})(), ft, "nose", ce.POSE_LAYOUTS["xyz3d_spd"])


_SUB = r"""
import os, sys, numpy as np
os.environ.update({"COMBINED_ENCODING_TONGUE": "3d", "COMBINED_ENCODING_JAW": "3d", "COMBINED_ENCODING_NOSE": "3d",
                   "COMBINED_ENCODING_POSE": "xyz3d_spd", "FROZEN_LICKS_ALL": os.environ.get("_LA", "none")})
from wfield_local import frozen_encoder as fe
from wfield_local import combined_encoding as ce
assert fe.POSE_SPD
assert "jaw_dir" not in fe.SPLIT_VARS and "nose_dir" not in fe.SPLIT_VARS
assert not any(v.startswith(("jaw_dir", "nose_dir")) for v in fe.FGROUP), sorted(fe.FGROUP)
assert fe.FGROUP["jaw_mag:dev"] == "lick" and fe.FGROUP["nose_mag"] == "other"
assert fe.FPEN["jaw_mag:dev"] == "orofacial" and fe.FPEN["nose_mag"] == "uninstructed"
assert "|xyz3d_spd" in fe._run_tag()
nl, n = len(ce.CLAGS), 5
raw = {v: np.zeros((n, nl)) for v in ("cue", "arrival", "pos_enl", "pos_cue", "running", "exec_angle")}
raw["licks"] = np.zeros((n, 2 * nl))
raw["tongue"] = np.tile(np.repeat(np.arange(5.0), nl), (n, 1))
raw["jaw"] = np.tile(np.repeat(np.arange(1.0), nl), (n, 1)) + 7
raw["nose"] = np.tile(np.repeat(np.arange(1.0), nl), (n, 1)) + 9
if fe.SPONT:
    for v in fe.SPONT_VARS:
        raw[v] = np.zeros((n, nl))
out = fe.split_blocks(raw)
assert "jaw_dir" not in out and "nose_dir" not in out
assert out["jaw_mag"].shape == (n, nl) and set(out["jaw_mag"][0]) == {7.0}
assert out["nose_mag"].shape == (n, nl) and set(out["nose_mag"][0]) == {9.0}
assert set(out["tongue_dir"][0]) == {0, 1} and set(out["tongue_mag"][0]) == {2, 3, 4}
assert set(out) == {v.split(":")[0] for v in fe.FGROUP} - {"push"}
raw["jaw"] = np.zeros((n, 4 * nl))
try:
    fe.split_blocks(raw)
except ValueError:
    pass
else:
    raise AssertionError("a 4-trace jaw block must be refused under xyz3d_spd")
# the variant list has no novar_jaw_dir / novar_nose_dir, and the FR / posinfo super-groups still build
assert "novar_jaw_dir" not in fe.VARIANTS and "novar_jaw_mag" in fe.VARIANTS
print("SPD-OK", len(fe.FGROUP))
"""


@pytest.mark.parametrize("licks_all", ["none", "split"])
def test_frozen_variables_drop_jaw_nose_dir_at_import(licks_all):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COMBINED_ENCODING_", "FROZEN_"))}
    env.update({"PYTHONPATH": str(REPO), "_LA": licks_all, "MKL_THREADING_LAYER": "SEQUENTIAL"})
    p = subprocess.run([sys.executable, "-c", _SUB], env=env, capture_output=True, text=True, cwd=str(REPO))
    assert p.returncode == 0, p.stderr[-3000:]
    assert "SPD-OK" in p.stdout


def test_split_blocks_refuses_layout_changed_after_import(monkeypatch):
    """The frozen variables are fixed at import: an xyz3d_spd layout set afterwards is refused, never half-applied."""
    from wfield_local import combined_encoding as ce
    from wfield_local import frozen_encoder as fe
    assert not fe.POSE_SPD
    _clear(monkeypatch)
    for k, v in ENV3D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("COMBINED_ENCODING_POSE", "xyz3d_spd")
    nl, n = len(ce.CLAGS), 3
    raw = {v: np.zeros((n, nl)) for v in ("cue", "arrival", "pos_enl", "pos_cue", "running", "exec_angle")}
    raw["licks"] = np.zeros((n, 2 * nl))
    raw["tongue"], raw["jaw"], raw["nose"] = np.zeros((n, 5 * nl)), np.zeros((n, nl)), np.zeros((n, nl))
    with pytest.raises(ValueError, match="changed after import"):
        fe.split_blocks(raw)


def test_o2_frozen_spd_frame_definition():
    import importlib.util
    spec = importlib.util.spec_from_file_location("o2f_spd", REPO / "scripts" / "o2_frozen.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    f = m.FRAMES["spont_frA_spd"]
    a = m.FRAMES["spont_frA"]
    assert f["tag"] == "gate-attempted_v7g3_t3d_spont_frA_spd" and f["jobs"] == a["jobs"] and f["args"] == a["args"]
    assert f["env"] == {**a["env"], "COMBINED_ENCODING_POSE": "xyz3d_spd"}
    assert m.V36_RUN_MODE["spont_frA_spd"] == m.V36_RUN_MODE["spont_frA"]
    os.environ["COMBINED_ENCODING_POSE"] = "xyz3d_spd"
    try:
        assert m.expected_run_tag("spont_frA_spd") == m.expected_run_tag("spont_frA").replace("|xyz3d|", "|xyz3d_spd|")
        assert "|jaw3d|xyz3d_spd|nose3d" in m.expected_run_tag("spont_frA_spd")
    finally:
        os.environ.pop("COMBINED_ENCODING_POSE", None)
    assert "release_lock_on_exit" in (REPO / "scripts" / "o2_encoding.py").read_text(encoding="utf-8")


def test_release_lock_on_exit_handles_sigterm(tmp_path):
    import signal

    from wfield_local import o2_bundle
    lock = tmp_path / "x.lock"
    lock.write_text("1")
    h = o2_bundle.release_lock_on_exit(lock)
    assert signal.getsignal(signal.SIGTERM) is h
    with pytest.raises(SystemExit) as ex:
        h(int(signal.SIGTERM), None)
    assert ex.value.code == 128 + int(signal.SIGTERM) and not lock.exists()
    h(None, None)                                     # atexit path: idempotent
    assert not lock.exists()


_SUB_LAG = r"""
import os, numpy as np
os.environ.update({"COMBINED_ENCODING_TONGUE": "3d", "COMBINED_ENCODING_JAW": "3d", "COMBINED_ENCODING_NOSE": "3d",
                   "COMBINED_ENCODING_POSE": "xyz3d", "FROZEN_LICKS_ALL": "split", "FROZEN_FR": "1",
                   "COMBINED_ENCODING_FR": "master_pre_v1", "COMBINED_ENCODING_FR_GROUPS": "master_v1:A",
                   "FROZEN_FR_LAGS": "-0.3,-0.2,-0.1,0.1,0.2,0.3"})
from wfield_local import frozen_encoder as fe
from wfield_local import movement_encoding as ME
assert fe.FR_LAGS == (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3) and "|frlags=-0.3,-0.2,-0.1,0.1,0.2,0.3" in fe._run_tag()
ft = np.arange(0, 20, 1 / 30.0)
rng = np.random.default_rng(0)
cols = {v: rng.normal(size=(len(ft), 1)) for v in fe.FR_VARS}
out = fe.fr_lag_columns(cols, ft)
fs = ME.frame_fs(ft)
for v in fe.FR_VARS:
    assert out[v] is cols[v]
    assert out[f"{v}:facelead"].shape == (len(ft), 3) and out[f"{v}:cortexlead"].shape == (len(ft), 3)
    np.testing.assert_array_equal(out[f"{v}:facelead"][:, 0], ME._shift(cols[v][:, 0], int(round(0.1 * fs))))
    np.testing.assert_array_equal(out[f"{v}:cortexlead"][:, 2], ME._shift(cols[v][:, 0], int(round(-0.1 * fs))))
assert all(f"{v}:facelead" in fe.FGROUP and fe.FGROUP[f"{v}:facelead"] == fe.FGROUP[v] for v in fe.FR_VARS)
assert "no_frfacelead" in fe.VARIANTS and "no_frcortexlead" in fe.VARIANTS
print("LAG-OK")
"""


def test_fr_lag_columns_outside_the_loader():
    """FROZEN_FR_LAGS copies are built by `fr_lag_columns` from the lag-0 column (so a replayed `_fr_columns` recorded
    without lags serves a lagged frame); without lags the columns pass through unchanged."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COMBINED_ENCODING_", "FROZEN_"))}
    env.update({"PYTHONPATH": str(REPO), "MKL_THREADING_LAYER": "SEQUENTIAL"})
    p = subprocess.run([sys.executable, "-c", _SUB_LAG], env=env, capture_output=True, text=True, cwd=str(REPO))
    assert p.returncode == 0, p.stderr[-3000:]
    assert "LAG-OK" in p.stdout
    from wfield_local import frozen_encoder as fe
    if not fe.FR_LAGS:
        cols = {"a": np.zeros((3, 1))}
        assert fe.fr_lag_columns(cols, np.arange(3.0)) is cols
    import inspect
    src = inspect.getsource(fe._fr_columns)
    # merged 11-26 onto main's design_raw: the lag copies are built from the lag-0 FR columns already in raw
    assert "FR_LAGS" not in src and "fr_lag_columns({v: raw[v] for v in FR_VARS}" in inspect.getsource(fe.session_design)


def test_o2_frozen_frlag_frame_definition():
    import importlib.util
    spec = importlib.util.spec_from_file_location("o2f_lag", REPO / "scripts" / "o2_frozen.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    f, a = m.FRAMES["spont_frA_frlag"], m.FRAMES["spont_frA"]
    assert f["env"] == {**a["env"], "FROZEN_FR_LAGS": "-0.3,-0.2,-0.1,0.1,0.2,0.3"} and f["jobs"] == a["jobs"]
    assert f["args"] == ["--variants", "joint", "no_fr", "no_fr_dir", "no_frfacelead", "no_frcortexlead", "no_movement"]
    assert m.V36_RUN_MODE["spont_frA_frlag"] == m.V36_RUN_MODE["spont_frA"].replace(
        "|licksall-split", "|frlags=-0.3,-0.2,-0.1,0.1,0.2,0.3|licksall-split")
