"""The RESTnl option leaves the page's Fig 12.1 decoder (`rest_frozen_decoder`) byte-identical by default.

`_collect` gained two opt-in keywords (`rest_mask`, `period_frames`). With the defaults it must return EXACTLY what
the pre-change source (commit 0e592a20) returns: same arrays, same dtypes, same bytes. The source check runs anywhere;
the data check needs MICROSCOPE and is skipped without it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REV = "0e592a20"
REL = "scripts/rest_migration/rest_frozen_decoder.py"


def _old_module(tmp_path):
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{REV}:{REL}"], capture_output=True, text=True,
                             check=True, encoding="utf-8").stdout
    except Exception:                                                    # noqa: BLE001
        pytest.skip("git history not available")
    p = tmp_path / "rest_frozen_decoder_old.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("rest_frozen_decoder_old", p)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(m)
    return m


def test_defaults_are_off():
    rf = pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")
    sig = inspect.signature(rf._collect)
    assert sig.parameters["rest_mask"].default is None
    assert sig.parameters["period_frames"].default is False


#: Functions allowed to differ from REV. `_collect`: the two opt-in keywords. `main`: since 11-02 its pre-stroke models
#: come from the frozen store by default (`frozen_rest_payload`; tests/test_frozen_default.py pins that route and that
#: the stored fit is the in-memory one). Everything else -- nulls, `_fit_frozen`, `_usable`, `training_pool`, `_plot`,
#: confusion -- stays source-identical.
CHANGED = {"_collect", "main"}


def test_only_collect_changed(tmp_path):
    """Every other function of the module (_plot, nulls, fits, confusion) is source-identical to REV."""
    rf = pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")
    old = _old_module(tmp_path)
    for name, obj in vars(old).items():
        if inspect.isfunction(obj) and obj.__module__ == old.__name__ and name not in CHANGED:
            assert inspect.getsource(obj) == inspect.getsource(getattr(rf, name)), name


def _digest(got):
    h = hashlib.sha256()
    for a in got:
        a = np.asarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("label", ["PS92_0814", "PS94_0821"])
def test_default_collect_bytes_identical(tmp_path, label):
    if not Path("M:/MICROSCOPE").exists():
        pytest.skip("needs MICROSCOPE")
    rf = pytest.importorskip("scripts.rest_migration.rest_frozen_decoder")
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    old = _old_module(tmp_path)
    s = next((x for x in SESSIONS if x["label"] == label), None)
    if s is None:
        pytest.skip(f"{label} not registered")
    b = joint_locanmf.load(label.split("_")[0], sessions=SESSIONS)
    new, why_n = rf._collect(s, b, 4)
    ref, why_o = old._collect(s, b, 4)
    assert why_n == why_o
    assert len(new) == len(ref) == 6
    assert _digest(new) == _digest(ref)
    # and the opt-in frame counts do not perturb the default tuple
    ext, _ = rf._collect(s, b, 4, period_frames=True)
    assert _digest(ext[:6]) == _digest(ref)
