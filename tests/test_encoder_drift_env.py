"""COMBINED_ENCODING_DRIFT_MIN_PERIOD_S override (2026-11-03): unset = the adopted 600 s; set = that period."""
import subprocess
import sys

import numpy as np
import pytest

from wfield_local import combined_encoding as ce


def _read(env_value):
    import os
    env = {k: v for k, v in os.environ.items() if k != ce.DRIFT_ENV}
    if env_value is not None:
        env[ce.DRIFT_ENV] = env_value
    r = subprocess.run([sys.executable, "-c",
                        "from wfield_local import combined_encoding as ce; print(ce.DRIFT_MIN_PERIOD_S)"],
                       env=env, capture_output=True, text=True)
    return r


def test_default_unchanged(monkeypatch):
    monkeypatch.delenv(ce.DRIFT_ENV, raising=False)
    assert ce._drift_min_period_s() == 600.0
    r = _read(None)
    assert r.returncode == 0 and float(r.stdout.strip()) == 600.0


def test_override(monkeypatch):
    monkeypatch.setenv(ce.DRIFT_ENV, "1500")
    assert ce._drift_min_period_s() == 1500.0
    r = _read("1500")
    assert r.returncode == 0 and float(r.stdout.strip()) == 1500.0


@pytest.mark.parametrize("bad", ["10", "nan", "abc"])
def test_bad_values_refused(monkeypatch, bad):
    monkeypatch.setenv(ce.DRIFT_ENV, bad)
    with pytest.raises(ValueError):
        ce._drift_min_period_s()


def test_basis_size_scales():
    t = np.linspace(0.0, 6000.0, 187381)
    assert ce._slow_basis(t, 600.0).shape[1] == 20 + 1
    assert ce._slow_basis(t, 1500.0).shape[1] == 8 + 1
