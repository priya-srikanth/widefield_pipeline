"""o2_partial.same_design: rounding noise on centered columns passes; a real change fails (11-10 O2 refusal)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import o2_partial as op  # noqa: E402


def _design(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(5000, 12))
    X -= X.mean(0)                                  # centered, as the frozen design
    Rd = rng.normal(size=(5000, 3))
    Rd -= Rd.mean(0)
    return {"X": X, "Rd": Rd, "col_var": ["a"] * 12}


def _recorded(d):
    return {"design": op.design_digest(d), "design_sums": op.design_sums(d)}


def test_exact():
    d = _design()
    assert op.same_design(d, _recorded(d)) == "exact"


def test_rounding_noise_passes():
    d = _design()
    rec = _recorded(d)
    noisy = {**d, "X": d["X"] * (1 + 1e-15 * np.random.default_rng(1).normal(size=d["X"].shape))}
    assert op.same_design(noisy, rec) == "float"


def test_real_change_fails():
    d = _design()
    rec = _recorded(d)
    X = d["X"].copy()
    X[:100, 3] += 0.5                               # a shifted event / wrong regressor on 100 frames
    assert op.same_design({**d, "X": X}, rec) == ""


def test_shape_change_fails():
    d = _design()
    rec = _recorded(d)
    assert op.same_design({**d, "X": d["X"][:, :11]}, rec) == ""
