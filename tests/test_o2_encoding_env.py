"""O2 bundles record their input settings (COMBINED_ENCODING_* / OROFACIAL_*) and copy env-named tables."""
import importlib.util
from pathlib import Path


def _mod():
    spec = importlib.util.spec_from_file_location("o2e", Path(__file__).resolve().parents[1] / "scripts" / "o2_encoding.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bundle_env_records_inputs_and_copies_tables(tmp_path, monkeypatch):
    m = _mod()
    tab = tmp_path / "calib.csv"
    tab.write_text("x\n1\n")
    for k in list(__import__("os").environ):
        if k.startswith(("COMBINED_ENCODING_", "OROFACIAL_")):
            monkeypatch.delenv(k)
    monkeypatch.setenv("COMBINED_ENCODING_LICKS", "combined")
    monkeypatch.setenv("COMBINED_ENCODING_BINS_CALIB", str(tab))
    monkeypatch.setenv("COMBINED_ENCODING_ONLY", "nokin:")             # per-submit: not recorded
    monkeypatch.setenv("OROFACIAL_GATE", "attempted")
    d = tmp_path / "bundle"
    d.mkdir()
    env = m.bundle_env(d)
    assert env == {"COMBINED_ENCODING_BINS_CALIB": "@tables/calib.csv", "COMBINED_ENCODING_LICKS": "combined",
                   "OROFACIAL_GATE": "attempted"}
    assert (d / "tables" / "calib.csv").read_text() == "x\n1\n"


def test_bins_width_env(monkeypatch):
    from wfield_local import combined_encoding as ce
    monkeypatch.delenv("COMBINED_ENCODING_BINS_WIDTH", raising=False)
    assert ce.bins_width() == 1.0
    monkeypatch.setenv("COMBINED_ENCODING_BINS_WIDTH", "0.5")
    assert ce.bins_width() == 0.5


def test_bins_is_a_submit_option(monkeypatch, tmp_path):
    """The binned contest is switched per submit, so one bundle serves contests and full models."""
    m = _mod()
    monkeypatch.setenv("COMBINED_ENCODING_BINS", "1")
    monkeypatch.setenv("COMBINED_ENCODING_BINS_WIDTH", "2")
    assert "COMBINED_ENCODING_BINS" not in m.bundle_env(None)
    assert "COMBINED_ENCODING_BINS_WIDTH" not in m.bundle_env(None)


def test_bundled_fr_grouped_at_design_time(monkeypatch):
    """Raw bundled factors are grouped per submit (master_v1 mode B = lick sum + others), not at bundle time."""
    import numpy as np
    from wfield_local import combined_encoding as ce
    from wfield_local import fr_groups
    G = fr_groups.GROUPINGS["master_v1"]
    X = np.arange(12 * 3, dtype=float).reshape(3, 12)
    monkeypatch.delenv(fr_groups.ENV, raising=False)
    assert ce.group_bundled_fr(X, G["set"]) is X
    monkeypatch.setenv(fr_groups.ENV, "master_v1:B")
    Y = ce.group_bundled_fr(X, G["set"])
    assert Y.shape == (3, 1 + len(G["other"]))
    assert np.allclose(Y[:, 0], X[:, list(G["lick"])].sum(1))
    assert ce.group_bundled_fr(X, G["set"], grouped=True) is X
