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
        if k.startswith(("COMBINED_ENCODING_", "OROFACIAL_", "WIDEFIELD_", "WFIELD_", "LOCANMF_")):
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


def _profile_env(monkeypatch):
    import os

    from wfield_local import render_profile
    from wfield_local import trial_classes as tc
    for k in list(os.environ):
        if k.startswith(("COMBINED_ENCODING_", "OROFACIAL_", "WIDEFIELD_", "WFIELD_", "LOCANMF_")):
            monkeypatch.delenv(k)
    for k, v in render_profile.load("cohort1_rebuild_20261103")["env"].items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(tc, "class_source_signature", lambda *a, **k: "sig")


def test_rebuild_env_round_trip_and_tag_refusal(tmp_path, monkeypatch):
    """Coordinator 11-04: env.json records the FULL profile env (+ COMBINED_ENCODING_* / OROFACIAL_*), never this
    machine's store paths or submit options; applied alone (as the runner does) it reproduces the recorded run identity;
    a changed setting is REFUSED."""
    import os

    import pytest

    from wfield_local import o2_bundle
    m = _mod()
    _profile_env(monkeypatch)
    tab = tmp_path / "exec3d.csv"
    tab.write_text("x" + chr(10))
    monkeypatch.setenv("COMBINED_ENCODING_EXEC3D_TABLE", str(tab))
    monkeypatch.setenv("OROFACIAL_GATE", "attempted")
    monkeypatch.setenv("WFIELD_INPUT_CACHE", str(tmp_path / "cache"))           # local-only: not recorded
    monkeypatch.setenv("COMBINED_ENCODING_ONLY", "drop:")                        # per submit: not recorded
    d = tmp_path / "b"
    d.mkdir()
    env = m.bundle_env(d)
    ident = o2_bundle.identity()
    assert ident["run_tag"].endswith("stopA2jrt3500c70") and ident["area_basis"] == "splitr8"
    for k in ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LOCANMF_PROJECT", "WIDEFIELD_TRIAL_GATE", "WIDEFIELD_TRIAL_WINDOW",
              "WIDEFIELD_TRIAL_END", "WIDEFIELD_TRIAL_STOPPED", "WIDEFIELD_WORKING_TRIALS", "WIDEFIELD_PRE_FROM",
              "WIDEFIELD_LP_FROZEN", "OROFACIAL_GATE"):
        assert env[k] == os.environ[k], k
    assert env["COMBINED_ENCODING_EXEC3D_TABLE"] == "@tables/exec3d.csv" and (d / "tables" / "exec3d.csv").exists()
    assert "WFIELD_INPUT_CACHE" not in env and "COMBINED_ENCODING_ONLY" not in env
    # the runner: clear every setting, apply env.json alone, verify
    for k in list(os.environ):
        if k.startswith(o2_bundle.RECORD_PREFIXES):
            monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(k, str(d / v[1:]) if v.startswith("@") else v)
    assert o2_bundle.verify(ident) == ident
    monkeypatch.setenv("WIDEFIELD_TRIAL_STOPPED", "any")                         # another run: refused
    with pytest.raises(o2_bundle.BundleMismatch, match="run_tag"):
        o2_bundle.verify(ident)
    with pytest.raises(SystemExit, match="not of the current run"):              # collect / check refuse it too
        m._refuse_other_run({"identity": ident})


def test_runner_applies_settings_before_importing_and_verifies():
    m = _mod()
    r = m.RUNNER
    assert r.index("for k, v in bundle_env.items():") < r.index("o2_bundle.verify(") < r.index("from wfield_local import combined_encoding as ce")
    assert "del os.environ[k]" in r                                             # nothing from the O2 shell leaks in
    tags = [t for t, *_ in m.FULL_RUNS]
    assert len(tags) == 14 and len(set(tags)) == 14 and {"main", "fr_A", "bins_w05", "musall_shuf"} <= set(tags)


def test_bundle_goes_under_the_rebuild_root(tmp_path, monkeypatch):
    from wfield_local import rebuild_root, run_mode, writeguard
    m = _mod()
    monkeypatch.setattr(rebuild_root, "root", lambda: tmp_path / "rebuild_x")
    d = m.bundle_dir(None, "enc_t")
    assert d == tmp_path / "rebuild_x" / "o2" / "enc_t"
    assert run_mode.data_store_ok(writeguard._normalize(d / "inputs" / "a.pkl"))
