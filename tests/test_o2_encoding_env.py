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
    assert ident["run_tag"].endswith("stopRun6j") and ident["area_basis"] == "splitr8"
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


def test_profile_keeps_the_callers_encoder_settings(tmp_path, monkeypatch):
    """Coordinator 11-04: `o2_encoding --profile` applied the render profile, which CLEARS every COMBINED_ENCODING_*
    variable, so the first enc_rebuild_20261104 bundle lost the encoder's input settings. They are re-applied on top
    and env.json carries them."""
    import os

    import pytest

    m = _mod()
    for k in list(os.environ):
        if k.startswith(("COMBINED_ENCODING_", "OROFACIAL_", "WIDEFIELD_", "WFIELD_", "LOCANMF_")):
            monkeypatch.delenv(k)
    tabs = {}
    for k in ("EXEC3D_TABLE", "TARGET3D_TABLE", "BINS_CALIB"):
        f = tmp_path / f"{k.lower()}.csv"
        f.write_text("x\n")
        tabs[f"COMBINED_ENCODING_{k}"] = str(f)
    caller = {"OROFACIAL_GATE": "attempted", "COMBINED_ENCODING_LICKS": "combined", "COMBINED_ENCODING_TONGUE": "3d",
              "COMBINED_ENCODING_EXEC": "polar", "COMBINED_ENCODING_RUN": "lp_r4_heatmap_20261013__clean_v36", **tabs,
              "WFIELD_INPUT_CACHE": str(tmp_path / "cache"), "WIDEFIELD_LEFTOVER": "x"}
    env = dict(caller)
    kept = m.apply_profile("cohort1_rebuild_20261103", environ=env)      # a dict: no fence installed in this process
    assert "WIDEFIELD_LEFTOVER" not in env and "WIDEFIELD_LEFTOVER" not in kept            # leftovers still cleared
    for k, v in caller.items():
        if k != "WIDEFIELD_LEFTOVER":
            assert env[k] == v, k
    assert env["WIDEFIELD_TRIAL_GATE"] == "camera"                                          # the profile is applied
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    d = tmp_path / "b"
    d.mkdir()
    j = m.bundle_env(d)
    for k in ("COMBINED_ENCODING_LICKS", "COMBINED_ENCODING_TONGUE", "COMBINED_ENCODING_EXEC", "COMBINED_ENCODING_RUN",
              "OROFACIAL_GATE"):
        assert j[k] == caller[k], k
    for k in tabs:
        assert j[k] == f"@tables/{Path(tabs[k]).name}", k
    assert "WFIELD_INPUT_CACHE" not in j
    # an arm-owned encoder setting left in the shell (it would re-tag the run) is refused
    with pytest.raises(SystemExit, match="arm owns"):
        m.apply_profile("cohort1_rebuild_20261103", environ={"COMBINED_ENCODING_DRIFT_MIN_PERIOD_S": "1500"})


def _fake_bundle(tmp_path, monkeypatch, m):
    """A bundle dir with the rebuild profile's env.json / identity, an OLD code snapshot and inputs that must survive."""
    import json
    import os

    from wfield_local import o2_bundle
    _profile_env(monkeypatch)
    d = tmp_path / "bundle"
    (d / "code" / "wfield_local").mkdir(parents=True)
    (d / "code" / "wfield_local" / "old.py").write_text("x = 1\n")
    (d / "inputs").mkdir()
    (d / "inputs" / "PS93_0814.pkl").write_bytes(b"inputs")
    env = m.bundle_env(d)
    (d / "env.json").write_text(json.dumps(env))
    ident = o2_bundle.identity()
    (d / "bundle.json").write_text(json.dumps({"identity": ident, "commit": "0" * 40}))
    (d / "COMMANDS.md").write_text(f"code {'0' * 40} (wf_rebuild)\n")
    real = m._git
    monkeypatch.setattr(m, "_git", lambda repo, *a: "" if a[0] == "status" else real(repo, *a))
    return d, ident, os


def test_refresh_code_replaces_only_code(tmp_path, monkeypatch):
    """refresh-code (11-04): code/ = HEAD's archive, runner rewritten, commit recorded; inputs untouched."""
    import argparse
    import json
    m = _mod()
    d, ident, _os = _fake_bundle(tmp_path, monkeypatch, m)
    head = m._git(Path(m.__file__).resolve().parents[1], "rev-parse", "HEAD")
    assert m.cmd_refresh_code(argparse.Namespace(name="", bundle=str(d), force=False)) == 0
    assert not (d / "code" / "wfield_local" / "old.py").exists()
    assert (d / "code" / "wfield_local" / "combined_encoding.py").exists() and (d / "code" / "configs").is_dir()
    assert not (d / "code.new").exists()
    assert (d / "inputs" / "PS93_0814.pkl").read_bytes() == b"inputs"
    meta = json.loads((d / "bundle.json").read_text())
    assert meta["commit"] == head and meta["commit_inputs"] == "0" * 40 and meta["identity"] == ident
    assert meta["code_refreshes"][-1]["to"] == head
    assert head in (d / "COMMANDS.md").read_text() and (d / "enc_runner.py").read_text() == m.RUNNER


def test_refresh_code_refuses_identity_change_and_dirty_tree(tmp_path, monkeypatch):
    import argparse
    import json

    import pytest
    m = _mod()
    d, ident, _os = _fake_bundle(tmp_path, monkeypatch, m)
    (d / "bundle.json").write_text(json.dumps({"identity": dict(ident, run_tag="another-run"), "commit": "0" * 40}))
    with pytest.raises(SystemExit, match="does not compute the bundle's run identity"):
        m.cmd_refresh_code(argparse.Namespace(name="", bundle=str(d), force=False))
    assert (d / "code" / "wfield_local" / "old.py").exists() and not (d / "code.new").exists()   # untouched
    monkeypatch.setattr(m, "_git", lambda repo, *a: " M wfield_local/x.py" if a[0] == "status" else "abc")
    with pytest.raises(SystemExit, match="uncommitted"):
        m.cmd_refresh_code(argparse.Namespace(name="", bundle=str(d), force=False))


def test_compare_trials():
    import numpy as np
    m = _mod()
    a = {"x": np.array([1.0, np.nan, 3.0]), "n": np.array(["a", "b"]), "i": np.arange(3)}
    assert m.compare_trials(a, dict(a)) == (0.0, [])
    w, bad = m.compare_trials(a, dict(a, x=np.array([1.0, np.nan, 3.0 + 1e-3])))
    assert bad == ["x"] and w > 1e-4
    assert m.compare_trials(a, dict(a, x=np.array([1.0, 2.0, 3.0])))[1] == ["x"]          # NaN pattern differs
    assert m.compare_trials(a, dict(a, n=np.array(["a", "c"])))[1] == ["n"]


def test_runner_ships_attempts_option_and_records_it():
    """ATTEMPTS (submit option) -> COMBINED_ENCODING_ATTEMPTS inside the bundle, recorded with its hash in run.json;
    never an env.json setting."""
    from wfield_local import o2_bundle
    m = _mod()
    assert "COMBINED_ENCODING_ATTEMPTS" in o2_bundle.SUBMIT_ENVS
    r = m.RUNNER
    assert 'os.environ["COMBINED_ENCODING_ATTEMPTS"] = str(f_)' in r and '"attempts_sha256"' in r
    assert r.index("COMBINED_ENCODING_ATTEMPTS") < r.index("run_session")
    assert r.index("**attempts") < r.index("run_session")


def test_refresh_code_ships_the_attempts_table(tmp_path, monkeypatch):
    import argparse
    import json

    from wfield_local import combined_encoding as ce
    m = _mod()
    d, _ident, _os = _fake_bundle(tmp_path, monkeypatch, m)
    (d / "tables").mkdir(exist_ok=True)
    (d / "tables" / "trial_attempts.csv").write_text("label,trial_id,attempted,canonical\nA,1,True,True\n")
    old = ce.file_sha256(d / "tables" / "trial_attempts.csv")
    new = tmp_path / "new_attempts.csv"
    new.write_text("label,trial_id,attempted,canonical\nA,1,False,True\n")
    assert m.cmd_refresh_code(argparse.Namespace(name="", bundle=str(d), force=False, attempts=str(new))) == 0
    meta = json.loads((d / "bundle.json").read_text())
    assert meta["attempts_sha256"] == ce.file_sha256(new) == ce.file_sha256(d / "tables" / "trial_attempts.csv")
    assert (d / "tables" / f"trial_attempts_{old[:12]}.csv").exists()                  # the earlier table is kept
