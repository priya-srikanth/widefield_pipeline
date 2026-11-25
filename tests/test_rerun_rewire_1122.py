"""The 11-22 four-camera re-run rewiring (docs/status/RERUN_PLAN_RECONCILED_2026-11-21.md section 2): consumers read the
fused train through `session_licks` when it is selected; skip-if-exists caches key on their inputs; runbook command
bugs fixed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import session_licks as SL

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def test_lp_licks_reads_the_fused_train(monkeypatch, tmp_path):
    from wfield_local import lp_licks
    monkeypatch.setenv(SL.FUSED_ENV, "fz")
    monkeypatch.setattr(SL, "session_onsets_s", lambda label, rv=None, **k: np.array([1.0, 2.0, 3.0]))
    import wfield_local.orofacial_cohort as oc
    monkeypatch.setattr(oc, "_cue_table", lambda sd: pd.DataFrame({"trial_id": [1], "cue_daq": [0.5]}))
    t, cue = lp_licks._read_times("PS92_0818", tmp_path, tmp_path / "f4", tmp_path / "f1")
    assert t.tolist() == [1.0, 2.0, 3.0] and cue.index.tolist() == [1]          # no cam1 events added on top


def test_lp_spec_token_carries_the_fused_signature(monkeypatch):
    from wfield_local import lp_licks
    monkeypatch.setattr(lp_licks, "_cam1_signature", lambda rv=None: "c1")
    monkeypatch.setattr(lp_licks, "attempts_path", lambda rv=None: Path("nonexistent_attempts.csv"))
    monkeypatch.setenv(SL.FUSED_ENV, "off")
    off = lp_licks.spec_token()
    monkeypatch.setenv(SL.FUSED_ENV, "fz")
    monkeypatch.setattr(SL, "signature", lambda rv=None: "fused:fz:abc")
    on = lp_licks.spec_token()
    assert lp_licks.LP_VERSION == 4 and off != on and on.endswith(":fused:fz:abc") and "fused" not in off


def test_restnl_lick_set_defaults_to_fused_under_a_fused_train(monkeypatch):
    from wfield_local import restnl_licks as RL
    monkeypatch.delenv(RL.ENV, raising=False)
    monkeypatch.setenv(SL.FUSED_ENV, "off")
    assert RL.lick_set() == "cam4" and not RL.is_fused()
    monkeypatch.setenv(SL.FUSED_ENV, "fz")
    assert RL.lick_set() == "fused" and RL.is_fused()
    monkeypatch.setenv(RL.ENV, "cam4cam1")                 # an explicit set still wins
    assert RL.lick_set() == "cam4cam1"


def test_quiet_chain_lick_times_use_the_fused_train(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts" / "cohort2_qc"))
    import importlib
    ql = importlib.import_module("lp_licks")
    assert Path(ql.__file__).parent.name == "cohort2_qc"
    monkeypatch.setenv(SL.FUSED_ENV, "fz")
    monkeypatch.setattr(SL, "session_onsets_s", lambda label, rv=None, **k: np.array([1.0, 1.02, 2.0]))
    t, parts = ql.lick_times("PS92_0818", return_parts=True)
    assert t.tolist() == [1.0, 2.0] and parts["cam1_src"] == "fused"            # the 40 ms dedup still applies


def test_glm_hmm_sequence_key_moves_with_the_lick_set(monkeypatch):
    from wfield_local import glm_hmm, trial_classes
    monkeypatch.setattr(trial_classes, "class_source_signature", lambda *a, **k: "cls1")
    monkeypatch.setattr(SL, "signature", lambda rv=None: "")
    a = glm_hmm.sequence_key(["PS92_0818"])
    monkeypatch.setattr(SL, "signature", lambda rv=None: "fused:fz:abc")
    b = glm_hmm.sequence_key(["PS92_0818"])
    monkeypatch.setattr(trial_classes, "class_source_signature", lambda *a, **k: "cls2")
    c = glm_hmm.sequence_key(["PS92_0818"])
    assert len({a, b, c}) == 3                    # labels alone no longer key the sequences (plan item 4)


def test_c2_phase_name_parse_accepts_suffix_and_kind_flag():
    from c2_phase_sessions_aligned import _parse_name
    assert _parse_name(Path("c2_phase_cued_side0_peak_sessions.csv")) == ("cued", "side0", "peak")
    assert _parse_name(Path("c2_phase_cued_side0_peak_sessions_NEW.csv")) == ("cued", "side0", "peak")
    assert _parse_name(Path("c2_phase_all_mid_sessions.csv")) == ("all", "mid", "norm")
    with pytest.raises(ValueError):
        _parse_name(Path("c2_phase_all_mid.csv"))


def test_learned_compensation_summarize_alias_and_unknown_command(monkeypatch):
    import learned_compensation as lc
    called = []
    monkeypatch.setattr(lc, "cmd_summarise", lambda: called.append("s"))
    monkeypatch.setattr(lc, "cmd_plot", lambda: called.append("p"))
    monkeypatch.setattr(sys, "argv", ["learned_compensation.py", "summarize"])
    assert lc.main() == 0 and called == ["s", "p"]                  # used to exit 0 having done nothing
    monkeypatch.setattr(sys, "argv", ["learned_compensation.py", "sumarise"])
    with pytest.raises(SystemExit) as e:
        lc.main()
    assert "unknown command" in str(e.value)


def test_skip_if_exists_caches_follow_the_lick_set(monkeypatch):
    import revup_behaviour as rb
    import within_lick_correction as wlc
    monkeypatch.setenv(rb.JAW_ENV, "2d")                            # the jaw rule is in the folder name too (default classes, 11-23)
    monkeypatch.setattr(SL, "signature", lambda rv=None: "")
    assert rb.sess_dir().name == rb.SESS and wlc.licks_name() == "licks"
    monkeypatch.setattr(SL, "signature", lambda rv=None: "fused:fz:abc")
    assert rb.sess_dir().name.startswith(rb.SESS + "_fused_") and wlc.licks_name().startswith("licks_fused_")


def test_iti_extract_refuses_another_lick_set(monkeypatch, tmp_path):
    import iti_licks as il
    f = tmp_path / "x.npz"
    np.savez(f, lick_sig="old|kinematics_all:1:2")
    monkeypatch.setattr(il, "_lick_sig", lambda: "new|kinematics_fused:3:4")
    with pytest.raises(RuntimeError):
        il.check_extract_licks(f)
    np.savez(f, other=1)                                            # a pre-11-22 extract (no lick_sig)
    monkeypatch.setenv(SL.FUSED_ENV, "off")
    il.check_extract_licks(f)                                       # accepted on the legacy lick set
    monkeypatch.setenv(SL.FUSED_ENV, "fz")
    with pytest.raises(RuntimeError):
        il.check_extract_licks(f)                                   # refused under the fused train


def test_pose_chain_rebuilds_outputs_older_than_their_inputs(tmp_path):
    import os
    import time

    import pose_chain as pc
    src, out = tmp_path / "windows_LP.csv", tmp_path / "frames.csv.gz"
    src.write_text("a")
    assert pc._stale(out, src)                                      # missing
    out.write_text("b")
    t = time.time()
    os.utime(out, (t, t))
    os.utime(src, (t - 100, t - 100))
    assert not pc._stale(out, src)
    os.utime(src, (t + 100, t + 100))
    assert pc._stale(out, src)                                      # the input changed after the output was made


def test_jaw_onsets_centre_blanking_is_source_aware(monkeypatch):
    import jaw_onsets as jo
    from wfield_local import pose3d
    monkeypatch.setenv(pose3d.SOURCE_ENV, "2cam")
    assert jo.centre_blanked() == jo.CENTRE
    monkeypatch.setenv(pose3d.SOURCE_ENV, "4cam")
    assert jo.centre_blanked() == ()


def test_adopted_pose3d_source_is_untagged_and_others_are_tagged(monkeypatch):
    from wfield_local import pose3d, run_mode
    monkeypatch.delenv(pose3d.SOURCE_ENV, raising=False)
    monkeypatch.delenv(pose3d.TAG_ENV, raising=False)
    monkeypatch.setattr(pose3d, "ADOPTED_SOURCE", "4cam")
    monkeypatch.setattr(pose3d, "ADOPTED_TAG", "_3net_20261121")
    assert pose3d.pose3d_source() == "4cam" and pose3d.pose3d_dir(root="R").name == "pose3d_4cam_3net_20261121"
    assert run_mode.pose3d_part() == ""                              # the adopted run's outputs are untagged
    monkeypatch.setenv(pose3d.SOURCE_ENV, "2cam")
    assert pose3d.pose3d_dir(root="R").name == "pose3d" and run_mode.pose3d_part() == "p3d-2cam"
    monkeypatch.setenv(pose3d.SOURCE_ENV, "4cam")
    monkeypatch.setenv(pose3d.TAG_ENV, "_other")
    assert run_mode.pose3d_part() == "p3d-4cam_other"


def test_fused_default_is_cohort1_only(monkeypatch):
    monkeypatch.delenv(SL.FUSED_ENV, raising=False)
    monkeypatch.setattr(SL, "FUSED_DEFAULT", "fz")
    monkeypatch.setenv("WIDEFIELD_COHORT", "cohort1")
    assert SL.fused_run() == "fz"
    monkeypatch.setenv("WIDEFIELD_COHORT", "cohort2")
    assert SL.fused_run() is None
