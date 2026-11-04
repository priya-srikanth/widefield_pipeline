"""THE REBUILD ROOT (coordinator 11-03, fourth review A + C): under a render profile with ``rebuild_root`` every derived
root resolves into one fresh folder, a write to an old default derived root RAISES, a read of a default derived product
RAISES, and a derived-product read resolves inside the rebuild root. Also: the writer register extended to
`scripts/` subfolders and `wfield_local/` (C), R1-R5 and the O2 runner guard (B).

The fence is a process audit hook, so the tests that exercise it run in a SUBPROCESS (never in the pytest process).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROFILE = "cohort1_rebuild_20261103"


def _run(code: str, env_extra: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WIDEFIELD_", "WFIELD_"))}
    env.update(env_extra)
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(tmp_path), capture_output=True, text=True,
                          timeout=300)


PROBE = r"""
import json, os
from wfield_local import render_profile
d = render_profile.apply("%s")
os.environ["WIDEFIELD_REBUILD_ROOT"] = %r                       # the profile's root, pointed at a scratch folder
os.environ["WIDEFIELD_FROZEN_MODEL_DIR"] = os.environ["WIDEFIELD_REBUILD_ROOT"] + "/frozen_models"
os.environ["WIDEFIELD_SESSION_CACHE"] = os.environ["WIDEFIELD_REBUILD_ROOT"] + "/session_cache"
from wfield_local import config, rebuild_root, writeguard
from wfield_local import orofacial_outputs as oo
rv = config.resolver()
R = rebuild_root.root().as_posix().lower()
out = {"roots": {n: rv.root(n) for n in ("figures_working", "cue_analysis_out", "cohort_out", "labcams", "orofacial")},
       "enc": oo.encoding_root().as_posix(), "writes": {}, "reads": {}}
old = {"grant": rv.input_root("cohort_out") + "/grant_figures/epoch/zz_rebuild_probe.png",
       "fw": rv.input_root("figures_working") + "/zz_rebuild_probe.png",
       "enc": rv.input_root("orofacial") + "/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/zz_probe/x.csv"}
for k, p in old.items():
    try:
        open(p, "wb")
        out["writes"][k] = "WRITTEN"
    except PermissionError as e:
        out["writes"][k] = "refused"
    out["writes"][k + "_guard"] = writeguard.is_writable(p)
for k, p in {"grant": rv.input_root("cohort_out") + "/grant_figures/epoch/data/zz_absent.csv",
             "enc": rv.input_root("orofacial") + "/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/qc/zz_absent.csv",
             "fw": rv.input_root("figures_working") + "/zz_absent.json"}.items():
    try:
        open(p)
        out["reads"][k] = "READ"
    except PermissionError:
        out["reads"][k] = "refused"
    except FileNotFoundError:
        out["reads"][k] = "not fenced"
new = rv.root("cohort_out") + "/grant_figures/epoch/data/probe.csv"
os.makedirs(os.path.dirname(new), exist_ok=True)
open(new, "w").write("x")
out["reads"]["derived_in_root"] = open(new).read()
out["R"] = R
print("JSON" + json.dumps(out))
"""


def test_profile_rebuild_root_resolves_and_fences(tmp_path):
    root = (tmp_path / "rebuild").as_posix()
    r = _run(PROBE % (PROFILE, root), {}, tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.split("JSON", 1)[1])
    R = out["R"]
    for n in ("figures_working", "cue_analysis_out", "cohort_out"):
        assert out["roots"][n].replace("\\", "/").lower().startswith(R), (n, out["roots"][n])
    for n in ("labcams", "orofacial"):                                   # INPUT roots stay where they are
        assert not out["roots"][n].replace("\\", "/").lower().startswith(R), n
    assert out["enc"].lower().startswith(R) and out["enc"].endswith("lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
    for k in ("grant", "fw", "enc"):
        assert out["writes"][k] == "refused", (k, out["writes"])           # the audit hook: any Python open()
        assert out["writes"][k + "_guard"] is False, k                    # and writeguard agree
    assert out["reads"]["grant"] == "refused" and out["reads"]["enc"] == "refused" and out["reads"]["fw"] == "refused"
    assert out["reads"]["derived_in_root"] == "x"


def test_rebuild_root_spec_and_inputs():
    from wfield_local import rebuild_root as rr
    s = rr.spec()
    assert set(s["derived_roots"]) == {"figures_working", "cue_analysis_out", "cohort_out"}
    assert s["derived_subtrees"]["orofacial"]["patterns"] == ["*/imaging_encoding"]
    assert "joint_bases" in s["inputs_in_place"]["cohort_out"]
    assert "epoch_boundaries.json" in s["seed_inputs"]["cohort_out"]
    assert any(x.endswith("qc/jaw_onsets.csv") for x in s["seed_inputs"]["orofacial"])


def test_read_and_write_rules_unit(monkeypatch, tmp_path):
    """The rule functions (not the hook): session data and listed inputs read fine; default derived products do not."""
    from wfield_local import config
    from wfield_local import rebuild_root as rr
    rv = config.resolver()
    lab, oro = rv.input_root("labcams"), rv.input_root("orofacial")
    monkeypatch.setenv(rr.ENV, str(tmp_path / "rb"))
    rr._fence.cache_clear()
    assert rr.read_refusal(lab + "/20260818/PS93_20260818_x/motion_corrected/wfield_local_results/SVTcorr.npy") is None
    assert rr.read_refusal(lab + "/joint_bases/PS93/basis.npz") is None
    assert rr.read_refusal(lab + "/grant_figures/epoch/data/a.csv")
    assert rr.read_refusal(lab + "/frozen_models/x.joblib")
    assert rr.read_refusal(oro + "/lp_r4_heatmap_20261013__clean_v36/sessions/PS92/20260818/licks.csv") is None
    assert rr.read_refusal(oro + "/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/a.csv")
    assert rr.write_refusal(lab + "/20260818/PS93_x/motion_corrected/wfield_local_results/hemo_v/SVTcorr.npy")
    monkeypatch.setenv(rr.VARIANT_BUILD_ENV, "1")                     # the variant-build step: hemo_* dirs only
    assert rr.write_refusal(lab + "/20260818/PS93_x/motion_corrected/wfield_local_results/hemo_v/SVTcorr.npy") is None
    assert rr.write_refusal(lab + "/20260818/PS93_x/motion_corrected/wfield_local_results/SVTcorr.npy")
    assert rr.write_refusal(str(tmp_path / "rb" / "labcams" / "grant_figures" / "x.png")) is None
    assert rr.write_refusal(str(Path(__import__("tempfile").gettempdir()) / "scratch.txt")) is None
    monkeypatch.delenv(rr.ENV)
    assert rr.write_refusal(lab + "/grant_figures/x.png") is None and rr.read_refusal(lab + "/grant_figures/x") is None


def test_init_seeds_inputs_with_hashes(monkeypatch, tmp_path):
    from wfield_local import rebuild_root as rr
    src = tmp_path / "orig" / "epoch_boundaries.json"
    src.parent.mkdir(parents=True)
    src.write_text('{"boundaries": {}}')
    dst = tmp_path / "rb" / "labcams" / "epoch_boundaries.json"
    monkeypatch.setenv(rr.ENV, str(tmp_path / "rb"))
    monkeypatch.setattr(rr, "seed_plan", lambda: [("cohort_out/epoch_boundaries.json", src, dst)])
    rec = rr.init()
    assert dst.read_text() == src.read_text() and rec["seeded"][0]["sha256"]
    assert (tmp_path / "rb" / rr.MANIFEST).exists()
    dst.write_text("changed")
    with pytest.raises(SystemExit):
        rr.init()                                                       # never overwrites a different seed


def test_unseeded_root_raises_for_a_silently_optional_input(monkeypatch, tmp_path):
    """The jaw calls / coverage table are optional for a reader (absent = skipped): inside a rebuild root an absent one
    means the root was not seeded, so it RAISES instead of silently dropping the jaw-only class."""
    from wfield_local import rebuild_root as rr
    from wfield_local import trial_classes as TC
    monkeypatch.setenv(rr.ENV, str(tmp_path / "rb"))
    with pytest.raises(FileNotFoundError, match="seed"):
        rr.require_seed(tmp_path / "rb" / "LP_analysis" / "x" / "imaging_encoding" / "qc" / "jaw_onsets.csv")
    rr.require_seed(tmp_path / "elsewhere.csv")                                  # outside the root: no-op
    with pytest.raises(FileNotFoundError, match="seed"):
        TC.coverage_note(tmp_path / "rb" / "camera_coverage.csv")
    monkeypatch.delenv(rr.ENV)
    assert "missing" in TC.coverage_note(tmp_path / "rb" / "camera_coverage.csv")


def test_package_import_installs_the_fence(tmp_path):
    code = ("import os, sys; import wfield_local; from wfield_local import rebuild_root as rr; "
            "print('INSTALLED' if rr._installed else 'NOT')")
    assert "INSTALLED" in _run(code, {"WIDEFIELD_REBUILD_ROOT": str(tmp_path / "rb")}, tmp_path).stdout
    assert "NOT" in _run(code, {}, tmp_path).stdout


# ---------------------------------------------------------------------------------------------------- R5
@pytest.mark.parametrize("env", [{"WIDEFIELD_LICK_SOURCE": "lp"}, {"WIDEFIELD_GATE_SOURCE": "lp"},
                                 {"WIDEFIELD_ENL_WINDOW_S": "1.5"}])
def test_require_default_covers_lp_gate_lp_and_enl(monkeypatch, env):
    from wfield_local import run_mode
    for k in ("WIDEFIELD_TRIAL_GATE", "WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_GATE_SOURCE",
              "WIDEFIELD_ENL_WINDOW_S"):
        monkeypatch.delenv(k, raising=False)
    run_mode.require_default("x")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(RuntimeError):
        run_mode.require_default("x")


def test_render_profile_clears_leftovers_and_checks_the_full_tag():
    from wfield_local import render_profile
    env = {"WIDEFIELD_GATE_SOURCE": "lp", "WIDEFIELD_ENL_WINDOW_S": "1.5", "WIDEFIELD_FROZEN_OUT_ROOT": "x",
           "WFIELD_ITI_OUT_W": "y", "PATH": "keep"}
    d = render_profile.apply(PROFILE, env)
    assert {"WIDEFIELD_GATE_SOURCE", "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_FROZEN_OUT_ROOT", "WFIELD_ITI_OUT_W"} <= set(
        d["cleared"])
    assert "WIDEFIELD_GATE_SOURCE" not in env and env["PATH"] == "keep"
    assert env["WIDEFIELD_REBUILD_ROOT"] == "M:/MICROSCOPE/Priya/Widefield/rebuild_20261103"
    assert env["WIDEFIELD_FROZEN_MODEL_DIR"].startswith(env["WIDEFIELD_REBUILD_ROOT"])
    assert env["WIDEFIELD_SESSION_CACHE"].startswith(env["WIDEFIELD_REBUILD_ROOT"])
    bash = render_profile.shell_lines(PROFILE, "bash")
    ps = render_profile.shell_lines(PROFILE, "powershell")
    assert "unset" in bash[0] and "Remove-Item" in ps[0]


def test_render_profile_check_fails_on_a_tag_mismatch(tmp_path):
    code = ("import sys; from wfield_local import render_profile as rp; import wfield_local.lp_licks as L; "
            "L.output_tag = lambda: 'WRONG'; ok, lines = rp.check('%s'); print('OK' if ok else 'MISMATCH')" % PROFILE)
    r = _run(code, {"WIDEFIELD_GATE_SOURCE": "lp"}, tmp_path)
    assert "MISMATCH" in r.stdout, r.stderr[-2000:]
    r = _run(code.replace("L.output_tag = lambda: 'WRONG'; ", ""), {"WIDEFIELD_GATE_SOURCE": "lp"}, tmp_path)
    assert "OK" in r.stdout, r.stderr[-2000:]                          # the leftover GATE_SOURCE was cleared


# ---------------------------------------------------------------------------------------------------- R1-R4
def test_r1_rest_frozen_decoder_nl_is_routed_and_keyed():
    src = (REPO / "scripts" / "rest_migration" / "rest_frozen_decoder_nl.py").read_text(encoding="utf-8")
    assert "M:/MICROSCOPE" not in src.split("# R1")[1].split("EPOCHS =")[0]
    assert "_run_mode.subdir(G / \"rest_frozen_decoder_restnl\")" in src
    assert "_run_mode.cache_tag()" in src and "rf._epoch_dir()" in src


def test_r2_deck_refuses_a_tagged_build_on_default_inputs(monkeypatch, tmp_path):
    from wfield_local import locanmf_analysis_deck as deck
    monkeypatch.delenv("WIDEFIELD_REBUILD_ROOT", raising=False)
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    with pytest.raises(RuntimeError, match="rebuild root"):
        deck.build_analysis_deck(tmp_path, tmp_path / "x.pptx")


def test_r3_rebuild_figs_defaults_follow_the_run(monkeypatch, tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "rebuild_figs"))
    import _common
    for k in ("WIDEFIELD_TRIAL_GATE", "WIDEFIELD_REBUILD_ROOT", "WIDEFIELD_HEMO_VARIANT"):
        monkeypatch.delenv(k, raising=False)
    d0 = _common.epoch_data_dir()
    _common.require_explicit(tmp_path, tmp_path, "--src")              # default run: fine
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    assert _common.epoch_data_dir() != d0                              # the tag folder
    with pytest.raises(SystemExit):
        _common.require_explicit(tmp_path, tmp_path, "--src")
    _common.require_explicit(tmp_path / "given", tmp_path, "--src")    # explicit: fine
    monkeypatch.setenv("WIDEFIELD_REBUILD_ROOT", str(tmp_path / "rb"))
    _common.require_explicit(tmp_path, tmp_path, "--src", ok_in_rebuild=True)
    for f in ("fig_frozen_refit_pooled", "fig_s_per_animal", "fig_state_position_sessions"):
        s = (REPO / "scripts" / "rebuild_figs" / f"{f}.py").read_text(encoding="utf-8")
        assert "_common.epoch_data_dir()" in s and 'Path("M:/MICROSCOPE/Priya/Widefield/labcams/grant' not in s, f


def test_r4_attempt_maps_keys_every_product_and_raises_on_missing(monkeypatch, tmp_path):
    from wfield_local import attempt_maps, run_mode
    monkeypatch.setattr(run_mode, "cache_tag", lambda: "csig-abc")
    assert attempt_maps.maps_path(tmp_path, "lick", "PS92").name == "maps_lick_PS92__csig-abc.npz"
    assert attempt_maps.hemispheres_path(tmp_path, "lick").name == "hemispheres_lick__csig-abc.csv"
    assert attempt_maps.product_name("counts_lick_PS92", ".csv") == "counts_lick_PS92__csig-abc.csv"
    import numpy as np
    monkeypatch.setattr(attempt_maps, "_mask_edges", lambda: (None, None))
    np.savez(tmp_path / "maps_lick_PS92__csig-abc.npz", **{"pre|far_contra|pre_hit": np.zeros((4, 4))})
    with pytest.raises(FileNotFoundError, match="PS93"):
        attempt_maps.hemisphere_table(tmp_path, "lick", ["PS92", "PS93"])
    assert len(attempt_maps.hemisphere_table(tmp_path, "lick", ["PS92"])) == 1
    src = (REPO / "wfield_local" / "attempt_maps.py").read_text(encoding="utf-8")
    assert 'od / f"counts_' not in src and 'od / f"hemispheres_' not in src


def test_o2_runner_text_carries_no_wfield_local_guard():
    sys.path.insert(0, str(REPO / "scripts"))
    src = (REPO / "scripts" / "o2_face_rhythm.py").read_text(encoding="utf-8")
    runner = src.split("RUNNER = r'''", 1)[1].split("'''", 1)[0]
    assert "require_default" not in runner and "wfield_local" not in runner
    main = src.split("\ndef main(argv=None) -> int:", 1)[1].split("\ndef ", 1)[0]
    assert "require_default(\"o2_face_rhythm\")" in main


# ---------------------------------------------------------------------------------------------------- C
# Writers outside top-level scripts/ (scripts/ subfolders, wfield_local/) whose share paths are NOT routed by run_mode
# and that do not refuse (`require_default`). Under the rebuild root they are SAFE either way:
#   RESOLVED  every share path goes through the resolver (PathResolver / encoding_root): it lands in the rebuild root
#   LITERAL   carries a literal share / machine path: the rebuild-root fence REFUSES its writes (and reads of default
#             derived products), so it fails loudly -- route it before it joins a render.
# A new writer fails this test until it is classified (C, fourth review).
W = re.compile(r"\.(to_csv|savefig|write_text|write_bytes|to_pickle|to_parquet)\(|np\.save(z|z_compressed)?\(|save_fd\(|"
               r"pd\.to_pickle\(")
SHARE = re.compile(r"M:/MICROSCOPE|oo\.root\(|PathResolver\(\)\.root\(|_run_dir\(\)|encoding_root\(|_enc_dir\(|"
                   r"root\(\"(cohort_out|orofacial|dlc|figures_working|behavior_out|labcams)")
LIT = re.compile(r"[MN]:[/\\\\]+MICROSCOPE|C:/Users/Sabatini")
ROUTED = re.compile(r"run_mode|output_dir\(|routed_output|lp_licks\.output_tag|require_default")


RESOLVED = {
    "scripts/recovered_20261103/fig_goal_epochs.py",
    "scripts/recovered_20261103/fig_goal_summary.py",
    "scripts/rest_migration/lick_bout_structure.py",
    "scripts/rest_migration/quit_prodrome.py",
    "wfield_local/animal_weights.py",
    "wfield_local/behavior_clip_deck.py",
    "wfield_local/behavior_events.py",
    "wfield_local/crossday_intensity.py",
    "wfield_local/dlc_frames.py",
    "wfield_local/dlc_hard_frames.py",
    "wfield_local/dlc_iti_frames.py",
    "wfield_local/epochs.py",
    "wfield_local/frozen_relabel.py",
    "wfield_local/histology_area.py",
    "wfield_local/histology_map/__main__.py",
    "wfield_local/joint_locanmf.py",
    "wfield_local/locanmf_contralateral.py",
    "wfield_local/orofacial_clean.py",
    "wfield_local/orofacial_cohort.py",
    "wfield_local/orofacial_heatmaps.py",
    "wfield_local/orofacial_jaw_lateral.py",
    "wfield_local/orofacial_outputs.py",
    "wfield_local/orofacial_w2.py",
    "wfield_local/paint_exclusion.py",
    "wfield_local/pose_windows.py",
    "wfield_local/publish_basis.py",
    "wfield_local/rebuild_root.py",
    # (wfield_local/recovery_trajectory.py: ROUTED since review 8B B1b, lp_licks.output_dir)
    "wfield_local/spout_behavior.py",
    "wfield_local/stray_licks.py",
    "wfield_local/sync_audit.py",
}
LITERAL = {
    "scripts/cohort2_qc/channel_drift.py",
    "scripts/cohort2_qc/cohort1_impact.py",
    "scripts/cohort2_qc/compare_hemo_variants_1029.py",
    "scripts/cohort2_qc/epileptiform_gallery.py",
    "scripts/cohort2_qc/epileptiform_scan.py",
    "scripts/cohort2_qc/fast_events.py",
    "scripts/cohort2_qc/indicator_compare.py",
    "scripts/cohort2_qc/indicator_kinetics_spread.py",
    "scripts/cohort2_qc/iso_check.py",
    "scripts/cohort2_qc/masked_vs_adopted_1029.py",
    "scripts/cohort2_qc/native_maps_1029.py",
    "scripts/cohort2_qc/quiet_f0_noise.py",
    "scripts/cohort2_qc/quiet_modes.py",
    "scripts/cohort2_qc/quiet_task_events.py",
    "scripts/cohort2_qc/quiet_task_events_followup.py",
    "scripts/cohort2_qc/running_sign_check.py",
    "scripts/cohort2_qc/seizure_events_video.py",
    "scripts/cohort2_qc/session_qc.py",
    "scripts/cohort2_qc/slowband_behaviour.py",
    "scripts/cohort2_qc/spout_compare_1029.py",
    "scripts/cohort2_qc/spout_maps.py",
    "scripts/cohort2_qc/swapmask_1029.py",
    "scripts/cohort2_qc/swapmask_cohort1.py",
    "scripts/cohort2_qc/swapmask_compare_quiet.py",
    "scripts/drift_check/drift_cutoff_check.py",
    "scripts/imaging_page/s_per_animal.py",
    "scripts/locanmf_refit_test/refit_vs_projection.py",
    "scripts/rebuild_figs/fig_behaviour_hits.py",
    "scripts/rebuild_figs/fig_decoding_sessions.py",
    "scripts/rebuild_figs/fig_enl_move.py",
    "scripts/rebuild_figs/fig_frozen_refit_pooled.py",
    "scripts/rebuild_figs/fig_goal_epochs.py",
    "scripts/rebuild_figs/fig_s_per_animal.py",
    "scripts/rebuild_figs/fig_state_position_sessions.py",
    "scripts/rebuild_figs/fig_state_ps94_diagnostic.py",
    "wfield_local/dlc_prelabel.py",
    "wfield_local/frozen_models.py",
    "wfield_local/locanmf_event_master.py",
    "wfield_local/locanmf_lick_cards.py",
    "wfield_local/locanmf_lick_pool.py",
    "wfield_local/lp_multiview.py",
    "wfield_local/o2_inference.py",
}


def _scan():
    out = {}
    for p in sorted(list((REPO / "scripts").rglob("*.py")) + list((REPO / "wfield_local").rglob("*.py"))):
        rel = p.relative_to(REPO).as_posix()
        if "/archive/" in rel or p.parent == REPO / "scripts":
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        if W.search(t) and SHARE.search(t) and not ROUTED.search(t):
            out[rel] = "LITERAL" if LIT.search(t) else "RESOLVED"
    return out


def test_subfolder_and_package_writers_are_classified():
    got = _scan()
    res = {k for k, v in got.items() if v == "RESOLVED"}
    lit = {k for k, v in got.items() if v == "LITERAL"}
    assert res == RESOLVED and lit == LITERAL, {"new_resolved": sorted(res - RESOLVED), "new_literal": sorted(lit - LITERAL),
                                                "stale": sorted((RESOLVED | LITERAL) - set(got))}


SEED_UNDER_FENCE = r"""
import os, json
from pathlib import Path
from wfield_local import render_profile
render_profile.apply("%s")                                       # installs the fence
os.environ["WIDEFIELD_REBUILD_ROOT"] = %r
from wfield_local import rebuild_root as rr
orig, dst = Path(%r), Path(%r)
src, other = orig / "epoch_boundaries.json", orig / "not_a_seed.json"
f0 = rr._fence
rr._fence = lambda key: dict(f0(key), deny_read=list(f0(key)["deny_read"]) + [rr._norm(orig)])   # orig = a default product
rr.seed_plan = lambda: [("cohort_out/epoch_boundaries.json", src, dst)]
seen = {}
c0 = rr._seed_copy
def probe(plan, rec, dry_run):
    try:
        open(other).read()
        seen["other"] = "READ"
    except rr.RebuildFenceError:
        seen["other"] = "REFUSED"
    return c0(plan, rec, dry_run)
rr._seed_copy = probe
try:
    open(src).read()
    seen["before"] = "READ"
except rr.RebuildFenceError:
    seen["before"] = "REFUSED"
rec = rr.init()
try:
    open(src).read()
    seen["after"] = "READ"
except rr.RebuildFenceError:
    seen["after"] = "REFUSED"
seen["copied"] = dst.exists()
print(json.dumps(seen))
"""


def test_init_seeds_under_an_applied_profile_and_only_the_seed_sources(tmp_path):
    """Coordinator 11-04: `rebuild_root init --profile ...` applies the profile (the fence) BEFORE init; init reads
    exactly the seed_plan() sources (a scoped allow-list), never any other default derived product, also during init."""
    orig = tmp_path / "orig"
    orig.mkdir()
    (orig / "epoch_boundaries.json").write_text('{"boundaries": {}}')
    (orig / "not_a_seed.json").write_text("{}")
    rb = tmp_path / "rb"
    dst = rb / "labcams" / "epoch_boundaries.json"
    r = _run(SEED_UNDER_FENCE % (PROFILE, str(rb), str(orig), str(dst)), {}, tmp_path)
    out = r.stdout.strip().splitlines()
    assert out, r.stderr[-3000:]
    import json
    seen = json.loads(out[-1])
    assert seen == {"before": "REFUSED", "other": "REFUSED", "after": "REFUSED", "copied": True}, (seen, r.stderr[-2000:])
    assert dst.read_text() == '{"boundaries": {}}'


SEED_CLI = r"""
import os, runpy, sys
from pathlib import Path
from wfield_local import rebuild_root as rr          # the IMPORTED module (the one the audit hook reads)
orig, dst = Path(%r), Path(%r)
f0 = rr._fence
rr._fence = lambda key: dict(f0(key), deny_read=list(f0(key)["deny_read"]) + [rr._norm(orig)])
rr.seed_plan = lambda: [("cohort_out/epoch_boundaries.json", orig / "epoch_boundaries.json", dst)]
_apply = __import__("wfield_local.render_profile", fromlist=["apply"]).apply
def apply(name, *a, **k):
    d = _apply(name, *a, **k)
    os.environ["WIDEFIELD_REBUILD_ROOT"] = %r            # the profile's root, pointed at the temp root
    return d
sys.modules["wfield_local.render_profile"].apply = apply
sys.argv = ["rebuild_root", "init", "--profile", "%s"]
try:
    runpy.run_module("wfield_local.rebuild_root", run_name="__main__")   # exactly `python -m wfield_local.rebuild_root`
except SystemExit as ex:
    print("EXIT", ex.code)
print("COPIED", dst.exists())
"""


def test_init_cli_as_main_seeds_under_the_fence(tmp_path):
    """Coordinator 11-04: `python -m wfield_local.rebuild_root init` (runpy: the file runs as __main__, a SECOND copy of
    the module) must seed through the IMPORTED module's allow-list, the one the installed hook reads."""
    orig = tmp_path / "orig"
    orig.mkdir()
    (orig / "epoch_boundaries.json").write_text('{"boundaries": {}}')
    rb = tmp_path / "rb"
    dst = rb / "labcams" / "epoch_boundaries.json"
    r = _run(SEED_CLI % (str(orig), str(dst), str(rb), PROFILE), {}, tmp_path)
    assert "EXIT 0" in r.stdout and "COPIED True" in r.stdout, r.stdout[-1500:] + r.stderr[-2500:]
