"""Library code must not import from `scripts/` (code review 11-04, modularity item 1).

WHY. `analysis_kit.session_behavior` -- the shared quit / trial definition -- imported `session_trials` /
`session_quit` from `scripts/rest_migration/`, so archiving a "one-off" script would have broken production numbers
(they moved to `wfield_local/engagement.py` on 2026-11-05). `scripts/README.md` says scripts depend on the library, never
the reverse.

KNOWN OFFENDERS still to move, each owned by another window (pose / encoder / FaceRhythm files; move with a shim and a
pinned check, then delete the entry). The set must match exactly: a NEW offender fails, and so does a stale entry.
"""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

KNOWN = {
    "wfield_local/combined_encoding.py",     # pose_kinematics_demo, movement_encoding_session, session_poses
    "wfield_local/face_rhythm_readouts.py",  # face_rhythm_mesh_qc
    "wfield_local/orofacial_cohort.py",      # pose_kinematics_demo (canonical DAQ contacts, split_precue_by_peak)
    "wfield_local/orofacial_outputs.py",     # pose_kinematics_demo
    "wfield_local/pose3d_side.py",           # apparatus_mouth._rig, pose3d.PROD / _cgroup
}


def _imports_scripts(path: Path) -> bool:
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom) and n.module and (n.module == "scripts" or n.module.startswith("scripts.")):
            return True
        if isinstance(n, ast.Import) and any(a.name == "scripts" or a.name.startswith("scripts.") for a in n.names):
            return True
    return False


def test_library_imports_no_scripts_beyond_the_known_list():
    found = {p.relative_to(REPO).as_posix() for p in (REPO / "wfield_local").rglob("*.py") if _imports_scripts(p)}
    assert found == KNOWN, {"new": sorted(found - KNOWN), "fixed (remove from KNOWN)": sorted(KNOWN - found)}


def test_engagement_names_are_one_object_everywhere():
    """The scripts re-export the moved functions: one definition, so the old import paths cannot drift."""
    from scripts.rest_migration import engagement_decomposition as ed
    from scripts.rest_migration import quit_point as qp
    from wfield_local import engagement as eng
    assert ed.session_trials is eng.session_trials and ed.near_codes is eng.near_codes
    assert qp.session_quit is eng.session_quit and qp._sharpness is eng._sharpness
    assert qp._early_rate is eng._early_rate and qp.EARLY_S == eng.EARLY_S
