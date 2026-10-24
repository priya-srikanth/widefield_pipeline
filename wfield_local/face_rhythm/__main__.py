"""FaceRhythm analysis catalogue: what each analysis answers, how long it takes, where its figures land.

    python -m wfield_local.face_rhythm list                 # every analysis, grouped by stage
    python -m wfield_local.face_rhythm show peak            # one entry in full (+ the script's own --help)
    python -m wfield_local.face_rhythm run peak --split side [--plot-only]   # run it (arguments pass through)

Environments: the `fr` conda env does not install the repo, so set ``PYTHONPATH=<repo>`` (this module adds the repo to
the script's path when it runs one); `push` needs the `locanmf` env (behaviour loaders). Outputs land under
``<dlc>/face_rhythm/cam4/readouts/<set>/`` unless an entry says otherwise. Guide: docs/FACE_RHYTHM_GUIDE.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]


class Analysis(NamedTuple):
    script: str            # scripts/<script>.py
    stage: str             # setup / factors / readouts / statistics
    question: str          # what it answers, in the words a user would ask it
    outputs: str           # where the figures and tables land (readouts/<set>/... unless absolute)
    runtime: str           # measured on the analysis desktop, 10-24
    cache: str = ""        # how to redraw without recomputing, if it can
    findings: str = ""     # FINDINGS section that reports it


CATALOGUE: dict[str, Analysis] = {
    # ---- setup: before any factor exists (run once per camera)
    "prep": Analysis("face_rhythm_prep", "setup", "Per-session reference faces + LP landmarks for registration",
                     "face_rhythm/<cam>/", "minutes per session"),
    "register": Analysis("face_rhythm_register", "setup", "Warp every session's face onto ONE master face",
                         "face_rhythm/<cam>/register/", "~10 min"),
    "overlay": Analysis("face_rhythm_overlay", "setup", "Spouts / nose / jaw / tongue of every session on the master, "
                        "to draw the master ROI", "face_rhythm/<cam>/", "~2 min"),
    "warp-rois": Analysis("face_rhythm_warp_rois", "setup", "Move ROIs between a session and the master face",
                          "face_rhythm/<cam>/rois/", "seconds"),
    "mesh-qc": Analysis("face_rhythm_mesh_qc", "setup", "Is the master mesh, warped to every session, sound?",
                        "face_rhythm/cam4/rois_qc_*/", "~5 min"),
    "param-qc": Analysis("face_rhythm_param_qc", "setup", "Which tracking parameters? (clips + factor QC of the "
                         "O2 parameter test)", "o2 bundle param_test/", "~15 min"),
    # ---- factors: the frozen master and the per-session fits (O2; the desktop runs plan / select / report)
    "factors": Analysis("face_rhythm_factors", "factors", "Fit / select / project the frozen pre-stroke factor set; "
                        "per-session (fitsession) and per-epoch (fitepoch) own fits; the factor report",
                        "o2/<bundle>/factors/<set>/", "O2: hours (array jobs)", findings="2, 3, 4a"),
    "clips": Analysis("face_rhythm_factor_clips", "factors", "Video clips of each factor at its strongest moments "
                      "(for naming)", "face_rhythm/cam4/factor_clips/<set>/", "~20 min"),
    "validate": Analysis("face_rhythm_validate", "factors", "What is each factor? (licks by direction / time / bout "
                         "position, running)", "face_rhythm/cam4/validation/<set>/", "~15 min", findings="4a"),
    # ---- readouts: figures from the projections
    "readouts": Analysis("face_rhythm_readouts", "readouts", "Amplitudes, EVR own vs frozen, movement maps, task vs "
                         "non-task, severity -- the session table the EVR statistics read",
                         "readouts/<set>/", "~8 min", findings="6-master, 6d"),
    "evr": Analysis("face_rhythm_evr_epochs", "statistics", "Does the pre-stroke vocabulary fit worse after the "
                    "stroke? (EVR ratio / gap / error vs own pre, by epoch)", "readouts/<set>/evr_epochs.png",
                    "seconds (needs readouts)", findings="6-master"),
    "peak": Analysis("face_rhythm_peak", "statistics", "How does each factor's peak amplitude change, and WHEN after "
                     "the cue is it expressed? (--split none / side / position)", "readouts/<set>/peak/",
                     "3-6 min per split", "--plot-only (data/peak_cache*.pkl)", "5d, 5g"),
    "per-lick": Analysis("face_rhythm_lick_epochs", "statistics", "Fewer licks or different licks? (per-lick "
                         "expression and licks per trial, by epoch)", "face_rhythm/cam4/validation/<set>/lick_epochs/",
                         "~15 min", "--stats-only (data/sessions.csv)", "5b-master"),
    "asym": Analysis("face_rhythm_asym_aligned", "statistics", "Does the contralesional half of the face move less, "
                     "aligned to the cue / each lick / bout onset?", "readouts/<set>/asym_aligned/", "~6 min",
                     "--plot-only (data/asym_traces.pkl)", "5e"),
    "maps": Analysis("face_rhythm_aligned_maps", "statistics", "WHERE on the face does movement change, per spout "
                     "and alignment? (patch maps + contra / ipsi halves)", "readouts/<set>/aligned_maps/",
                     "~8 min (needs asym's map cache)", findings="5f"),
    "bout-split": Analysis("face_rhythm_bout_split", "statistics", "Why is close_R asymmetric before a bout starts? "
                           "(bout classes)", "readouts/<set>/asym_aligned/bout_split/", "~10 min",
                           "--plot-only (data/bout_split_traces.npz)", "5f"),
    "position-maps": Analysis("face_rhythm_position_maps", "statistics", "Per-position face speed maps, epoch vs pre",
                              "readouts/<set>/position_maps/", "~1 min", findings="5e"),
    "match": Analysis("face_rhythm_match", "statistics", "Do the factors change SHAPE? (own fits Hungarian-matched to "
                      "the master: spatial / spectral angles, loading asymmetry)", "readouts/<set>/matching/",
                      "~1 min", findings="6c"),
    "nontask": Analysis("face_rhythm_nontask_state", "statistics", "Is the non-task drop just less activity? "
                        "(between-trial expression by running / rest)", "readouts/<set>/nontask_state/", "~2 min",
                        "--plot-only (data/nontask_state_sessions.csv)", "6d"),
    "push": Analysis("face_rhythm_push_check", "statistics", "Does the pre-stroke basis describe the face at treadmill "
                     "pushes? (locanmf env)", "readouts/<set>/push_check/", "~1 h",
                     "--plot-only (data/push_check_traces.npz)", "6b"),
}
STAGES = ("setup", "factors", "readouts", "statistics")


def _list() -> None:
    for st in STAGES:
        print(f"\n{st.upper()}")
        for name, a in CATALOGUE.items():
            if a.stage == st:
                extra = f"  [{a.cache}]" if a.cache else ""
                print(f"  {name:14s} {a.question}\n  {'':14s} -> {a.outputs}  ({a.runtime}){extra}")
    print("\nrun one: python -m wfield_local.face_rhythm run <name> [its arguments]; details: show <name>")


def _show(name: str) -> None:
    a = CATALOGUE[name]
    print(f"{name}: {a.question}\n  script   scripts/{a.script}.py\n  outputs  {a.outputs}\n  runtime  {a.runtime}")
    if a.cache:
        print(f"  redraw   {a.cache}")
    if a.findings:
        print(f"  FINDINGS docs/FACE_RHYTHM_FINDINGS.md section {a.findings}")
    _run(name, ["--help"])


def _run(name: str, args: list[str]) -> int:
    """Run the script as its OWN process (same interpreter, repo on PYTHONPATH): the per-session scripts fan out over
    spawned workers that look their worker function up in ``__main__`` by name, which only works when the script IS
    ``__main__`` -- running it in-process (runpy) would break every parallel analysis."""
    import os
    import subprocess
    script = REPO / "scripts" / f"{CATALOGUE[name].script}.py"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p))
    return subprocess.call([sys.executable, str(script), *args], env=env)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "list"):
        print(__doc__) if argv[:1] in (["-h"], ["--help"]) else _list()
        return 0
    cmd, *rest = argv
    if cmd == "show" and rest and rest[0] in CATALOGUE:
        _show(rest[0])
        return 0
    if cmd == "run" and rest and rest[0] in CATALOGUE:
        return _run(rest[0], rest[1:])
    print(f"unknown: {' '.join(argv)!r}. Analyses: {', '.join(CATALOGUE)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
