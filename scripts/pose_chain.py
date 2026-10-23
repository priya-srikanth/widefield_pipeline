"""The pose / 3-D / cam1 chain in ONE command, in dependency order (STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF §4; Priya
10-21: "use saved arrays, caches, parallelization ... the things that make this most efficient and less redundant").

    python -m scripts.pose_chain                              # after an O2 copy-back: new cam1 sessions -> every output
    python -m scripts.pose_chain --from kinematics            # resume at a step
    python -m scripts.pose_chain --only jaw_onsets phase      # just these steps
    python -m scripts.pose_chain --dry-run                    # print the commands

Steps (each a separate process, the right conda env; per-session steps are CACHED on their input files
(`wfield_local.result_cache`), so after a batch only the new sessions are computed and the cohort tables re-assembled):
  collect        o2_inference collect --missing-only (+ binary windows.npz sidecars, `pose_windows`)      locanmf
  pose3d         3-D production for sessions without 3-D frames yet                                       dlc
  kinematics     pose3d_kinematics (cam4 licks), --licks combined, --licks all                                   locanmf
  cam1_licks     cam1_licks run (sessions given, or those without events yet) + cam1_incomplete_licks    locanmf
  cue_traces     pose3d_cue_traces --align cue, then first_lick                                           locanmf
  spout_error    pose3d_spout_error                                                                       locanmf
  figures        pose3d_summary_fig, pose3d_jaw_crosscheck   (AFTER spout_error: the summary reads it)     locanmf
  phase          pose3d_phase                                                                              locanmf
  cohort_checks  first_lick_latency, pose3d_crosscheck, pose3d_vs_cam4 (cohort tables that read the steps above;
                 10-23: last run 10-19 / 10-20 on PS92 + part of PS93 only, so they were added here)             locanmf
  jaw_onsets     jaw_onsets (WATCHED by the encoding window: written last, once)                          locanmf
--jobs (default 6) is per step: sessions in parallel, collect included.
Each step's log goes to <dlc>/session_poses/pose_chain_logs/<stamp>_<step>.log; the run stops at the first failure.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ENVS = {"locanmf": r"C:\Users\SabatiniLab\miniconda3\envs\locanmf\python.exe",
        "dlc": r"C:\Users\SabatiniLab\miniconda3\envs\dlc\python.exe"}
BUNDLE = "lp_cam1_ctx_prod_20261019"
STEPS = ("collect", "pose3d", "kinematics", "cam1_licks", "cue_traces", "spout_error", "figures", "phase",
         "cohort_checks", "jaw_onsets")


def _pending(rv) -> dict:
    """Sessions that have cam1 windows but no 3-D frames / no cam1 lick events yet (curated)."""
    from wfield_local import analysis_kit as ak
    sp = Path(rv.root("dlc")) / "session_poses"
    cam1 = sorted(d.name[:-len("_cam1_full")] for d in sp.glob("PS*_cam1_full") if (d / "windows_LP.csv").exists())
    cam1 = [s for s in cam1 if s in ak.curated_ids()]
    return {"pose3d": [s for s in cam1 if not (sp / "pose3d" / f"{s}_frames.csv.gz").exists()],
            "cam1_licks": [s for s in cam1 if not (sp / "cam1_licks" / f"{s}_events.csv").exists()]}


def commands(rv, jobs: int, sessions: list[str] | None, step: str | None = None) -> dict:
    # the pending scan walks the share (minutes when it is busy): only the two steps that need it trigger it
    need = step in (None, "pose3d", "cam1_licks") and not sessions
    pend = _pending(rv) if need else {"pose3d": [], "cam1_licks": []}
    p3 = sessions or pend["pose3d"]
    cl = sessions or pend["cam1_licks"]
    arg = lambda ss: [s.replace("_", ":", 1) for s in ss]  # noqa: E731
    J = ["--jobs", str(jobs)]
    return {
        "collect": [("locanmf", ["-m", "wfield_local.o2_inference", "collect", "--name", BUNDLE, "--missing-only", *J]),
                    ("locanmf", ["-m", "wfield_local.pose_windows", "--jobs", str(jobs)])],
        "pose3d": [("dlc", ["-m", "scripts.pose3d", "production", *arg(p3), *J])] if p3 else [],
        "kinematics": [("locanmf", ["-m", "scripts.pose3d_kinematics", *J]),
                       ("locanmf", ["-m", "scripts.pose3d_kinematics", "--licks", "combined", *J]),
                       ("locanmf", ["-m", "scripts.pose3d_kinematics", "--licks", "all", *J])],
        "cam1_licks": ([("locanmf", ["-m", "scripts.cam1_licks", "run", *arg(cl), *J])] if cl else [])
        + [("locanmf", ["-m", "scripts.cam1_incomplete_licks"])],
        "cue_traces": [("locanmf", ["-m", "scripts.pose3d_cue_traces", "--align", "cue", *J]),
                       ("locanmf", ["-m", "scripts.pose3d_cue_traces", "--align", "first_lick", *J])],
        "spout_error": [("locanmf", ["-m", "scripts.pose3d_spout_error", *J])],
        "figures": [("locanmf", ["-m", "scripts.pose3d_summary_fig"]),
                    ("locanmf", ["-m", "scripts.pose3d_jaw_crosscheck"])],
        "phase": [("locanmf", ["-m", "scripts.pose3d_phase", *J])],
        "cohort_checks": [("locanmf", ["-m", "scripts.first_lick_latency", *J]),
                          ("locanmf", ["-m", "scripts.pose3d_crosscheck"]),
                          ("locanmf", ["-m", "scripts.pose3d_vs_cam4"])],
        "jaw_onsets": [("locanmf", ["-m", "scripts.jaw_onsets", *J])],
    }


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    ap.add_argument("--only", nargs="*", choices=STEPS, default=None)
    ap.add_argument("--sessions", nargs="*", default=None, metavar="ANIMAL_YYYYMMDD",
                    help="for pose3d / cam1_licks (default: the sessions still missing their output)")
    # 10-21 evening: 3 -> 6 (Priya: the box had plenty of free CPU / RAM; collect was serial and the slowest step).
    # Use --jobs 3 when the other windows need the box.
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rv = PathResolver()
    steps = a.only or list(STEPS[STEPS.index(a.start):])
    sessions = [s.replace(":", "_", 1) for s in a.sessions] if a.sessions else None
    logs = Path(rv.root("dlc")) / "session_poses" / "pose_chain_logs"
    stamp = time.strftime("%Y%m%d_%H%M")
    repo = Path(__file__).resolve().parents[1]
    once = commands(rv, a.jobs, sessions) if a.dry_run else None
    for step in steps:
        # re-resolved per step in a real run: earlier steps create what later ones need
        cmds = (once or commands(rv, a.jobs, sessions, step))[step]
        for env, args in cmds:
            line = f"{ENVS[env]} {' '.join(args)}"
            print(f"=== {time.strftime('%H:%M')} {step}: {' '.join(args)}", flush=True)
            if a.dry_run:
                continue
            logs.mkdir(parents=True, exist_ok=True)
            log = logs / f"{stamp}_{step}.log"
            t0 = time.time()
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"$ {line}\n")
                fh.flush()
                r = subprocess.run([ENVS[env], "-W", "ignore", *args], cwd=repo, stdout=fh, stderr=subprocess.STDOUT,
                                   env={**__import__("os").environ, "PYTHONPATH": str(repo)})
            tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            print(f"    {time.time() - t0:.0f} s, exit {r.returncode}; log {log.name}")
            for t in tail:
                print(f"    | {t[:160]}")
            if r.returncode:
                print(f"STOPPED at {step}")
                return r.returncode
    print("=== chain done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
