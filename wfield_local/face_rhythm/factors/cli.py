"""The command line (`python scripts/face_rhythm_factors.py <stage>` on the desktop, `python <set>/fr_factors.py <stage>` on O2)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import TRIAL_DS
from .files import set_dir
from .fit import cmd_build, cmd_fit, cmd_select
from .o2 import cmd_plan
from .project import cmd_project
from .report import cmd_report
from .validate import cmd_fitepoch, cmd_fitsession, cmd_loso, cmd_splithalf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=sys.modules[__package__].__doc__,      # the package docstring: stages
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "build", "fit", "loso", "splithalf", "select", "project", "fitsession",
                                    "fitepoch", "report"])
    ap.add_argument("--bundle", default="fr_cam4_20261021")
    ap.add_argument("--set", default="master_pre_v1")
    ap.add_argument("--dir", type=Path, default=None, help="factor-set folder (default: this file's folder on O2, "
                                                           "or <bundle>/factors/<set> on the desktop)")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--face", type=Path, default=None, help="report: master face image under the spatial maps")
    ap.add_argument("--absolute", action="store_true", help="report: time courses as absolute expression (x D)")
    ap.add_argument("--match", type=Path, default=None,
                    help="report: another factor-set folder (master.npz [+ labels.json]): label each factor with its "
                         "best match there (e.g. the named PS92 preview)")
    ap.add_argument("--trials-per-animal", type=int, default=60)
    ap.add_argument("--session-trials", type=int, default=60)
    ap.add_argument("--views", nargs="*", default=None,
                    help="plan: MULTI-VIEW set -- extra cameras as cam=bundle (e.g. cam1=fr_cam1_20261101); points "
                         "stacked with the --bundle camera's (wfield_local/face_rhythm_multiview.py)")
    ap.add_argument("--arrival", action="store_true",
                    help="plan: add PRE-CUE 5 Hz rows per fit trial: spout arrival (strobe -1..+2 s) + cue -4..-2 s")
    ap.add_argument("--even-days", action="store_true",
                    help="plan: spread every fit draw evenly over each animal's pre-stroke recording DAYS")
    ap.add_argument("--run-share", type=float, default=None,
                    help="plan: share of the between-trial rows from RUNNING bins, the rest STILL (None = uniform)")
    ap.add_argument("--trial-ds", type=int, default=TRIAL_DS,
                    help="plan: 25 Hz trial bins per FIT row (4 -> 2x the trials for the same tensor; session rows stay 2)")
    ap.add_argument("--between-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranks", type=int, nargs="*", default=[6, 8, 10, 12, 14, 16])
    ap.add_argument("--inits", type=int, default=8, help="random starts per rank (8: per-factor stability readable)")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--point-pcs", type=int, default=20, help="plan: PCs of the dense-position basis")
    ap.add_argument("--points-bins", type=int, default=20000, help="plan: 25 Hz bins per animal for its fit")
    ap.add_argument("--user", default="ps150")
    ap.add_argument("--animals", nargs="*", default=None, help="plan: only these animals' tasks (a pilot)")
    ap.add_argument("--with-outputs", action="store_true", help="plan: only tasks whose outputs are on the share")
    ap.add_argument("--role", default="master", choices=("master", "pilot"),
                    help="plan: 'pilot' for a restricted set (stamped into plan + master.npz; never the cohort master)")
    ap.add_argument("--no-loso", action="store_true", help="plan: no leave-one-pre-session-out jobs")
    ap.add_argument("--no-loao", action="store_true", help="fit: skip leave-one-animal-out (a plain rank sweep)")
    ap.add_argument("--split-inits", type=int, default=2, help="splithalf: random starts per half")
    ap.add_argument("--loso-basis", action="store_true",
                    help="project: a PRE task on its left-out-day basis -> proj_loso/ (FR's own scoring only)")
    ap.add_argument("--own-inits", type=int, default=3, help="fitsession: random starts of the session's own fit")
    ap.add_argument("--own-dir", default="own", help="fitsession: output folder in the set (own8 = 8-start rerun, 10-24)")
    ap.add_argument("--epoch-rows", type=int, default=11000,
                    help="fitepoch: rows per pooled animal x epoch fit (~2 sessions; equal for every fit)")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        cmd_plan(a.bundle, a.set, a.trials_per_animal, a.session_trials, a.between_frac, a.seed, a.ranks, a.inits,
                 a.n_iter, a.user, a.point_pcs, a.points_bins, a.animals, a.with_outputs, a.role, not a.no_loso,
                 a.trial_ds, a.views, a.run_share, a.arrival, a.even_days)
        return 0
    sd = a.dir
    if sd is None:
        if (set_dir() / "plan.json").exists():
            sd = set_dir()
        else:
            from wfield_local import o2_inference as oi
            from wfield_local.paths import PathResolver
            sd = oi.bundle_dir(PathResolver(), a.bundle) / "factors" / a.set
    task = a.task or (int(os.environ["SLURM_ARRAY_TASK_ID"]) if a.cmd in ("project", "fitsession", "fitepoch", "loso") else None)
    if a.cmd == "project" and a.loso_basis:
        cmd_project(sd, task, loso=True)
        return 0
    {"build": lambda: cmd_build(sd), "fit": lambda: cmd_fit(sd, a.rank, not a.no_loao), "select": lambda: cmd_select(sd, a.rank),
     "project": lambda: cmd_project(sd, task),
     "fitepoch": lambda: cmd_fitepoch(sd, task, a.own_inits, a.epoch_rows), "fitsession": lambda: cmd_fitsession(sd, task, a.own_inits, a.own_dir),
     "loso": lambda: cmd_loso(sd, task), "splithalf": lambda: cmd_splithalf(sd, a.rank, a.split_inits),
     "report": lambda: cmd_report(sd, a.face, a.absolute, a.match)}[a.cmd]()
    return 0
