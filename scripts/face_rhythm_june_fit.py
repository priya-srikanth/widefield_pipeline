"""June-only COHORT factor fit on O2 (Priya 11-09: "yes" -- test the lick_right SHAPE change against June as a ~2-month
pre-stroke drift floor). Same procedure as the production cohort epoch fits (`fr_factors_lib.validate._cohort_fit`:
5500 rows per animal drawn evenly over its sessions, each animal / its own master scale, 8 HALS starts at the master's
rank, best kept, columns matched to the master), on the 12 June sessions:
  task 1  COHORT_june          all four animals
  task 2-5 LOAO-<animal>_june  leave one animal out (jackknife, as the production cohort fits)
The rows come from each June session's ``session_rows``, drawn exactly as `fr_factors.py plan` draws them for the
production set (same params: session_trials, between_frac, run_share, pre-cue arrival rows, trial_ds; same seed rule
default_rng([seed, crc32(key)])). Output: ``<june set>/own_epoch_cohort/<tag>_june.npz`` (same fields as production
``own_epoch_cohort/<tag>_<group>.npz``) -> angle vs the production COHORT pre / preEarly fits with
`face_rhythm_drift_floor.py`'s matching.

    desktop  python scripts/face_rhythm_june_fit.py plan          (adds session_rows to the June set's plan.json,
                                                                   deploys the code, writes run_fitjune.sbatch)
    O2       python <june set>/fr_june_fit.py fit --task N
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAGS = ("COHORT", "LOAO-PS92", "LOAO-PS93", "LOAO-PS94", "LOAO-PS95")
JUNE_BUNDLE, PROD, SET = "fr_cam4_june_20261108", "fr_cam4_20261021", "master_pre_v1"
USER = "ps150"


def cmd_plan() -> None:
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(HERE.parent))
    from wfield_local import o2_inference as oi
    from wfield_local import writeguard
    from wfield_local.face_rhythm.factors.files import _sidecar_module
    from wfield_local.face_rhythm.factors.o2 import _sbatch, deploy_code
    from wfield_local.face_rhythm.factors.sample import between_states, draw_rows, session_candidates, trial_strobes
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    bd = oi.bundle_dir(rv, JUNE_BUNDLE)
    sd = bd / "factors" / f"{SET}_june"
    writeguard.assert_writable(sd)
    prod = json.loads((oi.bundle_dir(rv, PROD) / "factors" / SET / "plan.json").read_text())["params"]
    pl = json.loads((sd / "plan.json").read_text())
    if "session_rows" in pl:
        sys.exit("session_rows already in the June plan -- not redrawn (delete them on purpose to redraw)")
    frt = _sidecar_module("face_rhythm_timing")
    T = pd.read_csv(bd / "tasks.tsv", sep="\t", dtype=str).set_index("key")
    run_share, arrival = prod["run_share"], prod.get("precue_rows") is not None
    rows = {}
    for t in pl["tasks"]:
        k, r = t["key"], T.loc[t["key"]]
        tm = frt.load(bd, k)
        assert tm is not None, f"{k}: no timing (run o2_face_rhythm.py timing first)"
        st = between_states(rv, r.animal, r.date, tm["t5"][:int(r.n_frames) // 50]) if run_share is not None else None
        zt = np.load(bd / "trials" / f"{k}.npz")
        sb = trial_strobes(rv, r.animal, r.date, zt["trial_id"], zt["cue_daq_s"]) if arrival else None
        cand = session_candidates(k, zt, int(r.n_frames), tm, st, sb)
        rows[k] = draw_rows(*cand, prod["session_trials"], prod["between_frac"],
                            np.random.default_rng([prod["seed"], zlib.crc32(k.encode())]), run_share=run_share)
        n_tr = sum(map(len, rows[k]["trials"].values()))
        print(f"  {k}: {n_tr} trials, {sum(map(len, rows[k].get('arrival', {}).values()))} pre-cue bins, "
              f"{sum(map(len, rows[k]['between'].values()))} between bins")
    pl["session_rows"] = rows
    pl["params"] = prod
    pl["june_fit"] = {"tags": list(TAGS), "rows_per_animal": 5500, "inits": 8, "source": "scripts/face_rhythm_june_fit.py"}
    (sd / "plan.json").write_text(json.dumps(pl, indent=1), encoding="utf-8", newline="\n")
    deploy_code(sd)
    (sd / "fr_june_fit.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    (sd / "logs").mkdir(exist_ok=True)
    b = f"{oi.scratch(USER)}/o2/{JUNE_BUNDLE}/factors/{SET}_june"
    sb_txt = _sbatch(b, f"-J frf_june -c 16 --mem=96G -t 0-12:00:00 --array=1-{len(TAGS)}", "x")
    sb_txt = sb_txt.replace(f"python {b}/fr_factors.py x", f"python {b}/fr_june_fit.py fit --task $SLURM_ARRAY_TASK_ID")
    (sd / "run_fitjune.sbatch").write_text(sb_txt, encoding="utf-8", newline="\n")
    share = "/n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2"
    (sd / "COMMANDS_JUNE_FIT.md").write_text(f"""# June-only cohort factor fit (11-09)

Transfer node (`ssh {USER}@transfer.rc.hms.harvard.edu`): the set folder to scratch (the June `out/` is already there
from the June run; check it holds the 12 sessions)
```
ls {oi.scratch(USER)}/o2/{JUNE_BUNDLE}/out/*_summary.json | wc -l     # 12
rsync -ah {share}/{JUNE_BUNDLE}/factors/{SET}_june/ {b}/
```
Login node
```
sbatch {b}/run_fitjune.sbatch
```
5 tasks (COHORT + 4 leave-one-animal-out), 16 CPUs / 96 GB each, ~1-3 h (as the production cohort fits). Copy back:
```
rsync -ah {b}/own_epoch_cohort/ {share}/{JUNE_BUNDLE}/factors/{SET}_june/own_epoch_cohort/
```
""", encoding="utf-8", newline="\n")
    print("plan ->", sd)


def cmd_fit(task: int) -> None:
    sd = HERE
    sys.path.insert(0, str(sd))
    from fr_factors_lib.files import load_plan
    from fr_factors_lib.validate import _cohort_fit
    pl = load_plan(sd)
    tag = TAGS[task - 1]
    by = {}
    for t in pl["tasks"]:
        if tag == "COHORT" or t["animal"] != tag.split("-")[1]:
            by.setdefault(t["animal"], []).append(t["key"])
    by = {a: sorted(v) for a, v in sorted(by.items())}
    dst = sd / "own_epoch_cohort" / f"{tag}_june.npz"
    if dst.exists():
        print("done already", dst)
        return
    dst.parent.mkdir(exist_ok=True)
    cfg = pl["june_fit"]
    _cohort_fit(sd, pl, by, f"{tag}_june", dst, cfg["inits"], cfg["rows_per_animal"], animal=tag, group="june")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "fit"))
    ap.add_argument("--task", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan()
    else:
        cmd_fit(a.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
