"""Desktop stage `plan`: every task's epoch, the fit / session rows, plan.json, the O2 sbatch files, and the copy of the code the O2 jobs run (fr_factors.py + this package as fr_factors_lib/ + the sidecar modules)."""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from .config import ARRIVAL_S, BUSY_S, PRECUE_EXT_S, PRECUE_S, RESP_S, SIDECAR_MODULES, TRIAL_DS
from .files import _sidecar_module
from .sample import between_states, draw_rows, even_draw, session_candidates, session_group, trial_strobes


# ------------------------------------------------------------------------------------- desktop: plan / select
def _sbatch(b: str, opts: str, cmd: str) -> str:
    thr = opts.split(" -c ")[1].split()[0]
    parts, lines, i = opts.split(), [], 0
    while i < len(parts):
        step = 1 if parts[i].startswith("--") else 2
        lines.append(" ".join(parts[i:i + step]))
        i += step
    return ("#!/bin/bash -l\n#SBATCH -p short\n" + f"#SBATCH -o {b}/logs/%x_%A_%a.out\n#SBATCH -e {b}/logs/%x_%A_%a.err\n"
            + "".join(f"#SBATCH {x}\n" for x in lines)
            + "set -eo pipefail\nmodule load conda/miniforge3/24.11.3-0\nconda activate fr\n"
            + f"export OMP_NUM_THREADS={thr} MKL_NUM_THREADS={thr} OPENBLAS_NUM_THREADS={thr}\n"
            + f"python {b}/fr_factors.py {cmd}\n")


LIB = "fr_factors_lib"      # the package's folder name inside a set on O2 (imported there by fr_factors.py)


def deploy_code(sd: Path) -> list[Path]:
    """O2 has no repo: copy what the O2 stages run into the set folder -- the launcher as ``fr_factors.py``, this
    package as ``fr_factors_lib/`` and the SIDECAR modules (`_sidecar_module`). `plan` calls it; call it on its own
    to update the code of an existing set without re-planning (the 10-24 fitepoch copy). Overwrites only those
    files; LF endings for O2."""
    repo, pkg = Path(__file__).resolve().parents[3], Path(__file__).resolve().parent
    pairs = [(repo / "scripts" / "face_rhythm_factors.py", sd / "fr_factors.py")]
    pairs += [(p, sd / LIB / p.name) for p in sorted(pkg.glob("*.py"))]
    pairs += [(repo / "wfield_local" / f"{m}.py", sd / f"{m}.py") for m in SIDECAR_MODULES]
    (sd / LIB).mkdir(parents=True, exist_ok=True)
    for src, dst in pairs:
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    return [d for _, d in pairs]


def cmd_plan(bundle: str, name: str, trials_per_animal: int, session_trials: int, between_frac: float, seed: int,
             ranks: list[int], inits: int, n_iter: int, user: str, point_pcs: int = 20,
             points_bins: int = 20000, animals: list[str] | None = None, with_outputs: bool = False,
             role: str = "master", loso: bool = True, trial_ds: int = TRIAL_DS,
             views: list[str] | None = None, run_share: float | None = None, arrival: bool = False,
             even_days: bool = False) -> Path:
    import pandas as pd
    from wfield_local import epochs, writeguard
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    bd = oi.bundle_dir(rv, bundle)
    sd = bd / "factors" / name
    writeguard.assert_writable(sd.parent)
    if (sd / "master.npz").exists():
        sys.exit(f"{sd} is FROZEN (master.npz exists) -- choose a new set name")
    T = pd.read_csv(bd / "tasks.tsv", sep="\t", dtype=str)
    if animals:                                             # a PILOT on some animals (10-22: PS92 + PS93)
        T = T[T.animal.isin(animals)]
    if with_outputs:                                        # only tasks whose outputs are on the share now
        T = T[[(bd / "out" / f"{k}_summary.json").exists() for k in T.key]]
    if role == "master" and (animals or with_outputs):
        sys.exit("a set restricted by --animals / --with-outputs is not the cohort master: pass --role pilot")
    view_spec = None
    if views:                                               # MULTI-VIEW (face_rhythm_multiview): cam=bundle each
        fmv = _sidecar_module("face_rhythm_multiview")
        rc = next((q for q in bundle.split("_") if q.startswith("cam")), None)
        cams = [dict(zip(("cam", "bundle"), v.split("=", 1))) for v in views]
        keep = np.ones(len(T), bool)
        for v in cams:
            Tv = pd.read_csv(oi.bundle_dir(rv, v["bundle"]) / "tasks.tsv", sep="\t", dtype=str)
            has = set(Tv.key)
            if with_outputs:
                has = {k for k in has if (oi.bundle_dir(rv, v["bundle"]) / "out" / f"{k}_summary.json").exists()}
            keep &= np.array([fmv.view_key(k, rc, v["cam"]) in has for k in T.key])
        if (~keep).any():
            print(f"  multi-view: {(~keep).sum()} task(s) lack a view, left out: {', '.join(T.key[~keep])}")
        T = T[keep]
        view_spec = {"ref": rc, "cams": cams}
    frt = _sidecar_module("face_rhythm_timing")
    rng = np.random.default_rng(seed)
    tasks, cand = [], {}
    no_timing = [k for k in T.key if frt.load(bd, k) is None]
    if no_timing:
        sys.exit(f"{len(no_timing)} task(s) without timing ({', '.join(no_timing[:4])} ...): run "
                 f"`o2_face_rhythm.py timing --name {bundle}` first (true bin times + lossy bins, 10-23)")
    for r in T.itertuples():
        tasks.append({"key": r.key, "animal": r.animal, "date": r.date, "epoch": epochs.epoch_of(f"{r.animal}_{r.date[4:]}")})
        tm = frt.load(bd, r.key)
        st = between_states(rv, r.animal, r.date, tm["t5"][:int(r.n_frames) // 50]) if run_share is not None else None
        zt = np.load(bd / "trials" / f"{r.key}.npz")
        sb = trial_strobes(rv, r.animal, r.date, zt["trial_id"], zt["cue_daq_s"]) if arrival else None
        cand[r.key] = session_candidates(r.key, zt, int(r.n_frames), tm, st, sb)
    fit = {}
    for a in sorted({t["animal"] for t in tasks}):
        keys = [t["key"] for t in tasks if t["animal"] == a and t["epoch"] == "pre"]
        if not keys:
            continue
        picks = draw_rows(sum((cand[k][0] for k in keys), []), sum((cand[k][1] for k in keys), []),
                          trials_per_animal, between_frac, rng, ds=trial_ds, run_share=run_share,
                          even_days=even_days)
        fit[a] = {"sessions": keys, **picks}
        n_tr = sum(map(len, picks["trials"].values()))
        print(f"  {a}: {len(keys)} pre sessions, {n_tr} trials (fit rows every {trial_ds * 40} ms), "
              f"{sum(map(len, picks.get('arrival', {}).values()))} pre-cue 5 Hz bins (arrival + cue-4..-2 s), "
              f"{sum(map(len, picks['between'].values()))} between bins {picks['n_between']}")
    if arrival and fit:                                     # EQUAL pre-cue rows per animal (10-23): the count follows
        n_min = min(sum(map(len, f_["arrival"].values())) for f_ in fit.values())   # each animal's strobe -> cue lead
        for a_, f_ in fit.items():
            allb = [(k, b) for k, v in sorted(f_["arrival"].items()) for b in v]
            if even_days:                                   # ... and evenly over the animal's days
                kept = even_draw(allb, n_min, rng)
            else:
                kept = [allb[j] for j in (sorted(rng.choice(len(allb), n_min, replace=False))
                                          if len(allb) > n_min else range(len(allb)))]
            f_["arrival"] = {}
            for k_, b_ in kept:
                f_["arrival"].setdefault(k_, []).append(int(b_))
            f_["arrival"] = {k: sorted(v) for k, v in sorted(f_["arrival"].items())}
            print(f"  {a_}: pre-cue rows {len(allb)} -> {sum(map(len, f_['arrival'].values()))} (equal per animal)")
    points_fit = {}
    for a in fit:                                           # equal 25 Hz bins per animal, spread over its pre sessions
        keys = fit[a]["sessions"]
        nb = {k: int(next(r for r in T.itertuples() if r.key == k).n_frames) // 10 for k in keys}
        per = points_bins // len(keys)
        points_fit[a] = {k: sorted(int(b) for b in rng.choice(nb[k], min(per, nb[k]), replace=False)) for k in keys}
    session_rows = {t["key"]: draw_rows(*cand[t["key"]], session_trials, between_frac,
                                        np.random.default_rng([seed, zlib.crc32(t["key"].encode())]),
                                        run_share=run_share) for t in tasks}
    days = sorted({session_group(k) for a in fit.values() for k in a["sessions"]})
    loso_jobs = [[r, d] for r in ranks for d in days] if loso else []
    pl = {"set": name, "bundle": bundle, "created": time.strftime("%Y-%m-%d %H:%M"), "role": role,
          **({"views": view_spec} if view_spec else {}),
          "animals": sorted(fit), "loso_jobs": loso_jobs,
          "params": {"trials_per_animal": trials_per_animal, "session_trials": session_trials,
                     "between_frac": between_frac, "seed": seed, "trial_ds": trial_ds, "run_share": run_share,
                     "still": "behavior_events rest periods", "timing": frt.TIMING_VERSION, "even_days": even_days,
                     "precue_rows": {"strobe_s": list(ARRIVAL_S), "from_cue_s": PRECUE_EXT_S, "until": "trial grid start"}
                     if arrival else None, "ranks": ranks, "inits": inits,
                     "n_iter": n_iter, "precue_s": PRECUE_S, "resp_s": RESP_S, "busy_s": BUSY_S,
                     "scale": "per animal: RMS of its fit rows", "point_pcs": point_pcs,
                     "points_bins_per_animal": points_bins,
                     "vqt_norm": float(json.loads((bd / "params.json").read_text())["vqt"]["normalization_factor"]),
                     "one_over_f": float(json.loads((bd / "params.json").read_text())["vqt"]["one_over_f_exponent"])},
          "fit": fit, "points_fit": points_fit, "session_rows": session_rows, "tasks": tasks}
    (sd / "logs").mkdir(parents=True, exist_ok=True)
    (sd / "plan.json").write_text(json.dumps(pl, indent=1), encoding="utf-8", newline="\n")
    deploy_code(sd)
    b = f"{oi.scratch(user)}/o2/{bundle}/factors/{name}"
    n = len(tasks)
    jobs = {"run_build.sbatch": ("-J frf_build -c 4 --mem=40G -t 0-03:00:00", "build"),
            "run_fit.sbatch": (f"-J frf_fit -c 16 --mem=64G -t 0-12:00:00 --array={','.join(map(str, ranks))}",
                               "fit --rank $SLURM_ARRAY_TASK_ID"),
            "run_project.sbatch": (f"-J frf_proj -c 4 --mem=16G -t 0-02:00:00 --array=1-{n}%40",
                                   "project --task $SLURM_ARRAY_TASK_ID"),
            "run_fitsession.sbatch": (f"-J frf_own -c 8 --mem=24G -t 0-04:00:00 --array=1-{n}%40",
                                      "fitsession --task $SLURM_ARRAY_TASK_ID")}
    if loso_jobs:                                           # after the fit (needs rank<R>.npz): --dependency=afterok
        jobs["run_loso.sbatch"] = (f"-J frf_loso -c 8 --mem=48G -t 0-06:00:00 --array=1-{len(loso_jobs)}%40",
                                   "loso --task $SLURM_ARRAY_TASK_ID")
    jobs["run_splithalf.sbatch"] = ("-J frf_split -c 16 --mem=64G -t 0-12:00:00", "splithalf --rank $RANK")
    jobs["run_select.sbatch"] = ("-J frf_select -c 1 --mem=8G -t 0-00:30:00", "select --rank $RANK")
    jobs["run_project_loso.sbatch"] = (f"-J frf_projlo -c 4 --mem=16G -t 0-02:00:00 --array=1-{n}%40",
                                       "project --task $SLURM_ARRAY_TASK_ID --loso-basis")
    for fn, (opts, cmd) in jobs.items():
        (sd / fn).write_text(_sbatch(b, opts, cmd), encoding="utf-8", newline="\n")
    # the whole chain with dependencies (10-23, Priya: encoder projections FIRST, held-out scoring after):
    # build -> fit R (+ leave-one-animal-out) -> {split half, select -> project (full basis, for the encoder),
    # loso -> project --loso-basis (pre, FR scoring) -> fitsession}
    (sd / "run_chain.sh").write_text(f"""#!/bin/bash
# usage: bash {b}/run_chain.sh <rank>    (login node; prints the job ids)
set -eo pipefail
b={b}; R=${{1:?rank}}
j1=$(sbatch --parsable $b/run_build.sbatch)
j2=$(sbatch --parsable --dependency=afterok:$j1 --array=$R $b/run_fit.sbatch)
j3=$(sbatch --parsable --dependency=afterok:$j2 --export=ALL,RANK=$R $b/run_splithalf.sbatch)
j4=$(sbatch --parsable --dependency=afterok:$j2 --export=ALL,RANK=$R $b/run_select.sbatch)
j5=$(sbatch --parsable --dependency=afterok:$j4 $b/run_project.sbatch)
""" + ("""j6=$(sbatch --parsable --dependency=afterok:$j2 $b/run_loso.sbatch)
j7=$(sbatch --parsable --dependency=afterok:$j6:$j4 $b/run_project_loso.sbatch)
j8=$(sbatch --parsable --dependency=afterok:$j7 $b/run_fitsession.sbatch)
echo "build $j1 fit $j2 split $j3 select $j4 project $j5 loso $j6 project_loso $j7 fitsession $j8"
""" if loso_jobs else """j8=$(sbatch --parsable --dependency=afterok:$j5 $b/run_fitsession.sbatch)
echo "build $j1 fit $j2 split $j3 select $j4 project $j5 fitsession $j8"
"""), encoding="utf-8", newline="\n")
    rows = sum(len(v) for a in fit.values() for v in a.get("arrival", {}).values()) + \
        sum(len(v) for a in fit.values() for v in a["trials"].values()) * (151 // trial_ds) + \
        sum(len(v) for a in fit.values() for v in a["between"].values())
    print(f"plan [{role}] -> {sd}  (fit rows ~{rows}, tensor ~{rows * 2 * 1282 * 36 * 4 / 1e9:.1f} GB float32; {n} tasks; "
          f"{len(loso_jobs)} leave-one-session-out jobs = {len(ranks)} ranks x {len(days)} pre days)")
    return sd
