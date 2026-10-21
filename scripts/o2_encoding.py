"""The combined encoding model (`wfield_local/combined_encoding.py`) on HMS O2 (Priya 2026-10-19: "yes build the o2
version"). A local pass is 7-8 h on the shared desktop; on O2 every session is its own array task.

    python scripts/o2_encoding.py bundle --name enc_v36_20261019 --user ps150 [--cams cam1,cam4] [--jobs 4] [LABELS ...]
    python scripts/o2_encoding.py check  --name enc_v36_20261019 LABEL [--only drop:task]
    python scripts/o2_encoding.py collect --name enc_v36_20261019 --tag <tag>

bundle (this desktop): per session, everything `run_session` reads from the share (`combined_encoding.session_inputs`:
imaging tuple, Behaviour, epoch / day, and the video motion-energy matrix where it exists) -> inputs/<label>.pkl; a
snapshot of the code (`git archive HEAD`: wfield_local/, configs/) -> code/; the runner; a PARAMETRISED sbatch; and
COMMANDS.md -> <DeepLabCut/Widefield>/o2/<name>/ on the share. The model options are NOT baked in: each submit sets
ORDER / ONLY / VIDEO / FR / TAG (`sbatch --export=...`), so one bundle serves every run (main, nokin, nodir, trmodels,
musall, video, fr) and outputs land in out/<TAG>/sessions/ with out/<TAG>/run.json recording the options. FR (FaceRhythm
factor time courses of a frozen set, `wfield_local/face_rhythm_outputs.py`) is bundled only with `--fr <set>`; inputs
already in a bundle are not rebuilt, so an FR run needs a bundle made with --fr (a new name, or delete inputs/).
check (this desktop): runs one session from its bundled inputs AND from the share, with the same options, and compares
the region tables -- the bundle must reproduce the local result before an O2 run is trusted.
collect (this desktop): out/<TAG>/sessions/* -> <LP_analysis>/<run>/imaging_encoding/<order[_video-..][_subset-..]>/
sessions/, the folder `combined_encoding.out_dir` gives for those options.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

RUNNER = r'''"""Runs on O2: the combined encoding model for one session from its bundled inputs (task = SLURM_ARRAY_TASK_ID).
Options from the environment: ORDER (movement_first | task_first | musall), ONLY (model-name prefixes or 'trmodels'),
VIDEO (cameras, e.g. cam1,cam4), FR (FaceRhythm factor set, e.g. master_pre_v1), TAG (output subfolder)."""
import json, os, pickle, sys, time
from pathlib import Path

b = Path(__file__).resolve().parent
sys.path.insert(0, str(b / "code"))
task = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
labels = [x.strip() for x in (b / "labels.txt").read_text().splitlines() if x.strip()]
label = labels[task - 1]
order, tag = os.environ.get("ORDER", "movement_first"), os.environ["TAG"]
if os.environ.get("ONLY"):
    os.environ["COMBINED_ENCODING_ONLY"] = os.environ["ONLY"]
if os.environ.get("VIDEO"):
    os.environ["COMBINED_ENCODING_VIDEO"] = os.environ["VIDEO"]
if os.environ.get("FR"):
    os.environ["COMBINED_ENCODING_FR"] = os.environ["FR"]
if os.environ.get("MATCH"):                      # lick-matched subsampling: MATCH = a JSON in the bundle (match.json)
    os.environ["COMBINED_ENCODING_MATCH"] = str(b / os.environ["MATCH"])
out = b / "out" / tag
(out / "sessions").mkdir(parents=True, exist_ok=True)
(out / "run.json").write_text(json.dumps({"order": order, "only": os.environ.get("ONLY", ""),
                                          "video": os.environ.get("VIDEO", ""), "fr": os.environ.get("FR", ""),
                                          "match": os.environ.get("MATCH", ""),
                                          "tag": tag}))
if (out / "sessions" / f"{label}.json").exists():
    print("done already", label); sys.exit(0)
from wfield_local import combined_encoding as ce
inputs = pickle.loads((b / "inputs" / f"{label}.pkl").read_bytes())
t0 = time.time()
res = ce.run_session(label, order, inputs=inputs)
ce.save(res, out)
R = res["region"]
full = float(R[(R.model == "full") & (R.window == "all") & (R.area == "cortex")].r2.iloc[0])
print(f"{label}: {order} only={os.environ.get('ONLY', '')} video={os.environ.get('VIDEO', '')} "
      f"fr={os.environ.get('FR', '')} "
      f"full cvR2 {full:.3f} in {(time.time() - t0) / 60:.1f} min", flush=True)
'''


def _inputs_worker(job):
    """Module-level worker: one session's inputs -> pickle (float32 analog traces to halve the size)."""
    import numpy as np

    from wfield_local import combined_encoding as ce
    from wfield_local.orofacial_cohort import _repo_on_path
    _repo_on_path()
    label, dst, cams, fr = job
    x = ce.session_inputs(label, cams=cams.split(",") if cams else None, fr=fr or None)
    B = x["behaviour"]
    B.analog = {k: (np.asarray(t, np.float64), np.asarray(v, np.float32)) for k, (t, v) in B.analog.items()}
    Path(dst).write_bytes(pickle.dumps(x, protocol=5))
    return {"label": label, "MB": round(Path(dst).stat().st_size / 1e6), "video": x["video"] is not None,
            "fr": x.get("fr") is not None}


def cmd_bundle(a) -> Path:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import o2_inference as oi
    from wfield_local import writeguard
    from wfield_local.orofacial_cohort import RUN
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    os.environ[ce.RUN_ENV] = a.run or RUN
    d = oi.bundle_dir(rv, a.name)
    writeguard.assert_writable(d.parent)
    for sub in ("inputs", "logs", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    labels = a.labels or ce.sessions_available()
    todo = [(lab, str(d / "inputs" / f"{lab}.pkl"), a.cams or "", a.fr or "") for lab in labels
            if not (d / "inputs" / f"{lab}.pkl").exists()]
    got, fail = ak.fan_sessions(todo, _inputs_worker, jobs=a.jobs, key=lambda j: j[0])
    for it, err in fail:
        print(f"  FAILED {it[0]}: {err}")
    ok = sorted(p.stem for p in (d / "inputs").glob("*.pkl"))
    (d / "labels.txt").write_text("".join(f"{x}\n" for x in ok), encoding="utf-8", newline="\n")
    repo = Path(__file__).resolve().parents[1]
    code = d / "code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir()
    tar = d / "code.tar"
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tar), "HEAD", "wfield_local", "configs"],
                   check=True)
    shutil.unpack_archive(str(tar), str(code))
    tar.unlink()
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    (d / "enc_runner.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    s = oi.scratch(a.user)
    b = f"{s}/o2/{a.name}"
    n = len(ok)
    sb = f"""#!/bin/bash -l
#SBATCH -J enc_{a.name}
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-06:00:00
#SBATCH -o {b}/logs/%x_%A_%a.out
#SBATCH -e {b}/logs/%x_%A_%a.err
# Options via sbatch --export=ALL,TAG=...,ORDER=...,ONLY=...,VIDEO=...,FR=...  (see COMMANDS.md). Login shell + conda activate
# (a plain shell with `source activate` fell back to the system Python, 10-18).
set -eo pipefail
module load conda/miniforge3/24.11.3-0
conda activate lp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "task $SLURM_ARRAY_TASK_ID TAG=$TAG ORDER=${{ORDER:-movement_first}} ONLY=${{ONLY:-}} VIDEO=${{VIDEO:-}} FR=${{FR:-}} on $(hostname)"
python {b}/enc_runner.py
"""
    (d / "run_enc.sbatch").write_text(sb, encoding="utf-8", newline="\n")
    share_b = oi.o2_path(d, rv)
    runs = [("nodir", "", "nodir:", "", ""), ("trmodels", "", "trmodels", "", ""),
            ("musall_shuf", "musall", "shuf:", "", ""), ("main", "", "", "", ""), ("video", "", "", "cam1,cam4", "")]
    if a.fr:                  # FR vs motion energy vs both in ONE run (video:*, fr:*, frvideo:* models)
        runs.append((f"fr_{a.fr}", "", "", "cam1,cam4", a.fr))
    sub = "\n".join(f"sbatch --array=1-{n}%60 --export=ALL,TAG={t},ORDER={o or 'movement_first'},ONLY={only},"
                    f"VIDEO={v},FR={f} {b}/run_enc.sbatch      # {t}" for t, o, only, v, f in runs)
    cmds = f"""# O2 combined encoding model: {a.name} ({n} sessions; code {commit}; run {os.environ[ce.RUN_ENV]})

One bundle, many runs: each submit sets TAG (output folder out/<TAG>/), ORDER, ONLY, VIDEO, FR.

## 1. Transfer node  (ssh {a.user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b}
rsync -ah {share_b}/ {b}/
ls {b}/inputs | wc -l       # expect {n}
```

## 2. Login node  (ssh {a.user}@o2.hms.harvard.edu) -- ONE task of one run first
```bash
sbatch --array=1 --export=ALL,TAG=nodir,ORDER=movement_first,ONLY=nodir:,VIDEO=,FR= {b}/run_enc.sbatch
squeue -u {a.user}
tail -n 5 {b}/logs/enc_*_1.out
```
When `{b}/out/nodir/sessions/` has the first session (~20-60 min), submit what you need:
```bash
{sub}
```
(video only finds motion energy for the sessions bundled with --cams; others are skipped by the model.)

## 3. Transfer node, when a run is done
```bash
ls {b}/out/<TAG>/sessions/*.json | wc -l
rsync -ah {b}/out/ {share_b}/out/
```

## 4. Desktop
`python scripts/o2_encoding.py collect --name {a.name} --tag <TAG>`
"""
    (d / "COMMANDS.md").write_text(cmds, encoding="utf-8", newline="\n")
    print(f"{n} sessions bundled -> {d}\n{cmds}")
    return d


def _set_env(order_only_video: dict) -> None:
    from wfield_local import combined_encoding as ce
    for k, env in (("only", ce.ONLY_ENV), ("video", ce.VIDEO_ENV), ("fr", ce.FR_ENV), ("match", ce.MATCH_ENV)):
        if order_only_video.get(k):
            os.environ[env] = order_only_video[k]
        else:
            os.environ.pop(env, None)


def cmd_check(a) -> int:
    """One session from bundled inputs vs from the share, same options: region tables must match."""
    import numpy as np

    from wfield_local import combined_encoding as ce
    from wfield_local import o2_inference as oi
    from wfield_local.orofacial_cohort import RUN, _repo_on_path
    from wfield_local.paths import PathResolver
    _repo_on_path()
    os.environ[ce.RUN_ENV] = a.run or RUN
    _set_env({"only": a.only, "video": a.video, "fr": a.fr})
    d = oi.bundle_dir(PathResolver(), a.name)
    inputs = pickle.loads((d / "inputs" / f"{a.label}.pkl").read_bytes())
    r1 = ce.run_session(a.label, a.order, inputs=inputs)["region"]
    r2 = ce.run_session(a.label, a.order)["region"]
    m = r1.merge(r2, on=["model", "window", "area"], suffixes=("_bundle", "_share"))
    diff = np.abs(m.r2_bundle - m.r2_share)
    print(f"{len(m)} region rows; max |r2 bundle - share| = {diff.max():.2e}; "
          f"rows differing > 1e-6: {(diff > 1e-6).sum()}")
    return 0 if diff.max() < 1e-4 else 1


def cmd_collect(a) -> int:
    from wfield_local import combined_encoding as ce
    from wfield_local import o2_inference as oi
    from wfield_local import writeguard
    from wfield_local.orofacial_cohort import RUN
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    os.environ[ce.RUN_ENV] = a.run or RUN
    src = oi.bundle_dir(rv, a.name) / "out" / a.tag
    meta = json.loads((src / "run.json").read_text())
    _set_env(meta)
    dst = ce.out_dir(rv, meta["order"] or "movement_first") / "sessions"
    writeguard.assert_writable(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((src / "sessions").iterdir()):
        shutil.copy2(f, dst / f.name)
        n += 1
        if f.suffix == ".json":
            # two array tasks writing the same session at once (a resubmitted array overlapping an earlier one) left
            # a stray trailing byte after an otherwise complete JSON (trmodels PS92_1002, 10-19): keep the first object
            s = (dst / f.name).read_text(encoding="utf-8")
            obj, end = json.JSONDecoder().raw_decode(s)
            if s[end:].strip():
                (dst / f.name).write_text(json.dumps(obj, indent=1), encoding="utf-8")
                print(f"  repaired {f.name}: dropped {len(s) - end} trailing byte(s)")
    print(f"collected {n} files ({len(list((src / 'sessions').glob('*.json')))} sessions) -> {dst}")
    return 0


def cmd_match(a) -> int:
    """Per animal, target usable licks = frac x the smallest session (August pre + post epochs) -> bundle/match.json.
    Counts from licks.csv with the model's lick filter (engaged trials, measured peak, reach >= MIN_REACH_PX)."""
    import numpy as np
    import pandas as pd

    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import o2_inference as oi
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    rv = PathResolver()
    run = a.run or oc.RUN
    d = oi.bundle_dir(rv, a.name)
    labels = [x.strip() for x in (d / "labels.txt").read_text().splitlines() if x.strip()]
    n = {}
    for lab in labels:
        ep = epochs.epoch_of(lab)
        if ep not in ("pre", "acute", "subacute", "chronic") or (ep == "pre" and lab[5:] < oc.LATE_PRE_FROM):
            continue
        sd = oo.session_dir(rv, run, lab[:4], "2026" + lab[5:])
        keep, _ = oc.engagement_keep(lab[:4], "2026" + lab[5:], sd)
        L = pd.read_csv(sd / "licks.csv")
        if keep is not None:
            L = L[L.trial_id.isin(keep)]
        ok = (~L.peak_interp.astype(bool)) & (L.protrusion_px >= ce.MIN_REACH_PX) & np.isfinite(L.peak_angle_deg)
        n[lab] = int(ok.sum())
    S = pd.Series(n)
    tgt = {lab: int(a.frac * S[S.index.str[:4] == lab[:4]].min()) for lab in S.index}
    (d / "match.json").write_text(json.dumps(tgt, indent=1), encoding="utf-8")
    for an in sorted({x[:4] for x in S.index}):
        s = S[S.index.str[:4] == an]
        print(f"{an}: {len(s)} sessions, usable licks {s.min()}..{s.max()} -> target {tgt[s.idxmin()]}")
    print(f"-> {d / 'match.json'} ({len(tgt)} sessions; June pre sessions not matched = left out of the matched run)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--name", required=True)
    b.add_argument("--user", required=True)
    b.add_argument("--run", default=None)
    b.add_argument("--cams", default="cam1,cam4", help="video motion energy to bundle where it exists ('' = none)")
    b.add_argument("--fr", default="", help="FaceRhythm factor set to bundle where projected, e.g. master_pre_v1")
    b.add_argument("--jobs", type=int, default=4)
    b.add_argument("labels", nargs="*")
    c = sub.add_parser("check")
    c.add_argument("--name", required=True)
    c.add_argument("--run", default=None)
    c.add_argument("--order", default="movement_first")
    c.add_argument("--only", default="drop:task")
    c.add_argument("--video", default="")
    c.add_argument("--fr", default="")
    c.add_argument("label")
    k = sub.add_parser("collect")
    k.add_argument("--name", required=True)
    k.add_argument("--run", default=None)
    k.add_argument("--tag", required=True)
    m = sub.add_parser("match", help="write match.json (per-session lick targets) into the bundle")
    m.add_argument("--name", required=True)
    m.add_argument("--run", default=None)
    m.add_argument("--frac", type=float, default=0.95, help="target = frac x the animal's smallest session")
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        cmd_bundle(a)
        return 0
    if a.cmd == "match":
        return cmd_match(a)
    return cmd_check(a) if a.cmd == "check" else cmd_collect(a)


if __name__ == "__main__":
    raise SystemExit(main())
