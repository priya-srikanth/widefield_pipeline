"""The combined encoding model (`wfield_local/combined_encoding.py`) on HMS O2 (Priya 2026-10-19: "yes build the o2
version"). A local pass is 7-8 h on the shared desktop; on O2 every session is its own array task.

    python scripts/o2_encoding.py [--profile <render profile>] bundle --name <name> --user ps150 [--cams cam1,cam4] [--jobs 4] [LABELS ... or a label file]
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

REBUILD RUNS (coordinator 11-04; `--profile <render profile>`, applied before anything is imported): the bundle goes to
<rebuild root>/o2/<name> (a registered data area of the root); env.json records EVERY setting `run_session` depends on
(`wfield_local.o2_bundle.settings`: the full profile env + COMBINED_ENCODING_* / OROFACIAL_*) and bundle.json the run
identity (run tag, selection, area basis), the commit and the attempts table; the runner applies env.json before
importing wfield_local and REFUSES unless the bundled code computes that identity; inputs carry the area-basis masks
(built here: the v8 rim reads the share); code/ = `git archive` of the worktree's HEAD (refused if wfield_local / configs
are dirty). check audits the bundled run for share reads; collect refuses a bundle of another run.
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
# the bundle's INPUT settings (COMBINED_ENCODING_* / OROFACIAL_* at bundle time, e.g. lick set, executed set, binned
# contest + its calibration table): '@'-paths point into the bundle; submit-time options below still override
bundle_env = json.loads((b / "env.json").read_text()) if (b / "env.json").exists() else {}
for k in [k for k in os.environ if k.startswith(("WIDEFIELD_", "WFIELD_", "LOCANMF_", "COMBINED_ENCODING_", "OROFACIAL_"))]:
    del os.environ[k]                            # ONLY the bundle's settings (+ the submit options below) apply
for k, v in bundle_env.items():
    os.environ[k] = str(b / v[1:]) if isinstance(v, str) and v.startswith("@") else v
# REFUSE unless the bundled code computes the recorded run identity from these settings (run tag, selection, area basis)
if (b / "bundle.json").exists():
    from wfield_local import o2_bundle
    o2_bundle.verify(json.loads((b / "bundle.json").read_text())["identity"])
order, tag = os.environ.get("ORDER", "movement_first"), os.environ["TAG"]
if os.environ.get("ONLY"):
    os.environ["COMBINED_ENCODING_ONLY"] = os.environ["ONLY"]
if os.environ.get("VIDEO"):
    os.environ["COMBINED_ENCODING_VIDEO"] = os.environ["VIDEO"]
if os.environ.get("FR"):
    os.environ["COMBINED_ENCODING_FR"] = os.environ["FR"]
if os.environ.get("FR_POINTS"):
    os.environ["COMBINED_ENCODING_FR_POINTS"] = os.environ["FR_POINTS"]
if os.environ.get("FR_SHARE"):
    os.environ["COMBINED_ENCODING_FR_SHARE"] = os.environ["FR_SHARE"]
if os.environ.get("FR_GROUPS"):                  # FaceRhythm grouping:mode (submit-time; wfield_local/fr_groups.py)
    os.environ["COMBINED_ENCODING_FR_GROUPS"] = os.environ["FR_GROUPS"]
if os.environ.get("BINS"):                       # binned contest on (submit-time; combined_encoding.BINS_ENV)
    os.environ["COMBINED_ENCODING_BINS"] = os.environ["BINS"]
if os.environ.get("BINS_WIDTH"):                 # soft-bin width sweep (submit-time; combined_encoding.BINS_WIDTH_ENV)
    os.environ["COMBINED_ENCODING_BINS_WIDTH"] = os.environ["BINS_WIDTH"]
if os.environ.get("MATCH"):                      # lick-matched subsampling: MATCH = a JSON in the bundle (match.json)
    os.environ["COMBINED_ENCODING_MATCH"] = str(b / os.environ["MATCH"])
out = b / "out" / tag
(out / "sessions").mkdir(parents=True, exist_ok=True)
(out / "run.json").write_text(json.dumps({"order": order, "only": os.environ.get("ONLY", ""),
                                          "video": os.environ.get("VIDEO", ""), "fr": os.environ.get("FR", ""),
                                          "fr_points": os.environ.get("FR_POINTS", ""),
                                          "fr_share": os.environ.get("FR_SHARE", ""),
                                          "match": os.environ.get("MATCH", ""),
                                          "bins": os.environ.get("BINS", ""),
                                          "fr_groups": os.environ.get("FR_GROUPS", ""),
                                          "bins_width": os.environ.get("BINS_WIDTH", ""),
                                          "bundle_env": bundle_env, "tag": tag}))
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
    label, dst, cams, fr, fr_points, fr_share = job
    if fr_share:
        os.environ[ce.FR_SHARE_ENV] = "1"
    if fr_points:
        os.environ[ce.FR_POINTS_ENV] = str(fr_points)
    x = ce.session_inputs(label, cams=cams.split(",") if cams else None, fr=fr or None)
    B = x["behaviour"]
    B.analog = {k: (np.asarray(t, np.float64), np.asarray(v, np.float32)) for k, (t, v) in B.analog.items()}
    Path(dst).write_bytes(pickle.dumps(x, protocol=5))
    return {"label": label, "MB": round(Path(dst).stat().st_size / 1e6), "video": x["video"] is not None,
            "fr": x.get("fr") is not None}


def bundle_env(d: Path | None = None) -> dict:
    """The bundle's INPUT settings = `wfield_local.o2_bundle.settings()`: the full render-profile env (WIDEFIELD_* /
    WFIELD_* / LOCANMF_*) + COMBINED_ENCODING_* / OROFACIAL_*, minus this machine's store paths and the per-submit
    options. A value naming an existing file is copied to <bundle>/tables/ and recorded as '@tables/<name>' (the runner
    resolves it inside the bundle on O2); a value naming a DIRECTORY cannot be bundled and is refused. Recorded so one
    bundle = one input configuration, the runner reproduces it, and `collect` names folders the same."""
    from wfield_local import o2_bundle
    env = {}
    for k, v in o2_bundle.settings().items():
        if Path(v).is_absolute() and Path(v).is_dir():
            raise SystemExit(f"{k}={v} names a directory: O2 cannot see it and it is not a bundle setting")
        if d is not None and Path(v).is_absolute() and Path(v).is_file():
            (d / "tables").mkdir(exist_ok=True)
            shutil.copy2(v, d / "tables" / Path(v).name)
            v = f"@tables/{Path(v).name}"
        env[k] = v
    return env


def bundle_dir(rv, name: str) -> Path:
    """Where bundle ``name`` lives: in a rebuild-root run <root>/o2/<name> (a registered data area of the root, so the
    tag rule accepts it; coordinator 11-04), else `o2_inference.bundle_dir` (DeepLabCut/Widefield/o2/<name>)."""
    from wfield_local import o2_inference as oi
    from wfield_local import rebuild_root, run_mode
    r = rebuild_root.root()
    if r is None:
        return oi.bundle_dir(rv, name)
    run_mode.register_data_store(Path(r) / "o2")
    return Path(r) / "o2" / name


def _git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


#: the runs of a full rebuild bundle (TAG, ORDER, ONLY, extra exports): the 10-24 submit_all.sh set (fr_A_tr dropped,
#: coordinator 11-04: this code computes the FR trial-to-trial outputs in every fr run, so fr_A carries them)
FULL_RUNS = (("bins_w1", "movement_first", "nokin:", "BINS=1"),
             ("bins_w05", "movement_first", "nokin:", "BINS=1,BINS_WIDTH=0.5"),
             ("bins_w2", "movement_first", "nokin:", "BINS=1,BINS_WIDTH=2"),
             ("main", "movement_first", "", ""),
             ("droppos", "movement_first", "droppos-", ""),
             ("shufpos", "movement_first", "shufpos-", ""),
             ("nodir", "movement_first", "nodir:", ""),
             ("drop", "movement_first", "drop:", ""),
             ("trmodels", "movement_first", "trmodels", ""),
             ("musall_shuf", "musall", "shuf:", ""),
             ("fr_A", "movement_first", "fr:", "FR=master_pre_v1,FR_GROUPS=master_v1:A"),
             ("fr_B", "movement_first", "fr:", "FR=master_pre_v1,FR_GROUPS=master_v1:B"),
             ("fr_S", "movement_first", "fr:", "FR=master_pre_v1,FR_GROUPS=master_v1:S"),
             ("fr_D", "movement_first", "fr:", "FR=master_pre_v1,FR_GROUPS=master_v1:D"))


def cmd_bundle(a) -> Path:
    import datetime

    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import o2_bundle
    from wfield_local import o2_inference as oi
    from wfield_local import orofacial_cohort as oc
    from wfield_local import writeguard
    from wfield_local.orofacial_cohort import RUN
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    os.environ[ce.RUN_ENV] = a.run or RUN
    # CODE SNAPSHOT = HEAD of the worktree this runs from; refuse uncommitted changes to what is archived
    repo = Path(__file__).resolve().parents[1]
    dirty = _git(repo, "status", "--porcelain", "--", "wfield_local", "configs", "scripts/o2_encoding.py")
    if dirty:
        raise SystemExit(f"uncommitted changes in {repo} (the bundle archives HEAD):\n{dirty}")
    commit = _git(repo, "rev-parse", "HEAD")
    # the run's attempts table (OROFACIAL_GATE=attempted) is read at BUNDLE time (Behaviour.trials.engaged)
    attempts = None
    if oc.ENGAGEMENT_GATE != "reference":
        from wfield_local import lp_licks
        attempts = lp_licks.attempts_path()
        if not Path(attempts).exists():
            raise SystemExit(f"{attempts} missing: run scripts/trial_attempts.py under this profile first")
    d = bundle_dir(rv, a.name)
    writeguard.assert_writable(d)
    for sub in ("inputs", "logs", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    ident = o2_bundle.identity()
    meta_f = d / "bundle.json"
    if meta_f.exists():                       # resuming: the inputs already there must be of the same run
        old = json.loads(meta_f.read_text())
        if old["identity"] != ident:
            raise SystemExit(f"{d} holds a bundle of another run: {old['identity']} vs {ident}")
    labels = a.labels or ce.sessions_available()
    if len(labels) == 1 and Path(labels[0]).is_file():                   # a label FILE (whitespace-separated)
        labels = Path(labels[0]).read_text().split()
    if not a.no_inputs:
        todo = [(lab, str(d / "inputs" / f"{lab}.pkl"), a.cams or "", a.fr or "", a.fr_points or 0, a.fr_share)
                for lab in labels
                if not (d / "inputs" / f"{lab}.pkl").exists()]
        got, fail = ak.fan_sessions(todo, _inputs_worker, jobs=a.jobs, key=lambda j: j[0])
        for it, err in fail:
            print(f"  FAILED {it[0]}: {err}")
    ok = sorted(p.stem for p in (d / "inputs").glob("*.pkl"))
    (d / "labels.txt").write_text("".join(f"{x}\n" for x in ok), encoding="utf-8", newline="\n")
    env = bundle_env(d)
    (d / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8", newline="\n")
    if attempts is not None:                  # provenance only: the gate is already inside each inputs/<label>.pkl
        (d / "tables").mkdir(exist_ok=True)
        shutil.copy2(attempts, d / "tables" / "trial_attempts.csv")
    from wfield_local import rebuild_root
    meta = {"identity": ident, "commit": commit, "code_root": str(repo), "name": a.name,
            "profile": os.environ.get("O2_BUNDLE_PROFILE", ""), "rebuild_root": str(rebuild_root.root() or ""),
            "attempts_table": str(attempts or ""), "cams": a.cams, "fr": a.fr, "n_sessions": len(ok),
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    code = d / "code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir()
    tar = d / "code.tar"
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tar), commit, "wfield_local", "configs"],
                   check=True)
    shutil.unpack_archive(str(tar), str(code))
    tar.unlink()
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

    def line(t, o, only, extra, arr):
        return (f"sbatch --array={arr} --export=ALL,TAG={t},ORDER={o},ONLY={only},VIDEO=,FR="
                + (f",{extra}" if extra else "") + f" {b}/run_enc.sbatch")
    # FR= first, then the run's own FR=... overrides it (sbatch --export: the last assignment wins)
    sub = "\n".join(line(t, o, only, extra, f"1-{n}%60") + f"      # {t}" for t, o, only, extra in FULL_RUNS)
    prof = f" --profile {meta['profile']}" if meta["profile"] else ""
    cmds = f"""# O2 combined encoding model: {a.name}

{n} sessions; code {commit} ({repo.name}); run tag {ident['run_tag']}
selection {ident['selection']}; area basis {ident['area_basis']}; lick run {os.environ[ce.RUN_ENV]}
Settings: env.json (applied by the runner before wfield_local is imported; it REFUSES if the bundled code does not
compute the recorded run tag / selection / area basis from them). Gate table: tables/trial_attempts.csv (provenance).

## 1. Transfer node  (ssh {a.user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b}
rsync -ah {share_b}/ {b}/
ls {b}/inputs | wc -l       # expect {n}
```

## 2. Login node  (ssh {a.user}@o2.hms.harvard.edu) -- ONE task first
```bash
{line("main", "movement_first", "", "", "1")}
squeue -u {a.user}
tail -n 5 {b}/logs/enc_{a.name}_*_1.out      # must end in "full cvR2 ..."; a settings mismatch stops here
```
When `{b}/out/main/sessions/` has its first session (~20-60 min), submit all {len(FULL_RUNS)} runs (the 1-task run above is part
of `main`; its session is skipped as done):
```bash
{sub}
```
fr_A also carries the FR trial-to-trial / PETH outputs (Figs 21.4 / 21.11; the 10-25 fr_A_tr re-run is not needed with
this code). A task that runs out of memory: resubmit that index with --mem=64G.

## 3. Transfer node, when a run is done
```bash
ls {b}/out/<TAG>/sessions/*.json | wc -l      # expect {n}
rsync -ah {b}/out/ {share_b}/out/
```

## 4. Desktop (rebuild worktree, PYTHONPATH=<code_root>)
```bash
python scripts/o2_encoding.py{prof} collect --name {a.name} --tag <TAG>
```
collect writes to the routed run-tagged encoding folder (combined_encoding.out_dir) and refuses a bundle whose run tag
differs from the current profile's.
"""
    (d / "COMMANDS.md").write_text(cmds, encoding="utf-8", newline="\n")
    print(f"{n} sessions bundled -> {d}\n{cmds}")
    return d


def _set_env(order_only_video: dict) -> None:
    from wfield_local import combined_encoding as ce
    for k, env in (("only", ce.ONLY_ENV), ("video", ce.VIDEO_ENV), ("fr", ce.FR_ENV),
                   ("fr_points", ce.FR_POINTS_ENV), ("fr_share", ce.FR_SHARE_ENV), ("match", ce.MATCH_ENV),
                   ("bins", ce.BINS_ENV), ("bins_width", ce.BINS_WIDTH_ENV),
                   ("fr_groups", "COMBINED_ENCODING_FR_GROUPS")):
        if order_only_video.get(k):
            os.environ[env] = order_only_video[k]
        else:
            os.environ.pop(env, None)


def _bundle_meta(d: Path) -> dict:
    f = d / "bundle.json"
    if not f.exists():
        raise SystemExit(f"{f} missing: a bundle without its recorded run identity (before 11-04) cannot be checked or "
                         "collected into a rebuild run")
    return json.loads(f.read_text())


def _refuse_other_run(meta: dict) -> dict:
    """REFUSE unless this process (the current profile) computes the bundle's recorded run identity."""
    from wfield_local import o2_bundle
    try:
        return o2_bundle.verify(meta["identity"])
    except o2_bundle.BundleMismatch as ex:
        raise SystemExit(f"refusing: this bundle is not of the current run ({ex})") from None


_SHARE_READS: list = []
_AUDIT = {"on": False}


def _audit(event, args):
    if _AUDIT["on"] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).replace("\\", "/")
        if p[:2].upper() in ("M:", "N:") or "/MICROSCOPE/" in p.upper() or p.startswith("//"):
            _SHARE_READS.append(p)


def cmd_check(a) -> int:
    """One session from its bundled inputs (as on O2: env.json settings, no share reads -- audited) vs loaded here
    under the current profile, same options: the region tables must agree to < 1e-4."""
    import sys

    import numpy as np

    from wfield_local import combined_encoding as ce
    from wfield_local.orofacial_cohort import RUN, _repo_on_path
    from wfield_local.paths import PathResolver
    _repo_on_path()
    os.environ[ce.RUN_ENV] = a.run or RUN
    d = bundle_dir(PathResolver(), a.name)
    _refuse_other_run(_bundle_meta(d))
    env = json.loads((d / "env.json").read_text())
    for k, v in env.items():                                           # the bundle's settings, as the runner applies
        if os.environ.get(k) != v and not (isinstance(v, str) and v.startswith("@")):
            raise SystemExit(f"env.json {k}={v!r} differs from this shell's {os.environ.get(k)!r}")
    _set_env({"only": a.only, "video": a.video, "fr": a.fr,          # submit-time options from this shell's env
              **{k: os.environ.get(e, "") for k, e in (("bins", ce.BINS_ENV), ("bins_width", ce.BINS_WIDTH_ENV),
                                                         ("fr_groups", "COMBINED_ENCODING_FR_GROUPS"))}})
    inputs = pickle.loads((d / "inputs" / f"{a.label}.pkl").read_bytes())
    sys.addaudithook(_audit)
    _AUDIT["on"] = True
    try:
        r1 = ce.run_session(a.label, a.order, inputs=inputs)["region"]
    finally:
        _AUDIT["on"] = False
    r2 = ce.run_session(a.label, a.order)["region"]
    m = r1.merge(r2, on=["model", "window", "area"], suffixes=("_bundle", "_share"))
    diff = np.abs(m.r2_bundle - m.r2_share)
    print(f"{a.label}: {len(m)} region rows ({len(r1)} bundle / {len(r2)} local); max |r2 bundle - local| = "
          f"{diff.max():.2e}; rows differing > 1e-6: {(diff > 1e-6).sum()}; share reads during the bundle run: "
          f"{len(_SHARE_READS)} {sorted(set(_SHARE_READS))[:5]}")
    ok = diff.max() < 1e-4 and len(m) == len(r1) == len(r2) and not _SHARE_READS
    print("CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_collect(a) -> int:
    from wfield_local import combined_encoding as ce
    from wfield_local import writeguard
    from wfield_local.orofacial_cohort import RUN
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    os.environ[ce.RUN_ENV] = a.run or RUN
    d = bundle_dir(rv, a.name)
    _refuse_other_run(_bundle_meta(d))                                 # the bundle's run == the current profile's
    src = d / "out" / a.tag
    meta = json.loads((src / "run.json").read_text())
    for k, v in (meta.get("bundle_env") or {}).items():        # same input settings -> same folder name as a local run
        if not (isinstance(v, str) and v.startswith("@")):
            os.environ[k] = v
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
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    rv = PathResolver()
    run = a.run or oc.RUN
    d = bundle_dir(rv, a.name)
    labels = [x.strip() for x in (d / "labels.txt").read_text().splitlines() if x.strip()]
    n = {}
    for lab in labels:
        ep = epochs.epoch_of(lab)
        if ep not in ("pre", "acute", "subacute", "chronic") or (ep == "pre" and lab[5:] < oc.LATE_PRE_FROM):
            continue
        sd = oo.session_dir(rv, run, lab.split("_")[0], "2026" + lab[5:])
        keep, _ = oc.engagement_keep(lab.split("_")[0], "2026" + lab[5:], sd)
        L = pd.read_csv(sd / "licks.csv")
        if keep is not None:
            L = L[L.trial_id.isin(keep)]
        ok = (~L.peak_interp.astype(bool)) & (L.protrusion_px >= ce.MIN_REACH_PX) & np.isfinite(L.peak_angle_deg)
        n[lab] = int(ok.sum())
    S = pd.Series(n)
    tgt = {lab: int(a.frac * S[S.index.str[:4] == lab.split("_")[0]].min()) for lab in S.index}
    (d / "match.json").write_text(json.dumps(tgt, indent=1), encoding="utf-8")
    for an in sorted({x[:4] for x in S.index}):
        s = S[S.index.str[:4] == an]
        print(f"{an}: {len(s)} sessions, usable licks {s.min()}..{s.max()} -> target {tgt[s.idxmin()]}")
    print(f"-> {d / 'match.json'} ({len(tgt)} sessions; June pre sessions not matched = left out of the matched run)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="", help="render profile applied BEFORE anything is imported (e.g. "
                    "cohort1_rebuild_20261103): the bundle records its settings, run tag and rebuild-root location")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--name", required=True)
    b.add_argument("--user", required=True)
    b.add_argument("--run", default=None)
    b.add_argument("--cams", default="cam1,cam4", help="video motion energy to bundle where it exists ('' = none)")
    b.add_argument("--fr", default="", help="FaceRhythm factor set to bundle where projected, e.g. master_pre_v1")
    b.add_argument("--fr-points", type=int, default=0, help="with --fr: also K point-displacement PCs (frpts)")
    b.add_argument("--fr-share", action="store_true", help="with --fr: bundle the per-frame SHARE (the paper) instead "
                   "of absolute expression (a bundle holds one; make a second bundle for the other)")
    b.add_argument("--jobs", type=int, default=4)
    b.add_argument("--no-inputs", action="store_true", help="refresh code / env / COMMANDS only (inputs kept)")
    b.add_argument("labels", nargs="*", help="session labels, or ONE file of whitespace-separated labels")
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
    if a.profile:
        from wfield_local import render_profile
        render_profile.apply(a.profile)
        os.environ["O2_BUNDLE_PROFILE"] = a.profile
    if a.cmd == "bundle":
        cmd_bundle(a)
        return 0
    if a.cmd == "match":
        return cmd_match(a)
    return cmd_check(a) if a.cmd == "check" else cmd_collect(a)


if __name__ == "__main__":
    raise SystemExit(main())
