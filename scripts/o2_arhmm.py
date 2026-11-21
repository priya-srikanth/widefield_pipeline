"""The AR-HMM (`wfield_local/arhmm_model.py`) on HMS O2 (Priya 11-16: "we can run on o2"). Same layout as
`scripts/o2_encoding.py`: a bundle on the share that the transfer node pulls, a code snapshot, parametrised sbatch
scripts and COMMANDS.md with the exact lines.

    python scripts/o2_arhmm.py bundle --name arhmm_20261116 --user ps150 --feat <features dir>

bundle (this desktop): <DeepLabCut/Widefield>/o2/<name>/
    inputs/features/sessions/*.npz   the `arhmm_features` matrices (25 Hz; encoder behaviour variables + FR)
    inputs/epochs.json               {label: epoch} (written here; the arhmm env has no wfield config)
    code/                            `git archive HEAD` of wfield_local + configs (refused if wfield_local is dirty)
    scan.sbatch / fit.sbatch / decode.sbatch, labels.txt, COMMANDS.md
Runs (O2 conda env `arhmm`, CPU JAX + jax-moseq; COMMANDS.md creates it once):
    scan    array over KAPPAS (one kappa per task) on the pooled_all fit set -> out/scan_*.json (median syllable)
    fit     array over seeds, FRAME (pooled_all | pre), KAPPA, PER_CELL -> out/model_<frame>_s<seed>.pkl
    decode  array over sessions with TAG = a fitted model -> out/states_<TAG>/<label>.npz
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

KAPPAS = ("1e5", "1e6", "1e7", "1e8", "1e9", "1e10", "1e11")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--name", required=True)
    b.add_argument("--user", required=True)
    b.add_argument("--feat", required=True, help="arhmm_features output dir (holds sessions/)")
    a = ap.parse_args(argv)
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs as ep
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    repo = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "wfield_local", "configs"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing: wfield_local / configs have uncommitted changes:\n{dirty}")
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    d = oi.bundle_dir(rv, a.name)
    src = Path(a.feat) / "sessions"
    files = sorted(src.glob("*.npz"))
    if not files:
        raise SystemExit(f"no feature files in {src}")
    dst = d / "inputs" / "features" / "sessions"
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        t = dst / f.name
        if not t.exists() or t.stat().st_size != f.stat().st_size:
            shutil.copy2(f, t)
    labels = [f.stem for f in files]
    (d / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")
    (d / "inputs" / "epochs.json").write_text(json.dumps({lab: ep.epoch_of(lab) for lab in sorted(ak.curated_ids())}))
    code = d / "code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir()
    tar = d / "code.tar"
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tar), "HEAD", "wfield_local", "configs"],
                   check=True)
    shutil.unpack_archive(str(tar), str(code))
    tar.unlink()
    (d / "logs").mkdir(exist_ok=True)
    s = oi.scratch(a.user)
    o = f"{s}/o2/{a.name}"
    head = f"""#SBATCH -o {o}/logs/%x_%A_%a.out
#SBATCH -e {o}/logs/%x_%A_%a.err
set -eo pipefail
module load conda/miniforge3/26.9.4-1
conda activate arhmm
export PYTHONPATH={o}/code JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
COMMON="--feat {o}/inputs/features --out {o}/out --epochs {o}/inputs/epochs.json"
"""
    kap = ",".join(KAPPAS)
    scripts = {
        "scan.sbatch": f"""#SBATCH -J arhmm_scan_{a.name}
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00
{head}KS=({' '.join(KAPPAS)})
K=${{KS[$((SLURM_ARRAY_TASK_ID-1))]}}
python -u -m wfield_local.arhmm_model scan $COMMON --frame pooled_all --kappas $K --iters ${{ITERS:-30}} \\
    --scan-sessions ${{SCAN_SESSIONS:-6}}
""",
        "fit.sbatch": f"""#SBATCH -J arhmm_fit_{a.name}
#SBATCH -p medium
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
{head}python -u -m wfield_local.arhmm_model fit $COMMON --frame $FRAME --kappa $KAPPA --iters ${{ITERS:-100}} \\
    --per-cell ${{PER_CELL:-2}} --seed $SLURM_ARRAY_TASK_ID
""",
        "foldfit.sbatch": f"""#SBATCH -J arhmm_fold_{a.name}
#SBATCH -p medium
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
{head}python -u -m wfield_local.arhmm_model fit $COMMON --frame pre --kappa $KAPPA --iters ${{ITERS:-100}} \\
    --seed 1 --holdout-fold $SLURM_ARRAY_TASK_ID
""",
        "decode.sbatch": f"""#SBATCH -J arhmm_dec_{a.name}
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-02:00:00
{head}L=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {o}/labels.txt)
python -u -m wfield_local.arhmm_model decode $COMMON --tag $TAG --label $L
""",
    }
    for n, txt in scripts.items():
        (d / n).write_text("#!/bin/bash -l\n" + txt, encoding="utf-8", newline="\n")   # shebang first, then #SBATCH
    share_b = oi.o2_path(d, rv)
    n = len(labels)
    cmds = f"""# O2 AR-HMM: {a.name}

{n} sessions; code {commit}; features {src}
Design + decisions: docs/status/ENCODING_DESIGN_TIM_STATES_2026-11-16.md (section AR-HMM); model `wfield_local/arhmm_model.py`.

## 0. Login node, ONCE: the `arhmm` conda env (CPU JAX + jax-moseq; OpenBLAS -- the MKL build crashed on the desktop)
```bash
module load conda/miniforge3/26.9.4-1
conda create -y -n arhmm -c conda-forge python=3.11 numpy scipy pandas pyarrow h5py pyyaml scikit-learn \\
    "libblas=*=*openblas" "liblapack=*=*openblas"
conda activate arhmm && pip install "jax[cpu]" jax-moseq
python -c "import numpy as np, jax, jax_moseq; np.linalg.eigh(np.eye(3)); print('ok', jax.__version__)"
```

## 1. Transfer node  (ssh {a.user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {o}
rsync -ah {share_b}/ {o}/
ls {o}/inputs/features/sessions | wc -l      # expect {n}
```

## 2. Login node: kappa scan ({len(KAPPAS)} tasks, one kappa each: {kap})
```bash
sbatch --array=1-{len(KAPPAS)} {o}/scan.sbatch
squeue -u {a.user}
cat {o}/out/scan_pooled_all_k*.json | grep -E '"kappa"|"median_s"'
```
Pick the kappa whose median syllable is ~0.5 s (tell the desktop window; it records the choice).

## 3. Login node: fits (3 seeds each) at the chosen kappa
```bash
sbatch --array=1-3 --export=ALL,FRAME=pooled_all,KAPPA=<kappa>,PER_CELL=2 {o}/fit.sbatch
sbatch --array=1-3 --export=ALL,FRAME=pre,KAPPA=<kappa> {o}/fit.sbatch
```

## 3b. Login node: held-out pre baseline (8 folds; fold k drops each animal's k-th pre day; seed 1)
```bash
sbatch --array=1-8 --export=ALL,KAPPA=5e7 {o}/foldfit.sbatch
```
Outputs `out/model_pre_s1_f<k>.pkl`; the desktop decodes + scores each fold's held-out days locally.

## 4. Login node: decode every session with each fitted model (TAG = <frame>_s<seed>)
```bash
sbatch --array=1-{n}%40 --export=ALL,TAG=pooled_all_s1 {o}/decode.sbatch
sbatch --array=1-{n}%40 --export=ALL,TAG=pre_s1 {o}/decode.sbatch
```

## 5. Transfer node, when done
```bash
rsync -ah {o}/out/ {share_b}/out/
```
"""
    (d / "COMMANDS.md").write_text(cmds, encoding="utf-8", newline="\n")
    (d / "bundle.json").write_text(json.dumps({"name": a.name, "commit": commit, "n_sessions": n,
                                               "features": str(src), "kappas": list(KAPPAS)}, indent=1))
    print(f"{n} sessions bundled -> {d}\n\n{cmds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
