"""June (06/06-06/08) FaceRhythm sessions on the FROZEN pre-stroke factor set ``master_pre_v1`` (11-08, Priya: June as
a reference epoch).

The frozen set's ``project`` stage reads its task list from the set's own ``plan.json``, which holds no June sessions,
and its outputs from the production bundle's ``out/``. So this builds a SEPARATE set folder inside the June bundle,
``o2/fr_cam4_june_20261108/factors/master_pre_v1_june/``:
  - ``master.npz`` = a byte copy of ``master_pre_v1/master.npz`` (the frozen basis, per-animal scales, point PCs; the
    spec hash travels with it), never refitted;
  - ``plan.json`` = the frozen plan's params + the 12 June tasks (epoch "june", basis "full": June was never in the fit).
Then ``project`` runs per task (``--dir`` that folder; ``out_dir`` resolves to the June bundle's ``out/``) ->
``master_pre_v1_june/proj/<key>.npz``, the same fields as the production ``proj/``. ``master_pre_v1`` is not touched.

SCALE: the June face is ~1.13x larger in image px than August (camera scale 1.129; `face_rhythm_register` JUNE notes),
so June displacement spectra -- and the projection weights -- are inflated by about that factor. The projection is
kept RAW here; readouts divide by the session's face scale (``register/qc_june_fields.csv``) or use scale-free ratios.

    python scripts/face_rhythm_june_project.py [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BUNDLE, PROD, SET = "fr_cam4_june_20261108", "fr_cam4_20261021", "master_pre_v1"
REPO = Path(__file__).resolve().parent.parent


def set_dir() -> tuple[Path, Path]:
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    return oi.bundle_dir(rv, PROD) / "factors" / SET, oi.bundle_dir(rv, BUNDLE) / "factors" / f"{SET}_june"


def build() -> Path:
    from wfield_local import writeguard
    src, dst = set_dir()
    writeguard.assert_writable(dst)
    dst.mkdir(parents=True, exist_ok=True)
    m = dst / "master.npz"
    if not m.exists() or m.stat().st_size != (src / "master.npz").stat().st_size:
        shutil.copy2(src / "master.npz", m)
    P = json.loads((src / "plan.json").read_text())
    T = pd.read_csv(dst.parent.parent / "tasks.tsv", sep="\t", dtype=str)
    plan = {"set": f"{SET}_june", "bundle": BUNDLE, "source_set": f"{PROD}/factors/{SET}", "role": "projection only",
            "animals": sorted(T.animal.unique()), "loso_jobs": False, "params": P["params"],
            "tasks": [{"key": r.key, "animal": r.animal, "date": r.date, "epoch": "june"} for r in T.itertuples()]}
    (dst / "plan.json").write_text(json.dumps(plan, indent=1))
    return dst


def project(sd: Path, task: int) -> str:
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "face_rhythm_factors.py"), "project", "--dir", str(sd),
                        "--task", str(task)], capture_output=True, text=True)
    return (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else f"task {task} FAILED: {r.stderr[-800:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2, help="tasks in parallel (RAM-bound shared box: 2)")
    a = ap.parse_args()
    sd = build()
    n = len(json.loads((sd / "plan.json").read_text())["tasks"])
    with ThreadPoolExecutor(a.jobs) as ex:
        for line in ex.map(lambda t: project(sd, t), range(1, n + 1)):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
