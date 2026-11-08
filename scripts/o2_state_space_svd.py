"""Per-trial first-lick activity in ONE COHORT SVD basis, on HMS O2 (state-space arm; Priya 11-08: "what happened to
using svd" / "could we use one svd basis" -> "yes").

WHY. The joint-LocaNMF basis (`o2_state_space_lnmf.py`) is per ANIMAL, so its components differ across animals and every
cross-animal readout (the cohort loops) needs an alignment step (`state_space_trajectories.align_maps`, Procrustes on
pre; leave-one-animal-out r .79-.86 there vs .92-.98 in Allen areas). Every session's U_atlas is on the SAME Allen
pixel grid (affine8v1), so one basis can serve all four animals: component k is the same pixel pattern in every animal.

WHAT. `joint_basis`'s exact trick, across animals instead of within one. Basis job (stage 1, one task): for each AUGUST
PRE session s of all four animals (rows epoch == "pre"; June excluded, the August-only baseline rule), V_s = the adopted
SVTcorr with the slow drift removed (`combined_encoding._slow_basis`, as the extraction), block_s = U_s (V_s^T V_s)^(1/2)
scaled so every ANIMAL carries the same total variance (each session ||block||_F = 1 / sqrt(n pre sessions of its
animal)); U_cohort = top KC left singular vectors of [block_1 | block_2 | ...] (via the Gram of the stack; KC x 345600
pixels). Per component: share of its footprint energy in the left hemisphere (atlas > 0 = left, the convention of
`state_space_sim._cols`) -> side left (> 2/3) / right (< 1/3) / both. Projection (stage 2, array, after stage 1):
C_s = (U_cohort^T U_s) V_s^T, drift removed, sampled at the SAME rows / lags as the LocaNMF files (rows reused from that
bundle, so row i of every basis is the same trial). variance_captured as `joint_locanmf.Basis.project_uv`.

    python scripts/o2_state_space_svd.py bundle --name ss_svd_20261108 [--user ps150] [--kc 100]
    python scripts/o2_state_space_svd.py bundle --name ss_svdbm_20261108 --local &&         python scripts/o2_state_space_svd.py run-local --name ss_svdbm_20261108      (here, inputs off the share)

PIXELS (Priya 11-08 "we should be looking only at the brain mask (without glue areas)"): `beta_maps.brain_mask()` --
the Allen brain mask minus the olfactory bulbs and the hand-painted fibre glue (union over animals, the pooled-figure
rule), the mask the encoder fits and the map statistics use. The first run (ss_svd_20261108) used every pixel of the
frame: off-brain frame edges entered the basis (PS94_0917: 47 % of its variance off-brain, early-session) -- superseded.

Inputs on O2 scratch are the LocaNMF bundle's (uatlas/, June inputs/) and the encoding bundle's inputs/ -- nothing is
re-staged. Output -> rsync back to <bundle>/out/: basis/U_cohort.npy (npix, KC) + basis.npz (singular values, sides,
labels), sessions/<label>.npz (keys as the LocaNMF files + cols = svd#<k>_<side>). EXPLORATORY.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o2_state_space_lnmf as L  # noqa: E402

LNMF_NAME = "ss_lnmf_20261108"                     # rows, uatlas, June inputs reused from this bundle
KC = 100

COMMON = r'''import json, os, pickle, sys, time
from pathlib import Path

import numpy as np

b = Path(__file__).resolve().parent
sys.path.insert(0, str(b / "code"))
meta = json.loads((b / "bundle.json").read_text())
for k in [k for k in os.environ if k.startswith(("WIDEFIELD_", "WFIELD_", "LOCANMF_", "COMBINED_ENCODING_", "OROFACIAL_"))]:
    del os.environ[k]
for k, v in json.loads((b / "env.json").read_text()).items():
    os.environ[k] = v
from wfield_local import combined_encoding as ce
if ce.DRIFT_MIN_PERIOD_S != meta["drift_min_period_s"]:
    sys.exit(f"drift period {ce.DRIFT_MIN_PERIOD_S} != bundle {meta['drift_min_period_s']}")
lb = Path(meta["lnmf_scratch"])


def load(label):
    """(u (npix, K) float32, V (T, K) drift-removed float64, ft) -- the inputs of the LocaNMF extraction."""
    src = lb / "inputs" / f"{label}.pkl"                              # June: the LocaNMF bundle's own inputs
    if not src.exists():
        src = Path(meta["enc_scratch"]) / "inputs" / f"{label}.pkl"
    x = pickle.loads(src.read_bytes())
    V, ft = np.asarray(x["imaging"][0], np.float64), np.asarray(x["imaging"][1], float)
    del x
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    V -= Q @ (Q.T @ V)
    U = np.load(Path(meta.get("uatlas_root") or lb / "uatlas") / meta["uatlas"][label])
    u = np.nan_to_num(U.reshape(-1, U.shape[2])).astype(np.float32)[MASK]          # brain-mask pixels only
    return u, V, ft


MASK = np.load(b / "mask.npy").ravel().astype(bool)
'''

BASIS_RUNNER = '''"""Runs on O2 (stage 1, one task): the cohort SVD basis from the August pre sessions of all animals.
See scripts/o2_state_space_svd.py for the definitions."""
''' + COMMON + r'''
t0 = time.time()
pre = meta["pre_labels"]
n_an = {}
for lab in pre:
    n_an[lab.split("_")[0]] = n_an.get(lab.split("_")[0], 0) + 1
blocks = []
for lab in pre:
    u, V, _ = load(lab)
    w, E = np.linalg.eigh(V.T @ V)
    blk = u @ ((E * np.sqrt(np.clip(w, 0, None))) @ E.T).astype(np.float32)          # U_s (V_s^T V_s)^(1/2)
    blk /= np.linalg.norm(blk) * np.sqrt(n_an[lab.split("_")[0]])
    blocks.append(blk)
    print(f"  {lab} loaded ({time.time() - t0:.0f} s)", flush=True)
B = np.concatenate(blocks, axis=1)
del blocks
G = (B.T @ B).astype(np.float64)
lam, E = np.linalg.eigh(G)
lam, E = lam[::-1], E[:, ::-1]
kc = meta["kc"]
Uc = (B @ (E[:, :kc] / np.sqrt(lam[:kc])).astype(np.float32)).astype(np.float32)        # npix x kc, orthonormal
atlas = np.load(b / "atlas.npy").ravel()[MASK]
e = Uc ** 2
left = e[atlas > 0].sum(0) / np.maximum(e[atlas != 0].sum(0), 1e-30)
side = np.where(left > 2 / 3, "left", np.where(left < 1 / 3, "right", "both"))
o = b / "out" / "basis"
o.mkdir(parents=True, exist_ok=True)
np.save(o / "U_cohort.npy", Uc)
np.save(o / "mask.npy", MASK)
np.savez(o / "basis.npz", sv=np.sqrt(np.clip(lam, 0, None)), var_ratio=lam / lam.sum(), side=side, left_share=left,
         labels=np.array(pre), kc=kc)
print(f"cohort basis: {len(pre)} pre sessions, {B.shape[1]} columns, top {kc} = {lam[:kc].sum() / lam.sum():.4f} of the "
      f"weighted pre variance; sides {dict(zip(*np.unique(side, return_counts=True)))}; {time.time() - t0:.0f} s", flush=True)
'''

RUNNER = '''"""Runs on O2 (stage 2, task = SLURM_ARRAY_TASK_ID): one session projected onto the cohort SVD basis.
See scripts/o2_state_space_svd.py for the definitions."""
''' + COMMON + r'''
task = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
label = [x.strip() for x in (b / "labels.txt").read_text().splitlines() if x.strip()][task - 1]
out = b / "out" / "sessions" / f"{label}.npz"
if out.exists():
    print("done already", label); sys.exit(0)
t0 = time.time()
rows = np.load(b / "rows" / f"{label}.npz", allow_pickle=True)
u, V, ft = load(label)
Uc = np.load(b / "out" / "basis" / "U_cohort.npy")
bz = np.load(b / "out" / "basis" / "basis.npz")
P = (Uc.T @ u).astype(np.float64)                                   # (kc, K)
R = V @ P.T                                                         # (T, kc); drift already removed (linear)
W = V.T @ V
var = float(np.sum((P.T @ P) * W) / max(np.sum((u.T @ u).astype(np.float64) * W), 1e-30))
cols = np.array([f"svd#{k + 1}_{s}" for k, s in enumerate(bz["side"])])


def sample(t, lags):                                                # iti_licks._sample (NaN where t is NaN)
    o = np.full((len(t), len(lags), R.shape[1]), np.nan, np.float32)
    ok = np.isfinite(t) & (t + lags[0] > ft[0]) & (t + lags[-1] < ft[-1])
    tt = (t[ok][:, None] + lags[None, :]).ravel()
    for j in range(R.shape[1]):
        o[ok, :, j] = np.interp(tt, ft, R[:, j]).reshape(ok.sum(), len(lags))
    return o


cue, lick = rows["cue_s"].astype(float), rows["lick_s"].astype(float)
an = label.split("_")[0]
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, first_cue=sample(cue, rows["lags_c"]), first_peth=sample(lick, rows["lags_p"]),
                    lags_c=rows["lags_c"], lags_p=rows["lags_p"], first_rt=rows["first_rt"], first_pos=rows["first_pos"],
                    cue_s=cue, lick_s=lick, src=rows["src"], epoch=str(rows["epoch"]), animal=an,
                    basis_id=meta["basis_id"], variance_captured=var, cols=cols)
print(f"{label}: {len(cue)} rows ({int(np.isfinite(lick).sum())} timed), {R.shape[1]} comps, var captured {var:.3f}, "
      f"{time.time() - t0:.0f} s", flush=True)
'''


def analysis_mask():
    """`beta_maps.brain_mask()` (Allen brain - bulbs - painted glue, union) resolved WITHOUT the rebuild profile, and
    refused if the glue is missing. Found 11-08: under the profile `paint_exclusion.mask_dir` points at the rebuild's
    own grant_figures, where the hand-painted masks do not exist, so `painted_exclusion` silently returns an empty mask
    (192,110 px instead of 151,447: glue IN). Call before `L._env`."""
    from wfield_local import beta_maps
    painted = beta_maps.painted_exclusion()
    if not painted.any():
        raise SystemExit("painted glue mask is empty (profile applied? paint_exclusion.mask_dir) -- refusing to build")
    return beta_maps.brain_mask()


def cmd_bundle(a) -> None:
    mask = analysis_mask()                                             # before the profile (see analysis_mask)
    L._env(L.CACHE)                                                    # the rebuild profile: bundle_dir under its root
    import numpy as np

    from wfield_local import config, o2_bundle, writeguard
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    repo = Path(__file__).resolve().parents[1]
    dirty = L._git(repo, "status", "--porcelain", "--", "wfield_local", "configs")
    if dirty:
        raise SystemExit(f"uncommitted changes in {repo} (the bundle archives HEAD):\n{dirty}")
    import o2_encoding as oe
    src = oe.bundle_dir(rv, LNMF_NAME)
    d = oe.bundle_dir(rv, a.name)
    writeguard.assert_writable(d)
    for sub in ("logs", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    if not (d / "rows").exists():
        shutil.copytree(src / "rows", d / "rows")                      # our own small files: same trials as LocaNMF
    smeta = json.loads((src / "bundle.json").read_text())
    labels = [x for x in (src / "labels.txt").read_text().split() if x]
    pre = [lab for lab in labels if str(np.load(d / "rows" / f"{lab}.npz", allow_pickle=True)["epoch"]) == "pre"]
    sess = {s["label"]: s for s in config.load_sessions()}
    at = Path(sess[pre[0]]["mc"]) / "wfield_local_results" / "allen_aligned_affine8v1" / "allen_area_atlas_native_grid.npy"
    shutil.copyfile(at, d / "atlas.npy")
    np.save(d / "mask.npy", mask)                                      # Allen brain - bulbs - painted glue (union)
    (d / "labels.txt").write_text("".join(f"{x}\n" for x in labels), encoding="utf-8", newline="\n")
    shutil.copyfile(src / "env.json", d / "env.json")                  # same profile settings as the LocaNMF run
    s = oi.scratch(a.user)
    b = f"{s}/o2/{a.name}"
    commit = L._git(repo, "rev-parse", "HEAD")
    meta = {"name": a.name, "commit": commit, "code_root": str(repo), "profile": L.PROFILE, "kc": a.kc,
            "basis_id": f"cohort_svd_pre{len(pre)}_k{a.kc}", "pre_labels": pre, "n_sessions": len(labels),
            "lnmf_bundle": LNMF_NAME, "lnmf_scratch": str(src) if a.local else f"{s}/o2/{LNMF_NAME}",
            "enc_scratch": str(L.ENC_SHARE) if a.local else smeta["enc_scratch"],
            "uatlas_root": str(Path(rv.root("microscope"))) if a.local else None, "mask_px": int(mask.sum()),
            "drift_min_period_s": smeta["drift_min_period_s"], "uatlas": smeta["uatlas"],
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    if json.loads(json.dumps(o2_bundle.settings())) != json.loads((d / "env.json").read_text()):
        print("  NOTE: the current profile settings differ from the LocaNMF bundle's env.json -- using the bundle's")
    (d / "bundle.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    code = d / "code"
    if code.exists():
        shutil.rmtree(code)
    code.mkdir()
    tar = d / "code.tar"
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tar), commit, "wfield_local", "configs"],
                   check=True)
    shutil.unpack_archive(str(tar), str(code))
    tar.unlink()
    (d / "basis_runner.py").write_text(BASIS_RUNNER, encoding="utf-8", newline="\n")
    (d / "runner.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    head = """#!/bin/bash -l
#SBATCH -p short
#SBATCH -o {b}/logs/%x_%A_%a.out
#SBATCH -e {b}/logs/%x_%A_%a.err
"""
    tail = """set -eo pipefail
module load conda/miniforge3/26.9.4-1
conda activate lp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
"""
    (d / "basis.sbatch").write_text(head.format(b=b) + f"""#SBATCH -J ssb_{a.name}
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-03:00:00
""" + tail + f"python {b}/basis_runner.py\n", encoding="utf-8", newline="\n")
    (d / "run.sbatch").write_text(head.format(b=b) + f"""#SBATCH -J ss_{a.name}
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00
""" + tail + f"python {b}/runner.py\n", encoding="utf-8", newline="\n")
    share_b = oi.o2_path(d, rv)
    n = len(labels)
    cmds = f"""# O2 state-space cohort-SVD extraction: {a.name}

{n} sessions; basis from {len(pre)} August pre sessions (all four animals, equal weight per animal), top {a.kc};
code {commit}; rows / uatlas / June inputs from the LocaNMF bundle on scratch ({LNMF_NAME}), imaging from
{Path(smeta['enc_scratch']).name}/inputs.

## 1. Transfer node  (ssh {a.user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b}
rsync -ah {share_b}/ {b}/
find {s}/o2/{LNMF_NAME}/uatlas -name U_atlas.npy | wc -l      # expect {n}; else re-run the LocaNMF COMMANDS.md step 1
ls {smeta['enc_scratch']}/inputs | wc -l                        # expect 107
```

## 2. Login node  (ssh {a.user}@o2.hms.harvard.edu)
```bash
J=$(sbatch --parsable {b}/basis.sbatch)                         # stage 1: the basis (one task, ~30-60 min)
sbatch --dependency=afterok:$J --array=1-{n}%40 {b}/run.sbatch  # stage 2: every session, after the basis
squeue -u {a.user}
tail -n 2 {b}/logs/ssb_{a.name}_*.out                           # "cohort basis: ... top {a.kc} = ... sides ..."
ls {b}/out/sessions | wc -l                                     # expect {n}
```

## 3. Transfer node, when done
```bash
rsync -ah {b}/out/ {share_b}/out/
```
Desktop reads `{d}/out/sessions/<label>.npz` with `WF_SS_BASIS=svd`.
"""
    (d / "COMMANDS.md").write_text(cmds, encoding="utf-8", newline="\n")
    print(f"{n} sessions bundled -> {d}\n{cmds}")


def cmd_run_local(a) -> None:
    """The bundle's own runners, here: basis_runner.py once, then runner.py per session, `jobs` at a time (each a
    separate process, BLAS pinned to 4 threads). Same code as on O2; only the input paths differ (bundle --local)."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    import o2_encoding as oe
    from wfield_local.paths import PathResolver
    L._env(L.CACHE)
    d = oe.bundle_dir(PathResolver(), a.name)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    (d / "logs").mkdir(exist_ok=True)
    if not (d / "out" / "basis" / "U_cohort.npy").exists():
        with open(d / "logs" / "basis.log", "w") as f:
            subprocess.run([sys.executable, str(d / "basis_runner.py")], stdout=f, stderr=subprocess.STDOUT, env=env,
                           check=True)
    n = len((d / "labels.txt").read_text().split())

    def one(i):
        with open(d / "logs" / f"task_{i}.log", "w") as f:
            return subprocess.run([sys.executable, str(d / "runner.py"), str(i)], stdout=f, stderr=subprocess.STDOUT,
                                  env=env).returncode
    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(one, range(1, n + 1)))
    bad = [i for i, c in enumerate(codes, 1) if c]
    print(f"{n - len(bad)} / {n} sessions ok" + (f"; FAILED tasks {bad} (logs/task_<i>.log)" if bad else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bundle")
    p.add_argument("--name", required=True)
    p.add_argument("--user", default="ps150")
    p.add_argument("--kc", type=int, default=KC)
    p.add_argument("--local", action="store_true", help="paths on the share: run here with run-local")
    p = sp.add_parser("run-local", help="basis then every session on this box (inputs read off the share)")
    p.add_argument("--name", required=True)
    p.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "bundle":
        cmd_bundle(a)
    else:
        cmd_run_local(a)


if __name__ == "__main__":
    main()
