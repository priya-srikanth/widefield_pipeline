"""Per-trial, time-resolved first-lick activity in the JOINT-LocaNMF basis, on HMS O2 (state-space arm, option 4; Priya
11-07: "we can cue to run on o2"; 11-08: "add june where possible ... may need to use the contact-only trials").

WHY. The state-space analyses (`state_space_*.py`) read Allen-area means (62 areas) of the rebuild's per-trial extract
(`iti_licks.extract`: first_cue / first_peth). The finer, functionally defined basis (each animal's PINNED joint-LocaNMF
footprints, `joint_locanmf`, ~90 components) exists per trial only around ITI / random post-cue licks
(`iti_licks/basis_locanmf`, pre-rebuild, no first-lick / spout identity), and the decoder caches are time-averaged.

WHAT. Per session: project the adopted SVTcorr onto the animal's pinned joint basis (`joint_locanmf.Basis.project_uv`,
the arithmetic of `Basis.project`), remove the slow drift (`combined_encoding._slow_basis`, as `iti_basis_decoder.features`),
and sample every first-lick row at `iti_licks.LAGS_C` from the cue and `iti_licks.LAGS_P` from the lick onset.
  August + later (the rebuild set, 107 sessions): rows = EXACTLY the rows of the Allen extract (`state_space_sim.ITI`),
  timed by the run's `cued_first_lick_times.csv` (`iti_licks._first_lick_times`, cue = onset - first_rt), so row i of a
  LocaNMF file is row i of the Allen file (landing join, per-trial tables carry over). Imaging from the rebuild's O2 encoding
  bundle (`enc_rebuild_20261104b`, inputs/<label>.pkl on scratch: same profile, same hemo variant).
  June pre (`--june`, 6/6-6/8 x 4 animals; no-stroke drift reference, includes learning -> an upper bound): no 3-D /
  camera first licks of the rebuild quality, so rows = CONTACT-ONLY: first DAQ spout contact (`Behaviour.contact_s`) in
  cue .. stop of engaged trials (iti_licks' rule with contacts for licks). Their V / ft travel in this bundle (inputs/).
Desktop work: August rows from tables only; June loads behaviour + imaging per session (~2.5 min each, 12 sessions; the
local input cache misses across code versions). The heavy reads (U_atlas, ~135 MB x 119; joint bases) happen on the O2
transfer node, share -> scratch.

    python scripts/o2_state_space_lnmf.py bundle --name ss_lnmf_20261108 --user ps150 [--june] [--jobs 3] [LABELS ...]

Output on O2 -> rsync back to <bundle>/out/sessions/<label>.npz (first_cue (n, LAGS_C, ncomp), first_peth (n, LAGS_P,
ncomp), first_rt, first_pos, cue_s, lick_s, src, basis_id, variance_captured, epoch, animal). Units: LocaNMF component
amplitude (footprint-scaled, `_footprint_scale`), NOT x1e3 dF/F -- the state-space transforms are per-animal anyway.
EXPLORATORY (state-space branch); bundle -> <rebuild root>/o2/<name> (`o2_encoding.bundle_dir`).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROFILE = "cohort1_rebuild_20261103"
ENC = "enc_rebuild_20261104b"                       # the rebuild's O2 encoding bundle (its inputs/ are reused)
ENC_SHARE = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/o2") / ENC
CACHE = "C:/Users/SabatiniLab/wfield_cache/input_cache_rebuild_20261104"
JUNE = ("0606", "0607", "0608")                     # the curated June pre days (configs/animals.yaml date_policy)
TIMES = ("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
         "iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-"
         "incomplete-stopA2jexrt3500c70_daqlk/cued_first_lick_times.csv")   # rebuild run's OUT_W (cleanup item 1: route)

RUNNER = r'''"""Runs on O2: one session (task = SLURM_ARRAY_TASK_ID) of the state-space joint-LocaNMF extraction.
See scripts/o2_state_space_lnmf.py (bundle side) for the definitions."""
import json, os, pickle, sys, time
from pathlib import Path

import numpy as np

b = Path(__file__).resolve().parent
sys.path.insert(0, str(b / "code"))
task = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
label = [x.strip() for x in (b / "labels.txt").read_text().splitlines() if x.strip()][task - 1]
meta = json.loads((b / "bundle.json").read_text())
for k in [k for k in os.environ if k.startswith(("WIDEFIELD_", "WFIELD_", "LOCANMF_", "COMBINED_ENCODING_", "OROFACIAL_"))]:
    del os.environ[k]
for k, v in json.loads((b / "env.json").read_text()).items():   # the profile settings (hemo variant, drift period, ...)
    os.environ[k] = v
out = b / "out" / "sessions" / f"{label}.npz"
if out.exists():
    print("done already", label); sys.exit(0)
from wfield_local import combined_encoding as ce
from wfield_local import joint_locanmf as jl
if ce.DRIFT_MIN_PERIOD_S != meta["drift_min_period_s"]:
    sys.exit(f"drift period {ce.DRIFT_MIN_PERIOD_S} != bundle {meta['drift_min_period_s']}")
t0 = time.time()
rows = np.load(b / "rows" / f"{label}.npz")
src = b / "inputs" / f"{label}.pkl"                                    # June: shipped in this bundle
if not src.exists():
    src = Path(meta["enc_scratch"]) / "inputs" / f"{label}.pkl"        # rebuild set: the encoding bundle's inputs
x = pickle.loads(src.read_bytes())
V, ft = x["imaging"][0], np.asarray(x["imaging"][1], float)           # V (T, K) adopted SVTcorr (truncated to ft)
del x
an = label.split("_")[0]
U = np.load(b / "uatlas" / meta["uatlas"][label])
u = np.nan_to_num(U.reshape(-1, U.shape[2])).astype(np.float32)       # as joint_basis._load_session
del U
B = jl.Basis(b / "bases" / an / meta["bases"][an])
C, diag = B.project_uv(u, np.ascontiguousarray(V.T, dtype=np.float32), label, with_diagnostics=True)
R = np.nan_to_num(np.asarray(C, np.float64).T)                        # (T, ncomp)
Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]       # drift removal as iti_basis_decoder.features
R -= Q @ (Q.T @ R)


def sample(t, lags):                                                  # iti_licks._sample (NaN where t is NaN)
    o = np.full((len(t), len(lags), R.shape[1]), np.nan, np.float32)
    ok = np.isfinite(t) & (t + lags[0] > ft[0]) & (t + lags[-1] < ft[-1])
    tt = (t[ok][:, None] + lags[None, :]).ravel()
    for j in range(R.shape[1]):
        o[ok, :, j] = np.interp(tt, ft, R[:, j]).reshape(ok.sum(), len(lags))
    return o


cue, lick = rows["cue_s"].astype(float), rows["lick_s"].astype(float)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, first_cue=sample(cue, rows["lags_c"]), first_peth=sample(lick, rows["lags_p"]),
                    lags_c=rows["lags_c"], lags_p=rows["lags_p"], first_rt=rows["first_rt"], first_pos=rows["first_pos"],
                    cue_s=cue, lick_s=lick, src=rows["src"], epoch=rows["epoch"], animal=an, basis_id=B.basis_id,
                    variance_captured=diag["variance_captured"], regions=B.regions if (B.root / "regions.npy").exists()
                    else np.array([]))
print(f"{label}: {len(cue)} rows ({int(np.isfinite(lick).sum())} timed), {R.shape[1]} comps, "
      f"var captured {diag['variance_captured']:.3f}, {time.time() - t0:.0f} s", flush=True)
'''


def _env(cache: str) -> None:
    """The rebuild profile + the encoding bundle's COMBINED_ENCODING_* / OROFACIAL_* settings ('@' tables resolved in
    that bundle on the share), so behaviour / imaging hit the rebuild's local input cache (same keys)."""
    import o2_encoding as oe
    enc = json.loads((ENC_SHARE / "env.json").read_text())
    for k, v in enc.items():
        if k.startswith(oe.KEEP_PREFIXES):
            os.environ[k] = str(ENC_SHARE / v[1:]) if v.startswith("@") else v
    os.environ.setdefault("WFIELD_INPUT_CACHE", cache)
    oe.apply_profile(PROFILE)


def _rows_worker(item):
    """Module-level (spawn): one session -> rows/<label>.npz (+ inputs/<label>.pkl for June)."""
    label, d, june = item
    _env(os.environ.get("WFIELD_INPUT_CACHE", CACHE))
    import pickle

    import iti_licks as il
    import state_space_sim as S
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    d = Path(d)
    if june:
        # contact-only rows need no 3-D: June has no spout 3-D positions, so the encoder's 3-D executed / target / tongue
        # settings (read at call time) are dropped for these loads; trials, engagement and DAQ contacts are unaffected
        for k in (ce.EXEC_ENV, ce.EXEC3D_TABLE_ENV, ce.TARGET3D_TABLE_ENV, ce.TONGUE_ENV, ce.BINS_CALIB_ENV):
            os.environ.pop(k, None)
        B = il._behaviour(label)
        T = B.trials
        co = np.sort(np.asarray(B.contact_s, float)[np.isfinite(B.contact_s)])
        Eg = T[T.engaged & np.isfinite(T.cue_s)]
        cue, lick, pos = [], [], []
        for c, s, p in zip(Eg.cue_s.to_numpy(float), Eg.stop_s.to_numpy(float), Eg.pos.to_numpy()):
            s = s if np.isfinite(s) else c + 3.5                          # iti_licks.extract's rule, contacts for licks
            j = np.searchsorted(co, c)
            if j < len(co) and co[j] <= s:
                cue.append(c); lick.append(co[j]); pos.append(p)
        cue, lick = np.asarray(cue, float), np.asarray(lick, float)
        rt, pos, src = lick - cue, np.asarray(pos, str), "contact"
        V, ft = ce.load_imaging(label)[:2]
        (d / "inputs").mkdir(exist_ok=True)
        (d / "inputs" / f"{label}.pkl").write_bytes(pickle.dumps({"imaging": (V, ft)}, protocol=4))
    else:
        # first-lick onsets of every extract row, already computed by `iti_licks._first_lick_times` for this run
        # (lick_vigor_perspout.times_cache -> <OUT_W>/cued_first_lick_times.csv; row = the extract row); cue = t - rt
        import pandas as pd
        z = np.load(Path(S.ITI) / f"{label}.npz")
        rt, pos, src = z["first_rt"].astype(float), z["first_pos"].astype(str), "camera"
        t = pd.read_csv(TIMES, float_precision="round_trip").query("label == @label")
        lick = np.full(len(rt), np.nan)
        lick[t.row.to_numpy(int)] = t.t.to_numpy(float)
        cue = lick - rt
    np.savez(d / "rows" / f"{label}.npz", cue_s=cue, lick_s=lick, first_rt=rt, first_pos=pos, src=src,
             epoch=epochs.epoch_of(label), lags_c=il.LAGS_C, lags_p=il.LAGS_P)
    return {"n": len(rt), "matched": int(np.isfinite(lick).sum())}


def _git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def cmd_bundle(a) -> None:
    _env(a.cache)
    import state_space_sim as S
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import config, joint_locanmf, o2_bundle, writeguard
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    repo = Path(__file__).resolve().parents[1]
    dirty = _git(repo, "status", "--porcelain", "--", "wfield_local", "configs")
    if dirty:
        raise SystemExit(f"uncommitted changes in {repo} (the bundle archives HEAD):\n{dirty}")
    import o2_encoding as oe
    d = oe.bundle_dir(rv, a.name)                      # <rebuild root>/o2/<name> (the profile's write fence)
    writeguard.assert_writable(d)
    for sub in ("rows", "logs", "out", "uatlas", "bases"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    enc_labels = set(ENC_SHARE.joinpath("labels.txt").read_text().split())
    labels = a.labels or sorted(p.stem for p in Path(S.ITI).glob("*.npz") if p.stem in enc_labels)
    june = [f"{an}_{dd}" for an in ("PS92", "PS93", "PS94", "PS95") for dd in JUNE] if a.june else []
    todo = [(lab, str(d), lab in june) for lab in [*labels, *june] if not (d / "rows" / f"{lab}.npz").exists()]
    got, fail = ak.fan_sessions(todo, _rows_worker, jobs=a.jobs, key=lambda j: j[0])
    for it, r in got:
        print(f"  {it[0]}: {r['matched']} / {r['n']} rows timed")
    for it, err in fail:
        print(f"  FAILED {it[0]}: {err}")
    ok = sorted(p.stem for p in (d / "rows").glob("*.npz"))
    sess = {s["label"]: s for s in config.load_sessions()}
    # staged copies on scratch: <bundle>/uatlas/<label>/U_atlas.npy  (rsync --files-from, relative to the share root)
    rel, ua = [], {}
    share = oi.o2_path(Path(rv.root("microscope")), rv)
    for lab in ok:
        p = Path(sess[lab]["mc"]) / "wfield_local_results" / "allen_aligned_affine8v1" / "U_atlas.npy"
        rel.append(oi.o2_path(p, rv)[len(share) + 1:])
        ua[lab] = rel[-1]
    (d / "uatlas.txt").write_text("".join(f"{r}\n" for r in rel), encoding="utf-8", newline="\n")
    pins = {an: joint_locanmf.pinned(an)["id"] for an in sorted({lab.split("_")[0] for lab in ok})}
    srv = joint_locanmf.server_basis_dir()
    (d / "labels.txt").write_text("".join(f"{x}\n" for x in ok), encoding="utf-8", newline="\n")
    (d / "env.json").write_text(json.dumps(o2_bundle.settings(), indent=1), encoding="utf-8", newline="\n")
    s = oi.scratch(a.user)
    b = f"{s}/o2/{a.name}"
    commit = _git(repo, "rev-parse", "HEAD")
    meta = {"name": a.name, "commit": commit, "code_root": str(repo), "profile": PROFILE, "enc_bundle": ENC,
            "enc_scratch": f"{s}/o2/{ENC}", "drift_min_period_s": ce.DRIFT_MIN_PERIOD_S, "bases": pins,
            "uatlas": ua, "june": june, "n_sessions": len(ok), "created": datetime.datetime.now().isoformat(timespec="seconds")}
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
    (d / "runner.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    (d / "run.sbatch").write_text(f"""#!/bin/bash -l
#SBATCH -J ss_{a.name}
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00
#SBATCH -o {b}/logs/%x_%A_%a.out
#SBATCH -e {b}/logs/%x_%A_%a.err
set -eo pipefail
module load conda/miniforge3/26.9.4-1
conda activate lp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
python {b}/runner.py
""", encoding="utf-8", newline="\n")
    share_b = oi.o2_path(d, rv)
    srv_o2 = oi.o2_path(srv, rv)
    n = len(ok)
    bases = "\n".join(f"rsync -ah --exclude 'C_*' {srv_o2}/{an}/{bid}/ {b}/bases/{an}/{bid}/" for an, bid in pins.items())
    cmds = f"""# O2 state-space joint-LocaNMF extraction: {a.name}

{n} sessions ({len([x for x in ok if x in june])} June contact-only); code {commit}; profile {PROFILE}; imaging from
{ENC}/inputs on scratch (June: this bundle's inputs/). Bases (pinned): {pins}.

## 1. Transfer node  (ssh {a.user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b}
rsync -ah {share_b}/ {b}/
{bases}
rsync -ah --files-from={b}/uatlas.txt {share}/ {b}/uatlas/
ls {s}/o2/{ENC}/inputs | wc -l      # expect 107; if the scratch copy was purged:
# rsync -ah {oi.o2_path(ENC_SHARE, rv)}/inputs/ {s}/o2/{ENC}/inputs/
find {b}/uatlas -name U_atlas.npy | wc -l      # expect {n}
```

## 2. Login node  (ssh {a.user}@o2.hms.harvard.edu) -- ONE task first
```bash
sbatch --array=1 {b}/run.sbatch
squeue -u {a.user}
tail -n 3 {b}/logs/ss_{a.name}_*_1.out       # "<label>: N rows ..., var captured ..."
```
then the rest:
```bash
sbatch --array=2-{n}%40 {b}/run.sbatch
ls {b}/out/sessions | wc -l      # expect {n}
```

## 3. Transfer node, when done
```bash
rsync -ah {b}/out/ {share_b}/out/
```
Desktop reads `{d}/out/sessions/<label>.npz` directly (no collect step).
"""
    (d / "COMMANDS.md").write_text(cmds, encoding="utf-8", newline="\n")
    print(f"{n} sessions bundled -> {d}\n{cmds}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("bundle")
    p.add_argument("--name", required=True)
    p.add_argument("--user", default="ps150")
    p.add_argument("--june", action="store_true")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("labels", nargs="*")
    a = ap.parse_args()
    if a.cmd == "bundle":
        cmd_bundle(a)


if __name__ == "__main__":
    main()
