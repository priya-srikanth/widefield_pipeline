"""Video MOTION-ENERGY components on HMS O2, for the encoding model's uninstructed group (Priya 2026-10-18: "sure can
prep a pilot o2 ME").

    python scripts/o2_video_me.py bundle --name me_pilot_20261018 --user ps150 [--cams cam1 cam4] [SESSIONS ...]
    python scripts/o2_video_me.py collect --name me_pilot_20261018

bundle (this desktop): per session x camera, every video frame's IMAGING-frame bin on the DAQ clock (camera alignment
template -> `video_motion.frame_bins`) is computed HERE, where the imaging frame clock and templates live, and saved as
a small int32 array; plus a copy of `wfield_local/video_motion.py` (numpy + OpenCV only), the runner, an sbatch array
(CPU, one task per session x camera) and COMMANDS.md -> <DeepLabCut/Widefield>/o2/<name>/ on the share. Default
sessions = the PILOT: per animal 2 August pre, 2 acute, 2 chronic (earliest / latest), cam1 + cam4 (the faces).
collect (this desktop): out/<animal>_<date>_<cam>.npz -> session_poses/<animal>_<date>_full/video_motion_<cam>.npz,
the layout `scripts.video_motion_session` writes.

Masks (configs/defaults.yaml video_motion, Priya 10-06): none in space -- the treadmill reflects paw motion, and the
spout is handled in TIME (regressors blanked outside strobe .. trial end in the model). In the encoding model, video
goes in LAST, orthogonalised against every other regressor (Musall 2019: "cannot enhance the explanatory power of the
video variables -- it can only reduce it"), so the spout's static-occlusion position leak (DECISIONS 10-07: ENL motion
energy decodes position 0.92-0.96) cannot take position variance.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RUNNER = r'''"""Runs on O2: binned motion energy + motion SVD for one session x camera (task = SLURM_ARRAY_TASK_ID)."""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import video_motion as VM

b = Path(__file__).parent
task = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
T = pd.read_csv(b / "tasks.tsv", sep="\t", dtype=str)
r = T.iloc[task - 1]
out = b / "out"; out.mkdir(exist_ok=True)
dst = out / f"{r.animal}_{r.date}_{r.cam}.npz"
if dst.exists():
    print("done already", dst); sys.exit(0)
vid = Path(r.video_o2)
bins = np.load(b / "bins" / f"{r.animal}_{r.date}_{r.cam}.npy")
n_bins = int(r.n_bins)
scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"me_{r.animal}_{r.date}_{r.cam}.npy"
t0 = time.time()
nw = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
mm, filled = VM.binned_motion_energy_parallel(vid, bins, n_bins, scratch, ds=int(r.ds), n_workers=nw)
print(f"motion energy {(time.time() - t0) / 60:.1f} min, {filled.mean():.1%} of imaging frames covered", flush=True)
res = VM.motion_svd(mm, filled, k=int(r.k), keep=None)
shape = VM.downsample(next(VM.video_frames(vid)), int(r.ds)).shape
np.savez_compressed(dst, filled=filled, image_shape=shape, ds=int(r.ds), video=vid.name, **res)
del mm
scratch.unlink(missing_ok=True)
print(f"saved {dst} in {(time.time() - t0) / 60:.1f} min; top-{r.k} explained {res['explained'].sum():.1%}", flush=True)
'''


def _frame_times(label: str) -> np.ndarray:
    """Imaging frame DAQ seconds, exactly as `combined_encoding.load_imaging` (no SVD loaded; length = the SVD's)."""
    import glob

    from wfield_local import config
    from wfield_local.hemo_variants import ADOPTED
    from wfield_local.locanmf_lick_aligned import _corrected_frame_samples
    from wfield_local.plot_spout_trial_averages import _load_daq_events
    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    nV = np.load(res / f"hemo_{ADOPTED}" / "SVTcorr.npy", mmap_mode="r").shape[1]
    cue = _load_daq_events(s["h5"])
    fmdir = s["fmdir"] or s["mc"]
    fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
    off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
    ft = np.asarray(_corrected_frame_samples(fm, cue["pco_samples"], int(off)), float) / float(cue["sample_rate_hz"])
    return ft[:min(len(ft), nV)]


def pilot_sessions() -> list[tuple[str, str]]:
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs
    rows = [(config.animal_of(s["label"]), "2026" + s["label"].split("_")[-1], epochs.epoch_of(s["label"]))
            for s in ak.curated_sessions(order="sorted")]
    D = pd.DataFrame(rows, columns=["animal", "date", "epoch"]).sort_values(["animal", "date"])
    pick = []
    for a, g in D.groupby("animal"):
        pre = g[(g.epoch == "pre") & (g.date.str[4:] >= "0801")].date.tolist()
        acu = g[g.epoch == "acute"].date.tolist()
        chr_ = g[g.epoch == "chronic"].date.tolist()
        for d in dict.fromkeys(pre[:1] + pre[-1:] + acu[:2] + chr_[:1] + chr_[-1:]):
            pick.append((a, d))
    return pick


def cmd_bundle(name: str, user: str, sessions: list[tuple[str, str]], cams: list[str], ds: int, k: int) -> Path:
    from wfield_local import movement_inputs as MI
    from wfield_local import o2_inference as oi
    from wfield_local import video_motion as VM
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    d = oi.bundle_dir(rv, name)
    writeguard.assert_writable(d.parent)
    (d / "bins").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    rows = []
    for animal, date in sessions:
        label = f"{animal}_{date[4:]}"
        try:
            ft = _frame_times(label)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {label}: no imaging frame clock ({type(ex).__name__}: {ex}) -- skipped")
            continue
        for cam in cams:
            tp = Path(rv.root("alignment_templates")) / cam / animal / f"{date}.npz"
            vids = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{cam}_*.avi"))
            if not tp.exists() or len(vids) != 1:
                print(f"  !! {label} {cam}: template {tp.exists()}, videos {len(vids)} -- skipped")
                continue
            tpl = dict(np.load(tp, allow_pickle=True))
            n_cam = int(tpl["n_cam_frames"])
            bins = VM.frame_bins(MI.cam_frames_to_daq_s(tpl, np.arange(n_cam)), ft).astype(np.int32)
            np.save(d / "bins" / f"{animal}_{date}_{cam}.npy", bins)
            v_o2 = oi.o2_path(vids[0], rv)
            rows.append({"animal": animal, "date": date, "cam": cam, "video": vids[0].name, "n_bins": len(ft),
                         "ds": ds, "k": k, "video_share": v_o2,
                         "video_o2": f"{oi.scratch(user)}/o2_videos/{date}/{animal}/{vids[0].name}"})
            print(f"  {label} {cam}: {n_cam} frames -> {len(ft)} imaging bins ({(bins >= 0).mean():.1%} inside)")
    T = pd.DataFrame(rows)
    T.to_csv(d / "tasks.tsv", sep="\t", index=False, lineterminator="\n")
    shutil.copy2(Path(__file__).resolve().parents[1] / "wfield_local" / "video_motion.py", d / "video_motion.py")
    (d / "me_runner.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    rel = sorted({"/".join(Path(v).parts[-3:]) for v in T.video_share})
    (d / "videos.txt").write_text("".join(f"{r}\n" for r in rel), encoding="utf-8", newline="\n")
    s = oi.scratch(user)
    b = f"{s}/o2/{name}"
    share_b = oi.o2_path(d, rv)
    cam_root = oi.o2_path(Path(rv.root("behavior_cameras")), rv)
    sb = f"""#!/bin/bash
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 0-08:00:00
#SBATCH --array=1-{len(T)}%24
#SBATCH -o {b}/logs/%A_%a.out
#SBATCH -e {b}/logs/%A_%a.err
module load conda/miniforge3/24.11.3-0
source activate lp
python {b}/me_runner.py
"""
    (d / "run_me.sbatch").write_text(sb, encoding="utf-8", newline="\n")
    cmds = f"""# O2 video motion-energy pilot: {name} ({len(T)} tasks = session x camera)

## 1. Transfer node  (ssh {user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b} {s}/o2_videos
rsync -ah {share_b}/ {b}/
rsync -ah --files-from={b}/videos.txt {cam_root}/ {s}/o2_videos/
ls {b}; wc -l {b}/videos.txt
```
(cam4 videos already on scratch from the LP runs are skipped by rsync; cam1 is copied.)

## 2. Login node  (ssh {user}@o2.hms.harvard.edu) -- one task first, then the rest
```bash
sbatch --array=1 {b}/run_me.sbatch
squeue -u {user}
tail -n 5 {b}/logs/*_1.out
```
When task 1 has written `{b}/out/*.npz` (expect ~10-40 min):
```bash
sbatch --array=2-{len(T)}%24 {b}/run_me.sbatch
```

## 3. Transfer node, when all are done
```bash
ls {b}/out | wc -l          # expect {len(T)}
rsync -ah {b}/out/ {share_b}/out/
```

## 4. Desktop
`python scripts/o2_video_me.py collect --name {name}`
"""
    (d / "COMMANDS.md").write_text(cmds, encoding="utf-8", newline="\n")
    print(f"{len(T)} tasks -> {d}\n{cmds}")
    return d


def cmd_collect(name: str) -> None:
    from scripts.session_poses import session_dir
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    d = oi.bundle_dir(rv, name)
    T = pd.read_csv(d / "tasks.tsv", sep="\t", dtype=str)
    n = 0
    for r in T.itertuples():
        f = d / "out" / f"{r.animal}_{r.date}_{r.cam}.npz"
        if not f.exists():
            print(f"  missing {f.name}")
            continue
        z = dict(np.load(f, allow_pickle=True))
        z["frame_times_s"] = _frame_times(f"{r.animal}_{r.date[4:]}")
        out = session_dir(rv, f"{r.animal}:{r.date}:full")[2]
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"video_motion_{r.cam}.npz", **z)
        n += 1
    print(f"collected {n} of {len(T)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--name", required=True)
    b.add_argument("--user", required=True)
    b.add_argument("--cams", nargs="+", default=["cam1", "cam4"])
    b.add_argument("--ds", type=int, default=8)
    b.add_argument("--k", type=int, default=200, help="components (Musall: 200 per camera)")
    b.add_argument("sessions", nargs="*", help="animal:YYYYMMDD (default: the pilot set)")
    c = sub.add_parser("collect")
    c.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        ss = [tuple(x.split(":")[:2]) for x in a.sessions] if a.sessions else pilot_sessions()
        cmd_bundle(a.name, a.user, ss, a.cams, a.ds, a.k)
    else:
        cmd_collect(a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
