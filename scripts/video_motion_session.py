"""Video motion-energy components for one session and camera (`wfield_local/video_motion.py`), on the session's
imaging-frame clock, for the movement encoding models (`scripts.movement_encoding_session --video ...`).

    python -m scripts.video_motion_session PS93:20260814 --cam cam1      # one camera per process (run 4 at once)

Output: session_poses/<animal>_<date>_full/video_motion_<cam>.npz (timecourses (n_imaging_frames, k), components,
mean image, explained variance, the imaging frame times) + video_motion_<cam>.png (the top components as images,
so one can see WHAT each regressor is: paw, whisker pad, jaw, ...). The binned motion-energy memmap is local
scratch (`~/lp_stage/video_me/`) and removed after the projection unless --keep.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import scripts.movement_encoding_session as MS
    from scripts.session_poses import session_dir
    from wfield_local import movement_inputs as MI
    from wfield_local import video_motion as VM
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="animal:date")
    ap.add_argument("--cam", required=True, choices=["cam1", "cam2", "cam3", "cam4"])
    ap.add_argument("--ds", type=int, default=8, help="spatial downsampling factor (area average)")
    ap.add_argument("--k", type=int, default=50, help="components kept")
    ap.add_argument("--keep", action="store_true", help="keep the binned motion-energy memmap")
    a = ap.parse_args(argv)
    rv = PathResolver()
    animal, date = a.session.split(":")[:2]
    _, _, ft = MS.imaging(f"{animal}_{date[4:]}")
    tpl = dict(np.load(Path(rv.root("alignment_templates")) / a.cam / animal / f"{date}.npz", allow_pickle=True))
    vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{a.cam}_*.avi"))[0]
    n_cam = int(tpl["n_cam_frames"])
    bins = VM.frame_bins(MI.cam_frames_to_daq_s(tpl, np.arange(n_cam)), ft)
    scratch = Path.home() / "lp_stage" / "video_me"
    scratch.mkdir(parents=True, exist_ok=True)
    mpath = scratch / f"{animal}_{date}_{a.cam}_me.npy"
    t0 = time.time()
    print(f"{a.cam}: {vid.name}, {n_cam} frames -> {len(ft)} imaging bins ({(bins >= 0).mean():.1%} of frames "
          f"inside imaging)", flush=True)
    mm, filled = VM.binned_motion_energy(VM.video_frames(vid), bins, len(ft), mpath, ds=a.ds)
    print(f"{a.cam}: motion energy in {(time.time() - t0) / 60:.1f} min; {filled.mean():.1%} of imaging frames "
          f"covered", flush=True)
    res = VM.motion_svd(mm, filled, k=a.k)
    shape = VM.downsample(next(VM.video_frames(vid)), a.ds).shape
    out = session_dir(rv, f"{animal}:{date}:full")[2]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"video_motion_{a.cam}.npz", frame_times_s=ft, filled=filled, image_shape=shape,
                        ds=a.ds, video=vid.name, **res)
    print(f"{a.cam}: top-{a.k} explained {res['explained'].sum():.1%} (first 5: "
          f"{np.round(res['explained'][:5], 3).tolist()}) -> {out / f'video_motion_{a.cam}.npz'}", flush=True)
    n_show = min(12, a.k)
    fig, axs = plt.subplots(2, n_show // 2 + 1, figsize=(2.2 * (n_show // 2 + 1), 4.6), squeeze=False)
    axs = axs.ravel()
    axs[0].imshow(res["mean"].reshape(shape), cmap="gray")
    axs[0].set_title("mean motion energy", fontsize=7)
    for j in range(n_show):
        c = res["components"][j].reshape(shape)
        lim = np.abs(c).max()
        axs[j + 1].imshow(c, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axs[j + 1].set_title(f"PC{j} ({res['explained'][j]:.1%})", fontsize=7)
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"{animal} {date} {a.cam}: motion-energy components", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"video_motion_{a.cam}.png", dpi=110)
    plt.close(fig)
    del mm
    if not a.keep:
        mpath.unlink()
    print(f"{a.cam}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
