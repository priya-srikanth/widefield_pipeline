"""IMAGING BOX: are any pixels clipped at the camera ceiling (65535)? READ-ONLY on the raw .dat.

Why (QC doc §0m): PS121 10-29's mean 470 image reaches 63,895 / 65,535, and an SVD reconstruction of its brightest 1%
of pixels exceeds 65,535 on 14% of sampled frames -- the raw data must clip there, which flattens exactly the hotspot
responses. This reads every STEP-th clean pair (DAQ frame map, so channel identity is right) and reports, per
channel: the fraction of frames with any clipped pixel, the fraction of brain pixels ever clipped, and a map of how
often each pixel clips.

DO NOT RUN WHILE ACQUIRING (reads the .dat from E:).

    conda activate wfield
    python scripts/cohort2_qc/saturation_check.py E:/labcams_data/20261029/PS121_20261029_160848 [more] [--step 50]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

CEIL = 65500          # within 35 counts of 65535 counts as clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from wfield_local.writeguard import assert_writable
    allres = {}
    for sess in a.sessions:
        out = a.out or sess.parent / "_saturation_check"
        assert_writable(out)
        out.mkdir(parents=True, exist_ok=True)
        dat = next((sess / "raw_widefield_data").glob("*_uint16.dat"))
        m = re.search(r"_(\d+)_(\d+)_(\d+)_uint16", dat.name)
        H, W = int(m[2]), int(m[3])
        n = dat.stat().st_size // (H * W * 2)
        raw = np.memmap(dat, dtype=np.uint16, mode="r", shape=(n, H, W))           # READ-ONLY
        fm = np.load(next((sess / "motion_corrected").glob("*cleanpairs_frame_map.npz")))
        res = {}
        for slot, nm in ((0, "415"), (1, "470")):
            idx = fm[f"original_frame_index_ch{slot}"][::a.step]
            count = np.zeros((H, W), np.int32)
            any_clip = 0
            mx = 0
            for k in idx:
                f = np.asarray(raw[int(k)])
                c = f >= CEIL
                count += c
                any_clip += bool(c.any())
                mx = max(mx, int(f.max()))
            res[nm] = dict(frames_sampled=int(len(idx)), frac_frames_with_clipping=any_clip / len(idx),
                           frac_pixels_ever_clipped=float((count > 0).mean()),
                           frac_pixels_clipped_over_10pct_of_frames=float((count > 0.1 * len(idx)).mean()),
                           max_count=mx)
            np.save(out / f"{sess.name}_{nm}_clip_count.npy", count)
            print(sess.name, nm, json.dumps(res[nm]), flush=True)
        allres[sess.name] = res
    (Path(a.out or a.sessions[0].parent / "_saturation_check") / "saturation_summary.json").write_text(
        json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
