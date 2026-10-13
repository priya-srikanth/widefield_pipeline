"""The side-camera assignment: each moment to ONE of cam2 / cam3, alternating per session x position, every 5th to
both, context never a target."""
from __future__ import annotations

import pandas as pd

from wfield_local import dlc_sidecam as SC


def _manifest(n_moments=10, positions=("close_L", "far_R")):
    rows = []
    for j, pos in enumerate(positions):
        for k0 in range(n_moments):
            k = 100 * j + k0                      # trial ids are unique across positions, as in real data
            for cam in ("cam2", "cam3", "cam4"):
                rows.append({"animal": "PS93", "date": "20260814", "cam": cam, "epoch": "pre",
                             "video_stem": f"{cam}_2026-08-14T17_54_22", "image": f"img{1000 + k:07d}.png",
                             "frame": str(1000 + k), "trial_id": str(k), "position": pos, "category": "success",
                             "phase": "early", "t_from_cue_s": "0.4"})
            rows.append({**rows[-2], "category": "context", "image": f"img{5000 + k:07d}.png", "phase": "ctx_on-1"})
    return pd.DataFrame(rows)


def test_each_moment_one_camera_except_every_fifth():
    a = SC.assign(_manifest())
    lab = a[a.label]
    per_moment = lab.groupby("moment_id").cam.nunique()
    assert set(per_moment.unique()) <= {1, 2}
    assert (per_moment == 2).sum() == 4            # 10 moments x 2 positions -> every 5th = 2 per position
    assert lab.category.ne("context").all()        # context is never a target
    assert set(a.cam) == {"cam2", "cam3"}          # cam4 rows are not side frames


def test_both_cameras_get_every_position():
    lab = SC.assign(_manifest())
    lab = lab[lab.label]
    assert lab.groupby("position").cam.nunique().eq(2).all()
    n = lab.groupby("cam").size()
    assert abs(n["cam2"] - n["cam3"]) <= 2
