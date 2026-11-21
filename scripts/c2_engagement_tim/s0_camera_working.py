"""Step 0: per-trial camera working period + contact (the manuscript hit-rate trial set, Fig 3C / epoch_1b:
`trial_classes.camera_period_trials`) for every analysis session, and a check that it reproduces the manuscript's
per-session hit rates (epoch_1b_behaviour_by_position_sessions.csv; compared per animal x epoch x position mean).
Usage: WF_REPO=... python s0_camera_working.py <out_dir>   -> <out_dir>/camera_working.csv
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline")))
import pandas as pd  # noqa: E402
from wfield_local import behavior_states as B, trial_classes as tc  # noqa: E402

out = Path(sys.argv[1]) / "camera_working.csv"
if out.exists():
    sys.exit(f"refusing to overwrite {out}")
T = B.load_collected()["trials"]
T = T[T.label.map(B.session_epoch).notna()].copy()
T["epoch"] = T.label.map(B.session_epoch)
rows = []
for lab in sorted(T.label.unique()):
    S = tc.session_trials(lab).dropna(subset=["cue_daq"])
    w, c, _h = tc.camera_period_trials(lab[:5] + lab[-4:], S.cue_daq.to_numpy(float))
    rows.append(pd.DataFrame({"label": lab, "trial_id": S.trial_id.astype(int).to_numpy(), "work": w, "contact": c}))
R = pd.concat(rows)
R.to_csv(out, index=False)
T = T.merge(R, on=["label", "trial_id"])
T = T[T.work]
POSN = {"close_L": "nI", "close_center": "nM", "close_R": "nC", "far_L": "fI", "far_center": "fM", "far_R": "fC"}
T["position"] = T.position.map(POSN)
mine = T.groupby(["label", "animal", "epoch", "position"]).contact.mean().groupby(["animal", "epoch", "position"]).mean()
ms = pd.read_csv(r"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/hv-meegkit_hpfit_lag_refitT_swapmask_"
                 r"lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/epoch/data/"
                 r"epoch_1b_behaviour_by_position_sessions.csv")
ref = ms.groupby(["animal", "epoch", "position"]).value.mean()
print("max |animal x epoch x position mean hit rate - manuscript| =", float((mine - ref).abs().max()))
