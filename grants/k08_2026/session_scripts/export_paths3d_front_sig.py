"""Re-export `paths3d_front_sig.csv`, the marker table read by `paths_panel.py` / `paths_narrow.py` /
`paths_short.py` (Fig 3O/3P). No saved script wrote the original: it was a one-off export of
`kinematics_pooled_figs._paths3d_sig`, which the pipeline computes for the figure but never writes to CSV.
This reproduces it from the same inputs, with the same gate and the same test (cluster permutation on each
session's change from its animal's own August pre, labels permuted within animal, p < .05/3).

    PYTHONPATH=. python scripts/../grants/k08_2026/session_scripts/export_paths3d_front_sig.py
"""
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import kinematics_pooled_figs as K   # noqa: E402

C = pd.read_csv(K._tc().kin_out_dir(K.pose3d_dir(), "phase") / "curves.csv.gz")
# the gate fig_paths3d applies before calling _paths3d_sig -- read it off fig_paths3d, do not guess
aug = K.sessions3d()
aug = aug[K._is_aug_pre(aug)]
sig = K._paths3d_sig(C, aug)

rows = []
for (pos, ep, meas), s in sig.items():
    rows.append(pd.DataFrame({"position": pos, "epoch": ep, "measure": meas,
                              "x": s.index.astype(float), "sig": s.to_numpy()}))
out = pd.concat(rows, ignore_index=True)
out.to_csv("paths3d_front_sig.csv", index=False)
print("wrote paths3d_front_sig.csv", out.shape)
