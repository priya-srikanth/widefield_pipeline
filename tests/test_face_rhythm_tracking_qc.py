"""o2_face_rhythm.tracking_qc: mesh folds are counted from the 25 Hz points output (a point pushed across its neighbour
flips the orientation of the triangles around it)."""
import numpy as np

from scripts import o2_face_rhythm as o2


def test_fold_detected_only_in_the_folded_bin(tmp_path):
    gx, gy = np.meshgrid(np.arange(5) * 12.0, np.arange(5) * 12.0)
    home = np.column_stack([gx.ravel(), gy.ravel()])
    disp = np.zeros((50, len(home), 2), np.float16)
    disp[25, 12, 0] = 30.0                                   # centre point jumps two columns right -> folds
    np.savez(tmp_path / "k_points.npz", disp=disp, viol_frac=np.zeros(50, np.float16), home=home)
    q = o2.tracking_qc(tmp_path / "k_points.npz", step=25)   # samples bins 0 and 25
    assert q["n_sampled"] == 2
    assert q["fold_bins_frac"] == 0.5 and q["fold_frac_mean"] > 0
    assert q["disp_med_px"] == 0.0
    assert q["fold_bins_frac_interior"] == 0.5               # the centre point is interior
