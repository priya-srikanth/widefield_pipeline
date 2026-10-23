"""Is spout CONTACT predictable from lick DIRECTION? (Priya 10-23, cheap check of why PS93's unique contact R2 is ~0
although PS93 misses most: if contact is predictable from the lick's direction error, contact and the within-spout
direction group share variance and unique R2 credits neither.)

Per animal, August pre post-cue licks with 3-D (pose window per-lick table): cross-validated AUC of contact vs no
contact from direction-only features (|midline|, |frontal|, |sagittal| angular errors to the aimed spout, signed
midline error), and, for comparison, reach alone and direction + reach. AUC 0.5 = contact unrelated to direction;
higher = more shared. Reported with the contact fraction. Five-fold CV, folds by session (no leakage within a session).

Usage: python scripts/contact_direction_predictability.py
Output: <E>/qc/contact_direction_predictability.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KIN = "M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/kinematics/licks.csv.gz"
DIR = ["abs_midline", "abs_frontal", "abs_sagittal", "midline_err_ipsi"]
SETS = {"direction": DIR, "reach": ["reach_mm"], "direction+reach": DIR + ["reach_mm"]}


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = np.full(len(y), np.nan)
    k = min(5, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    ok = np.isfinite(p)
    return float(roc_auc_score(y[ok], p[ok])) if len(np.unique(y[ok])) == 2 else np.nan


def main() -> int:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    K = pd.read_csv(KIN, usecols=["animal", "date", "epoch", "session", "contact", "reach_mm", "midline_err_ipsi",
                                  "frontal_err_ipsi", "sagittal_err"])
    K = K[(K.epoch == "pre") & (K.date.astype(str) >= "20260806")].copy()
    K["abs_midline"], K["abs_frontal"], K["abs_sagittal"] = (K.midline_err_ipsi.abs(), K.frontal_err_ipsi.abs(),
                                                             K.sagittal_err.abs())
    K["y"] = K.contact.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
    rows = []
    for an, A in K.groupby("animal"):
        A = A.dropna(subset=DIR + ["reach_mm"])
        row = {"animal": an, "n_licks": len(A), "n_sessions": A.session.nunique(), "contact_frac": A.y.mean()}
        for name, cols in SETS.items():
            row[f"auc_{name}"] = cv_auc(A[cols].to_numpy(float), A.y.to_numpy(), A.session.to_numpy())
        rows.append(row)
    T = pd.DataFrame(rows)
    dst = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc" / "contact_direction_predictability.csv"
    assert_writable(dst)
    T.to_csv(dst, index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
