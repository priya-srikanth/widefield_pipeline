"""Aim lick tables FROM A 3-D KINEMATICS TABLE, in the where-decoder / first-lick schema, so the manuscript's aim
producers (`lick_cued_aim_by_spout.py`, `aim_first_vs_all.py`; THE aim definition `iti_recovery_measure.aim_licks`) run
on them unchanged (Priya 2026-11-22: the aim panels need not wait on the encoder -- the angle is purely kinematic).

    python scripts/aim_tables_from_kinematics.py --kin <pose3d>/kinematics_fused_postcue/licks.csv.gz \
        --labels <where_preds table defining the session set + epochs> --out DIR

Per post-cue lick with a 3-D point: act_lat = side_ml (mm, + = right = contralesional; all four lesions are left),
act_sag = hypot(out_ap, down_dv) (the where-decoder's SAGITTAL extension, `iti_where_decoder.coords`), so
atan2(act_lat, act_sag) is the same angle the manuscript used. pos = lesion-relative spout (`orofacial_cohort.pos_key`,
lesion side L). lick_idx = the kinematics table's post-cue lick index (0 = the trial's first post-cue lick).

DIFFERENCES from the where-decoder path (documented, Priya 11-22):
  * ALL post-cue licks of the kinematics table with a 3-D point, not the where-decoder's random draw of at most
    `iti_licks.N_TRIAL_MAX` (1,500) post-cue licks per session (seed 0);
  * licks need not fall inside the imaging recording window (the where table only keeps licks with imaging);
  * the four-camera table carries 3-D only for licks seen by >= 2 cameras (`kin_ok`); cam1-only events are not licks.
The session set and each session's epoch come from ``--labels`` (the where-decoder table of a run), so the sessions,
epochs and the pre reference are the manuscript's.

Writes DIR/aim_preds_kinematics.csv (set, label, animal, epoch, pos, act_lat, act_sag, lick_idx) and
DIR/aim_first_preds_kinematics.csv.gz (label, animal, epoch, pos, lick_idx, act = act_lat) -- the two inputs of
`aim_licks(preds, first_preds)`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build(kin: Path, labels: Path) -> pd.DataFrame:
    from wfield_local import orofacial_cohort as oc
    W = pd.read_csv(labels, usecols=["set", "label", "epoch"])
    ep = W[W.set == "trial"].drop_duplicates("label").set_index("label").epoch
    K = pd.read_csv(kin, usecols=lambda c: c in ("session", "animal", "trial_id", "lick_idx", "position", "context",
                                                 "side_ml", "out_ap", "down_dv"))
    if "context" in K:
        K = K[K.context == "post_cue"]
    K = K.dropna(subset=["side_ml", "out_ap", "down_dv", "lick_idx"])
    K["label"] = K.session.str.split("_").str[0] + "_" + K.session.str.split("_").str[1].str[-4:]
    K = K[K.label.isin(ep.index)].copy()
    P = pd.DataFrame({"set": "trial", "label": K.label.to_numpy(), "animal": K.animal.to_numpy(),
                      "epoch": ep.reindex(K.label).to_numpy(),
                      "pos": [oc.pos_key(p, "L") for p in K.position.astype(str)],
                      "act_lat": K.side_ml.to_numpy(float),
                      "act_sag": np.hypot(K.out_ap.to_numpy(float), K.down_dv.to_numpy(float)),
                      "lick_idx": K.lick_idx.to_numpy(int)})
    missing = sorted(set(ep.index) - set(P.label))
    if missing:
        print(f"{len(missing)} sessions of --labels have no kinematics licks: {missing}")
    return P


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kin", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    f1, f2 = a.out / "aim_preds_kinematics.csv", a.out / "aim_first_preds_kinematics.csv.gz"
    for f in (f1, f2):
        if f.exists() and not a.force:
            sys.exit(f"refusing to overwrite {f} (--force)")
    P = build(a.kin, a.labels)
    print(f"{len(P)} post-cue licks with 3-D, {P.label.nunique()} sessions; first licks {(P.lick_idx == 0).sum()}")
    print(P.groupby("epoch").label.nunique().to_dict())
    a.out.mkdir(parents=True, exist_ok=True)
    P.to_csv(f1, index=False)
    P.assign(act=P.act_lat)[["label", "animal", "epoch", "pos", "lick_idx", "act"]].to_csv(f2, index=False)
    print("wrote", f1, f2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
