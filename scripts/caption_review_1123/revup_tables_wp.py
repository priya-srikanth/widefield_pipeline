"""Rev-up tables on the adopted jaw rule and the camera working period (CC-B 11-23 late, inconsistency fixes task B).

The rev-up neural follow-ups (`revup_followup.py`, `revup_neural.py --join`) read `<REV>/revup_trials.csv.gz`. The live
one (`imaging_encoding/revup_fourcam_cam3on/`) gates with the near-spout reference `canonical` and takes the first
movement from the 2-D per-camera jaw + any cam1 event + FaceRhythm on the own-session scale. Priya 11-23 (decisions 2, 3):
gate = the camera working period; first movement = the 3-D jaw (cap + 3-D re-onset) OR an in-mouth lick (`3d_mouth`);
FaceRhythm vigor is NOT part of the first movement.

What this does (no per-session recompute; the per-session pickles never read the attempts table, `revup_behaviour`):
1. `revup_behaviour.assemble` on the live per-session pickles with the attempts table = the manuscript camera-tagged
   table on the new rule (`qc/jaw3dmouth_ccB1123/sel-..._daqlk/trial_attempts.csv`): canonical (= working period),
   cls, attempted, and the pre-stroke p90 `late` flag follow it.
2. First movement: the S10A overnight m3d onset (`s10_jaw3dmouth_wp.add_jaw` on the full cap jaw table and the cached
   in-mouth licks) replaces jaw / cam1 / FR: move_ms = wait_ms = the m3d onset at or before the first lick (else the
   latency when the trial was measured), revup_ms = latency - move_ms, revup_class as `revup_behaviour._revup`
   (aborted = >= 2 cam1-only openings before the lick; ramp = rev-up >= RAMP_MS; unknown = not measured). The old
   columns are kept in revup_trials_old_movement.csv.gz.
3. `revup_behaviour.summarise` + plots into the NEW folder `imaging_encoding/revup_fourcam_jaw3dmouth_wp_ccB1123/`.

    python scripts/caption_review_1123/revup_tables_wp.py [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding")
SRC = ENC / "revup_fourcam_cam3on"
OUT = ENC / "revup_fourcam_jaw3dmouth_wp_ccB1123"
ATT = (ENC / "qc" / "jaw3dmouth_ccB1123" / "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
       / "trial_attempts.csv")
HERE = Path(__file__).resolve().parent
OLD_MOVE = ["jaw_ms", "jaw_judged", "move_ms", "wait_ms", "revup_ms", "revup_class", "canonical", "cls", "late"]


def _mod(name: str, f: Path):
    spec = importlib.util.spec_from_file_location(name, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="build the table and print the checks; write nothing")
    a = ap.parse_args(argv)
    os.environ["REVUP_OUT_DIR"] = str(OUT)             # before revup_behaviour is imported (REV is set at import)
    sys.path.insert(0, str(HERE.parents[0]))
    import revup_behaviour as rb
    from wfield_local import writeguard
    S = _mod("s10wp", HERE / "s10_jaw3dmouth_wp.py")
    src_sess = S.A.REVUP_SESS                          # the live per-session pickles (read only)
    rb.sess_dir = lambda: src_sess
    rb.attempts_csv = lambda: ATT
    live = pd.read_csv(SRC / "revup_trials.csv.gz", low_memory=False)
    labels = sorted(live.label.unique())
    T, F = rb.assemble(labels)
    assert len(T) == len(live) and (T.label.to_numpy() == live.label.to_numpy()).all() \
        and (T.trial_id.to_numpy() == live.trial_id.to_numpy()).all(), "row order differs from the live table"
    Z = np.load(SRC / "revup_fr.npz")
    assert (Z["label"] == T.label.to_numpy(str)).all() and (Z["trial_id"] == T.trial_id.to_numpy(int)).all()
    # m3d first movement (S10A overnight: same per-trial table, same rows)
    X = S.trials()
    S.add_jaw(X, S.CAP_FULL)
    X = X.set_index(["label", "trial_id"]).reindex(pd.MultiIndex.from_frame(T[["label", "trial_id"]]))
    assert X.lat_ms.notna().sum() == T.lat_ms.notna().sum(), "first-lick latencies differ from the live table"
    old = T[["label", "trial_id"]].join(live[OLD_MOVE].add_suffix("_old"))
    T["m3d_ms"] = X.m3d_ms.to_numpy(float)
    T["m3d_judged"] = X.m3_judged.fillna(False).to_numpy(bool)
    lat = T.lat_ms.to_numpy(float)
    cand = np.where(T.m3d_ms.to_numpy(float) <= lat + 1e-6, T.m3d_ms.to_numpy(float), np.nan)
    measured = T.m3d_judged.to_numpy(bool)
    mv = np.where(np.isfinite(cand), cand, np.where(np.isfinite(lat) & measured, lat, np.nan))
    T["move_ms"] = mv
    T["wait_ms"] = mv
    T["revup_ms"] = lat - mv
    cls = np.where(T.n_cam1_pre.fillna(0) >= 2, "aborted", np.where(T.revup_ms >= rb.RAMP_MS, "ramp", "wait"))
    T["revup_class"] = np.where(np.isfinite(lat), np.where(measured, cls, "unknown"), "")
    T["jaw_ms"], T["jaw_judged"] = T.m3d_ms, T.m3d_judged          # the jaw column = the adopted rule from here on
    fc = T[(T.pos == "far_contra") & T.canonical & np.isfinite(T.lat_ms)]
    chk = fc.groupby(["animal", "epoch"])[["wait_ms", "revup_ms"]].median().unstack("epoch").round(0)
    print("== gate: canonical (working period) n =", int(T.canonical.sum()), "| live reference n =",
          int(live.canonical.astype(str).str.lower().eq("true").sum()))
    print("== far contra, per mouse median wait / rev-up (ms), working period, m3d:\n" + chk.to_string())
    if a.check:
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    writeguard.assert_writable(OUT / "revup_trials.csv.gz")
    old.to_csv(OUT / "revup_trials_old_movement.csv.gz", index=False, float_format="%.5g")
    rb.summarise(T, F)
    rb.plots()
    (OUT / "PROVENANCE.txt").write_text(
        "Rev-up tables on the adopted jaw rule (3d_mouth: cap 3-D jaw + 3-D re-onset OR an in-mouth lick; FaceRhythm "
        "not in the first movement) and the camera working period (canonical of the manuscript camera-tagged "
        f"attempts table {ATT}). Per-session pickles reused from {src_sess}. Script "
        "scripts/caption_review_1123/revup_tables_wp.py (branch ccb/revup-wp-1123). CC-B 11-23 late.\n")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
