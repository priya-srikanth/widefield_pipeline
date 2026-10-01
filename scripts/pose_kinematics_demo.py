"""First run of the ported kinematics on the PS93 0908 cue clip, DLC round 3 vs LP: clean -> spout frame ->
tongue v7 kinematics -> jaw per-trial -> tongue-jaw mismatch. A smoke test of the chain, NOT a result: every px
threshold is still the old rig's, and the clip holds 24 trials (4 per position).

    python -m scripts.pose_kinematics_demo --lp <LP csv for the clip>

The clip is 24 cue windows (cue -0.5 .. +3.5 s). Windows are joined with 2,200 NaN frames (8.8 s): the tongue
module slices each trial to +8 s and would otherwise overwrite the neighbouring window in its session arrays.
The spout frame comes from DLC's spout in these windows (all inside position strobe .. trial end).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wfield_local import jaw_kinematics as jk
from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import tongue_jaw_mismatch as tm
from wfield_local import tongue_kinematics as tk

SEP = 2200


def run(pose_csv: Path, idx: pd.DataFrame, frame: SF.SpoutFrame | None):
    df = oc.read_pose(pose_csv).iloc[:len(idx)]
    clean = {bp: oc.clean_windows(df, idx, bp, sep=SEP) for bp in ("tongue", "jaw")}
    cues = clean["tongue"][2]
    pos = idx.groupby("trial_k", sort=True).position.first().to_numpy()
    tid = idx.groupby("trial_k", sort=True).trial_id.first().to_numpy()
    trials_df = pd.DataFrame({"trial_id": tid, "cue_frame": cues.astype(float), "position": pos})
    trials = [tk.Trial(int(t), float(c), str(p)) for t, c, p in zip(tid, cues, pos)]
    sfr = tk.SpoutFrame(tuple(frame.origin), tuple(frame.ap_axis)) if frame is not None else None
    T = tk.from_clean(clean["tongue"][0], trials, spout_frame=sfr)
    J, _ = jk.jaw_pertrial(clean["jaw"][0].y_final, clean["jaw"][0].fps, trials_df)
    M = tm.classify_from_clean(clean["jaw"][0], clean["tongue"][0], trials_df)
    return T, J, M, clean


def main(argv=None) -> int:
    from wfield_local import dlc_project
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lp", type=Path, required=True)
    a = ap.parse_args(argv)
    rv = PathResolver()
    d = dlc_project.project_dir(rv).parent / "lp_vs_dlc_cue_traces" / "PS93_20260908"
    idx = pd.read_csv(d / "cue_windows_index.csv")
    # spout frame from DLC's spout over the clip (one span per window)
    dlc = oc.read_pose(d / "cue_windows_DLC.csv")
    xyp = dlc[["spout_x", "spout_y", "spout_likelihood"]].to_numpy()
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(xyp, spans))
    print(f"spout frame: origin {frame.origin.round(1)}  ap {frame.ap_axis.round(3)}  rms {frame.rms_px:.2f} px  "
          f"mouse-left sign {frame.mouse_left_sign:+d}")
    rows = []
    for name, f in (("DLC", d / "cue_windows_DLC.csv"), ("LP", a.lp)):
        T, J, M, _c = run(f, idx, frame)
        pt = T.per_trial.merge(J[["trial_id", "jaw_pass_qc", "jaw_peak_selected_deflection"]], on="trial_id")
        mm = M.trials
        pt = pt.merge(mm[["trial_id", "candidate_no_lick_with_jaw_move"]], on="trial_id", how="left")
        pt["model"] = name
        rows.append(pt)
        pl = T.per_lick
        print(f"\n== {name}: {len(pt)} trials, {len(pl)} kept licks")
    out = pd.concat(rows, ignore_index=True)
    order = {p: i for i, p in enumerate(tk.POSITIONS)}
    summ = (out.groupby(["model", "position"])
            .agg(n=("trial_id", "size"), kept_licks=("n_kept_licks", "mean"), lick1_ms=("lick1_t_peak_ms", "median"),
                 lick1_angle=("lick1_angle_at_ypeak", "median"), angle_max_l1=("angle_max_signed_lick1", "median"),
                 vxy_l1=("vxy_peak_lick1", "median"), jaw_moved=("jaw_pass_qc", "mean"),
                 mismatch=("candidate_no_lick_with_jaw_move", "sum"))
            .reset_index().sort_values(["model", "position"], key=lambda c: c.map(order) if c.name == "position" else c))
    print("\nper position (medians; angle + = image-right = mouse LEFT on cam4):")
    print(summ.round(1).to_string(index=False))
    out.to_csv(d / "kinematics_demo_per_trial.csv", index=False)
    print(f"-> {d / 'kinematics_demo_per_trial.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
