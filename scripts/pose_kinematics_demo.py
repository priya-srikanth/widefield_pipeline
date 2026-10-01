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

import numpy as np
import pandas as pd

from wfield_local import jaw_kinematics as jk
from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import tongue_jaw_mismatch as tm
from wfield_local import tongue_kinematics as tk

SEP = 2200


def run(pose_csv: Path, idx: pd.DataFrame, frame: SF.SpoutFrame | None, stop_after_cue_s: dict | None = None):
    """Tongue zero = the MOUTH (spout-frame origin); jaw keeps the data-driven baseline. With
    ``stop_after_cue_s`` ({trial_id: stop - cue, s}), every post-cue window ends at that trial's own stop."""
    df = oc.read_pose(pose_csv).iloc[:len(idx)]
    clean = {"tongue": oc.clean_windows(df, idx, "tongue", sep=SEP, x0y0=tuple(frame.origin) if frame else None),
             "jaw": oc.clean_windows(df, idx, "jaw", sep=SEP)}
    cues = clean["tongue"][2]
    pos = idx.groupby("trial_k", sort=True).position.first().to_numpy()
    tid = idx.groupby("trial_k", sort=True).trial_id.first().to_numpy()
    fps = clean["tongue"][0].fps
    stops = np.array([cues[k] + stop_after_cue_s[t] * fps if stop_after_cue_s and t in stop_after_cue_s else np.nan
                      for k, t in enumerate(tid)])
    trials_df = pd.DataFrame({"trial_id": tid, "cue_frame": cues.astype(float), "position": pos, "stop_frame": stops})
    trials = tk.trials_from_frame(trials_df)
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
    from wfield_local import trial_windows as TW
    b = TW.trial_bounds("PS93", "20260908", rv)
    stop_after = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    rows = []
    for name, f in (("DLC", d / "cue_windows_DLC.csv"), ("LP", a.lp)):
        T, J, M, c = run(f, idx, frame, stop_after)
        tg = c["tongue"][0]
        conf = tg.fill_method[c["tongue"][1]] == oc.FILL_NONE
        yv = tg.y_final[c["tongue"][1]][conf]
        xv = tg.x_final[c["tongue"][1]][conf]
        print(f"{name} tongue rel. MOUTH (raw confident frames): y p5/50/95 {np.percentile(yv, [5, 50, 95]).round(0)}  "
              f"x p5/50/95 {np.percentile(xv, [5, 50, 95]).round(0)};  kept-lick peak y p5/50/95 "
              f"{np.percentile(T.per_lick.y, [5, 50, 95]).round(0) if len(T.per_lick) else '-'}")
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
