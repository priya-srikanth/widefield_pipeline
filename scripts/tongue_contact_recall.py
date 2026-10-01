"""Recall of kept tongue licks against DAQ spout contacts (every contact is a lick) on the PS93 0908 cue clip,
per model / cleaning cutoff. A lower bound on missed licks: incomplete licks never touch the spout.

    PYTHONPATH=. python scripts/tongue_contact_recall.py
"""
import numpy as np, pandas as pd
import scripts.pose_kinematics_demo as D
from wfield_local import orofacial_clean as oc, spout_frame as SF, trial_windows as TW, dlc_project, dlc_frames
from wfield_local.paths import PathResolver
rv = PathResolver(); d = dlc_project.project_dir(rv).parent / "lp_vs_dlc_cue_traces" / "PS93_20260908"
idx = pd.read_csv(d / "cue_windows_index.csv"); dlc = oc.read_pose(d / "cue_windows_DLC.csv")
spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
fr = SF.from_medians(SF.position_medians(dlc[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
b = TW.trial_bounds("PS93", "20260908", rv); st = dict(zip(b.trial_id, b.stop_s - b.cue_s)); cue = dict(zip(b.trial_id, b.cue_s))
sid = sorted((rv.root("behavior_out") and __import__("pathlib").Path(rv.root("behavior_out")) / "sessions" / "PS93" / "20260908").glob("*_trials.csv"))[-1].name[:-11]
licks = dlc_frames.lick_onsets("PS93", "20260908", sid, rv)
orig = oc.params
rows = []
for name, f, lk in (("DLC", d / "cue_windows_DLC.csv", 0.6), ("DLC", d / "cue_windows_DLC.csv", 0.4),
                    ("LP", __import__("pathlib").Path.home() / "lp_cue_tmp" / "cue_windows_LP.csv", 0.6)):
    oc.params = lambda bp, lk=lk: {**orig(bp), "lk_thr": lk}
    T, _, _, _ = D.run(f, idx, fr, st)
    oc.params = orig
    pl = T.per_lick
    n_c = n_hit = 0
    per_pos = {}
    for tid in T.per_trial.trial_id:
        c_rel = (licks[(licks >= cue[tid] + 0.07) & (licks <= cue[tid] + st[tid])] - cue[tid]) * 1000
        k = pl.loc[pl.trial_id == tid, "t_ms"].to_numpy()
        # a contact is "found" if a kept lick PEAK lies within 60 ms of it (the peak follows contact onset closely)
        hit = sum(bool(len(k)) and np.min(np.abs(k - c)) <= 60 for c in c_rel)
        n_c += len(c_rel); n_hit += hit
        pos = T.per_trial.loc[T.per_trial.trial_id == tid, "position"].iloc[0]
        a_, b_ = per_pos.get(pos, (0, 0)); per_pos[pos] = (a_ + len(c_rel), b_ + hit)
    rows.append((name, lk, n_c, n_hit, len(pl), per_pos))
for name, lk, n_c, n_hit, n_k, per_pos in rows:
    print(f"{name} lk {lk}: DAQ contacts in response windows {n_c}; found by a kept lick {n_hit} ({100*n_hit/n_c:.1f}%); kept licks {n_k} (non-contact licks >= {n_k - n_hit})")
    print("   per position recall:", {p: f"{h}/{c}" for p, (c, h) in per_pos.items()})
