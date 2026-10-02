"""Per-session lick summary across sessions (scripts.session_poses clips), both models, by spout position:
licks per trial, DAQ contact rate, protrusion, rise / retraction speed, peak angle, and REACH ACCURACY -- tongue
minus spout-tip angle at the peak for contact vs no-contact licks, size-matched (licks reaching >= 100 px).

    python -m scripts.session_lick_summary PS93:20260814 PS93:20260821 PS93:20260908

Writes session_poses/lick_summary_<animal>.csv and prints the table. Epoch labels come from `epochs.epoch_of`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import trial_windows as TW

MIN_EXT = 100.0


def session_tables(animal: str, date: str, rv):
    import scripts.pose_kinematics_demo as D
    from wfield_local import dlc_project
    d = dlc_project.project_dir(rv).parent / "session_poses" / f"{animal}_{date}"
    idx = pd.read_csv(d / "windows_index.csv")
    raw = oc.read_pose(d / "windows_DLC.csv").iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(raw[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds(animal, date, rv)
    stop = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    out = {}
    for m in ("DLC", "LP"):
        f = d / f"windows_{m}.csv"
        if f.exists():
            T, J, M, _ = D.run(f, idx, frame, stop, contacts_ms=contacts)
            out[m] = (T, J, M)
    return out, frame


def summarize(animal, date, epoch, model, T, J, M) -> pd.DataFrame:
    pl, pt = T.per_lick, T.per_trial
    rows = []
    for pos, g in pt.groupby("position"):
        lk = pl[pl.position == pos]
        big = lk[lk.protrusion_max_px >= MIN_EXT]
        dc = big.loc[big.contact == True, "delta_angle_deg"]      # noqa: E712  (object column after NaN merge)
        dn = big.loc[big.contact == False, "delta_angle_deg"]     # noqa: E712
        mm = M.trials[M.trials.position == pos].candidate_no_lick_with_jaw_move
        rows.append({
            "animal": animal, "date": date, "epoch": epoch, "model": model, "position": pos, "n_trials": len(g),
            "licks_per_trial": len(lk) / len(g), "contact_rate": lk.contact.mean(),
            "frac_trials_no_lick": float((g.n_kept_licks == 0).mean()),
            "protrusion_med_px": lk.protrusion_px.median(), "reach_frac_med": lk.reach_frac.median(),
            "v_rise_med": lk.max_velocity_y_px_per_s.median(),
            "v_retract_med": lk.max_retract_velocity_y_px_per_s.median(),
            "angle_peak_med": lk.peak_angle_deg.median(),
            "dA_contact_med": dc.median(), "dA_nocontact_med": dn.median(), "n_contact_big": len(dc),
            "n_nocontact_big": len(dn), "dA_miss_minus_hit": dn.median() - dc.median() if len(dn) and len(dc) else np.nan,
            "jaw_moved": pd.to_numeric(J[J.position == pos].jaw_pass_qc, errors="coerce").mean(),
            "mismatch_trials": int(pd.to_numeric(mm, errors="coerce").sum()),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    from wfield_local import dlc_project, epochs
    from wfield_local.paths import PathResolver
    from wfield_local.tongue_kinematics import POSITIONS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    a = ap.parse_args(argv)
    rv = PathResolver()
    parts = []
    for s in a.sessions:
        animal, date = s.split(":")
        res, _ = session_tables(animal, date, rv)
        ep = epochs.epoch_of(f"{animal}_{date[4:]}")
        for m, (T, J, M) in res.items():
            parts.append(summarize(animal, date, ep, m, T, J, M))
    tab = pd.concat(parts, ignore_index=True)
    order = {p: i for i, p in enumerate(POSITIONS)}
    tab = tab.sort_values(["model", "position", "date"], key=lambda c: c.map(order) if c.name == "position" else c)
    out = dlc_project.project_dir(rv).parent / "session_poses" / f"lick_summary_{a.sessions[0].split(':')[0]}.csv"
    tab.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    cols = ["model", "position", "epoch", "licks_per_trial", "contact_rate", "frac_trials_no_lick", "protrusion_med_px",
            "v_rise_med", "v_retract_med", "angle_peak_med", "dA_contact_med", "dA_nocontact_med", "n_contact_big",
            "n_nocontact_big", "dA_miss_minus_hit", "jaw_moved", "mismatch_trials"]
    print(tab[cols].round(2).to_string(index=False))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
