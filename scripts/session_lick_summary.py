"""Per-session lick summary across sessions (scripts.session_poses clips), both models, by spout position:
licks per trial, DAQ contact rate, protrusion, rise / retraction speed, peak angle, and REACH ACCURACY -- tongue
minus spout-tip angle at the peak for contact vs no-contact licks, size-matched (licks reaching >= 100 px).

    python -m scripts.session_lick_summary PS93:20260814 PS93:20260821 PS93:20260908

Writes session_poses/lick_summary_<animal>.csv and prints the table. Epoch labels come from `epochs.epoch_of`.
Figures (session_poses/):
  lick_summary_<animal>.png       per position, each metric across the sessions (DLC solid, LP dashed)
  lick_phase_epochs_<animal>.png  per position, over the lick phase, one line per session (DLC): angle,
                                  tongue - spout angle (contact licks reaching >= 100 px), protrusion
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


EPOCH_COLORS = {"pre": "0.2", "acute": "tab:red", "subacute": "tab:orange", "chronic": "tab:blue"}
METRICS = [("licks_per_trial", "licks / trial"), ("contact_rate", "DAQ contact rate"),
           ("frac_trials_no_lick", "trials with no lick"), ("protrusion_med_px", "protrusion at peak (px)"),
           ("v_rise_med", "rise speed (px/s)"), ("v_retract_med", "retraction speed (px/s)"),
           ("angle_peak_med", "angle at peak (deg)"), ("dA_contact_med", "tongue - spout angle, contact (deg)"),
           ("jaw_moved", "jaw moved (frac trials)")]


def plot_summary(tab: pd.DataFrame, positions, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sess = list(dict.fromkeys(tab.sort_values("date").apply(lambda r: f"{r.date[4:]}\n{r.epoch}", axis=1)))
    fig, axs = plt.subplots(len(METRICS), len(positions), figsize=(2.6 * len(positions), 2.1 * len(METRICS)),
                            squeeze=False, sharex=True, sharey="row")   # one scale per metric: positions comparable
    for i, (col, lab) in enumerate(METRICS):
        for j, pos in enumerate(positions):
            ax = axs[i, j]
            for m, ls in (("DLC", "-"), ("LP", "--")):
                g = tab[(tab.position == pos) & (tab.model == m)].sort_values("date")
                ax.plot(range(len(g)), g[col], ls, marker="o", ms=4, color="k" if m == "DLC" else "tab:purple",
                        label=m)
            ax.set_xticks(range(len(sess)))
            ax.set_xticklabels(sess, fontsize=7)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(pos, fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Per-session lick summary by spout position (10 trials / position / session); "
                 "angles: + = image-right = mouse LEFT on cam4", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_phase_epochs(phases: dict, positions, out, min_ext=MIN_EXT, min_phase_px=50.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("angle_deg", "angle (deg)", False), ("delta_angle_deg", "tongue - spout angle,\ncontact licks (deg)", True),
            ("protrusion_px", "protrusion (px)", False)]
    fig, axs = plt.subplots(len(rows), len(positions), figsize=(3.0 * len(positions), 3.0 * len(rows)),
                            squeeze=False, sharex=True)
    for (label, epoch), ph in phases.items():
        ph = ph[ph.protrusion_max_px >= min_ext]
        for i, (var, lab, contact_only) in enumerate(rows):
            d = ph[ph.contact == True] if contact_only else ph                  # noqa: E712
            if var != "protrusion_px":
                d = d.assign(**{var: d[var].where(d.protrusion_px >= min_phase_px)})
            for j, pos in enumerate(positions):
                g = d[d.position == pos].groupby("phase")[var]
                if not len(g):
                    continue
                mu, se = g.mean(), g.std() / np.sqrt(g.count())
                ax = axs[i, j]
                c = EPOCH_COLORS.get(epoch, "0.5")
                ax.plot(mu.index, mu, color=c, lw=2, label=f"{label} {epoch} (n={g.count().max()})")
                ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.15, lw=0)
                ax.axvline(0.5, color="k", lw=0.4, ls=":")
                if i == 0:
                    ax.set_title(pos, fontsize=10)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=9)
                if i == len(rows) - 1:
                    ax.set_xlabel("lick phase")
    for j in range(len(positions)):
        axs[0, j].legend(fontsize=6)
    fig.suptitle(f"DLC: tongue over the lick by session (licks reaching >= {min_ext:.0f} px; angle points < "
                 f"{min_phase_px:.0f} px from the mouth masked; mean +- SEM; + = image-right = mouse LEFT)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import dlc_project, epochs
    from wfield_local.paths import PathResolver
    from wfield_local.tongue_kinematics import POSITIONS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    a = ap.parse_args(argv)
    rv = PathResolver()
    parts, phases = [], {}
    for s in a.sessions:
        animal, date = s.split(":")
        res, _ = session_tables(animal, date, rv)
        ep = epochs.epoch_of(f"{animal}_{date[4:]}")
        for m, (T, J, M) in res.items():
            parts.append(summarize(animal, date, ep, m, T, J, M))
        if "DLC" in res and res["DLC"][0].lick_phase is not None:
            T = res["DLC"][0]
            phases[(date[4:], ep)] = T.lick_phase.merge(
                T.per_lick[["trial_id", "lick_idx", "contact", "protrusion_max_px"]], on=["trial_id", "lick_idx"])
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
    positions = [p for p in POSITIONS if p in set(tab.position)]
    plot_summary(tab, positions, out.with_suffix(".png"))
    print(f"-> {out.with_suffix('.png')}")
    out2 = out.parent / f"lick_phase_epochs_{a.sessions[0].split(':')[0]}.png"
    plot_phase_epochs(phases, positions, out2)
    print(f"-> {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
