"""Per-session lick summary across sessions (scripts.session_poses clips), both models, by spout position:
licks per trial, DAQ contact rate, protrusion, rise / retraction speed, peak angle, and DIRECTION DEVIATION from the
successful-lick path (`lick_reference`: vs pooled pre-stroke contact licks, and vs the same session's contact
licks). The tongue - spout-tip angle columns (dA_*) are kept but are NOT accuracy on cam4 (DECISIONS 2026-10-02).

    python -m scripts.session_lick_summary PS93:20260814 PS93:20260821 PS93:20260908
    python -m scripts.session_lick_summary PS93:20260814:full ...     # whole-session (O2) folders

Writes session_poses/lick_summary_<animal>.csv and prints the table. Epoch labels come from `epochs.epoch_of`.
Figures (session_poses/):
  lick_summary_<animal>.png       per position, each metric across the sessions (DLC solid, LP dashed)
  lick_phase_modes_<animal>.png   protrusion over the lick, per-lick-extent vs centered phase (shape check)
  lick_direction_shift_<animal>.png  direction vs the pre-stroke successful path, ALL licks: mean paths in the
                                  spout frame, deviation over the lick, peak deviation median + 95% CI
  lick_phase_epochs_<animal>.png  per position, over the lick phase, one line per session (DLC), ALL licks
                                  (incomplete / short included): angle, tongue - spout angle (all; no-contact),
                                  protrusion
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import trial_windows as TW

MIN_EXT = 100.0


def session_tables(spec: str, rv):
    import scripts.pose_kinematics_demo as D
    from scripts.session_poses import session_dir
    animal, date, d = session_dir(rv, spec)
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


def summarize(animal, date, epoch, model, T, J, M, pl=None) -> pd.DataFrame:
    pl = T.per_lick if pl is None else pl
    pt = T.per_trial
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
            "dA_all_med": lk.delta_angle_deg.median(),
            "dA_nocontact_all_med": lk.loc[lk.contact == False, "delta_angle_deg"].median(),   # noqa: E712
            "n_nocontact_all": int((lk.contact == False).sum()),                                # noqa: E712
            "dA_contact_med": dc.median(), "dA_nocontact_med": dn.median(), "n_contact_big": len(dc),
            "n_nocontact_big": len(dn), "dA_miss_minus_hit": dn.median() - dc.median() if len(dn) and len(dc) else np.nan,
            # deviation from the successful-lick direction (lick_reference): vs the PRE-STROKE contact licks and
            # vs this session's own contact licks, at the peak
            **{f"dev_{r}_{w}_med": (lk.loc[sel, f"dev_{r}_deg"].median() if f"dev_{r}_deg" in lk else np.nan)
               for r in ("pre", "session")
               for w, sel in (("all", lk.contact.notna()), ("contact", lk.contact == True),       # noqa: E712
                              ("nocontact", lk.contact == False))},                               # noqa: E712
            "jaw_moved": pd.to_numeric(J[J.position == pos].jaw_pass_qc, errors="coerce").mean(),
            "mismatch_trials": int(pd.to_numeric(mm, errors="coerce").sum()),
        })
    return pd.DataFrame(rows)


EPOCH_COLORS = {"pre": "0.2", "acute": "tab:red", "subacute": "tab:orange", "chronic": "tab:blue"}
_SHADES = {"pre": ["0.2", "0.45", "0.65"], "acute": ["#d62728", "#ff7f7f", "#8b0000"],
           "subacute": ["#ff7f0e", "#ffbb78", "#a65300"], "chronic": ["#1f77b4", "#7fb8e6", "#0b3d66"]}
_SESSION_COLOR: dict = {}


def session_color(date: str, ep: str) -> str:
    """One colour per session: the epoch's hue, a different shade for each further session of that epoch."""
    key = (date, ep)
    if key not in _SESSION_COLOR:
        k = sum(1 for (_d, e) in _SESSION_COLOR if e == ep)
        _SESSION_COLOR[key] = _SHADES.get(ep, ["0.5"])[min(k, 2)]
    return _SESSION_COLOR[key]
METRICS = [("licks_per_trial", "licks / trial"), ("contact_rate", "DAQ contact rate"),
           ("frac_trials_no_lick", "trials with no lick"), ("protrusion_med_px", "protrusion at peak (px)"),
           ("v_rise_med", "rise speed (px/s)"), ("v_retract_med", "retraction speed (px/s)"),
           ("angle_peak_med", "angle at peak (deg)"),
           ("dev_pre_contact_med", "deviation vs pre-stroke\nsuccessful, contact (deg)"),
           ("dev_pre_nocontact_med", "deviation vs pre-stroke\nsuccessful, no contact (deg)"),
           ("dev_session_nocontact_med", "no-contact deviation vs\nsame-session contact (deg)"),
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


def _boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, (n, len(v))), axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def plot_direction_shift(devs: dict, positions, out, lip_px: float = 30.0) -> None:
    """Priya 2026-10-02: "the leftward deviation is what I see by eye but it's not clear on the graphs" + "plot the
    delta vs pre-stroke for all licks". One figure, ALL licks:
      row 1  mean tongue PATH in the spout frame (LR across, AP down = out of the mouth), per session (DLC) --
             what the eye sees on the video; the pre-stroke successful (contact) path dotted
      row 2  deviation from the pre-stroke successful path over the lick phase, all licks, one symmetric scale
      row 3  deviation at the peak: median + bootstrap 95% CI per session -- all / contact / no-contact, DLC and LP
    + = image-right = mouse LEFT on cam4; - = image-left = mouse RIGHT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(devs, key=lambda k: k[0])
    sess = list(dict.fromkeys((k[0], k[1]) for k in keys))
    fig, axs = plt.subplots(3, len(positions), figsize=(3.2 * len(positions), 11), squeeze=False,
                            gridspec_kw={"height_ratios": [1.3, 1, 1]})
    lim2 = 0.0
    for (date, ep, m), (pl, ph) in devs.items():
        if m != "DLC":
            continue
        c = session_color(date, ep)
        for j, pos in enumerate(positions):
            d = ph[(ph.position == pos) & (ph.protrusion_px >= lip_px)]
            g = d.groupby("phase")
            ax = axs[0, j]
            ax.plot(g.lr_px.mean(), g.ap_px.mean(), "-", color=c, lw=2, label=f"{date} {ep}")
            pk = d[np.isclose(d.phase, 0.5)]
            ax.plot(pk.lr_px.mean(), pk.ap_px.mean(), "o", color=c, ms=6)
            if ep == "pre":
                cc = d.merge(pl.loc[pl.contact == True, ["trial_id", "lick_idx"]], on=["trial_id", "lick_idx"])   # noqa: E712
                gc = cc.groupby("phase")
                ax.plot(gc.lr_px.mean(), gc.ap_px.mean(), ":", color="k", lw=1.2, label="pre contact (reference)")
            if "dev_pre_deg" in d:
                gd = d.groupby("phase").dev_pre_deg
                mu, se = gd.mean(), gd.std() / np.sqrt(gd.count())
                ax2 = axs[1, j]
                ax2.plot(mu.index, mu, color=c, lw=2)
                ax2.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.15, lw=0)
                if len(mu) and np.isfinite(mu).any():
                    lim2 = max(lim2, float(np.nanmax(np.abs(mu))))
    for j, pos in enumerate(positions):
        ax = axs[0, j]
        ax.plot(0, 0, "*", ms=12, mfc="yellow", mec="k")
        ax.axvline(0, color="0.7", lw=0.6, ls="--")
        ax.set_ylim(220, -10)
        ax.set_xlim(-80, 80)
        ax.set_aspect("equal")
        ax.set_title(pos, fontsize=11)
        ax.set_xlabel("LR px  (<- image-left = mouse R | image-right = mouse L ->)", fontsize=7)
        if j == 0:
            ax.set_ylabel("AP px (out of the mouth)")
            ax.legend(fontsize=6, loc="lower left")
        ax2 = axs[1, j]
        ax2.axhline(0, color="k", lw=0.8)
        ax2.axvline(0.5, color="k", lw=0.4, ls=":")
        ax2.set_ylim(-1.1 * lim2, 1.1 * lim2)
        ax2.set_xlabel("lick phase (peak = 0.5)")
        if j == 0:
            ax2.set_ylabel("deviation vs pre-stroke\nsuccessful path, ALL licks (deg)\n(- = image-left = mouse R)")
    # row 3: peak deviation per session, median + 95% CI
    lim3 = 0.0
    for j, pos in enumerate(positions):
        ax = axs[2, j]
        for k_m, (m, mk) in enumerate((("DLC", "o"), ("LP", "s"))):
            for k_g, (lab, sel, col) in enumerate((("all", lambda d: d.contact.notna(), "k"),
                                                   ("contact", lambda d: d.contact == True, "tab:green"),     # noqa: E712
                                                   ("no contact", lambda d: d.contact == False, "tab:red"))):  # noqa: E712
                for i, (date, ep) in enumerate(sess):
                    if (date, ep, m) not in devs:
                        continue
                    pl = devs[(date, ep, m)][0]
                    d = pl[pl.position == pos]
                    if "dev_pre_deg" not in d:
                        continue
                    med, lo, hi = _boot_ci(d.loc[sel(d), "dev_pre_deg"])
                    xx = i + (k_g - 1) * 0.22 + (k_m - 0.5) * 0.08
                    ax.errorbar(xx, med, yerr=[[med - lo], [hi - med]] if np.isfinite(lo) else None, fmt=mk,
                                color=col, ms=4, mfc=col if m == "DLC" else "white", lw=1,
                                label=f"{lab} ({m})" if (i == 0 and j == 0) else None)
                    if np.isfinite(hi):
                        lim3 = max(lim3, abs(lo), abs(hi))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(sess)))
        ax.set_xticklabels([f"{d}\n{e}" for d, e in sess], fontsize=7)
        if j == 0:
            ax.set_ylabel("deviation at the peak vs pre-stroke\nsuccessful, median + 95% CI (deg)")
            ax.legend(fontsize=5, ncol=2, loc="lower left")
    for j in range(len(positions)):
        axs[2, j].set_ylim(-1.05 * min(lim3, 90), 1.05 * min(lim3, 90))
    fig.suptitle("Tongue direction relative to the PRE-STROKE successful (contact) licks at each position -- all licks "
                 "(incomplete / short included; lip zone < 30 px masked). Row 1 DLC paths, row 2 DLC over the lick, "
                 "row 3 both models (filled DLC, open LP). Head-pose / camera rotation between sessions is not removed.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_phase_modes(centered: dict, extent: dict, positions, out, cycle_ms) -> None:
    """Protrusion over the lick, the two phase definitions side by side (DLC), one line per session: does a
    peak-shape difference between sessions survive putting every lick on one real-time scale?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, len(positions), figsize=(3.0 * len(positions), 6.4), squeeze=False, sharex=True)
    for row, (lab, src) in enumerate((("per-lick visible extent (rise -> 0-0.5, fall -> 0.5-1)", extent),
                                      (f"centered on peak, fixed {cycle_ms:.0f} ms window (stroke_orofacial)", centered))):
        for key, ph in src.items():
            date, ep = key[0], key[1]
            if len(key) == 3 and key[2] != "DLC":
                continue
            for j, pos in enumerate(positions):
                g = ph[ph.position == pos].groupby("phase").protrusion_px
                if not len(g):
                    continue
                mu, se = g.mean(), g.std() / np.sqrt(g.count())
                ax = axs[row, j]
                c = session_color(date, ep)
                ax.plot(mu.index, mu, color=c, lw=2, label=f"{date} {ep}")
                ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.15, lw=0)
                ax.axvline(0.5, color="k", lw=0.4, ls=":")
                if row == 0:
                    ax.set_title(pos, fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"protrusion (px)\n{lab}", fontsize=7)
                if row == 1:
                    ax.set_xlabel("lick phase")
    axs[0, 0].legend(fontsize=6)
    fig.suptitle("DLC: lick shape under the two phase definitions. In the centered version NaN (tongue in) frames are "
                 "left out of the mean, so the curve ends sit higher than the per-lick lows.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_phase_epochs(phases: dict, positions, out, min_ext=0.0, min_phase_px=30.0, cycle_ms=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Priya 2026-10-02: ALL licks, incomplete / short ones included; the tongue-spout angle wherever the tongue was
    rows = [("angle_deg", "angle (deg), all licks", None),
            ("dev_pre_deg", "deviation vs pre-stroke\nsuccessful path, contact (deg)", True),
            ("dev_pre_deg", "deviation vs pre-stroke\nsuccessful path, no contact (deg)", False),
            ("dev_session_deg", "deviation vs same-session\ncontact path, no contact (deg)", False),
            ("protrusion_px", "protrusion (px), all licks", None)]
    fig, axs = plt.subplots(len(rows), len(positions), figsize=(3.0 * len(positions), 3.0 * len(rows)),
                            squeeze=False, sharex=True)
    for (label, epoch), ph in phases.items():
        ph = ph[ph.protrusion_max_px >= min_ext]
        for i, (var, lab, contact) in enumerate(rows):
            d = ph if contact is None else ph[ph.contact == contact]           # noqa: E712
            if var not in d:
                continue
            if var != "protrusion_px":
                d = d.assign(**{var: d[var].where(d.protrusion_px >= min_phase_px)})
            for j, pos in enumerate(positions):
                g = d[d.position == pos].groupby("phase")[var]
                if not len(g):
                    continue
                mu, se = g.mean(), g.std() / np.sqrt(g.count())
                ax = axs[i, j]
                c = session_color(label, epoch)
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
    fig.suptitle(f"DLC: tongue over the lick by session (phase centered on each peak, window = pre-stroke median ILI "
                 f"{cycle_ms or float('nan'):.0f} ms); deviation = angle - the mean SUCCESSFUL (contact) lick path at "
                 f"that position (pre-stroke, or same session) -- all licks incl. incomplete / short (angle points < "
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
    from wfield_local import lick_reference as LR
    runs = []                                    # (animal, date, epoch, model, T, J, M)
    for s in a.sessions:
        animal, date = s.split(":")[:2]
        res, _ = session_tables(s, rv)
        ep = epochs.epoch_of(f"{animal}_{date[4:]}")
        runs += [(animal, date, ep, m, T, J, M) for m, (T, J, M) in res.items()]
    # lick phase on ONE time scale for every session (stroke_orofacial): centered on each peak, width = the animal's
    # PRE-STROKE median within-bout ILI (per model), so a shape change between sessions is real timing, not a
    # per-session normalisation. The per-lick visible-extent version is kept for the comparison figure.
    from wfield_local import tongue_kinematics as tk
    ili_pre, extent_phase = {}, {}
    for m in {r[3] for r in runs}:
        pre = [r for r in runs if r[3] == m and r[2] == "pre"]
        il = np.concatenate([tk.within_bout_ili_ms(r[4].trial_results) for r in (pre or [r for r in runs if r[3] == m])])
        ili_pre[m] = float(np.median(il))
        print(f"{m}: phase cycle = {'pre-stroke' if pre else 'all-session'} median within-bout ILI "
              f"{ili_pre[m]:.0f} ms (n={len(il)})")
    new_runs = []
    for animal, date, ep, m, T, J, M in runs:
        extent_phase[(date[4:], ep, m)] = tk.lick_phase_table(
            T.trial_results, T.X0, T.Y0, T.spout_frame, T.fps, n_points=21, mode="extent",
            spout_angle_by_trial=T.spout_angle_by_trial)
        T.lick_phase = tk.lick_phase_table(
            T.trial_results, T.X0, T.Y0, T.spout_frame, T.fps, n_points=int(T.params["phase"]["n_points"]),
            mode="centered", cycle_ms=ili_pre[m], spout_angle_by_trial=T.spout_angle_by_trial)
        new_runs.append((animal, date, ep, m, T, J, M))
    runs = new_runs
    # references (lick_reference): pooled PRE-STROKE contact licks per model, and each session's own contact licks
    ref_pre = {}
    for m in {r[3] for r in runs}:
        pre = [r for r in runs if r[3] == m and r[2] == "pre"]
        if pre:
            ref_pre[m] = LR.build_reference([r[4].per_lick for r in pre], [r[4].lick_phase for r in pre])
            print(f"{m}: pre-stroke reference from {', '.join(r[1] for r in pre)} -- contact licks per position "
                  f"{ref_pre[m]['n'].to_dict()}")
    parts, phases, devs = [], {}, {}
    for animal, date, ep, m, T, J, M in runs:
        pl, ph = T.per_lick, T.lick_phase
        pl, ph = LR.add_deviation(pl, ph, LR.build_reference([pl], [ph]), "session")
        if m in ref_pre:
            pl, ph = LR.add_deviation(pl, ph, ref_pre[m], "pre")
        parts.append(summarize(animal, date, ep, m, T, J, M, pl=pl))
        devs[(date[4:], ep, m)] = (pl, ph)
        if m == "DLC":
            phases[(date[4:], ep)] = ph.merge(pl[["trial_id", "lick_idx", "contact", "protrusion_max_px"]],
                                              on=["trial_id", "lick_idx"])
    tab = pd.concat(parts, ignore_index=True)
    order = {p: i for i, p in enumerate(POSITIONS)}
    tab = tab.sort_values(["model", "position", "date"], key=lambda c: c.map(order) if c.name == "position" else c)
    out = dlc_project.project_dir(rv).parent / "session_poses" / f"lick_summary_{a.sessions[0].split(':')[0]}.csv"
    tab.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    cols = ["model", "position", "epoch", "licks_per_trial", "contact_rate", "frac_trials_no_lick", "protrusion_med_px",
            "v_rise_med", "v_retract_med", "angle_peak_med", "dev_pre_contact_med", "dev_pre_nocontact_med",
            "dev_session_nocontact_med", "n_nocontact_all", "jaw_moved", "mismatch_trials"]
    print(tab[cols].round(2).to_string(index=False))
    print(f"-> {out}")
    positions = [p for p in POSITIONS if p in set(tab.position)]
    plot_summary(tab, positions, out.with_suffix(".png"))
    print(f"-> {out.with_suffix('.png')}")
    out2 = out.parent / f"lick_phase_epochs_{a.sessions[0].split(':')[0]}.png"
    plot_phase_epochs(phases, positions, out2, cycle_ms=ili_pre.get("DLC"))
    print(f"-> {out2}")
    out4 = out.parent / f"lick_direction_shift_{a.sessions[0].split(':')[0]}.png"
    plot_direction_shift(devs, positions, out4)
    print(f"-> {out4}")
    out3 = out.parent / f"lick_phase_modes_{a.sessions[0].split(':')[0]}.png"
    plot_phase_modes({k: v for k, v in phases.items()}, extent_phase, positions, out3, ili_pre.get("DLC"))
    print(f"-> {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
