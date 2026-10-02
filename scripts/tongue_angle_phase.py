"""Tongue angle: (1) why `angle_max_signed_lick1` and the lick-1 angle at the peak can disagree in sign -- a visual
example per disagreeing trial; (2) angle and protrusion over the LICK PHASE, per spout position, both models.

    python -m scripts.tongue_angle_phase            # PS93 0908 cue clip
    python -m scripts.tongue_angle_phase --session PS93:20260814

Priya, 2026-10-01: "show me the angle sign flips with a visual example ... the angle phase plots are actually some
of the most informative". Angle = atan2(lr, ap) in the spout frame, origin = the mouth, 0 = straight out toward
far_center, + = image-right (= mouse LEFT on cam4).
Figure 1, one row per trial where the signs disagree (DLC): lick-1 path in the spout frame coloured by phase
(ring = peak, X = the max-|angle| frame, +-52 ms, square = the same with the outer-half restriction); angle vs
time (smoothed per-frame angle; distance from the mouth on the right axis); the video frame at the peak and at
the max-|angle| frame, with the tongue point, the mouth and the 0-degree line.
Figure 2: angle and protrusion vs lick phase (0 = tongue appears, 0.5 = peak, 1 = gone), mean +- SEM per position,
DLC and LP, from `TongueKinematics.lick_phase`; thin grey = individual licks of one position.
Figure 3 (Priya: reach accuracy -- "does the tongue deviate from the spout during the trajectory even if contact is
normal"): delta angle = tongue angle - that trial's spout-tip angle (both from the mouth, spout frame) over the
lick phase, per position, licks WITH a DAQ spout contact vs WITHOUT (incomplete / missed), DLC and LP -- only licks
reaching >= MIN_EXTENSION_PX, and phase points < MIN_PHASE_PX from the mouth masked (near the lips the vector is
short and the tongue sits image-left of the mouth point, which alone makes small licks read ~-50 deg). On cam4 the
absolute delta is dominated by depth projection (spout tips sit 60-100 px from the mouth at +-40 deg, the tongue
reaches 140-185 px mostly down the image), so read CHANGES (between groups, sessions, epochs), not the level.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import tongue_kinematics as tk
from wfield_local import trial_windows as TW

POS_COLORS = {"far_L": "#1f77b4", "close_L": "#6baed6", "far_center": "#2ca02c", "close_center": "#98df8a",
              "close_R": "#ff9896", "far_R": "#d62728"}
MAX_ROWS = 5
MIN_EXTENSION_PX = 100.0   # fig 3: only licks reaching this far (small "tip at the lips" licks have short, unstable vectors)
MIN_PHASE_PX = 50.0        # figs 2-3: phase points closer than this to the mouth are masked (angle unstable there)


def _load(a, rv):
    from wfield_local import dlc_project
    if a.session:
        animal, date = a.session.split(":")
        d = dlc_project.project_dir(rv).parent / "session_poses" / f"{animal}_{date}"
        return animal, date, d, pd.read_csv(d / "windows_index.csv"), \
            {"DLC": d / "windows_DLC.csv", "LP": d / "windows_LP.csv"}, d / "windows.mp4"
    d = dlc_project.project_dir(rv).parent / "lp_vs_dlc_cue_traces" / "PS93_20260908"
    return "PS93", "20260908", d, pd.read_csv(d / "cue_windows_index.csv"), \
        {"DLC": d / "cue_windows_DLC.csv", "LP": a.lp}, d / "cue_windows.mp4"


def main(argv=None) -> int:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import scripts.pose_kinematics_demo as D
    from wfield_local.paths import PathResolver
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--session", default=None)
    ap_.add_argument("--lp", type=Path, default=Path.home() / "lp_cue_tmp" / "cue_windows_LP.csv")
    a = ap_.parse_args(argv)
    rv = PathResolver()
    animal, date, d, idx, poses, video = _load(a, rv)
    poses = {m: f for m, f in poses.items() if f.exists()}
    raw = {m: oc.read_pose(f).iloc[:len(idx)] for m, f in poses.items()}
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(raw["DLC"][["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds(animal, date, rv)
    stop = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    res = {m: D.run(f, idx, frame, stop, contacts_ms=contacts) for m, f in poses.items()}
    tag = f"{animal}_{date}"

    # ---------------------------------------------------------------- figure 1: sign disagreements (DLC)
    T, _J, _M, clean = res["DLC"]
    tg, pos_map, _ = clean["tongue"]
    fps = tg.fps
    clip_row = np.full(len(tg.y_final), -1)
    clip_row[pos_map] = np.arange(len(pos_map))
    xa = T.x_clean + tg.X0
    ya = np.where(T.is_baseline_fill_clean, np.nan, T.y_clean) + tg.Y0
    AP, LR = tk.spout_coords(xa, ya, tk.SpoutFrame(tuple(frame.origin), tuple(frame.ap_axis)))
    dist = np.hypot(AP, LR)
    ang = T.angle_per_frame
    pad = int(round(float(T.params["detector"]["peak_pad_ms"]) / 1000.0 * fps))
    frac = T.params["angle"].get("max_signed_min_frac_of_peak") or 0.5
    pt = T.per_trial
    flips = [k for k, r in pt.iterrows() if np.isfinite(r.lick1_angle_at_ypeak) and np.isfinite(r.angle_max_signed_lick1)
             and np.sign(r.lick1_angle_at_ypeak) != np.sign(r.angle_max_signed_lick1)][:MAX_ROWS]
    cap = cv2.VideoCapture(str(video))
    ox, oy = frame.origin
    rx, ry = -frame.ap_axis[0], -frame.ap_axis[1]
    fig, axs = plt.subplots(len(flips), 4, figsize=(15, 3.6 * len(flips)), squeeze=False,
                            gridspec_kw={"width_ratios": [1, 1.4, 1, 1]})
    for row, k in enumerate(flips):
        r, tres = pt.loc[k], T.trial_results[k]
        lk = tres.kept_licks[0]
        c = int(round(tres.cue_frame + lk["t_ms"] / 1000.0 * fps))
        on = tres.session_frame_lo + int(lk["on_frame"])
        off = tres.session_frame_lo + int(lk["off_frame"])
        w = np.arange(c - pad, c + pad + 1)
        j_old = w[int(np.nanargmax(np.abs(ang[w])))]
        wr = np.where(dist[w] >= frac * dist[c], ang[w], np.nan)
        j_new = w[int(np.nanargmax(np.abs(wr)))] if np.isfinite(wr).any() else c
        # path
        ax = axs[row, 0]
        ff = np.arange(on, off + 1)
        ax.plot(LR[w], AP[w], "-", color="0.85", lw=1)
        sc = ax.scatter(LR[ff], AP[ff], c=np.linspace(0, 1, len(ff)), cmap="viridis", s=14, zorder=3)
        ax.plot(LR[c], AP[c], "o", ms=11, mfc="none", mec="k", mew=1.5, label="peak")
        ax.plot(LR[j_old], AP[j_old], "X", ms=10, color="red", label="max |angle| (+-52 ms)")
        ax.plot(LR[j_new], AP[j_new], "s", ms=8, mfc="none", mec="orange", mew=2, label=f"max |angle|, >= {frac:g} x peak dist")
        ax.plot(0, 0, "*", ms=12, mfc="yellow", mec="k")
        ax.axvline(0, color="0.6", lw=0.6, ls="--")
        ax.set_xlabel("LR px (+ image-right)")
        ax.set_ylabel("AP px (out of mouth)")
        ax.set_xlim(-90, 90)
        ax.set_ylim(-10, 230)
        ax.set_title(f"trial {r.trial_id} {r.position}: lick 1 at {lk['t_ms']:.0f} ms\n"
                     f"angle at peak {r.lick1_angle_at_ypeak:+.0f}, max-signed {r.angle_max_signed_lick1:+.0f}", fontsize=8)
        if row == 0:
            ax.legend(fontsize=6, loc="upper right")
            fig.colorbar(sc, ax=ax, fraction=0.04, label="lick phase")
        # angle vs time
        ax = axs[row, 1]
        tt = (w - c) * 1000.0 / fps
        ax.plot(tt, ang[w], "-", color="tab:purple", lw=1.5, label="smoothed angle")
        ax.plot((j_old - c) * 1000 / fps, ang[j_old], "X", ms=10, color="red")
        ax.plot((j_new - c) * 1000 / fps, ang[j_new], "s", ms=8, mfc="none", mec="orange", mew=2)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_ylabel("angle (deg)", color="tab:purple")
        ax.set_xlabel("ms from lick-1 peak")
        a2 = ax.twinx()
        a2.plot(tt, dist[w], "-", color="0.5", lw=1)
        a2.axhline(frac * dist[c], color="orange", lw=0.6, ls=":")
        a2.set_ylabel("distance from mouth (px)", color="0.4")
        # frames
        for col, (fj, lab) in zip((2, 3), ((c, "peak"), (j_old, "max |angle| frame"))):
            ax = axs[row, col]
            ax.axis("off")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(clip_row[fj]))
            ok, im = cap.read()
            if not ok:
                continue
            x0, y0 = int(ox) - 110, int(oy) - 40
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[y0:y0 + 260, x0:x0 + 220])
            ax.plot(ox - x0, oy - y0, "*", ms=11, mfc="yellow", mec="k")
            ax.plot([ox - x0, ox - x0 + 240 * rx], [oy - y0, oy - y0 + 240 * ry], "--", color="yellow", lw=0.8)
            ax.plot([ox - x0, xa[fj] - x0], [oy - y0, ya[fj] - y0], "-", color="cyan", lw=1.2)
            ax.plot(xa[fj] - x0, ya[fj] - y0, "+", ms=12, mew=2, color="cyan")
            ax.set_xlim(0, 220)
            ax.set_ylim(260, 0)
            ax.set_title(f"{lab}: {(fj - c) * 1000 / fps:+.0f} ms, {dist[fj]:.0f} px out, angle {ang[fj]:+.0f}",
                         fontsize=8)
    cap.release()
    fig.suptitle(f"{animal} {date} DLC: trials where angle_max_signed_lick1 and the lick-1 angle at the peak "
                 f"disagree in sign (yellow dashed = 0 deg, toward far_center; cyan = mouth -> tongue)", fontsize=10)
    fig.tight_layout()
    out1 = d / f"tongue_angle_sign_examples_{tag}.png"
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"-> {out1} ({len(flips)} trials)")

    # ---------------------------------------------------------------- figure 2: angle / protrusion vs phase
    models = list(res)
    fig, axs = plt.subplots(2, len(models), figsize=(6.5 * len(models), 8), squeeze=False, sharex=True)
    for col, m in enumerate(models):
        ph = res[m][0].lick_phase.dropna(subset=["angle_deg"])
        ph = ph.assign(angle_deg=ph.angle_deg.where(ph.protrusion_px >= MIN_PHASE_PX))
        for row, (var, lab) in enumerate((("angle_deg", "angle (deg, + image-right)"),
                                          ("protrusion_px", "protrusion from mouth (px)"))):
            ax = axs[row, col]
            g0 = ph[ph.position == "far_L"]
            for _, gl in list(g0.groupby(["trial_id", "lick_idx"]))[:60]:
                ax.plot(gl.phase, gl[var], "-", color="0.85", lw=0.5, zorder=1)
            for pos in tk.POSITIONS:
                g = ph[ph.position == pos].groupby("phase")[var]
                if not len(g):
                    continue
                mu, se = g.mean(), g.std() / np.sqrt(g.count())
                ax.plot(mu.index, mu, "-", color=POS_COLORS[pos], lw=2, label=f"{pos} (n={g.count().max()})", zorder=3)
                ax.fill_between(mu.index, mu - se, mu + se, color=POS_COLORS[pos], alpha=0.2, lw=0, zorder=2)
            ax.axvline(0.5, color="k", lw=0.5, ls=":")
            ax.set_ylabel(lab)
            if row == 0:
                ax.axhline(0, color="0.6", lw=0.6)
                ax.set_title(f"{m}: every kept lick, phase 0 = tongue appears, 0.5 = peak, 1 = gone", fontsize=9)
                ax.legend(fontsize=7, ncol=2)
            else:
                ax.set_xlabel("lick phase")
    fig.suptitle(f"{animal} {date}: tongue angle and protrusion over the lick (mean +- SEM per spout position; "
                 f"grey = individual far_L licks)", fontsize=10)
    fig.tight_layout()
    out2 = d / f"tongue_angle_phase_{tag}.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"-> {out2}")

    # ---------------------------------------------------------------- figure 3: delta angle, contact vs not
    fig, axs = plt.subplots(len(models), 6, figsize=(20, 4.2 * len(models)), squeeze=False, sharex=True)
    for row, m in enumerate(models):
        T_ = res[m][0]
        ph = T_.lick_phase.merge(T_.per_lick[["trial_id", "lick_idx", "contact", "protrusion_max_px"]],
                                 on=["trial_id", "lick_idx"])
        ph = ph[(ph.protrusion_max_px >= MIN_EXTENSION_PX) & (ph.protrusion_px >= MIN_PHASE_PX)]
        ph = ph.dropna(subset=["delta_angle_deg"])
        for col, pos in enumerate(tk.POSITIONS):
            ax = axs[row, col]
            for cflag, colr, lab in ((True, POS_COLORS[pos], "contact"), (False, "k", "no contact")):
                g = ph[(ph.position == pos) & (ph.contact == cflag)].groupby("phase").delta_angle_deg
                if not len(g):
                    continue
                mu, se = g.mean(), g.std() / np.sqrt(g.count())
                ax.plot(mu.index, mu, color=colr, lw=2, ls="-" if cflag else "--",
                        label=f"{lab} (n={g.count().max()})")
                ax.fill_between(mu.index, mu - se, mu + se, color=colr, alpha=0.15, lw=0)
            ax.axvline(0.5, color="k", lw=0.5, ls=":")
            ax.set_title(f"{m} {pos}", fontsize=9)
            ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel("tongue - spout angle (deg)")
            if row == len(models) - 1:
                ax.set_xlabel("lick phase")
    fig.suptitle(f"{animal} {date}: tongue angle relative to the trial's spout tip over the lick, licks with vs "
                 f"without DAQ spout contact -- licks reaching >= {MIN_EXTENSION_PX:.0f} px only, phase points "
                 f"< {MIN_PHASE_PX:.0f} px from the mouth masked (mean +- SEM; + = image-right)", fontsize=10)
    fig.tight_layout()
    out3 = d / f"tongue_delta_angle_contact_{tag}.png"
    fig.savefig(out3, dpi=110)
    plt.close(fig)
    print(f"-> {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
