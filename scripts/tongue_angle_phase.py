"""Tongue angle: (1) why `angle_max_signed_lick1` and the lick-1 angle at the peak can disagree in sign -- a visual
example per disagreeing trial; (2) angle and protrusion over the LICK PHASE, per spout position, both models.

    python -m scripts.tongue_angle_phase            # PS93 0908 cue clip
    python -m scripts.tongue_angle_phase --session PS93:20260814

Priya, 2026-10-01: "show me the angle sign flips with a visual example ... the angle phase plots are actually some
of the most informative". Angle = atan2(lr, ap) in the spout frame, origin = the mouth, 0 = straight out toward
far_center, + = image-right (= mouse LEFT on cam4).
Figure 1, one row per trial where the signs disagree (DLC): lick-1 path in IMAGE coordinates around the mouth
(mouth at top centre, image-right on the right, the tongue going DOWN -- oriented like the video frames beside it;
Priya 2026-10-06; the dashed line = the 0-degree / far_center axis), coloured by phase
(ring = peak, X = the max-|angle| frame, +-52 ms, square = the same with the outer-half restriction); angle vs
time (smoothed per-frame angle; distance from the mouth on the right axis); the video frame at the peak and at
the max-|angle| frame, with the tongue point, the mouth and the 0-degree line.
Figure 2: angle and protrusion vs lick phase (0 = tongue appears, 0.5 = peak, 1 = gone), mean +- SEM per position,
DLC and LP, from `TongueKinematics.lick_phase`; thin grey = individual licks of one position.
Figure 3 (Priya: reach accuracy -- "does the tongue deviate ... even if contact is normal"): DEVIATION = tongue
angle - the mean angle of this session's CONTACT licks at that position and lick phase (`lick_reference`; the
pre-stroke reference is in `session_lick_summary`). Replaced tongue - spout-tip angle, which is ~±30-35° even on
contact licks on cam4 (DECISIONS 2026-10-02) -- the earlier text follows for the record: over the
lick phase, per position, by outcome: contact licks, and no-contact (incomplete / missed) licks split by reach
(>= 100 px, 60-100, < 60) -- ALL licks, the angle taken wherever the tongue was (Priya 2026-10-02); phase points
< MIN_PHASE_PX from the mouth masked. Near the lips the vector is short and the tongue sits image-left of the mouth
point, so SHORT licks read strongly negative from geometry alone -- compare like with like. On cam4 the
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

from wfield_local.spout_behavior import POSITIONS as _SB_POS  # noqa: E402
from wfield_local.spout_behavior import position_style as _style  # noqa: E402

#: The cohort palette and order (`spout_behavior.position_style`): hue = side, lightness = ring, style = side.
POS_ORDER = [q["name"] for q in _SB_POS]
POS_COLORS = {n: _style(n)[0] for n in POS_ORDER}
POS_LS = {n: _style(n)[2] for n in POS_ORDER}
MAX_ROWS = 5
EXT_BINS = (60.0, 100.0)   # fig 3: no-contact licks split into short (< 60 px) / partial / full-length (>= 100 px)
MIN_COV = 0.2              # figs 2-3: phase points where < 20 % of licks show the tongue are hidden
MIN_PHASE_PX = 30.0        # figs 2-3: phase points inside the lip zone (< 30 px from the mouth, where the tongue first
                           # appears) are masked -- the angle of a vector that short is unstable. Was 50 px with a
                           # >= 100 px lick filter; Priya 2026-10-02 asked to include short / incomplete licks.


def _load(a, rv):
    from wfield_local import dlc_project
    if a.session:
        from scripts.session_poses import read_index, session_dir
        animal, date, d = session_dir(rv, a.session)
        video = d / "windows.mp4"
        if not video.exists():          # whole-video (O2) folders: frames come from the original video
            from wfield_local.o2_inference import videos_for
            video = videos_for(rv, animal, date)[0]
        return animal, date, d, read_index(d), \
            {"DLC": d / "windows_DLC.csv", "LP": d / "windows_LP.csv"}, video
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
    tag = d.name

    # ---------------------------------------------------------------- figure 1: sign disagreements (DLC)
    T, _J, _M, clean = res["DLC"]
    tg, pos_map, _ = clean["tongue"]
    fps = tg.fps
    clip_row = np.full(len(tg.y_final), -1)
    # index row -> frame of `video`: the clip's own row, or the source frame when reading the original video
    clip_row[pos_map] = (idx.src_frame.to_numpy() if video.name.startswith("cam4_") else np.arange(len(pos_map)))
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
        # path, oriented like the image: mouth-centred image px, y down (Priya 2026-10-06)
        ax = axs[row, 0]
        ff = np.arange(on, off + 1)
        dx, dy = xa - ox, ya - oy
        ax.plot(dx[w], dy[w], "-", color="0.85", lw=1)
        sc = ax.scatter(dx[ff], dy[ff], c=np.linspace(0, 1, len(ff)), cmap="viridis", s=14, zorder=3)
        ax.plot(dx[c], dy[c], "o", ms=11, mfc="none", mec="k", mew=1.5, label="peak")
        ax.plot(dx[j_old], dy[j_old], "X", ms=10, color="red", label="max |angle| (+-52 ms)")
        ax.plot(dx[j_new], dy[j_new], "s", ms=8, mfc="none", mec="orange", mew=2,
                label=f"max |angle|, >= {frac:g} x peak dist")
        ax.plot(0, 0, "*", ms=12, mfc="yellow", mec="k")
        ax.plot([0, 240 * rx], [0, 240 * ry], "--", color="0.6", lw=0.8)     # 0 deg (toward far_center)
        ax.set_xlabel("image x from mouth (px, + = image-right)")
        ax.set_ylabel("image y from mouth (px, down)")
        ax.set_xlim(-110, 110)
        ax.set_ylim(240, -20)
        ax.set_aspect("equal")
        ax.set_title(f"trial {r.trial_id} {r.position}: lick 1 at {lk['t_ms']:.0f} ms\n"
                     f"angle at peak {r.lick1_angle_at_ypeak:+.0f}, max-signed {r.angle_max_signed_lick1:+.0f}", fontsize=8)
        if row == 0:
            ax.legend(fontsize=6, loc="lower right")
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
        a2.set_ylabel("distance from mouth (px, down = further)", color="0.4")
        a2.invert_yaxis()                     # down = further from the mouth, as in the image (Priya 2026-10-06)
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
        ph = res[m][0].lick_phase
        ph = ph.assign(angle_deg=ph.angle_deg.where(ph.protrusion_px >= MIN_PHASE_PX))
        pvar = "protrusion_filled_px" if "protrusion_filled_px" in ph else "protrusion_px"
        for row, (var, lab) in enumerate((("angle_deg", "angle (deg, + image-right)"),
                                          (pvar, "protrusion from mouth (px, down = further)"))):
            ax = axs[row, col]
            g0 = ph[ph.position == "far_L"]
            for _, gl in list(g0.groupby(["trial_id", "lick_idx"]))[:60]:
                ax.plot(gl.phase, gl[var], "-", color="0.85", lw=0.5, zorder=1)
            for pos in POS_ORDER:
                pp = ph[ph.position == pos]
                if not len(pp):
                    continue
                mu, se = tk.phase_mean(pp, var, min_coverage=MIN_COV)
                n_l = pp.groupby(["trial_id", "lick_idx"]).ngroups
                ax.plot(mu.index, mu, POS_LS[pos], color=POS_COLORS[pos], lw=2, label=f"{pos} (n={n_l})", zorder=3)
                ax.fill_between(mu.index, mu - se, mu + se, color=POS_COLORS[pos], alpha=0.2, lw=0, zorder=2)
            ax.axvline(0.5, color="k", lw=0.5, ls=":")
            ax.set_ylabel(lab)
            if row == 1:
                ax.invert_yaxis()             # down = further from the mouth, as in the image
            if row == 0:
                ax.axhline(0, color="0.6", lw=0.6)
                mode = res[m][0].params["phase"].get("mode", "extent")
                ttl = (f"centered on the peak, window {ph.cycle_ms.iloc[0]:.0f} ms (0.5 = peak)"
                       if mode == "centered" and "cycle_ms" in ph and len(ph) else "0 = tongue appears, 0.5 = peak, 1 = gone")
                ax.set_title(f"{m}: every kept lick, {ttl}", fontsize=9)
                ax.legend(fontsize=7, ncol=2)
            else:
                ax.set_xlabel("lick phase")
    fig.suptitle(f"{animal} {date}: tongue angle and protrusion over the lick (mean +- SEM per spout position; grey = "
                 f"individual far_L licks; tongue-in frames count as at the lips for protrusion; angle hidden where "
                 f"< {MIN_COV:.0%} of licks show the tongue)", fontsize=10)
    fig.tight_layout()
    out2 = d / f"tongue_angle_phase_{tag}.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"-> {out2}")

    # ---------------------------------------------------------------- figure 3: delta angle by lick outcome
    # Priya 2026-10-02: include incomplete / shorter licks, and take the tongue-spout angle wherever the tongue was,
    # contact or not. Groups: contact licks (any size) and no-contact licks split by how far they reached; points
    # inside the lip zone (< MIN_PHASE_PX from the mouth, where the tongue first appears) are masked.
    groups = [("contact", lambda g: g.contact == True, None, "-"),                                    # noqa: E712
              (f"no contact, >= {EXT_BINS[1]:.0f} px", lambda g: (g.contact == False) & (g.protrusion_max_px >= EXT_BINS[1]), "k", "--"),  # noqa: E712,E501
              (f"no contact, {EXT_BINS[0]:.0f}-{EXT_BINS[1]:.0f} px",
               lambda g: (g.contact == False) & (g.protrusion_max_px >= EXT_BINS[0]) & (g.protrusion_max_px < EXT_BINS[1]),  # noqa: E712
               "0.45", "--"),
              (f"no contact, < {EXT_BINS[0]:.0f} px", lambda g: (g.contact == False) & (g.protrusion_max_px < EXT_BINS[0]), "tab:orange", ":")]  # noqa: E712,E501
    fig, axs = plt.subplots(len(models), 6, figsize=(20, 4.4 * len(models)), squeeze=False, sharex=True)
    for row, m in enumerate(models):
        T_ = res[m][0]
        from wfield_local import lick_reference as LR
        pl_, ph_ = LR.add_deviation(T_.per_lick, T_.lick_phase,
                                    LR.build_reference([T_.per_lick], [T_.lick_phase]), "session")
        ph = ph_.merge(pl_[["trial_id", "lick_idx", "contact", "protrusion_max_px"]], on=["trial_id", "lick_idx"])
        ph = ph.assign(dev_session_deg=ph.dev_session_deg.where(ph.protrusion_px >= MIN_PHASE_PX))
        for col, pos in enumerate(POS_ORDER):
            ax = axs[row, col]
            pp = ph[ph.position == pos]
            for lab, sel, colr, ls in groups:
                sub = pp[sel(pp)]
                if not len(sub):
                    continue
                n_l = sub.groupby(["trial_id", "lick_idx"]).ngroups
                mu, se = tk.phase_mean(sub, "dev_session_deg", min_coverage=MIN_COV)
                c = colr or POS_COLORS[pos]
                ax.plot(mu.index, mu, color=c, lw=2, ls=ls, label=f"{lab} (n={n_l})")
                ax.fill_between(mu.index, mu - se, mu + se, color=c, alpha=0.12, lw=0)
            ax.axvline(0.5, color="k", lw=0.5, ls=":")
            ax.set_title(f"{m} {pos}", fontsize=9)
            ax.legend(fontsize=6)
            if col == 0:
                ax.set_ylabel("deviation from the session's\ncontact-lick path (deg)")
            if row == len(models) - 1:
                ax.set_xlabel("lick phase")
    fig.suptitle(f"{animal} {date}: tongue direction relative to the mean CONTACT-lick path at that position (same session; "
                 f"contact licks ~0 by construction), by outcome -- ALL "
                 f"licks incl. incomplete / short ones (points < {MIN_PHASE_PX:.0f} px from the mouth masked; mean +- "
                 f"SEM; + = image-right). Short licks sit near the lips, where the angle is least stable.", fontsize=10)
    fig.tight_layout()
    out3 = d / f"tongue_delta_angle_contact_{tag}.png"
    fig.savefig(out3, dpi=110)
    plt.close(fig)
    print(f"-> {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
