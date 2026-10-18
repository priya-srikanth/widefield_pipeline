"""First cam1 behavioural readout (Priya 2026-10-18, PRELIMINARY: the cam1 model is not chosen yet): where does each lick
point, relative to its spout, seen from below? cam1 sees left-right and up-down, which cam4 (side view) cannot.

    python -m scripts.cam1_lick_aim PS93:20260814=pre PS93:20260821=acute PS93:20260908=chronic \\
        --model "<cleaned-from pose csv with {d} / {s}>" [--ref-model "{d}/dlc_r2/*.csv"] [--tag _lp_c1r2]

Per session (cam1 trial clip, `scripts.session_poses clip --cam cam1`): the tongue is cleaned with the cam1 values
(`orofacial_clean_cam1`); frame = `spout_frame.from_medians_lr` on ``--ref-model``'s spout (the cam1 mouth, Priya 10-18),
axes ``out`` (from the mouth toward the spouts, = -ap) and ``lat`` (+ = the mouse's LEFT).

cam1 GEOMETRY (checked on frames 10-18): image up/down = dorsal/ventral. The spouts sit almost level with the mouth
(``out`` ~ +15 px) and differ only laterally (close ~ +-45 px, far ~ +-90); the tongue reaches DOWN, under the spout tip
toward the camera, 50-100 px below the mouth. So an angle of the tongue against the spout direction is meaningless on cam1
(the first version of this script used one: -50 .. -220 deg) -- only the LATERAL comparison is valid.

A LICK = a run of cleaned (measured or interpolated) tongue frames after the cue (0 .. trial stop) within one trial,
>= 3 frames, whose most VENTRAL point (the PEAK, max ``out``) is > 20 px below the mouth. Runs that never get there are
counted (``n_dorsal_runs``) but not licks: on frames they are a pale bump at the lips with the mouth nearly closed (out -30
.. -45 px; LP confident). Per lick: ``contact`` = a DAQ spout contact within the run +- 20 ms; ``lat_err_px`` = peak lateral
- spout lateral, + = more LATERAL (outward) than the spout for side positions, + = mouse's left at the centre (spout ~0);
``reach_px`` = the peak's ``out`` (ventral reach).

Writes ``session_poses/cam1_lick_aim<tag>.csv`` (one row per lick), ``..._summary.csv`` and ``....png``.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

POS_ORDER = ["far_L", "close_L", "close_center", "far_center", "close_R", "far_R"]
EPOCH_COL = {"pre": "#2b6cb0", "acute": "#d64545", "subacute": "#dd8a2b", "chronic": "#3b9c5a"}
OUT_MIN_PX, MIN_FRAMES, CONTACT_TOL_MS = 20.0, 3, 20.0


def licks_for_session(rv, session: str, epoch: str, model: str, ref_model: str, cam4_licks: str | None = None) -> pd.DataFrame:
    import scripts.pose_kinematics_demo as D
    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local import trial_windows as TW
    animal, date, _ = session_dir(rv, session)
    _, _, d = session_dir(rv, f"{session}:cam1")
    idx = read_index(d)

    def pose(pat):
        hits = glob.glob(pat.replace("{d}", str(d)).replace("{s}", f"{animal}_{date}"))
        if len(hits) != 1:
            raise SystemExit(f"{pat!r}: {len(hits)} matches")
        P = oc.read_pose(Path(hits[0]))
        if len(P) < len(idx):
            raise SystemExit(f"{hits[0]}: {len(P)} rows < {len(idx)} (still being written?)")
        return P.iloc[:len(idx)]

    ref = pose(ref_model)
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    med = SF.position_medians(ref[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans)
    fr = SF.from_medians_lr(med)
    sgn = fr.mouse_left_sign or 1

    def to_frame(x, y):
        dx, dy = np.asarray(x, float) - fr.origin[0], np.asarray(y, float) - fr.origin[1]
        return -(dx * fr.ap_axis[0] + dy * fr.ap_axis[1]), sgn * (dx * fr.lr_axis[0] + dy * fr.lr_axis[1])

    spout = {p: to_frame(*med[p][:2]) for p in med}
    spout_img = {p: (med[p][0] - fr.origin[0], med[p][1] - fr.origin[1]) for p in med}     # for plotting, image orientation
    P = pose(model)
    flat = pd.DataFrame({f"tongue_{c}": P[f"tongue_{c}"].to_numpy(float) for c in ("x", "y", "likelihood")})
    c, pos, _ = oc.clean_windows(flat, idx, "tongue", x0y0=tuple(fr.origin), cam="cam1")
    x, y, fm = c.x_final[pos] + c.X0, c.y_final[pos] + c.Y0, c.fill_method[pos]
    out_, lat = to_frame(x, y)
    b = TW.trial_bounds(animal, date, rv)
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    t, stop = idx.t_ms.to_numpy(float), idx.stop_ms.to_numpy(float)
    src, tid_a, pos_a = idx.src_frame.to_numpy(), idx.trial_id.to_numpy(), idx.position.to_numpy()
    if cam4_licks:
        return _cam4_timed(cam4_licks.replace("{animal}", animal).replace("{date}", date), idx, epoch,
                           f"{animal}_{date}", x, y, fm, out_, lat, spout, spout_img, fr)
    live = (fm != oc.FILL_BASELINE) & (t >= 0) & (t <= stop)
    rows, dorsal, i, n = [], {}, 0, len(t)
    while i < n:
        if not live[i]:
            i += 1
            continue
        j = i + 1
        while j < n and live[j] and tid_a[j] == tid_a[i] and src[j] - src[j - 1] == 1:
            j += 1
        k = i + int(np.nanargmax(out_[i:j]))
        p = pos_a[i]
        if j - i >= MIN_FRAMES and out_[k] <= OUT_MIN_PX:
            dorsal[p] = dorsal.get(p, 0) + 1
        if j - i >= MIN_FRAMES and out_[k] > OUT_MIN_PX and p in spout:
            so, sl = spout[p]
            cs = [cc for cc in contacts.get(tid_a[i], []) if t[i] - CONTACT_TOL_MS <= cc <= t[j - 1] + CONTACT_TOL_MS]
            row = {"session": f"{animal}_{date}", "epoch": epoch, "trial_id": int(tid_a[i]), "position": p,
                   "t_onset_ms": t[i], "t_peak_ms": t[k], "dur_ms": t[j - 1] - t[i], "contact": bool(cs),
                   "peak_out_px": out_[k], "peak_lat_px": lat[k], "reach_px": out_[k],
                   "peak_dx_img": x[k] - fr.origin[0], "peak_dy_img": y[k] - fr.origin[1],
                   "lat_err_px": (lat[k] - sl) * (np.sign(sl) if "center" not in p else 1.0),
                   "measured_share": float(np.mean(fm[i:j] == oc.FILL_NONE))}
            sgn_err = np.sign(sl) if "center" not in p else 1.0
            row["lat_err_mean_px"] = (float(np.mean(lat[i:j])) - sl) * sgn_err          # whole-run mean (peak-free)
            if cs:                                                                    # at the first DAQ contact
                kc = i + int(np.argmin(np.abs(t[i:j] - cs[0])))
                row["lat_err_contact_frame_px"] = (lat[kc] - sl) * sgn_err
                row["ipsi_px_contact_frame"] = lat[kc]
            rows.append(row)
        i = j
    L = pd.DataFrame(rows)
    L.attrs["spout"] = spout_img
    L.attrs["dorsal"] = dorsal
    return L


def _cam4_timed(path, idx, epoch, sess, x, y, fm, out_, lat, spout, spout_img, fr) -> pd.DataFrame:
    """cam4-TIMED peaks (Priya 10-18: the most ventral cam1 frame is not peak protrusion -- cam1 sees the tongue at an
    oblique angle). Each cam4 lick (production `licks.csv`: ``t_ms`` = cam4's protrusion peak, ms from the cue) in the clip's
    trials -> the cam1 frame of the same trial at the same cue-relative time (both clips are cut on the template-aligned DAQ
    clock, one 4 ms frame grid: exact match for 99.99 % of frames; nearest within 2 ms otherwise). The cam1 cleaned tongue
    there gives the lateral landing; ``cam1_seen`` = False when cam1's cleaned tongue is mouth fill at that frame."""
    from wfield_local import orofacial_clean as oc
    C4 = pd.read_csv(path)
    C4 = C4[C4.trial_id.isin(set(idx.trial_id))]
    key = pd.DataFrame({"trial_id": idx.trial_id.to_numpy(), "t_ms": idx.t_ms.to_numpy(float), "row": np.arange(len(idx))})
    rows = []
    for tid, g in C4.groupby("trial_id"):
        h = key[key.trial_id == tid].sort_values("t_ms")
        m = pd.merge_asof(g.sort_values("t_ms")[["t_ms", "position", "contact", "peak_angle_deg"]], h[["t_ms", "row"]],
                          on="t_ms", direction="nearest", tolerance=2.0)
        for r in m.itertuples():
            p = r.position
            if p not in spout or not np.isfinite(r.row):
                continue
            k = int(r.row)
            so, sl = spout[p]
            seen = bool(fm[k] != oc.FILL_BASELINE)
            sgn_err = np.sign(sl) if "center" not in p else 1.0
            rows.append({"session": sess, "epoch": epoch, "trial_id": int(tid), "position": p, "t_peak_ms": r.t_ms,
                         "contact": bool(r.contact) if pd.notna(r.contact) else False, "cam4_angle_deg": r.peak_angle_deg,
                         "cam1_seen": seen, "cam1_fill": int(fm[k]),
                         "peak_out_px": out_[k] if seen else np.nan, "peak_lat_px": lat[k] if seen else np.nan,
                         "peak_dx_img": x[k] - fr.origin[0] if seen else np.nan,
                         "peak_dy_img": y[k] - fr.origin[1] if seen else np.nan,
                         "reach_px": out_[k] if seen else np.nan,
                         "lat_err_px": (lat[k] - sl) * sgn_err if seen else np.nan})
    L = pd.DataFrame(rows)
    L.attrs["spout"] = spout_img
    L.attrs["dorsal"] = {}
    return L


def summarise(L: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ep, p), g in L.groupby(["epoch", "position"], sort=False):
        m = "lat_err_px"
        gc = g[g.contact]
        out.append({"epoch": ep, "position": p, "n_licks": len(g), "n_trials": g.trial_id.nunique(),
                    "licks_per_trial": round(len(g) / g.trial_id.nunique(), 1),
                    "contact_%": round(100 * g.contact.mean(), 1),
                    "lat_err_median": round(g[m].median(), 1), "lat_err_iqr": round(g[m].quantile(.75) - g[m].quantile(.25), 1),
                    "lat_err_contact": round(gc[m].median(), 1), "lat_err_no_contact": round(g[~g.contact][m].median(), 1),
                    "reach_px_median": round(g.reach_px.median(), 1),
                    **({"cam1_seen_%": round(100 * g.cam1_seen.mean(), 1),
                        "cam4_angle_contact_median": round(gc.cam4_angle_deg.median(), 1)} if "cam1_seen" in g else {})})
    S = pd.DataFrame(out)
    S["position"] = pd.Categorical(S.position, POS_ORDER)
    return S.sort_values(["position", "epoch"], key=lambda s: s if s.name == "position" else
                         s.map({e: i for i, e in enumerate(EPOCH_COL)})).reset_index(drop=True)


def figure(L: pd.DataFrame, spouts: dict, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 6, figsize=(24, 9), gridspec_kw={"height_ratios": [3, 2]})
    epochs = [e for e in EPOCH_COL if e in set(L.epoch)]
    for c, p in enumerate(POS_ORDER):
        ax, bx = axs[0, c], axs[1, c]
        for e in epochs:
            g = L[(L.position == p) & (L.epoch == e)]
            ax.scatter(g.peak_dx_img, g.peak_dy_img, s=np.where(g.contact, 14, 8), c=EPOCH_COL[e],
                       marker="o", alpha=0.45, lw=0, label=e)
            sx, sy = spouts[e].get(p, (np.nan, np.nan))
            ax.plot(sx, sy, "*", ms=16, mec="k", mfc=EPOCH_COL[e])
        ax.plot(0, 0, "k+", ms=12, mew=2)
        ax.set_aspect("equal")
        ax.invert_yaxis()                              # cam1 IMAGE orientation: down = ventral, as in the video
        ax.set_title(p)
        ax.set_xlabel("image x - mouth (px; right = mouse left)")
        if c == 0:
            ax.set_ylabel("image y - mouth (px, down = ventral)")
            ax.legend(fontsize=8, loc="lower left")
        m = "lat_err_px"
        data = [L[(L.position == p) & (L.epoch == e)][m].dropna().to_numpy() for e in epochs]
        bp = bx.boxplot(data, tick_labels=epochs, showfliers=False, widths=0.6, patch_artist=True)
        for patch, e in zip(bp["boxes"], epochs):
            patch.set_facecolor(EPOCH_COL[e])
            patch.set_alpha(0.5)
        for k, (e, v) in enumerate(zip(epochs, data)):
            bx.scatter(np.full(len(v), k + 1) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=5,
                       c=EPOCH_COL[e], alpha=0.5)
        bx.axhline(0, color="k", lw=0.8)
        bx.set_ylabel("lateral px at peak (+ mouse left)" if "center" in p else "lateral error px (+ outward of spout)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=90)


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD=EPOCH")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref-model", default="{d}/dlc_r2/*.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cam4-licks", default=None, metavar="PATH with {animal} {date}",
                    help="cam4-timed peaks: production licks.csv per session (see `_cam4_timed`)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    Ls, spouts, dors = [], {}, []
    for spec in a.sessions:
        s, ep = spec.split("=", 1)
        L = licks_for_session(rv, s, ep, a.model, a.ref_model, a.cam4_licks)
        spouts[ep] = L.attrs["spout"]
        dors += [{"epoch": ep, "position": p, "n_dorsal_runs": n} for p, n in L.attrs["dorsal"].items()]
        Ls.append(L)
    L = pd.concat(Ls, ignore_index=True)
    S = summarise(L)
    S = S.merge(pd.DataFrame(dors), on=["epoch", "position"], how="left") if dors else S
    pd.set_option("display.width", 220)
    print(S.to_string(index=False))
    base = Path(rv.root("dlc")) / "session_poses" / f"cam1_lick_aim{a.tag}"
    L.to_csv(f"{base}.csv", index=False)
    S.to_csv(f"{base}_summary.csv", index=False)
    figure(L, spouts, Path(f"{base}.png"),
           f"cam1 lick peaks relative to the spout ({', '.join(a.sessions)}; {a.tag.strip('_') or 'model'}) -- PRELIMINARY: "
           "one animal, cam1 model not chosen. cam1 image orientation; dots = lick peaks (big = DAQ contact), stars = spout, + = mouth")
    print(f"-> {base}.png / .csv / _summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
