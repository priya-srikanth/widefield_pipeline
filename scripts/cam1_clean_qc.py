"""QC figures for tuning cam1 clean traces (`docs/POSE_MODELS.md` D2; Priya 2026-10-18: "build tongue/jaw clean traces for
cam1"). Model-INDEPENDENT geometry first, settled on any decent model (DLC c1r2 for now); model-specific rules wait for the
chosen cam1 model. Each figure is for Priya's approval before a value goes into `orofacial_clean_cam1`.

    python -m scripts.cam1_clean_qc mouth PS93:20260814 --model "{d}/dlc_r2/*.csv"

``mouth``: is the cam1 spout-line intersection the MOUTH? One no-lick ENL frame per spout position (spout parked, tongue not
detected; the frame nearest that position's median spout tip), with all six position medians, the three close->far lines
extended, and their least-squares intersection (`spout_frame`, as on cam4). Also prints how well the lines meet (rms) and
each line's distance from the point. Writes ``<cam1 clip folder>/eval/qc_mouth_origin.png``.

``jaw``: which way and how far does the cam1 jaw move when the mouth opens, and how often does the jaw point land ON the
tongue (Priya 10-18, labelled clip: "sometimes jaw ends up tracking tongue")? Rest = each trial's median confident jaw over
cue -500 .. 0 ms (the v3.4 baseline window). Per frame after the cue: jaw displacement from that trial's rest, split by
tongue state (out = confident tongue > 30 px from the mouth origin). Riding candidates = jaw within 25 px of a confident
out tongue (cam4's `jaw_off_tongue` test). Panels: displacement clouds by state, |dy| / |dx| percentiles, jaw-tongue distance
histogram, and 8 example riding frames. Writes ``<cam1 clip folder>/eval/qc_jaw.png``.

``gaps`` (several sessions): which tongue gaps may be interpolated on cam1, judged by cam4 (`cmd_gaps`).

    python -m scripts.cam1_clean_qc gaps PS93:20260814 PS93:20260821 PS93:20260908 --model "<path with {s}>" --tag _lp
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

POS_ORDER = ["far_L", "close_L", "close_center", "far_center", "close_R", "far_R"]
SIDE_COL = {"L": (0.95, 0.45, 0.1), "center": (0.2, 0.7, 0.95), "R": (0.75, 0.3, 0.9)}


def cmd_mouth(session: str, model: str) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date, d = session_dir(rv, f"{session}:cam1")
    idx = read_index(d)
    hits = glob.glob(model.replace("{d}", str(d)))
    if len(hits) != 1:
        raise SystemExit(f"{model!r}: {len(hits)} matches")
    P = oc.read_pose(Path(hits[0])).iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    fr = SF.from_medians(SF.position_medians(P[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    print(f"{animal} {date} cam1: origin ({fr.origin[0]:.1f}, {fr.origin[1]:.1f}), lines meet rms {fr.rms_px:.2f} px, "
          f"distance per line {', '.join(f'{k} {v:.2f}' for k, v in fr.line_dist_px.items())} px")
    # one parked, no-lick frame per position: ENL (strobe + 300 ms .. cue), tongue not detected, spout confident, nearest the median
    t, sb = idx.t_ms.to_numpy(float), idx.strobe_ms.to_numpy(float)
    enl = (t >= sb + 300) & (t < 0) & (P.tongue_likelihood.to_numpy() < 0.1) & (P.spout_likelihood.to_numpy() >= 0.9)
    cap = cv2.VideoCapture(str(d / "windows.mp4"))                    # READ-ONLY
    fig, axs = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, pos in zip(axs.ravel(), POS_ORDER):
        m = enl & (idx.position.to_numpy() == pos)
        if not m.any() or pos not in fr.positions:
            ax.set_title(f"{pos}: no parked frame"); ax.axis("off"); continue
        px, py = fr.positions[pos][:2]
        rows = np.flatnonzero(m)
        r = rows[np.argmin(np.hypot(P.spout_x.to_numpy()[rows] - px, P.spout_y.to_numpy()[rows] - py))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r))
        ok, im = cap.read()
        ax.imshow(im[:, :, ::-1])
        for side, (c, f) in SF.SIDES.items():
            if c in fr.positions and f in fr.positions:
                a, b = np.array(fr.positions[f][:2]), np.array(fr.origin)
                ext = a + 1.25 * (b - a)
                ax.plot([a[0], ext[0]], [a[1], ext[1]], "-", color=SIDE_COL[side], lw=1.2, alpha=0.9)
        for q, (x, y, _) in fr.positions.items():
            ax.plot(x, y, "o", ms=6, mfc="none", mec="w" if q != pos else "yellow", mew=1.5)
        ax.plot(*fr.origin, "x", color="red", ms=14, mew=2.5)
        jx, jy, jl = (P[f"jaw_{c}"].to_numpy() for c in ("x", "y", "likelihood"))
        ax.plot(jx[r], jy[r], "+", color="lime", ms=12, mew=2) if jl[r] >= 0.6 else None
        ax.set_title(f"{pos} (trial {int(idx.trial_id[r])}, {t[r]:+.0f} ms from cue)", fontsize=10)
        ax.set_xlim(150, 470); ax.set_ylim(560, 240); ax.axis("off")
    cap.release()
    fig.suptitle(f"{animal} {date} cam1 -- is the spout-line intersection (red X) the MOUTH?  Lines: far -> close per side "
                 f"(orange L, blue centre, purple R), extended; rings = the six median spout tips (yellow = this panel's); "
                 f"green + = jaw (p >= 0.6)\nlines meet to rms {fr.rms_px:.2f} px; origin ({fr.origin[0]:.0f}, {fr.origin[1]:.0f})",
                 fontsize=11)
    fig.tight_layout()
    out = d / "eval" / "qc_mouth_origin.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"-> {out}")


def cmd_jaw(session: str, model: str) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date, d = session_dir(rv, f"{session}:cam1")
    idx = read_index(d)
    hits = glob.glob(model.replace("{d}", str(d)))
    P = oc.read_pose(Path(hits[0])).iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    mouth = SF.from_medians(SF.position_medians(P[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans)).origin
    jx, jy, jl = (P[f"jaw_{c}"].to_numpy(float) for c in ("x", "y", "likelihood"))
    tx, ty, tl = (P[f"tongue_{c}"].to_numpy(float) for c in ("x", "y", "likelihood"))
    t = idx.t_ms.to_numpy(float)
    jok = jl >= 0.6
    tout = (tl >= 0.4) & (np.hypot(tx - mouth[0], ty - mouth[1]) > 30)
    rx, ry = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    for _, g in idx.groupby("trial_k"):
        r = g.index.to_numpy()
        pre = r[(t[r] >= -500) & (t[r] < 0) & jok[r]]
        if len(pre) >= 5:
            rx[r], ry[r] = np.median(jx[pre]), np.median(jy[pre])
    post = (t >= 70) & jok & np.isfinite(rx)
    dx, dy = jx - rx, jy - ry
    dist = np.hypot(jx - tx, jy - ty)
    ride = post & tout & (dist < 25)
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(2, 3, 1)
    for m, col, lab in ((post & ~tout & (tl < 0.4), "0.6", "tongue not out"), (post & tout & ~ride, "tab:red", "tongue out"),
                        (ride, "tab:green", "jaw within 25 px of tongue")):
        ax.plot(dx[m], dy[m], ".", ms=1, alpha=0.3, color=col, label=f"{lab} (n={int(m.sum())})")
    ax.invert_yaxis(); ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5); ax.set_aspect("equal")
    ax.set_xlabel("jaw dx from trial rest (px, + image right)"); ax.set_ylabel("jaw dy (px, + image DOWN)")
    ax.legend(markerscale=10, fontsize=8); ax.set_title("jaw displacement after the cue")
    ax = fig.add_subplot(2, 3, 2)
    q = [50, 90, 95, 99, 99.9]
    for m, lab, col in ((post & ~tout, "tongue not out", "0.4"), (post & tout & ~ride, "tongue out, not riding", "tab:red")):
        ax.plot(q, np.percentile(dy[m], q), "o-", color=col, label=f"dy {lab}")
        ax.plot(q, np.percentile(np.abs(dx[m]), q), "s--", color=col, label=f"|dx| {lab}")
    ax.set_xlabel("percentile"); ax.set_ylabel("px"); ax.legend(fontsize=8); ax.set_title("how far the jaw moves")
    ax = fig.add_subplot(2, 3, 3)
    ax.hist(dist[post & tout], bins=np.arange(0, 200, 4), color="tab:red", alpha=.7)
    ax.axvline(25, color="k", ls="--"); ax.set_xlabel("jaw - tongue distance, tongue out (px)")
    ax.set_title(f"riding candidates (< 25 px): {100 * ride.sum() / max((post & tout).sum(), 1):.1f} % of tongue-out frames")
    rows = np.flatnonzero(ride)
    ex = rows[np.linspace(0, len(rows) - 1, min(8, len(rows))).astype(int)] if len(rows) else []
    cap = cv2.VideoCapture(str(d / "windows.mp4"))                    # READ-ONLY
    for k, r in enumerate(ex):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r)); ok, im = cap.read()
        ax = fig.add_axes([0.02 + 0.12 * k, 0.03, 0.115, 0.4])
        ax.imshow(im[230:560, 160:470, ::-1], extent=(160, 470, 560, 230))
        ax.plot(tx[r], ty[r], "o", mfc="none", mec="red", ms=8, mew=1.5)
        ax.plot(jx[r], jy[r], "+", color="lime", ms=10, mew=2)
        ax.plot(rx[r], ry[r], "x", color="cyan", ms=8, mew=1.5)
        ax.set_title(f"{idx.position[r]} {t[r]:+.0f} ms", fontsize=7); ax.axis("off")
    cap.release()
    print(f"{animal} {date}: jaw rest-relative after cue, tongue NOT out: dy p50/p99 {np.percentile(dy[post & ~tout], 50):.1f}/"
          f"{np.percentile(dy[post & ~tout], 99):.1f}, |dx| p99 {np.percentile(np.abs(dx[post & ~tout]), 99):.1f} | tongue OUT (not riding): "
          f"dy p50/p95/p99 {np.percentile(dy[post & tout & ~ride], 50):.1f}/{np.percentile(dy[post & tout & ~ride], 95):.1f}/"
          f"{np.percentile(dy[post & tout & ~ride], 99):.1f}, dy<0 share {100 * np.mean(dy[post & tout & ~ride] < 0):.0f} %, "
          f"|dx| p99 {np.percentile(np.abs(dx[post & tout & ~ride]), 99):.1f} | riding {int(ride.sum())} frames = "
          f"{100 * ride.sum() / max((post & tout).sum(), 1):.1f} % of tongue-out")
    fig.suptitle(f"{animal} {date} cam1 jaw ({Path(hits[0]).parent.name}): bottom = example 'jaw within 25 px of tongue' frames "
                 f"(red o tongue, green + jaw, cyan x trial rest)", fontsize=11)
    out = d / "eval" / "qc_jaw.png"
    fig.savefig(out, dpi=100)
    print(f"-> {out}")


def cmd_tongue_mouth(session: str, model: str, tag: str = "") -> None:
    """The cam1 MOUTH as where the tongue paths meet (`spout_frame.tongue_path_origin`, Priya 10-18 option 1), drawn on real
    lick-onset frames: per spout position, the onset frame nearest that position's median onset point (row 1) and the same
    lick 16 ms later (row 2). Red X = tongue-path mouth; white x = spout-line intersection (for contrast); orange dot + line =
    the position's median onset point and its path toward the spout tip; red circle = this frame's tongue point.
    Writes ``<cam1 clip folder>/eval/qc_tongue_mouth<tag>.png``."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date, d = session_dir(rv, f"{session}:cam1")
    idx = read_index(d)
    pat = model.replace("{d}", str(d)).replace("{s}", f"{animal}_{date}")
    hits = glob.glob(pat)
    if len(hits) != 1:
        raise SystemExit(f"{pat!r}: {len(hits)} matches")
    P = oc.read_pose(Path(hits[0])).iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    med = SF.position_medians(P[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans)
    sl = SF.from_medians(med).origin
    tx, ty, tl = (P[f"tongue_{c}"].to_numpy(float) for c in ("x", "y", "likelihood"))
    X, rms, per = SF.tongue_path_origin(tx, ty, tl, idx.position, idx.t_ms, idx.src_frame, idx.trial_k, med)
    on = SF.tongue_onsets(tl, idx.t_ms, idx.src_frame, idx.trial_k)
    pos_on = idx.position.to_numpy()[on]
    cap = cv2.VideoCapture(str(d / "windows.mp4"))                    # READ-ONLY
    fig, axs = plt.subplots(2, 6, figsize=(24, 9.5))
    for c, pos in enumerate(POS_ORDER):
        if pos not in per:
            for r in range(2):
                axs[r, c].axis("off")
            continue
        e, n = per[pos]
        cand = on[pos_on == pos]
        k = cand[np.argmin(np.hypot(tx[cand] - e[0], ty[cand] - e[1]))]
        for r, row in enumerate((k, k + 4)):
            ax = axs[r, c]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row))
            ok, im = cap.read()
            ax.imshow(im[260:500, 200:430, ::-1], extent=(200, 430, 500, 260))
            sp = np.array(med[pos][:2])
            ax.plot([e[0], sp[0]], [e[1], sp[1]], "-", color="orange", lw=1.3)
            ax.plot(*e, "o", color="orange", ms=5)
            ax.plot(*sl, "x", color="white", ms=9, mew=1.5)
            ax.plot(*X, "x", color="red", ms=13, mew=2.5)
            if tl[row] >= 0.4:
                ax.plot(tx[row], ty[row], "o", mfc="none", mec="red", ms=10, mew=1.5)
            ax.set_title(f"{pos} {'onset' if r == 0 else '+16 ms'} ({idx.t_ms[row]:+.0f} ms, n onsets {n})", fontsize=9)
            ax.axis("off")
    cap.release()
    fig.suptitle(f"{animal} {date} cam1, {Path(hits[0]).parent.parent.name if 'video_preds' in hits[0] else Path(hits[0]).parent.name}: "
                 f"red X = MOUTH where the tongue paths meet ({X[0]:.0f}, {X[1]:.0f}; paths meet to {rms:.1f} px rms); white x = "
                 f"spout-line intersection ({sl[0]:.0f}, {sl[1]:.0f}); orange = each position's median onset point + path to its "
                 "spout; red o = this frame's tongue", fontsize=11)
    fig.tight_layout()
    out = d / "eval" / f"qc_tongue_mouth{tag}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"-> {out}")


GAP_MAX_MS = (12, 16, 20, 24, 28, 32, 40, 56)


def _runs(mask: np.ndarray):
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        yield i, j
        i = j


def gap_table(session: str, model: str, ref_model: str):
    """Every tongue gap the cam1 cleaning would interpolate at cam4's 56 ms with the between-lick veto OFF (post-cue,
    within one trial), with: its length, whether the configured veto blanks it, and cam4's verdict -- the share of the
    gap's frames where the cam4 DLC r3 tongue is NOT confident (< 0.4; same trial, nearest cue-relative time within
    2.5 ms). cam4 sees licks side-on, so a gap cam4 calls mostly tongue-absent was the tongue back in the mouth: a
    bridge. Mouth = `spout_frame.from_medians_lr` on ``ref_model``'s spout (as `cleaned_label_clips --cam cam1`)."""
    import pandas as pd

    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date, c4 = session_dir(rv, session)
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
    mouth = tuple(SF.from_medians_lr(SF.position_medians(ref[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(),
                                                         spans)).origin)
    i4 = read_index(c4)
    P4 = oc.read_pose(c4 / "windows_DLC.csv").iloc[:len(i4)]
    i4 = i4.iloc[:len(P4)].assign(lk4=P4.tongue_likelihood.to_numpy(float))
    lk4 = np.full(len(idx), np.nan)
    for tid, g in idx.groupby("trial_id"):
        h = i4[i4.trial_id == tid].sort_values("t_ms")
        if len(h):
            m = pd.merge_asof(g[["t_ms"]].reset_index().sort_values("t_ms"), h[["t_ms", "lk4"]], on="t_ms",
                              direction="nearest", tolerance=2.5)
            lk4[m["index"].to_numpy()] = m.lk4.to_numpy()
    P = pose(model)
    flat = pd.DataFrame({f"tongue_{c}": P[f"tongue_{c}"].to_numpy(float) for c in ("x", "y", "likelihood")})
    p_on = oc.params("tongue", "cam1")
    p_off = {**p_on, "interp_veto_interlick": False, "max_gap_ms": 56.0}
    con, pos, _ = oc.clean_windows(flat, idx, "tongue", x0y0=mouth, cam="cam1", p={**p_on, "max_gap_ms": 56.0})
    coff, _, _ = oc.clean_windows(flat, idx, "tongue", x0y0=mouth, cam="cam1", p=p_off)
    fon, foff = con.fill_method[pos], coff.fill_method[pos]
    t, stop, tk = idx.t_ms.to_numpy(float), idx.stop_ms.to_numpy(float), idx.trial_k.to_numpy()
    post = (t >= 0) & (t <= stop)
    rows = []
    for i, j in _runs(np.isin(foff, (oc.FILL_PCHIP, oc.FILL_LINEAR))):
        if post[i] and tk[i] == tk[j - 1]:
            rows.append({"session": f"{animal}_{date}", "trial_id": int(idx.trial_id[i]), "position": idx.position[i],
                         "t_ms": t[i], "n": j - i, "ms": 1000 * (j - i) / 250.0,
                         "vetoed": bool((fon[i:j] == oc.FILL_BASELINE).all()),
                         "cam4_absent_share": float(np.mean(~(lk4[i:j] >= 0.4)))})
    return pd.DataFrame(rows)


def cmd_gaps(sessions: list[str], model: str, ref_model: str, tag: str = "") -> None:
    """How long may a cam1 tongue gap be before interpolation bridges two licks? (Priya 10-18, cleaned clip trial 251.)
    Per candidate max gap (with / without the between-lick veto): BRIDGED frames = filled although cam4 says the tongue
    was in the mouth (>= half the gap); LOST frames = left blank (-> mouth fill) although cam4 saw the tongue out.
    Writes ``session_poses/cam1_gap_check<tag>.csv`` (one row per gap) and prints the threshold table per session."""
    import pandas as pd

    from wfield_local.paths import PathResolver
    G = pd.concat([gap_table(s, model, ref_model) for s in sessions], ignore_index=True)
    G["tongue_in"] = G.cam4_absent_share >= 0.5
    G["centre"] = G.position.str.contains("center")
    out = []
    for s, g in G.groupby("session"):
        for X in GAP_MAX_MS:
            for veto in (True, False):
                fill = (g.ms <= X) & ~(g.vetoed & veto)
                br, lost = fill & g.tongue_in, ~fill & ~g.tongue_in
                out.append({"session": s, "max_gap_ms": X, "veto": veto, "bridged_frames": int(g.n[br].sum()),
                            "lost_frames": int(g.n[lost].sum()), "total": int(g.n[br].sum() + g.n[lost].sum()),
                            "centre_bridged": int(g.n[br & g.centre].sum())})
    T = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(G.assign(len_ms=pd.cut(G.ms, [0, 12, 24, 40, 56])).groupby(["centre", "len_ms"], observed=True)
          .tongue_in.agg(["size", "mean"]).round(2).rename(columns={"mean": "share_tongue_in_by_cam4"}).to_string())
    print(T.pivot_table(index=["max_gap_ms", "veto"], columns="session", values="total").to_string())
    out_csv = Path(PathResolver().root("dlc")) / "session_poses" / f"cam1_gap_check{tag}.csv"
    G.to_csv(out_csv, index=False)
    print(f"-> {out_csv}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mouth", "jaw", "tongue_mouth", "gaps"])
    ap.add_argument("session", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--model", required=True, metavar="PATH (with {d} / {s})")
    ap.add_argument("--ref-model", default="{d}/dlc_r2/*.csv", help="gaps: model whose spout sets the mouth")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    s = a.session[0]
    {"mouth": lambda: cmd_mouth(s, a.model), "jaw": lambda: cmd_jaw(s, a.model),
     "tongue_mouth": lambda: cmd_tongue_mouth(s, a.model, a.tag),
     "gaps": lambda: cmd_gaps(a.session, a.model, a.ref_model, a.tag)}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
