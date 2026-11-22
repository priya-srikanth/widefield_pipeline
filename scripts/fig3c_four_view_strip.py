"""VLS paper Figure 3C (Priya 11-22): a 1 x 4 strip of the four camera views (front = cam4, bottom = cam1, animal's
left side = cam2, animal's right side = cam3; docs/FACE_RHYTHM_PIPELINE.md camera table) at ONE pre-stroke moment of
one six-position animal, with the tracked tongue / jaw / spout keypoints marked as in the two-view panel it replaces
(grants/k08_2026/make_camframes.py: open rings + bold coloured labels with a leader line, same colours).

Everything drawn comes from the machinery of `scripts.pose3d_lick_clips` (production four-camera 3-D run, preset
``3net_20261121_sync60``): its per-session prep (``prep_session``: `pose3d_side.fuse_session` re-run for the cleaned 2-D
keypoints and the fusion's used flags, CHECKED against the production frames table and refused if different), cached
in ``pose3d_lick_clips.PREP_DIR``; this script reads that cache (or builds it the same way when absent) and the raw
video frames.

FRAME RULE (never chosen by eye): candidates = frames of the session's per-position trial clips (cue -100 .. +700 ms,
the clips' own seeded trial choice) where the tongue, jaw and spout 2-D keypoints exist AND were used by the robust
fusion in all four cameras, and the 3-D tongue is at a local maximum of its distance from the mouth (a lick peak).
They are ordered with ``SEED`` + crc32("<session>:fig3c") and the first is drawn (``--pick k`` takes the k-th); the
JSON written next to the figure lists every candidate in that order with its trial, position and time from the cue.

    conda activate dlc   # MKL_THREADING_LAYER=SEQUENTIAL, PYTHONPATH=<worktree>
    python -m scripts.fig3c_four_view_strip --session PS93:20260814 --out <folder> --stem S_cam_frames_4view_v26
    python -m scripts.fig3c_four_view_strip --pick 1 --layout v26b --out <folder> --stem S_cam_frames_4view_v26b

LAYOUT v26b (Priya 11-22: "trial 64, add nose marker, take out lines pointing to keypoints (just label with color-coded
words), and show whole face in cam 2-4 and include nose in cam1"): fixed per-camera crops (`CROPS_V26B`, source
pixels, chosen on the full frames of that moment so the whole face is inside in cam4 / cam2 / cam3 and the snout tip
in cam1), tiles of equal height and crop-proportional width, the nose ring wherever the nose keypoint is tracked (the
cam1 network has no nose output, so cam1 shows the nose but cannot mark it), and colour-coded word labels at fixed
offsets from their keypoint (`LABELS_V26B`, tile pixels) with no leader line; a thin dark outline keeps them legible on fur.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import zlib
from pathlib import Path

import numpy as np

SEED = 20261122
VIEWS = (("cam4", "front"), ("cam1", "bottom"), ("cam2", "left side"), ("cam3", "right side"))   # animal's left / right
#: the two-view panel's palette (make_camframes.COL): epoch colours stay reserved
COL = {"tongue": "#ff3b1f", "jaw": "#2b8cff", "spout": "#b45ff0", "nose": "#00e5ff"}
FIG_W, FIG_H = 6.5, 1.2
#: v26b crops (x0, y0, w, h) in source pixels for PS93 0814 trial 64 +456 ms (see LAYOUT v26b)
CROPS_V26B = {"cam4": (20, 10, 640, 650), "cam1": (110, 50, 420, 520), "cam2": (30, 0, 380, 290),
              "cam3": (220, 0, 380, 290)}
#: v26b label offsets (dx, dy, ha) from each keypoint in source pixels
LABELS_V26B = {
    "cam4": {"nose": (40, -10, "left"), "tongue": (-62, -50, "right"), "jaw": (22, 18, "left"),
             "spout": (24, 100, "left")},
    "cam1": {"tongue": (-58, 10, "right"), "jaw": (26, 20, "left"), "spout": (24, -40, "left")},
    "cam2": {"nose": (-6, -26, "left"), "tongue": (-4, 26, "right"), "jaw": (10, 24, "left"),
             "spout": (-14, -20, "right")},
    "cam3": {"nose": (6, -26, "right"), "tongue": (4, 26, "left"), "jaw": (-10, 24, "right"),
             "spout": (14, -20, "left")},
}


def load_prep(session: str, preset: str) -> dict:
    from scripts import pose3d_lick_clips as LC
    animal, date = session.split(":")
    t0, t1 = -100.0, 700.0
    for f in sorted(LC.PREP_DIR.glob(f"{preset}_{animal}_{date}_{t0:+.0f}_{t1:+.0f}_tr*.pkl")):
        P = pickle.loads(f.read_bytes())
        print(f"prep cache: {f}")
        return P
    # absent: build it exactly as pose3d_lick_clips does for its trial clips (writes the same cache file)
    os.environ.update(LC.PRESETS[preset]["env"])
    from wfield_local.paths import PathResolver
    pr = LC.PRESETS[preset]
    T = LC.load_tables(LC.DLC / "session_poses" / pr["p3d"], LC.DLC / "session_poses" / "camera_licks" / pr["licks"],
                       animal, date)
    cands = LC.trial_candidates(T, session, t0, t1)
    specs = [{"name": p, "kind": "trials", "cands": [(tr, t0, t1, r) for tr, r in cands.get(p, [])]}
             for p in LC.POS_ORDER if p in cands]
    P = LC.prep_session(PathResolver(), session, T, specs, pr["floor_ms"])
    LC.PREP_DIR.mkdir(parents=True, exist_ok=True)
    (LC.PREP_DIR / f"{preset}_{animal}_{date}_{t0:+.0f}_{t1:+.0f}_tr.pkl").write_bytes(pickle.dumps(P))
    return P


def candidates(P: dict, parts: tuple) -> list:
    out = []
    for ci, cl in enumerate(P["clips"]):
        if cl["kind"] != "trials":
            continue
        ok = np.ones(len(cl["rows"]), bool)
        for p in parts:
            ok &= np.isfinite(cl["pts"][p]).all(-1).all(0) & cl["used"][p].all(0)
        d = np.linalg.norm(cl["X"]["tongue"], axis=1)          # mm from the mouth origin
        d = np.where(np.isfinite(d), d, -np.inf)
        for k in np.flatnonzero(ok):
            if 0 < k < len(d) - 1 and d[k] >= d[k - 1] and d[k] >= d[k + 1] and np.isfinite(d[k]):
                out.append((ci, int(k)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="PS93:20260814")
    ap.add_argument("--preset", default="3net_20261121_sync60")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stem", default="S_cam_frames_4view")
    ap.add_argument("--parts", default="tongue,jaw,spout")
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--layout", default="v26", choices=("v26", "v26b"), help="see LAYOUT v26b")
    a = ap.parse_args(argv)
    parts = tuple(a.parts.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    clash = [str(a.out / f"{a.stem}.{x}") for x in ("pdf", "svg", "png", "json") if (a.out / f"{a.stem}.{x}").exists()]
    if clash:
        raise SystemExit("refusing to overwrite:\n  " + "\n  ".join(clash))

    import matplotlib
    matplotlib.use("Agg")
    import cv2
    import matplotlib.pyplot as plt

    from scripts.pose3d_lick_clips import fit_crops
    from scripts.pose3d_side_video import Reader
    P = load_prep(a.session, a.preset)
    if not P["lick_check"][0] or not all(c["same_missing"] for c in P["checks"] if c["clip"] != "3-D licks"):
        raise SystemExit("prep does not match the production table")
    cand = candidates(P, parts)
    if not cand:
        raise SystemExit("no frame with every keypoint used in all four views")
    rng = np.random.default_rng(SEED + zlib.crc32(f"{a.session}:fig3c".encode()))
    cand = [cand[i] for i in rng.permutation(len(cand))]
    ci, k = cand[a.pick]
    cl = P["clips"][ci]
    order = list(P["order"])
    crop, centre = fit_crops(P, [cl])

    if a.layout == "v26b":
        return render_v26b(a, P, cl, k, cand, parts, plt)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W, FIG_H))
    title_in, gap_in = 0.16, 0.04
    tile_h = FIG_H - title_in - 0.01
    tile_w = (FIG_W - 3 * gap_in - 0.02) / 4
    asp = tile_w / tile_h
    info = {"session": a.session, "preset": a.preset, "trial": cl["trial"], "position": cl["position"],
            "t_from_cue_ms": float(cl["t_ms"][k]), "pick": a.pick, "rule": __doc__.split("FRAME RULE")[1].split("\n\n")[0],
            "views": {}, "candidates": [{"trial": P["clips"][c]["trial"], "position": P["clips"][c]["position"],
                                         "t_from_cue_ms": float(P["clips"][c]["t_ms"][kk])} for c, kk in cand]}
    for i, (ax, (cam, title)) in enumerate(zip(axes, VIEWS)):
        j = order.index(cam)
        s_ = cl["src"][cam][k]
        im = Reader(P["videos"][cam]).get(int(s_))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        H, W = g.shape
        ch = min(crop[cam], H)
        cw = min(int(round(ch * asp)), W)
        ch = min(ch, int(round(cw / asp)))
        x0 = int(np.clip(centre[cam][0] - cw / 2, 0, W - cw))
        y0 = int(np.clip(centre[cam][1] - ch / 2, 0, H - ch))
        ax.set_position([(0.01 + i * (tile_w + gap_in)) / FIG_W, 0.01 / FIG_H, tile_w / FIG_W, tile_h / FIG_H])
        ax.imshow(g[y0:y0 + ch, x0:x0 + cw], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        xy = {p: cl["pts"][p][j, k] - (x0, y0) for p in parts}
        # labels stacked in one column on the roomier side of the keypoint cluster (no two labels overlap, none leave
        # the tile), ordered by the points' height so leader lines do not cross
        xs = np.array([xy[p][0] for p in parts]); ys = np.array([xy[p][1] for p in parts])
        sgn = -1 if xs.mean() > cw / 2 else 1
        lx = (xs.min() - 0.10 * cw) if sgn < 0 else (xs.max() + 0.10 * cw)
        step = 0.13 * ch
        slots = ys.mean() + step * (np.arange(len(parts)) - (len(parts) - 1) / 2)
        slots = np.clip(slots, 0.08 * ch, 0.92 * ch)
        for p, ty in zip([parts[i] for i in np.argsort(ys)], slots):
            x, y = xy[p]
            ax.plot(x, y, "o", ms=4.2, mfc="none", mec=COL[p], mew=1.0, zorder=3)
            ax.annotate(p, (x, y), (lx, ty), color=COL[p], fontsize=6.2, fontweight="bold",
                        ha="left" if sgn > 0 else "right", va="center", zorder=4,
                        arrowprops=dict(arrowstyle="-", color=COL[p], lw=0.6, shrinkA=0.5, shrinkB=3.0))
        ax.set_xlim(-0.5, cw - 0.5)
        ax.set_ylim(ch - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=7.0, pad=2)
        info["views"][cam] = {"title": title, "video": P["videos"][cam], "frame": int(s_), "crop_xywh": [x0, y0, cw, ch],
                              "keypoints_px": {p: [float(v) for v in cl["pts"][p][j, k]] for p in parts}}
    for x in ("pdf", "svg", "png"):
        fig.savefig(a.out / f"{a.stem}.{x}", dpi=600 if x == "png" else 300)
    plt.close(fig)
    (a.out / f"{a.stem}.json").write_text(json.dumps(info, indent=1))
    print(f"frame: {a.session} trial {cl['trial']} {cl['position']} t = {cl['t_ms'][k]:+.0f} ms from cue "
          f"(candidate {a.pick} of {len(cand)}) -> {a.out / a.stem}.*")
    return 0


def render_v26b(a, P, cl, k, cand, parts, plt) -> int:
    import cv2
    import matplotlib.patheffects as pe

    from scripts.pose3d_side_video import Reader
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    order = list(P["order"])
    draw = tuple(dict.fromkeys(("nose",) + parts))
    title_in, gap_in, pad = 0.16, 0.04, 0.01
    asp = [CROPS_V26B[c][2] / CROPS_V26B[c][3] for c, _ in VIEWS]
    tile_h = (FIG_W - 2 * pad - 3 * gap_in) / sum(asp)
    fig_h = round(tile_h + title_in + pad, 3)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    info = {"session": a.session, "preset": a.preset, "layout": "v26b", "trial": cl["trial"], "position": cl["position"],
            "t_from_cue_ms": float(cl["t_ms"][k]), "pick": a.pick, "size_in": [FIG_W, fig_h],
            "rule": __doc__.split("FRAME RULE")[1].split("\n\n")[0], "views": {},
            "candidates": [{"trial": P["clips"][c]["trial"], "position": P["clips"][c]["position"],
                            "t_from_cue_ms": float(P["clips"][c]["t_ms"][kk])} for c, kk in cand]}
    x = pad
    stroke = [pe.withStroke(linewidth=1.4, foreground="black")]
    for (cam, title), r in zip(VIEWS, asp):
        j = order.index(cam)
        s_ = cl["src"][cam][k]
        g = cv2.cvtColor(Reader(P["videos"][cam]).get(int(s_)), cv2.COLOR_BGR2GRAY)
        x0, y0, cw, ch = CROPS_V26B[cam]
        H, W = g.shape
        x0, y0 = min(x0, W - cw), min(y0, H - ch)
        ax = fig.add_axes([x / FIG_W, pad / fig_h, tile_h * r / FIG_W, tile_h / fig_h])
        x += tile_h * r + gap_in
        ax.imshow(g[y0:y0 + ch, x0:x0 + cw], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        marked = {}
        for p in draw:
            px, py = cl["pts"][p][j, k] - (x0, y0)
            if not np.isfinite(px):
                continue
            marked[p] = [float(v) for v in cl["pts"][p][j, k]]
            ax.plot(px, py, "o", ms=4.2, mfc="none", mec=COL[p], mew=1.0, zorder=3)
            dx, dy, ha = LABELS_V26B[cam].get(p, (12, 0, "left"))
            ax.text(px + dx, py + dy, p, color=COL[p], fontsize=6.2, fontweight="bold", ha=ha, va="center", zorder=4,
                    path_effects=stroke)
        ax.set_xlim(-0.5, cw - 0.5)
        ax.set_ylim(ch - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=7.0, pad=2)
        info["views"][cam] = {"title": title, "video": P["videos"][cam], "frame": int(s_), "crop_xywh": [x0, y0, cw, ch],
                              "keypoints_px": marked}
    for ext in ("pdf", "svg", "png"):
        fig.savefig(a.out / f"{a.stem}.{ext}", dpi=600 if ext == "png" else 300)
    plt.close(fig)
    (a.out / f"{a.stem}.json").write_text(json.dumps(info, indent=1))
    print(f"v26b: {a.session} trial {cl['trial']} {cl['position']} t = {cl['t_ms'][k]:+.0f} ms, {FIG_W} x {fig_h} in "
          f"-> {a.out / a.stem}.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
