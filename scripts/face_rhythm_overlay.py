"""Every session's spout tips / rest nose / rest jaw / tongue reach mapped INTO the master face, for drawing the master
ROI (Priya 10-21: "should we do it with the close spout locations projected so i exclude those?").

    <fr env python> scripts/face_rhythm_overlay.py [--root <dlc>/face_rhythm/cam4]

The registration fields (`scripts/face_rhythm_register.py`, ``register/fields/<s>.npy``) map MASTER px -> SESSION px.
A session point q goes back to the master by inverting that field numerically: the master pixel p whose R(p) is
nearest q (KD-tree on R sampled every 2 px, then the local affine of R refines it to sub-pixel). Rod path per spout
position = the cam4 rod hangs straight down from the tip (to the image bottom). Tongue reach = per session the 95th
percentile tongue distance from the rest jaw, scaled into master px by the local field scale.
Output ``register/master_overlay.npz``: positions, tip_xy (n_sessions x n_positions x 2, master px; NaN where missing),
nose_xy / jaw_xy (n_sessions x 2), reach_px (n_sessions), sessions, image (master raw); + ``master_overlay.png``.
In the ROI notebook (`notebooks/face_rhythm_draw_roi.ipynb`, SESSION = the master) plot it with:

    O = np.load(DLC / "face_rhythm" / "cam4" / "register" / "master_overlay.npz")
    for j, p in enumerate(O["positions"]):
        ax.plot(*O["tip_xy"][:, j].T, ".", ms=2, alpha=.5)          # every session's tip for spout p
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(r"M:\MICROSCOPE\Priya\DeepLabCut\Widefield\face_rhythm\cam4")


def inverse_points(field: np.ndarray, pts: np.ndarray, step: int = 2) -> np.ndarray:
    """Session px -> master px for a (master -> session) field: nearest sampled R(p), then one local-affine step."""
    from scipy.spatial import cKDTree
    H, W, _ = field.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    q = field[ys, xs].reshape(-1, 2)
    tree = cKDTree(q)
    out = np.full((len(pts), 2), np.nan)
    for i, t in enumerate(np.asarray(pts, float)):
        if not np.isfinite(t).all():
            continue
        _, k = tree.query(t)
        y, x = int(ys.flat[k]), int(xs.flat[k])
        x0, y0 = min(max(x, 1), W - 2), min(max(y, 1), H - 2)
        J = np.c_[(field[y0, x0 + 1] - field[y0, x0 - 1]) / 2, (field[y0 + 1, x0] - field[y0 - 1, x0]) / 2]
        try:
            d = np.linalg.solve(J, t - field[y0, x0])
        except np.linalg.LinAlgError:
            d = np.zeros(2)
        out[i] = [x0 + d[0], y0 + d[1]]
    return out


def main(argv=None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    reg = a.root / "register"
    master = json.loads((reg / "templates.json").read_text())["master"]
    M = np.load(reg / "master.npz")
    sessions = sorted(f.stem for f in (reg / "fields").glob("PS*.npy"))
    Z = {s: np.load(a.root / "frames" / f"{s}.npz") for s in sessions}
    positions = sorted({str(p) for z in Z.values() for p in z["tip_positions"]})
    tip = np.full((len(sessions), len(positions), 2), np.nan)
    nose, jaw, reach = (np.full((len(sessions), 2), np.nan), np.full((len(sessions), 2), np.nan),
                        np.full(len(sessions), np.nan))
    for i, s in enumerate(sessions):
        F = np.load(reg / "fields" / f"{s}.npy")
        z = Z[s]
        pts = [z["nose"], z["jaw"]] + [z["tip_xy"][list(map(str, z["tip_positions"])).index(p)]
                                      if p in list(map(str, z["tip_positions"])) else np.full(2, np.nan)
                                      for p in positions]
        back = inverse_points(F, np.asarray(pts, float))
        nose[i], jaw[i], tip[i] = back[0], back[1], back[2:]
        if np.isfinite(z["tongue_reach_px"]) and np.isfinite(jaw[i]).all():
            y, x = int(np.clip(jaw[i, 1], 1, F.shape[0] - 2)), int(np.clip(jaw[i, 0], 1, F.shape[1] - 2))
            sc = np.sqrt(abs(np.linalg.det(np.c_[(F[y, x + 1] - F[y, x - 1]) / 2, (F[y + 1, x] - F[y - 1, x]) / 2])))
            reach[i] = float(z["tongue_reach_px"]) / max(sc, 1e-6)
    np.savez_compressed(reg / "master_overlay.npz", positions=np.array(positions), tip_xy=tip, nose_xy=nose,
                        jaw_xy=jaw, reach_px=reach, sessions=np.array(sessions), image=M["image"], master=master)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(M["image"], cmap="gray")
    cols = plt.cm.tab10(np.arange(len(positions)))
    H = M["image"].shape[0]
    for j, p in enumerate(positions):
        ax.plot(*tip[:, j].T, ".", ms=3, color=cols[j], alpha=0.6, label=p)
        med = np.nanmedian(tip[:, j], 0)
        if np.isfinite(med).all():
            ax.plot([med[0], med[0]], [med[1], H - 1], "-", color=cols[j], lw=1, alpha=0.5)    # rod path
    jm = np.nanmedian(jaw, 0)
    if np.isfinite(jm).all():
        ax.add_patch(plt.Circle(jm, np.nanmedian(reach), fill=False, ls="--", color="w", lw=1))
    ax.plot(*nose.T, "+", color="c", ms=4)
    ax.plot(*jaw.T, "+", color="lime", ms=4)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"master {master}: every session's spout tips (dots), rod paths (lines), rest nose / jaw (+), "
                 "tongue reach (dashed)", fontsize=8)
    fig.tight_layout()
    fig.savefig(reg / "master_overlay.png", dpi=90)
    plt.close(fig)
    print("->", reg / "master_overlay.npz", len(sessions), "sessions,", len(positions), "positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
