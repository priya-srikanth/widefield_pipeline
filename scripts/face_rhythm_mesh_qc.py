"""QC of the master mesh (12 px, 1282 points) warped to every session, BEFORE tracking:

    python scripts/face_rhythm_mesh_qc.py -> <dlc>/face_rhythm/cam4/rois_qc_12px_20261021/

contact sheets of points on each session's own median face, plus per-session distortion against the master (flipped
Delaunay triangles, local area ratio, points outside the image / the session's own face polygon). The triangle helpers
(`mesh_triangles`, `signed_area`) are shared with the AFTER-tracking QC (`o2_face_rhythm.py qc`: folds per frame)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(r"M:\MICROSCOPE\Priya\DeepLabCut\Widefield\face_rhythm\cam4")
OUT = ROOT / "rois_qc_12px_20261021"
MAX_EDGE_PX = 40                                       # drop long hull triangles of the Delaunay mesh


def mesh_triangles(home: np.ndarray, max_edge: float = MAX_EDGE_PX) -> np.ndarray:
    """Delaunay triangles (n, 3) of the home points, without the long ones along the hull."""
    from scipy.spatial import Delaunay
    tri = Delaunay(home).simplices
    return tri[np.ptp(home[tri], axis=1).max(1) < max_edge]


def signed_area(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Signed area of each triangle for points P (..., n_points, 2); a sign flip vs home = the mesh folded there."""
    a, b, c = P[..., tri[:, 0], :], P[..., tri[:, 1], :], P[..., tri[:, 2], :]
    return 0.5 * ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]))


def load(s: str):
    import h5py
    with h5py.File(ROOT / "rois" / s / "ROIs.h5", "r") as f:
        return f["point_positions"][()], f["exampleImage"][()], {k: f["roi_points"][k][()] for k in f["roi_points"]}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.path import Path as MPath
    OUT.mkdir(exist_ok=True)
    master = json.loads((ROOT / "register" / "templates.json").read_text())["master"]
    P0, _, _ = load(master)
    tri = mesh_triangles(P0)
    A0 = signed_area(P0, tri)
    sessions = sorted(p.name for p in (ROOT / "rois").iterdir() if (p / "ROIs.h5").exists())
    rows, data = [], {}
    for s in sessions:
        P, img, polys = load(s)
        if len(P) != len(P0):
            rows.append({"session": s, "n_points": len(P), "note": "point count differs"})
            continue
        r = signed_area(P, tri) / A0
        H, W = img.shape[:2]
        poly = list(polys.values())
        rows.append({"session": s, "animal": s[:4], "date": s[5:], "n_points": len(P), "flipped_tri": int((r < 0).sum()),
                     "area_ratio_p05": np.percentile(r, 5), "area_ratio_p95": np.percentile(r, 95),
                     "area_ratio_med": np.median(r),
                     "out_of_image": ((P[:, 0] < 0) | (P[:, 1] < 0) | (P[:, 0] >= W) | (P[:, 1] >= H)).mean(),
                     "in_face_poly": MPath(poly[0]).contains_points(P).mean() if poly else np.nan,
                     "shift_med_px": np.median(np.hypot(*(P - P0).T))})
        data[s] = (P, img)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "mesh_qc.csv", index=False, float_format="%.3f")
    for a in sorted({s[:4] for s in data}):
        ss = [s for s in data if s.startswith(a)]
        nc = 7
        nr = int(np.ceil(len(ss) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3.15))
        for ax in np.ravel(axs):
            ax.axis("off")
        for ax, s in zip(np.ravel(axs), ss):
            P, img = data[s]
            ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5))
            r = signed_area(P, tri) / A0
            bad = np.unique(tri[(r < 0) | (r > 2.5) | (r < 0.4)])
            ax.scatter(P[:, 0], P[:, 1], s=0.6, c="#33ff66", lw=0)
            if len(bad):
                ax.scatter(P[bad, 0], P[bad, 1], s=2, c="red", lw=0)
            row = T[T.session == s].iloc[0]
            ax.set_title(f"{s[5:]}{' (master)' if s == master else ''}  flip {row.flipped_tri}  "
                         f"area {row.area_ratio_p05:.2f}-{row.area_ratio_p95:.2f}", fontsize=7)
        fig.suptitle(f"{a}: master mesh (12 px, {len(P0)} pts) warped to each session; red = flipped / >2.5x / <0.4x "
                     f"local area vs master", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"mesh_qc_{a}.png", dpi=110)
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(T.drop(columns=[c for c in ("note",) if c in T]).to_string(index=False))
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
