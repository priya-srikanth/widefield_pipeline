"""Video check sheet for three open 3-D questions (handoff 10-20 §4a / §7; Priya 10-20 "let's do 1 and 2").

    PYTHONPATH=. python -m scripts.pose3d_check_sheet [--per 4]          # dlc env (aniposelib)

A. far_center DIAGONAL (PS92 chronic): closest-approach points on a line parallel to the rod in the side view (28-53 % of
   far_center licks per chronic session). On-line vs off-line licks at the 3-D lick peak -- is the tongue sliding along
   the rod?
B. acute far_R PRE-CUE tongue (PS92 acute): the 3-D tongue sits ~0.25 mm out before the cue (pre ~0.05). Pre-cue frames
   with a 3-D tongue point -- tongue at the lips, a real lick, or a false point?
C. ANCHORED JAW at the centre spouts (PS92 pre): frames placed by the rest-depth anchor (`pose3d.anchor_single_view`,
   src 4 = cam1 + anchor, 5 = cam4 + anchor), the one place the held-out test could not check. The anchored 3-D jaw is
   projected into BOTH cameras -- including the one that did not see it -- does it land on the chin?

Each tile: the 3-D point (frames file, PnP stage frame) projected with the session's PnP cameras (`pose3d.pnp_group`, as
the 3-D step) -- cyan ring; the camera's own 2-D LP point when it has one (likelihood >= 0.4) -- magenta dot. Video
frames from ``src1`` / ``src4`` (template-time pairing). Writes `session_poses/pose3d/check_sheet/{A,B,C}_*.png`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TILE, CROP = 300, (260, 220)


def _session(rv, s: str):
    from scripts.apparatus_mouth import _rig
    from scripts.cam1_licks import PROD
    from scripts.pose3d import _cgroup
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import spout_frame as SF
    a, d = s.split("_")
    c1 = oi.session_windows(rv, PROD["cam1"], a, d, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], a, d, "cam4")
    ix1, f1 = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
    ix4, f4 = c4[0].reset_index(drop=True), c4[1].reset_index(drop=True)
    m1, m4 = SF.ruler_medians(f1, ix1), SF.ruler_medians(f4, ix4)
    cg, _ = P3.pnp_group(_cgroup(rv).subset_cameras_names(["cam1", "cam4"]), {"cam1": m1, "cam4": m4},
                         _rig(rv, a, d)["geometry"])
    S = pd.read_csv(Path(rv.root("dlc")) / "session_poses" / "pose3d" / f"{s}_frames.csv.gz")
    return {"cg": cg, "S": S, "cams": {"cam1": (ix1, f1), "cam4": (ix4, f4)}, "a": a, "d": d}


def _tiles(rv, ses, row, part: str, label: str) -> np.ndarray:
    """cam4 | cam1 tiles for one frames-file row: ``part``'s 3-D point projected (cyan ring) + the 2-D point (magenta)."""
    import cv2

    from scripts.cam1_licks import _video
    X = np.array([[row[f"{part}_ml"], row[f"{part}_ap"], row[f"{part}_dv"]]], float)
    proj = ses["cg"].project(X)[:, 0] if np.isfinite(X).all() else np.full((2, 2), np.nan)     # (cams, 2)
    out = []
    for ci, cam in ((1, "cam4"), (0, "cam1")):
        ix, fl = ses["cams"][cam]
        src = int(row["src4" if cam == "cam4" else "src1"])
        cap = cv2.VideoCapture(str(_video(rv, ses["a"], ses["d"], cam, pd.DataFrame(), row["trial_id"])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, src)
        ok, im = cap.read()
        cap.release()
        if not ok:
            im = np.zeros((1024, 1280, 3), np.uint8)
        k = np.flatnonzero(ix.src_frame.to_numpy() == src)
        q = fl.iloc[k[0]] if len(k) else None
        p2 = (np.array([q[f"{part}_x"], q[f"{part}_y"]], float)
              if q is not None and q[f"{part}_likelihood"] >= 0.4 else None)
        pp = proj[ci]
        if np.isfinite(pp).all():
            cv2.circle(im, tuple(pp.astype(int)), 9, (255, 255, 0), 1, cv2.LINE_AA)
        if p2 is not None:
            cv2.circle(im, tuple(p2.astype(int)), 3, (255, 0, 255), -1, cv2.LINE_AA)
        c = pp if np.isfinite(pp).all() else (p2 if p2 is not None else np.array([640.0, 512.0]))
        x0, y0 = max(int(c[0]) - CROP[0] // 2, 0), max(int(c[1]) - CROP[1] // 2, 0)
        t = cv2.resize(im[y0:y0 + CROP[1], x0:x0 + CROP[0]], (TILE, int(TILE * CROP[1] / CROP[0])))
        seen = "2-D yes" if p2 is not None else "2-D none"
        cv2.putText(t, f"{cam} ({seen})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(t)
    band = np.full((20, TILE * 2, 3), 30, np.uint8)
    cv2.putText(band, label[:80], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, np.hstack(out)])


def _grid(tiles: list, ncol: int, title: str, path: Path) -> None:
    import cv2
    if not tiles:
        print(f"{path.name}: nothing to show")
        return
    h, w = tiles[0].shape[:2]
    rows = []
    for i in range(0, len(tiles), ncol):
        r = tiles[i:i + ncol] + [np.zeros((h, w, 3), np.uint8)] * (ncol - len(tiles[i:i + ncol]))
        rows.append(np.hstack(r))
    head = np.zeros((24, w * ncol, 3), np.uint8)
    cv2.putText(head, title[:170], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([head] + rows))
    print("->", path)


def section_a(rv, out: Path, per: int, rng) -> None:
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    K = pd.read_csv(d3 / "kinematics" / "licks.csv.gz", dtype={"date": str})
    g = K[(K.animal == "PS92") & (K.position == "far_center") & (K.epoch == "chronic")].copy()
    g["on_line"] = (np.abs(g.dv_err_mm + g.ap_err_mm + 1.6) < 0.15) & (g.ap_err_mm < -0.3)
    s = g.groupby("session").on_line.sum().idxmax()
    ses = _session(rv, s)
    S = ses["S"].set_index(["trial_id", "t_ms"])
    tiles = []
    gs = g[g.session == s]
    for lab, sub, n in (("ON the diagonal", gs[gs.on_line], per), ("off the diagonal", gs[~gs.on_line], max(per // 2, 2))):
        for r in sub.sample(min(n, len(sub)), random_state=int(rng.integers(1 << 30))).itertuples():
            t = 4.0 * round(r.t_peak_3d / 4.0)
            if (r.trial_id, t) not in S.index:
                continue
            row = S.loc[(r.trial_id, t)].to_dict() | {"trial_id": r.trial_id}
            tiles.append(_tiles(rv, ses, row, "tongue", f"{lab}: trial {r.trial_id} t {t:+.0f} ms  ap {r.ap_err_mm:+.2f}"
                                                         f" dv {r.dv_err_mm:+.2f} mm  contact {bool(r.contact)}"))
    _grid(tiles, 2, f"A. {s} far_center (chronic): 3-D lick peak, tongue (cyan = 3-D projected, magenta = 2-D)",
          out / f"A_far_center_diagonal_{s}.png")


def section_b(rv, out: Path, per: int, rng) -> None:
    from wfield_local import epochs
    d3 = Path(rv.root("dlc")) / "session_poses" / "pose3d"
    cands = [f.name.replace("_frames.csv.gz", "") for f in sorted(d3.glob("PS92_*_frames.csv.gz"))]
    acute = [s for s in cands if epochs.epoch_of(f"PS92_{s.split('_')[1][4:]}") == "acute"]
    tiles = []
    for s in acute[:2]:
        ses = _session(rv, s)
        S = ses["S"]
        m = (S.position == "far_R") & (S.t_ms < 0) & S.tongue_ml.notna()
        g = S[m]
        print(f"{s}: far_R pre-cue frames with a 3-D tongue {m.sum()} in {g.trial_id.nunique()} trials of "
              f"{S[S.position == 'far_R'].trial_id.nunique()}")
        pick = g.groupby("trial_id").head(1)
        for _, row in pick.sample(min(per, len(pick)), random_state=int(rng.integers(1 << 30))).iterrows():
            dist = float(np.linalg.norm([row.tongue_ml, row.tongue_ap, row.tongue_dv]))
            tiles.append(_tiles(rv, ses, row, "tongue", f"{s} trial {row.trial_id} t {row.t_ms:+.0f} ms  |tip| "
                                                         f"{dist:.2f} mm  src {int(row.tongue_src)}"))
    _grid(tiles, 2, "B. PS92 acute far_R, BEFORE the cue: frames with a 3-D tongue point", out / "B_acute_farR_precue.png")


def section_c(rv, out: Path, per: int, rng) -> None:
    s = "PS92_20260811"
    ses = _session(rv, s)
    S = ses["S"]
    tiles = []
    for code, lab in ((5, "cam4 + anchor (cam1 missed)"), (4, "cam1 + anchor (cam4 missed)")):
        g = S[S.position.isin(["far_center", "close_center"]) & (S.jaw_src == code) & (S.t_ms > 70)]
        print(f"{s}: centre-spout anchored jaw frames src {code}: {len(g)}")
        g = g.groupby("trial_id").head(1)
        for _, row in g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 30))).iterrows():
            tiles.append(_tiles(rv, ses, row, "jaw", f"{lab}: {row.position} trial {row.trial_id} t {row.t_ms:+.0f} ms"))
    _grid(tiles, 2, f"C. {s} (pre) centre spouts: ANCHORED 3-D jaw projected into both cameras (cyan); 2-D LP jaw "
                    "(magenta) where the camera had one", out / f"C_anchored_jaw_{s}.png")


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = Path(rv.root("dlc")) / "session_poses" / "pose3d" / "check_sheet"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    section_a(rv, out, a.per, rng)
    section_b(rv, out, a.per, rng)
    section_c(rv, out, a.per, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
