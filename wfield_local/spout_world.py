"""The 3-D SPOUT WORLD FRAME: triangulated poses in mouth-centred millimetres, set by the apparatus.

    from wfield_local import spout_world as SW
    rig = SW.session_rig(log_dir)                                   # gui_config: mouth + geometry
    world = SW.fit(points, rig["geometry"])                         # points: pos, X (calib units), [dist_mm]
    print(world.report());  P_mm = world.to_world(P_calib)          # columns ml, ap, dv

WHY (DECISIONS 2026-10-05, "3-D world frame from the spout positions"). The six spout positions are commanded
by the stage in millimetres relative to a per-session `mouth` point (gui_config `geometry`, `mouth`), and the
stage read-back in events.csv equals the command to 0.001 mm during pre_cue. That makes them a ground truth
that does not come from the camera calibration:

  * AXES. Mouth-centred stage axes: ``ml`` = stage x (+ = mouse's right; the *_L spouts sit at -x),
    ``ap`` = -stage y (+ = out of the mouth toward the spouts, the same sign as `spout_frame`'s ap), ``dv`` =
    stage z (+ = up). Built from the apparatus, so nose / jaw are MEASURED in it, not used to define it
    (Priya, 2026-10-01: they change after the stroke).
  * SCALE CHECK / CORRECTION. With the 09-11 calibration, within-session distances between positions match
    the command to 0.98-1.01 in Aug-Sep but come out 4-9 % too LARGE in June -- the camera geometry drifted
    between June and September. A per-session similarity fit (rotation, translation, SCALE) to the commanded
    layout absorbs that, so June sessions are usable without a June calibration.
  * CONSTANT LABEL OFFSETS CANCEL. The spout moves by pure translation, so the labelled point (cam1 lower /
    cam4 upper front edge, not the commanded tip) is a fixed vector from the commanded point and goes into
    the translation.

THE STAGE AXES ARE LEFT-HANDED, SO THE FIT IS DONE IN (ml, ap, dv). With L at -x, the spouts at -y (in front of
the mouth) and z up, stage (x, y, z) is a mirror of any camera space, and a proper rotation cannot map one onto
the other: the first fit in stage axes returned scale 0.87-0.94 and 0.5-1 mm residuals although all 15 pairwise
distances matched to ~2 % (distances are mirror-blind). Fitted to (ml, ap, dv) = (x, -y, z), right-handed, the
residuals fell to 0.03-0.08 mm per session (2026-10-05, labelled spout, 09-11 calibration). The fit also
confirms the naming: *_L is the mouse's left in the right-handed camera space.

FIT PER SESSION, mouth-relative. A first pooled fit across sessions in absolute stage coordinates gave a
spurious scale of 0.83: the sessions' mouth points differ by only ~1 mm, comparable to the triangulation
noise, and that dilution shrank the fitted scale. The stage frame itself also moved ~15 mm in z between June
and August (mouth set point -81 vs -66 mm). Per session, mouth-relative, avoids both.

ADAPTIVE SESSIONS (PS93 0817 / 0820, PS92 0820): adaptive distance changed only far_L (3.5 / 3.75 mm). Only
samples at the position's NOMINAL distance define the frame (Priya, 2026-10-05: "only use trials with the
full 4 mm distance for the LR axis"); the shorter ones are kept as a CHECK -- the frame must place them at
their read-back distance (`step_check`). Per-sample distance comes from events.csv, never trials.csv, whose
position labels are wrong on ~15 % of trials (docs/GUI_TRIALS_LOGGING.md).

NOT YET: joining camera frames to events.csv needs the controller clock (``device_t_ms``) on the DAQ clock;
reuse the trial pairing in `daq_trials`, do not write a second matcher.

TRIAL IDS IN events.csv LAG ON ``trial_start`` (Priya, 2026-10-05; docs/GUI_TRIALS_LOGGING.md). The firmware emits
``trial_start`` BEFORE incrementing its trial counter, so that row carries the PREVIOUS trial's id, while the
trial's settle / pre_cue / cue / lick / reward rows carry the right one. GUI v47 (bb16533) fixed how trials.csv
is ASSEMBLED from these rows; events.csv itself keeps the lag in every session. `stage_samples` therefore reads
only ``pre_cue`` rows (correct id, own ``pos_name`` / ``pos_dist_mm`` / read-back), never ``trial_start``, and a
frame join should go by TIME, with trial ids only as a cross-check against the DAQ trials.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

POSITIONS = ["close_center", "close_L", "close_R", "far_center", "far_L", "far_R"]
_AZ_KEY = {"center": "az_center", "L": "az_left", "R": "az_right"}
FLIP = np.diag([1.0, -1.0, 1.0])     # stage (x, y, z) <-> world (ml, ap, dv); its own inverse
_PAIRS = {  # name -> (a, b): the distances checked against the command
    "ML far_L-far_R": ("far_L", "far_R"),
    "ML close_L-close_R": ("close_L", "close_R"),
    "radial centre close-far": ("close_center", "far_center"),
    "radial L close-far": ("close_L", "far_L"),
    "radial R close-far": ("close_R", "far_R"),
}


# ------------------------------------------------------------------------------- the command

def nominal_distance(geometry: dict, pos: str) -> float:
    tier = pos.split("_")[0]
    return float(geometry["dist_close"] if tier == "close" else geometry["dist_far"])


def commanded_world(geometry: dict, pos: str, r_mm: float | None = None) -> np.ndarray:
    """`commanded_offset` in world axes (ml, ap, dv): + mouse right, + out of the mouth, + up."""
    return FLIP @ commanded_offset(geometry, pos, r_mm)


def commanded_offset(geometry: dict, pos: str, r_mm: float | None = None) -> np.ndarray:
    """Mouth-relative spout position in STAGE mm, exactly as the firmware computes it.

    Port of ``recomputePosition`` (mobile_spout_behavior Teensy firmware, v39):
    ``(r cos(theta) sin(phi), -r cos(theta) cos(phi), -r sin(theta))`` then a roll about the fore-aft axis.
    ``r_mm`` overrides the nominal distance (adaptive trials).
    """
    side = pos.split("_")[1]
    r = nominal_distance(geometry, pos) if r_mm is None else float(r_mm)
    phi = np.radians(float(geometry[_AZ_KEY[side]]))
    th = np.radians(float(geometry["down_angle"]))
    roll = np.radians(float(geometry.get("head_roll", 0.0)))
    x0, y0, z0 = r * np.cos(th) * np.sin(phi), -r * np.cos(th) * np.cos(phi), -r * np.sin(th)
    return np.array([x0 * np.cos(roll) + z0 * np.sin(roll), y0, -x0 * np.sin(roll) + z0 * np.cos(roll)])


def session_rig(log_dir: str | Path) -> dict:
    """``{"mouth": (3,), "geometry": {...}, "adaptive": bool}`` from the session's gui_config.json."""
    c = json.loads((Path(log_dir) / "gui_config.json").read_text(encoding="utf-8"))
    return {"mouth": np.array(c["mouth"], float), "geometry": c["geometry"],
            "adaptive": bool((c.get("adaptive") or {}).get("enabled", False))}


def stage_samples(log_dir: str | Path, states=("pre_cue",)) -> pd.DataFrame:
    """Stage read-back per events.csv row in ``states``: device_t_ms, trial_id, pos_name, x/y/z_mm,
    pos_dist_mm and ``nominal`` (distance equals the position's nominal tier distance)."""
    cols = ["device_t_ms", "state", "trial_id", "pos_name", "x_mm", "y_mm", "z_mm", "pos_dist_mm"]
    e = pd.read_csv(Path(log_dir) / "events.csv", usecols=lambda c: c in cols, low_memory=False)
    e = e[e.state.isin(states)].copy()
    for c in ["device_t_ms", "x_mm", "y_mm", "z_mm", "pos_dist_mm"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    geom = session_rig(log_dir)["geometry"]
    nom = e.pos_name.map(lambda p: nominal_distance(geom, p) if p in POSITIONS else np.nan)
    e["nominal"] = np.isclose(e.pos_dist_mm, nom, atol=1e-3)
    return e.reset_index(drop=True)


# ----------------------------------------------------------------------------------- the fit

def umeyama(X: np.ndarray, Y: np.ndarray, scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """``s, R, t`` minimising ``|| s R X + t - Y ||`` (Umeyama 1991), proper rotation only."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / Xc.var(0).sum()) if scale else 1.0
    return s, R, my - s * R @ mx


@dataclass
class SpoutWorld:
    scale: float                  # mm per calibration unit (1.0 = calibration metric scale is right)
    R: np.ndarray                 # (3, 3) proper rotation, calibration -> world (ml, ap, dv)
    t: np.ndarray                 # (3,) mm, mouth-relative, world axes
    residual_mm: dict             # per position: |fitted - commanded|
    distances: pd.DataFrame       # _PAIRS: commanded mm, triangulated units, ratio
    n: dict                       # samples per position used
    excluded: dict = field(default_factory=dict)   # non-nominal (adaptive) samples per position

    def to_world(self, P: np.ndarray) -> np.ndarray:
        """Calibration-space points (..., 3) -> (ml, ap, dv) mm: + mouse right, + out of the mouth, + up."""
        P = np.asarray(P, float)
        return (self.scale * (self.R @ P.reshape(-1, 3).T)).T.reshape(P.shape) + self.t

    def to_stage(self, P: np.ndarray) -> np.ndarray:
        """Calibration-space points (..., 3) -> mouth-relative stage mm (x, y, z) (left-handed)."""
        return self.to_world(P) @ FLIP

    def report(self) -> str:
        res = ", ".join(f"{p} {v:.2f}" for p, v in self.residual_mm.items())
        lines = [f"scale {self.scale:.3f} stage mm / calib unit; residual mm: {res}"]
        for _, r in self.distances.iterrows():
            lines.append(f"  {r.pair:24s} commanded {r.cmd_mm:5.2f} mm  triangulated {r.tri:5.2f}  ratio {r.ratio:.3f}")
        if self.excluded:
            lines.append("  excluded (non-nominal distance): "
                         + ", ".join(f"{p} {n}" for p, n in self.excluded.items()))
        return "\n".join(lines)


def position_medians(points: pd.DataFrame, geometry: dict, min_n: int = 1) -> tuple[dict, dict, dict]:
    """Per-position median of nominal-distance samples. ``points``: columns ``pos``, ``X`` ((3,) arrays),
    optional ``dist_mm`` (per-sample stage distance; absent = all nominal)."""
    med, n, excl = {}, {}, {}
    for pos, g in points.groupby("pos"):
        if pos not in POSITIONS:
            continue
        ok = np.ones(len(g), bool)
        if "dist_mm" in g:
            ok = np.isclose(g.dist_mm.to_numpy(float), nominal_distance(geometry, pos), atol=1e-3)
        if (~ok).any():
            excl[pos] = int((~ok).sum())
        X = np.stack(g.X.to_numpy()[ok]) if ok.any() else np.empty((0, 3))
        X = X[np.isfinite(X).all(1)]
        if len(X) >= min_n:
            med[pos], n[pos] = np.median(X, axis=0), len(X)
    return med, n, excl


def distance_check(med: dict, geometry: dict) -> pd.DataFrame:
    rows = []
    for name, (a, b) in _PAIRS.items():
        if a in med and b in med:
            cmd = float(np.linalg.norm(commanded_offset(geometry, a) - commanded_offset(geometry, b)))
            tri = float(np.linalg.norm(med[a] - med[b]))
            rows.append({"pair": name, "cmd_mm": cmd, "tri": tri, "ratio": cmd / tri if tri else np.nan})
    return pd.DataFrame(rows, columns=["pair", "cmd_mm", "tri", "ratio"])


def fit(points: pd.DataFrame, geometry: dict, min_positions: int = 5, min_n: int = 1,
        scale: bool = True) -> SpoutWorld:
    """Similarity fit of one SESSION's triangulated spout positions to the commanded layout.

    ``min_positions`` 5 of 6: a similarity transform has 7 parameters and the six points are coplanar-ish
    on a cone, so fewer positions leave the axes poorly fixed; refuse rather than return a frame that
    looks fine and is not.
    """
    med, n, excl = position_medians(points, geometry, min_n)
    if len(med) < min_positions:
        raise ValueError(f"only {len(med)} spout positions with data ({sorted(med)}); need {min_positions}")
    pos = [p for p in POSITIONS if p in med]
    X = np.stack([med[p] for p in pos])
    Y = np.stack([commanded_world(geometry, p) for p in pos])       # right-handed: see module docstring
    s, R, t = umeyama(X, Y, scale)
    fitted = (s * (R @ X.T)).T + t
    res = {p: float(np.linalg.norm(fitted[i] - Y[i])) for i, p in enumerate(pos)}
    return SpoutWorld(s, R, t, res, distance_check(med, geometry), n, excl)


def step_check(world: SpoutWorld, points: pd.DataFrame, geometry: dict) -> pd.DataFrame:
    """For NON-nominal (adaptive) samples: frame-predicted distance from the mouth origin along the
    position's ray vs the stage read-back distance. Columns pos, dist_mm, fitted_mm, error_mm."""
    if "dist_mm" not in points:
        return pd.DataFrame(columns=["pos", "dist_mm", "fitted_mm", "error_mm"])
    rows = []
    for _, r in points.iterrows():
        if r.pos not in POSITIONS or np.isclose(r.dist_mm, nominal_distance(geometry, r.pos), atol=1e-3):
            continue
        u = commanded_world(geometry, r.pos, 1.0)                # unit ray from the mouth
        d = float(world.to_world(np.asarray(r.X, float)) @ u)
        rows.append({"pos": r.pos, "dist_mm": float(r.dist_mm), "fitted_mm": d, "error_mm": d - float(r.dist_mm)})
    return pd.DataFrame(rows, columns=["pos", "dist_mm", "fitted_mm", "error_mm"])
