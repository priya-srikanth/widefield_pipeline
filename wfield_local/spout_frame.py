"""The SPOUT reference frame: per session, an apparatus-defined origin and axes for tongue / jaw geometry.

    from wfield_local import spout_frame as SF
    frame = SF.from_medians(SF.position_medians(spout_xyp, spans))   # spans = SF.trial_spans(...)
    ap, lr = SF.to_frame(x, y, frame);  ang = SF.angle_deg(x, y, frame)

WHY (DECISIONS 2026-10-01, "the spout reference frame, tested"). The six spout positions lie on three lines
(L, centre, R; close and far on each) that meet at one point -- the mouth -- to within 0.1-0.8 px in every
session tested, and stay within 4 px across a session. Priya: nose and jaw can themselves change after the
stroke (facial weakness), so the frame comes from the APPARATUS, rebuilt per session, and nose / jaw are
MEASURED in it, not used to define it.

DEFINITIONS (image coordinates, y grows downward):
  origin   least-squares point nearest the three far->close lines (the "mouth").
  ap_axis  unit vector along the centre line, pointing from far_center TOWARD the origin (into the animal).
  lr_axis  ap_axis rotated so that +lr points to image-right.
  to_frame (ap, lr) of a point: ap = component along -ap_axis, i.e. OUT of the mouth toward the spouts (so a
           protruded tongue has ap > 0); lr = component along lr_axis.
  angle    atan2(lr, ap) in degrees: 0 = straight out along the centre line, + = toward image-right.
  Which image side is the mouse's left: cam4 is frontal, so the mouse's LEFT is image-RIGHT (the `close_L` /
  `far_L` spouts sit on the image right). Report angles with that in mind; `mouse_left_sign` says which sign it is.

LIMITS. cam4 is 2-D and frontal: AP (toward/away from the mouth) is mostly along the optical axis and
foreshortened, so cam4 angles are dominated by the LR component; horizontal-plane angles need cam1 (ventral)
or 3-D. The same construction applies to any camera that sees the spout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SIDES = {"L": ("close_L", "far_L"), "center": ("close_center", "far_center"), "R": ("close_R", "far_R")}
PCUT = 0.6


@dataclass
class SpoutFrame:
    origin: np.ndarray            # (2,) image px
    ap_axis: np.ndarray           # (2,) unit, far_center -> origin
    lr_axis: np.ndarray           # (2,) unit, +image-right
    positions: dict               # {position: (x, y, n_frames)}
    rms_px: float                 # how well the three lines meet
    line_dist_px: dict            # per side: distance of the origin from that line

    @property
    def mouse_left_sign(self) -> int:
        """+1 if the mouse's left (the *_L spouts) is at positive lr in this frame, else -1."""
        if "close_L" not in self.positions:
            return 0
        p = np.array(self.positions["close_L"][:2])
        return int(np.sign(np.dot(p - self.origin, self.lr_axis)))


def ls_intersection(lines) -> tuple[np.ndarray, float, list[float]]:
    """Least-squares point nearest to lines given as (point, direction); (point, rms, per-line distances)."""
    A, b, proj = np.zeros((2, 2)), np.zeros(2), []
    for p, d in lines:
        d = np.asarray(d, float) / np.linalg.norm(d)
        M = np.eye(2) - np.outer(d, d)                 # projector onto the line's normal
        A += M
        b += M @ np.asarray(p, float)
        proj.append((np.asarray(p, float), M))
    x = np.linalg.solve(A, b)
    dist = [float(np.linalg.norm(M @ (x - p))) for p, M in proj]
    return x, float(np.sqrt(np.mean(np.square(dist)))), dist


def trial_spans(trials: pd.DataFrame, strobe_s, tpl: dict, n_frames: int) -> list[tuple[int, int, str]]:
    """[(first, last_exclusive, position)] camera-frame spans from each trial's position strobe to trial end.

    Strobe = the most recent strobe at or before the cue (`daq_trials` pairing rule); trial end = the next
    trial's `trial_start_s` (the session's last frame for the final trial).
    """
    fs = float(tpl["fs_daq"])
    slope, icept = float(tpl["slope_daqSample_per_camFrame"]), float(tpl["intercept_daqSample"])
    to_f = lambda t: int(round((t * fs - icept) / slope))           # noqa: E731
    strobe_s = np.sort(np.asarray(strobe_s, float))
    t = trials.sort_values("cue_s").reset_index(drop=True)
    out = []
    for i, r in t.iterrows():
        j = np.searchsorted(strobe_s, float(r.cue_s), side="right") - 1
        if j < 0:
            continue
        end_s = float(t.trial_start_s.iloc[i + 1]) if i + 1 < len(t) else None
        f0 = max(0, to_f(strobe_s[j]))
        f1 = n_frames if end_s is None else min(n_frames, to_f(end_s))
        if f1 > f0:
            out.append((f0, f1, str(r.pos_name)))
    return out


def position_medians(spout_xyp: np.ndarray, spans, pcut: float = PCUT) -> dict:
    """{position: (x, y, n)} median confident spout tip over all of that position's spans.
    ``spout_xyp`` is (n_frames, 3) x, y, likelihood for the spout part."""
    acc: dict[str, list[np.ndarray]] = {}
    for f0, f1, pos in spans:
        seg = spout_xyp[f0:f1]
        acc.setdefault(pos, []).append(seg[seg[:, 2] > pcut, :2])
    out = {}
    for pos, arrs in acc.items():
        a = np.concatenate(arrs) if arrs else np.empty((0, 2))
        if len(a):
            out[pos] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])), int(len(a)))
    return out


def from_medians(med: dict) -> SpoutFrame:
    """The frame from per-position medians; needs close+far for at least two sides, and the centre line."""
    lines = {s: (np.array(med[c][:2]), np.array(med[c][:2]) - np.array(med[f][:2]))
             for s, (c, f) in SIDES.items() if c in med and f in med}
    if "center" not in lines or len(lines) < 2:
        raise ValueError(f"need close+far for the centre and one more side; have {sorted(med)}")
    X, rms, dist = ls_intersection(list(lines.values()))
    ap = lines["center"][1] / np.linalg.norm(lines["center"][1])
    lr = np.array([-ap[1], ap[0]])
    if lr[0] < 0:                                       # make +lr point to image-right
        lr = -lr
    return SpoutFrame(X, ap, lr, dict(med), rms, dict(zip(lines, dist)))


def to_frame(x, y, frame: SpoutFrame) -> tuple[np.ndarray, np.ndarray]:
    """(ap, lr) of image points: ap > 0 out of the mouth toward the spouts, lr > 0 toward image-right."""
    v = np.stack([np.asarray(x, float) - frame.origin[0], np.asarray(y, float) - frame.origin[1]], -1)
    return v @ (-frame.ap_axis), v @ frame.lr_axis


def angle_deg(x, y, frame: SpoutFrame) -> np.ndarray:
    """Angle of the point from the origin, relative to the centre line: 0 straight out, + toward image-right."""
    ap, lr = to_frame(x, y, frame)
    return np.degrees(np.arctan2(lr, ap))
