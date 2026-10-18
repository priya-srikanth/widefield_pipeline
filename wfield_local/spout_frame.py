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

LIMITS. Each camera sees the spout lines foreshortened differently. Measured 2026-10-18 (PS93 0814, close->far
length per line): cam4 L 66 / centre 56 / R 68 px; **cam1 L 48 / centre 3.4 / R 48 px -- cam1 looks almost straight down
the centre line**, so it sees LR and DV, hardly AP (this docstring used to say the opposite: "horizontal-plane angles
need cam1 (ventral)"). On cam1 the ORIGIN is still the mouth (projection keeps incidence: L and R meet within 0.3 px, both
centre spouts 4-7 px from it), but ``ap_axis`` -- taken from the centre line -- is noise there; a cam1 axis must come from
the L / R lines (their bisector). True AP-plane angles need 3-D.
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


#: no-lick ENL ruler (Priya 2026-10-18): the spout tip is measured only while it is parked and untouched --
#: strobe + ENL_SETTLE_MS .. cue, raw tongue not detected, confident spout.
ENL_SETTLE_MS = 300.0
RULER_TONGUE_MAX_LK = 0.4
RULER_SPOUT_MIN_LK = 0.9


def ruler_medians(flat: pd.DataFrame, ix: pd.DataFrame) -> dict:
    """{position: (x, y, n)} median spout tip over NO-LICK ENL frames (spout parked, tongue not detected), image px."""
    t, sb = ix.t_ms.to_numpy(float), ix.strobe_ms.to_numpy(float)
    m = ((t >= sb + ENL_SETTLE_MS) & (t < 0) & (flat.tongue_likelihood.to_numpy(float) < RULER_TONGUE_MAX_LK)
         & (flat.spout_likelihood.to_numpy(float) >= RULER_SPOUT_MIN_LK))
    out = {}
    for p, g in pd.DataFrame({"x": flat.spout_x.to_numpy(float), "y": flat.spout_y.to_numpy(float),
                              "pos": ix.position.astype(str).to_numpy()})[m].groupby("pos"):
        out[p] = (float(g.x.median()), float(g.y.median()), int(len(g)))
    return out


def ruler_spacing(med: dict) -> float:
    """Mean pairwise distance (px) between the six spout positions' medians -- the session's physical ruler. Origin-
    and rotation-free; NaN unless all six positions are present."""
    import itertools
    if len(med) < 6:
        return float("nan")
    pts = [np.array(med[p][:2]) for p in sorted(med)]
    return float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(pts, 2)]))


# --------------------------------------------------------------------------- cam1 (view from below), 2026-10-18
#: cam1 looks almost straight down the CENTRE spout line (close->far 3.4 px vs L / R 48 px), so the centre line gives no
#: direction there, and the spout-label lines meet ~20-40 px toward the jaw, unstably across sessions: cam1 labels the
#: spout tip's BOTTOM edge, cam4 its top front edge (Priya 2026-10-18). Interim cam1 mouth = where the TONGUE paths meet
#: (Priya: "option 1 then option 2" -- option 2, for 3-D, is cam4's apparatus mouth projected into cam1).
TONGUE_LK, ONSET_GAP_F, ONSET_AFTER_MS = 0.4, 3, 70.0


def tongue_onsets(tongue_lk, t_ms, src_frame, trial_k, lk: float = TONGUE_LK, gap_f: int = ONSET_GAP_F,
                  after_ms: float = ONSET_AFTER_MS) -> np.ndarray:
    """Rows where the tongue BECOMES confident after >= ``gap_f`` non-confident contiguous frames of the same trial,
    later than ``after_ms`` after the cue -- each lick's first visible frame."""
    ok = np.asarray(tongue_lk, float) >= lk
    t, src, tk = (np.asarray(a) for a in (t_ms, src_frame, trial_k))
    k = np.arange(gap_f, len(ok))
    prev_none = np.ones(len(k), bool)
    for j in range(1, gap_f + 1):
        prev_none &= ~ok[k - j]
    same = (tk[k - gap_f] == tk[k]) & (src[k] - src[k - gap_f] == gap_f)
    return k[ok[k] & prev_none & same & (t[k] > after_ms)]


def tongue_path_origin(tongue_x, tongue_y, tongue_lk, positions, t_ms, src_frame, trial_k, spout_med: dict,
                       min_onsets: int = 20) -> tuple[np.ndarray, float, dict]:
    """The MOUTH as the point where the tongue's paths meet: per spout position, the line from the median lick-onset
    tongue point toward that position's median spout tip; least-squares intersection over positions with
    >= ``min_onsets`` onsets. Returns (origin, rms px, {position: (onset point, n)})."""
    on = tongue_onsets(tongue_lk, t_ms, src_frame, trial_k)
    pos = np.asarray(positions)[on]
    tx, ty = np.asarray(tongue_x, float)[on], np.asarray(tongue_y, float)[on]
    lines, per = [], {}
    for p, sp in spout_med.items():
        m = pos == p
        if m.sum() < min_onsets:
            continue
        e = np.array([np.median(tx[m]), np.median(ty[m])])
        per[p] = (e, int(m.sum()))
        lines.append((e, np.array(sp[:2]) - e))
    if len(lines) < 3:
        raise ValueError(f"tongue paths for {len(lines)} positions; need >= 3")
    X, rms, _ = ls_intersection(lines)
    return X, rms, per


def from_medians_lr_bisector(med: dict, origin) -> SpoutFrame:
    """A frame for a camera that sees the centre line END-ON (cam1): given ``origin``, ``ap_axis`` = the bisector of the
    L and R far->close directions (into the animal, symmetric by construction), +lr toward image-right."""
    dirs = []
    for side in ("L", "R"):
        c, f = SIDES[side]
        v = np.array(med[c][:2]) - np.array(med[f][:2])
        dirs.append(v / np.linalg.norm(v))
    ap = dirs[0] + dirs[1]
    ap = ap / np.linalg.norm(ap)
    lr = np.array([-ap[1], ap[0]])
    if lr[0] < 0:
        lr = -lr
    return SpoutFrame(np.asarray(origin, float), ap, lr, dict(med), float("nan"), {})
