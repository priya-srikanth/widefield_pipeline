"""Per-camera settings for the FaceRhythm chain (Priya 10-21: "prep for all 3 cameras" -- cam1, cam2, cam3 after cam4).

ONE place for what differs between the views; every FR script takes ``--cam`` and reads it from here:

  prep        `scripts/face_rhythm_prep.py frames --cam <cam>`: median rest face per session. REST FRAMES ARE ALWAYS
              CHOSEN ON cam4's LP (tongue absent, -1.5 .. -0.5 s from the cue): for another camera their DAQ times go
              through that camera's own alignment template (`camera_sync.frame_daq_seconds`) to its frame indices, so
              no pose model is needed on that view. ``rest_exclude`` drops trials whose spout rod sits in front of the
              face in this view (cam1: the centre spouts).
  register    `scripts/face_rhythm_register.py --cam <cam>`: LP landmarks exist for cam4 only (``lp_landmarks``);
              the other views start the template -> master step from CLICKED landmarks (``click_landmarks``, on the
              animal templates + master, ``register_inputs/landmarks.json``) and the session -> template step from an
              image-only ECC fit (same rig and headplate every day: small shifts).
  rois / O2   `notebooks/face_rhythm_draw_roi.ipynb` (CAM), `scripts/face_rhythm_warp_rois.py --cam`,
              `scripts/o2_face_rhythm.py bundle --cam` (trial grids from the camera's template, same DAQ trial grid).
  sides       ``face_side`` / ``ipsi_image_side``: all four lesions are LEFT (configs/animals.yaml), so the
              ipsilesional face is the animal's LEFT: cam1 / cam4 image RIGHT (memory: image-side-conventions);
              cam2 sees only the LEFT side (ipsilesional), cam3 only the RIGHT (contralesional; PS93's deficit side).

All four cameras: 250 fps, encoder keyframe every 12 frames (checked 10-21 on PS92 0818) -> the runner's 20.83 Hz notch
applies to every view. Frame sizes (H x W): cam1 600 x 600, cam2 / cam3 450 x 600, cam4 680 x 680.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CAMS = {
    "cam4": {"view": "frontal", "shape": (680, 680), "lp_landmarks": True, "click_landmarks": ("eye_l", "eye_r"),
             "rest_exclude": (), "face_side": "both", "ipsi_image_side": "right"},
    # cam1 looks UP at nose / lips / chin (whiskers both sides); the centre spout rod hangs through the middle of the
    # face, so centre trials are not used for its rest face. Side views (cam2 faces image-left, cam3 image-right) show
    # the whole head AND the body + forepaws on the treadmill: the face ROI / mask must leave out body and treadmill
    # (paws there are a possible grooming route later -- deferred, FACE_RHYTHM_PLAN).
    "cam1": {"view": "ventral", "shape": (600, 600), "lp_landmarks": False,
             "click_landmarks": ("nose", "mouth", "chin", "cheek_l", "cheek_r"),
             "rest_exclude": ("close_center", "far_center"), "face_side": "both", "ipsi_image_side": "right"},
    "cam2": {"view": "side_left", "shape": (450, 600), "lp_landmarks": False,
             "click_landmarks": ("eye", "nose", "mouth", "chin", "ear"),
             "rest_exclude": (), "face_side": "left", "ipsi_image_side": None},
    "cam3": {"view": "side_right", "shape": (450, 600), "lp_landmarks": False,
             "click_landmarks": ("eye", "nose", "mouth", "chin", "ear"),
             "rest_exclude": (), "face_side": "right", "ipsi_image_side": None},
}


def cam(name: str) -> dict:
    if name not in CAMS:
        raise ValueError(f"unknown camera {name!r}; one of {sorted(CAMS)}")
    return CAMS[name]


def root(rv, name: str) -> Path:
    """``<dlc>/face_rhythm/<cam>`` -- frames/, register/, register_inputs/, rois/, readouts ... for that view."""
    cam(name)
    return Path(rv.root("dlc")) / "face_rhythm" / name


def video_for(cam4_video: str, name: str) -> str:
    """The ``name`` camera's file of the same recording: all four cameras share the recording timestamp
    (``cam4_2026-08-18T17_23_46.avi`` -> ``cam2_2026-08-18T17_23_46.avi``)."""
    assert cam4_video.startswith("cam4_"), cam4_video
    return f"{name}_{cam4_video[5:]}"


def frame_daq_seconds(rv, animal: str, date: str, name: str, video: str) -> np.ndarray:
    """DAQ time (s) of every frame of ``video`` (camera ``name``) through that camera's alignment template(s)
    (`camera_sync.frame_daq_seconds`: per-frame timestamps, right across dropped frames). A recording with several
    DAQ-segment templates (split days): each frame takes the template whose segment holds it; NaN where none does."""
    from wfield_local import camera_sync as cs
    rec = Path(video).stem.split("_", 1)[1]
    tpls = cs.templates_for(rv, date, animal, name).get(rec, [])
    if not tpls:
        raise FileNotFoundError(f"no {name} alignment template for {animal} {date} recording {rec}")
    out = None
    for tpl in tpls:
        fd = cs.frame_daq_seconds(rv, animal, date, tpl)
        seg = tpl.get("segment_daq_sec")
        if out is None:
            out = np.full(len(fd), np.nan)
        ok = np.isnan(out) if seg is None else np.isnan(out) & (fd >= seg[0]) & (fd <= seg[1])
        out[ok] = fd[ok]
    return out


def frames_at(fd: np.ndarray, daq_s: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Frame index (fractional; ``nearest`` -> int, -1 outside) at each DAQ time, from per-frame DAQ times ``fd``
    (`frame_daq_seconds`); NaN outside the video or a segment gap."""
    ok = np.isfinite(fd)
    f_ok = np.flatnonzero(ok)
    x = np.interp(daq_s, fd[ok], f_ok.astype(float), left=np.nan, right=np.nan)
    if nearest:
        return np.where(np.isfinite(x), np.rint(np.nan_to_num(x)), -1).astype(int)
    return x


def default_root(name: str) -> Path:
    """`root` with its own PathResolver (the fr-env scripts put the repo on sys.path first)."""
    from wfield_local.paths import PathResolver
    return root(PathResolver(), name)
