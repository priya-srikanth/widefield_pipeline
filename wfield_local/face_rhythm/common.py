"""Cohort constants and session bookkeeping shared by every FaceRhythm readout.

Conventions (all four animals PS92-PS95 have LEFT-hemisphere lesions):
  * spout positions are named by the ANIMAL's side; the right spouts (close_R, far_R) are CONTRALESIONAL
  * on the cam4 image the ipsilesional face is IMAGE RIGHT: mesh points with x <= `MASTER_MIDLINE_X` are contralesional
  * epochs come from the shared epoch boundaries (`wfield_local.epochs`); "pre" is the August-only pre-stroke baseline
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPOCHS = ("pre", "acute", "subacute", "chronic")
POST_EPOCHS = EPOCHS[1:]
#: spout positions, the ANIMAL's left (ipsilesional) to right (contralesional) -- the one canonical order
POSITIONS = ("far_L", "close_L", "close_center", "far_center", "close_R", "far_R")
#: the same, contralesional first (headline figures read top-down from the impaired side)
POSITIONS_CONTRA_FIRST = POSITIONS[::-1]
#: spout position -> side relative to the lesion
SIDE = {"far_R": "contra", "close_R": "contra", "far_center": "centre", "close_center": "centre",
        "far_L": "ipsi", "close_L": "ipsi"}
SIDES = ("contra", "centre", "ipsi")
#: trial grid of the projections (`proj/` ``trials``: cue -2 .. +4 s, 151 bins)
FS = 25.0


def default_set_dir(rv=None, bundle: str | None = None, set_name: str | None = None) -> Path:
    """The frozen factor set's folder on the share (default: the cohort master, `face_rhythm_outputs.SET`)."""
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.paths import PathResolver
    return fro.factor_dir(rv or PathResolver(), bundle or fro.BUNDLE, set_name or fro.SET)


def readouts_dir(set_name: str | None = None, rv=None) -> Path:
    """Where FR readout figures land: ``<dlc>/face_rhythm/cam4/readouts/<set>/``."""
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.paths import PathResolver
    return Path((rv or PathResolver()).root("dlc")) / "face_rhythm" / "cam4" / "readouts" / (set_name or fro.SET)


def factor_names(fdir: Path) -> list[str]:
    """The set's factor names in factor order (labels.json, without the ``fr_`` prefix)."""
    from wfield_local import face_rhythm_outputs as fro
    return [n.removeprefix("fr_") for n in fro.labels(fdir)]


def label(animal: str, date: str) -> str:
    """Session label ``PS92_0806`` (a split video, e.g. PS92_0812 v1 / v2, shares one label: one session)."""
    return f"{animal}_{str(date)[4:]}"


def day(label_: str):
    """Days since stroke of a session label (negative / pre for pre-stroke)."""
    from wfield_local import epochs
    return epochs.days_since_stroke(label_)


def sessions(fdir: Path, sub: str = "proj") -> pd.DataFrame:
    """One row per task file in ``<set>/<sub>/`` (proj, own, own8, proj_loso): key, file, animal, date, epoch, label.
    Sorted by file name, so every consumer iterates in the same order (CLAUDE.md rule 9)."""
    rows = []
    for f in sorted((fdir / sub).glob("*.npz")) if (fdir / sub).exists() else []:
        if f.name.endswith(".part.npz"):                 # an in-flight atomic write
            continue
        with np.load(f) as z:
            rows.append({"key": f.stem, "file": f, "animal": str(z["animal"]), "date": str(z["date"]),
                         "epoch": str(z["epoch"])})
    T = pd.DataFrame(rows, columns=["key", "file", "animal", "date", "epoch"])
    T["label"] = [label(a, d) for a, d in zip(T.animal, T.date)]
    return T


def session_keys(fdir: Path, sub: str = "proj") -> list[str]:
    """Sorted task keys of ``<set>/<sub>/`` without opening the files (for `analysis_kit.fan_sessions`)."""
    return sorted(f.stem for f in (fdir / sub).glob("*.npz") if not f.name.endswith(".part.npz"))


def position_groups(pos: np.ndarray, split: str) -> dict[str, np.ndarray]:
    """Trial masks by spout: ``none`` -> {"all"}, ``side`` -> contra / centre / ipsi, ``position`` -> each spout
    (close and far contralesional spouts are different behaviours, Priya 10-24)."""
    if split == "none":
        return {"all": np.ones(len(pos), bool)}
    if split == "side":
        return {s: np.array([SIDE.get(p) == s for p in pos]) for s in SIDES}
    if split == "position":
        return {p: pos == p for p in POSITIONS_CONTRA_FIRST}
    raise ValueError(f"split must be none / side / position, not {split!r}")


def animal_colors() -> dict[str, str]:
    """The repo's one animal palette (configs/animals.yaml)."""
    from wfield_local import config
    return config.animal_color()


def epoch_colors() -> dict[str, str]:
    """Epoch colours of every FR figure: the orofacial / 3-D kinematics line palette (`orofacial_cohort.EPOCH_LINE`:
    pre black, acute pink, subacute goldenrod, chronic teal). Priya 10-24: easier to tell apart than the grey ramp
    (`epoch_figures.EPOCH_GREY`, used by FR until then), especially overlaid traces (FINDINGS 5e / 5f). Animals keep
    `animal_colors()`."""
    from wfield_local.orofacial_cohort import EPOCH_LINE
    return dict(EPOCH_LINE)
