"""MULTI-VIEW FaceRhythm factors: one factor set over several cameras' points (Priya 10-23: a colleague concatenated the
four videos into one 2 x 2 video with four masks; we chose to combine the views at the FACTOR stage instead).

Each camera is tracked + spectrogrammed by its OWN FaceRhythm run (`o2_face_rhythm.py bundle --cam`; separate bundles
``fr_<cam>_<date>`` side by side under ``o2/``). A multi-view factor set (`face_rhythm_factors.py plan --views ...`)
stacks the views' points along the point axis of every row: (2, P_ref + P_cam1 + ..., F). Why not a mosaic video:
the cameras have independent clocks and drop frames independently (`camera_sync`), so a mosaic needs every video
resampled onto a common clock; here every view is already on the SAME DAQ grid:

  trial rows   each camera's run re-expresses cam4's trial grid in its own frames (`o2_face_rhythm.other_cam_grid`), so
               trial i / bin j is the same DAQ window in every view: rows match by index (checked: cue times agree).
  session rows each view's 5 Hz session bins come from its own frames: a reference bin takes the view's bin NEAREST in
               DAQ time, NaN when none is within ``SESSION_TOL_S`` (then the stacked row is NaN and dropped, as any
               NaN row).

VIEW BALANCE: FR normalises every frame within its own run (VQT normaliser over that view's points), so views differ
in point count and scale. Per animal, view v is divided by d_v = RMS(view v's fit rows) x sqrt(E_v / mean E), E_v =
2 x P_v x F its elements: every view then carries the same total energy in the fit (``view_scales.json``); the
animal's one scalar (RMS of the balanced rows, ``scales.json``) follows as in a single-view set.
ABSOLUTE expression: FR's per-frame divisor D is per VIEW; a multi-view row's D is the MEAN of its views' D
(`view_normaliser`) -- provisional, documented in the set's report.
Points (dense-position PCA) stay the reference view's (cam4).
"""
from __future__ import annotations

import numpy as np

SESSION_TOL_S = 0.1                      # half a 5 Hz bin
CUE_TOL_S = 0.02                         # trial grids must agree (same DAQ cues) to this


def view_key(ref_key: str, ref_cam: str, cam: str) -> str:
    """``PS92_20260812_cam4_v2`` -> ``PS92_20260812_cam1_v2`` (same session, same video split)."""
    tag = f"_{ref_cam}"
    if tag not in ref_key:
        raise ValueError(f"{ref_key!r} has no {tag!r}")
    return ref_key.replace(tag, f"_{cam}", 1)


def session_row_map(t_ref: np.ndarray, t_view: np.ndarray, tol: float = SESSION_TOL_S) -> np.ndarray:
    """For every reference bin time, the index of the view's nearest bin (both sorted DAQ s), -1 if > ``tol`` away."""
    t_view = np.asarray(t_view, float)
    j = np.clip(np.searchsorted(t_view, t_ref), 1, max(len(t_view) - 1, 1))
    j = np.where(np.abs(t_view[j - 1] - t_ref) <= np.abs(t_view[np.minimum(j, len(t_view) - 1)] - t_ref), j - 1, j)
    j = np.minimum(j, len(t_view) - 1)
    return np.where(np.abs(t_view[j] - t_ref) <= tol, j, -1).astype(np.int64)


def check_trials_match(cues_ref: np.ndarray, cues_view: np.ndarray, what: str) -> None:
    """Trial rows are matched by index: the views' trial grids must be the same DAQ cues."""
    if len(cues_ref) != len(cues_view) or np.nanmax(np.abs(np.asarray(cues_ref) - np.asarray(cues_view))) > CUE_TOL_S:
        raise ValueError(f"{what}: trial grid differs from the reference view ({len(cues_view)} vs {len(cues_ref)} "
                         f"trials) -- rebuild that camera's bundle from the same cam4 grid")


class ViewStack:
    """Lazy stand-in for one (lead..., 2, P, F) spectrogram memmap: the views' memmaps stacked along the POINT axis,
    view v divided by ``divs[v]``. ``maps[v]``: None (rows match by index) or an int array over axis 0 (reference row
    -> view row, -1 = missing -> NaN). Supports what the factor code does with a memmap: ``shape``, ``len``, indexing
    axis 0 (int, slice, int array), and ``reshape`` merging leading axes (index-matched views only)."""

    def __init__(self, arrays: list, maps: list | None = None, divs: list | None = None):
        self.arrays = list(arrays)
        self.maps = list(maps) if maps is not None else [None] * len(arrays)
        self.divs = [float(d) for d in divs] if divs is not None else [1.0] * len(arrays)
        a0 = self.arrays[0]
        lead = a0.shape[:-3] if self.maps[0] is None else (len(self.maps[0]),)
        for a, m in zip(self.arrays, self.maps):
            if a.shape[-3] != 2 or a.shape[-1] != a0.shape[-1]:
                raise ValueError(f"view shapes {a.shape} vs {a0.shape}: need (..., 2, P, F) with the same F")
            if m is None and a.shape[:-3] != lead:
                raise ValueError(f"index-matched view has leading shape {a.shape[:-3]}, reference {lead}")
            if m is not None and len(m) != lead[0]:
                raise ValueError("a view's row map must cover every reference row")
        self.P = [a.shape[-2] for a in self.arrays]
        self.shape = (*lead, 2, sum(self.P), a0.shape[-1])
        self.dtype = np.dtype(np.float32)

    def __len__(self):
        return self.shape[0]

    def slices(self) -> list[slice]:
        o = np.concatenate([[0], np.cumsum(self.P)])
        return [slice(int(o[i]), int(o[i + 1])) for i in range(len(self.P))]

    def __getitem__(self, idx):
        parts = []
        for a, m, d in zip(self.arrays, self.maps, self.divs):
            if m is None:
                x = np.asarray(a[idx], np.float32)
            else:
                j = np.atleast_1d(m[idx])
                x = np.full((len(j), *a.shape[1:]), np.nan, np.float32)
                ok = j >= 0
                if ok.any():
                    x[ok] = np.asarray(a[j[ok]], np.float32)
                if np.ndim(m[idx]) == 0:
                    x = x[0]
            parts.append(x / d)
        return np.concatenate(parts, axis=-2)

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        if any(m is not None for m in self.maps):
            raise ValueError("reshape needs index-matched views (trial grids)")
        if tuple(shape[-3:]) != self.shape[-3:]:
            raise ValueError(f"reshape keeps (2, P, F): {shape} vs {self.shape}")
        return ViewStack([a.reshape(*shape[:-3], *a.shape[-3:]) for a in self.arrays], None, self.divs)


def view_slices(P: list[int]) -> list[slice]:
    o = np.concatenate([[0], np.cumsum(P)])
    return [slice(int(o[i]), int(o[i + 1])) for i in range(len(P))]


def balance_divisors(X: np.ndarray, P: list[int]) -> list[float]:
    """Per-view divisors d_v (see module doc) from one animal's UNWEIGHTED stacked rows ``X`` (n, 2, sum P, F)."""
    E = np.array([2 * p * X.shape[-1] for p in P], float)
    out = []
    for sl, e in zip(view_slices(P), E):
        rms = float(np.sqrt(np.nanmean(np.asarray(X[:, :, sl], np.float64) ** 2)))
        out.append(rms * float(np.sqrt(e / E.mean())))
    return out


def view_normaliser(rows: np.ndarray, P: list[int], divs: list[float], s: float, normaliser, *args) -> np.ndarray:
    """FR's per-frame divisor D of a stacked row = mean over views of each view's D, from the view's columns un-scaled
    back to that run's saved values (x s x d_v). ``rows`` (n, 2 sum P, F) xy-major; ``normaliser(rows, *args)``."""
    x = np.asarray(rows, np.float32).reshape(len(rows), 2, sum(P), rows.shape[-1])
    Ds = [normaliser((x[:, :, sl] * (s * d)).reshape(len(x), -1, x.shape[-1]), *args)
          for sl, d in zip(view_slices(P), divs)]
    return np.mean(Ds, axis=0).astype(np.float32)
