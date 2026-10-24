"""Reading the runner's npz files without copies, the set / output folders, plan + frozen-set loading, and the sidecar modules O2 imports from the set folder."""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------- npz access (no copies)
def npz_memmap(path: Path, name: str) -> np.memmap:
    """Memory-map array ``name`` of an UNCOMPRESSED npz (the runner writes them with np.savez)."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(f"{name}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{name} is compressed -- cannot memory-map")
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        h = fh.read(30)
        n_name, n_extra = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(fh)
        off = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")


def npz_small(path: Path, names: tuple) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in names if k in z.files}


def rows_of(spec: np.ndarray) -> np.ndarray:
    """(..., 2, P, F) -> (n, 2P, F) float32 (xy-major: row index xy * P + point)."""
    a = np.asarray(spec, dtype=np.float32)
    return a.reshape(-1, a.shape[-3] * a.shape[-2], a.shape[-1])


def _sidecar_module(name: str):
    """``wfield_local.<name>`` on the desktop; on O2 (no repo) the copy `plan` put next to ``fr_factors.py``."""
    import importlib
    try:
        return importlib.import_module(f"wfield_local.{name}")
    except ImportError:
        sys.path.insert(0, str(set_dir()))
        return importlib.import_module(name)


def set_dir() -> Path:
    """The folder holding this package. On O2 that is <bundle>/factors/<set>/ (`plan` copies the package there as
    ``fr_factors_lib/`` beside ``fr_factors.py``, plan.json and the sidecar modules); on the desktop it is
    wfield_local/face_rhythm/, which has no plan.json, so the CLI falls back to the bundle on the share."""
    return Path(__file__).resolve().parent.parent


def out_dir(sd: Path) -> Path:
    return Path(os.environ.get("FR_OUT", sd.parent.parent / "out"))


def view_out_dir(sd: Path, cam: str, bundle: str) -> Path:
    """Outputs of another camera's run: its bundle sits next to this set's bundle (``FR_OUT_<CAM>`` overrides)."""
    return Path(os.environ.get(f"FR_OUT_{cam.upper()}", sd.parent.parent.parent / bundle / "out"))


def plan_views(sd: Path) -> list[dict]:
    """[] for a single-view set; else [{"cam", "bundle"}] of the EXTRA views (the reference view is the set's bundle)."""
    return load_plan(sd).get("views", {}).get("cams", [])


def load_plan(sd: Path) -> dict:
    return json.loads((sd / "plan.json").read_text())


def load_master(sd: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """(A (2P, R), B (F, R), animal scales, the npz as a dict)."""
    M = dict(np.load(sd / "master.npz"))
    A = M["spatial"].reshape(-1, M["spatial"].shape[-1]).astype(np.float64)
    return A, M["spectral"].astype(np.float64), json.loads(str(M["scales_json"])), M


def _atomic_savez(dst: Path, **arrs) -> None:
    """Write ``dst`` atomically via a PER-PROCESS temp file (10-22: two local drivers projecting the same task shared
    one ``.part.npz`` and the slower writer failed; the result was intact, the error noise was not)."""
    dst.parent.mkdir(exist_ok=True)
    tmp = dst.with_suffix(f".{os.getpid()}.part.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, dst)
