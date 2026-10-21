"""Binary sidecar for a session_poses folder's trial windows (Priya 10-21: "let's use saved arrays, caches ... the things
that make this most efficient and less redundant").

`o2_inference collect` writes ``windows_index.csv`` (~220 MB) + ``windows_LP.csv`` (~240 MB) per cam1 session, and a
dozen scripts re-parsed both on every run (~17 s a session). This keeps a ``windows.npz`` next to them -- built ONCE from
the CSVs (which stay untouched, the original output) and rebuilt automatically when either CSV is newer -- and returns
the SAME pandas objects the CSV reads did:

    ix = pose_windows.index(d)            # == pd.read_csv(d / "windows_index.csv")  [usecols= supported]
    P = pose_windows.lp(d)                # == pd.read_csv(d / "windows_LP.csv", header=[1, 2], index_col=0)

Precision: LP coordinates / likelihoods are stored float32 -- the model's own precision; the CSV is a decimal print of
those float32 values, so a CSV parse and this load differ by < 1e-5 px. Times (t_ms, strobe_ms, stop_ms, daq_s,
t_daq_ms) stay float64 (DAQ seconds need it). Uncompressed npz, ~175 MB a session; `np.load` reads it in ~1 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIDECAR = "windows.npz"
VERSION = 1
_STR = ("position", "stop_source")


def _stale(d: Path) -> bool:
    f = d / SIDECAR
    if not f.exists():
        return True
    t = f.stat().st_mtime
    return any((d / c).exists() and (d / c).stat().st_mtime > t for c in ("windows_index.csv", "windows_LP.csv"))


def build(d: Path) -> Path:
    """(Re)write ``windows.npz`` from the folder's CSVs. Atomic: written to a temp name, then renamed."""
    d = Path(d)
    ix = pd.read_csv(d / "windows_index.csv")
    out = {"version": np.int64(VERSION), "ix_columns": np.array(list(ix.columns))}
    for c in ix.columns:
        v = ix[c]
        if c in _STR or v.dtype == object:
            cats, codes = np.unique(v.astype(str).to_numpy(dtype=str), return_inverse=True)
            out[f"ix_{c}__cats"], out[f"ix_{c}__codes"] = cats.astype(str), codes.astype(np.int16)
        else:
            out[f"ix_{c}"] = v.to_numpy()
    if (d / "windows_LP.csv").exists():
        P = pd.read_csv(d / "windows_LP.csv", header=[1, 2], index_col=0)
        out["lp"] = P.to_numpy(np.float32)
        out["lp_parts"] = np.array([a for a, _ in P.columns])
        out["lp_coords"] = np.array([b for _, b in P.columns])
        out["lp_index"] = P.index.to_numpy()
    tmp = d / (SIDECAR + ".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(d / SIDECAR)
    return d / SIDECAR


def _z(d: Path):
    d = Path(d)
    if _stale(d):
        build(d)
    return np.load(d / SIDECAR, allow_pickle=False)


def index(d: Path, usecols=None) -> pd.DataFrame:
    """The windows_index table (same columns / dtypes as the CSV read)."""
    z = _z(d)
    cols = [str(c) for c in z["ix_columns"]]
    if usecols is not None:
        cols = [c for c in cols if c in set(usecols)]
    data = {}
    for c in cols:
        if f"ix_{c}__codes" in z:
            data[c] = z[f"ix_{c}__cats"][z[f"ix_{c}__codes"]].astype(object)
        else:
            data[c] = z[f"ix_{c}"]
    return pd.DataFrame(data)


def lp(d: Path) -> pd.DataFrame:
    """The LP windows table: (bodypart, coord) MultiIndex columns, float64 values (from float32 storage)."""
    z = _z(d)
    if "lp" not in z:
        raise FileNotFoundError(f"{d}: no windows_LP.csv")
    cols = pd.MultiIndex.from_arrays([z["lp_parts"].astype(str), z["lp_coords"].astype(str)],
                                     names=["bodyparts", "coords"])
    return pd.DataFrame(z["lp"].astype(np.float64), index=z["lp_index"], columns=cols)


def has_lp(d: Path) -> bool:
    return (Path(d) / "windows_LP.csv").exists()


def _build_one(d) -> str:
    d = Path(d)
    if _stale(d):
        build(d)
        return f"{d.name}: built"
    return f"{d.name}: current"


def main(argv=None) -> int:
    """``python -m wfield_local.pose_windows [--jobs 3]``: build / refresh the sidecar of every session_poses folder that
    has a windows_index.csv."""
    import argparse

    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--pattern", default="*_cam1_full")
    a = ap.parse_args(argv)
    root = Path(PathResolver().root("dlc")) / "session_poses"
    dirs = sorted(str(d) for d in root.glob(a.pattern) if (d / "windows_index.csv").exists())
    res, fails = ak.fan_sessions(dirs, _build_one, jobs=a.jobs)
    for d, e in fails:
        print(f"{d}: FAILED {e}")
    print(f"{len(res)} folders ok, {len(fails)} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
