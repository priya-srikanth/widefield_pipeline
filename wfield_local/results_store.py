"""Persist the NUMBERS behind each analysis figure, next to the figure.

Why: every analysis step used to emit only a ``.png``. Regenerating a figure with a tweaked axis, or
re-checking a number six weeks later, meant recomputing the whole thing from ``SVTcorr`` -- tens of
minutes to hours, on a machine that still has the inputs. Worse, an aggregate that lives only inside
a picture cannot be merged, audited, or re-plotted, which is exactly how the 2026-08-11 photobleach
summary silently lost three animals (see :mod:`wfield_local.photobleach`).

The rule this module encodes: **a figure is a VIEW of a saved result, never the only copy of it.**

    from wfield_local import results_store as rs
    rs.save(out_dir, "rsa", tag, {"rdm": {...}, "reliability": {...}}, meta={"align": "lick"})
    d = rs.load(out_dir, "rsa", tag)          # -> dict, ready to re-plot

Layout: ``<out_dir>/results/<name>{_<tag>}.json`` for scalars/small tables, plus a sibling ``.npz``
when any value is an ndarray (JSON would bloat and lose dtype). Arrays are restored transparently by
:func:`load`, so a caller sees one dict either way.

Everything is plain JSON/NPZ -- readable from MATLAB, a notebook, or a future rewrite -- and carries
a ``_meta`` block recording the pipeline params that produced it, so a stale result can be detected
rather than silently re-plotted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

SUBDIR = "results"
SCHEMA = 1


def _stem(out_dir, name, tag=None):
    d = Path(out_dir) / SUBDIR
    return d / (f"{name}_{tag}" if tag else name)


def _split_arrays(obj, arrays, prefix=""):
    """Recursively replace ndarrays with a marker, collecting them for the .npz sidecar."""
    if isinstance(obj, np.ndarray):
        key = prefix or "root"
        arrays[key] = obj
        return {"__ndarray__": key}
    if isinstance(obj, dict):
        return {k: _split_arrays(v, arrays, f"{prefix}/{k}" if prefix else str(k))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_split_arrays(v, arrays, f"{prefix}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _restore_arrays(obj, npz):
    if isinstance(obj, dict):
        if "__ndarray__" in obj and len(obj) == 1:
            return npz[obj["__ndarray__"]] if npz is not None else None
        return {k: _restore_arrays(v, npz) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_arrays(v, npz) for v in obj]
    return obj


def save(out_dir, name, tag, payload, meta=None):
    """Write ``payload`` (nested dict/list; ndarrays allowed) as the saved result for a figure.

    ``meta`` should carry the params that determined the numbers (alignment, source/basis, window,
    date span) so a later reader can tell whether a stored result matches what they want.
    Returns the json path.
    """
    stem = _stem(out_dir, name, tag)
    stem.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    body = _split_arrays(payload, arrays)
    body["_meta"] = {"schema": SCHEMA, "name": name, "tag": tag,
                     "written": time.strftime("%Y-%m-%d %H:%M:%S"), **(meta or {})}
    if arrays:
        np.savez_compressed(str(stem) + ".npz", **arrays)
    with open(str(stem) + ".json", "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=2, default=float)
    return str(stem) + ".json"


def load(out_dir, name, tag=None):
    """Read back a saved result (arrays restored). Returns None when it was never written."""
    stem = _stem(out_dir, name, tag)
    jp = Path(str(stem) + ".json")
    if not jp.exists():
        return None
    with open(jp, encoding="utf-8") as fh:
        body = json.load(fh)
    npz_path = Path(str(stem) + ".npz")
    npz = np.load(npz_path) if npz_path.exists() else None
    try:
        return _restore_arrays(body, npz)
    finally:
        if npz is not None:
            npz.close()


def listing(out_dir):
    """Every saved result under ``out_dir`` as {filename: _meta} -- what exists and how it was made."""
    d = Path(out_dir) / SUBDIR
    out = {}
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            with open(p, encoding="utf-8") as fh:
                out[p.name] = json.load(fh).get("_meta", {})
        except (OSError, ValueError) as e:
            out[p.name] = {"error": str(e)}
    return out


__all__ = ["SCHEMA", "SUBDIR", "listing", "load", "save"]
