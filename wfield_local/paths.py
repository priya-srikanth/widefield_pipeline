"""Path resolution: logical roots -> per-machine mounts.

Machines share this repo and the same `configs/*.yaml`:

  - the ANALYSIS box (behavior-GPU): MICROSCOPE is mapped to ``M:``; it reads
    LocaNMF inputs/outputs and runs decode/encode/RSA. It has no local raw imaging.
  - the IMAGING box (PCO microscope): MICROSCOPE is mapped to ``N:``; local raw
    acquisition lives on ``E:``. It runs preprocessing (motion -> SVD -> register -> push).
  - a MAC/Linux client with MICROSCOPE mounted over SMB at ``/Volumes/Priya`` — same shared
    tree, no local raw imaging. The SMB share is rooted at ``...\\MICROSCOPE\\Priya``, so its
    paths carry no ``MICROSCOPE`` segment (note for anything matching on that, e.g. writeguard).

The SAME logical root (e.g. ``labcams``) therefore resolves to a different drive on each
box. :class:`PathResolver` picks the mount for the current machine, chosen by (in order):

  1. an explicit ``machine=`` argument (used by tests),
  2. the ``WIDEFIELD_MACHINE`` env var (``"analysis"`` | ``"imaging"``),
  3. auto-detection from a signature mount on disk (E:\\labcams_data -> imaging;
     M:\\MICROSCOPE -> analysis; /Volumes/Priya -> mac), defaulting to ``"analysis"``.

Mount values in ``paths.yaml`` may be a string, a list of strings (first existing wins),
or ``null`` (root unavailable on this machine -> :meth:`root` raises).

Mounts are stored and returned as forward-slash strings; :meth:`resolve` joins with ``/``
so resolved session paths are byte-identical to the legacy hard-coded ``M:/...`` strings
that downstream ``np.load`` / ``h5py`` callers already accept on Windows.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
MACHINES = ("analysis", "imaging", "mac")

# Signature mounts used to auto-detect the machine when WIDEFIELD_MACHINE is unset.
_SIGNATURE = (("imaging", "E:/labcams_data"), ("analysis", "M:/MICROSCOPE/Priya"),
              ("mac", "/Volumes/Priya"))


def detect_machine() -> str:
    """Resolve the current machine: env override, else on-disk signature, else 'analysis'."""
    env = os.environ.get("WIDEFIELD_MACHINE")
    if env:
        if env not in MACHINES:
            raise ValueError(f"WIDEFIELD_MACHINE must be one of {MACHINES}, got {env!r}")
        return env
    for machine, sig in _SIGNATURE:
        if Path(sig).exists():
            return machine
    return "analysis"


class PathResolver:
    """Resolve logical roots from ``configs/paths.yaml`` to this machine's mounts."""

    def __init__(self, config_path: str | Path | None = None, machine: str | None = None):
        path = Path(config_path) if config_path else CONFIG_DIR / "paths.yaml"
        with open(path, encoding="utf-8") as fh:
            self._cfg = yaml.safe_load(fh)
        self.machine = machine or detect_machine()
        if self.machine not in MACHINES:
            raise ValueError(f"machine must be one of {MACHINES}, got {self.machine!r}")
        self._candidates: dict[str, list[str]] = {}
        for name, spec in (self._cfg.get("logical_roots") or {}).items():
            mount = (spec.get("mounts") or {}).get(self.machine)
            if mount is None:
                self._candidates[name] = []
            elif isinstance(mount, list):
                self._candidates[name] = [str(m) for m in mount]
            else:
                self._candidates[name] = [str(mount)]

    def root_names(self) -> list[str]:
        return list(self._candidates)

    def root(self, name: str) -> str:
        """Return the resolved mount string for a logical root (forward-slash form).

        Single-mount roots return directly (no existence check — a write destination may
        not exist yet). List mounts return the first candidate that exists on disk.
        """
        if name not in self._candidates:
            raise KeyError(f"Unknown logical root {name!r}. Known: {self.root_names()}")
        cands = self._candidates[name]
        if not cands:
            raise RuntimeError(f"Root {name!r} has no mount for machine {self.machine!r}.")
        if len(cands) == 1:
            return cands[0]
        for c in cands:
            if Path(c).exists():
                return c
        raise FileNotFoundError(
            f"Root {name!r}: none of the mounts exist on {self.machine!r}:\n  "
            + "\n  ".join(cands)
        )

    def resolve(self, name: str, rel: str) -> str:
        """Join a root-relative path onto its logical root, as a forward-slash string.

        An already-absolute ``rel`` (drive-letter, UNC, or POSIX-absolute) is returned
        unchanged, so partially-migrated configs keep working.
        """
        rel = str(rel)
        if is_absolute(rel):
            return rel.replace("\\", "/")
        return f"{self.root(name).rstrip('/')}/{rel.lstrip('/')}"

    def staging(self, name: str) -> str:
        """Return a ``staging:`` path (imaging-box local acquisition dirs)."""
        val = (self._cfg.get("staging") or {}).get(name)
        if val is None:
            raise KeyError(f"Unknown staging path {name!r}")
        return str(val)


def is_absolute(p: str) -> bool:
    """True for a drive-letter (``M:``), UNC (``\\\\host``), or POSIX-absolute path."""
    p = str(p)
    if len(p) >= 2 and p[1] == ":":
        return True
    if p.startswith("\\\\") or p.startswith("//"):
        return True
    return p.startswith("/")
