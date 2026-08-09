"""Runtime guard: refuse writes/deletes that land on the MICROSCOPE / standby shares OUTSIDE
the Priya subtree — CLAUDE.md ground rule 1 ("only ever write inside MICROSCOPE/Priya/…; never
another person's folder").

Why location-based (not a deny-list of source files like stroke_orofacial's `_writeguard`): this
pipeline WRITES INTO ``…/MICROSCOPE/Priya/Widefield/…`` (that IS its output tree), and raw ``.dat`` /
DAQ ``.h5`` files legitimately move to standby during archival — so a file-pattern deny would
false-positive on normal operation. The robust, false-positive-free rule is purely about location:

  - on the research MICROSCOPE share (any path containing ``/microscope/``), the target MUST be under
    ``/microscope/priya/`` — this holds for both mounts (``M:`` on the analysis box, ``N:`` on the
    imaging box) and the raw UNC form, machine-independently;
  - on the standby share (the standby host, or a ``…/neurobio/sabatini/…`` path), the target MUST be
    under a ``/priya/`` segment;
  - everything else (local ``C:``/``D:``/``E:``, non-share paths) is always allowed.

The pipeline only ever writes under Priya, so ``assert_writable`` never raises in normal operation;
it exists to catch a path-construction bug that would drop the ``Priya`` segment or resolve into
another person's folder / a share root. Cheap (a few substring checks) and idempotent — safe to call
in a loop. Call it at a write/delete site BEFORE the destructive op:

    from wfield_local import writeguard
    writeguard.assert_writable(dest)
    shutil.rmtree(dest)
"""
from __future__ import annotations

import os

__all__ = ["WriteGuardError", "assert_writable", "is_writable"]

# The one owner segment this pipeline may write under, on every shared filesystem (CLAUDE.md rule 1).
_OWNER = "priya"

# Substrings (in the normalized, lowercased, forward-slash path) that mark a recognized network share.
_MICROSCOPE_MARK = "/microscope/"                       # M:/MICROSCOPE/…, N:/MICROSCOPE/…, \\research…\Neurobio\MICROSCOPE\…
_STANDBY_MARKS = ("standby.files.med.harvard.edu", "/neurobio/sabatini/")


class WriteGuardError(RuntimeError):
    """Raised by :func:`assert_writable` when a write/delete target resolves onto a MICROSCOPE /
    standby share but OUTSIDE the Priya subtree (another person's folder, or a share-root path bug).
    The message names the offending path and which rule matched, to diagnose the path construction.
    """


def _normalize(path: str | os.PathLike[str]) -> str:
    """Forward-slash, lowercase, trailing-slash form for unambiguous substring matching."""
    s = str(path)
    if not s:
        return ""
    normed = os.path.normpath(s).replace("\\", "/").lower()
    return normed.rstrip("/") + "/"


def assert_writable(path: str | os.PathLike[str]) -> None:
    """Raise :class:`WriteGuardError` if ``path`` is on a recognized share but outside Priya's subtree.

    No-op for local paths and for legitimate ``…/Priya/…`` share destinations. ``path`` need not
    exist; the check is purely syntactic.
    """
    c = _normalize(path)
    if not c:
        return
    owner = f"/{_OWNER}/"
    if _MICROSCOPE_MARK in c and f"{_MICROSCOPE_MARK}{_OWNER}/" not in c:
        raise WriteGuardError(
            f"refusing to write outside MICROSCOPE/Priya:\n  path: {path}\n"
            f"  a '/microscope/' path must be under '/MICROSCOPE/{_OWNER.title()}/' (CLAUDE.md rule 1: "
            f"never write to another person's folder). This usually means a path-construction bug "
            f"dropped the 'Priya' segment or resolved to a share root.")
    if any(m in c for m in _STANDBY_MARKS) and owner not in c:
        raise WriteGuardError(
            f"refusing to write outside the standby Priya subtree:\n  path: {path}\n"
            f"  a standby-share path must contain a '/{_OWNER}/' segment (CLAUDE.md rule 1).")


def is_writable(path: str | os.PathLike[str]) -> bool:
    """True iff :func:`assert_writable` would NOT raise on ``path`` (branch instead of raising)."""
    try:
        assert_writable(path)
    except WriteGuardError:
        return False
    return True
