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

Note for the ``mac`` machine profile: its SMB mount is rooted AT ``…\\MICROSCOPE\\Priya`` and appears
as ``/Volumes/Priya/…``, so the path carries no ``/microscope/`` segment and the share rule above does
not fire. That is safe as far as rule 1 goes — the mount root is already inside Priya, so every path
under it is too — but it does mean the guard cannot catch a resolver bug that pointed at a DIFFERENT
``/Volumes`` share on that client. Keep the mac mounts in ``paths.yaml`` pointed at ``/Volumes/Priya``.

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

__all__ = ["WriteGuardError", "assert_writable", "is_writable",
           "warn_if_partial_aggregate", "covers_all",
           "assert_deletable", "is_original_data"]

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


# --------------------------------------------------------------------------- partial-aggregate guard
#
# A SECOND failure mode, unrelated to path location: a SHARED, cross-animal output written by a run
# that only covered SOME animals. Two machines split a date (imaging box: PS92/PS93; helper box:
# PS94/PS95), each rebuilds the shared file from its own in-memory list, and the last writer wins --
# silently discarding the other's contribution. Observed 2026-08-11 on the 8/11 photobleach summary,
# which ended up describing a single animal.
#
# The structural cure is to derive shared aggregates from the per-item artifacts ON DISK (see
# photobleach.load_records). These helpers are the backstop for sites that cannot do that: they make
# a partial write LOUD instead of silent, and let a caller decline a destructive prune.

def covers_all(covered, expected) -> bool:
    """True iff ``covered`` includes every member of ``expected`` (order/duplicates irrelevant)."""
    return set(expected) <= set(covered)


def warn_if_partial_aggregate(path, covered, expected, what="aggregate") -> bool:
    """Print a loud warning when a SHARED output is about to cover only part of ``expected``.

    Returns True when the write is partial (caller may downgrade to a merge, or skip a prune).
    Deliberately warns rather than raises: a partial aggregate is legitimate mid-night, when the
    other machine simply has not finished yet. What must never happen is it going UNNOTICED.

        if writeguard.warn_if_partial_aggregate(out, done, config.animals(), "photobleach summary"):
            ...  # merge with what is already on disk instead of replacing it
    """
    missing = sorted(set(expected) - set(covered))
    if not missing:
        return False
    print(f"[writeguard] *** PARTIAL {what}: {path}\n"
          f"             covers {sorted(set(covered))} but {missing} are missing.\n"
          f"             If another machine also writes this file, the last writer WINS and the\n"
          f"             other's results are lost. Prefer merging from the per-item artifacts on\n"
          f"             disk, or re-run once every contributor has finished.", flush=True)
    return True


# --------------------------------------------------------------------- irreplaceable-data deletion

#: File kinds that are ACQUIRED, not computed. Losing one loses an experiment: it cannot be
#: regenerated from anything else in the pipeline. (Priya, 2026-08-14.)
ORIGINAL_SUFFIXES = (".dat", ".h5", ".camlog", ".avi")
ORIGINAL_NAMES = ("trials.csv", "events.csv", "gui_config.json", "session_manifest.json")


def is_original_data(path) -> bool:
    """Is this an ACQUIRED file (raw imaging, DAQ, camera, behavior log) rather than a derived one?

    Deliberately errs toward YES. A false positive costs one extra verification; a false negative
    deletes an experiment. Derived ``.dat``/``.h5`` do exist (a cleanpairs rebuild, a repaired file),
    and callers state that explicitly via ``derived=True`` rather than the guard trying to infer it
    from a filename -- inferring it is how a real original gets classified as scratch.
    """
    # NOT _normalize(): it appends a trailing "/" for directory substring matching, which would make
    # every endswith(".dat") fail and every basename come out empty -- an inert guard that still passes
    # its permissive tests.
    p = str(path).replace("\\", "/").lower().rstrip("/")
    name = p.rsplit("/", 1)[-1]
    return name in ORIGINAL_NAMES or p.endswith(ORIGINAL_SUFFIXES)


def assert_deletable(path, verified_copies=(), derived: bool = False,
                     approved: bool = False) -> None:
    """Refuse to delete ACQUIRED data without a verified server copy or explicit human approval.

    HARD RULE (Priya, 2026-08-14): never delete original data -- ``.dat``, ``.h5``, behavior logs,
    camera files -- without either a copy confirmed to exist on a server, or explicit permission.

    ``verified_copies`` must be paths the CALLER has already confirmed (existence AND size/hash); this
    guard re-checks that each exists and is non-empty, but it cannot re-do the caller's byte
    comparison, so passing an unverified path defeats the purpose. ``derived=True`` declares the file
    reproducible from inputs that are themselves archived (a repaired or cleanpairs ``.dat``).
    ``approved=True`` records explicit human permission for THIS deletion.

    Raises :class:`WriteGuardError` unless one of those three is satisfied. Derived files and
    anything not matching :func:`is_original_data` pass straight through.
    """
    if derived or approved or not is_original_data(path):
        return
    good = [c for c in verified_copies if c and os.path.exists(c) and os.path.getsize(c) > 0]
    if good:
        return
    raise WriteGuardError(
        f"refusing to delete ORIGINAL data with no verified copy: {path}\n"
        f"  Acquired files (.dat/.h5/camlog/avi/behavior logs) may only be deleted when a copy is "
        f"confirmed on a server, or with explicit permission.\n"
        f"  Pass verified_copies=[<confirmed server path>], or derived=True if it is reproducible "
        f"from archived inputs, or approved=True for an explicitly authorised deletion.")
