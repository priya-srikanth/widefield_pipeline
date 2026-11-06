"""ONE verified copy for acquired data: size check always, SHA-256 read-back on request.

DATA-SAFETY PATH (CLAUDE.md ground rule 0). Both upload tools copy ORIGINAL data to the server, and
`archive_day clean` later deletes the local source on the strength of that copy, so the copy rule must
be one rule. Until 2026-11-04 `archive_day._copy_one` and `camera_nightly._copy_one` were two copies of
it that had drifted (code review 2026-11-04):

  * read buffers 64 MB vs 1 MB (copy) / 16 MB (hash) -- harmless, but a sign they were edited apart;
  * the "already there, same size" check: archive_day FAILED when the source could not be hashed,
    camera_nightly compared ``_sha256(src) != _sha256(dst)`` and so PASSED when both were unreadable
    (``None != None`` is False) -- a skip reported as verified with nothing measured;
  * archive_day returned the source hash for its manifest; camera_nightly threw it away.

What this module does, in order (`verified_copy`):

  1. refuse a missing source (``FileNotFoundError``) before touching the destination;
  2. a destination that already exists AT THE SOURCE'S SIZE is a ``skip`` -- with ``readback`` both are
     hashed and anything but two equal, real digests is ``FAIL``;
  3. otherwise stream src -> ``<dst>.part`` (hashing the bytes as they are read, so the source hash is
     free), ``os.replace`` it into place, then size-check the destination and, with ``readback``, hash
     the destination back off the disk and compare it to the source hash.

A ``.sha256`` value returned here is a MEASUREMENT of the source made during this copy. A ``skip`` without
``readback`` returns ``None``: nothing was measured, and the caller must not record a hash.

NOT here, deliberately: what gets deleted and when. `archive_day.cmd_clean` re-verifies size (and bytes
with ``--hash``) at the delete site and passes the destination to `writeguard.assert_deletable`; that
stays at the delete site so a copy that was fine at upload and damaged later is still caught.
"""
from __future__ import annotations

import hashlib
import os

#: read/write block for the copy and for hashing. Only speed depends on it, never a result.
BUF = 16 * 1024 * 1024


def sha256(path) -> str | None:
    """Full SHA-256 of a file (streamed). None if it cannot be read -- never a digest of nothing."""
    h = hashlib.sha256()
    try:
        with open(path, "rb", buffering=0) as f:
            while True:
                b = f.read(BUF)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def size(path) -> int:
    """File size in bytes, or -1 if it does not exist / cannot be stat'ed."""
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def same_bytes(a, b) -> bool:
    """True only when BOTH files hash and the digests match. An unreadable side is never a match: comparing
    ``sha256(a) != sha256(b)`` lets two unreadable files (None, None) through as identical, which at a delete
    site would remove an original whose copy was never measured (archive_day cmd_clean, fixed 2026-11-04)."""
    ha = sha256(a)
    return ha is not None and ha == sha256(b)


def verified_copy(src, dst, *, readback: bool = False) -> tuple[str, str | None]:
    """Copy ``src`` -> ``dst`` and verify it. Returns ``(status, src_sha256)``.

    status: ``"ok"`` copied and verified; ``"skip"`` destination already present at the source's size
    (and, with ``readback``, byte-identical); ``"FAIL"`` the destination is not a verified copy.
    ``readback`` = hash the destination back and compare to the source (slow on a cold link; callers
    switch it off for very large files and then rely on the size check alone).
    Raises ``FileNotFoundError`` when ``src`` is missing; nothing is created in that case.
    """
    src, dst = os.fspath(src), os.fspath(dst)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"verified_copy: source missing: {src}")
    ssz = os.path.getsize(src)
    if os.path.exists(dst) and size(dst) == ssz:
        if not readback:
            return "skip", None
        sh = sha256(src)
        if sh is None or sh != sha256(dst):          # an unreadable side is a FAIL, not a match
            return "FAIL", sh
        return "skip", sh
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".part"
    h = hashlib.sha256()
    with open(src, "rb", buffering=0) as fi, open(tmp, "wb", buffering=0) as fo:
        while True:
            b = fi.read(BUF)
            if not b:
                break
            fo.write(b)
            h.update(b)
    os.replace(tmp, dst)
    src_hash = h.hexdigest()
    if size(dst) != ssz:
        return "FAIL", src_hash
    if readback and sha256(dst) != src_hash:
        return "FAIL", src_hash
    return "ok", src_hash
