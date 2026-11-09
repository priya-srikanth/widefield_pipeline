"""Stale-read guards for caches whose CONTENT depends on the analysis masks but whose KEY / PATH does not (2026-11-09).

THE BUG THIS GUARDS AGAINST. Under the rebuild profile the painted fiber-glue mask was silently empty (GLUE_MASK_FIX
2026-11-09), so every brain / stat mask, every `roi_masks` area and every `component_exclusion` drop kept the glue. Many
caches are skip-if-exists on a path (or keyed on code / version only), so after the fix they would hand back glue-in
content as if it were current. The rule, the same one `scripts/iti_licks.check_extract_masks` introduced:

* a cache written from now on STORES the digest of the masks it was built on (`signature()`, the sha1 of the brain,
  stat and painted-union mask content = `beta_maps.mask_signature`; `exclusion_signature(animal)` for caches built on
  the per-animal `component_exclusion.excluded_pixels`);
* on reuse the stored digest must equal the current one: a mismatch is REFUSED (RuntimeError), never recomputed over;
* a cache WITHOUT a digest is refused inside a rebuild root (every rebuild-root cache before the fix was built with the
  glue in) and accepted at the default root (those were built on the glue-out masks, which the fix did not change).

Never deletes or moves anything: the error names the folder to move aside, and the coordinator does that.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

#: the field name every guarded cache stores its digest under (npz key / pickle dict key)
FIELD = "mask_sig"


def signature() -> str:
    """Digest of the ANALYSIS masks (brain + stat + painted union) = `beta_maps.mask_signature()`."""
    from wfield_local import beta_maps
    return beta_maps.mask_signature()


def _digest(arrays, n: int = 12) -> str:
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, bool))
        h.update(repr(a.shape).encode())
        h.update(np.packbits(a).tobytes())
    return h.hexdigest()[:int(n)]


def exclusion_signature(animal, include_bulbs: bool = True, n: int = 12) -> str:
    """Digest of one animal's `component_exclusion.excluded_pixels` content (painted glue [+ bulbs]).

    Before the fix this was bulbs-only for every animal under the rebuild profile (15,106 px), so the occluded-component
    drop it feeds differs between glue-in and glue-out runs while `signature()` alone would not see it per animal.
    """
    from wfield_local import component_exclusion
    return _digest([component_exclusion.excluded_pixels(animal, include_bulbs)], n)


def combined_signature(animal=None) -> str:
    """`signature()` plus, when `animal` is given, that animal's exclusion digest: one token for a filename / key."""
    s = signature()
    return s if animal is None else f"{s}-{exclusion_signature(animal)}"


def check(have, where, now=None, what: str = "analysis masks") -> None:
    """REFUSE a cache built on other masks. `have` = the stored digest (None = the cache has none), `where` = the cache
    path (for the message), `now` = the current digest (default `signature()`)."""
    if have is not None and not isinstance(have, str):
        have = str(np.asarray(have).item()) if np.ndim(have) == 0 else str(have)
    if have in (None, ""):
        from wfield_local import rebuild_root
        if not rebuild_root.active():
            return
        raise RuntimeError(f"{where}: a rebuild-root cache without {FIELD} = built before the 11-09 glue-mask fix (glue "
                           f"IN). Move {Path(where).parent} aside and rebuild it.")
    now = signature() if now is None else now
    if have != now:
        raise RuntimeError(f"{where}: built on {what} {have}, current {what} are {now}. Move {Path(where).parent} aside "
                           f"and rebuild it.")


def stored(obj):
    """The digest stored in an npz (NpzFile / dict of arrays) or a pickled dict, else None."""
    try:
        keys = obj.files if hasattr(obj, "files") else obj.keys()
    except AttributeError:
        keys = ()
    if FIELD not in keys:
        return None
    v = obj[FIELD]
    return str(np.asarray(v).item()) if isinstance(v, np.ndarray) else str(v)


def check_npz(f, now=None) -> None:
    """`check` on the digest stored in an .npz file."""
    with np.load(f, allow_pickle=False) as z:
        have = stored(z)
    check(have, f, now)


def check_obj(obj, where, now=None) -> None:
    """`check` on the digest stored in an already-loaded npz / dict (`obj`) read from `where`."""
    check(stored(obj), where, now)


def stamp(now=None) -> dict:
    """The field to add to a cache when writing it: ``{FIELD: digest}``."""
    return {FIELD: signature() if now is None else now}
