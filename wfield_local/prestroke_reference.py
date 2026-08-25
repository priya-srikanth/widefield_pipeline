"""Pre-stroke reference numbers and models are computed ONCE and then held fixed.

WHY (Priya, 2026-08-25). Every post-stroke result in this project is read against a PRE-STROKE
reference: the leave-one-session-out band a session is called "inside" or "outside", the frozen
decoder's transfer cost, the encoder's cross-day cost, the no-lick reference. Those references were
being recomputed every night over a session set that grows every night, which has two costs.

  SCIENTIFIC.  A reference that moves after the comparison data arrive is not a reference. The
               pipeline already accepts this for the no-lick arm -- `nolick_reference_prestroke.json`
               is written once and never overwritten, on exactly this reasoning -- but the decoder
               band and the frozen decoder/encoder costs were not covered by it. A post-stroke z
               score computed against a band that shifted last night is not comparable with the same
               z score quoted in a deck from three nights ago.
  TIME.        Measured 2026-08-25 over one night: `joint_xsession` 134 min, `poststroke_section_g`
               110 min, `precue_lickfree` 93 min, the pooled encoder 50 min. The pre-stroke share of
               that is recomputed nightly to land on (nearly) the same numbers.

WHAT "FROZEN" MEANS HERE, because the word is already overloaded in this repo. A *frozen decoder*
(`locanmf_frozen_decoder`) is frozen ACROSS DAYS -- trained on an animal's other days, applied to a
held-out day. That is a modelling choice and it refits every run. A *frozen reference*, this module,
is frozen IN TIME: computed once, stored, and reused verbatim until someone deliberately supersedes
it. The two are unrelated and a decoder can be both.

THE HOLE THIS CLOSES. The existing freeze is `if not frozen.exists(): write`. A bare existence check
cannot tell "the reference is current" from "the reference was computed from a session set that no
longer exists" -- so a re-curation, a new pre-stroke session, or an upstream re-preprocess (the
2026-08-14 flip to the meegkit_hpfit SVTcorr changed the DATA under unchanged labels) would leave a
stale reference in place with nothing to notice. Every frozen artifact here therefore stores the
signature of what it was computed FROM, and a mismatch is reported rather than resolved.

A MISMATCH IS NEVER RESOLVED SILENTLY, in either direction. Recomputing would move the reference,
which is the thing being prevented; using it blindly would score today's data against a reference
built from data that is gone. So `load_or_freeze` returns the frozen payload AND a status of STALE,
leaving the caller to refuse, and superseding is an explicit act (``refreeze=True``) that renames the
old artifact to ``<stem>.SUPERSEDED_<reason>_<stamp>.json`` rather than deleting it -- matching the
convention already on disk (`nolick_reference_prestroke.SUPERSEDED_contaminated_late_arm.json`).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from wfield_local import config

#: Bump when the MEANING of a frozen reference changes -- i.e. when the computation behind it is
#: altered such that an old artifact, though built from the same sessions, no longer answers the same
#: question. Mtimes cannot see a code change (the lesson `session_cache.CACHE_VERSION` records).
FREEZE_VERSION = 1

STATUS_NEW, STATUS_HIT, STATUS_STALE = "frozen-new", "frozen-hit", "STALE"


def _stat_sig(path) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "MISSING"


def _session_sig(s) -> str:
    """Signature of the INPUTS a reference is computed from, so a re-preprocess invalidates it.

    Hashing labels alone is not enough and this repo has the scar: on 2026-08-14 the switch to the
    meegkit_hpfit SVTcorr changed the underlying data while every label stayed identical.
    """
    import glob
    ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
    u_atlas = f"{ad[0]}/U_atlas.npy" if ad else ""
    return f"{_stat_sig(u_atlas)}|{_stat_sig(config.svtcorr_path(s['mc']))}"


def prestroke_signature(animals=None, extra=None) -> str:
    """Stable id over the pre-stroke session set, its inputs, and the freeze version.

    ``extra`` folds in whatever else determines the reference (window, alignment, basis, the decode
    params from defaults.yaml) so two references that differ only in a parameter cannot collide.
    """
    labels = set(config.phase_labels("pre"))
    sess = [s for s in config.load_sessions() if s["label"] in labels]
    if animals:
        sess = [s for s in sess if config.animal_of(s["label"]) in set(animals)]
    parts = [f"freeze{FREEZE_VERSION}"]
    if extra is not None:
        parts.append("extra=" + json.dumps(extra, sort_keys=True, default=str))
    for s in sorted(sess, key=lambda x: x["label"]):
        parts.append(f"{s['label']}:{_session_sig(s)}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def supersede(path: Path, reason: str) -> Path:
    """Retire a frozen artifact by RENAMING it, never by deleting or overwriting.

    The old numbers are what an earlier deck was built against, so they stay reachable; ``reason``
    goes in the filename because a superseded reference with no recorded reason is indistinguishable
    from a stray file.
    """
    path = Path(path)
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in reason)[:60] or "unspecified"
    dest = path.with_name(f"{path.stem}.SUPERSEDED_{safe}_{_stamp()}{path.suffix}")
    path.rename(dest)
    return dest


def load_or_freeze(path, compute, animals=None, extra=None, refreeze=None, log=print):
    """Return ``(payload, status)`` for a pre-stroke reference, computing it only the first time.

    ``compute`` is a zero-argument callable returning a JSON-serialisable payload; it is called ONLY
    when there is no current frozen artifact, which is where the time is saved. ``refreeze`` is the
    deliberate supersede path: pass a reason string to retire the existing artifact and mint a new
    one.

    Status is ``frozen-new`` (first computation), ``frozen-hit`` (reused verbatim -- the fast path),
    or ``STALE`` (an artifact exists but was built from different inputs; the payload is returned
    unchanged so nothing is silently rescored, and it is the CALLER's job to refuse).
    """
    path = Path(path)
    want = prestroke_signature(animals=animals, extra=extra)

    if path.exists():
        doc = json.loads(path.read_text())
        meta = doc.get("_freeze") or {}
        got = meta.get("signature")
        if got == want and not refreeze:
            return doc.get("payload", doc), STATUS_HIT
        if refreeze:
            old = supersede(path, refreeze)
            log(f"[prestroke] superseded {path.name} -> {old.name} ({refreeze})")
        else:
            log(f"[prestroke] !! {path.name} was frozen against a DIFFERENT pre-stroke input set "
                f"(frozen {got}, now {want}). NOT recomputed and NOT overwritten: recomputing would "
                f"move a reference that post-stroke results are already quoted against. Re-freeze "
                f"deliberately with refreeze='<reason>' once you know which set is right.")
            return doc.get("payload", doc), STATUS_STALE

    payload = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"_freeze": {"signature": want, "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
                     "freeze_version": FREEZE_VERSION, "animals": sorted(animals) if animals else None,
                     "extra": extra, "n_pre_sessions": len(config.phase_labels("pre"))},
         "payload": payload}, indent=2, default=float))
    log(f"[prestroke] froze {path.name} ({len(config.phase_labels('pre'))} pre-stroke sessions)")
    return payload, STATUS_NEW
