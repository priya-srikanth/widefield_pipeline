"""Block identity for position blocks — including the case where two adjacent blocks share a position.

Priya, 2026-08-18: *"I want to be sure we are appropriately labeling two blocks when the GUI randomly
put two blocks of the same position next to each other (ie sometimes far L block is followed by far L
block)."*

They were not. The pipeline started a new block whenever the POSITION changed, so a far_L block
followed by another far_L block became one block. Audited against the firmware's own count over all 48
curated + 8/17 sessions: **118 merges / 4216 blocks = 2.8%**, 0–8.2% per session.

WHY THE FIRMWARE COUNT IS THE GROUND TRUTH, AND WHY IT IS ONLY A TOTAL.
`device_snapshot_end.json` carries `block_number`, the scheduler's own count of blocks it ran. It is a
session TOTAL, not per-trial: the GUI polls device status every second but `logging.timeseries_enabled`
is off, so the polls are never written. (Turning it on would give per-trial block IDs, and was
considered and rejected — it would stream enough to risk destabilising the log. Priya, 2026-08-18.)
So `firmware_blocks - observed_runs` gives the exact merge COUNT per session with no inference, but
not where the boundaries are.

WHAT THIS MODULE DOES, AND WHAT IT CANNOT DO.
A run longer than `block_size_max` cannot be a single block, so it is split into chunks of at most
that length. That catches ~92% of the merges (108 of 118). Two limits remain, documented rather than
hidden:

  * 4+4 merges land at run-length exactly `block_size_max` and are indistinguishable from one genuine
    maximal block. About 10 of the 118 are of this kind and stay merged.
  * The PLACEMENT of a split inside an over-long run is a choice: a run of 11 could have been 4+7,
    5+6, 6+5 or 7+4. Chunking from the left at `block_size_max` picks one. Since blocks are only ever
    used as CV GROUPS, the placement affects which trials are held out together and not what is
    measured, so a defensible arbitrary choice is enough.

DIRECTION OF THE ERROR THIS CORRECTS. Merging made GroupKFold groups LARGER, holding more correlated
data out together, so the pre-fix numbers were CONSERVATIVE rather than inflated. Measured over the
eight worst-affected sessions and both alignments, splitting moved accuracy by a mean of +0.0105
(10/16 positive) — with the caveat that the ±0.05 scatter is probably fold-reassignment noise, so that
is evidence of no inflation rather than a measured merge effect.
"""
from __future__ import annotations

import glob as _glob
import json as _json
import re as _re
from pathlib import Path

import numpy as np

from wfield_local import config

DEFAULT_BLOCK_SIZE_MAX = 8          # gui_config timing.block_size_max on every session recorded so far


def _behavior_dir(s):
    """The behaviour-log directory for this session, or None.

    Resolved from the session's OWN date, never by globbing the animal — that shortcut silently gave
    every session its animal's earliest config once already (see nolick_decoder.response_window_for).
    """
    for cand in config.load_sessions():
        if cand["label"] == s["label"] and cand.get("behavior_trials"):
            return Path(cand["behavior_trials"]).parent
    animal = s["label"][:4]
    m = _re.search(rf"{animal}_(\d{{8}})_", str(s.get("h5") or "")) or \
        _re.search(r"[/\\](\d{8})[/\\]", str(s.get("h5") or s.get("mc") or ""))
    if not m:
        return None
    hits = sorted(_glob.glob(f"{config.resolver().root('behavior_logs')}/{animal}_{m.group(1)}_*"))
    return Path(hits[0]) if hits else None


def block_size_max_for(s, default=DEFAULT_BLOCK_SIZE_MAX):
    """This session's scheduler `block_size_max`, from its own gui_config.json."""
    d = _behavior_dir(s)
    if d is None:
        return int(default)
    try:
        cfg = _json.load(open(d / "gui_config.json"))
        return int(cfg.get("timing", {}).get("block_size_max", default))
    except Exception:                                                  # noqa: BLE001
        return int(default)


def firmware_block_count(s):
    """The scheduler's own block count from device_snapshot_end.json, or None.

    This is ground truth for how many blocks ran, and the only place block identity survives at all.
    """
    d = _behavior_dir(s)
    if d is None:
        return None
    try:
        st = _json.load(open(d / "device_snapshot_end.json"))
        st = st.get("latest_status", st)
        n = int(st.get("block_number", -1))
        return n if n > 0 else None
    except Exception:                                                  # noqa: BLE001
        return None


def block_ids(codes, block_size_max=DEFAULT_BLOCK_SIZE_MAX):
    """Block id per trial from a per-trial position `codes` array (-1 = unusable trial).

    A new block starts when the position changes OR when the current run reaches `block_size_max`,
    because a run longer than that cannot be one block.
    """
    out = np.full(len(codes), -1, dtype=int)
    b, prev, run = -1, None, 0
    for k, c in enumerate(codes):
        if c < 0:
            continue
        if prev is None or c != prev or run >= block_size_max:
            b += 1
            run = 0
        out[k] = b
        run += 1
        prev = int(c)
    return out


def audit(s, codes, block_size_max=None, verbose=True):
    """Compare the reconstructed block count with the firmware's. Returns a dict; never raises.

    A MISMATCH IS NOT ALWAYS A BUG HERE, which is why this warns rather than asserts:
      * reconstructed < firmware -> residual 4+4 merges, the known limit above.
      * reconstructed > firmware -> IMPOSSIBLE from merging alone, so it indicates damaged position
        labels. It fired on exactly the two sessions already known to be damaged: PS93_0806 (dead
        spout_bit1, behaviour-log fallback) and PS92_0812 (crash + concat). Worth keeping as a
        position-labelling guard independently of the block question.
    """
    bmax = block_size_max or block_size_max_for(s)
    ids = block_ids(np.asarray(codes), bmax)
    n_rec = int(len({int(i) for i in ids if i >= 0}))
    n_fw = firmware_block_count(s)
    out = {"label": s["label"], "block_size_max": bmax, "reconstructed": n_rec, "firmware": n_fw}
    if n_fw is None:
        out["status"] = "no firmware count"
        return out
    out["residual_merges"] = n_fw - n_rec
    if n_rec > n_fw:
        out["status"] = "MORE blocks than the firmware ran -- position labels are suspect"
        if verbose:
            print(f"  [block_ids] {s['label']}: {n_rec} reconstructed vs {n_fw} firmware blocks -- "
                  f"impossible from merging; check position labelling", flush=True)
    elif n_rec < n_fw:
        out["status"] = f"{n_fw - n_rec} residual merge(s) (4+4 at run-length {bmax})"
    else:
        out["status"] = "exact"
    return out
