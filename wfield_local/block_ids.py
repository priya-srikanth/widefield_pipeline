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
    5+6, 6+5 or 7+4. This module splits EVENLY (6+5), which is not arbitrary — see below.

WHY THE SPLIT IS EVEN AND NOT LEFT-CHUNKED (2026-09-26). It used to chunk from the left at
`block_size_max`, on the argument that "blocks are only ever used as CV GROUPS, so the placement
affects which trials are held out together and not what is measured". That argument was sound when
it was written and is no longer true: blocks are ALSO the exchangeable unit of the block-label
permutation nulls (`precue_significance.permute_block_labels`, `decode_ci.frozen_ci`,
`rest_frozen_decoder`'s `blockperm`), where the SIZE of a unit is exactly what is measured.

Left-chunking produced decompositions the scheduler could not have generated. With
`block_size_min: 4`, a run of 9 became **8+1** when the only legal splits are 4+5 and 5+4; measured
over the 177 `balanced_block_cycles` sessions, **286 of 491 over-long runs (58.2%)** came out
illegal — essentially every run of 9, 10 and 11.

WHICH WAY IT ERRED, MEASURED — and NOT the way it was first argued. The obvious reading is that a
one-trial permutation unit is trial-level shuffling for that trial, destroying within-block
correlation and UNDERSTATING the null, so the old rule would err toward false positives. A paired
A/B says otherwise: over the six worst-affected sessions the old rule sat the null MEAN **0.0038
HIGHER** with the SD unchanged, so it was CONSERVATIVE. The width argument fails because the width
does not move. What dominates is that the null REFITS per permutation, so left-chunking's larger
coherent chunk (8, against 5+4) is more learnable from drift and scores higher. No `p_perm` changed.
DECISIONS.md 2026-09-26 has the table. **The fix stands on legality regardless of that direction.**

Splitting a run of n into ceil(n / block_size_max) pieces as evenly as possible is optimal against
`block_size_min` and needs no knowledge of it: the largest achievable minimum piece is floor(n / k),
k is as small as the maximum allows, and an even split attains it. Where no legal decomposition
exists (a run of 6 with min=max=5, which the scheduler cannot have produced either) this still
returns the best available rather than raising — the run is already evidence of a damaged session,
and `audit` is the place that says so.

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
DEFAULT_BLOCK_SIZE_MIN = 4          # timing.block_size_min; 4 on all but a handful of sessions, which use 5
N_POSITIONS = 6                     # the scheduler's cycle length -- six spout positions, one block each


def _behavior_dir(s):
    """The behaviour-log directory for this session, or None.

    ``s`` is a session MAPPING, or a Path to the behaviour directory itself. The second form is
    what `spout_behavior` holds -- it is already standing in the session's log directory -- and
    accepting it here keeps `block_size_max_for` the single reader of `timing.block_size_max`
    rather than having the behaviour path grow its own copy of the same three lines.

    Resolved from the session's OWN date, never by globbing the animal — that shortcut silently gave
    every session its animal's earliest config once already (see nolick_decoder.response_window_for).
    """
    if isinstance(s, (str, Path)):
        d = Path(s)
        return d if (d / "gui_config.json").exists() else None
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


def block_size_min_for(s, default=DEFAULT_BLOCK_SIZE_MIN):
    """This session's scheduler `block_size_min`, from its own gui_config.json.

    `block_ids` does NOT take this — an even split satisfies it automatically wherever anything can
    (see the module docstring). It is here so `audit` can report a run whose pieces fall outside the
    scheduler's own bounds, which means the run is longer than two maximal blocks and the session's
    position labels are suspect rather than merely merged.
    """
    d = _behavior_dir(s)
    if d is None:
        return int(default)
    try:
        cfg = _json.load(open(d / "gui_config.json"))
        return int(cfg.get("timing", {}).get("block_size_min", default))
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


def split_lengths(n, block_size_max=DEFAULT_BLOCK_SIZE_MAX):
    """Lengths of the blocks a run of `n` same-position trials is made of.

    The fewest blocks that can hold the run, as EVENLY as possible — which is the split that keeps
    every piece furthest from `block_size_min`. Exposed for the tests and the audit; `block_ids` is
    the caller that matters.
    """
    if n <= block_size_max:
        return [n]
    k = -(-n // block_size_max)                 # ceil: fewest maximal blocks that can cover the run
    base, rem = divmod(n, k)
    return [base + 1] * rem + [base] * (k - rem)


def block_ids(codes, block_size_max=DEFAULT_BLOCK_SIZE_MAX):
    """Block id per trial from a per-trial position `codes` array (-1 = unusable trial).

    A new block starts when the position changes, and a run too long to be one block is divided
    evenly (see `split_lengths` and the module docstring on why evenly and not from the left).

    Unusable trials get no block and do NOT break the surrounding run: a `-1` in the middle of a
    far_L run is a position we could not resolve, not a change of position, and treating it as a
    boundary would invent a block the scheduler never ran.
    """
    codes = np.asarray(codes)
    out = np.full(len(codes), -1, dtype=int)
    usable = np.flatnonzero(codes >= 0)
    if not usable.size:
        return out
    c = codes[usable]
    # run boundaries over the USABLE trials only, so a -1 gap cannot split a run
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    b = -1
    for i, st in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(c)
        at = st
        for ln in split_lengths(stop - st, block_size_max):
            b += 1
            out[usable[at:at + ln]] = b
            at += ln
    return out


def cycle_ids(codes, block_size_max=DEFAULT_BLOCK_SIZE_MAX, n_positions=N_POSITIONS):
    """Cycle id per trial (-1 = unusable), for the scheduler's `balanced_block_cycles` design.

    The scheduler presents all six positions in a random order, re-randomises, and starts again
    (`gui_config.json timing.scheduling_mode: balanced_block_cycles`, `stop_mode:
    end_of_balanced_cycle`). A cycle is therefore six blocks covering six DISTINCT positions, and a
    position cannot repeat inside one.

    **THIS DOES NOT REUSE `block_ids`, AND THE REASON IS MEASURED.** The obvious implementation --
    take `block_ids`' output and close a cycle when a position repeats -- scores 86.0%, not 95.1%.
    `block_ids` splits EVERY run longer than `block_size_max`, but only **330 of the 491** long runs
    sit at a cycle boundary; the other 161 do not, so splitting them fires the repeat rule mid-cycle
    and truncates the cycle (it produces cycles of 1-5 blocks, 332 of them). Those 161 are evidence
    that a long run is not always a merge -- some are genuine over-long blocks, and some are the
    damaged position labels `audit` reports.

    So this splits a long run ONLY where the cycle says it must be a merge: the run completes the
    cycle (five positions already seen) and is too long to be one block. That is the one case where
    two same-position blocks are known to be adjacent. Everywhere else a long run is left whole, and
    a cycle that is genuinely malformed is reported as such rather than silently re-cut.

    Consequence to know about: for those 161 runs this function's internal block boundaries differ
    from `block_ids`'. That is deliberate -- they answer different questions. `block_ids` is a CV
    group and a permutation unit, where splitting unconditionally is the conservative choice; here
    the question is which cycle a trial belongs to, and an unconditional split destroys the answer.

    **DO NOT CHUNK THE BLOCK SEQUENCE INTO FIXED GROUPS OF SIX.** Measured over the 177
    `balanced_block_cycles` sessions: greedy closing gives exactly six blocks in **95.1%** of cycles,
    fixed chunking in **53.7%**. The first boundary merge shifts the phase and every later group
    looks broken, which is also why a session can appear to have no cycle structure at all.

    The residual 4.9% is the same limit `block_ids` documents -- a 4+4 merge lands at run-length
    exactly `block_size_max` and cannot be split, so its two blocks stay merged and the cycle they
    straddle comes out five blocks long rather than six.

    **NOTHING USES THIS FOR A NULL, DELIBERATELY.** It was written to answer whether the permutation
    tests should permute WITHIN cycle rather than within session; measured on all 44 curated
    sessions, no verdict sits in the band where null width could matter, so the within-session nulls
    stand. See DECISIONS.md 2026-09-26. It is kept because it is the prerequisite for reopening that
    question, and because per-cycle is a natural unit for a behavioural one (does performance drift
    within a cycle?).
    """
    codes = np.asarray(codes)
    out = np.full(len(codes), -1, dtype=int)
    usable = np.flatnonzero(codes >= 0)
    if not usable.size:
        return out
    c = codes[usable]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])       # runs over the USABLE trials only
    cyc, seen, at = 0, set(), 0
    for i, st in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(c)
        p, n = int(c[st]), stop - st
        if p in seen:                       # a position cannot repeat inside a cycle
            cyc += 1
            seen = set()
        if len(seen) == n_positions - 1 and n > block_size_max:
            # Completes the cycle AND is too long to be one block, so it is the boundary merge.
            # Split it the same way `split_lengths` would, first piece closing the open cycle.
            head = split_lengths(n, block_size_max)[0]
            out[usable[at:at + head]] = cyc
            cyc += 1
            out[usable[at + head:at + n]] = cyc
            seen = {p}
            at += n
            continue
        out[usable[at:at + n]] = cyc
        seen.add(p)
        at += n
        if len(seen) >= n_positions:
            cyc += 1
            seen = set()
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
    # A piece under `block_size_min` means the run was too long for the blocks it was split into --
    # i.e. longer than two maximal blocks -- which merging alone cannot produce. Same class of
    # evidence as `reconstructed > firmware` below, and it fires on the same damaged sessions.
    bmin = block_size_min_for(s)
    sizes = np.bincount(ids[ids >= 0]) if n_rec else np.zeros(0, dtype=int)
    out["block_size_min"] = bmin
    out["undersized_blocks"] = int((sizes < bmin).sum())
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
