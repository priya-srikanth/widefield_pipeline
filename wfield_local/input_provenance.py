"""WHICH LICK SET AND 3-D TABLE A FIGURE ACTUALLY READ (lane E, 11-24; Priya: "the captions and suptitles exist so that
inputs and settings can be traced", so the lick set / 3-D source on a figure must come from the inputs LOADED, not from
the intended settings).

Every ITI session file (`iti_licks.extract`, since 11-22) carries ``lick_sig`` = `session_licks.signature()` + the
whole-session kinematics table it was built on (folder name, size, mtime; `iti_licks._lick_sig`). Readers that load a
session file call `record` with that value; `footer_text` turns every signature seen in this process into one line that
`trial_set_text.stamp` adds to every saved figure, so a figure states the lick set and 3-D table its inputs were built on.
A session file without ``lick_sig`` is recorded as the legacy (pre-11-22, cam4) lick set.

`stamp_keys` / `check_same_source` are the write / read halves for DERIVED per-session extracts (basis extracts, ...):
the extract stores the lick_sig of the session file it was sampled from plus the 3-D run folder of its environment, and
a reader refuses an extract whose lick_sig differs from the session file it is read with (as `check_extract_licks` does
for the session files themselves)."""
from __future__ import annotations

import datetime as _dt

LEGACY = "legacy:cam4 (no lick_sig; session file built before 11-22)"
_SEEN: dict[str, int] = {}


def record(sig: str | None) -> None:
    """Note one loaded input's lick signature (None / "" = a legacy session file without one)."""
    s = str(sig) if sig not in (None, "") else LEGACY
    _SEEN[s] = _SEEN.get(s, 0) + 1


def seen() -> dict[str, int]:
    return dict(_SEEN)


def reset() -> None:
    _SEEN.clear()


def describe(sig: str) -> str:
    """'fused:<run>:<hash>|<kin folder>:<size>:<mtime_ns>' -> a readable phrase; anything else is returned as is."""
    if sig == LEGACY:
        return sig
    licks, _, kin = sig.partition("|")
    parts = licks.split(":")
    lk = f"fused four-camera lick list {parts[1]} ({parts[2]})" if parts[0] == "fused" and len(parts) >= 3 else \
        (licks or "legacy cam4 lick set")
    if kin:
        k = kin.split(":")
        when = ""
        if len(k) >= 3 and k[2].isdigit() and int(k[2]):
            when = _dt.datetime.fromtimestamp(int(k[2]) / 1e9).strftime("%Y-%m-%d %H:%M")
        size = f"{int(k[1]):,} B" if len(k) >= 2 and k[1].isdigit() else ""
        kt = f"3-D table {k[0]}/licks.csv.gz ({', '.join(x for x in (size, when) if x)})"
    else:
        kt = "3-D table: not recorded"
    return f"{lk}; {kt}"


def pose3d_run() -> str:
    """The 3-D run folder of THIS environment (`pose3d.pose3d_dir`), e.g. pose3d_4cam_3net_20261121_sync60."""
    try:
        from wfield_local import pose3d
        return pose3d.pose3d_dir().name
    except Exception:                                              # noqa: BLE001 - provenance never kills a figure
        return "unknown"


def footer_text() -> str:
    """One line naming every lick signature read in this process ("" when none was recorded)."""
    if not _SEEN:
        return ""
    body = "; ".join(f"{describe(s)} [{n} session file(s)]" for s, n in sorted(_SEEN.items()))
    return f"inputs read: {body}; 3-D run {pose3d_run()}"


def stamp_keys(source_sig: str | None) -> dict:
    """npz keys a derived per-session extract stores: the lick_sig of the session file it was sampled from and the 3-D
    run folder of its environment."""
    return {"lick_sig": str(source_sig) if source_sig not in (None, "") else LEGACY, "pose3d_run": pose3d_run()}


def check_same_source(extract_sig, session_sig, what: str) -> None:
    """REFUSE an extract sampled from another lick set / 3-D table than the session file it is read with. An extract
    written before 11-24 has no lick_sig (``extract_sig`` None): accepted, the caller's own alignment check (same licks,
    same 3-D landing) is then the guard."""
    if extract_sig is None:
        return
    s = str(session_sig) if session_sig not in (None, "") else LEGACY
    if str(extract_sig) != s:
        raise RuntimeError(f"{what}: built on {describe(str(extract_sig))}, but the session file is "
                           f"{describe(s)}. Move the extract folder aside and re-extract.")
