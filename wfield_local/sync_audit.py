"""Audit imaging-frame synchronisation across every processed session.

THE QUESTION (Priya, 2026-08-21): "have there been dropped frames in other recordings? are we
syncing using sync pulse to ensure alignment?"

HOW ALIGNMENT ACTUALLY WORKS, because there are two clocks and only one of them is the Arduino:

  * The PCO imaging camera needs no sync pulse. Every exposure raises the DAQ digital
    ``pco_exposure`` line, so the frame train IS on the DAQ clock -- alignment is an index, not a
    fit. ``chosen_exposure_offset`` records the shift the frame map had to apply to line the two up.
  * The Blackfly behaviour cameras are on their own clock and DO use the Arduino heartbeat (GPIO
    bit0 against DAQ digital ``sync``), matched by ``camera_sync``/``frame_sync``. That is a
    different mechanism for a different camera, and conflating the two is the mistake this docstring
    exists to prevent.

WHAT THE AUDIT FOUND, 80 sessions, 2026-08-23:

  * ``chosen_exposure_offset`` is 0 in ALL 80. The exposure train and the .dat frames line up
    index-for-index; no session has ever needed a shift.
  * DAQ exposures minus physical frames: +1 in 69 sessions, 0 in 8, +2 in one, +3 in two. The +1 is
    the ordinary case -- the last exposure fires and the recording stops before that frame is
    written. The worst disagreement anywhere is THREE frames out of ~300-500k, i.e. 0.001%.
  * So there is no dropped-frame problem in the imaging stream, and no session's alignment rests on
    a fitted offset.

THE ONE REAL OUTLIER IS NOT A SYNC PROBLEM. PS95_0813 skips 119438 illuminated frames of 532220
(22.4%) -- that is the documented blue-LED-only prefix (docs/EXPERIMENT_ERRORS.md), where the 415
channel is missing for the first ~32 min so those frames cannot be paired. Its exposure count still
matches its frame count exactly. Median skipped elsewhere is 265.

Run: ``python -m wfield_local.sync_audit`` (add ``--json <path>`` to save the table).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from wfield_local import config

#: A disagreement bigger than this between DAQ exposures and written frames is worth a look.
#: The observed maximum over 80 sessions is 3; 10 is loose enough not to cry wolf and tight enough
#: that a genuinely dropped chunk cannot hide under it.
MAX_EXPECTED_DELTA = 10


def _label(path: str) -> str:
    """PSxx_MMDD from the path STRUCTURE, not from a literal root name.

    The summary always sits at ``<root>/<YYYYMMDD>/<PSxx_...>/motion_corrected/<file>``, so the two
    directories above it carry everything needed. Keying on the string "labcams" instead made this
    depend on where the tree happens to be mounted -- the class of assumption PathResolver exists to
    remove -- and when it missed it silently returned "motion_corrected" as the session label.
    """
    q = Path(path)
    sess, date = q.parents[1].name, q.parents[2].name
    animal = sess.split("_")[0]
    return f"{animal}_{date[4:]}" if len(date) == 8 and date.isdigit() else sess


def collect(labcams_root=None) -> list[dict]:
    """One record per session that has a cleanpairs summary."""
    root = labcams_root or config.resolver().root("labcams")
    out = []
    for f in sorted(glob.glob(str(Path(root) / "2026*" / "*" / "motion_corrected"
                                  / "*cleanpairs_summary.json"))):
        try:
            d = json.load(open(f))
        except Exception:                                            # noqa: BLE001
            continue
        daq, dat = d.get("daq_pco_exposure_count"), d.get("dat_physical_frame_count")
        if daq is None or dat is None:
            continue
        out.append({"label": _label(f), "daq_exposures": int(daq), "dat_frames": int(dat),
                    "delta": int(daq) - int(dat),
                    "clean_pairs": d.get("clean_pairs"),
                    "skipped_illuminated": d.get("skipped_illuminated_frames"),
                    "exposure_offset": d.get("chosen_exposure_offset")})
    return out


def audit(records) -> dict:
    """Summarise, and flag only what actually deserves it."""
    bad_delta = [r for r in records if abs(r["delta"]) > MAX_EXPECTED_DELTA]
    shifted = [r for r in records if r.get("exposure_offset") not in (0, None)]
    return {"n": len(records), "bad_delta": bad_delta, "shifted": shifted,
            "deltas": sorted({r["delta"] for r in records})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labcams", default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rec = collect(args.labcams)
    if not rec:
        print("no cleanpairs summaries found")
        return 1
    a = audit(rec)
    print(f"{a['n']} session(s) with both counts")
    for d in a["deltas"]:
        n = sum(1 for r in rec if r["delta"] == d)
        print(f"   DAQ exposures - written frames = {d:+d}   in {n} session(s)")
    print()
    print(f"exposure_offset values seen: {sorted({r['exposure_offset'] for r in rec})} "
          f"(0 means the frame map needed no shift)")
    sk = [r["skipped_illuminated"] for r in rec if r["skipped_illuminated"] is not None]
    if sk:
        worst = max(rec, key=lambda r: r["skipped_illuminated"] or 0)
        print(f"skipped illuminated frames: median {sorted(sk)[len(sk)//2]}, "
              f"worst {worst['label']} at {worst['skipped_illuminated']}")
    print()
    if a["shifted"]:
        print("!! sessions whose frame map needed a SHIFT (alignment was not index-for-index):")
        for r in a["shifted"]:
            print(f"   {r['label']}  offset={r['exposure_offset']}")
    if a["bad_delta"]:
        print(f"!! sessions disagreeing by more than {MAX_EXPECTED_DELTA} frames:")
        for r in a["bad_delta"]:
            print(f"   {r['label']}  daq={r['daq_exposures']} dat={r['dat_frames']} "
                  f"delta={r['delta']:+d}")
    if not a["shifted"] and not a["bad_delta"]:
        print("No session needed an alignment shift, and none disagrees by more than "
              f"{MAX_EXPECTED_DELTA} frames.")
    if args.json:
        args.json.write_text(json.dumps(rec, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
