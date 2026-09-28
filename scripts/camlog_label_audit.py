"""Audit every session's ASSIGNED per-frame channel labels against the camera's own camlog LED record.

    python -m scripts.camlog_label_audit            # every session with a frame map + camlog on the share

WHAT IT COMPARES. The relabel step labels each DAT frame from the DAQ's LED TTLs measured during that
frame's exposure pulse (`trim_illuminated_labcams.load_daq_labels`) and stores them in the frame map as
`labels_per_original_frame`. labcams writes a `#LED:<id>,<n>,<t>` line per LED command into the camlog.
The two are independent records of the same alternation, so index-wise agreement should be ~100%.

WHAT IT FOUND (2026-09-28, 124 sessions): 116 agree on 100.00000% of frames, whole session, including
PS92_0922 at its head offset of 154. The eight that do not are of two kinds, neither a pipeline error:
  * camlogs at chance from the first frame (PS92_0605, and the early-June trial-gated sessions) --
    labcams not writing one #LED line per frame; the DAQ covers the whole DAT and stays authoritative;
  * camlogs that DIVERGE mid-session (PS93_0805 at frame 180,080; PS92_0904 at 429,023; PS92_0911 at
    421,567; PS93_0904 at 84,614; PS94_0817 repeatedly). At each point the camlog carries two #LED
    lines for one frame: labcams' LED toggle ran twice within one exposure, the DAQ saw both LEDs in
    that exposure (label 3, skipped by the pairing) and then measured the true LED on every later
    exposure. Cue-triggered responses of the two RAW channels before vs after each divergence are
    unchanged (PS92_0904 470: +0.041 -> +0.040, 415: +0.010 -> +0.009; PS93_0805 470: +0.013 ->
    +0.017, 415: +0.003 -> +0.002), so the pipeline's labels are right and the camlog's index is off.

WHAT THIS MEANS FOR THE RULES. The ~150-300 "alternation hiccups" per session are LED-driver repeats
the DAQ measures per exposure; `make_clean_pairs` skips the unpaired frame, so a hiccup costs one frame
and never a mislabel. The camlog is a software toggle log: decisive for placing a HEAD surplus (a
parity-neutral index error does not disturb that search, and PS92_0922's 154 was found with it), a
check and no more everywhere else -- exactly how `load_daq_labels` treats it.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from wfield_local.trim_illuminated_labcams import camlog_led_ids


def audit_session(frame_map: str, camlog: str) -> dict:
    z = np.load(frame_map)
    lpf = z["labels_per_original_frame"].astype(int)
    head = int(z["dat_head_offset"]) if "dat_head_offset" in z.files else 0
    ids = camlog_led_ids(camlog)
    n = min(len(lpf), len(ids))
    L, ids_n = lpf[:n], ids[:n]
    lit = L != 0
    if not lit.any():
        return {"error": "no lit frames"}
    # camlog id -> wavelength by best agreement over lit frames; polarity is not assumed
    best = None
    for a in np.unique(ids_n[lit]):
        for b in np.unique(ids_n[lit]):
            if a == b:
                continue
            pred = np.where(ids_n == a, 415, np.where(ids_n == b, 470, -1))
            agree = float((pred[lit] == L[lit]).mean())
            if best is None or agree > best[0]:
                best = (agree, pred)
    agree, pred = best
    lit_idx = np.flatnonzero(lit)
    bad = lit_idx[pred[lit] != L[lit]]
    tail = lit_idx[int(0.9 * len(lit_idx)):]
    repeats_daq = int((L[lit_idx[1:]] == L[lit_idx[:-1]]).sum())
    return {"frames": len(lpf), "camlog_lines": len(ids), "head": head, "agree": agree,
            "agree_last10": float((pred[tail] == L[tail]).mean()), "n_disagree": int(len(bad)),
            "first_bad": int(bad[0]) if len(bad) else -1, "label_repeats_daq": repeats_daq,
            "skipped": int(len(z["skipped_original_frame_index"])) if "skipped_original_frame_index" in z.files else -1}


def main() -> int:
    from wfield_local.paths import PathResolver
    root = PathResolver().root("labcams")
    maps = sorted(glob.glob(f"{root}/*/PS9*/motion_corrected/*cleanpairs_frame_map.npz"))
    n_ok = 0
    for fm in maps:
        sess_dir = os.path.dirname(os.path.dirname(fm))
        sess = os.path.basename(sess_dir)
        lab = f"{sess[:4]}_{sess[9:13]}"
        cams = glob.glob(os.path.join(sess_dir, "raw_widefield_data", "*.camlog"))
        if not cams:
            print(f"{lab}  no camlog")
            continue
        try:
            r = audit_session(fm, cams[0])
        except Exception as ex:                                             # noqa: BLE001
            print(f"{lab}  {type(ex).__name__}: {str(ex)[:80]}")
            continue
        if "error" in r:
            print(f"{lab}  {r['error']}")
            continue
        ok = r["agree"] == 1.0
        n_ok += ok
        print(f"{lab}  frames {r['frames']:7d}  camlog {r['camlog_lines']:7d}  head {r['head']:3d} | agree {100*r['agree']:9.5f}%  "
              f"last10% {100*r['agree_last10']:9.5f}%  n_disagree {r['n_disagree']:6d}  first_bad {r['first_bad']:7d} | "
              f"DAQ label-repeats {r['label_repeats_daq']:5d}  skipped {r['skipped']:5d}" + ("" if ok else "   <-- see module docstring"))
    print(f"\n{n_ok}/{len(maps)} sessions agree with the camlog on every frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
