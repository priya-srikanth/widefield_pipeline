"""Does the DOCKED rest term produce a usable mask, and how much rest does it cost?

RUN BEFORE FLIPPING THE CONFIG, not after. Turning on `segmentation.rest.docked` plus
`variant: restdock` regenerates every mask and every behavior_events npz and then needs a full
re-render -- so the question "does every session survive it, and with how much rest left" has to be
answered on the cheap path first. The docked window is ~1.35 s against the loose ~2.0 s and it is
intersected with the behavioural terms, so the cost is not predictable from the window ratio.

Reports per session: loose rest fraction, docked rest fraction, the ratio, and which path the
docked window came from. A session that RAISES is a bug, not a bonus -- docked is a strict subset.
"""
from __future__ import annotations

import sys
import time

import numpy as np


def main() -> int:
    import h5py

    from wfield_local import config, daq_io
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS, _load_cue_events
    from wfield_local.plot_spout_trial_averages import _classify_cues
    from wfield_local.quiet_periods import quiet_dir

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")][:limit]

    from wfield_local.docked_periods import docked_mask_any
    from wfield_local.spout_behavior import discover_sessions

    t0 = time.time()
    rows, failed = [], []
    for s in todo:
        lab = s["label"]
        try:
            import glob
            qs = sorted(glob.glob(f"{quiet_dir(s['mc'])}/*quiet_sample.npy"))
            if not qs:
                failed.append(f"{lab}: no rest mask on disk")
                continue
            rest = np.load(qs[0]).astype(bool)
            with h5py.File(s["h5"], "r") as f:
                dn = [x.decode() for x in f["digital/channel_names"][:]]
                packed = f["digital/packed_samples"][:, 0]
            sync = daq_io.rising_edges((packed >> dn.index("sync")) & 1)
            ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
            cue = _load_cue_events(s["h5"])
            codes = np.asarray(_classify_cues(cue["cue_samples"], cue["strobe_samples"],
                                              cue["strobe_codes"]))
            cs = np.asarray(cue["cue_samples"], np.int64)
            an, mmdd = lab.split("_")[0], lab.split("_")[1]
            cands = discover_sessions(config.resolver(), f"2026{mmdd}", [an])
            dm, source = docked_mask_any(cands[0] if cands else None, sync, cs, codes, ts,
                                         rest.shape[0])
        except Exception as ex:                                        # noqa: BLE001
            failed.append(f"{lab}: {type(ex).__name__} {str(ex)[:60]}")
            continue
        if dm is None:
            failed.append(f"{lab}: NO DOCKED WINDOW -- would drop out")
            continue
        loose = float(rest.mean())
        dock = float((rest & dm[: rest.shape[0]]).mean())
        rows.append((lab, loose, dock, source))
        print(f"  .. {lab:<12} loose {loose:.4f}  docked {dock:.4f}  "
              f"kept {100 * dock / max(1e-12, loose):5.1f}%  [{source}]", flush=True)

    print(f"\n{'=' * 72}\nDOCKED REST TERM -- feasibility\n{'=' * 72}")
    print(f"{len(rows)} session(s) produced a docked window; {len(failed)} did not")
    for x in failed:
        print("   !!", x)
    if not rows:
        print("\nNOTHING TESTED -- a failed run, not a negative result.")
        return 1
    keep = np.array([r[2] / max(1e-12, r[1]) for r in rows])
    print(f"\nrest retained: mean {100 * keep.mean():.1f}%  "
          f"min {100 * keep.min():.1f}%  max {100 * keep.max():.1f}%")
    nrec = sum(1 for r in rows if r[3] == "reconstructed")
    print(f"reconstructed windows: {nrec}/{len(rows)}")
    if keep.max() > 1.0:
        print("!! a session RETAINED MORE than it started with -- docked is a strict subset, "
              "so this is a bug")
    print(f"[done in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
