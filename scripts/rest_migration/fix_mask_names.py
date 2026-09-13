"""Rename the rebuilt masks to the TAGGED convention the rest of the pipeline constructs.

`recompute_masks` passed `--label <animal>_<mmdd>`, giving `PS92_0806_quiet_frame.npy`. The legacy
masks -- and every path `preprocess.py` builds -- use `<animal>_<mmdd>_<tag>`, i.e.
`PS92_0806_affine8v1_quiet_frame.npy`.

WHY IT MATTERS EVEN THOUGH THE RESOLVER COPES. `quiet_periods.quiet_frame_path` globs `*quiet_frame.npy`
and finds either name, so every ANALYSIS consumer was fine. But `preprocess.py` constructs the path
explicitly -- `qf = f"{quiet}/{lab}_quiet_frame.npy"` -- and passes it to `framemap_event_maps
--quiet-frame`. Under the new variant that file would not exist, so the imaging box's lick-map step
would fail (or silently skip its quiet-normalised panel) on every session. Found by diffing the two
directories' filenames rather than by running it.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from wfield_local import config
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.quiet_periods import quiet_dir

    tag = config.defaults()["preprocess"]["maps"]["tag"]
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    renamed, already, missing = 0, 0, []

    for s in SESSIONS:
        if s["label"] not in want:
            continue
        d = Path(quiet_dir(s["mc"]))
        if not d.exists():
            missing.append(s["label"])
            continue
        for kind in ("quiet_frame", "quiet_sample", "quiet_periods_summary"):
            ext = ".json" if kind.endswith("summary") else ".npy"
            good = d / f"{s['label']}_{tag}_{kind}{ext}"
            if good.exists():
                already += 1
                continue
            bad = d / f"{s['label']}_{kind}{ext}"
            if not bad.exists():
                continue
            if args.dry_run:
                print(f"  DRY {bad.name} -> {good.name}")
            else:
                os.rename(bad, good)
            renamed += 1
        # the QC png too, so the directory is self-consistent
        for p in glob.glob(f"{d}/{s['label']}_quiet_periods.png"):
            t = d / f"{s['label']}_{tag}_quiet_periods.png"
            if not t.exists():
                if args.dry_run:
                    print(f"  DRY {Path(p).name} -> {t.name}")
                else:
                    os.rename(p, t)

    print(f"renamed {renamed}, already correct {already}, "
          f"{len(missing)} session(s) without a variant directory")
    if missing:
        print("  " + ", ".join(missing[:10]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
