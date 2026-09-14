"""Print EXACTLY what the LocaNMF / Allen-ROI arm reads, resolved through the real config.

Not a reasoning exercise: resolves `config.svtcorr_path` and `config.locanmf_dir` for real sessions,
checks the files exist, and prints the variant manifest that records how each product was built.

    python -m scripts.rest_migration.what_locanmf_reads [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wfield_local import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    a = ap.parse_args()

    print("CONFIG SWITCHES")
    print(f"  defaults hemo.variant      = {(config.defaults().get('hemo') or {}).get('variant')!r}")
    print(f"  defaults hemo.build_variant= {(config.defaults().get('hemo') or {}).get('build_variant')!r}")
    print(f"  env WIDEFIELD_HEMO_VARIANT = {os.environ.get('WIDEFIELD_HEMO_VARIANT')!r}")
    print(f"  config.hemo_variant()      = {config.hemo_variant()!r}")
    print(f"  config.locanmf_dir_name()  = {config.locanmf_dir_name()!r}")
    print()

    sessions = config.load_sessions()
    shown = 0
    for s in sessions:
        mc = s.get("mc")
        if not mc:
            continue
        svt = Path(config.svtcorr_path(mc))
        bare = Path(mc) / "wfield_local_results" / "SVTcorr.npy"
        ldir = Path(config.locanmf_dir(mc))
        print(f"{s['label']}")
        print(f"  READS   {svt}")
        print(f"          exists={svt.exists()}  "
              f"{svt.stat().st_size/1e6:.1f} MB" if svt.exists() else "          exists=False")
        print(f"  BARE    {bare}  exists={bare.exists()}   <- zerophase product, NOT read")
        print(f"  LocaNMF {ldir}  exists={ldir.exists()}")
        man = svt.parent / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            keep = ("variant", "refit_t", "drift", "fit_drift", "mask", "mask_frac",
                    "meegkit_order", "win_s", "freq_highpass", "freq_lowpass",
                    "functional_channel")
            print("  MANIFEST " + "  ".join(f"{k}={m.get(k)!r}" for k in keep))
        shown += 1
        if shown >= a.limit:
            break


if __name__ == "__main__":
    main()
