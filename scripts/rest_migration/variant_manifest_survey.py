"""Survey the hemo-variant manifests the LocaNMF/ROI arm actually reads.

Reports, per session, the drift-fit MASK FRACTION -- the share of frames the order-10 polynomial is
allowed to see. The `strobedetrend` mask removes whole trials, so what survives is essentially the
ITI. A low and UNIFORM fraction is fine (11 free parameters against thousands of well-spread
samples); a low and CLUSTERED one would let the polynomial extrapolate through masked stretches.

    python -m scripts.rest_migration.variant_manifest_survey
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wfield_local import config


def main():
    rows, missing = [], []
    for s in config.load_sessions():
        mc = s.get("mc")
        if not mc:
            continue
        man = Path(config.svtcorr_path(mc)).parent / "manifest.json"
        if not man.exists():
            missing.append(s["label"])
            continue
        try:
            m = json.loads(man.read_text())
        except Exception as ex:                                   # noqa: BLE001
            missing.append(f"{s['label']} (unreadable: {ex})")
            continue
        rows.append((s["label"], m.get("mask_frac"), m.get("meegkit_order"),
                     m.get("refit_t"), m.get("variant"), m.get("fit_drift"),
                     m.get("functional_channel")))

    print(f"{len(rows)} sessions with a variant manifest; {len(missing)} without\n")
    fr = np.array([r[1] for r in rows if r[1] is not None], float)
    if fr.size:
        print(f"MASK FRACTION (share of frames the drift polynomial may fit)")
        print(f"  min {fr.min()*100:.2f}%   median {np.median(fr)*100:.2f}%   "
              f"max {fr.max()*100:.2f}%   mean {fr.mean()*100:.2f}%")
        print(f"  below 2%: {int((fr < 0.02).sum())} sessions")
    print("\nHOMOGENEITY of the other fields (any spread here would mean two incompatible products):")
    for i, name in ((2, "meegkit_order"), (3, "refit_t"), (4, "variant"),
                    (5, "fit_drift"), (6, "functional_channel")):
        vals = {}
        for r in rows:
            vals[r[i]] = vals.get(r[i], 0) + 1
        print(f"  {name:20s} {vals}")

    worst = sorted((r for r in rows if r[1] is not None), key=lambda r: r[1])[:8]
    print("\nLOWEST mask fractions:")
    for lab, f, *_ in worst:
        print(f"  {lab:12s} {f*100:.2f}%")
    if missing:
        print(f"\nNO MANIFEST ({len(missing)}): {', '.join(missing[:20])}"
              + (" ..." if len(missing) > 20 else ""))


if __name__ == "__main__":
    main()
