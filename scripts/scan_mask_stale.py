"""READ-ONLY: which existing per-session .npz cache folders the 11-09 mask guards would refuse (GLUE_MASK_FIX_2026-11-09).

Prints ``stale/total<TAB>folder`` for every folder matching the globs under ROOT, where stale = .npz files whose
``mask_sig`` is missing or differs from SIG (the current `mask_keys.signature()`). Reads only each zip's index and the
small mask_sig member; never writes, moves or deletes.

Usage: python scripts/scan_mask_stale.py ROOT SIG GLOB [GLOB ...]
  e.g. python scripts/scan_mask_stale.py <iti_licks run folder> 5f9eb36f36b2 "*" "*/*" "*/*/*"
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np


def npz_sig(f: Path) -> str | None:
    try:
        with zipfile.ZipFile(f) as z:
            if "mask_sig.npy" not in z.namelist():
                return None
            return str(np.load(io.BytesIO(z.read("mask_sig.npy"))).item())
    except Exception as ex:                                            # noqa: BLE001
        return f"ERR {type(ex).__name__}"


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) < 3:
        print(__doc__)
        return 2
    root, sig, pats = Path(a[0]), a[1], a[2:]
    for pat in pats:
        for d in sorted(root.glob(pat)):
            if not d.is_dir():
                continue
            fs = sorted(d.glob("*.npz"))
            if fs:
                print(f"{sum(1 for f in fs if npz_sig(f) != sig)}/{len(fs)}\t{d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
