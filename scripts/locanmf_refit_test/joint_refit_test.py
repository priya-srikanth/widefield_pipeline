"""Joint-basis refit on the corrected (swapmask) data for ONE animal, timed, compared with the current basis.

Production procedure = `joint_locanmf.build` (joint SVD over the basis's own session set, rank resolved as in
production, seed 0, LocaNMF params from defaults.yaml) with WIDEFIELD_HEMO_VARIANT pointing at the corrected
SVTcorr. Written to a SEPARATE out_dir (LOCAL_OUT/joint/<animal>/<bid>), never under the basis store, so
`joint_locanmf.load()` cannot pick it up; the script checks that load() still returns the current basis afterwards.

Compares: footprints (Hungarian-matched spatial r), and per session the new basis's fitted C against the CURRENT
basis's projection of the same corrected session (the joint analogue of refit vs projection).

    WIDEFIELD_HEMO_VARIANT=meegkit_hpfit_lag_refitT_swapmask python joint_refit_test.py PS94
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from refit_vs_projection import LOCAL_OUT, SHARE_OUT, VARIANT, match  # noqa: E402

from wfield_local import joint_locanmf as jl  # noqa: E402
from wfield_local.locanmf_cue_lick_analysis import SESSIONS  # noqa: E402


def main(animal):
    assert os.environ.get("WIDEFIELD_HEMO_VARIANT") == VARIANT, "set WIDEFIELD_HEMO_VARIANT to the swapmask variant"
    assert os.environ.get("WIDEFIELD_SESSION_CACHE", "").endswith("locanmftest")
    assert not os.environ.get("WIDEFIELD_JOINT_BASIS_DIR")
    cur = jl.load(animal)
    cur_id = cur.basis_id
    sessions = [next(s for s in SESSIONS if s["label"] == lab) for lab in cur.labels]
    bid = jl.basis_id(sessions, None)
    out_dir = LOCAL_OUT / "joint" / animal / bid
    for root in (jl.BASIS_DIR, jl.server_basis_dir()):
        assert root is None or not str(out_dir).lower().startswith(str(root).lower()), out_dir
    print(f"[joint] {animal}: current {cur!r}; new id {bid} -> {out_dir}", flush=True)
    t0 = time.time()
    new = jl.build(animal, sessions, rank=None, out_dir=out_dir)
    total = time.time() - t0
    print(f"[joint] built {new!r} in {total:.0f}s (LocaNMF {new.manifest['fit_seconds']}s)", flush=True)

    A0 = np.asarray(cur.A); A1 = np.asarray(new.A)
    inside = np.isfinite(A0.reshape(-1, A0.shape[2])).any(1) | np.isfinite(A1.reshape(-1, A1.shape[2])).any(1)
    # footprints + concatenated in-fit C (stored current C on the ADOPTED data vs new C on the corrected data)
    C0 = np.concatenate([np.load(cur.root / f"C_{lab}.npy") for lab in cur.labels], axis=1)
    C1 = np.concatenate([np.load(new.root / f"C_{lab}.npy") for lab in new.labels], axis=1)
    res = {"animal": animal, "variant": VARIANT, "current_basis": cur_id, "current_ncomp": cur.ncomp,
           "new_basis": new.basis_id, "new_dir": str(new.root), "new_ncomp": new.ncomp,
           "n_sessions": len(new.labels), "build_seconds_total": round(total, 1),
           "locanmf_seconds": new.manifest["fit_seconds"], "rank": new.manifest["rank"],
           "vs_current_storedC": match(A0, C0, A1, C1, inside)}
    del C0
    # current basis PROJECTION of the corrected sessions vs the new basis's fitted C
    os.environ[jl.PROJECT_ALL_ENV] = "1"
    # project() returns footprint-scaled rows; a per-row scale does not change a correlation
    Cp = np.concatenate([cur.project(s) for s in sessions], axis=1)
    res["vs_current_projection"] = match(A0, Cp, A1, C1, inside)
    del Cp
    # load() must still give the current basis
    after = jl.load(animal)
    res["load_after"] = after.basis_id
    res["load_unchanged"] = after.basis_id == cur_id
    for root in (LOCAL_OUT / "results", SHARE_OUT / "results"):
        root.mkdir(parents=True, exist_ok=True)
        (root / f"joint_{animal}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
