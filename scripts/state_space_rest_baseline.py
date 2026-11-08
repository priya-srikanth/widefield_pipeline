"""RESTnl rest baseline per session in the cohort SVD basis, for the state-space maps (Priya 11-08: "let's try the rest
baseline and no subtraction").

THE DEFINITION IS `wfield_local.restnl_mask` (the primary rest reference of the imaging-page maps since 11-01): the
restdock05 rest sample mask minus every DAQ and cam4 (licks_all) lick, widened [0.5 s before, 2.0 s after], engaged
period only, frames per spout position (`frames_by_position`), per-position MEDIAN, rule `wr` (equal-weight mean over
positions with >= 50 frames, >= 4 of 6). Its functions are called as they are; only the SIGNAL differs: the state-space
per-trial data are the SVTcorr with the slow drift projected out (cosines >= 600 s INCLUDING the constant, as the O2
extraction: `combined_encoding._slow_basis`), so the level is taken on that same drift-removed signal -- the cached
RESTnl baselines (raw SVT) would mix two references. b (K, the session's own SVD) -> cohort coordinates
c = U_cohort^T (U_s b) on the analysis mask (Allen brain - bulbs - painted glue).

Run under the rebuild profile (the rows / imaging / licks_all of the rebuild run), like `o2_state_space_lnmf._rows_worker`.
Sessions without licks_all.csv or failing rule wr get ok=False (the map then drops them for the rest reference).
Output: <state_space_recovery>/rest_baseline_svd/<label>.npz (c_rest (K_cohort,), ok, nfr_pos, positions_used, info).

    python scripts/state_space_rest_baseline.py [--jobs 3] [LABELS ...]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o2_state_space_lnmf as L  # noqa: E402
import state_space_sim as S  # noqa: E402

OUT = S.ROOT / "rest_baseline_svd"
BUNDLE = S.SVD.parents[1]


def _worker(label):
    """Module-level (spawn): one session -> its arrays. The profile's write fence forbids writing outside the rebuild
    root inside this process, so the parent (no profile) writes OUT/<label>.npz."""
    L._env(L.CACHE)
    from wfield_local import combined_encoding as ce
    from wfield_local import restnl_mask as rn
    meta = json.loads((BUNDLE / "bundle.json").read_text())
    src = Path(meta["lnmf_scratch"]) / "inputs" / f"{label}.pkl"
    if not src.exists():
        src = Path(meta["enc_scratch"]) / "inputs" / f"{label}.pkl"
    x = pickle.loads(src.read_bytes())
    V, ft = np.asarray(x["imaging"][0], np.float64), np.asarray(x["imaging"][1], float)
    del x
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    V -= Q @ (Q.T @ V)                                                  # (T, K) drift-removed, as the extraction
    s = rn.session_of(label)
    try:
        mask, parts, info = rn.lickfree_mask(s)
    except FileNotFoundError as ex:
        return {"label": label, "ok": False, "reason": str(ex)}
    per = rn.frames_by_position(s, V.shape[0], mask, parts)
    pm, nfr = rn.position_medians(V.T, per)
    b, ok, used = rn.combine(pm, nfr, "wr")
    msk = np.load(BUNDLE / "mask.npy").ravel().astype(bool)
    U = np.load(Path(meta["uatlas_root"]) / meta["uatlas"][label])
    u = np.nan_to_num(U.reshape(-1, U.shape[2])).astype(np.float64)[msk]
    Uc = np.load(S.SVD.parent / "basis" / "U_cohort.npy").astype(np.float64)
    c = Uc.T @ (u @ np.nan_to_num(b)) if ok else np.full(Uc.shape[1], np.nan)
    return {"label": label, "ok": bool(ok), "c_rest": c, "nfr_pos": nfr, "positions_used": np.asarray(used, int),
            "info": json.dumps(info)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = a.labels or [x for x in (BUNDLE / "labels.txt").read_text().split() if x]
    todo = [lab for lab in labels if not (OUT / f"{lab}.npz").exists()]
    from wfield_local import analysis_kit as kit
    if a.jobs > 1 and len(todo) > 1:
        got, fail = kit.fan_sessions(todo, _worker, jobs=a.jobs, key=lambda x: x)
    else:                         # never in THIS process: the profile's write fence would block the writes below
        from concurrent.futures import ProcessPoolExecutor
        got, fail = [], []
        with ProcessPoolExecutor(1) as ex:
            for lab in todo:
                try:
                    got.append((lab, ex.submit(_worker, lab).result()))
                except Exception as e:                                 # noqa: BLE001
                    fail.append((lab, e))
    for _i, r in got:
        np.savez(OUT / f"{r['label']}.npz", **{k: v for k, v in r.items() if k != "label"})
    bad = [r["label"] for _i, r in got if not r["ok"]]
    print(f"{len(got)} done, {len(bad)} without a usable rest baseline {bad}, {len(fail)} failed "
          f"{[(i, str(e)[:80]) for i, e in fail]}")


if __name__ == "__main__":
    main()
