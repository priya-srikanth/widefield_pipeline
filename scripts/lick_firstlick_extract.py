"""Re-extract EVERY first post-cue lick (no 1500-lick cap) into a SEPARATE cache (Priya 10-29 "do the re-extraction").

iti_licks.extract keeps a seeded random N_TRIAL_MAX = 1500 of each session's 3-D tracked post-cue licks for the decoder
(`tr_*` in the session files), so only ~25-65 % of first post-cue licks reached the balance, unevenly across epochs
(acute bouts are shorter; session_pace_table.py). The shared session files and the decoder's lick set are NOT changed
(every decoder result depends on them); this writes <OUT_W>/firstlick_full/<label>.npz with, for every 3-D tracked
post-cue lick that is the FIRST lick of its bout (lo._bout_idx == 0, the order rule of the decoder's lick set):
  X (n, len(LAGS_D), cols) -- the same area traces as iti_licks.extract (same ROI masks, drift removal, x1e3, float16),
  t (exact DAQ peak), side_ml, reach_mm, speed_peak, elev_deg, pos (target spout), epoch, animal.
Validation per session: licks also present in the capped set must reproduce its tr_X exactly (max |diff| reported).
Usage: python scripts/lick_firstlick_extract.py [--jobs N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402

OUT = il.OUT_W / "firstlick_full"


def extract(label: str) -> dict:
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    f = OUT / f"{label}.npz"
    if f.exists():
        return {"label": label, "cached": True}
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    K = pd.read_csv(Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz",
                    usecols=["session", "context", "lick_uid", "side_ml", "out_ap", "down_dv", "position", "reach_mm",
                             "speed_peak"])
    K = K[K.session == f"{an}_{date}"].copy()
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["context", "daq_peak_s"])
    uid = K.lick_uid.to_numpy(int)
    if not (A.context.to_numpy()[uid] == K.context.to_numpy()).all():
        raise RuntimeError(f"{label}: lick_uid join to licks_all.csv does not match on context")
    K["daq_peak_s"] = A.daq_peak_s.to_numpy(float)[uid]
    dlo, dhi = ft[0] - il.LAGS_D[0], ft[-1] - il.LAGS_D[-1]
    tr = K[(K.context == "post_cue") & np.isfinite(K.side_ml) & (K.daq_peak_s > dlo) & (K.daq_peak_s < dhi)]
    tr = tr.sort_values("daq_peak_s").drop_duplicates("daq_peak_s")
    order = lo._bout_idx(label, tr.daq_peak_s.to_numpy(float))
    first = tr[order == 0]
    t = first.daq_peak_s.to_numpy(float)
    X = il._sample(R, ft, t, il.LAGS_D).astype(np.float16)
    elev = np.degrees(np.arctan2(first.down_dv.to_numpy(float), np.hypot(first.side_ml.to_numpy(float),
                                                                          first.out_ap.to_numpy(float))))
    # validation against the capped decoder set
    d = il._load(label)
    tt = d["tr_t"].astype(float)
    common, i_new, i_old = np.intersect1d(t, tt, return_indices=True)
    dmax = float(np.nanmax(np.abs(X[i_new].astype(float) - d["tr_X"][i_old].astype(float)))) if len(common) else np.nan
    if len(common) and list(d["cols"]) != cols:
        raise RuntimeError(f"{label}: column order differs from the session file")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, cols=np.array(cols), X=X, t=t, side_ml=first.side_ml.to_numpy(float),
                        reach_mm=first.reach_mm.to_numpy(float), speed_peak=first.speed_peak.to_numpy(float),
                        elev_deg=elev, pos=first.position.to_numpy(dtype=str), epoch=epochs.epoch_of(label), animal=an)
    return {"label": label, "n_first_full": len(t), "n_first_in_capped": len(common), "max_abs_diff_vs_capped": dmax}


def main() -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT)
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
    labels = sorted(pd.read_csv(lo.preds_path(), usecols=["label"]).label.unique())
    res, fail = ak.fan_sessions(labels, extract, jobs=jobs, label="session")
    if fail:
        print(f"!! {len(fail)} failed: {fail[:5]}")
    S = pd.DataFrame([v for _k, v in res])
    S.to_csv(il.OUT_W / "firstlick_full_summary.csv", index=False)
    print(S.describe().round(3).to_string())
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
