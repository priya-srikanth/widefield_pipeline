"""FaceRhythm per-point face movement maps (Fig 5B, Supplementary S13) on the FOUR-CAMERA basis (Priya 11-23: every
six-position analysis uses the four-camera lick list, trial classes and gates; FaceRhythm itself is 2-D video, so the
point speeds are NOT recomputed from video, only re-windowed).

Same per-point speed as `face_rhythm_asym_aligned.py` (2-D image speed of each mesh point from the session's frozen
20 point PCs, `face_rhythm.face`), same windows (cue 0..+2 s; lick -0.1..+0.2 s, per lick), but the EVENTS change:

  cue_eng       production cue set: the FR bundle's engaged trials (behavior-pipeline engagement flag; `proj/` cues).
                Plumbing check: must equal the production cache (asym_aligned/data/aligned_point_maps.npz, cue).
  lick_v36eng   production lick set: LP v36 licks (`face_rhythm.licks.lick_table`) in cue -2..+4 s of an engaged cue.
                Plumbing check against the production cache (lick).
  cue_wp        FOUR-CAMERA: every cue inside the camera working period (`trial_classes.OUT_OF_WORKING_PERIOD`, final
                class of `imaging_encoding/qc/s1_trial_classes_cue_stop.csv.gz`; trials without video dropped).
  lick_4c_eng   fused four-camera licks (`session_licks.session_licks`, cam1-only events dropped) in cue -2..+4 s of an
                engaged cue: isolates the lick-list change.
  lick_4c_wp    FOUR-CAMERA primary: fused licks in cue -2..+4 s of a working-period cue (the production window rule).
  lick_4c_wpp   as lick_4c_wp, post-cue licks only (0..+4 s): sensitivity.

A lick belongs to the latest cue of the session's FULL cue train (class table) with cue - 2 s <= lick onset, and is
kept when it starts < 4 s after that cue and the cue is in the set; its position is that trial's spout position.

It also writes per-session 5E side-template candidates: per spout position, the RMS point speed 0..+4 s after the cue
(PC reconstruction) over engaged cues (``tmpl_eng``, compare with the stored full-point ``speed_response``) and over
working-period cues (``tmpl_wp``).

    python scripts/face_rhythm_aligned_fourcam.py [--jobs 2] [--keys PS92_20260806_cam4 ...]
-> <readouts>/<set>/aligned_fourcam_ccB1123/data/aligned_point_maps_fourcam.npz (maps (sessions, aligns, 6, P),
   counts, templates), data/event_counts_fourcam.csv (per session x position: events per set).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.face_rhythm import face as frface  # noqa: E402

CLASS_TABLE = ("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/"
               "qc/s1_trial_classes_cue_stop.csv.gz")
OUT_TAG = "aligned_fourcam_ccB1123"
POSITIONS = frc.POSITIONS
CUE_WIN, LICK_WIN, TMPL_WIN = (0.0, 2.0), (-0.1, 0.2), (0.0, 4.0)
ALIGNS = ("cue_eng", "lick_v36eng", "cue_wp", "lick_4c_eng", "lick_4c_wp", "lick_4c_wpp")
WIN = {a: (CUE_WIN if a.startswith("cue") else LICK_WIN) for a in ALIGNS}
MATCH_S = 0.05
#: classes outside the camera working period (`trial_classes.OUT_OF_WORKING_PERIOD`) plus trials without video
NOT_WP = ("stopped", "stopped_licking", "early_run", "after_end", "unknown", "no_video")


def assign_licks(t_on: np.ndarray, cue_all: np.ndarray, keep_cue: np.ndarray, lo: float = -2.0,
                 hi: float = 4.0) -> np.ndarray:
    """Index into ``cue_all`` (sorted) of each lick's trial, -1 when the lick is outside [cue + lo, cue + hi) of the
    latest cue with cue + lo <= onset, or that cue is not kept. Pure."""
    k = np.searchsorted(cue_all, t_on - lo, side="right") - 1
    kk = np.clip(k, 0, None)
    rel = t_on - cue_all[kk]
    ok = (k >= 0) & (rel >= lo) & (rel < hi)
    ok &= keep_cue[kk]
    return np.where(ok, k, -1)


def match_cues(cue: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Index into sorted ``ref`` of the nearest cue within MATCH_S, -1 otherwise. Pure."""
    if len(ref) == 0:
        return np.full(len(cue), -1)
    hi = np.clip(np.searchsorted(ref, cue), 0, len(ref) - 1)
    lo = np.clip(hi - 1, 0, None)
    j = np.where(np.abs(ref[lo] - cue) <= np.abs(ref[hi] - cue), lo, hi)
    return np.where(np.abs(ref[j] - cue) <= MATCH_S, j, -1)


def speed_sums(pcs, t, comps, W1, W2):
    """Per-point weighted sums of speed (W1 columns) and of speed^2 (W2 columns); NaN frames skipped."""
    dt = np.diff(t)
    P = comps.shape[1] // 2
    S1, S2 = np.zeros((P, W1.shape[1])), np.zeros((P, W2.shape[1]))
    for c0 in range(1, len(t), frface.CHUNK):
        c1 = min(c0 + frface.CHUNK, len(t))
        dv = (pcs[c0:c1] - pcs[c0 - 1:c1 - 1]) / dt[c0 - 1:c1 - 1, None]
        sp = np.sqrt(((dv @ comps).reshape(-1, P, 2) ** 2).sum(-1))
        ok = np.isfinite(sp).all(1)
        s = sp[ok].T.astype(np.float64)
        S1 += s @ W1[c0:c1][ok]
        S2 += (s ** 2) @ W2[c0:c1][ok]
    return S1, S2


def job(item: tuple) -> dict:
    fdir, key = item
    from wfield_local import session_licks as SL
    from wfield_local.face_rhythm import licks as frl
    from wfield_local.paths import PathResolver
    fdir = Path(fdir)
    comps, _ = frface.master_points(fdir)
    with np.load(fdir / "proj" / f"{key}.npz", allow_pickle=True) as z:
        animal, date, epoch = str(z["animal"]), str(z["date"]), str(z["epoch"])
        pcs, t = z["points_pc"].astype(np.float32), z["points_t_daq_s"].astype(float)
        cues, pos = z["cue_daq_s"].astype(float), z["position"].astype(str)
        have = list(z["power_positions"].astype(str))
        sresp = np.full((6, z["speed_response"].shape[1]), np.nan)
        for i, p in enumerate(POSITIONS):
            if p in have:
                sresp[i] = z["speed_response"][have.index(p)]
    label = frc.label(animal, date)
    C = pd.read_csv(CLASS_TABLE, usecols=["label", "cue_s", "cls_final", "position"])
    C = C[C.label == label].sort_values("cue_s")
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    C = C[(C.cue_s >= t0) & (C.cue_s <= t1)]              # split videos: the cues this video covers
    cue_all = C.cue_s.to_numpy(float)
    pos_all = C.position.astype(str).to_numpy()
    wp = ~np.isin(C.cls_final.astype(str).to_numpy(), NOT_WP)
    # engaged (production) cues on the full train
    o = np.argsort(cues)
    cues, pos = cues[o], pos[o]
    j_eng = match_cues(cues, cue_all)
    eng = np.zeros(len(cue_all), bool)
    eng[j_eng[j_eng >= 0]] = True
    events = {}
    # production v36 licks (as face_rhythm_asym_aligned.job)
    # (`lick_table` reads the fused train since 20bc145f; the production maps were made on the legacy cam4 v36 set)
    import os
    old = os.environ.get(SL.FUSED_ENV)
    os.environ[SL.FUSED_ENV] = "off"
    try:
        L = frl.lick_table(PathResolver(), animal, date, cues)
    finally:
        if old is None:
            os.environ.pop(SL.FUSED_ENV, None)
        else:
            os.environ[SL.FUSED_ENV] = old
    L = L[np.isfinite(L.t_from_cue)]
    # fused four-camera licks (cam1-only dropped)
    A = SL.session_licks(label)
    ton = np.sort(A.daq_on_s.to_numpy(float)[np.isfinite(A.daq_on_s.to_numpy(float))])
    k_eng = assign_licks(ton, cue_all, eng)
    k_wp = assign_licks(ton, cue_all, wp)
    k_wpp = assign_licks(ton, cue_all, wp, lo=0.0)
    for p in POSITIONS:
        events[("cue_eng", p)] = cues[pos == p]
        events[("lick_v36eng", p)] = L[L.position.astype(str) == p].daq_on_s.to_numpy(float)
        events[("cue_wp", p)] = cue_all[wp & (pos_all == p)]
        for nm, k in (("lick_4c_eng", k_eng), ("lick_4c_wp", k_wp), ("lick_4c_wpp", k_wpp)):
            sel = k >= 0
            events[(nm, p)] = ton[sel][pos_all[k[sel]] == p]
    okf = np.r_[False, np.isfinite(pcs[1:]).all(1) & np.isfinite(pcs[:-1]).all(1)]
    cols = [(a, p) for a in ALIGNS for p in POSITIONS]
    W1 = np.column_stack([frface.window_weights(t, events[c], WIN[c[0]], okf) for c in cols])
    tcols = [(a, p) for a in ("cue_eng", "cue_wp") for p in POSITIONS]
    W2 = np.column_stack([frface.window_weights(t, events[c], TMPL_WIN, okf) for c in tcols])
    S1, S2 = speed_sums(pcs, t, comps, W1, W2)
    w1, w2 = W1.sum(0), W2.sum(0)
    maps = np.full((len(ALIGNS), 6, S1.shape[0]), np.nan, np.float32)
    counts = np.zeros((len(ALIGNS), 6), int)
    for j, (a, p) in enumerate(cols):
        ai, pi = ALIGNS.index(a), POSITIONS.index(p)
        counts[ai, pi] = len(events[(a, p)])
        if w1[j] > 0:
            maps[ai, pi] = S1[:, j] / w1[j]
    tmpl = np.full((2, 6, S1.shape[0]), np.nan, np.float32)
    for j, (a, p) in enumerate(tcols):
        if w2[j] > 0:
            tmpl[0 if a == "cue_eng" else 1, POSITIONS.index(p)] = np.sqrt(S2[:, j] / w2[j])
    n_v36_all = int(len(L))
    return {"key": key, "animal": animal, "date": date, "epoch": epoch, "label": label, "maps": maps,
            "counts": counts, "tmpl": tmpl, "speed_response": sresp.astype(np.float32),
            "n_cues_all": len(cue_all), "n_eng": int(eng.sum()), "n_eng_unmatched": int((j_eng < 0).sum()),
            "n_wp": int(wp.sum()), "n_eng_and_wp": int((eng & wp).sum()), "n_v36": n_v36_all,
            "n_4c_eng": int((k_eng >= 0).sum()), "n_4c_wp": int((k_wp >= 0).sum()), "n_4c_wpp": int((k_wpp >= 0).sum())}


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--keys", nargs="*", default=None, help="only these proj keys (timing / test); writes *_test.npz")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, fro.BUNDLE, fro.SET)
    out = frc.readouts_dir(fro.SET, rv) / OUT_TAG / "data"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    keys = a.keys or frc.session_keys(fdir)
    tag = "_test" if a.keys else ""
    npz = out / f"aligned_point_maps_fourcam{tag}.npz"
    if npz.exists():
        raise SystemExit(f"refusing to overwrite {npz}")
    got, fail = ak.fan_sessions([(str(fdir), k) for k in keys], job, jobs=a.jobs, key=lambda it: it[1],
                                label="fr4c")
    R = [r for _, r in got]
    if fail:
        print("FAILED:", [f[0] for f in fail], flush=True)
    M = np.load(fdir / "master.npz")
    np.savez(npz, maps=np.stack([r["maps"] for r in R]), counts=np.stack([r["counts"] for r in R]),
             tmpl=np.stack([r["tmpl"] for r in R]), speed_response=np.stack([r["speed_response"] for r in R]),
             key=np.array([r["key"] for r in R]), label=np.array([r["label"] for r in R]),
             animal=np.array([r["animal"] for r in R]), epoch=np.array([r["epoch"] for r in R]),
             aligns=np.array(ALIGNS), positions=np.array(POSITIONS),
             windows=np.array([WIN[x] for x in ALIGNS]), home=M["home"])
    cnt = pd.DataFrame([{k: r[k] for k in r if k.startswith("n_") or k in ("key", "label", "animal", "epoch")}
                        for r in R])
    cnt.to_csv(out / f"event_counts_fourcam{tag}.csv", index=False)
    print(cnt.to_string(), flush=True)
    print("->", npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
