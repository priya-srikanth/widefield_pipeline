"""State-space per-row LocaNMF + cohort-SVD extraction for a NEW lick lineage, by RE-SAMPLING (CC-B 11-25, four-camera
rebuild; plan run_records/2026-11-25/STATESPACE_FOURCAM_PLAN.md).

WHY. The state-space bases `lnmf` / `svd` read per-row files (`o2_state_space_lnmf.py` / `o2_state_space_svd.py`): for
every first-lick row of the Allen extract, the component activity at `iti_licks.LAGS_C` from the cue (`first_cue`) and
`LAGS_P` from the first lick (`first_peth`). A new lick set (11-22 fused four-camera licks) keeps the imaging and the
cue times but moves first-lick times, so only the SAMPLING changes. Measured 11-25 (four-camera vs v36 extracts): 90 / 107
sessions keep the same rows, 17 differ by 1-3 rows; on matched rows `first_cue` is identical and `first_rt` moves (tails
to -1 s). Nothing saved in v36 can be re-selected into the new lick windows (only the sampled windows were kept; lane E's
basis extracts are ITI / random-lick windows without first-lick rows), so each session's projected trace is rebuilt -- one
pass, BOTH bases from one read of V and U_atlas (the O2 jobs read them twice) -- and its projection OPERATORS are saved
(`proj/<label>.npz`, content-keyed), so the next trial-set change re-samples from V alone.

REUSED unchanged: the pinned joint-LocaNMF bases (`bundle.json` bases of the v36 LocaNMF bundle), the cohort SVD basis
(`out/basis/` of the v36 SVD bundle, copied: built from August pre imaging only), the 12 June contact-only files (DAQ
contacts, no LP licks; copied), the cue time of every row matched to a v36 row.

  times    TABLES/cued_first_lick_times.csv (label, row, t = first-lick onset; the v36 format): per row of the NEW extract,
           the matched v36 row's cue time (match = byte-identical `first_cue`) + the new `first_rt`; rows without a timed
           v36 match get `iti_licks._first_lick_times` (behaviour reload, profile env, those sessions only), and on those
           sessions the matched rows are cross-checked against it.
  extract  <WF_SS_BUNDLES>/<lnmf-name>/out/sessions/<label>.npz and <WF_SS_BUNDLES>/<svd-name>/out/sessions/<label>.npz (keys as
           the O2 files), proj/<label>.npz operators, check.csv (new vs v36 `first_cue` on matched rows: must agree to
           float32 rounding, else the session is refused).

    python scripts/state_space_fourcam_extract.py times --tables <dir> [--jobs 3]
    python scripts/state_space_fourcam_extract.py extract --tables <dir> --lnmf-name N --svd-name N [--jobs 3] [LABELS]

Env: WF_SS_PROFILE / WF_SS_RUN (the lineage, as every state-space script), WF_SS_ENC_INPUTS (the G4 input build's
inputs/: V, ft per session). Writes only under --tables and the two named bundles (WFIELD_RUN_OUT must cover them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_sim as S  # noqa: E402

V36_ROOT = Path(str(S.SHARE)) / "rebuild_20261103"
V36_RUN = "lp_r4_heatmap_20261013__clean_v36"
V36_ITI = V36_ROOT / "LP_analysis" / V36_RUN / "imaging_encoding" / "iti_licks" / S.XTAG / "sessions_roi8fr"
V36_TIMES = V36_ROOT / "LP_analysis" / V36_RUN / "imaging_encoding" / "iti_licks" / S.WTAG / "cued_first_lick_times.csv"
V36_LNMF = V36_ROOT / "o2" / "ss_lnmf_20261108"
V36_SVD = V36_ROOT / "o2" / "ss_svdbm_20261108"
DRIFT_S = 600.0
TOL = 1e-3          # max |new - v36| / SD of the v36 cue windows (float32 / BLAS-order rounding only)


def _keys(fc):
    """Byte key per row of a `first_cue` array (rows identical <=> same trial: same imaging, same cue time)."""
    return [hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in fc]


def _match(label):
    """(new_rt, new_n, v36 row index per new row or -1, v36 first-lick time per v36 row)."""
    a = np.load(Path(S.ITI) / f"{label}.npz")
    b = np.load(V36_ITI / f"{label}.npz")
    ka, kb = _keys(a["first_cue"]), _keys(b["first_cue"])
    where = {}
    for i, k in enumerate(kb):
        where.setdefault(k, []).append(i)
    m = np.array([where[k].pop(0) if where.get(k) else -1 for k in ka], int)
    return a["first_rt"].astype(float), b["first_rt"].astype(float), m


def cmd_times(a) -> None:
    out = Path(a.tables)
    out.mkdir(parents=True, exist_ok=True)
    t36 = pd.read_csv(V36_TIMES, float_precision="round_trip")
    labels = sorted(p.stem for p in Path(S.ITI).glob("PS9*.npz"))
    rows, need, audit = {}, [], []
    for lab in labels:
        rt, rt36, m = _match(lab)
        g = t36[t36.label == lab]
        tv = np.full(len(rt36), np.nan)
        tv[g.row.to_numpy(int)] = g.t.to_numpy(float)
        cue36 = tv - rt36
        cue = np.where(m >= 0, cue36[np.maximum(m, 0)], np.nan)
        t = cue + rt
        rows[lab] = t
        miss = int((~np.isfinite(t)).sum())
        audit.append({"label": lab, "n_new": len(rt), "n_v36": len(rt36), "matched": int((m >= 0).sum()),
                      "untimed_after_match": miss, "rt_changed": int((np.abs(rt - np.where(m >= 0, rt36[np.maximum(m, 0)],
                                                                                         np.nan)) > 1e-9).sum())})
        if miss:
            need.append(lab)
    print(f"{len(labels)} sessions; {len(need)} need the behaviour reload for untimed rows: {need}", flush=True)
    if need:
        env = dict(os.environ)
        from wfield_local import render_profile
        render_profile.apply(S.PROFILE, environ=env)
        # the extract's own env (lane I i_iti_extract, run log 11-23 06:55): the first-lick rule must see the same licks
        env.update({"OROFACIAL_GATE": "attempted", "COMBINED_ENCODING_LICKS": "fourcam", "COMBINED_ENCODING_ONLY": "full",
                    "COMBINED_ENCODING_SUFFIX": "_gate-attempted", "COMBINED_ENCODING_TONGUE": "3d", "FROZEN_FR": "1",
                    "COMBINED_ENCODING_FR": "master_pre_v1", "COMBINED_ENCODING_FR_GROUPS": "master_v1:A",
                    "FROZEN_LICKSPOUT": "1"})
        # the profile process may not write outside the rebuild root (its fence): it returns the times on stdout
        cmd = [sys.executable, __file__, "behaviour-times", "--jobs", str(a.jobs), *need]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        print(r.stderr[-4000:], flush=True)
        if r.returncode:
            raise SystemExit(f"behaviour-times failed (rc {r.returncode}): {r.stdout[-2000:]}")
        bt = json.loads(next(x for x in r.stdout.splitlines() if x.startswith("BTJSON "))[7:])
        for lab in need:
            tb = np.asarray(bt[lab], float)
            t = rows[lab]
            both = np.isfinite(t) & np.isfinite(tb)
            dev = float(np.max(np.abs(t[both] - tb[both]))) if both.any() else 0.0
            if dev > 1e-6:
                raise SystemExit(f"{lab}: matched-row times disagree with the behaviour reload (max {dev:.3g} s)")
            rows[lab] = np.where(np.isfinite(t), t, tb)
            r = next(x for x in audit if x["label"] == lab)
            r["behaviour_filled"] = int((~np.isfinite(t) & np.isfinite(tb)).sum())
            r["untimed_final"] = int((~np.isfinite(rows[lab])).sum())
            r["crosscheck_max_s"] = dev
    pd.concat([pd.DataFrame({"label": k, "row": np.arange(len(v)), "t": v}) for k, v in rows.items()],
              ignore_index=True).to_csv(out / "cued_first_lick_times.csv", index=False, float_format="%.17g")
    pd.DataFrame(audit).to_csv(out / "cued_first_lick_times_audit.csv", index=False)
    print("wrote", out / "cued_first_lick_times.csv", flush=True)


def _bt_worker(label):
    import iti_licks as il
    rt = np.load(Path(S.ITI) / f"{label}.npz")["first_rt"].astype(float)
    return il._first_lick_times(label, rt)[1].tolist()


def cmd_behaviour_times(a) -> None:
    from wfield_local import analysis_kit as kit
    res, fail = kit.fan_sessions(a.labels, _bt_worker, jobs=min(a.jobs, len(a.labels)), label="times")
    if fail:
        raise SystemExit(f"behaviour times failed: {fail}")
    print("BTJSON " + json.dumps(dict(res)), flush=True)


# ------------------------------------------------------------------------------------------------------------ extract
def _sample(R, ft, t, lags):                                        # iti_licks._sample (NaN where t is NaN)
    o = np.full((len(t), len(lags), R.shape[1]), np.nan, np.float32)
    ok = np.isfinite(t) & (t + lags[0] > ft[0]) & (t + lags[-1] < ft[-1])
    tt = (t[ok][:, None] + lags[None, :]).ravel()
    for j in range(R.shape[1]):
        o[ok, :, j] = np.interp(tt, ft, R[:, j]).reshape(ok.sum(), len(lags))
    return o


def _ex_worker(item):
    label, cfg = item
    import iti_licks as il
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import joint_locanmf as jl
    if ce.DRIFT_MIN_PERIOD_S != DRIFT_S:
        raise RuntimeError(f"drift period {ce.DRIFT_MIN_PERIOD_S} != {DRIFT_S}")
    t0 = time.time()
    lb, sb = Path(cfg["lnmf"]), Path(cfg["svd"])
    z = np.load(Path(S.ITI) / f"{label}.npz")
    rt, pos = z["first_rt"].astype(float), z["first_pos"].astype(str)
    tt = pd.read_csv(cfg["times"], float_precision="round_trip").query("label == @label")
    lick = np.full(len(rt), np.nan)
    lick[tt.row.to_numpy(int)] = tt.t.to_numpy(float)
    cue = lick - rt
    x = pickle.loads((Path(cfg["inputs"]) / f"{label}.pkl").read_bytes())
    V, ft = x["imaging"][0], np.asarray(x["imaging"][1], float)
    del x
    an = label.split("_")[0]
    up = Path(cfg["priya"]) / cfg["uatlas"][label]
    U = np.load(up)
    u = np.nan_to_num(U.reshape(-1, U.shape[2])).astype(np.float32)
    del U
    Q = np.linalg.qr(ce._slow_basis(ft, DRIFT_S))[0]
    common = dict(lags_c=il.LAGS_C, lags_p=il.LAGS_P, first_rt=rt, first_pos=pos, cue_s=cue, lick_s=lick, src="camera",
                  epoch=str(epochs.epoch_of(label)), animal=an)
    # LocaNMF: the O2 runner's arithmetic (project_uv on the full V, then drift removal of the traces)
    B = jl.Basis(Path(cfg["bases_root"]) / an / cfg["bases"][an])
    C, diag = B.project_uv(u, np.ascontiguousarray(V.T, dtype=np.float32), label, with_diagnostics=True)
    R = np.nan_to_num(np.asarray(C, np.float64).T)
    del C
    R -= Q @ (Q.T @ R)
    Uj = np.load(B.root / "U_joint.npy")
    from wfield_local.joint_locanmf import _footprint_scale
    op_l = (_footprint_scale(np.asarray(B.A), B.ncomp)[:, None] * (B._operator @ (Uj.T @ u))).astype(np.float32)
    lab_out = lb / "out" / "sessions" / f"{label}.npz"
    fc_l = _sample(R, ft, cue, il.LAGS_C)
    np.savez_compressed(lab_out, first_cue=fc_l, first_peth=_sample(R, ft, lick, il.LAGS_P), basis_id=B.basis_id,
                        variance_captured=diag["variance_captured"],
                        regions=B.regions if (B.root / "regions.npy").exists() else np.array([]), **common)
    del R
    # SVD: the O2 runner's arithmetic (drift removed from V first, brain-mask pixels, P = U_cohort^T u)
    Vd = np.asarray(V, np.float64)
    Vd -= Q @ (Q.T @ Vd)
    mask = np.load(sb / "out" / "basis" / "mask.npy").ravel().astype(bool)
    Uc = np.load(sb / "out" / "basis" / "U_cohort.npy")
    bz = np.load(sb / "out" / "basis" / "basis.npz")
    um = u[mask]
    P = (Uc.T @ um).astype(np.float64)
    R = Vd @ P.T
    W = Vd.T @ Vd
    var = float(np.sum((P.T @ P) * W) / max(np.sum((um.T @ um).astype(np.float64) * W), 1e-30))
    cols = np.array([f"svd#{k + 1}_{s}" for k, s in enumerate(bz["side"])])
    fc_s = _sample(R, ft, cue, il.LAGS_C)
    np.savez_compressed(sb / "out" / "sessions" / f"{label}.npz", first_cue=fc_s,
                        first_peth=_sample(R, ft, lick, il.LAGS_P), basis_id=cfg["svd_basis_id"],
                        variance_captured=var, cols=cols, **common)
    # operators for the next re-selection (content key: basis ids + U_atlas size / mtime + drift period)
    st = up.stat()
    key = f"{B.basis_id}|{cfg['svd_basis_id']}|{up.name}:{st.st_size}:{st.st_mtime_ns}|drift{DRIFT_S}"
    np.savez(lb / "proj" / f"{label}.npz", op_lnmf=op_l, op_svd=P.astype(np.float32), key=key,
             note="LocaNMF traces = op_lnmf @ V.T then drift removed; SVD traces = (drift-removed V) @ op_svd.T")
    # check vs v36 on matched rows (cue windows: same imaging, same cue times)
    chk = {"label": label, "n": len(rt), "timed": int(np.isfinite(lick).sum()), "seconds": round(time.time() - t0, 1),
           "var_lnmf": diag["variance_captured"], "var_svd": var}
    _, _, m = _match(label)
    for nm, new, old in (("lnmf", fc_l, V36_LNMF), ("svd", fc_s, V36_SVD)):
        o = np.load(old / "out" / "sessions" / f"{label}.npz")["first_cue"]
        ok = m >= 0
        a_, b_ = new[ok], o[m[ok]]
        fin = np.isfinite(a_) & np.isfinite(b_)
        sd = float(np.nanstd(b_)) or 1.0
        chk[f"cue_dev_{nm}"] = float(np.max(np.abs(a_[fin] - b_[fin])) / sd) if fin.any() else np.nan
        chk[f"nan_mismatch_{nm}"] = int((np.isfinite(a_) != np.isfinite(b_)).sum())
    bad = [k for k in ("cue_dev_lnmf", "cue_dev_svd") if not chk[k] <= TOL] + \
          [k for k in ("nan_mismatch_lnmf", "nan_mismatch_svd") if chk[k]]
    if bad:
        for f in (lab_out, sb / "out" / "sessions" / f"{label}.npz"):
            f.rename(f.with_suffix(".FAILEDCHECK.npz"))
        raise RuntimeError(f"{label}: v36 cue-window check failed {bad}: {chk}")
    return chk


def cmd_extract(a) -> None:
    lb, sb = S.BUNDLES / a.lnmf_name, S.BUNDLES / a.svd_name
    if (lb, sb) != (S.LNMF.parents[1], S.SVD.parents[1]):
        raise SystemExit(f"--lnmf-name / --svd-name must match WF_SS_LNMF_NAME / WF_SS_SVD_NAME ({S.LNMF}, {S.SVD})")
    for d in (lb / "out" / "sessions", lb / "proj", sb / "out" / "sessions", sb / "out" / "basis"):
        d.mkdir(parents=True, exist_ok=True)
    for f in ("U_cohort.npy", "basis.npz", "mask.npy"):                       # the unchanged cohort basis (copy)
        if not (sb / "out" / "basis" / f).exists():
            shutil.copy2(V36_SVD / "out" / "basis" / f, sb / "out" / "basis" / f)
    if not (sb / "atlas.npy").exists():
        shutil.copy2(V36_SVD / "atlas.npy", sb / "atlas.npy")
    labels = a.labels or sorted(p.stem for p in Path(S.ITI).glob("PS9*.npz"))
    ml = json.loads((V36_LNMF / "bundle.json").read_text())
    ms = json.loads((V36_SVD / "bundle.json").read_text())
    june = [lab for lab in ml["uatlas"] if S._epoch(lab, "") == "june" or lab.split("_")[1].startswith("06")]
    for lab in june:                                                          # DAQ-contact rows: no lick input (copy)
        for src, dst in ((V36_LNMF, lb), (V36_SVD, sb)):
            f = dst / "out" / "sessions" / f"{lab}.npz"
            if not f.exists():
                shutil.copy2(src / "out" / "sessions" / f"{lab}.npz", f)
    inputs = Path(os.environ["WF_SS_ENC_INPUTS"])
    todo = [x for x in labels if not ((lb / "out" / "sessions" / f"{x}.npz").exists()
                                      and (sb / "out" / "sessions" / f"{x}.npz").exists())]
    from wfield_local.paths import PathResolver
    cfg = {"lnmf": str(lb), "svd": str(sb), "times": str(Path(a.tables) / "cued_first_lick_times.csv"),
           "inputs": str(inputs), "priya": str(PathResolver().root("microscope")), "uatlas": ml["uatlas"],
           "bases": ml["bases"], "bases_root": str(Path(str(S.SHARE)) / "labcams" / "joint_bases"),
           "svd_basis_id": ms["basis_id"]}
    # bundle metadata: the v36 bundle's keys that the readers use (uatlas map / root, rows, labels, masks), re-pointed:
    # enc_scratch = the G4 build (its inputs/ = V, ft), lnmf_scratch = the v36 LocaNMF bundle (June inputs/ only)
    for d, m0, extra in ((lb, ml, {"lnmf_v36": str(V36_LNMF)}), (sb, ms, {"svd_basis_from": str(V36_SVD / "out" / "basis")})):
        meta = dict(m0)
        meta.update({"name": d.name, "made_by": "scripts/state_space_fourcam_extract.py (re-sampling; see docstring)",
                     "profile": S.PROFILE, "run": S.REBUILD_RUN, "allen_extract": S.ITI, "times": cfg["times"],
                     "enc_scratch": str(inputs.parent), "lnmf_scratch": str(V36_LNMF), "uatlas_root": cfg["priya"],
                     "drift_min_period_s": DRIFT_S, "june_copied_from_v36": june,
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "v36_meta": {k: m0.get(k) for k in ("name", "commit")},
                     **extra})
        (d / "bundle.json").write_text(json.dumps(meta, indent=1))
        (d / "labels.txt").write_text((V36_LNMF / "labels.txt").read_text())
        (d / "rows").mkdir(exist_ok=True)
        for lab in june:
            if not (d / "rows" / f"{lab}.npz").exists():
                shutil.copy2(V36_LNMF / "rows" / f"{lab}.npz", d / "rows" / f"{lab}.npz")
    for f in ("mask.npy", "atlas.npy"):
        if not (sb / f).exists():
            shutil.copy2(V36_SVD / f, sb / f)
    # rows/<label>.npz of every August session (the readers' row timing: enl_baseline reads cue_s), from the tables
    import iti_licks as il
    from wfield_local import epochs
    tt_all = pd.read_csv(cfg["times"], float_precision="round_trip")
    for lab in labels:
        z = np.load(Path(S.ITI) / f"{lab}.npz")
        rt = z["first_rt"].astype(float)
        g = tt_all[tt_all.label == lab]
        lick = np.full(len(rt), np.nan)
        lick[g.row.to_numpy(int)] = g.t.to_numpy(float)
        for d in (lb, sb):
            np.savez(d / "rows" / f"{lab}.npz", cue_s=lick - rt, lick_s=lick, first_rt=rt,
                     first_pos=z["first_pos"].astype(str), src="camera", epoch=str(epochs.epoch_of(lab)),
                     lags_c=il.LAGS_C, lags_p=il.LAGS_P)
    print(f"{len(todo)} of {len(labels)} sessions to extract", flush=True)
    from wfield_local import analysis_kit as kit
    res, fail = kit.fan_sessions([(x, cfg) for x in todo], _ex_worker, jobs=min(a.jobs, max(1, len(todo))),
                                 label="extract")
    rows = [r for _, r in res]
    f = lb / "check.csv"
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("label", keep="last").to_csv(f, index=False)
    print(pd.DataFrame(rows).describe().T.to_string() if rows else "nothing new", flush=True)
    if fail:
        raise SystemExit(f"{len(fail)} sessions failed: {fail}")


# --------------------------------------------------------------------------------------- whole-session trace cache
# Priya 11-25 ("cache appropriately"): every session's whole-session projected component traces, both bases, float32,
# so a later trial-set / first-lick change re-samples rows (`_sample`) without re-projecting. Content key = the basis
# ids, the U_atlas file (path, size, mtime), the drift period, and the V source (G4 input pickle: path, size, mtime).
# `load_traces` is the ONLY reader: it recomputes the key from the live files and REFUSES a stale entry; it never
# computes into the cache (the cache is written only by `cache`, a declared run).
TRACE_VERSION = 1


class StaleCacheError(RuntimeError):
    """A cached trace whose content key no longer matches its inputs."""


def _stat(p) -> dict:
    st = Path(p).stat()
    return {"path": str(Path(p)).replace("\\", "/"), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def trace_key(label: str, lnmf_bundle, svd_bundle, inputs) -> dict:
    """The content key of ``label``'s traces from the bundles' metadata and the live input files."""
    from wfield_local.paths import PathResolver
    ml = json.loads((Path(lnmf_bundle) / "bundle.json").read_text())
    ms = json.loads((Path(svd_bundle) / "bundle.json").read_text())
    an = label.split("_")[0]
    return {"version": TRACE_VERSION, "label": label, "lnmf_basis": ml["bases"][an], "svd_basis": ms["basis_id"],
            "uatlas": _stat(Path(PathResolver().root("microscope")) / ml["uatlas"][label]), "drift_s": DRIFT_S,
            "v_source": _stat(Path(inputs) / f"{label}.pkl")}


def load_traces(label: str, cache_dir, lnmf_bundle, svd_bundle, inputs):
    """(ft, {'lnmf': (T, ncomp), 'svd': (T, 100)}) float32 drift-removed traces of ``label``; raises FileNotFoundError on a
    miss and StaleCacheError when the stored key differs from the live inputs. Never computes."""
    f = Path(cache_dir) / f"{label}.npz"
    z = np.load(f, allow_pickle=False)
    have, want = json.loads(str(z["key"])), trace_key(label, lnmf_bundle, svd_bundle, inputs)
    if have != want:
        raise StaleCacheError(f"{f}: stored key {have} != live {want}; rebuild with `state_space_fourcam_extract.py cache`")
    return z["ft"], {"lnmf": z["lnmf"], "svd": z["svd"]}


def _cache_worker(item):
    label, cfg = item
    import iti_licks as il
    from wfield_local import combined_encoding as ce
    t0 = time.time()
    lb, sb, out = Path(cfg["lnmf"]), Path(cfg["svd"]), Path(cfg["cache"])
    key = trace_key(label, lb, sb, cfg["inputs"])
    x = pickle.loads((Path(cfg["inputs"]) / f"{label}.pkl").read_bytes())
    V, ft = np.asarray(x["imaging"][0], np.float64), np.asarray(x["imaging"][1], float)
    del x
    pz = np.load(lb / "proj" / f"{label}.npz")
    Q = np.linalg.qr(ce._slow_basis(ft, DRIFT_S))[0]
    L = V @ np.asarray(pz["op_lnmf"], np.float64).T                     # (T, ncomp) = project_uv's C.T
    L -= Q @ (Q.T @ L)
    Vd = V - Q @ (Q.T @ V)
    Sv = Vd @ np.asarray(pz["op_svd"], np.float64).T                    # (T, 100)
    # check: re-sampled from the cached traces, every row's first_cue / first_peth must equal the extraction's files
    rows = np.load(lb / "rows" / f"{label}.npz")
    dev = {}
    for nm, R, d in (("lnmf", L, lb), ("svd", Sv, sb)):
        o = np.load(d / "out" / "sessions" / f"{label}.npz")
        for k, t, lags in (("first_cue", rows["cue_s"], il.LAGS_C), ("first_peth", rows["lick_s"], il.LAGS_P)):
            a_, b_ = _sample(R, ft, np.asarray(t, float), lags), o[k]
            fin = np.isfinite(a_) & np.isfinite(b_)
            dev[f"{nm}_{k}"] = float(np.max(np.abs(a_[fin] - b_[fin])) / (np.nanstd(b_) or 1.0)) if fin.any() else 0.0
    if max(dev.values()) > TOL:
        raise RuntimeError(f"{label}: cached traces do not reproduce the extraction files: {dev}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{label}.npz", lnmf=L.astype(np.float32), svd=Sv.astype(np.float32), ft=ft,
             key=json.dumps(key, sort_keys=False))
    return {"label": label, "T": len(ft), "mb": round((out / f"{label}.npz").stat().st_size / 2**20, 1),
            "seconds": round(time.time() - t0, 1), **dev}


def cmd_cache(a) -> None:
    lb, sb = S.LNMF.parents[1], S.SVD.parents[1]
    out = Path(a.out)
    inputs = Path(os.environ["WF_SS_ENC_INPUTS"])
    labels = a.labels or sorted(p.stem for p in Path(S.ITI).glob("PS9*.npz"))
    todo = [x for x in labels if not (out / f"{x}.npz").exists()]
    cfg = {"lnmf": str(lb), "svd": str(sb), "cache": str(out), "inputs": str(inputs)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(f"""# State-space whole-session projected traces (four-camera rebuild, CC-B 11-25)

Priya 11-25: "go ahead with state space locaNMF/SVD re-extraction and cache appropriately".
One file per session `<label>.npz`: `lnmf` (T, ncomp) and `svd` (T, 100) float32 drift-removed component traces (the
arithmetic of `o2_state_space_lnmf` / `o2_state_space_svd`), `ft` (T,) frame times (s, DAQ clock), `key` (JSON).

Content key (`scripts/state_space_fourcam_extract.py:trace_key`): version {TRACE_VERSION}; label; LocaNMF pinned basis id;
cohort SVD basis id; U_atlas path + size + mtime_ns; drift period (s); V source = the G4 input pickle path + size +
mtime_ns ({inputs}).
Reader: `state_space_fourcam_extract.load_traces(label, cache_dir, lnmf_bundle, svd_bundle, inputs)`. It recomputes the
key from the live files and raises StaleCacheError on any difference; it never computes into this folder. Re-sample rows
with `state_space_fourcam_extract._sample(trace, ft, times, lags)`.
Written by `state_space_fourcam_extract.py cache` (a declared run, WIDEFIELD_REBUILD_ROOT = this root); each file was
checked to reproduce the per-row extraction files ({lb.name}, {sb.name}) on every row (tolerance {TOL} x SD).
Operators: `{lb}/proj/<label>.npz`.
""")
    from wfield_local import analysis_kit as kit
    res, fail = kit.fan_sessions([(x, cfg) for x in todo], _cache_worker, jobs=min(a.jobs, max(1, len(todo))),
                                 label="cache")
    rows = [r for _, r in res]
    if rows:
        D = pd.DataFrame(rows)
        f = out / "check.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        pd.concat([old, D], ignore_index=True).drop_duplicates("label", keep="last").to_csv(f, index=False)
        print(D.describe().T.to_string(), flush=True)
    if fail:
        raise SystemExit(f"{len(fail)} sessions failed: {fail}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("times")
    p.add_argument("--tables", required=True)
    p.add_argument("--jobs", type=int, default=3)
    p = sp.add_parser("behaviour-times")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("labels", nargs="+")
    p = sp.add_parser("extract")
    p.add_argument("--tables", required=True)
    p.add_argument("--lnmf-name", required=True)
    p.add_argument("--svd-name", required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("labels", nargs="*")
    p = sp.add_parser("cache")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    {"times": cmd_times, "behaviour-times": cmd_behaviour_times, "extract": cmd_extract, "cache": cmd_cache}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
