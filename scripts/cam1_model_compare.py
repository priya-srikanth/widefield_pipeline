"""Score cam1 pose models on the measures agreed with Priya 2026-10-18 (`docs/POSE_MODELS.md` Part C), on the cam1 trial clips
(`scripts.session_poses clip --cam cam1` -> `session_poses/<animal>_<date>_cam1/`; the same 60 trials as the cam4 clips).

    python -m scripts.cam1_model_compare PS93:20260814=pre PS93:20260821=acute PS93:20260908=chronic \\
        --model "DLC c1r1={d}/dlc_r1/*.h5" --model "DLC c1r2={d}/dlc_r2/*.h5" --model "LP c1r2=<path with {d}>" \\
        [--cam4 "LP r4={c4}/<csv>"] [--tag _1018]

``{d}`` = the session's cam1 clip folder, ``{c4}`` = its cam4 clip folder; a glob must match exactly one file.
Every model gets the same trials, DAQ contacts and frames, so differences are tracking. Per session (= one epoch) x model:

* **contacts** (metric 2): share of DAQ spout contacts (cue + 70 ms .. trial stop) with a confident tongue (>= 0.4) within
  +-20 ms of the contact, overall and per spout position. From below, a tongue at the spout should be visible.
* **tip jumps** (metric 3): tongue steps > 40 px (the cam1 p99 of genuine motion, B3) between CONTIGUOUS source frames, both
  confident, per 10k confident tongue frames -- one-frame tip-end switches -- beside tongue coverage.
* **cam1 vs cam4 timing** (metric 3, with ``--cam4``): per contact seen by both, the onset of the confident tongue run that
  contains it, cam1 minus cam4 (ms on the shared cue clock); median |difference| and share within 12 ms (3 frames).
* **jaw** (metric 4): coverage (>= 0.6) of post-cue frames per position next to jaw steps > 20 px per 10k confident frames,
  so abstaining cannot pass for stability.

Output: `<cam1 clip folder>/eval/cam1_model_compare<tag>.csv` (one row per model, per-position columns) and a pooled
`session_poses/cam1_model_compare<tag>.csv`. Epoch labels come from the command line (``ANIMAL:DATE=EPOCH``).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

POS_ORDER = ["far_L", "close_L", "close_center", "far_center", "close_R", "far_R"]
CUT = {"tongue": 0.4, "jaw": 0.6}            # cam1 analysis cutoffs (DECISIONS 2026-10-05)
TIP_JUMP_PX, JAW_JUMP_PX = 40.0, 20.0
CONTACT_TOL_MS, SYNC_TOL_MS = 20.0, 12.0


def resolve(pattern: str, d: Path, c4: Path) -> Path:
    hits = glob.glob(pattern.replace("{d}", str(d)).replace("{c4}", str(c4)))
    if len(hits) != 1:
        raise SystemExit(f"{pattern!r} in {d.name}: {len(hits)} matches, need exactly one")
    return Path(hits[0])


def onsets_at_contacts(P: pd.DataFrame, idx: pd.DataFrame, contacts: dict) -> dict:
    """{(trial_id, contact_ms): onset_ms or NaN}: onset of the confident tongue run holding the frame nearest the contact
    (within CONTACT_TOL_MS); runs never cross a gap in source frames."""
    ok = P["tongue_likelihood"].to_numpy(float)[:len(idx)] >= CUT["tongue"]
    t = idx.t_ms.to_numpy(float)
    src = idx.src_frame.to_numpy()
    out = {}
    for tid, g in idx.groupby("trial_id", sort=False):
        rows = g.index.to_numpy()
        for c in contacts.get(tid, []):
            k = rows[np.argmin(np.abs(t[rows] - c))]
            near = rows[np.abs(t[rows] - c) <= CONTACT_TOL_MS]
            if not ok[near].any():
                out[(tid, float(c))] = np.nan
                continue
            k = near[ok[near]][np.argmin(np.abs(t[near[ok[near]]] - c))]
            while k - 1 >= rows[0] and ok[k - 1] and src[k] - src[k - 1] == 1:
                k -= 1
            out[(tid, float(c))] = t[k]
    return out


def score(name: str, P: pd.DataFrame, idx: pd.DataFrame, contacts: dict, stops: dict) -> dict:
    n = min(len(P), len(idx))
    P, idx = P.iloc[:n], idx.iloc[:n]
    r: dict = {"model": name}
    on = onsets_at_contacts(P, idx, contacts)
    pos_of = idx.groupby("trial_id").position.first().to_dict()
    seen = pd.DataFrame([(pos_of[t], np.isfinite(v)) for (t, _), v in on.items()], columns=["position", "seen"])
    r["contacts"] = len(seen)
    r["contact_seen_%"] = round(100 * seen.seen.mean(), 1) if len(seen) else np.nan
    for p in POS_ORDER:
        s = seen[seen.position == p].seen
        r[f"contact_seen_%_{p}"] = round(100 * s.mean(), 1) if len(s) else np.nan
    contig = np.diff(idx.src_frame.to_numpy()) == 1
    t_ms = idx.t_ms.to_numpy(float)
    stop_ms = idx.trial_id.map(lambda t: 1000 * stops.get(t, np.inf)).to_numpy(float)
    post = (t_ms >= 70) & (t_ms <= stop_ms)
    for part, thr in (("tongue", TIP_JUMP_PX), ("jaw", JAW_JUMP_PX)):
        x, y = P[f"{part}_x"].to_numpy(float), P[f"{part}_y"].to_numpy(float)
        ok = P[f"{part}_likelihood"].to_numpy(float) >= CUT[part]
        both = contig & ok[1:] & ok[:-1]
        big = both & (np.hypot(np.diff(x), np.diff(y)) > thr)
        r[f"{part}_conf_%_postcue"] = round(100 * ok[post].mean(), 1)
        r[f"{part}_jumps>{thr:.0f}_per10k"] = round(1e4 * big.sum() / max(ok.sum(), 1), 1)
    jok = P["jaw_likelihood"].to_numpy(float) >= CUT["jaw"]
    for p in POS_ORDER:
        m = post & (idx.position.to_numpy() == p)
        r[f"jaw_conf_%_{p}"] = round(100 * jok[m].mean(), 1) if m.any() else np.nan
    r["_onsets"] = on
    return r


def main(argv=None) -> int:
    import scripts.pose_kinematics_demo as D
    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD=EPOCH")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--cam4", default=None, metavar="NAME=PATH", help="cam4 reference for the timing comparison")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    rv = PathResolver()
    pooled = []
    for spec in a.sessions:
        sess, epoch = spec.split("=", 1)
        animal, date, c4 = session_dir(rv, sess)
        _, _, d = session_dir(rv, f"{sess}:cam1")
        idx = read_index(d)
        b = TW.trial_bounds(animal, date, rv)
        stops = dict(zip(b.trial_id, b.stop_s - b.cue_s))
        contacts = D.daq_contacts_ms(animal, date, b, rv)
        ref = None
        if a.cam4:
            rname, rpat = a.cam4.split("=", 1)
            ref = onsets_at_contacts(oc.read_pose(resolve(rpat, d, c4)), read_index(c4), contacts)
        rows = []
        for m in a.model:
            name, pat = m.split("=", 1)
            r = score(name, oc.read_pose(resolve(pat, d, c4)), idx, contacts, stops)
            on = r.pop("_onsets")
            if ref is not None:
                dt = np.array([on[k] - ref[k] for k in on if k in ref and np.isfinite(on[k]) and np.isfinite(ref[k])])
                r["vs_cam4_n"] = len(dt)
                r["vs_cam4_onset_median_abs_ms"] = round(float(np.median(np.abs(dt))), 1) if len(dt) else np.nan
                r[f"vs_cam4_within_{SYNC_TOL_MS:.0f}ms_%"] = round(100 * float((np.abs(dt) <= SYNC_TOL_MS).mean()), 1) if len(dt) else np.nan
            rows.append({"session": f"{animal}_{date}", "epoch": epoch, **r})
        S = pd.DataFrame(rows)
        (d / "eval").mkdir(exist_ok=True)
        S.to_csv(d / "eval" / f"cam1_model_compare{a.tag}.csv", index=False)
        pooled.append(S)
        print(f"\n{animal} {date} ({epoch})\n" + S.drop(columns=["session", "epoch"]).set_index("model").T.to_string())
    out = d.parent / f"cam1_model_compare{a.tag}.csv"
    pd.concat(pooled).to_csv(out, index=False)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
