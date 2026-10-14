"""Compare pose models on the measures the analyses use: lick detection against DAQ spout contacts, and tongue angle
at peak protrusion by spout position -- the same kinematics pipeline (`pose_kinematics_demo.run`) on each model's
predictions for one `session_poses` clip.

    PYTHONPATH=. python -m scripts.model_lick_compare PS93:20260814 --model "DLC r3=<csv>" --model "LP r4=<csv>" ...

Priya, 2026-10-13: judge the cam4 models (DLC / LP x round 3 / 4, and the temporal-context LP) downstream, not by
pixel error. Every model gets the SAME spout frame (mouth = the close->far spout-line intersection, from round-3 DLC's
spout, as the kinematics pipeline), trial stops and DAQ contacts, so differences are tracking.

Per model: DAQ contacts inside the response windows, how many a kept lick matched (`per_lick.contact`, the pipeline's
own matching: lick window peak -60 .. +120 ms), kept licks, licks with no contact (incomplete or missed-contact
licks, or false detections), per-position contact recall, and the median / IQR of the angle at peak protrusion per
position (a tighter IQR at the same median = less tracking noise).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POS_ORDER = ["far_L", "close_L", "close_center", "far_center", "close_R", "far_R"]


def run_model(csv, idx, frame, stops, contacts):
    import scripts.pose_kinematics_demo as D
    T, _J, _M, _clean = D.run(Path(csv), idx, frame, stops, contacts)
    return T


def summarise(name, T, contacts, stops) -> tuple[dict, pd.DataFrame]:
    """Recall = UNIQUE DAQ contacts (70 ms after cue .. trial stop) inside a kept lick's window (onset - 60 ms ..
    max(offset, peak + 120 ms)); a contact is never counted twice. Angles from CONTACT licks (the same licks for
    every model; no-contact licks are short and their angles noisy) and from all kept licks."""
    pl = T.per_lick
    n_c = found = 0
    for tid in T.per_trial.trial_id:
        cs = np.array([c for c in contacts.get(tid, []) if 70 <= c <= 1000 * stops.get(tid, np.inf)])
        if not len(cs):
            continue
        L = pl[pl.trial_id == tid]
        lo = L.on_ms.fillna(L.t_ms).to_numpy() - 60
        hi = np.maximum(L.off_ms.fillna(L.t_ms).to_numpy(), L.t_ms.to_numpy() + 120)
        n_c += len(cs)
        found += int(sum(bool(((lo <= c) & (c <= hi)).any()) for c in cs))
    is_c = pl.contact.fillna(False).astype(bool)
    row = {"model": name, "contacts": n_c, "found": found, "recall_%": round(100 * found / max(n_c, 1), 1),
           "kept_licks": len(pl), "contact_licks": int(is_c.sum()), "no_contact_licks": int((~is_c).sum())}

    def stats(df):
        return (df.dropna(subset=["peak_angle_deg"]).groupby("position").peak_angle_deg
                .agg(median="median", iqr=lambda a: np.subtract(*np.percentile(a, [75, 25])), n="size")
                .reindex(POS_ORDER).round(1))
    ang = pd.concat({"contact": stats(pl[is_c]), "all": stats(pl)}, names=["licks"]).reset_index()
    ang.insert(0, "model", name)
    return row, ang


def jumps(csv, idx, cut: float = 0.4, thr=(30, 60)) -> dict:
    """Frame-to-frame |dx| > thr px between CONTIGUOUS source frames where both are confident (>= cut): jaw side-flips
    and tongue teleports (selection rule 5)."""
    from wfield_local import orofacial_clean as oc
    P = oc.read_pose(Path(csv)).iloc[:len(idx)]
    contig = np.diff(idx.src_frame.to_numpy()[:len(P)]) == 1
    out = {}
    for part in ("jaw", "tongue"):
        x, p = P[f"{part}_x"].to_numpy(float), P[f"{part}_likelihood"].to_numpy(float)
        ok = contig & (p[1:] >= cut) & (p[:-1] >= cut)
        dx = np.abs(np.diff(x))
        for t in thr:
            out[f"{part}_x_jumps>{t}"] = int((ok & (dx > t)).sum())
    return out


def review_score(name, T, review: pd.DataFrame, win_ms: float = 40.0) -> dict:
    """Against Priya's lick review (no-contact candidates of this session): a candidate is DETECTED when a kept lick in
    the same trial peaks within `win_ms`. Precision = real / (real + not) among detected; share = detected real / real."""
    pl = T.per_lick
    det = np.array([bool((np.abs(pl.loc[pl.trial_id == r.trial_id, "t_ms"].to_numpy() - r.t_ms) <= win_ms).any())
                    for r in review.itertuples()])
    lab = review.label.to_numpy()
    real, nott = lab == "real", lab == "not"
    return {"rev_detected": int(det.sum()), "rev_real": int((det & real).sum()), "rev_not": int((det & nott).sum()),
            "rev_precision_%": round(100 * (det & real).sum() / max((det & (real | nott)).sum(), 1), 1),
            "rev_share_real_%": round(100 * (det & real).sum() / max(real.sum(), 1), 1)}


def main(argv=None) -> int:
    from scripts.session_poses import read_index, session_dir
    import scripts.pose_kinematics_demo as D
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--review", type=Path, help="review_scored.csv: add precision / share vs Priya's lick review")
    ap.add_argument("--tag", default="", help="suffix for the output csvs (keep several comparisons)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    animal, date, d = session_dir(rv, a.session)
    idx = read_index(d)
    raw = oc.read_pose(d / "windows_DLC.csv").iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(raw[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds(animal, date, rv)
    stops = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    rows, angs = [], []
    for spec in a.model:
        name, csv = spec.split("=", 1)
        T = run_model(csv, idx, frame, stops, contacts)
        r, g = summarise(name, T, contacts, stops)
        r.update(jumps(csv, idx))
        if a.review is not None:
            rv_ = pd.read_csv(a.review)
            rv_ = rv_[(rv_.kind == "no_contact") & rv_.id.str.startswith(f"{animal}_{date}_")]
            r.update(review_score(name, T, rv_))
        rows.append(r)
        angs.append(g)
    S = pd.DataFrame(rows)
    A = pd.concat(angs)
    out = d / "r4_eval"
    out.mkdir(exist_ok=True)
    S.to_csv(out / f"model_lick_compare{a.tag}.csv", index=False)
    A.to_csv(out / f"model_angle_compare{a.tag}.csv", index=False)
    print(f"{animal} {date}\n" + S.set_index("model").T.to_string())
    A = A.assign(v=A["median"].astype(str) + " [" + A["iqr"].astype(str) + "] (" + A["n"].astype("Int64").astype(str) + ")")
    for kind in ("contact", "all"):
        print(f"\nangle at peak protrusion (deg), {kind} licks: median [IQR] (n)")
        print(A[A.licks == kind].pivot(index="model", columns="position", values="v")
              .reindex(columns=POS_ORDER).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
