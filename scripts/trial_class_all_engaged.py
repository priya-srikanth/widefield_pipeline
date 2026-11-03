"""Trial classes over ALL ENGAGED trials, including NO RESPONSE (Priya 11-03: "what about fraction of trials (engaged)
with NO response").

Why a second table: `jaw_only_incomplete_figs` keeps only trials with ``jaw_pass_qc`` true, but that column is the
"jaw moved" call itself (`jaw_kinematics`: the sustained-density predicate), not a quality gate. So its denominator is
"engaged trials on which the jaw moved", its `none` class is empty by construction, and its shares are of jaw-moved
trials. Here the denominator is every engaged trial (`orofacial_cohort.engagement_keep`), with the same class rules,
first match wins:
  contact      >= 1 lick touched the spout (cam4 v36 lick with DAQ contact, or a cam1-only event with contact)
  incomplete   >= 1 lick (cam4, or cam1-only with droplets removed), none touched
  jaw_only     no lick, jaw moved (``candidate_no_lick_with_jaw_move``), cam1 saw no tongue (same as the source)
  no_response  no lick and the jaw did NOT move (``jaw_pass_qc`` == 0)
  unknown      no lick and the jaw call is undecidable (``jaw_pass_qc`` NaN: jaw not tracked)
Shares sum to 1 per session x spout. Same sessions as the source table (its sessions.csv); epochs copied from it.

    python scripts/trial_class_all_engaged.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.jaw_only_incomplete_figs import V36  # noqa: E402

SRC = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/cam1_licks")
CLS = ("contact", "incomplete", "jaw_only", "no_response", "unknown")


def session_counts(s: str, v36: Path) -> list[dict]:
    from wfield_local import orofacial_cohort as oc
    animal, date = s.split("_")
    sd = v36 / animal / date
    keep, _ = oc.engagement_keep(animal, date, sd)
    T = pd.read_csv(sd / "trials.csv")[["trial_id", "position"]]
    if keep is not None:
        T = T[T.trial_id.isin(keep)]
    M = pd.read_csv(sd / "tongue_jaw_mismatch.csv").set_index("trial_id")
    L = (pd.read_csv(sd / "licks.csv", usecols=["trial_id", "contact"]) if (sd / "licks.csv").exists()
         else pd.DataFrame(columns=["trial_id", "contact"]))
    E = pd.read_csv(SRC / f"{s}_events.csv")
    for c in ("cam4_matched", "in_3d", "contact", "jaw_only_trial", "droplet"):
        E[c] = E[c].astype("boolean").fillna(False).astype(bool)
    c1o = E[~E.cam4_matched & ~E.in_3d & ~E.droplet]
    seen = set(E.loc[E.jaw_only_trial & ~E.droplet, "trial_id"])
    touched = (set(L.loc[L.contact.astype("boolean").fillna(False).astype(bool), "trial_id"])
               | set(c1o.loc[c1o.contact, "trial_id"]))
    licked = set(L.trial_id) | set(c1o.trial_id)
    cand = set(M.index[M.candidate_no_lick_with_jaw_move.astype("boolean").fillna(False).astype(bool)]) - seen
    moved = pd.to_numeric(M.jaw_pass_qc, errors="coerce")

    def cls(t):
        if t in touched:
            return "contact"
        if t in licked:
            return "incomplete"
        if t in cand:
            return "jaw_only"
        q = moved.get(t, np.nan)
        return "unknown" if not np.isfinite(q) else ("jaw_only" if q > 0 else "no_response")

    T = T.assign(cls=T.trial_id.map(cls))
    return [{"session": s, "animal": animal, "date": date, "position": p, "n_trials": len(g),
             **{f"share_{c}": float((g.cls == c).mean()) for c in CLS},
             **{f"n_{c}": int((g.cls == c).sum()) for c in CLS}} for p, g in T.groupby("position")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SRC / "trial_classes_all_engaged"))
    a = ap.parse_args(argv)
    from wfield_local.paths import PathResolver
    v36 = Path(PathResolver().root("orofacial")) / V36 / "sessions"
    src = pd.read_csv(SRC / "trial_classes" / "sessions.csv")
    ep = src.drop_duplicates("session").set_index("session").epoch
    rows = []
    for s in ep.index:
        rows += session_counts(s, v36)
    S = pd.DataFrame(rows)
    S["epoch"] = S.session.map(ep)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "sessions.csv", index=False, float_format="%.4g")
    print(out / "sessions.csv", len(S), "rows")
    am = S.groupby(["position", "epoch", "animal"])[[f"share_{c}" for c in CLS]].mean().groupby(["position", "epoch"]).mean()
    print(am.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
