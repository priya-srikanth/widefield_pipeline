"""June drift reference for the chronic change (running list 2026-11-08 section C; Priya 11-08: "add june where possible
(dn't have good kinematics so may need to use the contact-only trials?)"). LocaNMF basis only (`WF_SS_BASIS=lnmf`): the
Allen extract has no June sessions.

Is the August-pre -> chronic change more than the no-stroke drift over a similar span? Per animal, in its FIXED pre space
(the August pre template, top-K, `state_space_recovery_trials.fixed_space`), with like-for-like trials: every session
restricted to CONTACT first licks (June: rows are first DAQ contacts by construction; August + later: the 3-D first lick
touched the spout, landing table `contact`, `relabel(..., "contact")`).
  june_to_pre    median noise-corrected distance (`vdist`) June session <-> August pre session
  chronic_to_pre median distance chronic session <-> August pre session
  pre_to_pre     median distance between August pre sessions (the unit; both distances are divided by it)
  cos_june_chr   cosine between (June template - pre template) and (chronic template - pre template), flattened over
                 the spouts all three have; cos_june_acute the same for acute (comparison); 1/sqrt(dim) = chance scale
  frac_along     share of the chronic change's squared length along the June -> pre direction
  gap_june_days / gap_chronic_days   median day gaps (June -> August pre, August pre -> chronic)
June -> August includes LEARNING, so june_to_pre is an UPPER bound on drift over ~2 months. Reading (fixed before the
data): chronic_to_pre >> june_to_pre and the change not along the June direction -> beyond drift; comparable or along
it -> not separable. Per-animal table only (rule 8; n = 4, no pooled test). EXPLORATORY.
Output: <state_space_recovery>/june_drift{tag}/june_drift.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

OUT = S.ROOT / f"june_drift{S.tag()}"


def direction(Ta, Tb, P, mu, ok):
    """(Ta - Tb) in the fixed space, flattened over the spouts in ``ok``."""
    return (((Ta - mu) @ P)[ok] - ((Tb - mu) @ P)[ok]).ravel()


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def animal_row(an, sess):
    by = {ep: [s for s in sess if s[1] == ep] for ep in ("june", "pre", "acute", "chronic")}
    if not (by["june"] and len(by["pre"]) > 1 and by["chronic"]):
        return {"animal": an, **{f"n_{ep}": len(v) for ep, v in by.items()}}
    T = {ep: R.template(v) for ep, v in by.items() if v}
    P, mu = R.fixed_space(T["pre"])
    vec = {ep: [R.vec(s[3], s[4], P, mu) for s in v] for ep, v in by.items()}
    pp = np.nanmedian([R.vdist(a, b) for i, a in enumerate(vec["pre"]) for b in vec["pre"][i + 1:]])
    to_pre = {ep: np.nanmedian([R.vdist(a, b) for a in vec[ep] for b in vec["pre"]]) / pp
              for ep in ("june", "acute", "chronic") if vec.get(ep)}
    fin = {ep: np.isfinite(t).all(axis=(1, 2)) for ep, t in T.items()}
    ok = fin["june"] & fin["pre"] & fin["chronic"]
    dj, dc = direction(T["june"], T["pre"], P, mu, ok), direction(T["chronic"], T["pre"], P, mu, ok)
    row = {"animal": an, **{f"n_{ep}": len(v) for ep, v in by.items()}, "n_spouts": int(ok.sum()),
           "pre_to_pre": float(pp), "june_to_pre": to_pre["june"], "chronic_to_pre": to_pre["chronic"],
           "acute_to_pre": to_pre.get("acute", np.nan), "ratio_chronic_june": to_pre["chronic"] / to_pre["june"],
           "cos_june_chr": cos(dj, dc), "frac_along": float((dc @ dj) ** 2 / ((dj @ dj) * (dc @ dc))),
           "chance_cos": 1 / np.sqrt(len(dj)),
           "gap_june_days": float(np.median([p[2] - j[2] for j in by["june"] for p in by["pre"]])),
           "gap_chronic_days": float(np.median([c[2] - p[2] for p in by["pre"] for c in by["chronic"]]))}
    oka = ok & fin.get("acute", ok & False)
    if oka.any():
        row["cos_june_acute"] = cos(direction(T["june"], T["pre"], P, mu, oka), direction(T["acute"], T["pre"], P, mu, oka))
    return row


def main():
    if S.BASIS == "allen":
        sys.exit("state_space_june_drift: June exists only in the O2 bases -- run with WF_SS_BASIS=lnmf or svd")
    D = pd.read_csv(R.LANDING)
    L = {lab: g for lab, g in D[D.contact.notna()].groupby("label")}
    hook = lambda lab, Y, pos: (Y, pos) if S._epoch(lab, "") == "june" else R.relabel(lab, Y, pos, "contact", L)  # noqa: E731
    data, _ = S.load_sessions(hook, june=True)
    res = pd.DataFrame([animal_row(an, sess) for an, sess in sorted(data.items())])
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "june_drift.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(f"=== June drift reference ({S.ALIGN_LABEL[S.ALIGN]}, contact first licks, distances in pre-to-pre units) ===")
        print(res.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
