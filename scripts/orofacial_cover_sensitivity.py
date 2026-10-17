"""Sensitivity of the cohort change curves to the coverage rule (Priya 10-17: "how did you decide ... at least half of an
epoch's sessions, and at least 3").

    python scripts/orofacial_cover_sensitivity.py [--run lp_r4_heatmap_20261013__clean_v35] [--fracs 0 0.25 0.5 0.75]

For each epoch-level threshold (`min_cover_frac`; 0 = no epoch rule, min n then 1) this re-runs exactly what the cohort
figures do -- `_animal_delta_curves` (session-weighted) and the cluster permutation vs pre -- on the saved cohort tables,
for every trace family x in-mouth convention x variable x position x post epoch, and reports

  * drawn bins (bins of the epoch's mean that survive the rule),
  * significant bins and cells (variable x position x epoch) with any significant cluster,
  * per threshold, the cells whose significance differs from the current 0.5 rule.

The per-animal rule (>= min(3, n sessions) of an animal's sessions) is held fixed. Outputs to
<LP_analysis>/<run>/cohort/qc/cover_sensitivity/: summary.csv, cells.csv, changed_cells.csv, SUMMARY.md and, per
threshold, the tongue-out cue-aligned and lick-phase figures for visual comparison.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wfield_local import orofacial_cohort as oc
from wfield_local.orofacial_stats import cluster_permutation

FAMILIES = {   # family -> (table key, x column, {convention: [variables]})
    "cue": ("CT", "t_ms", {"excl": ["protrusion_out_px", "lat_out_px", "angle_deg"],
                           "incl": ["protrusion_px", "lat_px", "angle_mouth0_deg"]}),
    "lick_phase": ("PHa", "t_ms", {"excl": ["protrusion_out_px", "lat_out_px", "angle_deg"],
                                   "incl": ["protrusion_px", "lat_px", "angle_mouth0_deg"]}),
    "phase_norm": ("PHf", "phase", {"excl": ["protrusion_out_px", "angle_deg"],
                                    "incl": ["protrusion_px", "angle_mouth0_deg"]}),
}


def _set_rule(frac: float) -> None:
    oc.MIN_COVER_FRAC = frac
    oc.MIN_COVER_N = 3 if frac > 0 else 1


def cells(T: dict, frac: float) -> list[dict]:
    _set_rule(frac)
    rows = []
    for fam, (key, xcol, conv) in FAMILIES.items():
        D = T[key]
        for im, cols in conv.items():
            for col in cols:
                for p in oc.POS_ORDER:
                    cur = oc._animal_delta_curves(D, xcol, col, p, "session")
                    if "pre" not in cur:
                        continue
                    Mp, _, anp = cur["pre"]
                    for e in oc.EPOCH_ORDER[1:]:
                        if e not in cur:
                            continue
                        Me, _, ane = cur[e]
                        c_ = Mp.columns.intersection(Me.columns)
                        drawn = int(np.isfinite(Me[c_].mean().to_numpy(float)).sum())
                        mask = cluster_permutation(np.vstack([Mp[c_].to_numpy(float), Me[c_].to_numpy(float)]),
                                                   np.r_[np.zeros(len(Mp), bool), np.ones(len(Me), bool)],
                                                   np.r_[anp, ane], n_perm=oc.CLUSTER_PERM)
                        rows.append({"frac": frac, "family": fam, "inmouth": im, "var": col, "pos": p, "epoch": e,
                                     "n_bins": len(c_), "drawn_bins": drawn, "sig_bins": int(mask.sum()),
                                     "sig": bool(mask.any()), "sig_bins_list": ",".join(f"{x:g}" for x in c_[mask])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75])
    a = ap.parse_args()
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    base = oo.root(rv) / a.run / "cohort"
    tab = base / "tables" / "prepost"
    out = base / "qc" / "cover_sensitivity"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda n: pd.read_csv(tab / n, dtype={"date": str})                    # noqa: E731
    CT = rd("cue_aligned_traces.csv")
    CTw = CT.pivot_table(index=["animal", "date", "epoch", "pos", "t_ms"], columns="var", values="value").reset_index()
    PH, PHN = rd("lick_phase.csv"), rd("lick_phase_normalised.csv")
    T = {"CT": CTw, "PHa": PH[PH.set == "postcue"], "PHf": PHN[PHN["mode"] == "first_multi"]}

    C = pd.DataFrame([r for f in a.fracs for r in cells(T, f)])
    C.to_csv(out / "cells.csv", index=False)
    S = (C.groupby(["family", "inmouth", "frac"])
          .agg(cells=("sig", "size"), drawn_bins=("drawn_bins", "sum"), total_bins=("n_bins", "sum"),
               sig_bins=("sig_bins", "sum"), sig_cells=("sig", "sum")).reset_index())
    S["drawn_frac"] = S.drawn_bins / S.total_bins
    S.to_csv(out / "summary.csv", index=False, float_format="%.3f")
    key = ["family", "inmouth", "var", "pos", "epoch"]
    ref = C[C.frac == 0.5].set_index(key)
    ch = []
    for f in a.fracs:
        if f == 0.5:
            continue
        x = C[C.frac == f].set_index(key)
        j = x.join(ref, rsuffix="_ref", how="inner")
        d = j[(j.sig != j.sig_ref) | ((j.sig_bins - j.sig_bins_ref).abs() >= 5)]
        for k, r in d.iterrows():
            ch.append({"frac": f, **dict(zip(key, k)), "sig": r.sig, "sig_ref_0.5": r.sig_ref, "sig_bins": r.sig_bins,
                       "sig_bins_ref_0.5": r.sig_bins_ref, "drawn_bins": r.drawn_bins,
                       "drawn_bins_ref_0.5": r.drawn_bins_ref})
    CH = pd.DataFrame(ch)
    CH.to_csv(out / "changed_cells.csv", index=False)

    lines = ["# Coverage-rule sensitivity (cohort change curves)", "",
             "Rule tested: a bin of an epoch's mean change curve is kept only when >= `frac` of the epoch's sessions AND "
             ">= 3 of them have data there (frac 0 = no epoch rule). Per-animal rule fixed (>= 3 of the animal's "
             "sessions). Session-weighted; significance = the figures' cluster permutation (500 perms, within-animal).",
             "", "## Summary per family / convention", "", "```", S.to_string(index=False, float_format="%.3f"), "```", "",
             f"## Cells whose significance differs from frac 0.5 (or by >= 5 significant bins): {len(CH)}", ""]
    if len(CH):
        lines += ["```", CH.to_string(index=False), "```"]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    t = "coverage sensitivity"
    for f in a.fracs:
        _set_rule(f)
        tag = f"frac{int(round(f * 100)):02d}"
        oc.fig_cohort_delta([(CTw, "t_ms", "protrusion_out_px", "protrusion (px)", False),
                             (CTw, "t_ms", "lat_out_px", "lateral tongue position (px; + = ipsi)", False),
                             (CTw, "t_ms", "angle_deg", "tongue angle (deg; + = ipsi)", False)],
                            out / f"cue_inmouth_excl_{tag}.png", t,
                            f"cue-aligned, in-mouth EXCLUDED; coverage rule frac = {f} (min n = {oc.MIN_COVER_N})",
                            "ms from cue", "session")
        oc.fig_cohort_delta([(T["PHa"], "t_ms", "protrusion_out_px", "protrusion (px)", False),
                             (T["PHa"], "t_ms", "lat_out_px", "lateral tip position (px; + = L)", False),
                             (T["PHa"], "t_ms", "angle_deg", "tongue angle (deg; + = L)", False)],
                            out / f"lick_phase_inmouth_excl_{tag}.png", t,
                            f"lick-peak-aligned, in-mouth EXCLUDED; coverage rule frac = {f} (min n = {oc.MIN_COVER_N})",
                            "ms from lick peak", "session")
    print("->", out)


if __name__ == "__main__":
    main()
