"""Per-spout reliability of the within-spout lick deviation, and DISATTENUATED per-spout slopes (Priya 10-21: "sure add
the corrected version - but why not for far contra?").

WHY. Measurement noise in the lick angle is roughly constant, so when the true within-spout scatter shrinks (PS94 /
PS95 far_ipsi halve by chronic) a larger share of the measured deviation is noise and the slope is biased toward zero.
Per session x lateral spout: reliability = Spearman-Brown of r(odd-lick, even-lick within-spout deviations), on the
trials the within-spout analysis uses (cam4, lick-aligned, >= 2 usable licks). Corrected slope = b / reliability,
reliability floored at RMIN (below it the correction is unstable and the row is NaN). The odd / even split counts real
lick-to-lick variation as noise, so the corrected slope is an UPPER bound (within_spout_deviation.split_half).

Joins the z-scored slopes (within_spout_mixed.add_z) of the given folder; prints per-animal tables for all four
lateral spouts. Output: <folder>/spout_reliability.csv (+ _slopes.csv).
Usage: WFIELD_INPUT_CACHE=... PYTHONPATH=. python scripts/spout_reliability.py [--folder within_spout_deviation_lick_contact]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

LATERAL = ("close_contra", "close_ipsi", "far_contra", "far_ipsi")
RMIN = 0.2


def one(label):
    from scripts import within_spout_deviation as W
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.paths import PathResolver
    B = ce.load_behaviour(config.animal_of(label), "2026" + label.split("_")[-1], PathResolver())
    E = B.trials[B.trials.engaged & np.isfinite(B.trials.cue_s)].reset_index(drop=True)
    T = W.trial_table(B, E, "", "lick")
    rows = []
    for p in LATERAL:
        G = T[(T.pos == p)].dropna(subset=["lat_odd", "lat_even"])
        if len(G) < W.MIN_TRIALS:
            rows.append({"label": label, "spouts": p, "n_rel": len(G), "r_half": np.nan, "rel": np.nan})
            continue
        o, e = G.lat_odd - G.lat_odd.mean(), G.lat_even - G.lat_even.mean()
        r = float(np.corrcoef(o, e)[0, 1])
        rows.append({"label": label, "spouts": p, "n_rel": len(G), "r_half": r,
                     "rel": 2 * r / (1 + r) if r > -1 else np.nan})
    return rows


def main(argv=None) -> int:
    from scripts import within_spout_mixed as wm
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="within_spout_deviation_lick_contact")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    folder = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc" / a.folder
    S = pd.read_csv(folder / "within_spout_deviation.csv")
    labels = sorted(S.label.unique())
    got, fail = ak.fan_sessions(labels, one, jobs=a.jobs)
    for it, err in fail:
        print("FAILED", it, err)
    R = pd.DataFrame([r for _, rr in got for r in rr])
    R.to_csv(folder / "spout_reliability.csv", index=False, float_format="%.4g")
    D = wm.add_z(S[S.variant == "mean"])
    D = D[D.spouts.isin(LATERAL) & D.area.str.match(r"^(MOs|SSp-m)_(left|right)$")].merge(R, on=["label", "spouts"])
    sg = np.where(D.spouts.str.endswith("contra"), 1.0, -1.0)
    D["mid_z"] = D.b_z * sg
    D["mid_z_corr"] = np.where(D.rel >= RMIN, D.mid_z / D.rel, np.nan)
    # PER-POSITION DIRECTION (Priya 10-21: per-position, not toward-midline): + = licks toward the mouse's LEFT (ipsi)
    D["dir_z"] = D.b_z
    D["dir_z_corr"] = np.where(D.rel >= RMIN, D.dir_z / D.rel, np.nan)
    D["epoch"] = D.label.map(epochs.epoch_of)
    D.to_csv(folder / "spout_reliability_slopes.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 230)
    order = ["pre", "acute", "subacute", "chronic"]
    rel = R.assign(animal=R.label.str.split("_").str[0], epoch=R.label.map(epochs.epoch_of))
    print("=== per-spout reliability of the within-spout deviation (Spearman-Brown, session medians)")
    print(rel.groupby(["spouts", "animal", "epoch"]).rel.median().unstack("epoch").reindex(columns=order).round(2)
          .to_string())
    D["reg"] = D.area.str.split("_").str[0]
    for col, title in (("dir_z", "z-scored per-position DIRECTION slope (+ = licks toward the mouse's LEFT)"),
                       ("dir_z_corr", f"DISATTENUATED direction: / per-spout reliability (rel >= {RMIN})")):
        print(f"\n=== {title}, hemispheres averaged, session means")
        print(D.groupby(["spouts", "reg", "animal", "epoch"])[col].mean().unstack("epoch").reindex(columns=order)
              .round(3).to_string())
    print(f"\n-> {folder / 'spout_reliability_slopes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
