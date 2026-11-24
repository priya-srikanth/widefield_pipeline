"""Recovery epochs (Supplementary Figure S4C) and the Fig 4F hit-rate input on the CAMERA WORKING PERIOD (CC-B 11-23).

Production: `epoch_audit` / `epochs.derive_*_boundaries` read `cohort_session_metrics.csv` ``hit__<pos>``, the DAQ
spout-contact hit rate on the behavior pipeline's `engaged` trials (`spout_behavior.session_metrics` ->
`reference_engagement`: judged at the reference spouts close_L / close_center only). Fig 4F joins the same column.

Here the ONLY change is the trial gate: ``hit__<pos>`` = fraction of the session's working-period trials at that
position whose four-camera class is ``contact`` (spout contact). Checked: on the engagement-flag trials this
fraction reproduces ``hit__far_R`` (107 sessions, max |diff| 0.005). Sessions without a class table (the June pre
sessions 0606-0608, which the derivation's full pre set includes) keep the production value, and say so
(``wp_source``). Free-reward trials: none in these sessions.

Writes under --out:
  cohort_session_metrics_working_period.csv  production table with hit__<pos> replaced (+ hit_flag__<pos>, n_wp__<pos>,
                                             wp_source); the --metrics input for `c2_manuscript_correlations.py`
  s4c_series_working_period.csv              per post/pre session: far_R, far_L hit rate (flag vs working period),
                                             as a fraction of the animal's pre mean (the S4C traces)
  s4c_boundaries_working_period.csv          per animal: acute range, subacute_from, chronic_from derived on each
                                             gate, and the stored boundaries (animals.yaml / epoch_boundaries)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
MANU = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}


def main(argv=None) -> int:
    from wfield_local import config, epoch_audit, epochs
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tally", required=True, help="gate_tally_trials.csv.gz (working_gate_tally_ccB1123.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C = pd.read_csv(a.tally)
    W = C[C.in_wp]
    hw = W.assign(c=W.cls.eq("contact")).groupby(["label", "position"]).c.agg(["mean", "size"]).reset_index()
    M0 = pd.read_csv(epoch_audit._cohort_path())
    M = M0.copy()
    M["label"] = M.animal + "_" + M.date.astype(str).str[-4:]
    M["wp_source"] = np.where(M.label.isin(set(C.label)), "working_period", "production (no class table)")
    for p in POS:
        M[f"hit_flag__{p}"] = M[f"hit__{p}"]
        s = hw[hw.position == p].set_index("label")
        has = M.label.isin(s.index)
        M.loc[has, f"hit__{p}"] = M.loc[has, "label"].map(s["mean"]).to_numpy()
        M.loc[has, f"n_wp__{p}"] = M.loc[has, "label"].map(s["size"]).to_numpy()
    M.drop(columns="label").to_csv(out / "cohort_session_metrics_working_period.csv", index=False)

    def tables(df):
        df = df.sort_values("n_engaged").drop_duplicates(["animal", "date"], keep="last")
        hit, lick = {}, {}
        for _i, r in df.iterrows():
            lab = f"{r['animal']}_{str(int(r['date']))[4:]}"
            hit[lab] = {p: float(r[f"hit__{p}"]) for p in POS if pd.notna(r[f"hit__{p}"])}
            if pd.notna(r["lpt__far_R"]):
                lick[lab] = {"far_R": float(r["lpt__far_R"])}
        return hit, lick

    res = {}
    with epoch_audit.full_cohort():
        for g, df in (("flag", M0), ("wp", M.drop(columns="label"))):
            hit, lick = tables(df)
            res[g] = (hit, epochs.derive_acute_boundaries(hit), epochs.derive_chronic_boundaries(hit, lick))
        rows = []
        for an in sorted(epochs.EPOCH_SPEC):
            spec = epochs.spec_for(an) or {}
            r = {"animal": an, "mouse": MANU.get(an, an), "stored_acute": spec.get("acute"),
                 "stored_subacute_from": spec.get("subacute_from"), "stored_chronic_from": spec.get("chronic_from")}
            for g in ("flag", "wp"):
                ac, ch = res[g][1].get(an, {}), res[g][2].get(an, {})
                r[f"{g}_acute"] = ac.get("acute")
                r[f"{g}_subacute_from"] = ac.get("subacute_from")
                r[f"{g}_acute_threshold"] = ac.get("threshold")
                r[f"{g}_relapse"] = ac.get("relapse")
                r[f"{g}_chronic_from"] = ch.get("derived_day")
                r[f"{g}_chronic_sd_pre"] = (ch.get("hit") or {}).get("sd_pre")
                r[f"{g}_chronic_level"] = (ch.get("hit") or {}).get("level")
            rows.append(r)
        B = pd.DataFrame(rows)
        # S4C series: every pooled session, both gates, fraction of own pre mean (the derivation's pre set)
        S = []
        for an in sorted(epochs.EPOCH_SPEC):
            pre = [lab for lab in config.phase_labels("pre") if config.animal_of(lab) == an]
            for lab in sorted(config.pooled_labels(an), key=lambda x: x.split("_")[-1]):
                rec = {"label": lab, "animal": an, "mouse": MANU.get(an, an), "epoch": epochs.epoch_of(lab),
                       "day": epochs.days_since_stroke(lab), "in_pre_baseline": lab in pre,
                       "has_class_table": lab in set(C.label)}
                for g in ("flag", "wp"):
                    hit = res[g][0]
                    for p in ("far_R", "far_L"):
                        base = [hit[x][p] for x in pre if x in hit and p in hit[x]]
                        v = hit.get(lab, {}).get(p)
                        rec[f"hit_{g}__{p}"] = v
                        rec[f"rel_{g}__{p}"] = (v / np.mean(base)) if (v is not None and base) else np.nan
                S.append(rec)
    S = pd.DataFrame(S)
    S.to_csv(out / "s4c_series_working_period.csv", index=False)
    B.to_csv(out / "s4c_boundaries_working_period.csv", index=False)
    pd.set_option("display.width", 250)
    print(B.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
