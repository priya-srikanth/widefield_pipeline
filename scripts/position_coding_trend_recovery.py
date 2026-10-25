"""Is the chronic gain in cortical position coding beyond the pre-stroke practice trend, and does it lead or follow
behavioural recovery? (Priya 2026-10-19: tests 1 and 4 of the "what does this reveal about the biology" list.)

Position coding per session (cortex, lick window unless noted), from the outputs already on the share:
  ceiling   nokin:category - nokin:none        (6-category ceiling; `movement_first_subset-nokin`)
  spread    RMS over the six spouts of the signed position deviation (respdev, cue 0.5-1.5 s; main run)
  spread_enl  the same in the pre-cue window (spout arrival .. cue)
Behaviour per session (engaged trials, licks 0 .. 3.5 s after the cue, LP run licks.csv):
  contact_far_contra   fraction of licks at far contra (far_R; lesions LEFT) that touch the spout
  contact_all          the same over all six spouts

TEST 1 (practice trend): per animal, a line through the PRE sessions (August-only, primary; June + August as the
long-span sensitivity -- June is otherwise superseded, here it only lengthens the pre-stroke time axis), extrapolated to
each chronic session's day; the chronic excess = observed - predicted, with the prediction's t-interval. Also the
within-chronic slope against the pre slope (the F18 drift-vs-reorganisation logic: a gain that merely continues the
pre-stroke rate is practice / drift, not recovery).

TEST 4 (lead / lag): per animal, each measure's post-stroke course as the fraction of its acute -> chronic change
(acute = mean of the acute sessions, plateau = mean of the last 4 chronic sessions), on a 2-session running mean; the
day it first crosses 50 %. Neural earlier than behaviour = cortical gain could support recovery; later = it follows it.
Undefined when the measure does not change (|plateau - acute| below its pre-stroke session SD).

Output: <imaging_encoding>/qc/position_coding_trend_recovery.csv (+ the printed per-animal tables; rule 8).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

POS6 = ("close_contra", "close_mid", "close_ipsi", "far_contra", "far_mid", "far_ipsi")
RESP_WIN = 3.5
#: source encoder runs (10-25: the O2 final-frame runs -- 3-D polar exec, combined licks, rebuilt attempts gate, 107
#: August+ sessions, so the "June+August pre" baseline equals August here; the 10-19 cam4 runs `movement_first_subset-nokin`
#: / `movement_first` were STALE, OUTPUT_AUDIT 2a)
NOKIN_RUN = "movement_first_subset-nokin_exec-polar_licks-combined_bins"
MAIN_RUN = "movement_first_exec-polar_licks-combined"


def _enc_root():
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return oo.root(PathResolver()) / oc.RUN / "imaging_encoding"


def neural(E: Path, nokin: str = NOKIN_RUN, main: str = MAIN_RUN) -> pd.DataFrame:
    rows = {}
    for f in glob.glob(str(E / nokin / "sessions" / "*_regions.csv")):
        R = pd.read_csv(f)
        R = R[(R.area == "cortex") & (R.window == "lick")].set_index("model").r2
        if {"nokin:category", "nokin:none"} <= set(R.index):
            rows.setdefault(Path(f).stem[:-8], {})["ceiling"] = float(R["nokin:category"] - R["nokin:none"])
    for f in glob.glob(str(E / main / "sessions" / "*.npz")):
        z = np.load(f)
        mk = [str(k) for k in z["map_keys"]]
        for w, col in (("c0.5-1.5", "spread"), ("enl", "spread_enl")):
            try:
                S = np.stack([z["maps"][mk.index(f"respdev:{p}|{w}")] for p in POS6])
            except ValueError:
                continue
            rows.setdefault(Path(f).stem, {})[col] = float(np.nanmean(np.sqrt(np.nanmean(S ** 2, 0))))
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("label").reset_index()


def behaviour(labels) -> pd.DataFrame:
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    out = []
    for lab in labels:
        animal, date = config.animal_of(lab), "2026" + lab.split("_")[-1]
        sd = oo.session_dir(rv, oc.RUN, animal, date)
        if not (sd / "licks.csv").exists():
            continue
        L = pd.read_csv(sd / "licks.csv", usecols=["trial_id", "position", "t_ms", "contact"])
        keep, _gate = oc.engagement_keep(animal, date, sd)
        if keep is not None:
            L = L[L.trial_id.isin(keep)]
        L = L[(L.t_ms >= 0) & (L.t_ms <= RESP_WIN * 1000)]
        far_c = L[L.position == "far_R"]                      # lesions LEFT -> contralesional = right
        out.append({"label": lab, "contact_far_contra": float(far_c.contact.mean()) if len(far_c) else np.nan,
                    "contact_all": float(L.contact.mean()) if len(L) else np.nan, "n_far_contra": len(far_c)})
    return pd.DataFrame(out)


def trend_test(D: pd.DataFrame, col: str, pre_mask) -> dict:
    from scipy import stats
    P = D[pre_mask & D[col].notna()]
    C = D[(D.epoch == "chronic") & D[col].notna()]
    if len(P) < 3 or len(C) < 2:
        return {}
    sl, ic, _r, _p, _se = stats.linregress(P.day, P[col])
    n = len(P)
    resid = P[col] - (ic + sl * P.day)
    s = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    xm, sxx = P.day.mean(), ((P.day - P.day.mean()) ** 2).sum()
    dc = C.day.mean()
    pred = ic + sl * dc
    se_pred = s * np.sqrt(1 / n + (dc - xm) ** 2 / sxx + 1 / len(C))    # chronic MEAN vs the extrapolated line
    tcrit = stats.t.ppf(0.975, n - 2)
    csl = stats.linregress(C.day, C[col]).slope
    return {"pre_level": float(P[col].mean()), "pre_slope_per_day": float(sl), "pre_span_days": float(np.ptp(P.day)),
            "chronic_mean": float(C[col].mean()), "chronic_pred": float(pred),
            "excess": float(C[col].mean() - pred), "excess_ci": float(tcrit * se_pred),
            "beyond_trend": bool(C[col].mean() - pred > tcrit * se_pred), "chronic_slope_per_day": float(csl)}


def half_recovery_day(D: pd.DataFrame, col: str) -> float:
    pre = D[(D.epoch == "pre") & (D.label.str[-4:] >= "0806")][col].dropna()
    post = D[(D.day > 0) & D[col].notna()].sort_values("day")
    ac = post[post.epoch == "acute"][col]
    ch = post[post.epoch == "chronic"][col].tail(4)
    if len(ac) == 0 or len(ch) == 0 or len(pre) < 2:
        return np.nan
    a, b = ac.mean(), ch.mean()
    if abs(b - a) < pre.std():
        return np.nan
    frac = (post[col].rolling(2, min_periods=1).mean() - a) / (b - a)
    after = post[post.epoch != "acute"]
    hit = after.index[frac.loc[after.index] >= 0.5]
    return float(post.loc[hit[0], "day"]) if len(hit) else np.nan


def main(argv=None) -> int:
    import argparse

    from wfield_local import epochs, writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nokin", default=NOKIN_RUN, help="encoder run with nokin:category / nokin:none (ceiling)")
    ap.add_argument("--main", default=MAIN_RUN, help="main encoder run with respdev maps (spread)")
    a = ap.parse_args(argv)
    E = _enc_root()
    N = neural(E, a.nokin, a.main)
    N["src_nokin"], N["src_main"] = a.nokin, a.main
    N["animal"] = N.label.str[:4]
    N["epoch"] = N.label.map(epochs.epoch_of)
    N["day"] = N.label.map(epochs.days_since_stroke)
    B = behaviour(N.label)
    D = N.merge(B, on="label", how="left").sort_values(["animal", "day"]).reset_index(drop=True)
    pd.set_option("display.width", 220)
    rows = []
    for col in ("ceiling", "spread", "spread_enl"):
        for base, pre_mask in (("August pre", (D.epoch == "pre") & (D.label.str[-4:] >= "0806")),
                               ("June+August pre", D.epoch == "pre")):
            for a, g in D.groupby("animal"):
                r = trend_test(g, col, pre_mask.loc[g.index])
                if r:
                    rows.append({"measure": col, "baseline": base, "animal": a, **r})
    T = pd.DataFrame(rows)
    print("TEST 1 -- chronic level vs the extrapolated PRE-STROKE trend (per animal; excess = chronic mean - predicted,"
          " +- 95 % prediction interval)")
    show = ["measure", "baseline", "animal", "pre_span_days", "pre_level", "pre_slope_per_day", "chronic_mean",
            "chronic_pred", "excess", "excess_ci", "beyond_trend", "chronic_slope_per_day"]
    print(T[show].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    H = []
    for a, g in D.groupby("animal"):
        H.append({"animal": a, **{f"half_day:{c}": half_recovery_day(g, c)
                                  for c in ("ceiling", "spread", "spread_enl", "contact_far_contra", "contact_all")}})
    H = pd.DataFrame(H)
    print("\nTEST 4 -- day each measure first reaches 50 % of its acute -> chronic change (2-session running mean; NaN ="
          " no change beyond the pre-stroke session SD)")
    print(H.to_string(index=False))
    print("\nper-session series (post-stroke):")
    print(D[D.day > 0][["animal", "day", "epoch", "ceiling", "spread", "contact_far_contra", "contact_all"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = E / "qc"
    writeguard.assert_writable(out)
    out.mkdir(exist_ok=True)
    D.to_csv(out / "position_coding_trend_recovery_sessions.csv", index=False, float_format="%.5g")
    T.to_csv(out / "position_coding_trend_recovery.csv", index=False, float_format="%.5g")
    H.to_csv(out / "position_coding_half_recovery.csv", index=False, float_format="%.5g")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
