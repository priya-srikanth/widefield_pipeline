"""PRIMARY figures for the ITI / lick read-out results: mixed models WITH TIME (Priya 10-27: mixed-model versions of the
ITI / lick figures as the primary figures; the epoch-bar figures stay as supplementary).

Per measure, the SHARED models of wfield_local.mixed_models (nothing is fitted here; rule 9), sessions as observations,
animal random effects (ladder: intercept + step + slope -> intercept + step -> intercept -> fixed animal intercepts):
  time       y ~ post + post:lday      `post` = day-1 step vs pre, `post:lday` = recovery slope per e-fold of days
  behaviour  y ~ post + post:deficit   deficit = far-contra time to contact vs pre (mx.DEFICIT); `post` = the change
                                       still present at FULL behavioural recovery
Figures (mx.plot_fits: session values vs day / deficit, coloured by animal, fitted fixed effects), one per group x model,
<E>/iti_licks/figures/lmm_{time,behaviour}_<group>.png:
  readout     decoder r (lateral, sagittal, angle) / the animal's pre mean and lateral offset beyond the pull, ITI + cued
  response    ITI post-lick response / pre adopted response: key_mean, cortex, key ROIs (adopted), key_mean at 415 nm
  lickclass   first vs later licks (beyond the pull, landing), first - later, lick-order slopes; all spouts + far contra
  perspout    cued beyond-pull offset at far contra and close contra
  behaviour   cued lick angle offset from own pre landing -- TIME model only (indexing a behavioural measure by
              behavioural recovery is circular)
Session values come from the producing scripts' session tables via the readers in scripts/iti_mixed_models.py.
Fitted tables: <E>/iti_licks/lmm_figures/lmm_<group>.csv. Rule 8: lmm_table prints the per-animal epoch means first.
Usage: python scripts/iti_lmm_figures.py [--groups readout response ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_mixed_models as imm  # noqa: E402

from wfield_local import mixed_models as mx  # noqa: E402

SET = {"iti": "ITI licks", "trial": "cued licks"}
MM = "mm, + = contralesional"
RATIO = "/ animal's pre mean, 1 = pre"
RESP = "/ animal's pre adopted resp., 1 = pre"


def _readout() -> list:
    out = []
    for st in ("iti", "trial"):
        out.append((f"r_lat_{st}", f"lateral decoder r, {SET[st]} ({RATIO})", imm.where_r(st, "lateral")))
    for st in ("iti", "trial"):
        out.append((f"off_lat_{st}", f"lateral offset beyond the pull, {SET[st]}\n(change vs pre, {MM})",
                    imm.recovery_link(st, "bey_lat")))
    for tg, lab in (("sagittal", "sagittal"), ("angle", "angle")):
        for st in ("iti", "trial"):
            out.append((f"r_{tg[:3]}_{st}", f"{lab} decoder r, {SET[st]} ({RATIO})", imm.where_r(st, tg)))
    return out


def _response() -> list:
    out = [("resp_key_mean", "ITI post-lick response, key_mean ROI, adopted\n(" + RESP + ")",
            imm.hemo_resp("adopted", "key_mean")),
           ("resp_cortex", "ITI post-lick response, whole cortex, adopted\n(" + RESP + ")",
            imm.hemo_resp("adopted", "cortex"))]
    for roi in ("ALM_right", "ALM_left", "tjM1_right", "tjM1_left", "SSp-m_right", "SSp-m_left"):
        out.append((f"resp_{roi}", f"ITI post-lick response, {roi}, adopted\n({RESP})", imm.hemo_resp("adopted", roi)))
    out.append(("resp_key_mean_415", "ITI post-lick response, key_mean ROI, RAW 415 nm\n(" + RESP + ")",
                imm.hemo_resp("raw415", "key_mean")))
    return out


def _lickclass() -> list:
    out = []
    for scope in ("all spouts", "far contra"):
        sc = scope.replace(" ", "_")
        cells = []
        for val, vlab in (("beyond", "offset beyond the pull"), ("landing", "landing vs pre at the spout")):
            for st in ("trial", "iti"):
                for k in ("first", "later"):
                    cells.append((f"{val}_{k}", st, f"{vlab}, {SET[st]}, {k.upper()}\n({scope}; {MM})"))
        for st in ("trial", "iti"):
            cells.append(("beyond_diff", st, f"first - later, beyond the pull, {SET[st]}\n({scope}; {MM})"))
        for val in ("beyond", "landing", "decoded"):
            for st in ("trial", "iti"):
                cells.append((f"{val}_slope", st, f"lick-order slope, {val}, {SET[st]}\n({scope}; mm per lick)"))
        for col, st, title in cells:
            d = imm.lick_order(scope, st, col)
            if len(d):
                out.append((f"{sc}_{st}_{col}", title, d))
            else:
                print(f"  lickclass: {col} absent from lick_order_sessions.csv -- panel skipped")
    return out


def _perspout() -> list:
    return [(f"beyond_{sp}", f"offset beyond the pull, cued licks, {sp.replace('_', '-')}\n({MM})",
             imm.overcorrection(sp)) for sp in ("far_contra", "close_contra")]


def _behaviour() -> list:
    return [("aim_weighted", "weighted signed aim, cued licks (BEHAVIOUR)\n(deg vs own pre landing, + = contralesional)",
             imm.recovery_measure("weighted_signed")),
            ("aim_far_contra", "signed aim, far-contra cued licks (BEHAVIOUR)\n(deg vs own pre landing, + = contralesional)",
             imm.recovery_measure("far_contra_signed"))]


#: group -> (builder, models, ncols, figure title)
GROUPS = {"readout": (_readout, ("time", "behaviour"), 4, "ITI / cued read-out: decoder r and offset beyond the pull"),
          "response": (_response, ("time", "behaviour"), 3, "ITI post-lick cortical response"),
          "lickclass": (_lickclass, ("time", "behaviour"), 4, "first vs later licks, ITI and cued (rows 1-4 all spouts, "
                                                              "5-8 far contra)"),
          "perspout": (_perspout, ("time", "behaviour"), 2, "cued beyond-pull offset per spout"),
          "behaviour": (_behaviour, ("time",), 2, "behavioural aim (time model only)")}


def run_group(name: str, dfc: dict) -> pd.DataFrame:
    build, models, ncols, title = GROUPS[name]
    print(f"\n==================== {name} ====================")
    items = build()
    titles = {m: t for m, t, _ in items}
    L = pd.concat([d.assign(measure=m) for m, _, d in items], ignore_index=True).rename(columns={"value": "y"})
    L = mx.add_time_covariates(L.dropna(subset=["y"]))
    L = mx.add_deficit(L, dfc)
    T = mx.lmm_table(L, ["measure"], models=models, fdr_by=())
    missing = [m for m in titles if T.empty or m not in set(T.measure)]
    if missing:
        print(f"  NOT FITTED (too few sessions / animals): {missing}")
    if T.empty:
        return T
    T.insert(1, "group", name)
    T["title"] = T.measure.map(lambda m: titles[m].replace("\n", " "))
    pd.set_option("display.width", 240)
    print(mx.summary_table(T, ["measure"]).to_string())
    od = il.OUT / "lmm_figures"
    od.mkdir(parents=True, exist_ok=True)
    T.to_csv(od / f"lmm_{name}.csv", index=False)
    panels = [(titles[m], {"measure": m}) for m in titles]
    for model in models:
        mx.plot_fits(L, T, panels, model, il.OUT / "figures" / f"lmm_{model}_{name}.png", title=title,
                     ylabel="session value (units in panel title)", ncols=ncols)
    return T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    a = ap.parse_args()
    dfc = mx.deficit(il.OUT.parent / "qc")
    for g in a.groups:
        run_group(g, dfc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
