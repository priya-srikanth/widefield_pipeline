"""PRIMARY figures for the ITI / lick read-out results: mixed models WITH TIME (Priya 10-27: mixed-model versions of the
ITI / lick figures as the primary figures; the epoch-bar figures stay as supplementary).

Per measure, the SHARED models of wfield_local.mixed_models (nothing is fitted here; rule 9), sessions as observations,
animal random effects (ladder: intercept + step + slope -> intercept + step -> intercept -> fixed animal intercepts):
  time       y ~ post + post:lday      `post` = day-1 step vs pre, `post:lday` = recovery slope per e-fold of days
  behavior  y ~ post + post:deficit   deficit = the cued licks' weighted signed AIM (deg, pre - session; 10-27 late,
                                       Priya), FIRST post-cue lick by default (DECISIONS 2026-10-27 late; env
                                       WFIELD_AIM=all = all-lick sensitivity, every output tagged _aimall); `post` = the
                                       change still present at FULL recovery of aim (--deficit ttc = far-contra time to
                                       contact, the earlier default)
Figures (mx.plot_fits: session values vs day / deficit -- open markers, shape = animal, color = epoch -- with the
fitted fixed effects), one per group x model,
<E>/iti_licks/figures/lmm_{time,behavior}_<group>.png:
  readout     decoder r (lateral, sagittal, angle) / the animal's pre mean and the like-for-like (same-place) read-out,
              ITI + cued
  response    ITI post-lick response / pre adopted response: key_mean, cortex, key ROIs (adopted), key_mean at 415 nm
  lickclass   first vs later licks (like-for-like read-out, landing), lick-order slopes; all spouts + far contra
  perspout    like-for-like read-out per spout, cued + ITI
Decoder basis = iti_lick_order.SOURCE (Allen + ALM / tjM1 split by default since 10-27 late; its outputs are untagged, so
the figure stems are unchanged; plain Allen areas write *_plain).
  behavior   cued lick angle offset from own pre landing, WFIELD_AIM lick set -- TIME model only (indexing a
              behavioral measure by behavioral recovery is circular)
Session values come from the producing scripts' session tables via the readers in scripts/iti_mixed_models.py.
Fitted tables: <E>/iti_licks/lmm_figures/lmm_<group>.csv. Rule 8: lmm_table prints the per-animal epoch means first.
Usage: [WFIELD_AIM=all] python scripts/iti_lmm_figures.py [--groups readout response ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_mixed_models as imm  # noqa: E402
import iti_recovery_measure as rm  # noqa: E402

from wfield_local import mixed_models as mx  # noqa: E402

WOUT = il.OUT_W if il.ITI_SUBSET else il.OUT     # subset runs (dock, ...) write to their own folder (10-29)
SET = {"iti": "ITI licks", "trial": "cued licks"}
MM = "mm, + = contralesional"
RATIO = "/ animal's pre mean, 1 = pre"
RESP = "/ animal's pre adopted resp., 1 = pre"


def _readout() -> list:
    out = []
    for st in ("iti", "trial"):
        out.append((f"r_lat_{st}", f"lateral decoder r, {SET[st]} ({RATIO})", imm.where_r(st, "lateral")))
    for st in ("iti", "trial"):
        out.append((f"place_{st}", f"like-for-like read-out (same place), {SET[st]}\n(vs pre licks landing there, {MM})",
                    imm.matched("all spouts", "all", st)))
    out.append(("place_trial_first", f"like-for-like read-out, FIRST cued lick\n(same place, {MM})",
                imm.matched("all spouts", "first", "trial")))
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
        for st in ("trial", "iti"):
            for k in ("first", "later"):
                d = imm.matched(scope.replace("far contra", "far_contra"), k, st)
                if len(d):
                    out.append((f"{sc}_{st}_place_{k}", f"like-for-like read-out (same place), {SET[st]}, {k.upper()}"
                                f"\n({scope}; {MM})", d))
        for st in ("trial", "iti"):
            for k in ("first", "later"):
                cells.append((f"landing_{k}", st, f"landing vs pre at the spout, {SET[st]}, {k.upper()}\n({scope}; {MM})"))
        for val in ("landing", "decoded"):
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
    import iti_lick_order as lo
    return [(f"place_{st}_{sp}", f"like-for-like read-out, {SET[st]}, {sp.replace('_', ' ')}\n(same place, {MM})",
             imm.matched(sp, "all", st)) for st in ("trial", "iti") for sp in lo.POS]


def _behaviour() -> list:
    w = rm.AIM_LABEL[rm.AIM]
    return [("aim_weighted", f"weighted signed aim, {w} (BEHAVIOR)\n(deg vs own pre landing, + = contralesional)",
             imm.recovery_measure("weighted_signed")),
            ("aim_far_contra", f"signed aim, far contra, {w} (BEHAVIOR)\n(deg vs own pre landing, + = contralesional)",
             imm.recovery_measure("far_contra_signed"))]


#: group -> (builder, models, ncols, figure title)
GROUPS = {"readout": (_readout, ("time", "behaviour", "epoch"), 4, "ITI / cued read-out: decoder r and like-for-like (same-place) "
                                                          "read-out"),
          "response": (_response, ("time", "behaviour"), 3, "ITI post-lick cortical response"),
          "lickclass": (_lickclass, ("time", "behaviour"), 4, "first vs later licks, ITI and cued (rows 1-4 all spouts, "
                                                              "5-8 far contra)"),
          "perspout": (_perspout, ("time", "behaviour"), 6, "like-for-like read-out per spout (row 1 cued, row 2 ITI)"),
          "behaviour": (_behaviour, ("time",), 2, "behavioral aim (time model only)")}


def lo_tag() -> str:
    """Decoder-basis tag + the aim lick-set tag (_aimall = all-lick sensitivity; DECISIONS 2026-10-27 late)."""
    import iti_lick_order as lo
    return lo.SRC_TAG + rm.AIM_TAG


XLAB = {"aim": f"aim deficit: {rm.AIM_LABEL[rm.AIM]} weighted signed aim, pre - session (deg; 0 = pre aim, + = aim "
               "shifted ipsilesional)",
        "ttc": None}
DEFICIT = "aim"


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
    T["aim_licks"] = rm.AIM
    T["title"] = T.measure.map(lambda m: titles[m].replace("\n", " "))
    pd.set_option("display.width", 240)
    print(mx.summary_table(T, ["measure"]).to_string())
    od = WOUT / "lmm_figures"
    od.mkdir(parents=True, exist_ok=True)
    T.to_csv(od / f"lmm_{name}{lo_tag()}.csv", index=False)
    panels = [(titles[m], {"measure": m}) for m in titles]
    for model in models:
        if model == "epoch":                    # supplementary per-epoch view of the same mixed model
            mx.plot_epoch(L, T, panels, WOUT / "figures" / f"lmm_epoch_{name}{lo_tag()}.png", title=title,
                          ylabel="session value (units in panel title)", ncols=ncols)
            continue
        mx.plot_fits(L, T, panels, model, WOUT / "figures" / f"lmm_{model}_{name}{lo_tag()}.png", title=title,
                     ylabel="session value (units in panel title)", ncols=ncols, xlabel=XLAB[DEFICIT])
    return T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--deficit", default="aim", choices=["aim", "ttc"])
    a = ap.parse_args()
    # behaviour-model deficit (Priya 10-27: "mixed models should use our new measure - the angle one"): the cued licks'
    # weighted signed aim (iti_recovery_measure.py; far contra x3, far mid x2, others x1), pre - session in degrees
    # (+ = aim shifted ipsilesional, 0 = pre level). --deficit ttc keeps the earlier far-contra time to contact.
    global DEFICIT
    DEFICIT = a.deficit
    if a.deficit == "aim":
        # FIRST post-cue lick weighted aim by default (Priya 10-27: reliability 0.98, shows the chronic deficit that later
        # corrective licks hide); THE definition + the WFIELD_AIM switch live in iti_recovery_measure (rule 9)
        dfc = rm.aim_deficit()
    else:
        from wfield_local import run_mode as _rm
        dfc = mx.deficit(_rm.deficit_qc_dir())   # THIS run's trial_attempts table (was iti_licks/../qc)
    for g in a.groups:
        run_group(g, dfc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
