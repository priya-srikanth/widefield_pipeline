"""MIXED-EFFECTS models for the headline ITI / lick read-out results (Priya 10-27: "mixed-effects model with sessions
nested in animals ... for any headline result" -> "yes").

Per measure, the SHARED models of wfield_local.mixed_models (time: y ~ post + post:lday; behaviour: y ~ post +
post:deficit, deficit = the cued licks' weighted signed AIM, pre - session in degrees, FIRST lick by default --
DECISIONS 2026-10-27 late; WFIELD_AIM=all = the all-lick sensitivity, outputs tagged _aimall; it was far-contra time to
contact before; epoch: y ~ C(epoch)), sessions as observations, random-effects
ladder per animal (10-27 late: was an epoch-only random-intercept model written here -- replaced, rule 9). Reported: each post epoch's coefficient (change vs pre), 95 % Wald CI and p, beside the
paired nested bootstrap used on the page (boot_delta) for the same session values. With 4 animals the random-effect
variance is poorly estimated, so the Wald p is anti-conservative relative to the bootstrap; both are shown, and the
per-animal direction count (rule 8).
Measures (session values, all on the frozen pre-stroke decoder with z-scored Allen-area features -- ALM / tjM1 split
basis by default since 10-27 late, iti_licks.AREA_BASIS -- unless stated):
  r_lat_iti / r_lat_cued      lateral r / the animal's pre mean (zscore_check/where_decoder_z.csv)
  off_lat_iti / off_lat_cued  lateral LIKE-FOR-LIKE offset (same place; replaces 'beyond the pull', Priya 10-27 late),
                              change vs pre (recovery_link_sessions_withpre_angle_z, column lfl_lat)
  off_farcontra_cued          far-contra cued like-for-like offset (overcorrection_sessions.csv, column lfl)
  resp_key                    ITI post-lick response, key ROIs, / the animal's pre (hemo_check, adopted variant)
  aim_weighted                behaviour: weighted signed angle offset from own pre landing, WFIELD_AIM lick set
                              (iti_recovery_measure.sessions_csv: first lick = PRIMARY, all licks = sensitivity)
Output <E>/iti_licks/mixed_models<basis tag><aim tag>.csv (+ _boot). Usage: [WFIELD_AIM=all] python scripts/iti_mixed_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_recovery_measure as rm  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

EP = ["acute", "subacute", "chronic"]


COLS = ["animal", "epoch", "label", "value"]
# decoder session tables are read from the subset folder in a subset run (dock, ...: iti_dock_where_preds,
# iti_lick_order, lick_matched_readout write there); il.OUT otherwise (unchanged)
RIN = il.OUT_W if il.ITI_SUBSET else il.OUT


# --- session-level readers, one per producing table (shared with scripts/iti_lmm_figures.py; one row per session) ---
def where_r(st: str, target: str = "lateral") -> pd.DataFrame:
    """Decoder r (Allen areas, z-scored features) for lick set `st` (iti / trial) / the animal's pre mean."""
    d = pd.read_csv(RIN / "zscore_check" / il.tagged("where_decoder_z.csv"))
    g = d[(d.source == "areas") & (d.target == target) & (d.set == st)].copy()
    g["value"] = g.r / g.animal.map(g[g.epoch == "pre"].groupby("animal").r.mean())
    return g[COLS]


def recovery_link(st: str, col: str = "lfl_lat") -> pd.DataFrame:
    """recovery_link_sessions_withpre_angle_z.csv column `col` (lfl_lat = lateral like-for-like offset, change vs pre)."""
    r = pd.read_csv(il.OUT / il.tagged("recovery_link_sessions_withpre_angle_z.csv"))
    return r[r.set == st].rename(columns={col: "value"})[COLS]


def overcorrection(spout: str, col: str = "lfl") -> pd.DataFrame:
    """overcorrection_sessions.csv (cued licks) at one spout (lfl = like-for-like offset, same place)."""
    oc = pd.read_csv(il.OUT / il.tagged("overcorrection_sessions.csv"))
    return oc[oc.spout == spout].rename(columns={col: "value"})[COLS]


def hemo_resp(variant: str = "adopted", roi: str = "key_mean") -> pd.DataFrame:
    """ITI post-lick response / the animal's pre ADOPTED response (hemo_check), one variant x ROI (mean of ROIs when
    `roi` is absent)."""
    h = pd.read_csv(il.OUT / "hemo_check" / "variant_resp_sessions.csv")
    h = h[h.variant == variant]
    if roi in set(h.roi):
        h = h[h.roi == roi]
    else:
        h = h.groupby(["animal", "epoch", "label"]).resp_rel_adopted.mean().reset_index().assign(roi="mean")
    return h.rename(columns={"resp_rel_adopted": "value"})[COLS]


def lick_order(scope: str, st: str, col: str) -> pd.DataFrame:
    """lick_order_sessions<SRC_TAG>.csv (scripts/iti_lick_order.py; split basis = untagged) column `col` for scope (all
    spouts / far contra) x set."""
    import iti_lick_order as lo_
    lo = pd.read_csv(RIN / f"lick_order_sessions{lo_.SRC_TAG}.csv")
    if col not in lo.columns:
        return pd.DataFrame(columns=COLS)
    return lo[(lo.scope == scope) & (lo.set == st)].rename(columns={col: "value"})[COLS]


def recovery_measure(col: str = "weighted_signed") -> pd.DataFrame:
    """recovery_measure_sessions<aim tag>.csv (behaviour: cued lick angle offset from own pre landing, degrees; lick set
    = WFIELD_AIM, first lick by default)."""
    m = pd.read_csv(rm.sessions_csv())   # behaviour only: no decoder, no basis tag
    return m.rename(columns={col: "value"})[COLS]


def matched(scope: str = "all spouts", cls: str = "all", st: str = "iti", ref: str = "gauss") -> pd.DataFrame:
    """Like-for-like read-out per session (scripts/lick_matched_readout.py; PRIMARY offset measure from 10-27): resid =
    decoded minus the animal's pre licks that landed at the same place (ref gauss = Gaussian same-place reference, Priya 10-27
    late "ok gaussian" (confirmed); loclin = local-linear supplement; was the binned 'place'), mm, + = contralesional. Decoder
    source follows iti_lick_order.SOURCE (env WFIELD_LICK_DECODER)."""
    import iti_lick_order as lo
    tag = ("" if ref == "pooled" else f"_{ref}") + lo.SRC_TAG
    S = pd.read_csv(RIN / f"matched{tag}_sessions.csv")
    S = S[(S.scope == scope) & (S.cls == cls) & (S.set == st)]
    return S.rename(columns={"resid": "value"})[["animal", "epoch", "label", "value"]]


def measures() -> dict:
    return {"r_lat_iti": where_r("iti"), "r_lat_cued": where_r("trial"),
            "off_lat_iti": recovery_link("iti"), "off_lat_cued": recovery_link("trial"),
            "off_farcontra_cued": overcorrection("far_contra"),
            "resp_key": hemo_resp("adopted", "key_mean"),
            "aim_weighted": recovery_measure()}


def main() -> int:
    """Shared models (wfield_local.mixed_models; DECISIONS 2026-10-20 night: mixed models with time are the PRIMARY
    readout, epochs descriptive): time (y ~ post + post:lday), behaviour (y ~ post + post:deficit, deficit = weighted
    signed aim, pre - session, WFIELD_AIM lick set) and epoch (y ~ C(epoch)), random-effects ladder per animal; the paired nested bootstrap per epoch
    alongside for the page figures."""
    from wfield_local import mixed_models as mx
    M = measures()
    L = pd.concat([d.assign(measure=nm) for nm, d in M.items()], ignore_index=True).rename(columns={"value": "y"})
    L = mx.add_time_covariates(L.dropna(subset=["y"]))
    L = mx.add_deficit(L, rm.aim_deficit())
    # the aim measure is not fitted against the aim deficit (deficit = pre - aim: slope -1 by construction); fitted apart
    # so the behaviour model's FDR family holds only the real measures
    T = pd.concat([mx.lmm_table(L, ["measure"], models=("time",), fdr_by=()),
                   mx.lmm_table(L[L.measure != "aim_weighted"], ["measure"], models=("behaviour",), fdr_by=(),
                                print_animals=False),
                   mx.lmm_table(L, ["measure"], models=("epoch",), fdr_by=(), print_animals=False)], ignore_index=True)
    T.assign(aim_licks=rm.AIM).to_csv(il.OUT / il.tagged("mixed_models.csv", il.BASIS_TAG + rm.AIM_TAG), index=False)
    pd.set_option("display.width", 240)
    print(mx.summary_table(T, ["measure"]).to_string())
    rows = []
    for nm, df in M.items():
        prd = {a: h.value.dropna().to_numpy() for a, h in df[df.epoch == "pre"].groupby("animal")}
        for ep in EP:
            post = {a: h.value.dropna().to_numpy() for a, h in df[df.epoch == ep].groupby("animal") if a in prd}
            post = {a: v for a, v in post.items() if len(v)}
            ci = ak.boot_delta(post, prd, np.random.default_rng(0))
            am = (df[df.epoch == ep].groupby("animal").value.mean() - df[df.epoch == "pre"].groupby("animal").value.mean()
                  ).dropna()
            rows.append(dict(measure=nm, epoch=ep, boot_point=ci.point, boot_lo=ci.lo, boot_hi=ci.hi,
                             n_dir=int((am * np.sign(ci.point) > 0).sum()), n_animals=len(am)))
    pd.DataFrame(rows).assign(aim_licks=rm.AIM).to_csv(il.OUT / il.tagged("mixed_models_boot.csv", il.BASIS_TAG + rm.AIM_TAG),
                                                       index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
