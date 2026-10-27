"""MIXED-EFFECTS models for the headline ITI / lick read-out results (Priya 10-27: "mixed-effects model with sessions
nested in animals ... for any headline result" -> "yes").

Per measure, the SHARED models of wfield_local.mixed_models (time: y ~ post + post:lday; behaviour: y ~ post +
post:deficit, deficit = far-contra time to contact; epoch: y ~ C(epoch)), sessions as observations, random-effects
ladder per animal (10-27 late: was an epoch-only random-intercept model written here -- replaced, rule 9). Reported: each post epoch's coefficient (change vs pre), 95 % Wald CI and p, beside the
paired nested bootstrap used on the page (boot_delta) for the same session values. With 4 animals the random-effect
variance is poorly estimated, so the Wald p is anti-conservative relative to the bootstrap; both are shown, and the
per-animal direction count (rule 8).
Measures (session values, all on the frozen pre-stroke decoder with z-scored Allen-area features unless stated):
  r_lat_iti / r_lat_cued      lateral r / the animal's pre mean (zscore_check/where_decoder_z.csv)
  off_lat_iti / off_lat_cued  lateral offset beyond the pre pull, change vs pre (recovery_link_sessions_withpre_angle_z)
  off_farcontra_cued          far-contra cued offset beyond the pull (overcorrection_sessions.csv)
  resp_key                    ITI post-lick response, key ROIs, / the animal's pre (hemo_check, adopted variant)
  aim_weighted                behaviour: weighted signed angle offset from own pre landing (recovery_measure_sessions)
Output <E>/iti_licks/mixed_models.csv. Usage: python scripts/iti_mixed_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

EP = ["acute", "subacute", "chronic"]


COLS = ["animal", "epoch", "label", "value"]


# --- session-level readers, one per producing table (shared with scripts/iti_lmm_figures.py; one row per session) ---
def where_r(st: str, target: str = "lateral") -> pd.DataFrame:
    """Decoder r (Allen areas, z-scored features) for lick set `st` (iti / trial) / the animal's pre mean."""
    d = pd.read_csv(il.OUT / "zscore_check" / "where_decoder_z.csv")
    g = d[(d.source == "areas") & (d.target == target) & (d.set == st)].copy()
    g["value"] = g.r / g.animal.map(g[g.epoch == "pre"].groupby("animal").r.mean())
    return g[COLS]


def recovery_link(st: str, col: str = "bey_lat") -> pd.DataFrame:
    """recovery_link_sessions_withpre_angle_z.csv column `col` (bey_lat = lateral offset beyond the pull vs pre)."""
    r = pd.read_csv(il.OUT / "recovery_link_sessions_withpre_angle_z.csv")
    return r[r.set == st].rename(columns={col: "value"})[COLS]


def overcorrection(spout: str, col: str = "beyond") -> pd.DataFrame:
    """overcorrection_sessions.csv (cued licks) at one spout."""
    oc = pd.read_csv(il.OUT / "overcorrection_sessions.csv")
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
    """lick_order_sessions.csv (scripts/iti_lick_order.py) column `col` for scope (all spouts / far contra) x set."""
    lo = pd.read_csv(il.OUT / "lick_order_sessions.csv")
    if col not in lo.columns:
        return pd.DataFrame(columns=COLS)
    return lo[(lo.scope == scope) & (lo.set == st)].rename(columns={col: "value"})[COLS]


def recovery_measure(col: str = "weighted_signed") -> pd.DataFrame:
    """recovery_measure_sessions.csv (behaviour: cued lick angle offset from own pre landing, degrees)."""
    m = pd.read_csv(il.OUT / "recovery_measure_sessions.csv")
    return m.rename(columns={col: "value"})[COLS]


def matched(scope: str = "all spouts", cls: str = "all", st: str = "iti", ref: str = "place") -> pd.DataFrame:
    """Like-for-like read-out per session (scripts/lick_matched_readout.py; PRIMARY offset measure from 10-27): resid =
    decoded minus the animal's pre licks that landed at the same place (ref place), mm, + = contralesional. Decoder
    source follows iti_lick_order.SOURCE (env WFIELD_LICK_DECODER)."""
    import iti_lick_order as lo
    tag = ("" if ref == "pooled" else f"_{ref}") + lo.SRC_TAG
    S = pd.read_csv(il.OUT / f"matched{tag}_sessions.csv")
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
    readout, epochs descriptive): time (y ~ post + post:lday), behaviour (y ~ post + post:deficit, deficit = far-contra
    time to contact) and epoch (y ~ C(epoch)), random-effects ladder per animal; the paired nested bootstrap per epoch
    alongside for the page figures."""
    from wfield_local import mixed_models as mx
    M = measures()
    L = pd.concat([d.assign(measure=nm) for nm, d in M.items()], ignore_index=True).rename(columns={"value": "y"})
    L = mx.add_time_covariates(L.dropna(subset=["y"]))
    L = mx.add_deficit(L, mx.deficit(il.OUT.parent / "qc"))
    T = mx.lmm_table(L, ["measure"], models=("time", "behaviour", "epoch"), fdr_by=())
    T.to_csv(il.OUT / "mixed_models.csv", index=False)
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
    pd.DataFrame(rows).to_csv(il.OUT / "mixed_models_boot.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
