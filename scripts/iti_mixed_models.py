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


def measures() -> dict:
    O = il.OUT
    out = {}
    d = pd.read_csv(O / "zscore_check" / "where_decoder_z.csv")
    d = d[(d.source == "areas") & (d.target == "lateral")]
    for st, nm in (("iti", "r_lat_iti"), ("trial", "r_lat_cued")):
        g = d[d.set == st].copy()
        g["value"] = g.r / g.animal.map(g[g.epoch == "pre"].groupby("animal").r.mean())
        out[nm] = g[["animal", "epoch", "label", "value"]]
    r = pd.read_csv(O / "recovery_link_sessions_withpre_angle_z.csv")
    for st, nm in (("iti", "off_lat_iti"), ("trial", "off_lat_cued")):
        out[nm] = r[r.set == st].rename(columns={"bey_lat": "value"})[["animal", "epoch", "label", "value"]]
    oc = pd.read_csv(O / "overcorrection_sessions.csv")
    out["off_farcontra_cued"] = oc[oc.spout == "far_contra"].rename(columns={"beyond": "value"})[
        ["animal", "epoch", "label", "value"]]
    h = pd.read_csv(O / "hemo_check" / "variant_resp_sessions.csv")
    h = h[(h.variant == "adopted")]
    if "key_mean" in set(h.roi):
        h = h[h.roi == "key_mean"]
    else:
        h = h.groupby(["animal", "epoch", "label"]).resp_rel_adopted.mean().reset_index().assign(roi="mean")
    out["resp_key"] = h.rename(columns={"resp_rel_adopted": "value"})[["animal", "epoch", "label", "value"]]
    m = pd.read_csv(O / "recovery_measure_sessions.csv")
    out["aim_weighted"] = m.rename(columns={"weighted_signed": "value"})[["animal", "epoch", "label", "value"]]
    return out


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
