"""Is the larger post-stroke ITI-lick dF/F response a smaller BASELINE (F0)? (handoff 00f to-do 2; pages 3.23-3.25)

dF/F divides by the session's mean image, so if raw fluorescence F0 fell after the stroke (bleaching, expression,
window clarity, LED power) the same dF would read as a larger dF/F. Per session: F0 = mean of `frames_average.npy`
(the pipeline's per-pixel temporal mean, raw counts) over the central 60 % of the frame, for the 470 nm (functional,
`hemo_variants.functional_channel`) and the 415 nm channel. Then per animal: F0 / pre mean by epoch, and the ITI-lick
post-lick response (`ramp_auc_sessions.csv`, orofacial ROIs averaged) in dF/F and in dF-equivalent units (dF/F x F0 /
pre F0). If the response increase vanishes in dF, it was an F0 artefact. Within animal across post sessions: r(F0,
response). Output <E>/iti_licks/f0_check.csv. Usage: python scripts/iti_f0_check.py
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


def f0(label: str) -> tuple[float, float]:
    from wfield_local import config
    from wfield_local.hemo_variants import functional_channel
    s = next(x for x in config.load_sessions() if x["label"] == label)
    a = np.load(Path(s["mc"]) / "wfield_local_results" / "frames_average.npy")
    h, w = a.shape[-2:]
    c = a[:, int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    fc = functional_channel(label)
    return float(np.nanmean(c[fc])), float(np.nanmean(c[1 - fc]))


def main() -> int:
    R = pd.read_csv(il.OUT / "ramp_auc_sessions.csv")
    resp = R.groupby(["label", "animal", "epoch"]).resp_auc.mean().reset_index()
    rows = []
    for x in resp.label:
        try:
            b, v = f0(x)
        except (OSError, StopIteration, ValueError) as e:
            print(f"{x}: {e}")
            b = v = np.nan
        rows.append(dict(label=x, f0_470=b, f0_415=v))
    S = resp.merge(pd.DataFrame(rows), on="label")
    pre = S[S.epoch == "pre"].groupby("animal")[["f0_470", "f0_415", "resp_auc"]].mean()
    for c in ("f0_470", "f0_415", "resp_auc"):
        S[f"{c}_rel"] = S[c] / S.animal.map(pre[c])
    S["resp_dF_rel"] = S.resp_auc_rel * S.f0_470_rel            # response in dF units, / pre
    S.to_csv(il.OUT / "f0_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("== per animal (rule 8): F0 and response / pre mean")
    print(S.groupby(["animal", "epoch"])[["f0_470_rel", "f0_415_rel", "resp_auc_rel", "resp_dF_rel"]].mean()
          .unstack("epoch").reindex(columns=["pre", *EP], level=1).round(2).to_string())
    tests = []
    for c in ("f0_470_rel", "resp_auc_rel", "resp_dF_rel"):
        prd = {a: g[c].to_numpy() for a, g in S[S.epoch == "pre"].groupby("animal")}
        for ep in EP:
            post = {a: g[c].to_numpy() for a, g in S[S.epoch == ep].groupby("animal")}
            ci = ak.boot_delta(post, prd, np.random.default_rng(0))
            tests.append(dict(measure=c, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi))
    print("\n== change vs pre (paired nested bootstrap)\n" + pd.DataFrame(tests).round(3).to_string(index=False))
    post = S[S.epoch != "pre"]
    print("\n== within animal over post sessions: r(F0_470, dF/F response)")
    print(post.groupby("animal").apply(lambda g: np.corrcoef(g.f0_470, g.resp_auc)[0, 1], include_groups=False)
          .round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
