"""Does the SLOW band of the adopted corrected signal carry decodable spout position? (Priya 2026-10-19.)

    python scripts/slowband_decode_check.py [--jobs 3] [labels ...]

Context: the two-band candidate (`hemo_variants` meegkit_2band) removes only the >= 300 s component that 470 shares with
415, and LOWERED pre-cue decoding in 26/32 pre sessions (-0.016, p=0.0004; post-cue unchanged). Position blocks last
~1-2 min (PS92 ~1.5x longer, tail to 10.5 min; `scripts/position_block_durations.py`), so genuine position coding should
mostly live faster than that. Here the ADOPTED SVTcorr (hemo_meegkit_hpfit) is split by a global cosine projection into
  slow300 / slow600   the component with period >= 300 / 600 s
  fast300             adopted minus slow300
and pre-cue / post-cue position are decoded from each with the 08-13 decoder (`filter_acausality_test.roi_signal` +
`decode`: Allen-ROI features, block CV). slow-only above chance = the slow band carries position (block-scheduling
structure, long blocks, or a slow position-correlated signal); fast300 ~ adopted = the headline code is fast.

Output -> <LP_analysis>/<run>/imaging_encoding/qc/slowband_decode.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = ("PS92_0806 PS92_0807 PS92_0809 PS92_0810 PS92_0811 PS92_0812 PS92_0813 PS92_0814 PS92_0818 PS92_0819 "
           "PS92_0820 PS92_0821 PS92_0822 PS92_0824 PS92_0826 PS92_0828 PS92_0901 PS93_0806 PS93_0807 PS93_0809 "
           "PS93_0810 PS93_0811 PS93_0812 PS93_0813 PS93_0814 PS94_0806 PS94_0807 PS94_0809 PS94_0810 PS94_0811 "
           "PS94_0812 PS94_0813 PS94_0814 PS94_0825 PS94_0831 PS94_0917 PS94_1001 PS95_0806 PS95_0807 PS95_0809 "
           "PS95_0810 PS95_0811 PS95_0812 PS95_0813 PS95_0814 PS95_1005").split()


def _slow(x: np.ndarray, fs: float, min_period_s: float) -> np.ndarray:
    t = np.arange(x.shape[1]) / fs
    T = max(t[-1], 1e-9)
    n = int(np.floor(2 * T / min_period_s))
    Q = np.linalg.qr(np.column_stack([np.cos(np.pi * k * t / T) for k in range(n + 1)]))[0]
    return (x @ Q) @ Q.T


def session_check(label: str) -> dict:
    """Module-level worker."""
    import glob

    from wfield_local import filter_acausality_test as F
    from wfield_local import hemo_variants as hv
    s = next(x for x in F.SESSIONS if x["label"] == label)
    res = s["mc"] + "/wfield_local_results"
    ad = glob.glob(res + "/allen_aligned_affine8v1")[0]
    c = np.load(res + f"/hemo_{hv.ADOPTED}/SVTcorr.npy").astype(np.float64)
    s300 = _slow(c, hv.FS, 300.0)
    arms = {"adopted": c, "slow300": s300, "fast300": c - s300, "slow600": _slow(c, hv.FS, 600.0)}
    out = {"label": label}
    for name, x in arms.items():
        sig, regs = F.roi_signal(ad, x.astype(np.float32))
        out[f"{name}_pre"] = F.decode(s, sig, regs, "precue")
        out[f"{name}_post"] = F.decode(s, sig, regs, "cue")
        del sig
    print("  {:10s} ".format(label) + "  ".join(f"{k} {v:.3f}" for k, v in out.items() if k != "label"), flush=True)
    return out


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--all", action="store_true",
                    help="every curated session from 08-06 (the epoch comparison, Priya 10-20); output "
                         "slowband_decode_all.csv, reusing the rows already in slowband_decode.csv")
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    from wfield_local import epochs
    from wfield_local import filter_acausality_test as F
    out = oo.root(PathResolver()) / a.run / "imaging_encoding" / "qc"
    writeguard.assert_writable(out)
    name = "slowband_decode_all.csv" if a.all else "slowband_decode.csv"
    labels = a.labels or ([s["label"] for s in F.SESSIONS if s["label"][-4:] >= "0806"] if a.all else list(DEFAULT))
    prev = pd.DataFrame()
    if a.all and (Path(out) / "slowband_decode.csv").exists():
        prev = pd.read_csv(Path(out) / "slowband_decode.csv")
        prev = prev[prev.label.isin(labels)].drop(columns=["animal", "set"], errors="ignore")
    todo = [x for x in labels if prev.empty or x not in set(prev.label)]
    got, fail = ak.fan_sessions(todo, session_check, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED {it}: {err}")
    D = pd.concat([prev, pd.DataFrame([r for _lab, r in got])], ignore_index=True).sort_values("label")
    D["animal"] = D.label.str[:4]
    D["set"] = ["pre" if x[5:] <= "0814" else "post" for x in D.label]
    D["epoch"] = D.label.map(epochs.epoch_of)
    D.to_csv(Path(out) / name, index=False, float_format="%.4g")
    if a.all:                                          # rule 8: per animal x epoch, adopted vs slow band removed
        cols = ["adopted_pre", "fast300_pre", "slow300_pre", "adopted_post", "fast300_post"]
        t = D.groupby(["animal", "epoch"])[cols].mean().unstack("epoch")
        print(t.reindex(columns=["pre", "acute", "subacute", "chronic"], level=1).round(3).to_string())
    pd.set_option("display.width", 200)
    cols = [c for c in D.columns if c.endswith(("_pre", "_post"))]
    print("chance 0.167; per set x animal:")
    print(D.groupby(["set", "animal"])[cols].mean().round(3).to_string())
    print(D.groupby("set")[cols].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
