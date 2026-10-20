"""How far does cam4's FRAME-INDEX time (``t_ms``, what v36 analyses use) sit from its CAMERA-TIMESTAMP time (``t_daq_ms``,
from the sync pulses) inside trial windows? (Priya 10-20, after PS92 0908: dropped frames slide every later frame's
index time.)

    python -m scripts.cam4_timing_check [--jobs 4]

Per session (every session with cam4 production output, `o2/lp_r4_prod_20261015`): over frames from the cue to 3 s after
(the v36 response window) -- the error at the cue frame (template placement) and the error at each frame (= cue error +
drops since the cue), as median / p95 / max |t_daq_ms - t_ms|, the share of trials whose error reaches 4 / 20 ms by 1 s
after the cue (a lick-latency-scale error), and the dropped frames inside windows. Writes
`session_poses/cam4_timing_check.csv` and prints sessions over 4 ms at p95.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE = "lp_r4_prod_20261015"


def one(s: str) -> dict:
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    a, d = s.split(":")
    c = oi.session_windows(rv, BUNDLE, a, d, "cam4")
    if c is None:
        return {"session": f"{a}_{d}", "error": "no cam4 output"}
    ix = c[0]
    w = ix[(ix.t_ms >= 0) & (ix.t_ms <= 3000)]
    e = (w.t_daq_ms - w.t_ms).to_numpy(float)
    cue = w[w.t_ms == 0].set_index("trial_id")
    e0 = (cue.t_daq_ms - cue.t_ms)
    one_s = w[w.t_ms == 1000].set_index("trial_id")
    e1 = (one_s.t_daq_ms - one_s.t_ms).abs()
    step = np.diff(w.t_daq_ms.to_numpy(float))[np.diff(w.t_ms.to_numpy(float)) == 4.0]
    return {"session": f"{a}_{d}", "trials": int(w.trial_id.nunique()),
            "cue_err_med_ms": float(np.median(e0)), "cue_err_absmax_ms": float(np.abs(e0).max()),
            "err_med_ms": float(np.median(e)), "err_abs_p95_ms": float(np.percentile(np.abs(e), 95)),
            "err_abs_max_ms": float(np.abs(e).max()),
            "trials_err_1s_ge4ms": float((e1 >= 4).mean()), "trials_err_1s_ge20ms": float((e1 >= 20).mean()),
            "drops_in_windows": int((step > 6).sum())}


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    rv = PathResolver()
    sess = sorted(p.name.replace("_", ":", 1) for p in (Path(rv.root("dlc")) / "o2" / BUNDLE / "out").glob("PS*_2026*")
                  if any(p.glob("*_done.json")))
    res, fails = ak.fan_sessions(sess, one, jobs=a.jobs)
    T = pd.DataFrame([v for _, v in res] + [{"session": i.replace(":", "_"), "error": e} for i, e in fails])
    T = T.sort_values("session")
    out = Path(rv.root("dlc")) / "session_poses" / "cam4_timing_check.csv"
    T.to_csv(out, index=False, float_format="%.3g")
    pd.set_option("display.width", 220)
    ok = T[T.get("error", pd.Series(np.nan, index=T.index)).isna()]
    print(ok.describe().round(2).to_string())
    print("\nsessions with p95 |error| > 4 ms:")
    print(ok[ok.err_abs_p95_ms > 4].round(2).to_string(index=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
