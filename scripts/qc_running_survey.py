"""Running-speed survey per animal x epoch (trial frames, lag-0 |speed|) for choosing the frozen-encoder running cap."""
import sys

import numpy as np


def main():
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import frozen_encoder as fe
    labels = [x for x in sorted(ce.sessions_available()) if x[-4:] >= "0801"]
    got, fail = ak.fan_sessions(labels, fe.template_job, jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 6,
                                label="survey")
    by = {}
    for lab, t in got:
        by.setdefault((t["animal"], epochs.epoch_of(lab)), []).append(t["run_abs"])
    print("animal epoch n_sess | p50 p90 p99 p99.9 max | frac>3 | p99 of frames>3 | frac>pre_p99 frac>pre_p999")
    for an in sorted({a for a, _ in by}):
        pre = np.concatenate(by[(an, "pre")])
        q99, q999 = np.percentile(pre, 99), np.percentile(pre, 99.9)
        for ep in ("pre", "acute", "subacute", "chronic"):
            if (an, ep) not in by:
                continue
            r = np.concatenate(by[(an, ep)])
            mov = r[r > 3]
            print(f"{an} {ep:9s} {len(by[(an, ep)]):2d} | " + " ".join(f"{np.percentile(r, q):6.2f}" for q in (50, 90, 99, 99.9))
                  + f" {r.max():6.2f} | {np.mean(r > 3):.3f} | {np.percentile(mov, 99) if len(mov) else np.nan:6.2f} | "
                  f"{np.mean(r > q99):.3f} {np.mean(r > q999):.3f}", flush=True)
    for it, e in fail:
        print("FAIL", it, e)


if __name__ == "__main__":
    main()
