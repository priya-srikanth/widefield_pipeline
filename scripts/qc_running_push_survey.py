"""Backward-push survey per animal x epoch (trial frames): how often (fraction of frames with backward speed > PUSH_FLOOR)
and how hard, vs the pre-stroke push cap. Run with FROZEN_PUSH=1 + the encoder env (OROFACIAL_GATE, COMBINED_ENCODING_LICKS).
Usage: python scripts/qc_running_push_survey.py [jobs]"""
import sys
import numpy as np


def main():
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import frozen_encoder as fe
    labels = [x for x in sorted(ce.sessions_available()) if x[-4:] >= "0806" and x in set(ak.curated_labels())]
    got, fail = ak.fan_sessions(labels, fe.template_job, jobs=int(sys.argv[1]), label="push")
    by = {}
    for lab, t in got:
        by.setdefault((t["animal"], epochs.epoch_of(lab)), []).append((np.asarray(t["push_abs"], float), len(t["run_abs"])))
    print("animal epoch n | fraction of trial frames with backward speed > 0.3 mm/s | fraction beyond the pre cap | magnitude of push frames")
    for an in sorted({a for a, _ in by}):
        pre = np.concatenate([b for b, _ in by[(an, "pre")]])
        cap = np.percentile(pre, 99)
        for ep in ("pre", "acute", "subacute", "chronic"):
            if (an, ep) not in by:
                continue
            r = np.concatenate([b for b, _ in by[(an, ep)]]); n = sum(m for _, m in by[(an, ep)])
            print(f"{an} {ep:9s} {len(by[(an, ep)]):2d} | push frames {len(r) / n:.3f} | beyond cap {np.sum(r > cap) / n:.4f} | "
                  f"mean {r.mean():.2f} p99 {np.percentile(r, 99):.2f} mm/s  (pre cap {cap:.2f})", flush=True)
    print("failed:", [f[0] for f in fail])


if __name__ == "__main__":
    main()
