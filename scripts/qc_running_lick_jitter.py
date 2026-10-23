"""Check 1 (Priya 10-23): are low treadmill speeds lick / cue jitter? Per session, treadmill |speed| on a 31.25 Hz grid
over the engaged-trial span; per speed band, the fraction of samples within +/-0.5 s of a cue or a lick onset (and
the same for random times = chance at that session's lick density)."""
import sys

import numpy as np

BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.0), (3.0, 1e9)]
WIN = 0.5


def job(label):
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    B = ce.load_behaviour(an, date, PathResolver())
    tr, vr = (np.asarray(x, float) for x in B.analog["running"])
    T = B.trials
    cue = T.cue_s.to_numpy(float)
    cue = cue[np.isfinite(cue)]
    lk = np.asarray(B.lick_on_s, float)
    ex = getattr(B, "extra_lick_on_s", None)
    if ex is not None:
        lk = np.concatenate([lk, np.asarray(ex, float)])
    ev = np.sort(np.concatenate([cue, lk]))
    g = np.arange(cue.min() - 5, cue.max() + 10, 1 / 31.25)
    s = np.abs(np.interp(g, tr, vr))
    j = np.clip(np.searchsorted(ev, g), 1, len(ev) - 1)
    near = np.minimum(np.abs(g - ev[j - 1]), np.abs(g - ev[j])) <= WIN
    out = {"label": label, "animal": an, "epoch": epochs.epoch_of(label), "n": len(g), "near_all": float(near.mean())}
    for lo, hi in BANDS:
        m = (s >= lo) & (s < hi)
        out[f"n_{lo:g}"] = int(m.sum())
        out[f"near_{lo:g}"] = float(near[m].mean()) if m.any() else np.nan
    return out


def main():
    import pandas as pd

    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    labels = [x for x in sorted(ce.sessions_available()) if x[-4:] >= "0801"]
    got, fail = ak.fan_sessions(labels, job, jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 4, label="jitter")
    D = pd.DataFrame([r for _l, r in got])
    D.to_csv(sys.argv[2] if len(sys.argv) > 2 else "lick_jitter.csv", index=False)
    pd.set_option("display.width", 220)
    print("fraction of samples within +/-0.5 s of a cue or lick, by speed band (mm/s); 'all' = any speed (chance)")
    for an, g in D.groupby("animal"):
        for ep in ("pre", "acute", "subacute", "chronic"):
            x = g[g.epoch == ep]
            if not len(x):
                continue
            w = {lo: (x[f"near_{lo:g}"] * x[f"n_{lo:g}"]).sum() / max(x[f"n_{lo:g}"].sum(), 1) for lo, _ in BANDS}
            frac = {lo: x[f"n_{lo:g}"].sum() / x.n.sum() for lo, _ in BANDS}
            print(f"{an} {ep:9s} all {np.average(x.near_all, weights=x.n):.2f} | " + " | ".join(
                f"{lo:g}-{hi if hi < 1e8 else 'inf'}: near {w[lo]:.2f} ({frac[lo]:.1%} of time)" for lo, hi in BANDS))
    for it, e in fail:
        print("FAIL", it, e)


if __name__ == "__main__":
    main()
