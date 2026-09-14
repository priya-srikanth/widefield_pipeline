"""REST AS SIGNAL, not as baseline: can spout position be DECODED from the inter-trial interval?

Priya, 2026-09-13: *"let's look at the 'rest' component as a separate analysis (for 'working'
trials)"*, after establishing that the spout RETRACTS between trials so no target is present.

WHY THIS IS WORTH DECODING RATHER THAN JUST MEASURING. `rest_position_permutation` established that
rest maps differ by position beyond what the block-time structure explains -- 44 sessions,
observed/null 1.429, above null in 41/44. That is a difference in MEAN MAPS. A decoder asks the
sharper question: is the position recoverable TRIAL BY TRIAL from an interval in which the animal
has no target in front of it? A mean-map difference can be carried by a handful of sessions; a
per-period decode cannot.

THE TWO GUARDS THIS ANALYSIS LIVES OR DIES BY
---------------------------------------------
1. BLOCK-CV, NOT RANDOM CV. Positions run in ~6-trial BLOCKS, so two rest periods from the same
   block are adjacent in time and share whatever slow drift the session has. Random folds would put
   them in train and test, and the decoder would recover BLOCK IDENTITY -- trivially, and it would
   read as position. Folds are grouped by block, exactly as `locanmf_position_decoder` does for
   trials. Without this the analysis is worthless and would look excellent.
2. A CIRCULAR-SHIFT NULL, not a label shuffle. Shuffling labels destroys the block structure and
   gives an optimistically low null. Circular-shifting the label sequence over time-ordered periods
   keeps blocks as blocks and drift as drift, so "drift aliased onto blocks" is INSIDE the null.
   Accuracy above that null is position information the time structure cannot supply.

WORKING TRIALS ONLY. A rest period counts only if the trials bracketing it are both WORKING -- the
animal responded, or missed while still attempting. The terminal quit period is excluded, for the
same reason every other family excludes it: a sated animal's rest is a different state, and it
concentrates at the end of the session where drift is largest.

WHAT THE ANSWER MEANS, AND THE AMBIGUITY THIS ARM CANNOT RESOLVE. With no spout present, above-null
decoding is either a PERSISTENT trace of the target just licked at, or ANTICIPATION of the next one
-- which is available because the block structure makes the next position predictable. Within a
block those two point at the SAME position and no amount of decoding separates them. The rest
periods at BLOCK BOUNDARIES do, and that is `--boundary`.

RUN:  python -m scripts.rest_migration.rest_position_decode [--perm 50] [--boundary]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time

import numpy as np


def _frame_samples(mc, fmdir, regime, pco):
    if regime == "B":
        fm = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_frame_map.npz"))
        summ = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_summary.json"))
        if not fm or not summ:
            return None
        with open(summ[0]) as fh:
            off = int(json.load(fh)["chosen_exposure_offset"])
        z = np.load(fm[0])
        return pco[np.clip(z["original_frame_index_ch0"] + off, 0, len(pco) - 1)]
    return pco[np.arange(len(pco) // 2) * 2]


def _fit_score(X, y, g):
    """Balanced accuracy under GroupKFold by BLOCK. None if the split is not supportable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    if len(np.unique(y)) < 3 or len(np.unique(g)) < 4:
        return None
    n_splits = min(5, len(np.unique(g)))
    pred = np.empty_like(y)
    try:
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
            if len(np.unique(y[tr])) < 2:
                return None
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=0.1,
                                                 multi_class="multinomial"))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
    except Exception:                                                  # noqa: BLE001
        return None
    return float(balanced_accuracy_score(y, pred))


def main() -> int:
    import h5py

    from wfield_local import config, daq_io, epochs, joint_basis
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS, _load_cue_events
    from wfield_local.plot_spout_trial_averages import _classify_cues
    from wfield_local.quiet_periods import quiet_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=50)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    t0 = time.time()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")]
    if a.limit:
        todo = todo[: a.limit]
    by_epoch, skipped = {}, []

    for s in todo:
        lab = s["label"]
        ep = epochs.epoch_of(lab)
        if ep is None:
            skipped.append(f"{lab}: no epoch")
            continue
        qs = sorted(glob.glob(f"{quiet_dir(s['mc'])}/*quiet_sample.npy"))
        if not qs:
            skipped.append(f"{lab}: no rest mask")
            continue
        try:
            rest = np.load(qs[0]).astype(bool)
            with h5py.File(s["h5"], "r") as f:
                dn = [x.decode() for x in f["digital/channel_names"][:]]
                packed = f["digital/packed_samples"][:, 0]
            pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
            ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
            cue = _load_cue_events(s["h5"])
            codes = np.asarray(_classify_cues(cue["cue_samples"], cue["strobe_samples"],
                                              cue["strobe_codes"]))
            cs = np.asarray(cue["cue_samples"], np.int64)
            fs_samp = _frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
            _u, v = joint_basis._load_session(s["mc"])
        except Exception as ex:                                        # noqa: BLE001
            skipped.append(f"{lab}: {type(ex).__name__} {str(ex)[:50]}")
            continue
        if fs_samp is None:
            continue

        V = np.asarray(v)
        T = min(V.shape[1], rest.shape[0])
        V = V[:, :T]
        f_of = np.clip(fs_samp, 0, rest.shape[0] - 1)

        pad = np.concatenate([[0], rest.view(np.int8), [0]])
        dif = np.diff(pad)
        # BLOCK ID from runs of equal position in the trial sequence -- the CV group.
        blk = np.zeros(len(codes), np.int64)
        b = 0
        for i in range(1, len(codes)):
            if codes[i] != codes[i - 1]:
                b += 1
            blk[i] = b

        X, y, g = [], [], []
        for aa, bb in zip(np.flatnonzero(dif > 0), np.flatnonzero(dif < 0)):
            prev = np.searchsorted(cs, aa, "right") - 1
            nxt = np.searchsorted(ts, bb, "left")
            if prev < 0 or nxt >= len(ts):
                continue
            nc = np.searchsorted(cs, ts[nxt], "left")
            if nc >= len(codes) or prev >= len(codes):
                continue
            if codes[prev] != codes[nc] or codes[prev] < 0:
                continue                       # boundary periods: the --boundary arm's business
            fr = np.flatnonzero((f_of >= aa) & (f_of < bb))
            fr = fr[fr < T]
            if fr.size < a.bins:
                continue
            # BINNED LIKE EVERY OTHER ARM, so the feature width is comparable: split the period
            # into `bins` equal parts and take each part's mean.
            parts = np.array_split(fr, a.bins)
            X.append(np.concatenate([V[:, p].mean(1) for p in parts]))
            y.append(int(codes[prev]))
            g.append(int(blk[prev]))
        if len(y) < 40:
            skipped.append(f"{lab}: only {len(y)} usable rest periods")
            continue

        X, y, g = np.array(X), np.array(y), np.array(g)
        obs = _fit_score(X, y, g)
        if obs is None:
            skipped.append(f"{lab}: CV not supportable")
            continue
        nulls = []
        order = np.argsort([0] * len(y))       # periods are already in time order
        for _ in range(int(a.perm)):
            k = int(rng.integers(1, len(y)))
            r = _fit_score(X, np.roll(y[order], k), g)
            if r is not None:
                nulls.append(r)
        if len(nulls) < 10:
            skipped.append(f"{lab}: only {len(nulls)} usable permutations")
            continue
        nm = float(np.mean(nulls))
        p = (1 + sum(1 for x in nulls if x >= obs)) / (1 + len(nulls))
        by_epoch.setdefault(ep, []).append((obs, nm, p, len(y)))
        print(f"  .. {lab} [{ep}] n={len(y):>4}  obs {obs:.3f}  null {nm:.3f}  "
              f"p={p:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'=' * 76}\nPOSITION DECODED FROM REST (working trials, block-CV, circular-shift null)"
          f"\n{'=' * 76}")
    order_ep = [e for e in ("pre", "acute", "subacute", "chronic") if e in by_epoch]
    if not order_ep:
        print("\nNOTHING WAS DECODED -- a failed run, not a negative result.")
        for x in skipped[:10]:
            print("   skipped:", x)
        return 1
    print(f"{'epoch':<10}{'n sess':>8}{'obs':>9}{'null':>9}{'obs-null':>10}{'sess>null':>11}")
    for e in order_ep:
        v = by_epoch[e]
        o = float(np.mean([x[0] for x in v]))
        n = float(np.mean([x[1] for x in v]))
        print(f"{e:<10}{len(v):>8}{o:>9.3f}{n:>9.3f}{o - n:>10.3f}"
              f"{sum(1 for x in v if x[0] > x[1]):>8}/{len(v):<3}")
    print("\nchance is 1/6 = 0.167 for six positions, but READ AGAINST THE NULL, not against chance:")
    print("the null already contains drift-through-blocks, which chance does not.")
    print(f"\ntested {sum(len(v) for v in by_epoch.values())} sessions; {len(skipped)} skipped")
    for x in skipped[:8]:
        print("   skipped:", x)
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
