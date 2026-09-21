"""Pre-cue window sweep ON THE PIPELINE WE ACTUALLY RUN: meegkit_hpfit + LocaNMF, curated set.

WHY THIS HAS TO BE RE-RUN. Three preprocessing variants appear across the existing evidence and NONE
of them is the one in production:

    withdrawn sweep (2026-08-11)   zerophase (acausal)   ROI features   -- "measuring the shadow"
    re-run sweep    (2026-08-13)   strobedetrend         ROI, 16 sess   -- reversed both answers
    EVERYTHING WE RUN NOW          meegkit_hpfit         LocaNMF        -- never swept

The 08-13 re-run found the asymmetry was an artefact (last1s-first1s went +0.245 -> -0.044),
concluded position information is spread EVENLY across the pre-cue window, and kept
`precue_post_s: 2.0` because shortening stopped helping (mean 1.0 s: -0.023, better in 3/16). Its own
caveats say ROI not LocaNMF, 16 sessions not 36, and a variant we did not adopt -- "confirm all three
before changing any default."

So the claim "the last second carries the most information" and its reversal have BOTH been measured
on pipelines we no longer use. Priya, 2026-09-13, on whether the maps should shorten to 1 s: this is
the test that decides it, and it must run on meegkit_hpfit + LocaNMF.

THE THREE ARMS ARE THE 08-13 ARMS, so the numbers are directly comparable:

    mean1.0     post_s 1.0, ONE bin   -- the last second only, averaged
    mean2.0     post_s 2.0, ONE bin   -- the whole window, averaged  <- what the MAPS do
    roll4x0.25  post_s 1.0, FOUR bins -- the last second, binned
    roll4x0.5   post_s 2.0, FOUR bins -- the time course            <- what the DECODER does

WHICH WINDOW OF THE ENL, and why NOT the early part. The ENL runs 2-3 s and there is visible activity
peaking 2-3 s before the cue -- i.e. at its START. That is tempting and it is the wrong end to take.
The ENL BEGINS AT THE LAST LICK VIOLATION, so its earliest moments sit immediately after licking, and
a calcium/haemodynamic tail outlives the lick that caused it. `precue_lickfree` slides the window off
any window CONTAINING a lick onset, but it cannot slide off a tail. So the early ENL is the most
lick-contaminated second of the trial, not the cleanest, and the 2 s ENDING at the cue stays the
safest choice (Priya, 2026-09-13).

THIS ALSO REFRAMES THE 08-13 RESULT. That sweep found "if anything the FIRST second is marginally
better" and read it as a maintained code spread evenly. The alternative reading is that the early
window carries LICK BLEED-THROUGH rather than more position code -- which would look identical in a
decoding score. Neither sweep separated those, so "spread evenly" should be held as one of two
explanations rather than as the finding.

AND THAT CONTRAST IS THE POINT (Priya: "the pre-cue decoder is ok to be 2s because it's binned. the
maps are not"). If roll4x0.5 > mean2.0 the decoder's binning is earning its keep; if mean1.0 >
mean2.0 the maps are diluting and should shorten; if mean2.0 >= mean1.0 the information is spread and
the maps should stay. Those are three different questions and one sweep answers all of them.

Block-CV (GroupKFold over ~6-trial position blocks), chance 0.167, paired per session so the
"better in N/M" column means the same thing it did in the 08-13 table.
"""
from __future__ import annotations

import sys
import time

import numpy as np

#: (name, post_s, n_bins). `roll4x0.25` is the cell that separates WINDOW LENGTH from BINNING
#: (Priya, 2026-09-13): if it matches roll4x0.5 the second second adds nothing and the maps can
#: shorten; if binning helps at 1 s too, the gain is temporal structure rather than duration.
ARMS = (("mean1.0", 1.0, 1), ("mean2.0", 2.0, 1),
        ("roll4x0.25", 1.0, 4), ("roll4x0.5", 2.0, 4))


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wfield_local import config
    from wfield_local import locanmf_position_decoder as lpd
    from wfield_local.block_ids import block_ids, block_size_max_for
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.locanmf_frozen_decoder import _args

    t0 = time.time()
    print(f"hemo.variant = {config.defaults()['hemo']['variant']!r}   "
          f"locanmf = {config.defaults()['locanmf']['output_dir_name']!r}", flush=True)
    want = set(config.phase_labels("pre"))
    todo = [s for s in SESSIONS if s["label"] in want]
    print(f"{len(todo)} pre-stroke sessions\n", flush=True)

    acc = {}
    for i, s in enumerate(todo, 1):
        for name, post_s, nbins in ARMS:
            # BINS ARE SET ON THE ARGS, which `_bins_for` honours before falling back to
            # `decode.bins[align]` -- so one arm gets a single mean and another a time course
            # without touching the config every other analysis reads, and without patching a
            # module function out from under a cached call.
            a = _args("locanmf", "precue", post_s)
            a.bins = nbins
            try:
                X, y, g, *_ = lpd.trial_features_cached(s, a)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! {s['label']} {name}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
                continue
            X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
            if len(y) < 40 or len(set(y.tolist())) < 6:
                continue
            blk = block_ids(g, block_size_max_for(s)) if g.ndim == 1 else g
            k = min(5, len(set(np.asarray(blk).tolist())))
            if k < 2:
                continue
            hits, n = 0, 0
            for tr, te in GroupKFold(n_splits=k).split(X, y, groups=blk):
                if len(set(y[tr].tolist())) < 2:
                    continue
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=0.5, max_iter=3000)).fit(X[tr], y[tr])
                hits += int((m.predict(X[te]) == y[te]).sum())
                n += len(te)
            if n:
                acc.setdefault(s["label"], {})[name] = hits / n
        print(f"  [{i}/{len(todo)}] {s['label']}: "
              + "  ".join(f"{k}={v:.3f}" for k, v in acc.get(s['label'], {}).items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    full = {lab: d for lab, d in acc.items() if len(d) == len(ARMS)}
    print(f"\n{len(full)} sessions with all three arms  (chance 0.167)")
    print(f"{'arm':<12}{'mean acc':>10}{'vs mean2.0':>12}{'better':>10}")
    base = "mean2.0"
    for name, _p, _b in ARMS:
        v = np.array([d[name] for d in full.values()])
        b = np.array([d[base] for d in full.values()])
        if name == base:
            print(f"{name:<12}{v.mean():>10.3f}{'--':>12}{'--':>10}")
        else:
            print(f"{name:<12}{v.mean():>10.3f}{v.mean() - b.mean():>+12.3f}"
                  f"{f'{int((v > b).sum())}/{len(v)}':>10}")
    if full:
        d1 = np.array([d["mean1.0"] for d in full.values()])
        d2 = np.array([d["mean2.0"] for d in full.values()])
        print(f"\nlast1s - whole2s : {d1.mean() - d2.mean():+.3f}   "
              f"(08-13 on strobedetrend/ROI: -0.023, better in 3/16)")
        print("  > 0  the last second carries more -- the MAPS are diluting and should shorten")
        print("  <=0  information is spread -- the maps should stay at 2 s")
    print(f"\n[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
