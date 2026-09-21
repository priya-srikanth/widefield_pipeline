"""Does the frozen STATE decoder read BEHAVIOUR, or does it read WHEN IN THE SESSION?

THE QUESTION, and why a count could not answer it. `state_time_bins` measured the confound's
INPUT: the three classes are not distributed alike over session time, and post-stroke the imbalance
is much steeper than pre (licking falls 60-64% from the first fifth of a session to the last, while
rest and running rise). Cortex drifts over the same axis -- rest early vs late differs at RMS
0.00282 with no behavioural difference at all. Those two facts together mean the decoder COULD be
separating classes on time. They do not show that it DOES.

THE TEST. Score the frozen decoder SEPARATELY WITHIN EACH FIFTH of the session, on exactly the
segments that fifth contains. The model is unchanged and the labels are unchanged; only the subset
moves.

    FLAT ACROSS BINS          time is not carrying the decoder. The class-composition drift is a
                              property of the data, not of what the model reads.
    SYSTEMATIC ACROSS BINS    accuracy tracks WHEN, and the state decoder's preservation claim has
                              to be re-read with that in mind.

WHY THIS IS THE RIGHT CONTROL RATHER THAN BALANCING. Per-session balancing retains 0.25-0.35 of the
segments and empties at least one class-bin in 30 of 91 sessions (`state_time_bins`), so it would
pay most of the dataset to remove a confound of unmeasured size. This measures the size first. It
also costs nothing extra: the expensive step is projecting every session onto the joint basis, and
that happens once here for both the overall numbers and the per-bin ones.

WHAT IT CANNOT DO. A flat profile does not prove the decoder ignores time -- within a fifth of a
session there is still a time axis, and a decoder reading slow drift would read it there too. What a
flat profile rules out is the version of the confound that matters: that post-stroke accuracy is
held up by the classes having moved to more separable parts of the session.

THE FROZEN DISCIPLINE IS `by_animal_day`'s, deliberately duplicated rather than imported: pre-stroke
days are leave-one-SESSION-out, post-stroke days are scored by a model fitted on ALL pre-stroke
segments. A pre column scored partly against itself would be too easy and the per-bin profile would
inherit that.

REST DEFINITION: whatever `behavior_events` holds on disk, which is schema v3 (trial-anchored REST)
since 2026-09-12. `by_animal_day` is `lru_cache`d IN-PROCESS ONLY, so a fresh run always re-reads
the events npz -- there is no disk cache here that could serve the retired definition.

RUN:  python -m scripts.rest_migration.archive.state_decoder_by_time [--bins 5]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

NBINS = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=NBINS)
    a = ap.parse_args()
    nb = int(a.bins)

    from wfield_local import behavior_events as be
    from wfield_local import config, joint_locanmf
    from wfield_local import locomotor_decoder as ld
    from wfield_local import locomotor_features as lf
    from wfield_local import locomotor_state as ls
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.paths import PathResolver
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events

    rv = PathResolver()
    t0 = time.time()
    fs_img = ld.FS_IMG
    got_all, skipped = {}, []

    for an in ANIMALS:
        try:
            basis = joint_locanmf.load(an, sessions=SESSIONS)
        except Exception as ex:                                        # noqa: BLE001
            skipped.append(f"{an}: basis {type(ex).__name__} {str(ex)[:60]}")
            continue
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        got = {}
        for s in [x for x in SESSIONS if x["label"] in want]:
            lab = s["label"]
            if lab in ls.EXCLUDE_SESSIONS:
                skipped.append(f"{lab}: EXCLUDE_SESSIONS")
                continue
            mmdd = lab.split("_")[-1]
            ev = be.get_or_compute(rv, an, "2026" + mmdd)
            if not ev:
                skipped.append(f"{lab}: no events")
                continue
            try:
                sig = np.asarray(joint_locanmf.BasisSource(basis, s).signal()[0])
                cues = np.asarray(_load_cue_events(s["h5"])["cue_samples"], np.int64)
                smp, y, _per, _nd = ls.three_way_segments(
                    ev, lick_mode="postcue", cue_samples=cues, lick_window_s=ls.SEGMENT_S)
                f0 = lf.sample_to_frame(s, smp)
            except Exception as ex:                                    # noqa: BLE001
                skipped.append(f"{lab}: {type(ex).__name__} {str(ex)[:60]}")
                continue
            post_n = max(1, round(ls.SEGMENT_S * fs_img))
            T = sig.shape[1]
            keep = (f0 >= 0) & (f0 + post_n <= T)
            if not keep.any():
                skipped.append(f"{lab}: no segment inside imaging coverage")
                continue
            nbin_ = min(int(ls.SEGMENT_BINS), int(post_n))
            from wfield_local.locanmf_position_decoder import _window_feature
            X = np.array([_window_feature(sig, int(w), post_n, nbin_, 0.0) for w in f0[keep]])
            # THE TIME AXIS IS THE FRAME INDEX, which is session time inside the imaging period --
            # the same axis `state_time_bins` binned, so the two reports are directly comparable.
            tb = np.clip((f0[keep] / max(1, T) * nb).astype(int), 0, nb - 1)
            d = _day(an, mmdd)
            if d is None:
                continue
            got[lab] = (X, y[keep], tb, int(d))
            print(f"  .. {lab} {len(tb)} segments ({time.time() - t0:.0f}s)", flush=True)
        if got:
            got_all[an] = got

    # ------------------------------------------------------------------ score
    from wfield_local import epochs
    # rows[epoch][bin] -> list of balanced accuracies, one per session
    rows, overall = {}, {}
    for an, got in got_all.items():
        pre = [(X, y) for X, y, _t, d in got.values() if d <= 0]
        if not pre:
            continue
        frozen = ld.fit_frozen(np.vstack([p for p, _q in pre]),
                               np.concatenate([q for _p, q in pre]))
        for lab, (X, y, tb, d) in sorted(got.items()):
            if d <= 0:
                keep = [(p, q) for l2, (p, q, _t, dq) in got.items() if l2 != lab and dq <= 0]
                if not keep:
                    continue
                m = ld.fit_frozen(np.vstack([p for p, _q in keep]),
                                  np.concatenate([q for _p, q in keep]))
            else:
                m = frozen
            ep = epochs.epoch_of(lab)
            if ep is None or m is None:
                continue
            r_all = ld.score(m, X, y)
            if r_all is None:
                continue
            overall.setdefault(ep, []).append(r_all["balacc"])
            for j in range(nb):
                sel = tb == j
                if sel.sum() < 20:
                    continue
                r = ld.score(m, X[sel], y[sel])
                if r is not None:
                    rows.setdefault(ep, {}).setdefault(j, []).append(r["balacc"])

    order = [e for e in ("pre", "acute", "subacute", "chronic") if e in rows]
    print(f"\n{'=' * 78}\nFROZEN STATE DECODER, SCORED WITHIN EACH FIFTH OF THE SESSION\n{'=' * 78}")
    if not order:
        print("\nNOTHING WAS SCORED. This is a failed run, NOT a negative result.")
        for x in skipped:
            print("   skipped:", x)
        return 1
    print(f"{'epoch':<10}{'overall':>9}" + "".join(f"{f'bin {j + 1}':>9}" for j in range(nb))
          + f"{'spread':>9}{'n sess':>8}")
    for ep in order:
        v = [float(np.mean(rows[ep][j])) if j in rows[ep] else float("nan") for j in range(nb)]
        fin = [x for x in v if np.isfinite(x)]
        print(f"{ep:<10}{np.mean(overall[ep]):>9.3f}"
              + "".join(f"{x:>9.3f}" if np.isfinite(x) else f"{'--':>9}" for x in v)
              + f"{(max(fin) - min(fin)) if fin else float('nan'):>9.3f}"
              + f"{len(overall[ep]):>8}")
    print("\nspread = max-min across bins. Chance is 1/3 for three classes.")
    print("  small spread  -> the decoder is not reading session time")
    print("  large spread  -> accuracy tracks WHEN, and the preservation claim needs re-reading")
    print(f"\ntested {sum(len(v) for v in overall.values())} session-scores over {len(order)} "
          f"epochs; {len(skipped)} skipped")
    for x in skipped:
        print("   skipped:", x)
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
