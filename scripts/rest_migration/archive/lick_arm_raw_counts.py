"""RAW per-position trial counts entering the LICK-arm fit — is far-contra 0, or thin-but-present?

Priya, 2026-09-16: *"for the lick-only analyses with far_R=0 class, the weighting doesn't really
make a difference anyway since there are no far_R trials to include in the data"*.

THE DISTINCTION THIS SETTLES, and it is not cosmetic. `meanref_balance` reported far_R acute as
`n_trials = 0`, but that column is `beta_maps.session_maps`'s ``used`` dict, which is populated
only when a position clears ``MIN_TRIALS_PER_CLASS = 20``:

    if n[q] and counts.get(q, 0) >= MIN_TRIALS_PER_CLASS:
        out[q] = ...;  used[q] = counts.get(q, 0)

**The floor governs which MAPS are emitted, not which trials enter the fit.** ``y`` is never
filtered by it, so a position with 1-19 trials IS a class in the multinomial — it simply gets no
map. So `used = 0` means "no session reached 20", which is NOT the same as "no trials exist".

WHY THE DIFFERENCE MATTERS. ``class_weight="balanced"`` sets each class's weight to
``n_samples / (n_classes * n_class)``, so a class with 5 trials against ~1100 elsewhere is
up-weighted ~200x. `beta_maps` already warns about exactly this at the ``MIN_TRIALS_PER_CLASS``
definition: *"`class_weight='balanced'` UP-weights a rare class, so a five-trial position is more
influential rather than less, which argues for the floor not against."* If far-contra is
thin-but-present acutely, then balancing the lick arm makes a near-empty class MORE influential over
the other five — an argument AGAINST the uniform weighting adopted 2026-09-16, not for it.

    truly 0 trials     -> weighting is irrelevant to far_R, as Priya says, and the measured
                          <= 0.035 moves are the remaining FIVE classes being equalised.
    1-19 trials        -> those trials are in the fit and balancing amplifies them ~200x. The
                          uniform change needs revisiting for this arm.

RUN:  python -m scripts.rest_migration.archive.lick_arm_raw_counts [--align lick] [--variant lick]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np


def main() -> int:
    from wfield_local import config, joint_basis
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached

    ap = argparse.ArgumentParser()
    ap.add_argument("--align", default="lick")
    ap.add_argument("--variant", default="lick")
    ap.add_argument("--post-s", type=float, default=2.0)
    a = ap.parse_args()

    from wfield_local import beta_maps as bm

    raw = defaultdict(lambda: defaultdict(int))     # epoch -> position -> raw trials in the FIT
    thin = defaultdict(list)                        # epoch -> (label, position, n) below the floor
    for an in ANIMALS:
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        for s in [x for x in SESSIONS if x["label"] in want]:
            d = _day(an, s["label"].split("_")[-1])
            if d is None:
                continue
            e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
            if e is None:
                continue
            try:
                _u, v = joint_basis._load_session(s["mc"])
                _X, y, _g, _Xn, _yn, _r, _ie, _idxn = trial_features_cached(
                    s, _args("locanmf", a.align, a.post_s), signal=np.asarray(v),
                    feat_region=np.arange(v.shape[0]), signal_key=f"svt:rank{v.shape[0]}",
                    with_indices=True)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
                continue
            y = np.asarray(y)
            # THE LICK ARM IS THE ENGAGED TRIALS ONLY -- no `working` augmentation. That is the
            # whole point: it conditions on a detected lick.
            cls, cnt = np.unique(y, return_counts=True)
            for c, n in zip(cls, cnt):
                q = POSITION_NAMES.get(int(c), str(c))
                raw[e][q] += int(n)
                if 0 < int(n) < bm.MIN_TRIALS_PER_CLASS:
                    thin[e].append((s["label"], q, int(n)))
            print(f"  .. {s['label']} [{e}] " +
                  " ".join(f"{POSITION_NAMES.get(int(c), c)}={n}" for c, n in zip(cls, cnt)),
                  flush=True)

    order = [q for q in ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")]
    print(f"\n{'=' * 80}\nRAW trials per position ENTERING THE FIT -- {a.align}/{a.variant}\n"
          f"(the MIN_TRIALS_PER_CLASS={bm.MIN_TRIALS_PER_CLASS} floor gates MAPS, not the fit)\n"
          f"{'=' * 80}")
    print(f"{'epoch':<10}" + "".join(f"{q:>14}" for q in order))
    for e in ("pre", "acute", "subacute", "chronic"):
        if e not in raw:
            continue
        print(f"{e:<10}" + "".join(f"{raw[e].get(q, 0):>14}" for q in order))

    print(f"\n{'=' * 80}\nSESSIONS WITH A THIN-BUT-PRESENT CLASS (1-{bm.MIN_TRIALS_PER_CLASS - 1} "
          f"trials): these ARE in the fit\n{'=' * 80}")
    any_thin = False
    for e in ("pre", "acute", "subacute", "chronic"):
        for lab, q, n in sorted(thin.get(e, []), key=lambda t: t[2]):
            any_thin = True
            w = f"~{sum(raw[e].values()) / max(1, len(order) * n):.0f}x"
            print(f"  {e:<10}{lab:<14}{q:<14}n={n:<5}balanced weight {w}")
    if not any_thin:
        print("  NONE -- every position is either absent or above the floor, so the floor and the")
        print("  fit agree and `class_weight='balanced'` cannot amplify a near-empty class.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
