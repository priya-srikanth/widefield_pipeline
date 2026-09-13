"""What the FINAL rest definition actually produced, read off the written masks.

Cheap, because it reads the per-corrected-frame masks and their summaries rather than re-deriving
anything from the DAQ. Run it after `recompute_masks` to get the numbers that go into the docs:

  * rest fraction by epoch, and the ACUTE/PRE ratio -- the quantity the whole redefinition targets.
    The retired definition measured 3.89; the sweep predicted 1.10 for a strobe-anchored
    approximation of this, and this is the real one.
  * bout durations, and what fraction admit a 1 s and a 2 s segment -- which is what decides whether
    the state decoder's window can move. `BEHAVIOURAL_STATE_CONTROL.md` set it to 1 s because the
    retired definition's quiet periods had a 1.10 s median and a 2 s window discarded 83%.
  * which ANCHOR each session used, because `trial_start`, the strobe fallback and the cue-only
    fallback are three different definitions and a cohort must not silently mix them.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter

import numpy as np


def main() -> int:
    from wfield_local import config, epoch_figures as ef
    from wfield_local.grant_figures import _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.quiet_periods import quiet_dir, quiet_variant

    fs_img = float(config.defaults()["preprocess"]["svd"]["fs"])
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    by_epoch, durs, anchors, missing = {}, {}, Counter(), []

    for s in SESSIONS:
        if s["label"] not in want:
            continue
        an, mmdd = s["label"].split("_")
        d = _day(an, mmdd)
        if d is None:
            continue
        e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
        if e is None:
            continue
        dd = quiet_dir(s["mc"])
        hits = sorted(glob.glob(f"{dd}/*quiet_frame.npy"))
        if not hits:
            missing.append(s["label"])
            continue
        q = np.load(hits[0]).astype(bool)
        by_epoch.setdefault(e, []).append(float(q.mean()))
        # bout lengths in IMAGING frames -> seconds
        pad = np.concatenate([[0], q.view(np.int8), [0]])
        dif = np.diff(pad)
        lens = (np.flatnonzero(dif < 0) - np.flatnonzero(dif > 0)) / fs_img
        durs.setdefault(e, []).append(lens)
        js = glob.glob(f"{dd}/*quiet_periods_summary.json")
        if js:
            try:
                anchors[json.load(open(js[0])).get("rest_anchor", "?")] += 1
            except Exception:                                          # noqa: BLE001
                anchors["(unreadable summary)"] += 1

    print(f"variant {quiet_variant()!r}; {sum(len(v) for v in by_epoch.values())} sessions with a "
          f"mask, {len(missing)} without")
    if missing:
        print(f"  missing: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    print(f"\nANCHOR USED: {dict(anchors)}")
    if len(anchors) > 1:
        print("  !! MORE THAN ONE ANCHOR IN THE COHORT -- these are different definitions and the "
              "sessions using each must be reported wherever the masks are used.")

    print("\nREST FRACTION OF CORRECTED FRAMES:")
    print(f"{'epoch':<10}{'n':>4}{'mean':>9}{'median':>9}{'min':>8}{'max':>8}")
    order = ("pre", "acute", "subacute", "chronic")
    for e in order:
        v = np.asarray(by_epoch.get(e, []))
        if v.size:
            print(f"{e:<10}{v.size:>4}{v.mean():>9.3f}{np.median(v):>9.3f}"
                  f"{v.min():>8.3f}{v.max():>8.3f}")
    pre = np.asarray(by_epoch.get("pre", []))
    if pre.size and pre.mean() > 0:
        print("\nRATIO TO PRE -- 1.00 means the definition does NOT track the deficit "
              "(the retired one measured 3.89 acute/pre):")
        for e in order[1:]:
            v = np.asarray(by_epoch.get(e, []))
            if v.size:
                print(f"   {e:<10} {v.mean() / pre.mean():5.2f}")

    print("\nREST BOUT DURATION (s), and what a segment window would keep:")
    print(f"{'epoch':<10}{'n bouts':>9}{'median':>9}{'p75':>8}{'p95':>8}{'>=1s':>8}{'>=2s':>8}")
    allb = []
    for e in order:
        if e not in durs:
            continue
        a = np.concatenate(durs[e])
        allb.append(a)
        print(f"{e:<10}{a.size:>9d}{np.median(a):>9.2f}{np.percentile(a, 75):>8.2f}"
              f"{np.percentile(a, 95):>8.2f}{float((a >= 1).mean()):>8.3f}"
              f"{float((a >= 2).mean()):>8.3f}")
    if allb:
        a = np.concatenate(allb)
        print(f"{'ALL':<10}{a.size:>9d}{np.median(a):>9.2f}{np.percentile(a, 75):>8.2f}"
              f"{np.percentile(a, 95):>8.2f}{float((a >= 1).mean()):>8.3f}"
              f"{float((a >= 2).mean()):>8.3f}")
        print("\nThe last two columns decide the state decoder's window. The retired definition gave"
              "\na 1.10 s median with 17% of periods admitting 2 s, which is why 1 s was chosen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
