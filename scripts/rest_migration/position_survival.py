"""Do the behavioural exclusions remove ITIs EVENLY across positions, or do they track the deficit?

Priya, 2026-09-13: *"the time-weighted by sweep will be affected by thrown-out ITIs due to running
etc"*.

THE CONCERN, AND WHY IT IS THE ONE THAT COULD STILL CHANGE THE DEFINITION. Sweep-binning guarantees
each position's BLOCK occurs in every bin (94.5% of sweeps are exactly 6 blocks). It guarantees
nothing about the FRAMES that survive: the treadmill and lick terms are applied AFTER the task
structure has done its balancing. If post-stroke the animal licks more after far-contralateral
trials -- extra attempts, frustration, longer consumption -- then far-contra ITIs are preferentially
removed by the lick buffer, and the rest baseline's COMPOSITION TRACKS THE DEFICIT.

**That is precisely the failure that retired the 8 s post-reward definition on 2026-09-12**, where
"quiet" measured 4.4% of frames pre-stroke against 17.1% acutely because the category was anchored
on the animal's PERFORMANCE. Re-introducing it one layer down, via a term nobody is looking at,
would be worse than the original because the definition now LOOKS behaviour-independent.

WHAT IS MEASURED. For every position, the fraction of its DOCKED ITI samples that survive into the
final rest mask, broken down by epoch:

    denominator  the docked interval following each of that position's trials, with NO behavioural
                 exclusion at all -- the spout is parked, that is all
    numerator    those same samples that also clear the treadmill and lick terms

AND ATTRIBUTED TO A TERM, because the two have different implications. Treadmill exclusion tracking
position would be odd; LICK exclusion tracking position is the expected route and the dangerous one,
since licking is what the lesion changes.

HOW TO READ THE ANSWER, stated before it is known:

  survival position-FLAT, and flat at every epoch    -> the concern collapses; the equal weighting
                                                        already handles what is left
  survival varies by position but the SPREAD is      -> a static imbalance; position-weighting
  stable across epochs                                  handles it, since it is not deficit-coupled
  the SPREAD GROWS post-stroke, and far-contra       -> the definition is still performance-coupled
  is the loser                                          and needs a further fix before any redo

THE THIRD OUTCOME IS THE ONE THAT MATTERS, and note that position-weighting does NOT rescue it:
weighting repairs unequal COUNTS, but if only the calmest far-contra ITIs survive then that
position's baseline is estimated from an unrepresentative subset however it is weighted.

RUN:  python -m scripts.rest_migration.position_survival [--limit 24]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

#: Which term to zero out, to attribute the loss. "full" applies everything.
ARMS = ("full", "no_lick_buffer", "no_treadmill_buffer")


def main() -> int:
    import h5py

    from wfield_local import config, daq_io, epochs
    from wfield_local.docked_periods import docked_mask_any
    from wfield_local.lick_detection import detect_licks
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS, _load_cue_events
    from wfield_local.plot_spout_trial_averages import _classify_cues
    from wfield_local.quiet_periods import rest_mask
    from wfield_local.spout_behavior import discover_sessions
    from wfield_local.treadmill import calibrate_treadmill, smooth_treadmill

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    a = ap.parse_args()

    seg = config.defaults()["segmentation"]
    tw, q = seg["treadmill"], seg["rest"]
    lk = config.defaults()["lick_detection"]
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")]
    step = max(1, len(todo) // max(1, a.limit))
    todo = todo[::step][: a.limit]

    # {epoch: {arm: {position: [surv_rate, ...]}}}
    acc, t0, skipped = {}, time.time(), []
    for s in todo:
        lab = s["label"]
        ep = epochs.epoch_of(lab)
        if ep is None:
            continue
        try:
            with h5py.File(s["h5"], "r") as f:
                fs = float(f.attrs["sample_rate_hz"])
                an = [x.decode() for x in f["analog/channel_names"][:]]

                def _ana(nm):
                    i = an.index(nm)
                    if "samples_int16" in f["analog"]:
                        sc = float(f["analog/int16_scale_volts_per_count"][i])
                        of = float(f["analog/int16_offset_volts"][i])
                        return f["analog/samples_int16"][:, i].astype(np.float32) * sc + of
                    return np.asarray(f["analog/samples"][:, i], np.float32)

                lick_v, tread_v = _ana(lk.get("channel", "lick_analog")), _ana(tw["channel"])
                dn = [x.decode() for x in f["digital/channel_names"][:]]
                packed = f["digital/packed_samples"][:, 0]
            n = int(lick_v.size)
            speed = smooth_treadmill(calibrate_treadmill(tread_v, tw["offset_v"],
                                                         tw["volt_sec_per_rot"], tw["mm_per_rot"]),
                                     fs, tw["smoothing_sigma_s"])
            det = detect_licks(lick_v, fs, thresh_upper=lk["thresh_upper"],
                               thresh_lower=lk["thresh_lower"],
                               lockout_s=tuple(lk["lockout_falling_edge_s"]),
                               min_ili_s=lk.get("min_ili_ms", 0) / 1000.0)
            lo = np.asarray(det["lick_onsets"], np.int64)
            cue_e = daq_io.rising_edges((packed >> dn.index("cue")) & 1)
            ts_e = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
            st_e = daq_io.rising_edges((packed >> dn.index("spout_strobe")) & 1)
            sync = daq_io.rising_edges((packed >> dn.index("sync")) & 1)
            cue = _load_cue_events(s["h5"])
            codes = np.asarray(_classify_cues(cue["cue_samples"], cue["strobe_samples"],
                                              cue["strobe_codes"]))
            cs = np.asarray(cue["cue_samples"], np.int64)
            an_, mmdd = lab.split("_")[0], lab.split("_")[1]
            cands = discover_sessions(config.resolver(), f"2026{mmdd}", [an_])
            dm, _src = docked_mask_any(cands[0] if cands else None, sync, cs, codes, ts_e, n)
            if dm is None:
                skipped.append(f"{lab}: no docked window")
                continue
        except Exception as ex:                                        # noqa: BLE001
            skipped.append(f"{lab}: {type(ex).__name__} {str(ex)[:50]}")
            continue

        masks = {}
        for arm in ARMS:
            p = dict(q)
            if arm == "no_lick_buffer":
                p["lick_buffer_s"] = [0.0, 0.0]
            elif arm == "no_treadmill_buffer":
                p["treadmill_buffer_s"] = [0.0, 0.0]
            try:
                m, _ = rest_mask(n, fs, speed, lo, cue_e / fs, ts_e / fs, st_e / fs, params=p,
                                 session_dir=None)
                masks[arm] = m
            except Exception:                                          # noqa: BLE001
                pass
        if "full" not in masks:
            skipped.append(f"{lab}: rest_mask failed")
            continue

        # DENOMINATOR: the docked interval after each trial, labelled by THAT trial's position, with
        # NO behavioural exclusion. This is "the ITI the task gave this position".
        den = {}
        num = {arm: {} for arm in masks}
        for k, c in enumerate(codes):
            if c < 0 or k >= len(cs):
                continue
            j = np.searchsorted(ts_e, cs[k], "right")
            if j >= ts_e.size:
                continue
            aa, bb = int(cs[k]), int(ts_e[j])
            if bb <= aa:
                continue
            seg_ = dm[aa:bb]
            tot = int(seg_.sum())
            if tot <= 0:
                continue
            den[int(c)] = den.get(int(c), 0) + tot
            for arm, m in masks.items():
                num[arm][int(c)] = num[arm].get(int(c), 0) + int((seg_ & m[aa:bb]).sum())
        if not den:
            skipped.append(f"{lab}: no docked ITI samples")
            continue

        line = []
        for c in sorted(den):
            r = num["full"].get(c, 0) / max(1, den[c])
            acc.setdefault(ep, {}).setdefault("full", {}).setdefault(c, []).append(r)
            for arm in masks:
                if arm != "full":
                    acc[ep].setdefault(arm, {}).setdefault(c, []).append(
                        num[arm].get(c, 0) / max(1, den[c]))
            line.append(f"{c}:{100 * r:.0f}%")
        print(f"  .. {lab:<12} [{ep:<8}] survival  " + "  ".join(line) +
              f"   ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'=' * 78}\nPER-POSITION SURVIVAL THROUGH THE REST MASK\n{'=' * 78}")
    if not acc:
        print("NOTHING MEASURED -- a failed run, not a negative result.")
        for x in skipped[:8]:
            print("   skipped:", x)
        return 1
    order = [e for e in ("pre", "acute", "subacute", "chronic") if e in acc]
    for arm in ARMS:
        if not any(arm in acc[e] for e in order):
            continue
        print(f"\n--- {arm} ---")
        print(f"{'epoch':<10}" + "".join(f"{('p%d' % c):>8}" for c in range(6)) +
              f"{'spread':>9}{'far-contra vs mean':>20}")
        for e in order:
            d = acc[e].get(arm) or {}
            vals = [float(np.mean(d[c])) if c in d else np.nan for c in range(6)]
            v = np.array(vals, float)
            fin = np.isfinite(v)
            if not fin.any():
                continue
            spread = float(np.nanmax(v) - np.nanmin(v))
            # FAR CONTRA is position 5 in CONF_LABELS order used elsewhere; reported as the
            # deficit-relevant one rather than assumed to be the loser.
            fc = v[5] - np.nanmean(v[fin]) if np.isfinite(v[5]) else np.nan
            print(f"{e:<10}" + "".join(f"{100 * x:>7.1f}%" if np.isfinite(x) else f"{'--':>8}"
                                       for x in v) +
                  f"{100 * spread:>8.1f}%{100 * fc:>19.1f}%")
    print("\nHOW TO READ IT:")
    print("  spread FLAT across epochs      ->  a static imbalance; position-weighting handles it")
    print("  spread GROWS post-stroke, and  ->  the definition is STILL PERFORMANCE-COUPLED, which")
    print("  far-contra is the loser            is the failure that retired the 8 s reward buffer.")
    print("                                     Weighting does NOT rescue this: it repairs unequal")
    print("                                     COUNTS, not an unrepresentative SAMPLE.")
    print("  compare `full` against `no_lick_buffer`: if the spread collapses without the lick")
    print("  term, licking is the route, which is the expected and dangerous one.")
    print(f"\n{sum(len(v.get('full', {}).get(0, [])) for v in acc.values())} session-positions; "
          f"{len(skipped)} sessions skipped")
    for x in skipped[:6]:
        print("   skipped:", x)
    print(f"[done in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
