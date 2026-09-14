"""Is the +-3 s treadmill buffer earning its cost, or is `speed < 1 mm/s` already doing the work?

Priya, 2026-09-13: *"should we reduce the +-3s criterion? < 1 mm/s is already pretty strict"*.

THREE REASONS TO SUSPECT IT, BEFORE MEASURING ANYTHING:

1. PROVENANCE. `treadmill_buffer_s: [3.0, 3.0]` was carried over from stroke_orofacial, the same
   source as the 8 s post-reward buffer that was retired on 2026-09-12 for being a different task's
   constant. It was never derived on this rig.
2. INTERNAL INCONSISTENCY. `lick_buffer_s` in the same config block is [1.0, 3.0] -- ASYMMETRIC,
   short before and long after, which is the physiologically shaped choice: the long tail exists to
   let an evoked response DECAY, and there is no symmetric reason to exclude time BEFORE an event.
   The treadmill buffer being [3.0, 3.0] is the odd one out.
3. INTERACTION WITH THE DOCKED WINDOW. Docked intervals run ~1.35 s. A +-3 s buffer means a single
   sample at 1 mm/s anywhere in a 7.7 s span deletes the whole interval. PS93_0806 retains 28% of
   its rest under the docked term; this is the likely mechanism.

NONE OF THAT IS EVIDENCE THE BUFFER IS WRONG. The buffer's JOB is to keep movement-related activity
out of the baseline, and widefield calcium plus its hemodynamic tail is genuinely slow. So the
question is empirical and has a sharp form:

    DO THE FRAMES A NARROWER BUFFER ADMITS, WHICH THE WIDER ONE EXCLUDED, ACTUALLY DIFFER
    IN CORTICAL ACTIVITY FROM THE FRAMES BOTH ADMIT?

If they are indistinguishable, the wider buffer is removing nothing but data. If they differ, it is
doing exactly what it is for and the cost is the price. THIS IS THE TEST, and it is stated in
advance of the answer: `marginal` vs `common` frames, compared as mean maps (RMS of the difference
against the RMS of the common map, plus their correlation) and as time-local baselines.

WHY THE ANSWER MATTERS NOW RATHER THAN LATER. Changing this buffer changes the rest mask, which is
a subtrahend for the maps AND a CLASS for the state decoder -- the same blast radius as the docked
redefinition. Deciding it before the redo costs one re-render instead of two.

RUN:  python -m scripts.rest_migration.treadmill_buffer_sweep [--limit 6] [--docked]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

#: Candidate (before, after) treadmill buffers, in seconds. [3,3] is the incumbent; [1,3] mirrors
#: the lick buffer's asymmetry; [1,1] and [0.5,1.5] probe how far it can fall before the marginal
#: frames start to look different. 0.25 is Priya's proposal, 2026-09-13.
CANDIDATES = ((3.0, 3.0), (2.0, 3.0), (1.0, 3.0), (1.0, 1.0), (0.5, 1.5), (0.25, 0.25), (0.0, 0.0))

#: THE LICK BUFFER, swept the same way. Priya, 2026-09-13: *"we shouldn't need a lick buffer at all
#: - the spout isn't close enough to lick in ITI"*, then *"maybe we just buffer 0.25s on either
#: side"*.
#:
#: THE PREMISE IS RIGHT AND THE BUFFER'S JOB IS A DIFFERENT ONE, which is why this is measured
#: rather than set. The spout retracts, so the animal CANNOT lick it during the ITI -- that is
#: established (`docked_periods`). But `lick_buffer_s` does not exist to exclude licks that happen
#: in the ITI; it exists to exclude the CORTICAL RESPONSE to licks that happened just before it.
#: GCaMP plus its hemodynamic tail decays over ~0.5-1.5 s, so a post-lick buffer is physiologically
#: motivated even in an interval where no licking is possible.
#:
#: AND THE DOCKED ANCHOR ALREADY SUPPLIES MOST OF IT. The response window closes at cue + 3.5 s and
#: dock lands at cue + 4.61-4.80 s, so a docked interval opens >= 1.1 s after the last lick the
#: task can produce. A 3 s post-lick buffer on top of that is largely redundant -- but "largely" is
#: not a measurement, because CONSUMPTION licking continues after the response window closes and is
#: not bounded by it. What decides this is the measured time from the last lick to docked onset,
#: reported below, and whether the frames a shorter buffer admits look like rest.
#:
#: WHY IT MATTERS MORE THAN THE TREADMILL BUFFER: the lick term is THE DOMINANT ONE, excluding
#: 73.7% of samples pre-stroke and 82.5% at chronic, and it is the stated reason chronic rest stays
#: ~0.5x of pre. If it can be shortened, the chronic caveat may go with it.
LICK_CANDIDATES = ((1.0, 3.0), (1.0, 1.5), (0.5, 1.0), (0.25, 0.5), (0.25, 0.25), (0.0, 0.0))


def _rest_for_buffer(n, fs, speed, lick_onsets, cue_s, ts_s, st_s, params, session_dir, buf,
                     sync_s=None, codes=None, which="treadmill"):
    from wfield_local.quiet_periods import rest_mask

    p = dict(params)
    p[f"{which}_buffer_s"] = list(buf)
    m, _note = rest_mask(n, fs, speed, lick_onsets, cue_s, ts_s, st_s, params=p,
                         session_dir=session_dir, sync_s=sync_s, position_codes=codes)
    return m


def _lick_timing_in_docked(lick_onsets, dm, fs):
    """``(n_licks_inside, median_gap_s, p05_gap_s, n_intervals)`` -- TESTS PRIYA'S PREMISE DIRECTLY.

    If the spout is retracted during the docked interval, essentially NO licks should fall inside
    one, and the time from the last lick to each interval's onset should already exceed any
    sensible decay buffer. Both are measured rather than assumed: the lick sensor is on the spout,
    so a detected lick inside a docked interval is either a real contact the retraction did not
    prevent or a sensor artefact, and either one would matter.

    p05 rather than the minimum, because one interval preceded by a late consumption lick should
    not set the buffer for the whole session -- but it is reported so the tail is visible.
    """
    pad = np.concatenate([[0], dm.view(np.int8), [0]])
    d = np.diff(pad)
    starts, stops = np.flatnonzero(d > 0), np.flatnonzero(d < 0)
    if starts.size == 0:
        return 0, np.nan, np.nan, 0
    lo = np.sort(np.asarray(lick_onsets, np.int64))
    inside = 0
    gaps = []
    for aa, bb in zip(starts, stops):
        i0, i1 = np.searchsorted(lo, aa, "left"), np.searchsorted(lo, bb, "left")
        inside += int(i1 - i0)
        if i0 > 0:
            gaps.append((aa - lo[i0 - 1]) / float(fs))
    g = np.asarray(gaps, float)
    return (inside, float(np.median(g)) if g.size else np.nan,
            float(np.percentile(g, 5)) if g.size else np.nan, int(starts.size))


def main() -> int:
    import h5py

    from wfield_local import config, daq_io, joint_basis
    from wfield_local.lick_detection import detect_licks
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS, _load_cue_events
    from wfield_local.plot_spout_trial_averages import _classify_cues
    from wfield_local.position_reference_maps import REST_BASELINE_BINS, _timelocal_from_mask

    from wfield_local.rest_by_position import frame_samples
    from wfield_local.treadmill import calibrate_treadmill, smooth_treadmill

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--docked", action="store_true")
    ap.add_argument("--which", choices=("treadmill", "lick"), default="treadmill")
    a = ap.parse_args()
    cands = CANDIDATES if a.which == "treadmill" else LICK_CANDIDATES
    incumbent = cands[0]
    print(f"sweeping {a.which}_buffer_s; incumbent {incumbent}; docked={a.docked}")

    seg = config.defaults()["segmentation"]
    tw, q = seg["treadmill"], seg["rest"]
    lk = config.defaults()["lick_detection"]
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")]
    # SPREAD ACROSS THE COHORT, not the first N by date -- the first N are all pre-stroke 0606 and
    # the buffer's cost is worst where rest is scarcest, which is post-stroke.
    step = max(1, len(todo) // max(1, a.limit))
    todo = todo[::step][: a.limit]

    t0 = time.time()
    frac = {b: [] for b in cands}
    marg = {b: [] for b in cands[1:]}
    prem = []
    for s in todo:
        lab = s["label"]
        print(f"\n=== {lab} " + "=" * (56 - len(lab)), flush=True)
        try:
            with h5py.File(s["h5"], "r") as f:
                fs = float(f.attrs["sample_rate_hz"])
                an_names = [x.decode() for x in f["analog/channel_names"][:]]

                def _ana(nm):
                    i = an_names.index(nm)
                    if "samples_int16" in f["analog"]:
                        sc = float(f["analog/int16_scale_volts_per_count"][i])
                        of = float(f["analog/int16_offset_volts"][i])
                        return f["analog/samples_int16"][:, i].astype(np.float32) * sc + of
                    return np.asarray(f["analog/samples"][:, i], np.float32)

                lick_v = _ana(lk.get("channel", "lick_analog"))
                tread_v = _ana(tw["channel"])
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
            pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
            cue = _load_cue_events(s["h5"])
            codes = np.asarray(_classify_cues(cue["cue_samples"], cue["strobe_samples"],
                                              cue["strobe_codes"]))
            fsamp = frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
            _u, v = joint_basis._load_session(s["mc"])
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {type(ex).__name__} {str(ex)[:70]}")
            continue
        if fsamp is None:
            print("  !! no frame map")
            continue

        qp = dict(q)
        qp["docked"] = bool(a.docked)
        # PRIYA'S PREMISE, TESTED DIRECTLY: with the spout retracted, are there licks inside the
        # docked interval at all, and how long since the last one when it opens?
        try:
            from wfield_local.docked_periods import docked_mask_any
            from wfield_local.spout_behavior import discover_sessions
            an_, mmdd_ = lab.split("_")[0], lab.split("_")[1]
            cds = discover_sessions(config.resolver(), f"2026{mmdd_}", [an_])
            dmm, dsrc = docked_mask_any(cds[0] if cds else None, sync, cue["cue_samples"], codes,
                                        ts_e, n)
            if dmm is not None:
                ins, med, p05, nint = _lick_timing_in_docked(lo, dmm, fs)
                prem.append((ins, med, p05, nint))
                print(f"  DOCKED PREMISE: {ins} lick(s) inside {nint} docked intervals; "
                      f"gap from last lick to onset  median {med:.2f}s  p05 {p05:.2f}s  [{dsrc}]")
        except Exception as ex:
            print(f"  .. docked premise check skipped: {type(ex).__name__} {str(ex)[:50]}")

        masks = {}
        for b in cands:
            try:
                masks[b] = _rest_for_buffer(n, fs, speed, lo, cue_e / fs, ts_e / fs, st_e / fs,
                                            qp, None if not a.docked else s.get("session_dir"),
                                            b, sync_s=sync / fs, codes=codes)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! buffer {b}: {type(ex).__name__} {str(ex)[:50]}")
        if incumbent not in masks:
            continue

        V = np.asarray(v)
        T = V.shape[1]
        f_of = np.clip(np.asarray(fsamp), 0, n - 1)

        def frames_of(m):
            """Frame indices whose DAQ sample falls inside the mask."""
            return np.flatnonzero(m[f_of[:T]])

        base_fr = frames_of(masks[incumbent])
        print(f"  incumbent {incumbent}: rest {masks[incumbent].mean():.4f} of samples, "
              f"{base_fr.size} frames")
        for b in cands:
            frac[b].append(float(masks[b].mean()) if b in masks else np.nan)

        common_map = V[:, base_fr].mean(1) if base_fr.size else None
        for b in cands[1:]:
            if b not in masks or common_map is None:
                continue
            fr = frames_of(masks[b])
            extra = np.setdiff1d(fr, base_fr, assume_unique=False)
            if extra.size < 30:
                print(f"  {str(b):<12} +{extra.size:>6} frames -- too few to compare")
                continue
            # THE TEST: do the frames this buffer ADMITS AND [3,3] EXCLUDED differ from the ones
            # both admit? Compared as mean maps in the SVT basis (RMS of the difference against the
            # RMS of the common map) and as a correlation.
            em = V[:, extra].mean(1)
            d = float(np.sqrt(np.mean((em - common_map) ** 2)))
            r0 = float(np.sqrt(np.mean(common_map ** 2)))
            cc = float(np.corrcoef(em, common_map)[0, 1])
            # And the quantity that actually matters: does the TIME-LOCAL BASELINE move?
            bl_c = _timelocal_from_mask(V, masks[incumbent][f_of[:T]], REST_BASELINE_BINS)
            bl_b = _timelocal_from_mask(V, masks[b][f_of[:T]], REST_BASELINE_BINS)
            bd = (float(np.sqrt(np.mean((bl_b - bl_c) ** 2))) / max(1e-12,
                  float(np.sqrt(np.mean(bl_c ** 2))))
                  if (bl_c is not None and bl_b is not None) else np.nan)
            marg[b].append((d / max(1e-12, r0), cc, bd, extra.size, fr.size))
            print(f"  {str(b):<12} +{extra.size:>6} frames ({100 * fr.size / max(1, base_fr.size) - 100:+5.1f}%)  "
                  f"marginal-vs-common RMS {100 * d / max(1e-12, r0):5.1f}%  corr {cc:+.4f}  "
                  f"baseline moves {100 * bd:4.1f}%")

    print(f"\n{'=' * 78}\nTREADMILL BUFFER SWEEP -- is +-3 s earning its cost?\n{'=' * 78}")
    print(f"{'buffer':<14}{'rest frac':>11}{'vs [3,3]':>11}{'marg RMS':>11}{'marg corr':>11}"
          f"{'baseline':>11}")
    base = np.nanmean(frac[incumbent]) if frac[incumbent] else np.nan
    for b in cands:
        f_ = np.nanmean(frac[b]) if frac[b] else np.nan
        if b == incumbent:
            print(f"{str(b):<14}{f_:>11.4f}{'--':>11}{'--':>11}{'--':>11}{'--':>11}")
            continue
        v_ = marg.get(b) or []
        if not v_:
            print(f"{str(b):<14}{f_:>11.4f}{100 * (f_ / base - 1):>10.1f}%{'n/a':>11}")
            continue
        print(f"{str(b):<14}{f_:>11.4f}{100 * (f_ / base - 1):>10.1f}%"
              f"{100 * np.mean([x[0] for x in v_]):>10.1f}%{np.mean([x[1] for x in v_]):>11.4f}"
              f"{100 * np.nanmean([x[2] for x in v_]):>10.1f}%")
    print("\nREAD IT LIKE THIS:")
    print("  marginal RMS small + correlation ~1  ->  the excluded frames look like rest;")
    print("                                          the wider buffer is removing DATA, not movement")
    print("  marginal RMS large + correlation low ->  the buffer is doing its job and the cost is the price")
    print(f"\n{len(todo)} session(s)  [done in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
