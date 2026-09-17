"""Does the REST baseline itself carry POSITION information? The control the reference assumes.

Priya, 2026-09-13: "can we test if the REST activity shows significant difference between trials of
different positions (ie for only rest times between two trials of the same position, do rest periods
between all 6 positions look similar to or different from each other)"

WHY THIS IS THE LOAD-BEARING CONTROL. The rest reference makes two claims. The first -- that the
subtrahend is IDENTICAL for all six positions -- is true by construction: it is one session mean, so
it cannot couple the positions. The second is the one everything rests on and has never been tested:
that it carries NO POSITION INFORMATION, so subtracting it removes nothing real. If rest activity
depends on which position the animal is sitting between, then that single session mean is a
position-WEIGHTED MIXTURE, and subtracting it biases each position's map by a different amount --
an F12-style over-subtraction hiding inside the reference we just adopted as primary.

It also bears on the position-graded amplitude change (1.48 near -> 0.48 far): a position-graded
REST baseline would manufacture exactly that, with no task-evoked change at all.

THE DESIGN, and why the bracketing rule is Priya's rather than mine. A rest period sits BETWEEN two
trials, so it has two candidate labels. Taking only periods whose PRECEDING and FOLLOWING trial share
a position removes the ambiguity entirely rather than picking one end and hoping. It costs little
here because positions are presented in ~6-trial BLOCKS (`block_ids`), so consecutive same-position
trials are the common case, not the exception.

    per session, per position p:  mean of U @ SVT over rest frames whose bracketing trials are BOTH p
    per animal:                   mean over that animal's sessions
    contrast tested:              each position's rest map MINUS the mean rest map over all positions

The contrast is one-vs-rest ON REST, so it asks exactly "does resting cortex look different depending
on which position is coming next". Tested with the deck's own nested animals->sessions bootstrap
against zero (`vs_zero_contour`), on the same eroded stat mask, with the same max-statistic
threshold -- so a positive here is directly comparable to a positive on any map figure.

A SCALAR SUMMARY TOO, because a contour map answers "where" and not "how much": between-position
RMS against a WITHIN-position split-half RMS. The split-half is the noise floor -- the same position's
rest, split randomly in two, differs by this much for nothing. If between-position is no larger, rest
carries no position information at the precision this cohort can measure.

RUN:  python -m scripts.rest_migration.rest_carries_position
"""
from __future__ import annotations

import glob
import json
import sys
import time

import numpy as np


def _frame_samples(mc, fmdir, regime, pco):
    """DAQ sample index of each corrected imaging frame -- the mapping `quiet_periods` uses."""
    if regime == "B":
        fm = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_frame_map.npz"))
        summ = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_summary.json"))
        if not fm or not summ:
            return None
        off = int(json.load(open(summ[0]))["chosen_exposure_offset"])
        z = np.load(fm[0])
        return pco[np.clip(z["original_frame_index_ch0"] + off, 0, len(pco) - 1)]
    return pco[np.arange(len(pco) // 2) * 2]


def main() -> int:
    import h5py

    from wfield_local import beta_maps as bm
    from wfield_local import config, daq_io, joint_basis
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS, _load_cue_events
    from wfield_local.quiet_periods import quiet_dir

    t0 = time.time()
    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    want = set(config.phase_labels("pre"))          # PRE-STROKE ONLY: the cleanest test of the
    # assumption, before any lesion can make rest position-dependent for a downstream reason.
    per_animal = {}
    n_ok, n_bouts_kept, n_bouts_all = 0, 0, 0

    for s in [x for x in SESSIONS if x["label"] in want and x.get("h5")]:
        d = quiet_dir(s["mc"])
        qs = sorted(glob.glob(f"{d}/*quiet_sample.npy"))
        if not qs:
            continue
        try:
            rest = np.load(qs[0]).astype(bool)
            with h5py.File(s["h5"], "r") as f:
                dn = [x.decode() for x in f["digital/channel_names"][:]]
                packed = f["digital/packed_samples"][:, 0]
            pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
            ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
            cue = _load_cue_events(s["h5"])
            # THE REPAIRED CLASSIFIER. Dead `spout_bit1` collapses six positions to four on the 0806
            # sessions -- ONE PER ANIMAL, ALL PRE-STROKE, 144-192 trials mislabelled each (measured
            # 2026-09-16). docs/REST_ENGAGEMENT_AUDIT.md; STATUS_2026-09-16 pitfall 6.
            codes = classify_cues_with_backup(s, cue, verbose=False)
            cs = np.asarray(cue["cue_samples"], np.int64)
            fs_samp = _frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
            u, v = joint_basis._load_session(s["mc"])
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if fs_samp is None:
            continue

        # ---- label every rest BOUT by its bracketing trials, keeping only unambiguous ones
        # ALWAYS GATED, no opt-out: `main()` here takes no argparse, and the patch that added this
        # originally wrote `getattr(a, ...)` where `a` is the rest-bout LOOP VARIABLE further down --
        # undefined on the first session, a stale sample index afterwards. Caught by ruff F821
        # before it ran. Copying a snippet between scripts whose argument names differ is the same
        # templating failure that spread the missing gate itself.
        from wfield_local.rest_engagement import engaged_by_cue
        engaged, _gn = engaged_by_cue(s, cs, codes)
        if "UNGATED" in _gn:
            print(f"  !! {s['label']}: {_gn}", flush=True)

        pad = np.concatenate([[0], rest.view(np.int8), [0]])
        dif = np.diff(pad)
        starts, stops = np.flatnonzero(dif > 0), np.flatnonzero(dif < 0)
        n_bouts_all += len(starts)
        lab = np.full(rest.shape[0], -1, np.int8)
        for a, b in zip(starts, stops):
            prev = np.searchsorted(cs, a, "right") - 1          # trial that just ended
            nxt = np.searchsorted(ts, b, "left")                # trial about to start
            if prev < 0 or nxt >= len(ts):
                continue
            nxt_cue = np.searchsorted(cs, ts[nxt], "left")
            if nxt_cue >= len(codes) or prev >= len(codes):
                continue
            if codes[prev] == codes[nxt_cue] and codes[prev] >= 0:
                # ENGAGEMENT GATE (2026-09-16). docs/REST_ENGAGEMENT_AUDIT.md.
                if engaged is not None and not (engaged[prev] and engaged[nxt_cue]):
                    continue
                lab[a:b] = codes[prev]
                n_bouts_kept += 1

        frame_lab = lab[np.clip(fs_samp, 0, rest.shape[0] - 1)]
        T = min(v.shape[1], frame_lab.shape[0])
        frame_lab = frame_lab[:T]
        maps = {}
        for q in CONF_LABELS:
            c = code_of.get(q)
            sel = frame_lab == c
            if int(sel.sum()) < 200:        # ~6 s of rest; below that a map is noise
                continue
            maps[q] = (u @ np.asarray(v)[:, :T][:, sel].mean(1)).reshape(bm.MAP_SHAPE)
        if len(maps) >= 4:
            per_animal.setdefault(s["label"].split("_")[0], []).append(maps)
            n_ok += 1
        print(f"  .. {s['label']}: {len(maps)}/6 positions ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{n_ok} pre-stroke sessions, {len(per_animal)} animals; "
          f"{n_bouts_kept}/{n_bouts_all} rest bouts had UNAMBIGUOUS position labels", flush=True)
    if len(per_animal) < 2:
        print("!! too few animals to test", flush=True)
        return 1

    mask = bm.stat_mask()
    # ---- per animal: mean over sessions, then each position minus the across-position mean
    # PER POSITION, OVER THE SESSIONS THAT HAVE IT -- not the intersection across all sessions.
    # The first version intersected, so one thin session shrank the whole animal's position set and
    # `len(common) < 4` then dropped that animal entirely. Every position had fewer than 3 animals
    # and the bootstrap silently tested NOTHING, printing a "no significant dependence" verdict that
    # was vacuous rather than negative. Priya: "how could so few animals share the same positions?
    # there should be dozens of each per session" -- correct, and the loss was aggregation, not data.
    by_pos = {}
    for an, sess in per_animal.items():
        am = {}
        for q in CONF_LABELS:
            got = [m[q] for m in sess if q in m]
            if got:
                am[q] = np.mean(got, axis=0)
        if len(am) < 4:
            continue
        # The grand mean is over the positions THIS animal has, so the contrast is always
        # "this position minus the animal's own across-position mean".
        grand = np.mean(list(am.values()), axis=0)
        for q in am:
            by_pos.setdefault(q, {})[an] = am[q] - grand
    print("  positions x animals available: "
          + ", ".join(f"{q}:{len(by_pos.get(q, {}))}" for q in CONF_LABELS), flush=True)

    print("\nDOES REST DIFFER BY POSITION? each position's rest map minus the across-position mean")
    print(f"{'position':<15}{'n animals':>10}   nested bootstrap vs zero")
    any_sig, n_tested = False, 0
    for q in CONF_LABELS:
        d = by_pos.get(q)
        if not d or len(d) < 3:
            continue
        cm, lab_txt = bm.vs_zero_contour({a: [m] for a, m in d.items()}, mask=mask)
        n = 0 if cm is None else int(cm.sum())
        any_sig |= n > 0
        n_tested += 1
        print(f"{q:<15}{len(d):>10}   {lab_txt}")

    # ---- the scalar: between-position RMS against a within-position split-half noise floor
    rng = np.random.default_rng(0)
    betw, within = [], []
    for an, sess in per_animal.items():
        am = {q: np.mean([m[q] for m in sess if q in m], axis=0)
              for q in CONF_LABELS if any(q in m for m in sess)}
        if len(am) < 4 or len(sess) < 2:
            continue
        grand = np.mean(list(am.values()), axis=0)
        betw += [float(np.sqrt(np.mean((am[q] - grand)[mask] ** 2))) for q in am]
        for q in am:
            have = [i for i, m in enumerate(sess) if q in m]
            if len(have) < 2:
                continue
            idx = rng.permutation(have)
            h1 = np.mean([sess[i][q] for i in idx[: len(idx) // 2]], axis=0)
            h2 = np.mean([sess[i][q] for i in idx[len(idx) // 2:]], axis=0)
            within.append(float(np.sqrt(np.mean(((h1 - h2) / 2)[mask] ** 2))))
    if betw and within:
        b, w = float(np.mean(betw)), float(np.mean(within))
        print(f"\nBETWEEN-position RMS {b:.5f}   WITHIN-position split-half RMS {w:.5f}   "
              f"ratio {b / w if w else float('nan'):.2f}")
        print("  ratio ~1 means rest differs between positions no more than the SAME position's")
        print("  rest differs from itself -- i.e. no position information the cohort can resolve.")
    print(f"\nVERDICT: {'REST CARRIES POSITION INFORMATION' if any_sig else 'no significant position dependence in rest'}")
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
