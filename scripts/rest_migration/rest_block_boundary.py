"""PERSISTENCE or ANTICIPATION? The rest periods at BLOCK BOUNDARIES separate them.

THE QUESTION THIS ANSWERS. Rest carries position information (`rest_position_permutation`:
observed/null 1.443 over 44 sessions, 41/44 above null, on the STRICT docked window where the spout
is away and stationary). With no target present, two readings remain:

    PERSISTENCE    a trace of the target the animal has just been licking at
    ANTICIPATION   preparation for the next one -- available because positions run in ~6-trial
                   BLOCKS, so the next position is predictable from recent history

WITHIN a block those are coextensive: the last target and the next target ARE THE SAME POSITION, and
no amount of decoding separates them. **At a block BOUNDARY they differ**, and that is the whole
design:

    rest resembles the PRECEDING position  -> persistence / post-movement trace
    rest resembles the FOLLOWING position  -> anticipation / preparation
    neither, or both equally               -> the effect is about block CONTEXT rather than either
                                              target, which is itself informative

THESE PERIODS ALREADY EXIST AND ARE CURRENTLY DISCARDED. Both `rest_carries_position` and
`rest_position_permutation` keep a rest period only where the bracketing trials AGREE, precisely so
the label is unambiguous. This script keeps the opposite set.

THE MEASURE. For each boundary rest period, correlate its map against the session's mean rest map
for the PRECEDING position and for the FOLLOWING one, both computed from WITHIN-BLOCK periods only
(so the references never contain boundary periods and cannot be circular). Report the difference,
per session and pooled over animals.

    r_prev - r_next  > 0   persistence
                     < 0   anticipation

WHY THE REFERENCES MUST EXCLUDE BOUNDARY PERIODS. If a boundary period contributed to either
reference it would correlate with itself, and the comparison would be decided by which reference had
more of the test period in it. Excluded by construction here, and asserted rather than assumed.

THE ANIMAL IS THE UNIT. Sessions within an animal share a basis, a window and a set of blocks, so
pooling them as independent would overstate the evidence -- the same nested rule every other family
in this deck uses.

RUN:  python -m scripts.rest_migration.rest_block_boundary [--docked]
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


def _corr(a, b, mask):
    x, y = a[mask].ravel(), b[mask].ravel()
    if x.size < 10 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan
    x = x - x.mean(); y = y - y.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else np.nan


def main() -> int:
    import h5py

    from wfield_local import beta_maps as bm
    from wfield_local import config, daq_io, joint_basis
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS, _load_cue_events
    from wfield_local.quiet_periods import quiet_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-engagement-gate", action="store_true",
                    help="keep the pre-2026-09-16 ungated behaviour, for measuring the "
                         "size of the correction only -- never for a reported result")
    ap.add_argument("--docked", action="store_true", default=True)
    ap.add_argument("--loose", dest="docked", action="store_false",
                    help="use the loose rest window instead of the strict docked one")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    mask = bm.stat_mask()
    want = set(config.phase_labels("pre"))
    todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")]
    if a.limit:
        todo = todo[: a.limit]
    t0 = time.time()
    per_animal, skipped, n_bound = {}, [], 0

    for s in todo:
        lab = s["label"]
        qs = sorted(glob.glob(f"{quiet_dir(s['mc'])}/*quiet_sample.npy"))
        if not qs:
            skipped.append(f"{lab}: no rest mask"); continue
        try:
            rest = np.load(qs[0]).astype(bool)
            with h5py.File(s["h5"], "r") as f:
                dn = [x.decode() for x in f["digital/channel_names"][:]]
                packed = f["digital/packed_samples"][:, 0]
            pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
            ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
            cue = _load_cue_events(s["h5"])
            # THE REPAIRED CLASSIFIER -- the 0806 sessions (one per animal, all PRE-STROKE)
            # collapse 6 positions to 4 under the raw one, 144-192 trials each. Measured
            # 2026-09-16; docs/REST_ENGAGEMENT_AUDIT.md.
            codes = np.asarray(classify_cues_with_backup(s, cue, verbose=False))
            cs = np.asarray(cue["cue_samples"], np.int64)
            fs_samp = _frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
            u, v = joint_basis._load_session(s["mc"])
            if a.docked:
                from wfield_local.docked_periods import docked_mask, docked_mask_reconstructed
                from wfield_local.spout_behavior import discover_sessions
                an_, mmdd_ = lab.split("_")[0], lab.split("_")[1]
                cands = discover_sessions(config.resolver(), f"2026{mmdd_}", [an_])
                dm = None
                if cands:
                    dm = docked_mask(cands[0],
                                     daq_io.rising_edges((packed >> dn.index("sync")) & 1),
                                     rest.shape[0])
                if dm is None:
                    dm = docked_mask_reconstructed(cs, codes, ts, rest.shape[0])
                if dm is None:
                    skipped.append(f"{lab}: no docked window"); continue
                rest = rest & dm[: rest.shape[0]]
        except Exception as ex:                                        # noqa: BLE001
            skipped.append(f"{lab}: {type(ex).__name__} {str(ex)[:50]}"); continue
        if fs_samp is None:
            continue

        V = np.asarray(v)
        T = min(V.shape[1], rest.shape[0])
        V = V[:, :T]
        f_of = np.clip(fs_samp, 0, rest.shape[0] - 1)

        engaged = None
        if not getattr(a, "no_engagement_gate", False):
            from wfield_local.rest_engagement import engaged_by_cue
            engaged, _gn = engaged_by_cue(s, cs, codes)
            if "UNGATED" in _gn:
                print(f"  !! {s['label']}: {_gn}", flush=True)

        pad = np.concatenate([[0], rest.view(np.int8), [0]])
        dif = np.diff(pad)
        within, boundary = {}, []                 # within[pos] -> [frame arrays]; boundary -> recs
        for aa, bb in zip(np.flatnonzero(dif > 0), np.flatnonzero(dif < 0)):
            prev = np.searchsorted(cs, aa, "right") - 1
            nxt = np.searchsorted(ts, bb, "left")
            if prev < 0 or nxt >= len(ts):
                continue
            nc = np.searchsorted(cs, ts[nxt], "left")
            if nc >= len(codes) or prev >= len(codes) or codes[prev] < 0 or codes[nc] < 0:
                continue
            # ENGAGEMENT GATE (2026-09-16). BOTH bracketing trials must be working -- and here that
            # matters twice over: a boundary period whose FOLLOWING trial is already in the quit
            # period is not a position CHANGE, it is a state change, and would be scored as
            # anticipation of a position the animal never went on to work.
            if engaged is not None and not (engaged[prev] and engaged[nc]):
                continue
            fr = np.flatnonzero((f_of >= aa) & (f_of < bb))
            fr = fr[fr < T]
            if fr.size < 4:
                continue
            # SELECTED ON POSITION CHANGE, NOT ON BLOCK ID -- which is what makes this immune to
            # the same-position-adjacent-blocks problem (Priya, 2026-09-13; `block_ids.py`, 2.8% of
            # blocks). A far_L block scheduled straight after another far_L block is a BLOCK
            # transition but not a POSITION transition, so it has codes[prev] == codes[nc] and lands
            # in `within`. It can never enter the boundary set, where it would be a period whose
            # "preceding" and "following" references are the same map and whose difference is
            # therefore exactly zero by construction -- diluting the effect toward no-difference.
            if codes[prev] == codes[nc]:
                within.setdefault(int(codes[prev]), []).append(fr)
            else:
                boundary.append((fr, int(codes[prev]), int(codes[nc])))

        # REFERENCES FROM WITHIN-BLOCK PERIODS ONLY -- asserted, not assumed.
        ref = {}
        for q, frs in within.items():
            idx = np.concatenate(frs)
            if idx.size >= 100:
                ref[q] = (u @ V[:, idx].mean(1)).reshape(bm.MAP_SHAPE)
        usable = [(fr, p, n) for fr, p, n in boundary if p in ref and n in ref]
        if len(usable) < 8:
            skipped.append(f"{lab}: only {len(usable)} usable boundary periods"); continue
        n_bound += len(usable)

        d = []
        for fr, p, n in usable:
            m = (u @ V[:, fr].mean(1)).reshape(bm.MAP_SHAPE)
            rp, rn = _corr(m, ref[p], mask), _corr(m, ref[n], mask)
            if np.isfinite(rp) and np.isfinite(rn):
                d.append((rp, rn))
        if len(d) < 8:
            skipped.append(f"{lab}: only {len(d)} scorable boundary periods"); continue
        rp = float(np.mean([x[0] for x in d])); rn = float(np.mean([x[1] for x in d]))
        per_animal.setdefault(lab.split("_")[0], []).append((rp, rn, len(d)))
        print(f"  .. {lab}: n={len(d):>3} boundary  r_prev {rp:+.3f}  r_next {rn:+.3f}  "
              f"diff {rp - rn:+.3f}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'=' * 76}\nPERSISTENCE vs ANTICIPATION at block boundaries "
          f"({'DOCKED' if a.docked else 'loose'} window)\n{'=' * 76}")
    if len(per_animal) < 2:
        print("\nTOO FEW ANIMALS -- a failed run, not a negative result.")
        for x in skipped[:10]:
            print("   skipped:", x)
        return 1
    print(f"{'animal':<8}{'sessions':>10}{'n periods':>11}{'r_prev':>9}{'r_next':>9}{'diff':>9}")
    diffs = []
    for an in sorted(per_animal):
        v = per_animal[an]
        rp = float(np.mean([x[0] for x in v])); rn = float(np.mean([x[1] for x in v]))
        print(f"{an:<8}{len(v):>10}{sum(x[2] for x in v):>11}{rp:>9.3f}{rn:>9.3f}{rp - rn:>9.3f}")
        diffs.append(rp - rn)
    m = float(np.mean(diffs))
    print(f"\nmean over animals: r_prev - r_next = {m:+.4f}   ({sum(1 for d in diffs if d > 0)}"
          f"/{len(diffs)} animals positive)")
    print(f"total boundary periods used: {n_bound}")
    print("\n  > 0  PERSISTENCE -- rest resembles the position just licked at")
    print("  < 0  ANTICIPATION -- rest resembles the position coming next")
    print("  ~ 0  neither: the signal is about block CONTEXT, not either target")
    print(f"\ntested {sum(len(v) for v in per_animal.values())} sessions; {len(skipped)} skipped")
    for x in skipped[:6]:
        print("   skipped:", x)
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
