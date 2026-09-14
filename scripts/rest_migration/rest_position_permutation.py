"""Does REST carry POSITION information, tested against a null that contains the drift confound?

WHY THIS EXISTS: THE EARLIER CONTROL DOES NOT SUPPORT THE CONCLUSION DRAWN FROM IT.
`rest_position_vs_drift` compared two RMS magnitudes and reported

    DRIFT     same position, EARLY half vs LATE half        RMS 0.00282
    POSITION  different positions, MATCHED halves           RMS 0.00288     ratio 1.02

and that ratio was read as "rest differs between positions by no more than it differs from itself
over time, therefore drift, not position coding". That reading is wrong for two independent reasons.

FIRST, THE TWO CONTRASTS ARE NOT MATCHED ON TIME SEPARATION -- and the asymmetry runs the WRONG WAY
for the conclusion. The DRIFT contrast separates its two estimates by roughly HALF A SESSION. The
POSITION contrast does not: positions are interleaved in ~6-trial blocks throughout each half, so
within a half every position's rest frames average to nearly the SAME mean time. It compares
estimates separated by ~zero. A near-zero-separation contrast matching a half-session-separation
contrast is, if anything, evidence that something OTHER than drift contributes to the former.

SECOND, A RATIO OF MAGNITUDES IS NOT A TEST. If both quantities are dominated by estimation NOISE --
which is plausible at RMS ~0.003 in raw dF/F -- the ratio is ~1 whatever the truth, and the
comparison cannot distinguish "position matters as much as drift" from "neither is resolvable".

Priya, 2026-09-13, looking at `epoch_15x_REST_by_position_by_animal`: "I thought we looked and
decided rest did not have position information...but that doesn't seem clear to me." It is not
clear, and the figure is why: the per-animal cells carry large, spatially coherent, hemisphere-
asymmetric structure that does not look like noise.

THE TEST THIS RUNS. Build the observed between-position RMS, then compare it to a null that
CONTAINS the drift confound rather than one that removes it:

    CIRCULAR SHIFT the sequence of position labels over rest periods ordered in time, by a random
    offset, within each session. The labels keep their block structure and the signal keeps its
    drift; only the CORRESPONDENCE between them is broken.

Under that null, "drift aliased onto blocks" is fully present -- a shifted label sequence still
assigns whole blocks to whole stretches of session time. So:

    observed >> null   rest carries position information the block-time structure cannot explain
    observed ~= null   the apparent structure IS drift-through-blocks, which is what the earlier
                       control claimed but did not show

WHAT TURNS ON IT. The REST reference subtracts ONE baseline from all six positions, so its
INDEPENDENCE claim survives either way -- a common subtrahend cannot couple them. What does not
survive a positive result is the second claim: that the subtrahend carries nothing real. If rest
during a position's blocks genuinely differs, then subtracting a session-common baseline leaves that
difference IN the position maps, as a contaminant that looks exactly like position coding.

RUN:  python -m scripts.rest_migration.rest_position_permutation [--perm 200]
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
        with open(summ[0]) as _fh:
            off = int(json.load(_fh)["chosen_exposure_offset"])
        z = np.load(fm[0])
        return pco[np.clip(z["original_frame_index_ch0"] + off, 0, len(pco) - 1)]
    return pco[np.arange(len(pco) // 2) * 2]


def _between_rms(maps, mask):
    """RMS of each position's deviation from the across-position mean, averaged over positions."""
    ms = list(maps.values())
    if len(ms) < 3:
        return np.nan
    gm = np.mean(ms, axis=0)
    return float(np.mean([np.sqrt(np.mean((m - gm)[mask] ** 2)) for m in ms]))


def main() -> int:
    import h5py

    from wfield_local import beta_maps as bm
    from wfield_local import config, daq_io, joint_basis
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS, _load_cue_events
    from wfield_local.plot_spout_trial_averages import _classify_cues
    from wfield_local.quiet_periods import quiet_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--docked", action="store_true",
                    help="restrict rest to the STRICT spout-docked interval (dock -> next "
                         "trial_start): no target present AND no spout movement. A session whose "
                         "GUI/DAQ clocks cannot be aligned DROPS OUT rather than falling back.")
    a = ap.parse_args()

    rng = np.random.default_rng(0)          # FIXED SEED: a null that moves between runs is not one
    mask = bm.stat_mask()
    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    want = set(config.phase_labels("pre"))
    t0 = time.time()
    obs_all, null_all, skipped = [], [], []

    todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")]
    if a.limit:
        todo = todo[: a.limit]

    for s in todo:
        lab = s["label"]
        qs = sorted(glob.glob(f"{quiet_dir(s['mc'])}/*quiet_sample.npy"))
        if not qs:
            skipped.append(f"{lab}: no rest mask")
            continue
        try:
            rest = np.load(qs[0]).astype(bool)
            with h5py.File(s["h5"], "r") as f:
                dn = [x.decode() for x in f["digital/channel_names"][:]]
                packed = f["digital/packed_samples"][:, 0]
            if a.docked:
                # DOCKED IS A SUBSET OF REST, never a replacement: it says where the SPOUT is and
                # nothing about the animal, so the not-running / not-licking conditions still apply.
                from wfield_local.docked_periods import docked_mask
                from wfield_local.spout_behavior import discover_sessions
                an_, mmdd_ = lab.split("_")[0], lab.split("_")[1]
                cands = discover_sessions(config.resolver(), f"2026{mmdd_}", [an_])
                sdir = cands[0] if cands else None
                dm = None if sdir is None else docked_mask(
                    sdir, daq_io.rising_edges((packed >> dn.index("sync")) & 1), rest.shape[0])
                if dm is None:
                    skipped.append(f"{lab}: no usable docked window (log/clock)")
                    continue
                rest = rest & dm[: rest.shape[0]]
            pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
            ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
            cue = _load_cue_events(s["h5"])
            codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
            cs = np.asarray(cue["cue_samples"], np.int64)
            fs_samp = _frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
            u, v = joint_basis._load_session(s["mc"])
        except Exception as ex:                                        # noqa: BLE001
            skipped.append(f"{lab}: {type(ex).__name__} {str(ex)[:50]}")
            continue
        if fs_samp is None:
            continue

        # REST PERIODS IN TIME ORDER, each with the position of its bracketing trials (only where
        # the preceding and following trial AGREE, so the label is unambiguous).
        pad = np.concatenate([[0], rest.view(np.int8), [0]])
        dif = np.diff(pad)
        starts, stops = np.flatnonzero(dif > 0), np.flatnonzero(dif < 0)
        periods = []
        for aa, bb in zip(starts, stops):
            prev = np.searchsorted(cs, aa, "right") - 1
            nxt = np.searchsorted(ts, bb, "left")
            if prev < 0 or nxt >= len(ts):
                continue
            nc = np.searchsorted(cs, ts[nxt], "left")
            if nc < len(codes) and prev < len(codes) and codes[prev] == codes[nc] >= 0:
                periods.append((aa, bb, int(codes[prev])))
        if len(periods) < 30:
            skipped.append(f"{lab}: only {len(periods)} labelled rest periods")
            continue

        T = min(v.shape[1], rest.shape[0])
        V = np.asarray(v)[:, :T]
        f_of = np.clip(fs_samp, 0, rest.shape[0] - 1)

        # Frame indices per period, once -- the permutation only relabels them.
        per_frames = []
        for aa, bb, c in periods:
            fr = np.flatnonzero((f_of >= aa) & (f_of < bb))
            fr = fr[fr < T]
            if fr.size:
                per_frames.append((fr, c))
        if len(per_frames) < 30:
            skipped.append(f"{lab}: only {len(per_frames)} periods with frames")
            continue

        # BOUND AS DEFAULTS, not captured: `u`, `V` and `per_frames` are rebound every session,
        # and a closure over the loop variable would silently map one session's frames onto
        # another's basis if this were ever called after the loop moved on.
        def _maps(labels, u=u, V=V, per_frames=per_frames):
            out = {}
            for q in CONF_LABELS:
                c = code_of.get(q)
                fr = [f for f, lc in zip([p[0] for p in per_frames], labels) if lc == c]
                if not fr:
                    continue
                idx = np.concatenate(fr)
                if idx.size >= 100:
                    out[q] = (u @ V[:, idx].mean(1)).reshape(bm.MAP_SHAPE)
            return out

        lab_seq = np.array([c for _f, c in per_frames])
        obs = _between_rms(_maps(lab_seq), mask)
        if not np.isfinite(obs):
            skipped.append(f"{lab}: fewer than 3 positions reached the frame floor")
            continue

        # THE NULL: circular shift of the LABEL sequence over time-ordered periods. Blocks stay
        # blocks and drift stays drift; only their correspondence is broken.
        n = len(lab_seq)
        nulls = []
        for _ in range(int(a.perm)):
            k = int(rng.integers(1, n))
            r = _between_rms(_maps(np.roll(lab_seq, k)), mask)
            if np.isfinite(r):
                nulls.append(r)
        if len(nulls) < 20:
            skipped.append(f"{lab}: only {len(nulls)} usable permutations")
            continue
        obs_all.append(obs)
        null_all.append(float(np.mean(nulls)))
        p = (1 + sum(1 for x in nulls if x >= obs)) / (1 + len(nulls))
        print(f"  .. {lab}: obs {obs:.5f}  null {np.mean(nulls):.5f}  ratio "
              f"{obs / max(1e-12, np.mean(nulls)):.2f}  p={p:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    print(f"tested {len(obs_all)} sessions; {len(skipped)} skipped")
    for x in skipped[:8]:
        print("   skipped:", x)
    if len(obs_all) < 3:
        print("\nTOO FEW SESSIONS TESTED -- a failed run, not a negative result.")
        return 1
    o, nl = np.array(obs_all), np.array(null_all)
    print(f"\nobserved between-position RMS  mean {o.mean():.5f}")
    print(f"circular-shift null            mean {nl.mean():.5f}")
    print(f"OBSERVED / NULL                     {o.mean() / max(1e-12, nl.mean()):.3f}")
    print(f"sessions with observed > null:      {int((o > nl).sum())}/{len(o)}")
    print("\n  >> 1  rest carries position information the block-time structure cannot explain;")
    print("        the REST subtrahend is not position-neutral and leaves a contaminant in the maps")
    print("  ~= 1  the structure IS drift aliased onto blocks, as the earlier control claimed")
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
