"""Is rest's position dependence POSITION, or session DRIFT aliased through the block structure?

THE CONFOUND THIS EXISTS TO RESOLVE. `rest_carries_position` found rest differs by position in 6/6
positions (up to 1,042 of 2,022 bins; between/within RMS ratio 1.45). But positions are presented in
~6-TRIAL BLOCKS, so POSITION IS CONFOUNDED WITH TIME-WITHIN-SESSION. Slow drift -- photobleaching,
arousal, the rest baseline's own documented drift -- makes rest during block k differ from block j,
and since blocks carry position labels that reads as "position information" with nothing
position-specific about it. The first test cannot tell the two apart.

THE DISCRIMINATION. Split each position's rest frames at the session's midpoint and measure two
differences in the same units, on the same mask:

    DRIFT     same position, EARLY half vs LATE half          -- pure time, position held fixed
    POSITION  different positions, matched halves (early-early -- pure position, time matched
              and late-late, averaged)

    POSITION >> DRIFT   the dependence is really about position; the REST reference is
                        conceptually wrong and a session-mean subtrahend cannot be unbiased
    POSITION ~= DRIFT   it is drift aliased onto blocks; the reference is salvageable by
                        detrending or block-balancing rather than abandoning

Both are computed WITHIN animal and then averaged over animals, so neither is inflated by
between-animal differences.

ALSO RENDERS THE PICTURE (Priya, 2026-09-13: "can you also generate a figure for a visual, comparing
the rest maps across positions, per animal?"): rows = animals, columns = positions, each cell that
animal's mean rest map MINUS its own across-position mean -- the exact contrast the bootstrap tested,
so the figure and the statistic show the same quantity.

RUN:  python -m scripts.rest_migration.rest_position_vs_drift
"""
from __future__ import annotations

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
        off = int(json.load(open(summ[0]))["chosen_exposure_offset"])
        z = np.load(fm[0])
        return pco[np.clip(z["original_frame_index_ch0"] + off, 0, len(pco) - 1)]
    return pco[np.arange(len(pco) // 2) * 2]


def main() -> int:
    import h5py

    from wfield_local import beta_maps as bm
    from wfield_local import config, daq_io, epoch_figures as ef, joint_basis
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS, _load_cue_events
    from wfield_local.paths import PathResolver
    from wfield_local.plot_spout_trial_averages import _classify_cues
    from wfield_local.quiet_periods import quiet_dir

    t0 = time.time()
    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    want = set(config.phase_labels("pre"))
    # per animal: list of {position: (early_map, late_map)}
    per_animal = {}

    for s in [x for x in SESSIONS if x["label"] in want and x.get("h5")]:
        qs = sorted(glob.glob(f"{quiet_dir(s['mc'])}/*quiet_sample.npy"))
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
            codes = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
            cs = np.asarray(cue["cue_samples"], np.int64)
            fs_samp = _frame_samples(s["mc"], s.get("fmdir"), s.get("regime"), pco)
            u, v = joint_basis._load_session(s["mc"])
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            continue
        if fs_samp is None:
            continue

        pad = np.concatenate([[0], rest.view(np.int8), [0]])
        dif = np.diff(pad)
        lab = np.full(rest.shape[0], -1, np.int8)
        for a, b in zip(np.flatnonzero(dif > 0), np.flatnonzero(dif < 0)):
            prev = np.searchsorted(cs, a, "right") - 1
            nxt = np.searchsorted(ts, b, "left")
            if prev < 0 or nxt >= len(ts):
                continue
            nc = np.searchsorted(cs, ts[nxt], "left")
            if nc < len(codes) and prev < len(codes) and codes[prev] == codes[nc] >= 0:
                lab[a:b] = codes[prev]

        frame_lab = lab[np.clip(fs_samp, 0, rest.shape[0] - 1)]
        T = min(v.shape[1], frame_lab.shape[0])
        frame_lab, V = frame_lab[:T], np.asarray(v)[:, :T]
        mid = T // 2                       # SESSION MIDPOINT: the time axis, in frames
        halves = {}
        for q in CONF_LABELS:
            sel = frame_lab == code_of.get(q)
            e, l = sel.copy(), sel.copy()
            e[mid:] = False
            l[:mid] = False
            # BOTH halves need enough frames or the split is noise, not drift.
            if int(e.sum()) < 100 or int(l.sum()) < 100:
                continue
            halves[q] = ((u @ V[:, e].mean(1)).reshape(bm.MAP_SHAPE),
                         (u @ V[:, l].mean(1)).reshape(bm.MAP_SHAPE))
        if len(halves) >= 4:
            per_animal.setdefault(s["label"].split("_")[0], []).append(halves)
        print(f"  .. {s['label']}: {len(halves)}/6 positions with both halves "
              f"({time.time() - t0:.0f}s)", flush=True)

    mask = bm.stat_mask()
    drift, pos = [], []
    for an, sess in per_animal.items():
        for h in sess:
            qs_ = sorted(h)
            # DRIFT: same position, early vs late
            for q in qs_:
                e, l = h[q]
                drift.append(float(np.sqrt(np.mean(((e - l) / 2)[mask] ** 2))))
            # POSITION: different positions, SAME half -- so time is matched
            for i in range(len(qs_)):
                for j in range(i + 1, len(qs_)):
                    for k in (0, 1):
                        d = h[qs_[i]][k] - h[qs_[j]][k]
                        pos.append(float(np.sqrt(np.mean((d / 2)[mask] ** 2))))

    print(f"\n{sum(len(v) for v in per_animal.values())} sessions, {len(per_animal)} animals")
    if drift and pos:
        d, p = float(np.mean(drift)), float(np.mean(pos))
        print(f"\nDRIFT    (same position, early vs late)      RMS {d:.5f}   n={len(drift)}")
        print(f"POSITION (different positions, matched time) RMS {p:.5f}   n={len(pos)}")
        # THIS RATIO IS RETIRED AS A TEST -- printed for continuity, NOT to be read as a verdict.
        # It is WITHDRAWN for two independent reasons (2026-09-13): (1) the two contrasts are not
        # matched on TIME SEPARATION -- DRIFT separates its estimates by ~half a session while
        # POSITION compares estimates separated by ~zero, because positions interleave in ~6-trial
        # blocks throughout each half; (2) a ratio of MAGNITUDES is not a test -- if both quantities
        # are noise-dominated (RMS ~0.003 in raw dF/F) the ratio sits near 1 whatever the truth.
        #
        # It read 1.02 on the retired rest definition and was taken as "drift, not position". On
        # `restdock05` the same arithmetic gives ~1.09. NEITHER number licenses a conclusion, and
        # the old verdict text printed here would have invited the opposite reading from the same
        # broken statistic. `rest_position_permutation` is the test: it keeps the block-time
        # structure INSIDE the null and gives observed/null 1.622 over 44 pre-stroke sessions,
        # above null in 44/44, 4/4 animals (restdock05, 2026-09-15).
        print(f"\nPOSITION / DRIFT = {p / d if d else float('nan'):.2f}   [RETIRED STATISTIC -- "
              f"NOT a verdict]")
        print("  This ratio does NOT decide drift vs position: its two contrasts are unmatched on")
        print("  time separation, and a ratio of magnitudes cannot separate 'both real' from")
        print("  'neither resolvable'. Use `rest_position_permutation` (circular-shift null):")
        print("  observed/null 1.622, 44/44 sessions, 4/4 animals -- REST CARRIES POSITION.")

    # ------------------------------------------------------------------ the figure
    out = None
    cells, rows = {}, []
    for an in sorted(per_animal):
        am = {}
        for q in CONF_LABELS:
            got = [np.mean(h[q], axis=0) for h in per_animal[an] if q in h]
            if got:
                am[q] = np.mean(got, axis=0)
        if len(am) < 4:
            continue
        grand = np.mean(list(am.values()), axis=0)
        rows.append(an)
        for q in am:
            cells[(an, q)] = am[q] - grand
    if cells:
        d = __import__("pathlib").Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"
        out = ef.map_grid(
            cells, d, name="epoch_15x_REST_by_position_by_animal",
            title="REST activity by spout position, per animal -- does the baseline carry position?",
            row_labels=rows, col_labels=[q for q in CONF_LABELS],
            edges=bm.atlas_edges(), blank=bm.excluded_mask(), row_scaled=True,
            cbar_label="rest activity minus that\nanimal's across-position mean",
            subtitle=(
                "EACH CELL IS THAT ANIMAL'S MEAN REST MAP FOR ONE POSITION, MINUS ITS OWN "
                "ACROSS-POSITION MEAN -- the exact contrast the bootstrap tests, so figure and "
                "statistic show the same quantity. Rest periods are labelled by position ONLY when "
                "the preceding and following trial share one, so the label is unambiguous. "
                "PRE-STROKE SESSIONS ONLY. Colour scale is PER ANIMAL: these are raw dF/F units and "
                "animals differ in expression, so rows are not comparable to each other. "
                "IF THE REST BASELINE WERE POSITION-INDEPENDENT -- which is what the REST reference "
                "assumes when it subtracts ONE session mean from all six positions -- every cell "
                "here would be noise. CAVEAT: positions come in ~6-trial BLOCKS, so position is "
                "confounded with time-within-session; see the DRIFT vs POSITION comparison printed "
                "by this script, which is what separates them."))
        print(f"\nfigure: {out}", flush=True)
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
