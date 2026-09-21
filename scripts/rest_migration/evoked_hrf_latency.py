"""THE HAEMODYNAMIC DIP, TIMED DIRECTLY: does the evoked HRF slow after the stroke?

Priya, 2026-09-19: *"why does the lag relate to the neurovascular coupling - i thought the vascular
part was the later dip"*, then *"yes run the dip comparison"*.

**THIS EXISTS BECAUSE `rest_coupling` MEASURES A QUANTITY WHOSE ABSOLUTE VALUE IS NOT
INTERPRETABLE.** That module found the HRF lag lengthens by +0.156 s acutely -- the strongest
evidence so far for decreased coupling -- but it works on REST frames and reads the trough of a
cross-correlation that the hemodynamic regression has notched at lag zero, so its lobes are pushed
outward by the fit. Its own docstring says only CHANGES are interpretable, and then the absolute
0.31-0.47 s got quoted as "an ordinary HRF latency", which was loose.

The DIP is the interpretable version. It is the thing visible in a cue-aligned trace: 470 rises
with calcium, then 415 falls as blood arrives and absorbs. The latency between those two events is
a physiologically transparent number that can be checked against the literature, and it shares NO
TRIALS with the rest estimate -- rest is by definition outside every trial. Agreement would be
replication rather than a re-slice of the same data.

THREE DESIGN CHOICES THAT MATTER:

    PARENCHYMA ONLY      Vessels carry no GCaMP, so what the camera collects there is scattered
                         light from neighbouring cortex -- a diluted copy of the parenchymal
                         signal by the same factor in BOTH channels (0.41 vs 0.43 at 0.4 s, see
                         `channel_vessel_sign`). Including them adds no information and drags both
                         traces toward each other. The mask is the LOCAL-contrast one, because an
                         absolute intensity threshold selects the dim anterior edge instead.

    ENGAGEMENT-GATED     Via `_quit_trials`, the same gate the rest of the deck uses. The quit
                         period is 3.1% of frames pre-stroke and 18.7% acutely, so an UNGATED
                         epoch comparison has its composition track the independent variable.

    THE TROUGH MUST BE   PS94 has no late negative deflection at all (`channel_evoked_sign`), so
    NEGATIVE TO COUNT    `argmin` over the late window would return its smallest POSITIVE value --
                         a number with no trough under it. Sessions are flagged, counted, and the
                         summary is reported both ways.

    python -m scripts.rest_migration.evoked_hrf_latency [--animals PS92 ...]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

#: THE BOOTSTRAP LIVES IN ONE PLACE NOW. `analysis_kit` holds the nested animals->sessions
#: draw this module used to define for itself, bit-for-bit -- `tests/test_analysis_kit.py`
#: pins it against the pre-extraction source. Read that module before touching a draw: the
#: point-estimate convention DIFFERS between `boot_ci` (flat pool, for LEVELS) and
#: `boot_delta` (animal-weighted, for CHANGES), and the difference has retracted a result.
from wfield_local import analysis_kit as ak

PRE_S, POST_S = 2.0, 3.5
BASE = (-1.0, -0.2)
PEAK_WIN = (0.0, 1.5)        # s; where the 470 calcium peak is looked for
DIP_WIN = (0.4, 3.0)         # s; where the 415 haemodynamic trough is looked for


def session_latency(s):
    """``dict`` for one session: 470 peak time, 415 trough time, and their difference."""
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.hemo_variants import FS
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.rest_by_position import _session_daq
    from scripts.rest_migration.channel_position_maps import _quit_trials, _signals
    from scripts.rest_migration.channel_vessel_sign import _vesselness, PAREN_Q

    res = Path(s["mc"]) / "wfield_local_results"
    allen = res / "allen_aligned_affine8v1"
    sig, _ev = _signals(res, allen)
    S = {"415": sig[0][1], "470": sig[1][1], "SVTcorr": sig[2][1]}

    U = np.asarray(np.load(allen / "U_atlas.npy", mmap_mode="r"), np.float64)
    brain = np.asarray(np.load(allen / "allen_brain_mask_native_grid.npy")).astype(bool)
    favg = np.asarray(np.load(allen / "frames_average_atlas.npy"), np.float64)
    from wfield_local.hemo_variants import functional_channel
    mean_img = favg[functional_channel(s)] if favg.ndim == 3 else favg
    vness = _vesselness(mean_img, brain)
    paren = brain & (vness <= np.quantile(vness[brain], PAREN_Q))
    op = U[paren].mean(0)

    _rest, cs, _c, _ts, fs_samp, _sy = _session_daq(s)
    f_of = np.asarray(fs_samp)
    cue = _load_cue_events(s["h5"])
    codes = np.asarray(classify_cues_with_backup(s, cue, verbose=False))
    from wfield_local.plot_lick_aligned_averages import _load_daq_events
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    not_engaged = _quit_trials(s, np.asarray(cs), codes, np.asarray(lk["lick_samples"]))

    T = min(v.shape[1] for v in S.values())
    cue_f = np.searchsorted(f_of[:T], np.asarray(cs, np.int64))
    a, b = int(round(PRE_S * FS)), int(round(POST_S * FS))
    ok = (codes >= 0) & ~not_engaged & (cue_f > a) & (cue_f + b < T)
    ev = cue_f[ok]
    if ev.size < 30:
        return None

    t = np.arange(-a, b) / FS
    jb = (t >= BASE[0]) & (t <= BASE[1])
    tr = {}
    for k, v in S.items():
        x = op @ np.asarray(v[:, :T], np.float64)
        seg = np.stack([x[f - a:f + b] for f in ev])
        tr[k] = (seg - seg[:, jb].mean(1, keepdims=True)).mean(0) * 100.0

    jp = (t >= PEAK_WIN[0]) & (t <= PEAK_WIN[1])
    jd = (t >= DIP_WIN[0]) & (t <= DIP_WIN[1])
    k470 = int(np.flatnonzero(jp)[int(np.argmax(tr["470"][jp]))])
    k415 = int(np.flatnonzero(jd)[int(np.argmin(tr["415"][jd]))])
    return dict(label=s["label"], n_cue=int(ev.size),
                t_peak_470=float(t[k470]), amp_peak_470=float(tr["470"][k470]),
                t_dip_415=float(t[k415]), amp_dip_415=float(tr["415"][k415]),
                latency=float(t[k415] - t[k470]),
                # THE FLAG THAT KEEPS PS94 FROM CONTRIBUTING A LATENCY WITH NO TROUGH UNDER IT.
                dip_is_negative=bool(tr["415"][k415] < 0),
                dip_at_edge=bool(abs(t[k415] - DIP_WIN[1]) < 1.5 / FS))


def session_latency_row(lab):
    """One row for `lab`, or ``None``. MODULE-LEVEL for `parallel.fan_out` (ground rule 6).

    Spawn pickles the worker BY NAME and re-imports this module in each child, so the label is
    passed and the session record looked up inside rather than shipped across.
    """
    from wfield_local import config, epochs

    s = next((x for x in config.load_sessions() if x["label"] == lab), None)
    if s is None:
        return None
    r = session_latency(s)
    if r is None:
        return None
    r.update(animal=config.animal_of(lab), epoch=epochs.epoch_of(lab))
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes; default `parallel.default_jobs()` (cores-2, cap 8)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    # `analysis_kit.curated_sessions` is this filter, once. IT PRESERVES `load_sessions`
    # ORDER on purpose -- that list is NOT sorted, and the pools below are iterated into a
    # seeded RNG, so quietly sorting here would move published CIs.
    labels = ak.curated_labels(a.animals)
    # `fan_sessions` IS `fan_out` PLUS THE SORT -- see `analysis_kit`. `fan_out`
    # returns COMPLETION order; every bootstrap pool below is built by iterating this
    # collection, and a seeded RNG over a differently-ordered list gives different
    # draws (measured on `quit_prodrome`: the CI moved from [-22.9,-13.7] to
    # [-23.1,-13.6] while the point estimate stayed exact). No longer forgettable.
    res, fail = ak.fan_sessions(labels, session_latency_row, jobs=a.jobs)
    rows = [r for _lab, r in res if r]
    for r in rows:
        print(f"   {r['label']:14s} {r['epoch']:9s} {r['n_cue']:4d} cues  "
              f"470 peak {r['t_peak_470']:+.2f}s ({r['amp_peak_470']:+.2f}%)  "
              f"415 dip {r['t_dip_415']:+.2f}s ({r['amp_dip_415']:+.2f}%)  "
              f"latency {r['latency']:+.2f}s"
              + ("" if r["dip_is_negative"] else "   NO NEGATIVE DIP"), flush=True)
    n_none = sum(1 for x in res if not x[1])
    if n_none:
        print(f"  .. {n_none} session(s) had too few engaged cues -- skipped", flush=True)
    if fail:
        print(f"  !! {len(fail)} session(s) failed: "
              + ", ".join(f"{x[0]} ({x[1][:40]})" for x in fail[:4]), flush=True)

    if not rows:
        print("no sessions -- a failed run, not a result")
        return 1
    q = out_dir / "epoch_22_evoked_hrf_latency.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {q}")

    rng = np.random.default_rng(a.seed)
    eps = [e for e in ("pre", "acute", "subacute", "chronic") if any(r["epoch"] == e for r in rows)]
    bar = "=" * 96

    for only_neg in (False, True):
        use = [r for r in rows if r["dip_is_negative"]] if only_neg else rows
        tag = "SESSIONS WITH A GENUINE NEGATIVE DIP ONLY" if only_neg else "ALL SESSIONS"
        print(f"\n{bar}\n{tag}  (n={len(use)} of {len(rows)})\n{bar}")
        print(f"  {'epoch':<10}{'n':>4}{'470 peak (s)':>24}{'415 dip (s)':>24}{'latency (s)':>24}")
        for e in eps:
            line = f"  {e:<10}{sum(1 for r in use if r['epoch'] == e):>4}"
            for key in ("t_peak_470", "t_dip_415", "latency"):
                d = defaultdict(list)
                for r in use:
                    if r["epoch"] == e:
                        d[r["animal"]].append(float(r[key]))
                g = ak.boot_ci(d, rng)
                line += f"{g[0]:>11.3f} [{g[1]:+.2f},{g[2]:+.2f}]" if g else f"{'--':>24}"
            print(line)

        print(f"\n  CHANGE FROM PRE, paired within animal")
        for e in [x for x in eps if x != "pre"]:
            line = f"  {e:<10}{'':>4}"
            for key in ("t_peak_470", "t_dip_415", "latency"):
                post, pre_ = defaultdict(list), defaultdict(list)
                for r in use:
                    (post if r["epoch"] == e else pre_)[r["animal"]].append(float(r[key])) \
                        if r["epoch"] in (e, "pre") else None
                g = ak.boot_delta(post, pre_, rng)
                star = " *" if g and (g[1] > 0 or g[2] < 0) else "  "
                line += (f"{g[0]:>+11.3f} [{g[1]:+.2f},{g[2]:+.2f}]{star}" if g
                         else f"{'--':>24}")
            print(line)

    n_noneg = defaultdict(int)
    for r in rows:
        if not r["dip_is_negative"]:
            n_noneg[r["animal"]] += 1
    print(f"\n  SESSIONS WITH NO NEGATIVE DIP, by animal: {dict(n_noneg) or 'none'}")
    print(f"  SESSIONS WITH THE DIP AT THE WINDOW EDGE ({DIP_WIN[1]} s): "
          f"{sum(1 for r in rows if r['dip_at_edge'])}")
    print("\n  THE LATENCY IS 415-dip MINUS 470-peak, both from the SAME trials, so a shift in")
    print("  when the animal responds cannot move it. Compare against `rest_coupling`'s lag,")
    print("  which uses REST frames and shares no trials with this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
