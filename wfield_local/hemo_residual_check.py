"""Does the hemodynamic correction still work under each drift-removal variant? (Priya, 2026-08-13)

The variants in :mod:`wfield_local.hemo_variants` change only how slow DRIFT is removed before the
470-vs-415 regression -- none of them touch the regression itself, which is applied sample-by-sample at
31.23 Hz per channel. So in principle the FAST hemodynamic correction (heartbeat ~10 Hz, breathing
~3 Hz, vasomotion ~0.1-0.3 Hz) is untouched. This module checks that empirically instead of asserting
it, because "in principle" has been wrong repeatedly on this problem.

THE TEST. Take the global brain-mean trace of each signal, bandpass into physiological bands, and ask
how much of the 415 (isosbestic, calcium-free, therefore hemodynamic + noise) survives in the output:

    before   |r| between the UNCORRECTED 470 and the 415        -- how much there was to remove
    after    |r| between the CORRECTED SVTcorr and the 415      -- how much is left

A working correction drives `after` toward zero in every band. The band that answers Priya's question
is 1-14 Hz: if `after` stays low there under a variant, that variant has kept the fast-kinetics
correction. A variant that only fixed slow drift would show a low residual at <0.1 Hz and a HIGH one at
1-14 Hz.

CAVEAT ON INTERPRETATION. The 415 channel is dimmer, so at high frequency much of its power is shot
noise, which is uncorrelated with anything and drags `before` down on its own. Read the BEFORE->AFTER
reduction within a band, never the absolute value across bands.

    python -m wfield_local.hemo_residual_check --sessions PS95_0810 --variants zerophase,strobedetrend
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
from scipy.signal import butter, filtfilt

from wfield_local import hemo_variants as hv
from wfield_local.locanmf_cue_lick_analysis import SESSIONS

FS = hv.FS
BANDS = [(0.01, 0.1, "<0.1Hz drift/vaso"), (0.1, 1.0, "0.1-1Hz vasomotion"),
         (1.0, 5.0, "1-5Hz breathing"), (5.0, 13.0, "5-13Hz heartbeat")]


def _band(x, lo, hi):
    b, a = butter(2, [lo / (FS / 2), min(hi, FS / 2 - 0.01) / (FS / 2)], btype="bandpass")
    return filtfilt(b, a, x, padlen=50)


def _global_trace(U_atlas, mask, V):
    """Spatial mean over the brain mask, contracted through the SVD basis (never forms npix x T)."""
    Uf = np.nan_to_num(np.asarray(U_atlas).reshape(-1, np.asarray(U_atlas).shape[-1]))
    return (Uf[mask].mean(0)) @ V


def check(lab, variants, verbose=True):
    s = next((x for x in SESSIONS if x["label"] == lab), None)
    if s is None:
        return None
    res = s["mc"] + "/wfield_local_results"
    ad = glob.glob(res + "/allen_aligned_affine8v1")
    if not ad:
        return None
    U = np.load(ad[0] + "/U_atlas.npy")
    mask = np.load(ad[0] + "/allen_brain_mask_native_grid.npy").astype(bool).reshape(-1)
    svt = np.load(res + "/SVT.npy")
    raw470 = _global_trace(U, mask, svt[:, hv.FUNC::2].astype(np.float64))
    raw415 = _global_trace(U, mask, svt[:, (hv.FUNC + 1) % 2::2].astype(np.float64))
    n = min(raw470.size, raw415.size)
    raw470, raw415 = raw470[:n], raw415[:n]

    rows = {}
    if verbose:
        print(f"\n=== {lab}", flush=True)
        print(f"  {'variant':15s} " + "  ".join(f"{d:>20s}" for _a, _b, d in BANDS), flush=True)
        print(f"  {'(uncorrected)':15s} " + "  ".join(
            f"{abs(np.corrcoef(_band(raw470, a, b), _band(raw415, a, b))[0,1]):>20.3f}"
            for a, b, _d in BANDS), flush=True)
    for v in variants:
        svtc, _T, _rc, _m = hv.compute(s, v, refit_t=True, verbose=False)
        cor = _global_trace(U, mask, svtc.astype(np.float64))[:n]
        del svtc
        vals = []
        for a, b, _d in BANDS:
            before = abs(np.corrcoef(_band(raw470, a, b), _band(raw415, a, b))[0, 1])
            after = abs(np.corrcoef(_band(cor, a, b), _band(raw415, a, b))[0, 1])
            vals.append((before, after))
        rows[v] = vals
        if verbose:
            print(f"  {v:15s} " + "  ".join(f"{af:>20.3f}" for _bf, af in vals), flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+",
                    default=["PS92_0810", "PS93_0810", "PS94_0810", "PS95_0810"])
    ap.add_argument("--variants", default="zerophase,fitonly,strobedetrend,meegkit")
    a = ap.parse_args(argv)
    variants = a.variants.split(",")

    allrows = {}
    for lab in a.sessions:
        r = check(lab, variants)
        if r:
            for v, vals in r.items():
                allrows.setdefault(v, []).append(vals)

    print(f"\n=== |r| with the 415 channel, mean over {len(a.sessions)} sessions ===")
    print("  (lower = more hemodynamic signal removed. Compare WITHIN a band, not across:")
    print("   the 415 is dimmer so its high-frequency power is largely shot noise.)")
    print(f"  {'variant':15s} " + "  ".join(f"{d:>20s}" for _a, _b, d in BANDS))
    if allrows:
        first = next(iter(allrows.values()))
        bef = np.array(first)[:, :, 0].mean(0)
        print(f"  {'UNCORRECTED':15s} " + "  ".join(f"{x:>20.3f}" for x in bef))
    for v, lst in allrows.items():
        aft = np.array(lst)[:, :, 1].mean(0)
        print(f"  {v:15s} " + "  ".join(f"{x:>20.3f}" for x in aft))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
