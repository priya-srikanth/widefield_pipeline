"""IS THE CUE-EVOKED 415 RESPONSE VESSEL-SHAPED OR PARENCHYMA-SHAPED? The decisive test.

Priya, 2026-09-19: *"one argument is that I can clearly see the cortical veins and sinuses as
increased in 415 nm aligned to cue"* -- against my claim that blood, which absorbs, must DECREASE
415 fluorescence.

**THIS IS THE OBSERVATION THAT SETTLES IT, AND IT IS SPATIAL RATHER THAN TEMPORAL.** The
time-course version (`channel_evoked_sign`) could not decide: 415 has a fast positive component in
every animal, and whether the late deflection goes negative differs by animal. But the two
candidate sources have completely different SPATIAL signatures, and the vessels are where they
separate:

    calcium bleed-through  ->  PARENCHYMA-shaped, matching the 470 activation map
    haemodynamic           ->  VESSEL-shaped, concentrated on veins and sinuses

and the SIGN at the vessels then says which way blood moves the signal. A POSITIVE vessel-shaped
415 response means blood INCREASES 415 fluorescence and my absorption argument is wrong -- which
would invalidate reading the TROUGH in `rest_coupling._hrf_lag`, since that choice rests entirely
on blood entering 415 negatively.

VESSELS ARE DEFINED FROM THE MEAN IMAGE, not from an atlas. Haemoglobin absorbs, so vessels are
DARK in `frames_average_atlas` whatever happens dynamically -- a static fact that does not assume
the answer to the dynamic question being asked.

**BUT "DARK" MUST MEAN LOCALLY DARK, AND THE FIRST VERSION OF THIS GOT IT WRONG.** Thresholding
raw intensity at the darkest decile does NOT select vessels, because the illumination and the
preparation have a large slow spatial gradient that dwarfs the vessels: that mask came out at row
183 +/- 198 with a mean intensity of 2046 against the brain's 13048 -- the DIM ANTERIOR EDGE, six
times darker than the brain, and only 49% of it overlapped the real vasculature. It made the
vessel compartment look flat when the maps plainly show the sinus and the big cortical veins
moving, which is how the error was caught (Priya: *"there are clearly veins there"*).

`_vesselness` compares each pixel to a Gaussian-smoothed copy of its own neighbourhood, so a slow
gradient cancels and only LOCAL darkness survives. That mask lands at column 316 +/- 71 against a
brain centred at 319 +/- 136 -- tight on the midline, which is the superior sagittal sinus.

    python -m scripts.rest_migration.channel_vessel_sign [--sessions PS93_0607 ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PRE, POST = 1.0, 3.0
BASE = (-1.0, -0.2)
VESSEL_Q = 0.90          # top decile of LOCAL darkness (see `_vesselness`) = vessels
PAREN_Q = 0.50           # bottom half of local darkness = parenchyma
VESSEL_SIGMA = 8.0       # px; neighbourhood the local-contrast comparison is made against
LATENCIES = (0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5)


def _vesselness(mean_img, brain, sigma=VESSEL_SIGMA):
    """How much DARKER than its own neighbourhood each pixel is. High = vessel.

    A raw intensity threshold selects the dim periphery, not the vasculature -- see the module
    docstring. Dividing by the smoothed image makes this a relative contrast, so it does not
    care that the anterior cortex is globally dimmer than the posterior.
    """
    from scipy.ndimage import gaussian_filter

    m = np.where(brain, mean_img, np.nan)
    filled = np.where(brain, mean_img, float(np.nanmedian(m)))
    sm = gaussian_filter(filled, sigma)
    return (sm - filled) / np.maximum(sm, 1e-9)


def session_maps(s):
    """``(latencies, {channel: maps}, vessel_mask, paren_mask, brain_mask, vesselness, n_cue)``."""
    from wfield_local import config
    from wfield_local.hemo_variants import FS, functional_channel
    from wfield_local.rest_by_position import _session_daq

    res = Path(s["mc"]) / "wfield_local_results"
    allen = res / "allen_aligned_affine8v1"
    svt = np.load(res / "SVT.npy", mmap_mode="r")
    fc = functional_channel(s)
    d = {"470": np.asarray(svt[:, fc::2], np.float64),
         "415": np.asarray(svt[:, (fc + 1) % 2::2], np.float64),
         "SVTcorr": np.asarray(np.load(config.svtcorr_in(res), mmap_mode="r"), np.float64)}
    U = np.asarray(np.load(allen / "U_atlas.npy", mmap_mode="r"), np.float64)   # (H, W, k)
    brain = np.asarray(np.load(allen / "allen_brain_mask_native_grid.npy")).astype(bool)
    favg = np.asarray(np.load(allen / "frames_average_atlas.npy"), np.float64)
    mean_img = favg[fc] if favg.ndim == 3 else favg
    _rest, cs, _codes, _ts, fs_samp, _sy = _session_daq(s)

    f_of = np.asarray(fs_samp)
    n = min(min(v.shape[1] for v in d.values()), f_of.size)
    a, b = int(round(PRE * FS)), int(round(POST * FS))
    cue_f = np.searchsorted(f_of[:n], np.asarray(cs, np.int64))
    ok = cue_f[(cue_f > a) & (cue_f < n - b)]
    if ok.size == 0:
        return None

    vness = _vesselness(mean_img, brain)
    vessel = brain & (vness >= np.quantile(vness[brain], VESSEL_Q))
    paren = brain & (vness <= np.quantile(vness[brain], PAREN_Q))

    t = np.arange(-a, b) / FS
    jb = (t >= BASE[0]) & (t <= BASE[1])
    out = {}
    for k, v in d.items():
        seg = np.stack([v[:, f - a:f + b] for f in ok]).mean(0)      # (k, n_t) cue-average in SVT
        seg = seg - seg[:, jb].mean(1, keepdims=True)
        js = [int(np.argmin(np.abs(t - x))) for x in LATENCIES]
        out[k] = np.stack([U @ seg[:, j] for j in js]) * 100.0       # (n_lat, H, W) per cent
    return np.array(LATENCIES), out, vessel, paren, brain, vness, ok.size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+",
                    default=["PS92_0828", "PS93_0607", "PS94_0820"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import config
    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    bar = "=" * 100
    got = {}
    for lab in a.sessions:
        s = next((x for x in config.load_sessions() if x["label"] == lab), None)
        if s is None:
            print(f"  !! {lab}: not in sessions.yaml")
            continue
        try:
            r = session_maps(s)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if r is None:
            continue
        got[lab] = r
        lat, mp, vess, par, brain, vness, ncue = r
        print(f"\n{bar}\n{lab}   {ncue} cues   vessel px {int(vess.sum())}  "
              f"parenchyma px {int(par.sum())}\n{bar}")
        print(f"  {'latency (s)':<14}" + "".join(f"{x:>8.1f}" for x in lat))
        for k in ("470", "415", "SVTcorr"):
            for nm, m in (("vessel", vess), ("parench", par)):
                print(f"  {k + ' ' + nm:<14}"
                      + "".join(f"{mp[k][i][m].mean():>8.2f}" for i in range(lat.size)))
        # THE DISCRIMINATOR: does the 415 map look like the 470 activation map, or like the
        # vessels? `vness` is high where the tissue is LOCALLY dark, i.e. on the vessels.
        r470 = [float(np.corrcoef(mp["415"][i][brain], mp["470"][i][brain])[0, 1])
                for i in range(lat.size)]
        rves = [float(np.corrcoef(mp["415"][i][brain], vness[brain])[0, 1])
                for i in range(lat.size)]
        print(f"  {'r(415, 470)':<14}" + "".join(f"{x:>8.2f}" for x in r470))
        print(f"  {'r(415, vessel)':<14}" + "".join(f"{x:>8.2f}" for x in rves))

    if not got:
        print("no sessions -- a failed run, not a result")
        return 1

    nlat = len(LATENCIES)
    fig, axes = plt.subplots(2 * len(got), nlat, figsize=(1.7 * nlat, 3.4 * len(got)),
                             squeeze=False)
    for gi, (lab, (lat, mp, vess, par, brain, vness, ncue)) in enumerate(got.items()):
        for ci, k in enumerate(("470", "415")):
            # ONE SHARED SCALE PER CHANNEL PER SESSION, computed INSIDE the brain mask -- the
            # registered edge of U_atlas holds extreme values and an all-pixel percentile
            # inflates some panels and not others (DECISIONS, the colour-scale bug).
            v = np.nanpercentile(np.abs(np.stack([mp[k][i][brain] for i in range(nlat)])), 99)
            for i in range(nlat):
                ax = axes[2 * gi + ci][i]
                img = np.where(brain, mp[k][i], np.nan)
                ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if ci == 0:
                    ax.set_title(f"{lat[i]:.1f} s", fontsize=8)
                if i == 0:
                    ax.set_ylabel(f"{lab}\n{k}  (+/-{v:.2f}%)", fontsize=7.5)
    fig.suptitle("CUE-ALIGNED MAPS. If the 415 row is PARENCHYMA-shaped and tracks the 470 row it "
                 "is calcium bleed-through;\nif it is VESSEL-shaped it is haemodynamic, and its "
                 "SIGN at the vessels says which way blood moves 415.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    q = out_dir / "epoch_21_channel_vessel_sign.png"
    fig.savefig(q, dpi=170)
    print(f"\nwrote {q}")
    print("\n  r(415, 470) HIGH and r(415, vessel) LOW -> calcium bleed-through.")
    print("  r(415, vessel) HIGH and POSITIVE vessel values -> blood INCREASES 415, and the")
    print("  trough reading in `rest_coupling` is WRONG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
