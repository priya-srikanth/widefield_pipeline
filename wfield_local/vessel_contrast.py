"""Do the surface vessels get fainter after the stroke? A gain-invariant measure from the mean image.

Priya, 2026-08-18: *"Another casual observation - the vessels look significantly smaller/fainter
post-stroke."*

WHY THIS IS WORTH MEASURING RATHER THAN NOTING. Vessels appear DARK because haemoglobin absorbs, so
their contrast against the surrounding cortex is a direct optical readout of how much blood sits in
the light path. Fainter vessels mean less absorption, i.e. less blood. That makes this the one
observation in this line that speaks to PERFUSION directly -- and it is independent of the evoked
argument that forced the retraction in `hemispheric_intensity`, where the raw 415 signal turned out to
RISE with activation and left the perfusion direction unresolved. Vessel contrast does not depend on
knowing that sign.

THE MEASURE. Vessels are dark, thin and high-spatial-frequency. Subtracting a blurred copy isolates
them, and only the NEGATIVE deviations are kept, because a vessel is darker than its surround while a
bright specular highlight is not a vessel:

    contrast = median( max(blur(I) - I, 0) ) / median(I)      within a hemisphere mask

Dividing by the local median makes it invariant to LED power, exposure and camera gain -- the
confound that forces every cross-day comparison in this project onto ratios. A Frangi vesselness
filter is computed alongside as an independent estimator with different failure modes; where the two
disagree, neither should be trusted.

WHAT WOULD FOOL IT, AND WHY BOTH HEMISPHERES ARE REPORTED. Focus drift, a clouding window and a
changed working distance all reduce apparent vessel contrast, and all of them do it BILATERALLY. So a
bilateral drop is uninformative -- it is the optics. The interpretable quantity is the L/R RATIO,
which a symmetric optical change leaves alone. That is why the ratio is the headline and the absolute
values are shown beside it rather than instead of it.

BOTH CHANNELS. Haemoglobin absorbs far more strongly at 415 nm than at 470, so vessel contrast should
be higher at 415 in every session. If it is not, the channel assignment or the image is wrong, and
that check is run rather than assumed.

CLI::

    python -m wfield_local.vessel_contrast
    python -m wfield_local.vessel_contrast --animals PS94 PS95
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from wfield_local import config                                    # noqa: E402
from wfield_local.hemispheric_intensity import (CHANNELS, _session_dirs,  # noqa: E402
                                                hemisphere_masks)
from wfield_local.paths import PathResolver                        # noqa: E402

BLUR_SIGMA = 6.0            # px; larger than a vessel, smaller than a cortical area
MIN_PX = 200


def _dark_contrast(img, mask, sigma=BLUR_SIGMA):
    """Median depth of DARK structure relative to a blurred background, as a fraction of the median.

    Only negative deviations count: a vessel is darker than its surround, whereas a specular highlight
    or a bright edge artefact is not a vessel and would inflate a symmetric measure.
    """
    im = img.astype(float)
    bg = gaussian_filter(im, sigma)
    dark = np.clip(bg - im, 0, None)
    med = float(np.median(im[mask]))
    if med <= 0:
        return float("nan")
    # A HIGH PERCENTILE, not the median. Vessels are a MINORITY of pixels, so the median of a
    # mostly-zero dark-deviation map is exactly 0 -- the first version returned 0.0 for whole
    # hemispheres and an L/R ratio of 0/0. The 90th percentile measures how deep the prominent dark
    # structure is, which is what "fainter vessels" means; the mean is returned beside it because a
    # percentile alone cannot distinguish fewer vessels from shallower ones.
    d = dark[mask]
    return float(np.percentile(d, 90) / med)


def _dark_mean(img, mask, sigma=BLUR_SIGMA):
    """Total dark-structure energy as a fraction of the median: falls if vessels get FEWER as well as
    fainter, where the 90th percentile only falls if they get shallower."""
    im = img.astype(float)
    bg = gaussian_filter(im, sigma)
    dark = np.clip(bg - im, 0, None)
    med = float(np.median(im[mask]))
    if med <= 0:
        return float("nan")
    return float(dark[mask].mean() / med)


def _frangi_score(img, mask, sigma=BLUR_SIGMA):
    """Frangi vesselness on the INVERTED image (vessels are dark), median within the mask.

    An independent estimator with different failure modes from the blur-difference: it responds to
    tubular structure specifically rather than to any dark blob, but it is more sensitive to noise and
    to the sigma range. Agreement between the two is the check; disagreement means neither is safe.
    """
    try:
        from skimage.filters import frangi
    except Exception:                                              # noqa: BLE001
        return float("nan")
    im = img.astype(float)
    im = (im - im.min()) / max(float(np.ptp(im)), 1e-9)   # ndarray.ptp gone in numpy 2
    v = frangi(1.0 - im, sigmas=range(1, 5), black_ridges=False)
    return float(np.median(v[mask]))


def session_vessels(s):
    fa_path, allen_dir = _session_dirs(s)
    if not fa_path.exists() or not allen_dir.exists():
        return None
    favg = np.load(fa_path)
    left, right, _groups = hemisphere_masks(allen_dir)
    if favg.shape[1:] != left.shape or left.sum() < MIN_PX or right.sum() < MIN_PX:
        return None
    out = {"label": s["label"], "animal": s["label"][:4], "date": s["label"].split("_")[-1]}
    for ch, nm in CHANNELS.items():
        for side, m in (("left", left), ("right", right)):
            out[f"dark_{nm}_{side}"] = _dark_contrast(favg[ch], m)
            out[f"darkmean_{nm}_{side}"] = _dark_mean(favg[ch], m)
            out[f"frangi_{nm}_{side}"] = _frangi_score(favg[ch], m)
        for est in ("dark", "darkmean", "frangi"):
            l, r = out[f"{est}_{nm}_left"], out[f"{est}_{nm}_right"]
            out[f"{est}_{nm}_LR"] = l / r if r else float("nan")
            out[f"{est}_{nm}_mean"] = float(np.mean([l, r]))
    # 415 must show MORE vessel contrast than 470 -- haemoglobin absorbs far more strongly there.
    # A violation means the channels are swapped or the image is not what we think it is.
    out["sanity_415_gt_470"] = bool(out["dark_415_mean"] > out["dark_470_mean"])
    return out


KEYS = [("dark_415_LR", "415 vessel DEPTH (p90)  L/R"),
        ("darkmean_415_LR", "415 vessel ENERGY (mean)  L/R"),
        ("dark_415_mean", "415 vessel depth  (both hemispheres)"),
        ("frangi_415_LR", "Frangi vesselness  L/R (415)")]


def collect(animals=None):
    sessions = config.load_sessions()
    if animals:
        sessions = [x for x in sessions if x["label"][:4] in set(animals)]
    rows = []
    for s in sessions:
        try:
            r = session_vessels(s)
        except Exception as ex:                                    # noqa: BLE001
            print(f"  {s['label']}: skip ({str(ex)[:60]})", flush=True)
            continue
        if r is None:
            continue
        r["phase"] = config.session_phase(r["animal"], r["date"])
        rows.append(r)
        print(f"  {r['label']}  415 dark L/R {r['dark_415_LR']:.3f}  "
              f"mean {r['dark_415_mean']:.4f}  415>470 {r['sanity_415_gt_470']}", flush=True)
    return sorted(rows, key=lambda r: (r["animal"], r["date"]))


def summarise(rows):
    out = {}
    for a in sorted({r["animal"] for r in rows}):
        pre = [r for r in rows if r["animal"] == a and r["phase"] == "pre"]
        post = [r for r in rows if r["animal"] == a and r["phase"] == "post"]
        excl = [r for r in rows if r["animal"] == a and r["phase"] == "excluded"]
        if len(pre) < 3:
            out[a] = {"note": f"only {len(pre)} pre-stroke sessions"}
            continue
        rec = {"n_pre": len(pre)}
        for key, _lab in KEYS:
            v = np.array([r.get(key, np.nan) for r in pre], float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            band = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max())}
            rec[key] = {"pre_band": band,
                        "post": {r["date"]: float(r.get(key, np.nan)) for r in post},
                        "small_lesion": {r["date"]: float(r.get(key, np.nan)) for r in excl}}
            for r in post:
                x = r.get(key, np.nan)
                rec[key].setdefault("post_z", {})[r["date"]] = (
                    float((x - band["mean"]) / band["sd"]) if band["sd"] else float("nan"))
                rec[key].setdefault("outside_pre_range", {})[r["date"]] = bool(
                    x < band["min"] or x > band["max"])
        out[a] = rec
    return out


def plot(rows, out_dir):
    animals = sorted({r["animal"] for r in rows})
    fig, axes = plt.subplots(len(KEYS), len(animals), figsize=(4.2 * len(animals), 3.0 * len(KEYS)),
                             squeeze=False)
    colours = {"pre": "tab:blue", "post": "tab:red", "excluded": "grey"}
    for c, a in enumerate(animals):
        sub = [r for r in rows if r["animal"] == a]
        dates, phases = [r["date"] for r in sub], [r["phase"] for r in sub]
        x = np.arange(len(sub))
        first_non_pre = next((i for i, p in enumerate(phases) if p != "pre"), None)
        for k, (key, lab) in enumerate(KEYS):
            ax = axes[k][c]
            y = np.array([r.get(key, np.nan) for r in sub], float)
            m = np.array([p == "pre" for p in phases])
            if m.sum() >= 3 and np.isfinite(y[m]).any():
                ax.axhspan(np.nanmin(y[m]), np.nanmax(y[m]), color="tab:blue", alpha=0.15)
                ax.axhline(np.nanmean(y[m]), color="tab:blue", lw=1.2)
            for xi, yi, ph in zip(x, y, phases):
                ax.plot(xi, yi, "o", ms=7, color=colours.get(ph, "k"), markeredgecolor="k", lw=0.4,
                        zorder=3)
            ax.plot(x, y, "-", color="k", lw=0.7, alpha=0.4, zorder=1)
            if first_non_pre not in (None, 0):
                ax.axvline(first_non_pre - 0.5, color="firebrick", ls="--", lw=1.6)
            ax.set_xticks(x)
            ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7.5)
            if k == 0:
                ax.set_title(a, fontsize=11, fontweight="bold")
    fig.suptitle(
        "SURFACE VESSEL CONTRAST across days. Vessels are DARK because haemoglobin absorbs, so their "
        "contrast is an optical readout of blood in the light path: fainter vessels = less blood. "
        "Measured as the median depth of dark structure against a blurred background, divided by the "
        "median so it is invariant to LED power, exposure and gain. FOCUS DRIFT AND A CLOUDING WINDOW "
        "REDUCE IT BILATERALLY, so the L/R RATIO is the interpretable row and the two-hemisphere mean "
        "is shown only to reveal such a global change. GREY = the small-lesion animals (PS92/PS93 "
        "8/17), which were lesioned too and are not a no-lesion control.", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out_dir) / "vessel_contrast.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    rows = collect(args.animals)
    if not rows:
        print("[vessel_contrast] no usable sessions")
        return 0
    bad = [r["label"] for r in rows if not r.get("sanity_415_gt_470")]
    if bad:
        print(f"\n[vessel_contrast] WARNING: 415 vessel contrast is NOT above 470 in {len(bad)} "
              f"session(s) -- haemoglobin absorbs far more at 415, so check the channel assignment "
              f"before reading anything below: {bad[:6]}")
    summ = summarise(rows)
    print(f"\n=== summary ({len(rows)} sessions) ===")
    for a, rec in summ.items():
        if "note" in rec:
            print(f"  {a}: {rec['note']}")
            continue
        for key, lab in KEYS:
            if key not in rec:
                continue
            b = rec[key]["pre_band"]
            ps = "  ".join(
                f"{d}={v:.4f} (z={rec[key]['post_z'].get(d, float('nan')):+.1f}"
                f"{', OUTSIDE' if rec[key]['outside_pre_range'].get(d) else ''})"
                for d, v in rec[key]["post"].items()) or "no post-stroke session"
            sl = "  ".join(f"{d}={v:.4f}" for d, v in rec[key]["small_lesion"].items())
            print(f"  {a} {lab:42s} pre {b['mean']:.4f} +- {b['sd']:.4f} "
                  f"[{b['min']:.4f}-{b['max']:.4f}]   POST {ps}"
                  + (f"   SMALL-LESION {sl}" if sl else ""))
    json.dump({"summary": summ, "sessions": rows},
              open(Path(out) / "vessel_contrast.json", "w"), indent=1, default=float)
    print(f"  wrote {plot(rows, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
