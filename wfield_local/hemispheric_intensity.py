"""LEFT-vs-RIGHT raw fluorescence, per channel, across days — does the lesioned hemisphere change?

Priya, 2026-08-18: "Anecdotally I feel like post-stroke there is higher GCaMP 470 nm signal in the L
hemisphere, especially parietally (maybe around SS?) — it would be interesting to look at changes in
hemispheric activity relative to pre-stroke, as well as changes in 415 nm hemodynamic signal (is
there evidence of L hemisphere hypoperfusion after striatal stroke?)."

THE TWO QUESTIONS ARE NOT INDEPENDENT, AND THAT IS THE POINT OF THIS MODULE.
415 nm is the GCaMP isosbestic: fluorescence there is insensitive to calcium and is dominated by
HAEMOGLOBIN ABSORPTION. Less blood in the light path means less absorption, so hypoperfusion RAISES
the raw counts at 415 -- and it raises them at 470 too, because the same blood absorbs both. So a
left-sided rise in 470 is exactly what hypoperfusion predicts WITH NO CHANGE IN NEURAL ACTIVITY. The
anecdotal observation may BE the hypoperfusion rather than evidence of activity beside it.

Separating them needs the ratio of ratios:

    R_470 = median(470, LEFT) / median(470, RIGHT)      optical + neural
    R_415 = median(415, LEFT) / median(415, RIGHT)      optical only (no calcium sensitivity)
    G     = R_470 / R_415                               GCaMP-specific, absorption divided out

  * R_415 rising post-stroke  -> left hemisphere is less absorbing == HYPOPERFUSED.
  * R_470 rising with G flat  -> the 470 rise is entirely optical. Not an activity change.
  * G rising                  -> a real left-sided GCaMP increase on top of whatever the blood did.

WHY RATIOS AND NOT ABSOLUTE COUNTS. `crossday_intensity` already tracks the absolute brain-ROI median
and warns on its own figure that LED power is titrated by hand day to day, so its trend may be the
LED setting. That confound is fatal for a cross-day claim about one hemisphere -- but it is COMMON to
both hemispheres within a session, so it cancels in a within-session L/R ratio. Exposure, gain, LED
drift and bleaching cancel with it. What does NOT cancel is anything spatially asymmetric: window
clarity, focus tilt, uneven illumination, or a headplate shift. Those are the real threats here, so
the pre-stroke sessions are the reference -- the question is never "is L/R != 1" (it never is) but
"did L/R MOVE relative to this animal's own pre-stroke range".

Regions come from the Allen atlas aligned to each session's native grid, so "parietal / SSp" is a
named area rather than a hand-drawn box.

CLI::

    python -m wfield_local.hemispheric_intensity                 # all curated + post-stroke sessions
    python -m wfield_local.hemispheric_intensity --animals PS94 PS95
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from wfield_local import config                                   # noqa: E402
from wfield_local.paths import PathResolver                       # noqa: E402

#: Region groups reported separately. "SSp" is Priya's specific hunch (parietal / somatosensory);
#: "all" is every atlas area, which is the number to read first because a whole-hemisphere shift and
#: a focal one mean different things.
REGION_GROUPS = {
    "all": None,
    "SSp": ("SSp",),                      # primary somatosensory -- the parietal hunch
    "MO": ("MOp", "MOs"),                 # motor, for contrast
    "VIS": ("VISp", "VISa", "VISal", "VISam", "VISl", "VISpm", "VISrl"),
}

CHANNELS = {0: 415, 1: 470}               # frames_average.npy channel order (crossday_intensity)


def _session_dirs(s):
    """(mean image, allen dir). The image is the ATLAS-grid one.

    `frames_average.npy` is on the camera grid (460x480) and the atlas is 540x640, so the two cannot
    be combined; `frames_average_atlas.npy` is the same mean image under the registration transform
    and is what the masks index.
    """
    mc = Path(s["mc"])
    allen = mc / "wfield_local_results" / "allen_aligned_affine8v1"
    return allen / "frames_average_atlas.npy", allen


def hemisphere_masks(allen_dir: Path):
    """(left_mask, right_mask, {group: (left, right)}) on the session's native grid.

    Hemisphere comes from the Allen area NAME suffix (_left / _right), the same rule
    `locanmf_rsa._session_hemi_compute` uses for components -- not from an image midline, which would
    be wrong whenever the headplate is even slightly rotated.
    """
    atlas = np.load(allen_dir / "allen_area_atlas_native_grid.npy")
    raw = json.load(open(allen_dir / "allen_area_names.json"))
    # list of [signed_code, name]; +code = _left, -code = _right
    names = [(int(c), str(n)) for c, n in raw]
    ai = np.rint(atlas).astype(np.int32)                 # stored float32; compare as integers

    left = np.zeros(ai.shape, bool)
    right = np.zeros(ai.shape, bool)
    per_group = {}
    for code, nm in names:
        if code == 0:
            continue
        side = "left" if nm.endswith("_left") else "right" if nm.endswith("_right") else None
        if side is None:
            continue
        # the sign convention and the name suffix must agree; if the exporter ever flips one, this
        # is where it surfaces rather than silently swapping the hemispheres in every result
        assert (code > 0) == (side == "left"), (
            f"allen_area_names.json: code {code} is signed {'+' if code > 0 else '-'} but named {nm}")
        m = ai == code
        if not m.any():
            continue
        (left if side == "left" else right)[m] = True

    for g, prefixes in REGION_GROUPS.items():
        if prefixes is None:
            per_group[g] = (left, right)
            continue
        gl = np.zeros(ai.shape, bool)
        gr = np.zeros(ai.shape, bool)
        for code, nm in names:
            base = nm.rsplit("_", 1)[0]
            if not any(base == p or base.startswith(p) for p in prefixes):
                continue
            m = ai == code
            (gl if code > 0 else gr)[m] = True
        per_group[g] = (gl, gr)
    return left, right, per_group


def session_ratios(s, min_px=200):
    """Per-channel L/R median-count ratios for one session, per region group.

    Medians, not means: a bright vessel or a specular highlight moves a mean and barely moves a
    median, and those are exactly the asymmetric artefacts this analysis cannot otherwise control.
    """
    fa_path, allen_dir = _session_dirs(s)
    if not fa_path.exists() or not allen_dir.exists():
        return None
    favg = np.load(fa_path)                                   # (2, H, W): ch0 = 415, ch1 = 470
    if favg.ndim != 3 or favg.shape[0] < 2:
        return None
    _l, _r, groups = hemisphere_masks(allen_dir)
    if favg.shape[1:] != _l.shape:
        raise ValueError(f"{s['label']}: mean image {favg.shape[1:]} does not match atlas "
                         f"{_l.shape} -- wrong grid")
    out = {"label": s["label"], "animal": s["label"][:4], "date": s["label"].split("_")[-1]}
    for g, (ml, mr) in groups.items():
        if ml.sum() < min_px or mr.sum() < min_px:
            out[g] = None
            continue
        rec = {"n_px_left": int(ml.sum()), "n_px_right": int(mr.sum())}
        for ch, nm in CHANNELS.items():
            lv = float(np.median(favg[ch][ml]))
            rv = float(np.median(favg[ch][mr]))
            rec[f"left_{nm}"] = lv
            rec[f"right_{nm}"] = rv
            rec[f"ratio_{nm}"] = lv / rv if rv else float("nan")
        # GCaMP-specific: divide the 470 asymmetry by the 415 asymmetry, removing the part of the
        # 470 change that any absorption difference would have produced on its own.
        r470, r415 = rec.get("ratio_470", float("nan")), rec.get("ratio_415", float("nan"))
        rec["ratio_gcamp_specific"] = r470 / r415 if r415 else float("nan")
        out[g] = rec
    return out


def collect(animals=None):
    sessions = config.load_sessions()
    if animals:
        sessions = [s for s in sessions if s["label"][:4] in set(animals)]
    rows = []
    for s in sessions:
        try:
            r = session_ratios(s)
        except Exception as ex:                                   # noqa: BLE001
            print(f"  {s['label']}: skip ({str(ex)[:60]})", flush=True)
            continue
        if r is None:
            continue
        r["phase"] = config.session_phase(r["animal"], r["date"])
        rows.append(r)
    return sorted(rows, key=lambda r: (r["animal"], r["date"]))


def summarise(rows, group="all"):
    """Post-stroke ratio against that animal's own PRE-stroke range, per channel.

    The comparison is always within-animal: L/R is never 1.0 even in a healthy mouse (illumination
    and window clarity are not symmetric), so an absolute value carries no information and only a
    MOVE relative to the animal's own baseline does.
    """
    out = {}
    for a in sorted({r["animal"] for r in rows}):
        pre = [r[group] for r in rows if r["animal"] == a and r["phase"] == "pre" and r.get(group)]
        post = [(r["date"], r[group]) for r in rows
                if r["animal"] == a and r["phase"] == "post" and r.get(group)]
        excl = [(r["date"], r[group]) for r in rows
                if r["animal"] == a and r["phase"] == "excluded" and r.get(group)]
        if len(pre) < 3:
            out[a] = {"note": f"only {len(pre)} pre-stroke sessions"}
            continue
        rec = {"n_pre": len(pre)}
        for key in ("ratio_415", "ratio_470", "ratio_gcamp_specific"):
            v = np.array([p[key] for p in pre], float)
            v = v[np.isfinite(v)]
            band = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max())}
            rec[key] = {"pre_band": band,
                        "post": {d: float(p[key]) for d, p in post},
                        "excluded": {d: float(p[key]) for d, p in excl}}
            for d, p in post:
                z = (p[key] - band["mean"]) / band["sd"] if band["sd"] else float("nan")
                rec[key].setdefault("post_z", {})[d] = float(z)
                rec[key].setdefault("post_outside_pre_range", {})[d] = bool(
                    p[key] < band["min"] or p[key] > band["max"])
        out[a] = rec
    return out


def plot(rows, out_dir, group="all"):
    animals = sorted({r["animal"] for r in rows})
    fig, axes = plt.subplots(3, len(animals), figsize=(4.2 * len(animals), 9.0), squeeze=False,
                             sharex="col")
    keys = [("ratio_415", "415 nm  L/R  (absorption; UP = left less absorbing = HYPOperfused)"),
            ("ratio_470", "470 nm  L/R  (optical + neural, NOT interpretable alone)"),
            ("ratio_gcamp_specific", "470/415  L/R  (GCaMP-specific, absorption divided out)")]
    for c, a in enumerate(animals):
        sub = [r for r in rows if r["animal"] == a and r.get(group)]
        dates = [r["date"] for r in sub]
        x = np.arange(len(dates))
        phases = [r["phase"] for r in sub]
        first_non_pre = next((i for i, p in enumerate(phases) if p != "pre"), None)
        for k, (key, lab) in enumerate(keys):
            ax = axes[k][c]
            y = np.array([r[group][key] for r in sub], float)
            pre_m = np.array([p == "pre" for p in phases])
            if pre_m.sum() >= 3:
                mu, sd = np.nanmean(y[pre_m]), np.nanstd(y[pre_m], ddof=1)
                ax.axhspan(y[pre_m].min(), y[pre_m].max(), color="tab:blue", alpha=0.15,
                           label="pre-stroke range")
                ax.axhline(mu, color="tab:blue", lw=1.2)
            colours = {"pre": "tab:blue", "post": "tab:red", "excluded": "grey"}
            for i, (xi, yi, ph) in enumerate(zip(x, y, phases)):
                ax.plot(xi, yi, "o", ms=7, color=colours.get(ph, "k"),
                        markeredgecolor="k", lw=0.4, zorder=3)
            ax.plot(x, y, "-", color="k", lw=0.7, alpha=0.4, zorder=1)
            if first_non_pre not in (None, 0):
                ax.axvline(first_non_pre - 0.5, color="firebrick", ls="--", lw=1.6)
                ax.text(first_non_pre - 0.5, ax.get_ylim()[1], " LESION", color="firebrick",
                        fontsize=7, fontweight="bold", va="top")
            ax.axhline(1.0, color="k", ls=":", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=6.5)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7)
            if k == 0:
                ax.set_title(a, fontsize=11, fontweight="bold")
    fig.suptitle(
        f"LEFT/RIGHT raw fluorescence across days — region group '{group}'. Ratios, not absolute "
        "counts: LED power is titrated by hand day to day, and that cancels within a session but not "
        "across one. 415 nm is the GCaMP isosbestic, so it tracks HAEMOGLOBIN ABSORPTION — "
        "hypoperfusion raises it, and raises 470 with it. A left-sided 470 rise is therefore NOT "
        "evidence of activity unless the 470/415 ratio moves too. L/R is never 1.0 in a healthy "
        "mouse; only a MOVE from the animal's own pre-stroke range means anything.",
        fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out_dir) / f"hemispheric_intensity_{group}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--group", nargs="+", default=["all", "SSp"], choices=list(REGION_GROUPS))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    rows = collect(args.animals)
    if not rows:
        print("[hemispheric_intensity] no sessions with frames_average + allen alignment")
        return 0
    print(f"[hemispheric_intensity] {len(rows)} sessions")
    for g in args.group:
        s = summarise(rows, g)
        print(f"\n=== region group: {g} ===")
        for a, rec in s.items():
            if "note" in rec:
                print(f"  {a}: {rec['note']}")
                continue
            for key in ("ratio_415", "ratio_470", "ratio_gcamp_specific"):
                b = rec[key]["pre_band"]
                post = rec[key]["post"]
                z = rec[key].get("post_z", {})
                outside = rec[key].get("post_outside_pre_range", {})
                ps = "  ".join(f"{d}={v:.4f} (z={z.get(d, float('nan')):+.1f}"
                               f"{', OUTSIDE pre range' if outside.get(d) else ''})"
                               for d, v in post.items()) or "no post-stroke session"
                print(f"  {a} {key:22s} pre {b['mean']:.4f} +- {b['sd']:.4f} "
                      f"[{b['min']:.4f}-{b['max']:.4f}]   POST {ps}")
        json.dump({"group": g, "summary": s, "sessions": rows},
                  open(Path(out) / f"hemispheric_intensity_{g}.json", "w"), indent=1, default=float)
        print(f"  wrote {plot(rows, out, g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
