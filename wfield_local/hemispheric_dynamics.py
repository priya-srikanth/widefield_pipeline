"""Per-hemisphere DYNAMICS and cross-hemisphere COUPLING — the measures a mean image cannot give.

Priya, 2026-08-18: "Let's next look at per-hemisphere temporal SD; and if possible think about a
hemispheric 'concordance' measure — eg are areas that are usually coupled cross-hemisphere pre-stroke
now de-coupled post-stroke — either with ROI or locaNMF frozen regions."

WHY THIS AND NOT `hemispheric_intensity`. That module measures the session MEAN image: static baseline
fluorescence. An impression that one hemisphere is "more active" is about DYNAMICS, and a mean image is
exactly the statistic that throws dynamics away. These two measures put it back:

  TEMPORAL SD   per-hemisphere standard deviation of the corrected (SVTcorr) signal over the session,
                i.e. how much the signal actually moves. Reported as an L/R ratio for the same reason
                the intensity module does: LED power, exposure and gain are titrated by hand and
                cancel within a session, not across one.

  CONCORDANCE   correlation between HOMOTOPIC pairs -- the same Allen area in the two hemispheres.
                Those are strongly coupled in an intact cortex, largely through the corpus callosum,
                so a drop is a specific and interpretable claim: the hemispheres have decoupled. It is
                also robust in a way an amplitude measure is not, because a correlation does not care
                about the gain of either channel.

WHAT MAKES CONCORDANCE THE BETTER OF THE TWO. Temporal SD inherits every asymmetric optical confound
that the intensity ratio has (window clarity, focus tilt, uneven illumination) -- it is an amplitude,
and amplitudes are what those confounds move. A homotopic CORRELATION is invariant to per-hemisphere
scaling: dim the left hemisphere by any factor and its correlation with the right is unchanged. So if
the two disagree, believe concordance.

THE CONTROL THAT MAKES EITHER READABLE. Both are reported against the animal's own pre-stroke range,
and PS92/PS93 8/17 -- same surgery day, no deficit -- are analysed alongside as the negative control.
A post-stroke drop that also appears in PS92/PS93 is a property of 8/17, not of the lesion.

BASIS. Allen-ROI by default, because its regions are anatomically named and homotopic pairing is then
exact (`SSp_left` <-> `SSp_right`). The joint LocaNMF basis is available via ``--source joint``; its
components are functionally defined, so pairing goes through each component's dominant Allen area and
is approximate. Where the two disagree, the ROI answer is the conservative one.

CLI::

    python -m wfield_local.hemispheric_dynamics
    python -m wfield_local.hemispheric_dynamics --animals PS94 PS95 --source joint
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from wfield_local import config                                    # noqa: E402
from wfield_local.paths import PathResolver                        # noqa: E402

MIN_PAIR_FRAMES = 1000       # a session shorter than this cannot support a stable correlation


def _region_names(s):
    """{component index -> Allen area name} for this session's feature set."""
    import glob
    f = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")
    if not f:
        return None
    raw = json.load(open(f[0]))
    return {int(c): str(n) for c, n in raw}


def session_dynamics(s, source="roi"):
    """Per-hemisphere temporal SD and homotopic-pair correlations for one session.

    Uses the same feature construction as the decoders (`_build_signal`), so "the signal" here is the
    same object every other result in this deck is computed from, on the adopted hemodynamic variant.
    """
    from wfield_local.locanmf_position_decoder import _build_signal

    sig, feat_reg = _build_signal(s, source)          # (nfeat, T)
    names = _region_names(s)
    if names is None or sig.shape[1] < MIN_PAIR_FRAMES:
        return None
    rn = np.array([names.get(int(feat_reg[i]), "?") for i in range(sig.shape[0])])
    left = np.array([n.endswith("_left") for n in rn])
    right = np.array([n.endswith("_right") for n in rn])
    if left.sum() < 3 or right.sum() < 3:
        return None

    sd = sig.std(axis=1)
    out = {"label": s["label"], "animal": s["label"][:4], "date": s["label"].split("_")[-1],
           "source": source, "n_frames": int(sig.shape[1]),
           "sd_left": float(np.median(sd[left])), "sd_right": float(np.median(sd[right])),
           "n_left": int(left.sum()), "n_right": int(right.sum())}
    out["sd_ratio_LR"] = out["sd_left"] / out["sd_right"] if out["sd_right"] else float("nan")

    # ---- homotopic concordance: same area, opposite hemispheres
    by_area = defaultdict(dict)
    for i, n in enumerate(rn):
        if n.endswith("_left"):
            by_area[n[:-5]].setdefault("L", []).append(i)
        elif n.endswith("_right"):
            by_area[n[:-6]].setdefault("R", []).append(i)
    pairs = {}
    for area, d in by_area.items():
        if "L" not in d or "R" not in d:
            continue
        # a region may map to several components; average them so one pair is one number
        l = sig[d["L"]].mean(axis=0)
        r = sig[d["R"]].mean(axis=0)
        if l.std() == 0 or r.std() == 0:
            continue
        pairs[area] = float(np.corrcoef(l, r)[0, 1])
    if not pairs:
        return out
    v = np.array(list(pairs.values()), float)
    out["homotopic_r"] = pairs
    out["homotopic_r_median"] = float(np.median(v))
    out["n_pairs"] = int(len(v))

    # WITHIN-hemisphere coupling as the specificity control: if EVERYTHING decorrelates post-stroke
    # (movement, arousal, a noisier recording) then a homotopic drop says nothing about the callosum.
    # The interpretable result is homotopic falling while within-hemisphere holds.
    def _within(mask):
        idx = np.flatnonzero(mask)
        if len(idx) < 4:
            return float("nan")
        c = np.corrcoef(sig[idx])
        iu = np.triu_indices(len(idx), k=1)
        return float(np.median(c[iu]))

    out["within_left_r_median"] = _within(left)
    out["within_right_r_median"] = _within(right)
    out["homotopic_minus_within"] = float(
        out["homotopic_r_median"] - np.nanmean([out["within_left_r_median"],
                                                out["within_right_r_median"]]))
    return out


def collect(animals=None, source="roi"):
    sessions = config.load_sessions()
    if animals:
        sessions = [x for x in sessions if x["label"][:4] in set(animals)]
    rows = []
    for s in sessions:
        try:
            r = session_dynamics(s, source)
        except Exception as ex:                                    # noqa: BLE001
            print(f"  {s['label']}: skip ({str(ex)[:60]})", flush=True)
            continue
        if r is None:
            continue
        r["phase"] = config.session_phase(r["animal"], r["date"])
        rows.append(r)
        print(f"  {r['label']}  SD L/R {r.get('sd_ratio_LR', float('nan')):.3f}   "
              f"homotopic r {r.get('homotopic_r_median', float('nan')):.3f} "
              f"({r.get('n_pairs', 0)} pairs)", flush=True)
    return sorted(rows, key=lambda r: (r["animal"], r["date"]))


KEYS = [("sd_ratio_LR", "temporal SD  L/R"),
        ("homotopic_r_median", "homotopic r (cross-hemisphere coupling)"),
        ("homotopic_minus_within", "homotopic minus within-hemisphere r")]


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
                        "excluded_control": {r["date"]: float(r.get(key, np.nan)) for r in excl}}
            for r in post:
                x = r.get(key, np.nan)
                rec[key].setdefault("post_z", {})[r["date"]] = (
                    float((x - band["mean"]) / band["sd"]) if band["sd"] else float("nan"))
                rec[key].setdefault("post_outside_pre_range", {})[r["date"]] = bool(
                    x < band["min"] or x > band["max"])
        out[a] = rec
    return out


def plot(rows, out_dir, source="roi"):
    animals = sorted({r["animal"] for r in rows})
    fig, axes = plt.subplots(len(KEYS), len(animals), figsize=(4.2 * len(animals), 3.2 * len(KEYS)),
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
                ax.set_ylabel(lab, fontsize=8)
            if k == 0:
                ax.set_title(a, fontsize=11, fontweight="bold")
    fig.suptitle(
        f"Per-hemisphere DYNAMICS and cross-hemisphere COUPLING ({source} basis). What a mean image "
        "cannot show. GREY points = the excluded sessions (PS92/PS93 8/17, same surgery day, no "
        "deficit) which are the NEGATIVE CONTROL: a post-stroke drop that also appears in them is a "
        "property of 8/17, not the lesion. Homotopic r is invariant to per-hemisphere gain, so it "
        "survives the optical asymmetries that make the amplitude measures fragile — where the two "
        "disagree, believe the correlation. The third row is the specificity check: if EVERYTHING "
        "decorrelates, a homotopic drop says nothing about the callosum.",
        fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out_dir) / f"hemispheric_dynamics_{source}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--source", default="roi", choices=["roi", "locanmf", "joint"])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    rows = collect(args.animals, args.source)
    if not rows:
        print("[hemispheric_dynamics] no usable sessions")
        return 0
    summ = summarise(rows)
    print(f"\n=== summary ({args.source}) ===")
    for a, rec in summ.items():
        if "note" in rec:
            print(f"  {a}: {rec['note']}")
            continue
        for key, lab in KEYS:
            if key not in rec:
                continue
            b = rec[key]["pre_band"]
            ps = "  ".join(
                f"{d}={v:.3f} (z={rec[key]['post_z'].get(d, float('nan')):+.1f}"
                f"{', OUTSIDE' if rec[key]['post_outside_pre_range'].get(d) else ''})"
                for d, v in rec[key]["post"].items()) or "no post-stroke session"
            ctl = "  ".join(f"{d}={v:.3f}" for d, v in rec[key]["excluded_control"].items())
            print(f"  {a} {lab:42s} pre {b['mean']:.3f} +- {b['sd']:.3f} "
                  f"[{b['min']:.3f}-{b['max']:.3f}]   POST {ps}"
                  + (f"   CONTROL(excluded) {ctl}" if ctl else ""))
    json.dump({"source": args.source, "summary": summ, "sessions": rows},
              open(Path(out) / f"hemispheric_dynamics_{args.source}.json", "w"),
              indent=1, default=float)
    print(f"  wrote {plot(rows, out, args.source)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
