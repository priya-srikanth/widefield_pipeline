"""Per-AREA evoked amplitude, pre vs post-stroke — the measure that matches what the maps show.

Priya, 2026-08-18, from the preprocessing decks: *"there's a big difference in the post-cue and
especially post-lick activity maps 8/17 compared to our included pre-stroke SVD maps. Generally, I see
more R sensorimotor activity for multiple spout positions, esp the far center and far R positions, and
in some cases maybe relatively less L activity. I also noted that the amplitude of the activity
measurement bars is MUCH larger."*

WHY THE EXISTING HEMISPHERIC MEASURES COULD NOT HAVE SEEN THIS. `hemispheric_intensity`,
`hemispheric_dynamics` and `vessel_contrast` all returned nulls, and all three collapse across SPACE --
a per-hemisphere median or a whole-hemisphere ratio. A focal right-sensorimotor increase averaged with
32 other areas vanishes. Those nulls say nothing about this observation, and this module is the one
that tests it.

THREE QUANTITIES, BECAUSE "BIGGER" IS AMBIGUOUS.

  ABSOLUTE   mean windowed response per area x position, in the same fractional units the maps are
             drawn in. This is what the colourbars show.
  SHARE      each area's |response| as a fraction of that session's total across areas, per position.
             A bigger response and a bigger slice of a bigger total are different claims and the maps
             cannot separate them.
  R-L INDEX  (right - left) / (|right| + |left|) per HOMOTOPIC pair. This is Priya's specific claim,
             and it is scale-free: it survives any change that scales a session uniformly.

THE CONFOUND THE ABSOLUTE NUMBER CARRIES AND THE OTHER TWO DO NOT. The signal is a deviation from the
session's own mean, so the denominator is session-specific, and LED power is titrated by hand daily. A
rise in ABSOLUTE amplitude can come from a larger response or from a smaller baseline. SHARE and the
R-L index are immune. **If ABSOLUTE moves while the other two do not, the change is a gain change and
not a redistribution of activity.** That is the single most important thing this module can tell us,
and it is why all three are computed rather than the one that was observed.

PS92/PS93 ARE THE SMALL-LESION ARM, NOT A CONTROL (Priya, 2026-08-18) -- they were lesioned too, just
mildly. Same recording day, rig, anaesthesia and preprocessing, so they control for the DAY; if their
amplitudes jump equally the effect is the session. If they jump LESS, that is a severity relationship,
which is a stronger result than a clean control would have given.

CLI::

    python -m wfield_local.evoked_amplitude
    python -m wfield_local.evoked_amplitude --align cue lick --animals PS94 PS95
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from wfield_local import config                                     # noqa: E402
from wfield_local.paths import PathResolver                         # noqa: E402
from wfield_local.plot_lick_aligned_averages import (DISPLAY_ORDER,  # noqa: E402
                                                     POSITION_NAMES)

#: The areas Priya's observation is about. Reported as a group as well as per area, because a claim
#: about "sensorimotor" should not rest on whichever single area happened to move most.
SENSORIMOTOR = ("SSp", "SSs", "MOp", "MOs")


def _names(s):
    f = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")
    if not f:
        return None
    return {int(c): str(n) for c, n in json.load(open(f[0]))}


def session_amplitudes(s, align="cue", source="roi"):
    """Per-area x per-position evoked amplitude for one session, plus its share and R-L decomposition.

    Sub-bins are averaged back to one value per area: `_trial_features` tiles each region across the
    window's bins (4 at cue, 8 at lick), and treating those as separate features would weight an area
    by how many bins it happens to have.
    """
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import _trial_features

    names = _names(s)
    if names is None:
        return None
    X, y, _g, _Xn, _yn, reg = _trial_features(s, _args(source=source, align=align, post_s=2.0))
    if len(y) < 30:
        return None
    reg = np.asarray(reg)
    codes = sorted(set(int(c) for c in reg.tolist()))
    area = {c: names.get(c, "?") for c in codes}

    # collapse bins -> one column per area
    A = np.stack([X[:, reg == c].mean(axis=1) for c in codes], axis=1)      # (trials, areas)

    amp, share, npos = {}, {}, {}
    for p in DISPLAY_ORDER:
        m = y == p
        pname = POSITION_NAMES[p]
        npos[pname] = int(m.sum())
        if m.sum() < 5:
            continue
        a = A[m].mean(axis=0)                                              # signed mean response
        amp[pname] = {area[c]: float(v) for c, v in zip(codes, a)}
        tot = float(np.abs(a).sum()) or 1.0
        share[pname] = {area[c]: float(abs(v) / tot) for c, v in zip(codes, a)}

    # R-L index per homotopic pair
    pairs = defaultdict(dict)
    for c in codes:
        n = area[c]
        if n.endswith("_left"):
            pairs[n[:-5]]["L"] = n
        elif n.endswith("_right"):
            pairs[n[:-6]]["R"] = n
    rl = {}
    for pname, d in amp.items():
        rl[pname] = {}
        for base, sides in pairs.items():
            if "L" not in sides or "R" not in sides:
                continue
            l, r = d.get(sides["L"]), d.get(sides["R"])
            if l is None or r is None:
                continue
            den = abs(l) + abs(r)
            rl[pname][base] = float((r - l) / den) if den > 1e-12 else float("nan")

    out = {"label": s["label"], "animal": s["label"][:4], "date": s["label"].split("_")[-1],
           "align": align, "n_per_position": npos,
           "amplitude": amp, "share": share, "rl_index": rl}
    # headline scalars: the sensorimotor group, which is the claim being tested
    sm = [b for b in pairs if any(b == p or b.startswith(p) for p in SENSORIMOTOR)]
    out["sensorimotor_areas"] = sorted(sm)
    out["rl_sensorimotor"] = {p: float(np.nanmean([v[b] for b in sm if b in v]))
                              for p, v in rl.items() if sm}
    out["abs_total"] = {p: float(np.abs(list(d.values())).sum()) for p, d in amp.items()}
    return out


def collect(animals=None, align="cue", source="roi", curated_only=True):
    """Sessions to measure. CURATED dates only by default.

    `session_phase(...) == "pre"` is TRUE for every date before the lesion, including the noisy early
    June sessions and 8/5 that `curated_dates()` exists to exclude -- and one of them, PS95_0605, has a
    mean |amplitude| of 16.3 against ~0.53 for every other session. Including it put PS95's pre-stroke
    band at [0.145, 18.09], inside which no post-stroke value could ever fall. Every other analysis in
    this project builds its reference from the curated set; this one has to as well.
    """
    keep_dates = set(config.curated_dates()) | {"0817"}
    sessions = config.load_sessions()
    if curated_only:
        sessions = [x for x in sessions if x["label"].split("_")[-1] in keep_dates]
    if animals:
        sessions = [x for x in sessions if x["label"][:4] in set(animals)]
    rows = []
    for s in sessions:
        try:
            r = session_amplitudes(s, align=align, source=source)
        except Exception as ex:                                          # noqa: BLE001
            print(f"  {s['label']}: skip ({str(ex)[:60]})", flush=True)
            continue
        if r is None:
            continue
        r["phase"] = config.session_phase(r["animal"], r["date"])
        rows.append(r)
        tot = np.mean(list(r["abs_total"].values())) if r["abs_total"] else float("nan")
        print(f"  {r['label']} {align:5s} total|amp| {tot:.4f}   "
              f"R-L sensorimotor {np.nanmean(list(r['rl_sensorimotor'].values())):+.3f}", flush=True)
    return sorted(rows, key=lambda r: (r["animal"], r["date"]))


def _band(vals):
    v = np.array([x for x in vals if x == x], float)
    if len(v) < 3:
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def summarise(rows):
    """Post-stroke value against that animal's own pre-stroke session range, for all three quantities."""
    out = {}
    for a in sorted({r["animal"] for r in rows}):
        pre = [r for r in rows if r["animal"] == a and r["phase"] == "pre"]
        post = [r for r in rows if r["animal"] == a and r["phase"] == "post"]
        excl = [r for r in rows if r["animal"] == a and r["phase"] == "excluded"]
        if len(pre) < 3 or not (post or excl):
            out[a] = {"note": f"{len(pre)} pre / {len(post)} post / {len(excl)} small-lesion"}
            continue
        rec = {"n_pre": len(pre)}
        for qty in ("abs_total", "rl_sensorimotor"):
            rec[qty] = {}
            for p in [POSITION_NAMES[c] for c in DISPLAY_ORDER]:
                b = _band([r[qty].get(p, np.nan) for r in pre])
                if b is None:
                    continue
                e = {"pre_band": b}
                for tag, group in (("post", post), ("small_lesion", excl)):
                    for r in group:
                        v = r[qty].get(p, np.nan)
                        e.setdefault(tag, {})[r["date"]] = float(v)
                        if tag == "post":
                            e.setdefault("post_z", {})[r["date"]] = (
                                float((v - b["mean"]) / b["sd"]) if b["sd"] else float("nan"))
                            e.setdefault("outside", {})[r["date"]] = bool(v < b["min"] or v > b["max"])
                rec[qty][p] = e
        # per-area z on the SHARE, so a redistribution shows even if the total moved
        rec["share_z"] = {}
        for p in [POSITION_NAMES[c] for c in DISPLAY_ORDER]:
            if not any(p in r["share"] for r in pre) or not post:
                continue
            areas = sorted(post[0]["share"].get(p, {}))
            zz = {}
            for ar in areas:
                b = _band([r["share"].get(p, {}).get(ar, np.nan) for r in pre])
                if b is None or not b["sd"]:
                    continue
                v = post[0]["share"][p].get(ar, np.nan)
                zz[ar] = float((v - b["mean"]) / b["sd"])
            if zz:
                rec["share_z"][p] = zz
        out[a] = rec
    return out


def plot(rows, summ, out_dir, align="cue"):
    """Three rows: absolute total, R-L sensorimotor index, and a per-area SHARE z heatmap."""
    animals = sorted({r["animal"] for r in rows if r["phase"] == "post"}) or \
              sorted({r["animal"] for r in rows})
    POS = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    fig, axes = plt.subplots(3, len(animals), figsize=(5.2 * len(animals), 11.0), squeeze=False)
    for k, a in enumerate(animals):
        rec = summ.get(a, {})
        for row, (qty, lab) in enumerate((("abs_total", "total |amplitude| (ABSOLUTE)"),
                                          ("rl_sensorimotor", "R-L index, sensorimotor (SCALE-FREE)"))):
            ax = axes[row][k]
            x = np.arange(len(POS))
            for i, p in enumerate(POS):
                e = rec.get(qty, {}).get(p)
                if not e:
                    ax.text(i, 0, "n/a", ha="center", fontsize=6.5, color="firebrick", rotation=90)
                    continue
                b = e["pre_band"]
                ax.add_patch(plt.Rectangle((i - 0.32, b["min"]), 0.64, max(b["max"] - b["min"], 1e-9),
                                           color="tab:blue", alpha=0.18, zorder=1))
                ax.plot([i - 0.32, i + 0.32], [b["mean"]] * 2, color="tab:blue", lw=1.8, zorder=2)
                for d, v in (e.get("post") or {}).items():
                    z = e.get("post_z", {}).get(d, float("nan"))
                    ax.plot(i, v, "o", ms=10, color="tab:red", markeredgecolor="k", zorder=4)
                    ax.text(i, v, f"  z{z:+.1f}", fontsize=6.5, va="center",
                            color=("firebrick" if e.get("outside", {}).get(d) else "dimgrey"))
                for d, v in (e.get("small_lesion") or {}).items():
                    ax.plot(i + 0.18, v, "s", ms=7, color="grey", markeredgecolor="k", zorder=3)
            if qty == "rl_sensorimotor":
                ax.axhline(0, color="k", lw=1, ls=":")
            ax.set_xticks(x)
            ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{a} — {lab}", fontsize=9)
            if k == 0:
                ax.set_ylabel(lab, fontsize=8)
        # per-area SHARE z heatmap
        ax = axes[2][k]
        sz = rec.get("share_z", {})
        areas = sorted({ar for p in sz for ar in sz[p]})
        if areas:
            M = np.full((len(areas), len(POS)), np.nan)
            for j, p in enumerate(POS):
                for i, ar in enumerate(areas):
                    if p in sz and ar in sz[p]:
                        M[i, j] = sz[p][ar]
            im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
            ax.set_yticks(range(len(areas)))
            ax.set_yticklabels(areas, fontsize=4.5)
            ax.set_xticks(range(len(POS)))
            ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{a} — per-area SHARE z (red = larger share post-stroke)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "no share data", ha="center", transform=ax.transAxes)
    fig.suptitle(
        f"Per-AREA evoked amplitude, {align}-aligned. THE HEMISPHERIC NULLS CANNOT SPEAK TO THIS: "
        "intensity, dynamics, concordance and vessels all collapse across space, so a focal change "
        "averages away in every one of them.\nROW 1 is what the map colourbars show and carries a "
        "confound the others do not — the signal is a deviation from the session's own mean and LED "
        "power is set by hand, so a rise can be a larger response OR a smaller baseline. ROWS 2-3 are "
        "scale-free. IF ROW 1 MOVES AND ROWS 2-3 DO NOT, the change is a GAIN change, not a "
        "redistribution of activity. Grey squares = the small-lesion animals (same day, mild lesion, "
        "NOT a no-lesion control).", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out_dir) / f"evoked_amplitude_{align}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--align", nargs="+", default=["cue", "lick"])
    ap.add_argument("--source", default="roi")
    ap.add_argument("--all-dates", action="store_true",
                    help="include non-curated dates (they carry the June outliers)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    for align in args.align:
        print(f"\n=== {align}-aligned ===", flush=True)
        rows = collect(args.animals, align=align, source=args.source,
                       curated_only=not args.all_dates)
        if not rows:
            continue
        summ = summarise(rows)
        for a, rec in summ.items():
            if "note" in rec:
                print(f"  {a}: {rec['note']}")
                continue
            print(f"  --- {a} ---")
            for qty in ("abs_total", "rl_sensorimotor"):
                for p, e in rec.get(qty, {}).items():
                    b = e["pre_band"]
                    ps = "  ".join(
                        f"{d}={v:+.4f} (z={e['post_z'].get(d, float('nan')):+.1f}"
                        f"{', OUTSIDE' if e.get('outside', {}).get(d) else ''})"
                        for d, v in (e.get("post") or {}).items())
                    if ps:
                        print(f"    {qty:16s} {p:13s} pre {b['mean']:+.4f}±{b['sd']:.4f} "
                              f"[{b['min']:+.4f},{b['max']:+.4f}]  POST {ps}")
        json.dump({"align": align, "summary": summ, "sessions": rows},
                  open(Path(out) / f"evoked_amplitude_{align}.json", "w"), indent=1, default=float)
        print(f"  wrote {plot(rows, summ, out, align)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
