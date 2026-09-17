"""The `epoch_15k` figure: the three reference families as region x position cohort maps.

READS THE SAVED TABLES, NEVER RECOMPUTES. `reference_family_roi` costs ~35 min because it pays a
whole `maps_by_epoch` pass; this module opens `epoch_15k_cohort_<align>.csv` and draws. It is the
same separation `epoch_15h --replot` exists for, and for the same reason -- on 2026-09-17 two of
the day's three full passes bought nothing but a change to how something was DRAWN.

It REFUSES if the table is absent rather than computing one, so a colour change can never silently
cost half an hour.

THE COLOUR SCALE IS PER FAMILY, DELIBERATELY. `raw`, `precue` and `restw` are not in the same
units: `raw` has no subtrahend, `precue` is a cue-evoked INCREMENT over a per-trial baseline, and
`restw` is activity above rest. A shared scale would invite reading one family's amplitude against
another's, which is the one comparison these three cannot support. What IS comparable across them
is WHICH CELLS MOVE -- and that is what the agreement panel shows.

THE MARK IS AN INTERVAL, NOT A p. With four animals an animal-level permutation has 2^4 = 16
assignments and a floor of 0.0625, so no cell can reach p < 0.05 at any effect size. A dot means
the nested animals->sessions bootstrap CI excludes zero, per cell and UNCORRECTED across regions;
the guard against that is agreement across three references with non-overlapping failure modes,
not a threshold.

    python -m scripts.rest_migration.reference_family_figure [--align cue]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

FAMILIES = ("raw", "precue", "restw")
EPOCHS = ("acute", "subacute", "chronic")


def families_in(rows):
    """The families actually PRESENT, in canonical order.

    NOT the `FAMILIES` constant. The pre-cue ALIGNMENT runs on two families, because the `precue`
    REFERENCE would there subtract the feature window from itself -- so a figure that assumed
    three would draw an empty column and, worse, would count "all three agree" out of three when
    only two exist.
    """
    got = {r["family"] for r in rows}
    return tuple([f for f in FAMILIES if f in got] + sorted(got - set(FAMILIES)))


def load_cohort(out_dir, align="cue", tag=""):
    """Rows of `epoch_15k_cohort_<align><tag>.csv`, typed. ``None`` if it does not exist."""
    p = Path(out_dir) / f"epoch_15k_cohort_{align}{tag}.csv"
    if not p.exists():
        return None, p
    rows = []
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({**r,
                         "cohort_delta": float(r["cohort_delta"]),
                         "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
                         "ci_excludes_zero": r["ci_excludes_zero"] == "True",
                         "n_animals": int(r["n_animals"]),
                         "n_animals_same_sign": int(r["n_animals_same_sign"])})
    return rows, p


def grid(rows, family, epoch, regions, positions):
    """``(delta, sig, same_sign)`` as region x position arrays. NaN where the cell is absent."""
    ri = {r: i for i, r in enumerate(regions)}
    pi = {q: i for i, q in enumerate(positions)}
    d = np.full((len(regions), len(positions)), np.nan)
    s = np.zeros(d.shape, bool)
    n = np.full(d.shape, np.nan)
    for r in rows:
        if r["family"] != family or r["epoch"] != epoch:
            continue
        i, j = ri.get(r["region"]), pi.get(r["position"])
        if i is None or j is None:
            continue
        d[i, j] = r["cohort_delta"]
        s[i, j] = r["ci_excludes_zero"]
        n[i, j] = r["n_animals_same_sign"]
    return d, s, n


def agreement(rows, epoch, regions, positions, families=FAMILIES):
    """How many of the three families put a CI clear of zero on each cell (0-3)."""
    out = np.zeros((len(regions), len(positions)))
    for fam in families:
        _d, s, _n = grid(rows, fam, epoch, regions, positions)
        out += s.astype(float)
    return out


def signed_agreement(rows, epoch, regions, positions, families=FAMILIES):
    """``(signed, mixed)`` -- agreement count carrying the DIRECTION, -3..+3.

    Priya, 2026-09-17: *"can that be color-coded red shades or blue shades for increase or
    decrease?"* -- a count alone says three references agree that SOMETHING changed without saying
    which way, and the direction is the whole content of the cell.

    A CELL WHOSE SIGNIFICANT FAMILIES DISAGREE IN SIGN IS NOT AVERAGED INTO ONE. It returns 0 and
    is flagged in `mixed`, because two references pointing opposite ways is a fact about the
    references, not a weaker version of agreement -- averaging would hide exactly the disagreement
    this three-family design was built to surface.
    """
    n = np.zeros((len(regions), len(positions)))
    pos_n = np.zeros_like(n)
    neg_n = np.zeros_like(n)
    for fam in families:
        d, s, _ = grid(rows, fam, epoch, regions, positions)
        n += s.astype(float)
        pos_n += (s & (d > 0)).astype(float)
        neg_n += (s & (d < 0)).astype(float)
    mixed = (pos_n > 0) & (neg_n > 0)
    signed = np.where(pos_n >= neg_n, pos_n, -neg_n)
    signed[mixed] = 0.0
    return signed, mixed


def global_shift_test(out_dir, align="cue", tag="", families=FAMILIES, n_boot=2000, seed=7,
                      log=print):
    """Is the ALL-REGION per-position shift itself significant? The largest effect nothing tested.

    `global_vs_regional` showed that ~75% of the cue arm's acute change is a shift common to every
    region -- near positions up, far positions down, the same sign in all 30. Every statistic in
    `15k` is per REGION, so that component was measured and then left untested while the residual
    quarter got all the inference. This is the test it was missing.

    THE UNIT IS ONE SESSION'S MEAN OVER THE 30 REGIONS -- the same unweighted mean
    `global_vs_regional` subtracts -- and the estimator is the project's nested animals -> sessions
    bootstrap, so it reports an INTERVAL, never a p (four animals: an animal-level permutation
    floors at 1/16 = 0.0625).

    IT IS FRAMING (1) AND ONLY FRAMING (1). `15g`/`15h` score a position RELATIVE to the other
    five, so a shift that raises all six equally is invisible to them by construction. This one is
    NOT uniform across positions -- it reverses sign between near and far -- so the two framings
    should agree here, and that agreement is worth checking rather than assuming.

    Returns ``{(family, epoch, position): (delta, lo, hi, n_animals, n_same_sign)}``.
    """
    import json

    from scripts.rest_migration.reference_family_roi import _ordered, cohort_delta

    p = Path(out_dir) / f"epoch_15k_region_vectors_{align}{tag}.npz"
    if not p.exists():
        log(f"!! no vectors at {p}")
        return {}
    with np.load(p, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        V = np.asarray(f["V"])
    # ONE SCALAR PER SESSION: the mean over regions. Kept 2-D so `cohort_delta` -- which expects a
    # vector per session -- takes it unchanged rather than through a second code path.
    g = np.nanmean(V, axis=1)[:, None]

    animals = sorted({m["animal"] for m in meta})
    out = {}
    log(f"\n   THE GLOBAL PER-POSITION SHIFT ITSELF ({align}), nested animals->sessions bootstrap")
    log(f"   {'family':<8}{'epoch':<10}{'position':<15}{'delta':>10}{'95% CI':>22}   animals")
    for fam in families:
        for ep in EPOCHS:
            poss = _ordered({m["position"] for m in meta if m["epoch"] == ep})
            for pos in poss:
                pa = {}
                for an in animals:
                    pre_i = [k for k, m in enumerate(meta) if m["family"] == fam
                             and m["animal"] == an and m["epoch"] == "pre"
                             and m["position"] == pos]
                    ep_i = [k for k, m in enumerate(meta) if m["family"] == fam
                            and m["animal"] == an and m["epoch"] == ep
                            and m["position"] == pos]
                    if pre_i and ep_i:
                        pa[an] = (g[pre_i], g[ep_i])
                got = cohort_delta(pa, n_boot=n_boot, seed=seed)
                if got is None:
                    continue
                real, lo, hi, n_an, rep = got
                out[(fam, ep, pos)] = (float(real[0]), float(lo[0]), float(hi[0]), n_an,
                                       int(rep[0]))
                star = "*" if (lo[0] > 0 or hi[0] < 0) else " "
                log(f"   {fam:<8}{ep:<10}{pos:<15}{real[0]:>+10.5f}"
                    f"  [{lo[0]:>+8.5f},{hi[0]:>+8.5f}]{star}  {rep[0]}/{n_an}")
    n_sig = sum(1 for v in out.values() if v[1] > 0 or v[2] < 0)
    log(f"   -> {n_sig}/{len(out)} cells with the interval clear of zero "
        f"(* marks them; per cell, uncorrected)")
    return out


def hemisphere_check(out_dir, align="lick", tag="", family="restw", log=print):
    """Does atlas `_left` mean the ANIMAL'S left hemisphere? Measured, not assumed.

    THIS IS NOT PEDANTRY: every lesion in the cohort is LEFT-sided, so the suffix decides whether
    a regional result reads as IPSILESIONAL or CONTRALESIONAL, and `locanmf_analysis_deck` carried
    the inverted word from an unknown date until 2026-09-17. An image-space flip anywhere in the
    Allen warp would silently swap it, and no figure would look wrong.

    THE TEST NEEDS NO NEW COMPUTATION and no assumption about image handedness. Somatosensation is
    crossed, so a spout on the animal's RIGHT must drive the LEFT cortex harder. The spout half of
    the mapping is already fixed elsewhere -- `anatomical_labels` calls close_R/far_R 'near/far
    contra', and contra is the right side for a left lesion -- so the hemisphere suffix is the
    only unknown. Pre-stroke sessions only.

    Measured 2026-09-17 on the lick arm: EVERY `_left` area positive and EVERY `_right` area
    negative, over six independent pairs (SSp-n +0.34/-0.35, MOp +0.23/-0.14, SSp-un +0.18/-0.17,
    SSp-bfd, SSp-ul, and SSp-m_right -0.21 with no left partner in the vocabulary). So `_left` is
    the animal's left, the LEFT hemisphere is IPSILESIONAL, and it is also the one representing
    the impaired right side.

    Returns ``{region: laterality_index}``; positive = driven more by a right-side spout.
    """
    import json

    p = Path(out_dir) / f"epoch_15k_region_vectors_{align}{tag}.npz"
    if not p.exists():
        log(f"!! no vectors at {p}")
        return {}
    with np.load(p, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        regions = json.loads(str(f["regions"]))
        V = np.asarray(f["V"])
    ri = {r: i for i, r in enumerate(regions)}

    def side_mean(region, positions):
        idx = [k for k, m in enumerate(meta)
               if m["family"] == family and m["epoch"] == "pre" and m["position"] in positions]
        if not idx or region not in ri:
            return np.nan
        return float(np.nanmean(V[idx, ri[region]]))

    out = {}
    log(f"\n   HEMISPHERE CONVENTION -- pre-stroke, {family}. A RIGHT-side spout must drive the "
        f"animal's LEFT cortex harder (somatosensation is crossed).")
    log(f"   {'region':<16}{'L-spout':>10}{'R-spout':>10}{'index':>9}   expected")
    for base in ("SSp-m", "SSp-n", "SSp-un", "SSp-bfd", "SSp-ul", "MOp"):
        for sfx in ("left", "right"):
            nm = f"{base}_{sfx}"
            if nm not in ri:
                continue
            lv = side_mean(nm, ("close_L", "far_L"))
            rv = side_mean(nm, ("close_R", "far_R"))
            den = abs(rv) + abs(lv)
            li = (rv - lv) / den if den > 0 else np.nan
            out[nm] = li
            want = "+" if sfx == "left" else "-"
            ok = "ok" if (np.isfinite(li) and (li > 0) == (sfx == "left")) else "!! FLIPPED"
            log(f"   {nm:<16}{lv:>10.5f}{rv:>10.5f}{li:>9.3f}   {want}  {ok}")
    bad = [k for k, v in out.items()
           if np.isfinite(v) and (v > 0) != k.endswith("_left")]
    log(f"   -> {len(out) - len(bad)}/{len(out)} areas match the crossed expectation"
        + (f"; DISAGREEING: {', '.join(bad)}" if bad else "; '_left' IS the animal's left"))
    return out


def global_vs_regional(rows, family, epoch, regions, positions):
    """``(frac_explained, ranked_regions)`` -- how much of the change is GLOBAL, not anatomical.

    THE QUESTION THIS ANALYSIS EXISTS TO ANSWER IS "WHERE", so the first thing to measure is
    whether "where" has an answer at all. Take the mean over regions at each position -- a shift
    that hits the whole cortex the same way -- and ask how much of the region x position table it
    accounts for. What is left is the part that is genuinely about anatomy, and the regions are
    ranked by how far they depart from the global pattern.

    Measured on the cue arm 2026-09-17: acute 69-76%, chronic 26-46%. So the ACUTE change is
    mostly a global per-position shift with a modest regional overlay, and the CHRONIC change is
    where the anatomy actually carries the signal -- the opposite of what one would guess from the
    acute panel being the loudest.
    """
    ri = {r: i for i, r in enumerate(regions)}
    pi = {q: i for i, q in enumerate(positions)}
    d = np.full((len(regions), len(positions)), np.nan)
    for r in rows:
        if r["family"] == family and r["epoch"] == epoch:
            i, j = ri.get(r["region"]), pi.get(r["position"])
            if i is not None and j is not None:
                d[i, j] = r["cohort_delta"]
    if not np.isfinite(d).any():
        return np.nan, []
    # AN ENTIRELY ABSENT COLUMN IS A RESULT, NOT A GAP -- the lick arm has NO acute far-contra
    # cell in any animal, because acutely the animals do not lick at that spout. Dropping it here
    # keeps `nanmean` from warning on an empty slice; the figure LABELS it rather than leaving it
    # as anonymous grey.
    have = np.isfinite(d).any(axis=0)
    d = d[:, have]
    res = d - np.nanmean(d, axis=0)[None, :]
    tot = np.nansum((d - np.nanmean(d)) ** 2)
    frac = 1 - np.nansum(res ** 2) / tot if tot > 0 else np.nan
    rank = np.sqrt(np.nansum(res ** 2, axis=1))
    return float(frac), [regions[i] for i in np.argsort(-rank)]


def _figure(rows, out_dir, align="cue", tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from wfield_local import epoch_figures as ef
    from wfield_local import transfer_matrix as tm
    from wfield_local.grant_figures import CONF_LABELS

    rank = {q: i for i, q in enumerate(CONF_LABELS)}
    positions = sorted({r["position"] for r in rows},
                       key=lambda q: (rank.get(q, len(rank)), q))
    # ALPHABETICAL GROUPS THE HEMISPHERES UNDER THEIR AREA (MOp_left, MOp_right, ...), which is
    # what makes a left/right asymmetry visible as two adjacent rows rather than two distant ones.
    regions = sorted({r["region"] for r in rows})
    short = dict(zip(CONF_LABELS, ef.anatomical_labels(CONF_LABELS, short=True)))
    xt = [short.get(q, q) for q in positions]

    fams = families_in(rows)
    nrow, ncol = len(EPOCHS), len(fams) + 1
    fig_h = 1.15 + 0.148 * len(regions) * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol + 1.2, fig_h), squeeze=False,
                             gridspec_kw={"wspace": 0.12, "hspace": 0.16})
    # THE HEADER GETS ITS OWN INCHES. Placing the caption by figure FRACTION on a 14-inch figure
    # put it straight through the column titles -- the axes have to be pushed down in the same
    # units the text is measured in, not left at the default top.
    fig.subplots_adjust(top=1 - 1.75 / fig_h)

    # ONE SCALE PER FAMILY, over all three epochs of that family -- so an epoch can be read
    # against another epoch (the comparison this figure is for) but a family cannot be read
    # against another family (the comparison it cannot support).
    vmax = {}
    for fam in fams:
        v = np.abs(np.asarray([r["cohort_delta"] for r in rows
                               if r["family"] == fam and np.isfinite(r["cohort_delta"])]))
        vmax[fam] = float(np.nanpercentile(v, 99)) if v.size else 1.0

    # SIGNED, so the panel says which WAY. Blue shades = decrease, red = increase, darkness = how
    # many references agree. Grey is the middle because 0 means "no reference cleared zero" -- and
    # a sign DISAGREEMENT between references also lands there, hatched, rather than being averaged
    # into a direction it does not have.
    ag_cmap = ListedColormap(["#08519c", "#6baed6", "#c6dbef", "0.93",
                              "#fcbba1", "#fb6a4a", "#a50f15"])
    ag_norm = BoundaryNorm([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], ag_cmap.N)

    for i, ep in enumerate(EPOCHS):
        ag = agreement(rows, ep, regions, positions, fams)
        for j, fam in enumerate(fams):
            ax = axes[i][j]
            d, s, _n = grid(rows, fam, ep, regions, positions)
            cm = plt.get_cmap(tm.CMAP_CHANGE).copy()
            cm.set_bad("0.90")                     # ABSENT, and grey so it cannot read as zero
            im = ax.imshow(np.ma.masked_invalid(d), cmap=cm, vmin=-vmax[fam], vmax=vmax[fam],
                           aspect="auto", interpolation="nearest")
            yy, xx = np.where(s)
            ax.plot(xx, yy, "o", ms=2.6, mfc="k", mec="none", ls="none")
            # A RING WHERE EVERY FAMILY AGREES: the cell no single subtrahend's failure
            # explains. `len(fams)`, not 3 -- the pre-cue alignment runs on two.
            yy3, xx3 = np.where(ag >= len(fams))
            ax.plot(xx3, yy3, "o", ms=7.5, mfc="none", mec="k", mew=1.0, ls="none")
            # SAY WHY A COLUMN IS EMPTY. The lick arm has no acute far-contra cell in ANY animal
            # -- acutely they do not lick at that spout -- and unlabelled grey reads as a
            # pipeline failure when it is the deficit itself.
            for j2 in np.flatnonzero(~np.isfinite(d).any(axis=0)):
                ax.text(j2, len(regions) / 2, "no trials in any animal", rotation=90,
                        ha="center", va="center", fontsize=6.5, color="0.35")
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(xt, fontsize=7)
            ax.set_yticks(range(len(regions)))
            ax.set_yticklabels(regions if j == 0 else [], fontsize=5.4)
            ax.tick_params(length=1.5, pad=1)
            if i == 0:
                ax.set_title(fam, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ep, fontsize=11, fontweight="bold")
            if i == nrow - 1:
                cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.032, pad=0.10)
                cb.ax.tick_params(labelsize=6.5)
                cb.set_label(f"cohort delta, {fam} scale", fontsize=7)

        ax = axes[i][ncol - 1]
        sag, mixed = signed_agreement(rows, ep, regions, positions, fams)
        ax.imshow(sag, cmap=ag_cmap, norm=ag_norm, aspect="auto", interpolation="nearest")
        ym, xm = np.where(mixed)
        if ym.size:
            ax.plot(xm, ym, "x", ms=4, color="k", mew=1.0, ls="none")
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(xt, fontsize=7)
        ax.set_yticks(range(len(regions)))
        ax.set_yticklabels([], fontsize=5.4)
        ax.tick_params(length=1.5, pad=1)
        if i == 0:
            ax.set_title("families agreeing, SIGNED", fontsize=11, fontweight="bold")
        if i == nrow - 1:
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=ag_norm, cmap=ag_cmap), ax=ax,
                              orientation="horizontal", fraction=0.032, pad=0.10,
                              ticks=[-3, -2, -1, 0, 1, 2, 3])
            cb.ax.tick_params(labelsize=6.5)
            cb.set_label("n families clear of 0 (blue = decrease, red = increase)", fontsize=7)

    fig.text(0.5, 1 - 0.30 / fig_h,
             f"epoch_15k -- WHERE the {align} position map changes, three references "
             f"with different failure modes",
             ha="center", va="top", fontsize=15, fontweight="bold")
    fig.text(0.5, 1 - 0.72 / fig_h,
             "Rows = epoch, columns = reference. Cell = COHORT delta (epoch - pre) for one Allen "
             "region x spout position, nested animals->sessions bootstrap.\n"
             "DOT = that bootstrap CI excludes zero (per cell, UNCORRECTED across regions). RING = "
             "all three references agree -- the cell no single subtrahend explains.\n"
             "LAST COLUMN: blue = decrease, red = increase, darker = more references agreeing; "
             "x = the significant references DISAGREE in sign.\n"
             "COLOUR SCALES DIFFER BETWEEN COLUMNS ON PURPOSE: raw has no subtrahend, precue is "
             "an increment over the per-trial pre-cue window, restw is activity above rest.\n"
             "Epochs within a column ARE comparable. The unit is a LocaNMF COMPONENT grouped by "
             "its Allen label, never a pixel mean over the area.\n"
             + ("THE LICK ARM IS CONDITIONED ON TRIALS THE ANIMAL LICKED, so a cell is the map "
                "of the attempts that HAPPENED -- acute far-contra has no cell at all because "
                "there were none."
                if align == "lick" else
                "Cue-aligned, so every trial contributes whether or not the animal responded."),
             ha="center", va="top", fontsize=8.5)
    out = Path(out_dir) / f"epoch_15k_reference_families_{align}{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _ccf_figure(rows, out_dir, align="cue", tag="", family="restw", n_families=None):
    """The all-three-families cells PAINTED ON THE ALLEN CCF. Rows = epoch, columns = position.

    Priya, 2026-09-17: *"can we make a version of the agree in all 3 that is on the Allen CCF
    picture?"* A 30-row region list is a table of names; the same result on a brain is an
    anatomical claim you can read at a glance, and the two SSp subdivisions that dominate every
    chronic list are neighbours -- which the alphabetical row order actively hides.

    ONE FAMILY SUPPLIES THE COLOUR (`restw` by default) AND THE OTHER TWO SUPPLY THE GATE. Only
    cells whose CI clears zero in ALL THREE are painted, so the map is the agreed set; the number
    shown is one reference's, because the three are not in the same units and averaging them would
    invent a quantity none of them measures.

    A PAINTED REGION IS A LocaNMF-COMPONENT MEAN SPREAD OVER ITS ALLEN FOOTPRINT -- it is NOT a
    pixel map, and the flat colour inside a region is the point rather than a rendering shortcut.
    The sharp edges are the parcellation's, not the data's.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import beta_maps as bm
    from wfield_local import epoch_figures as ef
    from wfield_local import transfer_matrix as tm
    from wfield_local.atlas_overlay import overlay_regions
    from wfield_local.grant_figures import CONF_LABELS

    # DERIVED, NEVER A LITERAL 3. The pre-cue alignment carries TWO families, and the hardcoded 3
    # painted nothing there while the console line above it reported 13 cells agreeing in all 2 --
    # a figure disagreeing with its own summary, with only the figure visible.
    present = families_in(rows)
    if n_families is None:
        n_families = len(present)

    atlas, names = bm._atlas_names()
    if atlas is None or not names:
        print("!! no Allen atlas available -- skipping the CCF figure")
        return None
    inv = {v: k for k, v in names.items()}
    try:
        smask = np.asarray(bm.stat_mask(), bool)
    except Exception:                                                  # noqa: BLE001
        smask = None
    edges = bm.atlas_edges()

    rank = {q: i for i, q in enumerate(CONF_LABELS)}
    positions = sorted({r["position"] for r in rows}, key=lambda q: (rank.get(q, 99), q))
    pretty = dict(zip(CONF_LABELS, [x.title()
                                    for x in ef.anatomical_labels(CONF_LABELS, short=False)]))

    # WHICH CELLS ARE AGREED, and the value one reference gives them.
    agreed = {}
    for r in rows:
        if r["ci_excludes_zero"]:
            agreed.setdefault((r["epoch"], r["position"], r["region"]), set()).add(r["family"])
    val = {(r["epoch"], r["position"], r["region"]): r["cohort_delta"]
           for r in rows if r["family"] == family}

    # "0 regions" IS TWO DIFFERENT FACTS and they must not share a label: no cell AGREED, versus
    # no cell EXISTED. The lick arm has no acute far-contra data at all, and an unqualified zero
    # there reads as a null result when it is the deficit.
    have_data = {(r["epoch"], r["position"]) for r in rows}

    imgs, painted = {}, []
    for ep in EPOCHS:
        for pos in positions:
            img = np.full(atlas.shape, np.nan, float)
            hit = 0
            for (e, q, reg), fams in agreed.items():
                if e != ep or q != pos or len(fams) < n_families:
                    continue
                sid = inv.get(reg)
                v = val.get((ep, pos, reg))
                if sid is None or v is None:
                    continue
                m = np.rint(atlas).astype(np.int32) == int(sid)
                if not m.any():
                    continue
                img[m] = v
                painted.append(v)
                hit += 1
            imgs[(ep, pos)] = (img, hit)
    if not painted:
        print("!! no cell agreed in all three families -- nothing to paint")
        return None
    vmax = float(np.nanpercentile(np.abs(painted), 99))

    nrow, ncol = len(EPOCHS), len(positions)
    fig_h = 1.95 + 2.25 * nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.25 * ncol + 0.9, fig_h), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    fig.subplots_adjust(top=1 - 1.55 / fig_h, bottom=0.075)
    cm = plt.get_cmap(tm.CMAP_CHANGE).copy()
    cm.set_bad("0.955")            # AGREED-ON-NOTHING, and pale so it cannot read as a zero value

    for i, ep in enumerate(EPOCHS):
        for j, pos in enumerate(positions):
            ax = axes[i][j]
            img, hit = imgs[(ep, pos)]
            ax.imshow(np.ma.masked_invalid(img), cmap=cm, vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            if edges is not None:
                overlay_regions(ax, edges)
            if smask is not None and smask.shape == atlas.shape:
                ax.contour(smask.astype(float), levels=[0.5], colors="k",
                           linewidths=0.7, linestyles="--")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            lab = (f"{hit} region{'' if hit == 1 else 's'}" if (ep, pos) in have_data
                   else "NO TRIALS IN ANY ANIMAL")
            ax.text(0.02, 0.02, lab, transform=ax.transAxes,
                    fontsize=7, color="0.25", ha="left", va="bottom",
                    bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.2})
            if i == 0:
                ax.set_title(pretty.get(pos, pos), fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ep, fontsize=12, fontweight="bold")

    cax = fig.add_axes([0.35, 0.035, 0.30, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(-vmax, vmax), cmap=cm), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7)
    cb.set_label(f"cohort delta (epoch - pre), {family} reference", fontsize=8)

    fig.text(0.5, 1 - 0.30 / fig_h,
             f"epoch_15k -- {align} position map: the regions ALL "
             f"{'TWO' if n_families == 2 else 'THREE' if n_families == 3 else n_families} "
             f"references agree on",
             ha="center", va="top", fontsize=15, fontweight="bold")
    fig.text(0.5, 1 - 0.70 / fig_h,
             f"Painted only where the cohort CI excludes zero in ALL {n_families} references "
             f"({' AND '.join(present)}) -- the change no single subtrahend's failure mode "
             f"explains. Colour is the {family} value.\n"
             "FLAT COLOUR INSIDE A REGION IS THE RESULT, NOT A RENDERING SHORTCUT: the unit is a "
             "LocaNMF component mean, spread over its Allen footprint. The edges are the "
             "parcellation's.\n"
             "Dashed line = the analysed mask (olfactory bulbs and the glue/window edge "
             "excluded). Pale grey = agreed on nothing there.\n"
             "'_left' IS THE ANIMAL'S LEFT and every lesion is left-sided, so left = "
             "IPSILESIONAL, and it is also the hemisphere representing the impaired right side.",
             ha="center", va="top", fontsize=8.5)
    out = Path(out_dir) / f"epoch_15k_agreed_ccf_{align}{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    from wfield_local.paths import PathResolver

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--align", default="cue", choices=["cue", "lick", "precue"],
                    help="`precue` is the ENL window and carries TWO families -- the precue "
                         "REFERENCE is excluded there because its baseline lies inside the "
                         "feature window.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--global-test", action="store_true",
                    help="test the ALL-REGION per-position shift itself -- ~75%% of the cue "
                         "arm's acute change, and nothing else tests it. Reads the saved "
                         "vectors; costs nothing.")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--hemisphere-check", action="store_true",
                    help="verify that atlas '_left' is the ANIMAL'S left and exit. Reads the "
                         "saved vectors; costs nothing.")
    a = ap.parse_args()
    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")

    if a.hemisphere_check:
        return 0 if hemisphere_check(out_dir, a.align, a.tag) else 1
    if a.global_test:
        return 0 if global_shift_test(out_dir, a.align, a.tag, n_boot=a.boot) else 1

    rows, p = load_cohort(out_dir, a.align, a.tag)
    if rows is None:
        print(f"!! no cohort table at {p} -- run\n"
              f"   python -m scripts.rest_migration.reference_family_roi --align {a.align}\n"
              f"   first. REFUSING to recompute: that is a ~35 min pass and this module draws.")
        return 1
    n_reg = len({r["region"] for r in rows})
    fams = families_in(rows)
    print(f"[15k] {len(rows)} cohort cells over {n_reg} regions from {p.name}")
    print(f"[15k] families present: {list(fams)}"
          + ("   (the precue REFERENCE is absent by design under the precue ALIGNMENT -- its "
             "baseline lies inside the feature window)" if "precue" not in fams else ""))
    for ep in EPOCHS:
        by = {}
        for r in rows:
            if r["epoch"] == ep and r["ci_excludes_zero"]:
                by.setdefault((r["region"], r["position"]), set()).add(r["family"])
        allf = sum(1 for v in by.values() if len(v) == len(fams))
        # A SIGN DISAGREEMENT BETWEEN REFERENCES IS THE ONE OUTCOME THIS DESIGN CANNOT ABSORB, so
        # it is counted out loud rather than left to be noticed on the figure.
        rg = sorted({r["region"] for r in rows})
        pp = sorted({r["position"] for r in rows})
        _sg, mixed = signed_agreement(rows, ep, rg, pp, fams)
        print(f"   {ep:<9} {len(by):>4} region-positions with any family clear of 0, "
              f"{allf:>3} clear in ALL {len(fams)}, {int(mixed.sum()):>3} with references "
              f"DISAGREEING in sign")

    # IS THERE A "WHERE" AT ALL? Printed on every draw, because a table of significant regions
    # reads as a localisation even when the same shift hits every region equally.
    from wfield_local.grant_figures import CONF_LABELS
    rank = {q: i for i, q in enumerate(CONF_LABELS)}
    positions = sorted({r["position"] for r in rows}, key=lambda q: (rank.get(q, 99), q))
    regions = sorted({r["region"] for r in rows})
    print("\n   GLOBAL vs REGIONAL -- how much of the change is a per-position shift common to "
          "every region")
    for ep in EPOCHS:
        for fam in fams:
            frac, ranked = global_vs_regional(rows, fam, ep, regions, positions)
            if not np.isfinite(frac):
                continue
            print(f"      {ep:<9}{fam:<8}{frac * 100:5.1f}% global   "
                  f"most-departing: {', '.join(ranked[:4])}")
    out = _figure(rows, out_dir, a.align, a.tag)
    print(f"[15k] wrote {out}")
    ccf = _ccf_figure(rows, out_dir, a.align, a.tag)
    if ccf is not None:
        print(f"[15k] wrote {ccf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
