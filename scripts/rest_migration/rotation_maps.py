"""WHERE the position code rotates -- Haufe decoder PATTERNS in the shared joint basis.

WHAT THE TRANSFER MATRIX LEFT OPEN. `epoch_15g` established that the chronic code is a PARTIAL
ROTATION of the pre-stroke one (transfer symmetric in both directions, ~0.78-0.82 of ceiling for
cue and lick, 0.64/0.57 for ENL) and that every window ends ABOVE its pre-stroke ceiling. Both are
scalars. Neither says WHICH CORTEX the rotated fifth occupies, or where the extra information came
from. This does.

WHAT A HAUFE PATTERN IS, and why the decoder's weights are the wrong thing to plot. A weight is a
FILTER: a logistic regression can put large weight on a channel carrying no position signal at all,
purely to cancel correlated noise in a channel that does -- a suppressor variable. So a weight map
answers "what does the readout multiply?". The Haufe transform (Haufe et al. 2014) converts it to a
PATTERN, ``A = Cov(X) @ beta``, the covariance between each channel and the decoder's output, which
is the anatomical question: "which cortex actually co-varies with position". **In this dataset the
filter and the pattern correlate at only r = 0.245**, so this is not a refinement.

HOW THIS DIFFERS FROM `epoch_14` / `epoch_15r`, WHICH ARE ALSO HAUFE MAPS AND ALREADY EXIST. Those
fit PER SESSION in that session's OWN SVD basis and reference each epoch's map to a baseline; they
answer "where is each position's pattern, and how did it change". This fits ONE model per EPOCH in
the SHARED JOINT basis -- the same models the transfer matrix scores -- so the maps are directly
comparable across epochs and the difference between two of them IS the rotation the matrix measured.
Different question, same transform. Do not read one as a re-render of the other.

SHAPE, NOT GAIN. Each position's pattern is normalised to unit length before differencing, because
a decoder trained on a higher-SNR epoch produces a larger pattern for reasons that have nothing to
do with rotation, and the transfer result is about DIRECTIONS. The un-normalised amplitudes are in
the CSV for anyone who wants the gain question instead; `evoked_amplitude` and the rescale analyses
already own that one.

THE COSINE IS THE SUMMARY, THE MAP IS THE LOCALISATION. ``cos(pattern_pre, pattern_epoch)`` per
position is "how far did this position's readout direction turn"; the map shows where the turn
lives. Reported for acute, subacute AND chronic against pre (Priya, 2026-09-17), because the
matrix's own diagonal shows the three epochs are not on one trajectory -- acute DIPS and chronic
overshoots.

RUN AS:  python -m scripts.rest_migration.rotation_maps [--arms ENL cue lick rest]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from scripts.rest_migration.transfer_arms import ARMS, collect_rest, collect_task


def _xyg(data, labels):
    """Pool sessions into ``(X, y, blocks)``, block ids made unique across sessions."""
    X = np.concatenate([data[k][1] for k in labels])
    y = np.concatenate([data[k][2] for k in labels])
    g = np.concatenate([np.asarray(data[k][3], np.int64) + 1_000_000 * (i + 1)
                        for i, k in enumerate(labels)])
    return X, y, g


def _patterns_from(fit, X, basis, positions):
    """``({position: unit pattern}, {position: raw norm})`` for one fitted model."""
    from wfield_local import transfer_matrix as tm

    pats, norms = {}, {}
    for pos in positions:
        raw = tm.haufe_pattern(fit, X, pos)
        if raw is None:
            continue
        comp = tm.fold_bins(raw, basis.ncomp)
        nrm = float(np.linalg.norm(comp))
        if nrm > 0:
            pats[pos], norms[pos] = comp / nrm, nrm
    return pats, norms


def epoch_patterns(data, basis, pipe_fn, *, min_sessions=2, seed_ns="", log=print):
    """``(patterns, norms, ceiling)`` -- per epoch, per position, unit-norm Haufe patterns.

    ONE MODEL PER EPOCH, trained on that epoch's pooled sessions at the SAME block-matched size the
    transfer matrix uses, so these are the very models whose transfer was measured.

    ``ceiling`` IS WHAT MAKES THE COSINES READABLE, and the first version of this analysis omitted
    it. Two decoders fit on finite data disagree even when nothing changed, so a cross-epoch cosine
    is attenuated by estimation noise by an unknown amount -- 0.64 could be a large rotation or a
    noisy estimate of none. The ceiling splits the PRE-STROKE sessions into two DISJOINT halves,
    fits each on its own half, and takes the cosine between them: what an unchanged code yields
    with this much data. **Read every cross-epoch cosine against it, never against 1.0.**

    SPLIT BY SESSION, NOT BY TRIAL. A trial-level split leaves both halves carrying the same day's
    drift, which inflates the ceiling and would make every real rotation look larger than it is.
    """
    import hashlib

    from wfield_local import transfer_matrix as tm

    by_ep = {e: [k for k, v in data.items() if v[0] == e] for e in tm.EPOCH_ORDER}
    usable = [e for e in tm.EPOCH_ORDER if len(by_ep[e]) >= min_sessions]
    if "pre" not in usable:
        return {}, {}, {}
    n_target = int(min(sum(len(data[k][2]) for k in by_ep[e]) for e in usable))

    pats, norms = {}, {}
    for ep in usable:
        X, y, g = _xyg(data, by_ep[ep])
        seed = int(hashlib.sha1(f"{seed_ns}|{ep}".encode()).hexdigest()[:8], 16)
        got = tm._matched(pipe_fn, X, y, g, n_target, np.random.default_rng(seed))
        if got is None:
            continue
        pp, nn = _patterns_from(got[0], X, basis, sorted(np.unique(y).tolist()))
        if pp:
            pats[ep], norms[ep] = pp, nn

    # RELIABILITY PER EPOCH, not just for pre -- and the first version's pre-only ceiling was
    # actively misleading. It split pre in half, so its models saw 5-6 sessions while the epoch
    # models saw 11, making it systematically PESSIMISTIC: for PS92's ENL arm the "ceiling" came
    # out at +0.260 while the cross-epoch cosines were 0.410-0.437, i.e. ABOVE their own ceiling,
    # which is not interpretable at all.
    #
    # Each epoch's own split-half reliability is measured instead, and the cross-epoch cosine is
    # corrected for attenuation in BOTH: cos / sqrt(rel_pre * rel_epoch). That is the standard
    # correction and it handles epochs of different reliability, which is exactly the situation
    # here (pre has 11 sessions, acute 4-6, subacute 2-7).
    reliability = {}
    for ep in usable:
        labs = sorted(by_ep[ep])
        if len(labs) < 4:
            continue                      # cannot split; reliability stays unknown for this epoch
        rng = np.random.default_rng(
            int(hashlib.sha1(f"{seed_ns}|rel|{ep}".encode()).hexdigest()[:8], 16))
        shuf = [str(x) for x in rng.permutation(labs)]
        halves, half = [shuf[: len(shuf) // 2], shuf[len(shuf) // 2:]], []
        for hi, hl in enumerate(halves):
            Xh, yh, gh = _xyg(data, hl)
            got = tm._matched(pipe_fn, Xh, yh, gh, min(n_target, len(yh)),
                              np.random.default_rng(1000 + hi))
            if got is None:
                half = []
                break
            half.append(_patterns_from(got[0], Xh, basis, sorted(np.unique(yh).tolist()))[0])
        if len(half) == 2:
            reliability[ep] = {pos: float(half[0][pos] @ half[1][pos])
                               for pos in sorted(set(half[0]) & set(half[1]))}
    ceiling = reliability.get("pre", {})
    log(f"   patterns for {sorted(pats)}; split-half reliability "
        + (", ".join(f"{e}:{np.mean(list(v.values())):+.2f}"
                     for e, v in sorted(reliability.items())) or "NOT COMPUTABLE"))
    # THE HALVES ARE RETURNED, not just their cosine, because the |difference| between them is the
    # NULL MAP -- what "where did it change" looks like when nothing changed. Measured 2026-09-17:
    # the null map and the acute-minus-pre map correlate at r = +0.83 (PS92) and +0.81 (PS93), so
    # the localisation is mostly basis-shaped estimation noise and CANNOT carry a regional claim
    # on its own. Retrosplenial topping every panel was this, not the lesion.
    return pats, norms, ceiling, reliability


def null_delta(data, basis, pipe_fn, *, n_draws=12, n_target=None, seed_ns="", log=print):
    """``{position: (n_draws, ncomp)}`` of |pattern change| between two PRE-STROKE halves.

    THE FIX FOR THE MAP CONFOUND, and the confound is real: measured 2026-09-17, the raw
    |acute - pre| map correlates with this null at **r = +0.83**, so two thirds of its spatial
    structure is present when nothing changed. Retrosplenial topped every panel because the basis
    puts its largest components there -- footprint mass spans 67x across components -- and a
    component-space unit norm projects them to correspondingly large pixel mass whatever their
    coefficient. The bright region was where the NOISE is, not where the lesion acted.

    So the map is not compared against zero. Each COMPONENT is compared against ITS OWN null: a
    large, noisy component has to clear a correspondingly large bar, which removes the area
    weighting rather than merely flagging it.

    Each draw splits the pre-stroke sessions into two disjoint halves BY SESSION, fits a model on
    each, and records the per-component |difference| of their sign-aligned unit patterns. Different
    split every draw, so the spread reflects which sessions land together as well as the fit noise.
    """
    import hashlib

    from wfield_local import transfer_matrix as tm

    pre_labs = sorted([k for k, v in data.items() if v[0] == "pre"])
    if len(pre_labs) < 4:
        log("   null: NOT COMPUTABLE (<4 pre sessions)")
        return {}
    out = {}
    for d in range(n_draws):
        rng = np.random.default_rng(
            int(hashlib.sha1(f"{seed_ns}|null{d}".encode()).hexdigest()[:8], 16))
        shuf = [str(x) for x in rng.permutation(pre_labs)]
        halves, ok = [shuf[: len(shuf) // 2], shuf[len(shuf) // 2:]], True
        pats = []
        for hi, labs in enumerate(halves):
            X, y, g = _xyg(data, labs)
            tgt = min(n_target or len(y), len(y))
            got = tm._matched(pipe_fn, X, y, g, tgt, np.random.default_rng(7919 * d + hi))
            if got is None:
                ok = False
                break
            pats.append(_patterns_from(got[0], X, basis, sorted(np.unique(y).tolist()))[0])
        if not ok:
            continue
        for pos in sorted(set(pats[0]) & set(pats[1])):
            c = float(pats[0][pos] @ pats[1][pos])
            dd = np.abs((pats[1][pos] if c >= 0 else -pats[1][pos]) - pats[0][pos])
            out.setdefault(pos, []).append(dd)
    res = {p: np.asarray(v) for p, v in out.items() if len(v) >= 3}
    log(f"   null: {len(res)} positions x {min((len(v) for v in out.values()), default=0)} draws")
    return res


def excess_z(real_delta, null_draws):
    """Per-component z of the observed change against its OWN null. None if the null is unusable.

    ``(real - null_mean) / null_sd``. A component whose change is ordinary for the noise scores ~0
    however large it is in absolute terms, which is precisely what the raw map failed to do.
    """
    if null_draws is None or len(null_draws) < 3:
        return None
    mu, sd = null_draws.mean(0), null_draws.std(0)
    # THE SD IS FLOORED, and without it the map is unreadable. From ~10 draws a component can get a
    # near-zero SD by luck, and dividing by it produced z values above 30 -- which are a small
    # denominator, not a large effect, and they set the colour scale for every other panel. The
    # floor is the 25th percentile of the non-zero SDs in this position's own null, so it adapts to
    # the arm's noise level instead of being a constant that would be lenient for one arm and
    # crushing for another.
    pos_sd = sd[sd > 1e-12]
    floor = float(np.percentile(pos_sd, 25)) if pos_sd.size else np.inf
    sd = np.maximum(sd, floor)
    sd = np.where(sd > 1e-12, sd, np.inf)      # a component with no null spread cannot be scored
    return (np.asarray(real_delta) - mu) / sd


def to_pixels(comp_pattern, basis, mask=True):
    """Component-space pattern -> ``(H, W)`` cortical map through the SHARED footprints.

    MASKED TO `beta_maps.stat_mask` BY DEFAULT -- the eroded brain mask every other map family in
    this project uses. It removes the olfactory bulbs and the glue/window edge, which are exactly
    where `U` is smallest and the Allen warp least constrained, so they carry partial-volume
    mixing and read as signal (Priya, 2026-09-12 and again 2026-09-17). Outside the mask is NaN
    rather than zero, so a masked pixel cannot drag a mean or set a colour limit.
    """
    A = np.nan_to_num(np.asarray(basis.A, dtype=np.float32))
    H, W = A.shape[0], A.shape[1]
    out = (A.reshape(-1, basis.ncomp) @ np.asarray(comp_pattern, np.float32)).reshape(H, W)
    if mask:
        from wfield_local import beta_maps as bm
        m = np.asarray(bm.stat_mask(), bool)
        if m.shape == out.shape:
            out = np.where(m, out, np.nan)
    return out


def by_region(comp_pattern, basis):
    """``{allen_region: summed |pattern|}`` -- the regional answer, without a pixel map.

    Uses the basis's OWN region assignment (`Basis.regions`, derived from the footprints), so the
    labels are the same ones every other joint-basis figure uses.
    """
    out = {}
    regs = list(basis.regions)
    for c, r in enumerate(regs):
        out[str(r)] = out.get(str(r), 0.0) + abs(float(comp_pattern[c]))
    return out


def main() -> int:
    from wfield_local import config, joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.locanmf_frozen_decoder import _pipe
    from wfield_local.paths import PathResolver

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["ENL", "cue", "lick"], choices=list(ARMS))
    ap.add_argument("--bins", type=int, default=4, help="rest arm only")
    ap.add_argument("--null-draws", type=int, default=12,
                    help="pre-stroke split-half draws building the per-component null the maps "
                         "are scored against; below ~6 the SD is too noisy to divide by")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    animals = a.animals or sorted({s["label"].split("_")[0] for s in SESSIONS if s["label"] in want})
    t0 = time.time()
    print(f"ROTATION MAPS -- Haufe patterns in the joint basis, arms {a.arms}\n")

    rows, maps = [], {}
    for arm in a.arms:
        align, variant = ARMS[arm]
        print(f"\n== ARM {arm}")
        for an in animals:
            try:
                basis = joint_locanmf.load(an, sessions=SESSIONS)
                data = (collect_rest(an, a.bins) if arm == "rest"
                        else collect_task(an, align, variant))
            except Exception as ex:                                      # noqa: BLE001
                print(f"   !! {an}: {type(ex).__name__} {str(ex)[:80]} -- SKIPPED", flush=True)
                continue
            if not data:
                continue
            print(f"   {an}:", flush=True)
            pats, norms, ceiling, reliability = epoch_patterns(
                data, basis, _pipe, seed_ns=f"{arm}|{an}", log=lambda m: print(m, flush=True))
            if "pre" not in pats:
                continue
            nulls = null_delta(data, basis, _pipe, n_draws=a.null_draws,
                               seed_ns=f"{arm}|{an}", log=lambda m: print(m, flush=True))
            for ep in ("acute", "subacute", "chronic"):
                if ep not in pats:
                    continue
                shared = sorted(set(pats["pre"]) & set(pats[ep]))
                if not shared:
                    continue
                zmaps, cosines = [], []
                for pos in shared:
                    a_pre, a_ep = pats["pre"][pos], pats[ep][pos]
                    cos = float(a_pre @ a_ep)
                    cosines.append(cos)
                    # SIGN-ALIGNED before differencing. A Haufe pattern's overall sign is set by
                    # the class coding, and a flipped sign would register as a total rotation.
                    d = np.abs((a_ep if cos >= 0 else -a_ep) - a_pre)
                    z = excess_z(d, nulls.get(pos))
                    if z is not None:
                        zmaps.append((pos, z))
                    rel_a = reliability.get("pre", {}).get(pos)
                    rel_b = reliability.get(ep, {}).get(pos)
                    # CORRECTED FOR ATTENUATION IN BOTH EPOCHS. None when either reliability is
                    # unknown or non-positive -- a negative reliability means the split-half
                    # estimate is pure noise and no correction can rescue the cosine.
                    corr = None
                    if rel_a and rel_b and rel_a > 0 and rel_b > 0:
                        corr = round(cos / float(np.sqrt(rel_a * rel_b)), 4)
                    for reg, v in by_region(d, basis).items():
                        rows.append({"arm": arm, "animal": an, "contrast": f"{ep} - pre",
                                     "position": int(pos), "region": reg,
                                     "abs_delta": round(v, 6),
                                     "cosine_pre_vs_epoch": round(cos, 4),
                                     "rel_pre": (round(rel_a, 4) if rel_a else None),
                                     "rel_epoch": (round(rel_b, 4) if rel_b else None),
                                     "cos_attenuation_corrected": corr,
                                     "noise_ceiling_cos": (round(ceiling[pos], 4)
                                                           if pos in ceiling else None),
                                     "norm_pre": round(norms["pre"][pos], 4),
                                     "norm_epoch": round(norms[ep][pos], 4)})
                # PER POSITION, not averaged over them. A mean over the six would have hidden the
                # cue arm's far-contralateral cosine of -0.042 among five values near +0.6.
                for pos, z in zmaps:
                    maps.setdefault((arm, f"{ep} - pre", pos), []).append(to_pixels(z, basis))
                cb = np.mean([ceiling[p] for p in shared if p in ceiling]) if ceiling else np.nan
                print(f"      {ep}-pre: mean cos {np.mean(cosines):+.3f} "
                      f"(noise ceiling {cb:+.3f}) over {len(shared)} pos", flush=True)

    if not rows:
        print("no patterns computed")
        return 1
    p = out_dir / f"epoch_15h_rotation_regions{a.tag}.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[15h] wrote {p}")

    print("\nROTATION (cosine between the pre-stroke and the epoch pattern, 1.0 = no turn)")
    print(f"{'arm':<7}{'contrast':<18}" + "".join(f"{an:>10}" for an in animals))
    for arm in a.arms:
        for ep in ("acute", "subacute", "chronic"):
            sel = [r for r in rows if r["arm"] == arm and r["contrast"] == f"{ep} - pre"]
            if not sel:
                continue
            cells = []
            for an in animals:
                v = {(r["position"]): r["cosine_pre_vs_epoch"]
                     for r in sel if r["animal"] == an}
                cells.append(f"{np.mean(list(v.values())):>10.3f}" if v else f"{'--':>10}")
            print(f"{arm:<7}{ep + ' - pre':<18}" + "".join(cells))

    fig = _figure(maps, rows, out_dir, a.tag)
    print(f"\n[15h] wrote {fig}\n[15h] {time.time() - t0:.0f}s")
    return 0


def _figure(maps, rows, out_dir, tag=""):
    """ONE FIGURE PER ARM: rows = spout position, columns = epoch contrast, cells = the z-map.

    PER POSITION, because averaging over positions is what hid the cue arm's far-contralateral
    cosine of -0.042 among five values near +0.6 -- and far-contralateral is the lesion-relevant
    one, so the average was hiding exactly the cell worth seeing.

    THE CELLS ARE EXCESS-OVER-NOISE, not raw change. The raw |change| map correlates with its own
    pre-stroke null at r = +0.83, so it localises the BASIS rather than the lesion; each component
    is scored against its own split-half null instead. See `null_delta`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import epoch_figures as ef
    from wfield_local import transfer_matrix as tm
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES

    # THE CANONICAL ORDER AND NAMES, not a numeric sort. Every other position figure in this
    # project reads Near Ipsi -> Far Contra (`CONF_LABELS` order, `anatomical_labels` naming), and
    # `POSITION_NAMES` codes do NOT follow it -- 0 is close_CENTER, so sorting by code puts Middle
    # before Ipsi and silently transposes this figure against its neighbours.
    _pretty = dict(zip(CONF_LABELS, [x.title() for x in ef.anatomical_labels(CONF_LABELS,
                                                                            short=False)]))
    _rank = {lab: i for i, lab in enumerate(CONF_LABELS)}

    def _key(pos):
        return _rank.get(POSITION_NAMES.get(pos, ""), 99)

    def _name(pos):
        return _pretty.get(POSITION_NAMES.get(pos, ""), str(pos))

    cons = ["acute - pre", "subacute - pre", "chronic - pre"]
    made = []
    for arm in sorted({k[0] for k in maps}, key=lambda x: list(ARMS).index(x)):
        poss = sorted({k[2] for k in maps if k[0] == arm}, key=_key)
        if not poss:
            continue
        fig = plt.figure(figsize=(3.1 * len(cons) + 6.0, 2.5 * len(poss) + 2.6))
        # A SPACER COLUMN between the colorbar and the bar panel. Without it the bar panel's
        # position tick-labels land on top of the colorbar's ticks -- one `wspace` serves every
        # gap, and it has to stay small to keep the map columns adjacent.
        gs = fig.add_gridspec(len(poss), len(cons) + 3,
                              width_ratios=[1] * len(cons) + [0.07, 0.55, 2.0],
                              top=0.855, bottom=0.05, hspace=0.12, wspace=0.10)
        vals = [np.nanmean(np.asarray(v), axis=0) for k, v in maps.items() if k[0] == arm]
        flat = np.concatenate([v.ravel() for v in vals])
        flat = flat[np.isfinite(flat)]
        vmax = float(np.nanpercentile(np.abs(flat), 99)) if flat.size else 1.0
        im = None
        for i, pos in enumerate(poss):
            for j, con in enumerate(cons):
                ax = fig.add_subplot(gs[i, j])
                v = maps.get((arm, con, pos))
                ax.set_xticks([])
                ax.set_yticks([])
                if not v:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8, color="0.6")
                    continue
                cm = plt.get_cmap(tm.CMAP_CHANGE).copy()
                cm.set_bad("white")
                im = ax.imshow(np.nanmean(np.asarray(v), axis=0), cmap=cm,
                               vmin=-vmax, vmax=vmax)
                if i == 0:
                    ax.set_title(con, fontsize=10, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(_name(pos), fontsize=9, fontweight="bold")
        if im is not None:
            cax = fig.add_subplot(gs[:, len(cons)])
            fig.colorbar(im, cax=cax).set_label(
                "change in readout pattern, z vs its own pre-stroke split-half null", fontsize=8)

        # right: the cosine per position, against the noise ceiling
        ax = fig.add_subplot(gs[:, len(cons) + 2])
        w = 0.82 / len(cons)
        for j, con in enumerate(cons):
            ys = []
            for pos in poss:
                sel = [r["cosine_pre_vs_epoch"] for r in rows
                       if r["arm"] == arm and r["contrast"] == con and r["position"] == pos]
                ys.append(float(np.mean(sel)) if sel else np.nan)
            ax.barh(np.arange(len(poss)) - (j - (len(cons) - 1) / 2) * w, ys, height=w,
                    label=con.replace(" - pre", ""),
                    color=tm.epoch_color(con.replace(" - pre", "")))
        ceil = [r["noise_ceiling_cos"] for r in rows
                if r["arm"] == arm and r["noise_ceiling_cos"] not in (None, "")]
        if ceil:
            ax.axvline(float(np.mean([float(c) for c in ceil])), color="k", ls="--", lw=1.4,
                       label="noise ceiling")
        ax.set_yticks(range(len(poss)), [_name(p) for p in poss], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("cosine(pre pattern, epoch pattern)", fontsize=9)
        ax.set_title("HOW FAR THE READOUT TURNED\n(read against the dashed ceiling, not 1.0)",
                     fontsize=9, fontweight="bold")
        ax.axvline(0, color="k", lw=0.8)
        # BELOW THE AXES so it cannot sit on the bars or collide with the title.
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.06))
        ax.grid(axis="x", alpha=0.25)

        fig.text(0.5, 0.985, f"WHERE the position readout changes -- {arm} window",
                 ha="center", va="top", fontsize=13, fontweight="bold")
        fig.text(0.5, 0.945,
                 "Haufe patterns (A = Cov(X)b) from one block-matched decoder per epoch, in the "
                 "shared joint LocaNMF basis; unit-normalised per position, so this is SHAPE, not "
                 "gain.\n"
                 "MAPS ARE EXCESS OVER NOISE: each component is scored against its own "
                 "pre-stroke split-half null, because the RAW change map correlates with that null "
                 "at r = 0.83 and localises the basis, not the lesion.\n"
                 "A decoder WEIGHT is a filter, not a pattern (r = 0.245 here). Cohort mean; "
                 "three animals carry chronic data.",
                 ha="center", va="top", fontsize=8.5)
        out = out_dir / f"epoch_15h_rotation_maps_{arm}{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(out.name)
    return ", ".join(made)


if __name__ == "__main__":
    raise SystemExit(main())
