"""WHERE the position map changes, in Allen ROIs, under THREE references with DIFFERENT failure modes.

Priya, 2026-09-17: *"we could look at `raw` (corrected SVD) cue maps, precue-normalized cue
('increment') maps, and rest-normalized cue maps"* -- three analyses, not one with options, and
*"this is why we will do the 3 families of map statistical comparisons"*.

THE ARGUMENT IS THE DISAGREEMENT, NOT THE AGREEMENT OF ANY ONE. Each reference is wrong in its own
way, and the ways do not overlap:

  raw      no subtrahend, so nothing in it can drift between epochs. Carries cross-day
           MULTIPLICATIVE scaling instead -- which NO subtraction removes either, so it is the
           confound left standing rather than one this family adds (`crossday_intensity` owns it).
  precue   subtrahend is PER TRIAL, so it cannot drift across epochs by construction. But it
           violates F12: the pre-cue window carries real anticipatory position signal (LOSO 0.510),
           so it subtracts real code and measures the cue-evoked INCREMENT, not the position map.
  restw    activity above the resting state, positions kept independent. Its baseline MOVES across
           epochs -- measured 2026-09-17 at 7-43% of the evoked signal, significant in 6 of 11
           animal-epoch cells (`rest_baseline_epoch_drift`), which BOUNDS the distortion it can
           impose on an across-epoch amplitude.

A regional effect present in ALL THREE is not a property of any one subtrahend. One present in only
one IS, and the table says which. That is the same logic the repo already applies when it reports
"two independent references, sharing no subtrahend, give the SAME ordering".

THE UNIT IS A LocaNMF COMPONENT, GROUPED BY ITS ALLEN LABEL -- not a pixel ROI (Priya, 2026-09-17:
*"let's abandon ROI for the basis.regions"*). The first version averaged PIXELS inside each Allen
area, and MOs is 15,613 px: a focal change in part of it was swamped by the rest. Averaging
COMPONENTS gives each functional parcel one vote, so the same change survives. It also puts this
analysis on the unit `epoch_15h` already tests, so a disagreement between them is about FRAMING
(1 vs 2) rather than about parcellation.

COMPONENTS DO NOT CORRESPOND ACROSS ANIMALS (PS92 95, PS93 87, PS94 90, PS95 95) but their ALLEN
LABELS do, and that label is what keeps a cohort test possible. The vocabulary is the INTERSECTION
across animals -- 34 regions carrying an in-mask component in all four -- so no region's cohort mean
silently rests on a different subset of animals.

A component's value is its FOOTPRINT-WEIGHTED mean of the map, `sum(|A_c|*map)/sum(|A_c|)`; the
region is then the MEAN over its components, NEVER a sum. A sum would rank regions by how many
components they happen to hold, which is the retrosplenial artefact in a third guise (footprint mass
spans 67x).

MASKED FIRST with `beta_maps.stat_mask` via `in_mask_components`: olfactory bulbs and the
glue/window edge, where `U` is smallest and the Allen warp least constrained.

TWO STATISTICS, AND THEY ANSWER DIFFERENT QUESTIONS. The PER-ANIMAL max-statistic corrects across
regions within an animal. The COHORT delta (`cohort_delta`) resamples ANIMALS -> SESSIONS and is the
only one that speaks for the group -- the first version had neither, pooling instead with an
ANY-ANIMAL rule whose per-cell false-positive rate was ~18%, not 5%.

PER-SESSION VECTORS ARE SAVED so the cohort, a different correction or another aggregation is a
re-read rather than another pass over `maps_by_epoch`.

    python -m scripts.rest_migration.reference_family_roi [--align cue] [--perm 2000]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

#: The three families, in the order they should be read.
FAMILIES = ("raw", "precue", "restw")

EPOCHS = ("acute", "subacute", "chronic")

#: A region must keep this many pixels inside `stat_mask` to be testable at all.
MIN_REGION_PX = 200


def shared_regions(animals):
    """``(sids, labels, mask)`` -- the Allen regions carrying an IN-MASK component in EVERY animal.

    THE INTERSECTION, not the union, and that is deliberate. A region present in three animals and
    absent in the fourth cannot enter a cohort mean without the fourth silently contributing
    nothing, which would make the cohort estimate rest on a different set of animals per region --
    the same "which animals is this number actually about" failure the any-animal pooling rule had.

    Measured 2026-09-17: all four animals carry the same 64 sids, and every component has a known
    one, so the intersection costs nothing here. It is a guard, not a filter.
    """
    from scripts.rest_migration.rotation_maps import in_mask_components
    from wfield_local import beta_maps as bm
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    _atlas, names = bm._atlas_names()
    if not names:
        raise RuntimeError("no Allen names available -- cannot label components")
    mask = np.asarray(bm.stat_mask(), bool)
    per = []
    for an in animals:
        basis = joint_locanmf.load(an, sessions=SESSIONS)
        keep = in_mask_components(basis)
        regs = [int(x) for x in basis.regions]
        per.append({regs[c] for c in np.flatnonzero(keep)
                    if c < len(regs) and regs[c] in names})
    shared = sorted(set.intersection(*per)) if per else []
    return np.asarray(shared), [names[s] for s in shared], mask


def cohort_delta(per_animal, n_boot=2000, seed=7, alpha=0.05):
    """``(mean, lo, hi, n_animals, n_replicating)`` -- the COHORT delta by nested bootstrap.

    WHY THIS EXISTS, and it is a correction to the first version of this script. The agreement
    table originally called a cell significant in a family if ANY ONE ANIMAL cleared p < 0.05
    there. The per-animal max-statistic corrects across REGIONS but nothing corrected across the
    ANIMAL axis, so with four animals the per-cell false-positive rate was ~1 - 0.95^4 = 18%, not
    5% -- and "significant in all three families" could mean three different animals. That is a
    replication count wearing a cohort test's clothes.

    THE UNIT RESAMPLED IS THE ANIMAL, with sessions resampled inside it -- the project's standard
    nested bootstrap, the same object the bar families use. Animals are the replicates; sessions
    within an animal are not independent of each other.

    WHY A BOOTSTRAP CI AND NOT A PERMUTATION p: with FOUR animals an animal-level sign-flip null
    has 2^4 = 16 possible assignments, so its smallest attainable p is 1/16 = 0.0625 and NO CELL
    CAN REACH 0.05 at any effect size. That is the same draw-count floor that made "nothing
    survived Bonferroni" meaningless in the 15h arm, arriving through the cohort size instead of
    the draw count. A CI has no such floor -- it can exclude zero -- so the cohort claim is
    "the interval excludes zero", never "p < 0.05".

    ``per_animal`` is ``{animal: (pre_sessions, epoch_sessions)}``, each an array of per-session
    region vectors. THE DELTA IS FORMED INSIDE EACH DRAW, from independently resampled pre and
    epoch sessions -- the two groups are different recordings, so there is nothing to pair and
    resampling a precomputed delta would understate the uncertainty in the pre mean.
    """
    animals = sorted(per_animal)
    if len(animals) < 2:
        return None
    k = np.asarray(per_animal[animals[0]][0]).shape[1]
    per_mean = {a: (np.nanmean(np.asarray(v[1]), axis=0)
                    - np.nanmean(np.asarray(v[0]), axis=0)) for a, v in per_animal.items()}
    real = np.nanmean(np.asarray([per_mean[a] for a in animals]), axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, k))
    for b in range(n_boot):
        pick = rng.choice(len(animals), size=len(animals), replace=True)
        acc = []
        for i in pick:
            pre_s, ep_s = (np.asarray(x) for x in per_animal[animals[i]])
            ip = rng.choice(pre_s.shape[0], size=pre_s.shape[0], replace=True)
            ie = rng.choice(ep_s.shape[0], size=ep_s.shape[0], replace=True)
            acc.append(np.nanmean(ep_s[ie], axis=0) - np.nanmean(pre_s[ip], axis=0))
        draws[b] = np.nanmean(np.asarray(acc), axis=0)
    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    # REPLICATION IS REPORTED BESIDE THE INTERVAL, not instead of it: how many animals share the
    # cohort's SIGN. With n = 4 this is the more legible number and the CI is the formal one.
    rep = np.sum([np.sign(np.asarray([per_mean[a] for a in animals])) == np.sign(real)], axis=1)[0]
    return real, lo, hi, len(animals), rep


def _ordered(positions):
    """Positions in the CANONICAL Near Ipsi -> Far Contra order, not alphabetical.

    `CONF_LABELS` is the project's order and every neighbouring figure uses it. Sorting these
    strings alphabetically gives close_L, close_center, close_R, far_L, far_center, far_R -- which
    happens to coincide here, but silently would not if a label were renamed, and a table whose
    rows are in a different order from the figure beside it is the kind of mismatch nobody checks.
    """
    from wfield_local.grant_figures import CONF_LABELS

    rank = {q: i for i, q in enumerate(CONF_LABELS)}
    return sorted(positions, key=lambda q: (rank.get(q, len(rank)), str(q)))


def component_weights(animal, region_labels):
    """``(W, keep_region)`` -- per-component footprint weights and each component's region index.

    ADOPTED 2026-09-17 IN PLACE OF PIXEL-ROI AVERAGING (Priya: *"let's abandon ROI for the
    basis.regions"*). A pixel mean over MOs -- 15,613 px, certainly not one functional unit --
    swamps a focal change in a small part of it. One vote per COMPONENT does not.

    Components do not correspond ACROSS animals (PS92 95, PS93 87, PS94 90, PS95 95), so the
    component index is useless as a shared axis -- but `Basis.regions` labels each one with an Allen
    area, and THOSE correspond. The label is what keeps a cohort test possible.

    Only components passing `in_mask_components` contribute: the rest are olfactory bulb and
    glue/window edge.
    """
    from scripts.rest_migration.rotation_maps import in_mask_components
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    basis = joint_locanmf.load(animal, sessions=SESSIONS)
    keep = in_mask_components(basis)
    A = np.nan_to_num(np.asarray(basis.A, dtype=np.float32))
    regs = list(basis.regions)
    idx_of = {int(r): i for i, r in enumerate(region_labels)}
    cols, rid = [], []
    for c in np.flatnonzero(keep):
        r = regs[c] if c < len(regs) else None
        if r not in idx_of:
            continue
        cols.append(np.abs(A[:, :, c]).reshape(-1))
        rid.append(idx_of[r])
    if not cols:
        return None, None
    return np.asarray(cols), np.asarray(rid)


def region_means(pixmap, W, rid, n_regions, mask):
    """``(n_regions,)`` -- the MEAN over each region's COMPONENTS, never a sum.

    A component's value is its FOOTPRINT-WEIGHTED mean of the map, `sum(|A_c|*map)/sum(|A_c|)`, so a
    component contributes where it actually lives. Regions then average their components with EQUAL
    weight -- a sum would rank regions by how many components they happen to contain, which is the
    retrosplenial artefact in a third guise (footprint mass spans 67x).

    NaN-safe: a region with no finite component value returns NaN, because "no data" and "no change"
    must not render as the same number.
    """
    v = np.asarray(pixmap).reshape(-1)
    m = mask.reshape(-1)
    good = m & np.isfinite(v)
    num = W[:, good] @ v[good]
    den = W[:, good].sum(axis=1)
    comp = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    out = np.full(n_regions, np.nan)
    for i in range(n_regions):
        sel = comp[rid == i]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            out[i] = float(sel.mean())
    return out


def maxstat_p(real, null_draws):
    """Family-wise p per region by the PERMUTATION MAX-STATISTIC across regions.

    Each draw contributes its single most extreme |value| across the whole region family; the
    family-wise null is the distribution of those maxima. This respects the fact that neighbouring
    regions share components and drift -- correlated regions give correlated draws, so the max
    distribution is narrower than independence implies -- and it costs NO extra draws, because it
    reuses the ones already computed. Strictly better than Bonferroni here, not merely different.
    """
    real = np.asarray(real, float)
    nd = np.asarray(null_draws, float)
    if nd.ndim != 2 or nd.shape[0] < 10:
        return np.full(real.shape, np.nan)
    per_draw_max = np.nanmax(np.abs(nd), axis=1)
    return np.array([float(np.mean(per_draw_max >= abs(r))) if np.isfinite(r) else np.nan
                     for r in real])


def main() -> int:
    from pathlib import Path

    from wfield_local.paths import PathResolver
    from wfield_local.position_reference_maps import maps_by_epoch

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--align", default="cue", choices=["cue", "lick", "precue"],
                    help="`precue` is the ENL window [cue - 2 s, cue] -- the motor-independent "
                         "readout. It runs on TWO families, not three: see below.")
    ap.add_argument("--variant", default=None,
                    help="default: working for cue and precue, lick for lick")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917)

    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    variant = a.variant or ("lick" if a.align == "lick" else "working")
    # THE PRE-CUE ARM HAS TWO FAMILIES, NOT THREE, AND THE MISSING ONE IS NOT AN OVERSIGHT. With
    # `align=precue` the feature window is [cue - 2 s, cue] and the precue BASELINE is its own
    # final second, so that reference would subtract the window from itself. Dropped loudly here
    # as well as in `session_raw_maps`, because a silently two-family run would still print an
    # "ALL THREE families" table -- with three meaning two.
    if a.align == "precue" and "precue" in a.families:
        a.families = [f for f in a.families if f != "precue"]
        print("!! PRECUE ALIGNMENT: dropping the `precue` FAMILY -- its baseline lies inside the "
              f"feature window. Families are {a.families}, and 'all families agree' below means "
              f"ALL {len(a.families)}, not three.")
    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")

    by_epoch, _rel, _n = maps_by_epoch(a.align, variant)
    animals = sorted(by_epoch)
    sids, labels, mask = shared_regions(animals)
    # THE THRESHOLD AND THE SHA GO IN THE HEADER. On 2026-09-17 the only way to tell which
    # MIN_IN_MASK_FRAC a finished run had used was to count its regions (30 = 0.75, 34 = 0.5) --
    # `shared_regions` imports `in_mask_components` lazily, so the run's START time does not
    # settle it. An artefact should say what produced it.
    from scripts.rest_migration.rotation_maps import MIN_IN_MASK_FRAC, git_sha
    print(f"REFERENCE FAMILIES BY Basis.regions -- {a.align}/{variant}, families {a.families}")
    print(f"   MIN_IN_MASK_FRAC = {MIN_IN_MASK_FRAC}   git {git_sha()}")
    print(f"{len(sids)} Allen regions carrying an in-mask COMPONENT in all {len(animals)} animals "
          f"(the correction family)\n{'=' * 78}", flush=True)
    wts = {}
    for an in animals:
        W, rid = component_weights(an, sids)
        if W is None:
            print(f"   !! {an}: no in-mask components map to a shared region -- EXCLUDED",
                  flush=True)
            continue
        wts[an] = (W, rid)
        print(f"   {an}: {W.shape[0]} components over {len(set(rid.tolist()))} regions", flush=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    # PER-SESSION VECTORS KEPT, so any later statistic -- the cohort bootstrap, a different
    # correction, another aggregation -- is a re-read of this npz rather than another pass over
    # `maps_by_epoch`. The first version stored only the per-cell MEANS, which is why the cohort
    # analysis could not reuse it; that is the same mistake `epoch_15h`'s cache was built to end.
    vectors = []

    for fam in a.families:
        print(f"\n== FAMILY {fam}", flush=True)
        for an in animals:
            if an not in wts:
                continue
            W, rid = wts[an]
            per = {}
            for ep, positions in by_epoch[an].items():
                if ep not in ("pre",) + EPOCHS:
                    continue
                for pos, labmaps in positions.items():
                    # `labmaps` is {session_label: {reference: map}} -- one entry per SESSION, so
                    # the list below is the per-session region vectors the sign-flip null resamples.
                    for _lab, refs in labmaps.items():
                        mp = refs.get(fam)
                        if mp is None:
                            continue
                        vec = region_means(mp, W, rid, len(sids), mask)
                        per.setdefault((ep, pos), []).append(vec)
                        vectors.append({"family": fam, "animal": an, "epoch": ep,
                                        "position": str(pos), "session": str(_lab), "v": vec})
            if not per:
                print(f"   .. {an}: no maps for {fam} -- skipped", flush=True)
                continue
            for ep in EPOCHS:
                for pos in _ordered({p for (e, p) in per if e == ep}):
                    if ("pre", pos) not in per:
                        continue
                    pre = np.nanmean(np.asarray(per[("pre", pos)]), axis=0)
                    now = np.nanmean(np.asarray(per[(ep, pos)]), axis=0)
                    delta = now - pre
                    n_pre = len(per[("pre", pos)])
                    n_ep = len(per[(ep, pos)])
                    # SIGN-FLIP NULL over the pooled per-session region vectors: under the null of
                    # no epoch change the delta's sign is arbitrary, and flipping preserves the
                    # spatial covariance between regions -- which is exactly what the max-statistic
                    # needs and what a region-shuffle would destroy.
                    pool = np.asarray(per[("pre", pos)] + per[(ep, pos)])
                    draws = np.empty((a.perm, len(sids)))
                    for d in range(a.perm):
                        sgn = rng.choice([-1.0, 1.0], size=pool.shape[0])[:, None]
                        draws[d] = np.nanmean(pool * sgn, axis=0)
                    p = maxstat_p(delta, draws)
                    for i, nm in enumerate(labels):
                        rows.append({"family": fam, "align": a.align, "animal": an, "epoch": ep,
                                     "position": str(pos), "region": nm,
                                     "n_pre_sessions": n_pre, "n_epoch_sessions": n_ep,
                                     "pre": round(float(pre[i]), 6),
                                     "epoch_value": round(float(now[i]), 6),
                                     "delta": round(float(delta[i]), 6),
                                     "p_fwe_maxstat": (round(float(p[i]), 4)
                                                       if np.isfinite(p[i]) else None)})
            print(f"   {an}: {sum(1 for r in rows if r['animal'] == an and r['family'] == fam)} "
                  f"region cells", flush=True)

    if not rows:
        print("\nno rows -- a failed result, not a result")
        return 1

    # ---- SAVE THE PER-SESSION VECTORS -------------------------------------------------------
    import json
    meta = [{k: d[k] for k in ("family", "animal", "epoch", "position", "session")}
            for d in vectors]
    np.savez_compressed(
        out_dir / f"epoch_15k_region_vectors_{a.align}{a.tag}.npz",
        meta=json.dumps(meta), regions=json.dumps([str(x) for x in labels]),
        sids=np.asarray(sids), V=np.asarray([d["v"] for d in vectors], dtype=np.float32))
    print(f"\nwrote per-session vectors: {len(vectors)} rows x {len(sids)} regions", flush=True)

    # ---- THE COHORT DELTA, the statistic the per-animal cells cannot provide -----------------
    print(f"\n{'=' * 78}\nCOHORT delta (epoch - pre), nested animals->sessions bootstrap\n"
          f"{'=' * 78}")
    coh = []
    for fam in a.families:
        for ep in EPOCHS:
            for pos in _ordered({d["position"] for d in vectors if d["epoch"] == ep}):
                pa = {}
                for an in animals:
                    pre_v = [d["v"] for d in vectors if d["family"] == fam and d["animal"] == an
                             and d["epoch"] == "pre" and d["position"] == pos]
                    ep_v = [d["v"] for d in vectors if d["family"] == fam and d["animal"] == an
                            and d["epoch"] == ep and d["position"] == pos]
                    if pre_v and ep_v:
                        pa[an] = (np.asarray(pre_v), np.asarray(ep_v))
                got = cohort_delta(pa, n_boot=min(a.perm, 2000), seed=a.seed)
                if got is None:
                    continue
                real, lo, hi, n_an, rep = got
                for i, nm in enumerate(labels):
                    excl = bool(np.isfinite(lo[i]) and np.isfinite(hi[i])
                                and (lo[i] > 0 or hi[i] < 0))
                    coh.append({"family": fam, "align": a.align, "epoch": ep, "position": pos,
                                "region": nm, "n_animals": n_an,
                                "cohort_delta": round(float(real[i]), 6),
                                "ci_lo": round(float(lo[i]), 6), "ci_hi": round(float(hi[i]), 6),
                                "ci_excludes_zero": excl,
                                "n_animals_same_sign": int(rep[i])})
    if coh:
        pc = out_dir / f"epoch_15k_cohort_{a.align}{a.tag}.csv"
        with open(pc, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(coh[0]))
            w.writeheader()
            w.writerows(coh)
        print(f"wrote {pc}", flush=True)
        print("\nCOHORT cells whose CI EXCLUDES ZERO in ALL THREE families "
              "(the defensible result):")
        for ep in EPOCHS:
            byc = {}
            for r in coh:
                if r["epoch"] == ep and r["ci_excludes_zero"]:
                    byc.setdefault((r["region"], r["position"]), set()).add(r["family"])
            allf = sorted(k for k, v in byc.items() if len(v) == len(a.families))
            print(f"  {ep}: {len(allf)} region-positions in all {len(a.families)} families")
            for k in allf[:14]:
                d = [r for r in coh if r["epoch"] == ep and r["region"] == k[0]
                     and r["position"] == k[1] and r["family"] == "restw"]
                s = f" restw {d[0]['cohort_delta']:+.5f} ({d[0]['n_animals_same_sign']}/4 same sign)" \
                    if d else ""
                print(f"     {k[0]:<16} {k[1]:<14}{s}")
        print("\n  NOTE the CI is per cell and NOT corrected across regions. With 4 animals an")
        print("  animal-level permutation cannot reach p<0.05 (2^4=16, floor 0.0625), so the")
        print("  interval is the claim and the all-three-families agreement is the guard.")
    p_out = out_dir / f"epoch_15k_reference_family_roi_{a.align}{a.tag}.csv"
    with open(p_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {p_out}", flush=True)

    # ---- THE TABLE THAT IS THE POINT: which regions survive in HOW MANY families ------------
    print(f"\n{'=' * 78}\nAGREEMENT ACROSS FAMILIES -- a region in all three is not a property "
          f"of any subtrahend\n{'=' * 78}")
    for ep in EPOCHS:
        hits = {}
        for fam in a.families:
            sig = {(r["region"], r["position"]) for r in rows
                   if r["family"] == fam and r["epoch"] == ep
                   and r["p_fwe_maxstat"] is not None and r["p_fwe_maxstat"] < 0.05}
            for k in sig:
                hits.setdefault(k, set()).add(fam)
        allthree = sorted(k for k, v in hits.items() if len(v) == len(a.families))
        some = sorted((k, sorted(v)) for k, v in hits.items() if 1 <= len(v) < len(a.families))
        print(f"\n{ep}:  {len(allthree)} region-positions significant in ALL "
              f"{len(a.families)} families, {len(some)} in only some")
        for k in allthree[:12]:
            print(f"    ALL  {k[0]:<16} pos {k[1]}")
        for k, v in some[:8]:
            print(f"    only {','.join(v):<20} {k[0]:<16} pos {k[1]}")
    print("\nREAD IT AS: ALL-three = a regional change no single reference's failure mode can")
    print("explain. only-one = a property of that subtrahend -- check its failure mode in the")
    print("module docstring before believing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
