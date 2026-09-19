"""Do `15k`'s all-family regions sit UNDER `15r`'s significant blobs? The spatial cross-check.

Two DIFFERENT test units and two DIFFERENT correction families agreeing is much stronger evidence
than either alone, and disagreement would be diagnostic rather than embarrassing. The two share the
trial data and the rest baseline and nothing else:

    15r   ~2,022 PIXEL bins        bootstrap max-statistic over bins, FWE threshold
    15k   33 ALLEN REGIONS from    nested animals->sessions CI, gated on agreement across three
          LocaNMF components       references whose failure modes differ

PANEL-LEVEL CONCORDANCE IS NOT THE QUESTION. Counting 15k regions against 15r significant bins per
panel gives Spearman +0.810 (lick) and +0.916 (cue), which says the two families flag the same
POSITIONS and EPOCHS. It does not say they agree about WHERE INSIDE THE BRAIN -- and agreeing on
the panel while differing on the place is exactly the failure this test exists to catch.

THE TEST. For one panel (position x epoch-contrast), 15r's significant-bin mask comes from the
figure bundle's stored `contours` and each region's footprint from the Allen atlas, both on the
same 540x640 grid. Score every region in 15k's vocabulary by the fraction of its footprint lying
inside 15r's significant bins, then ask whether the regions 15k called AGREED-IN-ALL-FAMILIES score
higher than the ones it did not.

THE NULL SHUFFLES THE AGREED LABEL **WITHIN PANEL**, and that is the whole design. Panels differ
enormously in how much cortex 15r flags -- cue acute far-contra is 1,186 bins against 20 for
far-ipsi -- so a global shuffle would only rediscover that agreed regions live in loud panels,
which the Spearman already showed. Shuffling inside a panel holds the blob size AND the number of
agreed regions fixed, leaving LOCATION as the only thing that can produce a difference.

RESULT, 2026-09-18: agreed regions have 52% of their footprint inside 15r's bins against 3-5% for
the rest, a within-panel difference of +0.56 (lick) and +0.51 (cue), p = 0.0001 at 10,000 draws
(the empirical floor). READ THE LICK ARM AS THE LOAD-BEARING ONE: 15k measures the cue acute panel
as ~75% GLOBAL, and a whole-cortex shift agrees with any anatomical claim, so concordance there is
close to uninformative. The lick arm is 11-31% global and gives the LARGER effect.

ATLAS KEYING IS BY SIGNED ID, NOT INDEX (`beta_maps._atlas_names`) -- getting that wrong once
silently mapped "FRP" onto primary motor cortex.

    python -m scripts.rest_migration.cross_15k_15r [--perm 10000] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = (("lick", "lick"), ("cue", "working"))
MIN_FOOTPRINT_PX = 50          # below this a footprint fraction is not a stable quantity


def sig_masks(d, stem, in_mask):
    """``{(position_label, contrast): bool mask}`` -- 15r's significant bins, from its bundle."""
    meta = json.loads((d / f"{stem}_bundle.json").read_text(encoding="utf-8"))
    z = np.load(d / f"{stem}_bundle.npz")
    shape = tuple(meta["shape"])
    n = int(np.prod(shape))
    out = {}
    for c in meta["cells"]:
        if c.get("kind") != "contours":
            continue
        m = np.unpackbits(z[f"contours{c['i']}"])[:n].astype(bool).reshape(shape)
        out[(c["row"], c["col"])] = m & in_mask
    return out


def collect(d, align, variant, atlas, name2sid, in_mask, anat):
    """``(recs, panels, n_fams, n_vocab)`` with ``recs = [(panel, region, frac, agreed)]``."""
    sig = sig_masks(d, f"epoch_15r_position_RESTWref_{align}_{variant}", in_mask)
    rows = list(csv.DictReader(open(d / f"epoch_15k_cohort_{align}.csv", encoding="utf-8")))
    fams = sorted({r["family"] for r in rows})
    vocab = sorted({r["region"] for r in rows})
    clear = defaultdict(set)
    for r in rows:
        if str(r.get("ci_excludes_zero", "")).lower() == "true":
            clear[(r["epoch"], r["position"], r["region"])].add(r["family"])

    foot = {}
    for reg in vocab:
        sid = name2sid.get(reg)
        if sid is None:
            continue
        m = (atlas == sid) & in_mask
        if m.sum() >= MIN_FOOTPRINT_PX:
            foot[reg] = m

    recs = []
    for (pos_raw, ep) in sorted({(p, e) for (e, p, _r) in clear}):
        key = (anat.get(pos_raw, pos_raw), f"{ep} - pre")
        if key not in sig:
            continue
        s = sig[key]
        for reg, m in foot.items():
            agreed = len(clear.get((ep, pos_raw, reg), ())) == len(fams)
            recs.append((key, reg, float((m & s).sum()) / float(m.sum()), agreed))
    return recs, sorted({r[0] for r in recs}), len(fams), (len(foot), len(vocab))


def within_panel_diff(recs, panels, flags):
    """Mean over panels of (agreed mean - not-agreed mean); panels needing both classes."""
    idx_by_panel = {p: [i for i, r in enumerate(recs) if r[0] == p] for p in panels}
    diffs = []
    for idx in idx_by_panel.values():
        a = [recs[i][2] for i in idx if flags[i]]
        b = [recs[i][2] for i in idx if not flags[i]]
        if a and b:
            diffs.append(np.mean(a) - np.mean(b))
    return (float(np.mean(diffs)) if diffs else np.nan), len(diffs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local import beta_maps as bm
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.paths import PathResolver

    d = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    atlas, names = bm._atlas_names()
    if atlas is None:
        print("!! no Allen atlas on the shared grid -- cannot score footprints")
        return 1
    in_mask = np.asarray(bm.stat_mask(), bool)
    name2sid = {v: k for k, v in names.items()}
    anat = dict(zip(CONF_LABELS,
                    [x.title() for x in ef.anatomical_labels(CONF_LABELS, short=False)]))
    rng = np.random.default_rng(7)

    out_rows = []
    for align, variant in ARMS:
        recs, panels, n_fams, (n_foot, n_vocab) = collect(
            d, align, variant, atlas, name2sid, in_mask, anat)
        if not recs:
            print(f"!! {align}: nothing to score")
            continue
        flags = np.array([r[3] for r in recs])
        obs, npan = within_panel_diff(recs, panels, flags)
        by_panel = {p: np.array([i for i, r in enumerate(recs) if r[0] == p]) for p in panels}
        null = []
        for _ in range(a.perm):
            f2 = flags.copy()
            for idx in by_panel.values():
                f2[idx] = rng.permutation(flags[idx])
            v, _n = within_panel_diff(recs, panels, f2)
            if np.isfinite(v):
                null.append(v)
        null = np.asarray(null)
        p = (1 + int((null >= obs).sum())) / (1 + null.size)
        ag = [r[2] for r in recs if r[3]]
        no = [r[2] for r in recs if not r[3]]
        print("=" * 76)
        print(f"{align.upper()} ARM -- {len(panels)} panels, {n_foot}/{n_vocab} regions scorable, "
              f"{n_fams} families")
        print("  footprint fraction inside 15r significant bins:")
        print(f"     agreed in all {n_fams}      {np.mean(ag):.3f}  (n={len(ag)})")
        print(f"     not agreed           {np.mean(no):.3f}  (n={len(no)})")
        print(f"  within-panel difference over {npan} panels: {obs:+.3f}")
        print(f"  permutation p (label shuffled WITHIN panel, {null.size} draws): {p:.4f}")
        out_rows.append(dict(arm=align, n_panels=len(panels), n_families=n_fams,
                             mean_frac_agreed=round(float(np.mean(ag)), 4),
                             mean_frac_not_agreed=round(float(np.mean(no)), 4),
                             n_agreed=len(ag), n_not_agreed=len(no),
                             within_panel_diff=round(obs, 4), n_panels_scored=npan,
                             perm_p=round(p, 5), n_perm=int(null.size)))
    if out_rows:
        q = d / "epoch_15kr_cross_check.csv"
        with open(q, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n[15k x 15r] wrote {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
