"""Does a post-stroke spout's ACTIVITY look like the pre-stroke pattern of its TARGET, of WHERE ITS LICKS LANDED, or of
NEITHER? (Priya 10-21: "we want to know if the brain activity after stroke looks like pre-stroke target vs pre-stroke
execution vs neither".) All four animals, map / pattern level -- the companion of the lick-template test
(scripts/lick_template_test.py, PS92 / PS93 in 3-D).

BRAIN: figure 10b's destination -- for each post-stroke session and aimed spout, the pre-stroke spout whose activity
pattern it matches best (`grant_figures._matrices_best_match_destination`: argmax of the frozen pattern-correlation row,
one-hot per session; pre = per-session leave-one-out one-hots). Pure brain activity, labelled by the COMMANDED spout.
BEHAVIOUR: where that session's licks to that spout LANDED -- cam4 peak angle + protrusion, soft-assigned to the animal's
PRE-STROKE landing centres (median per spout over the August pre sessions; a pre session uses the centres without
itself; distance in pooled within-spout SD units; weights as combined_encoding.exec_bins: exp(-d^2 / 2), rows sum to 1).
Engaged trials (reference gate), licks cue .. 3.5 s.
ROW CLASS (session x aimed spout): target-like = best match is its own spout; execution-like = best match is another spout
AND that spout is where most licks landed; neither = best match is another spout that is not the landing mode.
exec_mass = landing weight at the best-match spout (how much of the licking went where the brain pattern points).

Output <run>/imaging_encoding/qc/best_match_destination.csv (+ _summary.csv).
Usage: WIDEFIELD_PRE_FROM=0801 PYTHONPATH=. python scripts/best_match_destination.py [--arm cue/working]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

RESP_MS = 3500.0


def landing(labels: list) -> dict:
    """{label: {aimed raw position: soft landing weights over grant_kit.POS}} from cam4 licks."""
    from wfield_local import epochs
    from wfield_local import grant_kit as gk
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    L = {}
    for lab in labels:
        a, d = lab[:4], "2026" + lab[5:]
        sd = oo.session_dir(rv, oc.RUN, a, d)
        if not (sd / "licks.csv").exists():
            continue
        K = pd.read_csv(sd / "licks.csv", usecols=["trial_id", "position", "t_ms", "peak_angle_deg", "protrusion_px"])
        keep, _g = oc.engagement_keep(a, d, sd, gate="reference")
        if keep is not None:
            K = K[K.trial_id.isin(keep)]
        L[lab] = K[(K.t_ms >= 0) & (K.t_ms <= RESP_MS)].dropna(subset=["peak_angle_deg", "protrusion_px"])
    out = {}
    for an in sorted({x[:4] for x in L}):
        pre = [x for x in L if x[:4] == an and epochs.epoch_of(x) == "pre" and x[-4:] >= "0806"]
        for lab in [x for x in L if x[:4] == an]:
            ref = pd.concat([L[x] for x in pre if x != lab], ignore_index=True)
            if ref.empty:
                continue
            cen = ref.groupby("position")[["peak_angle_deg", "protrusion_px"]].median().reindex(gk.POS)
            dev = ref[["peak_angle_deg", "protrusion_px"]] - ref.groupby("position")[
                ["peak_angle_deg", "protrusion_px"]].transform("median")
            sc = np.maximum(dev.std().to_numpy(float), 1e-9)
            G = cen.to_numpy(float) / sc
            K = L[lab]
            X = K[["peak_angle_deg", "protrusion_px"]].to_numpy(float) / sc
            d2 = ((X[:, None, :] - G[None]) ** 2).sum(2)
            d2 = np.where(np.isfinite(d2), d2, np.inf)
            w = np.exp(-0.5 * (d2 - d2.min(1, keepdims=True)))
            w = w / w.sum(1, keepdims=True)
            out[lab] = {p: w[(K.position == p).to_numpy()].mean(0) for p in gk.POS if (K.position == p).sum() >= 5}
    return out


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import grant_figures as gf
    from wfield_local import grant_kit as gk
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--arm", default="cue/working", help="align/variant of the 10b matrices (cue/working, lick/lick)")
    a = ap.parse_args(argv)
    align, variant = a.arm.split("/")
    out = oo.root(PathResolver()) / a.run / "imaging_encoding" / "qc"
    writeguard.assert_writable(out)
    labels = [x for x in ak.curated_labels() if x[-4:] >= "0806"]
    lab_of = {(x[:4], epochs.days_since_stroke(x)): x for x in labels}
    Lw = landing(labels)
    dest, _days = gf._matrices_best_match_destination(align, variant)
    rows = []
    for an, by in dest.items():
        for day, H in by.items():
            if day == "PRE":
                continue
            lab = lab_of.get((an, int(day)))
            if lab is None or lab not in Lw:
                continue
            for i, p in enumerate(gk.POS):
                if not np.isfinite(H[i]).any() or p not in Lw[lab]:
                    continue
                j = int(np.nanargmax(H[i]))
                w = Lw[lab][p]
                mode = int(np.argmax(w))
                cls = "target" if j == i else ("execution" if j == mode else "neither")
                rows.append({"label": lab, "animal": an, "epoch": epochs.epoch_of(lab), "day": int(day),
                             "aimed": oc.pos_key(p, "L"), "best_match": oc.pos_key(gk.POS[j], "L"),
                             "landing_mode": oc.pos_key(gk.POS[mode], "L"), "own_land": float(w[i]),
                             "exec_mass": float(w[j]), "cls": cls})
    D = pd.DataFrame(rows)
    sfx = "" if a.arm == "cue/working" else "_" + a.arm.replace("/", "_")
    D.to_csv(out / f"best_match_destination{sfx}.csv", index=False, float_format="%.4g")
    order = ["acute", "subacute", "chronic"]
    pd.set_option("display.width", 220)
    S = D.groupby(["animal", "epoch", "aimed"]).agg(n=("cls", "size"), target=("cls", lambda c: (c == "target").mean()),
                                                     execution=("cls", lambda c: (c == "execution").mean()),
                                                     neither=("cls", lambda c: (c == "neither").mean()),
                                                     own_land=("own_land", "mean"), exec_mass=("exec_mass", "mean"))
    S.to_csv(out / f"best_match_destination{sfx}_summary.csv", float_format="%.3g")
    for p in ("far_contra", "far_mid"):
        t = S.xs(p, level="aimed")[["n", "target", "execution", "neither", "own_land"]]
        t = t.reindex([(an, ep) for an in sorted(D.animal.unique()) for ep in order if (an, ep) in t.index])
        print(f"\n  {p}: share of sessions whose activity best matches the pre-stroke TARGET / the LANDING zone /"
              f" NEITHER ({a.arm}); own_land = licks landing in their own zone\n" + t.round(2).to_string())
    allp = D.groupby(["epoch"]).cls.value_counts(normalize=True).unstack().reindex(order)
    print("\n  all spouts, all animals\n" + allp.round(2).to_string())
    print(f"\n-> {out / f'best_match_destination{sfx}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
