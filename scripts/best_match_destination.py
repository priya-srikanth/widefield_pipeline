"""Does a post-stroke spout's ACTIVITY look like the pre-stroke pattern of its TARGET, of WHERE ITS LICKS LANDED, or of
NEITHER? (Priya 10-21: "we want to know if the brain activity after stroke looks like pre-stroke target vs pre-stroke
execution vs neither".) PS92 / PS93 (3-D landing zones), map / pattern level -- the companion of the lick-template test
(scripts/lick_template_test.py, PS92 / PS93 in 3-D).

BRAIN: figure 10b's destination -- for each post-stroke session and aimed spout, the pre-stroke spout whose activity
pattern it matches best (`grant_figures._matrices_best_match_destination`: argmax of the frozen pattern-correlation row,
one-hot per session; pre = per-session leave-one-out one-hots). Pure brain activity, labelled by the COMMANDED spout.
BEHAVIOUR: where that session's licks to that spout LANDED -- the 3-D calibrated landing zones (`landing`, CONTEST_3D);
so PS92 / PS93 until PS94 / PS95 have 3-D.
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

from wfield_local.orofacial_cohort import POS_ORDER as POS   # canonical spout order (the 3-D weights' keys)


#: 3-D landing zones (Priya 10-21: "we should be using 3d for the lick zone based analyses"): the soft weights the 3-D
#: polar contest stores per session (`bins.soft_weight`, calibrated pre-stroke landing centres, leave-one-session-out for
#: pre sessions; combined_encoding.exec_bins) -- ONE landing definition with the binned contest and the template test.
#: cam4 angle + reach zones were tried first and FAILED the pre-stroke check (own-zone weight 0.17-0.63, PS93 ~chance).
CONTEST_3D = "movement_first_subset-nokin_exec-polar_licks-combined_bins"


def landing(E) -> dict:
    """{label: {aimed canonical position: soft landing weights over POS}} from the 3-D contest session JSONs."""
    import json
    out = {}
    for f in sorted((E / CONTEST_3D / "sessions").glob("*.json")):
        js = json.loads(f.read_text())
        sw = (js.get("bins") or {}).get("soft_weight") or {}
        if (js.get("bins") or {}).get("frame") != "3d_calibrated" or not sw:
            continue
        out[js["label"]] = {t: np.array([w.get(z, 0.0) for z in POS]) for t, w in sw.items()}
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
    E = oo.root(PathResolver()) / a.run / "imaging_encoding"
    out = E / "qc"
    writeguard.assert_writable(out)
    labels = [x for x in ak.curated_labels() if x[-4:] >= "0806"]
    lab_of = {(x[:4], epochs.days_since_stroke(x)): x for x in labels}
    Lw = landing(E)
    dest, _days = gf._matrices_best_match_destination(align, variant)
    rows = []
    for an, by in dest.items():
        for day, H in by.items():
            if day == "PRE":
                continue
            lab = lab_of.get((an, int(day)))
            if lab is None or lab not in Lw:
                continue
            for i, p in enumerate(gk.POS):                     # grant_kit raw order -> canonical names
                q = oc.pos_key(p, "L")
                if not np.isfinite(H[i]).any() or q not in Lw[lab]:
                    continue
                j = int(np.nanargmax(H[i]))
                qj = oc.pos_key(gk.POS[j], "L")
                w = Lw[lab][q]
                mode = POS[int(np.argmax(w))]
                cls = "target" if qj == q else ("execution" if qj == mode else "neither")
                rows.append({"label": lab, "animal": an, "epoch": epochs.epoch_of(lab), "day": int(day),
                             "aimed": q, "best_match": qj, "landing_mode": mode, "own_land": float(w[POS.index(q)]),
                             "exec_mass": float(w[POS.index(qj)]), "cls": cls})
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
