"""Does spout CONTACT carry the frozen-decoder contrast of page 3.8 / 3.9 (ITI licks stay remapped, in-trial licks
recover)? (handoff 10-26 B(i): "in-trial recovery may ride on spout contact".)

Each lick's DAQ `contact` flag from the session's licks_all.csv (joined on the exact DAQ peak time, which
`scripts/iti_licks.py` stores as iti_t / tr_t). The frozen decoder is the one of `iti_nmatch_check.py` / page 3.9:
trained per animal on ALL its pre-stroke licks of the set (pre sessions leave-one-session-out), lags 0..0.2 s, all Allen
areas; it is then SCORED on contact and no-contact licks separately (r per session, sessions with >= MIN_N licks of the
subset). ITI licks touch the spout too (it is still approaching or leaving at the window edges), so both sets are split.
Writes <E>/iti_licks/contact_check.csv; prints the per-animal table (rule 8) and nested-bootstrap changes from pre.
Usage: python scripts/iti_contact_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
MIN_N = 15


def contact_of(label: str, t: np.ndarray) -> np.ndarray:
    """DAQ contact flag of each lick (exact DAQ peak time match into licks_all.csv); NaN where unmatched."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["daq_peak_s", "contact"])
    A = A.dropna(subset=["daq_peak_s"]).drop_duplicates("daq_peak_s").set_index("daq_peak_s").contact
    c = A.reindex(np.asarray(t, float)).map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return c.to_numpy(float)


def main() -> int:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from wfield_local import analysis_kit as ak
    W = (il.LAGS_D >= 0.0) & (il.LAGS_D <= 0.2)
    labels = sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))
    D = {x: il._load(x) for x in labels}
    out = []
    for st, kx, ky, kt in (("iti", "iti_X", "iti_side", "iti_t"), ("trial", "tr_X", "tr_side", "tr_t")):
        for a in AN:
            S = {x: d for x, d in D.items() if str(d["animal"]) == a and len(d[ky]) >= 20}
            pre = [x for x, d in S.items() if str(d["epoch"]) == "pre"]
            if len(pre) < 3:
                continue
            feat = [i for i, c in enumerate(next(iter(S.values()))["cols"])
                    if c != "cortex" and not str(c).startswith(("ALM", "tjM1"))]
            X = {x: S[x][kx].astype(float)[:, W][:, :, feat].mean(1) for x in S}
            fits = {}
            for held in [*pre, None]:
                trn = [q for q in pre if q != held]
                Xt = np.vstack([X[q] for q in trn])
                sc = StandardScaler().fit(Xt)
                fits[held] = (sc, RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(
                    sc.transform(Xt), np.concatenate([S[q][ky] for q in trn])))
            for x in S:
                sc, m = fits[x if x in pre else None]
                pr, y = m.predict(sc.transform(X[x])), S[x][ky]
                c = contact_of(x, S[x][kt])
                row = {"set": st, "label": x, "animal": a, "epoch": str(S[x]["epoch"]), "n": len(y),
                       "share_contact": float(np.nanmean(c)), "share_matched": float(np.isfinite(c).mean()),
                       "r_all": float(np.corrcoef(pr, y)[0, 1])}
                for nm, mk in (("contact", c == 1), ("nocontact", c == 0)):
                    row[f"n_{nm}"] = int(mk.sum())
                    row[f"r_{nm}"] = float(np.corrcoef(pr[mk], y[mk])[0, 1]) if mk.sum() >= MIN_N else np.nan
                out.append(row)
    R = pd.DataFrame(out)
    R.to_csv(il.OUT / "contact_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("== matched share (licks found in licks_all by exact peak time):", R.share_matched.describe().round(3).to_dict())
    for st in ("iti", "trial"):
        t = R[R.set == st]
        print(f"\n== {st}: share of licks with spout contact (median per animal x epoch)")
        print(t.groupby(["animal", "epoch"]).share_contact.median().unstack().reindex(columns=EP).round(2).to_string())
        for val in ("r_all", "r_contact", "r_nocontact"):
            print(f"-- {st} {val}: per animal mean (rule 8)")
            print(t.groupby(["animal", "epoch"])[val].mean().unstack().reindex(columns=EP).round(3).to_string())
            ref = {a: t[(t.animal == a) & (t.epoch == "pre")][val].dropna().to_numpy() for a in AN}
            cells = []
            for ep in EP[1:]:
                post = {a: t[(t.animal == a) & (t.epoch == ep)][val].dropna().to_numpy() for a in AN}
                com = [a for a in AN if len(ref[a]) and len(post[a])]
                if len(com) < 3:
                    cells.append(f"{ep} n/a ({len(com)} animals)")
                    continue
                iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
                cells.append(f"{ep} {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}] ({len(com)} an)")
            print("   change vs pre: " + "; ".join(cells))
    # B(ii): SESSION-PAIRED ITI - in-trial difference (same session, same areas, same decoder recipe), change from pre
    P = R.pivot_table(index=["label", "animal", "epoch"], columns="set", values=["r_all", "r_nocontact"]).dropna(how="any")
    for val in ("r_all", "r_nocontact"):
        dd = (P[(val, "iti")] - P[(val, "trial")]).rename("d").reset_index()
        print(f"\n== ITI - in-trial, session-paired ({val}): per animal mean (rule 8)")
        print(dd.groupby(["animal", "epoch"]).d.mean().unstack().reindex(columns=EP).round(3).to_string())
        ref = {a: dd[(dd.animal == a) & (dd.epoch == "pre")].d.to_numpy() for a in AN}
        cells = []
        for ep in EP[1:]:
            post = {a: dd[(dd.animal == a) & (dd.epoch == ep)].d.to_numpy() for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
            cells.append(f"{ep} {iv.point:+.3f} [{iv.lo:+.3f}, {iv.hi:+.3f}] ({len(com)} an)")
        print("   change vs pre: " + "; ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
