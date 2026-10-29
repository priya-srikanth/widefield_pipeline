"""First-lick latency three ways (Priya 10-20: "these median first lick times are based on keypoint tracking right? not
spout contact"): per trial, time from the cue to the first
  cam4      LP cam4 lick onset (session licks.csv `on_ms`; what the lick-aligned within-spout check uses)
  combined  earliest of cam4 and the cam1 2-D detector's events (session_poses/cam1_licks/<a>_<d>_events.csv, droplets
            excluded) -- small / incomplete licks cam4 misses
  contact   first DAQ spout contact
A late cam4 first lick with an earlier cam1 event = attempts cam4 did not register as licks; a late contact with an
early lick = licks that miss the spout. Curated sessions from 08-06; cam1 columns only where events exist.
Output: <run>/imaging_encoding/qc/first_lick_latency.csv (per trial) + a per animal x epoch x spout summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPOCHS = ("pre", "acute", "subacute", "chronic")


def session_latency(label: str) -> pd.DataFrame:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    import scripts.pose_kinematics_demo as D
    rv = PathResolver()
    a, d = label.split("_")[0], "2026" + label[5:]
    b = TW.trial_bounds(a, d, rv)
    pos = dict(zip(b.trial_id.astype(int), [oc.pos_key(p, "L") for p in b.pos_name]))
    stop = dict(zip(b.trial_id.astype(int), (b.stop_s - b.cue_s).to_numpy(float) * 1000))
    L = pd.read_csv(oo.session_dir(rv, oc.RUN, a, d) / "licks.csv", usecols=["trial_id", "on_ms"])
    cam4 = L[L.on_ms >= 0].groupby("trial_id").on_ms.min()
    f = Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / f"{a}_{d}_events.csv"
    cam1 = None
    if f.exists():
        E = pd.read_csv(f, usecols=["trial_id", "on_ms", "droplet"])
        E = E[(E.on_ms >= 0) & ~E.droplet.fillna(False).astype(bool)]
        cam1 = E.groupby("trial_id").on_ms.min()
    con = D.daq_contacts_ms(a, d, b, rv)
    rows = []
    for t in b.trial_id.astype(int):
        c4 = cam4.get(t, np.nan)
        c1 = cam1.get(t, np.nan) if cam1 is not None else np.nan
        ct = [x for x in np.asarray(con.get(t, []), float) if x >= 0]
        rows.append({"label": label, "trial_id": t, "pos": pos[t], "stop_ms": stop[t], "cam4": c4,
                     "combined": np.nanmin([c4, c1]) if cam1 is not None else np.nan,
                     "contact": min(ct) if ct else np.nan, "has_cam1": cam1 is not None})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / a.run / "imaging_encoding" / "qc"
    writeguard.assert_writable(out)
    cur = set(ak.curated_labels())
    labels = [x for x in ce.sessions_available() if x in cur and x[-4:] >= "0806"]
    got, fail = ak.fan_sessions(labels, session_latency, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED {it}: {err}")
    D = pd.concat([r for _l, r in got], ignore_index=True)
    D["animal"], D["epoch"] = D.label.str.split("_").str[0], D.label.map(epochs.epoch_of)
    D.to_csv(out / "first_lick_latency.csv", index=False, float_format="%.1f")
    rows = []
    for (an, ep, p), g in D.groupby(["animal", "epoch", "pos"]):
        r = {"animal": an, "epoch": ep, "pos": p, "n_trials": len(g), "has_cam1": bool(g.has_cam1.all())}
        for m in ("cam4", "combined", "contact"):
            x = g[m]
            r[f"{m}_med_s"] = x.median() / 1000
            r[f"{m}_late"] = float(((x > 1500) | x.isna()).mean())     # no lick by 1.5 s (incl. none at all)
            r[f"{m}_none"] = float(x.isna().mean())
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(out / "first_lick_latency_summary.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    F = S[S.pos.isin(["far_contra", "far_mid", "close_contra"])]
    for c in ("med_s", "late"):
        t = F.pivot_table(index=["animal", "pos"], columns="epoch",
                          values=[f"cam4_{c}", f"combined_{c}", f"contact_{c}"]).round(2)
        print(f"\n  first-lick {c} (late = no lick by 1.5 s, incl. none)\n"
              + t.reindex(columns=list(EPOCHS), level=1).to_string())
    print(f"\n-> {out / 'first_lick_latency_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
