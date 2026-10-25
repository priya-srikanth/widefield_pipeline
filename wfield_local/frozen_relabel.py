"""RELABEL TEST: is cortex on a spout's trials in THAT spout's state, or in another spout's? (Priya 10-23, on the acute
far-contra trials: does the brain look like "trying to do far contra, tongue came out far mid" or "trying to do far mid,
and the tongue does far mid"?)

The frozen pre-stroke encoder (joint, v7 frame) predicts each session with its true spout labels. Here every spout q's
trials are additionally predicted AS IF they were spout j's: q's trials get j's TARGET columns and j's pre-stroke
SPOUT-LOCKED templates, while the executed traces stay exactly as recorded (template + deviation = the recorded trace;
`frozen_encoder.session_design(relabel=)`). Every cyclic shift of the six labels gives all 30 (q -> j) relabellings in 5
extra designs per session. Score = held-out SSE on q's OWN trial frames (all / cue / lick windows; cortex + areas), pre
sessions with their leave-one-session-out betas, post sessions with the frozen all-pre betas.

  dSSE(q -> j) = (SSE[q as j] - SSE[q as q]) / SST[q]     > 0: the TRUE label fits better (cortex is in q's state)
                                                          < 0: label j fits better (cortex looks like spout j)

Pre-stroke this is the validation (the true label should win everywhere). Post-stroke, far contra: A ("far contra,
execution off") predicts far contra still wins; B ("re-targeted to far mid") predicts far mid fits better.
Caveat: the executed traces carry their own spout structure (the templates only take their spout-locked part), so a
relabel moves only the target + spout-locked parts -- the test asks which spout's pre-stroke TARGET + TYPICAL-MOVEMENT
pattern the cortex is closer to, given the licks actually made.

Usage: python -m wfield_local.frozen_relabel --from gate-attempted_v7fwd15 --tag v7 [--jobs 2] [--labels ...]
Output: <E>/frozen_relabel_<tag>/{sessions/<label>.json, relabel_by_session.csv, relabel_by_animal.csv}
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

WINDOWS = ("all", "cue", "lick")


ALL = "all_areas"


def all_areas_dsse(sse_as: np.ndarray, sse_true: np.ndarray, sst: np.ndarray, idx) -> float:
    """Relabel cost over several areas jointly: summed SSE increase / summed SST (fraction of variance lost)."""
    return float((sse_as[idx] - sse_true[idx]).sum() / max(float(sst[idx].sum()), 1e-12))


def shifts(pos_order: list) -> list[dict]:
    """The 5 non-identity cyclic relabellings {q: j}; together they map every q to every other j once."""
    n = len(pos_order)
    return [{q: pos_order[(i + k) % n] for i, q in enumerate(pos_order)} for k in range(1, n)]


def relabel_job(item) -> dict:
    """One session: SSE on each true spout's frames under its own label and under every other spout's label."""
    from wfield_local import config
    from wfield_local import frozen_encoder as fe
    label, fit_path, tpath = item
    an = config.animal_of(label)
    fit = pickle.loads(Path(fit_path).read_bytes())[an]
    templates = fe._templates_of(tpath, an)
    beta = fit["beta_loso"].get(label, fit["beta_all"])
    d0 = fe.session_design(label, templates)
    if len(d0["col_var"]) != len(fit["col_var"]) or (d0["col_var"] != fit["col_var"]).any():
        raise ValueError(f"{label}: column layout differs from the animal's pre fit")
    ix = [d0["area_names"].index(a) for a in fit["areas"]]
    Rd = d0["Rd"][:, ix]
    fpos = d0["frame_pos"]
    order = list(d0["pos_order"])
    masks = {(w, q): d0["win"][w] & (fpos == q) for w in WINDOWS for q in order}

    # PER TRIAL (Priya 10-24: encoder match vs executed best bin, trial by trial): each trial's SSE in the lick and cue
    # windows, summed over the parcels (cortex mean excluded, as all_areas_dsse), under every label
    ftri = d0["frame_trial"]
    ia = np.array([a != "cortex" for a in fit["areas"]])
    tw = {w: [(int(d0["trial_id"][t]), np.flatnonzero(d0["win"][w] & (ftri == t))) for t in range(len(d0["trial_id"]))]
          for w in ("lick", "cue")}
    trial_sse: dict = {w: {} for w in tw}

    def sse(d, lab_of=None):
        P = (d["X"] / fit["scale"]) @ (fit["C"] @ beta)
        E2 = (Rd - P) ** 2
        for w, lst in tw.items():
            for tid, fr in lst:
                if len(fr) >= 5:
                    q = fpos[fr[0]]
                    lab = q if lab_of is None else lab_of.get(q, q)
                    trial_sse[w].setdefault(str(tid), {"spout": q})[lab] = float(E2[np.ix_(fr, ia)].sum())
        return {f"{w}|{q}": (E2[m].sum(0)).tolist() for (w, q), m in masks.items() if m.any()}
    out = {"label": label, "animal": an, "epoch": d0["epoch"], "areas": fit["areas"], "pos_order": order,
           "frozen_from": "loso" if label in fit["beta_loso"] else "all_pre",
           "sst": {f"{w}|{q}": (((Rd[m] - Rd[m].mean(0)) ** 2).sum(0)).tolist() for (w, q), m in masks.items() if m.any()},
           "sse": {"true": sse(d0)}}
    for rl in shifts(order):
        s = sse(fe.session_design(label, templates, relabel=rl), lab_of=rl)
        for q, j in rl.items():
            out["sse"].setdefault(f"as:{j}", {}).update({k: v for k, v in s.items() if k.endswith(f"|{q}")})
    out["trial_sse"] = trial_sse
    return out


def summarise(out: Path) -> None:
    import pandas as pd
    rows = []
    for js in sorted((out / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        for key, sst in m["sst"].items():
            w, q = key.split("|")
            base = np.asarray(m["sse"]["true"][key])
            for lab, ss in m["sse"].items():
                if lab == "true" or key not in ss:
                    continue
                j = lab.split(":", 1)[1]
                d = (np.asarray(ss[key]) - base) / np.maximum(np.asarray(sst), 1e-12)
                for ai, ar in enumerate(m["areas"]):
                    rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "window": w,
                                 "spout": q, "as": j, "area": ar, "dsse_x1e3": d[ai] * 1e3})
                # ALL AREAS jointly (smoke 10-23: the whole-cortex MEAN trace averages away the left / right pattern
                # that tells spouts apart and fails the pre check; summed SSE over the parcels keeps it)
                ia = [i for i, ar in enumerate(m["areas"]) if ar != "cortex"]
                rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "window": w,
                             "spout": q, "as": j, "area": ALL,
                             "dsse_x1e3": all_areas_dsse(np.asarray(ss[key]), base, np.asarray(sst), ia) * 1e3})
    T = pd.DataFrame(rows)
    T.to_csv(out / "relabel_by_session.csv", index=False, float_format="%.5g")
    order = ["pre", "acute", "subacute", "chronic"]
    A = T.groupby(["animal", "epoch", "window", "spout", "as", "area"]).dsse_x1e3.mean().reset_index()
    A.to_csv(out / "relabel_by_animal.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    V = A[(A.area == ALL) & (A.epoch == "pre")]
    if not V.empty:
        V = V.assign(true_wins=V.dsse_x1e3 > 0)
        print("\n=== VALIDATION (pre, all areas): fraction of (q -> j) relabellings where the TRUE label fits better\n"
              + V.pivot_table(index="animal", columns="window", values="true_wins").round(2).to_string())
        for an in sorted(V.animal.unique()):
            M = V[(V.animal == an) & (V.window == "lick")].pivot_table(index="spout", columns="as", values="dsse_x1e3")
            print(f"\n  {an} pre, lick window, all areas: rows = true spout, cols = relabelled as (x1e3 of SST; "
                  "small = weakly separated)\n" + M.round(1).to_string())
    for w in ("lick", "cue"):
        for area in (ALL, "SSp-m_left"):
            S = A[(A.window == w) & (A.area == area) & (A.spout == "far_contra")]
            if S.empty:
                continue
            P = S.pivot_table(index=["animal", "epoch"], columns="as", values="dsse_x1e3")
            P = P.reindex(sorted(P.index, key=lambda k: (k[0], order.index(k[1]) if k[1] in order else 9)))
            print(f"\n=== RELABEL far_contra trials as spout j, {w} window, {area}: dSSE x1e3 (> 0 = TRUE label fits "
                  "better; < 0 = j fits better)\n" + P.round(2).to_string())
    B = T[(T.area == ALL) & (T.window == "lick")]
    best = B.loc[B.groupby(["label", "spout"]).dsse_x1e3.idxmin()]
    best = best.assign(winner=np.where(best.dsse_x1e3 < 0, best["as"], best.spout))
    W = best.groupby(["animal", "epoch", "spout"]).winner.agg(lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items()))
    print("\n=== best-fitting label per session (all areas, lick window): true unless another label lowers SSE\n"
          + W.unstack("spout").reindex(columns=sorted(B.spout.unique())).to_string())
    print(f"\n-> {out}")


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import frozen_encoder as fe
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args(argv)
    E = oo.root(PathResolver()) / oc.RUN / "imaging_encoding"
    src = E / f"frozen_encoder_{a.src}"
    out = E / f"frozen_relabel_{a.tag}"
    assert_writable(out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    if a.summarise_only:
        summarise(out)
        return 0
    pre_stats = pickle.loads((src / "fit" / "pre_stats.pkl").read_bytes())
    fp = out / "joint_fits.pkl"
    if not fp.exists():
        fits = {}
        for an in sorted({s["animal"] for s in pre_stats}):
            f = fe.fit_animal([s for s in pre_stats if s["animal"] == an], variant="joint")
            fits[an] = {k: f[k] for k in ("scale", "C", "col_var", "areas", "beta_all", "beta_loso", "alphas")}
            print(f"  {an}: joint pre fit, LOSO R2 {f['loso_r2']:.3f}", flush=True)
        fp.write_bytes(pickle.dumps(fits, protocol=pickle.HIGHEST_PROTOCOL))
    cur = set(ak.curated_labels())
    labels = a.labels or [x for x in sorted(ce.sessions_available()) if x in cur and x[-4:] >= "0806"]
    done = {js.stem for js in (out / "sessions").glob("*.json")}
    todo = [x for x in labels if x not in done]
    got, fail = ak.fan_sessions([(x, str(fp), str(src / "fit" / "templates.pkl")) for x in todo], relabel_job,
                                jobs=a.jobs, label="relabel")
    for _it, r in got:
        (out / "sessions" / f"{r['label']}.json").write_text(json.dumps(r, default=float), encoding="utf-8")
    for it, err in fail:
        print(f"  !! {it}: {err}", flush=True)
    summarise(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
