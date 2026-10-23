"""ALL-SESSION ENCODER: shared kernels + per-epoch kernel CHANGES (Layer 2; Priya 10-23: "i like the shared kernels with
per-epoch changes idea!", then "should move up layer 2" -- the frozen encoder splits a cortical change into the parts
each BEHAVIOURAL change explains through the pre-stroke mapping (Layer 1) and leaves any change in how cortex RESPONDS
in one unattributed residual; this model attributes that residual to groups).

Per animal, on ALL its sessions (August pre + post), in the frozen encoder's frame (same columns, the pre-stroke
spout-locked templates, the pre-stroke column scaling; `frozen_encoder.session_design`):

    y = X beta  +  sum_e 1[session in epoch e] X dbeta_e          e in acute / subacute / chronic (pre = reference)

beta = kernels shared by every session; dbeta_e = how every kernel CHANGED in epoch e (every column has one, so every
group can change). Ridge: beta with the frozen fit's per-penalty-group alphas, dbeta_e with KAPPA x those alphas (kappa
chosen by held-out R^2 on the post sessions), so changes shrink to "no change" unless the data need them. The
augmented Gram is assembled exactly from per-session sufficient statistics (session k in epoch e fills blocks
(0,0), (0,e), (e,0), (e,e) with its X'X), so no pass over the imaging beyond the post sessions' statistics.

READOUTS (all leave-one-session-out held out; per animal x epoch; cortex + every area; all frames):
  change      R2(shared + all changes) - R2(shared only)          how much the epoch's kernel changes matter at all
  change_<g>  R2(all changes) - R2(all changes except group g's)  how much cortex's RESPONSE to group g changed
  dbeta sums  per group x epoch x area, summed over lags (raw units, x1e3): the sign / size of each kernel change
DRIFT NULL: the same fit on the PRE sessions only with the later half of them as a pseudo-epoch ("pre_late"):
  its change / change_<g> = what day-to-day drift alone produces.
Limits: areas, all frames (window readouts need a prediction pass); an epoch with ONE session (PS95 acute) cannot be
held out -- its change is then pure shrinkage (flagged n_epoch = 1).

Usage: python -m wfield_local.allsession_encoder --from gate-attempted_v7pushfree --tag v7push [--jobs 2] [--reuse-stats]
  (`--from` = a frozen run whose fit/templates.pkl + fit/pre_stats.pkl define the frame; run with the same running /
  push environment, e.g. FROZEN_PUSH=1 -- checked against the saved run tag.)
Output: <E>/allsession_encoder_<tag>/{allsession_scores.csv, allsession_dbeta.csv, fit/post_stats.pkl, fit/<an>.npz}
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

KAPPA_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
POST_EPOCHS = ("acute", "subacute", "chronic")


def _fe():
    from wfield_local import frozen_encoder as fe
    return fe


def augment(G: np.ndarray, b: np.ndarray, e: int, n_ep: int) -> tuple[np.ndarray, np.ndarray]:
    """One session's (X'X, X'Y) in the augmented layout [beta | dbeta_1 .. dbeta_n_ep]; e = 0 for the reference epoch
    (no change block), 1..n_ep otherwise."""
    p = G.shape[0]
    Ga = np.zeros(((n_ep + 1) * p, (n_ep + 1) * p))
    ba = np.zeros(((n_ep + 1) * p, b.shape[1]))
    blocks = [0] if e == 0 else [0, e]
    for i in blocks:
        ba[i * p:(i + 1) * p] = b
        for j in blocks:
            Ga[i * p:(i + 1) * p, j * p:(j + 1) * p] = G
    return Ga, ba


def ridge(G: np.ndarray, b: np.ndarray, lam: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Ridge on the columns in ``keep`` (others zero)."""
    keep = np.ones(len(lam), bool) if keep is None else keep
    beta = np.zeros((len(lam), b.shape[1]))
    ix = np.flatnonzero(keep)
    beta[ix] = np.linalg.solve(G[np.ix_(ix, ix)] + np.diag(lam[ix]) + 1e-8 * np.eye(len(ix)), b[ix])
    return beta


def sse(beta: np.ndarray, G: np.ndarray, b: np.ndarray, yy: np.ndarray) -> np.ndarray:
    return yy - 2 * (beta * b).sum(0) + np.einsum("pr,pq,qr->r", beta, G, beta)


def fit_animal(sessions: list[dict], alphas: dict, groups_of: np.ndarray, pen_of: np.ndarray, ok: np.ndarray,
               epochs_ref: tuple, kappa: float | None = None) -> dict:
    """sessions: [{"label", "epoch", "G", "b", "yy"}] (scaled design, common areas); epochs_ref = the change epochs
    (sessions of any other epoch are the reference). Returns kappa, held-out SSE per session x model, beta, dbeta."""
    n_ep, p = len(epochs_ref), len(groups_of)
    eidx = {e: i + 1 for i, e in enumerate(epochs_ref)}
    aug = [augment(s["G"], s["b"], eidx.get(s["epoch"], 0), n_ep) for s in sessions]
    Gt = sum(a[0] for a in aug)
    bt = sum(a[1] for a in aug)
    lam0 = np.array([alphas[g] for g in pen_of]) * ok + (~ok) * 1.0
    grp = np.tile(groups_of, n_ep + 1)
    is_d = np.repeat(np.arange(n_ep + 1) > 0, p)
    models = {"shared": ~is_d, "all": np.ones(len(grp), bool)}
    for g in sorted(set(groups_of)):
        models[f"no_d_{g}"] = ~(is_d & (grp == g))
    change_sessions = [k for k, s in enumerate(sessions) if s["epoch"] in eidx]

    def lam_of(kap):
        return np.concatenate([lam0] + [kap * lam0] * n_ep)

    def held_out(k, lam, keep):
        Gk, bk = aug[k]
        beta = ridge(Gt - Gk, bt - bk, lam, keep)
        return sse(beta, Gk, bk, sessions[k]["yy"])
    if kappa is None:                                    # kappa by held-out SSE over the change-epoch sessions
        def score(kap):
            lam = lam_of(kap)
            return sum(held_out(k, lam, models["all"]).sum() for k in change_sessions)
        kappa = min(KAPPA_GRID, key=score)
    lam = lam_of(kappa)
    out = {m: {sessions[k]["label"]: held_out(k, lam, keep) for k in range(len(sessions))} for m, keep in models.items()}
    full = ridge(Gt, bt, lam)
    return {"kappa": kappa, "sse": out, "beta": full[:p], "dbeta": {e: full[i * p:(i + 1) * p] for e, i in eidx.items()},
            "models": list(models)}


def main(argv=None) -> int:
    import pandas as pd

    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    fe = _fe()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, help="frozen run tag whose templates / pre stats define the frame")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--reuse-stats", action="store_true")
    ap.add_argument("--animals", nargs="*", default=None)
    a = ap.parse_args(argv)
    E = oo.root(PathResolver()) / oc.RUN / "imaging_encoding"
    src = E / f"frozen_encoder_{a.src}"
    out = E / f"allsession_encoder_{a.tag}"
    assert_writable(out)
    (out / "fit").mkdir(parents=True, exist_ok=True)
    pre_stats = pickle.loads((src / "fit" / "pre_stats.pkl").read_bytes())
    tags = {s.get("run_mode") for s in pre_stats}
    if tags != {fe._run_tag()} or {s.get("stats_version") for s in pre_stats} != {fe.STATS_VERSION}:
        raise SystemExit(f"{src}: pre statistics are {tags} / v{ {s.get('stats_version') for s in pre_stats} }, this "
                         f"environment is {fe._run_tag()} / v{fe.STATS_VERSION} -- set the same FROZEN_* environment")
    tp = src / "fit" / "templates.pkl"
    cur = set(ak.curated_labels())
    post = [x for x in sorted(ce.sessions_available()) if x in cur and x[-4:] >= "0806" and epochs.epoch_of(x) != "pre"]
    if a.animals:
        post = [x for x in post if config.animal_of(x) in a.animals]
    sp = out / "fit" / "post_stats.pkl"
    if a.reuse_stats and sp.exists():
        post_stats = pickle.loads(sp.read_bytes())
    else:
        got, fail = ak.fan_sessions([(x, str(tp)) for x in post], fe.stats_job, jobs=a.jobs, label="post stats")
        for it, err in fail:
            print(f"  !! stats {it}: {err}", flush=True)
        post_stats = [s for _it, s in sorted(got, key=lambda kv: kv[1]["label"])]
        for s in post_stats:
            s["stats_version"], s["run_mode"] = fe.STATS_VERSION, fe._run_tag()
        sp.write_bytes(pickle.dumps(post_stats, protocol=pickle.HIGHEST_PROTOCOL))
    rows, drows = [], []
    for an in sorted({s["animal"] for s in pre_stats}):
        if a.animals and an not in a.animals:
            continue
        S = [s for s in pre_stats + post_stats if s["animal"] == an]
        pre_s = [s for s in S if s["epoch"] == "pre"]
        col_var = pre_s[0]["col_var"]
        if any(len(s["col_var"]) != len(col_var) or (s["col_var"] != col_var).any() for s in S):
            raise SystemExit(f"{an}: column layout differs between sessions")
        areas = [x for x in pre_s[0]["area_names"] if all(x in s["area_names"] for s in S)]
        sd = np.sqrt(sum(s["sxx"] for s in pre_s) / sum(s["n"] for s in pre_s))   # the PRE scaling (frozen frame)
        ok = sd > 1e-9
        scale = np.where(ok, sd, np.inf)
        sess = []
        for s in S:
            xy, y2 = fe._subset(s, areas)
            sess.append({"label": s["label"], "epoch": s["epoch"], "G": s["XtX"] / np.outer(scale, scale),
                         "b": xy / scale[:, None], "yy": y2})
        js = next((src / "joint" / "sessions").glob(f"{an}_*.json"))
        alphas = json.loads(js.read_text(encoding="utf-8"))["alphas"]
        groups_of = np.array([fe.FGROUP[v] for v in col_var])
        pen_of = np.array([fe.FPEN[v] for v in col_var])
        sets = {"post": (sess, POST_EPOCHS)}
        pre_sorted = sorted([x for x in sess if x["epoch"] == "pre"], key=lambda x: epochs.days_since_stroke(x["label"]))
        half = len(pre_sorted) // 2
        if half >= 2:                                    # drift null: later half of the pre sessions as a pseudo-epoch
            sets["null"] = ([{**x, "epoch": "pre_late" if i >= half else "pre"} for i, x in enumerate(pre_sorted)],
                            ("pre_late",))
        fits = {k: fit_animal(ss, alphas, groups_of, pen_of, ok, eps) for k, (ss, eps) in sets.items()}
        for fit_name, f in fits.items():
            ss = sets[fit_name][0]
            n_ep = {}
            for x in ss:
                n_ep[x["epoch"]] = n_ep.get(x["epoch"], 0) + 1
            for x in ss:
                for m in f["models"]:
                    for ai, ar in enumerate(areas):
                        rows.append({"fit": fit_name, "animal": an, "label": x["label"], "epoch": x["epoch"],
                                     "n_epoch": n_ep[x["epoch"]], "model": m, "area": ar, "kappa": f["kappa"],
                                     "r2": 1.0 - f["sse"][m][x["label"]][ai] / max(x["yy"][ai], 1e-12)})
            for e, dB in f["dbeta"].items():
                raw = dB / scale[:, None]
                for g in sorted(set(groups_of)):
                    m = groups_of == g
                    for ai, ar in enumerate(areas):
                        drows.append({"fit": fit_name, "animal": an, "epoch": e, "group": g, "area": ar,
                                      "dbeta_sum_x1e3": float(raw[m, ai].sum() * 1e3),
                                      "beta_sum_x1e3": float((f["beta"][m, ai] / scale[m]).sum() * 1e3)})
            np.savez_compressed(out / "fit" / f"{an}_{fit_name}.npz", beta=f["beta"], scale=scale, col_var=col_var,
                                areas=np.array(areas), kappa=f["kappa"],
                                **{f"dbeta_{e}": d for e, d in f["dbeta"].items()})
        print(f"  {an}: kappa post {fits['post']['kappa']:g}"
              + (f", null {fits['null']['kappa']:g}" if "null" in fits else "") + "; sessions per epoch "
              + ", ".join(f"{e} {sum(x['epoch'] == e for x in sess)}" for e in ("pre", *POST_EPOCHS)), flush=True)
    R = pd.DataFrame(rows)
    D = pd.DataFrame(drows)
    R.to_csv(out / "allsession_scores.csv", index=False, float_format="%.5g")
    D.to_csv(out / "allsession_dbeta.csv", index=False, float_format="%.5g")
    summarise(out)
    return 0


def summarise(out: Path) -> None:
    """Per animal x epoch held-out R2 changes (x1e3), cortex and contralesional SSp-m; drift null alongside."""
    import pandas as pd
    R = pd.read_csv(out / "allsession_scores.csv")
    W = R.pivot_table(index=["fit", "animal", "label", "epoch", "area", "n_epoch"], columns="model", values="r2").reset_index()
    W["change"] = (W["all"] - W["shared"]) * 1e3
    gcols = [c for c in W.columns if str(c).startswith("no_d_")]
    for c in gcols:
        W[f"change_{c[5:]}"] = (W["all"] - W[c]) * 1e3
    keep = ["change"] + [f"change_{c[5:]}" for c in gcols]
    pd.set_option("display.width", 250)
    for area in ("cortex", "SSp-m_left"):
        A = W[W.area == area].groupby(["fit", "animal", "epoch"])[keep + ["n_epoch"]].mean().round(2)
        print(f"\n=== all-session encoder: held-out R2 gain x1e3 from per-epoch kernel CHANGES, {area} "
              "(change = all - shared; change_<g> = all - all-but-g's-change; fit null = pre late vs early)\n"
              + A.to_string())
    W.to_csv(out / "allsession_changes_by_session.csv", index=False, float_format="%.5g")
    D = pd.read_csv(out / "allsession_dbeta.csv")
    print("\n=== kernel change, summed over lags x1e3 (raw units), cortex\n"
          + D[D.area == "cortex"].pivot_table(index=["fit", "animal", "epoch"], columns="group",
                                               values="dbeta_sum_x1e3").round(1).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
