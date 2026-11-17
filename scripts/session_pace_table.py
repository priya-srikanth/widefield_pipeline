"""SESSION PACE vs the decoder's first-lick count (Priya 10-28: acute sessions have MORE first post-cue licks in the decoder's
lick set -- recprim_sessions.csv n_licks, panel 'cued first lick -> aim': per-animal epoch means pre / acute / subacute /
chronic PS92 128/196/120/122, PS93 157/198/138/114, PS94 167/218/137/128, PS95 203/213/152/131. Why?).

Candidates, one column each, per session:
  SESSION LENGTH / PACE (DAQ, `analysis_kit.session_behavior` -> engagement_decomposition.session_trials, the shared DAQ
    trial source): n_trials, dur_min (first to last cue), trials_per_min, med_iti_s, hit_rate, med_latency_s (first
    response-window lick, hits), quit_min (quit_point.session_quit; NaN when censored).
  ENGAGEMENT (the gate the imaging lick sets use: `iti_licks._behaviour` -> combined_encoding.load_behaviour, T.engaged,
    OROFACIAL_GATE of the frozen env = "attempted"): n_trials_img, n_engaged, eng_frac, n_eng_lick (engaged trials with a
    behavior lick onset in cue .. stop, extract's rule without the imaging-edge check), n_first_file (extract's stored
    first licks, with the imaging-edge check), n_rewards (DAQ reward TTLs).
  3-D TRACKING (pose3d kinematics_all licks table, table 'licks', context post_cue, finite side_ml): n_pc_tracked (all
    tracked post-cue licks), n_first_tracked (lick_idx == 0), licks_per_trial = n_pc_tracked / n_first_tracked.
  DECODER SET (iti_lick_order.preds_path(), set 'trial'): n_trial_set (all cued licks kept; extract caps the cued set at
    iti_licks.N_TRIAL_MAX = 1500 by a seeded random subsample), n_first_set (order == 0), tracked_frac = n_first_set /
    n_eng_lick, keep_frac = n_trial_set / n_pc_tracked.
Sessions: the labels of <OUT_W>/recprim_sessions.csv panel 'cued first lick -> aim' (with their epoch).
Usage: python scripts/session_pace_table.py [--jobs 8]
Output: <OUT_W>/session_pace.csv, <OUT_W>/session_pace_epoch_means.csv, <OUT_W>/figures/session_pace.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = il.EP4
PANEL = "cued first lick -> aim"
SHOW = [("n_licks", "decoder first licks (recprim n_licks)"), ("n_first_set", "tracked first licks in decoder set"),
        ("n_trials", "DAQ trials"), ("dur_min", "duration (min, first-last cue)"), ("trials_per_min", "trials / min"),
        ("n_engaged", "engaged trials (attempted gate)"), ("eng_frac", "engaged fraction"),
        ("n_eng_lick", "engaged trials with a post-cue lick"), ("tracked_frac", "tracked fraction (set / eng. w. lick)"),
        ("n_first_tracked", "3-D tracked first licks (all, before cap)"), ("n_pc_tracked", "3-D tracked post-cue licks"),
        ("licks_per_trial", "tracked post-cue licks per first lick"), ("keep_frac", "decoder keep fraction (1500 cap)"),
        ("hit_rate", "hit rate (DAQ)"), ("med_latency_s", "median first-lick latency (s)"),
        ("quit_min", "time to quit (min; censored = NaN)")]
TABLE = ["n_trials", "dur_min", "trials_per_min", "n_engaged", "eng_frac", "n_eng_lick", "n_first_set", "tracked_frac",
         "n_pc_tracked", "licks_per_trial", "keep_frac"]


def worker(label: str) -> dict:
    """One session (module-level for spawn)."""
    from wfield_local import analysis_kit as ak
    r = {"label": label}
    B = ak.session_behavior(label, min_trials=1, min_gated=1)
    if B is not None:
        tr = sorted(B.trials, key=lambda x: float(x["elapsed_s"]))
        el = np.array([float(x["elapsed_s"]) for x in tr])
        hit = np.array([x["hit"] for x in tr], float)
        lat = np.array([x["latency_s"] for x in tr], float)
        dur = (el[-1] - el[0]) / 60.0 if len(el) > 1 else np.nan
        q = B.quit
        r.update(n_trials=len(tr), dur_min=dur, trials_per_min=(len(tr) - 1) / dur if dur > 0 else np.nan,
                 med_iti_s=float(np.median(np.diff(el))) if len(el) > 1 else np.nan, hit_rate=float(hit.mean()),
                 med_latency_s=float(np.nanmedian(lat)) if np.isfinite(lat).any() else np.nan,
                 quit_min=np.nan if q["censored"] else float(q["quit_elapsed_s"]) / 60.0, censored=bool(q["censored"]))
    Bh = il._behaviour(label)
    T = Bh.trials
    bo = np.sort(np.asarray(Bh.lick_on_s, float)[np.isfinite(Bh.lick_on_s)])
    Eg = T[T.engaged & np.isfinite(T.cue_s)]
    n_lick, rts = 0, []
    for c, s in zip(Eg.cue_s.to_numpy(float), Eg.stop_s.to_numpy(float)):
        s = s if np.isfinite(s) else c + 3.5
        j = np.searchsorted(bo, c)
        if j < len(bo) and bo[j] <= s:
            n_lick += 1
            rts.append(bo[j] - c)
    r.update(n_trials_img=len(T), n_engaged=int(T.engaged.sum()), eng_frac=float(T.engaged.mean()),
             n_eng_lick=n_lick, med_rt_eng_s=float(np.median(rts)) if rts else np.nan,
             n_first_file=int(len(il._load(label)["first_rt"])), n_rewards=int(len(Bh.reward_s)))
    return r


def tracking_counts(labels) -> pd.DataFrame:
    from wfield_local import config
    from wfield_local.paths import PathResolver
    p = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
    K = pd.read_csv(p, usecols=["session", "table", "context", "lick_uid", "lick_idx", "side_ml"])
    K = K[(K.table == "licks") & (K.context == "post_cue") & np.isfinite(K.side_ml)].drop_duplicates(["session", "lick_uid"])
    sid = {f"{config.animal_of(x)}_2026{x.split('_')[-1]}": x for x in labels}
    K = K[K.session.isin(sid)].assign(label=lambda d: d.session.map(sid))
    return pd.DataFrame({"n_pc_tracked": K.groupby("label").size(),
                         "n_first_tracked": K[K.lick_idx == 0].groupby("label").size()})


def plot(S: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    from wfield_local import config
    col = config.animal_color()
    nc = 4
    nr = int(np.ceil(len(SHOW) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 2.9 * nr))
    rng = np.random.default_rng(0)
    for ax, (m, lab) in zip(axs.ravel(), SHOW):
        for k, a in enumerate(AN):
            g = S[S.animal == a]
            off = (k - 1.5) * 0.12
            for i, e in enumerate(EP):
                v = g[g.epoch == e][m].to_numpy(float)
                ax.scatter(i + off + rng.uniform(-0.03, 0.03, len(v)), v, s=7, color=col.get(a, "k"), alpha=0.35,
                           lw=0)
            mu = [g[g.epoch == e][m].mean() for e in EP]
            ax.plot(np.arange(4) + off, mu, "-o", color=col.get(a, "k"), ms=4, lw=1.4, label=a)
        ax.set_xticks(range(4), EP, fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axs.ravel()[len(SHOW):]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Why acute sessions have more first post-cue licks in the decoder set (BEHAVIOR): per-session dots, "
                 "per-animal epoch means (lines).\nThe cued decoder set is capped at 1500 randomly drawn tracked post-cue "
                 "licks per session; first licks kept = 1500 / (tracked post-cue licks per first lick)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=95)
    plt.close(fig)


def main() -> int:
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    R = pd.read_csv(il.OUT_W / "recprim_sessions.csv")
    R = R[R.panel == PANEL][["label", "animal", "epoch", "day", "n_licks"]].drop_duplicates("label")
    labels = sorted(R.label)
    res, fail = ak.fan_sessions(labels, worker, jobs=min(args.jobs, 8))
    if fail:
        raise RuntimeError(f"failed sessions: {fail[:5]}")
    S = R.merge(pd.DataFrame([v for _i, v in res]), on="label", how="left")
    S = S.merge(tracking_counts(labels), left_on="label", right_index=True, how="left")
    P = pd.read_csv(lo.preds_path(), usecols=["set", "label", "order"])
    P = P[P.set == "trial"]
    S["n_trial_set"] = S.label.map(P.groupby("label").size())
    S["n_first_set"] = S.label.map(P[P.order == 0].groupby("label").size())
    S["tracked_frac"] = S.n_first_set / S.n_eng_lick
    S["keep_frac"] = S.n_trial_set / S.n_pc_tracked
    S["licks_per_trial"] = S.n_pc_tracked / S.n_first_tracked
    S["first_tracked_frac"] = S.n_first_tracked / S.n_eng_lick
    S = S.sort_values(["animal", "day", "label"]).reset_index(drop=True)
    S.to_csv(il.OUT_W / "session_pace.csv", index=False)
    M = S.groupby(["animal", "epoch"])[[m for m, _ in SHOW] + ["first_tracked_frac", "n_trial_set"]].mean()
    M = M.reindex(pd.MultiIndex.from_product([AN, EP], names=["animal", "epoch"]))
    M.assign(n_sessions=S.groupby(["animal", "epoch"]).size()).to_csv(il.OUT_W / "session_pace_epoch_means.csv")
    pd.set_option("display.width", 250)
    for m in ["n_licks", *TABLE, "n_first_tracked", "first_tracked_frac", "hit_rate", "med_latency_s", "quit_min"]:
        print(f"\n== {m} (per-animal epoch means)\n" + M[m].unstack()[EP].round(3).to_string())
    plot(S, il.OUT_W / "figures" / "session_pace.png")
    print(f"\nwrote {il.OUT_W / 'session_pace.csv'} ({len(S)} sessions) and figures/session_pace.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
