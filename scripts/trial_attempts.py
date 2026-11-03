"""Per-trial attempt classes, a movement-based engagement label and an attempt / completion / latency split of the
behavioural deficit (Priya 10-20: "fraction of trials with lick as well as incomplete vs complete licks at far mid and
far contra"; "call a trial engaged if there is any evidence of attempted lick ... analyze no-try trials in the
canonical 'engaged' period separately"; "will we include latency?"). Plan: docs/DLC_IN_WIDEFIELD_ANALYSES.md 3B.

Per trial, cue .. trial stop: the class comes from THE trial classification, `wfield_local.trial_classes`
(Priya 11-03, one definition for every analysis -- read its docstring for the rules). This table keeps its historical
names for its readers: hit = contact (DAQ spout contact), incomplete, jaw_only, no_try = no_response, unknown.
attempted = hit | incomplete. jaw_only is a REACTION, not an attempt (Priya's 10-21 blind review, 24 post-stroke
trials, cam1 + cam4 to the trial stop: 0 / 24 judged attempts; all 11 jaw_only trials were jaw responses without opening
(6), other movement (4, 2 walking) or still (1); 4 / 6 no_try trials just under the floor also showed a jaw response --
key session_poses/attempt_review/key.csv). Confounds: sub-threshold walking moves the jaw; some jaw responses follow
the spout leaving (it starts moving 60-80 ms after stop; detection ends at stop). Until 10-21 jaw_only counted as an
attempt (outputs copied to qc/superseded_attempts_jaw_20261021/). `canonical` = the behaviour pipeline's
reference-position gate (`orofacial_cohort.engagement_keep`), which stays the DENOMINATOR: no-try trials inside it are
kept as their own class, never dropped, so failure to initiate stays measurable.

Per session x position (canonical trials): attempt_rate, completion (hit / attempted), incomplete_share (no-contact
licks / all licks, cam4 + cam1-only), latency_attempt_ms (median first lick, cam4 + cam1), latency_contact_ms (median
first contact, hit trials). Wide columns `<measure>_<pos>` per session feed `mixed_models.behaviour_deficit`
(latency with higher_is_better=False).

Output: <run>/imaging_encoding/qc/trial_attempts.csv (per trial), trial_attempts_sessions.csv (wide, per session),
trial_attempts_gate.csv (canonical vs attempted trial counts per animal x epoch x position).
Usage: PYTHONPATH=. python scripts/trial_attempts.py [--jobs 2]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from wfield_local import trial_classes as TC  # noqa: E402  THE trial classification (11-03)

CLASSES = ("hit", "incomplete", "jaw_only", "no_try", "unknown")
ATTEMPT_CLASSES = ("hit", "incomplete")             # jaw_only = reacted (jaw), NOT an attempt (10-21 blind review)
TA_NAMES = {"contact": "hit", "no_response": "no_try"}   # trial_classes names -> this table's historical names
MEASURES = ("attempt_rate", "ttc_ms", "completion", "incomplete_share", "latency_attempt_ms", "latency_move_ms",
            "latency_contact_ms",
            "lateral_bias", "lateral_scatter")
RESP_MS = 3500.0                                    # the task's response window (gui_config timing.response_window)


def session_trials(label: str) -> pd.DataFrame:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from scripts.first_lick_latency import session_latency
    rv = PathResolver()
    a, d = label.split("_")[0], "2026" + label[5:]
    sd = oo.session_dir(rv, oc.RUN, a, d)
    T = session_latency(label)
    stop = dict(zip(T.trial_id, T.stop_ms))
    L = pd.read_csv(sd / "licks.csv", usecols=["trial_id", "on_ms", "peak_angle_deg"])
    L = L[(L.on_ms >= 0) & (L.on_ms <= L.trial_id.map(stop))]
    ang1 = L.sort_values("on_ms").groupby("trial_id").peak_angle_deg.first()   # first cam4 lick, + = toward ipsi
    # canonical = the REFERENCE gate, explicitly: never the OROFACIAL_GATE env setting (other window 10-21)
    keep, gate = oc.engagement_keep(a, d, sd, gate="reference")
    t = T.trial_id
    T["angle_first"] = t.map(ang1)
    # CLASSES, lick counts and jaw calls come from THE trial classification (wfield_local.trial_classes, Priya 11-03:
    # one definition for every analysis; cue -> stop window, DAQ contact, jaw cam4 OR cam1, close_center unjudged,
    # at-cue jaw only through its re-onset). Only the names differ here, kept for this table's ~20 readers:
    # contact -> "hit", no_response -> "no_try". Regression 11-03: 62,746 / 62,771 trials identical to the old in-script
    # rule; the 25 others are trials where cam1 saw a tongue, now "incomplete".
    C = TC.session_trials(f"{a}_{d}", window="cue_stop").set_index("trial_id").reindex(t.to_numpy())
    T["n_licks"] = C.n_licks.fillna(0).to_numpy()
    T["n_nocontact"] = C.n_nocontact.fillna(0).to_numpy()
    T["jaw_cam4"], T["jaw_cam1"] = C.jaw_cam4.to_numpy(float), C.jaw_cam1.to_numpy(float)
    T["jaw_at_cue"] = C.jaw_at_cue.fillna(False).astype(bool).to_numpy()
    T["jaw_onset_ms"] = C.jaw_onset_ms.to_numpy(float)
    T["jaw_reonset"] = T.jaw_at_cue & T.jaw_onset_ms.notna()
    T["cls"] = C.cls.fillna("unknown").replace(TA_NAMES).to_numpy()
    T["attempted"] = T.cls.isin(ATTEMPT_CLASSES)
    T["canonical"] = True if keep is None else t.isin(keep)
    T["gate"] = gate
    return T


def session_summary(D: pd.DataFrame) -> pd.DataFrame:
    """Wide per-session table, `<measure>_<pos>` (Priya 10-20: engaged = attempted trials; the canonical period is used
    ONLY as the attempt-rate denominator, so a sated no-try trial is not scored as a far-contra deficit):
      attempt_rate        attempted / canonical trials                                   (INITIATION axis)
      ttc_ms              mean time to first contact on attempted trials; no contact by
                          RESP_MS, later contacts capped                       (EXECUTION axis = the deficit)
      completion, incomplete_share, latency_attempt_ms (first lick), latency_move_ms (first lick or after-cue jaw onset),
      latency_contact_ms                                                                attempted trials (components)
      lateral_bias / lateral_scatter   first cam4 lick peak angle on attempted trials vs the animal's August-pre mean /
                          sd at that spout (+ = toward ipsi). DESCRIPTIVE ONLY -- never a behavioural index: the executed
                          angle enters the neural measures' construction (DECISIONS 10-20, no circular index)
    """
    rows = []
    for (lab, p), g in D.groupby(["label", "pos"]):
        can, att = g[g.canonical], g[g.attempted]
        nl = att.n_licks.sum()
        rows.append({"label": lab, "pos": p, "n_trials": len(g), "n_attempted": len(att),
                     "attempt_rate": can.attempted.mean() if len(can) else np.nan,
                     "ttc_ms": np.minimum(att.contact.fillna(RESP_MS), RESP_MS).mean() if len(att) else np.nan,
                     "completion": (att.cls == "hit").mean() if len(att) else np.nan,
                     "incomplete_share": att.n_nocontact.sum() / nl if nl else np.nan,
                     "latency_attempt_ms": att.combined.where(att.has_cam1, att.cam4).median(),
                     "latency_contact_ms": att.contact.median(),
                     "latency_move_ms": np.fmin(att.combined.where(att.has_cam1, att.cam4), att.jaw_onset_ms).median(),
                     "angle_mean": att.angle_first.mean(), "angle_sd": att.angle_first.std(),
                     **{f"frac_{c}": (can.cls == c).mean() if len(can) else np.nan for c in CLASSES}})
    S = pd.DataFrame(rows)
    W = S.pivot_table(index="label", columns="pos", values=[c for c in S.columns if c not in ("label", "pos")])
    W.columns = [f"{m}_{p}" for m, p in W.columns]
    W = W.reset_index()
    an, pre = W.label.str.split("_").str[0], W.label.map(_is_aug_pre)
    for p in sorted({c[len("angle_mean_"):] for c in W.columns if c.startswith("angle_mean_")}):
        for col, out, f in ((f"angle_mean_{p}", f"lateral_bias_{p}", lambda x, r: x - r),
                            (f"angle_sd_{p}", f"lateral_scatter_{p}", lambda x, r: x / r)):
            ref = W[col].where(pre).groupby(an).transform("mean")
            W[out] = f(W[col], ref)
    return W


def _is_aug_pre(label: str) -> bool:
    from wfield_local import epochs
    return epochs.epoch_of(label) == "pre" and label[-4:] >= "0806"


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
    labels = [x for x in ce.sessions_available() if x in cur]
    got, fail = ak.fan_sessions(labels, session_trials, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED {it}: {err}")
    D = pd.concat([r for _l, r in got], ignore_index=True)
    D["animal"], D["epoch"] = D.label.str.split("_").str[0], D.label.map(epochs.epoch_of)
    D["day"] = D.label.map(epochs.days_since_stroke)
    D.to_csv(out / "trial_attempts.csv", index=False, float_format="%.1f")
    W = session_summary(D)
    W.insert(1, "animal", W.label.str.split("_").str[0])
    W.insert(2, "epoch", W.label.map(epochs.epoch_of))
    W.insert(3, "day", W.label.map(epochs.days_since_stroke))
    W.to_csv(out / "trial_attempts_sessions.csv", index=False, float_format="%.4g")
    Aug = D[(D.epoch != "pre") | (D.label.str[-4:] >= "0806")]           # August-only pre baseline
    G = Aug.groupby(["animal", "epoch", "pos"]).apply(
        lambda g: pd.Series({"n_trials": len(g), "canonical": g.canonical.sum(),
                             "attempted": g.attempted.sum(), "canonical_and_attempted": (g.canonical & g.attempted).sum(),
                             "no_try_in_canonical": (g.canonical & (g.cls == "no_try")).sum(),
                             "unknown_in_canonical": (g.canonical & (g.cls == "unknown")).sum(),
                             "attempted_outside_canonical": (~g.canonical & g.attempted).sum()}))
    G.to_csv(out / "trial_attempts_gate.csv")
    pd.set_option("display.width", 230)
    order = {e: i for i, e in enumerate(("pre", "acute", "subacute", "chronic"))}
    WA = W[W.label.isin(set(Aug.label))].drop(columns=["animal", "epoch", "day"])
    for m in MEASURES:
        P = WA.melt(id_vars="label")
        P = P[P.variable.str.startswith(m + "_")]
        P["pos"] = P.variable.str[len(m) + 1:]
        P["animal"], P["epoch"] = P.label.str.split("_").str[0], P.label.map(epochs.epoch_of)
        t = P.pivot_table(index=["animal", "epoch"], columns="pos", values="value")
        t = t.reindex(columns=[p for p in oc.POS_ORDER if p in t.columns])
        t = t.iloc[sorted(range(len(t)), key=lambda i: (t.index[i][0], order.get(t.index[i][1], 9)))]
        print(f"\n  {m} (mean over sessions, pre = August)\n" + t.round(3).to_string())
    print("\n  gate: canonical vs attempted (far positions)\n"
          + G[G.index.get_level_values("pos").str.startswith("far")].to_string())
    print(f"\n-> {out / 'trial_attempts_sessions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
