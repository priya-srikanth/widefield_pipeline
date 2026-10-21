"""Per-trial attempt classes, a movement-based engagement label and an attempt / completion / latency split of the
behavioural deficit (Priya 10-20: "fraction of trials with lick as well as incomplete vs complete licks at far mid and
far contra"; "call a trial engaged if there is any evidence of attempted lick ... analyze no-try trials in the
canonical 'engaged' period separately"; "will we include latency?"). Plan: docs/DLC_IN_WIDEFIELD_ANALYSES.md 3B.

Per trial, cue .. trial stop (`trial_windows.trial_bounds`; latencies from `scripts/first_lick_latency.session_latency`):
  hit         a DAQ spout contact
  incomplete  an LP lick (cam4 `licks.csv`, or a cam1-only event -- droplets out) and no contact
  jaw_only    no lick, jaw moved on EITHER camera: cam4 (v36 `tongue_jaw_mismatch.csv` jaw_pass_qc = the "jaw moved"
              call; NaN = cannot judge; not used at close_center, chin hidden) or cam1 (`cam1_jaw_moved`, same rule)
  no_try      no lick, jaw judged and still
  unknown     no lick, no usable jaw
attempted = hit | incomplete | jaw_only (the proposed movement engagement label). `canonical` = the behaviour pipeline's
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
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("hit", "incomplete", "jaw_only", "no_try", "unknown")
JAW_UNRELIABLE = ("close_mid",)                     # canonical pos keys (close_center); cam4 occludes the jaw
CAM1_JAW_CUT = 0.6                                  # cam1 jaw likelihood cutoff (scripts/cam1_model_compare.CUT)
MEASURES = ("attempt_rate", "ttc_ms", "completion", "incomplete_share", "latency_attempt_ms", "latency_contact_ms",
            "lateral_bias", "lateral_scatter")
RESP_MS = 3500.0                                    # the task's response window (gui_config timing.response_window)


def cam1_jaw_moved(a: str, d: str, rv, stop: dict) -> pd.Series:
    """{trial_id: 1 moved / 0 still / NaN cannot judge} from the cam1 LP jaw (session_poses/<a>_<d>_cam1_full
    windows_LP.csv; likelihood < CAM1_JAW_CUT -> NaN), by the SAME per-trial rule as cam4 (`jaw_kinematics.jaw_pertrial`:
    |y - pre-cue baseline| > max(4 sd, 4 px), sustained 5 / 8 frames, cue .. trial stop; degenerate baseline = unknown).
    Trial windows are laid end to end with NaN gaps so no snippet reaches into a neighbour."""
    from wfield_local import jaw_kinematics as jk
    d1 = Path(rv.root("dlc")) / "session_poses" / f"{a}_{d}_cam1_full"
    if not (d1 / "windows_LP.csv").exists():
        return pd.Series(dtype=float)
    P = pd.read_csv(d1 / "windows_LP.csv", header=[1, 2], index_col=0)
    idx = pd.read_csv(d1 / "windows_index.csv", usecols=["trial_id", "t_ms"])
    y = P[("jaw", "y")].to_numpy(float)
    y[P[("jaw", "likelihood")].to_numpy(float) < CAM1_JAW_CUT] = np.nan
    gap = int(np.ceil(6000 / 4))                                  # > the longest snippet, 250 fps
    parts, rows, off = [], [], 0
    for tid, g in idx.groupby("trial_id", sort=False):
        k = g.index.to_numpy()
        c = int(np.argmin(np.abs(g.t_ms.to_numpy())))
        parts += [y[k], np.full(gap, np.nan)]
        rows.append({"trial_id": int(tid), "position": "", "cue_frame": float(off + c),
                     "stop_frame": float(off + c + stop.get(int(tid), 3500) / 4)})
        off += len(k) + gap
    if not rows:
        return pd.Series(dtype=float)
    J, _ = jk.jaw_pertrial(np.concatenate(parts), 250.0, pd.DataFrame(rows), {"degenerate_as_unknown": True})
    return pd.Series(pd.to_numeric(J.jaw_pass_qc, errors="coerce").to_numpy(), index=J.trial_id)


def session_trials(label: str) -> pd.DataFrame:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from scripts.first_lick_latency import session_latency
    rv = PathResolver()
    a, d = label[:4], "2026" + label[5:]
    sd = oo.session_dir(rv, oc.RUN, a, d)
    T = session_latency(label)
    stop = dict(zip(T.trial_id, T.stop_ms))
    L = pd.read_csv(sd / "licks.csv", usecols=["trial_id", "on_ms", "contact", "peak_angle_deg"])
    L = L[(L.on_ms >= 0) & (L.on_ms <= L.trial_id.map(stop))]
    ang1 = L.sort_values("on_ms").groupby("trial_id").peak_angle_deg.first()   # first cam4 lick, + = toward ipsi
    n4 = L.groupby("trial_id").size()
    nc4 = (~L.contact.astype(bool)).groupby(L.trial_id).sum()
    f = Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / f"{a}_{d}_events.csv"
    n1 = nc1 = pd.Series(dtype=float)
    if f.exists():
        E = pd.read_csv(f, usecols=["trial_id", "on_ms", "cam4_matched", "in_3d", "contact", "droplet"])
        for c in ("cam4_matched", "in_3d", "contact", "droplet"):
            E[c] = E[c].astype("boolean").fillna(False).astype(bool)
        E = E[~E.cam4_matched & ~E.in_3d & ~E.droplet & (E.on_ms >= 0) & (E.on_ms <= E.trial_id.map(stop))]
        n1, nc1 = E.groupby("trial_id").size(), (~E.contact).groupby(E.trial_id).sum()
    jaw_ok = jaw_move = pd.Series(dtype=bool)
    if (sd / "tongue_jaw_mismatch.csv").exists():
        M = pd.read_csv(sd / "tongue_jaw_mismatch.csv",
                        usecols=["trial_id", "jaw_pass_qc", "candidate_no_lick_with_jaw_move"]).set_index("trial_id")
        # jaw_pass_qc IS the "jaw moved" call (wfield_local/jaw_kinematics.py); NaN = cannot judge (degenerate baseline)
        jq = pd.to_numeric(M.jaw_pass_qc.replace({"True": 1, "False": 0}), errors="coerce")
        jaw_ok, jaw_move = jq.notna(), jq == 1
    keep, gate = oc.engagement_keep(a, d, sd)
    t = T.trial_id
    T["n_licks"] = t.map(n4).fillna(0) + t.map(n1).fillna(0)
    T["n_nocontact"] = t.map(nc4).fillna(0) + t.map(nc1).fillna(0)
    T["angle_first"] = t.map(ang1)
    # jaw from EITHER camera (Priya 10-20): moved if either says moved; judged if either can judge. cam4 is not used at
    # close_center (chin hidden); cam1 sees the jaw there
    j4 = t.map(jaw_ok).fillna(False).astype(bool) & ~T.pos.isin(JAW_UNRELIABLE)
    m4 = j4 & t.map(jaw_move).fillna(False).astype(bool)
    J1 = cam1_jaw_moved(a, d, rv, stop)
    T["jaw_cam1"] = t.map(J1)
    T["jaw_cam4"] = np.where(j4, m4.astype(float), np.nan)
    ok = j4 | T.jaw_cam1.notna()
    mv = m4 | (T.jaw_cam1 == 1)
    T["cls"] = np.select([T.contact.notna(), T.n_licks > 0, ok & mv, ok], ["hit", "incomplete", "jaw_only", "no_try"],
                         "unknown")
    T["attempted"] = T.cls.isin(["hit", "incomplete", "jaw_only"])
    T["canonical"] = True if keep is None else t.isin(keep)
    T["gate"] = gate
    return T


def session_summary(D: pd.DataFrame) -> pd.DataFrame:
    """Wide per-session table, `<measure>_<pos>` (Priya 10-20: engaged = attempted trials; the canonical period is used
    ONLY as the attempt-rate denominator, so a sated no-try trial is not scored as a far-contra deficit):
      attempt_rate        attempted / canonical trials                                   (INITIATION axis)
      ttc_ms              mean time to first contact on attempted trials; no contact by
                          RESP_MS, later contacts capped                       (EXECUTION axis = the deficit)
      completion, incomplete_share, latency_attempt_ms, latency_contact_ms   attempted trials (components)
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
                     "angle_mean": att.angle_first.mean(), "angle_sd": att.angle_first.std(),
                     **{f"frac_{c}": (can.cls == c).mean() if len(can) else np.nan for c in CLASSES}})
    S = pd.DataFrame(rows)
    W = S.pivot_table(index="label", columns="pos", values=[c for c in S.columns if c not in ("label", "pos")])
    W.columns = [f"{m}_{p}" for m, p in W.columns]
    W = W.reset_index()
    an, pre = W.label.str[:4], W.label.map(_is_aug_pre)
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
    D["animal"], D["epoch"] = D.label.str[:4], D.label.map(epochs.epoch_of)
    D["day"] = D.label.map(epochs.days_since_stroke)
    D.to_csv(out / "trial_attempts.csv", index=False, float_format="%.1f")
    W = session_summary(D)
    W.insert(1, "animal", W.label.str[:4])
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
        P["animal"], P["epoch"] = P.label.str[:4], P.label.map(epochs.epoch_of)
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
