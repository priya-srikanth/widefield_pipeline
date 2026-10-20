"""Incomplete licks and jaw-without-lick trials with cam1 added (Priya 10-19/20; handoff to-do 4).

    python -m scripts.cam1_incomplete_licks

Per session x spout position, per trial: cam4 licks (v36 ``licks.csv``) and its NO-CONTACT licks; cam1-ONLY events
(`scripts/cam1_licks.py`: no cam4 lick and no 3-D lick overlaps them; water droplets removed -- blind review, 2 rounds)
split into ``lips`` (never > 20 px below the cam1 mouth) and ``out``; INCOMPLETE licks = cam4 no-contact + cam1-only
without a DAQ contact; and the jaw-without-lick share of jaw-QC'd trials (v36 ``tongue_jaw_mismatch.csv``) as cam4 has it
and CORRECTED (trials where cam1 saw the tongue in the quiet window removed from the numerator). Sessions failing a 3-D
gate (`pose3d.failed_sessions`) are left out (the cam4 / 3-D match flags are unreliable there).

Writes `session_poses/cam1_licks/incomplete/{sessions.csv, by_epoch.csv}` and `figures/incomplete_<animal>.png` (x =
days from the stroke, June compressed at the left as in `pose3d_summary_fig`; grey = August pre range).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
MEAS = {"licks_cam4": "cam4 licks / trial", "nocontact_cam4": "cam4 no-contact licks / trial",
        "cam1_only": "cam1-only events / trial", "cam1_only_lips": "cam1-only at the lips / trial",
        "incomplete": "incomplete licks / trial\n(cam4 no-contact + cam1-only no-contact)",
        "incomplete_frac": "incomplete share of all licks", "jaw_only_cam4": "jaw-without-lick share (cam4)",
        "jaw_only_corr": "jaw-without-lick share\n(cam1-corrected)"}


def session_rows(rv, s: str, E: pd.DataFrame) -> list[dict]:
    a, d = s.split("_")
    sd = Path(rv.root("orofacial")) / V36 / "sessions" / a / d
    if not (sd / "trials.csv").exists():
        return []
    T = pd.read_csv(sd / "trials.csv")[["trial_id", "position"]]
    L = pd.read_csv(sd / "licks.csv") if (sd / "licks.csv").exists() else pd.DataFrame(columns=["trial_id", "contact"])
    M = pd.read_csv(sd / "tongue_jaw_mismatch.csv") if (sd / "tongue_jaw_mismatch.csv").exists() else None
    g = E[E.session == s]
    c1 = g[g.cam1_only]
    seen = set(g.loc[g.jaw_only_trial, "trial_id"])
    out = []
    for p, tp in T.groupby("position"):
        ids = set(tp.trial_id)
        n = len(ids)
        lp = L[L.trial_id.isin(ids)]
        cp = c1[c1.trial_id.isin(ids)]
        r = {"session": s, "animal": a, "date": d, "position": p, "n_trials": n,
             "licks_cam4": len(lp) / n, "nocontact_cam4": float((~lp.contact.astype(bool)).sum()) / n,
             "cam1_only": len(cp) / n, "cam1_only_lips": float((cp.size_class == "lips").sum()) / n,
             "cam1_only_nocontact": float((~cp.contact).sum()) / n}
        r["incomplete"] = r["nocontact_cam4"] + r["cam1_only_nocontact"]
        tot = r["licks_cam4"] + r["cam1_only"]
        r["incomplete_frac"] = r["incomplete"] / tot if tot > 0 else np.nan
        if M is not None:
            q = M[M.trial_id.isin(ids) & M.jaw_pass_qc.astype(bool)]
            cand = q.candidate_no_lick_with_jaw_move.astype(bool)
            r["jaw_qc_trials"] = len(q)
            r["jaw_only_cam4"] = float(cand.mean()) if len(q) else np.nan
            r["jaw_only_corr"] = float((cand & ~q.trial_id.isin(seen)).mean()) if len(q) else np.nan
        out.append(r)
    return out


def figure(S: pd.DataFrame, animal: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d_summary_fig import EP_COL, _x, _xaxis
    A = S[S.animal == animal].copy()
    A["x"] = [_x(animal, d) for d in A.date]
    fig, axes = plt.subplots(len(MEAS), len(POS), figsize=(3.0 * len(POS), 1.9 * len(MEAS)), squeeze=False)
    for i, (m, lab) in enumerate(MEAS.items()):
        for j, p in enumerate(POS):
            ax = axes[i, j]
            g = A[A.position == p]
            pre = g[(g.epoch == "pre") & (g.date >= "20260801")][m].dropna()
            if len(pre):
                ax.axhspan(pre.min(), pre.max(), color="0.9", zorder=0)
                ax.axhline(pre.median(), color="0.5", lw=0.8, ls="--")
            for ep, ge in g.groupby("epoch"):
                june = ge.date < "20260801"
                ax.scatter(ge.x[~june], ge[m][~june], s=16, color=EP_COL.get(ep, "C0"), zorder=3)
                ax.scatter(ge.x[june], ge[m][june], s=16, facecolor="none", edgecolor=EP_COL.get(ep, "C0"), zorder=3)
            _xaxis(ax)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(p, fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("days from stroke", fontsize=7)
    fig.suptitle(f"{animal}: incomplete licks and jaw-without-lick trials, cam4 + cam1 (droplets removed); grey = August "
                 "pre range, dashed = its median; open = June", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import epochs
    from wfield_local import pose3d as P3
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    d = Path(rv.root("dlc")) / "session_poses" / "cam1_licks"
    E = pd.concat([pd.read_csv(f, dtype={"date": str}) for f in sorted(d.glob("PS*_events.csv"))], ignore_index=True)
    for c in ("cam4_matched", "in_3d", "contact", "jaw_only_trial", "droplet"):
        E[c] = E[c].astype("boolean").fillna(False).astype(bool)
    E["cam1_only"] = ~E.cam4_matched & ~E.in_3d & ~E.droplet
    E["jaw_only_trial"] &= ~E.droplet
    bad = P3.failed_sessions(Path(rv.root("dlc")) / "session_poses" / "pose3d")
    rows = []
    from wfield_local import analysis_kit as ak
    for s in sorted((set(E.session) - set(bad)) & ak.curated_ids()):            # curated cohort only (Priya 10-20)
        rows += session_rows(rv, s, E)
    S = pd.DataFrame(rows)
    S["epoch"] = [epochs.epoch_of(f"{a}_{dt[4:]}") for a, dt in zip(S.animal, S.date)]
    out = d / "incomplete"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "sessions.csv", index=False, float_format="%.4g")
    Aug = S[(S.epoch != "pre") | (S.date >= "20260801")]                    # August-only pre baseline
    B = Aug.groupby(["animal", "epoch", "position"])[list(MEAS)].mean()
    B["sessions"] = Aug.groupby(["animal", "epoch", "position"]).size()
    B.to_csv(out / "by_epoch.csv", float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    order = {"pre": 0, "acute": 1, "subacute": 2, "chronic": 3}
    P = B.reset_index()
    P["o"] = P.epoch.map(order)
    for m in ("incomplete", "incomplete_frac", "cam1_only_lips", "jaw_only_cam4", "jaw_only_corr"):
        print(f"\n{MEAS[m].replace(chr(10), ' ')} (mean over sessions; pre = August):")
        print(P.sort_values("o").pivot_table(index=["animal", "epoch"], columns="position", values=m, sort=False)
              .reindex(columns=list(POS)).round(3).to_string())
    for a in sorted(S.animal.unique()):
        figure(S, a, out / "figures" / f"incomplete_{a}.png")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
