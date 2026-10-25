"""POST-STROKE CHANGE CLAIMS REDONE IN THE FROZEN FRAME (Priya 10-25: "redo all post-stroke change claims with the frozen
model"). The claims below were made with the per-session refit encoder (`combined_encoding`), which re-learns every
session's weights, so a change there mixes changed movement with changed cortex. Here: one model per animal from its
August pre sessions (pre held out leave-one-session-out), applied unchanged; unique R2 of g = R2[joint] - R2[no_g].
Frames: noFR = `frozen_encoder_gate-attempted_v7g3_t3d`, FR-A = `..._frA` (FaceRhythm mode A in the frozen model).

A. TARGET vs EXECUTED (refit claim: binned contest, chronic target-beyond-executed gain, FINDINGS 4.11): frozen unique R2
   of target (the spout itself, beyond every executed regressor), exec_locked (the executed movement's spout-locked
   part) and exec_within (where the tongue / jaw went relative to that spout's usual path).
B. TRIAL-TO-TRIAL vs PETH (refit claim: the acute spout loss is a PETH loss, FINDINGS 4.12 / 8.9): the same unique R2 on
   `trial` (around each spout's mean trace) and `avg` (the spout-mean traces = PETH) segments of the frozen prediction.
C. RECOVERY TREND (refit claim: chronic position-coding gain beyond the practice trend, follows behaviour, ENCODING 4.1-4.2):
   per session spout unique R2 (cortex, frame), pre -> chronic per animal, and the day it reaches half its acute ->
   chronic change vs far-contra contact (behaviour, `qc/position_coding_trend_recovery_sessions.csv`).
Usage: python scripts/frozen_claims.py
Output: <E>/frozen_fr_compare/claims_*.csv, figures/frozen_claims_{contest,peth,trend}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402
from frozen_fr_compare import WINDOWS, bars, boot, session_r2  # noqa: E402

RUNS = {"noFR": "frozen_encoder_gate-attempted_v7g3_t3d", "FR-A": "frozen_encoder_gate-attempted_v7g3_t3d_frA"}
KEY = ["label", "animal", "epoch", "window", "area"]
AR = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left"]


def unique(folder: str, groups: dict, measure: str) -> pd.DataFrame:
    J = session_r2(folder, "joint", measure).set_index(KEY).r2
    out = []
    for g, v in groups.items():
        if (E / folder / v / "sessions").is_dir():
            u = (J - session_r2(folder, v, measure).set_index(KEY).r2).rename("u").reset_index()
            out.append(u.assign(q=g))
    return pd.concat(out)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from position_coding_trend_recovery import half_recovery_day
    out = E / "frozen_fr_compare"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    # A. target vs executed
    GA = {"target": "no_target", "exec_locked": "no_exec_locked", "exec_within": "no_exec_within"}
    A = pd.concat([unique(f, GA, "frame").assign(frame=k) for k, f in RUNS.items()])
    BA = pd.concat([boot(A[(A.frame == k) & (A.q == q)], "u").assign(frame=k, q=q) for k in RUNS for q in GA])
    BA.to_csv(out / "claims_contest.csv", index=False)
    print("== A. frozen unique R2, epoch - pre\n" + BA[BA.area.isin(["cortex", "SSp-m_right", "SSp-m_left"])][
        ["q", "frame", "window", "area", "epoch", "ref", "point", "lo", "hi", "n_up"]].round(4).to_string(index=False))

    # B. trial-to-trial vs PETH
    GB = {"target": "no_target", "spout": "no_spout", "movement": "no_movement", "fr": "no_fr_all"}
    Bt = pd.concat([unique(f, GB, m).assign(frame=k, measure=m) for k, f in RUNS.items() for m in ("trial", "avg")])
    BB = pd.concat([boot(Bt[(Bt.frame == k) & (Bt.q == q) & (Bt.measure == m)], "u").assign(frame=k, q=q, measure=m)
                    for k in RUNS for q in GB for m in ("trial", "avg") if ((Bt.frame == k) & (Bt.q == q)).any()])
    BB.to_csv(out / "claims_peth.csv", index=False)
    lev = Bt[Bt.epoch == "pre"].groupby(["frame", "q", "measure", "window", "area", "animal"]).u.mean() \
        .groupby(["frame", "q", "measure", "window", "area"]).mean().unstack("measure")
    print("\n== B. pre-stroke unique R2, trial-to-trial vs PETH (mean of animals)\n"
          + lev.loc[(slice(None), slice(None), slice(None), ["cortex", "SSp-m_right", "SSp-m_left"]), :].round(4).to_string())
    print("\n== B. epoch - pre\n" + BB[BB.area.isin(["cortex", "SSp-m_right", "SSp-m_left"]) & (BB.epoch != "subacute")][
        ["q", "frame", "measure", "window", "area", "epoch", "ref", "point", "lo", "hi", "n_up"]].round(4).to_string(index=False))

    # C. recovery trend (spout unique, cortex, frame R2, lick window) vs behaviour
    C = A[A.q == "target"].iloc[:0]
    C = pd.concat([unique(f, {"spout": "no_spout"}, "frame").assign(frame=k) for k, f in RUNS.items()])
    from wfield_local import epochs
    C["day"] = C.label.map(epochs.days_since_stroke)
    beh = pd.read_csv(E / "qc" / "position_coding_trend_recovery_sessions.csv")[["label", "contact_far_contra"]]
    H, L = [], []
    for (k, w, ar), t in C[C.area.isin(["cortex", "SSp-m_right", "SSp-m_left"])].groupby(["frame", "window", "area"]):
        for an, g in t.groupby("animal"):
            g = g.merge(beh, on="label", how="left").sort_values("day").rename(columns={"u": "spout_u"})
            H.append({"frame": k, "window": w, "area": ar, "animal": an, "half_day_spout": half_recovery_day(g, "spout_u"),
                      "half_day_contact": half_recovery_day(g, "contact_far_contra")})
            L.append({"frame": k, "window": w, "area": ar, "animal": an,
                      **{ep: g[g.epoch == ep].spout_u.mean() for ep in ("pre", "acute", "subacute", "chronic")}})
    H, L = pd.DataFrame(H), pd.DataFrame(L)
    H.to_csv(out / "claims_trend_halfday.csv", index=False)
    L.to_csv(out / "claims_trend_levels.csv", index=False)
    print("\n== C. spout unique R2 per animal (frozen; rule 8)\n" + L.round(4).to_string(index=False))
    print("\n== C. day of 50 % acute -> chronic change: spout unique vs far-contra contact\n" + H.to_string(index=False))

    # figures
    for name, Bx, qs, sub in (("contest", BA, list(GA), None), ("peth", BB, list(GB), "measure")):
        for w in WINDOWS:
            rows_ = [(q, m) for q in qs for m in ((None,) if sub is None else ("avg", "trial"))
                     if ((Bx.q == q) & ((Bx[sub] == m) if sub else True)).any()]
            fig, axs = plt.subplots(len(rows_), 1, figsize=(15, 2.7 * len(rows_) + 1.3), squeeze=False)
            for ax, (q, m) in zip(axs[:, 0], rows_):
                b = Bx[(Bx.q == q) & (Bx.window == w) & ((Bx[sub] == m) if sub else True)]
                fr_ = [f for f in RUNS if (b.frame == f).any()]
                lab = {"avg": "PETH (spout-mean traces)", "trial": "TRIAL-TO-TRIAL (around each spout's mean)"}.get(m, "")
                bars(ax, b, fr_, AR, "change from pre (R2)", f"{q} unique R2 {lab}")
            h, _l = axs[0, 0].get_legend_handles_labels()
            fig.legend(h, ["noFR = frozen final frame", "FR-A = frozen + FaceRhythm"], loc="upper center", ncol=2, fontsize=9,
                       frameon=False, bbox_to_anchor=(0.5, 1 - 0.6 / (2.7 * len(rows_) + 1.3)))
            what = {"contest": "TARGET vs EXECUTED (target = the spout beyond every executed regressor)",
                    "peth": "TRIAL-TO-TRIAL vs PETH"}[name]
            fig.suptitle(f"NEURAL: {what}, FROZEN frame, {w} window; epoch - pre, mean of animals, 95 % nested CI, * excludes 0, "
                         "dots = animals. contra = right (lesion left).", fontsize=9.5)
            fig.subplots_adjust(left=0.05, right=0.995, top=1 - 1.45 / (2.7 * len(rows_) + 1.3), bottom=0.05, hspace=0.75)
            fig.savefig(out / "figures" / f"frozen_claims_{name}_{w}.png", dpi=105)
            plt.close(fig)
    # trend figure: per animal spout unique over days (cortex, lick), both frames + far-contra contact
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, an in zip(axs, AN):
        for k, col in (("noFR", "#999999"), ("FR-A", "#e7298a")):
            g = C[(C.frame == k) & (C.animal == an) & (C.area == "cortex") & (C.window == "lick")].sort_values("day")
            ax.plot(g.day, g.u, "o-", color=col, ms=3, lw=1, label=f"spout unique, {k}")
        ax2 = ax.twinx()
        b = beh.assign(day=beh.label.map(epochs.days_since_stroke))
        b = b[b.label.str.startswith(an)].sort_values("day")
        ax2.plot(b.day, b.contact_far_contra, "s--", color="#1b9e77", ms=3, lw=0.8, label="far-contra contact (behaviour)")
        ax2.set_ylim(0, 1)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(an, fontsize=10)
        ax.set_xlabel("day from stroke")
        if an == AN[0]:
            ax.set_ylabel("frozen spout unique R2 (cortex, lick)")
            ax.legend(fontsize=7, loc="upper left")
        if an == AN[-1]:
            ax2.set_ylabel("far-contra contact fraction", color="#1b9e77")
            ax2.legend(fontsize=7, loc="lower right")
    fig.suptitle("NEURAL + LICK: recovery course in the FROZEN frame -- spout unique R2 (target + executed spout part) per session vs "
                 "far-contra contact (green).", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.96, top=0.85, bottom=0.14, wspace=0.12)
    fig.savefig(out / "figures" / "frozen_claims_trend.png", dpi=105)
    plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
