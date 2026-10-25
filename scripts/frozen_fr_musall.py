"""Musall-style UNIQUE R2 in the FROZEN frame, with vs without FaceRhythm (Priya 10-25: "make the frozen model the headline").
The per-session refit versions (Figs 16, fr_contra_orofacial.py) re-learn every session's weights, so a post-stroke
change there can be changed movement OR changed cortex; here one model per animal from its August pre sessions (pre held
out leave-one-session-out) is applied unchanged, and each group's unique R2 = R2[joint] - R2[no_<group>] (order-free;
shared variance credited to nobody), per session x area x window (frame R2 over the window's frames).

FRAMES  noFR = `frozen_encoder_gate-attempted_v7g3_t3d` (final frame), FR-A = `..._frA` (FaceRhythm mode A inside)
QUANTITIES  target (the six position blocks = "position unique"), spout (target + spout-locked + within-spout executed),
            task (timing + target), movement (every movement group; with FR includes FR), FR (all FR; FR-A only),
            FR direction (side / distance; FR-A only)
Areas incl. contra - ipsi laterality (lesion LEFT: *_right = contralesional). Epoch - pre, nested bootstrap, rule-8 table.
Usage: python scripts/frozen_fr_musall.py
Output: <E>/frozen_fr_compare/musall_{sessions,change}.csv, figures/frozen_musall_{cue,lick}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402
from frozen_fr_compare import ALAB, WINDOWS, bars, boot, session_r2  # noqa: E402

RUNS = {"noFR": "frozen_encoder_gate-attempted_v7g3_t3d", "FR-A": "frozen_encoder_gate-attempted_v7g3_t3d_frA"}
Q = [("target", "no_target", "position unique (target)"), ("spout", "no_spout", "spout unique (target + executed spout part)"),
     ("task", "no_task", "task unique (timing + target)"), ("movement", "no_movement", "movement unique (all movement)"),
     ("fr", "no_fr_all", "FaceRhythm unique (all)"), ("fr_dir", "no_fr_dir", "FaceRhythm direction unique")]
AREAS2 = ["cortex", "SSp-m_right", "SSp-m_left", "SSp-m_c-i", "MOp_right", "MOp_left", "MOp_c-i"]
ALAB.update({"SSp-m_c-i": "SSp-m contra-ipsi", "MOp_c-i": "MOp contra-ipsi"})


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = E / "frozen_fr_compare"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    rows = []
    for fr, folder in RUNS.items():
        J = session_r2(folder, "joint").set_index(["label", "animal", "epoch", "window", "area"]).r2
        for q, v, _lab in Q:
            if not (E / folder / v / "sessions").is_dir():
                continue
            u = (J - session_r2(folder, v).set_index(["label", "animal", "epoch", "window", "area"]).r2).rename("u")
            rows.append(u.reset_index().assign(frame=fr, q=q))
    U = pd.concat(rows)
    w_ = U.pivot_table(index=["frame", "q", "label", "animal", "epoch", "window"], columns="area", values="u")
    for b in ("SSp-m", "MOp"):
        w_[f"{b}_c-i"] = w_[f"{b}_right"] - w_[f"{b}_left"]
    U = w_.reset_index().melt(id_vars=["frame", "q", "label", "animal", "epoch", "window"], var_name="area", value_name="u")
    U.to_csv(out / "musall_sessions.csv", index=False)
    per = U[U.area.isin(["SSp-m_right", "SSp-m_c-i"])].groupby(["q", "frame", "window", "area", "animal", "epoch"]).u.mean() \
        .unstack("epoch")[["pre", "acute", "subacute", "chronic"]]
    print("== per animal (rule 8), contralesional SSp-m and laterality\n" + per.round(4).to_string())
    B = pd.concat([boot(U[(U.frame == fr) & (U.q == q)], "u").assign(frame=fr, q=q) for fr in RUNS for q, _v, _l in Q
                   if ((U.frame == fr) & (U.q == q)).any()])
    B.to_csv(out / "musall_change.csv", index=False)
    print("\n== epoch - pre (nested bootstrap)\n" + B[B.area.isin(["SSp-m_right", "SSp-m_left", "SSp-m_c-i"])][
        ["q", "frame", "window", "area", "epoch", "ref", "point", "lo", "hi", "p", "n_up"]].round(4).to_string(index=False))
    for w in WINDOWS:
        fig, axs = plt.subplots(len(Q), 1, figsize=(17, 3.1 * len(Q) + 1.2))
        for ax, (q, _v, lab) in zip(axs, Q):
            fr_ = [f for f in RUNS if ((B.frame == f) & (B.q == q)).any()]
            bars(ax, B[(B.q == q) & (B.window == w)].assign(), fr_, AREAS2, "change from pre (R2)", lab)
            pre = B[(B.q == q) & (B.window == w) & (B.epoch == "chronic")].set_index(["frame", "area"]).ref
            ax.text(0.005, 0.03, "pre level: " + "; ".join(f"{f} {ALAB[a]} {pre.get((f, a), float('nan')):.3f}"
                                                         for f in fr_ for a in ("SSp-m_right", "cortex")),
                    transform=ax.transAxes, fontsize=7)
        h, _l = axs[0].get_legend_handles_labels()
        fig.legend(h, ["noFR = frozen final frame", "FR-A = frozen + FaceRhythm (mode A)"], loc="upper center", ncol=2,
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))
        fig.suptitle(f"NEURAL: Musall-style UNIQUE R2 in the FROZEN frame (pre-stroke model applied unchanged), {w} window. Bars = "
                     "mean of animals' epoch - pre; whisker 95 % nested CI; * excludes 0; dots = animals. contra = right "
                     "(lesion left).", fontsize=10)
        fig.subplots_adjust(left=0.05, right=0.995, top=0.92, bottom=0.04, hspace=0.6)
        fig.savefig(out / "figures" / f"frozen_musall_{w}.png", dpi=105)
        plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
