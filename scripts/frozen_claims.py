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
ALM / tjM1 / MOs_rest / MOp_rest ROWS (Priya 10-27 / 10-28 "yes let's redo musall"): the area run saved area means only,
so these rows come from the frame's BASIS twin re-run with FROZEN_PIX_MUSALL=1 (`<run>_basis_pixmusall`, ridge alphas
held at the area run's) -- ROI R2 = 1 - w'Ew / w'Sw (`encoder_split_rows.musall_rows`), ROI v8 on the stat mask. Placed
right after their parent (MOp -> tjM1, MOp_rest; MOs -> ALM, MOs_rest) on a grey band; MOs (area run) added as their
parent. Validation (basis route vs the area run's MOs / MOp / SSp-m / cortex): docs/status/ALM_TJM1_ROWS_PLAN_20261027.md.
Usage: python scripts/frozen_claims.py [--out DIR] [--pix-root DIR]
Output: <E>/frozen_fr_compare/claims_*.csv, figures/frozen_claims_{contest,peth,trend}.png
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
from wfield_local.frozen_encoder import design_tag as _design_tag  # frame tag of the FROZEN_LICKS_ALL design (8.51)

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402
from frozen_fr_compare import WINDOWS, bars, boot, session_r2  # noqa: E402

RUNS = {"noFR": _design_tag("frozen_encoder_gate-attempted_v7g3_t3d"), "FR-A": _design_tag("frozen_encoder_gate-attempted_v7g3_t3d_frA")}
KEY = ["label", "animal", "epoch", "window", "area"]
#: split rows (encoder_split_rows) right after their parent's pair, as fr_direction: MOp -> tjM1, MOp_rest; MOs -> ALM, MOs_rest
SPLIT = ["tjM1_right", "tjM1_left", "MOp_rest_right", "MOp_rest_left", "ALM_right", "ALM_left", "MOs_rest_right",
         "MOs_rest_left"]
ALLEN = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "MOs_right", "MOs_left"]
AR = ALLEN[:5] + SPLIT[:4] + ALLEN[5:7] + SPLIT[4:]
#: where the FROZEN_PIX_MUSALL basis re-runs live (`--pix-root` for a dry run); the basis itself stays in the original
PIX = {k: f"{v}_basis_pixmusall" for k, v in RUNS.items()}
PIX_ROOT = E


def unique(folder: str, groups: dict, measure: str) -> pd.DataFrame:
    """unique R2 of each group (joint - no_<g>) per session x window x area: Allen rows from the area run ``folder``,
    split rows from its FROZEN_PIX_MUSALL basis twin when that exists."""
    pix = PIX_ROOT / f"{folder}_basis_pixmusall"

    def r2(v):
        t = session_r2(folder, v, measure, areas=ALLEN)
        if (pix / v / "sessions").is_dir():
            import encoder_split_rows as esr
            t = pd.concat([t, esr.musall_rows(pix, v, measure, tuple(SPLIT), fit_run=E / f"{folder}_basis")],
                          ignore_index=True)
        return t.set_index(KEY).r2
    J = r2("joint")
    out = []
    for g, v in groups.items():
        if (E / folder / v / "sessions").is_dir():
            u = (J - r2(v)).rename("u").reset_index()
            out.append(u.assign(q=g))
    return pd.concat(out)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import encoder_split_rows as esr
    import frozen_fr_compare as ffc
    from position_coding_trend_recovery import half_recovery_day
    global PIX_ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output folder (default <E>/frozen_fr_compare)")
    ap.add_argument("--pix-root", default=None, help="folder holding the *_basis_pixmusall runs (default <E>)")
    a = ap.parse_args(argv)
    PIX_ROOT = Path(a.pix_root) if a.pix_root else E
    out = Path(a.out) if a.out else E / "frozen_fr_compare"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    ffc.ALAB.update({x: esr.side_label(x) for x in AR if x not in ffc.ALAB})
    print("split rows from:", {k: str(PIX_ROOT / v) for k, v in PIX.items() if (PIX_ROOT / v).is_dir()} or "NONE (Allen rows only)")

    # A. target vs executed
    GA = {"target": "no_target", "exec_locked": "no_exec_locked", "exec_within": "no_exec_within"}
    A = pd.concat([unique(f, GA, "frame").assign(frame=k) for k, f in RUNS.items()])
    BA = pd.concat([boot(A[(A.frame == k) & (A.q == q)], "u").assign(frame=k, q=q) for k in RUNS for q in GA])
    BA.to_csv(out / "claims_contest.csv", index=False)
    print("== A. frozen unique R2, epoch - pre\n" + BA[BA.area.isin(["cortex", "SSp-m_right", "SSp-m_left"])][
        ["q", "frame", "window", "area", "epoch", "ref", "point", "lo", "hi", "n_up"]].round(4).to_string(index=False))
    MOT = AR[3:]                                         # MOp / MOs and their split rows
    perA = A[A.area.isin(MOT)].groupby(["q", "frame", "window", "area", "animal", "epoch"]).u.mean().unstack("epoch")
    print("\n== A. per animal (rule 8): frozen unique R2, MOp / MOs and the ALM / tjM1 split rows\n"
          + perA.reindex(columns=["pre", "acute", "subacute", "chronic"]).round(4).to_string())

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
    perB = Bt[Bt.area.isin(MOT)].groupby(["q", "frame", "measure", "window", "area", "animal", "epoch"]).u.mean() \
        .unstack("epoch")
    print("\n== B. per animal (rule 8): unique R2, MOp / MOs and the split rows\n"
          + perB.reindex(columns=["pre", "acute", "subacute", "chronic"]).round(4).to_string())

    # C. recovery trend (spout unique, cortex, frame R2, lick window) vs behaviour
    C = A[A.q == "target"].iloc[:0]
    C = pd.concat([unique(f, {"spout": "no_spout"}, "frame").assign(frame=k) for k, f in RUNS.items()])
    from wfield_local import epochs
    C["day"] = C.label.map(epochs.days_since_stroke)
    beh = pd.read_csv(E / "qc" / "position_coding_trend_recovery_sessions.csv")[["label", "contact_far_contra"]]
    H, L = [], []
    # (the figure plots cortex only; the MOp / MOs / split rows are in the two CSVs -- 10-28)
    for (k, w, ar), t in C[C.area.isin(["cortex", "SSp-m_right", "SSp-m_left", *MOT])].groupby(["frame", "window", "area"]):
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
            ar_ = [x for x in AR if (Bx.area == x).any()]   # split rows only when their basis re-run exists
            fig, axs = plt.subplots(len(rows_), 1, figsize=(max(15, 1.6 * len(ar_)), 2.7 * len(rows_) + 1.3),
                                    squeeze=False)
            for ax, (q, m) in zip(axs[:, 0], rows_):
                b = Bx[(Bx.q == q) & (Bx.window == w) & ((Bx[sub] == m) if sub else True)]
                fr_ = [f for f in RUNS if (b.frame == f).any()]
                lab = {"avg": "PETH (spout-mean traces)", "trial": "TRIAL-TO-TRIAL (around each spout's mean)"}.get(m, "")
                bars(ax, b, fr_, ar_, "change from pre (R2)", f"{q} unique R2 {lab}")
                ne = len(ffc.EPS) + 1
                for i, x in enumerate(ar_):                 # split rows on a light grey band (as fr_direction)
                    if x in SPLIT:
                        ax.axvspan(i * ne - 0.5, i * ne + ne - 1.5, color="#eeeeee", zorder=-1, lw=0)
            h, lb = axs[0, 0].get_legend_handles_labels()
            nm = {"noFR": "noFR = frozen final frame", "FR-A": "FR-A = frozen + FaceRhythm"}
            fig.legend(h, [nm.get(x, x) for x in lb], loc="upper center", ncol=3, fontsize=9,
                       frameon=False, bbox_to_anchor=(0.5, 1 - 0.6 / (2.7 * len(rows_) + 1.3)))
            what = {"contest": "TARGET vs EXECUTED (target = the spout beyond every executed regressor)",
                    "peth": "TRIAL-TO-TRIAL vs PETH"}[name]
            fig.suptitle(f"NEURAL: {what}, FROZEN frame, {w} window; epoch - pre, mean of animals, 95 % nested CI, * excludes 0, "
                         "dots = animals. contra = right (lesion left)."
                         + ("\nGrey band: ALM / tjM1 / MOs_rest / MOp_rest (ROI v8 on the stat mask), from the basis twin "
                            "re-run (FROZEN_PIX_MUSALL, area-run alphas), placed after their parent." if set(SPLIT) & set(ar_)
                            else ""), fontsize=9.5)
            fig.subplots_adjust(left=0.05, right=0.995, top=1 - 1.45 / (2.7 * len(rows_) + 1.3), bottom=0.05, hspace=0.75)
            save_png_svg(fig, out / "figures" / f"frozen_claims_{name}_{w}.png", dpi=105)
            plt.close(fig)
    trend_figure(plt, C, beh, out)
    print("->", out)
    return 0


def trend_figure(plt, C: pd.DataFrame, beh: pd.DataFrame, out: Path) -> Path:
    """Fig 0.11: spout unique R2 over days (cortex, lick window) in each frozen frame, and far-contra contact, ALL animals
    pooled per panel. EPOCH-COLOUR CONVENTION (A13, Priya 11-07; before: one panel per animal, lines across days):
    one OPEN marker per session, shape = animal, colour = epoch, no line joining sessions; per epoch a filled dot in the
    epoch colour +- SEM over animals (`cat_points.day_dots`). Plotting only."""
    from wfield_local import cat_points as cp
    from wfield_local import epochs
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, k in zip(axs[:2], ("noFR", "FR-A")):
        g = C[(C.frame == k) & (C.area == "cortex") & (C.window == "lick")]
        cp.day_dots(ax, g.day, g.u, g.animal, g.epoch)
        ax.set_title(f"spout unique R2, frozen frame {k}", fontsize=10)
        ax.set_ylabel("frozen spout unique R2 (cortex, lick)" if k == "noFR" else "")
    b = beh.assign(day=beh.label.map(epochs.days_since_stroke), animal=beh.label.str.split("_").str[0])
    b = b.assign(epoch=b.label.map(epochs.epoch_of))
    cp.day_dots(axs[2], b.day, b.contact_far_contra, b.animal, b.epoch)
    axs[2].set_ylim(0, 1.02)
    cp.fit_ylim(axs[2], b.contact_far_contra)
    axs[2].set_title("far-contra contact (behaviour)", fontsize=10)
    axs[2].set_ylabel("far-contra contact fraction")
    for ax in axs:
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("day from stroke")
        ax.spines[["top", "right"]].set_visible(False)
    eps = [e for e in cp.EPOCH_ORDER if (C.epoch == e).any()]
    fig.legend(handles=cp.legend_handles(sorted(C.animal.unique()), eps, epoch_label="{e} mean +- SEM"), loc="center left",
               bbox_to_anchor=(0.87, 0.5),
               fontsize=8, frameon=False)
    fig.suptitle("NEURAL + LICK: recovery course in the FROZEN frame -- spout unique R2 (target + executed spout part) per "
                 "session, both frames, and far-contra contact.\nOpen markers = sessions (shape = animal, colour = epoch); "
                 "filled = epoch mean +- SEM over animals (each animal's sessions averaged first).", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.86, top=0.82, bottom=0.14, wspace=0.22)
    p = out / "figures" / "frozen_claims_trend.png"
    save_png_svg(fig, p, dpi=105)
    plt.close(fig)
    return p


if __name__ == "__main__":
    raise SystemExit(main())
