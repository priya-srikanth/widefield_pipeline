"""FROZEN pre-stroke encoder WITH vs WITHOUT FaceRhythm (handoff 000t plan item 4, Priya 10-25: "should we be mixing frozen
and refit encoder vs FR models" -> frozen-with-FR run). Same frame for every number here: one model per animal fitted on
its August pre sessions (pre scored leave-one-session-out), applied unchanged to every post session.

FRAMES  noFR = final frame `frozen_encoder_gate-attempted_v7g3_t3d` (joint)
        FR-B = `..._frA` variant no_fr_dir (directionless FR: lick sum + other factors)
        FR-A = `..._frA` joint (all FR, side + distance included)
1. frozen R2 (held-out, fraction) per area x window: level per epoch, and epoch - pre (nested bootstrap, rule 8 table)
2. RESIDUAL (data - full, evoked, far-contra trials; x1e3 dF/F, `frozen_residual_trend.py` sessions tables): epoch - pre.
   Main question: is the chronic residual (cortex responding beyond what the pre-stroke mapping predicts from the
   movements made) partly unmodelled face movement? -> does it shrink with FR in the frozen model?
3. FR unique in the frozen frame: directionless (joint - no_fr) and direction (joint - no_fr_dir), level + change
4. LAYER 2 (`allsession_encoder`, combined fit: early pre = reference; pre_late = drift null in the same fit): change_ref of
   each post epoch minus pre_late's, noFR (`_v7g3_t3d_ref`) vs FR-A (`_v7g3_t3d_frA`); FR-A's change in the fr / fr_dir
   weights (how much the face -> cortex mapping itself changes). Layer 2 CSVs are x1e3 -> divided here.
5. RELABEL (`frozen_relabel`): far-contra trials relabelled to each spout, dSSE (relabelled - true; > 0 = the true target
   fits better), noFR vs FR-A.
Usage: python scripts/frozen_fr_compare.py
Output: <E>/frozen_fr_compare/ (tables + figures)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

RUNS = {"noFR": ("frozen_encoder_gate-attempted_v7g3_t3d", "joint"),
        "FR-B": ("frozen_encoder_gate-attempted_v7g3_t3d_frA", "no_fr_dir"),
        "FR-A": ("frozen_encoder_gate-attempted_v7g3_t3d_frA", "joint")}
FR_VARIANTS = ("joint", "no_fr", "no_fr_dir", "no_fr_all")
AREAS = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left"]
ALAB = {"cortex": "cortex", "SSp-m_right": "SSp-m contra", "SSp-m_left": "SSp-m ipsi", "MOp_right": "MOp contra",
        "MOp_left": "MOp ipsi"}
EP4 = ["pre", "acute", "subacute", "chronic"]
EPS = EP4[1:]
WINDOWS = ["cue", "lick"]                                 # cue before lick (Priya 10-25)
COL = {"noFR": "#999999", "FR-B": "#7570b3", "FR-A": "#e7298a"}


def session_r2(folder: str, variant: str, measure: str = "frame") -> pd.DataFrame:
    """Per session x window x area frozen R2; measure = frame (all frames of the window) | trial (single-trial window
    means) | avg (trial-averaged = PETH), as `frozen_encoder.musall_r2`."""
    rows = []
    for f in sorted(glob.glob(str(E / folder / variant / "sessions" / "*.json"))):
        m = json.loads(Path(f).read_text())
        for w in WINDOWS:
            fr = m["r2_musall"][w][measure]
            for a in AREAS:
                if a in m["areas"]:
                    rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "window": w, "area": a,
                                 "r2": fr[m["areas"].index(a)]})
    return pd.DataFrame(rows)


def boot(df: pd.DataFrame, val: str, ref_epoch: str = "pre", epochs=EPS, keys=("window", "area")) -> pd.DataFrame:
    """epoch - ref per animal (nested bootstrap, analysis_kit.boot_delta), per group of ``keys``."""
    from wfield_local import analysis_kit as ak
    rows = []
    for k, t in df.groupby(list(keys)):
        ref = {x: t[(t.epoch == ref_epoch) & (t.animal == x)][val].dropna().to_numpy() for x in AN}
        for ep in epochs:
            post = {x: t[(t.epoch == ep) & (t.animal == x)][val].dropna().to_numpy() for x in AN}
            com = [x for x in AN if len(ref[x]) and len(post[x])]
            if len(com) < 2:
                continue
            iv, d = ak.boot_delta({x: post[x] for x in com}, {x: ref[x] for x in com}, np.random.default_rng(0), draws=True)
            per = {x: float(post[x].mean() - ref[x].mean()) for x in com}
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "epoch": ep,
                         "ref": float(np.mean([ref[x].mean() for x in com])), "point": iv.point, "lo": iv.lo, "hi": iv.hi,
                         "p": float(ak.boot_p(d)), "n_up": sum(v > 0 for v in per.values()), "n": len(com), **per})
        # PRE-STROKE NOISE BAND (Priya 10-25: "there's no pre in those plots ... would it be helpful to see pre variability"):
        # random half-splits of each animal's ref sessions (>= 4), half B - half A, mean over animals, 500 splits -> 95 % range
        rng = np.random.default_rng(1)
        el = [x for x in AN if len(ref[x]) >= 4]
        if len(el) >= 2:
            nd = []
            for _ in range(500):
                dd = []
                for x in el:
                    v = rng.permutation(ref[x])
                    h = len(v) // 2
                    dd.append(v[h:].mean() - v[:h].mean())
                nd.append(np.mean(dd))
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "epoch": f"{ref_epoch}_split",
                         "point": float(np.median(nd)), "lo": float(np.percentile(nd, 2.5)),
                         "hi": float(np.percentile(nd, 97.5)), "n": len(el)})
    return pd.DataFrame(rows)


def bars(ax, B: pd.DataFrame, frames, area_list, ylabel, title, scale=1.0):
    """Grouped bars: x = area x epoch, one bar per frame; whisker 95 % CI; * CI excludes 0; dots = animals."""
    nf = len(frames)
    xt, xl = [], []
    band = False
    for i, a in enumerate(area_list):
        nb = B[B.area.eq(a) & B.epoch.astype(str).str.endswith("_split") & B.frame.isin(frames)]
        if not nb.empty:                                  # pre-stroke split-half 95 % range, widest over frames
            x_0, x_1 = i * (len(EPS) + 1) - 0.5, i * (len(EPS) + 1) + len(EPS) - 0.5
            ax.fill_between([x_0, x_1], nb.lo.min() * scale, nb.hi.max() * scale, color="#bbbbbb", alpha=0.35, lw=0,
                            zorder=0, label=None if band else "pre-stroke split-half 95 % range")
            band = True
        for j, ep in enumerate(EPS):
            x0 = i * (len(EPS) + 1) + j
            xt.append(x0); xl.append(f"{ep[:3]}" + ("\n" + ALAB[a] if j == 1 else "\n"))
            for k, fr in enumerate(frames):
                r = B[(B.frame == fr) & (B.area == a) & (B.epoch == ep)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = x0 + (k - (nf - 1) / 2) * 0.8 / nf
                ax.bar(x, r.point * scale, width=0.8 / nf, color=COL.get(fr, f"C{k}"), label=fr if (i == 0 and j == 0) else None)
                ax.plot([x, x], [r.lo * scale, r.hi * scale], color="k", lw=0.8)
                ax.scatter([x] * 4, [r.get(an, np.nan) * scale for an in AN], s=5, c="k", zorder=5)
                if r.lo > 0 or r.hi < 0:
                    ax.text(x, max(r.hi, 0) * scale, "*", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(title, fontsize=9)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = E / "frozen_fr_compare"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    # 1. frozen R2
    R = pd.concat([session_r2(f, v).assign(frame=k) for k, (f, v) in RUNS.items()])
    R.to_csv(out / "r2_sessions.csv", index=False)
    lev = R.groupby(["frame", "window", "area", "animal", "epoch"]).r2.mean().reset_index()
    L = lev.groupby(["frame", "window", "area", "epoch"]).r2.mean().unstack("epoch")[EP4]
    print("== 1. frozen R2 level (mean of animals)\n" + L.round(3).to_string())
    B1 = pd.concat([boot(R[R.frame == k], "r2").assign(frame=k) for k in RUNS])
    B1.to_csv(out / "r2_change.csv", index=False)

    # 2. residual
    res = []
    for k, (f, v) in RUNS.items():
        sfx = "" if v == "joint" else f"_{v}"
        t = pd.read_csv(E / f / f"residual_trend_sessions{sfx}.csv")
        res.append(t.assign(frame=k))
    Rs = pd.concat(res)
    Rs = Rs[Rs.region.isin(AREAS)].rename(columns={"region": "area"})
    per = Rs.groupby(["frame", "window", "area", "animal", "epoch"]).far_contra.mean().unstack("epoch")[EP4]
    print("\n== 2. residual far contra (x1e3), per animal (rule 8)\n" + per.round(1).to_string())
    B2 = pd.concat([boot(Rs[Rs.frame == k], "far_contra").assign(frame=k) for k in RUNS])
    B2.to_csv(out / "residual_change.csv", index=False)
    print("\n== residual epoch - pre\n" + B2[B2.area.isin(["SSp-m_right", "SSp-m_left", "cortex"])][
        ["frame", "window", "area", "epoch", "ref", "point", "lo", "hi", "n_up"]].round(2).to_string(index=False))

    # 3. FR unique, frozen frame
    U = {v: session_r2(RUNS["FR-A"][0], v).set_index(["label", "animal", "epoch", "window", "area"]).r2 for v in FR_VARIANTS}
    Uq = pd.DataFrame({"directionless (joint - no_fr)": U["joint"] - U["no_fr"],
                       "direction (joint - no_fr_dir)": U["joint"] - U["no_fr_dir"],
                       "all FR (joint - no_fr_all)": U["joint"] - U["no_fr_all"]}).reset_index()
    Uq.to_csv(out / "fr_unique_sessions.csv", index=False)
    UL = Uq.melt(id_vars=["label", "animal", "epoch", "window", "area"], var_name="frame", value_name="u")
    print("\n== 3. FR unique R2, frozen frame (mean of animals)\n" + UL.groupby(["frame", "window", "area", "animal", "epoch"])
          .u.mean().groupby(["frame", "window", "area", "epoch"]).mean().unstack("epoch")[EP4].round(4).to_string())
    B3 = pd.concat([boot(UL[UL.frame == k], "u").assign(frame=k) for k in UL.frame.unique()])
    B3.to_csv(out / "fr_unique_change.csv", index=False)

    # 4. Layer 2
    l2 = []
    for k, d in (("noFR", "allsession_encoder_v7g3_t3d_ref"), ("FR-A", "allsession_encoder_v7g3_t3d_frA")):
        t = pd.read_csv(E / d / "allsession_changes_by_session.csv")
        t = t[(t.fit == "combined") & t.area.isin(AREAS)].copy()
        for c in [c for c in t.columns if c.startswith("change")]:
            t[c] = t[c] / 1e3
        l2.append(t.assign(frame=k))
    L2 = pd.concat(l2)
    B4 = pd.concat([boot(L2[L2.frame == k], "change_ref", ref_epoch="pre_late", keys=("area",)).assign(frame=k)
                    for k in ("noFR", "FR-A")])
    B4.to_csv(out / "layer2_change_vs_prelate.csv", index=False)
    print("\n== 4. Layer 2 change_ref, epoch - pre_late (R2 units)\n" + B4[["frame", "area", "epoch", "ref", "point", "lo", "hi",
                                                                          "n_up"]].round(4).to_string(index=False))
    fa = L2[L2.frame == "FR-A"]
    B4b = pd.concat([boot(fa, c, ref_epoch="pre_late", keys=("area",)).assign(frame=c.replace("change_", ""))
                     for c in ("change_fr", "change_fr_dir")])
    B4b.to_csv(out / "layer2_fr_weights_vs_prelate.csv", index=False)

    # 5. relabel
    rl = []
    for k, d in (("noFR", "frozen_relabel_v7g3_t3d"), ("FR-A", "frozen_relabel_v7g3_t3d_frA")):
        t = pd.read_csv(E / d / "relabel_by_animal.csv")
        rl.append(t[(t.spout == "far_contra") & (t.window == "lick") & t.area.isin(AREAS)].assign(frame=k))
    RL = pd.concat(rl)
    RL["dsse"] = RL.dsse_x1e3 / 1e3
    rlm = RL.groupby(["frame", "area", "as", "epoch"]).dsse.mean().unstack("epoch")[EP4]
    rlm.to_csv(out / "relabel_far_contra.csv")
    print("\n== 5. relabel far contra -> each spout, dSSE (mean of animals; > 0 = target fits better)\n"
          + rlm.loc[(slice(None), ["cortex", "SSp-m_right", "SSp-m_left"]), :].round(4).to_string())

    # figures
    for w in WINDOWS:
        fig, axs = plt.subplots(2, 2, figsize=(19, 9.5))
        b = B1[B1.window == w]
        bars(axs[0, 0], b, list(RUNS), AREAS, "change from pre (R2)", "1. frozen held-out R2, epoch - pre")
        lv = lev[(lev.window == w)].groupby(["frame", "area", "epoch"]).r2.mean()
        txt = "pre level: " + "; ".join(f"{fr} {lv[(fr, 'cortex', 'pre')]:.3f}" for fr in RUNS) + " (cortex)"
        axs[0, 0].text(0.01, 0.02, txt, transform=axs[0, 0].transAxes, fontsize=7.5)
        bars(axs[0, 1], B2[B2.window == w], list(RUNS), AREAS, "change from pre (x1e3 dF/F)",
             "2. RESIDUAL (data - full), far-contra trials, epoch - pre (> 0 = cortex responds MORE than predicted)")
        bars(axs[1, 0], B3[B3.window == w].replace({"frame": {"directionless (joint - no_fr)": "FR-B",
                                                               "direction (joint - no_fr_dir)": "FR-A",
                                                               "all FR (joint - no_fr_all)": "noFR"}}),
             ["FR-B", "FR-A", "noFR"], AREAS, "change from pre (R2)",
             "3. FR UNIQUE in the frozen frame, epoch - pre: purple = directionless, pink = direction, grey = all FR")
        bars(axs[1, 1], B4, ["noFR", "FR-A"], AREAS, "change_ref minus pre_late (R2)",
             "4. LAYER 2: how much each epoch's mapping changes beyond the pre-stroke drift (pre_late), all windows")
        h, lb = axs[0, 0].get_legend_handles_labels()
        nm = {"noFR": "noFR = final frame (no FaceRhythm)", "FR-B": "FR-B = frozen + directionless FR", "FR-A": "FR-A = frozen + all FR"}
        fig.legend(h, [nm.get(x, x) for x in lb],
                   loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.925))
        fig.suptitle(f"NEURAL: FROZEN pre-stroke encoder with vs without FaceRhythm, {w} window (panel 4: all frames). Bars = "
                     "mean of animals' epoch - pre (panel 4: - pre_late); whisker 95 % nested CI; * excludes 0; dots = animals.",
                     fontsize=10)
        fig.subplots_adjust(left=0.045, right=0.995, top=0.86, bottom=0.07, hspace=0.42, wspace=0.14)
        fig.savefig(out / "figures" / f"frozen_fr_compare_{w}.png", dpi=110)
        plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
