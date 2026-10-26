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
    from wfield_local.sig_marks import sig_star
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
                dots = [r.get(an, np.nan) * scale for an in AN]
                ax.scatter([x] * 4, dots, s=5, c="k", zorder=5)
                sig_star(ax, x, r.lo * scale, r.hi * scale, dots)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(title, fontsize=9)


def epoch_lines(ax, B: pd.DataFrame, series, area: str, scale: float = 1.0, key: str = "frame"):
    """EPOCH LINES (Priya 10-25: "the partial lines is much better" than grouped bars): x = pre (0 by construction, a
    change from pre) -> acute -> subacute -> chronic; per series a line (mean of animals), a shaded 95 % nested CI, faint
    per-animal lines; FILLED marker = CI excludes 0, OPEN = includes 0. ``series`` = [(value of B[key], label, colour)];
    B = ``boot`` output (columns area, epoch, point, lo, hi and one column per animal). Grey band = the pre-stroke
    split-half range when B has `<ref>_split` rows."""
    x = np.arange(len(EP4))
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    nb = B[B.area.eq(area) & B.epoch.astype(str).str.endswith("_split") & B[key].isin([s[0] for s in series])]
    if not nb.empty:
        ax.axhspan(nb.lo.min() * scale, nb.hi.max() * scale, color="#bbbbbb", alpha=0.25, lw=0, zorder=0)
    ns = len(series)
    for k, (q, _lab, col) in enumerate(series):
        t = B[(B[key] == q) & (B.area == area) & B.epoch.isin(EPS)].set_index("epoch").reindex(EPS)
        if t.point.isna().all():
            continue
        xs = x + (k - (ns - 1) / 2) * 0.06
        pt = np.r_[0, t.point.to_numpy() * scale]
        lo, hi = np.r_[0, t.lo.to_numpy() * scale], np.r_[0, t.hi.to_numpy() * scale]
        for an in AN:
            if an in t:
                ax.plot(xs, np.r_[0, t[an].to_numpy() * scale], color=col, lw=0.6, alpha=0.25)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(xs, pt, color=col, lw=2)
        sig = np.r_[False, ((t.lo > 0) | (t.hi < 0)).to_numpy()]
        ax.scatter(xs[sig], pt[sig], s=44, color=col, edgecolor="k", lw=0.6, zorder=5)
        ns_ = ~sig
        ns_[0] = False
        ax.scatter(xs[ns_], pt[ns_], s=38, facecolor="w", edgecolor=col, lw=1.5, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["pre", "acute", "sub", "chronic"], fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def lines_legend(fig, series, y: float = 0.01, split: bool = True):
    """One legend row under the panels: series colours + filled / open marker key + animals (+ pre split-half band)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=c, lw=2, label=lab) for _q, lab, c in series]
    h += [Line2D([], [], marker="o", ls="", mfc="0.4", mec="k", label="95% nested CI excludes 0"),
          Line2D([], [], marker="o", ls="", mfc="w", mec="0.4", mew=1.5, label="CI includes 0"),
          Line2D([], [], color="0.4", lw=0.6, alpha=0.5, label="each animal")]
    if split:
        h.append(Patch(color="#bbbbbb", alpha=0.4, label="pre-stroke split-half range"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=min(len(h), 6), frameon=False, fontsize=8.5)


def lines_figure(plt, w, B1, B2, B3, B4, lev, out):
    """MAIN figure (Priya 10-25: lines "much better" than grouped bars): rows = the four measures, columns = areas, one
    line per frame; the grouped-bar version stays as the supplement `frozen_fr_compare_<w>.png`."""
    fr3 = [(k, k, COL[k]) for k in RUNS]
    uq = [("directionless (joint - no_fr)", "FR directionless unique", COL["FR-B"]),
          ("direction (joint - no_fr_dir)", "FR direction unique", COL["FR-A"]),
          ("all FR (joint - no_fr_all)", "all FR unique", COL["noFR"])]
    rows = [(B1[B1.window == w], fr3, 1.0, "1. frozen R2\nepoch - pre"),
            (B2[B2.window == w], fr3, 1.0, "2. residual far contra\nepoch - pre (x1e3 dF/F)"),
            (B3[B3.window == w], uq, 1.0, "3. FR unique R2\nepoch - pre"),
            (B4, [s for s in fr3 if s[0] != "FR-B"], 1.0, "4. Layer 2 change_ref\nepoch - pre_late (all windows)")]
    fig, axs = plt.subplots(len(rows), len(AREAS), figsize=(17, 13), sharey="row")
    lv = lev[lev.window == w].groupby(["frame", "area", "epoch"]).r2.mean()
    for i, (B, ser, sc, ylab) in enumerate(rows):
        for j, a in enumerate(AREAS):
            ax = axs[i, j]
            epoch_lines(ax, B, ser, a, sc)
            if i == 0:
                ax.set_title(ALAB[a] + "\npre R2: " + " / ".join(f"{lv.get((fr, a, 'pre'), np.nan):.2f}" for fr in RUNS),
                             fontsize=9)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8.5)
            if i == 3:
                ax.set_xticklabels(["pre_late", "acute", "sub", "chronic"], fontsize=8)
            ax.tick_params(axis="y", labelsize=7.5)
    nm = {"noFR": "noFR = final frame (no FaceRhythm)", "FR-B": "FR-B = frozen + directionless FR",
          "FR-A": "FR-A = frozen + all FR"}
    lines_legend(fig, [(q, nm[q], c) for q, _l, c in fr3] + uq, y=0.005)
    fig.suptitle(f"NEURAL: FROZEN pre-stroke encoder with vs without FaceRhythm, {w} window (row 4: all frames). Line = mean "
                 "of animals' epoch - pre (row 4: - pre_late), band 95 % nested CI; title: pre R2 noFR / FR-B / FR-A.",
                 fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.91, bottom=0.075, hspace=0.35, wspace=0.12)
    fig.savefig(out / "figures" / f"frozen_fr_compare_lines_{w}.png", dpi=110)
    plt.close(fig)


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
        lines_figure(plt, w, B1, B2, B3, B4, lev, out)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
