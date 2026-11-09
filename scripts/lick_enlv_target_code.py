"""TARGET CODE WITHOUT REWARD on ENLv licks (Priya 10-29, item 2). ENLv = ENL-violation lick (pre_cue context: spout
stationary at the trial's spout, no cue, no reward); never 'ITI'. Same tests as lick_balance_target_model.py (imported,
not copied -- rule 9), on the PRIMARY ENLv set of lick_enlv_balance_offset.py (set "enlv") AND its bout starts (set
"enlv_bouts", clean_start: >= 1 s since any lick = the ITI first-after-quiet analog) (onset >= 1.0 s after arrival, lick window
ends before the cue, raw signals; target = the trial's spout, which is in place):
  1 PRE models of the post-lick primary-pool response c, i, d = c - i (ALM + tjM1 + SSp-m + SSp-n + SSs, 0.1..0.45 s
    minus -1..-0.5 s): angle line / cubic, TARGET (6 levels), angle + target, mix; cvR2 leave-one-pre-session-out; target
    weight w = a / (a + b) of y = k + a * tgt + b * angle (0 = landed angle only, 1 = target only), session-bootstrap CI.
  2 angle-matched target effect g (per deg of target at the same 5-deg landed-angle bin) per epoch.
  3 balance with a target-blind (angle) vs target-aware (both / mix) pre reference: change vs pre (all spouts, far contra).
  5 decoder: per frozen reader (ITI, cued, ENLv-trained) the same-place read-out (lick_enlv_balance_offset off_<r>), pre
    within-session x spout slope on the landing error lerr_mm -> w_dec = -slope (target component of the decoder: > 0 =
    read toward the spout rather than where the lick landed); session offset minus k * lerr (off_resid) by epoch.
The SAME TARGET + SAME LANDING test (bal_spout, offsp_<r>) is in lick_enlv_balance_offset (enlv_tests.csv).
Compared with the cued / ITI rows of baltarget_premodel / _weights / _boot / _decoder / _matched (OUT root).
Per-animal first (rule 8); change vs pre = analysis_kit.boot_delta.
Outputs <OUT_W>/enlv_licks/enlv_target_{premodel,weights,matched,boot,decoder,sessions}.csv; figure
figures/lick_enlv_target.png (figdata/lick_enlv_target.pkl; --plot-only).
Usage: python scripts/lick_enlv_target_code.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import iti_licks as il  # noqa: E402
import lick_balance_target_model as tm  # noqa: E402
import lick_enlv_balance_offset as bo  # noqa: E402

OUT = bo.OUT
AN = bo.AN
EPS = bo.EPS
MK = bo.MK
SETCOL = {"trial": "#555555", "iti": "#1f78b4", "enlv": "#d95f02", "enlv_bouts": "#8c2d04"}
SETLAB = {"trial": "cued first lick", "iti": "ITI first after quiet", "enlv": "ENLv (primary, all licks)",
          "enlv_bouts": "ENLv bout starts (>= 1 s quiet)"}
# ENLv sets: the PRIMARY (all licks, onset >= 1 s) and its BOUT STARTS (the analogue of ITI first-after-quiet)
ENLV_SETS = {"enlv": bo.PRIMARY, "enlv_bouts": bo.BOUTS}


def licks(variant: str, setname: str) -> pd.DataFrame:
    import likeforlike as lfl
    L = pd.read_csv(OUT / "enlv_table.csv.gz")
    for c in ("first", "clean_start", "contact"):
        L[c] = L[c].astype(bool)
    L = L[bo.select(L, variant)].dropna(subset=["angle", "c", "i", "act"])
    L = L[L.pos.isin(bo.POS)].reset_index(drop=True)
    L["set"], L["pre"], L["d"] = setname, L.epoch == "pre", L.c - L.i
    L["aim"], L["iti_ang"] = np.nan, np.nan
    sig = pd.read_csv(OUT / "enlv_sigma.csv").set_index("reader").sigma_mm
    for r in bo.READERS:
        R = lfl.ref_table(L.assign(_y=L[f"dec_{r}"]), "act", "_y", float(sig[r]))
        L[f"off_{r}"] = L[f"dec_{r}"] - R.ref
    return L


def decoder(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    rows, srows = [], []
    for r in bo.READERS:
        for (st, an), g in L.groupby(["set", "animal"]):
            g = g.copy()
            pre = g[g.pre]
            g["lerr_mm"] = g.act - g.pos.map(pre.groupby("pos").act.mean())
            pre = g[g.pre]
            xw = pre.lerr_mm - pre.groupby(["label", "pos"]).lerr_mm.transform("mean")
            yw = pre[f"off_{r}"] - pre.groupby(["label", "pos"])[f"off_{r}"].transform("mean")
            k = float((xw * yw).sum() / (xw ** 2).sum())
            sg = g.groupby(["label", "epoch"]).agg(off=(f"off_{r}", "mean"), lerr_mm=("lerr_mm", "mean"),
                                                   n=("act", "size")).reset_index()
            sg = sg[sg.n >= bo.MIN_S]
            sg["off_resid"] = sg.off - k * sg.lerr_mm
            sg["reader"], sg["animal"], sg["set"] = r, an, st
            srows.append(sg)
            rows.append({"set": st, "reader": r, "animal": an, "k_pre_within": k, "w_dec": -k})
    Sd = pd.concat(srows, ignore_index=True)
    brow = []
    for (st, r), g in Sd.groupby(["set", "reader"]):
        for c in ("off", "off_resid", "lerr_mm"):
            pre = {a: h[c].to_numpy() for a, h in sorted(g[g.epoch == "pre"].groupby("animal"))}
            for ep in EPS[1:]:
                post = {a: h[c].to_numpy() for a, h in sorted(g[g.epoch == ep].groupby("animal")) if a in pre}
                ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                d = {a: post[a].mean() - pre[a].mean() for a in post}
                brow.append({"set": st, "reader": r, "measure": c, "epoch": ep, "point": ci.point, "lo": ci.lo, "hi": ci.hi,
                             "n_same": int(sum(np.sign(v) == np.sign(ci.point) for v in d.values())), "n": len(d),
                             **d})
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(brow)], ignore_index=True), Sd


def compute() -> dict:
    L = pd.concat([licks(v, s) for s, v in ENLV_SETS.items()], ignore_index=True)
    CV, W = tm.premodels(L)
    Lp = tm.predict_all(L)
    S, _E = tm.sessions(Lp)
    B = tm.boots(S)
    M = tm.matched(Lp, W)
    D, Sd = decoder(L)
    for name, df in (("premodel", CV), ("weights", W), ("sessions", S), ("boot", B), ("matched", M),
                     ("decoder", D), ("decoder_sessions", Sd)):
        df.to_csv(OUT / f"enlv_target_{name}.csv", index=False)
    ref = {k: pd.read_csv(il.OUT / f"baltarget_{k}.csv") for k in ("premodel", "weights", "boot", "decoder", "matched")}
    F = {"CV": pd.concat([ref["premodel"], CV], ignore_index=True),
         "W": pd.concat([ref["weights"], W], ignore_index=True),
         "B": pd.concat([ref["boot"], B], ignore_index=True),
         "M": pd.concat([ref["matched"], M], ignore_index=True),
         "D": D, "Dref": ref["decoder"]}
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print("== PRE cvR2 (leave-one-pre-session-out), per animal\n" + F["CV"][F["CV"].y == "d"].pivot_table(
        index=["set", "animal"], columns="model", values="cvR2").round(3).to_string())
    print("== target weight w (pre), per animal\n" + F["W"][["set", "animal", "y", "w", "w_lo", "w_hi",
                                                             "within_over_total"]].round(2).to_string(index=False))
    print("== balance by reference, change vs pre\n" + F["B"][F["B"].measure.isin(
        ["bal_angle", "bal_both", "bal_mix", "lerr"])].round(3).to_string(index=False))
    print("== angle-matched target effect\n" + F["M"][F["M"].y == "d"][["set", "animal", "epoch", "n_matched",
                                                                        "g_target_per_deg", "w_matched"]]
          .round(3).to_string(index=False))
    print("== decoder target component (w_dec) + offset residual\n" + D.round(3).to_string(index=False))
    return F


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(20, 11), squeeze=False)
    sets = ["trial", "iti", "enlv", "enlv_bouts"]
    # (a) pre cvR2 for d by model
    ax = axs[0, 0]
    models = ["angle", "target", "both", "mix"]
    CV = F["CV"][F["CV"].y == "d"]
    for k, st in enumerate(sets):
        for j, m in enumerate(models):
            v = CV[(CV.set == st) & (CV.model == m)].set_index("animal").reindex(AN).cvR2
            x = j + (k - 1.5) * 0.2
            ax.bar(x, v.mean(), 0.18, color=SETCOL[st], alpha=0.35, label=SETLAB[st] if j == 0 else None)
            for a in AN:
                ax.plot(x, v[a], MK[a], color=SETCOL[st], ms=4, mfc="none")
    ax.set_xticks(range(len(models)), models)
    ax.set_title("PRE cvR2 of d = contra - ipsi post-lick response\n(leave-one-pre-session-out; bar = mean, markers = "
                 "animals)", fontsize=9)
    # (b) target weight w per animal, y = c, i, d
    ax = axs[0, 1]
    W = F["W"]
    for k, st in enumerate(sets):
        for j, y in enumerate(("c", "i", "d")):
            g = W[(W.set == st) & (W.y == y)].set_index("animal").reindex(AN)
            for q, a in enumerate(AN):
                x = j + (k - 1.5) * 0.22 + (q - 1.5) * 0.05
                ax.errorbar(x, g.w[a], yerr=[[g.w[a] - g.w_lo[a]], [g.w_hi[a] - g.w[a]]], fmt=MK[a], color=SETCOL[st],
                            ms=4, capsize=1.5, lw=0.8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axhline(1, color="0.6", lw=0.6, ls=":")
    ax.set_ylim(-1.5, 2.5)
    ax.set_xticks(range(3), ["c (contra)", "i (ipsi)", "d = c - i"])
    ax.set_title("PRE target weight w = a / (a + b) (0 = landed angle only, 1 = target only)\nper animal, session-"
                 "bootstrap 95% CI (clipped at -1.5 / 2.5)", fontsize=9)
    # (c, d) balance by reference
    B = F["B"]
    for ax, pos in ((axs[0, 2], "all"), (axs[1, 0], "far_contra")):
        refs = ["bal_angle", "bal_both", "bal_mix"]
        for k, st in enumerate(sets):
            for j, m in enumerate(refs):
                g = B[(B.set == st) & (B.pos == pos) & (B.measure == m)].set_index("epoch").reindex(EPS[1:])
                x = np.arange(3) + (k - 1.5) * 0.22 + (j - 1) * 0.08
                ax.errorbar(x, g.point, yerr=[g.point - g.lo, g.hi - g.point], fmt=["o", "s", "^"][j],
                            color=SETCOL[st], ms=4, capsize=1.5, lw=0.8,
                            label=f"{SETLAB[st]}: {m}" if pos == "all" else None)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks(range(3), EPS[1:])
        ax.set_title(f"balance change vs pre, {pos}: angle (target-blind) vs both / mix (target-aware) reference\n"
                     "(+ = contralesional hemisphere larger; 95% nested bootstrap)", fontsize=9)
    # (e) decoder w_dec
    ax = axs[1, 1]
    D, Dr = F["D"], F["Dref"]
    Dr = Dr[Dr.animal.isin(AN)]
    groups = [("cued", "trial", Dr[Dr.set == "trial"]), ("ITI", "iti", Dr[Dr.set == "iti"])]
    rl = {"iti": "ITI rdr", "trial": "cued rdr", "enlv": "ENLv dec"}
    groups += [(f"{'ENLv' if s == 'enlv' else 'ENLv bouts'}\n{rl[r]}", s, D[(D.set == s) & (D.reader == r)
                                                                          & D.animal.isin(AN)])
               for s in ENLV_SETS for r in bo.READERS]
    for k, (_lab, s, g) in enumerate(groups):
        for a in AN:
            v = g[g.animal == a].w_dec
            if len(v):
                ax.plot(k, float(v.iloc[0]), MK[a], color=SETCOL[s], ms=5, mfc="none")
        ax.plot([k - 0.2, k + 0.2], [g.w_dec.astype(float).mean()] * 2, color=SETCOL[s], lw=2)
    ax.set_xticks(range(len(groups)), [x[0] for x in groups], fontsize=7)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_title("DECODER target component w_dec = -(pre within-spout slope of the same-place read-out on landing "
                 "error)\n> 0 = read toward the spout, not where the lick landed (markers = animals)", fontsize=9)
    # (f) angle-matched target effect g per epoch
    ax = axs[1, 2]
    M = F["M"][F["M"].y == "d"]
    for k, st in enumerate(sets):
        g = M[M.set == st]
        for a in AN:
            h = g[g.animal == a].set_index("epoch").reindex(EPS)
            ax.plot(np.arange(4) + (k - 1.5) * 0.1, h.g_target_per_deg, "-", marker=MK[a], color=SETCOL[st], lw=0.7,
                    ms=4, mfc="none")
        ax.plot(np.arange(4) + (k - 1.5) * 0.1, g.groupby("epoch").g_target_per_deg.mean().reindex(EPS), "-",
                color=SETCOL[st], lw=2.2, label=SETLAB[st])
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(range(4), EPS)
    ax.set_title("target effect on d at the SAME landed angle (5-deg bins)\nper deg of target (thin = animals, thick = "
                 "mean)", fontsize=9)
    for ax in axs.flat:
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle(textwrap.fill(
        "NEURAL: TARGET CODE WITHOUT REWARD -- ENLv licks (PRIMARY: onset >= 1.0 s after spout arrival, window before "
        "the cue; spout in place, no cue, no reward; all licks and bout starts) vs cued first licks and ITI first-after-quiet licks "
        "(lick_balance_target_model, same functions). Post-lick primary pool (ALM, tjM1, SSp-m, SSp-n, SSs) 0.1-0.45 s. "
        "Decoders = frozen pre-stroke readers (ITI, cued) and the frozen ENLv-trained decoder. PROVISIONAL", 220),
        fontsize=9.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hl = [Patch(color=SETCOL[st], label=SETLAB[st]) for st in sets]
    hl += [Line2D([], [], ls="", marker=mk_, color="0.3", mfc="none", label=f"{a}") for a, mk_ in MK.items()]
    hl += [Line2D([], [], ls="", marker=mk_, color="0.3", label=f"(c, d) {m}") for m, mk_ in
           zip(("bal_angle", "bal_both", "bal_mix"), ("o", "s", "^"))]
    fig.legend(handles=hl, loc="upper center", ncol=len(hl), fontsize=8.5, frameon=False, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    bo._header(fig, 2)
    save_png_svg(fig, bo.FIG / "lick_enlv_target.png", dpi=90)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The lick sets of lick_enlv_target.png (Priya 11-09): the cued / ITI rows of lick_balance_target_model (cued first
    post-cue lick, ITI first lick after >= il.GAP_MAIN s quiet) and the two ENLv sets (`ENLV_SETS`)."""
    from wfield_local import trial_set_text as tst
    have = "3-D angle, landing and pooled responses present"
    parts = [tst.cued_licks(first_only=True, extra=[f"[cued: {tm.SETS['trial']}]"]),
             tst.iti_licks(extra=[f"[ITI: {tm.SETS['iti']}]",
                                  f"no lick onset in the preceding {il.GAP_MAIN:g} s"])]
    parts += [bo.variant_sentence(v, extra=[f"spouts {', '.join(bo.POS)}", have]) for v in ENLV_SETS.values()]
    return tst.union(*parts)


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if figdata.plot_only():
        F = figdata.load_fd("lick_enlv_target", bo.FD)
    else:
        F = compute()
        figdata.save_fd("lick_enlv_target", F, bo.FD)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
