"""SIMPLE versions of the crowded KEY encoder bar charts + the correction-lick dot plot (Priya 10-29: "make simple
versions -- key areas only, per-animal dots + the saved CI, one panel per claim"). PLOTTING ONLY: every number is read
from the saved CSVs of the producing scripts; nothing is re-fit and no new statistic is computed.

  frozen_musall_<w>_simple.png         <- frozen_fr_compare/musall_change.csv   (scripts/frozen_fr_musall.py; Fig 0.2 / 0.8)
  frozen_claims_contest_<w>_simple.png <- frozen_fr_compare/claims_contest.csv  (scripts/frozen_claims.py;   Fig 0.3 / 0.9)
  frozen_claims_peth_<w>_simple.png    <- frozen_fr_compare/claims_peth.csv     (scripts/frozen_claims.py;   Fig 0.4 / 0.10)
  lick_correction_spouts_dots.png      <- iti_licks/lick_correction/corr_beh_{boot,per_animal}.csv (scripts/lick_correction.py)
  partial_refit_cue_numbers.md         <- <frA run>/partial_refit/boot.csv      (scripts/partial_refit.py;   Fig 1.12)

Encoder panels: point = mean of animals' epoch - pre (the saved `point`), bar = the saved 95 % nested CI (`lo`, `hi`),
grey dots = the four animals (saved per-animal columns), * = CI excludes 0 (`wfield_local.sig_marks.sig_star`), k/n =
saved `n_up` / `n`. Marker: filled circle = contralesional (right, lesion left), open circle = ipsilesional, square =
whole cortex, diamond = contra - ipsi. Colour = epoch (lick_enlv_curves / orofacial_cohort.EPOCH_LINE).
ALM / tjM1 rows exist only in the claims CSVs (FROZEN_PIX_MUSALL basis re-runs); musall_change.csv has cortex, SSp-m and
MOp only, and no saved table has SSp-n for these encoder quantities -- the simple figures say so rather than substitute.
  frozen_musall_<w>_simple_v2.png      <- frozen_fr_compare/musall_change_v2.csv (scripts/frozen_fr_musall_v2.py, 10-30):
                                          adds SSp-n (area run) and ALM / tjM1 (basis twin), target + spout, both frames;
                                          plotted rows saved as frozen_musall_<w>_simple_v2_data.csv.
Usage: python scripts/simple_key_figures.py [--only-v2] [--copy-to _goal_page]
"""
from __future__ import annotations
from wfield_local.frozen_encoder import design_tag as _design_tag  # frame tag of the FROZEN_LICKS_ALL design (8.51)

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPOCH_COL = {"pre": "k", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}   # = EPOCH_LINE
FC = E / "frozen_fr_compare"
FIG_FC = FC / "figures"
LC = E / "iti_licks" / "lick_correction"
FIG_LC = E / "iti_licks" / "figures"
PR = E / _design_tag("frozen_encoder_gate-attempted_v7g3_t3d_frA") / "partial_refit"
KEY_AREAS = [("cortex", "cortex"), ("ALM_right", "ALM"), ("ALM_left", "ALM"), ("tjM1_right", "tjM1"),
             ("tjM1_left", "tjM1"), ("SSp-m_right", "SSp-m"), ("SSp-m_left", "SSp-m")]
WLAB = {"cue": "cue window", "lick": "first-lick window"}


def _style(area: str) -> tuple[str, bool]:
    area = area.split("|")[0]
    if area == "cortex":
        return "s", True
    if area.endswith("_c-i"):
        return "D", True
    return "o", area.endswith("_right")


def _side(area: str) -> str:
    area = area.split("|")[0]
    return "" if area == "cortex" else ("contra-ipsi" if area.endswith("_c-i") else
                                       ("contra" if area.endswith("_right") else "ipsi"))


def dot_panel(ax, B: pd.DataFrame, items: list, title: str, ylabel: str | None) -> tuple[float, float]:
    """items = [(x, area, short name, epoch)]; B rows keyed by area + epoch (one row each). Returns the data range
    (CI ends, dots, 0) so the caller can set one y range per row (`set_range`)."""
    from wfield_local.sig_marks import sig_star
    rng = np.random.default_rng(0)
    vals = [0.0]
    for x, area, _nm, ep in items:
        r = B[(B.area == area) & (B.epoch == ep)]
        if r.empty:
            continue
        r = r.iloc[0]
        col = EPOCH_COL[ep]
        mk, filled = _style(area)
        dots = np.array([r.get(a, np.nan) for a in AN], float)
        vals += [r.lo, r.hi, *dots[np.isfinite(dots)]]
        ax.scatter(x + 0.18 + rng.uniform(-0.04, 0.04, 4), dots, s=10, c="0.55", zorder=3, lw=0)
        ax.plot([x, x], [r.lo, r.hi], color=col, lw=1.6, zorder=4)
        ax.plot(x, r.point, marker=mk, ms=7, color=col, mfc=col if filled else "white", mew=1.6, zorder=5, ls="none")
        sig_star(ax, x, r.lo, r.hi, dots, fontsize=12, color=col)
        ax.annotate(f"{int(r.n_up)}/{int(r.n)}", (x, 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color="0.35")
    ax.axhline(0, color="0.5", lw=0.7, zorder=1)
    ax.set_xticks([it[0] for it in items], [f"{it[2]}\n{_side(it[1])}" for it in items], fontsize=7.5)
    ax.set_xlim(min(it[0] for it in items) - 0.6, max(it[0] for it in items) + 0.7)
    ax.set_title(title, fontsize=9, pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return float(np.nanmin(vals)), float(np.nanmax(vals))


def set_range(axs, rngs) -> None:
    """One y range over the given axes, with room below for the k/n row and above / below for the stars."""
    lo, hi = min(r[0] for r in rngs), max(r[1] for r in rngs)
    d = hi - lo
    for ax in axs:
        ax.set_ylim(lo - 0.16 * d, hi + 0.10 * d)


def _wrap(t: str, n: int = 105) -> str:
    import textwrap
    return textwrap.fill(t, n)


def _foot(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=7, color="0.3", ha="left", va="bottom", wrap=True)


def fig_musall(plt, w: str, claim: str) -> Path:
    B = pd.read_csv(FC / "musall_change.csv")
    B = B[(B.window == w) & (B.epoch == "chronic")]
    areas = [("cortex", "cortex"), ("SSp-m_right", "SSp-m"), ("SSp-m_left", "SSp-m"), ("SSp-m_c-i", "SSp-m")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, (q, lab) in zip(axs, [("target", "(a) position (target) unique R², chronic - pre"),
                                  ("spout", "(b) spout unique R² (target + executed spout part), chronic - pre")]):
        items, x = [], 0.0
        for _fr in ("noFR", "FR-A"):
            for a, nm in areas:
                items.append((x, a, nm, "chronic"))
                x += 1
            x += 0.8
        b = pd.concat([B[(B.q == q) & (B.frame == fr) & B.area.isin([a for a, _ in areas])]
                       .assign(area=lambda d, fr=fr: d.area if fr == "noFR" else d.area + "|FR") for fr in ("noFR", "FR-A")])
        items = [(it[0], it[1] if i < len(areas) else it[1] + "|FR", it[2], it[3]) for i, it in enumerate(items)]
        b2 = b.copy()
        set_range([ax], [dot_panel(ax, b2, items, lab, "change from pre (R²)")])
        ax.set_xticklabels([f"{it[2]}\n{_side(it[1].split('|')[0])}" for it in items], fontsize=7.5)
        ax.text(1.5, 1.0, "frozen model", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        ax.text(6.3, 1.0, "frozen + FaceRhythm", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        ax.axvline(3.9, color="0.85", lw=0.8)
    fig.suptitle(f"{_wrap(claim)}\nFrozen pre-stroke encoder, {WLAB[w]}, chronic epoch. PROVISIONAL, n = 4", fontsize=10)
    _foot(fig, "Marker = mean of animals' chronic - pre; bar = 95% nested CI (animals > sessions); grey dots = animals; "
               "* CI excludes 0; k/n = animals above 0. Filled = contralesional, open = ipsilesional, square = cortex, "
               "diamond = contra - ipsi. ALM / tjM1 / SSp-n are not in the saved table for these quantities "
               "(musall_change.csv). Source: scripts/frozen_fr_musall.py.")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.2, wspace=0.18)
    p = FIG_FC / f"frozen_musall_{w}_simple.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


V2_AREAS = [("cortex", "cortex")] + [(f"{b}_{s}", b) for b in ("SSp-m", "SSp-n", "ALM", "tjM1", "MOp")
                                     for s in ("right", "left", "c-i")]


def fig_musall_v2(plt, w: str, claim: str) -> Path:
    """v2 of fig_musall (Priya 10-30: add SSp-n, ALM, tjM1): musall_change_v2.csv (scripts/frozen_fr_musall_v2.py, a
    re-summary of the saved per-session outputs; ALM / tjM1 from the FROZEN_PIX_MUSALL basis twin). Rows = target /
    spout unique, columns = frame (frozen / frozen + FaceRhythm), chronic - pre. The plotted rows are saved beside the PNG."""
    B = pd.read_csv(FC / "musall_change_v2.csv")
    B = B[(B.window == w) & (B.epoch == "chronic") & B.area.isin([a for a, _ in V2_AREAS])]
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.4), sharey="row")
    used = []
    for i, (q, lab) in enumerate([("target", "position (target) unique R²"),
                                  ("spout", "spout unique R² (target + executed spout part)")]):
        rr = []
        for j, (fr, frl) in enumerate([("noFR", "frozen model"), ("FR-A", "frozen + FaceRhythm")]):
            b = B[(B.q == q) & (B.frame == fr)]
            used.append(b)
            items, x, prev = [], 0.0, None
            for a, nm in V2_AREAS:
                if prev is not None and nm != prev:
                    x += 0.6
                items.append((x, a, nm, "chronic"))
                prev, x = nm, x + 1
            rr.append(dot_panel(axs[i, j], b, items, f"({'abcd'[2 * i + j]}) {lab}, chronic - pre; {frl}",
                                "change from pre (R²)" if j == 0 else None))
            ax = axs[i, j]
            ax.set_xticklabels(["" if a == "cortex" else {"contra": "c", "ipsi": "i", "contra-ipsi": "c-i"}[_side(a)]
                                for _x, a, _n, _e in items], fontsize=7.5)
            for x0, a, nm, _ep in items:                 # area name once per group, under its three ticks
                if a == "cortex" or a.endswith("_right"):
                    ax.text(x0 + (0 if a == "cortex" else 1), -0.075 if a != "cortex" else -0.03, nm,
                            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)
                if a.endswith("_right") and a.startswith(("ALM", "tjM1")):
                    ax.axvspan(x0 - 0.45, x0 + 2.45, color="#f0f0f0", zorder=-1, lw=0)
        set_range(axs[i], rr)
    pd.concat(used).to_csv(FIG_FC / f"frozen_musall_{w}_simple_v2_data.csv", index=False)
    fig.suptitle(f"{_wrap(claim, 150)}\nFrozen pre-stroke encoder, {WLAB[w]}, chronic epoch. PROVISIONAL, n = 4",
                 fontsize=10)
    _foot(fig, "Marker = mean of animals' chronic - pre; bar = 95% nested CI (animals > sessions); grey dots = animals; "
               "* CI excludes 0; k/n = animals above 0. Filled = contralesional, open = ipsilesional, square = cortex, "
               "diamond = contra - ipsi (ticks: c = contra, i = ipsi, c-i). cortex / SSp-m / SSp-n / MOp: area run (Allen area on the stat mask); ALM / tjM1 "
               "(grey band): ROI v8 on the stat mask from the FROZEN_PIX_MUSALL basis twin (same route as the claims "
               "figures; checked per session on SSp-m / SSp-n / MOp: r >= 0.987, median |diff| <= 0.0004, worst single session 0.13). "
               "Source: scripts/frozen_fr_musall_v2.py -> musall_change_v2.csv.")
    fig.subplots_adjust(left=0.05, right=0.995, top=0.86, bottom=0.12, wspace=0.05, hspace=0.55)
    p = FIG_FC / f"frozen_musall_{w}_simple_v2.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def _grid(plt, B: pd.DataFrame, rows: list, w: str, claim: str, foot: str, name: str) -> Path:
    """rows = [(row title, selector dict)]; columns = acute / chronic; key areas, frozen final frame (noFR)."""
    fig, axs = plt.subplots(len(rows), 2, figsize=(11, 3.3 * len(rows) + 1.3), sharey="row", squeeze=False)
    for i, (rt, sel) in enumerate(rows):
        b = B[(B.window == w) & (B.frame == "noFR")]
        for k, v in sel.items():
            b = b[b[k] == v]
        rr = []
        for j, ep in enumerate(("acute", "chronic")):
            items = [(n, a, nm, ep) for n, (a, nm) in enumerate(KEY_AREAS)]
            rr.append(dot_panel(axs[i, j], b, items, f"{rt}, {ep.upper()} - pre",
                                "change from pre (R²)" if j == 0 else None))
        set_range(axs[i], rr)
    fig.suptitle(f"{_wrap(claim)}\nFrozen pre-stroke encoder (no FaceRhythm), {WLAB[w]}. PROVISIONAL, n = 4", fontsize=10)
    _foot(fig, "Marker = mean of animals' epoch - pre; bar = 95% nested CI (animals > sessions); grey dots = animals; "
               "* CI excludes 0; k/n = animals above 0. Filled = contralesional, open = ipsilesional, square = whole "
               "cortex. SSp-n is not in the saved table. " + foot)
    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 1.35 / (3.3 * len(rows) + 1.3), bottom=0.55 / (3.3 * len(rows) + 1.3) + 0.04,
                        wspace=0.06, hspace=0.5)
    p = FIG_FC / f"{name}_{w}_simple.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_contest(plt, w: str, claim: str) -> Path:
    B = pd.read_csv(FC / "claims_contest.csv")
    rows = [("(a) target unique R² (the spout beyond every executed regressor)", {"q": "target"}),
            ("(b) executed movement's spout-locked part, unique R²", {"q": "exec_locked"})]
    return _grid(plt, B, rows, w, claim, "Source: scripts/frozen_claims.py (claims_contest.csv).", "frozen_claims_contest")


def fig_peth(plt, w: str, claim: str) -> Path:
    B = pd.read_csv(FC / "claims_peth.csv")
    rows = [("(a) target unique R² on the PETH (spout-mean traces)", {"q": "target", "measure": "avg"}),
            ("(b) target unique R² trial-to-trial (around each spout's mean)", {"q": "target", "measure": "trial"}),
            ("(c) movement unique R² on the PETH", {"q": "movement", "measure": "avg"})]
    return _grid(plt, B, rows, w, claim, "Source: scripts/frozen_claims.py (claims_peth.csv).", "frozen_claims_peth")


def fig_lick_dots(plt) -> Path:
    B = pd.read_csv(LC / "corr_beh_boot.csv")
    PA = pd.read_csv(LC / "corr_beh_per_animal.csv")
    B, PA = B[B.order == "any"], PA[PA.order == "any"]
    eps = ["pre", "acute", "subacute", "chronic"]
    groups = [("far_contra", "far contra", True, -0.13), ("other5", "other five spouts", False, 0.13)]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (m, title, yl) in zip(axs, [
            ("miss_rate", "(a) miss rate of lick k (no spout contact)", "fraction of licks"),
            ("step_b", "(b) extra step toward the target after a MISS vs a HIT\nat the same error "
             "(previous lick's error controlled)", "deg toward target")]):
        for g, lab, filled, off in groups:
            b = B[(B.group == g) & (B.metric == m)].set_index("epoch").reindex(eps)
            pa = PA[PA.group == g]
            for i, ep in enumerate(eps):
                r = b.loc[ep]
                if not np.isfinite(r.level):
                    continue
                col = EPOCH_COL[ep]
                x = i + off
                v = pa[pa.epoch == ep].set_index("animal")[f"{m}_mean"].reindex(AN).to_numpy(float)
                ax.scatter(np.full(4, x + (0.07 if off > 0 else -0.07)), v, s=9, c="0.6", lw=0, zorder=3)
                ax.plot([x, x], [r.level_lo, r.level_hi], color=col, lw=1.6, zorder=4)
                ax.plot(x, r.level, "o", ms=7, color=col, mfc=col if filled else "white", mew=1.6, zorder=5)
                if ep != "pre" and np.isfinite(r.delta) and (r.delta_lo > 0 or r.delta_hi < 0):
                    top = np.nanmax([r.level_hi, *v])
                    ax.text(x, top, "*", color=col, ha="center", va="bottom", fontsize=13, fontweight="bold")
                if ep != "pre" and int(r.n_animals) < 4:
                    ax.annotate(f"{int(r.n_animals)} animals", (x, r.level_lo), xytext=(0, -9), textcoords="offset points",
                                ha="center", fontsize=6.5, color="0.35")
            # direct series label at the chronic end (no legend on the data)
            r = b.loc["chronic"]
            ax.annotate(lab + (" (filled)" if filled else " (open)"), (3.32, r.level), xytext=(0, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color="0.2")
        ax.axhline(0, color="0.6", lw=0.6, zorder=1)
        ax.set_xticks(range(4), eps, fontsize=8)
        for t, ep in zip(ax.get_xticklabels(), eps):
            t.set_color(EPOCH_COL[ep])
        ax.set_xlim(-0.5, 4.75)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="y", labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Correction licks: misses rise acutely only at the far contra spout; after a miss the next lick "
                 "corrects LESS than after a hit pre-stroke,\nand this deficit shrinks after the stroke "
                 "(far contra chronic +4.5 deg vs pre, 4/4). Cued post-cue lick pairs k -> k+1. PROVISIONAL, n = 4",
                 fontsize=9.5)
    _foot(fig, "Marker = pooled session level, bar = 95% nested bootstrap CI (saved corr_beh_boot.csv, all lick orders); "
               "grey dots = animals (session means, corr_beh_per_animal.csv); * = change vs pre CI excludes 0. Session "
               "cells need >= 10 misses and >= 10 hits; per-spout close cells (median 1-13 misses / session) are not "
               "shown, the five non-far-contra spouts are the script's pooled group. Source: scripts/lick_correction.py.")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.2, wspace=0.28)
    p = FIG_LC / "lick_correction_spouts_dots.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def _fmt(r) -> str:
    return f"{r.point:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] {int(r.n_up)}/{int(r.n)}"


def partial_refit_md() -> Path:
    B = pd.read_csv(PR / "boot.csv")
    B = B[B.window == "cue"]
    sets = [("target", "target refit"), ("exec", "executed position refit"), ("exec_fr", "executed + FR side / distance"),
            ("target_exec", "target + executed"), ("target_exec_fr", "target + executed + FR"),
            ("target_perm", "NULL target (block permutation)"), ("exec_perm", "NULL executed (block permutation)"),
            ("target_shuf", "NULL target (half-session shift)"), ("exec_shuf", "NULL executed (half-session shift)"),
            ("target - null", "target minus its null"), ("exec - null", "executed minus its null"),
            ("target - shift null", "target minus shift null"), ("exec - shift null", "executed minus shift null"),
            ("exec - target", "executed minus target")]
    areas = [("cortex", "cortex"), ("SSp-m_right", "SSp-m contra"), ("SSp-m_left", "SSp-m ipsi"),
             ("MOp_right", "MOp contra"), ("MOp_left", "MOp ipsi")]
    L = ["# Fig 1.12 partial refit, CUE window: numbers behind partial_refit_bars_cue.png", "",
         "Source: `boot.csv` in this folder (scripts/partial_refit.py, frame `frozen_encoder_gate-attempted_v7g3_t3d_frA`). "
         "Value = R2 gain of the refit set over the frozen model, as epoch - pre within animal: mean of animals "
         "[95% nested CI animals > sessions] k/n animals with a positive change. Nulls refit the same columns with the "
         "spout labels block-permuted (perm, 5 draws) or shifted by half a session (shuf). Copied from the saved "
         "table, no new statistics. PROVISIONAL (n = 4).", ""]
    for a, an in areas:
        L += [f"## {an}", "", "| refit set | acute | subacute | chronic | pre split-half 95% range |", "|---|---|---|---|---|"]
        for q, ql in sets:
            t = B[(B.area == a) & (B.q == q)].set_index("epoch")
            if t.empty:
                continue
            cells = [_fmt(t.loc[e]) if e in t.index else "" for e in ("acute", "subacute", "chronic")]
            ns = t.loc["pre_split"] if "pre_split" in t.index else None
            cells.append(f"[{ns.lo:+.3f}, {ns.hi:+.3f}]" if ns is not None else "")
            L.append(f"| {ql} | " + " | ".join(cells) + " |")
        L.append("")
    p = PR / "partial_refit_cue_numbers.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


CLAIMS = {
    ("musall", "cue"): "Position (target) unique R² does not rise chronically; spout unique shifts "
                       "contralesionally (SSp-m contra - ipsi, 4/4)",
    ("musall", "lick"): "Lick window: no chronic target rise; spout unique shifts contralesionally with FaceRhythm "
                        "(SSp-m contra - ipsi +0.048, 4/4)",
    ("contest", "cue"): "Target unique R² falls acutely and stays down chronically; the executed spout-locked part "
                        "rises acutely (cortex 4/4) and stays up in ALM and tjM1",
    ("contest", "lick"): "Lick window: target unique falls and stays down (0-1/4 up); the chronic executed spout-locked "
                         "gain sits in ALM and contralesional tjM1 (4/4)",
    ("peth", "cue"): "The target loss is a PETH loss that does not recover chronically (ALM, contralesional tjM1, 0/4 "
                     "up); trial-to-trial target unique stays near 0; tjM1 gains movement PETH unique (4/4)",
    ("peth", "lick"): "Lick window: target PETH unique is lost chronically in every key area (0-1/4 up); "
                      "contralesional ALM and tjM1 gain movement PETH unique (4/4)",
}


CLAIMS_V2 = {
    "cue": "Target unique R² does not rise chronically in any key area (contralesional ALM 0/4 up in both frames); spout unique shifts "
           "contralesionally in SSp-m and tjM1 (contra - ipsi 4/4 with FaceRhythm), not detectably in SSp-n or ALM",
    "lick": "Lick window: target unique stays below pre chronically (contralesional ALM, SSp-n, tjM1 1/4 up); spout unique shifts "
            "contralesionally in SSp-m and tjM1 (contra - ipsi 4/4 with FaceRhythm), not detectably in SSp-n or ALM",
}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    if "--only-v2" in sys.argv:                  # the v2 figures alone (leaves every other output untouched)
        out = [fig_musall_v2(plt, w, CLAIMS_V2[w]) for w in ("cue", "lick")]
        print("\n".join(str(p) for p in out))
        if "--copy-to" in sys.argv:
            dst = Path(sys.argv[sys.argv.index("--copy-to") + 1])
            for p in out:
                shutil.copy2(p, dst / p.name)
        return 0
    for w in ("cue", "lick"):
        out.append(fig_musall(plt, w, CLAIMS[("musall", w)]))
        if (FC / "musall_change_v2.csv").exists():
            out.append(fig_musall_v2(plt, w, CLAIMS_V2[w]))
        out.append(fig_contest(plt, w, CLAIMS[("contest", w)]))
        out.append(fig_peth(plt, w, CLAIMS[("peth", w)]))
    out.append(fig_lick_dots(plt))
    md = partial_refit_md()
    print("\n".join(str(p) for p in out), "\n", md)
    if "--copy-to" in sys.argv:
        dst = Path(sys.argv[sys.argv.index("--copy-to") + 1])
        for p in out:
            shutil.copy2(p, dst / p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
