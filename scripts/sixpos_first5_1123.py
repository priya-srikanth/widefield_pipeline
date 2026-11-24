"""CANDIDATE "licks 1-5" six-position all-lick kinematic scalars (Priya 11-23, candidate main-figure lick measure):
peak 3-D tongue speed (Fig 4B, `speed_all`) and peak 3-D reach (Fig 4C / S11A, `reach_all`) over the FIRST FIVE post-cue
licks of each trial (fused four-camera lick list, ``lick_idx < 5`` = order within the trial), session MEAN over those
licks, compared with production (session median over all licks) and the session mean over all licks.

Each endpoint keeps its production lick set and rules; only the lick subset and the session statistic change:
  * speed: `c2_lick3d_session_summary` set = every post-cue lick row of ``kinematics_fused_postcue`` (UNGATED), August
    pre only (`config.early_pre` dropped); n = licks with 3-D support (finite reach_mm); cell kept if n >= 20
    (`c2_scalar_family_aligned.lick3d`);
  * reach: `kinematics_pooled_figs` set = the camera working-period gate (`trial_classes.keep_working_period`, the
    `pose3d_summary_fig.gated_sessions` gate); production has NO minimum lick count for reach_all; the licks 1-5
    variant applies n >= 20 (reported, and the no-cut variant is also tested).
Change = session value minus the animal's own pre mean at that spout (August pre sessions). Test = the family's
`orofacial_stats.session_change_tests` (session MWU + 4,000-draw nested animal -> session `analysis_kit.boot_delta`,
seed 0, deltas_given) per endpoint; BH ONCE over the 162-cell scalar family, the other seven endpoints' rows taken from
the production family table (their p / CI do not depend on these two endpoints). Star = family q < 0.05 AND CI
excludes 0.

Stages (all outputs into --out, a NEW folder; refuses to overwrite files):
  ids    working-period trial ids per session (parallel, --jobs) -> working_ids.json
  summ   per session x position table: n / median / mean over all licks and over licks 1-5, both lick sets
  stats  the three variants' families + comparison table + figure

    python scripts/sixpos_first5_1123.py {ids,summ,stats,all} --out DIR [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
LICKS = P3D / "kinematics_fused_postcue/licks.csv.gz"
PROD_SUMMARY = P3D / "kinematics_fused_postcue/c2_lick3d_session_summary_ungated_cam3on.csv"
PROD_REACH_SESS = P3D / "figures/pooled_fused_cam3on/pose3d_sessions.csv"
STATS = Path("M:/MICROSCOPE/Priya/Widefield/manuscript_stats/c2_4cam_sync60_20261122")
PROD_FAMILY = STATS / "scalar_family_final_boot4000_cam3on/c2_scalar_family_aligned.csv"
#: 11-23 evening: the new production family (session means + first-lick speed, 180 cells; ccb/caption-review-1123)
FINAL_FAMILY = STATS / "scalar_family_final_boot4000_sessmean_speedfirst_cam3on/c2_scalar_family_aligned.csv"
LR_POS = dict(zip(["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"],
                  ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]))
POS = list(LR_POS.values())
EPOCHS = ("acute", "subacute", "chronic")
FIRST = 5
NMIN = 20
KEY = ["animal", "date", "session", "epoch", "position"]
VARIANTS = {"prod_median_all": "production (median, all licks)", "mean_all": "mean, all licks (new production)",
            "mean_first5": "mean, post-cue licks 1-5 (candidate)",
            "median_first5": "median, post-cue licks 1-5 (candidate)"}


def _ids(session: str):
    from wfield_local import trial_classes as tc
    return sorted(int(i) for i in tc.working_period_ids(session))


def _safe_write(df: pd.DataFrame, path: Path):
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
    df.to_csv(path, index=False, lineterminator="\n")
    print("wrote", path)


def _load_licks() -> pd.DataFrame:
    from wfield_local import config
    L = pd.read_csv(LICKS, usecols=["trial_id", "lick_idx", "position", "reach_mm", "speed_peak", "animal", "date",
                                    "session", "epoch"], dtype={"date": str})
    return L[~((L.epoch == "pre") & L.date.astype(str).map(config.early_pre))].reset_index(drop=True)


def stage_ids(out: Path, jobs: int):
    from wfield_local import parallel
    f = out / "working_ids.json"
    if f.exists():
        print("exists", f)
        return
    sess = sorted(pd.read_csv(LICKS, usecols=["session"]).session.astype(str).unique())
    res, fail = parallel.fan_out(sess, _ids, jobs=jobs, label="session")
    if fail:
        sys.exit(f"working-period ids failed: {fail[:3]}")
    json.dump({s: r for s, r in sorted(res)}, open(f, "w"))
    print("wrote", f, len(res), "sessions")


def _summ(L: pd.DataFrame, m: str) -> pd.DataFrame:
    rows = []
    for key, d in L.groupby(KEY, sort=True):
        f5 = d[d.lick_idx < FIRST]
        va, v5 = d[m].dropna(), f5[m].dropna()
        nr, nr5 = d.reach_mm.notna(), f5.reach_mm.notna()     # n = licks with 3-D support (production n_all)
        rows.append(dict(zip(KEY, key), n_all=int(nr.sum()), n_first5=int(nr5.sum()), n_val_all=len(va),
                         n_val_first5=len(v5), median_all=va.median(), mean_all=va.mean(), mean_first5=v5.mean(),
                         median_first5=v5.median(), n_trials=d.trial_id.nunique(),
                         n_trials_first5=f5[f5.reach_mm.notna()].trial_id.nunique()))
    return pd.DataFrame(rows)


def stage_summ(out: Path):
    L = _load_licks()
    ids = json.load(open(out / "working_ids.json"))
    keep = np.zeros(len(L), bool)
    sv, tid = L.session.astype(str).to_numpy(), L.trial_id.to_numpy(float)
    for s in np.unique(sv):
        m = sv == s
        keep[m] = np.isin(tid[m], np.asarray(ids.get(s, []), float))
    Lg = L[keep]
    print(f"licks {len(L)} (August pre + post), working-period gated {len(Lg)}")
    sp = _summ(L, "speed_peak").assign(endpoint="speed_all", lick_set="ungated")
    rc = _summ(Lg, "reach_mm").assign(endpoint="reach_all", lick_set="working_period")
    S = pd.concat([sp, rc], ignore_index=True)
    # reproduction checks against the production session tables (median, all licks)
    P = pd.read_csv(PROD_SUMMARY, dtype={"date": str})
    chk = sp.merge(P[KEY + ["speed_all", "n_all"]], on=KEY, how="outer", suffixes=("", "_prod"))
    d1 = np.nanmax(np.abs(chk.median_all - chk.speed_all))
    print(f"speed median-all vs production summary: {len(chk)} rows, max |diff| {d1:.3g}, n_all mismatches "
          f"{int((chk.n_all != chk.n_all_prod).sum())}")
    R = pd.read_csv(PROD_REACH_SESS, dtype={"date": str})
    rl = pd.concat([R[["session", f"reach_mm|{lr}"]].rename(columns={f"reach_mm|{lr}": "prodv"}).assign(position=lr)
                    for lr in LR_POS if f"reach_mm|{lr}" in R], ignore_index=True).dropna(subset=["prodv"])
    chk2 = rc.merge(rl, on=["session", "position"], how="inner")
    d2 = np.nanmax(np.abs(chk2.median_all - chk2.prodv))
    print(f"reach median-all vs production pose3d_sessions: {len(chk2)} matched of {len(rc)} rows / {len(rl)} prod, "
          f"max |diff| {d2:.3g}")
    _safe_write(S, out / "sixpos_first5_session_summary.csv")


# ------------------------------------------------------------------------------------------- cohort-2 grid (11-23 late)
#: the cohort-1 grid (Priya 11-23 late): measure x lick set x session summary. "trialmed" = median over each trial's
#: licks of the set, then the mean over trials (no per-trial median for the first lick: one lick per trial)
LICKSETS = ("all", "first5", "first")
SUMMARIES = ("mean", "median", "trialmed")
SUMMARY_NAME = {"mean": "session mean", "median": "session median", "trialmed": "per-trial median, session mean"}
LICKSET_NAME = {"all": "all post-cue licks", "first5": "post-cue licks 1-5", "first": "first post-cue lick"}
#: the family endpoint each (measure, lick set) occupies; licks 1-5 is a candidate REPLACEMENT of the all-lick cells
ENDPOINT = {("speed", "all"): "speed_all", ("reach", "all"): "reach_all", ("speed", "first5"): "speed_all",
            ("reach", "first5"): "reach_all", ("speed", "first"): "speed_first", ("reach", "first"): "reach_first"}
SESSMEAN_SUMMARY = P3D / "kinematics_fused_postcue/c2_lick3d_session_summary_ungated_sessmean_cam3on.csv"


def _grid_rows(L: pd.DataFrame, col: str, measure: str, lickset: str, gate: str) -> list[dict]:
    rows = []
    for key, d in L.groupby(KEY, sort=True):
        if lickset == "first5":
            d = d[d.lick_idx < FIRST]
        elif lickset == "first":
            d = d[d.lick_idx == 0]
        v = d[[col, "trial_id"]].dropna(subset=[col])
        tm = v.groupby("trial_id")[col].median()
        base = dict(zip(KEY, key), measure=measure, lickset=lickset, lick_gate=gate,
                    n=int(d.reach_mm.notna().sum()), n_trials=int(d.loc[d.reach_mm.notna(), "trial_id"].nunique()))
        rows.append(base | {"summary": "mean", "value": v[col].mean()})
        rows.append(base | {"summary": "median", "value": v[col].median()})
        if lickset != "first":
            rows.append(base | {"summary": "trialmed", "value": tm.mean()})
    return rows


def _nrule(measure: str, lickset: str):
    """Minimum licks per session x spout: production n >= 20 for every lick3d endpoint; all-lick reach has none."""
    return None if (measure, lickset) == ("reach", "all") else NMIN


def stage_grid(out: Path):
    """The cohort-2 grid: session table, one 180-cell family per (lick set, summary) with that lick set's two cells
    swapped in (BH once over 180), cells table, star counts and figures."""
    from wfield_local import orofacial_stats as st
    f = out / "sixpos_grid_session_summary.csv"
    if f.exists():
        G = pd.read_csv(f, dtype={"date": str})
    else:
        L = _load_licks()
        ids = json.load(open(out / "working_ids.json"))
        keep = np.zeros(len(L), bool)
        sv, tid = L.session.astype(str).to_numpy(), L.trial_id.to_numpy(float)
        for s in np.unique(sv):
            m = sv == s
            keep[m] = np.isin(tid[m], np.asarray(ids.get(s, []), float))
        Lg = L[keep]
        rows = []
        for ls in LICKSETS:
            rows += _grid_rows(L, "speed_peak", "speed", ls, "ungated")
            # reach: all-lick reach is production's working-period table; first-lick reach is the ungated lick3d one
            rows += _grid_rows(L if ls == "first" else Lg, "reach_mm", "reach", ls,
                               "ungated" if ls == "first" else "working_period")
        G = pd.DataFrame(rows)
        # reproduction checks vs production session tables
        P = pd.read_csv(PROD_SUMMARY, dtype={"date": str})
        Q = pd.read_csv(SESSMEAN_SUMMARY, dtype={"date": str})
        for (ms, ls, sm), (tab, colp) in {("speed", "all", "median"): (P, "speed_all"),
                                          ("speed", "first", "median"): (P, "speed_first"),
                                          ("reach", "first", "median"): (P, "reach_first"),
                                          ("speed", "all", "mean"): (Q, "speed_all"),
                                          ("speed", "first", "mean"): (Q, "speed_first"),
                                          ("reach", "first", "mean"): (Q, "reach_first")}.items():
            g = G[(G.measure == ms) & (G.lickset == ls) & (G.summary == sm)].merge(tab[KEY + [colp]], on=KEY)
            print(f"check {ms}/{ls}/{sm} vs {colp}: {len(g)} rows, max |diff| {np.nanmax(np.abs(g.value - g[colp])):.3g}")
        _safe_write(G, f)
    F0 = pd.read_csv(FINAL_FAMILY)
    keep = ["metric", "pos", "epoch", "n_sessions", "n_pre_sessions", "n_animals", "mean_change", "p", "boot_change",
            "boot_lo", "boot_hi", "q_fdr_metric", "q_family", "boot_sig", "agree", "star_aligned"]
    cells, fams = [], {}
    for ls in LICKSETS:
        for sm in SUMMARIES:
            if ls == "first" and sm == "trialmed":
                continue
            parts, swap = [], []
            for ms in ("speed", "reach"):
                ep = ENDPOINT[(ms, ls)]
                d = G[(G.measure == ms) & (G.lickset == ls) & (G.summary == sm)]
                d = _chg(d, "value", _nrule(ms, ls), "n")
                parts.append(_test(d, ep))
                swap.append(ep)
            R = pd.concat(parts, ignore_index=True)
            R = R[R.epoch.isin(EPOCHS) & R.pos.isin(POS)]
            other = F0[~F0.metric.isin(swap)].copy()
            other["q_fdr"] = other.q_fdr_metric
            T = pd.concat([other[[c for c in other.columns if c in R.columns or c == "q_fdr"]], R], ignore_index=True)
            F = st.apply_family_fdr(T).rename(columns={"q_fdr": "q_family"})
            F["star_aligned"] = (F.agree == "both").astype(int)
            fams[(ls, sm)] = F
            _safe_write(F[keep], out / f"c2_scalar_family_grid_{ls}_{sm}_final180_boot4000_cam3on.csv")
            oth = F[~F.metric.isin(swap)].merge(F0[["metric", "pos", "epoch", "star_aligned"]],
                                                on=["metric", "pos", "epoch"], suffixes=("", "_f0"))
            c = F[F.metric.isin(swap)][keep].copy()
            c["measure"] = c.metric.str.split("_").str[0]
            c["lickset"], c["summary"] = ls, sm
            cells.append(c)
            print(f"{ls}/{sm}: stars speed {int(c[c.measure == 'speed'].star_aligned.sum())}, reach "
                  f"{int(c[c.measure == 'reach'].star_aligned.sum())}, family {int(F.star_aligned.sum())}; other-cell "
                  f"flips vs final family {int((oth.star_aligned != oth.star_aligned_f0).sum())}")
    C = pd.concat(cells, ignore_index=True)
    # the production / final-family cells reproduced?
    for ls, sm, ep in (("all", "mean", "speed_all"), ("all", "mean", "reach_all"), ("first", "mean", "speed_first"),
                       ("first", "mean", "reach_first")):
        a = C[(C.lickset == ls) & (C.summary == sm) & (C.metric == ep)].merge(
            F0[F0.metric == ep], on=["pos", "epoch"], suffixes=("", "_f0"))
        print(f"reproduce final family {ep} ({ls}/{sm}): max |dCI| "
              f"{np.nanmax(np.abs(a.boot_lo - a.boot_lo_f0)):.2g}, star mismatches "
              f"{int((a.star_aligned != a.star_aligned_f0).sum())}")
    P0 = pd.read_csv(PROD_FAMILY)
    for ls, ep in (("all", "speed_all"), ("all", "reach_all"), ("first", "reach_first")):
        a = C[(C.lickset == ls) & (C.summary == "median") & (C.metric == ep)].merge(
            P0[P0.metric == ep], on=["pos", "epoch"], suffixes=("", "_p0"))
        print(f"median {ep} ({ls}) vs 162-cell production: max |dCI| {np.nanmax(np.abs(a.boot_lo - a.boot_lo_p0)):.2g},"
              f" star mismatches {int((a.star_aligned != a.star_aligned_p0).sum())}")
    C = C[["measure", "lickset", "summary"] + keep]
    _safe_write(C, out / "sixpos_grid_cells.csv")
    cnt = C.groupby(["measure", "lickset", "summary"]).star_aligned.sum().unstack("summary")
    cnt = cnt.reindex(pd.MultiIndex.from_product([["speed", "reach"], LICKSETS]))[list(SUMMARIES)]
    _safe_write(cnt.reset_index().rename(columns={"level_0": "measure", "level_1": "lickset"}),
                out / "sixpos_grid_star_counts.csv")
    print(cnt.to_string())
    grid_figures(out, C, G)


def _panel(ax, d: pd.DataFrame, r: pd.DataFrame, title: str | None, ylab: str | None, ep_col, mk):
    from wfield_local import sig_marks
    for pi, p in enumerate(POS):
        for ei, e in enumerate(EPOCHS):
            x0 = pi + (ei - 1) * 0.25
            dd = d[(d.pos == p) & (d.epoch == e)]
            rng = np.random.default_rng(pi * 10 + ei)
            for a_, g in dd.groupby("animal"):
                ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(g)), g.chg, s=8, marker=mk.get(a_, "o"),
                           facecolors="none", edgecolors=ep_col[e], linewidths=0.6, alpha=0.7)
            rr = r[(r.pos == p) & (r.epoch == e)]
            if len(rr):
                rr = rr.iloc[0]
                ax.errorbar(x0, rr.boot_change, yerr=[[rr.boot_change - rr.boot_lo], [rr.boot_hi - rr.boot_change]],
                            fmt="_", color="k", ms=9, lw=1.1, capsize=0, zorder=5)
                if rr.star_aligned:
                    sig_marks.star_top(ax, x0, "*", fontsize=11)
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.axvline(2.5, color="0.85", lw=0.7)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["close\nipsi", "close\nmid", "close\ncontra", "far\nipsi", "far\nmid", "far\ncontra"], fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.5)
    ax.spines["right"].set_visible(False)


def grid_figures(out: Path, C: pd.DataFrame, G: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from wfield_local import sig_marks
    ep_col = {"acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4"}
    mk = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    unit = {"speed": "peak 3-D speed change (mm/s)", "reach": "peak 3-D reach change (mm)"}
    h = [Line2D([], [], ls="", marker="o", mfc="none", mec=c, label=e) for e, c in ep_col.items()]
    h += [Line2D([], [], ls="", marker=m, mfc="none", mec="0.3", label=f"#{9 + k}") for k, m in enumerate(mk.values())]
    note = ("change from own pre (mice #9-#12, four-camera post-cue licks); bar = nested bootstrap mean and 95% CI, "
            "4,000 draws; * = 180-cell family q < 0.05 and CI excludes 0")

    def data(ms, ls, sm):
        d = G[(G.measure == ms) & (G.lickset == ls) & (G.summary == sm)]
        d = _chg(d, "value", _nrule(ms, ls), "n")
        return d[d.epoch.isin(EPOCHS)], C[(C.measure == ms) & (C.lickset == ls) & (C.summary == sm)]

    def finish(fig, png):
        sig_marks.reserve_headroom(fig)
        fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.035, 1, 0.95))
        fig.savefig(png, dpi=200)
        fig.savefig(png.with_suffix(".pdf"))
        plt.close(fig)
        print("wrote", png)

    # 1) the comparison: rows speed / reach, columns median all / mean all / mean 1-5 / median 1-5
    png = out / "sixpos_first5_comparison.png"
    if png.exists():
        sys.exit(f"refusing to overwrite {png}")
    cols = (("all", "median", "median, all licks (old production)"), ("all", "mean", "mean, all licks (production)"),
            ("first5", "mean", "mean, licks 1-5 (candidate)"), ("first5", "median", "median, licks 1-5 (candidate)"))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2), sharey="row")
    for i, ms in enumerate(("speed", "reach")):
        for j, (ls, sm, t) in enumerate(cols):
            d, r = data(ms, ls, sm)
            _panel(axes[i, j], d, r, t if i == 0 else None, unit[ms] if j == 0 else None, ep_col, mk)
    fig.suptitle("Six-position peak 3-D speed and reach, " + note, fontsize=8.5)
    finish(fig, png)
    # 2) the full grid, one figure per measure: rows lick set, columns summary
    for ms in ("speed", "reach"):
        png = out / f"sixpos_grid_{ms}.png"
        if png.exists():
            sys.exit(f"refusing to overwrite {png}")
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey="row")
        for i, ls in enumerate(LICKSETS):
            for j, sm in enumerate(SUMMARIES):
                ax = axes[i, j]
                if ls == "first" and sm == "trialmed":
                    ax.axis("off")
                    ax.text(0.5, 0.5, "first lick: one lick per trial,\nno per-trial median", ha="center",
                            va="center", fontsize=9, color="0.4", transform=ax.transAxes)
                    continue
                d, r = data(ms, ls, sm)
                _panel(ax, d, r, SUMMARY_NAME[sm] if i == 0 else None,
                       f"{LICKSET_NAME[ls]}\n{unit[ms]}" if j == 0 else None, ep_col, mk)
        fig.suptitle(f"Six-position peak 3-D {ms}: lick set x session summary; " + note, fontsize=8.5)
        finish(fig, png)


def stage_focus(out: Path):
    """Priya's main comparison (11-23 late): all licks vs licks 1-5, both as per-trial median -> session mean; rows
    speed / reach (same y scale per row), stars from the grid's 180-cell families."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from wfield_local import sig_marks
    png = out / "C2_alls_vs_15_pertrialmedian_ccB1123.png"
    if png.exists():
        sys.exit(f"refusing to overwrite {png}")
    G = pd.read_csv(out / "sixpos_grid_session_summary.csv", dtype={"date": str})
    C = pd.read_csv(out / "sixpos_grid_cells.csv")
    ep_col = {"acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4"}
    mk = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    unit = {"speed": "peak 3-D speed, onset to peak\nchange from pre (mm/s)",
            "reach": "peak 3-D reach\nchange from pre (mm)"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey="row")
    for i, ms in enumerate(("speed", "reach")):
        for j, (ls, t) in enumerate((("all", "all post-cue licks"), ("first5", "post-cue licks 1-5"))):
            d = _chg(G[(G.measure == ms) & (G.lickset == ls) & (G.summary == "trialmed")], "value", _nrule(ms, ls), "n")
            r = C[(C.measure == ms) & (C.lickset == ls) & (C.summary == "trialmed")]
            _panel(axes[i, j], d[d.epoch.isin(EPOCHS)], r,
                   f"{t}\n(per-trial median, then session mean)" if i == 0 else None,
                   unit[ms] if j == 0 else None, ep_col, mk)
    sig_marks.reserve_headroom(fig)
    h = [Line2D([], [], ls="", marker="o", mfc="none", mec=c, label=e) for e, c in ep_col.items()]
    h += [Line2D([], [], ls="", marker=m, mfc="none", mec="0.3", label=f"#{9 + k}") for k, m in enumerate(mk.values())]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Mice #9-#12, four-camera post-cue licks; change from own pre; bar = nested bootstrap mean and 95% CI "
                 "(4,000 draws); * = 180-cell family q < 0.05 and CI excludes 0", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(png, dpi=200)
    fig.savefig(png.with_suffix(".pdf"))
    print("wrote", png)


def _chg(S: pd.DataFrame, val: str, nmin: int | None, ncol: str) -> pd.DataFrame:
    d = S.copy()
    if nmin:
        d = d[d[ncol] >= nmin]
    d = d.assign(pos=d.position.map(LR_POS), v=d[val]).dropna(subset=["v"])
    b = d[d.epoch == "pre"].groupby(["animal", "pos"]).v.mean()
    d["chg"] = d.v - [b.get((x, p), np.nan) for x, p in zip(d.animal, d.pos)]
    return d.dropna(subset=["chg"])


def _test(d: pd.DataFrame, name: str) -> pd.DataFrame:
    from wfield_local import orofacial_stats as st
    x = d[["animal", "date", "epoch", "pos", "chg"]].rename(columns={"chg": name})
    return st.session_change_tests(x, [name], deltas_given=True, n_boot=4000)


def stage_stats(out: Path):
    from wfield_local import orofacial_stats as st
    S = pd.read_csv(out / "sixpos_first5_session_summary.csv", dtype={"date": str})
    F0 = pd.read_csv(PROD_FAMILY)
    sp, rc = S[S.endpoint == "speed_all"], S[S.endpoint == "reach_all"]
    # (value column, n rule) per variant x endpoint; reach production rule = no minimum count
    spec = {"prod_median_all": {"speed_all": ("median_all", NMIN, "n_all"), "reach_all": ("median_all", None, None)},
            "mean_all": {"speed_all": ("mean_all", NMIN, "n_all"), "reach_all": ("mean_all", None, None)},
            "mean_first5": {"speed_all": ("mean_first5", NMIN, "n_first5"),
                            "reach_all": ("mean_first5", NMIN, "n_first5")},
            "mean_first5_reach_nocut": {"speed_all": ("mean_first5", NMIN, "n_first5"),
                                        "reach_all": ("mean_first5", None, None)}}
    keep = ["metric", "pos", "epoch", "n_sessions", "n_pre_sessions", "n_animals", "mean_change", "p", "boot_change",
            "boot_lo", "boot_hi", "q_fdr_metric", "q_family", "boot_sig", "agree", "star_aligned"]
    fams, cells, drops = {}, [], []
    for var, ep in spec.items():
        parts = []
        for metric, (val, nmin, ncol) in ep.items():
            src = sp if metric == "speed_all" else rc
            d = _chg(src, val, nmin, ncol)
            nall = len(_chg(src, val, None, None))
            drops.append(dict(variant=var, endpoint=metric, rule=f"n >= {nmin} ({ncol})" if nmin else "none",
                              cells_kept=len(d), cells_without_cut=nall, cells_dropped=nall - len(d),
                              post_kept=int((d.epoch != "pre").sum()), pre_kept=int((d.epoch == "pre").sum())))
            parts.append(_test(d, metric))
        R = pd.concat(parts, ignore_index=True)
        R = R[R.epoch.isin(EPOCHS) & R.pos.isin(POS)]
        other = F0[~F0.metric.isin(["speed_all", "reach_all"])].copy()
        other["q_fdr"] = other.q_fdr_metric
        T = pd.concat([other[[c for c in other.columns if c in R.columns or c == "q_fdr"]], R], ignore_index=True)
        F = st.apply_family_fdr(T).rename(columns={"q_fdr": "q_family"})
        F["star_aligned"] = (F.agree == "both").astype(int)
        F["variant"] = var
        fams[var] = F
        _safe_write(F[["variant"] + keep], out / f"c2_scalar_family_{var}_boot4000_cam3on.csv")
        cells.append(F[F.metric.isin(["speed_all", "reach_all"])][["variant"] + keep])
    # reproduction check: the production variant must equal the production family table
    a = fams["prod_median_all"].merge(F0, on=["metric", "pos", "epoch"], suffixes=("", "_f0"))
    for c in ("p", "boot_lo", "boot_hi", "q_family"):
        print(f"prod reproduction {c}: max |diff| {np.nanmax(np.abs(a[c] - a[c + '_f0'])):.3g}")
    print("prod reproduction star mismatches:", int((a.star_aligned != a.star_aligned_f0).sum()),
          "; family stars", int(a.star_aligned.sum()), "vs", int(a.star_aligned_f0.sum()))
    C = pd.concat(cells, ignore_index=True)
    _safe_write(pd.DataFrame(drops), out / "sixpos_first5_min_lick_rule.csv")
    W = C.pivot_table(index=["metric", "pos", "epoch"], columns="variant",
                      values=["boot_change", "boot_lo", "boot_hi", "q_family", "star_aligned", "n_sessions"],
                      aggfunc="first")
    W.columns = [f"{v}__{m}" for m, v in W.columns]
    W = W.reset_index()
    W["pos"] = pd.Categorical(W.pos, POS, ordered=True)
    W["metric"] = pd.Categorical(W.metric, ["speed_all", "reach_all"], ordered=True)
    W = W.sort_values(["metric", "pos", "epoch"]).reset_index(drop=True)
    for v in ("mean_all", "mean_first5"):
        W[f"flip_{v}_vs_prod"] = W[f"{v}__star_aligned"] - W["prod_median_all__star_aligned"]
    W["flip_first5_vs_mean_all"] = W["mean_first5__star_aligned"] - W["mean_all__star_aligned"]
    _safe_write(W, out / "sixpos_first5_comparison.csv")
    st_ = {v: {m: int(f[f.metric == m].star_aligned.sum()) for m in ("speed_all", "reach_all")} | {
        "family_total": int(f.star_aligned.sum())} for v, f in fams.items()}
    json.dump(st_, open(out / "sixpos_first5_star_counts.json", "w"), indent=1)
    print(json.dumps(st_, indent=1))


def stage_final(out: Path):
    """The candidate inside the NEW production family (11-23 evening: session MEANS + first-lick speed, 180 cells,
    `FINAL_FAMILY`): speed_all / reach_all replaced by licks 1-5, BH over the 180 cells. The mean-all variant must
    reproduce that family exactly. Comparison table + figure: production (median, 162-cell family) / mean all (final
    family) / licks 1-5 (in the final family)."""
    from wfield_local import orofacial_stats as st
    S = pd.read_csv(out / "sixpos_first5_session_summary.csv", dtype={"date": str})
    F0 = pd.read_csv(FINAL_FAMILY)
    P0 = pd.read_csv(PROD_FAMILY)
    sp, rc = S[S.endpoint == "speed_all"], S[S.endpoint == "reach_all"]
    spec = {"mean_all": {"speed_all": ("mean_all", NMIN, "n_all"), "reach_all": ("mean_all", None, None)},
            "mean_first5": {"speed_all": ("mean_first5", NMIN, "n_first5"),
                            "reach_all": ("mean_first5", NMIN, "n_first5")},
            # 11-23 late (Priya): session MEDIAN over licks 1-5, same lick set / minimum / family
            "median_first5": {"speed_all": ("median_first5", NMIN, "n_first5"),
                              "reach_all": ("median_first5", NMIN, "n_first5")}}
    keep = ["metric", "pos", "epoch", "n_sessions", "n_pre_sessions", "n_animals", "mean_change", "p", "boot_change",
            "boot_lo", "boot_hi", "q_fdr_metric", "q_family", "boot_sig", "agree", "star_aligned"]
    fams = {}
    for var, ep in spec.items():
        R = pd.concat([_test(_chg(sp if m == "speed_all" else rc, v, n, c), m) for m, (v, n, c) in ep.items()],
                      ignore_index=True)
        R = R[R.epoch.isin(EPOCHS) & R.pos.isin(POS)]
        other = F0[~F0.metric.isin(["speed_all", "reach_all"])].copy()
        other["q_fdr"] = other.q_fdr_metric
        T = pd.concat([other[[c for c in other.columns if c in R.columns or c == "q_fdr"]], R], ignore_index=True)
        F = st.apply_family_fdr(T).rename(columns={"q_fdr": "q_family"})
        F["star_aligned"] = (F.agree == "both").astype(int)
        F["variant"] = var
        fams[var] = F
        _safe_write(F[["variant"] + keep], out / f"c2_scalar_family_{var}_in_final180_boot4000_cam3on.csv")
    a = fams["mean_all"].merge(F0, on=["metric", "pos", "epoch"], suffixes=("", "_f0"))
    for c in ("p", "boot_lo", "boot_hi", "q_family"):
        print(f"final-family reproduction {c}: max |diff| {np.nanmax(np.abs(a[c] - a[c + '_f0'])):.3g}")
    print("final-family reproduction star mismatches:", int((a.star_aligned != a.star_aligned_f0).sum()),
          "; stars", int(a.star_aligned.sum()), "vs", int(a.star_aligned_f0.sum()), "cells", len(a), len(F0))
    fa = fams["mean_all"]
    for v in ("mean_first5", "median_first5"):
        oth = fams[v][~fams[v].metric.isin(["speed_all", "reach_all"])].merge(
            fa[["metric", "pos", "epoch", "star_aligned"]], on=["metric", "pos", "epoch"], suffixes=("", "_meanall"))
        print(f"other endpoints: star flips {v} vs mean-all family:",
              int((oth.star_aligned != oth.star_aligned_meanall).sum()))
    C = pd.concat([P0[P0.metric.isin(["speed_all", "reach_all"])].assign(variant="prod_median_all")[["variant"] + keep]]
                  + [f[f.metric.isin(["speed_all", "reach_all"])][["variant"] + keep] for f in fams.values()],
                  ignore_index=True)
    W = C.pivot_table(index=["metric", "pos", "epoch"], columns="variant",
                      values=["boot_change", "boot_lo", "boot_hi", "q_family", "star_aligned", "n_sessions"],
                      aggfunc="first")
    W.columns = [f"{v}__{m}" for m, v in W.columns]
    W = W.reset_index()
    W["pos"] = pd.Categorical(W.pos, POS, ordered=True)
    W["metric"] = pd.Categorical(W.metric, ["speed_all", "reach_all"], ordered=True)
    W = W.sort_values(["metric", "pos", "epoch"]).reset_index(drop=True)
    for v in ("mean_all", "mean_first5", "median_first5"):
        W[f"flip_{v}_vs_prod"] = W[f"{v}__star_aligned"] - W["prod_median_all__star_aligned"]
    W["flip_first5_vs_mean_all"] = W["mean_first5__star_aligned"] - W["mean_all__star_aligned"]
    W["flip_median_first5_vs_mean_all"] = W["median_first5__star_aligned"] - W["mean_all__star_aligned"]
    W["flip_median_first5_vs_mean_first5"] = W["median_first5__star_aligned"] - W["mean_first5__star_aligned"]
    _safe_write(W, out / "sixpos_first5_comparison_final180.csv")
    st_ = {"prod_median_all (162-cell family)": {m: int(P0[P0.metric == m].star_aligned.sum())
                                                 for m in ("speed_all", "reach_all")} | {
        "family_total": int(P0.star_aligned.sum())}}
    st_ |= {f"{v} (180-cell final family)": {m: int(f[f.metric == m].star_aligned.sum())
                                             for m in ("speed_all", "reach_all")} | {
        "family_total": int(f.star_aligned.sum())} for v, f in fams.items()}
    json.dump(st_, open(out / "sixpos_first5_star_counts_final180.json", "w"), indent=1)
    print(json.dumps(st_, indent=1))
    figure(out, C, S)


def figure(out: Path, C: pd.DataFrame, S: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    png = out / "sixpos_first5_comparison.png"
    if png.exists():
        sys.exit(f"refusing to overwrite {png}")
    ep_col = {"acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4"}
    mk = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    vals = {"prod_median_all": ("median_all", NMIN, "n_all", None), "mean_all": ("mean_all", NMIN, "n_all", None),
            "mean_first5": ("mean_first5", NMIN, "n_first5", NMIN),
            "median_first5": ("median_first5", NMIN, "n_first5", NMIN)}
    vals = {k: v for k, v in vals.items() if k in set(C.variant)}
    fig, axes = plt.subplots(2, len(vals), figsize=(4.4 * len(vals), 7.2), sharey="row")
    for i, (metric, unit) in enumerate((("speed_all", "peak 3-D speed change (mm/s)"),
                                        ("reach_all", "peak 3-D reach change (mm)"))):
        src = S[S.endpoint == metric]
        for j, var in enumerate(vals):
            ax = axes[i, j]
            val, nmin, ncol, rnmin = vals[var]
            if metric == "reach_all":
                nmin, ncol = rnmin, ("n_first5" if rnmin else None)
            d = _chg(src, val, nmin, ncol)
            d = d[d.epoch.isin(EPOCHS)]
            r = C[(C.variant == var) & (C.metric == metric)]
            for pi, p in enumerate(POS):
                for ei, e in enumerate(EPOCHS):
                    x0 = pi + (ei - 1) * 0.25
                    dd = d[(d.pos == p) & (d.epoch == e)]
                    rng = np.random.default_rng(pi * 10 + ei)
                    for a_, g in dd.groupby("animal"):
                        ax.scatter(x0 + rng.uniform(-0.07, 0.07, len(g)), g.chg, s=9, marker=mk.get(a_, "o"),
                                   facecolors="none", edgecolors=ep_col[e], linewidths=0.6, alpha=0.7)
                    rr = r[(r.pos == p) & (r.epoch == e)]
                    if len(rr):
                        rr = rr.iloc[0]
                        ax.errorbar(x0, rr.boot_change, yerr=[[rr.boot_change - rr.boot_lo], [rr.boot_hi - rr.boot_change]],
                                    fmt="_", color="k", ms=10, lw=1.2, capsize=0, zorder=5)
            ax.axhline(0, color="0.5", lw=0.7, ls="--")
            ax.set_xticks(range(6))
            ax.set_xticklabels(["close\nipsi", "close\nmid", "close\ncontra", "far\nipsi", "far\nmid", "far\ncontra"],
                               fontsize=8)
            ax.axvline(2.5, color="0.85", lw=0.7)
            if i == 0:
                ax.set_title(VARIANTS[var], fontsize=10)
            if j == 0:
                ax.set_ylabel(unit, fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    for i, metric in enumerate(("speed_all", "reach_all")):
        for j, var in enumerate(vals):
            ax = axes[i, j]
            r = C[(C.variant == var) & (C.metric == metric)]
            for pi, p in enumerate(POS):
                for ei, e in enumerate(EPOCHS):
                    rr = r[(r.pos == p) & (r.epoch == e)]
                    if len(rr) and rr.iloc[0].star_aligned:
                        sig_marks.star_top(ax, pi + (ei - 1) * 0.25, "*", fontsize=11)
    sig_marks.reserve_headroom(fig)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc="none", mec=c, label=e) for e, c in ep_col.items()]
    h += [Line2D([], [], ls="", marker=m, mfc="none", mec="0.3", label=f"#{9 + k}") for k, m in enumerate(mk.values())]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle("Six-position all-lick 3-D kinematics, change from own pre (mice #9-#12, four-camera licks, "
                 "post-cue; bar = nested bootstrap mean and 95% CI, 4,000 draws; * = family q < 0.05 and CI excludes 0)",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(png, dpi=200)
    fig.savefig(png.with_suffix(".pdf"))
    print("wrote", png)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["ids", "summ", "stats", "final", "grid", "focus", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("ids", "all"):
        stage_ids(a.out, a.jobs)
    if a.stage in ("summ", "all"):
        stage_summ(a.out)
    if a.stage in ("stats", "all"):
        stage_stats(a.out)
    if a.stage in ("final", "all"):
        stage_final(a.out)
    if a.stage == "grid":
        stage_grid(a.out)
    if a.stage == "focus":
        stage_focus(a.out)


if __name__ == "__main__":
    main()
