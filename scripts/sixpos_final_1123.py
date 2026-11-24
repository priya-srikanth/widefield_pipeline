"""FINAL six-position (cohort 2, mice #9-#12) scalar family and the Fig 4B / 4C / S11A / S11B tables (CC-B, 2026-11-23).

Combines every change Priya decided on 11-23 into ONE 180-cell family, BH once:
  * Fig 4B  speed_all    ALL post-cue licks, peak 3-D tip speed (onset -> extension peak): median over each trial's
                         licks, then the MEAN over trials (session x spout); ungated lick3d set; >= 20 licks with 3-D
                         support (production rule).
  * Fig 4C  reach_all    ALL post-cue licks, peak 3-D reach: per-trial median, then session mean; camera working-period
                         set; no minimum (production rule).
  * S11A    reach_first  first post-cue lick, peak 3-D reach: session MEDIAN over trials; ungated; >= 20 first licks.
  * S11B    speed_first  first post-cue lick, peak 3-D speed: session MEDIAN over trials; ungated; >= 20 first licks.
  * S10C    ili_ms       within-bout ILI with the cross-cohort bound (intervals <= 300 ms; item 29,
                         `scripts/c2_ili_le300.py`, rows taken from its c2_ili_le300_stats.csv, 4,000 draws).
  * every other endpoint (hit rate, latency, licks per responded trial, first-lick and all-lick angle = session means)
    as in the production 180-cell family `scalar_family_final_boot4000_sessmean_speedfirst_cam3on` (identical cells to
    the live 162-cell family for those endpoints).
Trial classes, jaw calls and gates are the production ones (3-D jaw class rebuild on hold).

Session values come from the 11-23 grid table (`scripts/sixpos_first5_1123.py grid`, sixpos_grid_session_summary.csv;
reused, not recomputed). Each swapped endpoint is tested alone with `orofacial_stats.session_change_tests`
(`deltas_given`, 4,000-draw nested animal -> session `analysis_kit.boot_delta`, seed 0), exactly as the grid; the tests
are checked against the grid cells. Star = family q < 0.05 AND bootstrap CI excludes 0.

Stages (outputs are NEW files; the script refuses to overwrite):
  family   -> <stats>/c2_scalar_family_FINAL_ccB1123_boot4000_cam3on.csv (+ live-vs-final comparison, star counts)
  tables   -> <data>/kinematics/c2_lick3d_session_summary_FINAL_ccB1123.csv (wide, per session x spout),
              <data>/kinematics/c2_kin_session_changes_FINAL_ccB1123.csv (long, the cells each test used),
              <data>/scalar_family/c2_scalar_family_FINAL_ccB1123_boot4000_cam3on.csv (copy),
              <data>/scalar_family/C2_{speed_all,reach_all,reach_first,speed_first}_stats_FINAL_ccB1123.csv,
              <data>/scalar_family/c2_scalar_family_live_vs_FINAL_ccB1123.csv, ..._star_counts_...csv

    python scripts/sixpos_final_1123.py {family,tables,all} [--stats DIR] [--data DIR] [--grid DIR] [--ili CSV]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sixpos_first5_1123 as F5  # noqa: E402  (grid helpers: _chg, _test, _nrule, LR_POS, POS, EPOCHS)

STATS = Path("M:/MICROSCOPE/Priya/Widefield/manuscript_stats/c2_4cam_sync60_20261122")
PAPER = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper")
REVIEW = PAPER / "ccB_caption_review_2026-11-23"
DATA = PAPER / "data_4cam_20261122"
GRID = REVIEW / "sixpos_first5"
ILI = REVIEW / "crosscohort/tables/c2_ili_le300_stats.csv"
BASE = STATS / "scalar_family_final_boot4000_sessmean_speedfirst_cam3on/c2_scalar_family_aligned.csv"
LIVE = STATS / "scalar_family_final_boot4000_cam3on/c2_scalar_family_aligned.csv"          # live v28, 4,000 draws
LIVE2000 = STATS / "scalar_family_final/c2_scalar_family_aligned.csv"                       # rendered live panels
TAG = "FINAL_ccB1123"
FAMILY_NAME = f"c2_scalar_family_{TAG}_boot4000_cam3on.csv"
MOUSE = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}

#: (metric) -> (grid measure, grid lick set, grid summary, panel in the final figures, definition)
SWAP = {
    "speed_all": ("speed", "all", "trialmed", "Fig 4B C2_speed_all",
                  "peak 3-D tip speed (onset -> extension peak), all post-cue licks; per-trial median, session mean; "
                  ">= 20 licks"),
    "reach_all": ("reach", "all", "trialmed", "Fig 4C C2_reach_all",
                  "peak 3-D reach, all post-cue licks (camera working period); per-trial median, session mean; "
                  "no minimum"),
    "reach_first": ("reach", "first", "median", "S11A C2_reach_first",
                    "peak 3-D reach, first post-cue lick; session median over trials; >= 20 first licks"),
    "speed_first": ("speed", "first", "median", "S11B C2_speed_first",
                    "peak 3-D tip speed (onset -> extension peak), first post-cue lick; session median over trials; "
                    ">= 20 first licks"),
}
PANEL = {"hit_rate": "Fig 3B C2_hit_rate", "latency_ms": "Fig 4A C2_latency_first", "aim": "Fig 4D C2_aim",
         "aim_all": "Fig 4E C2_aim_all", "ili_ms": "S10C C2_ili", "licks_per_resp": "S10D C2_licks_per_resp"}
PANEL |= {m: v[3] for m, v in SWAP.items()}
DEFN = {"hit_rate": "fraction of trials with a spout contact (production)",
        "latency_ms": "latency to the onset of the first post-cue fused lick; session median over trials (production)",
        "aim": "first-lick angle; session mean (production)", "aim_all": "all-lick angle; session mean (production)",
        "ili_ms": "within-bout ILI, session median of within-trial intervals <= 300 ms (item 29)",
        "licks_per_resp": "licks per responded trial; session mean (production)"}
DEFN |= {m: v[4] for m, v in SWAP.items()}
#: live (v28) panel of each endpoint; 4C and S11A swap endpoints (item 25), S11B is new
LIVE_PANEL = {"hit_rate": "Fig 3B", "latency_ms": "Fig 4A", "speed_all": "Fig 4B", "reach_first": "Fig 4C",
              "aim": "Fig 4D", "aim_all": "Fig 4E", "ili_ms": "S10C", "licks_per_resp": "S10D", "reach_all": "S11A",
              "speed_first": "(none)"}
KEEP = ["metric", "panel", "pos", "epoch", "n_sessions", "n_pre_sessions", "n_animals", "mean_change", "p",
        "boot_change", "boot_lo", "boot_hi", "q_fdr_metric", "q_family", "boot_sig", "agree", "star_aligned",
        "definition"]
CELL = ["metric", "pos", "epoch"]


def _safe_write(df: pd.DataFrame, path: Path):
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    print("wrote", path)


def _order(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    o = {tuple(k): i for i, k in enumerate(ref[CELL].to_numpy())}
    return df.assign(_o=[o.get(tuple(k), 1e9) for k in df[CELL].to_numpy()]).sort_values("_o").drop(columns="_o")


def _grid_cells(G: pd.DataFrame, metric: str) -> pd.DataFrame:
    ms, ls, sm = SWAP[metric][:3]
    d = G[(G.measure == ms) & (G.lickset == ls) & (G.summary == sm)]
    return F5._chg(d, "value", F5._nrule(ms, ls), "n")


def stage_family(stats: Path, grid: Path, ili: Path):
    from wfield_local import orofacial_stats as st
    out = stats / FAMILY_NAME
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    B = pd.read_csv(BASE)
    G = pd.read_csv(grid / "sixpos_grid_session_summary.csv", dtype={"date": str})
    GC = pd.read_csv(grid / "sixpos_grid_cells.csv")
    # 0) BH over the base family reproduces its stored q (so one BH over the new rows is the same procedure)
    b = B.copy()
    b["q_fdr"] = b.q_fdr_metric
    bq = st.apply_family_fdr(b)
    print(f"base family {len(B)} cells: BH reproduces q_family max |dq| {np.nanmax(np.abs(bq.q_fdr - B.q_family)):.2g}, "
          f"star mismatches {int(((bq.agree == 'both').astype(int) != B.star_aligned).sum())}")
    # 1) the four swapped lick endpoints, each tested alone (as the grid)
    parts = []
    for metric, (ms, ls, sm, *_r) in SWAP.items():
        R = F5._test(_grid_cells(G, metric), metric)
        R = R[R.epoch.isin(F5.EPOCHS) & R.pos.isin(F5.POS)]
        g = GC[(GC.measure == ms) & (GC.lickset == ls) & (GC.summary == sm) & (GC.metric == metric)]
        c = R.merge(g, on=["pos", "epoch"], suffixes=("", "_g"))
        print(f"{metric} ({ls}/{sm}): {len(R)} cells; vs grid p max|d| {np.nanmax(np.abs(c.p - c.p_g)):.2g}, "
              f"CI max|d| {np.nanmax(np.abs(np.r_[c.boot_lo - c.boot_lo_g, c.boot_hi - c.boot_hi_g])):.2g}, "
              f"matched {len(c)}")
        parts.append(R)
    # 2) ILI <= 300 ms rows (item 29; 4,000 draws, the stored fused-test replay)
    I = pd.read_csv(ili)
    I = I[(I.metric == "ili_ms") & I.epoch.isin(F5.EPOCHS) & I.pos.isin(F5.POS)]
    print(f"ili_ms (<= 300 ms): {len(I)} cells from {ili.name}")
    parts.append(I)
    swap = [*SWAP, "ili_ms"]
    other = B[~B.metric.isin(swap)].copy()
    other["q_fdr"] = other.q_fdr_metric
    new = pd.concat(parts, ignore_index=True)
    cols = sorted(set(other.columns) & set(new.columns) | {"q_fdr"})
    T = pd.concat([other[[c for c in other.columns if c in cols]], new[[c for c in new.columns if c in cols]]],
                  ignore_index=True)
    F = st.apply_family_fdr(T).rename(columns={"q_fdr": "q_family"})
    F["star_aligned"] = (F.agree == "both").astype(int)
    F["panel"] = F.metric.map(PANEL)
    F["definition"] = F.metric.map(DEFN)
    F = _order(F, B)[KEEP].reset_index(drop=True)
    assert len(F) == 180 and not F.duplicated(CELL).any(), len(F)
    chk = (F.star_aligned == ((F.q_family < 0.05) & ((F.boot_lo > 0) | (F.boot_hi < 0))).astype(int)).all()
    print(f"FINAL family: {len(F)} cells, {int(F.star_aligned.sum())} stars; star == (q < .05 & CI excl 0): {chk}; "
          f"n_animals min {int(F.n_animals.min())}, cells with < 2 animals {int((F.n_animals < 2).sum())}, "
          f"< 4 animals {int((F.n_animals < 4).sum())}")
    _safe_write(F, out)
    compare(stats, F)


def compare(stats: Path, F: pd.DataFrame):
    L = pd.read_csv(LIVE)
    L2 = pd.read_csv(LIVE2000)
    B = pd.read_csv(BASE)
    k = ["boot_change", "boot_lo", "boot_hi", "q_family", "star_aligned"]
    C = F[CELL + ["panel", "n_sessions", "n_animals"] + k].rename(columns={c: f"final_{c}" for c in k})
    C = C.merge(L[CELL + k].rename(columns={c: f"live_{c}" for c in k}), on=CELL, how="left")
    C = C.merge(L2[CELL + ["star_aligned"]].rename(columns={"star_aligned": "live2000_star_aligned"}), on=CELL,
                how="left")
    C = C.merge(B[CELL + ["boot_change", "star_aligned"]].rename(
        columns={"boot_change": "sessmean_boot_change", "star_aligned": "sessmean_star_aligned"}), on=CELL, how="left")
    C = C.rename(columns={"panel": "final_panel"})
    C.insert(3, "live_panel", C.metric.map(LIVE_PANEL))
    ls, fs = C.live_star_aligned, C.final_star_aligned
    C["change"] = np.where(ls.isna(), "NEW", np.where((ls == 1) & (fs == 0), "LOST",
                                                      np.where((ls == 0) & (fs == 1), "GAINED", "")))
    _safe_write(C, stats / f"c2_scalar_family_live_vs_{TAG}.csv")
    rows = []
    for m in F.metric.unique():
        c = C[C.metric == m]
        rows.append(dict(metric=m, live_panel=LIVE_PANEL[m], final_panel=PANEL[m],
                         live_stars=int(c.live_star_aligned.sum()) if c.live_star_aligned.notna().any() else None,
                         live2000_stars=(int(c.live2000_star_aligned.sum())
                                         if c.live2000_star_aligned.notna().any() else None),
                         sessmean_stars=int(c.sessmean_star_aligned.sum()), final_stars=int(c.final_star_aligned.sum()),
                         gained="; ".join(f"{p} {e}" for p, e in c[c.change == "GAINED"][["pos", "epoch"]].to_numpy()),
                         lost="; ".join(f"{p} {e}" for p, e in c[c.change == "LOST"][["pos", "epoch"]].to_numpy())))
    S = pd.DataFrame(rows)
    _safe_write(S, stats / f"c2_scalar_family_star_counts_live_vs_{TAG}.csv")
    pd.set_option("display.width", 250)
    print(S.to_string(index=False))
    print("totals: live", int(L.star_aligned.sum()), "(162 cells); session-mean", int(B.star_aligned.sum()),
          "(180); FINAL", int(F.star_aligned.sum()), "(180)")


def stage_tables(stats: Path, data: Path, grid: Path):
    F = pd.read_csv(stats / FAMILY_NAME)
    G = pd.read_csv(grid / "sixpos_grid_session_summary.csv", dtype={"date": str})
    key = F5.KEY
    wide = None
    for metric, (ms, ls, sm, *_r) in SWAP.items():
        d = G[(G.measure == ms) & (G.lickset == ls) & (G.summary == sm)][key + ["value", "n", "n_trials", "lick_gate"]]
        d = d.rename(columns={"value": metric, "n": f"n_{metric}", "n_trials": f"n_trials_{metric}",
                              "lick_gate": f"gate_{metric}"})
        wide = d if wide is None else wide.merge(d, on=key, how="outer")
    assert (wide.n_reach_first.fillna(-1) == wide.n_speed_first.fillna(-1)).all()
    wide.insert(1, "mouse", wide.animal.map(MOUSE))
    wide.insert(len(key) + 1, "pos", wide.position.map(F5.LR_POS))
    print(f"wide table {len(wide)} session x spout rows; reach_all cells with n < 20: "
          f"{int((wide.n_reach_all < 20).sum())} (renderer n >= 20 step == production no-minimum rule if 0)")
    _safe_write(wide, data / "kinematics" / f"c2_lick3d_session_summary_{TAG}.csv")
    long = []
    for metric in SWAP:
        d = _grid_cells(G, metric)
        long.append(d.assign(metric=metric, mouse=d.animal.map(MOUSE))[
            ["metric", "animal", "mouse", "date", "session", "epoch", "position", "pos", "value", "n", "n_trials",
             "chg"]])
    Lg = pd.concat(long, ignore_index=True)
    for metric in SWAP:
        n = Lg[(Lg.metric == metric) & (Lg.epoch != "pre")].groupby(["pos", "epoch"]).size()
        f = F[F.metric == metric].set_index(["pos", "epoch"]).n_sessions
        bad = [(k, int(n.get(k, 0)), int(v)) for k, v in f.items() if int(n.get(k, 0)) != int(v)]
        print(f"{metric}: session n per cell vs family", "OK" if not bad else f"MISMATCH {bad}")
    _safe_write(Lg, data / "kinematics" / f"c2_kin_session_changes_{TAG}.csv")
    sf = data / "scalar_family"
    dst = sf / FAMILY_NAME
    if dst.exists():
        sys.exit(f"refusing to overwrite {dst}")
    shutil.copy2(stats / FAMILY_NAME, dst)
    print("wrote", dst)
    for metric in SWAP:
        stem = PANEL[metric].split()[-1]
        _safe_write(F[F.metric == metric], sf / f"{stem}_stats_{TAG}.csv")
    for nm in (f"c2_scalar_family_live_vs_{TAG}.csv", f"c2_scalar_family_star_counts_live_vs_{TAG}.csv"):
        if (sf / nm).exists():
            sys.exit(f"refusing to overwrite {sf / nm}")
        shutil.copy2(stats / nm, sf / nm)
        print("wrote", sf / nm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["family", "tables", "all"])
    ap.add_argument("--stats", type=Path, default=STATS)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--grid", type=Path, default=GRID)
    ap.add_argument("--ili", type=Path, default=ILI)
    a = ap.parse_args(argv)
    if a.stage in ("family", "all"):
        stage_family(a.stats, a.grid, a.ili)
    if a.stage in ("tables", "all"):
        stage_tables(a.stats, a.data, a.grid)


if __name__ == "__main__":
    main()
