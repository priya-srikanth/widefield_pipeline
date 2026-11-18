"""ITI recovery axes matched to the previous spout / ITI phase, docked ITI markers, and far-contra within-spout raw balance
vs lick angle (Priya 11-17; docs/status/deck_swap_20261117/ITI_MATCHED_ANGLE_MARKERS.md).

WHY: the primary recovery marker's ITI axis `iti_ang` (lick_recovery_markers.sessions) is the session mean signed angle of
ALL ITI licks minus the animal's pre mean -- not matched to the previous trial's spout and not restricted to an ITI
phase, so a session that happens to follow more contralesional spouts, or licks more while the spout moves, moves the
axis. The learned-compensation prereg (docs/status/LEARNED_COMPENSATION_PREREG_20261030.md) already defines matched ITI
angles; they are reused here (rule 9: learned_compensation._iti_base / _refs / t3_cells / PHASE_MAP), applied to the SAME
per-lick angle as iti_ang (3-D landing angle asin(side / reach) of the KEY ITI licks, lick_order_preds x
covref_covariates), signed like iti_ang (sign of the pre far-contra cued angle; + = contralesional).

ITI AXES (per session; pre sessions scored against their own animal's OTHER pre sessions where the reference allows):
  iti_ang               reproduced from the per-lick table (check vs recmark_sessions.csv)
  iti_ang_matched       mean over licks of (angle - the animal's pre mean angle of ITI licks after the SAME previous spout
                        in the SAME ITI phase (retract / docked / outbound)) = T2's per-lick d (_refs, keys prev_pos +
                        phase, own session left out for pre). Licks after an engaged previous trial at one of the six
                        spouts (_iti_base) in the three phases; sessions with >= MIN_CELL licks.
  iti_ang_dock          docked ITI licks only (iti_dock_audit phase 'docked'), unmatched: session mean minus the animal's
                        pre mean of the docked session means (as iti_ang).
  iti_ang_dock_matched  T3 D_off on this angle: docked licks after an engaged previous trial, per lick angle - pre at the
                        same previous spout; per session the mean over previous-spout cells with >= MIN_CELL licks, when
                        >= D_OFF_MIN_SPOUTS cells (learned_compensation.t3_cells). Lick-weighted mean kept as _lw.
PREDICTORS: the primary ITI marker (gain ratio 0.1..0.5 s, oro + SSs, first ITI lick after >= 2 s quiet;
lick_recovery_primary.marker on balsub_sessions_full.csv) on ALL ITI licks, and the same on DOCKED ITI licks (the
lick_balance_subsets WFIELD_ITI_FILTER=dock hook, `docked-balance`). Models = lick_recovery_primary.fit_panels (per-animal
r, concurrent MixedLM with log(day), per-animal spline, next-session change, LOAO cvR2 vs days with within-animal shuffle,
blocked CV, LOSO).

FAR-CONTRA WITHIN SPOUT (`balance-angle`): lick_balance_angle_lmm's session x spout table (dec = X_right - X_left
0..0.2 s; resp = log2 contra / ipsi response AUC, cued 0..1.0 s, ITI 0..0.5 s per DECISIONS 11-17) with the ITI subset
hook applied BEFORE the per-lick join; the full balance-angle LMMs (lick_balance_angle_lmm.compute) on that table; then
far-contra cells only (cued: target far contra; ITI: previous spout far contra): per-animal r, y ~ angle * post.
With --idx (Priya 11-17) both add the bounded indices dec_idx / resp_idx (lick_balance_angle_lmm.bounded_index; see
that docstring) as two more measures; without it the outputs are unchanged. --gap2: ITI bout starts with gap >=
il.GAP_MAIN (2 s) only (lick_balance_angle_lmm.gap_keep).

Usage (env of the matching rebuild step + WFIELD_ITI_OUT_W=<new folder>; PYTHONPATH=<worktree>;<worktree>/scripts):
  python scripts/lick_iti_matched_markers.py docked-balance      (+ WFIELD_ITI_FILTER=dock WFIELD_BAL_FULL=1)
  python scripts/lick_iti_matched_markers.py docked-primary      (same env; lick_recovery_primary on the docked table)
  python scripts/lick_iti_matched_markers.py axes
  python scripts/lick_iti_matched_markers.py models --bal-all <balsub_sessions_full.csv> --bal-dock <...csv>
  python scripts/lick_iti_matched_markers.py balance-angle --ratio <lateral_ratio_sessions.csv> [--idx] [--gap2]  [+ WFIELD_ITI_FILTER=dock]
  python scripts/lick_iti_matched_markers.py plot                 (figures from the saved CSVs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

# A WFIELD_ITI_OUT_W re-render under an ITI SUBSET reads upstream tables from OUT/itifilter_<f>/ (iti_licks.in_path),
# which the rebuild tree does not have. These tables do not depend on the ITI subset (per-lick covariates, cued first
# licks, the session day / aim / all-ITI angle table -- the pre-rebuild docked run also used the all-ITI iti_ang), so
# they fall back to the production copy in OUT. Any other missing table still fails. Module level, so spawned workers
# (which import this file as __mp_main__) get it too.
_SUBSET_SAFE = ("covref_covariates", "firstlick_full", "recmark_sessions.csv", "aim_deficit_sessions.csv")
_in_path0 = il.in_path


def _in_path(rel):
    p = _in_path0(rel)
    if p.exists() or not il.ITI_SUBSET or not str(rel).startswith(_SUBSET_SAFE):
        return p
    q = il.OUT / rel
    return q if q.exists() else p


il.in_path = _in_path

AN = ["PS92", "PS93", "PS94", "PS95"]
AXES = ["iti_ang", "iti_ang_matched", "iti_ang_dock", "iti_ang_dock_matched"]
YS = [("iti_ang", "ITI angle vs pre (all ITI, unmatched)"), ("aim", "far-contra first-lick aim vs pre"),
      ("iti_ang_matched", "ITI angle vs pre, same previous spout + phase"),
      ("iti_ang_dock", "docked ITI angle vs pre (unmatched)"),
      ("iti_ang_dock_matched", "docked ITI angle vs pre, same previous spout (T3 D_off)")]
XLAB = {"iti_ang": "ITI angle\n(all, unmatched)", "aim": "far-contra\ncued aim", "iti_ang_matched":
        "ITI angle,\nmatched spout\n+ phase", "iti_ang_dock": "docked ITI\nangle", "iti_ang_dock_matched":
        "docked ITI angle,\nmatched spout\n(T3 D_off)"}
PREDS = [("ITI", "ITI balance (all ITI)"), ("ITI docked", "ITI balance (docked ITI)")]
FC = "far_contra"
# learned_compensation writes iti_licks.csv.gz with float_format %.6g, so daq_on_s keeps 6 significant digits: 0.005 s
# resolution for onsets of 1000-9999 s (2 ms, its own audit tolerance, matched only ~50 %). Licks are >= 40 ms apart
# (lick_detection.min_ili_ms), so 6 ms stays unique.
JOIN_TOL_S = 0.006


def _arg(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


# ------------------------------------------------------------------------------------------------ ITI axes
def iti_lick_table() -> tuple[pd.DataFrame, float]:
    """KEY ITI licks (lick_order_preds rows of set 'iti', k = cache index) with the 3-D landing angle, signed angle sa
    (+ = contralesional), dock-audit phase, previous spout and whether the previous trial was engaged (joined from
    learned_compensation's ITI table by DAQ onset, |dt| < JOIN_TOL_S)."""
    import iti_lick_order as lo
    import learned_compensation as lc
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.in_path(f"covref_covariates{lo.SRC_TAG}.csv.gz"), usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    sgn = float(np.sign(P[(P.set == "trial") & (P.epoch == "pre") & (P.pos == FC)].angle.mean()))   # as lrm.sessions
    I = P[P.set == "iti"].assign(sa=lambda d: sgn * d.angle).copy()
    A = pd.read_csv(il.OUT / "iti_dock_audit.csv", usecols=["label", "k", "onset", "phase"])
    I = I.merge(A, on=["label", "k"], how="left", validate="one_to_one")
    I["phase_raw"] = I.phase.fillna("unmatched")
    I["phase"] = I.phase_raw.map(lc.PHASE_MAP).fillna("other")
    print(f"learned_compensation ITI table: {lc.OUT / 'iti_licks.csv.gz'}", flush=True)
    L = pd.read_csv(lc.OUT / "iti_licks.csv.gz", usecols=["label", "daq_on_s", "prev_pos", "prev_canonical"])
    L["prev_canonical"] = lc._b(L.prev_canonical)
    I["prev_canonical"], I["prev_pos_lc"] = False, None
    for lab, g in I.groupby("label"):
        h = L[L.label == lab].sort_values("daq_on_s")
        on = h.daq_on_s.to_numpy(float)
        x = g.onset.to_numpy(float)
        if not len(on):
            continue
        j = np.clip(np.searchsorted(on, x), 1, max(len(on) - 1, 1)) if len(on) > 1 else np.zeros(len(x), int)
        jl = np.maximum(j - 1, 0)
        near = np.where(np.abs(on[j] - x) <= np.abs(on[jl] - x), j, jl)
        hit = np.abs(on[near] - x) < JOIN_TOL_S
        I.loc[g.index, "prev_canonical"] = np.where(hit, h.prev_canonical.to_numpy()[near], False)
        I.loc[g.index, "prev_pos_lc"] = np.where(hit, h.prev_pos.to_numpy()[near], None)
    I["prev_canonical"] = I.prev_canonical.astype(bool)
    I["prev_pos"] = I.pos                    # the KEY cache's ITI 'pos' = the previous trial's spout
    m = I.prev_pos_lc.notna()
    print(f"== ITI licks {len(I)}: angle finite {I.sa.notna().mean():.3f}; dock-audit phase {I.phase.value_counts().to_dict()};"
          f" joined to the learned-compensation ITI table {m.mean():.3f}; previous spout agrees "
          f"{(I.prev_pos[m] == I.prev_pos_lc[m]).mean():.4f}; previous trial engaged {I.prev_canonical.mean():.3f}",
          flush=True)
    return I, sgn


def _vs_pre(S: pd.Series, animal: pd.Series, epoch: pd.Series) -> pd.Series:
    """Session values minus the animal's mean over its pre sessions (as lrm.sessions' iti_ang)."""
    pm = S[epoch == "pre"].groupby(animal[epoch == "pre"]).mean()
    return S - animal.map(pm)


def axes() -> pd.DataFrame:
    import learned_compensation as lc
    I, sgn = iti_lick_table()
    R = pd.read_csv(il.in_path("recmark_sessions.csv"), usecols=["label", "day", "iti_ang"]).drop_duplicates("label")
    I = I.merge(R[["label", "day"]], on="label", how="left")
    I = I.dropna(subset=["sa"])
    meta = I.groupby("label").agg(animal=("animal", "first"), epoch=("epoch", "first"), day=("day", "first"))
    out = meta.copy()
    s = I.groupby("label").sa.mean()
    out["iti_ang"] = _vs_pre(s, meta.animal, meta.epoch)
    out["n_iti"] = I.groupby("label").size()
    # T2-matched (prev spout + phase), _iti_base = previous trial engaged at one of the six spouts
    J = lc._iti_base(I.assign(onset_ok=True))
    J2 = J[J.phase.isin(lc.PHASES)].copy()
    J2["d"] = J2.sa - lc._refs(J2, "sa", ["prev_pos", "phase"])
    g = J2.dropna(subset=["d"]).groupby("label").d.agg(["mean", "size"])
    out["iti_ang_matched"] = g["mean"].where(g["size"] >= lc.MIN_CELL)
    out["n_matched"] = g["size"]
    # docked, unmatched (all docked KEY ITI licks, as iti_ang on the docked subset)
    D = I[I.phase == "docked"]
    gd = D.groupby("label").sa.agg(["mean", "size"])
    v = gd["mean"].where(gd["size"] >= lc.MIN_CELL).reindex(meta.index)
    out["iti_ang_dock"] = _vs_pre(v, meta.animal, meta.epoch)
    out["n_dock"] = gd["size"]
    # docked, matched to the previous spout = T3 D_off (t3_cells)
    K, cell = lc.t3_cells(J[J.phase == "docked"], "sa")
    cd = cell.dropna(subset=["d"]).groupby("label").d.agg(["mean", "size"])
    out["iti_ang_dock_matched"] = cd["mean"].where(cd["size"] >= lc.D_OFF_MIN_SPOUTS)
    out["n_dock_spouts"] = cd["size"]
    kl = K.dropna(subset=["dv"]).groupby("label").dv.agg(["mean", "size"])
    out["iti_ang_dock_matched_lw"] = kl["mean"].where(kl["size"] >= lc.MIN_CELL)
    out = out.reset_index()
    # checks: iti_ang reproduces the primary table; D_off vs the learned-compensation T3 D_off (theta_E, its own licks)
    chk = out.merge(R[["label", "iti_ang"]], on="label", suffixes=("", "_recmark"))
    dmax = float(np.nanmax(np.abs(chk.iti_ang - chk.iti_ang_recmark)))
    print(f"== check: iti_ang vs recmark_sessions, {len(chk)} sessions, max |diff| = {dmax:.2e}", flush=True)
    t3 = lc.OUT / "t3_sessions.csv"
    if t3.exists():
        T3 = pd.read_csv(t3, usecols=["label", "D_off"]).merge(out[["label", "iti_ang_dock_matched"]], on="label").dropna()
        print(f"== check: iti_ang_dock_matched (3-D landing angle, KEY licks) vs T3 D_off (theta_E, learned_compensation): "
              f"r {np.corrcoef(T3.D_off, T3.iti_ang_dock_matched)[0, 1]:.3f} over {len(T3)} sessions", flush=True)
    print(f"sign {sgn:+.0f}; sessions with a value: " + ", ".join(f"{c} {int(out[c].notna().sum())}" for c in AXES), flush=True)
    pd.set_option("display.width", 220)
    print("== per animal x epoch (rule 8): session means of each axis (deg, + = contralesional)\n"
          + out.groupby(["animal", "epoch"])[AXES].mean().reindex(columns=AXES).round(2).to_string(), flush=True)
    print("== across post sessions: r between axes\n" + out[out.epoch != "pre"][AXES].corr().round(2).to_string())
    out.to_csv(il.OUT_W / "itimatch_axes.csv", index=False)
    return out


# ------------------------------------------------------------------------------------------------ models
def models() -> None:
    import lick_recovery_markers as lrm
    import lick_recovery_primary as lrp
    X = pd.read_csv(il.OUT_W / "itimatch_axes.csv")
    rows = []
    for pk, f in (("ITI", _arg("--bal-all")), ("ITI docked", _arg("--bal-dock"))):
        S = pd.read_csv(f)
        S = S[S.set == lrm.QSET].copy()
        S["marker"] = lrp.marker(S, lrp.ITI_WIN).to_numpy()
        S = S.drop(columns=[c for c in ("iti_ang",) if c in S]).merge(X[["label", *AXES, "iti_ang_dock_matched_lw"]],
                                                                       on="label", how="left")
        for yv, _yl in [*YS, ("iti_ang_dock_matched_lw", "")]:
            rows.append(pd.DataFrame({"panel": f"{pk} -> {yv}", "set": pk, "yname": yv, "label": S.label,
                                      "animal": S.animal, "epoch": S.epoch, "day": S.day, "y": S[yv], "marker": S.marker,
                                      "n_licks": S.n_licks}))
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(il.OUT_W / "itimatch_sessions.csv", index=False)
    M, CV, ANr, LC = lrp.fit_panels(T)
    M.to_csv(il.OUT_W / "itimatch_models.csv", index=False)
    CV.to_csv(il.OUT_W / "itimatch_cv.csv", index=False)
    ANr.to_csv(il.OUT_W / "itimatch_animals.csv", index=False)
    LC.to_csv(il.OUT_W / "itimatch_lickcount.csv", index=False)
    pd.set_option("display.width", 240)
    print("== per animal (rule 8): r of the marker (stored hemispheric sign) / of log(day) with the axis\n"
          + ANr.pivot_table(index="panel", columns="animal", values=["r", "r_days"], sort=False).round(2).to_string())
    print("== mixed models\n" + M.round(4).to_string(index=False))
    print("== cross-validated\n" + summary_table(CV).to_string(index=False))


def summary_table(CV: pd.DataFrame) -> pd.DataFrame:
    c = CV[CV.pred == "marker"].set_index("panel")
    d = CV[CV.pred == "days"].set_index("panel")
    return pd.DataFrame({"n": c.n, "loao": c.loao, "loao_days": d.loao.reindex(c.index), "null95": c.null95,
                         "p_shuffle": c.p_shuffle, "loblk": c.loblk, "loblk_days": d.loblk.reindex(c.index),
                         "loblk_plus_days": c.loblk_plus_days}).round(3).reset_index()


# ------------------------------------------------------------------------------------------------ balance vs angle
# resp column per set = lick_balance_angle_lmm.RESP_COL (DECISIONS 11-17: cued 0..1.0 s, ITI 0..0.5 s)


def bal_angle_points() -> pd.DataFrame:
    """lick_balance_angle_lmm.points with the ITI SUBSET hook applied to the per-lick cache BEFORE the joins (k = cache
    index kept), resp read per set from lba.RESP_COL of the given lateral-ratio table (--ratio; for a docked run the docked
    lateral ratio, whose ITI bout starts are docked)."""
    import iti_lick_order as lo
    import lick_balance_angle_lmm as lba
    import lick_balance_reference as lbr
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    P = P[il.iti_subset_keep(P)].reset_index(drop=True)
    P = P[lba.gap_keep(P)].reset_index(drop=True)      # --gap2 (11-17): bout starts with gap >= il.GAP_MAIN only
    print(f"ITI subset {il.ITI_SUBSET or 'none'}: {int((P.set == 'iti').sum())} ITI licks", flush=True)
    C = pd.read_csv(il.in_path(f"covref_covariates{lo.SRC_TAG}.csv.gz"), usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    # --idx (11-17): lba's per-session worker / columns / resp table also carry dec_idx / resp_idx (lba.bounded_index)
    res, fail = ak.fan_sessions(sorted(P.label.unique()), lba.session_worker())
    if fail:
        raise RuntimeError(f"balance failed for {len(fail)} sessions: {fail[:5]}")
    P = P.merge(pd.concat([v for _i, v in res], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    P["cls"] = np.where(P.order == 0, "", " later")
    g = P.groupby(["set", "cls", "label", "animal", "epoch", "pos"])
    S = g[["angle", *lba.bal_cols()]].mean().join(g.size().rename("n")).reset_index()
    S = S[S.n >= lba.MIN_LICKS]
    S["set"] = S.set.map(lba.SETMAP) + S.cls
    S = S.drop(columns="cls")
    S = S.merge(lba.resp_tables(pd.read_csv(_arg("--ratio")), lba.RESP_COL), on=["set", "label", "pos"], how="left")
    S = S.rename(columns={f"bal_{a}": f"dec_{a}" for a in lbr.PAIRS})
    S["dec_orofacial"] = S[[f"dec_{a}" for a in lbr.PAIRS]].mean(1)
    S = lba.add_dec_idx(S)
    S["post"] = (S.epoch != "pre").astype(int)
    return S


def farcontra(S: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Far-contra session x spout cells only (cued: target far contra; ITI: previous spout far contra): per animal r of
    balance with angle (pre, post, all) and OLS slope, then y ~ ac * post + (1 | animal) (ac = angle - the pre mean
    far-contra angle of the set): pre slope, slope change post, post shift at the same angle."""
    import lick_balance_angle_lmm as lba
    rows, an = [], []
    for st in ("cued", "iti"):
        G0 = S[(S.set == st) & (S.pos == FC)]
        if G0.empty:
            continue
        mu = G0[G0.epoch == "pre"].angle.mean()
        for meas in lba.meas_list():                  # + dec_idx / resp_idx with --idx (11-17)
            for a in lba.PAIRS:
                y = f"{meas}_{a}"
                if y not in G0:
                    continue
                G = G0.dropna(subset=["angle", y]).rename(columns={y: "y"})
                G["ac"] = G.angle - mu
                for animal, h in G.groupby("animal"):
                    rec = {"set": st, "measure": meas, "pair": a, "animal": animal, "n": len(h)}
                    for nm, hh in (("pre", h[h.post == 0]), ("post", h[h.post == 1]), ("all", h)):
                        ok = len(hh) >= 4 and hh.ac.std() > 0 and hh.y.std() > 0
                        rec[f"r_{nm}"] = np.corrcoef(hh.ac, hh.y)[0, 1] if ok else np.nan
                        rec[f"slope_{nm}"] = np.polyfit(hh.ac, hh.y, 1)[0] if ok else np.nan
                        rec[f"n_{nm}"] = len(hh)
                    rec["angle_shift_post"] = h[h.post == 1].angle.mean() - h[h.post == 0].angle.mean()
                    rec["y_shift_post"] = h[h.post == 1].y.mean() - h[h.post == 0].y.mean()
                    an.append(rec)
                try:
                    f = lba._fit(G, "y ~ ac * post")
                    fe, p_ = f.fe_params, f.pvalues
                    rows.append({"set": st, "measure": meas, "pair": a, "n": len(G), "n_sessions_post": int(G.post.sum()),
                                 "slope_pre": fe["ac"], "p_slope_pre": p_["ac"], "slope_change_post": fe["ac:post"],
                                 "p_slope_change": p_["ac:post"], "shift_post_same_angle": fe["post"],
                                 "p_shift": p_["post"], "angle_centre": mu,
                                 "angle_shift_post_raw": G[G.post == 1].angle.mean() - G[G.post == 0].angle.mean()})
                    f0 = lba._fit(G, "y ~ post")
                    rows[-1].update({"shift_post_no_angle": f0.fe_params["post"], "p_shift_no_angle": f0.pvalues["post"]})
                    # INSIDE the pre angle range: post sessions whose angle lies within the animal's pre sessions' range
                    # (acute far-contra licks fall beyond it, where the pre line is extrapolated)
                    rows[-1].update(lba.inside_shift(G))
                except Exception as ex:                                   # noqa: BLE001  reported, not hidden
                    rows.append({"set": st, "measure": meas, "pair": a, "note": f"FAILED {ex}"})
    return pd.DataFrame(rows), pd.DataFrame(an)


def balance_angle() -> None:
    import lick_balance_angle_lmm as lba
    from figdata import save_fd
    from wfield_local import trial_set_text as tst
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    tst.declare(text=lba.trial_set_sentence())
    lba.points = bal_angle_points                 # the full LMMs of lick_balance_angle_lmm on the subset table
    F = lba.compute()
    save_fd("lick_balance_angle", F)
    lba.plot(F)
    M, A = farcontra(F["S"])
    M.to_csv(il.OUT_W / "farcontra_lmm.csv", index=False)
    A.to_csv(il.OUT_W / "farcontra_animals.csv", index=False)
    pd.set_option("display.width", 240)
    print("== FAR CONTRA per animal (rule 8): r / OLS slope of balance on angle (pre, post), post - pre shifts\n"
          + A.round(3).to_string(index=False))
    print("== FAR CONTRA mixed models y ~ angle * post + (1 | animal)\n" + M.round(4).to_string(index=False))
    plot_farcontra(F["S"], M, A)


# ------------------------------------------------------------------------------------------------ figures
def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _sess(ax, x, y, an, ep, pre_gray=True) -> None:
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    for (a_, e_), ix in pd.DataFrame({"a": an, "e": ep}).groupby(["a", "e"]).groups.items():
        c = "0.6" if (e_ == "pre" and pre_gray) else EPOCH_LINE.get(e_, "k")
        ax.plot(np.asarray(x)[ix], np.asarray(y)[ix], ANIMAL_MARKER.get(a_, "o"), mfc="none", mec=c, mew=1.0, ms=5.5,
                zorder=1 if e_ == "pre" else 2)


def _legend(fig) -> None:
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    h = [Line2D([], [], color="0.6", marker="o", mfc="none", lw=0, label="pre (not fitted)")]
    h += [Line2D([], [], color=EPOCH_LINE[e], marker="o", mfc="none", lw=0, label=e) for e in ("acute", "subacute", "chronic")]
    h += [Line2D([], [], color="0.3", marker=ANIMAL_MARKER[a], mfc="none", lw=0, label=a) for a in AN]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=8.5, frameon=False)


def plot_models() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import lick_recovery_primary as lrp
    from wfield_local import sig_marks
    from wfield_local.figure_layout import save_png_svg
    T = pd.read_csv(il.OUT_W / "itimatch_sessions.csv")
    CV = pd.read_csv(il.OUT_W / "itimatch_cv.csv")
    ANr = pd.read_csv(il.OUT_W / "itimatch_animals.csv")
    M = pd.read_csv(il.OUT_W / "itimatch_models.csv")
    fig, axs = plt.subplots(len(PREDS), len(YS), figsize=(4.6 * len(YS), 4.9 * len(PREDS)), squeeze=False)
    for i, (pk, plab) in enumerate(PREDS):
        for j, (yv, ylab) in enumerate(YS):
            ax = axs[i, j]
            pn = f"{pk} -> {yv}"
            G = T[T.panel == pn].dropna(subset=["y", "marker"])
            _sess(ax, lrp.bal_disp(G.marker), G.y, G.animal.to_numpy(), G.epoch.to_numpy())
            r = ANr[ANr.panel == pn].set_index("animal").r
            c = CV[(CV.panel == pn)].set_index("pred")
            m = M[(M.panel == pn) & (M.model == "concurrent, log(day)")]
            ttl = f"{plab} -> {ylab}\nper-animal r " + " / ".join(f"{lrp.bal_disp(r.get(a, np.nan)):+.2f}" for a in AN)
            if "marker" in c.index:
                ttl += (f"\nunseen-animal cvR² {c.loc['marker', 'loao']:.2f} (days {c.loc['days', 'loao']:.2f}), "
                        f"shuffle p {c.loc['marker', 'p_shuffle']:.2g}")
            if len(m) and np.isfinite(m.p.iloc[0]):
                ttl += f"\nwithin-animal slope {lrp.bal_disp(m.slope.iloc[0]):+.1f}, p {m.p.iloc[0]:.2g} (log day)"
            ax.set_title(ttl, fontsize=7.5)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.axvline(0, color="0.8", lw=0.6)
            ax.set_xlabel(f"{lrp.BAL_SHORT}, {plab} (gain ratio 0.1-0.5 s, vs pre)", fontsize=7.5)
            ax.set_ylabel(ylab + " (deg)", fontsize=7.5)
            if "marker" in c.index and c.loc["marker", "p_shuffle"] < 0.05 and c.loc["marker", "loao"] > max(c.loc["days", "loao"], 0):
                sig_marks.star_top(ax, float(np.nanmean(ax.get_xlim())))
            _style(ax)
    sig_marks.reserve_headroom(fig)
    _legend(fig)
    fig.suptitle("ITI recovery axes matched to the previous spout / ITI phase and docked ITI licks, vs the ITI hemispheric "
                 "balance (first ITI lick after >= 2 s quiet, oro + SSs, 0.1-0.5 s; DAQ licks; post-stroke sessions fit, "
                 "gray = pre).\n* = unseen-animal cvR² above days since stroke AND within-animal shuffle p < 0.05. Balance "
                 "shown as implied direction (= -balance); r / slopes vs the value shown. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save_png_svg(fig, il.OUT_W / "figures" / "iti_matched_markers_scatter.png", dpi=100)
    plt.close(fig)
    # cvR2 summary
    fig, axs = plt.subplots(1, len(PREDS), figsize=(13, 4.6), squeeze=False, sharey=True)
    for i, (pk, plab) in enumerate(PREDS):
        ax = axs[0, i]
        x = np.arange(len(YS))
        cm = [CV[(CV.panel == f"{pk} -> {yv}") & (CV.pred == "marker")] for yv, _ in YS]
        cd = [CV[(CV.panel == f"{pk} -> {yv}") & (CV.pred == "days")] for yv, _ in YS]
        vm = [c.loao.iloc[0] if len(c) else np.nan for c in cm]
        vd = [c.loao.iloc[0] if len(c) else np.nan for c in cd]
        ax.bar(x - 0.2, vd, 0.38, color="0.65", label="days since stroke")
        ax.bar(x + 0.2, vm, 0.38, color="#C44E52", label="ITI balance")
        ax.plot(x + 0.2, [c.null95.iloc[0] if len(c) else np.nan for c in cm], "k_", ms=12, mew=1.4,
                label="shuffle null, 95th pct")
        for xi, c in enumerate(cm):
            if len(c) and c.p_shuffle.iloc[0] < 0.05 and vm[xi] > max(vd[xi], 0):
                sig_marks.star_top(ax, xi + 0.2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-0.8, 1)
        ax.set_xticks(x, [XLAB[yv] for yv, _ in YS], fontsize=7.5)
        ax.set_title(f"{plab}: leave-one-animal-out cvR²", fontsize=9)
        _style(ax)
    axs[0, 0].legend(fontsize=7.5, frameon=False, loc="lower left")
    sig_marks.reserve_headroom(fig)
    fig.suptitle("Does matching ITI licks to the previous spout / phase, or using docked licks only, change the ITI "
                 "marker result? Post-stroke sessions; * = above days and shuffle p < 0.05. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_svg(fig, il.OUT_W / "figures" / "iti_matched_markers_cv.png", dpi=110)
    plt.close(fig)


def plot_farcontra(S: pd.DataFrame, M: pd.DataFrame, A: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import lick_balance_angle_lmm as lba
    from wfield_local import sig_marks
    from wfield_local.figure_layout import save_png_svg
    sets = [st for st in ("cued", "iti") if ((S.set == st) & (S.pos == FC)).any()]
    meas = [("resp", "orofacial", "response-AUC balance log2(contra / ipsi), orofacial pool"),
            ("dec", "orofacial", "decoder-window difference X_right - X_left, 0-0.2 s, orofacial (x1e3 dF/F)"),
            ("resp_idx", "orofacial", "response-AUC bounded index (c - i) / (|c| + |i|), orofacial pool"),
            ("dec_idx", "orofacial", "decoder-window bounded index (c - i) / (|c| + |i|), 0-0.2 s,\n"
                                     f"baseline {il.win_label(lba.cr.BASE_SPOUT)} before the peak, orofacial")]
    meas = [m for m in meas if f"{m[0]}_{m[1]}" in S]          # the indices only with --idx (11-17)
    fig, axs = plt.subplots(len(sets), len(meas), figsize=(6.2 * len(meas), 5.0 * len(sets)), squeeze=False)
    for i, st in enumerate(sets):
        G = S[(S.set == st) & (S.pos == FC)]
        for j, (ms, pr, lab) in enumerate(meas):
            ax = axs[i, j]
            y = f"{ms}_{pr}"
            g = G.dropna(subset=["angle", y])
            _sess(ax, g.angle, g[y], g.animal.to_numpy(), g.epoch.to_numpy(), pre_gray=False)
            m = M[(M.set == st) & (M.measure == ms) & (M.pair == pr)]
            a = A[(A.set == st) & (A.measure == ms) & (A.pair == pr)].set_index("animal")
            win = "0-1.0 s" if st == "cued" else "0-0.5 s"
            ttl = (f"{'cued first lick, target far contra' if st == 'cued' else 'ITI bout start, previous spout far contra'}"
                   f" ({win if ms.startswith('resp') else '0-0.2 s'})\nper-animal r pre " +
                   " / ".join(f"{a.r_pre.get(x, np.nan):+.2f}" for x in AN) + "; post " +
                   " / ".join(f"{a.r_post.get(x, np.nan):+.2f}" for x in AN))
            if len(m) and "p_shift" in m and np.isfinite(m.p_shift.iloc[0]):
                r_ = m.iloc[0]
                ttl += (f"\npre slope {r_.slope_pre:+.4f}/deg (p {r_.p_slope_pre:.2g}); slope change {r_.slope_change_post:+.4f} "
                        f"(p {r_.p_slope_change:.2g})\npost shift at same angle {r_.shift_post_same_angle:+.3f} "
                        f"(p {r_.p_shift:.2g}); without angle {r_.shift_post_no_angle:+.3f} (p {r_.p_shift_no_angle:.2g})")
                if r_.p_shift < 0.05:
                    sig_marks.star_top(ax, float(np.nanmean(ax.get_xlim())))
                xx = np.linspace(g.angle.min(), g.angle.max(), 20)
                pre = g[g.post == 0]
                b0 = pre[y].mean() - r_.slope_pre * (pre.angle.mean() - r_.angle_centre)
                ax.plot(xx, b0 + r_.slope_pre * (xx - r_.angle_centre), "k-", lw=1.4)
                ax.plot(xx, b0 + r_.shift_post_same_angle + (r_.slope_pre + r_.slope_change_post) * (xx - r_.angle_centre),
                        "--", color="#C44E52", lw=1.4)
            ax.set_title(ttl, fontsize=7.5)
            ax.axhline(0, color="0.8", lw=0.6)
            ax.set_xlabel("session mean lick angle at this spout (deg, + = contralesional)", fontsize=8)
            ax.set_ylabel(lab, fontsize=7.5)
            _style(ax)
    sig_marks.reserve_headroom(fig)
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    h = [Line2D([], [], color=EPOCH_LINE[e], marker="o", mfc="none", lw=0, label=e) for e in ("pre", "acute", "subacute", "chronic")]
    h += [Line2D([], [], color="0.3", marker=ANIMAL_MARKER[a], mfc="none", lw=0, label=a) for a in AN]
    h += [Line2D([], [], color="k", lw=1.4, label="pre fit"), Line2D([], [], color="#C44E52", ls="--", lw=1.4, label="post fit")]
    fig.legend(handles=h, loc="lower center", ncol=10, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle(f"Far-contra only (target fixed): raw hemispheric balance vs the session's lick angle at that spout "
                 f"\n(ITI subset: {il.ITI_SUBSET or 'all ITI'}; DAQ licks; one point per session). Mixed model y ~ angle * post "
                 f"+ (1 | animal); * = post shift at the same angle p < 0.05. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    save_png_svg(fig, il.OUT_W / "figures" / "farcontra_balance_angle.png", dpi=105)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    from wfield_local import writeguard
    writeguard.assert_writable(il.OUT_W)
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    if cmd == "docked-balance":                       # lick_balance_subsets session table under the ITI subset hook
        if il.ITI_SUBSET != "dock":
            raise SystemExit("docked-balance needs WFIELD_ITI_FILTER=dock")
        import lick_balance_subsets as lbs
        sys.argv = [sys.argv[0], "--sessions-only"]
        return lbs.main()
    if cmd == "docked-primary":                       # lick_recovery_primary on that table (reproduces itifilter_dock)
        if il.ITI_SUBSET != "dock":
            raise SystemExit("docked-primary needs WFIELD_ITI_FILTER=dock")
        import lick_recovery_primary as lrp
        sys.argv = [sys.argv[0]]
        return lrp.main()
    if cmd == "axes":
        axes()
    elif cmd == "models":
        models()
        plot_models()
    elif cmd == "plot":
        plot_models()
    elif cmd == "balance-angle":
        balance_angle()
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
