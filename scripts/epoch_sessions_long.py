"""Per-SESSION long tables for the epoch scalar families + CD trajectories, for the mixed models
(encoding window via Priya, 10-20 late: "change the collectors to also emit a per-session table ... a new
*_sessions_long.csv next to the existing files; don't change the current files -- the deck reads them").

    WIDEFIELD_PRE_FROM=0801 python -m scripts.epoch_sessions_long [--arms cue lick ...] [--families 8g 10 10b 11 11c cd]
    python -m scripts.epoch_sessions_long --check-arm cue        # all-pre run: post rows / LOO means == current files

WHAT IT DOES. Repeats each collector's loop over `grant_kit._collect_7` (the same cached pool, same reference, same
rng seed and draw ORDER for the random ones) and keeps every session instead of averaging the pre sessions:
  11  / 11amp / 11pos   `grant_encoder._enc_tables`    frozen EV (slot 0), EV after rescale (2), gain a (1), per-position
                                                       shape r2 (3)
  8g                    `grant_geometry._rdm_rows`     per-position RDM row r (rng: default_rng(_seed(an, align, variant,
                                                       "8g")) -- full reference first, then (pat, LOO ref) per pre session,
                                                       then each post session; identical draws)
  10 / 10b              `grant_kit._matrices_pattern`  best-match accuracy + rank (`_best_match`), per-position best==own
  11c                   `grant_figures._enc_half_tables` ceiling / frozen (matched) slot 2 (rng default_rng(_seed(an,
                                                       align, variant)): pre sessions then post) + frozen (all pre) = 11
PRE SESSIONS are each scored against the OTHER pre sessions (`_pre_reference(exclude=s)`), exactly the per-session terms
the collectors average into today's PRE bar -- so for 8g / 11 / 11c the mean of the pre rows per animal IS today's PRE
value. 10 / 10b DIFFER ON PURPOSE: today's PRE is `_best_match` of the AVERAGED LOO matrix (argmax of a mean -- the
bias `grant_figures._matrices_best_match_destination` documents); here each pre session's own matrix is scored, so the
pre mean is lower.
BASELINE: run with WIDEFIELD_PRE_FROM=0801 (August pre from 0806; 0805 is excluded by date_policy). The pre pool, the
reference and therefore EVERY value (post too) differ from the deck's all-pre files; the long files say so
(``baseline``).
CD (`cd_trajectories` dumps, `scripts/cd_cross_animal_figure.collect` settings: orth, contrast reference, gate lick):
per session x position the mean projection on its OWN direction (diagonal cell) in fixed windows -- pre-cue [-2, 0) s
(precue dump), cue [0, 1) s (cue dump), lick [0, 1) s (lick dump). CAVEAT: the dumps were fitted on ALL pre sessions
(June included; their cohort guard refuses an August-only load) and pre values are IN-SAMPLE (the directions were fitted
on them), flagged ``pre_in_sample``. Only August pre + curated post rows are emitted.

Output, next to each family's existing data/<name>_sessions.csv (deck files untouched):
data/<name>_sessions_long.csv -- family, arm, align, variant, label (PSxx_MMDD), animal, epoch, day
(`epochs.days_since_stroke`), position, measure, value, n_trials, pre_loo, baseline.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = {"ENL": ("precue", "working"), "cue": ("cue", "working"), "lick": ("lick", "lick"),
        "cuelick": ("cue", "lick"), "ENLlick": ("precue", "lick")}
FAMILIES = ("11", "8g", "10", "11c", "cd")
MIN_TRIALS = 10
CD_WINDOWS = {"precue": ("pre-cue [-2,0) s", -2.0, 0.0), "cue": ("cue [0,1) s", 0.0, 1.0),
              "lick": ("lick [0,1) s", 0.0, 1.0)}


def _short(q):
    from wfield_local.epoch_matching import _short_labels
    from wfield_local.grant_figures import CONF_LABELS
    return dict(zip(CONF_LABELS, _short_labels()))[q]


def _post_dates(an: str) -> dict:
    """{day: mmdd} for the animal's post sessions (`grant_kit._day` is the collectors' key)."""
    from wfield_local import config
    from wfield_local import grant_kit as GK
    return {GK._day(an, lab.split("_")[1]): lab.split("_")[1]
            for lab in config.phase_labels("post") if lab.startswith(an + "_")}


def _rows_for(store, an, align, variant, fam_rows):
    """Common bookkeeping: yields (label, mmdd, epoch_key, pat, pre_loo, ref) for every session of ``an``."""
    from wfield_local import grant_kit as GK
    pre_by_sess, by_day = store[an]
    days = _post_dates(an)
    full = GK._pre_reference(pre_by_sess)
    for s, pat in pre_by_sess.items():
        yield f"{an}_{s}", pat, True, GK._pre_reference(pre_by_sess, exclude=s)
    for d, pat in by_day.items():
        yield f"{an}_{days.get(d, f'day{d}')}", pat, False, full


def fam_11(store, align, variant):
    from wfield_local import grant_encoder as GE
    from wfield_local.grant_figures import CONF_LABELS
    out = []
    for an, (pre_by_sess, by_day) in store.items():
        if len(pre_by_sess) < 2 or not by_day:
            continue
        for lab, pat, loo, ref in _rows_for(store, an, align, variant, out):
            t = GE._enc_scores(pat, ref)
            if not np.isfinite(t[0]):
                continue
            n = int(sum(len(v) for v in pat.values()))
            for meas, v, fam in (("frozen EV", t[0], "11"), ("EV after rescale", t[2], "11"), ("gain", t[1], "11amp")):
                out.append((fam, lab, "", meas, v, n, loo))
            for q in CONF_LABELS:
                if q in t[3]:
                    out.append(("11pos", lab, _short(q), "shape r2", t[3][q], len(pat.get(q, [])), loo))
    return out


def fam_8g(store, align, variant):
    from wfield_local import grant_geometry as GG
    from wfield_local import grant_kit as GK
    from wfield_local.grant_figures import CONF_LABELS
    out = []
    for an in GK.ANIMALS:                                  # _rdm_rows' loop order (rng per animal)
        if an not in store:
            continue
        pre_by_sess, by_day = store[an]
        rng = np.random.default_rng(GK._seed(an, align, variant, "8g"))
        Dpre = GG._crossnobis_within(GK._pre_reference(pre_by_sess), rng, CONF_LABELS)
        days = _post_dates(an)
        sess = []
        for s, pat in pre_by_sess.items():               # SAME draw order: pat, then its LOO reference
            D = GG._crossnobis_within(pat, rng, CONF_LABELS)
            Dref = GG._crossnobis_within(GK._pre_reference(pre_by_sess, exclude=s), rng, CONF_LABELS)
            sess.append((f"{an}_{s}", pat, GG._rdm_scores(D, Dref)[1], True))
        for d, pat in by_day.items():
            D = GG._crossnobis_within(pat, rng, CONF_LABELS)
            sess.append((f"{an}_{days.get(d, f'day{d}')}", pat, GG._rdm_scores(D, Dpre)[1], False))
        for lab, pat, rows, loo in sess:
            for i, q in enumerate(CONF_LABELS):
                v = rows.get(q) if isinstance(rows, dict) else rows[i]
                if v is not None and np.isfinite(v):
                    out.append(("8g", lab, _short(q), "row correlation", float(v), len(pat.get(q, [])), loo))
    return out


def fam_10(store, align, variant):
    from wfield_local import grant_figures as G
    from wfield_local import grant_kit as GK
    from wfield_local.grant_figures import CONF_LABELS
    out = []
    for an in store:
        for lab, pat, loo, ref in _rows_for(store, an, align, variant, out):
            M = GK._corr_matrix(GK._means(pat), GK._means(ref))
            best, rank = G._best_match(np.asarray(M, float))
            ok = np.isfinite(rank)
            n = int(sum(len(v) for v in pat.values()))
            if ok.any():
                out.append(("10", lab, "", "best-match accuracy", float(np.mean(best[ok] == np.flatnonzero(ok))), n, loo))
                out.append(("10", lab, "", "rank of correct position", float(np.nanmean(rank)), n, loo))
            for i, q in enumerate(CONF_LABELS):
                row = np.asarray(M, float)[i]
                if np.isfinite(row).any():
                    out.append(("10b", lab, _short(q), "best match is own position",
                                float(np.nanargmax(row) == i), len(pat.get(q, [])), loo))
    return out


def fam_11c(store, align, variant):
    from wfield_local import grant_figures as G
    from wfield_local import grant_kit as GK
    out = []
    for an, (pre_by_sess, by_day) in store.items():   # _enc_half_tables' loop order
        rng = np.random.default_rng(GK._seed(an, align, variant))
        days = _post_dates(an)
        items = [(f"{an}_{s}", pat, True, GK._pre_reference(pre_by_sess, exclude=s)) for s, pat in pre_by_sess.items()]
        full = GK._pre_reference(pre_by_sess)
        items += [(f"{an}_{days.get(d, f'day{d}')}", pat, False, full) for d, pat in by_day.items()]
        for lab, pat, loo, ref in items:
            dd = G._enc_half_scores(pat, rng, ref_pool=ref)
            n = int(sum(len(v) for v in pat.values()))
            for meas, t in (("ceiling", dd["ceiling"]), ("frozen (matched)", dd["matched"])):
                if t is not None and np.isfinite(t[0]):
                    out.append(("11c", lab, "", meas, float(t[2]), n, loo))
    return out


def fam_cd(_arm=None):
    """CD window means per session x position (diagonal cell), all three alignments."""
    import importlib
    from wfield_local import cd_trajectories as cdt
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    cx = importlib.import_module("scripts.cd_cross_animal_figure")
    out = []
    for align, (meas, lo, hi) in CD_WINDOWS.items():
        got, problems = cx.collect(cdt.default_out(), align=align)
        for p in problems:
            print(f"   CD {align}: {p}")
        for _a, res in got.items():
            t = np.arange(-res["pre_n"], res["post_n"]) / float(res["fs"])
            m = (t >= lo) & (t < hi)
            for (lab, ep, cd_p, tr_p), d in (res.get("session_traces") or {}).items():
                if cd_p != tr_p:
                    continue
                v = float(np.nanmean(np.asarray(d["mean"], float)[m]))
                out.append(("cd", lab, POSITION_NAMES.get(cd_p, str(cd_p)) if isinstance(POSITION_NAMES, dict)
                            else str(cd_p), f"CD projection, {meas}", v, int(d.get("n", 0)), ep == "pre",
                            align))
    return out


def _frame(rows, arm, align, variant, baseline):
    from wfield_local import epochs
    D = pd.DataFrame(rows, columns=["family", "label", "position", "measure", "value", "n_trials", "pre_loo"])
    D.insert(1, "arm", arm)
    D.insert(2, "align", align)
    D.insert(3, "variant", variant)
    D.insert(5, "animal", D.label.str[:4])
    D.insert(6, "epoch", [epochs.epoch_of(x) for x in D.label])
    D.insert(7, "day", [epochs.days_since_stroke(x) for x in D.label])
    D["baseline"] = baseline
    return D


STEMS = {"11": "epoch_11_encoder_gain_shape_{a}_{v}", "11amp": "epoch_11amp_encoder_amplitude_{a}_{v}",
         "11pos": "epoch_11pos_encoder_shape_{a}_{v}", "8g": "epoch_8g_geometry_by_position_{a}_{v}",
         "10": "epoch_10_best_match_{a}_{v}", "10b": "epoch_10b_best_match_by_position_{a}_{v}",
         "11c": "epoch_11c_encoder_ceiling_{a}_{v}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--out", type=Path, default=None, help="figure dir (default <labcams>/grant_figures/epoch; files -> data/)")
    ap.add_argument("--check-arm", default=None, help="validate one arm against the current all-pre files, no writing")
    a = ap.parse_args(argv)
    from wfield_local import analysis_kit as ak
    from wfield_local import cd_trajectories as cdt
    from wfield_local import config
    from wfield_local import grant_kit as GK
    pre_from = config.pre_from() if hasattr(config, "pre_from") else None
    baseline = f"pre from {pre_from}" if pre_from else "all pre (June + August)"
    if a.check_arm is None and not pre_from and set(a.families) - {"cd"}:
        raise SystemExit("set WIDEFIELD_PRE_FROM=0801 (August-only baseline, Priya's rule) or pass --check-arm")
    from wfield_local import figure_layout as FL
    fig_dir = a.out or Path(cdt.default_out())          # the families' figure dir; files go to its data/
    cur = set(ak.curated_labels())
    arms = [a.check_arm] if a.check_arm else ([] if set(a.families) <= {"cd"} else a.arms)
    fams = {"11": fam_11, "8g": fam_8g, "10": fam_10, "11c": fam_11c}
    for arm in arms:
        align, variant = ARMS[arm]
        print(f"=== {arm} ({align}, {variant}): collecting", flush=True)
        store, _days = GK._collect_7(align, variant, MIN_TRIALS)
        rows = []
        for f in a.families:
            if f in fams:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rows += fams[f](store, align, variant)
        D = _frame(rows, arm, align, variant, baseline)
        D = D[D.label.isin(cur)]
        if a.check_arm:
            _check(D, fig_dir, align, variant)
            continue
        for fam, g in D.groupby("family"):
            stem = STEMS[fam].format(a=align, v=variant)
            if fam == "10":
                for meas, gm in g.groupby("measure"):
                    key = "acc" if meas.startswith("best") else "rank"
                    p = FL.sidecar_for(fig_dir, stem.replace('best_match_', f'best_match_{key}_'), "_sessions_long.csv")
                    gm.to_csv(p, index=False)
                    print("  ->", p.name, len(gm))
                continue
            p = FL.sidecar_for(fig_dir, stem, "_sessions_long.csv")
            g.to_csv(p, index=False)
            print("  ->", p.name, len(g))
    if "cd" in a.families and not a.check_arm:
        C = pd.DataFrame(fam_cd(), columns=["family", "label", "position", "measure", "value", "n_trials", "pre_loo",
                                            "align"])
        from wfield_local import epochs
        C["pre_loo"] = False
        C["pre_in_sample"] = C.label.map(lambda x: epochs.epoch_of(x) == "pre")
        C = C[C.label.isin(cur) & ~((C.pre_in_sample) & (C.label.str[5:9] < "0806"))].copy()
        C.insert(1, "arm", C["align"])          # NOT C.align: that is DataFrame.align (a method)
        C.insert(3, "variant", "gate lick, orth, contrast")
        C.insert(5, "animal", C.label.str[:4])
        C.insert(6, "epoch", [epochs.epoch_of(x) for x in C.label])
        C.insert(7, "day", [epochs.days_since_stroke(x) for x in C.label])
        C["baseline"] = "directions fitted on ALL pre (June + August); pre rows in-sample; August pre rows only"
        p = FL.sidecar_for(fig_dir, "cd_window_projection", "_sessions_long.csv")
        C.to_csv(p, index=False)
        print("  ->", p.name, len(C))
    return 0


def _check(D, fig_dir, align, variant):
    """All-pre run: per animal, post rows in day order == the current _sessions.csv post rows, and the mean of the
    pre LOO rows == its PRE row (8g / 11 / 11c; 10 / 10b differ by design)."""
    from wfield_local import epochs
    for fam, meas, stem, key in (("11", "frozen EV", STEMS["11"], "frozen EV"), ("8g", "row correlation", STEMS["8g"], None),
                                 ("11amp", "gain", STEMS["11amp"], "gain")):
        from wfield_local import figure_layout as FL
        f = FL.sidecar_for(fig_dir, stem.format(a=align, v=variant), "_sessions.csv", mkdir=False)
        if not f.exists():
            print(f"  {fam}: no {f.name}")
            continue
        S = pd.read_csv(f)
        g = D[(D.family == fam) & (D.measure == meas)]
        for an, ga in g.groupby("animal"):
            for pos in (ga.position.unique() if key is None else [""]):
                gp = ga[ga.position == pos]
                sp = S[(S.animal == an) & (S.position == (key or pos))]
                pre_ok = np.isclose(gp[gp.pre_loo].value.mean(), sp[sp.epoch == "pre"].value.mean(), atol=1e-6)
                post_new = gp[~gp.pre_loo].sort_values("day").value.to_numpy()
                post_old = sp[sp.epoch != "pre"].value.to_numpy()
                same = len(post_new) == len(post_old) and np.allclose(post_new, post_old, atol=1e-6)
                if not (pre_ok and same):
                    print(f"  {fam} {an} {pos or key}: pre_mean_ok={pre_ok} post_same={same} "
                          f"(n {len(post_new)} vs {len(post_old)})")
        print(f"  {fam}: checked {g.animal.nunique()} animals x {g.position.nunique() or 1} positions")
    _ = epochs


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    raise SystemExit(main())
