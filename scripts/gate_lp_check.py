"""GATE vs LP CHECK (Priya-approved 11-01): does the DAQ-contact engagement gate / quit point mislabel trials that
contain real LP (video) lick attempts?

Every "engaged / working" gate and the S28 quit point is built from DAQ spout-CONTACT licks. Post-stroke far-contra
attempts often miss the spout, so a trial with a real LP attempt can be scored as a non-response. Four measurements,
per session (curated PS92-95, August-only pre via WIDEFIELD_PRE_FROM=0801; cohort 2 never touched):

  Q1  trials the gate scores NOT engaged that hold an LP attempt in the response window; and the share of all
      LP-attempt trials that fall outside the gate. Gates measured (both reference-restricted, both unchanged code):
        img   `precue_engagement_states.engagement_gate`, responded = DAQ lick within decode.max_rt_s (3.5 s) --
              the imaging classes lick / miss_working / stopped (`position_coding_directions._gate_all`)
        quit  `quit_point.session_quit` on `engagement_decomposition.session_trials` (2.0 s scoring) -- the S28 quit;
              not-engaged = trials at/after the quit onset
  Q2  within working periods (img gate engaged): DAQ misses (no contact within 3.5 s) that hold an LP attempt.
  Q3  quit point under an LP rule: the SAME detector with responded = DAQ hit OR LP attempt within 2.0 s.
  Q4  pre-cue (ENL) windows declared DAQ-lick-free (`precue_window_start`, same rule, ms units, 2.0 and 1.0 s): the
      fraction holding an LP lick (cam4 every-context train + cam1-only events; cam1 is post-cue only in practice).

LP ATTEMPT = trial_attempts class in lp_licks.ATTEMPT_CLASSES (hit / incomplete) with the first LP lick (combined,
else cam4) inside the window -- the LP gate's own definition (`lp_licks.session_lp`, cue-time matched). A trial with
no LP verdict (no attempts row / unmatched) is `lp_cov = False` and is excluded from every LP fraction (counted).
Secondary: `lp_any` = any raw LP lick onset in [cue, cue + window] (includes jaw_only-class tongue events).

Trial universe = DAQ cues with a resolved position (classify_cues_with_backup), NOT the imaging-kept subset (coverage
/ lick-free drops), so the img gate here is the gate on the full behavioural trial list; stated in the results doc.

    python scripts/gate_lp_check.py extract [--jobs 4]     per-session pickles (local cache)
    python scripts/gate_lp_check.py analyse                tables -> <E>/gate_lp_check/, figdata + figure
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("WIDEFIELD_PRE_FROM", "0801")          # AUGUST-ONLY pre (before config import)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

OUT = E / "gate_lp_check"
CACHE = Path(os.environ.get("WFIELD_INPUT_CACHE", "C:/Users/SabatiniLab/wfield_cache")) / "gate_lp_check"
VERSION = 1
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPS = ("pre", "acute", "subacute", "chronic")
CANON = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
         "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
POS_ORDER = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
MAX_RT = 3.5            # decode.max_rt_s, the imaging engaged rule (checked against config below)
QUIT_RESP = 2.0         # quit detector scoring window
ENL_W = (2.0, 1.0)
NEAR_DAQ_S = 0.10       # an LP lick within this of a DAQ contact is that contact's tongue-out, not an extra lick
SEED = 20261101

# thresholds, fixed before looking (stated in the results doc)
THR_GATED_OUT = 0.05    # > 5 % of gated-out trials hold an LP attempt -> gate materially affected
THR_QUIT_MIN = 5.0      # median per-animal quit shift > 5 min, or any quit disappears in >= 2 animals
THR_ENL = 0.10          # > 10 % of DAQ-clean pre-cue windows hold an LP lick


# ============================================================================================ extraction
def one(label):
    from wfield_local import config, epochs, lp_licks
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_position_decoder import precue_window_start
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _load_licks
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue
    from wfield_local.precue_engagement_states import engagement_gate
    from scripts.rest_migration.engagement_decomposition import session_trials
    from scripts.rest_migration.quit_point import session_quit

    f = CACHE / f"{label}.pkl"
    if f.exists():
        d = pickle.load(open(f, "rb"))
        if d.get("version") == VERSION:
            return d
    s = next(x for x in config.load_sessions() if x["label"] == label)
    ep = epochs.epoch_of(label)
    cue = _load_cue(s["h5"])
    sr = float(cue["sample_rate_hz"])
    lk = _load_licks(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    if lk.get("lick_source") not in (None, "daq"):
        raise RuntimeError("must read DAQ contact: unset WIDEFIELD_LICK_SOURCE")
    cs = np.asarray(cue["cue_samples"], np.int64)
    cue_s = cs / sr
    daq = np.sort(np.asarray(lk["lick_samples"], float)) / sr
    codes = np.asarray(classify_cues_with_backup(s, cue))
    out = {"version": VERSION, "label": label, "animal": label[:4], "epoch": ep, "lp": True, "trials": None,
           "quit_daq": None, "quit_lp": None, "n_cues": int(cs.size)}
    try:
        lp = lp_licks.session_lp(label, cs, sr, MAX_RT)
        lp_t, _ = lp_licks.lp_lick_times(label)
    except lp_licks.LPUnavailable as e:
        out["lp"] = False
        out["why"] = str(e)
        lp = None
        lp_t = np.zeros(0)

    # ---- per-cue table
    j = np.searchsorted(daq, cue_s, side="right")
    first_daq = np.where(j < daq.size, daq[np.clip(j, 0, max(daq.size - 1, 0))] - cue_s, np.nan) if daq.size \
        else np.full(cue_s.size, np.nan)
    ss = np.asarray(cue["strobe_samples"])
    js = np.searchsorted(ss, cs, side="right") - 1
    st_ms = np.where(js >= 0, np.round(ss[np.clip(js, 0, None)] / sr * 1e3), np.nan)
    c_ms = np.round(cue_s * 1e3).astype(np.int64)
    daq_ms = np.sort(np.round(daq * 1e3).astype(np.int64))
    lp_ms = np.sort(np.round(lp_t * 1e3).astype(np.int64))
    keep = codes >= 0
    rows = []
    for k in np.flatnonzero(keep):
        r = {"k": int(k), "pos": CANON[POSITION_NAMES[int(codes[k])]], "t_min": (cue_s[k] - cue_s[0]) / 60.0,
             "first_daq": float(first_daq[k])}
        if lp is not None:
            r["lp_cov"] = bool(not lp["drop"][k])
            r["lp_cls"] = str(lp["cls"][k])
            r["lp_first"] = float(lp["first_s"][k]) if lp["attempted"][k] else np.nan
            r["lp_any35"] = bool(np.any((lp_t >= cue_s[k]) & (lp_t <= cue_s[k] + MAX_RT)))
            r["lp_any20"] = bool(np.any((lp_t >= cue_s[k]) & (lp_t <= cue_s[k] + QUIT_RESP)))
        for w in ENL_W:
            wm = int(round(w * 1e3))
            r0 = precue_window_start(c_ms[k], st_ms[k], daq_ms, wm, lickfree=True)
            tag = f"{w:g}"
            r[f"enl{tag}_kept"] = r0 is not None
            if r0 is not None and lp is not None:
                inw = lp_ms[(lp_ms >= r0) & (lp_ms < r0 + wm)]
                r[f"enl{tag}_lp"] = bool(inw.size)
                if inw.size:
                    jj = np.searchsorted(daq_ms, inw)
                    dprev = np.abs(inw - daq_ms[np.clip(jj - 1, 0, max(daq_ms.size - 1, 0))]) if daq_ms.size \
                        else np.full(inw.size, 1e9)
                    dnext = np.abs(daq_ms[np.clip(jj, 0, max(daq_ms.size - 1, 0))] - inw) if daq_ms.size \
                        else np.full(inw.size, 1e9)
                    r[f"enl{tag}_lponly"] = bool(np.any(np.minimum(dprev, dnext) > NEAR_DAQ_S * 1e3))
                else:
                    r[f"enl{tag}_lponly"] = False
        rows.append(r)
    T = pd.DataFrame(rows)
    T["daq35"] = (T.first_daq > 0) & (T.first_daq <= MAX_RT)
    T["daq20"] = (T.first_daq > 0) & (T.first_daq <= QUIT_RESP)
    # imaging gate, on the full behavioural trial list (order = cue index)
    pos_raw = np.array([{v: kk for kk, v in CANON.items()}[p] for p in T.pos])
    from wfield_local import trial_classes as _tc
    # a diagnostic OF the DAQ / LP gate rules themselves (not a selection under the run's gate; rebuild 11-03)
    with _tc.daq_quit_rule("gate_lp_check: compares the DAQ and LP gate rules"):
        T["ne_img"] = engagement_gate(T.k.to_numpy(), T.daq35.to_numpy(bool), pos_raw)
    if lp is not None:
        att = T.lp_cls.isin(lp_licks.ATTEMPT_CLASSES) & np.isfinite(T.lp_first)
        T["lp_att35"] = att & (T.lp_first <= MAX_RT)
        T["lp_att20"] = att & (T.lp_first <= QUIT_RESP)
        # LP-or-DAQ img gate (what an LP-based gate would call engaged)
        resp_lp = (T.daq35 | (T.lp_cov & T.lp_att35)).to_numpy(bool)
        with _tc.daq_quit_rule("gate_lp_check: compares the DAQ and LP gate rules"):
            T["ne_img_lp"] = engagement_gate(T.k.to_numpy(), resp_lp, pos_raw)

    # ---- the S28 quit, DAQ and LP rule
    tr = session_trials(s, QUIT_RESP)
    T["ne_quit"] = False
    if len(tr) >= 60:
        q = session_quit(s, tr)
        out["quit_daq"] = q
        out["n_quit_rows"] = len(tr)
        if q is not None and not q["censored"]:
            T["ne_quit"] = T.t_min * 60.0 >= q["quit_elapsed_s"] - 1e-6
        if lp is not None:
            m = T.set_index("k")
            tr_lp = []
            for r in tr:
                r2 = dict(r)
                kk = r["order"]
                if kk in m.index and bool(m.at[kk, "lp_cov"]) and bool(m.at[kk, "lp_att20"]):
                    r2["hit"] = 1
                tr_lp.append(r2)
            out["quit_lp"] = session_quit(s, tr_lp)
            out["n_flip_quit_rows"] = int(sum(a["hit"] != b["hit"] for a, b in zip(tr, tr_lp)))
    T.insert(0, "label", label)
    T.insert(1, "animal", label[:4])
    T.insert(2, "epoch", ep)
    out["trials"] = T
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def labels():
    from wfield_local import analysis_kit as ak
    return [lab for lab in ak.curated_labels(list(ANIMALS), order="sorted")]


def extract(jobs):
    from wfield_local import analysis_kit as ak, config
    assert abs(float(config.defaults()["decode"]["max_rt_s"]) - MAX_RT) < 1e-9, "decode.max_rt_s moved"
    labs = labels()
    print(f"{len(labs)} curated sessions (August-only pre)", flush=True)
    if jobs <= 1:
        res = [(lab, one(lab)) for lab in labs]
        fail = []
    else:
        res, fail = ak.fan_sessions(labs, one, jobs=min(jobs, 4))
    if fail:
        print("FAILURES:", fail)
    return res


# ============================================================================================== analysis
def _load_all():
    labs = labels()
    D = {}
    for lab in labs:
        f = CACHE / f"{lab}.pkl"
        if f.exists():
            D[lab] = pickle.load(open(f, "rb"))
    missing = [x for x in labs if x not in D]
    if missing:
        raise SystemExit(f"missing extractions: {missing}")
    return D


def _frac(num, den):
    return float(num) / float(den) if den else np.nan


def per_session(D):
    """One row per session x spout group with every Q1 / Q2 / Q4 count."""
    rows = []
    for lab, d in D.items():
        T = d["trials"]
        if not d["lp"]:
            continue
        T = T[T.lp_cov]
        for grp, sub in (("all", T), ("far_contra", T[T.pos == "far_contra"]),
                         ("other", T[T.pos != "far_contra"])) + tuple((p, T[T.pos == p]) for p in POS_ORDER):
            r = {"label": lab, "animal": d["animal"], "epoch": d["epoch"], "grp": grp, "n": len(sub)}
            for g in ("img", "quit"):
                ne = sub[f"ne_{g}"].to_numpy(bool)
                att = sub.lp_att35.to_numpy(bool)
                anyl = sub.lp_any35.to_numpy(bool)
                r[f"n_ne_{g}"] = int(ne.sum())
                r[f"n_ne_{g}_att"] = int((ne & att).sum())
                r[f"n_ne_{g}_any"] = int((ne & anyl).sum())
                r["n_att"] = int(att.sum())
                r[f"n_att_ne_{g}"] = int((att & ne).sum())
                r[f"f_ne_{g}_att"] = _frac((ne & att).sum(), ne.sum())          # Q1a
                r[f"f_ne_{g}_any"] = _frac((ne & anyl).sum(), ne.sum())
                r[f"f_att_out_{g}"] = _frac((att & ne).sum(), att.sum())        # Q1b
            # Q1 PRIMARY: the imaging `stopped` class proper = gated out AND no DAQ contact within max_rt (a DAQ hit
            # after the onset is `success`, not stopped); and the S28 quit period's no-contact trials
            miss = ~sub.daq35.to_numpy(bool)
            att = sub.lp_att35.to_numpy(bool)
            for g in ("img", "quit"):
                st = sub[f"ne_{g}"].to_numpy(bool) & miss
                r[f"n_stop_{g}"] = int(st.sum())
                r[f"n_stop_{g}_att"] = int((st & att).sum())
                r[f"f_stop_{g}_att"] = _frac((st & att).sum(), st.sum())
                r[f"n_ne_{g}_daqhit"] = int((sub[f"ne_{g}"].to_numpy(bool) & ~miss).sum())
            # Q2: working (img-engaged) DAQ misses with an LP attempt
            wk = ~sub.ne_img.to_numpy(bool)
            r["n_work_miss"] = int((wk & miss).sum())
            r["n_work_miss_att"] = int((wk & miss & att).sum())
            r["f_work_miss_att"] = _frac((wk & miss & att).sum(), (wk & miss).sum())
            r["f_work_att_of_trials"] = _frac((wk & miss & att).sum(), wk.sum())
            # LP-based img gate: how many trials move engaged
            r["n_ne_img_lp"] = int(sub.ne_img_lp.sum())
            # Q4
            for w in ENL_W:
                t = f"{w:g}"
                kept = sub[f"enl{t}_kept"].to_numpy(bool)
                r[f"n_enl{t}"] = int(kept.sum())
                r[f"f_enl{t}_lp"] = _frac(sub.loc[kept, f"enl{t}_lp"].astype(bool).sum(), kept.sum())
                r[f"f_enl{t}_lponly"] = _frac(sub.loc[kept, f"enl{t}_lponly"].astype(bool).sum(), kept.sum())
                # same within the decoder's two arms (DAQ engaged within max_rt vs no-lick)
                for arm, m in (("eng", sub.daq35.to_numpy(bool)), ("nolick", ~sub.daq35.to_numpy(bool))):
                    kk = kept & m
                    r[f"f_enl{t}_lp_{arm}"] = _frac(sub.loc[kk, f"enl{t}_lp"].astype(bool).sum(), kk.sum())
            rows.append(r)
    return pd.DataFrame(rows)


def quit_table(D):
    rows = []
    for lab, d in D.items():
        qd, ql = d.get("quit_daq"), d.get("quit_lp")
        if qd is None and ql is None:
            continue
        r = {"label": lab, "animal": d["animal"], "epoch": d["epoch"], "lp": d["lp"],
             "n_rows": d.get("n_quit_rows"), "n_flipped": d.get("n_flip_quit_rows")}
        for tag, q in (("daq", qd), ("lp", ql)):
            if q is None:
                r[f"{tag}_quit"] = np.nan
                r[f"{tag}_censored"] = np.nan
                r[f"{tag}_engaged_min"] = np.nan
                r[f"{tag}_quit_trial"] = np.nan
                continue
            r[f"{tag}_censored"] = int(q["censored"])
            r[f"{tag}_quit"] = int(not q["censored"])
            r[f"{tag}_engaged_min"] = q["time_engaged_s"] / 60.0
            r[f"{tag}_quit_trial"] = q["quit_trial"] if not q["censored"] else q["n_trials"]
            r["dur_min"] = q["session_dur_s"] / 60.0
        r["shift_min"] = r["lp_engaged_min"] - r["daq_engaged_min"]
        r["shift_trials"] = r["lp_quit_trial"] - r["daq_quit_trial"]
        r["quit_lost"] = int(r.get("daq_quit") == 1 and r.get("lp_quit") == 0)
        rows.append(r)
    return pd.DataFrame(rows)


def _by_animal(df, col, grp=None, agg="mean"):
    x = df if grp is None else df[df.grp == grp]
    return x.groupby(["animal", "epoch"])[col].agg(agg).unstack("epoch").reindex(columns=list(EPS))


def _boot(df, col, rng):
    """boot_delta per post epoch vs pre (animal-weighted, nested animal -> session)."""
    from wfield_local import analysis_kit as ak
    out = []
    pre = {a: df[(df.animal == a) & (df.epoch == "pre")][col].dropna().to_numpy(float) for a in ANIMALS}
    pre = {a: v for a, v in pre.items() if v.size}
    lv = {}
    for ep in EPS:
        x = df[df.epoch == ep]
        lv[ep] = {a: x[x.animal == a][col].dropna().to_numpy(float) for a in ANIMALS}
        lv[ep] = {a: v for a, v in lv[ep].items() if v.size}
        ci = ak.boot_ci(lv[ep], rng) if lv[ep] else None
        if ci is not None:
            out.append({"metric": col, "epoch": ep, "kind": "level", "est": ci.point, "lo": ci.lo, "hi": ci.hi,
                        "n_animals": ci.n_animals})
    for ep in EPS[1:]:
        post = lv[ep]
        common = {a: post[a] for a in post if a in pre}
        if len(common) < 2:
            continue
        iv, o = ak.boot_delta(common, {a: pre[a] for a in common}, rng, draws=True)
        if iv is None:
            continue
        out.append({"metric": col, "epoch": ep, "kind": "delta_vs_pre", "est": iv.point, "lo": iv.lo, "hi": iv.hi,
                    "p": ak.boot_p(o), "n_animals": iv.n_animals})
    return out


def analyse():
    D = _load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    no_lp = sorted(lab for lab, d in D.items() if not d["lp"])
    cov = pd.DataFrame([{"label": lab, "animal": d["animal"], "epoch": d["epoch"], "lp": d["lp"],
                         "n_trials": len(d["trials"]),
                         "n_lp_cov": int(d["trials"].lp_cov.sum()) if d["lp"] else 0}
                        for lab, d in D.items()])
    cov.to_csv(OUT / "coverage.csv", index=False)
    print(f"sessions: {len(D)}; without LP tables: {no_lp}")
    print("LP verdict coverage per animal x epoch (trials with LP verdict / trials):")
    print(cov.groupby(["animal", "epoch"])[["n_lp_cov", "n_trials"]].sum().unstack("epoch").to_string())

    S = per_session(D)
    S.to_csv(OUT / "per_session.csv", index=False)
    Q = quit_table(D)
    Q.to_csv(OUT / "quit_shift.csv", index=False)

    # pooled-count tables per animal x epoch x grp (counts summed, then fractions)
    cnt_cols = [c for c in S.columns if c.startswith("n_")]
    A = S.groupby(["animal", "epoch", "grp"])[cnt_cols].sum().reset_index()
    for g in ("img", "quit"):
        A[f"f_ne_{g}_att"] = A[f"n_ne_{g}_att"] / A[f"n_ne_{g}"].replace(0, np.nan)
        A[f"f_ne_{g}_any"] = A[f"n_ne_{g}_any"] / A[f"n_ne_{g}"].replace(0, np.nan)
        A[f"f_att_out_{g}"] = A[f"n_att_ne_{g}"] / A["n_att"].replace(0, np.nan)
    A["f_work_miss_att"] = A.n_work_miss_att / A.n_work_miss.replace(0, np.nan)
    for g in ("img", "quit"):
        A[f"f_stop_{g}_att"] = A[f"n_stop_{g}_att"] / A[f"n_stop_{g}"].replace(0, np.nan)
    A.to_csv(OUT / "per_animal_epoch_counts.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    for g in ("img", "quit"):
        print(f"\n=== Q1 PRIMARY  {g}-gate STOPPED trials (gated out AND no DAQ contact <= 3.5 s) with an LP attempt")
        for grp in ("all", "far_contra", "other"):
            x = A[A.grp == grp].pivot(index="animal", columns="epoch", values=f"f_stop_{g}_att").reindex(columns=list(EPS))
            n = A[A.grp == grp].pivot(index="animal", columns="epoch", values=f"n_stop_{g}").reindex(columns=list(EPS))
            na = A[A.grp == grp].pivot(index="animal", columns="epoch", values=f"n_stop_{g}_att").reindex(columns=list(EPS))
            print(f"-- {grp}\n{x.to_string()}\n   n stopped / n stopped with attempt:\n{n.to_string()}\n{na.to_string()}")
        print("   gated-out trials WITH a DAQ hit (success class after onset), all spouts:")
        print(A[A.grp == "all"].pivot(index="animal", columns="epoch", values=f"n_ne_{g}_daqhit")
              .reindex(columns=list(EPS)).to_string())
    print("\n=== Q1a  fraction of IMG-gate NOT-engaged trials with an LP attempt (<=3.5 s), per animal (trial-pooled)")
    for grp in ("all", "far_contra", "other"):
        x = A[A.grp == grp].pivot(index="animal", columns="epoch", values="f_ne_img_att").reindex(columns=list(EPS))
        n = A[A.grp == grp].pivot(index="animal", columns="epoch", values="n_ne_img").reindex(columns=list(EPS))
        print(f"-- {grp}\n{x.to_string()}\n   n gated-out trials:\n{n.to_string()}")
    print("\n=== Q1a' same for the QUIT gate (trials at/after the S28 quit onset)")
    for grp in ("all", "far_contra", "other"):
        x = A[A.grp == grp].pivot(index="animal", columns="epoch", values="f_ne_quit_att").reindex(columns=list(EPS))
        print(f"-- {grp}\n{x.to_string()}")
    print("\n=== Q1b  fraction of LP-attempt trials that fall OUTSIDE the img gate")
    for grp in ("all", "far_contra", "other"):
        x = A[A.grp == grp].pivot(index="animal", columns="epoch", values="f_att_out_img").reindex(columns=list(EPS))
        print(f"-- {grp}\n{x.to_string()}")
    print("\n=== Q2  working-period DAQ misses holding an LP attempt, per spout (trial-pooled)")
    for p in POS_ORDER:
        x = A[A.grp == p].pivot(index="animal", columns="epoch", values="f_work_miss_att").reindex(columns=list(EPS))
        print(f"-- {p}\n{x.to_string()}")
    print("\n=== Q3  quit under LP rule, per animal x epoch")
    if len(Q):
        qa = Q.groupby(["animal", "epoch"]).agg(n=("label", "size"), daq_quits=("daq_quit", "sum"),
                                                 lp_quits=("lp_quit", "sum"), lost=("quit_lost", "sum"),
                                                 shift_min_mean=("shift_min", "mean"),
                                                 shift_min_max=("shift_min", "max"),
                                                 shift_tr_mean=("shift_trials", "mean")).reset_index()
        qa.to_csv(OUT / "quit_shift_per_animal.csv", index=False)
        print(qa.to_string(index=False))
        print("sessions with a nonzero shift:")
        print(Q[(Q.shift_min.abs() > 1e-9) | (Q.quit_lost > 0)][
            ["label", "epoch", "daq_quit", "lp_quit", "daq_engaged_min", "lp_engaged_min", "shift_min",
             "shift_trials", "n_flipped"]].to_string(index=False))
    print("\n=== Q4  DAQ-clean pre-cue windows holding an LP lick (any / LP-only, >100 ms from a contact)")
    for w in ENL_W:
        t = f"{w:g}"
        for col in (f"f_enl{t}_lp", f"f_enl{t}_lponly"):
            for grp in ("all", "far_contra", "other"):
                x = _by_animal(S, col, grp)
                print(f"-- {col} {grp} (session mean)\n{x.to_string()}")

    # ---- pooled: boot_ci levels + boot_delta vs pre (session values; animal-weighted deltas)
    rng = np.random.default_rng(SEED)
    B = []
    metrics = [("f_stop_img_att", "all"), ("f_stop_img_att", "far_contra"), ("f_stop_img_att", "other"),
               ("f_stop_quit_att", "all"), ("f_ne_img_att", "all"), ("f_ne_img_att", "far_contra"), ("f_ne_img_att", "other"),
               ("f_ne_quit_att", "all"), ("f_att_out_img", "all"), ("f_att_out_img", "far_contra"),
               ("f_work_miss_att", "far_contra"), ("f_work_miss_att", "other"),
               ("f_enl2_lp", "all"), ("f_enl1_lp", "all"), ("f_enl2_lponly", "all"), ("f_enl1_lponly", "all"),
               ("f_enl2_lp", "far_contra"), ("f_enl1_lp", "far_contra")]
    for col, grp in metrics:
        for r in _boot(S[S.grp == grp], col, rng):
            r["grp"] = grp
            B.append(r)
    for col in ("shift_min", "shift_trials"):
        for r in _boot(Q[Q.lp == True], col, rng):                          # noqa: E712
            r["grp"] = "session"
            B.append(r)
    B = pd.DataFrame(B)
    B.to_csv(OUT / "pooled_boot.csv", index=False)
    print("\n=== POOLED (boot_ci levels, boot_delta vs pre; nested animal -> session, n = 4)")
    print(B.to_string(index=False))

    # figdata + figure
    fd = {"S": S, "Q": Q, "B": B, "A": A, "thresholds": dict(gated_out=THR_GATED_OUT, quit_min=THR_QUIT_MIN,
                                                               enl=THR_ENL)}
    (OUT / "figdata").mkdir(exist_ok=True)
    pickle.dump(fd, open(OUT / "figdata" / "gate_lp_check.pkl", "wb"))
    plot(fd)


def plot(fd=None):
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    if fd is None:
        fd = pickle.load(open(OUT / "figdata" / "gate_lp_check.pkl", "rb"))
    S, Q, B = fd["S"], fd["Q"], fd["B"]
    MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    CO = {"pre": "#555555", "acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4"}
    panels = [("f_stop_img_att", "all", "STOPPED trials (gated out,\nno contact) with an LP attempt", THR_GATED_OUT),
              ("f_stop_img_att", "far_contra", "STOPPED far-contra trials\nwith an LP attempt", THR_GATED_OUT),
              ("f_work_miss_att", "far_contra", "working far-contra DAQ misses\nwith an LP attempt", None),
              ("f_enl2_lp", "all", "DAQ-clean 2 s ENL windows\nwith an LP lick", THR_ENL),
              ("f_enl1_lp", "all", "DAQ-clean 1 s ENL windows\nwith an LP lick", THR_ENL),
              ("shift_min", None, "quit shift under LP rule (min)", THR_QUIT_MIN)]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.6))
    rng = np.random.default_rng(0)
    for i, (ax, (col, grp, title, thr)) in enumerate(zip(axs, panels)):
        df = (Q[Q.lp == True] if grp is None else S[S.grp == grp])                # noqa: E712
        for xi, ep in enumerate(EPS):
            for a in ANIMALS:
                v = df[(df.epoch == ep) & (df.animal == a)][col].dropna().to_numpy(float)
                ax.scatter(xi + rng.uniform(-0.18, 0.18, v.size), v, marker=MK[a], facecolors="none",
                           edgecolors=CO[ep], s=18, lw=0.9)
            lv = B[(B.metric == col) & (B.grp == (grp or "session")) & (B.epoch == ep) & (B.kind == "level")]
            if len(lv):
                ax.errorbar(xi + 0.3, lv.est.iloc[0], yerr=[[lv.est.iloc[0] - lv.lo.iloc[0]],
                                                             [lv.hi.iloc[0] - lv.est.iloc[0]]],
                            fmt="_", color="k", ms=10, lw=1.2)
        if thr is not None:
            ax.axhline(thr, color="#8a2be2", ls=":", lw=1)
        ax.set_xticks(range(len(EPS)))
        ax.set_xticklabels(EPS, rotation=30, fontsize=8)
        ax.set_title(f"{chr(65 + i)}. {title}", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        for xi, ep in enumerate(EPS[1:], start=1):
            d = B[(B.metric == col) & (B.grp == (grp or "session")) & (B.epoch == ep) & (B.kind == "delta_vs_pre")]
            if len(d) and (d.lo.iloc[0] > 0 or d.hi.iloc[0] < 0):        # boot_delta CI excludes 0
                sig_marks.star_top(ax, xi, "*")
    for a in ANIMALS:
        axs[0].scatter([], [], marker=MK[a], facecolors="none", edgecolors="k", label=a)
    axs[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle("DAQ-contact gate vs LP lick attempts (PS92-95, August pre; open markers = sessions, shape = "
                 "animal, colour = epoch; bar = boot_ci; stars = boot_delta vs pre; dotted = decision threshold)",
                 fontsize=8)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "gate_lp_check.png", dpi=160)
    print("figure ->", OUT / "figures" / "gate_lp_check.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("extract", "analyse", "plot", "one"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--label")
    a = ap.parse_args(argv)
    if a.stage == "one":
        d = one(a.label)
        T = d["trials"]
        print({k: v for k, v in d.items() if k != "trials"})
        print(T.head())
        print(T.describe(include="all").T.to_string())
    elif a.stage == "extract":
        extract(a.jobs)
    elif a.stage == "analyse":
        analyse()
    else:
        plot()


if __name__ == "__main__":
    main()
