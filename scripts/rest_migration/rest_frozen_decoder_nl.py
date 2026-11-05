"""Fig 12.1 (imaging page) on LICK-FREE rest: the frozen pre-stroke rest decoder re-run on RESTnl periods.

Priya 11-01: "rest frames used as DATA: rest decoder 12.1 re-run on lick-free rest first". A NEW sensitivity
version beside `rest_frozen_decoder` (the page's 12.1, `--perm 200 --refit --match-train --match-duration
--match-lickgap`, restdock05, all pre); that module's default output is unchanged (its `_collect` gained two
opt-in keywords, pinned in tests/test_rest_frozen_decoder_nl.py).

WHAT CHANGES, AND ONLY THIS
  REST MASK  `--mask restnl`: the restdock05 rest sample mask (docked, not running, DAQ contact licks excluded
             [0.5, 1.0] s, min_rest_s) MINUS every DAQ lick onset (`restnl_mask.daq_lick_onsets`) and EVERY cam4
             Lightning-Pose lick of `licks_all.csv` (`daq_on_s`), each widened [0.5 s before, 2.0 s after], the
             min_rest_s floor re-applied after each intersection -- exactly `restnl_mask.lickfree_mask` (the map
             baseline's mask), applied to the decoder's own mask file. A rest PERIOD is then one lick-free run of that
             mask, kept only when bracketed by two engaged working trials at the same position (the decoder's own rule).
             Sessions with no licks_all.csv (PS93_0813) get no RESTnl periods; sessions whose RESTnl map baseline fails
             rule wr (>= 50 frames at >= 4 of 6 positions) are dropped (none do).
  LICK GAP   the lick-gap matching measures the gap to the nearest lick of DAQ contact UNION cam4 LP (the licks the
             mask removed), not DAQ contact alone.
  PRE        `--pre aug` (PRIMARY): pre-stroke = August sessions only (>= 0801, the cohort rule since 10-18); `--pre
             all` reproduces the PRE-REBUILD page's pre set (June + August), used to validate this driver against
             that page CSV. In the cohort-1 rebuild page 12.1 is August-only (the primary tag); `--compare` reads it
             there (`page_sessions`), while this block itself runs with PRE_FROM unset (review 8B N6).
  FROZEN     every pre-stroke model (full + one leave-one-session-out model per pre session) goes through
             `frozen_models.load_or_fit` (CLAUDE.md rule 10; kind `rest_frozen`, align carries mask / bins / matching),
             never refit in memory per render. The page's 12.1 did refit in memory; the fitted models are identical
             (same data, same lbfgs fit), the difference is that these are stored references. The size-MATCHED and
             within-session REFIT arms are controls fitted per scored session (unchanged from the page script).

Everything else (joint LocaNMF basis, 4 bins per period, StandardScaler + logistic C = 0.1, balanced accuracy, the
blockperm / shift / trial nulls with predictions fixed, 200 permutations, duration and lick-gap matching to the central
80 % of PRE, minimum 40 periods / 5 per class) is `rest_frozen_decoder`'s, imported, not copied.

STAGES (each writes what the next reads, so figures redraw without data):
  collect  per-session features -> `<cache>/<mask>/<label>.npz` (4 workers, `fan_sessions`)
  counts   periods / frames per animal x epoch x position BEFORE fitting -> `<out>/data/*_counts*.csv`
  fit      frozen models + scoring -> `<out>/data/<stem>_sessions.csv`, confusion counts
  plot     per-animal table, nested bootstrap (analysis_kit), figures   (`--plot-only` = this stage alone)

    python scripts/rest_migration/rest_frozen_decoder_nl.py --mask restnl --pre aug [--plot-only]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from scripts.rest_migration import rest_frozen_decoder as rf  # noqa: E402

CHANCE = rf.CHANCE
CODES = [0, 1, 2, 3, 4, 5]
# R1 (fourth review, 11-03): every root through the resolver and the run mode -- the output folder carries the run tag
# (`run_mode.subdir`), the feature cache sits under THIS run's figures_working (inside the rebuild root under one) and its
# key carries `run_mode.cache_tag` (hemo / lnmfproj / selection / csig), and the page-12.1 sessions table is read from
# the routed 15f folder (`rest_frozen_decoder._epoch_dir`), never the default copy
from wfield_local import config as _config  # noqa: E402
from wfield_local import run_mode as _run_mode  # noqa: E402

G = Path(_config.resolver().root("cohort_out")) / "grant_figures"
OUT_ROOT = _run_mode.subdir(G / "rest_frozen_decoder_restnl")
CACHE_ROOT = Path(_config.resolver().root("figures_working")) / "rest_decoder_nl_cache"
PAGE_SESSIONS = rf._epoch_dir() / "data" / "epoch_15f_rest_frozen_restdock05_durmatched_gapmatched_sessions.csv"
_PAGE_CSV = "epoch_15f_rest_frozen_restdock05_durmatched_gapmatched_sessions.csv"


def page_sessions():
    """Page 12.1's 15f sessions table of the PRIMARY run (review 7B N2): in a tagged run this block keeps its own June +
    August pre (PRE_FROM unset), so its own tag's 15f folder is not the page's; the primary tag's (August-only pre,
    `run_mode.PRIMARY_PRE_FROM`) is. The default run reads the default folder, as before."""
    from wfield_local import lp_licks
    return _run_mode.primary_output_dir(Path(_config.resolver().root("cohort_out")) / "grant_figures" / "epoch") / \
        "data" / _PAGE_CSV if lp_licks.output_tag() else PAGE_SESSIONS


def page_pre_text() -> str:
    """The page's pre set: August-only in a tagged (rebuild) run, June + August in the default run."""
    from wfield_local import lp_licks
    return "Aug pre" if lp_licks.output_tag() else "June+Aug pre"
EPOCHS = ("pre", "acute", "subacute", "chronic")
MASK_TEXT = {
    "restnl": "RESTnl rest (lick-free: DAQ contact + cam4 LP video licks removed [0.5 s, 2.0 s])",
    "rest": "REST rest (restdock05; DAQ contact licks removed [0.5 s, 1.0 s])",
}
LICK_TEXT = {"restnl": "licks: DAQ contact + cam4 LP video", "rest": "licks: DAQ spout contact"}
PRE_TEXT = {"aug": "August-only pre", "all": "June + August pre"}
# THE PAGE RUN'S FLAGS (nightly_figs: --perm 200 --refit --match-train --match-duration --match-lickgap)
PERM, BINS, MIN_PERIODS, MIN_PER_CLASS, PCT, LICK_FAR_S = 200, 4, 40, 5, 10.0, 3.0
#: FAIRNESS GATE (11-02): every position needs >= this many matched PRE training periods, or the animal is not fitted
#: (unless --allow-thin). Measured: RESTnl Aug-pre PS93 far contra has 9 (REST 348), so a frozen model barely predicts
#: that class, pre test sets hardly contain it and post test sets do -- a downward bias at the stroke position.
MIN_TRAIN_PER_POS = 30
NULLS = ("blockperm", "shift", "trial")


# ------------------------------------------------------------------------------------------------ the RESTnl mask
def lickfree_rest(session, rest, rate):
    """``(lick-free rest sample mask, DAQ + cam4 lick onsets in samples)`` -- `restnl_mask.lickfree_mask` applied to
    the decoder's rest mask (the same restdock05 quiet_sample.npy). Raises FileNotFoundError without licks_all.csv."""
    import pandas as pd

    from wfield_local import config
    from wfield_local import restnl_mask as rn
    lp = rn.licks_all_path(session["label"])
    if not lp.exists():
        raise FileNotFoundError(f"no licks_all.csv ({lp})")
    n = rest.shape[0]
    nb, na = int(rn.BUF_S[0] * rate), int(rn.BUF_S[1] * rate)
    minrun = int(float(config.defaults()["segmentation"]["rest"]["min_rest_s"]) * rate)
    daq_on = rn.daq_lick_onsets(session, rate)
    cam4 = pd.read_csv(lp, usecols=["daq_on_s"]).daq_on_s.to_numpy(float)
    cam4 = np.round(cam4[np.isfinite(cam4)] * rate).astype(np.int64)
    m = rn.floor_runs(rest & ~rn.widen(daq_on, n, nb, na), minrun)
    m = rn.floor_runs(m & ~rn.widen(cam4, n, nb, na), minrun)
    return m, np.unique(np.concatenate([daq_on, cam4]))


# ------------------------------------------------------------------------------------------------ collect
def _code_key(mask):
    from wfield_local import restnl_mask as rn
    src = inspect.getsource(rf._collect) + inspect.getsource(rf._frame_samples)
    if mask == "restnl":
        src += inspect.getsource(lickfree_rest) + repr((rn.BUF_S, rn.MIN_FR, rn.MIN_POS_WR))
    tag = _run_mode.cache_tag()                       # R1: another hemo / projection / selection never meets this key
    return hashlib.sha1((f"{src}|{BINS}|gate" + (f"|{tag}" if tag else "")).encode()).hexdigest()[:12]   # default key kept


_BASIS = {}


def _basis(an):
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    if an not in _BASIS:
        _BASIS.clear()
        _BASIS[an] = joint_locanmf.load(an, sessions=SESSIONS)
    return _BASIS[an]


def _collect_job(item):
    """One session -> cache npz (module-level for spawn). Returns a small dict."""
    label, mask = item
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    fp = CACHE_ROOT / mask / f"{label}__{_code_key(mask)}.npz"
    if fp.exists():
        return {"label": label, "cached": True}
    s = next(x for x in SESSIONS if x["label"] == label)
    t0 = time.perf_counter()
    basis = _basis(label.split("_")[0])
    got, why = rf._collect(s, basis, BINS, gate=True, rest_mask=lickfree_rest if mask == "restnl" else None,
                           period_frames=True)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(f".{os.getpid()}.tmp.npz")
    if got is None:
        np.savez(tmp, ok=False, why=str(why), basis_id=str(basis.basis_id))
    else:
        X, y, g, dur, gap, lick_hz, nfr = got
        np.savez(tmp, ok=True, why="", X=X, y=y, g=g, dur=dur, gap=gap, lick_hz=lick_hz, nfr=nfr,
                 basis_id=str(basis.basis_id))
    os.replace(tmp, fp)
    return {"label": label, "cached": False, "ok": got is not None, "why": why, "s": round(time.perf_counter() - t0, 1)}


def wanted_labels():
    """The page script's session set (pre + post of the curated cohort-1 set), in SESSIONS order."""
    from wfield_local import config
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    if os.environ.get("WIDEFIELD_PRE_FROM"):
        raise SystemExit("collect with WIDEFIELD_PRE_FROM unset (the pre set is chosen at fit time)")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    return [s["label"] for s in SESSIONS if s["label"] in want and s.get("h5")]


def collect(mask, jobs=4):
    from wfield_local import analysis_kit as ak
    from wfield_local import parallel
    parallel.pin_blas()
    labs = wanted_labels()
    t0 = time.time()
    res, fails = ak.fan_sessions([(x, mask) for x in labs], _collect_job, jobs=min(int(jobs), 4),
                                 key=lambda it: it[0], label=f"collect-{mask}")
    print(f"[collect {mask}] {len(res)} sessions, {len(fails)} failures, {(time.time() - t0) / 60:.1f} min", flush=True)
    for it, err in fails:
        print("   FAIL", it, str(err)[:200])
    return labs


def load_collected(mask, labs):
    """``{label: dict}`` in `labs` order; failures carry ``why``."""
    key = _code_key(mask)
    out = {}
    for lab in labs:
        fp = CACHE_ROOT / mask / f"{lab}__{key}.npz"
        if not fp.exists():
            out[lab] = {"ok": False, "why": f"{lab}: not collected"}
            continue
        z = np.load(fp, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["ok"] = bool(d["ok"])
        d["why"] = str(d["why"])
        out[lab] = d
    return out


# ------------------------------------------------------------------------------------------------ counts
def wr_status(label):
    """``(ok_wr or None when no RESTnl baseline cache, nfr_pos)`` from the map baseline cache (read only)."""
    from wfield_local import restnl_mask as rn
    try:
        fp = rn.cache_path(label)
    except StopIteration:
        return None, None
    if not fp.exists():
        return None, None
    z = np.load(fp)
    return bool(z["ok_wr"]), z["nfr_pos"].tolist()


def counts_table(mask, data, out, stem):
    """Periods / frames per session x position (all decoder-valid periods, before matching), and the animal x epoch
    x position sums. Written before any fit."""
    from wfield_local import epochs
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    rows = []
    for lab, d in data.items():
        an, ep = lab.split("_")[0], epochs.epoch_of(lab)
        base = {"animal": an, "label": lab, "epoch": ep, "mask": mask}
        if not d["ok"]:
            rows.append({**base, "position": "ALL", "n_periods": 0, "n_frames": 0, "median_dur_s": np.nan,
                         "note": d["why"]})
            continue
        y, nfr, dur = d["y"], d["nfr"], d["dur"]
        for c in CODES:
            m = y == c
            rows.append({**base, "position": POSITION_NAMES.get(c, str(c)), "n_periods": int(m.sum()),
                         "n_frames": int(nfr[m].sum()),
                         "median_dur_s": float(np.median(dur[m])) if m.any() else np.nan, "note": ""})
    p = out / "data" / f"{stem}_counts_by_session.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows, p


# ------------------------------------------------------------------------------------------------ fit
def _pre_ok(lab, pre):
    return pre == "all" or lab.split("_")[1] >= "0801"


def frozen_payload(an, basis_id, mask, pre, data_pre, windows):
    """Full + LOSO pre-stroke models through `frozen_models.load_or_fit`. ``data_pre`` = {label: (X, y, g, dur, gap)}
    for the animal's usable pre sessions. Returns (payload, status)."""
    from wfield_local import frozen_models as fm
    lo_s, hi_s, glo, ghi = windows
    labs = sorted(data_pre)
    nfeat = int(next(iter(data_pre.values()))[0].shape[1])
    spec = fm.make_spec(an, "rest_frozen", align=f"rest-{mask}-b{BINS}-dur{PCT:g}-gap{PCT:g}-gate",
                        source=f"joint_{basis_id}", train_labels=labs, basis_id=basis_id, n_features=nfeat,
                        post_s=0.0, zscore=True, alpha=0.1,
                        train_data=rf.rest_train_data({k: (data_pre[k][0], data_pre[k][1], data_pre[k][3],
                                                           data_pre[k][4]) for k in labs}, windows))

    def pool(keep):
        X = np.concatenate([data_pre[k][0] for k in keep])
        y = np.concatenate([data_pre[k][1] for k in keep])
        dt = np.concatenate([data_pre[k][3] for k in keep])
        gt = np.concatenate([data_pre[k][4] for k in keep])
        m = (dt >= lo_s) & (dt <= hi_s)
        m2 = (gt[m] >= glo) & (gt[m] <= ghi)
        return X[m][m2], y[m][m2]

    def fit():
        res = {"windows": tuple(float(v) for v in windows), "full": None, "loso": {}}
        X, y = pool(labs)
        if rf._usable(y, MIN_PERIODS, MIN_PER_CLASS)[0]:
            res["full"] = rf._fit_frozen(X, y)
        for k in labs:
            Xk, yk = pool([x for x in labs if x != k])
            res["loso"][k] = rf._fit_frozen(Xk, yk) if rf._usable(yk, MIN_PERIODS, MIN_PER_CLASS)[0] else None
        return res

    return fm.load_or_fit(spec, fit, meta={"mask": mask, "pre": pre, "windows": list(map(float, windows)),
                                           "script": "scripts/rest_migration/rest_frozen_decoder_nl.py"})


def fit_and_score(mask, pre, data, out, stem, allow_thin=False):
    """The page script's scoring loop (same rng sequence, same nulls, same controls) with frozen stored models."""
    import hashlib as _h

    from wfield_local import epochs
    rng = np.random.default_rng(0)
    per_session, skipped, frozen_log = [], [], []
    conf = {e: np.zeros((6, 6), np.int64) for e in EPOCHS}
    cidx = {c: i for i, c in enumerate(CODES)}
    animals = sorted({lab.split("_")[0] for lab in data})
    for an in animals:
        dd = {}
        for lab, d in data.items():
            if not lab.startswith(an + "_"):
                continue
            ep = epochs.epoch_of(lab)
            if ep is None:
                skipped.append(f"{lab}: no epoch")
                continue
            if ep == "pre" and not _pre_ok(lab, pre):
                continue
            if mask == "restnl":
                ok_wr, _n = wr_status(lab)
                if ok_wr is False:
                    skipped.append(f"{lab}: RESTnl rule wr fails")
                    continue
            if not d["ok"]:
                skipped.append(d["why"])
                continue
            ok, bad = rf._usable(d["y"], MIN_PERIODS, MIN_PER_CLASS)
            if not ok:
                skipped.append(f"{lab}: {bad}")
                continue
            dd[lab] = (ep, d["X"], d["y"], d["g"], d["dur"], d["gap"], float(d["lick_hz"]))
        pre_labs = [k for k, v in dd.items() if v[0] == "pre"]
        if len(pre_labs) < 2:
            skipped.append(f"{an}: {len(pre_labs)} usable pre sessions")
            continue
        pre_dur = np.concatenate([dd[k][4] for k in pre_labs])
        lo_s, hi_s = float(np.percentile(pre_dur, PCT)), float(np.percentile(pre_dur, 100 - PCT))
        pre_gap = np.concatenate([dd[k][5] for k in pre_labs])
        pre_gap = pre_gap[np.isfinite(pre_gap)]
        glo, ghi = float(np.percentile(pre_gap, PCT)), float(np.percentile(pre_gap, 100 - PCT))
        _y = np.concatenate([dd[k][2][(dd[k][4] >= lo_s) & (dd[k][4] <= hi_s) & (dd[k][5] >= glo) & (dd[k][5] <= ghi)]
                             for k in pre_labs])
        _per = np.bincount(_y, minlength=6)
        if _per.min() < MIN_TRAIN_PER_POS and not allow_thin:
            print(f"!! {an}: matched pre training periods per position {_per.tolist()} < {MIN_TRAIN_PER_POS} -- "
                  "NOT FITTED (fairness gate; --allow-thin overrides)", flush=True)
            skipped.append(f"{an}: thin pre training pool {_per.tolist()}")
            continue
        basis_id = str(data[pre_labs[0]]["basis_id"])
        payload, status = frozen_payload(an, basis_id, mask, pre,
                                         {k: (dd[k][1], dd[k][2], dd[k][3], dd[k][4], dd[k][5]) for k in pre_labs},
                                         (lo_s, hi_s, glo, ghi))
        frozen_log.append({"animal": an, "status": status, "n_pre": len(pre_labs), "dur_lo": lo_s, "dur_hi": hi_s,
                           "gap_lo": glo, "gap_hi": ghi, "pre_labels": " ".join(sorted(pre_labs))})
        print(f"{an}: {len(pre_labs)} pre, frozen {status}; duration [{lo_s:.2f}, {hi_s:.2f}] s, "
              f"lick gap [{glo:.2f}, {ghi:.2f}] s", flush=True)
        for lab, (ep, X, y, g, dur, gap, lick_hz) in dd.items():
            train = rf.training_pool(pre_labs, lab)
            Xt = np.concatenate([dd[k][1] for k in train])
            yt = np.concatenate([dd[k][2] for k in train])
            gt_pool = np.concatenate([np.asarray(dd[k][3], np.int64) + 1_000_000 * (i + 1)
                                      for i, k in enumerate(train)])
            dt = np.concatenate([dd[k][4] for k in train])
            m_tr = (dt >= lo_s) & (dt <= hi_s)
            Xt, yt, gt_pool = Xt[m_tr], yt[m_tr], gt_pool[m_tr]
            keep_test = (dur >= lo_s) & (dur <= hi_s)
            gtt = np.concatenate([dd[k][5] for k in train])[m_tr]
            m_g = (gtt >= glo) & (gtt <= ghi)
            Xt, yt, gt_pool = Xt[m_g], yt[m_g], gt_pool[m_g]
            keep_test &= (gap >= glo) & (gap <= ghi)
            ok, bad = rf._usable(yt, MIN_PERIODS, MIN_PER_CLASS)
            if not ok:
                skipped.append(f"{lab}: training pool {bad}")
                continue
            m_test = np.isin(y, np.unique(yt)) & keep_test
            if m_test.sum() < MIN_PERIODS:
                skipped.append(f"{lab}: only {int(m_test.sum())} periods in trained classes")
                continue
            model = payload["loso"].get(lab) if ep == "pre" else payload["full"]
            if model is None:
                skipped.append(f"{lab}: no frozen model")
                continue
            pred = model.predict(X[m_test])
            acc, ncls = rf._balanced_accuracy(y[m_test], pred)
            chance = 1.0 / ncls
            yt_test, gt_test = y[m_test], g[m_test]
            for t_c, p_c in zip(yt_test, pred):
                if int(t_c) in cidx and int(p_c) in cidx:
                    conf[ep][cidx[int(t_c)], cidx[int(p_c)]] += 1
            null_stats = {}
            for kind in NULLS:
                draws = [rf._balanced_accuracy(rf.null_labels(yt_test, gt_test, rng, kind), pred)[0]
                         for _ in range(PERM)]
                null_stats[kind] = (float(np.mean(draws)), (1 + sum(1 for x in draws if x >= acc)) / (1 + len(draws)),
                                    float(np.percentile(draws, 95)))
            nm, p, n95 = null_stats["blockperm"]
            rfp = rf.refit_predictions(X[m_test], y[m_test], g[m_test])
            r_acc = rf._balanced_accuracy(y[m_test], rfp)[0] if rfp is not None else float("nan")
            k = min(5, int(np.unique(g[m_test]).size))
            n_target = round(int(m_test.sum()) * max(k - 1, 1) / max(k, 1))
            seed = int(_h.sha1(lab.encode()).hexdigest()[:8], 16)
            got = rf.matched_frozen(Xt, yt, gt_pool, n_target, np.random.default_rng(seed))
            m_acc, m_used = (float("nan"), 0)
            if got is not None:
                m_acc = rf._balanced_accuracy(y[m_test], got[0].predict(X[m_test]))[0]
                m_used = got[1]
            gp = gap[m_test]
            fin = np.isfinite(gp)
            acc_near = acc_far = acc_veryfar = float("nan")
            n_veryfar = 0
            if fin.sum() >= 2 * MIN_PER_CLASS:
                cut = float(np.median(gp[fin]))
                near, far = fin & (gp <= cut), fin & (gp > cut)
                if near.sum() >= MIN_PERIODS // 2 and far.sum() >= MIN_PERIODS // 2:
                    acc_near = rf._balanced_accuracy(yt_test[near], pred[near])[0]
                    acc_far = rf._balanced_accuracy(yt_test[far], pred[far])[0]
                vf = fin & (gp >= LICK_FAR_S)
                n_veryfar = int(vf.sum())
                if n_veryfar >= MIN_PERIODS // 2 and len(np.unique(yt_test[vf])) >= 3:
                    acc_veryfar = rf._balanced_accuracy(yt_test[vf], pred[vf])[0]
            per_session.append({"animal": an, "label": lab, "epoch": ep, "n": int(m_test.sum()),
                                "n_classes": ncls, "acc": acc, "null": nm, "p": p,
                                "above_chance_frac": (acc - chance) / (1 - chance),
                                "refit_acc": r_acc, "refit_above_chance": (r_acc - chance) / (1 - chance),
                                "gap_refit_minus_frozen": r_acc - acc,
                                "matched_frozen_acc": m_acc, "matched_n_train": m_used,
                                "matched_above_chance": (m_acc - chance) / (1 - chance),
                                "gap_refit_minus_matched": r_acc - m_acc,
                                "n_train_full": len(yt), "median_dur_s": float(np.median(dur)),
                                "median_dur_scored_s": float(np.median(dur[m_test])), "lick_hz": lick_hz,
                                "median_lick_gap_s": float(np.median(gp[fin])) if fin.any() else float("nan"),
                                "acc_near_lick": acc_near, "acc_far_lick": acc_far,
                                "acc_veryfar_lick": acc_veryfar, "n_veryfar": n_veryfar,
                                "null_kind": "blockperm", "null_p95": n95,
                                **{f"null_{k_}": v[0] for k_, v in null_stats.items()},
                                **{f"p_{k_}": v[1] for k_, v in null_stats.items()},
                                "frozen_status": status})
            print(f"  == {lab} [{ep}] frozen acc {acc:.3f} (null {nm:.3f}, p={p:.3f}, n={int(m_test.sum())}), "
                  f"refit {r_acc:.3f}", flush=True)
    dp = out / "data"
    dp.mkdir(parents=True, exist_ok=True)
    _write(dp / f"{stem}_sessions.csv", per_session)
    _write(dp / f"{stem}_frozen.csv", frozen_log)
    _write(dp / f"{stem}_skipped.csv", [{"skipped": x} for x in skipped] or [{"skipped": ""}])
    with open(dp / f"{stem}_confusion.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "true", "pred", "count"])
        for e in EPOCHS:
            for i, tc in enumerate(CODES):
                for j, pc in enumerate(CODES):
                    w.writerow([e, tc, pc, int(conf[e][i, j])])
    print(f"[fit] {len(per_session)} scored, {len(skipped)} skipped", flush=True)
    return per_session


def _write(p, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ tables + stats
def per_animal(per_session):
    """{(animal, epoch): dict} -- mean acc, null, above-chance frac, retained vs own pre, n sessions, n > null."""
    out = {}
    for an in sorted({r["animal"] for r in per_session}):
        pf = [r["above_chance_frac"] for r in per_session if r["animal"] == an and r["epoch"] == "pre"]
        for e in EPOCHS:
            v = [r for r in per_session if r["animal"] == an and r["epoch"] == e]
            if not v:
                continue
            fr = float(np.mean([r["above_chance_frac"] for r in v]))
            out[(an, e)] = {"animal": an, "epoch": e, "n_sessions": len(v),
                            "acc": float(np.mean([r["acc"] for r in v])),
                            "null": float(np.mean([r["null"] for r in v])),
                            "above_chance_frac": fr,
                            "retained": fr / float(np.mean(pf)) if pf else float("nan"),
                            "n_above_null_p05": int(sum(1 for r in v if r["p"] < 0.05)),
                            "n_periods_scored": int(sum(r["n"] for r in v))}
    return out


def cohort(per_session, seed=0):
    """Pooled rows (the page's flat-pool definition) + nested animals -> sessions bootstrap (analysis_kit.boot_delta
    of the above-chance fraction, post - pre, animal-weighted, paired) and animal-weighted retained."""
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(seed)
    pre = {}
    for r in per_session:
        if r["epoch"] == "pre":
            pre.setdefault(r["animal"], []).append(r["above_chance_frac"])
    pre_flat = float(np.mean([r["above_chance_frac"] for r in per_session if r["epoch"] == "pre"]))
    pa = per_animal(per_session)
    rows = []
    for e in EPOCHS:
        v = [r for r in per_session if r["epoch"] == e]
        if not v:
            continue
        fr = float(np.mean([r["above_chance_frac"] for r in v]))
        row = {"epoch": e, "n_sessions": len(v), "n_animals": len({r["animal"] for r in v}),
               "acc": float(np.mean([r["acc"] for r in v])), "null": float(np.mean([r["null"] for r in v])),
               "above_chance_frac": fr, "retained": fr / pre_flat,
               "retained_animal_weighted": float(np.mean([pa[(a, e)]["retained"] for a in sorted(pre)
                                                          if (a, e) in pa])),
               "n_animals_retained_lt1": int(sum(1 for a in sorted(pre) if (a, e) in pa and pa[(a, e)]["retained"] < 1)),
               "n_sessions_above_null_p05": int(sum(1 for r in v if r["p"] < 0.05)),
               "delta_frac": np.nan, "delta_lo": np.nan, "delta_hi": np.nan, "delta_n_animals": 0,
               "delta_excludes_zero": False}
        if e != "pre":
            post = {}
            for r in v:
                post.setdefault(r["animal"], []).append(r["above_chance_frac"])
            iv = ak.boot_delta(post, pre, rng)
            if iv is not None:
                row.update({"delta_frac": iv.point, "delta_lo": iv.lo, "delta_hi": iv.hi,
                            "delta_n_animals": iv.n_animals, "delta_excludes_zero": bool(iv.excludes_zero)})
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ figures
def _read(p):
    from wfield_local import analysis_kit as ak
    return ak.read_rows(p)


def plot_main(per_session, crow, mask, pre, out, stem):
    """A balanced accuracy (sessions + pooled + null), B per-animal retained, C pooled change with bootstrap CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import config as _cfg
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    order = [e for e in EPOCHS if any(r["epoch"] == e for r in per_session)]
    x = np.arange(len(order))
    animals = sorted({r["animal"] for r in per_session})
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.9))
    rs = np.random.default_rng(1)
    # A
    for i, e in enumerate(order):
        for an in animals:
            v = [r["acc"] for r in per_session if r["epoch"] == e and r["animal"] == an]
            ax[0].scatter(i + rs.uniform(-0.18, 0.18, len(v)), v, s=26, facecolors="none",
                          edgecolors=EPOCH_LINE[e], marker=ANIMAL_MARKER.get(an, "o"), lw=1.0, zorder=2)
        c = next(r for r in crow if r["epoch"] == e)
        ax[0].plot([i - 0.3, i + 0.3], [c["acc"], c["acc"]], color=EPOCH_LINE[e], lw=3, zorder=3)
        ax[0].plot([i - 0.3, i + 0.3], [c["null"], c["null"]], color="0.45", lw=1.5, ls="--", zorder=3)
    ax[0].axhline(CHANCE, color="r", ls=":", lw=1)
    ax[0].set_ylabel("balanced accuracy (chance 1/6)", fontsize=11)
    ax[0].set_title("A  Frozen pre-stroke decoder, per session", fontsize=11.5, fontweight="bold", loc="left")
    h = [Line2D([], [], color="0.3", lw=3, label="pooled mean (epoch colour)"),
         Line2D([], [], color="0.45", lw=1.5, ls="--", label="block-permutation null (mean)"),
         Line2D([], [], color="r", ls=":", lw=1, label="chance 1/6")]
    h += [Line2D([], [], ls="", marker=ANIMAL_MARKER[a], mfc="none", mec="0.3", label=a) for a in animals]
    ax[0].legend(handles=h, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    # B
    colors = _cfg.animal_color()
    pa = per_animal(per_session)
    for an in animals:
        ys = [pa[(an, e)]["retained"] if (an, e) in pa else np.nan for e in order]
        ax[1].plot(x, ys, "-", color=colors.get(an, "0.4"), lw=1.6, zorder=2)
        ax[1].scatter(x, ys, marker=ANIMAL_MARKER.get(an, "o"), s=40, facecolors="none",
                      edgecolors=colors.get(an, "0.4"), lw=1.4, zorder=3, label=an)
    ax[1].axhline(1.0, color="0.6", ls=":", lw=1)
    ax[1].axhline(0.0, color="0.8", ls="-", lw=0.8)
    ax[1].set_ylabel("above-chance fraction retained (vs own pre)", fontsize=11)
    ax[1].set_title("B  Per animal", fontsize=11.5, fontweight="bold", loc="left")
    ax[1].legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=len(animals))
    # C
    for i, e in enumerate(order):
        c = next(r for r in crow if r["epoch"] == e)
        if e == "pre":
            ax[2].scatter([i], [0], s=60, color=EPOCH_LINE[e], zorder=3)
            continue
        ax[2].plot([i, i], [c["delta_lo"], c["delta_hi"]], color=EPOCH_LINE[e], lw=2.5, zorder=2)
        ax[2].scatter([i], [c["delta_frac"]], s=60, color=EPOCH_LINE[e], zorder=3)
        for an in animals:
            if (an, e) in pa and (an, "pre") in pa:
                ax[2].scatter([i + 0.12], [pa[(an, e)]["above_chance_frac"] - pa[(an, "pre")]["above_chance_frac"]],
                              marker=ANIMAL_MARKER.get(an, "o"), s=24, facecolors="none", edgecolors=EPOCH_LINE[e],
                              lw=1.0, zorder=2)
        if c["delta_excludes_zero"] in (True, "True", 1):
            sig_marks.star_top(ax[2], i)
    ax[2].axhline(0, color="0.6", ls=":", lw=1)
    ax[2].set_ylabel("above-chance fraction, epoch - pre", fontsize=11)
    ax[2].set_title("C  Pooled change (nested bootstrap 95% CI)", fontsize=11.5, fontweight="bold", loc="left")
    ax[2].legend(handles=[Line2D([], [], color="0.3", marker="o", lw=2.5, label="animal-weighted mean, 95% CI"),
                          Line2D([], [], ls="", marker="o", mfc="none", mec="0.3", label="per animal (shape)"),
                          Line2D([], [], ls="", marker="*", color="k", label="* CI excludes 0")],
                 fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels(order, fontsize=10)
        a_.spines[["top", "right"]].set_visible(False)
    sig_marks.reserve_headroom(fig)
    ns = ", ".join(f"{c['epoch']} {c['n_sessions']}" for c in crow)
    fig.suptitle(f"Position decoded from {MASK_TEXT[mask]} by a FROZEN pre-stroke decoder -- {LICK_TEXT[mask]}; "
                 f"baseline: {PRE_TEXT[pre]}\nduration- and lick-gap-matched, 4 bins / period, joint LocaNMF, "
                 f"200 block permutations; sessions {ns}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    p = out / f"{stem}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_confusion(out, stem, mask, pre):
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    conf = rf._read_conf(out / "data" / f"{stem}_confusion.csv", list(EPOCHS), CODES)
    rank = {q: i for i, q in enumerate(CONF_LABELS)}
    idx = sorted(range(6), key=lambda i: rank.get(POSITION_NAMES.get(CODES[i], ""), 99))
    counts = {e: np.asarray(m)[np.ix_(idx, idx)] for e, m in conf.items() if np.asarray(m).sum()}
    labels = ef.anatomical_labels([POSITION_NAMES.get(CODES[i], str(i)) for i in idx], short=True)
    return ef.confusion_row(counts, out, name=f"{stem}_confusion",
                            title=f"Frozen pre-stroke rest decoder confusion, pooled across animals -- "
                                  f"{MASK_TEXT[mask]}; {LICK_TEXT[mask]}; baseline {PRE_TEXT[pre]}",
                            delta=True, chance=CHANCE, labels=labels)


def plot_compare(out, runs):
    """Per animal x epoch retained under each run (page REST all-pre, REST Aug-pre, RESTnl Aug-pre)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    rows = []
    for name, ps in runs.items():
        pa = per_animal(ps)
        for _key, d in pa.items():
            rows.append({"run": name, **d})
    _write(out / "data" / "rest_frozen_compare_per_animal.csv", rows)
    animals = sorted({r["animal"] for r in rows})
    names = list(runs)
    fig, ax = plt.subplots(1, len(animals), figsize=(4.2 * len(animals), 4.6), sharey=True)
    styles = {names[0]: ("s", "0.55", ":"), names[1]: ("^", "0.25", "--"), names[-1]: ("o", "k", "-")}
    for j, an in enumerate(animals):
        for nm in names:
            mk, col, ls = styles[nm]
            ys = [next((r["retained"] for r in rows if r["run"] == nm and r["animal"] == an and r["epoch"] == e),
                       np.nan) for e in EPOCHS]
            ax[j].plot(range(4), ys, ls=ls, color=col, lw=1.4, zorder=2)
            for i, e in enumerate(EPOCHS):
                ax[j].scatter([i], [ys[i]], marker=mk, s=42, facecolors="none", edgecolors=EPOCH_LINE[e], lw=1.4,
                              zorder=3)
        ax[j].axhline(1, color="0.7", ls=":", lw=1)
        ax[j].axhline(0, color="0.85", lw=0.8)
        ax[j].set_title(an, fontsize=12, fontweight="bold")
        ax[j].set_xticks(range(4))
        ax[j].set_xticklabels(EPOCHS, fontsize=9.5)
        ax[j].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("above-chance fraction retained (vs own pre)", fontsize=10.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=styles[nm][1], ls=styles[nm][2], marker=styles[nm][0], mfc="none", label=nm)
         for nm in names]
    fig.legend(handles=h, loc="lower center", ncol=len(names), frameon=False, fontsize=9.5)
    fig.suptitle(f"Frozen pre-stroke rest decoder, retained per animal: page REST (DAQ contact licks, {page_pre_text()}) vs "
                 "REST (Aug pre) vs RESTnl (DAQ + cam4 LP licks, Aug pre)\nmarker colour = epoch; duration- and "
                 "lick-gap-matched", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    p = out / "rest_frozen_compare_per_animal.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------------------------ main
def stem_of(mask, pre):
    return f"rest_frozen_{mask}_{pre}pre_durgap"


def report(per_session, crow, title):
    print(f"\n=== {title} ===\nPER ANIMAL (retained vs own pre; mean acc / null; sessions with p<.05 of n)")
    pa = per_animal(per_session)
    for an in sorted({k[0] for k in pa}):
        print(an, "  ".join(f"{e}: ret {pa[(an, e)]['retained']:.2f} acc {pa[(an, e)]['acc']:.3f}/{pa[(an, e)]['null']:.3f}"
                            f" p<.05 {pa[(an, e)]['n_above_null_p05']}/{pa[(an, e)]['n_sessions']}"
                            for e in EPOCHS if (an, e) in pa))
    print("POOLED (flat retained | animal-weighted retained | delta frac [95% CI] nested bootstrap)")
    for c in crow:
        print(f"  {c['epoch']:<9} n={c['n_sessions']:>3} acc {float(c['acc']):.3f} null {float(c['null']):.3f} "
              f"ret {float(c['retained']):.2f} | aw {float(c['retained_animal_weighted']):.2f} | "
              f"d {float(c['delta_frac']):+.3f} [{float(c['delta_lo']):+.3f}, {float(c['delta_hi']):+.3f}]"
              f"{' *' if c['delta_excludes_zero'] in (True, 'True') else ''}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mask", choices=["restnl", "rest"], default="restnl")
    ap.add_argument("--pre", choices=["aug", "all"], default="aug")
    ap.add_argument("--stage", choices=["collect", "counts", "fit", "all"], default="all")
    ap.add_argument("--plot-only", action="store_true", help="redraw tables + figures from the CSVs (no data read)")
    ap.add_argument("--compare", action="store_true", help="also draw the per-animal comparison figure")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--allow-thin", action="store_true", help="fit animals that fail the fairness gate")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = a.out / f"{a.mask}_{a.pre}pre"
    stem = stem_of(a.mask, a.pre)
    if not a.plot_only:
        labs = collect(a.mask, a.jobs)          # cached per session: a no-op when already collected
        if a.stage == "collect":
            return 0
        data = load_collected(a.mask, labs)
        _rows, cp = counts_table(a.mask, data, out, stem)
        print(f"wrote {cp}", flush=True)
        if a.stage == "counts":
            return 0
        fit_and_score(a.mask, a.pre, data, out, stem, allow_thin=a.allow_thin)
    ps = _read(out / "data" / f"{stem}_sessions.csv")
    crow = cohort(ps)
    _write(out / "data" / f"{stem}_per_animal.csv", list(per_animal(ps).values()))
    _write(out / "data" / f"{stem}_cohort.csv", crow)
    report(ps, crow, f"{a.mask} / {PRE_TEXT[a.pre]}")
    print("wrote", plot_main(ps, crow, a.mask, a.pre, out, stem))
    print("wrote", plot_confusion(out, stem, a.mask, a.pre))
    (out / "data" / f"{stem}_figdata.json").write_text(json.dumps(
        {"mask": a.mask, "pre": a.pre, "mask_text": MASK_TEXT[a.mask], "lick_text": LICK_TEXT[a.mask],
         "inputs": [f"{stem}_sessions.csv", f"{stem}_confusion.csv"], "redraw": "--plot-only"}, indent=1))
    if a.compare:
        runs = {f"page 12.1: REST, {page_pre_text()}": _read(page_sessions()),
                "REST, Aug pre": _read(a.out / "rest_augpre" / "data" / f"{stem_of('rest', 'aug')}_sessions.csv"),
                "RESTnl, Aug pre": _read(a.out / "restnl_augpre" / "data" / f"{stem_of('restnl', 'aug')}_sessions.csv")}
        print("wrote", plot_compare(a.out, runs))
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
