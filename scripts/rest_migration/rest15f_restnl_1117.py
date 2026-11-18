"""Fig 12.1 (epoch family 15f) re-run on the cohort-1 rebuild: the frozen pre-stroke REST position decoder on RESTnl
rest bracketed by two camera WORKING trials (Priya 11-17).

WHAT CHANGES against the page's 15f (`rest_frozen_decoder`, restdock05, DAQ contacts only, in-memory refits):
  REST SET   restdock05 (docked, not running) MINUS every DAQ lick (the behaviour DAQ detector, `restnl_mask.
             daq_lick_onsets`) AND every LP video lick, each widened [0.5 s before, 2.0 s after] (`restnl_mask.BUF_S`), the
             min_rest_s floor re-applied after each step. `--lp cam4` (PRIMARY, Priya 11-17) = cam4 `licks_all.csv` only,
             exactly the rebuild's RESTnl (`rest_frozen_decoder_nl.lickfree_rest` = `restnl_mask.lickfree_mask`);
             `--lp cam4cam1` (stricter SENSITIVITY, the 11-17 first run) adds the cam1-only events
             (`lp_licks.lp_lick_times`). Each set has its own cache folder, frozen align token and output folder.
  BRACKETS   a period is kept only when the cue before it and the next trial are the SAME position (the 15f rule), both
             inside the camera working period (`rest_engagement.engaged_by_cue`, as 12.1b) AND both camera WORKING
             trials (`trial_classes.in_sel(cue_classes_for(...), "working")` = contact | incomplete in the working
             period; the rebuild's one working definition).
  MATCHING   duration and lick gap matched to the central 80 % of the animal's pre periods; the gap is to the nearest
             DAQ-or-LP lick (the licks the mask removed).
  FROZEN     every pre-stroke model (full + one LOSO model per pre session) through `frozen_models.load_or_fit` in the
             rebuild store (kind rest_frozen, align "rest-restnlc1wk-..."): stored, reloaded, nothing old overwritten.
  PRE        `--pre aug` (August-only) and `--pre all` (June + August); which one is primary is decided from the counts.

Everything else is `rest_frozen_decoder_nl`'s (imported): 4 bins / period, joint LocaNMF (projected), StandardScaler +
logistic C = 0.1, balanced accuracy, retained = (acc - 1/6)/(5/6) relative to own pre, LOSO pre baseline, 200 block-label
permutations (predictions fixed; shift / trial alongside), minimum 40 periods / 5 per class, the fairness gate (>= 30
matched pre training periods per position), nested animals -> sessions `boot_delta` for the stars.

STAGES: collect (fan_sessions, 4 workers, cached npz) -> counts (printed per animal x epoch x position BEFORE any fit,
CLAUDE.md rule 8) -> fit (fan_sessions over ANIMALS) -> tables + figures (`--plot-only` redraws from the CSVs).

    python scripts/rest_migration/rest15f_restnl_1117.py --pre all|aug [--stage collect|counts|fit|all] [--animals ..]
"""
from __future__ import annotations

import argparse
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
from scripts.rest_migration import rest_frozen_decoder_nl as nl  # noqa: E402
from wfield_local import config as _config  # noqa: E402
from wfield_local import run_mode as _run_mode  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402

#: WHICH LP LICKS LEAVE THE REST MASK (Priya 11-17): "cam4" = cam4 licks_all only, the rebuild's `restnl_mask` definition
#: (PRIMARY); "cam4cam1" = + cam1-only events (`lp_licks.lp_lick_times`), the stricter SENSITIVITY run. Read from the
#: env so spawned workers see the caller's choice (`main` sets it from --lp before any fan-out).
LP_ENV = "R15F_LP_SET"
#: "c1all" (11-17 night, coordinator / lick-source audit) = cam4 + the cam1-only, non-droplet events of cam1 events_all.csv
#: (the WHOLE trial windows; `restnl_licks.extend`, the RESTnlc1 mask). "cam4cam1" added only POST-CUE cam1 events
#: (`lp_lick_times` reads events.csv), so it is NOT the cam4 + cam1 rest version; kept as run, not re-used.
LP_SETS = ("cam4", "cam4cam1", "c1all")


def lp_set() -> str:
    v = os.environ.get(LP_ENV, "cam4").strip() or "cam4"
    if v not in LP_SETS:
        raise SystemExit(f"{LP_ENV}={v!r}; expected one of {LP_SETS}")
    return v


def _mode():
    """(mask name = cache folder + frozen align token, output folder name, mask text) of the active LP set."""
    if lp_set() == "c1all":                    # mask name follows the RESTnlc1 cache tag (v2 -> v3 11-17)
        from wfield_local import restnl_licks as _rl
        return (f"{_rl.CACHE_TAG}wk", "rest15f_restnlc1all_20261117",
                "RESTnlc1 rest (restdock05: docked, not running; minus every DAQ lick, every cam4 LP lick and every "
                "cam1-only LP lick of the whole trial windows (droplets and far-from-mouth labels removed), -0.5 s / +2.0 s) between two camera "
                "working trials at the same position")
    if lp_set() == "cam4cam1":
        return ("restnlc1wk", "rest15f_restnl_20261117",
                "RESTnl rest (restdock05: docked, not running; minus every DAQ lick and every cam4 + cam1 LP lick, "
                "-0.5 s / +2.0 s) between two camera working trials at the same position")
    return ("restnlwk", "rest15f_restnl_cam4_20261117",
            "RESTnl rest (restdock05: docked, not running; minus every DAQ lick and every cam4 LP lick, "
            "-0.5 s / +2.0 s) between two camera working trials at the same position")


CODES = nl.CODES
LP_TXT = {"cam4": "cam4 LP", "cam4cam1": "cam4 + post-cue cam1 LP", "c1all": "cam4 + cam1 LP (whole trial windows)"}
EPOCHS = nl.EPOCHS
CACHE_ROOT = Path(_config.resolver().root("figures_working")) / "rest_decoder_nl_cache"
#: the primary run tag's epoch folder (where the rebuild's 15f lives), a NEW subfolder per LP set
EPOCH_DIR = _run_mode.primary_output_dir(Path(_config.resolver().root("cohort_out")) / "grant_figures" / "epoch")
MASK, _OUT_NAME, MASK_TEXT = _mode()
OUT_ROOT = EPOCH_DIR / _OUT_NAME
PRE_TEXT = nl.PRE_TEXT
POS_SHORT = {"close_L": "close ipsi", "close_center": "close mid", "close_R": "close contra",
             "far_L": "far ipsi", "far_center": "far mid", "far_R": "far contra"}
POS_ORDER = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]   # lesion LEFT: L = ipsi


def _apply_mode():
    """Re-resolve the LP-set globals from LP_ENV (main, and any spawned worker, call this)."""
    global MASK, _OUT_NAME, MASK_TEXT, OUT_ROOT
    MASK, _OUT_NAME, MASK_TEXT = _mode()
    OUT_ROOT = EPOCH_DIR / _OUT_NAME


# ------------------------------------------------------------------------------------------------ mask
def lickfree_rest_lp(session, rest, rate):
    """``(lick-free rest mask, DAQ + LP lick onsets in samples)``: `rest_frozen_decoder_nl.lickfree_rest` with the LP
    set = `lp_licks.lp_lick_times` (cam4 licks_all + cam1-only events). Raises without LP tables."""
    from wfield_local import config, lp_licks
    from wfield_local import restnl_mask as rn
    n = rest.shape[0]
    nb, na = int(rn.BUF_S[0] * rate), int(rn.BUF_S[1] * rate)
    minrun = int(float(config.defaults()["segmentation"]["rest"]["min_rest_s"]) * rate)
    daq_on = rn.daq_lick_onsets(session, rate)
    lp_s, _cue = lp_licks.lp_lick_times(session["label"])
    lp = np.round(np.asarray(lp_s, float) * rate).astype(np.int64)
    m = rn.floor_runs(rest & ~rn.widen(daq_on, n, nb, na), minrun)
    m = rn.floor_runs(m & ~rn.widen(lp, n, nb, na), minrun)
    return m, np.unique(np.concatenate([daq_on, lp]))


def _code_key():
    from wfield_local import lp_licks
    from wfield_local import restnl_mask as rn
    # the cam4cam1 key is the 11-17 first run's, unchanged (its cache stays valid); cam4 swaps in the cam4-only mask
    mask_fn = _mask_fn()
    src = (inspect.getsource(rf._collect) + inspect.getsource(rf._frame_samples) + inspect.getsource(mask_fn)
           + inspect.getsource(_working_flags) + repr(rn.BUF_S))
    tag = _run_mode.cache_tag()
    if lp_set() == "c1all":                    # + the RESTnlc1 lick-train code (restnl_licks); other sets' keys unchanged
        from wfield_local import restnl_licks
        src += restnl_licks.code_digest()
    return hashlib.sha1(f"{src}|{nl.BINS}|gate|{tag}|{lp_licks.cache_tag()}".encode()).hexdigest()[:12]


def _mask_fn():
    """The rest-mask callable of the active LP set."""
    return {"cam4": nl.lickfree_rest, "cam4cam1": lickfree_rest_lp, "c1all": nl.lickfree_rest_c1}[lp_set()]


def _cache_path(label):
    return CACHE_ROOT / MASK / f"{label}__{_code_key()}.npz"


def _working_flags(session):
    """bool per DAQ cue: a camera WORKING trial (contact | incomplete inside the working period; `in_sel`)."""
    from wfield_local import trial_classes as tc
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    cs = np.asarray(_load_cue_events(session["h5"])["cue_samples"], np.int64)
    full = tc.daq_cue_seconds(session["label"])
    if len(full) != len(cs):
        raise RuntimeError(f"{session['label']}: {len(full)} DAQ cues (classes) vs {len(cs)} (collector)")
    cls = tc.cue_classes_for(session["label"], full)
    return np.asarray(tc.in_sel(cls, "working"), bool), cls


def _collect_job(label):
    _apply_mode()
    from wfield_local import lp_licks
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    fp = _cache_path(label)
    if fp.exists():
        return {"label": label, "cached": True}
    s = next(x for x in SESSIONS if x["label"] == label)
    t0 = time.perf_counter()
    basis = nl._basis(label.split("_")[0])
    got, why = rf._collect(s, basis, nl.BINS, gate=True, rest_mask=_mask_fn(), period_frames=True,
                           period_cues=True)
    n_cam1 = -1
    try:
        _sd, f4, f1 = lp_licks._inputs(label, _config.resolver())
        if lp_set() == "c1all":
            import pandas as _pd

            from wfield_local import restnl_licks
            from wfield_local import restnl_mask as _rn
            _c4 = _pd.read_csv(_rn.licks_all_path(label), usecols=["daq_on_s"]).daq_on_s.to_numpy(float)
            n_cam1 = int(len(restnl_licks.extend(label, _c4[np.isfinite(_c4)])[0]))
        else:
            n_cam1 = int(len(lp_licks.lp_lick_times(label)[0]))
    except Exception:                                                    # noqa: BLE001
        pass
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.with_suffix(f".{os.getpid()}.tmp.npz")
    if got is None:
        np.savez(tmp, ok=False, why=str(why), basis_id=str(basis.basis_id))
    else:
        X, y, g, dur, gap, lick_hz, nfr, cprev, cnext = got
        wk, cls = _working_flags(s)
        np.savez(tmp, ok=True, why="", X=X, y=y, g=g, dur=dur, gap=gap, lick_hz=lick_hz, nfr=nfr, cprev=cprev,
                 cnext=cnext, wk_prev=wk[cprev], wk_next=wk[cnext], n_lp_licks=n_cam1,
                 basis_id=str(basis.basis_id))
    os.replace(tmp, fp)
    return {"label": label, "ok": got is not None, "why": why, "s": round(time.perf_counter() - t0, 1)}


def collect(labs, jobs=4):
    from wfield_local import analysis_kit as ak
    from wfield_local import parallel
    parallel.pin_blas()
    t0 = time.time()
    res, fails = ak.fan_sessions(list(labs), _collect_job, jobs=min(int(jobs), 4), key=lambda it: it,
                                 label=f"collect-{MASK}")
    secs = [v["s"] for _it, v in res if isinstance(v, dict) and "s" in v]
    print(f"[collect] {len(res)} sessions ({len(secs)} computed, median {np.median(secs) if secs else 0:.0f} s), "
          f"{len(fails)} failures, {(time.time() - t0) / 60:.1f} min", flush=True)
    for it, err in fails:
        print("   FAIL", it, str(err)[:300], flush=True)


def load_collected(labs, working=True):
    """{label: dict} with the period arrays restricted to both-working brackets when ``working``."""
    out = {}
    for lab in labs:
        fp = _cache_path(lab)
        if not fp.exists():
            out[lab] = {"ok": False, "why": f"{lab}: not collected"}
            continue
        z = np.load(fp, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["ok"], d["why"] = bool(d["ok"]), str(d["why"])
        if d["ok"]:
            d["n_engaged_span"] = int(len(d["y"]))
            if working:
                m = np.asarray(d["wk_prev"], bool) & np.asarray(d["wk_next"], bool)
                for k in ("X", "y", "g", "dur", "gap", "nfr", "cprev", "cnext", "wk_prev", "wk_next"):
                    d[k] = d[k][m]
                if not m.any():
                    d["ok"], d["why"] = False, f"{lab}: 0 rest periods between two working trials"
        out[lab] = d
    return out


# ------------------------------------------------------------------------------------------------ counts
def _pname(c):
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    return POSITION_NAMES.get(int(c), str(c))


def matching_windows(data, an, pre):
    """(lo_s, hi_s, glo, ghi) exactly as `nl.fit_and_score` computes them (pre sessions passing `_usable`)."""
    from wfield_local import epochs
    pre_labs = []
    for lab, d in data.items():
        if not lab.startswith(an + "_") or not d["ok"] or epochs.epoch_of(lab) != "pre" or not nl._pre_ok(lab, pre):
            continue
        if rf._usable(d["y"], nl.MIN_PERIODS, nl.MIN_PER_CLASS)[0]:
            pre_labs.append(lab)
    if len(pre_labs) < 2:
        return None, pre_labs
    dur = np.concatenate([data[k]["dur"] for k in pre_labs])
    gap = np.concatenate([data[k]["gap"] for k in pre_labs])
    gap = gap[np.isfinite(gap)]
    return (float(np.percentile(dur, nl.PCT)), float(np.percentile(dur, 100 - nl.PCT)),
            float(np.percentile(gap, nl.PCT)), float(np.percentile(gap, 100 - nl.PCT))), pre_labs


def counts(data, data_span, pre, out):
    """Per animal x epoch x position: rest periods (engaged span; + both working; + duration / gap matched under this
    pre set's windows) and sessions. Pre rows count only this pre set's sessions."""
    from wfield_local import epochs
    rows = []
    animals = sorted({lab.split("_")[0] for lab in data})
    for an in animals:
        win, pre_labs = matching_windows(data, an, pre)
        for lab in data:
            if not lab.startswith(an + "_"):
                continue
            ep = epochs.epoch_of(lab)
            if ep is None or (ep == "pre" and not nl._pre_ok(lab, pre)):
                continue
            d, d0 = data[lab], data_span[lab]
            for c in CODES:
                n0 = int((d0["y"] == c).sum()) if d0["ok"] else 0
                n1 = int((d["y"] == c).sum()) if d["ok"] else 0
                n2 = 0
                if d["ok"] and win is not None:
                    lo, hi, glo, ghi = win
                    m = (d["y"] == c) & (d["dur"] >= lo) & (d["dur"] <= hi) & (d["gap"] >= glo) & (d["gap"] <= ghi)
                    n2 = int(m.sum())
                rows.append({"animal": an, "epoch": ep, "label": lab, "position": _pname(c), "n_span": n0,
                             "n_working": n1, "n_working_matched": n2,
                             "pre_train_session": lab in pre_labs})
    import pandas as pd
    df = pd.DataFrame(rows)
    p = out / "data" / f"counts_by_session_{pre}pre.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    piv = {}
    for col in ("n_span", "n_working", "n_working_matched"):
        t = df.pivot_table(index=["animal", "epoch"], columns="position", values=col, aggfunc="sum").reindex(
            columns=POS_ORDER)
        t = t.rename(columns=POS_SHORT)
        t["sessions"] = df.groupby(["animal", "epoch"]).label.nunique()
        order = [(a, e) for a in animals for e in EPOCHS if (a, e) in t.index]
        piv[col] = t.loc[order]
        piv[col].to_csv(out / "data" / f"counts_animal_epoch_position_{col}_{pre}pre.csv")
    return piv


# ------------------------------------------------------------------------------------------------ fit
def _write_tolerant(p, rows):
    """`nl._write`, but an empty table (an animal the fairness gate refuses) writes a one-column stub instead of raising
    (nl.fit_and_score would otherwise lose its skipped list with an IndexError)."""
    if not rows:
        Path(p).write_text("none\n", encoding="utf-8")
        return
    _NL_WRITE(p, rows)


_NL_WRITE = nl._write


def _fit_job(item):
    """One animal (module-level for spawn): `nl.fit_and_score` on its sessions, own stem."""
    an, pre, out, cfg = item
    _apply_mode()
    _apply_cfg(cfg)                                # floor + match window travel in the item (spawned workers)
    nl._write = _write_tolerant
    from wfield_local import parallel
    parallel.pin_blas()
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    labs = [s["label"] for s in SESSIONS if s["label"].startswith(an + "_")]
    data = {k: v for k, v in load_collected([x for x in nl.wanted_labels() if x in labs]).items()}
    t0 = time.perf_counter()
    ps = nl.fit_and_score(MASK, pre, data, Path(out), f"{stem(pre, cfg)}__{an}")
    return {"animal": an, "n": len(ps), "s": round(time.perf_counter() - t0, 1)}


#: THE FIG 12.1 PRIMARY SETTINGS (Priya 11-18), both options of this script; `rest_frozen_decoder_nl` keeps its own
#: defaults (MIN_PERIODS 40, PCT 10) so other consumers do not change.
#:   SCORED POST-STROKE SESSION FLOOR = 30: a post-stroke session is scored with >= 30 matched lick-free rest periods (in
#:     trained classes); pre sessions, training pools, LOSO and the frozen models keep nl.MIN_PERIODS (40).
#:   MATCH WINDOW = 5 % per tail (central 90 %) of the animal's pre duration and lick-gap distributions, for training AND
#:     scoring. nl.PCT is in the frozen align ("dur5-gap5"), so it selects its own stored models.
SCORE_MIN_POST_DEFAULT = 30
MATCH_PCT_DEFAULT = 5.0


def _cfg(cfg=None):
    """(floor, pct) with the nl defaults (40, 10) filled in."""
    floor, pct = (None, None) if cfg is None else cfg
    return (nl.MIN_PERIODS if floor is None else int(floor)), (10.0 if pct is None else float(pct))


def _apply_cfg(cfg):
    """Set the scored floor and match window on `rest_frozen_decoder_nl` (main and every worker call this)."""
    floor, pct = _cfg(cfg)
    nl.SCORE_MIN_POST = floor
    nl.PCT = pct


def floor_tag(cfg=None):
    """'' at (40, 10) (the 11-17 names), else '_floor<n>' and / or '_pct<p>'."""
    floor, pct = _cfg(cfg)
    return ("" if floor == nl.MIN_PERIODS else f"_floor{floor}") + ("" if pct == 10.0 else f"_pct{pct:g}")


def out_name(cfg=None):
    """Output folder of the active LP set and settings: the 11-17 folder at (40, 10), else '<base><tag>_20261118'."""
    tag = floor_tag(cfg)
    return _OUT_NAME if not tag else f"{_OUT_NAME.rsplit('_', 1)[0]}{tag}_20261118"


def stem(pre, cfg=None):
    return f"rest15f_{MASK}_{pre}pre_durgap{floor_tag(cfg)}"


def merge(pre, out, animals, cfg=None, kinds=("sessions", "frozen", "skipped", "confusion")):
    import pandas as pd
    dp = out / "data"
    for kind in kinds:
        parts = [dp / f"{stem(pre, cfg)}__{an}_{kind}.csv" for an in animals]
        dfs = [pd.read_csv(p) for p in parts if p.exists() and p.stat().st_size > 0]
        dfs = [d for d in dfs if len(d.columns) > 1 or kind == "skipped"]
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        if kind == "confusion":
            df = df.groupby(["epoch", "true", "pred"], as_index=False, sort=False)["count"].sum()
        if kind == "skipped":
            df = df[df.skipped.notna() & (df.skipped.astype(str) != "")]
        df.to_csv(dp / f"{stem(pre, cfg)}_{kind}.csv", index=False)


# ------------------------------------------------------------------------------------------------ figure
#: widest suptitle line (characters) that fits the 17 in figure at fontsize 10 with a margin; longer lines are wrapped
#: (the c1all mask text, ~260 characters, was clipped at both edges, 11-18)
TITLE_WRAP = 185
#: height (inches) one more suptitle line takes at fontsize 10
TITLE_LINE_IN = 0.18


def suptitle_text(pre, ns, cfg=None, width=TITLE_WRAP):
    """The main figure's suptitle, every line wrapped to ``width`` characters."""
    floor, pct = _cfg(cfg)
    import textwrap
    lines = [f"Rest position read by a FROZEN (stored) pre-stroke decoder; baseline: {PRE_TEXT[pre]}", MASK_TEXT,
             f"duration- and lick-gap-matched to the central {100 - 2 * pct:g}% of pre (gap to nearest DAQ or LP lick), "
             f"4 bins / period, joint LocaNMF, "
             f"200 block permutations; stars: nested animal -> session bootstrap of the change vs own pre; "
             f"scored sessions need >= {floor} matched periods post-stroke, >= {nl.MIN_PERIODS} pre; sessions {ns}"]
    return "\n".join(textwrap.fill(s, width=width, break_on_hyphens=False) for s in lines)


def plot_main(ps, crow, pre, out, cfg=None):
    """A per-session balanced accuracy + pooled mean + block-permutation null; B per animal (sessions as open markers,
    retained vs own pre; animal mean large); C pooled change (nested bootstrap CI, star = CI excludes 0; animal means; sessions faded, each minus its
    animal's own pre mean, pre sessions included)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    order = [e for e in EPOCHS if any(r["epoch"] == e for r in ps)]
    animals = sorted({r["animal"] for r in ps})
    pa = nl.per_animal(ps)
    ns = ", ".join(f"{c['epoch']} {c['n_sessions']}" for c in crow)
    title = suptitle_text(pre, ns, cfg)
    # 3 title lines = the original 6.2 in figure with the panels below 0.87; each wrapped line past the third adds
    # TITLE_LINE_IN to the height (the panels keep their size) and lowers the panels' top by the same amount
    extra = max(0, title.count("\n") - 2) * TITLE_LINE_IN
    fig_h = 6.2 + extra
    fig = plt.figure(figsize=(17, fig_h))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1])
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    rs = np.random.default_rng(1)
    x = np.arange(len(order))
    # A
    for i, e in enumerate(order):
        for an in animals:
            v = [float(r["acc"]) for r in ps if r["epoch"] == e and r["animal"] == an]
            ax[0].scatter(i + rs.uniform(-0.2, 0.2, len(v)), v, s=26, facecolors="none", edgecolors=EPOCH_LINE[e],
                          marker=ANIMAL_MARKER.get(an, "o"), lw=1.0, zorder=2)
        c = next(r for r in crow if r["epoch"] == e)
        ax[0].plot([i - 0.32, i + 0.32], [c["acc"], c["acc"]], color=EPOCH_LINE[e], lw=3, zorder=3)
        ax[0].plot([i - 0.32, i + 0.32], [c["null"], c["null"]], color="0.45", lw=1.5, ls="--", zorder=3)
    ax[0].axhline(rf.CHANCE, color="r", ls=":", lw=1)
    ax[0].set_ylabel("balanced accuracy (chance 1/6)", fontsize=11)
    ax[0].set_title("A  Frozen pre-stroke decoder, per session", fontsize=11.5, fontweight="bold", loc="left")
    h = [Line2D([], [], color="0.3", lw=3, label="pooled session mean (color = epoch)"),
         Line2D([], [], color="0.45", lw=1.5, ls="--", label="block-permutation null (mean)"),
         Line2D([], [], color="r", ls=":", lw=1, label="chance 1/6")]
    ax[0].legend(handles=h, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1)
    # B: per animal, sessions (small open) + animal mean (large open), retained vs own pre
    off = {an: (j - (len(animals) - 1) / 2) * 0.19 for j, an in enumerate(animals)}
    for an in animals:
        pre_f = pa.get((an, "pre"), {}).get("above_chance_frac")
        if not pre_f:
            continue
        for i, e in enumerate(order):
            v = [float(r["above_chance_frac"]) / pre_f for r in ps if r["epoch"] == e and r["animal"] == an]
            if not v:
                continue
            ax[1].scatter(i + off[an] + rs.uniform(-0.04, 0.04, len(v)), v, s=14, facecolors="none",
                          edgecolors=EPOCH_LINE[e], marker=ANIMAL_MARKER.get(an, "o"), lw=0.7, alpha=0.7, zorder=2)
            ax[1].scatter([i + off[an]], [pa[(an, e)]["retained"]], s=95, facecolors="none", edgecolors=EPOCH_LINE[e],
                          marker=ANIMAL_MARKER.get(an, "o"), lw=2.0, zorder=3)
    ax[1].axhline(1.0, color="0.6", ls=":", lw=1)
    ax[1].axhline(0.0, color="0.75", ls="-", lw=0.8)
    ax[1].set_ylabel("above-chance fraction retained (vs own pre)", fontsize=11)
    ax[1].set_title("B  Per animal (left to right: " + ", ".join(animals) + ")", fontsize=11.5, fontweight="bold",
                    loc="left")
    hb = [Line2D([], [], ls="", marker=ANIMAL_MARKER[a], mfc="none", mec="0.3", ms=8, mew=1.6, label=a)
          for a in animals]
    hb += [Line2D([], [], ls="", marker="o", mfc="none", mec="0.3", ms=4, label="session"),
           Line2D([], [], ls="", marker="o", mfc="none", mec="0.3", ms=9, mew=1.8, label="animal mean")]
    ax[1].legend(handles=hb, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09),
                 ncol=len(hb))
    # C: sessions (small, faded, left of the CI; pre sessions too = the no-stroke spread around 0) are drawn only;
    # the mean, CI and stars come from `crow` (boot_delta) unchanged (Priya 11-18)
    rc = np.random.default_rng(2)                    # own generator: panels A / B keep their jitter
    for i, e in enumerate(order):
        for an in animals:
            pre_m = pa.get((an, "pre"), {}).get("above_chance_frac")
            v = [float(r["above_chance_frac"]) - pre_m for r in ps if r["epoch"] == e and r["animal"] == an] \
                if pre_m is not None else []
            if v:
                ax[2].scatter(i - 0.17 + rc.uniform(-0.07, 0.07, len(v)), v, s=14, facecolors="none",
                              edgecolors=EPOCH_LINE[e], marker=ANIMAL_MARKER.get(an, "o"), lw=0.7, alpha=0.45,
                              zorder=1)
    for i, e in enumerate(order):
        c = next(r for r in crow if r["epoch"] == e)
        if e == "pre":
            ax[2].scatter([i], [0], s=60, color=EPOCH_LINE[e], zorder=3)
            continue
        ax[2].plot([i, i], [c["delta_lo"], c["delta_hi"]], color=EPOCH_LINE[e], lw=2.5, zorder=2)
        ax[2].scatter([i], [c["delta_frac"]], s=60, color=EPOCH_LINE[e], zorder=3)
        for an in animals:
            if (an, e) in pa and (an, "pre") in pa:
                ax[2].scatter([i + 0.15], [pa[(an, e)]["above_chance_frac"] - pa[(an, "pre")]["above_chance_frac"]],
                              marker=ANIMAL_MARKER.get(an, "o"), s=26, facecolors="none", edgecolors=EPOCH_LINE[e],
                              lw=1.0, zorder=2)
        if c["delta_excludes_zero"] in (True, "True", 1):
            sig_marks.star_top(ax[2], i)
    ax[2].axhline(0, color="0.6", ls=":", lw=1)
    ax[2].set_ylabel("above-chance fraction, epoch - own pre", fontsize=11)
    ax[2].set_title("C  Change vs own pre (95% CI)", fontsize=11.5, fontweight="bold", loc="left")
    ax[2].legend(handles=[Line2D([], [], color="0.3", marker="o", lw=2.5, label="animal-weighted mean, 95% CI"),
                          Line2D([], [], ls="", marker="o", mfc="none", mec="0.3", mew=1.0, ms=6,
                                 label="animal mean (shape = animal)"),
                          Line2D([], [], ls="", marker="o", mfc="none", mec="0.3", mew=0.7, ms=4, alpha=0.45,
                                 label="session - own pre mean (faded; pre = no-stroke spread)"),
                          Line2D([], [], ls="", marker="$*$", color="k", ms=9, label="CI excludes 0")],
                 fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1)
    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels(order, fontsize=10)
        a_.spines[["top", "right"]].set_visible(False)
    sig_marks.reserve_headroom(fig)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.02 * 6.2 / fig_h, 1, 1 - (0.13 * 6.2 + extra) / fig_h))
    p = out / f"{stem(pre, cfg)}.png"
    save_png_svg(fig, p, dpi=150)
    plt.close(fig)
    return p


def plot_confusion(out, pre, cfg=None):
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import CONF_LABELS
    conf = rf._read_conf(out / "data" / f"{stem(pre, cfg)}_confusion.csv", list(EPOCHS), CODES)
    rank = {q: i for i, q in enumerate(CONF_LABELS)}
    idx = sorted(range(6), key=lambda i: rank.get(_pname(CODES[i]), 99))
    cnt = {e: np.asarray(m)[np.ix_(idx, idx)] for e, m in conf.items() if np.asarray(m).sum()}
    labels = ef.anatomical_labels([_pname(CODES[i]) for i in idx], short=True)
    flo, pct = _cfg(cfg)
    return ef.confusion_row(cnt, out, name=f"{stem(pre, cfg)}_confusion",
                            title=f"Frozen pre-stroke rest decoder confusion, pooled across animals -- RESTnl "
                                  f"(DAQ + {LP_TXT[lp_set()]} licks removed), working brackets; "
                                  f"baseline {PRE_TEXT[pre]}; matched to the central {100 - 2 * pct:g}% of pre; "
                                  f"scored sessions >= {flo} matched periods post-stroke, >= {nl.MIN_PERIODS} pre",
                            delta=True, chance=rf.CHANCE, labels=labels)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre", choices=["aug", "all"], default="all")
    ap.add_argument("--stage", choices=["collect", "counts", "fit", "all"], default="all")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--lp", choices=LP_SETS, default="cam4",
                    help="LP licks removed from rest: cam4 (= restnl_mask), cam4cam1 (post-cue cam1 only; superseded) or c1all "
                         "(cam4 + cam1 events_all = RESTnlc1, 11-17 night)")
    ap.add_argument("--score-min-post", type=int, default=SCORE_MIN_POST_DEFAULT,
                    help="scored POST-stroke session floor (matched lick-free rest periods); pre sessions, training "
                         f"pools and frozen models keep {nl.MIN_PERIODS}. Default 30 (Priya 11-18, primary)")
    ap.add_argument("--match-pct", type=float, default=MATCH_PCT_DEFAULT,
                    help="percent trimmed from EACH tail of the pre duration and lick-gap distributions (training and "
                         "scoring; in the frozen align). Default 5 = central 90 %% (Priya 11-18, primary); 10 = 11-17")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    os.environ[LP_ENV] = a.lp                      # before any fan-out: spawned workers read it at import
    _apply_mode()
    cfg = (int(a.score_min_post), float(a.match_pct))
    _apply_cfg(cfg)                                # main process too (counts / matching_windows); workers: the item
    a.out = a.out or EPOCH_DIR / out_name(cfg)
    print(f"scored post-stroke floor: {cfg[0]} (pre {nl.MIN_PERIODS}); match window: {cfg[1]:g} % per tail "
          f"(nl.PCT = {nl.PCT:g})", flush=True)
    print(f"LP set: {a.lp} -> mask {MASK}", flush=True)
    import wfield_local
    print(f"wfield_local: {wfield_local.__file__}\nout: {a.out}\ncache: {CACHE_ROOT / MASK} key {_code_key()}",
          flush=True)
    from wfield_local import frozen_models as fm
    print(f"frozen store: {fm.local_dir()} exists={fm.local_dir().exists()}", flush=True)
    t0 = time.time()
    out = a.out / f"{a.pre}pre"
    labs = nl.wanted_labels()
    if a.animals:
        labs = [x for x in labs if x.split("_")[0] in a.animals]
    animals = sorted({x.split("_")[0] for x in labs})
    if not a.plot_only:
        collect(labs, a.jobs)
        if a.stage == "collect":
            return 0
        data = load_collected(labs)
        span = load_collected(labs, working=False)
        import pandas as pd
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            piv = counts(data, span, a.pre, out)
            for col, t in piv.items():
                print(f"\n=== REST PERIODS per animal x epoch x position: {col} ({PRE_TEXT[a.pre]}) ===\n{t}",
                      flush=True)
        if a.stage == "counts":
            return 0
        from wfield_local import analysis_kit as ak
        res, fails = ak.fan_sessions([(an, a.pre, str(out), cfg) for an in animals], _fit_job, jobs=len(animals),
                                     key=lambda it: it[0], label="fit-animals")
        for r in res:
            print("[fit]", r, flush=True)
        for it, err in fails:
            print("   FIT FAIL", it, str(err)[:400], flush=True)
        merge(a.pre, out, animals, cfg)
    else:
        # `ef.confusion_row` writes its panel table to data/<stem>_confusion.csv, over the merged counts it was drawn
        # from (11-18: a --plot-only redraw then failed on the panel table); rebuild the counts from the per-animal parts
        merge(a.pre, out, animals, cfg, kinds=("confusion",))
    ps = nl._read(out / "data" / f"{stem(a.pre, cfg)}_sessions.csv")
    crow = nl.cohort(ps)
    nl._write(out / "data" / f"{stem(a.pre, cfg)}_per_animal.csv", list(nl.per_animal(ps).values()))
    nl._write(out / "data" / f"{stem(a.pre, cfg)}_cohort.csv", crow)
    nl.report(ps, crow, f"{MASK} / {PRE_TEXT[a.pre]}")
    print("wrote", plot_main(ps, crow, a.pre, out, cfg))
    print("wrote", plot_confusion(out, a.pre, cfg))
    (out / "data" / f"{stem(a.pre, cfg)}_figdata.json").write_text(json.dumps(
        {"mask": MASK, "pre": a.pre, "mask_text": MASK_TEXT, "run_tag": _run_mode.tag(),
         "inputs": [f"{stem(a.pre, cfg)}_sessions.csv", f"{stem(a.pre, cfg)}_confusion.csv"], "redraw": "--plot-only",
         "score_min_post": cfg[0], "min_periods_pre": nl.MIN_PERIODS, "match_pct_per_tail": cfg[1],
         "script": "scripts/rest_migration/rest15f_restnl_1117.py"}, indent=1))
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
