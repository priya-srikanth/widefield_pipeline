"""DISENGAGEMENT TESTS 1-4 (Priya 10-31): why do animals quit faster after the stroke, and what is the quit state?

PRE-REGISTERED in docs/status/DISENGAGEMENT_PREREG_20261031.md (committed 34e80dd9 BEFORE any test was computed) --
read it first; this script implements it, and every departure is logged there under "Deviations". PROVISIONAL, n = 4.

    TEST 1 SATIETY      P1 collected water at quit-or-end, EARLY POST (acute + subacute) minus pre
    TEST 2 FRUSTRATION  P2 reference-block lapse rate after a scheduled FAR-CONTRA block minus after another
                           non-reference block, early post minus pre; + giving-up rate vs the pre working distribution
    TEST 3 FATIGUE      P3 within-bout deceleration in the 10 min before the quit minus time-matched controls
    TEST 4 STATE        P4 global (cortex) pre-cue level, RESTnl-referenced, quit period minus time-matched controls
    + a DESCRIPTIVE behaviour figure: licks per responded trial by spout and epoch.

Primary family {P1..P4}, Holm. Per-animal tables are PRINTED (and written) before every bootstrap (ground rule 8).

STAGES (all outputs under <imaging_encoding>/disengagement/):
    python scripts/disengagement_tests.py extract [--jobs 4]   per-session caches  sessions/beh_<label>.pkl, img_<label>.npz
    python scripts/disengagement_tests.py analyse              tables/*.csv + figdata/disengagement.pkl (+ draws figures)
    python scripts/disengagement_tests.py plot                 figures from figdata only (also: --plot-only)

DEFINITIONS (prereg section 2): trials = `engagement_decomposition.session_trials` through `analysis_kit.session_behavior`;
QUIT = the CAMERA quit since 11-05 (Priya 11-05: behaviour-only figures move to the camera working period;
`trial_classes.camera_quit`, A6 of docs/status/reanalysis_20261105/ENGAGEMENT_CAMERA_2026-11-05.md: the first trial after
the session's last camera lick, a quit only when that tail is >= 6 trials and starts >= 300 s after the first cue, else
censored); sessions without video (PS93_0813) have no camera quit and are left out (listed as DAQ-only rows).
DISENGAGEMENT_QUIT=daq reruns the prereg's DAQ detector (`quit_point.session_quit`, 2.0 s scoring) as the labelled
robustness row, into <imaging_encoding>/disengagement_daqquit/ (session caches shared). Pre = August only (unchanged).
Until 11-05 the DAQ quit was the only quit. Responded / hit for every covariate =
DAQ lick inside the session's own task window (gui_config timing.response_window, 3.5 s), cut at the next cue. Water: a
drop is delivered at the cue on every trial (auto_after_delay, delay 0; held only after 6 misses) and docks at a wick, so
COLLECTED water = DAQ reward_ttl deliveries on responded trials x the reward_ul in force (gui_config + config_changes
set_ack), DELIVERED = all deliveries. Positions by lesion side (all lesions LEFT: L = ipsi).

IMAGING: `combined_encoding.load_imaging` (WFIELD_INPUT_CACHE), ROI v8 masks + split rests, x1e3. Two constructions:
'b' = the 600-s slow basis removed (the construction RESTnl's level was measured on -> PRIMARY, see Deviations) and
'0' = no extra basis (sensitivity, referenced within session to its own early engaged pre-cue level).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("WIDEFIELD_PRE_FROM", "0801")          # AUGUST-ONLY pre baseline (primary); before config import

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

#: which quit the tests use (Priya 11-05): "camera" (primary, `trial_classes.camera_quit`) | "daq" (the prereg's S28
#: detector, labelled robustness row, written beside the primary)
QUIT_ENV = "DISENGAGEMENT_QUIT"
QUIT = os.environ.get(QUIT_ENV, "camera").strip().lower()
if QUIT not in ("camera", "daq"):
    raise SystemExit(f"{QUIT_ENV}={QUIT!r}: expected camera or daq")
OUT = E / ("disengagement" if QUIT == "camera" else "disengagement_daqquit")
SESS = E / "disengagement" / "sessions"  # per-session caches carry BOTH quits: one extraction serves both rows
FIG, TAB, FDIR = OUT / "figures", OUT / "tables", OUT / "figdata"
VERSION = 1
BEH_VERSION = 4                          # 2: + n_pre2 (DAQ licks in the 2 s before the cue), 10-31; 3: lick_samples = the behaviour DAQ detector (11-05, 8.53); 4: + quit_cam (camera quit, 11-05)
EPS = ("pre", "acute", "subacute", "chronic")
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
REF = ("close_ipsi", "close_mid")
OTHER_NONREF = ("close_contra", "far_ipsi", "far_mid")
CANON = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
         "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
FR_OTHER = (0, 1, 3, 6, 9, 10)          # fr_groups.master_v1 "other" (uninstructed) factors of master_pre_v1
BOUT_GAP_S, MIN_BOUT, MIN_ILI_LICKS = 0.5, 6, 4
WIN_S = 600.0                            # prodrome / quit-period window
RATE_S = 300.0                           # local collected reward rate window
MIN_BLOCKS = 3                           # reference blocks per arm (P2)
MIN_TRIALS_WIN = 3                       # trials (or bouts) a window needs to carry a value
SEED = 20261031
AREAS = ("cortex", "tjM1_right", "ALM_right", "SSp-m_right", "SSp-n_right",
         "tjM1_left", "ALM_left", "SSp-m_left", "SSp-n_left", "VISpm_right", "VISpm_left", "RSPd_right", "RSPd_left")


# =============================================================================================== extraction
def _beh_dir(s):
    from wfield_local import block_ids
    return block_ids._behavior_dir(s)


def _resp_window_s(d) -> float:
    try:
        return float(json.load(open(d / "gui_config.json"))["timing"]["response_window"]) / 1000.0
    except Exception:                                                  # noqa: BLE001
        return 3.5


def _reward_volumes(d, n_daq: int):
    """Per-DAQ-delivery nominal volume (µl), the GUI's own water total, and whether the counts matched."""
    g = json.load(open(d / "gui_config.json"))
    ul0 = float(g.get("reward_ul", 2.0))
    rw = []
    with open(d / "events.csv", encoding="utf-8", errors="ignore") as fh:
        for r in csv.DictReader(fh):
            if r.get("event_name") == "reward":
                rw.append(float(r["gui_timestamp_unix"]))
    t_ch, v_ch = [], []
    p = d / "config_changes.csv"
    if p.exists():
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for r in csv.DictReader(fh):
                if r.get("change_type") == "set_ack" and r.get("key") == "task.reward_ul":
                    try:
                        t_ch.append(float(r["gui_timestamp_unix"]))
                        v_ch.append(float(r["value"]))
                    except (TypeError, ValueError):
                        pass
    t_ch, v_ch, rw = np.asarray(t_ch), np.asarray(v_ch), np.asarray(rw)
    if t_ch.size:
        i = np.searchsorted(t_ch, rw, side="right") - 1
        vols = np.where(i >= 0, v_ch[np.clip(i, 0, None)], ul0)
    else:
        vols = np.full(rw.size, ul0)
    gui_water = np.nan
    try:
        gui_water = float(json.load(open(d / "summary_end.json"))["water_ul"])
    except Exception:                                                  # noqa: BLE001
        pass
    match = rw.size == n_daq
    if not match:
        vols = np.full(n_daq, float(np.mean(vols)) if vols.size else ul0)
    return vols, gui_water, match, float(vols.sum()) if match else float("nan"), int(rw.size)


def _bouts(t):
    """Split sorted lick times (s) into bouts at gaps > BOUT_GAP_S."""
    if t.size == 0:
        return []
    cut = np.flatnonzero(np.diff(t) > BOUT_GAP_S) + 1
    return np.split(t, cut)


def _trial_rows(rows, cue, nxt, t0, rwin, deliv, lk, lk_resp):
    """Per-trial behaviour rows. RESPONSE measures (hit, lat, n_post, n_pre2) from ``lk_resp`` (the deck's
    discriminator, `analysis_kit.lick_samples`); lick COUNTS (ILIs, bouts, licks / bout, deceleration) from ``lk`` (the
    behaviour detector, `analysis_kit.lick_count_samples`). Times in s."""
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    T = []
    for i, r in enumerate(rows):
        end = min(cue[i] + rwin, nxt[i])
        post = lk_resp[(lk_resp >= cue[i]) & (lk_resp <= end)]          # response measures: the old detector
        hit35 = int(post.size > 0)
        post_c = lk[(lk >= cue[i]) & (lk <= end)]                       # counts: detector A
        dec = []
        ilis = []
        nb = []
        for bt in _bouts(post_c):
            iv = np.diff(bt) * 1e3
            nb.append(bt.size)
            if iv.size:
                ilis.extend(iv.tolist())
            if bt.size >= MIN_BOUT:
                th = max(2, iv.size // 3)
                dec.append(float(np.median(iv[-th:]) - np.median(iv[:th])))
        npre2 = int(((lk_resp >= cue[i] - 2.0) & (lk_resp < cue[i])).sum())
        T.append(dict(order=int(r["order"]), n_pre2=npre2, pos=CANON.get(POSITION_NAMES.get(r["pos"], str(r["pos"])), str(r["pos"])),
                      t=float(cue[i] - t0), hit2=int(r["hit"]), hit=hit35, n_post=int(post.size),
                      lat=float(post[0] - cue[i]) if post.size else np.nan,
                      dec_sum=float(np.sum(dec)), dec_n=len(dec),
                      ili=float(np.median(ilis)) if post_c.size >= MIN_ILI_LICKS and ilis else np.nan,
                      lpb=float(np.mean(nb)) if nb else np.nan,
                      deliv=float(deliv[i]), coll=float(deliv[i]) * hit35))
    return T


def beh_worker(lab: str) -> dict:
    """One session -> SESS/beh_<label>.pkl (module-level for spawn)."""
    f = SESS / f"beh_{lab}.pkl"
    if f.exists():
        try:
            with open(f, "rb") as fh:
                if pickle.load(fh).get("version") == BEH_VERSION:
                    return {"label": lab, "cached": True}
        except Exception:                                              # noqa: BLE001
            pass
    from wfield_local import analysis_kit as ak
    from wfield_local import behavior_events as be
    from wfield_local import config, daq_io, epochs
    from wfield_local import face_rhythm_outputs as fro

    b = ak.session_behavior(lab)
    if b is None:
        raise RuntimeError(f"{lab}: session_behavior None")
    sr = b.sample_rate
    cs = np.asarray(b.cue_samples, float) / sr
    # TWO DETECTORS, ONE PER QUANTITY (coordinator 11-03, rule 9): the lick COUNTS (ILIs, bouts, licks / bout) use THE
    # behaviour detector (`session_behavior.lick_samples` = `analysis_kit.lick_count_samples`); the RESPONSE measures
    # (hit35, latency, n_post, n_pre2) stay on the deck's discriminator (`analysis_kit.lick_samples`), unchanged
    lk = np.sort(np.asarray(b.lick_samples, float) / sr)
    lk_resp = np.sort(np.asarray(ak.lick_samples(b.session), float) / sr)
    t0 = cs[0]
    s = b.session
    d = _beh_dir(s)
    rwin = _resp_window_s(d) if d is not None else 3.5
    rows = b.trials                                                     # session_trials order (= quit index order)
    order = np.array([r["order"] for r in rows], int)
    cue = cs[order]
    nxt = np.append(cs[1:], np.inf)[order]

    # rewards (DAQ), assigned to the trial whose cue precedes them
    with daq_io.open_daq(s["h5"]) as fh:
        rv = daq_io.analog_channel(fh, "reward_ttl", required=False)
    edges = (np.asarray(daq_io.rising_edges((np.asarray(rv) > 2.5).astype(np.int8)), float) / sr
             if rv is not None else np.array([]))
    if d is not None:
        vols, gui_water, vmatch, vsum, n_ev = _reward_volumes(d, edges.size)
    else:
        vols, gui_water, vmatch, vsum, n_ev = np.full(edges.size, 2.0), np.nan, False, np.nan, 0
    ti = np.searchsorted(cue, edges + 0.05, side="right") - 1
    deliv = np.zeros(len(rows))
    np.add.at(deliv, ti[ti >= 0], vols[ti >= 0])

    T = _trial_rows(rows, cue, nxt, t0, rwin, deliv, lk, lk_resp)
    q = b.quit
    quit_cam = camera_quit_rows(lab, cs, rows)

    # running: 1-s bins on the trial clock (t - first cue)
    try:
        ts, sp = be.session_speed(Path(s["h5"]))
        k = np.floor(ts - t0).astype(int)
        ok = k >= 0
        nbin = int(k[ok].max()) + 1 if ok.any() else 0
        cnt = np.bincount(k[ok], minlength=nbin)
        run_mean = np.bincount(k[ok], weights=sp[ok], minlength=nbin) / np.maximum(cnt, 1)
        run_frac = np.bincount(k[ok], weights=(sp[ok] > 3.0).astype(float), minlength=nbin) / np.maximum(cnt, 1)
        run = dict(t=np.arange(nbin) + 0.5, mean=np.where(cnt > 0, run_mean, np.nan),
                   frac=np.where(cnt > 0, run_frac, np.nan))
    except Exception as ex:                                            # noqa: BLE001
        run = dict(err=f"{type(ex).__name__}: {str(ex)[:80]}")

    # FaceRhythm master_pre_v1 whole-session projections, between-trial samples only, 10-s bins
    fr = {}
    try:
        an, date8 = config.animal_of(lab), "2026" + lab.split("_")[-1]
        keys = fro.session_keys(fro.factor_dir(), an, date8)
        if keys:
            X, tt, lossy = [], [], []
            for kf in keys:
                z = np.load(kf, allow_pickle=True)
                X.append(np.asarray(z["session"], float))
                tt.append(np.asarray(z["session_t_daq_s"], float))
                lossy.append(np.asarray(z["session_lossy"], bool))
            X, tt, lossy = np.concatenate(X), np.concatenate(tt), np.concatenate(lossy)
            j = np.searchsorted(cs, tt + 0.5, side="right") - 1          # last cue at or before t + 0.5
            near = np.zeros(tt.size, bool)
            okj = j >= 0
            near[okj] = (tt[okj] - cs[j[okj]]) < 4.0
            jn = np.searchsorted(cs, tt, side="left")                   # next cue within 0.5 s
            okn = jn < cs.size
            near[okn] |= (cs[jn[okn]] - tt[okn]) < 0.5
            use = ~near & ~lossy & np.isfinite(X).all(1)
            kb = np.floor((tt - t0) / 10.0).astype(int)
            use &= kb >= 0
            nb10 = int(kb[use].max()) + 1 if use.any() else 0
            c10 = np.bincount(kb[use], minlength=nb10)
            oth = np.bincount(kb[use], weights=X[use][:, list(FR_OTHER)].sum(1), minlength=nb10) / np.maximum(c10, 1)
            alls = np.bincount(kb[use], weights=X[use].sum(1), minlength=nb10) / np.maximum(c10, 1)
            fr = dict(t=np.arange(nb10) * 10.0 + 5.0, other=np.where(c10 > 0, oth, np.nan),
                      all=np.where(c10 > 0, alls, np.nan), n=c10)
    except Exception as ex:                                            # noqa: BLE001
        fr = dict(err=f"{type(ex).__name__}: {str(ex)[:80]}")

    term = []
    if d is not None:
        with open(d / "events.csv", encoding="utf-8", errors="ignore") as fh:
            for r in csv.DictReader(fh):
                if r.get("event_name") in ("target_trials_reached", "max_duration_reached", "water_limit_reached"):
                    term.append(r["event_name"])
    out = dict(version=BEH_VERSION, label=lab, animal=lab[:4], epoch=epochs.epoch_of(lab), resp_window_s=rwin,
               trials=pd.DataFrame(T), quit_idx=int(q["quit_trial"]), censored=bool(q["censored"]),
               quit_t=float(q["quit_elapsed_s"]), dur=float(q["session_dur_s"]), quit_cam=quit_cam,
               n_reward_daq=int(edges.size), n_reward_log=n_ev, vol_match=bool(vmatch), water_vol_sum=vsum,
               water_gui=gui_water, term="|".join(sorted(set(term))), run=run, fr=fr, t0=float(t0))
    SESS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh)
    tmp.replace(f)
    return {"label": lab, "cached": False}


def img_worker(lab: str) -> dict:
    """One session -> SESS/img_<label>.npz: per-cue pre-cue level and cue response per area, both constructions."""
    f = SESS / f"img_{lab}.npz"
    if f.exists():
        z = np.load(f)
        if int(z["version"]) == VERSION:
            return {"label": lab, "cached": True}
    import iti_licks as il
    import restnl
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.analysis_kit import daq_rate
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events

    s = next(x for x in config.load_sessions() if x["label"] == lab)
    cs = np.asarray(_load_cue_events(s["h5"])["cue_samples"], float) / daq_rate(s)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(lab)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R0 = V.astype(np.float64) @ Ra
    del V, Ub
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    Rb = R0 - Q @ (Q.T @ R0)
    R0 *= 1e3
    Rb *= 1e3
    R0s, cs_names = il.with_rest(R0, cols, lab, basis="split")
    Rbs, _ = il.with_rest(Rb, cols, lab, basis="split")
    ai = [cs_names.index(a) for a in AREAS]
    R0s, Rbs = R0s[:, ai], Rbs[:, ai]
    lev, ok = restnl.level(lab, list(AREAS))

    def mwin(R, a, b):
        i0, i1 = np.searchsorted(ft, a), np.searchsorted(ft, b)
        out = np.full((a.size, R.shape[1]), np.nan)
        cum = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R, 0)])
        good = i1 > i0
        out[good] = (cum[i1[good]] - cum[i0[good]]) / (i1[good] - i0[good])[:, None]
        return out
    pre_b, pre_0 = mwin(Rbs, cs - 2.0, cs), mwin(R0s, cs - 2.0, cs)
    resp_b = mwin(Rbs, cs, cs + 0.5) - mwin(Rbs, cs - 0.5, cs)
    # 1-Hz global ('cortex') trace, no extra basis, on the trial clock
    k = np.floor(ft - cs[0]).astype(int)
    okk = k >= 0
    nb = int(k[okk].max()) + 1
    c1 = np.bincount(k[okk], minlength=nb)
    g1 = np.bincount(k[okk], weights=R0s[okk, 0], minlength=nb) / np.maximum(c1, 1)
    np.savez_compressed(f, version=VERSION, cue_s=cs, pre_b=pre_b, pre_0=pre_0, resp_b=resp_b, areas=np.array(AREAS),
                        lev=lev, lev_ok=ok, g1=np.where(c1 > 0, g1, np.nan), ft_first=ft[0] - cs[0],
                        ft_last=ft[-1] - cs[0])
    return {"label": lab, "cached": False}


def extract(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    labs = ak.curated_labels(order="sorted")
    SESS.mkdir(parents=True, exist_ok=True)
    for name, w in (("behaviour", beh_worker), ("imaging", img_worker)):
        res, fail = ak.fan_sessions(labs, w, jobs=jobs, label=name)
        print(f"{name}: {len(res)} ok ({sum(not v.get('cached') for _, v in res)} new), {len(fail)} failed", flush=True)
        for fl in fail:
            print("   FAIL", fl, flush=True)


def camera_quit_rows(lab: str, cs, rows) -> dict:
    """The camera quit (`trial_classes.camera_quit` on the session's full DAQ cue train ``cs``, s) in this script's
    terms: {"quit_idx": index into ``rows`` (session_trials order) of the first trial at / after the quit cue, -1 when
    censored; "quit_t": that trial's elapsed_s; "censored"; "has_classes"; "n_tail"}."""
    from wfield_local import trial_classes as tc
    c = tc.camera_quit(lab, cs)
    order = np.array([r["order"] for r in rows], int)
    out = {"quit_idx": -1, "quit_t": float("nan"), "censored": True, "has_classes": c["has_classes"],
           "n_tail": c["n_tail"]}
    if not c["censored"]:
        k = int(np.flatnonzero(order >= c["quit_cue"])[0]) if np.any(order >= c["quit_cue"]) else len(rows)
        if 0 < k < len(rows):
            out.update(quit_idx=k, quit_t=float(rows[k]["elapsed_s"]), censored=False)
    return out


def apply_quit(s: dict, source: str = QUIT) -> dict | None:
    """A session record with its quit fields (quit_idx / quit_t / censored) from ``source``: "daq" = the stored S28
    fields unchanged; "camera" = `quit_cam`; None for a session without camera classes (no video: a DAQ-only row)."""
    if source == "daq":
        return s
    c = s.get("quit_cam")
    if c is None:
        raise SystemExit(f"{s['label']}: cache predates the camera quit (BEH_VERSION {BEH_VERSION}): re-run extract")
    if not c["has_classes"]:
        return None
    return {**s, "quit_idx": c["quit_idx"], "quit_t": c["quit_t"], "censored": c["censored"]}


# =============================================================================================== analysis helpers
def _load_all():
    from wfield_local import analysis_kit as ak
    B, I = {}, {}
    daq_only = []
    for lab in ak.curated_labels(order="sorted"):
        p = SESS / f"beh_{lab}.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                rec = apply_quit(pickle.load(fh))
            if rec is None:
                daq_only.append(lab)
                continue
            B[lab] = rec
        q = SESS / f"img_{lab}.npz"
        if q.exists():
            I[lab] = dict(np.load(q))
    if daq_only:
        print(f"== quit = {QUIT}: no camera classes (no video) -> DAQ-only rows, left out here: {daq_only}", flush=True)
    return B, I


def _ep_group(ep):
    return "early" if ep in ("acute", "subacute") else ep


def _by_animal(df, col, mask=None):
    d = df if mask is None else df[mask]
    out = {}
    for an in ANIMALS:
        v = d.loc[d.animal == an, col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            out[an] = v.tolist()
    return out


def _delta(S, col, rng, cells=("early", "acute", "subacute", "chronic")):
    """boot_delta of `col` (per session) for each post cell vs pre, + per-animal deltas."""
    from wfield_local import analysis_kit as ak
    pre = _by_animal(S, col, S.epoch == "pre")
    res = {}
    for c in cells:
        m = S.epoch.isin(("acute", "subacute")) if c == "early" else S.epoch == c
        post = _by_animal(S, col, m)
        iv, dr = ak.boot_delta(post, pre, rng, draws=True)
        per = {a: float(np.mean(post[a]) - np.mean(pre[a])) for a in sorted(set(post) & set(pre))}
        res[c] = dict(iv=iv, p=ak.boot_p(dr) if iv is not None else np.nan, per=per,
                      n_post={a: len(v) for a, v in post.items()}, n_pre={a: len(v) for a, v in pre.items()})
    return res


def _print_animal_table(title, S, col, extra=None):
    print(f"\n=== PER-ANIMAL TABLE: {title} ({col}) ===", flush=True)
    rows = []
    for an in ANIMALS:
        line = {"animal": an}
        for ep in EPS:
            v = S.loc[(S.animal == an) & (S.epoch == ep), col].to_numpy(float)
            v = v[np.isfinite(v)]
            line[ep] = f"{np.mean(v):.3g} (n={v.size})" if v.size else "-"
        if extra is not None:
            line.update(extra.get(an, {}))
        rows.append(line)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False), flush=True)
    return t


def _win(T, a, b):
    return T[(T.t >= a) & (T.t < b)]


def _val(sess, I, key, a, b, area=0, ref_level=None):
    """One measure over the window [a, b) (trial clock, s) of one session. NaN when too thin."""
    T = sess["trials"]
    W = _win(T, a, b)
    if key == "decel":
        R = W[W.pos.isin(REF) & (W.hit == 1)]
        n = R.dec_n.sum()
        return float(R.dec_sum.sum() / n) if n >= MIN_TRIALS_WIN else np.nan
    if key in ("ili", "lpb", "lat"):
        R = W[W.pos.isin(REF) & (W.hit == 1)][key].to_numpy(float)
        R = R[np.isfinite(R)]
        if R.size < MIN_TRIALS_WIN:
            return np.nan
        return float(np.median(R)) if key == "lat" else float(np.mean(R))
    if key == "rate":
        return float(W.coll.sum() / ((b - a) / 60.0)) if len(W) else np.nan
    if key in ("run_mean", "run_frac"):
        r = sess["run"]
        if "t" not in r:
            return np.nan
        m = (r["t"] >= a) & (r["t"] < b)
        v = r["mean" if key == "run_mean" else "frac"][m]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size >= 60 else np.nan
    if key in ("fr_other", "fr_all"):
        r = sess["fr"]
        if "t" not in r:
            return np.nan
        m = (r["t"] >= a) & (r["t"] < b)
        v = r["other" if key == "fr_other" else "all"][m]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size >= 6 else np.nan
    # imaging keys
    im = I.get(sess["label"])
    if im is None:
        return np.nan
    cue_t = im["cue_s"] - im["cue_s"][0]
    m = (cue_t >= a) & (cue_t < b)
    if key in ("level_still", "level_quiet"):
        # EXPLORATORY (post hoc, 10-31): only cues whose pre-cue 2 s had no running (every 1-s bin < 3 mm/s), to ask
        # whether the quit-period level rise is the running that comes with it.
        r = sess["run"]
        if "t" not in r or not bool(im["lev_ok"]):
            return np.nan
        idx = np.flatnonzero(m)
        npre = T.set_index("order").n_pre2 if key == "level_quiet" else None
        still = []
        for i in idx:
            c = cue_t[i]
            k0, k1 = int(np.floor(c - 2.0)), int(np.ceil(c))
            sp = r["mean"][max(k0, 0):max(k1, 0)]
            ok_i = sp.size > 0 and np.all(np.isfinite(sp)) and np.all(sp < 3.0)
            if npre is not None:
                ok_i = ok_i and int(npre.get(i, 1)) == 0
            still.append(ok_i)
        arr = im["pre_b"][idx[np.asarray(still, bool)], area] if idx.size else np.array([])
        arr = arr[np.isfinite(arr)]
        return float(arr.mean() - im["lev"][area]) if arr.size >= MIN_TRIALS_WIN else np.nan
    if key in ("level", "level0", "sd"):
        arr = im["pre_b" if key in ("level", "sd") else "pre_0"][m, area]
        arr = arr[np.isfinite(arr)]
        if arr.size < MIN_TRIALS_WIN:
            return np.nan
        if key == "sd":
            return float(np.std(arr, ddof=1))
        if key == "level":
            if not bool(im["lev_ok"]) or not np.isfinite(im["lev"][area]):
                return np.nan
            return float(arr.mean() - im["lev"][area])
        return float(arr.mean() - (ref_level if ref_level is not None else 0.0))
    if key == "resp":
        ref = T.set_index("order").pos
        idx = np.flatnonzero(m)
        keep = [i for i in idx if ref.get(i, "") in REF]
        arr = im["resp_b"][keep, area]
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size >= MIN_TRIALS_WIN else np.nan
    if key == "band":
        g = im["g1"]
        i0, i1 = int(max(a, 0)), int(min(b, g.size))
        x = g[i0:i1]
        if x.size < 400 or not np.isfinite(x).all():
            return np.nan
        from scipy.signal import welch
        fq, pw = welch(x - x.mean(), fs=1.0, nperseg=200)
        sel = (fq >= 0.01) & (fq <= 0.05)
        return float(np.trapezoid(pw[sel], fq[sel]))
    raise KeyError(key)


def _early_level0(sess, I, area=0):
    """Within-session reference for the no-basis construction: mean pre-cue level over the session's engaged trials in
    its first 30 min."""
    im = I.get(sess["label"])
    if im is None:
        return np.nan
    cue_t = im["cue_s"] - im["cue_s"][0]
    tq = sess["quit_t"] if not sess["censored"] else np.inf
    m = (cue_t < min(1800.0, tq))
    arr = im["pre_0"][m, area]
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size >= 10 else np.nan


def _engaged_through(sess, b):
    """No quit before b and recorded at least to b (trial clock)."""
    if sess["dur"] < b:
        return False
    return sess["censored"] or sess["quit_t"] >= b


def _pairs(B, I, key, window, same_epoch=True, area=0, rate_match=False):
    """Quit-vs-time-matched-control values. window = 'prodrome' [tq-600, tq) | 'quit' [tq, tq+600) (>= 300 s needed)."""
    rows = []
    for lab, sq in B.items():
        if sq["censored"]:
            continue
        tq = sq["quit_t"]
        if window == "prodrome":
            a, b = tq - WIN_S, tq
            if a < 0:
                continue
        else:
            a, b = tq, min(tq + WIN_S, sq["dur"])
            if b - a < 300.0:
                continue
        ref_q = _early_level0(sq, I, area) if key == "level0" else None
        vq = _val(sq, I, key, a, b, area, ref_q)
        if not np.isfinite(vq):
            continue
        rq = _val(sq, I, "rate", a, b) if rate_match else None
        cv = []
        for lc, sc in B.items():
            if lc == lab or sc["animal"] != sq["animal"]:
                continue
            if same_epoch and sc["epoch"] != sq["epoch"]:
                continue
            if not _engaged_through(sc, b):
                continue
            if rate_match:
                rc = _val(sc, I, "rate", a, b)
                if not (np.isfinite(rc) and rq > 0 and abs(rc - rq) <= 0.2 * rq):
                    continue
            ref_c = _early_level0(sc, I, area) if key == "level0" else None
            v = _val(sc, I, key, a, b, area, ref_c)
            if np.isfinite(v):
                cv.append(v)
        if cv:
            rows.append(dict(label=lab, animal=sq["animal"], epoch=sq["epoch"], quit=vq, control=float(np.mean(cv)),
                             n_ctrl=len(cv), diff=vq - float(np.mean(cv))))
    return pd.DataFrame(rows)


def _pair_test(P, rng):
    from wfield_local import analysis_kit as ak
    if not len(P):
        return dict(iv=None, p=np.nan, per={}, n={})
    q = {a: P.loc[P.animal == a, "quit"].tolist() for a in ANIMALS if (P.animal == a).any()}
    c = {a: P.loc[P.animal == a, "control"].tolist() for a in ANIMALS if (P.animal == a).any()}
    iv, dr = ak.boot_delta(q, c, rng, draws=True)
    per = {a: float(np.mean(q[a]) - np.mean(c[a])) for a in q}
    return dict(iv=iv, p=ak.boot_p(dr) if iv is not None else np.nan, per=per, n={a: len(v) for a, v in q.items()})


def _trajectory(B, I, key, area=0, lo=-1800, hi=1200, step=300, same_epoch=True):
    """Quit-aligned (quit - control) per quit per 5-min bin -> per-animal mean curves."""
    edges = np.arange(lo, hi + 1, step)
    out = {}
    for lab, sq in B.items():
        if sq["censored"]:
            continue
        tq = sq["quit_t"]
        for j in range(edges.size - 1):
            a, b = tq + edges[j], tq + edges[j + 1]
            if a < 0 or b > sq["dur"]:
                continue
            ref_q = _early_level0(sq, I, area) if key == "level0" else None
            vq = _val(sq, I, key, a, b, area, ref_q)
            if not np.isfinite(vq):
                continue
            cv = []
            for lc, sc in B.items():
                if lc == lab or sc["animal"] != sq["animal"] or (same_epoch and sc["epoch"] != sq["epoch"]):
                    continue
                if not _engaged_through(sc, b):
                    continue
                v = _val(sc, I, key, a, b, area, _early_level0(sc, I, area) if key == "level0" else None)
                if np.isfinite(v):
                    cv.append(v)
            if cv:
                out.setdefault(sq["animal"], {}).setdefault(j, []).append(vq - float(np.mean(cv)))
    centres = (edges[:-1] + edges[1:]) / 2.0 / 60.0
    curves = {a: np.array([np.mean(d[j]) if j in d else np.nan for j in range(centres.size)]) for a, d in out.items()}
    counts = {a: np.array([len(d.get(j, [])) for j in range(centres.size)]) for a, d in out.items()}
    return dict(x_min=centres, curves=curves, counts=counts)


# =============================================================================================== the tests
def analyse():
    from wfield_local import analysis_kit as ak
    B, I = _load_all()
    print(f"sessions: behaviour {len(B)}, imaging {len(I)}", flush=True)
    rng = np.random.default_rng(SEED)
    TAB.mkdir(parents=True, exist_ok=True)
    FD = {}

    # ---------------------------------------------------------------- session table
    W = pd.read_csv(Path(__file__).resolve().parents[1] / "configs" / "animal_weights.csv", dtype={"date": str})
    rows = []
    for lab, s in B.items():
        T = s["trials"]
        k = len(T) if s["censored"] else s["quit_idx"]
        Tb = T.iloc[:k]
        t_at = s["dur"] if s["censored"] else s["quit_t"]
        rows.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"], censored=int(s["censored"]),
                         quit_min=np.nan if s["censored"] else s["quit_t"] / 60, dur_min=s["dur"] / 60,
                         n_trials=len(T), w_coll=Tb.coll.sum(), w_deliv=Tb.deliv.sum(), n_rew=int((Tb.deliv > 0).sum()),
                         n_hit=int(Tb.hit.sum()), w_coll_total=T.coll.sum(), w_deliv_total=T.deliv.sum(),
                         accrual=Tb.coll.sum() / max(t_at / 60.0, 1e-9), vol_match=int(s["vol_match"]),
                         water_vol_sum=s["water_vol_sum"], water_gui=s["water_gui"], term=s["term"],
                         date="2026" + lab.split("_")[-1]))
    S = pd.DataFrame(rows)
    # 80-min horizon sensitivity (quit counted only if <= 80 min; sessions < 80 min dropped)
    h = []
    for s in B.values():
        if s["dur"] < 4800:
            h.append(np.nan)
            continue
        T = s["trials"]
        if not s["censored"] and s["quit_t"] <= 4800:
            h.append(T.iloc[:s["quit_idx"]].coll.sum())
        else:
            h.append(T[T.t <= 4800].coll.sum())
    S["w_coll_h80"] = h
    pre_dates = {a: set(S.loc[(S.animal == a) & (S.epoch == "pre"), "date"]) for a in ANIMALS}
    wt = {}
    for a in ANIMALS:
        wa = W[W.animal == a]
        base = wa[wa.date.isin(pre_dates[a])].pre_weight_g.astype(float).mean()
        for _, r in wa.iterrows():
            wt[(a, r.date)] = 100.0 * float(r.pre_weight_g) / base
    S["weight_pct"] = [wt.get((a, dte), np.nan) for a, dte in zip(S.animal, S.date)]
    S.to_csv(TAB / "session_table.csv", index=False)
    print("\nWATER LOG CHECK: DAQ deliveries == events.csv rewards in "
          f"{int(S.vol_match.sum())}/{len(S)} sessions; |sum(vol) - GUI water_ul| <= 1%: "
          f"{int((np.abs(S.water_vol_sum - S.water_gui) <= 0.01 * S.water_gui).sum())}", flush=True)
    print(S.loc[S.vol_match == 0, ["label", "epoch", "water_gui", "w_deliv_total"]].to_string(index=False), flush=True)

    # censoring per animal x epoch
    cens = S.groupby(["animal", "epoch"]).censored.agg(["sum", "count"]).reset_index()
    cens.to_csv(TAB / "censoring.csv", index=False)

    # ---------------------------------------------------------------- TEST 1
    t1_tab = _print_animal_table("TEST 1 collected water at quit-or-end, µl", S, "w_coll")
    _print_animal_table("TEST 1 delivered water at quit-or-end, µl", S, "w_deliv")
    _print_animal_table("TEST 1 collected-water accrual, µl/min to quit-or-end", S, "accrual")
    _print_animal_table("TEST 1 pre-session weight, % August pre", S, "weight_pct")
    P1 = _delta(S, "w_coll", rng)
    t1_sens = {"delivered": _delta(S, "w_deliv", rng), "h80": _delta(S, "w_coll_h80", rng),
               "accrual": _delta(S, "accrual", rng), "weight": _delta(S, "weight_pct", rng)}
    pre_mean = float(S.loc[S.epoch == "pre", "w_coll"].mean())
    # KM on the collected-water scale per animal x epoch
    km = {}
    for (an, ep), g in S.groupby(["animal", "epoch"]):
        x = g.w_coll.to_numpy(float)
        ev = (1 - g.censored).to_numpy(int)
        o = np.argsort(x)
        x, ev = x[o], ev[o]
        surv, at, xs, ys = 1.0, len(x), [0.0], [1.0]
        for xi, ei in zip(x, ev):
            if ei:
                surv *= (at - 1) / at
                xs.append(xi)
                ys.append(surv)
            at -= 1
        km[(an, ep)] = dict(x=np.array(xs), y=np.array(ys), cens_x=x[ev == 0])
    FD["t1"] = dict(S=S, P1=P1, sens=t1_sens, pre_mean=pre_mean, km=km, table=t1_tab)

    # ---------------------------------------------------------------- TEST 2
    t2rows, trig = [], []
    for lab, s in B.items():
        T = s["trials"].reset_index(drop=True)
        k = len(T) if s["censored"] else s["quit_idx"]
        blk = (T.pos != T.pos.shift()).cumsum()
        bpos = T.groupby(blk).pos.first()
        bstart = T.index.to_series().groupby(blk).first()
        arms = {"far": [], "other": []}
        arms2 = {"far": [], "other": []}
        for bi in bpos.index:
            if bpos[bi] not in REF or (bi - 1) not in bpos.index:
                continue
            prev = bpos[bi - 1]
            arm = "far" if prev == "far_contra" else ("other" if prev in OTHER_NONREF else None)
            if arm is None:
                continue
            rows_b = T[(blk == bi) & (T.index < k)]
            if not len(rows_b) or bstart[bi] >= k:
                continue
            arms[arm].append(1.0 - rows_b.hit.mean())
            arms2[arm].append(1.0 - rows_b.hit2.mean())
        ok = len(arms["far"]) >= MIN_BLOCKS and len(arms["other"]) >= MIN_BLOCKS
        t2rows.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"],
                           n_far=len(arms["far"]), n_other=len(arms["other"]),
                           lapse_far=np.mean(arms["far"]) if arms["far"] else np.nan,
                           lapse_other=np.mean(arms["other"]) if arms["other"] else np.nan,
                           contrast=(np.mean(arms["far"]) - np.mean(arms["other"])) if ok else np.nan,
                           contrast_2s=(np.mean(arms2["far"]) - np.mean(arms2["other"])) if ok else np.nan))
        if not s["censored"] and k >= 16:
            pre16 = T.iloc[k - 16:k]
            sched = (T.iloc[:k].pos == "far_contra").mean()
            trig.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"],
                             exposure16=(pre16.pos == "far_contra").mean(), sched=sched,
                             excess=(pre16.pos == "far_contra").mean() - sched,
                             contra_fail16=int(((pre16.pos.isin(("close_contra", "far_contra"))) & (pre16.hit == 0)).sum())))
    S2 = pd.DataFrame(t2rows)
    TRIG = pd.DataFrame(trig)
    S2.to_csv(TAB / "test2_block_lapse.csv", index=False)
    TRIG.to_csv(TAB / "test2_quit_triggered_exposure.csv", index=False)
    t2_tab = _print_animal_table("TEST 2 lapse(after far-contra block) - lapse(after other non-ref block)", S2, "contrast")
    _print_animal_table("TEST 2 lapse after far-contra block", S2, "lapse_far")
    _print_animal_table("TEST 2 lapse after other non-ref block", S2, "lapse_other")
    P2 = _delta(S2, "contrast", rng)
    t2_sens = {"2s": _delta(S2, "contrast_2s", rng)}
    if len(TRIG):
        _print_animal_table("TEST 2 quit-triggered far-contra excess (16 trials before quit)", TRIG, "excess")

    # giving-up rate vs pre working distribution
    thr_rows, pre_dist = [], {}
    for lab, s in B.items():
        T = s["trials"]
        k = len(T) if s["censored"] else s["quit_idx"]
        ct = np.concatenate([[0.0], np.cumsum(T.coll.to_numpy(float))])
        tt = T.t.to_numpy(float)

        def rate_at(t, tt=tt, ct=ct):
            i1 = np.searchsorted(tt, t, side="left")
            i0 = np.searchsorted(tt, t - RATE_S, side="left")
            return (ct[i1] - ct[i0]) / (RATE_S / 60.0)
        if s["epoch"] == "pre":
            ts = tt[:k][tt[:k] >= RATE_S]
            pre_dist.setdefault(s["animal"], []).extend([rate_at(x) for x in ts])
        if not s["censored"] and s["quit_t"] >= RATE_S:
            thr_rows.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"], giveup=rate_at(s["quit_t"])))
    TH = pd.DataFrame(thr_rows)
    pct = {a: (np.percentile(v, 5), np.percentile(v, 50)) for a, v in pre_dist.items()}
    TH["pre_p5"] = [pct[a][0] for a in TH.animal]
    TH["pre_pctile"] = [100.0 * np.mean(np.asarray(pre_dist[a]) <= g) for a, g in zip(TH.animal, TH.giveup)]
    TH["above_p5"] = (TH.giveup >= TH.pre_p5).astype(int)
    TH.to_csv(TAB / "test2_giveup_rate.csv", index=False)
    print("\n=== PER-ANIMAL TABLE: TEST 2 giving-up rate (µl/min, 5 min before quit) vs pre working distribution ===")
    thr_tab = []
    for a in ANIMALS:
        g = TH[(TH.animal == a) & (TH.epoch != "pre")]
        gp = TH[(TH.animal == a) & (TH.epoch == "pre")]
        thr_tab.append(dict(animal=a, pre_p5=round(pct[a][0], 1), pre_p50=round(pct[a][1], 1),
                            n_post_quits=len(g), post_giveup_med=round(g.giveup.median(), 1) if len(g) else np.nan,
                            post_frac_above_p5=round(g.above_p5.mean(), 2) if len(g) else np.nan,
                            post_pctile_med=round(g.pre_pctile.median(), 1) if len(g) else np.nan,
                            n_pre_quits=len(gp), pre_giveup_med=round(gp.giveup.median(), 1) if len(gp) else np.nan))
    thr_tab = pd.DataFrame(thr_tab)
    print(thr_tab.to_string(index=False), flush=True)

    # trial-level logistic of lapse (descriptive)
    lg = []
    try:
        from sklearn.linear_model import LogisticRegression
        for an in ANIMALS:
            for ep in EPS:
                Xs, ys = [], []
                for s in (B[x] for x in sorted(B) if B[x]["animal"] == an and B[x]["epoch"] == ep):
                    T = s["trials"].reset_index(drop=True)
                    k = len(T) if s["censored"] else s["quit_idx"]
                    far = (T.pos == "far_contra").astype(float).to_numpy()
                    cfail = (T.pos.isin(("close_contra", "far_contra")) & (T.hit == 0)).astype(float).to_numpy()
                    ofail = (T.pos.isin(("far_ipsi", "far_mid")) & (T.hit == 0)).astype(float).to_numpy()
                    ct = np.concatenate([[0.0], np.cumsum(T.coll.to_numpy(float))])
                    tt = T.t.to_numpy(float)
                    for i in range(16, k):
                        if T.pos[i] not in REF:
                            continue
                        i0 = np.searchsorted(tt, tt[i] - RATE_S)
                        Xs.append([far[i - 12:i].mean(), cfail[i - 16:i].sum(), ofail[i - 16:i].sum(),
                                   (ct[i] - ct[i0]) / 5.0, tt[i] / 60.0, ct[i]])
                        ys.append(1 - int(T.hit[i]))
                ys = np.asarray(ys)
                if ys.sum() < 10 or (1 - ys).sum() < 10:
                    continue
                X = np.asarray(Xs, float)
                sd = X.std(0)
                sd[sd == 0] = 1.0
                Z = (X - X.mean(0)) / sd
                m = LogisticRegression(C=1e6, max_iter=2000).fit(Z, ys)
                lg.append(dict(animal=an, epoch=ep, n=len(ys), n_lapse=int(ys.sum()),
                               **{f"b_{n}": float(c) for n, c in zip(
                                   ("far_exposure12", "contra_fail16", "farIM_fail16", "local_rate", "time", "cum_water"),
                                   m.coef_[0])}))
    except Exception as ex:                                            # noqa: BLE001
        print("logistic skipped:", ex)
    LG = pd.DataFrame(lg)
    LG.to_csv(TAB / "test2_lapse_logistic.csv", index=False)
    if len(LG):
        print("\n=== TEST 2 trial-level lapse logistic (standardised coefficients; descriptive) ===")
        print(LG.round(2).to_string(index=False), flush=True)
    FD["t2"] = dict(S2=S2, P2=P2, sens=t2_sens, TRIG=TRIG, TH=TH, pre_dist=pre_dist, pct=pct, LG=LG,
                    thr_tab=thr_tab, table=t2_tab)

    # ---------------------------------------------------------------- TEST 3
    t3 = {}
    for key in ("decel", "ili", "lpb", "lat", "rate"):
        P = _pairs(B, I, key, "prodrome")
        P.to_csv(TAB / f"test3_{key}_pairs.csv", index=False)
        t3[key] = dict(P=P, test=None)
    print("\n=== PER-ANIMAL TABLE: TEST 3 prodrome (10 min before quit) minus time-matched control ===")
    for key in ("decel", "ili", "lpb", "lat", "rate"):
        P = t3[key]["P"]
        g = P.groupby("animal").agg(n=("diff", "size"), quit=("quit", "mean"), control=("control", "mean"),
                                    diff=("diff", "mean"))
        print(f"-- {key}\n{g.round(3).to_string()}", flush=True)
    for key in ("decel", "ili", "lpb", "lat", "rate"):
        t3[key]["test"] = _pair_test(t3[key]["P"], rng)
    Prm = _pairs(B, I, "decel", "prodrome", rate_match=True)
    Pany = _pairs(B, I, "decel", "prodrome", same_epoch=False)
    print("-- decel, controls ALSO rate-matched (+/- 20 %)\n"
          f"{Prm.groupby('animal')['diff'].agg(['size', 'mean']).round(3).to_string() if len(Prm) else 'none'}")
    t3["decel_ratematched"] = dict(P=Prm, test=_pair_test(Prm, rng))
    t3["decel_anyepoch"] = dict(P=Pany, test=_pair_test(Pany, rng))
    t3["traj"] = {k: _trajectory(B, I, k, hi=0) for k in ("decel", "ili", "lpb", "lat")}
    FD["t3"] = t3

    # ---------------------------------------------------------------- TEST 4
    t4 = {}
    print("\n=== PER-ANIMAL TABLE: TEST 4 quit period (10 min after quit) minus time-matched control ===")
    keys4 = [("level", 0), ("level0", 0), ("level_still", 0), ("level_quiet", 0), ("sd", 0), ("run_mean", 0), ("run_frac", 0), ("fr_other", 0),
             ("fr_all", 0), ("band", 0)]
    for key, ar in keys4:
        P = _pairs(B, I, key, "quit", area=ar)
        P.to_csv(TAB / f"test4_{key}_pairs.csv", index=False)
        t4[key] = dict(P=P)
        if len(P):
            g = P.groupby("animal").agg(n=("diff", "size"), quit=("quit", "mean"), control=("control", "mean"),
                                        diff=("diff", "mean"))
            print(f"-- {key}\n{g.round(3).to_string()}", flush=True)
    for key, _ar in keys4:
        t4[key]["test"] = _pair_test(t4[key]["P"], rng)
    t4["level_anyepoch"] = dict(P=(pa := _pairs(B, I, "level", "quit", same_epoch=False)), test=_pair_test(pa, rng))
    areas = {}
    for ai, a in enumerate(AREAS):
        if ai == 0:
            continue
        P = _pairs(B, I, "level", "quit", area=ai)
        areas[a] = dict(P=P, test=_pair_test(P, rng))
    t4["areas"] = areas
    pro = {}
    for key in ("level", "resp", "run_mean", "fr_other"):
        P = _pairs(B, I, key, "prodrome")
        pro[key] = dict(P=P, test=_pair_test(P, rng))
    t4["prodrome"] = pro
    t4["traj"] = {k: _trajectory(B, I, k) for k in ("level", "sd", "resp", "run_mean", "run_frac", "fr_other")}
    FD["t4"] = t4

    # ---------------------------------------------------------------- PRIMARY FAMILY + Holm
    prim = {"P1": P1["early"], "P2": P2["early"], "P3": t3["decel"]["test"], "P4": t4["level"]["test"]}
    ps = np.array([prim[k]["p"] for k in ("P1", "P2", "P3", "P4")], float)
    ph = ak.holm(np.where(np.isfinite(ps), ps, 1.0))
    print("\n=== PRIMARY FAMILY (Holm) ===")
    fam = []
    for (k, v), pholm in zip(prim.items(), ph):
        iv = v["iv"]
        per = v["per"]
        npos = sum(x > 0 for x in per.values())
        nneg = sum(x < 0 for x in per.values())
        fam.append(dict(test=k, point=iv.point if iv else np.nan, lo=iv.lo if iv else np.nan,
                        hi=iv.hi if iv else np.nan, n_animals=iv.n_animals if iv else 0, p=v["p"], p_holm=pholm,
                        animals_pos=npos, animals_neg=nneg,
                        per_animal="; ".join(f"{a} {x:+.3g}" for a, x in per.items())))
    FAM = pd.DataFrame(fam)
    print(FAM.to_string(index=False), flush=True)
    FAM.to_csv(TAB / "primary_family.csv", index=False)
    FD["family"] = FAM

    # secondary cells of P1 / P2 + sensitivities -> one table
    sec = []
    for name, R in (("P1 w_coll", P1), ("P1s delivered", t1_sens["delivered"]), ("P1s h80", t1_sens["h80"]),
                    ("accrual", t1_sens["accrual"]), ("weight_pct", t1_sens["weight"]), ("P2 contrast", P2),
                    ("P2s 2s window", t2_sens["2s"])):
        for c, v in R.items():
            iv = v["iv"]
            sec.append(dict(measure=name, cell=c, point=iv.point if iv else np.nan, lo=iv.lo if iv else np.nan,
                            hi=iv.hi if iv else np.nan, p=v["p"],
                            per_animal="; ".join(f"{a} {x:+.3g}" for a, x in v["per"].items())))
    for name, v in [("P3 decel", t3["decel"]["test"]), ("P3s ili", t3["ili"]["test"]), ("P3s lpb", t3["lpb"]["test"]),
                    ("P3s lat", t3["lat"]["test"]), ("P3s local rate", t3["rate"]["test"]),
                    ("P3s decel rate-matched", t3["decel_ratematched"]["test"]),
                    ("P3s decel any-epoch ctrl", t3["decel_anyepoch"]["test"])] + \
            [(f"P4s {k}", t4[k]["test"]) for k, _ in keys4] + \
            [("P4s level any-epoch ctrl", t4["level_anyepoch"]["test"])] + \
            [(f"P4s area {a}", d["test"]) for a, d in areas.items()] + \
            [(f"prodrome {k}", d["test"]) for k, d in pro.items()]:
        iv = v["iv"]
        sec.append(dict(measure=name, cell="all quits", point=iv.point if iv else np.nan, lo=iv.lo if iv else np.nan,
                        hi=iv.hi if iv else np.nan, p=v["p"],
                        per_animal="; ".join(f"{a} {x:+.3g} (n={v['n'].get(a, 0)})" for a, x in v["per"].items())))
    SEC = pd.DataFrame(sec)
    SEC.to_csv(TAB / "all_contrasts.csv", index=False)
    print("\n=== ALL CONTRASTS (secondary unless in the primary family) ===")
    print(SEC.to_string(index=False, float_format=lambda x: f"{x:.3g}"), flush=True)
    FD["sec"] = SEC

    # ---------------------------------------------------------------- descriptive: licks per responded trial
    from wfield_local import orofacial_cohort as oc_     # the cam4 cohort's gate-named tables (A27)
    pm = pd.read_csv(oc_.cohort_dir("tables") / "position_metrics.csv", dtype={"date": str})
    pm["label"] = pm.animal + "_" + pm.date.str[4:]
    keep = set(B)
    pm = pm[pm.label.isin(keep)].copy()
    pm["epoch"] = [B[x]["epoch"] for x in pm.label]
    daq = []
    for lab, s in B.items():
        T = s["trials"]
        k = len(T) if s["censored"] else s["quit_idx"]
        R = T.iloc[:k]
        R = R[R.hit == 1]
        for p, g in R.groupby("pos"):
            daq.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"], pos=p, licks_per_resp=g.n_post.mean(),
                            n=len(g)))
    FD["lpr"] = dict(lp=pm[["label", "animal", "epoch", "pos", "licks_per_resp"]], daq=pd.DataFrame(daq))
    pd.DataFrame(daq).to_csv(TAB / "licks_per_responded_daq.csv", index=False)

    from figdata import save_fd
    save_fd("disengagement", FD, root=FDIR)
    print(f"figdata -> {FDIR / 'disengagement.pkl'}", flush=True)
    return FD


# =============================================================================================== figures
def _style():
    from wfield_local import orofacial_cohort as oc
    return oc.EPOCH_LINE, oc.ANIMAL_MARKER, oc.POS_ORDER


def _stars():
    from within_lick_correction import reserve_headroom, star_top
    return star_top, reserve_headroom


def _epoch_scatter(ax, S, col, label, cens_col=None):
    """Sessions as open markers (shape = animal, colour = epoch), pooled flat mean + boot_ci as a filled marker."""
    from wfield_local import analysis_kit as ak
    EL, AM, _ = _style()
    rng = np.random.default_rng(SEED)
    for i, ep in enumerate(EPS):
        for j, an in enumerate(ANIMALS):
            g = S[(S.epoch == ep) & (S.animal == an)]
            v = g[col].to_numpy(float)
            x = i - 0.27 + 0.11 * j + np.linspace(-0.03, 0.03, max(len(v), 1))[:len(v)]
            ax.scatter(x, v, marker=AM[an], facecolors="none", edgecolors=EL[ep], s=26, lw=1.0, zorder=3)
            if cens_col is not None:
                c = g[cens_col].to_numpy(bool)
                for xx, vv in zip(x[c], v[c]):           # lower bound: a small up-arrow ABOVE the open marker
                    ax.annotate("↑", (xx, vv), xytext=(0, 4), textcoords="offset points", ha="center",
                                va="bottom", fontsize=7, color=EL[ep], zorder=4)
        iv = ak.boot_ci({a: S.loc[(S.epoch == ep) & (S.animal == a), col].dropna().tolist() for a in ANIMALS
                         if len(S.loc[(S.epoch == ep) & (S.animal == a), col].dropna())}, rng)
        if iv is not None:
            ax.errorbar(i + 0.25, iv.point, yerr=[[iv.point - iv.lo], [iv.hi - iv.point]], fmt="o", color=EL[ep],
                        mfc=EL[ep], ms=7, capsize=3, zorder=5)
    ax.set_xticks(range(len(EPS)))
    ax.set_xticklabels(EPS)
    ax.set_ylabel(label)


def _delta_panel(ax, R, label, primary_cell=None, p_holm=None, cells=("early", "acute", "subacute", "chronic")):
    EL, AM, _ = _style()
    star_top, _ = _stars()
    for i, c in enumerate(cells):
        v = R.get(c)
        if v is None:
            continue
        col = EL["acute"] if c == "early" else EL[c]
        for j, (an, d) in enumerate(sorted(v["per"].items())):
            ax.scatter(i - 0.2 + 0.08 * j, d, marker=AM[an], facecolors="none", edgecolors=col, s=28, lw=1.0, zorder=3)
        iv = v["iv"]
        if iv is not None:
            ax.errorbar(i + 0.22, iv.point, yerr=[[iv.point - iv.lo], [iv.hi - iv.point]], fmt="o", color=col,
                        mfc=col, ms=7, capsize=3, zorder=5)
            sig = (p_holm is not None and c == primary_cell and p_holm < 0.05) or \
                  (c != primary_cell and iv.excludes_zero)
            if sig:
                star_top(ax, i + 0.22, "*" if c == primary_cell else "+")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([("early post\n(PRIMARY)" if c == primary_cell else "early post\n(acute+subacute)")
                        if c == "early" else c for c in cells])
    ax.set_ylabel(label)


def _pair_panel(ax, items, ylabel, primary_holm=None):
    """items: [(name, test dict, P frame)] -> per-animal mean diffs (open) + boot_delta CI (filled)."""
    EL, AM, _ = _style()
    star_top, _ = _stars()
    for i, (_nm, v, _P) in enumerate(items):
        for j, (an, d) in enumerate(sorted(v["per"].items())):
            ax.scatter(i - 0.2 + 0.08 * j, d, marker=AM[an], facecolors="none", edgecolors="k", s=28, lw=1.0, zorder=3)
        iv = v["iv"]
        if iv is not None:
            ax.errorbar(i + 0.22, iv.point, yerr=[[iv.point - iv.lo], [iv.hi - iv.point]], fmt="o", color="k",
                        ms=7, capsize=3, zorder=5)
            if i == 0 and primary_holm is not None:
                if primary_holm < 0.05:
                    star_top(ax, i + 0.22, "*")
            elif iv.excludes_zero:
                star_top(ax, i + 0.22, "+")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(items)))
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.set_xticklabels([nm for nm, _v, _P in items], fontsize=8)
    ax.set_ylabel(ylabel)


def _traj_panel(ax, tr, ylabel, quit_line=True):
    _, AM, _ = _style()
    x = tr["x_min"]
    M = []
    for an in ANIMALS:
        c = tr["curves"].get(an)
        if c is None:
            continue
        ax.plot(x, c, color="0.6", lw=0.8, marker=AM[an], mfc="none", ms=4)
        M.append(c)
    if M:
        M = np.vstack(M)
        n = np.isfinite(M).sum(0)
        mu = np.nanmean(M, 0)
        se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.maximum(n, 1)) if M.shape[0] > 1 else np.zeros_like(mu)
        ax.plot(x, mu, color="k", lw=2)
        ax.fill_between(x, mu - se, mu + se, color="k", alpha=0.15, lw=0)
    ax.axhline(0, color="0.6", lw=0.8)
    if quit_line:
        ax.axvline(0, color="#d01c8b", lw=1, ls="--")
    ax.set_xlabel("min from quit onset")
    ax.set_ylabel(ylabel)


def _legend_animals(fig, epochs=True):
    from matplotlib.lines import Line2D
    EL, AM, _ = _style()
    h = [Line2D([], [], marker=AM[a], ls="", mfc="none", mec="k", label=a) for a in ANIMALS]
    if epochs:
        h += [Line2D([], [], marker="o", ls="", mfc=EL[e], mec=EL[e], label=e) for e in EPS]
    else:
        h += [Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="mean over animals, 95 % CI (all epochs pooled)")]
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    _, reserve = _stars()
    reserve(fig)
    fig.savefig(FIG / f"{name}.png", dpi=140)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("  ->", FIG / f"{name}.png", flush=True)


def plot(FD=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if FD is None:
        from figdata import load_fd
        FD = load_fd("disengagement", root=FDIR)
    EL, AM, POS = _style()
    fam = FD["family"].set_index("test")

    # 1 satiety
    t1 = FD["t1"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    _epoch_scatter(ax[0], t1["S"], "w_coll", "collected water at quit (or session end), µl", cens_col="censored")
    ax[0].set_title("water collected before the quit\n(arrow = no quit: lower bound)", fontsize=10)
    _delta_panel(ax[1], t1["P1"], "change vs own August pre, µl", primary_cell="early", p_holm=fam.loc["P1", "p_holm"])
    ax[1].set_title(f"P1 collected water (Holm p = {fam.loc['P1', 'p_holm']:.3g})", fontsize=10)
    _delta_panel(ax[2], t1["sens"]["delivered"], "change vs pre, µl (delivered)")
    ax[2].set_title("delivered water (secondary)", fontsize=10)
    fig.suptitle("BEHAVIOUR -- Test 1 satiety: water before the quit, pre vs post", fontsize=12)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, "t1_satiety_water")

    fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, an in enumerate(ANIMALS):
        for ep in EPS:
            k = t1["km"].get((an, ep))
            if k is None:
                continue
            xm = max(k["x"].max(), k["cens_x"].max() if k["cens_x"].size else 0)
            ax[i].step(np.append(k["x"], xm), np.append(k["y"], k["y"][-1]), where="post", color=EL[ep], lw=1.6,
                       label=ep)
            if k["cens_x"].size:
                yv = k["y"][np.searchsorted(k["x"], k["cens_x"], side="right") - 1]   # step value, not linear
                ax[i].scatter(k["cens_x"], yv, marker="|", color=EL[ep], s=40)
        ax[i].set_title(an, fontsize=10)
        ax[i].set_xlabel("cumulative collected water, µl")
    ax[0].set_ylabel("fraction of sessions not yet quit")
    ax[-1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("BEHAVIOUR -- Test 1: Kaplan-Meier on the water scale (| = session ended without a quit)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "t1_km_water")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    _epoch_scatter(ax[0], t1["S"], "accrual", "collected water / min to quit-or-end")
    ax[0].set_title("water accrual rate", fontsize=10)
    _epoch_scatter(ax[1], t1["S"], "weight_pct", "pre-session body weight, % August pre")
    ax[1].set_title("body weight", fontsize=10)
    fig.suptitle("BEHAVIOUR -- Test 1: water accrual and body weight", fontsize=12)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    _save(fig, "t1_accrual_weight")

    # 2 frustration
    t2 = FD["t2"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    _epoch_scatter(ax[0], t2["S2"], "contrast", "lapse after far-contra block - after other block")
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_title("reference-spout lapses after a scheduled far-contra block", fontsize=10)
    _delta_panel(ax[1], t2["P2"], "change vs own August pre", primary_cell="early", p_holm=fam.loc["P2", "p_holm"])
    ax[1].set_title(f"P2 (Holm p = {fam.loc['P2', 'p_holm']:.3g})", fontsize=10)
    TR = t2["TRIG"]
    if len(TR):
        _epoch_scatter(ax[2], TR, "excess", "far-contra share, 16 trials before quit - scheduled")
        ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].set_title("do quits follow far-contra runs? (one point per quit)", fontsize=10)
    fig.suptitle("BEHAVIOUR -- Test 2 frustration: scheduled failure runs and disengagement", fontsize=12)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save(fig, "t2_frustration_blocks")

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2), sharey=False)
    for i, an in enumerate(ANIMALS):
        dist = np.asarray(t2["pre_dist"].get(an, []))
        if dist.size:
            ax[i].hist(dist, bins=30, color="0.8", density=True)
            p5, p50 = t2["pct"][an]
            ax[i].axvline(p5, color="k", ls=":", lw=1)
            ax[i].axvline(p50, color="k", ls="-", lw=1)
        g = t2["TH"][t2["TH"].animal == an]
        ymax = ax[i].get_ylim()[1]
        for j, ep in enumerate(EPS):
            v = g[g.epoch == ep].giveup.to_numpy(float)
            ax[i].scatter(v, np.full(v.size, ymax * (0.55 + 0.1 * j)), marker=AM[an], facecolors="none",
                          edgecolors=EL[ep], s=30)
        ax[i].set_title(f"{an}  (dotted = pre 5 %, solid = 50 %)", fontsize=9)
        ax[i].set_xlabel("collected µl / min, previous 5 min")
    ax[0].set_ylabel("pre working distribution (density)")
    fig.suptitle("BEHAVIOUR -- Test 2 foraging threshold: giving-up rate of each quit (markers, rows = epoch) on the "
                 "animal's pre-stroke working rates", fontsize=11)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    _save(fig, "t2_giveup_threshold")

    # 3 fatigue
    t3 = FD["t3"]
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    for i, (k, lab) in enumerate((("decel", "within-bout deceleration, ms"), ("ili", "within-bout ILI, ms"),
                                  ("lpb", "licks per bout"), ("lat", "first-lick latency, s"))):
        _traj_panel(ax[0, i], t3["traj"][k], f"quit - control: {lab}")
        ax[0, i].set_title(k + " (reference spouts, responded)", fontsize=10)
    _pair_panel(ax[1, 0], [("decel\n(P3)", t3["decel"]["test"], None),
                           ("decel\nrate-matched", t3["decel_ratematched"]["test"], None),
                           ("decel\nany-epoch ctrl", t3["decel_anyepoch"]["test"], None)], "prodrome - control, ms",
                primary_holm=fam.loc["P3", "p_holm"])
    ax[1, 0].set_title(f"P3 (Holm p = {fam.loc['P3', 'p_holm']:.3g})", fontsize=10)
    _pair_panel(ax[1, 1], [("ILI", t3["ili"]["test"], None)], "prodrome - control, ms")
    _pair_panel(ax[1, 2], [("licks/bout", t3["lpb"]["test"], None)], "prodrome - control")
    _pair_panel(ax[1, 3], [("latency", t3["lat"]["test"], None), ], "prodrome - control, s")
    fig.suptitle("BEHAVIOUR -- Test 3 motor fatigue: lick motor measures before the quit vs time-matched controls "
                 "(top: grey = animals, black = mean +/- SEM over animals; + = 95 % CI excludes 0, * = Holm primary)", fontsize=11)
    _legend_animals(fig, epochs=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, "t3_fatigue_prodrome")

    # 4 state, neural
    t4 = FD["t4"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    _traj_panel(ax[0, 0], t4["traj"]["level"], "quit - control, pre-cue level (RESTnl)")
    ax[0, 0].set_title("global (cortex) pre-cue level", fontsize=10)
    _traj_panel(ax[0, 1], t4["traj"]["sd"], "quit - control, trial-to-trial SD")
    ax[0, 1].set_title("global pre-cue level: trial-to-trial SD", fontsize=10)
    _traj_panel(ax[0, 2], t4["traj"]["resp"], "quit - control, cue response")
    ax[0, 2].set_title("global cue response (reference spouts)", fontsize=10)
    _pair_panel(ax[1, 0], [("level\n(P4)", t4["level"]["test"], None),
                           ("any-epoch\nctrl", t4["level_anyepoch"]["test"], None),
                           ("no-basis\nwithin-sess", t4["level0"]["test"], None),
                           ("still\n(post hoc)", t4["level_still"]["test"], None),
                           ("still, no lick\n(post hoc)", t4["level_quiet"]["test"], None),
                           ("SD", t4["sd"]["test"], None)], "quit period - control",
                primary_holm=fam.loc["P4", "p_holm"])
    ax[1, 0].set_title(f"P4 (Holm p = {fam.loc['P4', 'p_holm']:.3g})", fontsize=10)
    ar = t4["areas"]
    _pair_panel(ax[1, 1], [(a.replace("_right", "\ncontra").replace("_left", "\nipsi"), d["test"], None)
                           for a, d in ar.items()], "quit period - control (RESTnl)")
    ax[1, 1].set_title("per area (contra = right hemisphere)", fontsize=10)
    _pair_panel(ax[1, 2], [(k, d["test"], None) for k, d in t4["prodrome"].items()], "prodrome - control")
    ax[1, 2].set_title("10 min BEFORE the quit", fontsize=10)
    fig.suptitle("NEURAL -- Test 4 state: cortex in the quit period vs time-matched engaged controls", fontsize=12)
    _legend_animals(fig, epochs=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, "t4_state_neural")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for i, (k, lab) in enumerate((("run_mean", "treadmill speed, mm/s"), ("run_frac", "fraction of time running"),
                                  ("fr_other", "FaceRhythm 'other' factors (between trials)"))):
        _traj_panel(ax[0, i], t4["traj"][k], f"quit - control: {lab}")
        ax[0, i].set_title(lab, fontsize=10)
    _pair_panel(ax[1, 0], [("speed (mm/s)", t4["run_mean"]["test"], None)], "quit period - control, mm/s")
    _pair_panel(ax[1, 1], [("FR other", t4["fr_other"]["test"], None), ("FR all", t4["fr_all"]["test"], None)],
                "quit period - control")
    _pair_panel(ax[1, 2], [("0.01-0.05 Hz\nglobal power", t4["band"]["test"], None)], "quit period - control")
    ax[1, 2].set_title("NEURAL global 0.01-0.05 Hz power: rises although evoked\nresponses stop (running transients may contribute)",
                       fontsize=9)
    fig.suptitle("BEHAVIOUR -- Test 4 state: movement in the quit period vs time-matched engaged controls "
                 "(no pupil measure exists)", fontsize=12)
    _legend_animals(fig, epochs=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, "t4_state_movement")

    # 8 licks per responded trial
    L = FD["lpr"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for axi, (src, D) in zip(ax, (("cam4 LP licks (orofacial_cohort licks_per_resp)", L["lp"]),
                                  ("DAQ spout contacts in the task window", L["daq"]))):
        for pi, p in enumerate(POS):
            for ei, ep in enumerate(EPS):
                g = D[(D.pos == p) & (D.epoch == ep)]
                x0 = pi + (ei - 1.5) * 0.2
                for j, an in enumerate(ANIMALS):
                    v = g[g.animal == an].licks_per_resp.to_numpy(float)
                    axi.scatter(x0 - 0.06 + 0.03 * j + np.zeros(v.size), v, marker=AM[an], facecolors="none",
                                edgecolors=EL[ep], s=12, lw=0.7, alpha=0.8)
                if len(g):
                    axi.scatter([x0 + 0.07], [g.licks_per_resp.mean()], marker="o", color=EL[ep], s=40, zorder=5)
        axi.set_xticks(range(len(POS)))
        axi.set_xticklabels([p.replace("_", "\n") for p in POS])
        axi.set_ylabel("licks per responded trial")
        axi.set_title(src, fontsize=10)
    fig.suptitle("BEHAVIOUR -- licks per responded trial by spout and epoch (engaged trials; DESCRIPTIVE)", fontsize=12)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save(fig, "licks_per_responded_trial")


# =============================================================================================== ADDENDUM 10-31
# Pre-registered in the prereg's "ADDENDUM 10-31 (follow-up)" (A0-A6) BEFORE it was run. Family A (Holm) = {P5, P6,
# P8}; family B (Holm) = {P7a, P7b, P7c}. Uses only the per-session caches written by `extract`.
OROF = ("tjM1_right", "ALM_right", "SSp-m_right", "SSp-n_right", "tjM1_left", "ALM_left", "SSp-m_left", "SSp-n_left")
REFC = ("VISpm_right", "VISpm_left", "RSPd_right", "RSPd_left")
CAP_S = 5400.0                           # P5 restriction (90 min)
H80_S = 4800.0
CAP_TR = 300                             # A7 common trial cap
POS_ORDER_CANON = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
UNCANON = {v: k for k, v in CANON.items()}


def end_class(s) -> dict:
    """A0 rule: Q detected quit | W water limit | T rule / time | H hand-stopped at a collapse | U unknown."""
    T = s["trials"]
    R = T[T.pos.isin(REF)]
    last5 = float(R.tail(5).hit.mean()) if len(R) else np.nan
    trail = 0
    for h in R.hit.to_numpy()[::-1]:
        if h:
            break
        trail += 1
    all12 = float(T.tail(12).hit.mean())
    term = s["term"] or ""
    manual = not any(x in term for x in ("max_duration_reached", "target_trials_reached", "water_limit_reached"))
    if not s["censored"]:
        c = "Q"
    elif "water_limit_reached" in term:
        c = "W"
    elif not manual or s["label"] == "PS92_0812":
        c = "T"
    elif (last5 <= 0.4) or (trail >= 2) or (all12 <= 0.5):
        c = "H"
    elif last5 >= 0.8 and trail == 0:
        c = "T"
    else:
        c = "U"
    # A7: post-session end class from the logs (100 per spout or 120 min, whichever first)
    per = T.pos.value_counts()
    capped = ("target_trials_reached" in term) or (len(per) == 6 and int(per.min()) >= 100)
    timed = ("max_duration_reached" in term) or (manual and s["dur"] >= 118 * 60)
    stop = "trial-cap" if capped else ("120-min limit" if timed else "other")
    post_end = ("quit" if not s["censored"] else stop) if s["epoch"] != "pre" else ""
    return dict(label=s["label"], animal=s["animal"], epoch=s["epoch"], end_class=c, post_end=post_end,
                stop_rule=stop, manual_stop=int(manual),
                term=term or "manual", dur_min=round(s["dur"] / 60, 1),
                quit_min=round(s["quit_t"] / 60, 1) if not s["censored"] else np.nan,
                n_trials=len(T), ref_last5=round(last5, 2), trailing_ref_miss=trail, all_last12=round(all12, 2),
                **{f"n_{p}": int(per.get(p, 0)) for p in CANON.values()})


def _redetect(T, cut_s):
    """Quit onset on the record cut at cut_s (the `quit_point --horizon-min` logic; detector unchanged)."""
    from wfield_local.precue_engagement_states import engagement_gate
    C = T[T.t <= cut_s]
    if len(C) < 10:
        return None
    from wfield_local import trial_classes as _tc
    with _tc.daq_quit_rule("disengagement_tests: quit onset (a measure of engagement)"):
        ne = np.asarray(engagement_gate(C.order.to_numpy(), C.hit2.astype(bool).to_numpy(),
                                        np.array([UNCANON.get(p, p) for p in C.pos])), bool)
    if not ne.any():
        return dict(censored=True, k=len(C))
    k = len(ne)
    while k > 0 and ne[k - 1]:
        k -= 1
    if k == 0 or k >= len(C):
        return None
    return dict(censored=False, k=k, t=float(C.t.iloc[k]))


def _run_bins(sess):
    r = sess["run"]
    return (r["mean"], r["frac"]) if "t" in r else (None, None)


def _bin_mean(arr, a, b):
    if arr is None:
        return np.nan
    i0, i1 = int(max(np.floor(a), 0)), int(max(np.floor(b), 0))
    x = arr[i0:i1]
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else np.nan


def _prep_imaging(B, I):
    """Per-trial arrays for P6 / RESTW / P7c, stored into I[label]."""
    import iti_licks as il
    for lab, im in I.items():
        s = B.get(lab)
        if s is None:
            continue
        cue_t = im["cue_s"] - im["cue_s"][0]
        sp, fr = _run_bins(s)
        sp2 = np.array([_bin_mean(sp, c - 2, c) for c in cue_t])
        fr2 = np.array([_bin_mean(fr, c - 2, c) for c in cue_t])
        sp10 = np.array([_bin_mean(sp, c - 12, c - 2) for c in cue_t])
        q = (cue_t >= s["quit_t"]) if not s["censored"] else np.zeros(cue_t.size, bool)
        lev = im["pre_b"] - im["lev"][None, :] if bool(im["lev_ok"]) else np.full_like(im["pre_b"], np.nan)
        im["lev_nl"] = lev
        im["sp2"], im["q"] = sp2, q
        # RESTW reference
        try:
            z = np.load(il.OUT / "precue_restw" / f"{lab}.npz")
            cs = [str(c) for c in z["cols"]]
            rw = np.array([float(z["restw"][cs.index(a)]) if a in cs else np.nan for a in AREAS])
            ok = bool(np.asarray(z["restw_ok"]).all()) if np.asarray(z["restw_ok"]).size else False
            im["lev_rw"] = im["pre_b"] - rw[None, :] if ok or np.isfinite(rw).all() else np.full_like(lev, np.nan)
        except Exception:                                              # noqa: BLE001
            im["lev_rw"] = np.full_like(lev, np.nan)
        for src, dst in (("lev_nl", "res_nl"), ("lev_rw", "res_rw")):
            Y = im[src]
            Xr = np.column_stack([sp2, fr2, sp10])
            ok = np.isfinite(Xr).all(1) & np.isfinite(Y[:, 0])
            res = np.full_like(Y, np.nan)
            res_e = np.full_like(Y, np.nan)
            if ok.sum() >= 20:
                X = np.column_stack([np.ones(ok.sum()), Xr[ok], q[ok].astype(float)])
                bta = np.linalg.lstsq(X, Y[ok], rcond=None)[0]
                res[ok] = Y[ok] - Xr[ok] @ bta[1:4]
                eg = ok & ~q
                if eg.sum() >= 20:
                    Xe = np.column_stack([np.ones(eg.sum()), Xr[eg]])
                    be_ = np.linalg.lstsq(Xe, Y[eg], rcond=None)[0]
                    res_e[ok] = Y[ok] - Xr[ok] @ be_[1:4]
            im[dst] = res
            im[dst + "_eng"] = res_e
        # P7c: z-scored level vs pre-cue speed, per area
        ok = np.isfinite(sp2) & np.isfinite(im["pre_b"]).all(1)
        if ok.sum() >= 30 and np.std(sp2[ok]) > 0:
            Z = (im["pre_b"][ok] - im["pre_b"][ok].mean(0)) / im["pre_b"][ok].std(0)
            x = (sp2[ok] - sp2[ok].mean())
            im["run_beta"] = (x @ Z) / (x @ x)
        else:
            im["run_beta"] = np.full(len(AREAS), np.nan)


def _arr_val(I, lab, name, a, b, area=0, still=None):
    im = I.get(lab)
    if im is None or name not in im:
        return np.nan
    cue_t = im["cue_s"] - im["cue_s"][0]
    m = (cue_t >= a) & (cue_t < b)
    if still is not None:
        m &= np.isfinite(im["sp2"]) & (im["sp2"] < 3.0)
    x = im[name][m, area]
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size >= MIN_TRIALS_WIN else np.nan


def _pairs_arr(B, I, name, area=0, still=None, same_epoch=True):
    rows = []
    for lab, sq in B.items():
        if sq["censored"]:
            continue
        a, b = sq["quit_t"], min(sq["quit_t"] + WIN_S, sq["dur"])
        if b - a < 300.0:
            continue
        vq = _arr_val(I, lab, name, a, b, area, still)
        if not np.isfinite(vq):
            continue
        cv = [v for lc, sc in B.items() if lc != lab and sc["animal"] == sq["animal"]
              and (not same_epoch or sc["epoch"] == sq["epoch"]) and _engaged_through(sc, b)
              for v in [_arr_val(I, lc, name, a, b, area, still)] if np.isfinite(v)]
        if cv:
            rows.append(dict(label=lab, animal=sq["animal"], epoch=sq["epoch"], quit=vq, control=float(np.mean(cv)),
                             n_ctrl=len(cv), diff=vq - float(np.mean(cv))))
    return pd.DataFrame(rows)


def _periods(binary, min_len=2):
    """[(start, end)) runs of True in a 1-s bin boolean array, length >= min_len."""
    out, i, n = [], 0, binary.size
    while i < n:
        if binary[i]:
            j = i
            while j < n and binary[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _test4b_session(s):
    """P7a / P7b / descriptive running for one session's quit period."""
    sp, _fr = _run_bins(s)
    if sp is None or s["censored"]:
        return None
    T = s["trials"]
    tq, te = s["quit_t"], s["dur"]
    fin = np.isfinite(sp)
    run = fin & (sp > 3.0)
    still = fin & (sp <= 3.0)
    lo, hi = int(np.ceil(tq)), int(min(np.floor(te), sp.size))
    if hi - lo < 120:
        return None
    rb = [(a + lo, b + lo) for a, b in _periods(run[lo:hi])]
    sb = [(a + lo, b + lo) for a, b in _periods(still[lo:hi])]
    cue = T.t.to_numpy(float)
    hit = T.hit.to_numpy(int)

    def nxt(t_end):
        j = np.searchsorted(cue, t_end, side="left")
        return int(hit[j]) if j < cue.size and cue[j] - t_end <= 60.0 else None
    after_run, after_still = [], []
    for a, b in rb:
        d = b - a
        r = nxt(b)
        if r is None:
            continue
        # matched still window: duration d inside a still period, ending >= 5 s before that period ends, nearest in time
        best = None
        for sa, sbb in sb:
            if sbb - sa >= d + 5:
                end = min(max(sa + d, a), sbb - 5)          # as close to the bout as the still period allows
                dist = abs(end - b)
                if best is None or dist < best[0]:
                    best = (dist, end)
        if best is None:
            continue
        rs = nxt(best[1])
        if rs is None:
            continue
        after_run.append(r)
        after_still.append(rs)
    # P7b: trials by their preceding 10 s
    pr_q, ps_q, pr_e, ps_e, lat_r, lat_s = [], [], [], [], [], []
    for c, h, la in zip(cue, hit, T.lat.to_numpy(float)):
        k0, k1 = int(np.floor(c - 10)), int(np.floor(c))
        if k0 < 0 or k1 > sp.size:
            continue
        w = sp[k0:k1]
        if not np.isfinite(w).all():
            continue
        isr, iss = (w > 3.0).sum() >= 1, (w <= 3.0).all()
        inq = c >= tq
        if isr:
            (pr_q if inq else pr_e).append(h)
            if not inq and h:
                lat_r.append(la)
        elif iss:
            (ps_q if inq else ps_e).append(h)
            if not inq and h:
                lat_s.append(la)
    durq = (hi - lo) / 60.0
    return dict(label=s["label"], animal=s["animal"], epoch=s["epoch"], n_runbouts=len(rb), n_pairs=len(after_run),
                p_after_run=np.mean(after_run) if len(after_run) >= 3 else np.nan,
                p_after_still=np.mean(after_still) if len(after_run) >= 3 else np.nan,
                n_q_run=len(pr_q), n_q_still=len(ps_q),
                pq_run=np.mean(pr_q) if len(pr_q) >= 3 else np.nan, pq_still=np.mean(ps_q) if len(ps_q) >= 3 else np.nan,
                pe_run=np.mean(pr_e) if len(pr_e) >= 3 else np.nan, pe_still=np.mean(ps_e) if len(ps_e) >= 3 else np.nan,
                late_run=np.median(lat_r) if len(lat_r) >= 3 else np.nan,
                late_still=np.median(lat_s) if len(lat_s) >= 3 else np.nan,
                q_runfrac=float(run[lo:hi].mean()), q_bouts_per_min=len(rb) / durq,
                q_bout_s=float(np.mean([b - a for a, b in rb])) if rb else 0.0)


def _rmst(x, e, tau):
    """Kaplan-Meier restricted mean of x up to tau (e = 1 event, 0 censored)."""
    o = np.argsort(x)
    x, e = np.asarray(x, float)[o], np.asarray(e, int)[o]
    t_prev, s, area, at = 0.0, 1.0, 0.0, len(x)
    for xi, ei in zip(x, e):
        if xi >= tau:
            break
        if ei:
            area += s * (xi - t_prev)
            t_prev = xi
            s *= (at - 1) / at
        at -= 1
    return area + s * (tau - t_prev)


def _rmst_delta(S, col, rng, cells=("early", "acute", "subacute", "chronic"), n_boot=None):
    """KM restricted-mean difference (post cell - pre) per animal, averaged over animals, with the animals ->
    sessions resample. A KM statistic cannot go through `analysis_kit.boot_delta` (which reduces by MEAN), so this is
    the same draw ORDER (sorted animals; draw animals, then sessions within each arm) with a KM reduction -- the
    `matrix_bootstrap` exception documented in analysis_kit, not a ninth copy of the mean bootstrap."""
    from wfield_local import analysis_kit as ak
    n_boot = n_boot or ak.N_BOOT
    out = {}
    for c in cells:
        post_m = S.epoch.isin(("acute", "subacute")) if c == "early" else (S.epoch == c)
        arms = {}
        for a in sorted(ANIMALS):
            P = S[(S.animal == a) & (S.epoch == "pre")]
            Q = S[(S.animal == a) & post_m]
            if len(P) and len(Q):
                tau = min(P[col].max(), Q[col].max())
                arms[a] = (P[col].to_numpy(float), P.ev_primary.to_numpy(int), Q[col].to_numpy(float),
                           Q.ev_primary.to_numpy(int), float(tau))
        if not arms:
            out[c] = dict(iv=None, p=np.nan, per={}, per_tau={})
            continue
        per = {a: _rmst(v[2], v[3], v[4]) - _rmst(v[0], v[1], v[4]) for a, v in arms.items()}
        an = sorted(arms)
        draws = []
        for _ in range(n_boot):
            d = []
            for a in (an[i] for i in rng.integers(0, len(an), len(an))):
                px, pe, qx, qe, tau = arms[a]
                i1 = rng.integers(0, len(px), len(px))
                i2 = rng.integers(0, len(qx), len(qx))
                d.append(_rmst(qx[i2], qe[i2], tau) - _rmst(px[i1], pe[i1], tau))
            draws.append(float(np.mean(d)))
        dr = np.asarray(draws)
        iv = ak.Interval(float(np.mean(list(per.values()))), float(np.percentile(dr, 2.5)),
                         float(np.percentile(dr, 97.5)), len(an))
        out[c] = dict(iv=iv, p=ak.boot_p(dr), per=per, per_tau={a: (per[a], arms[a][4]) for a in an})
    return out


def _paired(df, a, b, rng):
    from wfield_local import analysis_kit as ak
    d = df[np.isfinite(df[a]) & np.isfinite(df[b])]
    x = {an: d.loc[d.animal == an, a].tolist() for an in ANIMALS if (d.animal == an).any()}
    y = {an: d.loc[d.animal == an, b].tolist() for an in ANIMALS if (d.animal == an).any()}
    iv, dr = ak.boot_delta(x, y, rng, draws=True)
    per = {an: float(np.mean(x[an]) - np.mean(y[an])) for an in x}
    return dict(iv=iv, p=ak.boot_p(dr) if iv is not None else np.nan, per=per, n={an: len(v) for an, v in x.items()})


def addendum():
    from wfield_local import analysis_kit as ak
    B, I = _load_all()
    rng = np.random.default_rng(SEED + 1)
    FD = {}

    # ---------------- A0
    A0 = pd.DataFrame([end_class(B[k]) for k in sorted(B)])
    A0.to_csv(TAB / "a0_end_class.csv", index=False)
    print("\n=== A0 END CLASS per animal x epoch (counts) ===")
    print(pd.crosstab([A0.animal, A0.epoch], A0.end_class).to_string(), flush=True)
    print(A0[A0.epoch == "pre"].to_string(index=False), flush=True)
    cls = dict(zip(A0.label, A0.end_class))

    # ---------------- A2: time / water survival + P5
    rows = []
    for lab, s in B.items():
        T = s["trials"]
        c = cls[lab]
        t_end = s["quit_t"] if c == "Q" else s["dur"]
        k = s["quit_idx"] if c == "Q" else len(T)
        w = T.iloc[:k].coll.sum()
        ev = {"primary": c in ("Q", "H"), "b_i": c in ("Q", "H", "U"), "b_ii": c == "Q"}
        cap = {"primary": min(t_end, CAP_S), "b_i": min(t_end, CAP_S),
               "b_ii": (CAP_S if (c in ("H", "U") and t_end < CAP_S) else min(t_end, CAP_S))}
        n_tr = int(k)
        n_resp = int(T.iloc[:k].hit.sum())
        short = (c != "Q") and n_tr < CAP_TR                       # censored below the trial cap
        prog = T.iloc[:k].pos.value_counts()
        rows.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"], cls=c, t_min=t_end / 60, water=w,
                         n_tr=n_tr, n_resp=n_resp,
                         ev_primary=int(ev["primary"]), ev_b_i=int(ev["b_i"]), ev_b_ii=int(ev["b_ii"]),
                         cap90_primary=cap["primary"] / 60, cap90_b_i=cap["b_i"] / 60, cap90_b_ii=cap["b_ii"] / 60,
                         cap300_primary=min(n_tr, CAP_TR), cap300_bound=(CAP_TR if short else min(n_tr, CAP_TR)),
                         **{f"prog_{p}": float(prog.get(p, 0)) / 100.0 for p in CANON.values()}))
    S = pd.DataFrame(rows)
    S.to_csv(TAB / "a2_time_water.csv", index=False)
    _print_animal_table("A2 time to quit-or-end, min (PS92/PS93 pre = time-censored lower bounds)", S, "t_min")
    _print_animal_table("A2 P5 engaged time capped at 90 min (primary reading)", S, "cap90_primary")
    _print_animal_table("A7 trials to quit-or-end (all trials)", S, "n_tr")
    _print_animal_table("A7 P5t trials capped at 300 (primary reading)", S, "cap300_primary")
    _print_animal_table("A7 responded trials to quit-or-end", S, "n_resp")
    P5 = _delta(S, "cap90_primary", rng)
    P5t = _delta(S, "cap300_primary", rng)
    P5s = {"b_i": _delta(S, "cap90_b_i", rng), "b_ii": _delta(S, "cap90_b_ii", rng), "uncapped": _delta(S, "t_min", rng),
           "trials bound": _delta(S, "cap300_bound", rng), "trials uncapped": _delta(S, "n_tr", rng),
           "responded trials": _delta(S, "n_resp", rng)}
    print("\n=== A7 per-spout progress at quit-or-end (fraction of 100), animal x epoch means ===")
    print(S.groupby(["animal", "epoch"])[[f"prog_{p}" for p in POS_ORDER_CANON]].mean().round(2).to_string(), flush=True)
    km = {}
    for (an, ep), g in S.groupby(["animal", "epoch"]):
        for scale in ("t_min", "water", "n_tr"):
            for rd in ("primary", "b_i", "b_ii"):
                x = g[scale].to_numpy(float)
                e = g[f"ev_{rd}"].to_numpy(int)
                o = np.argsort(x)
                x, e = x[o], e[o]
                sv, at, xs, ys = 1.0, len(x), [0.0], [1.0]
                for xi, ei in zip(x, e):
                    if ei:
                        sv *= (at - 1) / at
                        xs.append(xi)
                        ys.append(sv)
                    at -= 1
                km[(an, ep, scale, rd)] = dict(x=np.array(xs), y=np.array(ys), xmax=float(x.max()),
                                                cens=[(xi, cl) for xi, ei, cl in zip(x, e, g.cls.to_numpy()[o]) if not ei])
    # UNCAPPED with bounds (Priya 10-31, AFTER seeing the capped results; outside the Holm families):
    #   lower bound = every censored end treated as the quit (the raw value; = _delta on the column)
    #   censoring-aware = KM restricted mean to a per-animal tau (primary reading: Q, H = quit; T/W/U/120-min/
    #   trial-cap censored), tau = the smallest of the two groups' largest observations, so both groups are observed
    #   to tau and the pre long-session policy cannot inflate pre.
    unc = {}
    rng_u = np.random.default_rng(SEED + 2)        # own stream: adding this block must not move any earlier CI
    for col in ("t_min", "n_tr", "n_resp"):
        _print_animal_table(f"UNCAPPED lower bound ({col})", S, col)
        unc[col] = dict(lower=_delta(S, col, rng_u), km=_rmst_delta(S, col, rng_u))
        print(f"-- KM restricted mean ({col}), per animal [tau]:", flush=True)
        for c, v in unc[col]["km"].items():
            print(f"   {c}: " + "; ".join(f"{a} {d:+.1f} [tau {t:.0f}]" for a, (d, t) in v["per_tau"].items()),
                  flush=True)
    FD["a2"] = dict(S=S, P5=P5, P5t=P5t, sens=P5s, km=km, unc=unc)

    # ---------------- A3iii exact 80-min horizon
    h = []
    for lab, s in B.items():
        if s["dur"] < H80_S:
            continue
        T = s["trials"]
        q = _redetect(T, H80_S)
        if q is None:
            continue
        if q["censored"]:
            h.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"], quit80=0, t80=H80_S / 60,
                          w80=T[T.t <= H80_S].coll.sum()))
        else:
            h.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"], quit80=1, t80=q["t"] / 60,
                          w80=T.iloc[:q["k"]].coll.sum()))
    H = pd.DataFrame(h)
    H.to_csv(TAB / "a3iii_h80.csv", index=False)
    print("\n=== A3iii quits re-detected on records cut at 80 min (quits / sessions) ===")
    print(H.groupby(["animal", "epoch"]).quit80.agg(["sum", "count"]).unstack().to_string(), flush=True)
    _print_animal_table("A3iii water at quit-or-80 min", H, "w80")
    _print_animal_table("A3iii time at quit-or-80 min", H, "t80")
    FD["a3iii"] = dict(H=H, water=_delta(H, "w80", rng), time=_delta(H, "t80", rng))

    # ---------------- A1 normalised licks per responded trial
    from figdata import load_fd
    LPR = load_fd("disengagement", root=FDIR)["lpr"]
    A1 = {}
    for src in ("lp", "daq"):
        D = LPR[src].copy()
        base = D[D.epoch == "pre"].groupby(["animal", "pos"]).licks_per_resp.mean()
        D["log2r"] = [np.log2(v / base.get((a, p), np.nan)) if v > 0 else np.nan
                      for v, a, p in zip(D.licks_per_resp, D.animal, D.pos)]
        A1[src] = D
        tab = D.groupby(["animal", "pos", "epoch"]).log2r.mean().unstack("epoch").reindex(columns=list(EPS))
        print(f"\n=== A1 per-animal log2(licks per responded / own pre), source {src} ===")
        print(tab.round(2).to_string(), flush=True)
        tab.to_csv(TAB / f"a1_log2_licks_per_resp_{src}.csv")
    fc = A1["lp"][A1["lp"].pos == "far_contra"]
    P8 = _delta(fc, "log2r", rng, cells=("chronic",))["chronic"]
    P8daq = _delta(A1["daq"][A1["daq"].pos == "far_contra"], "log2r", rng, cells=("chronic",))["chronic"]
    FD["a1"] = dict(A1=A1, P8=P8, P8daq=P8daq)

    # ---------------- A3i / A3ii running-regressed level + RESTW
    _prep_imaging(B, I)
    a3 = {}
    print("\n=== A3i/ii quit period - time-matched control (global 'cortex') ===")
    for name, still in (("res_nl", None), ("res_nl_eng", None), ("lev_rw", None), ("res_rw", None),
                        ("lev_rw", True), ("lev_nl", None), ("lev_nl", True)):
        key = name + ("_still" if still else "")
        P = _pairs_arr(B, I, name, 0, still)
        P.to_csv(TAB / f"a3_{key}_pairs.csv", index=False)
        if len(P):
            g = P.groupby("animal").agg(n=("diff", "size"), diff=("diff", "mean"))
            print(f"-- {key}\n{g.round(3).to_string()}", flush=True)
        a3[key] = dict(P=P, test=_pair_test(P, rng))
    a3["areas"] = {a: _pair_test(_pairs_arr(B, I, "res_nl", ai), rng) for ai, a in enumerate(AREAS) if ai}
    FD["a3"] = a3

    # ---------------- Test 4b
    R4 = pd.DataFrame([r for r in (_test4b_session(B[k]) for k in sorted(B)) if r is not None])
    R4.to_csv(TAB / "t4b_sessions.csv", index=False)
    print("\n=== TEST 4b per-animal (quit sessions; pooled over epochs) ===")
    print(R4.groupby("animal")[["n_runbouts", "n_pairs", "p_after_run", "p_after_still", "n_q_run", "n_q_still",
                                "pq_run", "pq_still", "pe_run", "pe_still"]].mean().round(3).to_string(), flush=True)
    print(R4.groupby(["animal", "epoch"])[["q_runfrac", "q_bouts_per_min", "q_bout_s"]].mean().round(3).to_string())
    P7a = _paired(R4, "p_after_run", "p_after_still", rng)
    P7b = _paired(R4, "pq_run", "pq_still", rng)
    post = R4[R4.epoch != "pre"]
    sec4b = {"P7a post-only": _paired(post, "p_after_run", "p_after_still", rng),
             "P7b post-only": _paired(post, "pq_run", "pq_still", rng),
             "gain engaged": _paired(R4, "pe_run", "pe_still", rng),
             "latency engaged": _paired(R4, "late_run", "late_still", rng)}
    # engaged-period gain over ALL sessions (not only quit sessions)
    eg = []
    for lab in sorted(B):
        s = B[lab]
        sp, _ = _run_bins(s)
        if sp is None:
            continue
        T = s["trials"]
        k = len(T) if s["censored"] else s["quit_idx"]
        pr, ps = [], []
        for c, hh in zip(T.t.to_numpy(float)[:k], T.hit.to_numpy(int)[:k]):
            k0, k1 = int(np.floor(c - 10)), int(np.floor(c))
            if k0 < 0 or k1 > sp.size or not np.isfinite(sp[k0:k1]).all():
                continue
            w = sp[k0:k1]
            if (w > 3.0).sum() >= 1:
                pr.append(hh)
            elif (w <= 3.0).all():
                ps.append(hh)
        eg.append(dict(label=lab, animal=s["animal"], epoch=s["epoch"],
                       p_run=np.mean(pr) if len(pr) >= 3 else np.nan, p_still=np.mean(ps) if len(ps) >= 3 else np.nan))
    EG = pd.DataFrame(eg)
    sec4b["gain engaged, all sessions"] = _paired(EG, "p_run", "p_still", rng)
    bet = []
    for lab, im in I.items():
        if lab in B and np.isfinite(im.get("run_beta", np.array([np.nan]))).all():
            rb = im["run_beta"]
            bet.append(dict(label=lab, animal=B[lab]["animal"], epoch=B[lab]["epoch"],
                            orof=float(np.mean([rb[AREAS.index(a)] for a in OROF])),
                            ref=float(np.mean([rb[AREAS.index(a)] for a in REFC])),
                            **{f"b_{a}": float(rb[i]) for i, a in enumerate(AREAS)}))
    BE = pd.DataFrame(bet)
    BE["excess"] = BE.orof - BE.ref
    BE.to_csv(TAB / "t4b_running_betas.csv", index=False)
    _print_animal_table("4b P7c orofacial - reference running beta (z per mm/s)", BE, "excess")
    P7c = _paired(BE, "orof", "ref", rng)
    sec4b["P7c excess early post - pre"] = _delta(BE, "excess", rng, cells=("early",))["early"]
    FD["t4b"] = dict(R4=R4, EG=EG, BE=BE, P7a=P7a, P7b=P7b, P7c=P7c, sec=sec4b)

    # ---------------- families
    fams = {}
    for fname, items in (("A", {"P5": P5["early"], "P5t": P5t["early"], "P6": a3["res_nl"]["test"], "P8": P8}),
                         ("B", {"P7a": P7a, "P7b": P7b, "P7c": P7c})):
        ps = np.array([v["p"] for v in items.values()], float)
        ph = ak.holm(np.where(np.isfinite(ps), ps, 1.0))
        rows = []
        for (k, v), pholm in zip(items.items(), ph):
            iv = v["iv"]
            rows.append(dict(family=fname, test=k, point=iv.point if iv else np.nan, lo=iv.lo if iv else np.nan,
                             hi=iv.hi if iv else np.nan, p=v["p"], p_holm=pholm,
                             animals_pos=sum(x > 0 for x in v["per"].values()),
                             animals_neg=sum(x < 0 for x in v["per"].values()),
                             per_animal="; ".join(f"{a} {x:+.3g}" for a, x in v["per"].items())))
        fams[fname] = pd.DataFrame(rows)
        print(f"\n=== ADDENDUM FAMILY {fname} (Holm) ===\n{fams[fname].to_string(index=False)}", flush=True)
    FAM = pd.concat(fams.values())
    FAM.to_csv(TAB / "addendum_families.csv", index=False)
    FD["fam"] = FAM
    sec = []
    for nm, v in ([(f"P5 {c}", P5[c]) for c in P5] + [(f"P5t {c}", P5t[c]) for c in P5t] + [(f"P5 {r} {c}", P5s[r][c]) for r in P5s for c in P5s[r]]
                  + [(f"h80 water {c}", FD["a3iii"]["water"][c]) for c in FD["a3iii"]["water"]]
                  + [(f"h80 time {c}", FD["a3iii"]["time"][c]) for c in FD["a3iii"]["time"]]
                  + [("P8 DAQ far-contra chronic", P8daq)]
                  + [(f"A3 {k}", d["test"]) for k, d in a3.items() if k != "areas"]
                  + [(f"A3 area {a}", t) for a, t in a3["areas"].items()]
                  + list(sec4b.items())):
        iv = v["iv"]
        sec.append(dict(measure=nm, point=iv.point if iv else np.nan, lo=iv.lo if iv else np.nan,
                        hi=iv.hi if iv else np.nan, p=v["p"],
                        per_animal="; ".join(f"{a} {x:+.3g}" for a, x in v["per"].items())))
    SEC = pd.DataFrame(sec)
    SEC.to_csv(TAB / "addendum_all_contrasts.csv", index=False)
    print("\n=== ADDENDUM ALL CONTRASTS ===")
    print(SEC.to_string(index=False, float_format=lambda x: f"{x:.3g}"), flush=True)
    FD["sec"] = SEC
    from figdata import save_fd
    save_fd("disengagement_addendum", FD, root=FDIR)
    return FD


def plot_addendum(FD=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if FD is None:
        from figdata import load_fd
        FD = load_fd("disengagement_addendum", root=FDIR)
    EL, AM, POS = _style()
    fam = FD["fam"].set_index("test")

    # A1
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    for axi, (src, title) in zip(ax, (("lp", "cam4 LP licks"), ("daq", "DAQ spout contacts"))):
        D = FD["a1"]["A1"][src]
        for pi, p in enumerate(POS):
            for ei, ep in enumerate(EPS):
                g = D[(D.pos == p) & (D.epoch == ep)]
                x0 = pi + (ei - 1.5) * 0.2
                for j, an in enumerate(ANIMALS):
                    v = g[g.animal == an].log2r.to_numpy(float)
                    axi.scatter(x0 - 0.06 + 0.03 * j + np.zeros(v.size), v, marker=AM[an], facecolors="none",
                                edgecolors=EL[ep], s=12, lw=0.7, alpha=0.8)
                if len(g):
                    axi.scatter([x0 + 0.07], [np.nanmean(g.log2r)], marker="o", color=EL[ep], s=40, zorder=5)
        axi.axhline(0, color="0.5", lw=0.8)
        axi.set_xticks(range(len(POS)))
        axi.set_xticklabels([p.replace("_", "\n") for p in POS])
        axi.set_title(title, fontsize=10)
    ax[0].set_ylabel("log2(licks per responded trial / own August-pre mean, same spout)")
    p8 = fam.loc["P8"]
    fig.suptitle(f"BEHAVIOUR -- licks per responded trial relative to each animal's own pre, per spout "
                 f"(P8 chronic far contra, LP: {p8.point:+.2f} [{p8.lo:+.2f}, {p8.hi:+.2f}], Holm p = {p8.p_holm:.3g})",
                 fontsize=11)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    _save(fig, "a1_licks_per_resp_normalised")

    # A2 KM
    fig, ax = plt.subplots(3, 4, figsize=(17, 11.5))
    style = {"primary": "-", "b_i": ":", "b_ii": "--"}
    cm = {"T": "|", "W": "x", "U": "+", "H": "|"}
    for row, scale, xl in ((0, "t_min", "minutes from first cue"), (1, "water", "collected water, µl"),
                           (2, "n_tr", "trials from first cue")):
        for i, an in enumerate(ANIMALS):
            a_ = ax[row, i]
            for ep in EPS:
                for rd, ls in style.items():
                    k = FD["a2"]["km"].get((an, ep, scale, rd))
                    if k is None:
                        continue
                    a_.step(np.append(k["x"], k["xmax"]), np.append(k["y"], k["y"][-1]), where="post", color=EL[ep],
                            lw=1.6 if rd == "primary" else 0.9, ls=ls)
                    if rd == "primary":
                        for xc, cl in k["cens"]:
                            yv = float(k["y"][np.searchsorted(k["x"], xc, side="right") - 1])   # step value
                            a_.scatter([xc], [yv], marker=cm.get(cl, "|"), color=EL[ep], s=45, zorder=4)
            a_.set_ylim(-0.03, 1.12)
            a_.set_xlabel(xl)
            if row == 0:
                a_.set_title(an + ("  (pre = time-censored lower bounds)" if an in ("PS92", "PS93") else ""), fontsize=9)
        ax[row, 0].set_ylabel("fraction not yet quit")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=EL[e], label=e) for e in EPS] + \
        [Line2D([], [], color="k", ls=ls, label={"primary": "primary reading (Q, H = quit)",
                                                 "b_i": "bound: U also quit", "b_ii": "bound: H censored"}[rd])
         for rd, ls in style.items()] + \
        [Line2D([], [], color="k", marker=m, ls="", label=lab) for m, lab in (("|", "censored: time / rule (T)"),
                                                                              ("x", "water limit (W)"),
                                                                              ("+", "unknown (U)"))]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("BEHAVIOUR -- Kaplan-Meier of the quit in minutes (top), collected water (middle) and trials (bottom); "
                 "pre-session ends classified per A0", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(fig, "a2_km_time_water_trials")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9.2))
    _epoch_scatter(ax[0, 0], FD["a2"]["S"], "cap90_primary", "engaged minutes, capped at 90")
    ax[0, 0].set_title("time to quit (or end), capped at 90 min", fontsize=10)
    _delta_panel(ax[0, 1], FD["a2"]["P5"], "change vs own August pre, min", primary_cell="early",
                 p_holm=fam.loc["P5", "p_holm"])
    ax[0, 1].set_title(f"P5 minutes (Holm p = {fam.loc['P5', 'p_holm']:.3g}, family A)", fontsize=10)
    _delta_panel(ax[0, 2], FD["a3iii"]["time"], "change vs pre, min")
    ax[0, 2].set_title("exact 80-min horizon: minutes to quit-or-80", fontsize=10)
    _epoch_scatter(ax[1, 0], FD["a2"]["S"], "cap300_primary", "trials to quit (or end), capped at 300")
    ax[1, 0].set_title("trials to quit (or end), capped at 300", fontsize=10)
    _delta_panel(ax[1, 1], FD["a2"]["P5t"], "change vs own August pre, trials", primary_cell="early",
                 p_holm=fam.loc["P5t", "p_holm"])
    ax[1, 1].set_title(f"P5t trials (Holm p = {fam.loc['P5t', 'p_holm']:.3g}, family A)", fontsize=10)
    _delta_panel(ax[1, 2], FD["a2"]["sens"]["responded trials"], "change vs pre, responded trials")
    ax[1, 2].set_title("responded trials to quit-or-end (uncapped, secondary)", fontsize=10)
    fig.suptitle("BEHAVIOUR -- engagement before the quit, CAPPED (pre-registered P5 / P5t), pre vs post", fontsize=12)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, "a2_time_trials_capped")

    # UNCAPPED with bounds (Priya 10-31, after the capped results; outside the Holm families)
    S2 = FD["a2"]["S"].copy()
    S2["cens"] = (S2.ev_primary == 0)
    U = FD["a2"]["unc"]
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    for r, (col, lab) in enumerate((("t_min", "minutes"), ("n_tr", "trials (all)"), ("n_resp", "responded trials"))):
        _epoch_scatter(ax[r, 0], S2, col, f"{lab} to quit (or session end)", cens_col="cens")
        ax[r, 0].set_title(f"{lab}, uncapped (arrow = censored end; PS92 / PS93 pre = time-censored)", fontsize=9)
        _delta_panel(ax[r, 1], U[col]["lower"], f"change vs own pre, {lab}")
        ax[r, 1].set_title("LOWER BOUND: every censored end counted as the quit", fontsize=9)
        _delta_panel(ax[r, 2], U[col]["km"], f"KM restricted-mean change, {lab}")
        ax[r, 2].set_title("CENSORING-AWARE: KM restricted mean to a per-animal common horizon", fontsize=9)
    for r in range(3):
        lo_, hi_ = ax[r, 0].get_ylim()
        ax[r, 0].set_ylim(lo_, hi_ + 0.08 * (hi_ - lo_))            # room for the censoring arrows
    fig.suptitle("BEHAVIOUR -- time and trials to the quit, UNCAPPED, with censoring bounds (outside the Holm families)\n"
                 "Uncapped minutes are confounded: pre sessions ran to ~150 min, post stopped at 120 min / 100 per spout. "
                 "The KM restricted mean\ncompares pre and post only up to the shorter group's longest observation "
                 "(per animal), which removes that confound.", fontsize=10)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    _save(fig, "a2_time_trials_to_quit")

    S = FD["a2"]["S"]
    fig, ax = plt.subplots(1, 1, figsize=(13, 4.8))
    for pi, p in enumerate(POS):
        for ei, ep in enumerate(EPS):
            g = S[S.epoch == ep]
            x0 = pi + (ei - 1.5) * 0.2
            for j, an in enumerate(ANIMALS):
                v = g.loc[g.animal == an, f"prog_{p}"].to_numpy(float)
                ax.scatter(x0 - 0.06 + 0.03 * j + np.zeros(v.size), v, marker=AM[an], facecolors="none",
                           edgecolors=EL[ep], s=12, lw=0.7, alpha=0.8)
            ax.scatter([x0 + 0.07], [g[f"prog_{p}"].mean()], marker="o", color=EL[ep], s=40, zorder=5)
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax.set_xticks(range(len(POS)))
    ax.set_xticklabels([p.replace("_", "\n") for p in POS])
    ax.set_ylabel("trials done at quit (or end) / 100")
    ax.set_title("BEHAVIOUR -- per-spout trial progress at the quit (dotted = the post-stroke 100-per-spout stop rule)",
                 fontsize=11)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "a7_spout_progress")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    _epoch_scatter(ax[0], FD["a3iii"]["H"], "w80", "collected water at quit-or-80 min, µl")
    ax[0].set_title("water, exact 80-min horizon", fontsize=10)
    _delta_panel(ax[1], FD["a3iii"]["water"], "change vs own pre, µl")
    ax[1].set_title("sensitivity of P1 (not tested)", fontsize=10)
    fig.suptitle("BEHAVIOUR -- P1 with quits re-detected on records cut at 80 min", fontsize=12)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    _save(fig, "a3iii_h80_water")

    # A3 neural
    a3 = FD["a3"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
    _pair_panel(ax[0], [("running-\nregressed\n(P6)", a3["res_nl"]["test"], None),
                        ("regr. on\nengaged only", a3["res_nl_eng"]["test"], None),
                        ("raw level\nRESTnl (=P4)", a3["lev_nl"]["test"], None),
                        ("still only\nRESTnl", a3["lev_nl_still"]["test"], None),
                        ("raw level\nRESTW", a3["lev_rw"]["test"], None),
                        ("running-\nregressed\nRESTW", a3["res_rw"]["test"], None),
                        ("still only\nRESTW", a3["lev_rw_still"]["test"], None)],
                "quit period - time-matched control", primary_holm=fam.loc["P6", "p_holm"])
    ax[0].set_title(f"global pre-cue level (P6 Holm p = {fam.loc['P6', 'p_holm']:.3g}, family A)", fontsize=10)
    _pair_panel(ax[1], [(a.replace("_right", "\ncontra").replace("_left", "\nipsi"), t, None)
                        for a, t in a3["areas"].items()], "quit period - control, running-regressed (RESTnl)")
    ax[1].set_title("running-regressed level per area (contra = right hemisphere)", fontsize=10)
    fig.suptitle("NEURAL -- cortical pre-cue level in the quit period after regressing out running", fontsize=12)
    _legend_animals(fig, epochs=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    _save(fig, "a3_running_regressed_level")

    # 4b
    t = FD["t4b"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    _pair_panel(ax[0, 0], [("P7a next trial\nafter run - still", t["P7a"], None),
                           ("post quits\nonly", t["sec"]["P7a post-only"], None)],
                "difference in P(next trial responded)", primary_holm=fam.loc["P7a", "p_holm"])
    ax[0, 0].set_title(f"(1) re-engagement in quit periods (P7a Holm p = {fam.loc['P7a', 'p_holm']:.3g})", fontsize=10)
    _pair_panel(ax[0, 1], [("P7b quit\nperiods", t["P7b"], None), ("post quits\nonly", t["sec"]["P7b post-only"], None),
                           ("engaged,\nquit sessions", t["sec"]["gain engaged"], None),
                           ("engaged,\nall sessions", t["sec"]["gain engaged, all sessions"], None)],
                "P(responded | run in prior 10 s) - P(.. | still)", primary_holm=fam.loc["P7b", "p_holm"])
    ax[0, 1].set_title(f"(2) gain after running (P7b Holm p = {fam.loc['P7b', 'p_holm']:.3g})", fontsize=10)
    BE = t["BE"]
    ar = list(OROF) + list(REFC)
    for i, a in enumerate(ar):
        for j, an in enumerate(ANIMALS):
            v = BE.loc[BE.animal == an, f"b_{a}"].mean()
            ax[1, 0].scatter(i - 0.2 + 0.13 * j, v, marker=AM[an], facecolors="none",
                             edgecolors="#b2182b" if a in OROF else "#2166ac", s=28)
    ax[1, 0].axhline(0, color="0.6", lw=0.8)
    ax[1, 0].axvline(len(OROF) - 0.5, color="0.7", lw=0.8, ls=":")
    ax[1, 0].set_xticks(range(len(ar)))
    ax[1, 0].set_xticklabels([a.replace("_right", "\ncontra").replace("_left", "\nipsi") for a in ar], fontsize=7)
    ax[1, 0].set_ylabel("running beta (z level per mm/s), animal means")
    pc = fam.loc["P7c"]
    ax[1, 0].set_title(f"(3) orofacial (red) vs reference (blue): P7c {pc.point:+.3f} [{pc.lo:+.3f}, {pc.hi:+.3f}], "
                       f"Holm p = {pc.p_holm:.3g}", fontsize=9)
    R4 = t["R4"]
    _epoch_scatter(ax[1, 1], R4, "q_runfrac", "fraction of quit period running")
    ax[1, 1].set_title("(4) running in quit periods by epoch (pre = PS93 1, PS94 4, PS95 6 quits)", fontsize=9)
    fig.suptitle("BEHAVIOUR + NEURAL -- Test 4b: is quit-period running self-activation to get back to licking?",
                 fontsize=12)
    _legend_animals(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(fig, "t4b_self_activation")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", nargs="?", default="all",
                    choices=("extract", "analyse", "plot", "all", "addendum", "plot-addendum"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "plot-addendum":
        plot_addendum()
        return 0
    if a.mode == "addendum":
        plot_addendum(addendum())
        return 0
    if a.plot_only or a.mode == "plot":
        plot()
        return 0
    if a.mode in ("extract", "all"):
        extract(min(a.jobs, 4))
    if a.mode in ("analyse", "all"):
        plot(analyse())
    return 0


if __name__ == "__main__":
    sys.exit(main())
