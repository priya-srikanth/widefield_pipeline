"""Quiet-event detector inside TASK windows (T12) and pre-cue events vs the trial's outcome / position code (T13)
(cohort1 PS92-95, 2026-11-01; docs/status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md section 4).

T12 -- do the brief orofacial events that rise ~5x in QUIET periods also occur inside the task? Same traces and detector as
quiet_spikes_v2 (quiet_area_spikes.session_traces, dF/F % minus a 2-s running median; quiet_spikes_v2.detect: height above
the median of the preceding 1.0-0.2 s, half-width 1.5 frames .. 0.5 s, merged within 0.3 s, both polarities; threshold
k = 4 x the animal's AUGUST-pre robust SD from the v2 quiet sessions, quiet_fluct.animal_sd). Detection masks:
  PRE-CUE  frames in [spout strobe, cue) of each trial (wfield_local.trial_windows.trial_bounds; the strobe is sent once
           the spout has arrived), with no DAQ contact lick / running bout within 0.5 s and no swapped pair within 0.5 s
  ITI      frames in [trial stop, next trial's strobe), same exclusions
  QUIET    the v2 rule (no lick / running within 1.5 s), recomputed here from the same pass as the reference
Rates = events / min of mask frames, ENGAGED part of the session only (trial cue before the session's quit, as v2).
PRE-CUE events are also split by whether the frame is itself a v2 QUIET frame, so "task-window events" are not just the
quiet events that happen to fall in a pre-cue window.

T13 -- does a pre-cue orofacial event on a trial predict (a) a failed far-contra attempt (video attempt classes,
LP run imaging_encoding/qc/trial_attempts.csv: hit / incomplete / no_try / jaw_only; also all-spout hit vs miss) and
(b) a weaker pre-cue position code (attempt_coding trials_precue_<animal>.csv.gz: per-trial p_true of the stored frozen
pre-stroke decoder on the pre-cue ENL window)? Trial event flag = any event (orofacial ipsi OR contra, either
polarity, > 4 x pre SD) in the trial's clean pre-cue window ("ev_pre"); for (b) also restricted to the decoder's window
("ev_enl"): since 11-04 the shared LICK-FREE ENL of the run's lick source (LP by default: 1.0 s, slid to the latest
lick-free gap like the decoder's, trials with none out; was the fixed 2 s before the cue). Trials: engagement-gated
(trial_attempts `canonical`), >= 1.0 s of clean pre-cue frames (>= MIN_CLEAN_ENL_S = half the window clean for
ev_enl). Outputs keyed by source and window (task_events_<src>lf<W>s/; the DAQ 2 s run keeps task_events/). WITHIN-SESSION, SPOUT-MATCHED: strata = session x spout; effect =
EXCESS = mean over event trials of (outcome - its stratum mean), i.e. event trials vs the other trials of the same
spout in the same session; p = permutation of the event flags within strata (N_PERM). STRICT strata add the
within-session trial-order quartile and a clean-time bin (events and misses both rise late in the session and with the
lick-free time available), reported as excess_strict / p_strict and used for the figure's stars. Pooled per epoch: conditional
logistic regression (statsmodels ConditionalLogit, groups = session x spout) outcome ~ event + log(clean pre-cue s).
Per-animal x epoch tables first (CLAUDE.md rule 8).

OUTPUT M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes/v2_quiet/task_events/

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_task_events.py [--jobs 3] [--summarise] [--only PS92_0811 ...]
"""
import argparse
import os
import sys
from pathlib import Path

# THE RUN'S LICK SOURCE DEFAULTS TO LP (Priya 11-04), as in quiet_task_events_followup: the T13 p_true is attempt_coding's
# (an LP-only family), so the ENL window and its lick-free gate follow LP unless a run says otherwise.
os.environ.setdefault("WIDEFIELD_LICK_SOURCE", "lp")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import EP, EVENTS, columns, session_traces  # noqa: E402
from quiet_spikes_v2 import LICK_SOURCE, V2, busy_frames, detect, detrend  # noqa: E402
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # noqa: E402 - rebuild root (11-03)
from wfield_local.attempt_coding import trials_name as attempt_trials_name  # noqa: E402
from wfield_local import lp_licks as _lp_licks  # noqa: E402  (coordinator 11-04: the run's attempts table)
from wfield_local import config  # noqa: E402

#: THE ENL window, s (Priya 11-04): the shared `lp_licks.enl_post_s()` of the run's lick source -- 1.0 s under LP, the
#: frozen attempt_coding decoder's window -- LICK-FREE on that source (`lp_licks.gate_licks_s`) and SLID like the
#: decoder's (`lp_licks.enl_window_starts_s` = `precue_window_start`; per trial `enl_w0` in the npz, NaN = no lick-free
#: window -> out of the ENL rows). Until 11-04: a fixed, DAQ 2.0 s [cue - 2, cue).
ENL_S = _lp_licks.enl_post_s()
ENL_SRC = _lp_licks.lick_source()
#: OUTPUTS ARE KEYED BY (source, window): the pre-11-04 DAQ 2.0 s outputs stay in task_events/ and are never mixed in.
TE_SUFFIX = "" if (ENL_SRC, ENL_S) == ("daq", 2.0) else f"_{ENL_SRC}lf{ENL_S:g}s"
TE = V2 / f"task_events{TE_SUFFIX}"
K = 4.0
TASK_BUF_S = 0.5
COLS = ["orofacial_left", "orofacial_right", "orofacial", "limb_left", "limb_right"]
TRIAL_COLS = ("orofacial_left", "orofacial_right")       # T13 event flag: either hemisphere
WIN = {0: "precue", 1: "iti", 2: "quiet"}
MIN_CLEAN_S = 1.0
CLEAN_BINS = (0, 1.5, 2.0, 2.5, 99)
#: "CLEAN" = frames of the detection mask: no DAQ contact lick (+ LP licks under QS_LICK_SOURCE=daq+lp) or running bout
#: within TASK_BUF_S, no swapped pair within 0.5 s, finite -- NOT the lick-free test (that is the window's own gate).
#: So the requirement inside the ENL window stays, SCALED with it: MIN_CLEAN_ENL_FRAC x ENL_S = 0.5 s of a 1.0 s window,
#: the same fraction as the 1.0 s of 2.0 s used until 11-04 (PROPOSED 11-04; MIN_CLEAN_S stays for the whole pre-cue).
MIN_CLEAN_ENL_FRAC = 0.5
MIN_CLEAN_ENL_S = MIN_CLEAN_ENL_FRAC * ENL_S
#: CLEAN_BINS for the ENL-window strata, scaled with the window (identical to CLEAN_BINS for the 2.0 s window).
CLEAN_ENL_BINS = tuple(x * ENL_S / 2.0 for x in CLEAN_BINS)
N_PERM = 2000
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
LP = _encoding_root()
LESION_NAME = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
               "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
HEMI = {"orofacial_left": "orofacial ipsilesional", "orofacial_right": "orofacial contralesional",
        "orofacial": "orofacial bilateral", "limb_left": "limb ipsilesional", "limb_right": "limb contralesional"}


def near_frames(label, ft, buf):
    """Frames within `buf` s of a DAQ contact lick onset or a running bout."""
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fsd = float(z["fs"])
    m = np.zeros(len(ft), bool)
    licks = z["lick_onsets"] / fsd
    if LICK_SOURCE == "daq+lp":                       # Priya 11-01: LP video licks too (same switch as busy_frames)
        from lp_licks import lick_times
        licks = np.r_[licks, lick_times(label)]
    for x in licks:
        m[np.searchsorted(ft, x - buf):np.searchsorted(ft, x + buf)] = True
    for r0, r1 in np.c_[z["running_starts"], z["running_stops"]] / fsd:
        m[np.searchsorted(ft, r0 - buf):np.searchsorted(ft, r1 + buf)] = True
    return m


def _npz_enl_s(f) -> float:
    """The ENL window a per-session npz was built with (files written before 11-04 carry none: 2.0 s)."""
    with np.load(f) as z:
        return float(z["enl_s"]) if "enl_s" in z.files else 2.0


def enl_starts(label, cue, strobe):
    """Per trial the start (DAQ s) of its lick-free ENL window, NaN = none: licks of the run's source
    (`lp_licks.gate_licks_s` on the session's DAQ contact onsets; under LP the edited LP train, cues without an LP
    verdict -> NaN; a session with no LP tables -> all NaN), searched like the decoders' window."""
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    daq = z["lick_onsets"] / float(z["fs"])
    out = np.full(len(cue), np.nan)
    ok = np.isfinite(cue)
    try:
        licks, has = _lp_licks.gate_licks_s(label, cue[ok], daq, float(_cfg_max_rt()))
    except _lp_licks.LPUnavailable as ex:
        print(f"  {label}: no LP licks ({ex}) -> no ENL rows", flush=True)
        return out
    w = _lp_licks.enl_window_starts_s(cue[ok], strobe[ok], licks, ENL_S)
    out[np.flatnonzero(ok)] = np.where(has, w, np.nan)
    return out


def _cfg_max_rt():
    from wfield_local import config
    return config.defaults()["decode"]["max_rt_s"]


def extract(label):
    """One session (spawn worker): events in pre-cue / ITI / quiet masks + per-trial clean-frame bookkeeping."""
    f = TE / "sessions" / f"{label}.npz"
    if f.exists() and _npz_enl_s(f) == ENL_S:
        return label
    from wfield_local import trial_windows as TW
    Xr, ft, fs, _rest, engaged, quit_s = session_traces(label, detrend=False)
    X = detrend(Xr, fs)
    T = X.shape[1]
    busy15, badw = busy_frames(label, ft, fs)
    near = near_frames(label, ft, TASK_BUF_S)
    finite = np.isfinite(X).all(0)
    quiet = ~busy15 & ~badw & finite
    animal, mmdd = label.split("_")
    tb = TW.trial_bounds(animal, f"2026{mmdd}")
    cue = tb.cue_s.to_numpy(float)
    strobe = tb.strobe_s.to_numpy(float)
    stop = tb.stop_s.to_numpy(float)
    n = len(tb)
    prev_stop = np.r_[-np.inf, stop[:-1]]
    # pre-cue start = the strobe (spout arrived); must follow the previous trial's stop and precede the cue
    start = np.where(np.isfinite(strobe) & (strobe < cue) & (strobe >= prev_stop), strobe, np.nan)
    tri_pre = np.full(T, -1, int)
    tri_iti = np.full(T, -1, int)
    for i in range(n):
        if np.isfinite(start[i]):
            tri_pre[np.searchsorted(ft, start[i]):np.searchsorted(ft, cue[i])] = i
        if i + 1 < n and np.isfinite(start[i + 1]) and np.isfinite(stop[i]) and start[i + 1] > stop[i]:
            tri_iti[np.searchsorted(ft, stop[i]):np.searchsorted(ft, start[i + 1])] = i
    clean = ~near & ~badw & finite
    ok_pre = (tri_pre >= 0) & clean
    ok_iti = (tri_iti >= 0) & clean
    cols = columns()
    ev = []
    for c in COLS:
        ci = cols.index(c)
        if not np.isfinite(X[ci]).all():
            continue
        for code, ok in ((0, ok_pre), (1, ok_iti), (2, quiet)):
            for sign in (1, -1):
                pk, h = detect(sign * X[ci], ok, fs)
                for p, v in zip(pk, h):
                    tr = tri_pre[p] if code == 0 else tri_iti[p] if code == 1 else -1
                    trel = ft[p] - cue[tr] if code == 0 else np.nan
                    ev.append((ci, code, int(p), sign * float(v), int(tr), int(engaged[p]), int(quiet[p]), trel))
    # per trial: clean pre-cue frames (all / inside the last ENL_S / also v2-quiet), clean ITI frames
    npre = np.bincount(tri_pre[ok_pre], minlength=n)[:n]
    w0 = enl_starts(label, cue, strobe)
    in_enl = np.zeros(T, bool)
    for i in np.flatnonzero(np.isfinite(w0)):
        in_enl[np.searchsorted(ft, w0[i]):np.searchsorted(ft, w0[i] + ENL_S)] = True
    npre_enl = np.bincount(tri_pre[ok_pre & in_enl], minlength=n)[:n]
    npre_q = np.bincount(tri_pre[ok_pre & quiet], minlength=n)[:n]
    npre_all = np.bincount(tri_pre[tri_pre >= 0], minlength=n)[:n]
    niti = np.bincount(tri_iti[ok_iti], minlength=n)[:n]
    eng_tr = ~(cue >= quit_s) if np.isfinite(quit_s) else np.ones(n, bool)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f, ev=np.array(ev, float).reshape(-1, 8), fs=fs, cols=np.array(cols), quit_s=quit_s,
        n_quiet_eng=int((quiet & engaged).sum()), trial_id=tb.trial_id.to_numpy(int),
        pos_name=tb.pos_name.astype(str).to_numpy(), cue=cue, start=start, stop=stop, npre=npre, npre_enl=npre_enl,
        npre_q=npre_q, npre_all=npre_all, niti=niti, eng_tr=eng_tr, total_min=T / fs / 60, enl_s=float(ENL_S),
        enl_src=ENL_SRC, enl_w0=w0)
    return label


# ----------------------------------------------------------------------------------------------------------------- summary
def load_sessions():
    from quiet_fluct import animal_sd
    from wfield_local import epochs
    sd = animal_sd()
    D = {}
    for f in sorted((TE / "sessions").glob("*.npz")):
        lab = f.stem
        if lab[:4] not in ANIMALS:
            continue
        if _npz_enl_s(f) != ENL_S:
            raise SystemExit(f"{f}: built with ENL window {_npz_enl_s(f)} s, this run uses {ENL_S} s -- re-extract")
        ep = epochs.epoch_of(lab)
        if ep is None or (ep == "pre" and config.early_pre(lab)):
            continue
        D[lab] = dict(np.load(f, allow_pickle=True), epoch=ep, day=epochs.days_since_stroke(lab))
        if "enl_w0" not in D[lab]:                    # a pre-11-04 DAQ 2 s file: the fixed window ending at the cue
            D[lab]["enl_w0"] = np.asarray(D[lab]["cue"], float) - ENL_S
    return D, sd


def session_rates(D, sd):
    """One row per session: event rate (either polarity, > K x pre SD) per column x window, engaged part."""
    import pandas as pd
    cols = columns()
    rows = []
    for lab, d in D.items():
        an, fs, ev = lab[:4], float(d["fs"]), d["ev"]
        eng = d["eng_tr"].astype(bool)
        pre_min = d["npre"][eng].sum() / fs / 60
        preq_min = d["npre_q"][eng].sum() / fs / 60
        pren_min = pre_min - preq_min
        iti_min = d["niti"][eng].sum() / fs / 60
        q_min = d["n_quiet_eng"] / fs / 60
        r = dict(label=lab, animal=an, epoch=d["epoch"], day=d["day"], precue_min=pre_min, iti_min=iti_min,
                 quiet_min=q_min, precue_quiet_frac=preq_min / max(pre_min, 1e-9),
                 precue_clean_frac=d["npre"][eng].sum() / max(1, d["npre_all"][eng].sum()))
        for c in COLS:
            ci = cols.index(c)
            big = (ev[:, 0] == ci) & (np.abs(ev[:, 3]) > K * sd[an][ci])
            tr = ev[:, 4].astype(int)
            tr_eng = np.where(tr >= 0, eng[np.clip(tr, 0, len(eng) - 1)], False)
            n_pre = big & (ev[:, 1] == 0) & tr_eng
            n_iti = big & (ev[:, 1] == 1) & tr_eng
            n_q = big & (ev[:, 1] == 2) & (ev[:, 5] == 1)
            r[f"{c}|precue"] = n_pre.sum() / pre_min if pre_min > 0.25 else np.nan
            r[f"{c}|precue_inquiet"] = (n_pre & (ev[:, 6] == 1)).sum() / preq_min if preq_min > 0.25 else np.nan
            r[f"{c}|precue_notquiet"] = (n_pre & (ev[:, 6] == 0)).sum() / pren_min if pren_min > 0.25 else np.nan
            r[f"{c}|iti"] = n_iti.sum() / iti_min if iti_min > 0.25 else np.nan
            r[f"{c}|quiet"] = n_q.sum() / q_min if q_min > 0.25 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


WINDOWS = ("precue", "precue_notquiet", "precue_inquiet", "iti", "quiet")


def animal_table(S):
    """Per animal x epoch: mean session rate per window (orofacial columns) + post / pre fold per animal."""
    import pandas as pd
    rows = []
    for c in COLS:
        for w in WINDOWS:
            col = f"{c}|{w}"
            M = S.groupby(["animal", "epoch"])[col].mean().unstack("epoch").reindex(columns=EP)
            N = S.dropna(subset=[col]).groupby(["animal", "epoch"]).size().unstack("epoch").reindex(columns=EP)
            for an in M.index:
                pre = M.loc[an, "pre"]
                for ep in EP:
                    rows.append(dict(column=c, window=w, animal=an, epoch=ep, rate=M.loc[an, ep],
                                     n_sessions=N.loc[an, ep] if an in N.index else 0,
                                     fold_vs_pre=M.loc[an, ep] / pre if pre > 0 else np.nan))
    return pd.DataFrame(rows)


def t12_stats(S):
    """LMM (mixed_models.lmm_table, epoch + time models, August pre) on log10(rate + 0.2) per window x column, and on the
    session's log ratio precue / quiet (does the task-window change differ from the quiet change?)."""
    import pandas as pd
    from wfield_local import mixed_models as mm
    L = []
    for c in COLS:
        for w in WINDOWS:
            L.append(S[["label"]].assign(column=c, measure=w, y=np.log10(S[f"{c}|{w}"] + 0.2).to_numpy()))
        L.append(S[["label"]].assign(column=c, measure="log_precue_over_quiet",
                                     y=(np.log10(S[f"{c}|precue"] + 0.2) - np.log10(S[f"{c}|quiet"] + 0.2)).to_numpy()))
    L = mm.add_time_covariates(pd.concat(L), august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "column"], models=("epoch", "time"), print_animals=False, fdr_by=("measure",))
    return T


def t12_figure(S, A, T, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    cs = ["orofacial_left", "orofacial_right"]
    ws = [("precue", "PRE-CUE (spout in place, before the cue)"), ("iti", "ITI (trial end to next spout arrival)"),
          ("quiet", "QUIET (v2 rule, reference)")]
    fig, ax = plt.subplots(3, 3, figsize=(13, 11.5))
    rng = np.random.default_rng(0)
    for r_, c in enumerate(cs):
        for j, (w, wname) in enumerate(ws):
            a = ax[r_, j]
            col = f"{c}|{w}"
            for x, ep in enumerate(EP):
                B = S[(S.epoch == ep) & S[col].notna()]
                for an, G in B.groupby("animal"):
                    a.scatter(x + rng.uniform(-0.18, 0.18, len(G)), G[col], marker=ANIMAL_MARKER[an],
                              facecolors="none", edgecolors=EPOCH_LINE[ep], s=18, lw=0.8, alpha=0.8)
                ci = boot_ci({an: list(G[col]) for an, G in B.groupby("animal")}, rng)
                if ci is not None:
                    a.errorbar(x + 0.32, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="o",
                               color=EPOCH_LINE[ep], ms=6, capsize=3, lw=1.6, zorder=5)
                t = T[(T.model == "epoch") & (T.measure == w) & (T.column == c) & (T.term == f"C(epoch)[T.{ep}]")]
                if x > 0 and len(t) and t.q.iloc[0] < 0.05:
                    q = t.q.iloc[0]
                    sig_marks.star_top(a, x, "***" if q < 0.001 else "**" if q < 0.01 else "*", color=EPOCH_LINE[ep])
            a.set_xticks(range(len(EP)), EP, fontsize=8)
            a.set_title(f"{wname}\n{HEMI[c]}", fontsize=9)
            if j == 0:
                a.set_ylabel("events / min (either polarity)")
    # bottom row: per-animal fold vs pre, pre-cue vs quiet (and ITI vs quiet), and the pre-cue split by quiet status
    for j, (w, name) in enumerate((("precue", "pre-cue"), ("iti", "ITI"))):
        a = ax[2, j]
        for c, fill in (("orofacial_left", "none"), ("orofacial_right", None)):
            Q = A[(A.column == c) & (A.window == "quiet")].set_index(["animal", "epoch"]).fold_vs_pre
            P = A[(A.column == c) & (A.window == w)].set_index(["animal", "epoch"]).fold_vs_pre
            for (an, ep), q in Q.items():
                if ep == "pre" or not np.isfinite(q) or not np.isfinite(P.get((an, ep), np.nan)):
                    continue
                a.scatter(q, P[(an, ep)], marker=ANIMAL_MARKER[an], s=40, lw=1.0, edgecolors=EPOCH_LINE[ep],
                          facecolors="none" if fill == "none" else EPOCH_LINE[ep])
        lim = [0.5, max(2, float(np.nanmax(A[A.window.isin(["quiet", w])].fold_vs_pre)) * 1.1)]
        a.plot(lim, lim, color="0.6", lw=0.8, ls="--")
        a.axhline(1, color="0.85", lw=0.6); a.axvline(1, color="0.85", lw=0.6)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("QUIET rate, fold vs the animal's pre"); a.set_ylabel(f"{name.upper()} rate, fold vs pre")
        a.set_title(f"Per animal x epoch: {name} vs quiet change\n(open = ipsilesional, filled = contralesional;\n"
                    "dashed = equal change)", fontsize=9)
    # bottom-right (Priya 11-03 rules): no lines; per window x epoch the SESSION rates as faded open markers (shape =
    # animal, colour = epoch) behind the per-animal means (small filled); window = lateral position inside the epoch
    a = ax[2, 2]
    from wfield_local import cat_points as cp
    br = {}
    wins = (("precue_notquiet", "not\nquiet"), ("precue_inquiet", "quiet"))
    for k, (w, _short) in enumerate(wins):
        for x, ep in enumerate(EP):
            xk = x + (k - 0.5) * 0.4
            G = S[S.epoch == ep]
            cp.session_dots(a, xk, G[f"orofacial|{w}"], G.animal, EPOCH_LINE[ep], jitter=0.06, seed=10 * x + k)
            for an in sorted(S.animal.unique()):
                v = A[(A.column == "orofacial") & (A.window == w) & (A.animal == an) & (A.epoch == ep)].rate
                if len(v) and np.isfinite(v.iloc[0]):
                    br[(w, ep, an)] = float(v.iloc[0])
                    a.plot(xk, v.iloc[0], cp.animal_marker(an), ls="none", color=EPOCH_LINE[ep], mec="w", mew=0.5,
                           ms=6, zorder=4)
    a.set_xticks([x + (k - 0.5) * 0.4 for x in range(len(EP)) for k in range(2)],
                 [f"{ep}\n{s_}" for ep in EP for _w, s_ in wins], fontsize=7)
    for x in range(len(EP) - 1):
        a.axvline(x + 0.5, color="0.92", lw=0.6, zorder=0)
    a.set_ylabel("events / min")
    a.set_title("Pre-cue events (orofacial bilateral), split by\nwhether the frame is itself a v2 QUIET frame\n"
                "(faded open = sessions, small filled = animal means)", fontsize=9)
    fig._wf_bottom_right = br
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    fig.legend(handles=handles, fontsize=8, loc="upper right", bbox_to_anchor=(0.995, 0.935), ncol=4, frameon=False)
    sig_marks.reserve_headroom(fig)
    fig.suptitle("T12: orofacial events in TASK windows. v2 detector (2-s median detrended dF/F %, height > 4 x the "
                 "animal's August-pre robust SD, either polarity); task masks exclude frames within 0.5 s of a contact "
                 "lick / running / swapped pair; engaged part of the session. Open markers = sessions (shape = animal, "
                 "color = epoch); filled = pooled mean, nested animal -> session bootstrap 95% CI; stars = LMM epoch "
                 "vs pre on log10(rate + 0.2), FDR.", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0, h_pad=1.5)
    fig.savefig(TE / "t12_task_window_event_rates.png" if out is None else out, dpi=110)
    plt.close(fig)
    return fig._wf_bottom_right


# -------------------------------------------------------------------------------------------------------------------- T13
def trial_table(D, sd):
    """One row per trial with event flags, clean seconds, attempt class (trial_attempts) and p_true (attempt_coding)."""
    import pandas as pd
    cols = columns()
    rows = []
    for lab, d in D.items():
        an, fs, ev = lab[:4], float(d["fs"]), d["ev"]
        big = np.zeros(len(ev), bool)
        for c in TRIAL_COLS:
            ci = cols.index(c)
            big |= (ev[:, 0] == ci) & (np.abs(ev[:, 3]) > K * sd[an][ci])
        E = ev[big & (ev[:, 1] == 0)]
        n = len(d["trial_id"])
        tr = E[:, 4].astype(int)
        ev_pre = np.bincount(tr, minlength=n)[:n] > 0
        # an event inside the trial's OWN (slid, lick-free) ENL window; trials without one are not ENL rows
        w0r = np.asarray(d["enl_w0"], float) - np.asarray(d["cue"], float)          # window start rel. to the cue
        enl = (E[:, 7] >= w0r[tr]) & (E[:, 7] < w0r[tr] + ENL_S)
        ev_enl = np.bincount(tr[enl], minlength=n)[:n] > 0
        enl_ok = np.isfinite(w0r)
        rows.append(pd.DataFrame(dict(label=lab, animal=an, epoch=d["epoch"], day=d["day"], trial_id=d["trial_id"],
                                      spout_daq=[LESION_NAME.get(p, p) for p in d["pos_name"]],
                                      clean_s=d["npre"] / fs, clean_enl_s=d["npre_enl"] / fs, eng_tr=d["eng_tr"],
                                      ev_pre=ev_pre, ev_enl=ev_enl, enl_ok=enl_ok, n_ev=np.bincount(tr, minlength=n)[:n])))
    Tr = pd.concat(rows, ignore_index=True)
    A = pd.read_csv(_lp_licks.attempts_path(), usecols=["label", "trial_id", "pos", "cls", "canonical"])
    Tr = Tr.merge(A, on=["label", "trial_id"], how="left")
    P = []
    for an in ANIMALS:
        from wfield_local import attempt_coding as _ac
        f = _ac.out_dir() / attempt_trials_name("precue", an)           # the writer's folder (review N1)
        if f.exists():
            x = pd.read_csv(f, usecols=["label", "trial_id", "spout", "cls", "p_true", "method"])
            P.append(x[x.method == "dom"].drop(columns="method").rename(columns={"spout": "spout_ac", "cls": "cls_ac"}))
    Tr = Tr.merge(pd.concat(P).drop_duplicates(["label", "trial_id"]), on=["label", "trial_id"], how="left")
    return Tr


def excess(y, e, strata, n_perm=N_PERM, seed=0):
    """Within-stratum effect of a binary flag: mean over flagged trials of (y - stratum mean); strata without both
    flagged and unflagged trials are dropped. p = two-sided permutation of flags within strata. -> dict."""
    y, e, strata = np.asarray(y, float), np.asarray(e, bool), np.asarray(strata)
    ok = np.isfinite(y)
    y, e, strata = y[ok], e[ok], strata[ok]
    _, s = np.unique(strata, return_inverse=True)
    ne, nt = np.bincount(s, weights=e), np.bincount(s)
    keep = (ne > 0) & (ne < nt)
    m = keep[s]
    y, e, s = y[m], e[m], s[m]
    out = dict(n_trials=int(m.sum()), n_event=int(e.sum()), n_strata=int(keep.sum()))
    if e.sum() < 3:
        return {**out, "excess": np.nan, "p": np.nan, "rate_event": np.nan, "rate_noevent": np.nan}
    mu = np.bincount(s, weights=y) / np.maximum(np.bincount(s), 1)
    yc = y - mu[s]
    stat = yc[e].mean()
    o = np.argsort(s, kind="stable")
    yc_o, e_o, s_o = yc[o], e[o], s[o]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.lexsort((rng.random(len(s_o)), s_o))
        null[k] = yc_o[e_o[perm]].mean()
    p = (1 + np.sum(np.abs(null) >= abs(stat))) / (n_perm + 1)
    return {**out, "excess": float(stat), "p": float(p), "rate_event": float(y[e].mean()),
            "rate_noevent": float(y[~e].mean())}


def clogit(y, e, logc, strata):
    """Conditional logistic regression y ~ event + log(clean s), groups = strata. -> (coef, se, p) of event."""
    import warnings
    from statsmodels.discrete.conditional_models import ConditionalLogit
    y = np.asarray(y, float)
    X = np.c_[np.asarray(e, float), np.asarray(logc, float)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ConditionalLogit(y, X, groups=np.asarray(strata)).fit(disp=False)
        return float(r.params[0]), float(r.bse[0]), float(r.pvalues[0])
    except Exception:                                   # noqa: BLE001
        return np.nan, np.nan, np.nan


def outcomes(Tr):
    """{name: (subset frame, outcome column, flag column, clean column)} for T13."""
    import pandas as pd
    B = Tr[(Tr.canonical == True) & Tr.eng_tr & (Tr.pos == Tr.spout_daq)]          # noqa: E712
    B = B.assign(stratum=B.label + "|" + B.pos)
    # STRICT strata (+ within-session trial-order quartile + clean-time bin): events and misses both rise late in a
    # session and with how much lick-free pre-cue time there is, so the session x spout match alone can be confounded
    q = B.groupby("label").trial_id.transform(lambda x: np.floor(4 * (x.rank(method="first") - 1) / len(x)))
    B = B.assign(tq=q.astype(int).astype(str))
    a = B[B.clean_s >= MIN_CLEAN_S]
    a = a.assign(strict=a.stratum + "|" + a.tq + "|" + pd.cut(a.clean_s, CLEAN_BINS, labels=False).astype(str))
    fc = a[(a.pos == "far_contra") & a.cls.isin(["hit", "incomplete", "no_try", "jaw_only"])]
    O = {"farcontra_fail": (fc.assign(y=(fc.cls != "hit").astype(float)), "ev_pre", "clean_s"),
         "farcontra_incomplete_vs_hit": (fc[fc.cls.isin(["hit", "incomplete"])]
                                         .assign(y=lambda x: (x.cls == "incomplete").astype(float)), "ev_pre", "clean_s"),
         "farcontra_noattempt_vs_hit": (fc[fc.cls.isin(["hit", "no_try"])]
                                        .assign(y=lambda x: (x.cls == "no_try").astype(float)), "ev_pre", "clean_s")}
    al = a[a.cls.isin(["hit", "incomplete", "no_try", "jaw_only"])]
    O["allspout_miss"] = (al.assign(y=(al.cls != "hit").astype(float)), "ev_pre", "clean_s")
    b = B[B.enl_ok & (B.clean_enl_s >= MIN_CLEAN_ENL_S) & B.p_true.notna() & (B.spout_ac == B.pos)]
    b = b.assign(strict=b.stratum + "|" + b.tq + "|" + pd.cut(b.clean_enl_s, CLEAN_ENL_BINS, labels=False).astype(str))
    O["p_true_enl_event"] = (b.assign(y=b.p_true), "ev_enl", "clean_enl_s")
    O["p_true_enl_event_hits"] = (b[b.cls_ac.isin(["hit", "pre_hit"])].assign(y=lambda x: x.p_true), "ev_enl",
                                  "clean_enl_s")
    O["p_true_any_precue_event"] = (b.assign(y=b.p_true), "ev_pre", "clean_enl_s")
    return O


def t13(Tr):
    import pandas as pd
    O = outcomes(Tr)
    rows = []
    for name, (B, flag, cl) in O.items():
        binary = not name.startswith("p_true")
        for (an, ep), G in list(B.groupby(["animal", "epoch"])) + [(("ALL", ep), G) for ep, G in B.groupby("epoch")]:
            r = excess(G.y, G[flag], G.stratum)
            rs = excess(G.y, G[flag], G.strict, n_perm=1000)
            row = dict(outcome=name, animal=an, epoch=ep, n_sessions=G.label.nunique(),
                       event_frac=float(G[flag].mean()), **r, excess_strict=rs["excess"], p_strict=rs["p"])
            if binary:
                row["clogit_coef"], row["clogit_se"], row["clogit_p"] = clogit(G.y, G[flag], np.log(G[cl]), G.stratum)
            rows.append(row)
        # acute + subacute together (where failures exist)
        G = B[B.epoch.isin(["acute", "subacute"])]
        for an, H in list(G.groupby("animal")) + [("ALL", G)]:
            r = excess(H.y, H[flag], H.stratum)
            rs = excess(H.y, H[flag], H.strict, n_perm=1000)
            row = dict(outcome=name, animal=an, epoch="acute+subacute", n_sessions=H.label.nunique(),
                       event_frac=float(H[flag].mean()), **r, excess_strict=rs["excess"], p_strict=rs["p"])
            if binary:
                row["clogit_coef"], row["clogit_se"], row["clogit_p"] = clogit(H.y, H[flag], np.log(H[cl]), H.stratum)
            rows.append(row)
    R = pd.DataFrame(rows)
    order = {e: i for i, e in enumerate(list(EP) + ["acute+subacute"])}
    R["_o"] = R.epoch.map(order)
    return R.sort_values(["outcome", "_o", "animal"]).drop(columns="_o").reset_index(drop=True)


def t13_by_spout(Tr):
    import pandas as pd
    from wfield_local.orofacial_cohort import POS_ORDER
    O = outcomes(Tr)
    rows = []
    for name in ("allspout_miss", "p_true_enl_event"):
        B, flag, _ = O[name]
        for ep in ("pre", "acute", "subacute", "chronic"):
            for sp in POS_ORDER:
                G = B[(B.epoch == ep) & (B.pos == sp)]
                rows.append(dict(outcome=name, epoch=ep, spout=sp, **excess(G.y, G[flag], G.stratum, n_perm=1000)))
    return pd.DataFrame(rows)


def t13_figure(R, RS, out=None):
    """Priya 11-03 rules: no connecting lines; per-animal values small FILLED (shape = animal, color = epoch);
    all-animal summaries large; by-spout epochs at cat_points.epoch_offsets in the canonical spout order. Per-SESSION
    values are not stored for T13 (the excess is computed over the pooled session x spout strata), so there are no
    session dots. Returns {(panel, kind, animal | epoch, epoch | spout): plotted value} for the --plot-only check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import cat_points as cp
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import EPOCH_LINE, POS_ORDER
    panels = [("farcontra_fail", "far-contra FAILED attempt (incomplete / no attempt / reaction)\nexcess P(fail) on "
               "event trials"),
              ("farcontra_incomplete_vs_hit", "far-contra INCOMPLETE vs hit\nexcess P(incomplete)"),
              ("allspout_miss", "all spouts, MISS vs hit (spout-matched)\nexcess P(miss)"),
              ("p_true_enl_event", "frozen pre-stroke decoder p(true spout), 2-s pre-cue window\nexcess p_true on "
               "trials with an event in that window")]
    fig, ax = plt.subplots(2, 3, figsize=(16, 11))
    xs = list(EP)
    plotted = {}
    for k, (name, title) in enumerate(panels):
        a = ax.flat[k]
        G = R[R.outcome == name]
        for x, ep in enumerate(xs):
            for an in ANIMALS:
                v = G[(G.animal == an) & (G.epoch == ep)]
                xa = x - 0.2 + 0.08 * ANIMALS.index(an)
                if len(v) and np.isfinite(v.excess.iloc[0]):
                    plotted[(name, "excess", an, ep)] = float(v.excess.iloc[0])
                    a.plot(xa, v.excess.iloc[0], cp.animal_marker(an), ls="none", color=EPOCH_LINE[ep], ms=4.5,
                           mec="none", alpha=0.35, zorder=3)
                if len(v) and np.isfinite(v.excess_strict.iloc[0]):
                    plotted[(name, "strict", an, ep)] = float(v.excess_strict.iloc[0])
                    a.plot(xa, v.excess_strict.iloc[0], cp.animal_marker(an), ls="none", color=EPOCH_LINE[ep], ms=4.5,
                           mec="w", mew=0.4, zorder=3)
            v = G[(G.animal == "ALL") & (G.epoch == ep)]
            if len(v) and np.isfinite(v.excess.iloc[0]):
                plotted[(name, "excess", "ALL", ep)] = float(v.excess.iloc[0])
                plotted[(name, "strict", "ALL", ep)] = float(v.excess_strict.iloc[0])
                a.plot(x + 0.22, v.excess.iloc[0], "_", color=EPOCH_LINE[ep], ms=16, mew=2.2, zorder=4)
                cp.summary_point(a, x + 0.22, v.excess_strict.iloc[0], None, EPOCH_LINE[ep], ms=8)
                a.text(x + 0.33, v.excess_strict.iloc[0], f"n={int(v.n_event.iloc[0])}", fontsize=6, va="center")
                if v.p_strict.iloc[0] < 0.05:
                    p = v.p_strict.iloc[0]
                    sig_marks.star_top(a, x + 0.22, "***" if p < 0.001 else "**" if p < 0.01 else "*",
                                       color=EPOCH_LINE[ep])
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xticks(range(len(xs)), xs, fontsize=8)
        a.set_xlim(-0.5, len(xs) - 0.5)
        a.set_title(title, fontsize=9)
        a.set_ylabel("event trials minus same-session same-spout mean")
    off = cp.epoch_offsets(len(EP), step=0.2)
    for k, (name, title) in enumerate((("allspout_miss", "excess P(miss) by spout"),
                                       ("p_true_enl_event", "excess p_true by spout"))):
        a = ax.flat[4 + k]
        G = RS[RS.outcome == name]
        for j, ep in enumerate(EP):
            H = G[G.epoch == ep].set_index("spout").reindex(POS_ORDER)
            x = np.arange(len(POS_ORDER)) + off[j]
            for sp, val in zip(POS_ORDER, H.excess):
                plotted[(name, "by_spout", ep, sp)] = float(val)
            cp.summary_point(a, x, H.excess.to_numpy(float), None, EPOCH_LINE[ep], ms=5.5, label=ep)
            for xi, p in zip(x, H.p):
                if np.isfinite(p) and p < 0.05:
                    sig_marks.star_top(a, xi, "*", color=EPOCH_LINE[ep], row=j, fontsize=10)
        a.axhline(0, color="0.6", lw=0.8)
        for xb in np.arange(len(POS_ORDER) - 1) + 0.5:
            a.axvline(xb, color="0.92", lw=0.6, zorder=0)
        a.set_xticks(range(len(POS_ORDER)), [p.replace("_", "\n") for p in POS_ORDER], fontsize=8)
        a.set_title(f"{title}, all animals pooled\n(stars, one row per epoch = within-stratum permutation\n"
                    "p < 0.05, uncorrected)", fontsize=9)
    handles = []
    for an in ANIMALS:
        handles.append(plt.Line2D([], [], marker=cp.animal_marker(an), ls="", color="0.35", mec="w", label=an))
    handles.append(plt.Line2D([], [], marker="o", ls="", color="0.35", alpha=0.35, mec="none",
                              label="animal, session x spout strata (faded)"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color="0.35", mec="w", label="animal, STRICT strata"))
    handles.append(plt.Line2D([], [], marker="_", ls="", color="k", ms=14, mew=2.2,
                              label="all animals, session x spout"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color="k", ms=8, mec="w", label="all animals, STRICT"))
    handles += [plt.Line2D([], [], marker="o", ls="", color=EPOCH_LINE[ep], label=ep) for ep in EP]
    fig.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=7, frameon=False)
    sig_marks.reserve_headroom(fig)
    fig.suptitle("T13: does a pre-cue orofacial event predict the trial's outcome or its pre-cue position code? Event = "
                 "any orofacial (ipsi or contra) event > 4 x pre SD, either polarity, in the trial's clean pre-cue "
                 "window (>= 1 s clean, no lick / run within 0.5 s). Effect = within-session, spout-matched excess "
                 "(event trials minus the mean of the same spout in the same session). Small filled = per animal "
                 "(shape; faded = session x spout strata, solid = STRICT strata: + trial-order quartile + clean-time "
                 "bin); large = all animals; stars = STRICT within-stratum permutation p (top band). Per-session values "
                 "are not stored (the excess pools session x spout strata), so no session dots. Engaged (canonical) "
                 "trials.", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.5, h_pad=2.0)
    fig.savefig(TE / "t13_precue_event_vs_outcome.png" if out is None else out, dpi=110)
    plt.close(fig)
    return plotted


def plot_only():
    """Re-render T12 / T13 from the STORED tables only (no extraction, no LMM, no permutations) as *_nolines.png and
    check the plotted numbers against those tables. (T12's top-row pooled CIs are drawn by boot_ci at render time, as
    in the original figure; they are not stored.)"""
    import pandas as pd
    S = pd.read_csv(TE / "t12_session_rates.csv")
    A = pd.read_csv(TE / "t12_animal_epoch_rates.csv")
    T = pd.read_csv(TE / "t12_lmm.csv")
    br = t12_figure(S, A, T, out=TE / "t12_task_window_event_rates_nolines.png")
    for (w, ep, an), v in br.items():
        sess = S[(S.animal == an) & (S.epoch == ep)][f"orofacial|{w}"].dropna()
        assert np.isclose(v, sess.mean()), (w, ep, an)           # stored animal mean == mean of its stored sessions
    print(f"T12 bottom-right: {len(br)} animal means (t12_animal_epoch_rates.csv) == mean of stored session rates")
    R = pd.read_csv(TE / "t13_effects.csv")
    RS = pd.read_csv(TE / "t13_by_spout.csv")
    P = t13_figure(R, RS, out=TE / "t13_precue_event_vs_outcome_nolines.png")
    for key, v in P.items():
        if key[1] == "by_spout":
            want = RS[(RS.outcome == key[0]) & (RS.epoch == key[2]) & (RS.spout == key[3])].excess
        else:
            col = "excess" if key[1] == "excess" else "excess_strict"
            want = R[(R.outcome == key[0]) & (R.animal == key[2]) & (R.epoch == key[3])][col]
        assert len(want) == 1 and np.isclose(v, want.iloc[0], equal_nan=True), key
    print(f"T13: {len(P)} plotted values == t13_effects.csv / t13_by_spout.csv")


def summarise():
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    D, sd = load_sessions()
    print(f"{len(D)} sessions:", pd.Series([d['epoch'] for d in D.values()]).value_counts().to_dict())
    S = session_rates(D, sd)
    S.to_csv(TE / "t12_session_rates.csv", index=False)
    A = animal_table(S)
    A.to_csv(TE / "t12_animal_epoch_rates.csv", index=False)
    print("\nclean pre-cue minutes / session (median) and fraction of clean pre-cue frames that are also v2-quiet:")
    print(S.groupby(["animal", "epoch"])[["precue_min", "precue_clean_frac", "precue_quiet_frac", "iti_min",
                                          "quiet_min"]].median().round(2).to_string())
    for c in ("orofacial_left", "orofacial_right", "orofacial"):
        print(f"\n=== T12 {c}: rate / min (mean over sessions), per animal x epoch; fold vs that animal's pre ===")
        X = A[(A.column == c) & A.window.isin(["precue", "precue_notquiet", "iti", "quiet"])]
        print(X.pivot_table(index=["window", "animal"], columns="epoch", values="rate")[EP].round(2).to_string())
        print(X.pivot_table(index=["window", "animal"], columns="epoch", values="fold_vs_pre")[EP[1:]].round(2)
              .to_string())
    T = t12_stats(S)
    T.to_csv(TE / "t12_lmm.csv", index=False)
    show = T[(T.term != "pre_level") & T.column.isin(["orofacial_left", "orofacial_right", "orofacial"])]
    print("\nT12 LMM (log10(rate + 0.2); log_precue_over_quiet = does the task-window change differ from quiet):\n",
          show[["model", "measure", "column", "term", "coef", "se", "p", "q", "re", "n"]].round(4).to_string(index=False))
    t12_figure(S, A, T)

    Tr = trial_table(D, sd)
    print(f"\nT13 trials: {len(Tr)}; matched to trial_attempts {Tr.cls.notna().mean():.3f}; position agrees "
          f"{(Tr.pos == Tr.spout_daq)[Tr.pos.notna()].mean():.4f}; p_true available {Tr.p_true.notna().mean():.3f}")
    Tr.to_csv(TE / "t13_trials.csv.gz", index=False)
    R = t13(Tr)
    R.to_csv(TE / "t13_effects.csv", index=False)
    cols = ["outcome", "animal", "epoch", "n_sessions", "n_trials", "n_event", "event_frac", "rate_event",
            "rate_noevent", "excess", "p", "excess_strict", "p_strict", "clogit_coef", "clogit_se", "clogit_p"]
    print("\n=== T13 per animal x epoch (ALL = pooled strata), within-session spout-matched ===")
    print(R[cols].round(4).to_string(index=False))
    RS = t13_by_spout(Tr)
    RS.to_csv(TE / "t13_by_spout.csv", index=False)
    print("\nT13 by spout (pooled):\n", RS.round(4).to_string(index=False))
    t13_figure(R, RS)
    (TE / "README.txt").write_text(__doc__)
    print("wrote", TE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--plot-only", action="store_true", help="re-render T12 / T13 from the stored tables -> *_nolines.png")
    a = ap.parse_args()
    if a.plot_only:
        plot_only()
        return
    if not a.summarise:
        from wfield_local.analysis_kit import curated_labels
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = a.only or [k for k in sorted(curated_labels()) if not config.early_pre(k) and k[:4] in ANIMALS]
        res, fail = fan_sessions(labels, extract, jobs=min(a.jobs, 5), label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        if a.only:
            return
        if len(fail) > 3:
            raise SystemExit("too many failures")
    summarise()


if __name__ == "__main__":
    main()
