"""FAR-CONTRA "REV-UP": the structure of the post-stroke first-lick delay -- BEHAVIOUR ONLY (Priya 10-29: "after the
stroke animals seem to 'rev up' before licking the far-contra spout: much longer latencies than pre-stroke, sometimes
starting a lick only after the spout has left"; addition 10-29: do NOT censor the first-lick search at the response
window or at spout departure).

QUESTION. Known (FINDINGS, attempt-based deficit / time to contact): acutely the far-contra first lick is ~+490 ms later
and the first contact ~+900 ms later. What is the delay made of -- (i) one slow continuous ramp (slowed initiation),
(ii) repeated aborted attempts, or (iii) a quiet wait and then a normal lick -- and how much of it spills past the spout?

METHOD (per trial, every trial of the curated sessions from 08-06; reporting set = the CANONICAL trials of THIS RUN'S
trial-attempts table (`lp_licks.attempts_path()` = `run_mode.attempts_dir()`; coordinator 11-05, Priya: behaviour-only
figures move to the camera working period): under the camera trial gate `canonical` = the camera WORKING PERIOD
(every trial not stopped / stopped_licking / early_run / after_end / unknown, `scripts/trial_attempts.py`); in the
untagged default run the behaviour pipeline's reference gate `orofacial_cohort.engagement_keep` as before (until 11-05
this script always read that untagged table); pre = the AUGUST pre-stroke sessions; epochs = `epochs.epoch_of`):
  times      position strobe / cue / trial stop from `trial_windows.trial_bounds` (DAQ; stop = DAQ `trial_end` TTL)
  spout      SPOUT-LEAVE = the trial stop (the spout retracts at the trial-end TTL and starts moving 60-80 ms later,
             STATUS 10-21 jaw handoff); search BOUND = the NEXT trial's position strobe (the spout has arrived at the next
             position; last trial: stop + the session's median stop -> next-strobe gap)
  first lick FIRST cam4 LP lick ONSET (whole-session `licks_all.csv`, daq_on_s -- the video sees licks without the
             spout; the DAQ spout sensor cannot) whose PEAK is >= cue + 70 ms (the session licks.csv rule, so in-window
             latencies equal first_lick_latency.py's cam4 column) and before the bound. NOT censored at the window /
             the stop. first_lick_phase:
               window      onset <= cue + response window (gui_config timing.response_window, 3.5 s) and <= stop
               late_spout  cue + window < onset <= stop (spout still in place)
               after_left  stop < onset < bound (in the ITI, the spout has left)
               none        no lick before the bound
  contact    first DAQ spout contact in cue + 70 ms .. stop (`pose_kinematics_demo.daq_contacts_ms`' rule)
  movement before the first lick (the rev-up):
    cam1_small  cam1 2-D detector events (`cam1_licks/<s>_events_all.csv`, droplets out) NOT matched to a cam4 lick =
                small / incomplete mouth openings cam4 does not register, counted cue .. first lick (n_cam1_pre)
    jaw         `qc/jaw_onsets.csv` (cam4 + cam1, cam1 floor 15 px; searched cue .. stop): onset, or its re-onset when
                the jaw was already deflected at the cue
    FR vigour   FaceRhythm master_pre_v1 (cam4), UNBLANKED (`face_rhythm_outputs.fr_signals(blank=False)`, absolute),
                non-directional lick vigour = sum of the six lick factors (fr_direction_mechanism.LICK); z vs the
                session's pre-cue (-0.5..0 s) distribution (median / 1.4826 MAD); fr_onset = first time after
                cue + 40 ms with z > FR_Z for FR_K consecutive 40-ms samples, searched up to the first lick.
                Also fr_prep_jaw (the factor set's preparatory-jaw factor), same z.
    move_ms     earliest of jaw / first cam1 event / fr_onset (<= the first lick): wait_ms = move_ms (cue -> first
                movement), revup_ms = latency - move_ms (first movement -> first lick); a trial with no movement before
                the lick has revup 0 (wait = latency)
    class       (licked trials) aborted = >= 2 cam1-only openings before the first lick; ramp = otherwise revup_ms >=
                RAMP_MS (300 ms); wait = otherwise (quiet, then a lick within 300 ms of the first movement);
                unknown = no jaw, cam1 or FR measurement in the trial. late = latency > the animal's pre-stroke 90th
                percentile at that spout.
  chance     spontaneous ITI licking: per session, the rate of ITI bout starts (a lick with none in the preceding 1 s,
             in stop + 1 s .. bound) -> expected after_left share = 1 - exp(-rate x (bound - stop)), per trial.
STATS. Per animal x spout x epoch tables FIRST (rule 8), then paired nested bootstrap (`analysis_kit.boot_delta`) of
per-session values: far_contra epoch vs pre, and the far_contra - mean(other five spouts) contrast epoch vs pre.
Survival (Kaplan-Meier, censored at the bound) and discrete hazard (100-ms bins) of first-lick onset after the cue.
RECOVERY (5a): session far-contra measures vs aim / iti_ang (recmark_sessions.csv) on post sessions: per-animal r and
the within-animal MixedLM slope p (lick_sspn_checks._track's model), next to the primary balance marker
(recprim_sessions.csv `marker`).

OUTPUTS  <OUT_W>/revup/ : sessions/<label>.pkl (per-session cache), revup_trials.csv.gz (per trial; join key = label +
trial_id), revup_fr.npz (FR vigour cue- / lick-aligned per trial, row order = revup_trials), revup_cells.csv (animal x
spout x epoch), revup_session.csv (session x spout), revup_boot.csv, revup_survival.csv, revup_track.csv;
<OUT_W>/figures/revup_*.png.
Usage: python scripts/revup_behaviour.py [--jobs 6] [--labels ...]
       python scripts/revup_behaviour.py --plot-only        (redraw from the saved CSVs / npz)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # rebuild root (11-03)

RUN_ROOT = _encoding_root()
OUT_W = RUN_ROOT / "iti_licks"
REV = OUT_W / "revup"
SESS = "behaviour_sessions"        # per-session cache (revup/sessions/ belongs to the imaging helper; its *.pkl from a
#                                    first, superseded run of this script = the onset-only lick rule, not read)
LICK_MIN_S = 0.07                  # a post-cue lick needs its PEAK >= cue + 70 ms (the licks.csv / trial_windows rule)
FIG = OUT_W / "figures"
QC = RUN_ROOT / "qc"                 # jaw_onsets.csv (one table for every run)


def attempts_csv():
    """THIS run's per-trial attempts table (camera classes under the camera trial gate; `lp_licks.attempts_path`)."""
    from wfield_local import lp_licks
    return lp_licks.attempts_path()
EP4 = ["pre", "acute", "subacute", "chronic"]
PHASES = ["window", "late_spout", "after_left", "none"]
CLASSES = ["wait", "ramp", "aborted", "unknown"]
LICK = ("fr_lick_ongoing_7Hz", "fr_lick_right", "fr_lick_left", "fr_lick_fast_close", "fr_lick_far_nose",
        "fr_lick_slow_late")
FR_SET = "master_pre_v1"
DT = 0.04
LAGS_CUE = np.round(np.arange(-0.5, 8.0 + 1e-9, DT), 3)
LAGS_LICK = np.round(np.arange(-3.0, 0.5 + 1e-9, DT), 3)
FR_Z, FR_K = 3.0, 3
RAMP_MS = 300.0
BOUT_GAP = 1.0
HAZ_BIN = 0.1
SURV_T = np.round(np.arange(0, 12.0 + 1e-9, 0.05), 3)


def _pos_order():
    from wfield_local import orofacial_cohort as oc
    return list(oc.POS_ORDER)


# ------------------------------------------------------------------------------------------------ per session
def _resp_window(label: str) -> float:
    import json

    from wfield_local.docked_periods import behaviour_session_dir
    sd = behaviour_session_dir(label)
    try:
        return float(json.loads((Path(sd) / "gui_config.json").read_text())["timing"]["response_window"]) / 1000.0
    except (OSError, KeyError, ValueError, TypeError):
        return 3.5


def _first_at(sorted_t: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """first value of sorted_t in [lo, hi) per row (NaN if none)."""
    lo = np.asarray(lo, float)
    if not len(sorted_t):
        return np.full(len(lo), np.nan)
    j = np.searchsorted(sorted_t, lo, side="left")
    v = np.where(j < len(sorted_t), sorted_t[np.minimum(j, len(sorted_t) - 1)], np.nan)
    return np.where(v < hi, v, np.nan)


def _count(sorted_t: np.ndarray, lo, hi) -> np.ndarray:
    return np.searchsorted(sorted_t, hi, side="left") - np.searchsorted(sorted_t, lo, side="left")


def session(label: str) -> dict:
    """One session -> per-trial table + FR arrays; cached at REV/sessions/<label>.pkl (module-level for spawn)."""
    f = REV / SESS / f"{label}.pkl"
    if f.exists():
        return {"label": label, "cached": True}
    from wfield_local import config, dlc_frames, epochs
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    b = TW.trial_bounds(an, date, rv).sort_values("cue_s").reset_index(drop=True)
    cue, stop, strobe = (b[c].to_numpy(float) for c in ("cue_s", "stop_s", "strobe_s"))
    nxt = np.r_[strobe[1:], np.nan]
    gap = np.nanmedian(nxt - stop) if np.isfinite(nxt - stop).any() else 5.0
    nxt = np.where(np.isfinite(nxt) & (nxt > stop), nxt, stop + gap)
    rw = _resp_window(label)
    sd = oo.session_dir(rv, oc.RUN, an, date)
    A = pd.read_csv(sd / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s"])
    A = A[np.isfinite(A.daq_on_s) & np.isfinite(A.daq_peak_s)].sort_values("daq_peak_s")
    lk = np.sort(A.daq_on_s.to_numpy(float))
    pk, pk_on = A.daq_peak_s.to_numpy(float), A.daq_on_s.to_numpy(float)
    cov = {}
    cf = sd / "licks_all_coverage.json"
    if cf.exists():
        import json
        cov = json.loads(cf.read_text())
    # first lick (peak >= cue + 70 ms, the licks.csv rule; time = its ONSET), uncensored to the next strobe
    j = np.searchsorted(pk, cue + LICK_MIN_S, side="left")
    fl = np.where(j < len(pk), pk_on[np.minimum(j, len(pk) - 1)], np.nan)
    fl = np.where(np.isfinite(fl) & (pk[np.minimum(j, len(pk) - 1)] < nxt), np.maximum(fl, cue), np.nan)
    lat = fl - cue
    ph = np.where(~np.isfinite(fl), "none",
                  np.where(fl <= np.minimum(cue + rw, stop), "window", np.where(fl <= stop, "late_spout", "after_left")))
    # DAQ contacts (the repo's rule: cue + 70 ms .. stop)
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / an / date).glob("*_trials.csv"))[-1].name[:-11]
    ct = np.sort(np.asarray(dlc_frames.lick_onsets(an, date, sid, rv), float))
    fc = _first_at(ct, cue + 0.07, stop + 1e-9)
    # licks between first lick and first contact (missed licks); without a contact: licks cue .. stop
    n_miss = np.where(np.isfinite(fc), _count(lk, fl, fc), _count(lk, cue, stop))
    n_miss = np.where(np.isfinite(fl), n_miss, 0)
    # cam1 events
    e1 = Path(rv.root("dlc")) / "session_poses" / "cam1_licks" / f"{an}_{date}_events_all.csv"
    has_cam1 = e1.exists()
    if has_cam1:
        E = pd.read_csv(e1, usecols=["daq_on_s", "cam4_matched", "droplet"])
        E = E[~E.droplet.fillna(False).astype(bool) & np.isfinite(E.daq_on_s)]
        e_all = np.sort(E.daq_on_s.to_numpy(float))
        e_small = np.sort(E.loc[~E.cam4_matched.fillna(False).astype(bool), "daq_on_s"].to_numpy(float))
        upto = np.where(np.isfinite(fl), fl, nxt)
        n_cam1_pre = _count(e_small, cue, upto).astype(float)
        first_cam1 = _first_at(e_all, cue, upto + 1e-9) - cue
        n_cam1_pre_win = _count(e_small, cue, np.minimum(upto, stop)).astype(float)
    else:
        n_cam1_pre = first_cam1 = n_cam1_pre_win = np.full(len(b), np.nan)
    # jaw onsets (cue .. stop)
    J = _jaw(f"{an}_{date}")
    jt = b[["trial_id"]].merge(J, on="trial_id", how="left")
    jaw_ms = np.where(jt.onset_at_cue.fillna(False).astype(bool), jt.jaw_reonset_ms, jt.jaw_onset_ms).astype(float)
    jaw_judged = (jt.cam4_moved.notna() | jt.cam1_moved.notna()).to_numpy()
    # FaceRhythm vigour
    out_fr = {"cue": np.full((len(b), len(LAGS_CUE)), np.nan, np.float32),
              "lick": np.full((len(b), len(LAGS_LICK)), np.nan, np.float32),
              "prep_cue": np.full((len(b), len(LAGS_CUE)), np.nan, np.float32)}
    fr_onset = prep_onset = np.full(len(b), np.nan)
    fr_pre300 = np.full(len(b), np.nan)
    has_fr = False
    try:
        fdir = fro.factor_dir(rv, set_name=FR_SET)
        g1 = cue[:, None] + LAGS_CUE[None, :]
        g2 = np.where(np.isfinite(fl), fl, np.nan)[:, None] + LAGS_LICK[None, :]
        tt = np.concatenate([g1.ravel(), g2.ravel()])
        ok = np.isfinite(tt)
        o = np.argsort(tt[ok])
        sig = fro.fr_signals(an, date, rv, tt[ok][o], fdir=fdir, blank=False, absolute=True)
        vig = np.zeros(ok.sum())
        for k in LICK:
            vig = vig + np.asarray(sig[k][1], float)
        prep = np.asarray(sig["fr_prep_jaw"][1], float)
        V, P = np.full(len(tt), np.nan), np.full(len(tt), np.nan)
        idx = np.flatnonzero(ok)[o]
        V[idx], P[idx] = vig, prep
        n1 = g1.size
        vc, pc, vl = V[:n1].reshape(g1.shape), P[:n1].reshape(g1.shape), V[n1:].reshape(g2.shape)
        base = LAGS_CUE < 0

        def z(Xc, *Xs):
            bv = Xc[:, base].ravel()
            bv = bv[np.isfinite(bv)]
            med = np.median(bv)
            s = 1.4826 * np.median(np.abs(bv - med)) or np.std(bv) or 1.0
            return [(x - med) / s for x in (Xc, *Xs)]
        vcz, vlz = z(vc, vl)
        pcz, = z(pc)
        out_fr = {"cue": vcz.astype(np.float32), "lick": vlz.astype(np.float32), "prep_cue": pcz.astype(np.float32)}
        has_fr = bool(np.isfinite(vcz).any())
        fr_onset = _onset(vcz, lat)
        prep_onset = _onset(pcz, lat)
        w = (LAGS_LICK >= -0.3) & (LAGS_LICK < 0)
        fr_pre300 = np.nanmean(vlz[:, w], 1) if np.isfinite(vlz[:, w]).any() else fr_pre300
    except FileNotFoundError as ex:
        print(f"  {label}: no FaceRhythm ({ex})", flush=True)
    # spontaneous ITI bout-start rate
    prev = np.r_[-np.inf, lk[:-1]]
    starts = lk[(lk - prev) >= BOUT_GAP]
    lo_i, hi_i = stop + 1.0, nxt
    dur = np.clip(hi_i - lo_i, 0, None)
    n_st = _count(starts, lo_i, np.maximum(hi_i, lo_i))
    rate = float(n_st.sum() / dur.sum()) if dur.sum() > 0 else np.nan
    T = pd.DataFrame({
        "label": label, "animal": an, "epoch": epochs.epoch_of(label), "trial_id": b.trial_id.astype(int),
        "trial_k": np.arange(len(b)), "pos": [oc.pos_key(p, "L") for p in b.pos_name],
        "strobe_s": strobe, "cue_s": cue, "resp_end_s": cue + rw, "spout_leave_s": stop, "bound_s": nxt,
        "first_lick_s": fl, "first_contact_s": fc, "lat_ms": lat * 1000, "contact_ms": (fc - cue) * 1000,
        "first_lick_phase": ph, "stop_ms": (stop - cue) * 1000, "bound_ms": (nxt - cue) * 1000,
        "n_miss_before_contact": n_miss, "has_cam1": has_cam1, "n_cam1_pre": n_cam1_pre,
        "n_cam1_pre_spout": n_cam1_pre_win, "first_cam1_ms": first_cam1 * 1000, "jaw_ms": jaw_ms,
        "jaw_judged": jaw_judged, "has_fr": has_fr, "fr_onset_ms": fr_onset * 1000, "prep_onset_ms": prep_onset * 1000,
        "fr_pre300_z": fr_pre300, "iti_bout_rate_hz": rate,
        "p_spont_after_left": 1 - np.exp(-rate * np.clip(nxt - stop, 0, None)) if np.isfinite(rate) else np.nan,
        "coverage": float(cov.get("covered_frac", np.nan))})
    T = _revup(T)
    REV.joinpath(SESS).mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps({"T": T, "fr": out_fr}))
    return {"label": label, "cached": False, "n": len(T)}


def _onset(Z: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """first cue-aligned lag (s) >= 40 ms with FR_K consecutive samples > FR_Z, before the first lick (NaN if none)."""
    out = np.full(len(Z), np.nan)
    a = (Z > FR_Z).astype(int)
    run = np.zeros_like(a)
    for k in range(FR_K):
        run[:, :a.shape[1] - k] += a[:, k:]
    ok = (run >= FR_K) & (LAGS_CUE[None, :] >= 0.04)
    ok &= LAGS_CUE[None, :] <= np.where(np.isfinite(lat), lat, LAGS_CUE[-1])[:, None]
    has = ok.any(1)
    out[has] = LAGS_CUE[ok[has].argmax(1)]
    return out


def _revup(T: pd.DataFrame) -> pd.DataFrame:
    lat = T.lat_ms.to_numpy(float)
    cand = np.column_stack([T.jaw_ms, T.first_cam1_ms, T.fr_onset_ms]).astype(float)
    cand = np.where(cand <= lat[:, None] + 1e-6, cand, np.nan)
    measured = T.jaw_judged.to_numpy() | T.has_cam1.to_numpy() | T.has_fr.to_numpy()
    mv = np.nanmin(np.where(np.isfinite(cand), cand, np.inf), 1)
    mv = np.where(np.isfinite(mv), mv, np.where(np.isfinite(lat) & measured, lat, np.nan))
    T["move_ms"] = mv
    T["wait_ms"] = mv
    T["revup_ms"] = lat - mv
    licked = np.isfinite(lat)
    cls = np.where(T.n_cam1_pre.fillna(0) >= 2, "aborted", np.where(T.revup_ms >= RAMP_MS, "ramp", "wait"))
    cls = np.where(~measured, "unknown", cls)
    T["revup_class"] = np.where(licked, cls, "")
    return T


_JAW = {}


def _jaw(session: str) -> pd.DataFrame:
    if not _JAW:
        J = pd.read_csv(QC / "jaw_onsets.csv", usecols=["session", "trial_id", "cam4_moved", "cam1_moved",
                                                       "jaw_onset_ms", "onset_at_cue", "jaw_reonset_ms"],
                        low_memory=False)
        J["onset_at_cue"] = J.onset_at_cue.astype(str).str.lower().eq("true")
        _JAW.update({s: g.drop(columns="session") for s, g in J.groupby("session")})
    return _JAW.get(session, pd.DataFrame(columns=["trial_id", "cam4_moved", "cam1_moved", "jaw_onset_ms",
                                                   "onset_at_cue", "jaw_reonset_ms"]))


# ------------------------------------------------------------------------------------------------ assemble
def session_labels() -> list[str]:
    from wfield_local import config
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    A = pd.read_csv(attempts_csv(), usecols=["label"]).label.unique()
    cur = set(ak.curated_labels())
    return sorted(x for x in A if x in cur and not config.early_pre(x) and epochs.epoch_of(x) in EP4)


def assemble(labels: list[str]) -> tuple[pd.DataFrame, dict]:
    TA = pd.read_csv(attempts_csv(), usecols=["label", "trial_id", "cls", "canonical", "attempted", "day"])
    Ts, F = [], {"cue": [], "lick": [], "prep_cue": []}
    for x in labels:
        d = pickle.loads((REV / SESS / f"{x}.pkl").read_bytes())
        Ts.append(d["T"])
        for k in F:
            F[k].append(d["fr"][k])
    T = pd.concat(Ts, ignore_index=True)
    T = T.merge(TA, on=["label", "trial_id"], how="left")
    T["canonical"] = T.canonical.astype(str).str.lower().eq("true")
    T["day"] = T.groupby("label").day.transform("first")
    # late = latency above the animal's pre-stroke 90th percentile at that spout (canonical pre trials, any phase)
    P = T[(T.epoch == "pre") & T.canonical].groupby(["animal", "pos"]).lat_ms.quantile(0.9).rename("pre_p90_ms")
    T = T.merge(P.reset_index(), on=["animal", "pos"], how="left")
    T["late"] = np.where(np.isfinite(T.lat_ms), T.lat_ms > T.pre_p90_ms, np.nan)
    return T, {k: np.concatenate(v, 0) for k, v in F.items()}


# ------------------------------------------------------------------------------------------------ summaries
def _km(lat_s: np.ndarray, cens_s: np.ndarray) -> np.ndarray:
    """Kaplan-Meier S(t) on SURV_T: event = first lick at lat_s; censored at cens_s when no lick."""
    ev = np.isfinite(lat_s)
    t = np.where(ev, lat_s, cens_s)
    o = np.argsort(t)
    t, ev = t[o], ev[o]
    n = len(t)
    at_risk = n - np.arange(n)
    f = np.where(ev, 1 - 1 / at_risk, 1.0)
    S = np.cumprod(f)
    j = np.searchsorted(t, SURV_T, side="right")
    return np.where(j > 0, S[np.maximum(j - 1, 0)], 1.0)


def _cell(g: pd.DataFrame) -> dict:
    lat = g.lat_ms.to_numpy(float)
    lk = lat[np.isfinite(lat)]
    r = {"n": len(g), "lat_med_ms": np.median(lk) if len(lk) else np.nan,
         "lat_p75_ms": np.percentile(lk, 75) if len(lk) else np.nan,
         "lat_p90_ms": np.percentile(lk, 90) if len(lk) else np.nan,
         "contact_med_ms": np.nanmedian(g.contact_ms) if g.contact_ms.notna().any() else np.nan,
         "no_contact": float(g.contact_ms.isna().mean())}
    for p in PHASES:
        r[f"sh_{p}"] = float((g.first_lick_phase == p).mean())
    nl = g.first_lick_phase.isin(["after_left", "none"])
    r["after_left_given_nolick"] = float((g.first_lick_phase[nl] == "after_left").mean()) if nl.any() else np.nan
    r["sh_after_left_expected"] = float(np.nanmean(np.where(g.first_lick_phase.isin(["after_left", "none"]),
                                                            g.p_spont_after_left, np.nan))
                                        * g.first_lick_phase.isin(["after_left", "none"]).mean()) \
        if g.first_lick_phase.isin(["after_left", "none"]).any() else 0.0
    L = g[np.isfinite(g.lat_ms)]
    for c in ("wait_ms", "revup_ms"):
        r[f"{c[:-3]}_med_ms"] = np.nanmedian(L[c]) if L[c].notna().any() else np.nan
    r["n_cam1_pre_mean"] = float(np.nanmean(L.n_cam1_pre)) if L.n_cam1_pre.notna().any() else np.nan
    r["n_miss_mean"] = float(L.n_miss_before_contact.mean()) if len(L) else np.nan
    r["fr_pre300_z_med"] = np.nanmedian(L.fr_pre300_z) if L.fr_pre300_z.notna().any() else np.nan
    late = L[L.late == 1]
    r["sh_late"] = float((L.late == 1).mean()) if len(L) else np.nan
    r["n_late"] = len(late)
    for c in CLASSES:
        r[f"late_{c}"] = float((late.revup_class == c).mean()) if len(late) else np.nan
        r[f"all_{c}"] = float((L.revup_class == c).mean()) if len(L) else np.nan
    r["late_wait_med_ms"] = np.nanmedian(late.wait_ms) if len(late) else np.nan
    r["late_revup_med_ms"] = np.nanmedian(late.revup_ms) if len(late) else np.nan
    return r


SESS_MEAS = ["lat_med_ms", "contact_med_ms", "sh_window", "sh_late_spout", "sh_after_left", "sh_none",
             "after_left_given_nolick", "lat_p90_ms",
             "wait_med_ms", "revup_med_ms", "n_cam1_pre_mean", "n_miss_mean", "fr_pre300_z_med", "sh_late",
             "late_wait_med_ms", "late_revup_med_ms", "late_aborted", "late_ramp", "late_wait"]


def summarise(T: pd.DataFrame, F: dict) -> None:
    from wfield_local import analysis_kit as ak
    POS = _pos_order()
    C = T[T.canonical]
    # 1. per animal x spout x epoch (pooled trials)
    rows = [{"animal": a, "pos": p, "epoch": e, **_cell(g)} for (a, p, e), g in C.groupby(["animal", "pos", "epoch"])]
    cells = pd.DataFrame(rows)
    cells["pos"] = pd.Categorical(cells.pos, POS, ordered=True)
    cells["epoch"] = pd.Categorical(cells.epoch, EP4, ordered=True)
    cells = cells.sort_values(["animal", "pos", "epoch"])
    cells.to_csv(REV / "revup_cells.csv", index=False, float_format="%.4g")
    # 2. per session x spout
    S = pd.DataFrame([{"label": x, "animal": a, "epoch": e, "pos": p, "day": g.day.iloc[0], **_cell(g)}
                      for (x, a, e, p), g in C.groupby(["label", "animal", "epoch", "pos"])])
    S.to_csv(REV / "revup_session.csv", index=False, float_format="%.4g")
    # 3. survival / hazard per animal x spout x epoch
    sv = []
    for (a, p, e), g in C.groupby(["animal", "pos", "epoch"]):
        Sv = _km(g.lat_ms.to_numpy(float) / 1000, g.bound_ms.to_numpy(float) / 1000)
        sv.append(pd.DataFrame({"animal": a, "pos": p, "epoch": e, "t_s": SURV_T, "S": Sv}))
    SV = pd.concat(sv, ignore_index=True)
    st = SV.t_s.to_numpy()
    k = int(round(HAZ_BIN / 0.05))
    SV["hazard_per_s"] = np.nan
    for _, idx in SV.groupby(["animal", "pos", "epoch"]).groups.items():
        s = SV.loc[idx, "S"].to_numpy()
        h = np.full(len(s), np.nan)
        h[:-k] = np.where(s[:-k] > 0.02, (s[:-k] - s[k:]) / np.maximum(s[:-k], 1e-9) / HAZ_BIN, np.nan)
        SV.loc[idx, "hazard_per_s"] = h
    del st
    SV.to_csv(REV / "revup_survival.csv", index=False, float_format="%.4g")
    # 4. bootstrap
    rng_seed = 0
    out = []
    for m in SESS_MEAS:
        W = S.pivot_table(index=["label", "animal", "epoch"], columns="pos", values=m).reset_index()
        if "far_contra" not in W:
            continue
        oth = [p for p in POS if p != "far_contra" and p in W]
        W["fc"] = W["far_contra"]
        W["contrast"] = W["far_contra"] - W[oth].mean(1)
        for kind in ("fc", "contrast"):
            pre = {a: g[kind].dropna().tolist() for a, g in W[W.epoch == "pre"].groupby("animal")}
            for e in EP4[1:]:
                post = {a: g[kind].dropna().tolist() for a, g in W[W.epoch == e].groupby("animal")}
                pre_ = {a: v for a, v in pre.items() if v}
                post_ = {a: v for a, v in post.items() if v}
                iv, dr = ak.boot_delta(post_, pre_, np.random.default_rng(rng_seed), draws=True)
                per = {f"d_{a}": np.mean(post_[a]) - np.mean(pre_[a]) for a in sorted(set(pre_) & set(post_))}
                out.append({"measure": m, "kind": "far_contra" if kind == "fc" else "far_contra - others",
                            "epoch": e, "delta": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan,
                            "hi": iv.hi if iv else np.nan, "p": ak.boot_p(dr) if iv else np.nan,
                            "n_animals": iv.n_animals if iv else 0,
                            "n_same_sign": int(sum(np.sign(v) == np.sign(iv.point) for v in per.values()))
                            if iv else 0, **per})
    B = pd.DataFrame(out)
    B.to_csv(REV / "revup_boot.csv", index=False, float_format="%.4g")
    # 5. recovery tracking
    track(S).to_csv(REV / "revup_track.csv", index=False, float_format="%.4g")
    # trial table
    keep = ["label", "animal", "epoch", "day", "trial_id", "trial_k", "pos", "canonical", "cls", "attempted",
            "strobe_s", "cue_s", "resp_end_s", "spout_leave_s", "bound_s", "first_lick_s", "first_contact_s",
            "lat_ms", "contact_ms", "first_lick_phase", "stop_ms", "bound_ms", "late", "pre_p90_ms",
            "n_miss_before_contact", "has_cam1", "n_cam1_pre", "n_cam1_pre_spout", "first_cam1_ms", "jaw_ms",
            "jaw_judged", "has_fr", "fr_onset_ms", "prep_onset_ms", "fr_pre300_z", "move_ms", "wait_ms", "revup_ms",
            "revup_class", "p_spont_after_left", "iti_bout_rate_hz", "coverage"]
    T[keep].to_csv(REV / "revup_trials.csv.gz", index=False, float_format="%.5g")
    np.savez_compressed(REV / "revup_fr.npz", lags_cue=LAGS_CUE, lags_lick=LAGS_LICK,
                        cue=F["cue"].astype(np.float16), lick=F["lick"].astype(np.float16),
                        prep_cue=F["prep_cue"].astype(np.float16), label=T.label.to_numpy(str),
                        trial_id=T.trial_id.to_numpy(int))


def track(S: pd.DataFrame) -> pd.DataFrame:
    """lick_sspn_checks._track's model on far-contra session measures (and their far_contra - others contrast), plus
    the primary balance marker (recprim_sessions.csv) on the same post sessions for comparison."""
    import statsmodels.formula.api as smf
    POS = _pos_order()
    R = pd.read_csv(OUT_W / "recmark_sessions.csv", usecols=["label", "aim", "iti_ang"]).drop_duplicates("label")
    M = pd.read_csv(OUT_W / "recprim_sessions.csv")
    M = M[M.panel == "cued first lick -> aim"][["label", "marker"]].drop_duplicates("label")
    meas = ["lat_med_ms", "contact_med_ms", "sh_after_left", "sh_none", "sh_late", "wait_med_ms", "revup_med_ms",
            "late_wait_med_ms", "late_revup_med_ms"]
    W = S[S.pos == "far_contra"][["label", "animal", "epoch", "day", *meas]].copy()
    O = S[S.pos != "far_contra"].groupby("label")[meas].mean()
    for m in meas:
        W[f"{m}_vsoth"] = W[m].to_numpy() - O.reindex(W.label)[m].to_numpy()
    W = W.merge(R, on="label", how="left").merge(M, on="label", how="left")
    cols = [*meas, *[f"{m}_vsoth" for m in meas], "marker"]
    G0 = W[(W.epoch != "pre") & (W.day > 0)].copy()
    G0["logday"] = np.log(G0.day)
    out = []
    for yv in ("aim", "iti_ang"):
        G = G0.dropna(subset=[yv])
        for c in cols:
            g = G.dropna(subset=[c]).copy()
            if g.animal.nunique() < 4:
                continue
            rr = {a: np.corrcoef(h[c], h[yv])[0, 1] if len(h) > 2 else np.nan for a, h in g.groupby("animal")}
            g["w"] = g[c] - g.groupby("animal")[c].transform("mean")
            g["b"] = g.groupby("animal")[c].transform("mean")
            g["yy"] = g[yv]
            try:
                fit = smf.mixedlm("yy ~ w + b + logday", g, groups=g["animal"]).fit(reml=False)
                p, sl = fit.pvalues["w"], fit.params["w"]
            except Exception:                                          # noqa: BLE001
                p = sl = np.nan
            out.append({"y": yv, "measure": c, "n_sessions": len(g), "slope_within": sl, "p_within": p,
                        "n_neg": int(sum(v < 0 for v in rr.values() if np.isfinite(v))),
                        "n_pos": int(sum(v > 0 for v in rr.values() if np.isfinite(v))),
                        **{f"r_{a}": v for a, v in rr.items()}})
    del POS
    W.to_csv(REV / "revup_track_sessions.csv", index=False, float_format="%.4g")
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ figures
PH_COL = {"window": "#4c78a8", "late_spout": "#e8a33d", "after_left": "#c44e52", "none": "#bdbdbd"}
CL_COL = {"wait": "#8fb8de", "ramp": "#e8a33d", "aborted": "#c44e52", "unknown": "#d9d9d9"}


def _sup(fig, text: str, fontsize: float = 9) -> float:
    """Suptitle wrapped to the figure width (<= 150 chars / line; explicit line breaks kept); returns the top of the
    axes area for tight_layout(rect=(0, 0, 1, top)) so the title never overlaps or runs off the page."""
    import textwrap
    w = min(150, int(fig.get_figwidth() * 13.5 * 9 / fontsize))
    lines = [ln for para in text.split("\n") for ln in textwrap.wrap(para, w)]
    fig.suptitle("\n".join(lines), fontsize=fontsize)
    # tight_layout (matplotlib >= 3.3) already reserves the suptitle's own box, so reserving its height again here left a
    # large empty band between the title and the panels (figure-legibility pass, 10-29): return the full height.
    return 1.0


def _leg_below(fig, handles, ncol: int | None = None, fontsize: float = 8) -> float:
    """ONE figure-level legend centred below the panels (never on top of data); returns the bottom of the axes area for
    tight_layout(rect=(0, bottom, 1, top))."""
    ncol = ncol or len(handles)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=fontsize, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    nrow = -(-len(handles) // ncol)
    return (nrow * fontsize * 1.6 + 8) / 72 / fig.get_figheight()


def _grp_labels(ax, groups=(("other five", 1.5), ("far_contra", 6.2))) -> None:
    """Group names under the epoch tick labels of a two-group (other / far_contra) bar panel (x = data, y = axes)."""
    tr = ax.get_xaxis_transform()
    for nm, x in groups:
        ax.text(x, -0.17, nm, transform=tr, ha="center", va="top", fontsize=9, fontweight="bold")


def _ep_col():
    from wfield_local.epoch_figures import EPOCH_GREY
    return dict(EPOCH_GREY) | {"pre": "#4c78a8"}


def plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    POS = _pos_order()
    EC = _ep_col()
    cells = pd.read_csv(REV / "revup_cells.csv")
    SV = pd.read_csv(REV / "revup_survival.csv")
    T = pd.read_csv(REV / "revup_trials.csv.gz", low_memory=False)
    AN = sorted(cells.animal.unique())
    # Fig 1: survival of first lick, rows = spouts (canonical), cols = animals
    fig, axs = plt.subplots(len(POS), len(AN), figsize=(3.0 * len(AN), 2.0 * len(POS)), sharex=True, sharey=True)
    for i, p in enumerate(POS):
        for j, a in enumerate(AN):
            ax = axs[i, j]
            for e in EP4:
                g = SV[(SV.animal == a) & (SV.pos == p) & (SV.epoch == e)]
                if len(g):
                    ax.plot(g.t_s, g.S, color=EC[e], lw=1.6, label=e)
            ax.axvline(3.5, color="k", lw=0.6, ls=":")
            ax.set_xlim(0, 8)
            if i == 0:
                ax.set_title(a)
            if j == 0:
                ax.set_ylabel(f"{p}\nP(no lick yet)")
            if i == len(POS) - 1:
                ax.set_xlabel("time from cue (s)")
    bot = _leg_below(fig, [plt.Line2D([], [], color=EC[e], lw=1.6, label=e) for e in EP4], fontsize=9)
    top = _sup(fig, "Fig R1. First-lick survival after the cue (Kaplan-Meier; cam4 video licks, NOT censored at the window;"
                 "\ncensored at the next trial's position strobe). Dotted = end of the 3.5-s response window (the spout"
                 " leaves ~0.2 s later). Canonical engaged trials. BEHAVIOUR", fontsize=9)
    fig.tight_layout(rect=(0, bot, 1, top))
    save_png_svg(fig, FIG / "revup_survival.png", dpi=120)
    plt.close(fig)
    # Fig 2: phase shares per animal x epoch, per spout
    # 2 x 3 grid (was 1 x 6 at 3 in per panel: 16 bars per panel, unreadable tick labels)
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.6), sharey=True)
    axs = axs.ravel()
    for i, p in enumerate(POS):
        ax = axs[i]
        xs, labs = [], []
        x = 0
        for a in AN:
            for e in EP4:
                g = cells[(cells.animal == a) & (cells.pos == p) & (cells.epoch == e)]
                if len(g):
                    bot = 0.0
                    for ph in PHASES:
                        v = float(g[f"sh_{ph}"].iloc[0])
                        ax.bar(x, v, bottom=bot, color=PH_COL[ph], width=0.85, edgecolor="white", lw=0.5)
                        bot += v
                    xs.append(x)
                    labs.append(e[:3])
                x += 1
            x += 0.8
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, rotation=90, fontsize=8)
        ax.set_title(p, fontsize=11)
        for k, a in enumerate(AN):
            ax.text(k * 4.8 + 1.5, 1.02, a, ha="center", fontsize=9)
        ax.set_ylim(0, 1.09)
    axs[0].set_ylabel("share of trials")
    axs[3].set_ylabel("share of trials")
    bot = _leg_below(fig, [plt.Rectangle((0, 0), 1, 1, color=PH_COL[ph], label=ph) for ph in PHASES], fontsize=10)
    top = _sup(fig, "Fig R2. Where the first post-cue lick falls: response window / after the window with the spout still "
                 "there\n / after the spout left (ITI, before the next strobe) / never. Per animal x epoch. BEHAVIOUR",
                 fontsize=10)
    fig.tight_layout(rect=(0, bot, 1, top))
    save_png_svg(fig, FIG / "revup_phase_shares.png", dpi=120)
    plt.close(fig)
    # Fig 2 SIMPLE: only the three NON-window phases (the window share is 1 minus their sum), per spout x epoch:
    # bar = mean of the animals' cell shares (display only), markers = the animals. Same revup_cells.csv numbers.
    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    # late_spout is ~0 in every cell (max 0.011), so it is left out of the simple view (it stays in the full figure)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for k, ph in enumerate(("after_left", "none")):
        ax = axs[k]
        for i, p in enumerate(POS):
            for jj, e in enumerate(EP4):
                g = cells[(cells.pos == p) & (cells.epoch == e)]
                if not len(g):
                    continue
                x = i * 5 + jj
                v = g[f"sh_{ph}"].to_numpy(float)
                ax.bar(x, np.nanmean(v), color=EC[e], width=0.8)
                for an, vv in zip(g.animal, v):
                    ax.scatter(x, vv, s=16, color="white", edgecolor="k", lw=0.7, marker=ANIMAL_MARKER.get(an, "o"),
                               zorder=3)
        ax.set_xticks([i * 5 + 1.5 for i in range(len(POS))])
        ax.set_xticklabels([p.replace("_", "\n") for p in POS], fontsize=9)
        ax.set_title({"after_left": "first lick after the spout left (ITI)",
                      "none": "no lick before the next trial"}[ph], fontsize=11)
    axs[0].set_ylabel("share of trials (mean of animals)")
    hd = [plt.Rectangle((0, 0), 1, 1, color=EC[e], label=e) for e in EP4] + \
         [plt.Line2D([], [], ls="", marker=ANIMAL_MARKER.get(a, "o"), mfc="white", mec="k", label=a) for a in AN]
    bot = _leg_below(fig, hd, fontsize=9)
    top = _sup(fig, "Fig R2 (simple). Share of first post-cue licks that fall OUTSIDE the response window, by spout x epoch "
                 "(the rest fall in the window; 'after the window, spout still there' is <= 1.1 % in every cell and not drawn). Bar = mean "
                 "of animals, markers = animals. BEHAVIOUR", fontsize=10)
    fig.tight_layout(rect=(0, bot, 1, top))
    save_png_svg(fig, FIG / "revup_phase_shares_simple.png", dpi=120)
    plt.close(fig)
    # Fig 3: decomposition of latency (wait + revup) and rev-up classes, far_contra vs others
    C = T[T.canonical.astype(str).str.lower().eq("true") & np.isfinite(T.lat_ms)].copy()
    C["grp"] = np.where(C.pos == "far_contra", "far_contra", "other")
    fig, axs = plt.subplots(2, len(AN), figsize=(3.6 * len(AN), 7.6))
    for j, a in enumerate(AN):
        ax = axs[0, j]
        x = 0
        xt, xl = [], []
        for grp in ("other", "far_contra"):
            for e in EP4:
                g = C[(C.animal == a) & (C.grp == grp) & (C.epoch == e)]
                if not len(g):
                    x += 1
                    continue
                w, r = np.nanmedian(g.wait_ms), np.nanmedian(g.revup_ms)
                ax.bar(x, w, color="#bdbdbd", edgecolor="white")
                ax.bar(x, r, bottom=w, color="#e8a33d", edgecolor="white")
                ax.plot(x, np.nanmedian(g.lat_ms), "k_", ms=12)
                xt.append(x)
                xl.append(e[:3])
                x += 1
            x += 0.7
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=8)
        _grp_labels(ax)
        ax.set_title(a)
        if j == 0:
            ax.set_ylabel("median ms\n(black = median latency)")
        ax = axs[1, j]
        x = 0
        xt, xl = [], []
        for grp in ("other", "far_contra"):
            for e in EP4:
                g = C[(C.animal == a) & (C.grp == grp) & (C.epoch == e) & (C.late == 1)]
                if len(g) < 5:
                    x += 1
                    continue
                bot = 0
                for c in CLASSES:
                    v = float((g.revup_class == c).mean())
                    ax.bar(x, v, bottom=bot, color=CL_COL[c], edgecolor="white")
                    bot += v
                ax.text(x, 1.01, str(len(g)), ha="center", fontsize=7)
                xt.append(x)
                xl.append(e[:3])
                x += 1
            x += 0.7
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=8)
        _grp_labels(ax)
        ax.set_ylim(0, 1.1)
        if j == 0:
            ax.set_ylabel("LATE trials (> pre p90):\nclass share")
    hd = [plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", label="top: cue -> first movement"),
          plt.Rectangle((0, 0), 1, 1, color="#e8a33d", label="top: first movement -> lick"),
          plt.Line2D([], [], color="k", marker="_", ls="", ms=12, label="top: median latency")] + \
         [plt.Rectangle((0, 0), 1, 1, color=CL_COL[c], label=f"bottom: {c}") for c in CLASSES]
    bot = _leg_below(fig, hd, ncol=4, fontsize=8)
    top = _sup(fig, "Fig R3. What fills the delay.\nTop: median cue -> first movement (jaw / cam1 opening / FR lick vigour "
                 "z>3) and first movement -> first lick.\nBottom: late trials classed wait / ramp (>= 300 ms "
                 "movement before the lick) / aborted (>= 2 cam1-only openings). n above bars. BEHAVIOUR", fontsize=8)
    fig.tight_layout(rect=(0, bot, 1, top), h_pad=2.0)
    save_png_svg(fig, FIG / "revup_decomposition.png", dpi=120)
    plt.close(fig)
    # Fig 4: FR vigour lick-aligned + cue-aligned, far_contra, per animal, pre vs post epochs, late trials of each
    Z = np.load(REV / "revup_fr.npz")
    Tall = T
    can = Tall.canonical.astype(str).str.lower().eq("true").to_numpy()
    fig, axs = plt.subplots(2, len(AN), figsize=(3.6 * len(AN), 7.4), sharey="row")
    for j, a in enumerate(AN):
        for e in EP4:
            m = can & (Tall.animal.to_numpy() == a) & (Tall.pos.to_numpy() == "far_contra") & \
                (Tall.epoch.to_numpy() == e) & np.isfinite(Tall.lat_ms.to_numpy(float))
            if m.sum() < 5:
                continue
            Xc = Z["cue"][m].astype(float)
            axs[0, j].plot(Z["lags_cue"], np.nanmean(Xc, 0), color=EC[e], lw=1.5, label=f"{e} (n={m.sum()})")
            ml = m & (Tall.lat_ms.to_numpy(float) > 600)
            if ml.sum() >= 5:
                axs[1, j].plot(Z["lags_lick"], np.nanmean(Z["lick"][ml].astype(float), 0), color=EC[e], lw=1.5,
                               label=f"{e} (n={ml.sum()})")
        axs[0, j].set_xlim(-0.5, 4)
        axs[0, j].set_title(f"{a} far_contra")
        axs[0, j].axvline(0, color="k", lw=0.6)
        axs[1, j].axvline(0, color="k", lw=0.6)
        axs[0, j].set_xlabel("time from cue (s)")
        axs[1, j].set_xlabel("time from first lick (s)")
        # per-panel n's kept, but in a legend BELOW the panel (they sat on the traces)
        for r in (0, 1):
            axs[r, j].legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                             handlelength=1.2, columnspacing=0.8)
    axs[0, 0].set_ylabel("FR lick vigour (z vs pre-cue)")
    axs[1, 0].set_ylabel("FR lick vigour, trials with\nlatency > 600 ms (z)")
    top = _sup(fig, "Fig R4. FaceRhythm non-directional lick vigour (sum of the six lick factors, unblanked) at far_contra:\n"
                 "top cue-aligned (licked trials), bottom aligned to the first lick of slow trials -- a ramp would rise "
                 "well before 0. BEHAVIOUR", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=1.0)
    save_png_svg(fig, FIG / "revup_fr_vigour.png", dpi=120)
    plt.close(fig)
    # Fig 5: recovery tracking
    W = pd.read_csv(REV / "revup_track_sessions.csv")
    from wfield_local import config
    acol = config.animal_color()
    meas = [("lat_med_ms", "far-contra median first-lick latency (ms)"),
            ("lat_med_ms_vsoth", "far-contra - other spouts latency (ms)"),
            ("sh_after_left", "far-contra share first lick after spout left"),
            ("marker", "primary balance marker")]
    fig, axs = plt.subplots(1, len(meas), figsize=(3.6 * len(meas), 3.4))
    G = W[(W.epoch != "pre") & (W.day > 0)]
    for ax, (c, nm) in zip(axs, meas):
        for a, g in G.groupby("animal"):
            ax.plot(g[c], g.aim, "o", ms=5, color=acol.get(a, "k"), label=a)
        ax.set_xlabel(nm, fontsize=8)
        ax.set_ylabel("aim (deg; 0 = pre landing)")
    bot = _leg_below(fig, [plt.Line2D([], [], ls="", marker="o", color=acol.get(a, "k"), label=a)
                           for a in sorted(G.animal.unique())], fontsize=9)
    top = _sup(fig, "Fig R5. Does the far-contra delay track recovery?\npost-stroke sessions; aim = far-contra first-cued-lick"
                 " aim vs pre. Within-animal tests in revup_track.csv. BEHAVIOUR", fontsize=8)
    fig.tight_layout(rect=(0, bot, 1, top))
    save_png_svg(fig, FIG / "revup_recovery.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    _rm_guard.require_default("revup_behaviour")   # unrouted share writer: refuses in a non-default run mode
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    if a.plot_only:
        plots()
        return 0
    REV.mkdir(parents=True, exist_ok=True)
    labs = a.labels or session_labels()
    got, fail = ak.fan_sessions(labs, session, jobs=min(a.jobs, 8), label="revup")
    for it, err in fail:
        print(f"  FAILED {it}: {err}", flush=True)
    ok = [x for x, _r in got]
    print(f"[revup] {len(ok)} sessions ok, {len(fail)} failed", flush=True)
    T, F = assemble(ok)
    summarise(T, F)
    plots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
