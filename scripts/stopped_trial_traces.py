"""REGIONAL TRACES ON STOPPED (and other no-lick) TRIALS across epochs (Priya 10-29). DESCRIPTIVE, PROVISIONAL, n = 4.

TRIAL CLASSES (one definition per quantity -- the `enl_states` / `scripts/enl_state_counts` construction, verbatim):
  success        DAQ lick within decode.max_rt_s (3.5 s) of the cue                    (`session_trials` + max_rt)
  miss_working   no DAQ lick in the response window (`response_window_for`), BEFORE the session's terminal
                 non-recovering engagement collapse (`precue_engagement_states.engagement_gate`)
  stopped        no DAQ lick in the response window, INSIDE that terminal collapse
  Jaw split of the no-lick trials (Priya 10-29 addition), from the EXISTING per-trial call
  `<v36>/imaging_encoding/qc/trial_attempts.csv` `cls` (scripts/trial_attempts.py; jaw = qc/jaw_onsets.csv, cam4 4 px /
  cam1 15 px floors, close_center masked -> unknown; a jaw deflected at the cue counts only via its re-onset):
  jaw_nolick     no DAQ lick and cls in {incomplete (LP lick, no contact), jaw_only (jaw moved after the cue)}
                 -- split into _working / _stopped (inside vs outside the engaged period)
  stopped_nojaw  stopped and cls == no_try (jaw judged and still)
  working_nojaw  miss_working and cls == no_try (reference)
  Priya's 10-21 blind review judged 0 / 24 jaw_only trials attempts (jaw responses without opening / other movement /
  walking), so jaw_nolick is "a jaw reaction or an LP-only lick", NOT an attempt.
CAMERA TRIAL GATE (``WIDEFIELD_TRIAL_GATE=camera``; Priya 11-07: show the trial CLASSES directly). Every class comes
from `wfield_local.trial_classes` (rule 9; DECISIONS 2026-11-03 / 11-03 late / 11-06): the camera class of each DAQ cue =
`cue_classes_for` over the session's FULL DAQ cue train (the stop rule counts runs along it). Five groups:
  success        class contact (a DAQ spout-contact lick in the response window)          } only INSIDE the camera
  incomplete     a camera lick (cam4 LP / cam1), no contact                               } working period
  jaw_only       no lick, the jaw moved after the cue (its own class)                     } (`in_working_period`:
  no_response    no lick, the jaw judged and still                                        }  first .. last camera lick)
  stopped        `STOPPED_CLASSES`: the camera stopped-run rule (A2 + excise by default: mid-session stop runs + the quit
                 tail after the last camera lick)
  excluded       early_run / after_end / stopped_licking / unknown (no video, unmatched): in no group, counts printed
Window = the trial_classes response window (cue + 70 ms .. min(cue + max_rt, trial stop)). Jaw-onset alignment for the
incomplete and jaw_only groups. A group with < MIN_N trials for an animal x epoch is left out of that epoch's mean and
the panel says so. The DAQ gate (below) is unchanged. Outputs are routed by the run's selection tag (`run_mode`,
il.OUT), so camera and DAQ outputs never share a folder; session files carry ``gate`` and a file of the other gate is
recomputed.
CONFOUND: the engagement gate marks ONE terminal run, so miss_working always precedes stopped in a session; any
stopped vs working (and stopped_nojaw vs jaw_nolick_working) difference is partly a LATE-SESSION difference. The median
within-session position (fraction of the session's cues) of every class is reported per animal x epoch.

TRACE (per trial x area, ROI v8 masks, ALM / tjM1 split from MOs / MOp with MOs_rest / MOp_rest): the construction of
`scripts/lick_precue_restw_traces.session_worker` (area means of U_atlas on the adopted SVT, slow drift basis removed,
x1e3, NO per-trial baseline), sampled at
  cue      -2 .. +3 s from the cue
  arrival  -1 .. +4 s from the spout-arrival strobe, samples at / after the trial's cue masked (pre-cue only)
  jaw      -1 .. +2 s from the jaw onset (jaw_nolick trials with a finite trial_attempts jaw_onset_ms)
Baselines: "restw" = minus the session's RESTW level (<OUT_W>/precue_restw/<label>.npz, same as the pre-cue maps);
"raw" = the dF/F x1e3 trace itself. Only restw_ok sessions are used in BOTH versions (same trials in both).
Aggregation: per animal x epoch, TRIAL-weighted mean over that animal's sessions (stopped trials are sparse and
concentrated in few sessions; an animal enters an epoch with >= MIN_N trials); then mean over animals, shade = SEM over
animals; solid = all 4 animals contribute, dotted = fewer. Pre = August-only (il.labels()).

Outputs under <il.OUT>/stopped_traces/: sessions/<label>.npz, counts_*.csv, levels_per_animal.csv, figdata/*.pkl,
figures/*.png.   Usage: python scripts/stopped_trial_traces.py [--jobs 2] [--plot-only] [--rest-variant restnl]
RESTnl VARIANT (`--rest-variant restnl`, default restw unchanged; no re-extraction -- the session files hold RAW traces):
the rest-referenced versions subtract the LICK-FREE rest level (scripts/restnl.py) instead of RESTW, and only sessions
with a RESTnl level are used (raw columns too, so raw and RESTnl share trials). Figure stems '..._restw' become
'..._restnl', other outputs get the suffix _restnl; the raw-only trace figures are not redrawn (baseline-free).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata as _fd  # noqa: E402
import iti_licks as il  # noqa: E402
import restnl  # noqa: E402
from figdata import plot_only  # noqa: E402
from wfield_local import lp_licks as _lp_licks  # noqa: E402  (coordinator 11-04: the run's attempts table)

from wfield_local import mask_keys  # noqa: E402

# resolved at IMPORT from argv (`restnl.variant`), so a spawned worker -- which re-imports this module with the
# parent's argv -- sees the same value; main() no longer sets it (Priya 11-03 / parallel rule: options reach the
# workers)
SFX = restnl.SFX if restnl.variant() == "restnl" else ""


def _stem(stem: str) -> str:
    if not SFX:
        return stem
    return stem.replace("_restw", SFX) if "_restw" in stem else stem + SFX


def _o(name: str) -> str:
    stem, dot, ext = name.partition(".")
    return f"{_stem(stem)}{dot}{ext}"


def save_fd(stem, obj, root=None):
    if isinstance(obj, dict):
        obj = {**obj, "sfx": SFX}
        if _cam():   # camera only, so a DAQ figdata pickle is unchanged
            obj["gate"] = "camera"
    return _fd.save_fd(_stem(stem), obj, root)


def load_fd(stem, root=None):
    return _fd.load_fd(_stem(stem), root)


def _rl(F, txt: str) -> str:
    return restnl.relabel(txt) if F.get("sfx") else txt


def _lev(label: str, z) -> tuple[np.ndarray, bool]:
    """(rest level per column of the session file, ok) for the active variant."""
    if not SFX:
        return z["restw"], bool(z["restw_ok"])
    lev, ok = restnl.level(label, [str(c) for c in z["cols"]])
    return lev, ok and bool(z["restw_ok"])

OUTD = il.OUT / "stopped_traces"
SESS = OUTD / "sessions"
FIGD = OUTD / "figdata"
FIGS = OUTD / "figures"
VERSION = 1
LAGS = np.round(np.arange(-2.0, 3.0001, 0.05), 3)
LAGS_A = np.round(np.arange(-1.0, 4.0001, 0.05), 3)
LAGS_J = np.round(np.arange(-1.0, 2.0001, 0.05), 3)
ALIGN = {"cue": LAGS, "arrival": LAGS_A, "jaw": LAGS_J}
ALIGN_TXT = {"cue": "time from cue (s)", "arrival": "time from spout arrival (strobe) (s); samples at/after the cue masked",
             "jaw": "time from jaw onset (s)"}
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "MOs_rest", "MOp_rest", "SSp-bfd", "VISp"]
REF = {"MOs_rest", "MOp_rest", "SSp-bfd", "VISp"}
HEMI = (("right", "contra (R)"), ("left", "ipsi (L)"))
POS6 = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
SETS = (*POS6, "all")
CLASSES = ("success", "working", "stopped", "jaw_nolick", "jaw_nolick_working", "jaw_nolick_stopped", "stopped_nojaw",
           "working_nojaw")
CLS_TXT = {"success": "hit (lick <= 3.5 s)", "working": "miss_working", "stopped": "stopped",
           "jaw_nolick": "jaw response, no lick", "jaw_nolick_working": "jaw, no lick (engaged)",
           "jaw_nolick_stopped": "jaw, no lick (stopped)", "stopped_nojaw": "stopped, no jaw",
           "working_nojaw": "miss_working, no jaw"}
CLS_COL = {"success": "#1b7837", "working": "#2166ac", "stopped": "#b2182b", "jaw_nolick": "#e08214",
           "jaw_nolick_working": "#e08214", "jaw_nolick_stopped": "#8073ac", "stopped_nojaw": "#b2182b",
           "working_nojaw": "#2166ac"}
JAW_CLASSES = ("jaw_nolick", "jaw_nolick_working", "jaw_nolick_stopped")
MIN_N = {"all": 5}
MIN_N_POS = 3
N_AN = 4
CLASS_DEF = ("classes (enl_states): hit = DAQ lick <= 3.5 s; miss_working = no lick in the response window, before the "
             "terminal engagement collapse (engagement_gate); stopped = no lick, inside it; jaw split from "
             "qc/trial_attempts.csv cls (jaw_nolick = incomplete | jaw_only; no jaw = no_try; close_center = unknown)")
#: CAMERA GATE groups (Priya 11-07): the four working-period classes + stopped, all from trial_classes
CAM_GROUPS = ("success", "incomplete", "jaw_only", "no_response", "stopped")
CAM_WORKING = {"success": "contact", "incomplete": "incomplete", "jaw_only": "jaw_only", "no_response": "no_response"}
CAM_EXCLUDED = "excluded"
CAM_JAW = ("incomplete", "jaw_only")
CAM_TXT = {"success": "contact (DAQ contact)", "incomplete": "incomplete (camera lick, no contact)",
           "jaw_only": "jaw only", "no_response": "no response", "stopped": "stopped (camera stop run)"}
CAM_COL = {"success": "#1b7837", "incomplete": "#2166ac", "jaw_only": "#e08214", "no_response": "#762a83",
           "stopped": "#b2182b"}


def _gate() -> str:
    """The run's trial gate ("daq" | "camera"), `trial_classes.gate_mode` (env-resolved, so spawned workers agree)."""
    from wfield_local import trial_classes as tc
    return tc.gate_mode()


def _cam() -> bool:
    return _gate() == "camera"


def classes() -> tuple:
    """The per-session trace classes of the run's gate."""
    return CAM_GROUPS if _cam() else CLASSES


def jaw_classes() -> tuple:
    return CAM_JAW if _cam() else JAW_CLASSES


def class_figs() -> tuple:
    return CAM_GROUPS if _cam() else CLASS_FIGS


def compare_groups() -> dict:
    return {"camclasses": CAM_GROUPS} if _cam() else COMPARE


def jaw_compare() -> tuple:
    return CAM_JAW if _cam() else ("jaw_nolick_working", "jaw_nolick_stopped")


def lev_classes() -> tuple:
    return CAM_GROUPS if _cam() else LEV_CLASSES


def table_classes() -> list:
    """Columns of the rule-8 level tables."""
    return list(CAM_GROUPS) if _cam() else ["success", "working", "stopped", "jaw_nolick", "stopped_nojaw"]


def main_classes() -> tuple:
    """The classes of the pre-cue level / movement summaries (DAQ: hit / miss_working / stopped)."""
    return CAM_GROUPS if _cam() else ("success", "working", "stopped")


def _txt(k: str) -> str:
    return CAM_TXT[k] if _cam() and k in CAM_TXT else CLS_TXT[k]


def _col(k: str) -> str:
    return CAM_COL[k] if _cam() and k in CAM_COL else CLS_COL[k]


def class_def() -> str:
    """CLASS_DEF for the run's gate (captions / data block; figure-provenance rule)."""
    if not _cam():
        return CLASS_DEF
    from wfield_local import trial_classes as tc
    return (f"Trials: camera trial gate ({tc.run_selection()}), classes from trial_classes. Window = response window "
            f"cue + {tc.POSTCUE_START_MS:g} ms .. min(cue + {tc.max_rt_ms() / 1000:g} s, trial stop) (camera licks by "
            "peak, contacts by DAQ onset, jaw by onset). success = class contact (DAQ spout-contact lick); incomplete = "
            "camera lick (cam4 LP / cam1), no contact; jaw_only = no lick, jaw moved after the cue (cam4 or cam1); "
            "no_response = no lick, jaw judged still (close_center: default no_response); these four only inside the "
            "camera working period (first .. last camera lick, stop runs out). stopped = camera stopped-run rule "
            f"{tc.stopped_rule_name()} (>= {tc.stop_run_min()} consecutive no_response / jaw_only before the last camera "
            "lick, reference-spout guard, run starting >= 300 s after the first cue; plus the quit tail after the last "
            "camera lick; lick-touched trials excised). Excluded: early_run, after_end, stopped_licking, unknown / no "
            f"video. A group with < {MIN_N['all']} trials (all spouts; {MIN_N_POS} per spout) for an animal x epoch is "
            "left out of that epoch")


def confound_text() -> str:
    if not _cam():
        return "miss_working always precedes stopped within a session."
    return "the camera quit tail follows every working-period trial of its session; mid-session stop runs do not."


def _stopped_txt() -> str:
    if not _cam():
        return "stopped = no lick in the response window, inside the terminal engagement collapse (enl_states)."
    return "stopped = camera stopped-run rule (trial_classes; mid-session stop runs + quit tail)."


def _nolick(state) -> np.ndarray:
    """No-lick states of the jaw split: DAQ = not success (working / stopped); camera = the no-contact groups."""
    st = np.asarray(state, dtype=object)
    return np.isin(st, CAM_GROUPS[1:]) if _cam() else st != "success"


def camera_groups(cam_cls) -> np.ndarray:
    """Group per trial from its trial_classes camera class (Priya 11-07): the four classes inside the camera working
    period (`in_working_period`), stopped (`STOPPED_CLASSES`), else "excluded". Nothing is redefined here."""
    from wfield_local import trial_classes as tc
    c = np.asarray(cam_cls, dtype=object)
    inwp = tc.in_working_period(c)
    st = np.full(c.size, CAM_EXCLUDED, dtype=object)
    for g, k in CAM_WORKING.items():
        st[inwp & (c == k)] = g
    st[np.isin(c, list(tc.STOPPED_CLASSES))] = "stopped"
    return st.astype(str)


# ------------------------------------------------------------------------------------------------ per session
def _states(label: str) -> pd.DataFrame:
    """enl_state_counts._worker's per-trial state, plus the DAQ cue time of each trial (s)."""
    from scripts.rest_migration.engagement_decomposition import session_trials
    from wfield_local import config, daq_io
    from wfield_local.locanmf_cue_lick_analysis import _load_cue_events
    from wfield_local.nolick_decoder import response_window_for
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    from wfield_local.precue_engagement_states import engagement_gate
    s = next(x for x in config.load_sessions() if x["label"] == label)
    tr = session_trials(s, response_window_for(s))
    cs = np.asarray(_load_cue_events(s["h5"])["cue_samples"], float)
    with daq_io.open_daq(s["h5"]) as f:
        sr, _ = daq_io.session_attrs(f)
    order = np.array([t["order"] for t in tr])
    hit = np.array([bool(t["hit"]) for t in tr])
    lat = np.array([t["latency_s"] if t["latency_s"] is not None else np.nan for t in tr], float)
    pos = np.array([POSITION_NAMES.get(int(t["pos"]), str(t["pos"])) for t in tr])
    max_rt = float(config.defaults()["decode"].get("max_rt_s", 2.0))
    success = hit & np.isfinite(lat) & (lat <= max_rt)
    if _cam():
        # CAMERA GATE (Priya 11-07): trial_classes classes over the session's FULL DAQ cue train (the stop rule counts
        # runs along it) -> the five groups. No engagement_gate call (it refuses under camera).
        from wfield_local import trial_classes as tc
        cam = np.asarray(tc.cue_classes_for(label, cs / sr), dtype=object)[order]
        return pd.DataFrame({"order": order, "frac": order / max(cs.size - 1, 1), "cue_daq": cs[order] / sr,
                             "state": camera_groups(cam), "camcls": cam.astype(str)})
    not_eng = engagement_gate(order, hit, pos)
    state = np.where(success, "success", np.where(not_eng, "stopped", "working"))
    return pd.DataFrame({"order": order, "frac": order / max(cs.size - 1, 1), "cue_daq": cs[order] / sr,
                         "state": state})


def session_worker(item) -> dict:
    """One session (module-level for spawn) -> SESS/<label>.npz."""
    label, T, A = item
    f = SESS / f"{label}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if int(z["version"]) == VERSION and (str(z["gate"]) if "gate" in z.files else "daq") == _gate():
            mask_keys.check_obj(z, f)             # stale-read guard (11-09)
            return {"label": label, "cached": True}
    from wfield_local import combined_encoding as ce
    S = _states(label)
    # match the DAQ-side state rows to the imaging trial table (enl_trials: trial_id, pos, strobe_s, cue_s) by cue time
    T = T.sort_values("cue_s").reset_index(drop=True)
    tc = T.cue_s.to_numpy(float)
    j = np.clip(np.searchsorted(tc, S.cue_daq.to_numpy()), 1, len(tc) - 1)
    jj = np.where(np.abs(tc[j] - S.cue_daq) < np.abs(tc[j - 1] - S.cue_daq), j, j - 1)
    ok = np.abs(tc[jj] - S.cue_daq.to_numpy()) < 0.02
    n_unmatched = int((~ok).sum())
    S = S[ok].reset_index(drop=True)
    jj = jj[ok]
    if len(np.unique(jj)) != len(jj):
        raise ValueError(f"{label}: duplicate cue matches")
    S["trial_id"] = T.trial_id.to_numpy(int)[jj]
    S["pos"] = T.pos.astype(str).to_numpy()[jj]
    S["strobe_s"] = T.strobe_s.to_numpy(float)[jj]
    S["cue_s"] = tc[jj]
    S["canonical"] = T.engaged.astype(str).str.lower().eq("true").to_numpy()[jj]
    Am = A.set_index("trial_id")
    S["cls"] = S.trial_id.map(Am.cls).fillna("missing").astype(str)
    S["jaw_onset_ms"] = pd.to_numeric(S.trial_id.map(Am.jaw_onset_ms), errors="coerce")
    pos_agree = float((S.trial_id.map(Am.pos).astype(str) == S.pos).mean()) if len(Am) else np.nan
    nolick = pd.Series(_nolick(S.state), index=S.index)
    jaw = S.cls.isin(["incomplete", "jaw_only"])
    M = {"success": S.state == "success", "working": S.state == "working", "stopped": S.state == "stopped",
         "jaw_nolick": nolick & jaw, "jaw_nolick_working": (S.state == "working") & jaw,
         "jaw_nolick_stopped": (S.state == "stopped") & jaw, "stopped_nojaw": (S.state == "stopped") & (S.cls == "no_try"),
         "working_nojaw": (S.state == "working") & (S.cls == "no_try")}
    if _cam():
        M = {g: S.state == g for g in CAM_GROUPS}
    # imaging trace (== lick_precue_restw_traces.session_worker)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    Mk = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(Mk)
    Ra = np.stack([Ub[Mk[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    Rs, cs = il.with_rest(R, cols, label, basis="split")
    pc = [f"{a}_{h}" for a in AREAS for h, _ in HEMI if f"{a}_{h}" in cs]
    R = Rs[:, [cs.index(c) for c in pc]]
    src = il.OUT_W / "precue_restw" / f"{label}.npz"
    if src.exists():
        z = np.load(src)
        mask_keys.check_obj(z, src)               # stale-read guard (11-09)
        sc = [str(c) for c in z["cols"]]
        restw = np.array([z["restw"][sc.index(c)] if c in sc else np.nan for c in pc], float)
        restw_ok, animal, epoch = bool(z["restw_ok"]), str(z["animal"]), str(z["epoch"])
    else:
        from wfield_local import epochs
        restw, restw_ok, animal, epoch = np.full(len(pc), np.nan), False, label[:4], str(epochs.epoch_of(label))

    def samp(t, lags):
        X = il._sample(R, ft, t, lags).astype(np.float64)
        tt = t[:, None] + lags[None, :]
        X[(tt < ft[0]) | (tt > ft[-1]) | ~np.isfinite(tt)] = np.nan
        return X

    cue = S.cue_s.to_numpy(float)
    st = S.strobe_s.to_numpy(float)
    Xc = samp(cue, LAGS)
    Xa = samp(np.where(np.isfinite(st), st, cue - 10.0), LAGS_A)
    Xa[~np.isfinite(st)] = np.nan
    Xa[LAGS_A[None, :] >= (cue - st)[:, None]] = np.nan
    jt = cue + S.jaw_onset_ms.to_numpy(float) / 1000.0
    Xj = samp(np.where(np.isfinite(jt), jt, cue), LAGS_J)
    Xj[~np.isfinite(jt)] = np.nan
    out = {}
    posv = S.pos.to_numpy()
    for c, m in M.items():
        m = m.to_numpy(bool)
        for al, X in (("cue", Xc), ("arrival", Xa), ("jaw", Xj)):
            if al == "jaw" and c not in jaw_classes():
                continue
            mean = np.full((len(SETS), X.shape[1], X.shape[2]), np.nan)
            nl = np.zeros((len(SETS), X.shape[1]), int)
            for k, p in enumerate(SETS):
                sel = m & (posv == p) if p != "all" else m
                if sel.any():
                    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
                        __import__("warnings").simplefilter("ignore")
                        mean[k] = np.nanmean(X[sel], 0)
                    nl[k] = np.isfinite(X[sel][:, :, 0]).sum(0)
            out[f"{c}__{al}"] = mean.astype(np.float32)
            out[f"{c}__{al}__n"] = nl
    stp = M["stopped"].to_numpy(bool)
    SESS.mkdir(parents=True, exist_ok=True)
    if _cam():   # the gate + camera class per trial (a DAQ-gate file is written exactly as before)
        out = {**out, "gate": "camera", "t_camcls": S.camcls.to_numpy().astype(str)}
    np.savez_compressed(
        f, version=VERSION, cols=np.array(pc), restw=restw, restw_ok=restw_ok, animal=animal, epoch=epoch,
        n_unmatched=n_unmatched, pos_agree=pos_agree,
        t_trial_id=S.trial_id.to_numpy(int), t_pos=posv.astype(str), t_state=S.state.to_numpy().astype(str),
        t_cls=S.cls.to_numpy().astype(str), t_frac=S.frac.to_numpy(float), t_canonical=S.canonical.to_numpy(bool),
        t_jaw_ms=S.jaw_onset_ms.to_numpy(float), t_cue_minus_strobe=cue - st,
        stopped_X=Xc[stp].astype(np.float16), stopped_tid=S.trial_id.to_numpy(int)[stp], **out,
        **mask_keys.stamp())
    return {"label": label, "cached": False, "n": len(S), "n_unmatched": n_unmatched}


# ------------------------------------------------------------------------------------------------ tables
def trial_table(labels: list) -> pd.DataFrame:
    rows = []
    for x in labels:
        z = np.load(SESS / f"{x}.npz")
        mask_keys.check_obj(z, SESS / f"{x}.npz")  # stale-read guard (11-09)
        S = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("t_")})
        S["label"], S["animal"], S["epoch"], S["restw_ok"] = x, str(z["animal"]), str(z["epoch"]), _lev(x, z)[1]
        rows.append(S)
    D = pd.concat(rows, ignore_index=True)
    jaw = D.cls.isin(["incomplete", "jaw_only"])
    D["jaw_group"] = np.select([D.state == "success", jaw, D.cls == "no_try"], ["hit", "jaw", "no_jaw"], "jaw_unknown")
    return D


def class_masks(D: pd.DataFrame) -> dict:
    if _cam():
        return {g: D.state == g for g in CAM_GROUPS}
    jaw = D.cls.isin(["incomplete", "jaw_only"])
    return {"success": D.state == "success", "working": D.state == "working", "stopped": D.state == "stopped",
            "jaw_nolick": pd.Series(_nolick(D.state), index=D.index) & jaw, "jaw_nolick_working": (D.state == "working") & jaw,
            "jaw_nolick_stopped": (D.state == "stopped") & jaw, "stopped_nojaw": (D.state == "stopped") & (D.cls == "no_try"),
            "working_nojaw": (D.state == "working") & (D.cls == "no_try")}


def counts(D: pd.DataFrame) -> dict:
    D = D[D.restw_ok]
    M = class_masks(D)
    out = {}
    for tag, sel in (("all", np.ones(len(D), bool)), ("far_contra", (D.pos == "far_contra").to_numpy())):
        C = pd.DataFrame({c: (m.to_numpy() & sel) for c, m in M.items()})
        C["animal"], C["epoch"] = D.animal.to_numpy(), D.epoch.to_numpy()
        P = C.groupby(["animal", "epoch"])[list(M)].sum().astype(int)
        P = P.reindex(pd.MultiIndex.from_product([sorted(D.animal.unique()), il.EP4]), fill_value=0)
        P.index.names = ["animal", "epoch"]
        out[tag] = P
    # stopped sessions per animal x epoch
    ss = D[D.state == "stopped"].groupby(["animal", "epoch"]).label.nunique()
    out["sessions_with_stopped"] = ss
    # session-time confound: median within-session position of each class
    fr = {c: D[m.to_numpy()].groupby(["animal", "epoch"]).frac.median() for c, m in M.items()}
    out["frac"] = pd.DataFrame(fr).reindex(out["all"].index).round(2)
    # jaw-no-lick sub-types and canonical (reference-gate) split
    nl = D[_nolick(D.state)]
    out["nolick_cls"] = pd.crosstab([nl.animal, nl.epoch, nl.state], nl.cls)
    jl = nl[nl.cls.isin(["incomplete", "jaw_only"])]
    out["jaw_canonical"] = pd.crosstab([jl.animal, jl.epoch], [jl.state, jl.canonical.map({True: "canon", False: "noncanon"})])
    out["jaw_pos"] = pd.crosstab([jl.animal, jl.epoch], jl.pos)
    return out


def state_counts(D: pd.DataFrame) -> pd.DataFrame:
    """Rule 8: per animal x epoch trial counts of every group (all sessions, restw_ok or not); under the camera gate the
    five groups (all of them, zeros included) and the excluded trials by their trial_classes class."""
    T = pd.crosstab([D.animal, D.epoch], D.state)
    if "camcls" in D:
        T = T.reindex(columns=list(CAM_GROUPS), fill_value=0)
        o = D[D.state == CAM_EXCLUDED]
        if len(o):
            T = T.join(pd.crosstab([o.animal, o.epoch], o.camcls).add_prefix("excluded:"), how="left")
    return T.fillna(0).astype(int)


# ------------------------------------------------------------------------------------------------ aggregation
def pool(labels: list, klass: str, align: str, k: int, base: str, min_n: int) -> dict:
    """{epoch: {animal: (curve (lags, cols), n trials)}} trial-weighted over sessions."""
    acc = {}
    cols = None
    for x in labels:
        z = np.load(SESS / f"{x}.npz")
        mask_keys.check_obj(z, SESS / f"{x}.npz")  # stale-read guard (11-09)
        rw, ok = _lev(x, z)
        if not ok:
            continue
        key = f"{klass}__{align}"
        if key not in z.files:
            continue
        cols = [str(c) for c in z["cols"]] if cols is None else cols
        if [str(c) for c in z["cols"]] != cols:
            raise ValueError(f"{x}: column order differs")
        Y = z[key][k].astype(np.float64)
        n = z[key + "__n"][k].astype(float)
        if n.max() == 0:
            continue
        if base == "restw":
            Y = Y - rw[None, :]
        W = n[:, None] * np.where(np.isfinite(Y), 1.0, 0.0)
        a = acc.setdefault((str(z["epoch"]), str(z["animal"])), [np.zeros_like(Y), np.zeros_like(Y), 0])
        a[0] += np.nan_to_num(Y) * W
        a[1] += W
        a[2] += int(n.max())
    res = {}
    for (ep, an), (s, w, nt) in acc.items():
        if nt < min_n:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            res.setdefault(ep, {})[an] = (np.where(w > 0, s / w, np.nan), nt)
    return {"per": res, "cols": cols}


def summarise(P: dict) -> dict:
    """{epoch: (mean, sem, n_animals) (lags, cols)}"""
    out = {}
    for ep in il.EP4:
        if ep not in P["per"]:
            continue
        Y = np.stack([v[0] for v in P["per"][ep].values()])
        n = np.isfinite(Y).sum(0)
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            mu = np.nanmean(Y, 0)
            sd = np.nanstd(Y, 0, ddof=1) if Y.shape[0] > 1 else np.full(mu.shape, np.nan)
        out[ep] = (mu, sd / np.sqrt(np.maximum(n, 1)), n, {a: v[1] for a, v in P["per"][ep].items()})
    return out


# ------------------------------------------------------------------------------------------------ figures
def _style(ax):
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def _mark_missing(ax, F, ep, ks) -> None:
    """Camera gate: name the groups with no animal at >= min_n trials in this epoch (not drawn: no noisy mean)."""
    if F.get("gate") != "camera":
        return
    miss = [k for k in ks if ep not in (F["S"][k] if "classes" in F else F["S"])]
    if miss:
        ax.text(0.02, 0.02, f"< {F['min_n']} tr/animal: " + ", ".join(miss), transform=ax.transAxes, fontsize=5.5,
                color="0.35", va="bottom")


def _curve(ax, lags, mu, se, n, c, lab):
    ax.fill_between(lags, mu - se, mu + se, color=c, alpha=0.15, lw=0)
    ax.plot(lags, mu, color=c, lw=1.0, ls=":")
    ax.plot(lags, np.where(n >= N_AN, mu, np.nan), color=c, lw=1.4, label=lab)


def draw_class(stem: str) -> None:
    """Rows = areas, columns = hemispheres, color = epoch, one trial class."""
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    F = load_fd(stem, FIGD)
    lags, cols = ALIGN[F["align"]], F["cols"]
    fig, axs = plt.subplots(len(AREAS), 2, figsize=(10, 2.1 * len(AREAS)), sharex=True, squeeze=False,
                            layout="constrained")
    for i, a in enumerate(AREAS):
        for j, (h, hl) in enumerate(HEMI):
            ax = axs[i, j]
            _style(ax)
            ax.axvline(0, color="0.3", lw=0.8)
            ax.axhline(0, color="0.6", lw=0.5)
            c = f"{a}_{h}"
            ax.set_title(f"{a} {hl}" + ("  [reference]" if a in REF else ""), fontsize=8.5)
            if c not in cols:
                ax.text(0.5, 0.5, "no ROI in this hemisphere", transform=ax.transAxes, ha="center", fontsize=8)
                continue
            q = cols.index(c)
            for ep, (mu, se, n, na) in F["S"].items():
                _curve(ax, lags, mu[:, q], se[:, q], n[:, q], EPOCH_LINE[ep], f"{ep} ({len(na)} an, {sum(na.values())} tr)")
            if F.get("gate") == "camera" and i == 0 and j == 0:
                miss = [ep for ep in il.EP4 if ep not in F["S"]]
                if miss:
                    ax.text(0.02, 0.02, f"< {F['min_n']} tr/animal (not drawn): " + ", ".join(miss),
                            transform=ax.transAxes, fontsize=6, color="0.35", va="bottom")
            if j == 0:
                ax.set_ylabel(_rl(F, "dF/F x1e3 above restw") if F["base"] == "restw" else "dF/F x1e3 (raw)",
                              fontsize=7.5)
            ax.set_xlim(lags[0], lags[-1])
    for ax in axs[-1]:
        ax.set_xlabel(ALIGN_TXT[F["align"]], fontsize=8)
    h, lab = axs[0, 0].get_legend_handles_labels()   # one figure legend under the panels (was over ALM contra data)
    fig.legend(h, lab, fontsize=7.5, frameon=False, loc="outside lower center", ncol=len(h))
    fig.suptitle(textwrap.fill(_title(F, f"class = {_txt(F['klass']).upper()}, color = epoch"), 150), fontsize=8.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, FIGS / f"{_stem(stem)}.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


def _title(F, what):
    base = ("baseline = minus the session's RESTW level (as the pre-cue maps)" if F["base"] == "restw"
            else "RAW dF/F x1e3 (drift-removed, no per-trial or rest baseline)")
    spout = "all spouts pooled" if F["set"] == "all" else f"{F['set']} spout only"
    return _rl(F, f"NEURAL: {F['align']}-aligned area traces, {what}; {spout}; {base}. {class_def()}. Per animal x epoch "
            f"trial-weighted mean over sessions (>= {F['min_n']} trials), mean over animals, shade = SEM over animals; "
            f"solid = all 4 animals, dotted = fewer. Pre = August only. PROVISIONAL n = 4. Late-session confound: "
            f"{confound_text()}")


def draw_compare(stem: str) -> None:
    """Rows = areas, columns = hemisphere x epoch, color = class."""
    import matplotlib.pyplot as plt
    F = load_fd(stem, FIGD)
    lags, cols = ALIGN[F["align"]], F["cols"]
    ncol = 2 * len(il.EP4)
    fig, axs = plt.subplots(len(AREAS), ncol, figsize=(2.3 * ncol, 1.75 * len(AREAS)), sharex=True, squeeze=False,
                            layout="constrained")
    for i, a in enumerate(AREAS):
        for j, (h, hl) in enumerate(HEMI):
            c = f"{a}_{h}"
            for e, ep in enumerate(il.EP4):
                ax = axs[i, j * len(il.EP4) + e]
                _style(ax)
                ax.axvline(0, color="0.3", lw=0.7)
                ax.axhline(0, color="0.6", lw=0.5)
                if i == 0:
                    ax.set_title(f"{hl}\n{ep}", fontsize=8.5)
                if e == 0 and j == 0:
                    ax.set_ylabel(f"{a}\n" + (_rl(F, "above restw") if F["base"] == "restw" else "raw dF/F x1e3"),
                                  fontsize=7.5)
                elif e == 0:
                    ax.set_ylabel(a, fontsize=7.5)
                if c not in cols:
                    ax.text(0.5, 0.5, "no ROI", transform=ax.transAxes, ha="center", fontsize=7)
                    continue
                q = cols.index(c)
                if i == 0:
                    _mark_missing(ax, F, ep, F["classes"])
                for k in F["classes"]:
                    if ep not in F["S"][k]:
                        continue
                    mu, se, n, na = F["S"][k][ep]
                    lab = f"{_txt(k)} ({len(na)} an, {sum(na.values())} tr)" if (i == 0 and j == 0) else None
                    if lab:   # trial counts differ per epoch: legend label carries the epoch
                        lab = f"{ep}: {lab}"
                    _curve(ax, lags, mu[:, q], se[:, q], n[:, q], _col(k), lab)
                ax.set_xlim(lags[0], lags[-1])
        # share y within a row
        ys = [ax.get_ylim() for ax in axs[i]]
        lo, hi = min(y[0] for y in ys), max(y[1] for y in ys)
        for ax in axs[i]:
            ax.set_ylim(lo, hi)
    for ax in axs[-1]:
        ax.set_xlabel(ALIGN_TXT[F["align"]].split(";")[0], fontsize=7)
    # one figure legend under the panels (the per-epoch legends sat over the traces of the first four panels);
    # legend columns = epochs (pre..chronic), rows = classes, each with its animal / trial counts
    h, lab = [], []
    for e in range(len(il.EP4)):
        hh, ll = axs[0, e].get_legend_handles_labels()
        h += hh
        lab += ll
    fig.legend(h, lab, fontsize=7, frameon=False, loc="outside lower center", ncol=len(il.EP4))
    fig.suptitle(textwrap.fill(_title(F, "color = trial class: " + ", ".join(_txt(k) for k in F["classes"])), 230),
                 fontsize=9)
    FIGS.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, FIGS / f"{_stem(stem)}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_examples(stem: str) -> None:
    import matplotlib.pyplot as plt
    F = load_fd(stem, FIGD)
    ex = F["ex"]
    ncol = max([len(v) for v in ex.values()] + [1])
    fig, axs = plt.subplots(len(il.EP4), ncol, figsize=(3.4 * ncol, 3.0 * len(il.EP4)), squeeze=False,
                            layout="constrained")   # titles clear the tick labels of the row above
    vals = np.concatenate([e["X"].ravel() for v in ex.values() for e in v] or [np.zeros(1)])
    vm = float(np.nanpercentile(np.abs(vals), 98)) if np.isfinite(vals).any() else 1.0
    im = None
    for r, ep in enumerate(il.EP4):
        for k in range(ncol):
            ax = axs[r, k]
            if k >= len(ex.get(ep, [])):
                ax.axis("off")
                if k == 0:
                    ax.text(0.5, 0.5, f"{ep}: no stopped trials", transform=ax.transAxes, ha="center", fontsize=8)
                continue
            e = ex[ep][k]
            im = ax.imshow(e["X"].T, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest",
                           extent=(LAGS[0] - 0.025, LAGS[-1] + 0.025, len(F["cols"]) - 0.5, -0.5))
            ax.axvline(0, color="k", lw=0.7)
            ax.set_yticks(range(len(F["cols"])))
            ax.set_yticklabels(F["cols"] if k == 0 else [], fontsize=5.5)
            ax.set_title(f"{ep}: {e['label']} tr {e['tid']}\n{e['pos']}, session pos {e['frac']:.2f}", fontsize=7)
            ax.tick_params(labelsize=6)
            if r == len(il.EP4) - 1:
                ax.set_xlabel("time from cue (s)", fontsize=7)
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=0.5, label=_rl(F, "dF/F x1e3 above restw"))
    fig.suptitle(textwrap.fill(_rl(F, f"NEURAL: single STOPPED trials, {F['animal']} -- areas x time, cue-aligned, baseline = minus the "
                 f"session's RESTW level; trials evenly spaced through that animal x epoch's stopped trials "
                 f"(n stopped per epoch: {F['n']}). One color scale (98th pct |value|). {_stopped_txt()}"), 170), fontsize=8.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, FIGS / f"{_stem(stem)}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ compute
CLASS_FIGS = ("stopped", "working", "success", "jaw_nolick", "stopped_nojaw")
COMPARE = {"stopped_vs_working_hit": ("success", "working", "stopped"),
           "jaw_vs_nojaw": ("success", "jaw_nolick_working", "jaw_nolick_stopped", "stopped_nojaw")}


def stems() -> list:
    if SFX:   # RESTnl: only the rest-referenced figures (raw ones do not depend on the rest level)
        return [(k, s) for k, s in _stems() if not s.endswith("_raw") and "_raw_" not in s]
    return _stems()


def _stems() -> list:
    out = []
    for c in class_figs():
        for al in ("cue", "arrival") + (("jaw",) if c in jaw_classes() else ()):
            for b in ("restw", "raw"):
                out.append(("class", f"stopped_class_{c}_{al}_{b}"))
    for g in compare_groups():
        for al, st in (("cue", "all"), ("arrival", "all"), ("cue", "far_contra")):
            for b in ("restw", "raw"):
                out.append(("compare", f"stopped_compare_{g}_{al}_{b}" + ("_farcontra" if st != "all" else "")))
    out.append(("compare", "stopped_compare_jaw_aligned_jaw_restw"))
    out.append(("compare", "stopped_compare_jaw_aligned_jaw_raw"))
    for an in ("PS92", "PS93"):
        out.append(("examples", f"stopped_examples_{an}"))
    return out


def compute(labels: list) -> None:
    D = trial_table(labels)
    D.to_csv(OUTD / _o("trial_classes.csv.gz"), index=False)
    C = counts(D)
    pd.set_option("display.width", 200)
    print(f"== trial gate: {_gate()}" + (f" ({_tc_sel()})" if _cam() else " (DAQ quit-tail rule)"))
    print("== rule 8: per animal x epoch TRIAL COUNTS per group, all sessions"
          + (" (camera: 5 groups + excluded by trial_classes class)" if _cam() else "") + "\n"
          + state_counts(D).to_string())
    print("\n== rule 8: per animal x epoch TRIAL COUNTS, all spouts (restw_ok sessions)\n" + C["all"].to_string())
    print("\n== far_contra only\n" + C["far_contra"].to_string())
    print("\n== sessions containing >= 1 stopped trial\n" + C["sessions_with_stopped"].to_string())
    print("\n== median within-session position (fraction of the session's cues) per class: late-session confound\n"
          + C["frac"].to_string())
    print("\n== no-lick trials: trial_attempts cls by state\n" + C["nolick_cls"].to_string())
    print("\n== jaw_nolick: state (working / stopped) x reference gate (canonical)\n" + C["jaw_canonical"].to_string())
    print("\n== jaw_nolick by spout\n" + C["jaw_pos"].to_string())
    for k in ("all", "far_contra"):
        C[k].to_csv(OUTD / _o(f"counts_{k}.csv"))
    C["frac"].to_csv(OUTD / _o("counts_session_position_median.csv"))
    C["nolick_cls"].to_csv(OUTD / _o("counts_nolick_cls_by_state.csv"))
    C["jaw_canonical"].to_csv(OUTD / _o("counts_jaw_nolick_canonical.csv"))
    C["jaw_pos"].to_csv(OUTD / _o("counts_jaw_nolick_by_pos.csv"))
    match = pd.DataFrame([{"label": x, "n_unmatched": int(np.load(SESS / f"{x}.npz")["n_unmatched"]),
                           "pos_agree": float(np.load(SESS / f"{x}.npz")["pos_agree"])} for x in labels])
    print(f"\n== DAQ-state rows unmatched to the imaging trial table: {match.n_unmatched.sum()} total; trial_attempts "
          f"position agreement min {match.pos_agree.min():.3f}")
    cache = {}

    def S_of(c, al, st, b):
        key = (c, al, st, b)
        if key not in cache:
            mn = MIN_N["all"] if st == "all" else MIN_N_POS
            P = pool(labels, c, al, SETS.index(st), b, mn)
            cache[key] = (summarise(P), P["cols"], mn, P)
        return cache[key]

    # rule 8 level table: 0..1 s post cue (restw), key contra areas, per animal
    rec = []
    for c in table_classes():
        S, cols, _mn, P = S_of(c, "cue", "all", "restw")
        w = (LAGS >= -1e-9) & (LAGS <= 1.0 + 1e-9)
        wp = (LAGS >= -1.0 - 1e-9) & (LAGS <= 1e-9)
        for ep, d in P["per"].items():
            for an, (Y, nt) in d.items():
                for a in ("ALM_right", "tjM1_right", "SSp-m_right", "ALM_left"):
                    q = cols.index(a)
                    rec.append({"class": c, "epoch": ep, "animal": an, "area": a, "n": nt,
                                "pre_m1_0": np.nanmean(Y[wp, q]), "post_0_1": np.nanmean(Y[w, q])})
    L = pd.DataFrame(rec)
    L.to_csv(OUTD / _o("levels_per_animal.csv"), index=False)
    for a in ("ALM_right", "tjM1_right"):
        T = L[L.area == a].pivot_table(index=["animal", "epoch"], columns="class", values="post_0_1").reindex(
            columns=table_classes())
        print(f"\n== per animal (rule 8): {a} mean 0..1 s after cue, above restw (dF/F x1e3)\n" + T.round(2).to_string())
        T = L[L.area == a].pivot_table(index=["animal", "epoch"], columns="class", values="pre_m1_0").reindex(
            columns=table_classes())
        print(f"\n== per animal (rule 8): {a} mean -1..0 s before cue, above restw\n" + T.round(2).to_string())

    for kind, stem in stems():
        if kind == "class":
            parts = stem[len("stopped_class_"):].rsplit("_", 2)
            c, al, b = parts
            S, cols, mn, _P = S_of(c, al, "all", b)
            save_fd(stem, {"S": S, "cols": cols, "klass": c, "align": al, "base": b, "set": "all", "min_n": mn}, FIGD)
        elif kind == "compare":
            b = "raw" if "_raw" in stem else "restw"
            st = "far_contra" if stem.endswith("_farcontra") else "all"
            if "jaw_aligned" in stem:
                g, al, classes = "jaw", "jaw", jaw_compare()
            else:
                g = next(x for x in compare_groups() if f"_{x}_" in stem)
                al = "arrival" if "_arrival_" in stem else "cue"
                classes = compare_groups()[g]
            SS, cols, mn = {}, None, None
            for k in classes:
                S, cols, mn, _P = S_of(k, al, st, b)
                SS[k] = S
            save_fd(stem, {"S": SS, "cols": cols, "classes": classes, "align": al, "base": b, "set": st, "min_n": mn},
                    FIGD)
        else:
            an = stem.rsplit("_", 1)[1]
            ex, nn = {}, {}
            for ep in il.EP4:
                labs = [x for x in labels if x.startswith(an)]
                pool_ = []
                for x in labs:
                    z = np.load(SESS / f"{x}.npz")
                    mask_keys.check_obj(z, SESS / f"{x}.npz")  # stale-read guard (11-09)
                    rw, ok = _lev(x, z)
                    if str(z["epoch"]) != ep or not ok or len(z["stopped_tid"]) == 0:
                        continue
                    tid = z["t_trial_id"]
                    for i, t in enumerate(z["stopped_tid"]):
                        r = int(np.where(tid == t)[0][0])
                        pool_.append({"label": x, "tid": int(t), "pos": str(z["t_pos"][r]), "frac": float(z["t_frac"][r]),
                                      "X": z["stopped_X"][i].astype(np.float64) - rw[None, :]})
                nn[ep] = len(pool_)
                if pool_:
                    idx = np.unique(np.linspace(0, len(pool_) - 1, min(4, len(pool_))).round().astype(int))
                    ex[ep] = [pool_[i] for i in idx]
            cols = [str(c) for c in np.load(SESS / f"{labels[0]}.npz")["cols"]]
            save_fd(stem, {"ex": ex, "n": nn, "animal": an, "cols": cols}, FIGD)
        draw(kind, stem)


def draw(kind: str, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    {"class": draw_class, "compare": draw_compare, "examples": draw_examples}[kind](stem)


# ------------------------------------------------------------------------------------------------ pre-cue levels (10-29 late)
LEV_CLASSES = ("success", "working", "stopped", "stopped_nojaw", "jaw_nolick")
LEV_WIN = (-1.0, 0.0)
EP_PAIRS = (("pre", "acute"), ("pre", "subacute"), ("pre", "chronic"), ("acute", "subacute"), ("acute", "chronic"),
            ("subacute", "chronic"))
LEV_STEM = "stopped_precue_levels"


def precue_levels(labels: list) -> pd.DataFrame:
    """Per animal x epoch x class x area x hemi: -1..0 s pre-cue level (restw and raw), all spouts, trial-weighted
    over the animal's restw_ok sessions (the same pooling as the trace figures; no minimum count -- n is reported)."""
    w = (LAGS >= LEV_WIN[0] - 1e-9) & (LAGS <= LEV_WIN[1] + 1e-9)
    acc = {}
    for x in labels:
        z = np.load(SESS / f"{x}.npz")
        mask_keys.check_obj(z, SESS / f"{x}.npz")  # stale-read guard (11-09)
        rw, ok = _lev(x, z)
        if not ok:
            continue
        cols = [str(c) for c in z["cols"]]
        an, ep = str(z["animal"]), str(z["epoch"])
        for c in lev_classes():
            Y = z[f"{c}__cue"][SETS.index("all")].astype(np.float64)
            n = int(z[f"{c}__cue__n"][SETS.index("all")].max())
            if n == 0:
                continue
            raw = np.nanmean(Y[w], 0)
            for q, col in enumerate(cols):
                a = acc.setdefault((an, ep, c, col), [0.0, 0.0, 0])
                a[0] += n * raw[q]
                a[1] += n * (raw[q] - rw[q])
                a[2] += n
    rows = []
    for (an, ep, c, col), (sr, sw, n) in acc.items():
        area, hemi = col.rsplit("_", 1)
        rows.append({"animal": an, "epoch": ep, "class": c, "area": area, "hemi": hemi, "n": n, "raw": sr / n,
                     "restw": sw / n})
    return pd.DataFrame(rows)


def contra_minus_ipsi(L: pd.DataFrame) -> pd.DataFrame:
    P = L.pivot_table(index=["animal", "epoch", "class", "area"], columns="hemi", values=["raw", "restw"])
    out = pd.DataFrame({b: P[(b, "right")] - P[(b, "left")] for b in ("raw", "restw")}).dropna(how="all").reset_index()
    out["hemi"] = "R-L"
    return out


def within_animal(L: pd.DataFrame, base: str) -> pd.DataFrame:
    """Epoch-pair change per animal, ONLY animals present in both epochs of the pair (the stopped animal sets differ)."""
    rows = []
    for (c, a, h), g in L.groupby(["class", "area", "hemi"]):
        v = g.set_index(["animal", "epoch"])[base]
        for e1, e2 in EP_PAIRS:
            for an in sorted(g.animal.unique()):
                if (an, e1) in v.index and (an, e2) in v.index:
                    rows.append({"class": c, "area": a, "hemi": h, "pair": f"{e2}-{e1}", "animal": an,
                                 "delta": v[(an, e2)] - v[(an, e1)]})
    return pd.DataFrame(rows)


def movement_precue(labels: list) -> pd.DataFrame:
    """Pre-cue movement per trial class from qc/jaw_onsets.csv: treadmill -1..0 s (tread_pre_mm_s, DAQ, all spouts) and
    the 3-D jaw / nose pre-cue displacement (jaw3d_rest_mm / nose3d_rest_mm = 95th pct over -500..0 ms; side spouts only,
    blank at close/far center). Per animal x epoch x class: median over trials (restw_ok sessions)."""
    D = pd.read_csv(OUTD / _o("trial_classes.csv.gz"), low_memory=False)
    D = D[D.restw_ok.astype(str).str.lower().eq("true") & D.label.isin(labels)]
    from wfield_local import orofacial_outputs as oo   # <encoding root>/qc (D9 11-08: il.OUT.parent/qc never existed)
    J = pd.read_csv(oo.encoding_root() / "qc" / "jaw_onsets.csv", low_memory=False,
                    usecols=["session", "trial_id", "tread_pre_mm_s", "jaw3d_rest_mm", "nose3d_rest_mm"])
    J["label"] = J.session.str[:5] + J.session.str[-4:]
    D = D.merge(J.drop(columns="session"), on=["label", "trial_id"], how="left")
    M = class_masks(D)
    rows = []
    for c in lev_classes():
        g = D[M[c].to_numpy()]
        for (an, ep), h in g.groupby(["animal", "epoch"]):
            r = {"class": c, "animal": an, "epoch": ep, "n": len(h)}
            for m in ("tread_pre_mm_s", "jaw3d_rest_mm", "nose3d_rest_mm"):
                r[m] = h[m].median()
                r[f"n_{m}"] = int(h[m].notna().sum())
            r["frac_running_pre"] = float((h.tread_pre_mm_s > 3.0).mean()) if h.tread_pre_mm_s.notna().any() else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def levels(labels: list) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    L = precue_levels(labels)
    L.to_csv(OUTD / _o("precue_levels_per_animal_all_areas.csv"), index=False, float_format="%.3f")
    CI = contra_minus_ipsi(L)
    CI.to_csv(OUTD / _o("precue_levels_contra_minus_ipsi.csv"), index=False, float_format="%.3f")
    key = ["ALM", "tjM1", "SSp-m", "SSp-n", "MOs_rest", "MOp_rest", "SSp-bfd", "VISp"]
    for base in ("restw", "raw"):
        for c in (CAM_GROUPS if _cam() else ("stopped", "working", "success")):
            T = L[(L["class"] == c) & (L.hemi == "right")].pivot_table(index=["animal", "epoch"], columns="area",
                                                                       values=base).reindex(columns=key)
            T = T.reindex([(a, e) for a in sorted(L.animal.unique()) for e in il.EP4 if (a, e) in T.index])
            n = L[(L["class"] == c) & (L.area == "ALM") & (L.hemi == "right")].set_index(["animal", "epoch"]).n
            T.insert(0, "n", n.reindex(T.index))
            print(f"\n== PRE-CUE (-1..0 s) level, {c.upper()}, CONTRA (right), {base} (dF/F x1e3), per animal\n"
                  + T.round(2).to_string())
        T = CI[CI["class"] == "stopped"].pivot_table(index=["animal", "epoch"], columns="area", values=base).reindex(
            columns=key[:-1])
        T = T.reindex([(a, e) for a in sorted(L.animal.unique()) for e in il.EP4 if (a, e) in T.index])
        print(f"\n== STOPPED pre-cue CONTRA minus IPSI ({base}), per animal\n" + T.round(2).to_string())
    # within-animal epoch changes (animals present in both epochs only)
    Lc = pd.concat([L, CI.assign(n=np.nan)], ignore_index=True)
    out = []
    for base in ("restw", "raw"):
        W = within_animal(Lc, base)
        W["base"] = base
        out.append(W)
    W = pd.concat(out, ignore_index=True)
    W.to_csv(OUTD / _o("precue_levels_within_animal_deltas.csv"), index=False, float_format="%.3f")
    S = (W.groupby(["base", "class", "area", "hemi", "pair"]).delta
         .agg(n_animals="size", mean="mean", n_pos=lambda d: int((d > 0).sum()),
              animals=lambda d: "", per_animal=lambda d: "").reset_index())
    per = W.groupby(["base", "class", "area", "hemi", "pair"]).apply(
        lambda g: " ".join(f"{a}:{v:+.1f}" for a, v in zip(g.animal, g.delta)), include_groups=False)
    S["per_animal"] = per.reindex(pd.MultiIndex.from_frame(S[["base", "class", "area", "hemi", "pair"]])).to_numpy()
    S = S.drop(columns="animals")
    S.to_csv(OUTD / _o("precue_levels_within_animal_summary.csv"), index=False, float_format="%.3f")
    for base in ("restw", "raw"):
        sel = S[(S.base == base) & S.area.isin(["ALM", "tjM1", "SSp-m"]) & S.hemi.isin(["right", "R-L"])
                & S.pair.isin(["acute-pre", "subacute-pre", "chronic-pre", "chronic-acute", "subacute-acute",
                               "chronic-subacute"])
                & S["class"].isin(list(main_classes()))]
        print(f"\n== WITHIN-ANIMAL pre-cue change, {base}: animals present in BOTH epochs only (per animal listed)\n"
              + sel.drop(columns="base").round(2).to_string(index=False))
    Mv = movement_precue(labels)
    Mv.to_csv(OUTD / _o("precue_movement_by_class.csv"), index=False, float_format="%.3f")
    T = Mv[Mv["class"].isin(list(main_classes()))].pivot_table(
        index=["animal", "epoch"], columns="class", values=["tread_pre_mm_s", "frac_running_pre", "jaw3d_rest_mm",
                                                            "nose3d_rest_mm"])
    T = T.reindex([(a, e) for a in sorted(Mv.animal.unique()) for e in il.EP4 if (a, e) in T.index])
    print("\n== PRE-CUE movement by class (median over trials; treadmill mm/s -1..0 s; frac trials running > 3 mm/s; 3-D "
          "jaw / nose pre-cue 95th-pct displacement, mm, side spouts only)\n" + T.round(3).to_string())
    save_fd(LEV_STEM, {"L": L, "CI": CI, "W": W, "Mv": Mv}, FIGD)
    draw_levels(LEV_STEM)


def draw_levels(stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = load_fd(stem, FIGD)
    L, CI, Mv = F["L"], F["CI"], F["Mv"]
    areas = ["ALM", "tjM1", "SSp-m", "SSp-n", "MOp_rest", "SSp-bfd"]
    rows = [("restw", "right", "contra (R)"), ("restw", "R-L", "contra - ipsi"), ("raw", "right", "contra (R), RAW")]
    cls = main_classes()
    mk = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    fig, axs = plt.subplots(len(rows) * len(cls), len(areas), figsize=(2.6 * len(areas), 1.9 * len(rows) * len(cls)),
                            sharex=True, squeeze=False, layout="constrained")
    x = np.arange(len(il.EP4))
    for r, (base, hemi, hl) in enumerate(rows):
        D = CI if hemi == "R-L" else L[L.hemi == hemi]
        for k, c in enumerate(cls):
            i = r * len(cls) + k
            for j, a in enumerate(areas):
                ax = axs[i, j]
                _style(ax)
                ax.axhline(0, color="0.6", lw=0.5)
                g = D[(D["class"] == c) & (D.area == a)]
                for an, h in g.groupby("animal"):
                    v = h.set_index("epoch")[base].reindex(il.EP4).to_numpy(float)
                    ax.plot(x, v, color="0.45", lw=0.8, marker=mk.get(an, "o"), ms=3.5, label=an)
                m = g.groupby("epoch")[base].mean().reindex(il.EP4).to_numpy(float)
                ax.plot(x, m, color=_col(c), lw=2.0, marker="_", ms=14)
                if i == 0:
                    ax.set_title(a, fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{_txt(c)}\n{hl}", fontsize=7.5)
    for ax in axs[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(il.EP4, fontsize=7, rotation=30)
    hs = [plt.Line2D([], [], color="0.45", lw=0.8, marker=m, ms=3.5, label=an) for an, m in mk.items()]
    hs += [plt.Line2D([], [], color=_col(c), lw=2.0, label=f"mean, {_txt(c)}") for c in cls]
    fig.legend(handles=hs, fontsize=7, frameon=False, loc="outside lower center", ncol=len(hs))   # was over ALM data
    fig.suptitle(textwrap.fill(_rl(F,
        "NEURAL: pre-cue (-1..0 s) level per animal (gray, markers = animal) and mean of the animals present (color), "
        "all spouts pooled, trial-weighted over sessions. Rows: " + (" / ".join(_txt(c) for c in cls) if _cam() else "hit / miss_working / stopped") + " x (contralesional, above "
        "restw) / (contra minus ipsi, above restw) / (contralesional, RAW dF/F x1e3). RESTW is the engagement-gated "
        "WORKING-period rest, so for stopped (quit-period) trials it is a cross-state reference. Animal sets differ "
        "between epochs for stopped (PS92 acute only; PS95 no acute): compare within animals. " + class_def()
        + ". PROVISIONAL n = 4."), 200), fontsize=8.5)
    save_png_svg(fig, FIGS / f"{_stem(stem)}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # movement
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2), squeeze=False)
    for j, (m, t) in enumerate((("tread_pre_mm_s", "treadmill -1..0 s (mm/s, median)"),
                                ("frac_running_pre", "frac trials running pre-cue (> 3 mm/s)"),
                                ("jaw3d_rest_mm", "3-D jaw pre-cue disp (mm, median)"),
                                ("nose3d_rest_mm", "3-D nose pre-cue disp (mm, median)"))):
        ax = axs[0, j]
        _style(ax)
        for k, c in enumerate(cls):
            g = Mv[Mv["class"] == c]
            for an, h in g.groupby("animal"):
                v = h.set_index("epoch")[m].reindex(il.EP4).to_numpy(float)
                ax.plot(x + (k - (len(cls) - 1) / 2) * 0.12, v, color=_col(c), lw=0.7, marker=mk.get(an, "o"), ms=3.5,
                        alpha=0.8, label=f"{_txt(c)}" if an == "PS94" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(il.EP4, fontsize=7)
        ax.set_title(t, fontsize=8.5)
    axs[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("BEHAVIOR: pre-cue movement by trial class per animal (markers: PS92 o, PS93 s, PS94 ^, PS95 D); "
                 "qc/jaw_onsets.csv tread_pre_mm_s / jaw3d_rest_mm / nose3d_rest_mm (3-D: side spouts only)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, FIGS / f"{_stem(stem)}_movement.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _tc_sel() -> str:
    from wfield_local import trial_classes as tc
    return tc.run_selection()


def main() -> int:
    # wired to the camera trial gate 11-07 (Priya option a): `_states` takes the camera classes under gate=camera
    from wfield_local import analysis_kit as ak
    if plot_only():
        for kind, stem in stems():
            draw(kind, stem)
        draw_levels(LEV_STEM)
        return 0
    if "--levels" in sys.argv:
        levels(il.labels())
        return 0
    if SFX:   # no re-extraction: the session files hold raw traces
        labs = il.labels()
        compute(labs)
        levels(labs)
        return 0
    jobs = min(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 2, 2)
    labs = il.labels()
    if "--only" in sys.argv:
        labs = [x for x in labs if x in sys.argv[sys.argv.index("--only") + 1].split(",")]
    T = pd.read_csv(il.OUT / "enl_licks" / "enl_trials.csv.gz", usecols=["label", "trial_id", "pos", "strobe_s", "cue_s",
                                                                        "engaged"])
    A = pd.read_csv(_lp_licks.attempts_path(), usecols=["label", "trial_id", "pos", "cls",
                                                                         "jaw_onset_ms"])
    items = [(x, T[T.label == x].reset_index(drop=True), A[A.label == x].reset_index(drop=True)) for x in labs]
    OUTD.mkdir(parents=True, exist_ok=True)
    _res, fails = ak.fan_sessions(items, session_worker, jobs=jobs, key=lambda it: it[0], label="stopped traces")
    if fails:
        for it, err in fails:
            print(f"  !! {it[0]}: {err}", flush=True)
        raise SystemExit(f"{len(fails)} sessions failed")
    if "--extract-only" in sys.argv:
        return 0
    compute(labs)
    levels(labs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
