"""BEFORE the lick: is the contralesional read-out / hemispheric imbalance of SPONTANEOUS ITI licks already present
before a self-initiated lick? (Priya 10-29: "look at a time window before spontaneous ITI licks".)

QUESTION. After the stroke the frozen pre-stroke lateral lick decoder reads ITI licks as more CONTRALESIONAL than where
they land (like-for-like offset, chronic for ITI licks; Fig 12.17: mostly a left / right DIFFERENCE, strongest in SSp-n),
and the post-lick hemispheric balance (0.1..0.5 s) tracks recovery. Motor command / plan, or sensory reafference /
post-movement state? A self-initiated lick (no cue, no lick for >= 2 s) whose contralesional reading and imbalance are
there BEFORE the lick points to a motor / preparatory origin; one that appears only after, to reafference.

LICKS. Exactly the primary ITI set: "ITI first after quiet" (lick_recovery_markers.QSET) = ITI licks of the KEY decoder
cache (lick_order_preds_areas_splitr8) whose onset follows the previous lick onset by >= 2 s (lrm._gaps); angle =
asin(side_k / reach_mm) (covref_covariates), spout = the previous trial's spout (pos). Subset "clean" (sensitivity):
also >= 2 s after the last response-window end and >= 0.6 s before the next spout arrival (iti_timing_audit), so no
trial event falls inside -1.5..+0.6 s.

TWO ALIGNMENTS (both from the session files, sessions_roi8fr):
  peak   iti_X, aligned on the DAQ lick PEAK (contact), lags -1.0..+1.0 s: every QSET lick; the primary balance's own
         traces (its post window reproduces lick_recovery_primary exactly -- printed check).
  onset  bout_peth, aligned on the DAQ lick ONSET, lags -1.5..+0.75 s: the QSET licks that are also extraction bout
         starts (no cue / strobe within -2..+0.75 s of the onset; ~70-80 % of them). Peak - onset ~30 ms (median).
  Traces are linearly interpolated between imaging frames, so a sample < 1 frame before the lick can carry a frame
  after it, and the tongue protrudes ~0.1-0.15 s before contact: treat -0.2..0 as POSSIBLY MOVEMENT.
WINDOWS (s from the lick; lower edge inclusive, upper exclusive, except the decoder's own 0..0.2 s, inclusive):
  pre_100_050 -1.0..-0.5, pre_050_020 -0.5..-0.2, pre_020_000 -0.2..0 (possibly movement), dec_000_020 0..0.2 (the
  frozen decoder's training window), post_010_050 0.1..0.5 (the primary balance window).
BASELINES. Decoders: RAW window level (the frozen decoder is trained on raw levels, no per-lick baseline). Balance:
  "raw" = window level; "bs" = window minus an earlier baseline: peak -1.0..-0.5 s (= il.BASE_RESP, the primary's; so
  bs is undefined for pre_100_050 there), onset -1.5..-1.0 s (= il.BASE; starts >= 0.5 s after the previous lick).

DECODE (lateral landing, mm, + = contralesional; z-scored Allen areas with the ALM / tjM1 split, il.area_features / the
frame-level SD):
  (a) frozen  THE 12.17 decoder: iti_lick_order.predictions' per-animal ITI ridge (all pre ITI licks, lags 0..0.2 s,
              pre sessions leave-one-session-out), captured through lo.CONTRIB_HOOK and applied to each window's features
              (check: on dec_000_020 / peak it reproduces the cached predictions).
  (b) window  StandardScaler + RidgeCV (lick_refit_decoders' ridge) trained on the window itself, on the animal's PRE
              QSET licks, pre sessions leave-one-session-out.
  Pre: held-out r of the decoded value with the lateral landing, with the lick angle, and BEYOND SPOUT (pre per-spout
  means removed from both). Post: the like-for-like offset per session = lick_offset_contrib_map.session_dh (Gaussian
  0.1 mm, reference = the animal's PRE QSET licks in the SAME window, pre sessions referenced to the other pre
  sessions), and its exact split beta . dh into per-area terms -> left / right DIFFERENCE / SHARED parts
  (lick_offset_terms_recovery.components' definition; SSp-n difference; oro + SSs difference w/o SSp-n; ...).
  Change vs pre: analysis_kit.boot_delta (paired nested animals > sessions).
  RULE 10: neither decoder goes through frozen_models.load_or_fit. (a) is the 12.17 decoder, itself a deterministic
  in-memory refit in iti_lick_order (not in the frozen store) -- reproduced bit-for-bit here; (b) is an exploratory
  per-window decoder. Freezing either would write into the shared frozen-model store (outside this script's write
  root) and read every pre session's U_atlas / SVTcorr for the signature -- not done; flagged in the report.
BALANCE. Primary-style: per lick, area x side value referenced to the animal's PRE line on lick angle (per animal x lick
  order, pre sessions leave-one-session-out; vectorised lrm._residuals); pool ALM, tjM1, SSp-m, SSp-n, SSs (contra =
  right hemisphere). ratio = log2(sum obs contra / sum predicted contra) - log2(same, ipsi) (bs only; NaN when a sum is
  <= 0 -- a near-zero pre-lick response makes it unstable, see n_ratio_nan), diff = mean over areas of the session-mean
  referenced (contra - ipsi) (raw and bs). By epoch (boot_delta) and recovery tracking within animals
  (lick_sspn_checks._track: per-animal r with ITI lick angle and the cued far-contra aim, MixedLM within-animal slope).
TIME COURSE. Sliding 0.1 s bins (2 samples, step 0.05 s) from -1.5 (onset; -1.0 peak) to +0.6 s: frozen offset, its
  SSp-n difference and all-pairs difference terms, balance diff raw / bs, change vs pre per epoch (boot_delta_columns,
  the same draw as boot_delta).
Outputs <OUT_W>/iti_prelick_licks.csv.gz (lick set), iti_prelick_sessions.csv (window x measure per session),
  iti_prelick_decode_r.csv (pre held-out r per animal), iti_prelick_tests.csv (change vs pre), iti_prelick_animals.csv
  (per-animal epoch change), iti_prelick_track.csv (recovery tracking), iti_prelick_timecourse_sessions.csv,
  iti_prelick_timecourse_tests.csv; figure figures/lick_iti_prelick.png.

10-29 LATE -- WINDOW LIMIT (Priya: "we probably need to use a window that ends at least ~0.1 s before the lick onset").
  The v1 "pre 0.2..0" window and the sliding bins next to 0 overlap tongue protrusion. --window-limit L (default 0.1 s):
  every PRE window ends >= L s before the lick ONSET (no sample later than -L; both edges inclusive, so -0.3..-0.1 holds
  the samples -0.30 .. -0.10). Windows are referenced PER LICK to the DAQ lick onset (peak-aligned traces: the lags are
  shifted by that lick's onset - peak, ~-30 ms) or to the cam4 JAW-OPENING onset:
    pre_100_050 -1.0..-0.5, pre_050_020 -0.5..-0.2 (v1's), pre_050_015 -0.5..-0.15, pre_030_010 -0.3..-0.1,
    pre_025_015 -0.25..-0.15 (from lick onset); jaw_050_010 / jaw_030_010 / jaw_030_015 = -0.5..-0.1 / -0.3..-0.1 /
    -0.3..-0.15 from the jaw-opening onset; windows ending later than -L are dropped (L = 0.15 keeps the -0.15 ones).
  Reference only (NOT pre-lick): str_020_000 -0.2..0 (v1's pre_020_000; STRADDLES onset), dec_000_020, post_010_050.
  Imaging is 31.23 Hz per channel and the traces are linearly interpolated, so a sample at -0.10 s carries the frame
  after it (<= 32 ms later): the L = 0.1 windows can hold imaging up to ~-0.07 s; the -0.15 windows up to ~-0.12 s.
  bs (baseline-subtracted) per lick: NaN where the window's first sample lies before the baseline's end.
  Licks without a feature value in a window (no jaw onset, no onset time) are left out of that window.
  Sliding bins: unchanged (0.1 s = 2 samples), reported by their LAST sample; a bin counts as pre-lick only if its last
  sample is <= -L from onset (peak-aligned: <= -L - the median onset lag).
JAW ONSET (cam4 v36 cleaned jaw_y, + = opening; windows.npz frame DAQ times, nearest frame within 6 ms on a 4 ms grid;
  NOT at close_mid -- chin hidden, as jaw_onsets.py): jaw_kinematics' rule (threshold max(4 SD, 4 px), >= 5 of 8 frames)
  on the OPENING deflection only, baseline -1.0..-0.6 s and detect -0.6..+0.05 s from the DAQ lick onset; onset = the
  first suprathreshold frame of the first passing window; an onset on the first detect frame (already opening) is
  CENSORED and not used. Also the video TONGUE onset (licks_all rise_start_ms, through the trial's frame clock).
  cam1 is not used (its LP jaw is only in the trial windows).
FROZEN DECODER (10-29 late, RULE 10): now READ from the frozen store -- frozen_models.load_or_fit, kind ``lick_reader``,
  align ``iti`` (frozen by lick_enlv_extract.py --readers and verified there against the 12.17 cache to 4e-16); a miss
  raises instead of fitting (this script writes nothing outside iti_licks/). The CHECK below still compares it with the
  cached 12.17 predictions. The per-window decoder (b) remains an exploratory in-memory fit (not a pre-stroke reference
  that post data are scored in -- each window's own pre-trained read-out).
New outputs (L = 0.1): iti_prelick_w01_{licks.csv.gz, sessions, decode_r, tests, animals, track, timecourse_sessions,
  timecourse_tests, jaw}.csv; figure figures/lick_iti_prelick.png (v1 copied to lick_iti_prelick_v1.png). Other L:
  iti_prelick_w<L>_* and lick_iti_prelick_w<L>.png. --window-limit none = v1 exactly (v1 windows, v1 file names; the
  figure goes to lick_iti_prelick_v1.png).
Usage: python scripts/lick_iti_prelick.py [--plot-only] [--window-limit 0.1|0.15|none] [--jobs 4]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_offset_contrib_map as ocm  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
ORO = set(POOL)
WINS_V1 = [("pre_100_050", -1.0, -0.5), ("pre_050_020", -0.5, -0.2), ("pre_020_000", -0.2, 0.0),
           ("dec_000_020", 0.0, 0.2), ("post_010_050", 0.1, 0.5)]
# (name, lo, hi, reference, group): reference "lag" = fixed lags of the alignment (upper edge exclusive, as v1; dec
# inclusive), "onset" / "jaw" = per lick from the DAQ lick onset / the cam4 jaw-opening onset, both edges inclusive
WINS_NEW = [("pre_100_050", -1.0, -0.5, "onset", "pre"), ("pre_050_020", -0.5, -0.2, "onset", "pre"),
            ("pre_050_015", -0.5, -0.15, "onset", "pre"), ("pre_030_010", -0.3, -0.1, "onset", "pre"),
            ("pre_025_015", -0.25, -0.15, "onset", "pre"),
            ("jaw_050_010", -0.5, -0.1, "jaw", "jaw"), ("jaw_030_010", -0.3, -0.1, "jaw", "jaw"),
            ("jaw_030_015", -0.3, -0.15, "jaw", "jaw"),
            ("str_020_000", -0.2, 0.0, "lag", "straddle"), ("dec_000_020", 0.0, 0.2, "lag", "post"),
            ("post_010_050", 0.1, 0.5, "lag", "post")]
WLAB = {"pre_100_050": "-1.0..-0.5\nonset", "pre_050_020": "-0.5..-0.2\nonset", "pre_050_015": "-0.5..-0.15\nonset",
        "pre_030_010": "-0.3..-0.1\nonset", "pre_025_015": "-0.25..-0.15\nonset", "jaw_050_010": "-0.5..-0.1\nJAW",
        "jaw_030_010": "-0.3..-0.1\nJAW", "jaw_030_015": "-0.3..-0.15\nJAW", "str_020_000": "-0.2..0\nSTRADDLES",
        "dec_000_020": "0..0.2\n(post)", "post_010_050": "0.1..0.5\n(post)", "pre_020_000": "-0.2..0\n(straddles)"}
LIMIT: float | None = 0.1           # set by configure(); None = v1
WINS: list = []
TAG = ""
FIGNAME = "lick_iti_prelick.png"
SRC = {"peak": (il.LAGS_D, il.BASE_RESP), "onset": (il.LAGS_P, il.BASE)}
JAW_BASE = (-1.0, -0.6)             # s from the DAQ lick onset
JAW_DET = (-0.6, 0.05)
JAW_DT = 0.004
JAW_HIDDEN = ("close_mid",)          # cam4 chin hidden (jaw_onsets.py masks close_center)


def configure(limit: float | None) -> None:
    """Set the window set + output names for a window limit (None = v1)."""
    global LIMIT, WINS, TAG, FIGNAME
    LIMIT = limit
    if limit is None:
        WINS = [(n, lo_, hi_, "lag", "v1") for n, lo_, hi_ in WINS_V1]
        TAG, FIGNAME = "", "lick_iti_prelick_v1.png"
        return
    WINS = [w for w in WINS_NEW if w[4] not in ("pre", "jaw") or w[2] <= -limit + 1e-9]
    TAG = "w" + f"{limit:g}".replace("0.", "0")
    FIGNAME = "lick_iti_prelick.png" if abs(limit - 0.1) < 1e-9 else f"lick_iti_prelick_{TAG}.png"


def fn(stem: str) -> Path:
    return OUT / (f"iti_prelick_{TAG}_{stem}" if TAG else f"iti_prelick_{stem}")
SUBSETS = ("all", "clean")
SIGMA = ocm.SIGMA
MIN_CELL = 20                     # = lick_balance_regions.MIN_CELL (pre licks for a pre line)
MIN_PRE_LICKS = 10                # window decoder: pre licks per training session
TC_HI = 0.6
OUT = il.OUT_W
FIG = OUT / "figures"
COMP = ["total", "diff_all", "sum_all", "diff_SSp-n", "diff_oro_noSSpn", "sum_oro", "other", "unpaired"]
_RUN = {"jobs": 4}                # task cap: at most 4 workers (a dict: main() sets it without a module global)


# ------------------------------------------------------------------------------------------------------- lick set
def lick_set() -> pd.DataFrame:
    P = pd.read_csv(lo.preds_path())
    P["k"] = P.groupby(["set", "label"]).cumcount()
    P = P[P.set == "iti"].copy()
    C = pd.read_csv(OUT / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "t", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), lrm._gaps, jobs=_RUN["jobs"])
    if fail:
        raise RuntimeError(f"gaps failed: {fail[:5]}")
    G = pd.concat([v for _i, v in res], ignore_index=True)
    P = P.merge(G, on=["set", "label", "k"], how="left", validate="one_to_one")
    acts = P.groupby("label").act.apply(lambda s: s.to_numpy(float))
    Q = P[P.gap >= il.GAP_MAIN].copy()
    Q["angle"] = np.degrees(np.arcsin(np.clip(Q.side_k / Q.reach_mm, -1, 1)))
    Q["ord_c"] = np.minimum(Q.order, lo.ORDER_CAP)
    A = pd.read_csv(il.OUT / "iti_timing_audit.csv", usecols=["label", "t", "since_stop", "to_strobe"])
    A = A.replace(np.inf, 1e9).drop_duplicates(["label", "t"])
    Q = Q.merge(A, on=["label", "t"], how="left")
    Q["clean"] = (Q.since_stop >= 2.0) & (Q.to_strobe >= TC_HI)
    Q = Q.sort_values(["label", "k"]).reset_index(drop=True)
    return Q, acts


def _one(item):
    """One session (module-level for spawn): item = (label, peak times of its QSET licks). Returns the peak-aligned
    traces (n, LAGS_D, cols) and the onset-aligned bout traces (n, LAGS_P, cols; NaN where the lick is not a bout start)."""
    from wfield_local import config
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    label, tq, pos_q = item
    tq = np.asarray(tq, float)
    d = il._load(label)
    it = d["iti_t"].astype(float)
    o = np.argsort(it)
    j = np.clip(np.searchsorted(it[o], tq), 0, len(it) - 1)
    rows = o[j]
    okp = np.abs(it[rows] - tq) < 1e-9
    Xp = np.full((len(tq), len(il.LAGS_D), len(d["cols"])), np.nan, np.float32)
    Xp[okp] = d["iti_X"][rows[okp]].astype(np.float32)
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    sd = oo.session_dir(PathResolver(), oc.RUN, an, date)
    A = pd.read_csv(sd / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s", "trial_id", "rise_start_ms"])
    A = A[np.isfinite(A.daq_on_s)].drop_duplicates("daq_peak_s")
    A = A.set_index("daq_peak_s").reindex(tq)
    on = A.daq_on_s.to_numpy(float)
    jaw = jaw_and_tongue(sd, on, np.asarray(pos_q).astype(str), A.trial_id.to_numpy(float),
                         A.rise_start_ms.to_numpy(float))
    bt = d["bout_t"].astype(float)
    Xo = np.full((len(tq), len(il.LAGS_P), len(d["cols"])), np.nan, np.float32)
    oko = np.zeros(len(tq), bool)
    if len(bt):
        ob = np.argsort(bt)
        jb = np.clip(np.searchsorted(bt[ob], on), 0, len(bt) - 1)
        rb = ob[jb]
        oko = np.isfinite(on) & (np.abs(bt[rb] - on) < 1e-9)
        Xo[oko] = d["bout_peth"][rb[oko]].astype(np.float32)
    return {"Xp": Xp, "Xo": Xo, "okp": okp, "oko": oko, "on": on, "cols": [str(c) for c in d["cols"]], **jaw}


def jaw_and_tongue(sd: Path, on: np.ndarray, pos: np.ndarray, trial_id: np.ndarray, rise_ms: np.ndarray) -> dict:
    """Per lick (DAQ onset `on`): cam4 jaw-OPENING onset (DAQ s; jaw_kinematics' threshold + density rule, opening only)
    and the video tongue onset (licks_all rise_start_ms through the trial's frame clock), see the module docstring."""
    from wfield_local import jaw_kinematics as jk
    p = jk._resolve(None)
    n = len(on)
    out = {"jaw_on": np.full(n, np.nan), "jaw_state": np.full(n, "", dtype=object), "tongue_on": np.full(n, np.nan)}
    w = np.load(sd / "windows.npz")
    t = w["ix_daq_s"].astype(float)
    y = np.load(sd / "clean.npz")["jaw_y"].astype(float)
    tid, tms = w["ix_trial_id"].astype(float), w["ix_t_ms"].astype(float)
    o = np.argsort(t, kind="stable")
    ts, ys = t[o], y[o]
    rel = np.round(np.arange(JAW_BASE[0], JAW_DET[1] + 1e-9, JAW_DT), 4)
    t_ms = rel * 1000.0
    W, K = max(1, int(p["window_n"])), max(1, int(np.ceil(p["density_thr"] * p["window_n"])))
    det = (rel >= JAW_DET[0] - 1e-9) & (rel <= JAW_DET[1] + 1e-9)
    idx_det = np.flatnonzero(det)
    for i in range(n):
        if not np.isfinite(on[i]):
            out["jaw_state"][i] = "no_onset"
            continue
        if pos[i] in JAW_HIDDEN:
            out["jaw_state"][i] = "hidden"
            continue
        g = on[i] + rel
        j = np.clip(np.searchsorted(ts, g), 1, len(ts) - 1)
        j = np.where(np.abs(ts[j - 1] - g) < np.abs(ts[j] - g), j - 1, j)
        yw = np.where(np.abs(ts[j] - g) <= 0.006, ys[j], np.nan)
        bm, bsd = jk._compute_jaw_baseline(yw, t_ms, baseline_win_ms=(JAW_BASE[0] * 1000, JAW_BASE[1] * 1000))
        if not (np.isfinite(bm) and np.isfinite(bsd) and bsd > 0) or np.isfinite(yw[det]).mean() < 0.5:
            out["jaw_state"][i] = "no_data"
            continue
        thr, _ = jk._compute_jaw_threshold(bsd, sd_mult=p["sd_mult"], min_px=p["min_px"])
        s = (np.isfinite(yw[idx_det]) & ((yw[idx_det] - bm) > thr)).astype(int)
        passing = np.convolve(s, np.ones(W, int), mode="valid") >= K
        if not passing.any():
            out["jaw_state"][i] = "none"
            continue
        f0 = int(np.flatnonzero(passing)[0])
        k = f0 + int(np.flatnonzero(s[f0:f0 + W])[0])
        if k == 0:
            out["jaw_state"][i] = "censored"
            continue
        out["jaw_on"][i] = on[i] + rel[idx_det[k]]
        out["jaw_state"][i] = "onset"
    # video tongue onset: the lick's own trial frame clock (ix_t_ms -> ix_daq_s)
    ok = np.isfinite(trial_id) & np.isfinite(rise_ms)
    if ok.any():
        for tr in np.unique(trial_id[ok]):
            m = tid == tr
            if m.sum() < 2:
                continue
            tt, dd = tms[m], t[m]
            oo_ = np.argsort(tt, kind="stable")
            sel = ok & (trial_id == tr)
            r = rise_ms[sel]
            v = np.interp(r, tt[oo_], dd[oo_])
            v[(r < tt.min()) | (r > tt.max())] = np.nan
            out["tongue_on"][sel] = v
    return out


# ------------------------------------------------------------------------------------------------------- helpers
def wmask(lags: np.ndarray, lo_: float, hi_: float, incl: bool = False) -> np.ndarray:
    return (lags >= lo_ - 1e-9) & ((lags <= hi_ + 1e-9) if incl or hi_ >= lags.max() - 1e-9 else (lags < hi_ - 1e-9))


def _no_fit():
    raise RuntimeError("frozen lick_reader/iti not in the store -- run lick_enlv_extract.py --readers first; this script "
                       "never fits a frozen reference")


def frozen_fits() -> dict:
    """{animal: (labels the 12.17 decoder scores, fits)}: the 12.17 ITI decoder READ from the frozen store (RULE 10;
    kind lick_reader, align iti -- lick_enlv_extract.freeze's spec). A miss raises (nothing is fitted or written)."""
    import hashlib

    import iti_where_decoder as wd
    import lick_enlv_extract as lee

    from wfield_local import epochs
    from wfield_local import frozen_models as fm
    names, SD = wd.area_meta(basis="split")
    pre = [x for x in il.labels() if str(epochs.epoch_of(x)) == "pre"]
    res, fail = ak.fan_sessions([(x, "areas_split", "") for x in pre], wd._load_one, jobs=_RUN["jobs"], key=lambda it: it[0],
                                label="reader load", log=lambda m: None)
    if fail:
        raise RuntimeError(f"reader load failed: {fail[:2]}")
    D = {x: rec for _it, (x, rec) in res if rec is not None}
    P = pd.read_csv(lo.preds_path(), usecols=["set", "animal", "label"])
    F = {}
    for a in AN:
        S = lee.reader_training(a, "iti", D, names, SD)
        labs = sorted(S)
        dig = hashlib.sha1(b"".join(np.ascontiguousarray(S[x][0]).tobytes() + np.ascontiguousarray(S[x][1]).tobytes()
                                    for x in labs)).hexdigest()
        spec = fm.make_spec(a, "lick_reader", align="iti", source=f"areas_split_r{il.ROI_VERSION}", train_labels=labs,
                            post_s=0.2, n_features=S[labs[0]][0].shape[1],
                            train_data={"xy_sha1": dig})   # = lick_enlv_extract.py (same digest; FREEZE_VERSION 2)
        if fm.find(spec)[0] is None:
            _no_fit()
        Pz, status = fm.load_or_fit(spec, _no_fit)
        print(f"[frozen] {a} lick_reader iti: {status}; training data {'IDENTICAL' if Pz.get('xy_sha1') == dig else '!! DIFFERENT'}",
              flush=True)
        F[a] = (set(P[(P.set == "iti") & (P.animal == a)].label), Pz["fits"])
    return F


def fit_window(Zpre: dict, ypre: dict) -> dict:
    """{held-out pre label | None: (scaler, ridge)} trained on the window features of pre licks."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    labs = [x for x in sorted(Zpre) if len(ypre[x]) >= MIN_PRE_LICKS]
    out = {}
    if len(labs) < 3:
        return out
    for held in [*labs, None]:
        trn = [q for q in labs if q != held]
        Xt, yt = np.vstack([Zpre[q] for q in trn]), np.concatenate([ypre[q] for q in trn])
        sc = StandardScaler().fit(Xt)
        out[held] = (sc, RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), yt))
    return out


def split_terms(beta: np.ndarray, dh: np.ndarray, names: list[str]) -> dict:
    """The exact components of beta . dh (lick_offset_terms_recovery.components)."""
    rec = dict.fromkeys(COMP, 0.0)
    rec["total"] = float(beta @ dh)
    base = {}
    for i, n in enumerate(names):
        ar, h = n.rsplit("_", 1)
        base.setdefault(ar, {})[h] = i
    for ar, hh in base.items():
        if {"left", "right"} <= set(hh):
            r_, l_ = hh["right"], hh["left"]
            d_ = 0.5 * (beta[r_] - beta[l_]) * (dh[r_] - dh[l_])
            s_ = 0.5 * (beta[r_] + beta[l_]) * (dh[r_] + dh[l_])
            rec["diff_all"] += d_
            rec["sum_all"] += s_
            if ar == "SSp-n":
                rec["diff_SSp-n"] += d_
            elif ar in ORO:
                rec["diff_oro_noSSpn"] += d_
            if ar in ORO:
                rec["sum_oro"] += s_
            else:
                rec["other"] += d_ + s_
        else:
            rec["unpaired"] += float(sum(beta[i] * dh[i] for i in hh.values()))
    return rec


def residualize(V: np.ndarray, M: pd.DataFrame) -> np.ndarray:
    """V (n, m) minus the animal's PRE line on angle per animal x lick order; pre licks by the line fitted without their
    own session (vectorised lrm._residuals; rows with any NaN value or NaN angle are not fitted / get NaN)."""
    R = np.full(V.shape, np.nan)
    ok = np.isfinite(V).all(1) & np.isfinite(M.angle.to_numpy(float))
    ang = M.angle.to_numpy(float)
    lab = M.label.to_numpy()
    pre = (M.epoch == "pre").to_numpy()
    for _k, g in M.groupby(["animal", "ord_c"]).indices.items():
        g = np.asarray(g)
        pk = g[pre[g] & ok[g]]

        def fit(ix):
            if len(ix) < MIN_CELL or ang[ix].std() == 0:
                return None
            X = np.column_stack([np.ones(len(ix)), ang[ix]])
            return np.linalg.lstsq(X, V[ix], rcond=None)[0]
        B = fit(pk)
        post = g[~pre[g]]
        if B is not None and len(post):
            R[post] = V[post] - np.column_stack([np.ones(len(post)), ang[post]]) @ B
        for lb in np.unique(lab[pk]):
            Bq = fit(pk[lab[pk] != lb])
            sel = g[pre[g] & (lab[g] == lb)]
            if Bq is not None:
                R[sel] = V[sel] - np.column_stack([np.ones(len(sel)), ang[sel]]) @ Bq
    R[~ok] = np.nan
    return R


def balance_sessions(M: pd.DataFrame, vals: dict, kind: str) -> pd.DataFrame:
    """vals {area: (contra (n,), ipsi (n,))} -> per session diff (+ ratio for 'bs')."""
    V = np.column_stack([v for a in POOL for v in vals[a]])
    R = residualize(V, M)
    out = []
    for lab, ix in M.groupby("label").indices.items():
        o, r = V[ix], R[ix]
        okk = np.isfinite(r)
        rec = {"label": lab, "n_bal": int(okk.all(1).sum())}
        d = [np.nanmean(r[:, 2 * i] - r[:, 2 * i + 1]) if okk[:, 2 * i].any() else np.nan for i in range(len(POOL))]
        rec["bal_diff"] = float(np.mean(d)) if np.isfinite(d).all() else np.nan
        if kind == "bs":
            g = []
            for s in (0, 1):
                cs = [2 * i + s for i in range(len(POOL))]
                ob = np.where(okk[:, cs], o[:, cs], 0).sum()
                pr = np.where(okk[:, cs], o[:, cs] - r[:, cs], 0).sum()
                g.append(np.log2(ob / pr) if ob > 0 and pr > 0 else np.nan)
            rec["bal_ratio"] = g[0] - g[1]
        out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------------- compute
def win_values(A: np.ndarray, lags: np.ndarray, lo_: float, hi_: float, ref: str, incl: bool, off: np.ndarray | None,
               base_hi: float | None = None) -> np.ndarray:
    """(n, cols) window mean of A (n, lags, cols). ref 'lag' = fixed lags (v1's .mean(1), bit-identical); 'onset' /
    'jaw' = per lick, lags - off in [lo, hi] (both inclusive). base_hi: NaN for licks whose first window sample lies
    before it (bs would subtract a baseline that overlaps / follows the window)."""
    if ref == "lag":
        return A[:, wmask(lags, lo_, hi_, incl)].mean(1)
    r = lags[None, :] - off[:, None]
    M = (r >= lo_ - 1e-9) & (r <= hi_ + 1e-9)
    cnt = M.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(M[:, :, None], A, 0.0).sum(1) / cnt[:, None]
    bad = cnt == 0
    if base_hi is not None:
        first = np.where(M.any(1), lags[np.argmax(M, 1)], np.nan)
        bad |= ~(first >= base_hi - 1e-9)
    v[bad] = np.nan
    return v


def compute() -> None:
    import iti_where_decoder as wd
    Q, acts = lick_set()
    items = [(lab, tuple(g.t.to_numpy(float)), tuple(g.pos.astype(str))) for lab, g in Q.groupby("label", sort=True)]
    res, fail = ak.fan_sessions(items, _one, key=lambda it: it[0], jobs=_RUN["jobs"], label="prelick load")
    if fail:
        raise RuntimeError(f"session load failed: {[f[0] if isinstance(f, tuple) else f for f in fail][:5]}")
    T = {it[0]: v for it, v in res}
    COLS = T[items[0][0]]["cols"]
    Q["ok_peak"] = np.concatenate([T[lab]["okp"] for lab, *_ in items])
    Q["ok_onset"] = np.concatenate([T[lab]["oko"] for lab, *_ in items])
    Q["t_onset"] = np.concatenate([T[lab]["on"] for lab, *_ in items])
    Q["t_jaw"] = np.concatenate([T[lab]["jaw_on"] for lab, *_ in items])
    Q["jaw_state"] = np.concatenate([T[lab]["jaw_state"] for lab, *_ in items])
    Q["t_tongue"] = np.concatenate([T[lab]["tongue_on"] for lab, *_ in items])
    Q["jaw_lead"] = Q.t_onset - Q.t_jaw                       # s the jaw opens before the DAQ lick onset
    Q["tongue_lead"] = Q.t_onset - Q.t_tongue                 # s the video tongue starts before the DAQ lick onset
    print(f"== QSET licks: {len(Q)}; peak-matched {Q.ok_peak.mean():.3f}; onset (bout-start) matched {Q.ok_onset.mean():.3f}; "
          f"clean {Q.clean.mean():.3f}; peak - onset ms median {np.nanmedian((Q.t - Q.t_onset) * 1000):.1f}", flush=True)
    print(Q.groupby(["animal", "epoch"]).agg(n=("k", "size"), onset=("ok_onset", "mean"), clean=("clean", "mean"))
          .round(2).unstack("epoch").to_string(), flush=True)
    jaw_report(Q)
    Q.drop(columns=[c for c in ("gap",) if c in Q]).to_csv(fn("licks.csv.gz"), index=False)
    names, SD = wd.area_meta()
    F = frozen_fits()
    # z-scored decoder features over all lags, per source x label
    Z, X = {}, {}
    for src in SRC:
        key = "Xp" if src == "peak" else "Xo"
        X[src] = {lab: T[lab][key] for lab, *_ in items}
        Z[src] = {lab: (il.area_features(X[src][lab].astype(float), COLS, lab) /
                        SD.loc[lab, names].to_numpy(float)[None, None, :]).astype(np.float32) for lab, *_ in items}
    # CHECK: the frozen decoder on dec_000_020 / peak reproduces the cached predictions
    m = wmask(il.LAGS_D, 0.0, 0.2, incl=True)
    d1, d0 = [], []
    for lab, g in Q.groupby("label"):
        a = g.animal.iloc[0]
        if a not in F or lab not in F[a][0]:
            continue
        sc, mdl = F[a][1][lab if lab in F[a][1] else None]
        d1.append(mdl.predict(sc.transform(Z["peak"][lab][:, m].mean(1))))
        d0.append(g.dec.to_numpy(float))
    d1, d0 = np.concatenate(d1), np.concatenate(d0)
    print(f"== CHECK frozen decoder reproduction (dec_000_020, peak): max |diff| {np.nanmax(np.abs(d1 - d0)):.2e} over "
          f"{len(d1)} licks", flush=True)
    ci = {c: [COLS.index(f"{a}_right") for a in POOL] if c == "c" else [COLS.index(f"{a}_left") for a in POOL]
          for c in ("c", "i")}

    SESS, RR, TC = [], [], []
    for src, (lags, base) in SRC.items():
        okcol = "ok_peak" if src == "peak" else "ok_onset"
        t_al = "t" if src == "peak" else "t_onset"
        bm = wmask(lags, *base)
        bins = [(round(b, 3), round(b + 0.1, 3)) for b in np.arange(lags[0], TC_HI - 0.1 + 1e-9, 0.05)]
        for sub in SUBSETS:
            sel = Q[okcol].to_numpy(bool) & (Q.clean.to_numpy(bool) if sub == "clean" else True)
            pos_in, meta = {}, {}                  # row positions inside the per-label arrays (Q order), lick metadata
            for lab, ix in Q.groupby("label").indices.items():
                if sel[ix].any():
                    pos_in[lab] = np.flatnonzero(sel[ix])
                    meta[lab] = Q.iloc[ix[sel[ix]]]
            labs = sorted(pos_in)
            M = Q[sel]
            offs = {lab: {"onset": (meta[lab].t_onset - meta[lab][t_al]).to_numpy(float),
                          "jaw": (meta[lab].t_jaw - meta[lab][t_al]).to_numpy(float)} for lab in labs}

            def windows_of(lo_, hi_, ref, incl=False, bsb=None, lags=lags, src=src, pos_in=pos_in, labs=labs, offs=offs):
                return ({lab: win_values(Z[src][lab][pos_in[lab]], lags, lo_, hi_, ref, incl, offs[lab].get(ref))
                         for lab in labs},
                        {lab: win_values(X[src][lab][pos_in[lab]], lags, lo_, hi_, ref, incl, offs[lab].get(ref), bsb)
                         for lab in labs})
            Xb = {lab: X[src][lab][pos_in[lab]][:, bm].mean(1) for lab in labs}
            specs = [(n, lo_, hi_, n == "dec_000_020", "win", ref) for n, lo_, hi_, ref, _g in WINS] + \
                    [(f"tc_{b0:+.2f}", b0, b1, False, "tc", "lag") for b0, b1 in bins]
            for wname, lo_, hi_, incl, kind, ref in specs:
                if ref == "lag" and lo_ < lags[0] - 1e-9:
                    continue
                Zw, Xw = windows_of(lo_, hi_, ref, incl)
                Xw_bs = windows_of(lo_, hi_, ref, incl, base[1])[1] if ref != "lag" else Xw
                keep = {lab: np.isfinite(Zw[lab]).all(1) for lab in labs}
                rows = {lab: {"source": src, "subset": sub, "window": wname, "w_lo": lo_, "w_hi": hi_, "w_ref": ref,
                              "label": lab, "animal": meta[lab].animal.iloc[0], "epoch": meta[lab].epoch.iloc[0],
                              "n": len(meta[lab]), "n_win": int(keep[lab].sum())}
                        for lab in labs}
                labs_d = [lab for lab in labs if keep[lab].any()]
                Zd = {lab: Zw[lab][keep[lab]] for lab in labs_d}
                md = {lab: meta[lab][keep[lab]] for lab in labs_d}
                # ---- decoders
                for dname in (("frozen", "window") if kind == "win" else ("frozen",)):
                    for a in AN:
                        la = [lab for lab in labs_d if rows[lab]["animal"] == a]
                        if dname == "frozen":
                            if a not in F:
                                continue
                            la = [lab for lab in la if lab in F[a][0]]
                            fits = F[a][1]
                        else:
                            pre = [lab for lab in la if rows[lab]["epoch"] == "pre"]
                            fits = fit_window({x: Zd[x] for x in pre}, {x: md[x].act.to_numpy(float) for x in pre})
                            if not fits:
                                continue
                            la = [lab for lab in la if rows[lab]["epoch"] != "pre" or lab in fits]
                        if not any(rows[x]["epoch"] == "pre" for x in la):
                            continue
                        sc0, m0 = fits[None]
                        beta = m0.coef_ / sc0.scale_

                        def dec(x, Xs, fits=fits):
                            s_, m_ = fits[x if x in fits else None]
                            return m_.predict(s_.transform(Xs))
                        S = {x: (rows[x]["epoch"], Zd[x], md[x].act.to_numpy(float)) for x in la}
                        sess = ocm.session_dh(S, dec, SIGMA)
                        for x, (_e, dh, ro) in sess.items():
                            sp = split_terms(beta, dh, names)
                            rows[x].update({f"{dname}|readout": ro, **{f"{dname}|{k}": v for k, v in sp.items()}})
                        if kind == "win":                          # pre held-out r
                            pre = [x for x in la if rows[x]["epoch"] == "pre"]
                            dv = np.concatenate([dec(x, Zd[x]) for x in pre])
                            mm = pd.concat([md[x] for x in pre])
                            yv, av, pv = mm.act.to_numpy(float), mm.angle.to_numpy(float), mm.pos.to_numpy()
                            dvb = dv - pd.Series(dv).groupby(pv).transform("mean").to_numpy()
                            yvb = yv - pd.Series(yv).groupby(pv).transform("mean").to_numpy()
                            oka = np.isfinite(av)
                            RR.append({"source": src, "subset": sub, "window": wname, "decoder": dname, "animal": a,
                                       "n_pre": len(dv), "n_pre_sess": len(pre), "r_lateral": np.corrcoef(dv, yv)[0, 1],
                                       "r_angle": np.corrcoef(dv[oka], av[oka])[0, 1],
                                       "r_beyond_spout": np.corrcoef(dvb, yvb)[0, 1]})
                # ---- balance (all licks; rows without a window value are NaN and not fitted / averaged)
                Mw = pd.concat([meta[lab] for lab in labs], ignore_index=True)
                for bk in ("raw", "bs"):
                    if bk == "bs" and ref == "lag" and lo_ < base[1] - 1e-9:   # window overlaps / precedes the baseline
                        continue
                    Xs_ = Xw_bs if bk == "bs" else Xw
                    vals = {}
                    for i, a in enumerate(POOL):
                        vc = np.concatenate([Xs_[lab][:, ci["c"][i]] - (Xb[lab][:, ci["c"][i]] if bk == "bs" else 0)
                                             for lab in labs])
                        vi = np.concatenate([Xs_[lab][:, ci["i"][i]] - (Xb[lab][:, ci["i"][i]] if bk == "bs" else 0)
                                             for lab in labs])
                        vals[a] = (vc, vi)
                    B = balance_sessions(Mw, vals, bk)
                    for r in B.itertuples():
                        rows[r.label][f"bal_{bk}|diff"] = r.bal_diff
                        if bk == "bs":
                            rows[r.label]["bal_bs|ratio"] = r.bal_ratio
                        rows[r.label][f"n_bal_{bk}"] = r.n_bal
                (SESS if kind == "win" else TC).extend(rows.values())
            print(f"   {src} / {sub}: {len(labs)} sessions, {len(M)} licks", flush=True)
    S = pd.DataFrame(SESS)
    TCs = pd.DataFrame(TC)
    S.to_csv(fn("sessions.csv"), index=False)
    TCs.to_csv(fn("timecourse_sessions.csv"), index=False)
    pd.DataFrame(RR).to_csv(fn("decode_r.csv"), index=False)
    # CHECK vs the primary marker (post_010_050 bs ratio, peak, all) -- lick_recovery_primary.marker on balsub_sessions
    try:
        import lick_recovery_primary as lrp
        Bs = pd.read_csv(OUT / "balsub_sessions.csv")
        Bs = Bs[Bs.set == lrm.QSET]
        mk = pd.Series(lrp.marker(Bs, "ratio_010_050").to_numpy(), index=Bs.label)
        mine = S[(S.source == "peak") & (S.subset == "all") & (S.window == "post_010_050")].set_index("label")["bal_bs|ratio"]
        j = pd.concat([mk.rename("primary"), mine.rename("here")], axis=1).dropna()
        print(f"== CHECK vs primary marker (post 0.1-0.5, peak, all): n {len(j)}, r {np.corrcoef(j.primary, j.here)[0, 1]:.6f}, "
              f"max |diff| {np.abs(j.primary - j.here).max():.2e}", flush=True)
    except Exception as ex:                                        # noqa: BLE001
        print(f"== CHECK vs primary marker failed: {ex}")
    tests(S, TCs)


def jaw_report(Q: pd.DataFrame) -> None:
    """Per animal x epoch: jaw-opening and video-tongue lead before the DAQ lick onset (rule 8 table) -> <tag>_jaw.csv."""
    g = Q.groupby(["animal", "epoch"], sort=False)
    J = pd.DataFrame({
        "n": g.size(),
        "frac_jaw_onset": g.jaw_state.apply(lambda s: (s == "onset").mean()),
        "frac_censored": g.jaw_state.apply(lambda s: (s == "censored").mean()),
        "frac_none": g.jaw_state.apply(lambda s: (s == "none").mean()),
        "frac_hidden_or_nodata": g.jaw_state.apply(lambda s: s.isin(["hidden", "no_data", "no_onset"]).mean()),
        "jaw_lead_med_ms": g.jaw_lead.median() * 1000, "jaw_lead_q25_ms": g.jaw_lead.quantile(0.25) * 1000,
        "jaw_lead_q75_ms": g.jaw_lead.quantile(0.75) * 1000,
        "tongue_lead_med_ms": g.tongue_lead.median() * 1000, "n_tongue": g.tongue_lead.count(),
        "peak_minus_onset_med_ms": g.apply(lambda h: np.nanmedian(h.t - h.t_onset) * 1000),
    }).reset_index()
    J["epoch"] = pd.Categorical(J.epoch, ["pre", *EP])
    J = J.sort_values(["animal", "epoch"])
    J.to_csv(fn("jaw.csv"), index=False)
    print("== JAW: cam4 jaw-opening onset before the DAQ lick onset, per animal x epoch (lead > 0 = jaw first)")
    print(J.round(3).to_string(index=False), flush=True)


def measures(S: pd.DataFrame) -> list[str]:
    return [c for c in S.columns if "|" in c]


def tests(S: pd.DataFrame, TCs: pd.DataFrame) -> None:
    import lick_sspn_checks as ssc
    rng = np.random.default_rng(0)
    out, anim = [], []
    for (src, sub, w), g in S.groupby(["source", "subset", "window"], sort=False):
        for c in measures(g):
            h = g.dropna(subset=[c])
            prd = {a: q[c].to_numpy() for a, q in h[h.epoch == "pre"].groupby("animal")}
            pm = h[h.epoch == "pre"].groupby("animal")[c].mean()
            for ep in EP:
                post = {a: q[c].to_numpy() for a, q in h[h.epoch == ep].groupby("animal") if a in prd}
                am = (h[h.epoch == ep].groupby("animal")[c].mean() - pm).dropna()
                for a, v in am.items():
                    anim.append({"source": src, "subset": sub, "window": w, "measure": c, "epoch": ep, "animal": a,
                                 "change": v})
                if not post:
                    continue
                iv = ak.boot_delta(post, prd, rng)
                if iv is None:
                    continue
                out.append({"source": src, "subset": sub, "window": w, "measure": c, "epoch": ep, "point": iv.point,
                            "lo": iv.lo, "hi": iv.hi, "n_animals": iv.n_animals,
                            "n_dir": int((np.sign(am) == np.sign(iv.point)).sum()), "pre_level": float(pm.mean()),
                            "n_ratio_nan": int(g[c].isna().sum()) if c.endswith("ratio") else 0})
    Tt, A = pd.DataFrame(out), pd.DataFrame(anim)
    Tt.to_csv(fn("tests.csv"), index=False)
    A.to_csv(fn("animals.csv"), index=False)
    # recovery tracking (within animal), per source x subset x window
    trk = []
    for (src, sub, w), g in S.groupby(["source", "subset", "window"], sort=False):
        cols = [c for c in measures(g) if c.split("|")[1] in ("readout", "total", "diff_all", "diff_SSp-n", "diff", "ratio")]
        t = ssc._track(g.assign(set="iti"), cols, f"{src}|{sub}|{w}")
        trk.append(t.assign(source=src, subset=sub, window=w))
    TR = pd.concat(trk, ignore_index=True)
    TR.to_csv(fn("track.csv"), index=False)
    # time course: change vs pre per bin (boot_delta_columns; sessions with a NaN bin dropped per measure)
    rows = []
    for (src, sub), g in TCs.groupby(["source", "subset"], sort=False):
        wins = sorted(g.window.unique(), key=lambda s: float(s[3:]))
        for c in measures(g):
            Wd = g.pivot_table(index=["label", "animal", "epoch"], columns="window", values=c).reindex(columns=wins)
            Wd = Wd.dropna(axis=1, how="all")
            wins_c = list(Wd.columns)
            Wd = Wd.dropna(axis=0, how="any").reset_index()
            if Wd.empty:
                continue
            pre = {a: q[wins_c].to_numpy(float) for a, q in Wd[Wd.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: q[wins_c].to_numpy(float) for a, q in Wd[Wd.epoch == ep].groupby("animal") if a in pre}
                if not post:
                    continue
                r = ak.boot_delta_columns(post, pre, np.random.default_rng(0))
                if r is None:
                    continue
                pt, lo_, hi_, _dr, na = r
                am = np.array([post[a].mean(0) - pre[a].mean(0) for a in sorted(post)])
                for k, w in enumerate(wins_c):
                    rows.append({"source": src, "subset": sub, "measure": c, "epoch": ep, "bin_lo": float(w[3:]),
                                 "bin_mid": float(w[3:]) + 0.025, "bin_last": round(float(w[3:]) + 0.05, 3),
                                 "point": pt[k], "lo": lo_[k], "hi": hi_[k],
                                 "n_animals": na, "n_dir": int((np.sign(am[:, k]) == np.sign(pt[k])).sum()),
                                 **{f"chg_{a}": v for a, v in zip(sorted(post), am[:, k])}})
    TT = pd.DataFrame(rows)
    TT.to_csv(fn("timecourse_tests.csv"), index=False)
    report()


# ------------------------------------------------------------------------------------------------------- report + plot
def report() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    R = pd.read_csv(fn("decode_r.csv"))
    A = pd.read_csv(fn("animals.csv"))
    Tt = pd.read_csv(fn("tests.csv"))
    TR = pd.read_csv(fn("track.csv"))
    TT = pd.read_csv(fn("timecourse_tests.csv"))
    if LIMIT is not None and fn("jaw.csv").exists():
        print("== JAW lead before the DAQ lick onset (ms), per animal x epoch")
        print(pd.read_csv(fn("jaw.csv")).round(3).to_string(index=False))
    print("== PRE held-out r per animal (rule 8)")
    print(R.pivot_table(index=["source", "subset", "decoder", "window"], columns="animal",
                        values=["r_lateral", "r_beyond_spout"], sort=False).round(2).to_string())
    key = ["frozen|readout", "frozen|diff_all", "frozen|diff_SSp-n", "window|readout", "bal_raw|diff", "bal_bs|diff",
           "bal_bs|ratio"]
    for sub in SUBSETS:
        for src in SRC:
            a = A[(A.source == src) & (A.subset == sub) & A.measure.isin(key)]
            print(f"== per-animal change vs pre ({src}, {sub})")
            print(a.pivot_table(index=["measure", "window"], columns=["epoch", "animal"], values="change", sort=False)
                  .reindex(columns=EP, level=0).round(3).to_string())
            t = Tt[(Tt.source == src) & (Tt.subset == sub) & Tt.measure.isin(key)]
            print(t[["measure", "window", "epoch", "point", "lo", "hi", "n_dir", "n_animals", "pre_level", "n_ratio_nan"]]
                  .round(3).to_string(index=False))
    print("== recovery tracking (per-animal r; p_within = MixedLM within-animal slope)")
    print(TR[TR.measure.isin(key)].round(3).to_string(index=False))
    s = TT[((TT.lo > 0) | (TT.hi < 0)) & TT.measure.isin(["frozen|readout", "frozen|diff_SSp-n", "bal_raw|diff",
                                                          "bal_bs|diff"])]
    if LIMIT is None:
        print("== time course: bins whose CI excludes 0 (frozen readout, SSp-n diff, balance diff)")
        print(s.groupby(["source", "subset", "measure", "epoch"]).bin_lo.apply(lambda v: list(np.round(v, 2))).to_string())
        return
    lim = tc_limit()
    s = s.assign(pre_lick=s.bin_last <= s.source.map(lim) + 1e-9)
    print(f"== time course: bins whose CI excludes 0, by their LAST sample; PRE-LICK = last sample <= {lim} "
          "(onset / peak time)")
    for flag, h in s.groupby("pre_lick"):
        print(f"  -- {'PRE-LICK (fully before the limit)' if flag else 'within the limit / after the lick'}")
        print(h.groupby(["source", "subset", "measure", "epoch"]).bin_last.apply(lambda v: list(np.round(v, 2)))
              .to_string())


def tc_limit() -> dict:
    """Per source: the latest sliding-bin sample that is >= LIMIT before the lick ONSET (peak-aligned: minus the median
    onset -> peak lag, rounded down to the 0.05 s grid)."""
    L = pd.read_csv(fn("licks.csv.gz"), usecols=["t", "t_onset"])
    q = float(np.nanmedian(L.t - L.t_onset))
    return {"onset": round(-LIMIT, 3), "peak": float(np.floor((-LIMIT - q) / 0.05 + 1e-9) * 0.05).__round__(3)}


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    TT = pd.read_csv(fn("timecourse_tests.csv"))
    Tt = pd.read_csv(fn("tests.csv"))
    R = pd.read_csv(fn("decode_r.csv"))
    v1 = LIMIT is None
    lim = None if v1 else tc_limit()
    tcm = [("frozen|readout", "frozen decoder: like-for-like offset (mm)"),
           ("frozen|diff_SSp-n", "  of which SSp-n left/right DIFFERENCE (mm)"),
           ("bal_raw|diff", "balance, raw level: contra - ipsi vs pre line"),
           ("bal_bs|diff", "balance, minus baseline: contra - ipsi vs pre line")]
    fig = plt.figure(figsize=(22, 15.5))
    gs = fig.add_gridspec(3, 20, left=0.035, right=0.99, top=0.9, bottom=0.135, hspace=0.42, wspace=3.0,
                          height_ratios=(1, 1, 1.1))
    for i, src in enumerate(("onset", "peak")):
        for j, (m, lab) in enumerate(tcm):
            ax = fig.add_subplot(gs[i, 5 * j:5 * j + 5])
            xcol = "bin_mid" if v1 or "bin_last" not in TT else "bin_last"
            for ep in EP:
                t = TT[(TT.source == src) & (TT.subset == "all") & (TT.measure == m) & (TT.epoch == ep)]
                if t.empty:
                    continue
                ax.plot(t[xcol], t.point, color=EPOCH_LINE[ep], lw=2, label=ep)
                ax.fill_between(t[xcol], t.lo, t.hi, color=EPOCH_LINE[ep], alpha=0.15, lw=0)
                sig = (t.lo > 0) | (t.hi < 0)
                ax.plot(t[xcol][sig], t.point[sig], "o", color=EPOCH_LINE[ep], ms=4)
            ax.axhline(0, color="0.6", lw=0.7)
            ax.axvline(0, color="k", lw=0.8)
            if v1:
                ax.axvspan(-0.15, 0, color="0.85", alpha=0.6, lw=0)
            else:
                ax.axvspan(lim[src] + 0.001, 0, color="0.8", alpha=0.6, lw=0)
            base = SRC[src][1]
            if m == "bal_bs|diff":
                ax.axvspan(*base, color="#cfe3f0", alpha=0.5, lw=0)
            ax.set_title(f"{src.upper()}-aligned: {lab}\nchange vs pre, 95% paired nested bootstrap (dots: CI excl. 0)",
                         fontsize=8.5)
            ax.set_xlabel(f"time from lick {src} (s; 0.1 s bins at their sample mean)" if v1 else
                          f"time from lick {src} (s; 0.1 s bin plotted at its LAST sample)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    # row 3: windows (onset-aligned, all licks of that alignment)
    wins = [w[0] for w in WINS]
    grp = [w[4] for w in WINS]
    x0 = np.arange(len(wins))

    def deco(ax):
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xticks(x0, [WLAB.get(w, w).replace("\n", " ") for w in wins], fontsize=7, rotation=90)
        for k in range(1, len(wins)):
            if grp[k] != grp[k - 1]:
                ax.axvline(k - 0.5, color="0.75", lw=0.8, ls="--")
        for k, g in enumerate(grp):
            if g in ("straddle", "post") or (v1 and wins[k] == "pre_020_000"):
                ax.axvspan(k - 0.5, k + 0.5, color="0.92", lw=0, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    ax = fig.add_subplot(gs[2, 0:4])
    rcol = "r_lateral" if v1 else "r_beyond_spout"
    for k, (dname, mk) in enumerate((("frozen", "o"), ("window", "s"))):
        r = R[(R.source == "onset") & (R.subset == "all") & (R.decoder == dname)]
        col = "#7b3294" if dname == "frozen" else "#2c7fb8"
        for a, h in r.groupby("animal"):
            h = h.set_index("window").reindex(wins)
            ax.plot(x0 + (k - 0.5) * 0.2, h[rcol], mk, color=col, mfc="none" if a in ("PS93", "PS95") else None, ms=5)
        mu = r.groupby("window")[rcol].mean().reindex(wins)
        ax.plot(x0 + (k - 0.5) * 0.2, mu, "-", color=col, lw=1.5, label=f"{dname} decoder (mean of animals)")
    deco(ax)
    ax.set_title(("PRE held-out r, decoded vs lateral landing" if v1 else "PRE held-out r BEYOND the spout\n(per-spout "
                  "pre means removed)") + ", onset-aligned\none marker per animal (open = PS93 / PS95)", fontsize=8.5)
    hd_dec, lb_dec = ax.get_legend_handles_labels()
    panels = ([("frozen|readout", "frozen offset (mm)"), ("frozen|diff_SSp-n", "SSp-n difference (mm)"),
               ("bal_bs|ratio", "balance gain ratio (log2; bs)")] if v1 else
              [("frozen|readout", "frozen offset (mm)"), ("frozen|diff_SSp-n", "SSp-n difference (mm)"),
               ("bal_raw|diff", "balance raw diff"), ("bal_bs|diff", "balance diff, minus baseline")])
    width = 16 // len(panels)
    for j, (m, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[2, 4 + j * width:4 + (j + 1) * width])
        for k, ep in enumerate(EP):
            t = Tt[(Tt.source == "onset") & (Tt.subset == "all") & (Tt.measure == m) & (Tt.epoch == ep)].set_index(
                "window").reindex(wins)
            x = x0 + (k - 1) * 0.22
            ax.errorbar(x, t.point, yerr=[t.point - t.lo, t.hi - t.point], fmt="o", color=EPOCH_LINE[ep], capsize=2,
                        ms=4, label=ep)
            for xx, (_w, r) in zip(x, t.iterrows()):
                if np.isfinite(r.point):
                    ax.text(xx, r.hi, f"{int(r.n_dir)}/{int(r.n_animals)}", ha="center", va="bottom", fontsize=5.5)
        deco(ax)
        ax.set_title(f"onset-aligned windows: {lab}\nchange vs pre (k/n = animals with the sign of the mean)", fontsize=8.5)
    if v1:
        sub = ("Rows 1-2: sliding 0.1 s bins (onset-aligned: bout-start subset, baseline -1.5..-1.0 s, blue; peak-aligned: "
               "all licks, baseline -1.0..-0.5 s); grey = last 0.15 s (tongue already moving). Row 3: fixed windows. "
               "PROVISIONAL")
    else:
        J = pd.read_csv(fn("jaw.csv")) if fn("jaw.csv").exists() else None
        jt = "" if J is None else (f"; cam4 jaw opening leads the DAQ lick onset by a median "
                                   f"{np.nanmin(J.jaw_lead_med_ms):.0f}-{np.nanmax(J.jaw_lead_med_ms):.0f} ms "
                                   "(animal x epoch)")
        sub = (f"WINDOW LIMIT {LIMIT:g} s: every PRE window ends >= {LIMIT:g} s before the lick ONSET (per lick; JAW windows: "
               f"before the cam4 jaw-opening onset){jt}. Grey columns = reference windows that straddle / follow onset.\n"
               f"Rows 1-2: sliding 0.1 s bins at their LAST sample; grey = bins holding data later than {LIMIT:g} s before "
               "onset (onset-aligned: bout-start subset, baseline -1.5..-1.0 s, blue; peak-aligned: all licks, baseline "
               "-1.0..-0.5 s). Imaging 31 Hz, interpolated: a sample carries up to 1 frame (32 ms) later. PROVISIONAL")
    fig.suptitle("BEFORE the lick: frozen pre-stroke lateral decoder (12.17) and hemispheric balance (ALM, tjM1, SSp-m, SSp-n, "
                 "SSs; contra = right) around SPONTANEOUS ITI licks (first lick after >= 2 s quiet). + = contralesional.\n"
                 + sub, fontsize=9.5)
    # one legend below the panels (the per-panel legends sat on the CI shading / markers)
    hd_ep, lb_ep = fig.axes[0].get_legend_handles_labels()
    fig.legend(list(hd_ep) + list(hd_dec), list(lb_ep) + [f"row 3 panel 1: {s}" for s in lb_dec],
               loc="lower center", ncol=len(lb_ep) + len(lb_dec), fontsize=9, frameon=False)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / FIGNAME, dpi=90)
    plt.close(fig)
    print(f"figure -> {FIG / FIGNAME}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--window-limit", default="0.1", help="s before lick onset every pre window must end by; 'none' = v1")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    _RUN["jobs"] = max(1, min(a.jobs, 4))
    configure(None if a.window_limit.lower() == "none" else float(a.window_limit))
    print(f"== window limit {LIMIT}; windows {[w[0] for w in WINS]}; outputs {fn('*.csv').name}; figure {FIGNAME}",
          flush=True)
    if not a.plot_only:
        compute()
    else:
        report()
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
