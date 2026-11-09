"""Is the PRE-CUE (ENL) representation of the TARGET spout preserved after the stroke while execution deviates ("the plan
looks the same, execution differs")? (10-29 follow-up of lick_enl_angle_decode.py / lick_enl_offset_explore.py.)

F12 CAVEAT (CLAUDE.md): the spout arrives ~3 s before the cue, so pre-cue position information cannot separate a sensory
spout response from a held intention. "Target" below = the spout that is physically there; whatever carries it.

SIGNS: + angle = toward + side_ml = CONTRALESIONAL (lick_enl_angle_decode.prepare); lateral axis ipsi -1 / mid 0 /
contra +1. Spout order everywhere = orofacial_cohort.POS_ORDER. pre = the August pre sessions (not config.early_pre).

PART A  FROZEN PRE-STROKE POSITION DECODER (the repo's one reference, rule 10), pre-cue window. Per-trial outputs REUSED,
        not refitted: <E>/attempt_coding_pp/trials_precue_<animal>.csv.gz (attempt_coding.run_animal, align precue = the
        LP ENL window 1.0 s ending at the cue, LocaNMF features, frozen `position` models (frozen-hit), LOSO for pre;
        generated 10-29 with WIDEFIELD_LICK_SOURCE=lp `attempt_coding --aligns precue --out <E>/attempt_coding_pp`, the
        same call as the 10-24 lick / cue run; pred / p_true identical to the older pp-less <E>/attempt_coding copy, which
        is the fallback). decoded = argmax of pp_<spout>, as exec_vs_neural_match; cue / lick read side by side.
   A0   CLASSIFIER-SPACE TRACKING (cls_tracking): off-target trials placed between "reads like a pre trial at the
        target" (0) and "reads like a pre trial at the landing spout" (1), argmax and probability versions.
        Executed bin = argmax of the trial's mean 3-D LANDING softmax (width 1; softmax_neural_vs_lick.lick_weights) and
        of the TRAJECTORY softmax (path_weights), exactly as exec_vs_neural_match (8.2). Trials: pre_hit / hit /
        incomplete with 3-D licks.
   A1   off-target trials (executed bin != target): share decoded as the TARGET vs as the EXECUTED bin, per animal x
        epoch, all targets and far_contra targets, windows precue / cue / lick.
   A2   CLASS-MATCHED (time in session): every far_contra-target trial that LANDS IPSILESIONALLY of far_contra (executed
        bin lateral < contra) is paired with the nearest on-target far_contra trial of the same session in trial order;
        per session mean of [decoded as far_contra], [decoded as the off trial's landing bin] (for the matched on-target
        trial: decoded as that same bin = its base rate), p_true; off - matched on, boot_delta (failed vs hit style,
        as precue_class_matched.py). Epochs pre (a LEVEL test, LOSO predictions; few deviated trials), acute,
        subacute, chronic and post (= the three post epochs pooled, never pre). Session rule >= 2 deviated and >= 2
        on-target trials (min_off 2, primary); min_off 1 = sensitivity (column min_off in both class-matched tables).
   A3   confusion matrix (rows target, cols decoded; per-session row shares, mean of sessions; canonical order) per
        animal x epoch; epoch - pre (softmax_neural_vs_lick.boot_change, Bonferroni / 36); far_contra mean decoded
        lateral; accuracy per spout and overall per session -> boot_delta.
PART B  OWN 6-WAY CLASSIFIER (secondary check, area features): multinomial logistic regression (StandardScaler +
        LogisticRegression C 0.5, as locanmf_frozen_decoder._pipe) on the ENL area features of
        lick_enl_angle_decode (its extraction + `select`, imported unchanged), trained on all pre trials per animal,
        LOSO for pre, FROZEN via frozen_models.load_or_fit kind `enl_spout` (align enl_w<W>_lf<lf>). Same A1-A3 read-outs
        with probabilities; landing additionally = nearest pre spout mean angle WITHIN the target's distance row (the
        first-lick angle does not separate close from far).
PART C  GAIN-NORMALIZED TRACKING of the frozen ridge angle read-out (lick_enl_angle_decode.frozen_fits, kind
        enl_angle, loaded not refitted; pre = its LOSO predictions). Per animal the pre held-out GAIN g = slope of the
        six spout-mean decoded angles on the six spout-mean actual angles (ACROSS spouts; variant: the WITHIN session x
        spout slope). Per session x spout (>= MIN_CELL trials): mean decoded / g and mean actual angle. Per animal x
        spout x epoch: tracking = delta(decoded / g) / delta(actual), deltas vs pre; 0 = the read-out stays at the
        target's pre value, 1 = it moves as far as the executed angle would move it. CI: boot_delta_columns on the two
        columns (one draw for both, so the ratio of the draws is a valid bootstrap of the ratio). Pooled over animals
        (far_contra) = ratio of animal-mean deltas; pooled over spouts = least-squares projection
        sum_s dA_s dD_s / sum_s dA_s^2 on sessions holding all six spouts. A ratio is flagged ill-conditioned when the
        CI of delta(actual) spans 0 or |delta(actual)| < 5 deg.
WINDOWS / SUBSETS: W = 1.0 s ENL (primary) and 0.5 s (B, C), lick-free rule "slide" (Priya 11-04 ~20:00: the lick
        ENL family SLIDES like the imaging ENL decoders; `lick_enl_angle_decode` "slide", new frozen ids
        enl_w<W>_lfslide); `--lf 2s` reruns the pre-11-05 fixed window, no lick in the last 2 s. SETTLED = the spout strobe (firmware
        pulses it AFTER the move completes, daq_trials docstring; per trial from il._behaviour(label).trials.strobe_s, the
        last strobe before the cue) is >= 2 s before the ENL window opens: cue - strobe >= W + 2 s (W = 1.0 -> 3.0 s for
        A, B, C at W 1; 2.5 s at W 0.5). The trials are scored by the same frozen models (not refitted on the subset).
        The frozen decoder's pre-cue window may slide earlier when a lick sits in the last 1 s (precue_window_start), so
        for A the subset is approximate; so is it for B / C under "slide" (the subset is judged on the unslid window).
DECODABILITY (D): per session accuracy (A, B) and ridge raw r (C) -> change vs pre (boot_delta).

Outputs <OUT_W>/enl_target/: timing.csv.gz (label, trial_id, cue_s, strobe_s), ridge_trials.csv.gz, cls_trials.csv.gz,
precue_trials.csv.gz, cls_tracking.csv, tracking.csv, tracking_pooled.csv, tracking_shrink.csv, gains.csv, offtarget.csv, classmatched_sessions.csv,
classmatched.csv, confusion.csv, confusion_change.csv, lateral.csv, accuracy_sessions.csv, decodability_tests.csv;
figures <OUT_W>/figures/lick_enl_target_{tracking,confusion,offtarget,classmatched_epochs,decodability}.png.
Usage: python scripts/lick_enl_target_test.py [--jobs 8] [--lf slide|2s] [--plot-only | --classmatched-only]
--classmatched-only recomputes classmatched_sessions.csv / classmatched.csv from the saved precue_trials.csv.gz +
cls_trials.csv.gz (frozen-model outputs; nothing refitted) and redraws; --plot-only redraws every figure.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_enl_angle_decode as ead  # noqa: E402
from wfield_local.orofacial_outputs import encoding_root as _encoding_root  # noqa: E402 - rebuild root (11-03)
from wfield_local.attempt_coding import trials_name as attempt_trials_name  # noqa: E402

#: a non-primary lick-free rule (`--lf 2s`, the pre-11-05 fixed window) writes beside the primary, never over it
_LF_ARG = sys.argv[sys.argv.index("--lf") + 1] if "--lf" in sys.argv[:-1] else None
_LF_SFX = f"_lf{_LF_ARG}" if _LF_ARG and _LF_ARG != ead.LF_PRIMARY else ""
OUT = il.OUT_W / f"enl_target{_LF_SFX}"
FIG = il.OUT_W / f"figures{_LF_SFX}"
E = _encoding_root()


def _attempt_dir():
    """attempt_coding output folder, resolved by its WRITER (review N1 / N10): never a literal path."""
    from wfield_local import attempt_coding
    return attempt_coding.out_dir()


SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
LAB = ["cI", "cM", "cC", "fI", "fM", "fC"]
LAT = {s: {"ipsi": -1.0, "mid": 0.0, "contra": 1.0}[s.split("_")[1]] for s in SP}
AN, EP, POST = ead.AN, ead.EP, ead.EP[1:]
#: (window, lick-free rule): "slide" since 11-05 (`ead.LF_PRIMARY`); `--lf 2s` = the pre-11-05 fixed-window run
VARIANTS = ((1.0, ead.LF_PRIMARY), (0.5, ead.LF_PRIMARY))
SETTLE_GAP = 2.0
MIN_CELL = 3
MIN_OFF = 5
CM_MIN_OFF = (2, 1)  # class-matched session rule: >= this many deviated trials (primary 2; 1 = sensitivity), >= 2 on
SEED = 0


def _check_order():
    from wfield_local import orofacial_cohort as oc
    if list(oc.POS_ORDER) != SP:
        raise RuntimeError(f"POS_ORDER changed: {oc.POS_ORDER}")


# ------------------------------------------------------------------------------------------------------------ TIMING
def session_timing(label: str) -> pd.DataFrame:
    """One session (module-level for spawn): per DAQ trial, trial_id, cue_s and the spout strobe before it."""
    T = il._behaviour(label).trials
    return pd.DataFrame({"label": label, "trial_id": T.trial_id.astype(int).to_numpy(),
                         "cue_s": T.cue_s.to_numpy(float), "strobe_s": T.strobe_s.to_numpy(float)})


def timing(labels: list, jobs: int) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    f = OUT / "timing.csv.gz"
    if f.exists():
        T = pd.read_csv(f)
        if set(labels) <= set(T.label):
            return T
    res, fails = ak.fan_sessions(sorted(labels), session_timing, jobs=jobs)
    if fails:
        print(f"!! timing failed for {len(fails)} sessions: {[x for x, *_ in fails][:8]}", flush=True)
    T = pd.concat([v for _x, v in res], ignore_index=True)
    T.to_csv(f, index=False, float_format="%.6f")
    return T


# ------------------------------------------------------------------------------------------------------------ PART A
def precue_trials(TM: pd.DataFrame) -> dict:
    """{window: trial table} for the frozen position decoder: label, animal, epoch, trial_id, spout, cls, dec, p_true,
    exec_land, exec_traj, settle_s. precue from attempt_coding/, cue + lick from attempt_coding_pp/ (argmax pp_)."""
    from softmax_neural_vs_lick import lick_weights, path_weights
    LW, PW = lick_weights(1.0), path_weights(1.0)
    LW["exec_land"] = np.array(SP)[LW[[f"lw_{p}" for p in SP]].to_numpy().argmax(1)]
    LW = LW[["label", "trial_id", "exec_land"]]
    if PW is not None:
        PW["exec_traj"] = np.array(SP)[PW[[f"tw_{p}" for p in SP]].to_numpy().argmax(1)]
        PW = PW[["label", "trial_id", "exec_traj"]]
    out = {}
    from wfield_local import config
    from wfield_local import run_mode
    PP = run_mode.subdir(E / "attempt_coding_pp", lp_family=True)       # mode-routed, as attempt_coding.out_dir (N1)
    pq = PP if all((PP / attempt_trials_name("precue", an)).exists() for an in AN) \
        else _attempt_dir()
    print(f"== precue frozen-decoder trials from {pq}", flush=True)
    for w, d in (("precue", pq), ("cue", PP), ("lick", PP)):
        T = pd.concat([pd.read_csv(d / attempt_trials_name(w, an)) for an in AN], ignore_index=True)
        T = T[(T.method == "dom") & T.cls.isin(["pre_hit", "hit", "incomplete"]) & (~T.label.map(config.early_pre))]
        if "pp_far_contra" in T:
            T["dec"] = np.array(SP)[T[[f"pp_{p}" for p in SP]].to_numpy().argmax(1)]
        else:
            T["dec"] = T.pred.astype(str)
        T = T[["label", "animal", "epoch", "trial_id", "spout", "cls", "dec", "p_true"] +
              [f"pp_{q}" for q in SP if f"pp_{q}" in T]]
        T = T.merge(LW, on=["label", "trial_id"], how="left")
        T = T.merge(PW, on=["label", "trial_id"], how="left") if PW is not None else T.assign(exec_traj=np.nan)
        T = T.merge(TM[["label", "trial_id", "cue_s", "strobe_s"]], on=["label", "trial_id"], how="left")
        T["settle_s"] = T.cue_s - T.strobe_s
        T["window"] = w
        out[w] = T.reset_index(drop=True)
    return out


# ------------------------------------------------------------------------------------------------------------ PART B
def _pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def frozen_cls(a: str, W: float, lf: str, X, y, sess) -> dict:
    """RULE 10: the animal's all-pre ENL spout classifier and its LOSO siblings, through frozen_models.load_or_fit."""
    from wfield_local import frozen_models as fm
    labs = sorted(set(np.asarray(sess, str)))
    dig = ead._digest(X, y)
    spec = fm.make_spec(a, "enl_spout", align=f"enl_w{W:g}_lf{lf}", source=f"areas_split_r{il.ROI_VERSION}",
                        train_labels=labs, post_s=W, n_features=X.shape[1],
                        train_data={"xy_sha1": dig, "sess": np.asarray(sess, str)})  # training-data fingerprint (FREEZE_VERSION 2, rebuild 11-03)

    def fit():
        return {"full": _pipe().fit(X, y), "loso": {s: _pipe().fit(X[sess != s], y[sess != s]) for s in labs},
                "xy_sha1": dig}
    P, status = fm.load_or_fit(spec, fit, meta={"script": "lick_enl_target_test.py", "n_trials": int(len(y))})
    print(f"[frozen] {a} enl_spout W={W:g} lf={lf}: {status}", flush=True)
    if P.get("xy_sha1") != dig:
        print(f"[frozen] !! {a} enl_spout W={W:g}: stored model trained on different trial data; stored one used", flush=True)
    return P


def model_trials(T: pd.DataFrame, names: list, W: float, lf: str) -> pd.DataFrame:
    """Per selected ENL trial: frozen ridge angle prediction (pred) and frozen spout-classifier probabilities."""
    S = ead.select(T, W, lf)
    fc = ead.fcols(W, lf, names)
    out = []
    for a in AN:
        A = S[S.animal == a].reset_index(drop=True)
        Pr, Po = A[A.epoch == "pre"].reset_index(drop=True), A[A.epoch != "pre"].reset_index(drop=True)
        if Pr.label.nunique() < 2 or len(Pr) < 30:
            continue
        X, y = Pr[fc].to_numpy(float), Pr.angle.to_numpy(float)
        sess, pos = Pr.label.to_numpy(str), Pr.pos.to_numpy(str)
        cell = np.char.add(np.char.add(sess, "|"), pos)
        FZ = ead.frozen_fits(a, W, lf, X, y, sess, cell)
        Pr["pred"] = ead.loso(X, y, sess, cell, pos, fits=FZ["loso"])["raw"]
        Po["pred"] = ead._pred(FZ["raw"], Po[fc].to_numpy(float)) if len(Po) else []
        C = frozen_cls(a, W, lf, X, pos, sess)
        pp_pre = np.full((len(Pr), 6), np.nan)
        for s in np.unique(sess):
            m = sess == s
            M = C["loso"][s]
            pp_pre[m] = M.predict_proba(X[m])[:, [list(M.classes_).index(q) for q in SP]]
        M = C["full"]
        pp_post = M.predict_proba(Po[fc].to_numpy(float))[:, [list(M.classes_).index(q) for q in SP]] if len(Po) else \
            np.zeros((0, 6))
        B = pd.concat([Pr, Po], ignore_index=True)
        B[[f"pp_{q}" for q in SP]] = np.vstack([pp_pre, pp_post])
        out.append(B[["label", "animal", "epoch", "pos", "cue", "angle", "pred"] + [f"pp_{q}" for q in SP]])
    R = pd.concat(out, ignore_index=True)
    R["window_s"] = W
    R["lickfree"] = lf
    return R


def attach(R: pd.DataFrame, TM: pd.DataFrame) -> pd.DataFrame:
    """trial_id + strobe (exact cue match), 3-D landing / trajectory bins, angle landing within the target's row."""
    from softmax_neural_vs_lick import lick_weights, path_weights
    # nearest DAQ cue within 0.01 s (the ENL trial table stores cue with 6 significant digits)
    k = TM[["label", "cue_s", "trial_id", "strobe_s"]].sort_values("cue_s")
    R = pd.merge_asof(R.reset_index().sort_values("cue"), k, left_on="cue", right_on="cue_s", by="label",
                      direction="nearest", tolerance=0.01).sort_values("index").drop(columns=["index", "cue_s"])
    R = R.reset_index(drop=True)
    miss = R.trial_id.isna().mean()
    if miss > 0.01:
        print(f"!! {miss:.3f} of ENL trials without a timing match", flush=True)
    R["settle_s"] = R.cue - R.strobe_s
    LW, PW = lick_weights(1.0), path_weights(1.0)
    LW["exec_land"] = np.array(SP)[LW[[f"lw_{p}" for p in SP]].to_numpy().argmax(1)]
    R = R.merge(LW[["label", "trial_id", "exec_land"]], on=["label", "trial_id"], how="left")
    if PW is not None:
        PW["exec_traj"] = np.array(SP)[PW[[f"tw_{p}" for p in SP]].to_numpy().argmax(1)]
        R = R.merge(PW[["label", "trial_id", "exec_traj"]], on=["label", "trial_id"], how="left")
    R["exec_ang"] = ""
    for a, g in R.groupby("animal"):
        mu = g[g.epoch == "pre"].groupby("pos").angle.mean()
        for row in ("close", "far"):
            q = [s for s in SP if s.startswith(row)]
            m = (R.animal == a) & R.pos.str.startswith(row)
            d = np.abs(R.loc[m, "angle"].to_numpy(float)[:, None] - mu.reindex(q).to_numpy(float)[None])
            R.loc[m, "exec_ang"] = np.array(q)[d.argmin(1)]
    R["dec"] = np.array(SP)[R[[f"pp_{q}" for q in SP]].to_numpy().argmax(1)]
    R["p_true"] = R[[f"pp_{q}" for q in SP]].to_numpy()[np.arange(len(R)), [SP.index(p) for p in R.pos]]
    return R.rename(columns={"pos": "spout"})


# ------------------------------------------------------------------------------------------------------------ READ-OUTS
def subsets(T: pd.DataFrame, W: float):
    yield "all", T
    yield "settled", T[T.settle_s >= W + SETTLE_GAP]


def offtarget(T: pd.DataFrame, tag: dict, exec_cols=("exec_land", "exec_traj")) -> list:
    rows = []
    for ex in exec_cols:
        if ex not in T:
            continue
        J = T[T[ex].isin(SP)]
        for tgt in ("all", "far_contra"):
            g0 = J if tgt == "all" else J[J.spout == "far_contra"]
            for (a, ep), g in g0.groupby(["animal", "epoch"]):
                off, on = g[g[ex] != g.spout], g[g[ex] == g.spout]
                r = {**tag, "exec": ex, "target": tgt, "animal": a, "epoch": ep, "n_on": len(on), "n_off": len(off),
                     "on_decoded_target": float((on.dec == on.spout).mean()) if len(on) else np.nan,
                     "off_decoded_target": float((off.dec == off.spout).mean()) if len(off) else np.nan,
                     "off_decoded_executed": float((off.dec == off[ex]).mean()) if len(off) else np.nan}
                if "pp_far_contra" in g and len(off):
                    P = off[[f"pp_{q}" for q in SP]].to_numpy()
                    r["off_p_target"] = float(P[np.arange(len(off)), [SP.index(s) for s in off.spout]].mean())
                    r["off_p_executed"] = float(P[np.arange(len(off)), [SP.index(s) for s in off[ex]]].mean())
                rows.append(r)
    return rows


def class_matched(T: pd.DataFrame, ex: str, min_off: int = 2) -> pd.DataFrame:
    """far_contra-target trials landing ipsilesionally vs the nearest on-target trial (trial order) per session, every
    epoch INCLUDING pre (10-29: pre is the reference level; pre deviated far_contra trials are few). A session enters
    with >= min_off deviated and >= 2 on-target trials (primary min_off 2; 1 = sensitivity, CM_MIN_OFF)."""
    J = T[(T.spout == "far_contra") & T[ex].isin(SP) & T.trial_id.notna()]
    rows = []
    for (a, ep, lab), g in J.groupby(["animal", "epoch", "label"]):
        lat = g[ex].map(LAT)
        off, on = g[lat < 1].sort_values("trial_id"), g[g[ex] == "far_contra"].sort_values("trial_id")
        if len(off) < min_off or len(on) < 2:
            continue
        j = np.abs(off.trial_id.to_numpy()[:, None] - on.trial_id.to_numpy()[None]).argmin(1)
        m = on.iloc[j]
        r = {"animal": a, "epoch": ep, "label": lab, "n_off": len(off), "n_on": len(on),
             "off_dec_target": float((off.dec == "far_contra").mean()), "on_dec_target": float((m.dec == "far_contra").mean()),
             "off_dec_landing": float((off.dec.to_numpy() == off[ex].to_numpy()).mean()),
             "on_dec_landing": float((m.dec.to_numpy() == off[ex].to_numpy()).mean()),
             "off_p_true": float(off.p_true.mean()), "on_p_true": float(m.p_true.mean()),
             "off_lat": float(off.dec.map(LAT).mean()), "on_lat": float(m.dec.map(LAT).mean())}
        if "pp_far_contra" in g:
            P = off[[f"pp_{q}" for q in SP]].to_numpy(); Q = m[[f"pp_{q}" for q in SP]].to_numpy()
            jj = [SP.index(s) for s in off[ex]]
            r["off_p_landing"] = float(P[np.arange(len(off)), jj].mean())
            r["on_p_landing"] = float(Q[np.arange(len(off)), jj].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def shrink_decomposition(TRK: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    """SENSITIVITY: a weaker read-out regresses every spout's decoded angle toward the mean, which at far_contra (the most
    + decoded spout) looks like ipsilesional 'tracking'. Per animal x epoch, OLS across the six spouts:
    delta(decoded / g)_s = c + shrink * (pre decoded_s - mean over spouts) / g + track * delta(actual)_s.
    6 points, 3 parameters, and delta(actual) is largest where the pre offset is largest (far_contra), so `track` is
    poorly determined: read it as a sensitivity, not a test."""
    rows = []
    for (W, sub, ep), t in TRK[TRK.gain == "gain_across"].groupby(["window_s", "subset", "epoch"]):
        for a in AN:
            x = t[t.animal == a].set_index("spout").reindex(SP)
            g = G[(G.window_s == W) & (G.animal == a)]
            ok = x.d_actual.notna().to_numpy()
            if ok.sum() < 5 or not len(g):
                continue
            g = g.iloc[0]
            off = np.array([g[f"pred_{q}"] for q in SP], float)
            off = (off - off.mean()) / g.gain_across
            X = np.column_stack([np.ones(ok.sum()), off[ok], x.d_actual.to_numpy(float)[ok]])
            b = np.linalg.lstsq(X, x.d_decoded_over_gain.to_numpy(float)[ok], rcond=None)[0]
            rows.append({"window_s": W, "subset": sub, "epoch": ep, "animal": a, "n_spouts": int(ok.sum()),
                         "shrink": float(b[1]), "track": float(b[2]),
                         "r_offset_dactual": float(np.corrcoef(off[ok], x.d_actual.to_numpy(float)[ok])[0, 1])})
    return pd.DataFrame(rows)


def cls_tracking(T: pd.DataFrame, ex: str, tag: dict) -> list:
    """Classifier-space TRACKING of off-target post-stroke trials (the analog of PART C). Per off-target trial i with
    target t_i and executed bin q_i, against the animal's PRE confusion (LOSO; P_pre(c | target r) = mean over pre
    trials targeted at r of [decoded c] (argmax) or of pp_c (prob)):
      exec  = (x_i(q_i) - P_pre(q_i | t_i)) / (P_pre(q_i | q_i) - P_pre(q_i | t_i))   read as the landing bin
      targ  = (P_pre(t_i | t_i) - x_i(t_i)) / (P_pre(t_i | t_i) - P_pre(t_i | q_i))   loss of the target reading
    0 = the trial reads like a pre trial at its TARGET, 1 = like a pre trial at the spout it LANDED at. Numerator and
    denominator are averaged over trials within session, then boot_delta_columns (post = sessions, pre = a zero row:
    sessions resampled within animals, animals resampled; the PRE reference is a fixed point, its uncertainty is not
    propagated); ratio of the aligned draws."""
    from wfield_local import analysis_kit as ak
    rows = []
    for kind in ("argmax", "prob"):
        if kind == "prob" and "pp_far_contra" not in T:
            continue
        X = T[T[ex].isin(SP)].copy()
        if kind == "argmax":
            for q in SP:
                X[f"x_{q}"] = (X.dec == q).astype(float)
        else:
            for q in SP:
                X[f"x_{q}"] = X[f"pp_{q}"]
        for tgt in ("far_contra", "all"):
            num_e = {}
            for a, g in X.groupby("animal"):
                pre = g[g.epoch == "pre"]
                Pp = pre.groupby("spout")[[f"x_{q}" for q in SP]].mean().reindex(SP)
                Pp.columns = SP
                po = g[(g.epoch != "pre") & (g[ex] != g.spout)]
                if tgt == "far_contra":
                    po = po[(po.spout == "far_contra") & (po[ex].map(LAT) < 1)]
                if not len(po):
                    continue
                t, q = po.spout.to_numpy(str), po[ex].to_numpy(str)
                xq = po[[f"x_{c}" for c in SP]].to_numpy()[np.arange(len(po)), [SP.index(c) for c in q]]
                xt = po[[f"x_{c}" for c in SP]].to_numpy()[np.arange(len(po)), [SP.index(c) for c in t]]
                P = Pp.to_numpy(float)
                ti, qi = np.array([SP.index(c) for c in t]), np.array([SP.index(c) for c in q])
                D = pd.DataFrame({"label": po.label.to_numpy(), "epoch": po.epoch.to_numpy(),
                                  "en": xq - P[ti, qi], "ed": P[qi, qi] - P[ti, qi],
                                  "tn": P[ti, ti] - xt, "td": P[ti, ti] - P[qi, ti]})
                for ep, h in D.groupby("epoch"):
                    s = h.groupby("label")[["en", "ed", "tn", "td"]].mean()
                    num_e.setdefault(ep, {})[a] = s.to_numpy(float)
            for ep, dd in num_e.items():
                for who in sorted(dd) + ["pooled"]:
                    po = dd if who == "pooled" else {who: dd[who]}
                    zero = {a: np.zeros((1, 4)) for a in po}
                    res = ak.boot_delta_columns(po, zero, np.random.default_rng(SEED))
                    if res is None:
                        continue
                    obs, _l, _h, dr, n = res
                    r = {**tag, "exec": ex, "kind": kind, "target": tgt, "epoch": ep, "animal": who, "n_animals": n,
                         "n_sessions": sum(len(v) for v in po.values()),
                         "n_trials_note": "session means", "exec_den": float(obs[1]), "targ_den": float(obs[3])}
                    for nm, i, j in (("exec_tracking", 0, 1), ("target_loss", 2, 3)):
                        rr = dr[:, i] / dr[:, j]
                        r[nm] = float(obs[i] / obs[j])
                        r[f"{nm}_lo"], r[f"{nm}_hi"] = (float(v) for v in np.percentile(rr, [2.5, 97.5]))
                    rows.append(r)
    return rows


def cm_tests(CS: pd.DataFrame, tag: dict) -> list:
    from wfield_local import analysis_kit as ak
    rows = []
    # pre (added 10-29) draws from its OWN stream so the post-epoch CIs keep the draws they were published with
    rng, rng_pre = np.random.default_rng(SEED), np.random.default_rng([SEED, 1])
    for key in ("dec_target", "dec_landing", "p_true", "lat", "p_landing"):
        if f"off_{key}" not in CS:
            continue
        for ep in EP + ["post"]:
            # 'post' pools the three post epochs only (pre is its own LEVEL row, never pooled into post)
            g = CS[CS.epoch != "pre"] if ep == "post" else CS[CS.epoch == ep]
            off = {a: h[f"off_{key}"].dropna().tolist() for a, h in g.groupby("animal")}
            on = {a: h[f"on_{key}"].dropna().tolist() for a, h in g.groupby("animal")}
            ci = ak.boot_delta(off, on, rng_pre if ep == "pre" else rng)
            per = {a: float(np.mean(off[a]) - np.mean(on[a])) for a in sorted(off) if off[a] and on[a]}
            rows.append({**tag, "value": key, "epoch": ep, "off_minus_on": None if ci is None else ci.point,
                         "lo": None if ci is None else ci.lo, "hi": None if ci is None else ci.hi,
                         "n_animals": len(per), "n_neg": int(sum(v < 0 for v in per.values())),
                         **{f"n_off_{a}": int(g.loc[g.animal == a, "n_off"].sum()) for a in AN},
                         **{f"n_sess_{a}": int((g.animal == a).sum()) for a in AN},
                         **{f"d_{a}": per.get(a, np.nan) for a in AN},
                         **{f"off_{a}": float(np.mean(off[a])) if off.get(a) else np.nan for a in AN},
                         **{f"on_{a}": float(np.mean(on[a])) if on.get(a) else np.nan for a in AN}})
    return rows


def classmatched_all(PT: dict, RT: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A2 for every source / window / subset / exec / session rule. PT = {window: frozen position-decoder trial table}
    (precue_trials), RT = {W: own-classifier trial table} (cls_trials). Read-only on the models: both tables already hold
    the frozen decoders' outputs. Returns (classmatched_sessions, classmatched)."""
    jobs = []
    for w, D in PT.items():
        for sub, X in (subsets(D, 1.0) if w == "precue" else (("all", D),)):
            for ex in ("exec_land", "exec_traj"):
                jobs.append(({"source": "frozen_position", "window": w, "subset": sub}, X, ex))
    for W, R in RT.items():
        for sub, X in subsets(R, W):
            for ex in ("exec_land", "exec_ang"):
                jobs.append(({"source": "enl_spout", "window": f"{W:.1f}", "subset": sub}, X, ex))
    CMs, CMt = [], []
    for tag, X, ex in jobs:
        if ex not in X:
            continue
        for mo in CM_MIN_OFF:
            cs = class_matched(X, ex, mo)
            if len(cs):
                t = {**tag, "exec": ex, "min_off": mo}
                CMs.append(cs.assign(**t))
                CMt += cm_tests(cs, t)
    return pd.concat(CMs, ignore_index=True), pd.DataFrame(CMt)


def classmatched_only() -> None:
    """--classmatched-only: recompute classmatched_sessions.csv / classmatched.csv from the saved trial tables
    (precue_trials.csv.gz, cls_trials.csv.gz; frozen-model outputs, nothing refitted) and redraw. Without cls_trials the
    existing enl_spout rows are kept and only the frozen_position rows replaced."""
    P = pd.read_csv(OUT / "precue_trials.csv.gz")
    PT = {w: P[P.window == w].reset_index(drop=True) for w in ("precue", "cue", "lick")}
    RT = {}
    if (OUT / "cls_trials.csv.gz").exists():
        C = pd.read_csv(OUT / "cls_trials.csv.gz")
        RT = {float(W): g.reset_index(drop=True) for W, g in C.groupby("window_s")}
    CMs, CMt = classmatched_all(PT, RT)
    if not RT:
        print("!! cls_trials.csv.gz missing: enl_spout class-matched rows KEPT from the previous run", flush=True)
        def keep(new: pd.DataFrame, f: str) -> pd.DataFrame:
            old = pd.read_csv(OUT / f)
            return pd.concat([new, old[old.source != "frozen_position"]], ignore_index=True)
        CMs, CMt = keep(CMs, "classmatched_sessions.csv"), keep(CMt, "classmatched.csv")
    CMs.to_csv(OUT / "classmatched_sessions.csv", index=False)
    CMt.to_csv(OUT / "classmatched.csv", index=False)
    t = CMt[(CMt.value.isin(["dec_target", "dec_landing"])) & (CMt.exec == "exec_land") & (CMt.subset == "all")]
    cols = ["source", "window", "min_off", "value", "epoch", "off_minus_on", "lo", "hi", "n_animals", "n_neg"] + \
        [f"n_off_{a}" for a in AN]
    print("\n== CLASS-MATCHED (all trials, exec_land)\n" + t[cols].round(3).to_string(index=False))
    figures()


def confusion(T: pd.DataFrame, tag: dict, prob: bool) -> tuple[list, list]:
    """Session matrices rows = target, cols = decoded (one-hot argmax, or mean probabilities when prob)."""
    from softmax_neural_vs_lick import boot_change, session_matrices
    T = T.copy()
    if not prob:
        for q in SP:
            T[f"oh_{q}"] = (T.dec == q).astype(float)
    S = session_matrices(T, "pp_" if prob else "oh_")
    lev, chg = [], []
    for (a, ep), Ms in S.items():
        M = np.nanmean(Ms, 0)
        lev += [{**tag, "animal": a, "epoch": ep, "target": SP[r], "decoded": SP[c], "share": M[r, c],
                 "n_sessions": len(Ms)} for r in range(6) for c in range(6)]
    for ep in POST:
        if not any((a, ep) in S for a in AN):
            continue
        obs, sig, ag, n = boot_change(S, ep)
        chg += [{**tag, "epoch": ep, "target": SP[r], "decoded": SP[c], "change": obs[r, c], "bonf_sig": bool(sig[r, c]),
                 "n_same_sign": int(ag[r, c]), "n_animals": n} for r in range(6) for c in range(6)]
    return lev, chg


def accuracy_sessions(T: pd.DataFrame, tag: dict, ridge: bool) -> list:
    rows = []
    for (a, ep, lab), g in T.groupby(["animal", "epoch", "label"]):
        r = {**tag, "animal": a, "epoch": ep, "label": lab, "n": len(g), "acc": float((g.dec == g.spout).mean()),
             "p_true": float(g.p_true.mean())}
        fc = g[g.spout == "far_contra"]
        r["fc_lat"] = float(fc.dec.map(LAT).mean()) if len(fc) >= MIN_CELL else np.nan
        if "pp_far_contra" in g and len(fc) >= MIN_CELL:
            r["fc_lat_p"] = float((fc[[f"pp_{q}" for q in SP]].to_numpy() @ np.array([LAT[q] for q in SP])).mean())
        r.update({f"acc_{q}": float((g[g.spout == q].dec == q).mean()) if (g.spout == q).sum() >= MIN_CELL else np.nan
                  for q in SP})
        if ridge and len(g) >= ead.MIN_SESS:
            r["r_raw"] = ead._r(g.pred.to_numpy(float), g.angle.to_numpy(float))
        rows.append(r)
    return rows


def decod_tests(A: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(SEED)
    rows = []
    vals = [v for v in ("acc", "p_true", "fc_lat", "fc_lat_p", "r_raw") + tuple(f"acc_{q}" for q in SP) if v in A]
    keys = [k for k in ("source", "window", "subset") if k in A]
    for kk, g in A.groupby(keys, sort=True):
        for v in vals:
            pre = {a: h[v].dropna().tolist() for a, h in g[g.epoch == "pre"].groupby("animal")}
            pre = {a: x for a, x in pre.items() if x}
            for ep in POST:
                post = {a: h[v].dropna().tolist() for a, h in g[g.epoch == ep].groupby("animal")}
                post = {a: x for a, x in post.items() if x}
                if not (set(pre) & set(post)):
                    continue
                ci = ak.boot_delta(post, pre, rng)
                per = {a: float(np.mean(post[a]) - np.mean(pre[a])) for a in sorted(set(pre) & set(post))}
                rows.append({**dict(zip(keys, kk)), "value": v, "epoch": ep, "delta": None if ci is None else ci.point,
                             "lo": None if ci is None else ci.lo, "hi": None if ci is None else ci.hi,
                             "n_animals": len(per), "n_neg": int(sum(x < 0 for x in per.values())),
                             **{f"pre_{a}": float(np.mean(pre[a])) if a in pre else np.nan for a in AN},
                             **{f"d_{a}": per.get(a, np.nan) for a in AN}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------ PART C
def gains(R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, g in R[R.epoch == "pre"].groupby("animal"):
        m = g.groupby("spout")[["pred", "angle"]].mean().reindex(SP)
        g_across = float(np.polyfit(m.angle, m.pred, 1)[0])
        cell = (g.label + "|" + g.spout).to_numpy(str)
        x, y = ead._resid(g.angle.to_numpy(float), cell), ead._resid(g.pred.to_numpy(float), cell)
        rows.append({"animal": a, "gain_across": g_across, "gain_within": float(x @ y / (x @ x)), "n_pre": len(g),
                     "spread_pred": float(m.pred.max() - m.pred.min()), "spread_angle": float(m.angle.max() - m.angle.min()),
                     **{f"pred_{q}": m.pred[q] for q in SP}, **{f"angle_{q}": m.angle[q] for q in SP}})
    return pd.DataFrame(rows)


def _cells(R: pd.DataFrame, a: str, ep: str, g: float) -> pd.DataFrame:
    """session x spout means: decoded / gain and actual angle (>= MIN_CELL trials)."""
    x = R[(R.animal == a) & (R.epoch == ep)]
    c = x.groupby(["label", "spout"]).agg(n=("angle", "size"), d=("pred", "mean"), ang=("angle", "mean")).reset_index()
    c = c[c.n >= MIN_CELL]
    c["d"] = c.d / g
    return c


def _ratio(obs, draws, i_d, i_a):
    """tracking point + CI from aligned draws; flags."""
    t = obs[i_d] / obs[i_a]
    rd = draws[:, i_d] / draws[:, i_a]
    alo, ahi = np.percentile(draws[:, i_a], [2.5, 97.5])
    ill = bool(abs(obs[i_a]) < 5 or (alo < 0 < ahi))
    return float(t), float(np.percentile(rd, 2.5)), float(np.percentile(rd, 97.5)), float(alo), float(ahi), ill


def tracking(R: pd.DataFrame, G: pd.DataFrame, tag: dict) -> tuple[list, list]:
    from wfield_local import analysis_kit as ak
    rows, pooled = [], []
    Gi = G.set_index("animal")
    for gk in ("gain_across", "gain_within"):
        for ep in POST:
            # per animal x spout, and pooled over animals per spout
            for sp in SP:
                pre_d, post_d = {}, {}
                for a in Gi.index:
                    g = float(Gi.loc[a, gk])
                    cp = _cells(R, a, "pre", g); ce = _cells(R, a, ep, g)
                    cp, ce = cp[cp.spout == sp], ce[ce.spout == sp]
                    if len(cp) >= 2 and len(ce) >= 1:
                        pre_d[a], post_d[a] = cp[["d", "ang"]].to_numpy(float), ce[["d", "ang"]].to_numpy(float)
                for who in sorted(pre_d) + ["pooled"]:
                    pr, po = (pre_d, post_d) if who == "pooled" else ({who: pre_d[who]}, {who: post_d[who]})
                    if not pr:
                        continue
                    res = ak.boot_delta_columns(po, pr, np.random.default_rng(SEED))
                    if res is None:
                        continue
                    obs, _lo, _hi, dr, n = res
                    t, tlo, thi, alo, ahi, ill = _ratio(obs, dr, 0, 1)
                    rows.append({**tag, "gain": gk, "epoch": ep, "spout": sp, "animal": who, "n_animals": n,
                                 "n_post_sessions": sum(len(v) for v in po.values()),
                                 "d_decoded_over_gain": float(obs[0]), "d_actual": float(obs[1]), "d_actual_lo": alo,
                                 "d_actual_hi": ahi, "tracking": t, "lo": tlo, "hi": thi, "ill_conditioned": ill})
            # pooled over spouts: sessions with all six spouts
            pre_d, post_d = {}, {}
            for a in Gi.index:
                g = float(Gi.loc[a, gk])
                for ep_, dd in (("pre", pre_d), (ep, post_d)):
                    c = _cells(R, a, ep_, g)
                    w = c.pivot(index="label", columns="spout", values=["d", "ang"]).reindex(
                        columns=pd.MultiIndex.from_product([["d", "ang"], SP])).dropna()
                    if len(w):
                        dd[a] = w.to_numpy(float)
            for who in sorted(set(pre_d) & set(post_d)) + ["pooled"]:
                pr, po = (pre_d, post_d) if who == "pooled" else ({who: pre_d[who]}, {who: post_d[who]})
                if who != "pooled" and (len(pr[who]) < 2):
                    continue
                res = ak.boot_delta_columns(po, pr, np.random.default_rng(SEED))
                if res is None:
                    continue
                obs, _lo, _hi, dr, n = res

                def proj(v):
                    dD, dA = v[..., :6], v[..., 6:]
                    return (dA * dD).sum(-1) / (dA ** 2).sum(-1)
                b = proj(dr)
                pooled.append({**tag, "gain": gk, "epoch": ep, "animal": who, "n_animals": n,
                               "n_post_sessions": sum(len(v) for v in po.values()), "tracking": float(proj(obs)),
                               "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
                               "rms_d_actual": float(np.sqrt((obs[6:] ** 2).mean()))})
    return rows, pooled


# ------------------------------------------------------------------------------------------------------------ FIGURES
def _wrap_titles(fig, sup_width: int = 150) -> None:
    """Re-wrap every axis title (to the axis width) and the suptitle so no title runs off the page (Priya 10-29)."""
    for a_ in fig.axes:
        t = a_.get_title()
        if t:
            w = max(25, int(a_.get_position().width * fig.get_figwidth() * 72 / (0.6 * a_.title.get_fontsize())))
            a_.set_title(_fill(t, w), fontsize=a_.title.get_fontsize())
    if fig._suptitle is not None:
        fig._suptitle.set_text(_fill(fig._suptitle.get_text(), sup_width))


def _yerr(p, lo, hi):
    """errorbar extents, clipped at 0 (a ratio's percentile CI need not contain its point estimate) and the axis."""
    p, lo, hi = (float(np.clip(v, -50, 50)) for v in (p, lo, hi))
    return [[max(p - lo, 0.0)], [max(hi - p, 0.0)]]


def _fill(t: str, w: int) -> str:
    import textwrap
    return "\n".join(textwrap.fill(x, w) for x in t.split("\n"))


def _fig_classmatched_epochs(CM: pd.DataFrame, col: dict, mk: dict) -> None:
    """lick_enl_target_classmatched_epochs.png: frozen position decoder, class-matched deviated - matched hit for read as
    target / read as landing, windows precue / cue / lick x epochs pre..chronic; primary rule (black + animals) and the
    >= 1 deviated sensitivity (gray hollow); n deviated trials per animal under each epoch."""
    import matplotlib.pyplot as plt
    c = CM[(CM.source == "frozen_position") & (CM.subset == "all") & (CM.exec == "exec_land")]
    vals = (("dec_target", "read as TARGET (far contra)"), ("dec_landing", "read as the LANDING bin"))
    wins = ("precue", "cue", "lick")
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.2), sharey=True)
    for i, (v, vt) in enumerate(vals):
        for j, w in enumerate(wins):
            a_ = ax[i, j]
            ticks = []
            for e_i, ep in enumerate(EP):
                rr = {mo: c[(c.window.astype(str) == w) & (c.value == v) & (c.epoch == ep) & (c.min_off == mo)]
                      for mo in CM_MIN_OFF}
                lab = [ep]
                for mo, dx in zip(CM_MIN_OFF, (-0.12, 0.18)):
                    if not len(rr[mo]):
                        lab.append(f"n>={mo}: -")
                        continue
                    r = rr[mo].iloc[0]
                    lab.append(f"n>={mo}: " + "/".join(str(int(r.get(f"n_off_{a}", 0) or 0)) for a in AN))
                    if pd.isna(r.off_minus_on):
                        continue
                    prim = mo == CM_MIN_OFF[0]
                    a_.errorbar(e_i + dx, r.off_minus_on, yerr=_yerr(r.off_minus_on, r.lo, r.hi), fmt="o", ms=7,
                                color="k" if prim else "0.55", mfc="k" if prim else "none", capsize=3, zorder=3,
                                label=f"pooled, sessions with >= {mo} deviated" if (i, j, e_i) == (0, 0, 1) else None)
                    if prim:
                        for ia, a in enumerate(AN):
                            d = r.get(f"d_{a}")
                            if d is not None and np.isfinite(d):
                                a_.plot(e_i + dx - 0.2 + 0.05 * ia, d, mk[a], color=col[a], ms=5, alpha=0.9)
                ticks.append("\n".join(lab))
            a_.axhline(0, color="0.5", lw=0.8)
            a_.axvspan(-0.5, 0.5, color="0.93", zorder=0)
            a_.set_xlim(-0.5, len(EP) - 0.5)
            a_.set_xticks(range(len(EP)), ticks, fontsize=7.5)
            a_.set_title(f"{w} window: {vt}, deviated - matched hit", fontsize=9)
            if j == 0:
                a_.set_ylabel(f"{v}: deviated - matched hit\n(share of trials)")
    from matplotlib.lines import Line2D
    h, l_ = ax[0, 0].get_legend_handles_labels()
    h += [Line2D([], [], ls="", marker=mk[a], color=col[a]) for a in AN]
    l_ += [f"{a} (primary rule)" for a in AN]
    fig.legend(h, l_, fontsize=8, loc="lower center", ncol=len(h), frameon=False)
    fig.suptitle("Class-matched test by epoch, frozen pre-stroke POSITION decoder (pre = its LOSO predictions): far contra "
                 "TARGET trials whose 3-D landing bin is ipsilesional of far contra (deviated) minus the nearest on-target "
                 "far contra trial of the same session (matched hit), all trials. Black = pooled with 95% CI (nested animals "
                 "-> sessions bootstrap of off - on; sessions with >= 2 deviated and >= 2 on-target trials), color = per "
                 "animal; gray hollow = sensitivity with >= 1 deviated trial. Under each epoch: deviated trials per animal "
                 f"({'/'.join(AN)}) under each rule. Shaded pre = the pre-stroke LEVEL; few pre deviated trials.", fontsize=10)
    _wrap_titles(fig, sup_width=170)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    with _trial_set_for("A"):
        save_png_svg(fig, FIG / "lick_enl_target_classmatched_epochs.png", dpi=110)
    plt.close(fig)


def figures() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtrans
    from wfield_local import config
    col = {a: config.animal_color().get(a) for a in AN}
    mk = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    TR = pd.read_csv(OUT / "tracking.csv")
    TP = pd.read_csv(OUT / "tracking_pooled.csv")
    # 1 tracking
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5), sharey=True)
    for i, sub in enumerate(("all", "settled")):
        for k, ep in enumerate(POST):
            a_ = ax[i, k]
            t = TR[(TR.window_s == 1.0) & (TR.subset == sub) & (TR.gain == "gain_across") & (TR.epoch == ep)]
            for j, sp in enumerate(SP):
                for ia, an in enumerate(AN + ["pooled"]):
                    r = t[(t.spout == sp) & (t.animal == an)]
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    x = j + (ia - 2) * 0.14
                    c = "k" if an == "pooled" else col[an]
                    a_.errorbar(x, r.tracking, yerr=_yerr(r.tracking, r.lo, r.hi), fmt=mk.get(an, "*"),
                                color=c, mfc="none" if r.ill_conditioned else c, ms=6 if an != "pooled" else 9, lw=0.8,
                                label=an if j == 0 else None)
            p = TP[(TP.window_s == 1.0) & (TP.subset == sub) & (TP.gain == "gain_across") & (TP.epoch == ep) &
                   (TP.animal == "pooled")]
            if len(p):
                p = p.iloc[0]
                a_.axhspan(p.lo, p.hi, color="0.85", zorder=0)
                a_.text(0.01, 0.98, f"pooled over spouts + animals {p.tracking:+.2f} [{p.lo:+.2f}, {p.hi:+.2f}]",
                        transform=a_.transAxes, va="top", fontsize=8)
            a_.axhline(0, color="tab:blue", lw=1, ls="--"); a_.axhline(1, color="tab:red", lw=1, ls="--")
            a_.set_xticks(range(6), LAB)
            a_.set_ylim(-3, 4)
            a_.set_title(f"{ep}, {sub} trials" + (" (spout settled >= 2 s before ENL)" if sub == "settled" else ""),
                         fontsize=9)
            if k == 0:
                a_.set_ylabel("tracking = d(decoded / gain) / d(actual angle)\n0 = target (blue), 1 = execution (red)")
    h_, l_ = ax[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=9, loc="lower center", ncol=len(h_), frameon=False)
    fig.suptitle("Gain-normalized tracking of the frozen ENL ridge angle read-out (W 1.0 s, no lick 2 s; gain = pre held-out "
                 "slope ACROSS spouts). + angle = contralesional. Hollow = ill-conditioned (|d actual| < 5 deg or its CI "
                 "spans 0); clipped to [-3, 4]", fontsize=10)
    _wrap_titles(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    with _trial_set_for("C"):
        save_png_svg(fig, FIG / "lick_enl_target_tracking.png", dpi=110)
    plt.close(fig)
    # 2 confusion pre vs acute (mean of animals) + change
    CF, CC = pd.read_csv(OUT / "confusion.csv"), pd.read_csv(OUT / "confusion_change.csv")
    fig, ax = plt.subplots(2, 4, figsize=(17, 10))
    for i, (src, ttl) in enumerate((("frozen_position", "frozen POSITION decoder, argmax"),
                                    ("enl_spout", "ENL area classifier, mean prob."))):
        c = CF[(CF.source == src) & (CF.subset == "all") & (CF.window.astype(str).isin(["precue", "1.0"]))]
        for k, ep in enumerate(("pre", "acute", "chronic")):
            M = c[c.epoch == ep].groupby(["target", "decoded"]).share.mean().unstack().reindex(index=SP, columns=SP)
            a_ = ax[i, k]
            a_.imshow(M.to_numpy(float), vmin=0, vmax=0.6, cmap="viridis")
            for r in range(6):
                for cc in range(6):
                    a_.text(cc, r, f"{M.iloc[r, cc]:.2f}", ha="center", va="center", fontsize=6.5, color="w")
            a_.set_title(f"{ttl}\n{ep}: mean of animals", fontsize=9)
            a_.set_xticks(range(6), LAB); a_.set_yticks(range(6), LAB)
        d = CC[(CC.source == src) & (CC.subset == "all") & (CC.window.astype(str).isin(["precue", "1.0"])) &
               (CC.epoch == "acute")]
        M = d.pivot(index="target", columns="decoded", values="change").reindex(index=SP, columns=SP)
        S = d.pivot(index="target", columns="decoded", values="bonf_sig").reindex(index=SP, columns=SP)
        N = d.pivot(index="target", columns="decoded", values="n_same_sign").reindex(index=SP, columns=SP)
        a_ = ax[i, 3]
        a_.imshow(M.to_numpy(float), vmin=-0.3, vmax=0.3, cmap="RdBu_r")
        for r in range(6):
            for cc in range(6):
                a_.text(cc, r, f"{M.iloc[r, cc]:+.2f}\n{N.iloc[r, cc]}/4", ha="center", va="center", fontsize=6)
                if S.iloc[r, cc]:
                    a_.add_patch(plt.Rectangle((cc - .5, r - .5), 1, 1, fill=False, ec="k", lw=2))
        a_.set_title(f"{ttl}\nacute - pre (boxed = Bonferroni / 36)", fontsize=9)
        a_.set_xticks(range(6), LAB); a_.set_yticks(range(6), LAB)
    for a_ in ax.flat:
        a_.set_xlabel("decoded", fontsize=8, labelpad=1)
        a_.set_ylabel("target", fontsize=8, labelpad=1)
    fig.suptitle("PRE-CUE (ENL, 1 s) spout confusion, rows = target, cols = decoded; top row = frozen pre-stroke POSITION "
                 "decoder (LocaNMF, argmax); bottom row = own ENL area classifier (enl_spout, mean probability); "
                 "canonical order close ipsi..far contra (ipsi = ipsilesional). Chance 1/6", fontsize=10)
    _wrap_titles(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    save_png_svg(fig, FIG / "lick_enl_target_confusion.png", dpi=110)
    plt.close(fig)
    # 3 off-target far contra: decoded as target vs executed, by window; class-matched
    OT = pd.read_csv(OUT / "offtarget.csv")
    CM = pd.read_csv(OUT / "classmatched.csv")
    if "min_off" not in CM:
        CM["min_off"] = CM_MIN_OFF[0]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    o = OT[(OT.source == "frozen_position") & (OT.subset == "all") & (OT.exec == "exec_land") & (OT.target == "far_contra")]
    for j, w in enumerate(("precue", "cue", "lick")):
        for e_i, ep in enumerate(EP):
            g = o[(o.window == w) & (o.epoch == ep) & (o.n_off >= MIN_OFF)]
            x = j * 5 + e_i
            ax[0].bar(x - 0.2, g.off_decoded_target.mean(), 0.4, color="tab:blue", label="target" if x == 0 else None)
            ax[0].bar(x + 0.2, g.off_decoded_executed.mean(), 0.4, color="tab:red", label="executed" if x == 0 else None)
            for _, r in g.iterrows():
                ax[0].plot(x - 0.2, r.off_decoded_target, mk[r.animal], color="k", ms=3)
                ax[0].plot(x + 0.2, r.off_decoded_executed, mk[r.animal], color="k", ms=3)
    ax[0].set_xticks([j * 5 + e for j in range(3) for e in range(4)], [ep[:3] for _w in range(3) for ep in EP], fontsize=7)
    tr = mtrans.blended_transform_factory(ax[0].transData, ax[0].transAxes)
    for j, w in enumerate(("precue", "cue", "lick")):
        ax[0].text(j * 5 + 1.5, -0.09, f"{w} window", transform=tr, ha="center", va="top", fontsize=8)
    ax[0].axhline(1 / 6, color="0.5", ls=":")
    ax[0].set_title("frozen position decoder, far contra TARGET trials whose licks land elsewhere (3-D landing):\nshare "
                    f"decoded as target vs as executed bin (animals with >= {MIN_OFF} such trials)", fontsize=8)
    ax[0].legend(fontsize=7)
    for k, (src, wv) in enumerate((("frozen_position", "precue"), ("enl_spout", "1.0"))):
        c = CM[(CM.source == src) & (CM.window.astype(str) == wv) & (CM.subset == "all") & (CM.exec == "exec_land") &
               (CM.min_off == CM_MIN_OFF[0])]
        vals = ["dec_target", "dec_landing", "p_true"]
        for i, v in enumerate(vals):
            for e_i, ep in enumerate(POST + ["post"]):
                r = c[(c.value == v) & (c.epoch == ep)]
                if not len(r) or pd.isna(r.iloc[0].off_minus_on):
                    continue
                r = r.iloc[0]
                x = i * 5 + e_i
                ax[k + 1].errorbar(x, r.off_minus_on, yerr=_yerr(r.off_minus_on, r.lo, r.hi), fmt="o",
                                   color="k")
                for a in AN:
                    ax[k + 1].plot(x + 0.15, r[f"d_{a}"], mk[a], color=col[a], ms=4)
        ax[k + 1].axhline(0, color="0.5")
        # epoch on the tick, the read-out once per group underneath (the stacked "value\nepoch" ticks overlapped)
        ax[k + 1].set_xticks([i * 5 + e for i in range(3) for e in range(4)],
                             [ep[:3] for _v in vals for ep in POST + ["post"]], fontsize=7)
        tr = mtrans.blended_transform_factory(ax[k + 1].transData, ax[k + 1].transAxes)
        for i, v in enumerate(vals):
            ax[k + 1].text(i * 5 + 1.5, -0.09, v, transform=tr, ha="center", va="top", fontsize=8)
        ax[k + 1].set_title(f"{src} pre-cue: far contra trials landing IPSILESIONALLY minus the nearest on-target trial\n"
                            "(same session; dec_landing on the matched trial = its rate of being read as that bin)",
                            fontsize=8)
    _wrap_titles(fig)
    fig.tight_layout()
    save_png_svg(fig, FIG / "lick_enl_target_offtarget.png", dpi=110)
    plt.close(fig)
    _fig_classmatched_epochs(CM, col, mk)
    # 4 decodability
    DT = pd.read_csv(OUT / "decodability_tests.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for k, (src, w, v, ttl) in enumerate((("frozen_position", "precue", "acc", "frozen position decoder accuracy"),
                                          ("enl_spout", "1.0", "acc", "enl_spout accuracy"),
                                          ("ridge", "1.0", "r_raw", "ridge raw r (angle)"))):
        for i, sub in enumerate(("all", "settled")):
            d = DT[(DT.source == src) & (DT.window.astype(str) == w) & (DT.value == v) & (DT.subset == sub)]
            for e_i, ep in enumerate(POST):
                r = d[d.epoch == ep]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = e_i + (i - 0.5) * 0.3
                ax[k].errorbar(x, r.delta, yerr=_yerr(r.delta, r.lo, r.hi), fmt="o" if i == 0 else "s",
                               color="k" if i == 0 else "tab:green", label=sub if e_i == 0 else None)
                for a in AN:
                    ax[k].plot(x + 0.08, r[f"d_{a}"], mk[a], color=col[a], ms=4)
        ax[k].axhline(0, color="0.5")
        ax[k].set_xticks(range(3), POST)
        ax[k].set_title(f"{ttl}: change vs pre\n(paired nested bootstrap; color = animals)", fontsize=9)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, ["all trials", "settled trials"] if l_ == ["all", "settled"] else l_, fontsize=9,
               loc="lower center", ncol=len(h_), frameon=False)
    _wrap_titles(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_png_svg(fig, FIG / "lick_enl_target_decodability.png", dpi=110)
    plt.close(fig)
    print(f"== figures in {FIG}")


# ------------------------------------------------------------------------------------------------------------ CAPTIONS
#: the attempt_coding classes PART A keeps (`precue_trials`)
A_CLASSES = ("pre_hit", "hit", "incomplete")


def trial_set_sentence(part: str = "all") -> str:
    """The trials of this driver's figures (Priya 11-09): ``part`` "A" = the frozen position decoder's attempt_coding
    trials, "C" = the ENL ridge trials at W 1.0 (tracking figure, + its settled subset), "all" = A + the ENL trials of
    every window (confusion / off-target / decodability)."""
    from wfield_local import trial_set_text as tst
    lf = _LF_ARG or ead.LF_PRIMARY
    a = tst.events("attempt_coding per-trial tables of the frozen pre-stroke POSITION decoder (windows precue = its "
                   "lick-free ENL window, cue, lick; method dom)", unit="trials",
                   keep=["attempt classes " + " + ".join(A_CLASSES), "executed bin = argmax of the 3-D landing / "
                         "trajectory softmax (trials without 3-D licks have none)"],
                   excluded=["other attempt classes", "early (June) pre sessions"])
    settled = (f"settled subset: cue - spout strobe >= W + {SETTLE_GAP:g} s")
    if part == "A":
        return a
    if part == "C":
        return ead.trial_set_sentence(1.0, lf).replace("; excluded: ", f"; {settled}; excluded: ", 1)
    enl = [ead.trial_set_sentence(W, lf) for W in sorted({W for W, _lf in VARIANTS}, reverse=True)]
    return tst.union(a, *enl, unit="trials") + f"; {settled}"


def _trial_set_for(part: str):
    """Narrow the process declaration to ``part`` while a figure is saved (no-op when main() declared nothing)."""
    import contextlib

    from wfield_local import trial_set_text as tst
    if tst.current() is None:
        return contextlib.nullcontext()
    return tst.active(None, text=trial_set_sentence(part))


# ------------------------------------------------------------------------------------------------------------ MAIN
def main() -> int:
    from wfield_local import config
    from wfield_local import writeguard
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    _check_order()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    writeguard.assert_writable(OUT / "tracking.csv")
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if "--plot-only" in sys.argv:
        figures()
        return 0
    if "--classmatched-only" in sys.argv:
        classmatched_only()
        return 0
    il.split_only("lick_enl_target_test.py")
    jobs = min(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8, 8)
    T, names = ead.prepare(pd.read_csv(ead.TRIALS))
    from wfield_local import attempt_coding as _ac
    PQ = {an: pd.read_csv(_ac.out_dir() / attempt_trials_name("precue", an), usecols=["label"]) for an in AN}   # review N1
    labs = sorted(set(T.label) | {x for v in PQ.values() for x in v.label.unique() if not config.early_pre(x)})
    TM = timing(labs, jobs)
    # ---- A: frozen position decoder
    PT = precue_trials(TM)
    pd.concat(PT.values(), ignore_index=True).to_csv(OUT / "precue_trials.csv.gz", index=False, float_format="%.5g")
    OTr, CFl, CFc, ACC, CT = [], [], [], [], []
    for w, D in PT.items():
        for sub, X in subsets(D, 1.0) if w == "precue" else (("all", D),):
            tag = {"source": "frozen_position", "window": w, "subset": sub}
            OTr += offtarget(X, tag)
            for ex in ("exec_land", "exec_traj"):
                CT += cls_tracking(X, ex, tag)
            if w == "precue":
                lv, ch = confusion(X, tag, prob=False); CFl += lv; CFc += ch
            ACC += accuracy_sessions(X, tag, ridge=False)
    # ---- B + C: own classifier and ridge
    TRK, TRP, GN = [], [], []
    RT = {}
    for W, lf in ((W_, _LF_ARG or lf_) for W_, lf_ in VARIANTS):
        R = attach(model_trials(T, names, W, lf), TM)
        RT[W] = R
        G = gains(R).assign(window_s=W, lickfree=lf)
        GN.append(G)
        for sub, X in subsets(R, W):
            tag = {"source": "enl_spout", "window": f"{W:.1f}", "subset": sub}
            OTr += offtarget(X, tag, ("exec_land", "exec_traj", "exec_ang"))
            for ex in ("exec_land", "exec_ang"):
                CT += cls_tracking(X, ex, tag)
            lv, ch = confusion(X, tag, prob=True); CFl += lv; CFc += ch
            ACC += accuracy_sessions(X, tag, ridge=False)
            ACC += accuracy_sessions(X, {**tag, "source": "ridge"}, ridge=True)
            a_, p_ = tracking(X, G, {"window_s": W, "lickfree": lf, "subset": sub})
            TRK += a_; TRP += p_
    pd.concat(RT.values(), ignore_index=True).to_csv(OUT / "cls_trials.csv.gz", index=False, float_format="%.5g")
    GN = pd.concat(GN, ignore_index=True); GN.to_csv(OUT / "gains.csv", index=False)
    TRK, TRP = pd.DataFrame(TRK), pd.DataFrame(TRP)
    TRK.to_csv(OUT / "tracking.csv", index=False); TRP.to_csv(OUT / "tracking_pooled.csv", index=False)
    SH = shrink_decomposition(TRK, GN); SH.to_csv(OUT / "tracking_shrink.csv", index=False)
    print("\n== SHRINK vs TRACK (sensitivity)\n" + SH.round(2).to_string(index=False))
    OTr = pd.DataFrame(OTr); OTr.to_csv(OUT / "offtarget.csv", index=False)
    CMs, CMt = classmatched_all(PT, RT)
    CMs.to_csv(OUT / "classmatched_sessions.csv", index=False)
    CMt.to_csv(OUT / "classmatched.csv", index=False)
    CT = pd.DataFrame(CT); CT.to_csv(OUT / "cls_tracking.csv", index=False)
    pd.DataFrame(CFl).to_csv(OUT / "confusion.csv", index=False)
    pd.DataFrame(CFc).to_csv(OUT / "confusion_change.csv", index=False)
    ACC = pd.DataFrame(ACC); ACC.to_csv(OUT / "accuracy_sessions.csv", index=False)
    DT = decod_tests(ACC); DT.to_csv(OUT / "decodability_tests.csv", index=False)
    # ---- per-animal tables first (rule 8)
    print("\n== GAINS (pre held-out)\n" + GN.round(3).to_string(index=False))
    t = TRK[(TRK.spout == "far_contra") & (TRK.gain == "gain_across")]
    print("\n== TRACKING far_contra (gain across)\n" + t.round(2).to_string(index=False))
    print("\n== TRACKING pooled over spouts\n" + TRP.round(2).to_string(index=False))
    print("\n== OFF-TARGET far_contra\n" + OTr[OTr.target == "far_contra"].round(2).to_string(index=False))
    print("\n== CLASSIFIER TRACKING (0 = target, 1 = landing)\n" + CT.round(2).to_string(index=False))
    print("\n== CLASS-MATCHED\n" + CMt.round(3).to_string(index=False))
    print("\n== DECODABILITY\n" + DT[DT.value.isin(["acc", "r_raw", "fc_lat", "fc_lat_p", "acc_far_contra"])]
          .round(3).to_string(index=False))
    figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
