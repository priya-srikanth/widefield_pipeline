"""T13 follow-up: is the weaker frozen-decoder position code on trials with a pre-cue orofacial event a CONTAMINATION of
the decoder's features by the event transient, or a weaker PLAN? (cohort1 PS92-95, 2026-11-01;
docs/status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md section 4, T13 follow-up.)

THE DECODER INPUT, exactly as `wfield_local.attempt_coding.run_animal` builds it (LP_licks run
lp_r4_heatmap_20261013__clean_v36, the source of T13's p_true):
  * signal = the animal's joint-LocaNMF basis courses (`joint_locanmf.BasisSource`, 95 components for PS92)
  * window = the ENL pre-cue window. UNDER LP_licks THIS IS 1.0 s (`lp_licks.LP_ENL_WINDOW_S`, applied by
    `lp_licks.apply_enl_window` inside `trial_features_cached`; the frozen spec carries post_s = 1.0), NOT the 2 s T13
    assumed for its `ev_enl` flag. It ends at the cue, or is slid earlier when an LP lick falls inside it
    (`precue_window_start`).
  * feature = the window split into decode.bins.precue = 4 equal sub-bins (~0.25 s, 7-8 frames each) and the per-bin
    mean of every component, concatenated (`_window_feature`), z-scored with that session's own ENGAGED trials
    (`pool_sessions`), then the STORED frozen pre-stroke model (`frozen_models.load_or_fit` through `decoder_spec`, with
    a fit callable that RAISES: nothing is ever refitted, CLAUDE.md rule 10). Pre-stroke sessions use their stored
    leave-that-session-out model, post-stroke sessions the full model -- as attempt_coding.
  The input IS rebuildable per frame: this script re-runs `_trial_features` on the same signal with `_window_feature`
  and `_frames` instrumented, so it records each kept trial's real window start (incl. slid windows) and the raw window
  (components x frames) behind the cached feature. Check: rebuilt full-window p_true vs the stored p_true (printed).

EXCLUSION. Event = the T13 event (orofacial ipsi or contra, either polarity, > 4 x the animal's August-pre robust SD,
detected in the clean pre-cue mask, from task_events/sessions/<label>.npz); excluded span = peak -0.3 .. +0.6 s. A trial
is an EVENT trial ("ev_win") if any event's span overlaps its actual decoder window. With the event frames removed each
4-bin feature is rebuilt from the REMAINING frames of that bin; a bin with no remaining frame takes the mean of all
remaining frames of the window (primary, "fill = window mean"; sensitivity "fill = neutral" sets it to the session's
engaged-trial mean, z = 0). A 0.9-s span inside a 1.0-s window leaves only a few frames, so the readout after exclusion
is noisier for BOTH groups -- which is why the controls lose the same frames.
MATCHED CONTROLS: every no-event trial (no event span overlapping its window) gets the window-relative exclusion mask of
a randomly drawn event trial of the same session x spout (fallback: same session), N_DRAW draws averaged, so both groups
lose the same frames at the same place in the window (= a random pseudo-event time drawn from the real event times).
(b) Effect = the T13 statistic (`quiet_task_events.excess`): mean over event trials of (p_true - stratum mean), STRICT
strata = session x spout x within-session trial-order quartile x clean-time bin (T13's), p = within-stratum permutation.
Reported BEFORE exclusion (full window, rebuilt) and AFTER, on the same trials and the same flag.
(c) TIME COURSE: p_true from the window frames BEFORE the event span only, and AFTER it only (same fill rule; controls get
the drawn event trial's before / after masks). Also the contamination-free test: trials whose only pre-cue events lie
ENTIRELY BEFORE the decoder window (span ends before the window starts; possible because T13 flagged events in the
last 2 s and the clean pre-cue period) vs trials with no pre-cue event at all -- full-window p_true, nothing excluded.
Per-animal tables first (CLAUDE.md rule 8). Cohort1 only.

OUTPUT M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes/<v2_quiet | v2_quiet_lp>/task_events_followup/
    (follows QS_LICK_SOURCE: daq -> v2_quiet, daq+lp -> v2_quiet_lp, whose task_events/ carry LP-masked events. The
    windows/ cache is the decoder input and does NOT depend on the lick source -- copy it across instead of re-extracting.)
    windows/<label>.npz        raw decoder windows per kept trial (worker output; cache)
    trials.csv.gz              one row per T13 decoder trial: flags, p_true full / excl / before / after (+ neutral fill)
    effects.csv                per animal x epoch group x analysis x flag: excess, p (session x spout) and strict
    t13_followup_exclusion.png

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_task_events_followup.py [--jobs 4] [--summarise] [--only PS92_0811 ...]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("WIDEFIELD_LICK_SOURCE", "lp")      # attempt_coding's p_true is the LP_licks run
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_spikes_v2 import LICK_SOURCE, V2  # noqa: E402

TE = V2 / "task_events"
OUT = V2 / "task_events_followup"
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
TRIAL_COLS = ("orofacial_left", "orofacial_right")
K = 4.0
SPAN = (-0.3, 0.6)
N_DRAW = 20
CLEAN_BINS = (0, 1.5, 2.0, 2.5, 99)
SHORT = {"pre": "pre", "acute+subacute": "acute+sub", "chronic": "chronic"}
GROUPS = {"pre": ("pre",), "acute+subacute": ("acute", "subacute"), "chronic": ("chronic",),
          "acute": ("acute",), "subacute": ("subacute",)}
_BASIS = {}


def _basis(animal):
    from wfield_local import joint_locanmf
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
    if animal not in _BASIS:
        _BASIS[animal] = joint_locanmf.load(animal, sessions=_ALL)
    return _BASIS[animal]


def extract(label):
    """One session (spawn worker): re-run attempt_coding's precue `_trial_features` on the joint-basis signal with the
    window builder instrumented; save every kept trial's raw window + window start + frame times."""
    f = OUT / "windows" / f"{label}.npz"
    if f.exists():
        return label
    from wfield_local import joint_locanmf
    from wfield_local import locanmf_position_decoder as lpd
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS as _ALL
    from wfield_local.plot_spout_trial_averages import _load_daq_events
    s = next(x for x in _ALL if x["label"] == label)
    basis = _basis(label[:4])
    sig, reg = joint_locanmf.BasisSource(basis, s).signal()
    rec, fr = [], {}
    orig_wf, orig_frames = lpd._window_feature, lpd._frames

    def wf(sig_, w0, post_n, bins, base):
        rec.append((int(w0), np.asarray(sig_[:, w0:w0 + post_n], np.float32).copy(), int(bins)))
        return orig_wf(sig_, w0, post_n, bins, base)

    def frames(s_, cue, lk):
        out = orig_frames(s_, cue, lk)
        fr["csmp"], fr["cue_f"] = out[2], out[0]
        return out

    lpd._window_feature, lpd._frames = wf, frames
    try:
        args = _args("locanmf", "precue", 2.0)
        out = lpd._trial_features(s, args, signal=sig, feat_region=reg,
                                  with_indices=True, with_rt=True, nolick_ref="cue")
    finally:
        lpd._window_feature, lpd._frames = orig_wf, orig_frames
    X, y, _g, Xn, yn, _reg, ie, inl, _rt = out
    order = np.sort(np.r_[ie, inl])                     # the loop visits cues in order, one window per kept trial
    if len(order) != len(rec):
        raise RuntimeError(f"{label}: {len(rec)} windows recorded vs {len(order)} kept trials")
    cue = _load_daq_events(s["h5"])
    sr = float(cue["sample_rate_hz"])
    csmp = np.asarray(fr["csmp"], float)
    w0 = np.array([r[0] for r in rec])
    W = np.stack([r[1] for r in rec])                    # trials x comps x frames
    post_n = W.shape[2]
    tw = csmp[np.clip(w0[:, None] + np.arange(post_n)[None], 0, len(csmp) - 1)] / sr
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, k=order, w0=w0, W=W, tw=tw, bins=rec[0][2], cue_t=np.asarray(cue["cue_samples"], float)[order] / sr,
                        cue_f=np.asarray(fr["cue_f"])[order], ie=ie, inl=inl, X=X.astype(np.float32),
                        Xn=np.asarray(Xn, np.float32), y=y, yn=yn, fs=float(args.fs))
    return label


# ---------------------------------------------------------------------------------------------------------- features
def feat(W, keep, bins, fill="mean"):
    """4-bin features from the kept frames only. W trials x comps x frames, keep trials x frames (bool).
    Empty bin -> mean of all kept frames ("mean") or NaN ("neutral": set to z = 0 after z-scoring). All-empty -> NaN."""
    n, c, T = W.shape
    edges = np.linspace(0, T, bins + 1).astype(int)
    kf = keep.astype(np.float32)
    tot = kf.sum(1)
    allm = np.einsum("nct,nt->nc", W, kf) / np.maximum(tot, 1)[:, None]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        kb = kf[:, a:b]
        nb = kb.sum(1)
        m = np.einsum("nct,nt->nc", W[:, :, a:b], kb) / np.maximum(nb, 1)[:, None]
        empty = nb == 0
        m[empty] = allm[empty] if fill == "mean" else np.nan
        out.append(m)
    F = np.concatenate(out, 1)
    F[tot == 0] = np.nan
    return F


def predict(model, Z, code):
    """p(true spout) per row; NaN rows -> NaN; NaN entries (neutral fill) -> 0 (= the session's engaged mean)."""
    p = np.full(len(Z), np.nan)
    ok = np.isfinite(Z).any(1)
    if not ok.any():
        return p
    pr = model.predict_proba(np.nan_to_num(Z[ok], nan=0.0))
    cl = [int(c) for c in model.classes_]
    idx = np.array([cl.index(int(c)) if int(c) in cl else -1 for c in code[ok]])
    p[ok] = np.where(idx >= 0, pr[np.arange(ok.sum()), np.clip(idx, 0, None)], np.nan)
    return p


def frozen_models_for(animal):
    """(models, kept, pre labels, pooled XE by label) -- the STORED frozen precue decoder of attempt_coding. Load only."""
    from wfield_local import config, frozen_models, joint_locanmf  # noqa: F401
    from wfield_local.locanmf_frozen_decoder import decoder_spec, pool_sessions
    from wfield_local.precue_engagement_states import features_with_indices
    pre = [l for l in config.phase_labels("pre") if l.startswith(animal)]
    post = [l for l in config.phase_labels("post") if l.startswith(animal)]
    ft = features_with_indices(_basis(animal), nolick_ref="cue")
    XE, YE, GE, _BE, XU, YU, kept, _c, GU = pool_sessions(pre + post, source="locanmf", align="precue", post_s=2.0,
                                                          features=ft)
    pre_list = [i for i, lab in enumerate(kept) if lab in set(pre)]
    spec = decoder_spec(XE, kept, pre_list, YE=YE, GE=GE, align="precue", source="locanmf", post_s=2.0,
                        basis_id=getattr(ft, "basis_id", None))

    def _never():
        raise RuntimeError(f"{animal}: no stored frozen precue decoder for this spec -- refusing to fit (rule 10)")
    models, status = frozen_models.load_or_fit(spec, _never)
    print(f"[followup] {animal}: frozen precue decoder {status} (window {spec['post_s']} s, "
          f"{spec['n_features']} features, {len(pre_list)} pre sessions)", flush=True)
    raw = {lab: XE[GE == i] for i, lab in enumerate(kept)}
    return models, kept, set(pre), raw


def session_events(lab, sd):
    """Absolute DAQ times (s) of the T13 trial-flag events of one session (clean pre-cue mask)."""
    from quiet_area_spikes import columns
    d = np.load(TE / "sessions" / f"{lab}.npz", allow_pickle=True)
    ev, cols = d["ev"], columns()
    big = np.zeros(len(ev), bool)
    for c in TRIAL_COLS:
        ci = cols.index(c)
        big |= (ev[:, 0] == ci) & (np.abs(ev[:, 3]) > K * sd[lab[:4]][ci])
    E = ev[big & (ev[:, 1] == 0)]
    cue = d["cue"]
    return cue[E[:, 4].astype(int)] + E[:, 7], cue, d["trial_id"]


def score_session(lab, models, pre_set, raw, sd, rng):
    import pandas as pd
    z = np.load(OUT / "windows" / f"{lab}.npz", allow_pickle=True)
    W, tw, bins = z["W"], z["tw"], int(z["bins"])
    n, _c, T = W.shape
    ie = z["ie"]
    eng = np.isin(z["k"], ie)
    code = np.empty(n, int)
    code[eng] = z["y"]
    code[~eng] = z["yn"]
    # rebuilt full-window features must equal the trial_features output
    F_full = feat(W, np.ones((n, T), bool), bins)
    Xo = np.empty_like(F_full)
    Xo[eng] = z["X"]
    if (~eng).any():
        Xo[~eng] = z["Xn"]
    err = float(np.nanmax(np.abs(F_full - Xo)))
    mu, sdv = z["X"].mean(0), z["X"].std(0)
    sdv = np.where(sdv > 0, sdv, 1.0)
    # the pooled (cached) engaged features of this session must match too (same mu / sd as the stored run)
    err_pool = float(np.max(np.abs(raw[lab] - ((z["X"] - mu) / sdv)))) if lab in raw else np.nan
    mdl = models["loso"].get(lab, models["full"]) if lab in pre_set else models["full"]

    def P(keep, fill="mean", idx=None):
        idx = np.arange(n) if idx is None else idx
        return predict(mdl, (feat(W[idx], keep, bins, fill) - mu) / sdv, code[idx])

    # events vs this trial's window
    et, cue_tb, tid_tb = session_events(lab, sd)
    dt = 1.0 / float(z["fs"])
    w_lo, w_hi = tw[:, 0], tw[:, -1] + dt
    span_lo, span_hi = et + SPAN[0], et + SPAN[1]
    keep_ex = np.ones((n, T), bool)
    keep_bef = np.ones((n, T), bool)
    keep_aft = np.ones((n, T), bool)
    ev_win = np.zeros(n, bool)
    ev_out = np.zeros(n, bool)          # a pre-cue event of THIS trial entirely before its window
    first_lo = np.full(n, np.nan)
    for i in range(n):
        ov = (span_hi > w_lo[i]) & (span_lo < w_hi[i])
        if ov.any():
            ev_win[i] = True
            inside = np.zeros(T, bool)
            for a, b in zip(span_lo[ov], span_hi[ov]):
                inside |= (tw[i] >= a) & (tw[i] < b)
            keep_ex[i] = ~inside
            keep_bef[i] = tw[i] < span_lo[ov].min()
            keep_aft[i] = tw[i] >= span_hi[ov].max()
            first_lo[i] = (span_lo[ov].min() - w_lo[i])
    # trial id via the cue time (trial_bounds cue_s vs the decoder's DAQ cue)
    j = np.clip(np.searchsorted(cue_tb, z["cue_t"]), 1, len(cue_tb) - 1)
    j = np.where(np.abs(cue_tb[j - 1] - z["cue_t"]) < np.abs(cue_tb[j] - z["cue_t"]), j - 1, j)
    match = np.abs(cue_tb[j] - z["cue_t"]) < 0.05
    trial_id = np.where(match, tid_tb[j], -1)
    # events entirely before the window but after this trial's spout arrival (same trial's pre-cue period)
    for i in range(n):
        if not ev_win[i] and match[i]:
            prev_cue = cue_tb[j[i] - 1] if j[i] > 0 else -np.inf
            ev_out[i] = bool(np.any((span_hi <= w_lo[i]) & (et > prev_cue) & (et < z["cue_t"][i])))
    p_full = P(np.ones((n, T), bool))
    p_ex = np.full(n, np.nan); p_bef = np.full(n, np.nan); p_aft = np.full(n, np.nan); p_exn = np.full(n, np.nan)
    e_idx = np.flatnonzero(ev_win)
    if len(e_idx):
        p_ex[e_idx] = P(keep_ex[e_idx], idx=e_idx)
        p_exn[e_idx] = P(keep_ex[e_idx], "neutral", e_idx)
        hb = keep_bef[e_idx].any(1); ha = keep_aft[e_idx].any(1)
        p_bef[e_idx[hb]] = P(keep_bef[e_idx[hb]], idx=e_idx[hb])
        p_aft[e_idx[ha]] = P(keep_aft[e_idx[ha]], idx=e_idx[ha])
    # matched pseudo-event masks for the no-event trials: same session x spout event trials, else same session
    nb_idx = np.flatnonzero(~ev_win)
    n_frames_removed = (~keep_ex).sum(1).astype(float)
    pseudo_src = np.full(n, "", object)
    if len(e_idx) and len(nb_idx):
        acc = {k: np.zeros(n) for k in ("ex", "exn", "bef", "aft")}
        cnt = {k: np.zeros(n) for k in ("ex", "exn", "bef", "aft")}
        nrem = np.zeros(n)
        pools = {}
        for i in nb_idx:
            same = e_idx[code[e_idx] == code[i]]
            pools[i] = same if len(same) else e_idx
            pseudo_src[i] = "session x spout" if len(same) else "session"
        for _d in range(N_DRAW):
            pick = np.array([rng.choice(pools[i]) for i in nb_idx])
            for key, KM, fill in (("ex", keep_ex, "mean"), ("exn", keep_ex, "neutral"), ("bef", keep_bef, "mean"),
                                  ("aft", keep_aft, "mean")):
                km = KM[pick]
                has = km.any(1)
                v = np.full(len(nb_idx), np.nan)
                if has.any():
                    v[has] = predict(mdl, (feat(W[nb_idx[has]], km[has], bins, fill) - mu) / sdv, code[nb_idx[has]])
                okv = np.isfinite(v)
                acc[key][nb_idx[okv]] += v[okv]
                cnt[key][nb_idx[okv]] += 1
            nrem[nb_idx] += (~keep_ex[pick]).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_ex[nb_idx] = (acc["ex"] / cnt["ex"])[nb_idx]
            p_exn[nb_idx] = (acc["exn"] / cnt["exn"])[nb_idx]
            p_bef[nb_idx] = np.where(cnt["bef"][nb_idx] > 0, acc["bef"][nb_idx] / np.maximum(cnt["bef"][nb_idx], 1),
                                     np.nan)
            p_aft[nb_idx] = np.where(cnt["aft"][nb_idx] > 0, acc["aft"][nb_idx] / np.maximum(cnt["aft"][nb_idx], 1),
                                     np.nan)
        n_frames_removed[nb_idx] = nrem[nb_idx] / N_DRAW
    win_end_rel = (w_hi - z["cue_t"])
    return pd.DataFrame(dict(label=lab, trial_id=trial_id, k=z["k"], code=code, engaged_arm=eng, ev_win=ev_win,
                             ev_out=ev_out, win_end_rel_cue=win_end_rel, first_span_rel_win=first_lo,
                             n_frames=T, n_removed=n_frames_removed, pseudo_src=pseudo_src,
                             p_full=p_full, p_excl=p_ex, p_excl_neutral=p_exn, p_before=p_bef, p_after=p_aft)), \
        dict(label=lab, feat_err=err, pool_err=err_pool, n_trials=n, n_ev_win=int(ev_win.sum()),
             cue_match=float(match.mean()))


def build_table():
    import pandas as pd
    from quiet_fluct import animal_sd
    sd = animal_sd()
    rng = np.random.default_rng(0)
    rows, chk = [], []
    for an in ANIMALS:
        labs = sorted(f.stem for f in (OUT / "windows").glob(f"{an}_*.npz"))
        if not labs:
            continue
        models, kept, pre_set, raw = frozen_models_for(an)
        for lab in labs:
            if lab not in kept:
                print(f"  {lab}: not in the frozen-decoder pool -> skipped", flush=True)
                continue
            r, c = score_session(lab, models, pre_set, raw, sd, rng)
            rows.append(r); chk.append(c)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(chk)


# ------------------------------------------------------------------------------------------------------------- stats
def analysis_frame(R):
    """T13's p_true subset (canonical, engaged, spout agrees, >= 1 s clean in the last 2 s, p_true present) + strata."""
    import pandas as pd
    T = pd.read_csv(TE / "t13_trials.csv.gz")
    B = T[(T.canonical == True) & T.eng_tr & (T.pos == T.spout_daq)].copy()          # noqa: E712
    B["stratum"] = B.label + "|" + B.pos
    q = B.groupby("label").trial_id.transform(lambda x: np.floor(4 * (x.rank(method="first") - 1) / len(x)))
    B["tq"] = q.astype(int).astype(str)
    b = B[(B.clean_enl_s >= 1.0) & B.p_true.notna() & (B.spout_ac == B.pos)].copy()
    b["strict"] = b.stratum + "|" + b.tq + "|" + pd.cut(b.clean_enl_s, CLEAN_BINS, labels=False).astype(str)
    m = b.merge(R[R.trial_id >= 0].drop_duplicates(["label", "trial_id"]), on=["label", "trial_id"], how="inner")
    m["hit"] = m.cls_ac.isin(["hit", "pre_hit"])
    return m, b


ANALYSES = (
    # name, p column, flag, subset rule
    ("T13 replication (stored p_true, event in last 2 s)", "p_true", "ev_enl", "all"),
    ("before exclusion (full window, event in window)", "p_full", "ev_win", "all"),
    ("AFTER exclusion (fill = window mean)", "p_excl", "ev_win", "all"),
    ("after exclusion (fill = neutral)", "p_excl_neutral", "ev_win", "all"),
    ("before-event frames only", "p_before", "ev_win", "all"),
    ("after-event frames only", "p_after", "ev_win", "all"),
    ("event ENTIRELY BEFORE window vs no pre-cue event (full window)", "p_full", "ev_out", "out"),
    ("hits: before exclusion", "p_full", "ev_win", "hit"),
    ("hits: AFTER exclusion", "p_excl", "ev_win", "hit"),
)


def effects(M):
    import pandas as pd
    from quiet_task_events import excess
    rows = []
    for name, col, flag, sub in ANALYSES:
        G0 = M
        if sub == "hit":
            G0 = M[M.hit]
        if sub == "out":
            G0 = M[~M.ev_win & (M.ev_out | ~M.ev_pre)]          # contrast: event only before window vs no event
        for gname, eps in GROUPS.items():
            G = G0[G0.epoch.isin(eps)]
            for an, H in list(G.groupby("animal")) + [("ALL", G)]:
                H = H[np.isfinite(H[col])]
                r = excess(H[col], H[flag], H.stratum, n_perm=1000)
                rs = excess(H[col], H[flag], H.strict, n_perm=1000)
                rows.append(dict(analysis=name, p_col=col, flag=flag, animal=an, epoch=gname,
                                 n_sessions=H.label.nunique(), n_trials=r["n_trials"], n_event=r["n_event"],
                                 mean_event=r["rate_event"], mean_noevent=r["rate_noevent"], excess=r["excess"],
                                 p=r["p"], n_event_strict=rs["n_event"], excess_strict=rs["excess"],
                                 p_strict=rs["p"]))
    return pd.DataFrame(rows)


def figure(E, M, out=None):
    """Priya 11-03 rules: no lines; per-animal STRICT excess small FILLED (shape = animal, colour = epoch group), all
    animals large filled; the two conditions of a panel (full / removed, before / after) are separate x positions inside
    each epoch group. Per-SESSION values are not stored (the excess pools session x spout strata), so there are no
    session dots. Returns {(analysis, animal, epoch group): plotted excess_strict} for the --plot-only check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import cat_points as cp
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import EPOCH_LINE
    eg = ("pre", "acute+subacute", "chronic")
    col = {"pre": EPOCH_LINE["pre"], "acute+subacute": EPOCH_LINE["acute"], "chronic": EPOCH_LINE["chronic"]}
    panels = [
        ("A  before vs AFTER removing the event frames\n(STRICT strata; left = full window, right = event frames removed)",
         ("before exclusion (full window, event in window)", "AFTER exclusion (fill = window mean)")),
        ("B  hits only: before vs AFTER removal",
         ("hits: before exclusion", "hits: AFTER exclusion")),
        ("C  time course inside the window\n(left = frames BEFORE the event, right = frames AFTER it)",
         ("before-event frames only", "after-event frames only")),
        ("D  event ENTIRELY BEFORE the decoder window\n(nothing removed; vs trials with no pre-cue event)",
         ("event ENTIRELY BEFORE window vs no pre-cue event (full window)",)),
    ]
    plotted = {}
    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5))
    for a, (title, names) in zip(ax.flat, panels):
        nn = len(names)
        offs = [0.0] if nn == 1 else [-0.22, 0.22]
        for x, ep in enumerate(eg):
            for k, name in enumerate(names):
                G = E[(E.analysis == name) & (E.epoch == ep)]
                xs = x + offs[k]
                for ia, an in enumerate(ANIMALS):
                    v = G[G.animal == an]
                    if len(v) and np.isfinite(v.excess_strict.iloc[0]):
                        plotted[(name, an, ep)] = float(v.excess_strict.iloc[0])
                        a.plot(xs - 0.1 + 0.045 * ia, v.excess_strict.iloc[0], cp.animal_marker(an), ls="none",
                               color=col[ep], mec="w", mew=0.4, ms=5, zorder=3)
                v = G[G.animal == "ALL"]
                if len(v) and np.isfinite(v.excess_strict.iloc[0]):
                    plotted[(name, "ALL", ep)] = float(v.excess_strict.iloc[0])
                    cp.summary_point(a, xs + 0.08, v.excess_strict.iloc[0], None, col[ep], ms=8)
                    p = v.p_strict.iloc[0]
                    if p < 0.05:
                        sig_marks.star_top(a, xs + 0.08, "***" if p < 0.001 else "**" if p < 0.01 else "*",
                                           color=col[ep])
        if nn > 1:
            for x in range(len(eg)):
                a.axvline(x + 0.5, color="0.9", lw=0.6)
            a.set_xticks([x + o for x in range(len(eg)) for o in offs],
                         [f"{SHORT[e]}\n{'full' if o < 0 else 'removed'}" if "exclusion" in names[0]
                          else f"{SHORT[e]}\n{'before' if o < 0 else 'after'}" for e in eg for o in offs], fontsize=8)
        else:
            a.set_xticks(range(len(eg)), eg, fontsize=8)
        a.set_xlim(-0.5, len(eg) - 0.5)
        a.axhline(0, color="0.6", lw=0.8)
        a.set_ylabel("p_true: event trials minus stratum mean")
        a.set_title(title, fontsize=9)
    handles = [plt.Line2D([], [], marker=cp.animal_marker(an), ls="", color="0.35", mec="w", label=an)
               for an in ANIMALS]
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, label=e) for e, c in col.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="k", mec="w", ms=8, label="all animals"))
    fig.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=8, frameon=False)
    sig_marks.reserve_headroom(fig)
    nev = int(M.ev_win.sum())
    fig.suptitle("T13 follow-up: is the weaker frozen-decoder position code on pre-cue-event trials contamination by the "
                 "transient or a weaker plan? Frozen pre-stroke precue decoder (LP_licks, 1.0-s window, 4 sub-bins), "
                 "rebuilt per frame; event span = peak -0.3..+0.6 s; no-event controls lose the same frames (matched "
                 f"pseudo-event masks, {N_DRAW} draws). Small filled = animal (shape), colour = epoch; large = all "
                 "animals; STRICT strata; stars = STRICT within-stratum permutation p. Per-session values are not stored "
                 f"(the excess pools session x spout strata), so no session dots. {nev} event-in-window trials. "
                 "Events / masks: "
                 + ("DAQ contact + LP video licks (QS_LICK_SOURCE=daq+lp)." if LICK_SOURCE == "daq+lp"
                    else "DAQ contact licks."), fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "t13_followup_exclusion.png" if out is None else out, dpi=110)
    plt.close(fig)
    return plotted


def plot_only():
    """Re-render from the stored effects.csv / trials.csv.gz only -> t13_followup_exclusion_nolines.png, and check the
    plotted values against effects.csv."""
    import pandas as pd
    E = pd.read_csv(OUT / "effects.csv")
    M = pd.read_csv(OUT / "trials.csv.gz", low_memory=False)
    P = figure(E, M, out=OUT / "t13_followup_exclusion_nolines.png")
    for (name, an, ep), v in P.items():
        want = E[(E.analysis == name) & (E.animal == an) & (E.epoch == ep)].excess_strict
        assert len(want) == 1 and np.isclose(v, want.iloc[0]), (name, an, ep)
    print(f"{len(P)} plotted values == effects.csv")


def summarise():
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    R, C = build_table()
    print("\nper-session checks (feat_err = rebuilt vs trial_features features; pool_err = vs the pooled cached z-scored "
          "features):")
    print(C.round(6).to_string(index=False))
    M, b = analysis_frame(R)
    print(f"\nT13 p_true subset {len(b)} trials; matched to rebuilt decoder rows {len(M)}")
    d = (M.p_full - M.p_true).abs()
    print(f"rebuilt full-window p_true vs stored p_true: median |diff| {d.median():.2e}, max {d.max():.2e}, "
          f"frac > 1e-6 {(d > 1e-6).mean():.4f}")
    print("event-in-window trials (ev_win) vs T13 ev_enl (2-s):\n", pd.crosstab(M.ev_enl, M.ev_win))
    print("frames removed (window = %d frames): event trials mean %.1f, matched no-event mean %.1f" % (
        int(M.n_frames.iloc[0]), M[M.ev_win].n_removed.mean(), M[~M.ev_win].n_removed.mean()))
    print("pseudo-mask source:", M[~M.ev_win].pseudo_src.value_counts().to_dict())
    M.to_csv(OUT / "trials.csv.gz", index=False)
    E = effects(M)
    E.to_csv(OUT / "effects.csv", index=False)
    cols = ["animal", "epoch", "n_sessions", "n_trials", "n_event", "mean_event", "mean_noevent", "excess", "p",
            "excess_strict", "p_strict"]
    for name, *_ in ANALYSES:
        print(f"\n=== {name} ===")
        X = E[(E.analysis == name) & E.epoch.isin(["pre", "acute+subacute", "chronic"])]
        print(X[cols].round(4).to_string(index=False))
    figure(E, M)
    (OUT / "README.txt").write_text(__doc__)
    print("wrote", OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--figure", action="store_true", help="re-render the figure from effects.csv / trials.csv")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render from effects.csv / trials.csv -> *_nolines.png (and check the plotted values)")
    a = ap.parse_args()
    if a.plot_only:
        plot_only()
        return
    if a.figure:
        import pandas as pd
        figure(pd.read_csv(OUT / "effects.csv"), pd.read_csv(OUT / "trials.csv.gz", low_memory=False))
        return
    if not a.summarise:
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = a.only or sorted(f.stem for f in (TE / "sessions").glob("*.npz") if f.stem[:4] in ANIMALS)
        res, fail = fan_sessions(labels, extract, jobs=min(a.jobs, 4), label="session")
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
