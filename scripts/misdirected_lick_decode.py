"""Misdirected licks: does cortex look like the PLANNED far-contra lick or the lick the tongue EXECUTED? (Priya
2026-10-19: "in the acute far_contra ... the animal clearly has a lick but it doesn't contact the spout (deviated) --
does that activity look like a planned + executed far right lick or the lick that it executes?")

Per session, from the O2 bundle inputs (imaging + behaviour; `scripts/o2_encoding.py bundle`) and licks.csv (contact):
  features  region-mean activity (Allen areas x hemisphere, eroded stat mask, z-scored) per lick in three windows:
            evoked = mean over [onset, onset + EVOKED_S] minus mean over [onset - BASE_S, onset)   (lick-evoked, primary)
            post   = mean over [onset, onset + EVOKED_S]                                         (not baseline-subtracted)
            pre    = mean over [onset - BASE_S, onset)                                            (trial context only)
  decoder   multinomial logistic regression (standardised, L2) trained on CONTACTED licks of ENGAGED trials, label =
            spout position (6 classes); cross-validated predictions for the training licks (folds grouped by TRIAL)
  probe     MISDIRECTED licks = no contact, at FAR_CONTRA, executed peak angle deviated > DEV deg toward ipsi; for each,
            the EXECUTION-matched position = the FAR spout whose angle is nearest the executed angle (only licks whose
            executed angle is nearer another far spout than far_contra are kept: execution and plan then differ)
  readout   P(far_contra) = the plan / target vs P(exec-matched) = where the tongue went; controls: held-out contacted
            far_contra licks and contacted licks AT the exec-matched position (what each answer looks like), and the
            'pre' window (if misdirected licks already decode as far_contra BEFORE the lick, that is trial context, not
            a lick plan). Pre-stroke far-contra misses = the same analysis without the lesion.
Caveats: contacted licks carry a touch response that misdirected licks lack (both templates have it); widefield GCaMP is
slow relative to a ~7 Hz lick rhythm, so a lick's window overlaps its neighbours (baseline subtraction helps).

    python scripts/misdirected_lick_decode.py [--run ...] [--name enc_v36_20261019] [--dev 15] [--min-probe 15]

Output -> <LP_analysis>/<run>/cohort/tables/misdirected_licks/{decode_licks.csv, decode_sessions.csv, decode.png}
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

EVOKED_S, BASE_S, EARLY_S = 0.2, 0.1, 0.1
POSITIONS = ("close_contra", "close_mid", "close_ipsi", "far_contra", "far_mid", "far_ipsi")
FAR = ("far_contra", "far_mid", "far_ipsi")
EPOCHS = ("pre", "acute", "subacute", "chronic")


def region_traces(im) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(T, n_regions) region-mean activity from the SVD (V @ mean-U per region), the frame times and region names."""
    V, ft, Ub, _flat, parea, names, _shape, pstat = im
    codes = sorted(int(a) for a in np.unique(parea[pstat]) if a != 0)
    A = np.stack([Ub[(parea == c) & pstat].mean(0) for c in codes])          # (n_regions, k)
    X = (V @ A.T).astype(np.float64)
    return X, np.asarray(ft, float), [names[c] for c in codes]


def lick_table(rv, run: str, animal: str, date: str, B) -> pd.DataFrame:
    """Per lick (same order as the Behaviour arrays): onset (DAQ s), trial, position, executed angle, contact, engaged."""
    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    sd = oo.session_dir(rv, run, animal, date)
    cue_t = oc._cue_table(sd).set_index("trial_id")
    L = pd.read_csv(sd / "licks.csv")
    L = L[L.trial_id.isin(cue_t.index)].reset_index(drop=True)
    if len(L) != len(B.lick_on_s):
        raise RuntimeError(f"lick table / Behaviour mismatch {len(L)} vs {len(B.lick_on_s)}")
    ok = (~L.peak_interp.astype(bool)) & (L.protrusion_px >= ce.MIN_REACH_PX) & np.isfinite(L.peak_angle_deg)
    eng = set(B.trials.trial_id[B.trials.engaged].astype(int))
    return pd.DataFrame({"on_s": B.lick_on_s, "trial_id": L.trial_id.to_numpy(int), "pos": B.lick_pos,
                         "angle": np.where(ok, L.peak_angle_deg.to_numpy(float), np.nan),
                         "contact": L.contact.astype(bool).to_numpy(), "ok": ok.to_numpy(),
                         "engaged": L.trial_id.isin(eng).to_numpy(),
                         "reach": L.protrusion_px.to_numpy(float),
                         "lick_no": L.assign(_t=on_s_order(L)).groupby("trial_id")._t.rank(method="first")
                         .to_numpy(int) - 1})


def on_s_order(L: pd.DataFrame) -> np.ndarray:
    """Lick onset within the trial (ms) for ordering licks; peak time where onset is missing."""
    return np.where(np.isfinite(L.on_ms.to_numpy(float)), L.on_ms.to_numpy(float), L.t_ms.to_numpy(float))


def lick_features(X: np.ndarray, ft: np.ndarray, on_s: np.ndarray) -> dict:
    fs = 1.0 / np.median(np.diff(ft))
    ne, nb = max(1, int(round(EVOKED_S * fs))), max(1, int(round(BASE_S * fs)))
    i0 = np.searchsorted(ft, on_s)
    good = (i0 - nb >= 0) & (i0 + ne < len(ft))
    post = np.full((len(on_s), X.shape[1]), np.nan)
    pre = post.copy()
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    ii = i0[good]
    post[good] = (cs[ii + ne] - cs[ii]) / ne
    pre[good] = (cs[ii] - cs[ii - nb]) / nb
    # EARLY = the first EARLY_S after visible onset minus the same baseline: contact comes ~50-70 ms after onset and
    # GCaMP lags, so this window is largely pre-TOUCH (Priya 10-19: the touch asymmetry of contacted templates)
    nl = max(1, int(round(EARLY_S * fs)))
    early = post.copy()
    early[good] = (cs[ii + nl] - cs[ii]) / nl - pre[good]
    return {"evoked": post - pre, "post": post, "pre": pre, "early": early}


def decode_session(L: pd.DataFrame, F: dict, tgt: dict, dev: float, seed: int = 0) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fin = np.isfinite(F["evoked"]).all(1)
    base = L.engaged & L.ok & fin
    train = base & L.contact & L.pos.isin(POSITIONS)
    fc = tgt.get("far_contra")
    if fc is None or train.sum() < 60:
        return pd.DataFrame()
    d = L.angle - fc                                                     # + = toward ipsi (camera convention)
    far_t = {p: tgt[p] for p in FAR if p in tgt}
    near = L.angle.apply(lambda a: min(far_t, key=lambda p: abs(a - far_t[p])) if np.isfinite(a) else None)
    probe = base & ~L.contact & (L.pos == "far_contra") & (d > dev) & (near != "far_contra")
    pre_nc = base & ~L.contact & (L.pos == "far_contra") & (d.abs() <= dev)     # no contact but on target (short)
    out = []
    classes = [p for p in POSITIONS if (train & (L.pos == p)).sum() >= 10]
    tr = train & L.pos.isin(classes)
    y, g = L.pos[tr].to_numpy(), L.trial_id[tr].to_numpy()
    for feat in ("evoked", "post", "pre"):
        Xf = F[feat]
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        cvp = np.zeros((tr.sum(), len(classes)))
        nf = min(5, len(np.unique(g)))
        for a_, b_ in GroupKFold(nf).split(Xf[tr], y, g):
            m = mdl.fit(Xf[tr][a_], y[a_])
            cvp[b_] = m.predict_proba(Xf[tr][b_])[:, [list(m.classes_).index(c) for c in classes]]
        m = mdl.fit(Xf[tr], y)
        cols = [list(m.classes_).index(c) for c in classes]

        def rows(mask, kind, P, feat=feat):
            idx = np.flatnonzero(mask)
            for j, i in enumerate(idx):
                out.append({"feat": feat, "kind": kind, "pos": L.pos[i], "angle": L.angle[i], "trial_id": L.trial_id[i],
                            "exec_pos": near[i], **{f"p_{c}": P[j, k] for k, c in enumerate(classes)}})
        rows(tr.to_numpy(), "contacted_cv", cvp)
        for mask, kind in ((probe, "misdirected"), (pre_nc, "nocontact_on_target")):
            if mask.sum():
                rows(mask.to_numpy(), kind, m.predict_proba(Xf[mask.to_numpy()])[:, cols])
    return pd.DataFrame(out)


def session_licks(rv, run: str, lab: str, bdir, dev: float) -> pd.DataFrame | None:
    """Per usable ENGAGED lick of one session: position, contact, executed angle, kind (contacted / misdirected =
    far_contra no-contact deviated > dev toward ipsi and nearer another far spout / other), exec-matched far spout, and
    the lick-evoked region pattern (columns r_<region>)."""
    pk = bdir / f"{lab}.pkl"
    if not pk.exists():
        return None
    x = pickle.loads(pk.read_bytes())
    B = x["behaviour"]
    X, ft, names = region_traces(x["imaging"])
    eng = np.zeros(len(ft), bool)
    for _, t in B.trials[B.trials.engaged].iterrows():
        eng |= (ft >= t.strobe_s) & (ft <= t.stop_s)
    X = (X - X[eng].mean(0)) / X[eng].std(0)
    L = lick_table(rv, run, lab.split("_")[0], "2026" + lab[5:], B)
    FF = lick_features(X, ft, L.on_s.to_numpy())
    F, Fq, Fe = FF["evoked"], FF["pre"], FF["early"]
    tgt = B.target_angle or {}
    keep = (L.engaged & L.ok & np.isfinite(F).all(1)).to_numpy()
    L, F, Fq, Fe = L[keep].reset_index(drop=True), F[keep], Fq[keep], Fe[keep]
    far_t = {p: tgt[p] for p in FAR if p in tgt}
    near = L.angle.apply(lambda a: min(far_t, key=lambda p: abs(a - far_t[p])) if far_t else None)
    d = L.angle - tgt.get("far_contra", np.nan)
    own = L.angle - L.pos.map(tgt).astype(float)
    mis = ~L.contact & (L.pos == "far_contra") & (d > dev) & (near != "far_contra")
    nc_on = ~L.contact & (own.abs() <= dev)                 # no contact, but aimed at its own spout (short / glancing)
    kind = np.where(L.contact, "contacted", np.where(mis, "misdirected", np.where(nc_on, "nocontact", "other")))
    R = pd.DataFrame(F, columns=[f"r_{n}" for n in names])
    Q = pd.DataFrame(Fq, columns=[f"q_{n}" for n in names])
    E = pd.DataFrame(Fe, columns=[f"e_{n}" for n in names])
    return pd.concat([L[["trial_id", "pos", "angle", "reach", "lick_no"]].assign(kind=kind, exec_pos=near.to_numpy()),
                      R, Q, E], axis=1)


def matched(L: pd.DataFrame, probe: pd.Series, pool: pd.Series, tol_px: float = 15.0) -> pd.Series:
    """Comparison licks matched to the probe licks: for each probe lick, the pool lick with the same lick number
    (0 / 1) and the nearest reach (within tol_px), with replacement. Returns a count per pool lick (weights)."""
    w = pd.Series(0, index=L.index)
    P = L[pool]
    for i in np.flatnonzero(probe.to_numpy()):
        c = P[P.lick_no == L.lick_no[i]]
        if c.empty:
            continue
        j = (c.reach - L.reach[i]).abs().idxmin()
        if abs(c.reach[j] - L.reach[i]) <= tol_px:
            w[j] += 1
    return w


def _pc(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 else np.nan


def template_analysis(Ls: dict, meta: dict, min_n: int = 10, prefix: str = "r_") -> pd.DataFrame:
    """Per animal, PRE-stroke contacted-lick templates per far spout (mean lick-evoked region pattern), centred on the
    mean over the far spouts so only the position-SPECIFIC part is compared. Per test set (a session's misdirected
    licks, split by their exec-matched spout; and controls), r(pattern, T_far_contra) vs r(pattern, T_exec). Pre
    sessions are tested against templates built WITHOUT them (leave-one-session-out)."""
    rows = []
    for an in sorted({m["animal"] for m in meta.values()}):
        labs = [lb for lb in Ls if meta[lb]["animal"] == an]
        pre = [lb for lb in labs if meta[lb]["epoch"] == "pre"]
        if len(pre) < 2:
            continue
        rc = [c for c in Ls[labs[0]].columns if c.startswith(prefix)]

        def templates(excl, pre=pre, rc=rc):
            T = {}
            for p in FAR:
                pats = [Ls[lb].loc[(Ls[lb].kind == "contacted") & (Ls[lb].pos == p), rc].mean().to_numpy()
                        for lb in pre if lb != excl and ((Ls[lb].kind == "contacted") & (Ls[lb].pos == p)).sum() >= min_n]
                if pats:
                    T[p] = np.nanmean(pats, 0)
            if len(T) < 2:
                return None, None
            c = np.nanmean(list(T.values()), 0)
            return {p: v - c for p, v in T.items()}, c

        for lb in labs:
            T, c = templates(lb if lb in pre else None)
            if T is None or "far_contra" not in T:
                continue
            L = Ls[lb]
            sets = []
            for ex in ("far_mid", "far_ipsi"):
                mis = (L.kind == "misdirected") & (L.exec_pos == ex)
                cont = (L.kind == "contacted") & (L.pos == ex)
                sets.append((f"misdirected->{ex}", mis, ex))
                sets.append((f"contacted {ex}", cont, ex))
                sets.append((f"nocontact {ex}", (L.kind == "nocontact") & (L.pos == ex), ex))
                # FIRST ATTEMPTS + MATCHED (Priya 10-19: no-contact comparisons are short; misdirected licks are mostly
                # the 1st / 2nd lick): misdirected licks 0 / 1 vs contacted licks at the landing spot matched on lick
                # number and reach
                mis01 = mis & (L.lick_no <= 1)
                sets.append((f"misdirected01->{ex}", mis01, ex))
                sets.append((f"matched {ex}", matched(L, mis01, cont & (L.lick_no <= 1)), ex))
            sets.append(("contacted far_contra", (L.kind == "contacted") & (L.pos == "far_contra"), "far_mid"))
            for name, msk, ex in sets:
                w = msk.astype(float)
                if (w > 0).sum() < min_n or ex not in T:
                    continue
                pat = (L[rc].to_numpy() * w.to_numpy()[:, None]).sum(0) / w.sum() - c
                rows.append({"label": lb, "animal": an, "epoch": meta[lb]["epoch"], "set": name,
                             "n": int((w > 0).sum()), "exec_pos": ex, "r_target": _pc(pat, T["far_contra"]),
                             "r_exec": _pc(pat, T[ex])})
    D = pd.DataFrame(rows)
    if not D.empty:
        D["exec_minus_target"] = D.r_exec - D.r_target
    return D


def summarise(D: pd.DataFrame) -> pd.DataFrame:
    """Per session x feature: P(far_contra) and P(exec-matched) for misdirected licks, and the same two probabilities for
    held-out contacted far_contra licks and contacted licks AT each misdirected lick's exec-matched position."""
    rows = []
    for (lab, feat), g in D.groupby(["label", "feat"]):
        mis = g[g.kind == "misdirected"]
        if mis.empty:
            continue
        cv = g[g.kind == "contacted_cv"]
        ep = mis.exec_pos.value_counts().idxmax()                         # the usual exec-matched position
        r = {"label": lab, "animal": g.animal.iloc[0], "epoch": g.epoch.iloc[0], "feat": feat, "n_mis": len(mis),
             "exec_pos": ep, "mis_p_target": mis.p_far_contra.mean(),
             "mis_p_exec": np.nanmean([row[f"p_{row.exec_pos}"] if f"p_{row.exec_pos}" in row else np.nan
                                       for _, row in mis.iterrows()])}
        for name, sub in (("fc", cv[cv.pos == "far_contra"]), ("ex", cv[cv.pos == ep])):
            r[f"{name}_n"] = len(sub)
            r[f"{name}_p_target"] = sub.p_far_contra.mean() if len(sub) else np.nan
            r[f"{name}_p_exec"] = sub[f"p_{ep}"].mean() if len(sub) and f"p_{ep}" in sub else np.nan
        cls = [c for c in g.columns if c.startswith("p_")]
        r["cv_acc"] = (cv[cls].idxmax(1).str[2:] == cv.pos).mean()
        r["chance"] = 1 / len([c for c in cls if cv[c].notna().any()])
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import o2_inference as oi
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--name", default="enc_v36_20261019", help="O2 bundle holding the session inputs")
    ap.add_argument("--dev", type=float, default=15.0)
    ap.add_argument("--min-probe", type=int, default=15, help="sessions with fewer misdirected licks are skipped")
    ap.add_argument("--method", choices=("templates", "decoder"), default="templates",
                    help="templates (primary): pre-stroke contacted-lick templates per animal; decoder: within-session "
                         "logistic decoder (fails acutely: too few contacted far_contra licks to learn the class)")
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = oo.root(rv) / a.run / "cohort" / "tables" / "misdirected_licks"
    writeguard.assert_writable(out)
    cnt = pd.read_csv(out / "counts.csv")
    n_mis = cnt[cnt.pos == "far_contra"].groupby("label").ipsi_dev.sum()
    bdir = oi.bundle_dir(rv, a.name) / "inputs"
    if a.method == "templates":
        return main_templates(a, rv, out, bdir)
    labels = a.labels or [s["label"] for s in ak.curated_sessions() if n_mis.get(s["label"], 0) >= a.min_probe]
    parts = []
    for lab in labels:
        ep = epochs.epoch_of(lab)
        if ep not in EPOCHS or (ep == "pre" and lab[5:] < oc.LATE_PRE_FROM):
            continue
        pk = bdir / f"{lab}.pkl"
        if not pk.exists():
            print(f"{lab}: no bundle inputs yet -- skipped", flush=True)
            continue
        x = pickle.loads(pk.read_bytes())
        B = x["behaviour"]
        X, ft, _names = region_traces(x["imaging"])
        eng_frames = np.zeros(len(ft), bool)
        for _, t in B.trials[B.trials.engaged].iterrows():
            eng_frames |= (ft >= t.strobe_s) & (ft <= t.stop_s)
        X = (X - X[eng_frames].mean(0)) / X[eng_frames].std(0)
        L = lick_table(rv, a.run, lab.split("_")[0], "2026" + lab[5:], B)
        F = lick_features(X, ft, L.on_s.to_numpy())
        D = decode_session(L, F, B.target_angle or {}, a.dev)
        if D.empty:
            print(f"{lab}: too few licks -- skipped", flush=True)
            continue
        D = D.assign(label=lab, animal=lab.split("_")[0], epoch=ep)
        parts.append(D)
        m = D[(D.kind == "misdirected") & (D.feat == "evoked")]
        print(f"{lab} {ep}: misdirected {len(m)}, P(far_contra) {m.p_far_contra.mean():.2f}", flush=True)
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(out / "decode_licks.csv", index=False, float_format="%.4g")
    S = summarise(D)
    S.to_csv(out / "decode_sessions.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    for feat in ("evoked", "post", "pre"):
        T = S[S.feat == feat].groupby(["animal", "epoch"])[
            ["n_mis", "mis_p_target", "mis_p_exec", "fc_p_target", "ex_p_exec", "ex_p_target", "cv_acc"]].mean()
        print(f"\n== {feat}: per animal x epoch (mean over sessions)\n" + T.round(3).to_string())
    _figure(S, out / "decode.png", a.run, a.dev)
    print(f"-> {out}")
    return 0


def main_templates(a, rv, out, bdir) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    cache = out / "lick_patterns"
    cache.mkdir(parents=True, exist_ok=True)
    Ls, meta = {}, {}
    for s in ak.curated_sessions():
        lab = s["label"]
        ep = epochs.epoch_of(lab)
        if ep not in EPOCHS or (ep == "pre" and lab[5:] < oc.LATE_PRE_FROM) or (a.labels and lab not in a.labels):
            continue
        cf = cache / f"{lab}_dev{a.dev:g}_v3.pkl"
        if cf.exists():
            L = pd.read_pickle(cf)
        else:
            L = session_licks(rv, a.run, lab, bdir, a.dev)
            if L is None:
                print(f"{lab}: no bundle inputs -- skipped", flush=True)
                continue
            L.to_pickle(cf)
        Ls[lab], meta[lab] = L, {"animal": lab.split("_")[0], "epoch": ep}
        print(f"{lab} {ep}: contacted {int((L.kind == 'contacted').sum())}, misdirected "
              f"{int((L.kind == 'misdirected').sum())}", flush=True)
    # window 'evoked' = lick-evoked (post - pre, primary); 'pre' = the 100 ms BEFORE onset (trial context control)
    D = pd.concat([template_analysis(Ls, meta, prefix=p).assign(window=w) for w, p in (("evoked", "r_"), ("early", "e_"), ("pre", "q_"))],
                  ignore_index=True)
    D.to_csv(out / "template_match.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    T = D.groupby(["window", "set", "animal", "epoch"])[["n", "r_target", "r_exec", "exec_minus_target"]].mean()
    print("\nper window x set x animal x epoch (mean over sessions): r with the far_contra template (plan / target) vs "
          "with the exec-matched spout's template (execution)\n" + T.round(3).to_string())
    S = shift_table(D)
    S.to_csv(out / "template_shift.csv", index=False, float_format="%.4g")
    print("\nSHIFT = score(misdirected) - score(comparison licks at the exec-matched spot); score = r(exec) - r(far_contra)"
          "; < 0 = misdirected look more like far_contra than the comparison does. comparison 'contacted' (touch "
          "asymmetry) or 'nocontact' (no touch on either side)\n" + S.round(3).to_string(index=False))
    _figure_templates(D[D.window == "evoked"], out / "template_match.png", a.run, a.dev)
    print(f"-> {out / 'template_match.csv'}")
    return 0


def shift_table(D: pd.DataFrame) -> pd.DataFrame:
    """Per window x landing spot x comparison x animal x epoch: misdirected score minus the comparison set's score."""
    # per SESSION (both sets present), then mean over the animal's sessions
    A = D.set_index(["window", "set", "label"]).exec_minus_target
    meta = D.drop_duplicates("label").set_index("label")[["animal", "epoch"]]
    rows = []
    for (w, s, lb), v in A.items():
        if not s.startswith("misdirected"):
            continue
        probe, ex = s.split("->")
        for comp in (("matched",) if probe == "misdirected01" else ("contacted", "nocontact")):
            k = (w, f"{comp} {ex}", lb)
            if k in A.index:
                rows.append({"window": w, "landed": ex, "probe": probe, "vs": comp, "label": lb,
                             "animal": meta.animal[lb], "epoch": meta.epoch[lb], "mis": v, "comp": A[k],
                             "shift": v - A[k]})
    S = pd.DataFrame(rows)
    return (S.groupby(["window", "landed", "probe", "vs", "epoch", "animal"])
            .agg(sessions=("label", "size"), mis=("mis", "mean"), comp=("comp", "mean"), shift=("shift", "mean"))
            .reset_index())


def _figure_templates(D: pd.DataFrame, path, run: str, dev: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config
    from wfield_local.orofacial_cohort import save_png_svg
    acol = config.animal_color()
    sets = [s for s in ("contacted far_contra", "misdirected->far_mid", "contacted far_mid", "misdirected->far_ipsi",
                        "contacted far_ipsi") if s in set(D.set)]
    eps = [e for e in EPOCHS if e in set(D.epoch)]
    fig, axs = plt.subplots(1, len(sets), figsize=(3.9 * len(sets), 4.4), squeeze=False, sharey=True)
    for c, s in enumerate(sets):
        ax = axs[0][c]
        G = D[D.set == s]
        for ei, e in enumerate(eps):
            a = G[G.epoch == e].groupby("animal").exec_minus_target.mean()
            for k, (an, v) in enumerate(a.items()):
                ax.plot(ei + (k - 1.5) * 0.1, v, "o", ms=6, color=acol.get(an, "0.5"), mec="k", mew=0.5,
                        label=an if (c == 0 and ei == 0) else None)
            if len(a):
                ax.plot([ei - 0.25, ei + 0.25], [a.mean()] * 2, "-", color="k", lw=2)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels(eps)
        ex = s.split("->")[-1].replace("contacted ", "")
        ax.set_title(f"{s}\n(> 0 = like {'the exec-matched spout' if 'far_contra' in s else ex}, "
                     "< 0 = like far_contra)", fontsize=9.5)
        if c == 0:
            ax.set_ylabel("r(exec template) - r(far_contra template)")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Misdirected far-contra licks (no contact, > {dev:.0f} deg toward ipsi, nearer another far spout) vs "
                 f"PRE-stroke contacted-lick templates per animal (lick-evoked region pattern, position-specific part) "
                 f"-- {run}. Pre sessions tested against templates built without them. Dots = animals, bar = mean.",
                 fontsize=10, y=0.995, va="top", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_png_svg(fig, path, dpi=130)
    plt.close(fig)


def _figure(S: pd.DataFrame, path, run: str, dev: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config
    from wfield_local.orofacial_cohort import save_png_svg
    acol = config.animal_color()
    feats = [f for f in ("evoked", "post", "pre") if f in set(S.feat)]
    eps = [e for e in EPOCHS if e in set(S.epoch)]
    fig, axs = plt.subplots(1, len(feats), figsize=(5.6 * len(feats), 4.8), squeeze=False)
    groups = [("fc_p_target", "contacted far_contra\nP(far_contra)", "s"),
              ("mis_p_target", "MISDIRECTED\nP(far_contra) = plan", "o"),
              ("mis_p_exec", "MISDIRECTED\nP(exec-matched) = execution", "^"),
              ("ex_p_exec", "contacted at exec-matched\nP(exec-matched)", "D")]
    for c, feat in enumerate(feats):
        ax = axs[0][c]
        G = S[S.feat == feat]
        for k, (col, _lab, mk) in enumerate(groups):
            for ei, e in enumerate(eps):
                a = G[G.epoch == e].groupby("animal")[col].mean()
                x = ei + (k - 1.5) * 0.18
                for an, v in a.items():
                    ax.plot(x, v, mk, ms=6, color=acol.get(an, "0.5"), mec="k", mew=0.5, alpha=0.9)
                ax.plot([x - 0.07, x + 0.07], [a.mean()] * 2, "-", color="k", lw=2)
        ch = G.chance.mean()
        ax.axhline(ch, color="0.6", ls="--", lw=1, label=f"chance {ch:.2f}")
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels(eps)
        ax.set_ylim(0, 1)
        ax.set_ylabel("decoder probability")
        ax.set_title({"evoked": "lick-EVOKED (post - pre)", "post": "post-onset (no baseline)",
                      "pre": "PRE-lick only (trial context)"}[feat], fontsize=11)
        if c == 0:
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], marker=mk, ls="", color="0.4", label=lab) for _, lab, mk in groups]
                      + [Line2D([], [], ls="--", color="0.6", label="chance")], frameon=False, fontsize=8,
                      loc="upper right")
    fig.suptitle(f"Misdirected far-contra licks (no contact, deviated > {dev:.0f} deg toward ipsi, nearer another far "
                 f"spout): decoder trained on contacted licks (6 positions) -- {run}. Markers = animals (colour), bar = "
                 "mean of animals; sessions with too few misdirected licks skipped.", fontsize=10, y=0.995, va="top",
                 wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
