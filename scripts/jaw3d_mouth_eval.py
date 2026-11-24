"""EVALUATION of the jaw rule Priya adopted on 11-23 (CC-B): `trial_classes.jaw_call(mode="3d_mouth")` = the four-camera
3-D jaw call (CAP variant + 3-D re-onset) OR an IN-MOUTH LICK (kept cam1 event not matched to the fused four-camera
list; `trial_classes.mouth_lick_events`), onset = the earlier of the two.

    python scripts/jaw3d_mouth_eval.py classes [--jobs 6] [--labels ...]   # per-trial m3d classes + in-mouth windows
    python scripts/jaw3d_mouth_eval.py summary                             # clips, kappa, pseudo-cue, counts, adds
    python scripts/jaw3d_mouth_eval.py s10a                                # S10A decomposition with m3d as THE jaw
    python scripts/jaw3d_mouth_eval.py blind                               # INDEX_for_Priya_v2.csv + reveal_mouth key

Compared with the rules of `scripts/jaw3d_fix_eval.py` (reused from its fix_eval/classes_all_rules.csv.gz, not
recomputed): cur2d (production classes), cur3d, cap2d, cap3d (cap + 3-D re-onset). New rule: m3d.
Writes only under <review>/jaw3d_qc/fix_eval_mouth/ (and INDEX_for_Priya_v2.csv in blind_sixpos/).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REVIEW = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ccB_caption_review_2026-11-23")
JQ = REVIEW / "jaw3d_qc"
FIX = JQ / "fix_eval"
OUT = JQ / "fix_eval_mouth"
BLIND = JQ / "blind_sixpos"
QC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/qc")
CAP = JQ / "jaw_onsets_cap.csv"
POS = {"close_L": "near ipsi", "close_center": "near middle", "close_R": "near contra",
       "far_L": "far ipsi", "far_center": "far middle", "far_R": "far contra"}
POS_ORDER = list(POS.values())
EPOCHS = ["pre", "acute", "subacute", "chronic"]
NOLICK = ("jaw_only", "no_response", "unknown")
CLIPS = {"clip15": ("PS95_0817", 69), "clip28": ("PS92_0809", 209), "clip29": ("PS93_0807", 325),
         "clip30": ("PS94_0821", 328), "clip31": ("PS95_1005", 482)}


def session_job(label: str) -> pd.DataFrame | None:
    from wfield_local import session_licks as SL
    from wfield_local import trial_classes as tc
    T = tc.session_trials(label, jaw="3d_mouth", jaw_csv=CAP)
    if T is None:
        return None
    out = T[["trial_id", "position", "frame_gated", "n_contacts", "n_licks", "cls", "cls_detail", "jaw_onset_ms",
             "jaw_judged", "jaw_mouth_ms", "cue_daq", "stop_ms"]].rename(
        columns={"cls": "cls_m3d", "cls_detail": "det_m3d", "jaw_onset_ms": "on_m3d", "jaw_judged": "jud_m3d"})
    E = tc.mouth_lick_events(label)
    cue = T.cue_daq.to_numpy(float)
    # in-mouth licks in the two 1-s pseudo-cue test windows (onset): pseudo = -1000 .. 0 ms, real = 0 .. 1000 ms
    out["mouth_pseudo_ms"] = tc.mouth_lick_ms(E, cue - 1.0, np.full(len(T), 1000.0 - 1e-3)) - 1000.0
    out["mouth_real_ms"] = tc.mouth_lick_ms(E, cue, np.full(len(T), 1000.0))
    out["n_mouth_win"] = 0
    if E is not None and len(E):
        t = np.sort(E.daq_on_s.to_numpy(float))
        end = tc.response_end_ms(T.stop_ms.to_numpy(float)) / 1000.0
        out["n_mouth_win"] = np.searchsorted(t, cue + end, "right") - np.searchsorted(t, cue, "left")
        out["n_mouth_pre15"] = np.searchsorted(t, cue, "left") - np.searchsorted(t, cue - 1.5, "left")
    else:
        out["n_mouth_pre15"] = 0
    # pseudo-cue exclusion B: a fused LICK (cam1-only rows excluded) with its peak in -1500 .. 0 ms
    L = SL.session_licks(label)
    pk = pd.to_numeric(L.t_ms, errors="coerce")
    pre = set(L.trial_id[(pk >= -1500) & (pk < 0)].astype(int))
    out["precue_lick_nocam1"] = out.trial_id.isin(pre)
    out.insert(0, "label", label)
    return out


def classes(jobs: int, labels=None) -> None:
    from wfield_local import analysis_kit as ak
    labels = labels or sorted(pd.read_csv(QC / "s1_trial_classes_cue_stop.csv.gz", usecols=["label"]).label.unique())
    res, fails = ak.fan_sessions(labels, session_job, jobs=jobs)
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.concat([r for _, r in res if r is not None], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "classes_m3d" + ("_subset" if labels and len(labels) < 100 else "") + ".csv.gz"
    T.to_csv(OUT / name, index=False, float_format="%.6g")
    print(f"{len(T)} trials, {T.label.nunique()} sessions -> {OUT / name}")


def load() -> pd.DataFrame:
    A = pd.read_csv(FIX / "classes_all_rules.csv.gz", low_memory=False)
    M = pd.read_csv(OUT / "classes_m3d.csv.gz", low_memory=False)
    T = A.merge(M.drop(columns=["position", "frame_gated", "n_contacts", "n_licks"]), on=["label", "trial_id"],
                how="left", validate="one_to_one")
    assert T.cls_m3d.notna().all(), "m3d missing for some trials"
    return T


def _decided(T, a, b):
    return T[f"det_{a}"].isin(NOLICK) & T[f"det_{b}"].isin(NOLICK) & ~T.frame_gated.astype(bool)


def _kappa(x, y):
    po = float((x == y).mean())
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def _sort(D):
    D = D.copy()
    D["pos"] = pd.Categorical(D.pos, POS_ORDER)
    keys = ["pos"]
    if "epoch" in D:
        D["epoch"] = pd.Categorical(D.epoch, EPOCHS)
        keys.append("epoch")
    return D.sort_values(keys)


def summary() -> None:
    T = load()
    T = T[T.epoch.isin(EPOCHS)].copy()
    lines = []
    pd.set_option("display.width", 250)
    # 0. checks: m3d without the in-mouth lick must equal cap3d (same table, same re-onset)
    nomouth = ~np.isfinite(T.jaw_mouth_ms)
    same = (T.loc[nomouth, "cls_m3d"] == T.loc[nomouth, "cls_cap3d"]).mean()
    lines += [f"## check: m3d == cap3d on trials with no in-mouth lick in the window: {same:.5f} "
              f"({int(nomouth.sum())} trials); trials with an in-mouth lick in cue..end: {int((~nomouth).sum())}", ""]
    # 1. what the in-mouth lick adds, per spout x epoch: all trials, and no-lick (jaw-decided) trials
    mo = np.isfinite(T.jaw_mouth_ms)
    cap_on = T.on_cap3d.to_numpy(float)
    cap_mv = T.det_cap3d.isin(["jaw_only"]) | (np.isfinite(cap_on) & T.jud_cap3d.astype(bool))
    T["mouth_adds"] = mo & ~cap_mv
    T["mouth_earlier"] = mo & np.isfinite(cap_on) & (T.jaw_mouth_ms < cap_on)
    T["dec_m"] = _decided(T, "cap3d", "m3d")
    rows = []
    for (p, e), g in T.groupby(["pos", "epoch"]):
        d = g[g.dec_m]
        rows.append({"pos": p, "epoch": e, "n": len(g), "mouth_in_window": int(np.isfinite(g.jaw_mouth_ms).sum()),
                     "adds_call": int(g.mouth_adds.sum()), "earlier_onset": int(g.mouth_earlier.sum()),
                     "n_decided": len(d), "decided_mouth": int(np.isfinite(d.jaw_mouth_ms).sum()),
                     "decided_adds": int(d.mouth_adds.sum()), "decided_earlier": int(d.mouth_earlier.sum())})
    Ad = _sort(pd.DataFrame(rows))
    Ad.to_csv(OUT / "mouth_adds_pos_epoch.csv", index=False)
    lines += ["## in-mouth lick: adds a call the 3-D (cap + re-onset) missed / sets an earlier onset", Ad.to_string(index=False), ""]
    # earlier-onset size
    dd = (T.on_cap3d - T.jaw_mouth_ms)[T.mouth_earlier]
    lines += [f"   earlier by: median {dd.median():.0f} ms, IQR {dd.quantile(.25):.0f}-{dd.quantile(.75):.0f} (n {len(dd)})", ""]
    # 2. class changes
    X = pd.crosstab(T.det_cur2d, T.det_m3d, margins=True)
    X.to_csv(OUT / "class_change_cur2d_vs_m3d.csv")
    Y = pd.crosstab(T.det_cap3d, T.det_m3d, margins=True)
    Y.to_csv(OUT / "class_change_cap3d_vs_m3d.csv")
    lines += ["## class (cls_detail): production cur2d (rows) x m3d (cols)", X.to_string(), "",
              "## cap3d (rows) x m3d (cols)", Y.to_string(), ""]
    # 3. jaw-only counts per spout x epoch
    rules = ["cur2d", "cur3d", "cap2d", "cap3d", "m3d"]
    C = pd.concat({r: T[T[f"det_{r}"] == "jaw_only"].groupby(["pos", "epoch"]).size() for r in rules}, axis=1)
    C = _sort(C.fillna(0).astype(int).reset_index())
    C.to_csv(OUT / "jaw_only_counts_pos_epoch.csv", index=False)
    Cp = pd.concat({r: T[T[f"det_{r}"] == "jaw_only"].groupby("pos").size() for r in rules}, axis=1).reindex(POS_ORDER)
    U = pd.concat({r: T[T[f"cls_{r}"] == "unknown"].groupby("pos").size() for r in rules}, axis=1).reindex(POS_ORDER)
    lines += ["## jaw-only (jaw-called) per spout x epoch", C.to_string(index=False), "", "## per spout", Cp.to_string(),
              f"total {Cp.sum().to_dict()}", "", "## unknown per spout", U.fillna(0).astype(int).to_string(), ""]
    # 4. agreement / kappa on jaw-decided trials (two-camera spouts)
    rows = []
    for a, b in [("cur2d", "cur3d"), ("cap2d", "cap3d"), ("cur2d", "m3d"), ("cap2d", "m3d"), ("cap3d", "m3d")]:
        D = T[_decided(T, a, b) & T[f"cls_{a}"].isin(["jaw_only", "no_response"])
              & T[f"cls_{b}"].isin(["jaw_only", "no_response"]) & (T.position != "close_center")]
        for (p, e), g in D.groupby(["pos", "epoch"]):
            po, k = _kappa(g[f"cls_{a}"] == "jaw_only", g[f"cls_{b}"] == "jaw_only")
            rows.append({"pair": f"{a} vs {b}", "pos": p, "epoch": e, "n": len(g), "agree": round(po, 3),
                         "kappa": round(k, 3)})
    A = _sort(pd.DataFrame(rows))
    A.to_csv(OUT / "agreement_pos_epoch.csv", index=False)
    W = A.pivot_table(index=["pos", "epoch"], columns="pair", values="kappa", observed=True)
    S = A.groupby("pair").kappa.agg(["min", "median", "max"])
    lines += ["## kappa, jaw-decided", W.round(2).to_string(), "", S.round(2).to_string(), ""]
    # 5. onset 3-D-alone vs m3d, and vs 2-D, on jaw-decided both jaw-only
    rows = []
    for a, b in [("cur2d", "cap3d"), ("cur2d", "m3d"), ("cap2d", "m3d")]:
        D = T[_decided(T, a, b) & (T[f"cls_{a}"] == "jaw_only") & (T[f"cls_{b}"] == "jaw_only")
              & np.isfinite(T[f"on_{a}"]) & np.isfinite(T[f"on_{b}"]) & (T.position != "close_center")]
        for p, g in D.groupby("pos"):
            d = g[f"on_{b}"] - g[f"on_{a}"]
            rows.append({"pair": f"{b} - {a}", "pos": p, "n": len(d), "median": d.median(),
                         "within40": (d.abs() <= 40).mean()})
    O = _sort(pd.DataFrame(rows))
    O.round(3).to_csv(OUT / "onset_diff_decided.csv", index=False)
    lines += ["## onset difference, jaw-decided both jaw-only", O.round(3).to_string(index=False), ""]
    lines += pseudo(T)
    lines += clips(T)
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def pseudo(T: pd.DataFrame) -> list[str]:
    """Pseudo-cue test: the 1-s no-cue window (-1000 .. 0 ms, its own baseline -1500 .. -1000) vs the 1-s window after
    the real cue. m3d call = the cap 3-D call (re-onset taken at cue) OR an in-mouth lick in the window. Exclusion A =
    the section-9 set (no fused lick INCLUDING cam1-only events in -1.5 .. 0 s: in-mouth licks there drop the trial);
    exclusion B = no fused LICK (cam1-only rows excluded) in -1.5 .. 0 s, so spontaneous in-mouth licks stay in."""
    N = pd.read_csv(JQ / "floor_null_variants.csv.gz", keep_default_na=False, na_values=[""], low_memory=False)
    N = N[N.variant.isin(["prod", "cap"])]
    N["moved"] = pd.to_numeric(N.moved, errors="coerce")
    at = N.at_cue.astype(str).str.lower().eq("true")
    re_ = N.has_reonset.astype(str).str.lower().eq("true")
    reon_ok = np.where((N.signal == "3d") & (N.variant == "prod"), False, re_)
    N["call"] = np.where(N.moved.notna() & (~at | reon_ok), (N.moved == 1).astype(float), np.nan)
    N["precueA"] = N.precue_lick.astype(str).str.lower().eq("true")
    N["sig"] = np.where(N.signal == "3d", "3d", "2d")
    G = N.groupby(["label", "trial_id", "variant", "window", "sig"]).agg(
        call=("call", lambda s: np.nan if s.isna().all() else float((s == 1).any())),
        precueA=("precueA", "first")).reset_index()
    pa = G.groupby(["label", "trial_id"]).precueA.max()      # one flag per trial (same for every signal)
    P = G.set_index(["label", "trial_id", "variant", "window", "sig"]).call.unstack("sig").reset_index()
    P = P.rename(columns={"3d": "call3d", "2d": "call2d"})
    P["precueA"] = pd.MultiIndex.from_frame(P[["label", "trial_id"]]).map(pa).to_numpy(bool)
    assert not P.duplicated(["label", "trial_id", "variant", "window"]).any()
    K = T[["label", "trial_id", "pos", "epoch", "mouth_pseudo_ms", "mouth_real_ms", "precue_lick_nocam1",
           "n_mouth_pre15"]].copy()
    K["dec"] = _decided(T, "cur2d", "cur3d").to_numpy()
    # SYMMETRIC no-lick set: no contact and no fused LICK after the cue (cam1-only in-mouth licks allowed, i.e. the
    # tongue_at_lips trials kept), not frame-gated: an in-mouth lick can then count in BOTH windows
    K["noLick_lips"] = (T.det_m3d.isin(NOLICK + ("tongue_at_lips",)) & ~T.frame_gated.astype(bool)).to_numpy()
    P = P.merge(K, on=["label", "trial_id"], how="inner")
    P = P[P.epoch.isin(EPOCHS)]
    mo = np.where(P.window == "pseudo", np.isfinite(P.mouth_pseudo_ms), np.isfinite(P.mouth_real_ms))
    P["mouth"] = mo.astype(float)
    cap = P.variant == "cap"
    P.loc[cap, "callm3d"] = np.where(mo[cap], 1.0, P.loc[cap, "call3d"])
    P.to_csv(OUT / "pseudo_cue_calls_m3d.csv.gz", index=False)
    out = []
    rows = []
    for exn, ex in (("A", ~P.precueA.astype(bool)), ("B", ~P.precue_lick_nocam1.astype(bool))):
        for setn, D in (("all lick-free", P[ex]), ("jaw-decided", P[ex & P.dec]),
                        ("no-lick incl. in-mouth", P[ex & P.noLick_lips])):
            for (p, w), g in D[D.variant == "cap"].groupby(["pos", "window"]):
                rows.append({"excl": exn, "set": setn, "pos": p, "window": w, "n": len(g),
                             "cap3d": g.call3d.mean(), "m3d": g.callm3d.mean(), "mouth": g.mouth.mean(),
                             "cap2d": g.call2d.mean()})
            for (p, w), g in D[D.variant == "prod"].groupby(["pos", "window"]):
                rows.append({"excl": exn, "set": setn, "pos": p, "window": w, "n": len(g),
                             "cur3d": g.call3d.mean(), "cur2d": g.call2d.mean()})
    R = pd.DataFrame(rows).groupby(["excl", "set", "pos", "window"]).first().reset_index()
    V = R.pivot_table(index=["excl", "set", "pos"], columns="window",
                      values=["cur2d", "cur3d", "cap3d", "m3d", "mouth", "n"]).reset_index()
    V.columns = ["_".join([c for c in col if c]) for col in V.columns]
    for r in ("cur2d", "cur3d", "cap3d", "m3d", "mouth"):
        V[f"{r}_excess"] = V[f"{r}_real"] - V[f"{r}_pseudo"]
    V = _sort(V).sort_values(["excl", "set", "pos"])
    V.round(4).to_csv(OUT / "pseudo_cue_m3d.csv", index=False)
    show = ["excl", "set", "pos", "n_pseudo"] + [f"{r}_{w}" for r in ("cur2d", "cap3d", "m3d", "mouth")
                                                  for w in ("pseudo", "real", "excess")]
    out += ["## pseudo-cue vs real (1 s), cap 3-D vs m3d (+ in-mouth share); excl A = sec 9 set, B = in-mouth kept",
            V[show].round(3).to_string(index=False), ""]
    # per animal x epoch, far contra, excl B, decided
    D = P[~P.precue_lick_nocam1.astype(bool) & P.dec & (P.variant == "cap")].copy()
    D["animal"] = D.label.str[:4]
    Q = D.groupby(["animal", "epoch", "window"]).agg(n=("m3d_dummy" if False else "callm3d", "size"),
                                                     m3d=("callm3d", "mean"), cap3d=("call3d", "mean"),
                                                     mouth=("mouth", "mean")).unstack("window")
    Q.columns = [f"{a}_{b}" for a, b in Q.columns]
    Q.round(3).to_csv(OUT / "pseudo_cue_m3d_decided_animal_epoch_exclB.csv")
    out += ["## jaw-decided, excl B, per animal x epoch (all spouts)", Q.round(3).to_string(), ""]
    # spontaneous in-mouth licks: per animal x spout, all lick-free (excl B) and the symmetric no-lick set
    rows = []
    for setn, D in (("all lick-free", P[~P.precue_lick_nocam1.astype(bool)]),
                    ("no-lick incl. in-mouth", P[~P.precue_lick_nocam1.astype(bool) & P.noLick_lips])):
        D = D[D.variant == "cap"].copy()
        D["animal"] = D.label.str[:4]
        for (a, p, w), g in D.groupby(["animal", "pos", "window"]):
            rows.append({"set": setn, "animal": a, "pos": p, "window": w, "n": len(g), "mouth": g.mouth.mean(),
                         "cap3d": g.call3d.mean(), "m3d": g.callm3d.mean(),
                         "mouth_and_3d_still": float(((g.mouth == 1) & (g.call3d != 1)).mean())})
    Z = pd.DataFrame(rows)
    Z = Z.pivot_table(index=["set", "animal", "pos"], columns="window",
                      values=["n", "mouth", "mouth_and_3d_still", "cap3d", "m3d"]).reset_index()
    Z.columns = ["_".join([c for c in col if c]) for col in Z.columns]
    for r in ("cap3d", "m3d"):
        Z[f"{r}_excess"] = Z[f"{r}_real"] - Z[f"{r}_pseudo"]
    Z = _sort(Z).sort_values(["set", "pos", "animal"])
    Z.round(4).to_csv(OUT / "pseudo_cue_m3d_animal_pos_exclB.csv", index=False)
    out += ["## per animal x spout, excl B (mouth = in-mouth lick in the window)", Z.round(3).to_string(index=False), ""]
    return out


def clips(T: pd.DataFrame) -> list[str]:
    rows = []
    for clip, (lab, tid) in CLIPS.items():
        if tid is None:
            continue
        t = T[(T.label == lab) & (T.trial_id == tid)]
        if not len(t):
            continue
        t = t.iloc[0]
        rows.append({"clip": clip, "label": lab, "trial_id": tid, "pos": t.pos, "cls_cur2d": t.cls_cur2d,
                     "on_cur2d": t.on_cur2d, "on_cur3d": t.on_cur3d, "on_cap2d": t.on_cap2d, "on_cap3d": t.on_cap3d,
                     "jud_cap3d": t.jud_cap3d, "mouth_first_ms": t.jaw_mouth_ms, "n_mouth_win": t.n_mouth_win,
                     "on_m3d": t.on_m3d, "jud_m3d": t.jud_m3d, "cls_m3d": t.cls_m3d})
    K = pd.DataFrame(rows)
    K.to_csv(OUT / "clips_15_28_29_30_31.csv", index=False)
    return ["## clips (onsets ms from the cue)", K.to_string(index=False), ""]


def s10a() -> None:
    """S10A decomposition with m3d as THE jaw signal (the in-mouth lick is part of it, not its own category): cap
    table + 3-D re-onset + in-mouth (cue .. first lick, `s10a_trials` mouth_ms); against j3_mouth (cap) of section 9."""
    import importlib.util
    f = Path(__file__).resolve().parent / "caption_review_1123" / "s10a_onsets_analysis.py"
    spec = importlib.util.spec_from_file_location("s10a_an", f)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    from wfield_local import trial_classes as tc
    W = pd.read_csv(REVIEW / "s10a_onsets/s10a_trials_ccB1123.csv.gz", low_memory=False)
    T0 = A.load_trials()[["label", "trial_id", "has_cam1", "has_fr", "jaw_judged"]]
    W = W.merge(T0, on=["label", "trial_id"], how="left")
    W["sess"] = W.label.str[:5] + "2026" + W.label.str[5:]
    J = pd.read_csv(CAP, low_memory=False, usecols=lambda c: c in (
        "session", "trial_id", "position", "cam4_moved", "cam1_moved", "jaw_onset_ms", "onset_at_cue",
        "jaw_reonset_ms", "jaw3d_moved", "jaw3d_onset_ms", "jaw3d_at_cue", "jaw3d_source", "jaw3d_reonset_ms"))
    J["onset_at_cue"] = J.onset_at_cue.astype(str).str.lower().eq("true")
    J["jaw3d_at_cue"] = J.jaw3d_at_cue.astype(str).str.lower().eq("true")
    j = J.set_index(["session", "trial_id"]).reindex(pd.MultiIndex.from_arrays([W.sess, W.trial_id]))
    pos = j.position.fillna("").astype(str).to_numpy()
    jr = j.reset_index(drop=True)
    c3 = tc.jaw_call(jr, pos, "3d", reonset3d=True)
    cm = tc.jaw_call(jr, pos, "3d_mouth", mouth_ms=W.mouth_ms.to_numpy(float))
    W["jaw3d_ms"], W["j3_judged"] = c3["jaw_onset_ms"], c3["judged"]
    W["m3d_ms"], W["m3_judged"] = cm["jaw_onset_ms"], cm["judged"] | W.has_cam1.to_numpy(bool)
    A.AVAIL["m3d"] = "m3_judged"
    allc = []
    for name, cols in (("j3_mouth_cap", ["jaw3d", "mouth"]), ("m3d_jaw", ["m3d"]), ("j3_only_cap", ["jaw3d"])):
        mv, src = A.move(W, cols)
        C = A.cells(W, mv, src, cols)
        C.insert(0, "variant", name)
        allc.append(C)
        if name == "m3d_jaw":
            W["mv_m3d"], W["src_m3d"] = mv, src
    C = pd.concat(allc, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUT / "s10a_cells_m3d.csv", index=False, float_format="%.4g")
    F = C[C.pos == "far_contra"].copy()
    F["epoch"] = pd.Categorical(F.epoch, EPOCHS)
    F = F.sort_values(["variant", "animal", "epoch"])
    F.to_csv(OUT / "s10a_far_contra_per_animal_m3d.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    X = F.pivot_table(index=["animal", "epoch"], columns="variant", values=["wait_med_ms", "revup_med_ms"],
                      observed=True)
    print(X.round(0).to_string())
    M = F.groupby(["variant", "epoch"], observed=True)[["wait_med_ms", "revup_med_ms", "lat_med_ms"]].mean()
    M.round(1).to_csv(OUT / "s10a_far_contra_mean_of_animals_m3d.csv")
    print(M.round(0).to_string())
    sh = [c for c in C.columns if c.startswith("share_")]
    S = C.groupby(["variant", "pos", "epoch"])[sh].mean().round(3)
    S.to_csv(OUT / "s10a_shares_mean_of_animals_m3d.csv")
    print(S.loc[["j3_mouth_cap", "m3d_jaw"]].xs("far_contra", level="pos").to_string())
    # same-trial check: m3d first movement == j3_mouth_cap first movement
    mv1, _ = A.move(W, ["jaw3d", "mouth"])
    ok = np.isclose(mv1, W.mv_m3d, equal_nan=True, atol=1e-6).mean()
    print(f"m3d first movement == j3_mouth (cap) first movement on {ok:.5f} of trials")


def blind() -> None:
    """m3d for the 50 blind trials -> INDEX_for_Priya_v2.csv (a copy + new columns) and key_mouth.csv (for the clips)."""
    from wfield_local import trial_classes as tc
    K = pd.read_csv(BLIND / "key.csv")
    M = pd.read_csv(OUT / "classes_m3d.csv.gz", low_memory=False)
    A = pd.read_csv(FIX / "classes_all_rules.csv.gz", low_memory=False,
                    usecols=["label", "trial_id", "cls_cap3d", "on_cap3d"])
    K2 = K.merge(M[["label", "trial_id", "cls_m3d", "on_m3d", "jaw_mouth_ms", "n_mouth_win"]],
                 on=["label", "trial_id"], how="left").merge(A, on=["label", "trial_id"], how="left")
    ev = {}
    for lab in sorted(K2.label.unique()):
        E = tc.mouth_lick_events(lab)
        tt = tc._trial_times(Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/"
                                  "sessions") / lab[:4] / ("2026" + lab[5:]))
        for tid in K2.trial_id[K2.label == lab]:
            c = float(tt.cue_daq[tt.trial_id == tid].iloc[0])
            e = [] if E is None else ((E.daq_on_s.to_numpy(float) - c) * 1000.0).tolist()
            ev[(lab, int(tid))] = ";".join(f"{x:.0f}" for x in sorted(e) if -500 <= x <= 3700)
    K2["mouth_events_ms"] = [ev.get((a, int(b)), "") for a, b in zip(K2.label, K2.trial_id)]
    K2["m3d_moved"] = np.where(K2.cls_m3d == "jaw_only", "moved", np.where(K2.cls_m3d == "no_response", "still",
                                                                         K2.cls_m3d))
    K2.to_csv(BLIND / "key_mouth.csv", index=False)
    I = pd.read_csv(BLIND / "INDEX_for_Priya.csv")
    I["clip_reveal_mouth (open AFTER the blind columns)"] = "clips_reveal/" + I.id + "_reveal_mouth.mp4"
    I["new_rule_3d_or_inmouth (open AFTER the blind columns)"] = I.id.map(dict(zip(K2.id, [
        f"{m}; onset {o:.0f} ms" if np.isfinite(o) else f"{m}" for m, o in zip(K2.m3d_moved, K2.on_m3d)])))
    I.to_csv(BLIND / "INDEX_for_Priya_v2.csv", index=False)
    print(K2[["id", "group", "cls_2d", "cls_3d", "cls_cap3d", "on_cap3d", "cls_m3d", "on_m3d", "mouth_events_ms"]]
          .to_string(index=False))


def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    _rm_guard.require_default("jaw3d_mouth_eval")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("classes", "summary", "s10a", "blind"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--labels", nargs="*", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    {"classes": lambda: classes(a.jobs, a.labels), "summary": summary, "s10a": s10a, "blind": blind}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
