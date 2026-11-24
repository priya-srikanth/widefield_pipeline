"""RE-EVALUATION of the jaw call with Priya's two "let's try" fixes (CC-B 11-23), next to the current rules.

    python scripts/jaw3d_fix_eval.py classes [--jobs 5]   # per-trial classes under every rule (both calls x variants)
    python scripts/jaw3d_fix_eval.py summary              # agreement, counts, onset timing, pseudo-cue, in-mouth, clips
    python scripts/jaw3d_fix_eval.py s10a                 # S10A first-movement decomposition with the fixed jaw

RULES (``trial_classes.session_trials``; nothing in production changes):
  cur2d / cur3d   the production jaw_onsets.csv, jaw_call mode "classes" / "3d" (3-D: no re-onset)
  <v>2d / <v>3d   the onset variant <v> in {mad, cap} (`scripts/jaw_onsets.py` JAW_ONSET_VARIANT: robust MAD pre-cue
                  noise / the 4 x noise term capped at 2.5 x the floor, applied to cam4, cam1 AND the 3-D call; tables
                  jaw3d_qc/jaw_onsets_<v>.csv) + the 3-D RE-ONSET (``reonset3d``; the 2-D calls always had theirs)
Jaw-decided trials: no contact, no lick / tongue, not frame-gated (under both rules compared). Writes only under
<review>/jaw3d_qc/fix_eval/.
"""
from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd

REVIEW = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ccB_caption_review_2026-11-23")
JQ = REVIEW / "jaw3d_qc"
OUT = JQ / "fix_eval"
QC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/qc")
TABLES = {"cur": QC / "jaw_onsets.csv", "mad": JQ / "jaw_onsets_mad.csv", "cap": JQ / "jaw_onsets_cap.csv"}
RULES = {f"{v}{c}": (v, "classes" if c == "2d" else "3d") for v in TABLES for c in ("2d", "3d")}
POS = {"close_L": "near ipsi", "close_center": "near middle", "close_R": "near contra",
       "far_L": "far ipsi", "far_center": "far middle", "far_R": "far contra"}
POS_ORDER = list(POS.values())
EPOCHS = ["pre", "acute", "subacute", "chronic"]
NOLICK = ("jaw_only", "no_response", "unknown")
CLIPS = {"clip15": ("PS95_20260817", 69), "clip28": ("PS92_20260809", 209), "clip29": ("PS93_20260807", 325),
         "clip30": ("PS94_20260821", 328)}


def session_job(label: str) -> pd.DataFrame | None:
    from wfield_local import epochs
    from wfield_local import trial_classes as tc
    if getattr(tc.jaw_table, "cache_info", None) and tc.jaw_table.cache_info().maxsize < 4:
        tc.jaw_table = functools.lru_cache(maxsize=4)(tc.jaw_table.__wrapped__)     # 3 tables per worker
    out = None
    for r, (v, mode) in RULES.items():
        T = tc.session_trials(label, jaw=mode, jaw_csv=TABLES[v], reonset3d=(v != "cur"))
        if T is None:
            return None
        if out is None:
            out = T[["trial_id", "position", "frame_gated", "n_contacts", "n_licks"]].copy()
        out[f"cls_{r}"] = T.cls.to_numpy()
        out[f"det_{r}"] = T.cls_detail.to_numpy()
        out[f"on_{r}"] = T.jaw_onset_ms.to_numpy(float)
        out[f"jud_{r}"] = T.jaw_judged.to_numpy(bool)
    out.insert(0, "label", label)
    out["epoch"] = epochs.epoch_of(label)
    out["pos"] = out.position.map(POS)
    return out


def classes(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    for v, f in TABLES.items():
        if not f.exists():
            raise SystemExit(f"missing {f} (run scripts/jaw_onsets.py with JAW_ONSET_VARIANT={v} first)")
    labels = sorted(pd.read_csv(QC / "s1_trial_classes_cue_stop.csv.gz", usecols=["label"]).label.unique())
    res, fails = ak.fan_sessions(labels, session_job, jobs=jobs)
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.concat([r for _, r in res if r is not None], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "classes_all_rules.csv.gz", index=False, float_format="%.4g")
    print(f"{len(T)} trials -> {OUT / 'classes_all_rules.csv.gz'}")


def _decided(T: pd.DataFrame, a: str, b: str) -> pd.Series:
    return (T[f"det_{a}"].isin(NOLICK) & T[f"det_{b}"].isin(NOLICK) & ~T.frame_gated.astype(bool))


def _kappa(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    po = float((x == y).mean())
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def _sort(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    D["pos"] = pd.Categorical(D.pos, POS_ORDER)
    if "epoch" in D:
        D["epoch"] = pd.Categorical(D.epoch, EPOCHS)
        return D.sort_values(["pos", "epoch"])
    return D.sort_values("pos")


def summary() -> None:
    T = pd.read_csv(OUT / "classes_all_rules.csv.gz", low_memory=False)
    T = T[T.epoch.isin(EPOCHS)]
    lines = []
    pd.set_option("display.width", 250)
    pairs = [("cur2d", "cur3d"), ("mad2d", "mad3d"), ("cap2d", "cap3d")]
    # 1. agreement / kappa (jaw-only vs no-response, judged in both), two-camera spouts
    rows = []
    for a, b in pairs:
        D = T[_decided(T, a, b) & T[f"cls_{a}"].isin(["jaw_only", "no_response"])
              & T[f"cls_{b}"].isin(["jaw_only", "no_response"]) & (T[f"det_{a}"] != "tongue_at_lips")
              & (T.position != "close_center")]
        for (p, e), g in D.groupby(["pos", "epoch"]):
            po, k = _kappa(g[f"cls_{a}"] == "jaw_only", g[f"cls_{b}"] == "jaw_only")
            rows.append({"pair": f"{a} vs {b}", "pos": p, "epoch": e, "n": len(g), "agree": round(po, 3),
                         "kappa": round(k, 3)})
    A = _sort(pd.DataFrame(rows))
    A.to_csv(OUT / "agreement_pos_epoch.csv", index=False)
    W = A.pivot_table(index=["pos", "epoch"], columns="pair", values="kappa", observed=True)
    lines += ["## kappa (jaw-only vs no-response), jaw-decided trials", W.round(2).to_string(), ""]
    # 2. jaw-only counts (jaw-called, cls_detail == jaw_only) per spout x epoch, every rule
    C = pd.concat({r: T[T[f"det_{r}"] == "jaw_only"].groupby(["pos", "epoch"]).size() for r in RULES}, axis=1)
    C = C.fillna(0).astype(int).reset_index()
    C = _sort(C)
    C.to_csv(OUT / "jaw_only_counts_pos_epoch.csv", index=False)
    lines += ["## jaw-only trials (jaw-called) per spout x epoch", C.to_string(index=False), ""]
    # unknown counts (the at-cue fix)
    U = pd.concat({r: T[T[f"cls_{r}"] == "unknown"].groupby("pos").size() for r in RULES}, axis=1).fillna(0).astype(int)
    lines += ["## unknown trials per spout", U.reindex(POS_ORDER).to_string(), ""]
    # 3. onset timing, 3-D minus 2-D (same variant), decided trials both jaw-only / all trials both moved
    rows = []
    for a, b in pairs:
        for setn, m in (("all_both_moved", np.ones(len(T), bool)),
                        ("jaw_decided_both_jaw_only", (_decided(T, a, b) & (T[f"cls_{a}"] == "jaw_only")
                                                       & (T[f"cls_{b}"] == "jaw_only")).to_numpy())):
            D = T[m & np.isfinite(T[f"on_{a}"]) & np.isfinite(T[f"on_{b}"]) & (T.position != "close_center")]
            for p, g in D.groupby("pos"):
                d = g[f"on_{b}"] - g[f"on_{a}"]
                rows.append({"pair": f"{b} - {a}", "set": setn, "pos": p, "n": len(d), "median": d.median(),
                             "q25": d.quantile(.25), "q75": d.quantile(.75), "late_gt50": (d > 50).mean(),
                             "early_gt50": (d < -50).mean(), "within40": (d.abs() <= 40).mean()})
    O = _sort(pd.DataFrame(rows))
    O.round(3).to_csv(OUT / "onset_diff_3d_minus_2d.csv", index=False)
    lines += ["## onset 3-D minus 2-D (ms)", O.round(3).to_string(index=False), ""]
    # 4. pseudo-cue: class-equivalent calls per variant (floor_null_variants)
    lines += pseudo(T)
    # 5. in-mouth licks
    lines += inmouth()
    # 6. clips
    lines += clips(T)
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def pseudo(T: pd.DataFrame) -> list[str]:
    f = JQ / "floor_null_variants.csv.gz"
    if not f.exists():
        return ["## pseudo-cue: floor_null_variants.csv.gz missing", ""]
    N = pd.read_csv(f, keep_default_na=False, na_values=[""], low_memory=False)
    N["moved"] = pd.to_numeric(N.moved, errors="coerce")
    N = N[~N.precue_lick.astype(str).str.lower().eq("true")]
    at = N.at_cue.astype(str).str.lower().eq("true")
    re_ = N.has_reonset.astype(str).str.lower().eq("true")
    # class-equivalent call per signal: at cue -> the re-onset (2-D always; 3-D only in the fixed variants) or unjudged
    reon_ok = np.where((N.signal == "3d") & (N.variant == "prod"), False, re_)
    N["judged"] = N.moved.notna() & (~at | reon_ok)
    N["call"] = np.where(N.judged, (N.moved == 1).astype(float), np.nan)
    rows = []
    for (lab, tid, v, w), g in N.groupby(["label", "trial_id", "variant", "window"], sort=False):
        c3 = g.loc[g.signal == "3d", "call"]
        c2 = g.loc[g.signal.isin(["cam4", "cam1"]), "call"].dropna()
        rows.append((lab, tid, v, w, float(c3.iloc[0]) if len(c3) else np.nan,
                     (float((c2 == 1).any()) if len(c2) else np.nan), g.position.iloc[0]))
    P = pd.DataFrame(rows, columns=["label", "trial_id", "variant", "window", "call3d", "call2d", "position"])
    dec = T[_decided(T, "cur2d", "cur3d")][["label", "trial_id"]].assign(dec=True)
    P = P.merge(dec, on=["label", "trial_id"], how="left")
    P["dec"] = P.dec.fillna(False).astype(bool)
    P["pos"] = P.position.map(POS)
    P.to_csv(OUT / "pseudo_cue_calls.csv.gz", index=False)
    out = []
    for setn, D in (("all lick-free", P), ("jaw-decided", P[P.dec])):
        R = D.groupby(["pos", "variant", "window"])[["call3d", "call2d"]].mean().unstack("window")
        R.columns = [f"{s}_{w}" for s, w in R.columns]
        R["excess3d"] = R.call3d_real - R.call3d_pseudo
        R["excess2d"] = R.call2d_real - R.call2d_pseudo
        R = R.reset_index()
        R["pos"] = pd.Categorical(R.pos, POS_ORDER)
        R = R.sort_values(["pos", "variant"])
        R.round(4).to_csv(OUT / f"pseudo_cue_{'all' if setn.startswith('all') else 'decided'}.csv", index=False)
        out += [f"## pseudo-cue vs real (1 s windows), {setn}", R.round(3).to_string(index=False), ""]
    return out


def inmouth() -> list[str]:
    m = pd.read_csv(REVIEW / "s10a_onsets/tables_ccB1123/mouth_events_jaw3d_match.csv.gz")
    out = ["## in-mouth licks (659, before the first lick): 3-D rise within +-60 ms vs the applied 3-D threshold"]
    rows = []
    for v, f in (("prod", JQ / "jaw3d_reonset_trials.csv.gz"), ("mad", JQ / "jaw3d_reonset_trials_mad.csv.gz"),
                 ("cap", JQ / "jaw3d_reonset_trials_cap.csv.gz")):
        if not f.exists():
            out.append(f"  {v}: {f.name} missing")
            continue
        R = pd.read_csv(f, usecols=["label", "trial_id", "thr_um"])
        M = m.merge(R, on=["label", "trial_id"], how="left")
        rise = M.rise_60
        rows.append({"variant": v, "n": len(M), "reach_threshold": float((rise >= M.thr_um).mean()),
                     "above_floor_below_thr": float(((rise >= 200) & (rise < M.thr_um)).mean()),
                     "below_floor": float((rise < 200).mean()), "no_3d": float((M.thr_um.isna() | rise.isna()).mean()),
                     "median_thr_um": float(M.thr_um.median())})
    I = pd.DataFrame(rows)
    I.round(3).to_csv(OUT / "inmouth_threshold_share.csv", index=False)
    return out + [I.round(3).to_string(index=False), ""]


def clips(T: pd.DataFrame) -> list[str]:
    rows = []
    for v, f in TABLES.items():
        J = pd.read_csv(f, low_memory=False)
        for clip, (s, tid) in CLIPS.items():
            r = J[(J.session == s) & (J.trial_id == tid)]
            if not len(r):
                continue
            r = r.iloc[0]
            lab = f"{s[:4]}_{s[9:]}"
            t = T[(T.label == lab) & (T.trial_id == tid)]
            rows.append({"clip": clip, "table": v, "cam4_moved": r.cam4_moved, "cam1_moved": r.cam1_moved,
                         "2d_onset": r.jaw_onset_ms, "2d_at_cue": r.onset_at_cue, "2d_reonset": r.jaw_reonset_ms,
                         "3d_moved": r.jaw3d_moved, "3d_onset": r.jaw3d_onset_ms, "3d_at_cue": r.jaw3d_at_cue,
                         "3d_reonset": r.get("jaw3d_reonset_ms", np.nan),
                         "class_2d": t[f"cls_{v}2d"].iloc[0] if len(t) else "", "onset_2d_rule": t[f"on_{v}2d"].iloc[0] if len(t) else np.nan,
                         "class_3d": t[f"cls_{v}3d"].iloc[0] if len(t) else "", "onset_3d_rule": t[f"on_{v}3d"].iloc[0] if len(t) else np.nan})
    K = pd.DataFrame(rows)
    K.to_csv(OUT / "clips_15_28_29_30.csv", index=False)
    return ["## clips 15 / 28 / 29 / 30 (production table = cur; classes are the rule's class of that trial)",
            K.to_string(index=False), ""]


def s10a() -> None:
    """S10A first-movement decomposition (the S10A helper's `move` / `cells`, scripts/caption_review_1123) with the
    3-D jaw onset (and the class-rule jaw) from each table + the 3-D re-onset; far contra per animal x epoch."""
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
    W0 = W[["jaw3d_ms", "jawcls_ms"]].copy()
    allc = []
    for v, ftab in TABLES.items():
        J = pd.read_csv(ftab, low_memory=False, usecols=lambda c: c in (
            "session", "trial_id", "position", "cam4_moved", "cam1_moved", "jaw_onset_ms", "onset_at_cue",
            "jaw_reonset_ms", "jaw3d_moved", "jaw3d_onset_ms", "jaw3d_at_cue", "jaw3d_source", "jaw3d_reonset_ms"))
        J["onset_at_cue"] = J.onset_at_cue.astype(str).str.lower().eq("true")
        J["jaw3d_at_cue"] = J.jaw3d_at_cue.astype(str).str.lower().eq("true")
        J = J.set_index(["session", "trial_id"])
        j = J.reindex(pd.MultiIndex.from_arrays([W.sess, W.trial_id]))
        pos = j.position.fillna("").astype(str).to_numpy()
        for mode, col in (("3d", "jaw3d"), ("classes", "jawcls")):
            jc = tc.jaw_call(j.reset_index(drop=True), pos, mode, reonset3d=(v != "cur"))
            if v == "cur":                                   # reproduction check against the S10A table
                ref = W0[f"{col}_ms"].to_numpy(float)
                ok = np.isclose(jc["jaw_onset_ms"], ref, atol=0.5, equal_nan=True).mean()
                print(f"S10A {col}_ms reproduced on the production table: {ok:.4f} of trials")
            W[f"{col}_ms"] = jc["jaw_onset_ms"]
            W["j3_judged" if col == "jaw3d" else "jc_judged"] = jc["judged"]
        for name, cols in (("j3_only", ["jaw3d"]), ("j3_mouth", ["jaw3d", "mouth"]), ("cls_jaw", ["jawcls"]),
                           ("cls_mouth", ["jawcls", "mouth"])):
            mv, src = A.move(W, cols)
            C = A.cells(W, mv, src, cols)
            C.insert(0, "variant", name)
            C.insert(0, "table", v)
            allc.append(C)
    C = pd.concat(allc, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUT / "s10a_cells_all.csv", index=False, float_format="%.4g")
    F = C[C.pos == "far_contra"].copy()
    F["epoch"] = pd.Categorical(F.epoch, EPOCHS)
    F = F.sort_values(["variant", "table", "animal", "epoch"])
    F.to_csv(OUT / "s10a_far_contra_per_animal.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    for name in ("j3_mouth", "j3_only", "cls_jaw"):
        X = F[F.variant == name].pivot_table(index=["animal", "epoch"], columns="table",
                                             values=["wait_med_ms", "revup_med_ms"], observed=True)
        print(f"## S10A far contra, {name}: wait / rev-up medians per animal x epoch\n{X.round(0).to_string()}\n")
    M = F.groupby(["variant", "table", "epoch"], observed=True)[["wait_med_ms", "revup_med_ms", "lat_med_ms"]].mean()
    M.round(1).to_csv(OUT / "s10a_far_contra_mean_of_animals.csv")
    print(M.round(0).to_string())


def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    _rm_guard.require_default("jaw3d_fix_eval")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("classes", "summary", "s10a"))
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args(argv)
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    {"classes": lambda: classes(a.jobs), "summary": summary, "s10a": s10a}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
