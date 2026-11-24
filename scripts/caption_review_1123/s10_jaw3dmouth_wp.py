"""S10A and S10E on the adopted jaw rule and the camera WORKING PERIOD (CC-B overnight rebuild 11-23, item 3e).

S10A (first-lick delay decomposition, cue -> first movement = "wait", first movement -> first lick = "rev-up"):
  first movement = THE six-position jaw call `trial_classes.jaw_call(mode="3d_mouth")`: the four-camera 3-D jaw of the
  CAP jaw table (cap + 3-D re-onset) OR an in-mouth lick (kept cam1 event not matched to the fused four-camera list),
  onset = the earlier; the earliest onset <= the fused first lick (a measured trial with none = the lick, rev-up 0),
  exactly `s10a_onsets_analysis.move` with the single signal "m3d" (as `jaw3d_mouth_eval.py s10a`).
  FaceRhythm (pooled pre-stroke scale, `frpool`) is reported SEPARATELY (supplementary variant m3d + frpool, and the
  frpool onset alone); it is not part of the first movement.
  first lick = the fused four-camera list (the cam3-on rev-up per-session pickles, full precision).
S10E: within-animal Spearman rho of the far-contra first-lick aim (`iti_recovery_measure.aim_licks`, first post-cue
  lick of the four-camera kinematics) vs the median far-contra first-lick latency over post-stroke sessions, raw and
  after regressing both on log(day) (`c2_manuscript_correlations.angle_latency`), with latency AND aim per session
  from working-period trials.

Trial gates (per trial, all computed; `--gate` picks the reported one):
  ref    production rev-up `canonical` (= production qc/trial_attempts.csv canonical, the near-spout reference gate)
  wpcls  the new class snapshot's `working` flag (matched to the rev-up trials by cue time)
  wp     the new camera-tagged attempts table's `canonical` (= working period; PRIMARY)

Per-trial features that do not depend on the gate are REUSED (checked in the code: the rev-up per-session pickles,
`revup_behaviour.session`, never read the attempts table; the S10A per-session analysis pickle holds the in-mouth
licks and FR series of every trial; FR baselines use all trials of a session).

    python s10_jaw3dmouth_wp.py repro            # reproduce the live / section 10.5 numbers on the production gate
    python s10_jaw3dmouth_wp.py run [--gate wp]  # tables under data_4cam_20261122/s10a_jaw3dmouth_wp/
Writes only under vls_stroke_paper/data_4cam_20261122/s10a_jaw3dmouth_wp/.
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1]))

PAPER = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper")
REVIEW = PAPER / "ccB_caption_review_2026-11-23"
OUT = PAPER / "data_4cam_20261122" / "s10a_jaw3dmouth_wp"
TAG = "jaw3dmouth_wp_ccB1123"
ENC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding")
CAP_FULL = ENC / "qc" / "jaw_onsets_cap_ccB1123.csv"
CAP_LEAN = REVIEW / "jaw3d_qc" / "jaw_onsets_cap.csv"
NEW = ENC / "qc" / "jaw3dmouth_ccB1123"
CLS_NEW = NEW / "s1_trial_classes_cue_stop.csv.gz"
ATT_NEW = NEW / "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk" / "trial_attempts.csv"
CACHE = REVIEW / "s10a_onsets" / "cache_ccB1123"
KIN = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60/"
           "kinematics_fused_postcue/licks.csv.gz")
AIM_LIVE = ENC / "aim_kinematics_fourcam_boot4000_cam3on" / "aim_by_spout_sessions.csv"
TRACK_LIVE = PAPER / "data_4cam_20261122" / "correlations" / "revup_track_sessions_boot4000_cam3on.csv"
CELLS_LIVE = PAPER / "manuscript_code" / "data" / "c2_revup_cells.csv"
EP = ["pre", "acute", "subacute", "chronic"]
MOUSE = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}
JCOLS = ("session", "trial_id", "position", "cam4_moved", "cam1_moved", "jaw_onset_ms", "onset_at_cue",
         "jaw_reonset_ms", "jaw3d_moved", "jaw3d_onset_ms", "jaw3d_at_cue", "jaw3d_source", "jaw3d_reonset_ms")


def _mod(name: str, f: Path):
    spec = importlib.util.spec_from_file_location(name, f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod("s10a_an", HERE / "s10a_onsets_analysis.py")


# ------------------------------------------------------------------------------------------------ per-trial table
def trials() -> pd.DataFrame:
    """Every rev-up trial (licked or not, any gate) with the reused per-trial features: fused first-lick latency,
    in-mouth lick (mouth_ms), FR onsets on the own-session and the pooled pre-stroke scale (z 3, K 3)."""
    T0 = A.load_trials()
    res = pickle.loads((CACHE / "_analysis_per_session_ccB1123.pkl").read_bytes())
    R = pd.concat([r["R"] for r in res], ignore_index=True)
    T = T0.merge(R[["label", "trial_id", "mouth_ms", "jaw3d_ms", "j3_judged"]].rename(
        columns={"jaw3d_ms": "jaw3d_uncap_ms", "j3_judged": "j3_uncap_judged"}), on=["label", "trial_id"], how="left")
    assert len(T) == len(T0)
    LAG = A.lags()
    ep_of = T0.groupby("label").epoch.first()
    pre_pool = {}
    for an in sorted(T.animal.unique()):
        vals = np.concatenate([r["base_vals"] for r in res if r["animal"] == an and "V" in r
                               and ep_of[r["label"]] == "pre"])
        pre_pool[an] = A.robust(vals)
    Zo, Zp = [], []
    for r in res:                       # res is in label order, as R (and T0 sorted by label in load_trials)
        n = len(r["R"])
        if "V" not in r:
            Zo.append(np.full((n, len(LAG)), np.nan, np.float32))
            Zp.append(Zo[-1])
            continue
        pm, ps = pre_pool[r["animal"]]
        Zo.append((r["V"] - r["base_med"]) / r["base_s"])
        Zp.append((r["V"] - pm) / ps)
    order = pd.concat([r["R"][["label", "trial_id"]] for r in res], ignore_index=True)
    assert (order.label.to_numpy() == T.label.to_numpy()).all() and (order.trial_id.to_numpy() == T.trial_id.to_numpy()).all()
    lim = T.lat_ms.to_numpy(float) / 1000
    T["fr_ms"] = A.onset(np.concatenate(Zo), lim, 3.0, 3, LAG) * 1000
    T["frpool_ms"] = A.onset(np.concatenate(Zp), lim, 3.0, 3, LAG) * 1000
    T["sess"] = T.label.str[:5] + "2026" + T.label.str[5:]
    return T


def add_jaw(T: pd.DataFrame, cap: Path, suffix: str = "") -> None:
    """m3d (cap 3-D + re-onset OR in-mouth) and the cap 3-D-only call, from jaw table ``cap``."""
    from wfield_local import trial_classes as tc
    J = pd.read_csv(cap, low_memory=False, usecols=lambda c: c in JCOLS)
    for c in ("onset_at_cue", "jaw3d_at_cue"):
        J[c] = J[c].astype(str).str.lower().eq("true")
    j = J.set_index(["session", "trial_id"]).reindex(pd.MultiIndex.from_arrays([T.sess, T.trial_id]))
    pos = j.position.fillna("").astype(str).to_numpy()
    jr = j.reset_index(drop=True)
    c3 = tc.jaw_call(jr, pos, "3d", reonset3d=True)
    cm = tc.jaw_call(jr, pos, "3d_mouth", mouth_ms=T.mouth_ms.to_numpy(float))
    T[f"jaw3d{suffix}_ms"], T[f"j3{suffix}_judged"] = c3["jaw_onset_ms"], c3["judged"]
    T[f"m3d{suffix}_ms"] = cm["jaw_onset_ms"]
    T[f"m3{suffix}_judged"] = cm["judged"] | T.has_cam1.to_numpy(bool)
    A.AVAIL[f"m3d{suffix}"] = f"m3{suffix}_judged"
    A.AVAIL[f"jaw3d{suffix}"] = f"j3{suffix}_judged"


def add_gates(T: pd.DataFrame) -> dict:
    """ref (production canonical), wpcls (new class snapshot `working`, cue-time match), wp (new attempts canonical)."""
    info = {}
    T["g_ref"] = T.canonical.astype(bool)
    C = pd.read_csv(CLS_NEW, usecols=["label", "cue_s", "working", "cls", "cls_final"])
    parts = []
    for lab, g in T[["label", "trial_id", "cue_s"]].groupby("label"):
        c = C[C.label == lab].sort_values("cue_s")
        g = g.sort_values("cue_s")
        if c.empty:
            parts.append(g.assign(working=np.nan, cls_new=np.nan, cls_final_new=np.nan, dcue=np.nan))
            continue
        m = pd.merge_asof(g, c.rename(columns={"cue_s": "cue_c", "cls": "cls_new", "cls_final": "cls_final_new"})
                          .drop(columns="label"), left_on="cue_s", right_on="cue_c", direction="nearest",
                          tolerance=0.05)
        m["dcue"] = (m.cue_s - m.cue_c).abs()
        parts.append(m.drop(columns="cue_c"))
    M = pd.concat(parts).set_index(["label", "trial_id"]).reindex(pd.MultiIndex.from_frame(T[["label", "trial_id"]]))
    T["g_wpcls"] = M.working.astype(str).str.lower().eq("true").to_numpy()
    T["cls_final_new"] = M.cls_final_new.to_numpy()
    info["wpcls_matched"] = float(M.working.notna().mean())
    info["wpcls_max_dcue_s"] = float(np.nanmax(M.dcue))
    if ATT_NEW.exists():
        G = pd.read_csv(ATT_NEW, usecols=["label", "trial_id", "canonical", "gate"])
        info["att_gate_values"] = G.gate.astype(str).str.split(" ").str[0].value_counts().to_dict()
        G["canonical"] = G.canonical.astype(str).str.lower().eq("true")
        X = T[["label", "trial_id"]].merge(G, on=["label", "trial_id"], how="left")
        info["wp_matched"] = float(X.canonical.notna().mean())
        T["g_wp"] = X.canonical.fillna(False).to_numpy(bool)
        info["wp_vs_wpcls_agree"] = float((T.g_wp == T.g_wpcls).mean())
    else:
        info["wp"] = "attempts table not found; wp = wpcls"
        T["g_wp"] = T.g_wpcls
    for g in ("g_ref", "g_wpcls", "g_wp"):
        info[f"n_{g}"] = int(T[g].sum())
        info[f"n_{g}_licked"] = int((T[g] & np.isfinite(T.lat_ms)).sum())
    return info


VARIANTS = {"m3d": ["m3d"], "m3d_frpool": ["m3d", "frpool"], "frpool_only": ["frpool"], "j3cap_only": ["jaw3d"]}


def decompose(T: pd.DataFrame, gate: str, variants=VARIANTS) -> tuple[dict, pd.DataFrame]:
    W = T[T[gate] & np.isfinite(T.lat_ms)].copy()
    out = {}
    for name, cols in variants.items():
        mv, src = A.move(W, cols)
        W[f"mv_{name}"], W[f"src_{name}"] = mv, src
        out[name] = A.cells(W, mv, src, cols)
    return out, W


def far(C: pd.DataFrame, pos: str = "far_contra") -> pd.DataFrame:
    g = C[C.pos == pos].groupby("epoch")[["wait_med_ms", "revup_med_ms", "lat_med_ms"]].mean()
    return g.reindex(EP)


# ------------------------------------------------------------------------------------------------ S10E
def aim_sessions(gate_tab: pd.DataFrame | None) -> pd.DataFrame:
    """Far-contra first-lick aim per session (`off`, deg vs the animal's all-pre-lick mean at the spout) from the
    four-camera kinematics; ``gate_tab`` (label, trial_id) = the trials kept (None = all trials, the live path)."""
    import aim_tables_from_kinematics as atk
    import iti_recovery_measure as rm
    labels = sorted(pd.read_csv(AIM_LIVE).label.unique())
    W = pd.DataFrame({"set": "trial", "label": labels})
    ep = pd.read_csv(AIM_LIVE).drop_duplicates("label").set_index("label").epoch
    W["epoch"] = ep.reindex(W.label).to_numpy()
    tmp = OUT / f"_labels_{TAG}.csv"
    W.to_csv(tmp, index=False)
    P = atk.build(KIN, tmp)
    tmp.unlink()
    if gate_tab is not None:
        K = pd.read_csv(KIN, usecols=["session", "trial_id", "context", "side_ml", "out_ap", "down_dv", "lick_idx"])
        K = K[K.context == "post_cue"].dropna(subset=["side_ml", "out_ap", "down_dv", "lick_idx"])
        K["label"] = K.session.str.split("_").str[0] + "_" + K.session.str.split("_").str[1].str[-4:]
        K = K[K.label.isin(set(labels))]
        assert len(K) == len(P), (len(K), len(P))
        key = set(zip(gate_tab.label, gate_tab.trial_id))
        keep = np.array([(a, b) in key for a, b in zip(K.label, K.trial_id)])
        P = P[keep]
    f = OUT / f"_preds_{TAG}.csv"
    P.to_csv(f, index=False)
    p = rm.aim_licks(str(f))
    f.unlink()
    p = p[p["first"] & (p.pos == "far_contra")]
    g = p.groupby(["animal", "label", "epoch"]).off.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= rm.MIN_LICKS["first"]].rename(columns={"mean": "aim", "size": "n_aim"})
    return g


def lat_sessions(T: pd.DataFrame, gate: str | None) -> pd.DataFrame:
    X = T[(T.pos == "far_contra")] if gate is None else T[T[gate] & (T.pos == "far_contra")]
    rows = []
    for (lab, an, e), g in X.groupby(["label", "animal", "epoch"]):
        lk = g.lat_ms.to_numpy(float)
        lk = lk[np.isfinite(lk)]
        rows.append({"label": lab, "animal": an, "epoch": e, "day": g.day.iloc[0], "n_trials": len(g),
                     "n_licked": len(lk), "lat_med_ms": np.median(lk) if len(lk) else np.nan})
    return pd.DataFrame(rows)


def angle_latency(d: pd.DataFrame) -> pd.DataFrame:
    cm = _mod("c2mc", HERE.parents[0] / "c2_manuscript_correlations.py")
    f = OUT / f"_track_{TAG}.csv"
    d.to_csv(f, index=False)
    r = cm.angle_latency(f)
    f.unlink()
    return r


# ------------------------------------------------------------------------------------------------ commands
def repro() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    T = trials()
    add_jaw(T, CAP_LEAN, "lean")
    add_jaw(T, CAP_FULL, "")
    cols = ["m3d_ms", "jaw3d_ms"]
    for c in cols:
        a, b = T[c].to_numpy(float), T[c[:-3] + "lean_ms"].to_numpy(float)
        lines.append(f"{c}: full cap table == lean cap table on {np.mean(np.isclose(a, b, equal_nan=True)):.5f} "
                     f"of {len(T)} trials")
    T["g_ref"] = T.canonical.astype(bool)
    C, W = decompose(T, "g_ref", {"m3d": ["m3d"], "m3dlean": ["m3dlean"], "j3cap_only": ["jaw3d"]})
    # vs section 10.5 (fix_eval_mouth/s10a_cells_m3d.csv, variant m3d_jaw)
    S = pd.read_csv(REVIEW / "jaw3d_qc" / "fix_eval_mouth" / "s10a_cells_m3d.csv")
    S = S[S.variant == "m3d_jaw"].drop(columns="variant")
    X = C["m3d"].merge(S, on=["animal", "pos", "epoch"], suffixes=("", "_105"))
    for c in ("n_licked", "wait_med_ms", "revup_med_ms", "lat_med_ms"):
        d = np.nanmax(np.abs(X[c] - X[f"{c}_105"]))
        lines.append(f"section 10.5 cells (m3d_jaw) vs this run, ref gate, full cap table: max |diff| {c} = {d:.4g} "
                     f"({len(X)} of {len(S)} cells)")
    F = far(C["m3d"]).round(0)
    lines.append("far contra m3d, ref gate (mean of mice): wait " + " / ".join(f"{v:.0f}" for v in F.wait_med_ms)
                 + "; rev-up " + " / ".join(f"{v:.0f}" for v in F.revup_med_ms) + "  (10.5: 158/405/156/138, 32/65/33/28)")
    # live S10A
    L = pd.read_csv(CELLS_LIVE)
    FL = far(L)
    lines.append("LIVE S10A (c2_revup_cells.csv) far contra: wait " + " / ".join(f"{v:.0f}" for v in FL.wait_med_ms)
                 + "; rev-up " + " / ".join(f"{v:.0f}" for v in FL.revup_med_ms))
    # S10E live: angle_latency on the live track table; then this script's aim / latency on the live (ungated aim,
    # ref-gated latency) inputs
    TL = pd.read_csv(TRACK_LIVE)
    r_live = angle_latency(TL)
    lines.append("S10E live track table: " + r_live.round(3).to_string(index=False).replace("\n", "\n    "))
    g0 = aim_sessions(None)
    Q = TL[TL.epoch != "pre"][["label", "aim"]].merge(g0[["label", "aim"]], on="label", suffixes=("_live", "_me"))
    lines.append(f"aim (ungated, this script) vs live track aim: {len(Q)} post sessions, max |diff| "
                 f"{np.nanmax(np.abs(Q.aim_live - Q.aim_me)):.3g} deg")
    l0 = lat_sessions(T, "g_ref")
    Q = TL[["label", "lat_med_ms"]].merge(l0[["label", "lat_med_ms"]], on="label", suffixes=("_live", "_me"))
    lines.append(f"far-contra median latency (ref gate, this script) vs live track: {len(Q)} sessions, max |diff| "
                 f"{np.nanmax(np.abs(Q.lat_med_ms_live - Q.lat_med_ms_me)):.3g} ms (live table keeps 4 significant digits)")
    (OUT / f"checks_repro_{TAG}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def run(gate: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    T = trials()
    add_jaw(T, CAP_FULL, "")
    info = add_gates(T)
    lines = [f"{k}: {v}" for k, v in info.items()]
    gcol = f"g_{gate}"
    # ---- S10A cells: new rule on the reported gate, and on the production gate (to split rule vs gate)
    allc = []
    for g in ("g_ref", gcol) if gcol != "g_ref" else ("g_ref",):
        C, W = decompose(T, g)
        for name, c in C.items():
            allc.append(c.assign(variant=name, gate=g[2:]))
        if g == gcol:
            Wkeep = W
    CC = pd.concat(allc, ignore_index=True)
    L = pd.read_csv(CELLS_LIVE)[["animal", "pos", "epoch", "n", "lat_med_ms", "wait_med_ms", "revup_med_ms"]]
    CC = pd.concat([CC, L.rename(columns={"n": "n_licked"}).assign(variant="live_S10A", gate="ref_cam4lick")],
                   ignore_index=True)
    CC["mouse"] = CC.animal.map(MOUSE)
    CC.to_csv(OUT / f"s10a_cells_all_{TAG}.csv", index=False, float_format="%.4g")
    prim = CC[(CC.variant == "m3d") & (CC.gate == gate)].drop(columns=["variant", "gate"])
    prim.to_csv(OUT / f"s10a_cells_{TAG}.csv", index=False, float_format="%.4g")   # the render input (S10A bars)
    supp = CC[(CC.variant == "m3d_frpool") & (CC.gate == gate)].drop(columns=["variant", "gate"])
    supp.to_csv(OUT / f"s10a_cells_m3d_frpool_{TAG}.csv", index=False, float_format="%.4g")
    M = CC.groupby(["variant", "gate", "pos", "epoch"])[["n_licked", "lat_med_ms", "wait_med_ms", "revup_med_ms"]] \
        .mean().reset_index()
    M["pos"] = pd.Categorical(M.pos, A.POS, ordered=True)
    M["epoch"] = pd.Categorical(M.epoch, EP, ordered=True)
    M = M.sort_values(["variant", "gate", "pos", "epoch"])
    M.to_csv(OUT / f"s10a_mean_of_mice_{TAG}.csv", index=False, float_format="%.4g")
    # per mouse far contra and old vs new wide table
    keys = ["animal", "mouse", "pos", "epoch"]
    O = CC[CC.variant == "live_S10A"][keys + ["wait_med_ms", "revup_med_ms", "lat_med_ms", "n_licked"]]
    N = prim[keys + ["wait_med_ms", "revup_med_ms", "lat_med_ms", "n_licked"]]
    OV = O.merge(N, on=keys, how="outer", suffixes=("_old", "_new"))
    OV["pos"] = pd.Categorical(OV.pos, A.POS, ordered=True)
    OV["epoch"] = pd.Categorical(OV.epoch, EP, ordered=True)
    OV.sort_values(["pos", "epoch", "animal"]).to_csv(OUT / f"s10a_old_vs_new_per_mouse_{TAG}.csv", index=False,
                                                       float_format="%.4g")
    # shares (which signal set the first movement), supplementary variant
    sh = [c for c in CC.columns if c.startswith("share_")]
    CC[(CC.gate == gate) & CC.variant.isin(["m3d", "m3d_frpool"])].groupby(["variant", "pos", "epoch"])[sh] \
        .mean().round(4).to_csv(OUT / f"s10a_shares_mean_of_mice_{TAG}.csv")
    # per-trial table
    keep = ["label", "animal", "epoch", "day", "trial_id", "pos", "lat_ms", "g_ref", "g_wpcls", "g_wp",
            "cls_final_new", "m3d_ms", "m3_judged", "jaw3d_ms", "mouth_ms", "fr_ms", "frpool_ms",
            "mv_m3d", "src_m3d", "mv_m3d_frpool", "src_m3d_frpool"]
    T2 = T.merge(Wkeep[["label", "trial_id", "mv_m3d", "src_m3d", "mv_m3d_frpool", "src_m3d_frpool"]],
                 on=["label", "trial_id"], how="left")
    T2[keep].to_csv(OUT / f"s10a_trials_{TAG}.csv.gz", index=False, float_format="%.10g")
    # ---- S10E
    lat = lat_sessions(T, gcol)
    aim = aim_sessions(T[T[gcol]][["label", "trial_id"]])
    TR = lat.merge(aim[["label", "aim", "n_aim"]], on="label", how="left")
    TR.to_csv(OUT / f"revup_track_sessions_{TAG}.csv", index=False, float_format="%.6g")
    rn = angle_latency(TR)
    ro = angle_latency(pd.read_csv(TRACK_LIVE))
    E = ro.merge(rn, on="animal", suffixes=("_old", "_new"))
    E.insert(1, "mouse", E.animal.map(MOUSE))
    E.to_csv(OUT / f"angle_latency_per_animal_{TAG}.csv", index=False, float_format="%.4g")
    # split: gate on latency only / aim only
    lat_ref = lat_sessions(T, "g_ref")
    aim_all = aim_sessions(None)
    rows = []
    for nm, la, am in (("lat ref, aim all (live inputs rebuilt)", lat_ref, aim_all), ("lat wp, aim all", lat, aim_all),
                       ("lat ref, aim wp", lat_ref, aim), ("lat wp, aim wp (NEW)", lat, aim)):
        r = angle_latency(la.merge(am[["label", "aim"]], on="label", how="left"))
        rows.append(r.assign(inputs=nm))
    pd.concat(rows).to_csv(OUT / f"angle_latency_split_{TAG}.csv", index=False, float_format="%.4g")
    lines.append("S10E old vs new:\n" + E.round(3).to_string(index=False))
    lines.append("S10E split:\n" + pd.concat(rows).round(3).to_string(index=False))
    for v in ("m3d", "m3d_frpool", "frpool_only"):
        for g in sorted(CC[CC.variant == v].gate.unique()):
            F = far(CC[(CC.variant == v) & (CC.gate == g)])
            lines.append(f"far contra {v} [{g}]: wait " + " / ".join(f"{x:.0f}" for x in F.wait_med_ms)
                         + "; rev-up " + " / ".join(f"{x:.0f}" for x in F.revup_med_ms))
    F = far(CC[CC.variant == "live_S10A"])
    lines.append("far contra LIVE: wait " + " / ".join(f"{x:.0f}" for x in F.wait_med_ms)
                 + "; rev-up " + " / ".join(f"{x:.0f}" for x in F.revup_med_ms))
    (OUT / f"log_run_{TAG}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["repro", "run"])
    ap.add_argument("--gate", default="wp", choices=["wp", "wpcls", "ref"])
    a = ap.parse_args(argv)
    repro() if a.cmd == "repro" else run(a.gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
