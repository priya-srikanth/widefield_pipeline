"""Fig 5B variants (CC-B 2026-11-26; Priya 11-26 "let's make both"): far-contralesional lick-aligned face-speed maps
(A) on SUCCESSFUL licks only (hits) and (B) ANGLE-MATCHED to each animal's pre-stroke executed 3-D angle.

Live 5B = `face_rhythm_aligned_fourcam.py` align ``lick_4c_wp`` (fused four-camera licks, cam1-only dropped, onset in
cue -2..+4 s of a camera-working-period cue; per-point 2-D image speed from the frozen point PCs, mean over -0.1..+0.2 s
of each lick), position far_R, then `face_rhythm_aligned_maps.py --perm 10000 --perm-points` (log2 vs the animal's
own pre mean, within-animal session-label shuffles, cluster mass |t| > 2).

This script keeps every one of those rules and changes only the lick SET / lick WEIGHTS:

  stage ``licks``  per session (fanned, `analysis_kit.fan_sessions`): every far_R lick of the live 5B set with its
                   own window speed sum per mesh point (S1, sum over its window frames) and its frame count (nf), so
                   any lick subset / weighting of a session map is sum_i w_i S1_i / sum_i w_i nf_i -- exactly the
                   live `window_weights` arithmetic (overlapping windows count per lick). Plus the lick's DAQ
                   contact flag (`pose3d_kinematics.fused_lick_table`: a DAQ contact onset within the lick span +-
                   20 ms; the flag the fused 3-D lick table carries).
                   -> <cache>/perlick/<key>.npz  (cache = a NEW folder under lp_stage; refuses to overwrite)
  stage ``maps``   joins the 3-D executed angle (fused four-camera lineage, `kinematics_fused_postcue/licks.csv.gz`,
                   on session + lick_key), builds the event sets, the per-animal angle weights, the coverage table,
                   and writes a maps npz in the `face_rhythm_aligned_fourcam` format (far_R filled, other positions
                   NaN except the two copied production aligns) for `face_rhythm_aligned_maps.py --maps-npz`.
                   -> <out>/data/...

Aligns written (``maps`` stage):
  cue_wp, lick_4c_wp       copied from the production npz (so the permutation run reproduces the production rows with
                           the same seed / loop order before the variants: plumbing check)
  lick_live_repro          the live set rebuilt from the per-lick sums (must equal lick_4c_wp)
  lick_hit                 (A) live set, DAQ-contact licks only (pre-cue licks included, as live)
  lick_post                post-cue licks only (0..+4 s), all outcomes (= production lick_4c_wpp; reference)
  lick_post_hit            post-cue contact licks
  lick_post_ang            post-cue licks that have a 3-D angle, unweighted (reference of the matched map)
  lick_matched             (B) post-cue licks with a 3-D angle, each post-stroke session's licks weighted per animal x
                           epoch so the epoch's angle histogram equals the animal's pre histogram (pre weights = 1)

    python scripts/fr5b_variants_ccb1126.py licks --cache <new dir> [--jobs 2] [--keys ...]
    python scripts/fr5b_variants_ccb1126.py maps --cache <dir> --out <new dir> [--bin-deg 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POS = "far_R"
LO, HI = -0.1, 0.2
PROD_NPZ = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/"
                "aligned_fourcam_ccB1123/data/aligned_point_maps_fourcam.npz")
LICKS3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60/"
               "kinematics_fused_postcue/licks.csv.gz")
NEW_ALIGNS = ("lick_live_repro", "lick_hit", "lick_post", "lick_post_hit", "lick_post_ang", "lick_matched",
              "lick_matched_all")
EPOCHS = ("pre", "acute", "subacute", "chronic")


def lick_frame_sums(pcs, t, comps, okf, ev):
    """Per lick: (S1 (n, P) = sum of per-point speed over the lick's window frames, nf (n,) = frames). Same frames and
    same per-frame speed as `face_rhythm_aligned_fourcam.speed_sums` with `face.window_weights` (frames with
    t in [ev + LO, ev + HI], ok frames only)."""
    P = comps.shape[1] // 2
    i0 = np.searchsorted(t, ev + LO)
    i1 = np.searchsorted(t, ev + HI, side="right")
    fr = np.unique(np.concatenate([np.arange(a, b) for a, b in zip(i0, i1)] or [np.empty(0, int)])).astype(int)
    fr = fr[(fr >= 1) & (fr < len(t))]
    fr = fr[okf[fr]]
    dt = np.diff(t)
    dv = (pcs[fr] - pcs[fr - 1]) / dt[fr - 1, None]
    sp = np.sqrt(((dv @ comps).reshape(-1, P, 2) ** 2).sum(-1))          # (frames, P)
    good = np.isfinite(sp).all(1)
    pos_of = {f: j for j, f in enumerate(fr)}
    S1 = np.zeros((len(ev), P))
    nf = np.zeros(len(ev))
    for k, (a, b) in enumerate(zip(i0, i1)):
        jj = [pos_of[f] for f in range(a, b) if f in pos_of]
        jj = [j for j in jj if good[j]]
        if jj:
            S1[k] = sp[jj].sum(0)
            nf[k] = len(jj)
    return S1, nf


def job(item: tuple) -> dict:
    fdir, key, cache = item
    import scripts.face_rhythm_aligned_fourcam as F4
    from scripts.pose3d_kinematics import fused_lick_table
    from wfield_local.face_rhythm import common as frc
    from wfield_local.face_rhythm import face as frface
    from wfield_local.paths import PathResolver
    f_out = Path(cache) / "perlick" / f"{key}.npz"
    if f_out.exists():
        raise RuntimeError(f"refusing to overwrite {f_out}")
    fdir = Path(fdir)
    comps, _ = frface.master_points(fdir)
    with np.load(fdir / "proj" / f"{key}.npz", allow_pickle=True) as z:
        animal, date, epoch = str(z["animal"]), str(z["date"]), str(z["epoch"])
        pcs, t = z["points_pc"].astype(np.float32), z["points_t_daq_s"].astype(float)
    label = frc.label(animal, date)
    C = pd.read_csv(F4.CLASS_TABLE, usecols=["label", "cue_s", "cls_final", "position"])
    C = C[C.label == label].sort_values("cue_s")
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    C = C[(C.cue_s >= t0) & (C.cue_s <= t1)]
    cue_all = C.cue_s.to_numpy(float)
    pos_all = C.position.astype(str).to_numpy()
    wp = ~np.isin(C.cls_final.astype(str).to_numpy(), F4.NOT_WP)
    A = fused_lick_table(PathResolver(), animal, date)              # = session_licks(label) + DAQ contact flag
    A = A[np.isfinite(A.daq_on_s.to_numpy(float))].sort_values("daq_on_s", kind="stable").reset_index(drop=True)
    ton = A.daq_on_s.to_numpy(float)
    k = F4.assign_licks(ton, cue_all, wp)
    sel = k >= 0
    sel[sel] = pos_all[k[sel]] == POS
    L = A[sel].reset_index(drop=True)
    kk = k[sel]
    ev = ton[sel]
    okf = np.r_[False, np.isfinite(pcs[1:]).all(1) & np.isfinite(pcs[:-1]).all(1)]
    S1, nf = lick_frame_sums(pcs, t, comps, okf, ev)
    f_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f_out, S1=S1.astype(np.float32), nf=nf, daq_on_s=ev, rel_s=ev - cue_all[kk], cue_ix=kk,
             trial_id=L.trial_id.to_numpy(), lick_key=np.array(L.lick_key.astype(str).tolist(), dtype=str),
             contact=L.contact.astype(bool).to_numpy(), onset_context=np.array(L.onset_context.astype(str).tolist(), dtype=str),
             animal=animal, date=date, epoch=epoch, label=label, key=key)
    return {"key": key, "label": label, "animal": animal, "epoch": epoch, "n_far_R": int(sel.sum()),
            "n_contact": int(L.contact.sum()), "n_post": int((ev - cue_all[kk] >= 0).sum())}


def cmd_licks(a) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.face_rhythm import common as frc
    from wfield_local.paths import PathResolver
    cache = Path(a.cache)
    if (cache / "perlick").exists() and not a.keys:
        raise SystemExit(f"refusing: {cache / 'perlick'} exists (a run writes only to its own new folder)")
    rv = PathResolver()
    fdir = fro.factor_dir(rv, fro.BUNDLE, fro.SET)
    keys = a.keys or frc.session_keys(fdir)
    got, fail = ak.fan_sessions([(str(fdir), k, str(cache)) for k in keys], job, jobs=a.jobs, key=lambda it: it[1],
                                label="fr5b")
    R = pd.DataFrame([r for _, r in got])
    if fail:
        print("FAILED:", [f[0] for f in fail], flush=True)
    R.to_csv(cache / ("perlick_counts_test.csv" if a.keys else "perlick_counts.csv"), index=False)
    print(R.to_string(), flush=True)
    return 0


def angle_bins(bin_deg: float) -> np.ndarray:
    return np.arange(-90.0, 90.0 + bin_deg, bin_deg)


def match_weights(T: pd.DataFrame, edges: np.ndarray, min_bin: int = 1,
                  min_cover: float = 0.0) -> tuple[pd.Series, pd.DataFrame]:
    """Per lick weight (post-cue licks with an angle): pre = 1; post = p_pre(b) / p_post(animal, epoch, b) for bins
    the animal's pre populates AND that hold >= ``min_bin`` post licks of the animal x epoch; 0 otherwise. Bins with
    pre mass but no usable post licks get no weight (their pre mass is the uncovered fraction). An animal x epoch whose
    covered pre mass is < ``min_cover`` is not matched (weights 0; ``used`` False). -> (weights aligned to T.index,
    coverage table). Pure."""
    T = T.assign(b=np.digitize(T.angle_deg.to_numpy(float), edges) - 1)
    w = pd.Series(0.0, index=T.index)
    rows = []
    for an, g in T.groupby("animal"):
        pre = g[g.epoch == "pre"]
        p_pre = pre.b.value_counts(normalize=True)
        w[pre.index] = 1.0
        for ep in EPOCHS[1:]:
            q = g[g.epoch == ep]
            if not len(q):
                continue
            cnt = q.b.value_counts()
            p_post = cnt / cnt.sum()
            shared = p_pre.index.intersection(cnt[cnt >= min_bin].index)
            wb = (p_pre[shared] / p_post[shared])
            used = bool(p_pre[shared].sum() >= min_cover)
            wq = q.b.map(wb).fillna(0.0) * float(used)
            w[q.index] = wq.to_numpy()
            ww = wq.to_numpy()
            n_eff = float(ww.sum() ** 2 / (ww ** 2).sum()) if (ww > 0).any() else 0.0
            rows.append({"animal": an, "epoch": ep, "used": used, "n_pre": len(pre), "n_post": len(q),
                         "n_post_weighted": int((ww > 0).sum()),
                         "frac_post_in_pre_bins": float((ww > 0).mean()),
                         "pre_mass_covered": float(p_pre[shared].sum()),
                         "n_eff": n_eff, "n_eff_frac": n_eff / len(q),
                         "max_weight": float(ww.max()),
                         "pre_median_deg": float(pre.angle_deg.median()), "post_median_deg": float(q.angle_deg.median()),
                         "matched_post_median_deg": float(_wmedian(q.angle_deg.to_numpy(float), ww)),
                         "n_sessions": q.label.nunique()})
    return w, pd.DataFrame(rows)


def _wmedian(x, w):
    m = w > 0
    if not m.any():
        return np.nan
    o = np.argsort(x[m])
    c = np.cumsum(w[m][o])
    return float(x[m][o][np.searchsorted(c, c[-1] / 2)])


def cmd_maps(a) -> int:
    from wfield_local import config
    cache, out = Path(a.cache), Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists (a run writes only to its own new folder)")
    (out / "data").mkdir(parents=True)
    Z = np.load(PROD_NPZ)
    keys = list(Z["key"].astype(str))
    positions = list(Z["positions"].astype(str))
    pi = positions.index(POS)
    prod_aligns = list(Z["aligns"].astype(str))
    # per-lick table
    rows, S1s, NFs = [], [], []
    for i, k in enumerate(keys):
        with np.load(cache / "perlick" / f"{k}.npz") as z:
            rows.append(pd.DataFrame({"row": i, "key": k, "label": str(z["label"]), "animal": str(z["animal"]),
                                      "epoch": str(z["epoch"]), "date": str(z["date"]), "lick_key": z["lick_key"],
                                      "daq_on_s": z["daq_on_s"], "rel_s": z["rel_s"], "contact": z["contact"],
                                      "onset_context": z["onset_context"], "nf": z["nf"]}))
            S1s.append(z["S1"].astype(np.float64))
            NFs.append(z["nf"])
            assert str(z["label"]) == str(Z["label"][i])
    T = pd.concat(rows, ignore_index=True)
    S1 = np.concatenate(S1s)
    T["session"] = T.animal + "_" + T.date
    K = pd.read_csv(LICKS3D, usecols=["session", "lick_key", "position", "context", "daq_peak_s", "side_ml", "out_ap",
                                      "down_dv", "contact", "kin_ok"])
    K = K[K.position == POS].drop_duplicates(["session", "lick_key"])
    K["angle_deg"] = np.degrees(np.arctan2(K.side_ml, np.hypot(K.out_ap, K.down_dv)))   # + = contralesional
    K = K.rename(columns={"contact": "contact_3d", "position": "position_3d"})
    T = T.merge(K[["session", "lick_key", "angle_deg", "contact_3d", "kin_ok", "context"]], on=["session", "lick_key"],
                how="left", validate="many_to_one")
    T["post"] = T.rel_s >= 0
    T["in3d"] = T.contact_3d.notna()
    T["has_angle"] = T.angle_deg.notna()
    T["early_pre"] = (T.epoch == "pre") & T.date.map(config.early_pre)
    # contact agreement with the 3-D table's flag (same DAQ rule): plumbing check
    m = T.in3d
    agree = float((T.contact[m] == T.contact_3d[m].astype(bool)).mean()) if m.any() else np.nan
    edges = angle_bins(a.bin_deg)
    TA = T[T.post & T.has_angle]
    w, cov = match_weights(TA, edges, a.min_bin, a.min_cover)
    T["w_match"] = 0.0
    T.loc[TA.index, "w_match"] = w
    w1, cov1 = match_weights(TA, edges, 1, 0.0)            # sensitivity: every bin with a post lick, no cut
    T["w_match_all"] = 0.0
    T.loc[TA.index, "w_match_all"] = w1
    cov = cov.merge(cov1.drop(columns="used").add_suffix("_allbins").rename(
        columns={"animal_allbins": "animal", "epoch_allbins": "epoch"}), on=["animal", "epoch"])
    sets = {"lick_live_repro": np.ones(len(T)),
            "lick_hit": T.contact.to_numpy(float),
            "lick_post": T.post.to_numpy(float),
            "lick_post_hit": (T.post & T.contact).to_numpy(float),
            "lick_post_ang": (T.post & T.has_angle).to_numpy(float),
            "lick_matched": T.w_match.to_numpy(float),
            "lick_matched_all": T.w_match_all.to_numpy(float)}
    nS = len(keys)
    P = S1.shape[1]
    aligns = ["cue_wp", "lick_4c_wp", *NEW_ALIGNS]
    maps = np.full((nS, len(aligns), 6, P), np.nan, np.float32)
    counts = np.zeros((nS, len(aligns), 6), int)
    for j, al in enumerate(("cue_wp", "lick_4c_wp")):
        maps[:, j] = Z["maps"][:, prod_aligns.index(al)]
        counts[:, j] = Z["counts"][:, prod_aligns.index(al)]
    nf = T.nf.to_numpy(float)
    for j, al in enumerate(NEW_ALIGNS, start=2):
        wv = sets[al]
        for i in range(nS):
            ii = np.flatnonzero((T.row.to_numpy() == i) & (wv > 0))
            counts[i, j, pi] = len(ii)
            den = (wv[ii] * nf[ii]).sum()
            if den > 0:
                maps[i, j, pi] = (wv[ii, None] * S1[ii]).sum(0) / den
    repro = np.nanmax(np.abs(maps[:, 2, pi] - maps[:, 1, pi]) / np.maximum(np.abs(maps[:, 1, pi]), 1e-12))
    cnt_ok = bool((counts[:, 2, pi] == counts[:, 1, pi]).all())
    print(f"live repro: max rel diff {repro:.3g}; counts equal {cnt_ok}; contact vs 3-D flag agreement {agree:.4f}")
    wins = np.array([(0.0, 2.0)] + [(LO, HI)] * (len(aligns) - 1))
    npz = out / "data" / "fr5b_variant_maps_ccB1126.npz"
    np.savez(npz, maps=maps, counts=counts, key=Z["key"], label=Z["label"], animal=Z["animal"], epoch=Z["epoch"],
             aligns=np.array(aligns), positions=Z["positions"], windows=wins, home=Z["home"],
             tmpl=Z["tmpl"], speed_response=Z["speed_response"])
    T.drop(columns=["row"]).to_csv(out / "data" / "fr5b_far_contra_licks_ccB1126.csv.gz", index=False)
    cov.to_csv(out / "data" / "fr5b_angle_coverage_ccB1126.csv", index=False)
    # per animal x epoch counts
    g = T.groupby(["animal", "epoch"])
    C = pd.DataFrame({"sessions": g.label.nunique(), "live_licks": g.size(), "hits": g.contact.sum(),
                      "pre_cue": g.post.apply(lambda s: int((~s).sum())), "post_cue": g.post.sum(),
                      "post_hits": (T.post & T.contact).groupby([T.animal, T.epoch]).sum(),
                      "post_with_angle": (T.post & T.has_angle).groupby([T.animal, T.epoch]).sum(),
                      "matched_weighted": (T.w_match > 0).groupby([T.animal, T.epoch]).sum(),
                      "matched_all_weighted": (T.w_match_all > 0).groupby([T.animal, T.epoch]).sum()}).reset_index()
    C["hit_frac"] = C.hits / C.live_licks
    C["epoch"] = pd.Categorical(C.epoch, EPOCHS)
    C = C.sort_values(["animal", "epoch"])
    C.to_csv(out / "data" / "fr5b_lick_counts_ccB1126.csv", index=False)
    pd.set_option("display.width", 220)
    print(C.to_string(index=False))
    print(cov.round(3).to_string(index=False))
    (out / "data" / "fr5b_checks_ccB1126.txt").write_text(
        f"live repro max rel diff {repro:.3g}; counts equal {cnt_ok}\ncontact (fused_lick_table) vs 3-D table contact "
        f"agreement on joined post-cue licks: {agree:.6f} (n = {int(m.sum())})\npost-cue licks joined to 3-D table: "
        f"{int((T.post & T.in3d).sum())} / {int(T.post.sum())}; with angle {int((T.post & T.has_angle).sum())}\n"
        f"bins: {a.bin_deg} deg from -90 to 90; primary matched: bins with >= {a.min_bin} post licks, animal x epoch "
        f"used when >= {a.min_cover} of its pre mass is covered; lick_matched_all = every bin, no cut\n")
    print("->", npz)
    return 0


SUMMARY_ALIGNS = ("lick", "hit", "post", "post_hit", "post_ang", "matched", "matched_all")


def cmd_summary(a) -> int:
    """Per animal x epoch (rule 8, before the pooled maps), then pooled: per variant, the log2 change vs own pre
    averaged over (i) the whole face, (ii) the contralesional / ipsilesional halves, (iii) the live-5B ringed points of
    that epoch split by sign (fixed region from the live map), plus ringed-point counts and the spatial correlation of
    each variant map with the live map. -> <out>/data/fr5b_per_animal_ccB1126.csv, fr5b_pooled_ccB1126.csv."""
    import scripts.face_rhythm_aligned_maps as AM
    from wfield_local import face_rhythm_readouts as frr
    out = Path(a.out)
    Z = np.load(out / "data" / "fr5b_variant_maps_ccB1126.npz")
    PP = pd.read_csv(out / "data" / "aligned_maps_perm_points.csv")
    PP = PP[PP["position"] == POS]
    maps, counts = Z["maps"].astype(float), Z["counts"]
    animal, epoch, label = Z["animal"].astype(str), Z["epoch"].astype(str), Z["label"].astype(str)
    aligns = list(Z["aligns"].astype(str))
    names = dict(zip(aligns, ["cue", "lick", "live_repro", "hit", "post", "post_hit", "post_ang", "matched",
                              "matched_all"]))
    pi = list(Z["positions"].astype(str)).index(POS)
    contra = Z["home"][:, 0] <= float(frr.MASTER_MIDLINE_X)
    live = {e: PP[(PP["align"] == "lick") & (PP.epoch == e)].sort_values("point") for e in EPOCHS[1:]}
    rows, pooled = [], []
    for al in aligns[1:]:
        nm = names[al]
        if nm not in SUMMARY_ALIGNS:
            continue
        lg, an_s, ep_s = AM.session_log2(maps, counts, label, animal, epoch, aligns.index(al), pi)
        for ep in EPOCHS[1:]:
            L = live[ep]
            up = (L.sig_cluster.to_numpy(bool) & (L.mean_log2.to_numpy() > 0))
            dn = (L.sig_cluster.to_numpy(bool) & (L.mean_log2.to_numpy() <= 0))
            for an in np.unique(an_s):
                m = (an_s == an) & (ep_s == ep)
                if not m.any():
                    continue
                v = np.nanmean(lg[m], 0)
                rows.append({"variant": nm, "epoch": ep, "animal": an, "n_sessions": int(m.sum()),
                             "whole_face": np.nanmean(v), "contra_half": np.nanmean(v[contra]),
                             "ipsi_half": np.nanmean(v[~contra]),
                             "live_up_ring": np.nanmean(v[up]) if up.any() else np.nan,
                             "live_down_ring": np.nanmean(v[dn]) if dn.any() else np.nan})
            g = PP[(PP["align"] == nm) & (PP.epoch == ep)].sort_values("point")
            if not len(g):
                pooled.append({"variant": nm, "epoch": ep, "n_animals": 0})
                continue
            mv, sg = g.mean_log2.to_numpy(), g.sig_cluster.to_numpy(bool)
            pooled.append({"variant": nm, "epoch": ep, "n_animals": int(len(np.unique(an_s[ep_s == ep]))),
                           "whole_face": mv.mean(), "contra_half": mv[contra].mean(), "ipsi_half": mv[~contra].mean(),
                           "ring_up": int((sg & (mv > 0)).sum()), "ring_down": int((sg & (mv <= 0)).sum()),
                           "ring_up_contra": int((sg & (mv > 0) & contra).sum()),
                           "ring_down_contra": int((sg & (mv <= 0) & contra).sum()),
                           "live_up_ring": mv[up].mean() if up.any() else np.nan,
                           "live_down_ring": mv[dn].mean() if dn.any() else np.nan,
                           "overlap_with_live_rings": int((sg & L.sig_cluster.to_numpy(bool)).sum()),
                           "r_with_live_map": float(np.corrcoef(mv, L.mean_log2.to_numpy())[0, 1])})
    R = pd.DataFrame(rows)
    Pd = pd.DataFrame(pooled)
    R.to_csv(out / "data" / "fr5b_per_animal_ccB1126.csv", index=False, float_format="%.4f")
    Pd.to_csv(out / "data" / "fr5b_pooled_ccB1126.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print("PER ANIMAL (log2 change vs own pre):")
    print(R.round(3).to_string(index=False))
    print("POOLED (mean of animals; rings = permutation clusters):")
    print(Pd.round(3).to_string(index=False))
    # staged 5B-style tables (epoch, point, log2_ratio, ring) per variant
    for nm in SUMMARY_ALIGNS:
        T = PP[PP["align"] == nm].sort_values(["epoch", "point"])
        T = T.rename(columns={"mean_log2": "log2_ratio", "sig_cluster": "ring"})[["epoch", "point", "log2_ratio",
                                                                                  "ring"]]
        T.to_csv(out / "data" / f"fig5B_far_contra_lick_maps_{nm}_ccB1126.csv", index=False, float_format="%.17g")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s1 = sp.add_parser("licks")
    s1.add_argument("--cache", required=True)
    s1.add_argument("--jobs", type=int, default=2)
    s1.add_argument("--keys", nargs="*", default=None)
    s2 = sp.add_parser("maps")
    s2.add_argument("--cache", required=True)
    s2.add_argument("--out", required=True)
    s2.add_argument("--bin-deg", type=float, default=5.0)
    s2.add_argument("--min-bin", type=int, default=5, help="post licks a bin needs (per animal x epoch)")
    s2.add_argument("--min-cover", type=float, default=0.5, help="pre mass an animal x epoch must cover")
    s3 = sp.add_parser("summary")
    s3.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"licks": cmd_licks, "maps": cmd_maps, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
