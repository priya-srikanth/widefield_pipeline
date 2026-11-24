"""Fig 5E side specificity with LEAVE-ONE-SESSION-OUT templates and the template WINDOW as a choice (Priya 11-23:
LOO templates; windows 0-2 s (the Fig 5B / S13 maps) and 0-3.5 s (the response window) against the current 0-4 s),
on the four-camera working period.

Templates: per pre-stroke session and spout position, the RMS 2-D point speed after the cue, from the session's frozen
20 point PCs at 25 Hz (`face_rhythm_aligned_fourcam.speed_sums`), over the working-period cues (four-camera trial
classes); per animal the mean over its pre sessions, LEAVING OUT the scored pre session (post sessions use all pre
sessions), minus the mean over the six spouts. Side specificity = mean r with the two own-side templates minus mean r
with the two opposite-side ones (`face_rhythm_cohort_null.side_spec`), on the matched per-session loading
(`face_rhythm_session_shape_tests.load_sessions`; fits unchanged). Test as `scripts/caption_review_1123/fr5e_side_spec_loso.py`
(e88bbb77): within-animal session-label shuffle, 10,000 shuffles, seed 0 per variant, two-sided; Holm over the 24-test
lick family with the other 18 p values from the stored session_scalars.csv.

Variants: prod (stored full-point 0-4 s engaged templates, all pre: plumbing), loso_full04 (stored, LOO = the e88bbb77
result), pc_eng04_loso (PC, engaged, 0-4 s: the PC check), wp04_loso, wp035_loso, wp02_loso (PC, working period).

    python scripts/face_rhythm_sidespec_windows.py [--jobs 4] [--keys ...]   (--keys: timing only, no tests)
-> <readouts>/<set>/aligned_fourcam_ccB1123/sidespec_windows/: data/pre_templates.npz, side_spec_per_session.csv,
   side_spec_tests.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.face_rhythm import face as frface  # noqa: E402

WINDOWS = {"wp02": (0.0, 2.0), "wp035": (0.0, 3.5), "wp04": (0.0, 4.0), "eng04": (0.0, 4.0)}


def job(item):
    fdir, key = item
    import face_rhythm_aligned_fourcam as fa
    fdir = Path(fdir)
    comps, _ = frface.master_points(fdir)
    with np.load(fdir / "proj" / f"{key}.npz", allow_pickle=True) as z:
        pcs, t = z["points_pc"].astype(np.float32), z["points_t_daq_s"].astype(float)
        cues, pos = z["cue_daq_s"].astype(float), z["position"].astype(str)
        animal, date = str(z["animal"]), str(z["date"])
    label = frc.label(animal, date)
    C = pd.read_csv(fa.CLASS_TABLE, usecols=["label", "cue_s", "cls_final", "position"])
    C = C[(C.label == label) & (C.cue_s >= np.nanmin(t)) & (C.cue_s <= np.nanmax(t))]
    wp = ~np.isin(C.cls_final.astype(str).to_numpy(), fa.NOT_WP)
    cw, pw = C.cue_s.to_numpy(float)[wp], C.position.astype(str).to_numpy()[wp]
    okf = np.r_[False, np.isfinite(pcs[1:]).all(1) & np.isfinite(pcs[:-1]).all(1)]
    cols = [(w, p) for w in WINDOWS for p in frc.POSITIONS]
    W = np.column_stack([frface.window_weights(t, (cues[pos == p] if w == "eng04" else cw[pw == p]), WINDOWS[w], okf)
                         for w, p in cols])
    _, S2 = fa.speed_sums(pcs, t, comps, W[:, :1], W)
    ws = W.sum(0)
    out = np.full((len(WINDOWS), 6, S2.shape[0]), np.nan, np.float32)
    for j, (w, p) in enumerate(cols):
        if ws[j] > 0:
            out[list(WINDOWS).index(w), frc.POSITIONS.index(p)] = np.sqrt(S2[:, j] / ws[j])
    return {"key": key, "tmpl": out}


def main(argv=None) -> int:
    import time

    import face_rhythm_cohort_null as N
    import face_rhythm_session_shape_tests as st
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import writeguard
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--keys", nargs="*", default=None)
    a = ap.parse_args(argv)
    st.es.RULE = "v2"
    fdir = frc.default_set_dir(set_name=fro.SET)
    root = frc.readouts_dir(fro.SET)
    out = root / "aligned_fourcam_ccB1123" / "sidespec_windows"
    writeguard.assert_writable(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    S = frc.sessions(fdir, "proj")
    pre_keys = list(S[S.epoch == "pre"].key) if not a.keys else list(a.keys)
    t0 = time.time()
    cache = out / "data" / "pre_templates.npz"
    if a.keys or not cache.exists():
        got, fail = ak.fan_sessions([(str(fdir), k) for k in pre_keys], job, jobs=a.jobs, key=lambda it: it[1],
                                    label="tmpl")
        if fail:
            raise SystemExit(f"failed: {[f[0] for f in fail]}")
        R = {r["key"]: r["tmpl"] for _, r in got}
        print(f"templates: {len(R)} sessions in {time.time() - t0:.0f} s", flush=True)
        if a.keys:
            return 0
        np.savez(cache, key=np.array(list(R)), tmpl=np.stack(list(R.values())), windows=np.array(list(WINDOWS)))
    Z = np.load(cache)
    PCT = dict(zip(Z["key"].astype(str), Z["tmpl"]))
    T, names, _ = st.load_sessions(fdir)                     # side_spec column = production (stored, all pre)
    stored = {}
    for t in S[S.epoch == "pre"].itertuples():
        with np.load(t.file, allow_pickle=True) as z:
            have = list(z["power_positions"].astype(str))
            m = np.full((6, z["speed_response"].shape[1]), np.nan)
            for i, p in enumerate(frc.POSITIONS):
                if p in have:
                    m[i] = z["speed_response"][have.index(p)]
            stored[t.key] = m
    an_of = dict(zip(S.key, S.animal))
    src = {"full04": stored, **{w: {k: v[list(WINDOWS).index(w)].astype(float) for k, v in PCT.items()}
                                for w in WINDOWS}}
    variants = {"prod": ("full04", False), "loso_full04": ("full04", True), "pc_eng04_loso": ("eng04", True),
                "wp04_loso": ("wp04", True), "wp035_loso": ("wp035", True), "wp02_loso": ("wp02", True)}
    L = T[T.unit.isin(st.LICK)].copy()
    per = L[["key", "animal", "date", "epoch", "unit", "recovered"]].copy()
    for v, (s, loso) in variants.items():
        vals = []
        for r in L.itertuples():
            use = [k for k in src[s] if an_of[k] == r.animal and not (loso and k == r.key)]
            m = np.nanmean([src[s][k] for k in use], 0)
            vals.append(N.side_spec(r.load, m - np.nanmean(m, 0, keepdims=True), st.OWN_SIDE[r.unit]))
        per[v] = vals
    chk = np.nanmax(np.abs(per["prod"].to_numpy() - L.side_spec.to_numpy()))
    print("prod side_spec == load_sessions side_spec: max diff", chk, flush=True)
    per.to_csv(out / "side_spec_per_session.csv", index=False, float_format="%.6g")
    Rs = pd.read_csv(root / "session_shape_v2" / "data" / "session_scalars.csv")
    other = Rs[(Rs.family == "lick pair") & (Rs.measure != "side_spec")]
    rows = []
    for v in variants:
        rng = np.random.default_rng(0)
        for ep in frc.POST_EPOCHS:
            for nm in st.LICK:
                d = per[(per.unit == nm) & per.recovered & per.epoch.isin(["pre", ep])]
                groups = [(g[v].to_numpy(float), (g.epoch == "pre").to_numpy()) for _, g in d.groupby("animal")]
                obs, p, n_an = st.shuffle_test(groups, "value", a.perm, rng, "two")
                per_an = {an: float(g[g.epoch == ep][v].mean() - g[g.epoch == "pre"][v].mean())
                          for an, g in d.groupby("animal")}
                rows.append(dict(variant=v, epoch=ep, unit=nm, delta=obs, p=p, n_animals=n_an, **per_an))
    R = pd.DataFrame(rows)
    R["p_holm24"] = np.nan
    for _v, g in R.groupby("variant"):
        R.loc[g.index, "p_holm24"] = ak.holm(np.concatenate([g.p.to_numpy(), other.p.to_numpy()]))[:len(g)]
    R["stars"] = np.where(R.p_holm24 < .01, "**", np.where(R.p_holm24 < .05, "*", ""))
    R.to_csv(out / "side_spec_tests.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))
    print(f"-> {out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
