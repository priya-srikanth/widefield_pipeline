"""Window or model? The bout-summary movement regression (WF_SS_FR=KBs) applied to the FROZEN ENCODER's own per-trial
data, in the encoder's window (Priya 11-09 "yes run": item 40 left open whether the encoder removes more of the acute
far-contra deficit because of its lick-locked movement model or because its window, -0.1..+1.0 s after the first lick,
includes the late contact-dependent response that the 0.1-0.5 s maps mostly exclude).

Same trials, same window, same area targets (run `frozen_encoder_gate-attempted_v7g3_t3d_spont_frA`, `trial_parts`,
lick window, 58 Allen areas), three conditions:
  data     the measured values (trials that also have the KBs covariates)
  -enc     minus the encoder's predicted lick + exec_locked + exec_within + contact + fr + fr_dir (item 40's "-both")
  -KBs     minus the KBs regression: per animal a PRE-ONLY least-squares fit of the trial values on the 15 bout-summary
           covariates (8 kinematics + contact, 7 FaceRhythm, `state_space_chronic_maps` KBs), covariates minus the pre mean
           at the trial's spout, pre sessions leave-session-out -- exactly `apply_fr`'s KBs, on these targets
Encoder trials join the extract rows on cue time (< 20 ms). Readouts as item 40: spout-specific (spout minus the
session's mean over spouts), change vs own pre in units of the animal's pre data RMS; kept along the data change (per
animal, cohort mean); SSp-m / SSp-n ipsi far contra with `boot_delta`. If -KBs keeps about what it keeps on the maps
(~.75) while -enc keeps ~.35 on the SAME trials and window, the difference is the movement MODEL; if -KBs drops to ~.35
here, it is the WINDOW.

    WF_SS_ALIGN=lick WF_SS_BASIS=svd WF_SS_MAPBASE=rest python scripts/state_space_encoder_window_kbs.py
Output: <state_space_recovery>/encoder_subtract/window_kbs.csv. EXPLORATORY.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["WF_SS_FR"] = "KBs"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_chronic_maps as C  # noqa: E402
import state_space_encoder_subtract as E  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402

ENC = ["lick", "exec_locked", "exec_within", "contact", "fr", "fr_dir"]


def main():
    if S.ALIGN != "lick":
        sys.exit("run with WF_SS_ALIGN=lick (the KBs covariates' cache key)")
    src = E.run_dir()
    sess = {}
    for f in sorted(src.glob("PS9*.npz")):
        lab = f.stem
        if lab.endswith("_trials") or not (S.SVD / f"{lab}.npz").exists():
            continue
        x = np.load(S.SVD / f"{lab}.npz", allow_pickle=True)
        ep = S._epoch(lab, str(x["epoch"]))
        if ep not in ("pre", *E.EPS):
            continue
        cue = x["cue_s"].astype(float)
        Xf = C.fr_covariates(lab, len(cue))
        if Xf is None:
            continue
        X = np.column_stack([C.kin_covariates(lab, len(cue)), C.fr_design(Xf)])
        z = np.load(f, allow_pickle=True)
        t = np.load(src / f"{lab}_trials.npz", allow_pickle=True)
        parts = [str(p) for p in z["parts"]]
        ar = [str(a) for a in z["areas"]]
        ka = [i for i, a in enumerate(ar) if a != "cortex"]
        wl = [str(w) for w in z["trial_windows"]].index("lick")
        tp = z["trial_parts"][:, :, wl][:, :, ka].astype(float)
        ec = t["t_cue_s"].astype(float)
        j = np.abs(ec[:, None] - cue[None]).argmin(1)
        ok = (np.abs(ec - cue[j]) < 0.02) & np.isfinite(X[j]).all(1) & np.isfinite(tp[0]).all(1)
        data = tp[parts.index("data")][ok]
        enc = data - sum(tp[parts.index(p)][ok] for p in ENC)
        sess.setdefault(lab.split("_")[0], []).append(
            {"lab": lab, "ep": ep, "data": data, "-enc": enc, "X": X[j][ok],
             "pos": np.asarray(z["trial_pos"]).astype(str)[ok]})
        E_areas = [ar[i] for i in ka]
    # -KBs: apply_fr's KBs on these targets
    for ss in sess.values():
        pre = [s for s in ss if s["ep"] == "pre"]
        for s in ss:
            train = [q for q in pre if q["lab"] != s["lab"]]
            Xp = np.concatenate([q["X"] for q in train])
            Rp = np.concatenate([q["data"] for q in train])
            pp = np.concatenate([q["pos"] for q in train])
            smu = {sp: Xp[pp == sp].mean(0) for sp in np.unique(pp)}
            Xp = Xp - np.stack([smu[sp] for sp in pp])
            Xt = np.nan_to_num(s["X"] - np.stack([smu.get(sp, np.full(Xp.shape[1], np.nan)) for sp in s["pos"]]))
            mu = Xp.mean(0)
            B = np.linalg.lstsq(np.column_stack([np.ones(len(Xp)), Xp - mu]), Rp, rcond=None)[0][1:]
            s["-KBs"] = s["data"] - (Xt - mu) @ B
    conds = ["data", "-enc", "-KBs"]

    def spouts(s, c):
        out = np.full((6, s["data"].shape[1]), np.nan)
        for i, p in enumerate(S.POS):
            m = s["pos"] == p
            if m.sum() >= E.MIN_TR:
                out[i] = s[c][m].mean(0)
        return out - np.nanmean(out, 0)                                              # spout-specific

    unit = {}
    for an, ss in sess.items():
        P = np.stack([spouts(s, "data") + 0 for s in ss if s["ep"] == "pre"])
        full = np.stack([np.stack([s["data"][s["pos"] == p].mean(0) if (s["pos"] == p).sum() >= E.MIN_TR
                                   else np.full(P.shape[-1], np.nan) for p in S.POS]) for s in ss if s["ep"] == "pre"])
        unit[an] = float(np.sqrt(np.nanmean(np.nanmean(full, 0) ** 2)))
        del P
    rng = np.random.default_rng(0)
    rows = []
    fc = S.POS.index("far_contra")
    for col, ci in [("far_contra", fc), ("close_ipsi", S.POS.index("close_ipsi"))]:
        for ep in E.EPS:
            V = {c: {} for c in conds}
            for an, ss in sess.items():
                for c in conds:
                    pre = np.nanmean([spouts(s, c) for s in ss if s["ep"] == "pre"], 0)
                    post = np.nanmean([spouts(s, c) for s in ss if s["ep"] == ep], 0)
                    V[c][an] = (post - pre)[ci] / unit[an]
            for c in conds[1:]:
                k = {an: float(V[c][an] @ V["data"][an] / (V["data"][an] @ V["data"][an])) for an in sess}
                rows.append({"column": col, "epoch": ep, "condition": c, "readout": "kept",
                             "value": float(np.mean(list(k.values()))), "per_animal": {a: round(v, 2) for a, v in k.items()}})
            for c in conds:
                for area in ["SSp-m_left", "SSp-n_left"]:
                    a = E_areas.index(area)
                    pre = {an: [spouts(s, c)[ci, a] / unit[an] for s in ss if s["ep"] == "pre"] for an, ss in sess.items()}
                    post = {an: [spouts(s, c)[ci, a] / unit[an] for s in ss if s["ep"] == ep] for an, ss in sess.items()}
                    pre = {x: [v for v in vv if np.isfinite(v)] for x, vv in pre.items()}
                    post = {x: [v for v in vv if np.isfinite(v)] for x, vv in post.items() if pre.get(x)}
                    iv = kit.boot_delta({x: v for x, v in post.items() if v}, pre, rng)
                    per = {x: round(float(np.mean(post[x]) - np.mean(pre[x])), 2) for x in post if post[x]}
                    rows.append({"column": col, "epoch": ep, "condition": c, "readout": area, "value": iv.point,
                                 "lo": iv.lo, "hi": iv.hi, "per_animal": per})
    T = pd.DataFrame(rows)
    T.to_csv(E.OUT / "window_kbs.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 2, "display.max_rows", 200):
        print(f"trials matched: {sum(len(s['data']) for ss in sess.values() for s in ss)}")
        print(T.to_string(index=False))


if __name__ == "__main__":
    main()
