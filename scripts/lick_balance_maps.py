"""Where does the hemispheric BALANCE carry TASK vs MOVEMENT information? Musall-style alone / unique maps of the per-lick
balance per area pair (Priya 10-28: "can we plot this R2 on the map, a la Musall? ... task alone, task-independent,
task-unique ... make maps as we / they do for that").

Per lick (the frozen KEY-row decoder's lick set; cued licks and ITI licks separately): balance of each bilateral area pair
(split basis: ALM, tjM1, MOs_rest, MOp_rest and every Allen area with both a left and a right ROI) = X_right - X_left in
the decoder's 0..0.2 s window (x1e3 dF/F). Regressor groups:
  TASK      the spout (cued: the target; ITI: the PREVIOUS trial's spout), one-hot
  MOVEMENT  the lick's lateral angle (asin(side / reach) of the 3-D tongue vector), 3-D reach, peak tongue speed
Per animal x epoch x set, ordinary least squares on all pairs at once, cross-validated leave-one-SESSION-out (epochs with
a single session are skipped): cvR2 of FULL (task + movement), TASK ALONE, MOVEMENT ALONE; TASK UNIQUE = full - movement
alone, MOVEMENT UNIQUE = full - task alone (Musall 2019's alone / unique logic). Maps = mean over animals; each pair's value
painted on BOTH hemispheres (it is a pair property). Image left = ipsilesional.
Outputs <OUT_W>/balmap_r2.csv; figures lick_balance_maps_{cued,iti}.png (figdata + --plot-only).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

SETS = {"trial": "cued", "iti": "ITI"}
MEASURES = [("full", "FULL (task + movement)"), ("task_alone", "TASK alone (spout)"),
            ("task_unique", "TASK unique (full - movement)"), ("move_alone", "MOVEMENT alone (angle, reach, speed)"),
            ("move_unique", "MOVEMENT unique (full - task)")]


def _session(label: str) -> pd.DataFrame:
    acts = lbr.cv._acts(label)
    d = il._load(label)
    out = []
    for st, (xk, sk) in {"iti": ("iti_X", "iti_side"), "trial": ("tr_X", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        X = d[xk].astype(float)[:, lbr.W, :].mean(1)
        X, cols = il.with_rest(X, d["cols"], label, basis="split")
        pairs = sorted({c.rsplit("_", 1)[0] for c in cols if c.endswith("_right")} &
                       {c.rsplit("_", 1)[0] for c in cols if c.endswith("_left")} - {"MOs", "MOp"})
        R = pd.DataFrame({"set": st, "label": label, "k": np.arange(len(act))})
        B = np.column_stack([X[:, cols.index(f"{a}_right")] - X[:, cols.index(f"{a}_left")] for a in pairs])
        B = np.where((m >= 0)[:, None], B[np.maximum(m, 0)], np.nan)
        R = pd.concat([R, pd.DataFrame(B, columns=[f"b|{a}" for a in pairs])], axis=1)
        out.append(R)
    return pd.concat(out, ignore_index=True)


def _cv_r2(Y: np.ndarray, Xd: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-group-out cvR2 per column of Y for OLS on Xd (+ intercept); pooled SSE / SST over held-out licks."""
    Xd = np.column_stack([np.ones(len(Xd)), Xd])
    pred = np.full(Y.shape, np.nan)
    for gname in np.unique(groups):
        te = groups == gname
        beta = np.linalg.lstsq(Xd[~te], Y[~te], rcond=None)[0]
        pred[te] = Xd[te] @ beta
    sse = np.nansum((Y - pred) ** 2, 0)
    sst = np.nansum((Y - np.nanmean(Y, 0)) ** 2, 0)
    return 1 - sse / sst


def compute() -> dict:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz",
                    usecols=["set", "label", "k", "reach_mm", "speed_peak", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), _session)
    if fail:
        raise RuntimeError(f"balance failed for {len(fail)} sessions: {fail[:5]}")
    B = pd.concat([v for _i, v in res], ignore_index=True)
    P = P.merge(B, on=["set", "label", "k"], how="left", validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    bcols = [c for c in P if c.startswith("b|")]
    rows = []
    for (st, an, ep), g in P.groupby(["set", "animal", "epoch"]):
        g = g.dropna(subset=["angle", "reach_mm", "speed_peak", *bcols])
        if g.label.nunique() < 2 or len(g) < 100:
            continue
        task = pd.get_dummies(g.pos.astype(str), drop_first=True).to_numpy(float)
        mv = g[["angle", "reach_mm", "speed_peak"]].to_numpy(float)
        mv = (mv - mv.mean(0)) / mv.std(0)
        Y = g[bcols].to_numpy(float)
        grp = g.label.to_numpy()
        r_full = _cv_r2(Y, np.column_stack([task, mv]), grp)
        r_task = _cv_r2(Y, task, grp)
        r_move = _cv_r2(Y, mv, grp)
        for j, c in enumerate(bcols):
            rows.append({"set": SETS[st], "animal": an, "epoch": ep, "pair": c[2:], "n": len(g),
                         "full": r_full[j], "task_alone": r_task[j], "move_alone": r_move[j],
                         "task_unique": r_full[j] - r_move[j], "move_unique": r_full[j] - r_task[j]})
    R = pd.DataFrame(rows)
    R.to_csv(il.OUT_W / "balmap_r2.csv", index=False)
    pd.set_option("display.width", 220)
    key = R[R.pair.isin(["ALM", "tjM1", "SSp-m", "SSp-n", "VISam", "RSPd", "SSp-bfd", "SSp-ul", "ACAd"])]
    print("== per animal (rule 8), PRE: movement alone / task alone cvR2 of the balance")
    print(key[key.epoch == "pre"].pivot_table(index=["set", "pair"], columns="animal",
                                              values=["move_alone", "task_alone"]).round(3).to_string())
    M = R.groupby(["set", "epoch", "pair"])[[m for m, _t in MEASURES]].mean().reset_index()
    from iti_area_maps import split_image
    img, nm = split_image(il.labels()[0])
    return {"M": M, "img": img, "nm": nm}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import outline, paint

    M, img, nm = F["M"], F["img"], F["nm"]
    ol = outline(img)
    for st in ("cued", "ITI"):
        fig, axs = plt.subplots(len(MEASURES), 4, figsize=(15, 3.6 * len(MEASURES)), squeeze=False)
        for i, (m, ttl) in enumerate(MEASURES):
            vals = M[M.set == st]
            vmax = max(np.nanpercentile(vals[m], 99), 0.02)
            for j, ep in enumerate(il.EP4):
                ax = axs[i, j]
                v = vals[vals.epoch == ep].set_index("pair")[m]
                dv = {f"{p}_{h}": float(x) for p, x in v.items() for h in ("left", "right")}
                im = ax.imshow(paint(img, nm, dv), cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                ax.set_title(f"{ttl}\n{ep}", fontsize=8.5)
                ax.axis("off")
            fig.colorbar(im, ax=axs[i, :], fraction=0.015, pad=0.01).set_label("cvR2 of the balance", fontsize=8)
        fig.suptitle(f"{st.upper()} licks: how much of each area pair's hemispheric BALANCE (X_right - X_left, decoder's "
                     "0..0.2 s window) is explained by TASK (spout) vs MOVEMENT (lick angle, reach, speed); Musall-style "
                     "alone / unique cvR2 (leave-one-session-out, per animal x epoch; mean of animals).\nEach pair's value "
                     "is painted on both hemispheres; ALM / tjM1 separate; image left = ipsilesional. PROVISIONAL",
                     fontsize=10)
        from wfield_local import trial_set_text as tst
        with tst.active(None, text=trial_set_sentence(st)):        # this figure's lick set only
            save_png_svg(fig, il.OUT_W / "figures" / f"lick_balance_maps_{st.lower()}.png", dpi=95, bbox_inches="tight")
        plt.close(fig)


def trial_set_sentence(st: str | None = None) -> str:
    """The lick set of lick_balance_maps' figures (Priya 11-09): the KEY decoder's lick set, licks with the lick angle,
    reach, peak speed and every pair balance finite, animal x epoch groups with >= 2 sessions and >= 100 licks (the
    literals of `compute`); ``st`` = "cued" / "ITI" (one figure) or None (both)."""
    import lick_covariate_reference as cv
    keep = ["lick angle, reach, peak speed and every pair balance finite",
            "animal x epoch groups with >= 2 sessions and >= 100 licks"]
    sets = {"cued": ("trial",), "ITI": ("iti",), None: ("iti", "trial")}[st]
    return cv.key_lick_set(iti_extra=keep, cued_extra=keep, sets=sets)


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    if plot_only():
        plot(load_fd("lick_balance_maps"))
        return 0
    F = compute()
    save_fd("lick_balance_maps", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
