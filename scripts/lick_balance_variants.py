"""Which hemispheric BALANCE definition tracks recovery? (Priya 10-28: "we may need to try more possible balance decoders --
ratio, temporal binned ratio, temporal binned difference, AUC ratio, AUC difference"; and "I really want to think about
how to predict / what is correlating with ITI lick recovery -- the decoder still works on it, so it's somewhere").

Per lick (the frozen KEY decoder's lick set, -1..+1 s at 50 ms, x1e3 dF/F, drift-removed, no per-lick baseline in the
stored traces), per orofacial area (ALM, tjM1, SSp-m, SSp-n; pool = their mean) and hemisphere (_right = contralesional,
_left = ipsilesional):
  raw   window mean of the stored trace (the current balance = raw 0..0.2 s)
  bs    window mean minus the -1.0..-0.5 s baseline (il.BASE_RESP)
  auc   bs over 0..0.2 / 0..0.5 / 0..1.0 s
in 8 time bins (-0.5..-0.2, -0.2..0, 0..0.1, 0.1..0.2, 0.2..0.35, 0.35..0.5, 0.5..0.75, 0.75..1.0 s) and the 3 AUC windows.
Every lick value is referenced as the balance is: per animal x lick set x lick order, the PRE line on lick angle (pre
sessions leave-one-pre-session-out; lick_recovery_markers._residuals).
  DIFFERENCE  session mean of the referenced (contra - ipsi)
  RATIO       angle-matched GAIN ratio: log2(sum contra / sum of its pre-line prediction) - log2(same for ipsi), i.e. how
              much each side changed relative to what pre licks at the same angle & order predict; NaN when a predicted
              or observed sum is <= 0 (the ratio needs positive values: used for bs / auc, not raw levels). A plain session
              ratio would partly follow which angles were licked, which is circular for the ITI angle axis.
Also: the ITI / cued like-for-like decoder OFFSET (balreg_sessions.csv).
Targets: cued sets -> far-contra first-lick aim; ITI sets -> ITI's own axis (iti_ang: session mean ITI lick angle vs
pre) and the cued aim. Models as lick_recovery_markers.models (per-animal r, within-animal MixedLM with log(day), LOAO
with a within-animal shuffle null, leave-one-epoch-out, LOSO); shuffle reduced to N_SHUF.
Outputs <OUT_W>/balvar_sessions.csv, balvar_cv.csv, balvar_models.csv; figure lick_balance_variants.png
(figdata + --plot-only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
import lick_balance_regions as lbg  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

ORO = lrm.ORO
BINS = [(-0.5, -0.2), (-0.2, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.75), (0.75, 1.0)]
AUCW = [(0.0, 0.2), (0.0, 0.5), (0.0, 1.0)]
N_SHUF = 500
SETS = {"trial": "cued", "trial_1": lrm.C1SET, "iti": "ITI", "iti_q": lrm.QSET}
TARGETS = {"cued": ["aim"], lrm.C1SET: ["aim"], "ITI": ["iti_ang", "aim"], lrm.QSET: ["iti_ang", "aim"]}


def _bl(lo_: float, hi_: float) -> str:
    return f"{lo_:+.2f}_{hi_:+.2f}"


def _wmask(lo_: float, hi_: float) -> np.ndarray:
    L = il.LAGS_D
    return (L >= lo_ - 1e-9) & ((L < hi_ - 1e-9) if hi_ < L.max() else (L <= hi_ + 1e-9))


def _session(label: str) -> pd.DataFrame:
    acts = lbr.cv._acts(label)
    d = il._load(label)
    cols = list(d["cols"])
    base_m = (il.LAGS_D >= il.BASE_RESP[0] - 1e-9) & (il.LAGS_D <= il.BASE_RESP[1] + 1e-9)
    out = []
    for st, (xk, sk) in {"iti": ("iti_X", "iti_side"), "trial": ("tr_X", "tr_side")}.items():
        act = np.asarray(acts.get(st, []), float)
        if not len(act):
            continue
        m = il.subseq_match(d[sk].astype(float), act)
        X = d[xk].astype(float)
        R = {"set": st, "label": label, "k": np.arange(len(act))}
        ok = m >= 0
        mm = np.maximum(m, 0)
        for a in ORO:
            for side, h in (("c", "right"), ("i", "left")):
                tr = X[:, :, cols.index(f"{a}_{h}")]
                base = tr[:, base_m].mean(1)
                for lo_, hi_ in BINS:
                    v = tr[:, _wmask(lo_, hi_)].mean(1)
                    R[f"raw{_bl(lo_, hi_)}|{a}|{side}"] = np.where(ok, v[mm], np.nan)
                    R[f"bs{_bl(lo_, hi_)}|{a}|{side}"] = np.where(ok, (v - base)[mm], np.nan)
                for lo_, hi_ in AUCW:
                    v = (tr[:, _wmask(lo_, hi_)].mean(1) - base) * (hi_ - lo_)
                    R[f"auc{_bl(lo_, hi_)}|{a}|{side}"] = np.where(ok, v[mm], np.nan)
        out.append(pd.DataFrame(R))
    return pd.concat(out, ignore_index=True)


def sessions() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), _session)
    if fail:
        raise RuntimeError(f"failed for {len(fail)} sessions: {fail[:5]}")
    P = P.merge(pd.concat([v for _i, v in res], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    P["ord_c"] = np.minimum(P.order, lbg.CAP)
    feats = sorted({c.split("|")[0] for c in P.columns if "|" in c})
    # pool = mean of the 4 areas; per-area variables kept only for the 0..1 s AUC and the current raw 0..0.2 balance window
    keep_area = {f"auc{_bl(0.0, 1.0)}", f"raw{_bl(0.0, 0.1)}", f"raw{_bl(0.1, 0.2)}"}
    vs = []
    for f in feats:
        for s in ("c", "i"):
            P[f"{f}|oro|{s}"] = P[[f"{f}|{a}|{s}" for a in ORO]].mean(1)
        for a in ["oro", *ORO]:
            if a != "oro" and f not in keep_area:
                continue
            P[f"{f}|{a}|d"] = P[f"{f}|{a}|c"] - P[f"{f}|{a}|i"]
            vs += [f"{f}|{a}|{s}" for s in ("c", "i", "d")]
    gres, gfail = ak.fan_sessions(sorted(P[P.set == "iti"].label.unique()), lrm._gaps)
    if gfail:
        raise RuntimeError(f"gaps failed: {gfail[:5]}")
    G = pd.concat([v for _i, v in gres], ignore_index=True)
    Q = P[P.set == "iti"].merge(G, on=["set", "label", "k"], how="left", validate="one_to_one")
    Q = Q[Q.gap >= il.GAP_MAIN].assign(set="iti_q").drop(columns="gap")
    Q1 = P[(P.set == "trial") & (P.order == 0)].assign(set="trial_1")
    P = pd.concat([P, Q, Q1], ignore_index=True)
    P = lrm._residuals(P, vs)
    # session table: DIFFERENCE = mean referenced (c - i); RATIO = log2 gain contra - log2 gain ipsi (bs / auc only)
    rows = []
    for (st, lab, an, ep), g in P.groupby(["set", "label", "animal", "epoch"]):
        rec = {"set": SETS[st], "label": lab, "animal": an, "epoch": ep}
        for v in {x.rsplit("|", 1)[0] for x in vs}:
            rec[f"diff:{v}"] = g[f"res_{v}|d"].mean()
            if not v.startswith("raw"):
                gains = []
                for s in ("c", "i"):
                    obs = g[f"{v}|{s}"]
                    pred = obs - g[f"res_{v}|{s}"]
                    okk = obs.notna() & pred.notna()
                    so, sp = obs[okk].sum(), pred[okk].sum()
                    gains.append(np.log2(so / sp) if (so > 0 and sp > 0) else np.nan)
                rec[f"ratio:{v}"] = gains[0] - gains[1]
        rows.append(rec)
    S = pd.DataFrame(rows)
    B = pd.read_csv(il.OUT_W / "balreg_sessions.csv", usecols=["set", "label", "offset"])
    B["set"] = B.set.replace({"cued": "cued", "ITI": "ITI"})
    S = S.merge(B.rename(columns={"offset": "offset:decoder"}), on=["set", "label"], how="left")
    R = pd.read_csv(il.OUT_W / "recmark_sessions.csv", usecols=["set", "label", "day", "aim", "iti_ang"])
    S = S.merge(R, on=["set", "label"], how="left")
    return S


def compute(models_only: bool = False) -> dict:
    if models_only:                                    # re-fit the models on the saved session table
        S = pd.read_csv(il.OUT_W / "balvar_sessions.csv")
    else:
        S = sessions()
        S.to_csv(il.OUT_W / "balvar_sessions.csv", index=False)
    # self-check: diff of the current window (raw 0..0.2) ~ the published balance (window differs slightly: 0..0.2 incl.)
    chk = S[["set", "label", f"diff:raw{_bl(0.0, 0.1)}|oro", f"diff:raw{_bl(0.1, 0.2)}|oro"]].merge(
        pd.read_csv(il.OUT_W / "recmark_sessions.csv", usecols=["set", "label", "oro_bal"]), on=["set", "label"])
    cur = chk[[f"diff:raw{_bl(0.0, 0.1)}|oro", f"diff:raw{_bl(0.1, 0.2)}|oro"]].mean(1)
    print(f"== self-check: raw 0..0.2 diff vs published balance r = {np.corrcoef(cur, chk.oro_bal)[0, 1]:.3f} "
          f"(n {len(chk)})", flush=True)
    lrm.N_SHUF = N_SHUF
    out = {k: [] for k in ("M", "CV", "CVA", "AN")}
    keep = ["set", "label", "animal", "epoch", "day"]
    preds = [c for c in S.columns if ":" in c]
    for st, ys in TARGETS.items():
        for y in ys:
            T = S[S.set == st][[*keep, *preds]].assign(aim=S.loc[S.set == st, y])
            F = lrm.models(T)
            for k in out:
                out[k].append(F[k].assign(y=y))
            print(f"[{st} -> {y}] done", flush=True)
    F = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    F["CV"].to_csv(il.OUT_W / "balvar_cv.csv", index=False)
    F["M"].to_csv(il.OUT_W / "balvar_models.csv", index=False)
    pd.set_option("display.width", 240)
    top = F["CV"].sort_values("loao", ascending=False).groupby(["set", "y"]).head(8)
    print("== best LOAO per set x target\n" + top[["set", "y", "pred", "loao", "p_shuffle", "loblk", "loso_unknown"]]
          .round(3).to_string(index=False))
    print("== per animal r, top predictors (rule 8)\n" + F["AN"].merge(top[["set", "y", "pred"]], on=["set", "y", "pred"])
          .pivot_table(index=["set", "y", "pred"], columns="animal", values="r").round(2).to_string())
    return {"S": S, **F}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CV, M = F["CV"], F["M"]
    panels = [(st, y) for st, ys in TARGETS.items() for y in ys]
    mid = [(a + b) / 2 for a, b in BINS]
    fig, axs = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 8.5), squeeze=False)
    for j, (st, y) in enumerate(panels):
        c = CV[(CV.set == st) & (CV.y == y)].set_index("pred")
        m = M[(M.set == st) & (M.y == y)].set_index("pred")
        dd = c.loc["days", "loao"] if "days" in c.index else np.nan
        for i, area in enumerate(("oro", None)):
            ax = axs[i, j]
            if area is None:                                   # AUC windows + areas, difference vs ratio
                labs, vals_d, vals_r = [], [], []
                for lo_, hi_ in AUCW:
                    labs.append(f"AUC {lo_:g}-{hi_:g}")
                    vals_d.append(c.loc[f"diff:auc{_bl(lo_, hi_)}|oro", "loao"])
                    vals_r.append(c.loc[f"ratio:auc{_bl(lo_, hi_)}|oro", "loao"])
                for a in ORO:
                    labs.append(f"{a} AUC 0-1")
                    vals_d.append(c.loc[f"diff:auc{_bl(0.0, 1.0)}|{a}", "loao"])
                    vals_r.append(c.loc[f"ratio:auc{_bl(0.0, 1.0)}|{a}", "loao"])
                labs.append("decoder offset")
                vals_d.append(c.loc["offset:decoder", "loao"] if "offset:decoder" in c.index else np.nan)
                vals_r.append(np.nan)
                x = np.arange(len(labs))
                ax.bar(x - 0.2, vals_d, width=0.38, color="#C44E52", label="difference")
                ax.bar(x + 0.2, vals_r, width=0.38, color="#4C72B0", label="ratio (gain)")
                ax.set_xticks(x, labs, rotation=45, ha="right", fontsize=7)
                ax.set_title(f"{st} -> {y}: windows / areas (LOAO cvR²)", fontsize=9)
            else:
                for kind, col, lab in (("diff:raw", "0.3", "raw difference"), ("diff:bs", "#C44E52", "baseline-sub difference"),
                                       ("ratio:bs", "#4C72B0", "baseline-sub ratio (gain)")):
                    vals = [c.loc[f"{kind}{_bl(a, b)}|oro", "loao"] if f"{kind}{_bl(a, b)}|oro" in c.index else np.nan
                            for a, b in BINS]
                    ax.plot(mid, vals, "o-", color=col, label=lab, ms=4)
                    for xm, (a, b) in zip(mid, BINS, strict=True):
                        k = f"{kind}{_bl(a, b)}|oro"
                        if k in m.index and "p" in m and np.isfinite(m.loc[k, "p"]) and m.loc[k, "p"] < 0.05:
                            ax.plot(xm, c.loc[k, "loao"], "o", mfc="none", mec=col, ms=10)
                ax.axvline(0, color="0.6", lw=0.6)
                ax.set_xlabel("time from lick (s), bin center", fontsize=8)
                ax.set_title(f"{st} -> {y}: orofacial, by time bin (LOAO cvR²)", fontsize=9)
            ax.axhline(dd, color="0.4", ls="--", lw=1, label="days since stroke")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylim(-1, 1)
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("Which BALANCE definition estimates recovery in an UNSEEN animal (leave-one-animal-out cvR²)? Top: by "
                 "time bin around the lick (open ring = within-animal mixed-model p < 0.05 with log day). Bottom: response "
                 "AUC windows and areas, DIFFERENCE vs angle-matched RATIO, and the decoder offset.\nAll values at the same "
                 "lick angle & order as pre. Targets: cued -> far-contra first-lick aim; ITI -> ITI's own lick-angle axis "
                 "(iti_ang) or the cued aim. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT_W / "figures" / "lick_balance_variants.png", dpi=90)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_balance_variants"))
        return 0
    F = compute(models_only="--models-only" in sys.argv)
    save_fd("lick_balance_variants", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
