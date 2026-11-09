"""Is the SSp-n result (12.17: the SSp-n left/right difference term of the decoder offset tracks recovery within every
animal) EXCITING or BANAL? Three checks (Priya 10-29 "do all 3"):

1. DECODER WITHOUT SSp-n: refit the frozen lateral decoder without the SSp-n features (and, for comparison, without each
   other orofacial area), same fits / like-for-like offset as lick_area_importance (11.12); does the refit decoder's
   session offset still track recovery within animals? If yes, SSp-n is only the decoder's convenient carrier.
2. WEIGHT-FREE AREA COMPARISON: per area, the same-place left/right difference of the scaled features dh_r - dh_l
   (terms_sessions_<tag>.csv; NO decoder weights) per session; within-animal r with recovery, every area ranked.
3. FACE-MOVEMENT CONTROL: per lick, the SSp-n difference term regressed (within animal) on the FaceRhythm factors around
   the lick (covref covariates fr_lick, fr_right, fr_left, fr_close, fr_far); session mean of the residual vs recovery.
Recovery axes as 12.17: ITI -> ITI lick-angle axis (iti_ang) and cued aim; cued -> cued aim (recmark_sessions.csv).
Per-animal r first (rule 8) + within-animal MixedLM with log(day). Outputs <OUT_W>/sspn_check{1,2,3}.csv; figure
lick_sspn_checks.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
import lick_area_importance as lai  # noqa: E402
import lick_refit_decoders as rd  # noqa: E402

TAG = "areas_splitr8"
AN = ["PS92", "PS93", "PS94", "PS95"]
ORO = ["SSp-n", "SSp-m", "ALM", "tjM1", "SSs"]
TARGETS = [("iti", "iti_ang"), ("iti", "aim"), ("trial", "aim")]


def _recovery() -> pd.DataFrame:
    return pd.read_csv(il.OUT_W / "recmark_sessions.csv", usecols=["label", "day", "aim", "iti_ang"]).drop_duplicates("label")


def _track(S: pd.DataFrame, cols: list[str], tag: str) -> pd.DataFrame:
    """per (set, y, col): per-animal r of col with y on post sessions + within-animal MixedLM slope p (log day)."""
    import statsmodels.formula.api as smf
    R = _recovery()
    S = S.merge(R, on="label", how="left")
    out = []
    for st, yv in TARGETS:
        G = S[(S.set == st) & (S.epoch != "pre") & (S.day > 0)].dropna(subset=[yv]).copy()
        G["logday"] = np.log(G.day)
        for c in cols:
            g = G.dropna(subset=[c]).copy()
            if g.animal.nunique() < 4:
                continue
            rr = {a: np.corrcoef(h[c], h[yv])[0, 1] for a, h in g.groupby("animal")}
            g["w"] = g[c] - g.groupby("animal")[c].transform("mean")
            g["b"] = g.groupby("animal")[c].transform("mean")
            g["yy"] = g[yv]
            try:
                p = smf.mixedlm("yy ~ w + b + logday", g, groups=g["animal"]).fit(reml=False).pvalues["w"]
            except Exception:                                          # noqa: BLE001
                p = np.nan
            out.append({"check": tag, "set": st, "y": yv, "measure": c, "p_within": p,
                        "n_same_neg": int(sum(v < 0 for v in rr.values())), **{f"r_{a}": v for a, v in rr.items()}})
    return pd.DataFrame(out)


def check1() -> pd.DataFrame:
    names = list(wd.area_meta()[0])
    base = [n.rsplit("_", 1)[0] for n in names]
    D = rd.load_z()
    drops = {"full": []} | {f"without {a}": [i for i, b in enumerate(base) if b == a] for a in ORO}
    drops["without SSp-n + SSp-m"] = [i for i, b in enumerate(base) if b in ("SSp-n", "SSp-m")]
    rows = []
    for st in rd.SETS:
        for a in AN:
            S = {x: (d["epoch"], *d[st]) for x, d in sorted(D.items()) if d["animal"] == a and len(d[st][1]) >= rd.MIN_LICKS}
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            for mode, drop in drops.items():
                keep = np.setdiff1d(np.arange(len(names)), drop)
                fits = {h: rd._fit(np.vstack([S[q][1][:, keep] for q in pre if q != h]),
                                   np.vstack([S[q][2][:, :1] for q in pre if q != h])) for h in [*pre, None]}
                dec = {x: rd._pred(fits[x if v[0] == "pre" else None], v[1][:, keep])[:, 0] for x, v in S.items()}
                for x, (ep, _X, Y) in S.items():
                    ref = [q for q in pre if q != x]
                    off = lai._offset(np.concatenate([S[q][2][:, 0] for q in ref]), np.concatenate([dec[q] for q in ref]),
                                      Y[:, 0], dec[x])
                    rows.append({"set": st, "animal": a, "label": x, "epoch": ep, "mode": mode,
                                 "pre_r": float(np.corrcoef(dec[x], Y[:, 0])[0, 1]), "offset": float(np.nanmean(off))})
            print(f"[check1 {st} {a}] done", flush=True)
    R = pd.DataFrame(rows)
    W = R.pivot_table(index=["set", "animal", "label", "epoch"], columns="mode", values="offset").reset_index()
    W.columns = [str(c) for c in W.columns]
    pr = R[R.epoch == "pre"].groupby(["set", "mode"]).pre_r.mean().unstack("mode").round(3)
    print("== check 1: held-out pre r per decoder\n" + pr.to_string())
    return _track(W, [c for c in W.columns if c == "full" or c.startswith("without")], "1: decoder without area")


def check2() -> pd.DataFrame:
    T = pd.read_csv(il.OUT_W / "offset_contrib" / f"terms_sessions_{TAG}.csv")
    T["area"] = T.feature.str.rsplit("_", n=1).str[0]
    T["hemi"] = T.feature.str.rsplit("_", n=1).str[1]
    P = T.pivot_table(index=["set", "animal", "label", "epoch", "area"], columns="hemi", values="dh").reset_index()
    P = P.dropna(subset=["right", "left"])
    P["d"] = P.right - P.left                         # weight-free same-place left / right difference (scaled units)
    W = P.pivot_table(index=["set", "animal", "label", "epoch"], columns="area", values="d").reset_index()
    W.columns = [str(c) for c in W.columns]
    areas = [c for c in W.columns if c not in ("set", "animal", "label", "epoch")]
    return _track(W, areas, "2: weight-free difference")


def check3() -> pd.DataFrame:
    L = pd.read_csv(il.OUT_W / "offset_contrib" / f"terms_licks_{TAG}.csv.gz")
    Pr = pd.read_csv(lo.preds_path(), usecols=["set", "label", "act"])
    Pr["k"] = Pr.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz",
                    usecols=["set", "label", "k", "fr_lick", "fr_right", "fr_left", "fr_close", "fr_far"])
    Pr = Pr.merge(C, on=["set", "label", "k"], how="left")
    Pr = Pr.drop_duplicates(["set", "label", "act"])
    L = L.merge(Pr.drop(columns="k"), on=["set", "label", "act"], how="left")
    fr = ["fr_lick", "fr_right", "fr_left", "fr_close", "fr_far"]
    print(f"== check 3: licks with FaceRhythm {L[fr].notna().all(1).mean():.2f}", flush=True)
    L["sspn_resid"] = np.nan
    for (_st, _an), g in L.groupby(["set", "animal"]):
        ok = g[fr].notna().all(1) & g.ssp_n_diff.notna()
        if ok.sum() < 100:
            continue
        X = np.column_stack([np.ones(ok.sum()), g.loc[ok, fr].to_numpy(float)])
        b = np.linalg.lstsq(X, g.loc[ok, "ssp_n_diff"].to_numpy(float), rcond=None)[0]
        L.loc[g.index[ok], "sspn_resid"] = g.loc[ok, "ssp_n_diff"].to_numpy(float) - X @ b + b[0]
    S = L.groupby(["set", "animal", "label", "epoch"])[["ssp_n_diff", "sspn_resid"]].mean().reset_index()
    return _track(S, ["ssp_n_diff", "sspn_resid"], "3: face-movement partialled")


def trial_set_sentence() -> str:
    """THE lick set of the figure (Priya 11-09): the KEY per-lick cache licks behind the offset terms (terms_*_<TAG>,
    unfiltered) + the recovery axes."""
    return rd.key_sets(filt="", more=[rd.recovery_axes()])


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    out = []
    for k, f in (("3", check3), ("2", check2), ("1", check1)):
        R = f()
        R.to_csv(il.OUT_W / f"sspn_check{k}.csv", index=False)
        out.append(R)
    A = pd.concat(out, ignore_index=True)
    pd.set_option("display.width", 230)
    print(A.round(3).to_string(index=False))
    plot(A)
    return 0


def plot(A: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = A.copy()
    A["mean_r"] = A[[f"r_{a}" for a in AN]].mean(1)
    fig, axs = plt.subplots(3, len(TARGETS), figsize=(6.5 * len(TARGETS), 13), squeeze=False)
    for i, chk in enumerate(sorted(A.check.unique())):
        for j, (st, yv) in enumerate(TARGETS):
            ax = axs[i, j]
            t = A[(A.check == chk) & (A.set == st) & (A.y == yv)].sort_values("mean_r")
            x = np.arange(len(t))
            col = ["#C44E52" if ("SSp-n" in m or m in ("ssp_n_diff", "sspn_resid")) else "#4C72B0" for m in t.measure]
            ax.bar(x, t.mean_r, color=col)
            for xi, (_, r) in enumerate(t.iterrows()):
                ax.plot(np.full(4, xi), r[[f"r_{a}" for a in AN]].to_numpy(float), "k.", ms=4)
                if np.isfinite(r.p_within) and r.p_within < 0.05:
                    ax.text(xi, max(r.mean_r, 0) + 0.04, "*", ha="center")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylim(-1, 1)
            ax.set_xticks(x, t.measure, rotation=60, ha="right", fontsize=7)
            ax.set_title(f"check {chk}\n{'ITI' if st == 'iti' else 'cued'} -> {yv}: within-animal r with recovery "
                         "(bar = mean of animals, dots = animals, * = mixed p < 0.05)", fontsize=8.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("Is the SSp-n result special? (1) decoders refit WITHOUT an area: does their offset still track recovery; "
                 "(2) weight-free same-place left/right difference per area; (3) SSp-n term with face movement "
                 "(FaceRhythm) partialled out. Red = SSp-n. Negative r = larger value with worse (more ipsilesional) "
                 "aim. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(il.OUT_W / "figures" / "lick_sspn_checks.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
