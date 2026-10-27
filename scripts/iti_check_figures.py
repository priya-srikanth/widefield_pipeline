"""Figures for three checks that were table-only (Priya 10-27: "figure?"):
  rmse   the where-decoder's error split into |bias| and scatter (SD) per session (`iti_where_rmse_split.py`)
  f0     baseline fluorescence F0 (470, 415 nm) and the post-lick response in dF/F vs dF units (`iti_f0_check.py`)
  3axis  the where-decoder on lateral / forward / down axes: r / pre and offset beyond the pre pull (`iti_where_3axis.py`)
Every panel: thin lines = animals (session means per epoch), black = mean of animals, diamonds = mean of animals' pre +
change vs pre with the paired nested bootstrap 95% CI (animals > sessions, `analysis_kit.boot_delta`; * = CI excludes
0), and the count of animals changing in the point's direction. Reads the session CSVs; writes
<E>/iti_licks/figures/iti_check_{rmse,f0,3axis}.png. Usage: python scripts/iti_check_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]


def epoch_panel(ax, df: pd.DataFrame, val: str, ref: float | None = None, title: str = "", ylab: str = "") -> None:
    import matplotlib.pyplot as plt
    cols = plt.get_cmap("tab10")
    am = df.groupby(["animal", "epoch"])[val].mean()
    for k, a in enumerate(AN):
        ax.plot(range(4), [am.get((a, e), np.nan) for e in EPS], "o-", color=cols(k), ms=4, lw=1, alpha=0.7, label=a)
    mu = [am.xs(e, level="epoch").mean() if e in am.index.get_level_values("epoch") else np.nan for e in EPS]
    ax.plot(range(4), mu, "-", color="k", lw=2.2, zorder=5, label="mean of animals")
    prd = {a: h[val].dropna().to_numpy() for a, h in df[df.epoch == "pre"].groupby("animal")}
    for x, e in enumerate(EPS[1:], start=1):
        post = {a: h[val].dropna().to_numpy() for a, h in df[df.epoch == e].groupby("animal")}
        post = {a: v for a, v in post.items() if len(v) and a in prd}
        if not post:
            continue
        ci = ak.boot_delta(post, prd, np.random.default_rng(0))
        d = (am.xs(e, level="epoch") - am.xs("pre", level="epoch")).dropna()
        n_dir = int((d * np.sign(ci.point) > 0).sum())
        ax.errorbar(x + 0.13, mu[0] + ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="D", color="k",
                    ms=5, capsize=3, zorder=6)
        sig = ci.lo > 0 or ci.hi < 0
        ax.text(x + 0.13, mu[0] + ci.hi, f"{ci.point:+.2f}{' *' if sig else ''}\n{n_dir}/{len(d)}", ha="center",
                va="bottom", fontsize=7)
    if ref is not None:
        ax.axhline(ref, color="0.7", lw=0.7, ls=":")
    ax.set_xticks(range(4), EPS, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylab, fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def _save(fig, name: str, top: float) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(il.OUT / "figures" / f"iti_check_{name}.png", dpi=100)
    plt.close(fig)


def rmse() -> None:
    import matplotlib.pyplot as plt
    S = pd.read_csv(il.OUT / "where_rmse_split_sessions.csv")
    rows = [(st, ax_) for st in ("iti", "trial") for ax_ in ("lat", "sag")]
    fig, axs = plt.subplots(len(rows), 3, figsize=(13, 3.3 * len(rows)), squeeze=False)
    for i, (st, ax_) in enumerate(rows):
        g = S[(S.set == st) & (S.axis == ax_)]
        for j, (val, lab) in enumerate((("rmse", "RMSE"), ("bias", "|bias| = |mean error|"),
                                        ("sd", "scatter = SD of error"))):
            epoch_panel(axs[i, j], g, val, title=f"{'ITI' if st == 'iti' else 'cued'} "
                        f"{'LATERAL' if ax_ == 'lat' else 'SAGITTAL'}: {lab}", ylab="z-scored decoder, mm")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("NEURAL: why does the frozen where-decoder's error grow after the stroke? Per session, error = decoded - "
                 "actual; RMSE^2 = bias^2 + SD^2.\nDecoder on z-scored Allen-area features (lags 0..0.2 s; KEY-row "
                 "decoder). A growing |bias| with flat SD = predictions SHIFTED, not more scattered. PROVISIONAL",
                 fontsize=9.5)
    _save(fig, "rmse", 0.94)


def f0() -> None:
    import matplotlib.pyplot as plt
    S = pd.read_csv(il.OUT / "f0_check.csv")
    panels = (("f0_470_rel", "F0 470 nm (/ animal's pre mean)"), ("f0_415_rel", "F0 415 nm (/ pre)"),
              ("resp_auc_rel", "post-lick response, dF/F (/ pre)"), ("resp_dF_rel", "post-lick response, dF = "
                                                                     "dF/F x F0 (/ pre)"))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3), squeeze=False)
    for j, (val, lab) in enumerate(panels):
        epoch_panel(axs[0, j], S, val, ref=1.0, title=lab, ylab="/ pre" if j == 0 else "")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("NEURAL / MEASUREMENT: is the larger post-lick response an F0 artefact (dF/F inflated by a fall in "
                 "baseline fluorescence)? F0 = mean of frames_average.npy over the central 60% of the image;\nresponse "
                 "= the ITI-lick response area (as 3.24). If F0 fell, dF/F would rise with no change in dF: the dF "
                 "panel answers it. PROVISIONAL", fontsize=9.5)
    _save(fig, "f0", 0.86)


def three_axis() -> None:
    import matplotlib.pyplot as plt
    S = pd.read_csv(il.OUT / "where_3axis_sessions.csv")
    axes = (("lat", "LATERAL (+ = contralesional)"), ("fwd", "FORWARD (+ = further forward)"),
            ("down", "DOWN (+ = further down)"))
    fig, axs = plt.subplots(4, 3, figsize=(14, 13.5), squeeze=False)
    for i, (st, val, lab) in enumerate((("iti", "r_norm", "r / animal's pre"), ("iti", "beyond_d", "offset beyond pre "
                                        "pull, mm (change vs pre)"), ("trial", "r_norm", "r / animal's pre"),
                                        ("trial", "beyond_d", "offset beyond pre pull, mm (change vs pre)"))):
        for j, (ax_, an_) in enumerate(axes):
            g = S[(S.set == st) & (S.axis == ax_)]
            epoch_panel(axs[i, j], g, val, ref=1.0 if val == "r_norm" else 0.0,
                        title=f"{'ITI' if st == 'iti' else 'cued'} licks, {an_}", ylab=lab if j == 0 else "")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("NEURAL: the frozen where-decoder on all THREE axes (z-scored Allen-area features, lags 0..0.2 s; one "
                 "ridge per axis, trained on August pre licks, pre leave-one-session-out).\nSagittal extension ~ |down| "
                 "(r 0.96); forward is nearly independent of it (r -0.05). Offset = decoded - (a + b x actual), a, b fit "
                 "on pre licks. PROVISIONAL", fontsize=9.5)
    _save(fig, "3axis", 0.95)


def _days() -> pd.DataFrame:
    from recovery_timing import series
    return series("lick")[["label", "day"]].drop_duplicates("label")


def _aim() -> pd.DataFrame:
    """Per session: far-contra cued lick angle / animal's pre mean (the 3.37 behaviour)."""
    return pd.read_csv(il.OUT / "recovery_link_sessions_withpre_angle_z.csv")[["label", "beh"]].drop_duplicates("label")


def course(rows: list, title: str, name: str, ref_of: dict) -> None:
    """One column per animal, one row per (label, df, value): the measure per SESSION over days, pre shaded."""
    import matplotlib.pyplot as plt
    D = _days()
    fig, axs = plt.subplots(len(rows), 4, figsize=(17, 2.6 * len(rows)), squeeze=False, sharex="col")
    for i, (lab, df, val) in enumerate(rows):
        g0 = df.merge(D, on="label", how="left") if "day" not in df.columns else df
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = g0[g0.animal == a].groupby(["label", "day", "epoch"])[val].mean().reset_index().sort_values("day")
            ax.axvspan(g.day.min() - 1 if len(g) else -12, 0, color="0.93", zorder=0)
            ax.axvline(0, color="k", lw=0.7, ls="--")
            ax.plot(g.day, g[val], "o-", color="k", ms=3.5, lw=1)
            if ref_of.get(val) is not None:
                ax.axhline(ref_of[val], color="0.6", lw=0.6, ls=":")
            if len(g):
                pm = g[g.epoch == "pre"][val].mean()
                ax.axhline(pm, color="#2980b9", lw=0.8, ls="-", alpha=0.6)
            ax.set_title(f"{a}: {lab}", fontsize=8.5)
            if i == len(rows) - 1:
                ax.set_xlabel("day after stroke", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle(title + "\nOne dot per SESSION; grey band = pre-stroke sessions; blue line = the animal's pre mean.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if len(rows) > 3 else 0.9))
    fig.savefig(il.OUT / "figures" / f"iti_check_{name}_sessions.png", dpi=95)
    plt.close(fig)


def error_hist() -> None:
    """Per LICK: signed decoding error (decoded - actual, lateral) distributions, pre vs acute vs chronic, per animal.
    A SHIFTED distribution = bias; a WIDER one = scatter."""
    import matplotlib.pyplot as plt
    p = pd.read_csv(il.OUT / "zscore_check" / "where_preds_z.csv")
    p["err"] = p.dec_lat - p.act_lat
    col = {"pre": "0.35", "acute": "#c0392b", "subacute": "#e67e22", "chronic": "#2980b9"}
    fig, axs = plt.subplots(2, 4, figsize=(17, 6.4), squeeze=False, sharex=True)
    bins = np.linspace(-3, 3, 61)
    for i, st in enumerate(("iti", "trial")):
        for j, a in enumerate(AN):
            ax = axs[i, j]
            for ep in EPS:
                e = p[(p.set == st) & (p.animal == a) & (p.epoch == ep)].err.dropna().to_numpy()
                if len(e) < 20:
                    continue
                ax.hist(e, bins=bins, density=True, histtype="step", lw=1.4, color=col[ep],
                        label=f"{ep}: mean {e.mean():+.2f}, SD {e.std():.2f} (n {len(e)})")
                ax.axvline(e.mean(), color=col[ep], lw=0.8, ls="--")
            ax.axvline(0, color="k", lw=0.5)
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
            ax.set_title(f"{a}, {'ITI' if st == 'iti' else 'cued'} licks", fontsize=9)
            if i == 1:
                ax.set_xlabel("decoded - actual LATERAL position, mm (+ = read more contralesional)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("NEURAL: per-LICK decoding error of the frozen where-decoder (z-scored Allen areas; lateral). A shifted "
                 "distribution = BIAS (predictions displaced), a wider one = SCATTER.\nNB the error includes the decoder's "
                 "pull toward the pre mean, so a changed lick distribution moves the mean error too; the offset BEYOND "
                 "the pull (3.21 / 3.37) removes that. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(il.OUT / "figures" / "iti_check_error_per_lick.png", dpi=95)
    plt.close(fig)


def f0_severity() -> pd.DataFrame:
    """Does the post-lick response increase track the far-contra aim deficit? Within animal across POST sessions
    (log(day) removed from both, as 3.37), and per animal acute rise vs histology lesion burden (descriptive, n = 4)."""
    S = pd.read_csv(il.OUT / "f0_check.csv").merge(_days(), on="label").merge(_aim(), on="label")
    S = S[(S.epoch != "pre") & (S.day > 0)]
    rows = []
    for a, g in S.groupby("animal"):
        g = g.dropna(subset=["resp_auc_rel", "beh"])
        if len(g) < 5:
            continue
        d = np.log(g.day.to_numpy(float))
        res = {}
        for v in ("resp_auc_rel", "beh"):
            y = g[v].to_numpy(float)
            X = np.column_stack([np.ones(len(y)), d])
            res[v] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append(dict(animal=a, n=len(g), r_raw=float(np.corrcoef(g.resp_auc_rel, g.beh)[0, 1]),
                         r_detrended=float(np.corrcoef(res["resp_auc_rel"], res["beh"])[0, 1])))
    return pd.DataFrame(rows)


def main() -> int:
    rmse()
    f0()
    three_axis()
    error_hist()
    R = pd.read_csv(il.OUT / "where_rmse_split_sessions.csv")
    R = R[R.set == "iti"]
    course([(f"ITI {'lateral' if ax_ == 'lat' else 'sagittal'} {lab}", R[R.axis == ax_], v)
            for ax_ in ("lat", "sag") for v, lab in (("bias", "|bias|"), ("sd", "scatter (SD)"))],
           "NEURAL: frozen where-decoder error per session, ITI licks (z-scored decoder; mm).", "rmse", {})
    F = pd.read_csv(il.OUT / "f0_check.csv").merge(_aim(), on="label", how="left")
    course([("F0 470 / pre", F, "f0_470_rel"), ("post-lick response dF/F / pre", F, "resp_auc_rel"),
            ("post-lick response dF / pre", F, "resp_dF_rel"), ("far-contra lick angle / pre (behaviour)", F, "beh")],
           "NEURAL / MEASUREMENT: F0 and the ITI post-lick response per session, with far-contra aim for comparison.",
           "f0", {"f0_470_rel": 1, "resp_auc_rel": 1, "resp_dF_rel": 1, "beh": 1})
    T = pd.read_csv(il.OUT / "where_3axis_sessions.csv")
    T = T[T.set == "iti"]
    course([(f"ITI {ax_}: {lab}", T[T.axis == ax_], v) for ax_ in ("lat", "fwd", "down")
            for v, lab in (("r_norm", "r / pre"), ("beyond_d", "offset beyond pull (mm)"))],
           "NEURAL: frozen where-decoder on three axes per session, ITI licks (z-scored).", "3axis",
           {"r_norm": 1, "beyond_d": 0})
    sev = f0_severity()
    pd.set_option("display.width", 200)
    print("== does the ITI post-lick response (dF/F / pre) track far-contra aim within animal? (post sessions)")
    print(sev.round(2).to_string(index=False))
    print("wrote iti_check_{rmse,f0,3axis}[_sessions].png, iti_check_error_per_lick.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
