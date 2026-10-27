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


def main() -> int:
    rmse()
    f0()
    three_axis()
    print("wrote iti_check_{rmse,f0,3axis}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
