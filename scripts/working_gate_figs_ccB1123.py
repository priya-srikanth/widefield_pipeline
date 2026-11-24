"""Review figures for the working-period gate check (CC-B 11-23; not paper panels).

  gate_tally_classes.png  trials in one gate but not the other, by four-camera class, per epoch and animal
  s4c_far_contra_gates.png far-contralesional hit rate / own pre mean per session, engagement flag vs working period,
                          with the chronic onset each gate derives

    PYTHONPATH=. python scripts/working_gate_figs_ccB1123.py --data <working_gate/data> --out <working_gate>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CLS = ("contact", "incomplete", "jaw_only", "no_response", "unknown")
CLS_COL = {"contact": "#2a78d6", "incomplete": "#eb6834", "jaw_only": "#1baf7a", "no_response": "#eda100",
           "unknown": "#9a9a92"}
EPOCHS = ("pre", "acute", "subacute", "chronic")
MICE = ("#9", "#10", "#11", "#12")
INK, MUTED = "#222222", "#8a8a85"


def tally(data: Path, out: Path) -> None:
    T = pd.read_csv(data / "gate_tally_by_class.csv")
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharey="row")
    for r, (st, title) in enumerate((("wp_not_flag", "In the working period, not in the engagement flag"),
                                     ("flag_not_wp", "In the engagement flag, not in the working period"))):
        for c, ep in enumerate(EPOCHS):
            ax = axes[r, c]
            s = T[(T.set == st) & (T.epoch == ep)].pivot_table(index="mouse", columns="cls", values="n",
                                                                aggfunc="sum").reindex(MICE).fillna(0)
            bottom = pd.Series(0.0, index=MICE)
            for k in CLS:
                v = s[k] if k in s.columns else pd.Series(0.0, index=MICE)
                ax.bar(range(4), v, bottom=bottom, color=CLS_COL[k], width=0.7, edgecolor="white", linewidth=1,
                       label=k.replace("_", " ") if (r == 0 and c == 0) else None)
                bottom = bottom + v
            ax.set_xticks(range(4), MICE)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors=INK, labelsize=8)
            if r == 0:
                ax.set_title(ep, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel("trials", fontsize=9, color=INK)
            tot = int(bottom.sum()) if hasattr(bottom, "sum") else 0
            ax.text(0.98, 0.95, f"n = {tot}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
        axes[r, 0].annotate(title, xy=(0, 1.13 if r == 0 else 1.04), xycoords="axes fraction", fontsize=9,
                            color=INK, fontweight="bold")
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out / "gate_tally_classes.png", dpi=200)
    plt.close(fig)


def s4c(data: Path, out: Path) -> None:
    S = pd.read_csv(data / "epochs" / "s4c_series_working_period.csv")
    B = pd.read_csv(data / "epochs" / "s4c_boundaries_working_period.csv").set_index("animal")
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
    for ax, an in zip(axes, ("PS92", "PS93", "PS94", "PS95")):
        d = S[(S.animal == an) & (S.day > 0)].sort_values("day")
        ax.plot(d.day, d.rel_flag__far_R, color=MUTED, lw=2, marker="o", ms=4, label="engagement flag (production)")
        ax.plot(d.day, d.rel_wp__far_R, color="#2a78d6", lw=2, marker="o", ms=4, label="working period")
        ax.axhline(0.25, color=MUTED, lw=1, ls="--")
        for g, col in (("flag", MUTED), ("wp", "#2a78d6")):
            v = B.loc[an, f"{g}_chronic_from"]
            if pd.notna(v):
                ax.axvline(v, color=col, lw=1.2, ls=":")
        ax.set_title(f"{B.loc[an, 'mouse']}: chronic from day {B.loc[an, 'flag_chronic_from']:.0f} (flag), "
                     f"{B.loc[an, 'wp_chronic_from']:.0f} (working)", fontsize=8.5, color=INK)
        ax.set_xlabel("day after stroke", fontsize=9, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8)
    axes[0].set_ylabel("far contra hit rate / pre", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "s4c_far_contra_gates.png", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tally(Path(a.data), Path(a.out))
    s4c(Path(a.data), Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
