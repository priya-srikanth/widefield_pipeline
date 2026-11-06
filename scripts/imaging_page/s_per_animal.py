"""Imaging page Fig 3.3 (`s_per_animal`): the pre-stroke position TEMPLATE and the frozen BEHAVIOURAL-STATE decoder,
per animal across epochs -- redrawn under Priya's 11-03 categorical-x rules (no lines across epochs; session values as
faded open markers, shape = animal, colour = epoch; summary = mean of animals +- SEM over animals, marker + interval).

Brought into the repo from M:/MICROSCOPE/Priya/Widefield/imaging_page_src/s_per_animal_regen_20261102/per_animal_fig.py
(and the session-c1a9581a `per_animal_tables.py` that built its tables). NOTHING IS REFITTED: every value is read from
the stored per-session sidecars in <labcams>/grant_figures/epoch/data/:

  template amplitude  epoch_11amp_encoder_amplitude_cue_working_sessions_long.csv, measure == "gain"
  variance captured   epoch_11c_encoder_ceiling_cue_working_sessions_long.csv, "frozen (matched)" / "ceiling"
  state decoder       epoch_13_state_decoder_cue_sessions.csv, (acc - 1/3) / (2/3)

Pre = leave-one-session-out pre sessions, as in the figures that wrote these sidecars. Before drawing, the per-animal
session means are CHECKED against the stored per-animal tables beside the source script (4 dp, as they were written),
so the redraw cannot silently show different numbers from the page's earlier version.

    python -m scripts.imaging_page.s_per_animal [--src DIR] [--out PNG]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/data")
SRC = Path("M:/MICROSCOPE/Priya/Widefield/imaging_page_src/s_per_animal_regen_20261102")
EP = ["pre", "acute", "subacute", "chronic"]
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
#: the epoch palette of the epoch figures this page's CD / window panels use (`scripts.cd_loo_figures.EP_COL`)
EP_COL = {"pre": "0.45", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}


def session_tables(data: Path = DATA) -> dict[str, pd.DataFrame]:
    """``{panel: DataFrame(animal, epoch, value)}``, one row per scored session."""
    amp = pd.read_csv(data / "epoch_11amp_encoder_amplitude_cue_working_sessions_long.csv")
    amp = amp[amp.measure == "gain"][["animal", "epoch", "value"]]
    cei = pd.read_csv(data / "epoch_11c_encoder_ceiling_cue_working_sessions_long.csv")
    w = cei.pivot_table(index=["label", "animal", "epoch"], columns="measure", values="value").reset_index()
    cap = w.assign(value=w["frozen (matched)"] / w["ceiling"])[["animal", "epoch", "value"]]
    st = pd.read_csv(data / "epoch_13_state_decoder_cue_sessions.csv")
    st = st.assign(value=(st.value - 1 / 3) / (1 - 1 / 3))[["animal", "epoch", "value"]]
    return {"template_amplitude_per_animal.csv": amp, "template_captured_per_animal.csv": cap,
            "state_decoder_abovechance_per_animal.csv": st}


def check_against_stored(tabs: dict[str, pd.DataFrame], src: Path = SRC) -> None:
    """Per-animal session means must equal the stored per-animal tables (written at 4 dp)."""
    for f, t in tabs.items():
        m = t.groupby(["animal", "epoch"]).value.mean().unstack()[EP].loc[list(ANIMALS)]
        ref = pd.read_csv(src / f, index_col=0).loc[list(ANIMALS), EP]
        d = float(np.nanmax(np.abs(m.round(4).to_numpy() - ref.to_numpy())))
        if d > 1e-4:
            raise SystemExit(f"{f}: session means differ from the stored per-animal table by {d:.4g}")
        print(f"   {f}: session means match the stored table (max |diff| {d:.1e})")


def figure(tabs: dict[str, pd.DataFrame], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import cat_points as cp

    panels = [("template_amplitude_per_animal.csv", "Pre-stroke template amplitude a", "amplitude (pre ~0.94)"),
              ("template_captured_per_animal.csv", "Variance the pre-stroke template captures",
               "frozen (matched) / ceiling"),
              ("state_decoder_abovechance_per_animal.csv", "Frozen state decoder (rest / run / lick)",
               "above chance, (acc - 1/3) / (2/3)")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8))
    for k, (ax, (f, title, yl)) in enumerate(zip(axs, panels)):
        t = tabs[f]
        M = t.groupby(["animal", "epoch"]).value.mean().unstack().reindex(columns=EP)
        for i, e in enumerate(EP):
            g = t[t.epoch == e]
            cp.session_dots(ax, i, g.value.to_numpy(), g.animal.to_numpy(), EP_COL[e], jitter=0.13,
                            seed=31 * k + i)
            v = M[e].dropna()
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            cp.summary_point(ax, i, v.mean(), se, color=EP_COL[e], ms=7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(EP)
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(yl)
        ax.spines[["top", "right"]].set_visible(False)
    h = [Line2D([], [], ls="none", marker="o", color=EP_COL[e], mec="w", ms=7, label=f"{e} (mean of animals +- SEM)")
         for e in EP] + cp.animal_handles(ANIMALS)
    fig.legend(handles=h, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.suptitle("Per animal, cue window, engaged trials (pre = leave-one-session-out). Faded open = sessions "
                 "(shape = animal);\nfilled = mean of the four animal means +- SEM. PS95 acute = 1 session.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.11, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    from wfield_local.writeguard import assert_writable

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--src", type=Path, default=SRC, help="folder holding the stored per-animal tables")
    ap.add_argument("--out", type=Path, default=SRC / "s_per_animal_nolines.png")
    a = ap.parse_args(argv)
    tabs = session_tables(a.data)
    for f, t in tabs.items():
        print(f, "n sessions:\n", t.groupby(["animal", "epoch"]).value.count().unstack()[EP].to_string())
    check_against_stored(tabs, a.src)
    assert_writable(a.out)
    print("->", figure(tabs, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
