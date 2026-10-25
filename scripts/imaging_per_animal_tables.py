"""Per-animal tables for two imaging readouts the results page reported pooled only (Priya 10-25, imaging page
"pending"): the pre-stroke position TEMPLATE (epoch family 11 / 11amp / 11c, cue window, engaged trials) and the frozen
BEHAVIOURAL-STATE decoder (family 13). Read from the existing per-session sidecars -- nothing is refitted; the frozen
models are references (CLAUDE.md rule 10).

Session means per animal x epoch; pre = the leave-one-session-out pre sessions the figures use (template: the 8
August pre sessions per animal, `_sessions_long.csv`; state decoder: 11 pre sessions per animal incl. June,
`epoch_13_state_decoder_cue_sessions.csv`). "captured" = frozen (matched) / ceiling per session (reproduces the
page's pooled 61 % pre / 18 % acute). State "above chance" = (balanced accuracy - 1/3) / (2/3), the 13n scale.

Findings when written (10-25): the cohort-level state-decoder drop is carried by PS94 (chronic -.53 above chance;
the other three -.01 to -.09); template amplitude falls acutely 4 / 4 but is back chronically only in PS92 / PS93
(PS94 .68, PS95 .73 of ~.93 pre); template captured fraction stays below pre chronically 4 / 4.

    python scripts/imaging_per_animal_tables.py [--out DIR]
Output: <labcams>/grant_figures/epoch/per_animal_tables/ (CSVs + per_animal_template_state.png).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EP = ["pre", "acute", "subacute", "chronic"]
ANIMALS = ("PS92", "PS93", "PS94", "PS95")


def tables(data: Path) -> dict[str, pd.DataFrame]:
    amp = pd.read_csv(data / "epoch_11amp_encoder_amplitude_cue_working_sessions_long.csv")
    cei = pd.read_csv(data / "epoch_11c_encoder_ceiling_cue_working_sessions_long.csv")
    w = cei.pivot_table(index=["label", "animal", "epoch"], columns="measure", values="value").reset_index()
    w["captured"] = w["frozen (matched)"] / w["ceiling"]
    a = amp[amp.measure == "gain"][["label", "animal", "epoch", "value"]].rename(columns={"value": "amplitude"})
    T = w.merge(a, on=["label", "animal", "epoch"], how="outer")
    st = pd.read_csv(data / "epoch_13_state_decoder_cue_sessions.csv")
    out = {}
    for name, frame, col in (("template_amplitude", T, "amplitude"), ("template_ceiling", T, "ceiling"),
                             ("template_frozen_matched", T, "frozen (matched)"), ("template_captured", T, "captured"),
                             ("state_balacc", st, "value")):
        t = frame.groupby(["animal", "epoch"])[col].mean().unstack()[EP]
        out[name] = t
        out[name + "_n"] = frame.groupby(["animal", "epoch"])[col].count().unstack()[EP]
    out["state_abovechance"] = (out["state_balacc"] - 1 / 3) / (2 / 3)
    for k in [k for k in out if not k.endswith("_n")]:
        out[k].loc["mean of animals"] = out[k].loc[list(ANIMALS)].mean()
    return out


def figure(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config
    col = config.animal_color()
    panels = [("template_amplitude", "Pre-stroke template amplitude a", "amplitude"),
              ("template_captured", "Variance the pre-stroke template captures", "frozen (matched) / ceiling"),
              ("state_abovechance", "Frozen state decoder (rest / run / lick)", "above chance, (acc - 1/3) / (2/3)")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (k, title, yl) in zip(axs, panels):
        t = out[k]
        for an in ANIMALS:
            ax.plot(range(4), t.loc[an, EP], "-o", color=col[an], label=an, lw=1.6, ms=5)
        ax.plot(range(4), t.loc["mean of animals", EP], "-s", color="k", lw=2.4, ms=6, label="mean of animals")
        ax.set_xticks(range(4))
        ax.set_xticklabels(EP)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(yl)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Per animal, session means (cue window, engaged trials; pre = leave-one-session-out). "
                 "PS95 acute = 1 session.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    epoch = Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"
    out_dir = a.out or epoch / "per_animal_tables"
    assert_writable(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = tables(epoch / "data")
    for k, t in out.items():
        t.round(4).to_csv(out_dir / f"{k}.csv")
        if not k.endswith("_n"):
            print(f"\n== {k}\n{t.round(3).to_string()}")
    figure(out, out_dir / "per_animal_template_state.png")
    print(f"\n-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
