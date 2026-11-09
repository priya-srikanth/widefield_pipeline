"""Imaging results page Fig 3.3 (`s_per_animal.png`): per animal, pre-stroke template amplitude, the variance the
pre-stroke template captures, and the frozen behavioral-state decoder (cue window), session means per animal x epoch.

PROVENANCE: recreated 2026-11-03 (cohort-1 rebuild, step 3) from three session scripts, none in the repo:
  * per_animal_tables.py  (session c1a9581a scratchpad imaging/, 10-25): template tables from the epoch sidecars.
  * state_table_zscored.py (imaging_page_src/s_per_animal_regen_20261102/, 11-02): the state table from the
    re-rendered (per-session z-scored, commit abda8c9c) epoch_13 sidecar; same computation as per_animal_tables.py.
  * per_animal_fig.py (imaging_page_src/s_per_animal_regen_20261102/ = wf_attempt scripts/recovered_20261103/): the
    plot; it read the three CSVs from its own folder and wrote per_animal_template_state.png, which the page copy
    step (docs/results_pages/imaging/page_figs.py) turned into s_per_animal.png.
  Here the three steps are one script: tables -> CSVs -> figure. Computation, layout, dpi unchanged. Changes:
  argparse (--out / --plot-only / --data-dir), output PNG named per_animal_template_state.png as before (the page name
  s_per_animal.png is the copy step's), figdata pickle (<out>/data/s_per_animal_figdata.pkl). The tables step's
  extra outputs that the figure does not use (template_ceiling / template_frozen / state_decoder_balacc tables,
  template_sessions.csv) are also written, as per_animal_tables.py did.

INPUTS (per-session figdata sidecars, read as is), in --data-dir (default = the originals' hard-coded
M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/data):
  epoch_11amp_encoder_amplitude_cue_working_sessions_long.csv   measure "gain" = template amplitude a (Fig 6.1)
  epoch_11c_encoder_ceiling_cue_working_sessions_long.csv       "frozen (matched)" / "ceiling" = captured (Fig 6.2)
  epoch_13_state_decoder_cue_sessions.csv                       frozen state decoder balanced accuracy (Fig 3.1);
                                                                above chance = (acc - 1/3) / (2/3)
  Session means per animal x epoch; pre = leave-one-session-out pre sessions (as the figures); the "pooled (animal
  mean)" row = unweighted mean of the four animals.
REBUILD: re-render epoch families 11amp, 11c, 13 and point --data-dir at their data/.
RENDERING ENVIRONMENT: the page image (11-02 21:47) was drawn with matplotlib 3.11.2 (PNG metadata; the `fr` /
  `roicat` envs on this box), not the `locanmf` env's 3.10.9. The tables are identical either way; the PNG is
  pixel-identical to the page only under matplotlib 3.11.2 (under 3.10.9 ~7 % of pixels differ: line antialiasing /
  text placement, same values).
NOTE (as found, not changed): the suptitle says "engaged trials"; the sidecars are the WORKING-trial class
(`_cue_working_`), which the page caption states correctly.

OUT: <out>/per_animal_template_state.png + template_amplitude_per_animal.csv, template_captured_per_animal.csv,
     state_decoder_abovechance_per_animal.csv (the three tables the figure draws).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

NAME = "s_per_animal"
PNG = "per_animal_template_state.png"
D_DEF = _common.epoch_data_dir()   # R3: this run's epoch data (rebuild root under the profile)
EP = ["pre", "acute", "subacute", "chronic"]
PANELS = [("template_amplitude_per_animal.csv", "Pre-stroke template amplitude a", "amplitude (pre ~0.94)"),
          ("template_captured_per_animal.csv", "Variance the pre-stroke template captures",
           "frozen (matched) / ceiling"),
          ("state_decoder_abovechance_per_animal.csv", "Frozen state decoder (rest / run / lick)",
           "above chance, (acc - 1/3) / (2/3)")]


def tables(D: Path, out: Path) -> dict:
    """per_animal_tables.py (template part) + state_table_zscored.py; CSVs rounded to 4 dp as the originals wrote, and
    the figure reads the ROUNDED tables back (as per_animal_fig.py did), so the figdata holds exactly what was drawn."""
    amp = pd.read_csv(D / "epoch_11amp_encoder_amplitude_cue_working_sessions_long.csv")
    cei = pd.read_csv(D / "epoch_11c_encoder_ceiling_cue_working_sessions_long.csv")
    print("October days in long tables:", sorted(amp[amp.label.str.endswith(("_1002", "_1006"))].day.unique()))
    w = cei.pivot_table(index=["label", "animal", "epoch", "day"], columns="measure", values="value").reset_index()
    w["captured"] = w["frozen (matched)"] / w["ceiling"]
    a = amp[amp.measure == "gain"][["label", "animal", "epoch", "value"]].rename(columns={"value": "amplitude"})
    T = w.merge(a, on=["label", "animal", "epoch"], how="outer")

    def table(col):
        t = T.groupby(["animal", "epoch"])[col].mean().unstack()[EP]
        n = T.groupby(["animal", "epoch"])[col].count().unstack()[EP]
        t.loc["pooled (animal mean)"] = t.mean()
        return t, n
    for col in ("amplitude", "ceiling", "frozen (matched)", "captured"):
        t, n = table(col)
        print(f"\n== template {col} (cue, engaged), session means\n", t.round(3).to_string(), "\n n sessions:\n",
              n.to_string())
        t.round(4).to_csv(out / f"template_{col.split()[0]}_per_animal.csv")
    T.to_csv(out / "template_sessions.csv", index=False)
    st = pd.read_csv(D / "epoch_13_state_decoder_cue_sessions.csv")
    s = st.groupby(["animal", "epoch"]).value.mean().unstack()[EP]
    print("n sessions\n", st.groupby(["animal", "epoch"]).value.count().unstack()[EP].to_string())
    norm = (s - 1 / 3) / (1 - 1 / 3)
    s.loc["pooled (animal mean)"] = s.mean()
    norm.loc["pooled (animal mean)"] = norm.mean()
    print(norm.round(3).to_string())
    s.round(4).to_csv(out / "state_decoder_balacc_per_animal.csv")
    norm.round(4).to_csv(out / "state_decoder_abovechance_per_animal.csv")
    return {f: pd.read_csv(out / f, index_col=0) for f, _t, _y in PANELS}


def draw(tabs: dict, png: Path) -> None:
    from wfield_local import config
    col = config.animal_color()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (f, title, yl) in zip(axs, PANELS):
        t = tabs[f]
        for an in ("PS92", "PS93", "PS94", "PS95"):
            ax.plot(range(4), t.loc[an, EP], "-o", color=col[an], label=an, lw=1.6, ms=5)
        ax.plot(range(4), t.loc["pooled (animal mean)", EP], "-s", color="k", lw=2.4, ms=6, label="mean of animals")
        ax.set_xticks(range(4))
        ax.set_xticklabels(EP)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(yl)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Per animal, session means (cue window, engaged trials; pre = leave-one-session-out). "
                 "PS95 acute = 1 session.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png, dpi=170)
    plt.close(fig)


def trial_set_sentence() -> str:
    """Template amplitude / captured (camera working post, lick-trial pre) and the state decoder's 1 s segments."""
    from wfield_local import trial_set_text as tst
    post_inc, post_exc = _common._split(tst.sentence("working"))
    pre_inc, pre_exc = _common._split(tst.sentence("lick"))
    tmpl = tst.events(f"template panels, cue window: POST sessions = {post_inc} (>= 10 trials per position; not: "
                      f"{post_exc}); PRE sessions = {pre_inc} (not: {pre_exc}), each left out against the other pre "
                      "sessions (the 11amp / 11c pre side is the lick class, not camera working)", unit="trials")
    return tst.union(tmpl, _common.state_segments(), unit="trials / frames")


def main(argv=None) -> int:
    ap = _common.parser(__doc__)
    ap.add_argument("--data-dir", type=Path, default=D_DEF, help="grant_figures/epoch/data folder with the sidecars")
    a = ap.parse_args(argv)
    _common.declare(trial_set_sentence())          # THE TRIAL SET of every figure (Priya 11-09)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        tabs = _common.load_figdata(a.out, NAME)["tables"]
    else:
        tabs = tables(a.data_dir, a.out)
        _common.save_figdata(a.out, NAME, {"tables": tabs, "src": str(a.data_dir)})
    draw(tabs, a.out / PNG)
    print("->", a.out / PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
