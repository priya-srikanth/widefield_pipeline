from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
from pathlib import Path
from wfield_local import config
S = Path(__file__).parent; EP = ["pre", "acute", "subacute", "chronic"]; col = config.animal_color()
panels = [("template_amplitude_per_animal.csv", "Pre-stroke template amplitude a", "amplitude (pre ~0.94)", None),
          ("template_captured_per_animal.csv", "Variance the pre-stroke template captures", "frozen (matched) / ceiling", None),
          ("state_decoder_abovechance_per_animal.csv", "Frozen state decoder (rest / run / lick)", "above chance, (acc - 1/3) / (2/3)", None)]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, (f, title, yl, _) in zip(axs, panels):
    t = pd.read_csv(S / f, index_col=0)
    for an in ("PS92", "PS93", "PS94", "PS95"):
        ax.plot(range(4), t.loc[an, EP], "-o", color=col[an], label=an, lw=1.6, ms=5)
    ax.plot(range(4), t.loc["pooled (animal mean)", EP], "-s", color="k", lw=2.4, ms=6, label="mean of animals")
    ax.set_xticks(range(4)); ax.set_xticklabels(EP); ax.set_title(title, fontsize=10.5); ax.set_ylabel(yl)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("Per animal, session means (cue window, engaged trials; pre = leave-one-session-out). "
             "PS95 acute = 1 session.", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save_png_svg(fig, S / "per_animal_template_state.png", dpi=170)
print("ok")
