"""Slide 21, normalized-to-own-pre version (Priya 11-17): each session's above-chance retained divided by its animal's
own pre mean (pre ~ 1, chance = 0); stars = boot_delta paired animals -> sessions on that ratio, Holm over the 6 tests.
Same data (rebuild state_position_sessions.csv) and same drawing as scripts/rebuild_figs/fig_state_position_sessions.py."""
import inspect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "C:/Users/SabatiniLab/Github/wf_cleanup2/scripts/rebuild_figs")
import fig_state_position_sessions as F  # noqa: E402

CSV = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/rebuild_figs/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_"
           "pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/state_position_sessions.csv")
OUT = Path(sys.argv[1])

D = pd.read_csv(CSV)
pre = D[D.epoch == "pre"].groupby(["readout", "animal"]).retained.mean().rename("pre_mean")
D = D.join(pre, on=["readout", "animal"])
D["retained"] = D.retained / D.pre_mean
D["epoch"] = pd.Categorical(D.epoch, F.EPOCHS, ordered=True)
print("change vs own pre (ratio - 1), boot_delta animals -> sessions:")
T = F.stats(D)

src = inspect.getsource(F.draw)
src = src.replace('ax.axhline(0, color="0.45", ls="--", lw=1.1, zorder=1)',
                  'ax.axhline(0, color="0.45", ls="--", lw=1.1, zorder=1); '
                  'ax.axhline(1, color="0.6", ls=":", lw=1.1, zorder=1)')
src = src.replace("axs[0].set_ylim(lo, 1.05)", "hi = max(1.1, float(D.retained.max()) + 0.05); axs[0].set_ylim(lo, hi)")
src = src.replace("np.arange(np.ceil(lo * 4) / 4, 1.01, 0.25)", "np.arange(np.ceil(lo * 4) / 4, hi, 0.25)")
src = src.replace('"Above-chance performance retained"', '"Performance vs own pre (pre = 1)"')
src = src.replace("def draw(", "def draw_rel(")
ns = dict(vars(F))
exec(src, ns)
OUT.mkdir(parents=True, exist_ok=True)
ns["draw_rel"](D, T, OUT / "state_position_sessions_rel_own_pre.png")
D.to_csv(OUT / "state_position_sessions_rel_own_pre.csv", index=False, float_format="%.4f")
print("->", OUT / "state_position_sessions_rel_own_pre.png")
