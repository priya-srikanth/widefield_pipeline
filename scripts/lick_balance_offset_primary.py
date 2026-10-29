"""Does the PRIMARY balance (lick_recovery_primary: oro + SSs gain ratio 0.1..0.5 s after the first lick) track the
decoder's like-for-like OFFSET? -- the 12.18 question re-asked with the new balance (Priya 10-28: "replace the balance
figures with this new balance").

Session balance (cued first post-cue lick; ITI first lick after >= 2 s quiet; all licks and the inside-pre-range control)
vs the session like-for-like offset of the frozen KEY decoder (balreg_sessions.csv offset: all cued / all ITI licks).
Post-stroke sessions; per-animal r first (rule 8); MixedLM offset ~ within + between + log(day) + (1 | animal).
Outputs <OUT_W>/balprim_offset_models.csv; figure lick_balance_offset_primary.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

PAIRS = [("cued first lick -> aim", "cued", "cued: first post-cue lick"),
         ("ITI first after quiet -> iti_ang", "ITI", "ITI: first lick after quiet")]
AN = ["PS92", "PS93", "PS94", "PS95"]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    B = pd.read_csv(il.OUT_W / "balreg_sessions.csv", usecols=["set", "label", "offset"])
    rows, an = [], []
    fig, axs = plt.subplots(1, len(PAIRS), figsize=(7 * len(PAIRS), 5.6), squeeze=False)
    for j, (panel, oset, lab) in enumerate(PAIRS):
        ax = axs[0, j]
        for sfx, ver in (("", "all licks"), ("_inside", "inside pre range")):
            S = pd.read_csv(il.OUT_W / f"recprim_sessions{sfx}.csv")
            S = S[S.panel == panel][["label", "animal", "epoch", "day", "marker"]]
            G = S.merge(B[B.set == oset][["label", "offset"]], on="label")
            post = G[(G.epoch != "pre") & (G.day > 0)].dropna(subset=["marker", "offset"]).copy()
            post["logday"] = np.log(post.day)
            post["w"] = post.marker - post.groupby("animal").marker.transform("mean")
            post["b"] = post.groupby("animal").marker.transform("mean")
            r = {a: np.corrcoef(h.marker, h.offset)[0, 1] for a, h in post.groupby("animal")}
            an.append({"panel": lab, "version": ver, **r})
            m = smf.mixedlm("offset ~ w + b + logday", post, groups=post["animal"]).fit(reml=False)
            rows.append({"panel": lab, "version": ver, "n": len(post), "slope": m.fe_params["w"],
                         "lo": m.conf_int().loc["w", 0], "hi": m.conf_int().loc["w", 1], "p": m.pvalues["w"],
                         **{f"r_{a}": v for a, v in r.items()}})
            if sfx == "":
                pre = G[G.epoch == "pre"]
                ax.plot(pre.marker, pre.offset, "o", color="0.82", mec="0.6", mew=0.4, ms=6, zorder=0)
                for (ep, a_), h in post.groupby(["epoch", "animal"]):
                    ax.plot(h.marker, h.offset, ANIMAL_MARKER.get(a_, "o"), color=EPOCH_LINE[ep], ms=6.5, mec="k", mew=0.3)
        R = pd.DataFrame(rows)
        ttl = lab
        for ver in ("all licks", "inside pre range"):
            q = R[(R.panel == lab) & (R.version == ver)].iloc[0]
            s_ = f"{ver}: slope {q.slope:+.2f} mm per log2 unit, p {q.p:.2g}"
            ttl += ("\n" + (rf"$\mathbf{{{s_.replace(' ', '~')}}}$" if q.p < 0.05 else s_)
                    + "\n   per-animal r " + " / ".join(f"{q[f'r_{a}']:+.2f}" for a in AN))
        ax.set_title(ttl, fontsize=8.5)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel("primary balance (gain ratio, oro + SSs, 0.1-0.5 s after the first lick; log2 vs pre)", fontsize=8.5)
        ax.set_ylabel(f"session like-for-like decoder offset, all {oset} licks (mm)", fontsize=8.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    R = pd.DataFrame(rows)
    R.to_csv(il.OUT_W / "balprim_offset_models.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))
    fig.suptitle("Does the PRIMARY balance track the decoder's like-for-like offset? One point per session (grey = pre, not "
                 "fitted); mixed model offset ~ balance (within animal) + log(day), random intercept per animal; also with "
                 "only licks inside the pre angle range (no extrapolation). PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT_W / "figures" / "lick_balance_offset_primary.png", dpi=95)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
