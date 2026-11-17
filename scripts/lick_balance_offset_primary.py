"""Does the PRIMARY balance (lick_recovery_primary: oro + SSs gain ratio 0.1..0.5 s after the first lick) track the
decoder's like-for-like OFFSET? -- the 12.18 question re-asked with the new balance (Priya 10-28: "replace the balance
figures with this new balance").

Session balance (cued first post-cue lick; ITI first lick after >= 2 s quiet; all licks and the inside-pre-range control)
vs the session like-for-like offset of the frozen KEY decoder (balreg_sessions.csv offset: all cued / all ITI licks).
Post-stroke sessions; per-animal r first (rule 8); MixedLM offset ~ within + between + log(day) + (1 | animal).
Outputs <OUT_W>/balprim_offset_models.csv; figure lick_balance_offset_primary.png. The balance axis is shown under
lick_recovery_primary.BAL_DISPLAY (default: implied direction = -balance, Priya 10-29; slopes and r drawn are vs the value
shown; the CSV keeps the hemispheric sign). --plot-only redraws from balprim_offset_models.csv (no fit, CSV untouched).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import plot_only  # noqa: E402
from lick_recovery_primary import BAL_DISPLAY, BAL_LABEL, bal_disp  # noqa: E402

PAIRS = [("cued first lick -> aim", "cued", "cued: first post-cue lick"),
         ("ITI first after quiet -> iti_ang", "ITI", "ITI: first lick after quiet")]
AN = ["PS92", "PS93", "PS94", "PS95"]


def trial_set_sentence() -> str:
    """The lick sets of lick_balance_offset_primary (Priya 11-09): the primary balance on the first post-cue lick and the
    ITI first lick after quiet (recprim_sessions; and its inside-pre-range version), against the offset of ALL KEY-set
    cued / ITI licks (balreg_sessions)."""
    import lick_balance_subsets as lbs
    import lick_covariate_reference as cv
    from wfield_local import trial_set_text as tst
    bal = lbs.set_sentence(("trial_1", "iti_q"), full=False, inside=False, subset=bool(il.ITI_SUBSET),
                           extra=["balance axis (second version: post licks inside the pre 2.5-97.5 % angle range)"])
    return tst.union(bal, cv.key_lick_set(iti_extra=["offset axis"], cued_extra=["offset axis"]))


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    PO = plot_only()
    R0 = pd.read_csv(il.OUT_W / "balprim_offset_models.csv") if PO else None
    B = pd.read_csv(il.OUT_W / "balreg_sessions.csv", usecols=["set", "label", "offset"])
    rows = []
    fig, axs = plt.subplots(1, len(PAIRS), figsize=(7 * len(PAIRS), 5.9), squeeze=False)
    for j, (panel, oset, lab) in enumerate(PAIRS):
        ax = axs[0, j]
        for sfx, ver in (("", "all licks"), ("_inside", "inside pre range")):
            if il.ITI_SUBSET and not (il.OUT_W / f"recprim_sessions{sfx}.csv").exists():
                continue                                # subset run without that sensitivity table (10-29)
            S = pd.read_csv(il.OUT_W / f"recprim_sessions{sfx}.csv")
            S = S[S.panel == panel][["label", "animal", "epoch", "day", "marker"]]
            G = S.merge(B[B.set == oset][["label", "offset"]], on="label")
            post = G[(G.epoch != "pre") & (G.day > 0)].dropna(subset=["marker", "offset"]).copy()
            if not PO:
                import statsmodels.formula.api as smf
                post["logday"] = np.log(post.day)
                post["w"] = post.marker - post.groupby("animal").marker.transform("mean")
                post["b"] = post.groupby("animal").marker.transform("mean")
                r = {a: np.corrcoef(h.marker, h.offset)[0, 1] for a, h in post.groupby("animal")}
                m = smf.mixedlm("offset ~ w + b + logday", post, groups=post["animal"]).fit(reml=False)
                rows.append({"panel": lab, "version": ver, "n": len(post), "slope": m.fe_params["w"],
                             "lo": m.conf_int().loc["w", 0], "hi": m.conf_int().loc["w", 1], "p": m.pvalues["w"],
                             **{f"r_{a}": v for a, v in r.items()}})
            if sfx == "":
                pre = G[G.epoch == "pre"]
                ax.plot(bal_disp(pre.marker), pre.offset, "o", color="0.82", mec="0.6", mew=0.4, ms=6, zorder=0)
                for (ep, a_), h in post.groupby(["epoch", "animal"]):
                    ax.plot(bal_disp(h.marker), h.offset, ANIMAL_MARKER.get(a_, "o"), color=EPOCH_LINE[ep], ms=6.5,
                            mec="k", mew=0.3)
        R = R0 if PO else pd.DataFrame(rows)
        ttl = lab
        for ver in ("all licks", "inside pre range"):
            if not ((R.panel == lab) & (R.version == ver)).any():
                continue                                # (subset run without the inside-range table)
            q = R[(R.panel == lab) & (R.version == ver)].iloc[0]
            s_ = f"{ver}: slope {bal_disp(q.slope):+.2f} mm per log2 unit, p {q.p:.2g}"
            ttl += ("\n" + (rf"$\mathbf{{{s_.replace(' ', '~')}}}$" if q.p < 0.05 else s_)
                    + "\n   per-animal r " + " / ".join(f"{bal_disp(q[f'r_{a}']):+.2f}" for a in AN))
        ax.set_title(ttl, fontsize=8.5)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        ax.set_xlabel(f"primary {BAL_LABEL.replace(chr(10), ' ')}\n(gain ratio, oro + SSs, 0.1-0.5 s after the first "
                      "lick; log2 vs pre)", fontsize=8)
        ax.set_ylabel(f"session like-for-like decoder offset, all {oset} licks (mm)", fontsize=8.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    if not PO:
        R = pd.DataFrame(rows)
        R.to_csv(il.OUT_W / "balprim_offset_models.csv", index=False)
        pd.set_option("display.width", 200)
        print(R.round(3).to_string(index=False))
    shown = ("\nBalance shown as " + BAL_LABEL.replace("\n", " ") + ";\nslopes and r are vs the value shown (+ slope = "
             "activity reading more contralesional goes with a more contralesional decoder offset)"
             if BAL_DISPLAY == "direction" else "")
    fig.suptitle("Does the PRIMARY balance track the decoder's like-for-like offset? One point per session (gray = pre, not "
                 "fitted);\nmixed model offset ~ balance (within animal) + log(day), random intercept per animal; also with "
                 "only licks inside the pre angle range (no extrapolation). PROVISIONAL" + shown, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.82 if shown else 0.88))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_balance_offset_primary.png", dpi=95)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
