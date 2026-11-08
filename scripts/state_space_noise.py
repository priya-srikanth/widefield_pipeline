"""Which noise model holds for the trial-spread readout? (Priya 11-07: "run noise checks".)

spread_raw vs spread_norm (= spread / amplitude^2) disagree whenever amplitude changes: raw spread rises a little
post-stroke, amplitude-free spread falls. spread_norm assumes trial noise scales with the signal (an imaging gain:
F0, expression). Two measurements decide it, per session, within spout, first-lick trials (contact + incomplete), all 62
Allen areas, alignment from WF_SS_ALIGN (state_space_sim):

1. BASELINE NOISE: trial-to-trial variance around the spout mean in the baseline bins (state_space_sim.BAL_BINS: cue
   -0.5..0 s before the cue; lick -1.0..-0.6 s before the first lick), summed over areas. An imaging gain scales it by
   ~amplitude^2; a change in the response alone leaves it flat.
   Also RESPONSE NOISE (same, in the response bins) and RESPONSE SIZE^2 (norm^2 over areas of the spout mean, response
   minus baseline) for scale.
2. MEAN-VARIANCE: per animal x area, PRE cells (session x spout x time bin): variance = a + b * mean^2 (least squares,
   mean = the cell's trial mean minus the spout's baseline mean). Each session's cells are scored log2(observed /
   predicted at their own mean); median over cells and areas. Pre sessions are scored with a fit that leaves them out.
   ~0 post = trial noise is what the larger responses predict (noise scales with signal: spread_norm is the right
   model); > 0 = extra variability beyond the signal (raw spread is the right model).
All values per session as log2 vs the animal's own pre mean (except the mean-variance excess, already pre-referenced).
Output: Widefield/state_space_recovery/noise{tag}/noise_sessions.csv, page_figs/ss_noise_<align>.png. Exploratory.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import epochs  # noqa: E402

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
OUT = ROOT / f"noise{S.tag()}"


def cells(Y, pos, base, resp):
    """Per spout (>= MIN_TRIALS): baseline / response residual variance (summed over areas), response size^2, and the
    per-area cell table (bins x areas) of trial mean (minus the spout baseline) and trial variance."""
    out, cm, cv = {}, [], []
    for p in S.POS:
        m = pos == p
        if m.sum() < R.MIN_TRIALS:
            continue
        X = Y[m]                                                      # trials x bins x areas
        mu = X.mean(0)
        var = X.var(0, ddof=1)
        b0 = mu[base].mean(0)
        out[p] = {"base_noise": float(var[base].mean(0).sum()), "resp_noise": float(var[resp].mean(0).sum()),
                  "size2": float(((mu[resp].mean(0) - b0) ** 2).sum())}
        cm.append(mu - b0)
        cv.append(var)
    return out, (np.concatenate(cm) if cm else None), (np.concatenate(cv) if cv else None)


def fit_mv(M, V):
    """Per area least squares var = a + b * mean^2 (b >= 0 not enforced); M, V = cells x areas."""
    X2 = M ** 2
    coef = []
    for j in range(M.shape[1]):
        A = np.column_stack([np.ones(len(X2)), X2[:, j]])
        coef.append(np.linalg.lstsq(A, V[:, j], rcond=None)[0])
    return np.array(coef)                                            # areas x 2


def main():
    base, resp = S.BAL_BINS[S.ALIGN]
    rows, cellset = [], {}
    for f in sorted(glob.glob(S.ITI + "/PS9*.npz")):
        Y, pos, cols, ep, an = S.load(f)
        lab = Path(f).stem
        sp, M, V = cells(Y, pos, base, resp)
        if not sp:
            continue
        cellset[lab] = (an, ep, M, V)
        for p, v in sp.items():
            rows.append({"animal": an, "label": lab, "epoch": ep, "day": epochs.days_since_stroke(lab), "spout": p, **v})
    d = pd.DataFrame(rows)
    # session level: mean over spouts, then log2 vs the animal's pre mean
    s = d.groupby(["animal", "label", "epoch", "day"])[["base_noise", "resp_noise", "size2"]].mean().reset_index()
    for c in ("base_noise", "resp_noise", "size2"):
        pre = s[s.epoch == "pre"].groupby("animal")[c].mean()
        s[f"l2_{c}"] = np.log2(s[c] / s.animal.map(pre))
    # mean-variance excess
    exc = {}
    for an in s.animal.unique():
        labs = [lab for lab, v in cellset.items() if v[0] == an]
        pre = [lab for lab in labs if cellset[lab][1] == "pre"]
        for lab in labs:
            fit_on = [p for p in pre if p != lab]
            C = fit_mv(np.concatenate([cellset[p][2] for p in fit_on]), np.concatenate([cellset[p][3] for p in fit_on]))
            M, V = cellset[lab][2], cellset[lab][3]
            pred = C[:, 0] + C[:, 1] * M ** 2
            ok = (pred > 0) & (V > 0)
            r = np.where(ok, np.log2(np.where(ok, V, 1) / np.where(ok, pred, 1)), np.nan)
            exc[lab] = float(np.nanmedian(np.nanmedian(r, axis=0)))
    s["mv_excess"] = s.label.map(exc)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "noise_sessions.csv", index=False)
    d.to_csv(OUT / "noise_spouts.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(s.groupby(["animal", "epoch"])[["l2_base_noise", "l2_resp_noise", "l2_size2", "mv_excess"]].mean()
              .unstack().reindex(columns=R.EPOCHS, level=1))
    figure(s)


def figure(s):
    import state_space_page_figs as F
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    panels = [("l2_size2", "response size$^2$ (signal)\n(log2 vs own pre)"),
              ("l2_base_noise", "baseline trial noise\n(log2 vs own pre)"),
              ("l2_resp_noise", "response-window trial noise\n(log2 vs own pre)"),
              ("mv_excess", "trial noise beyond the pre\nmean-variance relation (log2 obs / pred)")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, (c, t) in zip(axs, panels):
        F.epoch_panel(ax, s, c, t, rng)
        ax.set_ylabel("")
    axs[0].set_ylabel("log2 vs own pre")
    F.legend_animals_epochs(fig, y=1.08)
    fig.suptitle(f"Does trial noise scale with the signal? ({F.ALIGN_LABEL[S.ALIGN]}; 62 areas, within spout)", y=1.15)
    fig.tight_layout()
    F.flush_marks(fig)
    F.sig_marks.reserve_headroom(fig)
    fig.savefig(F.ROOT / "page_figs" / f"ss_noise_{S.ALIGN}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
