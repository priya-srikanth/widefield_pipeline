"""CLASS-MATCHED PRE-CUE TEST (Priya 10-25: "then do class-matched pre cue"): is the frozen decoder's far-contra code
weaker BEFORE the cue on trials that go on to be failed attempts than on hits?

Far-contra target trials, frozen pre-stroke spout decoder (`attempt_coding` trials, LOSO for pre), windows pre-cue (ENL),
cue and lick for comparison. Within each SESSION every failed attempt (incomplete) is paired with the NEAREST hit in trial
order (time-in-session matched: engagement, drift, satiety); the session value is the mean paired difference failed - hit
in raw P(true) (PRIMARY: retention units explode where the pre-stroke pre-cue code is near chance, e.g. PS92
0.24 - 1/6 = 0.07); retention kept in the session table. Acute hits are few (4-12 per session), so the same few hits
serve as the match for every failed attempt in a session: read the median session too. Sessions need >= MIN_EACH of both classes.
Test: failed vs matched hits on the same sessions, paired animals -> sessions bootstrap (`analysis_kit.boot_delta`,
animal-weighted), per epoch; per animal (rule 8) first. Pre-stroke trials are hits only, so there is no pre-stroke
failed-vs-hit comparison. Caveat: pre-cue activity is not movement-removed here (anticipatory movement could differ by class).

Usage: python scripts/precue_class_matched.py
Output: <E>/attempt_coding/precue_class_matched{,_sessions}.csv, figures/precue_class_matched.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
A = E / "attempt_coding"
AN = ["PS92", "PS93", "PS94", "PS95"]
POST = ["acute", "subacute", "chronic"]
WIN = [("precue", "pre-cue (ENL)"), ("cue", "cue window"), ("lick", "lick window")]
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
MIN_EACH = 3


def session_pairs(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (an, ep, lab), g in T.groupby(["animal", "epoch", "label"]):
        h, f = g[g.cls == "hit"].sort_values("trial_id"), g[g.cls == "incomplete"]
        if len(h) < MIN_EACH or len(f) < MIN_EACH:
            continue
        hid = h.trial_id.to_numpy()
        j = np.abs(f.trial_id.to_numpy()[:, None] - hid[None, :]).argmin(1)           # nearest hit in trial order
        m = h.iloc[j]
        rows.append({"animal": an, "epoch": ep, "label": lab, "n_failed": len(f), "n_hit": len(h),
                     "p_failed": f.p_true.mean(), "p_hit": m.p_true.mean(),
                     "ret_failed": f.ret.mean(), "ret_hit": m.ret.mean(),
                     "gap_trials": float(np.abs(f.trial_id.to_numpy() - m.trial_id.to_numpy()).mean())})
    return pd.DataFrame(rows)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import analysis_kit as ak
    res, ses = [], []
    for w, _wl in WIN:
        T = pd.concat([pd.read_csv(A / f"trials_{w}_{an}.csv.gz") for an in AN])
        T = T[(T.method == "dom") & (T.spout == "far_contra") & (T.label.str[-4:] >= "0801")].copy()
        ref = T[T.cls == "pre_hit"].groupby("animal").p_true.mean()
        T["ret"] = (T.p_true - 1 / 6) / (T.animal.map(ref) - 1 / 6)
        P = session_pairs(T[T.epoch.isin(POST)]).assign(window=w)
        ses.append(P)
        for ep in POST:
            q = P[P.epoch == ep]
            fa = {an: q[q.animal == an].p_failed.to_numpy() for an in AN if (q.animal == an).any()}
            hi = {an: q[q.animal == an].p_hit.to_numpy() for an in AN if (q.animal == an).any()}
            iv = ak.boot_delta(fa, hi, np.random.default_rng(0)) if len(fa) >= 2 else None
            per = {an: float(fa[an].mean() - hi[an].mean()) for an in fa}
            res.append({"window": w, "epoch": ep, "n_sessions": len(q), "animals": "/".join(sorted(fa)),
                        "delta_p": None if iv is None else iv.point, "lo": None if iv is None else iv.lo,
                        "hi": None if iv is None else iv.hi, "n_lower": sum(v < 0 for v in per.values()),
                        "median_session_d": float((q.p_failed - q.p_hit).median()) if len(q) else None, **per})
    S = pd.concat(ses, ignore_index=True)
    R = pd.DataFrame(res)
    S.to_csv(A / "precue_class_matched_sessions.csv", index=False)
    R.to_csv(A / "precue_class_matched.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): failed - matched hits, P(true), mean over sessions (sessions)")
    print(S.assign(d=S.p_failed - S.p_hit).groupby(["window", "epoch", "animal"]).d.agg(["mean", "size"])
          .unstack("animal").round(2).to_string())
    print("== pooled: failed - time-matched hits (raw P(true); paired animals > sessions bootstrap)")
    print(R.round(3).to_string(index=False))
    print("mean trial gap to the matched hit:", S.groupby("window").gap_trials.mean().round(1).to_dict())

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for k, (w, wl) in enumerate(WIN):
        ax = axs[k]
        ax.axhline(0, color="k", lw=0.7)
        for i, ep in enumerate(POST):
            q = S[(S.window == w) & (S.epoch == ep)]
            for j, an in enumerate(AN):
                v = q[q.animal == an]
                if len(v):
                    xs = i + (j - 1.5) * 0.08 + np.random.default_rng(10 * i + j).uniform(-0.02, 0.02, len(v))
                    ax.scatter(xs, v.p_failed - v.p_hit, marker=MK[an], s=22, facecolor="w", edgecolor="#c0392b", lw=1.1)
            r = R[(R.window == w) & (R.epoch == ep)].iloc[0]
            if r.delta_p is not None and np.isfinite(r.delta_p):
                sig = r.lo > 0 or r.hi < 0
                ax.plot([i - 0.25, i + 0.25], [r.delta_p] * 2, color="#c0392b", lw=2.4)
                ax.plot([i, i], [r.lo, r.hi], color="k", lw=1.2)
                ax.text(i, 0.58, f"{r.delta_p:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]\nmedian session {r.median_session_d:+.2f}\n"
                        f"{int(r.n_lower)}/{len(r.animals.split('/'))} animals lower",
                        ha="center", va="top", fontsize=7.5, fontweight="bold" if sig else "normal")
        ax.set_xticks(range(3)); ax.set_xticklabels(POST); ax.set_ylim(-0.75, 0.6)
        ax.set_title(wl, fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("failed attempts minus time-matched hits\n(decoder P(far contra))")
    fig.suptitle(f"NEURAL: is the far-contra code weaker on trials that FAIL? Within session, each failed attempt vs the nearest hit in "
                 f"trial order (sessions with >= {MIN_EACH} of each);\ndots = sessions; bar = paired animals > sessions bootstrap mean, "
                 "whisker 95% CI (bold = excludes 0). Pre-cue = before the cue (ENL), not movement-removed", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(A / "figures" / "precue_class_matched.png", dpi=125)
    plt.close(fig)
    print("->", A / "figures" / "precue_class_matched.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
