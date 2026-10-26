"""Is the frozen pre-stroke lick-direction decoder a MARKER OF RECOVERY? (Priya 10-26: "perhaps this is a marker of
recovery?" about the in-trial recovery of page 3.9.)

Within each animal, across its POST-stroke sessions: does that session's frozen-decoder r (in-trial licks; ITI licks
as the comparison) track that day's behaviour = far-contra hit rate / the animal's pre-stroke rate
(`recovery_timing.series`)? Chronic behaviour has no between-animal spread, so this is a within-animal time-course test.
Both measures recover with time, so the correlation is also computed on residuals after regressing log(day) out of
each, per animal (does the decoder track day-to-day behaviour BEYOND the shared recovery trend?). SPREAD control: r
is a correlation, so a session with a wider spread of lick directions scores higher by itself, and more far-contra
hits widen the in-trial spread (PS92: r(behaviour, spread) 0.79) -> also partial out the session's SD of lick lateral
direction ("spread"), and spread + log(day) ("spread+day").
Per animal Pearson r; summary = mean of animals with a within-animal permutation test (behaviour shuffled across that
animal's post sessions, N_PERM draws, same shuffle for every statistic of a draw).
Inputs: <E>/iti_licks/contact_check.csv (`iti_contact_check.py`). Output: <E>/iti_licks/recovery_marker.csv,
figures/iti_recovery_marker.png.
Usage: python scripts/iti_recovery_marker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from recovery_timing import series  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
N_PERM = 5000


KINDS = ("raw", "detrended", "spread", "spread+day")


def _resid(y, *cov):
    X = np.column_stack([np.ones(len(y)), *cov])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def _stats(G: pd.DataFrame, beh: dict) -> dict:
    """Per animal r (raw, detrended) between behaviour and r_norm for each lick set; plus their means."""
    out = {}
    for st in ("trial", "iti"):
        for kind in KINDS:
            rs = []
            for a in AN:
                g = G[(G.animal == a) & (G.set == st)]
                if len(g) < 5:
                    rs.append(np.nan)
                    continue
                b, y, d = beh[a][g.index_in_animal.to_numpy()], g.r_norm.to_numpy(), np.log(g.day.to_numpy(float))
                cov = {"raw": (), "detrended": (d,), "spread": (g.side_sd.to_numpy(),),
                       "spread+day": (g.side_sd.to_numpy(), d)}[kind]
                if cov:
                    b, y = _resid(b, *cov), _resid(y, *cov)
                rs.append(float(np.corrcoef(b, y)[0, 1]))
            out[(st, kind)] = rs
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    R = pd.read_csv(il.OUT / "contact_check.csv")
    pre = R[R.epoch == "pre"].groupby(["set", "animal"]).r_all.mean()
    R["r_norm"] = R.r_all / [pre.get((s_, a_), np.nan) for s_, a_ in zip(R.set, R.animal)]
    sd = {}
    for x in R.label.unique():
        d_ = il._load(x)
        sd[(x, "trial")], sd[(x, "iti")] = np.nanstd(d_["tr_side"].astype(float)), np.nanstd(d_["iti_side"].astype(float))
    R["side_sd"] = [sd[(x, s_)] for x, s_ in zip(R.label, R.set)]
    B = series("lick")[["label", "day", "beh"]]
    G = R[R.epoch != "pre"].merge(B, on="label", how="inner").dropna(subset=["beh", "r_norm", "day"])
    G = G[G.day > 0].copy()
    # one behaviour vector per animal over its post sessions (shared by both lick sets, so one shuffle serves both)
    sess = {a: sorted(G[G.animal == a].label.unique()) for a in AN}
    beh = {a: G.drop_duplicates("label").set_index("label").beh.reindex(sess[a]).to_numpy(float) for a in AN}
    G["index_in_animal"] = [sess[a].index(x) for a, x in zip(G.animal, G.label)]
    print("== post sessions per animal x set:\n" + G.groupby(["animal", "set"]).size().unstack().to_string())
    obs = _stats(G, beh)
    rng = np.random.default_rng(0)
    null = {k: [] for k in obs}
    diff_null = {kind: [] for kind in KINDS}
    for _ in range(N_PERM):
        sh = {a: rng.permutation(beh[a]) for a in AN}
        s_ = _stats(G, sh)
        for k, v in s_.items():
            null[k].append(np.nanmean(v))
        for kind in diff_null:
            diff_null[kind].append(np.nanmean(s_[("trial", kind)]) - np.nanmean(s_[("iti", kind)]))
    rows = []
    for (st, kind), rs in obs.items():
        m = np.nanmean(rs)
        p = (np.sum(np.abs(null[(st, kind)]) >= abs(m)) + 1) / (N_PERM + 1)
        rows.append({"set": st, "kind": kind, **dict(zip(AN, np.round(rs, 3))), "mean_r": round(m, 3), "p_perm": p})
    for kind in KINDS:
        dm = np.nanmean(obs[("trial", kind)]) - np.nanmean(obs[("iti", kind)])
        p = (np.sum(np.abs(diff_null[kind]) >= abs(dm)) + 1) / (N_PERM + 1)
        rows.append({"set": "trial - iti", "kind": kind, "mean_r": round(dm, 3), "p_perm": p})
    T = pd.DataFrame(rows)
    T.to_csv(il.OUT / "recovery_marker.csv", index=False)
    pd.set_option("display.width", 200)
    print("== within-animal correlation of frozen-decoder r (/ pre) with far-contra hit rate (/ pre), post sessions\n"
          + T.to_string(index=False))
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey="row")
    for i, st in enumerate(("trial", "iti")):
        for j, a in enumerate(AN):
            ax = axs[i, j]
            g = G[(G.animal == a) & (G.set == st)]
            for ep in ("acute", "subacute", "chronic"):
                h = g[g.epoch == ep]
                ax.scatter(h.beh, h.r_norm, s=22, color=colors[ep], label=ep)
            rr = obs[(st, "raw")][j]
            rd = obs[(st, "detrended")][j]
            ax.set_title(f"{a} {'in-trial' if st == 'trial' else 'ITI'}: r {rr:+.2f} (detrended {rd:+.2f})", fontsize=9)
            ax.axhline(1, color="0.8", lw=0.6)
            ax.axvline(1, color="0.8", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"frozen decoder r / pre\n({'in-trial' if st == 'trial' else 'ITI'} licks)", fontsize=9)
            if i == 1:
                ax.set_xlabel("far-contra hit rate / pre", fontsize=9)
    axs[0, 0].legend(fontsize=7, frameon=False)
    t_ = T.set_index(["set", "kind"])
    fig.suptitle("NEURAL vs BEHAVIOUR: does the frozen pre-stroke lick-direction decoder track recovery? One dot per post-stroke "
                 "session. Mean within-animal r: in-trial "
                 f"{t_.loc[('trial', 'raw'), 'mean_r']:+.2f} (p {t_.loc[('trial', 'raw'), 'p_perm']:.3f}), ITI "
                 f"{t_.loc[('iti', 'raw'), 'mean_r']:+.2f} (p {t_.loc[('iti', 'raw'), 'p_perm']:.3f}); after removing the shared "
                 f"log(day) trend: in-trial {t_.loc[('trial', 'detrended'), 'mean_r']:+.2f} (p "
                 f"{t_.loc[('trial', 'detrended'), 'p_perm']:.3f}), ITI {t_.loc[('iti', 'detrended'), 'mean_r']:+.2f} (p "
                 f"{t_.loc[('iti', 'detrended'), 'p_perm']:.3f}); also removing the session's spread of lick directions: in-trial "
                 f"{t_.loc[('trial', 'spread+day'), 'mean_r']:+.2f} (p {t_.loc[('trial', 'spread+day'), 'p_perm']:.3f}). "
                 "p = within-animal permutation of behaviour.",
                 fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT / "figures" / "iti_recovery_marker.png", dpi=110)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
