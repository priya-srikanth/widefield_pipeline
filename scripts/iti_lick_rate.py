"""Does INTER-TRIAL licking track recovery, and does it exceed pre-stroke levels? (Priya 10-26.)

BEHAVIOUR only. Per session: ITI licks = licks_all.csv context "iti" (every cam4-detected lick between trials; the
spout is retracted, except at the window edges); ITI time = sum over consecutive trials of next arrival (strobe) minus
this trial's stop, clipped at 0 and at 60 s (long pauses excluded); rate = ITI licks / ITI minutes. Also the share of
ITI licks that touch the spout (DAQ contact flag) and the no-contact rate. Per-trial counts (ITI licks per trial) as a
second definition. Normalised to the animal's pre mean.
Tests: change from pre per epoch (nested bootstrap, per-animal table first); within animal across POST sessions, the
correlation with far-contra hit rate / pre (`recovery_timing.series`), raw and with log(day) removed (permutation of
behaviour within animal, as `iti_recovery_marker.py`).
Outputs <E>/iti_licks/iti_lick_rate.csv, figures/iti_lick_rate.png.
Usage: python scripts/iti_lick_rate.py [--plot-only]   (--plot-only redraws from the saved CSVs)
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
EP = ["pre", "acute", "subacute", "chronic"]
GAP_MAX = 60.0
N_PERM = 5000


def session_row(label: str) -> dict:
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv", usecols=["context", "contact"])
    T = il._behaviour(label).trials.sort_values("cue_s")
    st = np.where(np.isfinite(T.strobe_s), T.strobe_s, T.cue_s - 3.0)
    sp = T.stop_s.to_numpy(float)
    gaps = st[1:] - sp[:-1]
    gaps = gaps[np.isfinite(gaps)]
    iti_min = float(np.clip(gaps, 0, GAP_MAX).sum() / 60.0)
    a = A[A.context == "iti"]
    c = a.contact.astype(str).str.lower().eq("true")
    return {"label": label, "animal": an, "epoch": epochs.epoch_of(label), "n_trials": len(T),
            "n_iti": len(a), "share_contact": float(c.mean()) if len(a) else np.nan, "iti_min": iti_min,
            "rate": len(a) / iti_min if iti_min > 0 else np.nan,
            "rate_nocontact": int((~c).sum()) / iti_min if iti_min > 0 else np.nan,
            "per_trial": len(a) / max(len(T), 1)}


def _resid(y, d):
    X = np.column_stack([np.ones(len(d)), d])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def main() -> int:
    from wfield_local import analysis_kit as ak
    labels = sorted(p.stem for p in (il.OUT / "sessions").glob("*.npz"))
    R = pd.DataFrame([session_row(x) for x in labels])
    VALS = ("rate", "rate_nocontact", "per_trial", "share_contact")
    for v in VALS:
        pre = R[R.epoch == "pre"].groupby("animal")[v].mean()
        R[f"{v}_n"] = R[v] / R.animal.map(pre)
    R = R.merge(series("lick")[["label", "day", "beh"]], on="label", how="left")
    R.to_csv(il.OUT / "iti_lick_rate.csv", index=False)
    pd.set_option("display.width", 200)
    print("== ITI lick rate (licks / ITI minute), per animal mean (rule 8)")
    print(R.groupby(["animal", "epoch"]).rate.mean().unstack().reindex(columns=EP).round(1).to_string())
    print("== share of ITI licks touching the spout")
    print(R.groupby(["animal", "epoch"]).share_contact.mean().unstack().reindex(columns=EP).round(2).to_string())
    rows = []
    for v in VALS:
        ref = {a: R[(R.animal == a) & (R.epoch == "pre")][f"{v}_n"].dropna().to_numpy() for a in AN}
        for ep in EP[1:]:
            post = {a: R[(R.animal == a) & (R.epoch == ep)][f"{v}_n"].dropna().to_numpy() for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
            per = {a: round(float(post[a].mean() - ref[a].mean()), 2) for a in com}
            rows.append({"value": v, "epoch": ep, "delta_vs_pre": iv.point, "lo": iv.lo, "hi": iv.hi, **per})
    T = pd.DataFrame(rows)
    print("\n== change from pre (normalised to the animal's pre = 1)\n" + T.round(3).to_string(index=False))
    G = R[(R.epoch != "pre") & R.beh.notna() & (R.day > 0)].copy()
    rng = np.random.default_rng(0)
    out = []
    for v in ("rate_n", "rate_nocontact_n", "per_trial_n"):
        def stat(sh, v=v):
            rr = {}
            for kind in ("raw", "detrended"):
                rs = []
                for a in AN:
                    g = G[G.animal == a]
                    b, y, d = sh[a], g[v].to_numpy(), np.log(g.day.to_numpy(float))
                    if kind == "detrended":
                        b, y = _resid(b, d), _resid(y, d)
                    rs.append(float(np.corrcoef(b, y)[0, 1]))
                rr[kind] = rs
            return rr
        beh = {a: G[G.animal == a].beh.to_numpy() for a in AN}
        obs = stat(beh)
        null = {k: [] for k in obs}
        for _ in range(N_PERM):
            s_ = stat({a: rng.permutation(beh[a]) for a in AN})
            for k in null:
                null[k].append(np.nanmean(s_[k]))
        for k, rs in obs.items():
            m = np.nanmean(rs)
            out.append({"value": v, "kind": k, **dict(zip(AN, np.round(rs, 2))), "mean_r": round(m, 3),
                        "p_perm": (np.sum(np.abs(null[k]) >= abs(m)) + 1) / (N_PERM + 1)})
    C = pd.DataFrame(out)
    print("\n== within-animal correlation with far-contra hit rate / pre (post sessions)\n" + C.to_string(index=False))
    T.to_csv(il.OUT / "iti_lick_rate_tests.csv", index=False)
    C.to_csv(il.OUT / "iti_lick_rate_recovery.csv", index=False)
    plot(R, T, C)
    return 0


def plot(R: pd.DataFrame, T: pd.DataFrame, C: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    rng2 = np.random.default_rng(1)
    for j, a in enumerate(AN):
        ax = axs[0, j]
        g = R[R.animal == a].sort_values("day")
        ax.plot(g.day, g.rate_n, color="0.7", lw=0.8, zorder=0)
        for ep in EP:
            h = g[g.epoch == ep]
            ax.scatter(h.day, h.rate_n, s=22, color=colors[ep], label=ep, zorder=2)
        ax2 = ax.twinx()
        ax2.plot(g.day, g.beh, color="0.3", lw=1, ls=":")
        ax2.set_ylim(-0.1, 1.5)
        ax2.tick_params(labelsize=7)
        if j == 3:
            ax2.set_ylabel("far-contra hit rate / pre (dotted)", fontsize=8)
        ax.axhline(1, color="0.6", lw=0.6, ls="--")
        ax.axvline(0, color="0.6", lw=0.6)
        c_ = C[(C.value == "rate_n") & (C.kind == "raw")].iloc[0][a]
        ax.set_title(f"{a}: r with behaviour (post) {c_:+.2f}", fontsize=9)
        ax.set_xlabel("day from stroke", fontsize=8)
        if j == 0:
            ax.set_ylabel("ITI lick rate / pre", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
    for j, v in enumerate(("rate", "rate_nocontact", "per_trial", "share_contact")):
        ax = axs[1, j]
        for e, ep in enumerate(EP):
            vv = R[R.epoch == ep][f"{v}_n"].dropna().to_numpy()
            ax.scatter(e + rng2.uniform(-0.18, 0.18, len(vv)), vv, s=10, color=colors[ep], alpha=0.7, lw=0)
            if len(vv):
                ax.errorbar(e + 0.3, vv.mean(), vv.std(ddof=1) / np.sqrt(len(vv)), fmt="o", color="k", ms=4, capsize=2)
            tt = T[(T.value == v) & (T.epoch == ep)]
            if len(tt) and (tt.lo.iloc[0] > 0 or tt.hi.iloc[0] < 0):
                ax.text(e, 1.02, "*", transform=ax.get_xaxis_transform(), ha="center", fontsize=12)
        ax.axhline(1, color="0.6", lw=0.6, ls="--")
        ax.set_xticks(range(4), EP, fontsize=8)
        ax.set_title({"rate": "ITI licks per ITI minute", "rate_nocontact": "... without spout contact",
                      "per_trial": "ITI licks per trial", "share_contact": "share of ITI licks touching the spout"}[v]
                     + " / pre", fontsize=9, pad=14)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle("BEHAVIOUR: inter-trial (self-initiated) licking across recovery. Top: per session vs day from stroke, divided "
                 "by the animal's pre mean\n(dotted = far-contra hit rate / pre). Bottom: one dot per session; black = mean +- "
                 "SEM; * = nested-bootstrap change from pre.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(il.OUT / "figures" / "iti_lick_rate.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(pd.read_csv(il.OUT / "iti_lick_rate.csv"), pd.read_csv(il.OUT / "iti_lick_rate_tests.csv"),
             pd.read_csv(il.OUT / "iti_lick_rate_recovery.csv"))
        raise SystemExit(0)
    raise SystemExit(main())
