"""Do animals "try harder" with the body when they cannot lick? Running and limb cortex by trial outcome (cohort1, 11-01).

Priya 11-01: "acutely I also feel like I see the animal 'trying' with more treadmill movement and maybe more limb
activation when it's unable to lick -- do we see this on either jaw activation but no lick, or lick but not contact
post-stroke". Trial classes from the encoding window's attempt table (imaging_encoding/qc/trial_attempts.csv):
hit (contact), incomplete (lick, no contact), jaw_only (jaw movement, no lick; blind review = reactions), no_try.
Per trial, window = cue -> cue + 3.5 s (the response window), baseline = cue - 2 .. cue:
  * RUNNING: fraction of the window inside a DAQ running bout (behavior_events running_starts / stops, 3 mm/s), the same
    in the baseline, and whether a bout STARTS in the window.
  * LIMB CORTEX: mean dF/F % of the bilateral limb ROI (SSp-ul + SSp-ll) in cue .. +1.5 s minus cue - 1 .. 0 s; the
    same for orofacial and whole cortex (to tell a limb-specific rise from a global one).
Compared within epoch across classes (does a failed attempt carry more running / limb activity than a hit or a no-try?)
and per class across epochs. Far-contra trials reported separately (where acute failures concentrate).

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/attempt_effort.py [--jobs 5] [--summarise]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import EVENTS, columns, session_traces  # noqa: E402
from quiet_spikes_v2 import V2  # noqa: E402
from wfield_local import lp_licks as _lp_licks  # noqa: E402  (coordinator 11-04: the run's attempts table)

OUTE = V2.parent / "attempt_effort"
TA = _lp_licks.attempts_path()      # THIS run's attempts table (coordinator 11-04)
WIN_S, BASE_S, CTX_S, CTX_BASE_S = 3.5, 2.0, 1.5, 1.0
CLS = ["hit", "incomplete", "jaw_only", "no_try"]
EP = ["pre", "acute", "subacute", "chronic"]


def frac_in(bouts, a, b):
    if not len(bouts):
        return 0.0
    ov = np.clip(np.minimum(bouts[:, 1], b) - np.maximum(bouts[:, 0], a), 0, None)
    return float(ov.sum() / (b - a))


def extract(label):
    f = OUTE / "sessions" / f"{label}.csv"
    if f.exists():
        return label
    import pandas as pd
    from wfield_local import trial_windows as TW
    animal, mmdd = label.split("_")
    T = pd.read_csv(TA)
    T = T[T.label == label]
    if T.empty:
        return label
    tb = TW.trial_bounds(animal, f"2026{mmdd}").set_index("trial_id")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fsd = float(z["fs"])
    run = np.c_[z["running_starts"], z["running_stops"]] / fsd
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    cols = columns()
    ci = {k: cols.index(k) for k in ("limb", "orofacial", "cortex")}
    rows = []
    for _, r in T.iterrows():
        if r.trial_id not in tb.index:
            continue
        cue = float(tb.loc[r.trial_id, "cue_s"])
        if not np.isfinite(cue):
            continue
        d = dict(label=label, trial_id=int(r.trial_id), pos=r.pos, cls=r.cls, animal=animal, epoch=r.epoch, day=r.day,
                 run_win=frac_in(run, cue, cue + WIN_S), run_base=frac_in(run, cue - BASE_S, cue),
                 run_start=bool(np.any((run[:, 0] >= cue) & (run[:, 0] < cue + WIN_S))) if len(run) else False)
        a0, a1 = np.searchsorted(ft, [cue, cue + CTX_S])
        b0, b1 = np.searchsorted(ft, [cue - CTX_BASE_S, cue])
        for k, i in ci.items():
            if a1 > a0 and b1 > b0:
                d[f"{k}_resp"] = float(np.nanmean(Xr[i, a0:a1]) - np.nanmean(Xr[i, b0:b1]))
        rows.append(d)
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f, index=False)
    return label


def summarise():
    import pandas as pd
    pd.set_option("display.width", 220)
    S = pd.concat([pd.read_csv(f) for f in sorted((OUTE / "sessions").glob("*.csv"))], ignore_index=True)
    S = S[S.label.str[-4:] >= "0806"]
    S["run_delta"] = S.run_win - S.run_base
    S.to_csv(OUTE / "attempt_effort_trials.csv", index=False)
    for sub, G in (("ALL spouts", S), ("FAR CONTRA", S[S.pos == "far_contra"])):
        for m in ("run_win", "run_delta", "run_start", "limb_resp", "orofacial_resp", "cortex_resp"):
            A = G.groupby(["label", "animal", "epoch", "cls"])[m].mean().reset_index()
            P = A.pivot_table(index=["animal", "cls"], columns="epoch", values=m, aggfunc="mean").reindex(columns=EP)
            print(f"\n[{sub}] {m}: session means -> animal x class x epoch:\n",
                  P.reindex([(a, c) for a in sorted(G.animal.unique()) for c in CLS if (a, c) in P.index])
                  .round(3).to_string())
    return S


def contrasts(S):
    """Within-session paired differences: failed class minus hits (same session), per animal x epoch."""
    import pandas as pd
    out = []
    for m in ("run_win", "run_delta", "limb_resp", "cortex_resp"):
        A = S.groupby(["label", "animal", "epoch", "cls"])[m].mean().unstack("cls")
        for c in ("incomplete", "jaw_only", "no_try"):
            if c in A and "hit" in A:
                d = (A[c] - A["hit"]).dropna().reset_index()
                g = d.groupby(["animal", "epoch"])[0].mean().unstack().reindex(columns=EP)
                for an, r in g.iterrows():
                    out.append(dict(measure=m, cls=c, animal=an, **{ep: r[ep] for ep in EP}))
    T = pd.DataFrame(out)
    T.to_csv(OUTE / "contrast_vs_hit.csv", index=False)
    print("\nWITHIN-SESSION class minus hit (animal mean of session differences):\n", T.round(3).to_string())
    return T


def figure(S, out=None):
    """Priya 11-03 rules: no lines; per epoch, the four trial classes side by side; each class's SESSION means (mean over
    that session's trials) as faded open markers (shape = animal, colour = class), the animal means (mean over the
    animal's trials, as before) small filled, and the mean of animals as a large marker +- SEM across animals.
    Returns {(subset, measure, epoch, class): (animal means, mean of animals)} for the --plot-only check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import cat_points as cp
    col = {"hit": "#1b9e77", "incomplete": "#d95f02", "jaw_only": "#7570b3", "no_try": "0.5"}
    ms = [("run_win", "fraction of response window running"), ("run_delta", "running: window - pre-cue baseline"),
          ("limb_resp", "limb cortex (SSp-ul/ll) dF/F %, cue..+1.5 s - baseline"),
          ("cortex_resp", "whole cortex dF/F %, same window")]
    off = cp.epoch_offsets(len(CLS), step=0.2)
    plotted = {}
    fig, ax = plt.subplots(2, len(ms), figsize=(4.4 * len(ms), 8.4))
    for r, (sub, G) in enumerate((("all spouts", S), ("far contra", S[S.pos == "far_contra"]))):
        for a, (m, lab) in zip(ax[r], ms):
            A = G.groupby(["animal", "epoch", "cls"])[m].mean().reset_index()
            Ss = G.groupby(["label", "animal", "epoch", "cls"])[m].mean().reset_index()
            for k, ep in enumerate(EP):
                for j, c in enumerate(CLS):
                    x = k + off[j]
                    s_ = Ss[(Ss.epoch == ep) & (Ss.cls == c)]
                    cp.session_dots(a, x, s_[m], s_.animal, col[c], jitter=0.035, seed=100 * k + j, ms=3.0,
                                    alpha=0.35)
                    g = A[(A.epoch == ep) & (A.cls == c)]
                    for _, rr in g.iterrows():
                        a.plot(x + 0.05, rr[m], cp.animal_marker(rr.animal), ls="none", color=col[c], mec="w",
                               mew=0.4, ms=4, zorder=3)
                    if len(g):
                        v = g[m].to_numpy(float)
                        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
                        plotted[(sub, m, ep, c)] = (dict(zip(g.animal, v)), float(v.mean()))
                        cp.summary_point(a, x, v.mean(), se, col[c], ms=6, label=c if (k == 1 and r == 0) else None)
            for x in range(len(EP) - 1):
                a.axvline(x + 0.5, color="0.92", lw=0.6, zorder=0)
            # y range from the animals + the central 98 % of sessions: a session with one or two trials of a class
            # can sit far out and would flatten every summary; those are drawn as edge carets and counted
            sv = Ss[m].dropna().to_numpy(float)
            av = A[m].dropna().to_numpy(float)
            lo = min(np.quantile(sv, 0.01), av.min(), 0.0)
            hi = max(np.quantile(sv, 0.99), av.max(), 0.0)
            pad = 0.06 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            n_out = 0
            for k, ep in enumerate(EP):
                for j, c in enumerate(CLS):
                    s_ = Ss[(Ss.epoch == ep) & (Ss.cls == c)][m].dropna()
                    for y_, mk in ((hi, "^"), (lo, "v")):
                        n = int((s_ > hi).sum() if mk == "^" else (s_ < lo).sum())
                        if n:
                            n_out += n
                            a.plot(k + off[j], y_, mk, ls="none", mfc="none", mec=col[c], ms=4, alpha=0.7,
                                   clip_on=False)
            a.set_ylim(lo, hi)
            a.axhline(0, color="0.6", lw=0.6)
            a.set_xticks(range(4))
            a.set_xticklabels(EP, fontsize=8)
            a.set_title(f"{sub}: {lab}" + (f"\n({n_out} session values off-axis, carets)" if n_out else ""),
                        fontsize=8)
    h, l_ = ax[0, 0].get_legend_handles_labels()
    fig.legend(h + cp.animal_handles(sorted(S.animal.unique())), l_ + [f"{x} sessions" for x in
                                                                      sorted(S.animal.unique())],
               fontsize=8, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 0.935), frameon=False)
    fig.suptitle("Do animals recruit the body when they cannot lick? Running and limb cortex by trial outcome (hit / "
                 "lick-no-contact / jaw-only / no try). Large = mean of animals +- SEM across animals; small filled = "
                 "animal (shape; mean over its trials); faded open = sessions (shape = animal, colour = class)",
                 fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = OUTE / "attempt_effort.png" if out is None else out
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)
    return plotted


def plot_only():
    """Re-render from the stored attempt_effort_trials.csv only (no extraction) -> attempt_effort_nolines.png, and check
    the plotted animal means against an independent pivot of that table."""
    import pandas as pd
    S = pd.read_csv(OUTE / "attempt_effort_trials.csv", low_memory=False)
    P = figure(S, out=OUTE / "attempt_effort_nolines.png")
    for (sub, m, ep, c), (am, mu) in P.items():
        G = S if sub == "all spouts" else S[S.pos == "far_contra"]
        W = G[(G.epoch == ep) & (G.cls == c)].pivot_table(index="animal", values=m, aggfunc="mean")[m]
        assert np.allclose([am[a] for a in W.index], W.to_numpy()) and np.isclose(mu, W.mean()), (sub, m, ep, c)
    print(f"{len(P)} class x epoch summaries == attempt_effort_trials.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="re-render from attempt_effort_trials.csv -> *_nolines.png")
    a = ap.parse_args()
    if a.plot_only:
        plot_only()
        return
    if not a.summarise:
        import pandas as pd
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = sorted(x for x in pd.read_csv(TA, usecols=["label"]).label.unique() if x[-4:] >= "0806")
        if a.reverse:
            fan_sessions(labels[::-1], extract, jobs=a.jobs, label="session")
            return
        res, fail = fan_sessions(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
    S = summarise()
    contrasts(S)
    figure(S)
    (OUTE / "README.txt").write_text(__doc__)


if __name__ == "__main__":
    main()
