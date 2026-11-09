"""Does INTER-TRIAL licking track recovery, and does it exceed pre-stroke levels? (Priya 10-26.)

BEHAVIOR only. Per session: ITI licks = licks_all.csv context "iti" (every cam4-detected lick between trials; the
spout is retracted, except at the window edges); ITI time = sum over consecutive trials of next arrival (strobe) minus
this trial's stop, clipped at 0 and at 60 s (long pauses excluded); rate = ITI licks / ITI minutes. Also the share of
ITI licks that touch the spout (DAQ contact flag) and the no-contact rate. Per-trial counts (ITI licks per trial) as a
second definition. Normalized to the animal's pre mean.
CAMERA WORKING PERIOD (Priya 11-05, after (f) 2): only the ITIs of trials inside the camera working period count --
first cue -> the session's last camera lick, minus the stopped runs (`trial_classes.camera_period_trials` on the trial
cues; an ITI lick belongs to the trial it follows, `trial_classes.working_trial_ids` on its licks_all trial_id). So
licking after the camera quit and inside stopped runs leaves both the numerator and the ITI time. A session without
camera classes is left out and listed. Until 11-05 the whole session counted.
Tests: change from pre per epoch (nested bootstrap, per-animal table first); within animal across POST sessions, the
correlation with far-contra hit rate / pre (`recovery_timing.series`), raw and with log(day) removed (permutation of
behavior within animal, as `iti_recovery_marker.py`).
Outputs <E>/iti_licks/iti_lick_rate.csv, figures/lick_iti_rate.png.
Figure data (figdata.py convention): <E>/iti_licks/figdata/lick_iti_rate.pkl = dict(R sessions, T change tests, C recovery
correlations); `--plot-only` redraws from that file alone (no bootstrap, no permutation).
BEHAVIOR AXIS (Priya 10-28: "pooled please, and plot over time and also vs recovery axis"): the recovery axis = the
far-contra FIRST-cued-lick aim offset (`iti_recovery_measure`, degrees vs the animal's pre landing, - = ipsilesional; as
Fig 10.5); far-contra hit rate / pre kept as a sensitivity (column `beh`, rows behavior="hit" in the recovery table).
Figure: row 1 over days (per animal + POOLED), row 2 against the aim (per animal + POOLED), row 3 per-epoch summaries.
Usage: python scripts/iti_lick_rate.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402
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
    from wfield_local import trial_classes as tc
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["trial_id", "context", "contact"])
    T = il._behaviour(label).trials.sort_values("cue_s")
    # the camera working period (11-05): trial i's ITI (its stop -> the next arrival) counts only when trial i is in it
    work, _contact, has = tc.camera_period_trials(label, T.cue_s.to_numpy(float))
    if not has:
        return None                                   # no video: no camera working period (left out, listed)
    st = np.where(np.isfinite(T.strobe_s), T.strobe_s, T.cue_s - 3.0)
    sp = T.stop_s.to_numpy(float)
    gaps = st[1:] - sp[:-1]
    gaps = gaps[np.isfinite(gaps) & work[:-1]]
    iti_min = float(np.clip(gaps, 0, GAP_MAX).sum() / 60.0)
    wk, _h = tc.working_trial_ids(label)
    a = A[(A.context == "iti") & A.trial_id.map(lambda t: bool(wk.get(int(t), False)) if pd.notna(t) else False)]
    T = T[work]
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
    labels = il.labels()
    rows = {x: session_row(x) for x in labels}
    skipped = sorted(x for x, r in rows.items() if r is None)
    if skipped:
        print(f"== no camera classes (no camera working period), left out: {skipped}", flush=True)
    R = pd.DataFrame([r for r in rows.values() if r is not None])
    VALS = ("rate", "rate_nocontact", "per_trial", "share_contact")
    for v in VALS:
        pre = R[R.epoch == "pre"].groupby("animal")[v].mean()
        R[f"{v}_n"] = R[v] / R.animal.map(pre)
    R = R.merge(series("lick")[["label", "day", "beh"]], on="label", how="left")
    import iti_recovery_measure as rm
    R = R.merge(pd.read_csv(rm.sessions_csv("first"))[["label", "far_contra_signed"]].rename(
        columns={"far_contra_signed": "aim"}), on="label", how="left")
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
    print("\n== change from pre (normalized to the animal's pre = 1)\n" + T.round(3).to_string(index=False))
    rng = np.random.default_rng(0)
    out = []
    for bcol, bname in (("aim", "fc_aim"), ("beh", "hit")):     # recovery axis (primary), hit rate (sensitivity)
        G = R[(R.epoch != "pre") & R[bcol].notna() & (R.day > 0)].copy()
        for v in ("rate_n", "rate_nocontact_n", "per_trial_n"):
            def stat(sh, v=v, G=G):
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
            beh = {a: G[G.animal == a][bcol].to_numpy() for a in AN}
            obs = stat(beh)
            null = {k: [] for k in obs}
            for _ in range(N_PERM):
                s_ = stat({a: rng.permutation(beh[a]) for a in AN})
                for k in null:
                    null[k].append(np.nanmean(s_[k]))
            for k, rs in obs.items():
                m = np.nanmean(rs)
                out.append({"behaviour": bname, "value": v, "kind": k, **dict(zip(AN, np.round(rs, 2))),
                            "mean_r": round(m, 3), "p_perm": (np.sum(np.abs(null[k]) >= abs(m)) + 1) / (N_PERM + 1)})
    C = pd.DataFrame(out)
    print("\n== within-animal correlation with the far-contra first-lick aim (fc_aim) / hit rate / pre (hit), post "
          "sessions\n" + C.to_string(index=False))
    T.to_csv(il.OUT / "iti_lick_rate_tests.csv", index=False)
    C.to_csv(il.OUT / "iti_lick_rate_recovery.csv", index=False)
    D = dict(R=R, T=T, C=C)
    save_fd("lick_iti_rate", D)
    plot(D)
    return 0


def plot(D: dict) -> None:
    R, T, C = D["R"], D["T"], D["C"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    Ca = C[(C.behaviour == "fc_aim") & (C.value == "rate_n")].set_index("kind")
    marks = dict(zip(AN, "osD^"))
    fig, axs = plt.subplots(3, 5, figsize=(19, 11.5))
    rng2 = np.random.default_rng(1)
    # ROW 1: over days, per animal (twin axis = far-contra first-lick aim, dotted) + POOLED
    for j, a in enumerate(AN):
        ax = axs[0, j]
        g = R[R.animal == a].sort_values("day")
        ax.plot(g.day, g.rate_n, color="0.7", lw=0.8, zorder=0)
        for ep in EP:
            h = g[g.epoch == ep]
            ax.scatter(h.day, h.rate_n, s=22, color=colors[ep], label=ep, zorder=2)
        ax2 = ax.twinx()
        ga = g.dropna(subset=["aim"])
        ax2.plot(ga.day, ga.aim, color="0.3", lw=1, ls=":")
        ax2.tick_params(labelsize=7)
        if j == 3:
            ax2.set_ylabel("far-contra first-lick aim offset, deg (dotted)", fontsize=8)
        ax.axhline(1, color="0.6", lw=0.6, ls="--")
        ax.axvline(0, color="0.6", lw=0.6)
        ax.set_title(f"{a}: ITI lick rate over days", fontsize=9)
        ax.set_xlabel("day from stroke", fontsize=8)
        if j == 0:
            ax.set_ylabel("ITI lick rate / pre", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
    ax = axs[0, 4]                                        # POOLED over days; log y (PS95 reaches 13x its pre rate)
    for a in AN:
        for ep in EP:
            h = R[(R.animal == a) & (R.epoch == ep)]
            ax.scatter(h.day, h.rate_n, s=14, color=colors[ep], marker=marks[a], alpha=0.75, lw=0)
    E_ = R.groupby(["animal", "epoch"]).agg(day=("day", "median"), v=("rate_n", "mean")).reset_index()
    M_ = E_.groupby("epoch").agg(day=("day", "median"), v=("v", "mean")).reindex(EP)
    ax.plot(M_.day, M_.v, color="k", lw=2, marker="o", zorder=3)
    ax.set_yscale("log")
    ax.axhline(1, color="0.6", lw=0.6, ls="--")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_title("POOLED over days (marker = animal; black = mean of\nanimals per epoch, at the epoch's median day)",
                 fontsize=8.5)
    ax.set_xlabel("day from stroke", fontsize=8)
    ax.legend([Line2D([], [], ls="", marker=marks[a], color="0.4") for a in AN], AN, fontsize=7, frameon=False)
    # ROW 2: against the recovery axis (post sessions), per animal + POOLED
    P_ = R[(R.epoch != "pre") & R.aim.notna()]
    for j, a in enumerate(AN + ["pooled"]):
        ax = axs[1, j]
        g = P_ if a == "pooled" else P_[P_.animal == a]
        for ep in EP[1:]:
            h = g[g.epoch == ep]
            if a == "pooled":
                for a_ in AN:
                    hh = h[h.animal == a_]
                    ax.scatter(hh.aim, hh.rate_n, s=14, color=colors[ep], marker=marks[a_], alpha=0.75, lw=0)
            else:
                ax.scatter(h.aim, h.rate_n, s=22, color=colors[ep], label=ep)
        ax.axhline(1, color="0.6", lw=0.6, ls="--")
        ax.axvline(0, color="0.6", lw=0.6)
        if a == "pooled":
            ax.set_yscale("log")
            ax.set_title(f"POOLED vs aim: mean within-animal r {Ca.loc['raw', 'mean_r']:+.2f} (p "
                         f"{Ca.loc['raw', 'p_perm']:.3f})\nlog(day) removed {Ca.loc['detrended', 'mean_r']:+.2f} (p "
                         f"{Ca.loc['detrended', 'p_perm']:.3f})", fontsize=8.5)
        else:
            ax.set_title(f"{a}: vs aim, r {Ca.loc['raw', a]:+.2f} (log(day) removed {Ca.loc['detrended', a]:+.2f})",
                         fontsize=9)
        ax.set_xlabel("far-contra first-lick aim offset (deg; - = ipsilesional)", fontsize=8)
        if j == 0:
            ax.set_ylabel("ITI lick rate / pre (post sessions)", fontsize=8)
    axs[2, 4].axis("off")
    # ROW 3: per-epoch summaries, sessions pooled
    for j, v in enumerate(("rate", "rate_nocontact", "per_trial", "share_contact")):
        ax = axs[2, j]
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
    fig.suptitle("BEHAVIOR: inter-trial (self-initiated) licking across recovery, divided by the animal's pre mean. Row 1: "
                 "over days (dotted = far-contra first-lick aim offset). Row 2: against the recovery axis (far-contra\n"
                 "first-lick aim offset; post sessions; r within animal, p = within-animal permutation). Row 3: one dot per "
                 "session; black = mean +- SEM; * = nested-bootstrap change from pre. Pooled panels on a log scale.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_svg(fig, il.OUT / "figures" / "lick_iti_rate.png", dpi=110)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The lick set of lick_iti_rate (Priya 11-09): every cam4 ITI lick of licks_all.csv that follows a trial of the
    camera working period (`session_row`), over the ITI time of those trials; sessions without camera classes out."""
    from wfield_local import trial_classes as tc
    from wfield_local import trial_set_text as tst
    end = tst._END_TEXT.get(tc.end_anchor(), tc.end_anchor())
    return tst.events(
        "ITI licks (cam4 licks_all.csv, context iti: every camera-detected lick between trials; no 3-D requirement)",
        keep=[f"only licks following a trial of the camera working period (first cue to the {end}, stopped runs "
              f"({tst.stop_text()}) removed); ITI time = those trials' stop to the next spout arrival, each gap clipped "
              f"to 0..{GAP_MAX:g} s"],
        excluded=["post-cue licks", "ITI licks after the camera working period or inside stopped runs",
                  "sessions without camera classes (no video)"])


def cli() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())                # declared before any draw: the --plot-only replot too
    if plot_only():                                       # redraw from the figure data alone
        plot(load_fd("lick_iti_rate"))
        return 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())
