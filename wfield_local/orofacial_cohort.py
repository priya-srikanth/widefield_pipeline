"""Pre- vs post-stroke orofacial kinematics across the cohort, from the organised pose outputs (`orofacial_outputs`).

    python -m wfield_local.orofacial_cohort [--run lp_r4_heatmap_20261013__clean_v35] [--jobs 4]

Reads every CURATED session (`analysis_kit.curated_sessions`, the same pre + post set the imaging analyses use -- early
June 6/1-6/5 and 8/5 excluded) that has a processed session folder, reduces each to per-session numbers, and writes

    <LP_analysis>/<run>/cohort/tables/prepost/  session_metrics.csv     one row per session (animal, epoch, day, metrics)
                                              position_metrics.csv    one row per session x spout position
                                              epoch_by_animal.csv     per-animal epoch means (ground rule 8: read FIRST)
                                              epoch_deltas.csv        paired animal-weighted change vs pre (`boot_delta`)
    <LP_analysis>/<run>/cohort/figures/prepost/ timecourse.png, epochs.png, angle_tuning.png, by_position.png

Re-run as sessions arrive; every number is recomputed from the session folders (nothing cached here).

WHAT EACH METRIC IS (per session; post-cue = cue + 70 ms .. DAQ trial stop, `licks.csv`):
  resp_rate          fraction of trials with >= 1 post-cue lick (engagement as much as ability)
  licks_per_resp     post-cue licks per RESPONDED trial
  post_rate_hz       post-cue licks / summed response-window time
  latency_ms         median first post-cue lick (peak) time, responded trials
  ili_ms             median inter-lick interval inside a trial, intervals < 300 ms (within-bout rhythm)
  protrusion_px      median per-lick max protrusion, MEASURED peaks only (`peak_interp` False; Priya 10-15: be wary
                     of quantities read off interpolated points)
  contact_frac       fraction of post-cue licks with a DAQ spout contact (reach success, video-independent)
  angle_gain         executed-angle vs target: slope of the per-position median peak tongue angle (measured peaks) on
                     the per-position spout angle. 1 = the tongue angle scales with the target; 0 = no steering
  angle_gain_far     the same from the two far side positions only (far_L - far_R)
  precue_rate_hz     spontaneous licks, position strobe (spout arrived) .. cue + 70 ms, per second (`licks_precue.csv`)
  precue_trial_frac  fraction of trials with >= 1 pre-cue lick
  iti_rate_hz        inter-trial licks per second (`licks_iti.csv`; stop .. window end + window start .. strobe)
  peak_interp_frac   fraction of post-cue licks whose peak is interpolated / imputed (tracking-quality context)

SIDES: positions are close_/far_ x L / center / R; ipsi / contra from `animals.yaml stroke_laterality` (all four lesions
are LEFT, so R = contralateral). PS93 also has a pre-existing right orofacial deficit -- read its contra numbers with
that in mind. At n=4, the per-animal table is printed before any bootstrap (CLAUDE.md rule 8).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN = "lp_r4_heatmap_20261013__clean_v35"
EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")
METRICS = ("resp_rate", "licks_per_resp", "post_rate_hz", "latency_ms", "ili_ms", "protrusion_px", "contact_frac",
           "angle_gain", "angle_gain_far", "precue_rate_hz", "precue_trial_frac", "iti_rate_hz", "peak_interp_frac")
LABEL = {"resp_rate": "responded trials (frac)", "licks_per_resp": "licks / responded trial",
         "post_rate_hz": "post-cue lick rate (Hz)", "latency_ms": "first-lick latency (ms)",
         "ili_ms": "within-bout ILI (ms)", "protrusion_px": "max protrusion, measured (px)",
         "contact_frac": "licks with spout contact (frac)", "angle_gain": "angle gain (executed / target)",
         "angle_gain_far": "angle gain, far L vs R", "precue_rate_hz": "pre-cue lick rate (Hz)",
         "precue_trial_frac": "trials with pre-cue lick (frac)", "iti_rate_hz": "ITI lick rate (Hz)",
         "peak_interp_frac": "interpolated peaks (frac)"}
ILI_MAX_MS = 300.0
FRAME_MS = 4.0


def side_of(position: str, laterality: str) -> str:
    """'ipsi' / 'contra' / 'center' for a `close_L`-style position, relative to the lesion side."""
    s = position.rsplit("_", 1)[-1]
    if s == "center":
        return "center"
    return "ipsi" if s == laterality else "contra"


def _windows_durations(sd: Path) -> pd.DataFrame:
    """Per trial: pre-cue (strobe .. cue + 70 ms), response (cue + 70 .. stop) and ITI (rest of the window) seconds."""
    w = np.load(sd / "windows.npz")
    ix = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "t_ms", "strobe_ms", "stop_ms")})
    g = ix.groupby("trial_id", sort=True).agg(t0=("t_ms", "min"), t1=("t_ms", "max"), strobe=("strobe_ms", "first"),
                                               stop=("stop_ms", "first"))
    t1 = g.t1 + FRAME_MS
    pre = (70.0 - g.strobe).where(np.isfinite(g.strobe), np.nan).clip(lower=0)
    resp = (g.stop.clip(upper=t1) - 70.0).clip(lower=0)
    iti = (g.strobe - g.t0).clip(lower=0).fillna(0) + (t1 - g.stop).clip(lower=0).fillna(0)
    return pd.DataFrame({"pre_s": pre / 1000, "resp_s": resp / 1000, "iti_s": iti / 1000})


def _gain(med: pd.DataFrame) -> float:
    """Slope of median peak angle on spout angle over positions (>= 3 positions with a value)."""
    m = med.dropna(subset=["peak_angle_deg", "spout_angle_deg"])
    if len(m) < 3 or np.ptp(m.spout_angle_deg) < 10:
        return np.nan
    return float(np.polyfit(m.spout_angle_deg, m.peak_angle_deg, 1)[0])


def session_metrics(job) -> tuple[dict, list[dict]] | None:
    """(session row, position rows) for one session folder; module-level for `fan_sessions`."""
    sd, animal, date, epoch, day, lat = job
    rd = lambda n: pd.read_csv(sd / n) if (sd / n).exists() else pd.DataFrame()   # noqa: E731
    L, P, Q, T = rd("licks.csv"), rd("licks_precue.csv"), rd("licks_iti.csv"), rd("trials.csv")
    if not len(T) or "t_ms" not in L:
        return None
    dur = _windows_durations(sd)
    meas = L[~L.peak_interp.astype(bool)] if "peak_interp" in L else L
    by_trial = L.groupby("trial_id")
    resp_trials = by_trial.size()
    ili = (L.sort_values(["trial_id", "t_ms"]).groupby("trial_id").t_ms.diff())
    pos_med = (meas.groupby("position")[["peak_angle_deg", "spout_angle_deg"]].median())
    far = pos_med.reindex(["far_L", "far_R"])
    gain_far = (float((far.peak_angle_deg.iloc[0] - far.peak_angle_deg.iloc[1])
                      / (far.spout_angle_deg.iloc[0] - far.spout_angle_deg.iloc[1]))
                if far.notna().all().all() and abs(far.spout_angle_deg.iloc[0] - far.spout_angle_deg.iloc[1]) > 10
                else np.nan)
    n_tr = int(len(T))
    row = {"animal": animal, "date": date, "label": f"{animal}_{date[4:]}", "epoch": epoch, "day": day,
           "n_trials": n_tr, "n_licks": int(len(L)), "n_precue_licks": int(len(P)), "n_iti_licks": int(len(Q)),
           "resp_rate": len(resp_trials) / n_tr,
           "licks_per_resp": float(resp_trials.mean()) if len(resp_trials) else np.nan,
           "post_rate_hz": len(L) / dur.resp_s.sum() if dur.resp_s.sum() > 0 else np.nan,
           "latency_ms": float(by_trial.t_ms.min().median()) if len(L) else np.nan,
           "ili_ms": float(ili[ili < ILI_MAX_MS].median()) if ili.notna().any() else np.nan,
           "protrusion_px": float(meas.protrusion_max_measured_px.median()) if len(meas) else np.nan,
           "contact_frac": float(L.contact.astype(float).mean()) if "contact" in L else np.nan,
           "angle_gain": _gain(pos_med), "angle_gain_far": gain_far,
           "precue_rate_hz": len(P) / dur.pre_s.sum() if dur.pre_s.sum() > 0 else np.nan,
           "precue_trial_frac": (P.trial_id.nunique() / n_tr) if len(P) else 0.0,
           "iti_rate_hz": len(Q) / dur.iti_s.sum() if dur.iti_s.sum() > 0 else np.nan,
           "peak_interp_frac": float(L.peak_interp.astype(float).mean()) if "peak_interp" in L else np.nan}
    prow = []
    for pos, g in L.groupby("position"):
        gm = meas[meas.position == pos]
        prow.append({"animal": animal, "date": date, "epoch": epoch, "day": day, "position": pos,
                     "side": side_of(pos, lat), "ring": pos.split("_")[0], "n_licks": int(len(g)),
                     "contact_frac": float(g.contact.astype(float).mean()) if "contact" in g else np.nan,
                     "protrusion_px": float(gm.protrusion_max_measured_px.median()) if len(gm) else np.nan,
                     "peak_angle_deg": float(gm.peak_angle_deg.median()) if len(gm) else np.nan,
                     "spout_angle_deg": float(gm.spout_angle_deg.median()) if len(gm) else np.nan,
                     "reach_frac": float(g.reach_frac.median()) if "reach_frac" in g else np.nan,
                     "latency_ms": float(g.groupby("trial_id").t_ms.min().median())})
    return row, prow


def collect(rv, run: str = RUN, jobs: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs
    from wfield_local import orofacial_outputs as oo
    lat = {a: str(v.get("stroke_laterality", "L")) for a, v in config.animals().items()}
    items = []
    for s in ak.curated_sessions(order="sorted"):
        a = config.animal_of(s["label"])
        date = "2026" + s["label"].split("_")[-1]
        sd = oo.session_dir(rv, run, a, date)
        if (sd / "manifest.json").exists():
            items.append((sd, a, date, epochs.epoch_of(s["label"]), epochs.days_since_stroke(s["label"]), lat[a]))
    res, fail = ak.fan_sessions(items, session_metrics, jobs=jobs, key=lambda it: (it[1], it[2]), label="session")
    if fail:
        raise RuntimeError(f"{len(fail)} session(s) failed -- refusing to write a partial table: {fail}")
    rows = [v for _, v in res if v is not None]
    S = pd.DataFrame([r for r, _ in rows])
    Pm = pd.DataFrame([p for _, ps in rows for p in ps])
    return S, Pm


def epoch_tables(S: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-animal epoch means, and the paired animal-weighted change of each post epoch vs pre (`boot_delta`)."""
    from wfield_local import analysis_kit as ak
    A = S.groupby(["animal", "epoch"])[list(METRICS)].mean().reset_index()
    A["n_sessions"] = S.groupby(["animal", "epoch"]).size().to_numpy()
    rng = np.random.default_rng(seed)
    out = []
    for m in METRICS:
        pre = {a: g[m].dropna().tolist() for a, g in S[S.epoch == "pre"].groupby("animal", sort=True)}
        for ep in EPOCH_ORDER[1:]:
            post = {a: g[m].dropna().tolist() for a, g in S[S.epoch == ep].groupby("animal", sort=True)}
            pre_e = {a: v for a, v in pre.items() if v and post.get(a)}
            post_e = {a: v for a, v in post.items() if v and a in pre_e}
            iv = ak.boot_delta(post_e, pre_e, rng) if len(post_e) >= 2 else None
            out.append({"metric": m, "epoch": ep, "n_animals": len(post_e),
                        "delta": iv.point if iv else (float(np.mean([np.mean(post_e[a]) - np.mean(pre_e[a])
                                                                   for a in post_e])) if post_e else np.nan),
                        "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan,
                        "animals": ",".join(sorted(post_e))})
    return A, pd.DataFrame(out)


# --------------------------------------------------------------------------- figures

def _colors():
    from wfield_local import config
    return config.animal_color()


def fig_timecourse(S: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = _colors()
    n = len(METRICS)
    nc = 4
    fig, axs = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(4.2 * nc, 2.9 * np.ceil(n / nc)), squeeze=False)
    for ax, m in zip(axs.flat, METRICS):
        for a, g in S.sort_values("day").groupby("animal"):
            ax.plot(g.day, g[m], "o-", ms=3, lw=1, color=col.get(a, "k"), label=a)
        ax.axvline(0.5, color="0.4", ls="--", lw=0.8)
        ax.set_title(LABEL[m], fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("days since stroke", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_epochs(A: pd.DataFrame, D: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = _colors()
    n = len(METRICS)
    nc = 4
    fig, axs = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(4.0 * nc, 2.9 * np.ceil(n / nc)), squeeze=False)
    xi = {e: i for i, e in enumerate(EPOCH_ORDER)}
    for ax, m in zip(axs.flat, METRICS):
        for a, g in A.groupby("animal"):
            g = g[g.epoch.isin(xi)].assign(x=lambda d: d.epoch.map(xi)).sort_values("x")
            ax.plot(g.x, g[m], "o-", color=col.get(a, "k"), ms=4, lw=1.2, label=a)
        lab = []
        for e in EPOCH_ORDER[1:]:
            r = D[(D.metric == m) & (D.epoch == e)]
            if len(r) and np.isfinite(r.lo.iloc[0]):
                ci = (r.lo.iloc[0], r.hi.iloc[0])
                star = "*" if ci[0] > 0 or ci[1] < 0 else ""
                lab.append(f"{e[:3]} {r.delta.iloc[0]:+.3g}{star} (n={int(r.n_animals.iloc[0])})")
        ax.set_xticks(range(len(EPOCH_ORDER)), EPOCH_ORDER, fontsize=7)
        ax.set_title(LABEL[m] + ("\n" + "; ".join(lab) if lab else ""), fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in axs.flat[n:]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title + "\nper-animal epoch means; title: paired change vs pre, animal-weighted bootstrap "
                 "(* = 95% CI excludes 0; n = animals)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_angle_tuning(Pm: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ecol = dict(zip(EPOCH_ORDER, ("k", "tab:red", "tab:orange", "tab:blue")))
    animals = sorted(Pm.animal.unique())
    fig, axs = plt.subplots(1, len(animals), figsize=(3.6 * len(animals), 4.0), squeeze=False)
    for ax, a in zip(axs[0], animals):
        g = Pm[Pm.animal == a]
        for e in EPOCH_ORDER:
            ge = g[g.epoch == e]
            if not len(ge):
                continue
            # close and far spouts sit at the SAME image angle (~+-40 deg), so one line through all six zigzags:
            # one line per ring (far solid, close dashed)
            for ring, ls in (("far", "-"), ("close", "--")):
                m = (ge[ge.ring == ring].groupby("position")[["spout_angle_deg", "peak_angle_deg"]].mean()
                     .sort_values("spout_angle_deg"))
                ax.plot(m.spout_angle_deg, m.peak_angle_deg, "o" + ls, color=ecol[e], ms=4, lw=1.2,
                        label=f"{e} ({ge.date.nunique()})" if ring == "far" else None)
        lim = [-50, 50]
        ax.plot(lim, lim, color="0.8", lw=0.8, zorder=0)
        ax.set_xlim(lim)
        ax.set_title(a, fontsize=9)
        ax.set_xlabel("spout angle (deg)", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)
        ax.tick_params(labelsize=7)
    axs[0][0].set_ylabel("executed peak tongue angle (deg)", fontsize=8)
    fig.suptitle(title + "\nexecuted vs target angle: per-position medians (measured peaks), mean over sessions per "
                 "epoch\nfar = solid, close = dashed; grey = identity; + = toward the L positions (image right; "
                 "ipsilesional)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


POS_ORDER = ("far_ipsi", "close_ipsi", "close_center", "far_center", "close_contra", "far_contra")


def fig_by_position(Pm: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ecol = dict(zip(EPOCH_ORDER, ("k", "tab:red", "tab:orange", "tab:blue")))
    P = Pm.assign(pos=Pm.ring + "_" + Pm.side)
    animals = sorted(P.animal.unique())
    rows = ("contact_frac", "protrusion_px", "latency_ms")
    ylab = {"contact_frac": "contact (frac)", "protrusion_px": "protrusion (px)", "latency_ms": "latency (ms)"}
    fig, axs = plt.subplots(len(rows), len(animals), figsize=(3.6 * len(animals), 2.6 * len(rows)), squeeze=False)
    for j, a in enumerate(animals):
        for i, m in enumerate(rows):
            ax = axs[i][j]
            for e in EPOCH_ORDER:
                g = P[(P.animal == a) & (P.epoch == e)]
                if len(g):
                    v = g.groupby("pos")[m].mean().reindex(POS_ORDER)
                    ax.plot(range(len(POS_ORDER)), v.to_numpy(), "o-", color=ecol[e], ms=3.5, lw=1.1, label=e)
            ax.set_xticks(range(len(POS_ORDER)), [p.replace("_", "\n") for p in POS_ORDER], fontsize=6)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(a, fontsize=9)
            if j == 0:
                ax.set_ylabel(ylab[m], fontsize=8)
    axs[0][0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(title + "\nby spout position (ipsi / contra to the lesion), mean over sessions per epoch", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    base = oo.root(rv) / a.run / "cohort"
    tables, figs = base / "tables" / "prepost", base / "figures" / "prepost"
    writeguard.assert_writable(base)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    S, Pm = collect(rv, a.run, a.jobs)
    A, D = epoch_tables(S)
    pd.set_option("display.width", 200)
    print(f"{len(S)} curated sessions with pose outputs:")
    print(S.groupby(["animal", "epoch"]).size().unstack(fill_value=0).reindex(columns=list(EPOCH_ORDER), fill_value=0))
    print("\nPER-ANIMAL epoch means (rule 8 -- read before the deltas):")
    print(A.set_index(["animal", "epoch"])[["n_sessions", *METRICS]].round(3).to_string())
    print("\nchange vs pre (animal-weighted; CI only with >= 2 animals):")
    print(D.round(4).to_string(index=False))
    S.to_csv(tables / "session_metrics.csv", index=False)
    Pm.to_csv(tables / "position_metrics.csv", index=False)
    A.to_csv(tables / "epoch_by_animal.csv", index=False)
    D.to_csv(tables / "epoch_deltas.csv", index=False)
    animals = ", ".join(f"{x} {int((S.animal == x).sum())}" for x in sorted(S.animal.unique()))
    t = f"Orofacial kinematics, pre vs post stroke -- {a.run} -- sessions: {animals}"
    fig_timecourse(S, figs / "timecourse.png", t)
    fig_epochs(A, D, figs / "epochs.png", t)
    fig_angle_tuning(Pm, figs / "angle_tuning.png", t)
    fig_by_position(Pm, figs / "by_position.png", t)
    (tables / "provenance.json").write_text(json.dumps({"run": a.run, "n_sessions": int(len(S)),
                                                        "code_commit": oo.code_commit(),
                                                        "sessions": S.label.tolist()}, indent=1), encoding="utf-8")
    print(f"\ntables -> {tables}\nfigures -> {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
