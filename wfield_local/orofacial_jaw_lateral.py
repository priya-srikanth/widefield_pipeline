"""Jaw lateral readout on WHOLE sessions with the production pose model (handoff 10-16 to-do 4; 10-14 to-do 10).

    python -m wfield_local.orofacial_jaw_lateral [--run lp_r4_heatmap_20261013__clean_v35] [--jobs 6]

Same quantities as `scripts/jaw_lateral.py deviation` (DLC r3 on 60-trial clips, DECISIONS 2026-10-15), now from the
organised session folders (`orofacial_outputs`: LP r4, cleaning v35, every trial of every curated session):

  per trial  lr_wide  median sideways jaw displacement from rest over frames with opening > OPEN_PX (50 px), in the
                      SPOUT frame (lr axis perpendicular to the apparatus centre line; + = image right = the mouse's
                      LEFT = ipsilesional for all four animals), measured jaw frames only, cue + 70 .. 3000 ms;
             angle    jaw opening angle, atan2(lr, opening) from a least-squares fit over open frames (> 15 px);
             open_max widest opening; reach (max tongue protrusion / spout distance over the trial's licks).
  per session FAR_DIR  = (far_L - far_R) / 2 of the per-trial medians: the jaw's movement TOWARD the target side.
                       Positive when the jaw goes left for the left spout and right for the right spout.
              FAR_BIAS = (far_L + far_R) / 2: a side bias independent of the target.
  Close / centre spouts are not used: they hide part of the chin and the model then marks its visible edge.
  Control (Priya 10-15: "jaw following a shorter reach?"): FAR_DIR recomputed on trials whose tongue reached the spout
  (reach >= 1), so an acute drop cannot come only from shorter licks.

Outputs: <LP_analysis>/<run>/cohort/tables/jaw_lateral/ (trials.csv, sessions.csv, epoch_by_animal.csv, epoch_deltas.csv)
and figures/jaw_lateral/ (timecourse.png, epochs.png, by_position.png). Per-animal table printed first (rule 8).
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

RUN = "lp_r4_heatmap_20261013__clean_v35"
OPEN_PX = 50.0
OPEN_FIT_PX = 15.0
WIN_MS = (70.0, 3000.0)                    # = orofacial_kinematics lick12_detect_win_ms (the 10-15 readout)
EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")
EPOCH_LINE = {"pre": "k", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}   # = orofacial_cohort
POS_ORDER = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
MEASURES = {"lr_wide": "jaw sideways at wide opening (px; + = ipsi)", "angle": "jaw opening angle (deg; + = ipsi)"}


def session_trials(job) -> list[dict]:
    """Per-trial jaw lateral numbers for one session folder (module-level for `fan_sessions`)."""
    sd, animal, date, epoch, day, lat = job
    from wfield_local.orofacial_cohort import pos_key
    c = np.load(sd / "clean.npz")
    w = np.load(sd / "windows.npz")
    jx, jy, fill = c["jaw_x"].astype(float), c["jaw_y"].astype(float), c["jaw_fill"]
    ap = np.asarray(c["spout_ap_axis"], float)
    ap = ap / np.hypot(*ap)
    lra = np.array([-ap[1], ap[0]])                 # perpendicular to the apparatus centre line ...
    if lra[0] < 0:                                  # ... with + pointing to image right (spout_frame's convention)
        lra = -lra
    opening, lr = jx * ap[0] + jy * ap[1], jx * lra[0] + jy * lra[1]
    kept = fill == 0
    if np.nanmedian(np.where(kept & (jy > 30), opening, np.nan)) < 0:        # + opening = jaw dropping
        opening = -opening
    sgn = 1.0 if lat == "L" else -1.0                                      # + = ipsilesional
    lr = sgn * lr
    t = w["ix_t_ms"].astype(float)
    tid = w["ix_trial_id"]
    pos = w["ix_position"].astype(str)
    win = kept & (t >= WIN_MS[0]) & (t <= WIN_MS[1]) & np.isfinite(opening) & np.isfinite(lr)
    L = pd.read_csv(sd / "licks.csv")
    reach = L.groupby("trial_id").reach_frac.max() if len(L) else pd.Series(dtype=float)
    prot = (L[~L.peak_interp.astype(bool)].groupby("trial_id").protrusion_max_measured_px.max()
            if len(L) else pd.Series(dtype=float))
    out = []
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    ends = np.r_[starts[1:], len(tid)]
    for a, b in zip(starts, ends):
        m = np.zeros(len(tid), bool)
        m[a:b] = win[a:b]
        o, lv = opening[m], lr[m]
        wide, op = o > OPEN_PX, o > OPEN_FIT_PX
        ang = (float(np.degrees(np.arctan2(np.sum(lv[op] * o[op]), np.sum(o[op] ** 2)))) if op.sum() >= 10 else np.nan)
        t_id = int(tid[a])
        out.append({"animal": animal, "date": date, "epoch": epoch, "day": day, "trial_id": t_id,
                    "position": pos[a], "pos": pos_key(pos[a], lat),
                    "lr_wide": float(np.median(lv[wide])) if wide.sum() >= 3 else np.nan, "n_wide": int(wide.sum()),
                    "angle": ang, "open_max": float(np.max(o)) if o.size else np.nan,
                    "reach": float(reach.get(t_id, np.nan)), "protrusion_px": float(prot.get(t_id, np.nan))})
    return out


def far_pair(T: pd.DataFrame, col: str, rng) -> dict:
    """FAR_DIR / FAR_BIAS (+ trial-bootstrap 95% CI) from one session's trials; ipsi / contra -> far ipsi / contra."""
    a_ = T.loc[T.pos == "far_ipsi", col].dropna().to_numpy(float)
    b_ = T.loc[T.pos == "far_contra", col].dropna().to_numpy(float)
    if min(a_.size, b_.size) < 3:
        return {}
    ma = np.median(rng.choice(a_, (1000, a_.size)), axis=1)
    mb = np.median(rng.choice(b_, (1000, b_.size)), axis=1)
    d, bias = (np.median(a_) - np.median(b_)) / 2, (np.median(a_) + np.median(b_)) / 2
    return {f"{col}_far_dir": float(d), f"{col}_far_dir_lo": float(np.percentile((ma - mb) / 2, 2.5)),
            f"{col}_far_dir_hi": float(np.percentile((ma - mb) / 2, 97.5)), f"{col}_far_bias": float(bias),
            f"{col}_n": int(min(a_.size, b_.size))}


def summarise(T: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (a, d), g in T.groupby(["animal", "date"], sort=True):
        r = {"animal": a, "date": d, "epoch": g.epoch.iloc[0], "day": g.day.iloc[0], "n_trials": len(g)}
        for col in MEASURES:
            r.update(far_pair(g, col, rng))
            # control: only trials whose tongue reached the spout (reach >= 1)
            fp = far_pair(g[g.reach >= 1.0], col, rng)
            r.update({k.replace("_far_", "_reached_far_"): v for k, v in fp.items() if "_far_" in k})
        far = g[g.pos.isin(["far_ipsi", "far_contra"])]
        r.update(open_max_far=float(far.open_max.median()), reach_far=float(far.reach.median()),
                 reached_frac_far=float((far.reach >= 1).mean()) if len(far) else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


SUMMARY_COLS = {"lr_wide_far_dir": "jaw steering FAR_DIR, wide opening (px)",
                "lr_wide_reached_far_dir": "FAR_DIR, trials where the tongue reached the spout (px)",
                "angle_far_dir": "jaw steering FAR_DIR, opening angle (deg)",
                "angle_reached_far_dir": "FAR_DIR angle, tongue reached the spout (deg)",
                "lr_wide_far_bias": "jaw side bias FAR_BIAS (px; + = ipsi)",
                "open_max_far": "widest jaw opening, far spouts (px)",
                "reach_far": "tongue reach, far spouts (protrusion / spout distance)",
                "reached_frac_far": "far trials where the tongue reached the spout (frac)"}


def epoch_tables(S: pd.DataFrame, seed: int = 0):
    from wfield_local import analysis_kit as ak
    cols = [c for c in SUMMARY_COLS if c in S]
    g = S.groupby(["animal", "epoch"])
    A = g[cols].mean().reset_index()
    A.insert(2, "n_sessions", g.size().to_numpy())
    rng = np.random.default_rng(seed)
    D = []
    for c in cols:
        pre = {a: x[c].dropna().tolist() for a, x in S[S.epoch == "pre"].groupby("animal", sort=True)}
        for ep in EPOCH_ORDER[1:]:
            post = {a: x[c].dropna().tolist() for a, x in S[S.epoch == ep].groupby("animal", sort=True)}
            pre_e = {a: v for a, v in pre.items() if v and post.get(a)}
            post_e = {a: v for a, v in post.items() if v and a in pre_e}
            iv = ak.boot_delta(post_e, pre_e, rng) if len(post_e) >= 2 else None
            D.append({"metric": c, "epoch": ep, "n_animals": len(post_e),
                      "delta": iv.point if iv else (float(np.mean([np.mean(post_e[a]) - np.mean(pre_e[a])
                                                                 for a in post_e])) if post_e else np.nan),
                      "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan, "animals": ",".join(sorted(post_e))})
    return A, pd.DataFrame(D)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _sem(x) -> float:
    x = pd.Series(x).dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def fig_epochs(S: pd.DataFrame, D: pd.DataFrame, path: Path, title: str) -> None:
    from wfield_local import config
    col = config.animal_color()
    cols = [c for c in SUMMARY_COLS if c in S]
    plt = _plt()
    nc = 4
    fig, axs = plt.subplots(int(np.ceil(len(cols) / nc)), nc, figsize=(4.0 * nc, 3.0 * np.ceil(len(cols) / nc)),
                            squeeze=False)
    for ax in axs.flat[len(cols):]:
        ax.axis("off")
    for ax, c in zip(axs.flat, cols):
        for k, (a, g) in enumerate(S.groupby("animal")):
            mu = g.groupby("epoch")[c].mean().reindex(EPOCH_ORDER)
            se = g.groupby("epoch")[c].agg(_sem).reindex(EPOCH_ORDER)
            x = np.arange(len(EPOCH_ORDER)) + (k - 1.5) * 0.06
            ok = mu.notna().to_numpy()
            ax.errorbar(x[ok], mu[ok], yerr=se[ok], fmt="o-", color=col.get(a, "k"), ms=4, lw=1.2, capsize=2,
                        elinewidth=0.8, label=a, zorder=3)
        if "far_dir" in c or "bias" in c:
            ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        lab = []
        for e in EPOCH_ORDER[1:]:
            r = D[(D.metric == c) & (D.epoch == e)]
            if len(r) and np.isfinite(r.lo.iloc[0]):
                star = "*" if r.lo.iloc[0] > 0 or r.hi.iloc[0] < 0 else ""
                lab.append(f"{e[:3]} {r.delta.iloc[0]:+.3g}{star}")
        ax.set_xticks(range(len(EPOCH_ORDER)), EPOCH_ORDER, fontsize=7)
        ax.set_title(SUMMARY_COLS[c] + ("\n" + "; ".join(lab) if lab else "\n"), fontsize=7.5)
        ax.set_ylabel(SUMMARY_COLS[c], fontsize=6.5)
        ax.tick_params(labelsize=7)
    axs.flat[0].legend(fontsize=7, frameon=False)
    note = ("trials: EVERY trial at the far spouts, cue + 70 .. 3000 ms, measured jaw frames only (LP r4, cleaning v35); "
            "FAR_DIR = (far ipsi - far contra) / 2 of the per-trial jaw sideways displacement = jaw movement TOWARD the "
            "target side; per-animal epoch mean +- SEM over sessions; title: paired change vs pre (animal-weighted "
            "bootstrap, * = 95% CI excludes 0; >= 2 animals)")
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 150)), fontsize=9)
    from wfield_local.orofacial_cohort import _tidy, save_png_svg
    _tidy(fig)
    fig.tight_layout()
    save_png_svg(fig, path)
    plt.close(fig)


def fig_timecourse(S: pd.DataFrame, path: Path, title: str) -> None:
    from wfield_local import config
    col = config.animal_color()
    plt = _plt()
    cs = ("lr_wide_far_dir", "lr_wide_reached_far_dir", "angle_far_dir", "lr_wide_far_bias")
    fig, axs = plt.subplots(1, len(cs), figsize=(4.2 * len(cs), 3.2), squeeze=False)
    for ax, c in zip(axs[0], cs):
        for a, g in S.sort_values("day").groupby("animal"):
            ax.plot(g.day, g[c], "o-", ms=3, lw=1, color=col.get(a, "k"), label=a, zorder=3)
            if f"{c}_lo" in g:
                ax.fill_between(g.day, g[f"{c}_lo"], g[f"{c}_hi"], color=col.get(a, "k"), alpha=0.12, lw=0)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0.5, color="0.6", ls="--", lw=0.8, zorder=0)
        ax.set_title(SUMMARY_COLS[c], fontsize=8)
        ax.set_ylabel(SUMMARY_COLS[c], fontsize=7)
        ax.set_xlabel("days since stroke", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axs[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(title + "\none point per session (band = trial-bootstrap 95% CI where available); dashed = lesion",
                 fontsize=9)
    from wfield_local.orofacial_cohort import _tidy, save_png_svg
    _tidy(fig)
    fig.tight_layout()
    save_png_svg(fig, path)
    plt.close(fig)


def fig_by_position(T: pd.DataFrame, path: Path, title: str) -> None:
    from wfield_local.orofacial_cohort import _pos_color, pos_disp
    animals = sorted(T.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(2, len(animals), figsize=(3.5 * len(animals), 5.4), squeeze=False)
    x = np.arange(len(POS_ORDER))
    for j, a in enumerate(animals):
        for i, m in enumerate(MEASURES):
            ax = axs[i][j]
            for e in EPOCH_ORDER:
                g = T[(T.animal == a) & (T.epoch == e)]
                if not len(g):
                    continue
                per = g.groupby(["date", "pos"])[m].median().unstack()      # session medians, then epoch mean +- SEM
                mu, se = per.mean().reindex(POS_ORDER), per.agg(_sem).reindex(POS_ORDER)
                ax.errorbar(x, mu.to_numpy(), yerr=se.to_numpy(), fmt="o-", color=EPOCH_LINE[e], ms=3.5, lw=1.2,
                            capsize=2, elinewidth=0.8, label=f"{e} ({len(per)})", zorder=3)
            ax.axhline(0, color="0.85", lw=0.8, zorder=0)
            ax.axvline(2.5, color="0.85", lw=0.8, zorder=0)
            ax.set_xticks(x, [pos_disp(p).replace(" ", "\n") for p in POS_ORDER], fontsize=6)
            for tk, p in zip(ax.get_xticklabels(), POS_ORDER):
                tk.set_color(_pos_color(p, "L"))
            ax.set_ylabel(MEASURES[m], fontsize=7)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(a, fontsize=9)
        axs[0][j].legend(fontsize=6, frameon=False)
    note = ("trials: EVERY trial, cue + 70 .. 3000 ms, measured jaw frames; per-session median per position, epoch mean "
            "+- SEM over sessions. Close / mid spouts partly hide the chin (the model then marks its visible edge) -- "
            "read the FAR positions; the readout (FAR_DIR) uses far ipsi vs far contra only")
    fig.suptitle(title + "\n" + "\n".join(textwrap.wrap(note, 140)), fontsize=9)
    from wfield_local.orofacial_cohort import _tidy, save_png_svg
    _tidy(fig)
    fig.tight_layout()
    save_png_svg(fig, path)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs, writeguard
    from wfield_local import orofacial_outputs as oo
    from wfield_local.orofacial_cohort import _repo_on_path
    from wfield_local.paths import PathResolver
    _repo_on_path()
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--run", default=RUN)
    ap_.add_argument("--jobs", type=int, default=None)
    a = ap_.parse_args(argv)
    rv = PathResolver()
    base = oo.root(rv) / a.run / "cohort"
    tables, figs = base / "tables" / "jaw_lateral", base / "figures" / "jaw_lateral"
    writeguard.assert_writable(base)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    lat = {k: str(v.get("stroke_laterality", "L")) for k, v in config.animals().items()}
    items = []
    for s in ak.curated_sessions(order="sorted"):
        an = config.animal_of(s["label"])
        date = "2026" + s["label"].split("_")[-1]
        sd = oo.session_dir(rv, a.run, an, date)
        if (sd / "manifest.json").exists():
            items.append((sd, an, date, epochs.epoch_of(s["label"]), epochs.days_since_stroke(s["label"]), lat[an]))
    res, fail = ak.fan_sessions(items, session_trials, jobs=a.jobs, key=lambda it: (it[1], it[2]), label="session")
    if fail:
        raise RuntimeError(f"{len(fail)} session(s) failed -- refusing to write a partial table: {fail}")
    T = pd.DataFrame([r for _, v in res for r in v])
    S = summarise(T)
    A, D = epoch_tables(S)
    pd.set_option("display.width", 250)
    print(f"{S.shape[0]} sessions, {len(T)} trials")
    print("\nPER-ANIMAL epoch means (rule 8 -- read before the deltas):")
    print(A.set_index(["animal", "epoch"]).round(2).to_string())
    print("\nchange vs pre (animal-weighted; CI only with >= 2 animals):")
    print(D.round(3).to_string(index=False))
    T.to_csv(tables / "trials.csv", index=False, float_format="%.3f")
    S.to_csv(tables / "sessions.csv", index=False, float_format="%.3f")
    A.to_csv(tables / "epoch_by_animal.csv", index=False)
    D.to_csv(tables / "epoch_deltas.csv", index=False)
    animals = ", ".join(f"{x} {int((S.animal == x).sum())}" for x in sorted(S.animal.unique()))
    t = f"Jaw lateral readout (whole sessions) -- {a.run} -- sessions: {animals}"
    fig_epochs(S, D, figs / "epochs.png", t)
    fig_timecourse(S, figs / "timecourse.png", t)
    fig_by_position(T, figs / "by_position.png", t)
    (tables / "provenance.json").write_text(json.dumps({"run": a.run, "n_sessions": int(len(S)),
                                                        "code_commit": oo.code_commit()}, indent=1), encoding="utf-8")
    print(f"\ntables -> {tables}\nfigures -> {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
