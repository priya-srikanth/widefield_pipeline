"""The TRIAL-MATCHED pre-vs-pre null for the CD MIGRATION matrix (each spout's trials on each pre spout direction).

Priya, 2026-11-26 ("overlap null - yes"): the 11-25 lick-window migration on the four-camera root reads far-contra
trials onto far-ipsi at +0.46 and onto close-ipsi at +0.37 acutely, 4/4 animals, while far-contra's own projection
fell +0.87 -> +0.24. Those are CHANGES against pre, and like the shared-mode overlap (`cd_overlap_null`) a change has
no scale of its own: a mean over the few acute far-contra trials is noisier than the pre mean it is compared with, and
two sets of PRE sessions differ too. This asks whether the observed change is outside what pre-vs-pre resampling,
matched in trials, produces.

THE CONSTRUCTION is `cd_overlap_null`'s, cell by cell. Per animal, per post epoch E, per draw:

    B   a random set of PRE sessions, then per ROW (trial position i) a random subset of B's row-i trials of the SAME
        SIZE as E's row-i trial count (`n_E[i]`) -- trials are the unit the noise scales with, and the per-trial
        window means (`cd_trajectories.analyse_animal(keep_trials=True)`) let B be trimmed to it exactly
    A   every remaining pre session, all trials (the reference both sides share)
    null       dM_null[i, j] = M_B[i, j] - M_A[i, j]
    observed   dM_obs[i, j]  = M_E[i, j] - M_A[i, j]

Disjoint at SESSION level (a session in both groups would correlate them and shrink the null). B takes sessions in
random order until every row can be covered, but never more than half the pre sessions, so A stays a real reference.
Where B cannot reach a row's target (a large chronic epoch against 7-8 pre sessions) B keeps all its trials and the
achieved fraction is reported: an under-sized B is NOISIER than E, so its null is WIDER -- conservative for the
"outside the null" reading, never flattering.

DIRECTIONS ARE THE FROZEN ONES, loaded (never refit; `WIDEFIELD_FROZEN_LOAD_ONLY=1`). They were fitted on ALL pre
sessions, so B's and A's trials are IN-SAMPLE for the directions and E's are not. Both null groups carry the same
in-sample bias, so it cancels in M_B - M_A but not in M_E - M_A. For an OFF-diagonal cell (i on j, i != j) i's trials
were in j's "rest" group, so in-sample they score slightly LOWER on j than fresh trials would -- the observed
increase is then a little larger than a fresh-sample null would give. Stated, not corrected: refitting a direction per
draw would need a reference the store does not hold.

WHAT IS NOT CLAIMED. The draws resample which pre sessions / trials stand in for E; they are not a permutation of an
exchangeable label and they share sessions. `frac_null_ge` = the fraction of draws whose null change is at least as
large as the observed one IN THE OBSERVED DIRECTION -- a fraction of draws, NOT a p-value. A cell "clears" its null in
an animal when that fraction is <= 0.025 (two-sided 95% band); the cohort reading counts animals (rule 8).

    python -m wfield_local.cd_migration_null --align lick --out <NEW folder>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Draws per (animal, epoch). Sessions are few (7-8 pre), so draws buy resolution, not independent evidence.
N_DRAW = 1000
#: B never takes more than this fraction of the pre sessions (A must remain a reference).
B_MAX_FRAC = 0.5
#: A cell clears its null in one animal when at most this fraction of draws reach the observed change.
CLEAR_FRAC = 0.025
EPOCHS_POST = ("acute", "subacute", "chronic")


def trial_table(res, window_positions=None):
    """``{label: {"epoch": ep, "rows": {i_pos: (n_i, n_cd) array}}}`` and the ordered position list.

    Columns follow `cd_migration.DISPLAY_ORDER` restricted to the fitted positions, rows likewise, so the matrices here
    index exactly like `cd_migration.matrices`.
    """
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

    pos = [p for p in DISPLAY_ORDER if p in res["positions"]] if window_positions is None else list(window_positions)
    st = res.get("session_trials") or {}
    out = {}
    for (lab, ep, cd, tr), v in st.items():
        if cd not in pos or tr not in pos:
            continue
        d = out.setdefault(lab, {"epoch": ep, "cols": {}})
        d["cols"].setdefault(tr, {})[cd] = np.asarray(v, float)
    tab = {}
    for lab, d in out.items():
        rows = {}
        for tr, cols in d["cols"].items():
            if len(cols) != len(pos):
                continue
            A = np.stack([cols[c] for c in pos], axis=1)            # (n_trials, n_cd)
            A = A[np.isfinite(A).all(1)]
            if A.shape[0]:
                rows[tr] = A
        tab[lab] = {"epoch": d["epoch"], "rows": rows}
    return tab, pos


def _mean_matrix(groups, pos):
    """Row-wise trial-pooled mean over ``{tr: [(n, n_cd) arrays]}`` -> (n_pos, n_pos), NaN for an empty row."""
    M = np.full((len(pos), len(pos)), np.nan)
    for i, tr in enumerate(pos):
        parts = [g for g in groups.get(tr, []) if g.shape[0]]
        if parts:
            M[i] = np.concatenate(parts).mean(0)
    return M


def epoch_null(tab, pos, ep, n_draw=N_DRAW, rng=None):
    """Observed and null dM for one epoch over `n_draw` trial-matched splits of the pre sessions, or None."""
    rng = np.random.default_rng(0) if rng is None else rng
    pre = sorted(lab for lab, d in tab.items() if d["epoch"] == "pre")
    post = sorted(lab for lab, d in tab.items() if d["epoch"] == ep)
    if len(pre) < 2 or not post:
        return None
    n = len(pos)
    E = {tr: [tab[lab]["rows"][tr] for lab in post if tr in tab[lab]["rows"]] for tr in pos}
    n_E = np.array([sum(a.shape[0] for a in E[tr]) for tr in pos], float)
    M_E = _mean_matrix(E, pos)
    M_pre_all = _mean_matrix({tr: [tab[lab]["rows"][tr] for lab in pre if tr in tab[lab]["rows"]] for tr in pos},
                             pos)
    k_max = max(1, int(np.floor(B_MAX_FRAC * len(pre))))
    obs, null, nB, kB = [], [], [], []
    for _ in range(n_draw):
        order = [pre[i] for i in rng.permutation(len(pre))]
        B, have = [], np.zeros(n)
        for lab in order:
            if len(B) >= k_max or (have >= n_E).all():
                break
            B.append(lab)
            have += [tab[lab]["rows"][tr].shape[0] if tr in tab[lab]["rows"] else 0 for tr in pos]
        A = [lab for lab in order if lab not in set(B)]
        gB = {}
        for i, tr in enumerate(pos):
            pool = [tab[lab]["rows"][tr] for lab in B if tr in tab[lab]["rows"]]
            if not pool or n_E[i] == 0:
                continue
            P = np.concatenate(pool)
            take = int(min(n_E[i], P.shape[0]))
            gB[tr] = [P[rng.choice(P.shape[0], take, replace=False)]]
        M_B = _mean_matrix(gB, pos)
        M_A = _mean_matrix({tr: [tab[lab]["rows"][tr] for lab in A if tr in tab[lab]["rows"]] for tr in pos}, pos)
        obs.append(M_E - M_A)
        null.append(M_B - M_A)
        nB.append([gB[tr][0].shape[0] if tr in gB else 0 for tr in pos])
        kB.append(len(B))
    obs, null = np.stack(obs), np.stack(null)
    d = obs - null
    sgn = np.sign(np.nanmedian(obs, 0))
    with np.errstate(invalid="ignore"):
        frac_null_ge = np.nanmean((sgn[None] * d) <= 0, axis=0)
    return {"obs": obs, "null": null, "n_E": n_E, "n_B": np.asarray(nB, float), "k_B": np.asarray(kB, float),
            "k_E": len(post), "n_pre_sessions": len(pre), "M_E": M_E, "M_pre_all": M_pre_all,
            "dM_vs_all_pre": M_E - M_pre_all, "frac_null_ge": frac_null_ge,
            "obs_med": np.nanmedian(obs, 0), "null_med": np.nanmedian(null, 0),
            "null_lo": np.nanpercentile(null, 2.5, axis=0), "null_hi": np.nanpercentile(null, 97.5, axis=0),
            "match_frac": np.nanmedian(np.asarray(nB, float), 0) / np.where(n_E > 0, n_E, np.nan)}


def run_animal(animal, align="lick", gate="lick", n_draw=N_DRAW, seed=0, verbose=True):
    """Every post epoch's migration null for one animal, from `analyse_animal(keep_trials=True)` (load-only)."""
    from wfield_local import cd_trajectories as cdt

    res = cdt.analyse_animal(animal, align, orth=True, gate=gate, reference="contrast", method="dom",
                             mask_occluded=True, verbose=False, keep_trials=True)
    if res.get("skipped"):
        raise RuntimeError(f"{animal} {align}: {res['skipped']}")
    tab, pos = trial_table(res)
    from scripts.cd_migration import matrices          # the published reduction, for the reproduction check
    mats = matrices(res)
    out = {"animal": animal, "align": align, "gate": gate, "positions": pos, "n_draw": int(n_draw),
           "win_s": float(res["win_s"]), "basis_id": res["basis_id"], "epochs": {}, "repro": {}}
    rng = np.random.default_rng(seed)
    for ep in EPOCHS_POST:
        r = epoch_null(tab, pos, ep, n_draw=n_draw, rng=rng)
        if r is None:
            continue
        out["epochs"][ep] = r
    # REPRODUCTION: the trial-pooled M from per-trial window means must equal the dump reduction (`matrices`).
    for ep in ("pre", *EPOCHS_POST):
        if ep not in mats:
            continue
        labs = [lab for lab, d in tab.items() if d["epoch"] == ep]
        M = _mean_matrix({tr: [tab[lab]["rows"][tr] for lab in labs if tr in tab[lab]["rows"]] for tr in pos}, pos)
        out["repro"][ep] = float(np.nanmax(np.abs(M - mats[ep][0])))
    if verbose:
        print(report(out), flush=True)
    return out


def _names(pos):
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    return [POSITION_NAMES.get(p, str(p)) for p in pos]


def report(out, cells=None):
    """Per-animal table of every cell (rule 8): observed dM (vs A), null median [2.5, 97.5], fraction of draws."""
    pos = out["positions"]
    nm = _names(pos)
    L = [f"{out['animal']} {out['align']}  draws={out['n_draw']}  reproduction max|dM| vs dump: "
         + ", ".join(f"{k} {v:.1e}" for k, v in out["repro"].items())]
    for ep, r in out["epochs"].items():
        L.append(f"  {ep}: E = {r['k_E']} sessions, trials per row {r['n_E'].astype(int).tolist()}; "
                 f"B = median {np.median(r['k_B']):.0f} of {r['n_pre_sessions']} pre sessions, achieved "
                 f"{np.round(r['match_frac'], 2).tolist()} of the row targets")
        for i in range(len(pos)):
            for j in range(len(pos)):
                if cells is not None and (nm[i], nm[j]) not in cells:
                    continue
                L.append(f"    {nm[i]:13s} on {nm[j]:13s} dM(all pre) {r['dM_vs_all_pre'][i, j]:+.2f}  "
                         f"dM(vs A) {r['obs_med'][i, j]:+.2f}  null {r['null_med'][i, j]:+.2f} "
                         f"[{r['null_lo'][i, j]:+.2f},{r['null_hi'][i, j]:+.2f}]  "
                         f"frac draws null>=obs {r['frac_null_ge'][i, j]:.3f}")
    return "\n".join(L)


def save(out, out_dir):
    from wfield_local import results_store as rs

    payload = {k: v for k, v in out.items() if k != "epochs"}
    payload["epochs"] = {ep: {k: v for k, v in r.items()} for ep, r in out["epochs"].items()}
    return rs.save(out_dir, "cd_migration_null", f"{out['animal']}_{out['align']}_{out['gate']}", payload,
                   meta={"n_draw": out["n_draw"], "animal": out["animal"], "align": out["align"], "gate": out["gate"],
                         "b_max_frac": B_MAX_FRAC, "clear_frac": CLEAR_FRAC})


#: Display labels in the canonical spout order (close ipsi, mid, contra, then far); ipsi = L, contra = R here.
LABEL = {"close_L": "close ipsi", "close_center": "close mid", "close_R": "close contra",
         "far_L": "far ipsi", "far_center": "far mid", "far_R": "far contra"}


def load_all(out_dir, align="lick", gate="lick", animals=("PS92", "PS93", "PS94", "PS95")):
    from wfield_local import results_store as rs

    got = {}
    for a in animals:
        body = rs.load(out_dir, "cd_migration_null", f"{a}_{align}_{gate}")
        if body and body.get("epochs"):
            got[a] = body
    return got


def cohort_table(got):
    """``(per_animal, pooled)`` DataFrames: every cell per animal (rule 8), then the pooled reading with
    ``k_clear_same_sign`` = animals whose change clears their own null in the pooled direction."""
    import pandas as pd

    recs = []
    for a, b in sorted(got.items()):
        nm = _names(b["positions"])
        for ep, r in b["epochs"].items():
            for i in range(len(nm)):
                for j in range(len(nm)):
                    fr = float(np.asarray(r["frac_null_ge"])[i, j])
                    recs.append({"epoch": ep, "trials": nm[i], "on_direction": nm[j], "animal": a,
                                 "dM_vs_all_pre_1125": float(np.asarray(r["dM_vs_all_pre"])[i, j]),
                                 "dM_vs_A_median": float(np.asarray(r["obs_med"])[i, j]),
                                 "null_median": float(np.asarray(r["null_med"])[i, j]),
                                 "null_lo95": float(np.asarray(r["null_lo"])[i, j]),
                                 "null_hi95": float(np.asarray(r["null_hi"])[i, j]),
                                 "frac_draws_null_ge_obs": fr, "clears": bool(fr <= CLEAR_FRAC),
                                 "n_trials_E_row": float(np.asarray(r["n_E"])[i]),
                                 "B_match_frac_row": float(np.asarray(r["match_frac"])[i]),
                                 "n_E_sessions": int(r["k_E"]), "n_pre_sessions": int(r["n_pre_sessions"])})
    df = pd.DataFrame(recs)
    if df.empty:
        return df, df
    keys = ["epoch", "trials", "on_direction"]
    pooled = df.groupby(keys, sort=False).agg(
        dM_vs_A_mean=("dM_vs_A_median", "mean"), dM_vs_all_pre_mean=("dM_vs_all_pre_1125", "mean"),
        n_animals=("animal", "count")).reset_index()
    ks = []
    for _, row in pooled.iterrows():
        sub = df[(df["epoch"] == row["epoch"]) & (df["trials"] == row["trials"])
                 & (df["on_direction"] == row["on_direction"])]
        sgn = np.sign(row["dM_vs_A_mean"])
        ks.append(int(((np.sign(sub["dM_vs_A_median"]) == sgn) & sub["clears"]).sum()))
    pooled["k_clear_same_sign"] = ks
    return df, pooled


def figure(got, path):
    """Top: pooled dM (vs A) per epoch, each cell labelled k/4 = animals whose change clears their own null in the
    pooled direction (boxed at >= 3/4). Bottom: the far-contra row per animal (open marker: shape = animal, color =
    epoch) with that animal's null 95% band as a gray bar beside it. No stars: the fractions are not p-values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import cat_points as cp
    from wfield_local.figure_layout import save_png_svg

    df, pooled = cohort_table(got)
    pos_names = _names(next(iter(got.values()))["positions"])
    eps = [e for e in EPOCHS_POST if (pooled["epoch"] == e).any()]
    fig, axes = plt.subplots(2, len(eps), figsize=(4.3 * len(eps), 8.6), squeeze=False,
                             gridspec_kw={"height_ratios": [1.0, 0.9]}, constrained_layout=True)
    lim = float(np.nanmax(np.abs(pooled["dM_vs_A_mean"])))
    for c, ep in enumerate(eps):
        ax = axes[0][c]
        P = pooled[pooled["epoch"] == ep]
        D = np.full((len(pos_names), len(pos_names)), np.nan)
        K = np.zeros(D.shape, int)
        for _, row in P.iterrows():
            i, j = pos_names.index(row["trials"]), pos_names.index(row["on_direction"])
            D[i, j], K[i, j] = row["dM_vs_A_mean"], row["k_clear_same_sign"]
        im = ax.imshow(D, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if np.isfinite(D[i, j]):
                    ax.text(j, i, f"{D[i, j]:+.2f}\n{K[i, j]}/4", ha="center", va="center", fontsize=6.5,
                            fontweight="bold" if K[i, j] >= 3 else "normal")
                    if K[i, j] >= 3:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.6, ec="k"))
        ax.set_xticks(range(len(pos_names)))
        ax.set_xticklabels([LABEL.get(n, n) for n in pos_names], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pos_names)))
        ax.set_yticklabels([LABEL.get(n, n) for n in pos_names], fontsize=7)
        ax.set_title(f"{ep}: change vs pre (A), mean of 4 animals", fontsize=9, color=cp.epoch_color(ep))
        if c == 0:
            ax.set_ylabel("trials at spout", fontsize=8)
        ax.set_xlabel("projected onto the frozen pre direction of", fontsize=7)
        if c == len(eps) - 1:
            fig.colorbar(im, ax=ax, shrink=0.75, label="change (pole units)")

        ax = axes[1][c]
        R = df[(df["epoch"] == ep) & (df["trials"] == "far_R")]
        animals = sorted(R["animal"].unique())
        offs = cp.epoch_offsets(len(animals), 0.17)
        ax.axhline(0, color="0.6", lw=0.8, ls=":")
        for k, a in enumerate(animals):
            for _, row in R[R["animal"] == a].iterrows():
                x = pos_names.index(row["on_direction"]) + offs[k]
                ax.plot([x + 0.06] * 2, [row["null_lo95"], row["null_hi95"]], color="0.6", lw=2.4,
                        solid_capstyle="butt", zorder=1)
                ax.plot(x, row["dM_vs_A_median"], ls="none", marker=cp.animal_marker(a), mfc="none",
                        mec=cp.epoch_color(ep), mew=1.4, ms=6, zorder=3)
        for j, n in enumerate(pos_names):
            sub = P[(P["trials"] == "far_R") & (P["on_direction"] == n)]
            if len(sub):
                ax.text(j, 1.0, f"{int(sub['k_clear_same_sign'].iloc[0])}/4", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(pos_names)))
        ax.set_xticklabels([LABEL.get(n, n) for n in pos_names], rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.6, len(pos_names) - 0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"far-contra trials, {ep}", fontsize=9, pad=14)
        if c == 0:
            ax.set_ylabel("change vs pre (A); gray = that animal's\ntrial-matched pre-vs-pre null, 95%", fontsize=7.5)
            for a in animals:
                ax.plot([], [], ls="none", marker=cp.animal_marker(a), mfc="none", mec="0.35", mew=1.2, ms=6,
                        label=f"{a} (one value per animal; color = epoch)")
            ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.suptitle("Lick-window migration vs its trial-matched pre-vs-pre null (four-camera root)\n"
                 "k/4 = animals whose change clears their own null (fraction of draws <= 0.025) in the pooled "
                 "direction; boxed = 3/4 or more. Fractions of draws, not p-values.", fontsize=8.5)
    save_png_svg(fig, Path(path), dpi=160)
    plt.close(fig)
    return path


def _one(item):
    from wfield_local import cd_migration_null as m

    out = m.run_animal(item["animal"], item["align"], item["gate"], n_draw=item["n_draw"], seed=item["seed"],
                       verbose=False)
    m.save(out, item["out"])
    return m.report(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--align", default="lick", choices=("precue", "cue", "lick"))
    ap.add_argument("--gate", default="lick")
    ap.add_argument("--draws", type=int, default=N_DRAW)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", required=True, help="a NEW folder (rule 11; there is no default)")
    ap.add_argument("--figure-only", action="store_true",
                    help="read the persisted draws under --out and write the figure + tables there")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        got = load_all(out, args.align, args.gate)
        df, pooled = cohort_table(got)
        df.to_csv(out / f"cd_migration_null_{args.align}_per_animal.csv", index=False)
        pooled.to_csv(out / f"cd_migration_null_{args.align}_pooled.csv", index=False)
        print(f"-> {figure(got, out / f'cd_migration_null_{args.align}.png')}")
        return 0
    items = [{"animal": a, "align": args.align, "gate": args.gate, "n_draw": args.draws, "seed": args.seed,
              "out": str(out)} for a in (args.animal or ["PS92", "PS93", "PS94", "PS95"])]
    from wfield_local import analysis_kit as ak

    res, fail = ak.fan_sessions(items, _one, jobs=args.jobs, key=lambda it: it["animal"], label="animal")
    for _it, val in res:
        if val:
            print(val, flush=True)
    for f in fail:
        print(f"!! {f}", flush=True)
    print(f"-> {out}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
