"""S12F caption review (Priya 11-23, item 31): the six-position within- vs between-session course on far-contralesional
FIRST-LICK ANGLE (change from pre-stroke), the measure of the two-tone S8E, instead of launch error.

Same trials, session thirds, mixed models and pre-stroke-thirds display as the current S12F (cam3-on Variant A run of
`scripts/learned_compensation.py`, T1; display as `s12f_pre_thirds.py`). Only the per-trial readout changes:
  * launch error e_L (deg, first 12 ms of protrusion vs pre contact licks at the spout)  ->
  * first-lick 3-D angle, THE aim definition of Fig 4D (`iti_recovery_measure.aim_licks`, built by
    `aim_tables_from_kinematics.py`): atan2(side_ml, hypot(out_ap, down_dv)) at the extension peak of the trial's first
    post-cue lick (lick_idx 0) in the four-camera fused kinematics (`kinematics_fused_postcue/licks.csv.gz`); + =
    contralesional (all four lesions are left, side_ml + = right).
The angle is joined to the T1 trial table (`trials_t1.csv.gz`: engaged = `canonical`, block structure, far_contra) on
(label, trial_id); the position of the joined lick is checked against the trial's. Everything downstream is the
pipeline's own code (`learned_compensation.t1_sessions / t1_pairs / t1_mixed / _change_row`), with the angle in the
`eL1` column. The launch-error version is recomputed through the same code as a reproduction check.

Added (not in the pipeline): a session-to-session mixed model on consecutive-day pairs, whole session d vs whole session
d+1 (y ~ second * post, random intercept per animal + pair), the trial-level analog of the pipeline's day step S.

    python s12f_firstangle.py <manuscript_code/figures> <learned_compensation dir> <kinematics licks.csv.gz>
                              <aim_by_spout_sessions.csv> <data_out_dir> <fig_out_stem>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_PRE = 3
AN = ["PS92", "PS93", "PS94", "PS95"]
NUM = dict(zip(AN, ["#9", "#10", "#11", "#12"]))


def angle_table(kin: str) -> pd.DataFrame:
    from wfield_local import orofacial_cohort as oc
    K = pd.read_csv(kin, usecols=["session", "trial_id", "lick_idx", "position", "context", "side_ml", "out_ap",
                                  "down_dv"])
    K = K[(K.context == "post_cue") & (K.lick_idx == 0)].dropna(subset=["side_ml", "out_ap", "down_dv"])
    K["label"] = K.session.str.split("_").str[0] + "_" + K.session.str.split("_").str[1].str[-4:]
    K["pos_k"] = [oc.pos_key(p, "L") for p in K.position.astype(str)]
    K["ang1"] = np.degrees(np.arctan2(K.side_ml, np.hypot(K.out_ap, K.down_dv)))
    dup = K.duplicated(["label", "trial_id"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate first licks")
    return K[["label", "trial_id", "pos_k", "ang1"]]


def boot_table(lc, S: pd.DataFrame, P1: pd.DataFrame, P2: pd.DataFrame) -> pd.DataFrame:
    """The pipeline's T1 boot_delta loop (cmd_summarise), same metric order and rng, so CIs reproduce."""
    rng = np.random.default_rng(lc.SEED)
    B1 = []
    pre_s, win_s = S[S.period == "pre"], S[S.period == "window"]
    for m in ("W_eL", "W_hit", "b_eL", "W_o5", "W_fc_minus_o5", "W_eE", "W_eL3", "W_eLnf", "M_eL", "M_hit"):
        if m not in S:
            B1.append({"metric": m, "set": "window d1-14 vs pre"})
            continue
        B1.append({"metric": m, "set": "window d1-14 vs pre",
                   **lc._change_row(lc._by_animal(win_s, m), lc._by_animal(pre_s, m), rng)})
    for nm, PP in (("gap1", P1), ("gap<=2", P2)):
        pre_p, win_p = PP[PP.period == "pre"], PP[PP.period == "window"]
        for m in ("O_eL", "O_hit", "S_eL", "S_hit", "O_eE", "O_eL3"):
            if m not in PP:
                continue
            B1.append({"metric": m, "set": f"window pairs vs pre pairs ({nm})",
                       **lc._change_row(lc._by_animal(win_p, m), lc._by_animal(pre_p, m), rng)})
    return pd.DataFrame(B1)


def s2s_mixed(lc, X: pd.DataFrame, P: pd.DataFrame) -> list[dict]:
    rows = []
    for k, pr in enumerate(P.itertuples()):
        for lab, sec in ((pr.label_d, 0), (pr.label_d1, 1)):
            v = X[X.label == lab].eL1.dropna().to_numpy(float)
            if len(v) < lc.MIN_SESS_T1:
                continue
            rows += [{"animal": pr.animal, "pair": k, "second": sec, "post": int(pr.period == "window"), "y": y}
                     for y in v]
    Q = pd.DataFrame(rows)
    return [{"test": "T1 session-to-session (whole-session pairs, added)", **r}
            for r in lc._mixed("y ~ second * post", Q, "pair", ["second", "second:post"])]


def run_models(lc, T: pd.DataFrame, tag: str) -> tuple:
    S = lc.t1_sessions(T)
    P1, P2 = lc.t1_pairs(S, 1), lc.t1_pairs(S, 2)
    M = lc.t1_mixed(T, S)
    M = M[M.readout.str.startswith("e_L")].copy()
    per = S.set_index("label").period
    X = T[T.canonical & (T.pos == "far_contra")].copy()
    X["period"] = X.label.map(per)
    X = X[X.period.isin(["pre", "window"])].sort_values(["label", "trial_id"])
    M = pd.concat([M, pd.DataFrame(s2s_mixed(lc, X, P1))], ignore_index=True)
    M.insert(0, "readout_set", tag)
    B = boot_table(lc, S, P1, P2)
    B.insert(0, "readout_set", tag)
    return S, P1, M, B


def render(figdir: str, S: pd.DataFrame, out_stem: str) -> pd.DataFrame:
    sys.path.insert(0, figdir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import vls_style as St
    St.apply()
    plt.rcParams.update({'font.size': 6.8, 'axes.labelsize': 6.8, 'axes.titlesize': 6.8, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'legend.fontsize': 6.3, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2,
                         'ytick.major.size': 2})
    C = St.EP
    cols = ["early_eL", "mid_eL", "late_eL"]
    fig, axs = plt.subplots(1, 4, figsize=(5.6, 1.35), sharey=False)
    base = []
    for ax, a in zip(axs, AN):
        d = S[S.animal == a].sort_values(["day", "label"])
        pre_all = d[d.epoch == "pre"].dropna(subset=cols)
        v = pre_all[cols].to_numpy(float).ravel()
        b0 = np.nanmean(v)
        base.append({"animal": a, "num": NUM[a], "pre_mean_deg": b0, "sd_thirds": np.nanstd(v),
                     "n_pre_sessions": len(pre_all)})
        pre = pre_all.tail(N_PRE)
        post = d[(d.epoch != "pre") & (d.day >= 1) & (d.day <= 14)].dropna(subset=cols)
        seq = pd.concat([pre, post]).reset_index(drop=True)
        prev, ticks, labels = None, [], []
        x0 = 0.0
        for _k, r in seq.iterrows():
            xs = x0 + np.array([-0.25, 0, 0.25])
            ys = r[cols].to_numpy(float) - b0
            col = "0.3" if r.epoch == "pre" else C[r.epoch]
            ax.plot(xs, ys, "-", color=col, lw=0.9, zorder=2)
            ax.plot(xs, ys, "o", ms=1.8, color=col, zorder=3)
            if prev is not None:
                ax.plot([prev[0], xs[0]], [prev[1], ys[0]], "--" if r.day - prev[2] == 1 else ":", color="0.55",
                        lw=0.6, zorder=1)
            prev = (xs[-1], ys[-1], r.day)
            if r.epoch != "pre":
                ticks.append(x0)
                labels.append(str(int(r.day)))
            x0 += 1.0
        ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
        ax.set_xticks(ticks[::2])
        ax.set_xticklabels(labels[::2])
        ax.text((N_PRE - 1) / 2, -0.02, "pre", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.5, color="0.3")
        ax.set_xlim(-0.6, x0 - 0.4)
        ax.set_title(NUM[a], pad=2)
        ax.set_xlabel("post-stroke day", labelpad=1)
    axs[0].set_ylabel("Δ first-lick\nangle (deg)")
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.27, top=0.86, wspace=0.32)
    for x in ("pdf", "svg", "png"):
        fig.savefig(f"{out_stem}.{x}", dpi=300 if x == "png" else None)
    return pd.DataFrame(base)


def main(figdir: str, lcdir: str, kin: str, aim_sessions: str, data_out: str, out_stem: str) -> None:
    for x in ("pdf", "svg", "png"):
        if Path(f"{out_stem}.{x}").exists():
            sys.exit(f"refusing to overwrite {out_stem}.{x}")
    out = Path(data_out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["LC_OUT_DIR"] = lcdir             # read-only here: nothing below writes to it
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import learned_compensation as lc
    T = pd.read_csv(Path(lcdir) / "trials_t1.csv.gz", low_memory=False)
    T["canonical"] = lc._b(T.canonical)
    A = angle_table(kin)
    T = T.merge(A, on=["label", "trial_id"], how="left", validate="one_to_one")
    bad = T.pos_k.notna() & (T.pos_k != T.pos)
    if bad.any():
        raise SystemExit(f"position mismatch on {int(bad.sum())} trials")
    fc = T[T.canonical & (T.pos == "far_contra")]
    cov = pd.DataFrame({"n_engaged_fc": fc.groupby("label").size(), "n_eL": fc.groupby("label").eL1.count(),
                        "n_ang": fc.groupby("label").ang1.count(),
                        "n_both": fc.dropna(subset=["eL1", "ang1"]).groupby("label").size()}).reset_index()
    cov.to_csv(out / "coverage_per_session.csv", index=False)
    print(f"far-contra engaged trials {len(fc)}: e_L {fc.eL1.notna().sum()}, angle {fc.ang1.notna().sum()}, "
          f"both {fc.dropna(subset=['eL1', 'ang1']).shape[0]}")
    print("per-trial r(e_L, angle) far-contra engaged:", round(fc[["eL1", "ang1"]].corr().iloc[0, 1], 3))

    # check vs Fig 4D table (all trials, no engaged gate): per-session far-contra mean first-lick angle minus pre ref
    G = pd.read_csv(aim_sessions)
    G = G[G.pos == "far_contra"].set_index("label")
    allfc = T[T.pos == "far_contra"].groupby("label").ang1.agg(["mean", "size", "count"])
    # aim_licks offset ref = animal's mean over ALL its pre cued licks at the spout (all lick_idx), so compare
    # the between-session differences (ref cancels)
    j = allfc.join(G[["off", "n", "animal", "epoch"]], how="inner")
    j["d_mine"] = j["mean"] - j.groupby("animal")["mean"].transform("mean")
    j["d_fig4d"] = j["off"] - j.groupby("animal")["off"].transform("mean")
    j.to_csv(out / "check_vs_fig4d_sessions.csv")
    print(f"check vs Fig 4D first-lick table ({len(j)} sessions): max |diff of centered session means| "
          f"{np.nanmax(np.abs(j.d_mine - j.d_fig4d)):.3g} deg; n licks equal {(j['count'] == j['n']).mean():.3f}")

    # launch-error reproduction (pipeline code, original eL1)
    _, _, M0, B0 = run_models(lc, T, "launch_error_eL (reproduction)")
    # first-lick angle
    Ta = T.copy()
    Ta["eL1"] = Ta["ang1"]
    Sa, Pa, Ma, Ba = run_models(lc, Ta, "first_lick_angle")
    M = pd.concat([M0, Ma], ignore_index=True)
    B = pd.concat([B0, Ba], ignore_index=True)
    M.to_csv(out / "t1_mixed_launch_vs_angle.csv", index=False)
    B.to_csv(out / "t1_boot_launch_vs_angle.csv", index=False)
    Sa.to_csv(out / "t1_sessions_firstangle.csv", index=False)
    Pa.to_csv(out / "t1_pairs_gap1_firstangle.csv", index=False)
    pd.set_option("display.width", 220)
    print(M[["readout_set", "test", "term", "coef", "lo", "hi", "p", "n", "n_groups", "converged"]].round(3)
          .to_string(index=False))
    keep = B.metric.isin(["W_eL", "O_eL", "S_eL", "M_eL"])
    print(B[keep][["readout_set", "metric", "set", "delta", "lo", "hi", "p", "n_same", "change",
                   *[f"d_{a}" for a in AN]]].round(2).to_string(index=False))
    base = render(figdir, Sa, out_stem)
    base.to_csv(out / "pre_reference_per_animal.csv", index=False)
    print(base.round(2).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:7])
