"""Do post-stroke licks cued to one spout but EXECUTED in another direction look, in cortex, like pre-stroke licks to
the same TARGET or like pre-stroke licks that went the same WAY? (`wfield_local/lick_templates.py`)

    python -m scripts.lick_template_match --pre PS93:20260814 --post PS93:20260821 PS93:20260908
    python -m scripts.lick_template_match --pre PS93:20260814:full PS93:20260812:full --post PS93:20260821:full \\
        --cued far_R far_L --baseline pre_cue

Priya, 2026-10-05: "I'm interested especially in post-stroke licks i.e. those licks that were triggered with far R
spout but that were executed leftward or centrally"; then "please note this finding and refine the code ... we will
want to expand on this when we have more complete dlc results". First result in DECISIONS 2026-10-05.

Per run, for RAW and MOVEMENT-REMOVED activity (cross-fitted residual of a within-session movement model: DAQ
contacts, DLC tongue onsets, continuous tongue / jaw):
  * area signals per session (64 Allen regions, common across sessions; robust-clipped);
  * every executed lick's pattern (tongue onset 0 .. +0.6 s, baseline-subtracted -- `--baseline pre_onset` (the
    -0.2 .. 0 s before the onset; falls inside the previous lick in fast bouts) or `pre_cue` (-0.5 .. 0 s before the
    trial's cue));
  * pre-stroke templates pooled over every `--pre` session: per target position, and per post lick a matched-angle
    template (pre licks within +-angle_tol of its executed angle, any position);
  * per post lick cued to each `--cued` position: r with the target template, r with the matched-angle template,
    executed-angle class vs the pre contact-lick direction at that position; paired statistics of
    (executed - target) per class (mean, bootstrap 95 % CI, Wilcoxon), descriptive only; the pre-stroke own-template accuracy as scale;
  * THE TEST (`_readout_test.csv`): pre templates per executed-angle bin -> each post lick's cortical angle readout
    -> r(executed angle, readout) within cued position, null = angles permuted within position (calibrated), pooled
    and per position with BH q.
  Baseline fixed IN ADVANCE to pre_cue (config default) after the first result held with pre_onset only.
Settings: configs/defaults.yaml `lick_template_match`. Outputs (session_poses/): lick_template_match[_<tag>].csv
(per lick), ..._stats.csv (per session x position x class x version) and a scatter figure per cued position.
Licks are DLC-detected (executed = reach >= movement_encoding.min_reach_px); DAQ contact is a label only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import lick_templates as LT
from wfield_local import movement_encoding as ME
from wfield_local import movement_inputs as MI


def load_session(spec: str, rv, pm: dict, pt: dict, movement_removed: bool) -> dict:
    """Executed DLC licks of one session with their patterns in the common area space."""
    import scripts.movement_encoding_session as MS
    animal, date = spec.split(":")[:2]
    label = f"{animal}_{date[4:]}"
    Ydff, reg, ft = MS.imaging(label)
    A, labs = LT.area_signals(Ydff, reg, float(pt["clip_mad"]))
    P = MS.session_pieces(animal, date, rv, ft, float(pm["min_reach_px"]), spec=spec)
    mask = P["mask"]
    if movement_removed:
        L = P["licks"]
        mov = ME.select_rows(ME.build_design(MI.build_inputs(
            ft, events={"contact": P["contact_s"], "tongue_onset": L.on_s.to_numpy()}, video_signals=P["sig"],
            video_t_s=P["vt"], trial_starts_s=P["trial_starts_s"])), mask)
        folds = ME.trial_block_folds(ft[mask], P["trial_starts_s"], int(pm["n_folds"]))
        a_mov = ME.choose_alphas(ME.standardise(mov.X)[0], A[mask], mov.group, folds, grid=tuple(pm["alpha_grid"]))
        R = np.full_like(A, np.nan)
        R[mask] = ME.cv_residual(mov, A[mask], folds, a_mov)
        A = R
    L = P["licks"][P["licks"].reach_ok].reset_index(drop=True)
    V, ok = LT.lick_vectors(A, ft, L.on_s.to_numpy(), win_s=tuple(pt["win_s"]), base_s=tuple(pt["base_s"]),
                            baseline=pt["baseline"], cue_s=L.cue_s.to_numpy(), precue_base_s=tuple(pt["precue_base_s"]))
    return {"label": label, "licks": L[ok].reset_index(drop=True), "V": V[ok], "labs": labs}


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", nargs="+", required=True, help="pre-stroke session(s), pooled for the templates")
    ap.add_argument("--post", nargs="+", required=True)
    ap.add_argument("--cued", nargs="+", default=["far_R"], help="cued positions to analyse ('all' = every one)")
    ap.add_argument("--baseline", choices=["pre_onset", "pre_cue"], default=None)
    ap.add_argument("--tag", default="", help="suffix for the output names")
    a = ap.parse_args(argv)
    rv = PathResolver()
    pm, pt = MI.params(), LT.params({"baseline": a.baseline} if a.baseline else None)
    cued = LT.POS_ORDER if a.cued == ["all"] else a.cued
    out_dir = Path(rv.root("microscope")) / "DeepLabCut" / "Widefield" / "session_poses"
    sfx = f"_{a.tag}" if a.tag else ""
    per_lick, stats_rows = [], []
    for version in ("raw", "movement_removed"):
        pres = [load_session(s, rv, pm, pt, version == "movement_removed") for s in a.pre]
        if any(len(p["labs"]) != len(pres[0]["labs"]) or not np.array_equal(p["labs"], pres[0]["labs"]) for p in pres):
            raise SystemExit("pre sessions do not share the same region set")
        Lp = pd.concat([p["licks"] for p in pres], ignore_index=True)
        Vp = np.concatenate([p["V"] for p in pres])
        acc = LT.own_template_accuracy(Vp, Lp.position.to_numpy())
        centres, Tang, _edges = LT.angle_bin_templates(Lp, Vp, int(pt["n_angle_bins"]))
        print(f"\n######## {version} ({pt['baseline']} baseline): pre {', '.join(p['label'] for p in pres)} -- "
              f"{len(Lp)} executed licks; own-position template accuracy {acc:.0%} (chance 17%)")
        for spec in a.post:
            post = load_session(spec, rv, pm, pt, version == "movement_removed")
            if not np.array_equal(post["labs"], pres[0]["labs"]):
                print(f"  {post['label']}: region set differs from the pre sessions -- skipped")
                continue
            for pos in cued:
                m = (post["licks"].position == pos).to_numpy()
                if not m.any():
                    continue
                D = LT.compare(Lp, Vp, post["licks"][m].reset_index(drop=True), post["V"][m],
                               angle_tol_deg=float(pt["angle_tol_deg"]), class_deg=float(pt["class_deg"]),
                               min_matched=int(pt["min_matched"]))
                D.insert(0, "version", version)
                D.insert(1, "session", post["label"])
                per_lick.append(D)
                print(f"  == {post['label']} {pos}: {len(D)} executed licks, classes {D.cls.value_counts().to_dict()}")
                D["readout_angle"] = LT.readout_angle(post["V"][m], centres, Tang)
                for cls, g in D.groupby("cls"):
                    st = LT.paired_stats(g.r_executed - g.r_target, n_boot=int(pt["n_boot"]))
                    stats_rows.append({"version": version, "baseline": pt["baseline"], "session": post["label"],
                                       "position": pos, "cls": cls, "r_target": g.r_target.mean(),
                                       "r_executed": g.r_executed.mean(), "matched_mostly":
                                       g.matched_mostly.mode().iloc[0] if g.matched_mostly.any() else "",
                                       "own_template_acc_pre": acc, **st})
                    print(f"     {cls:20s} n={st['n']:3d}  executed - target {st['mean']:+.3f} "
                          f"[{st['ci_lo']:+.3f}, {st['ci_hi']:+.3f}] (descriptive)  readout {g.readout_angle.mean():+.1f} deg "
                          f"vs executed {g.angle.mean():+.1f}  (r target "
                          f"{g.r_target.mean():+.3f}, r executed {g.r_executed.mean():+.3f}, matched mostly "
                          f"{stats_rows[-1]['matched_mostly']})")
    D = pd.concat(per_lick, ignore_index=True) if per_lick else pd.DataFrame()
    S = pd.DataFrame(stats_rows)
    # THE TEST: within cued position, does the cortical angle readout follow the executed angle? (lick-level null)
    T_rows = []
    if len(D):
        for (version, sess), g in D.groupby(["version", "session"]):
            res = LT.within_position_angle_test(g.position, g.angle, g.readout_angle, n_perm=int(pt["n_perm"]))
            T_rows.append({"version": version, "session": sess, "position": "ALL (within-position)", **res})
            for pos, gp in g.groupby("position"):
                if len(gp) >= 8:
                    T_rows.append({"version": version, "session": sess, "position": pos,
                                   **LT.within_position_angle_test(gp.position, gp.angle, gp.readout_angle,
                                                                   n_perm=int(pt["n_perm"]))})
        T = pd.DataFrame(T_rows)
        T["q_bh"] = np.nan
        for _, ix in T[T.position != "ALL (within-position)"].groupby(["version", "session"]).groups.items():
            T.loc[ix, "q_bh"] = LT.bh(T.loc[ix, "p_perm"].to_numpy())
        T.to_csv(out_dir / f"lick_template_match{sfx}_readout_test.csv", index=False)
        print("\n==== READOUT TEST: r(executed angle, cortical angle readout) WITHIN cued position; null = angles "
              "permuted within position (q_bh across positions per version x session)")
        print(T.round(3).to_string(index=False))
    D.to_csv(out_dir / f"lick_template_match{sfx}.csv", index=False)
    S.to_csv(out_dir / f"lick_template_match{sfx}_stats.csv", index=False)
    for pos in cued:
        g0 = D[D.position == pos] if len(D) else D
        if not len(g0):
            continue
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=False)
        for j, version in enumerate(("raw", "movement_removed")):
            ax = axs[0, j]
            g = g0[g0.version == version]
            for k, (sess, gs) in enumerate(g.groupby("session")):
                for cls, mk in (("toward mouse-LEFT", "o"), ("on-target", "s"), ("toward mouse-right", "^")):
                    gc = gs[gs.cls == cls]
                    if len(gc):
                        ax.scatter(gc.r_target, gc.r_executed, marker=mk, s=28, alpha=0.8,
                                   color=["tab:red", "tab:blue", "tab:green", "tab:purple"][k % 4],
                                   label=f"{sess} {cls} (n={len(gc)})")
            ax.plot([-1, 1], [-1, 1], "k:", lw=0.8)
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 0.8)
            ax.set_xlabel(f"r with pre-stroke {pos} template (TARGET)")
            ax.set_ylabel("r with pre-stroke matched-angle template (EXECUTED)")
            ax.set_title(f"{version} ({pt['baseline']} baseline): one point per post-stroke {pos}-cued lick", fontsize=8)
            ax.legend(fontsize=6)
        fig.suptitle("Above the diagonal: the lick's cortical pattern resembles pre-stroke licks that went the SAME WAY "
                     "more than pre-stroke licks to the same TARGET", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"lick_template_match_{pos}{sfx}.png", dpi=110)
        plt.close(fig)
    print(f"\n-> {out_dir / f'lick_template_match{sfx}.csv'} (+ _stats.csv, figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
