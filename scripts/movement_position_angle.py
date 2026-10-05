"""Position-specific encoding AFTER regressing out movement, and lick responses organised by EXECUTED tongue angle vs
target position -- LocaNMF dF/F on the frames inside the session's pose windows.

    python -m scripts.movement_position_angle PS93:20260814 PS93:20260821 PS93:20260908 [--perms 20]

Priya, 2026-10-05: "let's try looking at residual position activity - but is a decoder the right way? can we regress
out the 'movement' information and then look at encoding of position-specific information?" + "can we try a version
looking at continuous or binned tongue angle on executed licks?"

A. RESIDUAL POSITION ENCODING. Movement (DAQ contacts, DLC tongue onsets, continuous tongue / jaw) is regressed out
   with a CROSS-FITTED residual (`movement_encoding.cv_residual`: each fold predicted by a movement model fit on the
   other folds -- movement first = conservative, shared variance goes with movement). On the residual, two task models:
   one SHARED cue kernel for all positions vs a cue kernel PER POSITION. Position-specific information =
   CV R^2(per position) - CV R^2(shared), per component; null = the same with position labels shuffled across trials
   (`--perms`). Plus cue-aligned residual averages per position and area (the residual "position activity").
B. EXECUTED ANGLE vs TARGET. Base = cue per position + contacts + continuous tongue / jaw. Lick-onset kernels:
   `single` (one kernel for executed licks + one for short licks), `position` (one per target position), `angle`
   (one per executed-angle bin: 6 within-session quantiles of the peak angle, licks reaching >= min_reach_px), `both`.
   CV R^2 gain over `single`; ANGLE BEYOND POSITION = R^2(both) - R^2(position), null = angle bins shuffled WITHIN
   each position (the angle-position association is kept, only within-position angle information is destroyed).
Pose covers 60 trials (11-13 % of a session) until whole-session predictions exist -- a first look, not a result.
Outputs: session_poses/<a>_<d>/movement_position_angle.csv (+ residual_position_traces.png); a cross-session summary
session_poses/movement_position_angle_summary.csv and figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import movement_encoding as ME
from wfield_local import movement_inputs as MI

N_ANGLE_BINS = 6
POS_ORDER = ["far_L", "close_L", "far_center", "close_center", "close_R", "far_R"]
POS_COLORS = {"far_L": "#1f77b4", "close_L": "#6baed6", "far_center": "#2ca02c", "close_center": "#98df8a",
              "close_R": "#ff9896", "far_R": "#d62728"}


def _design(ft, mask, **kw):
    return ME.select_rows(ME.build_design(MI.build_inputs(ft, **kw)), mask)


def _cv_r2(d, Y, folds, alphas):
    Z, _, _ = ME.standardise(d.X)
    return ME.r2_score(Y, ME.cv_predict(Z, Y, d.group, alphas, folds))


def part_a(ft, mask, P, Y, folds, perms, rng, grid):
    tr, L = P["trials"], P["licks"]
    mov = _design(ft, mask, events={"contact": P["contact_s"], "tongue_onset": L.on_s.to_numpy()},
                  video_signals=P["sig"], video_t_s=P["vt"], trial_starts_s=P["trial_starts_s"])
    a_mov = ME.choose_alphas(ME.standardise(mov.X)[0], Y, mov.group, folds, grid=grid)
    R = ME.cv_residual(mov, Y, folds, a_mov)
    shared = _design(ft, mask, cues=tr.assign(pos_name="all")[["cue_s", "pos_name"]])
    spec = _design(ft, mask, cues=tr[["cue_s", "pos_name"]])
    a_task = ME.choose_alphas(ME.standardise(spec.X)[0], R, spec.group, folds, grid=grid)
    r2_sh, r2_sp = _cv_r2(shared, R, folds, a_task), _cv_r2(spec, R, folds, a_task)
    null = []
    for _ in range(perms):
        sh = tr[["cue_s", "pos_name"]].assign(pos_name=rng.permutation(tr.pos_name.to_numpy()))
        null.append(_cv_r2(_design(ft, mask, cues=sh), R, folds, a_task) - r2_sh)
    return R, r2_sh, r2_sp, np.array(null), a_mov, a_task


def _onset_events(L, mode, rng=None, shuffle_within_position=False):
    ex, short = L[L.reach_ok], L[~L.reach_ok]
    ev = {"tongue_onset_short": short.on_s.to_numpy()}
    if mode in ("single",):
        ev["tongue_onset_exec"] = ex.on_s.to_numpy()
    if mode in ("position", "both"):
        for pos, g in ex.groupby("position"):
            ev[f"onset_pos_{pos}"] = g.on_s.to_numpy()
    if mode in ("angle", "both"):
        b = ex.angle_bin.to_numpy()
        if shuffle_within_position:
            b = b.copy()
            for pos in ex.position.unique():
                m = (ex.position == pos).to_numpy()
                b[m] = rng.permutation(b[m])
        for k in range(N_ANGLE_BINS):
            ev[f"onset_ang{k}"] = ex.on_s.to_numpy()[b == k]
    return ev


def _overrides(ev):
    lags = {n: [-0.5, 1.0] for n in ev}
    groups = {n: "onset" for n in ev}
    return {"lags_s": lags, "groups": groups}


def part_b(ft, mask, P, Y, folds, perms, rng, grid):
    tr, L = P["trials"], P["licks"].copy()
    ex = L.reach_ok
    q = np.quantile(L.loc[ex, "angle"], np.linspace(0, 1, N_ANGLE_BINS + 1))
    L["angle_bin"] = -1
    L.loc[ex, "angle_bin"] = np.clip(np.searchsorted(q, L.loc[ex, "angle"], side="right") - 1, 0, N_ANGLE_BINS - 1)
    base = {"cues": tr[["cue_s", "pos_name"]], "video_signals": P["sig"], "video_t_s": P["vt"],
            "trial_starts_s": P["trial_starts_s"]}

    def design(ev):
        e = {"contact": P["contact_s"], **ev}
        ov = _overrides(ev)
        return _design(ft, mask, events=e, overrides=ov, **base)

    d_both = design(_onset_events(L, "both"))
    alphas = ME.choose_alphas(ME.standardise(d_both.X)[0], Y, d_both.group, folds, grid=grid)
    r2 = {m: _cv_r2(design(_onset_events(L, m)), Y, folds, alphas) for m in ("single", "position", "angle", "both")}
    null = np.array([_cv_r2(design(_onset_events(L, "both", rng, True)), Y, folds, alphas) - r2["position"]
                     for _ in range(perms)])
    return r2, null, alphas, q


def area_name(lab):
    names = {3: "MOp", 4: "MOs", 5: "SSp-n", 6: "SSp-m"}
    return f"{names.get(abs(int(lab)), abs(int(lab)))}_{'L' if lab > 0 else 'R'}"


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import scripts.movement_encoding_session as MS
    from scripts.session_poses import session_dir
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--areas", nargs="+", type=int, default=[4, 3, 6, 5])
    a = ap.parse_args(argv)
    rv = PathResolver()
    p = MI.params()
    grid = tuple(p["alpha_grid"])
    rng = np.random.default_rng(0)
    rows, traces = [], {}
    for spec in a.sessions:
        animal, date = spec.split(":")[:2]
        label = f"{animal}_{date[4:]}"
        Yall, reg, ft = MS.imaging(label)
        P = MS.session_pieces(animal, date, rv, ft, float(p["min_reach_px"]))
        mask = P["mask"]
        keep = [k for k, lab in enumerate(reg) if abs(int(lab)) in a.areas]
        Y = Yall[mask][:, keep]
        names = [area_name(reg[k]) for k in keep]
        folds = ME.trial_block_folds(ft[mask], P["trial_starts_s"], int(p["n_folds"]))
        R, r2_sh, r2_sp, nullA, a_mov, a_task = part_a(ft, mask, P, Y, folds, a.perms, rng, grid)
        r2b, nullB, a_b, q = part_b(ft, mask, P, Y, folds, a.perms, rng, grid)
        df = pd.DataFrame({"session": label, "component": keep, "area": names, "r2_resid_shared": r2_sh,
                           "r2_resid_specific": r2_sp, "pos_specific": r2_sp - r2_sh,
                           "pos_specific_null_mean": nullA.mean(0),
                           **{f"r2_onset_{m}": v for m, v in r2b.items()},
                           "angle_beyond_position": r2b["both"] - r2b["position"],
                           "angle_beyond_null_mean": nullB.mean(0)})
        out = session_dir(rv, spec)[2]
        df.to_csv(out / "movement_position_angle.csv", index=False)
        rows.append((df, nullA, nullB))
        # per-area medians + permutation p (median over the area's components vs the null's medians)
        print(f"\n== {label}: residual-position alphas mov {a_mov} task {a_task}; onset-model alphas {a_b}; angle "
              f"bins (deg) {np.round(q, 1).tolist()}")
        for ar in sorted(set(names)):
            m = np.array([n == ar for n in names])
            ps = np.median(df.pos_specific[m])
            pnull = np.median(nullA[:, m], axis=1)
            ab = np.median(df.angle_beyond_position[m])
            bnull = np.median(nullB[:, m], axis=1)
            print(f"  {ar:8s} n={m.sum():2d}  position-specific (resid) {ps:+.4f} (null {pnull.mean():+.4f}, "
                  f"p={(np.sum(pnull >= ps) + 1) / (len(pnull) + 1):.2f})  |  onset gain: position "
                  f"{np.median(df.r2_onset_position[m] - df.r2_onset_single[m]):+.4f}  angle "
                  f"{np.median(df.r2_onset_angle[m] - df.r2_onset_single[m]):+.4f}  angle-beyond-position {ab:+.4f} "
                  f"(null {bnull.mean():+.4f}, p={(np.sum(bnull >= ab) + 1) / (len(bnull) + 1):.2f})")
        # residual cue-aligned averages per position, per area (mean over the area's components)
        Rfull = np.full((len(ft), len(keep)), np.nan)
        Rfull[mask] = R
        lag = np.arange(int(-0.5 * 31.23), int(2.0 * 31.23))
        tr = P["trials"]
        cf = ME.nearest_frame(tr.cue_s.to_numpy(), ft)
        traces[label] = {}
        for ar in sorted(set(names)):
            m = np.array([n == ar for n in names])
            sig = np.nanmean(Rfull[:, m], axis=1)
            for pos in POS_ORDER:
                ff = cf[(tr.pos_name == pos).to_numpy() & (cf >= 0)]
                ii = ff[:, None] + lag[None, :]
                ii = ii[(ii.min(1) >= 0) & (ii.max(1) < len(ft))]
                traces[label][(ar, pos)] = np.nanmean(sig[ii], axis=0) if len(ii) else np.full(len(lag), np.nan)
        areas = sorted(set(names))
        fig, axs = plt.subplots(1, len(areas), figsize=(3.0 * len(areas), 3.2), squeeze=False, sharey=True)
        for j, ar in enumerate(areas):
            ax = axs[0, j]
            for pos in POS_ORDER:
                ax.plot(lag / 31.23, traces[label][(ar, pos)], color=POS_COLORS[pos], lw=1.5, label=pos)
            ax.axvline(0, color="k", lw=0.5)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_title(ar, fontsize=9)
            ax.set_xlabel("s from cue")
        axs[0, 0].set_ylabel("residual dF/F (movement removed)")
        axs[0, 0].legend(fontsize=6)
        fig.suptitle(f"{label}: cue-aligned activity AFTER regressing out movement (cross-fitted), mean per position "
                     f"(area = mean over its components; _L = left = ipsilesional)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "residual_position_traces.png", dpi=110)
        plt.close(fig)
    S = pd.concat([r[0] for r in rows], ignore_index=True)
    summ = S.groupby(["session", "area"]).median(numeric_only=True).reset_index()
    out = Path(rv.root("microscope")) / "DeepLabCut" / "Widefield" / "session_poses"
    summ.to_csv(out / "movement_position_angle_summary.csv", index=False)
    print(f"\n-> {out / 'movement_position_angle_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
