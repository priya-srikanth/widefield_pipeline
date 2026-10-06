"""Descriptive companions to `movement_position_angle` (NOT part of its pre-registered tests): where the residual
position activity sits in time, and what the lick kernels look like by target position vs executed angle.

    python -m scripts.residual_lick_aligned PS93:20260814:full

Priya, 2026-10-06: "why the far R and far center kernels don't have post-cue evoked residuals like the other
positions?" + "should we try aligning to first-lick peak protrusion (from DLC, not spout contact)?" + "is there a
graphical comparison of the fit based on spout position vs executed target angle?"

Figure 1 (residual_lick_aligned.png): the cross-fitted movement residual (same movement model as
`movement_position_angle` part A), mean per position and area, aligned (top) to the cue and (bottom) to the PEAK
PROTRUSION of each trial's first executed lick (DLC, reach >= min_reach_px; not the DAQ contact). Far-spout first
licks come 400-900 ms after the cue with a 0.6-1.1 s spread (close: ~150 ms, 50 ms spread), so a lick-locked
response is smeared away in a cue-aligned average; if the far positions' response reappears lick-aligned, their
"missing" post-cue transient was timing, not absence.
Figure 2 (onset_kernels_position_vs_angle.png): the lick-onset kernels of part B's `position` model (one kernel per
target position) and `angle` model (one per executed-angle bin at peak protrusion), per area, in dF/F PER LICK
(the fitted weight divided by the regressor's SD -- the standardised weights in movement_encoding_kernels.png are
per SD of a sparse 0/1 column, hence their tiny scale), with each model's CV R^2 gain over one shared kernel.
If cortex tracks the target, the position kernels separate and the angle-bin kernels look alike; if it tracks
the executed direction, the reverse.
"""
from __future__ import annotations

import argparse

import numpy as np

import scripts.movement_position_angle as MPA
from wfield_local import movement_encoding as ME
from wfield_local import movement_inputs as MI

FPS_IMG = 31.23


def first_lick_peaks(P) -> np.ndarray:
    """Per trial (P["trials"] order): DAQ time of the first EXECUTED lick's peak protrusion after the cue (NaN if
    none)."""
    L = P["licks"]
    L = L[L.reach_ok & (L.peak_s >= L.cue_s)].sort_values("peak_s")
    first = L.groupby("trial_id").peak_s.first()
    return P["trials"].trial_id.map(first).to_numpy(float)


def aligned_means(sig, ft, t_events, pos, lag):
    """{position: mean of ``sig`` around the imaging frame nearest each event}."""
    out = {}
    ok = np.isfinite(t_events)
    f = np.full(len(t_events), -1)
    f[ok] = ME.nearest_frame(t_events[ok], ft)
    for p in MPA.POS_ORDER:
        ff = f[(pos == p) & (f >= 0)]
        ii = ff[:, None] + lag[None, :]
        ii = ii[(ii.min(1) >= 0) & (ii.max(1) < len(sig))]
        out[p] = (np.nanmean(sig[ii], axis=0), len(ii))
    return out


def onset_kernels(ft, mask, P, Y, folds, grid):
    """Part B's position and angle models refit on all data -> ({model: {regressor: (lags, per-lick kernel)}},
    {model: CV R^2 per output})."""
    L = P["licks"].copy()
    ex = L.reach_ok
    q = np.quantile(L.loc[ex, "angle"], np.linspace(0, 1, MPA.N_ANGLE_BINS + 1))
    L["angle_bin"] = -1
    L.loc[ex, "angle_bin"] = np.clip(np.searchsorted(q, L.loc[ex, "angle"], side="right") - 1, 0,
                                     MPA.N_ANGLE_BINS - 1)
    base = {"cues": P["trials"][["cue_s", "pos_name"]], "video_signals": P["sig"], "video_t_s": P["vt"],
            "trial_starts_s": P["trial_starts_s"]}

    def design(ev):
        return MPA._design(ft, mask, events={"contact": P["contact_s"], **ev}, overrides=MPA._overrides(ev), **base)

    d_both = design(MPA._onset_events(L, "both"))
    alphas = ME.choose_alphas(ME.standardise(d_both.X)[0], Y, d_both.group, folds, grid=grid)
    ks, r2 = {}, {}
    for m in ("single", "position", "angle"):
        d = design(MPA._onset_events(L, m))
        r2[m] = MPA._cv_r2(d, Y, folds, alphas)
        model = ME.fit(d, Y, alphas=alphas)
        sd = dict(zip(model.names, model.sd))
        per = {}
        for reg_name, (lg, w) in ME.kernels(model).items():
            if reg_name.startswith("onset_"):
                cols = [n for n in model.names if model.regressor[model.names.index(n)] == reg_name]
                s = np.array([sd[c] for c in cols])[np.argsort([model.lag_s[model.names.index(c)] for c in cols])]
                per[reg_name] = (lg, w / s[:, None])
        ks[m] = per
    return ks, r2, q


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import scripts.movement_encoding_session as MS
    from scripts.session_poses import session_dir
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="animal:date[:tag], e.g. PS93:20260814:full")
    ap.add_argument("--areas", nargs="+", type=int, default=[4, 3, 6, 5])
    a = ap.parse_args(argv)
    rv = PathResolver()
    p = MI.params()
    grid = tuple(p["alpha_grid"])
    animal, date = a.session.split(":")[:2]
    label = f"{animal}_{date[4:]}"
    Yall, reg, ft = MS.imaging(label)
    label += "".join(f"_{t}" for t in a.session.split(":")[2:])
    P = MS.session_pieces(animal, date, rv, ft, float(p["min_reach_px"]), spec=a.session)
    mask = P["mask"]
    keep = [k for k, lab in enumerate(reg) if abs(int(lab)) in a.areas]
    Y = Yall[mask][:, keep]
    names = [MPA.area_name(reg[k]) for k in keep]
    areas = sorted(set(names))
    folds = ME.trial_block_folds(ft[mask], P["trial_starts_s"], int(p["n_folds"]))
    out = session_dir(rv, a.session)[2]

    # ---- figure 1: residual, cue- vs first-lick-peak-aligned
    L = P["licks"]
    mov = MPA._design(ft, mask, events={"contact": P["contact_s"], "tongue_onset": L.on_s.to_numpy()},
                      video_signals=P["sig"], video_t_s=P["vt"], trial_starts_s=P["trial_starts_s"])
    a_mov = ME.choose_alphas(ME.standardise(mov.X)[0], Y, mov.group, folds, grid=grid)
    R = ME.cv_residual(mov, Y, folds, a_mov)
    Rfull = np.full((len(ft), len(keep)), np.nan)
    Rfull[mask] = R
    tr = P["trials"]
    pos = tr.pos_name.to_numpy()
    t_cue, t_pk = tr.cue_s.to_numpy(float), first_lick_peaks(P)
    lat = (t_pk - t_cue) * 1000
    lag = np.arange(int(-0.5 * FPS_IMG), int(2.0 * FPS_IMG))
    lag_pk = np.arange(int(-1.0 * FPS_IMG), int(1.5 * FPS_IMG))
    fig, axs = plt.subplots(2, len(areas), figsize=(3.0 * len(areas), 6.4), squeeze=False, sharey="row")
    for j, ar in enumerate(areas):
        sig = np.nanmean(Rfull[:, [n == ar for n in names]], axis=1)
        for row, (te, lg, xl) in enumerate([(t_cue, lag, "s from cue"),
                                            (t_pk, lag_pk, "s from first-lick PEAK protrusion")]):
            ax = axs[row, j]
            for pname, (m, n) in aligned_means(sig, ft, te, pos, lg).items():
                ax.plot(lg / FPS_IMG, m, color=MPA.POS_COLORS[pname], lw=1.5, label=f"{pname} (n={n})")
            ax.axvline(0, color="k", lw=0.5)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_xlabel(xl, fontsize=8)
            if row == 0:
                ax.set_title(ar, fontsize=9)
    axs[0, 0].set_ylabel("residual dF/F (movement removed)")
    axs[1, 0].set_ylabel("residual dF/F (movement removed)")
    axs[1, 0].legend(fontsize=6, loc="upper left")
    med = {p_: np.nanmedian(lat[pos == p_]) for p_ in MPA.POS_ORDER}
    fig.suptitle(f"{label}: movement residual (cross-fitted), cue-aligned (top) vs first executed lick's peak "
                 f"(bottom). Median first-peak latency (ms): "
                 + ", ".join(f"{k} {v:.0f}" for k, v in med.items()), fontsize=8)
    fig.tight_layout()
    f1 = out / "residual_lick_aligned.png"
    fig.savefig(f1, dpi=110)
    plt.close(fig)
    print("->", f1)

    # ---- figure 2: onset kernels, position vs angle bin
    ks, r2, q = onset_kernels(ft, mask, P, Y, folds, grid)
    cmap = plt.get_cmap("coolwarm")
    fig, axs = plt.subplots(2, len(areas), figsize=(3.0 * len(areas), 6.4), squeeze=False, sharey="row")
    for j, ar in enumerate(areas):
        m = np.array([n == ar for n in names])
        gp = np.median(r2["position"][m] - r2["single"][m])
        ga = np.median(r2["angle"][m] - r2["single"][m])
        ax = axs[0, j]
        for pname in MPA.POS_ORDER:
            k = ks["position"].get(f"onset_pos_{pname}")
            if k is not None:
                ax.plot(k[0], k[1][:, m].mean(1), color=MPA.POS_COLORS[pname], lw=1.5, label=pname)
        ax.set_title(f"{ar}\nby TARGET position (gain {gp:+.4f})", fontsize=8)
        ax = axs[1, j]
        for b in range(MPA.N_ANGLE_BINS):
            k = ks["angle"].get(f"onset_ang{b}")
            if k is not None:
                ax.plot(k[0], k[1][:, m].mean(1), color=cmap(b / (MPA.N_ANGLE_BINS - 1)), lw=1.5,
                        label=f"{q[b]:+.0f}..{q[b + 1]:+.0f} deg")
        ax.set_title(f"by EXECUTED angle bin (gain {ga:+.4f})", fontsize=8)
        for ax in axs[:, j]:
            ax.axvline(0, color="k", lw=0.5)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_xlabel("s from tongue onset", fontsize=8)
    axs[0, 0].set_ylabel("dF/F per lick")
    axs[1, 0].set_ylabel("dF/F per lick")
    axs[0, 0].legend(fontsize=6)
    axs[1, 0].legend(fontsize=6, title="angle at peak (+ = image-right = mouse L)", title_fontsize=6)
    fig.suptitle(f"{label}: lick-onset kernels (executed licks), one per target position (top) vs one per executed-"
                 f"angle bin (bottom); gain = median CV R^2 over one shared kernel", fontsize=8)
    fig.tight_layout()
    f2 = out / "onset_kernels_position_vs_angle.png"
    fig.savefig(f2, dpi=110)
    plt.close(fig)
    print("->", f2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
