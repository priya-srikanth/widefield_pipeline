"""Figures for the BEHAVIORAL STATES results page (Priya 11-17: GLM-HMM engagement states, AR-HMM syllables, treadmill
push, TIM). Reads only what `wfield_local.behavior_states` collected (`behavior_states.collect`) plus the GLM-HMM trial
table; every choice (models, roles, groups, thresholds, gates, epochs, statistics) comes from that module.

    python scripts/states_page_figs.py [--only occupancy push ...]

Page conventions (docs/templates/README.md): sessions = faded OPEN markers (shape = animal, color = epoch, or the
state's series color where states are the series), pooled = filled marker + 95 % CI (`analysis_kit.boot_ci`, nested
animals -> sessions), no lines across epochs or spouts (categorical x), time-continuous traces = mean over animals +-
SEM band; "*" in the top band = the paired change vs each animal's own pre (`orofacial_cohort.epoch_tables`, i.e.
`analysis_kit.boot_delta`) has a 95 % CI excluding 0; no top spine; no figure numbers inside images.
Writes PNG (+ svg/) to `behavior_states.out_dir()/page_figs/`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402
from wfield_local import behavior_states as bs  # noqa: E402
from wfield_local import cat_points as cp  # noqa: E402
from wfield_local import orofacial_cohort as oc  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402

EPS = list(oc.EPOCH_ORDER)
ECOL = oc.EPOCH_LINE
RNG_SEED = 0


def _save(fig, name: str, od: Path) -> None:
    oc._tidy(fig)                                   # legends out of the data + headroom for the star band
    sig_marks.reserve_headroom(fig)
    fig.tight_layout()
    save_png_svg(fig, od / f"{name}.png")
    plt.close(fig)
    print("  wrote", name)


def _by_animal(S: pd.DataFrame, col: str) -> dict:
    return {a: g[col].dropna().tolist() for a, g in S.groupby("animal", sort=True) if g[col].notna().any()}


def _pooled(ax, x, S: pd.DataFrame, col: str, color, label=None, marker="o"):
    """Session dots + pooled mean with its 95 % CI (nested bootstrap) at category x."""
    cp.session_dots(ax, x, S[col].to_numpy(float), S.animal.to_numpy(), color, jitter=0.07, seed=int(x * 97) % 1000)
    iv = ak.boot_ci(_by_animal(S, col), np.random.default_rng(RNG_SEED))
    if iv is None:
        return
    cp.summary_point(ax, x, iv.point, np.array([[iv.point - iv.lo], [iv.hi - iv.point]]), color, label=label,
                     marker=marker)


def _star_epochs(ax, D: pd.DataFrame, metric: str, xs: dict, row_of=None) -> None:
    """'*' in the top band over each post epoch whose paired change vs pre has a CI excluding 0."""
    for e in EPS[1:]:
        r = D[(D.metric == metric) & (D.epoch == e)]
        if len(r) and (r.lo.iloc[0] > 0 or r.hi.iloc[0] < 0):
            sig_marks.star_top(ax, xs[e], "*", color=ECOL[e], row=0 if row_of is None else row_of(e), fontsize=12)


def epoch_panels(S: pd.DataFrame, metrics: list, labels: dict, name: str, title: str, od: Path, nc: int = 4,
                 ylim01: bool = False) -> pd.DataFrame:
    """One panel per metric, x = epoch: sessions + pooled 95 % CI; stars from the paired change vs own pre."""
    A, D = oc.epoch_tables(S, metrics=metrics)
    nr = int(np.ceil(len(metrics) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.9 * nr), squeeze=False)
    xs = {e: k for k, e in enumerate(EPS)}
    for ax, m in zip(axs.flat, metrics):
        for e in EPS:
            _pooled(ax, xs[e], S[S.epoch == e], m, ECOL[e], label=e)
        _star_epochs(ax, D, m, xs)
        ax.set_xticks(range(len(EPS)), EPS, fontsize=8)
        ax.set_xlim(-0.6, len(EPS) - 0.4)
        if ylim01:
            ax.set_ylim(bottom=0)
        ax.set_title(labels.get(m, m), fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axs.flat[len(metrics):]:
        ax.axis("off")
    cp.add_animal_entries(axs.flat[0], S.animal.unique())
    axs.flat[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(title, fontsize=9)
    _save(fig, name, od)
    D.to_csv(od / f"{name}_delta_vs_pre.csv", index=False)
    A.to_csv(od / f"{name}_per_animal.csv", index=False)
    return D


def trace_panels(per: dict, channels: list, labels: list, name: str, title: str, od: Path, t: np.ndarray,
                 rows: list | None = None, negate: tuple = (), shade: dict | None = None, wrap: int = 0) -> None:
    """Time-continuous traces: mean over animals (each = mean of its sessions) +- SEM band (alpha .08, as the
    lick_matched_readout curves); ``rows`` = [(row key, row title)], ``per[row][label] -> (channels, time)``;
    ``wrap`` = panels per figure row when there is a single data row."""
    rows = rows or [(None, None)]
    nc = wrap if (wrap and len(rows) == 1) else len(channels)
    nr = int(np.ceil(len(channels) / nc)) if nc != len(channels) else len(rows)
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.9 * nr), squeeze=False, sharex=True)
    for ax in axs.flat[len(channels) * len(rows):]:
        ax.axis("off")
    for i, (rk, rt) in enumerate(rows):
        src = per[rk]
        for j, lab_ in enumerate(labels):
            ax = axs.flat[j] if nc != len(channels) else axs[i, j]
            for e in EPS:
                an = {}
                for lab, v in src.items():
                    if bs.session_epoch(lab) == e:
                        an.setdefault(lab.split("_")[0], []).append(v[channels[j]])
                if not an:
                    continue
                M = np.stack([np.mean(an[a], 0) for a in sorted(an)]) * (-1 if channels[j] in negate else 1)
                m = M.mean(0)
                se = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros_like(m)
                ax.fill_between(t, m - se, m + se, color=ECOL[e], alpha=0.08, lw=0, zorder=2)
                ax.plot(t, m, color=ECOL[e], lw=1.5, label=f"{e} (n = {len(M)} animals)", zorder=4)
            ax.axvline(0, color="0.6", lw=0.8, ls="--")
            for (a, b), txt in (shade or {}).items():
                ax.axvspan(a, b, color="0.85", alpha=0.35, lw=0, zorder=0)
                ax.text((a + b) / 2, 1.0, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=7, color="0.4")
            ax.set_ylabel(lab_, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("time from cue (s)", fontsize=8)
            if j == 0 and rt:
                ax.set_title(rt, loc="left", fontsize=9, fontweight="bold")
        (axs.flat[0] if nc != len(channels) else axs[i, -1]).legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(title, fontsize=9)
    _save(fig, name, od)


# ------------------------------------------------------------------------------------------------ figure families

def fig_occupancy(C: dict, od: Path) -> None:
    S = C["glmhmm_occupancy"]
    T = bs.glmhmm_table()
    roles = bs.glmhmm_roles(T)
    ms = list(bs.ROLES) + [f"p_state{k}" for k in sorted(roles)]
    lab = {r: f"{r} states (occupancy)" for r in bs.ROLES}
    lab.update({f"p_state{k}": f"state {k} ({v})" for k, v in roles.items()})
    epoch_panels(S, ms, lab, "glmhmm_occupancy_by_epoch",
                 f"GLM-HMM K{bs.GLMHMM_K} ({bs.GLMHMM_DESIGN}) state occupancy (mean posterior per session), every trial; "
                 "roles from each state's own profile", od, ylim01=True)


def _curve_table(T: pd.DataFrame, outputs: dict, min_w: float = 3.0) -> pd.DataFrame:
    """Posterior-weighted rate per session x state x spout for each output (+ P(state | spout))."""
    pc = bs._pcols(T)
    rows = []
    for (lab, an, ep, pos), g in T.groupby(["label", "animal", "epoch", "pos"]):
        W = g[pc].to_numpy(float)
        for k, c in enumerate(pc):
            w = W[:, k]
            r = {"label": lab, "animal": an, "epoch": ep, "pos": pos, "state": int(c[7:]), "p_state_given_spout": w.mean()}
            for name, y in outputs.items():
                yy = y.loc[g.index].to_numpy(float)
                ok = np.isfinite(yy)
                r[name] = np.sum(w[ok] * yy[ok]) / w[ok].sum() if w[ok].sum() >= min_w else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def fig_state_curves(C: dict, od: Path) -> None:
    T = bs.glmhmm_table()
    roles = bs.glmhmm_roles(T)
    outs = {"contact": T.cls.eq("contact").astype(float), "incomplete": T.cls.eq("incomplete").astype(float),
            "jaw-only": T.cls.eq("jaw_only").astype(float), "no response": T.cls.eq("no_response").astype(float),
            "pre-cue lick": T.precue_lick.eq(1).astype(float).where(T.precue_lick >= 0),
            "ITI lick": T.iti_lick.eq(1).astype(float).where(T.iti_lick >= 0)}
    CT = _curve_table(T, outs)
    CT.to_csv(od / "glmhmm_state_curves_per_session.csv", index=False)
    pos = list(oc.POS_ORDER)
    x = np.arange(len(pos))
    states = sorted(roles)
    lat = bs.laterality()
    # (1) all epochs pooled, series = states
    off = cp.epoch_offsets(len(states), step=0.13)
    cols = list(outs)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for ax, name in zip(axs.flat, cols):
        for si, k in enumerate(states):
            D = CT[CT.state == k]
            for j, p in enumerate(pos):
                _pooled(ax, x[j] + off[si], D[D.pos == p], name, cp.model_color(k),
                        label=f"state {k} ({roles[k]})" if j == 0 else None)
        _spout_axis(ax, x, pos, lat)
        ax.set_title(name, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
    axs.flat[-1].axis("off")
    h, lab = axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(h, lab, fontsize=8, frameon=False, loc="center")
    fig.suptitle(f"GLM-HMM K{bs.GLMHMM_K} ({bs.GLMHMM_DESIGN}) per-state response by spout, all epochs pooled: rate "
                 "weighted by each trial's state posterior; series color = state", fontsize=9)
    _save(fig, "glmhmm_state_curves", od)
    # (2) by epoch: rows = states, columns = outputs + P(state | spout); epochs side by side, stars per spout
    cols2 = ["contact", "incomplete", "jaw-only", "no response", "p_state_given_spout"]
    eoff = cp.epoch_offsets(len(EPS))
    fig, axs = plt.subplots(len(states), len(cols2), figsize=(3.3 * len(cols2), 2.6 * len(states)), squeeze=False)
    for i, k in enumerate(states):
        D = CT[CT.state == k]
        for j, name in enumerate(cols2):
            ax = axs[i, j]
            for ei, e in enumerate(EPS):
                for q, p in enumerate(pos):
                    _pooled(ax, x[q] + eoff[ei], D[(D.pos == p) & (D.epoch == e)], name, ECOL[e],
                            label=e if q == 0 else None)
            W = D.pivot_table(index=["label", "animal", "epoch"], columns="pos", values=name).reset_index()
            _, DD = oc.epoch_tables(W, metrics=[p for p in pos if p in W])
            for q, p in enumerate(pos):
                _star_epochs(ax, DD, p, {e: x[q] + eoff[ei] for ei, e in enumerate(EPS)},
                             row_of=lambda e: EPS.index(e) - 1)
            _spout_axis(ax, x, pos, lat)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title("P(state | spout)" if name == "p_state_given_spout" else name, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"state {k}: {roles[k]}", fontsize=9, fontweight="bold")
    axs[0, -1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(f"GLM-HMM K{bs.GLMHMM_K} per-state response by spout and epoch: does a state mean the same thing after "
                 "the stroke? (emissions are shared across epochs); last column = which states the animals are in at "
                 "each spout", fontsize=9)
    _save(fig, "glmhmm_state_curves_by_epoch", od)


def _spout_axis(ax, x, pos, lat):
    ax.set_xticks(x, [oc.pos_disp(p).replace(" ", "\n") for p in pos], fontsize=7)
    for t_, p in zip(ax.get_xticklabels(), pos):
        t_.set_color(oc._pos_color(p, next(iter(lat.values()))))
    ax.axvline(2.5, color="0.85", lw=0.8, zorder=0)
    ax.set_xlim(-0.6, len(pos) - 0.4)
    ax.tick_params(labelsize=7)


def fig_push(C: dict, od: Path) -> None:
    pp, walk = bs.push_params(), bs.walk_threshold()
    lab = {"push_rate": f"push rate (backward > {pp['thresh']:g} mm/s in cue window)",
           "push_control_rate": "same test in the control window", "push_cue_minus_control": "cue minus control",
           "push_peak": "peak backward speed on push trials (mm/s)", "push_latency_s": "push onset re cue (s)",
           "walk_cue": f"walking (forward > {walk:g} mm/s), cue window", "walk_control": "walking, control window"}
    for gate, gname in (("lick_period", "lick working period"), ("glm_engaged", "GLM-HMM engaged")):
        S = bs.per_session(C["trials"], gate)
        epoch_panels(S, list(lab), lab, f"push_by_epoch_{gate}",
                     f"Treadmill around the cue, trials inside the {gname}: cue window {pp['cue'][0]:g} .. "
                     f"{pp['cue'][1]:g} s, control {pp['control'][0]:g} .. {pp['control'][1]:g} s", od)


def _traces(C: dict, prefix: str) -> dict:
    out = {}
    for k, v in C["traces"].items():
        parts = k.split("|")
        if parts[0] == prefix:
            key = parts[1] if len(parts) == 3 else None
            out.setdefault(key, {})[parts[-1]] = v
    return out


def fig_wheel_traces(C: dict, od: Path) -> None:
    W = _traces(C, "wheel")
    fs, (a, _b) = C["meta"]["fs"], bs.CUE_ALIGN_S
    t = a + (np.arange(next(iter(next(iter(W.values())).values())).shape[1]) + 0.5) / fs
    pp, walk = bs.push_params(), bs.walk_threshold()
    labels = ["forward speed (mm/s)", "backward velocity (mm/s)", f"P(walking: forward > {walk:g} mm/s)",
              f"P(push: backward > {pp['thresh']:g} mm/s)"]
    per = {k: {lab: v for lab, v in d.items()} for k, d in W.items()}
    shade = {pp["control"]: "control", pp["cue"]: "cue window"}
    for gate, rows in (("lick", [("lick_in", "inside the lick working period"), ("lick_out", "outside it")]),
                       ("glm", [("glm_in", "GLM-HMM engaged"), ("glm_out", "GLM-HMM partial + disengaged")])):
        trace_panels(per, [0, 1, 2, 3], labels, f"wheel_cue_aligned_{gate}gate",
                     "Treadmill around the cue (DAQ wheel), engaged vs not; backward plotted as negative velocity",
                     od, t, rows=rows, negate=(1,), shade=shade)


def fig_arhmm(C: dict, od: Path) -> None:
    U = C["usage"]
    groups = C["meta"]["groups"]
    gn = list(groups)
    epoch_panels(U, [f"engaged:{g}" for g in gn], {f"engaged:{g}": f"{g.replace('_', ' ')} ({'-'.join(map(str, groups[g]))})"
                                                   for g in gn}, "arhmm_usage_engaged_by_epoch",
                 f"AR-HMM syllable groups ({C['meta']['tag']}): share of frames on GLM-HMM-engaged trial cycles", od,
                 ylim01=True)
    lk = [g for g in bs.LICK_GROUPS if g in gn]
    epoch_panels(U, [f"engaged:share_{g}" for g in lk], {f"engaged:share_{g}": f"{g.replace('_', ' ')}: share of lick "
                                                         "frames" for g in lk}, "arhmm_lick_direction_by_epoch",
                 "AR-HMM lick syllables as a share of all lick frames (direction, independent of how much the "
                 "mouse licks), GLM-HMM-engaged trial cycles", od, nc=3, ylim01=True)
    E = _traces(C, "etho")[None]
    fs = C["meta"]["fs"]
    t = bs.CUE_ALIGN_S[0] + (np.arange(next(iter(E.values())).shape[1]) + 0.5) / fs
    trace_panels({None: E}, list(range(len(gn))), [f"P({g.replace('_', ' ')})" for g in gn],
                 "arhmm_cue_aligned_groups", "AR-HMM syllable groups around the cue, GLM-HMM-engaged trials", od, t,
                 wrap=3)
    Tr = C["transitions"]
    epoch_panels(Tr, ["entropy_bits"], {"entropy_bits": "transition entropy (bits)"}, "arhmm_transition_entropy",
                 "AR-HMM group transitions on engaged trial cycles (runs collapsed): entropy of the next group", od, nc=1)
    jaw = C["trials"][C["trials"].cls == "jaw_only"]
    J = jaw.groupby(["label", "animal", "epoch"])[[f"jaw:{g}" for g in gn]].mean().reset_index()
    epoch_panels(J, [f"jaw:{g}" for g in gn], {f"jaw:{g}": g.replace("_", " ") for g in gn},
                 "arhmm_groups_at_jaw_onset_jawonly",
                 f"Jaw-only trials: AR-HMM syllable groups at the jaw onset (+- {bs.JAW_HALF_S:g} s)", od, ylim01=True)


def fig_frozen(C: dict, od: Path) -> None:
    S = bs.frozen_distance()
    S.to_csv(od / "arhmm_frozen_distance_per_session.csv", index=False)
    epoch_panels(S, ["ll_rel"], {"ll_rel": "log-likelihood per frame under the pre-stroke AR-HMM (relative, nats)"},
                 "arhmm_frozen_likelihood_by_epoch",
                 "How pre-stroke-like is each session's movement? AR-HMM marginal log-likelihood under held-out "
                 "pre-stroke fits (8 folds); pre days scored only by the fold that left them out", od, nc=1)


FAMILIES = {"occupancy": fig_occupancy, "curves": fig_state_curves, "push": fig_push, "wheel": fig_wheel_traces,
            "arhmm": fig_arhmm, "frozen": fig_frozen}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("--only", nargs="*", default=list(FAMILIES), choices=list(FAMILIES))
    a = ap.parse_args(argv)
    C = bs.load_collected()
    od = bs.out_dir() / "page_figs"
    od.mkdir(parents=True, exist_ok=True)
    print("families:", a.only)
    for f in a.only:
        FAMILIES[f](C, od)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
