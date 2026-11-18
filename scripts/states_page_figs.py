"""Figures for the BEHAVIORAL STATES results page (Priya 11-17: GLM-HMM engagement states, AR-HMM syllables, treadmill
push, TIM). Reads only what `wfield_local.behavior_states` collected (`behavior_states.collect`) plus the GLM-HMM trial
table; every choice (models, roles, groups, thresholds, gates, epochs, statistics) comes from that module.

    python scripts/states_page_figs.py [--only occupancy push ...]

Page conventions (docs/templates/README.md): sessions = faded OPEN markers (shape = animal, color = epoch, or the
state's series color where states are the series), pooled = filled marker + 95 % CI (`analysis_kit.boot_ci`, nested
animals -> sessions), no lines across epochs or spouts (categorical x), time-continuous traces = mean over animals +-
SEM band; "*" in the top band = the paired change vs each animal's own pre (`orofacial_cohort.epoch_tables`, i.e.
`analysis_kit.boot_delta`) has a 95 % CI excluding 0; no top spine; no figure numbers inside images.
Every by-epoch figure is written twice (Priya 11-18): ``<name>.png`` = levels, ``<name>_dpre.png`` = each session minus
its animal's own mean pre (`orofacial_cohort.delta_from_pre`; pre sessions scatter around 0 = the noise floor), pooled
= the paired `boot_delta` change and CI (the same numbers as the stars). Spout tick labels are plain black unless
spouts are overlaid in one panel (Priya 11-18); GLM-HMM states use `behavior_states.glmhmm_state_colors`.
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
    delta_panels(S, D, metrics, labels, name, title, od, nc=nc)
    return D


def delta_panels(S: pd.DataFrame, D: pd.DataFrame, metrics: list, labels: dict, name: str, title: str, od: Path,
                 nc: int = 4) -> None:
    """The change-vs-own-pre version of `epoch_panels`: sessions minus their animal's mean pre; pooled = the paired
    `boot_delta` change + 95 % CI from ``D`` (the stars' numbers); pre = sessions only (noise floor around 0)."""
    Z = oc.delta_from_pre(S.assign(pos="all"), tuple(metrics))
    nr = int(np.ceil(len(metrics) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.9 * nr), squeeze=False)
    xs = {e: k for k, e in enumerate(EPS)}
    for ax, m in zip(axs.flat, metrics):
        ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=1)
        for e in EPS:
            G = Z[Z.epoch == e]
            cp.session_dots(ax, xs[e], G[f"{m}_dpre"].to_numpy(float), G.animal.to_numpy(), ECOL[e], jitter=0.07,
                            seed=xs[e] * 97)
            r = D[(D.metric == m) & (D.epoch == e)]
            if e != "pre" and len(r) and np.isfinite(r.delta.iloc[0]):
                d, lo, hi = r.delta.iloc[0], r.lo.iloc[0], r.hi.iloc[0]
                err = np.array([[d - lo], [hi - d]]) if np.isfinite(lo) else None
                cp.summary_point(ax, xs[e], d, err, ECOL[e], label=e)
        _star_epochs(ax, D, m, xs)
        ax.set_xticks(range(len(EPS)), EPS, fontsize=8)
        ax.set_xlim(-0.6, len(EPS) - 0.4)
        ax.set_title(labels.get(m, m) + ",\nchange vs own pre", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axs.flat[len(metrics):]:
        ax.axis("off")
    cp.add_animal_entries(axs.flat[0], S.animal.unique())
    axs.flat[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(title + " -- change vs each animal's own pre", fontsize=9)
    _save(fig, f"{name}_dpre", od)


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
                ax.set_title(rt, loc="left", fontsize=9, fontweight="bold", pad=16 if shade else 6)  # above the window labels
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
    lab.update({f"p_state{k}": f"state {bs.glmhmm_state_label(k)} ({v})" for k, v in roles.items()})
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
    scol = bs.glmhmm_state_colors(roles)
    lat = bs.laterality()
    # (1) all epochs pooled, series = states
    off = cp.epoch_offsets(len(states), step=0.13)
    cols = list(outs)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for ax, name in zip(axs.flat, cols):
        for si, k in enumerate(states):
            D = CT[CT.state == k]
            for j, p in enumerate(pos):
                _pooled(ax, x[j] + off[si], D[D.pos == p], name, scol[k],
                        label=f"state {bs.glmhmm_state_label(k)} ({roles[k]})" if j == 0 else None)
        _spout_axis(ax, x, pos, lat)
        ax.set_title(name, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
    for ax in axs.flat[len(cols):]:
        ax.axis("off")
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
                ax.set_ylabel(f"state {bs.glmhmm_state_label(k)}: {roles[k]}", fontsize=9, fontweight="bold")
    axs[0, -1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle(f"GLM-HMM K{bs.GLMHMM_K} per-state response by spout and epoch: does a state mean the same thing after "
                 "the stroke? (emissions are shared across epochs); last column = which states the animals are in at "
                 "each spout", fontsize=9)
    _save(fig, "glmhmm_state_curves_by_epoch", od)


def _spout_axis(ax, x, pos, lat, colored: bool = False):
    """Spout x axis (`orofacial_cohort.POS_ORDER`). Tick labels are colored by spout only when ``colored`` (spouts
    overlaid in one panel); otherwise black (Priya 11-18)."""
    ax.set_xticks(x, [oc.pos_disp(p).replace(" ", "\n") for p in pos], fontsize=7)
    if colored:
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
    rows = [(f"role_{r}", f"GLM-HMM {r} state") for r in bs.ROLES]
    if all(r in per for r, _t in rows):
        trace_panels(per, [0, 1, 2, 3], labels, "wheel_cue_aligned_glmroles",
                     "Treadmill around the cue by GLM-HMM state role (each trial's most probable state); backward "
                     "plotted as negative velocity", od, t, rows=rows, negate=(1,), shade=shade)
    rows = [(f"out_{o}", {"working": "working trials (contact or incomplete camera lick)", "jaw_only": "jaw-only trials",
                          "no_response": "no-response trials"}[o]) for o in bs.OUTCOME_ROWS]
    if all(r in per for r, _t in rows):
        trace_panels(per, [0, 1, 2, 3], labels, "wheel_cue_aligned_outcome",
                     "Treadmill around the cue by trial outcome, inside the lick working period (camera licks, cam4 + "
                     "cam1); backward plotted as negative velocity", od, t, rows=rows, negate=(1,), shade=shade)


def fig_glmhmm_cue_traces(C: dict, od: Path) -> None:
    """What each GLM-HMM state looks like around the cue: tongue, jaw, nose and treadmill per state (series), one row per
    epoch; mean over animals (each = mean of its sessions) +- SEM band."""
    T = bs.glmhmm_table()
    roles = bs.glmhmm_roles(T)
    scol = bs.glmhmm_state_colors(roles)
    W, F = _traces(C, "wheel"), _traces(C, "feat")
    fs, (a, _b) = C["meta"]["fs"], bs.CUE_ALIGN_S
    t = a + (np.arange(next(iter(next(iter(W.values())).values())).shape[1]) + 0.5) / fs
    pp = bs.push_params()
    chans = [("feat", 0, "tongue extension (mm)", 1), ("feat", 1, "tongue lateral (mm, + = contra)", 1),
             ("feat", 2, "jaw vertical (cam4 px, down +)", 1), ("feat", 3, "nose vs mouth (cam4 px)", 1),
             ("wheel", 0, "forward speed (mm/s)", 1), ("wheel", 1, "backward velocity (mm/s)", -1),
             ("wheel", 3, f"P(push: backward > {pp['thresh']:g} mm/s)", 1)]
    fig, axs = plt.subplots(len(EPS), len(chans), figsize=(2.9 * len(chans), 2.4 * len(EPS)), squeeze=False,
                            sharex=True)
    for i, e in enumerate(EPS):
        for j, (kind, ch, lab_, sg) in enumerate(chans):
            ax = axs[i, j]
            src = W if kind == "wheel" else F
            for k in sorted(roles):
                d = src.get(f"state_{k}", {})
                an = {}
                for lab, v in d.items():
                    if bs.session_epoch(lab) == e:
                        an.setdefault(lab.split("_")[0], []).append(v[ch])
                if not an:
                    continue
                M = np.stack([np.nanmean(an[x], 0) for x in sorted(an)]) * sg
                m = np.nanmean(M, 0)
                se = np.nanstd(M, 0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros_like(m)
                ax.fill_between(t, m - se, m + se, color=scol[k], alpha=0.08, lw=0, zorder=2)
                ax.plot(t, m, color=scol[k], lw=1.3, label=f"{bs.glmhmm_state_label(k)}: {roles[k]} (n = {len(M)})",
                        zorder=4)
            ax.axvline(0, color="0.6", lw=0.8, ls="--")
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(lab_, fontsize=8)
            if j == 0:
                ax.set_ylabel(e, fontsize=9, fontweight="bold", color=ECOL[e])
            if i == len(EPS) - 1:
                ax.set_xlabel("time from cue (s)", fontsize=8)
        axs[i, -1].legend(fontsize=6, frameon=False, loc="upper right")
    for j in range(len(chans)):                                       # one y range per column (epochs comparable)
        lo = min(ax.get_ylim()[0] for ax in axs[:, j])
        hi = max(ax.get_ylim()[1] for ax in axs[:, j])
        for ax in axs[:, j]:
            ax.set_ylim(lo, hi)
    fig.suptitle(f"GLM-HMM K{bs.GLMHMM_K}: tongue, jaw, nose and treadmill around the cue, per state (each trial's most "
                 "probable state; every trial)", fontsize=9)
    _save(fig, "glmhmm_cue_aligned_states", od)


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
    S = bs.frozen_distance_frames()
    S.to_csv(od / "arhmm_frozen_distance_per_session.csv", index=False)
    groups = [c for c in S if c.startswith("ll_rel:")]
    epoch_panels(S, ["ll_rel", "ll_rel_moving"],
                 {"ll_rel": "all frames", "ll_rel_moving": "movement frames only (quiet excluded)"},
                 "arhmm_frozen_likelihood_by_epoch",
                 "How pre-stroke-like is each session's movement? log-likelihood per frame (relative, nats) under "
                 "held-out pre-stroke AR-HMM fits (8 folds); pre days scored only by the fold that left them out", od,
                 nc=2)
    epoch_panels(S, groups, {g: g.split(":")[1].replace("_", " ") for g in groups},
                 "arhmm_frozen_likelihood_by_group",
                 "Held-out pre-stroke AR-HMM log-likelihood per frame (relative, nats) by syllable group of the frame "
                 "(reviewed pooled-fit groups): which kinds of movement are least pre-stroke-like", od, nc=5)


def fig_frozen_stats(C: dict, od: Path, tag: str = "pre_s1") -> None:
    """The FROZEN pre-stroke AR-HMM applied to every session, combined into the reviewed groups (each frozen state
    joins the group it overlaps most): usage and bout length per group, whole sessions."""
    S, N = bs.group_stats(tag)
    S.to_csv(od / f"frozen_{tag}_group_stats.csv", index=False)
    N.to_csv(od / f"frozen_{tag}_state_names.csv", index=False)
    gn = [c.split(":")[1] for c in S if c.startswith("use:")]
    members = {g: "-".join(str(int(s)) for s in N[N.group == g].state) for g in gn}
    for kind, lab_, ylim in (("use", "share of frames", True), ("bout", "median bout (s)", False)):
        epoch_panels(S, [f"{kind}:{g}" for g in gn],
                     {f"{kind}:{g}": f"{g.replace('_', ' ')} (frozen {members[g]}): {lab_}" for g in gn},
                     f"frozen_{tag}_{kind}_by_epoch",
                     f"Frozen pre-stroke AR-HMM ({tag}) decoded on every session (whole session), combined into the "
                     f"reviewed groups: {lab_}", od, nc=5, ylim01=ylim)
    epoch_panels(S, ["entropy_bits"], {"entropy_bits": "group transition entropy (bits)"},
                 f"frozen_{tag}_summary_by_epoch", f"Frozen pre-stroke AR-HMM ({tag}), groups: transition entropy, "
                 "whole sessions", od, nc=1)


def fig_glmhmm_model(C: dict, od: Path) -> None:
    """The identified GLM-HMM states: held-out K sweep, model emission profiles, P(attempt) by spout, transitions."""
    from wfield_local import glm_hmm
    T = bs.glmhmm_table(merge=False)                  # the fitted model: all K states (readouts combine GLMHMM_MERGE)
    roles = bs.glmhmm_roles(T)
    scol = bs.glmhmm_state_colors(roles)
    P = bs.glmhmm_params()
    R = bs.glmhmm_select_k()
    fig = plt.figure(figsize=(16, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.25, 1.9, 1.15])
    # (a) K sweep
    ax = fig.add_subplot(gs[0])
    ks = sorted(R.K.unique())
    for k in ks:
        v = R[R.K == k].pR2_total.to_numpy(float)
        ax.plot(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, "o", mfc="none", mec="0.55", ms=4, mew=0.8)
        ax.plot(k, v.mean(), "o", color="k", ms=6, zorder=4)
    ax.axvline(bs.GLMHMM_K, color=scol[0], lw=1, ls="--", zorder=0)
    ax.set_xticks(ks)
    ax.set_xlabel("number of states K")
    ax.set_ylabel("held-out pseudo-R$^2$ (all outputs)")
    ax.set_title("K sweep: open = fold, filled = mean", fontsize=9)
    ax.text(bs.GLMHMM_K + 0.1, ax.get_ylim()[0] + 0.02, f"K = {bs.GLMHMM_K}", color=scol[0], fontsize=8)
    if P is None:
        fig.suptitle("GLM-HMM parameters not written (run glm_hmm --pooled-all --k K --params-only)", fontsize=9)
        _save(fig, "glmhmm_model", od)
        return
    names = [f"{k}: {roles[k]}" for k in range(P["K"])]
    # (b) emission profile at the average input
    E = bs.glmhmm_emission(P)
    cols = [("outcome", 0, "attempt"), ("outcome", 1, "jaw-only"), ("outcome", 2, "no response"),
            ("precue_lick", 1, "pre-cue lick"), ("iti_lick", 1, "ITI lick"),
            ("run_precue", None, "running pre-cue"), ("run_iti", None, "running ITI")]
    M = np.array([[(1 - E[o][k, 0]) if c is None else E[o][k, c] for (o, c, _n) in cols] for k in range(P["K"])])
    ax = fig.add_subplot(gs[1])
    ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] > 0.55 else "k")
    ax.set_xticks(range(len(cols)), [n for *_x, n in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(P["K"]), names, fontsize=8)
    for t_, k in zip(ax.get_yticklabels(), range(P["K"])):
        t_.set_color(scol[k])
    ax.set_title("model emission probability (average spout, animal)", fontsize=9)
    # (c) P(attempt | spout, state) from the model
    ax = fig.add_subplot(gs[2])
    lat = bs.laterality()
    a0 = next(iter(lat.values()))
    pos = list(oc.POS_ORDER)
    raw = {oc.pos_key(p, a0): p for p in glm_hmm.POSITIONS}
    x = np.arange(len(pos))
    off = cp.epoch_offsets(P["K"], step=0.11)
    for k in range(P["K"]):
        y = [bs.glmhmm_emission(P, raw[q])["outcome"][k, 0] for q in pos]
        ax.plot(x + off[k], y, "o", color=scol[k], ms=6, label=names[k])
    _spout_axis(ax, x, pos, lat)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("model P(attempt)")
    ax.set_title("model P(attempt) by spout and state", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    # (d) transitions
    ax = fig.add_subplot(gs[3])
    A = np.asarray(P["A"], float)
    ax.imshow(np.log10(A), cmap="Greys", vmin=-3, vmax=0, aspect="equal")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.2f}" if A[i, j] >= 0.005 else "<.01", ha="center", va="center", fontsize=7,
                    color="w" if A[i, j] > 0.3 else "k")
    ax.set_xticks(range(P["K"]), range(P["K"]), fontsize=8)
    ax.set_yticks(range(P["K"]), names, fontsize=8)
    for t_, k in zip(ax.get_yticklabels(), range(P["K"])):
        t_.set_color(scol[k])
    ax.set_xlabel("to state")
    ax.set_title("transitions, trial to trial", fontsize=9)
    merged = "; ".join(f"states {' and '.join(map(str, g))} combined in every readout" for g in bs.GLMHMM_MERGE)
    fig.suptitle(f"GLM-HMM K{bs.GLMHMM_K} ({bs.GLMHMM_DESIGN}), one model over all animals and epochs: the identified "
                 f"states ({merged})", fontsize=9)
    _save(fig, "glmhmm_model", od)


MOTIF_LABEL = {"tongue_ext": "tongue extension (mm, 3-D)", "tongue_ml": "tongue lateral (mm, + = contra)",
               "jaw_y": "jaw vertical (cam4 px, down +)", "nose_y": "nose vs mouth (cam4 px)",
               "run": "treadmill (mm/s, push < 0)"}


def fig_arhmm_motifs(C: dict, od: Path, tag: str | None = None) -> None:
    """The identified AR-HMM motifs, one row per reviewed group: example frames from the review reels (mid-syllable,
    one per member state) and the onset-aligned feature traces, pre-stroke sessions."""
    import matplotlib.image as mpimg
    from wfield_local import arhmm_reels as ar
    groups = bs.state_groups(tag)
    pre = sorted(lab for lab in C["usage"].label.unique() if bs.session_epoch(lab) == "pre")
    res, _fail = ak.fan_sessions([(lab, groups, tag) for lab in pre], bs.motif_traces, key=lambda it: it[0])
    per = [r for _i, r in res if r]
    rdir = ar.reels_dir(tag or "pooled_all_s1")
    nf = len(bs.MOTIF_FEATURES)
    nimg = 3
    t = np.arange(int(round(bs.MOTIF_WIN_S[0] * 25)), int(round(bs.MOTIF_WIN_S[1] * 25)) + 1) / 25.0
    gn = list(groups)
    fig, axs = plt.subplots(len(gn), nimg + nf, figsize=(1.55 * nimg + 2.1 * nf, 1.45 * len(gn)), squeeze=False,
                            gridspec_kw={"width_ratios": [0.75] * nimg + [1] * nf})
    rows = []
    for i, g in enumerate(gn):
        S = [r["groups"][g] for r in per if g in r["groups"]]
        frac = np.mean([s["frac"] for s in S]) if S else np.nan
        dur = np.median([s["dur_med"] for s in S]) if S else np.nan
        rows.append({"group": g, "states": "-".join(map(str, groups[g])), "pre_frac": frac, "bout_med_s": dur,
                     "n_sessions": len(S), "n_onsets": int(sum(s["n"] for s in S))})
        # example frames: the first clip of each member state, cycling
        order = []
        for j in range(nimg):
            for k in groups[g]:
                c = sorted((rdir / f"state_{k:02d}").glob("clip_*.png"))
                if j < len(c):
                    order.append(c[j])
        for j in range(nimg):
            ax = axs[i, j]
            ax.axis("off")
            if j < len(order):
                ax.imshow(mpimg.imread(order[j]), cmap="gray")
                ax.text(0.02, 0.02, f"state {order[j].parent.name[6:].lstrip('0') or '0'}", transform=ax.transAxes,
                        fontsize=6, color="w")
        axs[i, 0].text(-0.08, 0.5, f"{g.replace('_', ' ')}\nstates {rows[-1]['states']}\n{100 * frac:.1f} % pre, "
                       f"bout {dur:.2f} s", transform=axs[i, 0].transAxes, ha="right", va="center", fontsize=7)
        for f, feat in enumerate(bs.MOTIF_FEATURES):
            ax = axs[i, nimg + f]
            if S:
                M = np.stack([s["mean"][:, f] for s in S])
                m, se = np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(np.sum(np.isfinite(M), 0).clip(1))
                ax.fill_between(t, m - se, m + se, color="k", alpha=0.08, lw=0)
                ax.plot(t, m, color="k", lw=1.2)
            ax.axvline(0, color="0.6", lw=0.8, ls="--")
            if feat == "run":
                ax.axhline(0, color="0.8", lw=0.6)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(MOTIF_LABEL[feat], fontsize=8)
            if i == len(gn) - 1:
                ax.set_xlabel("s from bout onset", fontsize=7)
    # shared y per feature column, except the treadmill (walking would flatten the push)
    for f in range(nf - 1):
        col = axs[:, nimg + f]
        lo = min(a.get_ylim()[0] for a in col)
        hi = max(a.get_ylim()[1] for a in col)
        for a in col:
            a.set_ylim(lo, hi)
    pd.DataFrame(rows).to_csv(od / "arhmm_motifs.csv", index=False)
    fig.suptitle(f"AR-HMM syllable groups (reviewed, {tag or 'pooled_all_s1'}): example mid-syllable frames (cam4) and "
                 f"onset-aligned features, pre-stroke sessions (mean over {len(per)} sessions +- SEM)", fontsize=9)
    fig.tight_layout()
    save_png_svg(fig, od / "arhmm_motifs.png")
    plt.close(fig)
    print("  wrote arhmm_motifs")


TIM_PERIODS = ("precue", "response", "iti")      # windows always ordered pre-cue, cue / response, then ITI
TIM_PARTS = ("tongue", "jaw", "nose")


def fig_tim(C: dict, od: Path) -> None:
    D = bs.tim_trials("within")
    W = D[D.working.astype(bool)]
    S = W.pivot_table(index=["label", "animal", "epoch"], columns=["period", "part"], values="tim_px").reset_index()
    S.columns = [c if isinstance(c, str) else (c[0] if not c[1] else f"{c[0]}:{c[1]}") for c in S.columns]
    ms = [f"{p}:{q}" for p in TIM_PERIODS for q in TIM_PARTS if f"{p}:{q}" in S]
    epoch_panels(S, ms, {m: f"TIM {m.split(':')[1]}, {m.split(':')[0]} (px)" for m in ms}, "tim_within_by_epoch",
                 "Task-independent movement (within-session fit, cross-validated), working trials: mean residual "
                 "position not predicted by the task, per session", od, nc=3)
    # by engagement (Yin et al.: TIM rises when mice disengage): per session, not-engaged minus engaged, pre-cue
    P = D[D.period == "precue"].copy()
    P["eng"] = np.where(P.role == "engaged", "engaged", np.where(P.role.notna(), "not", None))
    P = P[P.eng.notna()]
    G = P.pivot_table(index=["label", "animal", "epoch", "part"], columns="eng", values="tim_px").reset_index()
    G = G[G.engaged.notna() & G["not"].notna()]
    G["diff"] = G["not"] - G["engaged"]
    H = G.pivot_table(index=["label", "animal", "epoch"], columns="part", values="diff").reset_index()
    epoch_panels(H, [q for q in TIM_PARTS if q in H], {q: f"TIM {q}, pre-cue: not engaged minus engaged (px)"
                                                       for q in TIM_PARTS}, "tim_by_engagement_by_epoch",
                 "Task-independent movement by GLM-HMM engagement (pre-cue period, within-session fit): per session, "
                 "partial + disengaged trials minus engaged trials", od, nc=3)


FAMILIES = {"model": fig_glmhmm_model, "glmcue": fig_glmhmm_cue_traces, "motifs": fig_arhmm_motifs, "occupancy": fig_occupancy, "curves": fig_state_curves, "push": fig_push, "wheel": fig_wheel_traces,
            "arhmm": fig_arhmm, "frozen": fig_frozen, "tim": fig_tim,
            "frozenstats": fig_frozen_stats}


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
