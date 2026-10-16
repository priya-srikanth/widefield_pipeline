"""Pre- vs post-stroke orofacial kinematics across the cohort, from the organised pose outputs (`orofacial_outputs`).

    python -m wfield_local.orofacial_cohort [--run lp_r4_heatmap_20261013__clean_v35] [--jobs 4]

Reads every CURATED session (`analysis_kit.curated_sessions`, the same pre + post set the imaging analyses use -- early
June 6/1-6/5 and 8/5 excluded) that has a processed session folder, reduces each to per-session numbers, and writes

    <LP_analysis>/<run>/cohort/tables/prepost/  session_metrics.csv     one row per session (animal, epoch, day, metrics)
                                                position_metrics.csv    one row per session x spout position
                                                lick_phase.csv          lick-peak-aligned mean protrusion / velocity per
                                                                        session x position (measured-peak licks)
                                                epoch_by_animal.csv     per-animal epoch mean + SEM over sessions (rule 8)
                                                epoch_deltas.csv        paired animal-weighted change vs pre (`boot_delta`)
    <LP_analysis>/<run>/cohort/figures/prepost/ epochs, timecourse (session level), epochs_by_position_<group>,
                                                timecourse_by_position_<group> (group = engagement / kinematics /
                                                accuracy), by_position, angle_tuning, lick_phase, phase_portrait,
                                                mismatch (.png)

Re-run as sessions arrive; every number is recomputed from the session folders (nothing cached here).

POSITIONS are named and ordered by the lesion: close ipsi / close mid / close contra, far ipsi / far mid / far contra
(Priya, 2026-10-16). ipsi / contra come from `animals.yaml stroke_laterality` (all four lesions are LEFT: L = ipsi,
R = contra; PS93 also has a pre-existing right orofacial deficit). Colours follow the widefield figures: positions
`spout_behavior.position_style` (hue = side, lightness = ring), epochs the grey ramp `epoch_figures.EPOCH_GREY`, animals
`config.animal_color`. Error bars are SEM over SESSIONS within an animal (none where an animal has one session).

WHAT EACH METRIC IS (per session; post-cue = cue + 70 ms .. DAQ trial stop, `licks.csv`):
  resp_rate          fraction of trials with >= 1 post-cue lick (engagement as much as ability)
  licks_per_resp     post-cue licks per RESPONDED trial
  post_rate_hz       post-cue licks / summed response-window time
  latency_ms         median time of the first post-cue LP lick's PEAK (comparable to a spout contact)
  latency_contact_ms median time of the first post-cue DAQ spout contact (video-independent)
  ili_ms             median inter-lick interval inside a trial, intervals < 300 ms (within-bout rhythm)
  neighbor_frac      fraction of licks with another lick in the same trial within 250 ms (bout continuation vs
                     isolated licks -- the flanks of the lick-phase plot)
  lick_dur_ms        median lick duration (onset .. offset)
  protrusion_px      median per-lick max protrusion, MEASURED peaks only (`peak_interp` False; Priya 10-15: be wary
                     of quantities read off interpolated points). Compare WITHIN a position: the spouts sit at
                     different image distances from the mouth (far mid ~109 px vs far sides ~132 px, PS92)
  reach_frac         protrusion / mouth-spout distance -- comparable ACROSS positions (> 1 = past the spout)
  vel_protrude       median peak protrusion speed (px/s), measured peaks
  vel_retract        median peak retraction speed (px/s), measured peaks
  contact_frac       fraction of post-cue licks with a DAQ spout contact
  nocontact_frac     1 - contact_frac: the tongue came out and missed (tongue-side mismatch)
  jaw_no_lick_frac   fraction of jaw-QC'd trials with a jaw movement but no tongue lick (jaw-side mismatch,
                     `tongue_jaw_mismatch.csv candidate_no_lick_with_jaw_move`)
  angle_gain_far     executed-angle vs target from the far side positions: (angle far L - far R) / (spout far L - far R)
  ipsi_bias_deg      mean over positions of the median executed peak angle, signed + = toward the IPSILESIONAL side --
                     a target-independent turn of the tongue
  precue_rate_hz     spontaneous licks, position strobe (spout arrived) .. cue + 70 ms, per second (`licks_precue.csv`)
  precue_trial_frac  fraction of trials with >= 1 pre-cue lick
  iti_rate_hz        inter-trial licks per second (`licks_iti.csv`)
  peak_interp_frac   fraction of post-cue licks whose peak is interpolated / imputed (tracking-quality context)
At n=4, the per-animal table is printed before any bootstrap (CLAUDE.md rule 8).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN = "lp_r4_heatmap_20261013__clean_v35"
EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")
POS_ORDER = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
METRICS = ("resp_rate", "licks_per_resp", "post_rate_hz", "precue_rate_hz", "precue_trial_frac", "iti_rate_hz",
           "latency_ms", "latency_contact_ms", "ili_ms", "neighbor_frac", "lick_dur_ms",
           "protrusion_px", "reach_frac", "vel_protrude", "vel_retract",
           "contact_frac", "nocontact_frac", "jaw_no_lick_frac", "angle_gain_far", "ipsi_bias_deg", "peak_interp_frac")
LABEL = {"resp_rate": "responded trials (frac)", "licks_per_resp": "licks / responded trial",
         "post_rate_hz": "post-cue lick rate (Hz)", "precue_rate_hz": "pre-cue lick rate (Hz)",
         "precue_trial_frac": "trials with pre-cue lick (frac)", "iti_rate_hz": "ITI lick rate (Hz)",
         "latency_ms": "first LP lick peak (ms from cue)", "latency_contact_ms": "first spout contact (ms from cue)",
         "ili_ms": "within-bout ILI (ms)", "lick_dur_ms": "lick duration (ms)",
         "neighbor_frac": "licks with a neighbour < 250 ms (frac)",
         "protrusion_px": "max protrusion, measured (px)", "reach_frac": "reach (protrusion / spout distance)",
         "vel_protrude": "peak protrusion speed (px/s)", "vel_retract": "peak retraction speed (px/s)",
         "contact_frac": "licks with spout contact (frac)", "nocontact_frac": "no-contact licks (frac)",
         "jaw_no_lick_frac": "jaw move, no lick (frac trials)", "angle_gain_far": "angle gain, far ipsi vs contra",
         "ipsi_bias_deg": "tongue angle bias, + = ipsi (deg)", "peak_interp_frac": "interpolated peaks (frac)"}
LABEL["peak_angle_deg"] = "executed peak angle (deg; + = L / image right)"
#: per-position figures, one file per group (rows = metric, columns = the six positions)
POS_GROUPS = {"engagement": ("resp_rate", "licks_per_resp", "precue_rate_hz", "ili_ms", "neighbor_frac",
                             "latency_ms", "latency_contact_ms"),
              "kinematics": ("protrusion_px", "reach_frac", "vel_protrude", "vel_retract", "lick_dur_ms"),
              "accuracy": ("contact_frac", "nocontact_frac", "jaw_no_lick_frac", "peak_angle_deg")}
#: epoch colours for this module's line plots (angle tuning, by position, lick phase): the grey ramp's light "pre"
#: disappears on them (Priya 10-16: "ok to set another convention there"). Animal colours stay `config.animal_color`.
#: Chosen to collide with NO animal colour (PS92 blue, PS93 red, PS94 orange, PS95 green -- Priya 10-16: "confusing for
#: animals colors to be the same as the pre/acute/subacute/chronic colors").
EPOCH_LINE = {"pre": "k", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}
ILI_MAX_MS = 300.0
FRAME_MS = 4.0
PHASE_HALF = 38                       # frames each side of the lick peak (+-152 ms)
FPS = 250.0


def pos_key(position: str, laterality: str) -> str:
    """`close_L` -> 'close_ipsi' / 'close_mid' / 'close_contra', relative to the lesion side."""
    ring, s = position.split("_", 1)
    side = "mid" if s == "center" else ("ipsi" if s == laterality else "contra")
    return f"{ring}_{side}"


def pos_disp(key: str) -> str:
    return key.replace("_", " ")


def raw_position(key: str, laterality: str) -> str:
    """Inverse of `pos_key` -- for the position palette, which is keyed by the rig's names."""
    ring, side = key.split("_")
    if side == "mid":
        return f"{ring}_center"
    other = "R" if laterality == "L" else "L"
    return f"{ring}_{laterality if side == 'ipsi' else other}"


def _windows(sd: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-trial durations: pre-cue, response, ITI seconds; per-trial first row + first t_ms, for frame lookups)."""
    w = np.load(sd / "windows.npz")
    ix = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "t_ms", "strobe_ms", "stop_ms")})
    ix["row"] = np.arange(len(ix))
    g = ix.groupby("trial_id", sort=True).agg(t0=("t_ms", "min"), t1=("t_ms", "max"), strobe=("strobe_ms", "first"),
                                               stop=("stop_ms", "first"), row0=("row", "min"))
    t1 = g.t1 + FRAME_MS
    pre = (70.0 - g.strobe).where(np.isfinite(g.strobe), np.nan).clip(lower=0)
    resp = (g.stop.clip(upper=t1) - 70.0).clip(lower=0)
    iti = (g.strobe - g.t0).clip(lower=0).fillna(0) + (t1 - g.stop).clip(lower=0).fillna(0)
    return pd.DataFrame({"pre_s": pre / 1000, "resp_s": resp / 1000, "iti_s": iti / 1000}), g[["row0", "t0"]]


def _first_contacts(animal: str, date: str) -> pd.Series:
    """{trial_id: first DAQ spout-contact time, ms from the cue, inside cue + 70 ms .. trial stop}."""
    from scripts import pose_kinematics_demo as D
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    lick = np.sort(D.daq_lick_onsets_s(animal, date, rv))
    b = TW.trial_bounds(animal, date, rv)
    out = {}
    for t, c, s in zip(b.trial_id, b.cue_s, b.stop_s):
        if np.isfinite(c):
            k = np.searchsorted(lick, c + 0.07)
            if k < len(lick) and (not np.isfinite(s) or lick[k] <= s):
                out[t] = (lick[k] - c) * 1000.0
    return pd.Series(out, dtype=float)


#: below this protrusion the tongue direction is not defined (tip at the lips / mouth fill = 0 px): angle = NaN
ANGLE_MIN_PX = 20.0


def _lick_phase(sd: Path, meas: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """{position: (mean protrusion, mean velocity, mean angle, n licks)} around each measured lick PEAK (+-152 ms).
    Protrusion = distance of the cleaned tongue from the mouth, which reproduces `protrusion_px` at the peak frame;
    angle = `tongue_kinematics.compute_signed_tongue_angle_deg` (+ = image right = the L positions), NaN while the tip
    is within ANGLE_MIN_PX of the mouth."""
    from wfield_local import tongue_kinematics as tk
    c = np.load(sd / "clean.npz")
    tx, ty = c["tongue_x"].astype(float), c["tongue_y"].astype(float)
    dist = np.hypot(tx, ty)
    o = c["tongue_origin"]
    ang = tk.compute_signed_tongue_angle_deg(tx + o[0], ty + o[1],
                                             tk.SpoutFrame(tuple(c["spout_origin"]), tuple(c["spout_ap_axis"])))
    ang = np.where(dist >= ANGLE_MIN_PX, ang, np.nan)
    n = len(dist)
    r = (rows.row0.reindex(meas.trial_id).to_numpy() +
         np.round((meas.t_ms.to_numpy() - rows.t0.reindex(meas.trial_id).to_numpy()) / FRAME_MS)).astype(int)
    off = np.arange(-PHASE_HALF, PHASE_HALF + 1)
    ok = (r - PHASE_HALF >= 0) & (r + PHASE_HALF < n)
    out = {}
    for pos in meas.position.unique():
        m = ok & (meas.position.to_numpy() == pos)
        if m.sum() < 5:
            continue
        idx = r[m][:, None] + off[None, :]
        seg = dist[idx]
        vel = np.gradient(seg, axis=1) * FPS
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[pos] = (np.nanmean(seg, 0), np.nanmean(vel, 0), np.nanmean(ang[idx], 0), int(m.sum()))
    return out


NEIGHBOR_MS = 250.0


def _neighbor_frac(g: pd.DataFrame) -> float:
    """Fraction of licks with another lick of the same trial within NEIGHBOR_MS (either side)."""
    if not len(g):
        return np.nan
    o = g.sort_values(["trial_id", "t_ms"])
    d = o.groupby("trial_id").t_ms.diff()
    prev_ok = d < NEIGHBOR_MS
    next_ok = (-o.groupby("trial_id").t_ms.diff(-1)) < NEIGHBOR_MS
    return float((prev_ok | next_ok).mean())


def session_metrics(job):
    """(session row, position rows, lick-phase rows) for one session folder; module-level for `fan_sessions`."""
    sd, animal, date, epoch, day, lat = job
    rd = lambda n: pd.read_csv(sd / n) if (sd / n).exists() else pd.DataFrame()   # noqa: E731
    L, P, Q, T, M = rd("licks.csv"), rd("licks_precue.csv"), rd("licks_iti.csv"), rd("trials.csv"), \
        rd("tongue_jaw_mismatch.csv")
    if not len(T) or "t_ms" not in L:
        return None
    dur, rows = _windows(sd)
    contact1 = _first_contacts(animal, date)
    meas = L[~L.peak_interp.astype(bool)]
    by_trial = L.groupby("trial_id")
    resp_trials = by_trial.size()
    ili = L.sort_values(["trial_id", "t_ms"]).groupby("trial_id").t_ms.diff()
    pos_med = meas.groupby("position")[["peak_angle_deg", "spout_angle_deg"]].median()
    far = pos_med.reindex([f"far_{lat}", f"far_{'R' if lat == 'L' else 'L'}"])          # ipsi, contra
    dsp = far.spout_angle_deg.iloc[0] - far.spout_angle_deg.iloc[1]
    gain_far = (float((far.peak_angle_deg.iloc[0] - far.peak_angle_deg.iloc[1]) / dsp)
                if far.notna().all().all() and abs(dsp) > 10 else np.nan)
    # angles are + toward image right = the L positions; flip so + = ipsilesional whatever the lesion side
    sgn = 1.0 if lat == "L" else -1.0
    jq = M[M.jaw_pass_qc.astype(bool)] if len(M) and "jaw_pass_qc" in M else pd.DataFrame()
    n_tr = int(len(T))

    def kin(g, gm):
        return {"protrusion_px": float(gm.protrusion_max_measured_px.median()) if len(gm) else np.nan,
                "reach_frac": float(gm.reach_frac.median()) if len(gm) else np.nan,
                "vel_protrude": float(gm.max_velocity_y_px_per_s.median()) if len(gm) else np.nan,
                "vel_retract": float(gm.max_retract_velocity_y_px_per_s.median()) if len(gm) else np.nan,
                "lick_dur_ms": float((g.off_ms - g.on_ms).median()) if len(g) else np.nan,
                "contact_frac": float(g.contact.astype(float).mean()) if len(g) else np.nan,
                "neighbor_frac": _neighbor_frac(g),
                "latency_ms": float(g.groupby("trial_id").t_ms.min().median()) if len(g) else np.nan}

    row = {"animal": animal, "date": date, "label": f"{animal}_{date[4:]}", "epoch": epoch, "day": day,
           "n_trials": n_tr, "n_licks": int(len(L)), "n_precue_licks": int(len(P)), "n_iti_licks": int(len(Q)),
           "resp_rate": len(resp_trials) / n_tr,
           "licks_per_resp": float(resp_trials.mean()) if len(resp_trials) else np.nan,
           "post_rate_hz": len(L) / dur.resp_s.sum() if dur.resp_s.sum() > 0 else np.nan,
           "precue_rate_hz": len(P) / dur.pre_s.sum() if dur.pre_s.sum() > 0 else np.nan,
           "precue_trial_frac": (P.trial_id.nunique() / n_tr) if len(P) else 0.0,
           "iti_rate_hz": len(Q) / dur.iti_s.sum() if dur.iti_s.sum() > 0 else np.nan,
           "latency_contact_ms": float(contact1.median()) if len(contact1) else np.nan,
           "ili_ms": float(ili[ili < ILI_MAX_MS].median()) if ili.notna().any() else np.nan,
           **kin(L, meas),
           "jaw_no_lick_frac": float(jq.candidate_no_lick_with_jaw_move.astype(float).mean()) if len(jq) else np.nan,
           "angle_gain_far": gain_far,
           "ipsi_bias_deg": sgn * float(pos_med.peak_angle_deg.mean()) if len(pos_med) else np.nan,
           "peak_interp_frac": float(L.peak_interp.astype(float).mean())}
    row["nocontact_frac"] = 1.0 - row["contact_frac"]
    prow = []
    tpos = T.set_index("trial_id").position
    for pos, g in L.groupby("position"):
        tids = tpos[tpos == pos].index
        jp = jq[jq.position == pos] if len(jq) else jq
        rt = g.groupby("trial_id").size()
        ili_p = g.sort_values(["trial_id", "t_ms"]).groupby("trial_id").t_ms.diff()
        pre_s = dur.pre_s.reindex(tids).sum()
        r = {"animal": animal, "date": date, "epoch": epoch, "day": day, "position": pos, "pos": pos_key(pos, lat),
             "n_trials": int(len(tids)), "n_licks": int(len(g)), **kin(g, meas[meas.position == pos]),
             "resp_rate": len(rt) / len(tids) if len(tids) else np.nan,
             "licks_per_resp": float(rt.mean()) if len(rt) else np.nan,
             "precue_rate_hz": (int(P.trial_id.isin(tids).sum()) / pre_s) if pre_s > 0 else np.nan,
             "ili_ms": float(ili_p[ili_p < ILI_MAX_MS].median()) if ili_p.notna().any() else np.nan,
             "latency_contact_ms": float(contact1.reindex(tids).median()),
             "jaw_no_lick_frac": float(jp.candidate_no_lick_with_jaw_move.astype(float).mean()) if len(jp) else np.nan,
             "peak_angle_deg": float(meas[meas.position == pos].peak_angle_deg.median()),
             "spout_angle_deg": float(meas[meas.position == pos].spout_angle_deg.median()),
             "spout_dist_px": float(g.spout_dist_px.median())}
        r["nocontact_frac"] = 1.0 - r["contact_frac"]
        prow.append(r)
    t = np.arange(-PHASE_HALF, PHASE_HALF + 1) * FRAME_MS
    sp = {r["position"]: r["spout_angle_deg"] for r in prow}
    ph = [{"animal": animal, "date": date, "epoch": epoch, "position": pos, "pos": pos_key(pos, lat), "t_ms": tt,
           "protrusion_px": d, "velocity_px_s": v, "angle_deg": an, "delta_angle_deg": an - sp.get(pos, np.nan),
           "n_licks": n}
          for pos, (dd, vv, aa, n) in _lick_phase(sd, meas, rows).items() for tt, d, v, an in zip(t, dd, vv, aa)]
    return row, prow, ph


def collect(rv, run: str = RUN, jobs: int | None = None):
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
    return (pd.DataFrame([r for r, _, _ in rows]), pd.DataFrame([p for _, ps, _ in rows for p in ps]),
            pd.DataFrame([x for _, _, ph in rows for x in ph]), lat)


def _sem(x) -> float:
    x = pd.Series(x).dropna()
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def epoch_tables(S: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-animal epoch means (+ SEM over sessions), and the paired animal-weighted change vs pre (`boot_delta`)."""
    from wfield_local import analysis_kit as ak
    g = S.groupby(["animal", "epoch"])
    A = g[list(METRICS)].mean()
    A = A.join(g[list(METRICS)].agg(_sem).add_suffix("_sem")).reset_index()
    A.insert(2, "n_sessions", g.size().to_numpy())
    rng = np.random.default_rng(seed)
    out = []
    for m in METRICS:
        pre = {a: x[m].dropna().tolist() for a, x in S[S.epoch == "pre"].groupby("animal", sort=True)}
        for ep in EPOCH_ORDER[1:]:
            post = {a: x[m].dropna().tolist() for a, x in S[S.epoch == ep].groupby("animal", sort=True)}
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

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _palettes():
    from wfield_local import config
    from wfield_local.epoch_figures import EPOCH_GREY
    return config.animal_color(), dict(EPOCH_GREY)


def _pos_color(key: str, lat: str):
    from wfield_local.spout_behavior import position_style
    return position_style(raw_position(key, lat))[0]


def _grid(n: int, nc: int, w: float, h: float):
    plt = _plt()
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(w * nc, h * nr), squeeze=False)
    for ax in axs.flat[n:]:
        ax.axis("off")
    return plt, fig, axs


def _save(plt, fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_epochs(A: pd.DataFrame, D: pd.DataFrame, path: Path, title: str) -> None:
    col, _ = _palettes()
    plt, fig, axs = _grid(len(METRICS), 5, 3.5, 2.8)
    xi = {e: i for i, e in enumerate(EPOCH_ORDER)}
    for ax, m in zip(axs.flat, METRICS):
        for k, (a, g) in enumerate(A.groupby("animal")):
            g = g[g.epoch.isin(xi)].assign(x=lambda d: d.epoch.map(xi)).sort_values("x")
            ax.errorbar(g.x + (k - 1.5) * 0.06, g[m], yerr=g[f"{m}_sem"], fmt="o-", color=col.get(a, "k"), ms=4,
                        lw=1.2, capsize=2, elinewidth=0.8, label=a)
        lab = []
        for e in EPOCH_ORDER[1:]:
            r = D[(D.metric == m) & (D.epoch == e)]
            if len(r) and np.isfinite(r.lo.iloc[0]):
                star = "*" if r.lo.iloc[0] > 0 or r.hi.iloc[0] < 0 else ""
                lab.append(f"{e[:3]} {r.delta.iloc[0]:+.3g}{star}")
        ax.set_xticks(range(len(EPOCH_ORDER)), EPOCH_ORDER, fontsize=7)
        ax.set_title(LABEL[m] + ("\n" + "; ".join(lab) if lab else "\n"), fontsize=7.5)
        ax.tick_params(labelsize=7)
    axs.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title + "\nper-animal epoch mean +- SEM over sessions; title: paired change vs pre (animal-weighted "
                 "bootstrap, * = 95% CI excludes 0; only epochs with >= 2 animals)", fontsize=9)
    _save(plt, fig, path)


def fig_timecourse(S: pd.DataFrame, path: Path, title: str) -> None:
    col, _ = _palettes()
    plt, fig, axs = _grid(len(METRICS), 5, 3.5, 2.6)
    for ax, m in zip(axs.flat, METRICS):
        for a, g in S.sort_values("day").groupby("animal"):
            ax.plot(g.day, g[m], "o-", ms=2.5, lw=0.9, color=col.get(a, "k"), label=a)
        ax.axvline(0.5, color="0.4", ls="--", lw=0.8)
        ax.set_title(LABEL[m], fontsize=8)
        ax.tick_params(labelsize=7)
    axs.flat[0].legend(fontsize=7, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("days since stroke", fontsize=8)
    fig.suptitle(title + "\none point per session; dashed = lesion", fontsize=9)
    _save(plt, fig, path)


def _by_pos_panel(ax, P: pd.DataFrame, m: str, ecol: dict, lat: str) -> None:
    x = np.arange(len(POS_ORDER))
    for e in EPOCH_ORDER:
        g = P[P.epoch == e]
        if not len(g):
            continue
        mu = g.groupby("pos")[m].mean().reindex(POS_ORDER)
        se = g.groupby("pos")[m].agg(_sem).reindex(POS_ORDER)
        ax.errorbar(x, mu.to_numpy(), yerr=se.to_numpy(), fmt="o-", color=ecol[e], ms=3.5, lw=1.2, capsize=2,
                    elinewidth=0.8, label=f"{e} ({g.date.nunique()})", )
    ax.axvline(2.5, color="0.85", lw=0.8)
    ax.set_xticks(x, [pos_disp(p).replace(" ", "\n") for p in POS_ORDER], fontsize=6)
    for t, p in zip(ax.get_xticklabels(), POS_ORDER):
        t.set_color(_pos_color(p, lat))
    ax.tick_params(labelsize=7)


def _per_animal_figure(P: pd.DataFrame, rows: tuple, lat: dict, path: Path, title: str, note: str) -> None:
    ecol = EPOCH_LINE
    animals = sorted(P.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(rows), len(animals), figsize=(3.4 * len(animals), 2.4 * len(rows)), squeeze=False)
    for j, a in enumerate(animals):
        for i, m in enumerate(rows):
            _by_pos_panel(axs[i][j], P[P.animal == a], m, ecol, lat[a])
            if i == 0:
                axs[i][j].set_title(a, fontsize=9)
            if j == 0:
                axs[i][j].set_ylabel(LABEL[m], fontsize=7.5)
        axs[0][j].legend(fontsize=6, frameon=False)
    fig.suptitle(title + "\n" + note, fontsize=9)
    _save(plt, fig, path)


def fig_by_position(Pm, lat, path, title):
    _per_animal_figure(Pm, ("contact_frac", "reach_frac", "protrusion_px", "latency_ms", "latency_contact_ms",
                            "vel_protrude", "vel_retract"), lat, path, title,
                       "by spout position (close ipsi / mid / contra | far ipsi / mid / contra), epoch mean +- SEM over "
                       "sessions; compare protrusion WITHIN a position (spouts sit at different image distances), "
                       "reach across positions")


def fig_mismatch(Pm, lat, path, title):
    _per_animal_figure(Pm, ("nocontact_frac", "jaw_no_lick_frac"), lat, path, title,
                       "tongue-jaw mismatch: no-contact licks (tongue out, spout missed) and jaw movement without a "
                       "lick; epoch mean +- SEM over sessions")


def fig_angle_tuning(Pm: pd.DataFrame, lat: dict, path: Path, title: str) -> None:
    ecol = EPOCH_LINE
    animals = sorted(Pm.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(1, len(animals), figsize=(3.6 * len(animals), 4.0), squeeze=False)
    for ax, a in zip(axs[0], animals):
        g = Pm[Pm.animal == a]
        for e in EPOCH_ORDER:
            ge = g[g.epoch == e]
            if not len(ge):
                continue
            # close and far sit at the SAME image angle (~+-40 deg): one line per ring (far solid, close dashed)
            for ring, ls in (("far", "-"), ("close", "--")):
                gr = ge[ge.pos.str.startswith(ring)]
                mu = gr.groupby("pos")[["spout_angle_deg", "peak_angle_deg"]].mean().sort_values("spout_angle_deg")
                se = gr.groupby("pos").peak_angle_deg.agg(_sem).reindex(mu.index)
                ax.plot(mu.spout_angle_deg, mu.peak_angle_deg, ls, color=ecol[e], lw=1.3,
                        label=f"{e} ({ge.date.nunique()})" if ring == "far" else None)
                for p, r in mu.iterrows():
                    ax.errorbar(r.spout_angle_deg, r.peak_angle_deg, yerr=se[p], fmt="o", ms=4, capsize=2,
                                color=ecol[e], elinewidth=0.8)
        ax.plot([-50, 50], [-50, 50], color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(-50, 50)
        ax.set_title(a, fontsize=9)
        ax.set_xlabel("spout angle (deg; + = image right = L positions)", fontsize=7.5)
        ax.legend(fontsize=6.5, frameon=False)
        ax.tick_params(labelsize=7)
    axs[0][0].set_ylabel("executed peak tongue angle (deg)", fontsize=8)
    fig.suptitle(title + "\nexecuted vs target angle: per-position median (measured peaks), epoch mean +- SEM over "
                 "sessions\nfar = solid, close = dashed; grey = identity; left = contra positions, right = ipsi",
                 fontsize=9)
    _save(plt, fig, path)


def _phase_fig(PH: pd.DataFrame, lat: dict, path: Path, title: str, portrait: bool) -> None:
    ecol = EPOCH_LINE
    animals = sorted(PH.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(animals), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER), 2.3 * len(animals)),
                            squeeze=False, sharex=True, sharey="row" if not portrait else False)
    for i, a in enumerate(animals):
        for j, p in enumerate(POS_ORDER):
            ax = axs[i][j]
            for e in EPOCH_ORDER:
                g = PH[(PH.animal == a) & (PH.pos == p) & (PH.epoch == e)]
                if not len(g):
                    continue
                piv_d = g.pivot_table(index="date", columns="t_ms", values="protrusion_px")
                piv_v = g.pivot_table(index="date", columns="t_ms", values="velocity_px_s")
                t = piv_d.columns.to_numpy()
                md, mv = piv_d.mean().to_numpy(), piv_v.mean().to_numpy()
                if portrait:
                    ax.plot(md, mv / 1000, color=ecol[e], lw=1.3, label=f"{e} ({len(piv_d)})")
                    ax.plot(md[t == 0], mv[t == 0] / 1000, "o", color=ecol[e], ms=3)
                else:
                    sd = (piv_d.std(ddof=1) / np.sqrt(len(piv_d))).to_numpy() if len(piv_d) > 1 else None
                    ax.plot(t, md, color=ecol[e], lw=1.3, label=f"{e} ({len(piv_d)})")
                    if sd is not None:
                        ax.fill_between(t, md - sd, md + sd, color=ecol[e], alpha=0.25, lw=0)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=8.5, color=_pos_color(p, lat[a]))
            if j == 0:
                ax.set_ylabel(f"{a}\n" + ("speed (px/ms)" if portrait else "protrusion (px)"), fontsize=8)
            if portrait:
                ax.axhline(0, color="0.85", lw=0.8)
            else:
                ax.axvline(0, color="0.85", lw=0.8)
            ax.tick_params(labelsize=6.5)
        axs[i][0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("protrusion (px)" if portrait else "ms from lick peak", fontsize=7.5)
    fig.suptitle(title + ("\nlick phase portrait: protrusion speed vs protrusion, mean lick (measured peaks) per epoch; "
                          "dot = peak" if portrait else
                          "\nlick-peak-aligned protrusion, mean lick (measured peaks), epoch mean +- SEM over sessions"),
                 fontsize=9)
    _save(plt, fig, path)


def fig_epochs_by_position(Pm: pd.DataFrame, lat: dict, metrics: tuple, path: Path, title: str) -> None:
    """Rows = metric, columns = the six positions; x = epoch; one line per animal, mean +- SEM over sessions."""
    col, _ = _palettes()
    plt = _plt()
    fig, axs = plt.subplots(len(metrics), len(POS_ORDER), figsize=(2.5 * len(POS_ORDER), 2.2 * len(metrics)),
                            squeeze=False, sharey="row")
    for j, p in enumerate(POS_ORDER):
        for i, m in enumerate(metrics):
            ax = axs[i][j]
            for k, (a, g) in enumerate(Pm[Pm.pos == p].groupby("animal")):
                mu = g.groupby("epoch")[m].mean().reindex(EPOCH_ORDER)
                se = g.groupby("epoch")[m].agg(_sem).reindex(EPOCH_ORDER)
                x = np.arange(len(EPOCH_ORDER)) + (k - 1.5) * 0.07
                ok = mu.notna().to_numpy()
                ax.errorbar(x[ok], mu[ok], yerr=se[ok], fmt="o-", color=col.get(a, "k"), ms=3.5, lw=1.1, capsize=2,
                            elinewidth=0.7, label=a)
            ax.set_xticks(range(len(EPOCH_ORDER)), [e[:3] for e in EPOCH_ORDER], fontsize=6.5)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=9, color=_pos_color(p, "L"))
            if j == 0:
                ax.set_ylabel(LABEL[m], fontsize=7)
    axs[0][0].legend(fontsize=6, frameon=False)
    fig.suptitle(title + "\nby spout position: per-animal epoch mean +- SEM over sessions", fontsize=9)
    _save(plt, fig, path)


def fig_timecourse_by_position(Pm: pd.DataFrame, metrics: tuple, path: Path, title: str) -> None:
    """Rows = metric, columns = the six positions; x = days since stroke; one line per animal (one point per session)."""
    col, _ = _palettes()
    plt = _plt()
    fig, axs = plt.subplots(len(metrics), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER), 2.1 * len(metrics)),
                            squeeze=False, sharey="row", sharex=True)
    for j, p in enumerate(POS_ORDER):
        for i, m in enumerate(metrics):
            ax = axs[i][j]
            for a, g in Pm[Pm.pos == p].sort_values("day").groupby("animal"):
                ax.plot(g.day, g[m], "o-", ms=2.2, lw=0.8, color=col.get(a, "k"), label=a)
            ax.axvline(0.5, color="0.6", ls="--", lw=0.7)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=9, color=_pos_color(p, "L"))
            if j == 0:
                ax.set_ylabel(LABEL[m], fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("days since stroke", fontsize=7)
    axs[0][0].legend(fontsize=6, frameon=False)
    fig.suptitle(title + "\nby spout position, one point per session; dashed = lesion", fontsize=9)
    _save(plt, fig, path)


def fig_lick_phase_value(PH: pd.DataFrame, lat: dict, col: str, ylabel: str, path: Path, title: str,
                         note: str, zero_line: bool = True) -> None:
    """Rows = animals, columns = the six positions: ``col`` around the lick peak, epoch mean +- SEM over sessions."""
    ecol = EPOCH_LINE
    animals = sorted(PH.animal.unique())
    plt = _plt()
    fig, axs = plt.subplots(len(animals), len(POS_ORDER), figsize=(2.6 * len(POS_ORDER), 2.3 * len(animals)),
                            squeeze=False, sharex=True, sharey=True)
    for i, a in enumerate(animals):
        for j, p in enumerate(POS_ORDER):
            ax = axs[i][j]
            for e in EPOCH_ORDER:
                g = PH[(PH.animal == a) & (PH.pos == p) & (PH.epoch == e)]
                if not len(g):
                    continue
                piv = g.pivot_table(index="date", columns="t_ms", values=col)
                t, mu = piv.columns.to_numpy(), piv.mean().to_numpy()
                ax.plot(t, mu, color=ecol[e], lw=1.3, label=f"{e} ({len(piv)})")
                if len(piv) > 1:
                    se = (piv.std(ddof=1) / np.sqrt(piv.notna().sum())).to_numpy()
                    ax.fill_between(t, mu - se, mu + se, color=ecol[e], alpha=0.2, lw=0)
            ax.axvline(0, color="0.85", lw=0.8)
            if zero_line:
                ax.axhline(0, color="0.85", lw=0.8)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(pos_disp(p), fontsize=8.5, color=_pos_color(p, lat[a]))
            if j == 0:
                ax.set_ylabel(f"{a}\n{ylabel}", fontsize=8)
        axs[i][0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("ms from lick peak", fontsize=7.5)
    fig.suptitle(title + "\n" + note, fontsize=9)
    _save(plt, fig, path)


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
    S, Pm, PH, lat = collect(rv, a.run, a.jobs)
    A, D = epoch_tables(S)
    pd.set_option("display.width", 250)
    print(f"{len(S)} curated sessions with pose outputs:")
    print(S.groupby(["animal", "epoch"]).size().unstack(fill_value=0).reindex(columns=list(EPOCH_ORDER), fill_value=0))
    print("\nPER-ANIMAL epoch means (rule 8 -- read before the deltas):")
    print(A.set_index(["animal", "epoch"])[["n_sessions", *METRICS]].round(3).to_string())
    print("\nchange vs pre (animal-weighted; CI only with >= 2 animals):")
    print(D.round(4).to_string(index=False))
    S.to_csv(tables / "session_metrics.csv", index=False)
    Pm.to_csv(tables / "position_metrics.csv", index=False)
    PH.to_csv(tables / "lick_phase.csv", index=False)
    A.to_csv(tables / "epoch_by_animal.csv", index=False)
    D.to_csv(tables / "epoch_deltas.csv", index=False)
    animals = ", ".join(f"{x} {int((S.animal == x).sum())}" for x in sorted(S.animal.unique()))
    t = f"Orofacial kinematics, pre vs post stroke -- {a.run} -- sessions: {animals}"
    fig_epochs(A, D, figs / "epochs.png", t)
    fig_timecourse(S, figs / "timecourse.png", t)
    fig_by_position(Pm, lat, figs / "by_position.png", t)
    fig_mismatch(Pm, lat, figs / "mismatch.png", t)
    fig_angle_tuning(Pm, lat, figs / "angle_tuning.png", t)
    for name, ms in POS_GROUPS.items():
        fig_epochs_by_position(Pm, lat, ms, figs / f"epochs_by_position_{name}.png", t)
        fig_timecourse_by_position(Pm, ms, figs / f"timecourse_by_position_{name}.png", t)
    _phase_fig(PH, lat, figs / "lick_phase.png", t, portrait=False)
    _phase_fig(PH, lat, figs / "phase_portrait.png", t, portrait=True)
    fig_lick_phase_value(PH, lat, "angle_deg", "tongue angle (deg)", figs / "lick_phase_angle.png", t,
                         "lick-peak-aligned tongue angle (+ = image right = L / ipsi positions; undefined within "
                         f"{ANGLE_MIN_PX:.0f} px of the mouth), mean lick (measured peaks), epoch mean +- SEM over sessions")
    fig_lick_phase_value(PH, lat, "delta_angle_deg", "angle - spout angle (deg)", figs / "lick_phase_angle_delta.png",
                         t, "lick-peak-aligned angle ERROR: tongue angle minus that position's spout angle (0 = on "
                         "target; + = toward image right / L), epoch mean +- SEM over sessions")
    (tables / "provenance.json").write_text(json.dumps({"run": a.run, "n_sessions": int(len(S)),
                                                        "code_commit": oo.code_commit(),
                                                        "sessions": S.label.tolist()}, indent=1), encoding="utf-8")
    print(f"\ntables -> {tables}\nfigures -> {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
