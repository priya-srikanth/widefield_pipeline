"""Is the CHRONIC neural data stable, per animal, per session -- the same test behaviour gets?

Priya, 2026-09-28: *"behavior has stabilized so I'm wondering if we should deem the experiment
'complete' and stop recording (and sac the animals for histology). But I want to ensure the neural
data is also stable."* That is an irreversible decision, so this is deliberately NOT a new test. It
applies the rule that already sets `chronic_from` from hit rate -- `epochs._plateau_index`: a tail
that is FLAT (total drift over the window <= K_DRIFT x pre-stroke SD, one-sided so only a still-rising
series fails) and SETTLED (residual <= K_RES x pre-stroke SD) -- to every neural readout the deck
reports, per animal, expressed against that animal's own pre-stroke sessions.

READOUTS, and where each per-session value comes from:
  decoder frozen / refit / reorganisation G   recovery_trajectory_matched_cue_working.csv (has days)
  encoder ceiling / frozen-matched / gain     epoch_11c_* / epoch_11amp_*  _sessions.csv
  best-match accuracy                          epoch_10_best_match_acc_*    _sessions.csv
  crossnobis distance from own pre template    epoch_8diag_matrices_*        _sessions.csv
  behaviour far-contra hit rate                epoch_1b_behaviour_*         _sessions.csv
  cue-evoked map amplitude per position        computed here from the per-session npz that
                                               `epoch_15_evoked_CUEINCREMENT` aggregates

THE `_sessions.csv` ROWS CARRY NO LABEL. They are written in the caller's list order, which is
`load_sessions` date order; this was VERIFIED on 2026-09-28 by the PS92_0922 outlier landing in
slot 8 of 9 in every encoder/decoder sidecar. If a family's row count disagrees with the session
count for an (animal, epoch), that family is skipped for that group and says so -- never guessed.

THE ENCODER / TEMPLATE / CROSSNOBIS SIDECARS CARRY ONE POOLED PRE ROW, not per-session pre values,
so no pre-stroke SD exists for them. For those the pooled pre is the level and the animal's own
subacute+chronic scatter is the tolerance -- a weaker test that cannot fail "settled" by its own
construction, so for those rows read the DRIFT verdict and the level, not the settled flag. The
table marks them (`pre_sd_source`).

EXCLUDE names sessions whose imaging is known-bad and which are reported BOTH WAYS (with and
without). It held PS92_0922 from 2026-09-28 until that afternoon: as first processed it was a
frame<->DAQ misalignment of 154 frames (docs/EXPERIMENT_ERRORS.md, 2026-09-22: the DAQ was started
2.466 s after the camera and the relabel mapped the surplus to the wrong end), so both task decoders
and the encoder read at chance while behaviour was perfect -- the largest excursion in every PS92
chronic series, and instability that was not there. The session was re-preprocessed with the fixed
relabel and verified the same day (cue transient at +0.42 s, haemodynamic fit sign restored), so
it is back in. The mechanism stays so the next such session can be handled the same way.

Outputs: <labcams>/chronic_stability/chronic_stability.{csv,png} -- every plotted number is in the
CSV. A cohort p-value is NOT printed (four animals); consistency across animals is the evidence, as
everywhere else in this deck.

CLI::

    conda activate locanmf
    python -m scripts.chronic_stability            # writes to the share
    python -m scripts.chronic_stability --out DIR
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.epochs import (CHRONIC_K_DRIFT, CHRONIC_K_RES, _is_flat, _plateau_index, _pstdev,
                                 days_since_stroke, epoch_of)
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

warnings.filterwarnings("ignore")

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EXCLUDE: set[str] = set()      # was {"PS92_0922"} until its 2026-09-28 redo; see the module docstring

#: sidecar file, position filter, and whether its pre rows are per-session
SIDECARS = {
    "behaviour far-contra hit": ("epoch_1b_behaviour_by_position_sessions.csv", "fC"),
    "encoder ceiling (refit)": ("epoch_11c_encoder_ceiling_cue_working_sessions.csv", "ceiling"),
    "encoder frozen (matched)": ("epoch_11c_encoder_ceiling_cue_working_sessions.csv", "frozen (matched)"),
    "encoder gain": ("epoch_11amp_encoder_amplitude_cue_working_sessions.csv", "gain"),
    "best-match accuracy": ("epoch_10_best_match_acc_cue_working_sessions.csv", None),
    "crossnobis far-contra": ("epoch_8diag_matrices_crossnobis_cue_working_sessions.csv", "fC"),
}
POSITIONS = ("nI", "nM", "nC", "fI", "fM", "fC")


def _labels():
    """{animal: {epoch: [labels in date order]}} -- the order the sidecar rows are in."""
    out = {}
    for a in ANIMALS:
        out[a] = {"pre": sorted(l for l in config.phase_labels("pre") if l.startswith(a))}
        for l in sorted((l for l in config.pooled_labels(a) if epoch_of(l) != "pre"), key=lambda x: x.split("_")[-1]):
            out[a].setdefault(epoch_of(l), []).append(l)
    return out


def _sidecar(data_dir, fname, position, labs, log):
    """{animal: {label: value}}; a group whose row count disagrees with its session count is skipped."""
    df = pd.read_csv(data_dir / fname)
    if position is not None:
        df = df[df.position == position]
    out = {a: {} for a in ANIMALS}
    for a in ANIMALS:
        for ep, g in df[df.animal == a].groupby("epoch", sort=False):
            L = labs[a].get(ep, [])
            if len(L) != len(g):
                if ep != "pre":                      # pooled pre rows are expected to disagree
                    log(f"  !! {fname} {a} {ep}: {len(g)} rows vs {len(L)} sessions -- skipped")
                continue
            out[a].update(zip(L, g.value.astype(float).tolist()))
    return out


def _pooled_pre(data_dir, fname, position):
    df = pd.read_csv(data_dir / fname)
    df = df[df.epoch == "pre"]
    if position is not None:
        df = df[df.position == position]
    return {a: float(df[df.animal == a].value.mean()) for a in ANIMALS if (df.animal == a).any()}


def _map_series(labs, log):
    """Per-session cue-evoked map amplitude (post-cue minus pre-cue, mean over the eroded brain mask)."""
    from wfield_local import beta_maps as bm
    from wfield_local import position_evoked_maps as pem
    store, _counts = pem.maps_by_epoch()
    mask = np.asarray(bm.stat_mask()).astype(bool)
    out = {}
    for an, by_ep in store.items():
        for _ep, by_pos in by_ep.items():
            for pos, by_lab in by_pos.items():
                for lab, m in by_lab.items():
                    m = np.asarray(m, float)
                    if m.shape != mask.shape:
                        continue
                    v = m[mask]
                    out.setdefault(pos, {}).setdefault(an, {})[lab] = float(np.nanmean(v))
    log(f"  maps: {sum(len(d) for p in out.values() for d in p.values())} per-session amplitudes")
    return out


def build_series(data_dir, labs, log=print):
    series, pre_vals, pre_src = {}, {}, {}
    for name, (fname, pos) in SIDECARS.items():
        series[name] = _sidecar(data_dir, fname, pos, labs, log)
        pooled = _pooled_pre(data_dir, fname, pos)
        for a in ANIMALS:
            per = [series[name][a][l] for l in labs[a]["pre"] if l in series[name][a]]
            if len(per) >= 3:
                pre_vals[(name, a)], pre_src[(name, a)] = per, "per-session"
            elif a in pooled:
                pre_vals[(name, a)], pre_src[(name, a)] = [pooled[a]], "pooled+late-scatter"
    cn = {p: _sidecar(data_dir, SIDECARS["crossnobis far-contra"][0], p, labs, lambda *_: None) for p in POSITIONS}
    series["crossnobis mean(6 pos)"] = {a: {l: float(np.mean([cn[p][a][l] for p in POSITIONS if l in cn[p][a]]))
                                            for l in cn["fC"][a]} for a in ANIMALS}
    pooled = pd.read_csv(data_dir / SIDECARS["crossnobis far-contra"][0])
    pooled = pooled[pooled.epoch == "pre"].groupby("animal").value.mean()
    for a in ANIMALS:
        if a in pooled:
            pre_vals[("crossnobis mean(6 pos)", a)], pre_src[("crossnobis mean(6 pos)", a)] = [float(pooled[a])], "pooled+late-scatter"

    rt = pd.read_csv(data_dir / "recovery_trajectory_matched_cue_working.csv")
    rtp = pd.read_csv(data_dir / "recovery_trajectory_matched_cue_working_pre.csv")
    d2l = {a: {days_since_stroke(l): l for ep, L in labs[a].items() if ep != "pre" for l in L} for a in ANIMALS}
    for name, col in (("decoder frozen acc", "frozen"), ("decoder refit acc", "refit"), ("reorganisation G", "G_reorg")):
        series[name] = {a: {d2l[a][int(r.day)]: float(r[col]) for _, r in rt[rt.animal == a].iterrows() if int(r.day) in d2l[a]}
                        for a in ANIMALS}
    for a in ANIMALS:
        pf, pr = rt[rt.animal == a].pre_frozen.iloc[0], rt[rt.animal == a].pre_refit.iloc[0]
        p = rtp[rtp.animal == a]
        pre_vals[("decoder frozen acc", a)] = (pf - p.F_deficit).tolist()
        pre_vals[("decoder refit acc", a)] = (pr - (p.F_deficit - p.G_reorg)).tolist()
        pre_vals[("reorganisation G", a)] = p.G_reorg.tolist()
        for n in ("decoder frozen acc", "decoder refit acc", "reorganisation G"):
            pre_src[(n, a)] = "per-session"

    maps = _map_series(labs, log)
    for pos, nice in (("far_R", "far-contra"), ("close_L", "near-ipsi")):
        name = f"map amplitude {nice}"
        series[name] = {a: maps.get(pos, {}).get(a, {}) for a in ANIMALS}
        for a in ANIMALS:
            per = [series[name][a][l] for l in labs[a]["pre"] if l in series[name][a]]
            if len(per) >= 3:
                pre_vals[(name, a)], pre_src[(name, a)] = per, "per-session"
    return series, pre_vals, pre_src


#: readouts kept in raw units (differences, or quantities that cross zero) rather than as a fraction of pre
RAW_UNITS = {"reorganisation G", "map amplitude far-contra", "map amplitude near-ipsi"}


def evaluate(series, pre_vals, pre_src, labs, boundaries):
    rows, curves = [], {}
    for name, per in series.items():
        for a in ANIMALS:
            d, pv = per.get(a, {}), pre_vals.get((name, a))
            if not pv:
                continue
            raw = name in RAW_UNITS
            pre_mean = float(np.mean(pv))
            norm = (lambda v: v) if raw else (lambda v, _m=pre_mean: v / _m)
            if pre_src[(name, a)] == "per-session":
                sd_pre = _pstdev(pv) if raw else _pstdev(pv) / pre_mean
            else:
                late = [norm(d[l]) for ep, L in labs[a].items() if ep in ("subacute", "chronic") for l in L
                        if l in d and l not in EXCLUDE]
                sd_pre = _pstdev(late) if len(late) >= 4 else float("nan")
            if not np.isfinite(sd_pre) or sd_pre <= 0:
                continue
            cf, sf = boundaries[a]["chronic_from"], boundaries[a]["subacute_from"]
            for excl in (False, True):
                post = sorted((days_since_stroke(l), norm(d[l]), l) for ep, L in labs[a].items() if ep != "pre"
                              for l in L if l in d and not (excl and l in EXCLUDE))
                if not post:
                    continue
                days, vals = [p[0] for p in post], [p[1] for p in post]
                idx = _plateau_index(vals, sd_pre, None, days=days, min_day=sf)
                cv = [v for dd, v in zip(days, vals) if dd >= cf]
                flat = drift = resid = None
                if len(cv) >= 3:
                    flat, slope, resid = _is_flat(cv, sd_pre)
                    drift = slope * (len(cv) - 1)
                rows.append(dict(readout=name, animal=a, excl_0922=excl, pre_sd_source=pre_src[(name, a)],
                                 n_pre=len(pv), pre_mean=round(pre_mean, 4), sd_pre=round(sd_pre, 4), n_chronic=len(cv),
                                 chronic_level=round(float(np.mean(cv)), 4) if cv else np.nan,
                                 drift_preSD=round(drift / sd_pre, 2) if drift is not None else np.nan,
                                 resid_preSD=round(resid / sd_pre, 2) if resid is not None else np.nan,
                                 flat=flat, settled=(bool(resid <= CHRONIC_K_RES * sd_pre) if resid is not None else None),
                                 rule_plateau_day=(days[idx] if idx is not None else None), behav_chronic_from=cf))
                if not excl:
                    curves[(name, a)] = dict(days=days, vals=vals, labels=[p[2] for p in post],
                                             pre=[norm(v) for v in pv], sd=sd_pre, cf=cf, sf=sf)
    return pd.DataFrame(rows), curves


def figure(res, curves, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(n for n, _a in curves))
    fig, axes = plt.subplots(len(names), 4, figsize=(17, 2.1 * len(names)), sharex="col", squeeze=False)
    for i, name in enumerate(names):
        for j, a in enumerate(ANIMALS):
            ax, c = axes[i, j], curves.get((name, a))
            if not c:
                ax.set_visible(False)
                continue
            pm, ps = float(np.mean(c["pre"])), float(np.std(c["pre"]))
            ax.axhspan(pm - ps, pm + ps, color="0.85", zorder=0)
            ax.axhline(pm, color="0.5", lw=0.8)
            ax.axvline(c["sf"], color="tab:orange", lw=0.8, ls=":")
            ax.axvline(c["cf"], color="tab:red", lw=1.0, ls="--")
            ax.plot(c["days"], c["vals"], "-o", ms=3.5, lw=1, color="tab:blue")
            for dd, v, l in zip(c["days"], c["vals"], c["labels"]):
                if l in EXCLUDE:
                    ax.plot(dd, v, "x", ms=9, color="red", mew=2)
            r = res[(res.readout == name) & (res.animal == a) & (~res.excl_0922)].iloc[0]
            tag = ("flat" if r.flat else "RISING") + "/" + ("settled" if r.settled else "not settled")
            ax.set_title(f"{a}  chronic: {tag}   resid {r.resid_preSD}x   drift {r.drift_preSD}x", fontsize=8)
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("days since lesion", fontsize=8)
    fig.suptitle("Per-session trajectories against each animal's own pre-stroke band (mean +/- 1 SD). "
                 "dotted = subacute_from, dashed = behavioural chronic_from. red x = EXCLUDE (none at present).",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="output directory (default <labcams>/chronic_stability)")
    ap.add_argument("--data", default=None, help="epoch figure data dir (default <labcams>/grant_figures/epoch/data)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    data_dir = Path(a.data) if a.data else Path(rv.root("labcams")) / "grant_figures" / "epoch" / "data"
    out_dir = Path(a.out) if a.out else Path(rv.root("labcams")) / "chronic_stability"
    assert_writable(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = json.load(open(Path(rv.root("labcams")) / "epoch_boundaries.json"))["boundaries"]

    labs = _labels()
    series, pre_vals, pre_src = build_series(data_dir, labs)
    res, curves = evaluate(series, pre_vals, pre_src, labs, boundaries)
    res.to_csv(out_dir / "chronic_stability.csv", index=False)
    figure(res, curves, out_dir / "chronic_stability.png")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    print(f"\nK_RES={CHRONIC_K_RES}  K_DRIFT={CHRONIC_K_DRIFT}  -- settled: chronic residual <= K_RES x pre SD; "
          f"flat: total drift over the chronic window <= K_DRIFT x pre SD (one-sided)\n")
    cols = ["readout", "animal", "pre_sd_source", "n_chronic", "chronic_level", "drift_preSD", "resid_preSD", "flat", "settled",
            "rule_plateau_day", "behav_chronic_from"]
    for excl in (False, True):
        print("=" * 160)
        print("EXCLUDING PS92_0922" if excl else "ALL SESSIONS")
        print(res[res.excl_0922 == excl][cols].to_string(index=False))
    print(f"\nwrote {out_dir / 'chronic_stability.csv'} and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
