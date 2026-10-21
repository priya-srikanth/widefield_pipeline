"""CD window projections per session with the PRE sessions scored LEAVE-ONE-OUT (encoding window via Priya 10-20:
"pre sessions: each needs its own value"; 10-21 to-do 4 of STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF).

    WIDEFIELD_PRE_FROM=0801 python -m scripts.cd_loo_sessions [--animals PS92 ...] [--aligns precue cue lick] [--jobs 2]

`epoch_sessions_long --families cd` reads the saved `cd_trajectories` dumps, whose directions were fitted on ALL pre
sessions (June included) -- its pre rows are IN-SAMPLE (``pre_in_sample``). This refits instead, with the dumps'
settings (`cd_cross_animal_figure.collect`: method dom, contrast reference, gate lick, orth, occluded components
dropped) and the August-only baseline:
  post  one fit on every August pre session, projected onto each curated post session
  pre   per August pre session s: the fit WITHOUT s (directions, poles, z-score frame and condition-independent mode
        -- `analyse_animal(exclude_fit={s}, project_only={s})`), projected onto s
Value = per session x position the mean projection on its OWN direction (diagonal cell) in the windows of
`epoch_sessions_long.CD_WINDOWS` (pre-cue [-2, 0) s, cue [0, 1) s, lick [0, 1) s), exactly as ``fam_cd``.

Output `<labcams>/grant_figures/epoch/data/cd_window_projection_loo_sessions_long.csv` (the in-sample file is
untouched): family (``cd_loo`` -- NOT ``cd``: `scripts/imaging_lmm.py` reads every *_sessions_long.csv and would merge
the two), arm, align, variant, label, animal, epoch, day, position, measure, value, n_trials, pre_loo, baseline.
Plus ``data/cd_loo_traces.npz``: every session's diagonal TIME COURSE per alignment (``t_<align>`` s; ``trace_<align>``
rows = sessions x positions, with ``label_/epoch_/position_/n_/pre_loo_<align>``) for the trajectory figures
(`scripts/cd_loo_figures.py`). Cost: features, arms and grand means are session-cached, so a refit is cheap; each run projects only its
own session(s).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
SETTINGS = {"method": "dom", "reference": "contrast", "gate": "lick", "orth": True, "mask_occluded": True}


def _rows(res, align, loo):
    """Diagonal window means of one `analyse_animal` result (as `epoch_sessions_long.fam_cd`)."""
    from scripts.epoch_sessions_long import CD_WINDOWS
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    meas, lo, hi = CD_WINDOWS[align]
    t = np.arange(-res["pre_n"], res["post_n"]) / float(res["fs"])
    m = (t >= lo) & (t < hi)
    out = []
    for (lab, _ep, cd_p, tr_p), d in (res.get("session_traces") or {}).items():
        if cd_p != tr_p:
            continue
        pos = POSITION_NAMES.get(cd_p, str(cd_p)) if isinstance(POSITION_NAMES, dict) else str(cd_p)
        out.append(("cd_loo", lab, pos, f"CD projection, {meas}", float(np.nanmean(np.asarray(d["mean"], float)[m])),
                    int(d.get("n", 0)), loo, align))
    return out


def _traces(res, align, loo):
    """Diagonal per-session time courses of one result: (t s, [(label, epoch, position, n, loo, trace), ...])."""
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    t = np.arange(-res["pre_n"], res["post_n"]) / float(res["fs"])
    out = []
    for (lab, ep, cd_p, tr_p), d in (res.get("session_traces") or {}).items():
        if cd_p != tr_p:
            continue
        pos = POSITION_NAMES.get(cd_p, str(cd_p)) if isinstance(POSITION_NAMES, dict) else str(cd_p)
        out.append((lab, ep, pos, int(d.get("n", 0)), loo, np.asarray(d["mean"], np.float32)))
    return t, out


def run_animal(job):
    """All alignments for one animal (the animal is the parallel unit: one basis, warm caches)."""
    animal, aligns = job
    from wfield_local import analysis_kit as ak
    from wfield_local import cd_trajectories as cdt
    from wfield_local import epochs
    labs = [s["label"] for s in ak.curated_sessions() if s["label"].startswith(animal)]
    pre = [x for x in labs if epochs.epoch_of(x) == "pre"]
    post = [x for x in labs if epochs.epoch_of(x) not in ("pre", "excluded")]
    rows, notes, tr = [], [], {}
    for align in aligns:
        res = cdt.analyse_animal(animal, align, project_only=set(post), verbose=False, **SETTINGS)
        if res.get("skipped"):
            notes.append(f"{animal} {align} post: SKIPPED {res['skipped']}")
        else:
            rows += _rows(res, align, False)
            t, x = _traces(res, align, False)
            tr.setdefault(align, [t, []])[1].extend(x)
        for s in pre:
            res = cdt.analyse_animal(animal, align, exclude_fit={s}, project_only={s}, verbose=False, **SETTINGS)
            if res.get("skipped"):
                notes.append(f"{animal} {align} {s}: SKIPPED {res['skipped']}")
                continue
            rows += _rows(res, align, True)
            t, x = _traces(res, align, True)
            tr.setdefault(align, [t, []])[1].extend(x)
        print(f"  {animal} {align}: {len(pre)} pre LOO + {len(post)} post", flush=True)
    return rows, notes, tr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="*", default=list(ANIMALS))
    ap.add_argument("--aligns", nargs="*", default=["precue", "cue", "lick"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None, help="figure dir (default <labcams>/grant_figures/epoch)")
    a = ap.parse_args(argv)
    from wfield_local import analysis_kit as ak
    from wfield_local import cd_trajectories as cdt
    from wfield_local import config, epochs
    from wfield_local import figure_layout as FL
    pf = config.pre_from()
    if not pf:
        raise SystemExit("set WIDEFIELD_PRE_FROM=0801 (August-only baseline, Priya's rule)")
    res, fails = ak.fan_sessions([(x, tuple(a.aligns)) for x in a.animals], run_animal, jobs=a.jobs,
                                 key=lambda j: j[0], label="animal")
    for j, e in fails:
        print(f"{j[0]}: FAILED {e}")
    rows = [r for _, (rr, _n, _t) in res for r in rr]
    for _, (_r, notes, _t) in res:
        for n in notes:
            print("  ", n)
    C = pd.DataFrame(rows, columns=["family", "label", "position", "measure", "value", "n_trials", "pre_loo", "align"])
    C = C[C.label.isin(set(ak.curated_labels()))].copy()
    C.insert(1, "arm", C["align"])          # NOT C.align: that is DataFrame.align (a method)
    C.insert(3, "variant", "gate lick, orth, contrast")
    C.insert(5, "animal", C.label.str[:4])
    C.insert(6, "epoch", [epochs.epoch_of(x) for x in C.label])
    C.insert(7, "day", [epochs.days_since_stroke(x) for x in C.label])
    C["baseline"] = f"directions refitted on pre from {pf} (August); pre rows leave-one-out"
    fig_dir = a.out or Path(cdt.default_out())
    p = FL.sidecar_for(fig_dir, "cd_window_projection_loo", "_sessions_long.csv")
    C.to_csv(p, index=False)
    cur = set(ak.curated_labels())
    arrs = {}
    for align in a.aligns:
        t, items = None, []
        for _, (_r, _n, tr) in res:
            if align in tr:
                t = tr[align][0]
                items += [x for x in tr[align][1] if x[0] in cur]
        if t is None or not items:
            continue
        arrs[f"t_{align}"] = t
        arrs[f"trace_{align}"] = np.vstack([x[5] for x in items])
        for k, name in enumerate(("label", "epoch", "position", "n", "pre_loo")):
            arrs[f"{name}_{align}"] = np.array([x[k] for x in items])
    np.savez_compressed(FL.sidecar_for(fig_dir, "cd_loo_traces", ".npz"), **arrs)
    pd.set_option("display.width", 200)
    print(C.groupby(["animal", "align", "epoch"]).value.agg(["size", "mean"]).round(3).to_string())
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
