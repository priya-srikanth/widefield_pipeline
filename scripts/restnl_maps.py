"""RESTnl (lick-free rest, flat) versions of the imaging page's rest-referenced epoch maps (15r pooled + 15rpa per animal).

Priya 11-01: RESTnl primary, old REST kept as a supplement, FLAT, rule wr, AUGUST-ONLY pre. Imaging page figures 7.1,
7.3, 7.5, 7.7, 7.8, 7.10, 7.16, 7.18, 7.19 (drift twins: scripts/map_drift_prepre.py --refs restnl; attempt maps:
wfield_local.attempt_maps --rest-variant restnl).

    python scripts/restnl_maps.py --arm contact|lp|enl1s [--aligns precue cue lick] [--plot-only]

What it does: sets the arm env (as map_drift_prepre.ARMS), WIDEFIELD_PRE_FROM=0801 (August-only pre),
WIDEFIELD_MAP_REFS=restnl (only the derived RESTnl reference is attached and drawn; the other references are NOT redrawn),
(the WIDEFIELD_PRM_LEGACY_DATE8 shim it used to force was dropped in the cohort-1 rebuild, 11-03), then calls the page's own renderers `epoch_maps._fig_15r_reference_maps` / `_fig_15rpa_reference_by_animal` per alignment.
Needs the per-session baselines first: `python -m wfield_local.restnl_mask`.

Outputs (NEW folders; nothing existing is overwritten): <labcams>/grant_figures/epoch_restnl/ (contact),
grant_figures/LP_licks/epoch_restnl/, grant_figures/DAQ_ENL1s/epoch_restnl/; figure data = each figure's
data/<name>_bundle.{npz,json} + _meta / _stats sidecars (map_grid), plus per-animal amplitude tables
restnl_amplitude_per_animal_<align>_<variant>.csv (RESTnl beside REST / RESTW / RAW on the same August-only session
set). --plot-only redraws every figure in the folder from its bundle (no data load).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ARMS = {"contact": {}, "lp": {"WIDEFIELD_LICK_SOURCE": "lp"}, "enl1s": {"WIDEFIELD_ENL_WINDOW_S": "1.0"}}
GATE = {"contact": "trial gate: DAQ spout contact", "lp": "trial gate: LP video licks",
        "enl1s": "trial gate: DAQ spout contact, ENL 1 s"}
ALIGN_VARIANT = {"precue": ("working", "ENL (pre-cue), lick + miss-while-working"),
                 "cue": ("working", "post-cue, lick + miss-while-working"),
                 "lick": ("lick", "post-lick, lick trials only")}
PRE_FROM = "0801"


def setup_env(arm: str, refs: str = "restnl", august_only: bool = True) -> None:
    for k in ("WIDEFIELD_LICK_SOURCE", "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_PRE_FROM"):
        os.environ.pop(k, None)
    os.environ.update(ARMS[arm])
    if august_only:
        os.environ["WIDEFIELD_PRE_FROM"] = PRE_FROM
    os.environ["WIDEFIELD_MAP_REFS"] = refs
    os.environ.setdefault("WIDEFIELD_MAP_JOBS", "4")
    if int(os.environ["WIDEFIELD_MAP_JOBS"]) > 4:
        os.environ["WIDEFIELD_MAP_JOBS"] = "4"


def out_dir(root: Path | None = None) -> Path:
    from wfield_local import lp_licks
    from wfield_local.paths import PathResolver
    # RESTNL_OUT_ROOT (2026-11-02, cohort-1 swap-mask re-run): a run on another hemo variant (WIDEFIELD_HEMO_VARIANT)
    # writes under its OWN root (arm sub-folders as usual), never over the adopted figures. Unset = unchanged.
    env = os.environ.get("RESTNL_OUT_ROOT")
    base = root or (Path(env) if env else Path(PathResolver().root("cohort_out")) / "grant_figures" / "epoch_restnl")
    return lp_licks.output_dir(base)


def amplitude_table(store, align, variant, refs=("restnl", "rest", "restw", "raw")):
    """Per animal x position x post epoch: RMS(epoch mean map) / RMS(pre mean map) in the brain mask -- the figures'
    amplitude ratio (`beta_maps.within_animal_pooled`), one animal at a time (rule 8: per animal first)."""
    import pandas as pd

    from wfield_local import beta_maps as bm
    from wfield_local import position_reference_maps as prm
    from wfield_local.grant_figures import CONF_LABELS
    rows = []
    for ref in refs:
        for q in CONF_LABELS:
            pre = prm.by_animal(store, ref, q, "pre")
            for e in ("acute", "subacute", "chronic"):
                post = prm.by_animal(store, ref, q, e)
                for an in sorted(set(pre) & set(post)):
                    _p, _q, _d, ratio, _a = bm.within_animal_pooled({an: pre[an]}, {an: post[an]})
                    rows.append(dict(align=align, variant=variant, reference=ref, position=q, epoch=e, animal=an,
                                     n_pre=len(pre[an]), n_post=len(post[an]), ratio=float(ratio)))
    return pd.DataFrame(rows)


def render(arm: str, aligns) -> list:
    import pandas as pd

    from wfield_local import epoch_maps as em
    from wfield_local import lp_licks
    from wfield_local import position_reference_maps as prm
    from wfield_local.writeguard import assert_writable
    out = out_dir()
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[restnl_maps] arm {arm}: lick source {lp_licks.lick_source()}, ENL window {lp_licks.enl_window_s()}, "
          f"pre from {os.environ.get('WIDEFIELD_PRE_FROM')}, refs {prm.render_references()} -> {out}", flush=True)
    made = []
    for align in aligns:
        variant, wname = ALIGN_VARIANT[align]
        wname = f"{wname} | {GATE[arm]} | pre: August only"
        for fn in (em._fig_15r_reference_maps, em._fig_15rpa_reference_by_animal):
            for p in (fn(out, align, variant, wname) or []):
                print(f"  -> {p}", flush=True)
                made.append(p)
        os.environ["WIDEFIELD_MAP_REFS"] = "restnl"
        store, _rel, _ntr = prm.maps_by_epoch(align, variant)
        A = amplitude_table(store, align, variant)
        (out / "data").mkdir(exist_ok=True)
        A.to_csv(out / "data" / f"restnl_amplitude_per_animal_{align}_{variant}.csv", index=False)
        fc = A[(A.position == "far_R") & (A.epoch == "acute")].pivot_table(index="animal", columns="reference",
                                                                           values="ratio")
        print(f"  .. per-animal far-contra acute / pre amplitude ({align}/{variant}):\n{fc.round(3).to_string()}",
              flush=True)
    pd.DataFrame({"figure": [str(p) for p in made]}).to_csv(out / "data" / f"restnl_maps_made_{arm}.csv", index=False)
    return made


def plot_only(folder: Path) -> list:
    """Redraw every RESTnl map figure in ``folder`` from its bundle (same name; no data load)."""
    from wfield_local import epoch_figures as ef
    made = []
    for b in sorted((folder / "data").glob("*RESTNLref*_bundle.npz")):
        stem = b.stem.replace("_bundle", "")
        p = ef.replot_map(b, folder, name=stem)
        print(f"  -> {p}", flush=True)
        made.append(p)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=tuple(ARMS), required=True)
    ap.add_argument("--aligns", nargs="+", default=None, choices=tuple(ALIGN_VARIANT))
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    setup_env(a.arm)
    from wfield_local import parallel
    parallel.pin_blas()
    if a.plot_only:
        plot_only(out_dir())
        return 0
    aligns = a.aligns or (["precue"] if a.arm == "enl1s" else ["precue", "cue", "lick"])
    render(a.arm, aligns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
