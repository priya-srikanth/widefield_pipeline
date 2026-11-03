"""FINAL LP ARM render (Priya 11-02 "ok for the full job"): every LP-arm imaging family once, in final form.

Plan + results: docs/status/LP_FINAL_ARM_20261102.md. One configuration, set here, never by hand:

    licks     WIDEFIELD_LICK_SOURCE=lp       (LP video licks, cam4 + cam1; lick-aligned = first LP lick)
    gate      WIDEFIELD_GATE_SOURCE=lp       (engaged / working / stopped + quit = DAQ contact OR LP attempt)
    pre       WIDEFIELD_PRE_FROM=0801        (AUGUST-ONLY pre-stroke baseline; the drift figure is the exception,
                                              it needs June by construction: scripts/map_drift_prepre.py)
    refs      WIDEFIELD_MAP_REFS = mean rest restw precue raw restnl   (RESTnl = the lick-free flat rest reference,
                                              plus the existing references exactly as they are now)
    ENL       WIDEFIELD_ENL_WINDOW_S (1.0 default under LP; decoders also 1.5 and 2.0)
    decoders  WIDEFIELD_EPOCH_FROZEN=1 (`grant_confusion.EPOCH_FROZEN_ENV`): the epoch acc decoders go through the
              frozen store. DAQ-TRAINED = fitted by a DAQ-source run (`decoders --source daq`, same window + pre set),
              loaded by the LP run; LP-TRAINED (sensitivity) = WIDEFIELD_LP_FROZEN=lp. Store: WIDEFIELD_FROZEN_MODEL_DIR
              = <figures_working>/frozen_models_lpfinal (own store; the canonical ones are untouched).

Steps (each prints what it resolved):
    restnl                         gate-LP RESTnl baselines (own cache dir `.widefield_session_cache_restnl_glp<sha>`)
    maps     [--aligns ...]        15r + 15rpa for every reference incl. RESTnl; per-animal amplitude tables
    decoders --window W --source lp|daq --trained daq|lp     acc + accdelta, ENL arm (precue / working); trial counts
    compare  --window W            the CURRENT LP-arm decoders (plain LP_licks, all pre, in-memory) per animal
    --plot-only                    redraw a step's figures from their figdata (no data load)

Outputs: <labcams>/grant_figures/LPfinal/  (NEW; lp_licks.output_dir appends the run tag)
    maps            LPfinal/LP_licks_gateLP/epoch/
    decoders        LPfinal/trained_<daq|lp>/<LP_licks[_ENLw]_gateLP | DAQ_ENLws>/epoch/
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

PRE_FROM = "0801"
ALL_REFS = "mean rest restw precue raw restnl"
LP_TEXT = "licks: LP video (cam4 + cam1)"
DAQ_TEXT = "licks: DAQ spout contact"
GATE_TEXT = "gate: engaged = DAQ contact OR LP attempt"
ALIGN_VARIANT = {"precue": ("working", "ENL (pre-cue) {w:g} s lick-free, lick + miss-while-working"),
                 "cue": ("working", "post-cue, lick + miss-while-working"),
                 "lick": ("lick", "post-lick (first LP lick), lick trials only")}


def root() -> Path:
    from wfield_local.paths import PathResolver
    return Path(PathResolver().root("cohort_out")) / "grant_figures" / "LPfinal"


def setup(source="lp", window=None, trained=None, refs=ALL_REFS, august=True, gate=True):
    for k in ("WIDEFIELD_LICK_SOURCE", "WIDEFIELD_GATE_SOURCE", "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_PRE_FROM",
              "WIDEFIELD_LP_FROZEN", "WIDEFIELD_EPOCH_FROZEN", "WIDEFIELD_MAP_REFS"):
        os.environ.pop(k, None)
    if source == "lp":
        os.environ["WIDEFIELD_LICK_SOURCE"] = "lp"
        if gate:
            os.environ["WIDEFIELD_GATE_SOURCE"] = "lp"
    if window is not None:
        os.environ["WIDEFIELD_ENL_WINDOW_S"] = f"{float(window):g}"
    if august:
        os.environ["WIDEFIELD_PRE_FROM"] = PRE_FROM
    if refs:
        os.environ["WIDEFIELD_MAP_REFS"] = refs
    if trained:
        os.environ["WIDEFIELD_EPOCH_FROZEN"] = "1"
        wd = Path(os.environ.get("WIDEFIELD_FIGURES_WORKING", str(Path.home() / "cue_lick")))
        os.environ["WIDEFIELD_FROZEN_MODEL_DIR"] = str(wd / "frozen_models_lpfinal")
        Path(os.environ["WIDEFIELD_FROZEN_MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
        if trained == "lp":
            os.environ["WIDEFIELD_LP_FROZEN"] = "lp"
    os.environ["WIDEFIELD_PRM_LEGACY_DATE8"] = "1"
    jobs = min(int(os.environ.get("WIDEFIELD_MAP_JOBS", "4")), 4)
    os.environ["WIDEFIELD_MAP_JOBS"] = str(jobs)
    from wfield_local import parallel
    parallel.pin_blas()


def describe() -> str:
    from wfield_local import config, gate_source, lp_licks
    return (f"lick source {lp_licks.lick_source()}, gate {gate_source.source()}, ENL window {lp_licks.enl_window_s()}, "
            f"pre from {config.pre_from()}, refs {os.environ.get('WIDEFIELD_MAP_REFS')}, epoch frozen "
            f"{os.environ.get('WIDEFIELD_EPOCH_FROZEN')}, LP frozen {os.environ.get('WIDEFIELD_LP_FROZEN')}, "
            f"store {os.environ.get('WIDEFIELD_FROZEN_MODEL_DIR')}, map jobs {os.environ.get('WIDEFIELD_MAP_JOBS')}")


# ---------------------------------------------------------------------------------------------------------- RESTnl
def step_restnl(jobs=4):
    setup(source="lp", august=False, refs=None)
    from wfield_local import restnl_mask
    print(f"[lpfinal restnl] {describe()} -> {restnl_mask.cache_dir()}", flush=True)
    rows = restnl_mask.build(jobs=jobs)
    v = restnl_mask.validate(rows)
    bad = [r for r in rows if r.get("error") or ("ok_wr" in r and not r["ok_wr"])]
    print(f"[lpfinal restnl] {len(rows)} sessions; frame counts equal to restw_check (DAQ gate) in {v['equal']}, "
          f"differ in {len(v['differ'])} (expected where the LP gate moves rest frames); not ok: {bad}", flush=True)
    import json
    (restnl_mask.cache_dir() / "build_summary.json").write_text(json.dumps({"rows": rows, "validate": v}, indent=1,
                                                                           default=str), encoding="utf-8")
    return rows


# ------------------------------------------------------------------------------------------------------------ maps
def maps_out() -> Path:
    from wfield_local import lp_licks
    return lp_licks.output_dir(root() / "epoch")


def step_maps(aligns, plot_only=False):
    setup(source="lp")
    import pandas as pd

    from wfield_local import epoch_figures as ef
    from wfield_local import epoch_maps as em
    from wfield_local import lp_licks
    from wfield_local import position_reference_maps as prm
    from wfield_local.writeguard import assert_writable
    sys_path_scripts()
    from restnl_maps import amplitude_table
    out = maps_out()
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[lpfinal maps] {describe()} -> {out}", flush=True)
    if plot_only:
        for b in sorted((out / "data").glob("epoch_15r*_bundle.npz")):
            print(f"  -> {ef.replot_map(b, out, name=b.stem.replace('_bundle', ''))}", flush=True)
        return
    w = lp_licks.enl_window_s()
    for align in aligns:
        t0 = time.time()
        variant, wtxt = ALIGN_VARIANT[align]
        wname = f"{wtxt.format(w=w)} | {LP_TEXT} | {GATE_TEXT} | pre: August only"
        for fn in (em._fig_15r_reference_maps, em._fig_15rpa_reference_by_animal):
            for p in (fn(out, align, variant, wname) or []):
                print(f"  -> {p}", flush=True)
        store, _rel, _ntr = prm.maps_by_epoch(align, variant)
        A = amplitude_table(store, align, variant, refs=tuple(prm.render_references()))
        A.to_csv(out / "data" / f"lpfinal_amplitude_per_animal_{align}_{variant}.csv", index=False)
        fc = A[(A.position == "far_R")].pivot_table(index=["epoch", "animal"], columns="reference", values="ratio")
        print(f"  .. far-contra amplitude / pre per animal ({align}/{variant}):\n{fc.round(3).to_string()}", flush=True)
        print(f"  .. {align} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame({"done": [time.strftime("%Y-%m-%d %H:%M")]}).to_csv(out / "data" / "lpfinal_maps_done.csv")


def sys_path_scripts():
    import sys
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


# -------------------------------------------------------------------------------------------------------- decoders
def dec_out(trained) -> Path:
    from wfield_local import lp_licks
    return lp_licks.output_dir(root() / f"trained_{trained}" / "epoch")


def trial_table(per_animal):
    """Trials and sessions per animal x epoch x position in the decoder's own scored sets (rule 8)."""
    import numpy as np
    import pandas as pd

    from wfield_local import epoch_figures as ef
    from wfield_local.epoch_accuracy import _accuracy_of, _named
    from wfield_local.grant_figures import CONF_LABELS
    rows = []
    for an in sorted(per_animal):
        one = {an: per_animal[an]}
        cov = ef.epoch_coverage(one)["per_epoch"]
        for e in ef.PANELS:
            rec = ef.pool_records(one, e)
            if rec is None:
                continue
            y, _p = _named(rec)
            pre = per_animal[an][0]
            n_s = (int(np.unique(np.asarray(pre[2]) // 1_000_000).size) if e == "pre" and pre is not None
                   else cov.get(e, {}).get(an, 0))              # pre blocks = session index * 1e6 + block
            r = {"animal": an, "epoch": e, "n_sessions": n_s, "n_trials": int(len(y))}
            for q in CONF_LABELS:
                r[f"n_{q}"] = int(np.sum(y == q))
                r[f"acc_{q}"] = _accuracy_of(rec, q)
            rows.append(r)
    return pd.DataFrame(rows)


def step_decoders(window, source, trained, plot_only=False):
    setup(source=source, window=window, trained=trained, refs=None)
    import pandas as pd

    from wfield_local import epoch_figures as ef
    from wfield_local import lp_licks
    from wfield_local.epoch_accuracy import _epoch_arm, _per_position_accuracy
    from wfield_local.writeguard import assert_writable
    out = dec_out(trained)
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[lpfinal decoders] {describe()} -> {out}", flush=True)
    if plot_only:
        import pickle
        for pk in sorted((out / "data").glob("epoch_acc*_figdata.pkl")):
            t = pickle.loads(pk.read_bytes())["kwargs"]["title"]
            for tr in ("DAQ", "LP"):    # first renders said "frozen decoder DAQ-trained" (wrapped at the hyphen)
                t = t.replace(f"frozen decoder {tr}-trained", f"frozen decoder trained on {tr} pre trials")
            print(f"  -> {ef.replot_figdata(pk, out, title=t)}", flush=True)
        return
    t0 = time.time()
    per_animal = _epoch_arm("precue", "working")
    print(f"  .. collected in {(time.time() - t0) / 60:.1f} min", flush=True)
    T = trial_table(per_animal)
    T.insert(0, "window_s", float(window))
    T.insert(1, "source", source)
    T.insert(2, "trained", trained)
    (out / "data").mkdir(exist_ok=True)
    T.to_csv(out / "data" / "lpfinal_enl_trials_acc_per_animal.csv", index=False)
    print(T[["animal", "epoch", "n_sessions", "n_trials", "n_far_R", "acc_far_R"]].round(3).to_string(), flush=True)
    lick = LP_TEXT if lp_licks.is_lp() else DAQ_TEXT
    gate = GATE_TEXT if lp_licks.is_lp() else "gate: DAQ contact"
    wname = (f"ENL (pre-cue) {window:g} s lick-free ({'LP video' if lp_licks.is_lp() else 'DAQ contact'}), lick + "
             f"miss-while-working | {lick} | {gate} | pre: August only | frozen decoder trained on {trained.upper()} pre trials")
    p = _per_position_accuracy(per_animal, out, "ENL", "precue", "working", wname)
    print(f"  -> {p}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.DataFrame({"done": [time.strftime("%Y-%m-%d %H:%M")]}).to_csv(out / "data" / "lpfinal_decoders_done.csv")


def step_compare(window):
    """The CURRENT LP-arm decoders (plain LP_licks: LP gate as before, June + August pre, in-memory fit), per animal,
    to set beside the final ones. Writes a table only (no figure)."""
    setup(source="lp", window=window, refs=None, august=False, gate=False)
    # the CURRENT renders were fitted in memory; the stored path is the default since 11-02, so opt out explicitly
    os.environ["WIDEFIELD_EPOCH_FROZEN"] = "0"
    from wfield_local.epoch_accuracy import _epoch_arm
    print(f"[lpfinal compare] {describe()}", flush=True)
    T = trial_table(_epoch_arm("precue", "working"))
    T.insert(0, "window_s", float(window))
    from wfield_local import lp_licks
    out = lp_licks.output_dir(root() / "compare")
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / f"current_LParm_enl{window:g}s_acc_per_animal.csv", index=False)
    print(T[["animal", "epoch", "n_trials", "n_far_R", "acc_far_R"]].round(3).to_string(), flush=True)


def main(argv=None) -> int:
    from wfield_local.console import use_utf8_stdout
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("restnl", "maps", "decoders", "compare"))
    ap.add_argument("--aligns", nargs="+", default=["precue", "cue", "lick"], choices=tuple(ALIGN_VARIANT))
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--source", choices=("lp", "daq"), default="lp")
    ap.add_argument("--trained", choices=("daq", "lp"), default="daq")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    if a.source == "daq" and a.trained == "lp":
        ap.error("a DAQ-source run can only train DAQ models")
    t0 = time.time()
    if a.step == "restnl":
        step_restnl(min(a.jobs, 4))
    elif a.step == "maps":
        step_maps(a.aligns, a.plot_only)
    elif a.step == "decoders":
        step_decoders(a.window, a.source, a.trained, a.plot_only)
    else:
        step_compare(a.window)
    print(f"[lpfinal {a.step}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
