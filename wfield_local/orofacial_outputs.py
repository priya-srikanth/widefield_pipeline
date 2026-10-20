"""Orofacial (pose) ANALYSIS outputs on the share: run-versioned, session-first, one index.

    python -m wfield_local.orofacial_outputs process --bundle lp_r4_prod_20261015 [--sessions PS93:20260814 ...]
    python -m wfield_local.orofacial_outputs index                      # rebuild index.csv + README.md

Priya, 2026-10-15: "outputs should be organized rationally on the server, with subdirectories for different analysis
types ... we can improve on [stroke_orofacial's] structure to minimize redundancy and ensure organization as will be
most useful for finding figures and analyses in the future". Layout (root = paths.yaml `orofacial`):

    Widefield/LP_analysis/                     (paths.yaml root `orofacial`; Priya/Orofacial/Widefield until 10-16)
      README.md, index.csv                       generated: layout + one row per run x session
      <run>/                                     RUN = pose model x cleaning version, e.g. lp_r4_heatmap_20261013__clean_v35
        run.json                                 model, cleaning version, digest of the cleaning + kinematics params
        sessions/<PSxx>/<YYYYMMDD>/
          manifest.json                          provenance + checks (source videos, frame counts, counts, commit)
          windows.npz                            trial windows: index columns (incl. per-frame DAQ time daq_s) + raw
                                                 pose (n_rows, parts, x/y/lk), x/y x px_scale (June only; v36+)
          clean.npz                              cleaned tongue / jaw per window row: x, y, fill (0 meas, 1-2 interp, 3 fill)
          licks.csv                              post-cue licks: cue + 70 ms .. DAQ trial stop
          licks_precue.csv                       spontaneous licks: position strobe (spout arrived) .. cue + 70 ms
          licks_iti.csv                          inter-trial licks: trial stop .. window end ('tail'), window start ..
                                                 strobe ('lead'); reach fields blank (spout moving / retracted)
          trials.csv, jaw_trials.csv, tongue_jaw_mismatch.csv
          qc.png
        cohort/tables/<analysis>.csv, cohort/figures/<analysis>/...

WHY NOT stroke_orofacial's layout: it is TYPE-first (`dlc_kinematics/cleaned_traces/<bodypart>/<animal>/<date>`,
`.../lick_events/...`, `.../per_trial/...`), so one session is spread over ~8 trees and a re-run overwrites in place
with nothing saying which model / cleaning made a file. Here one session is one folder, and a different model or
cleaning is a different RUN folder, never an overwrite: `ensure_run` refuses to add sessions to a run whose recorded
params digest no longer matches the code's (bump CLEANING_VERSION -> new run). The raw whole-video O2 predictions stay
in `DeepLabCut/Widefield/o2/<bundle>/out/` (provenance; not copied). Per-frame data is compressed .npz (no parquet
library in these envs); `load_session` reads a session back into DataFrames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config

#: Bump when `orofacial_clean` / the kinematics change what they compute -> a NEW run folder (never an overwrite).
#: v35 = jaw x ceiling 75 + x jump test + bracketed flip rule (x-only) + hidden jaw NaN + inter-lick tongue veto +
#: jaw-off-tongue (all adopted 2026-10-14 / 15, DECISIONS).
#: v36 = v35 + the June camera-scale correction (`orofacial_camera`, 2026-10-18): June pose x 0.8853 before cleaning.
CLEANING_VERSION = "v36"
PARTS_ORDER = ("nose", "jaw", "tongue", "spout")


def root(rv) -> Path:
    return Path(rv.root("orofacial"))


def params_digest() -> str:
    """sha1 of every config block that shapes the outputs (cleaning, kinematics, lick detection)."""
    d = config.defaults()
    blob = {k: d.get(k) for k in ("orofacial_clean", "orofacial_kinematics", "lick_detection", "orofacial_camera")}
    return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:12]


def camera_scale(date: str) -> float:
    """Factor applied to this session's raw pose x, y (`orofacial_camera`): the June camera configuration -> Aug+ px;
    1.0 from `scale_before` on. Divide pose by it to draw on the raw video."""
    c = config.defaults().get("orofacial_camera") or {}
    return float(c.get("scale_factor", 1.0)) if str(date) < str(c.get("scale_before", "0")) else 1.0


def run_name(scorer: str, cleaning: str = CLEANING_VERSION) -> str:
    return f"{scorer.lower()}__clean_{cleaning}"


def session_dir(rv, run: str, animal: str, date: str) -> Path:
    return root(rv) / run / "sessions" / animal / date


def code_commit() -> str:
    """Short HEAD, with ``+dirty`` when the code (wfield_local / scripts / configs) has uncommitted changes -- a
    manifest must never claim clean code it did not run on."""
    try:
        here = Path(__file__).parent
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, text=True, capture_output=True,
                              timeout=20).stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "wfield_local", "scripts", "configs"],
                               cwd=here.parent, text=True, capture_output=True, timeout=20).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:                                                  # noqa: BLE001
        return "unknown"


def ensure_run(rv, run: str, meta: dict) -> Path:
    """Create ``<run>/run.json`` or check it: the same run must keep the same params digest."""
    from wfield_local import writeguard
    d = root(rv) / run
    writeguard.assert_writable(d.parent)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "run.json"
    dig = params_digest()
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("params_digest") != dig:
            raise SystemExit(f"{run}: params digest {old.get('params_digest')} on disk != {dig} in the code -- the "
                             f"cleaning / kinematics settings changed: bump CLEANING_VERSION (a new run), do not mix")
        return d
    f.write_text(json.dumps({**meta, "run": run, "cleaning_version": CLEANING_VERSION, "params_digest": dig,
                             "created": time.strftime("%Y-%m-%d %H:%M"), "code_commit": code_commit()}, indent=1),
                 encoding="utf-8")
    return d


# --------------------------------------------------------------------------- one session

def _flat(ix_len: int, sel: np.ndarray, parts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({f"{p}_{c}": sel[:ix_len, k, j].astype(float) for k, p in enumerate(parts)
                         for j, c in enumerate(("x", "y", "likelihood"))})


def process_session(rv, bundle: str, animal: str, date: str, force: bool = False, log=print) -> dict | None:
    """O2 bundle outputs of one session -> its folder in the run: windows, cleaned traces, kinematics tables, QC."""
    import cv2

    import scripts.pose_kinematics_demo as D
    from wfield_local import camera_sync, o2_inference as oi
    from wfield_local import spout_frame as SF
    from wfield_local import trial_windows as TW
    bd = oi.bundle_dir(rv, bundle)
    bj = json.loads((bd / "bundle.json").read_text(encoding="utf-8"))
    run = run_name(bj["scorer"])
    ensure_run(rv, run, {"model_scorer": bj["scorer"], "source_bundle": bundle, "reader": bj.get("reader"),
                         "source_model_dir": bj.get("source_model_dir")})
    sd = session_dir(rv, run, animal, date)
    if (sd / "manifest.json").exists() and not force:
        log(f"{animal} {date}: done already ({sd})")
        return json.loads((sd / "manifest.json").read_text(encoding="utf-8"))
    tasks = pd.read_csv(bd / "tasks.tsv", sep="\t", dtype=str)
    g = tasks[(tasks.animal == animal) & (tasks.date == date)]
    od = bd / "out" / f"{animal}_{date}"
    tpls = camera_sync.templates_for(rv, date, animal, "cam4")
    vids, srcs = [], []
    for r in g.itertuples():
        stem = Path(r.video).stem
        done = od / f"{stem}_done.json"
        if not done.exists():
            log(f"{animal} {date}: {r.video} not finished -- session skipped")
            return None
        info = json.loads(done.read_text(encoding="utf-8"))
        v = next(x for x in oi.videos_for(rv, animal, date) if x.name == r.video)
        n_vid = int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT))
        if info["n_frames"] != n_vid:
            log(f"{animal} {date}: {r.video} predicted {info['n_frames']} frames, video has {n_vid} -- skipped")
            return None
        rec = stem.split("_", 1)[1]
        if rec not in tpls:
            log(f"{animal} {date}: no alignment template for {rec} -- skipped")
            return None
        z = np.load(od / f"{stem}_{info['scorer']}.npz")
        for t in tpls[rec]:                  # one per DAQ segment on a split day (camera_sync.templates_for)
            vids.append((z["pose"], t, r.video))
        srcs.append({"video": r.video, "n_frames": int(info["n_frames"]), "wall_h": round(info["wall_s"] / 3600, 2),
                     "reader": info.get("reader"), "n_templates": len(tpls[rec]),
                     "template_quality_ok": bool(all(t.get("quality_ok", True) for t in tpls[rec]))})
        parts = [str(x) for x in z["parts"]]
    ix, sel = oi.build_windows(vids[0][0], animal, date, rv, videos=vids if len(vids) > 1 else None)
    if not len(ix):
        log(f"{animal} {date}: no trial windows -- skipped")
        return None
    px_scale = camera_scale(date)
    if px_scale != 1.0:                      # June camera configuration -> Aug+ px units, BEFORE cleaning (px thresholds)
        sel = sel.copy()
        sel[:, :, :2] *= px_scale
    flat = _flat(len(ix), sel, parts)
    spans = [(int(gg.index.min()), int(gg.index.max()) + 1, str(gg.position.iloc[0])) for _, gg in ix.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(flat[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds(animal, date, rv)
    stops = dict(zip(b.trial_id, b.stop_s - b.cue_s))
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    T, J, M, clean = D.run(flat, ix, frame, stops, contacts)
    lick_s, cue_s = D.daq_lick_onsets_s(animal, date, rv), dict(zip(b.trial_id, b.cue_s))
    P = D.precue_licks(clean, ix, frame, flat, lick_s, cue_s)          # spout arrival -> cue + 70 ms
    Q = D.iti_licks(clean, ix, frame, flat, lick_s, cue_s)             # trial stop -> next window; window start -> strobe
    from wfield_local import writeguard
    writeguard.assert_writable(sd.parent)
    sd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sd / "windows.npz", pose=sel.astype(np.float32), parts=np.array(parts),
                        **{f"ix_{c}": (ix[c].to_numpy() if pd.api.types.is_numeric_dtype(ix[c]) else np.asarray(ix[c].astype(str), dtype=str))
                           for c in ix.columns})                 # text as fixed-width str: loads without pickle
    ct, pt, _ = clean["tongue"]
    cj, pj, _ = clean["jaw"]
    np.savez_compressed(sd / "clean.npz", tongue_x=ct.x_final[pt].astype(np.float32),
                        tongue_y=ct.y_final[pt].astype(np.float32), tongue_fill=ct.fill_method[pt],
                        jaw_x=cj.x_final[pj].astype(np.float32), jaw_y=cj.y_final[pj].astype(np.float32),
                        jaw_fill=cj.fill_method[pj], tongue_origin=np.array([ct.X0, ct.Y0]),
                        jaw_rest=np.array([cj.X0, cj.Y0]), spout_origin=np.asarray(frame.origin),
                        spout_ap_axis=np.asarray(frame.ap_axis), px_scale=np.float64(px_scale))
    T.per_lick.to_csv(sd / "licks.csv", index=False)
    P.to_csv(sd / "licks_precue.csv", index=False)
    Q.to_csv(sd / "licks_iti.csv", index=False)
    T.per_trial.to_csv(sd / "trials.csv", index=False)
    J.to_csv(sd / "jaw_trials.csv", index=False)
    M.trials.to_csv(sd / "tongue_jaw_mismatch.csv", index=False)
    try:
        qc_figure(ix, flat, ct.y_final[pt], ct.x_final[pt], cj.y_final[pj], sd / "qc.png", f"{animal} {date} ({run})")
    except Exception as ex:                                            # noqa: BLE001 -- QC must not block outputs
        log(f"{animal} {date}: QC figure failed ({type(ex).__name__}: {ex})")
    man = {"run": run, "animal": animal, "date": date, "phase": config.session_phase(animal, date),
           "source_bundle": bundle, "videos": srcs, "n_trials": int(ix.trial_k.nunique()), "n_rows": int(len(ix)),
           "n_licks": int(len(T.per_lick)), "n_precue_licks": int(len(P)), "n_iti_licks": int(len(Q)), "n_contact_licks": int(T.per_lick.get("contact", pd.Series(dtype=bool))
                                                                    .fillna(False).astype(bool).sum()),
           "px_scale": px_scale, "params_digest": params_digest(), "code_commit": code_commit(),
           "created": time.strftime("%Y-%m-%d %H:%M")}
    (sd / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    log(f"{animal} {date}: {man['n_trials']} trials, {man['n_licks']} licks -> {sd}")
    return man


def load_session(rv, run: str, animal: str, date: str) -> dict:
    """{"index": DataFrame, "pose": flat DataFrame (<part>_x/y/likelihood), "clean": DataFrame, "licks": ..,
    "trials": .., "jaw_trials": .., "mismatch": .., "manifest": dict} for one processed session."""
    sd = session_dir(rv, run, animal, date)
    w = np.load(sd / "windows.npz", allow_pickle=False)
    ix = pd.DataFrame({k[3:]: w[k] for k in w.files if k.startswith("ix_")})
    parts = [str(x) for x in w["parts"]]
    c = np.load(sd / "clean.npz")
    clean = pd.DataFrame({k: c[k] for k in ("tongue_x", "tongue_y", "tongue_fill", "jaw_x", "jaw_y", "jaw_fill")})
    rd = lambda n: pd.read_csv(sd / n) if (sd / n).exists() else None       # noqa: E731
    return {"index": ix, "pose": _flat(len(ix), w["pose"], parts), "clean": clean, "licks": rd("licks.csv"),
            "licks_precue": rd("licks_precue.csv"),
            "licks_iti": rd("licks_iti.csv"),
            "trials": rd("trials.csv"), "jaw_trials": rd("jaw_trials.csv"), "mismatch": rd("tongue_jaw_mismatch.csv"),
            "manifest": json.loads((sd / "manifest.json").read_text(encoding="utf-8"))}


def qc_figure(ix, flat, tongue_y, tongue_x, jaw_y, out: Path, title: str) -> None:
    """Whole-session QC: coverage per keypoint per trial, spout clusters + drift, nose stability, cleaned tongue /
    jaw cue-aligned by position (the panel set Priya reviewed on PS93 0814 / 0821, 2026-10-15)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos_order = ["far_L", "close_L", "far_center", "close_center", "close_R", "far_R"]
    col = dict(zip(pos_order, plt.cm.tab10(np.arange(6))))
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    tk = ix.trial_k.to_numpy()
    post = ((ix.t_ms >= 0) & (ix.t_ms <= 3000)).to_numpy()
    a = ax[0, 0]
    for kp, thr in (("nose", 0.6), ("spout", 0.6), ("jaw", 0.4), ("tongue", 0.4)):
        cov = pd.Series((flat[f"{kp}_likelihood"].to_numpy() >= thr)[post]).groupby(tk[post]).mean()
        a.plot(cov.index, cov.values, lw=0.8, label=f"{kp} med {cov.median():.2f}")
    a.set_ylim(0, 1.02); a.legend(fontsize=7); a.set_title("confident fraction per trial (0-3 s post-cue)", fontsize=9)
    pre = ((ix.t_ms >= ix.strobe_ms + 300) & (ix.t_ms < 0)).to_numpy() & (flat.spout_likelihood.to_numpy() >= 0.9)
    sm = (pd.DataFrame({"k": tk, "pos": ix.position, "x": flat.spout_x, "y": flat.spout_y})[pre]
          .groupby("k").agg(pos=("pos", "first"), x=("x", "median"), y=("y", "median")))
    a = ax[0, 1]
    for p_ in pos_order:
        s_ = sm[sm.pos == p_]
        a.scatter(s_.index, s_.x, s=5, color=col[p_], label=p_)
    a.set_title("spout tip x per trial (at position): clusters + drift", fontsize=9); a.legend(fontsize=6)
    okn = flat.nose_likelihood.to_numpy() >= 0.6
    nz = pd.DataFrame({"k": tk, "x": flat.nose_x, "y": flat.nose_y})[okn].groupby("k").median()
    a = ax[0, 2]
    a.plot(nz.index, nz.x - nz.x.median(), lw=0.8, label="x"); a.plot(nz.index, nz.y - nz.y.median(), lw=0.8, label="y")
    a.legend(fontsize=7); a.set_title("nose per-trial median rel. session median (px)", fontsize=9)
    t = ix.t_ms.to_numpy(); w = (t >= -500) & (t <= 3500); tb = np.round(t / 8) * 8
    for i, (lab, arr) in enumerate((("cleaned tongue y (px from mouth; fill = 0)", tongue_y),
                                    ("cleaned tongue x (px from mouth)", tongue_x),
                                    ("cleaned jaw y (px from rest; hidden = NaN)", jaw_y))):
        a = ax[1, i]
        for p_ in pos_order:
            m = w & (ix.position == p_).to_numpy()
            s_ = pd.Series(arr[m]).groupby(tb[m]).mean()
            a.plot(s_.index, s_.values, lw=1, color=col[p_], label=p_)
        a.axvline(0, color="k", lw=0.8); a.set_title(lab, fontsize=9); a.set_xlabel("ms from cue")
    ax[1, 0].legend(fontsize=6)
    fig.suptitle(f"{title}: {ix.trial_k.nunique()} trials", fontsize=11)
    fig.tight_layout()
    from wfield_local.orofacial_cohort import save_png_svg
    save_png_svg(fig, out, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- index / README

def write_index(rv) -> Path:
    """index.csv: one row per run x session from the manifests; README.md describing the layout."""
    rows = []
    for m in sorted(root(rv).glob("*/sessions/*/*/manifest.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        rows.append({k: d.get(k) for k in ("run", "animal", "date", "phase", "n_trials", "n_licks", "n_contact_licks",
                                           "source_bundle", "params_digest", "code_commit", "created")}
                    | {"n_videos": len(d.get("videos", [])), "folder": str(m.parent.relative_to(root(rv)))})
    out = root(rv) / "index.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (root(rv) / "README.md").write_text(README, encoding="utf-8")
    return out


README = """# Orofacial (pose) analysis outputs -- widefield cohort

Generated by `widefield_pipeline` (`wfield_local/orofacial_outputs.py`); do not edit by hand.

* `index.csv` -- one row per RUN x session (trials, licks, provenance). Start here.
* `<run>/` -- a RUN is one pose model x one cleaning version (e.g. `lp_r4_heatmap_20261013__clean_v35`). A new model or
  a changed cleaning is a NEW run folder; nothing is overwritten. `run.json` records the model, the cleaning version and
  a digest of every cleaning / kinematics parameter.
* `<run>/sessions/<animal>/<YYYYMMDD>/` -- everything for one session: `manifest.json` (source videos, frame-count
  checks, counts, code commit), `windows.npz` (trial windows + raw pose; `ix_daq_s` = each frame's DAQ time, s,
  from the camera timestamps), `clean.npz` (cleaned tongue / jaw; fill 0 =
  measured, 1-2 = interpolated, 3 = filled: tongue in the mouth = 0 px, jaw hidden = NaN), `licks.csv` (post-cue
  licks, cue + 70 ms .. the DAQ trial stop; incl. `peak_interp`, `interp_frac`, `protrusion_max_measured_px`),
  `licks_precue.csv` (spontaneous licks from the position strobe -- spout in place -- to cue + 70 ms, same detector and
  columns, `t_ms` from the cue (negative) + `t_from_strobe_ms`), `licks_iti.csv` (inter-trial licks: `segment` 'tail' =
  trial stop .. the next trial's window, 'lead' = window start .. strobe; spout-reach fields blank), `trials.csv`, `jaw_trials.csv`,
  `tongue_jaw_mismatch.csv`, `qc.png`.
* `<run>/cohort/tables/<analysis>.csv`, `<run>/cohort/figures/<analysis>/` -- cross-session analyses, each figure
  with its data next to it.
* Read a session in Python: `from wfield_local import orofacial_outputs as oo; s = oo.load_session(rv, run, animal,
  date)`.
* Raw whole-video predictions: `DeepLabCut/Widefield/o2/<bundle>/out/` (not copied here).
"""


# --------------------------------------------------------------------------- CLI

def _process_worker(job):
    """Module-level for `parallel.fan_out` (spawn pickles by name)."""
    from wfield_local.paths import PathResolver
    bundle, animal, date, force = job
    try:
        m = process_session(PathResolver(), bundle, animal, date, force=force)
        return (animal, date, "ok" if m else "skipped", None)
    except Exception as ex:                                            # noqa: BLE001 -- report, do not stop the rest
        return (animal, date, "FAILED", f"{type(ex).__name__}: {str(ex)[:200]}")


def main(argv=None) -> int:
    from wfield_local import o2_inference as oi
    from wfield_local import parallel
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process")
    p.add_argument("--bundle", required=True)
    p.add_argument("--sessions", nargs="*", default=None, help="animal:date (default: every finished CURATED session)")
    p.add_argument("--all", action="store_true", help="every finished session, not just the curated cohort")
    p.add_argument("--force", action="store_true")
    p.add_argument("--jobs", type=int, default=4, help="parallel sessions (each holds ~1-2 GB)")
    sub.add_parser("index")
    a = ap.parse_args(argv)
    rv = PathResolver()
    if a.cmd == "process":
        bd = oi.bundle_dir(rv, a.bundle)
        tasks = pd.read_csv(bd / "tasks.tsv", sep="\t", dtype=str)
        sess = sorted({(r.animal, r.date) for r in tasks.itertuples()})
        if a.sessions:
            want = {tuple(s.split(":")) for s in a.sessions}
            sess = [s for s in sess if s in want]
        elif not a.all:                                   # curated cohort only (Priya 10-20); --all for every session
            from wfield_local import analysis_kit as ak
            cur = ak.curated_ids()
            sess = [s for s in sess if f"{s[0]}_{s[1]}" in cur]
        done = [s for s in sess if all((bd / "out" / f"{s[0]}_{s[1]}" / f"{Path(v).stem}_done.json").exists()
                                       for v in tasks[(tasks.animal == s[0]) & (tasks.date == s[1])].video)]
        print(f"{len(done)} of {len(sess)} sessions have every video finished")
        results, failures = parallel.fan_out([(a.bundle, an, d, a.force) for an, d in done], _process_worker,
                                             jobs=a.jobs, label="session")
        res = [r for _it, r in results] + [(it[1], it[2], "FAILED", err) for it, err in failures]
        for an, d, st, err in sorted(res):
            if st != "ok":
                print(f"  {an} {d}: {st}" + (f" -- {err}" if err else ""))
        print(f"ok {sum(r[2] == 'ok' for r in res)}, skipped {sum(r[2] == 'skipped' for r in res)}, "
              f"failed {sum(r[2] == 'FAILED' for r in res)}")
    print(f"index -> {write_index(rv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
