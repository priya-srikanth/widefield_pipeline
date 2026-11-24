"""Does leaving cam3 out change conclusions? (Priya 11-22.) Four-camera 3-D + per-lick kinematics re-run with a side
camera FORCED OFF (`pose3d_side.SIDE_OFF_ENV`), compared per lick (``lick_key``) with the adopted run.

    python scripts/qc_cam3_drop.py run S1 S2 ... [--variant no_cam3|control] [--workers 3]   # supervisor (CREATE_NO_WINDOW)
    python scripts/qc_cam3_drop.py kin ANIMAL:YYYYMMDD --variant no_cam3                     # one session's kinematics
    python scripts/qc_cam3_drop.py compare [--variant no_cam3]                                # tables vs production
    python scripts/qc_cam3_drop.py validate        # 11-23 cam3 re-inclusion table (a_forced / b_pnp4 / ref_on vs prod)

Writes ONLY under ``<P3D>/qc_cam3_drop_20261122/<variant>/`` (``cam3_on/<variant>/`` for a_forced / b_pnp4, 11-23) (P3D = the adopted run `pose3d_4cam_3net_20261121_sync60`);
inputs are read-only. Per session, two child processes with the production envs:
  1. FUSION = `scripts.pose3d production S --four-cam lp_side_r2b_prod_20261118 --out <variant> --jobs 1` under the
     whole-session camera re-run env (lp_stage/rerun_20261122/chain.py ENV), + ``WIDEFIELD_POSE3D_SIDE_OFF=cam3`` for
     ``no_cam3`` (``control`` = the same command without it: must reproduce production).
  2. KINEMATICS (`kin`) = `pose3d_kinematics._session_result(.., "fused", ..)` on the new frames + the frame-coverage
     gate (as `session_result`), then `pose3d_kin_postcue.postcue_licks` / `mask_single_view` (lick set, numbering and
     cam1-only rule = production's: the adopted fused lick list is NOT rebuilt). Env = the Phase B kin_fused env.
     ``no_cam3``: 3-D support (``kin_ok``, >= 2 cameras saw the lick) counts the views WITHOUT cam3
     (``n_views_nocam3`` = n_views - seen_cam3); a lick only cam1 + cam3 saw keeps its row but loses its kinematics.
Aim = atan2(side_ml, hypot(out_ap, down_dv)) in degrees. Compare: per session x position, fraction kin_ok, medians of
reach / speed_peak / aim for the first lick (production lick_idx 0) and all licks, no_cam3 - production, against the
session's change from its animal's August pre median (production) -- `compare_tables` (pure, tested).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
PY = "C:/Users/SabatiniLab/miniconda3/envs/dlc/python.exe"
CREATE_NO_WINDOW = 0x08000000
SIDE = "lp_side_r2b_prod_20261118"
TAG = "_3net_20261121_sync60"
CAM4_RUN = "lp_r4_heatmap_20261013__clean_v37_sync60"
P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses") / f"pose3d_4cam{TAG}"
QC = P3D / "qc_cam3_drop_20261122"
PROD_POSTCUE = P3D / "kinematics_fused_postcue" / "licks.csv.gz"
VARIANTS = {"no_cam3": "cam3", "control": "", "a_forced": "", "b_pnp4": "", "ref_on": ""}
#: 11-23 (Priya: "can we add back cam3 then?", PS95 0818 / 0820): cam3 FORCED ON past the spout gate
#: (`pose3d_side.SIDE_ON_ENV`), (a) with the standard six-spout pose, (b) posed from the four unbiased spouts
#: (`pose3d_side.PNP_EXCLUDE_ENV`: close_center / close_R carry the chin-fur tip bias). Output <QC>/cam3_on/<variant>/.
FORCED_ENV = {"a_forced": {"WIDEFIELD_POSE3D_SIDE_ON": "cam3"},
              "b_pnp4": {"WIDEFIELD_POSE3D_SIDE_ON": "cam3", "WIDEFIELD_POSE3D_PNP_EXCLUDE": "cam3:close_center,close_R"},
              # reference: forced on where cam3 PASSES the gate (a no-op on the result: must equal production), for the
              # forced-run QC columns (ray distances, per-spout gate px) on normal sessions; side events go to --out
              "ref_on": {"WIDEFIELD_POSE3D_SIDE_ON": "cam3"}}


def vdir(variant: str) -> Path:
    """The variant's output folder: <QC>/<variant>, or <QC>/cam3_on/<variant> for the forced-on variants."""
    return QC / "cam3_on" / variant if variant in FORCED_ENV else QC / variant
BASE_ENV = {"MKL_THREADING_LAYER": "SEQUENTIAL", "PYTHONPATH": str(WT), "PYTHONUNBUFFERED": "1"}
#: the whole-session camera re-run env (lp_stage/rerun_20261122/chain.py ENV) -- the production 3-D fusion
FUSION_ENV = {**BASE_ENV, "WIDEFIELD_WINDOWS_SPAN": "sync", "WIDEFIELD_RUN_LABEL": "60",
              "WIDEFIELD_LICK_INPUT_RUN": CAM4_RUN, "WIDEFIELD_CAM1_LICKS_DIR": "cam1_licks_v37_sync60",
              "WIDEFIELD_POSE3D_SOURCE": "4cam", "WIDEFIELD_POSE3D_TAG": TAG}
POS_ORDER = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
MEASURES = ["reach_mm", "speed_peak", "aim_deg"]


def aim_deg(side_ml, out_ap, down_dv) -> np.ndarray:
    """atan2(side_ml, hypot(out_ap, down_dv)) in degrees (the paper's aim angle)."""
    return np.degrees(np.arctan2(np.asarray(side_ml, float), np.hypot(np.asarray(out_ap, float),
                                                                       np.asarray(down_dv, float))))


def _env(base: dict, extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WIDEFIELD_", "WFIELD_"))}
    return {**env, **base, **(extra or {})}


def _child(cmd: list[str], env: dict, log: Path) -> subprocess.Popen:
    fh = open(log, "w", encoding="utf-8")
    return subprocess.Popen([PY, *cmd], cwd=str(WT), env=env, stdout=fh, stderr=subprocess.STDOUT,
                            creationflags=CREATE_NO_WINDOW)


def _session_jobs(s: str, variant: str, out: Path) -> list[tuple[str, list[str], dict]]:
    off = VARIANTS[variant]
    fus = FUSION_ENV | ({"WIDEFIELD_POSE3D_SIDE_OFF": off} if off else {}) | FORCED_ENV.get(variant, {})
    return [("fuse", ["-m", "scripts.pose3d", "production", s, "--four-cam", SIDE, "--out", str(out), "--jobs", "1"],
             _env(fus)),
            ("kin", ["scripts/qc_cam3_drop.py", "kin", s, "--variant", variant], _env(BASE_ENV))]


def run(sessions: list[str], variant: str, workers: int) -> int:
    """Supervisor: each session runs fuse -> kin; up to ``workers`` sessions at once; one retry per step. Polls only
    the PIDs it started. Timings -> <variant>/timings.json."""
    out = vdir(variant)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    tfile = out / "timings.json"
    tim = json.loads(tfile.read_text()) if tfile.exists() else {}
    todo = [s for s in sessions if not (tim.get(s, {}).get("kin_ok"))]
    queue = {s: _session_jobs(s, variant, out) for s in todo}
    stage = {s: 0 for s in todo}
    tries = {s: 0 for s in todo}
    running: dict = {}
    print(f"supervisor pid {os.getpid()}: {len(todo)} sessions, {workers} workers -> {out}", flush=True)
    while queue or running:
        while queue and len(running) < workers:
            s = next(iter(queue))
            name, cmd, env = queue[s][stage[s]]
            tries[s] += 1
            log = out / "logs" / f"{s.replace(':', '_')}_{name}_try{tries[s]}.log"
            p = _child(cmd, env, log)
            running[s] = (p, name, time.time())
            del queue[s]
            print(f"{time.strftime('%H:%M:%S')} {s} {name} start pid {p.pid}", flush=True)
        time.sleep(10)
        for s, (p, name, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[s]
            el = round((time.time() - t0) / 60, 2)
            tim.setdefault(s, {})[f"{name}_min"] = el
            print(f"{time.strftime('%H:%M:%S')} {s} {name} rc {rc} in {el} min", flush=True)
            jobs = _session_jobs(s, variant, out)
            if rc == 0:
                tries[s] = 0
                stage[s] += 1
                if stage[s] < len(jobs):
                    queue[s] = jobs
                else:
                    tim[s]["kin_ok"] = True
            elif tries[s] < 2:
                queue[s] = jobs
            else:
                tim[s]["failed"] = name
            tfile.write_text(json.dumps(tim, indent=1))
    return 0 if all(tim.get(s, {}).get("kin_ok") for s in sessions) else 1


def kin(s: str, variant: str) -> int:
    """One session's fused kinematics on the variant's 3-D frames -> <variant>/kinematics_fused[_postcue]/<s>_*.csv.gz."""
    from scripts import pose3d_kin_postcue as PC
    from scripts import pose3d_kinematics as K
    from wfield_local import frame_coverage as FC
    from wfield_local.paths import PathResolver
    out = vdir(variant)
    animal, date = s.split(":")
    fr = out / f"{animal}_{date}_frames.csv.gz"
    rv = PathResolver()
    L, J, row = K._session_result(rv, s, "fused", fr, None)
    cov = FC.session_coverage(rv, animal, date)
    L, G = K.gate_by_coverage(L, cov)
    # the rest of `pose3d_kinematics.session_result` (fused mode), so the session row / jaw trials splice as built
    row = {**row, "licks_3d": int((L.n_3d >= 3).sum()) if len(L) else 0, "licks_gated": len(G),
           "trials_gated": int(cov.gated.sum()), "cov_union_min": float(cov.cov_union.min()) if len(cov) else np.nan}
    if len(J):
        J = J.assign(frame_gated=J.trial_id.map(cov.set_index("trial_id").gated).eq(True))
    (out / "kinematics_fused").mkdir(exist_ok=True)
    (out / "kinematics_fused_postcue").mkdir(exist_ok=True)
    # daq_peak_s as production writes it (`pose3d_kinematics.main`: "{:.4f}"; "%.4g" kept 1 ms at 1-10 s, 0.1 s past 100 s)
    fmt = (lambda D: D.assign(daq_peak_s=D["daq_peak_s"].map("{:.4f}".format)) if "daq_peak_s" in D else D)
    fmt(L).to_csv(out / "kinematics_fused" / f"{animal}_{date}_licks.csv.gz", index=False, float_format="%.4g")
    spans = PC._spans(f"{animal}_{date}")
    P = PC.postcue_licks(L, spans)
    P["n_views_all"] = P.n_views
    if VARIANTS[variant]:
        P["n_views"] = P.n_views - P[f"seen_{VARIANTS[variant]}"].astype(bool).astype(int)
    P = PC.mask_single_view(P)
    fmt(P).to_csv(out / "kinematics_fused_postcue" / f"{animal}_{date}_licks.csv.gz", index=False, float_format="%.4g")
    J.to_csv(out / "kinematics_fused" / f"{animal}_{date}_jaw_trials.csv.gz", index=False, float_format="%.4g")
    if len(G):
        fmt(G).to_csv(out / "kinematics_fused" / f"{animal}_{date}_licks_gated.csv.gz", index=False, float_format="%.4g")
    pd.DataFrame([row]).to_csv(out / "kinematics_fused" / f"{animal}_{date}_session_row.csv", index=False)
    print(f"{s}: {len(L)} fused licks ({len(G)} gated), {len(P)} post-cue, kin_ok {int(P.kin_ok.sum())}")
    return 0


def session_medians(P: pd.DataFrame) -> pd.DataFrame:
    """Per session x position x lick set (first = lick_idx 0, all): n, fraction kin_ok, medians of `MEASURES` over
    kin_ok licks. Pure."""
    P = P.assign(aim_deg=aim_deg(P.side_ml, P.out_ap, P.down_dv))
    rows = []
    for lset, Q in (("first", P[P.lick_idx == 0]), ("all", P)):
        for (s, pos), g in Q.groupby(["session", "position"]):
            k = g[g.kin_ok.astype(bool)]
            rows.append({"session": s, "position": pos, "licks": lset, "n": len(g), "frac_kin": k.shape[0] / len(g),
                         **{m: float(k[m].median()) if k[m].notna().any() else np.nan for m in MEASURES}})
    return pd.DataFrame(rows)


def compare_tables(prod: pd.DataFrame, alt: pd.DataFrame, epochs: dict, pre_ref: pd.DataFrame) -> pd.DataFrame:
    """Matched per-lick tables (rows of ``alt`` joined to ``prod`` on session + lick_key; ``alt`` takes prod's lick_idx)
    -> per session x position x lick set: production and alternative medians, delta (alt - prod), and the session's
    effect = production median - its animal's August pre median (``pre_ref``: animal, position, licks, measure
    columns), with |delta| / |effect|. Pure."""
    a = alt.drop(columns=["lick_idx"]).merge(prod[["session", "lick_key", "lick_idx"]], on=["session", "lick_key"],
                                             how="inner", validate="one_to_one")
    mp, ma = session_medians(prod), session_medians(a)
    T = mp.merge(ma, on=["session", "position", "licks"], suffixes=("_prod", "_alt"))
    T["animal"] = T.session.str[:4]
    T["epoch"] = T.session.map(epochs)
    T = T.merge(pre_ref.rename(columns={m: f"{m}_pre" for m in MEASURES}), on=["animal", "position", "licks"],
                how="left")
    for m in ["frac_kin"] + MEASURES:
        T[f"d_{m}"] = T[f"{m}_alt"] - T[f"{m}_prod"]
    for m in MEASURES:
        T[f"eff_{m}"] = T[f"{m}_prod"] - T[f"{m}_pre"]
        T[f"ratio_{m}"] = (T[f"d_{m}"].abs() / T[f"eff_{m}"].abs()).where(T.epoch != "pre")
    T["position"] = pd.Categorical(T.position, POS_ORDER, ordered=True)
    return T.sort_values(["licks", "session", "position"]).reset_index(drop=True)


def compare(variant: str) -> int:
    from wfield_local import epochs as EP
    alt_dir = vdir(variant) / "kinematics_fused_postcue"
    A = pd.concat([pd.read_csv(f, dtype={"lick_key": str, "date": str}) for f in sorted(alt_dir.glob("PS*_licks.csv.gz"))],
                  ignore_index=True)
    sess = sorted(A.session.unique())
    cols = ["session", "animal", "date", "epoch", "position", "lick_key", "lick_idx", "kin_ok", "n_views", "seen_cam3",
            "cams", "reach_mm", "speed_peak", "side_ml", "out_ap", "down_dv"]
    P = pd.read_csv(PROD_POSTCUE, usecols=cols, dtype={"lick_key": str, "date": str})
    animals = sorted({s[:4] for s in sess})
    pre = P[(P.epoch == "pre") & (P.date >= "20260801") & P.animal.isin(animals)]
    pre_m = session_medians(pre).assign(animal=lambda d: d.session.str[:4])
    pre_ref = pre_m.groupby(["animal", "position", "licks"])[MEASURES].median().reset_index()
    P = P[P.session.isin(sess)]
    n_prod, n_alt = P.groupby("session").size(), A.groupby("session").size()
    T = compare_tables(P, A, {s: EP.epoch_of(f"{s[:4]}_{s[9:]}") for s in sess}, pre_ref)
    out = vdir(variant)
    T.to_csv(out / "compare_by_session_position.csv", index=False, float_format="%.4f")
    m = A.merge(P[["session", "lick_key"]], on=["session", "lick_key"], how="outer", indicator=True)
    info = {s: {"licks_prod": int(n_prod.get(s, 0)), "licks_alt": int(n_alt.get(s, 0)),
                "unmatched": int((m[m.session == s]._merge != "both").sum()),
                "lost_cam3_support": int(((A.session == s) & (A.n_views_all >= 2) & (A.n_views < 2)).sum())}
            for s in sess}
    (out / "compare_counts.json").write_text(json.dumps(info, indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = ["session", "epoch", "position", "n_prod", "frac_kin_prod", "d_frac_kin"] + \
        [c for m in MEASURES for c in (f"{m}_prod", f"d_{m}", f"eff_{m}")]
    for lset in ("first", "all"):
        print(f"\n== {lset} lick(s): {variant} - production ==")
        print(T[T.licks == lset][show].round(3).to_string(index=False))
    print(json.dumps(info, indent=1))
    print("->", out / "compare_by_session_position.csv")
    return 0


def tongue_jitter(F: pd.DataFrame, P: pd.DataFrame, min_idx: int = 1) -> dict:
    """Frame-to-frame tongue jitter in STEADY lick phases: over post-cue licks with ``lick_idx >= min_idx`` and kin_ok,
    every run of three consecutive 4 ms frames inside the lick span (on_ms .. off_ms) with a multi-view tongue point
    (``tongue_views >= 2``): |X[t-1] - 2 X[t] + X[t+1]| mm. Returns {"n", "med", "p95"}. Pure."""
    Q = P[(P.lick_idx >= min_idx) & P.kin_ok.astype(bool)].dropna(subset=["on_ms", "off_ms"])
    F = F.sort_values(["trial_id", "t_ms"])
    X = F[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    ok = (F.tongue_views.to_numpy() >= 2) & np.isfinite(X).all(1)
    tr, t = F.trial_id.to_numpy(), F.t_ms.to_numpy(float)
    same = (tr[:-2] == tr[2:]) & np.isclose(t[2:] - t[:-2], 8.0, atol=0.5) & ok[:-2] & ok[1:-1] & ok[2:]
    acc = np.full(len(F), np.nan)
    acc[1:-1] = np.where(same, np.linalg.norm(X[:-2] - 2 * X[1:-1] + X[2:], axis=1), np.nan)
    inl = np.zeros(len(F), bool)
    by = {k: i for k, i in F.groupby("trial_id").indices.items()}
    for r in Q.itertuples(index=False):
        i = by.get(r.trial_id)
        if i is None:
            continue
        inl[i[(t[i] >= r.on_ms) & (t[i] <= r.off_ms)]] = True
    v = acc[inl & np.isfinite(acc)]
    return {"n": int(len(v)), "med": float(np.median(v)) if len(v) else np.nan,
            "p95": float(np.percentile(v, 95)) if len(v) else np.nan}


#: validation runs (11-23): variant -> sessions; "prod" = the adopted production files (no cam3 on 0818 / 0820)
VALIDATE = {"prod": ["PS95:20260818", "PS95:20260820", "PS95:20260819", "PS95:20260821"],
            "a_forced": ["PS95:20260818", "PS95:20260820"],
            "b_pnp4": ["PS95:20260818", "PS95:20260820", "PS95:20260821"],
            "ref_on": ["PS95:20260819", "PS95:20260821"]}


def validate() -> int:
    """The cam3 re-inclusion validation table (per variant x session) -> <QC>/cam3_on/validation.csv + printed."""
    summ = {v: pd.read_csv(vdir(v) / f"summary_production_{SIDE}.csv") for v in VALIDATE if v != "prod"}
    summ["prod"] = pd.read_csv(P3D / f"summary_production_{SIDE}.csv")
    P_prod = pd.read_csv(PROD_POSTCUE, dtype={"lick_key": str, "date": str})
    rows = []
    for v, sess in VALIDATE.items():
        for s in sess:
            sid = s.replace(":", "_")
            r = summ[v].set_index("session").loc[sid]
            d = (P3D if v == "prod" else vdir(v))
            F = pd.read_csv(d / f"{sid}_frames.csv.gz", usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap",
                                                                  "tongue_dv", "tongue_views"])
            P = (P_prod[P_prod.session == sid] if v == "prod" else
                 pd.read_csv(d / "kinematics_fused_postcue" / f"{sid}_licks.csv.gz", dtype={"lick_key": str, "date": str}))
            jit = tongue_jitter(F, P)
            k = P[P.kin_ok.astype(bool)]
            out = {"variant": v, "session": sid, "side_dropped": r.get("side_dropped", ""),
                   "gate_rms600_cam3": r.side_rms600_cam3, "gate_max600_cam3": r.side_max600_cam3,
                   "pnp_rms600_fit_cam3": r.get("pnp_rms600_fit_cam3", np.nan),
                   "pnp_rms600_all_cam3": r.get("pnp_rms600_all_cam3", np.nan),
                   "pnp_max600_all_cam3": r.get("pnp_max600_all_cam3", np.nan),
                   "tongue_off_cam3_ap": r.tongue_off_cam3_ap, "nose_off_cam3_dv": r.nose_off_cam3_dv}
            for part in ("tongue", "jaw", "nose"):
                out[f"{part}_ray_med_cam3"] = r.get(f"{part}_ray_mm_med_cam3", np.nan)
                out[f"{part}_ray_p95_cam3"] = r.get(f"{part}_ray_mm_p95_cam3", np.nan)
                seen = r.get(f"{part}_seen_cam3", np.nan)
                out[f"{part}_rej_cam3"] = r[f"{part}_rejected_cam3"]
                out[f"{part}_rej_frac_cam3"] = r[f"{part}_rejected_cam3"] / seen if seen and seen == seen else np.nan
            out.update({"jitter_n": jit["n"], "jitter_med_mm": jit["med"], "jitter_p95_mm": jit["p95"],
                        "tongue_3d_frac": r.tongue_3d_frac, "licks_kin": len(k),
                        "speed_all": k.speed_peak.median(), "reach_all": k.reach_mm.median(),
                        "speed_first": k[k.lick_idx == 0].speed_peak.median(),
                        "reach_first": k[k.lick_idx == 0].reach_mm.median()})
            rows.append(out)
    T = pd.DataFrame(rows)
    o = QC / "cam3_on" / "validation.csv"
    T.to_csv(o, index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(T.set_index(["variant", "session"]).T.round(3).to_string())
    print("->", o)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "kin", "compare", "validate"])
    ap.add_argument("sessions", nargs="*", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--variant", choices=list(VARIANTS), default="no_cam3")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a.sessions, a.variant, a.workers)
    if a.cmd == "kin":
        return kin(a.sessions[0], a.variant)
    if a.cmd == "validate":
        return validate()
    return compare(a.variant)


if __name__ == "__main__":
    sys.exit(main())
