"""RETRACTION BOUTS (Priya 11-24): does licking that STARTS while the spout retracts to the dock go up after the stroke,
and is that explained by a delayed first lick (late response bouts running into the retraction)? Table work only.

Definitions (as iti_dock_audit's ITI class):
  bout       gap-based over ALL of the session's licks (THE session lick table = the fused four-camera train): a bout
             starts at an onset >= il.BOUT_GAP (1 s) after the previous onset of any context
  retraction a bout whose first lick falls in [trial stop, dock arrival) of that trial's cycle
             (iti_dock_audit's iti_dock_cycles.csv: dock = the behavior log's dock event, else the reconstruction)
  docked     a bout whose first lick falls in [dock arrival, next trial_start)
  responded  attempted trial (OROFACIAL_GATE=attempted table) with a finite first-lick latency (first_lick_ms of the
             camera-tagged attempts table = the four-camera first lick after the cue)
Per trial: retraction bout yes / no, docked bout yes / no, first-lick latency, class (cls / canonical / attempted), spout.
Per session: share of responded trials with a retraction bout, all spouts, per spout and far contra.
Change vs each animal's own pre: paired nested bootstrap (analysis_kit.boot_delta, animals -> sessions), * = CI excludes 0.
Latency: bins = quintiles of the animal's own PRE responded-trial latencies; (a) share per latency bin and epoch, change
vs pre within each bin; (b) LATENCY-ADJUSTED share: per session, observed share minus the share expected from the
session's own latency distribution under the animal's PRE bin rates (indirect standardization); its change vs pre is the
post-stroke increase NOT explained by latency.
Inputs: <in_path_x>/iti_dock_cycles.csv (iti_dock_audit.py), the attempts table (WFIELD_DAYS_ATTEMPTS_CSV or --attempts).
Writes under il.OUT_W (needs WFIELD_ITI_OUT_W): retraction_trials.csv.gz, retraction_sessions.csv,
retraction_tests.csv, retraction_latency_bins.csv.
Usage: WFIELD_ITI_OUT_W=<new dir> python scripts/iti_retraction_bouts.py [--attempts CSV]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

POS6 = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
N_BINS = 5


def trial_flags(on_all: np.ndarray, C: pd.DataFrame, gap: float | None = None) -> pd.DataFrame:
    """Per trial cycle (stop, dock, trial_start): retraction / docked bout started in the cycle (any bout start in
    [stop, dock) / [dock, trial_start)). on_all = every lick onset of the session (any context)."""
    gap = il.BOUT_GAP if gap is None else gap
    on = np.sort(np.asarray(on_all, float))
    on = on[np.isfinite(on)]
    bs = on[np.r_[True, np.diff(on) >= gap]] if len(on) else on
    st, dk = C.stop.to_numpy(float), C.dock.to_numpy(float)
    ts = np.where(np.isfinite(C.trial_start.to_numpy(float)), C.trial_start.to_numpy(float), np.inf)

    def any_in(a, b):
        ok = np.isfinite(a) & np.isfinite(b) & (b > a)
        i, j = np.searchsorted(bs, a), np.searchsorted(bs, b)      # starts in [a, b)
        return np.where(ok, j > i, False)
    return C.assign(retraction_bout=any_in(st, dk), docked_bout=any_in(dk, ts),
                    cycle_ok=np.isfinite(st) & np.isfinite(dk))


def _boot(S: pd.DataFrame, col: str, rng) -> list[dict]:
    from wfield_local import analysis_kit as ak
    rows = []
    pre = {a: g[col].dropna().to_list() for a, g in S[S.epoch == "pre"].groupby("animal")}
    for ep in ("acute", "subacute", "chronic"):
        post = {a: g[col].dropna().to_list() for a, g in S[S.epoch == ep].groupby("animal")}
        post = {a: v for a, v in post.items() if v}
        iv = ak.boot_delta(post, {a: v for a, v in pre.items() if v}, rng)
        per = {a: float(np.mean(post[a]) - np.mean(pre[a])) for a in post if pre.get(a)}
        pt = iv.point if iv else np.nan
        rows.append({"epoch": ep, "point": pt, "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan,
                     "n_animals": iv.n_animals if iv else 0,
                     "n_dir": int(sum(np.sign(v) == np.sign(pt) for v in per.values())) if np.isfinite(pt) else 0,
                     "pre_level": float(np.mean([np.mean(v) for v in pre.values() if v])) if pre else np.nan,
                     **{f"animal_{a}": per.get(a, np.nan) for a in ("PS92", "PS93", "PS94", "PS95")}})
    return rows


def main() -> int:
    if not os.environ.get("WFIELD_ITI_OUT_W"):
        raise SystemExit("set WFIELD_ITI_OUT_W to a NEW folder")
    from wfield_local import config, epochs
    from wfield_local import session_licks as SL
    att = (sys.argv[sys.argv.index("--attempts") + 1] if "--attempts" in sys.argv
           else os.environ.get("WFIELD_DAYS_ATTEMPTS_CSV", ""))
    if not att:
        raise SystemExit("needs --attempts CSV or WFIELD_DAYS_ATTEMPTS_CSV (the camera-tagged attempts table)")
    A = pd.read_csv(att, usecols=lambda c: c in ("label", "trial_id", "pos", "first_lick_ms", "cls", "attempted",
                                                  "canonical", "animal", "epoch"))
    CY = pd.read_csv(il.in_path_x("iti_dock_cycles.csv"))
    rows = []
    for lab, C in CY.groupby("label", sort=True):
        an, date = config.animal_of(lab), "2026" + lab.split("_")[-1]
        L = SL.session_licks(f"{an}_{date}")
        F = trial_flags(L.daq_on_s.to_numpy(float), C.reset_index(drop=True))
        F["label"], F["epoch"], F["animal"] = lab, str(epochs.epoch_of(lab)), an
        rows.append(F)
    T = pd.concat(rows, ignore_index=True)
    T = T.merge(A.drop(columns=[c for c in ("animal", "epoch", "pos") if c in A], errors="ignore"),
                on=["label", "trial_id"], how="left", validate="one_to_one")
    T["responded"] = T.attempted.fillna(False).astype(bool) & np.isfinite(T.first_lick_ms) & T.cycle_ok
    T.to_csv(il.OUT_W / "retraction_trials.csv.gz", index=False)
    R = T[T.responded].copy()
    print(f"trials {len(T)}, cycles with a dock time {int(T.cycle_ok.sum())}, responded {len(R)}", flush=True)
    # latency bins from each animal's own pre responded trials
    R["lat_bin"] = -1
    for _a, g in R.groupby("animal"):
        pre = g[g.epoch == "pre"].first_lick_ms.to_numpy(float)
        if len(pre) < 50:
            continue
        edges = np.quantile(pre, np.linspace(0, 1, N_BINS + 1)[1:-1])
        R.loc[g.index, "lat_bin"] = np.searchsorted(edges, g.first_lick_ms.to_numpy(float), side="right")
    prate = R[R.epoch == "pre"].groupby(["animal", "lat_bin"]).retraction_bout.mean()
    R["expected"] = [prate.get((a, b), np.nan) for a, b in zip(R.animal, R.lat_bin)]
    sess, binrows = [], []
    for scope, g0 in [("all spouts", R)] + [(p, R[R.pos == p]) for p in POS6]:
        for (lab, a, ep), h in g0.groupby(["label", "animal", "epoch"]):
            if len(h) < 10:
                continue
            sess.append({"scope": scope, "label": lab, "animal": a, "epoch": ep, "n_responded": len(h),
                         "share_retraction": h.retraction_bout.mean(), "share_docked": h.docked_bout.mean(),
                         "latency_median_ms": h.first_lick_ms.median(),
                         "adjusted": h.retraction_bout.mean() - h.expected.mean()})
        if scope == "all spouts":
            for (lab, a, ep, b), h in g0[g0.lat_bin >= 0].groupby(["label", "animal", "epoch", "lat_bin"]):
                if len(h) >= 5:
                    binrows.append({"label": lab, "animal": a, "epoch": ep, "lat_bin": int(b), "n": len(h),
                                    "share_retraction": h.retraction_bout.mean()})
    S = pd.DataFrame(sess)
    S.to_csv(il.OUT_W / "retraction_sessions.csv", index=False)
    rng = np.random.default_rng(0)
    tests = []
    for scope, g in S.groupby("scope", sort=False):
        for q in ("share_retraction", "adjusted", "latency_median_ms", "share_docked"):
            tests += [{"scope": scope, "measure": q, **r} for r in _boot(g, q, rng)]
    B = pd.DataFrame(binrows)
    for b, g in B.groupby("lat_bin"):
        tests += [{"scope": f"latency bin {b + 1} of {N_BINS}", "measure": "share_retraction", **r}
                  for r in _boot(g, "share_retraction", rng)]
    TT = pd.DataFrame(tests)
    TT.to_csv(il.OUT_W / "retraction_tests.csv", index=False)
    B.to_csv(il.OUT_W / "retraction_latency_bins.csv", index=False)
    pd.set_option("display.width", 250)
    show = TT[TT.measure.isin(["share_retraction", "adjusted"]) & TT.scope.isin(["all spouts", "far_contra"])]
    print(show[["scope", "measure", "epoch", "pre_level", "point", "lo", "hi", "n_dir"]].round(3).to_string(index=False))
    print(TT[TT.scope.str.startswith("latency")][["scope", "epoch", "pre_level", "point", "lo", "hi", "n_dir"]]
          .round(3).to_string(index=False))
    print("->", il.OUT_W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
