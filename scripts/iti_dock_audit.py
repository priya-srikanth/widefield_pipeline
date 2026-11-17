"""DOCKED-SPOUT AUDIT of the ITI lick set (Priya 10-29: "ITI offset results restricted to licks made while the spout is
DOCKED"; the primary ITI set keeps every 'iti'-context lick, incl. those during the spout's retraction and its outbound
move to the next target; far1 of iti_lick_timing_audit only approximates docked).

ITI cycle (DECISIONS 2026-09-13, wfield_local.docked_periods): trial stop (trial_end TTL; the spout starts moving at
`dock_start`, measured here ~0 ... +10 ms after the DAQ stop) -> RETRACTION `dock_start` -> `dock` (0.65-0.98 s by
position) -> DOCKED `dock` -> next `trial_start` -> OUTBOUND `trial_start` -> arrival (`position` = the DAQ position
strobe) -> cue ~3 s later.

Per decoder ITI lick (the rows of the frozen prediction cache: covref_covariates.csv.gz set 'iti', key label x k, exact
DAQ peak time t), its ONSET (licks_all peak -> onset join, as iti_lick_timing_audit) is placed in the cycle of the last
trial stop before it:
  pre_retract   stop .. dock_start          (spout still at the target, a few ms)
  retract       dock_start .. dock          (spout moving back to the dock)
  docked        dock .. next trial_start    (spout parked: THE STRICT DOCKED WINDOW)
  outbound      trial_start .. arrival      (spout moving out to the next target)
  at_target     arrival .. (next stop)      (should not occur for ITI-context licks; counted)
  before_first  before the first trial's stop;  after_last  after the last dock with no later trial_start
Event times: the behavior log mapped device -> DAQ by the sync heartbeat (trial_windows._log_events_daq: dock_start,
dock, trial_start, position; source 'log'). Sessions whose log clock does not align (source 'reconstructed'): trial_start
= the DAQ `trial_start` digital line (log vs DAQ agree to a few ms where both exist), arrival = DAQ strobe, and
dock / dock_start = stop + the per-spout 95th percentile / median of (event - stop) measured over the log sessions of this
run (conservative: the reconstructed docked window opens after the true dock on ~95 % of trials; docked_periods'
p95 rule, here anchored on the measured stop rather than the cue).
gap = time since the previous lick onset (licks_all, as lick_recovery_markers._gaps; 'first ITI lick after quiet' =
gap >= 2 s, il.GAP_MAIN).
Filter flags (iti_licks.ITI_SUBSET_FILTERS): dock = phase docked; dock01 = onset in [dock + 0.1, trial_start - 0.1];
gap2dock = dock & gap >= 2; far1s = the far1 rule of iti_lick_timing_audit (>= 1 s from the window end and the next
arrival) as a SUBSET of the frozen predictions (no refit).
Outputs <OUT_W>/iti_dock_audit.csv (per lick), iti_dock_census.csv (per animal x epoch x lick set: n and share by
phase), iti_dock_sessions.csv (per session: source, n licks, reconstruction constants used).
Usage: python scripts/iti_dock_audit.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

PHASES = ["pre_retract", "retract", "docked", "outbound", "at_target", "before_first", "after_last"]
MARGIN = 0.1


def one(item) -> dict:
    """(label, t array) -> licks (onset, gap) + the session's event arrays on the DAQ clock. Module-level for spawn."""
    from wfield_local import config, daq_io, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    label, t = item
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    rv = PathResolver()
    A = pd.read_csv(oo.session_dir(rv, oc.RUN, an, date) / "licks_all.csv", usecols=["daq_on_s", "daq_peak_s"])
    A = A[np.isfinite(A.daq_on_s)].sort_values("daq_on_s").reset_index(drop=True)
    on = A.daq_on_s.to_numpy(float)
    m_on = pd.Series(on, index=A.daq_peak_s.to_numpy(float))
    m_gap = pd.Series(np.r_[np.inf, np.diff(on)], index=A.daq_peak_s.to_numpy(float))
    m_on, m_gap = m_on[~m_on.index.duplicated()], m_gap[~m_gap.index.duplicated()]
    t = np.asarray(t, float)
    T = il._behaviour(label).trials
    E = TW._log_events_daq(an, date, ("dock_start", "dock", "trial_start", "position"), rv)
    h5 = daq_io.session_h5(an, date, rv)
    with daq_io.open_daq(h5) as f:
        names, bits = daq_io.digital_bits(f)
        fs, _ = daq_io.session_attrs(f)
    ts_daq = daq_io.rising_edges(bits[:, names.index("trial_start")]) / fs if "trial_start" in names else np.array([])
    ev = {k: np.sort(v.to_numpy(float)) for k, v in E.items()} if E is not None else None
    df = pd.DataFrame(dict(label=label, animal=an, epoch=str(epochs.epoch_of(label)), t=t,
                           onset=m_on.reindex(t).to_numpy(float), gap=m_gap.reindex(t).to_numpy(float)))
    return {"label": label, "df": df, "ev": ev, "ts_daq": np.sort(ts_daq),
            "stop": T.stop_s.to_numpy(float), "strobe": T.strobe_s.to_numpy(float), "pos": T.pos.astype(str).to_numpy()}


def _first_after(ev: np.ndarray, x: np.ndarray, lo: float = 0.0) -> np.ndarray:
    j = np.searchsorted(ev, x + lo)
    return np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)], np.nan)


def trial_cycle(r: dict, const: dict | None) -> tuple[pd.DataFrame, str]:
    """Per trial (sorted by stop): stop, pos, dock_start, dock, next trial_start, next arrival. Log events when the log
    aligned, else the reconstruction (const: per-spout offsets after the stop)."""
    o = np.argsort(r["stop"])
    stop, pos = r["stop"][o], r["pos"][o]
    strobe = np.sort(r["strobe"][np.isfinite(r["strobe"])])
    C = pd.DataFrame({"stop": stop, "pos": pos})
    if r["ev"] is not None:
        ev = r["ev"]
        C["dock_start"] = _first_after(ev["dock_start"], stop, -0.2)
        C["dock"] = _first_after(ev["dock"], C.dock_start.to_numpy(), 0.0)
        C["trial_start"] = _first_after(ev["trial_start"], C.dock.to_numpy(), 0.0)
        C["arrival"] = _first_after(np.sort(ev["position"]), C.trial_start.to_numpy(), 0.0)
        # a log event that belongs to a LATER cycle (missing event) -> NaN: dock must precede the next stop
        nxt = np.r_[stop[1:], np.inf]
        for c in ("dock_start", "dock"):
            C.loc[C[c].to_numpy() >= nxt, c] = np.nan
        return C, "log"
    if const is None:
        return C, "none"
    C["dock_start"] = stop + C.pos.map(const["ds_med"]).to_numpy(float)
    C["dock"] = stop + C.pos.map(const["dock_p95"]).to_numpy(float)
    C["trial_start"] = _first_after(r["ts_daq"], C.dock.to_numpy(), 0.0)
    C["arrival"] = _first_after(strobe, C.trial_start.to_numpy(), 0.0)
    return C, "reconstructed"


def phases(df: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    x = df.onset.to_numpy(float)
    stop = C.stop.to_numpy(float)
    j = np.searchsorted(stop, x, side="right") - 1
    ph = np.full(len(x), "unknown", dtype=object)
    cols = {c: np.full(len(x), np.nan) for c in ("dock_start", "dock", "trial_start", "arrival")}
    ok = j >= 0
    ph[~ok & np.isfinite(x)] = "before_first"
    jj = np.maximum(j, 0)
    for c in cols:
        cols[c] = np.where(ok, C[c].to_numpy(float)[jj], np.nan)
    ds, dk, ts, ar = cols["dock_start"], cols["dock"], cols["trial_start"], cols["arrival"]
    with np.errstate(invalid="ignore"):
        sel = ok & np.isfinite(x)
        ph[sel & (x < ds)] = "pre_retract"
        ph[sel & (x >= ds) & (x < dk)] = "retract"
        ph[sel & (x >= dk) & (x < ts)] = "docked"
        ph[sel & (x >= dk) & ~np.isfinite(ts)] = "after_last"
        ph[sel & (x >= ts) & (x < ar)] = "outbound"
        ph[sel & (x >= ts) & ((x >= ar) | ~np.isfinite(ar))] = "at_target"
    out = df.copy()
    out["phase"] = ph
    out["since_dock"], out["to_trial_start"] = x - dk, ts - x
    out["since_stop"], out["to_arrival"] = x - np.where(ok, stop[jj], np.nan), ar - x
    return out


def flags(D: pd.DataFrame) -> pd.DataFrame:
    D["dock"] = D.phase == "docked"
    D["dock01"] = D.dock & (D.since_dock >= MARGIN) & (D.to_trial_start >= MARGIN)
    D["gap2dock"] = D.dock & (D.gap >= il.GAP_MAIN)
    D["far1s"] = (D.since_stop >= 1.0) & (D.to_arrival >= 1.0)
    # COMPLEMENT (interpretation check): licks while the spout MOVES (retraction incl. the few ms before it, outbound)
    D["notdock"] = D.phase.isin(["pre_retract", "retract", "outbound"])
    # SIZE-MATCHED complement (power control): per session x stratum (gap >= 2 s or not), a random draw of moving-spout
    # licks of the same number as that session's docked licks (seed 0; all of them if fewer)
    rng = np.random.default_rng(0)
    D["notdock_n"] = False
    q = (D.gap >= il.GAP_MAIN).to_numpy()
    for (_lab, _s), g in D.groupby([D.label, q], sort=True):
        cand = g.index[g.notdock.to_numpy()]
        n = min(int(g.dock.sum()), len(cand))
        if n:
            D.loc[np.sort(rng.choice(cand, n, replace=False)), "notdock_n"] = True
    return D


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    from wfield_local import analysis_kit as ak
    cov = pd.read_csv(il.OUT / "covref_covariates.csv.gz", usecols=["set", "label", "k", "t"])
    cov = cov[cov.set == "iti"]
    items = [(lab, g.sort_values("k").t.to_numpy(float)) for lab, g in cov.groupby("label", sort=True)]
    got, fail = ak.fan_sessions(items, one, jobs=a.jobs, label="dock audit", log=lambda m: None)
    for it, err in fail:
        print(f"  !! {it[0]}: {err}", flush=True)
    R = {r["label"]: r for _it, r in got}
    # reconstruction constants from the log sessions: per spout, (dock_start - stop) median, (dock - stop) p95
    rows = []
    for r in R.values():
        if r["ev"] is None:
            continue
        C, _src = trial_cycle(r, None)
        rows.append(C.assign(label=r["label"]))
    L = pd.concat(rows, ignore_index=True).dropna(subset=["dock_start", "dock"])
    L["d_ds"], L["d_dk"] = L.dock_start - L.stop, L.dock - L.stop
    L["travel"] = L.dock - L.dock_start
    sess_p95 = L.groupby(["label", "pos"]).d_dk.quantile(0.95).groupby("pos").median()
    const = {"ds_med": L.groupby("pos").d_ds.median().to_dict(), "dock_p95": sess_p95.to_dict()}
    K = pd.DataFrame({"dockstart_minus_stop_median": L.groupby("pos").d_ds.median(),
                      "dock_minus_stop_median": L.groupby("pos").d_dk.median(),
                      "dock_minus_stop_p95_sessmedian": sess_p95, "travel_median": L.groupby("pos").travel.median(),
                      "docked_dur_median": (L.trial_start - L.dock).groupby(L.pos).median(),
                      "outbound_median": (L.arrival - L.trial_start).groupby(L.pos).median()})
    print("== measured cycle constants per spout (log sessions, s after the DAQ trial stop)\n" + K.round(3).to_string())
    out, srows = [], []
    for lab in sorted(R):
        r = R[lab]
        C, src = trial_cycle(r, const)
        if r["ev"] is not None and len(r["ts_daq"]):                # log vs DAQ trial_start agreement (check)
            ts_log = r["ev"]["trial_start"]
            j = np.clip(np.searchsorted(r["ts_daq"], ts_log), 1, len(r["ts_daq"]) - 1)
            near = np.where(np.abs(r["ts_daq"][j] - ts_log) < np.abs(r["ts_daq"][j - 1] - ts_log), r["ts_daq"][j],
                            r["ts_daq"][j - 1])
            d_ts = float(np.median(np.abs(near - ts_log)))
        else:
            d_ts = np.nan
        P = phases(r["df"], C) if src != "none" else r["df"].assign(phase="unknown")
        P["src"] = src
        out.append(P)
        srows.append({"label": lab, "animal": r["df"].animal.iloc[0], "epoch": r["df"].epoch.iloc[0], "src": src,
                      "n_iti": len(P), "n_trials": len(C), "ts_log_vs_daq_median_abs_s": d_ts,
                      "share_docked": float((P.phase == "docked").mean())})
    D = flags(pd.concat(out, ignore_index=True))
    D["k"] = D.groupby("label").cumcount()
    D.to_csv(il.OUT_W / "iti_dock_audit.csv", index=False)
    S = pd.DataFrame(srows)
    S.to_csv(il.OUT_W / "iti_dock_sessions.csv", index=False)
    print("== sessions by source:\n" + S.groupby(["animal", "src"]).size().unstack(fill_value=0).to_string())
    print("   reconstructed: " + ", ".join(S[S.src != "log"].label))
    print(f"   log vs DAQ trial_start, median |diff| over sessions: {S.ts_log_vs_daq_median_abs_s.median() * 1e3:.1f} ms")
    print(f"   onset not found (peak join): {int(D.onset.isna().sum())} of {len(D)}")
    cen = []
    for setname, sel in (("all ITI", np.ones(len(D), bool)), ("first after 2 s quiet", (D.gap >= il.GAP_MAIN).to_numpy())):
        G = D[sel]
        for (an, ep), g in G.groupby(["animal", "epoch"]):
            rec = {"set": setname, "animal": an, "epoch": ep, "n": len(g), "n_sessions": g.label.nunique(),
                   "n_sessions_reconstructed": g[g.src != "log"].label.nunique()}
            for p in [*PHASES, "unknown"]:
                rec[f"share_{p}"] = float((g.phase == p).mean())
            for fl in ("dock", "dock01", "gap2dock", "far1s"):
                rec[f"n_{fl}"] = int(g[fl].sum())
            rec["share_far1s_docked"] = float(g[g.far1s].dock.mean()) if g.far1s.any() else np.nan
            cen.append(rec)
    CEN = pd.DataFrame(cen)
    ep_order = {"pre": 0, "acute": 1, "subacute": 2, "chronic": 3}
    CEN = CEN.sort_values(["set", "animal", "epoch"], key=lambda s: s.map(ep_order) if s.name == "epoch" else s)
    CEN.to_csv(il.OUT_W / "iti_dock_census.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["set", "animal", "epoch", "n", "share_retract", "share_docked", "share_outbound", "share_pre_retract",
            "share_at_target", "share_before_first", "share_after_last", "n_dock", "n_dock01", "share_far1s_docked"]
    print("== census per animal x epoch (rule 8)\n" + CEN[show].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
