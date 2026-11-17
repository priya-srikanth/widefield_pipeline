"""ITI negative-AUC diagnostic for lick_lateral_ratio (Priya 11-17): why is the contra or ipsi response integral <= 0?

Read-only. Step 1 (`--extract <out.pkl>`): rebuild lick_lateral_ratio's session x set x spout x pair curves from the
same inputs (`lick_cued_ramp._sessions`, under the env of the production run: all-ITI, or WFIELD_ITI_FILTER=dock), keeping
per-row diagnostics -- per-lick AUC SEM, baseline level vs an earlier window, the 0..0.5 / 0.5..1 s parts, for ITI rows
the timing of each bout start relative to the previous trial stop / spout dock / next trial start (iti_dock_audit.csv).
Step 2 (`--analyze <all.pkl> <dock.pkl> <out_dir>`): characterise the undefined rows and recompute the ratio's key
tests for alternative metrics (no production file is written).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "scripts"))


# ------------------------------------------------------------------------------------------------------------ EXTRACT
def extract(out: str) -> None:
    import iti_licks as il
    import iti_lick_order as lo
    import lick_cued_ramp as cr
    import lick_lateral_ratio as lr
    L = cr.LAGS_W
    D = cr._sessions()
    A = pd.read_csv(il.OUT / "iti_dock_audit.csv")
    AU = {x: g.dropna(subset=["onset"]).sort_values("onset") for x, g in A.groupby("label")}
    rows = []
    for x, d in D.items():
        # ITI event timing (the kept bout starts, in D order)
        s = il._load(x)
        keep = np.asarray(cr.np.load(cr.RIN / cr.WIDE / f"{x}.npz")["iti_gap"]) >= il.GAP_MAIN
        if il.ITI_SUBSET:
            keep &= il.bout_subset_keep(x, s["bout_t"])
        bt = np.asarray(s["bout_t"], float)[keep]
        tim = pd.DataFrame({"onset": bt})
        if x in AU and len(bt):
            g = AU[x]
            j = np.clip(np.searchsorted(g.onset.to_numpy(), bt), 0, len(g) - 1)
            jl = np.maximum(j - 1, 0)
            on = g.onset.to_numpy()
            near = np.where(np.abs(on[j] - bt) <= np.abs(on[jl] - bt), j, jl)
            okm = np.abs(on[near] - bt) <= 0.003
            for c in ["gap", "phase", "since_dock", "to_trial_start", "since_stop", "to_arrival"]:
                v = g[c].to_numpy()[near]
                tim[c] = np.where(okm, v, np.nan if c != "phase" else "")
        d["iti_tim"] = tim
        for st, (key, base, _xl, _t) in lr.SETS.items():
            X = d[key]
            pos = d["iti_pos"] if key == "iti" else d["pos"]
            for p in [*lo.POS, "pooled"]:
                sel = np.ones(len(X), bool) if p == "pooled" else (pos == p)
                P = X[sel]
                if len(P) < (cr.MIN_SPOUT if p in lo.POS else cr.MIN_N):
                    continue
                bmask = (L >= base[0]) & (L <= base[1])
                b0 = np.nanmean(P[:, bmask, :], 1, keepdims=True)
                Pb = P - b0
                m = np.nanmean(Pb, 0)
                raw = np.nanmean(P, 0)          # not baseline-subtracted (session mean)
                tim_r = d["iti_tim"][sel] if key == "iti" else None
                for a in lr.PAIRS:
                    if a == "orofacial":
                        ci = [d["cols"].index(f"{q}_right") for q in lr.OROFACIAL]
                        ii = [d["cols"].index(f"{q}_left") for q in lr.OROFACIAL]
                    else:
                        ci, ii = [d["cols"].index(f"{a}_right")], [d["cols"].index(f"{a}_left")]
                    rec = {"label": x, "animal": d["animal"], "epoch": d["epoch"], "set": st, "spout": p, "pair": a,
                           "n": len(P)}
                    for side, idx in (("c", ci), ("i", ii)):
                        rec[f"curve_{side}"] = m[:, idx].mean(1)
                        rec[f"raw_{side}"] = raw[:, idx].mean(1)
                        per = Pb[:, :, idx].mean(2)          # licks x lags
                        for w0, w1 in ((0, 1.0), (0, 0.5), (0.5, 1.0)):
                            w = (L >= w0 - 1e-9) & (L <= w1 + 1e-9)
                            pa = np.trapezoid(np.nan_to_num(per[:, w]), L[w], axis=1)
                            rec[f"{side}_sem_{w0:g}-{w1:g}"] = float(np.std(pa, ddof=1) / np.sqrt(len(pa)))
                    if tim_r is not None and len(tim_r):
                        rec["since_stop_med"] = float(np.nanmedian(tim_r.since_stop))
                        rec["since_dock_med"] = float(np.nanmedian(tim_r.since_dock))
                        rec["to_trial_med"] = float(np.nanmedian(tim_r.to_trial_start))
                        rec["gap_med"] = float(np.nanmedian(tim_r.gap))
                        rec["frac_dock_in_base"] = float(np.mean((tim_r.since_dock > 0.3) & (tim_r.since_dock < 1.3)))
                        rec["frac_cue_in_win"] = float(np.mean((tim_r.to_trial_start > 0) & (tim_r.to_trial_start < 1.0)))
                    rows.append(rec)
    C = pd.DataFrame(rows)
    with open(out, "wb") as f:
        pickle.dump({"C": C, "L": L, "filter": il.ITI_SUBSET}, f)
    print(f"wrote {out}: {len(C)} rows, filter={il.ITI_SUBSET!r}")


def events(out: str) -> None:
    """Per ITI bout start (the kept set): following-lick count, timing, and per pair x hemisphere the baseline-subtracted
    AUCs 0..0.5 / 0.5..1 / 0..1 s, peak 0..0.5 s, trough 0.5..1.2 s, raw baseline (vs the session mean)."""
    import iti_licks as il
    import lick_cued_ramp as cr
    import lick_lateral_ratio as lr
    L = cr.LAGS_W
    D = cr._sessions()
    A = pd.read_csv(il.OUT / "iti_dock_audit.csv")
    AU = {x: g.dropna(subset=["onset"]).sort_values("onset") for x, g in A.groupby("label")}
    bm = (L >= -1.0) & (L <= -0.5)
    rows = []
    for x, d in D.items():
        s = il._load(x)
        z = np.load(cr.RIN / cr.WIDE / f"{x}.npz")
        keep = np.asarray(z["iti_gap"]) >= il.GAP_MAIN
        if il.ITI_SUBSET:
            keep &= il.bout_subset_keep(x, s["bout_t"])
        bt = np.asarray(s["bout_t"], float)[keep]
        nfol = np.asarray(z["iti_nfol"])[keep]
        X = d["iti"]
        assert len(X) == len(bt)
        tim = {}
        if x in AU and len(bt):
            g = AU[x]
            on = g.onset.to_numpy()
            j = np.clip(np.searchsorted(on, bt), 0, len(g) - 1)
            jl = np.maximum(j - 1, 0)
            near = np.where(np.abs(on[j] - bt) <= np.abs(on[jl] - bt), j, jl)
            okm = np.abs(on[near] - bt) <= 0.003
            for c in ["since_dock", "to_trial_start", "since_stop"]:
                tim[c] = np.where(okm, g[c].to_numpy()[near], np.nan)
        rec = pd.DataFrame({"label": x, "animal": d["animal"], "epoch": d["epoch"], "spout": d["iti_pos"],
                            "onset": bt, "nfol": nfol, **tim})
        for a in ("tjM1", "SSp-n", "orofacial"):
            for side, hemi in (("c", "right"), ("i", "left")):
                q = lr.OROFACIAL if a == "orofacial" else [a]
                v = X[:, :, [d["cols"].index(f"{k}_{hemi}") for k in q]].mean(2)
                vb = v - np.nanmean(v[:, bm], 1, keepdims=True)
                for w0, w1 in ((0, 0.5), (0.5, 1.0), (0, 1.0)):
                    w = (L >= w0 - 1e-9) & (L <= w1 + 1e-9)
                    rec[f"{a}_{side}_{w0:g}-{w1:g}"] = np.trapezoid(np.nan_to_num(vb[:, w]), L[w], axis=1)
                rec[f"{a}_{side}_peak"] = np.nanmax(vb[:, (L >= 0) & (L <= 0.5)], 1)
                rec[f"{a}_{side}_trough"] = np.nanmin(vb[:, (L >= 0.5) & (L <= 1.2)], 1)
                rec[f"{a}_{side}_rawbase"] = np.nanmean(v[:, bm], 1)
        rows.append(rec)
    E = pd.concat(rows, ignore_index=True)
    E.to_pickle(out)
    print(f"wrote {out}: {len(E)} events")


if __name__ == "__main__":
    if sys.argv[1] == "--extract":
        extract(sys.argv[2])
    elif sys.argv[1] == "--events":
        events(sys.argv[2])
    else:
        import neg_auc_analyze as na  # noqa: E402  (step 2 lives next to this file)
        na.main(sys.argv[2:])
