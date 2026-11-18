"""READ-ONLY CENSUS (Priya 11-18): what would change if the pre-cue lick-free (ENL) window and the ITI lick-free windows
also removed the KEPT cam1-only licks (cam1 `events_all`, `cam1_licks.keep_mask`: not a droplet, med_out_px >= -120 px,
not cam4-matched) -- the lick train RESTnlc1 already uses. Nothing is re-run; this only counts.

    python scripts/enl_iti_cam1_census.py [--jobs 4] [--labels ...]

Run under the rebuild render profile (`python -m wfield_local.render_profile cohort1_rebuild_20261103 --shell bash`).
WIDEFIELD_LICK_SOURCE is forced to lp (the 1.0 s ENL window is the LP source's window).

ENL. Current train = `lp_licks.gate_licks_s` (the edited LP train `_trial_features` / the time-based ENL scripts use:
cam4 licks_all + cam1 POST-CUE events only; cues without an LP verdict dropped). New train = current UNION the kept
cam1-only `events_all` onsets (`restnl_licks.cam1_only_onsets`). Window search = `lp_licks.enl_window_starts_s`
(the decoders' `precue_window_start`: fixed [cue - W, cue) if clean, else latest lick-free gap in [strobe, cue), else
drop) for W = 1.0 (primary) and 1.5 s; and the trace families' `enl_window.sliding_lags` (+ 0.3 s settle, >= -2 s).
Strobe / cue / stop = `trial_windows.trial_bounds`. Population: cues with an LP verdict; ENGAGED = next current-train
lick within decode.max_rt_s (the `enl_movement_multisession.enl_table` population; the camera working-trial gate is
not re-applied here).

ITI (two definitions in the code):
  A  iti_licks bout starts (goal Fig 14.x ITI family): cam4 licks_all ITI licks whose preceding no-lick gap (vs cam4
     licks_all + DAQ lick onsets; onsets within 50 ms = the same lick) is >= 2 s (main) / >= 1 s, no cue / strobe in
     [-2, +0.75] s; "isolated" = no lick 0.05..0.75 s after. Census: bout starts that lose that status when the kept
     cam1-only onsets join the train.
  B  quiet_task_events T12 ITI mask: ITI = trial stop .. next strobe; clean = no DAQ contact lick (QS_LICK_SOURCE=daq,
     the default) and no running bout within 0.5 s. Census on a 10 ms grid: clean ITI time removed by +-0.5 s around
     the kept cam1-only onsets; ITIs losing all clean time / losing their >= 1 s clean stretch.
COVERED = a kept cam1-only onset within 0.1 s of an existing DAQ or current-train (cam4) lick onset.
Output: <rebuild>/labcams/grant_figures/enl_iti_cam1_census_20261118/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["WIDEFIELD_LICK_SOURCE"] = "lp"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/enl_iti_cam1_census_20261118")
EVENTS = Path("M:/MICROSCOPE/Priya/Behavior_logs/Widefield/behavior_summary/events")
WINS = (1.0, 1.5)
COVER_S = 0.1
POS_NAME = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
            "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}
ITI_BUF = 0.5
GRID = 0.01


def _sid(animal, date, rv):
    csvs = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))
    if csvs:
        return csvs[-1].name[:-11]
    from wfield_local.docked_periods import behaviour_session_dir
    return Path(behaviour_session_dir(f"{animal}_{date[4:]}")).name


def _near(t, ref, tol):
    """bool per t: some ref within +-tol."""
    ref = np.sort(ref[np.isfinite(ref)])
    if not len(ref) or not len(t):
        return np.zeros(len(t), bool)
    j = np.clip(np.searchsorted(ref, t), 1, len(ref) - 1)
    return np.minimum(np.abs(ref[j - 1] - t), np.abs(ref[j] - t)) <= tol


def _count_in(L, a, b):
    return np.searchsorted(L, b, side="left") - np.searchsorted(L, a, side="left")


def session(label):
    from wfield_local import cam1_licks as C1
    from wfield_local import config, dlc_frames, enl_window, epochs, lp_licks
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import restnl_licks as RL
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, mmdd = label.split("_")
    date = config.date8(mmdd)
    ep = epochs.epoch_of(label)
    max_rt = float(config.defaults()["decode"]["max_rt_s"])
    b = TW.trial_bounds(animal, date, rv)
    b = b[np.isfinite(b.cue_s)].sort_values("cue_s").reset_index(drop=True)
    daq = np.sort(np.asarray(dlc_frames.lick_onsets(animal, date, _sid(animal, date, rv), rv), float))
    try:
        cur, has = lp_licks.gate_licks_s(label, b.cue_s.to_numpy(float), daq, max_rt)
    except lp_licks.LPUnavailable as ex:
        return {"label": label, "skip": f"no LP ({ex})"}
    cur = np.sort(cur[np.isfinite(cur)])
    f1 = RL.cam1_events_path(label)
    if not f1.exists():
        return {"label": label, "skip": "no cam1 events_all"}
    E = pd.read_csv(f1, usecols=lambda c: c in ("daq_on_s", "cam4_matched", "droplet", "spread_px", "range_px",
                                                "med_out_px", "context"))
    m4 = C1.flag_col(E["cam4_matched"])
    c1 = RL.cam1_only_onsets(E, label)
    n_c1_only_all = int((~m4).sum())
    new = np.union1d(cur, c1)
    cov_ref = np.union1d(cur, daq)
    c1_cov = _near(c1, cov_ref, COVER_S)

    cue = b.cue_s.to_numpy(float)
    strobe = b.strobe_s.to_numpy(float)
    stop = b.stop_s.to_numpy(float)
    pos = b.pos_name.astype(str).map(lambda p: POS_NAME.get(p, p)).to_numpy()
    j = np.searchsorted(cur, cue)
    nxt = np.where(j < len(cur), cur[np.minimum(j, len(cur) - 1)], np.inf)
    engaged = has & ((nxt - cue) <= max_rt)
    T = pd.DataFrame({"label": label, "animal": animal, "epoch": ep, "trial": np.arange(len(b)), "position": pos,
                      "has_lp": has, "engaged": engaged, "cue_s": cue, "strobe_s": strobe})
    # pre-cue kept cam1-only licks per trial: [strobe, cue) and [cue - 1, cue)
    st_lo = np.where(np.isfinite(strobe) & (strobe < cue), strobe, cue)
    T["c1_precue_n"] = _count_in(c1, st_lo, cue)
    T["c1_last1s_n"] = _count_in(c1, cue - 1.0, cue)
    c1_pre_mask = np.zeros(len(c1), bool)
    for a, z in zip(st_lo, cue):
        c1_pre_mask[np.searchsorted(c1, a):np.searchsorted(c1, z)] = True
    for W in WINS:
        tag = f"{W:g}"
        w0 = np.where(has, lp_licks.enl_window_starts_s(cue, strobe, cur, W), np.nan)
        w1 = np.where(has, lp_licks.enl_window_starts_s(cue, strobe, new, W), np.nan)
        T[f"w0cur_{tag}"] = w0
        T[f"w0new_{tag}"] = w1
        inwin = np.where(np.isfinite(w0), _count_in(c1, np.nan_to_num(w0), np.nan_to_num(w0) + W), 0)
        T[f"c1_inwin_{tag}"] = inwin
        # families' rule (settle 0.3 s, traces from -2 s)
        lo0, _ = enl_window.sliding_lags(cue, strobe, cur, W, span_lo=-2.0)
        lo1, _ = enl_window.sliding_lags(cue, strobe, new, W, span_lo=-2.0)
        T[f"fam_cur_{tag}"] = np.where(has, lo0, np.nan)
        T[f"fam_new_{tag}"] = np.where(has, lo1, np.nan)

    # ---------------- ITI A: iti_licks bout starts
    sd = oo.session_dir(rv, oc.RUN, animal, date)
    A = pd.read_csv(sd / "licks_all.csv", usecols=["context", "daq_on_s"])
    t_all = np.sort(np.concatenate([A.daq_on_s.to_numpy(float), daq]))
    t_all = t_all[np.isfinite(t_all)]
    t_new = np.union1d(t_all, c1)
    to = np.sort(A.daq_on_s[(A.context == "iti")].to_numpy(float))
    to = to[np.isfinite(to)]
    ev = np.sort(np.concatenate([cue, strobe]))
    ev = ev[np.isfinite(ev)]

    def gap_of(train, t):
        k = np.searchsorted(train, t - 0.05)
        return t - np.where(k > 0, train[np.maximum(k - 1, 0)], -np.inf)

    def nfol(train, t):
        return np.searchsorted(train, t + 0.75, side="right") - np.searchsorted(train, t + 0.05)
    je = np.searchsorted(ev, to - 2.0)
    near_ev = (je < len(ev)) & (ev[np.minimum(je, max(len(ev) - 1, 0))] <= to + 0.75) if len(ev) else np.zeros(len(to), bool)
    g0, g1 = gap_of(t_all, to), gap_of(t_new, to)
    f0, f1n = nfol(t_all, to), nfol(t_new, to)
    iti = {}
    for G in (1.0, 2.0):
        bs0 = (g0 >= G) & ~near_ev
        bs1 = (g1 >= G) & ~near_ev
        iti[f"itiA_bout{G:g}_cur"] = int(bs0.sum())
        iti[f"itiA_bout{G:g}_lost"] = int((bs0 & ~bs1).sum())
        iti[f"itiA_bout{G:g}_iso_cur"] = int((bs0 & (f0 == 0)).sum())
        iti[f"itiA_bout{G:g}_iso_lost"] = int((bs0 & (f0 == 0) & ~(bs1 & (f1n == 0))).sum())
    # ---------------- ITI B: quiet_task_events T12 mask
    z = np.load(EVENTS / animal / f"{date}.npz", allow_pickle=True)
    fsd = float(z["fs"])
    qlick = np.sort(z["lick_onsets"] / fsd)
    run = np.c_[z["running_starts"], z["running_stops"]] / fsd
    prev_stop = np.r_[-np.inf, stop[:-1]]
    start = np.where(np.isfinite(strobe) & (strobe < cue) & (strobe >= prev_stop), strobe, np.nan)
    clean0 = clean1 = 0.0
    n_iti = n_iti_any0 = n_iti_lose_all = n_iti_1s0 = n_iti_lose_1s = 0
    c1_iti_mask = np.zeros(len(c1), bool)
    for i in range(len(b) - 1):
        a, e = stop[i], start[i + 1]
        if not (np.isfinite(a) and np.isfinite(e) and e > a):
            continue
        n_iti += 1
        g = np.arange(a, e, GRID)
        bad = np.zeros(len(g), bool)
        for x in qlick[np.searchsorted(qlick, a - ITI_BUF):np.searchsorted(qlick, e + ITI_BUF)]:
            bad[np.searchsorted(g, x - ITI_BUF):np.searchsorted(g, x + ITI_BUF)] = True
        for r0, r1 in run[(run[:, 1] >= a - ITI_BUF) & (run[:, 0] <= e + ITI_BUF)]:
            bad[np.searchsorted(g, r0 - ITI_BUF):np.searchsorted(g, r1 + ITI_BUF)] = True
        bad1 = bad.copy()
        k0, k1 = np.searchsorted(c1, a - ITI_BUF), np.searchsorted(c1, e + ITI_BUF)
        c1_iti_mask[np.searchsorted(c1, a):np.searchsorted(c1, e)] = True
        for x in c1[k0:k1]:
            bad1[np.searchsorted(g, x - ITI_BUF):np.searchsorted(g, x + ITI_BUF)] = True
        cl0, cl1 = (~bad).sum() * GRID, (~bad1).sum() * GRID
        clean0 += cl0
        clean1 += cl1
        if cl0 > 0:
            n_iti_any0 += 1
            n_iti_lose_all += cl1 == 0

        def longest(m):
            if not m.any():
                return 0.0
            d = np.diff(np.r_[0, m.astype(np.int8), 0])
            return float((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max() * GRID)
        l0, l1 = longest(~bad), longest(~bad1)
        if l0 >= 1.0:
            n_iti_1s0 += 1
            n_iti_lose_1s += l1 < 1.0
    S = {"label": label, "animal": animal, "epoch": ep, "n_cues": len(b), "n_lp": int(has.sum()),
         "n_c1_only_events": n_c1_only_all, "n_c1_kept": len(c1),
         "n_c1_kept_precue": int(c1_pre_mask.sum()), "n_c1_kept_precue_covered": int((c1_pre_mask & c1_cov).sum()),
         "n_c1_kept_iti": int(c1_iti_mask.sum()), "n_c1_kept_iti_covered": int((c1_iti_mask & c1_cov).sum()),
         "n_c1_kept_covered": int(c1_cov.sum()),
         "itiB_n": n_iti, "itiB_clean_s_cur": clean0, "itiB_clean_s_new": clean1, "itiB_n_any_clean": n_iti_any0,
         "itiB_n_lose_all": int(n_iti_lose_all), "itiB_n_1s_clean": n_iti_1s0, "itiB_n_lose_1s": int(n_iti_lose_1s),
         **iti}
    return {"label": label, "S": S, "T": T}


def classify(T, W, fam=False):
    tag = f"{W:g}"
    a, b_ = (f"fam_cur_{tag}", f"fam_new_{tag}") if fam else (f"w0cur_{tag}", f"w0new_{tag}")
    v0, v1 = np.isfinite(T[a]), np.isfinite(T[b_])
    same = v0 & v1 & (np.abs(T[a] - T[b_]) < 1e-6)
    st = np.where(~v0, "invalid_now", np.where(same, "unchanged", np.where(v1, "shift", "drop")))
    return st, np.where(st == "shift", T[a] - T[b_], np.nan)


def summarize(T, S):
    rows = []
    for (an, ep), g in T.groupby(["animal", "epoch"], sort=False):
        for pop in ("engaged", "all_lp"):
            gg = g[g.engaged] if pop == "engaged" else g[g.has_lp]
            for W in WINS:
                for fam in (False, True):
                    st, sh = classify(gg, W, fam)
                    for sub, m in (("all", np.ones(len(gg), bool)), ("far_contra", gg.position.eq("far_contra").to_numpy()),
                                   ("not_far_contra", ~gg.position.eq("far_contra").to_numpy())):
                        nv = int(((st != "invalid_now") & m).sum())
                        nsh, ndr = int(((st == "shift") & m).sum()), int(((st == "drop") & m).sum())
                        rows.append({"animal": an, "epoch": ep, "pop": pop, "win_s": W,
                                     "rule": "families_settle" if fam else "decoder", "subset": sub,
                                     "n_trials": int(m.sum()), "n_valid_now": nv,
                                     "n_c1_in_window": nsh + ndr, "pct_c1_in_window": 100 * (nsh + ndr) / max(nv, 1),
                                     "n_shift": nsh, "n_drop": ndr, "pct_shift": 100 * nsh / max(nv, 1),
                                     "pct_drop": 100 * ndr / max(nv, 1),
                                     "median_shift_s": float(np.nanmedian(sh[m & (st == "shift")])) if ((st == "shift") & m).any() else np.nan,
                                     "c1_precue_per_trial": float(gg.c1_precue_n[m].mean()) if m.any() else np.nan,
                                     "pct_trials_any_c1_precue": 100 * float((gg.c1_precue_n[m] > 0).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--labels", nargs="*")
    a = ap.parse_args()
    from wfield_local import analysis_kit as ak
    labels = a.labels or ak.curated_labels(animals=("PS92", "PS93", "PS94", "PS95"))
    OUT.mkdir(parents=True, exist_ok=True)
    res, fail = ak.fan_sessions(labels, session, jobs=a.jobs)
    for f in fail:
        print("FAIL", f, flush=True)
    good = [v for _, v in res if "S" in v]
    skips = [(v["label"], v["skip"]) for _, v in res if "skip" in v]
    print("skipped:", skips, flush=True)
    T = pd.concat([v["T"] for v in good], ignore_index=True)
    S = pd.DataFrame([v["S"] for v in good])
    for W in WINS:
        st, sh = classify(T, W)
        T[f"status_{W:g}"], T[f"shift_s_{W:g}"] = st, sh
        st, sh = classify(T, W, fam=True)
        T[f"status_fam_{W:g}"], T[f"shift_fam_s_{W:g}"] = st, sh
    T.to_csv(OUT / "trials.csv.gz", index=False)
    # per-session ENL (engaged, decoder rule) into the session table
    for W in WINS:
        g = T[T.engaged].groupby("label")
        S = S.merge(pd.DataFrame({
            f"enl{W:g}_valid_now": g[f"status_{W:g}"].apply(lambda s: (s != "invalid_now").sum()),
            f"enl{W:g}_shift": g[f"status_{W:g}"].apply(lambda s: (s == "shift").sum()),
            f"enl{W:g}_drop": g[f"status_{W:g}"].apply(lambda s: (s == "drop").sum())}).reset_index(), on="label", how="left")
        S[f"enl{W:g}_pct_changed"] = 100 * (S[f"enl{W:g}_shift"] + S[f"enl{W:g}_drop"]) / S[f"enl{W:g}_valid_now"].clip(lower=1)
        S[f"enl{W:g}_pct_drop"] = 100 * S[f"enl{W:g}_drop"] / S[f"enl{W:g}_valid_now"].clip(lower=1)
    S["itiB_pct_clean_removed"] = 100 * (1 - S.itiB_clean_s_new / S.itiB_clean_s_cur.clip(lower=1e-9))
    S.to_csv(OUT / "sessions.csv", index=False)
    A = summarize(T, S)
    A.to_csv(OUT / "enl_animal_epoch.csv", index=False)
    num = [c for c in S.columns if c.startswith(("itiA_", "itiB_n", "itiB_clean", "n_"))]
    I = S.groupby(["animal", "epoch"], sort=False)[num].sum().reset_index()
    I["itiB_pct_clean_removed"] = 100 * (1 - I.itiB_clean_s_new / I.itiB_clean_s_cur)
    I["itiB_pct_lose_1s"] = 100 * I.itiB_n_lose_1s / I.itiB_n_1s_clean
    for G in ("1", "2"):
        I[f"itiA_bout{G}_pct_lost"] = 100 * I[f"itiA_bout{G}_lost"] / I[f"itiA_bout{G}_cur"].clip(lower=1)
        I[f"itiA_bout{G}_iso_pct_lost"] = 100 * I[f"itiA_bout{G}_iso_lost"] / I[f"itiA_bout{G}_iso_cur"].clip(lower=1)
    I.to_csv(OUT / "iti_animal_epoch.csv", index=False)
    print(f"wrote {OUT}; sessions {len(S)}; skipped {len(skips)}; failed {len(fail)}", flush=True)


if __name__ == "__main__":
    main()
