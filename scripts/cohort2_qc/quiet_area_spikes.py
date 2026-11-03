"""Spontaneous activity spikes in orofacial vs limb cortex during QUIET periods, cohort1 pre vs post stroke (2026-10-31).

Priya 10-31: "use our existing cohort 1 analyzed data to try to look at spontaneous activity spikes in alm/tjm1/ssp-m/n/ul
or in ssp-ul/ll during non-lick non-running periods pre vs post stroke". Follows quiet_modes.py (PS93 08-20 / 08-21
orofacial and limb cortex alternate while the animal sits still).

DATA: the encoder's inputs, unchanged -- `combined_encoding.load_imaging` (adopted hemo-corrected SVD on the shared
Allen grid, eroded stat mask; frame times on the DAQ clock). AREAS: Allen areas per hemisphere plus ALM / tjM1 =
anterolateral 40% of MOs / MOp (`iti_licks.roi_masks`, Priya 10-25). _left = ipsilesional, _right = contralesional.
GROUPS: orofacial = ALM, tjM1, SSp-m, SSp-n; limb = SSp-ul, SSp-ll (both hemispheres; each area also kept alone).
REST (Priya 10-31: "why not use restw"): the pipeline's canonical rest windows, restdock05 (`behavior_events`
rest_starts / rest_stops: spout docked -> next trial_start, not running, lick buffer [0.5, 1.0] s; ~2 s per trial),
swapped pairs (+/-3) dropped (cohort1 was never repaired before its SVD). Each rest frame is also flagged ENGAGED
(before the animal's quit, `analysis_kit.session_behavior` / quit_point) so the quit-tail state can be split out:
post-stroke quiet sits mostly in the disengaged tail, pre-stroke has almost none.
SIGNAL: area-mean dF/F % minus its 2-s running median over the whole session (Priya 10-31: the events are "very
brief").
SPIKE: peaks found on the WHOLE trace (so prominence has real context), half-width 1.5 frames (~50 ms) to 0.5 s, peak inside rest;
prominence > 5 x the ANIMAL's pre-stroke robust SD of that trace over ENGAGED rest frames (1.4826 x MAD).
LOCAL = cortex-mean fluctuation at the peak < 0.5 x the area's. INDEPENDENT (groups) = the OTHER group's fluctuation
at the peak < 0.5 x this group's (orofacial vs limb); CO-OCCUR = an other-group spike within 1 s.
Per session x rest set (all / engaged / tail): rest minutes, rate (/min of rest) of all and local spikes per area / group, quiet SD.
Printed per animal x epoch first (CLAUDE.md rule 8).

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_area_spikes.py [--jobs 6] [--summarise]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # scripts/ (iti_licks)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # repo root (analysis_kit imports scripts.*)
# QS_OUT_ROOT (2026-11-02, cohort-1 swap-mask re-run): a sensitivity run on another hemo variant
# (WIDEFIELD_HEMO_VARIANT) writes to its OWN root, so it can never overwrite or mix with the adopted outputs. Unset = the
# adopted location, unchanged.
def _out_root() -> Path:
    """QS_OUT_ROOT, else the adopted location routed by the run mode (rebuild 11-03: a tagged run -> .../<tag>/
    cohort1_quiet_spikes, so its skip-if-exists `sessions/` never serves or overwrites the default ones)."""
    if os.environ.get("QS_OUT_ROOT"):
        return Path(os.environ["QS_OUT_ROOT"])
    from wfield_local import lp_licks
    return lp_licks.output_dir(Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes"))


OUT = _out_root()
EVENTS = Path("M:/MICROSCOPE/Priya/Behavior_logs/Widefield/behavior_summary/events")
AREAS = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSp-ul", "SSp-ll"]
GROUPS = {"orofacial": ["ALM", "tjM1", "SSp-m", "SSp-n"], "limb": ["SSp-ul", "SSp-ll"]}
K_SD, MAX_HW_S, LOCAL, DETREND_S, CAND_SD = 5.0, 0.5, 0.5, 2.0, 2.0
MIN_HW_FR = 1.5        # half-width >= 1.5 frames (~50 ms): a one-frame swapped-pair flash cannot pass
EP = ["pre", "acute", "subacute", "chronic"]


def columns():
    cols = [f"{a}_{h}" for a in AREAS for h in ("left", "right")]
    return cols + [f"{g}_{h}" for g in GROUPS for h in ("left", "right")] + list(GROUPS) + ["cortex"]


def session_traces(label, detrend=True):
    """One session -> (X cols x T detrended dF/F %, ft, fs_img, rest mask, engaged mask, quit_s). Shared with
    quiet_spikes_fr.py (the FaceRhythm-regressed pass) so both passes read the identical traces and rest frames."""
    from scipy.ndimage import median_filter
    from iti_licks import roi_masks
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    m = roi_masks(parea, pstat, names, flat, shape)
    s = next(x for x in config.load_sessions() if x["label"] == label)
    W = []
    for c in columns():
        if c == "cortex":
            mk = m["cortex"]
        elif c in GROUPS:
            mk = np.any([m[f"{a}_{h}"] for a in GROUPS[c] for h in ("left", "right") if f"{a}_{h}" in m], 0)
        elif c.rsplit("_", 1)[0] in GROUPS:
            g, h = c.rsplit("_", 1)
            mk = np.any([m[f"{a}_{h}"] for a in GROUPS[g] if f"{a}_{h}" in m], 0)
        else:
            mk = m.get(c, np.zeros(len(flat), bool))
        W.append(Ub[mk].mean(0) if mk.any() else np.full(Ub.shape[1], np.nan, np.float32))
    X = 100 * (np.asarray(W, np.float64) @ V.T.astype(np.float64))                # cols x T, dF/F %
    fs_img = 1.0 / np.median(np.diff(ft))
    w = int(round(DETREND_S * fs_img)) | 1
    if detrend:                              # detrend=False: raw dF/F % (quiet_spikes_v2 injects synthetic events first)
        X = X - median_filter(X, (1, w), mode="nearest")
    T = X.shape[1]
    from wfield_local.analysis_kit import session_behavior
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fs = float(z["fs"])
    rest = np.zeros(T, bool)
    for a, b in zip(z["rest_starts"] / fs, z["rest_stops"] / fs):
        rest[np.searchsorted(ft, a):np.searchsorted(ft, b)] = True
    sw = Path(s["mc"]) / "wfield_local_results" / "swapped_pairs.npz"
    if sw.exists():
        bad = np.load(sw)["bad"][:T].astype(float)
        rest[:len(bad)] &= ~(np.convolve(bad, np.ones(7), "same") > 0)
    engaged = np.ones(T, bool)
    quit_s = np.nan
    bh = session_behavior(label)
    if bh is not None and not bh.quit["censored"]:
        k = sum(float(r["elapsed_s"]) < float(bh.quit["quit_elapsed_s"]) for r in bh.trials)
        if k < len(bh.cue_samples):
            quit_s = bh.cue_samples[k] / bh.sample_rate
            engaged = ft < quit_s
    return X, ft, fs_img, rest, engaged, quit_s


def mad(Y):
    """Robust SD per row (1.4826 x MAD)."""
    return 1.4826 * np.median(np.abs(Y - np.median(Y, 1, keepdims=True)), 1)


def extract(label):
    """One session -> candidate rest spikes per column (see the module docstring). Module-level for spawn."""
    f = OUT / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    from scipy.signal import find_peaks
    X, ft, fs_img, rest, engaged, quit_s = session_traces(label)
    sd_rest = mad(X[:, rest]) if rest.any() else np.full(len(X), np.nan)
    sd_eng = mad(X[:, rest & engaged]) if (rest & engaged).any() else np.full(len(X), np.nan)
    cols = columns()
    C = []                                  # candidate spikes: col, frame, prominence, cortex, other group
    for i, c in enumerate(cols):
        if c == "cortex" or not np.isfinite(X[i]).all() or not np.isfinite(sd_rest[i]):
            continue
        pk, pr = find_peaks(X[i], prominence=CAND_SD * sd_rest[i], width=(MIN_HW_FR, MAX_HW_S * fs_img), rel_height=0.5)
        keep = rest[pk]
        pk, prom = pk[keep], pr["prominences"][keep]
        oth = c.replace("orofacial", "@").replace("limb", "orofacial").replace("@", "limb")
        ov = X[cols.index(oth), pk] if oth != c and oth in cols else np.full(len(pk), np.nan)
        C += [(i, int(p), float(q), float(X[-1, p]), float(o)) for p, q, o in zip(pk, prom, ov)]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, cand=np.array(C, float).reshape(-1, 5), sd_rest=sd_rest, sd_eng=sd_eng, fs=fs_img,
                        cols=np.array(cols), n_rest=int(rest.sum()), n_rest_eng=int((rest & engaged).sum()),
                        engaged_frame=engaged.astype(np.uint8)[::1], quit_s=quit_s, total_min=X.shape[1] / fs_img / 60)
    return label


SETS = ("all", "engaged", "tail")
SHOW = ["orofacial_left", "orofacial_right", "limb_left", "limb_right", "orofacial", "limb"]


def summarise():
    import pandas as pd
    from wfield_local import epochs
    D = {f.stem: dict(np.load(f)) for f in sorted((OUT / "sessions").glob("*.npz"))}
    cols = columns()
    thr = {}                                    # 5 x the animal's pre-stroke ENGAGED-rest robust SD (median over sessions)
    for an in sorted({k.split("_")[0] for k in D}):
        pre = np.array([D[k]["sd_eng"] for k in D if k.startswith(an) and epochs.epoch_of(k) == "pre"])
        thr[an] = K_SD * np.nanmedian(pre, 0)
    rows = []
    for k, d in D.items():
        ep = epochs.epoch_of(k)
        if ep is None:
            continue
        an, fs, cand, eng = k.split("_")[0], float(d["fs"]), d["cand"], d["engaged_frame"].astype(bool)
        n_set = {"all": int(d["n_rest"]), "engaged": int(d["n_rest_eng"]), "tail": int(d["n_rest"] - d["n_rest_eng"])}
        ci, fr, prom, ctx, oth = (cand[:, j] for j in range(5))
        ci, fr = ci.astype(int), fr.astype(int)
        above = prom > thr[an][ci]
        in_eng = eng[np.minimum(fr, len(eng) - 1)]
        for st in SETS:
            mins = n_set[st] / fs / 60
            if mins < 0.25:
                continue
            inset = above & (in_eng if st == "engaged" else ~in_eng if st == "tail" else True)
            r = dict(session=k, animal=an, epoch=ep, day=epochs.days_since_stroke(k), set=st, rest_min=mins,
                     total_min=float(d["total_min"]))
            for i, c in enumerate(cols[:-1]):
                m = inset & (ci == i)
                r[f"{c}|sd_eng"] = float(d["sd_eng"][i])          # threshold-drift check: this session's rest SD
                r[f"{c}|rate"] = m.sum() / mins
                r[f"{c}|local_rate"] = (m & (ctx < LOCAL * prom)).sum() / mins
                if c.startswith(("orofacial", "limb")):
                    oc = c.replace("orofacial", "@").replace("limb", "orofacial").replace("@", "limb")
                    r[f"{c}|indep_rate"] = (m & (oth < LOCAL * prom)).sum() / mins
                    of = fr[inset & (ci == cols.index(oc))]
                    near = [np.any(np.abs(of - x) <= fs) for x in fr[m]]
                    r[f"{c}|cooccur"] = float(np.mean(near)) if len(near) else np.nan
            rows.append(r)
    S = pd.DataFrame(rows).sort_values(["session", "set"])
    S.to_csv(OUT / "rest_spikes_sessions.csv", index=False)
    for st in ("engaged", "all", "tail"):
        print(f"\n=== REST SET: {st} ===  per animal x epoch, median over sessions: spikes/min (all / local / "
              "independent of the other group), co = fraction with an other-group spike within 1 s")
        for an, A in S[S.set == st].groupby("animal"):
            for ep in EP:
                B = A[A.epoch == ep]
                if B.empty:
                    continue
                cells = "  ".join(f"{c}: {B[f'{c}|rate'].median():.2f}/{B[f'{c}|local_rate'].median():.2f}/"
                                  f"{B[f'{c}|indep_rate'].median():.2f} co {B[f'{c}|cooccur'].median():.2f}"
                                  for c in ("orofacial", "limb"))
                lr = "  ".join(f"{c}: {B[f'{c}|rate'].median():.2f}" for c in SHOW[:4])
                sd = "/".join(f"{B[f'{c}|sd_eng'].median():.2f}" for c in ("orofacial", "limb"))
                print(f"{an} {ep:9s} n={len(B):2d} rest {B.rest_min.sum():6.1f} min | {cells} | {lr} | engaged-rest SD "
                      f"oro/limb {sd}%")
    print("\nPER AREA, ENGAGED rest: post / pre ratio of the spike rate (animal medians over sessions), PS92/93/94/95")
    E = S[S.set == "engaged"]
    for c in [f"{a}_{h}" for a in AREAS for h in ("left", "right")] + SHOW:
        out = []
        for ep in EP[1:]:
            vals = []
            for _an, A in E.groupby("animal"):
                p_, q_ = A[A.epoch == "pre"][f"{c}|rate"].median(), A[A.epoch == ep][f"{c}|rate"].median()
                vals.append(f"{q_ / p_:.1f}" if p_ > 0 else ("inf" if q_ > 0 else "-"))
            out.append(f"{ep} " + "/".join(vals))
        print(f"{c:18s} " + "   ".join(out))
    figure(S)
    (OUT / "README.txt").write_text(__doc__)


def figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"PS92": "tab:blue", "PS93": "tab:red", "PS94": "tab:green", "PS95": "tab:purple"}
    show = SHOW[:4]
    panels = [("engaged", "rate", "spikes / min, ENGAGED rest"), ("all", "rate", "spikes / min, ALL rest"),
              ("engaged", "indep_rate", "independent spikes / min, engaged")]
    fig, ax = plt.subplots(len(panels), len(show), figsize=(4 * len(show), 3.2 * len(panels)), sharex=True)
    for i, (st, key, lab) in enumerate(panels):
        for j, c in enumerate(show):
            for an, A in S[S.set == st].groupby("animal"):
                ax[i, j].plot(A.day, A[f"{c}|{key}"], "o", mfc="none", color=col.get(an, "k"), ms=5, label=an)
            ax[i, j].axvline(0, color="k", lw=0.6)
            ax[i, j].set_ylabel(lab, fontsize=8)
            if i == 0:
                ax[i, j].set_title(c.replace("_left", " ipsilesional").replace("_right", " contralesional"), fontsize=9)
    for j in range(len(show)):
        ax[-1, j].set_xlabel("days since stroke")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("cohort1 rest-period spikes (restdock05; half-width 50-500 ms; > 5 x the animal's pre-stroke engaged-rest "
                 "robust SD): orofacial = ALM + tjM1 + SSp-m + SSp-n, limb = SSp-ul + SSp-ll", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "rest_spikes_by_day.png", dpi=90)
    print("wrote", OUT / "rest_spikes_by_day.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    from wfield_local.writeguard import assert_writable
    assert_writable(OUT)
    if not a.summarise:
        from wfield_local.analysis_kit import curated_labels, fan_sessions  # noqa: F401
        labels = sorted(curated_labels())
        _, fail = fan_sessions(labels, extract, jobs=a.jobs, label="session")
        if fail:
            raise SystemExit(f"{len(fail)} session(s) FAILED, not summarising a partial set: {fail[:3]}")
    summarise()


if __name__ == "__main__":
    main()
