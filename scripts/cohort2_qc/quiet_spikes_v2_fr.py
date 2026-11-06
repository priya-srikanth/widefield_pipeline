"""Quiet-period cortical events AFTER regressing out face movement, on the v2 definitions (cohort1, 2026-11-01).

Priya 11-01, on fr_regressed_own/rest_spikes_pooled_epochs.png: "can title/caption be clear and we can add to the page?";
"the spikes/min face regressed seems most impt". That figure used the v1 frames (restdock05 rest, ~1.5 s per trial) and
the v1 prominence detector at 5 x SD, positive only; the page uses v2 (quiet_spikes_v2.py). This script redoes the face
regression on the v2 definitions so the two agree:

FRAMES: v2 QUIET (no DAQ contact lick / running bout within 1.5 s before or after; spout may be out; no swapped pair
within 0.5 s; `quiet_spikes_v2.busy_frames`), engaged part of the session; additionally frames with a full face-regressor
row only, for BOTH the raw and the residual count (identical frames).
FACE MODEL: each session's OWN FaceRhythm fit (QS_FR_BASIS=own, `quiet_spikes_fr.face_regressors`: 12 own factors,
absolute, + 20 own point-displacement PCs + their speeds, all 2-s high-passed, lags -0.3 .. +0.3 s), ridge fitted on
engaged quiet frames, OUT OF FOLD (10 contiguous blocks), applied to the 2-s-median-detrended trace.
EVENTS: `quiet_spikes_v2.detect` (height above the median of the preceding 1.0 .. 0.2 s, half-width 1.5 frames .. 0.5 s,
merged within 0.3 s), positive and negative, threshold k = 4 x the animal's August-pre robust SD -- raw counts use the
raw SD, residual counts the RESIDUAL SD (both from this run's engaged quiet frames).
MOVEMENT AT A RAW EVENT: face speed max within +/-0.3 s, as a percentile of the session's quiet face speed; > 90th =
movement-associated; chance = the same at 3000 random engaged quiet frames.
STATISTICS: per-animal table first, `mixed_models.lmm_table` epoch + time models (August pre, log10(rate + 0.2)), FDR.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_spikes_v2_fr.py [--jobs 6] [--summarise]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QS_FR_BASIS", "own")            # before the import: quiet_spikes_fr reads it at import (spawn too)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import DETREND_S, EP, columns, mad, session_traces  # noqa: E402
from quiet_spikes_fr import MOVE_PCTL, MOVE_WIN_S, N_FOLD, face_regressors, lagged, ridge_oof  # noqa: E402
from quiet_spikes_v2 import V2, busy_frames, detect, detrend  # noqa: E402
from wfield_local import config  # noqa: E402

K = 4.0
FRV2 = V2 / f"fr_regressed_{os.environ['QS_FR_BASIS']}"
SHOW = ["orofacial_left", "orofacial_right", "limb_left", "limb_right"]
GROUPS = ["orofacial", "limb"]


def extract(label):
    f = FRV2 / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    X = detrend(Xr, fs)
    busy, badw = busy_frames(label, ft, fs)
    w = int(round(DETREND_S * fs)) | 1
    R, speed = face_regressors(label, ft, w)
    XL = lagged(R, fs)
    okX = np.isfinite(XL).all(1)
    quiet = ~busy & ~badw & np.isfinite(X).all(0) & okX
    qe = quiet & engaged
    T = X.shape[1]
    pred, alphas = ridge_oof(XL, X.T, qe, np.arange(T) * N_FOLD // T)
    Res = X - np.nan_to_num(pred.T)
    cols = columns()
    r2 = 1 - ((X[:, qe] - pred.T[:, qe]) ** 2).sum(1) / ((X[:, qe] - X[:, qe].mean(1, keepdims=True)) ** 2).sum(1)
    ev = {}
    for kind, Y in (("raw", X), ("res", Res)):
        for sign, nm in ((1, "pos"), (-1, "neg")):
            L = []
            for i, c in enumerate(cols):
                if c == "cortex":
                    continue
                pk, h = detect(sign * Y[i], quiet, fs)
                L += [(i, int(p), float(v), int(engaged[p])) for p, v in zip(pk, h)]
            ev[f"{kind}_{nm}"] = np.array(L, float).reshape(-1, 4)
    sp_q = np.sort(speed[qe & np.isfinite(speed)])
    hw = int(round(MOVE_WIN_S * fs))

    def pctl_at(frames):
        out = np.full(len(frames), np.nan)
        for j, p in enumerate(frames):
            v = speed[max(0, p - hw):p + hw + 1]
            if np.isfinite(v).any() and len(sp_q):
                out[j] = 100.0 * np.searchsorted(sp_q, np.nanmax(v)) / len(sp_q)
        return out
    rr = np.flatnonzero(qe)
    null = pctl_at(np.random.default_rng(0).choice(rr, min(3000, len(rr)), replace=False)) if len(rr) else np.array([])
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **ev, move_raw_pos=pctl_at(ev["raw_pos"][:, 1].astype(int)),
                        move_raw_neg=pctl_at(ev["raw_neg"][:, 1].astype(int)), move_null=null, r2=r2,
                        sd_raw=mad(X[:, qe]) if qe.sum() > 100 else np.full(len(cols), np.nan),
                        sd_res=mad(Res[:, qe]) if qe.sum() > 100 else np.full(len(cols), np.nan),
                        fs=fs, cols=np.array(cols), n_quiet_eng=int(qe.sum()), face_frac=float(okX.mean()),
                        alphas=np.array(alphas))
    return label


def session_table():
    import pandas as pd
    from wfield_local import epochs
    D = {f.stem: dict(np.load(f)) for f in sorted((FRV2 / "sessions").glob("*.npz"))}
    cols = columns()
    pre = {k: d for k, d in D.items() if epochs.epoch_of(k) == "pre" and not config.early_pre(k)}
    sd = {(an, kind): np.nanmedian([d[f"sd_{kind}"] for k, d in pre.items() if k.startswith(an)], 0)
          for an in sorted({k[:4] for k in D}) for kind in ("raw", "res")}
    rows = []
    for k, d in D.items():
        ep = epochs.epoch_of(k)
        if ep is None or (ep == "pre" and config.early_pre(k)):
            continue
        an, fs = k[:4], float(d["fs"])
        mins = int(d["n_quiet_eng"]) / fs / 60
        if mins < 0.25:
            continue
        r = dict(label=k, animal=an, epoch=ep, day=epochs.days_since_stroke(k), quiet_min=mins,
                 face_frac=float(d["face_frac"]), move_null=float(np.nanmean(d["move_null"] > MOVE_PCTL)))
        for i, c in enumerate(cols[:-1]):
            mv = []
            for kind in ("raw", "res"):
                n = {}
                for pol in ("pos", "neg"):
                    a = d[f"{kind}_{pol}"]
                    m = (a[:, 0] == i) & (a[:, 3] == 1) & (a[:, 2] > K * sd[an, kind][i])
                    n[pol] = int(m.sum())
                    if kind == "raw":
                        mv.append(d[f"move_raw_{pol}"][m])
                r[f"{c}|{kind}_pos"], r[f"{c}|{kind}_neg"] = n["pos"] / mins, n["neg"] / mins
                r[f"{c}|{kind}_both"] = (n["pos"] + n["neg"]) / mins
            mv = np.concatenate(mv)
            r[f"{c}|move_frac"] = float(np.mean(mv[np.isfinite(mv)] > MOVE_PCTL)) if np.isfinite(mv).any() else np.nan
        for i, c in enumerate(cols):
            r[f"{c}|r2"] = float(d["r2"][i])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


MEASURES = {"res_both": "events / min, face regressed\n(positive + negative)",
            "raw_both": "events / min, raw\n(positive + negative)",
            "res_pos": "face regressed, positive", "res_neg": "face regressed, negative",
            "move_frac": "fraction of raw events\nwith face movement", "r2": "face model\ncross-validated R2"}


def stats(S):
    import pandas as pd
    from wfield_local import mixed_models as mm
    L = []
    for c in SHOW + GROUPS:
        for m in MEASURES:
            col = f"{c}|{m}"
            y = np.log10(S[col] + 0.2) if m[:3] in ("res", "raw") else S[col]
            L.append(S[["label"]].assign(column=c, measure=m, y=y.to_numpy()))
    L = mm.add_time_covariates(pd.concat(L), august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "column"], models=("epoch", "time"), print_animals=True, fdr_by=("measure",))
    T.to_csv(FRV2 / "lmm_v2_fr.csv", index=False)
    return T


def figure(S, T, rows, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    fig, ax = plt.subplots(len(rows), len(SHOW), figsize=(3.5 * len(SHOW), 2.7 * len(rows) + 0.8), sharey="row",
                           squeeze=False)
    rng = np.random.default_rng(0)
    for i, m in enumerate(rows):
        for j, c in enumerate(SHOW):
            a, col = ax[i, j], f"{c}|{m}"
            for x, ep in enumerate(EP):
                B = S[S.epoch == ep]
                for an, A in B.groupby("animal"):
                    a.scatter(x + rng.uniform(-0.18, 0.18, len(A)), A[col], marker=ANIMAL_MARKER[an],
                              facecolors="none", edgecolors=EPOCH_LINE[ep], s=18, lw=0.8, alpha=0.8)
                ci = boot_ci({an: list(A[col].dropna()) for an, A in B.groupby("animal") if A[col].notna().any()}, rng)
                if ci is not None:
                    a.errorbar(x + 0.32, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="o",
                               color=EPOCH_LINE[ep], mfc=EPOCH_LINE[ep], ms=6, capsize=3, lw=1.6, zorder=5)
                if m == "move_frac":
                    a.plot([x - 0.3, x + 0.3], [B.move_null.mean()] * 2, color="0.5", ls="--", lw=1)
                t = T[(T.model == "epoch") & (T.measure == m) & (T.column == c) & (T.term == f"C(epoch)[T.{ep}]")]
                if x > 0 and len(t) and t.q.iloc[0] < 0.05:
                    q = t.q.iloc[0]
                    sig_marks.star_top(a, x, "***" if q < 0.001 else "**" if q < 0.01 else "*", color=EPOCH_LINE[ep])
            a.set_xticks(range(len(EP)), EP, fontsize=8)
            if j == 0:
                a.set_ylabel(MEASURES[m], fontsize=8)
            if i == 0:
                a.set_title(c.replace("_left", " ipsilesional").replace("_right", " contralesional"), fontsize=9)
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    ax[0, 0].legend(handles=handles, fontsize=7, loc="upper left")
    sig_marks.reserve_headroom(fig)
    fig.suptitle(title, fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if len(rows) < 3 else 0.95))
    fig.savefig(FRV2 / out, dpi=110)
    print("wrote", FRV2 / out)


def summarise():
    import pandas as pd
    pd.set_option("display.width", 240)
    S = session_table()
    S.to_csv(FRV2 / "quiet_events_fr_sessions.csv", index=False)
    print(f"{len(S)} sessions; face data cover median {S.face_frac.median():.2f} of frames")
    for c in ("orofacial", "limb"):
        print(f"\n{c}: per-animal epoch means (raw_both, res_both, move_frac, r2)")
        print(S.groupby(["animal", "epoch"])[[f"{c}|raw_both", f"{c}|res_both", f"{c}|move_frac", f"{c}|r2"]]
              .mean().round(2).to_string())
    T = stats(S)
    show = T[(T.model == "epoch") & (T.term != "pre_level") & T.column.isin(SHOW + GROUPS)]
    print("\nLMM epoch model vs August pre (rates log10(rate + 0.2)):")
    print(show[["measure", "column", "term", "coef", "se", "p", "q", "re", "n"]].round(4).to_string(index=False))
    cap = ("events = brief deflections in the 2-s-median-detrended dF/F trace whose height above the preceding "
           "1.0-0.2 s exceeds 4 x the animal's pre-stroke robust SD (1.4826 x MAD), positive or negative; quiet = no "
           "lick / running within 1.5 s before or after (spout may be out), engaged part of the session. Open markers "
           "= sessions (shape = animal, colour = epoch); filled = pooled mean, nested animal -> session bootstrap 95% "
           "CI; stars = mixed model, epoch vs August pre, FDR.")
    figure(S, T, ["res_both"], "quiet_events_face_regressed.png",
           "Quiet-period events per minute after regressing out face movement (each session's own FaceRhythm "
           "factors + face-point PCs + speeds, lags +-0.3 s, out of fold).\n" + cap)
    figure(S, T, ["raw_both", "res_both", "move_frac", "r2"], "quiet_events_face_regressed_detail.png",
           "Raw vs face-regressed quiet events, movement at raw events (dashed = chance) and face-model R2. " + cap)
    (FRV2 / "README.txt").write_text(__doc__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.analysis_kit import curated_labels
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = a.only or [k for k in sorted(curated_labels()) if not config.early_pre(k)]
        res, fail = fan_sessions(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        if len(fail) > 3:
            raise SystemExit("too many failures")
    summarise()


if __name__ == "__main__":
    main()
