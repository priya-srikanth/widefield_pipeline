"""SIDE-ONLY LICK GUARD, BLIND VALIDATION (Priya 11-21: "shape rule, validated"). The PS95 side-week check
(`scripts/pose3d_side_week_check.py`, qc_phaseA/ps95_side_week) read a rule off 41 judged side-only licks: keep a
side-only lick only if its side peak is <= 4 mm from the mouth AND the lick lasts >= 20 ms (5/5 real kept, 1/36 false
kept). Those thresholds were chosen on the same 41, so this script tests them on a FRESH sample, judged blind.
RESULT (11-21): the rule FAILED (12/25 real kept, 6/37 false kept) and was NOT adopted; the two-view fallback (cam2 AND
cam3) kept 19/25 real, 6/37 false, no paws / drops. README in the output folder; DECISIONS.md 2026-11-21.

READ ONLY on every existing output and on the raw videos; writes only to
``<pose3d_4cam_3net_20261121>/qc_phaseA/side_only_guard/``.

    python -m scripts.side_only_guard_check events            every side-only lick of the 126 sessions + features
    python -m scripts.side_only_guard_check pick              the fresh sample (stratified + near-threshold), blind ids
    python -m scripts.side_only_guard_check strips [--jobs 6] BLIND frame strips (blind id only; no rule features shown)
    python -m scripts.side_only_guard_check score             join verdicts_blind.csv (written by hand FIRST) to the
                                                              rule; sensitivity / false-keep; review sheet + answer key
    python -m scripts.side_only_guard_check sessions          per-session composition + what each candidate rule keeps

Sample (`pick`, seed 1): the 41 already-reviewed licks excluded (same ``ex_id``). RANDOM = up to 6 per animal x phase
(pre / post, `config.session_phase`; excluded dates dropped), 8 strata. NEAR = 15 drawn from the licks within ~1 mm /
8 ms of the keep region's edge, on either side: side peak <= 5 mm and fused span >= 12 ms, minus the deep interior
(peak < 3 mm and span > 28 ms). Strata are not shown on the strips.

Blindness: strips (`strips`) show a fixed window (+-28 ms around the lick's center, every frame) so the strip width does
not encode the span; the title is the blind id only; the traces are raw tongue likelihood per camera (no distances, no
shaded span). The verdict file ``verdicts_blind.csv`` (blind_id, verdict, note) is written BEFORE `score` joins it to
the features. Residual leak: the frames themselves show how long the label is present and where it sits; judging
needs that.

Features are the side-week ones: side peak = the largest ``peak_dist_mm`` of the side events inside the lick span
(px / `pose3d_side.px_per_mm`, a size scale at the mouth's depth), span = fused off - on (ms).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts import pose3d_side_week_check as W  # noqa: E402

SEED = 1
N_PER_STRATUM = 6
N_NEAR = 15
WIN_MS = 28.0
NEAR_MM, LONG_MS = W.NEAR_MM, W.LONG_MS            # the rule under test (4.0 mm, 20 ms)
VERDICTS = ("real", "paw", "grooming", "water_drop", "other")


def dirs() -> dict:
    D = W.dirs()
    out = D["4cam"] / "qc_phaseA" / "side_only_guard"
    assert "MICROSCOPE" in str(out) and "Priya" in str(out), out
    return {**D, "out": out, "strips": out / "strips"}


def _session_rows(f: str) -> pd.DataFrame:
    D = W.dirs()
    f = Path(f)
    sid = f.name[:13]
    U = pd.read_csv(f)
    E = pd.read_csv(D["licks"] / f"{sid}_side_events.csv")
    U["sid"], U["animal"], U["date"] = sid, sid[:4], sid[5:]
    U["side_only"] = W._side_only(U)
    U = U[U.side_only].copy()
    if not len(U):
        return U
    U = pd.concat([U, W._match_side(U, E)], axis=1)
    U["dur_ms"] = U.off_ms - U.on_ms
    return U


def events() -> None:
    from wfield_local import config, parallel
    D = dirs()
    D["out"].mkdir(parents=True, exist_ok=True)
    files = sorted(str(f) for f in D["licks"].glob("PS*_2026*_fused.csv"))
    res, fail = parallel.fan_out(files, _session_rows, jobs=6, label="session")
    assert not fail, fail
    A = pd.concat([r for _f, r in sorted(res, key=lambda x: x[0])], ignore_index=True)
    A["phase"] = [config.session_phase(a, str(d)) for a, d in zip(A.animal, A.date)]
    A["ex_id"] = [f"{r.sid}_t{int(r.trial_id)}_{int(r.on_ms)}" for r in A.itertuples()]
    A.to_csv(D["out"] / "side_only_all.csv.gz", index=False, float_format="%.4g")
    print(len(files), "sessions;", len(A), "side-only licks")
    print(A.groupby(["animal", "phase"]).size().unstack())


def pick() -> None:
    D = dirs()
    A = pd.read_csv(D["out"] / "side_only_all.csv.gz")
    seen = set(pd.read_csv(W.dirs()["out"] / "examples_review.csv").ex_id)
    A = A[~A.ex_id.isin(seen) & A.phase.isin(["pre", "post"]) & A.side_peak_dist_mm.notna()].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    near_pool = (A.side_peak_dist_mm <= 5.0) & (A.dur_ms >= 12.0) & ~((A.side_peak_dist_mm < 3.0) & (A.dur_ms > 28.0))
    # NEAR first, spread over animals (round robin), so RANDOM cannot take them
    near_idx = []
    pools = {a: list(rng.permutation(np.flatnonzero(near_pool & (A.animal == a)))) for a in sorted(A.animal.unique())}
    while len(near_idx) < N_NEAR and any(pools.values()):
        for a in sorted(pools):
            if pools[a] and len(near_idx) < N_NEAR:
                near_idx.append(pools[a].pop())
    rest = A.drop(index=near_idx)
    rand_idx = []
    for (_a, _ph), g in rest.groupby(["animal", "phase"], sort=True):
        k = rng.choice(len(g), size=min(N_PER_STRATUM, len(g)), replace=False)
        rand_idx += list(g.index[np.sort(k)])
    P = pd.concat([A.loc[near_idx].assign(stratum="near"), A.loc[rand_idx].assign(stratum="random")])
    P = P.iloc[rng.permutation(len(P))].reset_index(drop=True)
    P["blind_id"] = [f"B{i + 1:02d}" for i in range(len(P))]
    P.to_csv(D["out"] / "sample_key.csv", index=False, float_format="%.4g")
    print(len(P), "picked;", P.groupby(["animal", "phase", "stratum"]).size().to_string())


def strip_worker(sid: str) -> list:
    """Blind strips for the picked licks of one session (one `fuse_session`): cam2 / cam3 / cam1 / cam4 rows, every frame
    in +-WIN_MS of the lick's center, tracked tongue (cleaned = filled red, raw label = open red), jaw, mouth zero."""
    W._env()
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d import PROD
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d_side as PS
    D = dirs()
    rv = W._rv()
    an, date = sid.split("_")
    P = pd.read_csv(D["out"] / "sample_key.csv")
    P = P[P.sid == sid]
    c1 = oi.session_windows(rv, PROD["cam1"], an, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], an, date, "cam4")
    Z = PS.fuse_session(rv, an, date, side_bundle=W.SIDE_BUNDLE, c1=c1, c4=c4)
    F, order, pts, side = Z["F"], Z["order"], Z["pts"], Z["side"]
    cams = {c.get_name(): c for c in Z["g"].cameras}
    raw = {"cam1": c1, "cam4": c4}
    srcmap = {}
    for c in ("cam1", "cam4"):
        s = pd.Series(np.arange(len(raw[c][0])), index=raw[c][0].src_frame.to_numpy())
        srcmap[c] = s[~s.index.duplicated()]
    src = {"cam1": F.src1.to_numpy(), "cam4": F.src4.to_numpy(),
           **{c: np.where(F[f"i_{c}"].to_numpy() >= 0,
                          side[c][0].src_frame.to_numpy()[np.clip(F[f"i_{c}"].to_numpy(), 0, None)], -1) for c in W.SIDE}}

    def raw_xy(c: str, part: str, i: int):
        if c in W.SIDE:
            k, fl = int(F[f"i_{c}"].iat[i]), side[c][1]
        else:
            s = src[c][i]
            k, fl = (int(srcmap[c][s]) if (s >= 0 and s in srcmap[c].index) else -1), raw[c][1]
        if k < 0:
            return np.array([np.nan, np.nan]), np.nan
        return np.array([fl[f"{part}_x"].iat[k], fl[f"{part}_y"].iat[k]], float), float(fl[f"{part}_likelihood"].iat[k])

    mouth = {c: PS.project(cams[c], np.zeros((1, 3)))[0] for c in order}
    readers = {c: cv2.VideoCapture(str(oi.videos_for(rv, an, date, c)[0])) for c in order}
    show = ("cam2", "cam3", "cam1", "cam4")
    done = []
    D["strips"].mkdir(parents=True, exist_ok=True)
    tt_all, tid_all = F.t_ms.to_numpy(float), F.trial_id.to_numpy()
    for r in P.itertuples():
        rows_t = np.flatnonzero(tid_all == r.trial_id)
        tt = tt_all[rows_t]
        mid = (float(r.on_ms) + float(r.off_ms)) / 2.0
        sel = rows_t[(tt >= mid - WIN_MS) & (tt <= mid + WIN_MS)]
        win = rows_t[(tt >= mid - 200) & (tt <= mid + 200)]
        nf = max(len(sel), 1)
        fig = plt.figure(figsize=(1.25 * nf + 0.4, 1.25 * 4 + 1.6))
        gs = fig.add_gridspec(5, nf, height_ratios=[1, 1, 1, 1, 0.9], hspace=0.12, wspace=0.03)
        for jc, c in enumerate(show):
            j = order.index(c)
            cs = 240 if c in W.SIDE else 220
            cx, cy = mouth[c]
            for jj, i in enumerate(sel):
                ax = fig.add_subplot(gs[jc, jj])
                ax.set_xticks([])
                ax.set_yticks([])
                vf, im = src[c][i], None
                if vf >= 0:
                    readers[c].set(cv2.CAP_PROP_POS_FRAMES, int(vf))
                    _ok, im = readers[c].read()
                if im is None:
                    ax.text(0.5, 0.5, "no frame", ha="center", fontsize=6, transform=ax.transAxes)
                    continue
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                x0 = int(np.clip(cx - cs / 2, 0, max(im.shape[1] - cs, 0)))
                y0 = int(np.clip(cy - cs / 2, 0, max(im.shape[0] - cs, 0)))
                ax.imshow(im[y0:y0 + cs, x0:x0 + cs])
                q, ql = raw_xy(c, "tongue", i)
                if np.isfinite(q).all() and ql >= 0.1:
                    ax.plot(q[0] - x0, q[1] - y0, "o", ms=5, mfc="none", mec="#ff3c3c", mew=0.8)
                qc = pts["tongue"][j, i]
                if np.isfinite(qc).all():
                    ax.plot(qc[0] - x0, qc[1] - y0, "o", ms=3.5, color="#ff3c3c")
                qj, jl = raw_xy(c, "jaw", i)
                if np.isfinite(qj).all() and jl >= 0.6:
                    ax.plot(qj[0] - x0, qj[1] - y0, "^", ms=4, color="#ffd400")
                ax.plot(cx - x0, cy - y0, "+", color="cyan", ms=6, mew=1)
                if jc == 0:
                    ax.set_title(f"f{jj + 1}", fontsize=6)
                if jj == 0:
                    ax.set_ylabel(c, fontsize=7)
        t = (tt_all[win] - mid) / 4.0
        ax2 = fig.add_subplot(gs[4, :])
        colc = {"cam1": "#1b9e77", "cam4": "#7570b3", "cam2": "#d95f02", "cam3": "#e7298a"}
        for c in show:
            ax2.plot(t, [raw_xy(c, "tongue", i)[1] for i in win], lw=0.8, color=colc[c], label=c)
        ax2.axvline(0, color="k", lw=0.5, ls=":")
        ax2.axhline(0.4, color="gray", lw=0.6, ls=":")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_ylabel("raw tongue lk", fontsize=7)
        ax2.set_xlabel("frames from the strip center (4 ms each; strip = the frames within +-7)", fontsize=7)
        ax2.legend(fontsize=6, ncol=4, frameon=False, loc="upper left")
        fig.suptitle(f"{r.blind_id}   red dot = cleaned tongue (used for licks); open red = raw tongue label (lk >= 0.1);"
                     f" yellow = jaw; cyan + = mouth zero", fontsize=7.5)
        fig.savefig(D["strips"] / f"{r.blind_id}.png", dpi=95, bbox_inches="tight")
        plt.close(fig)
        done.append(r.blind_id)
    for cap in readers.values():
        cap.release()
    return done


def strips(jobs: int) -> None:
    from wfield_local import parallel
    D = dirs()
    P = pd.read_csv(D["out"] / "sample_key.csv")
    res, fail = parallel.fan_out(sorted(P.sid.unique()), strip_worker, jobs=jobs, label="session")
    if fail:
        print("FAILED:", fail)
    print("strips:", sum(len(x) for _s, x in res), "of", len(P))


def rule_keep(peak_mm, dur_ms) -> np.ndarray:
    return (np.asarray(peak_mm, float) <= NEAR_MM) & (np.asarray(dur_ms, float) >= LONG_MS)


def score() -> None:
    D = dirs()
    P = pd.read_csv(D["out"] / "sample_key.csv")
    V = pd.read_csv(D["out"] / "verdicts_blind.csv")
    assert set(V.verdict) <= set(VERDICTS) | {"unjudged"}, set(V.verdict) - set(VERDICTS)
    R = P.merge(V, on="blind_id", how="inner", validate="1:1")
    R["rule_keep"] = rule_keep(R.side_peak_dist_mm, R.dur_ms)
    R["real"] = R.verdict == "real"
    # "other" with a note starting "fragment": a real tongue that is the onset / retraction tail of a lick another
    # camera's event already counts -> a double count, scored FALSE (and, as a check, also dropped from the scoring)
    R["fragment"] = (R.verdict == "other") & R.note.astype(str).str.startswith("fragment")
    cols = ["blind_id", "verdict", "note", "fragment", "rule_keep", "stratum", "ex_id", "sid", "animal", "phase", "trial_id",
            "position", "on_ms", "off_ms", "dur_ms", "cams", "side_cam", "side_n_frames", "side_peak_dist_mm",
            "side_static", "side_far"]
    R = R.sort_values("blind_id")[cols]
    R.to_csv(D["out"] / "review.csv", index=False, float_format="%.4g")
    R[["blind_id", "ex_id", "verdict", "rule_keep", "stratum", "side_peak_dist_mm", "dur_ms", "cams", "position"]].to_csv(
        D["out"] / "blind_answer_key.csv", index=False, float_format="%.4g")
    pd.DataFrame({"blind_id": sorted(P.blind_id), "verdict (real/paw/grooming/water_drop/other)": "", "note": ""}).to_csv(
        D["out"] / "blind_review_sheet.csv", index=False)
    lines = []
    J = R[R.verdict != "unjudged"].assign(real=lambda d: d.verdict == "real")
    for name, g in (("ALL", J), ("random", J[J.stratum == "random"]), ("near", J[J.stratum == "near"]),
                    ("ALL_no_fragments", J[~J.fragment]), *((f"{a}_{ph}", J[(J.animal == a) & (J.phase == ph)])
                                                         for a in sorted(J.animal.unique()) for ph in ("pre", "post"))):
        nr, nf = int(g.real.sum()), int((~g.real).sum())
        kr, kf = int((g.real & g.rule_keep).sum()), int((~g.real & g.rule_keep).sum())
        lines.append({"subset": name, "n": len(g), "real": nr, "real_kept": kr,
                      "sensitivity": kr / nr if nr else np.nan, "false": nf, "false_kept": kf,
                      "false_keep_rate": kf / nf if nf else np.nan,
                      "kept": int(g.rule_keep.sum()), "precision": kr / max(int(g.rule_keep.sum()), 1)})
    S = pd.DataFrame(lines)
    S.to_csv(D["out"] / "validation_summary.csv", index=False, float_format="%.4g")
    print(S.to_string())
    print(pd.crosstab(J.verdict, J.rule_keep, margins=True))
    print(pd.crosstab([J.animal, J.phase], J.verdict))
    print(J[J.real][["blind_id", "stratum", "rule_keep", "side_peak_dist_mm", "dur_ms", "cams", "position"]].to_string())
    print(J[~J.real & J.rule_keep][["blind_id", "verdict", "stratum", "side_peak_dist_mm", "dur_ms", "note"]].to_string())


def sessions() -> None:
    """Per session, on the EXISTING 11-21 fusion (nothing re-run): kept licks and their composition (all cameras = cam1 +
    cam4 + a side view / cam1-only / cam4-only / side-only / other), and how many side-only licks each candidate rule
    would keep: the shape rule (side peak <= 4 mm AND span >= 20 ms) and the two-view rule (cam2 AND cam3 both saw it)."""
    from wfield_local import config
    D = dirs()
    V = pd.read_csv(D["licks"] / "views.csv")
    V["kept"] = V.n - V.guarded
    sets = V.cams.astype(str).str.split("+").map(set)
    V["kind"] = np.select([sets.map(lambda s: s <= set(W.SIDE)), V.cams == "cam1", V.cams == "cam4",
                           sets.map(lambda s: {"cam1", "cam4"} <= s and bool(s & set(W.SIDE)))],
                          ["side_only", "cam1_only", "cam4_only", "all_cams"], "other_multi")
    T = V.pivot_table(index="session", columns="kind", values="kept", aggfunc="sum", fill_value=0)
    T = T.reindex(columns=["all_cams", "other_multi", "cam1_only", "cam4_only", "side_only"], fill_value=0)
    T.insert(0, "n_kept", T.sum(1))
    A = pd.read_csv(D["out"] / "side_only_all.csv.gz")
    g = A.assign(shape=rule_keep(A.side_peak_dist_mm, A.dur_ms), two=A.cams == "cam2+cam3").groupby("sid")
    T["side_only_shape_kept"] = g["shape"].sum().reindex(T.index, fill_value=0)
    T["side_only_two_view_kept"] = g["two"].sum().reindex(T.index, fill_value=0)
    T = T.reset_index()
    T.insert(1, "animal", T.session.str[:4])
    T.insert(2, "phase", [config.session_phase(s[:4], s[5:]) for s in T.session])
    for c in ("side_only", "side_only_shape_kept", "side_only_two_view_kept"):
        T[f"{c}_pct"] = 100 * T[c] / T.n_kept
    T.to_csv(D["out"] / "sessions_rule_impact.csv", index=False, float_format="%.4g")
    num = ["n_kept", "all_cams", "other_multi", "cam1_only", "cam4_only", "side_only", "side_only_shape_kept",
           "side_only_two_view_kept"]
    print(T[num].sum().to_string())
    print(T.groupby(["animal", "phase"])[num].sum().to_string())
    print(T.sort_values("side_only_pct").tail(8)[["session", "n_kept", "cam4_only", "side_only", "side_only_pct",
                                                  "side_only_shape_kept", "side_only_two_view_kept"]].round(2).to_string())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["events", "pick", "strips", "score", "sessions"])
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    W._env()
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    {"events": events, "pick": pick, "strips": lambda: strips(a.jobs), "score": score, "sessions": sessions}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
