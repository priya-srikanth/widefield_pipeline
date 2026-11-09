"""Why does the raw rest level FALL after the stroke? (handoff 11-09 ~12:15 open item d; Priya 11-09 "go ahead with all")

B32: the raw RESTnl level (vs the session mean) falls -.48 / -.53 / -.76 in contralesional orofacial cortex. The signal
has no absolute level -- the slow drift is projected out per session INCLUDING the constant
(`state_space_rest_baseline.py`), so every frame is relative to the session's (slow, >= 600 s) mean. The mean is the
time-weighted sum of every behavioral state, so with frame classes k (fraction f_k of frames, mean level m_k):
    sum_k f_k m_k = 0   =>   m_rest = - sum_{k != rest} f_k (m_k - m_rest)
i.e. the rest level reads LOW when the session spends more time in states that sit above rest (time budget) or when
those states sit further above rest (bigger task responses). This script measures every term per session.

CLASSES (frames, disjoint, in this precedence), on the same drift-removed SVT as the RESTnl level:
  rest     the RESTnl frames themselves (`restnl_mask`: lick-free rest, engaged, position-labelled; every position)
  trial    cue to cue + 3.5 s (the response window) of an ENGAGED trial: the task response itself
  lick     within [0.5 s before, 2.0 s after] any DAQ or cam4 LP lick (RESTnl's own lick buffer), outside the trial
           window (ITI / ENL licking; ~90 % of frames sit in the lick buffer, smoke test PS92_0807)
  quit     after the cue of the first trial of the terminal non-engaged run (`rest_by_position._engaged_trials`)
  pretask  before the first cue
  other    the rest of the session (engaged, no lick nearby, not a RESTnl frame: running, undocked, short or ambiguously
           labelled rest, inter-trial time near the cue)
Per class: the fraction of frames and the MEAN level; for rest also the RESTnl level itself (per-position median, rule
wr) so the median-vs-mean gap is visible. Region means (orofacial = tjM1 + SSp-m + SSp-n, contra / ipsi; whole brain
= the analysis mask) in the cohort SVD basis, units of the animal's pre-template pixel RMS (as B32 / Figs 10.21-10.26).
Change vs own pre (`boot_delta`, animals -> sessions, 2,000; * = CI excludes 0; per-animal values printed first).
Counterfactual split of the rest change: TIME BUDGET = the epoch's fractions with the animal's pre class gaps; GAPS =
pre fractions with the epoch's gaps (the remainder = interaction).
Caveat: per-pixel dF/F is relative to that pixel's own session mean, so no anatomical contrast escapes the reference;
this explains the reference, it cannot recover an absolute level (hand-set LED power, B32).

    python scripts/state_space_rest_decomp.py [--jobs 2]          # per-session classes (cached), then the table + figure
Outputs: <state_space_recovery>/rest_decomp_svd/<label>.npz, sessions.csv, changes.csv; page_figs/ss_rest_decomp_svd.png.
EXPLORATORY.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o2_state_space_lnmf as L  # noqa: E402
import state_space_sim as S  # noqa: E402

OUT = S.ROOT / "rest_decomp_svd"
BUNDLE = S.SVD.parents[1]
CLASSES = ["rest", "trial", "lick", "quit", "pretask", "other"]
TRIAL_S = 3.5                     # the task's response window (gui_config timing.response_window, 3500 ms)
EPS = ["acute", "subacute", "chronic"]


def _worker(label):
    """One session -> per-class fractions + mean levels in cohort coordinates (module-level for spawn)."""
    L._env(L.CACHE)
    import pandas as pd

    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import restnl_mask as rn
    meta = json.loads((BUNDLE / "bundle.json").read_text())
    src = Path(meta["lnmf_scratch"]) / "inputs" / f"{label}.pkl"
    if not src.exists():
        src = Path(meta["enc_scratch"]) / "inputs" / f"{label}.pkl"
    x = pickle.loads(src.read_bytes())
    V, ft = np.asarray(x["imaging"][0], np.float64), np.asarray(x["imaging"][1], float)
    del x
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    V -= Q @ (Q.T @ V)                                                  # (T, K) drift-removed, as the extraction
    T = V.shape[0]
    s = rn.session_of(label)
    try:
        mask, parts, info = rn.lickfree_mask(s)
    except FileNotFoundError as ex:
        return {"label": label, "ok": False, "reason": str(ex)}
    per = rn.frames_by_position(s, T, mask, parts)
    pm, nfr = rn.position_medians(V.T, per)
    b, ok, _used = rn.combine(pm, nfr, "wr")
    cs, codes, ts, fs_samp, sync, eng = parts
    fs = ak.daq_rate(s)
    n = mask.shape[0]
    f_of = np.clip(np.asarray(fs_samp)[:T], 0, n - 1)
    daq_on = rn.daq_lick_onsets(s, fs)
    cam4 = pd.read_csv(rn.licks_all_path(label), usecols=["daq_on_s"]).daq_on_s.to_numpy(float)
    cam4 = np.round(cam4[np.isfinite(cam4)] * fs).astype(np.int64)
    nb, na = int(rn.BUF_S[0] * fs), int(rn.BUF_S[1] * fs)
    lick_s = rn.widen(np.concatenate([np.asarray(daq_on, np.int64), cam4]), n, nb, na)
    eng = np.asarray(eng, bool)
    quit0 = n
    if len(eng) and not eng[-1]:
        last = np.flatnonzero(eng)
        k = (last[-1] + 1) if last.size else 0
        quit0 = int(cs[k]) if k < len(cs) else n
    cls = np.full(T, "other", object)
    cls[f_of < (cs[0] if len(cs) else 0)] = "pretask"
    cls[f_of >= quit0] = "quit"
    cls[lick_s[f_of]] = "lick"
    tr_s = np.zeros(n, bool)
    for c0, e in zip(np.asarray(cs, np.int64), eng):
        if e:
            tr_s[max(c0, 0):min(c0 + int(TRIAL_S * fs), n)] = True
    cls[tr_s[f_of]] = "trial"
    rest_fr = np.concatenate([np.asarray(v, np.int64) for v in per.values()]) if per else np.zeros(0, np.int64)
    rest_fr = rest_fr[rest_fr < T]
    cls[rest_fr] = "rest"
    msk = np.load(BUNDLE / "mask.npy").ravel().astype(bool)
    U = np.load(Path(meta["uatlas_root"]) / meta["uatlas"][label])
    u = np.nan_to_num(U.reshape(-1, U.shape[2])).astype(np.float64)[msk]
    Uc = np.load(S.SVD.parent / "basis" / "U_cohort.npy").astype(np.float64)
    to_c = lambda v: Uc.T @ (u @ np.nan_to_num(v))  # noqa: E731
    out = {"label": label, "ok": bool(ok), "c_restnl": to_c(b) if ok else np.full(Uc.shape[1], np.nan),
           "c_all": to_c(V.mean(0)), "info": json.dumps(info)}
    for k in CLASSES:
        m = cls == k
        out[f"f_{k}"] = float(m.mean())
        out[f"c_{k}"] = to_c(V[m].mean(0)) if m.any() else np.full(Uc.shape[1], np.nan)
    return out


def collect(jobs):
    from wfield_local import analysis_kit as kit
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [x for x in (BUNDLE / "labels.txt").read_text().split() if x]
    todo = [lab for lab in labels if not (OUT / f"{lab}.npz").exists()]
    print(f"{len(todo)} sessions to do", flush=True)
    if not todo:
        return
    if jobs > 1 and len(todo) > 1:
        got, fail = kit.fan_sessions(todo, _worker, jobs=jobs, key=lambda x: x)
    else:                         # never in THIS process: the profile's write fence would block the writes below
        from concurrent.futures import ProcessPoolExecutor
        got, fail = [], []
        with ProcessPoolExecutor(1) as ex:
            for lab in todo:
                try:
                    got.append((lab, ex.submit(_worker, lab).result()))
                except Exception as e:                                 # noqa: BLE001
                    fail.append((lab, e))
    for _i, r in got:
        np.savez(OUT / f"{r['label']}.npz", **{k: v for k, v in r.items() if k != "label"})
    print(f"{len(got)} done, {len(fail)} failed {[(i, str(e)[:80]) for i, e in fail]}", flush=True)


def analyze():
    if S.BASIS != "svd" or S.ALIGN != "lick":
        sys.exit("state_space_rest_decomp: analyze under WF_SS_BASIS=svd WF_SS_ALIGN=lick (the B32 units)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    import state_space_chronic_maps as C
    import state_space_recovery_trials as R
    import state_space_shift_behavior as SB
    from wfield_local import analysis_kit as kit
    from wfield_local import area_basis, beta_maps, sig_marks
    from wfield_local.figure_layout import save_png_svg
    from wfield_local.orofacial_cohort import EPOCH_LINE
    Uc = np.load(C.BASIS_DIR / "U_cohort.npy").astype(np.float64)
    atlas = np.load(BUNDLE / "atlas.npy")
    mask = np.load(C.BASIS_DIR / "mask.npy").reshape(atlas.shape)
    flat = np.flatnonzero(mask.ravel())
    split = area_basis.basis_masks(area_basis.roi_masks(atlas.ravel()[flat], beta_maps.stat_mask().ravel()[flat],
                                                        dict(S._atlas_names()), flat, mask.shape))["split"]
    W = {}
    for side, hemi in SB.SIDES.items():
        px = np.logical_or.reduce([np.asarray(split[f"{a}_{hemi}"], bool) for a in SB.REGIONS["orofacial"]
                                   if f"{a}_{hemi}" in split])
        W[f"orofacial {side}"] = Uc[px].mean(0)
    W["whole brain"] = Uc.mean(0)
    data, _ = S.load_sessions()
    rows = []
    for an, sess in sorted(data.items()):
        Tpre = R.template([q for q in sess if q[1] == "pre"])
        ok = np.isfinite(Tpre).all(axis=(1, 2))
        B0 = Tpre[ok] - S.flat(Tpre[ok]).mean(0)
        unit = float(np.sqrt((B0 ** 2).sum() / B0.size * B0.shape[-1] / Uc.shape[0]))
        for lab, ep, day, _Y, _pos in sess:
            f = OUT / f"{lab}.npz"
            if ep not in ("pre", *EPS) or not f.exists():
                continue
            z = np.load(f)
            if not bool(z["ok"]):
                continue
            r = {"animal": an, "label": lab, "epoch": ep, "day": day,
                 **{f"f_{k}": float(z[f"f_{k}"]) for k in CLASSES}}
            for reg, w in W.items():
                r[f"{reg} | RESTnl"] = float(z["c_restnl"] @ w) / unit
                r[f"{reg} | all"] = float(z["c_all"] @ w) / unit
                for k in CLASSES:
                    r[f"{reg} | m_{k}"] = float(z[f"c_{k}"] @ w) / unit if r[f"f_{k}"] > 0 else 0.0
                for k in CLASSES[1:]:
                    r[f"{reg} | gap_{k}"] = r[f"{reg} | m_{k}"] - r[f"{reg} | m_rest"] if r[f"f_{k}"] > 0 else 0.0
                    r[f"{reg} | contrib_{k}"] = r[f"f_{k}"] * r[f"{reg} | gap_{k}"]
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "sessions.csv", index=False)
    # counterfactual split, per session against the animal's pre means
    for reg in W:
        tb, gp = [], []
        for _, r in D.iterrows():
            P = D[(D.animal == r.animal) & (D.epoch == "pre")]
            tb.append(-sum(r[f"f_{k}"] * P[f"{reg} | gap_{k}"].mean() for k in CLASSES[1:]))
            gp.append(-sum(P[f"f_{k}"].mean() * r[f"{reg} | gap_{k}"] for k in CLASSES[1:]))
        D[f"{reg} | rest via time budget"] = tb
        D[f"{reg} | rest via gaps"] = gp
    D.to_csv(OUT / "sessions.csv", index=False)
    rng = np.random.default_rng(0)

    def change(col, ep):
        pre = {a: g[g.epoch == "pre"][col].dropna().tolist() for a, g in D.groupby("animal")}
        post = {a: g[g.epoch == ep][col].dropna().tolist() for a, g in D.groupby("animal")}
        post = {a: v for a, v in post.items() if v and pre.get(a)}
        per = {a: round(float(np.mean(post[a]) - np.mean(pre[a])), 3) for a in post}
        iv = kit.boot_delta(post, pre, rng)
        return iv.point, iv.lo, iv.hi, sum(v > 0 for v in per.values()), len(per), per

    recs = []
    cols = ([f"f_{k}" for k in CLASSES]
            + [f"{reg} | {q}" for reg in W for q in ["RESTnl", "m_rest", "all"] + [f"m_{k}" for k in CLASSES[1:]]
               + [f"gap_{k}" for k in CLASSES[1:]] + [f"contrib_{k}" for k in CLASSES[1:]]
               + ["rest via time budget", "rest via gaps"]])
    for col in cols:
        for ep in EPS:
            pt, lo, hi, up, n, per = change(col, ep)
            pre_mean = float(D[D.epoch == "pre"].groupby("animal")[col].mean().mean())
            recs.append({"readout": col, "epoch": ep, "pre": pre_mean, "change": pt, "lo": lo, "hi": hi, "k_up": up,
                         "n": n, "star": bool(lo > 0 or hi < 0), "per_animal": per})
            print(f"  {col} | {ep}: pre {pre_mean:+.3f} {per} -> {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    Cg = pd.DataFrame(recs)
    Cg.to_csv(OUT / "changes.csv", index=False)

    # figure: left = class time fractions; then per region the rest-level change and its terms
    regs = list(W)
    fig, axs = plt.subplots(1, 1 + len(regs), figsize=(4.6 * (1 + len(regs)), 4.6), squeeze=False)
    ax = axs[0, 0]
    for e, ep in enumerate(EPS):
        for k, cl in enumerate(CLASSES):
            g = Cg[(Cg.readout == f"f_{cl}") & (Cg.epoch == ep)].iloc[0]
            x = k + (e - 1) * 0.22
            ax.errorbar(x, g.change, yerr=[[g.change - g.lo], [g.hi - g.change]], fmt="o", color=EPOCH_LINE[ep],
                        ms=5, lw=1.4, capsize=0)
            if g.star:
                sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=8)
    pre_f = [Cg[(Cg.readout == f"f_{cl}") & (Cg.epoch == "acute")].iloc[0].pre for cl in CLASSES]
    ax.set_xticks(range(len(CLASSES)), [f"{cl}\npre {p:.2f}" for cl, p in zip(CLASSES, pre_f)], fontsize=8)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("change vs own pre (fraction of frames)")
    ax.set_title("time budget: fraction of frames per class", fontsize=9.5)
    terms = ["RESTnl", "m_rest", "rest via time budget", "rest via gaps"] + [f"contrib_{k}" for k in CLASSES[1:]]
    tlab = ["RESTnl\nlevel", "rest\nmean", "via time\nbudget", "via\ngaps"] + [f"-f x gap\n{k}" for k in CLASSES[1:]]
    for j, reg in enumerate(regs):
        ax = axs[0, j + 1]
        for e, ep in enumerate(EPS):
            for k, t in enumerate(terms):
                g = Cg[(Cg.readout == f"{reg} | {t}") & (Cg.epoch == ep)].iloc[0]
                sgn = -1 if t.startswith("contrib") else 1                   # plotted as its effect on the rest level
                pt, lo, hi = sgn * g.change, sgn * (g.hi if sgn < 0 else g.lo), sgn * (g.lo if sgn < 0 else g.hi)
                x = k + (e - 1) * 0.22
                ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=EPOCH_LINE[ep], ms=5, lw=1.4,
                            capsize=0, mfc=EPOCH_LINE[ep] if k < 2 else "white")
                if g.star:
                    sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=8)
        ax.axvline(1.5, color="0.8", lw=0.8)
        ax.axvline(3.5, color="0.8", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(len(terms)), tlab, fontsize=7.5)
        ax.set_title(f"{reg}: rest level vs session mean and its terms", fontsize=9.5)
        if j == 0:
            ax.set_ylabel("change vs own pre (pre pixel RMS)")
    for a in axs.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Why the raw rest level falls: the session mean is a time-weighted sum of states "
                 "(rest = -sum f_k (m_k - m_rest)); change vs own pre, boot_delta * = 95 % CI excludes 0", fontsize=10)
    fig.tight_layout()
    FIG = S.ROOT / "page_figs" / f"ss_rest_decomp{S.tag()}.png"
    sig_marks.reserve_headroom(fig)                  # stars in the top band, no top spine (Priya 10-30 / 11-03)
    save_png_svg(fig, FIG)
    print(f"wrote {FIG}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if not a.analyze_only:
        collect(a.jobs)
    analyze()


if __name__ == "__main__":
    main()
