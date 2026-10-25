"""A NO-STROKE drift comparison group: June pre -> August pre, for the representational-distance and
frozen-vs-refit readouts (Priya 2026-10-24: "add in a june vs august pre comparison group (for effects of
drift over time)").

June and August pre-stroke sessions are ~2 months apart with no lesion in between -- about the same gap as
August pre -> chronic. So each post-stroke readout that compares a session to a FROZEN pre-stroke reference
gets a twin in which June plays "pre" and August plays "post": what that readout does from time alone.

  DISTANCE  crossnobis distance of each session from a pooled reference, own-position diagonal, normalised by
            the animal's pre-stroke between-position scale (the units of `epoch_8diag` and
            `scripts/chronic_drift_vs_reorg.py`, whose helpers this reuses):
              aug_from_june   each August pre session vs the JUNE template          (drift, ~2 months)
              aug_loso        each August pre session vs the OTHER August sessions  (short-gap floor)
              post_from_aug   each post-stroke session vs the AUGUST template       (stroke + drift)
  DECODER   the frozen-vs-refit comparison of `grant_confusion._collect_5c(mode="paired_matched")`, cue
            window, licking trials, with the frozen model trained on JUNE pre only and size-matched to the
            refit (whole blocks), scored on each August session:
              june_loso       frozen on the other June sessions, scored on a June session (no-gap baseline)
              aug_from_june   frozen on all June sessions, scored on an August session
            F_drift = frozen accuracy, june_loso minus aug_from_june; G_drift = (refit - frozen) gap,
            aug_from_june minus june_loso. The chronic twins (F, G vs the pooled June + August frozen model)
            are in `recovery_trajectory_matched_cue_working.csv`.

Per animal, every number written (rule 8). Curated June sessions only (06-06..06-08 + whatever is curated).

    python -m scripts.june_august_drift [--out DIR] [--n-split 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
ALIGN, VARIANT, MIN_TRIALS = "cue", "working", 10


def _month(mmdd: str) -> str:
    return {"06": "june", "08": "aug"}.get(str(mmdd)[:2], "other")


def distance_rows(n_split: int, log=print) -> pd.DataFrame:
    from scripts.chronic_drift_vs_reorg import _pair_dist, _scale_for
    from wfield_local.epoch_figures import epoch_of_day
    from wfield_local.epochs import days_since_stroke
    from wfield_local.grant_kit import _collect_7, _pre_reference
    store, _ = _collect_7(ALIGN, VARIANT, MIN_TRIALS)
    rows = []
    for an in ANIMALS:
        if an not in store:
            continue
        pre, by_day = store[an]
        scale = _scale_for(_pre_reference(pre), n_split, seed=abs(hash((an, "scale"))) % (2**32))
        day = {m: days_since_stroke(f"{an}_{m}") for m in pre}
        jun = {m: p for m, p in pre.items() if _month(m) == "june"}
        aug = {m: p for m, p in pre.items() if _month(m) == "aug"}
        if not jun or not aug:
            log(f"{an}: june {len(jun)} / aug {len(aug)} pre sessions -- skipped")
            continue
        jref, aref = _pre_reference(jun), _pre_reference(aug)
        jday, aday = np.mean([day[m] for m in jun]), np.mean([day[m] for m in aug])

        def d(pat, ref, key, an=an, scale=scale):
            return _pair_dist(pat, ref, scale, n_split, abs(hash((an, key))) % (2**32))
        for m, p in aug.items():
            rows.append({"animal": an, "group": "aug_from_june", "session": m, "gap_days": day[m] - jday,
                         "distance": d(p, jref, ("aj", m))})
            rows.append({"animal": an, "group": "aug_loso", "session": m, "gap_days": np.nan,
                         "distance": d(p, _pre_reference(aug, exclude=m), ("al", m))})
        for dd, p in by_day.items():
            rows.append({"animal": an, "group": "post_from_aug", "session": f"day{int(dd)}",
                         "epoch": epoch_of_day(an, int(dd)), "gap_days": int(dd) - aday,
                         "distance": d(p, aref, ("pa", dd))})
        log(f"{an}: june {len(jun)}, aug {len(aug)}, post {len(by_day)}; scale {scale:.4f}")
    return pd.DataFrame(rows)


def decoder_rows(log=print) -> pd.DataFrame:
    from wfield_local import epochs
    from wfield_local.grant_confusion import _matched_frozen, _pooled_bundle, _refit_pred
    from wfield_local.grant_kit import _seed
    from wfield_local.locanmf_frozen_decoder import _pipe
    rows = []
    for an in ANIMALS:
        bd = _pooled_bundle(an, ALIGN)
        XE, YE, GE, BE = bd["XE"], bd["YE"], bd["GE"], bd["BE"]
        kept, pre_i, e_pre = bd["kept"], bd["pre_i"], bd["e_pre"]
        mm = {i: kept[i].split("_")[-1] for i in pre_i}
        J = sorted(i for i in pre_i if _month(mm[i]) == "june")
        A = sorted(i for i in pre_i if _month(mm[i]) == "aug")
        if len(J) < 2 or not A:
            log(f"{an}: june {len(J)} / aug {len(A)} pre sessions -- skipped")
            continue

        def score(i, train_sessions, group, XE=XE, YE=YE, GE=GE, BE=BE, e_pre=e_pre, mm=mm, an=an, test_mask=None):
            te = (e_pre & (GE == i)) if test_mask is None else test_mask
            tr = e_pre & np.isin(GE, train_sessions)
            X, y, blk = XE[te], YE[te], BE[te]
            if len(y) < 5 or len(np.unique(YE[tr])) < 2:
                return
            rf = _refit_pred(_pipe, X, y, blk)
            if rf is None:
                return
            k = min(5, int(np.unique(blk).size))
            n_target = int(round(len(y) * max(k - 1, 1) / max(k, 1)))
            got = _matched_frozen(_pipe, XE[tr], YE[tr], BE[tr], n_target,
                                  np.random.default_rng(abs(_seed(an, ALIGN, "june_aug", group)) + 1000003 * len(y) + int(blk[0])))
            if got is None:
                return
            fz_m = np.asarray(got[0].predict(X))
            fz = np.asarray(_pipe().fit(XE[tr], YE[tr]).predict(X))
            rows.append({"animal": an, "group": group, "session": mm[i], "n_trials": int(len(y)),
                         "frozen_acc": float(np.mean(fz == y)), "frozen_matched_acc": float(np.mean(fz_m == y)),
                         "refit_acc": float(np.mean(np.asarray(rf) == y))})
        for i in J:
            score(i, [j for j in J if j != i], "june_loso")
        for i in A:
            score(i, J, "aug_from_june")
        # THE STROKE TWIN, built the same way (10-24): frozen on AUGUST pre only (size-matched like every arm
        # here), scored on each post-stroke session's engaged licking trials, against an August LOSO baseline.
        # Without it the drift group (frozen on 3 June sessions) is compared with chronic numbers from a model
        # frozen on all 11 pre sessions, a broader training set that generalises better by construction.
        for i in A:
            score(i, [j for j in A if j != i], "aug_loso")
        post = [i for i in range(len(kept)) if i not in pre_i]
        for i in post:
            mm[i] = kept[i].split("_")[-1]
            ep = epochs.epoch_of(f"{an}_{mm[i]}")
            if ep in ("acute", "subacute", "chronic"):
                te = GE == i
                if te.any():
                    score(i, A, f"{ep}_from_aug", test_mask=te)
        log(f"{an}: decoder june {len(J)} / aug {len(A)} / post {len(post)} sessions")
    return pd.DataFrame(rows)


def summarise(D: pd.DataFrame, R: pd.DataFrame, chronic_csv: Path | None) -> pd.DataFrame:
    out = []
    for an in ANIMALS:
        r = {"animal": an}
        d = D[D.animal == an]
        for g in ("aug_from_june", "aug_loso"):
            r[f"dist_{g}"] = d[d.group == g].distance.median()
        r["gap_aug_from_june_days"] = d[d.group == "aug_from_june"].gap_days.median()
        for ep in ("acute", "subacute", "chronic"):
            s = d[(d.group == "post_from_aug") & (d.epoch == ep)]
            r[f"dist_{ep}_from_aug"] = s.distance.median()
            if ep == "chronic":
                r["gap_chronic_days"] = s.gap_days.median()
        q = R[R.animal == an]
        jl, ag = q[q.group == "june_loso"], q[q.group == "aug_from_june"]
        if len(jl) and len(ag):
            r["frozen_june_loso"] = jl.frozen_matched_acc.mean()
            r["frozen_aug_from_june"] = ag.frozen_matched_acc.mean()
            r["F_drift"] = r["frozen_june_loso"] - r["frozen_aug_from_june"]
            r["G_drift"] = (ag.refit_acc - ag.frozen_matched_acc).mean() - (jl.refit_acc - jl.frozen_matched_acc).mean()
        al = q[q.group == "aug_loso"]
        for ep in ("acute", "subacute", "chronic"):
            pe = q[q.group == f"{ep}_from_aug"]
            if len(al) and len(pe):
                r[f"F_{ep}_augfrozen"] = al.frozen_matched_acc.mean() - pe.frozen_matched_acc.mean()
                r[f"G_{ep}_augfrozen"] = (pe.refit_acc - pe.frozen_matched_acc).mean() - (al.refit_acc - al.frozen_matched_acc).mean()
        out.append(r)
    S = pd.DataFrame(out)
    if chronic_csv is not None and chronic_csv.exists():
        C = pd.read_csv(chronic_csv)
        if {"animal", "epoch"} <= set(C.columns):
            num = [c for c in ("F_deficit", "G_reorg") if c in C.columns]
            if num:
                S = S.merge(C[C.epoch == "chronic"].groupby("animal")[num].mean().add_suffix("_chronic").reset_index(),
                            on="animal", how="left")
    return S


def figure(D: pd.DataFrame, R: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    col = {"aug_from_june": "#1b7837", "acute": "#b2182b", "subacute": "#ef8a62", "chronic": "#2166ac"}
    for k, an in enumerate(ANIMALS):
        ax = axes[0, k]
        d = D[D.animal == an]
        a = d[d.group == "aug_from_june"]
        ax.scatter(a.gap_days, a.distance, c=col["aug_from_june"], marker="s", label="Aug pre vs June template (no stroke)")
        al = d[d.group == "aug_loso"].distance
        if len(al):
            ax.axhline(al.median(), color="0.5", ls=":", lw=1, label="Aug pre vs other Aug (floor)")
        for ep in ("acute", "subacute", "chronic"):
            p = d[(d.group == "post_from_aug") & (d.epoch == ep)]
            ax.scatter(p.gap_days, p.distance, c=col[ep], s=18, label=f"{ep} vs Aug template")
        ax.set(title=an, xlabel="days between template and session", ylabel="distance (pre between-position units)" if k == 0 else "")
        if k == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax2 = axes[1, k]
        q = R[R.animal == an]
        groups = (("june_loso", "June\nLOSO"), ("aug_from_june", "Aug\n<- June"), ("aug_loso", "Aug\nLOSO"),
                  ("acute_from_aug", "acute\n<- Aug"), ("subacute_from_aug", "subac.\n<- Aug"),
                  ("chronic_from_aug", "chronic\n<- Aug"))
        for j, (g, _lab) in enumerate(groups):
            s = q[q.group == g]
            x = np.full(len(s), j, float)
            ax2.scatter(x - 0.12, s.frozen_matched_acc, c="#555", s=14, label="frozen (size-matched)" if j == 0 else None)
            ax2.scatter(x + 0.12, s.refit_acc, c="#c51b7d", s=14, marker="^", label="refit within session" if j == 0 else None)
        ax2.axvspan(-0.5, 1.5, color="#1b7837", alpha=0.07, lw=0)
        ax2.set(xticks=range(len(groups)), xticklabels=[g[1] for g in groups],
                ylabel="accuracy (cue, licking trials)" if k == 0 else "", ylim=(0, 1), xlim=(-0.5, len(groups) - 0.5))
        ax2.tick_params(axis="x", labelsize=7)
        ax2.axhline(1 / 6, color="0.6", ls=":", lw=1)
        if k == 0:
            ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle("No-stroke drift group (green): June pre -> August pre. Top: crossnobis distance from a template by gap. "
                 "Bottom: frozen (size-matched) vs refit; '<- X' = decoder frozen on X", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-split", type=int, default=20)
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument("--reuse-distance", action="store_true", help="read distance_rows.csv from --out")
    ap.add_argument("--reuse-decoder", action="store_true", help="read decoder_rows.csv from --out")
    a = ap.parse_args(argv)
    lab = Path(PathResolver().root("labcams"))
    out = a.out or lab / "chronic_stability" / "june_august_drift"
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(out / "distance_rows.csv") if a.reuse_distance else distance_rows(a.n_split)
    D.to_csv(out / "distance_rows.csv", index=False)
    if a.reuse_decoder:
        R = pd.read_csv(out / "decoder_rows.csv")
    else:
        R = pd.DataFrame(columns=["animal", "group"]) if a.skip_decoder else decoder_rows()
    R.to_csv(out / "decoder_rows.csv", index=False)
    S = summarise(D, R, lab / "grant_figures" / "epoch" / "data" / "recovery_trajectory_matched_cue_working.csv")
    S.to_csv(out / "summary_by_animal.csv", index=False)
    pd.set_option("display.width", 240)
    print(S.round(3).T.to_string())
    figure(D, R, out / "june_august_drift.png")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
