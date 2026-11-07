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
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

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


#: STORED BY DEFAULT (11-02 pass 2; Priya: "yes cache all frozen decoders"; rule 10). The full (not size-matched) frozen
#: decoder of `decoder_rows` -- trained on the animal's JUNE pre sessions or AUGUST pre sessions, all of them or all but
#: the scored one -- comes from `frozen_models.load_or_fit`, kind `drift_decoder`, ONE spec per month (align
#: "cue-june" / "cue-aug": two training sets never share an id), payload {"full", "loso": {label: model}}. OPT-OUT
#: WIDEFIELD_DRIFT_FROZEN=0 refits per scored session as before. The size-matched draws (`_matched_frozen`, seeded per
#: scored session) are STORED too, like 5rm's `matched_frozen` (merge review N4): ONE spec per animal (kind
#: `drift_matched`, its fingerprint covering the pooled bundle, + the scaling), keyed inside the payload by (group,
#: scored session, n_target, seed); a hit replays every draw, a miss fits them in memory and stores them. Under WIDEFIELD_LICK_SOURCE=lp the store serves
#: the LP-TRAINED model (default since 11-03, Priya), the DAQ-trained one (the animal's DAQ pre sessions of that month)
#: with WIDEFIELD_LP_FROZEN=daq.
DRIFT_FROZEN_ENV = "WIDEFIELD_DRIFT_FROZEN"
_DRIFT_MODELS: dict = {}


def drift_payload(an, bd, month, log=print, scaling=None):
    """{"full", "loso"} for one animal x month from the store (fitted once), or None when opted out. Under bal_em
    (``scaling``) the models are fitted on the bal_em TRAINING frame (`spout_scaling.train_frame`: each month session
    centred on its true class means, the frame `FrozenFrame` trains in) and the spec carries the scaling; under
    session_z on the production frame, as before."""
    from wfield_local import config, frozen_models, spout_scaling
    from wfield_local.locanmf_frozen_decoder import _pipe
    if not frozen_models.family_on(DRIFT_FROZEN_ENV):
        return None
    XE, YE, GE, kept, e_pre = bd["XE"], bd["YE"], bd["GE"], bd["kept"], bd["e_pre"]
    idx = sorted(i for i in bd["pre_i"] if _month(kept[i].split("_")[-1]) == month)
    train = [kept[i] for i in idx]
    if frozen_models.daq_trained_in_lp():
        train = [x for x in config.phase_labels("pre") if config.animal_of(x) == an and _month(x.split("_")[-1]) == month]
    spec = frozen_models.make_spec(an, "drift_decoder", align=f"{ALIGN}-{month}", source="locanmf",
                                   train_labels=train, post_s=2.0, basis_id=bd.get("basis_id"),
                                   n_features=int(XE.shape[1]),
                                   train_data=(XE[e_pre & np.isin(GE, idx)], YE[e_pre & np.isin(GE, idx)],
                                               GE[e_pre & np.isin(GE, idx)]),
                                   **spout_scaling.spec_kw(scaling))
    XF = spout_scaling.train_frame(XE, YE, GE, idx) if scaling else XE

    def _fit():
        full = e_pre & np.isin(GE, idx)
        return {"full": _pipe().fit(XF[full], YE[full]),
                "loso": {kept[i]: _pipe().fit(XF[full & (GE != i)], YE[full & (GE != i)]) for i in idx
                         if len(np.unique(YE[full & (GE != i)])) >= 2}}

    payload, status = frozen_models.load_or_fit(spec, _fit, meta={"month": month, "n_sessions": len(idx)})
    log(f"{an}: drift_decoder {frozen_models._slug(spec)} {status} "
        f"(trained on {spec.get('lick_source', 'daq')} {month} pre trials)")
    return payload


def _stored_month_model(an, bd, train_sessions, scaling=None):
    """The stored model for one `score` training set (all of one month's pre sessions, or all but one), or None.

    The in-memory path fits `_pipe()` on ``e_pre & isin(GE, train_sessions)``; the stored payload holds that exact fit
    for every training set `decoder_rows` uses (June LOSO / all June / August LOSO / all August)."""
    from wfield_local import frozen_models
    if not frozen_models.family_on(DRIFT_FROZEN_ENV):
        return None
    kept = bd["kept"]
    months = {_month(kept[i].split("_")[-1]) for i in train_sessions}
    if len(months) != 1:
        raise ValueError(f"{an}: a drift training set spans months {sorted(months)}")
    month = months.pop()
    key = (an, month, id(bd), scaling)
    if key not in _DRIFT_MODELS:
        _DRIFT_MODELS[key] = drift_payload(an, bd, month, scaling=scaling)
    pay = _DRIFT_MODELS[key]
    idx = sorted(i for i in bd["pre_i"] if _month(kept[i].split("_")[-1]) == month)
    left = sorted(set(idx) - set(train_sessions))
    if not left:
        return pay["full"]
    if len(left) == 1 and kept[left[0]] in pay["loso"]:
        return pay["loso"][kept[left[0]]]
    raise ValueError(f"{an} {month}: training set {sorted(train_sessions)} is neither all sessions nor all but one")


def decoder_rows(log=print) -> pd.DataFrame:
    from wfield_local import epochs
    from wfield_local.grant_confusion import _matched_frozen, _pooled_bundle, _refit_pred
    from wfield_local import spout_scaling as ss
    from wfield_local.grant_kit import _frozen_score_rows, _seed
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
        # THE SIZE-MATCHED DRAWS THROUGH THE STORE (merge review N4; as grant_confusion's 5rm matched_frozen)
        from wfield_local import frozen_models
        mm_spec, mm_old, mm_new = None, None, {}
        if frozen_models.family_on(DRIFT_FROZEN_ENV):
            mm_spec = frozen_models.make_spec(
                an, "drift_matched", align=f"{ALIGN}-june_aug-seed{abs(_seed(an, ALIGN, 'june_aug'))}",
                source="locanmf", train_labels=[kept[i] for i in sorted(pre_i)], post_s=2.0,
                basis_id=bd.get("basis_id"), n_features=int(XE.shape[1]),
                train_data=(XE, YE, GE, BE, sorted(pre_i), bd.get("cls")), **ss.spec_kw())
            _d, _o = frozen_models.find(mm_spec)
            if _d is not None:
                mm_old = frozen_models._read(_d)

        def _mf(key, pool, n_target, seed, mm_old=mm_old, mm_new=mm_new):
            if mm_old is not None and key in mm_old:
                return mm_old[key]
            got = _matched_frozen(_pipe, *pool, n_target, np.random.default_rng(seed))
            mm_new[key] = got
            return got

        def score(i, train_sessions, group, XE=XE, YE=YE, GE=GE, BE=BE, e_pre=e_pre, mm=mm, an=an, test_mask=None,
                  bd=bd, mf=_mf, kept=kept):
            te = (e_pre & (GE == i)) if test_mask is None else test_mask
            tr = e_pre & np.isin(GE, train_sessions)
            X, y, blk = XE[te], YE[te], BE[te]
            if len(y) < 5 or len(np.unique(YE[tr])) < 2:
                return
            rf = _refit_pred(_pipe, X, y, blk)
            if rf is None:
                return
            # THE FROZEN ARMS IN THE bal_em FRAMES (production since 2026-11-03, `spout_scaling`), as the chronic twins
            # in `_collect_5c`: training sessions centred on their true class means, the scored session centred by the
            # label-free EM centre of its engaged rows under each frozen model. Identity under session_z.
            # THE MONTH MODEL IS STORED (rule 10) in the frame it scores in: the FrozenFrame store hook serves
            # `drift_payload` (fitted on the bal_em training frame under bal_em, on the production frame under
            # session_z); WIDEFIELD_DRIFT_FROZEN=0 = the same fit in memory
            def _store(fit_all, an=an, bd=bd, ts=tuple(train_sessions)):
                st = _stored_month_model(an, bd, list(ts), scaling=ss.spout_scaling())
                return {"full": st, "loso": {}} if st is not None else fit_all()

            ff = ss.FrozenFrame(XE, YE, GE, train_sessions, _pipe, train_mask=e_pre, loso=False, store=_store,
                                labels=dict(enumerate(kept)),
                                cls_e=bd["cls"][0] if bd.get("cls") is not None else None)
            k = min(5, int(np.unique(blk).size))
            n_target = int(round(len(y) * max(k - 1, 1) / max(k, 1)))
            seed = abs(_seed(an, ALIGN, "june_aug", group)) + 1000003 * len(y) + int(blk[0])
            got = mf(f"{group}|{kept[i]}|{int(n_target)}|{int(seed)}", (ff.XT[tr], YE[tr], BE[tr]), n_target, seed)
            if got is None:
                return
            Xm = X
            if ff.scaling and [int(c) for c in got[0].classes_] == list(range(6)):
                Xm = _frozen_score_rows(bd, got[0], got[0].spout_pi_train_, i, X)[0]
            fz_m = np.asarray(got[0].predict(Xm))
            fz = np.asarray(ff.full.predict(ff.apply(X, i)))          # the stored month model, in its scoring frame
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
        if mm_spec is not None:
            if mm_old is None:
                _p, _st = frozen_models.load_or_fit(mm_spec, lambda mm_new=mm_new: dict(mm_new),
                                                    meta={"n_draws": len(mm_new)})
                log(f"{an}: drift_matched {frozen_models._slug(mm_spec)} {_st} ({len(mm_new)} draws)")
            elif mm_new:
                log(f"  !! {an}: {len(mm_new)} matched draws missing from the stored {frozen_models._slug(mm_spec)}; "
                    "fitted in memory (not stored)")
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
    # EPOCH-COLOUR CONVENTION (A13, Priya 11-07): colours are THE epoch palette (orofacial_cohort.EPOCH_LINE; the
    # no-stroke June -> August group is pre-stroke, so the pre colour), points OPEN; the group is told apart by marker
    # (panel = one animal, so the animal shape carries nothing here): square = Aug pre vs June, circle = post vs Aug
    from wfield_local.orofacial_cohort import EPOCH_LINE as col
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    for k, an in enumerate(ANIMALS):
        ax = axes[0, k]
        d = D[D.animal == an]
        a = d[d.group == "aug_from_june"]
        ax.scatter(a.gap_days, a.distance, facecolors="none", edgecolors=col["pre"], marker="s", lw=1.0,
                   label="Aug pre vs June template (no stroke)")
        al = d[d.group == "aug_loso"].distance
        if len(al):
            ax.axhline(al.median(), color="0.5", ls=":", lw=1, label="Aug pre vs other Aug (floor)")
        for ep in ("acute", "subacute", "chronic"):
            p = d[(d.group == "post_from_aug") & (d.epoch == ep)]
            ax.scatter(p.gap_days, p.distance, facecolors="none", edgecolors=col[ep], s=22, lw=1.0,
                       label=f"{ep} vs Aug template")
        ax.set(title=an, xlabel="days between template and session", ylabel="distance (pre between-position units)" if k == 0 else "")
        if k == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax2 = axes[1, k]
        q = R[R.animal == an]
        groups = (("june_loso", "June\nLOSO"), ("aug_from_june", "Aug\n<- June"), ("aug_loso", "Aug\nLOSO"),
                  ("acute_from_aug", "acute\n<- Aug"), ("subacute_from_aug", "subac.\n<- Aug"),
                  ("chronic_from_aug", "chronic\n<- Aug"))
        gep = {"june_loso": "pre", "aug_from_june": "pre", "aug_loso": "pre", "acute_from_aug": "acute",
               "subacute_from_aug": "subacute", "chronic_from_aug": "chronic"}
        for j, (g, _lab) in enumerate(groups):
            s = q[q.group == g]
            x = np.full(len(s), j, float)
            c_ = col[gep[g]]
            ax2.scatter(x - 0.12, s.frozen_matched_acc, facecolors="none", edgecolors=c_, s=16, lw=0.9, marker="o",
                        label="frozen (size-matched)" if j == 0 else None)
            ax2.scatter(x + 0.12, s.refit_acc, facecolors="none", edgecolors=c_, s=18, lw=0.9, marker="v",
                        label="refit within session" if j == 0 else None)
        ax2.axvspan(-0.5, 1.5, color="0.5", alpha=0.07, lw=0)
        ax2.set(xticks=range(len(groups)), xticklabels=[g[1] for g in groups],
                ylabel="accuracy (cue, licking trials)" if k == 0 else "", ylim=(0, 1), xlim=(-0.5, len(groups) - 0.5))
        ax2.tick_params(axis="x", labelsize=7)
        ax2.axhline(1 / 6, color="0.6", ls=":", lw=1)
        if k == 0:
            ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle("No-stroke drift group (grey band; pre colour): June pre -> August pre. Top: crossnobis distance from a "
                 "template by gap. Bottom: frozen (size-matched, circles) vs refit (triangles); '<- X' = decoder frozen on X; "
                 "colour = epoch", fontsize=10)
    fig.tight_layout()
    save_png_svg(fig, out_png, dpi=130)


def primary_chronic_csv(lab: Path) -> Path:
    """The stroke twin's input: the PRIMARY run's (August-only pre, `run_mode.PRIMARY_PRE_FROM`) matched recovery
    trajectory, ``grant_figures/<primary tag>/epoch/data/recovery_trajectory_matched_cue_working.csv`` (untagged in the
    default run). REFUSES if it is missing (review 7B B1: never a silent untagged / stale read)."""
    from wfield_local import run_mode
    f = run_mode.primary_output_dir(Path(lab) / "grant_figures" / "epoch") / "data" / \
        "recovery_trajectory_matched_cue_working.csv"
    if not f.exists():
        raise FileNotFoundError(f"june_august_drift: the primary run's {f} is missing -- render the primary epoch "
                                "families (recovery_trajectory) first")
    return f


def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    # review 7B B1: a TAGGED run is allowed only as the profile's june_august_drift block (PRE_FROM unset: its own
    # June + August pre set), writing under its own tag (`run_mode.june_august_drift_dir`); the default run is unchanged
    from wfield_local import config as _cfg0
    from wfield_local import lp_licks as _lpl0
    if _lpl0.output_tag() and _cfg0.pre_from():
        raise RuntimeError("june_august_drift: a tagged run must be the render profile's june_august_drift block "
                           "(WIDEFIELD_PRE_FROM unset: --block june_august_drift)")
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-split", type=int, default=20)
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument("--reuse-distance", action="store_true", help="read distance_rows.csv from --out")
    ap.add_argument("--reuse-decoder", action="store_true", help="read decoder_rows.csv from --out")
    a = ap.parse_args(argv)
    from wfield_local import config as _cfg
    _cfg.require_full_pre("june_august_drift (drift decoder)")   # June pre is its reference (rebuild 11-03)
    lab = Path(PathResolver().root("cohort_out"))        # review 8B B1: the DERIVED root (labcams is the input tree)
    out = a.out or _rm_guard.june_august_drift_dir(lab)
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(out / "distance_rows.csv") if a.reuse_distance else distance_rows(a.n_split)
    D.to_csv(out / "distance_rows.csv", index=False)
    if a.reuse_decoder:
        R = pd.read_csv(out / "decoder_rows.csv")
    else:
        R = pd.DataFrame(columns=["animal", "group"]) if a.skip_decoder else decoder_rows()
    R.to_csv(out / "decoder_rows.csv", index=False)
    S = summarise(D, R, primary_chronic_csv(lab))
    S.to_csv(out / "summary_by_animal.csv", index=False)
    pd.set_option("display.width", 240)
    print(S.round(3).T.to_string())
    figure(D, R, out / "june_august_drift.png")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
