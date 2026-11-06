"""What cortex LOOKS LIKE by what the animal did: trial-average maps per spout x video attempt class x epoch.

    WIDEFIELD_LICK_SOURCE=lp python -m wfield_local.attempt_maps [--animals ...] [--aligns cue lick precue] [--jobs 4]

Priya 10-24, goal question item 4 ("what does brain activity look like post stroke when an animal is attempting to do a
behavior it cannot execute or executes differently"). Family 15r pooled every attempted trial at a spout (hit +
incomplete under LP_licks); this splits them, with the same classes as `attempt_coding`:

  pre_hit      pre-stroke video hits at that spout (the reference)
  hit          post-stroke hits (executed, possibly differently)
  incomplete   attempted, tongue out within max_rt, no contact           -> "cannot execute"
  noattempt    no attempt while still working (cls no_try)
  reaction     jaw reaction, not an attempt (cls jaw_only), still working
  stopped      after the session's final collapse of attempting (`beta_maps._quit_mask`, which sees VIDEO attempts
               under the LP source because the engaged arm is the attempted arm)

MAP DEFINITION = 15r's rest-referenced form, not a copy of it: the SVT minus the time-local rest baseline
(`position_reference_maps.session_rest_svt_timelocal`), through `trial_features_cached` with the SAME args and signal
key as `session_raw_maps`' rest load, so the per-session trial features are shared with the 15r cache; window bins
averaged, projected with the session's atlas-registered U. ONE load per session x window gives every class (15r would
need a load per class). Lick window: hit / incomplete are aligned to their own first VIDEO lick; the no-lick classes
sit at the would-be-lick time only under `nolick_ref="would_be_lick"`, which 15r does not use -- so the lick window
here carries hit / incomplete only (as 15r's lick arm), and the cue and ENL windows carry every class.

AGGREGATION, per animal x epoch x spout x class: the mean over SESSIONS of each session's class mean (sessions with
>= MIN_TRIALS trials of that class at that spout; a session-weighted mean, so one long session cannot dominate), at
half resolution (270 x 320, 2 x 2 block mean) to keep the store small; trial and session counts alongside. Rule 8:
every figure is per animal first; the cohort panel is the mean of animal means.

OUTPUT <run>/imaging_encoding/attempt_maps/: maps_<align>_<animal>.npz (keys "<epoch>|<spout>|<class>" -> map, plus
"n|..." trial counts and "s|..." session counts), counts_<align>.csv, figures/ (per spout: rows = classes, columns =
epochs, per animal + cohort; and difference-from-pre-hit maps).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import run_mode
from wfield_local.attempt_coding import LESION_NAME, SPOUTS, POST_EPOCHS, _cue_classes

CLASSES = ("pre_hit", "pre_incomplete", "hit", "incomplete", "incomplete_pooled", "noattempt", "reaction", "stopped")
#: the TOUCH / REWARD control (10-24): pre-stroke attempts that missed the spout -- no contact, no water, like the
#: post-stroke failed attempts. Few per session (far contra 7 / 27 / 31 / 20 pre trials in all, PS92-95), so no
#: per-session floor and a TRIAL-weighted pool across sessions instead of the session-weighted mean.
POOLED = ("pre_incomplete", "incomplete_pooled")
#: derived classes: the same trials as the source class, pooled trial-weighted with no per-session floor (Priya 10-24
#: "why no chronic incomplete maps": chronic far-contra failed attempts are 8 / 12 / 38 / 9 trials in all, 1-3 per
#: session, so the session-weighted >= 5 floor drops almost every session). Kept beside the session-weighted class.
ALIAS = {"incomplete_pooled": "incomplete"}
ALIGNS = ("cue", "lick", "precue")
MIN_TRIALS = 5          # per session x spout x class to enter that session's class mean
DS = 2                  # 2 x 2 block mean -> 270 x 320


def _down(m):
    h, w = m.shape
    return m[: h // DS * DS, : w // DS * DS].reshape(h // DS, DS, w // DS, DS).mean((1, 3)).astype(np.float32)


def classes_for_rows(cls_e, cls_u, quit_u, pre, camera_u=None):
    """Class per engaged row and per no-lick row of ONE session (strings; "" = unused).

    CAMERA TRIAL GATE (rebuild 11-03): ``camera_u`` = the `trial_classes` class per no-lick row; then ``quit_u`` is not
    read, stopped = STOPPED_CLASSES (a sustained stop, Priya 11-03; "noattempt" = an isolated lapse) and the working split = class in
    GATES["working"] ("unknown" = neither), as `attempt_coding.classify`."""
    if pre:
        ce = np.select([cls_e == "hit", cls_e == "incomplete"], ["pre_hit", "pre_incomplete"], "").astype(object)
        return ce, np.full(len(cls_u), "", dtype=object)
    ce = np.where(np.isin(cls_e, ("hit", "incomplete")), cls_e, "").astype(object)
    cu = np.full(len(cls_u), "", dtype=object)
    if camera_u is not None:
        from wfield_local import trial_classes as _tc
        cam = np.asarray(camera_u, dtype=object)
        stop, work = np.isin(cam, list(_tc.STOPPED_CLASSES)), _tc.in_sel(cam, "miss_working")   # no contact (M6)
    else:
        stop, work = quit_u, ~quit_u
    cu[stop & ~np.isin(cls_u, ("", "unknown"))] = "stopped"
    cu[work & (cls_u == "no_try")] = "noattempt"
    if camera_u is not None:     # decision 3 (Priya 11-03): noattempt = isolated lapse (camera no_response in period)
        cu[(np.asarray(camera_u, dtype=object) == "no_response") & (cls_u == "no_try")] = "noattempt"
        # jaw_only its own class (Priya 11-03 late): camera jaw_only inside the working period -> "reaction"
        cu[(np.asarray(camera_u, dtype=object) == "jaw_only") & (cls_u == "jaw_only")] = "reaction"
    else:
        cu[work & (cls_u == "jaw_only")] = "reaction"
    return ce, cu


#: REST VARIANTS of the map reference (11-01). `rest` = the time-local REST above (the page's 10-24 figures 8.1 / 8.2,
#: default, unchanged). `restnl` = RESTnl (lick-free rest, flat; `restnl_mask`): features of the PLAIN SVT (the same
#: cached `svt:rank<K>` key as 15r's raw load), minus the flat baseline b_nl in every bin -- exact, because the window
#: features are linear bin means with no per-trial baseline; a session without a usable RESTnl baseline is skipped.
REST_VARIANTS = ("rest", "restnl")


def session_class_maps(label, align, post_s=None, rest_variant="rest"):
    """{(spout, class): (map at half resolution, n trials)} for one session and window."""
    from wfield_local import config, joint_basis, lp_licks
    from wfield_local.beta_maps import _quit_mask
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    from wfield_local.position_reference_maps import (MAP_SHAPE, REST_BASELINE_BINS,
                                                      session_rest_svt_timelocal)
    s = next(x for x in SESSIONS if x["label"] == label)
    post_s = post_s or lp_licks.post_s_for(align)
    u, v = joint_basis._load_session(s["mc"])
    a = _args("locanmf", align, post_s, baseline="none")
    b_nl = None
    if rest_variant == "restnl":
        from wfield_local import restnl_mask
        d = restnl_mask.baseline_or_build(label)      # built when missing (T4); None = excluded / rule wr only
        if d is None:
            print(f"  .. {label}: no usable RESTnl baseline (excluded or rule wr) -- skipped", flush=True)
            return {}
        b_nl = np.asarray(d["b_nl"], np.float64)
        X, y, _g, Xn, yn, _reg, idx_e, idx_n = trial_features_cached(
            s, a, signal=np.asarray(v), feat_region=np.arange(v.shape[0]),
            signal_key=f"svt:rank{v.shape[0]}", with_indices=True)
    else:
        base = session_rest_svt_timelocal(s, v)
        if base is None:
            return {}
        sig = np.asarray(v) - np.asarray(base)
        # the time-local REST baseline is built on the ENGAGED rest frames (rest_engagement), so it moves with
        # WIDEFIELD_GATE_SOURCE -- and `tf-*` kinds are exempt from the gate tag; key it here (unchanged by default)
        from wfield_local import gate_source
        gk = f":{gate_source.cache_tag()}" if gate_source.is_lp() else ""
        X, y, _g, Xn, yn, _reg, idx_e, idx_n = trial_features_cached(
            s, a, signal=sig, feat_region=np.arange(v.shape[0]),
            signal_key=f"svt:rank{v.shape[0]}:restlocal{REST_BASELINE_BINS}{gk}", with_indices=True)
    X, y, Xn, yn = np.asarray(X), np.asarray(y), np.asarray(Xn), np.asarray(yn)
    K = u.shape[1]
    if b_nl is not None:                      # flat baseline: the same (K,) vector off every bin of every trial
        if X.shape[1] % K or (len(yn) and Xn.shape[1] % K):
            raise ValueError(f"{label}: {X.shape[1]} features is not a multiple of {K}")
        X = X - np.tile(b_nl, X.shape[1] // K)[None, :]
        if len(yn):
            Xn = Xn - np.tile(b_nl, Xn.shape[1] // K)[None, :]
    cls_cue, _tid = _cue_classes([label], a.max_rt)[label]
    pick = lambda idx: np.array([cls_cue[k] if k < len(cls_cue) else "" for k in idx], object).astype(str)
    cls_e, cls_u = pick(idx_e), pick(idx_n)
    pre = config.session_phase(config.animal_of(label), label.split("_")[-1]) == "pre"
    from wfield_local import trial_classes as _tc
    if _tc.camera_gate_on():                  # camera trial gate (rebuild 11-03): no DAQ quit mask is built
        ce, cu = classes_for_rows(cls_e, cls_u, None, pre, camera_u=_tc.row_classes(label, idx_n))
    else:
        quit_u = (_quit_mask(s, idx_e, idx_n, y, yn) if len(yn) else np.zeros(0, bool))
        ce, cu = classes_for_rows(cls_e, cls_u, quit_u, pre)
    if align == "lick":                       # no-lick rows have no lick to align to (see module docstring)
        cu[:] = ""
    rows = [(X, y, ce)] + ([(Xn, yn, cu)] if len(yn) else [])
    out = {}
    for Xa, ya, ca in rows:
        if not len(ya):
            continue
        n_bins = Xa.shape[1] // K
        sp = np.array([LESION_NAME.get(POSITION_NAMES.get(int(c), str(c)), "?") for c in ya])
        for P in SPOUTS:
            for c in CLASSES:
                m = (sp == P) & (ca == ALIAS.get(c, c))
                if m.sum() < (1 if c in POOLED else MIN_TRIALS):
                    continue
                f = Xa[m].mean(0).reshape(n_bins, K).mean(0)
                out[(P, c)] = (_down((u @ f).reshape(MAP_SHAPE)), int(m.sum()))
    return out


def _job(args):
    label, align = args[:2]
    rv = args[2] if len(args) > 2 else "rest"
    try:
        return label, session_class_maps(label, align, rest_variant=rv)
    except Exception as ex:                                            # noqa: BLE001
        from wfield_local import trial_classes as _tc
        _tc.reraise_under_camera(ex, f"attempt_maps {label} {align}")   # review N9
        print(f"  !! {label} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
        return label, None


def collect(animal, align, jobs=4, rest_variant="rest"):
    """Aggregate one animal's sessions -> ({key: map}, counts DataFrame)."""
    from wfield_local import analysis_kit as ak, lp_licks
    if not lp_licks.is_lp():
        raise RuntimeError(f"set {lp_licks.LICK_SOURCE_ENV}=lp: the classes are video attempts")
    A = pd.read_csv(lp_licks.attempts_path(), usecols=["label", "epoch"]).drop_duplicates("label")
    ep = dict(zip(A.label, A.epoch))
    labels = [l for l in ak.curated_labels([animal]) if ep.get(l) in ("pre",) + POST_EPOCHS]
    if jobs > 1:
        pairs, fail = ak.fan_sessions([(l, align, rest_variant) for l in labels], _job, jobs=jobs)
        if fail:                                   # a short table looks like a legitimately short one: refuse
            raise RuntimeError(f"{animal} {align}: {len(fail)} session(s) failed in the pool: {fail}")
        res = [v for _item, v in pairs]
    else:
        res = [_job((l, align, rest_variant)) for l in labels]
    bad = [l for l, d in res if d is None]
    if bad:
        raise RuntimeError(f"{animal} {align}: no maps for {bad}")
    acc, rows = {}, []
    for label, d in sorted(res, key=lambda r: r[0]):
        if not d:
            continue
        for (P, c), (m, n) in d.items():
            k = (ep[label], P, c)
            acc.setdefault(k, []).append((m, n))
            rows.append({"animal": animal, "align": align, "label": label, "epoch": ep[label], "spout": P,
                         "cls": c, "n": n})
    maps = {}
    for (e, P, c), mn in acc.items():
        w = np.array([n for _m, n in mn], float) if c in POOLED else np.ones(len(mn))
        maps[f"{e}|{P}|{c}"] = (np.tensordot(w, np.stack([m for m, _n in mn]), 1) / w.sum()).astype(np.float32)
    return maps, pd.DataFrame(rows)


def out_dir(rest_variant="rest"):
    """`attempt_maps/` (REST, 10-24) or `attempt_maps_restnl/` (RESTnl, August-only pre; 11-01) -- never the same."""
    from wfield_local import lp_licks, run_mode
    # run-mode routing (rebuild 11-03): a tagged run (camera gate, hemo variant ...) writes <dir>/<tag>/; default unchanged
    return run_mode.subdir(lp_licks._enc_dir() / ("attempt_maps" if rest_variant == "rest"
                                                                        else f"attempt_maps_{rest_variant}"),
                           lp_family=True)                     # LP-only family (review N1)


EPOCHS = ("pre",) + POST_EPOCHS
SHOW = (("pre", "pre_hit"), ("pre", "pre_incomplete"), ("acute", "hit"), ("acute", "incomplete"), ("acute", "noattempt"),
        ("acute", "reaction"), ("acute", "stopped"), ("subacute", "hit"), ("subacute", "incomplete_pooled"),
        ("chronic", "hit"), ("chronic", "incomplete_pooled"))


def _mask_edges():
    """Brain mask + atlas edges at map resolution (beta_maps' shared-grid versions, 2 x 2 block-reduced)."""
    from wfield_local.beta_maps import atlas_edges, brain_mask
    m = brain_mask()
    e = atlas_edges()
    m = None if m is None else _down(np.asarray(m, float)) > 0.5
    e = None if e is None else _down(np.asarray(e, float)) > 0
    return m, e


def product_name(stem: str, ext: str) -> str:
    """R4 (fourth review): EVERY attempt_maps product -- maps npz, counts and hemisphere CSVs, figures -- carries the same
    per-session cache key (`run_mode.keyed_name(cache=True)`: hemo / projection / selection / class-code csig), so a
    class-code change can never pair new maps with old tables or figures. Unchanged names in a default run."""
    return run_mode.keyed_name(f"{stem}{ext}", cache=True)


def maps_path(od, align, an) -> Path:
    return Path(od) / product_name(f"maps_{align}_{an}", ".npz")


def hemispheres_path(od, align) -> Path:
    """The hemisphere table of ``align`` in output folder ``od`` (read by rebuild_figs/fig_goal_summary)."""
    return Path(od) / product_name(f"hemispheres_{align}", ".csv")


def _stored_maps(od, align, animals) -> dict:
    """{animal: loaded maps} -- RAISES when an expected keyed maps file is missing (R4: never a partial figure set)."""
    miss = [an for an in animals if not maps_path(od, align, an).exists()]
    if miss:
        raise FileNotFoundError(f"[attempt_maps] {align}: no maps for {miss} in {od} under this run's key "
                                f"({run_mode.cache_tag() or 'default'}); run attempt_maps without --figures first")
    return {an: np.load(maps_path(od, align, an)) for an in animals}


def hemisphere_table(od, align, animals):
    """Per animal x epoch x spout x class: mean map value in the image-LEFT (ipsilesional) and image-RIGHT
    (contralesional) halves inside the brain mask, and the same for the difference from that spout's pre-stroke hits."""
    m, _e = _mask_edges()
    rows = []
    for an, Z in _stored_maps(od, align, animals).items():
        for k in Z.files:
            e, P, c = k.split("|")
            M = Z[k]
            mm = np.ones_like(M, bool) if m is None else m
            half = M.shape[1] // 2
            ref = Z[f"pre|{P}|pre_hit"] if f"pre|{P}|pre_hit" in Z.files else None
            r = {"animal": an, "align": align, "epoch": e, "spout": P, "cls": c,
                 "ipsi": float(M[:, :half][mm[:, :half]].mean()), "contra": float(M[:, half:][mm[:, half:]].mean())}
            if ref is not None:
                D = M - ref
                r["d_ipsi"] = float(D[:, :half][mm[:, :half]].mean())
                r["d_contra"] = float(D[:, half:][mm[:, half:]].mean())
                r["ref_ipsi"] = float(ref[:, :half][mm[:, :half]].mean())
                r["ref_contra"] = float(ref[:, half:][mm[:, half:]].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def _run_text() -> str:
    """Lick source, gate and baseline of this run, for the title (lp-final 11-02: every title names all three)."""
    from wfield_local import config, gate_source, lp_licks
    return (f"licks: {'LP video (cam4 + cam1)' if lp_licks.is_lp() else 'DAQ contact'}; trial classes: LP video "
            f"attempts; gate: {'DAQ contact OR LP attempt' if gate_source.is_lp() else 'as lick source'}; pre: "
            f"{'August only' if config.pre_from() else 'June + August'}")


def figures(od, align, animals, spouts=("far_contra", "far_mid", "far_ipsi"), rest_variant="rest"):
    """Per spout: one figure per animal-set, rows = animals (+ mean of animal means), columns = SHOW (epoch x class);
    one absolute version (shared symmetric scale per row) and one minus that animal's pre-stroke hits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m, edges = _mask_edges()
    fd = od / "figures"
    fd.mkdir(exist_ok=True)
    Zs = _stored_maps(od, align, animals)
    out = []
    for P in spouts:
        for diff in (False, True):
            cols = [sc for sc in SHOW if not (diff and sc == ("pre", "pre_hit"))]
            fig, axes = plt.subplots(len(Zs) + 1, len(cols) + 1, figsize=(1.9 * len(cols) + 0.5, 1.75 * (len(Zs) + 1)),
                                     squeeze=False, gridspec_kw={"width_ratios": [1] * len(cols) + [0.07]})
            stack = {sc: [] for sc in cols}
            for i, (an, Z) in enumerate(list(Zs.items()) + [("mean", None)]):
                mats = []
                for sc in cols:
                    k = f"{sc[0]}|{P}|{sc[1]}"
                    if Z is None:
                        M = np.mean(stack[sc], 0) if len(stack[sc]) >= 2 else None
                    else:
                        M = Z[k] if k in Z.files else None
                        ref = Z[f"pre|{P}|pre_hit"] if f"pre|{P}|pre_hit" in Z.files else None
                        if M is not None and diff:
                            M = None if ref is None else M - ref
                        if M is not None:
                            stack[sc].append(M)
                    mats.append(M)
                vals = [np.abs(M[m] if m is not None else M) for M in mats if M is not None]
                v = float(np.percentile(np.concatenate(vals), 99)) if vals else 1.0
                im = None
                for j, (sc, M) in enumerate(zip(cols, mats)):
                    ax = axes[i, j]
                    ax.set_xticks([]); ax.set_yticks([])
                    if M is not None:
                        show = np.where(m, M, np.nan) if m is not None else M
                        im = ax.imshow(show, cmap="RdBu_r", vmin=-v, vmax=v)
                        if edges is not None:
                            ax.contour(edges, levels=[0.5], colors="k", linewidths=0.2)
                    else:
                        ax.set_facecolor("0.92")
                    if i == 0:
                        ax.set_title(f"{sc[0]}\n{sc[1]}", fontsize=7)
                    if j == 0:
                        ax.set_ylabel(f"{an}\n(+/-{v:.2g})", fontsize=7)
                if im is not None:                         # the row's shared symmetric scale
                    cb = fig.colorbar(im, cax=axes[i, -1])
                    cb.ax.tick_params(labelsize=5)
                    cb.set_label("dF", fontsize=6)
                else:
                    axes[i, -1].set_visible(False)
            if rest_variant == "restnl":
                what = ("RESTnl-referenced dF, minus pre-stroke hits" if diff else "RESTnl-referenced dF")
                fig.suptitle(f"{P}, {align} window: {what}. RESTnl = lick-free rest, flat (rest licks excluded: DAQ "
                             f"contact + cam4 LP video, [-0.5, +2.0] s); {_run_text()}"
                             f"\n(image left = ipsilesional; mean = mean of animal means, >= 2 animals; per-row "
                             f"symmetric 99th-pct scale)", fontsize=8)
                fig.tight_layout(rect=(0, 0, 1, 0.93))
            else:
                what = ("REST-referenced dF (time-local rest), minus pre-stroke hits" if diff
                        else "REST-referenced dF (time-local rest)")
                fig.suptitle(f"{P}, {align} window: {what}; {_run_text()}\n(image left = ipsilesional; mean = mean "
                             f"of animal means, >= 2 animals; per-row symmetric 99th-pct scale)", fontsize=8)
                fig.tight_layout(rect=(0, 0, 1, 0.93))
            sfx = "" if rest_variant == "rest" else f"_{rest_variant}"
            fn = fd / product_name(f"attempt_maps_{align}_{P}{'_minus_pre' if diff else ''}{sfx}", ".png")
            save_png_svg(fig, fn, dpi=130)
            plt.close(fig)
            out.append(fn)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--animals", nargs="+", default=["PS92", "PS93", "PS94", "PS95"])
    ap.add_argument("--aligns", nargs="+", default=list(ALIGNS), choices=ALIGNS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--figures", "--plot-only", action="store_true",
                    help="figures + hemisphere table from the stored maps (the figure data), then exit")
    ap.add_argument("--rest-variant", choices=REST_VARIANTS, default="rest",
                    help="map reference: rest (time-local REST, default) or restnl (lick-free flat RESTnl; run with "
                         "WIDEFIELD_PRE_FROM=0801 for the August-only pre; baselines from `restnl_mask`)")
    a = ap.parse_args(argv)
    # an explicit --out (e.g. attempt_coding_pp) is routed like out_dir() (LP family; T1 / T6, third review)
    od = run_mode.routed_output(a.out, guard=False, lp_family=True) if a.out else out_dir(a.rest_variant)
    from wfield_local.writeguard import assert_writable
    assert_writable(od / "_")   # review H3: a tagged run's share write must sit under its tag folder
    od.mkdir(parents=True, exist_ok=True)
    if a.figures:
        for align in a.aligns:
            hemisphere_table(od, align, a.animals).to_csv(hemispheres_path(od, align), index=False)
            for fn in figures(od, align, a.animals, rest_variant=a.rest_variant):
                print(f"[attempt_maps] wrote {fn.name}", flush=True)
        return 0
    for align in a.aligns:
        counts = []
        for an in a.animals:
            f = maps_path(od, align, an)
            if f.exists():
                print(f"[attempt_maps] {an} {align}: exists, skipped", flush=True)
                continue
            maps, C = collect(an, align, a.jobs, rest_variant=a.rest_variant)
            if not maps:
                # T4 (third review): an EMPTY maps file would be skipped as "exists" by every later run -- never write it
                raise RuntimeError(f"[attempt_maps] {an} {align}: no session produced maps (rest_variant "
                                   f"{a.rest_variant}); nothing written -- check the RESTnl baselines / inputs")
            np.savez_compressed(f, **maps)
            C.to_csv(od / product_name(f"counts_{align}_{an}", ".csv"), index=False)
            counts.append(C)
            print(f"[attempt_maps] {an} {align}: {len(maps)} maps from {C.label.nunique() if len(C) else 0} sessions",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
