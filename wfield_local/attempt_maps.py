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

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local.attempt_coding import LESION_NAME, SPOUTS, POST_EPOCHS, _cue_classes

CLASSES = ("pre_hit", "pre_incomplete", "hit", "incomplete", "noattempt", "reaction", "stopped")
#: the TOUCH / REWARD control (10-24): pre-stroke attempts that missed the spout -- no contact, no water, like the
#: post-stroke failed attempts. Few per session (far contra 7 / 27 / 31 / 20 pre trials in all, PS92-95), so no
#: per-session floor and a TRIAL-weighted pool across sessions instead of the session-weighted mean.
POOLED = ("pre_incomplete",)
ALIGNS = ("cue", "lick", "precue")
MIN_TRIALS = 5          # per session x spout x class to enter that session's class mean
DS = 2                  # 2 x 2 block mean -> 270 x 320


def _down(m):
    h, w = m.shape
    return m[: h // DS * DS, : w // DS * DS].reshape(h // DS, DS, w // DS, DS).mean((1, 3)).astype(np.float32)


def classes_for_rows(cls_e, cls_u, quit_u, pre):
    """Class per engaged row and per no-lick row of ONE session (strings; "" = unused)."""
    if pre:
        ce = np.select([cls_e == "hit", cls_e == "incomplete"], ["pre_hit", "pre_incomplete"], "").astype(object)
        return ce, np.full(len(cls_u), "", dtype=object)
    ce = np.where(np.isin(cls_e, ("hit", "incomplete")), cls_e, "").astype(object)
    cu = np.full(len(cls_u), "", dtype=object)
    cu[quit_u & ~np.isin(cls_u, ("", "unknown"))] = "stopped"
    cu[~quit_u & (cls_u == "no_try")] = "noattempt"
    cu[~quit_u & (cls_u == "jaw_only")] = "reaction"
    return ce, cu


def session_class_maps(label, align, post_s=None):
    """{(spout, class): (map at half resolution, n trials)} for one session and window."""
    from wfield_local import config, joint_basis
    from wfield_local.beta_maps import _quit_mask
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    from wfield_local.position_reference_maps import (MAP_SHAPE, REST_BASELINE_BINS,
                                                      session_rest_svt_timelocal)
    s = next(x for x in SESSIONS if x["label"] == label)
    post_s = post_s or float(config.defaults()["decode"].get(f"{align}_post_s", 2.0))
    u, v = joint_basis._load_session(s["mc"])
    base = session_rest_svt_timelocal(s, v)
    if base is None:
        return {}
    sig = np.asarray(v) - np.asarray(base)
    a = _args("locanmf", align, post_s, baseline="none")
    X, y, _g, Xn, yn, _reg, idx_e, idx_n = trial_features_cached(
        s, a, signal=sig, feat_region=np.arange(v.shape[0]),
        signal_key=f"svt:rank{v.shape[0]}:restlocal{REST_BASELINE_BINS}", with_indices=True)
    X, y, Xn, yn = np.asarray(X), np.asarray(y), np.asarray(Xn), np.asarray(yn)
    K = u.shape[1]
    cls_cue, _tid = _cue_classes([label], a.max_rt)[label]
    pick = lambda idx: np.array([cls_cue[k] if k < len(cls_cue) else "" for k in idx], object).astype(str)
    cls_e, cls_u = pick(idx_e), pick(idx_n)
    quit_u = (_quit_mask(s, idx_e, idx_n, y, yn) if len(yn) else np.zeros(0, bool))
    pre = config.session_phase(config.animal_of(label), label.split("_")[-1]) == "pre"
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
                m = (sp == P) & (ca == c)
                if m.sum() < (1 if c in POOLED else MIN_TRIALS):
                    continue
                f = Xa[m].mean(0).reshape(n_bins, K).mean(0)
                out[(P, c)] = (_down((u @ f).reshape(MAP_SHAPE)), int(m.sum()))
    return out


def _job(args):
    label, align = args
    try:
        return label, session_class_maps(label, align)
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! {label} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
        return label, None


def collect(animal, align, jobs=4):
    """Aggregate one animal's sessions -> ({key: map}, counts DataFrame)."""
    from wfield_local import analysis_kit as ak, lp_licks
    if not lp_licks.is_lp():
        raise RuntimeError(f"set {lp_licks.LICK_SOURCE_ENV}=lp: the classes are video attempts")
    A = pd.read_csv(lp_licks.attempts_path(), usecols=["label", "epoch"]).drop_duplicates("label")
    ep = dict(zip(A.label, A.epoch))
    labels = [l for l in ak.curated_labels([animal]) if ep.get(l) in ("pre",) + POST_EPOCHS]
    if jobs > 1:
        pairs, fail = ak.fan_sessions([(l, align) for l in labels], _job, jobs=jobs)
        if fail:                                   # a short table looks like a legitimately short one: refuse
            raise RuntimeError(f"{animal} {align}: {len(fail)} session(s) failed in the pool: {fail}")
        res = [v for _item, v in pairs]
    else:
        res = [_job((l, align)) for l in labels]
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


def out_dir():
    from wfield_local import lp_licks
    return lp_licks._run_dir() / "imaging_encoding" / "attempt_maps"


EPOCHS = ("pre",) + POST_EPOCHS
SHOW = (("pre", "pre_hit"), ("pre", "pre_incomplete"), ("acute", "hit"), ("acute", "incomplete"), ("acute", "noattempt"), ("acute", "reaction"),
        ("acute", "stopped"), ("subacute", "hit"), ("subacute", "incomplete"), ("chronic", "hit"))


def _mask_edges():
    """Brain mask + atlas edges at map resolution (beta_maps' shared-grid versions, 2 x 2 block-reduced)."""
    from wfield_local.beta_maps import atlas_edges, brain_mask
    m = brain_mask()
    e = atlas_edges()
    m = None if m is None else _down(np.asarray(m, float)) > 0.5
    e = None if e is None else _down(np.asarray(e, float)) > 0
    return m, e


def hemisphere_table(od, align, animals):
    """Per animal x epoch x spout x class: mean map value in the image-LEFT (ipsilesional) and image-RIGHT
    (contralesional) halves inside the brain mask, and the same for the difference from that spout's pre-stroke hits."""
    m, _e = _mask_edges()
    rows = []
    for an in animals:
        f = od / f"maps_{align}_{an}.npz"
        if not f.exists():
            continue
        Z = np.load(f)
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


def figures(od, align, animals, spouts=("far_contra", "far_mid", "far_ipsi")):
    """Per spout: one figure per animal-set, rows = animals (+ mean of animal means), columns = SHOW (epoch x class);
    one absolute version (shared symmetric scale per row) and one minus that animal's pre-stroke hits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m, edges = _mask_edges()
    fd = od / "figures"
    fd.mkdir(exist_ok=True)
    Zs = {an: np.load(od / f"maps_{align}_{an}.npz") for an in animals if (od / f"maps_{align}_{an}.npz").exists()}
    out = []
    for P in spouts:
        for diff in (False, True):
            cols = [sc for sc in SHOW if not (diff and sc == ("pre", "pre_hit"))]
            fig, axes = plt.subplots(len(Zs) + 1, len(cols), figsize=(1.9 * len(cols), 1.75 * (len(Zs) + 1)),
                                     squeeze=False)
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
                for j, (sc, M) in enumerate(zip(cols, mats)):
                    ax = axes[i, j]
                    ax.set_xticks([]); ax.set_yticks([])
                    if M is not None:
                        show = np.where(m, M, np.nan) if m is not None else M
                        ax.imshow(show, cmap="RdBu_r", vmin=-v, vmax=v)
                        if edges is not None:
                            ax.contour(edges, levels=[0.5], colors="k", linewidths=0.2)
                    else:
                        ax.set_facecolor("0.92")
                    if i == 0:
                        ax.set_title(f"{sc[0]}\n{sc[1]}", fontsize=7)
                    if j == 0:
                        ax.set_ylabel(f"{an}\n(+/-{v:.2g})", fontsize=7)
            what = "minus pre-stroke hits" if diff else "rest-referenced dF"
            fig.suptitle(f"{P}, {align} window: {what} (image left = ipsilesional; mean = mean of animal means, "
                         f">= 2 animals; per-row symmetric 99th-pct scale)", fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fn = fd / f"attempt_maps_{align}_{P}{'_minus_pre' if diff else ''}.png"
            fig.savefig(fn, dpi=130)
            plt.close(fig)
            out.append(fn)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--animals", nargs="+", default=["PS92", "PS93", "PS94", "PS95"])
    ap.add_argument("--aligns", nargs="+", default=list(ALIGNS), choices=ALIGNS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--figures", action="store_true", help="figures + hemisphere table from the stored maps, then exit")
    a = ap.parse_args(argv)
    od = a.out or out_dir()
    od.mkdir(parents=True, exist_ok=True)
    if a.figures:
        for align in a.aligns:
            hemisphere_table(od, align, a.animals).to_csv(od / f"hemispheres_{align}.csv", index=False)
            for fn in figures(od, align, a.animals):
                print(f"[attempt_maps] wrote {fn.name}", flush=True)
        return 0
    for align in a.aligns:
        counts = []
        for an in a.animals:
            f = od / f"maps_{align}_{an}.npz"
            if f.exists():
                print(f"[attempt_maps] {an} {align}: exists, skipped", flush=True)
                continue
            maps, C = collect(an, align, a.jobs)
            np.savez_compressed(f, **maps)
            C.to_csv(od / f"counts_{align}_{an}.csv", index=False)
            counts.append(C)
            print(f"[attempt_maps] {an} {align}: {len(maps)} maps from {C.label.nunique() if len(C) else 0} sessions",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
