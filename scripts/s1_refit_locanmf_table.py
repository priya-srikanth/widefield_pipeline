"""Per-session WITHIN-SESSION refit accuracy on the joint-LocaNMF basis (imaging page section 1, Priya 11-03: "we also
have locanmf based decoders"): the per-animal table of Figs 1.1 / 1.2 (family 5r), which only ever drew by-position bars.

    python scripts/s1_refit_locanmf_table.py [--fig-rule]

TRIALS: WORKING trials pre AND post (Priya 11-03: licking trials + no-lick trials outside the terminal quit period,
`grant_kit._class_select("working")`). Figs 1.1 / 1.2 score pre-stroke sessions on licking trials only (the 5c pre rule);
Camera lick gate: WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_ENL_WINDOW_S=2.0 (the bundle's trials, gate and quit then read
video licks; licking trials = camera attempts). `--fig-rule` reproduces that (output s1_refit_locanmf_figrule.csv) so the table can be checked record for record.

Same records as `epoch_accuracy._frozen_vs_refit` (`grant_confusion._collect_5c(align, "working", "paired")`, column 1 =
refit): per animal the shared pooled bundle (`_pooled_bundle`, cached), per session the SAME trial selection as the
figure post-stroke (`_class_select("working", ...)`), and the SAME refit (`_refit_pred`: GroupKFold by position block, deterministic),
so every prediction equals the one in the figure. Recomputed here, not read from `_collect_5c`, only because its pre
entry is an unlabelled list (a session that cannot be scored is skipped, so list position does not identify it).
Accuracy = share of trials predicted right, trials the session could not train on (REFIT_UNAVAILABLE) dropped, as
`_refit_at`; one number per session over all positions.
Output -> <run>/imaging_encoding/qc/s1_refit_locanmf[_<lick-source tag>][_figrule].csv (label, animal, epoch, align, n, acc).
SHOWN: imaging results page section 1, per-animal table (LocaNMF row). Writes a CSV only (no figure): the CSV is the
figdata, so there is no separate plot step for the cohort 1 rebuild inventory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALIGNS = ("precue", "cue")
EPS = ("pre", "acute", "subacute", "chronic")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-rule", action="store_true",
                    help="pre-stroke sessions on LICKING trials (the rule of Figs 1.1 / 1.2), to check against them")
    fig_rule = ap.parse_args(argv).fig_rule
    from wfield_local import epochs
    from wfield_local import grant_confusion as GC
    from wfield_local.locanmf_frozen_decoder import _pipe
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc"
    from wfield_local import lp_licks
    out = lp_licks.output_dir(out)       # camera-lick (LP) runs write under <parent>/<tag>/qc (writeguard rule)
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    # CAPTURE the bundle's own features object: its `.indices[label]` = (engaged cue idx, no-lick cue idx), row for row
    # with XE[GE == i] / XU[GU == i] -- the only way to map a bundle row to its DAQ cue (read-only; `_pooled_bundle`
    # imports `features_with_indices` at call time, so wrapping the module attribute sees the exact object it builds)
    from wfield_local import precue_engagement_states as PES
    import os
    # the shared A4 classes per DAQ cue (window cue_stop for both columns; `scripts/s1_trial_class_snapshot.py`, commit in
    # the .json beside it) -- read, never re-derived
    snap = pd.read_csv(os.environ.get("WFIELD_S1_CLASSES") or (oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc"
                                                               / "s1_trial_classes_cue_stop.csv.gz"))
    feats, _orig = [], PES.features_with_indices
    PES.features_with_indices = lambda *a, **k: feats.append(_orig(*a, **k)) or feats[-1]
    rows = []
    for align in ALIGNS:
        for an in GC.ANIMALS:
            feats.clear()
            GC._BUNDLE_CACHE.pop((an, align), None)            # rebuild so the wrapper sees this bundle's features
            bd = GC._pooled_bundle(an, align)
            assert len(feats) == 1, f"{an} {align}: captured {len(feats)} feature objects"
            idx = feats[0].indices
            XE, YE, GE, XU, YU, GU = (bd[k] for k in ("XE", "YE", "GE", "XU", "YU", "GU"))
            BE, BU, not_eng, e_pre = bd["BE"], bd["BU"], bd["not_eng"], bd["e_pre"]
            for i, lab in enumerate(bd["kept"]):
                se = (e_pre & (GE == i)) if i in bd["pre_i"] else (GE == i)
                su = (GU == i) if len(GU) else np.zeros(0, bool)
                arms = {}
                # QUIT-TAIL working (`_class_select`): robustness row; `--fig-rule` = licking trials at pre (Figs 1.1/1.2)
                arms["quittail"] = ((e_pre & (GE == i), np.zeros(len(GU), bool)) if i in bd["pre_i"] and fig_rule
                                    else GC._class_select("working", se, su, not_eng, rule="daq"))
                # CAMERA gates (Priya 11-03; working = DEFAULT): shared classes matched to this session's DAQ cues
                ie, inl = idx.get(lab, (np.zeros(0, int), np.zeros(0, int)))
                sn = snap[snap.label == lab].set_index("cue_idx")
                if len(ie) == int(se.sum()) and len(inl) == int(su.sum()) and len(sn):
                    for gate, tag in (("working", "camwork"), ("camera_lick", "camlick")):
                        me, mu = se.copy(), su.copy()
                        me[np.flatnonzero(se)] = sn[gate].reindex(np.asarray(ie, int)).fillna(False).to_numpy(bool)
                        if len(mu):
                            mu[np.flatnonzero(su)] = sn[gate].reindex(np.asarray(inl, int)).fillna(False).to_numpy(bool)
                        arms[tag] = (me, mu)
                elif len(sn):
                    print(f"  !! {lab} {align}: index / row count mismatch -- camera arms skipped", flush=True)
                for tag, (me, mu) in arms.items():
                    y = np.concatenate([YE[me]] + ([YU[mu]] if mu.any() else []))
                    if len(y) < 5:
                        continue
                    X = np.vstack([XE[me]] + ([XU[mu]] if mu.any() else []))
                    b = np.concatenate([BE[me]] + ([BU[mu]] if mu.any() else []))
                    p = GC._refit_pred(_pipe, X, y, b)
                    if p is None:
                        continue
                    ok = p != GC.REFIT_UNAVAILABLE
                    rows.append(dict(label=lab, animal=an, epoch=epochs.epoch_of(lab), align=align, trials=tag,
                                     n=int(ok.sum()), acc=float(np.mean(p[ok] == y[ok])) if ok.any() else np.nan))
            print(f"  {align} {an}: {sum(r['animal'] == an and r['align'] == align for r in rows)} rows", flush=True)
    PES.features_with_indices = _orig
    D = pd.DataFrame(rows)
    f = out / ("s1_refit_locanmf" + ("_figrule" if fig_rule else "") + ".csv")   # the run tag is in the directory
    D.to_csv(f, index=False, float_format="%.4g")
    D["aug"] = D.label.str[-4:] >= "0806"
    pd.set_option("display.width", 220)
    for pre_set, m in (("pre = August only", D.aug | (D.epoch != "pre")), ("pre = June + August", D.epoch.notna())):
        t = D[m].groupby(["trials", "align", "animal", "epoch"]).acc.mean().unstack("epoch").reindex(columns=list(EPS))
        print(f"\n== per animal x epoch, session means ({pre_set}); chance 0.167")
        print(t.round(3).to_string())
    print("->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
