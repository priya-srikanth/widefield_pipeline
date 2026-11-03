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
    writeguard.assert_writable(out)
    rows = []
    for align in ALIGNS:
        for an in GC.ANIMALS:
            bd = GC._pooled_bundle(an, align)
            XE, YE, GE, XU, YU, GU = (bd[k] for k in ("XE", "YE", "GE", "XU", "YU", "GU"))
            BE, BU, not_eng, e_pre = bd["BE"], bd["BU"], bd["not_eng"], bd["e_pre"]
            for i, lab in enumerate(bd["kept"]):
                # WORKING trials pre AND post (Priya 11-03). Figs 1.1 / 1.2 use licking trials at pre (the 5c pre rule);
                # `--fig-rule` keeps that, to check this table against the figure's records.
                if i in bd["pre_i"] and fig_rule:
                    me, mu = e_pre & (GE == i), np.zeros(len(GU), bool)
                else:
                    me, mu = GC._class_select("working", (e_pre & (GE == i)) if i in bd["pre_i"] else (GE == i),
                                              (GU == i) if len(GU) else np.zeros(0, bool), not_eng)
                y = np.concatenate([YE[me]] + ([YU[mu]] if mu.any() else []))
                if len(y) < 5:
                    continue
                X = np.vstack([XE[me]] + ([XU[mu]] if mu.any() else []))
                b = np.concatenate([BE[me]] + ([BU[mu]] if mu.any() else []))
                p = GC._refit_pred(_pipe, X, y, b)
                if p is None:
                    continue
                ok = p != GC.REFIT_UNAVAILABLE
                rows.append(dict(label=lab, animal=an, epoch=epochs.epoch_of(lab), align=align, n=int(ok.sum()),
                                 acc=float(np.mean(p[ok] == y[ok])) if ok.any() else np.nan))
            print(f"  {align} {an}: {sum(r['animal'] == an and r['align'] == align for r in rows)} sessions",
                  flush=True)
    D = pd.DataFrame(rows)
    from wfield_local import lp_licks
    tag = lp_licks.output_tag()                       # camera lick source -> its own file (never the DAQ table)
    f = out / (f"s1_refit_locanmf{'_' + tag if tag else ''}" + ("_figrule" if fig_rule else "") + ".csv")
    D.to_csv(f, index=False, float_format="%.4g")
    D["aug"] = D.label.str[-4:] >= "0806"
    pd.set_option("display.width", 220)
    for pre_set, m in (("pre = August only", D.aug | (D.epoch != "pre")), ("pre = June + August", D.epoch.notna())):
        t = D[m].groupby(["align", "animal", "epoch"]).acc.mean().unstack("epoch").reindex(columns=list(EPS))
        print(f"\n== per animal x epoch, session means ({pre_set}); chance 0.167")
        print(t.round(3).to_string())
    print("->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
