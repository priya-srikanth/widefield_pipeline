"""Imaging page section 1 per-animal table on the SPLIT Allen basis (Priya 11-03: "re-run on split basis").

    PYTHONPATH=. python scripts/s1_refit_basis_table.py [--jobs 6] [labels ...]

The section-1 table (within-session refit decoder, pre-cue / post-cue, per animal x epoch) was computed on PLAIN Allen
areas (`scripts/slowband_decode_check.py --all`, 10-20: `filter_acausality_test.roi_signal` + `decode`). The 10-27 late
decision (DECISIONS; `iti_licks` AREA BASIS) made Allen areas WITH the ALM / tjM1 split the primary decoder basis, plain
areas a robustness row. This re-runs the same decoder (`filter_acausality_test.decode`: Allen-area features, block CV,
C = 0.5, chance 1/6; same sessions, trials and windows) on three feature sets built from ONE load of the adopted SVTcorr:
  plain_orig   `roi_signal` exactly as the table (whole native brain mask, >= 20 px per area): reproduces the table
  plain_v8     the split basis's masks (eroded stat mask + the v8 lateral thin rim, `iti_licks.roi_masks`), every
               area but cortex, whole MOs / MOp, ALM / tjM1 out (`iti_licks.area_index(.., "plain")`)
  split        same masks; whole MOs / MOp out, ALM + tjM1 + MOs_rest / MOp_rest in (`area_index(.., "split")`); the rest
               areas are the parent's pixels outside the sub-ROI (= `iti_licks.with_rest`'s (parent - f sub) / (1 - f))
plain_v8 vs split differs ONLY in which areas enter, so it isolates the split; plain_orig vs plain_v8 is the mask change.
ROI version = `iti_licks.ROI_VERSION` (8, the final ROIs). Needs this branch (the v8 ROIs are not on main).
Sessions: curated, from 08-06 (as `slowband_decode_check --all`). Output -> <run>/imaging_encoding/qc/s1_refit_basis.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

EPS = ("pre", "acute", "subacute", "chronic")
SETS = ("plain_orig", "plain_v8", "split")


def basis_signals(label: str, SVTc: np.ndarray) -> dict:
    """{set: (signal (n_feat, T), names)} for the v8-mask feature sets of one session (SVTc: (K, T) adopted)."""
    import iti_licks as il

    from wfield_local import combined_encoding as ce
    _V, _ft, Ub, flat, parea, names, shape, pstat = ce._load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    for src, dst, h in il._rest_list():                       # rest = parent pixels outside the sub-ROI
        par, sub = M.get(f"{src}_{h}"), M.get(f"{dst}_{h}")
        if par is not None and sub is not None:
            M[f"{src}_rest_{h}"] = par & ~sub
    out = {}
    for basis in ("plain", "split"):
        cols = [n for n in M if M[n].any()]
        keep = [cols[i] for i in il.area_index(cols, basis)]
        Ra = np.stack([Ub[M[n]].mean(0) for n in keep], 0).astype(np.float64)       # (n_feat, K)
        out[f"{basis}_v8"] = (Ra @ SVTc, keep)
    out["split"] = out.pop("split_v8")
    return out


def session_job(label: str) -> dict:
    """Module-level worker: refit decode, pre-cue + post-cue, for each feature set."""
    import glob

    from wfield_local import filter_acausality_test as F
    from wfield_local import hemo_variants as hv
    s = next(x for x in F.SESSIONS if x["label"] == label)
    res = s["mc"] + "/wfield_local_results"
    ad = glob.glob(res + "/allen_aligned_affine8v1")[0]
    c = np.load(res + f"/hemo_{hv.ADOPTED}/SVTcorr.npy").astype(np.float64)
    sig, regs = F.roi_signal(ad, c.astype(np.float32))
    feats = {"plain_orig": (sig, regs), **{k: (v, np.arange(len(n))) for k, (v, n) in basis_signals(label, c).items()}}
    out = {"label": label}
    for k, (x, r) in feats.items():
        out[f"{k}_n"] = int(x.shape[0])
        out[f"{k}_pre"] = F.decode(s, x, r, "precue")
        out[f"{k}_post"] = F.decode(s, x, r, "cue")
    print("  {:10s} ".format(label) + "  ".join(f"{k} {v:.3f}" for k, v in out.items()
                                                  if k.endswith(("_pre", "_post"))), flush=True)
    return out


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    from wfield_local import filter_acausality_test as F
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc"
    writeguard.assert_writable(out)
    cur = set(ak.curated_labels())
    labels = a.labels or [s["label"] for s in F.SESSIONS if s["label"][-4:] >= "0806" and s["label"] in cur]
    got, fail = ak.fan_sessions(labels, session_job, jobs=a.jobs)
    for it, err in fail:
        print(f"  FAILED {it}: {err}")
    D = pd.DataFrame([r for _lab, r in got]).sort_values("label")
    D["animal"] = D.label.str.split("_").str[0]
    D["epoch"] = D.label.map(epochs.epoch_of)
    f = Path(out) / ("s1_refit_basis.csv" if not a.labels else "s1_refit_basis_subset.csv")
    D.to_csv(f, index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    for w in ("pre", "post"):
        cols = [f"{k}_{w}" for k in SETS]
        t = D.groupby(["animal", "epoch"])[cols].mean().unstack("epoch")
        print(f"\n== {w}-cue, per animal x epoch (session means), chance 0.167")
        print(t.reindex(columns=list(EPS), level=1).round(3).to_string())
    print("\nsessions per animal x epoch:\n", D.groupby(["animal", "epoch"]).size().unstack().reindex(columns=list(EPS)))
    print("->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
